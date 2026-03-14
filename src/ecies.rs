// -*- mode: rust; -*-
//
// This file is part of schnorrkel.
// Copyright (c) 2019 Web 3 Foundation
// See LICENSE for licensing information.
//
// Authors:
// - Jeff Burdges <jeff@web3.foundation>

//! ## ECIES encryption using sr25519 keys
//!
//! Implements the Elliptic Curve Integrated Encryption Scheme (ECIES)
//! over Ristretto255, providing high-level `encrypt` and `decrypt`
//! operations using sr25519 keypairs.
//!
//! The scheme uses:
//! - **Key agreement:** ECDH on Ristretto255 with ephemeral sender keys
//! - **Key derivation:** Merlin transcript (STROBE128/Keccak-based)
//! - **Symmetric cipher:** ChaCha20-Poly1305 (RFC 8439)
//!
//! ### Security properties
//!
//! - IND-CCA2 secure (ciphertext indistinguishability under adaptive chosen-ciphertext attack)
//! - Forward secrecy against sender key compromise (ephemeral keys)
//! - Authenticated encryption (Poly1305 tag)
//! - Domain-separated key derivation via Merlin transcripts
//! - Sender can re-read outgoing messages via OVK
//!
//! ### Wire format
//!
//! ```text
//! [ephemeral_pk: 32] [nonce: 12] [ciphertext + tag: N + 16]
//! [ovk_nonce: 12] [ovk_wrapped_esk + tag: 32 + 16]
//! ```
//!
//! Total overhead: 120 bytes.
//!
//! The trailing 60 bytes contain the ephemeral secret key encrypted
//! under the sender's [`OutgoingViewingKey`], allowing the sender to
//! later re-derive the shared secret and decrypt their own message.
//!
//! [`OutgoingViewingKey`]: crate::viewing_key::OutgoingViewingKey
//!
//! ### Example
//!
//! ```
//! # #[cfg(all(feature = "ecies", feature = "getrandom"))]
//! # fn main() {
//! use schnorrkel::Keypair;
//! use schnorrkel::ecies;
//! use schnorrkel::viewing_key::FullViewingKey;
//!
//! let alice = Keypair::generate();
//! let bob = Keypair::generate();
//! let alice_fvk = FullViewingKey::from_keypair(&alice);
//! let bob_fvk = FullViewingKey::from_keypair(&bob);
//!
//! let plaintext = b"hidden message";
//! let encrypted = ecies::encrypt(
//!     plaintext, bob_fvk.ivk_public(), b"my-app", alice_fvk.ovk(),
//! ).expect("encryption failed");
//!
//! // Bob decrypts with his IVK
//! let decrypted = bob_fvk.decrypt_incoming(&encrypted, b"my-app")
//!     .expect("decryption failed");
//! assert_eq!(&decrypted, plaintext);
//!
//! // Alice re-reads her outgoing message via OVK
//! let decrypted2 = alice_fvk.decrypt_outgoing(
//!     &encrypted, bob_fvk.ivk_public(), b"my-app",
//! ).expect("outgoing decryption failed");
//! assert_eq!(&decrypted2, plaintext);
//! # }
//! # #[cfg(not(all(feature = "ecies", feature = "getrandom")))]
//! # fn main() {}
//! ```

use rand_core::RngCore;
use zeroize::Zeroize;

use chacha20poly1305::{
    ChaCha20Poly1305,
    aead::{Aead, KeyInit, generic_array::GenericArray},
};

use curve25519_dalek::ristretto::{CompressedRistretto, RistrettoPoint};

use crate::keys::{PublicKey, SecretKey, Keypair, PUBLIC_KEY_LENGTH};
use crate::viewing_key::OutgoingViewingKey;

// ---------------------------------------------------------------------------
// Wire format constants
// ---------------------------------------------------------------------------

/// Size of the ChaCha20-Poly1305 nonce (96 bits).
const NONCE_LEN: usize = 12;

/// Size of the Poly1305 authentication tag.
const TAG_LEN: usize = 16;

/// Size of the ephemeral secret scalar.
const SCALAR_LEN: usize = 32;

/// Size of the OVK blob: 12 (nonce) + 32 (encrypted scalar) + 16 (tag).
const OVK_BLOB_LEN: usize = NONCE_LEN + SCALAR_LEN + TAG_LEN;

/// Fixed overhead added to every ECIES ciphertext:
/// 32 (ephemeral pk) + 12 (nonce) + 16 (tag) + 60 (OVK blob) = 120 bytes.
pub const ECIES_OVERHEAD: usize = PUBLIC_KEY_LENGTH + NONCE_LEN + TAG_LEN + OVK_BLOB_LEN;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors specific to ECIES encryption / decryption.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum EciesError {
    /// The ciphertext is too short to contain the ECIES header and tag.
    CiphertextTooShort,
    /// The ephemeral public key could not be decompressed.
    InvalidEphemeralKey,
    /// The recipient public key is the identity point, which is
    /// insecure: ECDH with the identity always yields the identity,
    /// so anyone can derive the same shared secret.
    IdentityRecipient,
    /// AEAD encryption failed.
    EncryptionFailed,
    /// AEAD decryption failed (wrong key, corrupted data, or tampered ciphertext).
    DecryptionFailed,
}

impl core::fmt::Display for EciesError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            EciesError::CiphertextTooShort =>
                write!(f, "ECIES ciphertext too short"),
            EciesError::InvalidEphemeralKey =>
                write!(f, "Invalid ephemeral public key in ECIES ciphertext"),
            EciesError::IdentityRecipient =>
                write!(f, "ECIES recipient public key is the identity point"),
            EciesError::EncryptionFailed =>
                write!(f, "ECIES encryption failed"),
            EciesError::DecryptionFailed =>
                write!(f, "ECIES decryption failed: authentication or key mismatch"),
        }
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Reject the identity point as a public key.
fn reject_identity(pk: &PublicKey) -> Result<(), EciesError> {
    if pk.as_point() == &RistrettoPoint::default() {
        return Err(EciesError::IdentityRecipient);
    }
    Ok(())
}

/// Build the AAD: `ephemeral_pk || recipient_pk`.
fn build_aad(
    ephemeral_pk: &CompressedRistretto,
    recipient_pk: &CompressedRistretto,
) -> [u8; PUBLIC_KEY_LENGTH + PUBLIC_KEY_LENGTH] {
    let mut aad = [0u8; PUBLIC_KEY_LENGTH + PUBLIC_KEY_LENGTH];
    aad[..32].copy_from_slice(ephemeral_pk.as_bytes());
    aad[32..64].copy_from_slice(recipient_pk.as_bytes());
    aad
}

/// Derive the main ChaCha20-Poly1305 AEAD from an ECDH shared secret.
///
/// Domain: `"sr25519-ecies"`.
fn derive_aead(
    shared_secret: &CompressedRistretto,
    ephemeral_pk: &CompressedRistretto,
    recipient_pk: &CompressedRistretto,
    ctx: &[u8],
) -> ChaCha20Poly1305 {
    let mut t = merlin::Transcript::new(b"sr25519-ecies");
    t.append_message(b"ctx", ctx);
    t.append_message(b"ephemeral-pk", ephemeral_pk.as_bytes());
    t.append_message(b"recipient-pk", recipient_pk.as_bytes());
    t.append_message(b"shared-secret", shared_secret.as_bytes());

    let mut key = [0u8; 32];
    t.challenge_bytes(b"aead-key", &mut key);
    let cipher = ChaCha20Poly1305::new(GenericArray::from_slice(&key));
    key.zeroize();
    cipher
}

/// Derive the OVK-wrap ChaCha20-Poly1305 AEAD.
///
/// Domain: `"sr25519-ecies-ovk-wrap"` — distinct from the main AEAD.
///
/// The `main_ciphertext_and_tag` binds the OVK blob to the specific main
/// ciphertext, preventing an attacker from swapping OVK blobs between
/// messages without detection.
fn derive_ovk_wrap_aead(
    ovk: &[u8; 32],
    ephemeral_pk: &CompressedRistretto,
    recipient_pk: &CompressedRistretto,
    main_ciphertext_and_tag: &[u8],
) -> ChaCha20Poly1305 {
    let mut t = merlin::Transcript::new(b"sr25519-ecies-ovk-wrap");
    t.append_message(b"ovk", ovk);
    t.append_message(b"ephemeral-pk", ephemeral_pk.as_bytes());
    t.append_message(b"recipient-pk", recipient_pk.as_bytes());
    t.append_message(b"main-ciphertext", main_ciphertext_and_tag);

    let mut key = [0u8; 32];
    t.challenge_bytes(b"ovk-wrap-key", &mut key);
    let cipher = ChaCha20Poly1305::new(GenericArray::from_slice(&key));
    key.zeroize();
    cipher
}

/// Parse and validate the ephemeral public key from the ciphertext header.
fn parse_ephemeral_pk(
    ephemeral_pk_bytes: &[u8],
) -> Result<PublicKey, EciesError> {
    let compressed = CompressedRistretto::from_slice(ephemeral_pk_bytes)
        .map_err(|_| EciesError::InvalidEphemeralKey)?;
    let point = compressed
        .decompress()
        .ok_or(EciesError::InvalidEphemeralKey)?;

    if point == RistrettoPoint::default() {
        return Err(EciesError::InvalidEphemeralKey);
    }

    Ok(PublicKey::from_point(point))
}

// ---------------------------------------------------------------------------
// Encrypt
// ---------------------------------------------------------------------------

/// Encrypt `plaintext` to `recipient` using ECIES over Ristretto255.
///
/// - `ctx`: domain separation byte string (e.g. `b"my-app-v1"`).
/// - `sender_ovk`: the sender's outgoing viewing key, embedded in the
///   ciphertext so the sender can later re-read their own message.
pub fn encrypt(
    plaintext: &[u8],
    recipient: &PublicKey,
    ctx: &[u8],
    sender_ovk: &OutgoingViewingKey,
) -> Result<alloc::vec::Vec<u8>, EciesError> {
    let ephemeral = Keypair::generate();
    encrypt_with(plaintext, recipient, ctx, &ephemeral.secret, sender_ovk)
}

/// Encrypt with a caller-supplied ephemeral secret key.
///
/// Deterministic core of [`encrypt`] — useful for testing or when the
/// ephemeral key is derived externally.
pub fn encrypt_with(
    plaintext: &[u8],
    recipient: &PublicKey,
    ctx: &[u8],
    ephemeral_secret: &SecretKey,
    sender_ovk: &OutgoingViewingKey,
) -> Result<alloc::vec::Vec<u8>, EciesError> {
    reject_identity(recipient)?;

    let ephemeral_pk = ephemeral_secret.to_public();

    // ECDH
    let shared = ephemeral_secret.raw_key_exchange(recipient);

    // Derive main AEAD
    let aead = derive_aead(
        &shared,
        ephemeral_pk.as_compressed(),
        recipient.as_compressed(),
        ctx,
    );

    // Random nonce
    let mut nonce_bytes = [0u8; NONCE_LEN];
    crate::getrandom_or_panic().fill_bytes(&mut nonce_bytes);
    let nonce = GenericArray::from_slice(&nonce_bytes);

    // AAD binds ephemeral pk and recipient pk
    let aad = build_aad(ephemeral_pk.as_compressed(), recipient.as_compressed());

    // Encrypt plaintext
    let ciphertext_and_tag = aead
        .encrypt(nonce, chacha20poly1305::aead::Payload { msg: plaintext, aad: &aad })
        .map_err(|_| EciesError::EncryptionFailed)?;

    // OVK-wrap the ephemeral secret scalar, binding to the main ciphertext
    let ovk_aead = derive_ovk_wrap_aead(
        sender_ovk.as_bytes(),
        ephemeral_pk.as_compressed(),
        recipient.as_compressed(),
        &ciphertext_and_tag,
    );

    let mut ovk_nonce_bytes = [0u8; NONCE_LEN];
    crate::getrandom_or_panic().fill_bytes(&mut ovk_nonce_bytes);
    let ovk_nonce = GenericArray::from_slice(&ovk_nonce_bytes);

    let mut esk_bytes = ephemeral_secret.key.to_bytes();
    let ovk_ciphertext = ovk_aead
        .encrypt(ovk_nonce, &esk_bytes[..])
        .map_err(|_| EciesError::EncryptionFailed)?;
    esk_bytes.zeroize();

    // Assemble: ephemeral_pk || nonce || ciphertext+tag || ovk_nonce || ovk_ciphertext+tag
    let mut out = alloc::vec::Vec::with_capacity(ECIES_OVERHEAD + plaintext.len());
    out.extend_from_slice(ephemeral_pk.as_compressed().as_bytes());
    out.extend_from_slice(&nonce_bytes);
    out.extend_from_slice(&ciphertext_and_tag);
    out.extend_from_slice(&ovk_nonce_bytes);
    out.extend_from_slice(&ovk_ciphertext);
    Ok(out)
}

// ---------------------------------------------------------------------------
// Decrypt (recipient)
// ---------------------------------------------------------------------------

/// Decrypt an ECIES ciphertext using the recipient's secret key.
///
/// The OVK blob is stripped and ignored.
pub fn decrypt(
    ciphertext: &[u8],
    secret: &SecretKey,
    ctx: &[u8],
) -> Result<alloc::vec::Vec<u8>, EciesError> {
    let recipient_pk = secret.to_public();
    decrypt_inner(
        ciphertext,
        |ephemeral_pk| secret.raw_key_exchange(ephemeral_pk),
        &recipient_pk,
        ctx,
    )
}

/// Decrypt using an [`IncomingViewingKey`](crate::viewing_key::IncomingViewingKey).
///
/// The ciphertext must have been encrypted to the IVK's public key.
pub fn decrypt_with_ivk(
    ciphertext: &[u8],
    ivk: &crate::viewing_key::IncomingViewingKey,
    ctx: &[u8],
) -> Result<alloc::vec::Vec<u8>, EciesError> {
    let recipient_pk = ivk.ivk_public();
    decrypt_inner(
        ciphertext,
        |ephemeral_pk| ivk.raw_key_exchange(ephemeral_pk),
        recipient_pk,
        ctx,
    )
}

/// Core decryption: parse header, ECDH, derive AEAD, decrypt payload.
/// The OVK blob (last `OVK_BLOB_LEN` bytes) is stripped before decryption.
fn decrypt_inner(
    ciphertext: &[u8],
    ecdh: impl FnOnce(&PublicKey) -> CompressedRistretto,
    recipient_pk: &PublicKey,
    ctx: &[u8],
) -> Result<alloc::vec::Vec<u8>, EciesError> {
    if ciphertext.len() < ECIES_OVERHEAD {
        return Err(EciesError::CiphertextTooShort);
    }

    let ephemeral_pk_bytes = &ciphertext[..PUBLIC_KEY_LENGTH];
    let nonce_bytes = &ciphertext[PUBLIC_KEY_LENGTH..PUBLIC_KEY_LENGTH + NONCE_LEN];
    let main_end = ciphertext.len() - OVK_BLOB_LEN;
    let encrypted = &ciphertext[PUBLIC_KEY_LENGTH + NONCE_LEN..main_end];

    let ephemeral_pk = parse_ephemeral_pk(ephemeral_pk_bytes)?;

    // ECDH
    let shared = ecdh(&ephemeral_pk);

    // Derive AEAD
    let aead_cipher = derive_aead(
        &shared,
        ephemeral_pk.as_compressed(),
        recipient_pk.as_compressed(),
        ctx,
    );

    let aad = build_aad(ephemeral_pk.as_compressed(), recipient_pk.as_compressed());
    let nonce = GenericArray::from_slice(nonce_bytes);
    aead_cipher
        .decrypt(nonce, chacha20poly1305::aead::Payload { msg: encrypted, aad: &aad })
        .map_err(|_| EciesError::DecryptionFailed)
}

// ---------------------------------------------------------------------------
// Decrypt (outgoing, via OVK)
// ---------------------------------------------------------------------------

/// Decrypt an outgoing ECIES ciphertext using an [`OutgoingViewingKey`].
///
/// Recovers the ephemeral secret from the OVK blob, re-derives the
/// ECDH shared secret, and decrypts the main payload.
pub fn decrypt_outgoing(
    ciphertext: &[u8],
    ovk: &OutgoingViewingKey,
    recipient_pk: &PublicKey,
    ctx: &[u8],
) -> Result<alloc::vec::Vec<u8>, EciesError> {
    if ciphertext.len() < ECIES_OVERHEAD {
        return Err(EciesError::CiphertextTooShort);
    }

    let ephemeral_pk_bytes = &ciphertext[..PUBLIC_KEY_LENGTH];
    let nonce_bytes = &ciphertext[PUBLIC_KEY_LENGTH..PUBLIC_KEY_LENGTH + NONCE_LEN];
    let main_end = ciphertext.len() - OVK_BLOB_LEN;
    let encrypted = &ciphertext[PUBLIC_KEY_LENGTH + NONCE_LEN..main_end];

    // Parse OVK blob
    let ovk_blob = &ciphertext[main_end..];
    let ovk_nonce_bytes = &ovk_blob[..NONCE_LEN];
    let ovk_ciphertext = &ovk_blob[NONCE_LEN..];

    let ephemeral_pk = parse_ephemeral_pk(ephemeral_pk_bytes)?;

    // Unwrap the ephemeral secret scalar, binding to the main ciphertext
    let ovk_aead = derive_ovk_wrap_aead(
        ovk.as_bytes(),
        ephemeral_pk.as_compressed(),
        recipient_pk.as_compressed(),
        encrypted,
    );

    let ovk_nonce = GenericArray::from_slice(ovk_nonce_bytes);
    let mut esk_bytes = ovk_aead
        .decrypt(ovk_nonce, ovk_ciphertext)
        .map_err(|_| EciesError::DecryptionFailed)?;

    if esk_bytes.len() != SCALAR_LEN {
        esk_bytes.zeroize();
        return Err(EciesError::DecryptionFailed);
    }

    let mut scalar_arr = [0u8; 32];
    scalar_arr.copy_from_slice(&esk_bytes);
    esk_bytes.zeroize();

    let mut esk_scalar = match crate::scalar_from_canonical_bytes(scalar_arr) {
        Some(s) => s,
        None => {
            scalar_arr.zeroize();
            return Err(EciesError::DecryptionFailed);
        }
    };
    scalar_arr.zeroize();

    // Re-derive ECDH shared secret: esk * recipient_pk
    let shared = (esk_scalar * recipient_pk.as_point()).compress();
    esk_scalar.zeroize();

    // Derive main AEAD
    let aead_cipher = derive_aead(
        &shared,
        ephemeral_pk.as_compressed(),
        recipient_pk.as_compressed(),
        ctx,
    );

    let aad = build_aad(ephemeral_pk.as_compressed(), recipient_pk.as_compressed());
    let nonce = GenericArray::from_slice(nonce_bytes);
    aead_cipher
        .decrypt(nonce, chacha20poly1305::aead::Payload { msg: encrypted, aad: &aad })
        .map_err(|_| EciesError::DecryptionFailed)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaChaRng;

    fn test_keypair(seed: u8) -> Keypair {
        let mut rng = ChaChaRng::from_seed([seed; 32]);
        Keypair::generate_with(&mut rng)
    }

    fn test_ovk(seed: u8) -> OutgoingViewingKey {
        OutgoingViewingKey::from_secret(&test_keypair(seed).secret)
    }

    fn test_ivk(seed: u8) -> crate::viewing_key::IncomingViewingKey {
        crate::viewing_key::IncomingViewingKey::from_secret(&test_keypair(seed).secret)
    }

    #[test]
    fn round_trip() {
        let bob_ivk = test_ivk(2);
        let alice_ovk = test_ovk(1);
        let ctx = b"test-round-trip";
        let plaintext = b"hello bob, this is alice";

        let encrypted = encrypt(plaintext, bob_ivk.ivk_public(), ctx, &alice_ovk).unwrap();
        assert_eq!(encrypted.len(), ECIES_OVERHEAD + plaintext.len());

        // Bob decrypts with IVK
        let decrypted = bob_ivk.decrypt(&encrypted, ctx).unwrap();
        assert_eq!(&decrypted[..], plaintext);

        // Alice decrypts outgoing with OVK
        let decrypted2 = decrypt_outgoing(&encrypted, &alice_ovk, bob_ivk.ivk_public(), ctx).unwrap();
        assert_eq!(&decrypted2[..], plaintext);
    }

    #[test]
    fn wrong_ivk_fails() {
        let bob_ivk = test_ivk(2);
        let alice_ovk = test_ovk(1);
        let charlie_ivk = test_ivk(3);

        let encrypted = encrypt(b"secret", bob_ivk.ivk_public(), b"ctx", &alice_ovk).unwrap();
        assert_eq!(
            charlie_ivk.decrypt(&encrypted, b"ctx"),
            Err(EciesError::DecryptionFailed)
        );
    }

    #[test]
    fn wrong_ovk_fails() {
        let bob_ivk = test_ivk(2);
        let alice_ovk = test_ovk(1);
        let charlie_ovk = test_ovk(3);

        let encrypted = encrypt(b"secret", bob_ivk.ivk_public(), b"ctx", &alice_ovk).unwrap();
        assert_eq!(
            decrypt_outgoing(&encrypted, &charlie_ovk, bob_ivk.ivk_public(), b"ctx"),
            Err(EciesError::DecryptionFailed)
        );
    }

    #[test]
    fn wrong_recipient_pk_in_outgoing_fails() {
        let bob_ivk = test_ivk(2);
        let alice_ovk = test_ovk(1);
        let charlie_ivk = test_ivk(3);

        let encrypted = encrypt(b"test", bob_ivk.ivk_public(), b"ctx", &alice_ovk).unwrap();
        // Wrong recipient pk → OVK wrap key mismatch
        assert_eq!(
            decrypt_outgoing(&encrypted, &alice_ovk, charlie_ivk.ivk_public(), b"ctx"),
            Err(EciesError::DecryptionFailed)
        );
    }

    #[test]
    fn deterministic_encryption() {
        let bob_ivk = test_ivk(2);
        let alice_ovk = test_ovk(1);
        let ctx = b"test-det";
        let plaintext = b"deterministic";

        let eph = SecretKey::generate_with(ChaChaRng::from_seed([42; 32]));
        let enc1 = encrypt_with(plaintext, bob_ivk.ivk_public(), ctx, &eph, &alice_ovk).unwrap();
        let enc2 = encrypt_with(plaintext, bob_ivk.ivk_public(), ctx, &eph, &alice_ovk).unwrap();

        // Same ephemeral pk
        assert_eq!(&enc1[..32], &enc2[..32]);
        // Different nonces
        assert_ne!(enc1, enc2);

        assert_eq!(bob_ivk.decrypt(&enc1, ctx).unwrap(), plaintext);
        assert_eq!(bob_ivk.decrypt(&enc2, ctx).unwrap(), plaintext);
    }

    #[test]
    fn empty_plaintext() {
        let bob_ivk = test_ivk(2);
        let alice_ovk = test_ovk(1);

        let encrypted = encrypt(b"", bob_ivk.ivk_public(), b"empty", &alice_ovk).unwrap();
        assert_eq!(encrypted.len(), ECIES_OVERHEAD);
        assert!(bob_ivk.decrypt(&encrypted, b"empty").unwrap().is_empty());
        assert!(
            decrypt_outgoing(&encrypted, &alice_ovk, bob_ivk.ivk_public(), b"empty")
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn wrong_context_fails() {
        let bob_ivk = test_ivk(2);
        let alice_ovk = test_ovk(1);

        let encrypted = encrypt(b"ctx test", bob_ivk.ivk_public(), b"ctx-a", &alice_ovk).unwrap();
        assert_eq!(
            bob_ivk.decrypt(&encrypted, b"ctx-b"),
            Err(EciesError::DecryptionFailed)
        );
    }

    #[test]
    fn truncated_ciphertext() {
        assert_eq!(
            decrypt(&[0u8; 10], &test_keypair(1).secret, b"ctx"),
            Err(EciesError::CiphertextTooShort)
        );
    }

    #[test]
    fn tampered_ciphertext() {
        let bob_ivk = test_ivk(2);
        let alice_ovk = test_ovk(1);

        let mut encrypted = encrypt(b"tamper", bob_ivk.ivk_public(), b"ctx", &alice_ovk).unwrap();
        // Tamper in the main ciphertext area (before OVK blob)
        let tamper_idx = PUBLIC_KEY_LENGTH + NONCE_LEN + 1;
        encrypted[tamper_idx] ^= 0x01;
        assert_eq!(
            bob_ivk.decrypt(&encrypted, b"ctx"),
            Err(EciesError::DecryptionFailed)
        );
    }

    #[test]
    fn tampered_ovk_blob() {
        let bob_ivk = test_ivk(2);
        let alice_ovk = test_ovk(1);

        let mut encrypted = encrypt(b"tamper ovk", bob_ivk.ivk_public(), b"ctx", &alice_ovk).unwrap();
        let last = encrypted.len() - 1;
        encrypted[last] ^= 0x01;

        // IVK decrypt still works (OVK blob is stripped)
        assert!(bob_ivk.decrypt(&encrypted, b"ctx").is_ok());
        // OVK decrypt fails
        assert_eq!(
            decrypt_outgoing(&encrypted, &alice_ovk, bob_ivk.ivk_public(), b"ctx"),
            Err(EciesError::DecryptionFailed)
        );
    }

    #[test]
    fn large_plaintext() {
        let bob_ivk = test_ivk(2);
        let alice_ovk = test_ovk(1);
        let plaintext = alloc::vec![0xAB; 65536];

        let encrypted = encrypt(&plaintext, bob_ivk.ivk_public(), b"lg", &alice_ovk).unwrap();
        assert_eq!(bob_ivk.decrypt(&encrypted, b"lg").unwrap(), plaintext);
        assert_eq!(
            decrypt_outgoing(&encrypted, &alice_ovk, bob_ivk.ivk_public(), b"lg").unwrap(),
            plaintext
        );
    }

    #[test]
    fn reject_identity_recipient() {
        let identity = PublicKey::from_bytes(&[0u8; 32]).unwrap();
        let ovk = test_ovk(1);
        assert_eq!(
            encrypt(b"test", &identity, b"ctx", &ovk),
            Err(EciesError::IdentityRecipient)
        );
    }

    #[test]
    fn reject_identity_ephemeral() {
        let bob_ivk = test_ivk(2);
        let alice_ovk = test_ovk(1);

        let mut encrypted = encrypt(b"test", bob_ivk.ivk_public(), b"ctx", &alice_ovk).unwrap();
        // Overwrite ephemeral pk with identity
        encrypted[..32].copy_from_slice(&[0u8; 32]);
        assert_eq!(
            bob_ivk.decrypt(&encrypted, b"ctx"),
            Err(EciesError::InvalidEphemeralKey)
        );
    }

    #[test]
    fn decrypt_with_secret_key_directly() {
        // Encrypt to Bob's raw public key (not IVK), decrypt with secret key
        let bob = test_keypair(2);
        let alice_ovk = test_ovk(1);
        let ctx = b"raw-sk";
        let plaintext = b"direct secret key";

        let encrypted = encrypt(plaintext, &bob.public, ctx, &alice_ovk).unwrap();
        let decrypted = decrypt(&encrypted, &bob.secret, ctx).unwrap();
        assert_eq!(&decrypted[..], plaintext);
    }
}
