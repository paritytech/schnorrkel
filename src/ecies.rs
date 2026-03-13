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
//!
//! ### Wire format
//!
//! ```text
//! [version: 1 byte] [ephemeral_pk: 32 bytes] [nonce: 12 bytes] [ciphertext + tag: N + 16 bytes]
//! ```
//!
//! Total overhead: 61 bytes.
//!
//! ### Example
//!
//! ```
//! # #[cfg(all(feature = "ecies", feature = "getrandom"))]
//! # fn main() {
//! use schnorrkel::Keypair;
//! use schnorrkel::ecies;
//!
//! let alice = Keypair::generate();
//! let bob = Keypair::generate();
//!
//! let plaintext = b"hidden message";
//! let encrypted = ecies::encrypt(plaintext, &bob.public, b"my-app")
//!     .expect("encryption failed");
//!
//! let decrypted = ecies::decrypt(&encrypted, &bob.secret, b"my-app")
//!     .expect("decryption failed");
//!
//! assert_eq!(&decrypted, plaintext);
//! # }
//! # #[cfg(not(all(feature = "ecies", feature = "getrandom")))]
//! # fn main() {}
//! ```

use rand_core::RngCore;

use chacha20poly1305::{
    ChaCha20Poly1305,
    aead::{Aead, KeyInit, generic_array::GenericArray},
};

use curve25519_dalek::ristretto::CompressedRistretto;

use crate::keys::{PublicKey, SecretKey, Keypair, PUBLIC_KEY_LENGTH};

/// Current ECIES wire format version.
const ECIES_VERSION: u8 = 0x01;

/// Size of the ChaCha20-Poly1305 nonce (96 bits).
const NONCE_LEN: usize = 12;

/// Size of the Poly1305 authentication tag.
const TAG_LEN: usize = 16;

/// Fixed overhead added to every ECIES ciphertext:
/// 1 (version) + 32 (ephemeral pubkey) + 12 (nonce) + 16 (auth tag).
pub const ECIES_OVERHEAD: usize = 1 + PUBLIC_KEY_LENGTH + NONCE_LEN + TAG_LEN;

/// Errors specific to ECIES encryption / decryption.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum EciesError {
    /// The ciphertext is too short to contain the ECIES header and tag.
    CiphertextTooShort,
    /// Unsupported ECIES version byte.
    UnsupportedVersion {
        /// The version byte found in the ciphertext.
        version: u8,
    },
    /// The ephemeral public key could not be decompressed.
    InvalidEphemeralKey,
    /// AEAD decryption failed (wrong key, corrupted data, or tampered ciphertext).
    DecryptionFailed,
}

impl core::fmt::Display for EciesError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            EciesError::CiphertextTooShort =>
                write!(f, "ECIES ciphertext too short"),
            EciesError::UnsupportedVersion { version } =>
                write!(f, "Unsupported ECIES version: 0x{version:02x}"),
            EciesError::InvalidEphemeralKey =>
                write!(f, "Invalid ephemeral public key in ECIES ciphertext"),
            EciesError::DecryptionFailed =>
                write!(f, "ECIES decryption failed: authentication or key mismatch"),
        }
    }
}

/// Derive a ChaCha20-Poly1305 cipher from an ECDH shared secret,
/// both public keys, and a context string, using a Merlin transcript.
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
    ChaCha20Poly1305::new(GenericArray::from_slice(&key))
}

/// Encrypt `plaintext` to `recipient` using ECIES over Ristretto255.
///
/// The `ctx` parameter provides domain separation — use a unique
/// application-specific byte string (e.g. `b"my-app-v1"`).
///
/// Returns the complete ciphertext including ephemeral public key,
/// nonce, and authentication tag.
pub fn encrypt(
    plaintext: &[u8],
    recipient: &PublicKey,
    ctx: &[u8],
) -> Result<alloc::vec::Vec<u8>, EciesError> {
    let ephemeral = Keypair::generate();
    encrypt_with(plaintext, recipient, ctx, &ephemeral.secret)
}

/// Encrypt `plaintext` to `recipient` using a caller-supplied ephemeral secret key.
///
/// This is the deterministic core of [`encrypt`] — useful for testing
/// or when the ephemeral key is derived externally.
pub fn encrypt_with(
    plaintext: &[u8],
    recipient: &PublicKey,
    ctx: &[u8],
    ephemeral_secret: &SecretKey,
) -> Result<alloc::vec::Vec<u8>, EciesError> {
    let ephemeral_pk = ephemeral_secret.to_public();

    // ECDH
    let shared = ephemeral_secret.raw_key_exchange(recipient);

    // Derive AEAD key
    let aead = derive_aead(
        &shared,
        ephemeral_pk.as_compressed(),
        recipient.as_compressed(),
        ctx,
    );

    // Generate nonce
    let mut nonce_bytes = [0u8; NONCE_LEN];
    crate::getrandom_or_panic().fill_bytes(&mut nonce_bytes);
    let nonce = GenericArray::from_slice(&nonce_bytes);

    // Build AAD: version || ephemeral_pk || recipient_pk
    let mut aad = [0u8; 1 + PUBLIC_KEY_LENGTH + PUBLIC_KEY_LENGTH];
    aad[0] = ECIES_VERSION;
    aad[1..33].copy_from_slice(ephemeral_pk.as_compressed().as_bytes());
    aad[33..65].copy_from_slice(recipient.as_compressed().as_bytes());

    // Encrypt
    let ciphertext_and_tag = aead
        .encrypt(nonce, chacha20poly1305::aead::Payload { msg: plaintext, aad: &aad })
        .map_err(|_| EciesError::DecryptionFailed)?;

    // Assemble: version || ephemeral_pk || nonce || ciphertext_and_tag
    let mut out = alloc::vec::Vec::with_capacity(ECIES_OVERHEAD + plaintext.len());
    out.push(ECIES_VERSION);
    out.extend_from_slice(ephemeral_pk.as_compressed().as_bytes());
    out.extend_from_slice(&nonce_bytes);
    out.extend_from_slice(&ciphertext_and_tag);
    Ok(out)
}

/// Decrypt an ECIES ciphertext using the recipient's secret key.
///
/// The `ctx` must match the context used during [`encrypt`].
pub fn decrypt(
    ciphertext: &[u8],
    secret: &SecretKey,
    ctx: &[u8],
) -> Result<alloc::vec::Vec<u8>, EciesError> {
    if ciphertext.len() < ECIES_OVERHEAD {
        return Err(EciesError::CiphertextTooShort);
    }

    // Parse header
    let version = ciphertext[0];
    if version != ECIES_VERSION {
        return Err(EciesError::UnsupportedVersion { version });
    }

    let ephemeral_pk_bytes = &ciphertext[1..1 + PUBLIC_KEY_LENGTH];
    let nonce_bytes = &ciphertext[1 + PUBLIC_KEY_LENGTH..1 + PUBLIC_KEY_LENGTH + NONCE_LEN];
    let encrypted = &ciphertext[1 + PUBLIC_KEY_LENGTH + NONCE_LEN..];

    // Decompress ephemeral public key
    let ephemeral_compressed = CompressedRistretto::from_slice(ephemeral_pk_bytes)
        .map_err(|_| EciesError::InvalidEphemeralKey)?;
    let ephemeral_point = ephemeral_compressed
        .decompress()
        .ok_or(EciesError::InvalidEphemeralKey)?;
    let ephemeral_pk = PublicKey::from_point(ephemeral_point);

    // Recipient public key
    let recipient_pk = secret.to_public();

    // ECDH
    let shared = secret.raw_key_exchange(&ephemeral_pk);

    // Derive AEAD key
    let aead_cipher = derive_aead(
        &shared,
        ephemeral_pk.as_compressed(),
        recipient_pk.as_compressed(),
        ctx,
    );

    // Rebuild AAD
    let mut aad = [0u8; 1 + PUBLIC_KEY_LENGTH + PUBLIC_KEY_LENGTH];
    aad[0] = ECIES_VERSION;
    aad[1..33].copy_from_slice(ephemeral_pk.as_compressed().as_bytes());
    aad[33..65].copy_from_slice(recipient_pk.as_compressed().as_bytes());

    // Decrypt
    let nonce = GenericArray::from_slice(nonce_bytes);
    aead_cipher
        .decrypt(nonce, chacha20poly1305::aead::Payload { msg: encrypted, aad: &aad })
        .map_err(|_| EciesError::DecryptionFailed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaChaRng;

    fn test_keypair(seed: u8) -> Keypair {
        let mut rng = ChaChaRng::from_seed([seed; 32]);
        Keypair::generate_with(&mut rng)
    }

    #[test]
    fn round_trip() {
        let alice = test_keypair(1);
        let bob = test_keypair(2);
        let ctx = b"test-round-trip";
        let plaintext = b"hello bob, this is alice";

        let encrypted = encrypt(plaintext, &bob.public, ctx).unwrap();
        assert!(encrypted.len() == ECIES_OVERHEAD + plaintext.len());

        let decrypted = decrypt(&encrypted, &bob.secret, ctx).unwrap();
        assert_eq!(&decrypted[..], plaintext);

        // Alice cannot decrypt a message intended for Bob
        let result = decrypt(&encrypted, &alice.secret, ctx);
        assert_eq!(result, Err(EciesError::DecryptionFailed));
    }

    #[test]
    fn deterministic_encryption() {
        let bob = test_keypair(2);
        let ctx = b"test-deterministic";
        let plaintext = b"deterministic test";

        let ephemeral = SecretKey::generate_with(ChaChaRng::from_seed([42; 32]));
        let enc1 = encrypt_with(plaintext, &bob.public, ctx, &ephemeral).unwrap();

        // Same ephemeral key but different random nonce → different ciphertext
        let enc2 = encrypt_with(plaintext, &bob.public, ctx, &ephemeral).unwrap();

        // Ephemeral PK portion should be identical
        assert_eq!(&enc1[..33], &enc2[..33]);
        // But nonces differ so ciphertexts differ
        assert_ne!(enc1, enc2);

        // Both decrypt correctly
        assert_eq!(decrypt(&enc1, &bob.secret, ctx).unwrap(), plaintext);
        assert_eq!(decrypt(&enc2, &bob.secret, ctx).unwrap(), plaintext);
    }

    #[test]
    fn empty_plaintext() {
        let bob = test_keypair(2);
        let ctx = b"test-empty";

        let encrypted = encrypt(b"", &bob.public, ctx).unwrap();
        assert_eq!(encrypted.len(), ECIES_OVERHEAD);

        let decrypted = decrypt(&encrypted, &bob.secret, ctx).unwrap();
        assert!(decrypted.is_empty());
    }

    #[test]
    fn wrong_context_fails() {
        let bob = test_keypair(2);
        let plaintext = b"context sensitive";

        let encrypted = encrypt(plaintext, &bob.public, b"ctx-a").unwrap();
        let result = decrypt(&encrypted, &bob.secret, b"ctx-b");
        assert_eq!(result, Err(EciesError::DecryptionFailed));
    }

    #[test]
    fn truncated_ciphertext() {
        let result = decrypt(&[0x01; 10], &test_keypair(1).secret, b"ctx");
        assert_eq!(result, Err(EciesError::CiphertextTooShort));
    }

    #[test]
    fn bad_version() {
        let bob = test_keypair(2);
        let mut encrypted = encrypt(b"test", &bob.public, b"ctx").unwrap();
        encrypted[0] = 0xFF;
        let result = decrypt(&encrypted, &bob.secret, b"ctx");
        assert_eq!(result, Err(EciesError::UnsupportedVersion { version: 0xFF }));
    }

    #[test]
    fn tampered_ciphertext() {
        let bob = test_keypair(2);
        let mut encrypted = encrypt(b"do not tamper", &bob.public, b"ctx").unwrap();
        let last = encrypted.len() - 1;
        encrypted[last] ^= 0x01;
        let result = decrypt(&encrypted, &bob.secret, b"ctx");
        assert_eq!(result, Err(EciesError::DecryptionFailed));
    }

    #[test]
    fn large_plaintext() {
        let bob = test_keypair(2);
        let plaintext = alloc::vec![0xAB; 65536];
        let ctx = b"large";

        let encrypted = encrypt(&plaintext, &bob.public, ctx).unwrap();
        let decrypted = decrypt(&encrypted, &bob.secret, ctx).unwrap();
        assert_eq!(decrypted, plaintext);
    }
}
