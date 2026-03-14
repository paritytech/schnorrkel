// -*- mode: rust; -*-
//
// This file is part of schnorrkel.
// Copyright (c) 2019 Web 3 Foundation
// See LICENSE for licensing information.
//
// Authors:
// - Jeff Burdges <jeff@web3.foundation>

//! ## Full viewing keys for sr25519
//!
//! A Penumbra-style viewing key hierarchy that separates
//! incoming decryption, outgoing decryption, and signing authority.
//!
//! ```text
//! SecretKey
//!   ├── FullViewingKey { ivk, ovk, signing_public }
//!   │   ├── IncomingViewingKey   ← decrypt messages TO you
//!   │   └── OutgoingViewingKey   ← decrypt messages BY you
//!   └── sign()                   ← spending, full secret only
//! ```
//!
//! All derivations are one-way via domain-separated Merlin transcripts:
//!
//! ```text
//! ivk_scalar = Merlin("sr25519-ivk", signing_scalar)   → ECDH scalar
//! ovk        = Merlin("sr25519-ovk", signing_scalar)   → 32-byte symmetric key
//! ```
//!
//! ### Example
//!
//! ```
//! # #[cfg(all(feature = "ecies", feature = "getrandom"))]
//! # fn main() {
//! use schnorrkel::{Keypair, ecies};
//! use schnorrkel::viewing_key::FullViewingKey;
//!
//! let alice = Keypair::generate();
//! let bob = Keypair::generate();
//! let bob_fvk = FullViewingKey::from_keypair(&bob);
//!
//! // Alice encrypts to Bob's incoming viewing public key,
//! // attaching her OVK so she can later re-read her own message.
//! let alice_fvk = FullViewingKey::from_keypair(&alice);
//! let plaintext = b"confidential data";
//! let encrypted = ecies::encrypt(
//!     plaintext, bob_fvk.ivk_public(), b"my-app",
//!     alice_fvk.ovk(),
//! ).expect("encryption failed");
//!
//! // Bob decrypts with his IVK
//! let decrypted = bob_fvk.decrypt_incoming(&encrypted, b"my-app")
//!     .expect("decryption failed");
//! assert_eq!(&decrypted, plaintext);
//!
//! // Alice re-reads her own outgoing message via OVK
//! let decrypted2 = alice_fvk.decrypt_outgoing(
//!     &encrypted, bob_fvk.ivk_public(), b"my-app",
//! ).expect("outgoing decryption failed");
//! assert_eq!(&decrypted2, plaintext);
//! # }
//! # #[cfg(not(all(feature = "ecies", feature = "getrandom")))]
//! # fn main() {}
//! ```

use core::fmt::Debug;

use curve25519_dalek::constants;
use curve25519_dalek::ristretto::CompressedRistretto;
use curve25519_dalek::scalar::Scalar;

use subtle::{ConstantTimeEq, Choice};
use zeroize::Zeroize;

use crate::keys::{PublicKey, SecretKey, PUBLIC_KEY_LENGTH};
use crate::errors::{SignatureError, SignatureResult};

// ---------------------------------------------------------------------------
// IncomingViewingKey
// ---------------------------------------------------------------------------

/// Length of a serialized [`IncomingViewingKey`]: 32 (scalar) + 32 (public).
pub const INCOMING_VIEWING_KEY_LENGTH: usize = 32 + PUBLIC_KEY_LENGTH;

/// An incoming viewing key derived one-way from an sr25519 secret key.
///
/// Allows decryption of ECIES messages encrypted to [`ivk_public`](Self::ivk_public)
/// but cannot produce signatures.
#[derive(Clone)]
pub struct IncomingViewingKey {
    /// The incoming viewing scalar.
    pub(crate) ivk_scalar: Scalar,
    /// The incoming viewing public key: `ivk_scalar * G`.
    pub(crate) ivk_public: PublicKey,
}

impl Debug for IncomingViewingKey {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "IncomingViewingKey {{ ivk_public: {:?} }}", self.ivk_public)
    }
}

impl Eq for IncomingViewingKey {}
impl PartialEq for IncomingViewingKey {
    fn eq(&self, other: &Self) -> bool {
        self.ct_eq(other).unwrap_u8() == 1u8
    }
}
impl ConstantTimeEq for IncomingViewingKey {
    fn ct_eq(&self, other: &Self) -> Choice {
        self.ivk_scalar.ct_eq(&other.ivk_scalar)
    }
}

impl Zeroize for IncomingViewingKey {
    fn zeroize(&mut self) {
        self.ivk_scalar.zeroize();
    }
}
impl Drop for IncomingViewingKey {
    fn drop(&mut self) {
        self.zeroize();
    }
}

impl IncomingViewingKey {
    /// Derive an `IncomingViewingKey` from a secret key.
    pub fn from_secret(secret: &SecretKey) -> Self {
        let ivk_scalar = Self::derive_scalar(secret);
        let ivk_public =
            PublicKey::from_point(&ivk_scalar * constants::RISTRETTO_BASEPOINT_TABLE);
        IncomingViewingKey { ivk_scalar, ivk_public }
    }

    /// One-way scalar derivation via Merlin transcript.
    fn derive_scalar(secret: &SecretKey) -> Scalar {
        let mut t = merlin::Transcript::new(b"sr25519-ivk");
        t.append_message(b"signing-scalar", &secret.key.to_bytes());
        let mut buf = [0u8; 64];
        t.challenge_bytes(b"ivk-scalar", &mut buf);
        let scalar = Scalar::from_bytes_mod_order_wide(&buf);
        buf.zeroize();
        scalar
    }

    /// The public key that senders should encrypt to.
    pub fn ivk_public(&self) -> &PublicKey {
        &self.ivk_public
    }

    /// Perform ECDH using the incoming viewing scalar.
    pub(crate) fn raw_key_exchange(&self, public: &PublicKey) -> CompressedRistretto {
        (self.ivk_scalar * public.as_point()).compress()
    }

    /// Decrypt an ECIES ciphertext encrypted to this key's
    /// [`ivk_public`](Self::ivk_public).
    #[cfg(feature = "ecies")]
    pub fn decrypt(
        &self,
        ciphertext: &[u8],
        ctx: &[u8],
    ) -> Result<alloc::vec::Vec<u8>, crate::ecies::EciesError> {
        crate::ecies::decrypt_with_ivk(ciphertext, self, ctx)
    }

    /// Serialize to bytes: `[ivk_scalar: 32] [ivk_public: 32]`.
    pub fn to_bytes(&self) -> [u8; INCOMING_VIEWING_KEY_LENGTH] {
        let mut bytes = [0u8; INCOMING_VIEWING_KEY_LENGTH];
        bytes[..32].copy_from_slice(&self.ivk_scalar.to_bytes());
        bytes[32..64].copy_from_slice(&self.ivk_public.to_bytes());
        bytes
    }

    /// Deserialize, verifying `ivk_scalar * G == ivk_public`.
    pub fn from_bytes(bytes: &[u8]) -> SignatureResult<Self> {
        if bytes.len() != INCOMING_VIEWING_KEY_LENGTH {
            return Err(SignatureError::BytesLengthError {
                name: "IncomingViewingKey",
                description: "A 64-byte sr25519 incoming viewing key",
                length: INCOMING_VIEWING_KEY_LENGTH,
            });
        }

        let mut scalar_bytes = [0u8; 32];
        scalar_bytes.copy_from_slice(&bytes[..32]);
        let ivk_scalar = crate::scalar_from_canonical_bytes(scalar_bytes)
            .ok_or(SignatureError::ScalarFormatError)?;

        let ivk_public = PublicKey::from_bytes(&bytes[32..64])?;

        let expected =
            PublicKey::from_point(&ivk_scalar * constants::RISTRETTO_BASEPOINT_TABLE);
        if expected != ivk_public {
            return Err(SignatureError::PointDecompressionError);
        }

        Ok(IncomingViewingKey { ivk_scalar, ivk_public })
    }
}

// ---------------------------------------------------------------------------
// OutgoingViewingKey
// ---------------------------------------------------------------------------

/// Length of a serialized [`OutgoingViewingKey`]: 32 bytes (symmetric key).
pub const OUTGOING_VIEWING_KEY_LENGTH: usize = 32;

/// An outgoing viewing key derived one-way from an sr25519 secret key.
///
/// This 32-byte symmetric key is used to wrap the ephemeral secret
/// inside ECIES ciphertexts, allowing the sender to later re-derive
/// the shared secret and decrypt their own outgoing messages.
#[derive(Clone)]
pub struct OutgoingViewingKey {
    /// The 32-byte symmetric OVK.
    pub(crate) ovk: [u8; 32],
}

impl Debug for OutgoingViewingKey {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "OutgoingViewingKey(***)")
    }
}

impl Eq for OutgoingViewingKey {}
impl PartialEq for OutgoingViewingKey {
    fn eq(&self, other: &Self) -> bool {
        self.ct_eq(other).unwrap_u8() == 1u8
    }
}
impl ConstantTimeEq for OutgoingViewingKey {
    fn ct_eq(&self, other: &Self) -> Choice {
        self.ovk.ct_eq(&other.ovk)
    }
}

impl Zeroize for OutgoingViewingKey {
    fn zeroize(&mut self) {
        self.ovk.zeroize();
    }
}
impl Drop for OutgoingViewingKey {
    fn drop(&mut self) {
        self.zeroize();
    }
}

impl OutgoingViewingKey {
    /// Derive an `OutgoingViewingKey` from a secret key.
    pub fn from_secret(secret: &SecretKey) -> Self {
        let mut t = merlin::Transcript::new(b"sr25519-ovk");
        t.append_message(b"signing-scalar", &secret.key.to_bytes());
        let mut ovk = [0u8; 32];
        t.challenge_bytes(b"ovk-key", &mut ovk);
        OutgoingViewingKey { ovk }
    }

    /// Access the raw 32-byte OVK material.
    pub(crate) fn as_bytes(&self) -> &[u8; 32] {
        &self.ovk
    }

    /// Serialize to 32 bytes.
    pub fn to_bytes(&self) -> [u8; OUTGOING_VIEWING_KEY_LENGTH] {
        self.ovk
    }

    /// Deserialize from 32 bytes.
    pub fn from_bytes(bytes: &[u8]) -> SignatureResult<Self> {
        if bytes.len() != OUTGOING_VIEWING_KEY_LENGTH {
            return Err(SignatureError::BytesLengthError {
                name: "OutgoingViewingKey",
                description: "A 32-byte sr25519 outgoing viewing key",
                length: OUTGOING_VIEWING_KEY_LENGTH,
            });
        }
        let mut ovk = [0u8; 32];
        ovk.copy_from_slice(bytes);
        Ok(OutgoingViewingKey { ovk })
    }
}

// ---------------------------------------------------------------------------
// FullViewingKey
// ---------------------------------------------------------------------------

/// Length of a serialized [`FullViewingKey`]:
/// 32 (ivk scalar) + 32 (ivk public) + 32 (ovk) + 32 (signing public) = 128.
pub const FULL_VIEWING_KEY_LENGTH: usize =
    INCOMING_VIEWING_KEY_LENGTH + OUTGOING_VIEWING_KEY_LENGTH + PUBLIC_KEY_LENGTH;

/// A full viewing key that bundles incoming and outgoing viewing
/// capabilities with the signing public key for identification.
///
/// Derived one-way from a [`SecretKey`]; cannot produce signatures.
#[derive(Clone)]
pub struct FullViewingKey {
    /// Incoming viewing key.
    ivk: IncomingViewingKey,
    /// Outgoing viewing key.
    ovk: OutgoingViewingKey,
    /// The original signing public key (advisory — not cryptographically
    /// bound to the IVK/OVK, only trustworthy if you derived this FVK
    /// yourself).
    signing_public: PublicKey,
}

impl Debug for FullViewingKey {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "FullViewingKey {{ ivk_public: {:?}, signing_public: {:?} }}",
            self.ivk.ivk_public, self.signing_public
        )
    }
}

impl Eq for FullViewingKey {}
impl PartialEq for FullViewingKey {
    fn eq(&self, other: &Self) -> bool {
        self.ct_eq(other).unwrap_u8() == 1u8
    }
}
impl ConstantTimeEq for FullViewingKey {
    fn ct_eq(&self, other: &Self) -> Choice {
        self.ivk.ct_eq(&other.ivk) & self.ovk.ct_eq(&other.ovk)
    }
}

impl Zeroize for FullViewingKey {
    fn zeroize(&mut self) {
        self.ivk.zeroize();
        self.ovk.zeroize();
    }
}
impl Drop for FullViewingKey {
    fn drop(&mut self) {
        self.zeroize();
    }
}

impl FullViewingKey {
    /// Derive a `FullViewingKey` from a [`SecretKey`].
    pub fn from_secret(secret: &SecretKey) -> Self {
        let ivk = IncomingViewingKey::from_secret(secret);
        let ovk = OutgoingViewingKey::from_secret(secret);
        let signing_public = secret.to_public();
        FullViewingKey { ivk, ovk, signing_public }
    }

    /// Derive a `FullViewingKey` from a [`Keypair`](crate::Keypair)
    /// (avoids recomputing the signing public key).
    pub fn from_keypair(keypair: &crate::Keypair) -> Self {
        let ivk = IncomingViewingKey::from_secret(&keypair.secret);
        let ovk = OutgoingViewingKey::from_secret(&keypair.secret);
        FullViewingKey {
            ivk,
            ovk,
            signing_public: keypair.public,
        }
    }

    /// The incoming viewing key.
    pub fn ivk(&self) -> &IncomingViewingKey {
        &self.ivk
    }

    /// The outgoing viewing key.
    pub fn ovk(&self) -> &OutgoingViewingKey {
        &self.ovk
    }

    /// Shortcut: the IVK's public key (what senders encrypt to).
    pub fn ivk_public(&self) -> &PublicKey {
        self.ivk.ivk_public()
    }

    /// The original signing public key (advisory).
    pub fn signing_public(&self) -> &PublicKey {
        &self.signing_public
    }

    /// Decrypt an incoming ECIES message.
    #[cfg(feature = "ecies")]
    pub fn decrypt_incoming(
        &self,
        ciphertext: &[u8],
        ctx: &[u8],
    ) -> Result<alloc::vec::Vec<u8>, crate::ecies::EciesError> {
        self.ivk.decrypt(ciphertext, ctx)
    }

    /// Decrypt an outgoing ECIES message using the OVK.
    ///
    /// `recipient_pk` is the IVK public key of the recipient
    /// (needed to re-derive the OVK-wrap AEAD key).
    #[cfg(feature = "ecies")]
    pub fn decrypt_outgoing(
        &self,
        ciphertext: &[u8],
        recipient_pk: &PublicKey,
        ctx: &[u8],
    ) -> Result<alloc::vec::Vec<u8>, crate::ecies::EciesError> {
        crate::ecies::decrypt_outgoing(ciphertext, &self.ovk, recipient_pk, ctx)
    }

    /// Serialize: `[ivk_scalar: 32] [ivk_public: 32] [ovk: 32] [signing_public: 32]`.
    pub fn to_bytes(&self) -> [u8; FULL_VIEWING_KEY_LENGTH] {
        let mut bytes = [0u8; FULL_VIEWING_KEY_LENGTH];
        let ivk_bytes = self.ivk.to_bytes();
        bytes[..64].copy_from_slice(&ivk_bytes);
        bytes[64..96].copy_from_slice(&self.ovk.to_bytes());
        bytes[96..128].copy_from_slice(&self.signing_public.to_bytes());
        bytes
    }

    /// Deserialize, verifying IVK consistency.
    pub fn from_bytes(bytes: &[u8]) -> SignatureResult<Self> {
        if bytes.len() != FULL_VIEWING_KEY_LENGTH {
            return Err(SignatureError::BytesLengthError {
                name: "FullViewingKey",
                description: "A 128-byte sr25519 full viewing key",
                length: FULL_VIEWING_KEY_LENGTH,
            });
        }

        let ivk = IncomingViewingKey::from_bytes(&bytes[..64])?;
        let ovk = OutgoingViewingKey::from_bytes(&bytes[64..96])?;
        let signing_public = PublicKey::from_bytes(&bytes[96..128])?;

        Ok(FullViewingKey { ivk, ovk, signing_public })
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Keypair;
    use rand::SeedableRng;
    use rand_chacha::ChaChaRng;

    fn test_keypair(seed: u8) -> Keypair {
        let mut rng = ChaChaRng::from_seed([seed; 32]);
        Keypair::generate_with(&mut rng)
    }

    // --- IncomingViewingKey tests ---

    #[test]
    fn ivk_derivation_is_deterministic() {
        let kp = test_keypair(1);
        let ivk1 = IncomingViewingKey::from_secret(&kp.secret);
        let ivk2 = IncomingViewingKey::from_secret(&kp.secret);
        assert_eq!(ivk1, ivk2);
    }

    #[test]
    fn ivk_public_differs_from_signing_public() {
        let kp = test_keypair(1);
        let ivk = IncomingViewingKey::from_secret(&kp.secret);
        assert_ne!(ivk.ivk_public(), &kp.public);
    }

    #[test]
    fn ivk_different_keys_differ() {
        let ivk1 = IncomingViewingKey::from_secret(&test_keypair(1).secret);
        let ivk2 = IncomingViewingKey::from_secret(&test_keypair(2).secret);
        assert_ne!(ivk1, ivk2);
    }

    #[test]
    fn ivk_serialization_round_trip() {
        let kp = test_keypair(1);
        let ivk = IncomingViewingKey::from_secret(&kp.secret);
        let bytes = ivk.to_bytes();
        let ivk2 = IncomingViewingKey::from_bytes(&bytes).unwrap();
        assert_eq!(ivk, ivk2);
    }

    #[test]
    fn ivk_tampered_scalar_rejected() {
        let kp = test_keypair(1);
        let ivk = IncomingViewingKey::from_secret(&kp.secret);
        let mut bytes = ivk.to_bytes();
        bytes[0] ^= 0x01;
        assert!(IncomingViewingKey::from_bytes(&bytes).is_err());
    }

    #[test]
    fn ivk_debug_does_not_leak_scalar() {
        let kp = test_keypair(1);
        let ivk = IncomingViewingKey::from_secret(&kp.secret);
        let debug = alloc::format!("{:?}", ivk);
        assert!(!debug.contains("ivk_scalar"));
    }

    // --- OutgoingViewingKey tests ---

    #[test]
    fn ovk_derivation_is_deterministic() {
        let kp = test_keypair(1);
        let ovk1 = OutgoingViewingKey::from_secret(&kp.secret);
        let ovk2 = OutgoingViewingKey::from_secret(&kp.secret);
        assert_eq!(ovk1, ovk2);
    }

    #[test]
    fn ovk_different_keys_differ() {
        let ovk1 = OutgoingViewingKey::from_secret(&test_keypair(1).secret);
        let ovk2 = OutgoingViewingKey::from_secret(&test_keypair(2).secret);
        assert_ne!(ovk1, ovk2);
    }

    #[test]
    fn ovk_serialization_round_trip() {
        let kp = test_keypair(1);
        let ovk = OutgoingViewingKey::from_secret(&kp.secret);
        let bytes = ovk.to_bytes();
        let ovk2 = OutgoingViewingKey::from_bytes(&bytes).unwrap();
        assert_eq!(ovk, ovk2);
    }

    #[test]
    fn ovk_debug_redacted() {
        let kp = test_keypair(1);
        let ovk = OutgoingViewingKey::from_secret(&kp.secret);
        let debug = alloc::format!("{:?}", ovk);
        assert_eq!(debug, "OutgoingViewingKey(***)");
    }

    // --- FullViewingKey tests ---

    #[test]
    fn fvk_from_secret_and_keypair_match() {
        let kp = test_keypair(1);
        let fvk1 = FullViewingKey::from_secret(&kp.secret);
        let fvk2 = FullViewingKey::from_keypair(&kp);
        assert_eq!(fvk1.ivk, fvk2.ivk);
        assert_eq!(fvk1.ovk, fvk2.ovk);
        assert_eq!(fvk1.signing_public, fvk2.signing_public);
    }

    #[test]
    fn fvk_serialization_round_trip() {
        let kp = test_keypair(1);
        let fvk = FullViewingKey::from_keypair(&kp);
        let bytes = fvk.to_bytes();
        assert_eq!(bytes.len(), FULL_VIEWING_KEY_LENGTH);
        let fvk2 = FullViewingKey::from_bytes(&bytes).unwrap();
        assert_eq!(fvk, fvk2);
    }

    #[test]
    fn fvk_debug_does_not_leak_secrets() {
        let kp = test_keypair(1);
        let fvk = FullViewingKey::from_keypair(&kp);
        let debug = alloc::format!("{:?}", fvk);
        let ivk_hex = alloc::format!("{:?}", fvk.ivk.ivk_scalar.to_bytes());
        assert!(!debug.contains(&ivk_hex));
    }

    #[test]
    fn ivk_and_ovk_are_independent() {
        // IVK is a scalar, OVK is a symmetric key — derived from the
        // same secret via different domain tags, they must differ.
        let kp = test_keypair(1);
        let ivk = IncomingViewingKey::from_secret(&kp.secret);
        let ovk = OutgoingViewingKey::from_secret(&kp.secret);
        assert_ne!(&ivk.ivk_scalar.to_bytes()[..], &ovk.ovk[..]);
    }

    // --- ECIES integration tests ---

    #[cfg(feature = "ecies")]
    #[test]
    fn ecies_ivk_round_trip() {
        let kp = test_keypair(1);
        let alice_ovk = OutgoingViewingKey::from_secret(&test_keypair(2).secret);
        let ivk = IncomingViewingKey::from_secret(&kp.secret);
        let ctx = b"test-ivk";
        let plaintext = b"incoming only";

        let encrypted =
            crate::ecies::encrypt(plaintext, ivk.ivk_public(), ctx, &alice_ovk).unwrap();

        let decrypted = ivk.decrypt(&encrypted, ctx).unwrap();
        assert_eq!(&decrypted[..], plaintext);

        // Signing secret key cannot decrypt (different key)
        let result = crate::ecies::decrypt(&encrypted, &kp.secret, ctx);
        assert!(result.is_err());
    }

    #[cfg(feature = "ecies")]
    #[test]
    fn ecies_fvk_round_trip() {
        let alice = test_keypair(1);
        let bob = test_keypair(2);
        let alice_fvk = FullViewingKey::from_keypair(&alice);
        let bob_fvk = FullViewingKey::from_keypair(&bob);
        let ctx = b"test-fvk";
        let plaintext = b"full viewing key round trip";

        let encrypted = crate::ecies::encrypt(
            plaintext,
            bob_fvk.ivk_public(),
            ctx,
            alice_fvk.ovk(),
        )
        .unwrap();

        // Bob decrypts incoming
        let decrypted = bob_fvk.decrypt_incoming(&encrypted, ctx).unwrap();
        assert_eq!(&decrypted[..], plaintext);

        // Alice decrypts outgoing
        let decrypted2 = alice_fvk
            .decrypt_outgoing(&encrypted, bob_fvk.ivk_public(), ctx)
            .unwrap();
        assert_eq!(&decrypted2[..], plaintext);
    }

    #[cfg(feature = "ecies")]
    #[test]
    fn wrong_ovk_fails() {
        let alice = test_keypair(1);
        let bob = test_keypair(2);
        let charlie = test_keypair(3);
        let alice_fvk = FullViewingKey::from_keypair(&alice);
        let bob_fvk = FullViewingKey::from_keypair(&bob);
        let charlie_fvk = FullViewingKey::from_keypair(&charlie);

        let encrypted = crate::ecies::encrypt(
            b"secret",
            bob_fvk.ivk_public(),
            b"ctx",
            alice_fvk.ovk(),
        )
        .unwrap();

        let result = charlie_fvk.decrypt_outgoing(
            &encrypted, bob_fvk.ivk_public(), b"ctx",
        );
        assert!(result.is_err());
    }

    #[cfg(feature = "ecies")]
    #[test]
    fn tampered_ovk_blob_fails() {
        let alice = test_keypair(1);
        let bob = test_keypair(2);
        let alice_fvk = FullViewingKey::from_keypair(&alice);
        let bob_fvk = FullViewingKey::from_keypair(&bob);

        let mut encrypted = crate::ecies::encrypt(
            b"tamper test",
            bob_fvk.ivk_public(),
            b"ctx",
            alice_fvk.ovk(),
        )
        .unwrap();

        let last = encrypted.len() - 1;
        encrypted[last] ^= 0x01;

        // IVK still works
        assert!(bob_fvk.decrypt_incoming(&encrypted, b"ctx").is_ok());
        // OVK fails
        assert!(alice_fvk.decrypt_outgoing(
            &encrypted, bob_fvk.ivk_public(), b"ctx",
        ).is_err());
    }
}
