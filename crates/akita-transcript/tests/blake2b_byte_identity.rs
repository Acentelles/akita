//! Byte-identity guard for the default Blake2b transcript backend.
//!
//! Adding the unaudited `transcript-poseidon2` backend must not perturb a
//! single byte of the shipped transcript. This test replays a fixed script
//! against `AkitaTranscript` and pins a digest of every squeezed byte.
//!
//! The golden digest was produced by running exactly this script against the
//! crate as it stands at git `HEAD` (that is, before the Poseidon2 backend
//! existed), built in a scratch copy so the working tree was not touched. If
//! this test fails, the default Fiat-Shamir transcript changed and every
//! previously produced proof is invalidated.
//!
//! The test is gated on `transcript-blake2b`, which is the default and also
//! wins under `--all-features`, so it runs in both configurations.

#![cfg(feature = "transcript-blake2b")]
#![allow(missing_docs)]

use akita_field::Prime128Offset275;
// `preview_challenge_bytes_after_absorb{,_chain}` are inherent methods on
// `AkitaTranscript`, so the `FoldChallengeSeedPreview` trait does not need to
// be in scope to call them.
use akita_transcript::AkitaTranscript;
use blake2::{Blake2b512, Digest};

type F = Prime128Offset275;

/// Replay a fixed absorb/squeeze script and digest everything observable.
fn transcript_fingerprint() -> String {
    let mut acc = Blake2b512::new();

    // Protocol tag is part of the wire behaviour.
    acc.update(akita_transcript::PROTOCOL_TAG);

    let mut transcript = AkitaTranscript::<F>::prover(b"identity/session", b"identity/instance");

    acc.update(transcript.squeeze_bytes(akita_transcript::label!("c0"), 64));

    transcript.absorb_bytes(akita_transcript::label!("a0"), b"");
    transcript.absorb_bytes(akita_transcript::label!("a1"), b"akita");
    transcript.absorb_bytes(akita_transcript::label!("a2"), &(0..=255u8).collect::<Vec<_>>());
    acc.update(transcript.squeeze_bytes(akita_transcript::label!("c1"), 1));
    acc.update(transcript.squeeze_bytes(akita_transcript::label!("c2"), 31));
    acc.update(transcript.squeeze_bytes(akita_transcript::label!("c3"), 97));

    let scalar = transcript.squeeze_scalar(akita_transcript::label!("c4"));
    acc.update(akita_field::CanonicalBytes::to_bytes_le_vec(&scalar));

    for value in [0u64, 1, 42, u64::MAX] {
        transcript.absorb_field(
            akita_transcript::label!("af"),
            &<F as akita_field::FromPrimitiveInt>::from_u64(value),
        );
    }
    acc.update(transcript.squeeze_bytes(akita_transcript::label!("c5"), 48));

    // Preview paths feed fold grinding; pin them too.
    acc.update(transcript.preview_challenge_bytes_after_absorb(b"grind-probe", 32));
    acc.update(transcript.preview_challenge_bytes_after_absorb_chain(
        &[b"chain-a".as_slice(), b"chain-b".as_slice()],
        &[16, 24],
    ));
    acc.update(transcript.squeeze_bytes(akita_transcript::label!("c6"), 8));

    // Re-binding must be reflected.
    transcript.bind_instance_bytes(b"rebound/instance");
    acc.update(transcript.squeeze_bytes(akita_transcript::label!("c7"), 32));

    // Verifier side must track the prover side exactly.
    let mut verifier =
        AkitaTranscript::<F>::verifier(b"identity/session", b"identity/instance");
    verifier.absorb_bytes(akita_transcript::label!("a1"), b"akita");
    acc.update(verifier.squeeze_bytes(akita_transcript::label!("cv"), 32));

    hex(&acc.finalize())
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// Golden digest captured from git HEAD (pre-Poseidon2).
const GOLDEN: &str = include_str!("blake2b_byte_identity.golden");

#[test]
fn default_blake2b_transcript_is_byte_identical_to_head() {
    assert_eq!(
        transcript_fingerprint(),
        GOLDEN.trim(),
        "the default Blake2b transcript changed; every existing proof is invalidated"
    );
}
