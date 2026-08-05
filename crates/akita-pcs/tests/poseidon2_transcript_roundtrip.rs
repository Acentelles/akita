//! Prover/verifier round trip under the algebraic Poseidon2 transcript.
//!
//! # UNAUDITED RESEARCH CRYPTOGRAPHY - NOT FOR PRODUCTION
//!
//! This test shows that swapping the Fiat-Shamir sponge for the Poseidon2
//! duplex over `p* = 2^64 - 59` is transparent to the protocol: same prove and
//! verify path, same claims, same acceptance, with a completely different
//! transcript hash underneath. It also checks the two negative directions that
//! make the positive result meaningful.
//!
//! # How this is reachable at all
//!
//! `AkitaTranscript` has always been generic over its sponge, so the backend is
//! chosen here by *naming the type*, not by feature selection:
//! `AkitaTranscript<F, Poseidon2ByteSponge<Alpha3>>`. The default
//! `TranscriptSponge` alias is untouched and still resolves to Blake2b in this
//! build, so nothing else in the workspace changes.
//!
//! Two small enabling changes made this possible, both authorized:
//!
//! - `akita_prover::ProverTranscriptGrind` was widened from
//!   `AkitaTranscript<F, TranscriptSponge>` to a generic sponge parameter
//!   bounded by `akita_transcript::TranscriptSpongeBackend`. It is an empty
//!   marker trait whose supertraits were already implemented generically, so
//!   this only adds implementors.
//! - `akita-pcs` gained an additive, non-default `transcript-poseidon2` feature
//!   forwarding to `akita-transcript/transcript-poseidon2`, so the module
//!   exists and this file can be `#[cfg]`-gated without tripping
//!   `unexpected_cfgs` under `-D warnings`.
//!
//! Run with:
//!
//! ```text
//! CARGO_TARGET_DIR=~/.cache/akita-sponge-target \
//!   cargo test -p akita-pcs --features transcript-poseidon2 \
//!   --test poseidon2_transcript_roundtrip
//! ```
//!
//! Note that making Poseidon2 the *selected* backend workspace-wide is a
//! separate and much larger change; see the recipe in
//! `crates/akita-transcript/README.md`. Nothing here requires it.

#![cfg(feature = "transcript-poseidon2")]
#![allow(missing_docs)]

use akita_prover::{ComputeBackendSetup, CpuBackend};

mod common;

use akita_pcs::AkitaCommitmentScheme;
use akita_serialization::{AkitaDeserialize, AkitaSerialize};
use akita_transcript::poseidon2::{Alpha3, Alpha7, Poseidon2ByteSponge, Poseidon2Instance};
use akita_transcript::{AkitaTranscript, TranscriptSpongeBackend, TRANSCRIPT_BACKEND};
use akita_types::AkitaBatchedProof;
use common::*;

type DenseCfg = fp128::D128Full;
const DENSE_D: usize = DenseCfg::D;

const NV: usize = 16;
const SESSION: &[u8] = b"poseidon2/roundtrip";
const INSTANCE: &[u8] = b"akita/default-instance";

/// Maps each Poseidon2 sponge to the other alpha variant, for negative
/// control 2.
trait OtherInstance {
    type Other: TranscriptSpongeBackend;
}
impl OtherInstance for Poseidon2ByteSponge<Alpha3> {
    type Other = Poseidon2ByteSponge<Alpha7>;
}
impl OtherInstance for Poseidon2ByteSponge<Alpha7> {
    type Other = Poseidon2ByteSponge<Alpha3>;
}

/// Prove under sponge `S`, then verify under `S` (must accept) and under two
/// negative controls (must reject).
fn round_trip<S>(tag: &str)
where
    S: TranscriptSpongeBackend + OtherInstance,
{
    let layout = DenseCfg::get_params_for_batched_commitment(
        &akita_types::OpeningClaimsLayout::new(NV, 1).expect("singleton opening batch"),
    )
    .expect("layout");

    let evals = dense_field_evals(NV, 0x9051_1d02_0000);
    let poly = DensePoly::<F>::from_field_evals(NV, DENSE_D, &evals).expect("dense poly");

    let pt = random_point(NV, 0x9051_1d02_0001);
    let expected_opening = opening_from_poly::<DENSE_D, _>(&poly, &pt, &layout);

    let setup = AkitaCommitmentScheme::<DenseCfg>::setup_prover(NV, 1).unwrap();
    let prepared = CpuBackend.prepare_setup(&setup).unwrap();
    let stack =
        akita_prover::UniformProverStack::uniform(&CpuBackend, &prepared, setup.expanded.as_ref())
            .expect("stack");
    let verifier_setup = AkitaCommitmentScheme::<DenseCfg>::setup_verifier(&setup);
    let commit_input = std::slice::from_ref(&poly);
    let (commitment, hint) =
        AkitaCommitmentScheme::<DenseCfg>::commit::<_, _>(&setup, commit_input, &stack)
            .expect("commit");

    let poly_refs: [&DensePoly<F>; 1] = [&poly];
    let commitments = [commitment];
    let openings = [expected_opening];
    let opening_groups = [&openings[..]];
    let hints = vec![hint];

    let mut prover_transcript = AkitaTranscript::<F, S>::new_prover(SESSION, INSTANCE);
    let proof = AkitaCommitmentScheme::<DenseCfg>::batched_prove::<_, _, _>(
        &setup,
        prove_input(
            &pt[..],
            &poly_refs[..],
            &commitments[0],
            hints.into_iter().next().unwrap(),
        ),
        &stack,
        &mut prover_transcript,
        BasisMode::Lagrange,
        akita_types::SetupContributionMode::Direct,
    )
    .unwrap_or_else(|e| panic!("{tag}: prove failed: {e:?}"));

    // The transcript is not on the wire, so serialization must be unaffected.
    let mut serialized = Vec::new();
    let proof_shape = proof.shape();
    proof
        .serialize_compressed(&mut serialized)
        .expect("serialize");
    let decoded = AkitaBatchedProof::<F, F>::deserialize_compressed(
        &mut std::io::Cursor::new(serialized),
        &proof_shape,
    )
    .expect("deserialize");

    let mut verifier_transcript = AkitaTranscript::<F, S>::new_prover(SESSION, INSTANCE);
    AkitaCommitmentScheme::<DenseCfg>::batched_verify(
        &decoded,
        &verifier_setup,
        &mut verifier_transcript,
        verify_input(&pt[..], opening_groups[0], &commitments[0]),
        BasisMode::Lagrange,
        akita_types::SetupContributionMode::Direct,
    )
    .unwrap_or_else(|e| panic!("{tag}: verify failed: {e:?}"));

    // Negative control 1: a tampered opening claim must reject. This is the
    // control that proves verification is genuinely exercised here.
    //
    // NOTE on instance size, learned the hard way. At NV = 10 with this
    // config, an Akita proof contains no Fiat-Shamir-derived material at all:
    // Blake2b and Poseidon2 produce BYTE-IDENTICAL proofs, and every
    // transcript-sensitivity assertion below passes vacuously or fails
    // spuriously. NV must be large enough that folding actually happens.
    // `diagnostic_do_backends_produce_different_proofs` guards this: if it
    // ever reports "differ = false", every other assertion in this file is
    // meaningless regardless of whether it passes.
    let other_pt = random_point(NV, 0xdead_0000_0001);
    assert!(
        AkitaCommitmentScheme::<DenseCfg>::batched_verify(
            &decoded,
            &verifier_setup,
            &mut AkitaTranscript::<F, S>::new_prover(SESSION, INSTANCE),
            verify_input(&other_pt[..], opening_groups[0], &commitments[0]),
            BasisMode::Lagrange,
            akita_types::SetupContributionMode::Direct,
        )
        .is_err(),
        "{tag}: verification must fail against a different opening point"
    );

    // Negative control 2: the same session under the *other* alpha instance
    // must reject. This is what the distinct 28-byte capacity IVs buy us.
    let mut wrong_backend =
        AkitaTranscript::<F, S::Other>::new_prover(SESSION, INSTANCE);
    assert!(
        AkitaCommitmentScheme::<DenseCfg>::batched_verify(
            &decoded,
            &verifier_setup,
            &mut wrong_backend,
            verify_input(&pt[..], opening_groups[0], &commitments[0]),
            BasisMode::Lagrange,
            akita_types::SetupContributionMode::Direct,
        )
        .is_err(),
        "{tag}: verification must fail under a different Poseidon2 instance"
    );
}

/// The transparency claim, stated as agreement: Blake2b and Poseidon2 must
/// reach the *same* accept/reject decision on both negative controls. A
/// mismatched session label and a tampered opening claim must each be
/// rejected, identically, by both backends.
#[test]
fn control_backends_agree_on_negative_controls() {
    init_rayon_pool();
    run_on_large_stack(|| {
        let accepted = session_mismatch_accepts::<akita_transcript::TranscriptSponge>();
        println!("BLAKE2B: mismatched session label accepted = {accepted}");
        assert!(!accepted, "blake2b must reject a mismatched session label");
        let accepted_p2 = session_mismatch_accepts::<Poseidon2ByteSponge<Alpha3>>();
        println!("POSEIDON2 alpha=3: mismatched session label accepted = {accepted_p2}");
        assert!(!accepted_p2, "poseidon2 must reject a mismatched session label");
        assert_eq!(
            accepted, accepted_p2,
            "the two backends must agree on session-label sensitivity"
        );
        // Harness sanity: a control that MUST reject, so we know the
        // transcript is load-bearing here at all.
        let tampered_b2 = tampered_claim_accepts::<akita_transcript::TranscriptSponge>();
        let tampered_p2 = tampered_claim_accepts::<Poseidon2ByteSponge<Alpha3>>();
        println!("BLAKE2B: tampered opening claim accepted = {tampered_b2}");
        println!("POSEIDON2 alpha=3: tampered opening claim accepted = {tampered_p2}");
        assert!(!tampered_b2, "blake2b must reject a tampered claim");
        assert!(!tampered_p2, "poseidon2 must reject a tampered claim");
    });
}

/// Prove under `SESSION`, verify under a different session label, and report
/// whether verification accepted.
fn session_mismatch_accepts<S: TranscriptSpongeBackend>() -> bool {
    let layout = DenseCfg::get_params_for_batched_commitment(
        &akita_types::OpeningClaimsLayout::new(NV, 1).expect("singleton opening batch"),
    )
    .expect("layout");
    let evals = dense_field_evals(NV, 0x9051_1d02_0000);
    let poly = DensePoly::<F>::from_field_evals(NV, DENSE_D, &evals).expect("dense poly");
    let pt = random_point(NV, 0x9051_1d02_0001);
    let expected_opening = opening_from_poly::<DENSE_D, _>(&poly, &pt, &layout);
    let setup = AkitaCommitmentScheme::<DenseCfg>::setup_prover(NV, 1).unwrap();
    let prepared = CpuBackend.prepare_setup(&setup).unwrap();
    let stack =
        akita_prover::UniformProverStack::uniform(&CpuBackend, &prepared, setup.expanded.as_ref())
            .expect("stack");
    let verifier_setup = AkitaCommitmentScheme::<DenseCfg>::setup_verifier(&setup);
    let (commitment, hint) =
        AkitaCommitmentScheme::<DenseCfg>::commit::<_, _>(&setup, std::slice::from_ref(&poly), &stack)
            .expect("commit");
    let poly_refs: [&DensePoly<F>; 1] = [&poly];
    let commitments = [commitment];
    let openings = [expected_opening];
    let opening_groups = [&openings[..]];

    let mut prover_transcript = AkitaTranscript::<F, S>::new_prover(SESSION, INSTANCE);
    let proof = AkitaCommitmentScheme::<DenseCfg>::batched_prove::<_, _, _>(
        &setup,
        prove_input(&pt[..], &poly_refs[..], &commitments[0], hint),
        &stack,
        &mut prover_transcript,
        BasisMode::Lagrange,
        akita_types::SetupContributionMode::Direct,
    )
    .expect("prove");

    let mut wrong_session =
        AkitaTranscript::<F, S>::new_prover(b"poseidon2/roundtrip-WRONG", INSTANCE);
    AkitaCommitmentScheme::<DenseCfg>::batched_verify(
        &proof,
        &verifier_setup,
        &mut wrong_session,
        verify_input(&pt[..], opening_groups[0], &commitments[0]),
        BasisMode::Lagrange,
        akita_types::SetupContributionMode::Direct,
    )
    .is_ok()
}

/// Verify a valid proof against a DIFFERENT opening point. Must reject on any
/// backend; if it does not, the harness is not exercising verification.
fn tampered_claim_accepts<S: TranscriptSpongeBackend>() -> bool {
    let layout = DenseCfg::get_params_for_batched_commitment(
        &akita_types::OpeningClaimsLayout::new(NV, 1).expect("singleton opening batch"),
    )
    .expect("layout");
    let evals = dense_field_evals(NV, 0x9051_1d02_0000);
    let poly = DensePoly::<F>::from_field_evals(NV, DENSE_D, &evals).expect("dense poly");
    let pt = random_point(NV, 0x9051_1d02_0001);
    let expected_opening = opening_from_poly::<DENSE_D, _>(&poly, &pt, &layout);
    let setup = AkitaCommitmentScheme::<DenseCfg>::setup_prover(NV, 1).unwrap();
    let prepared = CpuBackend.prepare_setup(&setup).unwrap();
    let stack =
        akita_prover::UniformProverStack::uniform(&CpuBackend, &prepared, setup.expanded.as_ref())
            .expect("stack");
    let verifier_setup = AkitaCommitmentScheme::<DenseCfg>::setup_verifier(&setup);
    let (commitment, hint) =
        AkitaCommitmentScheme::<DenseCfg>::commit::<_, _>(&setup, std::slice::from_ref(&poly), &stack)
            .expect("commit");
    let poly_refs: [&DensePoly<F>; 1] = [&poly];
    let commitments = [commitment];
    let openings = [expected_opening];
    let opening_groups = [&openings[..]];

    let mut prover_transcript = AkitaTranscript::<F, S>::new_prover(SESSION, INSTANCE);
    let proof = AkitaCommitmentScheme::<DenseCfg>::batched_prove::<_, _, _>(
        &setup,
        prove_input(&pt[..], &poly_refs[..], &commitments[0], hint),
        &stack,
        &mut prover_transcript,
        BasisMode::Lagrange,
        akita_types::SetupContributionMode::Direct,
    )
    .expect("prove");

    // Same session, but a different evaluation point.
    let other_pt = random_point(NV, 0xdead_0000_0001);
    let mut verifier = AkitaTranscript::<F, S>::new_prover(SESSION, INSTANCE);
    AkitaCommitmentScheme::<DenseCfg>::batched_verify(
        &proof,
        &verifier_setup,
        &mut verifier,
        verify_input(&other_pt[..], opening_groups[0], &commitments[0]),
        BasisMode::Lagrange,
        akita_types::SetupContributionMode::Direct,
    )
    .is_ok()
}

/// Guard against a vacuous instance. If two backends produce byte-identical
/// proofs then no Fiat-Shamir challenge is load-bearing at this size, and
/// every other test in this file proves nothing. Measured: at NV = 10 the
/// proofs ARE identical (16384 bytes); at NV = 16 they differ (94764 bytes).
#[test]
fn diagnostic_do_backends_produce_different_proofs() {
    init_rayon_pool();
    run_on_large_stack(|| {
        let blake = proof_bytes::<akita_transcript::TranscriptSponge>();
        let p2a3 = proof_bytes::<Poseidon2ByteSponge<Alpha3>>();
        let p2a7 = proof_bytes::<Poseidon2ByteSponge<Alpha7>>();
        println!("proof len = {}", blake.len());
        println!("blake2b vs poseidon2-a3 proofs differ = {}", blake != p2a3);
        assert_ne!(blake, p2a3, "vacuous instance: transcript does not affect the proof");
        println!("poseidon2-a3 vs a7 proofs differ      = {}", p2a3 != p2a7);
    });
}

fn proof_bytes<S: TranscriptSpongeBackend>() -> Vec<u8> {
    let layout = DenseCfg::get_params_for_batched_commitment(
        &akita_types::OpeningClaimsLayout::new(NV, 1).expect("singleton opening batch"),
    )
    .expect("layout");
    let evals = dense_field_evals(NV, 0x9051_1d02_0000);
    let poly = DensePoly::<F>::from_field_evals(NV, DENSE_D, &evals).expect("dense poly");
    let pt = random_point(NV, 0x9051_1d02_0001);
    let expected_opening = opening_from_poly::<DENSE_D, _>(&poly, &pt, &layout);
    let setup = AkitaCommitmentScheme::<DenseCfg>::setup_prover(NV, 1).unwrap();
    let prepared = CpuBackend.prepare_setup(&setup).unwrap();
    let stack =
        akita_prover::UniformProverStack::uniform(&CpuBackend, &prepared, setup.expanded.as_ref())
            .expect("stack");
    let (commitment, hint) =
        AkitaCommitmentScheme::<DenseCfg>::commit::<_, _>(&setup, std::slice::from_ref(&poly), &stack)
            .expect("commit");
    let poly_refs: [&DensePoly<F>; 1] = [&poly];
    let commitments = [commitment];
    let mut t = AkitaTranscript::<F, S>::new_prover(SESSION, INSTANCE);
    let proof = AkitaCommitmentScheme::<DenseCfg>::batched_prove::<_, _, _>(
        &setup,
        prove_input(&pt[..], &poly_refs[..], &commitments[0], hint),
        &stack,
        &mut t,
        BasisMode::Lagrange,
        akita_types::SetupContributionMode::Direct,
    )
    .expect("prove");
    let _ = expected_opening;
    let mut out = Vec::new();
    proof.serialize_compressed(&mut out).expect("serialize");
    out
}

#[test]
fn poseidon2_alpha3_transcript_round_trip() {
    init_rayon_pool();
    run_on_large_stack(|| round_trip::<Poseidon2ByteSponge<Alpha3>>("alpha=3"));
}

#[test]
fn poseidon2_alpha7_transcript_round_trip() {
    init_rayon_pool();
    run_on_large_stack(|| round_trip::<Poseidon2ByteSponge<Alpha7>>("alpha=7"));
}

/// Enabling this feature must not change which backend the workspace selects.
#[test]
fn default_backend_is_still_blake2b() {
    assert_eq!(
        TRANSCRIPT_BACKEND, "blake2b",
        "enabling transcript-poseidon2 must not change the selected backend"
    );
    // And the two instances really are distinct parameter sets.
    assert_ne!(Alpha3::ROUNDS_P, Alpha7::ROUNDS_P);
    assert_ne!(Alpha3::DOMAIN_TAG, Alpha7::DOMAIN_TAG);
}
