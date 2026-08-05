//! End-to-end coverage for recursive setup offloading at ring dimension 128
//! (`specs/mixed-d-setup-delegation.md`).
//!
//! Mirrors `recursive_setup_e2e.rs` (fp128 D64OneHot) at the Aerie fp64
//! preset `D128FullBound18`: the table-less dense recursive profile key is a
//! two-polynomial final group at `nv = 23` plus two singleton dense
//! precommits at `nv = 11`. A successful recursive proof exercises the
//! offloaded setup-contribution path (stage-3 setup-product sumcheck plus
//! carried setup-prefix opening) at `D = 128`, plus cross-mode rejection.

#![allow(missing_docs)]

use akita_config::proof_optimized::fp64;
use akita_config::{CommitmentConfig, ConservativeCommitmentConfig, RecursiveCommitmentConfig};
use akita_field::{CanonicalField, ExtField, LiftBase};
use akita_pcs::AkitaCommitmentScheme;
use akita_prover::{
    ComputeBackendSetup, CpuBackend, DensePoly, ProverOpeningData, UniformProverStack,
};
use akita_serialization::{AkitaDeserialize, AkitaSerialize};
use akita_transcript::AkitaTranscript;
use akita_types::{
    lagrange_weights, AkitaBatchedProof, AkitaBatchedRootProof, AkitaScheduleLookupKey,
    AkitaVerifierSetup, BasisMode, Commitment, OpeningClaims, PointVariableSelection,
    PolynomialGroupClaims, PolynomialGroupLayout, Schedule, SetupContributionMode, Step,
};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

const TRANSCRIPT_DOMAIN: &[u8] = b"recursive_setup_fp64_d128_e2e/dense_bound18";
const PRE_NV: usize = 11;
const FINAL_NV: usize = 23;
const PRE_GROUPS: usize = 2;
const FINAL_GROUP_SIZE: usize = 2;
const TOTAL_GROUP_SIZE: usize = PRE_GROUPS + FINAL_GROUP_SIZE;
/// Application-guaranteed source range for `D128FullBound18` data.
const BOUND18_MAX_EXCLUSIVE: u64 = 1u64 << 17;
const STACK_SIZE: usize = 256 * 1024 * 1024;

type F = fp64::Field;
type E = fp64::ExtensionField;
type MainCfg = fp64::D128FullBound18;
type RecursiveCfg = RecursiveCommitmentConfig<MainCfg>;
type ConservativeCfg = ConservativeCommitmentConfig<MainCfg>;
type RecursiveScheme = AkitaCommitmentScheme<RecursiveCfg>;
type ConservativeScheme = AkitaCommitmentScheme<ConservativeCfg>;

const RING_D: usize = MainCfg::D;

fn run_on_large_stack(f: impl FnOnce() + Send + 'static) {
    std::thread::Builder::new()
        .stack_size(STACK_SIZE)
        .spawn(f)
        .expect("failed to spawn thread")
        .join()
        .expect("test thread panicked");
}

/// Random extension opening point compatible with ring-subfield packing:
/// inactive inner coordinates in `[log2(ring_d/ext_degree), log2(ring_d))`
/// are zeroed (see `prepare_opening_point`).
fn subfield_random_point(num_vars: usize, seed: u64) -> Vec<E> {
    let ext_degree = <E as ExtField<F>>::EXT_DEGREE;
    let trace_inner = (RING_D / ext_degree).trailing_zeros() as usize;
    let alpha_bits = RING_D.trailing_zeros() as usize;
    let mut rng = StdRng::seed_from_u64(seed);
    let mut point: Vec<E> = (0..num_vars)
        .map(|_| {
            let limbs: Vec<F> = (0..ext_degree)
                .map(|_| F::from_canonical_u128_reduced(rng.r#gen::<u128>()))
                .collect();
            E::from_base_slice(&limbs)
        })
        .collect();
    for idx in trace_inner..alpha_bits {
        if idx < point.len() {
            point[idx] = E::zero();
        }
    }
    point
}

fn make_seeded_dense_poly_bounded(
    num_vars: usize,
    max_exclusive: u64,
    seed: u64,
) -> (DensePoly<F>, Vec<F>) {
    let len = 1usize << num_vars;
    let mut rng = StdRng::seed_from_u64(seed);
    let evals: Vec<F> = (0..len)
        .map(|_| F::from_u64(rng.gen_range(0..max_exclusive)))
        .collect();
    let poly = DensePoly::<F>::from_field_evals(num_vars, RING_D, &evals).expect("dense poly");
    (poly, evals)
}

fn dense_opening_ext(evals: &[F], point: &[E]) -> E {
    let weights = lagrange_weights(point).expect("dense ext weights");
    evals
        .iter()
        .zip(weights.iter())
        .fold(E::zero(), |acc, (&coeff, &weight)| {
            acc + weight * E::lift_base(coeff)
        })
}

fn dense_recursive_profile_key_at(pre_nv: usize, final_nv: usize) -> AkitaScheduleLookupKey {
    // Same static hook the prove-time schedule-key derivation uses
    // (`recursive_schedule_key` -> inner `precommitted_group_params`).
    let pre_key = PolynomialGroupLayout::new(pre_nv, 1);
    let precommitted =
        <MainCfg as CommitmentConfig>::precommitted_group_params(0, pre_key).expect("pre params");
    AkitaScheduleLookupKey {
        final_group: PolynomialGroupLayout::new(final_nv, FINAL_GROUP_SIZE),
        precommitteds: vec![precommitted, precommitted],
    }
}

fn dense_recursive_profile_key() -> AkitaScheduleLookupKey {
    dense_recursive_profile_key_at(PRE_NV, FINAL_NV)
}

fn schedule_uses_setup_prefix(schedule: &Schedule) -> bool {
    schedule.steps.iter().any(|step| {
        matches!(
            step,
            Step::Fold(fold) if fold.params.setup_prefix.is_some()
        )
    })
}

fn proof_has_recursive_setup_sumcheck(proof: &AkitaBatchedProof<F, E>) -> bool {
    let root_has_stage3 = match &proof.root {
        AkitaBatchedRootProof::Fold(fold) => fold.stage3_sumcheck_proof.is_some(),
        AkitaBatchedRootProof::Terminal(_) | AkitaBatchedRootProof::ZeroFold { .. } => false,
    };
    let suffix_has_stage3 = proof
        .steps
        .iter()
        .any(|step| step.stage3_sumcheck_proof().is_some());
    root_has_stage3 || suffix_has_stage3
}

struct ProvenInstance {
    pre_nv: usize,
    final_nv: usize,
    proof_bytes: Vec<u8>,
    shape: akita_types::AkitaBatchedProofShape,
    verifier_setup: AkitaVerifierSetup<F>,
    point: Vec<E>,
    pre_openings: Vec<Vec<E>>,
    final_openings: Vec<E>,
    pre_commitments: Vec<Commitment<F>>,
    final_commitment: Commitment<F>,
}

fn prove_dense_recursive_instance() -> ProvenInstance {
    prove_dense_recursive_instance_at(PRE_NV, FINAL_NV)
}

fn prove_dense_recursive_instance_at(pre_nv: usize, final_nv: usize) -> ProvenInstance {
    let schedule_key = dense_recursive_profile_key_at(pre_nv, final_nv);
    let schedule = RecursiveCfg::runtime_schedule(schedule_key)
        .expect("dense fp64 D128 recursive profile schedule");
    assert!(
        schedule_uses_setup_prefix(&schedule),
        "dense fp64 D128 recursive profile must carry setup-prefix metadata"
    );

    let setup =
        RecursiveScheme::setup_prover(final_nv, TOTAL_GROUP_SIZE).expect("recursive fp64 setup");
    assert!(
        !setup.prefix_slots.is_empty(),
        "recursive fp64 D128 setup must precompute setup-prefix slots"
    );
    let prepared = CpuBackend.prepare_setup(&setup).expect("prepared setup");
    let stack = UniformProverStack::uniform(&CpuBackend, &prepared, setup.expanded.as_ref())
        .expect("stack");

    let point = subfield_random_point(final_nv, 0x0ae2_2026_0731);

    // Dense bound-18 precommitted groups under the conservative adapter.
    let mut pre_polys_by_group = Vec::new();
    let mut pre_evals_by_group = Vec::new();
    let mut pre_commitments = Vec::new();
    let mut pre_hints = Vec::new();
    for group_idx in 0..PRE_GROUPS {
        let (poly, evals) = make_seeded_dense_poly_bounded(
            pre_nv,
            BOUND18_MAX_EXCLUSIVE,
            0x0ae2_2026_1000 + group_idx as u64,
        );
        let polys = vec![poly];
        let (commitment, hint) =
            ConservativeScheme::batched_commit(&setup, &polys[..], &stack).expect("precommit");
        pre_polys_by_group.push(polys);
        pre_evals_by_group.push(evals);
        pre_commitments.push(commitment);
        pre_hints.push(hint);
    }
    let pre_openings: Vec<Vec<E>> = pre_evals_by_group
        .iter()
        .map(|evals| vec![dense_opening_ext(evals, &point[..pre_nv])])
        .collect();

    // Dense bound-18 final group.
    let mut final_polys = Vec::new();
    let mut final_openings = Vec::new();
    for poly_idx in 0..FINAL_GROUP_SIZE {
        let (poly, evals) = make_seeded_dense_poly_bounded(
            final_nv,
            BOUND18_MAX_EXCLUSIVE,
            0x0ae2_2026_2000 + poly_idx as u64,
        );
        final_openings.push(dense_opening_ext(&evals, &point));
        final_polys.push(poly);
    }
    let pre_keys = vec![PolynomialGroupLayout::new(pre_nv, 1); PRE_GROUPS];
    let (final_commitment, final_hint) =
        RecursiveScheme::commit_final_group(&setup, &final_polys, &stack, pre_keys)
            .expect("final recursive fp64 commitment");

    let pre_refs_by_group: Vec<Vec<&DensePoly<F>>> = pre_polys_by_group
        .iter()
        .map(|polys| polys.iter().collect())
        .collect();
    let final_refs: Vec<&DensePoly<F>> = final_polys.iter().collect();

    let mut prover_groups = Vec::new();
    for (group_idx, openings) in pre_openings.iter().enumerate() {
        prover_groups.push(
            PolynomialGroupClaims::new(
                PointVariableSelection::prefix(pre_nv, final_nv).expect("pre point vars"),
                openings.clone(),
                pre_commitments[group_idx].clone(),
            )
            .expect("pre prover group"),
        );
    }
    prover_groups.push(
        PolynomialGroupClaims::new(
            PointVariableSelection::prefix(final_nv, final_nv).expect("final point vars"),
            final_openings.clone(),
            final_commitment.clone(),
        )
        .expect("final prover group"),
    );

    let mut prover_polys: Vec<&[&DensePoly<F>]> = Vec::new();
    for refs in &pre_refs_by_group {
        prover_polys.push(&refs[..]);
    }
    prover_polys.push(&final_refs[..]);
    let mut prover_hints = pre_hints;
    prover_hints.push(final_hint);

    let prover_claims = ProverOpeningData::new(
        OpeningClaims::from_groups(point.clone(), prover_groups).expect("prover claims"),
        prover_hints,
        prover_polys,
    )
    .expect("dense fp64 recursive prover data");

    let mut prover_transcript = AkitaTranscript::<F>::new(TRANSCRIPT_DOMAIN);
    let proof = RecursiveScheme::batched_prove(
        &setup,
        prover_claims,
        &stack,
        &mut prover_transcript,
        BasisMode::Lagrange,
        SetupContributionMode::Recursive,
    )
    .expect("dense fp64 recursive proof");
    assert!(
        proof_has_recursive_setup_sumcheck(&proof),
        "recursive fp64 D128 proof must carry stage-3 setup sumcheck evidence"
    );

    let shape = proof.shape();
    let mut proof_bytes = Vec::new();
    proof
        .serialize_compressed(&mut proof_bytes)
        .expect("serialize dense fp64 recursive proof");

    let verifier_setup = setup.verifier_setup().expect("verifier setup");
    ProvenInstance {
        pre_nv,
        final_nv,
        proof_bytes,
        shape,
        verifier_setup,
        point,
        pre_openings,
        final_openings,
        pre_commitments,
        final_commitment,
    }
}

fn verifier_claims(instance: &ProvenInstance) -> OpeningClaims<'_, E, &Commitment<F>> {
    let mut verifier_groups = Vec::new();
    for (group_idx, openings) in instance.pre_openings.iter().enumerate() {
        verifier_groups.push(
            PolynomialGroupClaims::new(
                PointVariableSelection::prefix(instance.pre_nv, instance.final_nv)
                    .expect("pre point vars"),
                openings.clone(),
                &instance.pre_commitments[group_idx],
            )
            .expect("pre verifier group"),
        );
    }
    verifier_groups.push(
        PolynomialGroupClaims::new(
            PointVariableSelection::prefix(instance.final_nv, instance.final_nv)
                .expect("final point vars"),
            instance.final_openings.clone(),
            &instance.final_commitment,
        )
        .expect("final verifier group"),
    );
    OpeningClaims::from_groups(instance.point.clone(), verifier_groups).expect("verifier claims")
}

/// Small-shape debug variant of the round trip (run explicitly with
/// `--ignored`): same pipeline at `nv = 20` for fast failure localization.
#[test]
#[ignore = "debug-size variant; the nv=23 round trip is the acceptance gate"]
fn recursive_fp64_d128_bound18_nv20_round_trip_debug() {
    run_on_large_stack(|| {
        let instance = prove_dense_recursive_instance_at(11, 22);
        let proof = AkitaBatchedProof::<F, E>::deserialize_compressed(
            &mut std::io::Cursor::new(instance.proof_bytes.clone()),
            &instance.shape,
        )
        .expect("deserialize");
        let mut verifier_transcript = AkitaTranscript::<F>::new(TRANSCRIPT_DOMAIN);
        RecursiveScheme::batched_verify(
            &proof,
            &instance.verifier_setup,
            &mut verifier_transcript,
            verifier_claims(&instance),
            BasisMode::Lagrange,
            SetupContributionMode::Recursive,
        )
        .expect("verify at nv=20");
    });
}

/// `D128FullBound6` planning at the Aerie `nv = 26` shape: the recursive
/// profile key must resolve to a schedule that carries setup-prefix metadata.
/// (The full prove/verify round trip is covered at `Bound18` `nv = 23`; the
/// two presets share every code path exercised by this change.)
#[test]
fn recursive_fp64_d128_bound6_nv26_plans_with_setup_prefix() {
    run_on_large_stack(|| {
        type Bound6Cfg = fp64::D128FullBound6;
        type RecursiveBound6Cfg = RecursiveCommitmentConfig<Bound6Cfg>;
        const BOUND6_FINAL_NV: usize = 26;
        const BOUND6_PRE_NV: usize = BOUND6_FINAL_NV / 2;

        let pre_key = PolynomialGroupLayout::new(BOUND6_PRE_NV, 1);
        let precommitted = <Bound6Cfg as CommitmentConfig>::precommitted_group_params(0, pre_key)
            .expect("bound6 pre params");
        let key = AkitaScheduleLookupKey {
            final_group: PolynomialGroupLayout::new(BOUND6_FINAL_NV, FINAL_GROUP_SIZE),
            precommitteds: vec![precommitted, precommitted],
        };
        let schedule = RecursiveBound6Cfg::runtime_schedule(key)
            .expect("dense fp64 D128 bound6 recursive profile schedule");
        assert!(
            schedule_uses_setup_prefix(&schedule),
            "bound6 nv=26 recursive profile must carry setup-prefix metadata"
        );
    });
}

/// `D128FullBound18` planning at the Aerie `nv = 23` shape (cheap: schedule
/// resolution only; the setup/prove pipeline is exercised by the ignored
/// round-trip test below).
#[test]
fn recursive_fp64_d128_bound18_nv23_plans_with_setup_prefix() {
    run_on_large_stack(|| {
        let schedule = RecursiveCfg::runtime_schedule(dense_recursive_profile_key())
            .expect("dense fp64 D128 bound18 recursive profile schedule");
        assert!(
            schedule_uses_setup_prefix(&schedule),
            "bound18 nv=23 recursive profile must carry setup-prefix metadata"
        );
    });
}

/// Full recursive round trip at fp64 D128 `nv = 23`.
///
/// Requires the shared extension-opening reduction over the carried
/// setup-prefix claim (`specs/eor-setup-prefix-absorption.md`): at
/// `EXT_DEGREE = 2` every suffix level runs the EOR, and the setup-prefix
/// level's carried claim joins it suffix-aligned; the reduced openings feed
/// the existing `G = 2` batched front-end. Heavy in debug; the release run is
/// the acceptance gate.
#[test]
fn recursive_fp64_d128_bound18_nv23_round_trips_and_rejects_stripped_stage3() {
    run_on_large_stack(|| {
        let instance = prove_dense_recursive_instance();

        let proof = AkitaBatchedProof::<F, E>::deserialize_compressed(
            &mut std::io::Cursor::new(instance.proof_bytes.clone()),
            &instance.shape,
        )
        .expect("deserialize dense fp64 recursive proof");

        let mut verifier_transcript = AkitaTranscript::<F>::new(TRANSCRIPT_DOMAIN);
        RecursiveScheme::batched_verify(
            &proof,
            &instance.verifier_setup,
            &mut verifier_transcript,
            verifier_claims(&instance),
            BasisMode::Lagrange,
            SetupContributionMode::Recursive,
        )
        .expect("dense fp64 recursive verify");

        // Mode is load-bearing through the schedule: a recursive-schedule
        // proof stripped of its stage-3 sumcheck must be rejected. (The
        // verifier takes the per-level mode from the schedule; the caller
        // argument does not override it.)
        let mut stripped = proof.clone();
        if let AkitaBatchedRootProof::Fold(fold) = &mut stripped.root {
            fold.stage3_sumcheck_proof = None;
        }
        let mut stripped_transcript = AkitaTranscript::<F>::new(TRANSCRIPT_DOMAIN);
        assert!(
            RecursiveScheme::batched_verify(
                &stripped,
                &instance.verifier_setup,
                &mut stripped_transcript,
                verifier_claims(&instance),
                BasisMode::Lagrange,
                SetupContributionMode::Recursive,
            )
            .is_err(),
            "recursive fp64 D128 proof without stage-3 evidence must be rejected"
        );

        // Tampered opening must reject.
        let mut tampered = instance;
        tampered.final_openings[0] += E::one();
        let mut tampered_transcript = AkitaTranscript::<F>::new(TRANSCRIPT_DOMAIN);
        assert!(
            RecursiveScheme::batched_verify(
                &proof,
                &tampered.verifier_setup,
                &mut tampered_transcript,
                verifier_claims(&tampered),
                BasisMode::Lagrange,
                SetupContributionMode::Recursive,
            )
            .is_err(),
            "tampered dense fp64 recursive opening must reject"
        );
    });
}
