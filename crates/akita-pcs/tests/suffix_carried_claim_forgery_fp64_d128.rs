//! Regression test for **Defect 1** of `aerie/_docs/ABSORPTION-AUDIT.md`
//! (see also the appended "Verification of Defect 1" section): the `G = 2`
//! recursive suffix must not combine the carried setup-prefix claim and the
//! carried folded-witness claim with unit coefficients.
//!
//! # The forgery
//!
//! Stage 3 of level `j` ends in a single linear equation in the two
//! prover-supplied carried values,
//! `final_claim = c1 * next_w_eval + c2 * setup_prefix_eval`. A prover that
//! runs the stage-3 sumcheck "honestly relative to a lie" lands at
//! `T + Gamma`, and can absorb `Gamma` by shipping
//! `(setup_prefix_eval, next_w_eval) = (O_S + delta, O_w - delta)` with
//! `delta = Gamma / (c2 - c1)`. At level `j + 1` the old fold enforced only
//! `O_S + O_w = s' + v_w'`, which the compensating pair satisfies identically,
//! and the EOR per-claim partial checks are head-independent, so a constant
//! `+/-delta` on each claim's column partials passes them too. This was
//! executed end to end against the unmodified verifier and **accepted**
//! (`Ok(())`) at `nv = 23`, on a schedule with two setup-prefix levels, both
//! attacked with independent deltas.
//!
//! # What must happen now
//!
//! The batching coefficients are squeezed from the transcript after the
//! claimed values are absorbed, so the fold enforces
//! `sum_l rho_l * O_l = sum_l rho_l * v_l`; a nonzero `(+delta, -delta)`
//! survives that only when `rho_0 = rho_1`, i.e. with probability `1/|E|`.
//! This test therefore requires the verifier to REJECT.
//!
//! # Running it
//!
//! Needs the test-only `attack-probe` feature (the adversarial prover
//! injections in `akita_prover::attack_probe`), and takes roughly 45 minutes
//! (two setups, two proofs, two verifies at `nv = 23`), so it is `#[ignore]`d:
//!
//! ```text
//! CARGO_TARGET_DIR=~/.cache/akita-attack-target \
//!   cargo test --release -p akita-pcs --features attack-probe \
//!   --test suffix_carried_claim_forgery_fp64_d128 \
//!   -- --nocapture --test-threads=1 --ignored
//! ```
//!
//! The cheap always-on companion is
//! `suffix_carried_claim_batching_coefficients_are_transcript_bound` in
//! `crates/akita-prover/src/protocol/core/tests.rs`, which pins the same
//! property at unit-test cost.

#![cfg(feature = "attack-probe")]
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
    lagrange_weights, AkitaBatchedProof, AkitaScheduleLookupKey, AkitaVerifierSetup, BasisMode,
    Commitment, OpeningClaims, PointVariableSelection, PolynomialGroupClaims,
    PolynomialGroupLayout, Schedule, SetupContributionMode, Step,
};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

const TRANSCRIPT_DOMAIN: &[u8] = b"recursive_setup_fp64_d128_e2e/dense_bound18";
const PRE_GROUPS: usize = 2;
const FINAL_GROUP_SIZE: usize = 2;
const TOTAL_GROUP_SIZE: usize = PRE_GROUPS + FINAL_GROUP_SIZE;
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
            <E as ExtField<F>>::from_base_slice(&limbs)
        })
        .collect();
    for coord in point
        .iter_mut()
        .take(alpha_bits.min(num_vars))
        .skip(trace_inner)
    {
        *coord = <E as LiftBase<F>>::lift_base(F::zero());
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
        .map(|_| F::from_canonical_u128_reduced(u128::from(rng.gen_range(0..max_exclusive))))
        .collect();
    let poly = DensePoly::<F>::from_field_evals(num_vars, RING_D, &evals).expect("dense poly");
    (poly, evals)
}

fn dense_opening_ext(evals: &[F], point: &[E]) -> E {
    let weights = lagrange_weights::<E>(point).expect("lagrange weights");
    assert_eq!(weights.len(), evals.len());
    weights
        .iter()
        .zip(evals.iter())
        .fold(E::zero(), |acc, (w, e)| {
            acc + *w * <E as LiftBase<F>>::lift_base(*e)
        })
}

fn dense_recursive_profile_key_at(pre_nv: usize, final_nv: usize) -> AkitaScheduleLookupKey {
    let pre_key = PolynomialGroupLayout::new(pre_nv, 1);
    let precommitted =
        <MainCfg as CommitmentConfig>::precommitted_group_params(0, pre_key).expect("pre params");
    AkitaScheduleLookupKey {
        final_group: PolynomialGroupLayout::new(final_nv, FINAL_GROUP_SIZE),
        precommitteds: vec![precommitted; PRE_GROUPS],
    }
}

fn schedule_uses_setup_prefix(schedule: &Schedule) -> bool {
    schedule
        .steps
        .iter()
        .any(|step| matches!(step, Step::Fold(fold) if fold.params.setup_prefix.is_some()))
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

fn prove_at(pre_nv: usize, final_nv: usize) -> ProvenInstance {
    let schedule_key = dense_recursive_profile_key_at(pre_nv, final_nv);
    let schedule = RecursiveCfg::runtime_schedule(schedule_key).expect("schedule");
    assert!(
        schedule_uses_setup_prefix(&schedule),
        "shape must reach a setup-prefix level, otherwise the forgery surface is absent"
    );

    let setup = RecursiveScheme::setup_prover(final_nv, TOTAL_GROUP_SIZE).expect("setup");
    let prepared = CpuBackend.prepare_setup(&setup).expect("prepared setup");
    let stack = UniformProverStack::uniform(&CpuBackend, &prepared, setup.expanded.as_ref())
        .expect("stack");

    let point = subfield_random_point(final_nv, 0x0ae2_2026_0731);

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
            .expect("final commitment");

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
    .expect("prover data");

    let mut prover_transcript = AkitaTranscript::<F>::new(TRANSCRIPT_DOMAIN);
    let proof = RecursiveScheme::batched_prove(
        &setup,
        prover_claims,
        &stack,
        &mut prover_transcript,
        BasisMode::Lagrange,
        SetupContributionMode::Recursive,
    )
    .expect("recursive proof");

    let shape = proof.shape();
    let mut proof_bytes = Vec::new();
    proof
        .serialize_compressed(&mut proof_bytes)
        .expect("serialize");

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

fn verify(instance: &ProvenInstance) -> Result<(), akita_field::AkitaError> {
    let proof = AkitaBatchedProof::<F, E>::deserialize_compressed(
        &mut std::io::Cursor::new(instance.proof_bytes.clone()),
        &instance.shape,
    )
    .expect("deserialize");
    let mut transcript = AkitaTranscript::<F>::new(TRANSCRIPT_DOMAIN);
    RecursiveScheme::batched_verify(
        &proof,
        &instance.verifier_setup,
        &mut transcript,
        verifier_claims(instance),
        BasisMode::Lagrange,
        SetupContributionMode::Recursive,
    )
}

fn run_case(pre_nv: usize, final_nv: usize) {
    // Control: the honest prover must still round-trip. If this fails the
    // rejection below proves nothing.
    akita_prover::attack_probe::disarm();
    let honest = prove_at(pre_nv, final_nv);
    let honest_verdict = verify(&honest);
    eprintln!("[defect1] honest verdict at nv={final_nv}: {honest_verdict:?}");
    assert!(honest_verdict.is_ok(), "control proof must verify");

    // Forgery: compensating (+delta, -delta) on the two carried claims.
    akita_prover::attack_probe::arm();
    let tampered = prove_at(pre_nv, final_nv);
    let notes = akita_prover::attack_probe::take_notes();
    akita_prover::attack_probe::disarm();
    for note in &notes {
        eprintln!("[defect1] probe note: {note}");
    }
    assert!(
        notes.iter().any(|n| n.contains("gamma_is_zero=false")),
        "probe must have produced a nonzero stage-3 sumcheck error"
    );
    assert!(
        notes.iter().any(|n| n.contains("delta_is_zero=false")),
        "probe must have produced a nonzero compensating delta"
    );
    assert!(
        notes
            .iter()
            .any(|n| n.contains("relation_holds_after_shift=true")),
        "the shifted pair must still satisfy the stage-3 final relation, otherwise the test \
         would be rejected for the wrong reason"
    );
    assert!(
        notes.iter().any(|n| n.contains("suffix EOR: shifted")),
        "probe must have reached the suffix EOR and re-based the column partials"
    );

    let attack_verdict = verify(&tampered);
    eprintln!("[defect1] forgery verdict at nv={final_nv}: {attack_verdict:?}");
    assert!(
        attack_verdict.is_err(),
        "DEFECT 1 REGRESSION at nv={final_nv}: the verifier ACCEPTED a proof whose two carried \
         claims are both false by +/-delta. The G = 2 recursive suffix is batching with unit \
         coefficients again; see aerie/_docs/ABSORPTION-AUDIT.md."
    );
}

/// `nv = 23` is the smallest shape in this profile family whose planner emits
/// a setup-prefix step at all (`nv = 22` emits none), so it is the smallest
/// shape that reaches the two-group carried-claim suffix batch. There is no
/// cheaper minimisation available inside this family; the cheap companion is
/// the prover unit fixture named in the module docs.
#[test]
#[ignore = "~45 min; run explicitly (see the module docs for the command)"]
fn carried_claim_compensating_shift_is_rejected_nv23() {
    run_on_large_stack(|| run_case(11, 23));
}
