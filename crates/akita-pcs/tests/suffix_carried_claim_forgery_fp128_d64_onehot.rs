//! `k = 1` half of the **Defect 1** regression
//! (`aerie/_docs/ABSORPTION-AUDIT.md`, and its "Verification of Defect 1"
//! appendix).
//!
//! At `EXT_DEGREE = 1` there is no extension-opening reduction, so the two
//! carried claims of the `G = 2` recursive suffix entered the fold straight
//! through `OpeningClaimsLayout::batched_eval_target(row_coefficients, ...)`.
//! With `row_coefficients = [1, 1]` the compensating `(+delta, -delta)` pair
//! produced by the cheating stage-3 sumcheck needed no partial fix-up at all:
//! the forgery is strictly easier here than at `k = 2`, and the unmodified
//! verifier accepted it (`Ok(())`) on this exact shape. That is why the defect
//! is upstream Akita's, not a consequence of the suffix-aligned EOR added on
//! this fork.
//!
//! The fix draws the batching coefficients from the transcript after the
//! claimed values are absorbed on this path too, so the pair must now be
//! rejected. This test requires the rejection.
//!
//! # Running it
//!
//! Needs the test-only `attack-probe` feature and the non-default
//! `schedules-fp128-d64-onehot-recursive` table (via `profile-ci`); roughly
//! 15 minutes, hence `#[ignore]`:
//!
//! ```text
//! CARGO_TARGET_DIR=~/.cache/akita-attack-target-k1 \
//!   cargo test --release -p akita-pcs \
//!   --features profile-ci,attack-probe \
//!   --test suffix_carried_claim_forgery_fp128_d64_onehot \
//!   -- --nocapture --test-threads=1 --ignored
//! ```

#![cfg(feature = "attack-probe")]
#![allow(missing_docs)]

use akita_config::{CommitmentConfig, ConservativeCommitmentConfig, RecursiveCommitmentConfig};
use akita_prover::{ComputeBackendSetup, CpuBackend};

mod common;

use akita_pcs::AkitaCommitmentScheme;
use akita_serialization::{AkitaDeserialize, AkitaSerialize};
use akita_transcript::AkitaTranscript;
use akita_types::{
    AkitaBatchedProof, AkitaScheduleLookupKey, BasisMode, OpeningClaims, OpeningClaimsLayout,
    PointVariableSelection, PolynomialGroupClaims, PolynomialGroupLayout, PrecommittedGroupParams,
    Schedule, SetupContributionMode, Step,
};
use common::*;

const TRANSCRIPT_DOMAIN: &[u8] = b"recursive_setup_e2e/generated_onehot";
const PRE_NV: usize = 16;
const FINAL_NV: usize = 32;
const PRE_GROUPS: usize = 2;
const PRE_GROUP_SIZE: usize = 1;
const FINAL_GROUP_SIZE: usize = 2;
const TOTAL_GROUP_SIZE: usize = PRE_GROUPS * PRE_GROUP_SIZE + FINAL_GROUP_SIZE;

type RecursiveOneHotCfg = RecursiveCommitmentConfig<OneHotCfg>;
type ConservativeOneHotCfg = ConservativeCommitmentConfig<OneHotCfg>;
type RecursiveOneHotScheme = AkitaCommitmentScheme<RecursiveOneHotCfg>;
type ConservativeOneHotScheme = AkitaCommitmentScheme<ConservativeOneHotCfg>;

fn multi_group_root_params(schedule: &Schedule) -> &LevelParams {
    match schedule.steps.first().expect("generated profile root step") {
        Step::Direct(direct) => direct.params.as_ref().expect("multi-group root params"),
        Step::Fold(fold) => &fold.params,
    }
}

fn schedule_uses_setup_prefix(schedule: &Schedule) -> bool {
    schedule
        .steps
        .iter()
        .any(|step| matches!(step, Step::Fold(fold) if fold.params.setup_prefix.is_some()))
}

fn generated_recursive_profile_key() -> (AkitaScheduleLookupKey, Vec<PolynomialGroupLayout>) {
    let pre_key = PolynomialGroupLayout::new(PRE_NV, PRE_GROUP_SIZE);
    let pre_params = ConservativeOneHotCfg::get_params_for_batched_commitment(
        &OpeningClaimsLayout::new(PRE_NV, PRE_GROUP_SIZE).expect("precommit batch"),
    )
    .expect("conservative precommit params");
    let pre_frozen = PrecommittedGroupParams::from_params(
        pre_key,
        &pre_params,
        akita_config::group_bound_policy_of::<ConservativeOneHotCfg>(),
    );
    let key = AkitaScheduleLookupKey {
        final_group: PolynomialGroupLayout::new(FINAL_NV, FINAL_GROUP_SIZE),
        precommitteds: vec![pre_frozen, pre_frozen],
    };
    (key, vec![pre_key; PRE_GROUPS])
}

/// Prove once, verify once. Returns the verifier verdict.
fn round_trip() -> Result<(), akita_field::AkitaError> {
    let (schedule_key, pre_keys) = generated_recursive_profile_key();
    let schedule =
        RecursiveOneHotCfg::runtime_schedule(schedule_key).expect("recursive profile schedule");
    assert!(
        schedule_uses_setup_prefix(&schedule),
        "shape must reach a setup-prefix level, otherwise the forgery surface is absent"
    );
    let root_params = multi_group_root_params(&schedule);

    let setup =
        RecursiveOneHotScheme::setup_prover(FINAL_NV, TOTAL_GROUP_SIZE).expect("recursive setup");
    let prepared = CpuBackend.prepare_setup(&setup).expect("prepared setup");
    let stack =
        akita_prover::UniformProverStack::uniform(&CpuBackend, &prepared, setup.expanded.as_ref())
            .expect("stack");

    let pre_layout = ConservativeOneHotCfg::get_params_for_batched_commitment(
        &OpeningClaimsLayout::new(PRE_NV, PRE_GROUP_SIZE).expect("precommit batch"),
    )
    .expect("conservative precommit params");
    let mut pre_polys_by_group = Vec::new();
    let mut pre_commitments = Vec::new();
    let mut pre_hints = Vec::new();
    for group_idx in 0..PRE_GROUPS {
        let poly = make_onehot_poly(&pre_layout, 0x0bee_fcaf_2026_0000 + group_idx as u64);
        let (commitment, hint) =
            ConservativeOneHotScheme::batched_commit(&setup, std::slice::from_ref(&poly), &stack)
                .expect("precommit group");
        pre_polys_by_group.push(vec![poly]);
        pre_commitments.push(commitment);
        pre_hints.push(hint);
    }

    let final_polys: Vec<OneHotPoly<F, u8>> = (0..FINAL_GROUP_SIZE)
        .map(|poly_idx| make_onehot_poly(root_params, 0x0bee_fcaf_2026_1000 + poly_idx as u64))
        .collect();
    let (final_commitment, final_hint) =
        RecursiveOneHotScheme::commit_final_group(&setup, &final_polys, &stack, pre_keys)
            .expect("final commitment");

    let point = random_point(FINAL_NV, 0xcafe_2026_0001);
    let pre_openings: Vec<Vec<F>> = pre_polys_by_group
        .iter()
        .map(|polys| {
            polys
                .iter()
                .map(|poly| opening_from_poly::<ONEHOT_D, _>(poly, &point[..PRE_NV], &pre_layout))
                .collect()
        })
        .collect();
    let final_openings: Vec<F> = final_polys
        .iter()
        .map(|poly| opening_from_poly::<ONEHOT_D, _>(poly, &point, root_params))
        .collect();

    let pre_refs_by_group: Vec<Vec<&OneHotPoly<F, u8>>> = pre_polys_by_group
        .iter()
        .map(|polys| polys.iter().collect())
        .collect();
    let final_refs: Vec<&OneHotPoly<F, u8>> = final_polys.iter().collect();

    let mut prover_groups = Vec::new();
    for (group_idx, openings) in pre_openings.iter().enumerate() {
        prover_groups.push(
            PolynomialGroupClaims::new(
                PointVariableSelection::prefix(PRE_NV, FINAL_NV).expect("pre point vars"),
                openings.clone(),
                pre_commitments[group_idx].clone(),
            )
            .expect("pre prover group"),
        );
    }
    prover_groups.push(
        PolynomialGroupClaims::new(
            PointVariableSelection::prefix(FINAL_NV, FINAL_NV).expect("final point vars"),
            final_openings.clone(),
            final_commitment.clone(),
        )
        .expect("final prover group"),
    );

    let mut prover_polys: Vec<&[&OneHotPoly<F, u8>]> = Vec::new();
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
    let proof = RecursiveOneHotScheme::batched_prove(
        &setup,
        prover_claims,
        &stack,
        &mut prover_transcript,
        BasisMode::Lagrange,
        SetupContributionMode::Recursive,
    )
    .expect("recursive proof");

    let shape = proof.shape();
    let mut bytes = Vec::new();
    proof.serialize_compressed(&mut bytes).expect("serialize");
    let proof =
        AkitaBatchedProof::<F, F>::deserialize_compressed(&mut std::io::Cursor::new(bytes), &shape)
            .expect("deserialize");

    let verifier_setup = setup.verifier_setup().expect("verifier setup");
    let mut verifier_groups = Vec::new();
    for (group_idx, openings) in pre_openings.iter().enumerate() {
        verifier_groups.push(
            PolynomialGroupClaims::new(
                PointVariableSelection::prefix(PRE_NV, FINAL_NV).expect("pre point vars"),
                openings.clone(),
                &pre_commitments[group_idx],
            )
            .expect("pre verifier group"),
        );
    }
    verifier_groups.push(
        PolynomialGroupClaims::new(
            PointVariableSelection::prefix(FINAL_NV, FINAL_NV).expect("final point vars"),
            final_openings,
            &final_commitment,
        )
        .expect("final verifier group"),
    );
    let verify_claims =
        OpeningClaims::from_groups(point, verifier_groups).expect("verifier claims");
    let mut verifier_transcript = AkitaTranscript::<F>::new(TRANSCRIPT_DOMAIN);
    RecursiveOneHotScheme::batched_verify(
        &proof,
        &verifier_setup,
        &mut verifier_transcript,
        verify_claims,
        BasisMode::Lagrange,
        SetupContributionMode::Recursive,
    )
}

#[test]
#[ignore = "~15 min; run explicitly (see the module docs for the command)"]
fn carried_claim_compensating_shift_is_rejected_at_k1() {
    init_rayon_pool();
    run_on_large_stack(|| {
        akita_prover::attack_probe::disarm();
        let honest = round_trip();
        eprintln!("[defect1-k1] honest verdict: {honest:?}");
        assert!(honest.is_ok(), "control proof must verify");

        akita_prover::attack_probe::arm();
        let attacked = round_trip();
        let notes = akita_prover::attack_probe::take_notes();
        akita_prover::attack_probe::disarm();
        for note in &notes {
            eprintln!("[defect1-k1] probe note: {note}");
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
            "the shifted pair must still satisfy the stage-3 final relation"
        );
        assert!(
            notes
                .iter()
                .any(|n| n.contains("k=1 suffix: absorbed the carried claim values as shipped")),
            "the k=1 attacker must absorb the FALSE carried claims, so that only the batching \
             coefficients can catch it"
        );
        eprintln!("[defect1-k1] forgery verdict: {attacked:?}");
        assert!(
            attacked.is_err(),
            "DEFECT 1 REGRESSION AT k = 1: the verifier ACCEPTED a proof whose two carried claims \
             are both false by +/-delta; see aerie/_docs/ABSORPTION-AUDIT.md."
        );
    });
}
