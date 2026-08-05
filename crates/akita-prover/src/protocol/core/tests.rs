use super::*;
use crate::RecursiveWitnessFlat;
use akita_config::{proof_optimized::fp128::D64OneHot, CommitmentConfig};
use akita_field::{Fp32, FpExt2, NegOneNr};
use akita_transcript::AkitaTranscript;
use akita_types::{
    OpeningClaims, OpeningClaimsLayout, PointVariableSelection, PolynomialGroupClaims,
    PolynomialGroupLayout,
};

type F = Fp32<251>;
type E = FpExt2<F, NegOneNr>;

#[test]
fn recursive_extension_opening_reduction_pads_to_opening_cube() {
    let logical_w = RecursiveWitnessFlat::from_i8_digits(vec![1, -1, 2, 0, 3, -2]);
    let point = [
        E::new(F::from_u64(2), F::from_u64(3)),
        E::new(F::from_u64(5), F::from_u64(7)),
        E::new(F::from_u64(11), F::from_u64(13)),
    ];
    let logical_polys = [&logical_w];

    let mut transcript =
        AkitaTranscript::<F>::new(b"test/recursive-extension-opening-reduction-padding");
    let opening_batch = OpeningClaims::from_groups(
        point.to_vec(),
        vec![PolynomialGroupClaims::new(
            PointVariableSelection::prefix(point.len(), point.len()).expect("point vars"),
            vec![E::zero()],
            (),
        )
        .expect("group claims")],
    )
    .expect("opening batch");
    let proved = prove_extension_opening_reduction::<F, E, _, RecursiveWitnessFlat, _, 2>(
        &crate::compute::CpuBackend,
        None,
        &logical_polys,
        &opening_batch,
        EorClaimAlignment::Flat,
        true,
        &mut transcript,
        "recursive",
    )
    .expect("padded logical witnesses should reduce over the opening cube");

    assert_eq!(
        proved.reduction.proof.partials.len(),
        <E as ExtField<F>>::EXT_DEGREE
    );
    assert_eq!(proved.reduction.proof.num_rounds(), point.len() - 1);
}

#[test]
fn proof_schedule_from_layout_includes_entire_batch() {
    let batch = OpeningClaimsLayout::from_groups(vec![
        PolynomialGroupLayout::new(16, 1),
        PolynomialGroupLayout::new(32, 2),
    ])
    .expect("multi-group shape");
    assert_eq!(batch.num_groups(), 2);
    let schedule = D64OneHot::get_params_for_prove(&batch).expect("multi-group schedule");
    let root_params =
        akita_types::multi_group_root_commit_params(&schedule).expect("multi-group root params");
    assert_eq!(root_params.precommitted_groups.len(), 1);
    assert_eq!(
        root_params.precommitted_groups[0].layout.group,
        PolynomialGroupLayout::new(16, 1)
    );
}

/// Suffix-aligned shared EOR (`specs/eor-setup-prefix-absorption.md`): a
/// two-group batch whose shorter group's point is a SUFFIX of the shared point
/// (the recursive carried-claim geometry). The shorter group lifts by strided
/// (LOW-variable) constant extension; the prover's internal identity checks
/// (input claim from terms, final-oracle and transparent-factor equality)
/// gate the whole reduction, and the emitted per-claim partials must open
/// each claim at its own routed point.
#[test]
fn suffix_aligned_grouped_reduction_reduces_carried_claims() {
    fn lift(digit: i8) -> E {
        if digit >= 0 {
            E::lift_base(F::from_u64(digit as u64))
        } else {
            -E::lift_base(F::from_u64((-(digit as i64)) as u64))
        }
    }

    // Shorter group A: 2 vars (claims at the shared point's suffix); full
    // group B: 3 vars.
    let digits_a: Vec<i8> = vec![1, -2, 3, 0];
    let digits_b: Vec<i8> = vec![2, 5, -1, 4, 0, -3, 7, 1];
    let w_a = RecursiveWitnessFlat::from_i8_digits(digits_a.clone());
    let w_b = RecursiveWitnessFlat::from_i8_digits(digits_b.clone());
    let point = [
        E::new(F::from_u64(2), F::from_u64(3)),
        E::new(F::from_u64(5), F::from_u64(7)),
        E::new(F::from_u64(11), F::from_u64(13)),
    ];

    let groups = vec![
        PolynomialGroupClaims::new(
            PointVariableSelection::suffix(2, 3).expect("suffix selection"),
            vec![E::zero()],
            (),
        )
        .expect("group a"),
        PolynomialGroupClaims::new(
            PointVariableSelection::suffix(3, 3).expect("full selection"),
            vec![E::zero()],
            (),
        )
        .expect("group b"),
    ];
    let opening_batch = OpeningClaims::from_groups_allow_custom_routing(point.to_vec(), groups)
        .expect("suffix batch");
    let layout = opening_batch.layout().expect("layout");

    let mut transcript = AkitaTranscript::<F>::new(b"test/suffix-aligned-grouped-eor");
    let logical_polys = [&w_a, &w_b];
    let proved = prove_extension_opening_reduction::<F, E, _, RecursiveWitnessFlat, _, 2>(
        &crate::compute::CpuBackend,
        None,
        &logical_polys,
        &opening_batch,
        EorClaimAlignment::Suffix(&layout),
        true,
        &mut transcript,
        "recursive",
    )
    .expect("suffix-aligned grouped reduction");

    // Two claims, `[E:F] = 2` partials each.
    assert_eq!(proved.reduction.proof.partials.len(), 4);
    // Joint tail has `3 - 1 = 2` rounds.
    assert_eq!(proved.reduction.proof.num_rounds(), 2);
    assert_eq!(proved.rho.len(), 2);

    // Each claim's partials must open the claim at ITS OWN routed point:
    // group A at `point[1..]`, group B at the full point.
    let evals_a: Vec<E> = digits_a.iter().copied().map(lift).collect();
    let evals_b: Vec<E> = digits_b.iter().copied().map(lift).collect();
    let expected_a = akita_algebra::poly::multilinear_eval(&evals_a, &point[1..]).expect("claim a");
    let expected_b = akita_algebra::poly::multilinear_eval(&evals_b, &point).expect("claim b");
    let via_partials_a = derive_tensor_extension_opening_claim_from_partials::<F, E>(
        &point[1..],
        &proved.reduction.proof.partials[..2],
    )
    .expect("derive a");
    let via_partials_b = derive_tensor_extension_opening_claim_from_partials::<F, E>(
        &point,
        &proved.reduction.proof.partials[2..],
    )
    .expect("derive b");
    assert_eq!(via_partials_a, expected_a);
    assert_eq!(via_partials_b, expected_b);

    // The final relation consumed by `compute_trace_target`: the batched
    // final claim must equal the RHO-WEIGHTED sum of each group's PACKED
    // polynomial at its own slice of `rho`, scaled by the shared transparent
    // factor. This is the identity the fold-side openings are checked
    // against, so it must hold on the EOR side by itself.
    fn to_base(digit: i8) -> F {
        if digit >= 0 {
            F::from_u64(digit as u64)
        } else {
            -F::from_u64((-(digit as i64)) as u64)
        }
    }
    let base_a: Vec<F> = digits_a.iter().copied().map(to_base).collect();
    let base_b: Vec<F> = digits_b.iter().copied().map(to_base).collect();
    let packed_a = akita_types::tensor_packed_witness_evals::<F, E>(2, &base_a).expect("pack a");
    let packed_b = akita_types::tensor_packed_witness_evals::<F, E>(3, &base_b).expect("pack b");
    // Group A occupies joint-tail vars [pad..), pad = 1; group B the full tail.
    let w_star_a =
        akita_algebra::poly::multilinear_eval(&packed_a, &proved.rho[1..]).expect("w* a");
    let w_star_b = akita_algebra::poly::multilinear_eval(&packed_b, &proved.rho).expect("w* b");
    assert_eq!(proved.row_coefficients.len(), 2);
    assert_eq!(
        proved.reduction.final_claim,
        (proved.row_coefficients[0] * w_star_a + proved.row_coefficients[1] * w_star_b)
            * proved.reduction.final_factor,
        "final claim must equal the rho-weighted sum of per-group packed openings times the \
         shared transparent factor"
    );
}

/// Regression test for **Defect 1** of `aerie/_docs/ABSORPTION-AUDIT.md`.
///
/// The `G = 2` recursive carried-claim suffix used to batch its two claims
/// with unit coefficients. The verifier then enforced only
/// `O_S + O_w = s' + v_w'`, so a prover shipping both carried claims false by
/// a compensating `(+delta, -delta)` was accepted with probability 1 — the
/// forgery was executed end to end at `k = 2` and `k = 1` (see the audit's
/// "Verification of Defect 1"). This pins the three properties the fix rests
/// on, at unit-test cost. The end-to-end forgeries themselves live in
/// `crates/akita-pcs/tests/suffix_carried_claim_forgery_*.rs` (`#[ignore]`d,
/// feature `attack-probe`).
#[test]
fn suffix_carried_claim_batching_coefficients_are_transcript_bound() {
    /// Run the two-claim suffix-aligned reduction over the given shorter-group
    /// witness and return the sampled batching coefficients. Changing
    /// `digits_a` changes the claim value the prover absorbs (and nothing
    /// absorbed before it), so it isolates the absorb-then-squeeze ordering.
    fn coefficients_for(digits_a: Vec<i8>) -> Vec<E> {
        let w_a = RecursiveWitnessFlat::from_i8_digits(digits_a);
        let w_b = RecursiveWitnessFlat::from_i8_digits(vec![2, 5, -1, 4, 0, -3, 7, 1]);
        let point = [
            E::new(F::from_u64(2), F::from_u64(3)),
            E::new(F::from_u64(5), F::from_u64(7)),
            E::new(F::from_u64(11), F::from_u64(13)),
        ];
        let groups = vec![
            PolynomialGroupClaims::new(
                PointVariableSelection::suffix(2, 3).expect("suffix selection"),
                vec![E::zero()],
                (),
            )
            .expect("group a"),
            PolynomialGroupClaims::new(
                PointVariableSelection::suffix(3, 3).expect("full selection"),
                vec![E::zero()],
                (),
            )
            .expect("group b"),
        ];
        let opening_batch = OpeningClaims::from_groups_allow_custom_routing(point.to_vec(), groups)
            .expect("suffix batch");
        let layout = opening_batch.layout().expect("layout");
        let mut transcript = AkitaTranscript::<F>::new(b"test/defect1-rho-binding");
        let logical_polys = [&w_a, &w_b];
        let proved = prove_extension_opening_reduction::<F, E, _, RecursiveWitnessFlat, _, 2>(
            &crate::compute::CpuBackend,
            None,
            &logical_polys,
            &opening_batch,
            EorClaimAlignment::Suffix(&layout),
            true,
            &mut transcript,
            "recursive",
        )
        .expect("suffix-aligned grouped reduction");
        proved.row_coefficients
    }

    // (1) The two claims are NOT combined with unit coefficients. With
    //     `rho_0 == rho_1` the fold enforces a single sum, which is exactly
    //     the freedom the forgery exploited.
    let rho = coefficients_for(vec![1, -2, 3, 0]);
    assert_eq!(rho.len(), 2);
    assert_ne!(
        rho[0], rho[1],
        "the two carried claims must carry DIFFERENT batching coefficients; equal coefficients \
         reinstate Defect 1 (aerie/_docs/ABSORPTION-AUDIT.md)"
    );

    // (2) The coefficients are squeezed AFTER the claimed values are absorbed,
    //     so a prover cannot pick its carried claims once rho is known.
    //     Perturbing a single claim value must move them.
    let rho_perturbed = coefficients_for(vec![1, -2, 3, 1]);
    assert_ne!(
        rho, rho_perturbed,
        "the batching coefficients must depend on the claimed values, i.e. be squeezed after \
         they are absorbed ([AK] Lemma 5.7's hypothesis)"
    );

    // (3) The algebra the fix turns on: a compensating `(+delta, -delta)`
    //     shift survives a unit-coefficient combination identically, and
    //     survives the rho-weighted one only if `rho_0 == rho_1`.
    let delta = E::new(F::from_u64(37), F::from_u64(41));
    assert!(!delta.is_zero());
    assert!(
        (delta + (-delta)).is_zero(),
        "unit-coefficient batching cancels the compensating pair"
    );
    assert!(
        !(rho[0] * delta - rho[1] * delta).is_zero(),
        "rho-weighted batching must NOT cancel the compensating pair"
    );
}

/// Fold-side counterpart of the suffix-aligned reduction: the committed flat
/// setup prefix, opened through the fold machinery at the column-major-routed
/// psi-packed point, must equal the packed polynomial's MLE at `rho`
/// (`specs/eor-setup-prefix-absorption.md`, per-group point construction).
#[test]
fn setup_prefix_fold_opening_matches_packed_mle_at_reduction_point() {
    setup_prefix_fold_opening_harness::<akita_field::Ext2<akita_field::Prime64Offset59>>();
}

/// Degree-1 control for the harness: with `E = F` the same routing must
/// reproduce the k=1-proven identity (fold at the routed raw point equals the
/// natural-order MLE).
#[test]
fn setup_prefix_fold_opening_matches_mle_at_degree_one() {
    setup_prefix_fold_opening_harness::<akita_field::Prime64Offset59>();
}

fn setup_prefix_fold_opening_harness<Ext>()
where
    Ext: akita_types::FpExtEncoding<akita_field::Prime64Offset59>
        + akita_field::ExtField<akita_field::Prime64Offset59>
        + akita_field::MulBaseUnreduced<akita_field::Prime64Offset59>
        + akita_field::LiftBase<akita_field::Prime64Offset59>
        + akita_field::FromPrimitiveInt,
{
    use crate::backend::RecursiveFoldSource;
    use crate::protocol::core::fold_kernels::{
        evaluate_claims_at_prepared_point, scalar_opening_from_folded_ring,
    };
    use akita_challenges::SparseChallengeConfig;
    use akita_field::{Ext2, Prime64Offset59};
    use akita_types::{
        prepare_opening_point, setup_prefix_precommitted_params, setup_prefix_slot_id,
        suffix_aligned_group_packed_point, tensor_packed_witness_evals, AkitaSetupSeed, BasisMode,
        BatchedStage3Geometry, BlockOrder, DigitBlocks, GroupBoundPolicy, LevelParams,
        SetupPrefixPublicCommitment, SetupPrefixSlot, SisModulusFamily,
    };
    use std::sync::Arc;

    type F = Prime64Offset59;
    type E1 = Ext2<F>;
    let _ = core::marker::PhantomData::<E1>;
    const D: usize = 128;
    const M_VARS: usize = 2;
    const R_VARS: usize = 1;
    let ring_bits = D.trailing_zeros() as usize; // 7
    let point_len = ring_bits + M_VARS + R_VARS; // 10
    let n_prefix = 1usize << point_len; // 1024 field coefficients

    // Expanded setup + slot over the flat prefix.
    let ring_slots = n_prefix / D;
    let seed = AkitaSetupSeed {
        max_num_vars: 16,
        max_num_batched_polys: 1,
        gen_ring_dim: D,
        max_setup_len: ring_slots,
        public_matrix_seed: [3u8; 32],
    };
    let shared =
        akita_types::derive_public_matrix_flat::<F, D>(ring_slots, &seed.public_matrix_seed);
    let expanded = Arc::new(
        akita_types::AkitaExpandedSetup::from_trusted_seed_derived_parts_unchecked(seed, shared),
    );
    let lp = LevelParams::params_only(
        SisModulusFamily::Q64,
        D,
        3,
        M_VARS,
        R_VARS,
        2,
        SparseChallengeConfig::pm1_only(3),
    )
    .with_decomp(2, 3, 2, 2, 3)
    .expect("level params");
    let mut commitment_params = setup_prefix_precommitted_params(
        &lp,
        n_prefix,
        GroupBoundPolicy {
            log_commit_bound: 1,
            onehot_chunk_size: 1,
            basis_range: (1, 8),
        },
    )
    .expect("prefix params");
    commitment_params.layout.m_vars = M_VARS;
    commitment_params.layout.r_vars = R_VARS;
    commitment_params.num_blocks = 1 << R_VARS;
    commitment_params.block_len = 1 << M_VARS;
    let id = setup_prefix_slot_id(D, n_prefix, commitment_params);
    let decomposed = DigitBlocks::from_blocks(vec![Vec::new()], D).expect("digit blocks");
    let slot = Arc::new(SetupPrefixSlot {
        id: id.clone(),
        natural_len: n_prefix,
        padded_len: n_prefix,
        commitment: SetupPrefixPublicCommitment {
            rows: vec![akita_types::RingVec::from_coeffs(vec![F::zero(); D])],
        },
        hint: akita_types::AkitaCommitmentHint::singleton(decomposed),
    });
    let committed = crate::backend::setup_prefix_committed_field_evals::<F, Ext, D>(
        expanded.as_ref(),
        slot.as_ref(),
    )
    .expect("committed evals");
    let source = RecursiveFoldSource::setup_prefix(
        Arc::clone(&expanded),
        Arc::clone(&slot),
        std::sync::Arc::new(committed),
    );

    // Column-major routing at offset 0 over its own length (the full-group
    // case of the carried-claim batch). Set `HARNESS_IDENTITY_ROWMAJOR` to
    // probe the natural layout (identity routing + RowMajor) instead.
    const HARNESS_IDENTITY_ROWMAJOR: bool = false;
    let point_vars = if HARNESS_IDENTITY_ROWMAJOR {
        akita_types::PointVariableSelection::prefix(point_len, point_len).expect("identity")
    } else {
        BatchedStage3Geometry::setup_prefix_column_major_point_vars(point_len, &id, 0, point_len)
            .expect("routing")
    };
    let block_order = if HARNESS_IDENTITY_ROWMAJOR {
        BlockOrder::RowMajor
    } else {
        BlockOrder::ColumnMajor
    };

    let kappa = <Ext as akita_field::ExtField<F>>::EXT_DEGREE.trailing_zeros() as usize;
    let rho: Vec<Ext> = (1..=(point_len - kappa) as u64)
        .map(|idx| {
            let limbs: Vec<F> = (0..<Ext as akita_field::ExtField<F>>::EXT_DEGREE)
                .map(|limb| F::from_u64(3 * idx + 1 + 5 * limb as u64))
                .collect();
            Ext::from_base_slice(&limbs)
        })
        .collect();
    assert_eq!(rho.len(), point_len - kappa);

    let group_point =
        suffix_aligned_group_packed_point::<F, Ext, D>(&rho, &point_vars).expect("packed point");
    let prepared = prepare_opening_point::<F, Ext, D>(
        &group_point,
        BasisMode::Lagrange,
        M_VARS,
        R_VARS,
        ring_bits,
        block_order,
    )
    .expect("prepared point");
    let sources: [&RecursiveFoldSource<F>; 1] = [&source];
    let (folded_rings, _) =
        evaluate_claims_at_prepared_point::<F, Ext, RecursiveFoldSource<F>, _, D>(
            &crate::compute::CpuBackend,
            None,
            &sources,
            &prepared,
            1 << M_VARS,
        )
        .expect("fold evaluate");
    let fold_opening = scalar_opening_from_folded_ring::<F, Ext, D>(
        &folded_rings[0],
        &prepared,
        &group_point[..ring_bits],
        BasisMode::Lagrange,
    )
    .expect("fold opening");

    // Packed polynomial of the same flat data at `rho` (natural order).
    let flat = expanded.shared_matrix().as_field_slice()[..n_prefix].to_vec();
    let packed = if kappa == 0 {
        flat.iter()
            .copied()
            .map(<Ext as akita_field::LiftBase<F>>::lift_base)
            .collect::<Vec<Ext>>()
    } else {
        tensor_packed_witness_evals::<F, Ext>(point_len, &flat).expect("packed")
    };
    let expected = akita_algebra::poly::multilinear_eval(&packed, &rho).expect("packed mle");

    assert_eq!(
        fold_opening, expected,
        "fold-side opening at the routed packed point must equal the packed MLE at rho"
    );
}
