//! Mixed per-group commit-bound multi-group root batching probes.
//!
//! One multi-group root mixing groups with different `log_commit_bound`:
//! (a) fp32 ext-4: dense `D128FullFastProver` main group (bound 32,
//!     basis range (3, 3)) plus a one-hot bound-1 precommitted group frozen
//!     under `ConservativeCommitmentConfig<fp32::D128OneHot>` (basis range
//!     (2, 6)); and
//! (b) fp64 ext-2: dense bound-18 main group plus a dense bound-6
//!     precommitted group (test-local presets; the application-side copies
//!     live in the aerie workspace).
//!
//! Both prove under `MixedPrecommitConfig<Main, Pre>`, whose static
//! per-group-index hook freezes every precommitted group under `Pre`'s
//! conservative adapter, so the mixed bound policies enter the schedule key
//! (and transcript instance descriptor) as config-declared statics.

use super::*;
use akita_config::proof_optimized::{fp32, fp64};
use akita_config::MixedPrecommitConfig;
use akita_field::ExtField;
use akita_prover::MultilinearPolynomial;

// ---------------------------------------------------------------------------
// Probe (a): fp32 dense main (bound 32) + one-hot bound-1 precommit.
// ---------------------------------------------------------------------------

type Fp32F = fp32::Field;
type Ext4 = fp32::ExtensionField;

type MainCfg32 = fp32::D128FullFastProver;
type PreCfg32 = fp32::D128OneHot;
type MixedCfg32 = MixedPrecommitConfig<MainCfg32, PreCfg32>;
type MixedScheme32 = AkitaCommitmentScheme<MixedCfg32>;
type ConservativePre32 = ConservativeCommitmentConfig<PreCfg32>;
type ConservativePreScheme32 = AkitaCommitmentScheme<ConservativePre32>;

const MIXED_D32: usize = MainCfg32::D;
const MIXED_ONEHOT_K: usize = 256;

type MixedPoly32 = MultilinearPolynomial<Fp32F, u8>;

/// Random extension opening point compatible with ring-subfield packing:
/// inactive inner coordinates in `[log2(ring_d/ext_degree), log2(ring_d))`
/// are zeroed (see `prepare_opening_point`).
fn subfield_random_point<F, E>(num_vars: usize, ring_d: usize, seed: u64) -> Vec<E>
where
    F: akita_field::FieldCore + akita_field::CanonicalField,
    E: ExtField<F>,
{
    let ext_degree = <E as ExtField<F>>::EXT_DEGREE;
    let trace_inner = (ring_d / ext_degree).trailing_zeros() as usize;
    let alpha_bits = ring_d.trailing_zeros() as usize;
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

fn make_mixed_onehot_poly(layout: &LevelParams, seed: u64) -> OneHotPoly<Fp32F, u8> {
    let total_ring = layout.num_blocks * layout.block_len;
    let total_field = total_ring * MIXED_D32;
    let num_vars = layout.m_vars + layout.r_vars + MIXED_D32.trailing_zeros() as usize;
    assert_eq!(total_field, 1usize << num_vars);
    let total_chunks = total_field / MIXED_ONEHOT_K;

    let mut rng = StdRng::seed_from_u64(seed);
    let indices: Vec<Option<u8>> = (0..total_chunks)
        .map(|_| Some(rng.gen_range(0..MIXED_ONEHOT_K) as u8))
        .collect();
    OneHotPoly::<Fp32F, u8>::new(MIXED_ONEHOT_K, MIXED_D32, indices).expect("mixed onehot poly")
}

/// Extension Lagrange opening of a one-hot polynomial at an extension point.
fn mixed_onehot_opening_ext(poly: &OneHotPoly<Fp32F, u8>, point: &[Ext4]) -> Ext4 {
    let onehot_k = MIXED_ONEHOT_K;
    assert_eq!(poly.indices().len() * onehot_k, 1usize << point.len());
    let low_vars = onehot_k.trailing_zeros() as usize;
    let low_weights = lagrange_weights(&point[..low_vars]).expect("low weights");
    let high_weights = lagrange_weights(&point[low_vars..]).expect("high weights");
    poly.indices()
        .iter()
        .enumerate()
        .filter_map(|(chunk_idx, hot_idx)| {
            hot_idx.map(|hot_idx| high_weights[chunk_idx] * low_weights[hot_idx as usize])
        })
        .fold(Ext4::zero(), |acc, weight| acc + weight)
}

/// Extension Lagrange opening of base-field evaluations at an extension point.
fn dense_opening_ext_generic<F, E>(evals: &[F], point: &[E]) -> E
where
    F: akita_field::FieldCore,
    E: ExtField<F> + akita_field::LiftBase<F> + std::ops::Mul<Output = E>,
{
    let weights = lagrange_weights(point).expect("dense ext weights");
    evals
        .iter()
        .zip(weights.iter())
        .fold(E::zero(), |acc, (&coeff, &weight)| {
            acc + weight * E::lift_base(coeff)
        })
}

fn make_seeded_dense_poly_bounded<F>(
    num_vars: usize,
    ring_d: usize,
    max_exclusive: u64,
    seed: u64,
) -> (DensePoly<F>, Vec<F>)
where
    F: akita_field::FieldCore + akita_field::CanonicalField,
{
    let len = 1usize << num_vars;
    let mut rng = StdRng::seed_from_u64(seed);
    let evals: Vec<F> = (0..len)
        .map(|_| F::from_u64(rng.gen_range(0..max_exclusive)))
        .collect();
    let poly = DensePoly::<F>::from_field_evals(num_vars, ring_d, &evals).expect("dense poly");
    (poly, evals)
}

/// Mixed fp32 root at ext-4: dense bound-32 main group plus one one-hot
/// bound-1 precommitted group; commit -> prove -> verify -> tamper-reject.
#[test]
fn multi_group_root_mixed_dense_main_onehot_precommit_fp32_round_trips() {
    // FINAL_NV must sit above the dense fold boundary: the planner finds no
    // fold candidate for this shape below 16 vars at 2 polys (the boundary
    // scan in this round measured nv=14/2 and nv=16/6 emitting Direct roots,
    // which grouped roots reject) and production dense shapes are 18-23 vars.
    const PRE_NV: usize = 10;
    const FINAL_NV: usize = 16;
    const FINAL_SIZE: usize = 2;
    let total = 1 + FINAL_SIZE;

    let setup = MixedScheme32::setup_prover(FINAL_NV, total).expect("mixed setup");
    let prepared = CpuBackend.prepare_setup(&setup).expect("prepared setup");
    let stack =
        akita_prover::UniformProverStack::uniform(&CpuBackend, &prepared, setup.expanded.as_ref())
            .expect("stack");

    let point = subfield_random_point::<Fp32F, Ext4>(FINAL_NV, MIXED_D32, 0x001f_32e4_0a01);

    // One-hot bound-1 precommit under the Pre preset's conservative adapter.
    let pre_key = akita_types::PolynomialGroupLayout::new(PRE_NV, 1);
    let pre_batch = OpeningClaimsLayout::new(PRE_NV, 1).expect("precommit batch");
    let pre_layout =
        ConservativePre32::get_params_for_batched_commitment(&pre_batch).expect("precommit layout");
    let pre_polys: Vec<MixedPoly32> = vec![MultilinearPolynomial::onehot(make_mixed_onehot_poly(
        &pre_layout,
        0x001f_32e4_0a02,
    ))];
    let pre_openings: Vec<Ext4> = pre_polys
        .iter()
        .map(|poly| match poly {
            MultilinearPolynomial::OneHot(poly) => mixed_onehot_opening_ext(poly, &point[..PRE_NV]),
            MultilinearPolynomial::Dense(_) => unreachable!("precommit group is one-hot"),
        })
        .collect();
    let (pre_commitment, pre_hint) =
        ConservativePreScheme32::batched_commit(&setup, &pre_polys[..], &stack)
            .expect("mixed one-hot conservative precommit");

    // Dense bound-32 final group committed under the mixed proving config.
    let mut final_polys: Vec<MixedPoly32> = Vec::new();
    let mut final_openings: Vec<Ext4> = Vec::new();
    for poly_idx in 0..FINAL_SIZE {
        let len = 1usize << FINAL_NV;
        let mut rng = StdRng::seed_from_u64(0x001f_32e4_0f00 + poly_idx as u64);
        let evals: Vec<Fp32F> = (0..len)
            .map(|_| Fp32F::from_canonical_u128_reduced(rng.r#gen::<u128>()))
            .collect();
        let poly =
            DensePoly::<Fp32F>::from_field_evals(FINAL_NV, MIXED_D32, &evals).expect("dense poly");
        final_openings.push(dense_opening_ext_generic(&evals, &point));
        final_polys.push(MultilinearPolynomial::dense(poly));
    }
    let (final_commitment, final_hint) =
        MixedScheme32::commit_final_group(&setup, &final_polys, &stack, vec![pre_key])
            .expect("mixed final multi-group commitment");

    let pre_refs: Vec<&MixedPoly32> = pre_polys.iter().collect();
    let final_refs: Vec<&MixedPoly32> = final_polys.iter().collect();

    let prover_groups = vec![
        PolynomialGroupClaims::new(
            PointVariableSelection::prefix(PRE_NV, FINAL_NV).expect("pre point vars"),
            pre_openings.clone(),
            pre_commitment.clone(),
        )
        .expect("pre prover group"),
        PolynomialGroupClaims::new(
            PointVariableSelection::prefix(FINAL_NV, FINAL_NV).expect("final point vars"),
            final_openings.clone(),
            final_commitment.clone(),
        )
        .expect("final prover group"),
    ];
    let prover_polys: Vec<&[&MixedPoly32]> = vec![&pre_refs[..], &final_refs[..]];
    let prover_hints = vec![pre_hint, final_hint];

    let prover_claims = ProverOpeningData::new(
        OpeningClaims::from_groups(point.clone(), prover_groups).expect("prover claims"),
        prover_hints,
        prover_polys,
    )
    .expect("mixed fp32 multi-group prover data");

    let mut prover_transcript = AkitaTranscript::<Fp32F>::new(b"test/multi-group-mixed-fp32");
    let proof = MixedScheme32::batched_prove(
        &setup,
        prover_claims,
        &stack,
        &mut prover_transcript,
        BasisMode::Lagrange,
        akita_types::SetupContributionMode::Direct,
    )
    .expect("mixed fp32 multi-group prove");
    assert!(!matches!(
        proof.root,
        akita_types::AkitaBatchedRootProof::ZeroFold { .. }
    ));

    let shape = proof.shape();
    let mut bytes = Vec::new();
    proof
        .serialize_uncompressed(&mut bytes)
        .expect("serialize mixed fp32 multi-group proof");
    let decoded =
        akita_types::AkitaBatchedProof::<Fp32F, Ext4>::deserialize_uncompressed(&bytes[..], &shape)
            .expect("deserialize mixed fp32 multi-group proof");
    assert_eq!(decoded, proof);

    let verifier_setup = MixedScheme32::setup_verifier(&setup);
    let build_verifier_claims = |final_openings: &[Ext4]| {
        let verifier_groups = vec![
            PolynomialGroupClaims::new(
                PointVariableSelection::prefix(PRE_NV, FINAL_NV).expect("pre point vars"),
                pre_openings.clone(),
                &pre_commitment,
            )
            .expect("pre verifier group"),
            PolynomialGroupClaims::new(
                PointVariableSelection::prefix(FINAL_NV, FINAL_NV).expect("final point vars"),
                final_openings.to_vec(),
                &final_commitment,
            )
            .expect("final verifier group"),
        ];
        OpeningClaims::from_groups(point.clone(), verifier_groups).expect("verifier claims")
    };

    let mut verifier_transcript = AkitaTranscript::<Fp32F>::new(b"test/multi-group-mixed-fp32");
    MixedScheme32::batched_verify(
        &decoded,
        &verifier_setup,
        &mut verifier_transcript,
        build_verifier_claims(&final_openings),
        BasisMode::Lagrange,
        akita_types::SetupContributionMode::Direct,
    )
    .expect("mixed fp32 multi-group verify");

    // Negative: tampering the final group's first opening must reject.
    let mut tampered_final = final_openings.clone();
    tampered_final[0] += Ext4::one();
    let mut tampered_transcript = AkitaTranscript::<Fp32F>::new(b"test/multi-group-mixed-fp32");
    assert!(
        MixedScheme32::batched_verify(
            &decoded,
            &verifier_setup,
            &mut tampered_transcript,
            build_verifier_claims(&tampered_final),
            BasisMode::Lagrange,
            akita_types::SetupContributionMode::Direct,
        )
        .is_err(),
        "tampered mixed fp32 final opening must reject"
    );
}

// ---------------------------------------------------------------------------
// Probe (b): fp64 dense bound-18 main + dense bound-6 precommit.
// ---------------------------------------------------------------------------

type Fp64F = fp64::Field;
type Ext2F = fp64::ExtensionField;

/// Test-local bound presets (the aerie application hosts its own copies).
#[derive(Clone, Copy, Debug, Default)]
struct TestBound18;
#[derive(Clone, Copy, Debug, Default)]
struct TestBound6;
akita_config::impl_proof_optimized_preset!(
    TestBound18,
    fp64::Field,
    fp64::ExtensionField,
    akita_types::SisModulusFamily::Q64,
    128,
    64,
    18,
    basis_range = (3, 3)
);
akita_config::impl_proof_optimized_preset!(
    TestBound6,
    fp64::Field,
    fp64::ExtensionField,
    akita_types::SisModulusFamily::Q64,
    128,
    64,
    6,
    basis_range = (3, 3)
);

type MainCfg64 = TestBound18;
type PreCfg64 = TestBound6;
type MixedCfg64 = MixedPrecommitConfig<MainCfg64, PreCfg64>;
type MixedScheme64 = AkitaCommitmentScheme<MixedCfg64>;
type ConservativePre64 = ConservativeCommitmentConfig<PreCfg64>;
type ConservativePreScheme64 = AkitaCommitmentScheme<ConservativePre64>;

const MIXED_D64: usize = MainCfg64::D;

/// Mixed fp64 dense root: bound-18 main group plus a bound-6 precommitted
/// group; both bounds are application-guaranteed source ranges, so the test
/// data respects them (bound-18: `[0, 2^17)`, bound-6: `[0, 2^5)`).
#[test]
fn multi_group_root_mixed_dense_bound18_bound6_fp64_round_trips() {
    const PRE_NV: usize = 8;
    const FINAL_NV: usize = 14;
    const FINAL_SIZE: usize = 2;
    let total = 1 + FINAL_SIZE;

    let setup = MixedScheme64::setup_prover(FINAL_NV, total).expect("mixed fp64 setup");
    let prepared = CpuBackend.prepare_setup(&setup).expect("prepared setup");
    let stack =
        akita_prover::UniformProverStack::uniform(&CpuBackend, &prepared, setup.expanded.as_ref())
            .expect("stack");

    let point = subfield_random_point::<Fp64F, Ext2F>(FINAL_NV, MIXED_D64, 0x001f_64e2_0b01);

    // Bound-6 precommit: signed magnitude within 6 bits.
    let pre_key = akita_types::PolynomialGroupLayout::new(PRE_NV, 1);
    let (pre_poly, pre_evals) =
        make_seeded_dense_poly_bounded::<Fp64F>(PRE_NV, MIXED_D64, 1u64 << 5, 0x001f_64e2_0b02);
    let pre_polys = vec![pre_poly];
    let pre_openings = vec![dense_opening_ext_generic(&pre_evals, &point[..PRE_NV])];
    let (pre_commitment, pre_hint) =
        ConservativePreScheme64::batched_commit(&setup, &pre_polys[..], &stack)
            .expect("mixed fp64 bound-6 conservative precommit");

    // Bound-18 final group: values within 2^17.
    let mut final_polys = Vec::new();
    let mut final_openings = Vec::new();
    for poly_idx in 0..FINAL_SIZE {
        let (poly, evals) = make_seeded_dense_poly_bounded::<Fp64F>(
            FINAL_NV,
            MIXED_D64,
            1u64 << 17,
            0x001f_64e2_0f00 + poly_idx as u64,
        );
        final_openings.push(dense_opening_ext_generic(&evals, &point));
        final_polys.push(poly);
    }
    let (final_commitment, final_hint) =
        MixedScheme64::commit_final_group(&setup, &final_polys, &stack, vec![pre_key])
            .expect("mixed fp64 final multi-group commitment");

    let pre_refs: Vec<&DensePoly<Fp64F>> = pre_polys.iter().collect();
    let final_refs: Vec<&DensePoly<Fp64F>> = final_polys.iter().collect();

    let prover_groups = vec![
        PolynomialGroupClaims::new(
            PointVariableSelection::prefix(PRE_NV, FINAL_NV).expect("pre point vars"),
            pre_openings.clone(),
            pre_commitment.clone(),
        )
        .expect("pre prover group"),
        PolynomialGroupClaims::new(
            PointVariableSelection::prefix(FINAL_NV, FINAL_NV).expect("final point vars"),
            final_openings.clone(),
            final_commitment.clone(),
        )
        .expect("final prover group"),
    ];
    let prover_polys: Vec<&[&DensePoly<Fp64F>]> = vec![&pre_refs[..], &final_refs[..]];
    let prover_hints = vec![pre_hint, final_hint];

    let prover_claims = ProverOpeningData::new(
        OpeningClaims::from_groups(point.clone(), prover_groups).expect("prover claims"),
        prover_hints,
        prover_polys,
    )
    .expect("mixed fp64 multi-group prover data");

    let mut prover_transcript = AkitaTranscript::<Fp64F>::new(b"test/multi-group-mixed-fp64");
    let proof = MixedScheme64::batched_prove(
        &setup,
        prover_claims,
        &stack,
        &mut prover_transcript,
        BasisMode::Lagrange,
        akita_types::SetupContributionMode::Direct,
    )
    .expect("mixed fp64 multi-group prove");
    assert!(!matches!(
        proof.root,
        akita_types::AkitaBatchedRootProof::ZeroFold { .. }
    ));

    let shape = proof.shape();
    let mut bytes = Vec::new();
    proof
        .serialize_uncompressed(&mut bytes)
        .expect("serialize mixed fp64 multi-group proof");
    let decoded = akita_types::AkitaBatchedProof::<Fp64F, Ext2F>::deserialize_uncompressed(
        &bytes[..],
        &shape,
    )
    .expect("deserialize mixed fp64 multi-group proof");
    assert_eq!(decoded, proof);

    let verifier_setup = MixedScheme64::setup_verifier(&setup);
    let build_verifier_claims = |final_openings: &[Ext2F]| {
        let verifier_groups = vec![
            PolynomialGroupClaims::new(
                PointVariableSelection::prefix(PRE_NV, FINAL_NV).expect("pre point vars"),
                pre_openings.clone(),
                &pre_commitment,
            )
            .expect("pre verifier group"),
            PolynomialGroupClaims::new(
                PointVariableSelection::prefix(FINAL_NV, FINAL_NV).expect("final point vars"),
                final_openings.to_vec(),
                &final_commitment,
            )
            .expect("final verifier group"),
        ];
        OpeningClaims::from_groups(point.clone(), verifier_groups).expect("verifier claims")
    };

    let mut verifier_transcript = AkitaTranscript::<Fp64F>::new(b"test/multi-group-mixed-fp64");
    MixedScheme64::batched_verify(
        &decoded,
        &verifier_setup,
        &mut verifier_transcript,
        build_verifier_claims(&final_openings),
        BasisMode::Lagrange,
        akita_types::SetupContributionMode::Direct,
    )
    .expect("mixed fp64 multi-group verify");

    // Negative: tampering the precommitted group's opening must reject.
    let mut tampered_pre = pre_openings.clone();
    tampered_pre[0] += Ext2F::one();
    let tampered_groups = vec![
        PolynomialGroupClaims::new(
            PointVariableSelection::prefix(PRE_NV, FINAL_NV).expect("pre point vars"),
            tampered_pre,
            &pre_commitment,
        )
        .expect("pre verifier group"),
        PolynomialGroupClaims::new(
            PointVariableSelection::prefix(FINAL_NV, FINAL_NV).expect("final point vars"),
            final_openings.clone(),
            &final_commitment,
        )
        .expect("final verifier group"),
    ];
    let tampered_claims = OpeningClaims::from_groups(point.clone(), tampered_groups)
        .expect("tampered verifier claims");
    let mut tampered_transcript = AkitaTranscript::<Fp64F>::new(b"test/multi-group-mixed-fp64");
    assert!(
        MixedScheme64::batched_verify(
            &decoded,
            &verifier_setup,
            &mut tampered_transcript,
            tampered_claims,
            BasisMode::Lagrange,
            akita_types::SetupContributionMode::Direct,
        )
        .is_err(),
        "tampered mixed fp64 precommitted opening must reject"
    );
}
