//! fp32 (degree-4 extension) multi-group root batching end-to-end probes.
//!
//! Mirrors the ext-1 multi-group round trips in `onehot.rs` and
//! `dense_multi_group.rs`, but under the fp32 `D=128` presets whose proof
//! scalar is `FpExt4`: grouped roots go through the tensor-projection
//! extension-opening reduction, with shorter precommitted groups lifted into
//! the joint reduction domain by constant extension.

use super::*;
use akita_config::proof_optimized::fp32;
use akita_field::ExtField;

type Fp32 = fp32::Field;
type Ext4 = fp32::ExtensionField;

type OneHotCfg32 = fp32::D128OneHot;
type ConservativeOneHotCfg32 = ConservativeCommitmentConfig<OneHotCfg32>;
type OneHotScheme32 = AkitaCommitmentScheme<OneHotCfg32>;
type ConservativeOneHotScheme32 = AkitaCommitmentScheme<ConservativeOneHotCfg32>;

type DenseCfg32 = fp32::D128FullFastProver;
type ConservativeDenseCfg32 = ConservativeCommitmentConfig<DenseCfg32>;
type DenseScheme32 = AkitaCommitmentScheme<DenseCfg32>;
type ConservativeDenseScheme32 = AkitaCommitmentScheme<ConservativeDenseCfg32>;

const D32: usize = OneHotCfg32::D;
const FP32_ONEHOT_K: usize = 256;

/// Random `FpExt4` opening point compatible with fp32 ring-subfield packing at
/// `ring_d`: inactive inner coordinates in `[log2(ring_d/4), log2(ring_d))`
/// are zeroed (see `fp32_ext4.rs` and `akita_types::prepare_opening_point`).
fn fp32_ext4_random_point(num_vars: usize, ring_d: usize, seed: u64) -> Vec<Ext4> {
    let ext_degree = <Ext4 as ExtField<Fp32>>::EXT_DEGREE;
    let trace_inner = (ring_d / ext_degree).trailing_zeros() as usize;
    let alpha_bits = ring_d.trailing_zeros() as usize;
    let mut rng = StdRng::seed_from_u64(seed);
    let mut point: Vec<Ext4> = (0..num_vars)
        .map(|_| {
            let limbs: [Fp32; 4] = [
                Fp32::from_canonical_u128_reduced(rng.r#gen::<u128>()),
                Fp32::from_canonical_u128_reduced(rng.r#gen::<u128>()),
                Fp32::from_canonical_u128_reduced(rng.r#gen::<u128>()),
                Fp32::from_canonical_u128_reduced(rng.r#gen::<u128>()),
            ];
            Ext4::from_base_slice(&limbs)
        })
        .collect();
    for idx in trace_inner..alpha_bits {
        if idx < point.len() {
            point[idx] = Ext4::zero();
        }
    }
    point
}

fn make_fp32_onehot_poly(layout: &LevelParams, seed: u64) -> OneHotPoly<Fp32, u8> {
    let total_ring = layout.num_blocks * layout.block_len;
    let total_field = total_ring * D32;
    let num_vars = layout.m_vars + layout.r_vars + D32.trailing_zeros() as usize;
    assert_eq!(total_field, 1usize << num_vars);
    let total_chunks = total_field / FP32_ONEHOT_K;

    let mut rng = StdRng::seed_from_u64(seed);
    let indices: Vec<Option<u8>> = (0..total_chunks)
        .map(|_| Some(rng.gen_range(0..FP32_ONEHOT_K) as u8))
        .collect();
    OneHotPoly::<Fp32, u8>::new(FP32_ONEHOT_K, D32, indices).expect("fp32 onehot poly")
}

/// `FpExt4` Lagrange opening of a one-hot polynomial at an extension point.
fn onehot_opening_ext(poly: &OneHotPoly<Fp32, u8>, point: &[Ext4]) -> Ext4 {
    let onehot_k = FP32_ONEHOT_K;
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

fn make_seeded_fp32_dense_poly(num_vars: usize, seed: u64) -> (DensePoly<Fp32>, Vec<Fp32>) {
    let len = 1usize << num_vars;
    let mut rng = StdRng::seed_from_u64(seed);
    let evals: Vec<Fp32> = (0..len)
        .map(|_| Fp32::from_canonical_u128_reduced(rng.r#gen::<u128>()))
        .collect();
    let poly = DensePoly::<Fp32>::from_field_evals(num_vars, D32, &evals).expect("dense poly");
    (poly, evals)
}

/// `FpExt4` Lagrange opening of base-field evaluations at an extension point.
fn dense_opening_ext(evals: &[Fp32], point: &[Ext4]) -> Ext4 {
    let weights = lagrange_weights(point).expect("dense ext weights");
    evals
        .iter()
        .zip(weights.iter())
        .fold(Ext4::zero(), |acc, (&coeff, &weight)| {
            acc + weight * Ext4::lift_base(coeff)
        })
}

/// Produce and verify a folded multi-group-root fp32/`FpExt4` one-hot
/// same-point proof: precommitted groups under the conservative config, the
/// final group with `commit_final_group`, grouped root through the
/// extension-opening reduction, singleton recursive suffix.
fn multi_group_root_round_trip_onehot_fp32_ext4(pre_sizes: &[usize], final_size: usize) {
    const PRE_NV: usize = 10;
    const FINAL_NV: usize = 16;
    let total: usize = pre_sizes.iter().sum::<usize>() + final_size;

    let setup = ConservativeOneHotScheme32::setup_prover(FINAL_NV, total).expect("setup");
    let prepared = CpuBackend.prepare_setup(&setup).expect("prepared setup");
    let stack =
        akita_prover::UniformProverStack::uniform(&CpuBackend, &prepared, setup.expanded.as_ref())
            .expect("stack");

    let point = fp32_ext4_random_point(FINAL_NV, D32, 0xf32e_4001);

    let mut pre_keys = Vec::new();
    let mut pre_commitments = Vec::new();
    let mut pre_hints = Vec::new();
    let mut pre_polys_by_group: Vec<Vec<OneHotPoly<Fp32, u8>>> = Vec::new();
    let mut pre_openings: Vec<Vec<Ext4>> = Vec::new();
    for (group_idx, &k) in pre_sizes.iter().enumerate() {
        let key = akita_types::PolynomialGroupLayout::new(PRE_NV, k);
        let opening_batch = OpeningClaimsLayout::new(PRE_NV, k).expect("precommit batch");
        let layout = ConservativeOneHotCfg32::get_params_for_batched_commitment(&opening_batch)
            .expect("precommit layout");
        let polys: Vec<OneHotPoly<Fp32, u8>> = (0..k)
            .map(|poly_idx| {
                make_fp32_onehot_poly(
                    &layout,
                    0xf32e_4a00_0000_0000 + ((group_idx as u64) << 8) + poly_idx as u64,
                )
            })
            .collect();
        let openings = polys
            .iter()
            .map(|poly| onehot_opening_ext(poly, &point[..PRE_NV]))
            .collect::<Vec<_>>();
        let (commitment, hint) =
            ConservativeOneHotScheme32::batched_commit(&setup, &polys[..], &stack)
                .expect("fp32 conservative precommit");
        pre_keys.push(key);
        pre_commitments.push(commitment);
        pre_hints.push(hint);
        pre_polys_by_group.push(polys);
        pre_openings.push(openings);
    }

    let multi_group_key = akita_types::AkitaScheduleLookupKey {
        final_group: akita_types::PolynomialGroupLayout::new(FINAL_NV, final_size),
        precommitteds: pre_keys
            .iter()
            .map(|key| {
                let batch = OpeningClaimsLayout::new(key.num_vars(), key.num_polynomials())
                    .expect("frozen batch");
                let layout = ConservativeOneHotCfg32::get_params_for_batched_commitment(&batch)
                    .expect("frozen layout");
                akita_types::PrecommittedGroupParams::from_params(
                    *key,
                    &layout,
                    akita_config::group_bound_policy_of::<ConservativeOneHotCfg32>(),
                )
            })
            .collect(),
    };
    let multi_group_schedule =
        OneHotCfg32::runtime_schedule(multi_group_key).expect("multi-group runtime schedule");
    let main_params = akita_types::multi_group_root_commit_params(&multi_group_schedule)
        .expect("multi-group root params");
    let final_polys: Vec<OneHotPoly<Fp32, u8>> = (0..final_size)
        .map(|poly_idx| {
            make_fp32_onehot_poly(&main_params, 0xf32e_4f00_0000_0000 + poly_idx as u64)
        })
        .collect();
    let final_openings: Vec<Ext4> = final_polys
        .iter()
        .map(|poly| onehot_opening_ext(poly, &point))
        .collect();
    let (final_commitment, final_hint) =
        OneHotScheme32::commit_final_group(&setup, &final_polys, &stack, pre_keys)
            .expect("fp32 final multi-group commitment");

    let pre_refs_by_group: Vec<Vec<&OneHotPoly<Fp32, u8>>> = pre_polys_by_group
        .iter()
        .map(|polys| polys.iter().collect())
        .collect();
    let final_refs: Vec<&OneHotPoly<Fp32, u8>> = final_polys.iter().collect();

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

    let mut prover_polys: Vec<&[&OneHotPoly<Fp32, u8>]> = Vec::new();
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
    .expect("fp32 multi-group prover data");

    let mut prover_transcript = AkitaTranscript::<Fp32>::new(b"test/multi-group-fp32-onehot");
    let proof = OneHotScheme32::batched_prove(
        &setup,
        prover_claims,
        &stack,
        &mut prover_transcript,
        BasisMode::Lagrange,
        akita_types::SetupContributionMode::Direct,
    )
    .expect("fp32 multi-group prove");
    assert!(!matches!(
        proof.root,
        akita_types::AkitaBatchedRootProof::ZeroFold { .. }
    ));
    // Grouped extension roots must prove the extension-opening reduction.
    let root_reduction = match &proof.root {
        akita_types::AkitaBatchedRootProof::Fold(fold) => fold.extension_opening_reduction.as_ref(),
        akita_types::AkitaBatchedRootProof::Terminal(terminal) => {
            terminal.extension_opening_reduction.as_ref()
        }
        akita_types::AkitaBatchedRootProof::ZeroFold { .. } => None,
    };
    assert!(
        root_reduction.is_some(),
        "fp32 multi-group root must carry an extension-opening reduction"
    );

    let shape = proof.shape();
    let mut bytes = Vec::new();
    proof
        .serialize_uncompressed(&mut bytes)
        .expect("serialize fp32 multi-group proof");
    let decoded =
        akita_types::AkitaBatchedProof::<Fp32, Ext4>::deserialize_uncompressed(&bytes[..], &shape)
            .expect("deserialize fp32 multi-group proof");
    assert_eq!(decoded, proof);

    let verifier_setup = OneHotScheme32::setup_verifier(&setup);
    let build_verifier_claims = |final_openings: &[Ext4]| {
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
                final_openings.to_vec(),
                &final_commitment,
            )
            .expect("final verifier group"),
        );
        OpeningClaims::from_groups(point.clone(), verifier_groups).expect("verifier claims")
    };

    let mut verifier_transcript = AkitaTranscript::<Fp32>::new(b"test/multi-group-fp32-onehot");
    OneHotScheme32::batched_verify(
        &decoded,
        &verifier_setup,
        &mut verifier_transcript,
        build_verifier_claims(&final_openings),
        BasisMode::Lagrange,
        akita_types::SetupContributionMode::Direct,
    )
    .expect("fp32 multi-group verify");

    // Negative: tampering the final group's first opening must reject.
    let mut tampered_final = final_openings.clone();
    tampered_final[0] += Ext4::one();
    let mut tampered_transcript = AkitaTranscript::<Fp32>::new(b"test/multi-group-fp32-onehot");
    assert!(
        OneHotScheme32::batched_verify(
            &decoded,
            &verifier_setup,
            &mut tampered_transcript,
            build_verifier_claims(&tampered_final),
            BasisMode::Lagrange,
            akita_types::SetupContributionMode::Direct,
        )
        .is_err(),
        "tampered fp32 one-hot group opening must reject"
    );
}

#[test]
fn multi_group_root_folded_two_group_onehot_fp32_ext4_round_trips() {
    multi_group_root_round_trip_onehot_fp32_ext4(&[1], 1);
}

/// Dense analogue of the fp32/`FpExt4` grouped-root probe under the
/// `D128FullFastProver` preset.
fn multi_group_root_round_trip_dense_fp32_ext4(pre_sizes: &[usize], final_size: usize) {
    const PRE_NV: usize = 8;
    const FINAL_NV: usize = 16;
    let total: usize = pre_sizes.iter().sum::<usize>() + final_size;

    let setup = DenseScheme32::setup_prover(FINAL_NV, total).expect("setup");
    let prepared = CpuBackend.prepare_setup(&setup).expect("prepared setup");
    let stack =
        akita_prover::UniformProverStack::uniform(&CpuBackend, &prepared, setup.expanded.as_ref())
            .expect("stack");

    let point = fp32_ext4_random_point(FINAL_NV, D32, 0xf32e_4d02);

    let mut pre_keys = Vec::new();
    let mut pre_commitments = Vec::new();
    let mut pre_hints = Vec::new();
    let mut pre_polys_by_group: Vec<Vec<DensePoly<Fp32>>> = Vec::new();
    let mut pre_openings: Vec<Vec<Ext4>> = Vec::new();
    for (group_idx, &k) in pre_sizes.iter().enumerate() {
        let key = akita_types::PolynomialGroupLayout::new(PRE_NV, k);
        let mut polys = Vec::new();
        let mut openings = Vec::new();
        for poly_idx in 0..k {
            let (poly, evals) = make_seeded_fp32_dense_poly(
                PRE_NV,
                0xf32e_4de0_0000_0000 + ((group_idx as u64) << 8) + poly_idx as u64,
            );
            openings.push(dense_opening_ext(&evals, &point[..PRE_NV]));
            polys.push(poly);
        }
        let (commitment, hint) =
            ConservativeDenseScheme32::batched_commit(&setup, &polys[..], &stack)
                .expect("fp32 dense conservative precommit");
        pre_keys.push(key);
        pre_commitments.push(commitment);
        pre_hints.push(hint);
        pre_polys_by_group.push(polys);
        pre_openings.push(openings);
    }

    let mut final_polys = Vec::new();
    let mut final_openings = Vec::new();
    for poly_idx in 0..final_size {
        let (poly, evals) =
            make_seeded_fp32_dense_poly(FINAL_NV, 0xf32e_4df0_0000_0000 + poly_idx as u64);
        final_openings.push(dense_opening_ext(&evals, &point));
        final_polys.push(poly);
    }
    let (final_commitment, final_hint) =
        DenseScheme32::commit_final_group(&setup, &final_polys, &stack, pre_keys)
            .expect("fp32 dense final multi-group commitment");

    let pre_refs_by_group: Vec<Vec<&DensePoly<Fp32>>> = pre_polys_by_group
        .iter()
        .map(|polys| polys.iter().collect())
        .collect();
    let final_refs: Vec<&DensePoly<Fp32>> = final_polys.iter().collect();

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

    let mut prover_polys: Vec<&[&DensePoly<Fp32>]> = Vec::new();
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
    .expect("fp32 dense multi-group prover data");

    let mut prover_transcript = AkitaTranscript::<Fp32>::new(b"test/multi-group-fp32-dense");
    let proof = DenseScheme32::batched_prove(
        &setup,
        prover_claims,
        &stack,
        &mut prover_transcript,
        BasisMode::Lagrange,
        akita_types::SetupContributionMode::Direct,
    )
    .expect("fp32 dense multi-group prove");
    assert!(!matches!(
        proof.root,
        akita_types::AkitaBatchedRootProof::ZeroFold { .. }
    ));

    let shape = proof.shape();
    let mut bytes = Vec::new();
    proof
        .serialize_uncompressed(&mut bytes)
        .expect("serialize fp32 dense multi-group proof");
    let decoded =
        akita_types::AkitaBatchedProof::<Fp32, Ext4>::deserialize_uncompressed(&bytes[..], &shape)
            .expect("deserialize fp32 dense multi-group proof");
    assert_eq!(decoded, proof);

    let verifier_setup = DenseScheme32::setup_verifier(&setup);
    let build_verifier_claims = |final_openings: &[Ext4]| {
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
                final_openings.to_vec(),
                &final_commitment,
            )
            .expect("final verifier group"),
        );
        OpeningClaims::from_groups(point.clone(), verifier_groups).expect("verifier claims")
    };

    let mut verifier_transcript = AkitaTranscript::<Fp32>::new(b"test/multi-group-fp32-dense");
    DenseScheme32::batched_verify(
        &decoded,
        &verifier_setup,
        &mut verifier_transcript,
        build_verifier_claims(&final_openings),
        BasisMode::Lagrange,
        akita_types::SetupContributionMode::Direct,
    )
    .expect("fp32 dense multi-group verify");

    // Negative: tampering the final group's first opening must reject.
    let mut tampered_final = final_openings.clone();
    tampered_final[0] += Ext4::one();
    let mut tampered_transcript = AkitaTranscript::<Fp32>::new(b"test/multi-group-fp32-dense");
    assert!(
        DenseScheme32::batched_verify(
            &decoded,
            &verifier_setup,
            &mut tampered_transcript,
            build_verifier_claims(&tampered_final),
            BasisMode::Lagrange,
            akita_types::SetupContributionMode::Direct,
        )
        .is_err(),
        "tampered fp32 dense group opening must reject"
    );
}

#[test]
fn multi_group_root_folded_dense_fp32_ext4_two_group_round_trips() {
    multi_group_root_round_trip_dense_fp32_ext4(&[1], 1);
}
