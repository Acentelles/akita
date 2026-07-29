//! Dense multi-group root batching end-to-end probes.
//!
//! Mirrors the one-hot multi-group round trips in `onehot.rs`, but under the
//! dense full-field preset (`log_commit_bound > 1`): every group commits a
//! dense polynomial, precommitted groups use the conservative-rank adapter,
//! and the final group is committed with `commit_final_group`.

use super::*;

type DenseCfg = Cfg; // fp128::D64Full, log_commit_bound = 128 (dense).
type ConservativeDenseCfg = ConservativeCommitmentConfig<DenseCfg>;
type DenseScheme = AkitaCommitmentScheme<DenseCfg>;
type ConservativeDenseScheme = AkitaCommitmentScheme<ConservativeDenseCfg>;

fn make_seeded_dense_poly(num_vars: usize, seed: u64) -> (DensePoly<F>, Vec<F>) {
    let len = 1usize << num_vars;
    let mut rng = StdRng::seed_from_u64(seed);
    let evals: Vec<F> = (0..len)
        .map(|_| F::from_canonical_u128_reduced(rng.r#gen::<u128>()))
        .collect();
    let poly = DensePoly::<F>::from_field_evals(num_vars, D, &evals).expect("dense poly");
    (poly, evals)
}

/// Produce and verify a folded multi-group-root dense same-point proof for the
/// given precommitted group sizes plus a final group size. Precommitted groups
/// are committed under the conservative config; the final group is committed
/// with `commit_final_group`; the multi-group root folds into a singleton
/// recursive suffix.
fn multi_group_root_round_trip_dense(pre_sizes: &[usize], final_size: usize) {
    const PRE_NV: usize = 8;
    const FINAL_NV: usize = PRE_NV * 2;
    let total: usize = pre_sizes.iter().sum::<usize>() + final_size;

    let setup = DenseScheme::setup_prover(FINAL_NV, total).expect("setup");
    let prepared = CpuBackend.prepare_setup(&setup).expect("prepared setup");
    let stack =
        akita_prover::UniformProverStack::uniform(&CpuBackend, &prepared, setup.expanded.as_ref())
            .expect("stack");

    // Commit every precommitted group under the conservative config.
    let mut pre_keys = Vec::new();
    let mut pre_commitments = Vec::new();
    let mut pre_hints = Vec::new();
    let mut pre_polys_by_group: Vec<Vec<DensePoly<F>>> = Vec::new();
    let mut pre_evals_by_group: Vec<Vec<Vec<F>>> = Vec::new();
    for (group_idx, &k) in pre_sizes.iter().enumerate() {
        let key = akita_types::PolynomialGroupLayout::new(PRE_NV, k);
        let mut polys = Vec::new();
        let mut evals = Vec::new();
        for poly_idx in 0..k {
            let (poly, poly_evals) = make_seeded_dense_poly(
                PRE_NV,
                0xdeb5_e000_0000_0000 + ((group_idx as u64) << 8) + poly_idx as u64,
            );
            polys.push(poly);
            evals.push(poly_evals);
        }
        let (commitment, hint) =
            ConservativeDenseScheme::batched_commit(&setup, &polys[..], &stack)
                .expect("dense conservative precommit");
        pre_keys.push(key);
        pre_commitments.push(commitment);
        pre_hints.push(hint);
        pre_polys_by_group.push(polys);
        pre_evals_by_group.push(evals);
    }

    let mut final_polys = Vec::new();
    let mut final_evals = Vec::new();
    for poly_idx in 0..final_size {
        let (poly, poly_evals) =
            make_seeded_dense_poly(FINAL_NV, 0xdeb5_ef00_0000_0000 + poly_idx as u64);
        final_polys.push(poly);
        final_evals.push(poly_evals);
    }
    let (final_commitment, final_hint) =
        DenseScheme::commit_final_group(&setup, &final_polys, &stack, pre_keys)
            .expect("dense final multi-group commitment");

    let point = debug_random_point(FINAL_NV);
    let pre_openings: Vec<Vec<F>> = pre_evals_by_group
        .iter()
        .map(|group_evals| {
            group_evals
                .iter()
                .map(|evals| dense_opening(evals, &point[..PRE_NV]))
                .collect()
        })
        .collect();
    let final_openings: Vec<F> = final_evals
        .iter()
        .map(|evals| dense_opening(evals, &point))
        .collect();

    let pre_refs_by_group: Vec<Vec<&DensePoly<F>>> = pre_polys_by_group
        .iter()
        .map(|polys| polys.iter().collect())
        .collect();
    let final_refs: Vec<&DensePoly<F>> = final_polys.iter().collect();

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
    .expect("dense multi-group prover data");

    let mut prover_transcript = AkitaTranscript::<F>::new(b"test/multi-group-dense");
    let proof = DenseScheme::batched_prove(
        &setup,
        prover_claims,
        &stack,
        &mut prover_transcript,
        BasisMode::Lagrange,
        akita_types::SetupContributionMode::Direct,
    )
    .expect("dense multi-group prove");
    assert!(!matches!(
        proof.root,
        akita_types::AkitaBatchedRootProof::ZeroFold { .. }
    ));
    if matches!(proof.root, akita_types::AkitaBatchedRootProof::Fold(_)) {
        assert!(
            !proof.steps.is_empty(),
            "intermediate dense multi-group root must hand off to a suffix"
        );
    }

    let shape = proof.shape();
    let mut bytes = Vec::new();
    proof
        .serialize_uncompressed(&mut bytes)
        .expect("serialize dense multi-group proof");
    let decoded =
        akita_types::AkitaBatchedProof::<F, F>::deserialize_uncompressed(&bytes[..], &shape)
            .expect("deserialize dense multi-group proof");
    assert_eq!(decoded, proof);

    let verifier_setup = DenseScheme::setup_verifier(&setup);
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
            final_openings.clone(),
            &final_commitment,
        )
        .expect("final verifier group"),
    );
    let verify_claims = OpeningClaims::from_groups(point.clone(), verifier_groups)
        .expect("dense multi-group verifier claims");
    let mut verifier_transcript = AkitaTranscript::<F>::new(b"test/multi-group-dense");
    DenseScheme::batched_verify(
        &decoded,
        &verifier_setup,
        &mut verifier_transcript,
        verify_claims,
        BasisMode::Lagrange,
        akita_types::SetupContributionMode::Direct,
    )
    .expect("dense multi-group verify");

    // Negative: tampering the final group's first opening must reject.
    let mut tampered_groups = Vec::new();
    for (group_idx, openings) in pre_openings.iter().enumerate() {
        tampered_groups.push(
            PolynomialGroupClaims::new(
                PointVariableSelection::prefix(PRE_NV, FINAL_NV).expect("pre point vars"),
                openings.clone(),
                &pre_commitments[group_idx],
            )
            .expect("pre verifier group"),
        );
    }
    let mut tampered_final = final_openings.clone();
    tampered_final[0] += F::one();
    tampered_groups.push(
        PolynomialGroupClaims::new(
            PointVariableSelection::prefix(FINAL_NV, FINAL_NV).expect("final point vars"),
            tampered_final,
            &final_commitment,
        )
        .expect("final verifier group"),
    );
    let tampered_claims = OpeningClaims::from_groups(point.clone(), tampered_groups)
        .expect("tampered verifier claims");
    let mut tampered_transcript = AkitaTranscript::<F>::new(b"test/multi-group-dense");
    assert!(
        DenseScheme::batched_verify(
            &decoded,
            &verifier_setup,
            &mut tampered_transcript,
            tampered_claims,
            BasisMode::Lagrange,
            akita_types::SetupContributionMode::Direct,
        )
        .is_err(),
        "tampered dense group opening must reject"
    );
}

#[test]
fn multi_group_root_folded_dense_two_group_round_trips() {
    multi_group_root_round_trip_dense(&[1], 1);
}

#[test]
fn multi_group_root_folded_dense_unequal_one_two_round_trips() {
    multi_group_root_round_trip_dense(&[1], 2);
}
