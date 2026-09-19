use super::resident_stage2::{scalar, Request};
use super::*;
use akita_config::proof_optimized::fp64::{ExtensionField as F, Field as B};

fn field(i: u64) -> F {
    fn mix(mut x: u64) -> u64 {
        x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
        x ^ (x >> 31)
    }
    F::new(
        B::from_u64(mix(i.wrapping_add(0x9e3779b97f4a7c15))),
        B::from_u64(mix(i ^ 0xa53d)),
    )
}

fn assert_norm(a: NormRoundTerms<F>, b: NormRoundTerms<F>) {
    match (a, b) {
        (NormRoundTerms::Full(a), NormRoundTerms::Full(b)) => assert_eq!(a, b),
        (NormRoundTerms::SkipLinear(a), NormRoundTerms::SkipLinear(b)) => assert_eq!(a, b),
        _ => panic!("norm variant changed"),
    }
}

fn fixture(
    lanes: usize,
    coefficient_bits: usize,
    basis: usize,
    full_norm: bool,
) -> (RelationRangeImageProver<F>, Vec<i8>) {
    fixture_with_linear(lanes, coefficient_bits, basis, full_norm, None)
}

fn fixture_with_linear(
    lanes: usize,
    coefficient_bits: usize,
    basis: usize,
    full_norm: bool,
    supplied: Option<(PreparedProverLinearTerms<F>, Vec<F>)>,
) -> (RelationRangeImageProver<F>, Vec<i8>) {
    let capacity = lanes.next_power_of_two();
    let c = 1usize << coefficient_bits;
    let lane_bits = capacity.trailing_zeros() as usize;
    let compact: Vec<i8> = (0..lanes * c)
        .map(|i| ((i * 13 + 7) % basis) as i8 - (basis / 2) as i8)
        .collect();
    let alpha: Vec<F> = (0..c).map(|i| field(i as u64 + 100)).collect();
    let weights: Vec<F> = (0..capacity).map(|i| field(i as u64 + 200)).collect();
    let (prepared_linear, linear) = supplied.unwrap_or_else(|| {
        let linear: Vec<F> = (0..lanes * c).map(|i| field(i as u64 + 300)).collect();
        (
            PreparedProverLinearTerms::from_dense(linear.clone(), lanes, c),
            linear,
        )
    });
    assert_eq!(linear.len(), lanes * c);
    let mut point: Vec<F> = (0..lane_bits + coefficient_bits)
        .map(|i| field(i as u64 + 400))
        .collect();
    if full_norm {
        point[2] = F::zero();
    }
    let mut range = F::zero();
    let mut relation = F::zero();
    let mut trace = F::zero();
    for (i, &v) in compact.iter().enumerate() {
        let w = F::from_i64(v as i64);
        let mut eq = F::one();
        for (bit, &r) in point.iter().enumerate() {
            eq *= if (i >> bit) & 1 == 1 { r } else { F::one() - r };
        }
        range += eq * w * (w + F::one());
        relation += w * alpha[i % c] * weights[i / c];
        trace += w * linear[i];
    }
    let prover = RelationRangeImageProver::new(
        field(19),
        compact.clone(),
        &point,
        range,
        basis,
        alpha,
        weights,
        lanes,
        lane_bits,
        coefficient_bits,
        relation,
        prepared_linear,
        trace,
        None,
    )
    .unwrap();
    (prover, compact)
}

#[test]
fn resident_stage2_compact_and_coefficient_requests_match_optimized_cpu() {
    for basis in [4, 8] {
        for lanes in [1, 3, 6] {
            for full_norm in [false, true] {
                let (mut p, compact) = fixture(lanes, 6, basis, full_norm);
                let r0 = field(901);
                let r1 = field(902);
                let alpha =
                    RelationRangeImageProver::fold_alpha_two_rounds(&p.common_alpha_factor, r0, r1);
                p.split_eq.bind(r0);
                p.split_eq.bind(r1);
                p.linear_terms.fold_two_coefficients(r0, r1);
                let request =
                    Request::compact(&p, &compact, &alpha, &p.linear_terms, r0, r1).unwrap();
                assert_eq!(request.skip_linear, !full_norm);
                let actual = scalar(&request).unwrap();
                let (expected, norm, relation) = p
                    .materialize_two_round_compact_prefix_and_compute_next_round(
                        &compact,
                        &alpha,
                        &p.linear_terms,
                        r0,
                        r1,
                    );
                assert_eq!(actual.witness, expected);
                assert_norm(actual.norm, norm);
                assert_eq!(actual.relation, relation);
                p.common_alpha_factor = alpha;
                p.rounds_completed = 2;
                let mut witness = actual.witness;
                for round in 0..3 {
                    let rho = field(1000 + round);
                    p.split_eq.bind(rho);
                    p.fold_linear_terms_for_current_round(rho);
                    let mut next = p.common_alpha_factor.clone();
                    fold_evals_in_place(&mut next, rho);
                    let request = Request::coefficients(&p, &witness, &next, rho).unwrap();
                    let actual = scalar(&request).unwrap();
                    let (expected, norm, relation) =
                        p.fuse_folded_coefficients_and_compute_next_round(&witness, &next, rho);
                    assert_eq!(actual.witness, expected);
                    assert_norm(actual.norm, norm);
                    assert_eq!(actual.relation, relation);
                    p.common_alpha_factor = next;
                    p.rounds_completed += 1;
                    witness = actual.witness;
                }
            }
        }
    }
}

#[test]
fn resident_stage2_requests_reject_shapes_before_output_or_state_change() {
    let (mut p, mut compact) = fixture(3, 4, 8, false);
    let r0 = field(91);
    let r1 = field(92);
    let alpha = RelationRangeImageProver::fold_alpha_two_rounds(&p.common_alpha_factor, r0, r1);
    p.split_eq.bind(r0);
    p.split_eq.bind(r1);
    p.linear_terms.fold_two_coefficients(r0, r1);
    let before = p.common_alpha_factor.clone();
    let rounds = p.rounds_completed;
    assert!(Request::compact(
        &p,
        &compact[..compact.len() - 1],
        &alpha,
        &p.linear_terms,
        r0,
        r1
    )
    .is_err());
    assert!(Request::compact(
        &p,
        &compact,
        &alpha[..alpha.len() - 1],
        &p.linear_terms,
        r0,
        r1
    )
    .is_err());
    compact[0] = 4;
    assert!(Request::compact(&p, &compact, &alpha, &p.linear_terms, r0, r1).is_err());
    compact[0] = -4;
    let wrong_linear = PreparedProverLinearTerms::zero(3, alpha.len() * 2);
    assert!(Request::compact(&p, &compact, &alpha, &wrong_linear, r0, r1).is_err());
    let mut request = Request::compact(&p, &compact, &alpha, &p.linear_terms, r0, r1).unwrap();
    assert!(request.linear_pair(3, 0).is_err());
    assert!(request.linear_pair(0, 1).is_err());
    request.eq_first = &[];
    assert!(scalar(&request).is_err());
    assert_eq!(p.common_alpha_factor, before);
    assert_eq!(p.rounds_completed, rounds);
}

fn check_factored(
    linear: PreparedProverLinearTerms<F>,
    dense: Vec<F>,
    c: usize,
    basis: usize,
    full_norm: bool,
) {
    assert!(c.is_power_of_two() && c >= 32);
    assert_eq!(dense.len() % c, 0);
    let lanes = dense.len() / c;
    let mut dense_linear = PreparedProverLinearTerms::from_dense(dense.clone(), lanes, c);
    let (mut p, compact) = fixture_with_linear(
        lanes,
        c.trailing_zeros() as usize,
        basis,
        full_norm,
        Some((linear, dense)),
    );
    let r0 = field(901);
    let r1 = field(902);
    let alpha = RelationRangeImageProver::fold_alpha_two_rounds(&p.common_alpha_factor, r0, r1);
    p.split_eq.bind(r0);
    p.split_eq.bind(r1);
    p.linear_terms.fold_two_coefficients(r0, r1);
    dense_linear.fold_two_coefficients(r0, r1);
    let req = Request::compact(&p, &compact, &alpha, &p.linear_terms, r0, r1).unwrap();
    let got = scalar(&req).unwrap();
    let dense_req = Request::compact(&p, &compact, &alpha, &dense_linear, r0, r1).unwrap();
    let independent = scalar(&dense_req).unwrap();
    assert_eq!(got.witness, independent.witness);
    assert_norm(got.norm, independent.norm);
    assert_eq!(got.relation, independent.relation);
    let (expected, norm, relation) = p.materialize_two_round_compact_prefix_and_compute_next_round(
        &compact,
        &alpha,
        &p.linear_terms,
        r0,
        r1,
    );
    assert_eq!(got.witness, expected);
    assert_norm(got.norm, norm);
    assert_eq!(got.relation, relation);
    p.common_alpha_factor = alpha;
    p.rounds_completed = 2;
    let mut witness = got.witness;
    for round in 0..3 {
        let rho = field(1000 + round);
        p.split_eq.bind(rho);
        p.fold_linear_terms_for_current_round(rho);
        dense_linear.fold_coefficients(rho);
        let mut alpha = p.common_alpha_factor.clone();
        fold_evals_in_place(&mut alpha, rho);
        let req = Request::coefficients(&p, &witness, &alpha, rho).unwrap();
        let got = scalar(&req).unwrap();
        std::mem::swap(&mut p.linear_terms, &mut dense_linear);
        let independent =
            scalar(&Request::coefficients(&p, &witness, &alpha, rho).unwrap()).unwrap();
        std::mem::swap(&mut p.linear_terms, &mut dense_linear);
        assert_eq!(got.witness, independent.witness);
        assert_norm(got.norm, independent.norm);
        assert_eq!(got.relation, independent.relation);
        let (expected, norm, relation) =
            p.fuse_folded_coefficients_and_compute_next_round(&witness, &alpha, rho);
        assert_eq!(got.witness, expected);
        assert_norm(got.norm, norm);
        assert_eq!(got.relation, relation);
        p.common_alpha_factor = alpha;
        p.rounds_completed += 1;
        witness = got.witness;
    }
}

#[test]
fn resident_stage2_factored_packing_and_sparse_match_dense_and_optimized_cpu() {
    for basis in [4, 8] {
        for full_norm in [false, true] {
            for packing in [false, true] {
                let values = (0..4 * 64).map(|i| field(7000 + i)).collect();
                let (linear, dense) =
                    super::evaluation_trace::resident_stage2_factored_fixture::fixture(
                        64, values, packing,
                    );
                check_factored(linear, dense, 64, basis, full_norm);
            }
            for packing_basis in [
                akita_types::BasisMode::Lagrange,
                akita_types::BasisMode::Monomial,
            ] {
                let (linear, dense, c) =
                    super::coefficient_packing_terms::tests::resident_semantic_fixture(
                        packing_basis,
                    );
                check_factored(linear, dense, c, basis, full_norm);
            }
        }
    }
}

#[path = "resident_stage2_fixture_export.rs"]
mod fixture_export;

#[path = "resident_stage2_compact_fixture_export.rs"]
mod compact_fixture_export;

#[path = "resident_stage2_owned_fixture_export.rs"]
mod owned_fixture_export;

#[cfg(feature = "resident-stage2-owned")]
#[path = "resident_stage2_owned_proof_tests.rs"]
mod owned_proof_tests;
