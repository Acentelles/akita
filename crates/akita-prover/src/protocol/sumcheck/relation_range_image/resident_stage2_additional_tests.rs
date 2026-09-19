use super::*;
use akita_algebra::offset_eq::{
    materialize_eq_tensor_left, EqPairTensorAxis, EqPairTensorFamily, OffsetEqWindow,
};
/// Uses the same checked additional-term boundary and L2 tensor materializer as stages.rs.
/// Compression-shaped sparse inputs are explicit fixtures, not a claimed real MLDSA layout.
pub(crate) fn fixture(witness: &[i8], point: &[F], mode: usize) -> AdditionalRelationTerms<F> {
    let domain = 1usize << point.len();
    assert!(domain >= 16);
    let mut linear = if mode == 0 {
        vec![]
    } else {
        vec![
            (1, F::from_u64(3)),
            (1, F::from_u64(5)),
            (3, F::from_u64(7)),
            (domain - 1, F::from_u64(11)),
        ]
    };
    if mode >= 2 {
        // Same ring/limb/row tensor shape used by PhysicalResponsePlan::virtualization_families.
        let family = EqPairTensorFamily::new(
            0,
            0,
            F::one(),
            vec![
                EqPairTensorAxis::unit(2, 1, 1),
                EqPairTensorAxis::dense(2, 0, vec![F::from_u64(13), F::from_u64(17)]),
                EqPairTensorAxis::unit(2, 4, 2),
            ],
        )
        .unwrap();
        linear.extend(
            materialize_eq_tensor_left(
                &OffsetEqWindow::new(point).unwrap(),
                &[family],
                witness.len(),
            )
            .unwrap()
            .into_iter()
            .enumerate()
            .filter(|(_, v)| !v.is_zero()),
        );
        linear.sort_unstable_by_key(|(i, _)| *i);
    }
    let intervals = if mode == 0 || mode == 3 {
        vec![]
    } else {
        vec![2..6, domain - 2..domain]
    };
    AdditionalRelationTerms::new(witness, domain, linear, &intervals, point, F::from_u64(19))
        .unwrap()
}
fn value(v: [u64; 2]) -> F {
    F::new(
        Prime64Offset59::from_u64(v[0]),
        Prime64Offset59::from_u64(v[1]),
    )
}
fn evaluate(wire: &AdditionalDescriptor, witness: &[F], x: F) -> F {
    let at = |i: usize| witness.get(i).copied().unwrap_or_else(F::zero);
    let mut sum = F::zero();
    let beta = value(wire.binary_batching);
    for p in &wire.pairs {
        let w0 = at(2 * p[0] as usize);
        let w1 = at(2 * p[0] as usize + 1);
        let w = w0 + x * (w1 - w0);
        let l0 = value([p[1], p[2]]);
        let l1 = value([p[3], p[4]]);
        let b0 = value([p[5], p[6]]);
        let b1 = value([p[7], p[8]]);
        sum += w * (l0 + x * (l1 - l0)) + beta * (b0 + x * (b1 - b0)) * w * (w + F::one());
    }
    sum
}
#[test]
fn resident_additional_constructor_export_matches_cpu_after_binds() {
    let point = (0..5)
        .map(|i| {
            F::new(
                Prime64Offset59::from_u64(i + 2),
                Prime64Offset59::from_u64(i + 31),
            )
        })
        .collect::<Vec<_>>();
    let compact = (0..23)
        .map(|i| if i % 2 == 0 { -1 } else { 0 })
        .collect::<Vec<i8>>();
    for mode in 0..4 {
        let mut p = fixture(&compact, &point, mode);
        let mut witness = compact
            .iter()
            .map(|&w| F::from_i64(w as i64))
            .collect::<Vec<_>>();
        for round in 0..4 {
            let wire = p.serialize_resident_additional(witness.len(), 64).unwrap();
            assert_eq!(wire.live_len, witness.len() as u64);
            let expected = p.round_polynomial_folded(&witness);
            for x in [
                F::zero(),
                F::one(),
                F::from_u64(7),
                F::new(
                    Prime64Offset59::from_u64(999),
                    Prime64Offset59::from_u64(101),
                ),
            ] {
                assert_eq!(evaluate(&wire, &witness, x), expected.evaluate(&x));
            }
            let rho = F::new(
                Prime64Offset59::from_u64(51 + round),
                Prime64Offset59::from_u64(79 + round),
            );
            p.bind(rho);
            witness = witness
                .chunks(2)
                .map(|v| v[0] + rho * (v.get(1).copied().unwrap_or_else(F::zero) - v[0]))
                .collect();
        }
    }
}
#[test]
fn resident_additional_rejects_malformed_shape_and_budget() {
    let point = vec![F::from_u64(3); 5];
    let make = || fixture(&[-1; 23], &point, 1);
    assert!(make().serialize_resident_additional(33, 64).is_err());
    assert!(make().serialize_resident_additional(23, 0).is_err());
    let mut p = make();
    p.weights.reverse();
    assert!(p.serialize_resident_additional(23, 64).is_err());
    let mut p = make();
    p.weights[0].index = 32;
    assert!(p.serialize_resident_additional(23, 64).is_err());
    let mut p = make();
    p.domain_len = 31;
    assert!(p.serialize_resident_additional(23, 64).is_err());
    let p = fixture(&[-1; 23], &point, 0);
    assert!(p
        .serialize_resident_additional(23, 0)
        .unwrap()
        .pairs
        .is_empty());
}

#[test]
fn resident_additional_production_compression_support_matches_cpu() {
    let (live, domain, weights, binary) =
        super::super::super::coefficient_packing_terms::tests::resident_compression_support_fixture(
        );
    assert!(live <= domain && domain.is_power_of_two());
    let point = (0..domain.trailing_zeros())
        .map(|i| {
            F::new(
                Prime64Offset59::from_u64(2 + i as u64),
                Prime64Offset59::from_u64(31 + i as u64),
            )
        })
        .collect::<Vec<_>>();
    let compact = (0..live)
        .map(|i| if i % 3 == 0 { -1i8 } else { 0i8 })
        .collect::<Vec<_>>();
    let mut terms =
        AdditionalRelationTerms::new(&compact, domain, weights, &binary, &point, F::from_u64(19))
            .unwrap();
    let mut witness = compact
        .iter()
        .map(|&x| F::from_i64(x as i64))
        .collect::<Vec<_>>();
    for round in 0..4 {
        let wire = terms
            .serialize_resident_additional(witness.len(), domain / 2)
            .unwrap();
        assert!(!wire.pairs.is_empty());
        let expected = terms.round_polynomial_folded(&witness);
        for x in [
            F::zero(),
            F::one(),
            F::from_u64(7),
            F::new(
                Prime64Offset59::from_u64(999),
                Prime64Offset59::from_u64(101),
            ),
        ] {
            assert_eq!(evaluate(&wire, &witness, x), expected.evaluate(&x));
        }
        let rho = F::from_u64(51 + round);
        terms.bind(rho);
        witness = witness
            .chunks(2)
            .map(|v| v[0] + rho * (v.get(1).copied().unwrap_or_else(F::zero) - v[0]))
            .collect();
    }
}
