use super::*;
const LIMITS: Limits = Limits {
    source_count: 1 << 16,
    source_elements: 1 << 20,
    references: 1 << 20,
    lanes: 1 << 16,
};
fn decode(x: [u64; 2]) -> F {
    const P: u64 = u64::MAX - 58;
    assert!(x[0] < P && x[1] < P);
    F::new(
        Prime64Offset59::from_u64(x[0]),
        Prime64Offset59::from_u64(x[1]),
    )
}
fn materialize(wire: &Serialized) -> Vec<F> {
    let lanes = wire.lanes as usize;
    let c = wire.coefficients as usize;
    assert_eq!(wire.lane_offsets.len(), lanes + 1);
    assert_eq!(wire.lane_offsets[0], 0);
    assert_eq!(
        *wire.lane_offsets.last().unwrap(),
        wire.references.len() as u64
    );
    let mut output = vec![F::zero(); lanes * c];
    for lane in 0..lanes {
        let start = wire.lane_offsets[lane] as usize;
        let end = wire.lane_offsets[lane + 1] as usize;
        assert!(start <= end);
        for &[source_index, source_lane, f0, f1] in &wire.references[start..end] {
            let factor = decode([f0, f1]);
            let [base, source_lanes] = wire.source_records[source_index as usize];
            assert!(source_lane < source_lanes);
            let source = base as usize + source_lane as usize * c;
            assert!(source + c <= wire.sources.len());
            for i in 0..c {
                output[lane * c + i] += factor * decode(wire.sources[source + i]);
            }
        }
    }
    output
}
fn check(mut terms: PreparedProverLinearTerms<F>, mut dense: Vec<F>, mut c: usize) {
    let lanes = dense.len() / c;
    while c >= 2 {
        let wire = terms.serialize_resident(LIMITS).unwrap();
        assert_eq!(
            wire.sources.len(),
            terms.sources.iter().map(|s| s.values.len()).sum()
        );
        assert_eq!(materialize(&wire), dense);
        assert_eq!(terms.materialize_dense(), dense);
        let rho = F::new(
            Prime64Offset59::from_u64(u64::MAX - 97 - c as u64),
            Prime64Offset59::from_u64(9123 + c as u64),
        );
        let next = c / 2;
        let mut folded = vec![F::zero(); lanes * next];
        for lane in 0..lanes {
            for i in 0..next {
                let a = dense[lane * c + 2 * i];
                let b = dense[lane * c + 2 * i + 1];
                folded[lane * next + i] = a + rho * (b - a);
            }
        }
        terms.fold_coefficients(rho);
        dense = folded;
        c = next;
    }
    assert!(terms.serialize_resident(LIMITS).is_err());
}
#[test]
fn resident_descriptors_match_authenticated_packing_sparse_and_cpu_evolution() {
    for packing in [false, true] {
        let values = (0..256)
            .map(|i| {
                F::new(
                    Prime64Offset59::from_u64(71 + i),
                    Prime64Offset59::from_u64(u64::MAX - 1000 - i),
                )
            })
            .collect();
        let (p, dense) =
            super::super::resident_stage2_factored_fixture::fixture(64, values, packing);
        check(p, dense, 64);
    }
    for basis in [
        akita_types::BasisMode::Lagrange,
        akita_types::BasisMode::Monomial,
    ] {
        let (p, dense, c) =
            super::super::super::coefficient_packing_terms::tests::resident_semantic_fixture(basis);
        check(p, dense, c);
    }
    let zero = PreparedProverLinearTerms::<F>::zero(3, 64);
    let wire = zero.serialize_resident(LIMITS).unwrap();
    assert!(wire.sources.is_empty() && wire.references.is_empty());
    assert_eq!(materialize(&wire), vec![F::zero(); 192]);
}
#[test]
fn resident_descriptors_reject_malformed_maps_and_budgets() {
    let make =
        || super::super::resident_stage2_factored_fixture::fixture(64, vec![F::one(); 256], true).0;
    assert!(make()
        .serialize_resident(Limits {
            source_elements: 255,
            ..LIMITS
        })
        .is_err());
    assert!(make()
        .serialize_resident(Limits {
            references: 0,
            ..LIMITS
        })
        .is_err());
    assert!(make()
        .serialize_resident(Limits { lanes: 5, ..LIMITS })
        .is_err());
    let mut p = make();
    p.sources[0].values.pop();
    assert!(p.serialize_resident(LIMITS).is_err());
    let mut p = make();
    if let PreparedLaneWeights::Packing(m) = &mut p.lane_weights {
        m.segments[0].source_index = usize::MAX;
    }
    assert!(p.serialize_resident(LIMITS).is_err());
    let mut p = make();
    if let PreparedLaneWeights::Packing(m) = &mut p.lane_weights {
        m.segments[0].target_lane_start = usize::MAX;
    }
    assert!(p.serialize_resident(LIMITS).is_err());
    let mut p = make();
    if let PreparedLaneWeights::Packing(m) = &mut p.lane_weights {
        m.overlapping_segments.remove(&1);
    }
    assert!(p.serialize_resident(LIMITS).is_err());
    let mut p = make();
    if let PreparedLaneWeights::Packing(m) = &mut p.lane_weights {
        m.overlapping_segments.get_mut(&1).unwrap().reverse();
    }
    assert!(p.serialize_resident(LIMITS).is_err());
    let mut p = PreparedProverLinearTerms::from_dense(vec![F::one(); 128], 2, 64);
    if let PreparedLaneWeights::Sparse(m) = &mut p.lane_weights {
        m[0][0].lane = 1;
    }
    assert!(p.serialize_resident(LIMITS).is_err());
}
