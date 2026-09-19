//! Test-only descriptor fixture: overlapping supports, shifted source lanes and holes.
use super::*;

pub(crate) fn fixture<E: FieldCore + FromPrimitiveInt>(
    c: usize,
    values: Vec<E>,
    packing: bool,
) -> (PreparedProverLinearTerms<E>, Vec<E>) {
    let lanes = 6;
    assert_eq!(values.len(), 4 * c);
    let segments = [
        (0, 1, 3, E::from_u64(3)),
        (1, 0, 3, E::from_u64(5)),
        (5, 2, 1, E::from_u64(7)),
    ];
    // Independent flat scatter oracle, not materialize_dense/get/pair_from_flat_index.
    let mut dense = vec![E::zero(); lanes * c];
    for &(target, source, count, factor) in &segments {
        for lane in 0..count {
            for i in 0..c {
                dense[(target + lane) * c + i] += factor * values[(source + lane) * c + i];
            }
        }
    }
    let prepared = if packing {
        let mut map = PreparedPackingLaneMap {
            segments: Vec::new(),
            lane_to_segment: vec![None; lanes],
            overlapping_segments: BTreeMap::new(),
        };
        for &(target, source, count, factor) in &segments {
            let index = map.segments.len();
            map.segments.push(PreparedPackingSegment {
                factor,
                source_index: 0,
                target_lane_start: target,
                source_lane_start: source,
                lane_count: count,
            });
            for lane in target..target + count {
                map.add_segment(lane, index).unwrap();
            }
        }
        assert!(map.overlapping_segments.contains_key(&1));
        assert!(map.lane_to_segment[4].is_none());
        PreparedProverLinearTerms {
            lane_weights: PreparedLaneWeights::Packing(map),
            sources: vec![PreparedTraceSource {
                values,
                lane_count: 4,
            }],
            live_lane_count: lanes,
            coeff_count: c,
        }
    } else {
        let weights = StructuredLinearWeights {
            sources: vec![Arc::from(values)],
            physical_field_len: lanes * c,
            segments: segments
                .iter()
                .map(|&(t, s, n, _)| StructuredLinearSegment {
                    physical_coefficient_start: t * c,
                    source_coefficient_start: s * c,
                    coefficient_count: n * c,
                })
                .collect(),
            terms: segments
                .iter()
                .enumerate()
                .map(|(i, &(_, _, _, factor))| StructuredLinearTerm {
                    factor,
                    source_index: 0,
                    segment_range: i..i + 1,
                })
                .collect(),
        };
        let p = PreparedProverLinearTerms::from_structured_weights(&weights, c).unwrap();
        assert!(matches!(p.lane_weights, PreparedLaneWeights::Sparse(_)));
        p
    };
    prepared.validate_len(lanes * c).unwrap();
    assert_eq!(prepared.source_count(), 1);
    (prepared, dense)
}
