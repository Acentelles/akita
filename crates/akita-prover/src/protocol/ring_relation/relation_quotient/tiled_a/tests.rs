use super::*;
use akita_challenges::SparseChallenge;
use akita_field::{Prime128OffsetA7F7, Prime32Offset99, Prime64Offset59};

fn rings<F: FieldCore + CanonicalField, const D: usize>(count: usize) -> Vec<CyclotomicRing<F, D>> {
    let mut state = 0x7bbd_15a8_632f_19c1u64;
    (0..count)
        .map(|_| {
            CyclotomicRing::from_coefficients(std::array::from_fn(|i| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                let value = F::from_u64(state);
                match i % 4 {
                    0 => -F::one(),
                    1 => F::zero(),
                    2 => value * value,
                    _ => -value,
                }
            }))
        })
        .collect()
}

fn challenges<const D: usize>(count: usize) -> Challenges {
    let sparse = (0..count)
        .map(|i| SparseChallenge {
            // Include degree-zero terms, the last coefficient, cancellation
            // at a repeated position and both production/fallback signs.
            positions: vec![0, (D - 1) as u32, (D / 2) as u32, (D / 2) as u32].into(),
            coeffs: vec![i as i8, (i.wrapping_mul(17)) as i8, 2, -2].into(),
        })
        .collect();
    // Nontrivial claim-major order when the count permits it.
    let claims = if count.is_multiple_of(3) && count != 0 {
        3
    } else {
        1
    };
    Challenges::from_sparse(sparse, count / claims, claims).unwrap()
}

fn independent_high_halves<F: FieldCore + CanonicalField, const D: usize>(
    challenges: &Challenges,
    rings: &[CyclotomicRing<F, D>],
    rows: usize,
) -> Vec<Vec<F>> {
    (0..rows)
        .map(|row| {
            let mut result = vec![F::zero(); D];
            for (i, challenge) in challenges.as_slice().iter().enumerate() {
                let Some(ring) = rings.get(i * rows + row) else {
                    continue;
                };
                for (&position, &coefficient) in
                    challenge.positions.iter().zip(challenge.coeffs.iter())
                {
                    let position = position as usize;
                    let scale = F::from_i64(i64::from(coefficient));
                    for source in D - position..D {
                        result[position + source - D] += ring.coefficients()[source] * scale;
                    }
                }
            }
            result
        })
        .collect()
}

fn assert_parity<F: FieldCore + CanonicalField, const D: usize>(
    challenges: &Challenges,
    source: &[CyclotomicRing<F, D>],
    rows: usize,
) {
    let scalar = accumulate_a_high_halves(challenges, source, rows, false).unwrap();
    let tiled = accumulate_a_high_halves(challenges, source, rows, true).unwrap();
    let independent = independent_high_halves(challenges, source, rows);
    assert_eq!(tiled, scalar, "D={D} rows={rows} rings={}", source.len());
    assert_eq!(tiled, independent);
    assert_eq!(tiled.len(), rows);
}

fn all_geometries<F: FieldCore + CanonicalField>() {
    for count in [0, 1, 3, 17, 65] {
        let challenges = challenges::<8>(count);
        for rows in [0, 1, 3, 4, 5, 9] {
            let source = rings::<F, 8>(count * rows + 2);
            // Extra source rings are ignored; missing/partial final A-row
            // blocks contribute zero exactly as the original optional lookup.
            for present in [0, rows.saturating_sub(1), count * rows, source.len()] {
                assert_parity(&challenges, &source[..present.min(source.len())], rows);
            }
        }
    }
    for rows in [1, 4, 5] {
        let challenges = challenges::<8>(256);
        assert_parity(&challenges, &rings::<F, 8>(256 * rows - 1), rows);
    }
}

#[test]
fn tiled_a_high_halves_match_full_field_scalar_and_dense_oracles() {
    all_geometries::<Prime32Offset99>();
    all_geometries::<Prime64Offset59>();
    all_geometries::<Prime128OffsetA7F7>();
}

fn large_dimension<const D: usize>() {
    let challenges = challenges::<D>(9);
    let source = rings::<Prime128OffsetA7F7, D>(9 * 5);
    assert_parity(&challenges, &source, 5);
    assert_parity(&challenges, &source[..source.len() - 3], 5);
}

#[test]
fn tiled_a_high_halves_cover_native_dimensions_and_incomplete_tiles() {
    large_dimension::<1>();
    large_dimension::<64>();
    large_dimension::<128>();
    large_dimension::<256>();
    large_dimension::<512>();
    large_dimension::<1024>();
    large_dimension::<2048>();
}

#[test]
fn tiled_a_high_halves_preserve_sparse_zip_and_zero_terms() {
    let challenges = Challenges::from_sparse(
        vec![
            SparseChallenge {
                positions: vec![0, 7, 3].into(),
                coeffs: vec![0, 1].into(),
            },
            SparseChallenge {
                positions: vec![7].into(),
                coeffs: vec![-1, -128].into(),
            },
            SparseChallenge {
                positions: vec![].into(),
                coeffs: vec![2].into(),
            },
        ],
        1,
        3,
    )
    .unwrap();
    assert_parity(&challenges, &rings::<Prime128OffsetA7F7, 8>(15), 5);
}
