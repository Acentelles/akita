use super::poly::DensePoly;
use akita_algebra::CyclotomicRing;
use akita_field::Prime128OffsetA7F7 as F;

fn ring<const D: usize>(offset: u64) -> CyclotomicRing<F, D> {
    CyclotomicRing::from_coefficients(std::array::from_fn(|idx| {
        F::from_u64(offset + idx as u64 + 1)
    }))
}

#[test]
fn ring_fold_matches_dense_multiplication_reference() {
    const D: usize = 8;
    let coeffs = (0..2).map(|idx| ring::<D>(10 * idx)).collect::<Vec<_>>();
    let poly = DensePoly::<F>::from_ring_coeffs(coeffs.clone());
    let scalars = vec![
        ring::<D>(100),
        ring::<D>(200),
        ring::<D>(300),
        ring::<D>(400),
    ];
    let got = poly.fold_blocks_ring(&scalars, 4);
    let expected = coeffs
        .chunks(4)
        .map(|block| {
            block
                .iter()
                .zip(scalars.iter())
                .fold(CyclotomicRing::<F, D>::zero(), |acc, (coeff, scalar)| {
                    acc + (*coeff * *scalar)
                })
        })
        .collect::<Vec<_>>();

    assert_eq!(got, expected);
}

#[test]
fn dense_constructor_reuses_owned_evaluation_buffer() {
    let evals = (0..2048).map(F::from_u64).collect::<Vec<_>>();
    let allocation = evals.as_ptr();
    let poly = DensePoly::<F>::from_field_evals(11, evals).unwrap();
    assert_eq!(poly.field_coeffs().as_ptr(), allocation);
}

#[test]
fn dense_source_has_exact_views_across_supported_ring_dimensions() {
    let evals = (1..=32).map(F::from_u64).collect::<Vec<_>>();
    let poly = DensePoly::<F>::from_field_evals(5, evals.clone()).unwrap();

    fn assert_view<const D: usize>(poly: &DensePoly<F>, evals: &[F]) {
        let rings = poly.ring_coeffs::<D>().expect("supported dense view");
        let flat = rings
            .iter()
            .flat_map(|ring| ring.coefficients().iter().copied())
            .collect::<Vec<_>>();
        assert_eq!(&flat[..evals.len()], evals);
        assert!(flat[evals.len()..].iter().all(|value| *value == F::zero()));
    }

    assert_view::<64>(&poly, &evals);
    assert_view::<128>(&poly, &evals);
    assert_view::<256>(&poly, &evals);
    assert_view::<512>(&poly, &evals);
    assert_view::<1024>(&poly, &evals);
}

/// Honest signed digits wider than i8 must survive the root fold exactly.
/// The reference is a direct signed-integer negacyclic convolution, independent
/// of the optimized digit decomposition and sparse accumulation helpers.
#[test]
fn dense_single_digit_wide_basis_matches_integer_negacyclic_reference() {
    use akita_challenges::SparseChallenge;
    const D: usize = 128;
    const POSITIONS: usize = 2;
    const BLOCKS: usize = 2;
    let challenges = [
        SparseChallenge {
            positions: vec![0, 3, 127].into(),
            coeffs: vec![1, -1, 2].into(),
        },
        SparseChallenge {
            positions: vec![1, 64, 126].into(),
            coeffs: vec![-1, 1, -2].into(),
        },
    ];
    for log_basis in [9u32, 10, 12, 16] {
        let half = 1i32 << (log_basis - 1);
        let boundary = [-half, -129, -128, -1, 0, 127, 128, half - 1];
        let coefficients = (0..BLOCKS * POSITIONS * D)
            .map(|index| boundary[(index + index / D) % boundary.len()])
            .collect::<Vec<_>>();
        let rings = coefficients
            .chunks_exact(D)
            .map(|row| {
                CyclotomicRing::<F, D>::from_coefficients(std::array::from_fn(|index| {
                    let value = row[index];
                    if value < 0 {
                        -F::from_u64(u64::from(value.unsigned_abs()))
                    } else {
                        F::from_u64(value as u64)
                    }
                }))
            })
            .collect();
        let poly = DensePoly::<F>::from_ring_coeffs(rings);
        let witness = poly.decompose_fold::<D>(&challenges, POSITIONS, 1, log_basis);
        let mut expected = vec![0i32; POSITIONS * D];
        for (block, challenge) in challenges.iter().enumerate() {
            for position in 0..POSITIONS {
                for coefficient in 0..D {
                    let value = coefficients[(block * POSITIONS + position) * D + coefficient];
                    for (&shift, &factor) in challenge.positions.iter().zip(&challenge.coeffs) {
                        let exponent = coefficient + shift as usize;
                        let sign = if exponent < D { 1 } else { -1 };
                        expected[position * D + exponent % D] += value * i32::from(factor) * sign;
                    }
                }
            }
        }
        assert_eq!(
            witness.centered_coeffs_flat(),
            expected,
            "basis {log_basis}"
        );
    }
}
