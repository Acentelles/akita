//! Pack live digit planes into bounded lazy dots without changing their columns.

use super::*;

pub(super) fn accumulate<W: PrimeWidth, const K: usize, const D: usize>(
    accs: &mut [CyclotomicCrtNtt<W, K, D>],
    matrix: &[&[CyclotomicCrtNtt<W, K, D>]],
    column_start: usize,
    digits: &[[i8; D]],
    params: &CrtNttParamSet<W, K, D>,
    lut: &DigitMontLut<W, K>,
    scratch: &mut I8ColumnScratch<W, K, D>,
) {
    let batch_size = params.pointwise_dot_batch_size();
    let empty = [0i8; D];
    let mut packed = [&empty; I32_LAZY_DOT_BATCH];
    let mut columns = [0usize; I32_LAZY_DOT_BATCH];
    let mut count = 0;
    for (offset, digit) in digits.iter().enumerate() {
        if is_zero_plane(digit) {
            continue;
        }
        packed[count] = digit;
        columns[count] = column_start + offset;
        count += 1;
        if count == batch_size {
            CyclotomicCrtNtt::add_assign_col_pointwise_dot_i8_multi_with_lut_scratch(
                accs,
                matrix,
                |index| columns[index],
                &packed[..count],
                params,
                lut,
                &mut scratch.lazy_dot,
            );
            count = 0;
        }
    }
    if count == 1 {
        CyclotomicCrtNtt::add_assign_col_pointwise_mul_i8_multi_with_lut_scratch(
            accs,
            matrix,
            columns[0],
            packed[0],
            params,
            lut,
            &mut scratch.rhs,
        );
    } else if count > 1 {
        CyclotomicCrtNtt::add_assign_col_pointwise_dot_i8_multi_with_lut_scratch(
            accs,
            matrix,
            |index| columns[index],
            &packed[..count],
            params,
            lut,
            &mut scratch.lazy_dot,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use akita_algebra::ntt::tables::{q128_primes, Q32_PRIMES, Q64_PRIMES};
    use akita_field::{Fp64, Prime128Offset275, Prime64Offset59};

    fn check<F: FieldCore + CanonicalField, const K: usize, const D: usize>(
        params: CrtNttParamSet<i32, K, D>,
    ) {
        // Signed boundary digits, separated live columns, empty/all-zero input,
        // every lazy-batch tail and a nonzero starting matrix column. Compare
        // against independent negacyclic schoolbook arithmetic, not an NTT path.
        const WIDTH: usize = 23;
        const START: usize = 3;
        let matrix: Vec<Vec<CyclotomicRing<F, D>>> = (0..3)
            .map(|row| {
                (0..WIDTH + START)
                    .map(|column| {
                        CyclotomicRing::from_coefficients(std::array::from_fn(|index| {
                            if (row + column + index) % 5 == 0 {
                                F::from_i128(i128::MAX - (column * D + index) as i128)
                            } else if (row + column + index) % 2 == 0 {
                                F::from_i64((29 * row + 13 * column + 7 * index) as i64)
                            } else {
                                -F::from_i64((31 * row + 17 * column + 11 * index + 1) as i64)
                            }
                        }))
                    })
                    .collect()
            })
            .collect();
        let transformed = precompute_dense_mat_ntt_with_params(&matrix, &params);
        let views: Vec<_> = transformed.iter().map(Vec::as_slice).collect();
        let lut = DigitMontLut::new_with_digit_bound(&params, 128);
        for length in 0..=WIDTH {
            for pattern in 0..4 {
                let digits: Vec<[i8; D]> = (0..length)
                    .map(|column| {
                        if pattern == 0 || (pattern != 3 && column % 3 == pattern - 1) {
                            [0; D]
                        } else {
                            std::array::from_fn(|index| match (column + index) % 4 {
                                0 => -128,
                                1 => 127,
                                2 => -1,
                                _ => 1,
                            })
                        }
                    })
                    .collect();
                let mut expected = vec![CyclotomicRing::<F, D>::zero(); matrix.len()];
                for (row, out) in matrix.iter().zip(&mut expected) {
                    for (column, digit) in digits.iter().enumerate() {
                        for (i, &a) in row[START + column].coefficients().iter().enumerate() {
                            for (j, &b) in digit.iter().enumerate() {
                                let product = a * F::from_i64(i64::from(b));
                                if i + j < D {
                                    out.coefficients_mut()[i + j] += product;
                                } else {
                                    out.coefficients_mut()[i + j - D] -= product;
                                }
                            }
                        }
                    }
                }
                for sparse_batch in [false, true] {
                    let mut scratch = I8ColumnScratch::new();
                    scratch.sparse_batch = sparse_batch;
                    let mut accs = vec![CyclotomicCrtNtt::zero(); matrix.len()];
                    // Artificial short CRT chunks also check nonzero column
                    // origins and final partial chunks. Each chunk lifts alone.
                    let mut actual = vec![CyclotomicRing::<F, D>::zero(); matrix.len()];
                    for (chunk, planes) in digits.chunks(11).enumerate() {
                        accs.fill(CyclotomicCrtNtt::zero());
                        accumulate_i8_columns::<_, K, D, true>(
                            &mut accs,
                            &views,
                            START + 11 * chunk,
                            planes,
                            &params,
                            &lut,
                            &mut scratch,
                        );
                        for (dst, acc) in actual.iter_mut().zip(&accs) {
                            *dst += acc.to_ring(&params);
                        }
                    }
                    assert_eq!(
                        actual, expected,
                        "length={length}, pattern={pattern}, packed={sparse_batch}"
                    );
                }
            }
        }
    }

    #[test]
    fn sparse_dot_matches_schoolbook_q32() {
        check::<Fp64<4294967197>, _, 64>(CrtNttParamSet::new(Q32_PRIMES));
    }

    #[test]
    fn sparse_dot_matches_schoolbook_q64() {
        check::<Prime64Offset59, _, 128>(CrtNttParamSet::new(Q64_PRIMES));
    }

    #[test]
    fn sparse_dot_matches_schoolbook_q128() {
        check::<Prime128Offset275, _, 64>(CrtNttParamSet::new(q128_primes()));
    }
}
