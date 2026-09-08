//! Sparse high-half products for a small tile of native A rows.
//!
//! Each challenge is decoded once per tile. Source rings are borrowed, and
//! every coefficient addition/subtraction uses the same field operations as
//! the scalar row oracle. No integer accumulator or delayed reduction is used.

use super::{parallel_high_half_accumulate, Challenges, CyclotomicRing};
use akita_error::{checked, AkitaError};
use akita_field::parallel::*;
use akita_field::{CanonicalField, FieldCore};

const ROWS_PER_TILE: usize = 4;

/// Explicit selector supports exact parity tests without changing process state.
/// Production calls the tiled path only under the opt-in Cargo feature; the
/// original scalar loop remains unchanged in `compute_group_a_relation_quotients`.
/// As in that prover kernel, sampled positions must be below the positive
/// native ring dimension. Malformed out-of-domain positions are not supported.
pub(super) fn accumulate_a_high_halves<F, const D: usize>(
    challenges: &Challenges,
    rings: &[CyclotomicRing<F, D>],
    a_rows: usize,
    tiled: bool,
) -> Result<Vec<Vec<F>>, AkitaError>
where
    F: FieldCore + CanonicalField,
{
    if !tiled {
        return (0..a_rows)
            .map(|row| {
                parallel_high_half_accumulate::<F, _, D>(challenges, |i| {
                    rings.get(checked::mul_add(i, a_rows, row)?).copied()
                })
            })
            .collect();
    }
    if D == 0 {
        return Err(AkitaError::InvalidSetup("zero A ring dimension".into()));
    }
    let mut output = Vec::with_capacity(a_rows);
    for first_row in (0..a_rows).step_by(ROWS_PER_TILE) {
        let row_count = (a_rows - first_row).min(ROWS_PER_TILE);
        let width = row_count.checked_mul(D).ok_or(AkitaError::InvalidProof)?;
        let tile = cfg_fold_reduce!(
            0..challenges.len(),
            || vec![F::zero(); width],
            |mut acc: Vec<F>, i: usize| {
                let source: [Option<&CyclotomicRing<F, D>>; ROWS_PER_TILE] =
                    std::array::from_fn(|local| {
                        if local >= row_count {
                            return None;
                        }
                        // As in the original optional ring lookup, absent or
                        // overflowing source positions contribute zero.
                        let row = first_row.checked_add(local)?;
                        rings.get(checked::mul_add(i, a_rows, row)?)
                    });
                let challenge = &challenges.as_slice()[i];
                for (&position, &coefficient) in
                    challenge.positions.iter().zip(challenge.coeffs.iter())
                {
                    let position = position as usize;
                    let rows = &source[..row_count];
                    match coefficient {
                        1 => add_term(&mut acc, rows, position, |dst, value| *dst += value),
                        -1 => add_term(&mut acc, rows, position, |dst, value| *dst -= value),
                        2 => add_term(&mut acc, rows, position, |dst, value| {
                            *dst += value;
                            *dst += value;
                        }),
                        -2 => add_term(&mut acc, rows, position, |dst, value| {
                            *dst -= value;
                            *dst -= value;
                        }),
                        _ => {
                            let scale = F::from_i64(i64::from(coefficient));
                            add_term(&mut acc, rows, position, |dst, value| {
                                *dst += value * scale;
                            });
                        }
                    }
                }
                acc
            },
            |mut left: Vec<F>, right: Vec<F>| {
                for (dst, value) in left.iter_mut().zip(right) {
                    *dst += value;
                }
                left
            }
        );
        output.extend(tile.chunks_exact(D).map(<[F]>::to_vec));
    }
    Ok(output)
}

#[inline(always)]
fn add_term<F: FieldCore, const D: usize>(
    tile: &mut [F],
    sources: &[Option<&CyclotomicRing<F, D>>],
    position: usize,
    add: impl Fn(&mut F, F),
) {
    for (row, source) in tile.chunks_exact_mut(D).zip(sources) {
        if let Some(source) = source {
            for (dst, &value) in row[..position]
                .iter_mut()
                .zip(&source.coefficients()[D - position..])
            {
                add(dst, value);
            }
        }
    }
}

#[cfg(test)]
mod tests;
