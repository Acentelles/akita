//! Dense polynomial inner commit.

use super::poly::DensePoly;
use crate::compute::{CpuBackend, CpuPreparedSetup, DenseCommitInput};
use akita_algebra::CyclotomicRing;
use akita_field::{AkitaError, CanonicalField, FieldCore};

impl<F> DensePoly<F>
where
    F: FieldCore + CanonicalField,
{
    pub(super) fn commit_rows<const D: usize>(
        &self,
        backend: &CpuBackend,
        prepared: &CpuPreparedSetup<F>,
        n_a: usize,
        num_positions_per_block: usize,
        num_digits_inner: usize,
        log_basis: u32,
    ) -> Result<Vec<Vec<CyclotomicRing<F, D>>>, AkitaError> {
        let coeffs = self.ring_coeffs::<D>()?;
        let n = coeffs.len();
        let num_live_blocks = n.div_ceil(num_positions_per_block);

        // Blocks made entirely of structurally-zero rings commit to zero
        // rows: under a declared live extent the matvec runs on the live
        // blocks only and the dead blocks' rows are emitted as zeros, so
        // the output shape (and every value) is identical.
        let block_mask: Option<Vec<bool>> = self.live_extent().map(|extent| {
            (0..num_live_blocks)
                .map(|block_idx| {
                    let start = block_idx * num_positions_per_block;
                    let end = (start + num_positions_per_block).min(n);
                    (start..end).any(|ring| extent.ring_is_live(ring))
                })
                .collect()
        });
        let scatter = |live_rows: Vec<Vec<CyclotomicRing<F, D>>>,
                       mask: &[bool]|
         -> Vec<Vec<CyclotomicRing<F, D>>> {
            let mut live_iter = live_rows.into_iter();
            mask.iter()
                .map(|&live| {
                    if live {
                        live_iter
                            .next()
                            .expect("one committed block per live mask entry")
                    } else {
                        vec![CyclotomicRing::<F, D>::zero(); n_a]
                    }
                })
                .collect()
        };

        if let Some(digit_planes) = self.digit_planes_for::<D>(num_digits_inner, log_basis) {
            let digit_block_slices =
                digit_block_slices(digit_planes, n, num_positions_per_block, num_digits_inner);
            return match &block_mask {
                Some(mask) if mask.iter().any(|&live| !live) => {
                    let live_slices = digit_block_slices
                        .into_iter()
                        .zip(mask)
                        .filter_map(|(slice, &live)| live.then_some(slice))
                        .collect();
                    let live_rows = backend.dense_commit_rows(
                        prepared,
                        n_a,
                        DenseCommitInput::CachedDigits {
                            digit_block_slices: live_slices,
                            log_basis_inner: log_basis,
                        },
                    )?;
                    Ok(scatter(live_rows, mask))
                }
                _ => backend.dense_commit_rows(
                    prepared,
                    n_a,
                    DenseCommitInput::CachedDigits {
                        digit_block_slices,
                        log_basis_inner: log_basis,
                    },
                ),
            };
        }

        let block_slices: Vec<&[CyclotomicRing<F, D>]> = (0..num_live_blocks)
            .map(|i| {
                let start = i * num_positions_per_block;
                if start >= n {
                    &[] as &[CyclotomicRing<F, D>]
                } else {
                    &coeffs[start..(start + num_positions_per_block).min(n)]
                }
            })
            .collect();

        match &block_mask {
            Some(mask) if mask.iter().any(|&live| !live) => {
                let live_slices = block_slices
                    .into_iter()
                    .zip(mask)
                    .filter_map(|(slice, &live)| live.then_some(slice))
                    .collect();
                let live_rows = backend.dense_commit_rows(
                    prepared,
                    n_a,
                    DenseCommitInput::CoeffBlocks {
                        block_slices: live_slices,
                        num_digits_inner,
                        log_basis_inner: log_basis,
                    },
                )?;
                Ok(scatter(live_rows, mask))
            }
            _ => backend.dense_commit_rows(
                prepared,
                n_a,
                DenseCommitInput::CoeffBlocks {
                    block_slices,
                    num_digits_inner,
                    log_basis_inner: log_basis,
                },
            ),
        }
    }
}

pub(super) fn digit_block_slices<const D: usize>(
    digit_planes: &[[i8; D]],
    num_rings: usize,
    num_positions_per_block: usize,
    num_digits: usize,
) -> Vec<&[[i8; D]]> {
    let num_live_blocks = num_rings.div_ceil(num_positions_per_block);
    (0..num_live_blocks)
        .map(|block_idx| {
            let ring_start = block_idx * num_positions_per_block;
            let ring_end = (ring_start + num_positions_per_block).min(num_rings);
            let digit_start = ring_start * num_digits;
            let digit_end = ring_end * num_digits;
            &digit_planes[digit_start..digit_end]
        })
        .collect()
}
