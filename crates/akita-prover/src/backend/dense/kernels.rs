//! CpuBackend kernels over dense polynomial views.

use super::views::{DenseBatchView, DenseView};
use super::DensePoly;
use crate::backend::coefficient_packing::{partials_from_position_source, weighted_i8};
use crate::compute::{
    BatchDecomposeFoldOutcome, CommitInnerPlan, CpuBackend, DecomposeFoldBatchPlan,
    DecomposeFoldPlan, OpeningBatchKernel, OpeningFoldKernel, OpeningFoldOutput, OpeningFoldPlan,
    RootCommitKernel, RootPolyMeta, SubringCoefficientPackingBatchKernel,
    SubringCoefficientPackingPartials, SubringCoefficientPackingPlan,
};
use crate::{CommitInnerWitness, DecomposeFoldWitness};
use akita_error::AkitaError;
use akita_field::parallel::*;
use akita_field::{CanonicalField, ExtField, FieldCore};

impl<F, const D: usize> RootCommitKernel<DenseView<'_, F, D>, F, D> for CpuBackend
where
    F: FieldCore + CanonicalField,
{
    fn commit_inner_group(
        &self,
        prepared: &Self::PreparedSetup,
        sources: Vec<DenseView<'_, F, D>>,
        plan: CommitInnerPlan,
    ) -> Result<Vec<CommitInnerWitness<F>>, AkitaError> {
        cfg_into_iter!(sources)
            .map(|source| {
                source
                    .poly
                    .commit_rows::<D>(
                        self,
                        prepared,
                        plan.n_a,
                        plan.num_positions_per_block,
                        plan.num_digits_inner,
                        plan.log_basis_inner,
                    )
                    .map(CommitInnerWitness::from_rows)
            })
            .collect()
    }
}

impl<F, const D: usize> OpeningFoldKernel<DenseView<'_, F, D>, F, D> for CpuBackend
where
    F: FieldCore + CanonicalField,
{
    fn evaluate_and_fold(
        &self,
        _prepared: Option<&Self::PreparedSetup>,
        source: DenseView<'_, F, D>,
        plan: OpeningFoldPlan<'_, F>,
    ) -> Result<OpeningFoldOutput<F, D>, AkitaError> {
        let num_positions_per_block = plan.num_positions_per_block();
        if num_positions_per_block == 0 {
            return Err(AkitaError::InvalidInput(
                "num_positions_per_block must be positive".to_string(),
            ));
        }
        let num_live_blocks = source
            .poly
            .ring_coeffs::<D>()?
            .len()
            .div_ceil(num_positions_per_block);
        plan.validate::<D>(num_live_blocks)?;
        let (eval, folded) = match plan {
            OpeningFoldPlan::Base {
                live_block_weights,
                position_weights,
                num_positions_per_block,
            } => source.poly.evaluate_and_fold::<D>(
                live_block_weights,
                position_weights,
                num_positions_per_block,
            ),
            OpeningFoldPlan::Subfield {
                multipliers,
                num_positions_per_block,
            } => source
                .poly
                .evaluate_and_fold_subfield(multipliers, num_positions_per_block)?,
        };
        Ok(OpeningFoldOutput { eval, folded })
    }

    fn decompose_fold(
        &self,
        _prepared: Option<&Self::PreparedSetup>,
        source: DenseView<'_, F, D>,
        plan: DecomposeFoldPlan<'_>,
    ) -> Result<DecomposeFoldWitness<F>, AkitaError> {
        Ok(source.poly.decompose_fold::<D>(
            plan.challenges,
            plan.num_positions_per_block,
            plan.num_digits,
            plan.log_basis,
        ))
    }
}

impl<F, const D: usize> OpeningBatchKernel<DenseBatchView<'_, F, D>, F, D> for CpuBackend
where
    F: FieldCore + CanonicalField,
{
    fn decompose_fold_batch(
        &self,
        _prepared: Option<&Self::PreparedSetup>,
        _source: DenseBatchView<'_, F, D>,
        _plan: DecomposeFoldBatchPlan<'_>,
    ) -> Result<BatchDecomposeFoldOutcome<F, D>, AkitaError> {
        Ok(BatchDecomposeFoldOutcome::FallbackPerPoly)
    }
}

impl<F, E, const D: usize> SubringCoefficientPackingBatchKernel<DenseBatchView<'_, F, D>, F, E, D>
    for CpuBackend
where
    F: FieldCore + CanonicalField,
    E: ExtField<F> + akita_types::FpExtEncoding<F>,
{
    fn coefficient_packing_partials_batch(
        &self,
        _prepared: Option<&Self::PreparedSetup>,
        source: DenseBatchView<'_, F, D>,
        plan: SubringCoefficientPackingPlan<'_, E>,
    ) -> Result<Vec<SubringCoefficientPackingPartials<F>>, AkitaError> {
        // Set once by the application before starting proving threads. Reading
        // once per batch keeps this experimental ablation out of the inner loop.
        let small_packing =
            std::env::var_os("AERIE_CACHED_SMALL_PACKING").is_some_and(|value| value == "1");
        source
            .polys
            .iter()
            .map(|poly| {
                let coordinates = poly.coefficient_packing_partials::<E, D>(plan, small_packing)?;
                SubringCoefficientPackingPartials::new(
                    plan.point.geometry(),
                    plan.point.num_live_blocks(),
                    coordinates,
                )
            })
            .collect()
    }
}

impl<F: FieldCore + CanonicalField> DensePoly<F> {
    pub(super) fn coefficient_packing_partials<E, const D: usize>(
        &self,
        plan: SubringCoefficientPackingPlan<'_, E>,
        small_packing: bool,
    ) -> Result<Vec<F>, AkitaError>
    where
        E: ExtField<F> + akita_types::FpExtEncoding<F>,
    {
        let rings = self.ring_coeffs::<D>()?;
        // Dense roots authenticate the complete Boolean hypercube, so every
        // stored ring is live. Exact prefixes belong to recursive witnesses.
        if rings.len() != plan.point.num_live_positions() {
            return Err(AkitaError::InvalidSize {
                expected: plan.point.num_live_positions(),
                actual: rings.len(),
            });
        }
        if let Some(small) = small_packing
            .then(|| self.small_i8_ring_coeffs::<D>())
            .flatten()
        {
            // The existing exact cache includes the same physical zero padding
            // as `rings`. Geometry is checked above before selecting the cache.
            partials_from_position_source::<F, E, i8, D>(
                plan,
                RootPolyMeta::<F>::num_vars(self),
                |position| small.get(position).ok_or(AkitaError::InvalidProof),
                |weight, _, _, coefficient| weighted_i8::<F, E>(weight, coefficient),
            )
        } else {
            partials_from_position_source::<F, E, F, D>(
                plan,
                RootPolyMeta::<F>::num_vars(self),
                |position| {
                    rings
                        .get(position)
                        .map(|ring| ring.coefficients())
                        .ok_or(AkitaError::InvalidProof)
                },
                |weight, _, _, coefficient| weight.mul_base(coefficient),
            )
        }
    }
}
