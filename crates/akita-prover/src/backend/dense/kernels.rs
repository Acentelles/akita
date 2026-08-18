//! CpuBackend kernels over dense polynomial views.

use super::poly::DensePoly;
use super::views::{DenseBatchView, DenseView};
use crate::backend::RootTensorProjectionPoly;
use crate::compute::{
    BatchDecomposeFoldOutcome, CommitInnerPlan, CpuBackend, DecomposeFoldBatchPlan,
    DecomposeFoldPlan, OpeningBatchKernel, OpeningFoldKernel, OpeningFoldOutput, OpeningFoldPlan,
    RootCommitKernel, TensorPackedWitness, TensorProjectionBatchKernel, TensorProjectionKernel,
};
use crate::protocol::extension_opening_reduction::SparseExtensionOpeningWitness;
use crate::{CommitInnerWitness, DecomposeFoldWitness};
use akita_field::parallel::*;
use akita_field::{
    AkitaError, CanonicalField, ExtField, FieldCore, FromPrimitiveInt, MulBaseUnreduced,
};
use akita_types::FpExtEncoding;

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
        let commit_one = |source: DenseView<'_, F, D>| {
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
        };
        // Commit the first source before fanning out. Every source of one
        // group requests the same shared-NTT cache key (the group layout is
        // uniform), so the first commit builds the slot from the calling
        // thread, outside the worker pool, and the parallel remainder only
        // ever reads it. Fanning all sources out cold can deadlock: sibling
        // workers park on the slot's `OnceLock` while the builder's nested
        // parallel work has no free worker left to run it.
        let mut sources = sources.into_iter();
        let Some(first) = sources.next() else {
            return Ok(Vec::new());
        };
        let first = commit_one(first)?;
        let rest = cfg_into_iter!(sources.collect::<Vec<_>>())
            .map(commit_one)
            .collect::<Result<Vec<_>, AkitaError>>()?;
        let mut witnesses = Vec::with_capacity(rest.len() + 1);
        witnesses.push(first);
        witnesses.extend(rest);
        Ok(witnesses)
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

impl<F, E, const D: usize> TensorProjectionKernel<DenseView<'_, F, D>, F, E, D> for CpuBackend
where
    F: FieldCore + CanonicalField + FromPrimitiveInt,
    E: ExtField<F>,
{
    fn column_partials(
        &self,
        _prepared: Option<&Self::PreparedSetup>,
        source: DenseView<'_, F, D>,
        logical_point: &[E],
    ) -> Result<Vec<E>, AkitaError>
    where
        E: MulBaseUnreduced<F>,
    {
        source
            .poly
            .tensor_extension_column_partials::<E, D>(logical_point)
    }

    fn packed_witness(
        &self,
        _prepared: Option<&Self::PreparedSetup>,
        source: DenseView<'_, F, D>,
    ) -> Result<TensorPackedWitness<E>, AkitaError> {
        Ok(TensorPackedWitness::Dense(
            source.poly.tensor_packed_extension_evals::<E, D>()?,
        ))
    }

    fn root_projection(
        &self,
        _prepared: Option<&Self::PreparedSetup>,
        source: DenseView<'_, F, D>,
    ) -> Result<RootTensorProjectionPoly<F>, AkitaError>
    where
        E: FpExtEncoding<F>,
    {
        source.poly.tensor_packed_extension_root_poly::<E, D>()
    }
}

impl<F, E, const D: usize> TensorProjectionBatchKernel<DenseBatchView<'_, F, D>, F, E, D>
    for CpuBackend
where
    F: FieldCore + CanonicalField,
    E: ExtField<F>,
{
    fn column_partials_batch(
        &self,
        _prepared: Option<&Self::PreparedSetup>,
        source: DenseBatchView<'_, F, D>,
        logical_point: &[E],
    ) -> Result<Vec<Vec<E>>, AkitaError>
    where
        E: MulBaseUnreduced<F>,
    {
        DensePoly::tensor_extension_column_partials_batch::<E, D>(source.polys, logical_point)
    }

    fn sparse_linear_combination(
        &self,
        _prepared: Option<&Self::PreparedSetup>,
        source: DenseBatchView<'_, F, D>,
        coeffs: &[E],
    ) -> Result<Option<SparseExtensionOpeningWitness<E>>, AkitaError> {
        DensePoly::tensor_packed_extension_sparse_linear_combination(source.polys, coeffs)
    }
}
