use std::array::from_fn;
use std::sync::Arc;

use akita_algebra::ring::cyclotomic::decompose_centering_threshold;
use akita_algebra::CyclotomicRing;
use akita_field::AkitaError;
use akita_field::{CanonicalField, ExtField, FieldCore, FromPrimitiveInt, MulBaseUnreduced};
use akita_types::{
    AkitaExpandedSetup, CleartextWitnessProof, FpExtEncoding, RingVec, SetupPrefixSlot,
};

use crate::backend::poly_helpers::{
    balanced_ring_decompose_fold_partitioned, build_decompose_fold_witness, DecomposeParams,
};
use crate::backend::{RecursiveWitnessFlat, SuffixWitnessBatchView, SuffixWitnessView};
use crate::compute::{
    BatchDecomposeFoldOutcome, CpuBackend, DecomposeFoldBatchPlan, DecomposeFoldPlan,
    DirectRootWitnessSource, OpeningBatchKernel, OpeningFoldKernel, OpeningFoldOutput,
    OpeningFoldPlan, RootOpeningSource, RootPolyMeta, RootPolyShape, RootTensorSource,
    TensorPackedWitness, TensorProjectionBatchKernel, TensorProjectionKernel,
};
use crate::protocol::extension_opening_reduction::SparseExtensionOpeningWitness;
use crate::RootTensorProjectionPoly;

#[doc(hidden)]
#[derive(Clone)]
pub enum RecursiveFoldSource<F: FieldCore> {
    SetupPrefix {
        expanded: Arc<AkitaExpandedSetup<F>>,
        slot: Arc<SetupPrefixSlot<F>>,
        /// Committed representation of the padded flat prefix: at `k = 1` the
        /// raw stream; at `k > 1` the psi-packed transform (the
        /// committed-transformed-witness discipline of the extension-opening
        /// cutover). The fold/commit kernels consume this; the tensor (EOR)
        /// kernels consume the raw stream.
        committed_evals: Arc<Vec<F>>,
    },
    Witness(Arc<RecursiveWitnessFlat>),
}

impl<F: FieldCore> RecursiveFoldSource<F> {
    pub(crate) fn setup_prefix(
        expanded: Arc<AkitaExpandedSetup<F>>,
        slot: Arc<SetupPrefixSlot<F>>,
        committed_evals: Arc<Vec<F>>,
    ) -> Self {
        Self::SetupPrefix {
            expanded,
            slot,
            committed_evals,
        }
    }

    pub(crate) fn witness(witness: Arc<RecursiveWitnessFlat>) -> Self {
        Self::Witness(witness)
    }
}

#[doc(hidden)]
#[derive(Clone, Copy)]
pub enum RecursiveFoldView<'a, F: FieldCore, const D: usize> {
    SetupPrefix {
        expanded: &'a AkitaExpandedSetup<F>,
        slot: &'a SetupPrefixSlot<F>,
        committed_evals: &'a [F],
    },
    Witness(SuffixWitnessView<'a, F, D>),
}

#[doc(hidden)]
#[derive(Clone, Copy)]
pub struct RecursiveFoldBatchView<'a, F: FieldCore, const D: usize> {
    polys: &'a [&'a RecursiveFoldSource<F>],
}

impl<F: FieldCore> RootPolyMeta<F> for RecursiveFoldSource<F> {
    fn num_ring_elems(&self) -> usize {
        match self {
            Self::SetupPrefix { slot, .. } => slot.id.n_prefix().unwrap_or(1),
            Self::Witness(witness) => RootPolyMeta::<F>::num_ring_elems(witness.as_ref()),
        }
    }

    fn num_vars(&self) -> usize {
        match self {
            Self::SetupPrefix { slot, .. } => {
                slot.id.n_prefix().unwrap_or(1).trailing_zeros() as usize
            }
            Self::Witness(witness) => RootPolyMeta::<F>::num_vars(witness.as_ref()),
        }
    }
}

impl<F: FieldCore, const D: usize> RootPolyShape<F, D> for RecursiveFoldSource<F> {
    fn num_ring_elems(&self) -> usize {
        match self {
            Self::SetupPrefix { slot, .. } => slot.id.n_prefix().map_or(1, |n| n / D),
            Self::Witness(witness) => RootPolyShape::<F, D>::num_ring_elems(witness.as_ref()),
        }
    }

    fn num_vars(&self) -> usize {
        RootPolyMeta::<F>::num_vars(self)
    }
}

impl<F: FieldCore, const D: usize> RootOpeningSource<F, D> for RecursiveFoldSource<F> {
    type OpeningView<'v>
        = RecursiveFoldView<'v, F, D>
    where
        Self: 'v;

    type OpeningBatchView<'v>
        = RecursiveFoldBatchView<'v, F, D>
    where
        Self: 'v;

    fn opening_view(&self) -> Result<Self::OpeningView<'_>, AkitaError> {
        match self {
            Self::SetupPrefix {
                expanded,
                slot,
                committed_evals,
            } => Ok(RecursiveFoldView::SetupPrefix {
                expanded: expanded.as_ref(),
                slot: slot.as_ref(),
                committed_evals: committed_evals.as_slice(),
            }),
            Self::Witness(witness) => {
                Ok(RecursiveFoldView::Witness(witness.as_ref().view::<F, D>()?))
            }
        }
    }

    fn opening_batch<'v>(polys: &'v [&'v Self]) -> Result<Self::OpeningBatchView<'v>, AkitaError> {
        Ok(RecursiveFoldBatchView { polys })
    }
}

impl<F: FieldCore, const D: usize> RootTensorSource<F, D> for RecursiveFoldSource<F> {
    type TensorView<'v>
        = RecursiveFoldView<'v, F, D>
    where
        Self: 'v;

    type TensorBatchView<'v>
        = RecursiveFoldBatchView<'v, F, D>
    where
        Self: 'v;

    fn tensor_view(&self) -> Result<Self::TensorView<'_>, AkitaError> {
        self.opening_view()
    }

    fn tensor_batch<'v>(polys: &'v [&'v Self]) -> Result<Self::TensorBatchView<'v>, AkitaError> {
        Ok(RecursiveFoldBatchView { polys })
    }
}

impl<F: FieldCore + CanonicalField, const D: usize> DirectRootWitnessSource<F, D>
    for RecursiveFoldSource<F>
{
    fn direct_root_witness(&self) -> Result<CleartextWitnessProof<F>, AkitaError> {
        match self {
            Self::SetupPrefix { expanded, slot, .. } => Ok(CleartextWitnessProof::FieldElements(
                RingVec::from_coeffs(setup_prefix_field_evals(expanded.as_ref(), slot.as_ref())?),
            )),
            Self::Witness(witness) => {
                DirectRootWitnessSource::<F, D>::direct_root_witness(witness.as_ref())
            }
        }
    }
}

fn setup_prefix_field_evals<F: FieldCore>(
    expanded: &AkitaExpandedSetup<F>,
    slot: &SetupPrefixSlot<F>,
) -> Result<Vec<F>, AkitaError> {
    let n_prefix = slot.id.n_prefix()?;
    let fields = expanded.shared_matrix().as_field_slice();
    if slot.natural_len > fields.len() || slot.natural_len > n_prefix {
        return Err(AkitaError::InvalidSetup(
            "setup-prefix slot exceeds shared setup matrix".to_string(),
        ));
    }
    let mut evals = vec![F::zero(); n_prefix];
    evals[..slot.natural_len].copy_from_slice(&fields[..slot.natural_len]);
    Ok(evals)
}

fn committed_evals_rings<F: FieldCore, const D: usize>(
    committed_evals: &[F],
) -> Result<Vec<CyclotomicRing<F, D>>, AkitaError> {
    if !committed_evals.len().is_multiple_of(D) {
        return Err(AkitaError::InvalidSize {
            expected: committed_evals.len().next_multiple_of(D),
            actual: committed_evals.len(),
        });
    }
    Ok(committed_evals
        .chunks_exact(D)
        .map(|chunk| CyclotomicRing::from_coefficients(from_fn(|idx| chunk[idx])))
        .collect())
}

/// Committed representation of the padded flat setup prefix at extension
/// degree `k = [E:F]`: the raw stream at `k = 1`; at `k > 1` the psi-packed
/// transform (adjacent-pack into `E`, chunk `D/k`, `psi`-embed each chunk),
/// mirroring `DensePoly::tensor_packed_extension_poly` so the fold opens the
/// committed transformed polynomial (`specs/eor-setup-prefix-absorption.md`).
pub(crate) fn setup_prefix_committed_field_evals<F, E, const D: usize>(
    expanded: &AkitaExpandedSetup<F>,
    slot: &SetupPrefixSlot<F>,
) -> Result<Vec<F>, AkitaError>
where
    F: FieldCore + CanonicalField + FromPrimitiveInt,
    E: FpExtEncoding<F>,
{
    let raw = setup_prefix_field_evals(expanded, slot)?;
    transform_committed_field_evals::<F, E, D>(raw)
}

/// The psi-pack transform on a padded flat stream (identity at `k = 1`).
pub(crate) fn transform_committed_field_evals<F, E, const D: usize>(
    raw: Vec<F>,
) -> Result<Vec<F>, AkitaError>
where
    F: FieldCore + CanonicalField + FromPrimitiveInt,
    E: FpExtEncoding<F>,
{
    let k = <E as ExtField<F>>::EXT_DEGREE;
    if k == 1 {
        return Ok(raw);
    }
    let num_vars = akita_types::num_rounds_from_table_len(raw.len())?;
    let packed = akita_types::tensor_packed_witness_evals::<F, E>(num_vars, &raw)?;
    let packed_len = D / k;
    if packed_len == 0 || !packed.len().is_multiple_of(packed_len) {
        return Err(AkitaError::InvalidInput(
            "setup prefix does not fill whole psi-packed ring slots".to_string(),
        ));
    }
    let mut out = Vec::with_capacity(raw.len());
    for chunk in packed.chunks(packed_len) {
        let ring = akita_types::embed_ring_subfield_vector::<F, E, D>(
            chunk,
            AkitaError::InvalidInput(
                "setup prefix does not encode in the ring-subfield basis".to_string(),
            ),
        )?;
        out.extend_from_slice(ring.coefficients());
    }
    Ok(out)
}

fn fold_setup_prefix_blocks<F: FieldCore, const D: usize>(
    coeffs: &[CyclotomicRing<F, D>],
    scalars: &[F],
    block_len: usize,
) -> Vec<CyclotomicRing<F, D>> {
    (0..coeffs.len().div_ceil(block_len))
        .map(|block_idx| {
            let start = block_idx * block_len;
            let end = (start + block_len).min(coeffs.len());
            let mut acc = CyclotomicRing::<F, D>::zero();
            for (ring, scalar) in coeffs[start..end].iter().zip(scalars.iter()) {
                acc += ring.scale(scalar);
            }
            acc
        })
        .collect()
}

fn fold_setup_prefix_blocks_ring<F: FieldCore + CanonicalField, const D: usize>(
    coeffs: &[CyclotomicRing<F, D>],
    scalars: &[CyclotomicRing<F, D>],
    block_len: usize,
) -> Vec<CyclotomicRing<F, D>> {
    (0..coeffs.len().div_ceil(block_len))
        .map(|block_idx| {
            let start = block_idx * block_len;
            let end = (start + block_len).min(coeffs.len());
            let mut acc = CyclotomicRing::<F, D>::zero();
            for (ring, scalar) in coeffs[start..end].iter().zip(scalars.iter()) {
                ring.mul_accumulate_sparse_rhs_into(scalar, &mut acc);
            }
            acc
        })
        .collect()
}

fn setup_prefix_evaluate_and_fold<F: FieldCore + CanonicalField, const D: usize>(
    committed_evals: &[F],
    plan: OpeningFoldPlan<'_, F, D>,
) -> Result<OpeningFoldOutput<F, D>, AkitaError> {
    let coeffs = committed_evals_rings::<F, D>(committed_evals)?;
    match plan {
        OpeningFoldPlan::Base {
            eval_outer_scalars,
            fold_scalars,
            block_len,
        } => {
            let folded = fold_setup_prefix_blocks(&coeffs, fold_scalars, block_len);
            let (eval, folded) = crate::backend::poly_helpers::fused_evaluate_and_fold_base(
                folded,
                eval_outer_scalars,
            );
            Ok(OpeningFoldOutput { eval, folded })
        }
        OpeningFoldPlan::Ring {
            eval_outer_scalars,
            fold_scalars,
            block_len,
        } => {
            let folded = fold_setup_prefix_blocks_ring(&coeffs, fold_scalars, block_len);
            let (eval, folded) = crate::backend::poly_helpers::fused_evaluate_and_fold_ring(
                folded,
                eval_outer_scalars,
            );
            Ok(OpeningFoldOutput { eval, folded })
        }
    }
}

fn setup_prefix_decompose_fold<F: CanonicalField, const D: usize>(
    committed_evals: &[F],
    plan: DecomposeFoldPlan<'_>,
) -> Result<crate::DecomposeFoldWitness<F>, AkitaError> {
    let coeffs = committed_evals_rings::<F, D>(committed_evals)?;
    let q = (-F::one()).to_canonical_u128() + 1;
    let threshold = decompose_centering_threshold(plan.num_digits, plan.log_basis, q);
    let params = DecomposeParams {
        threshold,
        q,
        mask: (1i128 << plan.log_basis) - 1,
        half_b: 1i128 << (plan.log_basis - 1),
        b_val: 1i128 << plan.log_basis,
        log_basis: plan.log_basis,
        overflow_possible: q.saturating_sub(threshold) > i128::MAX as u128,
    };
    let centered = balanced_ring_decompose_fold_partitioned::<F, D>(
        &coeffs,
        plan.challenges,
        plan.block_len,
        plan.num_digits,
        &params,
    );
    Ok(build_decompose_fold_witness::<F, D>(centered, q))
}

impl<F, const D: usize> OpeningFoldKernel<RecursiveFoldView<'_, F, D>, F, D> for CpuBackend
where
    F: FieldCore + CanonicalField,
{
    fn evaluate_and_fold(
        &self,
        prepared: Option<&Self::PreparedSetup>,
        source: RecursiveFoldView<'_, F, D>,
        plan: OpeningFoldPlan<'_, F, D>,
    ) -> Result<OpeningFoldOutput<F, D>, AkitaError> {
        match source {
            RecursiveFoldView::SetupPrefix {
                committed_evals, ..
            } => setup_prefix_evaluate_and_fold(committed_evals, plan),
            RecursiveFoldView::Witness(view) => <CpuBackend as OpeningFoldKernel<
                SuffixWitnessView<'_, F, D>,
                F,
                D,
            >>::evaluate_and_fold(
                self, prepared, view, plan
            ),
        }
    }

    fn decompose_fold(
        &self,
        prepared: Option<&Self::PreparedSetup>,
        source: RecursiveFoldView<'_, F, D>,
        plan: DecomposeFoldPlan<'_>,
    ) -> Result<crate::DecomposeFoldWitness<F>, AkitaError> {
        match source {
            RecursiveFoldView::SetupPrefix {
                committed_evals, ..
            } => setup_prefix_decompose_fold::<F, D>(committed_evals, plan),
            RecursiveFoldView::Witness(view) => {
                <CpuBackend as OpeningFoldKernel<SuffixWitnessView<'_, F, D>, F, D>>::decompose_fold(
                    self, prepared, view, plan,
                )
            }
        }
    }
}

impl<F, const D: usize> OpeningBatchKernel<RecursiveFoldBatchView<'_, F, D>, F, D> for CpuBackend
where
    F: FieldCore + CanonicalField,
{
    fn decompose_fold_batch(
        &self,
        _prepared: Option<&Self::PreparedSetup>,
        source: RecursiveFoldBatchView<'_, F, D>,
        _plan: DecomposeFoldBatchPlan<'_>,
    ) -> Result<BatchDecomposeFoldOutcome<F, D>, AkitaError> {
        let _ = source.polys;
        Ok(BatchDecomposeFoldOutcome::FallbackPerPoly)
    }
}

fn setup_prefix_extension_tensor_unsupported<T>() -> Result<T, AkitaError> {
    Err(AkitaError::InvalidSetup(
        "setup-prefix grouped suffix does not support extension tensor projection".to_string(),
    ))
}

/// Extract the homogeneous witness view of a recursive fold batch, or `None`
/// when the batch contains a setup-prefix source (heterogeneous batches take
/// the per-poly path).
fn recursive_fold_batch_witnesses<'a, F: FieldCore, const D: usize>(
    source: RecursiveFoldBatchView<'a, F, D>,
) -> Option<Vec<&'a RecursiveWitnessFlat>> {
    let mut witnesses = Vec::with_capacity(source.polys.len());
    for poly in source.polys {
        match poly {
            RecursiveFoldSource::Witness(witness) => witnesses.push(witness.as_ref()),
            RecursiveFoldSource::SetupPrefix { .. } => return None,
        }
    }
    Some(witnesses)
}

impl<F, E, const D: usize> TensorProjectionKernel<RecursiveFoldView<'_, F, D>, F, E, D>
    for CpuBackend
where
    F: FieldCore + CanonicalField + FromPrimitiveInt,
    E: ExtField<F>,
{
    fn column_partials(
        &self,
        prepared: Option<&Self::PreparedSetup>,
        source: RecursiveFoldView<'_, F, D>,
        logical_point: &[E],
    ) -> Result<Vec<E>, AkitaError>
    where
        E: MulBaseUnreduced<F>,
    {
        match source {
            RecursiveFoldView::SetupPrefix { expanded, slot, .. } => {
                // The claim object is the plain flat coefficient stream
                // (zero-padded to `n_prefix`); the EOR reduces the raw
                // adjacent-packed polynomial, while the fold opens the
                // committed transformed representation
                // (`specs/eor-setup-prefix-absorption.md`).
                let evals = setup_prefix_field_evals(expanded, slot)?;
                let num_vars = akita_types::num_rounds_from_table_len(evals.len())?;
                akita_types::tensor_column_partials_from_base_evals::<F, E>(
                    num_vars,
                    &evals,
                    logical_point,
                )
            }
            RecursiveFoldView::Witness(view) => <CpuBackend as TensorProjectionKernel<
                SuffixWitnessView<'_, F, D>,
                F,
                E,
                D,
            >>::column_partials(
                self, prepared, view, logical_point
            ),
        }
    }

    fn packed_witness(
        &self,
        prepared: Option<&Self::PreparedSetup>,
        source: RecursiveFoldView<'_, F, D>,
    ) -> Result<TensorPackedWitness<E>, AkitaError> {
        match source {
            RecursiveFoldView::SetupPrefix { expanded, slot, .. } => {
                let evals = setup_prefix_field_evals(expanded, slot)?;
                let num_vars = akita_types::num_rounds_from_table_len(evals.len())?;
                Ok(TensorPackedWitness::Dense(
                    akita_types::tensor_packed_witness_evals::<F, E>(num_vars, &evals)?,
                ))
            }
            RecursiveFoldView::Witness(view) => <CpuBackend as TensorProjectionKernel<
                SuffixWitnessView<'_, F, D>,
                F,
                E,
                D,
            >>::packed_witness(self, prepared, view),
        }
    }

    fn root_projection(
        &self,
        prepared: Option<&Self::PreparedSetup>,
        source: RecursiveFoldView<'_, F, D>,
    ) -> Result<RootTensorProjectionPoly<F>, AkitaError>
    where
        E: FpExtEncoding<F>,
    {
        match source {
            RecursiveFoldView::SetupPrefix { .. } => setup_prefix_extension_tensor_unsupported(),
            RecursiveFoldView::Witness(view) => <CpuBackend as TensorProjectionKernel<
                SuffixWitnessView<'_, F, D>,
                F,
                E,
                D,
            >>::root_projection(
                self, prepared, view
            ),
        }
    }
}

impl<F, E, const D: usize> TensorProjectionBatchKernel<RecursiveFoldBatchView<'_, F, D>, F, E, D>
    for CpuBackend
where
    F: FieldCore + CanonicalField + FromPrimitiveInt,
    E: ExtField<F>,
{
    fn column_partials_batch(
        &self,
        prepared: Option<&Self::PreparedSetup>,
        source: RecursiveFoldBatchView<'_, F, D>,
        logical_point: &[E],
    ) -> Result<Vec<Vec<E>>, AkitaError>
    where
        E: MulBaseUnreduced<F>,
    {
        if let Some(witnesses) = recursive_fold_batch_witnesses(source) {
            let batch = <RecursiveWitnessFlat as RootTensorSource<F, D>>::tensor_batch(&witnesses)?;
            return <CpuBackend as TensorProjectionBatchKernel<
                SuffixWitnessBatchView<'_, F, D>,
                F,
                E,
                D,
            >>::column_partials_batch(self, prepared, batch, logical_point);
        }
        // Heterogeneous (setup-prefix-bearing) batch: per-poly partials.
        source
            .polys
            .iter()
            .map(|poly| {
                <CpuBackend as TensorProjectionKernel<RecursiveFoldView<'_, F, D>, F, E, D>>::column_partials(
                    self,
                    prepared,
                    <RecursiveFoldSource<F> as RootTensorSource<F, D>>::tensor_view(poly)?,
                    logical_point,
                )
            })
            .collect()
    }

    fn sparse_linear_combination(
        &self,
        prepared: Option<&Self::PreparedSetup>,
        source: RecursiveFoldBatchView<'_, F, D>,
        coeffs: &[E],
    ) -> Result<Option<SparseExtensionOpeningWitness<E>>, AkitaError> {
        // Setup-prefix-bearing spans take the dense fallback.
        let Some(witnesses) = recursive_fold_batch_witnesses(source) else {
            return Ok(None);
        };
        let batch = <RecursiveWitnessFlat as RootTensorSource<F, D>>::tensor_batch(&witnesses)?;
        <CpuBackend as TensorProjectionBatchKernel<SuffixWitnessBatchView<'_, F, D>, F, E, D>>::sparse_linear_combination(
            self, prepared, batch, coeffs,
        )
    }
}

#[cfg(test)]
mod tensor_equivalence_tests {
    use super::*;
    use akita_algebra::poly::multilinear_eval;
    use akita_challenges::SparseChallengeConfig;
    use akita_field::{Ext2, LiftBase, Prime64Offset59};
    use akita_types::{
        derive_tensor_extension_opening_claim_from_partials, setup_prefix_precommitted_params,
        setup_prefix_slot_id, AkitaSetupSeed, DigitBlocks, GroupBoundPolicy, LevelParams,
        SetupPrefixPublicCommitment, SisModulusFamily,
    };

    type F = Prime64Offset59;
    type E = Ext2<F>;
    const D: usize = 128;

    fn test_slot(natural_len: usize, n_prefix: usize) -> SetupPrefixSlot<F> {
        let lp = LevelParams::params_only(
            SisModulusFamily::Q64,
            D,
            3,
            2,
            3,
            2,
            SparseChallengeConfig::pm1_only(3),
        )
        .with_decomp(2, 3, 2, 2, 3)
        .expect("level params");
        let commitment_params = setup_prefix_precommitted_params(
            &lp,
            n_prefix,
            GroupBoundPolicy {
                log_commit_bound: 1,
                onehot_chunk_size: 1,
                basis_range: (1, 8),
            },
        )
        .expect("prefix params");
        let id = setup_prefix_slot_id(D, natural_len, commitment_params);
        let decomposed = DigitBlocks::from_blocks(vec![Vec::new()], D).expect("digit blocks");
        SetupPrefixSlot {
            id,
            natural_len,
            padded_len: n_prefix,
            commitment: SetupPrefixPublicCommitment {
                rows: vec![RingVec::from_coeffs(vec![F::zero(); D])],
            },
            hint: akita_types::AkitaCommitmentHint::singleton(decomposed),
        }
    }

    fn test_expanded(min_field_len: usize) -> AkitaExpandedSetup<F> {
        let ring_slots = min_field_len.div_ceil(D).max(1);
        let seed = AkitaSetupSeed {
            max_num_vars: 16,
            max_num_batched_polys: 1,
            gen_ring_dim: D,
            max_setup_len: ring_slots,
            public_matrix_seed: [7u8; 32],
        };
        let shared =
            akita_types::derive_public_matrix_flat::<F, D>(ring_slots, &seed.public_matrix_seed);
        AkitaExpandedSetup::from_trusted_seed_derived_parts_unchecked(seed, shared)
    }

    fn random_ext_point(len: usize, seed: u64) -> Vec<E> {
        (0..len)
            .map(|idx| {
                E::from_base_slice(&[
                    F::from_u64(
                        seed.wrapping_mul(6364136223846793005)
                            .wrapping_add(idx as u64),
                    ),
                    F::from_u64(
                        seed.wrapping_mul(1442695040888963407)
                            .wrapping_add(97 + idx as u64),
                    ),
                ])
            })
            .collect()
    }

    /// Phase-1 equivalence oracle (`specs/eor-setup-prefix-absorption.md`):
    /// the committed prefix's flat data opened via the `k = 2` claim path
    /// (tensor column partials + head combination, the object the shared EOR
    /// reduces) equals the plain `k = 1`-style extension MLE of the same flat
    /// zero-padded stream. The commitment binds this one object at both
    /// extension degrees; the psi packing is a claim-side relabeling.
    #[test]
    fn setup_prefix_k2_claim_path_matches_k1_direct_mle() {
        let natural_len = 900usize;
        let n_prefix = 1024usize;
        let expanded = Arc::new(test_expanded(natural_len));
        let slot = Arc::new(test_slot(natural_len, n_prefix));
        let num_vars = n_prefix.trailing_zeros() as usize;
        let point = random_ext_point(num_vars, 0xae2_2026);

        // k = 2 claim path: partials at the point, combined over the packing
        // head — exactly the per-claim object of the shared EOR.
        let committed =
            setup_prefix_committed_field_evals::<F, E, D>(expanded.as_ref(), slot.as_ref())
                .expect("committed evals");
        let source = RecursiveFoldSource::setup_prefix(
            Arc::clone(&expanded),
            Arc::clone(&slot),
            Arc::new(committed),
        );
        let view =
            <RecursiveFoldSource<F> as RootTensorSource<F, D>>::tensor_view(&source).expect("view");
        let partials = <CpuBackend as TensorProjectionKernel<
            RecursiveFoldView<'_, F, D>,
            F,
            E,
            D,
        >>::column_partials(&CpuBackend, None, view, &point)
        .expect("column partials");
        let via_k2 = derive_tensor_extension_opening_claim_from_partials::<F, E>(&point, &partials)
            .expect("k2 claim");

        // k = 1-style opening of the same flat data: direct MLE of the
        // zero-padded stream lifted into `E`.
        let flat = setup_prefix_field_evals(expanded.as_ref(), slot.as_ref()).expect("flat evals");
        assert_eq!(flat.len(), n_prefix);
        assert_eq!(
            &flat[natural_len..],
            &vec![F::zero(); n_prefix - natural_len][..]
        );
        let lifted: Vec<E> = flat.iter().copied().map(E::lift_base).collect();
        let direct = multilinear_eval(&lifted, &point).expect("direct MLE");

        assert_eq!(
            via_k2, direct,
            "k=2 claim path must open the same flat object"
        );

        // Packed-witness identity: the psi-packed table is the same flat
        // stream reinterpreted pairwise (entry `tail*k + head`).
        let view =
            <RecursiveFoldSource<F> as RootTensorSource<F, D>>::tensor_view(&source).expect("view");
        let packed = <CpuBackend as TensorProjectionKernel<
            RecursiveFoldView<'_, F, D>,
            F,
            E,
            D,
        >>::packed_witness(&CpuBackend, None, view)
        .expect("packed witness");
        let TensorPackedWitness::Dense(packed) = packed else {
            panic!("setup-prefix packed witness must be dense");
        };
        assert_eq!(packed.len(), n_prefix / 2);
        for (tail, value) in packed.iter().enumerate() {
            assert_eq!(*value, E::from_base_slice(&flat[2 * tail..2 * tail + 2]));
        }
    }
}
