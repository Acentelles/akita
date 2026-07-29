use super::*;
use crate::compute::{
    ComputeBackendSetup, RootTensorSource, TensorPackedWitness, TensorProjectionBatchKernel,
    TensorProjectionKernel,
};
use crate::protocol::extension_opening_reduction::SparseExtensionOpeningWitness;
use akita_types::OpeningClaimsLayout;

pub(in crate::protocol::core) struct PreparedExtensionOpeningReduction<E: FieldCore> {
    pub(in crate::protocol::core) proof_partials: Vec<E>,
    pub(in crate::protocol::core) row_coefficients: Vec<E>,
    pub(in crate::protocol::core) terms: Vec<ExtensionOpeningReductionTerm<E>>,
    pub(in crate::protocol::core) padded_point: Vec<E>,
    pub(in crate::protocol::core) split_bits: usize,
    pub(in crate::protocol::core) eta: Vec<E>,
    pub(in crate::protocol::core) true_input_claim: E,
}

pub(in crate::protocol::core) struct ProvedExtensionOpeningReduction<E: FieldCore> {
    pub(in crate::protocol::core) reduction: ExtensionOpeningReduction<E>,
    pub(in crate::protocol::core) row_coefficients: Vec<E>,
    pub(in crate::protocol::core) protocol_point: Vec<E>,
}

/// Contiguous per-group claim spans `(start, end, num_vars)` in flat claim
/// order for the extension-opening reduction.
///
/// A single-group (or padded/homogeneous) batch collapses to one span covering
/// every claim at the shared padded arity, which keeps the historical
/// byte-identical reduction path. Multi-group roots pass their real layout so
/// each group is reduced at its own prefix of the shared point and lifted into
/// the joint sumcheck domain by constant extension (table tiling): a group
/// claim at a point prefix equals the claim of the constant-extended
/// polynomial at the full padded point.
fn eor_claim_spans<F, E>(
    claim_layout: Option<&OpeningClaimsLayout>,
    num_claims: usize,
    num_vars: usize,
) -> Result<Vec<(usize, usize, usize)>, AkitaError>
where
    F: FieldCore,
    E: ExtField<F>,
{
    let Some(layout) = claim_layout else {
        return Ok(vec![(0, num_claims, num_vars)]);
    };
    if layout.num_total_polynomials() != num_claims || layout.max_num_vars() != num_vars {
        return Err(AkitaError::InvalidInput(
            "extension-opening reduction claim layout does not match the padded batch".to_string(),
        ));
    }
    let (split_bits, _) = tensor_opening_split::<F, E>()?;
    let mut spans = Vec::with_capacity(layout.num_groups());
    let mut start = 0usize;
    for group_index in 0..layout.num_groups() {
        let group = layout.group_layout(group_index)?;
        let group_num_vars = group.num_vars();
        if group_num_vars < split_bits || group_num_vars > num_vars {
            return Err(AkitaError::InvalidInput(
                "extension-opening reduction group arity is outside the padded batch".to_string(),
            ));
        }
        let end = start
            .checked_add(group.num_polynomials())
            .ok_or_else(|| AkitaError::InvalidInput("EOR claim span overflow".to_string()))?;
        spans.push((start, end, group_num_vars));
        start = end;
    }
    if start != num_claims {
        return Err(AkitaError::InvalidSize {
            expected: num_claims,
            actual: start,
        });
    }
    Ok(spans)
}

/// Build the extension-opening reduction terms span by span.
///
/// Each claim span (group) yields terms over its *own* packed tail prefix.
/// Spans shorter than the padded batch are lifted to the joint sumcheck
/// domain by **virtual** constant extension instead of physically replicating
/// tables: the tiled table is the original table repeated end-to-end
/// (little-endian packing), so
///
/// - while the group still has unbound variables, round messages over the
///   tiled domain decompose into the group-domain accumulation against the
///   truncated-tail tensor factor (partition of unity of the equality kernel
///   in the replicated coordinates), and
/// - once the group is exhausted, the folded tiled witness is one constant
///   `w*`, so the remaining rounds have a closed form (quadratic coefficient
///   exactly zero, constant `w*` times the transparent even-half factor sum).
///
/// This keeps grouped roots at `>= 2^30` padded variables inside the
/// equality-table budget (no dense full-tail factor per term) and avoids the
/// `copies`-fold witness blow-up, while the emitted round messages, and hence
/// the proof bytes, stay identical to the materialized-tiling computation.
///
/// Sparse and dense spans are handled independently: a span whose backend
/// supports a sparse linear combination keeps the sparse/lazy-tensor path even
/// when another span in the same batch must fall back to dense witnesses.
pub(in crate::protocol::core) fn build_extension_opening_reduction_terms<
    F,
    E,
    P,
    B,
    const D: usize,
>(
    backend: &B,
    prepared: Option<&<B as ComputeBackendSetup<F>>::PreparedSetup>,
    polys: &[&P],
    row_coefficients: &[E],
    tail_point: &[E],
    eta: &[E],
    claim_spans: &[(usize, usize, usize)],
) -> Result<Vec<ExtensionOpeningReductionTerm<E>>, AkitaError>
where
    F: FieldCore + CanonicalField + AkitaSerialize,
    E: ExtField<F> + MulBaseUnreduced<F>,
    P: RootTensorSource<F, D>,
    B: ComputeBackendSetup<F>
        + for<'a> TensorProjectionBatchKernel<P::TensorBatchView<'a>, F, E, D>
        + for<'a> TensorProjectionKernel<P::TensorView<'a>, F, E, D>,
{
    let _span =
        tracing::info_span!("extension_opening_reduction_terms", num_terms = polys.len()).entered();
    if polys.len() != row_coefficients.len() {
        return Err(AkitaError::InvalidSize {
            expected: polys.len(),
            actual: row_coefficients.len(),
        });
    }
    let (split_bits, _) = tensor_opening_split::<F, E>()?;
    let mut terms = Vec::with_capacity(claim_spans.len());
    for &(start, end, group_num_vars) in claim_spans {
        let group_tail_vars = group_num_vars.checked_sub(split_bits).ok_or_else(|| {
            AkitaError::InvalidInput("EOR group arity below the tensor split".to_string())
        })?;
        if group_tail_vars > tail_point.len() {
            return Err(AkitaError::InvalidSize {
                expected: tail_point.len(),
                actual: group_tail_vars,
            });
        }
        let tiled = group_tail_vars < tail_point.len();

        let span_witness = {
            let _span = tracing::info_span!(
                "extension_opening_sparse_terms",
                num_terms = end - start,
                group_tail_vars
            )
            .entered();
            TensorProjectionBatchKernel::sparse_linear_combination(
                backend,
                prepared,
                P::tensor_batch(&polys[start..end])?,
                &row_coefficients[start..end],
            )?
        };
        if let Some(witness) = span_witness {
            terms.push(sparse_span_term::<F, E>(
                witness,
                tail_point,
                group_tail_vars,
                eta,
                tiled,
            )?);
            continue;
        }

        // Dense fallback for this span only.
        let _dense_span = tracing::info_span!(
            "extension_opening_dense_witnesses",
            num_terms = end - start,
            group_tail_vars
        )
        .entered();
        for (poly, coeff) in polys[start..end]
            .iter()
            .zip(row_coefficients[start..end].iter().copied())
        {
            let witness = {
                let _s = tracing::info_span!("eor_packed_witness").entered();
                TensorProjectionKernel::packed_witness(backend, prepared, poly.tensor_view()?)?
            };
            terms.push(if tiled {
                tiled_term_from_packed_witness::<F, E>(
                    witness,
                    tail_point,
                    group_tail_vars,
                    eta,
                    coeff,
                )?
            } else {
                extension_opening_term_from_packed_witness::<F, E>(witness, tail_point, eta, coeff)?
            });
        }
    }
    Ok(terms)
}

/// One reduction term for a sparse span, over its own group tail prefix.
///
/// Full-arity spans keep the historical byte-identical construction (dense or
/// lazy tensor factor over the full tail); shorter spans are lifted to the
/// joint domain by virtual constant extension.
fn sparse_span_term<F, E>(
    witness: SparseExtensionOpeningWitness<E>,
    tail_point: &[E],
    group_tail_vars: usize,
    eta: &[E],
    tiled: bool,
) -> Result<ExtensionOpeningReductionTerm<E>, AkitaError>
where
    F: FieldCore + CanonicalField,
    E: ExtField<F>,
{
    let lazy_rounds = group_tail_vars.min(SPARSE_TENSOR_FACTOR_MAX_LAZY_ROUNDS);
    if tiled {
        return ExtensionOpeningReductionTerm::new_tiled_sparse::<F>(
            witness,
            tail_point,
            eta,
            E::one(),
            lazy_rounds,
        );
    }
    if lazy_rounds == 0 {
        let factor_evals = {
            let _span = tracing::debug_span!(
                "extension_opening_factor_evals",
                tail_vars = tail_point.len()
            )
            .entered();
            tensor_equality_factor_evals::<F, E>(tail_point, eta)?
        };
        return ExtensionOpeningReductionTerm::new_sparse(witness, factor_evals, E::one());
    }
    let _span = tracing::debug_span!(
        "extension_opening_lazy_tensor_factor",
        tail_vars = tail_point.len(),
        lazy_rounds
    )
    .entered();
    ExtensionOpeningReductionTerm::new_sparse_tensor_factor::<F>(
        witness,
        tail_point.to_vec(),
        eta.to_vec(),
        E::one(),
        lazy_rounds,
    )
}

fn extension_opening_term_from_packed_witness<F, E>(
    witness: TensorPackedWitness<E>,
    tail_point: &[E],
    eta: &[E],
    coeff: E,
) -> Result<ExtensionOpeningReductionTerm<E>, AkitaError>
where
    F: FieldCore + CanonicalField,
    E: ExtField<F>,
{
    let factor_evals = tensor_equality_factor_evals::<F, E>(tail_point, eta)?;
    match witness {
        TensorPackedWitness::Dense(witness_evals) => {
            ExtensionOpeningReductionTerm::new(witness_evals, factor_evals, coeff)
        }
        TensorPackedWitness::Sparse(witness) => {
            ExtensionOpeningReductionTerm::new_sparse(witness, factor_evals, coeff)
        }
    }
}

/// Virtually tiled term for one packed witness of a shorter span.
fn tiled_term_from_packed_witness<F, E>(
    witness: TensorPackedWitness<E>,
    tail_point: &[E],
    group_tail_vars: usize,
    eta: &[E],
    coeff: E,
) -> Result<ExtensionOpeningReductionTerm<E>, AkitaError>
where
    F: FieldCore + CanonicalField,
    E: ExtField<F>,
{
    match witness {
        TensorPackedWitness::Dense(witness_evals) => {
            ExtensionOpeningReductionTerm::new_tiled_dense::<F>(
                witness_evals,
                tail_point,
                eta,
                coeff,
            )
        }
        TensorPackedWitness::Sparse(witness) => ExtensionOpeningReductionTerm::new_tiled_sparse::<F>(
            witness,
            tail_point,
            eta,
            coeff,
            group_tail_vars.min(SPARSE_TENSOR_FACTOR_MAX_LAZY_ROUNDS),
        ),
    }
}

#[allow(clippy::too_many_arguments)]
pub(in crate::protocol::core) fn prepare_extension_opening_reduction<
    F,
    E,
    T,
    P,
    B,
    const D: usize,
>(
    backend: &B,
    prepared: Option<&<B as ComputeBackendSetup<F>>::PreparedSetup>,
    polys: &[&P],
    opening_batch: &OpeningClaims<'_, E>,
    claim_layout: Option<&OpeningClaimsLayout>,
    pad_base_evals: bool,
    transcript: &mut T,
) -> Result<PreparedExtensionOpeningReduction<E>, AkitaError>
where
    F: FieldCore + CanonicalField + AkitaSerialize,
    E: ExtField<F> + MulBaseUnreduced<F>,
    T: Transcript<F>,
    P: RootTensorSource<F, D>,
    B: ComputeBackendSetup<F>
        + for<'a> TensorProjectionBatchKernel<P::TensorBatchView<'a>, F, E, D>
        + for<'a> TensorProjectionKernel<P::TensorView<'a>, F, E, D>,
{
    let num_claims = opening_batch.num_total_polynomials();
    let num_vars = opening_batch.num_vars();
    let _span =
        tracing::info_span!("prepare_extension_opening_reduction", num_claims, num_vars).entered();
    let (split_bits, width) = tensor_opening_split::<F, E>()?;
    if split_bits > num_vars {
        return Err(AkitaError::InvalidPointDimension {
            expected: split_bits,
            actual: num_vars,
        });
    }
    if polys.len() != num_claims {
        return Err(AkitaError::InvalidInput(
            "extension-opening reduction input lengths do not match".to_string(),
        ));
    }
    let claim_spans = eor_claim_spans::<F, E>(claim_layout, num_claims, num_vars)?;

    let padded_point = opening_batch.point().to_vec();

    let mut openings = Vec::with_capacity(num_claims);
    let mut partials = Vec::with_capacity(width.saturating_mul(num_claims));
    let mut row_partials_by_claim = Vec::with_capacity(num_claims);
    {
        let _span =
            tracing::info_span!("extension_opening_prepare_partials", width, split_bits).entered();
        // Each span's polynomials are reduced at their own prefix of the
        // shared padded point; a group claim at that prefix equals the claim
        // of the constant-extended polynomial at the full point, so the
        // derived column partials feed the joint reduction unchanged.
        let mut point_partials = Vec::with_capacity(num_claims);
        for &(start, end, group_num_vars) in &claim_spans {
            let span_partials = TensorProjectionBatchKernel::column_partials_batch(
                backend,
                prepared,
                P::tensor_batch(&polys[start..end])?,
                &padded_point[..group_num_vars],
            )?;
            if span_partials.len() != end - start {
                return Err(AkitaError::InvalidSize {
                    expected: end - start,
                    actual: span_partials.len(),
                });
            }
            point_partials.extend(span_partials);
        }
        if point_partials.len() != num_claims {
            return Err(AkitaError::InvalidSize {
                expected: num_claims,
                actual: point_partials.len(),
            });
        }
        for column_partials in point_partials {
            let opening = derive_tensor_extension_opening_claim_from_partials::<F, E>(
                &padded_point,
                &column_partials,
            )?;
            let row_partials = tensor_row_partials_from_columns::<F, E>(&column_partials)?;
            partials.extend(column_partials);
            openings.push(opening);
            row_partials_by_claim.push(row_partials);
        }
    }
    let proof_partials = partials.clone();
    let row_coefficients = if pad_base_evals {
        vec![E::one(); num_claims]
    } else {
        let transcript_openings = openings.as_slice();
        append_claim_values_to_transcript::<F, E, T>(transcript_openings, transcript);
        let opening_shape = opening_batch.layout()?;
        sample_public_row_coefficients::<F, E, T>(&opening_shape, transcript)?
    };
    if row_partials_by_claim.len() != row_coefficients.len() {
        return Err(AkitaError::InvalidSize {
            expected: row_partials_by_claim.len(),
            actual: row_coefficients.len(),
        });
    }
    let expected_partials = width
        .checked_mul(row_coefficients.len())
        .ok_or_else(|| AkitaError::InvalidInput("EOR partial count overflow".to_string()))?;
    if proof_partials.len() != expected_partials {
        return Err(AkitaError::InvalidSize {
            expected: expected_partials,
            actual: proof_partials.len(),
        });
    }
    let proof_row_partials_by_claim = proof_partials
        .chunks_exact(width)
        .map(tensor_row_partials_from_columns::<F, E>)
        .collect::<Result<Vec<_>, _>>()?;
    {
        let _span = tracing::debug_span!(
            "extension_opening_absorb_partials",
            partials_len = proof_partials.len()
        )
        .entered();
        for partial in &proof_partials {
            append_ext_field::<F, E, T>(transcript, ABSORB_EVALUATION_CLAIMS, partial);
        }
    }
    let eta = (0..split_bits)
        .map(|_| sample_ext_challenge::<F, E, T>(transcript, CHALLENGE_SUMCHECK_BATCH))
        .collect::<Vec<_>>();
    let input_claim = {
        let _span = tracing::debug_span!("extension_opening_input_claim").entered();
        proof_row_partials_by_claim
            .iter()
            .zip(row_coefficients.iter().copied())
            .try_fold(E::zero(), |acc, (row_partials, coeff)| {
                tensor_reduction_claim_from_rows::<F, E>(row_partials, &eta)
                    .map(|claim| acc + coeff * claim)
            })?
    };
    let true_input_claim = row_partials_by_claim
        .iter()
        .zip(row_coefficients.iter().copied())
        .try_fold(E::zero(), |acc, (row_partials, coeff)| {
            tensor_reduction_claim_from_rows::<F, E>(row_partials, &eta)
                .map(|claim| acc + coeff * claim)
        })?;
    debug_assert_eq!(input_claim, true_input_claim);

    let tail_point = &padded_point[split_bits..];
    let terms = build_extension_opening_reduction_terms::<F, E, P, B, D>(
        backend,
        prepared,
        polys,
        &row_coefficients,
        tail_point,
        &eta,
        &claim_spans,
    )?;

    Ok(PreparedExtensionOpeningReduction {
        proof_partials,
        row_coefficients,
        terms,
        padded_point,
        split_bits,
        eta,
        true_input_claim,
    })
}

#[allow(clippy::too_many_arguments)]
pub(in crate::protocol::core) fn prove_extension_opening_reduction<F, E, T, P, B, const D: usize>(
    tensor_backend: &B,
    tensor_prepared: Option<&<B as ComputeBackendSetup<F>>::PreparedSetup>,
    polys: &[&P],
    opening_batch: &OpeningClaims<'_, E>,
    claim_layout: Option<&OpeningClaimsLayout>,
    pad_base_evals: bool,
    transcript: &mut T,
    path: &'static str,
) -> Result<ProvedExtensionOpeningReduction<E>, AkitaError>
where
    F: FieldCore + CanonicalField + AkitaSerialize,
    E: ExtField<F> + HasUnreducedOps + HasOptimizedFold + MulBaseUnreduced<F> + AkitaSerialize,
    T: Transcript<F>,
    P: RootTensorSource<F, D>,
    B: ComputeBackendSetup<F>
        + for<'a> TensorProjectionBatchKernel<P::TensorBatchView<'a>, F, E, D>
        + for<'a> TensorProjectionKernel<P::TensorView<'a>, F, E, D>,
{
    let _span = tracing::info_span!(
        "prove_extension_opening_reduction",
        path,
        num_claims = opening_batch.num_total_polynomials()
    )
    .entered();
    let backend = tensor_backend;
    let prepared = prepare_extension_opening_reduction::<F, E, T, P, B, D>(
        backend,
        tensor_prepared,
        polys,
        opening_batch,
        claim_layout,
        pad_base_evals,
        transcript,
    )?;
    let tail_point = &prepared.padded_point[prepared.split_bits..];
    let prover_claim =
        ExtensionOpeningReductionProver::input_claim_from_terms(prepared.terms.as_slice())?;
    if prover_claim != prepared.true_input_claim {
        return Err(AkitaError::InvalidInput(
            "extension-opening reduction input claim mismatch".to_string(),
        ));
    }
    let mut prover = {
        let _span = tracing::info_span!("extension_opening_reduction_prover_new", path).entered();
        ExtensionOpeningReductionProver::new(prepared.terms, prover_claim)?
    };
    let _eor_sumcheck_span = tracing::info_span!(
        "extension_opening_reduction_sumcheck",
        path = path,
        num_rounds = prover.num_rounds()
    )
    .entered();
    let (sumcheck_proof, rho, final_claim) = prover.prove::<F, T, _>(transcript, |tr| {
        sample_ext_challenge::<F, E, T>(tr, CHALLENGE_SUMCHECK_ROUND)
    })?;
    let final_terms = prover.final_terms().ok_or_else(|| {
        AkitaError::InvalidInput(format!(
            "{path} extension-opening reduction has not reached a final point"
        ))
    })?;
    let final_factor =
        tensor_equality_factor_eval_at_point::<F, E>(tail_point, &prepared.eta, &rho)?;
    if final_terms
        .iter()
        .any(|(_, _, factor)| *factor != final_factor)
    {
        return Err(AkitaError::InvalidInput(format!(
            "{path} extension-opening reduction transparent factor mismatch"
        )));
    }
    let expected_final = final_terms
        .into_iter()
        .fold(E::zero(), |acc, (coeff, witness, factor)| {
            acc + coeff * witness * factor
        });
    if final_claim != expected_final {
        return Err(AkitaError::InvalidInput(format!(
            "{path} extension-opening reduction final oracle mismatch"
        )));
    }
    let protocol_point = {
        let _span = tracing::info_span!("extension_opening_protocol_point").entered();
        ring_subfield_packed_extension_opening_point::<F, E, D>(rho.len(), &rho)?
    };
    let reduction = ExtensionOpeningReduction {
        proof: ExtensionOpeningReductionProof {
            partials: prepared.proof_partials,
            sumcheck: sumcheck_proof,
        },
        final_claim,
        final_factor,
    };

    Ok(ProvedExtensionOpeningReduction {
        reduction,
        row_coefficients: prepared.row_coefficients,
        protocol_point,
    })
}

pub(in crate::protocol::core) type MultiplierWeightSlices<'a, F, const D: usize> =
    (&'a [CyclotomicRing<F, D>], &'a [CyclotomicRing<F, D>]);
pub(in crate::protocol::core) type FoldedClaimEvals<F, const D: usize> =
    (Vec<CyclotomicRing<F, D>>, Vec<Vec<CyclotomicRing<F, D>>>);
