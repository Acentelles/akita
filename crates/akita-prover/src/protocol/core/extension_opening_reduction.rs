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
    /// Raw reduction sumcheck point over the joint tail; suffix-aligned
    /// grouped callers derive per-group packed points from its slices.
    pub(in crate::protocol::core) rho: Vec<E>,
}

/// How grouped claims embed into the joint extension-opening reduction domain.
#[derive(Clone, Copy)]
pub(in crate::protocol::core) enum EorClaimAlignment<'a> {
    /// Historical single-span batch at the shared padded arity.
    Flat,
    /// Groups at nested prefixes of the shared point (grouped roots): shorter
    /// groups lift by constant extension in the missing HIGH tail variables
    /// (virtual tiling).
    Prefix(&'a OpeningClaimsLayout),
    /// Groups as contiguous suffixes of the shared point (recursive suffix
    /// carried-claim batches): shorter groups lift by constant extension in
    /// the missing LOW variables (strided expansion). At most one group is
    /// shorter than the padded arity.
    Suffix(&'a OpeningClaimsLayout),
}

/// One contiguous per-group claim span in flat claim order.
#[derive(Clone, Copy)]
pub(in crate::protocol::core) struct EorClaimSpan {
    start: usize,
    end: usize,
    /// The span's own arity (number of point variables it claims).
    num_vars: usize,
    /// First coordinate of the span's own point inside the shared padded
    /// point: `0` for prefix-aligned spans, `padded - num_vars` for
    /// suffix-aligned spans. Full-arity spans have offset `0` either way.
    offset: usize,
}

/// Contiguous per-group claim spans in flat claim order for the
/// extension-opening reduction.
///
/// A single-group (or padded/homogeneous) batch collapses to one span covering
/// every claim at the shared padded arity, which keeps the historical
/// byte-identical reduction path. Grouped batches pass their real layout so
/// each group is reduced at its own slice of the shared point and lifted into
/// the joint sumcheck domain by constant extension: prefix-aligned groups tile
/// in the missing HIGH variables, suffix-aligned groups expand in the missing
/// LOW variables. Either way a group claim at its point slice equals the claim
/// of the constant-extended polynomial at the full padded point (partition of
/// unity of the equality kernel in the replicated coordinates).
fn eor_claim_spans<F, E>(
    alignment: EorClaimAlignment<'_>,
    num_claims: usize,
    num_vars: usize,
) -> Result<Vec<EorClaimSpan>, AkitaError>
where
    F: FieldCore,
    E: ExtField<F>,
{
    let (layout, suffix_aligned) = match alignment {
        EorClaimAlignment::Flat => {
            return Ok(vec![EorClaimSpan {
                start: 0,
                end: num_claims,
                num_vars,
                offset: 0,
            }]);
        }
        EorClaimAlignment::Prefix(layout) => (layout, false),
        EorClaimAlignment::Suffix(layout) => (layout, true),
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
        spans.push(EorClaimSpan {
            start,
            end,
            num_vars: group_num_vars,
            offset: if suffix_aligned {
                num_vars - group_num_vars
            } else {
                0
            },
        });
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
    claim_spans: &[EorClaimSpan],
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
    for &EorClaimSpan {
        start,
        end,
        num_vars: group_num_vars,
        offset,
    } in claim_spans
    {
        let group_tail_vars = group_num_vars.checked_sub(split_bits).ok_or_else(|| {
            AkitaError::InvalidInput("EOR group arity below the tensor split".to_string())
        })?;
        if group_tail_vars > tail_point.len() {
            return Err(AkitaError::InvalidSize {
                expected: tail_point.len(),
                actual: group_tail_vars,
            });
        }
        let lifted = group_tail_vars < tail_point.len();

        if offset > 0 && lifted {
            // Suffix-aligned shorter span: strided (LOW-variable constant
            // extension) lifting; each packed table is materialized densely
            // over the joint tail against the full-tail transparent factor.
            let _dense_span = tracing::info_span!(
                "extension_opening_strided_witnesses",
                num_terms = end - start,
                group_tail_vars,
                offset
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
                terms.push(strided_term_from_packed_witness::<F, E>(
                    witness,
                    tail_point,
                    group_tail_vars,
                    eta,
                    coeff,
                )?);
            }
            continue;
        }

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
                lifted,
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
            terms.push(if lifted {
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

/// Strided (suffix-aligned) term for one packed witness of a shorter span.
///
/// The lifted table is `g~(x) = g(x >> pad)` over the joint tail (each entry
/// of `g` repeated `2^pad` times consecutively), paired against the full-tail
/// transparent factor. By partition of unity of the equality kernel in the
/// replicated LOW coordinates, `sum_x g~(x) * A_eta(x)` equals the span's own
/// claim `sum_w g(w) * A_eta^{own}(w)`, and the final folded value is
/// `g~(rho) = g(rho[pad..])` (`specs/eor-setup-prefix-absorption.md`).
fn strided_term_from_packed_witness<F, E>(
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
    let TensorPackedWitness::Dense(witness_evals) = witness else {
        return Err(AkitaError::InvalidInput(
            "strided extension-opening spans require a dense packed witness".to_string(),
        ));
    };
    let pad = tail_point.len() - group_tail_vars;
    let expected_len = 1usize
        .checked_shl(group_tail_vars as u32)
        .ok_or_else(|| AkitaError::InvalidInput("strided EOR table overflow".to_string()))?;
    if witness_evals.len() != expected_len {
        return Err(AkitaError::InvalidSize {
            expected: expected_len,
            actual: witness_evals.len(),
        });
    }
    let strided_len = expected_len
        .checked_shl(pad as u32)
        .ok_or_else(|| AkitaError::InvalidInput("strided EOR table overflow".to_string()))?;
    let copies = 1usize << pad;
    let mut strided = Vec::with_capacity(strided_len);
    for value in witness_evals {
        strided.extend(std::iter::repeat_n(value, copies));
    }
    let factor_evals = tensor_equality_factor_evals::<F, E>(tail_point, eta)?;
    ExtensionOpeningReductionTerm::new(strided, factor_evals, coeff)
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
        TensorPackedWitness::Sparse(witness) => {
            ExtensionOpeningReductionTerm::new_tiled_sparse::<F>(
                witness,
                tail_point,
                eta,
                coeff,
                group_tail_vars.min(SPARSE_TENSOR_FACTOR_MAX_LAZY_ROUNDS),
            )
        }
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
    alignment: EorClaimAlignment<'_>,
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
    // Recorded before `alignment` is consumed. ONLY the recursive
    // carried-claim suffix builds its EOR batch with real claimed values
    // (`recursive_suffix_eor_claims`); the root passes a shape carrier
    // (`OpeningClaims::with_padded_point`, evaluations are placeholders), so
    // the Defect-1 probe below must never fire there or it would corrupt the
    // root proof and make the regression verdict unattributable.
    #[cfg(feature = "attack-probe")]
    let probe_suffix_aligned = matches!(alignment, EorClaimAlignment::Suffix(_));
    let claim_spans = eor_claim_spans::<F, E>(alignment, num_claims, num_vars)?;

    let padded_point = opening_batch.point().to_vec();

    let mut openings = Vec::with_capacity(num_claims);
    let mut partials = Vec::with_capacity(width.saturating_mul(num_claims));
    let mut row_partials_by_claim = Vec::with_capacity(num_claims);
    {
        let _span =
            tracing::info_span!("extension_opening_prepare_partials", width, split_bits).entered();
        // Each span's polynomials are reduced at their own slice of the
        // shared padded point (prefix for grouped roots, suffix for the
        // recursive carried-claim batch); a group claim at that slice equals
        // the claim of the constant-extended polynomial at the full point, so
        // the derived column partials feed the joint reduction unchanged.
        let mut point_partials = Vec::with_capacity(num_claims);
        let mut claim_points = Vec::with_capacity(num_claims);
        for &EorClaimSpan {
            start,
            end,
            num_vars: group_num_vars,
            offset,
        } in &claim_spans
        {
            let span_point = &padded_point[offset..offset + group_num_vars];
            let span_partials = TensorProjectionBatchKernel::column_partials_batch(
                backend,
                prepared,
                P::tensor_batch(&polys[start..end])?,
                span_point,
            )?;
            if span_partials.len() != end - start {
                return Err(AkitaError::InvalidSize {
                    expected: end - start,
                    actual: span_partials.len(),
                });
            }
            claim_points.extend(std::iter::repeat_n(span_point, end - start));
            point_partials.extend(span_partials);
        }
        if point_partials.len() != num_claims {
            return Err(AkitaError::InvalidSize {
                expected: num_claims,
                actual: point_partials.len(),
            });
        }
        for (column_partials, claim_point) in point_partials.into_iter().zip(claim_points) {
            let opening = derive_tensor_extension_opening_claim_from_partials::<F, E>(
                claim_point,
                &column_partials,
            )?;
            let row_partials = tensor_row_partials_from_columns::<F, E>(&column_partials)?;
            partials.extend(column_partials);
            openings.push(opening);
            row_partials_by_claim.push(row_partials);
        }
    }
    #[allow(unused_mut)]
    let mut proof_partials = partials.clone();
    // ---- Defect-1 regression injection (feature `attack-probe`) ------------
    // The previous level's stage 3 shipped carried claims `(O_S + delta,
    // O_w - delta)` while the honest column partials open the true `O_*`.
    // Ship partials that match whatever the batch claims: a constant shift of
    // `claimed - true` on a claim's `width` partials moves its derived
    // opening by exactly that amount, because
    // `derive_tensor_extension_opening_claim_from_partials` weights them by an
    // `eq` kernel whose weights sum to 1 (head-independent). Under the old
    // unit-coefficient batching the two shifts then cancelled in the batched
    // EOR input claim, so the shared sumcheck was still the honest one and the
    // verifier accepted. Inert for an honest prover (`claimed == true`).
    #[cfg(feature = "attack-probe")]
    if crate::attack_probe::is_armed() && probe_suffix_aligned {
        let mut claimed = Vec::with_capacity(num_claims);
        for group_index in 0..opening_batch.num_groups() {
            claimed.extend_from_slice(opening_batch.group_evaluations(group_index)?);
        }
        if claimed.len() != num_claims {
            return Err(AkitaError::InvalidInput(
                "probe: claimed value count mismatch".to_string(),
            ));
        }
        let mut shifted = 0usize;
        for (claim_idx, (&claim, truth)) in claimed.iter().zip(openings.iter_mut()).enumerate() {
            let delta = claim - *truth;
            if delta.is_zero() {
                continue;
            }
            shifted += 1;
            for slot in proof_partials
                .iter_mut()
                .skip(claim_idx * width)
                .take(width)
            {
                *slot += delta;
            }
            // Absorb what the VERIFIER will absorb (the false carried claim),
            // not the truth: the strongest attacker keeps its transcript in
            // lockstep and relies on the batching being blind to the shift.
            *truth = claim;
        }
        if shifted != 0 {
            crate::attack_probe::note(format!(
                "suffix EOR: shifted {shifted}/{num_claims} claims' partials to match false \
                 carried claims"
            ));
        }
    }
    // ---- end Defect-1 regression injection ---------------------------------
    // Per-claim batching coefficients. A single-claim batch needs none (the
    // sum has one term, so `rho_0 = 1` loses nothing and keeps the historical
    // recursive-suffix bytes). Every multi-claim batch MUST squeeze them from
    // the transcript AFTER absorbing the claimed values, root and recursive
    // alike: with `rho_l = 1` the combination `sum_l v_l` is satisfied by any
    // error vector summing to zero, which is exactly the accepted forgery of
    // Defect 1 in `aerie/_docs/ABSORPTION-AUDIT.md` (a compensating
    // `(+delta, -delta)` on the two carried claims of the `G = 2` recursive
    // suffix). Absorbing first and squeezing after restores the
    // Schwartz-Zippel pricing of [AK] Lemma 5.7.
    let row_coefficients = if pad_base_evals && num_claims == 1 {
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
    // Only consumed by the honest-prover cross-check below; the `attack-probe`
    // build deliberately drops that check (see there).
    #[cfg_attr(feature = "attack-probe", allow(unused_variables))]
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
    // The shipped and the true input claim coincide for any honest prover.
    // The `attack-probe` regression build deliberately breaks that (shipped
    // partials encode false carried claims), and the divergence is exactly
    // what the rho-weighted batching is supposed to expose, so do not assert.
    #[cfg(not(feature = "attack-probe"))]
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
    alignment: EorClaimAlignment<'_>,
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
        alignment,
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
        rho,
    })
}

pub(in crate::protocol::core) type MultiplierWeightSlices<'a, F, const D: usize> =
    (&'a [CyclotomicRing<F, D>], &'a [CyclotomicRing<F, D>]);
pub(in crate::protocol::core) type FoldedClaimEvals<F, const D: usize> =
    (Vec<CyclotomicRing<F, D>>, Vec<Vec<CyclotomicRing<F, D>>>);
