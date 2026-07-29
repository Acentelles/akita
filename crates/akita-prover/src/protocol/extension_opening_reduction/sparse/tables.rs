use super::*;

///
/// - [`Dense`](Self::Dense): dense witness paired with a dense factor. The
///   initial shape for non-onehot terms and the steady state of the recursive
///   EOR path.
/// - [`Sparse`](Self::Sparse): sparse witness paired with a (dense or lazy
///   tensor) [`SparseFactor`]. The initial shape for onehot terms.
/// - [`Tiled`](Self::Tiled): a group-domain inner table lifted to the joint
///   tail domain by *virtual* constant extension. `inner` lives over the
///   group's tail prefix paired with the truncated-tail factor; `tail` tracks
///   the full-tail transparent factor state. Round messages equal the
///   physically tiled table's messages element-for-element (see
///   [`TiledTailFactor`]), without materializing the replication.
#[derive(Debug, Clone)]
pub(in crate::protocol::extension_opening_reduction) enum ExtensionOpeningTables<E: FieldCore> {
    Dense {
        witness: Vec<E>,
        factor: Vec<E>,
    },
    Sparse {
        witness: SparseExtensionOpeningWitness<E>,
        factor: SparseFactor<E>,
    },
    Tiled {
        inner: Box<ExtensionOpeningTables<E>>,
        tail: TiledTailFactor<E>,
    },
}

impl<E: FieldCore> ExtensionOpeningTables<E> {
    pub(in crate::protocol::extension_opening_reduction) fn len(&self) -> usize {
        match self {
            Self::Dense { witness, .. } => witness.len(),
            Self::Sparse { witness, .. } => witness.table_len(),
            Self::Tiled { tail, .. } => tail.remaining_len(),
        }
    }

    pub(in crate::protocol::extension_opening_reduction) fn claim(&self) -> Result<E, AkitaError> {
        match self {
            Self::Dense { witness, factor } => extension_opening_reduction_claim(witness, factor),
            Self::Sparse { witness, factor } => match factor {
                SparseFactor::Dense(factor_evals) => witness.claim_with_factor(factor_evals),
                SparseFactor::Tensor(factor) => {
                    if witness.table_len() != factor.len() {
                        return Err(AkitaError::InvalidSize {
                            expected: witness.table_len(),
                            actual: factor.len(),
                        });
                    }
                    Ok(witness.claim_with_factor_fn(|idx| factor.factor_at_index(idx)))
                }
            },
            // The tiled claim over the joint domain regroups onto the group
            // domain: `sum_{lo,hi} W(lo)·A(lo,hi) = sum_lo W(lo)·B(lo)` with
            // `B` the truncated-tail factor the inner tables already hold.
            Self::Tiled { inner, .. } => inner.claim(),
        }
    }

    /// Fully folded witness value, ignoring the factor's representation.
    pub(in crate::protocol::extension_opening_reduction) fn final_witness(&self) -> Option<E> {
        match self {
            Self::Dense { witness, .. } => (witness.len() == 1).then(|| witness[0]),
            Self::Sparse { witness, .. } => witness.final_eval(),
            Self::Tiled { inner, .. } => inner.final_witness(),
        }
    }

    pub(in crate::protocol::extension_opening_reduction) fn final_witness_and_factor_evals(
        &self,
    ) -> Option<(E, E)> {
        match self {
            Self::Dense { witness, factor } => {
                (factor.len() == 1 && witness.len() == 1).then(|| (witness[0], factor[0]))
            }
            Self::Sparse { witness, factor } => match factor {
                SparseFactor::Dense(factor_evals) => (factor_evals.len() == 1)
                    .then(|| witness.final_eval())
                    .flatten()
                    .map(|witness| (witness, factor_evals[0])),
                SparseFactor::Tensor(_) => None,
            },
            Self::Tiled { inner, tail } => tail
                .final_value()
                .and_then(|factor| inner.final_witness().map(|witness| (witness, factor))),
        }
    }
}

impl<E: FieldCore + HasUnreducedOps> ExtensionOpeningTables<E> {
    pub(in crate::protocol::extension_opening_reduction) fn accumulate_round(
        &self,
        coeff: E,
        constant: &mut E,
        quadratic: &mut E,
    ) {
        match self {
            Self::Dense { witness, factor } => {
                let (round_constant, round_quadratic) =
                    accumulate_dense_round(witness, factor, coeff);
                *constant += round_constant;
                *quadratic += round_quadratic;
            }
            Self::Sparse { witness, factor } => match factor {
                SparseFactor::Dense(factor_evals) => {
                    witness.accumulate_round(factor_evals, coeff, constant, quadratic);
                }
                SparseFactor::Tensor(factor) => {
                    witness.accumulate_round_with_factor(coeff, constant, quadratic, |pair| {
                        factor.factor_pair(pair)
                    });
                }
            },
            Self::Tiled { inner, tail } => {
                if inner.len() > 1 {
                    // Group rounds: the tiled round message regroups per high
                    // assignment onto the group domain against the
                    // truncated-tail factor, which is exactly the inner
                    // accumulation. Equal to the materialized-table message
                    // element-for-element.
                    inner.accumulate_round(coeff, constant, quadratic);
                } else {
                    // Constant-extension plateau: every remaining entry of the
                    // virtually tiled witness equals the fully folded group
                    // value `w*`, so the odd/even witness difference vanishes
                    // (the quadratic coefficient is exactly zero, matching the
                    // materialized table) and the constant coefficient is
                    // `w*` times the closed-form even-half factor sum.
                    debug_assert_eq!(inner.len(), 1);
                    let witness = inner.final_witness().unwrap_or_else(E::zero);
                    *constant += coeff * (witness * tail.even_half_sum());
                }
            }
        }
    }
}

impl<E: FieldCore + HasUnreducedOps + HasOptimizedFold> SparseFactor<E> {
    /// Fold the transparent factor by one sumcheck challenge, materializing the
    /// lazy tensor factor into a dense table once it reaches its split depth.
    pub(in crate::protocol::extension_opening_reduction) fn fold_in_place(&mut self, r_round: E) {
        match self {
            SparseFactor::Dense(factor_evals) => {
                fold_evals_in_place(factor_evals, r_round);
            }
            SparseFactor::Tensor(tensor_factor) => {
                tensor_factor.fold_in_place(r_round);
                if tensor_factor.is_ready_to_materialize() {
                    let dense = tensor_factor.materialize_dense();
                    *self = SparseFactor::Dense(dense);
                }
            }
        }
    }
}

impl<E: FieldCore + HasUnreducedOps + HasOptimizedFold> ExtensionOpeningTables<E> {
    pub(in crate::protocol::extension_opening_reduction) fn fold_in_place(&mut self, r_round: E) {
        match self {
            Self::Dense { witness, factor } => {
                fold_dense_reduction_tables_in_place(witness, factor, r_round);
            }
            Self::Sparse { witness, factor } => {
                witness.fold_in_place(r_round);
                factor.fold_in_place(r_round);
            }
            Self::Tiled { inner, tail } => {
                // The transparent tail state folds every round; the inner
                // group tables only fold while the group has unbound
                // variables (afterwards they hold the single value `w*`).
                if inner.len() > 1 {
                    inner.fold_in_place(r_round);
                }
                tail.fold_in_place(r_round);
            }
        }
    }
}

/// Fold a sparse term's factor and witness by one challenge AND compute the next
/// round's `(constant, quadratic)` in a single witness sweep.
///
/// Sparse counterpart of [`fused_fold_and_accumulate`], valid only inside the
/// merge-free plateau. The factor is folded first so the witness sweep reads the
/// next round's factor children while folding each entry. Returns the *unscaled*
/// next-round coefficients; the caller applies the term coefficient.
pub(in crate::protocol::extension_opening_reduction) fn fused_fold_and_accumulate_sparse<E>(
    witness: &mut SparseExtensionOpeningWitness<E>,
    factor: &mut SparseFactor<E>,
    r_round: E,
) -> (E, E)
where
    E: FieldCore + HasUnreducedOps + HasOptimizedFold,
{
    factor.fold_in_place(r_round);
    match factor {
        SparseFactor::Dense(factor_evals) => witness
            .fused_fold_accumulate_merge_free(r_round, &|pair| {
                (factor_evals[2 * pair], factor_evals[2 * pair + 1])
            }),
        SparseFactor::Tensor(tensor_factor) => witness
            .fused_fold_accumulate_merge_free(r_round, &|pair| tensor_factor.factor_pair(pair)),
    }
}
