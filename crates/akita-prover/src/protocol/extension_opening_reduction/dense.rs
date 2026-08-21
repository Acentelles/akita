use super::sparse::DenseEorFactor;
use super::*;

#[cfg(feature = "parallel")]
const DENSE_PARALLEL_PAIR_THRESHOLD: usize = 1 << 14;

pub(crate) fn accumulate_dense_round<E: FieldCore + HasUnreducedOps>(
    witness_evals: &[E],
    factor_evals: &[E],
    coeff: E,
) -> (E, E) {
    let _span = tracing::trace_span!(
        "dense_extension_reduction_accumulate_round",
        table_len = witness_evals.len()
    )
    .entered();
    debug_assert_eq!(witness_evals.len(), factor_evals.len());
    if coeff == E::zero() {
        return (E::zero(), E::zero());
    }

    // Sum the wide products in `E::ProductAccum` only when the field has proven
    // that delayed reduction is exact for these batch sizes; otherwise reduce
    // each product immediately so the coefficients stay byte-identical to
    // per-term `Mul` (the `DELAYED_PRODUCT_SUM_IS_EXACT` contract).
    let (constant, quadratic) = if E::DELAYED_PRODUCT_SUM_IS_EXACT {
        accumulate_dense_round_with::<E, DelayedDeg2<E>>(witness_evals, factor_evals)
    } else {
        accumulate_dense_round_with::<E, DirectDeg2<E>>(witness_evals, factor_evals)
    };
    (coeff * constant, coeff * quadratic)
}

fn accumulate_dense_round_with<E, A>(witness_evals: &[E], factor_evals: &[E]) -> (E, E)
where
    E: FieldCore + HasUnreducedOps,
    A: Deg2RoundAccum<E>,
{
    let half = witness_evals.len() / 2;

    #[cfg(feature = "parallel")]
    {
        if half >= DENSE_PARALLEL_PAIR_THRESHOLD {
            return (0..half)
                .into_par_iter()
                .fold(A::zero, |mut acc, i| {
                    let w0 = witness_evals[2 * i];
                    let w1 = witness_evals[2 * i + 1];
                    let a0 = factor_evals[2 * i];
                    let a1 = factor_evals[2 * i + 1];

                    acc.add_constant_product(w0, a0);
                    acc.add_quadratic_product(w1 - w0, a1 - a0);
                    acc
                })
                .reduce(A::zero, A::merge)
                .finish();
        }
    }

    let mut acc = A::zero();
    for i in 0..half {
        let w0 = witness_evals[2 * i];
        let w1 = witness_evals[2 * i + 1];
        let a0 = factor_evals[2 * i];
        let a1 = factor_evals[2 * i + 1];

        acc.add_constant_product(w0, a0);
        acc.add_quadratic_product(w1 - w0, a1 - a0);
    }
    acc.finish()
}

/// Round accumulation over a dense witness with factor values supplied by a
/// pair lookup instead of a materialized table. Same per-pair arithmetic and
/// accumulation policy as [`accumulate_dense_round`], so the coefficients are
/// byte-identical for equal factor values.
pub(in crate::protocol::extension_opening_reduction) fn accumulate_dense_round_with_factor_fn<
    E: FieldCore + HasUnreducedOps,
    FN: Fn(usize) -> (E, E) + Sync,
>(
    witness_evals: &[E],
    factor_fn: FN,
    coeff: E,
) -> (E, E) {
    let _span = tracing::trace_span!(
        "dense_extension_reduction_accumulate_round",
        table_len = witness_evals.len()
    )
    .entered();
    if coeff == E::zero() {
        return (E::zero(), E::zero());
    }
    let (constant, quadratic) = if E::DELAYED_PRODUCT_SUM_IS_EXACT {
        accumulate_dense_round_factor_fn_with::<E, DelayedDeg2<E>, FN>(witness_evals, &factor_fn)
    } else {
        accumulate_dense_round_factor_fn_with::<E, DirectDeg2<E>, FN>(witness_evals, &factor_fn)
    };
    (coeff * constant, coeff * quadratic)
}

fn accumulate_dense_round_factor_fn_with<E, A, FN>(witness_evals: &[E], factor_fn: &FN) -> (E, E)
where
    E: FieldCore + HasUnreducedOps,
    A: Deg2RoundAccum<E>,
    FN: Fn(usize) -> (E, E) + Sync,
{
    let half = witness_evals.len() / 2;

    #[cfg(feature = "parallel")]
    {
        if half >= DENSE_PARALLEL_PAIR_THRESHOLD {
            return (0..half)
                .into_par_iter()
                .fold(A::zero, |mut acc, i| {
                    let w0 = witness_evals[2 * i];
                    let w1 = witness_evals[2 * i + 1];
                    let (a0, a1) = factor_fn(i);
                    acc.add_constant_product(w0, a0);
                    acc.add_quadratic_product(w1 - w0, a1 - a0);
                    acc
                })
                .reduce(A::zero, A::merge)
                .finish();
        }
    }

    let mut acc = A::zero();
    for i in 0..half {
        let w0 = witness_evals[2 * i];
        let w1 = witness_evals[2 * i + 1];
        let (a0, a1) = factor_fn(i);
        acc.add_constant_product(w0, a0);
        acc.add_quadratic_product(w1 - w0, a1 - a0);
    }
    acc.finish()
}

/// Boolean-sum claim `sum_i witness[i] * factor(i)` over a dense witness with a
/// factor lookup. Exact field algebra: the value equals the materialized-table
/// claim for equal factor values.
pub(in crate::protocol::extension_opening_reduction) fn dense_claim_with_factor_fn<
    E: FieldCore,
    FN: Fn(usize) -> E + Sync,
>(
    witness_evals: &[E],
    factor_fn: FN,
) -> E {
    #[cfg(feature = "parallel")]
    {
        witness_evals
            .par_iter()
            .enumerate()
            .map(|(i, &w)| w * factor_fn(i))
            .sum()
    }
    #[cfg(not(feature = "parallel"))]
    {
        witness_evals
            .iter()
            .enumerate()
            .map(|(i, &w)| w * factor_fn(i))
            .sum()
    }
}

/// Fold the witness by one variable AND pre-compute the next round's
/// `(constant, quadratic)` accumulation in a single pass, with factor values
/// supplied by a pair lookup (the lazy-factor counterpart of
/// [`fused_fold_and_accumulate`]; the factor itself was already folded by the
/// caller). Witness fold arithmetic and accumulation policy are identical, so
/// for equal factor values the coefficients are byte-identical.
pub(in crate::protocol::extension_opening_reduction) fn fused_fold_witness_with_factor_fn<
    E: HasUnreducedOps + HasOptimizedFold,
    FN: Fn(usize) -> (E, E) + Sync,
>(
    witness_evals: &mut Vec<E>,
    factor_fn: FN,
    r_round: E,
) -> (E, E) {
    let _span = tracing::trace_span!("fused_fold_and_accumulate", table_len = witness_evals.len())
        .entered();
    debug_assert!(witness_evals.len().is_power_of_two());
    debug_assert!(witness_evals.len() >= 4);
    if E::DELAYED_PRODUCT_SUM_IS_EXACT {
        fused_fold_witness_factor_fn_with::<E, DelayedDeg2<E>, FN>(
            witness_evals,
            &factor_fn,
            r_round,
        )
    } else {
        fused_fold_witness_factor_fn_with::<E, DirectDeg2<E>, FN>(
            witness_evals,
            &factor_fn,
            r_round,
        )
    }
}

fn fused_fold_witness_factor_fn_with<E, A, FN>(
    witness_evals: &mut Vec<E>,
    factor_fn: &FN,
    r_round: E,
) -> (E, E)
where
    E: FieldCore + HasUnreducedOps + HasOptimizedFold,
    A: Deg2RoundAccum<E>,
    FN: Fn(usize) -> (E, E) + Sync,
{
    let half = witness_evals.len() / 2;
    let quarter = half / 2;
    let ctx = E::precompute_fold(r_round);

    #[cfg(feature = "parallel")]
    {
        if quarter >= DENSE_PARALLEL_PAIR_THRESHOLD {
            let mut folded_w = Vec::<E>::with_capacity(half);
            // SAFETY: allocated with capacity `half`; `half` is even (table
            // length is a power of two >= 4), so the `par_chunks_mut(2)` loop
            // writes all `half` slots before the first read. `E: FieldCore`
            // is `Copy` with a trivial drop.
            unsafe {
                folded_w.set_len(half);
            }
            let acc = {
                let input_w: &[E] = witness_evals;
                folded_w
                    .par_chunks_mut(2)
                    .enumerate()
                    .fold(A::zero, |mut acc, (i, w_out)| {
                        let fw0 = E::fold_one(&ctx, input_w[4 * i], input_w[4 * i + 1]);
                        let fw1 = E::fold_one(&ctx, input_w[4 * i + 2], input_w[4 * i + 3]);
                        let (fa0, fa1) = factor_fn(i);
                        acc.add_constant_product(fw0, fa0);
                        acc.add_quadratic_product(fw1 - fw0, fa1 - fa0);
                        w_out[0] = fw0;
                        w_out[1] = fw1;
                        acc
                    })
                    .reduce(A::zero, A::merge)
            };
            *witness_evals = folded_w;
            return acc.finish();
        }
    }

    let mut acc = A::zero();
    for i in 0..quarter {
        let fw0 = E::fold_one(&ctx, witness_evals[4 * i], witness_evals[4 * i + 1]);
        let fw1 = E::fold_one(&ctx, witness_evals[4 * i + 2], witness_evals[4 * i + 3]);
        let (fa0, fa1) = factor_fn(i);
        acc.add_constant_product(fw0, fa0);
        acc.add_quadratic_product(fw1 - fw0, fa1 - fa0);
        witness_evals[2 * i] = fw0;
        witness_evals[2 * i + 1] = fw1;
    }
    witness_evals.truncate(half);
    acc.finish()
}

/// Fold both tables by one variable AND pre-compute the next round's
/// `(constant, quadratic)` accumulation in a single pass over the data.
///
/// A [`DenseEorFactor::Shared`] factor is read in place and its folded half
/// is written to a fresh owned table (identical values); owned tables fold
/// in place exactly as before. A [`DenseEorFactor::Lazy`] factor folds its
/// transparent state first (exact multilinear folding, identical values to a
/// dense fold) and supplies the folded pair values by lookup; it materializes
/// into an owned dense table at its split depth and rejoins the shared path.
pub(in crate::protocol::extension_opening_reduction) fn fused_fold_and_accumulate<E: HasUnreducedOps + HasOptimizedFold>(
    witness_evals: &mut Vec<E>,
    factor: &mut DenseEorFactor<E>,
    r_round: E,
) -> (E, E) {
    if let DenseEorFactor::Lazy(tensor) = factor {
        tensor.fold_in_place(r_round);
        if tensor.is_ready_to_materialize() {
            let dense = tensor.materialize_dense();
            let acc = fused_fold_witness_with_factor_fn(
                witness_evals,
                |pair| (dense[2 * pair], dense[2 * pair + 1]),
                r_round,
            );
            *factor = DenseEorFactor::Owned(dense);
            return acc;
        }
        return fused_fold_witness_with_factor_fn(
            witness_evals,
            |pair| tensor.factor_pair(pair),
            r_round,
        );
    }

    let _span = tracing::trace_span!("fused_fold_and_accumulate", table_len = witness_evals.len())
        .entered();
    debug_assert_eq!(witness_evals.len(), factor.len());
    debug_assert!(witness_evals.len().is_power_of_two());
    debug_assert!(witness_evals.len() >= 4);

    // The fold itself (`E::fold_one`) is always exact; only the product
    // accumulation respects `DELAYED_PRODUCT_SUM_IS_EXACT`, matching
    // `accumulate_dense_round`.
    if E::DELAYED_PRODUCT_SUM_IS_EXACT {
        fused_fold_and_accumulate_with::<E, DelayedDeg2<E>>(witness_evals, factor, r_round)
    } else {
        fused_fold_and_accumulate_with::<E, DirectDeg2<E>>(witness_evals, factor, r_round)
    }
}

fn fused_fold_and_accumulate_with<E, A>(
    witness_evals: &mut Vec<E>,
    factor: &mut DenseEorFactor<E>,
    r_round: E,
) -> (E, E)
where
    E: FieldCore + HasUnreducedOps + HasOptimizedFold,
    A: Deg2RoundAccum<E>,
{
    let half = witness_evals.len() / 2;
    let quarter = half / 2;
    let ctx = E::precompute_fold(r_round);

    #[cfg(feature = "parallel")]
    {
        if quarter >= DENSE_PARALLEL_PAIR_THRESHOLD {
            let mut folded_w = Vec::<E>::with_capacity(half);
            let mut folded_f = Vec::<E>::with_capacity(half);
            // SAFETY: both vectors are allocated with capacity `half`. `half` is
            // even (table length is a power of two >= 4), so the `par_chunks_mut(2)`
            // loop below yields exactly `quarter` chunks of length 2 and writes all
            // `half` slots before the first read (`*witness_evals = folded_w`).
            // `E: FieldCore` is `Copy` with a trivial drop, so overwriting the
            // uninitialized slots is sound.
            unsafe {
                folded_w.set_len(half);
                folded_f.set_len(half);
            }

            let acc = {
                let input_w: &[E] = witness_evals;
                let input_f: &[E] = factor.as_slice();

                folded_w
                    .par_chunks_mut(2)
                    .zip(folded_f.par_chunks_mut(2))
                    .enumerate()
                    .fold(A::zero, |mut acc, (i, (w_out, f_out))| {
                        let fw0 = E::fold_one(&ctx, input_w[4 * i], input_w[4 * i + 1]);
                        let fw1 = E::fold_one(&ctx, input_w[4 * i + 2], input_w[4 * i + 3]);
                        let fa0 = E::fold_one(&ctx, input_f[4 * i], input_f[4 * i + 1]);
                        let fa1 = E::fold_one(&ctx, input_f[4 * i + 2], input_f[4 * i + 3]);

                        acc.add_constant_product(fw0, fa0);
                        acc.add_quadratic_product(fw1 - fw0, fa1 - fa0);

                        w_out[0] = fw0;
                        w_out[1] = fw1;
                        f_out[0] = fa0;
                        f_out[1] = fa1;

                        acc
                    })
                    .reduce(A::zero, A::merge)
            };

            *witness_evals = folded_w;
            *factor = DenseEorFactor::Owned(folded_f);
            return acc.finish();
        }
    }

    match factor {
        DenseEorFactor::Owned(factor_evals) => {
            let mut acc = A::zero();
            for i in 0..quarter {
                let fw0 = E::fold_one(&ctx, witness_evals[4 * i], witness_evals[4 * i + 1]);
                let fw1 = E::fold_one(&ctx, witness_evals[4 * i + 2], witness_evals[4 * i + 3]);
                let fa0 = E::fold_one(&ctx, factor_evals[4 * i], factor_evals[4 * i + 1]);
                let fa1 = E::fold_one(&ctx, factor_evals[4 * i + 2], factor_evals[4 * i + 3]);

                acc.add_constant_product(fw0, fa0);
                acc.add_quadratic_product(fw1 - fw0, fa1 - fa0);

                witness_evals[2 * i] = fw0;
                witness_evals[2 * i + 1] = fw1;
                factor_evals[2 * i] = fa0;
                factor_evals[2 * i + 1] = fa1;
            }
            witness_evals.truncate(half);
            factor_evals.truncate(half);
            acc.finish()
        }
        DenseEorFactor::Lazy(_) => {
            unreachable!("lazy factors are handled before the materialized fused path")
        }
        DenseEorFactor::Shared(shared) => {
            let input_f: &[E] = shared;
            let mut folded_f = Vec::<E>::with_capacity(half);
            let mut acc = A::zero();
            for i in 0..quarter {
                let fw0 = E::fold_one(&ctx, witness_evals[4 * i], witness_evals[4 * i + 1]);
                let fw1 = E::fold_one(&ctx, witness_evals[4 * i + 2], witness_evals[4 * i + 3]);
                let fa0 = E::fold_one(&ctx, input_f[4 * i], input_f[4 * i + 1]);
                let fa1 = E::fold_one(&ctx, input_f[4 * i + 2], input_f[4 * i + 3]);

                acc.add_constant_product(fw0, fa0);
                acc.add_quadratic_product(fw1 - fw0, fa1 - fa0);

                witness_evals[2 * i] = fw0;
                witness_evals[2 * i + 1] = fw1;
                folded_f.push(fa0);
                folded_f.push(fa1);
            }
            witness_evals.truncate(half);
            *factor = DenseEorFactor::Owned(folded_f);
            acc.finish()
        }
    }
}
