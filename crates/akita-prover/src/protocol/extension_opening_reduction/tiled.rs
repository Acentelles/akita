use super::*;

/// Transparent tail state for a *virtually tiled* extension-opening term.
///
/// A grouped root lifts a shorter group's packed witness `W` (over the group's
/// tail prefix) into the joint sumcheck domain by constant extension in the
/// missing high tail variables. Physically that is the table `W` repeated
/// `copies` times end-to-end; this struct lets the prover run the joint
/// sumcheck without ever materializing the replication or the full-domain
/// transparent factor `A_eta`:
///
/// - While the group still has unbound variables (the first `group_tail_vars`
///   rounds), the tiled round messages decompose over the original domain: the
///   even/odd witness children repeat per high assignment, so
///   `sum_x W'(x)·A(x)` terms collapse onto `sum_lo W(lo)·B(lo)` with
///   `B(lo) = sum_hi A(lo, hi)`. By partition of unity of the equality kernel
///   in the summed coordinates, `B` is exactly the tensor equality factor of
///   the *truncated* tail point, so the caller pairs `W` with that small
///   factor table and this struct only tracks the full-tail folding state.
/// - Once the group is exhausted, the folded tiled witness is one value `w*`
///   replicated across the remaining domain: the quadratic round coefficient
///   vanishes identically and the constant coefficient is
///   `w* · sum_{x_0 = 0} A_k(x)`, whose factor sum has the closed form
///   `sum_j state_j · proj_eta(basis_j · (1 - tail_k))` (partition of unity in
///   every remaining coordinate except the bound low bit).
///
/// The folding state is the same coordinate-redistribution recurrence used by
/// [`tensor_equality_factor_eval_at_point`]: coordinate extraction is only
/// `F`-linear while the challenges are extension-valued, so the state carries
/// one accumulator per base coordinate. All quantities here are exact field
/// values; the emitted round coefficients equal the materialized-table
/// computation element-for-element, keeping proofs byte-identical.
#[derive(Debug, Clone)]
pub(in crate::protocol::extension_opening_reduction) struct TiledTailFactor<E: FieldCore> {
    /// Per-round transition matrices over base coordinates:
    /// `transitions[k].zero[src][dst] = coord_dst(basis_src · (1 - tail_k))`
    /// lifted into `E` (`one` uses `tail_k`).
    transitions: Vec<TiledTailTransition<E>>,
    /// Per-round even-half factor weights:
    /// `even_weights[k][src] = proj_eta(basis_src · (1 - tail_k))`.
    even_weights: Vec<Vec<E>>,
    /// `proj_eta(basis_j) = eq-weights of eta`; closes the state at the end.
    eta_weights: Vec<E>,
    /// Current width-vector folding state (starts at the coordinates of `1`).
    state: Vec<E>,
    /// Number of sumcheck rounds already folded into `state`.
    round: usize,
}

#[derive(Debug, Clone)]
struct TiledTailTransition<E: FieldCore> {
    zero: Vec<Vec<E>>,
    one: Vec<Vec<E>>,
}

impl<E: FieldCore> TiledTailFactor<E> {
    pub(in crate::protocol::extension_opening_reduction) fn new<F>(
        tail_point: &[E],
        eta: &[E],
    ) -> Result<Self, AkitaError>
    where
        F: FieldCore,
        E: ExtField<F>,
    {
        let (split_bits, width) = tensor_opening_split::<F, E>()?;
        if eta.len() != split_bits {
            return Err(AkitaError::InvalidSize {
                expected: split_bits,
                actual: eta.len(),
            });
        }
        // `remaining_len` shifts by the remaining round count.
        checked_table_len(tail_point.len())?;

        let eta_weights = EqPolynomial::evals(eta)?;
        let basis = (0..width)
            .map(|idx| {
                let mut coords = vec![F::zero(); width];
                coords[idx] = F::one();
                E::from_base_slice(&coords)
            })
            .collect::<Vec<_>>();
        let one_coords = E::one().to_base_vec();
        if one_coords.len() != width {
            return Err(AkitaError::InvalidSize {
                expected: width,
                actual: one_coords.len(),
            });
        }
        let state = one_coords.into_iter().map(E::lift_base).collect::<Vec<_>>();

        let mut transitions = Vec::with_capacity(tail_point.len());
        let mut even_weights = Vec::with_capacity(tail_point.len());
        for &tail in tail_point {
            let tail_zero = E::one() - tail;
            let tail_one = tail;
            let mut zero = vec![vec![E::zero(); width]; width];
            let mut one = vec![vec![E::zero(); width]; width];
            let mut even = Vec::with_capacity(width);
            for (src_idx, &basis_elem) in basis.iter().enumerate() {
                let zero_value = basis_elem * tail_zero;
                let one_value = basis_elem * tail_one;
                let zero_coords = zero_value.to_base_vec();
                let one_coords = one_value.to_base_vec();
                if zero_coords.len() != width || one_coords.len() != width {
                    return Err(AkitaError::InvalidSize {
                        expected: width,
                        actual: zero_coords.len().max(one_coords.len()),
                    });
                }
                for dst_idx in 0..width {
                    zero[src_idx][dst_idx] = E::lift_base(zero_coords[dst_idx]);
                    one[src_idx][dst_idx] = E::lift_base(one_coords[dst_idx]);
                }
                even.push(project_tensor_factor_value::<F, E>(
                    zero_value,
                    &eta_weights,
                    width,
                )?);
            }
            transitions.push(TiledTailTransition { zero, one });
            even_weights.push(even);
        }

        Ok(Self {
            transitions,
            even_weights,
            eta_weights,
            state,
            round: 0,
        })
    }

    pub(in crate::protocol::extension_opening_reduction) fn total_rounds(&self) -> usize {
        self.transitions.len()
    }

    /// Remaining virtual table length `2^{tail_vars - round}`.
    pub(in crate::protocol::extension_opening_reduction) fn remaining_len(&self) -> usize {
        1usize << (self.total_rounds() - self.round)
    }

    /// Closed-form even-half factor sum for the current round:
    /// `sum_{x: x_0 = 0} A_k(x) = sum_j state_j · proj_eta(basis_j · (1 - tail_k))`.
    pub(in crate::protocol::extension_opening_reduction) fn even_half_sum(&self) -> E {
        debug_assert!(self.round < self.total_rounds());
        self.state
            .iter()
            .zip(self.even_weights[self.round].iter())
            .fold(E::zero(), |acc, (&coeff, &weight)| acc + coeff * weight)
    }

    /// Fold the transparent tail state by one sumcheck challenge.
    pub(in crate::protocol::extension_opening_reduction) fn fold_in_place(&mut self, r_round: E) {
        debug_assert!(self.round < self.total_rounds());
        let transition = &self.transitions[self.round];
        let one_minus = E::one() - r_round;
        let width = self.state.len();
        let mut next = vec![E::zero(); width];
        for (src_idx, &src) in self.state.iter().enumerate() {
            if src == E::zero() {
                continue;
            }
            for (dst_idx, dst) in next.iter_mut().enumerate() {
                let step = transition.zero[src_idx][dst_idx] * one_minus
                    + transition.one[src_idx][dst_idx] * r_round;
                *dst += src * step;
            }
        }
        self.state = next;
        self.round += 1;
    }

    /// Fully folded factor value `A(rho)`; `None` before the final round.
    ///
    /// Equals [`tensor_equality_factor_eval_at_point`] at the joint challenge
    /// point (same recurrence, same exact arithmetic).
    pub(in crate::protocol::extension_opening_reduction) fn final_value(&self) -> Option<E> {
        (self.round == self.total_rounds()).then(|| {
            self.state
                .iter()
                .zip(self.eta_weights.iter())
                .fold(E::zero(), |acc, (&coord_eval, &eta_weight)| {
                    acc + coord_eval * eta_weight
                })
        })
    }
}
