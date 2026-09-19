//! Isolated, default-off input boundary for a future resident coefficient backend.
//! No challenger/state mutation and no expanded linear-weight table. The caller
//! supplies the already-folded alpha/linear/equality state used by the CPU call.
use super::*;

#[derive(Clone, Copy)]
pub(super) enum Input<'a, E: FieldCore> {
    Coefficients {
        values: &'a [E],
        rho: E,
    },
    Compact {
        values: &'a [i8],
        r0: E,
        r1: E,
        basis: usize,
    },
}

/// All slices remain borrowed for one synchronous call. In particular linear
/// terms retain their existing sparse/packing/source factorization.
pub(super) struct Request<'a, E: FieldCore> {
    pub(super) input: Input<'a, E>,
    pub(super) lanes: usize,
    pub(super) input_coefficients: usize,
    pub(super) alpha: &'a [E],
    pub(super) lane_weights: &'a [E],
    pub(super) eq_first: &'a [E],
    pub(super) eq_second: &'a [E],
    linear: &'a PreparedProverLinearTerms<E>,
    pub(super) skip_linear: bool,
}

pub(super) struct Output<E: FieldCore> {
    pub(super) witness: Vec<E>,
    pub(super) norm: NormRoundTerms<E>,
    pub(super) relation: [E; 3],
}

impl<'a, E: FieldCore + FromPrimitiveInt + HasUnreducedOps> Request<'a, E> {
    pub(super) fn coefficients(
        prover: &'a RelationRangeImageProver<E>,
        values: &'a [E],
        next_alpha: &'a [E],
        rho: E,
    ) -> Result<Self, AkitaError> {
        if !prover.in_coefficient_round() || prover.current_coefficient_width() < 2 {
            return Err(AkitaError::InvalidInput("stage2 coefficient phase".into()));
        }
        let (eq_first, eq_second) = prover.split_eq.remaining_eq_tables();
        let request = Self {
            input: Input::Coefficients { values, rho },
            lanes: prover.live_lane_count,
            input_coefficients: prover.common_alpha_factor.len(),
            alpha: next_alpha,
            lane_weights: &prover.relation_lane_weights,
            eq_first,
            eq_second,
            linear: &prover.linear_terms,
            skip_linear: prover.can_skip_norm_linear_coeff(),
        };
        request.validate()?;
        Ok(request)
    }

    pub(super) fn compact(
        prover: &'a RelationRangeImageProver<E>,
        values: &'a [i8],
        alpha_round2: &'a [E],
        linear_round2: &'a PreparedProverLinearTerms<E>,
        r0: E,
        r1: E,
    ) -> Result<Self, AkitaError> {
        if prover.coefficient_bits() <= 2 {
            return Err(AkitaError::InvalidInput(
                "stage2 compact coefficient phase".into(),
            ));
        }
        let (eq_first, eq_second) = prover.split_eq.remaining_eq_tables();
        let request = Self {
            input: Input::Compact {
                values,
                r0,
                r1,
                basis: prover.b,
            },
            lanes: prover.live_lane_count,
            input_coefficients: prover.common_alpha_factor.len(),
            alpha: alpha_round2,
            lane_weights: &prover.relation_lane_weights,
            eq_first,
            eq_second,
            linear: linear_round2,
            skip_linear: prover.can_skip_norm_linear_coeff(),
        };
        request.validate()?;
        Ok(request)
    }

    pub(super) fn output_coefficients(&self) -> usize {
        self.alpha.len()
    }

    /// The representation stays factored: evaluate just the requested adjacent
    /// pair. No source vector is duplicated and no lane×coefficient t is built.
    pub(super) fn linear_pair(&self, lane: usize, left: usize) -> Result<(E, E), AkitaError> {
        if lane >= self.lanes || left % 2 != 0 || left >= self.alpha.len().saturating_sub(1) {
            return Err(AkitaError::InvalidInput(
                "stage2 linear pair address".into(),
            ));
        }
        let index = akita_error::checked::product([lane, self.alpha.len()])
            .ok_or_else(|| AkitaError::InvalidInput("stage2 checked size overflow".into()))?
            .checked_add(left)
            .ok_or_else(|| AkitaError::InvalidInput("stage2 address overflow".into()))?;
        Ok(self.linear.pair_from_flat_index(index, self.alpha.len()))
    }

    pub(super) fn validate(&self) -> Result<(), AkitaError> {
        let bad = || AkitaError::InvalidInput("stage2 resident request shape".into());
        let divisor = match &self.input {
            Input::Coefficients { .. } => 2,
            Input::Compact { .. } => 4,
        };
        if self.lanes == 0
            || !self.input_coefficients.is_power_of_two()
            || self.input_coefficients < 2 * divisor
            || self.alpha.len() != self.input_coefficients / divisor
            || self.lane_weights.len() < self.lanes
            || !self.lane_weights.len().is_power_of_two()
            || !self.eq_first.len().is_power_of_two()
            || !self.eq_second.len().is_power_of_two()
        {
            return Err(bad());
        }
        let input_len = akita_error::checked::product([self.lanes, self.input_coefficients])
            .ok_or_else(|| AkitaError::InvalidInput("stage2 checked size overflow".into()))?;
        let output_len = akita_error::checked::product([self.lanes, self.alpha.len()])
            .ok_or_else(|| AkitaError::InvalidInput("stage2 checked size overflow".into()))?;
        let eq_len = akita_error::checked::product([self.eq_first.len(), self.eq_second.len()])
            .ok_or_else(|| AkitaError::InvalidInput("stage2 checked size overflow".into()))?;
        let padded_pairs =
            akita_error::checked::product([self.lane_weights.len(), self.alpha.len() / 2])
                .ok_or_else(|| AkitaError::InvalidInput("stage2 checked size overflow".into()))?;
        if eq_len != padded_pairs {
            return Err(bad());
        }
        match &self.input {
            Input::Coefficients { values, .. } => {
                if values.len() != input_len {
                    return Err(bad());
                }
            }
            Input::Compact { values, basis, .. } => {
                if !matches!(*basis, 4 | 8) || values.len() != input_len {
                    return Err(bad());
                }
                let half = (*basis / 2) as i8;
                if values.iter().any(|&v| v < -half || v >= half) {
                    return Err(bad());
                }
            }
        }
        self.linear.validate_len(output_len)?;
        Ok(())
    }
}

/// Independent scalar expression oracle. This intentionally does not use CPU
/// fused lookup/reduction helpers; it is not a proposed performance backend.
pub(super) fn scalar<E: FieldCore + FromPrimitiveInt + HasUnreducedOps>(
    request: &Request<'_, E>,
) -> Result<Output<E>, AkitaError> {
    request.validate()?; // every admission check precedes allocation or writes
    let next = request.output_coefficients();
    let mut witness = vec![
        E::zero();
        akita_error::checked::product([request.lanes, next]).ok_or_else(
            || AkitaError::InvalidInput("stage2 checked size overflow".into())
        )?
    ];
    for lane in 0..request.lanes {
        for coefficient in 0..next {
            let base = lane * request.input_coefficients;
            witness[lane * next + coefficient] = match request.input {
                Input::Coefficients { values, rho } => {
                    let left = values[base + 2 * coefficient];
                    left + rho * (values[base + 2 * coefficient + 1] - left)
                }
                Input::Compact { values, r0, r1, .. } => {
                    let i = base + 4 * coefficient;
                    let a = E::from_i64(values[i] as i64);
                    let b = E::from_i64(values[i + 1] as i64);
                    let c = E::from_i64(values[i + 2] as i64);
                    let d = E::from_i64(values[i + 3] as i64);
                    let lo = a + r0 * (b - a);
                    let hi = c + r0 * (d - c);
                    lo + r1 * (hi - lo)
                }
            };
        }
    }
    let mut norm = [E::zero(); 3];
    let mut relation = [E::zero(); 3];
    for lane in 0..request.lanes {
        for pair in 0..next / 2 {
            let left = 2 * pair;
            let j = lane * (next / 2) + pair;
            let e = request.eq_first[j % request.eq_first.len()]
                * request.eq_second[j / request.eq_first.len()];
            let w0 = witness[lane * next + left];
            let dw = witness[lane * next + left + 1] - w0;
            norm[0] += e * w0 * (w0 + E::one());
            if !request.skip_linear {
                norm[1] += e * dw * (w0 + w0 + E::one());
            }
            norm[2] += e * dw * dw;
            let (t0, t1) = request.linear_pair(lane, left)?;
            let p0 = request.alpha[left] * request.lane_weights[lane] + t0;
            let p1 = request.alpha[left + 1] * request.lane_weights[lane] + t1;
            relation[0] += w0 * p0;
            relation[1] += w0 * (p1 - p0) + dw * p0;
            relation[2] += dw * (p1 - p0);
        }
    }
    let norm = if request.skip_linear {
        NormRoundTerms::SkipLinear([norm[0], norm[2]])
    } else {
        NormRoundTerms::Full(norm)
    };
    Ok(Output {
        witness,
        norm,
        relation,
    })
}
