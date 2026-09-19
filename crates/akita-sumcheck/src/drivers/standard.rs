//! Standard sumcheck transcript drivers.

use crate::traits::{
    FallibleSumcheckInstanceProver, SumcheckInstanceProver, SumcheckInstanceVerifier,
};
use crate::types::SumcheckProof;
use akita_error::AkitaError;
use akita_field::{CanonicalField, FieldCore};
use akita_serialization::AkitaSerialize;
use akita_transcript::labels;
use akita_transcript::Transcript;

/// Prove a single sumcheck using synchronous fallible operations.
///
/// The claim is absorbed first. Each accepted polynomial is absorbed and its
/// challenge sampled before ingestion. A failed compute or degree check absorbs
/// no polynomial for that round; failed ingestion leaves that round's challenge
/// in the transcript. Finalization is called only after all successful ingests.
/// No partial proof is returned, no retry occurs, and transcript/state rollback
/// is not attempted. Callers must discard a failed proof attempt.
///
/// # Errors
/// Propagates compute, ingest and finalization failures, or rejects a polynomial
/// above the declared degree bound.
#[tracing::instrument(skip_all, name = "prove_sumcheck")]
#[inline(never)]
pub fn prove_fallible_sumcheck<F, E, T, S, I>(
    instance: &mut I,
    transcript: &mut T,
    mut sample_challenge: S,
) -> Result<(SumcheckProof<E>, Vec<E>, E), AkitaError>
where
    F: FieldCore + CanonicalField,
    E: FieldCore + AkitaSerialize,
    T: Transcript<F>,
    S: FnMut(&mut T) -> E,
    I: FallibleSumcheckInstanceProver<E> + ?Sized,
{
    let num_rounds = instance.num_rounds();
    let mut claim = instance.input_claim();
    tracing::debug!(
        is_zero = claim.is_zero(),
        num_rounds,
        "prove_sumcheck input_claim"
    );
    transcript.append_serde(labels::ABSORB_SUMCHECK_CLAIM, &claim);

    let degree_bound = instance.degree_bound();
    let mut round_polys = Vec::with_capacity(num_rounds);
    let mut r = Vec::with_capacity(num_rounds);

    for round in 0..num_rounds {
        let _round_span = tracing::info_span!(
            "sumcheck_round",
            round,
            table_len = 1usize << (num_rounds - round)
        )
        .entered();
        let g = {
            let _s = tracing::info_span!("sumcheck_round_univariate").entered();
            instance.compute_round_univariate(round, claim)?
        };
        let round_sum = g.evaluate(&E::zero()) + g.evaluate(&E::one());
        debug_assert!(
            round_sum == claim,
            "sumcheck round {round} univariate does not match previous claim hint"
        );

        let compressed = g.compress();
        if compressed.degree() > degree_bound {
            return Err(AkitaError::InvalidInput(format!(
                "sumcheck round poly degree {} exceeds bound {}",
                compressed.degree(),
                degree_bound
            )));
        }

        transcript.append_serde(labels::ABSORB_SUMCHECK_ROUND, &compressed);
        let r_i = sample_challenge(transcript);
        r.push(r_i);

        claim = compressed.eval_from_hint(&claim, &r_i);
        {
            let _s = tracing::info_span!("sumcheck_round_fold").entered();
            instance.ingest_challenge(round, r_i)?;
        }
        round_polys.push(compressed);
    }

    instance.finalize()?;
    Ok((SumcheckProof { round_polys }, r, claim))
}

struct InfallibleProver<'a, I>(&'a mut I);
impl<E: FieldCore, I: SumcheckInstanceProver<E>> FallibleSumcheckInstanceProver<E>
    for InfallibleProver<'_, I>
{
    fn num_rounds(&self) -> usize {
        self.0.num_rounds()
    }
    fn degree_bound(&self) -> usize {
        self.0.degree_bound()
    }
    fn input_claim(&self) -> E {
        self.0.input_claim()
    }
    fn compute_round_univariate(
        &mut self,
        round: usize,
        claim: E,
    ) -> Result<akita_algebra::uni_poly::UniPoly<E>, AkitaError> {
        Ok(self.0.compute_round_univariate(round, claim))
    }
    fn ingest_challenge(&mut self, round: usize, challenge: E) -> Result<(), AkitaError> {
        self.0.ingest_challenge(round, challenge);
        Ok(())
    }
    fn finalize(&mut self) -> Result<(), AkitaError> {
        self.0.finalize();
        Ok(())
    }
}

/// Plain extension for standard sumcheck provers.
pub trait SumcheckInstanceProverExt<E>: SumcheckInstanceProver<E> + Sized
where
    E: FieldCore,
{
    /// Produce a sumcheck proof for a single instance.
    ///
    /// It returns the proof, the derived point `r`, and the final claimed value
    /// at `r`.
    ///
    /// # Errors
    ///
    /// Returns an error if any per-round polynomial exceeds the instance's degree bound.
    fn prove<F, T, S>(
        &mut self,
        transcript: &mut T,
        sample_challenge: S,
    ) -> Result<(SumcheckProof<E>, Vec<E>, E), AkitaError>
    where
        F: FieldCore + CanonicalField,
        T: Transcript<F>,
        E: AkitaSerialize,
        S: FnMut(&mut T) -> E,
    {
        prove_fallible_sumcheck::<F, E, T, S, _>(
            &mut InfallibleProver(self),
            transcript,
            sample_challenge,
        )
    }
}

impl<E, Inst> SumcheckInstanceProverExt<E> for Inst
where
    E: FieldCore,
    Inst: SumcheckInstanceProver<E>,
{
}

/// Plain extension for standard sumcheck verifiers.
pub trait SumcheckInstanceVerifierExt<E>: SumcheckInstanceVerifier<E> + Sized
where
    E: FieldCore,
{
    /// Verify a single-instance sumcheck proof.
    ///
    /// Returns the challenge point `r` on success.
    ///
    /// # Errors
    ///
    /// Returns [`AkitaError::InvalidProof`] if the final sumcheck claim does not
    /// match the oracle evaluation, or propagates any error from the per-round
    /// verification.
    #[tracing::instrument(skip_all, name = "verify_sumcheck")]
    #[inline(never)]
    fn verify<F, T, S>(
        &self,
        proof: &SumcheckProof<E>,
        transcript: &mut T,
        mut sample_challenge: S,
    ) -> Result<Vec<E>, AkitaError>
    where
        F: FieldCore + CanonicalField,
        T: Transcript<F>,
        E: AkitaSerialize,
        S: FnMut(&mut T) -> E,
    {
        let num_rounds = self.num_rounds();
        if proof.round_polys.len() != num_rounds {
            return Err(AkitaError::InvalidSize {
                expected: num_rounds,
                actual: proof.round_polys.len(),
            });
        }

        let mut claim = self.input_claim();
        tracing::debug!(
            is_zero = claim.is_zero(),
            num_rounds,
            "verify_sumcheck input_claim"
        );
        transcript.append_serde(labels::ABSORB_SUMCHECK_CLAIM, &claim);

        let degree_bound = self.degree_bound();
        let mut challenges = Vec::with_capacity(num_rounds);

        for poly in &proof.round_polys {
            if poly.degree() > degree_bound {
                return Err(AkitaError::InvalidInput(format!(
                    "sumcheck round poly degree {} exceeds bound {}",
                    poly.degree(),
                    degree_bound
                )));
            }

            transcript.append_serde(labels::ABSORB_SUMCHECK_ROUND, poly);
            let r_i = sample_challenge(transcript);
            challenges.push(r_i);
            claim = poly.eval_from_hint(&claim, &r_i);
        }

        check_sumcheck_output_claim(claim, self, &challenges)?;
        Ok(challenges)
    }
}

impl<E, Inst> SumcheckInstanceVerifierExt<E> for Inst
where
    E: FieldCore,
    Inst: SumcheckInstanceVerifier<E>,
{
}

/// Enforce the final sumcheck oracle equality for the provided challenge point.
///
/// This is useful when some prefix rounds are reconstructed outside the generic
/// verifier driver and the caller needs to check the final oracle value against
/// the full concatenated challenge vector.
///
/// # Errors
///
/// Returns any error produced by `verifier.expected_output_claim`, or
/// [`AkitaError::InvalidProof`] if the final claim does not match the oracle
/// evaluation at `challenges`.
pub fn check_sumcheck_output_claim<E, V>(
    final_claim: E,
    verifier: &V,
    challenges: &[E],
) -> Result<(), AkitaError>
where
    E: FieldCore + AkitaSerialize,
    V: SumcheckInstanceVerifier<E>,
{
    let expected = verifier.expected_output_claim(challenges)?;
    if final_claim != expected {
        tracing::error!(
            rounds = verifier.num_rounds(),
            degree_bound = verifier.degree_bound(),
            diff_is_zero = (final_claim - expected).is_zero(),
            "verify_sumcheck MISMATCH"
        );
        return Err(AkitaError::InvalidProof);
    }
    Ok(())
}

#[cfg(test)]
#[path = "fallible_tests.rs"]
mod fallible_tests;
#[cfg(test)]
#[path = "standard_original_test_oracle.rs"]
mod standard_original_test_oracle;
