use super::*;

pub(super) fn original_prove<F, E, T, S, I>(
    instance: &mut I,
    transcript: &mut T,
    mut sample_challenge: S,
) -> Result<(SumcheckProof<E>, Vec<E>, E), AkitaError>
where
    F: FieldCore + CanonicalField,
    E: FieldCore + AkitaSerialize,
    T: Transcript<F>,
    S: FnMut(&mut T) -> E,
    I: SumcheckInstanceProver<E>,
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
            instance.compute_round_univariate(round, claim)
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
            instance.ingest_challenge(round, r_i);
        }
        round_polys.push(compressed);
    }

    instance.finalize();
    Ok((SumcheckProof { round_polys }, r, claim))
}
