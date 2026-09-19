//! Full Stage2 sumcheck parity; not a complete PCS/aggregate proof claim.
use super::super::resident_owned::ResidentRelationProver;
use super::*;
use akita_serialization::AkitaSerialize;
use akita_sumcheck::{prove_fallible_sumcheck, SumcheckInstanceProverExt};
use akita_transcript::{labels, AkitaTranscript, Transcript};
type Tr = AkitaTranscript<B>;
fn transcript() -> Tr {
    Tr::new(labels::DOMAIN_AKITA_PROTOCOL)
}
fn sample(t: &mut Tr) -> F {
    F::new(
        t.challenge_scalar(labels::CHALLENGE_SUMCHECK_ROUND),
        t.challenge_scalar(labels::CHALLENGE_SUMCHECK_ROUND),
    )
}
fn prover(
    basis: akita_types::BasisMode,
    full_norm: bool,
    additional_mode: usize,
    digit_basis: usize,
    padded: bool,
) -> RelationRangeImageProver<F> {
    let (linear, dense, c) =
        super::super::coefficient_packing_terms::tests::resident_semantic_fixture(basis);
    let lanes = dense.len() / c;
    let (mut p, compact) = fixture_with_linear(
        lanes,
        c.trailing_zeros() as usize,
        digit_basis,
        full_norm,
        Some((linear, dense)),
    );
    let mut point = (0..p.num_vars)
        .map(|i| field(i as u64 + 400))
        .collect::<Vec<_>>();
    if full_norm {
        point[2] = F::zero();
    }
    if padded {
        let high = field(777);
        point.push(high);
        let mut weights = p.relation_lane_weights;
        weights.resize(weights.len() * 2, F::zero());
        // Re-enter the real checked constructor with an extra zero-padded lane
        // bit and the corresponding equality-weighted range claim.
        p = RelationRangeImageProver::new(
            p.batching_coeff,
            compact.clone(),
            &point,
            p.range_image_evaluation * (F::one() - high),
            digit_basis,
            p.common_alpha_factor,
            weights,
            lanes,
            p.lane_bits + 1,
            c.trailing_zeros() as usize,
            p.relation_linear_claim,
            p.linear_terms,
            F::zero(),
            None,
        )
        .unwrap();
    }
    let additional = super::super::additional_terms::resident_additional::tests::fixture(
        &compact,
        &point,
        additional_mode,
    );
    p.input_claim += additional.input_claim();
    p.additional_relation_terms = Some(additional);
    p
}
#[test]
fn resident_owned_full_sumcheck_matches_cpu_proof_and_next_challenge() {
    let mut cases = 0;
    for basis in [
        akita_types::BasisMode::Lagrange,
        akita_types::BasisMode::Monomial,
    ] {
        for full_norm in [false, true] {
            for digit_basis in [4, 8] {
                for padded in [false, true] {
                    for mode in 0..4 {
                        let mut cpu = prover(basis, full_norm, mode, digit_basis, padded);
                        let mut gpu = ResidentRelationProver::new(prover(
                            basis,
                            full_norm,
                            mode,
                            digit_basis,
                            padded,
                        ))
                        .unwrap();
                        let (mut ct, mut gt) = (transcript(), transcript());
                        let expected = cpu.prove::<B, _, _>(&mut ct, sample).unwrap();
                        let actual =
                            prove_fallible_sumcheck::<B, F, _, _, _>(&mut gpu, &mut gt, sample)
                                .unwrap();
                        let (mut a, mut b) = (Vec::new(), Vec::new());
                        expected.0.serialize_compressed(&mut a).unwrap();
                        actual.0.serialize_compressed(&mut b).unwrap();
                        assert_eq!(a, b);
                        assert_eq!(expected, actual);
                        let completed = gpu.into_completed().unwrap();
                        assert_eq!(cpu.final_w_eval(), completed.final_w_eval());
                        assert_eq!(
                            cpu.expected_final_claim().unwrap(),
                            completed.expected_final_claim().unwrap()
                        );
                        assert_eq!(actual.2, completed.expected_final_claim().unwrap());
                        assert_eq!(sample(&mut ct), sample(&mut gt));
                        cases += 1;
                    }
                }
            }
        }
    }
    assert_eq!(cases, 64);
    println!("RESIDENT_STAGE2_PARITY cases=64 exact_proof_bytes=true final_witness=true final_claim=true next_challenge=true");
}
#[test]
fn resident_owned_static_shape_rejects_before_transcript_or_device_route() {
    let mut p = prover(akita_types::BasisMode::Lagrange, false, 2, 4, false);
    p.rounds_completed = 1;
    assert!(ResidentRelationProver::new(p).is_err());
    let mut p = prover(akita_types::BasisMode::Lagrange, false, 2, 4, false);
    p.b = 16;
    assert!(ResidentRelationProver::new(p).is_err());
    let mut p = prover(akita_types::BasisMode::Lagrange, false, 2, 4, false);
    p.num_vars = usize::BITS as usize;
    assert!(ResidentRelationProver::new(p).is_err());
    // Constructor has no transcript handle; all three failures occur before Owner::create.
}
