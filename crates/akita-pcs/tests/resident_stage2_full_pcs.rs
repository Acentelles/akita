//! Full production-layout opening parity; no hand-built Stage2 witness or descriptor.
#![cfg(all(feature = "resident-stage2-observer", feature = "schedules-default"))]

use akita_config::proof_optimized::fp64;
use akita_field::LiftBase;
use akita_pcs::AkitaCommitmentScheme;
use akita_prover::{
    ComputeBackendSetup, CpuBackend, DensePoly, MultilinearPolynomial, SelectedProverOpeningData,
    UniformProverStack,
};
use akita_serialization::{AkitaDeserialize, AkitaSerialize};
use akita_transcript::{sample_ext_challenge, AkitaTranscript};
use akita_types::{
    lagrange_weights, AkitaBatchedProof, BasisMode, GroupBatchStatement, OpeningClaims,
    PolynomialGroupClaims,
};

type B = fp64::Field;
type E = fp64::ExtensionField;
type Scheme = AkitaCommitmentScheme<fp64::Dense>;
const NV: usize = 20; // Existing shipped full-opening fp64 Dense fixture, not a synthetic catalog.
const LABEL: &[u8] = b"resident-stage2/full-pcs/v1";

fn next(t: &mut AkitaTranscript<B>) -> E {
    sample_ext_challenge::<B, E, _>(t, b"resident-stage2/parity-next")
}

fn fixture(seed: u64) {
    let evals: Vec<B> = (0..1usize << NV)
        .map(|i| {
            let mut x = (i as u64).wrapping_add(seed);
            x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            B::from_u64(x ^ (x >> 31))
        })
        .collect();
    let poly: MultilinearPolynomial<B, u8> = MultilinearPolynomial::dense(
        DensePoly::from_field_evals(NV, &evals).expect("dense polynomial"),
    );
    let point: Vec<E> = (0..NV)
        .map(|i| {
            E::new(
                B::from_u64(seed + 3 * i as u64 + 1),
                B::from_u64(7 * i as u64 + 9),
            )
        })
        .collect();
    // Independent extension-field bit-MLE; no prover folding helper supplies expected claims.
    let weights = lagrange_weights::<E>(&point).expect("bit-MLE weights");
    let expected = weights
        .iter()
        .zip(&evals)
        .fold(E::from_u64(0), |sum, (&w, &v)| sum + w * E::lift_base(v));
    drop(weights);
    drop(evals);
    let setup = Scheme::setup_prover(NV, 1).expect("production catalog setup");
    let prepared = CpuBackend::DEFAULT
        .prepare_setup(&setup)
        .expect("prepared setup");
    let stack =
        UniformProverStack::uniform(&CpuBackend::DEFAULT, &prepared, setup.expanded.as_ref())
            .expect("CPU commitment and ring-switch stack");
    let verifier_setup = Scheme::setup_verifier(&setup).expect("verifier setup");
    let committed = Scheme::commit::<_, _>(
        &setup,
        std::slice::from_ref(&poly),
        &stack,
        akita_prover::GroupContext::scheduler_without_precommitted_groups(),
    )
    .expect("commit real polynomial");
    let poly_refs = [&poly];
    let input = || {
        let claims = OpeningClaims::from_groups(vec![PolynomialGroupClaims::new(
            point.clone(),
            vec![E::from_u64(0)],
            committed.committed_group.clone(),
        )
        .expect("prover group")])
        .expect("prover claims");
        SelectedProverOpeningData::from_committed_claims::<fp64::Dense>(
            claims,
            vec![committed.hint.clone()],
            vec![&poly_refs[..]],
        )
        .expect("checked selected opening")
    };
    assert!(akita_prover::stage2_observer::take().is_empty());
    let mut cpu_transcript = AkitaTranscript::<B>::new(LABEL);
    let cpu = Scheme::batched_prove(
        &setup,
        input(),
        &stack,
        &mut cpu_transcript,
        BasisMode::Lagrange,
    )
    .expect("CPU complete opening");
    assert!(
        akita_prover::stage2_observer::take().is_empty(),
        "CPU default must not use resident policy"
    );
    let gpu_input = input();
    let selection = gpu_input.selection();
    let mut resident_transcript = AkitaTranscript::<B>::new(LABEL);
    let resident = Scheme::batched_prove_resident_stage2(
        &setup,
        gpu_input,
        &stack,
        &mut resident_transcript,
        BasisMode::Lagrange,
    )
    .expect("resident supported folds and preselected CPU remainder");
    let routes = akita_prover::stage2_observer::take();
    assert_eq!(routes.len(), resident.recursive_folds.len() + 1);
    for (level, route) in routes.iter().enumerate() {
        assert_eq!(route.level, level);
        assert!(route.lanes > 0 && route.columns.is_power_of_two());
        assert!(route.domain.is_power_of_two() && route.domain >= route.lanes * route.columns);
        assert!(
            route.declined.is_none(),
            "all five scheduled folds are admitted"
        );
        assert!([4, 8, 16, 32, 64].contains(&route.basis));
        assert_eq!(route.compact_entries, 1);
        assert_eq!(route.exports, 1);
        assert_eq!(route.advances, route.columns.ilog2() as usize - 3);
        eprintln!("full_pcs_route seed={seed} level={} basis={} columns={} lanes={} domain={} compression_layers={} declined={:?} entry={} advance={} export={}",
            route.level, route.basis, route.columns, route.lanes, route.domain, route.compression_layers,
            route.declined, route.compact_entries, route.advances, route.exports);
    }
    assert!(
        routes[0].declined.is_none()
            && routes[0].compression_layers > 0
            && routes[0].negative_binary_intervals > 0,
        "actual compressed root must execute resident Stage2"
    );
    assert!(
        routes.iter().skip(1).any(|r| r.declined.is_none()
            && r.compression_layers > 0
            && r.negative_binary_intervals > 0),
        "an actual compressed recursive suffix must execute resident Stage2"
    );
    assert_eq!(routes.len(), 5, "shipped NV20 root plus four suffix folds");
    assert_eq!(
        routes.iter().map(|r| r.basis).collect::<Vec<_>>(),
        [8, 8, 16, 16, 32]
    );
    let (mut cpu_bytes, mut resident_bytes) = (Vec::new(), Vec::new());
    cpu.serialize_compressed(&mut cpu_bytes)
        .expect("CPU serialization");
    resident
        .serialize_compressed(&mut resident_bytes)
        .expect("resident serialization");
    assert_eq!(cpu_bytes, resident_bytes, "complete PCS proof bytes");
    assert_eq!(next(&mut cpu_transcript), next(&mut resident_transcript));
    let mut verifier_next = Vec::new();
    for bytes in [&cpu_bytes, &resident_bytes] {
        let proof =
            AkitaBatchedProof::<B, E>::deserialize_compressed(&bytes[..], &resident.shape())
                .expect("checked proof decode");
        let claims = OpeningClaims::from_groups(vec![PolynomialGroupClaims::new(
            point.clone(),
            vec![expected],
            &committed.committed_group,
        )
        .expect("verifier group")])
        .expect("verifier claims");
        let mut vt = AkitaTranscript::<B>::new(LABEL);
        Scheme::batched_verify(
            &proof,
            &verifier_setup,
            &mut vt,
            GroupBatchStatement::new(selection, claims).expect("statement"),
            BasisMode::Lagrange,
        )
        .expect("original verifier accepts independent expected opening");
        verifier_next.push(next(&mut vt));
    }
    assert_eq!(verifier_next[0], verifier_next[1]);
    eprintln!("full_pcs_parity seed={seed} all_folds=resident folds=5 cpu_first_fold_wide=retained proof_bytes=equal verified=2 next_challenge=equal");
}

#[test]
fn full_pcs_compressed_root_and_suffix_match_cpu_proof_verifier_and_transcript() {
    std::thread::Builder::new()
        .stack_size(256 * 1024 * 1024)
        .spawn(|| {
            fixture(0x4a31);
            fixture(0x5b73);
        })
        .expect("large-stack fixture thread")
        .join()
        .expect("full PCS fixture");
}
