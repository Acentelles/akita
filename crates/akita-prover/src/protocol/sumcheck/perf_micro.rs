//! Ignored micro-benchmarks driving the stage-1/stage-2 sumcheck provers and
//! the dense extension-opening-reduction round kernels at production-like
//! fp32 shapes (`FpExt4<Prime32Offset99>`, D = 128, b = 8).
//!
//! Run with:
//! `cargo test -p akita-prover --release --features parallel perf_micro -- --ignored --nocapture`
//!
//! Output is TSV (`label round/size ns`). A/B protocol: build old and new
//! binaries in the same session, run interleaved, compare ratios only.

use super::akita_stage1_tree::AkitaStage1Prover;
use super::akita_stage2::AkitaStage2Prover;
use crate::protocol::extension_opening_reduction::{
    accumulate_dense_round, fold_dense_reduction_tables_in_place, fused_fold_and_accumulate,
};
use akita_field::{FpExt4, Prime32Offset99, RandomSampling};
use akita_sumcheck::SumcheckInstanceProver;
use akita_transcript::AkitaTranscript;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::time::Instant;

type F = Prime32Offset99;
type E = FpExt4<F>;

const B: usize = 8;
const RING_BITS: usize = 7; // D = 128
const COL_BITS: usize = 14;

fn random_digits(len: usize, seed: u64) -> Vec<i8> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..len).map(|_| rng.gen_range(-4i8..=3i8)).collect()
}

fn random_ext(len: usize, seed: u64) -> Vec<E> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..len).map(|_| E::random(&mut rng)).collect()
}

#[test]
#[ignore]
fn perf_stage1_full_run() {
    let num_vars = COL_BITS + RING_BITS;
    let live_x_cols = 1usize << COL_BITS;
    let w = random_digits(live_x_cols << RING_BITS, 11);
    let tau0 = random_ext(num_vars, 12);

    for rep in 0..3 {
        let prover =
            AkitaStage1Prover::<E>::new(&w, &tau0, B, live_x_cols, COL_BITS, RING_BITS).unwrap();
        let mut transcript = AkitaTranscript::<F>::new(b"perf/stage1");
        let t_total = Instant::now();
        let out = prover.prove::<F, _>(&mut transcript).unwrap();
        let ns = t_total.elapsed().as_nanos();
        std::hint::black_box(&out);
        if rep == 2 {
            println!("stage1_total\t-\t{ns}");
        }
    }
}

#[test]
#[ignore]
fn perf_stage2_full_run() {
    let num_vars = COL_BITS + RING_BITS;
    let live_x_cols = 1usize << COL_BITS;
    let y_len = 1usize << RING_BITS;
    let w = random_digits(live_x_cols * y_len, 21);
    let stage1_point = random_ext(num_vars, 22);
    let alpha_evals_y = random_ext(y_len, 23);
    let relation_matrix_col_evals = random_ext(live_x_cols, 24);
    let challenges = random_ext(num_vars, 25);
    let mut rng = StdRng::seed_from_u64(26);
    let batching_coeff = E::random(&mut rng);

    for rep in 0..3 {
        let mut prover = AkitaStage2Prover::<E>::new(
            batching_coeff,
            w.clone(),
            &stage1_point,
            E::zero(),
            B,
            alpha_evals_y.clone(),
            relation_matrix_col_evals.clone(),
            live_x_cols,
            COL_BITS,
            RING_BITS,
            E::zero(),
            None,
            E::zero(),
        )
        .unwrap();
        let t_total = Instant::now();
        let mut claim = prover.input_claim();
        for (round, &r) in challenges.iter().enumerate() {
            let t = Instant::now();
            let poly = prover.compute_round_univariate(round, claim);
            let ns_poly = t.elapsed().as_nanos();
            claim = poly.evaluate(&r);
            std::hint::black_box(&poly);
            let t = Instant::now();
            prover.ingest_challenge(round, r);
            let ns_fold = t.elapsed().as_nanos();
            if rep == 2 {
                println!("stage2_round\t{round}\t{ns_poly}\t{ns_fold}");
            }
        }
        if rep == 2 {
            println!("stage2_total\t-\t{}", t_total.elapsed().as_nanos());
        }
    }
}

#[test]
#[ignore]
fn perf_eor_dense_ladder() {
    let start_len = 1usize << 20;
    let witness_master = random_ext(start_len, 31);
    let factor_master = random_ext(start_len, 32);
    let coeff = random_ext(1, 33)[0];
    let rs = random_ext(24, 34);

    // Unfused: accumulate then fold, per round, down to 4 entries.
    for rep in 0..3 {
        let mut witness = witness_master.clone();
        let mut factor = factor_master.clone();
        let mut round = 0usize;
        let t_total = Instant::now();
        while witness.len() >= 4 {
            let t = Instant::now();
            let acc = accumulate_dense_round(&witness, &factor, coeff);
            let ns_acc = t.elapsed().as_nanos();
            std::hint::black_box(&acc);
            let t = Instant::now();
            fold_dense_reduction_tables_in_place(&mut witness, &mut factor, rs[round % rs.len()]);
            let ns_fold = t.elapsed().as_nanos();
            if rep == 2 {
                println!("eor_unfused\t{}\t{ns_acc}\t{ns_fold}", witness.len() * 2);
            }
            round += 1;
        }
        if rep == 2 {
            println!("eor_unfused_total\t-\t{}", t_total.elapsed().as_nanos());
        }
    }

    // Fused fold+accumulate ladder.
    for rep in 0..3 {
        let mut witness = witness_master.clone();
        let mut factor = factor_master.clone();
        let mut round = 0usize;
        let t_total = Instant::now();
        while witness.len() >= 4 {
            let t = Instant::now();
            let acc = fused_fold_and_accumulate(&mut witness, &mut factor, rs[round % rs.len()]);
            let ns = t.elapsed().as_nanos();
            std::hint::black_box(&acc);
            if rep == 2 {
                println!("eor_fused\t{}\t{ns}", witness.len() * 2);
            }
            round += 1;
        }
        if rep == 2 {
            println!("eor_fused_total\t-\t{}", t_total.elapsed().as_nanos());
        }
    }
}
