//! Ignored micro-benchmarks for the parallel-threshold tuning A/B protocol.
//!
//! Run with:
//! `cargo test -p akita-algebra --release --features parallel --test perf_micro -- --ignored --nocapture`
//!
//! Each test prints TSV rows (`label size ns_per_call`). Compare the output of
//! two binaries (old vs new thresholds) built in the same session; ratios only.

use akita_algebra::eq_poly::EqPolynomial;
use akita_algebra::poly::{fold_evals_in_place, multilinear_eval};
use akita_field::{FpExt4, FromPrimitiveInt, Prime32Offset99, RandomSampling};
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::time::Instant;

type F = Prime32Offset99;
type E = FpExt4<F>;

fn random_table(len: usize, seed: u64) -> Vec<E> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..len).map(|_| E::random(&mut rng)).collect()
}

fn random_point(nv: usize, seed: u64) -> Vec<E> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..nv).map(|_| E::random(&mut rng)).collect()
}

/// Median-of-runs timer; returns ns per call.
fn time_median<Fun: FnMut()>(reps: usize, mut f: Fun) -> u128 {
    let mut samples = Vec::with_capacity(reps);
    for _ in 0..reps {
        let t = Instant::now();
        f();
        samples.push(t.elapsed().as_nanos());
    }
    samples.sort_unstable();
    samples[samples.len() / 2]
}

#[test]
#[ignore]
fn perf_fold_per_size() {
    let r = random_point(1, 7)[0];
    for log_len in 9..=17u32 {
        let len = 1usize << log_len;
        let master = random_table(len, 42 + u64::from(log_len));
        let reps = (1usize << 24) / len + 3;
        let mut sink = E::zero();
        let ns = time_median(reps.min(64), || {
            let mut t = master.clone();
            fold_evals_in_place(&mut t, r);
            sink += t[0];
        });
        // Subtract an estimate of the clone cost measured separately.
        let clone_ns = time_median(reps.min(64), || {
            let t = master.clone();
            std::hint::black_box(&t);
        });
        println!("fold\t{len}\t{}\t{}", ns.saturating_sub(clone_ns), sink == E::zero());
    }
}

#[test]
#[ignore]
fn perf_fold_ladder() {
    // Full descent (production mixture) and tail-only descent (threshold zone).
    let r = random_point(1, 9)[0];
    for &start_log in &[20u32, 13u32] {
        let len = 1usize << start_log;
        let master = random_table(len, 1000 + u64::from(start_log));
        let reps = if start_log >= 20 { 5 } else { 200 };
        let ns = time_median(reps, || {
            let mut t = master.clone();
            while t.len() >= 2 {
                fold_evals_in_place(&mut t, r);
            }
            std::hint::black_box(&t);
        });
        println!("fold_ladder\t{len}\t{ns}");
    }
}

#[test]
#[ignore]
fn perf_eq_evals() {
    for nv in 10..=18usize {
        let point = random_point(nv, 77 + nv as u64);
        let reps = ((1usize << 22) >> nv).max(3);
        let ns = time_median(reps.min(64), || {
            let t = EqPolynomial::evals(&point).unwrap();
            std::hint::black_box(&t);
        });
        println!("eq_evals\t{nv}\t{ns}");
    }
}

#[test]
#[ignore]
fn perf_mle_eval() {
    for nv in 10..=17usize {
        let len = 1usize << nv;
        let table = random_table(len, 300 + nv as u64);
        let point = random_point(nv, 400 + nv as u64);
        let reps = ((1usize << 22) >> nv).max(3);
        let mut sink = E::zero();
        let ns = time_median(reps.min(64), || {
            sink += multilinear_eval(&table, &point).unwrap();
        });
        println!("mle_eval\t{nv}\t{ns}\t{}", sink == E::from_u64(0));
    }
}
