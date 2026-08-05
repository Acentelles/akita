//! Native A/B benchmark: Poseidon2 (alpha = 3 and alpha = 7) against the
//! Blake2b baseline, in one process.
//!
//! Deliberately harness-free (no criterion dev-dependency, so the shared
//! `Cargo.lock` is untouched). Absolute numbers from a managed environment are
//! not trustworthy; only same-session ratios are. Everything below is measured
//! in one process so the ratios are meaningful.
//!
//! Run with:
//!
//! ```text
//! CARGO_TARGET_DIR=~/.cache/akita-sponge-target \
//!   cargo test --release -p akita-transcript \
//!   --features transcript-poseidon2 --test poseidon2_bench \
//!   -- --ignored --nocapture --test-threads=1
//! ```
//!
//! Both `transcript-blake2b` (default) and `transcript-poseidon2` must be on so
//! all three sponges are nameable in the same binary. `TranscriptSponge` still
//! resolves to Blake2b in that build; the Poseidon2 sponges are named directly.

#![cfg(all(feature = "transcript-blake2b", feature = "transcript-poseidon2"))]
#![allow(missing_docs)]

use std::hint::black_box;
use std::time::{Duration, Instant};

use akita_field::Prime128Offset275;
use akita_transcript::poseidon2::{
    Alpha3, Alpha7, Fp, Poseidon2, Poseidon2ByteSponge, Poseidon2Instance, WIDTH,
};
use akita_transcript::AkitaTranscript;
use spongefish::DuplexSpongeInterface;

type F = Prime128Offset275;

const PERM_ITERS: usize = 200_000;
const SPONGE_ITERS: usize = 20_000;

fn bench<T>(label: &str, iters: usize, mut f: impl FnMut() -> T) -> Duration {
    // Warm up so the first-call OnceLock init is not attributed to the loop.
    for _ in 0..iters / 20 + 1 {
        black_box(f());
    }
    let start = Instant::now();
    for _ in 0..iters {
        black_box(f());
    }
    let elapsed = start.elapsed();
    let per = elapsed.as_secs_f64() / iters as f64;
    println!(
        "{label:<44} {:>10.1} ns/op   {:>12.0} op/s",
        per * 1e9,
        1.0 / per
    );
    elapsed
}

#[test]
#[ignore = "benchmark; run explicitly with --ignored --nocapture --release"]
fn permutation_throughput() {
    println!("\n--- permutation (t = 12), {PERM_ITERS} iterations ---");

    let mut state: [Fp; WIDTH] =
        core::array::from_fn(|i| <Fp as akita_field::FromPrimitiveInt>::from_u64(i as u64));

    let a3 = bench("Poseidon2 alpha=3 (R_F=8, R_P=42)", PERM_ITERS, || {
        Poseidon2::<Alpha3>::permute(&mut state);
        state[0]
    });
    let a7 = bench("Poseidon2 alpha=7 (R_F=8, R_P=22)", PERM_ITERS, || {
        Poseidon2::<Alpha7>::permute(&mut state);
        state[0]
    });

    println!(
        "\nalpha=3 / alpha=7 permutation time ratio: {:.3}x",
        a3.as_secs_f64() / a7.as_secs_f64()
    );
    println!(
        "S-boxes: alpha=3 {} (deg 3), alpha=7 {} (deg 7)",
        Poseidon2::<Alpha3>::SBOX_COUNT,
        Poseidon2::<Alpha7>::SBOX_COUNT
    );
    println!(
        "field mults per permutation, S-box only: alpha=3 {}, alpha=7 {}",
        2 * Poseidon2::<Alpha3>::SBOX_COUNT,
        4 * Poseidon2::<Alpha7>::SBOX_COUNT
    );
    println!(
        "field mults per permutation, internal linear layer: alpha=3 {}, alpha=7 {}",
        WIDTH * Alpha3::ROUNDS_P,
        WIDTH * Alpha7::ROUNDS_P
    );
}

/// Duplex throughput on a payload shaped like a real transcript absorb.
#[test]
#[ignore = "benchmark; run explicitly with --ignored --nocapture --release"]
fn sponge_throughput() {
    println!("\n--- duplex: absorb 256 B, squeeze 32 B, {SPONGE_ITERS} iterations ---");
    let payload: Vec<u8> = (0..256u32).map(|i| i as u8).collect();

    let blake = bench("Blake2b512 (spongefish, baseline)", SPONGE_ITERS, || {
        let mut sponge = spongefish::instantiations::Blake2b512::default();
        sponge.absorb(&payload);
        sponge.squeeze_array::<32>()
    });
    let a3 = bench("Poseidon2 alpha=3 byte sponge", SPONGE_ITERS, || {
        let mut sponge = Poseidon2ByteSponge::<Alpha3>::default();
        sponge.absorb(&payload);
        sponge.squeeze_array::<32>()
    });
    let a7 = bench("Poseidon2 alpha=7 byte sponge", SPONGE_ITERS, || {
        let mut sponge = Poseidon2ByteSponge::<Alpha7>::default();
        sponge.absorb(&payload);
        sponge.squeeze_array::<32>()
    });

    println!("\nrelative to Blake2b (higher = slower):");
    println!(
        "  Poseidon2 alpha=3: {:.1}x    Poseidon2 alpha=7: {:.1}x    alpha3/alpha7: {:.3}x",
        a3.as_secs_f64() / blake.as_secs_f64(),
        a7.as_secs_f64() / blake.as_secs_f64(),
        a3.as_secs_f64() / a7.as_secs_f64()
    );
}

/// Full `AkitaTranscript` path, which is what the protocol actually pays.
#[test]
#[ignore = "benchmark; run explicitly with --ignored --nocapture --release"]
fn transcript_throughput() {
    println!("\n--- AkitaTranscript round: absorb 256 B + squeeze scalar, {SPONGE_ITERS} iterations ---");
    let payload: Vec<u8> = (0..256u32).map(|i| i as u8).collect();

    fn run<S>(payload: &[u8]) -> F
    where
        S: Default + DuplexSpongeInterface<U = u8> + Send + 'static,
    {
        let mut transcript =
            AkitaTranscript::<F, S>::new_prover(b"bench/session", b"bench/instance");
        transcript.absorb_bytes(akita_transcript::label!("bench"), payload);
        transcript.squeeze_scalar(akita_transcript::label!("bench"))
    }

    let blake = bench("Blake2b512 transcript round", SPONGE_ITERS, || {
        run::<spongefish::instantiations::Blake2b512>(&payload)
    });
    let a3 = bench("Poseidon2 alpha=3 transcript round", SPONGE_ITERS, || {
        run::<Poseidon2ByteSponge<Alpha3>>(&payload)
    });
    let a7 = bench("Poseidon2 alpha=7 transcript round", SPONGE_ITERS, || {
        run::<Poseidon2ByteSponge<Alpha7>>(&payload)
    });

    println!("\nrelative to Blake2b (higher = slower):");
    println!(
        "  Poseidon2 alpha=3: {:.1}x    Poseidon2 alpha=7: {:.1}x    alpha3/alpha7: {:.3}x",
        a3.as_secs_f64() / blake.as_secs_f64(),
        a7.as_secs_f64() / blake.as_secs_f64(),
        a3.as_secs_f64() / a7.as_secs_f64()
    );
}
