//! Host driver that compiles the Jolt guest program in
//! `profile/akita-recursion/guest`, feeds it the
//! [`akita_recursion_glue::AkitaJoltInputs`] blob produced by
//! `profile/akita-recursion/artifact`, and traces/proves the Akita verifier.
//!
//! fp64 port: `--preset bound18` (default) drives the `akita_verify`
//! monomorphization (`fp64::D128FullBound18`, Aerie main-group proxy);
//! `--preset bound6` drives `akita_verify_bound6` (`fp64::D128FullBound6`,
//! Aerie range-group proxy). Blobs are synthetic single-group proxies, not
//! Aerie/Falcon results.
//!
//! Per-marker cycle counts emitted by the guest's
//! `start_cycle_tracking` / `end_cycle_tracking` calls (including the
//! span-bridge markers, see guest/src/lib.rs) are forwarded through Jolt's
//! `tracing` infrastructure; we initialize a tracing subscriber here so they
//! show up on stdout.

#![allow(missing_docs)]

use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use akita_config::proof_optimized::fp64;
use akita_config::{CommitmentConfig, RecursiveCommitmentConfig};
use akita_recursion_glue::{AkitaJoltInputs, MAX_JOLT_BLOB_BYTES};
use akita_transcript::AkitaTranscript;
use akita_types::{BasisMode, SetupContributionMode};
use akita_verifier::batched_verify;
use clap::{Parser, ValueEnum};
use tracing::info;
use tracing_subscriber::EnvFilter;

const TRUSTED_BENCHMARK_ARTIFACT_ENV: &str = "AKITA_RECURSION_TRUSTED_BENCHMARK_ARTIFACT";
type F = fp64::Field;
type E = fp64::ExtensionField;
const D: usize = 128;
type CfgMain = fp64::D128FullBound18;
type CfgRange = fp64::D128FullBound6;

const _: () = {
    assert!(D == <CfgMain as CommitmentConfig>::D);
    assert!(D == <CfgRange as CommitmentConfig>::D);
};

/// Guest monomorphization to run; must match the blob's artifact preset.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum PresetArg {
    /// `fp64::D128FullBound18` — guest fn `akita_verify`.
    Bound18,
    /// `fp64::D128FullBound6` — guest fn `akita_verify_bound6`.
    Bound6,
}

#[derive(Debug, Parser)]
#[command(
    about = "Prove the Akita fp64 verifier inside Jolt and report cycle counts",
    long_about = None
)]
struct Args {
    /// Path to the verifier-input blob produced by the `artifact` binary
    /// (`profile/akita-recursion/artifact`).
    #[arg(long, default_value = "target/akita_recursion_inputs.bin")]
    input: PathBuf,

    /// fp64 preset (must match the artifact's `--preset`).
    #[arg(long, value_enum, default_value_t = PresetArg::Bound18)]
    preset: PresetArg,

    /// Directory used by Jolt for per-program build artifacts.
    #[arg(long, default_value = "/tmp/akita-recursion-targets")]
    target_dir: String,

    /// Trace file path for `--trace-only`; defaults to
    /// `<target-dir>/akita_verify.trace`.
    #[arg(long)]
    trace_output: Option<PathBuf>,

    /// Only trace the guest (skips the ~minute-long Jolt prover step).
    /// Useful when iterating on guest panics with `JOLT_BACKTRACE=full`.
    #[arg(long, default_value_t = false)]
    trace_only: bool,
}

fn run_native_guest(preset: PresetArg, blob: &[u8]) -> Result<(), String> {
    info!("running guest natively (sanity check)");
    let native_output = match preset {
        PresetArg::Bound18 => guest::akita_verify(blob),
        PresetArg::Bound6 => guest::akita_verify_bound6(blob),
    };
    info!(native_output, "native guest output");
    if native_output != 0 {
        return Err(format!(
            "native guest run reported failure code {native_output}"
        ));
    }
    Ok(())
}

fn path_to_utf8<'a>(path: &'a Path, context: &str) -> Result<&'a str, String> {
    match path.to_str() {
        Some(path) => Ok(path),
        None => Err(format!(
            "{context} must be valid UTF-8: `{}`",
            path.display()
        )),
    }
}

fn enable_trusted_benchmark_guest_build() {
    // The pinned Jolt SDK builds guest ELFs with a hard-coded `--features guest`.
    // This checked build-script cfg keeps plain `guest` strict while letting
    // this benchmark harness opt the RISC-V build into trusted setup decode.
    std::env::set_var(TRUSTED_BENCHMARK_ARTIFACT_ENV, "1");
}

fn load_blob(input: &Path) -> Result<Vec<u8>, String> {
    let file = match File::open(input) {
        Ok(file) => file,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
            return Err(format!(
                "verifier-input blob not found at `{}`.\n\
                     Generate one first with `akita-recursion-artifact`. For example:\n\n\
                         AKITA_NUM_VARS=23 ./target/release/akita-recursion-artifact --preset bound18\n\n\
                     or, for a different blob path / arity:\n\n\
                         AKITA_NUM_VARS=26 AKITA_RECURSION_BLOB={} \\\n\
                             ./target/release/akita-recursion-artifact --preset bound6",
                input.display(),
                input.display()
            ));
        }
        Err(err) => return Err(format!("failed to open `{}`: {err}", input.display())),
    };
    let metadata = file
        .metadata()
        .map_err(|err| format!("failed to stat `{}`: {err}", input.display()))?;
    if !metadata.file_type().is_file() {
        return Err(format!(
            "verifier-input blob `{}` must be a regular file",
            input.display()
        ));
    }
    if metadata.len() > MAX_JOLT_BLOB_BYTES {
        return Err(format!(
            "verifier-input blob `{}` is {} bytes, exceeding max {} bytes",
            input.display(),
            metadata.len(),
            MAX_JOLT_BLOB_BYTES
        ));
    }
    let mut reader = file.take(MAX_JOLT_BLOB_BYTES + 1);
    let mut bytes = Vec::with_capacity(metadata.len() as usize);
    reader
        .read_to_end(&mut bytes)
        .map_err(|err| format!("failed to read `{}`: {err}", input.display()))?;
    if bytes.len() as u64 > MAX_JOLT_BLOB_BYTES {
        return Err(format!(
            "verifier-input blob `{}` exceeded max {} bytes while reading",
            input.display(),
            MAX_JOLT_BLOB_BYTES
        ));
    }
    Ok(bytes)
}

fn strict_host_preflight_with_config<Cfg>(blob: &[u8]) -> Result<(), String>
where
    Cfg: CommitmentConfig<Field = F, ExtField = E>,
{
    info!("strictly decoding and verifying verifier-input blob before trusted benchmark replay");
    let decoded = AkitaJoltInputs::<F, E, D>::read_from_bytes(blob)
        .map_err(|err| format!("strict input decode failed: {err}"))?;
    let mut transcript = AkitaTranscript::<F>::unbound_verifier(&decoded.transcript_domain);
    let claims = decoded
        .verifier_opening_batch()
        .map_err(|err| format!("blob opening batch is malformed: {err}"))?;
    match decoded.setup_contribution_mode {
        SetupContributionMode::Direct => batched_verify::<Cfg, _>(
            &decoded.proof,
            &decoded.verifier_setup,
            &mut transcript,
            claims,
            BasisMode::Lagrange,
            decoded.setup_contribution_mode,
        ),
        SetupContributionMode::Recursive => batched_verify::<RecursiveCommitmentConfig<Cfg>, _>(
            &decoded.proof,
            &decoded.verifier_setup,
            &mut transcript,
            claims,
            BasisMode::Lagrange,
            decoded.setup_contribution_mode,
        ),
    }
    .map_err(|err| format!("strict host verifier rejected input blob: {err}"))?;
    info!("strict host preflight OK");
    Ok(())
}

fn strict_host_preflight(preset: PresetArg, blob: &[u8]) -> Result<(), String> {
    match preset {
        PresetArg::Bound18 => strict_host_preflight_with_config::<CfgMain>(blob),
        PresetArg::Bound6 => strict_host_preflight_with_config::<CfgRange>(blob),
    }
}

fn run() -> Result<(), String> {
    let args = Args::parse();

    info!(input = %args.input.display(), preset = ?args.preset, "loading verifier-input blob");
    let blob = load_blob(&args.input)?;
    info!(bytes = blob.len(), "blob loaded");
    strict_host_preflight(args.preset, &blob)?;

    info!(target_dir = %args.target_dir, "compiling Akita verifier guest program");
    enable_trusted_benchmark_guest_build();

    if args.trace_only {
        // No explicit compile here: `trace_*_to_file` builds the guest ELF
        // itself (into /tmp/jolt-guest-targets), so a separate compile would
        // just duplicate a ~10-minute RISC-V build into a second target dir.
        info!("trace-only mode: skipping preprocessing and proof generation");
        run_native_guest(args.preset, &blob)?;

        let trace_path = args
            .trace_output
            .unwrap_or_else(|| PathBuf::from(&args.target_dir).join("akita_verify.trace"));
        info!(trace_file = %trace_path.display(), "tracing guest under emulator");
        let trace_path_str = path_to_utf8(&trace_path, "--trace-output")?;
        match args.preset {
            PresetArg::Bound18 => guest::trace_akita_verify_to_file(trace_path_str, &blob),
            PresetArg::Bound6 => guest::trace_akita_verify_bound6_to_file(trace_path_str, &blob),
        }
        info!("trace done");
        return Ok(());
    }

    // Full prove pipeline; per-preset because the SDK macro generates one
    // helper family per provable fn.
    let (output, panic_flag, is_valid, prover_secs, verifier_secs) = match args.preset {
        PresetArg::Bound18 => {
            let mut program = guest::compile_akita_verify(&args.target_dir);
            info!("running shared / prover / verifier preprocessing");
            let shared_preprocessing = guest::preprocess_shared_akita_verify(&mut program)
                .map_err(|err| format!("shared preprocessing failed: {err}"))?;
            let prover_preprocessing =
                guest::preprocess_prover_akita_verify(shared_preprocessing.clone());
            let verifier_preprocessing = guest::preprocess_verifier_akita_verify(
                shared_preprocessing,
                prover_preprocessing.generators.to_verifier_setup(),
                None,
            );
            let prove = guest::build_prover_akita_verify(program, prover_preprocessing);
            let verify = guest::build_verifier_akita_verify(verifier_preprocessing);

            run_native_guest(args.preset, &blob)?;

            info!("invoking Jolt prover");
            let now = Instant::now();
            let (output, proof, program_io) = prove(&blob);
            let prover_secs = now.elapsed().as_secs_f64();
            info!(prover_secs, "prover finished");
            let now = Instant::now();
            let is_valid = verify(&blob, output, program_io.panic, proof);
            let verifier_secs = now.elapsed().as_secs_f64();
            (output, program_io.panic, is_valid, prover_secs, verifier_secs)
        }
        PresetArg::Bound6 => {
            let mut program = guest::compile_akita_verify_bound6(&args.target_dir);
            info!("running shared / prover / verifier preprocessing");
            let shared_preprocessing = guest::preprocess_shared_akita_verify_bound6(&mut program)
                .map_err(|err| format!("shared preprocessing failed: {err}"))?;
            let prover_preprocessing =
                guest::preprocess_prover_akita_verify_bound6(shared_preprocessing.clone());
            let verifier_preprocessing = guest::preprocess_verifier_akita_verify_bound6(
                shared_preprocessing,
                prover_preprocessing.generators.to_verifier_setup(),
                None,
            );
            let prove = guest::build_prover_akita_verify_bound6(program, prover_preprocessing);
            let verify = guest::build_verifier_akita_verify_bound6(verifier_preprocessing);

            run_native_guest(args.preset, &blob)?;

            info!("invoking Jolt prover");
            let now = Instant::now();
            let (output, proof, program_io) = prove(&blob);
            let prover_secs = now.elapsed().as_secs_f64();
            info!(prover_secs, "prover finished");
            let now = Instant::now();
            let is_valid = verify(&blob, output, program_io.panic, proof);
            let verifier_secs = now.elapsed().as_secs_f64();
            (output, program_io.panic, is_valid, prover_secs, verifier_secs)
        }
    };

    info!(
        guest_output = output,
        guest_panic = panic_flag,
        prover_secs,
        verifier_secs,
        is_valid,
        "Jolt prover/verifier finished"
    );

    if !is_valid {
        return Err("Jolt verifier rejected the proof".to_string());
    }
    if output != 0 {
        return Err(format!("guest reported Akita-verify failure: {output}"));
    }
    info!("Akita-in-Jolt proof OK");
    Ok(())
}

fn main() -> ExitCode {
    let filter =
        EnvFilter::try_from_env("AKITA_RECURSION_LOG").unwrap_or_else(|_| EnvFilter::new("info"));
    tracing_subscriber::fmt().with_env_filter(filter).init();

    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("error: {err}");
            ExitCode::FAILURE
        }
    }
}
