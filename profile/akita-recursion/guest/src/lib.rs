//! Jolt guest program that deserializes a serialized Akita verifier input
//! bundle (from [`akita_recursion_glue::AkitaJoltInputs`]) and runs the
//! Akita batched verifier inside the Jolt RISC-V emulator.
//!
//! fp64 port: two monomorphizations are exposed, matching the Aerie-
//! representative presets produced by `../artifact`:
//!
//! - `akita_verify`        — `fp64::D128FullBound18` (Aerie main group).
//! - `akita_verify_bound6` — `fp64::D128FullBound6` (Aerie range group).
//!
//! Both operate on **synthetic single-group proxy** blobs, not Aerie/Falcon
//! results: only the verifier schedule is representative.
//!
//! Cycle attribution:
//!
//! - Three top-level markers wrap `deserialize_input`, `transcript_init`, and
//!   `akita_verify` exactly as in the original fp128 harness.
//! - A guest-local `tracing` subscriber bridges every span the Akita verifier
//!   already emits (`prepare_fold_replay`, `ring_switch_verifier*`,
//!   `prepare_relation_matrix_evaluator`, `setup_contribution`,
//!   `stage1_sumcheck`, `stage2_sumcheck`, `stage2_relation_matrix_eval`, ...)
//!   into Jolt cycle markers, giving per-stage, per-fold-level cycles with no
//!   changes to `akita-verifier`.
//! - A transcript wrapper wraps every absorb/squeeze in
//!   `transcript_absorb` / `transcript_challenge` markers so Fiat-Shamir
//!   hashing is separable (these intervals are subsets of the enclosing stage
//!   markers; do not add them onto stage totals).
//!
//! Return code:
//!
//! - `0` — verification succeeded.
//! - `1` — decode failure.
//! - `2` — verifier rejected the proof (panics with the error instead).

use akita_config::proof_optimized::fp64;
use akita_config::{CommitmentConfig, RecursiveCommitmentConfig};
use akita_recursion_glue::AkitaJoltInputs;
use akita_serialization::AkitaSerialize;
use akita_transcript::{AkitaTranscript, Transcript};
use akita_types::{BasisMode, SetupContributionMode};
use akita_verifier::batched_verify;

use jolt::{end_cycle_tracking, start_cycle_tracking};

type F = fp64::Field;
type E = fp64::ExtensionField;
const D: usize = 128;
type CfgMain = fp64::D128FullBound18;
type CfgRange = fp64::D128FullBound6;

const _: () = {
    // Hard-fail at compile time if the guest monomorphizations drift away from
    // the config and host artifact generator (`../artifact/src/main.rs`).
    assert!(D == <CfgMain as CommitmentConfig>::D);
    assert!(D == <CfgRange as CommitmentConfig>::D);
};

// ---------------------------------------------------------------------------
// Cycle-marker bridging.
// ---------------------------------------------------------------------------

/// Shared `&'static str` labels so start/end marker calls pass the *same*
/// pointer (the Jolt emulator keys active markers by label pointer).
const TRANSCRIPT_ABSORB: &str = "transcript_absorb";
const TRANSCRIPT_CHALLENGE: &str = "transcript_challenge";

#[inline]
fn marked<R>(name: &'static str, f: impl FnOnce() -> R) -> R {
    start_cycle_tracking(name);
    let out = f();
    end_cycle_tracking(name);
    out
}

/// Transcript decorator that wraps every absorb/squeeze in a cycle marker.
///
/// The intervals emitted under `transcript_absorb` / `transcript_challenge`
/// lie *inside* the per-stage span markers; summing them measures the total
/// Fiat-Shamir (BLAKE2b sponge + encoding) share of the verifier.
struct MarkedTranscript(AkitaTranscript<F>);

impl MarkedTranscript {
    fn unbound_verifier(session_label: &[u8]) -> Self {
        Self(AkitaTranscript::<F>::unbound_verifier(session_label))
    }
}

impl Transcript<F> for MarkedTranscript {
    fn new(domain_label: &[u8]) -> Self {
        Self(AkitaTranscript::<F>::new(domain_label))
    }

    fn bind_instance_bytes(&mut self, instance_bytes: &[u8]) {
        marked(TRANSCRIPT_ABSORB, || {
            self.0.bind_instance_bytes(instance_bytes)
        })
    }

    fn append_bytes(&mut self, label: &[u8], bytes: &[u8]) {
        marked(TRANSCRIPT_ABSORB, || self.0.append_bytes(label, bytes))
    }

    fn append_field(&mut self, label: &[u8], x: &F) {
        marked(TRANSCRIPT_ABSORB, || self.0.append_field(label, x))
    }

    fn append_serde<S: AkitaSerialize>(&mut self, label: &[u8], s: &S) {
        marked(TRANSCRIPT_ABSORB, || self.0.append_serde(label, s))
    }

    fn challenge_scalar(&mut self, label: &[u8]) -> F {
        marked(TRANSCRIPT_CHALLENGE, || self.0.challenge_scalar(label))
    }

    fn challenge_bytes(&mut self, label: &[u8], len: usize) -> Vec<u8> {
        marked(TRANSCRIPT_CHALLENGE, || self.0.challenge_bytes(label, len))
    }
}

/// Bridge `tracing` spans emitted by `akita-verifier` to Jolt cycle markers.
///
/// Span metadata names are `&'static str`, so the enter/exit marker calls for
/// one span invocation pass the same label pointer, which is exactly the key
/// the Jolt emulator tracks active markers by. Only compiled for the RISC-V
/// guest build; the native host run keeps its own subscriber.
#[cfg(any(target_arch = "riscv32", target_arch = "riscv64"))]
mod span_markers {
    // BTreeMap, not HashMap: std's default RandomState hasher seeds from OS
    // randomness, which the zeroos guest runtime does not provide (the guest
    // aborts in `sys::random`).
    use std::collections::BTreeMap;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::Mutex;
    use tracing::span::{Attributes, Id, Record};
    use tracing::{Event, Metadata, Subscriber};

    pub(crate) struct SpanMarkerSubscriber {
        next_id: AtomicU64,
        names: Mutex<BTreeMap<u64, &'static str>>,
    }

    impl SpanMarkerSubscriber {
        fn new() -> Self {
            Self {
                next_id: AtomicU64::new(1),
                names: Mutex::new(BTreeMap::new()),
            }
        }

        fn name_of(&self, id: &Id) -> Option<&'static str> {
            self.names
                .lock()
                .ok()
                .and_then(|names| names.get(&id.into_u64()).copied())
        }
    }

    impl Subscriber for SpanMarkerSubscriber {
        fn enabled(&self, metadata: &Metadata<'_>) -> bool {
            metadata.is_span()
        }

        fn new_span(&self, attrs: &Attributes<'_>) -> Id {
            let id = self.next_id.fetch_add(1, Ordering::Relaxed);
            if let Ok(mut names) = self.names.lock() {
                names.insert(id, attrs.metadata().name());
            }
            Id::from_u64(id)
        }

        fn record(&self, _id: &Id, _record: &Record<'_>) {}

        fn record_follows_from(&self, _id: &Id, _follows: &Id) {}

        fn event(&self, _event: &Event<'_>) {}

        fn enter(&self, id: &Id) {
            if let Some(name) = self.name_of(id) {
                jolt::start_cycle_tracking(name);
            }
        }

        fn exit(&self, id: &Id) {
            if let Some(name) = self.name_of(id) {
                jolt::end_cycle_tracking(name);
            }
        }

        fn try_close(&self, id: Id) -> bool {
            if let Ok(mut names) = self.names.lock() {
                names.remove(&id.into_u64());
            }
            false
        }
    }

    pub(crate) fn install() {
        let _ = tracing::subscriber::set_global_default(SpanMarkerSubscriber::new());
    }
}

#[cfg(any(target_arch = "riscv32", target_arch = "riscv64"))]
fn install_span_markers() {
    span_markers::install();
}

#[cfg(not(any(target_arch = "riscv32", target_arch = "riscv64")))]
fn install_span_markers() {
    // Native sanity runs keep the host's own tracing subscriber.
}

// ---------------------------------------------------------------------------
// Shared verifier body.
// ---------------------------------------------------------------------------

macro_rules! akita_verify_body {
    ($cfg:ty, $input:expr) => {{
        install_span_markers();

        // `&[u8]` (rather than `Vec<u8>`) so the postcard-decoded input is a
        // zero-copy borrow into the guest's input region: no megabyte-scale
        // copy before verifier replay.
        start_cycle_tracking("deserialize_input");
        #[cfg(any(
            feature = "trusted-benchmark-artifact",
            akita_trusted_benchmark_artifact
        ))]
        let decoded_result = AkitaJoltInputs::<F, E, D>::read_trusted_host_artifact_bytes($input);
        #[cfg(not(any(
            feature = "trusted-benchmark-artifact",
            akita_trusted_benchmark_artifact
        )))]
        let decoded_result = AkitaJoltInputs::<F, E, D>::read_from_bytes($input);

        let decoded = match decoded_result {
            Ok(decoded) => decoded,
            Err(_) => {
                end_cycle_tracking("deserialize_input");
                return 1;
            }
        };
        end_cycle_tracking("deserialize_input");

        start_cycle_tracking("transcript_init");
        let mut transcript = MarkedTranscript::unbound_verifier(&decoded.transcript_domain);
        end_cycle_tracking("transcript_init");

        // We call `batched_verify` directly (rather than the public
        // `AkitaCommitmentScheme::batched_verify` wrapper) to skip its
        // `Instant::now()` + final `tracing::info!` wall-clock log. The
        // Jolt RISC-V runtime panics on `std::time::Instant::now()` (no
        // `clock_gettime` support), so the scheme entry point would abort
        // before any real verifier work runs.
        //
        // Recursive-mode blobs replay under `RecursiveCommitmentConfig` so
        // schedule resolution plans the recursive setup-offload path; the
        // Direct arm is byte-identical to the original single-group harness.
        start_cycle_tracking("akita_verify");
        let claims = match decoded.verifier_opening_batch() {
            Ok(claims) => claims,
            Err(_) => {
                end_cycle_tracking("akita_verify");
                return 1;
            }
        };
        let result = match decoded.setup_contribution_mode {
            SetupContributionMode::Direct => batched_verify::<$cfg, _>(
                &decoded.proof,
                &decoded.verifier_setup,
                &mut transcript,
                claims,
                BasisMode::Lagrange,
                decoded.setup_contribution_mode,
            ),
            SetupContributionMode::Recursive => {
                batched_verify::<RecursiveCommitmentConfig<$cfg>, _>(
                    &decoded.proof,
                    &decoded.verifier_setup,
                    &mut transcript,
                    claims,
                    BasisMode::Lagrange,
                    decoded.setup_contribution_mode,
                )
            }
        };
        end_cycle_tracking("akita_verify");

        match result {
            Ok(()) => 0,
            Err(err) => panic!("recursive verifier rejected proof: {err:?}"),
        }
    }};
}

// Memory limits sized for the fp64 D=128 verifier blobs. Blob size scales
// with `nv` (setup matrix prefix + proof). We keep the fp128 harness limits:
//   - `max_input_size` = 768 MiB (== `akita_recursion_glue::MAX_JOLT_BLOB_BYTES`).
//   - `heap_size`      = 1.5 GiB for the decoded verifier setup + transient
//                        verifier-internal allocations.
//   - `stack_size`     = 16 MiB for sumcheck recursion + extension-field
//                        arithmetic frames.
// `backtrace = "off"` strips DWARF symbols + `.eh_frame` and skips
// `-Cforce-frame-pointers=yes`; flip to `backtrace = "dwarf"` to symbolicate
// a guest panic.

/// Aerie main-group proxy: `fp64::D128FullBound18` (nv=23 by default).
#[jolt::provable(
    backtrace = "off",
    stack_size = 16777216,
    heap_size = 1610612736,
    max_input_size = 805306368,
    max_output_size = 1024,
    max_trace_length = 4294967296
)]
fn akita_verify(input: &[u8]) -> u32 {
    akita_verify_body!(CfgMain, input)
}

/// Aerie range-group proxy: `fp64::D128FullBound6` (nv=26 by default).
#[jolt::provable(
    backtrace = "off",
    stack_size = 16777216,
    heap_size = 1610612736,
    max_input_size = 805306368,
    max_output_size = 1024,
    max_trace_length = 4294967296
)]
fn akita_verify_bound6(input: &[u8]) -> u32 {
    akita_verify_body!(CfgRange, input)
}
