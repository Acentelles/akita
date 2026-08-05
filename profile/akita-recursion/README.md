# `akita-recursion` — Akita verifier inside Jolt

Runs the Akita PCS verifier inside a Jolt zkVM guest program and reports
per-phase cycle counts. End-to-end this can also produce a SNARK of the
verifier execution and confirm Jolt accepts it.

**Current target: Aerie-representative fp64 presets** (field
`Prime64Offset59`, extension `Ext2`, ring `D=128`):

- `--preset bound18` → `fp64::D128FullBound18`, guest fn `akita_verify`,
  default `nv=23` (Aerie's main-group arity);
- `--preset bound6` → `fp64::D128FullBound6`, guest fn `akita_verify_bound6`,
  default `nv=26` (Aerie's range-group arity).

**All blobs are synthetic proxies, not Aerie/Falcon results.** The committed
polynomials are bounded random data; only the verifier *schedule* (fold
levels, sum-check rounds, setup-scan or delegation shape) is representative
of Aerie's Akita opening.

Two setup-contribution modes (blob format `AKJOLTv2`, multi-group):

- `--setup-mode direct` (default): single-poly single-group opening, the
  original proxy; ships the full expanded setup matrix.
- `--setup-mode recursive`: the recursive setup-offload profile key (two
  singleton dense precommits at `nv/2` plus a two-poly final group at `nv`)
  proved and replayed under `RecursiveCommitmentConfig`, so the guest runs
  the stage-3 setup-product sumcheck plus the carried setup-prefix opening
  (shared suffix-aligned EOR). Recursive blobs can truncate the shipped
  setup matrix (`--setup-transport slice|seed`): the delegated root scan is
  replaced by the stage-3 sumcheck and the public prefix commitment `C_S`
  (shipped in the blob's `prefix_slots`), so only the sub-gate/terminal
  read prefix ships (slice) or is re-derived from the 32-byte seed in the
  guest (seed). The minimal prefix is found by binary search over native
  verification in the artifact generator.

The original fp128 `D64OneHot` harness lives in the git history (see
"fp128 history" below for its headline numbers).

This directory is a **standalone Cargo sub-workspace** (it's excluded
from the parent Akita workspace). It pins Rust `1.95` plus the
RISC-V targets and applies Jolt's `[patch.crates-io]` overrides for
`arkworks-algebra`.

## Crates

| Crate        | Kind | Purpose                                                          |
| ------------ | ---- | ---------------------------------------------------------------- |
| `glue/`      | lib  | Shared verifier-input blob format (`AkitaJoltInputs<F, E, D>`).  |
| `artifact/`  | bin  | Runs the Akita prover and writes the verifier-input blob.        |
| `host/`      | bin  | Compiles the guest, runs Jolt trace/prove, prints cycle counts.  |
| `guest/`     | bin  | `#[jolt::provable]` RISC-V program that runs the Akita verifier. |

## Quick start (fp64 bound18, `nv=23`)

You need the [Jolt CLI](https://github.com/a16z/jolt) installed. Build
artifacts should go outside the repo (it lives in Dropbox), e.g.:

```bash
cd profile/akita-recursion
export CARGO_TARGET_DIR=~/.cache/akita-recursion-target

# 1. Build the host binaries.
cargo build --release

# 2. Generate the verifier-input blob (artifact prints exact size).
#    REQUIRED before step 3 — `host` reads this file from disk.
AKITA_NUM_VARS=23 \
    AKITA_RECURSION_BLOB=$CARGO_TARGET_DIR/akita_recursion_inputs_fp64_nv23.bin \
    $CARGO_TARGET_DIR/release/akita-recursion-artifact --preset bound18

# 3. Compile the guest to RISC-V, emulate it, and report cycle markers.
#    `--trace-output /dev/null` keeps the raw trace bytes off disk while
#    preserving the cycle-marker output.
ZEROOS_GUEST_RUSTFLAGS=-Zunstable-options \
    AKITA_RECURSION_LOG=info $CARGO_TARGET_DIR/release/akita-recursion-host \
    --preset bound18 \
    --trace-only \
    --trace-output /dev/null \
    --input $CARGO_TARGET_DIR/akita_recursion_inputs_fp64_nv23.bin
```

For the range-group bracket, repeat with `--preset bound6` and
`AKITA_NUM_VARS=26` on both binaries.

For the recursive setup-offload measurement, generate all three transports
from one prover run and trace each blob:

```bash
AKITA_NUM_VARS=23 \
    AKITA_RECURSION_BLOB=$CARGO_TARGET_DIR/inputs_fp64_nv23_recursive.bin \
    $CARGO_TARGET_DIR/release/akita-recursion-artifact --preset bound18 \
    --setup-mode recursive --setup-transport full,slice,seed
# → inputs_fp64_nv23_recursive.bin (+ .slice / .seed variants)
```

Measured results for the fp64 presets live in
[`AERIE-FP64-MEASUREMENT.md`](AERIE-FP64-MEASUREMENT.md).

## Cycle markers

Three top-level guest markers (`deserialize_input`, `transcript_init`,
`akita_verify`) match the original harness. Two further mechanisms give the
per-stage split without any changes to `akita-verifier`:

1. **Span bridge.** The guest installs a minimal `tracing` subscriber that
   converts every span the Akita verifier already emits into a Jolt cycle
   marker (span names are `&'static str`, so start/end pass the same label
   pointer, which is how the Jolt emulator keys active markers). Spans of
   interest: `stage3_setup_sumcheck` (recursive mode: delegated
   setup-product sumcheck replay), `eor_replay` (shared extension-opening
   reduction replay), `derive_public_matrix_flat` (seed-derived transport:
   in-guest XOF prefix derivation), `prepare_fold_replay`, `ring_switch_verifier`,
   `ring_switch_verifier_terminal`, `ring_switch_verifier_core`,
   `prepare_relation_matrix_evaluator`, `structured_chunks`,
   `setup_contribution`, `r_structured`/`r_dense`, `stage1_sumcheck`,
   `stage2_sumcheck`, `AkitaStage2Verifier::new`,
   `stage2_expected_output_claim`, `stage2_witness_eval`,
   `stage2_relation_matrix_eval`. Markers repeat once per fold level, in
   execution order; sum the occurrences per label (they nest, so do **not**
   add nested labels onto their parents).
2. **Transcript wrapper.** The guest wraps the verifier transcript in a
   decorator that emits `transcript_absorb` / `transcript_challenge` markers
   around every absorb/squeeze, making the Fiat-Shamir (BLAKE2b) share
   separable. These intervals are subsets of the enclosing stage markers.

## Trusted-decoder benchmark path

This profile is a trusted host-artifact benchmark: the guest decodes the
verifier setup through the explicitly trusted cached-matrix path
(`AKITA_RECURSION_TRUSTED_BENCHMARK_ARTIFACT=1`, set by the host binary
before Jolt compiles the RISC-V ELF). Seed/matrix shape metadata and field
elements are still validated, but the guest skips checking that the expanded
setup matrix coefficients equal the matrix derived from the seed, because the
blob is produced and strictly round-trip-verified by the host-side artifact
generator. Plain `--features guest` builds use strict setup decoding. A
production recursion circuit must use strict setup validation or bind an
externally checked setup commitment.

## Debugging guest panics

The guest enables `jolt/stdout` so panic messages reach the host. The
`#[jolt::provable]` attribute currently uses `backtrace = "off"`; flip it to
`backtrace = "dwarf"` for a single diagnostic iteration if a panic comes
back, then run with:

```bash
ZEROOS_GUEST_RUSTFLAGS=-Zunstable-options \
    JOLT_BACKTRACE=full AKITA_RECURSION_LOG=info \
    $CARGO_TARGET_DIR/release/akita-recursion-host --trace-only \
    --input <blob>
```

To force a clean guest rebuild:

```bash
rm -rf /tmp/akita-recursion-targets /tmp/jolt-guest-targets
```

## Environment variables

| Variable                  | Default                                  | Effect                                  |
| ------------------------- | ---------------------------------------- | --------------------------------------- |
| `AKITA_NUM_VARS`          | `23` (bound18) / `26` (bound6)           | Polynomial arity for the prover.        |
| `AKITA_RECURSION_BLOB`    | `target/akita_recursion_inputs.bin`      | Output path for the blob (`artifact`).  |
| `AKITA_RECURSION_LOG`     | `info`                                   | `tracing-subscriber` filter (`host`).   |
| `ZEROOS_GUEST_RUSTFLAGS`  | unset                                    | Pass `-Zunstable-options` when Rust requires it for Jolt's custom `riscv64imac-zero-linux-musl` target. |
| `JOLT_BACKTRACE`          | unset                                    | `full` ⇒ symbolic guest backtraces.     |
| `AKITA_ALLOW_DEBUG_PROFILE` | unset                                  | `1` ⇒ bypass `--release` guard in `artifact`. |

## CLI flags (`akita-recursion-host`)

| Flag                  | Default                              | Description                                  |
| --------------------- | ------------------------------------ | -------------------------------------------- |
| `--input <path>`      | `target/akita_recursion_inputs.bin`  | Path to the blob produced by `artifact`.     |
| `--preset <p>`        | `bound18`                            | `bound18` or `bound6`; must match the blob.  |
| `--target-dir <path>` | `/tmp/akita-recursion-targets`       | Jolt's per-program build cache.              |
| `--trace-output <path>` | `<target-dir>/akita_verify.trace`  | Trace file path for `--trace-only`.          |
| `--trace-only`        | off                                  | Skip preprocessing + Jolt prove/verify.      |

## How it works

1. **`artifact`** runs `AkitaCommitmentScheme::<fp64 preset>` →
   `setup_prover` → `commit` → `batched_prove` over one synthetic dense
   bounded polynomial, sanity-verifies on the host, and serializes
   `(transcript_domain, num_vars, opening_point, opening, commitment,
   verifier_setup, proof_shape, proof)` into a single blob via
   [`AkitaJoltInputs::write_to_bytes`](glue/src/lib.rs).
2. **`host`** loads the blob, strictly re-verifies it, compiles the guest to
   `riscv64imac-zero-linux-musl` via the Jolt CLI, runs Jolt's
   preprocess/prove/verify (or just the trace under `--trace-only`),
   and forwards per-marker cycle counts through `tracing`.
3. **`guest`** (running inside the Jolt RISC-V emulator) decodes the
   blob and invokes `akita_verifier::batched_verify` directly —
   bypassing `akita-scheme::batched_verify`, which would otherwise
   call `Instant::now()` (the Jolt runtime doesn't implement
   `clock_gettime`, and the guest aborts there). The guest constructs an
   unbound verifier transcript and the verifier binds the canonical instance
   descriptor.

## fp128 history

The original harness targeted `fp128::D64OneHot`. Headline numbers kept for
comparison (Apple Silicon laptop):

- `nv=20`: 65,283,025-cycle trace (`backtrace = "off"`, `input: &[u8]`);
  the `Vec<u8> → &[u8]` input switch alone removed ~36 M cycles because
  postcard's `Vec<u8>` decode copied the 1.1 MiB input byte-by-byte.
- `nv=32`: ~8 G-cycle trace, ~22 min wall clock, dominated by
  `deserialize_input` (decoding the expanded verifier-setup matrix inside
  the blob); the proof itself is a tiny fraction.
- D=64 was pinned over D=32 because D=32 is not a valid A-role fold degree
  (`d_a ≥ 64`), took one more fold level, and had a ~4.5× larger
  verifier-setup matrix.

## Open follow-ups

1. **Full prove at large `nv`** requires bumping `max_trace_length` past the
   traced length in the `#[jolt::provable]` attribute (currently 4 G) and
   server-class memory.
2. **Make `deserialize_input` cheaper.** At large `nv` it dominates the
   trace; most of it is decoding the expanded verifier-setup matrix. Ship
   just the `public_matrix_seed` (32 bytes) and re-derive inside the guest,
   or bind an externally checked setup commitment (the
   `NATIVE-RECURSION-DESIGN.md` route in the Aerie repo).
3. **Upstreaming candidates** — if the public trait entry point ever becomes
   timer-free and verifier-only, the guest should delegate to it; the guest
   should remain free of `akita-scheme`, `akita-prover`, and `akita-setup`
   dependencies.
