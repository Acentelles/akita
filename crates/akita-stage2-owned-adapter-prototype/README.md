# Experimental resident Metal Stage2 owner

This opt-in adapter keeps the Stage2 coefficient witness in a local Metal owner from compact entry through coefficient folds. The default PCS API remains CPU-only. The explicit `AkitaCommitmentScheme::batched_prove_resident_stage2` sibling supports the Prime64Offset59/Ext2 field pair. It selects native execution for supported digit bases 4 and 8 before creating the owner; unsupported folds use the existing CPU route. Errors after owner creation abort the proof attempt without CPU replay.

The owner is confined to its creating thread and is neither `Send` nor `Sync`. Compact input is copied at admission. Canonical field factors and checked sparse descriptors are uploaded for each round. Only round messages and the final initialized witness are returned. This is not a zero-copy implementation. The `resident-stage2-observer` feature adds diagnostic execution counters and is not required by production callers.

## Build explicitly on Apple silicon macOS

The native build is intentionally separate from Cargo. It requires Python 3, Apple Command Line Tools, a macOS SDK and a Metal device. From the repository root:

```sh
native_output="$(mktemp -d "${TMPDIR:-/tmp}/akita-stage2.XXXXXX")/build"
SDKROOT="$(xcrun --sdk macosx --show-sdk-path)" \
  python3 crates/akita-stage2-owned-adapter-prototype/native/build.py "$native_output"
AKITA_STAGE2_NATIVE_LIB_DIR="$native_output" \
  cargo test --locked --release -p akita-pcs \
  --test resident_stage2_full_pcs --no-default-features \
  --features parallel,transcript-blake2b,schedules-default,resident-stage2-observer \
  -- --exact full_pcs_compressed_root_and_suffix_match_cpu_proof_verifier_and_transcript \
  --nocapture --test-threads=1
```

The build output directory must not exist; the script creates it. It contains `libakita_stage2_owned.a` and the bounded fixture client. Keep the same `AKITA_STAGE2_NATIVE_LIB_DIR` when building any Rust executable with the resident feature. The path is a caller-chosen build output, not a repository dependency or a private machine path. The build script checks the Apple silicon target and archive, then links Metal, Foundation and C++. It does not automatically invoke a native compiler or download an artifact. Omitting the environment variable does not supply native symbols and is insufficient for linking an executable that uses the owner.

To use the CPU-only library graph, omit the resident features. To type-check the resident API without diagnostic counters, use `--features parallel,transcript-blake2b,resident-stage2-owned` with the archive environment set.

## Validated scope

The bounded validation compared 64 complete Stage2 sumcheck cases against the original CPU implementation, plus 16 continuous exported fixtures against the native owner. The complete PCS fixture checks two distinct NV20 polynomials, exact CPU/resident proof bytes, the original verifier with independently evaluated opening claims, and next transcript challenges. In that schedule the root and first recursive suffix use Metal; later basis16/32 folds are explicitly selected for CPU. This revision does not claim all-fold native execution, full MLDSA aggregate correctness, throughput improvement or an end-to-end speedup. Wider-basis work remains a separate revision.

This package remains experimental. The full repository CI matrix, portability checks and post-submission device-failure tests require separate evidence beyond the bounded successful-path campaigns. The native ABI and Rust adapter must be built from matching sources; no stable cross-version ABI is promised.
