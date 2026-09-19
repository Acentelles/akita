# Experimental resident Metal Stage2 owner

This opt-in adapter keeps the Stage2 coefficient witness in a local Metal owner from compact entry through coefficient folds. The default PCS API remains CPU-only. The explicit `AkitaCommitmentScheme::batched_prove_resident_stage2` sibling supports the Prime64Offset59/Ext2 field pair. It selects native execution for supported digit bases 4, 8, 16, 32 and 64 before creating the owner; unsupported folds use the existing CPU route. Errors after owner creation abort the proof attempt without CPU replay.

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

The bounded validation compared 64 original and 96 wider-basis complete Stage2 sumcheck cases against the original CPU implementation. The wider matrix includes bases 16/32/64, both packing representations, full/skip norm modes, additional terms and padded domains. The native pair-table extension separately passed 40 scalar-oracle cases and eight rejection cases. The complete PCS fixture checks two distinct NV20 polynomials, exact CPU/resident proof bytes, four original-verifier checks with independently evaluated opening claims, and next transcript challenges. All five scheduled folds execute natively, with bases 8/8/16/16/32 and checked entry/advance/export counts. Basis 64 is covered by the Stage2 matrix, not this NV20 schedule. Full MLDSA aggregate correctness and end-to-end performance remain separate validation tasks.

For bases 16/32/64, the original CPU first fold, temporary N/2 field witness allocation, and next-message computation remain in the proof lifecycle. Native compact entry then consumes its original copied digits and both challenges. The pair lookup table stays within 64 KiB. This revision makes no claim that the first CPU fold or its costs were removed, and no speedup claim.

This package remains experimental. The full repository CI matrix, portability checks and post-submission device-failure tests require separate evidence beyond the bounded successful-path campaigns. The native ABI and Rust adapter must be built from matching sources; no stable cross-version ABI is promised.
