# akita-transcript

Fiat-Shamir transcript support for Akita.

## Before you edit `src/` here

This crate is a leaf dependency of nearly the whole workspace, so **any** change
under `src/` invalidates the build graph for `akita-prover`, `akita-verifier`,
`akita-config`, `akita-pcs` and their test targets, forcing a full recompile.
If long-running benchmarks, proof sweeps or attack tests are in flight on the
machine, that is expensive and it lands on somebody else.

The trap worth naming, because it is not obvious and it has already cost us
once: **Cargo fingerprints the crate, not the reachable subset.** Adding a
module that is entirely `cfg`-gated off, such as `#[cfg(feature = "...")] pub
mod poseidon2;`, still changes the crate's source fingerprint and still triggers
the same downstream rebuild, even though not one line of it compiles in a
default-feature build. "It is `cfg`'d out, so it is free" is false. So is "it is
only a comment" and "it is purely additive".

What is actually free: edits under `tests/` (they invalidate only that test
target), `tools/`, and this README.

When iterating on a change that will need several attempts to compile, do the
iteration outside the workspace and land one known-good edit. A standalone copy
of this crate with absolute-path dependencies on `akita-field` and
`akita-serialization`, its own `CARGO_TARGET_DIR`, and the workspace `[lints]`
tables copied in, compiles the crate faithfully while touching nothing in the
graph. Copy the lint tables, or you will reach green in the scratch copy and
still fail `-D warnings` on landing.

## Active Hardening Pillars

Akita's transcript hardening has three active pieces:

1. `AkitaInstanceDescriptor` bytes are bound into the spongefish preamble through `DomainSeparator.instance(...)`.
2. `AkitaTranscript` is backed by spongefish. The default backend is Blake2b; `--no-default-features --features transcript-keccak` selects Keccak instead. Cargo `--all-features` builds also resolve to Blake2b.
3. `LoggingTranscript` is available behind `logging-transcript` for tests and schedule inspection.

Production labels are diagnostics only. `Label` is a zero-sized type when `logging-transcript` is disabled, and labels are never absorbed into the production sponge. Positional order plus the instance descriptor preamble are the protocol transcript domain.

## Logging Checks

`LoggingTranscript` records:

- descriptor preamble events;
- transcript absorbs;
- challenge squeezes;
- verifier wire-use events registered by tests or verifier harnesses.

Its smell checks assert:

- the first event is a non-empty descriptor preamble;
- absorbs are non-empty;
- labels are in `labels::ALL_LABELS`;
- each tracked verifier wire use is followed by a matching absorb before the next squeeze;
- declared wire-coverage manifest labels are actually recorded.

The PCS integration tests enable this with:

```bash
cargo test -p akita-pcs --features logging-transcript --test transcript_hardening
cargo test -p akita-pcs --features logging-transcript --test transcript_hardening_proptest
```

For the full design and deferred follow-ups, see `specs/transcript-hardening.md`.

## Algebraic transcript backend (`transcript-poseidon2`)

> **UNAUDITED RESEARCH CRYPTOGRAPHY - NOT FOR PRODUCTION.**
> This is a new Poseidon2 instance over `p* = 2^64 - 59`. No
> arithmetization-oriented permutation has received third-party cryptanalysis
> over this prime at full parameters; the only public analysis over `p*`
> targets the deliberately round-reduced Ethereum Foundation bounty instances.
> Do not deploy it.

Why it exists: the Akita-in-Akita wrapped verifier must recompute the
Fiat-Shamir transcript hash *inside* a proof over the same field, and a
byte-oriented Blake2b duplex is not a feasible circuit. `transcript-poseidon2`
supplies an arithmetization-friendly alternative behind a non-default feature.

- Permutation: Poseidon2 (eprint 2023/323), `t = 12`, `alpha = 3` with
  `(R_F, R_P) = (8, 42)` by default; `transcript-poseidon2-alpha7` selects
  `alpha = 7` with `(8, 22)`.
- Sponge: duplex, rate 8 / capacity 4 (256 capacity bits = the 128-bit level
  under the SAFE bound, eprint 2023/520), overwrite absorption per
  [CO25] Construction 3.3, with a 28-byte domain tag packed into the capacity.
- Byte encoding: absorb rule (A1) is SAFE-style `10*` padding into 7-byte
  little-endian limbs; squeeze rule (S1) emits the low 7 bytes of each squeezed
  element, rejecting elements `>= 255 * 2^56` so the output bytes are *exactly*
  uniform. Rule (S2), `squeeze_field`, is the rejection-free exactly-uniform
  field-element path an in-circuit verifier should use.
- The sponge sits below `AkitaTranscript` at the `DuplexSpongeInterface`
  boundary, so absorb labels stay decorative and the swap is transparent to the
  protocol. Binding labels into the state would be cheap here and is a genuine
  improvement, but it is a wire-format change and is deliberately not done.

Backend selection priority is Blake2b > Keccak > Poseidon2, so `--all-features`
and every existing build keep Blake2b byte-for-byte. This is enforced by
`tests/blake2b_byte_identity.rs`, which pins a digest of a scripted transcript
session captured from git `HEAD` before this backend existed.

### Choosing alpha, and what this costs

**Selected: `alpha = 3`** (`R_F = 8`, `R_P = 42`). `alpha = 7` (`R_F = 8`,
`R_P = 22`) is generated and available behind
`transcript-poseidon2-alpha7`; both are formula-legitimate.

Measured with `tests/poseidon2_bench.rs`, release profile, all three sponges in
one process. Only the *ratios* are meaningful: absolute timings in this
environment run several times slower than bare host, and the machine is shared.

| | ratio |
|---|---|
| permutation, `alpha = 3` / `alpha = 7` | **1.077x** |
| duplex round (absorb 256 B, squeeze 32 B), `alpha = 3` / `alpha = 7` | 1.079x |
| full `AkitaTranscript` round, `alpha = 3` / `alpha = 7` | **1.014x** |
| full `AkitaTranscript` round, Poseidon2 / Blake2b | **~9.3x** |

Two things are worth knowing about these numbers.

First, `alpha = 7` is *cheaper natively*, which is the opposite of the naive
expectation that a smaller S-box exponent wins. The cause is the internal
linear layer: the generator draws the diagonal `mu` from the Grain stream, so
entries are full-size 64-bit values and every internal round costs `t = 12`
general multiplications. At `R_P = 42` that is 504 multiplications, more than
the whole S-box budget, and it outweighs the 2-versus-4 multiplication saving
per S-box. If native transcript cost ever matters, the fix is a power-of-two
diagonal (as Plonky3 uses for Goldilocks) that still passes
`check_minpoly_condition`; that is a new parameter search, not a
re-parameterisation, and has not been attempted.

Second, the native penalty for `alpha = 3` **collapses to 1.4% at the level
that actually matters**, because byte-encoding overhead dominates a full
transcript round. Against that, `alpha = 3` is roughly 2x cheaper in circuit on
area times degree (`127 * 3 = 381` versus `107 * 7 = 749`), and cheaper still
for a sum-check prover, where a degree-`d` constraint needs `d + 1` evaluations
per round: 4 against 8. That is the whole decision. (An earlier operation-count
model predicted a 15-20% permutation penalty; measurement corrected it to 7.7%.
The direction was right, the magnitude was overstated, and the measured figure
is the one to quote.)

**The honest headline: a Poseidon2 transcript round costs about 9x a Blake2b
one.** That is a real regression in native prover time and it is justified by
exactly one thing, the ability of a wrapped verifier to recompute the
transcript hash in-circuit over the same field. For any non-recursive use there
is no upside whatsoever, only the 9x and an unaudited permutation. This backend
must never become the default, and it should not be enabled for any workload
that is not paying for the arithmetization.

### Selecting a non-default backend downstream

Two different things are often conflated here, and they need different fixes.

**Using a non-default sponge in a test or tool.** Name it as the type
parameter: `AkitaTranscript<F, Poseidon2ByteSponge<Alpha3>>`. `AkitaTranscript`
has always been generic over the sponge, so this needs no feature juggling; the
crate only has to be built with `transcript-poseidon2` on so the module exists
(`--features akita-transcript/transcript-poseidon2` is enough, since it only
*adds* a feature). Bound on
[`TranscriptSpongeBackend`](crate::TranscriptSpongeBackend) if you need to be
generic over the sponge without depending on `spongefish` yourself.

**Making a non-default sponge the selected backend workspace-wide.** This is the
one that needs manifest surgery, and it is easy to get half right. Recipe:

1. Every crate that reaches `akita-transcript` needs `default-features = false`
   on that edge **and** on every edge to another transcript-carrying crate,
   plus pass-through features:

   ```toml
   akita-transcript = { path = "../akita-transcript", default-features = false }

   [features]
   default = ["transcript-blake2b"]          # preserves today's behaviour
   transcript-blake2b   = ["akita-transcript/transcript-blake2b", ...]
   transcript-keccak    = ["akita-transcript/transcript-keccak", ...]
   transcript-poseidon2 = ["akita-transcript/transcript-poseidon2", ...]
   ```

2. **The set is 10 crates, not the 7 that depend on `akita-transcript`
   directly.** This is the part that bites. Cargo unifies features across the
   whole graph, so a crate that merely depends on a transcript-carrying crate
   pulls in *that crate's* defaults, which re-enables `transcript-blake2b` and
   silently defeats a partial conversion. Direct: `akita-challenges`,
   `akita-config`, `akita-pcs`, `akita-prover`, `akita-sumcheck`, `akita-types`,
   `akita-verifier`. Transitive but equally required: `akita-planner` (via
   challenges, types), `akita-schedules` (via planner), `akita-setup` (via
   config, prover, types).

3. Dev-dependency edges unify too, in test builds. `akita-pcs` -> `akita-planner`
   and `akita-prover` -> `akita-config` also need `default-features = false`.
   (`akita-pcs` -> `akita-config` and `akita-verifier` -> `akita-config` already
   have it.)

4. `profile/akita-recursion` is a **separate workspace**, so its three edges do
   not participate in main-workspace unification and can be left alone.

5. Unaffected entirely: `akita-algebra`, `akita-field`, `akita-serialization`,
   `akita-sis-estimator`, `akita-witness`.

Do not instead flip `default` in this crate's manifest. It would work, and it
would silently make an unaudited experimental sponge the transcript for every
downstream build.

Note that `transcript-keccak` has the same reachability gap: it has never been
selectable above `akita-transcript` itself, so it has effectively never been
exercised at the protocol level.

### Parameter provenance

Constants are generated, never hand-written:

```bash
sage crates/akita-transcript/tools/poseidon2_params_p64m59.sage 3 > alpha3.json
sage crates/akita-transcript/tools/poseidon2_params_p64m59.sage 7 > alpha7.json
python3 crates/akita-transcript/tools/poseidon2_sponge_reference.py \
    alpha3.json alpha7.json > vectors.json
python3 crates/akita-transcript/tools/emit_rust_params.py \
    alpha3.json alpha7.json vectors.json crates/akita-transcript/src/poseidon2
```

`poseidon2_params_p64m59.sage` is a parameterised fork of the designers'
`poseidon2_rust_params.sage` (HorizenLabs/poseidon2), vendored alongside it as
`upstream_poseidon2_rust_params.sage.txt`; every deviation is listed in its
header. To re-check the committed artifacts without SageMath:

```bash
python3 crates/akita-transcript/tools/verify_generated_params.py
```

That script shares no code with the generator. It re-derives the round numbers
from the published inequalities, regenerates the round constants and the
internal diagonal from the Grain LFSR, re-checks MDS-ness of `M4`,
invertibility and the exact form of both matrices, `check_minpoly_condition`
via Rabin irreducibility, and the vendored permutation KATs.

[CO25]: https://eprint.iacr.org/2025/536
