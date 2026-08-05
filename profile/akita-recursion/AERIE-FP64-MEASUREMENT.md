# Measured RISC-V cycle breakdown: Akita fp64 verifier inside the Jolt guest

Date: 2026-07-31. Status: measured. **Every row in this file is a synthetic
single-group proxy, not an Aerie/Falcon result**: the committed polynomial is
bounded random data with no relation structure; only the verifier *schedule*
(fold levels, sum-check rounds, setup-scan shape) is representative of Aerie's
Akita opening. Aerie's production root opening is a merged multi-group batch
(bound-18 main groups at `main_num_vars=23` plus a bound-6 digit group at
`range_num_vars=26`); this harness measures each shape as a standalone
single-group opening, so the two runs bracket the real multi-group root rather
than reproduce it.

> **CORRECTION (2026-08-01), supersedes an earlier figure circulated from
> this file.** An intermediate revision reported a "2.06 GiB verifier
> requirement" for recursive mode at `nv = 23`. **That number is the
> layout-capacity precheck's, not the verifier's actual work.** The measured
> read high-water mark is **270,336 ring elements = 276,824,064 B (264 MB)**,
> an 8.0x over-estimate by the precheck, and it fits the 768 MiB blob cap.
> In particular the **root level's B matrix (2,162,688 ring elements) is
> never read at a folded root**; it is counted by the precheck only. Any
> note derived from the 2.06 GiB figure should be corrected. Details and
> method in "Read high-water mark vs layout capacity" below.

Purpose: validate/falsify the verifier cost ledger (V1-V7) in the Aerie repo's
`_docs/NATIVE-RECURSION-DESIGN.md` Section 2.

## Setup

- Harness: `profile/akita-recursion` (this directory), `--trace-only` (no Jolt
  proving), Jolt rev `2509bdce`, guest target `riscv64imac-zero-linux-musl`.
- Presets: `fp64::D128FullBound18` at `nv=23` and `fp64::D128FullBound6` at
  `nv=26` (field `Prime64Offset59`, extension `Ext2`, ring `D=128`,
  `SetupContributionMode::Direct`, basis Lagrange, single committed polynomial,
  one opening at a subfield-packed random extension point).
- **Trusted-decoder flag enabled** (`AKITA_RECURSION_TRUSTED_BENCHMARK_ARTIFACT=1`,
  set by the host binary): the guest decodes the expanded setup matrix from the
  blob and skips re-deriving it from the seed. Field elements and shape
  metadata are still validated; the blob is strictly decoded and re-verified on
  the host before replay. Consequence for the ledger: **V4 (in-circuit XOF
  regeneration) never executes**; its cost shows up instead as the
  `deserialize_input` row (the "ship the expanded matrix" alternative the
  design note discusses).
- Cycle attribution: Jolt cycle markers. Top-level markers
  (`deserialize_input`, `transcript_init`, `akita_verify`) plus (a) a
  guest-side tracing subscriber that converts every `akita-verifier` span into
  a cycle marker, and (b) a transcript decorator that marks every
  absorb/squeeze (`transcript_absorb` / `transcript_challenge`). Markers nest:
  do not add a nested label onto its parent. "Cycles" below are the emulator's
  *total* cycles (RV64IMAC + virtual instructions).
- Machine: M3 Max class laptop, shared load; wall-clock numbers are not
  meaningful, cycle counts are deterministic.

## Headline totals

| Run | Blob | Fold levels | Total trace | `deserialize_input` | `akita_verify` |
|-----|------|-------------|-------------|---------------------|----------------|
| bound18, nv=23 | 17.60 MiB | 6 (root + 4 intermediate + terminal) | 611,887,247 | 236,712,577 (38.7%) | 374,979,764 (61.3%) |
| bound6, nv=26 | 30.11 MiB | 7 (root + 5 intermediate + terminal) | 907,019,496 | 404,796,229 (44.6%) | 501,999,940 (55.4%) |

(`transcript_init` is negligible: 359 / 338 cycles.)

## Per-marker cycles, bound18 nv=23

Sums over all occurrences (n = occurrence count; one occurrence per fold level
for the per-level spans). Indentation shows nesting; a child is included in
its parent.

| Marker | n | Total cycles | Notes |
|---|---|---|---|
| `deserialize_input` | 1 | 236,712,577 | mostly expanded-setup-matrix decode |
| `akita_verify` | 1 | 374,979,764 | everything below nests here |
| . `prepare_fold_replay` (suffix levels) | 5 | 50,601,594 | EOR replay, transcript absorbs, point prep |
| .. `ring_opening_point` | 6 | 43,357,490 | first occurrence (4,999,490) is the root's, outside `prepare_fold_replay` |
| . `ring_switch_verifier` | 5 | 14,345,227 | root + intermediate levels |
| . `ring_switch_verifier_terminal` | 1 | 3,074,034 | terminal level |
| .. `ring_switch_verifier_core` | 6 | 14,974,414 | |
| ... `prepare_relation_matrix_evaluator` | 6 | 12,966,787 | |
| . `stage1_sumcheck` | 5 | 7,906,189 | terminal level is relation-only (no stage 1) |
| .. `verify_eq_factored_sumcheck` | 5 | 7,829,645 | |
| . `stage2_sumcheck` | 6 | 200,246,859 | |
| .. `verify_sumcheck` | 6 | 200,240,942 | |
| ... `stage2_expected_output_claim` | 6 | 194,361,918 | |
| .... `stage2_witness_eval` | 6 | 25,256,203 | 25,255,398 of it in the single terminal occurrence |
| .... `stage2_relation_matrix_eval` | 6 | 134,013,066 | |
| ..... `setup_contribution` | 6 | 126,832,338 | the V2 setup-matrix scan |
| ..... `structured_chunks` | 6 | 6,361,866 | |
| ..... `setup_prepare_plan` / `_e/t/z_weights` | 6 ea | 12,149,327 / 1,026,917 / 4,871,014 / 6,154,968 | |
| ..... `r_dense` | 6 | 493,512 | |
| . `relation_claim_from_layout_extension` | 6 | 265,117 | |
| `transcript_absorb` | 733 | 13,236,752 | subset of the spans above |
| `transcript_challenge` | 950 | 9,517,588 | subset of the spans above |

Residual inside `akita_verify` not covered by any span: ~93.5M cycles
(claim validation, schedule selection including the planner DP that runs at
verify time for presets without shipped schedule tables, root-level commitment
and point absorbs, `verify_fold_eor` at the root, relation-instance assembly,
grind-nonce check). Finer attribution would need markers inside
`akita-verifier`'s root prep; not needed for the ledger questions below.

### Per-fold-level sequences, bound18 nv=23 (total cycles)

Levels in execution order: L0 = root, L1-L4 intermediate, L5 terminal.

| Marker | L0 | L1 | L2 | L3 | L4 | L5 |
|---|---|---|---|---|---|---|
| `stage2_sumcheck` | 78.23M | 30.28M | 25.55M | 16.31M | 13.02M | 36.85M |
| `setup_contribution` | 70.26M | 20.94M | 16.94M | 8.70M | 5.55M | 4.44M |
| `stage2_relation_matrix_eval` | 70.89M | 23.16M | 18.75M | 9.76M | 6.33M | 5.12M |
| `stage1_sumcheck` | 2.08M | 1.72M | 1.54M | 1.35M | 1.22M | (none) |
| `prepare_fold_replay` | n/a | 10.65M | 19.00M | 10.30M | 5.96M | 4.70M |
| `ring_switch_verifier(_terminal)` | 2.56M | 3.89M | 4.00M | 2.25M | 1.65M | 3.07M |
| `stage2_witness_eval` | 161 | 161 | 161 | 161 | 161 | 25.26M |

The setup scan decays geometrically down the ladder (root dominates); the
terminal level's cost is dominated by the cleartext-witness MLE evaluation.

## Mapping onto the NATIVE-RECURSION-DESIGN.md ledger (bound18 nv=23)

| Ledger row | Measured proxy | Cycles | Share of `akita_verify` |
|---|---|---|---|
| V1 outer/stage sum-check replay (information-theoretic rounds) | `stage1_sumcheck` + (`stage2_sumcheck` − `stage2_expected_output_claim`) | 13,791,130 | 3.7% |
| V2 setup-matrix scan | `stage2_relation_matrix_eval` + `prepare_relation_matrix_evaluator` (core scan alone: `setup_contribution` 126.8M) | 146,979,853 | 39.2% |
| V3 Fiat-Shamir (BLAKE2b absorbs + squeezes) | `transcript_absorb` + `transcript_challenge` | 22,754,340 | 6.1% |
| V4 XOF setup regeneration | not executed (trusted decoder); the shipped-matrix alternative costs `deserialize_input` = 236,712,577 | — | (38.7% of the whole trace) |
| V5 terminal direct ring checks | terminal `stage2_witness_eval` + `ring_switch_verifier_terminal` (whole terminal stage-2: 36.9M) | 28,329,432 | 7.6% |
| V6/V7 | not in scope: single-group PCS opening only, no statement lanes | — | — |
| point prep + EOR + schedule/claims residual | `ring_opening_point` + `prepare_fold_replay` remainder + uncovered residual | ~143M | ~38% |

### Verdict on the ledger

1. **V2 dominance among in-verifier algebra: confirmed.** The setup scan is
   the largest single component (126.8M cycles core, 147M with its
   preparation), 3.5x the next-largest verifier-proper item. The [AK] "78-85%
   of total Akita verifier cost" figure is not reproduced inside the guest
   (39% of `akita_verify`) because RV64 emulation inflates the
   bookkeeping-heavy parts (point preparation, EOR, schedule selection) that
   are nearly free natively; the *ranking* is unchanged.
2. **V4/deserialize trade: confirmed and quantified.** Even with the trusted
   decoder (no XOF), shipping the expanded matrix costs 236.7M cycles of
   decode, 38.7% of the whole trace, dwarfing every verifier component except
   the verify body itself. This is the fp64 analogue of the fp128 `nv=32`
   observation ("dominated by decoding the expanded verifier-setup matrix")
   and directly supports the design note's conclusion that a recursion
   statement must ship the 32-byte seed plus a setup commitment `C_S`, never
   the matrix (and never re-derive it in-circuit).
3. **V5 measured (the ledger marked it low-confidence).** Terminal-level cost
   is 28.3M-36.9M cycles, dominated by the cleartext terminal-witness MLE
   evaluation (25.3M). This lands at the upper end of the ledger's
   $10^6$-$10^7$-mult estimate and **exceeds V3**: the measured in-guest
   ranking is V2 > V5 >= V3 > V1, versus the ledger's conjectured
   V4 > V3 > V5 > V1 (V4 removed here by the trusted decoder). Post-port,
   the terminal level is a bigger residual than the ledger assumed; the
   terminal-closure / pi-prime design in Section 4.1 carries correspondingly
   more of the budget.
4. **V3: consistent in magnitude, smaller in-guest share than the naive-wrap
   model.** 22.75M cycles of BLAKE2b transcript work across 1,683 calls. The
   ledger's $7 \times 10^7$ constraint-equivalent estimate is for a
   circuit-arithmetized sponge over a 224 KB envelope; cycles and constraints
   are different metrics, but the relative claims survive: V3 > V1 (1.6x
   here), V3 << V2 (6.5x smaller).
5. **V1: confirmed tiny.** Pure sum-check round replay is 13.8M cycles
   (3.7%), including its own transcript absorbs; nothing to optimize first.

## Setup-contribution mode: direct vs recursive (superseded 2026-08-01)

The 2026-07-31 revision of this file recorded that recursive mode was
*structurally rejected* at the fp64 presets, with two hard `D == 64` guards
(`recursive_commitment.rs`, `schedule_params/candidate.rs`). **That gate has
since been lifted** (`specs/mixed-d-setup-delegation.md` and
`specs/eor-setup-prefix-absorption.md`): the fp64 `D128FullBound18`
recursive path now plans, proves and verifies end to end at `nv = 23`, both
in `crates/akita-pcs/tests/recursive_setup_fp64_d128_e2e.rs` and in this
harness. The blocker is no longer the dimension gate but the setup envelope
documented in the section above.

## Comparison with the fp128 numbers in README.md

- fp128 D64OneHot `nv=20`: 65.3M-cycle trace on a 1.1 MiB blob.
- fp64 D128FullBound18 `nv=23`: 611.9M-cycle trace on a 17.6 MiB blob; the
  blob is 16x larger and deserialize is 236.7M cycles, i.e. the decode cost
  per byte (~13 cycles/B) matches the fp128 lineage; the verify body grew
  with the deeper 6-level D=128 ladder and the bound-18 dense schedule.
- fp128 `nv=32`: ~8G cycles dominated by `deserialize_input`; the fp64 runs
  sit between the two, with deserialize share growing in blob size exactly as
  that data point predicts.

## bound6 nv=26 (range-group bracket)

Same pipeline, `fp64::D128FullBound6` at `nv=26` (Aerie's range-group arity),
7 fold levels. Total trace 907,019,496 cycles; `deserialize_input`
404,796,229 (44.6%); `akita_verify` 501,999,940 (55.4%).

Per-marker sums (same nesting as the nv=23 table):

| Marker | n | Total cycles |
|---|---|---|
| `prepare_fold_replay` | 6 | 56,993,930 |
| `ring_opening_point` | 7 | 60,507,210 (root occurrence: 17,813,927) |
| `ring_switch_verifier` / `_terminal` | 6 / 1 | 23,600,647 / 2,937,509 |
| `prepare_relation_matrix_evaluator` | 7 | 21,749,886 |
| `stage1_sumcheck` | 6 | 9,449,061 |
| `stage2_sumcheck` | 7 | 284,560,232 |
| `stage2_expected_output_claim` | 7 | 277,297,355 |
| `stage2_witness_eval` | 7 | 24,374,863 (terminal occurrence: 24,373,897) |
| `stage2_relation_matrix_eval` | 7 | 211,597,050 |
| `setup_contribution` | 7 | 201,295,427 |
| `transcript_absorb` | 863 | 14,792,440 |
| `transcript_challenge` | 1,129 | 10,945,251 |

Per-level `setup_contribution` (L0 root ... L6 terminal): 121.99M, 37.69M,
15.55M, 10.67M, 6.54M, 4.74M, 4.12M. Per-level `stage2_sumcheck`: 130.68M,
48.67M, 24.19M, 18.76M, 14.24M, 12.14M, 35.88M (terminal again dominated by
the 24.4M cleartext-witness evaluation).

Ledger mapping (share of `akita_verify`):

| Ledger row | Cycles | Share |
|---|---|---|
| V1 (stage1 + stage2 pure rounds) | 9,449,061 + 7,262,877 = 16,711,938 | 3.3% |
| V2 (relation-matrix eval + evaluator prep; core scan 201.3M) | 233,346,936 | 46.5% |
| V3 (transcript absorb + challenge) | 25,737,691 | 5.1% |
| V5 (terminal witness eval + terminal ring switch; whole terminal stage-2: 35.9M) | 27,311,406 | 5.4% |
| uncovered residual (schedule/claims/root prep) | ~106.3M | ~21% |

The structure matches nv=23: V2 grows with the deeper 7-level ladder and the
larger root scan (root setup scan 122.0M vs 70.3M at nv=23); V5 is essentially
flat (terminal size is planner-capped, not `nv`-driven: 27.3M vs 28.3M); V3
and V1 grow mildly with rounds. The measured in-guest ranking
V2 > V5 ~= V3 > V1 holds at both arities.

## Reproduction

```bash
cd profile/akita-recursion
export CARGO_TARGET_DIR=~/.cache/akita-recursion-target
cargo build --release

AKITA_NUM_VARS=23 AKITA_RECURSION_BLOB=$CARGO_TARGET_DIR/inputs_nv23.bin \
  $CARGO_TARGET_DIR/release/akita-recursion-artifact --preset bound18
ZEROOS_GUEST_RUSTFLAGS=-Zunstable-options AKITA_RECURSION_LOG=info \
  $CARGO_TARGET_DIR/release/akita-recursion-host --preset bound18 --trace-only \
  --trace-output /dev/null --input $CARGO_TARGET_DIR/inputs_nv23.bin

AKITA_NUM_VARS=26 AKITA_RECURSION_BLOB=$CARGO_TARGET_DIR/inputs_nv26.bin \
  $CARGO_TARGET_DIR/release/akita-recursion-artifact --preset bound6
ZEROOS_GUEST_RUSTFLAGS=-Zunstable-options AKITA_RECURSION_LOG=info \
  $CARGO_TARGET_DIR/release/akita-recursion-host --preset bound6 --trace-only \
  --trace-output /dev/null --input $CARGO_TARGET_DIR/inputs_nv26.bin
```

The host binary must be launched with the harness workspace
(`profile/akita-recursion`) as the current directory; the Jolt CLI resolves
`-p akita-recursion-guest` against the cwd's workspace.

Raw logs: `~/.cache/akita-recursion-target/trace_fp64_nv23_bound18.log`,
`~/.cache/akita-recursion-target/trace_fp64_nv26_bound6.log` (kept out of the
repo; regenerate with the commands above).

## Direct vs Recursive setup contribution (2026-08-01, harness v2)

The harness now ships a multi-group blob (`AKJOLTv2`) and a recursive
artifact/guest path, so `--setup-mode recursive` plans and replays the
Recursive schedule at fp64 `D128FullBound18` `nv = 23`. Two new verifier
spans are bridged to Jolt markers:

- `stage3_setup_sumcheck` — the delegated setup-product sumcheck replay
  (`AkitaStage3Verifier::verify_batched_stage3`);
- `eor_replay` — the shared extension-opening reduction replay
  (`replay_eor_reduction`), which at `EXT_DEGREE = 2` runs at every level in
  both modes and, in recursive mode, additionally carries the setup-prefix
  claim (`specs/eor-setup-prefix-absorption.md`).

**Proxy caveat, read before comparing totals.** The Direct row is a
*single-group, single-polynomial* opening; the Recursive row is a
*three-group, four-polynomial* opening (two singleton dense precommits at
`nv/2 = 11` plus a two-poly final group at `nv = 23`), because recursive
setup planning only fires for multi-group keys. The two rows therefore
differ in both mode and batch shape; the mode-attributable comparison is the
per-level `setup_contribution` / `stage3_setup_sumcheck` split, not the
headline total.

### Direct baseline re-measured under v2 (`nv = 23`, bound18)

Blob 18,457,563 B (17.60 MiB), same content as the 2026-07-31 run plus the
v2 header. Totals reproduce the earlier measurement to within 0.15%
(`akita_verify` 375,814,952 vs 374,979,764; `deserialize_input` 236,788,640
vs 236,712,577), which also confirms the added span markers cost ~0.1% and
the v2 multi-group decode path costs nothing at one group.

| Marker | n | Total cycles |
|---|---|---|
| `deserialize_input` | 1 | 236,788,640 |
| `akita_verify` | 1 | 375,814,952 |
| . `stage2_sumcheck` | 6 | 200,240,146 |
| .. `stage2_expected_output_claim` | 6 | 194,359,975 |
| ... `stage2_relation_matrix_eval` | 6 | 133,995,959 |
| .... `setup_contribution` | 6 | 126,817,730 |
| ... `stage2_witness_eval` | 6 | 25,256,203 |
| . `prepare_fold_replay` | 5 | 51,533,599 |
| . `ring_opening_point` | 6 | 43,650,291 |
| . `eor_replay` | 6 | 7,055,767 |
| . `stage1_sumcheck` | 5 | 7,828,134 |
| . `ring_switch_verifier` / `_terminal` | 5 / 1 | 14,285,949 / 3,075,833 |
| `transcript_absorb` / `_challenge` | 733 / 950 | 13,419,794 / 9,333,206 |

Per-level `setup_contribution` (L0 root ... L5 terminal): 70.26M, 20.94M,
16.94M, 8.70M, 5.54M, 4.43M. Per-level `eor_replay`: 1.23M, 1.25M, 1.20M,
1.14M, 1.22M, 1.02M (flat, as expected: the reduction is per-claim-count,
not per-level-size). `stage3_setup_sumcheck` does not appear in Direct mode.

Total trace: 612,603,952 cycles.

### The recursive setup envelope cannot be shipped at all (measured)

First hard result from the recursive artifact run: at `nv = 23` the recursive
setup's expanded shared matrix is **4,194,304 ring elements = 4,294,967,296
bytes (4 GiB)**, against 17,920 ring elements (17.5 MiB) for the direct
setup at the same arity, a 234x inflation. Measured component sizes of the
recursive artifact (`blob component encoded sizes` log line):

| Component | Encoded bytes |
|---|---|
| expanded shared matrix | 4,294,967,312 |
| proof | 122,692 |
| setup-prefix commitments (`C_S`, 2 slots) | 3,572 |

The inflation is the prover-side envelope of the committed setup prefix
(`inflate_envelope_for_setup_prefix_slot` plus the delegating fold's D-matrix
width): the prover must materialize the prefix in order to commit to it.
Consequences:

1. **`--setup-mode recursive --setup-transport full` is impossible**, and not
   because of a harness cap: 4 GiB exceeds the 768 MiB blob limit by 5.6x and
   would dwarf every cycle number in this file. The artifact now skips the
   expanded-matrix transport with a warning when the envelope exceeds the cap.
2. **Seed/slice transport is a prerequisite for recursive mode, not an
   optimization.** This strengthens the design note's conclusion: a recursion
   statement must ship the seed plus `C_S` (3,572 B here), because in
   recursive mode there is no viable "ship the matrix" fallback at all.
3. The proof itself is 122,692 B, i.e. **0.003% of the direct blob's matrix
   payload**. Once the matrix is off the wire the blob is dominated by
   nothing in particular; deserialize should collapse accordingly.

To make the truncated transports work, the verifier-side capacity check was
split from the prover's: `ensure_schedule_fits_verifier_setup`
(`crates/akita-config/src/proof_optimized.rs`) excludes a consumed
setup-prefix slot from the required envelope exactly when the verifier holds
that slot's public commitment, since the delegated stage-3 sumcheck opens
`C_S` instead of reading the prefix rows. Slots missing from the registry are
still counted (fail-closed), and every remaining matrix access stays
bounds-checked at read time (`FlatMatrix::ring_view`).

### Recursive prove/verify works; the blob does not fit at any delegating arity

The recursive artifact path runs end to end at `nv = 23`
(`gen_recursive3.log`): recursive setup 335 s, **8 fold levels**, prove
758 s, native `host-side verify OK` in 0.53 s under
`RecursiveCommitmentConfig<D128FullBound18>` +
`SetupContributionMode::Recursive`, with the two-group profile key (two
singleton dense precommits at `nv/2 = 11` plus the two-poly final group).
The guest measurement is nevertheless **not reachable**, for a reason that is
itself the headline result.

Binary search over native verification with a truncated shared matrix (the
artifact's `--setup-transport slice|seed` path probes this automatically)
puts the verifier-side requirement at **2,162,688 ring elements =
2,214,592,512 B (2.06 GiB)**, against the 768 MiB blob cap. Probes below
that threshold fail in ~6 ms (the capacity check), probes at or above it
verify in ~0.5 s.

Per-term breakdown of the `nv = 23` recursive schedule (ring elements at
`D = 128`; the level-1/2 prefix slots are the delegated ones the verifier
does not materialize):

| Level | A | B | D | consumes prefix |
|---|---|---|---|---|
| L0 (root) | 672 | **2,162,688** | 180,312 | no |
| L1 | 24,675 | 28,160 | 50,688 | yes (prefix_ring_len 4,194,304) |
| L2 | 24,990 | 14,080 | 14,080 | yes (prefix_ring_len 1,048,576) |
| L3-L8 | ≤ 11,575 | ≤ 14,080 | ≤ 2,816 | no |

So the binding term is the **root level's B (outer) matrix**, not the
delegated prefix: the 4,194,304-element prefix materialization is already
excluded on the verifier side by `ensure_schedule_fits_verifier_setup`, and
removing it still leaves 2.06 GiB.

Shape sweep over the recursive profile key (schedule resolution only, no
proving), reporting the verifier-side requirement and whether the schedule
actually delegates:

| final_nv | verifier ring elements | binding term | bytes | fits 768 MiB | delegates |
|---|---|---|---|---|---|
| 14 | 1,008 | L1.A | 1.0 MB | yes | **no** |
| 16 | 3,584 | L0.A | 3.7 MB | yes | **no** |
| 18 | 4,480 | L0.A | 4.6 MB | yes | **no** |
| 20 | 8,960 | L0.A | 9.2 MB | yes | **no** |
| 22 | 17,920 | L0.A | 18.4 MB | yes | **no** |
| 23 | 2,162,688 | L0.B | 2.21 GB | no | yes |
| 24 | 2,162,688 | L0.B | 2.21 GB | no | yes |
| 26 | 2,162,688 | L0.B | 2.21 GB | no | yes |

(`pre_nv = final_nv/2` and `final_nv/2 + 1` give identical results;
`final_nv = 12` is unschedulable for this key.)

**Conclusions, stated as bad news first.**

1. **The delegating schedule appears only at `nv >= 23`, and exactly at the
   arities where it appears, the verifier's setup requirement jumps 120x
   (17,920 -> 2,162,688 ring elements).** Every arity whose blob would fit
   plans a non-delegating schedule, i.e. is not a recursive-mode measurement
   at all. There is therefore no arity at which this harness can compare
   Direct and Recursive in-guest today.
2. **No blob transport rescues it.** Slice ships 2.06 GiB (2.7x over cap).
   Seed-derived ships 32 bytes but obliges the guest to re-derive 2,162,688
   ring elements through SHAKE256 (121x the 17,920-element direct matrix
   whose in-guest XOF regeneration was already estimated at ~3e9 cycles),
   which is exactly the explosion the seed-only scope guard forbids. The
   seed-only path is implemented and validated (see below) but is not
   usable at this requirement.
3. **This falsifies, at these shapes, the premise that setup delegation
   shrinks the verifier's setup footprint.** Stage-3 delegation removes the
   root-level *scan* of the setup product, but the schedule that carries it
   inflates the root's B-matrix envelope far beyond what the scan cost. On
   the measured direct baseline the entire setup scan is 126.8M cycles
   (39% of `akita_verify`); the recursion route as planned today trades that
   against a setup artifact 120x larger.
4. **Not caused by the D=128 lift: the shipped upstream catalog behaves
   identically.** Resolving the same two-group key under the
   *upstream-blessed* `fp128::D64OneHot` recursive catalog reproduces the
   pattern exactly, and the same key under the non-recursive config does
   not:

   | Config / key | verifier ring elements | binding | delegates |
   |---|---|---|---|
   | fp128 D64OneHot recursive, nv=16 | 1,608 | L1.A | no |
   | fp128 D64OneHot recursive, nv=20 | 524,288 | L0.B | **yes** |
   | fp128 D64OneHot non-recursive, nv=20 | 8,192 | L0.A | no |
   | fp128 D64OneHot recursive, nv=23 | 1,048,576 | L0.B | **yes** |
   | fp128 D64OneHot non-recursive, nv=23 | 16,384 | L0.A | no |
   | fp64 D128 non-recursive, nv=23 (same key) | 28,160 | L0.B | no |
   | fp64 D128 recursive, nv=23 | 2,162,688 | L0.B | **yes** |

   The inflation switches on with delegation (64x at fp128 D64, 77x at fp64
   D128 against the same key), in a catalog that predates this work. So the
   root-B blowup is a property of the recursive schedule shape upstream, not
   of the mixed-D setup delegation added here. It also explains why the
   upstream +8.3% recursion figure was never contradicted: it was not
   measured through a blob that had to carry this envelope.
5. **Resolved by experiment: the capacity requirement over-estimates the
   real read footprint by 8x, and the real one fits.** See the next section.

### Seed-only / sliced setup transport (implemented, validated, not yet payable)

The blob format carries a `SetupTransport` discriminant:
`ExpandedMatrix` (full, required for Direct), `MatrixSlice` (verifier-read
prefix only) and `SeedDerived` (32-byte `public_matrix_seed` only, guest
re-derives the prefix). Truncated transports are accepted only for
`SetupContributionMode::Recursive` and are rejected at decode otherwise,
because Direct replay scans the matrix. Strict decoding re-derives the
prefix from the seed and rejects a mismatching slice; the trusted-benchmark
decoder skips only the slice comparison (the seed-derived path has no bytes
to trust and is strict by construction). `C_S` travels in `prefix_slots`
(3,572 B for the two slots at `nv = 23`), so the delegated opening is
publicly bound without the matrix. Unit coverage for all three transports,
the prefix-stability property of `derive_public_matrix_flat`, and the
mode/transport rejection lives in `profile/akita-recursion/glue/src/lib.rs`
(11 tests).

## Read high-water mark vs layout capacity (2026-08-01, measured)

**Question.** Is the 2,162,688-ring-element requirement a genuine read, or an
artifact of the layout-capacity precheck?

**Method.** Enumerate every read path into the expanded setup, then bypass
the precheck (temporary debug escape hatch in
`ensure_schedule_fits_verifier_setup`, since removed) and repeat the
truncated-matrix binary search, so verification fails at real reads instead.
The read paths are six, all bounds-checked, and none of them are the
precheck:

| # | Site | View |
|---|---|---|
| 1 | `akita-verifier/.../core/verify.rs:224` | A, direct-witness recommitment |
| 2 | `akita-verifier/.../core/verify.rs:283` | B, direct-witness recommitment |
| 3 | `akita-verifier/src/stages/stage3.rs:378` | setup MLE scan (non-delegated only) |
| 4 | `akita-types/.../setup_contribution/plan/scan/mod.rs:97` | setup-contribution scan |
| 5 | `akita-types/src/proof/relation_matrix_cols.rs:143` | D, relation-matrix columns |
| 6 | `akita-types/src/proof/relation_matrix_cols.rs:220-221` | A and B, per group |

**Result (outcome (a)).** The true read high-water mark at `nv = 23`
recursive is **270,336 ring elements = 276,824,064 B (264 MB)**, against the
2,162,688-element (2.06 GiB) capacity requirement: an **8.0x
over-estimate**, and 264 MB is **comfortably inside the 768 MiB blob cap**.
Probes are monotone and the boundary is sharp (270,335 fails, 270,336
passes).

**What actually binds.** Every failing probe reports
`suffix verify level 2 failed: InvalidSetup("shared matrix is too small
...")`, and 270,336 is exactly the **level-2 setup-prefix group's A matrix**
from the schedule breakdown above. So the binding read is the A matrix of
the carried setup-prefix group, i.e. the delegated prefix participating as a
precommitted group at its consuming level.

Two corrections to the earlier analysis follow, and both matter:

1. **The root's B matrix (2,162,688) is never read.** `relation_matrix_cols`
   would build a B view of exactly `n_b = 2` x `b_width = 1,081,344`
   (`k=2` x `n_a=6` x `digits_open=22` x `num_blocks=4096`) if that path ran
   at the root, but verification of this proof succeeds with 8x less, so it
   does not. The layout-capacity precheck counts a matrix the folded root
   never touches. Conclusion 3 above should therefore be read narrowly: the
   *envelope* inflates 120x, but the verifier's *work* does not.
2. **The verifier-side filter is too aggressive in one direction and the
   precheck too conservative in the other.** `ensure_schedule_fits_verifier_setup`
   currently excludes the whole prefix term (storage + A + B) whenever the
   slot commitment is in the registry, yet the prefix group's **A matrix is
   read**; conversely it counts root A/B that are not. Only the prefix
   *storage* (1,048,576 / 4,194,304 elements) is genuinely never read. The
   check happens to be safe today only because it over-counts globally.

**Consequence: the recursive in-guest measurement is unblocked in principle**
(264 MB slice fits), and needs exactly two changes, neither of which should
be made by guessing:

- Replace the layout-capacity precheck with a read-footprint bound: count
  each consumed setup-prefix slot's A/B (read) while excluding its storage
  (not read), and stop counting root A/B for folded roots. Reads stay
  bounds-checked at `ring_view*`, so the precheck is defense-in-depth, but
  narrowing a setup-sufficiency check is protocol-adjacent and wants an
  owner's decision rather than a harness-driven tweak.
- Raise the setup-prefix registry decode cap. With the 264 MB slice the run
  now gets past sizing and fails later, at blob round-trip:
  `Sequence length 536870912 exceeds maximum 67108864`. The level-1 slot's
  `n_prefix = 536,870,912` and `natural_len = 276,824,064` both exceed
  `MAX_SETUP_MATRIX_FIELD_ELEMENTS = 2^26` used when decoding
  `SetupPrefixVerifierRegistry`. This is a constant, not a cost, but the
  blob cannot round-trip the prefix metadata until it is raised.

With those two in place the Direct-vs-Recursive guest comparison is a
mechanical re-run of the pipeline already built here (artifact
`--setup-mode recursive --setup-transport slice,seed`, then the host trace).

## Precheck tightening + decode bound (2026-08-01): 3.4x closer, still one cap short

> **CODE-STATE LABEL.** Everything in this section was measured on a binary
> linked at 19:41 local. A soundness fix to the recursive suffix (an extra
> transcript absorb plus a squeezed batching coefficient) landed in
> `akita-verifier/src/protocol/core/{suffix,fold}.rs` at 19:47, *after* that
> link and while the prove was running, and `akita-prover` files were still
> being edited afterwards. The binary itself was therefore internally
> consistent (a linked binary does not change under source edits), but these
> numbers **predate the absorption fix** and describe a protocol revision now
> known to be unsound. The expected shift is negligible (one absorb, one
> squeeze) but it is stated here rather than assumed, and any recursive-mode
> figure below must be re-measured after that fix before being quoted.

### CHANGE 1: verifier precheck sized to reads, not prover layout

`ensure_schedule_fits_verifier_setup` now counts only verifier-read
footprints: a consumed setup-prefix slot's A/B (read at the consuming level),
its storage **only** when the slot's commitment is absent from the registry
(fail-closed), and the root direct-commitment footprint **only** for a direct
(zero-fold) root. It no longer counts fold levels' own A/B/D, which are
prover-side: `compute_relation_matrix_col_evals`, the only consumer that
views them, has no verifier caller. The read-time guards
(`FlatMatrix::ring_view*`, the setup-contribution scan, the stage-3 setup
MLE) are unchanged and remain the load-bearing check.

The filter bug found earlier was fixed in the same change: prefix A/B are now
always counted (they are read); only storage is conditional. Regression guard
`verifier_envelope_counts_prefix_a_even_when_committed`
(`crates/akita-config/src/setup_prefix_slots.rs`) fails if that regresses.

Measured effect at fp64 `D128FullBound18` `nv = 23` recursive:

| Quantity | Ring elements | Bytes | Fits 768 MiB cap |
|---|---|---|---|
| Old prover-shaped precheck | 2,162,688 | 2.21 GB | no |
| **New verifier precheck** | **630,784** | **646 MB** | **yes** |
| True read high-water mark | 270,336 | 277 MB | yes |

So the precheck went from 8.0x to 2.3x over the real read footprint, a 3.4x
reduction, and **crossed under the blob cap**. The residual 2.3x is the
level-1 prefix slot's B term (630,784), which the precheck counts and this
proof does not read; tightening further would need per-level knowledge of
which prefix role is touched, and was deliberately left conservative.

### CHANGE 2: bound the allocated prefix, not the declared capacity

The blob decoder previously rejected on `seed.matrix_field_elements()`, i.e.
the *prover's* declared envelope (4,194,304 x 128 = 2^29 > the 2^26 decode
cap), even though a truncated blob allocates only its shipped prefix. The
check now bounds `shipped_setup_ring_len * gen_ring_dim`, the quantity that
actually drives allocation. This is **not** a ceiling widening: for
`ExpandedMatrix` the two coincide, and for the truncated transports the
allocated length is strictly smaller. The declared capacity remains bound
into the transcript through the instance descriptor's setup-seed digest.
Covered by `decode_bounds_the_allocated_prefix_not_the_declared_capacity`
(`profile/akita-recursion/glue/src/lib.rs`).

Production note: the DoS surface at this boundary is the *materialized*
prefix, so bounding it (rather than the metadata) is the right guard; a
production decoder should additionally validate the declared slot geometry
against the schedule before allocating, which this harness cannot do because
the decoder has no schedule in hand.

### Still blocked, one layer deeper

With both changes the run gets past sizing (646 MB slice, under the cap) and
fails later, at blob round-trip:

```
error: decode jolt inputs blob (round-trip) failed:
       Sequence length 80740352 exceeds maximum 67108864
```

**Site pinpointed, and it was not `natural_len` nor `akita-types`.**
80,740,352 is the *field-element count of the shipped slice*:
630,784 ring elements x 128. It was rejected by the `max_field_elements`
argument of `FlatMatrix::deserialize_with_expected_shape`
(`crates/akita-types/src/layout/flat_matrix.rs:281`) - a **caller-supplied
parameter**, which the blob decoder was passing as
`MAX_SETUP_MATRIX_FIELD_ELEMENTS = 2^26`. The coincidence with the level-2
slot's `natural_len` (also 80,740,352, being the same quantity in field
elements) is what made the earlier reading look like slot metadata.

**Fixed in the blob decoder** (`profile/akita-recursion/glue/src/lib.rs`,
`max_blob_setup_field_elements`), same principle as the shipped-length bound:
the DoS surface is memory the decoder allocates, and the input is already
capped at `MAX_JOLT_BLOB_BYTES`, so the bound is derived from the blob cap
(768 MiB / 8 B = 100,663,296 field elements for fp64) rather than from an
unrelated setup constant. The tighter, input-derived guard
(`check_setup_matrix_bytes_available`, payload must fit the bytes actually
remaining) is unchanged and still runs first. Note the bound is not
uniformly looser: for a 16-byte field it is *tighter* than the old constant,
which is the point - it tracks what the input can carry. Seed-derived
transport is held to the same ceiling so in-guest derivation cannot be driven
unbounded by metadata alone. Regression test
`setup_matrix_decode_bound_is_derived_from_the_blob_cap` (glue, 13 tests
green). **`akita-types` was not modified**, so there is no overlap with the
in-flight soundness fix.

**Consequence: no recursive blob has been written yet, so the
Direct-vs-Recursive comparison table still does not exist.** The Direct
baseline stands (measured, above); the recursive rows remain unmeasured. All
known blockers are now cleared in code (precheck 646 MB < cap, decode bound
blob-derived), but the re-run is deliberately **not** being done on this code
state: it must sit on top of the recursive-suffix absorption soundness fix,
since numbers taken before it describe a protocol revision known to be
unsound.

## Re-run on the FIXED protocol (2026-08-02): blocked one layer further, new bug found

Fired against the post-soundness-fix tree (recursive-suffix absorb + squeezed
batching coefficient, including the sigma hoist; verifier `stage3.rs`/`fold.rs`
and prover `akita_stage3` rebuilt from source at 01:5x). Machine verified idle
before launch; single sequential detached pipeline.

Progress relative to the previous attempt: **both earlier blockers are
cleared.** The verifier precheck admits the recursive schedule at 630,784 ring
elements (646 MB, under the 768 MiB cap), and the blob-derived decode bound
lets the sliced matrix through. The run now reaches the prefix-registry
round-trip and fails there:

```
error: decode jolt inputs blob (round-trip) failed:
       Invalid data: setup prefix commitment row has 256 coefficients, expected 128
```

**Diagnosis.** `SetupPrefixVerifierSlot::check()`
(`crates/akita-types/src/proof/setup_prefix.rs`) requires every commitment row
to have exactly `id.d_setup` coefficients. But the commitment is materialized
as a **single flattened row holding all of `u`**
(`crates/akita-prover/src/api/setup_prefix.rs`:
`rows: vec![RingVec::from_ring_elems(&u)]`), so
`row.coeff_len() = n_b * d_setup`. Here `n_b = 2`, giving 256 against an
expected 128.

Note the 2x is `n_b`, **not** `EXT_DEGREE` - the two coincide numerically at
this preset and the coincidence is misleading. The invariant holds for the
only shipped recursive catalog (`fp128::D64OneHot`, whose setup-prefix
commitment has `n_b = 1`, so `coeff_len = 64 = d_setup`) and breaks for any
prefix commitment with more than one B row, which the fp64 D128 recursive
profile has.

**Why nothing caught it.** The `nv = 23` recursive e2e serializes the *proof*,
never the verifier setup, so the registry's `Validate::Yes` path is not
exercised anywhere in the test suite. The blob pipeline is the first consumer
that round-trips a `SetupPrefixVerifierRegistry`.

**Not resolved here, deliberately.** Two candidate resolutions, and choosing
between them is a representation decision rather than a harness detail:

1. Relax the check to `row.coeff_len() % id.d_setup == 0` with the quotient
   pinned to the slot's B row count (accepting the flattened-row encoding), or
2. Change the encoding to one `RingVec` per element of `u` and keep the strict
   per-row `d_setup` invariant.

Both touch `akita-types` setup-prefix code adjacent to files another agent has
been editing, so this is left for sequencing rather than taken unilaterally.

**Status of the comparison table: still not produced.** Steps 4-6 of the
pipeline (the three guest traces) never ran; the run aborted in step 3. The
Direct baseline was regenerated successfully on the fixed protocol
(18,457,563 B blob, prove 3.56 s, host verify OK) and is byte-identical in
size to the pre-fix Direct blob, consistent with the soundness fix touching
only the recursive suffix. Recursive probe numbers on the fixed protocol
match the pre-fix ones exactly (minimal read prefix 630,784 ring elements,
proof 123,262 B vs 122,692 B pre-fix, the difference being the added absorb
and squeezed coefficient), which is the expected negligible shift.

# MEASURED: Direct vs Recursive comparison (2026-08-02, fixed protocol)

**Protocol state: post-soundness-fix, post-sigma-hoist.** All rows below were
measured on one binary built at 05:48 on 2026-08-02, after the recursive-suffix
absorb + squeezed batching coefficient and the sigma hoist landed, and after
the transcript agent's final edits (its later change touched only an
`akita-pcs` test file, not linked into these binaries). **Synthetic proxy, not
an Aerie/Falcon result**, and note the shape asymmetry below.

## Headline

| Run | Blob | Total trace | `deserialize_input` | `akita_verify` |
|---|---|---|---|---|
| Direct, expanded matrix | 18,457,563 B | 608,038,367 | 232,222,123 | 375,815,884 |
| Recursive, matrix slice | 646,058,724 B | 12,445,161,517 | 8,116,805,137 | 4,328,356,020 |

**Shape asymmetry, stated up front:** Direct is 1 group / 1 polynomial;
Recursive is 3 groups / 4 polynomials (two dense precommits at `nv/2 = 11`
plus a two-poly final group at `nv = 23`), because recursive setup planning
only fires for multi-group keys. Part of the gap is batch shape, not
delegation. The per-level setup analysis below isolates the delegation effect
and does not depend on the asymmetry.

## Q1: does delegation reduce in-guest verifier cycles? No - it increases them 11.5x

`akita_verify` goes from 375,815,884 (Direct) to 4,328,356,020 (Recursive):
**11.5x worse**. The two components responsible are both delegation-specific:

| Component | Direct | Recursive | Ratio |
|---|---|---|---|
| `setup_contribution` (the scan) | 126,817,752 (n=6) | 1,213,967,481 (n=7) | 9.6x |
| `stage3_setup_sumcheck` (delegation) | absent | 1,863,803,105 (n=2) | new cost |
| `eor_replay` | 7,055,767 (n=6) | 10,949,109 (n=9) | 1.6x |
| `stage2_witness_eval` | 25,256,213 | 25,046,751 | 1.0x |

Delegation does not remove the setup scan; it adds a 1.86G-cycle stage-3
sumcheck **on top of** a setup scan that is itself 9.6x larger than the one it
was supposed to replace.

## Q3: the root scan does vanish - and reappears 15x larger one level down

Per-level `stage2_relation_matrix_eval` and `setup_contribution`
(L0 = root; recursive has 9 fold levels, Direct 6):

| Level | Recursive relation-matrix eval | Recursive setup scan | Direct setup scan |
|---|---|---|---|
| L0 (root) | 4,014,391 | **none (delegated)** | 70,262,965 |
| L1 | 11,884,218 | **none (delegated)** | 20,937,777 |
| L2 | **1,111,974,261** | **1,100,706,028** | 16,938,311 |
| L3 | 66,284,093 | 61,586,461 | 8,704,891 |
| L4 | 19,859,633 | 17,956,965 | 5,544,207 |
| L5 | 17,430,584 | 15,749,232 | 4,429,579 |
| L6 | 9,277,522 | 8,249,337 | - |
| L7 | 6,124,369 | 5,367,054 | - |
| L8 | 5,014,043 | 4,352,404 | - |

`stage3_setup_sumcheck` fires exactly twice, at L0 and L1 (the two levels
carrying setup-prefix slots): 1,421,419,757 and 442,383,348.

So the geometric-tail prediction is **half right and half wrong**:

* **Confirmed:** the root scan is gone. L0 has no `setup_contribution` at all,
  and its relation-matrix eval collapses from 70.3M (Direct) to 4.0M - a 17.5x
  reduction exactly where delegation was supposed to act.
* **Falsified:** the residual is not a decaying tail from a smaller base. The
  scan reappears at **L2 at 1,100,706,028 cycles**, 15.7x the Direct *root*
  scan it replaced and 65x the Direct scan at the same level. From L3 down the
  tail does decay geometrically (61.6M, 18.0M, 15.7M, 8.2M, 5.4M, 4.4M) and is
  broadly comparable in shape to Direct's (20.9M, 16.9M, 8.7M, 5.5M, 4.4M).

**Mechanism.** The level that consumes the carried prefix scans the *padded
prefix* rather than the original setup product: 630,784 ring elements against
the 17,920 the Direct root scans, a 35x larger scan. This is the same 630,784
figure that the verifier-read high-water measurement produced, now visible as
cycles. Delegation therefore does not eliminate the setup scan at these
shapes - it **relocates it one level down and enlarges it**, then charges an
extra 1.86G cycles of stage-3 sumcheck for the privilege.

## Q2: transports vs the 236.8M expanded-matrix baseline

| Transport | Blob | `deserialize_input` | vs 232.2M baseline |
|---|---|---|---|
| Direct, expanded matrix | 18,457,563 B | 232,222,123 | 1.0x |
| Recursive, matrix slice | 646,058,724 B | 8,116,805,137 | **34.9x worse** |
| Recursive, seed-derived | 135,892 B | not completed (see below) | - |

The slice transport is catastrophic for deserialization: the verifier-read
prefix is 630,784 ring elements (646 MB), 35x the Direct blob's matrix, and
deserialize scales with it at a near-constant ~12.6 cycles/byte (8.117G /
646 MB), matching the 12.9 cycles/byte measured on the Direct blob. Slicing
the matrix helps only relative to shipping the *recursive* prover envelope
(4 GiB, which does not fit the blob cap at all); it is far worse than the
Direct baseline it would have to beat.

### Seed-derived transport: measured blob, unmeasured deserialize, and why it no longer matters

The seed-derived blob is **135,892 B** - 136x smaller than the Direct blob and
4,754x smaller than the slice - because it carries only the 32-byte
`public_matrix_seed`, the schedule/instance metadata, the two setup-prefix
commitments `C_S` (3,572 B) and the proof (123,262 B). That is the transport
the design note wants.

Its `deserialize_input` was **not measured**: the guest must re-derive the
630,784-ring-element read prefix (80,740,352 field elements) through SHAKE256
inside the emulator. The trace ran 27 minutes without completing that single
marker and was stopped. Order-of-magnitude, ~645 MB of XOF output is ~4.75M
Keccak-f permutations; at the emulation throughput measured on the slice run
(8.117G cycles in ~75 min = 1.80M cycles/s) that is hours of wall clock and
of order 10^10 cycles - i.e. the seed-only path trades an 8.1G-cycle
deserialize for an XOF derivation of the same order or worse. This is exactly
the explosion the seed-only scope guard anticipated, now with the prefix size
(630,784 rather than 17,920 ring elements) that makes it 35x worse than the
case the guard was written for.

**It does not change the verdict, and that is why the run was stopped rather
than continued.** The transport only affects `deserialize_input`. Recursive
`akita_verify` alone is 4,328,356,020 cycles, which is **7.1x the entire
Direct trace** (608,038,367 including its deserialize). So even a transport
with *zero* deserialization cost leaves recursion 7.1x more expensive than
Direct at these shapes. No transport choice can rescue it; the cost is in the
verifier body, per Q1.

## Verdict

At Aerie's fp64 `D128FullBound18` `nv = 23` shape, on the fixed protocol,
**setup delegation as currently planned makes the in-guest verifier
substantially more expensive, not less**:

1. `akita_verify` 11.5x worse (4.33G vs 376M), of which the delegation-
   specific parts are a new 1.86G stage-3 sumcheck and a setup scan that is
   9.6x larger than the one delegation was meant to remove.
2. The root scan is genuinely eliminated (70.3M -> 0, relation-matrix eval
   17.5x lower at L0), confirming the mechanism works locally, but the scan
   reappears at L2 at 1.10G cycles because the consuming level scans the
   padded prefix (630,784 ring elements) instead of the setup product
   (17,920).
3. Every shipping route is worse than Direct: the prover envelope (4 GiB)
   does not fit the blob cap; the slice (646 MB) costs 8.1G cycles to
   deserialize, 34.9x the Direct baseline; and the seed-only blob (136 KB),
   while ideal on the wire, moves that cost into in-guest XOF derivation of
   the same order.

The one-line summary for the design note: **delegation relocates and enlarges
the setup scan rather than removing it, and the carried-prefix padding is
what makes it worse.** The lever that would change this verdict is the size of
the padded prefix the consuming level must scan, not the transport and not
the EOR absorption.
