# Spec: Shared Extension-Opening Reduction over the Carried Setup-Prefix Claim

| Field         | Value                          |
|---------------|--------------------------------|
| Author(s)     | research fork (aerie-mains-substrate) |
| Created       | 2026-08-01                     |
| Status        | active                         |
| PR            | (fork branch, no PR)           |
| Supersedes    |                                |
| Superseded-by |                                |
| Book-chapter  |                                |

## Summary

Recursive setup offloading is blocked at extension claim fields (fp64 Ext2,
`k = EXT_DEGREE = 2`): the recursive suffix disables the extension-opening
reduction (EOR) whenever the level carries a setup prefix, so the carried
stage-3 point violates the ring-subfield psi-packing constraint at the next
fold (`specs/mixed-d-setup-delegation.md`, "Discovered blocker"). The paper
specifies the fix ([AK] lines 2181-2183): when several claims are proved
together, the EOR runs as **one shared degree-2 sum-check across all
claims**; each claim keeps its own reduced value at the common challenge
point, and the reduced openings feed the batched front-end. The carried
setup-prefix claim is one more claim in the fold's shared EOR. This spec
implements that composition. The `G = 2` carried-opening absorption itself
(the two-group suffix batch with the prefix as a precommitted group) already
exists and is proven at `k = 1` (fp128 D64OneHot recursive e2e); the missing
protocol unit is exactly the EOR composition, so no split is needed — this
change lands the absorbed end state directly.

## Phase-1 determination (OPEN-IMPL from `_docs/EOR-SETUP-PREFIX.md`)

**Determination (corrected during implementation — the empirical fixtures
overruled the first reading): the slot identity does NOT need to key on `k`
(one config, one `k`, one setup artifact), but the committed PAYLOAD is
`k`-dependent.** At `k > 1` the slot must commit the committed-transformed
representation of the padded flat stream (adjacent-pack into `E`, chunk
`D/k`, psi-embed each chunk — exactly `DensePoly::tensor_packed_extension_poly`),
because at `k > 1` every fold opens "the committed transformed polynomial"
(`specs/extension-field-trace-cutover.md`,
`specs/extension-field-opening-batching.md`: root tensor projection moves
the transformed witness to the commitment boundary; recursive witnesses are
born in this representation). The claim/EOR side keeps the RAW flat stream:
stage 3 carries `s_rho = S~^flat(rho)` on the raw MLE, the shared EOR
reduces the raw adjacent-packed polynomial, and the fold-side opening of the
transformed commitment at the per-group psi-packed point equals that reduced
value (pinned by
`setup_prefix_fold_opening_matches_packed_mle_at_reduction_point` and its
degree-one control). At `k = 1` the transform is the identity, so the
fp128 D64 artifacts are byte-identical. Evidence for the raw-side claim
equivalence:

- Commit path (`akita-prover/src/api/setup_prefix.rs:78-93` and
  `extract_setup_prefix_ring_elems` at `:173-196`): contiguous
  `D`-coefficient chunks of `shared_matrix().as_field_slice()`, zero-padded,
  committed through `DenseCommitInput::CoeffBlocks` — the same kernel that
  commits ordinary dense witnesses
  (`akita-prover/src/backend/dense/commit.rs:59`). The function is generic
  over `F` only; no extension type appears anywhere on the commit path.
- Open path (`akita-prover/src/backend/recursive/setup_prefix_source.rs:158-172`,
  `setup_prefix_field_evals`): the same flat zero-padded stream feeds the
  fold kernels. At `k > 1` the psi packing is a claim-side basis relabeling:
  `tensor_packed_witness_evals`
  (`akita-types/src/extension_opening_reduction.rs:63-115`) reads entry
  `tail*k + head = f(head, tail)` — a reinterpretation of the same
  little-endian flat order with no data permutation — and
  `derive_tensor_extension_opening_claim` recovers exactly the plain
  extension MLE `f(p)` of the same table.
- Cross-check: `k = 2` openings of plain-committed base-field precommitted
  groups are already exercised end to end by the grouped-root EOR
  (`akita-pcs/src/scheme/tests/mixed_bound_multi_group.rs`, fp32 ext-4 and
  fp64 ext-2 probes) with the same commitment format.
- What the `k = 1` restriction actually gates is three claim-path shortcuts,
  not the commitment layout: (a) prover suffix
  `needs_extension_reduction = EXT_DEGREE != 1 && setup_prefix.is_none()`
  (`akita-prover/src/protocol/core/suffix.rs:305`); (b) the verifier mirror
  plus an explicit `EOR present && num_groups != 1 => InvalidProof` rejection
  (`akita-verifier/src/protocol/core/suffix.rs:245` and the guard below it);
  (c) stubbed tensor kernels for the `SetupPrefix` source
  (`setup_prefix_source.rs:340-427`, "does not support extension tensor
  projection/packing").

## Mathematical content (all pieces imported or elementary)

Notation: shared suffix point `P` of length `L` (the longer of the carried
setup point and the witness point; the shorter is a suffix of the longer,
`BatchedStage3Geometry::shared_suffix_point`). EOR split `kappa =
log2(k)`; joint tail `P[kappa..]` of length `N = L - kappa`. Group `g` has
its own point `p_g = P[pad_g..]` (`pad_g = L - len_g`; one group always has
`pad = 0`).

1. **Shared EOR with suffix-aligned spans.** The grouped EOR (already
   implemented for prefix-aligned root groups,
   `akita-prover/src/protocol/core/extension_opening_reduction.rs`,
   `eor_claim_spans` and the virtual tiling) generalizes to suffix-aligned
   spans by the mirrored lifting: lift the shorter group's packed table
   `g_S` (over its own tail, `n = len_g - kappa` vars) to the joint tail by
   constant extension over the LOW `N - n` variables,
   `g~_S(x) = g_S(x >> pad_g)`. Then, with the transparent factor `A_eta`
   built over the FULL tail `P[kappa..]`:
   `sum_x g~_S(x) * A_eta(x) = sum_w g_S(w) * A_eta^{own}(w)` where
   `A_eta^{own}` is the factor at the group's own tail `P[pad_g + kappa..]`
   — by partition of unity of the equality kernel in the summed low
   coordinates (the exact mirror of the tiled identity used by grouped
   roots, whose docstring records the prefix version). Consequently the
   proof format is unchanged: one shared sum-check, one `final_claim`, one
   `final_factor` (all terms share the full-tail factor), and the final
   per-term value is `g_S(rho[pad_g..])` because the MLE of the
   constant-extension is `g~_S(rho) = g_S(rho[pad_g..])`.
2. **Per-claim partial checks.** Each claim's column partials are computed
   at its OWN routed point (`p_g`), and the verifier's per-claim
   consistency check `derive_tensor_extension_opening_claim_from_partials`
   must use the claim's own head `P[pad_g..pad_g+kappa]` (for prefix-aligned
   root groups `pad_g = 0`, so the existing full-point call is the special
   case and stays byte-identical).
3. **Per-group next opening points.** After the reduction, group `g`'s
   opening point is the psi-packed form of its own slice of the sum-check
   point: `packed_g = [rho[pad_g .. pad_g + (alpha - kappa)], 0^kappa,
   rho[pad_g + (alpha - kappa) ..]]` (the per-group application of
   `ring_subfield_packed_extension_opening_point`), followed by the group's
   existing block-structure routing (the column-major m/r arrangement for
   the setup group, identity for the witness group), then
   `prepare_opening_point` — whose inactive-window zero constraint is now
   satisfied by construction. This is the suffix analogue of the grouped
   root's `prepare_multi_group_points` (which truncates the single packed
   point, valid only for prefix alignment).
4. **Final relation, and the per-claim binding that makes it sound.**
   `final_claim = (sum_g rho_g * opening_g(prepared_g)) * final_factor`,
   where the `rho_g` are **transcript-squeezed batching coefficients**, one
   independent `CHALLENGE_EVAL_BATCH` draw per claim
   (`sample_public_row_coefficients`). The prover-side check is
   `compute_trace_target`
   (`akita-prover/src/protocol/core/fold_kernels.rs`); the verifier consumes
   `trace_eval_target = final_claim` with per-claim scales `final_factor`
   and per-claim coefficients `rho_g`, mirroring the grouped root
   (`akita-verifier/src/protocol/core/root_fold.rs`), and the trace table
   multiplies them together at
   `akita-types/src/trace_weight/stage2.rs` (`output_scale *
   row_coefficients[claim] * scale`).

   **Corrected claim (this item previously said `coeff = 1` per claim, with
   "per-group binding continues to come from the stage-1 group challenges").
   That justification was FALSE and the unit coefficients were a soundness
   defect.** `derive_multi_group_stage1_challenges` takes
   `(transcript, v_coeffs, ring_d, opening_batch, lp, row_layout,
   grind_nonce)`: no claimed evaluation value reaches it, so the stage-1
   group challenges bind each group's *committed rows*, not its *claimed
   value*. With `rho_g = 1` the fold enforced the single scalar equation
   `O_S + O_w = s' + v_w'`, and a prover that shifts the two carried claims
   by a compensating `(+delta, -delta)` satisfies it identically. The
   remaining checks do not catch it either: the stage-3 final relation is
   one linear equation in the same two values, so `delta` can be solved for;
   and `derive_tensor_extension_opening_claim_from_partials` weights a
   claim's column partials by an `eq` kernel whose weights sum to `1`, so a
   constant `+/-delta` on the partials moves each derived opening by exactly
   `+/-delta` **at any head**, passing the per-claim checks, and the two
   shifts then cancel in the unit-weighted EOR input claim. This forgery was
   executed end to end against the real verifier and **accepted** at `k = 2`
   (fp64 `D128FullBound18`, `nv = 23`, both setup-prefix levels attacked
   independently) and at `k = 1` (fp128 `D64OneHot`, where no EOR fix-up is
   even needed). See `aerie/_docs/ABSORPTION-AUDIT.md`, Defect 1 and its
   "Verification of Defect 1" appendix.

   **The actual binding argument for the fixed protocol.** The verifier, on
   the recursive suffix with `num_total_polynomials() > 1`:
   (a) absorbs the two claimed values `v_S = s'` and `v_w = v_w'` under
   `ABSORB_EVAL_OPENINGS_FIELD`; (b) squeezes `rho_S, rho_w` under
   `CHALLENGE_EVAL_BATCH`; (c) only then runs the EOR replay (per-claim
   partial checks, `eta`, the shared sum-check) and the fold, both weighted
   by `rho`. The per-claim partial check pins `v_g` to the shipped partials
   of claim `g`; the `rho`-weighted EOR input claim pins the shipped
   partials to the shared sum-check; and the sum-check's output is pinned to
   the *committed* per-group openings by the stage-2 trace equation, whose
   per-claim coefficient is `evaluation_trace_weight * rho_g *
   final_factor`. Writing `e_g = v_g - O_g` for the per-claim error against
   the commitment-determined opening, every one of those checks is `F`-linear
   in the `e_g`, so acceptance forces `sum_g rho_g * e_g = 0`. Because the
   `e_g` are fixed before `rho` is drawn (the values were absorbed at step
   (a)), this is a Schwartz-Zippel event on a nonzero affine form.
   Equivalently: the shared EOR sum-check is run on the honest tables, whose
   true sum is `sum_g rho_g * T(cols_g^true)`, while the verifier
   reconstructs its input claim from the *shipped* partials; a `(+delta,
   -delta)` pair now contributes `(rho_S - rho_w) * T(delta * 1)`, which no
   longer cancels.

   The `k = 1` path (no EOR) reaches the same equation directly through
   `OpeningClaimsLayout::batched_eval_target(rho, openings)`, with the
   absorb-then-squeeze happening after the per-group prepared points go into
   the transcript (prover `compute_trace_target`, verifier
   `prepare_fold_replay`). A single-claim suffix batch keeps `rho_0 = 1`,
   which is lossless (the sum has one term) and byte-identical.

   Belt, independent of the above: `SetupSumcheckProof::setup_prefix_eval`
   is now absorbed at the end of stage 3, next to `next_w_eval`, whenever
   the next level carries a setup-prefix claim (Risk 3 of the audit). It was
   previously absorbed nowhere, which left it entirely unconstrained in the
   degenerate case `setup_scale * setup_index_weight * alpha_val = 0`. Note
   that this belt alone does **not** block the forgery above: the
   compensating pair is determined by challenges that are all fixed before
   the absorb, so binding it changes only downstream challenges the prover
   recomputes for free. Item 4's `rho` draw is the load-bearing fix.
5. **Soundness delta.** Two terms.
   - *Batching.* The per-claim combination above fails with probability at
     most `1/|E|` per suffix batch ([AK] Lemma 5.7, independent-coefficient
     case; the power-ladder variant would give `(Lambda-1)/|E|`, with
     `Lambda = 2` claims here). `|E| = q^2 ~ 2^127` at fp64 Ext2 and
     `|E| = q ~ 2^127` at fp128, so this is `< 2^-126` per offloaded level,
     i.e. the same order as the term already budgeted for the extra EOR
     claim, and absorbed by the existing budget. Before the fix this term
     was `1` (the hypothesis of Lemma 5.7 was violated), so the suffix
     carried no per-claim binding at all.
   - *Extra EOR claim.* As in `_docs/EOR-SETUP-PREFIX.md` Section 3: one
     extra claim in the shared EOR per offloaded level, `< 2^-126` per
     level. The suffix-aligned lifting is an exact algebraic identity
     (item 1) and adds no error term.

## Intent

### Goal

Setup-prefix suffix levels at `k > 1` run the shared EOR with the carried
setup-prefix claim included, producing per-group psi-packed opening points,
so `RecursiveCommitmentConfig<fp64::D128FullBound18>` proves and verifies
end to end at `nv = 23` (the currently `#[ignore]`d round trip).

Surfaces:

- `akita-prover/src/backend/recursive/setup_prefix_source.rs`: implement
  the tensor kernels for the `SetupPrefix` variant (column partials and
  dense packed witness over the RAW `setup_prefix_field_evals`; homogeneous
  span batches; sparse combination returns `None` so setup spans take the
  dense path); the source additionally carries `committed_evals` (the
  psi-transformed stream at `k > 1`) consumed by the fold/decompose kernels,
  and `commit_setup_prefix` commits that representation.
- `akita-prover/src/protocol/core/extension_opening_reduction.rs`: claim
  spans carry a joint offset (`0` for the existing prefix/full alignment);
  suffix-aligned spans evaluate partials at their routed point slice and
  build dense strided terms (`g~(x) = g(x >> pad)` against the full-tail
  factor). The prefix/tiled path is untouched.
- `akita-prover/src/protocol/core/{suffix,fold,fold_kernels}.rs`: drop the
  `setup_prefix.is_none()` conjunct; thread the alignment mode; per-group
  protocol points (and per-group inner claim points) in
  `finish_prepared_fold` / `compute_trace_target` when the suffix EOR ran.
- `akita-verifier/src/protocol/core/{suffix,fold}.rs`: drop the mirrored
  conjunct and the `EOR && num_groups != 1` rejection; `replay_eor_reduction`
  checks each claim at its own routed head; the suffix builds per-group
  packed points from `rho` slices and consumes the final relation
  root-style (`trace_claim_scales`).
- Tests: the Phase-1 equivalence fixture (k=2 vs k=1 opening of the same
  committed flat prefix data), a strided-vs-materialized EOR identity test,
  and the un-ignored fp64 D128 nv=23 round trip.

### Invariants

- **`k = 1` path byte-identical — VOID for two-group suffix levels, by
  design.** The original invariant said all new behaviour sat behind
  `EXT_DEGREE != 1`. That is no longer true and must not be restored: the
  `k = 1` two-group carried-claim suffix was part of the Defect-1 vulnerable
  surface (the forgery was demonstrated there end to end), so it now draws
  batching coefficients from the transcript and its bytes change. What
  remains byte-identical at `k = 1` is every *single-claim* suffix level and
  every level with no setup prefix. Protected by the fp128 D64OneHot
  recursive e2e (`recursive_setup_e2e.rs`), the full `k = 1` suites, and the
  negative test
  `suffix_carried_claim_forgery_fp128_d64_onehot.rs`.
  (`compute_trace_target`'s switch to the group's own routed inner point,
  Defect 2 of the audit, also changes `k = 1` behaviour whenever the two
  suffix groups have unequal arity; that change is a fix, not a regression —
  the verifier has always used the group's own prepared point.)
- **Existing `k > 1` EOR paths byte-identical.** Root grouped and singleton
  suffix EOR keep offset-0 spans and the single-point routing; the new span
  offset parameter is `0` there and the code paths reduce to the current
  ones. Protected by `mixed_bound_multi_group` and all fp32/fp64 suites.
- **Direct mode untouched.** No change to any `SetupContributionMode::Direct`
  path. Protected by the Direct-mode suites.
- **Prover/verifier symmetry.** Transcript event order for the new branch:
  stage-3 outputs (`next_w_eval`, then `setup_prefix_eval` when the next
  level carries one), suffix commitment, then **the two claimed carried
  values, then the batching coefficients `rho`**, then EOR partials (claim
  order: setup group, then witness group), eta, sum-check rounds, then
  per-group padded-point absorbs in `OpeningClaims` group order — mirrored
  exactly on both sides. At `k = 1` there is no EOR, so the
  claim-values-then-`rho` pair moves to *after* the per-group padded-point
  absorbs (prover `finish_prepared_fold` -> `compute_trace_target`, verifier
  `prepare_fold_replay` after its prepared-point loop). Protected by the
  fp64 and fp128 round-trip e2es.
- **Stage-3 presence load-bearing.** A recursive-schedule proof stripped of
  its stage-3 sumcheck is rejected (protected by the fp64 e2e). Upstream
  observation recorded during implementation: the verifier takes the
  per-level setup-contribution mode from the schedule
  (`root_fold.rs:25`, `verifier suffix.rs` underscore the caller argument),
  so the `batched_verify` mode parameter no longer cross-rejects by itself;
  the schedule-mode enforcement is the live invariant.
- **Verifier no-panic boundary.** All new verifier-reachable code returns
  `AkitaError`.
- **Setup-contribution value agreement** (materialized-vs-direct oracle)
  unaffected; stage 3 itself is unchanged.

### Non-Goals

- Dyadic segments; claim accumulation across more than the currently
  eligible offloaded levels; changes to the terminal-eligibility rule
  ("terminal levels never embed a stage-3 proof").
- Lazy/streamed strided terms: the first implementation materializes the
  strided table (size `2^(L - kappa)` extension elements at the suffix,
  bounded by the suffix witness domain, ~16-32 MB at Aerie shapes). A
  closed-form lazy variant (the mirror of `TiledTailFactor`) is a
  performance follow-up, not a correctness need.
- ZK feature coverage (unchanged from the base recursive spec).
- Changing the stage-3 batched geometry or its transcript (suffix-projection
  stays; the alignment is handled inside the EOR).

## Evaluation

### Acceptance Criteria

- [x] Phase-1 equivalence fixtures
      (`setup_prefix_k2_claim_path_matches_k1_direct_mle`,
      `setup_prefix_fold_opening_matches_packed_mle_at_reduction_point` plus
      its degree-one control): the raw-stream claim path, the committed
      transformed representation, and the fold-side opening identity are all
      pinned.
- [x] Suffix-aligned grouped EOR fixture
      (`suffix_aligned_grouped_reduction_reduces_carried_claims`): per-claim
      routed partials open the claims, and `final_claim` equals the
      **rho-weighted** sum of per-group packed openings times the shared
      transparent factor.
- [x] Defect-1 per-claim binding, unit cost
      (`suffix_carried_claim_batching_coefficients_are_transcript_bound`,
      `akita-prover/src/protocol/core/tests.rs`): the two carried claims get
      distinct coefficients; the coefficients move when a claim value moves
      (so they are squeezed after the absorb); and a compensating
      `(+delta, -delta)` cancels under unit weights but not under `rho`.
- [x] Defect-1 forgery rejected end to end, both `k`
      (`akita-pcs/tests/suffix_carried_claim_forgery_fp64_d128.rs` at
      `k = 2`, `nv = 23`; `..._fp128_d64_onehot.rs` at `k = 1`). Both run the
      honest pipeline as a control, then the malicious prover of
      `akita_prover::attack_probe`, and require the verifier to reject.
      `#[ignore]`d and behind the test-only `attack-probe` feature; run
      commands are in each file's module docs. `nv = 23` is the smallest
      shape in the fp64 family whose planner emits a setup-prefix step at
      all (`nv = 22` emits none), so no cheaper e2e minimisation exists; the
      unit fixture above is the CI-cheap companion.
- [x] `recursive_fp64_d128_bound18_nv23_round_trips_and_rejects_stripped_stage3`
      un-ignored and passing (release): prove + serialize round trip +
      recursive verify + stripped-stage-3 rejection + tampered-opening
      rejection. (Renamed from `..._rejects_cross_mode`: the caller-mode
      cross-rejection is vestigial upstream — see Invariants.)
- [x] Regression sweep green: fp128 D64OneHot recursive e2e (release),
      `akita_e2e` (13), `single_poly_e2e` (8), `batched_aggregated_e2e` (5),
      `akita-pcs --lib` (35, incl. the grouped-root EOR probes at k = 2 and
      k = 4), akita-prover (228), akita-verifier, akita-types (289),
      akita-setup, akita-planner, akita-config (envelope trio validated in
      release). `cargo clippy --workspace --all-targets -- -D warnings`
      green.
- [ ] Guest measurement at nv=23 `--setup-mode recursive`: BLOCKED on a
      harness limitation surfaced by this work, split out as the follow-up
      unit. The recursion harness ships a single-group blob
      (`profile/akita-recursion/glue/src/lib.rs`, singleton
      `opening/commitment` fields), but single-group keys plan Direct-only
      (`akita-planner/src/resolve.rs`: scalar recursive keys drop
      `recursive_setup_planning`), and the setup-contribution mode is
      schedule-borne on both sides (prover `fold.rs:799`, verifier
      `root_fold.rs:25` ignore the caller argument), so `--setup-mode
      recursive` on the current harness measures the Direct schedule.
      Required follow-up: extend `AkitaJoltInputs` + guest replay to the
      two-group recursive profile (two dense precommits at `nv/2` plus the
      final group, the same key as the e2e), then run Direct vs Recursive at
      nv=23 and append to
      `profile/akita-recursion/AERIE-FP64-MEASUREMENT.md`.

### Testing Strategy

Unit fixtures live next to the EOR prover tests
(`akita-prover/src/protocol/extension_opening_reduction/sparse/tests.rs`
conventions) and the setup-prefix source. E2e reuses
`crates/akita-pcs/tests/recursive_setup_fp64_d128_e2e.rs`. Everything else
per the previous spec.

### Performance

Adds one EOR claim (partials `k` elements, one strided dense table) per
offloaded suffix level at `k > 1`; no change at `k = 1` or in Direct mode.
The guest measurement (acceptance item 5) is the deliverable.

## Design

### Architecture

See "Mathematical content". Reused machinery (per the reuse mandate):
grouped-span EOR (`eor_claim_spans`, virtual tiling), the two-group suffix
batch (`new_recursive_suffix_with_setup_prefix`,
`recursive_suffix_eor_claims`, `shared_suffix_point`,
`setup_prefix_column_major_point_vars`), per-group point preparation
(`finish_prepared_fold` group loop; grouped-root replay conventions in
`verify_multi_group_root_inner`), the trace-target relation
(`compute_trace_target`, `check_extension_opening_reduction_output`), and
the whole `k = 1` absorption/fold algebra (untouched).

### Alternatives Considered

- Reordering the stage-3 batched rounds so carried points become prefixes
  (would let the root's prefix machinery apply verbatim): rejected — it
  changes the `k = 1` transcript (byte-identity invariant).
- Reversed/permuted joint domains: rejected — they trade one group's
  alignment for a physical permutation of the other group's table.
- Separate non-packed prefix opening: rejected in
  `_docs/EOR-SETUP-PREFIX.md` (candidate 3).

## References

- `_docs/EOR-SETUP-PREFIX.md` (aerie repo; design decision note).
- [AK] Section 3.7 (EOR), 5.1-5.2 (batched openings), lines 2181-2183
  (composition sentence), 6.6 (carried claim).
- `specs/mixed-d-setup-delegation.md`, `specs/batched-stage3-setup-opening.md`,
  `STACK.md` slices 02C/04.
