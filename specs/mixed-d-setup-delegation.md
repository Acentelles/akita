# Spec: Mixed-D Setup Delegation (D128 Recursive Setup Offload)

| Field         | Value                          |
|---------------|--------------------------------|
| Author(s)     | research fork (aerie-mains-substrate) |
| Created       | 2026-07-31                     |
| Status        | active                         |
| PR            | (fork branch, no PR)           |
| Supersedes    |                                |
| Superseded-by |                                |
| Book-chapter  |                                |

## Summary

The recursive setup-contribution path (stage-3 setup-product sumcheck plus
carried setup-prefix opening) is currently hard-gated to ring dimension
`D = 64`, encoded at two independent guards
(`RecursiveCommitmentConfig::runtime_schedule` and
`derive_setup_prefix_group`) plus several sites that use the constant
`SETUP_OFFLOAD_D_SETUP = 64` where they mean "the level's ring dimension".
This is the STACK.md slice-02B first-implementation restriction ("if the
proof level has `D != D_setup`, delegation is rejected"). The paper's final
design (Akita Section 6.6, "Preprocessed prefix commitments") removes the
dimension gate: the committed object is the flat coefficient vector
`S^flat`, every role reads it at its own ring dimension through the weight
side, and any level whose dimensions divide the generation dimension
delegates through the same commitments. This change lifts the substrate to
that design for the uniform-dimension case, enabling recursive setup
offload for the fp64 `D128FullBound18` / `D128FullBound6` presets (Aerie's
shapes) at `max_num_vars = 23` / `26`.

## Phase-0 findings this spec encodes

- The setup generation dimension is `gen_ring_dim =
  policy_of::<Cfg>().ring_dimension = Cfg::D` (`akita-setup/src/lib.rs`,
  `setup_gen_ring_dim`); for the fp64 D128 presets `d_gen = 128`. The
  paper's divisibility condition (level dimension divides `d_gen`) holds
  trivially: the planner emits uniform per-level ring dims
  (`CommitmentRingDims::uniform(policy.ring_dimension)`), so every level
  reads at `128 | 128`.
- The flat committed object is split-invariant: for a flat prefix of
  length `n_prefix`, the weight vector satisfies
  `w(lambda_128 * 128 + y) = idx_w_128(lambda_128) * alpha^y` and equally
  `w(lambda_64 * 64 + y') = idx_w_64(lambda_64) * alpha^{y'}` with
  `idx_w_64(2*lambda + lane) = idx_w_128(lambda) * alpha^{64*lane}`; the
  MLE claim `s_rho = S~^flat(rho)` over `log2(n_prefix)` variables does not
  depend on the split. The succinct weight evaluator
  (`SetupIndexWeightEvaluator`) already implements the lane projection for
  role dims that are multiples of a base setup dimension.
- What is *not* dimension-free in the substrate is the carried-opening
  absorption: the setup-prefix commitment is opened by the next fold as a
  precommitted group, and the fold's group batch is uniform in ring
  dimension per level (no per-group ring dims in the fold algebra). A
  `d_setup = 64` commitment therefore cannot be absorbed by a `D = 128`
  fold. Consequently this change instantiates the paper design with the
  prefix committed at the generation dimension: `d_setup := gen_ring_dim
  (= Cfg::D)` per config. For uniform-`D` schedules this is exactly the
  paper's object (same flat vector, same claims); the commitment chunking
  dimension is a commitment-internal detail bound into the slot identity.

## Intent

### Goal

Generalize the recursive setup-offload path from "level ring dimension is
the constant 64" to "level ring dimension equals the setup generation
dimension, and that dimension is in the supported offload set {64, 128}",
with the slot commitment dimension `d_setup` equal to the generation
dimension, so `RecursiveCommitmentConfig<fp64::D128FullBound18>` (and
`Bound6`) plan, prove, and verify recursive setup offload.

Surfaces touched:

- `akita-types::proof::setup_prefix`: new
  `SETUP_OFFLOAD_SUPPORTED_RING_DIMS` / `setup_offload_ring_dim_supported`
  predicate; `select_setup_prefix_slot` additionally cross-checks the
  planned slot's `d_setup` against the runtime dimension (fail-closed).
- `akita-config::recursive_commitment` guard: supported-dimension predicate
  instead of `Cfg::D != 64`.
- `akita-planner::schedule_params::{candidate, suffix_dp}` and
  `group_batch`: `derive_setup_prefix_group` guard generalized; slot ids
  stamped with `policy.ring_dimension`; every `active_setup_field_len`
  natural-length computation uses the level's ring dimension.
- `akita-config::setup_prefix_slots`: slot/schedule cross-checks use the
  fold's `ring_dimension`.
- `akita-config::generated_families`: recursive capacity candidates for
  catalog-less recursive configs; explicit enablement of the two fp64 D128
  presets via a profile key (final `(nv, 2)` plus two singleton precommits
  at `(nv/2, 1)`), mirroring the existing fp128 D64OneHot profile key.
- `akita-setup::recursive_prefixes`: commit prefix slots at
  `id.d_setup` (runtime dispatch) and require `id.d_setup == gen_ring_dim`;
  generation-dimension guard generalized to the supported set.
- `akita-prover` stage-3 / `akita-verifier` stage-3: the prefix-offload
  branch gates on `ring_d == gen_ring_dim` instead of `ring_d == 64`.
- Diagnostic fix (upstream bug, in passing):
  `proof_optimized_max_setup_matrix_size_uncached` reports the first
  per-shape schedule error when no shape is feasible, instead of the
  masking "no generated schedules" message.

### Invariants

- **Existing D=64 OneHot recursive behavior unchanged.** For
  `RecursiveCommitmentConfig<fp128::D64OneHot>` every generalized
  condition evaluates exactly as the old constant gate (`gen_ring_dim = 64
  = Cfg::D`), slot ids keep `d_setup = 64`, and candidate enumeration for
  catalog-backed configs is untouched. Protected by
  `crates/akita-pcs/tests/recursive_setup_e2e.rs` and the
  `setup_prefix_slots` unit tests.
- **Prover/verifier symmetry.** Prover and verifier derive the natural
  prefix length, padded domain, and slot identity from the same level
  parameters and the same ring dimension; the slot id (including
  `d_setup`) is transcript-bound (`ABSORB_SETUP_PREFIX_SLOT`). Protected
  by the new fp64 D128 e2e round-trip and the existing fp128 one.
- **Mode cross-rejection preserved.** A `Recursive` proof still fails
  under `Direct` and vice versa (stage-3 presence is load-bearing).
  Protected by the existing cross-mode test and a new fp64 D128
  cross-mode test.
- **Verifier no-panic boundary.** All new verifier-reachable checks return
  `AkitaError` (unsupported dimension, slot/dimension mismatch, missing
  slot) rather than panicking.
- **Setup-contribution value agreement at D=128.** The materialized
  `<S, omega_S>` inner product equals the direct packed scan at ring
  dimension 128, exactly (the mandatory equivalence oracle). Protected by
  the new `RING_DIM = 128` fixtures in
  `crates/akita-verifier/src/protocol/slice_mle/setup_contribution/`.
- **Guards are generalized, not deleted.** Delegation still requires: a
  recursive config, a supported offload dimension, single-chunk witness,
  level dimension equal to the generation dimension, and an exact
  registry slot; every failure is an explicit error.

### Non-Goals

- **Profile B / carried-opening batching changes.** The absorption
  mechanics (how the carried claim joins the next fold's opening batch)
  are unchanged; only the dimension gate is lifted.
- **Heterogeneous-dimension delegation** (level dimension a strict
  divisor/multiple of the commitment dimension, e.g. a D=64 level against
  a D=128-committed prefix). The flat object supports it in principle
  (lane projection), but the fold's group batch is uniform-D; this stays
  future work and is explicitly rejected by the equality condition.
- **New generated schedule tables.** The fp64 D128 recursive family
  resolves table-less through the planner DP (catalog `None`), like the
  scalar recursive fallback. Tables are added only if runtime cost demands
  it (it does not, at nv <= 26).
- **ZK feature coverage** (unchanged from the base recursive spec).
- **Changing `SETUP_OFFLOAD_MIN_PREFIX_FIELD_LEN`** or the eligibility
  levels (root and first recursive level).

## Evaluation

### Acceptance Criteria

- [x] `RecursiveCommitmentConfig<fp64::D128FullBound18>` plans a recursive
      schedule (with setup-prefix metadata) for the profile key at
      `max_num_vars = 23`; the setup materializes and commits the `d_setup
      = 128` prefix slots, and root-level recursive proving succeeds.
- [ ] Full prove + verify round trip at `nv = 23`: BLOCKED by a
      pre-existing, previously undocumented restriction orthogonal to the
      dimension gate — see "Discovered blocker" below. The round-trip test
      exists (`recursive_setup_fp64_d128_e2e.rs`) and is `#[ignore]`d with
      the exact reason.
- [x] `fp64::D128FullBound6` plans a recursive schedule with setup-prefix
      metadata at `max_num_vars = 26`.
- [x] Materialized-vs-direct setup-contribution equivalence fixtures pass
      at `RING_DIM = 128` (and continue passing at 64) — the mandatory
      mixed-D equivalence oracle.
- [x] All existing tests stay green, in particular
      `recursive_setup_e2e.rs` (fp128 D64OneHot) and the Direct-mode
      suites.
- [x] `cargo clippy --workspace --all-targets -- -D warnings` (after three
      semantics-free baseline lint fixes; HEAD itself was not clippy-clean
      under the pinned 1.95 toolchain). `cargo fmt` is NOT run
      workspace-wide: HEAD is not fmt-clean under rustfmt 1.9.0 (pinned
      toolchain), so formatting is normalized only in touched files.

### Discovered blocker (stop-and-report)

The fp64 round trip fails at suffix level 1 with
`InvalidInput("inactive extension inner coordinates must be zero after psi
packing")`. Root cause: the recursive suffix disables the extension-opening
reduction whenever the level carries a setup prefix —

```text
needs_extension_reduction = EXT_DEGREE != 1 && level_params.setup_prefix.is_none()
```

(`akita-prover/src/protocol/core/suffix.rs`, mirrored at
`akita-verifier/src/protocol/core/suffix.rs`). The carried stage-3 setup
point `(rho_y, rho_setup_idx)` consists of free sumcheck challenges; at
extension degree 2 the ring-subfield packed opening path requires the
coordinates in `[log2(D/e), log2(D))` to be zero, which only the
extension-opening reduction arranges for the witness side. The composition
of the carried setup-prefix opening with the extension-opening reduction is
defined nowhere (not in `specs/batched-stage3-setup-opening.md`, which is
extension-degree-silent, not in `specs/extension-field-opening-batching.md`,
and not in the paper's Section 6.6, which is dimension-focused). Candidate
designs, none of which is implemented here because each changes protocol
semantics: (a) run the extension-opening reduction jointly over the
two-group suffix batch including the prefix group; (b) make the stage-3
setup table ring-subfield packed (y-axis = `log2(D/e)` trace bits with the
extension slot handled by the psi embedding, changing the stage-3 closure
relation); (c) open the prefix through a separate non-packed path. This is
the same class of work as Profile B absorption and must be specified before
implementation. Consequence: recursive setup offload currently requires a
degree-1 claim field (fp128) regardless of ring dimension; the fp64 presets
gain planning, preprocessing, and root-level proving from this change, and
the remaining gap is extension-field carried-opening ingestion, not the
dimension gate.

### Testing Strategy

- New: fp64 D128 recursive e2e (modeled on `recursive_setup_e2e.rs` with
  the dense-poly machinery of `mixed_bound_multi_group.rs`): profile key
  final `(23, 2)` plus two `(11, 1)` dense precommits, bound-18 data,
  prove + serialize round-trip + verify + cross-mode rejection.
- New: `RING_DIM = 128` variants of the materialized-vs-direct and
  setup-index-weight-MLE fixtures (the equivalence oracle).
- Existing: full `cargo test`, no exceptions.

### Performance

No native-performance claim; this enables shapes previously rejected. The
recursive mode remains more expensive than Direct on a native verifier
(Remark 6.13 of the paper); its value is in-circuit. Schedule resolution
for the new family is pure-DP; the planner memoization keeps setup-time
schedule search acceptable at nv <= 26.

## Design

### Architecture

The generalized condition, stated once:

> Setup delegation at a fold level is allowed iff the config is
> recursive, single-chunk, `setup_offload_ring_dim_supported(d)` holds for
> `d = policy.ring_dimension`, and (at runtime) the level's inner ring
> dimension equals the setup generation dimension `gen_ring_dim`. The slot
> commitment dimension is `d_setup = gen_ring_dim`.

Since the planner emits uniform ring dims equal to `Cfg::D` and
`gen_ring_dim = Cfg::D`, the runtime equality is invariant for all
shipped presets; it is checked anyway (fail-closed) at slot commit, slot
match, and the stage-3 offload branches. All natural-length arithmetic
(`active_setup_field_len`, `stage3_offload_natural_field_len`) uses the
level ring dimension, which the D64 path already did implicitly via the
constant.

### Alternatives Considered

- **Commit the prefix at a fixed `d_setup = 64` and read at 128 through
  lanes** (the literal slice-02B constants). Rejected for this change: the
  stage-3 sumcheck itself is split-invariant, but the carried opening must
  be absorbed by the next fold's uniform-D group batch; supporting a
  64-committed group inside a 128 fold requires per-group ring dimensions
  in the fold algebra (a much larger change, and unnecessary for uniform
  presets). The lane projection in the weight evaluator is kept; it is the
  path to true mixed-D later.
- **Generated tables for the new family.** Rejected: the D64 recursive
  family works table-less per the catalog design; DP regen suffices.

## Documentation

This spec. `STACK.md` slice-02B's "initial constants" paragraph is
superseded by the generalized condition above (left in place upstream;
this is a fork).

## References

- Akita paper Section 6.6 (`raw/lattices/2026-akita-lattice-polynomial-
  commitment-scheme`, "Preprocessed prefix commitments").
- `specs/setup-product-sumcheck.md`, `specs/setup-layout-repack.md`,
  `specs/setup-prefix-ladder.md`, `STACK.md` slices 02B/03B/04.
- `_docs/NATIVE-RECURSION-DESIGN.md` (aerie repo) Section 2 measured
  update, 2026-07-31.
