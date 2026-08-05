//! Generate an Akita verifier-input blob to be consumed by the Jolt guest
//! program in `profile/akita-recursion/guest`.
//!
//! fp64 port of the original fp128 D64OneHot artifact under an
//! Aerie-representative fp64 preset (`fp64::D128FullBound18` by default,
//! `fp64::D128FullBound6` with `--preset bound6`). Two setup-contribution
//! modes are supported:
//!
//! - `--setup-mode direct` (default): a single-poly *dense* commitment opened
//!   at one random extension point; the original single-group proxy.
//! - `--setup-mode recursive`: the recursive setup-offload profile key (two
//!   singleton dense precommits at `nv/2` plus a two-poly final group at
//!   `nv`), proved under `RecursiveCommitmentConfig` so the schedule carries
//!   the stage-3 setup-product sumcheck and the carried setup-prefix opening.
//!
//! **Synthetic proxies, not Aerie/Falcon results.** The witness rows are
//! relation-free random values inside the preset's declared source bound;
//! only the verifier *schedule* (fold levels, sum-check rounds, setup scan or
//! delegation shape) is representative of Aerie's Akita opening, not the
//! statement.
//!
//! Recursive blobs can additionally truncate the shipped setup matrix
//! (`--setup-transport slice|seed`): the delegated root-level scan is
//! replaced by the stage-3 sumcheck plus the public setup-prefix commitment
//! `C_S`, so the verifier only reads the sub-gate/terminal prefix of the
//! shared matrix. The minimal prefix is found empirically by binary search
//! over native verification and re-checked before publishing.
//!
//! After running the prover end-to-end we re-run the host verifier as a
//! sanity check, then serialize all verifier-side state into one contiguous
//! blob via [`akita_recursion_glue::AkitaJoltInputs`].
//!
//! Output paths are controlled via `AKITA_RECURSION_BLOB` (defaults to
//! `target/akita_recursion_inputs.bin`). Set `AKITA_NUM_VARS` (default 23,
//! Aerie's main-group arity) to regenerate at a different polynomial arity.

#![allow(missing_docs)]

use akita_config::proof_optimized::fp64;
use akita_config::{CommitmentConfig, ConservativeCommitmentConfig, RecursiveCommitmentConfig};
use akita_field::{CanonicalField, ExtField, LiftBase, PseudoMersenneField};
use akita_pcs::AkitaCommitmentScheme;
use akita_prover::{
    ComputeBackendSetup, CpuBackend, DensePoly, ProverOpeningData, UniformProverStack,
};
use akita_recursion_glue::{AkitaJoltGroup, AkitaJoltInputs, SetupTransport};
use akita_transcript::AkitaTranscript;
use akita_types::{
    lagrange_weights, AkitaVerifierSetup, BasisMode, Commitment, FlatMatrix, OpeningClaims,
    PointVariableSelection, PolynomialGroupClaims, PolynomialGroupLayout, SetupContributionMode,
};
use clap::{Parser, ValueEnum};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::env;
use std::fs;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;
use tracing_subscriber::EnvFilter;

/// Setup-contribution mode the proof is generated under.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum SetupModeArg {
    /// Evaluate the setup contribution directly from the expanded matrix.
    Direct,
    /// Embed the recursive setup-product sumcheck (multi-group profile key).
    Recursive,
}

impl SetupModeArg {
    fn into_mode(self) -> SetupContributionMode {
        match self {
            SetupModeArg::Direct => SetupContributionMode::Direct,
            SetupModeArg::Recursive => SetupContributionMode::Recursive,
        }
    }
}

/// How the shared setup matrix travels inside the blob.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum SetupTransportArg {
    /// Ship the full expanded matrix (required for direct mode).
    Full,
    /// Ship only the verifier-read prefix slice (recursive mode only).
    Slice,
    /// Ship no matrix; the guest re-derives the prefix from the 32-byte
    /// public seed (recursive mode only).
    Seed,
}

impl SetupTransportArg {
    fn into_transport(self) -> SetupTransport {
        match self {
            SetupTransportArg::Full => SetupTransport::ExpandedMatrix,
            SetupTransportArg::Slice => SetupTransport::MatrixSlice,
            SetupTransportArg::Seed => SetupTransport::SeedDerived,
        }
    }
}

/// fp64 preset the artifact is generated under. Must match the guest
/// monomorphization that will consume the blob.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum PresetArg {
    /// `fp64::D128FullBound18` (Aerie main-group shape, values in `[0, 2^17)`).
    Bound18,
    /// `fp64::D128FullBound6` (Aerie range-group shape, values in `[0, 2^5)`).
    Bound6,
}

#[derive(Debug, Parser)]
#[command(
    about = "Generate an Akita fp64 verifier-input blob for the Jolt recursion guest",
    long_about = None
)]
struct Args {
    /// Setup-contribution mode the proof is generated under. The blob records
    /// this so host preflight and guest replay verify under the same mode.
    #[arg(long, value_enum, default_value_t = SetupModeArg::Direct)]
    setup_mode: SetupModeArg,

    /// Setup-matrix transports (full matrix, verifier-read slice, and/or
    /// seed-derived), comma-separated. One blob is published per transport
    /// from a single prover run; with several transports the output path
    /// gains a `.slice` / `.seed` suffix for the truncated variants.
    /// Truncated transports require `--setup-mode recursive`.
    #[arg(
        long,
        value_enum,
        num_args = 1..,
        value_delimiter = ',',
        default_value = "full"
    )]
    setup_transport: Vec<SetupTransportArg>,

    /// fp64 preset (bound18 = Aerie main group, bound6 = Aerie range group).
    #[arg(long, value_enum, default_value_t = PresetArg::Bound18)]
    preset: PresetArg,
}

type F = fp64::Field;
type E = fp64::ExtensionField;
const D: usize = 128;

/// Recursive profile-key shape (mirrors
/// `crates/akita-pcs/tests/recursive_setup_fp64_d128_e2e.rs`).
const PRE_GROUPS: usize = 2;
const FINAL_GROUP_SIZE: usize = 2;
const TOTAL_GROUP_SIZE: usize = PRE_GROUPS + FINAL_GROUP_SIZE;
const PROVER_STACK_BYTES: usize = 256 * 1024 * 1024;

const TRANSCRIPT_DOMAIN_BOUND18: &[u8] = b"akita-recursion/fp64-d128-bound18";
const TRANSCRIPT_DOMAIN_BOUND6: &[u8] = b"akita-recursion/fp64-d128-bound6";

/// Random extension opening point compatible with ring-subfield packing:
/// inactive inner coordinates in `[log2(D/ext_degree), log2(D))` are zeroed
/// (mirrors `subfield_random_point` in akita-pcs's mixed-bound tests).
fn subfield_random_point(num_vars: usize, seed: u64) -> Vec<E> {
    let ext_degree = <E as ExtField<F>>::EXT_DEGREE;
    let trace_inner = (D / ext_degree).trailing_zeros() as usize;
    let alpha_bits = D.trailing_zeros() as usize;
    let mut rng = StdRng::seed_from_u64(seed);
    let mut point: Vec<E> = (0..num_vars)
        .map(|_| {
            let limbs: Vec<F> = (0..ext_degree)
                .map(|_| F::from_canonical_u128_reduced(rng.r#gen::<u128>()))
                .collect();
            E::from_base_slice(&limbs)
        })
        .collect();
    for idx in trace_inner..alpha_bits {
        if idx < point.len() {
            point[idx] = E::zero();
        }
    }
    point
}

/// Extension Lagrange opening of base-field evaluations at an extension point.
fn dense_opening_ext(evals: &[F], point: &[E]) -> Result<E, String> {
    let weights =
        lagrange_weights(point).map_err(|err| format!("dense ext lagrange weights: {err}"))?;
    if weights.len() != evals.len() {
        return Err(format!(
            "lagrange weight count {} does not match eval count {}",
            weights.len(),
            evals.len()
        ));
    }
    Ok(evals
        .iter()
        .zip(weights.iter())
        .fold(E::zero(), |acc, (&coeff, &weight)| {
            acc + weight * E::lift_base(coeff)
        }))
}

fn make_seeded_dense_poly_bounded(
    num_vars: usize,
    max_exclusive: u64,
    seed: u64,
) -> Result<(DensePoly<F>, Vec<F>), String> {
    let len = 1usize << num_vars;
    let mut rng = StdRng::seed_from_u64(seed);
    let evals: Vec<F> = (0..len)
        .map(|_| F::from_u64(rng.gen_range(0..max_exclusive)))
        .collect();
    let poly = DensePoly::<F>::from_field_evals(num_vars, D, &evals)
        .map_err(|err| format!("dense poly construction: {err}"))?;
    Ok((poly, evals))
}

fn fp64_prime_label() -> String {
    format!(
        "q=2^64-{}",
        <F as PseudoMersenneField>::MODULUS_OFFSET
    )
}

fn env_usize(name: &str, default: usize) -> Result<usize, String> {
    match env::var(name) {
        Ok(value) => match value.parse() {
            Ok(parsed) => Ok(parsed),
            Err(err) => Err(format!(
                "{name} must be a non-negative integer, got `{value}`: {err}"
            )),
        },
        Err(env::VarError::NotPresent) => Ok(default),
        Err(env::VarError::NotUnicode(value)) => Err(format!(
            "{name} must be valid Unicode, got `{}`",
            value.to_string_lossy()
        )),
    }
}

fn env_string(name: &str, default: &str) -> Result<String, String> {
    match env::var(name) {
        Ok(value) => Ok(value),
        Err(env::VarError::NotPresent) => Ok(default.to_string()),
        Err(env::VarError::NotUnicode(value)) => Err(format!(
            "{name} must be valid Unicode, got `{}`",
            value.to_string_lossy()
        )),
    }
}

fn publish_blob(output_path: &std::path::Path, blob: &[u8]) -> Result<(), String> {
    if let Some(parent) = output_path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        fs::create_dir_all(parent).map_err(|err| {
            format!(
                "failed to create output directory `{}`: {err}",
                parent.display()
            )
        })?;
    }
    let mut tmp_name = output_path
        .file_name()
        .map(|name| name.to_os_string())
        .unwrap_or_else(|| "akita_recursion_inputs.bin".into());
    tmp_name.push(".tmp");
    let tmp_path = output_path.with_file_name(tmp_name);
    fs::write(&tmp_path, blob)
        .map_err(|err| format!("failed to write temp blob `{}`: {err}", tmp_path.display()))?;
    fs::rename(&tmp_path, output_path).map_err(|err| {
        let _ = fs::remove_file(&tmp_path);
        format!(
            "failed to publish blob `{}` from `{}`: {err}",
            output_path.display(),
            tmp_path.display()
        )
    })
}

/// One proven instance, independent of how it was produced.
struct ProvenArtifact {
    point: Vec<E>,
    groups: Vec<AkitaJoltGroup<F, E>>,
    verifier_setup: AkitaVerifierSetup<F>,
    proof: akita_types::AkitaBatchedProof<F, E>,
}

fn verifier_claims_from_groups<'a>(
    point: &[E],
    groups: &'a [AkitaJoltGroup<F, E>],
) -> Result<OpeningClaims<'static, E, &'a Commitment<F>>, String> {
    let mut claim_groups = Vec::with_capacity(groups.len());
    for (idx, group) in groups.iter().enumerate() {
        let point_num_vars = usize::try_from(group.point_num_vars)
            .map_err(|_| format!("group {idx} arity overflow"))?;
        claim_groups.push(
            PolynomialGroupClaims::new(
                PointVariableSelection::prefix(point_num_vars, point.len())
                    .map_err(|err| format!("group {idx} point selection: {err}"))?,
                group.openings.clone(),
                &group.commitment,
            )
            .map_err(|err| format!("group {idx} claims: {err}"))?,
        );
    }
    OpeningClaims::from_groups(point.to_vec(), claim_groups)
        .map_err(|err| format!("verifier opening batch: {err}"))
}

fn verify_artifact_with_mode<Cfg>(
    artifact: &ProvenArtifact,
    verifier_setup: &AkitaVerifierSetup<F>,
    transcript_domain: &[u8],
    setup_contribution_mode: SetupContributionMode,
) -> Result<(), String>
where
    Cfg: CommitmentConfig<Field = F, ExtField = E>,
    RecursiveCommitmentConfig<Cfg>: CommitmentConfig<Field = F, ExtField = E>,
{
    let claims = verifier_claims_from_groups(&artifact.point, &artifact.groups)?;
    let mut transcript = AkitaTranscript::<F>::unbound_verifier(transcript_domain);
    match setup_contribution_mode {
        SetupContributionMode::Direct => akita_verifier::batched_verify::<Cfg, _>(
            &artifact.proof,
            verifier_setup,
            &mut transcript,
            claims,
            BasisMode::Lagrange,
            setup_contribution_mode,
        ),
        SetupContributionMode::Recursive => {
            akita_verifier::batched_verify::<RecursiveCommitmentConfig<Cfg>, _>(
                &artifact.proof,
                verifier_setup,
                &mut transcript,
                claims,
                BasisMode::Lagrange,
                setup_contribution_mode,
            )
        }
    }
    .map_err(|err| format!("{setup_contribution_mode:?}-mode verifier rejected proof: {err}"))
}

/// Direct mode: the original single-poly, single-group dense proxy.
fn prove_direct<Cfg>(nv: usize, bound_max_exclusive: u64) -> Result<ProvenArtifact, String>
where
    Cfg: CommitmentConfig<Field = F, ExtField = E>,
{
    // Deterministic seeds for reproducibility (same as the original harness).
    let (poly, evals) = make_seeded_dense_poly_bounded(nv, bound_max_exclusive, 0xbeef_cafe)?;
    let opening_point = subfield_random_point(nv, 0xfeed_f00d);

    let t0 = Instant::now();
    let opening = dense_opening_ext(&evals, &opening_point)?;
    tracing::info!(
        elapsed_s = t0.elapsed().as_secs_f64(),
        "reference opening computed"
    );
    drop(evals);

    let t0 = Instant::now();
    let prover_setup = AkitaCommitmentScheme::<Cfg>::setup_prover(nv, 1)
        .map_err(|err| format!("prover setup failed: {err}"))?;
    let prepared = CpuBackend
        .prepare_setup(&prover_setup)
        .map_err(|err| format!("backend setup preparation failed: {err}"))?;
    let stack = UniformProverStack::uniform(&CpuBackend, &prepared, prover_setup.expanded.as_ref())
        .map_err(|err| format!("prover stack validation failed: {err}"))?;
    tracing::info!(
        elapsed_s = t0.elapsed().as_secs_f64(),
        "prover setup complete"
    );

    let t0 = Instant::now();
    let (commitment, hint) =
        AkitaCommitmentScheme::<Cfg>::commit(&prover_setup, std::slice::from_ref(&poly), &stack)
            .map_err(|err| format!("commit failed: {err}"))?;
    tracing::info!(elapsed_s = t0.elapsed().as_secs_f64(), "commit complete");

    let poly_refs: [&DensePoly<F>; 1] = [&poly];
    let openings = [opening];

    let t0 = Instant::now();
    let mut prover_transcript = AkitaTranscript::<F>::new(transcript_domain_for::<Cfg>()?);
    let prove_group = PolynomialGroupClaims::new(
        PointVariableSelection::prefix(opening_point.len(), opening_point.len())
            .map_err(|err| format!("invalid opening point shape: {err}"))?,
        openings.to_vec(),
        commitment.clone(),
    )
    .map_err(|err| format!("invalid prover opening group: {err}"))?;
    let prove_input = ProverOpeningData::new(
        OpeningClaims::from_groups(opening_point.clone(), vec![prove_group])
            .map_err(|err| format!("invalid prover opening claims: {err}"))?,
        vec![hint],
        vec![&poly_refs[..]],
    )
    .map_err(|err| format!("invalid prover opening data: {err}"))?;
    let proof = AkitaCommitmentScheme::<Cfg>::batched_prove(
        &prover_setup,
        prove_input,
        &stack,
        &mut prover_transcript,
        BasisMode::Lagrange,
        SetupContributionMode::Direct,
    )
    .map_err(|err| format!("batched_prove failed: {err}"))?;
    tracing::info!(elapsed_s = t0.elapsed().as_secs_f64(), "prove complete");

    let verifier_setup = AkitaCommitmentScheme::<Cfg>::setup_verifier(&prover_setup);
    Ok(ProvenArtifact {
        point: opening_point,
        groups: vec![AkitaJoltGroup {
            point_num_vars: nv as u64,
            openings: openings.to_vec(),
            commitment,
        }],
        verifier_setup,
        proof,
    })
}

/// Recursive mode: the multi-group recursive setup-offload profile key
/// (two singleton dense precommits at `nv/2` plus a two-poly final group),
/// mirroring `recursive_setup_fp64_d128_e2e.rs`.
fn prove_recursive<Cfg>(nv: usize, bound_max_exclusive: u64) -> Result<ProvenArtifact, String>
where
    Cfg: CommitmentConfig<Field = F, ExtField = E>,
    RecursiveCommitmentConfig<Cfg>: CommitmentConfig<Field = F, ExtField = E>,
    ConservativeCommitmentConfig<Cfg>: CommitmentConfig<Field = F, ExtField = E>,
{
    type RecursiveScheme<Cfg> = AkitaCommitmentScheme<RecursiveCommitmentConfig<Cfg>>;
    type ConservativeScheme<Cfg> = AkitaCommitmentScheme<ConservativeCommitmentConfig<Cfg>>;

    let pre_nv = nv / 2;

    let t0 = Instant::now();
    let setup = RecursiveScheme::<Cfg>::setup_prover(nv, TOTAL_GROUP_SIZE)
        .map_err(|err| format!("recursive prover setup failed: {err}"))?;
    if setup.prefix_slots.is_empty() {
        return Err("recursive setup did not precompute setup-prefix slots".to_string());
    }
    let prepared = CpuBackend
        .prepare_setup(&setup)
        .map_err(|err| format!("backend setup preparation failed: {err}"))?;
    let stack = UniformProverStack::uniform(&CpuBackend, &prepared, setup.expanded.as_ref())
        .map_err(|err| format!("prover stack validation failed: {err}"))?;
    tracing::info!(
        elapsed_s = t0.elapsed().as_secs_f64(),
        "recursive prover setup complete"
    );

    let point = subfield_random_point(nv, 0xfeed_f00d);

    // Precommitted singleton dense groups under the conservative adapter.
    let t0 = Instant::now();
    let mut groups: Vec<AkitaJoltGroup<F, E>> = Vec::with_capacity(PRE_GROUPS + 1);
    let mut pre_polys_by_group = Vec::with_capacity(PRE_GROUPS);
    let mut pre_hints = Vec::with_capacity(PRE_GROUPS);
    for group_idx in 0..PRE_GROUPS {
        let (poly, evals) = make_seeded_dense_poly_bounded(
            pre_nv,
            bound_max_exclusive,
            0xbeef_1000 + group_idx as u64,
        )?;
        let polys = vec![poly];
        let (commitment, hint) = ConservativeScheme::<Cfg>::batched_commit(&setup, &polys[..], &stack)
            .map_err(|err| format!("precommit {group_idx} failed: {err}"))?;
        let opening = dense_opening_ext(&evals, &point[..pre_nv])?;
        groups.push(AkitaJoltGroup {
            point_num_vars: pre_nv as u64,
            openings: vec![opening],
            commitment,
        });
        pre_polys_by_group.push(polys);
        pre_hints.push(hint);
    }
    tracing::info!(
        elapsed_s = t0.elapsed().as_secs_f64(),
        "precommits complete"
    );

    // Final two-poly dense group.
    let t0 = Instant::now();
    let mut final_polys = Vec::with_capacity(FINAL_GROUP_SIZE);
    let mut final_openings = Vec::with_capacity(FINAL_GROUP_SIZE);
    for poly_idx in 0..FINAL_GROUP_SIZE {
        let (poly, evals) =
            make_seeded_dense_poly_bounded(nv, bound_max_exclusive, 0xbeef_2000 + poly_idx as u64)?;
        final_openings.push(dense_opening_ext(&evals, &point)?);
        final_polys.push(poly);
    }
    let pre_keys = vec![PolynomialGroupLayout::new(pre_nv, 1); PRE_GROUPS];
    let (final_commitment, final_hint) =
        RecursiveScheme::<Cfg>::commit_final_group(&setup, &final_polys, &stack, pre_keys)
            .map_err(|err| format!("final recursive commitment failed: {err}"))?;
    groups.push(AkitaJoltGroup {
        point_num_vars: nv as u64,
        openings: final_openings.clone(),
        commitment: final_commitment.clone(),
    });
    tracing::info!(
        elapsed_s = t0.elapsed().as_secs_f64(),
        "final-group commit complete"
    );

    let pre_refs_by_group: Vec<Vec<&DensePoly<F>>> = pre_polys_by_group
        .iter()
        .map(|polys| polys.iter().collect())
        .collect();
    let final_refs: Vec<&DensePoly<F>> = final_polys.iter().collect();

    let mut prover_groups = Vec::with_capacity(groups.len());
    for group in &groups {
        prover_groups.push(
            PolynomialGroupClaims::new(
                PointVariableSelection::prefix(group.point_num_vars as usize, nv)
                    .map_err(|err| format!("prover point selection: {err}"))?,
                group.openings.clone(),
                group.commitment.clone(),
            )
            .map_err(|err| format!("prover group claims: {err}"))?,
        );
    }
    let mut prover_polys: Vec<&[&DensePoly<F>]> = Vec::with_capacity(groups.len());
    for refs in &pre_refs_by_group {
        prover_polys.push(&refs[..]);
    }
    prover_polys.push(&final_refs[..]);
    let mut prover_hints = pre_hints;
    prover_hints.push(final_hint);

    let prover_claims = ProverOpeningData::new(
        OpeningClaims::from_groups(point.clone(), prover_groups)
            .map_err(|err| format!("prover opening claims: {err}"))?,
        prover_hints,
        prover_polys,
    )
    .map_err(|err| format!("prover opening data: {err}"))?;

    let t0 = Instant::now();
    let mut prover_transcript = AkitaTranscript::<F>::new(transcript_domain_for::<Cfg>()?);
    let proof = RecursiveScheme::<Cfg>::batched_prove(
        &setup,
        prover_claims,
        &stack,
        &mut prover_transcript,
        BasisMode::Lagrange,
        SetupContributionMode::Recursive,
    )
    .map_err(|err| format!("recursive batched_prove failed: {err}"))?;
    tracing::info!(
        elapsed_s = t0.elapsed().as_secs_f64(),
        "recursive prove complete"
    );

    let verifier_setup = setup
        .verifier_setup()
        .map_err(|err| format!("verifier setup derivation failed: {err}"))?;
    if verifier_setup.prefix_slots.is_empty() {
        return Err("recursive verifier setup carries no setup-prefix commitments".to_string());
    }
    Ok(ProvenArtifact {
        point,
        groups,
        verifier_setup,
        proof,
    })
}

fn transcript_domain_for<Cfg: CommitmentConfig>() -> Result<&'static [u8], String> {
    use std::any::TypeId;
    let id = TypeId::of::<Cfg>();
    if id == TypeId::of::<fp64::D128FullBound18>() {
        Ok(TRANSCRIPT_DOMAIN_BOUND18)
    } else if id == TypeId::of::<fp64::D128FullBound6>() {
        Ok(TRANSCRIPT_DOMAIN_BOUND6)
    } else {
        Err("unknown preset config for transcript domain".to_string())
    }
}

fn truncated_verifier_setup(
    verifier_setup: &AkitaVerifierSetup<F>,
    ring_len: usize,
) -> Result<AkitaVerifierSetup<F>, String> {
    let expanded = verifier_setup.expanded.as_ref();
    let shared = expanded.shared_matrix();
    let field_len = ring_len
        .checked_mul(shared.gen_ring_dim())
        .filter(|&len| len <= shared.as_field_slice().len())
        .ok_or_else(|| format!("truncation length {ring_len} exceeds the expanded matrix"))?;
    let truncated = FlatMatrix::from_flat_data(
        shared.as_field_slice()[..field_len].to_vec(),
        shared.gen_ring_dim(),
    );
    Ok(AkitaVerifierSetup {
        expanded: Arc::new(
            akita_types::AkitaExpandedSetup::from_trusted_seed_derived_parts_unchecked(
                expanded.seed().clone(),
                truncated,
            ),
        ),
        prefix_slots: verifier_setup.prefix_slots.clone(),
    })
}

/// Find the minimal shared-matrix ring prefix the native verifier reads for
/// this proof, by binary search over native verification with a truncated
/// matrix. Verification reads are prefix views (`ring_view`), so the
/// pass/fail predicate is monotone in the prefix length.
fn minimal_verifier_setup_ring_len<Cfg>(
    artifact: &ProvenArtifact,
    transcript_domain: &[u8],
) -> Result<usize, String>
where
    Cfg: CommitmentConfig<Field = F, ExtField = E>,
    RecursiveCommitmentConfig<Cfg>: CommitmentConfig<Field = F, ExtField = E>,
{
    let full_len = artifact
        .verifier_setup
        .expanded
        .shared_matrix()
        .total_ring_elements();
    let verify_at = |ring_len: usize| -> Result<bool, String> {
        let truncated = truncated_verifier_setup(&artifact.verifier_setup, ring_len)?;
        let t0 = Instant::now();
        let outcome = verify_artifact_with_mode::<Cfg>(
            artifact,
            &truncated,
            transcript_domain,
            SetupContributionMode::Recursive,
        );
        let ok = outcome.is_ok();
        // The failure reason identifies which read path (or the capacity
        // precheck) binds at this prefix length.
        let reason = outcome.err().unwrap_or_default();
        tracing::info!(
            ring_len,
            ok,
            reason = %reason,
            elapsed_s = t0.elapsed().as_secs_f64(),
            "truncation probe verify"
        );
        Ok(ok)
    };
    // Exponential probing first: the verifier-read prefix is expected to be
    // orders of magnitude smaller than the prover's envelope (the recursive
    // setup materializes the whole setup-prefix; the verifier never reads
    // it), and each probe copies its own prefix, so starting from the top
    // would clone gigabytes per probe.
    let mut hi = 1usize;
    while hi < full_len && !verify_at(hi)? {
        hi = hi.saturating_mul(2).min(full_len);
    }
    if !verify_at(hi)? {
        return Err("proof does not verify against the full matrix".to_string());
    }
    let mut lo = if hi == 1 { 1 } else { hi / 2 + 1 };
    // Invariant: verify_at(hi) == true; lo is the smallest untested length.
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        if verify_at(mid)? {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }
    tracing::info!(
        minimal_ring_len = hi,
        full_ring_len = full_len,
        "minimal verifier-read setup prefix found"
    );
    Ok(hi)
}

fn transport_output_path(
    output_path: &std::path::Path,
    transport: SetupTransport,
    multiple: bool,
) -> PathBuf {
    if !multiple {
        return output_path.to_path_buf();
    }
    match transport {
        SetupTransport::ExpandedMatrix => output_path.to_path_buf(),
        SetupTransport::MatrixSlice => {
            let mut name = output_path.as_os_str().to_os_string();
            name.push(".slice");
            PathBuf::from(name)
        }
        SetupTransport::SeedDerived => {
            let mut name = output_path.as_os_str().to_os_string();
            name.push(".seed");
            PathBuf::from(name)
        }
    }
}

#[allow(clippy::too_many_lines)]
fn run_with_preset<Cfg>(
    nv: usize,
    output_path: &std::path::Path,
    setup_contribution_mode: SetupContributionMode,
    setup_transports: &[SetupTransport],
    bound_max_exclusive: u64,
    preset_name: &str,
) -> Result<(), String>
where
    Cfg: CommitmentConfig<Field = F, ExtField = E>,
    RecursiveCommitmentConfig<Cfg>: CommitmentConfig<Field = F, ExtField = E>,
    ConservativeCommitmentConfig<Cfg>: CommitmentConfig<Field = F, ExtField = E>,
{
    let transcript_domain = transcript_domain_for::<Cfg>()?;
    let prime = fp64_prime_label();
    let proxy_label = match setup_contribution_mode {
        SetupContributionMode::Direct => "synthetic single-group proxy",
        SetupContributionMode::Recursive => "synthetic multi-group recursive proxy",
    };
    tracing::info!(
        nv,
        d = D,
        preset = preset_name,
        mode = ?setup_contribution_mode,
        transports = ?setup_transports,
        bound_max_exclusive,
        prime = %prime,
        "generating Akita fp64 verifier-input artifact ({proxy_label}, not an Aerie/Falcon result)"
    );

    let artifact = match setup_contribution_mode {
        SetupContributionMode::Direct => prove_direct::<Cfg>(nv, bound_max_exclusive)?,
        SetupContributionMode::Recursive => prove_recursive::<Cfg>(nv, bound_max_exclusive)?,
    };

    // Sanity check: the proof should verify with the same domain label.
    let t0 = Instant::now();
    verify_artifact_with_mode::<Cfg>(
        &artifact,
        &artifact.verifier_setup,
        transcript_domain,
        setup_contribution_mode,
    )
    .map_err(|err| format!("host-side sanity verify failed: {err}"))?;
    tracing::info!(
        elapsed_s = t0.elapsed().as_secs_f64(),
        "host-side verify OK"
    );

    let full_ring_len = artifact
        .verifier_setup
        .expanded
        .shared_matrix()
        .total_ring_elements();
    let needs_minimal = setup_transports
        .iter()
        .any(|transport| *transport != SetupTransport::ExpandedMatrix);
    let minimal_ring_len = if needs_minimal {
        let minimal = minimal_verifier_setup_ring_len::<Cfg>(&artifact, transcript_domain)?;
        let gen_ring_dim = artifact
            .verifier_setup
            .expanded
            .shared_matrix()
            .gen_ring_dim();
        tracing::info!(
            minimal_ring_len = minimal,
            full_ring_len,
            slice_bytes = minimal * gen_ring_dim * std::mem::size_of::<F>(),
            full_bytes = full_ring_len * gen_ring_dim * std::mem::size_of::<F>(),
            "verifier-read setup prefix"
        );
        Some(minimal)
    } else {
        None
    };

    let proof_shape = artifact.proof.shape();
    {
        use akita_recursion_glue::BLOB_COMPRESS;
        use akita_serialization::AkitaSerialize;
        let proof_bytes = artifact.proof.serialized_size(BLOB_COMPRESS);
        let matrix_bytes = artifact
            .verifier_setup
            .expanded
            .shared_matrix()
            .serialized_size(BLOB_COMPRESS);
        let prefix_slot_bytes = artifact
            .verifier_setup
            .prefix_slots
            .serialized_size(BLOB_COMPRESS);
        tracing::info!(
            proof_bytes,
            matrix_bytes,
            prefix_slot_bytes,
            "blob component encoded sizes"
        );
        if proof_bytes > (64 << 20) {
            tracing::warn!(shape = ?proof_shape, "oversized proof; full shape dump");
        }
    }
    let mut published = 0usize;
    for &setup_transport in setup_transports {
        // The recursive setup's envelope includes the materialized
        // setup-prefix, which at the Aerie shapes is orders of magnitude
        // larger than the direct envelope and cannot fit the blob cap. Skip
        // (rather than abort) so a `full,slice,seed` run still delivers the
        // transports that do fit.
        if setup_transport == SetupTransport::ExpandedMatrix {
            let matrix_bytes = artifact
                .verifier_setup
                .expanded
                .shared_matrix()
                .as_field_slice()
                .len()
                .saturating_mul(std::mem::size_of::<F>());
            if matrix_bytes as u64 > akita_recursion_glue::MAX_JOLT_BLOB_BYTES {
                tracing::warn!(
                    matrix_bytes,
                    full_ring_len,
                    cap = akita_recursion_glue::MAX_JOLT_BLOB_BYTES,
                    "skipping expanded-matrix transport: the setup envelope exceeds the blob \
                     cap; use --setup-transport slice or seed"
                );
                continue;
            }
        }
        let shipped_setup_ring_len = match setup_transport {
            SetupTransport::ExpandedMatrix => full_ring_len,
            SetupTransport::MatrixSlice | SetupTransport::SeedDerived => minimal_ring_len
                .ok_or_else(|| "minimal setup length missing for truncated transport".to_string())?,
        };
        let inputs: AkitaJoltInputs<F, E, D> = AkitaJoltInputs {
            transcript_domain: transcript_domain.to_vec(),
            num_vars: nv as u64,
            setup_contribution_mode,
            setup_transport,
            shipped_setup_ring_len: shipped_setup_ring_len as u64,
            opening_point: artifact.point.clone(),
            groups: artifact.groups.clone(),
            verifier_setup: artifact.verifier_setup.clone(),
            proof_shape: proof_shape.clone(),
            proof: artifact.proof.clone(),
        };

        let blob = inputs
            .write_to_bytes()
            .map_err(|err| format!("encode jolt inputs blob failed: {err}"))?;
        // Round-trip before publishing so a buggy encoding fails on the host
        // instead of leaving a trusted benchmark artifact on disk. For
        // truncated transports this also exercises the sliced-setup verify
        // path natively.
        let decoded = AkitaJoltInputs::<F, E, D>::read_from_bytes(&blob)
            .map_err(|err| format!("decode jolt inputs blob (round-trip) failed: {err}"))?;
        let decoded_artifact = ProvenArtifact {
            point: decoded.opening_point.clone(),
            groups: decoded.groups.clone(),
            verifier_setup: decoded.verifier_setup.clone(),
            proof: decoded.proof,
        };
        verify_artifact_with_mode::<Cfg>(
            &decoded_artifact,
            &decoded_artifact.verifier_setup,
            &decoded.transcript_domain,
            decoded.setup_contribution_mode,
        )
        .map_err(|err| format!("decoded blob verify failed: {err}"))?;
        tracing::info!(transport = ?setup_transport, "decoded-blob verify OK");

        let transport_path =
            transport_output_path(output_path, setup_transport, setup_transports.len() > 1);
        publish_blob(&transport_path, &blob)?;

        let blob_kib = (blob.len() as f64) / 1024.0;
        let blob_mib = blob_kib / 1024.0;
        tracing::info!(
            nv,
            d = D,
            preset = preset_name,
            mode = ?setup_contribution_mode,
            transport = ?setup_transport,
            shipped_setup_ring_len,
            full_ring_len,
            bytes = blob.len(),
            kib = blob_kib,
            mib = blob_mib,
            path = %transport_path.display(),
            "wrote akita-recursion fp64 verifier-input blob ({proxy_label}, not an Aerie/Falcon result)"
        );
        eprintln!(
            "wrote {} bytes ({:.2} MiB) to {} [{} nv={} {:?}/{:?} {}]",
            blob.len(),
            blob_mib,
            transport_path.display(),
            preset_name,
            nv,
            setup_contribution_mode,
            setup_transport,
            proxy_label,
        );
        published += 1;
    }
    if published == 0 {
        return Err("no verifier-input blob could be published for the requested transports"
            .to_string());
    }
    Ok(())
}

fn run() -> Result<(), String> {
    let args = Args::parse();
    let setup_contribution_mode = args.setup_mode.into_mode();
    let setup_transports: Vec<SetupTransport> = args
        .setup_transport
        .iter()
        .map(|arg| arg.into_transport())
        .collect();
    for (idx, transport) in setup_transports.iter().enumerate() {
        if setup_transports[..idx].contains(transport) {
            return Err(format!("duplicate --setup-transport value {transport:?}"));
        }
        if *transport != SetupTransport::ExpandedMatrix
            && setup_contribution_mode != SetupContributionMode::Recursive
        {
            return Err(
                "--setup-transport slice/seed requires --setup-mode recursive (direct replay \
                 scans the full expanded matrix)"
                    .to_string(),
            );
        }
    }

    #[cfg(feature = "parallel")]
    rayon::ThreadPoolBuilder::new()
        .stack_size(64 * 1024 * 1024)
        .build_global()
        .ok();

    if cfg!(debug_assertions) && env::var("AKITA_ALLOW_DEBUG_PROFILE").as_deref() != Ok("1") {
        return Err(
            "akita-recursion-artifact must be run with --release for sane runtimes.\n\
             Re-run with: cargo run --release -p akita-recursion-artifact\n\
             Set AKITA_ALLOW_DEBUG_PROFILE=1 to override this guard."
                .to_string(),
        );
    }

    let log_filter =
        EnvFilter::try_new(env::var("AKITA_RECURSION_LOG").unwrap_or_else(|_| "info".to_string()))
            .unwrap_or_else(|_| EnvFilter::new("info"));
    let _ = tracing_subscriber::fmt()
        .with_env_filter(log_filter)
        .with_target(false)
        .try_init();

    let default_nv = match args.preset {
        PresetArg::Bound18 => 23,
        PresetArg::Bound6 => 26,
    };
    let nv: usize = env_usize("AKITA_NUM_VARS", default_nv)?;
    let output_path = PathBuf::from(env_string(
        "AKITA_RECURSION_BLOB",
        "target/akita_recursion_inputs.bin",
    )?);

    // Recursive prove/verify recurses deeply; run on a large dedicated stack
    // (mirrors the e2e test harness).
    let preset = args.preset;
    std::thread::Builder::new()
        .stack_size(PROVER_STACK_BYTES)
        .spawn(move || match preset {
            PresetArg::Bound18 => run_with_preset::<fp64::D128FullBound18>(
                nv,
                &output_path,
                setup_contribution_mode,
                &setup_transports,
                1u64 << 17,
                "fp64-d128-bound18",
            ),
            PresetArg::Bound6 => run_with_preset::<fp64::D128FullBound6>(
                nv,
                &output_path,
                setup_contribution_mode,
                &setup_transports,
                1u64 << 5,
                "fp64-d128-bound6",
            ),
        })
        .map_err(|err| format!("failed to spawn artifact thread: {err}"))?
        .join()
        .map_err(|_| "artifact thread panicked".to_string())?
}

fn main() {
    match run() {
        Ok(()) => {}
        Err(err) => {
            eprintln!("error: {err}");
            std::process::exit(2);
        }
    }
}
