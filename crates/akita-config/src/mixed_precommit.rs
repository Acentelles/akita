//! Mixed-bound multi-group proving config adapter.
//!
//! [`MixedPrecommitConfig<Main, Pre>`] proves multi-group roots whose final
//! group commits under preset `Main` while every precommitted group froze
//! under preset `Pre`'s conservative adapter
//! (`ConservativeCommitmentConfig<Pre>`). This is the config-declared static
//! per-group policy map required for mixed `log_commit_bound` roots: the
//! per-group bound policies enter the runtime schedule key (and, through it,
//! the transcript instance descriptor) purely from `Main` / `Pre` statics,
//! never from prover bytes.
//!
//! Mixing is only defined within one field / ring family: `Pre` must share
//! `Main`'s base and extension fields (enforced by the impl bounds) and its
//! ring dimension, SIS family, effective field width, and psi norm bound
//! (enforced at key-construction time; a mismatch is a loud
//! `InvalidSetup`, before any schedule is planned).

use crate::{CommitmentConfig, ConservativeCommitmentConfig};
use akita_challenges::{SparseChallengeConfig, TensorChallengeShape};
use akita_field::AkitaError;
use akita_types::{
    AkitaScheduleInputs, DecompositionParams, OpeningClaimsLayout, PolynomialGroupLayout,
    PrecommittedGroupParams, Schedule, SetupMatrixEnvelope, SisModulusFamily,
};
use std::marker::PhantomData;

/// Proving config whose precommitted groups froze under `Pre` while the final
/// group and the multi-group root plan under `Main`.
#[derive(Clone, Copy, Debug, Default)]
pub struct MixedPrecommitConfig<Main, Pre>(PhantomData<(Main, Pre)>);

/// Reject `(Main, Pre)` pairs that do not share the invariants mixed roots
/// assume. Field-type equality is already enforced by the impl bounds; the
/// remaining shared inputs are checked here so a bad pairing fails loudly at
/// schedule-key construction time.
fn validate_mixed_pair<Main, Pre>() -> Result<(), AkitaError>
where
    Main: CommitmentConfig,
    Pre: CommitmentConfig<Field = Main::Field, ExtField = Main::ExtField>,
{
    if Pre::D != Main::D {
        return Err(AkitaError::InvalidSetup(format!(
            "mixed precommit config requires matching ring dimension: main D={}, pre D={}",
            Main::D,
            Pre::D
        )));
    }
    if Pre::sis_modulus_family() != Main::sis_modulus_family() {
        return Err(AkitaError::InvalidSetup(
            "mixed precommit config requires matching SIS modulus family".to_string(),
        ));
    }
    if Pre::decomposition().field_bits() != Main::decomposition().field_bits() {
        return Err(AkitaError::InvalidSetup(
            "mixed precommit config requires matching effective field width".to_string(),
        ));
    }
    if Pre::ring_subfield_embedding_norm_bound() != Main::ring_subfield_embedding_norm_bound() {
        return Err(AkitaError::InvalidSetup(
            "mixed precommit config requires matching psi embedding norm bound".to_string(),
        ));
    }
    Ok(())
}

impl<Main, Pre> CommitmentConfig for MixedPrecommitConfig<Main, Pre>
where
    Main: CommitmentConfig,
    Pre: CommitmentConfig<Field = Main::Field, ExtField = Main::ExtField>,
{
    type Field = Main::Field;
    type ExtField = Main::ExtField;

    const D: usize = Main::D;

    fn decomposition() -> DecompositionParams {
        Main::decomposition()
    }

    fn ring_challenge_config(d: usize) -> Result<SparseChallengeConfig, AkitaError> {
        Main::ring_challenge_config(d)
    }

    fn fold_challenge_shape_at_level(inputs: AkitaScheduleInputs) -> TensorChallengeShape {
        Main::fold_challenge_shape_at_level(inputs)
    }

    fn sis_modulus_family() -> SisModulusFamily {
        Main::sis_modulus_family()
    }

    fn ring_subfield_embedding_norm_bound() -> u32 {
        Main::ring_subfield_embedding_norm_bound()
    }

    fn max_setup_matrix_size(
        max_num_vars: usize,
        max_num_batched_polys: usize,
    ) -> Result<SetupMatrixEnvelope, AkitaError> {
        // Sized under `Self`, not `Main`: the envelope scan's multi-group
        // shapes then freeze their precommitted groups under `Pre` (via
        // `Self::precommitted_group_params`), so mixed-bound root footprints
        // enter the shared setup envelope. This can only enlarge the
        // envelope relative to `Main`'s own scan (conservative over-sizing).
        crate::proof_optimized::proof_optimized_max_setup_matrix_size::<Self>(
            max_num_vars,
            max_num_batched_polys,
        )
    }

    fn basis_range() -> (u32, u32) {
        Main::basis_range()
    }

    fn onehot_chunk_size() -> usize {
        Main::onehot_chunk_size()
    }

    fn chunked_witness_cfg() -> akita_types::ChunkedWitnessCfg {
        Main::chunked_witness_cfg()
    }

    // No shipped catalog: mixed keys are never table entries; every schedule
    // regenerates through the DP with the per-group frozen bound policies.
    fn schedule_catalog() -> Option<akita_planner::GeneratedScheduleTable> {
        None
    }

    fn precommitted_group_params(
        _group_index: usize,
        group: PolynomialGroupLayout,
    ) -> Result<PrecommittedGroupParams, AkitaError> {
        validate_mixed_pair::<Main, Pre>()?;
        // Every precommitted group froze under `Pre`'s conservative adapter.
        // This must stay byte-identical to what
        // `ConservativeCommitmentConfig<Pre>` commits, which it is by
        // construction (both call the same freeze); the transcript instance
        // descriptor binds the result, so any drift rejects at verify time.
        crate::conservative_commitment::conservative_precommitted_group_params::<Pre>(group)
    }

    fn get_params_for_prove(layout: &OpeningClaimsLayout) -> Result<Schedule, AkitaError> {
        Self::runtime_schedule(
            crate::proof_optimized::proof_optimized_schedule_key::<Self>(layout)?,
        )
    }
}

/// Conservative committing adapter for the precommitted groups of a
/// [`MixedPrecommitConfig`]; provided so callers name one config family.
pub type MixedPrecommitConservative<Pre> = ConservativeCommitmentConfig<Pre>;

#[cfg(test)]
mod boundary_scan_tmp {
    use super::*;
    use crate::proof_optimized::fp32;
    use akita_types::{PolynomialGroupLayout, Step};

    type MixedCfg32 = MixedPrecommitConfig<fp32::D128FullFastProver, fp32::D128OneHot>;

    #[test]
    fn mixed_fp32_fold_boundary_scan() {
        for final_nv in [14usize, 16, 18, 19, 20, 21] {
            for polys in [2usize, 6] {
                let layout = OpeningClaimsLayout::from_root_groups(
                    &[PolynomialGroupLayout::new(10, 1)],
                    PolynomialGroupLayout::new(final_nv, polys),
                )
                .unwrap();
                match MixedCfg32::get_params_for_prove(&layout) {
                    Ok(s) => eprintln!(
                        "nv={final_nv} polys={polys} first_step_fold={}",
                        matches!(s.steps.first(), Some(Step::Fold(_)))
                    ),
                    Err(e) => eprintln!("nv={final_nv} polys={polys} err={e}"),
                }
            }
        }
    }
}
