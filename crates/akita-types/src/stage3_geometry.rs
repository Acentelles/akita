//! Shared batched Stage-3 point geometry.
//!
//! This module is the single source of truth for projecting the batched
//! Stage-3 challenge into witness/setup points and for routing those projected
//! points into the next recursive suffix opening batch.

use akita_field::AkitaError;
use akita_field::{FieldCore, FromPrimitiveInt};

use crate::{PointVariableSelection, SetupPrefixSlotId};

/// Geometry for one batched Stage-3 setup-product plus carried-witness sumcheck.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BatchedStage3Geometry {
    witness_rounds: usize,
    setup_rounds: usize,
    batched_rounds: usize,
}

impl BatchedStage3Geometry {
    /// Build the shared Stage-3 geometry.
    ///
    /// `batched_rounds` is the common padded cube dimension. Native witness and
    /// setup coordinates occupy the suffix of the batched challenge vector.
    pub fn new(witness_rounds: usize, setup_rounds: usize) -> Result<Self, AkitaError> {
        if witness_rounds == 0 || setup_rounds == 0 {
            return Err(AkitaError::InvalidSetup(
                "batched stage-3 native round counts must be nonzero".to_string(),
            ));
        }
        Ok(Self {
            witness_rounds,
            setup_rounds,
            batched_rounds: witness_rounds.max(setup_rounds),
        })
    }

    /// Native witness round count.
    #[must_use]
    pub fn witness_rounds(&self) -> usize {
        self.witness_rounds
    }

    /// Native setup round count.
    #[must_use]
    pub fn setup_rounds(&self) -> usize {
        self.setup_rounds
    }

    /// Common padded Stage-3 round count.
    #[must_use]
    pub fn batched_rounds(&self) -> usize {
        self.batched_rounds
    }

    /// Project the batched challenge onto the native witness point.
    pub fn witness_point<E: Clone>(&self, rho: &[E]) -> Result<Vec<E>, AkitaError> {
        self.project_native_point(rho, self.witness_rounds)
    }

    /// Project the batched challenge onto the native setup point.
    pub fn setup_point<E: Clone>(&self, rho: &[E]) -> Result<Vec<E>, AkitaError> {
        self.project_native_point(rho, self.setup_rounds)
    }

    /// Split `rho_setup` into ring-coordinate `rho_y` and setup-index tail.
    pub fn setup_y_and_index<'a, E>(
        &self,
        rho_setup: &'a [E],
        ring_bits: usize,
    ) -> Result<(&'a [E], &'a [E]), AkitaError> {
        if rho_setup.len() != self.setup_rounds {
            return Err(AkitaError::InvalidPointDimension {
                expected: self.setup_rounds,
                actual: rho_setup.len(),
            });
        }
        if ring_bits > rho_setup.len() {
            return Err(AkitaError::InvalidPointDimension {
                expected: rho_setup.len(),
                actual: ring_bits,
            });
        }
        Ok(rho_setup.split_at(ring_bits))
    }

    /// Lifting scale for the witness term embedded into the common cube.
    pub fn witness_lift_scale<E: FieldCore + FromPrimitiveInt>(&self) -> Result<E, AkitaError> {
        lift_scale(self.batched_rounds - self.witness_rounds)
    }

    /// Lifting scale for the setup term embedded into the common cube.
    pub fn setup_lift_scale<E: FieldCore + FromPrimitiveInt>(&self) -> Result<E, AkitaError> {
        lift_scale(self.batched_rounds - self.setup_rounds)
    }

    /// Merge Stage-3 setup/witness points into the shared point used by the
    /// next suffix opening batch.
    ///
    /// Returns `(shared_point, setup_offset)`, where `setup_offset` is the first
    /// coordinate of `setup_prefix_point` inside `shared_point`.
    pub fn shared_suffix_point<E: FieldCore>(
        setup_prefix_point: &[E],
        witness_point: &[E],
    ) -> Result<(Vec<E>, usize), AkitaError> {
        if setup_prefix_point.len() >= witness_point.len() {
            if &setup_prefix_point[setup_prefix_point.len() - witness_point.len()..]
                != witness_point
            {
                return Err(AkitaError::InvalidInput(
                    "stage-3 suffix opening points are inconsistent".to_string(),
                ));
            }
            Ok((setup_prefix_point.to_vec(), 0))
        } else {
            if &witness_point[witness_point.len() - setup_prefix_point.len()..]
                != setup_prefix_point
            {
                return Err(AkitaError::InvalidInput(
                    "stage-3 suffix opening points are inconsistent".to_string(),
                ));
            }
            Ok((
                witness_point.to_vec(),
                witness_point.len() - setup_prefix_point.len(),
            ))
        }
    }

    /// Coordinate routing for a setup-prefix group in the next suffix opening batch.
    pub fn setup_prefix_column_major_point_vars(
        setup_prefix_point_len: usize,
        setup_prefix_id: &SetupPrefixSlotId,
        offset: usize,
        shared_point_len: usize,
    ) -> Result<PointVariableSelection, AkitaError> {
        if setup_prefix_id.d_setup == 0 {
            return Err(AkitaError::InvalidSetup(
                "setup-prefix d_setup must be nonzero".to_string(),
            ));
        }
        let ring_bits = setup_prefix_id.d_setup.trailing_zeros() as usize;
        let params = &setup_prefix_id.commitment_params;
        let expected = ring_bits
            .checked_add(params.layout.r_vars)
            .and_then(|n| n.checked_add(params.layout.m_vars))
            .ok_or_else(|| AkitaError::InvalidSetup("setup-prefix point length overflow".into()))?;
        if setup_prefix_point_len != expected {
            return Err(AkitaError::InvalidPointDimension {
                expected,
                actual: setup_prefix_point_len,
            });
        }
        let mut indices = Vec::with_capacity(expected);
        indices.extend(offset..offset + ring_bits);
        indices.extend(
            offset + ring_bits + params.layout.m_vars
                ..offset + ring_bits + params.layout.m_vars + params.layout.r_vars,
        );
        indices.extend(offset + ring_bits..offset + ring_bits + params.layout.m_vars);
        PointVariableSelection::new(indices, shared_point_len)
    }

    fn project_native_point<E: Clone>(
        &self,
        rho: &[E],
        native_rounds: usize,
    ) -> Result<Vec<E>, AkitaError> {
        if rho.len() != self.batched_rounds {
            return Err(AkitaError::InvalidPointDimension {
                expected: self.batched_rounds,
                actual: rho.len(),
            });
        }
        Ok(rho[self.batched_rounds - native_rounds..].to_vec())
    }
}

/// Per-group packed opening point after a suffix-aligned shared
/// extension-opening reduction (`specs/eor-setup-prefix-absorption.md`).
///
/// `rho` is the reduction sumcheck point over the joint tail (shared point
/// minus the `log2([E:F])` packing head). A group whose own point occupies the
/// contiguous shared-coordinate slice `[offset, offset + len)` (its
/// `PointVariableSelection` indices, possibly permuted within the slice) has
/// its own tail at `rho[offset..]`; its packed opening point is the
/// psi-packed form of that slice, re-routed by the group's own within-slice
/// permutation.
///
/// # Errors
///
/// Returns an error when the selection is not a permutation of a contiguous
/// slice, or the slice does not fit `rho`'s implied shared arity.
pub fn suffix_aligned_group_packed_point<F, E, const D: usize>(
    rho: &[E],
    point_vars: &crate::PointVariableSelection,
) -> Result<Vec<E>, AkitaError>
where
    F: akita_field::FieldCore,
    E: akita_field::ExtField<F>,
{
    let kappa = E::EXT_DEGREE.trailing_zeros() as usize;
    let indices = point_vars.indices();
    let len = indices.len();
    if len == 0 {
        return Err(AkitaError::InvalidInput(
            "suffix-aligned group selection is empty".to_string(),
        ));
    }
    let offset = indices.iter().copied().min().unwrap_or(0);
    let end = indices.iter().copied().max().unwrap_or(0) + 1;
    if end - offset != len {
        return Err(AkitaError::InvalidInput(
            "suffix-aligned group selection is not a contiguous slice".to_string(),
        ));
    }
    let own_tail_len = len.checked_sub(kappa).ok_or_else(|| {
        AkitaError::InvalidInput("suffix-aligned group arity below the tensor split".to_string())
    })?;
    let slice = rho.get(offset..offset + own_tail_len).ok_or_else(|| {
        AkitaError::InvalidPointDimension {
            expected: offset + own_tail_len,
            actual: rho.len(),
        }
    })?;
    let packed = crate::ring_subfield_packed_extension_opening_point::<F, E, D>(slice.len(), slice)?;
    if packed.len() != len {
        return Err(AkitaError::InvalidPointDimension {
            expected: len,
            actual: packed.len(),
        });
    }
    indices
        .iter()
        .map(|&idx| {
            packed.get(idx - offset).copied().ok_or_else(|| {
                AkitaError::InvalidInput(
                    "suffix-aligned packed point routing out of range".to_string(),
                )
            })
        })
        .collect()
}

fn lift_scale<E: FieldCore + FromPrimitiveInt>(extra_rounds: usize) -> Result<E, AkitaError> {
    let inv_two = E::from_u64(2)
        .inverse()
        .ok_or_else(|| AkitaError::InvalidSetup("two is not invertible in Akita fields".into()))?;
    Ok((0..extra_rounds).fold(E::one(), |acc, _| acc * inv_two))
}

/// Bind the stage-3 setup-product claim `sigma`, then squeeze the stage-3
/// batching challenge `eta`. **In that order, and from one place.**
///
/// Fiat-Shamir hygiene: a challenge must be drawn after everything it binds.
/// `eta` weights the carried-witness half of the batched stage-3 sumcheck
/// against the setup-product half, so the stage-3 input claim is
/// `sigma + eta * stage2_next_w_eval`. `stage2_next_w_eval` is absorbed under
/// `ABSORB_STAGE2_NEXT_W_EVAL` before this point, but `sigma` previously
/// reached the transcript only *inside* that combined sum, i.e. after `eta`
/// had already been squeezed. A prover computes `eta` itself, so it could
/// choose `sigma` adaptively with `eta` in hand, and only the combination was
/// bound rather than each term.
///
/// **No attack was demonstrated against the old ordering** (see the upstream
/// attribution section of `aerie/_docs/ABSORPTION-AUDIT.md`); the linearised
/// two-equation argument suggested splitting the sum was a `1/|E|` event, but
/// that argument is incomplete because stage 2 also ships compressed round
/// polynomials, so its final claim is itself prover-steerable. This closes the
/// adaptivity unconditionally instead of relying on that argument.
///
/// Absorbing `sigma` *after* `eta` would not work: the prover already holds
/// `eta` at that point, so a later absorb looks present while closing nothing.
/// Only the ordering below removes the adaptivity, which is why prover and
/// verifier both call this helper rather than open-coding the pair.
pub fn bind_setup_product_claim_and_sample_eta<F, E, T>(setup_product_claim: &E, transcript: &mut T) -> E
where
    F: FieldCore + akita_field::CanonicalField,
    E: akita_field::ExtField<F> + akita_serialization::AkitaSerialize,
    T: akita_transcript::Transcript<F>,
{
    transcript.append_serde(
        akita_transcript::labels::ABSORB_SUMCHECK_CLAIM,
        setup_product_claim,
    );
    akita_transcript::sample_ext_challenge::<F, E, T>(
        transcript,
        akita_transcript::labels::CHALLENGE_SUMCHECK_BATCH,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        AjtaiKeyParams, PolynomialGroupLayout, PrecommittedGroupParams, PrecommittedLevelParams,
        SisModulusFamily,
    };
    use akita_field::Prime32Offset99 as F;

    fn test_prefix_id() -> SetupPrefixSlotId {
        let key = AjtaiKeyParams::new_unchecked(128, SisModulusFamily::Q32, 1, 1, 1, 32);
        SetupPrefixSlotId {
            d_setup: 32,
            natural_len: 1,
            commitment_params: PrecommittedLevelParams {
                layout: PrecommittedGroupParams {
                    group: PolynomialGroupLayout::singleton(8),
                    m_vars: 2,
                    r_vars: 1,
                    log_basis: 3,
                    n_a: 1,
                    conservative_n_b: 1,
                    log_commit_bound: 1,
                    onehot_chunk_size: 1,
                    basis_range: (1, 8),
                },
                a_key: key.clone(),
                b_key: key,
                num_blocks: 1,
                block_len: 1,
                num_digits_commit: 1,
                num_digits_open: 1,
                num_digits_fold_one: 1,
            },
        }
    }

    #[test]
    fn projects_suffix_points_for_unequal_domains() {
        let geometry = BatchedStage3Geometry::new(3, 5).expect("geometry");
        let rho = vec![
            F::from_u64(1),
            F::from_u64(2),
            F::from_u64(3),
            F::from_u64(4),
            F::from_u64(5),
        ];
        assert_eq!(
            geometry.witness_point(&rho).expect("witness"),
            vec![F::from_u64(3), F::from_u64(4), F::from_u64(5)]
        );
        assert_eq!(geometry.setup_point(&rho).expect("setup"), rho);
    }

    #[test]
    fn computes_lift_scales() {
        let geometry = BatchedStage3Geometry::new(3, 5).expect("geometry");
        let inv_four = F::from_u64(4).inverse().expect("inverse");
        assert_eq!(
            geometry.witness_lift_scale::<F>().expect("witness"),
            inv_four
        );
        assert_eq!(geometry.setup_lift_scale::<F>().expect("setup"), F::one());
    }

    #[test]
    fn shared_suffix_point_accepts_suffix_consistency() {
        let setup = vec![F::from_u64(1), F::from_u64(2), F::from_u64(3)];
        let witness = vec![F::from_u64(2), F::from_u64(3)];
        let (shared, offset) =
            BatchedStage3Geometry::shared_suffix_point(&setup, &witness).expect("shared");
        assert_eq!(shared, setup);
        assert_eq!(offset, 0);

        let setup = vec![F::from_u64(2), F::from_u64(3)];
        let witness = vec![F::from_u64(1), F::from_u64(2), F::from_u64(3)];
        let (shared, offset) =
            BatchedStage3Geometry::shared_suffix_point(&setup, &witness).expect("shared");
        assert_eq!(shared, witness);
        assert_eq!(offset, 1);
    }

    #[test]
    fn shared_suffix_point_rejects_inconsistent_suffix() {
        let setup = vec![F::from_u64(1), F::from_u64(2), F::from_u64(3)];
        let witness = vec![F::from_u64(4), F::from_u64(3)];
        assert!(BatchedStage3Geometry::shared_suffix_point(&setup, &witness).is_err());
    }

    #[test]
    fn setup_prefix_column_major_point_vars_match_existing_order() {
        let id = test_prefix_id();
        let selection = BatchedStage3Geometry::setup_prefix_column_major_point_vars(8, &id, 1, 10)
            .expect("selection");
        assert_eq!(selection.indices(), &[1, 2, 3, 4, 5, 8, 6, 7]);
    }

    /// Transcript that records the ORDER of absorbs and squeezes.
    ///
    /// Deliberately not `LoggingTranscript`: that lives behind the non-default
    /// `logging-transcript` feature, so a regression written with it would be
    /// silently SKIPPED by `cargo test --workspace` while reading as coverage.
    /// See the "decorative regression tests" section of
    /// `aerie/_docs/ABSORPTION-AUDIT.md`.
    #[derive(Default)]
    struct OrderRecordingTranscript {
        events: Vec<(&'static str, Vec<u8>)>,
        counter: u64,
    }

    impl akita_transcript::Transcript<F> for OrderRecordingTranscript {
        fn new(_domain_label: &[u8]) -> Self {
            Self::default()
        }
        fn bind_instance_bytes(&mut self, _instance_bytes: &[u8]) {}
        fn append_bytes(&mut self, label: &[u8], _bytes: &[u8]) {
            self.events.push(("absorb", label.to_vec()));
        }
        fn append_field(&mut self, label: &[u8], _x: &F) {
            self.events.push(("absorb", label.to_vec()));
        }
        fn append_serde<S: akita_serialization::AkitaSerialize>(
            &mut self,
            label: &[u8],
            _s: &S,
        ) {
            self.events.push(("absorb", label.to_vec()));
        }
        fn challenge_scalar(&mut self, label: &[u8]) -> F {
            self.events.push(("squeeze", label.to_vec()));
            self.counter += 1;
            F::from_u64(self.counter)
        }
        fn challenge_bytes(&mut self, label: &[u8], len: usize) -> Vec<u8> {
            self.events.push(("squeeze", label.to_vec()));
            vec![0u8; len]
        }
    }

    /// Regression for the stage-3 `sigma`/`eta` ordering
    /// (`aerie/_docs/ABSORPTION-AUDIT.md`, upstream attribution item 1).
    ///
    /// `eta` MUST be squeezed strictly after the setup-product claim is
    /// absorbed, otherwise a prover can pick `sigma` with `eta` already in
    /// hand. This fails if the absorb is deleted AND if it is moved after the
    /// squeeze — the latter is the case that matters, since a post-`eta`
    /// absorb looks present but closes nothing.
    #[test]
    fn eta_is_squeezed_strictly_after_the_setup_product_claim_is_absorbed() {
        let mut transcript = <OrderRecordingTranscript as akita_transcript::Transcript<F>>::new(
            b"test/stage3-sigma-eta-order",
        );
        let sigma = F::from_u64(0x0ae2_2026_0802);
        let _eta: F = bind_setup_product_claim_and_sample_eta::<F, F, _>(&sigma, &mut transcript);

        // The eta squeeze is emitted under EXT-LIMB labels derived from
        // CHALLENGE_SUMCHECK_BATCH, never the bare label. Matching on byte
        // equality would match nothing and pass vacuously.
        let absorb_idx = transcript
            .events
            .iter()
            .position(|(kind, label)| {
                *kind == "absorb"
                    && (label.as_slice() == akita_transcript::labels::ABSORB_SUMCHECK_CLAIM
                        || akita_transcript::is_ext_limb_label(
                            label,
                            akita_transcript::labels::ABSORB_SUMCHECK_CLAIM,
                        ))
            })
            .expect(
                "stage 3 must absorb the setup-product claim; without it `sigma` is never bound \
                 before `eta` and the prover may choose it adaptively",
            );
        let squeeze_idx = transcript
            .events
            .iter()
            .position(|(kind, label)| {
                *kind == "squeeze"
                    && (label.as_slice() == akita_transcript::labels::CHALLENGE_SUMCHECK_BATCH
                        || akita_transcript::is_ext_limb_label(
                            label,
                            akita_transcript::labels::CHALLENGE_SUMCHECK_BATCH,
                        ))
            })
            .expect("stage 3 must squeeze the eta batching challenge");

        assert!(
            absorb_idx < squeeze_idx,
            "eta must be squeezed AFTER the setup-product claim is absorbed; got absorb at {} \
             and squeeze at {} in {:?}. A post-eta absorb binds nothing: the prover already \
             holds eta and can still choose sigma adaptively.",
            absorb_idx,
            squeeze_idx,
            transcript.events,
        );
    }
}
