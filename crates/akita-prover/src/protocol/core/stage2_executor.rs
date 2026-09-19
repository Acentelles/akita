//! Typed execution boundary; both policies use the canonical sumcheck transcript driver.
use super::*;

#[derive(Clone, Copy)]
pub(super) struct Stage2Context {
    pub level: usize,
    pub basis: usize,
    pub columns: usize,
    pub lanes: usize,
    pub domain: usize,
    pub compression_layers: usize,
    pub negative_binary_intervals: usize,
}

impl Stage2Context {
    #[cfg(feature = "resident-stage2-observer")]
    fn event(
        self,
        declined: Option<&'static str>,
        counts: [usize; 3],
    ) -> crate::stage2_observer::Event {
        crate::stage2_observer::Event {
            level: self.level,
            basis: self.basis,
            columns: self.columns,
            lanes: self.lanes,
            domain: self.domain,
            compression_layers: self.compression_layers,
            negative_binary_intervals: self.negative_binary_intervals,
            declined,
            compact_entries: counts[0],
            advances: counts[1],
            exports: counts[2],
        }
    }
    #[cfg(feature = "resident-stage2-owned")]
    fn decline_reason(self) -> Option<&'static str> {
        if ![4, 8].contains(&self.basis) {
            Some("digit-basis")
        } else if self.columns < 8 || !self.columns.is_power_of_two() {
            Some("coefficient-width")
        } else if self.domain > (1 << 26) {
            Some("domain-capacity")
        } else {
            None
        }
    }
}

type Completed<E> = ((SumcheckProof<E>, Vec<E>, E), RelationRangeImageProver<E>);

pub(super) trait Stage2Executor<F, E>
where
    F: FieldCore + CanonicalField,
    E: ExtField<F> + HasUnreducedOps + HasOptimizedFold + FromPrimitiveInt + AkitaSerialize,
{
    fn prove<T: Transcript<F>>(
        &self,
        prover: RelationRangeImageProver<E>,
        transcript: &mut T,
        context: Stage2Context,
    ) -> Result<Completed<E>, AkitaError>;
}

pub(super) struct CpuStage2;

impl<F, E> Stage2Executor<F, E> for CpuStage2
where
    F: FieldCore + CanonicalField,
    E: ExtField<F> + HasUnreducedOps + HasOptimizedFold + FromPrimitiveInt + AkitaSerialize,
{
    fn prove<T: Transcript<F>>(
        &self,
        mut prover: RelationRangeImageProver<E>,
        transcript: &mut T,
        context: Stage2Context,
    ) -> Result<Completed<E>, AkitaError> {
        let _ = (
            context.level,
            context.basis,
            context.columns,
            context.lanes,
            context.domain,
            context.compression_layers,
            context.negative_binary_intervals,
        );
        let output = prover.prove::<F, T, _>(transcript, |tr| {
            sample_ext_challenge::<F, E, T>(tr, CHALLENGE_SUMCHECK_ROUND)
        })?;
        Ok((output, prover))
    }
}

#[cfg(feature = "resident-stage2-owned")]
pub(super) struct ResidentStage2;

#[cfg(feature = "resident-stage2-owned")]
impl Stage2Executor<akita_field::Prime64Offset59, akita_field::Ext2<akita_field::Prime64Offset59>>
    for ResidentStage2
{
    fn prove<T: Transcript<akita_field::Prime64Offset59>>(
        &self,
        prover: RelationRangeImageProver<akita_field::Ext2<akita_field::Prime64Offset59>>,
        transcript: &mut T,
        context: Stage2Context,
    ) -> Result<Completed<akita_field::Ext2<akita_field::Prime64Offset59>>, AkitaError> {
        use crate::protocol::sumcheck::relation_range_image::resident_owned::ResidentRelationProver;
        use akita_field::{Ext2, Prime64Offset59};
        if let Some(reason) = context.decline_reason() {
            // A static routing decision, before any owner creation or Stage2 transcript event.
            let result =
                <CpuStage2 as Stage2Executor<Prime64Offset59, Ext2<Prime64Offset59>>>::prove(
                    &CpuStage2, prover, transcript, context,
                )?;
            #[cfg(feature = "resident-stage2-observer")]
            crate::stage2_observer::record(context.event(Some(reason), [0; 3]))?;
            let _ = reason;
            return Ok(result);
        }
        // Admission and owner creation precede the Stage2 driver's claim absorption.
        // Earlier Stage1 events remain in the transcript on admission failure.
        let mut owner = ResidentRelationProver::new(prover)?;
        let output = akita_sumcheck::prove_fallible_sumcheck::<
            Prime64Offset59,
            Ext2<Prime64Offset59>,
            T,
            _,
            _,
        >(&mut owner, transcript, |tr| {
            sample_ext_challenge::<Prime64Offset59, Ext2<Prime64Offset59>, T>(
                tr,
                CHALLENGE_SUMCHECK_ROUND,
            )
        })?;
        #[cfg(feature = "resident-stage2-observer")]
        let counts = owner.execution_counts();
        let completed = owner.into_completed()?;
        #[cfg(feature = "resident-stage2-observer")]
        crate::stage2_observer::record(context.event(None, counts))?;
        Ok((output, completed))
    }
}
