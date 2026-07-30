//! Akita-specific sumcheck stage implementations.
//!
//! Generic sumcheck proof types, traits, and drivers live in `akita-sumcheck`.
//! This module keeps the Akita stage-1/stage-2 instances and the prover-internal
//! two-round-prefix optimization beside the protocol code they depend on.

pub mod akita_stage1;
pub mod akita_stage1_tree;
pub mod akita_stage2;
pub mod akita_stage3;
pub mod two_round_prefix;

#[cfg(test)]
mod perf_micro;

pub use akita_stage1_tree::AkitaStage1Prover;
pub use akita_stage2::AkitaStage2Prover;
pub use akita_stage3::AkitaStage3Prover;

// --- Shared helpers ------------------------------------------------------

use akita_field::unreduced::HasUnreducedOps;
use akita_field::{FieldCore, Zero};

/// Fixed-width product-accumulation lanes with reduction deferred to the end
/// of an accumulation window when the field's delayed product sum is exact
/// (`DELAYED_PRODUCT_SUM_IS_EXACT`), and per-term reduction otherwise.
///
/// The two behaviors produce the identical canonical values: per-term
/// reduction trivially, the deferred path by the field's exactness contract
/// (asserted against `PRODUCT_ACCUM_MAX_TERMS` at the call sites). The flag is
/// an associated `const`, so the branch monomorphizes away.
pub(crate) struct ProductLanes<E: HasUnreducedOps, const N: usize> {
    delayed: [E::ProductAccum; N],
    direct: [E; N],
}

impl<E: FieldCore + HasUnreducedOps, const N: usize> ProductLanes<E, N> {
    #[inline]
    pub(crate) fn zero() -> Self {
        Self {
            delayed: [E::ProductAccum::zero(); N],
            direct: [E::zero(); N],
        }
    }

    /// Accumulate `lhs · rhs` into lane `k`.
    #[inline]
    pub(crate) fn add_product(&mut self, k: usize, lhs: E, rhs: E) {
        if E::DELAYED_PRODUCT_SUM_IS_EXACT {
            self.delayed[k] += lhs.mul_to_product_accum(rhs);
        } else {
            self.direct[k] += lhs * rhs;
        }
    }

    /// Reduce every lane to its canonical field value.
    #[inline]
    pub(crate) fn finish(self) -> [E; N] {
        if E::DELAYED_PRODUCT_SUM_IS_EXACT {
            self.delayed.map(E::reduce_product_accum)
        } else {
            self.direct
        }
    }
}

/// Fold a pair of adjacent evaluations in a full-width row at a challenge `r`,
/// with implicit zero-padding when the index falls past the end.
#[inline]
pub(crate) fn fold_full_prefix_pair<E: FieldCore>(row: &[E], left: usize, r: E) -> E {
    let v0 = row.get(left).copied().unwrap_or_else(E::zero);
    let v1 = row.get(left + 1).copied().unwrap_or_else(E::zero);
    v0 + r * (v1 - v0)
}

#[cfg(test)]
mod product_lanes_tests {
    use super::*;
    use akita_field::unreduced::HasUnreducedOps;
    use akita_field::{FpExt4, Prime128Offset275, Prime32Offset99, RandomSampling};
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    /// R7 kernel differential: for the exact accumulation pattern used by the
    /// deferred stage-2 lanes (`Σ_j e_j · q_j`, reduced once per window), the
    /// deferred path must equal per-term reduce-then-add, elementwise, for
    /// window lengths up to and past the production block sizes.
    fn check_lanes_match_per_term<E>(seed: u64)
    where
        E: FieldCore + HasUnreducedOps + RandomSampling + std::fmt::Debug,
    {
        let mut rng = StdRng::seed_from_u64(seed);
        for window in [1usize, 2, 3, 64, 1000, 4096] {
            assert!(window <= E::PRODUCT_ACCUM_MAX_TERMS);
            let mut lanes = ProductLanes::<E, 3>::zero();
            let mut direct = [E::zero(); 3];
            for _ in 0..window {
                let e = E::random(&mut rng);
                for (k, slot) in direct.iter_mut().enumerate() {
                    let q = E::random(&mut rng);
                    lanes.add_product(k, e, q);
                    *slot += e * q;
                }
            }
            assert_eq!(lanes.finish(), direct, "window {window}");
        }
    }

    #[test]
    fn product_lanes_match_per_term_reduction() {
        // Deferred branch (DELAYED_PRODUCT_SUM_IS_EXACT = true).
        check_lanes_match_per_term::<FpExt4<Prime32Offset99>>(0xacc0_0001);
        // Per-term branch (flag false): trivially identical, pins the gate.
        check_lanes_match_per_term::<Prime128Offset275>(0xacc0_0002);
    }

    /// The exact-headroom window bound must accommodate every production
    /// block size (blocks are at most `e_first.len()`, far below 2^32).
    #[test]
    fn product_accum_window_covers_production_blocks() {
        assert!(FpExt4::<Prime32Offset99>::PRODUCT_ACCUM_MAX_TERMS > 1 << 32);
    }
}
