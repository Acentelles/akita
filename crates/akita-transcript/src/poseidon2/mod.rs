//! Poseidon2 permutation and algebraic duplex sponge over `p* = 2^64 - 59`.
//!
//! # UNAUDITED RESEARCH CRYPTOGRAPHY - NOT FOR PRODUCTION
//!
//! This module implements a **new** Poseidon2 instance at a prime that nobody
//! runs in production. There is **no third-party cryptanalysis of any
//! arithmetization-oriented permutation over `p* = 2^64 - 59` at full
//! parameters**; the only public analysis over this exact prime targets the
//! deliberately round-reduced Ethereum Foundation bounty instances
//! (`t = 3`, 8-40 bit tiers). The round numbers, round constants, and linear
//! layers here come from the designers' own parameter tooling, but running that
//! tooling is not the same thing as having the resulting instance analysed.
//! Do not enable this transcript backend for anything that needs to be secure.
//!
//! # Why this exists
//!
//! The Akita-in-Akita wrapped verifier has to recompute the Fiat-Shamir
//! transcript hash *inside* a proof over the same field. A byte-oriented
//! BLAKE2b duplex is ruinous there; a permutation over `F_{p*}` is not. This
//! backend is the arithmetization-friendly alternative, kept strictly behind a
//! non-default Cargo feature so the shipped BLAKE2b transcript is untouched.
//!
//! # (H1) for the sponge actually implemented
//!
//! `_docs/COMPOSITION-SOUNDNESS.md` hypothesis (H1) is a concrete-hash
//! Fiat-Shamir extraction assumption, and Section 7 requires it to be stated
//! for the sponge that is actually implemented. Here it reads:
//!
//! > Let `pi` be the Poseidon2 permutation over `F_{p*}` with `t = 12` and the
//! > generated parameters in [`generated_alpha3`] / [`generated_alpha7`]. The
//! > Fiat-Shamir compilation of the Akita core protocol using the rate-8 /
//! > capacity-4 duplex over `pi` ([`Poseidon2Duplex`]) admits an
//! > expected-polynomial-time extractor with the knowledge error of the
//! > ideal-permutation analysis of Chiesa-Orru (eprint 2025/536), i.e. `pi` is
//! > assumed to behave as an ideal permutation for this adversary class.
//!
//! That assumption is non-falsifiable and, at this prime, *not backed by any
//! third-party cryptanalysis*. It is strictly weaker evidence than the same
//! assumption for BLAKE2b. Record it as such.
//!
//! # Construction
//!
//! - Permutation: Poseidon2 (eprint 2023/323), `t = 12`, external matrix from
//!   the fixed `M_4` block structure, internal matrix `1 1^T + diag(mu - 1)`.
//! - Sponge: duplex, rate 8 / capacity 4. Capacity 4 field elements = 256 bits
//!   gives the 128-bit level under the SAFE bound (eprint 2023/520:
//!   indifferentiability up to `|F|^{c/2}` queries).
//! - Absorption: overwrite mode, i.e. [CO25] Construction 3.3, the same
//!   construction `spongefish::DuplexSponge` implements, with a nonzero
//!   capacity IV for domain separation.
//!
//! # Where this sits, and what it deliberately does not change
//!
//! [`Poseidon2ByteSponge`] implements `spongefish::DuplexSpongeInterface` with
//! `U = u8`, so it slots in *below* [`AkitaTranscript`](crate::AkitaTranscript)
//! exactly where the Blake2b and Keccak sponges do. Nothing above that boundary
//! moves: the domain separator, the instance binding, the length-framed absorb
//! encoding, and the challenge-derivation call sites are untouched.
//!
//! In particular, **absorb labels remain decorative**. `AkitaTranscript`
//! already drops the `Label` argument and never absorbs it; production
//! transcripts are *positional*, and separation comes from the protocol tag,
//! the instance descriptor, and absorb order alone. This backend preserves that
//! behaviour exactly, which is what makes the swap transparent to the protocol.
//! Two consequences worth stating plainly:
//!
//! - Any reordering of absorbs is soundness-relevant and no label mismatch will
//!   catch it, here or on the Blake2b path.
//! - Binding labels into the state would be cheap for an algebraic sponge (one
//!   extra absorbed element per event, or a label digest folded into the
//!   capacity at `START` as SAFE's IO pattern prescribes). That is a genuine
//!   improvement, but it is a **wire-format change**: it would make this
//!   backend produce different challenges than Blake2b for the same protocol
//!   run. It is deliberately *not* done here. Recommend it for a future wire
//!   revision, applied to all backends at once.
//!
//! # Choosing alpha: exact operation counts
//!
//! Both generated instances are formula-legitimate. The counts below are exact
//! for this implementation (not estimates), and they point in opposite
//! directions, which is why the choice needs stating rather than assuming.
//!
//! Per permutation, `t = 12`:
//!
//! | | `alpha = 3` (`R_F = 8`, `R_P = 42`) | `alpha = 7` (`R_F = 8`, `R_P = 22`) |
//! |---|---|---|
//! | S-boxes | 138 | 118 |
//! | mults per S-box | 2 (`x^2 * x`) | 4 (`x^2, x^3, x^6, x^7`) |
//! | S-box mults | 276 | 472 |
//! | internal-layer mults (`WIDTH * R_P`) | 504 | 264 |
//! | **total mults** | **780** | **736** |
//! | external-layer adds (`9 * 68`) | 612 | 612 |
//! | internal-layer adds (`23 * R_P`) | 966 | 506 |
//! | round-constant adds | 138 | 118 |
//! | **total adds** | **1716** | **1236** |
//! | degree-`alpha` Plonkish constraints | 127 | 107 |
//! | in-circuit area x degree | 127 * 3 = **381** | 107 * 7 = **749** |
//!
//! Natively, `alpha = 7` is cheaper on **both** counts, which is not the naive
//! expectation. The reason is the internal linear layer: the generator draws
//! the diagonal `mu` from the Grain stream, so the entries are full-size
//! 64-bit values and each internal round costs `WIDTH = 12` general
//! multiplications. At `R_P = 42` that is 504 multiplications, more than the
//! entire S-box budget, and it swamps the 2-versus-4 multiplication saving per
//! S-box.
//!
//! # Measured, not predicted
//!
//! `tests/poseidon2_bench.rs` times all three sponges in one process. Ratios
//! only: absolute timings in a managed environment run several times slower
//! than bare host.
//!
//! | | ratio |
//! |---|---|
//! | permutation, `alpha = 3` / `alpha = 7` | **1.077x** |
//! | duplex round (absorb 256 B, squeeze 32 B), `alpha = 3` / `alpha = 7` | 1.079x |
//! | full `AkitaTranscript` round, `alpha = 3` / `alpha = 7` | **1.014x** |
//! | full `AkitaTranscript` round, Poseidon2 / Blake2b | **~9.3x** |
//!
//! The counts above predicted a 15-20% permutation penalty for `alpha = 3`;
//! measurement says 7.7%, so the direction was right and the magnitude was
//! overstated (an addition is cheaper relative to a pseudo-Mersenne
//! multiplication than the model assumed). Quote the measured figure. More
//! importantly, the penalty **collapses to 1.4% on a full transcript round**,
//! because byte-encoding overhead dominates the S-box difference at that level.
//!
//! In-circuit, which is the entire reason this backend exists, `alpha = 3`
//! wins by about 2x on area times degree, and by more for a sum-check prover:
//! a degree-`d` constraint needs `d + 1` evaluations per sum-check round, so
//! degree 3 costs 4 evaluations against 8 for degree 7. The `alpha = 7` figure
//! of `118 * 7 = 826` matches the Monolith paper's published area-degree
//! product for Poseidon2 at `t = 12` (eprint 2023/1025, Table 6), which is a
//! useful sanity check on the accounting.
//!
//! **Recommendation: `alpha = 3`**, the default here. A 1.4% penalty on a
//! transcript round is nothing, while the in-circuit factor of about 2 lands
//! directly on the wrapped verifier this backend exists to make feasible.
//! `alpha = 3` is also the designers' smallest-coprime convention for `p*` and
//! the exponent the Ethereum Foundation bounty instances over this exact prime
//! used, so it inherits what little targeted cryptanalysis this prime has.
//!
//! **What this backend costs, stated plainly: about 9x Blake2b per transcript
//! round.** That is a real native regression, and it buys exactly one thing,
//! namely a verifier that can recompute the transcript hash in-circuit over the
//! same field. For non-recursive use there is no upside at all, only the 9x and
//! an unaudited permutation. Never make this the default; do not enable it for
//! any workload that is not paying for the arithmetization.
//!
//! One caveat worth acting on if native transcript cost ever matters: the
//! 504-multiplication internal layer is an artefact of Grain-drawn diagonal
//! entries. Plonky3's Goldilocks instances use power-of-two `mu`, turning
//! those multiplications into shifts. Searching for a small-entry or
//! power-of-two diagonal over `p*` that still passes
//! `check_minpoly_condition` would remove most of the `alpha = 3` native
//! penalty. That is a new parameter search, not a re-parameterisation, and is
//! deliberately not attempted here.
//!
//! [CO25]: https://eprint.iacr.org/2025/536

// `Fp64` provides inherent `from_u64` (reducing) and `square`, which shadow
// `FromPrimitiveInt` and `RingCore`, so those traits are not imported here.
// `CanonicalField` is still required for `to_canonical_u128`.
use akita_field::{CanonicalField, Prime64Offset59};

pub mod challenges;
pub mod duplex;
pub mod generated_alpha3;
pub mod generated_alpha7;
#[cfg(test)]
mod generated_vectors;

pub use challenges::Poseidon2ChallengeReader;
pub use duplex::{Poseidon2ByteSponge, Poseidon2Duplex};
pub use generated_alpha3::Alpha3;
pub use generated_alpha7::Alpha7;

/// Parameter set the `transcript-poseidon2` backend uses.
///
/// `alpha = 3` (`R_F = 8`, `R_P = 42`) by default, the designers'
/// smallest-coprime convention and the exponent the Ethereum Foundation bounty
/// instances over this exact prime used. Enable `transcript-poseidon2-alpha7`
/// to select `alpha = 7` (`R_F = 8`, `R_P = 22`), whose round structure is
/// byte-for-byte the published Goldilocks instance.
#[cfg(not(feature = "transcript-poseidon2-alpha7"))]
pub type SelectedInstance = Alpha3;

/// Parameter set the `transcript-poseidon2` backend uses. See the `alpha = 3`
/// variant of this alias for the selection rule.
#[cfg(feature = "transcript-poseidon2-alpha7")]
pub type SelectedInstance = Alpha7;

/// Base field of the transcript permutation: `p* = 2^64 - 59`.
pub type Fp = Prime64Offset59;

/// The transcript prime `p* = 2^64 - 59`, as an integer.
pub const MODULUS: u64 = 18_446_744_073_709_551_557;

/// Permutation state width in field elements.
pub const WIDTH: usize = 12;

/// Duplex rate in field elements.
pub const RATE: usize = 8;

/// Duplex capacity in field elements (`WIDTH - RATE`); 256 bits = 128-bit level.
pub const CAPACITY: usize = WIDTH - RATE;

const _: () = assert!(RATE + CAPACITY == WIDTH);

/// The fixed Poseidon2 `M_4` block (eprint 2023/323, Section 5.1).
pub const M4: [[u64; 4]; 4] = [[5, 7, 1, 3], [4, 6, 1, 1], [1, 3, 5, 7], [1, 1, 4, 6]];

/// A generated Poseidon2 parameter set over `p*` at `t = 12`.
///
/// Implementors are produced by
/// `crates/akita-transcript/tools/poseidon2_params_p64m59.sage`. Do not
/// hand-write one.
pub trait Poseidon2Instance:
    Copy + Clone + Default + PartialEq + Eq + core::fmt::Debug + Send + Sync + 'static
{
    /// S-box exponent (`gcd(ALPHA, p-1) = 1` is checked by the generator).
    const ALPHA: u64;
    /// Number of full (external) rounds.
    const ROUNDS_F: usize;
    /// Number of partial (internal) rounds.
    const ROUNDS_P: usize;
    /// Flat round constants, `(ROUNDS_F + ROUNDS_P) * WIDTH` canonical values.
    ///
    /// Internal-round slots carry a single nonzero constant at offset 0 and
    /// `WIDTH - 1` zeros, exactly as the generator emits them.
    const ROUND_CONSTANTS_U64: &'static [u64];
    /// Internal matrix diagonal minus one, `WIDTH` canonical values.
    const INTERNAL_DIAG_M1_U64: &'static [u64];
    /// 28-byte ASCII domain tag; packed into the capacity IV verbatim.
    const DOMAIN_TAG: &'static [u8; 28];

    /// Round constants as field elements (cached after first use).
    fn round_constants() -> &'static [Fp];
    /// Internal diagonal minus one, as field elements (cached after first use).
    fn internal_diag_m1() -> &'static [Fp; WIDTH];
    /// Capacity IV: `DOMAIN_TAG` split into 4 little-endian 7-byte chunks.
    fn capacity_iv() -> [Fp; CAPACITY] {
        let tag = Self::DOMAIN_TAG;
        core::array::from_fn(|i| {
            let mut limb = [0u8; 8];
            limb[..7].copy_from_slice(&tag[i * 7..i * 7 + 7]);
            Fp::from_u64(u64::from_le_bytes(limb))
        })
    }
}

/// Declare a generated parameter set. Used only by the generated modules.
#[macro_export]
#[doc(hidden)]
macro_rules! declare_poseidon2_instance {
    (
        $name:ident,
        alpha = $alpha:expr,
        rounds_f = $rf:expr,
        rounds_p = $rp:expr,
        tag = $tag:expr,
        constants = $rc:ident,
        diag = $diag:ident $(,)?
    ) => {
        /// Generated Poseidon2 parameter set. See the module header.
        #[derive(Clone, Copy, Default, PartialEq, Eq, Debug)]
        pub struct $name;

        impl $crate::poseidon2::Poseidon2Instance for $name {
            const ALPHA: u64 = $alpha;
            const ROUNDS_F: usize = $rf;
            const ROUNDS_P: usize = $rp;
            const ROUND_CONSTANTS_U64: &'static [u64] = &$rc;
            const INTERNAL_DIAG_M1_U64: &'static [u64] = &$diag;
            const DOMAIN_TAG: &'static [u8; 28] = $tag;

            fn round_constants() -> &'static [$crate::poseidon2::Fp] {
                static CACHE: std::sync::OnceLock<Vec<$crate::poseidon2::Fp>> =
                    std::sync::OnceLock::new();
                CACHE.get_or_init(|| {
                    $rc.iter()
                        .map(|&v| {
                            <$crate::poseidon2::Fp as akita_field::FromPrimitiveInt>::from_u64(v)
                        })
                        .collect()
                })
            }

            fn internal_diag_m1(
            ) -> &'static [$crate::poseidon2::Fp; $crate::poseidon2::WIDTH] {
                static CACHE: std::sync::OnceLock<
                    [$crate::poseidon2::Fp; $crate::poseidon2::WIDTH],
                > = std::sync::OnceLock::new();
                CACHE.get_or_init(|| {
                    core::array::from_fn(|i| {
                        <$crate::poseidon2::Fp as akita_field::FromPrimitiveInt>::from_u64(
                            $diag[i],
                        )
                    })
                })
            }
        }
    };
}

/// The Poseidon2 permutation for a generated instance.
#[derive(Clone, Copy, Default, PartialEq, Eq, Debug)]
pub struct Poseidon2<I: Poseidon2Instance>(core::marker::PhantomData<fn() -> I>);

impl<I: Poseidon2Instance> Poseidon2<I> {
    /// Number of S-box evaluations per permutation (`t * R_F + R_P`).
    pub const SBOX_COUNT: usize = WIDTH * I::ROUNDS_F + I::ROUNDS_P;

    /// Degree-`ALPHA` constraints in the optimized Plonkish representation
    /// (eprint 2023/323, Section 8.3: `t * R_F + R_P - t + 1`).
    pub const PLONKISH_CONSTRAINTS: usize = WIDTH * I::ROUNDS_F + I::ROUNDS_P - WIDTH + 1;

    /// Apply the permutation in place.
    #[inline]
    pub fn permute(state: &mut [Fp; WIDTH]) {
        let rc = I::round_constants();
        debug_assert_eq!(rc.len(), (I::ROUNDS_F + I::ROUNDS_P) * WIDTH);
        let half_f = I::ROUNDS_F / 2;

        // Poseidon2 prepends one external linear layer.
        external_linear_layer(state);

        for round in 0..half_f {
            Self::full_round(state, &rc[round * WIDTH..(round + 1) * WIDTH]);
        }
        for round in 0..I::ROUNDS_P {
            Self::partial_round(state, rc[(half_f + round) * WIDTH]);
        }
        for round in 0..half_f {
            let idx = half_f + I::ROUNDS_P + round;
            Self::full_round(state, &rc[idx * WIDTH..(idx + 1) * WIDTH]);
        }
    }

    /// Permute a fresh copy of `state`.
    #[inline]
    pub fn permuted(state: &[Fp; WIDTH]) -> [Fp; WIDTH] {
        let mut out = *state;
        Self::permute(&mut out);
        out
    }

    #[inline]
    fn full_round(state: &mut [Fp; WIDTH], rc: &[Fp]) {
        for (slot, c) in state.iter_mut().zip(rc.iter()) {
            *slot += *c;
        }
        for slot in state.iter_mut() {
            *slot = Self::sbox(*slot);
        }
        external_linear_layer(state);
    }

    #[inline]
    fn partial_round(state: &mut [Fp; WIDTH], rc0: Fp) {
        state[0] += rc0;
        state[0] = Self::sbox(state[0]);
        internal_linear_layer(state, I::internal_diag_m1());
    }

    /// `x^ALPHA`, minimal addition chain for the supported exponents.
    #[inline]
    fn sbox(x: Fp) -> Fp {
        match I::ALPHA {
            // x^3: 2 multiplications.
            3 => x.square() * x,
            // x^7: 4 multiplications.
            7 => {
                let x2 = x.square();
                let x3 = x2 * x;
                x3.square() * x
            }
            other => unreachable!("unsupported Poseidon2 S-box exponent {other}"),
        }
    }
}

/// External (full-round) linear layer: `circ(2*M4, M4, M4)` for `t = 12`.
///
/// Applies `M4` to each 4-element block, then adds the block-wise sum, which is
/// exactly `2*M4` on the diagonal blocks and `M4` elsewhere.
#[inline]
pub fn external_linear_layer(state: &mut [Fp; WIDTH]) {
    for block in state.chunks_exact_mut(4) {
        apply_m4(block);
    }
    let mut sum = [Fp::default(); 4];
    for block in state.chunks_exact(4) {
        for (slot, value) in sum.iter_mut().zip(block.iter()) {
            *slot += *value;
        }
    }
    for block in state.chunks_exact_mut(4) {
        for (slot, value) in block.iter_mut().zip(sum.iter()) {
            *slot += *value;
        }
    }
}

/// Multiply a 4-element block by `M4 = [[5,7,1,3],[4,6,1,1],[1,3,5,7],[1,1,4,6]]`.
///
/// Uses the standard 8-addition circuit; `verify_m4_circuit` in the tests below
/// checks it against the literal matrix product.
#[inline]
fn apply_m4(x: &mut [Fp]) {
    let t0 = x[0] + x[1];
    let t1 = x[2] + x[3];
    let t2 = x[1] + x[1] + t1;
    let t3 = x[3] + x[3] + t0;
    let t4 = t1 + t1 + t1 + t1 + t3;
    let t5 = t0 + t0 + t0 + t0 + t2;
    x[0] = t3 + t5;
    x[1] = t5;
    x[2] = t2 + t4;
    x[3] = t4;
}

/// Internal (partial-round) linear layer: `M_I = 1 1^T + diag(mu - 1)`.
///
/// `(M_I x)_i = sum_j x_j + (mu_i - 1) x_i`.
#[inline]
pub fn internal_linear_layer(state: &mut [Fp; WIDTH], diag_m1: &[Fp; WIDTH]) {
    let mut sum = Fp::default();
    for slot in state.iter() {
        sum += *slot;
    }
    for (slot, d) in state.iter_mut().zip(diag_m1.iter()) {
        *slot = sum + *slot * *d;
    }
}

/// Canonical `u64` of a field element.
#[inline]
pub(crate) fn to_u64(x: Fp) -> u64 {
    let v = x.to_canonical_u128();
    debug_assert!(v < u128::from(MODULUS));
    v as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    fn m4_reference(x: &[Fp; 4]) -> [Fp; 4] {
        core::array::from_fn(|i| {
            (0..4)
                .map(|j| Fp::from_u64(M4[i][j]) * x[j])
                .fold(Fp::default(), |a, b| a + b)
        })
    }

    #[test]
    fn m4_circuit_matches_matrix() {
        for seed in 0..64u64 {
            let x: [Fp; 4] = core::array::from_fn(|i| {
                Fp::from_u64(seed.wrapping_mul(0x9e37_79b9_7f4a_7c15).wrapping_add(i as u64))
            });
            let mut got = x;
            apply_m4(&mut got);
            assert_eq!(got, m4_reference(&x), "M4 circuit mismatch at seed {seed}");
        }
    }

    #[test]
    fn external_layer_matches_block_matrix() {
        // Literal circ(2*M4, M4, M4) applied naively.
        let m: [[Fp; WIDTH]; WIDTH] = core::array::from_fn(|i| {
            core::array::from_fn(|j| {
                let (bi, bj) = (i / 4, j / 4);
                let base = Fp::from_u64(M4[i % 4][j % 4]);
                if bi == bj {
                    base + base
                } else {
                    base
                }
            })
        });
        for seed in 0..16u64 {
            let x: [Fp; WIDTH] = core::array::from_fn(|i| {
                Fp::from_u64(seed.wrapping_mul(0xff51_afd7_ed55_8ccd).wrapping_add(i as u64))
            });
            let want: [Fp; WIDTH] = core::array::from_fn(|i| {
                (0..WIDTH)
                    .map(|j| m[i][j] * x[j])
                    .fold(Fp::default(), |a, b| a + b)
            });
            let mut got = x;
            external_linear_layer(&mut got);
            assert_eq!(got, want, "external layer mismatch at seed {seed}");
        }
    }

    #[test]
    fn internal_layer_matches_matrix() {
        let diag = Alpha3::internal_diag_m1();
        let m: [[Fp; WIDTH]; WIDTH] = core::array::from_fn(|i| {
            core::array::from_fn(|j| {
                if i == j {
                    Fp::from_u64(1) + diag[i]
                } else {
                    Fp::from_u64(1)
                }
            })
        });
        for seed in 0..16u64 {
            let x: [Fp; WIDTH] = core::array::from_fn(|i| {
                Fp::from_u64(seed.wrapping_mul(0xc2b2_ae3d_27d4_eb4f).wrapping_add(i as u64))
            });
            let want: [Fp; WIDTH] = core::array::from_fn(|i| {
                (0..WIDTH)
                    .map(|j| m[i][j] * x[j])
                    .fold(Fp::default(), |a, b| a + b)
            });
            let mut got = x;
            internal_linear_layer(&mut got, diag);
            assert_eq!(got, want, "internal layer mismatch at seed {seed}");
        }
    }

    #[test]
    fn sbox_matches_exponentiation() {
        for v in 0..97u64 {
            let x = Fp::from_u64(v.wrapping_mul(0x0123_4567_89ab_cdef));
            let x3 = x * x * x;
            assert_eq!(Poseidon2::<Alpha3>::sbox(x), x3);
            let x7 = x3 * x3 * x;
            assert_eq!(Poseidon2::<Alpha7>::sbox(x), x7);
        }
    }

    #[test]
    fn capacity_iv_packs_the_domain_tag() {
        let iv = Alpha3::capacity_iv();
        let mut bytes = Vec::new();
        for e in iv {
            bytes.extend_from_slice(&to_u64(e).to_le_bytes()[..7]);
        }
        assert_eq!(&bytes[..], &Alpha3::DOMAIN_TAG[..]);
        assert_ne!(Alpha3::capacity_iv(), Alpha7::capacity_iv());
    }

    #[test]
    fn instance_shapes_are_consistent() {
        assert_eq!(
            Alpha3::ROUND_CONSTANTS_U64.len(),
            (Alpha3::ROUNDS_F + Alpha3::ROUNDS_P) * WIDTH
        );
        assert_eq!(
            Alpha7::ROUND_CONSTANTS_U64.len(),
            (Alpha7::ROUNDS_F + Alpha7::ROUNDS_P) * WIDTH
        );
        assert_eq!(Alpha3::INTERNAL_DIAG_M1_U64.len(), WIDTH);
        assert_eq!(Alpha7::INTERNAL_DIAG_M1_U64.len(), WIDTH);
        for &c in Alpha3::ROUND_CONSTANTS_U64 {
            assert!(c < MODULUS);
        }
        for &c in Alpha7::ROUND_CONSTANTS_U64 {
            assert!(c < MODULUS);
        }
        // Internal-round constant slots are (c, 0, ..., 0).
        for inst_rc in [
            (
                Alpha3::ROUND_CONSTANTS_U64,
                Alpha3::ROUNDS_F,
                Alpha3::ROUNDS_P,
            ),
            (
                Alpha7::ROUND_CONSTANTS_U64,
                Alpha7::ROUNDS_F,
                Alpha7::ROUNDS_P,
            ),
        ] {
            let (rc, rf, rp) = inst_rc;
            for round in rf / 2..rf / 2 + rp {
                for i in 1..WIDTH {
                    assert_eq!(rc[round * WIDTH + i], 0);
                }
            }
        }
    }

    #[test]
    fn permutation_is_a_bijection_on_sampled_points() {
        // Distinct inputs must give distinct outputs (necessary condition).
        let mut seen = std::collections::HashSet::new();
        for v in 0..256u64 {
            let mut s: [Fp; WIDTH] = core::array::from_fn(|i| Fp::from_u64(v + i as u64));
            Poseidon2::<Alpha3>::permute(&mut s);
            assert!(seen.insert(s.map(to_u64)), "collision at {v}");
        }
    }

    /// Cross-check against the designers' own sage `poseidon2()` reference,
    /// independently re-derived by `tools/poseidon2_sponge_reference.py`.
    #[test]
    fn permutation_matches_generated_kats() {
        for (input, expected) in generated_alpha3::PERMUTATION_KATS {
            let mut state: [Fp; WIDTH] = core::array::from_fn(|i| Fp::from_u64(input[i]));
            Poseidon2::<Alpha3>::permute(&mut state);
            assert_eq!(state.map(to_u64), expected, "alpha=3 KAT mismatch");
        }
        for (input, expected) in generated_alpha7::PERMUTATION_KATS {
            let mut state: [Fp; WIDTH] = core::array::from_fn(|i| Fp::from_u64(input[i]));
            Poseidon2::<Alpha7>::permute(&mut state);
            assert_eq!(state.map(to_u64), expected, "alpha=7 KAT mismatch");
        }
    }

    /// The fast linear layers must agree with the literal matrices the
    /// generator emitted, not just with a matrix rebuilt from the same
    /// assumptions.
    #[test]
    fn fast_layers_match_generated_matrices() {
        fn naive(m: &[[u64; WIDTH]; WIDTH], x: &[Fp; WIDTH]) -> [Fp; WIDTH] {
            core::array::from_fn(|i| {
                (0..WIDTH)
                    .map(|j| Fp::from_u64(m[i][j]) * x[j])
                    .fold(Fp::default(), |a, b| a + b)
            })
        }
        for seed in 0..8u64 {
            let x: [Fp; WIDTH] = core::array::from_fn(|i| {
                Fp::from_u64(seed.wrapping_mul(0x9e37_79b9_7f4a_7c15).wrapping_add(i as u64))
            });

            let mut got = x;
            external_linear_layer(&mut got);
            assert_eq!(got, naive(&generated_alpha3::EXTERNAL_MATRIX, &x));
            assert_eq!(
                generated_alpha3::EXTERNAL_MATRIX,
                generated_alpha7::EXTERNAL_MATRIX,
                "the external matrix is alpha-independent"
            );

            let mut got = x;
            internal_linear_layer(&mut got, Alpha3::internal_diag_m1());
            assert_eq!(got, naive(&generated_alpha3::INTERNAL_MATRIX, &x));

            let mut got = x;
            internal_linear_layer(&mut got, Alpha7::internal_diag_m1());
            assert_eq!(got, naive(&generated_alpha7::INTERNAL_MATRIX, &x));
        }
    }

    /// The generated internal matrix must be exactly `1*1^T + diag(mu - 1)`.
    #[test]
    fn generated_internal_matrix_has_the_poseidon2_form() {
        for (m, diag) in [
            (
                &generated_alpha3::INTERNAL_MATRIX,
                Alpha3::INTERNAL_DIAG_M1_U64,
            ),
            (
                &generated_alpha7::INTERNAL_MATRIX,
                Alpha7::INTERNAL_DIAG_M1_U64,
            ),
        ] {
            for i in 0..WIDTH {
                for j in 0..WIDTH {
                    if i == j {
                        assert_eq!(m[i][i], (diag[i] + 1) % MODULUS);
                    } else {
                        assert_eq!(m[i][j], 1);
                    }
                }
            }
        }
    }

    /// `M4` is MDS over `p*`: every square submatrix is nonsingular.
    #[test]
    fn m4_is_mds() {
        assert!(is_mds(&M4.map(|r| r.map(u128::from)), 4));
    }

    /// The `t = 12` external matrix is invertible but **not** MDS, by design.
    ///
    /// eprint 2023/323 Section 5 states plainly that Poseidon2 uses "non-MDS
    /// matrices for the external and internal rounds"; the "This matrix is MDS
    /// for all primes we consider" sentence in Section 5.1 is about the `4 x 4`
    /// block `M4`, not the `t x t` block matrix. What the paper claims for
    /// `M_E` is a branch number `b = t/4 + 4` (Section 7.1, citing Griffin,
    /// eprint 2022/403, Proposition 1), so `b = 7` here against `13` for MDS.
    ///
    /// The obstruction is structural and holds over every field: within a block
    /// row, column `j` is `2*M4[:, j]` while column `j + 4` is `M4[:, j]`, so
    /// the `2 x 2` minor on rows `{0, 1}`, columns `{0, 4}` is
    /// `det [[10, 5], [8, 4]] = 0`. This test pins the fact so a future
    /// parameter change cannot silently alter it.
    #[test]
    fn external_matrix_is_invertible_but_not_mds() {
        let modulus = u128::from(MODULUS);
        let m: [[u128; WIDTH]; WIDTH] =
            generated_alpha3::EXTERNAL_MATRIX.map(|r| r.map(u128::from));
        let all: Vec<usize> = (0..WIDTH).collect();

        assert!(
            !singular(&m, &all, &all, modulus),
            "external matrix must be invertible"
        );
        assert!(
            singular(&m, &[0, 1], &[0, 4], modulus),
            "expected the documented singular 2x2 minor"
        );
        assert!(!is_mds(&m, WIDTH), "external matrix is not MDS for t = 12");

        // The obstruction starts at size 2: every entry is nonzero.
        for row in m {
            for entry in row {
                assert_ne!(entry % modulus, 0);
            }
        }
    }

    /// Upper-bound half of the designers' branch-number claim for `M_E`.
    ///
    /// eprint 2023/323 Section 7.1 claims `b = t/4 + 4 = 7` at `t = 12`, citing
    /// Griffin Proposition 1. Writing `M_E = M' M''` with
    /// `M' = diag(M4, M4, M4)` and `M'' = circ(2I, I, I)`, the vector
    /// `x = (-s, -s, 3s)` has `M'' x = (0, 0, 4s)`, so `M_E x = (0, 0, 4 M4 s)`.
    /// Taking `s = e_0` gives `wt(x) = 3` and `wt(M_E x) = wt(M4 e_0) = 4`, so
    /// `wt(x) + wt(M_E x) = 7` and hence `b <= 7`, matching the claim exactly.
    /// The matching lower bound `b >= 7` is imported from Griffin, not
    /// re-proved here.
    #[test]
    fn external_matrix_branch_number_witness_matches_the_claimed_value() {
        let mut x = [Fp::default(); WIDTH];
        x[0] = -Fp::from_u64(1);
        x[4] = -Fp::from_u64(1);
        x[8] = Fp::from_u64(3);

        let mut image = x;
        external_linear_layer(&mut image);

        let weight = |v: &[Fp; WIDTH]| v.iter().filter(|e| to_u64(**e) != 0).count();
        assert_eq!(weight(&x), 3);
        assert_eq!(weight(&image), 4, "expected the witness image to be 4 M4 e_0");
        assert_eq!(
            weight(&x) + weight(&image),
            7,
            "branch number witness must match the claimed b = t/4 + 4 = 7"
        );
        // The nonzero image entries are exactly the last block.
        for (i, e) in image.iter().enumerate() {
            assert_eq!(to_u64(*e) != 0, i >= 8, "witness image support at index {i}");
        }
    }

    /// Nonsingularity of every `k x k` submatrix, by Gaussian elimination
    /// modulo `p*`.
    fn is_mds<const N: usize>(m: &[[u128; N]; N], n: usize) -> bool {
        let modulus = u128::from(MODULUS);
        let mut rows = vec![0usize; n];
        let mut cols = vec![0usize; n];
        for k in 1..=n {
            for i in 0..k {
                rows[i] = i;
                cols[i] = i;
            }
            loop {
                loop {
                    if singular(m, &rows[..k], &cols[..k], modulus) {
                        return false;
                    }
                    if !next_combination(&mut cols[..k], n) {
                        break;
                    }
                }
                for (i, slot) in cols.iter_mut().enumerate().take(k) {
                    *slot = i;
                }
                if !next_combination(&mut rows[..k], n) {
                    break;
                }
            }
        }
        true
    }

    fn next_combination(idx: &mut [usize], n: usize) -> bool {
        let k = idx.len();
        let mut i = k;
        while i > 0 {
            i -= 1;
            if idx[i] != i + n - k {
                idx[i] += 1;
                for j in i + 1..k {
                    idx[j] = idx[j - 1] + 1;
                }
                return true;
            }
        }
        false
    }

    fn singular<const N: usize>(
        m: &[[u128; N]; N],
        rows: &[usize],
        cols: &[usize],
        modulus: u128,
    ) -> bool {
        let k = rows.len();
        let mut a: Vec<Vec<u128>> = rows
            .iter()
            .map(|&r| cols.iter().map(|&c| m[r][c] % modulus).collect())
            .collect();
        for col in 0..k {
            let Some(pivot) = (col..k).find(|&r| a[r][col] != 0) else {
                return true;
            };
            a.swap(col, pivot);
            let inv = mod_inverse(a[col][col], modulus);
            for row in col + 1..k {
                if a[row][col] == 0 {
                    continue;
                }
                let factor = a[row][col] * inv % modulus;
                // `col < row`, so split to borrow the pivot row and the target
                // row disjointly and eliminate by zipping their tails.
                let (upper, lower) = a.split_at_mut(row);
                let pivot_row = &upper[col];
                let target_row = &mut lower[0];
                for (target, pivot) in target_row[col..k].iter_mut().zip(&pivot_row[col..k]) {
                    let sub = factor * *pivot % modulus;
                    *target = (*target + modulus - sub) % modulus;
                }
            }
        }
        false
    }

    fn mod_inverse(a: u128, modulus: u128) -> u128 {
        // p* is prime, so a^(p-2) is the inverse.
        let mut result = 1u128;
        let mut base = a % modulus;
        let mut exp = modulus - 2;
        while exp > 0 {
            if exp & 1 == 1 {
                result = result * base % modulus;
            }
            base = base * base % modulus;
            exp >>= 1;
        }
        result
    }

    /// Whole-stack cross-check: permutation, duplex, and the (A1)/(S1)/(S2)
    /// encodings against `tools/poseidon2_sponge_reference.py`.
    #[test]
    fn sponge_matches_generated_vectors() {
        use crate::poseidon2::generated_vectors::{SpongeOp, SPONGE_KATS_ALPHA3, SPONGE_KATS_ALPHA7};
        use spongefish::DuplexSpongeInterface;

        fn run<I: Poseidon2Instance>(kats: &[crate::poseidon2::generated_vectors::SpongeKat]) {
            for kat in kats {
                let mut sponge = Poseidon2ByteSponge::<I>::new();
                for (step, op) in kat.ops.iter().enumerate() {
                    match op {
                        SpongeOp::Absorb(bytes) => {
                            sponge.absorb(bytes);
                        }
                        SpongeOp::SqueezeBytes(expected) => {
                            let mut got = vec![0u8; expected.len()];
                            sponge.squeeze(&mut got);
                            assert_eq!(
                                &got[..],
                                *expected,
                                "{} step {step}: byte squeeze mismatch",
                                kat.name
                            );
                        }
                        SpongeOp::SqueezeField(expected) => {
                            let mut got = vec![Fp::default(); expected.len()];
                            sponge.squeeze_field_elements(&mut got);
                            let got: Vec<u64> = got.into_iter().map(to_u64).collect();
                            assert_eq!(
                                &got[..],
                                *expected,
                                "{} step {step}: field squeeze mismatch",
                                kat.name
                            );
                        }
                        SpongeOp::Ratchet => {
                            sponge.ratchet();
                        }
                    }
                }
            }
        }

        run::<Alpha3>(SPONGE_KATS_ALPHA3);
        run::<Alpha7>(SPONGE_KATS_ALPHA7);
    }

    #[test]
    fn sbox_and_constraint_counts() {
        assert_eq!(Poseidon2::<Alpha3>::SBOX_COUNT, 138);
        assert_eq!(Poseidon2::<Alpha7>::SBOX_COUNT, 118);
        assert_eq!(Poseidon2::<Alpha3>::PLONKISH_CONSTRAINTS, 127);
        assert_eq!(Poseidon2::<Alpha7>::PLONKISH_CONSTRAINTS, 107);
    }
}
