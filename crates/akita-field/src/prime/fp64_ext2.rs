//! Scalar quadratic multiplication for the full-width offset-59 field.

use super::Fp64;
use crate::ext::{fp_ext2_mul_generic, FpExt2Config, FpExt2NonResidueKind};
use crate::FieldCore;

const OFFSET_59_PRIME: u64 = u64::MAX - 58;

/// Reduce `low + carry * 2^128`, where the caller establishes `carry <= 2`.
/// With `2^64 = 59 (mod p)`, the first fold is at most
/// `60 * 2^64 + 6902`; the second is below `2^64 + 3540 < 2p`.
/// Thus one conditional subtraction gives a canonical 64-bit residue.
#[inline(always)]
fn reduce130_offset59(low: u128, carry: u64) -> u64 {
    let folded = (low >> 64) * 59 + u128::from(low as u64) + u128::from(carry) * 3481;
    let settled = (folded >> 64) * 59 + u128::from(folded as u64);
    let prime = u128::from(OFFSET_59_PRIME);
    if settled >= prime {
        (settled - prime) as u64
    } else {
        settled as u64
    }
}

impl<const P: u64> FieldCore for Fp64<P> {
    #[inline(always)]
    fn fp_ext2_mul<C>(a0: Self, a1: Self, b0: Self, b1: Self) -> (Self, Self)
    where
        C: FpExt2Config<Self>,
    {
        if P != OFFSET_59_PRIME || C::NON_RESIDUE_KIND != FpExt2NonResidueKind::Two {
            return fp_ext2_mul_generic::<Self, C>(a0, a1, b0, b1);
        }

        let ac = u128::from(a0.0) * u128::from(b0.0);
        let bd = u128::from(a1.0) * u128::from(b1.0);
        let ad = u128::from(a0.0) * u128::from(b1.0);
        let bc = u128::from(a1.0) * u128::from(b0.0);
        let (twice_bd, carry0) = bd.overflowing_add(bd);
        let (real, carry1) = ac.overflowing_add(twice_bd);
        let (imaginary, carry2) = ad.overflowing_add(bc);
        // Each raw product is below 2^128. These carry counts represent
        // the exact sums ac + 2bd and ad + bc, including both real carries.
        (
            Self(reduce130_offset59(
                real,
                u64::from(carry0) + u64::from(carry1),
            )),
            Self(reduce130_offset59(imaginary, u64::from(carry2))),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ext::{FpExt2, NegOneNr, TwoNr};
    use crate::Fp32;

    type F = Fp64<OFFSET_59_PRIME>;

    struct GenericTwo;
    impl FpExt2Config<F> for GenericTwo {
        fn non_residue() -> F {
            F::from_u64(2)
        }
    }

    fn compare<F, C>(a: F, b: F, c: F, d: F)
    where
        F: FieldCore + std::fmt::Debug,
        C: FpExt2Config<F>,
    {
        // The pre-specialization Karatsuba calculation is the reference.
        let ac = a * c;
        let bd = b * d;
        let cross = (a + b) * (c + d);
        let expected =
            FpExt2::<F, C>::new(ac + C::mul_non_residue(bd, |base| base), cross - ac - bd);
        assert_eq!(FpExt2::<F, C>::new(a, b) * FpExt2::new(c, d), expected);
    }

    #[test]
    fn scalar_extension_hook_matches_boundary_tuples_and_full_field() {
        let boundary = [
            0,
            1,
            2,
            58,
            59,
            OFFSET_59_PRIME / 2,
            OFFSET_59_PRIME - 3,
            OFFSET_59_PRIME - 2,
            OFFSET_59_PRIME - 1,
        ];
        for a in boundary {
            for b in boundary {
                for c in boundary {
                    for d in boundary {
                        compare::<F, TwoNr>(
                            F::from_u64(a),
                            F::from_u64(b),
                            F::from_u64(c),
                            F::from_u64(d),
                        );
                    }
                }
            }
        }
        let mut state = 0x243f_6a88_85a3_08d3_u64;
        for _ in 0..100_000 {
            let mut next = || {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                F::from_u64(state)
            };
            let [a, b, c, d] = core::array::from_fn(|_| next());
            compare::<F, TwoNr>(a, b, c, d);
            compare::<F, TwoNr>(a, b, a, b);
            compare::<F, GenericTwo>(a, b, c, d);
        }
    }

    #[test]
    fn scalar_extension_hook_preserves_other_fields_and_non_residues() {
        type F61 = Fp64<{ (1_u64 << 61) - 1 }>;
        type F48 = Fp64<{ (1_u64 << 48) - 59 }>;
        type F31 = Fp32<{ (1_u32 << 31) - 1 }>;
        for i in 0..10_000_u64 {
            let raw: [u64; 4] =
                core::array::from_fn(|j| (4 * i + j as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15));
            let [a, b, c, d] = raw.map(F61::from_u64);
            compare::<F61, NegOneNr>(a, b, c, d);
            let [a, b, c, d] = raw.map(F48::from_u64);
            compare::<F48, TwoNr>(a, b, c, d);
            let [a, b, c, d] = raw.map(F31::from_u64);
            compare::<F31, NegOneNr>(a, b, c, d);
        }
    }
}
