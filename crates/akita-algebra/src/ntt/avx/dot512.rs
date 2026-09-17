//! Six-product bounded Montgomery dot accumulation in 512-bit vectors.

#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use super::montgomery::{mont_reduce_i32_products_avx512, reduce_range_16x_i32_avx512};
use crate::ntt::prime::I32_LAZY_DOT_BATCH;

/// # Safety
///
/// Requires AVX2 and AVX-512F/DQ/BW. The first `1 <= COUNT <= 6` pointers in each
/// input must cover `d` readable elements and `acc` must cover `d` writable
/// elements, with no input aliasing `acc`. Every residue has magnitude below
/// `p < 2^30`, and `pinv` is the Montgomery inverse of `p`.
#[inline]
#[target_feature(enable = "avx512f,avx512dq,avx512bw,avx2")]
pub(super) unsafe fn pointwise_dot_acc_i32_count<const COUNT: usize>(
    acc: *mut i32,
    lhs: *const *const i32,
    rhs: *const *const i32,
    d: usize,
    p: i32,
    pinv: i32,
) {
    let p_v = _mm512_set1_epi32(p);
    let pinv_v = _mm512_set1_epi32(pinv);
    let mut i = 0;
    while i + 16 <= d {
        let mut even_sum = _mm512_setzero_si512();
        let mut odd_sum = _mm512_setzero_si512();
        for product in 0..COUNT {
            // SAFETY: each input pointer covers d elements; i + 16 <= d.
            unsafe {
                let l = _mm512_loadu_si512((*lhs.add(product)).add(i).cast());
                let r = _mm512_loadu_si512((*rhs.add(product)).add(i).cast());
                even_sum = _mm512_add_epi64(even_sum, _mm512_mul_epi32(l, r));
                odd_sum = _mm512_add_epi64(
                    odd_sum,
                    _mm512_mul_epi32(_mm512_srli_epi64::<32>(l), _mm512_srli_epi64::<32>(r)),
                );
            }
        }
        // SAFETY: COUNT <= 6 and p < 2^30 bound the signed reduction numerator
        // by 6*2^60 + 2^61 < 2^63. acc covers the complete vector.
        unsafe {
            let even = mont_reduce_i32_products_avx512(even_sum, p_v, pinv_v);
            let odd = mont_reduce_i32_products_avx512(odd_sum, p_v, pinv_v);
            let batch = _mm512_or_si512(even, _mm512_slli_epi64::<32>(odd));
            let batch = reduce_range_16x_i32_avx512(batch, p_v);
            let accumulator = _mm512_loadu_si512(acc.add(i).cast());
            _mm512_storeu_si512(
                acc.add(i).cast(),
                reduce_range_16x_i32_avx512(_mm512_add_epi32(accumulator, batch), p_v),
            );
        }
        i += 16;
    }
    if i < d {
        let mut lhs_tail = [std::ptr::null(); I32_LAZY_DOT_BATCH];
        let mut rhs_tail = [std::ptr::null(); I32_LAZY_DOT_BATCH];
        for product in 0..COUNT {
            // SAFETY: i < d and each of the first COUNT pointers covers d values.
            unsafe {
                lhs_tail[product] = (*lhs.add(product)).add(i);
                rhs_tail[product] = (*rhs.add(product)).add(i);
            }
        }
        // SAFETY: AVX2 is enabled, the adjusted pointers cover d-i values, and
        // the existing kernel reads only the first COUNT pointers.
        unsafe {
            super::pointwise::pointwise_dot_acc_i32(
                acc.add(i),
                lhs_tail.as_ptr(),
                rhs_tail.as_ptr(),
                COUNT,
                d - i,
                p,
                pinv,
                false,
            );
        }
    }
}
