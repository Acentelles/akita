//! Host kernel selection for one prepared CRT+NTT parameter set.

use super::prime::PrimeWidth;

/// Host kernels selected once when a CRT+NTT parameter set is prepared.
///
/// AVX2 is the measured x86 production backend for both transforms and
/// pointwise arithmetic. AVX-512 i32 kernels require explicit opt-in and all
/// required CPU features; i16 keeps its AVX2 implementation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NttKernelPlan(NttKernelKind);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[allow(dead_code)] // Variants are selected on different target architectures.
enum NttKernelKind {
    /// Portable scalar kernels.
    Scalar,
    /// AArch64 NEON transforms and pointwise arithmetic.
    Neon,
    /// AVX2 transforms and pointwise arithmetic.
    Avx2,
    /// AVX-512 i32 transforms and pointwise arithmetic, with AVX2 tails.
    Avx512,
}

impl NttKernelPlan {
    pub(crate) const SCALAR: Self = Self(NttKernelKind::Scalar);

    /// Detect the best enabled host plan for residue width `W`.
    pub fn detect<W: PrimeWidth>() -> Self {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if matches!(core::mem::size_of::<W>(), 2 | 4) {
                match super::avx::avx_ntt_mode() {
                    Some(super::avx::AvxNttMode::Avx512) if core::mem::size_of::<W>() == 4 => {
                        return Self(NttKernelKind::Avx512);
                    }
                    Some(_) => return Self(NttKernelKind::Avx2),
                    None => {}
                }
            }
        }

        #[cfg(target_arch = "aarch64")]
        if super::neon::use_neon_ntt() {
            return Self(NttKernelKind::Neon);
        }

        Self::SCALAR
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    pub(crate) const fn uses_x86_transform(self) -> bool {
        matches!(self.0, NttKernelKind::Avx2 | NttKernelKind::Avx512)
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    pub(crate) const fn uses_avx512_i32(self) -> bool {
        matches!(self.0, NttKernelKind::Avx512)
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    pub(crate) const fn x86_pointwise_mode(self) -> Option<super::avx::AvxNttMode> {
        match self.0 {
            NttKernelKind::Avx2 => Some(super::avx::AvxNttMode::Avx2),
            NttKernelKind::Avx512 => Some(super::avx::AvxNttMode::Avx512),
            NttKernelKind::Scalar | NttKernelKind::Neon => None,
        }
    }

    #[cfg(target_arch = "aarch64")]
    pub(crate) const fn uses_neon(self) -> bool {
        matches!(self.0, NttKernelKind::Neon)
    }
}
