//! Challenge derivation from the algebraic sponge.
//!
//! # UNAUDITED RESEARCH CRYPTOGRAPHY - NOT FOR PRODUCTION
//!
//! See the [module header](super) for the standing risk statement.
//!
//! # What changes and what does not
//!
//! Akita derives three families of challenge:
//!
//! 1. **Uniform base-field elements** (`Transcript::challenge_scalar`).
//! 2. **Extension-field elements** (`akita_transcript::sample_ext_challenge`),
//!    assembled from `EXT_DEGREE` base-field draws.
//! 3. **Sparse ring challenges** (`akita_challenges::sample_sparse_challenges`):
//!    a fixed-weight signed-sparse family, `count_pm1` coefficients at `+-1`
//!    and `count_pm2` at `+-2`, drawn by partial Fisher-Yates over the ring
//!    dimension.
//!
//! Families 1 and 2 need **no port at all**. They route through
//! `AkitaTranscript::squeeze_bytes`, and encoding rule (S1) makes the squeezed
//! bytes *exactly* uniform, so `F::from_challenge_bytes` sees the same input
//! distribution it sees under Blake2b and produces the identical challenge
//! distribution. Nothing about family 1 or 2 is approximated here.
//!
//! Family 3 is a two-stage construction: the transcript supplies a 32-byte
//! seed, and a SHAKE256 XOF expands that seed into the Fisher-Yates draws.
//! Swapping the transcript backend moves the *seed* onto the algebraic sponge
//! automatically (the seed comes from `challenge_bytes`), so the sampled
//! distribution is unchanged and the binding is algebraic. What stays on
//! SHAKE256 is the *expansion*. See [`Poseidon2ChallengeReader`] for the
//! drop-in that removes the last SHAKE dependency, and the note below for why
//! it is not wired up here.
//!
//! # Why the expansion is not switched over in this change
//!
//! `akita_challenges::sampler::xof::XofCursor` is private to `akita-challenges`
//! and constructed inside `sample_sparse_challenges`. Replacing it is a
//! mechanical change in that crate, not in this one:
//!
//! ```text
//! -let seed = transcript.challenge_bytes(CHALLENGE_SPARSE_CHALLENGE, 32);
//! -let mut cursor = XofCursor::from_seed(&seed);
//! +let mut cursor = <the transcript's own byte reader>;
//! ```
//!
//! [`Poseidon2ChallengeReader`] is the byte reader that slots in there. It
//! reproduces `XofCursor`'s drawing primitives bit for bit (same bitmask
//! rejection, same read widths), so the sampled sparse challenges keep the
//! exact same distribution; only the source of uniform bytes changes.
//!
//! # Distribution argument
//!
//! `XofCursor::next_usize_mod(m)` and [`Poseidon2ChallengeReader::next_usize_mod`]
//! are the same algorithm: read `ceil(log2(m))` bits from a stream of uniform
//! independent bits, accept if the value is `< m`, else retry. Conditioned on
//! acceptance the result is uniform on `[0, m)` whatever the stream, provided
//! the stream is uniform. SHAKE256 output is uniform under the standard XOF
//! assumption; sponge output under rule (S1) is *exactly* uniform on
//! `[0, 2^56)` per element, hence uniform per byte, under the ideal-permutation
//! model for the Poseidon2 permutation. So both readers induce the identical
//! sparse-challenge distribution, and the weight and norm parameters
//! (`count_pm1`, `count_pm2`, the resulting `l1` and `linf` norms) are
//! untouched because the sampler above them is untouched.
//!
//! The one thing that is *not* preserved is the concrete challenge value for a
//! given transcript, which is the point of changing the hash.

use super::{Fp, Poseidon2ByteSponge, Poseidon2Instance};

/// Buffered byte reader over a Poseidon2 duplex, with the same drawing
/// primitives as `akita_challenges`' SHAKE256-backed `XofCursor`.
///
/// This is the drop-in that would remove SHAKE256 from the sparse
/// ring-challenge path. It is provided and tested here; wiring it into
/// `akita_challenges::sample_sparse_challenges` is a change in that crate.
pub struct Poseidon2ChallengeReader<I: Poseidon2Instance> {
    sponge: Poseidon2ByteSponge<I>,
    buf: Vec<u8>,
    pos: usize,
}

/// Internal buffer size, matching `XofCursor`'s 4 KB amortisation window.
const BUF_SIZE: usize = 4096;

impl<I: Poseidon2Instance> Poseidon2ChallengeReader<I> {
    /// Build a reader that draws from `sponge` by rule (S1).
    #[must_use]
    pub fn new(sponge: Poseidon2ByteSponge<I>) -> Self {
        let mut reader = Self {
            sponge,
            buf: vec![0u8; BUF_SIZE],
            pos: BUF_SIZE,
        };
        reader.refill();
        reader
    }

    /// Build a reader seeded by absorbing `seed`, mirroring
    /// `XofCursor::from_seed`.
    #[must_use]
    pub fn from_seed(seed: &[u8]) -> Self {
        use spongefish::DuplexSpongeInterface;
        let mut sponge = Poseidon2ByteSponge::<I>::default();
        sponge.absorb(SPARSE_PRG_DOMAIN);
        sponge.absorb(seed);
        Self::new(sponge)
    }

    fn refill(&mut self) {
        use spongefish::DuplexSpongeInterface;
        let mut buf = core::mem::take(&mut self.buf);
        self.sponge.squeeze(&mut buf);
        self.buf = buf;
        self.pos = 0;
    }

    #[inline]
    fn next_u8(&mut self) -> u8 {
        if self.pos >= BUF_SIZE {
            self.refill();
        }
        let b = self.buf[self.pos];
        self.pos += 1;
        b
    }

    #[inline]
    fn next_u32(&mut self) -> u32 {
        if self.pos + 4 <= BUF_SIZE {
            let val = u32::from_le_bytes(self.buf[self.pos..self.pos + 4].try_into().unwrap());
            self.pos += 4;
            val
        } else {
            let mut tmp = [0u8; 4];
            for b in &mut tmp {
                *b = self.next_u8();
            }
            u32::from_le_bytes(tmp)
        }
    }

    /// Copy `out.len()` bytes from the buffered stream.
    pub fn fill_bytes(&mut self, out: &mut [u8]) {
        let mut off = 0;
        while off < out.len() {
            if self.pos >= BUF_SIZE {
                self.refill();
            }
            let avail = BUF_SIZE - self.pos;
            let take = avail.min(out.len() - off);
            out[off..off + take].copy_from_slice(&self.buf[self.pos..self.pos + take]);
            self.pos += take;
            off += take;
        }
    }

    /// Uniform draw from `0..modulus` by bitmask rejection.
    ///
    /// Byte-for-byte the same algorithm as `XofCursor::next_usize_mod`,
    /// including the 1/2/4-byte read widths, so the induced distribution over
    /// Fisher-Yates positions is identical.
    #[inline]
    pub fn next_usize_mod(&mut self, modulus: usize) -> usize {
        debug_assert!(modulus > 0);
        if modulus == 1 {
            return 0;
        }
        let bits = usize::BITS - (modulus - 1).leading_zeros();
        if bits <= 8 {
            let mask = ((1u16 << bits) - 1) as u8;
            loop {
                let val = (self.next_u8() & mask) as usize;
                if val < modulus {
                    return val;
                }
            }
        } else if bits <= 16 {
            let mask = (1usize << bits) - 1;
            loop {
                let lo = self.next_u8() as usize;
                let hi = self.next_u8() as usize;
                let val = (lo | (hi << 8)) & mask;
                if val < modulus {
                    return val;
                }
            }
        } else {
            let mask: usize = (1 << bits) - 1;
            loop {
                let val = (self.next_u32() as usize) & mask;
                if val < modulus {
                    return val;
                }
            }
        }
    }

    /// Exactly-uniform, rejection-free field element, rule (S2).
    ///
    /// This is the derivation an in-circuit verifier should use.
    ///
    /// Note that this draws a fresh element straight from the duplex and does
    /// **not** consume the reader's 4 KB byte buffer, so interleaving it with
    /// [`Self::next_usize_mod`] or [`Self::fill_bytes`] in one reader gives a
    /// well-defined but confusing stream: the byte draws run ahead of the field
    /// draws by up to a buffer. Use one mode per reader.
    pub fn next_field(&mut self) -> Fp {
        self.sponge.squeeze_field()
    }
}

/// Domain separator mirroring `akita_challenges`' `SPARSE_PRG_DOMAIN`, so a
/// ported sparse sampler keeps a PRG stream distinct from transcript output.
const SPARSE_PRG_DOMAIN: &[u8] = b"akita/sparse-challenge-prg";

#[cfg(test)]
mod tests {
    use super::*;
    use crate::poseidon2::Alpha3;

    type R = Poseidon2ChallengeReader<Alpha3>;

    #[test]
    fn reader_is_deterministic_in_the_seed() {
        let mut a = R::from_seed(b"seed-1");
        let mut b = R::from_seed(b"seed-1");
        let mut c = R::from_seed(b"seed-2");
        let (mut xa, mut xb, mut xc) = ([0u8; 64], [0u8; 64], [0u8; 64]);
        a.fill_bytes(&mut xa);
        b.fill_bytes(&mut xb);
        c.fill_bytes(&mut xc);
        assert_eq!(xa, xb);
        assert_ne!(xa, xc);
    }

    #[test]
    fn draws_stay_in_range_across_all_read_widths() {
        // 8-bit, 16-bit and 32-bit paths of next_usize_mod.
        for modulus in [1usize, 2, 3, 64, 255, 256, 257, 4096, 65_535, 65_537, 1 << 20] {
            let mut reader = R::from_seed(b"range");
            for _ in 0..2_000 {
                assert!(reader.next_usize_mod(modulus) < modulus);
            }
        }
    }

    #[test]
    fn small_modulus_draws_look_uniform() {
        // Not a proof of uniformity, just a smoke test that no residue class is
        // starved. Exact uniformity is argued in the module header.
        const MODULUS: usize = 7;
        const DRAWS: usize = 70_000;
        let mut counts = [0usize; MODULUS];
        let mut reader = R::from_seed(b"uniformity");
        for _ in 0..DRAWS {
            counts[reader.next_usize_mod(MODULUS)] += 1;
        }
        let expected = DRAWS / MODULUS;
        for (value, &count) in counts.iter().enumerate() {
            let deviation = (count as isize - expected as isize).unsigned_abs();
            assert!(
                deviation < expected / 10,
                "residue {value} count {count} deviates too far from {expected}"
            );
        }
    }

    #[test]
    fn partial_fisher_yates_shape_is_reproducible() {
        // Mirrors how the sparse sampler consumes the reader: pick `weight`
        // distinct positions out of `ring_d`.
        fn positions(seed: &[u8], ring_d: usize, weight: usize) -> Vec<usize> {
            let mut reader = R::from_seed(seed);
            let mut pool: Vec<usize> = (0..ring_d).collect();
            let mut out = Vec::with_capacity(weight);
            for i in 0..weight {
                let j = i + reader.next_usize_mod(ring_d - i);
                pool.swap(i, j);
                out.push(pool[i]);
            }
            out
        }

        let left = positions(b"fy", 64, 31);
        let right = positions(b"fy", 64, 31);
        assert_eq!(left, right);
        assert_eq!(left.len(), 31);

        let mut sorted = left.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), 31, "positions must be distinct");
        assert!(sorted.iter().all(|&p| p < 64));

        assert_ne!(left, positions(b"fy-other", 64, 31));
    }

    #[test]
    fn field_draws_are_rejection_free_and_deterministic() {
        let mut a = R::from_seed(b"fields");
        let mut b = R::from_seed(b"fields");
        for _ in 0..32 {
            assert_eq!(a.next_field(), b.next_field());
        }
    }
}
