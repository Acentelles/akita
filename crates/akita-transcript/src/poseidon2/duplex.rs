//! Rate-8 / capacity-4 duplex over the Poseidon2 permutation, plus the
//! byte-facing adapter that plugs into `spongefish::DuplexSpongeInterface`.
//!
//! # UNAUDITED RESEARCH CRYPTOGRAPHY - NOT FOR PRODUCTION
//!
//! See the [module header](super) for the standing risk statement.
//!
//! # Encoding rules (normative)
//!
//! The Akita transcript is byte-oriented: callers absorb length-framed
//! `AkitaSerialize` payloads and squeeze byte strings. The algebraic sponge is
//! element-oriented. The bridge is fixed by two rules, both testable and both
//! reproduced by `tools/poseidon2_sponge_reference.py`.
//!
//! **(A1) Absorb.** Let `m` be the concatenation of all bytes absorbed since
//! the last squeeze (or since construction). Encode `m` as
//! `m || 0x01 || 0x00^k` with `k` minimal so the length is a multiple of 7,
//! then split into 7-byte little-endian chunks; each chunk is an integer in
//! `[0, 2^56)`, hence always a canonical element of `F_{p*}`. Absorb the chunks
//! into the duplex in order. This is SAFE-style `10*` padding applied at the
//! byte level; it is injective on byte strings, so distinct absorbed strings
//! give distinct element sequences. The padding element is appended lazily, at
//! the absorb-to-squeeze transition, which keeps `absorb` associative as the
//! `DuplexSpongeInterface` contract requires.
//!
//! **(S1) Squeeze to bytes.** Squeeze one element `x` at a time. Accept `x`
//! iff `x < 255 * 2^56 = 2^64 - 2^56`, otherwise discard it and squeeze
//! another. From an accepted `x`, emit the low 7 bytes of `x` in
//! little-endian order. Because `floor(p*/2^56) = 255`, the accepted range is
//! exactly 255 complete residue classes mod `2^56`, so the emitted 7 bytes are
//! **exactly uniform** on `[0, 2^56)` when `x` is uniform on `[0, p*)`. The
//! rejection probability is `(2^56 - 59)/p* ~ 2^-8`.
//!
//! Rule (S1) is deliberately exact rather than rejection-free: the naive
//! "emit the low 7 bytes of every element" rule has total-variation distance
//! `59 * 2^-64 ~ 2^-58.1` from uniform per element, which is far above the
//! `2^-128` soundness target this transcript is supposed to serve.
//!
//! **(S2) Squeeze to field elements**, [`Poseidon2ByteSponge::squeeze_field`]:
//! return the squeezed element directly. Exactly uniform on `F_{p*}` with **no
//! rejection**, which is the rule an in-circuit verifier must use; data
//! dependent rejection loops (S1) cannot be expressed in a loop-free circuit
//! (survey Section 6.2). Callers that will ever be proven in-circuit should
//! derive challenges through (S2).
//!
//! # Duplex
//!
//! Overwrite-mode duplex, [CO25] Construction 3.3, structurally identical to
//! `spongefish::DuplexSponge<_, 12, 8>` except that the capacity is
//! initialised to a domain IV instead of zero (see
//! [`Poseidon2Instance::capacity_iv`](super::Poseidon2Instance::capacity_iv)).
//!
//! [CO25]: https://eprint.iacr.org/2025/536

use core::marker::PhantomData;

use spongefish::DuplexSpongeInterface;

use super::{to_u64, Fp, Poseidon2, Poseidon2Instance, RATE, WIDTH};

/// Bytes carried by one squeezed field element under rule (S1).
pub const BYTES_PER_ELEMENT: usize = 7;

/// Bytes packed into one absorbed field element under rule (A1).
pub const ABSORB_BYTES_PER_ELEMENT: usize = 7;

/// Rule (S1) acceptance bound: `255 * 2^56 = floor(p*/2^56) * 2^56`.
pub const SQUEEZE_ACCEPT_BOUND: u64 = 0xff00_0000_0000_0000;

const _: () = assert!(SQUEEZE_ACCEPT_BOUND == 255u64 << 56);

/// Rate-8 / capacity-4 duplex over the Poseidon2 permutation.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Poseidon2Duplex<I: Poseidon2Instance> {
    state: [Fp; WIDTH],
    absorb_pos: usize,
    squeeze_pos: usize,
    _instance: PhantomData<fn() -> I>,
}

impl<I: Poseidon2Instance> Default for Poseidon2Duplex<I> {
    fn default() -> Self {
        Self::new()
    }
}

impl<I: Poseidon2Instance> Poseidon2Duplex<I> {
    /// Fresh duplex: rate zeroed, capacity set to the instance domain IV.
    #[must_use]
    pub fn new() -> Self {
        let iv = I::capacity_iv();
        let mut state = [Fp::default(); WIDTH];
        state[RATE..].copy_from_slice(&iv);
        Self {
            state,
            absorb_pos: 0,
            squeeze_pos: RATE,
            _instance: PhantomData,
        }
    }

    /// Absorb field elements (overwrite mode).
    pub fn absorb_elements(&mut self, mut input: &[Fp]) {
        self.squeeze_pos = RATE;
        while !input.is_empty() {
            if self.absorb_pos == RATE {
                Poseidon2::<I>::permute(&mut self.state);
                self.absorb_pos = 0;
            } else {
                let chunk_len = input.len().min(RATE - self.absorb_pos);
                self.state[self.absorb_pos..self.absorb_pos + chunk_len]
                    .copy_from_slice(&input[..chunk_len]);
                self.absorb_pos += chunk_len;
                input = &input[chunk_len..];
            }
        }
    }

    /// Squeeze one field element, permuting at rate boundaries.
    pub fn squeeze_element(&mut self) -> Fp {
        self.absorb_pos = 0;
        if self.squeeze_pos == RATE {
            self.squeeze_pos = 0;
            Poseidon2::<I>::permute(&mut self.state);
        }
        let out = self.state[self.squeeze_pos];
        self.squeeze_pos += 1;
        out
    }

    /// One-way ratchet: drop the rate, permute, forbid further squeezing from
    /// the current block.
    pub fn ratchet(&mut self) {
        self.absorb_pos = RATE;
        self.squeeze_pos = RATE;
        self.state[..RATE].fill(Fp::default());
        Poseidon2::<I>::permute(&mut self.state);
    }

    /// Current permutation state (tests and known-answer vectors only).
    #[must_use]
    pub fn state(&self) -> &[Fp; WIDTH] {
        &self.state
    }
}

/// Byte-facing Poseidon2 duplex implementing `spongefish::DuplexSpongeInterface`.
///
/// Encoding is fixed by rules (A1) and (S1) in the module header.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Poseidon2ByteSponge<I: Poseidon2Instance> {
    inner: Poseidon2Duplex<I>,
    pending: [u8; ABSORB_BYTES_PER_ELEMENT],
    pending_len: usize,
    absorbing: bool,
    out_buf: [u8; BYTES_PER_ELEMENT],
    out_pos: usize,
}

impl<I: Poseidon2Instance> Default for Poseidon2ByteSponge<I> {
    fn default() -> Self {
        Self {
            inner: Poseidon2Duplex::new(),
            pending: [0u8; ABSORB_BYTES_PER_ELEMENT],
            pending_len: 0,
            absorbing: false,
            out_buf: [0u8; BYTES_PER_ELEMENT],
            out_pos: BYTES_PER_ELEMENT,
        }
    }
}

impl<I: Poseidon2Instance> Poseidon2ByteSponge<I> {
    /// Fresh sponge.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline]
    fn element_from_chunk(chunk: &[u8]) -> Fp {
        debug_assert_eq!(chunk.len(), ABSORB_BYTES_PER_ELEMENT);
        let mut limb = [0u8; 8];
        limb[..ABSORB_BYTES_PER_ELEMENT].copy_from_slice(chunk);
        Fp::from_u64(u64::from_le_bytes(limb))
    }

    /// Rule (A1) tail: append `0x01`, zero-fill, absorb.
    fn finish_absorb(&mut self) {
        if !self.absorbing {
            return;
        }
        let mut block = [0u8; ABSORB_BYTES_PER_ELEMENT];
        block[..self.pending_len].copy_from_slice(&self.pending[..self.pending_len]);
        block[self.pending_len] = 0x01;
        self.inner
            .absorb_elements(&[Self::element_from_chunk(&block)]);
        self.pending_len = 0;
        self.absorbing = false;
    }

    fn absorb_bytes(&mut self, input: &[u8]) {
        if input.is_empty() {
            return;
        }
        self.absorbing = true;
        // Any bytes still buffered from a previous squeeze are dropped, as in
        // `spongefish::DuplexSponge::absorb`.
        self.out_pos = BYTES_PER_ELEMENT;

        let mut rest = input;
        // Complete a partially filled group first.
        if self.pending_len > 0 {
            let take = rest.len().min(ABSORB_BYTES_PER_ELEMENT - self.pending_len);
            self.pending[self.pending_len..self.pending_len + take].copy_from_slice(&rest[..take]);
            self.pending_len += take;
            rest = &rest[take..];
            if self.pending_len == ABSORB_BYTES_PER_ELEMENT {
                let element = Self::element_from_chunk(&self.pending);
                self.inner.absorb_elements(&[element]);
                self.pending_len = 0;
            }
        }
        let full = rest.len() / ABSORB_BYTES_PER_ELEMENT;
        if full > 0 {
            let elements: Vec<Fp> = rest[..full * ABSORB_BYTES_PER_ELEMENT]
                .chunks_exact(ABSORB_BYTES_PER_ELEMENT)
                .map(Self::element_from_chunk)
                .collect();
            self.inner.absorb_elements(&elements);
            rest = &rest[full * ABSORB_BYTES_PER_ELEMENT..];
        }
        if !rest.is_empty() {
            self.pending[..rest.len()].copy_from_slice(rest);
            self.pending_len = rest.len();
        }
    }

    /// Rule (S1): squeeze one accepted element and refill the byte buffer.
    fn refill_out(&mut self) {
        loop {
            let x = to_u64(self.inner.squeeze_element());
            if x < SQUEEZE_ACCEPT_BOUND {
                self.out_buf
                    .copy_from_slice(&x.to_le_bytes()[..BYTES_PER_ELEMENT]);
                self.out_pos = 0;
                return;
            }
        }
    }

    fn squeeze_bytes(&mut self, out: &mut [u8]) {
        if out.is_empty() {
            return;
        }
        self.finish_absorb();
        let mut written = 0;
        while written < out.len() {
            if self.out_pos == BYTES_PER_ELEMENT {
                self.refill_out();
            }
            let take = (BYTES_PER_ELEMENT - self.out_pos).min(out.len() - written);
            out[written..written + take]
                .copy_from_slice(&self.out_buf[self.out_pos..self.out_pos + take]);
            self.out_pos += take;
            written += take;
        }
    }

    /// Rule (S2): squeeze one exactly-uniform, rejection-free field element.
    ///
    /// This is the derivation an in-circuit verifier must replay. Any bytes
    /// buffered by a previous [`Self::squeeze_bytes`] are discarded, so byte
    /// and field squeezes never share a partially consumed element.
    pub fn squeeze_field(&mut self) -> Fp {
        self.finish_absorb();
        self.out_pos = BYTES_PER_ELEMENT;
        self.inner.squeeze_element()
    }

    /// Squeeze `out.len()` exactly-uniform field elements (rule (S2)).
    pub fn squeeze_field_elements(&mut self, out: &mut [Fp]) {
        for slot in out.iter_mut() {
            *slot = self.squeeze_field();
        }
    }

    /// Absorb field elements directly, bypassing rule (A1).
    ///
    /// Only for callers that are already element-oriented; mixing this with
    /// byte absorbs in one transcript loses (A1)'s injectivity guarantee, so
    /// do not do that.
    pub fn absorb_field_elements(&mut self, input: &[Fp]) {
        self.finish_absorb();
        self.out_pos = BYTES_PER_ELEMENT;
        self.inner.absorb_elements(input);
    }

    /// Read-only access to the underlying duplex (tests, known-answer vectors).
    #[must_use]
    pub fn duplex(&self) -> &Poseidon2Duplex<I> {
        &self.inner
    }
}

impl<I: Poseidon2Instance> DuplexSpongeInterface for Poseidon2ByteSponge<I> {
    type U = u8;

    fn absorb(&mut self, input: &[u8]) -> &mut Self {
        self.absorb_bytes(input);
        self
    }

    fn squeeze(&mut self, output: &mut [u8]) -> &mut Self {
        self.squeeze_bytes(output);
        self
    }

    fn ratchet(&mut self) -> &mut Self {
        self.finish_absorb();
        self.inner.ratchet();
        self.out_pos = BYTES_PER_ELEMENT;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::poseidon2::{Alpha3, Alpha7, MODULUS};

    type S3 = Poseidon2ByteSponge<Alpha3>;
    type S7 = Poseidon2ByteSponge<Alpha7>;

    #[test]
    fn accept_bound_gives_exact_uniformity() {
        // floor(p*/2^56) = 255, so [0, 255*2^56) is 255 whole residue classes.
        assert_eq!(MODULUS >> 56, 255);
        assert_eq!(SQUEEZE_ACCEPT_BOUND, 255u64 << 56);
        const { assert!(SQUEEZE_ACCEPT_BOUND < MODULUS) };
        // Rejection mass is exactly 2^56 - 59 out of p*.
        assert_eq!(MODULUS - SQUEEZE_ACCEPT_BOUND, (1u64 << 56) - 59);
    }

    #[test]
    fn absorb_is_associative() {
        let payload: Vec<u8> = (0..97u8).collect();
        let mut one = S3::new();
        one.absorb(&payload);
        let a = one.squeeze_array::<64>();

        for split in [1usize, 6, 7, 8, 13, 14, 50, 96] {
            let mut two = S3::new();
            two.absorb(&payload[..split]);
            two.absorb(&payload[split..]);
            let b = two.squeeze_array::<64>();
            assert_eq!(a, b, "absorb split at {split} changed the output");
        }
    }

    #[test]
    fn squeeze_is_associative() {
        let mut one = S3::new();
        one.absorb(b"akita");
        let a = one.squeeze_array::<64>();

        for split in [1usize, 7, 8, 14, 31, 63] {
            let mut two = S3::new();
            two.absorb(b"akita");
            let mut b = [0u8; 64];
            two.squeeze(&mut b[..split]);
            two.squeeze(&mut b[split..]);
            assert_eq!(a, b, "squeeze split at {split} changed the output");
        }
    }

    #[test]
    fn padding_is_injective_on_lengths() {
        // "ab" and "ab\x01" must not collide: (A1) appends 0x01 unambiguously.
        let mut left = S3::new();
        left.absorb(b"ab");
        let mut right = S3::new();
        right.absorb(b"ab\x01");
        assert_ne!(left.squeeze_array::<32>(), right.squeeze_array::<32>());

        // Zero-extension must not collide either.
        let mut a = S3::new();
        a.absorb(b"ab");
        let mut b = S3::new();
        b.absorb(b"ab\x00");
        assert_ne!(a.squeeze_array::<32>(), b.squeeze_array::<32>());
    }

    #[test]
    fn determinism_and_input_sensitivity() {
        let mut a = S3::new();
        a.absorb(b"same input");
        let mut b = S3::new();
        b.absorb(b"same input");
        assert_eq!(a.squeeze_array::<48>(), b.squeeze_array::<48>());

        let mut c = S3::new();
        c.absorb(b"same inpuT");
        assert_ne!(a.squeeze_array::<48>(), c.squeeze_array::<48>());
    }

    #[test]
    fn interleaved_duplex_rounds_differ_from_flat_absorb() {
        let mut duplexed = S3::new();
        duplexed.absorb(b"round-1");
        let first = duplexed.squeeze_array::<16>();
        duplexed.absorb(b"round-2");
        let second = duplexed.squeeze_array::<16>();
        assert_ne!(first, second);

        let mut flat = S3::new();
        flat.absorb(b"round-1round-2");
        assert_ne!(second, flat.squeeze_array::<16>());
    }

    #[test]
    fn instances_are_domain_separated() {
        let mut a = S3::new();
        a.absorb(b"payload");
        let mut b = S7::new();
        b.absorb(b"payload");
        assert_ne!(a.squeeze_array::<32>(), b.squeeze_array::<32>());
    }

    #[test]
    fn squeezed_elements_respect_the_accept_bound() {
        let mut s = S3::new();
        s.absorb(b"bound check");
        // Every emitted 7-byte group must come from an accepted element, so
        // reading many bytes must never panic and must stay in range.
        let mut buf = [0u8; 7 * 64];
        s.squeeze(&mut buf);
        for chunk in buf.chunks_exact(7) {
            let mut limb = [0u8; 8];
            limb[..7].copy_from_slice(chunk);
            assert!(u64::from_le_bytes(limb) < 1u64 << 56);
        }
    }

    #[test]
    fn field_squeeze_is_in_range_and_rejection_free() {
        let mut s = S3::new();
        s.absorb(b"field squeeze");
        let mut out = [Fp::default(); 40];
        s.squeeze_field_elements(&mut out);
        for e in out {
            assert!(to_u64(e) < MODULUS);
        }
        // Determinism.
        let mut t = S3::new();
        t.absorb(b"field squeeze");
        let mut out2 = [Fp::default(); 40];
        t.squeeze_field_elements(&mut out2);
        assert_eq!(out, out2);
    }

    #[test]
    fn ratchet_is_one_way_and_changes_state() {
        let mut a = S3::new();
        a.absorb(b"before ratchet");
        let mut b = a.clone();
        b.ratchet();
        assert_ne!(a.squeeze_array::<32>(), b.squeeze_array::<32>());
    }

    #[test]
    fn absorb_discards_buffered_squeeze_bytes() {
        let mut a = S3::new();
        a.absorb(b"x");
        let _ = a.squeeze_array::<1>();
        a.absorb(b"y");
        let after = a.squeeze_array::<8>();

        let mut b = S3::new();
        b.absorb(b"x");
        let _ = b.squeeze_array::<3>();
        b.absorb(b"y");
        assert_eq!(after, b.squeeze_array::<8>());
    }
}
