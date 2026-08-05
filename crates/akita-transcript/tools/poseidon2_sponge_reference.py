#!/usr/bin/env python3
"""Independent reference implementation of the Akita Poseidon2 byte sponge.

RESEARCH USE ONLY. See `crates/akita-transcript/src/poseidon2/mod.rs`.

This is deliberately a *second*, structurally different implementation of the
same specification as the Rust code:

  * the linear layers are plain O(t^2) matrix-vector products over the matrices
    emitted by `poseidon2_params_p64m59.sage`, not the fast M4 / sum-and-scale
    circuits the Rust uses;
  * the S-box is `pow(x, alpha, p)`, not an addition chain;
  * the duplex and the byte encodings are written from the normative rules in
    the Rust module header, not ported from the Rust source.

Agreement between the two is therefore a meaningful check on both.

Usage:
  python3 poseidon2_sponge_reference.py alpha3.json alpha7.json > vectors.json
"""

import json
import sys

P = 18446744073709551557  # 2^64 - 59
WIDTH = 12
RATE = 8
CAPACITY = WIDTH - RATE
BYTES_PER_ELEMENT = 7
ABSORB_BYTES_PER_ELEMENT = 7
SQUEEZE_ACCEPT_BOUND = 255 << 56

DOMAIN_TAGS = {
    3: b"akita-pcs/transcript/v1/p2a3",
    7: b"akita-pcs/transcript/v1/p2a7",
}


class Poseidon2:
    """Poseidon2 permutation, naive matrix form."""

    def __init__(self, params):
        self.p = params["prime"]
        self.t = params["t"]
        self.alpha = params["alpha"]
        self.rf = params["R_F"]
        self.rp = params["R_P"]
        self.rc = params["round_constants"]
        self.me = params["external_matrix"]
        self.mi = params["internal_matrix"]
        assert self.p == P
        assert self.t == WIDTH
        assert len(self.rc) == (self.rf + self.rp) * self.t

    def _matvec(self, m, x):
        return [sum(m[i][j] * x[j] for j in range(self.t)) % self.p for i in range(self.t)]

    def _sbox(self, x):
        return pow(x, self.alpha, self.p)

    def permute(self, state):
        s = [v % self.p for v in state]
        half = self.rf // 2
        s = self._matvec(self.me, s)
        for r in range(half):
            base = r * self.t
            s = [(s[i] + self.rc[base + i]) % self.p for i in range(self.t)]
            s = [self._sbox(v) for v in s]
            s = self._matvec(self.me, s)
        for r in range(self.rp):
            base = (half + r) * self.t
            s[0] = (s[0] + self.rc[base]) % self.p
            s[0] = self._sbox(s[0])
            s = self._matvec(self.mi, s)
        for r in range(half):
            base = (half + self.rp + r) * self.t
            s = [(s[i] + self.rc[base + i]) % self.p for i in range(self.t)]
            s = [self._sbox(v) for v in s]
            s = self._matvec(self.me, s)
        return s


class Duplex:
    """Overwrite-mode duplex, rate 8 / capacity 4, capacity IV from the tag."""

    def __init__(self, perm, tag):
        assert len(tag) == CAPACITY * 7
        iv = [
            int.from_bytes(tag[i * 7:(i + 1) * 7], "little")
            for i in range(CAPACITY)
        ]
        self.perm = perm
        self.state = [0] * RATE + iv
        self.absorb_pos = 0
        self.squeeze_pos = RATE

    def absorb_elements(self, elements):
        self.squeeze_pos = RATE
        i = 0
        while i < len(elements):
            if self.absorb_pos == RATE:
                self.state = self.perm.permute(self.state)
                self.absorb_pos = 0
            else:
                take = min(len(elements) - i, RATE - self.absorb_pos)
                self.state[self.absorb_pos:self.absorb_pos + take] = elements[i:i + take]
                self.absorb_pos += take
                i += take

    def squeeze_element(self):
        self.absorb_pos = 0
        if self.squeeze_pos == RATE:
            self.squeeze_pos = 0
            self.state = self.perm.permute(self.state)
        out = self.state[self.squeeze_pos]
        self.squeeze_pos += 1
        return out

    def ratchet(self):
        self.absorb_pos = RATE
        self.squeeze_pos = RATE
        self.state[:RATE] = [0] * RATE
        self.state = self.perm.permute(self.state)


class ByteSponge:
    """Byte-facing sponge implementing rules (A1) and (S1)."""

    def __init__(self, params):
        self.inner = Duplex(Poseidon2(params), DOMAIN_TAGS[params["alpha"]])
        self.pending = b""
        self.absorbing = False
        self.out_buf = b""

    def _finish_absorb(self):
        if not self.absorbing:
            return
        block = self.pending + b"\x01"
        block = block + b"\x00" * (ABSORB_BYTES_PER_ELEMENT - len(block))
        self.inner.absorb_elements([int.from_bytes(block, "little")])
        self.pending = b""
        self.absorbing = False

    def absorb(self, data):
        if not data:
            return
        self.absorbing = True
        self.out_buf = b""
        buf = self.pending + data
        full = len(buf) // ABSORB_BYTES_PER_ELEMENT
        elements = [
            int.from_bytes(buf[i * 7:(i + 1) * 7], "little") for i in range(full)
        ]
        if elements:
            self.inner.absorb_elements(elements)
        self.pending = buf[full * ABSORB_BYTES_PER_ELEMENT:]

    def squeeze(self, n):
        if n == 0:
            return b""
        self._finish_absorb()
        out = b""
        while len(out) < n:
            if not self.out_buf:
                while True:
                    x = self.inner.squeeze_element()
                    if x < SQUEEZE_ACCEPT_BOUND:
                        self.out_buf = x.to_bytes(8, "little")[:BYTES_PER_ELEMENT]
                        break
            take = min(len(self.out_buf), n - len(out))
            out += self.out_buf[:take]
            self.out_buf = self.out_buf[take:]
        return out

    def squeeze_field(self, n):
        self._finish_absorb()
        self.out_buf = b""
        return [self.inner.squeeze_element() for _ in range(n)]

    def ratchet(self):
        self._finish_absorb()
        self.inner.ratchet()
        self.out_buf = b""


# --- script -----------------------------------------------------------------

# (name, program). A program is a list of ops applied in order.
SCRIPTS = [
    ("empty_squeeze", [("squeeze", 32)]),
    ("single_byte", [("absorb", b"\x00"), ("squeeze", 32)]),
    ("short", [("absorb", b"akita"), ("squeeze", 32)]),
    ("exact_group", [("absorb", b"1234567"), ("squeeze", 14)]),
    ("rate_boundary", [("absorb", bytes(range(56))), ("squeeze", 64)]),
    ("multi_block", [("absorb", bytes(range(200))), ("squeeze", 100)]),
    (
        "duplexed",
        [
            ("absorb", b"round-1"),
            ("squeeze", 16),
            ("absorb", b"round-2"),
            ("squeeze", 16),
            ("absorb", b""),
            ("squeeze", 3),
        ],
    ),
    (
        "ratcheted",
        [("absorb", b"before"), ("ratchet", 0), ("squeeze", 32)],
    ),
    (
        "field_squeeze",
        [("absorb", b"field path"), ("squeeze_field", 20)],
    ),
    (
        "mixed_byte_and_field",
        [
            ("absorb", b"mixed"),
            ("squeeze", 5),
            ("squeeze_field", 3),
            ("absorb", b"more"),
            ("squeeze", 9),
        ],
    ),
]


def run(params):
    out = []
    for name, program in SCRIPTS:
        sponge = ByteSponge(params)
        ops = []
        for kind, arg in program:
            if kind == "absorb":
                sponge.absorb(arg)
                ops.append({"op": "absorb", "bytes": list(arg)})
            elif kind == "squeeze":
                got = sponge.squeeze(arg)
                ops.append({"op": "squeeze", "len": arg, "out": list(got)})
            elif kind == "squeeze_field":
                got = sponge.squeeze_field(arg)
                ops.append({"op": "squeeze_field", "len": arg, "out": got})
            elif kind == "ratchet":
                sponge.ratchet()
                ops.append({"op": "ratchet"})
            else:
                raise ValueError(kind)
        out.append({"name": name, "ops": ops})
    return out


def main():
    result = {}
    for path in sys.argv[1:]:
        with open(path) as fh:
            params = json.load(fh)
        alpha = params["alpha"]
        perm = Poseidon2(params)
        # Re-derive the permutation KATs the sage script emitted, as an
        # independent confirmation of the permutation itself.
        for kat in params["kats"]:
            got = perm.permute(kat["input"])
            assert got == kat["output"], (
                "python reference disagrees with the sage reference for alpha=%d" % alpha
            )
        result["alpha%d" % alpha] = {
            "permutation_kats": params["kats"],
            "sponge_kats": run(params),
        }
    json.dump(result, sys.stdout)


if __name__ == "__main__":
    main()
