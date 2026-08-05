#!/usr/bin/env python3
"""Independent verification of the VENDORED Poseidon2 parameters.

RESEARCH USE ONLY. See `crates/akita-transcript/src/poseidon2/mod.rs`.

This script re-derives and re-checks everything from the Rust files that are
actually committed, using only the Python standard library. It shares no code
with the SageMath generator: the Grain LFSR, the round-number inequalities, the
linear algebra (Gaussian elimination, Faddeev-LeVerrier characteristic
polynomials, Rabin irreducibility), and the permutation are all reimplemented
here. Agreement is therefore a real cross-check, not a tautology.

What it verifies, per alpha:

  1. Round constants regenerate bit-for-bit from the Grain LFSR seeded with
     (field, sbox, n, t, R_F, R_P), including the rejection rule and the
     Poseidon2 internal-round zero padding.
  2. The internal diagonal regenerates from the continuation of the same Grain
     stream, after the documented number of rejected draws.
  3. Round numbers (R_F, R_P) reproduce from the published inequalities plus
     the eprint 2023/537 correction and the designers' security margin.
  4. M4 is MDS (all 69 minors) and is the matrix from eprint 2023/323 Sec. 5.1.
  5. The external matrix is circ(2*M4, M4, M4), is invertible, and is NOT MDS;
     the minimal singular minor is reported, and the branch-number witness
     x = (-s, -s, 3s) giving wt(x) + wt(M_E x) = 7 is checked.
  6. The internal matrix is 1*1^T + diag(mu - 1), is invertible, and satisfies
     check_minpoly_condition: for i = 1..2t the characteristic polynomial of
     M_I^i has degree t and is irreducible over F_p.
  7. The vendored permutation KATs are reproduced by an independent naive
     implementation of the permutation.
  8. Every constant is a canonical field element (< p).

Usage (from the repository root):

  python3 crates/akita-transcript/tools/verify_generated_params.py
"""

import itertools
import os
import re
import sys

P = 18446744073709551557  # 2^64 - 59
T = 12
M4 = [[5, 7, 1, 3], [4, 6, 1, 1], [1, 3, 5, 7], [1, 1, 4, 6]]

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "src", "poseidon2")

FAILURES = []


def check(name, condition, detail=""):
    status = "ok  " if condition else "FAIL"
    print(f"  [{status}] {name}{(' - ' + detail) if detail else ''}")
    if not condition:
        FAILURES.append(name)
    return condition


# --- parsing the vendored Rust -------------------------------------------


def parse_u64_array(text, name):
    m = re.search(rf"const {name}: \[u64; (\d+)\] = \[(.*?)\];", text, re.S)
    if not m:
        raise SystemExit(f"could not find {name}")
    values = [int(v) for v in re.findall(r"\d+", m.group(2))]
    assert len(values) == int(m.group(1)), name
    return values


def parse_matrix(text, name):
    m = re.search(rf"const {name}: \[\[u64; 12\]; 12\] = \[(.*?)\n\];", text, re.S)
    if not m:
        raise SystemExit(f"could not find {name}")
    rows = re.findall(r"\[([^\]]*)\]", m.group(1))
    out = [[int(v) for v in re.findall(r"\d+", r)] for r in rows]
    assert len(out) == 12 and all(len(r) == 12 for r in out), name
    return out


def parse_kats(text):
    m = re.search(r"PERMUTATION_KATS: \[\(\[u64; 12\], \[u64; 12\]\); \d+\] = \[(.*?)\n\];", text, re.S)
    if not m:
        raise SystemExit("could not find PERMUTATION_KATS")
    out = []
    for pair in re.findall(r"\(\[([^\]]*)\], \[([^\]]*)\]\)", m.group(1)):
        out.append(([int(v) for v in re.findall(r"\d+", pair[0])],
                    [int(v) for v in re.findall(r"\d+", pair[1])]))
    return out


def parse_scalar(text, pattern):
    m = re.search(pattern, text)
    if not m:
        raise SystemExit(f"could not find {pattern}")
    return int(m.group(1))


# --- Grain LFSR (reimplemented from eprint 2019/458 Appendix F) -----------


class Grain:
    def __init__(self, field, sbox, n, t, rf, rp):
        bits = (
            [int(b) for b in bin(field)[2:].zfill(2)]
            + [int(b) for b in bin(sbox)[2:].zfill(4)]
            + [int(b) for b in bin(n)[2:].zfill(12)]
            + [int(b) for b in bin(t)[2:].zfill(12)]
            + [int(b) for b in bin(rf)[2:].zfill(10)]
            + [int(b) for b in bin(rp)[2:].zfill(10)]
            + [1] * 30
        )
        assert len(bits) == 80
        self.state = bits
        for _ in range(160):
            self._clock()

    def _clock(self):
        s = self.state
        new = s[62] ^ s[51] ^ s[38] ^ s[23] ^ s[13] ^ s[0]
        s.pop(0)
        s.append(new)
        return new

    def next_bit(self):
        # The reference generator drops every other bit: clock once, and while
        # the produced bit is 0, clock twice more; then clock once and yield.
        while True:
            new = self._clock()
            while new == 0:
                self._clock()
                new = self._clock()
            return self._clock()

    def next_int(self, num_bits):
        v = 0
        for _ in range(num_bits):
            v = (v << 1) | self.next_bit()
        return v

    def next_field(self):
        while True:
            v = self.next_int(64)
            if v < P:
                return v


def regenerate_constants(rf, rp):
    grain = Grain(1, 0, 64, T, rf, rp)
    out = []
    num_constants = rf * T + rp
    for i in range(num_constants):
        out.append(grain.next_field())
        if (rf / 2) * T <= i < ((rf / 2) * T) + rp:
            out.extend([0] * (T - 1))
    return out, grain


# --- round numbers (reimplemented from the designers' inequalities) -------


def sat_inequiv_alpha(t, rf, rp, alpha, M, field_size=64):
    import math

    log2 = math.log2
    logalpha2 = log2(2) / log2(alpha)  # log_alpha(2)

    rf1 = 6 if M <= (math.floor(log2(P) - ((alpha - 1) / 2.0))) * (t + 1) else 10
    rf2 = 1 + math.ceil(logalpha2 * min(M, field_size)) + math.ceil(log2(t) / log2(alpha)) - rp
    rf3 = (logalpha2 * min(M, log2(P))) - rp
    rf4 = t - 1 + logalpha2 * min(M / float(t + 1), log2(P) / 2.0) - rp
    rf5 = (t - 2 + (M / (2 * log2(alpha))) - rp) / float(t - 1)
    rf_max = max(math.ceil(rf1), math.ceil(rf2), math.ceil(rf3), math.ceil(rf4), math.ceil(rf5))

    # eprint 2023/537 correction, exact integer binomial
    r_temp = math.floor(t / 3.0)
    over = (rf - 1) * t + rp + r_temp + r_temp * (rf / 2.0) + rp + alpha
    under = r_temp * (rf / 2.0) + rp + alpha
    binom = comb_float(over, under)
    binom_log = math.log2(binom) if binom > 0 else float("-inf")
    cost_gb4 = math.ceil(2 * binom_log)
    return rf >= rf_max and cost_gb4 >= M


def comb_float(over, under):
    import math

    over_i, under_i = int(over), int(under)
    if over != over_i or under != under_i:
        raise ValueError("non-integral binomial arguments")
    if under_i < 0 or under_i > over_i:
        return 0
    return math.comb(over_i, under_i)


def find_round_numbers(t, alpha, M=128):
    import math

    best = None
    for rp_t in range(1, 500):
        for rf_t in range(4, 100, 2):
            if sat_inequiv_alpha(t, rf_t, rp_t, alpha, M):
                rf = rf_t + 2
                rp = int(math.ceil(rp_t * 1.075))
                cost = t * rf + rp
                if best is None or cost < best[0] or (cost == best[0] and rf < best[1]):
                    best = (cost, rf, rp)
    return best[1], best[2]


# --- linear algebra over F_p ---------------------------------------------


def det(m):
    n = len(m)
    a = [row[:] for row in m]
    d = 1
    for c in range(n):
        piv = next((r for r in range(c, n) if a[r][c] % P), None)
        if piv is None:
            return 0
        if piv != c:
            a[c], a[piv] = a[piv], a[c]
            d = -d
        d = d * a[c][c] % P
        inv = pow(a[c][c], P - 2, P)
        for r in range(c + 1, n):
            f = a[r][c] * inv % P
            if f:
                for cc in range(c, n):
                    a[r][cc] = (a[r][cc] - f * a[c][cc]) % P
    return d % P


def is_mds(m):
    n = len(m)
    for k in range(1, n + 1):
        for rs in itertools.combinations(range(n), k):
            for cs in itertools.combinations(range(n), k):
                if det([[m[r][c] for c in cs] for r in rs]) == 0:
                    return False, (k, rs, cs)
    return True, None


def min_singular_minor(m, max_k):
    n = len(m)
    for k in range(1, max_k + 1):
        for rs in itertools.combinations(range(n), k):
            for cs in itertools.combinations(range(n), k):
                if det([[m[r][c] for c in cs] for r in rs]) == 0:
                    return k, list(rs), list(cs)
    return None


def matmul(a, b):
    n = len(a)
    return [[sum(a[i][k] * b[k][j] for k in range(n)) % P for j in range(n)] for i in range(n)]


def matvec(m, v):
    return [sum(m[i][j] * v[j] for j in range(len(v))) % P for i in range(len(m))]


def charpoly(a):
    """Faddeev-LeVerrier: returns coefficients [c0, ..., c_{n-1}, 1]."""
    n = len(a)
    ident = [[1 if i == j else 0 for j in range(n)] for i in range(n)]
    coeffs = [0] * (n + 1)
    coeffs[n] = 1
    mk = [[0] * n for _ in range(n)]
    for k in range(1, n + 1):
        # M_k = A*M_{k-1} + c_{n-k+1} I
        if k == 1:
            mk = [row[:] for row in ident]
        else:
            mk = matmul(a, mk)
            c = coeffs[n - k + 1]
            for i in range(n):
                mk[i][i] = (mk[i][i] + c) % P
        am = matmul(a, mk)
        trace = sum(am[i][i] for i in range(n)) % P
        coeffs[n - k] = (-trace * pow(k, P - 2, P)) % P
    return coeffs


# --- polynomial arithmetic mod (p, f) for Rabin irreducibility ------------


def poly_trim(a):
    while len(a) > 1 and a[-1] == 0:
        a.pop()
    return a


def poly_mulmod(a, b, f):
    n = len(f) - 1
    res = [0] * (len(a) + len(b) - 1)
    for i, ai in enumerate(a):
        if ai:
            for j, bj in enumerate(b):
                if bj:
                    res[i + j] = (res[i + j] + ai * bj) % P
    # reduce by monic f
    for d in range(len(res) - 1, n - 1, -1):
        c = res[d]
        if c:
            res[d] = 0
            for j in range(n):
                res[d - n + j] = (res[d - n + j] - c * f[j]) % P
    return poly_trim(res[:n] if len(res) > n else res)


def poly_powmod(base, e, f):
    result = [1]
    b = base[:]
    while e:
        if e & 1:
            result = poly_mulmod(result, b, f)
        b = poly_mulmod(b, b, f)
        e >>= 1
    return result


def poly_sub(a, b):
    n = max(len(a), len(b))
    out = [0] * n
    for i in range(n):
        x = a[i] if i < len(a) else 0
        y = b[i] if i < len(b) else 0
        out[i] = (x - y) % P
    return poly_trim(out)


def poly_gcd(a, b):
    a, b = poly_trim(a[:]), poly_trim(b[:])
    while not (len(b) == 1 and b[0] == 0):
        # a mod b
        r = a[:]
        inv = pow(b[-1], P - 2, P)
        while len(r) >= len(b) and not (len(r) == 1 and r[0] == 0):
            shift = len(r) - len(b)
            c = r[-1] * inv % P
            for i, bi in enumerate(b):
                r[shift + i] = (r[shift + i] - c * bi) % P
            r = poly_trim(r)
            if len(r) < len(b):
                break
        a, b = b, r
    return a


def is_irreducible(f):
    """Rabin's test for monic f over F_p."""
    n = len(f) - 1
    if n <= 1:
        return n == 1
    x = [0, 1]
    # frob_k = x^(p^k) mod f
    frob = x[:]
    checkpoints = {n // q for q in (2, 3, 5, 7, 11) if n % q == 0}
    for k in range(1, n + 1):
        frob = poly_powmod(frob, P, f)
        if k in checkpoints:
            g = poly_gcd(poly_sub(frob, x), f)
            if not (len(g) == 1 and g[0] != 0):
                return False
    return poly_sub(frob, x) == [0]


# --- permutation (naive, independent of the Rust fast circuits) ----------


def permute(state, alpha, rf, rp, rc, me, mi):
    s = [v % P for v in state]
    half = rf // 2
    s = matvec(me, s)
    for r in range(half):
        base = r * T
        s = [(s[i] + rc[base + i]) % P for i in range(T)]
        s = [pow(v, alpha, P) for v in s]
        s = matvec(me, s)
    for r in range(rp):
        base = (half + r) * T
        s[0] = pow((s[0] + rc[base]) % P, alpha, P)
        s = matvec(mi, s)
    for r in range(half):
        base = (half + rp + r) * T
        s = [(s[i] + rc[base + i]) % P for i in range(T)]
        s = [pow(v, alpha, P) for v in s]
        s = matvec(me, s)
    return s


# --- main -----------------------------------------------------------------


def verify(alpha):
    print(f"\n=== alpha = {alpha} ===")
    path = os.path.join(SRC, f"generated_alpha{alpha}.rs")
    text = open(path).read()

    rc = parse_u64_array(text, "ROUND_CONSTANTS")
    diag_m1 = parse_u64_array(text, "INTERNAL_DIAG_M1")
    mi = parse_matrix(text, "INTERNAL_MATRIX")
    me = parse_matrix(text, "EXTERNAL_MATRIX")
    kats = parse_kats(text)
    rf = parse_scalar(text, r"rounds_f = (\d+),")
    rp = parse_scalar(text, r"rounds_p = (\d+),")
    file_alpha = parse_scalar(text, r"alpha = (\d+),")

    check("vendored alpha matches", file_alpha == alpha)
    check("gcd(alpha, p-1) == 1", __import__("math").gcd(alpha, P - 1) == 1)

    # 3. round numbers
    got_rf, got_rp = find_round_numbers(T, alpha)
    check(
        "round numbers reproduce from the published inequalities",
        (got_rf, got_rp) == (rf, rp),
        f"independent search gives (R_F, R_P) = ({got_rf}, {got_rp}), file has ({rf}, {rp})",
    )

    # 1 + 2. Grain regeneration
    regen, grain = regenerate_constants(rf, rp)
    check("round constants regenerate from the Grain LFSR", regen == rc,
          f"{len(rc)} constants")
    check("all round constants are canonical (< p)", all(0 <= c < P for c in rc))

    draws = 0
    while True:
        draws += 1
        cand = [grain.next_field() for _ in range(T)]
        m = [[(1 if i != j else 0) + (cand[i] if i == j else 0) for j in range(T)]
             for i in range(T)]
        # M = (J - I) + diag(cand)
        m = [[(0 if i == j else 1) for j in range(T)] for i in range(T)]
        for i in range(T):
            m[i][i] = cand[i]
        if all(x == y for x, y in zip([m[i][i] - 1 for i in range(T)], diag_m1)):
            break
        if draws > 200:
            break
    check("internal diagonal regenerates from the Grain stream continuation",
          [(m[i][i] - 1) % P for i in range(T)] == diag_m1,
          f"accepted on Grain draw {draws}")

    # 4. M4
    ok, witness = is_mds(M4)
    check("M4 is MDS (all 69 minors)", ok, "" if ok else str(witness))

    # 5. external matrix
    expected_me = [
        [(2 * M4[i % 4][j % 4] if i // 4 == j // 4 else M4[i % 4][j % 4]) % P for j in range(T)]
        for i in range(T)
    ]
    check("external matrix == circ(2*M4, M4, M4)", me == expected_me)
    check("external matrix is invertible", det(me) != 0, f"det = {det(me)}")
    msm = min_singular_minor(me, 2)
    check("external matrix is NOT MDS (by design; branch number claim, not MDS)",
          msm is not None, f"minimal singular minor: size {msm[0]} at rows {msm[1]}, cols {msm[2]}")
    x = [0] * T
    x[0], x[4], x[8] = P - 1, P - 1, 3
    img = matvec(me, x)
    weight = sum(1 for v in x if v) + sum(1 for v in img if v)
    check("branch-number witness gives wt(x) + wt(M_E x) = 7 (claimed b = t/4 + 4)",
          weight == 7, f"got {weight}")

    # 6. internal matrix
    expected_mi = [[1 if i != j else (diag_m1[i] + 1) % P for j in range(T)] for i in range(T)]
    check("internal matrix == 1*1^T + diag(mu - 1)", mi == expected_mi)
    check("internal matrix is invertible", det(mi) != 0)
    print("       running check_minpoly_condition for i = 1..24 (Rabin irreducibility)...")
    power = [row[:] for row in mi]
    minpoly_ok = True
    for i in range(1, 2 * T + 1):
        cp = charpoly(power)
        if len(cp) - 1 != T or not is_irreducible(cp):
            minpoly_ok = False
            print(f"       failed at i = {i}")
            break
        power = matmul(mi, power)
    check("check_minpoly_condition: charpoly(M_I^i) irreducible of degree 12, i = 1..24",
          minpoly_ok)

    # 7. KATs
    kats_ok = all(permute(inp, alpha, rf, rp, rc, me, mi) == out for inp, out in kats)
    check(f"vendored permutation KATs reproduce ({len(kats)} vectors)", kats_ok)


def main():
    print("Independent verification of the vendored Poseidon2 parameters")
    print(f"p* = {P} = 2^64 - 59, t = {T}")
    for alpha in (3, 7):
        verify(alpha)
    print()
    if FAILURES:
        print(f"FAILED: {len(FAILURES)} check(s): {FAILURES}")
        sys.exit(1)
    print("All checks passed.")


if __name__ == "__main__":
    main()
