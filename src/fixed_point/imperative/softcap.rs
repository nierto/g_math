//! Softcapped activations, correctly rounded: `softcap(x, c) = c tanh(x / c)`
//! and the softcapped SwiGLU `cg tanh(g / cg) sigmoid(g) cu tanh(u / cu)`.
//!
//! Every result is the exact value rounded once to storage (nearest, ties
//! toward +infinity). The exact stage is [`settle`] over the formulas below
//! (the format's compute tier with tracked bounds, then `Mpf` up to 4160
//! bits). On the 32-bit word a faster first stage runs before it: every
//! exponential `exp(-a)` (`a >= 0`) at 62 fractional bits from a table of
//! `exp(-k / 128)` and a degree-6 polynomial, the factors combined through
//! one reciprocal (interpolated from a table, two Newton steps; no division
//! instruction), and the result carried with an error bound in units of
//! `2^-(F + 32)`. Slices run four elements interleaved (the same arithmetic
//! per element) so their dependency chains overlap. A result is returned from there only when the bound keeps
//! it off a rounding boundary; the rest (about one in a hundred million at
//! ordinary caps) goes to [`settle`]. So both stages return the same word.
//!
//! Error budget of the fast stage, in absolute value at 62 fractional bits:
//! table entries within 2^-62 of `exp(-k/128)` (built by integer series at
//! 126 bits, checked against `Mpf` in the tests); the polynomial within
//! 2^-60 (truncated Taylor remainder below 2^-61.3; six steps with floored
//! `1/n!` coefficients, at most two units each, damped by `r < 2^-7`); the
//! product with the entry one more unit: `exp(-a)` within 2^-58.6 for an
//! exact argument (measured worst: 2 units of 2^-62). The reciprocal is
//! within 4 units of 2^-62 (measured worst: 1). A tanh argument
//! `2|x| / c` comes from a 63-bit reciprocal of `c`: relative error below
//! 2^-62, so below 2^-56.4 absolute for `a < 46`, and the exponential's
//! error stays below 2^-56 in all. Arguments from 45 up give 0 (the true
//! value is below 2^-64).

use crate::fixed_point::domains::binary_fixed::transcendental::mpf::MpfErr;
use crate::fixed_point::exact::{settle, Formula, Num};
use crate::fixed_point::fixed_num::FixedNum;
use crate::fixed_point::width::{Storage, WideInt};
use crate::fixed_point::Fixed;

type R<X> = Result<X, MpfErr>;

const ONE: u64 = 1 << 62;
/// Table steps per unit of the argument and the number of entries (to 45).
const STEP_BITS: u32 = 7;
const ENTRIES: usize = 45 << STEP_BITS;
/// `45 * 2^62`: from here on `exp(-a) < 2^-64` and the fast stage returns 0.
const SATURATE: u128 = 45u128 << 62;

/// `floor(a * b / 2^126)` for `a, b < 2^127`.
fn mul_shr126(a: u128, b: u128) -> u128 {
    let (a1, a0) = (a >> 64, a & u64::MAX as u128);
    let (b1, b0) = (b >> 64, b & u64::MAX as u128);
    let (hh, hl, lh, ll) = (a1 * b1, a1 * b0, a0 * b1, a0 * b0);
    // the 256-bit product: hh * 2^128 + (hl + lh) * 2^64 + ll
    let mid = (ll >> 64) + (hl & u64::MAX as u128) + (lh & u64::MAX as u128);
    let hi = hh + (hl >> 64) + (lh >> 64) + (mid >> 64);
    let lo_top = mid & u64::MAX as u128; // bits 64..128 of the product
    (hi << 2) | (lo_top >> 62)
}

/// `exp(-k / 128)` at 62 fractional bits, `k < ENTRIES`: powers of
/// `exp(-1/128)` (a Taylor sum at 126 bits) at 126 bits, rounded once.
fn exp_table() -> &'static [u64] {
    static TABLE: std::sync::OnceLock<Vec<u64>> = std::sync::OnceLock::new();
    TABLE.get_or_init(|| {
        let one126: u128 = 1 << 126;
        // exp(-1/128) = sum (-1)^n / (128^n n!)
        let (mut sum, mut term, mut n) = (one126, one126, 1u128);
        loop {
            term /= 128 * n;
            if term == 0 {
                break;
            }
            if n % 2 == 1 { sum -= term } else { sum += term }
            n += 1;
        }
        let mut p = one126;
        let mut t = Vec::with_capacity(ENTRIES);
        for _ in 0..ENTRIES {
            t.push(((p + (1 << 63)) >> 64) as u64);
            p = mul_shr126(p, sum);
        }
        t
    })
}

/// `exp(-a)` at 62 fractional bits for `a = a_q / 2^62 >= 0`, within 2^-58.6
/// of the exact value (see the module notes).
#[inline]
fn exp_neg(table: &[u64], a_q: u128) -> u64 {
    if a_q >= SATURATE {
        return 0;
    }
    let k = (a_q >> (62 - STEP_BITS)) as usize;
    let r = (a_q & ((1u128 << (62 - STEP_BITS)) - 1)) as u64;
    // e^-r = c0 - r(c1 - r(c2 - r(c3 - r(c4 - r(c5 - r c6))))), c_n = 1/n! (floored)
    let mut acc = INV_FACT[6];
    for n in (0..6).rev() {
        acc = INV_FACT[n] - ((r as u128 * acc as u128) >> 62) as u64;
    }
    ((table[k] as u128 * acc as u128) >> 62) as u64
}

/// [`exp_neg`] on `N` independent arguments at once: the same arithmetic per
/// lane, written lane-interleaved so the dependency chains overlap.
#[inline(always)]
fn exp_neg_n<const N: usize>(table: &[u64], a_q: [u128; N]) -> [u64; N] {
    let mut k = [0usize; N];
    let mut r = [0u64; N];
    let mut sat = [false; N];
    for i in 0..N {
        sat[i] = a_q[i] >= SATURATE;
        let a = if sat[i] { 0 } else { a_q[i] };
        k[i] = (a >> (62 - STEP_BITS)) as usize;
        r[i] = (a & ((1u128 << (62 - STEP_BITS)) - 1)) as u64;
    }
    let mut acc = [INV_FACT[6]; N];
    for n in (0..6).rev() {
        for i in 0..N {
            acc[i] = INV_FACT[n] - ((r[i] as u128 * acc[i] as u128) >> 62) as u64;
        }
    }
    let mut out = [0u64; N];
    for i in 0..N {
        out[i] = if sat[i] { 0 } else { ((table[k[i]] as u128 * acc[i] as u128) >> 62) as u64 };
    }
    out
}

/// [`recip`] on `N` independent divisors at once (same arithmetic per lane).
#[inline(always)]
fn recip_n<const N: usize>(rt: &[u64], d: [u64; N]) -> [u64; N] {
    let mut y = [0u64; N];
    for i in 0..N {
        let x = d[i] - ONE;
        let j = (x >> 54) as usize;
        let fr = (x & ((1u64 << 54) - 1)) as u128;
        let (a, b) = (rt[j], rt[(j + 1).min(256)]);
        y[i] = a - (((a - b) as u128 * fr) >> 54) as u64;
    }
    for _ in 0..2 {
        for i in 0..N {
            let t = ((d[i] as u128 * y[i] as u128) >> 62) as u64;
            y[i] = ((y[i] as u128 * ((2 * ONE) - t) as u128) >> 62) as u64;
        }
    }
    y
}

/// [`softcap_fast`] on `N` inputs at once (same arithmetic per lane).
#[inline(always)]
fn softcap_fast_n<const N: usize>(t: Tables, x: [i128; N], f: u32, cap: &Cap, frac: u32) -> [Option<i128>; N] {
    let mut a = [0u128; N];
    for i in 0..N {
        a[i] = cap.tanh_arg(x[i].unsigned_abs(), f, frac);
    }
    let e = exp_neg_n(t.exp, a);
    let mut d = [0u64; N];
    for i in 0..N {
        d[i] = ONE + e[i];
    }
    let y = recip_n(t.rec, d);
    let bound = (cap.c >> 21) + 3;
    let mut out = [None; N];
    for i in 0..N {
        let q = ((ONE - e[i]) as u128 * y[i] as u128) >> 62;
        out[i] = if x[i] == 0 { Some(0) } else { decide((cap.c * q) >> 30, bound, x[i] < 0) };
    }
    out
}

/// [`swiglu_fast`] on `N` gate/up pairs at once (same arithmetic per lane).
#[inline(always)]
fn swiglu_fast_n<const N: usize>(t: Tables, g: [i128; N], u: [i128; N], f: u32, gc: &Cap, uc: &Cap, frac: u32) -> [Option<i128>; N] {
    let one = ONE as u128;
    let mut a1 = [0u128; N];
    let mut a2 = [0u128; N];
    let mut a3 = [0u128; N];
    for i in 0..N {
        let (ga, ua) = (g[i].unsigned_abs(), u[i].unsigned_abs());
        a1[i] = gc.tanh_arg(ga, f, frac);
        a2[i] = plain_arg(ga, f);
        a3[i] = uc.tanh_arg(ua, f, frac);
    }
    let e1 = exp_neg_n(t.exp, a1);
    let e2 = exp_neg_n(t.exp, a3);
    let mode = if gc.c == 2u128 << frac { 1 } else if gc.c == 4u128 << frac { 2 } else { 0 };
    let sx = if mode == 0 { exp_neg_n(t.exp, a2) } else { [0u64; N] };
    let mut pn = [0u128; N];
    let mut d = [0u64; N];
    let mut sh = [0u32; N];
    for i in 0..N {
        let (e1i, e2i) = (e1[i] as u128, e2[i] as u128);
        let s = match mode {
            1 => e1i,
            2 => (e1i * e1i) >> 62,
            _ => sx[i] as u128,
        };
        let ns = if g[i] > 0 { one } else { s };
        pn[i] = ((((one - e1i) * ns) >> 62) * (one - e2i)) >> 62;
        let pd = ((((one + e1i) * (one + s)) >> 62) * (one + e2i)) >> 62;
        sh[i] = (128 - pd.leading_zeros()) - 63;
        d[i] = (pd >> sh[i]) as u64;
    }
    let y = recip_n(t.rec, d);
    let bound = ((gc.c * uc.c) >> (20 + frac)) + 3;
    let mut out = [None; N];
    for i in 0..N {
        let q = (pn[i] * y[i] as u128) >> (62 + sh[i]);
        out[i] = if g[i] == 0 || u[i] == 0 { Some(0) } else { decide((gc.c * uc.c * q) >> (30 + frac), bound, (g[i] < 0) != (u[i] < 0)) };
    }
    out
}

/// `x * sigmoid(g)` for `N` pairs of raws at `f` fraction bits, the result at
/// `frac`: sigmoid at 62 fraction bits from one exponential and one
/// reciprocal (within 2^-58: exponential 2^-58.6, reciprocal 2^-60,
/// floors), the product carried in units of 2^-(frac + 32) with the bound
/// `|x| 2^(frac - f - 26)`, doubled, plus the floors.
#[inline(always)]
fn sigmul_fast_n<const N: usize>(t: Tables, x: [i128; N], g: [i128; N], f: u32, frac: u32) -> [Option<i128>; N] {
    let mut a = [0u128; N];
    for i in 0..N {
        a[i] = plain_arg(g[i].unsigned_abs(), f);
    }
    let e = exp_neg_n(t.exp, a);
    let mut d = [0u64; N];
    for i in 0..N {
        d[i] = ONE + e[i];
    }
    let y = recip_n(t.rec, d);
    let mut out = [None; N];
    for i in 0..N {
        let num = if g[i] >= 0 { ONE } else { e[i] };
        let q = (num as u128 * y[i] as u128) >> 62;
        let xa = x[i].unsigned_abs();
        let w = (xa * q) >> (30 + f - frac);
        let bound = (xa >> (f - frac + 25)) + 3;
        out[i] = if x[i] == 0 { Some(0) } else { decide(w, bound, x[i] < 0) };
    }
    out
}

/// `1 / n!` at 62 fraction bits, floored.
const INV_FACT: [u64; 7] = [ONE, ONE, ONE / 2, ONE / 6, ONE / 24, ONE / 120, ONE / 720];

/// `floor(2^124 / (2^62 + i 2^54))`, `i = 0..=256`: `1/x` at 62 fraction bits
/// on a grid of 2^-8 over `x` in [1, 2].
fn recip_table() -> &'static [u64] {
    static TABLE: std::sync::OnceLock<Vec<u64>> = std::sync::OnceLock::new();
    TABLE.get_or_init(|| (0..=256u128).map(|i| ((1u128 << 124) / ((1u128 << 62) + (i << 54))) as u64).collect())
}

/// `2^124 / d` for `d` in [2^62, 2^63], within 4 units (2^-60 relative):
/// linear interpolation on the table (within 2^-17), then two Newton steps
/// `y (2 - x y)` (the relative error squares: 2^-34, then 2^-68, below the
/// steps' own truncation). No division.
#[inline]
fn recip(rt: &[u64], d: u64) -> u64 {
    let x = d - ONE;
    let i = (x >> 54) as usize;
    let fr = (x & ((1u64 << 54) - 1)) as u128;
    let (a, b) = (rt[i], rt[(i + 1).min(256)]);
    let mut y = a - (((a - b) as u128 * fr) >> 54) as u64;
    for _ in 0..2 {
        let t = ((d as u128 * y as u128) >> 62) as u64;
        y = ((y as u128 * ((2 * ONE) - t) as u128) >> 62) as u64;
    }
    y
}

/// A cap prepared for the fast stage: `|c|` and a 63-bit reciprocal.
#[derive(Clone, Copy)]
pub(crate) struct Cap {
    c: u128,
    rec: u128,
    len: u32,
}

impl Cap {
    /// `None` for a zero cap.
    pub(crate) fn new(raw: i128) -> Option<Self> {
        let c = raw.unsigned_abs();
        if c == 0 {
            return None;
        }
        let len = 128 - c.leading_zeros();
        // floor(2^(62 + len) / c), in (2^62, 2^63]
        let rec = (1u128 << (62 + len)) / c;
        Some(Self { c, rec, len })
    }

    /// `2|x| / |c|` at 62 fractional bits for a raw `|x|` at `f` fraction bits
    /// and the cap at `frac` (`f >= frac`), never above the exact value.
    #[inline]
    fn tanh_arg(&self, x_abs: u128, f: u32, frac: u32) -> u128 {
        // 2|x| 2^62 / c = |X| rec 2^(63 + frac - f - 62 - len)
        let sh = self.len - 1 + f - frac;
        let p = x_abs * self.rec;
        if sh >= 128 { 0 } else { p >> sh }
    }
}

/// `|x|` at 62 fractional bits, or the saturation point when larger.
#[inline]
fn plain_arg(x_abs: u128, f: u32) -> u128 {
    if x_abs >= 45u128 << f { SATURATE } else { x_abs << (62 - f) }
}

/// Nearest of `w / 2^32` when `w` is at least `bound + 1` from a half unit,
/// with the sign; `None` otherwise.
#[inline]
fn decide(w: u128, bound: u128, negative: bool) -> Option<i128> {
    // branch-free on the random parts (the sign and the side of the half)
    let rem = (w & 0xFFFF_FFFF) as i64;
    let off = rem - (1i64 << 31);
    if off.unsigned_abs() as u128 <= bound {
        return None;
    }
    let y = ((w >> 32) as i128) + (off > 0) as i128;
    let m = -(negative as i128); // 0 or -1
    Some((y ^ m) - m)
}

/// The fast stage of `softcap`: `x` a raw at `f` fraction bits, the cap and
/// the result at `frac`. `None` when it does not decide.
#[inline]
pub(crate) fn softcap_fast(t: Tables, x: i128, f: u32, cap: &Cap, frac: u32) -> Option<i128> {
    if x == 0 {
        return Some(0);
    }
    debug_assert!(cap.c <= 1 << 31 && f <= 62 && f >= frac);
    let e = exp_neg(t.exp, cap.tanh_arg(x.unsigned_abs(), f, frac));
    // tanh = (1 - e) / (1 + e) at 62 fraction bits, through the reciprocal
    // (softcap_fast_n is the same computation on several lanes)
    let q = ((ONE - e) as u128 * recip(t.rec, ONE + e) as u128) >> 62;
    // |y| 2^(frac + 32) = c tanh 2^32 = c q / 2^30
    let w = (cap.c * q) >> 30;
    // q within 128 units (tanh moves by at most 2 per unit of e: 2 * 2^-56)
    // + 4 (reciprocal) + 1 (floor): c 133 / 2^30 < c / 2^22.9; bounded by
    // c / 2^21 (four times that), plus the floors
    let bound = (cap.c >> 21) + 3;
    decide(w, bound, x < 0)
}

/// The fast stage of the softcapped SwiGLU (`g`, `u` raws at `f`, caps and
/// result at `frac`). `None` when it does not decide.
#[inline]
pub(crate) fn swiglu_fast(t: Tables, g: i128, u: i128, f: u32, gc: &Cap, uc: &Cap, frac: u32) -> Option<i128> {
    if g == 0 || u == 0 {
        return Some(0);
    }
    debug_assert!(gc.c <= 1 << 31 && uc.c <= 1 << 31 && f <= 62 && f >= frac);
    let (ga, ua) = (g.unsigned_abs(), u.unsigned_abs());
    let e1 = exp_neg(t.exp, gc.tanh_arg(ga, f, frac)) as u128;
    // exp(-|g|) = e1^(cg / 2): for a gate cap of 2 or 4 it is e1 or e1^2
    // (within 2^-55, inside the budget below), one exponential fewer
    let s = if gc.c == 2u128 << frac {
        e1
    } else if gc.c == 4u128 << frac {
        (e1 * e1) >> 62
    } else {
        exp_neg(t.exp, plain_arg(ga, f)) as u128
    };
    let e2 = exp_neg(t.exp, uc.tanh_arg(ua, f, frac)) as u128;
    let one = ONE as u128;
    // tanh(|g|/cg) = (1 - e1)/(1 + e1); sigmoid(g) = 1/(1 + s) or s/(1 + s); tanh(|u|/cu) likewise
    let ns = if g > 0 { one } else { s };
    let pn = ((((one - e1) * ns) >> 62) * (one - e2)) >> 62;
    let pd = ((((one + e1) * (one + s)) >> 62) * (one + e2)) >> 62;
    // pd in [2^62, 2^65]: normalise to [2^62, 2^63) (relative error 2^-62), q = pn / pd at 62 fraction bits
    let sh = (128 - pd.leading_zeros()) - 63;
    let q = (pn * recip(t.rec, (pd >> sh) as u64) as u128) >> (62 + sh);
    // |y| 2^(frac + 32) = cg cu q 2^(32 - frac - 62)
    let w = (gc.c * uc.c * q) >> (30 + frac);
    // q within 2^-52: the exponentials 2 + 1 (2^-55 when s = e1^2) + 2 times 2^-56,
    // truncations 2^-59, reciprocal 2^-60: below 2^-52.9; cg cu 2^(32 - frac) 2^-52, plus the floors
    let bound = ((gc.c * uc.c) >> (20 + frac)) + 3;
    decide(w, bound, (g < 0) != (u < 0))
}

/// The prepared tables for a call: `exp(-k/128)` and the reciprocal knots.
#[derive(Clone, Copy)]
pub(crate) struct Tables {
    exp: &'static [u64],
    rec: &'static [u64],
}

#[inline]
pub(crate) fn table() -> Tables {
    Tables { exp: exp_table(), rec: recip_table() }
}

// ---------------------------------------------------------------------------
// the exact stage
// ---------------------------------------------------------------------------

fn sigmoid<T: FixedNum, B: Num<T>>(g: B) -> R<B> {
    let one = B::int(1)?;
    if g.is_negative() {
        let e = g.exp()?;
        e.div(one.add(e)?)
    } else {
        one.div(one.add(g.neg().exp()?)?)
    }
}

struct SoftcapForm<W, S> {
    x: W,
    f: u32,
    cap: S,
}

impl<W: WideInt, S: Storage, const F: u32> Formula<Fixed<S, F>> for SoftcapForm<W, S>
where
    Fixed<S, F>: FixedNum<Storage = S>,
{
    fn eval<B: Num<Fixed<S, F>>>(&self) -> R<Vec<B>> {
        let x = B::from_word(self.x, self.f)?;
        let c = B::from_word(self.cap, F)?;
        Ok(vec![c.mul(x.div(c)?.tanh()?)?])
    }
}

struct SwigluForm<W, S> {
    g: W,
    u: W,
    f: u32,
    gc: S,
    uc: S,
}

impl<W: WideInt, S: Storage, const F: u32> Formula<Fixed<S, F>> for SwigluForm<W, S>
where
    Fixed<S, F>: FixedNum<Storage = S>,
{
    fn eval<B: Num<Fixed<S, F>>>(&self) -> R<Vec<B>> {
        let g = B::from_word(self.g, self.f)?;
        let u = B::from_word(self.u, self.f)?;
        let gc = B::from_word(self.gc, F)?;
        let uc = B::from_word(self.uc, F)?;
        let tg = gc.mul(g.div(gc)?.tanh()?)?;
        let tu = uc.mul(u.div(uc)?.tanh()?)?;
        Ok(vec![tg.mul(sigmoid::<Fixed<S, F>, B>(g)?)?.mul(tu)?])
    }
}

struct SigmulForm<W> {
    x: W,
    g: W,
    f: u32,
}

impl<W: WideInt, S: Storage, const F: u32> Formula<Fixed<S, F>> for SigmulForm<W>
where
    Fixed<S, F>: FixedNum<Storage = S>,
{
    fn eval<B: Num<Fixed<S, F>>>(&self) -> R<Vec<B>> {
        let x = B::from_word(self.x, self.f)?;
        let g = B::from_word(self.g, self.f)?;
        Ok(vec![x.mul(sigmoid::<Fixed<S, F>, B>(g)?)?])
    }
}

/// `x * sigmoid(g)` rounded once (raws at `f`, the result at `F`): the exact stage alone.
pub(crate) fn sigmul_exact<S: Storage, W: WideInt, const F: u32>(x: W, g: W, f: u32) -> S
where
    Fixed<S, F>: FixedNum<Storage = S>,
{
    settled::<S, F>(settle::<Fixed<S, F>, _>(&SigmulForm { x, g, f }), "sigmoid_mul_wide")
}

fn settled<S: Storage, const F: u32>(r: R<Vec<S>>, what: &str) -> S
where
    Fixed<S, F>: FixedNum<Storage = S>,
{
    match r {
        Ok(v) => v[0],
        Err(MpfErr::Overflow) => panic!("{what}: result exceeds storage range"),
        Err(e) => panic!("{what}: exact evaluation failed ({e:?})"),
    }
}

/// `cap tanh(x / cap)` rounded once (`x` at `f` fraction bits, the cap and
/// the result at `F`): the exact stage alone.
pub(crate) fn softcap_exact<S: Storage, W: WideInt, const F: u32>(x: W, f: u32, cap: S) -> S
where
    Fixed<S, F>: FixedNum<Storage = S>,
{
    settled::<S, F>(settle::<Fixed<S, F>, _>(&SoftcapForm { x, f, cap }), "softcap")
}

/// The softcapped SwiGLU rounded once: the exact stage alone.
pub(crate) fn swiglu_exact<S: Storage, W: WideInt, const F: u32>(g: W, u: W, f: u32, gc: S, uc: S) -> S
where
    Fixed<S, F>: FixedNum<Storage = S>,
{
    settled::<S, F>(settle::<Fixed<S, F>, _>(&SwigluForm { g, u, f, gc, uc }), "softcapped_swiglu")
}

// ---------------------------------------------------------------------------
// the 32-bit word: fast stage, then the exact one where undecided
// ---------------------------------------------------------------------------

/// Tests only: send every element to the exact stage (checks the fallback's wiring).
#[cfg(test)]
static FORCE_EXACT: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

#[inline(always)]
fn fast_allowed() -> bool {
    #[cfg(test)]
    {
        !FORCE_EXACT.load(std::sync::atomic::Ordering::Relaxed)
    }
    #[cfg(not(test))]
    {
        true
    }
}

fn narrow_i32(v: i128, what: &str) -> i32 {
    i32::try_from(v).unwrap_or_else(|_| panic!("{what}: result exceeds storage range"))
}

/// `out[i] = softcap(x[i])` for raws `x` at `f` fraction bits (`f` = F or 2F).
pub(crate) fn softcap_i32_into<W: WideInt + Into<i128>, const F: u32>(x: &[W], f: u32, cap: i32, out: &mut [i32])
where
    Fixed<i32, F>: FixedNum<Storage = i32>,
{
    let t = table();
    let c = Cap::new(cap as i128).expect("softcap: cap is zero");
    const L: usize = 4;
    let n4 = x.len() / L * L;
    for (oc, xc) in out[..n4].chunks_exact_mut(L).zip(x[..n4].chunks_exact(L)) {
        let r = softcap_fast_n::<L>(t, std::array::from_fn(|i| xc[i].into()), f, &c, F);
        for i in 0..L {
            oc[i] = match r[i].filter(|_| fast_allowed()) {
                Some(v) => narrow_i32(v, "softcap"),
                None => softcap_exact::<i32, W, F>(xc[i], f, cap),
            };
        }
    }
    for (o, &xi) in out[n4..].iter_mut().zip(&x[n4..]) {
        *o = match softcap_fast(t, xi.into(), f, &c, F).filter(|_| fast_allowed()) {
            Some(v) => narrow_i32(v, "softcap"),
            None => softcap_exact::<i32, W, F>(xi, f, cap),
        };
    }
}

/// `out[j] = swiglu(gate_up[2j], gate_up[2j + 1])` for raws at `f` fraction bits.
pub(crate) fn swiglu_i32_into<W: WideInt + Into<i128>, const F: u32>(gate_up: &[W], f: u32, gc: i32, uc: i32, out: &mut [i32])
where
    Fixed<i32, F>: FixedNum<Storage = i32>,
{
    let t = table();
    let g_cap = Cap::new(gc as i128).expect("softcapped_swiglu: gate cap is zero");
    let u_cap = Cap::new(uc as i128).expect("softcapped_swiglu: up cap is zero");
    const L: usize = 4;
    let n4 = out.len().min(gate_up.len() / 2) / L * L;
    for (oc, gu) in out[..n4].chunks_exact_mut(L).zip(gate_up[..2 * n4].chunks_exact(2 * L)) {
        let r = swiglu_fast_n::<L>(t, std::array::from_fn(|i| gu[2 * i].into()), std::array::from_fn(|i| gu[2 * i + 1].into()), f, &g_cap, &u_cap, F);
        for i in 0..L {
            oc[i] = match r[i].filter(|_| fast_allowed()) {
                Some(v) => narrow_i32(v, "softcapped_swiglu"),
                None => swiglu_exact::<i32, W, F>(gu[2 * i], gu[2 * i + 1], f, gc, uc),
            };
        }
    }
    for (o, pair) in out[n4..].iter_mut().zip(gate_up[2 * n4..].chunks_exact(2)) {
        *o = match swiglu_fast(t, pair[0].into(), pair[1].into(), f, &g_cap, &u_cap, F).filter(|_| fast_allowed()) {
            Some(v) => narrow_i32(v, "softcapped_swiglu"),
            None => swiglu_exact::<i32, W, F>(pair[0], pair[1], f, gc, uc),
        };
    }
}

/// `out[i] = x[i] * sigmoid(g[i])` for raws at `f` fraction bits.
pub(crate) fn sigmul_i32_into<W: WideInt + Into<i128>, const F: u32>(x: &[W], g: &[W], f: u32, out: &mut [i32])
where
    Fixed<i32, F>: FixedNum<Storage = i32>,
{
    let t = table();
    const L: usize = 4;
    let n4 = out.len() / L * L;
    for ((oc, xc), gc) in out[..n4].chunks_exact_mut(L).zip(x[..n4].chunks_exact(L)).zip(g[..n4].chunks_exact(L)) {
        let r = sigmul_fast_n::<L>(t, std::array::from_fn(|i| xc[i].into()), std::array::from_fn(|i| gc[i].into()), f, F);
        for i in 0..L {
            oc[i] = match r[i].filter(|_| fast_allowed()) {
                Some(v) => narrow_i32(v, "sigmoid_mul_wide"),
                None => sigmul_exact::<i32, W, F>(xc[i], gc[i], f),
            };
        }
    }
    for i in n4..out.len() {
        let [r] = sigmul_fast_n::<1>(t, [x[i].into()], [g[i].into()], f, F);
        out[i] = match r.filter(|_| fast_allowed()) {
            Some(v) => narrow_i32(v, "sigmoid_mul_wide"),
            None => sigmul_exact::<i32, W, F>(x[i], g[i], f),
        };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixed_point::domains::binary_fixed::transcendental::mpf::{Mpf, C3};

    /// Every table entry within 2^-62 of exp(-k/128), measured with Mpf at 192 bits.
    #[test]
    fn table_entries_within_one_unit() {
        let t = table().exp;
        let mut worst = 0u128;
        for (k, &v) in t.iter().enumerate() {
            let exact = Mpf::<C3>::from_raw(-(k as i128), STEP_BITS).exp().unwrap();
            let want: i128 = exact.round_to_raw::<i128>(80, false).unwrap(); // 2^80 units
            let got = (v as i128) << 18;
            worst = worst.max(got.abs_diff(want));
        }
        // 2^-62 = 2^18 units of 2^-80
        eprintln!("table: worst {worst} units of 2^-80 (bound 2^18)");
        assert!(worst <= 1 << 18, "worst {worst} units of 2^-80");
    }

    /// exp_neg within 2^-58.6 (bound used: 2^-56 with the argument error) on a
    /// sweep of arguments, against Mpf.
    #[test]
    fn exp_neg_within_budget() {
        let t = table().exp;
        let mut worst = 0u128;
        let mut s: u64 = 0x9E37_79B9_7F4A_7C15;
        for i in 0..20000u64 {
            s ^= s >> 12;
            s ^= s << 25;
            s ^= s >> 27;
            let a_q: u128 = if i < 1000 { (i as u128) << 55 } else { (s.wrapping_mul(0x2545_F491_4F6C_DD1D) as u128 * 45) >> 2 };
            let got = exp_neg(t, a_q) as i128;
            let exact = Mpf::<C3>::from_raw(-(a_q as i128), 62).exp().unwrap();
            let want: i128 = exact.round_to_raw::<i128>(62, false).unwrap();
            worst = worst.max(got.abs_diff(want));
        }
        // 2^-58.6 is 10.6 units of 2^-62; one more for the reference's rounding
        eprintln!("exp_neg: worst {worst} units of 2^-62 (bound 12)");
        assert!(worst <= 12, "worst {worst} units of 2^-62");
    }

    /// The reciprocal within 4 units of floor(2^124 / d) on the endpoints,
    /// every table knot, and 2 million random divisors.
    #[test]
    fn recip_within_four_units() {
        let rt = recip_table();
        let mut rng = Rng(0x2EC1);
        let mut worst = 0u64;
        let mut check = |d: u64| {
            let exact = ((1u128 << 124) / d as u128) as u64;
            worst = worst.max(recip(rt, d).abs_diff(exact));
        };
        for i in 0..=256u64 {
            for off in [0u64, 1, 1 << 30, (1 << 54) - 1] {
                let d = ONE + (i << 54) + off;
                if d <= 2 * ONE { check(d) }
            }
        }
        check(2 * ONE);
        for _ in 0..2_000_000 {
            check(ONE + (rng.next() >> 2));
        }
        eprintln!("recip: worst {worst} units of 2^-62 (bound 4)");
        assert!(worst <= 4);
    }

    /// ns per call of each piece (run with --ignored --nocapture).
    #[test]
    #[ignore]
    fn piece_timings() {
        use std::hint::black_box;
        let tb = table();
        let (t, rt) = (tb.exp, tb.rec);
        let mut rng = Rng(7);
        let n = 1 << 20;
        let a: Vec<u128> = (0..n).map(|_| (rng.next() as u128 * 20) >> 2).collect();
        let d: Vec<u64> = (0..n).map(|_| ONE + (rng.next() >> 2)).collect();
        let x: Vec<i128> = (0..n).map(|_| ((rng.next() >> 50) as i128) - (1 << 13)).collect();
        let c = Cap::new(30 << 10).unwrap();
        let time = |name: &str, f: &dyn Fn() -> u64| {
            let s = std::time::Instant::now();
            let v = f();
            eprintln!("{name}: {:.1} ns/call ({v})", s.elapsed().as_nanos() as f64 / n as f64);
        };
        time("exp_neg", &|| a.iter().map(|&q| black_box(exp_neg(t, q))).fold(0, |s, v| s ^ v));
        time("recip", &|| d.iter().map(|&q| black_box(recip(rt, q))).fold(0, |s, v| s ^ v));
        time("tanh_arg", &|| x.iter().map(|&q| black_box(c.tanh_arg(q.unsigned_abs(), 10, 10)) as u64).fold(0, |s, v| s ^ v));
        time("softcap_fast", &|| x.iter().map(|&q| black_box(softcap_fast(tb, q, 10, &c, 10)).unwrap_or(0) as u64).fold(0, |s, v| s ^ v));
        time("softcap_fast_n<4>", &|| x.chunks_exact(4).map(|q| black_box(softcap_fast_n::<4>(tb, [q[0], q[1], q[2], q[3]], 10, &c, 10)).iter().fold(0u64, |s, v| s ^ v.unwrap_or(0) as u64)).fold(0, |s, v| s ^ v));
        time("softcap_fast_n<8>", &|| x.chunks_exact(8).map(|q| black_box(softcap_fast_n::<8>(tb, std::array::from_fn(|i| q[i]), 10, &c, 10)).iter().fold(0u64, |s, v| s ^ v.unwrap_or(0) as u64)).fold(0, |s, v| s ^ v));
        time("softcap_fast_n<2>", &|| x.chunks_exact(2).map(|q| black_box(softcap_fast_n::<2>(tb, [q[0], q[1]], 10, &c, 10)).iter().fold(0u64, |s, v| s ^ v.unwrap_or(0) as u64)).fold(0, |s, v| s ^ v));
        let gu: Vec<i128> = (0..n).map(|_| ((rng.next() >> 49) as i128) - (1 << 14)).collect();
        let (g4, u25) = (Cap::new(4 << 10).unwrap(), Cap::new(25 << 10).unwrap());
        time("swiglu_fast (per pair)", &|| gu.chunks_exact(2).map(|q| black_box(swiglu_fast(tb, q[0], q[1], 10, &g4, &u25, 10)).unwrap_or(0) as u64).fold(0, |s, v| s ^ v) );
        time("swiglu_fast_n<4> (per pair)", &|| gu.chunks_exact(8).map(|q| black_box(swiglu_fast_n::<4>(tb, [q[0], q[2], q[4], q[6]], [q[1], q[3], q[5], q[7]], 10, &g4, &u25, 10)).iter().fold(0u64, |s, v| s ^ v.unwrap_or(0) as u64)).fold(0, |s, v| s ^ v));
        for q in gu.chunks_exact(8).take(100_000) {
            let r = swiglu_fast_n::<4>(tb, [q[0], q[2], q[4], q[6]], [q[1], q[3], q[5], q[7]], 10, &g4, &u25, 10);
            for i in 0..4 { assert_eq!(r[i], swiglu_fast(tb, q[2 * i], q[2 * i + 1], 10, &g4, &u25, 10)); }
            let c8 = Cap::new(8 << 10).unwrap();
            let r = swiglu_fast_n::<4>(tb, [q[0], q[2], q[4], q[6]], [q[1], q[3], q[5], q[7]], 10, &c8, &u25, 10);
            for i in 0..4 { assert_eq!(r[i], swiglu_fast(tb, q[2 * i], q[2 * i + 1], 10, &c8, &u25, 10)); }
        }
        // the lane form equals the single form
        for q in x.chunks_exact(4).take(200_000) {
            let n4 = softcap_fast_n::<4>(tb, [q[0], q[1], q[2], q[3]], 10, &c, 10);
            for i in 0..4 { assert_eq!(n4[i], softcap_fast(tb, q[i], 10, &c, 10)); }
        }
    }

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 >> 12;
            self.0 ^= self.0 << 25;
            self.0 ^= self.0 >> 27;
            self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
        }
    }

    /// Every input of softcap up to saturation (and a margin past it) at
    /// Q22.10 and Q16.16: where the fast stage decides, it equals the exact
    /// stage. Prints the inputs it leaves undecided.
    fn softcap_exhaustive<const F: u32>(cap_value: i64)
    where
        Fixed<i32, F>: FixedNum<Storage = i32>,
    {
        let t = table();
        let cap = (cap_value << F) as i32;
        let c = Cap::new(cap as i128).unwrap();
        // tanh(x / cap) leaves the exponential's range at 2|x|/cap = 45; go 10 percent past
        let lim = ((cap_value.abs() * 45 / 2 * 11 / 10) << F).min(i32::MAX as i64) as i32;
        let (mut decided, mut undecided) = (0u64, Vec::new());
        let mut x = -lim;
        while x <= lim {
            let want = softcap_exact::<i32, i32, F>(x, F, cap);
            match softcap_fast(t, x as i128, F, &c, F) {
                Some(v) => {
                    assert_eq!(v, want as i128, "softcap x {x} cap {cap_value} F {F}");
                    decided += 1;
                }
                None => undecided.push(x),
            }
            x += 1;
        }
        eprintln!("softcap Q.{F} cap {cap_value}: {decided} decided equal to the exact stage, {} undecided: {:?}", undecided.len(), &undecided[..undecided.len().min(12)]);
        assert!(undecided.len() as u64 * 100_000 < decided);
    }

    #[test]
    fn softcap_fast_equals_exact_everywhere_q22_10() {
        for cap in [30, 4, 25, -7] {
            softcap_exhaustive::<10>(cap);
        }
    }

    #[test]
    fn softcap_fast_equals_exact_everywhere_q16_16() {
        softcap_exhaustive::<16>(30);
    }

    /// Random gate/up pairs (and wide inputs) at Q22.10 and Q16.16: where the
    /// fast stage decides, it equals the exact stage.
    fn swiglu_random<const F: u32>(n: usize, wide: bool)
    where
        Fixed<i32, F>: FixedNum<Storage = i32>,
    {
        let t = table();
        let mut rng = Rng(0x5161_u64 + F as u64 + wide as u64);
        let caps = [(4i64, 25i64), (-4, 25), (1, 1), (8, 3)];
        let (mut decided, mut undecided) = (0u64, Vec::new());
        for i in 0..n {
            let (gcv, ucv) = caps[i % caps.len()];
            let (gc, uc) = ((gcv << F) as i32, (ucv << F) as i32);
            let (gk, uk) = (Cap::new(gc as i128).unwrap(), Cap::new(uc as i128).unwrap());
            // values mostly within +-64 (in units), some up to the word's range
            let draw = |rng: &mut Rng| -> i64 {
                let r = rng.next();
                let bits = if r % 16 == 0 { if F == 10 { 30 } else { 30 } } else { 6 + F };
                let m = ((r >> 8) & ((1u64 << bits) - 1)) as i64;
                if r & 1 == 1 { -m } else { m }
            };
            let (g, u) = (draw(&mut rng), draw(&mut rng));
            let (got, want) = if wide {
                let (gw, uw) = (g << F, u << F | (rng.next() & ((1 << F) - 1)) as i64);
                (swiglu_fast(t, gw as i128, uw as i128, 2 * F, &gk, &uk, F), std::panic::catch_unwind(|| swiglu_exact::<i32, i64, F>(gw, uw, 2 * F, gc, uc)).ok())
            } else {
                (swiglu_fast(t, g as i128, u as i128, F, &gk, &uk, F), std::panic::catch_unwind(|| swiglu_exact::<i32, i32, F>(g as i32, u as i32, F, gc, uc)).ok())
            };
            match (got, want) {
                (Some(v), Some(w)) => {
                    assert_eq!(v, w as i128, "swiglu g {g} u {u} caps {gcv} {ucv} F {F} wide {wide}");
                    decided += 1;
                }
                // the exact stage overflows exactly when the value leaves the word
                (Some(v), None) => assert!(i32::try_from(v).is_err(), "swiglu g {g} u {u}: fast {v}, exact overflow"),
                (None, _) => undecided.push((g, u, gcv, ucv)),
            }
        }
        eprintln!("swiglu Q.{F} wide {wide}: {decided} decided equal to the exact stage, {} undecided: {:?}", undecided.len(), &undecided[..undecided.len().min(8)]);
        assert!(undecided.len() as u64 * 10_000 < decided);
    }

    #[test]
    fn swiglu_fast_equals_exact_random() {
        std::panic::set_hook(Box::new(|_| {}));
        swiglu_random::<10>(400_000, false);
        swiglu_random::<10>(200_000, true);
        swiglu_random::<16>(200_000, false);
        swiglu_random::<16>(100_000, true);
        let _ = std::panic::take_hook();
    }

    /// The drivers' fallback: with every element forced to the exact stage,
    /// the public slice functions return what the fast stage returns.
    #[test]
    fn forced_exact_stage_equals_fast_stage_through_the_api() {
        use crate::fixed_point::imperative::fused::{sigmoid_mul_wide_slice, softcap_slice, softcap_wide_slice, softcapped_swiglu_interleaved, softcapped_swiglu_interleaved_wide};
        type X = Fixed<i32, 10>;
        let mut rng = Rng(0xFA11);
        let mut draw = |bits: u32| -> i64 {
            let r = rng.next();
            let m = ((r >> 8) & ((1u64 << bits) - 1)) as i64;
            if r & 1 == 1 { -m } else { m }
        };
        let xs: Vec<X> = (0..4000).map(|_| X::from_raw(draw(17) as i32)).collect();
        let ws: Vec<i64> = (0..4000).map(|_| draw(27)).collect();
        let (c30, c4, c25) = (X::from_raw(30 << 10), X::from_raw(4 << 10), X::from_raw(25 << 10));
        let gs: Vec<i64> = ws.iter().rev().map(|&w| w >> 3).collect();
        let run = || (softcap_slice(&xs, c30), softcap_wide_slice::<i32, 10>(&ws, c30), softcapped_swiglu_interleaved(&xs, c4, c25), softcapped_swiglu_interleaved_wide::<i32, 10>(&ws, c4, c25), sigmoid_mul_wide_slice::<i32, 10>(&ws, &gs));
        let fast = run();
        FORCE_EXACT.store(true, std::sync::atomic::Ordering::Relaxed);
        let exact = run();
        FORCE_EXACT.store(false, std::sync::atomic::Ordering::Relaxed);
        assert_eq!(fast.0, exact.0);
        assert_eq!(fast.1, exact.1);
        assert_eq!(fast.2, exact.2);
        assert_eq!(fast.3, exact.3);
        assert_eq!(fast.4, exact.4);
    }

    /// x * sigmoid(g), wide inputs: random raws at 2F (and at F) against the exact stage.
    #[test]
    fn sigmul_fast_equals_exact_random() {
        let t = table();
        let mut rng = Rng(0x5161);
        let (mut decided, mut undecided) = (0u64, 0u64);
        for i in 0..400_000u32 {
            let r = rng.next();
            let bits = if r % 8 == 0 { 34 } else { 26 };
            let draw = |rng: &mut Rng| -> i64 { let v = rng.next(); let m = ((v >> 8) & ((1u64 << bits) - 1)) as i64; if v & 1 == 1 { -m } else { m } };
            let (x, g) = (draw(&mut rng), draw(&mut rng));
            let (f, wide) = if i % 2 == 0 { (20, true) } else { (10, false) };
            let (x, g) = if wide { (x, g) } else { ((x >> 8) as i32 as i64, (g >> 8) as i32 as i64) };
            let [got] = sigmul_fast_n::<1>(t, [x as i128], [g as i128], f, 10);
            let want = std::panic::catch_unwind(|| if wide { sigmul_exact::<i32, i64, 10>(x, g, 20) } else { sigmul_exact::<i32, i32, 10>(x as i32, g as i32, 10) }).ok();
            match (got, want) {
                (Some(v), Some(w)) => {
                    assert_eq!(v, w as i128, "sigmul x {x} g {g} f {f}");
                    decided += 1;
                }
                (Some(v), None) => assert!(i32::try_from(v).is_err()),
                (None, _) => undecided += 1,
            }
        }
        // Undecided here are genuine near-ties: a wide x that is an exact half
        // unit at the output (1 in 1024 of these random raws) times a gate
        // within 2^-40 of 1; the exact stage settles them (checked above).
        eprintln!("sigmul: {decided} decided equal to the exact stage, {undecided} undecided (near-ties)");
        assert!(undecided * 1_000 < decided);
    }

    /// Wide-input softcap: random raws at 2F against the exact stage.
    #[test]
    fn softcap_wide_fast_equals_exact_random() {
        let t = table();
        let mut rng = Rng(0x50F7);
        let cap = 30i32 << 10;
        let c = Cap::new(cap as i128).unwrap();
        let (mut decided, mut undecided) = (0u64, 0u64);
        for _ in 0..400_000 {
            let r = rng.next();
            let m = ((r >> 8) & ((1u64 << 30) - 1)) as i64;
            let x = if r & 1 == 1 { -m } else { m };
            match softcap_fast(t, x as i128, 20, &c, 10) {
                Some(v) => {
                    assert_eq!(v, softcap_exact::<i32, i64, 10>(x, 20, cap) as i128, "wide softcap x {x}");
                    decided += 1;
                }
                None => undecided += 1,
            }
        }
        eprintln!("softcap wide Q.10: {decided} decided equal, {undecided} undecided");
        assert!(undecided * 10_000 < decided);
    }
}
