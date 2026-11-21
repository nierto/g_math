//! Speed-only releases against the published 0.7.2: the SAME outcome for
//! every input, nothing closer, nothing different.
//!
//! 0.7.3 changes how wide integers divide (word-at-a-time long division), how
//! the square-root engines stop, and how the decimal engine rescales. None of
//! that may change a returned integer: a changed output would open a new
//! numerics epoch for consumers that pin outputs. This gate links crates.io
//! 0.7.2 beside this tree and compares, per format, the Debug rendering of
//! every outcome (value, `Err` variant, or panic) of:
//! - the arithmetic operators and their `try_` forms, every transcendental and
//!   its `try_` form, `inv_sqrt`, `pow`, `atan2`, `Display`, at `Fixed<i32, 10>`,
//!   `Fixed<i32, 16>`, `Fixed<i32, 24>`, Q32.32, Q64.64, Q128.128, Q256.256;
//! - `DecimalFixed<D>` arithmetic and transcendentals across the decimal tiers;
//! - the canonical pipeline on binary and decimal literals;
//! - `g_math::wide` (the Q64.64 inference gates);
//! - the fused operations and the decompositions at the inference formats.
//!
//! Run: `cd differential && cargo test --release --test differential_072_validation`
//! (`GMATH_PROFILE` only matters to the 0.6.7 gate beside this one.)

use std::fmt::Debug;
use std::panic::{catch_unwind, AssertUnwindSafe};

use g_math as new;
use g_math_072 as old;

// ---- outcomes -----------------------------------------------------------------

fn silence_panics() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| std::panic::set_hook(Box::new(|_| {})));
}

/// The Debug rendering of a result, or "panic".
fn run<T: Debug>(f: impl FnOnce() -> T) -> String {
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(v) => format!("{v:?}"),
        Err(_) => "panic".to_string(),
    }
}

struct Tally {
    name: String,
    cases: usize,
    panics: usize,
    errs: usize,
    diffs: Vec<String>,
}

impl Tally {
    fn new(name: &str) -> Self { Tally { name: name.to_string(), cases: 0, panics: 0, errs: 0, diffs: Vec::new() } }
    fn same(&mut self, what: &str, input: &dyn Debug, old: String, new: String) {
        self.cases += 1;
        if old == "panic" { self.panics += 1; }
        if old.starts_with("Err") || old.contains("Err(") { self.errs += 1; }
        if old != new && self.diffs.len() < 20 {
            self.diffs.push(format!("{what} {input:?}: 0.7.2 {old} | now {new}"));
        } else if old != new {
            self.diffs.push(String::new());
        }
    }
    fn finish(self) {
        eprintln!("{}: {} cases ({} panic, {} with an Err), {} differ", self.name, self.cases, self.panics, self.errs, self.diffs.len());
        // a gate where nearly everything fails on both sides would pass vacuously
        assert!(self.panics + self.errs < self.cases * 3 / 4, "{}: too few ordinary outcomes", self.name);
        assert!(self.diffs.is_empty(), "{}: {} of {} outcomes differ from 0.7.2:\n{}", self.name, self.diffs.len(), self.cases,
            self.diffs.iter().filter(|s| !s.is_empty()).cloned().collect::<Vec<_>>().join("\n"));
    }
}

// ---- inputs ---------------------------------------------------------------------

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn below(&mut self, n: u64) -> u64 { self.next() % n }
}

/// A two's-complement raw of `bits` storage bits as little-endian words (at
/// least one), magnitude below `2^mag_bits`, either sign.
fn raw_words(rng: &mut Rng, bits: u32, mag_bits: u32) -> Vec<u64> {
    let n = ((bits + 63) / 64) as usize;
    let mag_bits = mag_bits.min(bits - 1);
    let len = rng.below(mag_bits as u64 + 1) as u32;
    let mut w = vec![0u64; n];
    for i in 0..((len + 63) / 64) as usize {
        w[i] = rng.next();
    }
    if len < 64 * n as u32 {
        let (top, b) = ((len / 64) as usize, len % 64);
        if top < n {
            w[top] &= (1u64 << b).wrapping_sub(1);
            for x in w.iter_mut().skip(top + 1) { *x = 0; }
        }
    }
    if rng.below(2) == 1 { negate(&mut w); }
    w
}

fn negate(w: &mut [u64]) {
    let mut carry = 1u64;
    for x in w.iter_mut() {
        let (s, c) = (!*x).overflowing_add(carry);
        *x = s;
        carry = c as u64;
    }
}

/// `v * 2^shift` (small `v`) as words of a `bits`-bit storage.
fn scaled(v: i64, shift: u32, bits: u32) -> Vec<u64> {
    let n = ((bits + 63) / 64) as usize;
    let mut w = vec![0u64; n];
    let m = v.unsigned_abs() as u128;
    let (word, b) = ((shift / 64) as usize, shift % 64);
    let lo = m << b;
    if word < n { w[word] = lo as u64; }
    if word + 1 < n { w[word + 1] = (lo >> 64) as u64; }
    if word + 2 < n && b > 0 { w[word + 2] = (m >> (128 - b)) as u64; }
    if v < 0 { negate(&mut w); }
    w
}

/// Special raws for a format of `bits` storage bits and `f` fractional bits:
/// zero, units, one and its neighbours, halves, small integers, the extremes.
fn specials(bits: u32, f: u32) -> Vec<Vec<u64>> {
    let n = ((bits + 63) / 64) as usize;
    let mut out = Vec::new();
    for v in [0i64, 1, -1, 2, -2, 3] { out.push(scaled(v, 0, bits)); }
    for v in [1i64, -1, 2, -2, 3, 10, -10] { out.push(scaled(v, f, bits)); }
    if f >= 1 { for v in [1i64, -1, 3] { out.push(scaled(v, f - 1, bits)); } }
    for v in [1i64, -1] {
        // one plus and minus a unit
        let mut a = scaled(v, f, bits);
        let mut b = a.clone();
        add_small(&mut a, 1);
        add_small(&mut b, -1);
        out.push(a);
        out.push(b);
    }
    let mut max = vec![u64::MAX; n];
    let mut min = vec![0u64; n];
    if bits % 64 == 0 {
        max[n - 1] = i64::MAX as u64;
        min[n - 1] = 1 << 63;
    } else {
        // 32-bit storage in one word, sign-extended
        max[0] = i32::MAX as u64;
        min[0] = i32::MIN as i64 as u64;
    }
    out.push(max);
    out.push(min);
    out
}

fn add_small(w: &mut [u64], d: i64) {
    let mut carry = d as i128;
    for x in w.iter_mut() {
        let s = *x as i128 + carry;
        *x = s as u64;
        carry = s >> 64;
        if carry == 0 { break; }
    }
}

/// Inputs for a format: specials, then values below 2^(f + 2) (the
/// transcendentals' interesting range), then values of any size.
fn inputs(seed: u64, bits: u32, f: u32, count: usize) -> Vec<Vec<u64>> {
    let mut rng = Rng(seed);
    let mut out = specials(bits, f);
    for i in 0..count {
        let mag = if i % 2 == 0 { f + 2 } else { bits - 1 };
        out.push(raw_words(&mut rng, bits, mag));
    }
    out
}

// ---- Fixed<S, F> ------------------------------------------------------------------

macro_rules! fixed_unary {
    ($t:ident, $xs:ident, $mk_old:ident, $mk_new:ident, $O:ty, $N:ty; $($m:ident),+ $(,)?) => {$(
        for w in &$xs {
            let (a, b): ($O, $N) = ($mk_old(w), $mk_new(w));
            $t.same(stringify!($m), w, run(|| a.$m()), run(|| b.$m()));
        }
    )+};
}

macro_rules! fixed_gate {
    ($test:ident, $bits:expr, $f:expr, $count:expr, $old:ty, $new:ty, $mk_old:expr, $mk_new:expr) => {
        #[test]
        fn $test() {
            silence_panics();
            type O = $old;
            type N = $new;
            let mk_old = $mk_old;
            let mk_new = $mk_new;
            let xs = inputs(0x7203 ^ $f as u64 ^ (($bits as u64) << 8), $bits, $f, $count);
            let mut t = Tally::new(stringify!($test));
            fixed_unary!(t, xs, mk_old, mk_new, O, N; exp, ln, sqrt, sin, cos, sincos, tan, atan, asin, acos, sinh, cosh, sinhcosh, tanh, asinh, acosh, atanh, inv_sqrt);
            fixed_unary!(t, xs, mk_old, mk_new, O, N; try_exp, try_ln, try_sqrt, try_sin, try_cos, try_sincos, try_tan, try_atan, try_asin, try_acos, try_sinh, try_cosh,
                   try_sinhcosh, try_tanh, try_asinh, try_acosh, try_atanh, try_inv_sqrt, abs, try_neg, to_int);
            for w in &xs {
                let (a, b): (O, N) = (mk_old(w), mk_new(w));
                t.same("Display", w, run(|| a.to_string()), run(|| b.to_string()));
            }
            // binary: pairs from the same pool
            let mut rng = Rng(0xB1 ^ $f as u64);
            let pairs: Vec<(usize, usize)> = (0..xs.len() * 2).map(|_| (rng.below(xs.len() as u64) as usize, rng.below(xs.len() as u64) as usize)).collect();
            for &(i, j) in &pairs {
                let (a, b): (O, N) = (mk_old(&xs[i]), mk_new(&xs[i]));
                let (c, d): (O, N) = (mk_old(&xs[j]), mk_new(&xs[j]));
                let input = (&xs[i], &xs[j]);
                t.same("+", &input, run(|| a + c), run(|| b + d));
                t.same("-", &input, run(|| a - c), run(|| b - d));
                t.same("*", &input, run(|| a * c), run(|| b * d));
                t.same("/", &input, run(|| a / c), run(|| b / d));
                t.same("try_mul", &input, run(|| a.try_mul(c)), run(|| b.try_mul(d)));
                t.same("try_div", &input, run(|| a.try_div(c)), run(|| b.try_div(d)));
                t.same("pow", &input, run(|| a.pow(c)), run(|| b.pow(d)));
                t.same("try_pow", &input, run(|| a.try_pow(c)), run(|| b.try_pow(d)));
                t.same("atan2", &input, run(|| a.atan2(c)), run(|| b.atan2(d)));
                t.same("try_atan2", &input, run(|| a.try_atan2(c)), run(|| b.try_atan2(d)));
            }
            t.finish();
        }
    };
}

fn w_i32(w: &[u64]) -> i32 { w[0] as i32 }
fn w_i64(w: &[u64]) -> i64 { w[0] as i64 }
fn w_i128(w: &[u64]) -> i128 { (w[0] as u128 | (w[1] as u128) << 64) as i128 }

fixed_gate!(fixed_i32_10_same_as_0_7_2, 32, 10, 3000, old::fixed_point::Fixed<i32, 10>, new::fixed_point::Fixed<i32, 10>,
    |w: &Vec<u64>| old::fixed_point::Fixed::<i32, 10>::from_raw(w_i32(w)), |w: &Vec<u64>| new::fixed_point::Fixed::<i32, 10>::from_raw(w_i32(w)));
fixed_gate!(fixed_i32_16_same_as_0_7_2, 32, 16, 3000, old::fixed_point::Fixed<i32, 16>, new::fixed_point::Fixed<i32, 16>,
    |w: &Vec<u64>| old::fixed_point::Fixed::<i32, 16>::from_raw(w_i32(w)), |w: &Vec<u64>| new::fixed_point::Fixed::<i32, 16>::from_raw(w_i32(w)));
fixed_gate!(fixed_i32_24_same_as_0_7_2, 32, 24, 3000, old::fixed_point::Fixed<i32, 24>, new::fixed_point::Fixed<i32, 24>,
    |w: &Vec<u64>| old::fixed_point::Fixed::<i32, 24>::from_raw(w_i32(w)), |w: &Vec<u64>| new::fixed_point::Fixed::<i32, 24>::from_raw(w_i32(w)));
fixed_gate!(q32_32_same_as_0_7_2, 64, 32, 3000, old::fixed_point::Q32_32, new::fixed_point::Q32_32,
    |w: &Vec<u64>| old::fixed_point::Q32_32::from_raw(w_i64(w)), |w: &Vec<u64>| new::fixed_point::Q32_32::from_raw(w_i64(w)));
fixed_gate!(q64_64_same_as_0_7_2, 128, 64, 2000, old::fixed_point::Q64_64, new::fixed_point::Q64_64,
    |w: &Vec<u64>| old::fixed_point::Q64_64::from_raw(w_i128(w)), |w: &Vec<u64>| new::fixed_point::Q64_64::from_raw(w_i128(w)));
fixed_gate!(q128_128_same_as_0_7_2, 256, 128, 800, old::fixed_point::Q128_128, new::fixed_point::Q128_128,
    |w: &Vec<u64>| old::fixed_point::Q128_128::from_raw(old::fixed_point::I256 { words: w[..].try_into().unwrap() }),
    |w: &Vec<u64>| new::fixed_point::Q128_128::from_raw(new::fixed_point::I256 { words: w[..].try_into().unwrap() }));
fixed_gate!(q256_256_same_as_0_7_2, 512, 256, 150, old::fixed_point::Q256_256, new::fixed_point::Q256_256,
    |w: &Vec<u64>| old::fixed_point::Q256_256::from_raw(old::fixed_point::I512 { words: w[..].try_into().unwrap() }),
    |w: &Vec<u64>| new::fixed_point::Q256_256::from_raw(new::fixed_point::I512 { words: w[..].try_into().unwrap() }));

// ---- DecimalFixed<D> ---------------------------------------------------------------

macro_rules! dec_unary {
    ($t:ident, $raws:ident, $O:ty, $N:ty; $($m:ident),+ $(,)?) => {$(
        for &r in &$raws {
            let (a, b) = (<$O>::from_raw(r), <$N>::from_raw(r));
            $t.same(stringify!($m), &r, run(|| a.$m()), run(|| b.$m()));
        }
    )+};
}

macro_rules! decimal_gate {
    ($test:ident, [$($d:literal),+], $count:expr) => {
        #[test]
        fn $test() {
            silence_panics();
            let mut t = Tally::new(stringify!($test));
            $({
                type O = old::fixed_point::DecimalFixed<$d>;
                type N = new::fixed_point::DecimalFixed<$d>;
                let mut rng = Rng(0xDEC0 ^ $d as u64);
                // raws: specials, values below 10^(D+2), any size
                let ten_d = 10i128.pow($d as u32);
                let mut raws: Vec<i128> = vec![0, 1, -1, ten_d, -ten_d, ten_d / 2, ten_d - 1, ten_d + 1, 2 * ten_d, -3 * ten_d, i128::MAX, i128::MIN, i128::MIN + 1];
                for i in 0..$count {
                    let cap: u32 = if i % 2 == 0 { (128 - (ten_d.leading_zeros())).min(124) + 4 } else { 127 };
                    let bits = rng.below(cap as u64 + 1) as u32;
                    let m = ((rng.next() as u128) << 64 | rng.next() as u128) >> (128 - bits.max(1)) >> (bits == 0) as u32;
                    let v = m as i128;
                    raws.push(if rng.below(2) == 1 { v.wrapping_neg() } else { v });
                }
                let label = concat!("D", stringify!($d));
                dec_unary!(t, raws, O, N; try_exp, try_ln, try_sqrt, try_sin, try_cos, try_sincos, try_tan, try_atan, try_asin, try_acos,
                       try_sinh, try_cosh, try_sinhcosh, try_tanh, try_asinh, try_acosh, try_atanh, try_neg);
                for &r in &raws {
                    let (a, b) = (O::from_raw(r), N::from_raw(r));
                    t.same(label, &r, run(|| a.to_string()), run(|| b.to_string()));
                }
                for k in 0..raws.len() * 2 {
                    let (i, j) = (rng.below(raws.len() as u64) as usize, (k * 7919) % raws.len());
                    let (a, b) = (O::from_raw(raws[i]), N::from_raw(raws[i]));
                    let (c, d) = (O::from_raw(raws[j]), N::from_raw(raws[j]));
                    let input = (raws[i], raws[j]);
                    t.same("try_mul", &input, run(|| a.try_mul(c)), run(|| b.try_mul(d)));
                    t.same("try_div", &input, run(|| a.try_div(c)), run(|| b.try_div(d)));
                    t.same("try_add", &input, run(|| a.try_add(c)), run(|| b.try_add(d)));
                    t.same("try_mul_div", &input, run(|| a.try_mul_div(c, a)), run(|| b.try_mul_div(d, b)));
                    t.same("try_atan2", &input, run(|| a.try_atan2(c)), run(|| b.try_atan2(d)));
                }
            })+
            t.finish();
        }
    };
}

decimal_gate!(decimal_word_tiers_same_as_0_7_2, [0, 2, 6, 9], 1500);
decimal_gate!(decimal_i256_tiers_same_as_0_7_2, [10, 12, 17, 19], 1000);
decimal_gate!(decimal_i512_tiers_same_as_0_7_2, [20, 28, 37, 38], 500);

// ---- canonical pipeline -------------------------------------------------------------

const LITERALS: &[&str] = &[
    "0", "1", "-1", "0.5", "-0.5", "2", "10", "0.1", "1.5", "2.5", "3.14159", "0.999", "-0.75", "7.389056098930650227",
    "0.12345678901234567890123456789012345", "1.4142135623730950488016887242096980786", "123456.789", "-42.0625",
    "0.00000000000000000001", "36.7", "-36.7", "1e-5", "0x1F", "1/3", "-2/7",
];

macro_rules! canon_unary {
    ($t:ident, $ofmt:ty, $fmt:ty; $($m:ident),+ $(,)?) => {$(
        for s in LITERALS {
            let o = run(|| old::canonical::gmath_parse::<$ofmt>(s).map(|e| old::canonical::evaluate(&e.$m()).map(|v| v.to_string())));
            let n = run(|| new::canonical::gmath_parse::<$fmt>(s).map(|e| new::canonical::evaluate(&e.$m()).map(|v| v.to_string())));
            $t.same(stringify!($m), s, o, n);
        }
    )+};
}

macro_rules! canonical_gate {
    ($test:ident, $fmt:ty, $ofmt:ty) => {
        #[test]
        fn $test() {
            silence_panics();
            let mut t = Tally::new(stringify!($test));
            canon_unary!(t, $ofmt, $fmt; exp, ln, sqrt, sin, cos, tan, atan, asin, acos, sinh, cosh, tanh, asinh, acosh, atanh);
            for a in LITERALS {
                for b in LITERALS {
                    let o = run(|| -> Result<_, old::fixed_point::OverflowDetected> {
                        let (x, y) = (old::canonical::gmath_parse::<$ofmt>(a)?, old::canonical::gmath_parse::<$ofmt>(b)?);
                        Ok([
                            old::canonical::evaluate(&(x.clone() / y.clone())).map(|v| v.to_string()),
                            old::canonical::evaluate(&(x.clone() * y.clone())).map(|v| v.to_string()),
                            old::canonical::evaluate(&x.pow(y)).map(|v| v.to_string()),
                        ])
                    });
                    let n = run(|| -> Result<_, new::fixed_point::OverflowDetected> {
                        let (x, y) = (new::canonical::gmath_parse::<$fmt>(a)?, new::canonical::gmath_parse::<$fmt>(b)?);
                        Ok([
                            new::canonical::evaluate(&(x.clone() / y.clone())).map(|v| v.to_string()),
                            new::canonical::evaluate(&(x.clone() * y.clone())).map(|v| v.to_string()),
                            new::canonical::evaluate(&x.pow(y)).map(|v| v.to_string()),
                        ])
                    });
                    t.same("div/mul/pow", &(a, b), o, n);
                }
            }
            t.finish();
        }
    };
}

canonical_gate!(canonical_i32_10_same_as_0_7_2, new::fixed_point::Fixed<i32, 10>, old::fixed_point::Fixed<i32, 10>);
canonical_gate!(canonical_q64_64_same_as_0_7_2, new::fixed_point::Q64_64, old::fixed_point::Q64_64);
canonical_gate!(canonical_q128_128_same_as_0_7_2, new::fixed_point::Q128_128, old::fixed_point::Q128_128);
canonical_gate!(canonical_q256_256_same_as_0_7_2, new::fixed_point::Q256_256, old::fixed_point::Q256_256);

// ---- g_math::wide ---------------------------------------------------------------------

#[test]
fn wide_q64_same_as_0_7_2() {
    silence_panics();
    let mut t = Tally::new("wide_q64");
    let mut rng = Rng(0x0064);
    let mut xs: Vec<i128> = vec![0, 1, -1, 1 << 64, -(1 << 64), 1 << 63, i128::MAX, i128::MIN, 41 << 64, -(41 << 64)];
    for i in 0..20_000 {
        let bits = rng.below(if i % 2 == 0 { 70 } else { 128 }) as u32;
        let m = (((rng.next() as u128) << 64 | rng.next() as u128) >> (127 - bits.min(126))) as i128;
        xs.push(if rng.below(2) == 1 { -m } else { m });
    }
    for &x in &xs {
        t.same("exp_q64", &x, run(|| old::wide::exp_q64(x)), run(|| new::wide::exp_q64(x)));
        t.same("ln_q64", &x, run(|| old::wide::ln_q64(x)), run(|| new::wide::ln_q64(x)));
        t.same("sincos_q64", &x, run(|| old::wide::sincos_q64(x)), run(|| new::wide::sincos_q64(x)));
        t.same("sigmoid_q64", &x, run(|| old::wide::sigmoid_q64(x)), run(|| new::wide::sigmoid_q64(x)));
        t.same("softplus_q64", &x, run(|| old::wide::softplus_q64(x)), run(|| new::wide::softplus_q64(x)));
        t.same("silu_q64", &x, run(|| old::wide::silu_q64(x)), run(|| new::wide::silu_q64(x)));
        t.same("sqrt_q64", &x, run(|| old::wide::sqrt_q64(x)), run(|| new::wide::sqrt_q64(x)));
        for f in [10u32, 20, 40, 64] {
            t.same("sqrt_q64_to", &(x, f), run(|| old::wide::sqrt_q64_to(x, f)), run(|| new::wide::sqrt_q64_to(x, f)));
            t.same("narrow_q64", &(x, f), run(|| old::wide::narrow_q64(x, f)), run(|| new::wide::narrow_q64(x, f)));
        }
    }
    for k in 0..xs.len() {
        let (a, b, d) = (xs[k], xs[(k * 31 + 7) % xs.len()], xs[(k * 17 + 3) % xs.len()]);
        t.same("mul_div_floor", &(a, b, d), run(|| old::wide::mul_div_floor(a, b, d)), run(|| new::wide::mul_div_floor(a, b, d)));
        t.same("mul_div_nearest", &(a, b, d), run(|| old::wide::mul_div_nearest(a, b, d)), run(|| new::wide::mul_div_nearest(a, b, d)));
        let (ua, ub, ud) = (a as u128, b as u128, d as u128);
        t.same("mul_div_floor_u128", &(ua, ub, ud), run(|| old::wide::mul_div_floor_u128(ua, ub, ud)), run(|| new::wide::mul_div_floor_u128(ua, ub, ud)));
    }
    for s in LITERALS {
        for f in [10u32, 64, 127] {
            t.same("try_from_str", &(s, f), run(|| old::wide::try_from_str(s, f)), run(|| new::wide::try_from_str(s, f)));
        }
        t.same("try_ratio_from_str", s, run(|| old::wide::try_ratio_from_str(s)), run(|| new::wide::try_ratio_from_str(s)));
    }
    t.finish();
}

// ---- fused operations and decompositions ----------------------------------------------

macro_rules! vector_gate {
    ($test:ident, $bits:expr, $f:expr, $old:ty, $new:ty, $mk_old:expr, $mk_new:expr) => {
        #[test]
        fn $test() {
            silence_panics();
            use old::fixed_point::imperative::{decompose as od, derived as odv, fused as of};
            use new::fixed_point::imperative::{decompose as nd, derived as ndv, fused as nf};
            type O = $old;
            type N = $new;
            let (mk_old, mk_new) = ($mk_old, $mk_new);
            let mut t = Tally::new(stringify!($test));
            let mut rng = Rng(0xF0 ^ $f as u64);
            for case in 0..400usize {
                let len = 1 + rng.below(48) as usize;
                let mag = if case % 3 == 0 { $f + 6 } else { $f + 2 };
                let ws: Vec<Vec<u64>> = (0..len).map(|_| raw_words(&mut rng, $bits, mag)).collect();
                let gs: Vec<Vec<u64>> = (0..len).map(|_| raw_words(&mut rng, $bits, $f + 3)).collect();
                let (xo, xn): (Vec<O>, Vec<N>) = (ws.iter().map(|w| mk_old(w)).collect(), ws.iter().map(|w| mk_new(w)).collect());
                let (go, gn): (Vec<O>, Vec<N>) = (gs.iter().map(|w| mk_old(w)).collect(), gs.iter().map(|w| mk_new(w)).collect());
                let eps = (rng.next() >> 20) as i128;
                let input = (&ws, &gs, eps);
                t.same("silu", &input, run(|| xo.iter().map(|&x| of::silu(x)).collect::<Vec<_>>()), run(|| xn.iter().map(|&x| nf::silu(x)).collect::<Vec<_>>()));
                t.same("softmax", &input, run(|| of::softmax(&xo)), run(|| nf::softmax(&xn)));
                t.same("entropy", &input, run(|| of::softmax(&xo).map(|w| of::entropy(&w))), run(|| nf::softmax(&xn).map(|w| nf::entropy(&w))));
                t.same("rms_norm", &input, run(|| of::rms_norm(&xo, &go, eps)), run(|| nf::rms_norm(&xn, &gn, eps)));
                t.same("rms_norm_factor_eps_wide", &input, run(|| of::rms_norm_factor_eps_wide(&xo, eps)), run(|| nf::rms_norm_factor_eps_wide(&xn, eps)));
                t.same("sigmoid_mul_slice", &input, run(|| of::sigmoid_mul_slice(&xo, &go)), run(|| nf::sigmoid_mul_slice(&xn, &gn)));
                t.same("dot", &input, run(|| of::dot(&xo, &go)), run(|| nf::dot(&xn, &gn)));
                t.same("sqrt_sum_sq", &input, run(|| of::sqrt_sum_sq(&xo)), run(|| nf::sqrt_sum_sq(&xn)));
                t.same("inv_sqrt_sum_sq", &input, run(|| of::inv_sqrt_sum_sq(&xo)), run(|| nf::inv_sqrt_sum_sq(&xn)));
                t.same("euclidean_distance", &input, run(|| of::euclidean_distance(&xo, &go)), run(|| nf::euclidean_distance(&xn, &gn)));
                // attention: one query of `dim` against len / dim keys
                let dim = 1 + rng.below(8) as usize;
                let keys = len / dim;
                if keys > 0 {
                    let q = (&go[..dim], &gn[..dim]);
                    let (ko, kn) = (&xo[..keys * dim], &xn[..keys * dim]);
                    t.same("dot_many", &input, run(|| of::dot_many(q.0, ko, dim)), run(|| nf::dot_many(q.1, kn, dim)));
                    let (so, sn) = (&go[..keys], &gn[..keys]);
                    t.same("softmax_mix_flat_values", &input, run(|| of::softmax_mix_flat_values(so, ko, dim)), run(|| nf::softmax_mix_flat_values(sn, kn, dim)));
                    t.same("softmax_mix_flat", &input, run(|| of::softmax_mix_flat(so, ko, dim)), run(|| nf::softmax_mix_flat(sn, kn, dim)));
                }
                // square systems from the first n*n values
                let n = (1..=5).rev().find(|n| n * n <= len).unwrap();
                if case % 2 == 0 {
                    let (ao, an) = (old::fixed_point::FixedMatrix::from_slice(n, n, &xo[..n * n]), new::fixed_point::FixedMatrix::from_slice(n, n, &xn[..n * n]));
                    let (bo, bn) = (old::fixed_point::FixedVector::from_slice(&go[..n]), new::fixed_point::FixedVector::from_slice(&gn[..n]));
                    // a symmetric, diagonally dominant twin for Cholesky / eigen
                    let (so, sn) = (&ao + &ao.transpose(), &an + &an.transpose());
                    let bump_o = old::fixed_point::FixedMatrix::<O>::identity(n) * O::from_int((8 * n) as i32);
                    let bump_n = new::fixed_point::FixedMatrix::<N>::identity(n) * N::from_int((8 * n) as i32);
                    let (po, pn) = (&so + &bump_o, &sn + &bump_n);
                    t.same("lu solve", &input, run(|| od::lu_decompose(&ao).and_then(|d| d.solve(&bo))), run(|| nd::lu_decompose(&an).and_then(|d| d.solve(&bn))));
                    t.same("determinant", &input, run(|| odv::determinant(&ao)), run(|| ndv::determinant(&an)));
                    t.same("inverse", &input, run(|| odv::inverse(&ao)), run(|| ndv::inverse(&an)));
                    t.same("qr solve", &input, run(|| od::qr_decompose(&ao).and_then(|d| d.solve(&bo))), run(|| nd::qr_decompose(&an).and_then(|d| d.solve(&bn))));
                    t.same("cholesky solve", &input, run(|| od::cholesky_decompose(&po).and_then(|d| d.solve(&bo))), run(|| nd::cholesky_decompose(&pn).and_then(|d| d.solve(&bn))));
                    t.same("cholesky det", &input, run(|| od::cholesky_decompose(&po).map(|d| d.determinant())), run(|| nd::cholesky_decompose(&pn).map(|d| d.determinant())));
                    t.same("eigen_symmetric", &input, run(|| od::eigen_symmetric(&so)), run(|| nd::eigen_symmetric(&sn)));
                    t.same("svd", &input, run(|| od::svd_decompose(&ao)), run(|| nd::svd_decompose(&an)));
                    t.same("frobenius_norm", &input, run(|| odv::frobenius_norm(&ao)), run(|| ndv::frobenius_norm(&an)));
                }
            }
            t.finish();
        }
    };
}

vector_gate!(fused_linalg_i32_10_same_as_0_7_2, 32, 10, old::fixed_point::Fixed<i32, 10>, new::fixed_point::Fixed<i32, 10>,
    |w: &Vec<u64>| old::fixed_point::Fixed::<i32, 10>::from_raw(w_i32(w)), |w: &Vec<u64>| new::fixed_point::Fixed::<i32, 10>::from_raw(w_i32(w)));
vector_gate!(fused_linalg_i32_16_same_as_0_7_2, 32, 16, old::fixed_point::Fixed<i32, 16>, new::fixed_point::Fixed<i32, 16>,
    |w: &Vec<u64>| old::fixed_point::Fixed::<i32, 16>::from_raw(w_i32(w)), |w: &Vec<u64>| new::fixed_point::Fixed::<i32, 16>::from_raw(w_i32(w)));
vector_gate!(fused_linalg_q32_32_same_as_0_7_2, 64, 32, old::fixed_point::Q32_32, new::fixed_point::Q32_32,
    |w: &Vec<u64>| old::fixed_point::Q32_32::from_raw(w_i64(w)), |w: &Vec<u64>| new::fixed_point::Q32_32::from_raw(w_i64(w)));
vector_gate!(fused_linalg_q64_64_same_as_0_7_2, 128, 64, old::fixed_point::Q64_64, new::fixed_point::Q64_64,
    |w: &Vec<u64>| old::fixed_point::Q64_64::from_raw(w_i128(w)), |w: &Vec<u64>| new::fixed_point::Q64_64::from_raw(w_i128(w)));
vector_gate!(fused_linalg_q128_128_same_as_0_7_2, 256, 128, old::fixed_point::Q128_128, new::fixed_point::Q128_128,
    |w: &Vec<u64>| old::fixed_point::Q128_128::from_raw(old::fixed_point::I256 { words: w[..].try_into().unwrap() }),
    |w: &Vec<u64>| new::fixed_point::Q128_128::from_raw(new::fixed_point::I256 { words: w[..].try_into().unwrap() }));
