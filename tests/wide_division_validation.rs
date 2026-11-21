//! Division of the wide integer types against exact BigInt arithmetic.
//!
//! 0.7.3 replaced the bit-at-a-time long division of `I256`, `I512`,
//! `I1024`, `I2048`, `D256` and `D512` with word-at-a-time long division
//! (Knuth's Algorithm D, `binary_fixed/word_div.rs`). Integer quotient and
//! remainder are unique, so the new division must return exactly what
//! num-bigint returns: quotient truncated toward zero, remainder with the
//! dividend's sign. This gate asks every public division entry point of the
//! six types on edge operands (every pair) and on seeded operands built to
//! reach the quotient-estimate corrections: divisors whose top word is
//! exactly `2^63` or all ones, dividends whose top word equals the
//! divisor's, quotients of one word and of every length.
//!
//! Defects the old loops had and the new division fixes (each confirmed on
//! crates.io 0.7.2 by `.gates/div_old_probe`):
//! - `I1024`: a dividend of `I1024::MIN` returned quotient 0 and remainder
//!   `MIN` for every divisor (the negated dividend stayed negative and
//!   compared below the divisor).
//! - Every type: `i128::MIN / -1` on operands that fit `i128` panicked in the
//!   `i128` shortcut, although `2^127` fits the wide type.
//! - `MIN / -1` of the type itself now panics as native integers do (the
//!   old loops returned `MIN` for `I256`/`I512`/`D256`/`D512`).
//!
//! Format-independent: these types exist in every build.

use g_math::fixed_point::domains::binary_fixed::i1024::divmod_i1024;
use g_math::fixed_point::domains::binary_fixed::i2048::i2048_divmod;
use g_math::fixed_point::domains::binary_fixed::i256::divmod_i256_by_i256;
use g_math::fixed_point::domains::binary_fixed::i512::divmod_i512_by_i512;
use g_math::fixed_point::domains::decimal_fixed::d256::divmod_d256_by_d256;
use g_math::fixed_point::domains::decimal_fixed::d512::divmod_d512_by_d512;
use g_math::fixed_point::domains::decimal_fixed::{D256, D512};
use g_math::fixed_point::{I1024, I2048, I256, I512};
use num_bigint::BigInt;

/// Two's-complement words (little-endian) as a signed integer.
fn to_big(w: &[u64]) -> BigInt {
    let mut v = BigInt::from(0u8);
    for word in w.iter().rev() {
        v = (v << 64u32) + BigInt::from(*word);
    }
    if (w[w.len() - 1] as i64) < 0 {
        v -= BigInt::from(1u8) << (64 * w.len() as u32);
    }
    v
}

/// Small deterministic generator (xorshift64*).
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
}

fn negate(w: &mut [u64]) {
    let mut carry = 1u64;
    for x in w.iter_mut() {
        let (s, c) = (!*x).overflowing_add(carry);
        *x = s;
        carry = c as u64;
    }
}

/// A non-negative value of `bits` significant bits (top bit set) in `n` words.
fn magnitude(rng: &mut Rng, n: usize, bits: usize) -> Vec<u64> {
    let mut w = vec![0u64; n];
    for i in 0..(bits + 63) / 64 {
        w[i] = rng.next();
    }
    let top = (bits - 1) / 64;
    let b = (bits - 1) % 64;
    w[top] &= if b == 63 { u64::MAX } else { (1u64 << (b + 1)) - 1 };
    w[top] |= 1u64 << b;
    w
}

/// Edge operands in `n` words: small values, the extremes, powers of two and
/// all-ones words at word boundaries, both signs.
fn edges(n: usize) -> Vec<Vec<u64>> {
    let mut out = Vec::new();
    let small = |x: i128| {
        let mut w = vec![if x < 0 { u64::MAX } else { 0 }; n];
        w[0] = x as u64;
        w[1] = (x >> 64) as u64;
        w
    };
    for x in [0i128, 1, -1, 2, -2, 3, -7, u64::MAX as i128, -(u64::MAX as i128), 1 << 64, (1 << 64) + 1, i128::MAX, i128::MIN, i128::MIN + 1] {
        out.push(small(x));
    }
    let mut max = vec![u64::MAX; n];
    max[n - 1] = i64::MAX as u64;
    let mut min = vec![0u64; n];
    min[n - 1] = 1 << 63;
    let mut min1 = min.clone();
    min1[0] = 1;
    out.push(max);
    out.push(min);
    out.push(min1);
    for e in [63usize, 64, 127, 128, 191, 64 * n / 2 - 1, 64 * n / 2, 64 * n - 65, 64 * n - 2] {
        let mut p = vec![0u64; n];
        p[e / 64] = 1 << (e % 64);
        let mut ones = vec![0u64; n];
        for k in 0..=e {
            ones[k / 64] |= 1 << (k % 64);
        }
        for w in [p, ones] {
            let mut neg = w.clone();
            negate(&mut neg);
            out.push(w);
            out.push(neg);
        }
    }
    out
}

/// Dividend/divisor pairs aimed at the quotient-estimate corrections.
fn seeded_pairs(n: usize, seed: u64, count: usize) -> Vec<(Vec<u64>, Vec<u64>)> {
    let mut rng = Rng(seed);
    let bits = 64 * n - 1; // magnitudes that keep the sign bit clear
    let mut out = Vec::with_capacity(count);
    for i in 0..count {
        let vb = 1 + (rng.next() as usize) % bits;
        let mut v = magnitude(&mut rng, n, vb);
        let top = (vb - 1) / 64;
        match i % 6 {
            // divisor top word exactly 2^63 (normalisation shift 0) or all ones
            0 if vb % 64 == 0 => { v[top] = 1 << 63; }
            1 => { for k in 0..top { v[k] = u64::MAX; } }
            _ => {}
        }
        let ub = vb + (rng.next() as usize) % (bits - vb + 1);
        let mut u = magnitude(&mut rng, n, ub);
        if i % 6 == 2 && ub > 64 && vb > 64 {
            // dividend top word equal to the divisor's top word (aligned)
            let shift_words = (ub - vb) / 64;
            if (ub - vb) % 64 == 0 && top + shift_words < n {
                u[top + shift_words] = v[top];
            }
        }
        if rng.next() % 2 == 1 { negate(&mut u); }
        if rng.next() % 2 == 1 { negate(&mut v); }
        out.push((u, v));
    }
    out
}

fn check(name: &str, a: &[u64], b: &[u64], q: &[u64], r: &[u64]) {
    let (ba, bb) = (to_big(a), to_big(b));
    let (eq, er) = (&ba / &bb, &ba % &bb);
    assert_eq!(to_big(q), eq, "{name}: quotient of {ba} / {bb}");
    assert_eq!(to_big(r), er, "{name}: remainder of {ba} % {bb}");
}

/// Skip a zero divisor and the overflowing MIN / -1.
fn skip(a: &[u64], b: &[u64]) -> bool {
    let n = a.len();
    let zero = b.iter().all(|&w| w == 0);
    let minus_one = b.iter().all(|&w| w == u64::MAX);
    let a_min = a[n - 1] == 1 << 63 && a[..n - 1].iter().all(|&w| w == 0);
    zero || (minus_one && a_min)
}

macro_rules! division_gate {
    ($name:ident, $n:expr, $seed:expr, $mk:expr, $words:expr, [$($f:expr),+ $(,)?]) => {
        #[test]
        fn $name() {
            let mk = $mk;
            let words = $words;
            let mut pairs: Vec<(Vec<u64>, Vec<u64>)> = Vec::new();
            let e = edges($n);
            for a in &e { for b in &e { pairs.push((a.clone(), b.clone())); } }
            pairs.extend(seeded_pairs($n, $seed, 40_000));
            let mut checked = 0usize;
            for (a, b) in &pairs {
                if skip(a, b) { continue; }
                let (x, y) = (mk(a), mk(b));
                $(
                    let (label, (q, r)) = $f(x, y);
                    check(label, a, b, &words(q), &words(r));
                )+
                checked += 1;
            }
            assert!(checked > 40_000);
        }
    };
}

division_gate!(i256_division_is_exact, 4, 0x256, |w: &Vec<u64>| I256 { words: w[..].try_into().unwrap() }, |v: I256| v.words.to_vec(), [
    |x: I256, y: I256| ("divmod_i256_by_i256", divmod_i256_by_i256(x, y)),
    |x: I256, y: I256| ("I256 / and %", (x / y, x % y)),
]);
division_gate!(i512_division_is_exact, 8, 0x512, |w: &Vec<u64>| I512 { words: w[..].try_into().unwrap() }, |v: I512| v.words.to_vec(), [
    |x: I512, y: I512| ("divmod_i512_by_i512", divmod_i512_by_i512(x, y)),
    |x: I512, y: I512| ("I512 / and %", (x / y, x % y)),
]);
division_gate!(i1024_division_is_exact, 16, 0x1024, |w: &Vec<u64>| I1024 { words: w[..].try_into().unwrap() }, |v: I1024| v.words.to_vec(), [
    |x: I1024, y: I1024| ("divmod_i1024", divmod_i1024(x, y)),
    |x: I1024, y: I1024| ("I1024 / and %", (x / y, x % y)),
]);
division_gate!(i2048_division_is_exact, 32, 0x2048, |w: &Vec<u64>| I2048::from_words(w[..].try_into().unwrap()), |v: I2048| v.words.to_vec(), [
    |x: I2048, y: I2048| ("i2048_divmod", i2048_divmod(x, y)),
]);
division_gate!(d256_division_is_exact, 4, 0xD256, |w: &Vec<u64>| D256 { words: w[..].try_into().unwrap() }, |v: D256| v.words.to_vec(), [
    |x: D256, y: D256| ("divmod_d256_by_d256", divmod_d256_by_d256(x, y)),
    |x: D256, y: D256| ("D256 / and %", (x / y, x % y)),
]);
division_gate!(d512_division_is_exact, 8, 0xD512, |w: &Vec<u64>| D512 { words: w[..].try_into().unwrap() }, |v: D512| v.words.to_vec(), [
    |x: D512, y: D512| ("divmod_d512_by_d512", divmod_d512_by_d512(x, y)),
    |x: D512, y: D512| ("D512 / and %", (x / y, x % y)),
]);

// The overflowing quotient MIN / -1 panics, as for native integers.
macro_rules! min_by_minus_one {
    ($name:ident, $n:expr, $f:expr) => {
        #[test]
        #[should_panic(expected = "attempt to divide with overflow")]
        fn $name() {
            let mut min = vec![0u64; $n];
            min[$n - 1] = 1 << 63;
            let minus_one = vec![u64::MAX; $n];
            $f(&min, &minus_one);
        }
    };
}
min_by_minus_one!(i256_min_by_minus_one_panics, 4, |a: &Vec<u64>, b: &Vec<u64>| { let _ = divmod_i256_by_i256(I256 { words: a[..].try_into().unwrap() }, I256 { words: b[..].try_into().unwrap() }); });
min_by_minus_one!(i512_min_by_minus_one_panics, 8, |a: &Vec<u64>, b: &Vec<u64>| { let _ = divmod_i512_by_i512(I512 { words: a[..].try_into().unwrap() }, I512 { words: b[..].try_into().unwrap() }); });
min_by_minus_one!(i1024_min_by_minus_one_panics, 16, |a: &Vec<u64>, b: &Vec<u64>| { let _ = divmod_i1024(I1024 { words: a[..].try_into().unwrap() }, I1024 { words: b[..].try_into().unwrap() }); });
min_by_minus_one!(i2048_min_by_minus_one_panics, 32, |a: &Vec<u64>, b: &Vec<u64>| { let _ = i2048_divmod(I2048::from_words(a[..].try_into().unwrap()), I2048::from_words(b[..].try_into().unwrap())); });
min_by_minus_one!(d256_min_by_minus_one_panics, 4, |a: &Vec<u64>, b: &Vec<u64>| { let _ = divmod_d256_by_d256(D256 { words: a[..].try_into().unwrap() }, D256 { words: b[..].try_into().unwrap() }); });
min_by_minus_one!(d512_min_by_minus_one_panics, 8, |a: &Vec<u64>, b: &Vec<u64>| { let _ = divmod_d512_by_d512(D512 { words: a[..].try_into().unwrap() }, D512 { words: b[..].try_into().unwrap() }); });

/// `i128::MIN / -1` on operands that fit `i128`: `2^127` fits every wide type
/// (the `i128` shortcut used to overflow and panic).
#[test]
fn i128_min_by_minus_one_fits_the_wide_types() {
    let expect = BigInt::from(1u8) << 127u32;
    assert_eq!(to_big(&(I256::from_i128(i128::MIN) / I256::from_i128(-1)).words), expect);
    assert_eq!(to_big(&(I512::from_i128(i128::MIN) / I512::from_i128(-1)).words), expect);
    assert_eq!(to_big(&(I1024::from_i128(i128::MIN) / I1024::from_i128(-1)).words), expect);
    assert_eq!(to_big(&i2048_divmod(I2048::from_i128(i128::MIN), I2048::from_i128(-1)).0.words), expect);
    assert_eq!(to_big(&divmod_d256_by_d256(D256::from_i128(i128::MIN), D256::from_i128(-1)).0.words), expect);
    assert_eq!(to_big(&divmod_d512_by_d512(D512::from_i128(i128::MIN), D512::from_i128(-1)).0.words), expect);
}
