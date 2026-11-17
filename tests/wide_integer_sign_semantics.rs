//! Sign semantics of the wide integer types: the permanent gate.
//!
//! Two silent-wrong-value bugs found on 2026-08-29 while building certified
//! decimal intervals, both in code that had never been asked the question:
//!
//! - `D256::Sub` and `D512::Sub` never propagated a borrow. The borrow test
//!   compared a `u128` against `u128::MAX`, which is never true, so
//!   `0 - 1` produced `2^64 - 1` (positive) and `2^128 - 1` lost a word.
//!   Used by the UGOD decimal tier 5 and 6 subtraction arms
//!   (`decimal_addition.rs`); whether canonical literals can reach those
//!   arms today is a separate question (they currently parse such magnitudes
//!   into the binary domain), so this gate asks the types directly.
//! - `Ord` on `I1024`, `I2048`, `D256` and `D512` reversed the word
//!   comparison for two negative operands. Two's-complement negatives already
//!   order correctly as unsigned words (`-1 = 0xFF..F > 0xFF..E = -2`), so
//!   the reversal made `-2 > -1`. `I256` and `I512` had been corrected
//!   earlier; the other four had not. At Q256.256 the compute
//!   tier is `I1024`, so any `<`, `min` or `max` between two negative
//!   compute-tier values there was wrong.
//!
//! Format-independent: these types exist in every build.
//!
//! Also here since 0.6.1: `I2048::checked_add`, added for the Q256.256
//! fused quadratic-form accumulator, which had no checked addition.

use g_math::fixed_point::domains::decimal_fixed::{D256, D512};
use g_math::fixed_point::{I1024, I2048, I256, I512};
use std::cmp::Ordering;

// ----------------------------------------------------------------------------
// Ordering of negatives, same sign, including values beyond i128
// ----------------------------------------------------------------------------

macro_rules! ord_gate {
    ($name:ident, $t:ty, $from:expr, $neg:expr, $shl:expr) => {
        #[test]
        fn $name() {
            let from = $from;
            let neg = $neg;
            let shl = $shl;
            // small negatives
            assert_eq!(from(-2).cmp(&from(-1)), Ordering::Less, "-2 < -1");
            assert_eq!(from(-1).cmp(&from(-2)), Ordering::Greater, "-1 > -2");
            assert_eq!(from(-1).cmp(&from(-1)), Ordering::Equal);
            assert_eq!(from(-1).cmp(&from(0)), Ordering::Less);
            assert_eq!(from(i128::MIN).cmp(&from(i128::MIN + 1)), Ordering::Less);
            // negatives beyond i128: -(2^130) < -(2^129) < -(2^128 + 1)
            let big = |e: u32| neg(shl(from(1), e));
            assert_eq!(big(130).cmp(&big(129)), Ordering::Less, "-(2^130) < -(2^129)");
            assert_eq!(big(129).cmp(&big(130)), Ordering::Greater);
            assert_eq!(big(128).cmp(&from(-1)), Ordering::Less, "-(2^128) < -1");
            // a sort of mixed signs comes out ascending
            let mut v = vec![from(3), big(129), from(-5), from(0), big(130), from(-1), shl(from(1), 129)];
            v.sort();
            let expect = vec![big(130), big(129), from(-5), from(-1), from(0), from(3), shl(from(1), 129)];
            assert_eq!(v, expect, "mixed-sign sort order");
            // min/max, the operation that exposed the bug
            let a = big(129);
            let b = big(130);
            assert_eq!(std::cmp::min(a, b), b);
            assert_eq!(std::cmp::max(a, b), a);
        }
    };
}

ord_gate!(i256_orders_negatives, I256, |x: i128| I256::from_i128(x), |v: I256| -v, |v: I256, e: u32| v << (e as usize));
ord_gate!(i512_orders_negatives, I512, |x: i128| I512::from_i128(x), |v: I512| -v, |v: I512, e: u32| v << (e as usize));
ord_gate!(i1024_orders_negatives, I1024, |x: i128| I1024::from_i128(x), |v: I1024| -v, |v: I1024, e: u32| v << (e as usize));
ord_gate!(i2048_orders_negatives, I2048, |x: i128| I2048::from_i128(x), |v: I2048| -v, |v: I2048, e: u32| v << (e as usize));
ord_gate!(
    d256_orders_negatives,
    D256,
    |x: i128| D256::from_i128(x),
    |v: D256| g_math::fixed_point::domains::decimal_fixed::negate_d256(v),
    |v: D256, e: u32| d256_shl(v, e)
);
ord_gate!(
    d512_orders_negatives,
    D512,
    |x: i128| D512::from_i128(x),
    |v: D512| g_math::fixed_point::domains::decimal_fixed::negate_d512(v),
    |v: D512, e: u32| d512_shl(v, e)
);

/// Left shift by whole words plus bits, for the decimal types, which have no
/// shift operator: multiply by 2^e via repeated doubling of the words.
fn d256_shl(v: D256, e: u32) -> D256 {
    let mut w = v.words;
    for _ in 0..e {
        let mut carry = 0u64;
        for word in w.iter_mut() {
            let next = *word >> 63;
            *word = (*word << 1) | carry;
            carry = next;
        }
    }
    D256::from_words(w)
}

fn d512_shl(v: D512, e: u32) -> D512 {
    let mut w = v.words;
    for _ in 0..e {
        let mut carry = 0u64;
        for word in w.iter_mut() {
            let next = *word >> 63;
            *word = (*word << 1) | carry;
            carry = next;
        }
    }
    D512::from_words(w)
}

// ----------------------------------------------------------------------------
// Subtraction with borrow across words
// ----------------------------------------------------------------------------

#[test]
fn d256_sub_propagates_borrow() {
    // 0 - 1 = -1 (all ones), not 2^64 - 1
    assert_eq!(D256::from_i128(0) - D256::from_i128(1), D256::from_i128(-1));
    // 2^64 - 1: borrow from word 1 into word 0
    let two64 = D256::from_words([0, 1, 0, 0]);
    assert_eq!(two64 - D256::from_i128(1), D256::from_words([u64::MAX, 0, 0, 0]));
    // 2^128 - 1: borrow chain through two words
    let two128 = D256::from_words([0, 0, 1, 0]);
    assert_eq!(two128 - D256::from_i128(1), D256::from_words([u64::MAX, u64::MAX, 0, 0]));
    // 2^192 - 2^64: borrow lands on word 1
    let two192 = D256::from_words([0, 0, 0, 1]);
    assert_eq!(two192 - two64, D256::from_words([0, u64::MAX, u64::MAX, 0]));
    // a - b == -(b - a)
    let a = D256::from_words([5, 7, 0, 0]);
    let b = D256::from_words([9, 2, 0, 0]);
    assert_eq!(a - b, g_math::fixed_point::domains::decimal_fixed::negate_d256(b - a));
    // consistency with i128 on in-range operands, both signs
    for (x, y) in [(5i128, 9i128), (-5, 9), (5, -9), (-5, -9), (i128::MIN + 1, 1), (0, i128::MAX)] {
        assert_eq!(D256::from_i128(x) - D256::from_i128(y), D256::from_i128(x - y), "{x} - {y}");
    }
}

#[test]
fn d512_sub_propagates_borrow() {
    assert_eq!(D512::from_i128(0) - D512::from_i128(1), D512::from_i128(-1));
    let two64 = D512::from_words([0, 1, 0, 0, 0, 0, 0, 0]);
    assert_eq!(two64 - D512::from_i128(1), D512::from_words([u64::MAX, 0, 0, 0, 0, 0, 0, 0]));
    let two256 = D512::from_words([0, 0, 0, 0, 1, 0, 0, 0]);
    assert_eq!(
        two256 - D512::from_i128(1),
        D512::from_words([u64::MAX, u64::MAX, u64::MAX, u64::MAX, 0, 0, 0, 0])
    );
    for (x, y) in [(5i128, 9i128), (-5, 9), (5, -9), (-5, -9), (i128::MIN + 1, 1), (0, i128::MAX)] {
        assert_eq!(D512::from_i128(x) - D512::from_i128(y), D512::from_i128(x - y), "{x} - {y}");
    }
}

/// `I2048::checked_add` (0.6.1, the scientific quadratic-form accumulator):
/// the same sign test `I512` and `I1024` use, at both ends of the range and
/// on in-range operands of every sign combination.
#[test]
fn i2048_checked_add_detects_signed_overflow() {
    let max = I2048::max_value();
    let min = I2048::min_value();
    let one = I2048::one();
    assert_eq!(max.checked_add(one), None, "max + 1 overflows");
    assert_eq!(min.checked_add(-one), None, "min - 1 overflows");
    assert_eq!(max.checked_add(-one), Some(max - one));
    assert_eq!(min.checked_add(one), Some(min + one));
    assert_eq!(max.checked_add(min), Some(-one));
    let big = I2048::from_i128(i128::MAX);
    assert_eq!(big.checked_add(big), Some(big + big));
    assert_eq!((-big).checked_add(-big), Some(-(big + big)));
    for (x, y) in [(5i128, 9i128), (-5, 9), (5, -9), (-5, -9), (i128::MIN + 1, 1), (0, i128::MAX)] {
        assert_eq!(I2048::from_i128(x).checked_add(I2048::from_i128(y)), Some(I2048::from_i128(x + y)), "{x} + {y}");
    }
    // the top half of the range: 2^2046 + 2^2046 = 2^2047 does not fit
    let mut w = [0u64; 32];
    w[31] = 1 << 62;
    let half_max = I2048::from_words(w);
    assert_eq!(half_max.checked_add(half_max), None, "2^2046 + 2^2046 overflows");
    assert_eq!(half_max.checked_add(-half_max), Some(I2048::zero()));
}

// ----------------------------------------------------------------------------
// Multiplication of signed operands, against exact BigInt products
// ----------------------------------------------------------------------------
//
// 0.7.2: `I256::checked_mul` returned `None` for every product with a
// negative operand, and `I256::mul_to_i512` (public) returned the product of
// the two's-complement bit patterns read as unsigned numbers: right modulo
// 2^256, wrong in the high half. Reported by a consumer against 0.7.0 and
// 0.7.1 (`-1 * 1`, `-3 * 5`, `(-2^100)^2` all `None`). `I512::mul_to_i1024`
// and `I1024::mul_to_i2048` had the same unsigned reading. The gate asks
// every widening and wrapping multiply of the four types, on every sign
// combination, for the exact product.

use num_bigint::BigInt;

/// Two's-complement words (little-endian) as a signed integer.
fn words_to_big(w: &[u64]) -> BigInt {
    let mut v = BigInt::from(0u8);
    for word in w.iter().rev() {
        v = (v << 64u32) + BigInt::from(*word);
    }
    if (w[w.len() - 1] as i64) < 0 {
        v -= BigInt::from(1u8) << (64 * w.len() as u32);
    }
    v
}

/// `v` reduced modulo 2^(64 n), read back as a signed n-word integer.
fn wrap_big(v: &BigInt, n: usize) -> BigInt {
    let m = BigInt::from(1u8) << (64 * n as u32);
    let mut r = v % &m;
    if r < BigInt::from(0u8) {
        r += &m;
    }
    if r >= (BigInt::from(1u8) << (64 * n as u32 - 1)) {
        r -= &m;
    }
    r
}

fn fits(v: &BigInt, n: usize) -> bool {
    wrap_big(v, n) == *v
}

/// Operands of n words: zero, plus and minus one, the extremes, powers of
/// two around the half-word boundaries, and seeded pseudo-random values of
/// every length and both signs.
fn operands(n: usize) -> Vec<Vec<u64>> {
    let mut out: Vec<Vec<u64>> = Vec::new();
    let small = |x: i128| {
        let mut w = vec![if x < 0 { u64::MAX } else { 0 }; n];
        w[0] = x as u64;
        w[1] = (x >> 64) as u64;
        w
    };
    for x in [0i128, 1, -1, 3, -3, 5, -5, i64::MAX as i128, i64::MIN as i128, i128::MAX, i128::MIN] {
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
    for e in [100usize, 127, 128, 129, 64 * n / 2 - 1, 64 * n / 2, 64 * n - 2] {
        let mut w = vec![0u64; n];
        w[e / 64] = 1 << (e % 64);
        let neg = words_to_big(&w) * BigInt::from(-1);
        let nw = (0..n)
            .map(|i| {
                let r = wrap_big(&neg, n);
                let m = BigInt::from(1u8) << (64 * n as u32);
                let u = if r < BigInt::from(0u8) { r + m } else { r };
                let s: BigInt = (u >> (64 * i as u32)) & BigInt::from(u64::MAX);
                u64::try_from(s).unwrap()
            })
            .collect::<Vec<_>>();
        out.push(w);
        out.push(nw);
    }
    let mut s: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut next = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s
    };
    for len in 1..=n {
        for _ in 0..6 {
            let neg = next() & 1 == 1;
            let mut w = vec![0u64; n];
            for word in w.iter_mut().take(len) {
                *word = next();
            }
            if len == n {
                w[n - 1] &= i64::MAX as u64;
            }
            if neg {
                // two's-complement negation
                let mut carry = 1u64;
                for word in w.iter_mut() {
                    let (v, c) = (!*word).overflowing_add(carry);
                    *word = v;
                    carry = c as u64;
                }
            }
            out.push(w);
        }
    }
    out
}

macro_rules! mul_gate {
    ($name:ident, $t:ident, $n:expr, $wide:ident, $wn:expr, $to_wide:ident, $mul_wide:ident) => {
        #[test]
        fn $name() {
            let ops = operands($n);
            let mut checked = 0usize;
            for a in &ops {
                for b in &ops {
                    let x = $t { words: a.clone().try_into().unwrap() };
                    let y = $t { words: b.clone().try_into().unwrap() };
                    let exact = words_to_big(a) * words_to_big(b);
                    let (ea, eb) = (words_to_big(a), words_to_big(b));
                    // wrapping product
                    assert_eq!(words_to_big(&(x * y).words), wrap_big(&exact, $n), "{} * {} (wrapping)", ea, eb);
                    // widening products, exact
                    assert_eq!(words_to_big(&x.$to_wide(y).words), exact, "{}::{}({}, {})", stringify!($t), stringify!($to_wide), ea, eb);
                    assert_eq!(words_to_big(&x.$mul_wide(&y).words), exact, "{}::{}({}, {})", stringify!($t), stringify!($mul_wide), ea, eb);
                    checked += 1;
                }
            }
            assert!(checked > 1000, "corpus too small: {checked}");
            // the reported cases
            let one = $t::from_i128(1);
            let neg_one = $t::from_i128(-1);
            assert_eq!(neg_one.$to_wide(one), $wide::from_i128(-1));
            assert_eq!(one.$to_wide(neg_one), $wide::from_i128(-1));
            assert_eq!(neg_one.$to_wide(neg_one), $wide::from_i128(1));
            assert_eq!($t::from_i128(-3).$to_wide($t::from_i128(5)), $wide::from_i128(-15));
            let _ = $wn;
        }
    };
}

mul_gate!(i256_multiplies_signed_operands, I256, 4, I512, 8, mul_to_i512, multiply_i256);
mul_gate!(i512_multiplies_signed_operands, I512, 8, I1024, 16, mul_to_i1024, multiply_i512);

#[test]
fn i1024_and_i2048_multiply_signed_operands() {
    let ops = operands(16);
    for a in &ops {
        for b in &ops {
            let x = I1024 { words: a.clone().try_into().unwrap() };
            let y = I1024 { words: b.clone().try_into().unwrap() };
            let exact = words_to_big(a) * words_to_big(b);
            assert_eq!(words_to_big(&(x * y).words), wrap_big(&exact, 16), "I1024 {} * {}", words_to_big(a), words_to_big(b));
            assert_eq!(words_to_big(&x.mul_to_i2048(y).words), exact, "I1024::mul_to_i2048({}, {})", words_to_big(a), words_to_big(b));
            // I2048 wrapping product, operands sign-extended from I1024
            let xx = I2048::from_i1024(x);
            let yy = I2048::from_i1024(y);
            assert_eq!(words_to_big(&(xx * yy).words), exact, "I2048 {} * {}", words_to_big(a), words_to_big(b));
        }
    }
    let ops = operands(32);
    for a in &ops {
        for b in &ops {
            let x = I2048 { words: a.clone().try_into().unwrap() };
            let y = I2048 { words: b.clone().try_into().unwrap() };
            let exact = words_to_big(a) * words_to_big(b);
            assert_eq!(words_to_big(&(x * y).words), wrap_big(&exact, 32), "I2048 {} * {}", words_to_big(a), words_to_big(b));
        }
    }
}

/// `checked_mul` is `Some` exactly when the product fits, for every sign.
#[test]
fn checked_mul_is_some_exactly_when_the_product_fits() {
    let ops = operands(4);
    let mut some_neg = 0usize;
    for a in &ops {
        for b in &ops {
            let x = I256 { words: a.clone().try_into().unwrap() };
            let y = I256 { words: b.clone().try_into().unwrap() };
            let exact = words_to_big(a) * words_to_big(b);
            let got = x.checked_mul(y);
            if fits(&exact, 4) {
                let v = got.unwrap_or_else(|| panic!("I256 {} * {} fits but checked_mul is None", words_to_big(a), words_to_big(b)));
                assert_eq!(words_to_big(&v.words), exact);
                if exact < BigInt::from(0u8) {
                    some_neg += 1;
                }
            } else {
                assert!(got.is_none(), "I256 {} * {} overflows but checked_mul is Some", words_to_big(a), words_to_big(b));
            }
        }
    }
    assert!(some_neg > 50, "negative in-range products checked: {some_neg}");
    let ops = operands(8);
    for a in &ops {
        for b in &ops {
            let x = I512 { words: a.clone().try_into().unwrap() };
            let y = I512 { words: b.clone().try_into().unwrap() };
            let exact = words_to_big(a) * words_to_big(b);
            let got = x.checked_mul(y);
            if fits(&exact, 8) {
                assert_eq!(got.map(|v| words_to_big(&v.words)), Some(exact.clone()), "I512 {} * {}", words_to_big(a), words_to_big(b));
            } else {
                assert!(got.is_none(), "I512 {} * {} overflows but checked_mul is Some", words_to_big(a), words_to_big(b));
            }
        }
    }
    // the reported cases
    let i = |x: i128| I256::from_i128(x);
    assert_eq!(i(-1).checked_mul(i(1)), Some(i(-1)));
    assert_eq!(i(1).checked_mul(i(-1)), Some(i(-1)));
    assert_eq!(i(-1).checked_mul(i(-1)), Some(i(1)));
    assert_eq!(i(-3).checked_mul(i(5)), Some(i(-15)));
    let p100 = -(i(1) << 100usize);
    assert_eq!(p100.checked_mul(p100), Some(i(1) << 200usize));
}

/// The decimal wide integers and the i128 widening product, same question.
#[test]
fn decimal_wide_integers_multiply_signed_operands() {
    use g_math::fixed_point::domains::decimal_fixed::d256::{mul_d256_to_d512, mul_i128_to_d256};
    use g_math::fixed_point::domains::binary_fixed::mul_i128_to_i256;
    let ops = operands(4);
    for a in &ops {
        for b in &ops {
            let x = D256::from_words(a.clone().try_into().unwrap());
            let y = D256::from_words(b.clone().try_into().unwrap());
            let exact = words_to_big(a) * words_to_big(b);
            let (ea, eb) = (words_to_big(a), words_to_big(b));
            assert_eq!(words_to_big(&(x * y).words), wrap_big(&exact, 4), "D256 {} * {}", ea, eb);
            assert_eq!(words_to_big(&x.mul_to_d512(y).words), exact, "D256::mul_to_d512({}, {})", ea, eb);
            assert_eq!(words_to_big(&mul_d256_to_d512(x, y).words), exact, "mul_d256_to_d512({}, {})", ea, eb);
            let xx = D512::from_words({
                let mut w = [if (a[3] as i64) < 0 { u64::MAX } else { 0 }; 8];
                w[..4].copy_from_slice(a);
                w
            });
            let yy = D512::from_words({
                let mut w = [if (b[3] as i64) < 0 { u64::MAX } else { 0 }; 8];
                w[..4].copy_from_slice(b);
                w
            });
            assert_eq!(words_to_big(&(xx * yy).words), exact, "D512 {} * {}", ea, eb);
        }
    }
    let ops = operands(8);
    for a in &ops {
        for b in &ops {
            let x = D512::from_words(a.clone().try_into().unwrap());
            let y = D512::from_words(b.clone().try_into().unwrap());
            let exact = words_to_big(a) * words_to_big(b);
            assert_eq!(words_to_big(&(x * y).words), wrap_big(&exact, 8), "D512 {} * {}", words_to_big(a), words_to_big(b));
        }
    }
    let smalls = [0i128, 1, -1, 3, -5, i64::MAX as i128, i64::MIN as i128, i128::MAX, i128::MIN, i128::MIN + 1, 1 << 100, -(1 << 100), 0x1234_5678_9ABC_DEF0_0FED_CBA9_8765_4321];
    for &a in &smalls {
        for &b in &smalls {
            let exact = BigInt::from(a) * BigInt::from(b);
            assert_eq!(words_to_big(&mul_i128_to_i256(a, b).words), exact, "mul_i128_to_i256({a}, {b})");
            assert_eq!(words_to_big(&mul_i128_to_d256(a, b).words), exact, "mul_i128_to_d256({a}, {b})");
        }
    }
}
