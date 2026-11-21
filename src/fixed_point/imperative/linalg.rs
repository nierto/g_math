//! The 32-bit word's long dot product path (the bounded unchecked AVX2 sum
//! behind `Storage::dot_acc_fast`) and the stagnation constant of the
//! iterative decompositions. Everything else that lived here is generic
//! over the format in `kernels` since the width design.

/// Below this length the per-term check is cheaper than the bound pass
/// (measured: the two cross between 16 and 32 elements). The length test
/// is made inline by `Storage::dot_acc_fast`; only longer vectors reach the
/// out-of-line [`long_dot_acc_i32`].
#[cfg(target_arch = "x86_64")]
pub(crate) const BOUNDED_DOT_MIN_LEN: usize = 32;

/// The 32-bit word's fast dot accumulator for `BOUNDED_DOT_MIN_LEN` or more
/// elements: the bounded unchecked AVX2 sum where AVX2 is present and the
/// bound holds, `None` otherwise (the generic checked loop then runs; it
/// panics where the sum really overflows). Out of line, so the short-vector
/// path of the callers stays as small as it was.
#[cfg(target_arch = "x86_64")]
#[inline(never)]
pub(crate) fn long_dot_acc_i32(a: &[i32], b: &[i32]) -> Option<i64> {
    if !std::is_x86_feature_detected!("avx2") {
        return None;
    }
    // SAFETY: AVX2 was just detected.
    unsafe { bounded_dot_avx2(a, b) }
}

/// Largest magnitude in a slice (`|i32::MIN|` is `2^31`, hence unsigned).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn max_abs_avx2(x: &[i32]) -> u32 {
    let mut m = 0u32;
    for &v in x {
        m = m.max(v.unsigned_abs());
    }
    m
}

/// `sum a_i * b_i` in i64 with no per-term check, for inputs whose bound
/// `len * max|a| * max|b| < 2^63` proves that no partial sum can leave i64 in
/// any order of summation: the same integer as the checked loop, in a form
/// the compiler vectorises. `None` when the bound does not hold (the caller
/// then takes the checked loop, which panics where the sum really overflows).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn bounded_dot_avx2(a: &[i32], b: &[i32]) -> Option<i64> {
    let bound = (a.len() as u128) * (max_abs_avx2(a) as u128) * (max_abs_avx2(b) as u128);
    if bound >= 1u128 << 63 {
        return None;
    }
    Some(unchecked_dot_avx2(a, b))
}

/// `sum a_i * b_i` in i64, wrapping adds: the caller has proven the bound.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn unchecked_dot_avx2(a: &[i32], b: &[i32]) -> i64 {
    let mut acc = 0i64;
    for (&x, &y) in a.iter().zip(b) {
        acc = acc.wrapping_add((x as i64) * (y as i64));
    }
    acc
}


/// `x >> f` (logical, 64-bit lanes) with an immediate count: the register-count
/// form is two micro-ops on Intel cores. `f` is a constant at every call site.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn srli_epi64_dyn(x: std::arch::x86_64::__m256i, f: u32) -> std::arch::x86_64::__m256i {
    use std::arch::x86_64::*;
    macro_rules! arms { ($($k:literal),+) => { match f { $($k => _mm256_srli_epi64::<$k>(x),)+ _ => _mm256_srl_epi64(x, _mm_cvtsi32_si128(f as i32)) } } }
    arms!(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)
}

/// One position of the softmax value mix on the i32 formats, `F <= 15`:
/// `num[d] += (e * v[d] + 2^(F - 1)) >> F + 2^(62 - F)` for every `d`,
/// wrapping. The caller removes the `2^(62 - F)` per position afterwards.
///
/// Why the offset: AVX2 has a signed 32 x 32 -> 64-bit multiply
/// (`vpmuldq`) but no 64-bit arithmetic right shift. With
/// `0 <= e <= 2^(2F) <= 2^30` and `|v| <= 2^31` the product is within
/// `2^61`, so `p + 2^(F - 1) + 2^62` lies in `(0, 2^63)`: its logical shift
/// is the floor, and `2^62` is a multiple of `2^F`, so the shifted value is
/// exactly `floor((p + 2^(F - 1)) / 2^F) + 2^(62 - F)`. The sums wrap
/// modulo 2^64; the true sum fits i64 (the caller's bound), so subtracting
/// the offsets restores it exactly: the same integers as the scalar loop.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
pub(crate) unsafe fn mix_row_offset_avx2(e: i64, v: &[i32], num: &mut [i64], f: u32) {
    use std::arch::x86_64::*;
    debug_assert!((0..=1i64 << 30).contains(&e) && (1..=15).contains(&f) && num.len() == v.len());
    let n = v.len();
    let ev = _mm256_set1_epi64x(e);
    let add = (1i64 << 62) + (1i64 << (f - 1));
    let bias = _mm256_set1_epi64x(add);
    let mut d = 0usize;
    while d + 8 <= n {
        // SAFETY: d + 8 <= n = v.len() = num.len(); unaligned loads and stores.
        let x = _mm256_loadu_si256(v.as_ptr().add(d) as *const __m256i);
        // products of elements 0, 2, 4, 6 and of 1, 3, 5, 7 (low 32 bits of
        // each 64-bit lane, sign-extended)
        let pe = _mm256_mul_epi32(x, ev);
        let po = _mm256_mul_epi32(_mm256_srli_epi64::<32>(x), ev);
        let qe = srli_epi64_dyn(_mm256_add_epi64(pe, bias), f);
        let qo = srli_epi64_dyn(_mm256_add_epi64(po, bias), f);
        // back to element order: [0 1 4 5] and [2 3 6 7], then [0 1 2 3] [4 5 6 7]
        let a = _mm256_unpacklo_epi64(qe, qo);
        let b = _mm256_unpackhi_epi64(qe, qo);
        let lo = _mm256_permute2x128_si256::<0x20>(a, b);
        let hi = _mm256_permute2x128_si256::<0x31>(a, b);
        let p = num.as_mut_ptr().add(d) as *mut __m256i;
        _mm256_storeu_si256(p, _mm256_add_epi64(_mm256_loadu_si256(p), lo));
        _mm256_storeu_si256(p.add(1), _mm256_add_epi64(_mm256_loadu_si256(p.add(1)), hi));
        d += 8;
    }
    while d < n {
        let q = ((e * v[d] as i64 + add) as u64 >> f) as i64;
        num[d] = num[d].wrapping_add(q);
        d += 1;
    }
}

/// [`mix_row_offset_avx2`] for four positions at once: each numerator chunk
/// is loaded and stored once per four rows instead of once per row. The
/// same offset terms, so the same integers.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
pub(crate) unsafe fn mix_rows4_offset_avx2(e: [i64; 4], v: [&[i32]; 4], num: &mut [i64], f: u32) {
    use std::arch::x86_64::*;
    let n = num.len();
    debug_assert!(e.iter().all(|e| (0..=1i64 << 30).contains(e)) && (1..=15).contains(&f) && v.iter().all(|r| r.len() == n));
    let ev = [_mm256_set1_epi64x(e[0]), _mm256_set1_epi64x(e[1]), _mm256_set1_epi64x(e[2]), _mm256_set1_epi64x(e[3])];
    let add = (1i64 << 62) + (1i64 << (f - 1));
    let bias = _mm256_set1_epi64x(add);
    let mut d = 0usize;
    while d + 8 <= n {
        // SAFETY: d + 8 <= n = every row's length = num.len(); unaligned access.
        let mut se = _mm256_setzero_si256();
        let mut so = _mm256_setzero_si256();
        for k in 0..4 {
            let x = _mm256_loadu_si256(v[k].as_ptr().add(d) as *const __m256i);
            let pe = _mm256_mul_epi32(x, ev[k]);
            let po = _mm256_mul_epi32(_mm256_srli_epi64::<32>(x), ev[k]);
            se = _mm256_add_epi64(se, srli_epi64_dyn(_mm256_add_epi64(pe, bias), f));
            so = _mm256_add_epi64(so, srli_epi64_dyn(_mm256_add_epi64(po, bias), f));
        }
        let a = _mm256_unpacklo_epi64(se, so);
        let b = _mm256_unpackhi_epi64(se, so);
        let lo = _mm256_permute2x128_si256::<0x20>(a, b);
        let hi = _mm256_permute2x128_si256::<0x31>(a, b);
        let p = num.as_mut_ptr().add(d) as *mut __m256i;
        _mm256_storeu_si256(p, _mm256_add_epi64(_mm256_loadu_si256(p), lo));
        _mm256_storeu_si256(p.add(1), _mm256_add_epi64(_mm256_loadu_si256(p.add(1)), hi));
        d += 8;
    }
    while d < n {
        for k in 0..4 {
            let q = ((e[k] * v[k][d] as i64 + add) as u64 >> f) as i64;
            num[d] = num[d].wrapping_add(q);
        }
        d += 1;
    }
}

/// Consecutive non-improving iterations after which an iterative decomposition
/// is taken to sit at its precision floor.
pub(crate) const STAGNATION_SWEEPS: usize = 5;
