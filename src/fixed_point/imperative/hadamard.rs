//! The Walsh-Hadamard transform on integer words, exact.
//!
//! `y = H x` with `H[r][c] = (-1)^popcount(r & c)` (the Sylvester order),
//! computed in place by `log2(n)` butterfly passes of additions and
//! subtractions. No rounding happens: the result is the exact integer
//! vector, `log2(n)` bits wider than the input at most. A normalised
//! transform `H / sqrt(n)` is this one followed by the caller's division,
//! which a caller usually folds into a later rounding (for `n = 4^k` it is
//! the exact shift by `k`).
//!
//! For a `Fixed<S, F>` slice transform the raw words
//! ([`Fixed::raw_slice_mut`](crate::fixed_point::Fixed::raw_slice_mut)):
//! the transform is linear with integer coefficients, so the raws of the
//! result are the result's raws at the same `F`.

use crate::fixed_point::core_types::errors::OverflowDetected;
use crate::fixed_point::width::WideInt;

/// The Walsh-Hadamard transform of `x` in place, exact.
///
/// `x` becomes `H x`, `H[r][c] = (-1)^popcount(r & c)`. The words are any
/// signed integer the crate knows (`i32`, `i64`, `i128`, `I256` ..).
///
/// # Errors
/// `TierOverflow` if a value of the transform (or of a butterfly pass) does
/// not fit the word; `x` is then left unchanged. When every `|x_i| <
/// 2^(BITS - 1 - log2(n))` no check is needed and none is made.
///
/// # Panics
/// Panics if `x.len()` is not a power of two (an empty slice is fine).
///
/// ```
/// use g_math::fixed_point::imperative::hadamard::walsh_hadamard_in_place;
/// let mut x = [1i64, 2, 3, 4];
/// walsh_hadamard_in_place(&mut x).unwrap();
/// assert_eq!(x, [10, -2, -4, 0]);
/// ```
pub fn walsh_hadamard_in_place<T: WideInt>(x: &mut [T]) -> Result<(), OverflowDetected> {
    let n = x.len();
    if n <= 1 {
        return Ok(());
    }
    assert!(n.is_power_of_two(), "walsh_hadamard_in_place: length {n} is not a power of two");
    let k = n.trailing_zeros();
    // each pass at most doubles the largest magnitude: below this bound no pass overflows
    let widest = x.iter().map(|v| v.magnitude_bit_length()).max().unwrap_or(0);
    if widest + k < T::BITS {
        butterflies(x);
        return Ok(());
    }
    let mut y = x.to_vec();
    butterflies_checked(&mut y)?;
    x.copy_from_slice(&y);
    Ok(())
}

/// [`walsh_hadamard_in_place`] on each consecutive block of `block` words
/// (a block-diagonal transform). The same errors; on an error the slice is
/// left unchanged.
///
/// # Panics
/// Panics if `block` is not a power of two or does not divide `x.len()`.
pub fn walsh_hadamard_blocks_in_place<T: WideInt>(x: &mut [T], block: usize) -> Result<(), OverflowDetected> {
    assert!(block.is_power_of_two(), "walsh_hadamard_blocks_in_place: block {block} is not a power of two");
    assert!(x.len() % block == 0, "walsh_hadamard_blocks_in_place: block {block} does not divide length {}", x.len());
    let k = block.trailing_zeros();
    let widest = x.iter().map(|v| v.magnitude_bit_length()).max().unwrap_or(0);
    if widest + k < T::BITS {
        for c in x.chunks_exact_mut(block) {
            butterflies(c);
        }
        return Ok(());
    }
    let mut y = x.to_vec();
    for c in y.chunks_exact_mut(block) {
        butterflies_checked(c)?;
    }
    x.copy_from_slice(&y);
    Ok(())
}

/// The butterfly passes without checks (the caller has bounded the input).
#[inline]
fn butterflies<T: WideInt>(x: &mut [T]) {
    let n = x.len();
    let mut h = 1;
    while h < n {
        for i in (0..n).step_by(2 * h) {
            let (lo, hi) = x[i..i + 2 * h].split_at_mut(h);
            for (a, b) in lo.iter_mut().zip(hi.iter_mut()) {
                let (u, v) = (*a, *b);
                *a = u + v;
                *b = u - v;
            }
        }
        h *= 2;
    }
}

/// The butterfly passes with a check on every sum.
fn butterflies_checked<T: WideInt>(x: &mut [T]) -> Result<(), OverflowDetected> {
    let n = x.len();
    let mut h = 1;
    while h < n {
        for i in (0..n).step_by(2 * h) {
            for j in i..i + h {
                let (u, v) = (x[j], x[j + h]);
                x[j] = u.checked_add_wide(v).ok_or(OverflowDetected::TierOverflow)?;
                x[j + h] = u.checked_sub_wide(v).ok_or(OverflowDetected::TierOverflow)?;
            }
        }
        h *= 2;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixed_point::{I256, I512};

    /// `sum_c (-1)^popcount(r & c) x_c` at 256 bits: the matrix definition.
    fn reference(x: &[i64]) -> Vec<I256> {
        let n = x.len();
        (0..n)
            .map(|r| {
                let mut s = I256::zero();
                for (c, &v) in x.iter().enumerate() {
                    let t = I256::from_i64(v);
                    s = if (r & c).count_ones() % 2 == 1 { s - t } else { s + t };
                }
                s
            })
            .collect()
    }

    fn lcg(s: &mut u64) -> u64 {
        *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        *s >> 11
    }

    #[test]
    fn equals_the_matrix_definition() {
        let mut seed = 0x4AD4_u64;
        for k in 0..=11u32 {
            let n = 1usize << k;
            for &bits in &[3u32, 20, 31, 52] {
                let x: Vec<i64> = (0..n).map(|_| (lcg(&mut seed) as i64 % (1i64 << bits)) - (1i64 << (bits - 1))).collect();
                let want = reference(&x);
                // i64: fast path when the bound allows, checked path otherwise
                let mut y = x.clone();
                match walsh_hadamard_in_place(&mut y) {
                    Ok(()) => {
                        for (a, b) in y.iter().zip(&want) { assert_eq!(I256::from_i64(*a), *b, "n {n} bits {bits}"); }
                    }
                    Err(_) => {
                        // an intermediate is half a sum or difference of the next pass: it overflows only if the result does
                        let fits_i64 = |w: &I256| w.fits_in_i128() && (w.as_i128() as i64) as i128 == w.as_i128();
                        assert!(want.iter().any(|w| !fits_i64(w)), "n {n} bits {bits}: refused a fitting transform");
                        assert_eq!(y, x, "an error leaves the input unchanged");
                    }
                }
                // I512 never overflows here
                let mut z: Vec<I512> = x.iter().map(|&v| I512::from_i64(v)).collect();
                walsh_hadamard_in_place(&mut z).unwrap();
                for (a, b) in z.iter().zip(&want) { assert_eq!(*a, I512::from_i256(*b)); }
            }
        }
    }

    #[test]
    fn checked_path_is_exact_at_the_edge_and_refuses_overflow() {
        // 2 elements at i32::MAX/2: the bound refuses the fast path, the checked path succeeds
        let mut x = [i32::MAX / 2, i32::MAX / 2];
        walsh_hadamard_in_place(&mut x).unwrap();
        assert_eq!(x, [i32::MAX - 1, 0]);
        // a sum past the word: error, input unchanged
        let mut y = [i32::MAX, 1];
        assert_eq!(walsh_hadamard_in_place(&mut y), Err(OverflowDetected::TierOverflow));
        assert_eq!(y, [i32::MAX, 1]);
        // the word's minimum transforms exactly when nothing is subtracted from it
        let mut m = [i64::MIN, 0, 0, 0];
        walsh_hadamard_in_place(&mut m).unwrap();
        assert_eq!(m, [i64::MIN; 4]);
        let mut m = [i64::MIN, 1, 0, 0];
        assert!(walsh_hadamard_in_place(&mut m).is_err());
        assert_eq!(m, [i64::MIN, 1, 0, 0]);
    }

    #[test]
    fn twice_is_n_times_identity_and_blocks_are_independent() {
        let mut seed = 7u64;
        let x: Vec<i64> = (0..4096).map(|_| (lcg(&mut seed) as i64 % 2_000_001) - 1_000_000).collect();
        let mut y = x.clone();
        walsh_hadamard_in_place(&mut y).unwrap();
        walsh_hadamard_in_place(&mut y).unwrap();
        for (a, b) in y.iter().zip(&x) { assert_eq!(*a, 4096 * b); }
        let mut blocks = x.clone();
        walsh_hadamard_blocks_in_place(&mut blocks, 1024).unwrap();
        for (c, (got, src)) in blocks.chunks(1024).zip(x.chunks(1024)).enumerate() {
            let mut one = src.to_vec();
            walsh_hadamard_in_place(&mut one).unwrap();
            assert_eq!(got, &one[..], "block {c}");
        }
    }

    #[test]
    #[should_panic(expected = "not a power of two")]
    fn rejects_other_lengths() {
        let mut x = [1i64, 2, 3];
        let _ = walsh_hadamard_in_place(&mut x);
    }
}
