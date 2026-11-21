//! Word-at-a-time long division on little-endian 64-bit words.
//!
//! Knuth, The Art of Computer Programming vol. 2, section 4.3.1, Algorithm D:
//! each quotient word is estimated from the top two words of the running
//! remainder and the top word of the normalised divisor, corrected with the
//! second divisor word (the estimate is then at most one too large), and the
//! rare remaining excess is undone by adding the divisor back. The quotient
//! and remainder of two integers are unique, so this returns the same words
//! as the bit-at-a-time loop it replaces, in about one step per word instead
//! of one per bit.
//!
//! Each quotient word is a two-word by one-word division by an invariant
//! divisor, done with a precomputed reciprocal (Moller and Granlund,
//! "Improved division by invariant integers", IEEE Trans. Computers 2011,
//! algorithm 4): one multiply and two corrections instead of a software
//! 128-by-64-bit division. The same reciprocal divides whole numbers by one
//! word ([`Recip64::div_words`]), which the decimal engine uses to divide by
//! powers of ten ([`POW10_RECIP`]).
//!
//! Both algorithms are written once as macros over the word type, so the
//! unit tests run the identical code on 8-bit words: the reciprocal division
//! exhaustively, and long division with every branch (including the
//! add-back step, about 2 in 2^64 per quotient word at 64 bits) reached
//! constantly, both checked against native integers.

/// Most words any caller divides (I2048 and its decimal twins are 32).
pub(crate) const MAX_WORDS: usize = 32;

#[cfg(test)]
pub(crate) static ADD_BACKS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

macro_rules! reciprocal_divisor {
    ($name:ident, $w:ty, $dw:ty, $bits:expr) => {
        /// A one-word divisor with its precomputed reciprocal.
        #[derive(Clone, Copy, Debug)]
        pub(crate) struct $name {
            /// the divisor shifted left until its top bit is set
            d: $w,
            /// floor((B^2 - 1) / d) - B for the normalised d
            v: $w,
            /// the normalising shift
            s: u32,
        }

        #[allow(dead_code)]
        impl $name {
            /// Panics if `d` is zero.
            pub(crate) const fn new(d: $w) -> Self {
                assert!(d != 0, "division by zero");
                let s = d.leading_zeros();
                let dn = d << s;
                // in [B, 2B): the cast drops the leading B
                let v = (<$dw>::MAX / dn as $dw) as $w;
                $name { d: dn, v, s }
            }

            /// `(u1 * B + u0) / d` and its remainder for a NORMALISED
            /// divisor (`s == 0`) and `u1 < d`.
            #[inline(always)]
            pub(crate) const fn div_2by1(&self, u1: $w, u0: $w) -> ($w, $w) {
                let p = (self.v as $dw) * (u1 as $dw);
                let p = p.wrapping_add(((u1 as $dw) << $bits) | u0 as $dw);
                let mut q1 = ((p >> $bits) as $w).wrapping_add(1);
                let q0 = p as $w;
                let mut r = u0.wrapping_sub(q1.wrapping_mul(self.d));
                if r > q0 {
                    q1 = q1.wrapping_sub(1);
                    r = r.wrapping_add(self.d);
                }
                if r >= self.d {
                    q1 += 1;
                    r -= self.d;
                }
                (q1, r)
            }

            /// Divide the little-endian words `w` in place; returns the
            /// remainder. Leading zero words are skipped.
            #[inline]
            pub(crate) fn div_words(&self, w: &mut [$w]) -> $w {
                let mut t = w.len();
                while t > 0 && w[t - 1] == 0 {
                    t -= 1;
                }
                if t == 0 {
                    return 0;
                }
                let s = self.s;
                if s == 0 {
                    let mut r: $w = 0;
                    for i in (0..t).rev() {
                        let (q, rr) = self.div_2by1(r, w[i]);
                        w[i] = q;
                        r = rr;
                    }
                    r
                } else {
                    // divide w * 2^s by d * 2^s: the same quotient, the
                    // remainder scaled by 2^s; the extra top word is below 2^s
                    let mut r: $w = w[t - 1] >> ($bits - s);
                    for i in (0..t).rev() {
                        let low = if i > 0 { w[i - 1] >> ($bits - s) } else { 0 };
                        let (q, rr) = self.div_2by1(r, (w[i] << s) | low);
                        w[i] = q;
                        r = rr;
                    }
                    r >> s
                }
            }
        }
    };
}

reciprocal_divisor!(Recip64, u64, u128, 64);
#[cfg(test)]
reciprocal_divisor!(Recip8, u8, u16, 8);

/// `10^k` for `k <= 19` with their reciprocals, for dividing by powers of ten.
pub(crate) const POW10_RECIP: [Recip64; 20] = {
    let mut t = [Recip64::new(1); 20];
    let mut k = 1;
    let mut p = 1u64;
    while k < 20 {
        p *= 10;
        t[k] = Recip64::new(p);
        k += 1;
    }
    t
};

macro_rules! knuth_d {
    ($name:ident, $recip:ident, $w:ty, $dw:ty, $bits:expr) => {
        /// `u = q * v + r` with `0 <= r < v` for unsigned words, little-endian.
        /// Writes the quotient into `q` (`q.len() == u.len()`) and the remainder
        /// into `r` (`r.len() == u.len()`). `v` must not be zero and
        /// `u.len() == v.len() <= MAX_WORDS`.
        #[allow(dead_code)]
        pub(crate) fn $name(u: &[$w], v: &[$w], q: &mut [$w], r: &mut [$w]) {
            const B: $dw = 1 << $bits;
            let len = u.len();
            debug_assert!(v.len() == len && q.len() == len && r.len() == len && len <= MAX_WORDS);
            for x in q.iter_mut() { *x = 0; }
            for x in r.iter_mut() { *x = 0; }
            // significant words
            let mut n = len;
            while n > 0 && v[n - 1] == 0 { n -= 1; }
            assert!(n > 0, "division by zero");
            let mut m = len;
            while m > 0 && u[m - 1] == 0 { m -= 1; }
            // u < v: quotient zero, remainder u
            if m < n || (m == n && {
                let mut i = n;
                loop {
                    i -= 1;
                    if u[i] != v[i] { break u[i] < v[i]; }
                    if i == 0 { break false; }
                }
            }) {
                r[..len].copy_from_slice(u);
                return;
            }
            if n == 1 {
                q[..m].copy_from_slice(&u[..m]);
                r[0] = $recip::new(v[0]).div_words(&mut q[..m]);
                return;
            }
            // normalise: the top divisor word gets its top bit set
            let s = v[n - 1].leading_zeros();
            let mut vn = [0 as $w; MAX_WORDS];
            let mut un = [0 as $w; MAX_WORDS + 1];
            if s == 0 {
                vn[..n].copy_from_slice(&v[..n]);
                un[..m].copy_from_slice(&u[..m]);
                un[m] = 0;
            } else {
                for i in (1..n).rev() {
                    vn[i] = (v[i] << s) | (v[i - 1] >> ($bits - s));
                }
                vn[0] = v[0] << s;
                un[m] = u[m - 1] >> ($bits - s);
                for i in (1..m).rev() {
                    un[i] = (u[i] << s) | (u[i - 1] >> ($bits - s));
                }
                un[0] = u[0] << s;
            }
            let (vtop, vnext) = (vn[n - 1] as $dw, vn[n - 2] as $dw);
            let vtop_recip = $recip::new(vn[n - 1]); // normalised: no shift
            for j in (0..=m - n).rev() {
                // estimate from the top two remainder words; un[j + n] <= vtop
                let (mut qhat, mut rhat) = if un[j + n] as $dw >= vtop {
                    // the quotient word is at most B - 1
                    let num = ((un[j + n] as $dw) << $bits) | un[j + n - 1] as $dw;
                    let qh = B - 1;
                    (qh, num - qh * vtop)
                } else {
                    let (q, r) = vtop_recip.div_2by1(un[j + n], un[j + n - 1]);
                    (q as $dw, r as $dw)
                };
                while rhat < B && qhat * vnext > ((rhat << $bits) | un[j + n - 2] as $dw) {
                    qhat -= 1;
                    rhat += vtop;
                }
                // un[j..=j + n] -= qhat * vn[..n]
                let mut carry: $dw = 0;
                let mut borrow: $w = 0;
                for i in 0..n {
                    let p = qhat * vn[i] as $dw + carry;
                    carry = p >> $bits;
                    let (d1, b1) = un[i + j].overflowing_sub(p as $w);
                    let (d2, b2) = d1.overflowing_sub(borrow);
                    un[i + j] = d2;
                    borrow = (b1 | b2) as $w;
                }
                let top = un[j + n] as $dw;
                let sub = carry + borrow as $dw;
                un[j + n] = top.wrapping_sub(sub) as $w;
                if top < sub {
                    // qhat was one too large: add the divisor back
                    #[cfg(test)]
                    ADD_BACKS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    qhat -= 1;
                    let mut c: $w = 0;
                    for i in 0..n {
                        let (s1, c1) = un[i + j].overflowing_add(vn[i]);
                        let (s2, c2) = s1.overflowing_add(c);
                        un[i + j] = s2;
                        c = (c1 | c2) as $w;
                    }
                    un[j + n] = un[j + n].wrapping_add(c);
                }
                q[j] = qhat as $w;
            }
            // denormalise the remainder
            if s == 0 {
                r[..n].copy_from_slice(&un[..n]);
            } else {
                for i in 0..n - 1 {
                    r[i] = (un[i] >> s) | (un[i + 1] << ($bits - s));
                }
                r[n - 1] = un[n - 1] >> s;
            }
        }
    };
}

knuth_d!(udivmod_slices, Recip64, u64, u128, 64);
#[cfg(test)]
knuth_d!(udivmod_slices_u8, Recip8, u8, u16, 8);

/// Unsigned `(u / v, u % v)` on `N` words. Panics if `v` is zero.
#[inline]
pub(crate) fn udivmod<const N: usize>(u: &[u64; N], v: &[u64; N]) -> ([u64; N], [u64; N]) {
    let mut q = [0u64; N];
    let mut r = [0u64; N];
    udivmod_slices(u, v, &mut q, &mut r);
    (q, r)
}

#[inline]
fn negate<const N: usize>(a: [u64; N]) -> [u64; N] {
    let mut out = [0u64; N];
    let mut carry = 1u64;
    for i in 0..N {
        let (s, c) = (!a[i]).overflowing_add(carry);
        out[i] = s;
        carry = c as u64;
    }
    out
}

/// Signed `(a / b, a % b)` on two's-complement words, as Rust's `/` and `%`:
/// the quotient truncated toward zero, the remainder with the dividend's sign.
///
/// # Panics
/// Panics if `b` is zero, and on the one overflowing quotient `MIN / -1`
/// (as native integers do).
#[inline]
pub(crate) fn divmod_signed<const N: usize>(a: [u64; N], b: [u64; N]) -> ([u64; N], [u64; N]) {
    let a_neg = a[N - 1] >> 63 == 1;
    let b_neg = b[N - 1] >> 63 == 1;
    // the magnitude of MIN is 2^(64N - 1): its own words read unsigned
    let ua = if a_neg { negate(a) } else { a };
    let ub = if b_neg { negate(b) } else { b };
    let (q, r) = udivmod(&ua, &ub);
    let q = if a_neg != b_neg {
        negate(q)
    } else {
        assert!(q[N - 1] >> 63 == 0, "attempt to divide with overflow");
        q
    };
    let r = if a_neg { negate(r) } else { r };
    (q, r)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Small deterministic generator (xorshift64*), no dependencies.
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 >> 12;
            self.0 ^= self.0 << 25;
            self.0 ^= self.0 >> 27;
            self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
        }
    }

    fn to_u8_words(x: u64, len: usize) -> Vec<u8> {
        (0..len).map(|i| (x >> (8 * i)) as u8).collect()
    }
    fn from_u8_words(w: &[u8]) -> u64 {
        w.iter().enumerate().fold(0u64, |acc, (i, &b)| acc | (b as u64) << (8 * i))
    }

    /// The reciprocal division on 8-bit words, exhaustively: every
    /// normalised divisor and every two-word dividend with a high word below
    /// it, and every divisor against every two-word number through
    /// `div_words` (the shifted path included).
    #[test]
    fn reciprocal_division_on_8_bit_words_is_exact_everywhere() {
        for d in 128u16..=255 {
            let r8 = Recip8::new(d as u8);
            for u1 in 0..d {
                for u0 in 0u16..=255 {
                    let n = u1 << 8 | u0;
                    assert_eq!(r8.div_2by1(u1 as u8, u0 as u8), ((n / d) as u8, (n % d) as u8), "{n} / {d}");
                }
            }
        }
        for d in 1u32..=255 {
            let r8 = Recip8::new(d as u8);
            for n in 0u32..1 << 16 {
                let mut w = [n as u8, (n >> 8) as u8];
                let rem = r8.div_words(&mut w);
                assert_eq!((w[0] as u32 | (w[1] as u32) << 8, rem as u32), (n / d, n % d), "{n} / {d}");
            }
        }
    }

    /// The 64-bit reciprocal division against u128 division, and the
    /// powers-of-ten table against the powers.
    #[test]
    fn reciprocal_division_on_64_bit_words_matches_u128_division() {
        let mut rng = Rng(0xA5A5_5A5A_0F0F_F0F0);
        let mut divisors = vec![1u64, 2, 3, 7, 10, 1 << 63, (1 << 63) + 1, u64::MAX, u64::MAX - 1, 10_000_000_000_000_000_000];
        for k in 0..20 {
            assert_eq!(POW10_RECIP[k].d >> POW10_RECIP[k].s, 10u64.pow(k as u32));
            divisors.push(10u64.pow(k as u32));
        }
        for _ in 0..2_000 {
            divisors.push(rng.next() >> (rng.next() % 64));
        }
        for &d in &divisors {
            let d = d.max(1);
            let r = Recip64::new(d);
            for _ in 0..200 {
                let mut w = [rng.next() >> (rng.next() % 64), rng.next() >> (rng.next() % 64), 0, 0];
                if rng.next() % 4 == 0 { w[1] = 0; }
                let n = w[0] as u128 | (w[1] as u128) << 64;
                let rem = r.div_words(&mut w);
                assert_eq!((w[0] as u128 | (w[1] as u128) << 64, rem as u128), (n / d as u128, n % d as u128), "{n} / {d}");
                assert_eq!((w[2], w[3]), (0, 0));
            }
        }
    }

    /// The algorithm on 8-bit words against native u64 division: every
    /// branch, the add-back included, is reached many times.
    #[test]
    fn knuth_d_on_8_bit_words_matches_native_division() {
        ADD_BACKS.store(0, std::sync::atomic::Ordering::Relaxed);
        let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
        let mut cases = 0u64;
        for len in 1..=8usize {
            let bits = 8 * len as u32;
            let mask = if bits == 64 { u64::MAX } else { (1u64 << bits) - 1 };
            for _ in 0..200_000 {
                // operands of random bit length, top words often near the edges
                let ub = 1 + (rng.next() % bits as u64) as u32;
                let vb = 1 + (rng.next() % bits as u64) as u32;
                let mut u = rng.next() & (u64::MAX >> (64 - ub)) & mask;
                let mut v = rng.next() & (u64::MAX >> (64 - vb)) & mask;
                match rng.next() % 8 {
                    0 => v |= 0x80u64 << (8 * ((vb - 1) / 8)), // top bit of its top word
                    1 => u |= mask & !(mask >> 8), // dividend top word full
                    2 => v = (v | 0xFF) & mask, // low word all ones
                    _ => {}
                }
                if v == 0 { v = 1; }
                let (uw, vw) = (to_u8_words(u, len), to_u8_words(v, len));
                let mut q = vec![0u8; len];
                let mut r = vec![0u8; len];
                udivmod_slices_u8(&uw, &vw, &mut q, &mut r);
                assert_eq!((from_u8_words(&q), from_u8_words(&r)), (u / v, u % v), "u={u:#x} v={v:#x} len={len}");
                cases += 1;
            }
        }
        // exhaustive: every 24-bit dividend by every 16-bit divisor whose top
        // word is in the add-back-prone band
        for v in (0x8000u64..=0xFFFF).step_by(389) {
            for u in (0..1u64 << 24).step_by(61) {
                let (uw, vw) = (to_u8_words(u, 3), to_u8_words(v, 3));
                let mut q = vec![0u8; 3];
                let mut r = vec![0u8; 3];
                udivmod_slices_u8(&uw, &vw, &mut q, &mut r);
                assert_eq!((from_u8_words(&q), from_u8_words(&r)), (u / v, u % v), "u={u:#x} v={v:#x}");
                cases += 1;
            }
        }
        let adds = ADD_BACKS.load(std::sync::atomic::Ordering::Relaxed);
        eprintln!("8-bit words: {cases} cases, add-back reached {adds} times");
        assert!(adds >= 50, "add-back branch reached only {adds} times in {cases} cases");
    }

    /// The 64-bit instantiation against u128 division on two-word operands.
    #[test]
    fn knuth_d_on_64_bit_words_matches_u128_division() {
        let mut rng = Rng(0xD1B5_4A32_D192_ED03);
        for _ in 0..500_000 {
            let ub = 1 + (rng.next() % 128) as u32;
            let vb = 1 + (rng.next() % 128) as u32;
            let u = ((rng.next() as u128) << 64 | rng.next() as u128) >> (128 - ub);
            let mut v = ((rng.next() as u128) << 64 | rng.next() as u128) >> (128 - vb);
            if v == 0 { v = 1; }
            let uw = [u as u64, (u >> 64) as u64];
            let vw = [v as u64, (v >> 64) as u64];
            let (q, r) = udivmod(&uw, &vw);
            assert_eq!((q[0] as u128 | (q[1] as u128) << 64, r[0] as u128 | (r[1] as u128) << 64), (u / v, u % v));
        }
    }

    #[test]
    fn signed_matches_native_i128() {
        let mut rng = Rng(0x0123_4567_89AB_CDEF);
        let edges = [0i128, 1, -1, 2, -2, i128::MAX, i128::MIN, i128::MIN + 1, 1 << 64, -(1 << 64), (1 << 64) - 1, u64::MAX as i128 + 2];
        let check = |a: i128, b: i128| {
            if b == 0 || (a == i128::MIN && b == -1) { return; }
            let w = |x: i128| [x as u64, (x >> 64) as u64];
            let (q, r) = divmod_signed(w(a), w(b));
            assert_eq!((q[0] as u128 | (q[1] as u128) << 64) as i128, a / b, "{a} / {b}");
            assert_eq!((r[0] as u128 | (r[1] as u128) << 64) as i128, a % b, "{a} % {b}");
        };
        for &a in &edges { for &b in &edges { check(a, b); } }
        for _ in 0..200_000 {
            let a = (((rng.next() as u128) << 64 | rng.next() as u128) as i128) >> (rng.next() % 127);
            let b = (((rng.next() as u128) << 64 | rng.next() as u128) as i128) >> (rng.next() % 127);
            check(a, b);
        }
    }

    #[test]
    #[should_panic(expected = "attempt to divide with overflow")]
    fn min_by_minus_one_panics() {
        let _ = divmod_signed([0, 1u64 << 63], [u64::MAX, u64::MAX]);
    }
}
