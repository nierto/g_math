//! Gate for `fused::softcap*` and `fused::softcapped_swiglu*`: every result
//! is the exact value rounded once to storage (nearest, ties toward
//! +infinity), at every format.
//!
//! References: `scripts/generate_softcap_refs.py` (mpmath, 150 digits),
//! parsed at the build's format by the exact decimal parser, which rounds
//! them once; so the expected word is the correctly rounded value and the
//! gate asks for equality, not a tolerance. Every form (scalar, slice, in
//! place, interleaved, `_into`, wide inputs at 2F fraction bits) is checked.
//! The exhaustive check of the 32-bit first stage against the exact stage is
//! in the library's unit tests (`imperative::softcap`). Integer arithmetic only.

macro_rules! suite {
    ($m:ident, $S:ty, $F:literal) => {
        mod $m {
            type FixedPoint = Fixed<$S, $F>;
            use g_math::fixed_point::imperative::fused::{
                softcap, softcap_in_place, softcap_slice, softcap_wide_slice, softcapped_swiglu, softcapped_swiglu_interleaved,
                softcapped_swiglu_interleaved_into, softcapped_swiglu_interleaved_wide, softcapped_swiglu_interleaved_wide_into,
            };
            use g_math::fixed_point::{Fixed, Storage, WideInt};

            #[allow(dead_code)]
            mod refs {
                include!("data/softcap_refs.rs");
            }

            type W = <$S as Storage>::Compute;

            fn fp(s: &str) -> FixedPoint {
                FixedPoint::from_str(s)
            }

            /// Every value of a reference row in range at this format (the
            /// narrow-range splits hold only part of the table).
            fn fits(vals: &[&str]) -> bool {
                vals.iter().all(|v| FixedPoint::try_from_str(v).is_ok())
            }

            /// The references are multiples of 2^-8 (2^-16 for the wide
            /// inputs): exact from 8 fraction bits up. Below that (the edge
            /// splits) the suite compiles and does nothing.
            const EXACT: bool = $F >= 8;

            /// Whether `k / 65536` lies within the storage range (the wide
            /// forms take any compute-word input; the references stay in range).
            fn wide_fits(k: i64) -> bool {
                FixedPoint::try_from_int(((k.unsigned_abs() >> 16) + 1) as i32).is_ok()
            }

            /// `k / 65536` as a raw at 2F fraction bits (exact: 2F >= 16).
            fn wide(k: i64) -> W {
                W::from_i128(k as i128).shl_bits((2 * $F as u32).saturating_sub(16))
            }

            #[test]
            fn softcap_is_correctly_rounded() {
                if !EXACT {
                    return;
                }
                let mut n = 0;
                for &(x, c, want) in refs::SOFTCAP.iter().filter(|r| fits(&[r.0, r.1, r.2])) {
                    let (x, c, want) = (fp(x), fp(c), fp(want));
                    assert_eq!(softcap(x, c), want, "softcap({x}, {c})");
                    n += 1;
                }
                // the slice and in-place forms per cap
                for cap in ["30", "4", "25", "-7", "0.75", "100"] {
                    let rows: Vec<_> = refs::SOFTCAP.iter().filter(|r| r.1 == cap && fits(&[r.0, r.1, r.2])).collect();
                    if rows.is_empty() {
                        continue;
                    }
                    let xs: Vec<FixedPoint> = rows.iter().map(|r| fp(r.0)).collect();
                    let want: Vec<FixedPoint> = rows.iter().map(|r| fp(r.2)).collect();
                    assert_eq!(softcap_slice(&xs, fp(cap)), want, "softcap_slice cap {cap}");
                    let mut ys = xs.clone();
                    softcap_in_place(&mut ys, fp(cap));
                    assert_eq!(ys, want, "softcap_in_place cap {cap}");
                }
                println!("softcap: {n} values equal the correctly rounded references");
                assert!(n > 0);
            }

            #[test]
            fn softcap_wide_is_correctly_rounded() {
                if !EXACT {
                    return;
                }
                let mut n = 0;
                for &(k, c, want) in refs::SOFTCAP_WIDE.iter().filter(|r| fits(&[r.1, r.2]) && wide_fits(r.0)) {
                    n += 1;
                    assert_eq!(softcap_wide_slice::<$S, $F>(&[wide(k)], fp(c)), vec![fp(want)], "softcap_wide({k}/65536, {c})");
                }
                if fits(&["30", "120"]) {
                    let xs: Vec<W> = refs::SOFTCAP_WIDE.iter().filter(|r| r.1 == "30").map(|r| wide(r.0)).collect();
                    let want: Vec<FixedPoint> = refs::SOFTCAP_WIDE.iter().filter(|r| r.1 == "30").map(|r| fp(r.2)).collect();
                    assert_eq!(softcap_wide_slice::<$S, $F>(&xs, fp("30")), want);
                }
                println!("softcap wide: {n} values equal");
                assert!(n > 0);
            }

            #[test]
            fn softcapped_swiglu_is_correctly_rounded() {
                if !EXACT {
                    return;
                }
                let mut n = 0;
                for &(g, u, gc, uc, want) in refs::SWIGLU.iter().filter(|r| fits(&[r.0, r.1, r.2, r.3, r.4])) {
                    n += 1;
                    assert_eq!(softcapped_swiglu(fp(g), fp(u), fp(gc), fp(uc)), fp(want), "swiglu({g}, {u}; {gc}, {uc})");
                }
                for (gc, uc) in [("4", "25"), ("-4", "25"), ("0.5", "3"), ("8", "12"), ("0.5", "1.5")] {
                    let rows: Vec<_> = refs::SWIGLU.iter().filter(|r| r.2 == gc && r.3 == uc && fits(&[r.0, r.1, r.2, r.3, r.4])).collect();
                    if rows.is_empty() {
                        continue;
                    }
                    let gate_up: Vec<FixedPoint> = rows.iter().flat_map(|r| [fp(r.0), fp(r.1)]).collect();
                    let want: Vec<FixedPoint> = rows.iter().map(|r| fp(r.4)).collect();
                    assert_eq!(softcapped_swiglu_interleaved(&gate_up, fp(gc), fp(uc)), want, "interleaved {gc} {uc}");
                    let mut out = vec![FixedPoint::ZERO; want.len()];
                    softcapped_swiglu_interleaved_into(&gate_up, fp(gc), fp(uc), &mut out);
                    assert_eq!(out, want, "interleaved_into {gc} {uc}");
                }
                println!("softcapped_swiglu: {n} values equal");
                assert!(n > 0);
            }

            #[test]
            fn softcapped_swiglu_wide_is_correctly_rounded() {
                if !EXACT {
                    return;
                }
                let mut n = 0;
                for (gc, uc) in [("4", "25"), ("-4", "25"), ("0.5", "3"), ("8", "12"), ("0.5", "1.5")] {
                    let rows: Vec<_> = refs::SWIGLU_WIDE.iter().filter(|r| r.2 == gc && r.3 == uc && fits(&[r.2, r.3, r.4]) && wide_fits(r.0) && wide_fits(r.1)).collect();
                    if rows.is_empty() {
                        continue;
                    }
                    n += rows.len();
                    let gate_up: Vec<W> = rows.iter().flat_map(|r| [wide(r.0), wide(r.1)]).collect();
                    let want: Vec<FixedPoint> = rows.iter().map(|r| fp(r.4)).collect();
                    assert_eq!(softcapped_swiglu_interleaved_wide::<$S, $F>(&gate_up, fp(gc), fp(uc)), want, "wide {gc} {uc}");
                    let mut out = vec![FixedPoint::ZERO; want.len()];
                    softcapped_swiglu_interleaved_wide_into::<$S, $F>(&gate_up, fp(gc), fp(uc), &mut out);
                    assert_eq!(out, want, "wide_into {gc} {uc}");
                }
                println!("softcapped_swiglu wide: {n} values equal");
                assert!(n > 0);
            }

            #[test]
            fn zero_caps_and_odd_lengths_panic() {
                let one = FixedPoint::one();
                assert!(std::panic::catch_unwind(|| softcap(one, FixedPoint::ZERO)).is_err());
                assert!(std::panic::catch_unwind(|| softcapped_swiglu(one, one, FixedPoint::ZERO, one)).is_err());
                assert!(std::panic::catch_unwind(|| softcapped_swiglu_interleaved(&[one, one, one], one, one)).is_err());
            }
        }
    };
}

for_each_format!(suite);
