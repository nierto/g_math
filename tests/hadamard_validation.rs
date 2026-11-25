//! Gate for `imperative::hadamard`: the exact Walsh-Hadamard transform
//! against references from the matrix definition (scripts/generate_hadamard_refs.py:
//! integers, confirmed by mpmath, the normalised floor checked against a real
//! sqrt for n = 4^j): lengths 1..1024, i32 at and past its limits (the
//! checked path; errors leave the input unchanged), i64, I256.

mod refs {
    include!("data/hadamard_refs.rs");
}

use g_math::fixed_point::imperative::hadamard::{walsh_hadamard_blocks_in_place, walsh_hadamard_in_place};
use g_math::fixed_point::imperative::OverflowDetected;
use g_math::fixed_point::I256;

fn parse(v: &[&str]) -> Vec<i128> {
    v.iter().map(|s| s.parse::<i128>().unwrap()).collect()
}

#[test]
fn equals_the_matrix_definition() {
    let mut checked = 0;
    for &(word, x, y) in refs::CASES {
        let xi = parse(x);
        let yi = y.map(parse);
        match word {
            "i64" => {
                let mut v: Vec<i64> = xi.iter().map(|&a| a as i64).collect();
                let r = walsh_hadamard_in_place(&mut v);
                match &yi {
                    Some(w) => {
                        assert_eq!(r, Ok(()));
                        assert_eq!(v.iter().map(|&a| a as i128).collect::<Vec<_>>(), *w, "i64 n {}", x.len());
                    }
                    None => assert_eq!(r, Err(OverflowDetected::TierOverflow)),
                }
            }
            "i32" => {
                let orig: Vec<i32> = xi.iter().map(|&a| a as i32).collect();
                let mut v = orig.clone();
                let r = walsh_hadamard_in_place(&mut v);
                match &yi {
                    Some(w) if w.iter().all(|&a| i32::try_from(a).is_ok()) => {
                        assert_eq!(r, Ok(()));
                        assert_eq!(v.iter().map(|&a| a as i128).collect::<Vec<_>>(), *w);
                    }
                    _ => {
                        assert_eq!(r, Err(OverflowDetected::TierOverflow), "i32 {x:?}");
                        assert_eq!(v, orig, "an error leaves the input unchanged");
                    }
                }
            }
            "i256" => {
                let mut v: Vec<I256> = xi.iter().map(|&a| I256::from_i128(a)).collect();
                walsh_hadamard_in_place(&mut v).unwrap();
                let w: Vec<I256> = yi.unwrap().iter().map(|&a| I256::from_i128(a)).collect();
                assert_eq!(v, w);
            }
            _ => unreachable!(),
        }
        checked += 1;
    }
    assert_eq!(checked, refs::CASES.len());
    assert!(checked >= 39);
}

#[test]
fn blocks_equal_each_block_alone() {
    // the 1024-element i64 cases as four blocks of 256 and as one block
    for &(word, x, _) in refs::CASES {
        if word != "i64" || x.len() != 1024 {
            continue;
        }
        let xi: Vec<i64> = parse(x).iter().map(|&a| a as i64).collect();
        let mut blocks = xi.clone();
        walsh_hadamard_blocks_in_place(&mut blocks, 256).unwrap();
        for (b, chunk) in blocks.chunks(256).enumerate() {
            let mut one = xi[b * 256..(b + 1) * 256].to_vec();
            walsh_hadamard_in_place(&mut one).unwrap();
            assert_eq!(chunk, &one[..]);
        }
        let mut whole = xi.clone();
        walsh_hadamard_blocks_in_place(&mut whole, 1024).unwrap();
        let mut direct = xi.clone();
        walsh_hadamard_in_place(&mut direct).unwrap();
        assert_eq!(whole, direct);
    }
}
