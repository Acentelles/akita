use akita_algebra::CyclotomicRing;
use akita_field::{CanonicalField, FieldCore, Prime128OffsetA7F7, Prime64Offset59};
use akita_prover::DensePoly;

// DensePoly equality compares the centered byte cache as well as the field
// coefficients. The unchanged ring constructor builds that cache through a
// separate implementation, providing a reference for the new field path.
fn reference<T: CanonicalField>(values: &[T]) -> DensePoly<T> {
    DensePoly::from_ring_coeffs(
        values
            .iter()
            .map(|&x| CyclotomicRing::from_coefficients([x]))
            .collect(),
    )
}

fn check_cache<T: FieldCore + CanonicalField>() {
    // Mix high index bits so different chunks carry different signed values.
    for num_vars in [0, 9, 13, 14, 15, 16, 17] {
        let len = 1 << num_vars;
        let signed: Vec<i8> = (0..len)
            .map(|i| i8::from_ne_bytes([u8::try_from((i ^ (i >> 8) ^ (i >> 16)) & 255).unwrap()]))
            .collect();
        let evals: Vec<T> = signed
            .iter()
            .map(|&value| {
                let magnitude = T::from_u64(u64::from(value.unsigned_abs()));
                if value < 0 {
                    -magnitude
                } else {
                    magnitude
                }
            })
            .collect();
        let expected = reference(&evals);
        let allocation = evals.as_ptr();
        let borrowed = DensePoly::<T>::from_field_evals(num_vars, evals.as_slice()).unwrap();
        let owned = DensePoly::<T>::from_field_evals(num_vars, evals).unwrap();
        assert_eq!(owned, borrowed);
        if len >= 1024 {
            assert_eq!(owned.field_coeffs().as_ptr(), allocation);
        }
        assert_eq!(owned, expected);
        assert!(owned.field_coeffs()[len..].iter().all(|&x| x == T::zero()));
        // Exercise failures before, on and after chunk boundaries, and in
        // the final chunk. A partial cache must never survive rejection.
        let mut positions = vec![0, len - 1];
        for boundary in [16384, 32768, 65536] {
            for index in [boundary - 1, boundary, boundary + 1] {
                if index < len {
                    positions.push(index);
                }
            }
        }
        for index in positions {
            for wide in [T::from_u64(128), -T::from_u64(129)] {
                let mut bad = owned.field_coeffs()[..len].to_vec();
                bad[index] = wide;
                let poly = DensePoly::<T>::from_field_evals(num_vars, bad.clone()).unwrap();
                assert_eq!(&poly.field_coeffs()[..len], bad);
                assert_eq!(poly, reference(&bad), "wide coefficient at {index}");
            }
        }
    }
}

#[test]
fn small_cache_fp64_matches_ring_reference_and_discards_partial_results() {
    check_cache::<Prime64Offset59>();
}

#[test]
fn small_cache_fp128_matches_ring_reference_and_discards_partial_results() {
    check_cache::<Prime128OffsetA7F7>();
}
