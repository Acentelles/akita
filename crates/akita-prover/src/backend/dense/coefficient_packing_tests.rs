//! Exact signed-byte packing comparisons against the field-source path.

use super::DensePoly;
use crate::backend::coefficient_packing::weighted_i8;
use crate::backend::RecursiveWitnessFlat;
use crate::compute::{
    CpuBackend, RootOpeningSource, SubringCoefficientPackingBatchKernel,
    SubringCoefficientPackingPlan,
};
use akita_field::{
    CanonicalField, Ext2, ExtField, FieldCore, FpExt4, FromPrimitiveInt, Prime128OffsetA7F7,
    Prime32Offset99, Prime64Offset59,
};
use akita_types::{
    coefficient_packing_partials, BasisMode, FpExtEncoding, PreparedSubringCoefficientPackingPoint,
    SubringCoefficientPackingGeometry,
};
use rand::{rngs::StdRng, RngCore, SeedableRng};

fn arbitrary_weight<F: FieldCore + FromPrimitiveInt, E: ExtField<F>>(rng: &mut StdRng) -> E {
    let coordinates = (0..E::EXT_DEGREE)
        .map(|_| F::from_u128((u128::from(rng.next_u64()) << 64) | u128::from(rng.next_u64())))
        .collect::<Vec<_>>();
    E::from_base_slice(&coordinates)
}

fn all_signed_byte_products<F, E>()
where
    F: FieldCore + FromPrimitiveInt,
    E: ExtField<F>,
{
    let mut rng = StdRng::seed_from_u64(0x69a2_30e7);
    let mut weights = vec![E::zero(), E::one(), -E::one()];
    weights.extend((0..64).map(|_| arbitrary_weight::<F, E>(&mut rng)));
    for weight in weights {
        for coefficient in i8::MIN..=i8::MAX {
            assert_eq!(
                weighted_i8::<F, E>(weight, coefficient),
                weight.mul_base(F::from_i8(coefficient)),
                "signed coefficient {coefficient}"
            );
        }
    }
}

#[test]
fn coefficient_packing_signed_bytes_cover_all_values_and_production_fields() {
    all_signed_byte_products::<Prime32Offset99, FpExt4<Prime32Offset99>>();
    all_signed_byte_products::<Prime64Offset59, Ext2<Prime64Offset59>>();
    all_signed_byte_products::<Prime128OffsetA7F7, Prime128OffsetA7F7>();
}

fn dense_cached_and_field_sources<F, E, const D: usize>(num_vars: usize, wide: Option<i64>)
where
    F: FieldCore + CanonicalField + FromPrimitiveInt,
    E: ExtField<F> + FpExtEncoding<F>,
{
    let mut rng = StdRng::seed_from_u64(0xc0ef_f1c1);
    let geometry = SubringCoefficientPackingGeometry::try_new(E::EXT_DEGREE, D, 64).unwrap();
    let public_point = (0..num_vars)
        .map(|_| arbitrary_weight::<F, E>(&mut rng))
        .collect::<Vec<_>>();
    let point = PreparedSubringCoefficientPackingPoint::new(
        geometry,
        BasisMode::Lagrange,
        (1usize << num_vars).div_ceil(D),
        2,
        num_vars,
        &public_point,
    )
    .unwrap();
    let mut coefficients = (0..1usize << num_vars)
        .map(|index| F::from_i8((index as u8).wrapping_add(128) as i8))
        .collect::<Vec<_>>();
    if let Some(wide) = wide {
        let last = coefficients.len() - 1;
        coefficients[last] = F::from_i64(wide);
    }
    let poly = DensePoly::from_field_evals(num_vars, coefficients).unwrap();
    assert_eq!(poly.small_i8_ring_coeffs::<D>().is_some(), wide.is_none());
    let mut field_only = poly.clone();
    field_only.small_i8_coeffs = None;
    let plan = SubringCoefficientPackingPlan { point: &point };
    let cached = poly
        .coefficient_packing_partials::<E, D>(plan, true)
        .unwrap();
    let disabled = poly
        .coefficient_packing_partials::<E, D>(plan, false)
        .unwrap();
    let uncached = field_only
        .coefficient_packing_partials::<E, D>(plan, true)
        .unwrap();
    assert_eq!(cached, disabled);
    assert_eq!(cached, uncached);

    let refs = [&poly, &field_only];
    let batch = <DensePoly<F> as RootOpeningSource<F, D>>::opening_batch(&refs).unwrap();
    let packed = CpuBackend::DEFAULT
        .coefficient_packing_partials_batch(None, batch, plan)
        .unwrap();
    assert_eq!(packed[0], packed[1]);
    assert_eq!(packed[0].coordinates(), cached);

    let expected = coefficient_packing_partials::<F, E>(
        geometry,
        point.num_live_positions(),
        point.num_positions_per_block(),
        &poly.field_coeffs()[..point.num_live_positions() * D],
        point.position_weights(),
        point.packing_weights(),
    )
    .unwrap();
    assert_eq!(packed[0].coordinates(), expected);
}

#[test]
fn coefficient_packing_opt_in_retains_arity_and_ring_validation() {
    type F = Prime64Offset59;
    type E = Ext2<F>;
    const D: usize = 128;
    let geometry = SubringCoefficientPackingGeometry::try_new(2, D, 64).unwrap();
    let point = PreparedSubringCoefficientPackingPoint::new(
        geometry,
        BasisMode::Lagrange,
        1,
        1,
        6,
        &[E::one(); 6],
    )
    .unwrap();
    let poly = DensePoly::from_field_evals(5, vec![F::one(); 32]).unwrap();
    for enabled in [false, true] {
        let plan = SubringCoefficientPackingPlan { point: &point };
        assert!(poly
            .coefficient_packing_partials::<E, D>(plan, enabled)
            .is_err());
        assert!(poly
            .coefficient_packing_partials::<E, 0>(plan, enabled)
            .is_err());
        assert!(poly
            .coefficient_packing_partials::<E, 3>(plan, enabled)
            .is_err());
        assert!(poly
            .coefficient_packing_partials::<E, 2048>(plan, enabled)
            .is_err());
    }
}

#[test]
fn coefficient_packing_cached_dense_matches_field_with_zero_padding_and_multiple_blocks() {
    for num_vars in [5, 8, 11] {
        dense_cached_and_field_sources::<Prime32Offset99, FpExt4<Prime32Offset99>, 256>(
            num_vars, None,
        );
        dense_cached_and_field_sources::<Prime32Offset99, FpExt4<Prime32Offset99>, 1024>(
            num_vars, None,
        );
        dense_cached_and_field_sources::<Prime64Offset59, Ext2<Prime64Offset59>, 128>(
            num_vars, None,
        );
        dense_cached_and_field_sources::<Prime128OffsetA7F7, Prime128OffsetA7F7, 128>(
            num_vars, None,
        );
    }
}

#[test]
fn coefficient_packing_wide_dense_retains_field_fallback() {
    for wide in [-129, 128, 65537] {
        dense_cached_and_field_sources::<Prime32Offset99, FpExt4<Prime32Offset99>, 256>(
            9,
            Some(wide),
        );
        dense_cached_and_field_sources::<Prime64Offset59, Ext2<Prime64Offset59>, 128>(
            9,
            Some(wide),
        );
        dense_cached_and_field_sources::<Prime128OffsetA7F7, Prime128OffsetA7F7, 128>(
            9,
            Some(wide),
        );
    }
}

#[test]
fn coefficient_packing_recursive_ignores_nonzero_inactive_tail_bytes() {
    type F = Prime64Offset59;
    type E = Ext2<F>;
    const D: usize = 128;
    let mut rng = StdRng::seed_from_u64(0xd191_75ac);
    let geometry = SubringCoefficientPackingGeometry::try_new(2, D, 64).unwrap();
    let public_point = (0..8)
        .map(|_| arbitrary_weight::<F, E>(&mut rng))
        .collect::<Vec<_>>();
    let point = PreparedSubringCoefficientPackingPoint::new(
        geometry,
        BasisMode::Lagrange,
        2,
        4,
        8,
        &public_point,
    )
    .unwrap();
    for live_len in [D + 1, D + 37, 2 * D - 1] {
        let digits = (0..2 * D)
            .map(|index| (index as u8).wrapping_add(128) as i8)
            .collect::<Vec<_>>();
        let mut expected_source = digits.iter().copied().map(F::from_i8).collect::<Vec<_>>();
        expected_source[live_len..].fill(F::zero());
        let recursive = RecursiveWitnessFlat::from_packed_i8_digits(digits, live_len)
            .unwrap()
            .align_for_commitment_ring_dim(D)
            .unwrap();
        let refs = [&recursive];
        let batch =
            <RecursiveWitnessFlat as RootOpeningSource<F, D>>::opening_batch(&refs).unwrap();
        let packed = CpuBackend::DEFAULT
            .coefficient_packing_partials_batch(
                None,
                batch,
                SubringCoefficientPackingPlan { point: &point },
            )
            .unwrap();
        let expected = coefficient_packing_partials::<F, E>(
            geometry,
            2,
            4,
            &expected_source,
            point.position_weights(),
            point.packing_weights(),
        )
        .unwrap();
        assert_eq!(packed[0].coordinates(), expected);
    }
}
