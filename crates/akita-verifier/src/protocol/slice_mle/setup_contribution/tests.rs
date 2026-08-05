use super::fixtures::{SetupContributionFixture, SetupContributionShape, TEST_RING_DIM_128};

#[test]
fn setup_contribution_matches_materialized_on_root_fixture() {
    SetupContributionFixture::from_shape(&SetupContributionShape::root_single_point())
        .assert_direct_matches_materialized();
}

#[test]
fn setup_contribution_matches_materialized_on_recursive_multi_group_fixture() {
    SetupContributionFixture::from_shape(&SetupContributionShape::recursive_multi_group())
        .assert_direct_matches_materialized();
}

#[test]
fn setup_index_weight_mle_matches_materialized() {
    SetupContributionFixture::from_shape(&SetupContributionShape::root_single_point())
        .assert_setup_index_weight_mle_matches_materialized();
    SetupContributionFixture::from_shape(&SetupContributionShape::recursive_multi_group())
        .assert_setup_index_weight_mle_matches_materialized();
}

// Mixed-D delegation equivalence oracle (`specs/mixed-d-setup-delegation.md`):
// the packed direct scan must equal the materialized `<S, omega_S>` inner
// product exactly at ring dimension 128 (the Aerie fp64 preset dimension),
// for both the root and the recursive multi-group shapes.
#[test]
fn setup_contribution_matches_materialized_on_root_fixture_d128() {
    SetupContributionFixture::from_shape_at::<TEST_RING_DIM_128>(
        &SetupContributionShape::root_single_point(),
    )
    .assert_direct_matches_materialized_at::<TEST_RING_DIM_128>();
}

#[test]
fn setup_contribution_matches_materialized_on_recursive_multi_group_fixture_d128() {
    SetupContributionFixture::from_shape_at::<TEST_RING_DIM_128>(
        &SetupContributionShape::recursive_multi_group(),
    )
    .assert_direct_matches_materialized_at::<TEST_RING_DIM_128>();
}

#[test]
fn setup_index_weight_mle_matches_materialized_d128() {
    SetupContributionFixture::from_shape_at::<TEST_RING_DIM_128>(
        &SetupContributionShape::root_single_point(),
    )
    .assert_setup_index_weight_mle_matches_materialized();
    SetupContributionFixture::from_shape_at::<TEST_RING_DIM_128>(
        &SetupContributionShape::recursive_multi_group(),
    )
    .assert_setup_index_weight_mle_matches_materialized();
}

#[test]
fn setup_contribution_matches_materialized_with_offset_carries_d128() {
    SetupContributionFixture::from_shape_at::<TEST_RING_DIM_128>(
        &SetupContributionShape::e_t_offset_carry(),
    )
    .assert_direct_matches_materialized_at::<TEST_RING_DIM_128>();
    SetupContributionFixture::from_shape_at::<TEST_RING_DIM_128>(
        &SetupContributionShape::pow2_z_offset_carry(),
    )
    .assert_direct_matches_materialized_at::<TEST_RING_DIM_128>();
}

#[test]
fn setup_contribution_matches_materialized_on_terminal_fixture() {
    SetupContributionFixture::from_shape(&SetupContributionShape::terminal_relation_only())
        .assert_direct_matches_materialized();
}

#[test]
fn setup_contribution_matches_materialized_on_dense_non_pow2_z_fixture() {
    SetupContributionFixture::from_shape(&SetupContributionShape::dense_non_pow2_z())
        .assert_direct_matches_materialized();
}

#[test]
fn setup_contribution_matches_materialized_on_batched_root_fixture() {
    SetupContributionFixture::from_shape(&SetupContributionShape::batched_root())
        .assert_direct_matches_materialized();
}

#[test]
fn setup_contribution_matches_materialized_with_e_t_offset_carries() {
    SetupContributionFixture::from_shape(&SetupContributionShape::e_t_offset_carry())
        .assert_direct_matches_materialized();
}

#[test]
fn setup_contribution_matches_materialized_with_pow2_z_offset_carry() {
    SetupContributionFixture::from_shape(&SetupContributionShape::pow2_z_offset_carry())
        .assert_direct_matches_materialized();
}
