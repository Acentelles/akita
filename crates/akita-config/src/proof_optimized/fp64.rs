//! fp64 presets used for small-field integration and profiling.

use super::*;

/// Base field for the fp64 scaffold presets.
pub type Field = Prime64Offset59;
/// ring-subfield used for fp64 public claims and Fiat-Shamir challenges.
pub type ExtensionField = Ext2<Field>;

/// Full-field `D=64` preset.
#[derive(Clone, Copy, Debug, Default)]
pub struct D64Full;

/// Onehot `D=64` preset.
#[derive(Clone, Copy, Debug, Default)]
pub struct D64OneHot;

/// Full-field `D=128` preset for planner-backed fp64 experiments.
#[derive(Clone, Copy, Debug, Default)]
pub struct D128Full;

/// Full-field `D=128` preset for explicitly range-proven 18-bit root data.
///
/// The application must supply a separate proof of this source bound; the PCS
/// configuration does not enforce it by itself.
#[derive(Clone, Copy, Debug, Default)]
pub struct D128FullBound18;

/// Full-field `D=128` preset for explicitly range-proven signed 6-bit data,
/// intended for balanced radix-8 digit planes whose values lie in `[-4,3]`.
///
/// Like [`D128FullBound18`], the application must supply a separate proof of
/// the source bound; the PCS configuration does not enforce it by itself.
/// The bound is deliberately two digits wide rather than one: single-digit
/// bound-3 schedules failed Akita's stage-2 consistency in earlier
/// experiments and remain unsupported.
#[derive(Clone, Copy, Debug, Default)]
pub struct D128FullBound6;

/// Onehot `D=128` preset for planner-backed fp64 experiments.
#[derive(Clone, Copy, Debug, Default)]
pub struct D128OneHot;

/// Full-field `D=256` preset for planner-backed fp64 experiments.
#[derive(Clone, Copy, Debug, Default)]
pub struct D256Full;

/// Onehot `D=256` preset for planner-backed fp64 experiments.
#[derive(Clone, Copy, Debug, Default)]
pub struct D256OneHot;

impl_proof_optimized_preset!(
    D64Full,
    Field,
    ExtensionField,
    akita_types::SisModulusFamily::Q64,
    64,
    64,
    64
);
impl_proof_optimized_preset!(
    D64OneHot,
    Field,
    ExtensionField,
    akita_types::SisModulusFamily::Q64,
    64,
    64,
    1
);
impl_proof_optimized_preset!(
    D128Full,
    Field,
    ExtensionField,
    akita_types::SisModulusFamily::Q64,
    128,
    64,
    64,
    schedules = ("schedules-fp64-d128", "fp64_d128", fp64_d128_table)
);
impl_proof_optimized_preset!(
    D128FullBound18,
    Field,
    ExtensionField,
    akita_types::SisModulusFamily::Q64,
    128,
    64,
    18,
    basis_range = (3, 3)
);
impl_proof_optimized_preset!(
    D128FullBound6,
    Field,
    ExtensionField,
    akita_types::SisModulusFamily::Q64,
    128,
    64,
    6,
    basis_range = (3, 3)
);
impl_proof_optimized_preset!(
    D128OneHot,
    Field,
    ExtensionField,
    akita_types::SisModulusFamily::Q64,
    128,
    64,
    1,
    schedules = (
        "schedules-fp64-d128-onehot",
        "fp64_d128_onehot",
        fp64_d128_onehot_table
    )
);
impl_proof_optimized_preset!(
    D256Full,
    Field,
    ExtensionField,
    akita_types::SisModulusFamily::Q64,
    256,
    64,
    64
);
impl_proof_optimized_preset!(
    D256OneHot,
    Field,
    ExtensionField,
    akita_types::SisModulusFamily::Q64,
    256,
    64,
    1,
    schedules = (
        "schedules-fp64-d256-onehot",
        "fp64_d256_onehot",
        fp64_d256_onehot_table
    )
);
