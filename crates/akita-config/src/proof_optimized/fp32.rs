//! fp32 presets used for small-field integration and profiling.

use super::*;

/// Base field for the fp32 scaffold presets.
pub type Field = Prime32Offset99;
/// Akita's degree-4 extension for fp32 public claims and Fiat-Shamir challenges.
pub type ExtensionField = FpExt4<Field>;

/// Full-field `D=64` preset for fp32 crossover profiling.
#[derive(Clone, Copy, Debug, Default)]
pub struct D64Full;

/// Onehot `D=64` preset for fp32 crossover profiling.
#[derive(Clone, Copy, Debug, Default)]
pub struct D64OneHot;

/// Full-field `D=128` preset for planner-backed fp32 experiments.
#[derive(Clone, Copy, Debug, Default)]
pub struct D128Full;

/// Full-field `D=128` preset that caps decomposition at `log_basis = 3`.
///
/// This trades proof size for prover latency and enables the specialized
/// two-round first-stage sumcheck at every recursive fold.
#[derive(Clone, Copy, Debug, Default)]
pub struct D128FullFastProver;

/// Full-field `D=128` preset for explicitly range-proven 18-bit root data.
///
/// The application must supply a separate proof of this source bound; the PCS
/// configuration does not enforce it by itself.
#[derive(Clone, Copy, Debug, Default)]
pub struct D128FullBound18;

/// Onehot `D=128` preset for planner-backed fp32 experiments.
#[derive(Clone, Copy, Debug, Default)]
pub struct D128OneHot;

/// Binary one-hot `D=128` preset with 256-entry logical chunks.
///
/// This is the natural root shape for Shout addresses split into bytes. Unlike
/// [`D128OneHot`], whose legacy generated schedule leaves the chunk-size hint
/// at `1`, this preset asks the runtime planner to price the actual sparse
/// witness norm for one hot bit among 256 positions.
#[derive(Clone, Copy, Debug, Default)]
pub struct D128OneHot256;

/// Full-field `D=256` preset for planner-backed fp32 experiments.
#[derive(Clone, Copy, Debug, Default)]
pub struct D256Full;

/// Onehot `D=256` preset for planner-backed fp32 experiments.
#[derive(Clone, Copy, Debug, Default)]
pub struct D256OneHot;

impl_proof_optimized_preset!(
    D64Full,
    Field,
    ExtensionField,
    akita_types::SisModulusFamily::Q32,
    64,
    32,
    32
);
impl_proof_optimized_preset!(
    D64OneHot,
    Field,
    ExtensionField,
    akita_types::SisModulusFamily::Q32,
    64,
    32,
    1
);
impl_proof_optimized_preset!(
    D128Full,
    Field,
    ExtensionField,
    akita_types::SisModulusFamily::Q32,
    128,
    32,
    32
);
impl_proof_optimized_preset!(
    D128FullFastProver,
    Field,
    ExtensionField,
    akita_types::SisModulusFamily::Q32,
    128,
    32,
    32,
    basis_range = (3, 3)
);
impl_proof_optimized_preset!(
    D128FullBound18,
    Field,
    ExtensionField,
    akita_types::SisModulusFamily::Q32,
    128,
    32,
    18,
    basis_range = (3, 3)
);
impl_proof_optimized_preset!(
    D128OneHot,
    Field,
    ExtensionField,
    akita_types::SisModulusFamily::Q32,
    128,
    32,
    1,
    schedules = (
        "schedules-fp32-d128-onehot",
        "fp32_d128_onehot",
        fp32_d128_onehot_table
    )
);
impl_proof_optimized_preset!(
    D128OneHot256,
    Field,
    ExtensionField,
    akita_types::SisModulusFamily::Q32,
    128,
    32,
    1,
    256
);
impl_proof_optimized_preset!(
    D256Full,
    Field,
    ExtensionField,
    akita_types::SisModulusFamily::Q32,
    256,
    32,
    32
);
impl_proof_optimized_preset!(
    D256OneHot,
    Field,
    ExtensionField,
    akita_types::SisModulusFamily::Q32,
    256,
    32,
    1,
    schedules = (
        "schedules-fp32-d256-onehot",
        "fp32_d256_onehot",
        fp32_d256_onehot_table
    )
);
