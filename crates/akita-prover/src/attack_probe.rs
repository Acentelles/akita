//! Adversarial prover injections used ONLY by the carried-claim forgery
//! regression tests. Compiled out unless the `attack-probe` feature is on.
//!
//! # What this exists for
//!
//! `aerie/_docs/ABSORPTION-AUDIT.md` (Defect 1, and the appended
//! "Verification of Defect 1") documents a soundness defect in the `G = 2`
//! recursive suffix: the carried setup-prefix claim and the carried
//! folded-witness claim were combined with unit coefficients, so a prover
//! could ship both carried claims false by a compensating `(+delta, -delta)`
//! and be accepted with probability 1. The forgery was executed end to end at
//! `k = 2` (fp64 `D128FullBound18`) and at `k = 1` (fp128 `D64OneHot`).
//!
//! The fix squeezes the per-claim batching coefficients from the transcript
//! after absorbing the claimed values. To keep that fix honest we keep the
//! attacking prover: the regression tests
//! `crates/akita-pcs/tests/suffix_carried_claim_forgery_*.rs` run it against
//! the real verifier and require a rejection.
//!
//! # Safety
//!
//! Every injection site is `#[cfg(feature = "attack-probe")]`, so a build
//! without the feature contains no branch, no atomic, and no code. The feature
//! must never be enabled outside these tests; nothing in the workspace's
//! default or `profile-ci` feature sets turns it on.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, OnceLock};

static ARMED: AtomicBool = AtomicBool::new(false);

fn log_lines() -> &'static Mutex<Vec<String>> {
    static CELL: OnceLock<Mutex<Vec<String>>> = OnceLock::new();
    CELL.get_or_init(|| Mutex::new(Vec::new()))
}

/// Arm the probe and clear any recorded diagnostics.
pub fn arm() {
    log_lines().lock().expect("attack probe poisoned").clear();
    ARMED.store(true, Ordering::SeqCst);
}

/// Disarm the probe.
pub fn disarm() {
    ARMED.store(false, Ordering::SeqCst);
}

/// Whether the probe is armed.
#[inline]
#[must_use]
pub fn is_armed() -> bool {
    ARMED.load(Ordering::SeqCst)
}

/// Record a diagnostic line (also echoed to stderr for `--nocapture` runs).
pub fn note(line: impl Into<String>) {
    let line = line.into();
    eprintln!("[attack-probe] {line}");
    log_lines()
        .lock()
        .expect("attack probe poisoned")
        .push(line);
}

/// Drain the recorded diagnostics.
#[must_use]
pub fn take_notes() -> Vec<String> {
    std::mem::take(&mut *log_lines().lock().expect("attack probe poisoned"))
}
