//! Shared verifier-input blob shipped from a host artifact generator into a
//! Jolt guest program.
//!
//! The host serializes the bundle once (`AkitaJoltInputs::write_to_bytes`) and
//! the Jolt guest deserializes it as the very first step of the program.
//! Per-component encoding is the existing [`AkitaSerialize`] /
//! [`AkitaDeserialize`] machinery in [`akita_serialization`]. The recursion
//! benchmark can opt into an explicitly trusted cached-matrix setup decoder;
//! strict decoding remains the default.
//!
//! Format v2 (`AKJOLTv2`) generalizes the original single-group blob:
//!
//! - **Multi-group claims.** The blob carries a vector of opening groups
//!   (per-group prefix arity, openings, commitment) so the recursive
//!   setup-offload profile key (two dense precommits at `nv/2` plus the final
//!   group) fits in one blob. A Direct-mode blob is the one-group special
//!   case.
//! - **Setup transport.** The verifier setup's shared matrix can travel as
//!   the full expanded matrix (as before), as a truncated prefix slice, or
//!   not at all (seed-derived in the guest). The truncated transports are
//!   only accepted for `SetupContributionMode::Recursive`, where the
//!   delegated stage-3 setup-product sumcheck plus the public setup-prefix
//!   commitment `C_S` (shipped in `prefix_slots`) replace the root-level
//!   matrix scan; Direct-mode replay scans the matrix and must ship it whole.

#![allow(clippy::missing_errors_doc)]

use akita_field::{CanonicalField, FieldCore, RandomSampling};
use akita_serialization::{
    AkitaDeserialize, AkitaSerialize, Compress, SerializationError, Valid, Validate,
};
use akita_types::{
    derive_public_matrix_flat, AkitaBatchedProof, AkitaBatchedProofShape, AkitaExpandedSetup,
    AkitaSetupSeed, AkitaVerifierSetup, Commitment, FlatMatrix, OpeningClaims,
    PointVariableSelection, PolynomialGroupClaims, SetupContributionMode,
    SetupPrefixVerifierRegistry, MAX_SETUP_MATRIX_FIELD_ELEMENTS,
};
use std::sync::Arc;

/// Encoding mode used for the verifier-input blob. Held constant on both ends
/// so the host and guest don't have to negotiate compression.
pub const BLOB_COMPRESS: Compress = Compress::No;

/// Validation mode used when decoding on the guest side. The blob is verifier
/// input, so malformed shape headers must be rejected before they drive
/// allocation or proof replay.
pub const BLOB_VALIDATE: Validate = Validate::Yes;

/// Maximum verifier-input blob bytes accepted by host and guest.
///
/// Mirrors the Jolt guest `max_input_size` literal in `guest/src/lib.rs`.
pub const MAX_JOLT_BLOB_BYTES: u64 = 805_306_368;

/// Maximum number of opening groups accepted in one blob.
pub const MAX_BLOB_GROUPS: usize = 16;

/// Maximum number of openings accepted per group.
pub const MAX_GROUP_OPENINGS: usize = 64;

/// Magic header so the guest fails fast if it gets the wrong bytes.
const BLOB_MAGIC: [u8; 8] = *b"AKJOLTv2";
const MAX_TRANSCRIPT_DOMAIN_BYTES: usize = 1024;
const MAX_BLOB_NUM_VARS: usize = 64;

/// How the verifier setup's shared matrix travels inside the blob.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SetupTransport {
    /// Full expanded matrix (`seed.max_setup_len` ring elements). The only
    /// transport valid for Direct-mode blobs.
    ExpandedMatrix,
    /// Truncated prefix slice of `shipped_setup_ring_len` ring elements.
    /// Recursive mode only.
    MatrixSlice,
    /// No matrix payload; the decoder re-derives the first
    /// `shipped_setup_ring_len` ring elements from `seed.public_matrix_seed`
    /// (the flat derivation is prefix-stable by construction). Recursive mode
    /// only.
    SeedDerived,
}

impl SetupTransport {
    fn to_u8(self) -> u8 {
        match self {
            SetupTransport::ExpandedMatrix => 0,
            SetupTransport::MatrixSlice => 1,
            SetupTransport::SeedDerived => 2,
        }
    }

    fn from_u8(byte: u8) -> Result<Self, SerializationError> {
        match byte {
            0 => Ok(SetupTransport::ExpandedMatrix),
            1 => Ok(SetupTransport::MatrixSlice),
            2 => Ok(SetupTransport::SeedDerived),
            other => Err(SerializationError::InvalidData(format!(
                "akita-jolt blob has invalid setup-transport byte {other}"
            ))),
        }
    }

    /// Whether the blob carries matrix payload bytes for this transport.
    #[must_use]
    pub fn ships_matrix_payload(self) -> bool {
        !matches!(self, SetupTransport::SeedDerived)
    }
}

fn setup_mode_to_u8(mode: SetupContributionMode) -> u8 {
    match mode {
        SetupContributionMode::Direct => 0,
        SetupContributionMode::Recursive => 1,
    }
}

fn setup_mode_from_u8(byte: u8) -> Result<SetupContributionMode, SerializationError> {
    match byte {
        0 => Ok(SetupContributionMode::Direct),
        1 => Ok(SetupContributionMode::Recursive),
        other => Err(SerializationError::InvalidData(format!(
            "akita-jolt blob has invalid setup-contribution mode byte {other}"
        ))),
    }
}

fn check_transport_mode(
    transport: SetupTransport,
    mode: SetupContributionMode,
) -> Result<(), SerializationError> {
    if transport != SetupTransport::ExpandedMatrix && mode != SetupContributionMode::Recursive {
        return Err(SerializationError::InvalidData(
            "akita-jolt truncated setup transports require recursive setup-contribution mode \
             (direct replay scans the full expanded matrix)"
                .to_string(),
        ));
    }
    Ok(())
}

fn reject_trailing_bytes(rest: &[u8]) -> Result<(), SerializationError> {
    if rest.is_empty() {
        return Ok(());
    }
    Err(SerializationError::InvalidData(format!(
        "akita-jolt blob has {} trailing bytes",
        rest.len()
    )))
}

/// One opening group: a prefix point-variable selection, its claimed
/// openings, and the group commitment.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AkitaJoltGroup<F: FieldCore, E: FieldCore> {
    /// Number of leading opening-point variables this group's polynomials
    /// depend on (`PointVariableSelection::prefix(point_num_vars, num_vars)`).
    pub point_num_vars: u64,
    /// Claimed opening value per polynomial in the group.
    pub openings: Vec<E>,
    /// Ring commitment covering the group's polynomials.
    pub commitment: Commitment<F>,
}

/// Bundled verifier inputs that travel from the host to the Jolt guest.
///
/// `D` is the cyclotomic ring dimension picked by the host config. The
/// guest must use the same `D` to decode commitments. `E` is the public-claim
/// extension field of the config (`Cfg::ExtField`); it collapses to `F` for
/// `EXT_DEGREE == 1` profiles (fp128 D64OneHot) and is `Ext2<F>` for the fp64
/// presets.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AkitaJoltInputs<F: FieldCore, E: FieldCore, const D: usize> {
    /// Domain label both prover and verifier transcripts were initialized with.
    pub transcript_domain: Vec<u8>,
    /// Number of variables of the joint opening point (max group arity).
    pub num_vars: u64,
    /// Setup-contribution mode the proof was generated under. Held in the blob
    /// so host preflight and guest replay verify under the same mode without a
    /// separate flag.
    pub setup_contribution_mode: SetupContributionMode,
    /// How the shared setup matrix travels (full / slice / seed-derived).
    pub setup_transport: SetupTransport,
    /// Number of shared-matrix ring elements shipped (or derived, for
    /// [`SetupTransport::SeedDerived`]). Must equal `seed.max_setup_len` for
    /// [`SetupTransport::ExpandedMatrix`].
    pub shipped_setup_ring_len: u64,
    /// Joint opening point in the multilinear basis, over the claim field `E`.
    pub opening_point: Vec<E>,
    /// Opening groups (Direct: one; recursive profile: two precommits plus
    /// the final group).
    pub groups: Vec<AkitaJoltGroup<F, E>>,
    /// Expanded verifier setup (matrix prefix usable by the verifier kernel)
    /// plus the public setup-prefix commitment registry (`C_S`).
    pub verifier_setup: AkitaVerifierSetup<F>,
    /// Proof shape descriptor; needed to deserialize `proof` without
    /// reconstructing a `Schedule` first.
    pub proof_shape: AkitaBatchedProofShape,
    /// The Akita batched proof itself over `(F, E)`.
    pub proof: AkitaBatchedProof<F, E>,
}

impl<F: FieldCore, E: FieldCore, const D: usize> AkitaJoltInputs<F, E, D> {
    /// Build the verifier claim batch represented by this blob.
    ///
    /// Keeping this projection here prevents host and guest replay from
    /// growing independent claim-shaping code.
    ///
    /// # Errors
    ///
    /// Returns a descriptive error when the decoded group shapes cannot form
    /// a valid opening batch.
    pub fn verifier_opening_batch(
        &self,
    ) -> Result<OpeningClaims<'static, E, &'_ Commitment<F>>, String> {
        let mut groups = Vec::with_capacity(self.groups.len());
        for (idx, group) in self.groups.iter().enumerate() {
            let point_num_vars = usize::try_from(group.point_num_vars)
                .map_err(|_| format!("group {idx} point arity overflows usize"))?;
            let point_vars = PointVariableSelection::prefix(point_num_vars, self.opening_point.len())
                .map_err(|err| format!("group {idx} point selection: {err}"))?;
            groups.push(
                PolynomialGroupClaims::new(point_vars, group.openings.clone(), &group.commitment)
                    .map_err(|err| format!("group {idx} claims: {err}"))?,
            );
        }
        OpeningClaims::from_groups(self.opening_point.clone(), groups)
            .map_err(|err| format!("opening batch: {err}"))
    }

    fn validate_blob_header_bounds(
        transcript_domain_len: usize,
        num_vars: usize,
        opening_point_len: usize,
    ) -> Result<(), SerializationError> {
        if transcript_domain_len > MAX_TRANSCRIPT_DOMAIN_BYTES {
            return Err(SerializationError::LengthLimitExceeded {
                len: u64::try_from(transcript_domain_len).unwrap_or(u64::MAX),
                max: MAX_TRANSCRIPT_DOMAIN_BYTES,
            });
        }
        if num_vars > MAX_BLOB_NUM_VARS {
            return Err(SerializationError::LengthLimitExceeded {
                len: u64::try_from(num_vars).unwrap_or(u64::MAX),
                max: MAX_BLOB_NUM_VARS,
            });
        }
        if opening_point_len != num_vars {
            return Err(SerializationError::InvalidData(format!(
                "akita-jolt blob num_vars={num_vars} does not match opening-point arity {opening_point_len}"
            )));
        }
        Ok(())
    }

    fn validate_group_bounds(
        num_vars: usize,
        point_num_vars: u64,
        num_openings: usize,
    ) -> Result<(), SerializationError> {
        if point_num_vars == 0 || point_num_vars > num_vars as u64 {
            return Err(SerializationError::InvalidData(format!(
                "akita-jolt group point arity {point_num_vars} out of range 1..={num_vars}"
            )));
        }
        if num_openings == 0 || num_openings > MAX_GROUP_OPENINGS {
            return Err(SerializationError::InvalidData(format!(
                "akita-jolt group opening count {num_openings} out of range 1..={MAX_GROUP_OPENINGS}"
            )));
        }
        Ok(())
    }

    fn validate_setup_transport_bounds(
        transport: SetupTransport,
        mode: SetupContributionMode,
        shipped_setup_ring_len: u64,
        seed_max_setup_len: usize,
    ) -> Result<(), SerializationError> {
        check_transport_mode(transport, mode)?;
        if shipped_setup_ring_len == 0 {
            return Err(SerializationError::InvalidData(
                "akita-jolt blob ships an empty setup matrix".to_string(),
            ));
        }
        match transport {
            SetupTransport::ExpandedMatrix => {
                if shipped_setup_ring_len != seed_max_setup_len as u64 {
                    return Err(SerializationError::InvalidData(format!(
                        "akita-jolt expanded-matrix transport must ship the full \
                         {seed_max_setup_len} ring elements, got {shipped_setup_ring_len}"
                    )));
                }
            }
            SetupTransport::MatrixSlice | SetupTransport::SeedDerived => {
                if shipped_setup_ring_len > seed_max_setup_len as u64 {
                    return Err(SerializationError::InvalidData(format!(
                        "akita-jolt shipped setup length {shipped_setup_ring_len} exceeds seed \
                         capacity {seed_max_setup_len}"
                    )));
                }
            }
        }
        Ok(())
    }
}

impl<F, E, const D: usize> AkitaJoltInputs<F, E, D>
where
    F: FieldCore + CanonicalField + AkitaSerialize,
    E: FieldCore + AkitaSerialize,
{
    fn shipped_matrix_prefix(&self) -> Result<Option<FlatMatrix<F>>, SerializationError> {
        if !self.setup_transport.ships_matrix_payload() {
            return Ok(None);
        }
        let shared = self.verifier_setup.expanded.shared_matrix();
        let shipped_len = usize::try_from(self.shipped_setup_ring_len).map_err(|_| {
            SerializationError::LengthLimitExceeded {
                len: self.shipped_setup_ring_len,
                max: usize::MAX,
            }
        })?;
        if shipped_len == shared.total_ring_elements() {
            return Ok(Some(shared.clone()));
        }
        let field_len = shipped_len
            .checked_mul(shared.gen_ring_dim())
            .filter(|&len| len <= shared.as_field_slice().len())
            .ok_or_else(|| {
                SerializationError::InvalidData(
                    "akita-jolt shipped setup slice exceeds the expanded matrix".to_string(),
                )
            })?;
        Ok(Some(FlatMatrix::from_flat_data(
            shared.as_field_slice()[..field_len].to_vec(),
            shared.gen_ring_dim(),
        )))
    }

    /// Encode the bundle into a single contiguous byte vector.
    pub fn write_to_bytes(&self) -> Result<Vec<u8>, SerializationError> {
        Self::validate_blob_header_bounds(
            self.transcript_domain.len(),
            usize::try_from(self.num_vars).map_err(|_| {
                SerializationError::LengthLimitExceeded {
                    len: self.num_vars,
                    max: usize::MAX,
                }
            })?,
            self.opening_point.len(),
        )?;
        if self.groups.is_empty() || self.groups.len() > MAX_BLOB_GROUPS {
            return Err(SerializationError::InvalidData(format!(
                "akita-jolt blob group count {} out of range 1..={MAX_BLOB_GROUPS}",
                self.groups.len()
            )));
        }
        Self::validate_setup_transport_bounds(
            self.setup_transport,
            self.setup_contribution_mode,
            self.shipped_setup_ring_len,
            self.verifier_setup.expanded.seed().max_setup_len,
        )?;
        let shipped_matrix = self.shipped_matrix_prefix()?;

        let mut bytes = Vec::new();
        bytes.extend_from_slice(&BLOB_MAGIC);
        // D is encoded so the guest can fail loudly on a mismatched
        // monomorphization.
        (D as u64).serialize_with_mode(&mut bytes, BLOB_COMPRESS)?;
        self.transcript_domain
            .serialize_with_mode(&mut bytes, BLOB_COMPRESS)?;
        self.num_vars
            .serialize_with_mode(&mut bytes, BLOB_COMPRESS)?;
        setup_mode_to_u8(self.setup_contribution_mode)
            .serialize_with_mode(&mut bytes, BLOB_COMPRESS)?;
        self.setup_transport
            .to_u8()
            .serialize_with_mode(&mut bytes, BLOB_COMPRESS)?;
        self.shipped_setup_ring_len
            .serialize_with_mode(&mut bytes, BLOB_COMPRESS)?;
        self.opening_point
            .serialize_with_mode(&mut bytes, BLOB_COMPRESS)?;
        (self.groups.len() as u64).serialize_with_mode(&mut bytes, BLOB_COMPRESS)?;
        for group in &self.groups {
            Self::validate_group_bounds(
                self.opening_point.len(),
                group.point_num_vars,
                group.openings.len(),
            )?;
            group
                .point_num_vars
                .serialize_with_mode(&mut bytes, BLOB_COMPRESS)?;
            group
                .openings
                .serialize_with_mode(&mut bytes, BLOB_COMPRESS)?;
            let num_commitment_coeffs = u64::try_from(group.commitment.rows().coeff_len())
                .map_err(|_| SerializationError::LengthLimitExceeded {
                    len: group.commitment.rows().coeff_len() as u64,
                    max: usize::MAX,
                })?;
            num_commitment_coeffs.serialize_with_mode(&mut bytes, BLOB_COMPRESS)?;
            group
                .commitment
                .serialize_with_mode(&mut bytes, BLOB_COMPRESS)?;
        }
        self.verifier_setup
            .expanded
            .seed()
            .serialize_with_mode(&mut bytes, BLOB_COMPRESS)?;
        if let Some(matrix) = &shipped_matrix {
            matrix.serialize_with_mode(&mut bytes, BLOB_COMPRESS)?;
        }
        self.verifier_setup
            .prefix_slots
            .serialize_with_mode(&mut bytes, BLOB_COMPRESS)?;
        self.proof_shape
            .serialize_with_mode(&mut bytes, BLOB_COMPRESS)?;
        self.proof.serialize_with_mode(&mut bytes, BLOB_COMPRESS)?;
        if bytes.len() as u64 > MAX_JOLT_BLOB_BYTES {
            return Err(SerializationError::LengthLimitExceeded {
                len: bytes.len() as u64,
                max: MAX_JOLT_BLOB_BYTES as usize,
            });
        }
        Ok(bytes)
    }
}

impl<F, E, const D: usize> AkitaJoltInputs<F, E, D>
where
    F: FieldCore + AkitaSerialize + AkitaDeserialize<Context = ()> + Valid,
    E: FieldCore + AkitaSerialize + AkitaDeserialize<Context = ()> + Valid,
{
    fn decode_capped_bytes(
        rest: &mut &[u8],
        max_len: usize,
        context: &'static str,
    ) -> Result<Vec<u8>, SerializationError> {
        let len = Self::decode_capped_len(rest, max_len)?;
        Self::ensure_remaining(rest, len, context)?;
        let (bytes, tail) = rest.split_at(len);
        *rest = tail;
        Ok(bytes.to_vec())
    }

    fn decode_capped_len(rest: &mut &[u8], max_len: usize) -> Result<usize, SerializationError> {
        let encoded = u64::deserialize_with_mode(rest, BLOB_COMPRESS, BLOB_VALIDATE, &())?;
        let len =
            usize::try_from(encoded).map_err(|_| SerializationError::LengthLimitExceeded {
                len: encoded,
                max: usize::MAX,
            })?;
        if len > max_len {
            return Err(SerializationError::LengthLimitExceeded {
                len: encoded,
                max: max_len,
            });
        }
        Ok(len)
    }

    fn ensure_remaining(
        rest: &[u8],
        len: usize,
        context: &'static str,
    ) -> Result<(), SerializationError> {
        if rest.len() < len {
            return Err(SerializationError::InvalidData(format!(
                "{context} claims {len} bytes but only {} remain",
                rest.len()
            )));
        }
        Ok(())
    }

    fn encoded_payload_len(
        field_elements: usize,
        elem_size: usize,
    ) -> Result<usize, SerializationError> {
        field_elements.checked_mul(elem_size).ok_or_else(|| {
            SerializationError::InvalidData(
                "akita-jolt blob field payload length overflow".to_string(),
            )
        })
    }

    fn encoded_field_payload_len(field_elements: usize) -> Result<usize, SerializationError> {
        Self::encoded_payload_len(field_elements, F::zero().serialized_size(BLOB_COMPRESS))
    }

    /// Upper bound on field elements this decoder may be asked to
    /// materialize for the shared setup matrix.
    ///
    /// Derived from the blob cap rather than from
    /// `MAX_SETUP_MATRIX_FIELD_ELEMENTS`, for the same metadata-vs-allocation
    /// reason as the shipped-length bound above: the DoS surface at this
    /// boundary is *memory the decoder allocates*, and the input itself is
    /// already capped at [`MAX_JOLT_BLOB_BYTES`], so no honest or hostile
    /// blob can require more elements than the cap can encode. Bounding by an
    /// unrelated setup constant instead rejected legitimate blobs: the fp64
    /// D128 nv=23 recursive slice is 630,784 ring elements = 80,740,352 field
    /// elements, over the 2^26 setup constant while being a 646 MB payload
    /// well inside the 768 MiB blob cap.
    ///
    /// This is a bound, not a ceiling raise: the shipped payload must still
    /// fit the bytes actually remaining in the blob
    /// (`check_setup_matrix_bytes_available`), which is the tighter,
    /// input-derived guard. Callers that materialize without reading payload
    /// (the seed-derived transport) are held to the same ceiling so in-guest
    /// derivation cannot be driven unbounded by metadata alone.
    fn max_blob_setup_field_elements() -> usize {
        let elem_size = F::zero().serialized_size(BLOB_COMPRESS).max(1);
        (MAX_JOLT_BLOB_BYTES as usize) / elem_size
    }

    fn decode_ext_elems(
        rest: &mut &[u8],
        len: usize,
        context: &'static str,
    ) -> Result<Vec<E>, SerializationError> {
        let payload_len =
            Self::encoded_payload_len(len, E::zero().serialized_size(BLOB_COMPRESS))?;
        Self::ensure_remaining(rest, payload_len, context)?;
        let mut elems = Vec::with_capacity(len);
        for _ in 0..len {
            elems.push(E::deserialize_with_mode(
                &mut *rest,
                BLOB_COMPRESS,
                BLOB_VALIDATE,
                &(),
            )?);
        }
        Ok(elems)
    }

    fn decode_opening_point(
        rest: &mut &[u8],
        transcript_domain_len: usize,
        num_vars: usize,
    ) -> Result<Vec<E>, SerializationError> {
        let len = Self::decode_capped_len(rest, MAX_BLOB_NUM_VARS)?;
        Self::validate_blob_header_bounds(transcript_domain_len, num_vars, len)?;
        Self::decode_ext_elems(rest, len, "akita-jolt opening point")
    }

    fn decode_groups(
        rest: &mut &[u8],
        num_vars: usize,
    ) -> Result<Vec<AkitaJoltGroup<F, E>>, SerializationError> {
        let num_groups = Self::decode_capped_len(rest, MAX_BLOB_GROUPS)?;
        if num_groups == 0 {
            return Err(SerializationError::InvalidData(
                "akita-jolt blob must carry at least one opening group".to_string(),
            ));
        }
        let mut groups = Vec::with_capacity(num_groups);
        for _ in 0..num_groups {
            let point_num_vars =
                u64::deserialize_with_mode(&mut *rest, BLOB_COMPRESS, BLOB_VALIDATE, &())?;
            let num_openings = Self::decode_capped_len(rest, MAX_GROUP_OPENINGS)?;
            Self::validate_group_bounds(num_vars, point_num_vars, num_openings)?;
            let openings = Self::decode_ext_elems(rest, num_openings, "akita-jolt group openings")?;
            let num_commitment_coeffs =
                Self::decode_capped_len(rest, MAX_SETUP_MATRIX_FIELD_ELEMENTS)?;
            let commitment = Commitment::<F>::deserialize_with_mode(
                &mut *rest,
                BLOB_COMPRESS,
                BLOB_VALIDATE,
                &num_commitment_coeffs,
            )?;
            groups.push(AkitaJoltGroup {
                point_num_vars,
                openings,
                commitment,
            });
        }
        Ok(groups)
    }

    fn setup_matrix_encoded_len(matrix_fields: usize) -> Result<usize, SerializationError> {
        let header_len = 0usize
            .serialized_size(BLOB_COMPRESS)
            .checked_mul(2)
            .ok_or_else(|| {
                SerializationError::InvalidData(
                    "akita-jolt setup matrix header length overflow".to_string(),
                )
            })?;
        let payload_len = Self::encoded_field_payload_len(matrix_fields)?;
        header_len.checked_add(payload_len).ok_or_else(|| {
            SerializationError::InvalidData(
                "akita-jolt setup matrix encoded length overflow".to_string(),
            )
        })
    }

    fn check_setup_matrix_bytes_available(
        rest: &[u8],
        matrix_fields: usize,
    ) -> Result<(), SerializationError> {
        let matrix_len = Self::setup_matrix_encoded_len(matrix_fields)?;
        if rest.len() < matrix_len {
            return Err(SerializationError::InvalidData(format!(
                "akita-jolt setup matrix claims {matrix_len} bytes but only {} remain",
                rest.len()
            )));
        }
        Ok(())
    }

    fn decode_seed_and_shipped_matrix(
        rest: &mut &[u8],
        transport: SetupTransport,
        shipped_setup_ring_len: usize,
    ) -> Result<(AkitaSetupSeed, Option<FlatMatrix<F>>), SerializationError> {
        let seed =
            AkitaSetupSeed::deserialize_with_mode(&mut *rest, BLOB_COMPRESS, BLOB_VALIDATE, &())?;
        if seed.gen_ring_dim != D {
            return Err(SerializationError::InvalidData(format!(
                "akita-jolt setup D={} does not match guest D={D}",
                seed.gen_ring_dim
            )));
        }
        // DoS guard: bound what this decode will actually ALLOCATE, which is
        // the shipped (or seed-derived) prefix, not the capacity the seed
        // declares. The declared `max_setup_len` drives no allocation here -
        // it is metadata, and it is bound into the transcript through the
        // instance descriptor's setup-seed digest, so a decoder that never
        // materializes it has nothing to over-allocate.
        //
        // Bounding the declared capacity instead (the earlier form) rejected
        // legitimate truncated blobs whenever the *prover's* envelope was
        // large: at fp64 D128 nv=23 recursive the seed declares
        // 4,194,304 x 128 = 2^29 field elements, over the 2^26 cap, even
        // though the blob ships a 270,336-element prefix. Checking the
        // allocated length is strictly no weaker for every transport
        // (`ExpandedMatrix` ships the whole matrix, so the two coincide) and
        // unblocks the truncated ones.
        let matrix_fields = shipped_setup_ring_len
            .checked_mul(seed.gen_ring_dim)
            .ok_or_else(|| {
                SerializationError::InvalidData(
                    "akita-jolt shipped setup matrix field count overflow".to_string(),
                )
            })?;
        let max_field_elements = Self::max_blob_setup_field_elements();
        if matrix_fields > max_field_elements {
            return Err(SerializationError::LengthLimitExceeded {
                len: u64::try_from(matrix_fields).unwrap_or(u64::MAX),
                max: max_field_elements,
            });
        }
        if !transport.ships_matrix_payload() {
            return Ok((seed, None));
        }
        Self::check_setup_matrix_bytes_available(rest, matrix_fields)?;
        let shipped_matrix = FlatMatrix::<F>::deserialize_with_expected_shape(
            &mut *rest,
            BLOB_COMPRESS,
            BLOB_VALIDATE,
            shipped_setup_ring_len,
            seed.gen_ring_dim,
            max_field_elements,
        )?;
        Ok((seed, Some(shipped_matrix)))
    }

    fn decode_prefix_slots(
        rest: &mut &[u8],
    ) -> Result<SetupPrefixVerifierRegistry<F>, SerializationError> {
        SetupPrefixVerifierRegistry::deserialize_with_mode(
            &mut *rest,
            BLOB_COMPRESS,
            BLOB_VALIDATE,
            &(),
        )
    }

    fn decode_from_bytes_with_setup(
        bytes: &[u8],
        build_expanded: impl FnOnce(
            SetupTransport,
            usize,
            AkitaSetupSeed,
            Option<FlatMatrix<F>>,
        ) -> Result<AkitaExpandedSetup<F>, SerializationError>,
    ) -> Result<Self, SerializationError> {
        if bytes.len() < BLOB_MAGIC.len() {
            return Err(SerializationError::InvalidData(
                "akita-jolt blob shorter than magic header".to_string(),
            ));
        }
        if bytes.len() as u64 > MAX_JOLT_BLOB_BYTES {
            return Err(SerializationError::LengthLimitExceeded {
                len: bytes.len() as u64,
                max: MAX_JOLT_BLOB_BYTES as usize,
            });
        }
        let (magic, mut rest) = bytes.split_at(BLOB_MAGIC.len());
        if magic != BLOB_MAGIC {
            return Err(SerializationError::InvalidData(
                "akita-jolt blob magic mismatch".to_string(),
            ));
        }
        let encoded_d = u64::deserialize_with_mode(&mut rest, BLOB_COMPRESS, BLOB_VALIDATE, &())?;
        if encoded_d != D as u64 {
            return Err(SerializationError::InvalidData(format!(
                "akita-jolt blob D={encoded_d} doesn't match guest D={D}"
            )));
        }
        let transcript_domain = Self::decode_capped_bytes(
            &mut rest,
            MAX_TRANSCRIPT_DOMAIN_BYTES,
            "akita-jolt transcript domain",
        )?;
        let num_vars = Self::decode_capped_len(&mut rest, MAX_BLOB_NUM_VARS)?;
        let setup_mode_byte =
            u8::deserialize_with_mode(&mut rest, BLOB_COMPRESS, BLOB_VALIDATE, &())?;
        let setup_contribution_mode = setup_mode_from_u8(setup_mode_byte)?;
        let transport_byte =
            u8::deserialize_with_mode(&mut rest, BLOB_COMPRESS, BLOB_VALIDATE, &())?;
        let setup_transport = SetupTransport::from_u8(transport_byte)?;
        check_transport_mode(setup_transport, setup_contribution_mode)?;
        let shipped_setup_ring_len =
            Self::decode_capped_len(&mut rest, MAX_SETUP_MATRIX_FIELD_ELEMENTS)?;
        let opening_point =
            Self::decode_opening_point(&mut rest, transcript_domain.len(), num_vars)?;
        let groups = Self::decode_groups(&mut rest, num_vars)?;
        let (seed, shipped_matrix) = Self::decode_seed_and_shipped_matrix(
            &mut rest,
            setup_transport,
            shipped_setup_ring_len,
        )?;
        Self::validate_setup_transport_bounds(
            setup_transport,
            setup_contribution_mode,
            shipped_setup_ring_len as u64,
            seed.max_setup_len,
        )?;
        let expanded =
            build_expanded(setup_transport, shipped_setup_ring_len, seed, shipped_matrix)?;
        let prefix_slots = Self::decode_prefix_slots(&mut rest)?;
        let proof_shape = AkitaBatchedProofShape::deserialize_with_mode(
            &mut rest,
            BLOB_COMPRESS,
            BLOB_VALIDATE,
            &(),
        )?;
        let proof = AkitaBatchedProof::<F, E>::deserialize_with_mode(
            &mut rest,
            BLOB_COMPRESS,
            BLOB_VALIDATE,
            &proof_shape,
        )?;
        reject_trailing_bytes(rest)?;
        Ok(Self {
            transcript_domain,
            num_vars: num_vars as u64,
            setup_contribution_mode,
            setup_transport,
            shipped_setup_ring_len: shipped_setup_ring_len as u64,
            opening_point,
            groups,
            verifier_setup: AkitaVerifierSetup {
                expanded: Arc::new(expanded),
                prefix_slots,
            },
            proof_shape,
            proof,
        })
    }

    fn missing_matrix_payload_error() -> SerializationError {
        SerializationError::InvalidData(
            "akita-jolt matrix-shipping transport decoded without matrix payload".to_string(),
        )
    }
}

impl<F, E, const D: usize> AkitaJoltInputs<F, E, D>
where
    F: FieldCore + RandomSampling + AkitaSerialize + AkitaDeserialize<Context = ()> + Valid,
    E: FieldCore + AkitaSerialize + AkitaDeserialize<Context = ()> + Valid,
{
    fn derive_matrix_prefix(seed: &AkitaSetupSeed, ring_len: usize) -> FlatMatrix<F> {
        // `derive_public_matrix_flat` is prefix-stable: a vector of length N
        // is a prefix of any longer vector derived from the same seed.
        derive_setup_matrix_prefix::<F, D>(ring_len, &seed.public_matrix_seed)
    }

    fn build_strict_expanded_setup(
        transport: SetupTransport,
        shipped_setup_ring_len: usize,
        seed: AkitaSetupSeed,
        shipped_matrix: Option<FlatMatrix<F>>,
    ) -> Result<AkitaExpandedSetup<F>, SerializationError> {
        match transport {
            SetupTransport::ExpandedMatrix => {
                let matrix = shipped_matrix.ok_or_else(Self::missing_matrix_payload_error)?;
                AkitaExpandedSetup::from_verified_parts(seed, matrix)
            }
            SetupTransport::MatrixSlice => {
                let matrix = shipped_matrix.ok_or_else(Self::missing_matrix_payload_error)?;
                let expected = Self::derive_matrix_prefix(&seed, shipped_setup_ring_len);
                if matrix.as_field_slice() != expected.as_field_slice() {
                    return Err(SerializationError::InvalidData(
                        "akita-jolt setup matrix slice does not match public matrix seed"
                            .to_string(),
                    ));
                }
                Ok(AkitaExpandedSetup::from_trusted_seed_derived_parts_unchecked(seed, matrix))
            }
            SetupTransport::SeedDerived => {
                let matrix = Self::derive_matrix_prefix(&seed, shipped_setup_ring_len);
                Ok(AkitaExpandedSetup::from_trusted_seed_derived_parts_unchecked(seed, matrix))
            }
        }
    }

    /// Strictly decode the bundle from bytes produced by [`Self::write_to_bytes`].
    ///
    /// This path rederives the public setup matrix (or shipped prefix slice)
    /// from its seed and rejects stale or corrupted cached matrix bytes.
    /// Host-side artifact checks should use this path.
    pub fn read_from_bytes(bytes: &[u8]) -> Result<Self, SerializationError> {
        Self::decode_from_bytes_with_setup(bytes, Self::build_strict_expanded_setup)
    }
}

#[cfg(any(
    feature = "trusted-benchmark-artifact",
    akita_trusted_benchmark_artifact
))]
impl<F, E, const D: usize> AkitaJoltInputs<F, E, D>
where
    F: FieldCore + RandomSampling + AkitaSerialize + AkitaDeserialize<Context = ()> + Valid,
    E: FieldCore + AkitaSerialize + AkitaDeserialize<Context = ()> + Valid,
{
    fn build_trusted_expanded_setup(
        transport: SetupTransport,
        shipped_setup_ring_len: usize,
        seed: AkitaSetupSeed,
        shipped_matrix: Option<FlatMatrix<F>>,
    ) -> Result<AkitaExpandedSetup<F>, SerializationError> {
        match transport {
            SetupTransport::ExpandedMatrix | SetupTransport::MatrixSlice => {
                let matrix = shipped_matrix.ok_or_else(Self::missing_matrix_payload_error)?;
                Ok(AkitaExpandedSetup::from_trusted_seed_derived_parts_unchecked(seed, matrix))
            }
            // Seed-derived transport carries no matrix bytes to trust: the
            // decoder always re-derives, so this path is strict by
            // construction.
            SetupTransport::SeedDerived => {
                let matrix = Self::derive_matrix_prefix(&seed, shipped_setup_ring_len);
                Ok(AkitaExpandedSetup::from_trusted_seed_derived_parts_unchecked(seed, matrix))
            }
        }
    }

    /// Decode a host-produced recursion artifact while trusting cached
    /// setup-matrix bytes.
    ///
    /// This is a benchmark/profile fast path, not a general recursion security
    /// boundary. It still validates the blob magic, ring dimension, serialized
    /// structure, field elements, and seed/matrix shape metadata, but for the
    /// matrix-shipping transports it deliberately skips checking that the
    /// shipped matrix coefficients equal the prefix derived from the seed.
    /// The seed-derived transport has no bytes to trust and is re-derived
    /// exactly as in [`Self::read_from_bytes`].
    pub fn read_trusted_host_artifact_bytes(bytes: &[u8]) -> Result<Self, SerializationError> {
        Self::decode_from_bytes_with_setup(bytes, Self::build_trusted_expanded_setup)
    }
}

/// Derive the first `ring_len` shared-matrix ring elements from a public
/// seed.
///
/// Thin wrapper over [`derive_public_matrix_flat`] so host-side slice sizing
/// and guest-side re-derivation share one call site.
#[must_use]
pub fn derive_setup_matrix_prefix<F: FieldCore + RandomSampling, const D: usize>(
    ring_len: usize,
    seed: &akita_types::PublicMatrixSeed,
) -> FlatMatrix<F> {
    derive_public_matrix_flat::<F, D>(ring_len, seed)
}

// `akita-algebra` is pulled in only so that downstream consumers can rely on
// `Commitment<F>` having all of its trait bounds satisfied; declare it
// here to avoid a `cargo machete` style trim.
#[doc(hidden)]
pub use akita_algebra as _akita_algebra_dep;

#[cfg(test)]
mod tests {
    use super::*;
    use akita_field::Prime128Offset275;
    use akita_types::{
        sample_public_matrix_seed, setup_prefix_slot_id, AjtaiKeyParams, PolynomialGroupLayout,
        PrecommittedGroupParams, PrecommittedLevelParams, RingVec, SetupPrefixPublicCommitment,
        SetupPrefixVerifierSlot, SisModulusFamily,
    };

    type TestF = Prime128Offset275;
    const TEST_D: usize = 32;

    fn blob_prefix() -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&BLOB_MAGIC);
        (TEST_D as u64)
            .serialize_with_mode(&mut bytes, BLOB_COMPRESS)
            .unwrap();
        bytes
    }

    fn prefix_commitment_params() -> PrecommittedLevelParams {
        PrecommittedLevelParams {
            layout: PrecommittedGroupParams {
                group: PolynomialGroupLayout::singleton(TEST_D.trailing_zeros() as usize),
                m_vars: 0,
                r_vars: 0,
                log_basis: 1,
                n_a: 1,
                conservative_n_b: 1,
                log_commit_bound: 1,
                onehot_chunk_size: 1,
                basis_range: (1, 1),
            },
            a_key: AjtaiKeyParams::new_unchecked(
                akita_types::DEFAULT_SIS_SECURITY_BITS,
                SisModulusFamily::Q128,
                1,
                1,
                1,
                TEST_D,
            ),
            b_key: AjtaiKeyParams::new_unchecked(
                akita_types::DEFAULT_SIS_SECURITY_BITS,
                SisModulusFamily::Q128,
                1,
                1,
                1,
                TEST_D,
            ),
            num_blocks: 1,
            block_len: 1,
            num_digits_commit: 1,
            num_digits_open: 1,
            num_digits_fold_one: 1,
        }
    }

    fn test_seed(max_setup_len: usize) -> AkitaSetupSeed {
        AkitaSetupSeed {
            max_num_vars: 8,
            max_num_batched_polys: 1,
            gen_ring_dim: TEST_D,
            max_setup_len,
            public_matrix_seed: sample_public_matrix_seed(),
        }
    }

    #[test]
    fn trailing_blob_bytes_are_rejected() {
        let err = reject_trailing_bytes(&[0]).unwrap_err();
        assert!(err.to_string().contains("trailing bytes"));
        reject_trailing_bytes(&[]).unwrap();
    }

    #[test]
    fn transcript_domain_len_is_capped_before_allocation() {
        let mut bytes = blob_prefix();
        ((MAX_TRANSCRIPT_DOMAIN_BYTES + 1) as u64)
            .serialize_with_mode(&mut bytes, BLOB_COMPRESS)
            .unwrap();

        let err = AkitaJoltInputs::<TestF, TestF, TEST_D>::read_from_bytes(&bytes).unwrap_err();
        assert!(err.to_string().contains("length"));
    }

    #[test]
    fn num_vars_is_capped_before_opening_point_allocation() {
        let mut bytes = blob_prefix();
        Vec::<u8>::new()
            .serialize_with_mode(&mut bytes, BLOB_COMPRESS)
            .unwrap();
        ((MAX_BLOB_NUM_VARS + 1) as u64)
            .serialize_with_mode(&mut bytes, BLOB_COMPRESS)
            .unwrap();

        let err = AkitaJoltInputs::<TestF, TestF, TEST_D>::read_from_bytes(&bytes).unwrap_err();
        assert!(err.to_string().contains("length"));
    }

    #[test]
    fn opening_point_len_must_match_num_vars_before_allocation() {
        let mut bytes = blob_prefix();
        Vec::<u8>::new()
            .serialize_with_mode(&mut bytes, BLOB_COMPRESS)
            .unwrap();
        2u64.serialize_with_mode(&mut bytes, BLOB_COMPRESS).unwrap();
        setup_mode_to_u8(SetupContributionMode::Direct)
            .serialize_with_mode(&mut bytes, BLOB_COMPRESS)
            .unwrap();
        SetupTransport::ExpandedMatrix
            .to_u8()
            .serialize_with_mode(&mut bytes, BLOB_COMPRESS)
            .unwrap();
        1u64.serialize_with_mode(&mut bytes, BLOB_COMPRESS).unwrap();
        3u64.serialize_with_mode(&mut bytes, BLOB_COMPRESS).unwrap();

        let err = AkitaJoltInputs::<TestF, TestF, TEST_D>::read_from_bytes(&bytes).unwrap_err();
        assert!(err.to_string().contains("opening-point arity 3"));
    }

    #[test]
    fn truncated_setup_transports_require_recursive_mode() {
        for transport in [SetupTransport::MatrixSlice, SetupTransport::SeedDerived] {
            let err =
                check_transport_mode(transport, SetupContributionMode::Direct).unwrap_err();
            assert!(err.to_string().contains("recursive"));
            check_transport_mode(transport, SetupContributionMode::Recursive).unwrap();
        }
        check_transport_mode(
            SetupTransport::ExpandedMatrix,
            SetupContributionMode::Direct,
        )
        .unwrap();
    }

    #[test]
    fn group_count_is_capped() {
        let mut bytes = blob_prefix();
        Vec::<u8>::new()
            .serialize_with_mode(&mut bytes, BLOB_COMPRESS)
            .unwrap();
        1u64.serialize_with_mode(&mut bytes, BLOB_COMPRESS).unwrap();
        setup_mode_to_u8(SetupContributionMode::Direct)
            .serialize_with_mode(&mut bytes, BLOB_COMPRESS)
            .unwrap();
        SetupTransport::ExpandedMatrix
            .to_u8()
            .serialize_with_mode(&mut bytes, BLOB_COMPRESS)
            .unwrap();
        1u64.serialize_with_mode(&mut bytes, BLOB_COMPRESS).unwrap();
        // Opening point: one element.
        1u64.serialize_with_mode(&mut bytes, BLOB_COMPRESS).unwrap();
        TestF::zero()
            .serialize_with_mode(&mut bytes, BLOB_COMPRESS)
            .unwrap();
        ((MAX_BLOB_GROUPS + 1) as u64)
            .serialize_with_mode(&mut bytes, BLOB_COMPRESS)
            .unwrap();

        let err = AkitaJoltInputs::<TestF, TestF, TEST_D>::read_from_bytes(&bytes).unwrap_err();
        assert!(err.to_string().contains("length"));
    }

    #[test]
    fn strict_setup_builder_verifies_full_matrix() {
        let seed = test_seed(2);
        let shared =
            derive_setup_matrix_prefix::<TestF, TEST_D>(2, &seed.public_matrix_seed);
        let built = AkitaJoltInputs::<TestF, TestF, TEST_D>::build_strict_expanded_setup(
            SetupTransport::ExpandedMatrix,
            2,
            seed.clone(),
            Some(shared),
        )
        .expect("verified full matrix");
        assert_eq!(built.seed().max_setup_len, 2);

        let wrong = FlatMatrix::from_flat_data(vec![TestF::zero(); 2 * TEST_D], TEST_D);
        let err = AkitaJoltInputs::<TestF, TestF, TEST_D>::build_strict_expanded_setup(
            SetupTransport::ExpandedMatrix,
            2,
            seed,
            Some(wrong),
        )
        .unwrap_err();
        assert!(err.to_string().contains("matrix"));
    }

    #[test]
    fn strict_setup_builder_checks_slice_against_seed_prefix() {
        let seed = test_seed(4);
        let slice =
            derive_setup_matrix_prefix::<TestF, TEST_D>(2, &seed.public_matrix_seed);
        // Prefix property: the 2-element slice must be accepted against the
        // 4-element seed capacity.
        let built = AkitaJoltInputs::<TestF, TestF, TEST_D>::build_strict_expanded_setup(
            SetupTransport::MatrixSlice,
            2,
            seed.clone(),
            Some(slice.clone()),
        )
        .expect("verified matrix slice");
        assert_eq!(built.shared_matrix().total_ring_elements(), 2);
        let full =
            derive_setup_matrix_prefix::<TestF, TEST_D>(4, &seed.public_matrix_seed);
        assert_eq!(
            full.as_field_slice()[..2 * TEST_D],
            *slice.as_field_slice(),
            "flat derivation must be prefix-stable"
        );

        let mut tampered_data = slice.as_field_slice().to_vec();
        tampered_data[0] += TestF::one();
        let tampered = FlatMatrix::from_flat_data(tampered_data, TEST_D);
        let err = AkitaJoltInputs::<TestF, TestF, TEST_D>::build_strict_expanded_setup(
            SetupTransport::MatrixSlice,
            2,
            seed,
            Some(tampered),
        )
        .unwrap_err();
        assert!(err.to_string().contains("does not match"));
    }

    #[test]
    fn seed_derived_setup_builder_matches_direct_derivation() {
        let seed = test_seed(4);
        let built = AkitaJoltInputs::<TestF, TestF, TEST_D>::build_strict_expanded_setup(
            SetupTransport::SeedDerived,
            3,
            seed.clone(),
            None,
        )
        .expect("seed-derived setup");
        let expected =
            derive_setup_matrix_prefix::<TestF, TEST_D>(3, &seed.public_matrix_seed);
        assert_eq!(built.shared_matrix(), &expected);
    }

    #[test]
    fn strict_setup_decoder_preserves_prefix_slots() {
        let seed = test_seed(2);
        let shared =
            derive_setup_matrix_prefix::<TestF, TEST_D>(2, &seed.public_matrix_seed);
        let id = setup_prefix_slot_id(TEST_D, 1, prefix_commitment_params());
        let mut prefix_slots = SetupPrefixVerifierRegistry::new();
        prefix_slots
            .insert(SetupPrefixVerifierSlot {
                id: id.clone(),
                natural_len: 1,
                padded_len: TEST_D,
                commitment: SetupPrefixPublicCommitment {
                    rows: vec![RingVec::from_coeffs(vec![TestF::zero(); TEST_D])],
                },
            })
            .expect("insert prefix slot");

        let mut bytes = Vec::new();
        prefix_slots
            .serialize_with_mode(&mut bytes, BLOB_COMPRESS)
            .unwrap();

        let mut rest = &bytes[..];
        let decoded =
            AkitaJoltInputs::<TestF, TestF, TEST_D>::decode_prefix_slots(&mut rest)
                .expect("decode prefix slots");
        assert!(rest.is_empty());
        assert!(decoded.get(&id).is_some());
        assert_eq!(decoded.len(), 1);

        // The strict builder still accepts the full-matrix payload the slots
        // ride along with.
        AkitaJoltInputs::<TestF, TestF, TEST_D>::build_strict_expanded_setup(
            SetupTransport::ExpandedMatrix,
            2,
            seed,
            Some(shared),
        )
        .expect("decode setup");
    }

    #[test]
    fn decode_bounds_the_allocated_prefix_not_the_declared_capacity() {
        // A seed may legitimately declare a capacity far beyond the decode
        // cap (the prover's envelope); what must be bounded is the prefix the
        // decoder actually materializes.
        let huge_capacity = MAX_SETUP_MATRIX_FIELD_ELEMENTS / TEST_D + 1024;
        let seed = test_seed(huge_capacity);
        assert!(
            seed.matrix_field_elements().unwrap() > MAX_SETUP_MATRIX_FIELD_ELEMENTS,
            "test is vacuous unless the declared capacity exceeds the cap"
        );
        // Seed-derived transport allocates only the shipped prefix: accepted.
        let mut bytes = Vec::new();
        seed.serialize_with_mode(&mut bytes, BLOB_COMPRESS).unwrap();
        let mut rest = &bytes[..];
        let (decoded_seed, matrix) =
            AkitaJoltInputs::<TestF, TestF, TEST_D>::decode_seed_and_shipped_matrix(
                &mut rest,
                SetupTransport::SeedDerived,
                4,
            )
            .expect("small prefix under a large declared capacity must decode");
        assert!(matrix.is_none());
        assert_eq!(decoded_seed.max_setup_len, huge_capacity);

        // An oversized shipped prefix is still rejected before allocation.
        let mut rest = &bytes[..];
        let err = AkitaJoltInputs::<TestF, TestF, TEST_D>::decode_seed_and_shipped_matrix(
            &mut rest,
            SetupTransport::SeedDerived,
            huge_capacity,
        )
        .unwrap_err();
        assert!(err.to_string().contains("exceeds maximum"));
    }

    /// Regression guard for the setup-matrix decode bound.
    ///
    /// The bound must come from what the blob can carry, not from
    /// `MAX_SETUP_MATRIX_FIELD_ELEMENTS`. Reverting to the setup constant
    /// re-breaks the fp64 D128 nv=23 recursive slice, which is 630,784 ring
    /// elements = 80,740,352 field elements: over the 2^26 constant, but a
    /// 646 MB payload well inside the 768 MiB blob cap.
    #[test]
    fn setup_matrix_decode_bound_is_derived_from_the_blob_cap() {
        use akita_field::Prime64Offset59;
        type Fp64 = Prime64Offset59;
        const PROD_D: usize = 128;

        let max_fields =
            AkitaJoltInputs::<Fp64, Fp64, PROD_D>::max_blob_setup_field_elements();
        assert_eq!(max_fields, MAX_JOLT_BLOB_BYTES as usize / 8);
        // The real slice must now be admissible where the setup constant
        // rejected it.
        let recursive_slice_fields = 630_784usize * PROD_D;
        assert!(
            recursive_slice_fields > MAX_SETUP_MATRIX_FIELD_ELEMENTS,
            "test is vacuous unless the slice exceeds the old constant"
        );
        assert!(
            recursive_slice_fields <= max_fields,
            "fp64 D128 nv=23 recursive slice must fit the blob-derived bound"
        );

        // Anything past the blob-derived bound is still rejected, before any
        // allocation and without reading payload.
        let seed = AkitaSetupSeed {
            max_num_vars: 23,
            max_num_batched_polys: 4,
            gen_ring_dim: PROD_D,
            max_setup_len: usize::MAX / PROD_D,
            public_matrix_seed: sample_public_matrix_seed(),
        };
        let mut bytes = Vec::new();
        seed.serialize_with_mode(&mut bytes, BLOB_COMPRESS).unwrap();
        let mut rest = &bytes[..];
        let err = AkitaJoltInputs::<Fp64, Fp64, PROD_D>::decode_seed_and_shipped_matrix(
            &mut rest,
            SetupTransport::MatrixSlice,
            max_fields / PROD_D + 1,
        )
        .unwrap_err();
        assert!(err.to_string().contains("exceeds maximum"));
    }

    #[test]
    fn setup_matrix_payload_must_fit_remaining_blob_before_allocation() {
        let err = AkitaJoltInputs::<TestF, TestF, TEST_D>::check_setup_matrix_bytes_available(&[], 1)
            .unwrap_err();
        assert!(err.to_string().contains("setup matrix claims"));
    }
}
