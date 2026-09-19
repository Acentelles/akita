//! Native interoperability fixtures from actual checked packing constructors.
//! Expected values come only from unchanged pinned optimized CPU kernels.
use super::super::evaluation_trace::resident_stage2_descriptors::Limits;
use super::*;
use akita_field::CanonicalU64;
use std::io::Write;
use std::path::Path;
fn words(out: &mut Vec<u8>, values: impl IntoIterator<Item = u64>) {
    for word in values {
        out.extend_from_slice(&word.to_le_bytes());
    }
}
fn ext(out: &mut Vec<u8>, values: &[F]) {
    for &x in values {
        words(
            out,
            [
                x.c0().to_canonical_u64_checked().unwrap(),
                x.c1().to_canonical_u64_checked().unwrap(),
            ],
        );
    }
}
fn write_new(path: &Path, bytes: &[u8]) {
    let mut f = std::fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(path)
        .unwrap();
    f.write_all(bytes).unwrap();
    f.sync_all().unwrap();
}
#[test]
fn resident_stage2_exports_compact_entry_fixtures() {
    let dir =
        std::env::var_os("AKITA_STAGE2_COMPACT_FIXTURE_DIR").expect("bound fixture directory");
    let dir = Path::new(&dir);
    std::fs::create_dir(dir).unwrap();
    let mut manifest = String::from(
        "case\tlanes\tcoefficients\tbasis\tskip_linear\tinput_bytes\texpected_bytes\n",
    );
    let mut case = 0;
    for packing_basis in [
        akita_types::BasisMode::Lagrange,
        akita_types::BasisMode::Monomial,
    ] {
        for digit_basis in [4, 8] {
            for full_norm in [false, true] {
                let (linear, dense, c) =
                    super::super::coefficient_packing_terms::tests::resident_semantic_fixture(
                        packing_basis,
                    );
                let lanes = dense.len() / c;
                let (mut p, compact) = fixture_with_linear(
                    lanes,
                    c.trailing_zeros() as usize,
                    digit_basis,
                    full_norm,
                    Some((linear, dense)),
                );
                let r0 = field(901);
                let r1 = field(902);
                let alpha =
                    RelationRangeImageProver::fold_alpha_two_rounds(&p.common_alpha_factor, r0, r1);
                p.split_eq.bind(r0);
                p.split_eq.bind(r1);
                p.linear_terms.fold_two_coefficients(r0, r1);
                let (witness, norm, relation) = p
                    .materialize_two_round_compact_prefix_and_compute_next_round(
                        &compact,
                        &alpha,
                        &p.linear_terms,
                        r0,
                        r1,
                    );
                let (skip, norm) = match norm {
                    NormRoundTerms::Full(v) => (false, v),
                    NormRoundTerms::SkipLinear([a, c]) => (true, [a, F::zero(), c]),
                };
                assert_eq!(skip, !full_norm);
                let wire = p
                    .linear_terms
                    .serialize_resident(Limits {
                        source_count: 1 << 16,
                        source_elements: 1 << 20,
                        references: 1 << 20,
                        lanes: 1 << 16,
                    })
                    .unwrap();
                assert_eq!(wire.coefficients, c as u64 / 4);
                assert_eq!(wire.lanes, lanes as u64);
                let (first, second) = p.split_eq.remaining_eq_tables();
                let pairs = lanes * (c / 8);
                let needed = pairs.div_ceil(first.len());
                assert!(
                    first.len().is_power_of_two() && first.len() <= pairs && needed <= second.len()
                );
                let mut input = Vec::new();
                words(
                    &mut input,
                    [0x31434b41, lanes as u64, c as u64, digit_basis as u64, 0],
                );
                input.extend(compact.iter().map(|&x| x as u8));
                words(
                    &mut input,
                    [
                        first.len() as u64,
                        needed as u64,
                        u64::from(skip),
                        r0.c0().to_canonical_u64_checked().unwrap(),
                        r0.c1().to_canonical_u64_checked().unwrap(),
                        r1.c0().to_canonical_u64_checked().unwrap(),
                        r1.c1().to_canonical_u64_checked().unwrap(),
                        wire.sources.len() as u64,
                        wire.source_records.len() as u64,
                        wire.references.len() as u64,
                    ],
                );
                ext(&mut input, &alpha);
                ext(&mut input, &p.relation_lane_weights[..lanes]);
                ext(&mut input, first);
                ext(&mut input, &second[..needed]);
                for x in wire.sources {
                    words(&mut input, x);
                }
                for x in wire.source_records {
                    words(&mut input, x);
                }
                words(&mut input, wire.lane_offsets);
                for x in wire.references {
                    words(&mut input, x);
                }
                let mut expected = Vec::new();
                ext(&mut expected, &norm);
                ext(&mut expected, &relation);
                ext(&mut expected, &witness);
                let name = format!("case-{case}");
                write_new(&dir.join(format!("{name}.input")), &input);
                write_new(&dir.join(format!("{name}.expected")), &expected);
                manifest += &format!(
                    "{name}\t{lanes}\t{c}\t{digit_basis}\t{}\t{}\t{}\n",
                    u8::from(skip),
                    input.len(),
                    expected.len()
                );
                case += 1;
            }
        }
    }
    assert_eq!(case, 8);
    write_new(&dir.join("manifest.tsv"), manifest.as_bytes());
}
