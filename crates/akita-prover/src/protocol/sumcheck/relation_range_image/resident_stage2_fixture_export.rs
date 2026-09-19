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
fn resident_stage2_exports_actual_constructor_additional_fixtures() {
    let dir = std::env::var_os("AKITA_STAGE2_FIXTURE_DIR")
        .expect("fixture output directory must be bound by runner");
    let dir = Path::new(&dir);
    std::fs::create_dir(dir).unwrap();
    let mut manifest =
        String::from("case\tlanes\tinitial_coefficients\trounds\tinput_bytes\texpected_bytes\n");
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
                assert!(c.is_power_of_two() && c >= 32);
                let lanes = dense.len() / c;
                let (mut p, compact) = fixture_with_linear(
                    lanes,
                    c.trailing_zeros() as usize,
                    digit_basis,
                    full_norm,
                    Some((linear, dense)),
                );
                let mut point = (0..lanes.next_power_of_two().trailing_zeros() as usize
                    + c.trailing_zeros() as usize)
                    .map(|i| field(i as u64 + 400))
                    .collect::<Vec<_>>();
                if full_norm {
                    point[2] = F::zero();
                }
                let mut additional =
                    super::super::additional_terms::resident_additional::tests::fixture(
                        &compact,
                        &point,
                        case % 4,
                    );
                let r0 = field(901);
                let r1 = field(902);
                let alpha =
                    RelationRangeImageProver::fold_alpha_two_rounds(&p.common_alpha_factor, r0, r1);
                p.split_eq.bind(r0);
                p.split_eq.bind(r1);
                p.linear_terms.fold_two_coefficients(r0, r1);
                additional.bind(r0);
                additional.bind(r1);
                let (mut witness, _, _) = p
                    .materialize_two_round_compact_prefix_and_compute_next_round(
                        &compact,
                        &alpha,
                        &p.linear_terms,
                        r0,
                        r1,
                    );
                p.common_alpha_factor = alpha;
                p.rounds_completed = 2;
                let initial_c = c / 4;
                let rounds = u64::from(initial_c.trailing_zeros() - 1).min(3);
                assert!(rounds >= 2);
                let mut input = Vec::new();
                words(
                    &mut input,
                    [0x35414753, lanes as u64, initial_c as u64, rounds, 0],
                );
                ext(&mut input, &witness);
                let mut expected = Vec::new();
                for round in 0..rounds {
                    let rho = field(1000 + round);
                    p.split_eq.bind(rho);
                    p.fold_linear_terms_for_current_round(rho);
                    additional.bind(rho);
                    let mut alpha = p.common_alpha_factor.clone();
                    fold_evals_in_place(&mut alpha, rho);
                    let request = Request::coefficients(&p, &witness, &alpha, rho).unwrap();
                    let wire = p
                        .linear_terms
                        .serialize_resident(Limits {
                            source_count: 1 << 16,
                            source_elements: 1 << 20,
                            references: 1 << 20,
                            lanes: 1 << 16,
                        })
                        .unwrap();
                    assert_eq!(wire.lanes, lanes as u64);
                    assert_eq!(wire.coefficients, alpha.len() as u64);
                    let pairs = lanes * (request.input_coefficients / 4);
                    let first = request.eq_first.len();
                    assert!(first.is_power_of_two() && first <= pairs);
                    let second = pairs.div_ceil(first);
                    assert!(second <= request.eq_second.len());
                    // Native visits live lanes only. Omit unused padded equality/weight suffixes.
                    words(
                        &mut input,
                        [
                            first as u64,
                            second as u64,
                            u64::from(request.skip_linear),
                            rho.c0().to_canonical_u64_checked().unwrap(),
                            rho.c1().to_canonical_u64_checked().unwrap(),
                            wire.sources.len() as u64,
                            wire.source_records.len() as u64,
                            wire.references.len() as u64,
                        ],
                    );
                    ext(&mut input, request.alpha);
                    ext(&mut input, &request.lane_weights[..lanes]);
                    ext(&mut input, request.eq_first);
                    ext(&mut input, &request.eq_second[..second]);
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
                    let additional_wire = additional
                        .serialize_resident_additional(lanes * alpha.len(), 1 << 20)
                        .unwrap();
                    words(
                        &mut input,
                        [
                            additional_wire.domain_len,
                            additional_wire.binary_batching[0],
                            additional_wire.binary_batching[1],
                            additional_wire.pairs.len() as u64,
                        ],
                    );
                    for pair in additional_wire.pairs {
                        words(&mut input, pair);
                    }
                    let (next, norm, relation) =
                        p.fuse_folded_coefficients_and_compute_next_round(&witness, &alpha, rho);
                    let norm = match norm {
                        NormRoundTerms::Full(v) => v,
                        NormRoundTerms::SkipLinear([a, c]) => [a, F::zero(), c],
                    };
                    ext(&mut expected, &norm);
                    ext(&mut expected, &relation);
                    let mut cubic = additional.round_polynomial_folded(&next).coeffs;
                    assert!(cubic.len() <= 4);
                    cubic.resize(4, F::zero());
                    ext(&mut expected, &cubic);
                    p.common_alpha_factor = alpha;
                    p.rounds_completed += 1;
                    witness = next;
                }
                ext(&mut expected, &witness);
                let name = format!("case-{case}");
                write_new(&dir.join(format!("{name}.input")), &input);
                write_new(&dir.join(format!("{name}.expected")), &expected);
                manifest += &format!(
                    "{name}\t{lanes}\t{initial_c}\t{rounds}\t{}\t{}\n",
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
