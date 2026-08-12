# Frozen family-H causal 21-day admission audit

## Contract

The raw all-family H residual score was joined to its exact side and
`policy_label_available_ts` lineage, then passed through the canonical
pooled-parent/side-shrunk 21-day causal map:

- 20 fixed score-rank bins;
- 5% trimmed outcome means;
- pooled common-bps parent with side shrinkage at 500 references;
- strict `label_available_ts < decision timestamp` boundary;
- pooled global ranking only after admission.

Fold-qualified candidate identities were used because candidate IDs recur across
outer folds.

## Result

The declared +50-bps admission floor admits **0 of 275,540 outer-test rows**.
The mapped test expected-net distribution has:

- mean −68.03 bps;
- median −69.56 bps;
- 95th percentile −36.40 bps;
- maximum +30.65 bps.

Thus the frozen H score does not merely fail the gate at the margin: its causal
21-day map never reaches the required +50-bps threshold.

## Sensitivity diagnostic (not a promotion result)

| Floor | Eligible test rows | Share | Realized net of all eligible rows |
|---:|---:|---:|---:|
| 0 bps | 2,133 | 0.774% | −7.53 bps |
| 10 bps | 634 | 0.230% | +20.25 bps |
| 25 bps | 315 | 0.114% | +6.47 bps |
| 50 bps | 0 | 0.000% | — |

At a zero floor, the first 0.5% mapped tail is +12.79 realized net bps/trade;
the full admitted set is negative. At 10 and 25 bps, realized economics are
well below the mapped hurdle, so lowering the floor would not establish a
reliable admission policy.

## Decision

The current family-H residual stack fails the causal admission-readiness gate.
Do not promote it or relax the +50-bps floor. The next repair must change the
score-to-net conversion or the short/side economics before another policy
replay; additional family leaf weighting is not justified by this evidence.

Artifacts:

- [admission input with lineage](/Users/remyroche/Documents/Ares/data_perp/artifacts/long_family_raw_h_admission_21d_20260808_v1/admission_input.parquet)
- [causal admission replay](/Users/remyroche/Documents/Ares/data_perp/artifacts/long_family_raw_h_admission_21d_20260808_v2)
- [input materializer](/Users/remyroche/Documents/Ares/scripts/materialize_family_admission_input.py)
