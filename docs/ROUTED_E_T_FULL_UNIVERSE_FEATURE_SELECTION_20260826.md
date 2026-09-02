# Routed E/T full-universe feature-selection audit

Status: rejected as an E/T replacement.  This is an offline research receipt;
it does not modify B0, the canonical research stack, or any live artifact.

## Contract

The test used the frozen strict-OOF timestamp-local top-50% router, then fit
the existing direct E and T targets only on routed rows whose supportive and
policy labels resolved before the 28-day reserve.  B0 was deliberately absent.

The source contained 1,407 numeric causal fields.  Of these, 1,199 cleared the
minimum coverage/variance hygiene test across the relevant months; only
near-duplicates at absolute Spearman correlation >= 0.995 were removed,
leaving 1,094 fields.  Screening used three strict held months (Feb--Apr
2026), full-model gain, general and precision-region OOF TreeSHAP, univariate
economic rescue, randomized stability, and semantic-family rescue.  It was
then followed by OOF within-timestamp economic and Top-10-boundary MDA and a
120/90/70/50/35/25 subset ladder.

All target/path fields remained outcome-only joins.  The 156 missing November
supportive rows were explicitly recorded label-invalid and excluded from model
fitting/evaluation rather than encoded as adverse outcomes.

## Results

The full 1,094-field cheap screen found signal, but it did not compress into a
portable 120-or-fewer contract.  The decisive comparison below uses the same
strict Feb--Apr folds, routed candidates, targets, reserve, sample caps and
model configuration.

| Head | Contract | Features | Top-1% policy EV | Top-5% policy EV | Top-10% policy EV | Stable Top-10 score | Q10 week | Worst month Top-10 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| E | existing frozen control | 120 | +120.22 | +103.28 | +80.38 | +74.35 | +40.20 | +71.28 |
| E | best new compact subset | 90 | +12.89 | +9.83 | +5.90 | +1.23 | -25.24 | -4.52 |
| T | existing frozen control | 120 | +79.65 | +68.06 | +48.49 | +42.22 | +6.69 | +24.34 |
| T | best new compact subset | 70 | -47.64 | -53.69 | -51.53 | -58.83 | -100.15 | -68.01 |

Therefore neither new E nor new T contract advances.  The existing 120-field
E/T contracts remain the correct inputs for the B0-replacement study.

## Funding result

Funding was present: 125 funding/carry-related candidate fields survived
hygiene.  The broad screen initially retained five E and two T funding/OI
interaction features.  Cross-fold MDA did not validate them:

- E funding-family MDA: Top-10 delta -0.26 bps; stable-Top-10 delta -0.20
  bps when the family is removed.
- T funding-family MDA: effectively zero incremental effect.

Consequently no *new* funding field is promoted.  This does not remove the
funding interactions already in the frozen control contract, including
`q_lower_tail__oi_3d_x_funding`, `q_lower_tail__oi_7d_x_funding`, and
`xs_dispersion__funding_per_hour`.

## Receipts and scripts

- Full-universe strict screens: `data_perp/artifacts/strict_r3_routed_et_fulluniverse_screen_20260826_v7_*`
- Cross-fold screen: `data_perp/artifacts/strict_r3_routed_et_fulluniverse_screen_crossfold_20260826_v2`
- MDA receipts: `data_perp/artifacts/strict_r3_routed_et_mda_20260826_v6_*`
- Cross-fold MDA and subset contracts: `data_perp/artifacts/strict_r3_routed_et_mda_crossfold_20260826_v1`
- Subset comparison: `data_perp/artifacts/strict_r3_routed_et_subset_ladder_crossfold_20260826_v1.parquet`
- Full screen producer: `scripts/run_strict_r3_routed_et_fulluniverse_screen.py`
- MDA producer: `scripts/run_strict_r3_routed_et_mda.py`
- Subset evaluator: `scripts/evaluate_strict_r3_routed_et_subset_ladder.py`

The next permitted research stage is a B0 replacement using the unchanged
frozen E/T controls and the predeclared TBM, policy-magnitude and path-quality
LambdaRank targets.  Its selection criterion is conditional contribution to
E+T, not standalone head quality.
