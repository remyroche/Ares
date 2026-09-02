# Short P0/F90 M4 Domain and Era Diagnostic

Status: **offline research only — no short live/admission promotion**.

## Completed cross-era result

Immutable run:
`data_perp/artifacts/strict_r3_short_p0_m4_domain_regime_diagnostic_20260821_v2`.

The conclusion is **do not add a regime/support gate to frozen M4**.  Its
causal train-p80 control has valid-label policy-net results of −15.04 bps per
trade in 2024 (1,984 trades), +92.48 in 2025 (1,360), and +92.51 in 2026
(344).  The later-era result is therefore not a mere selection artifact, but
the adverse 2024 regime does not transfer under any of the predeclared
controls.

The support-history probe uses the unchanged M4 model and OOF isotonic map:

| Train history | 2024 net bps/trade | 2025 | 2026 | Reading |
|---|---:|---:|---:|---|
| 3 months | −8.38 | +97.47 | −18.61 | Small 2024 improvement, but damages 2026. |
| 6 months | −17.81 | +100.57 | −20.12 | Does not repair 2024 and damages 2026. |
| 9 months | −19.22 | +112.29 | +71.24 | Strong later eras, worse 2024. |
| 12 months | −14.19 | +110.47 | +138.64 | Best later-era portability, still negative 2024. |

All requested 25k/50k/100k/all caps are non-binding in this top-1 hourly
population and hence exactly equivalent. They are retained in the immutable
artifact with their effective-cap audit; they are not independent evidence.

The raw-to-policy-net relationship is weak in 2024 (raw Spearman 0.063;
mapped-EV Spearman 0.022), but positive in 2025 (0.123/0.104) and 2026
(0.112/0.102). The policy decomposition also shows the economic shift: the
same frozen p80 selection averages +84.96 gross/−15.04 net bps in 2024 versus
about +192.5 gross/+92.5 net bps in each later era. Thus the evidence is
consistent with an era-level outcome/policy mismatch, not simply inadequate
history or an isotonic-map defect.

D9 rejected every predeclared marker: no two had a supported, same-direction
effect in 2024, 2025, and 2026. D10 is consequently empty. This is a correct
fail-closed outcome—not a missing replay—and leaves the frozen M4 control
unchanged.

## Purpose

The frozen short P0/F90 M4 conversion head is a useful later-era absolute
conversion model, yet its causal train-p80 admission was negative in 2024 and
strong in 2025–26. This work distinguishes support maturity, calibration
transport, score-domain semantics, market conditioning, OOD/support, and
population composition before any new target or model is considered.

M4 itself is frozen: six-class ordinal policy margin on the existing 41-field
base block, trained expanding-prequentially and converted to policy net bps
with chronological OOF isotonic calibration. P0 still selects one target-free
winner per hour. The runner may never rerank P0, alter policy geometry, add a
consensus, or create an admission through a later stage.

## Runner

`scripts/run_strict_r3_short_p0_m4_domain_diagnostic.py`

Example full diagnostic command:

```bash
python3 scripts/run_strict_r3_short_p0_m4_domain_diagnostic.py \
  --absolute-root data_perp/artifacts/strict_r3_short_p0_absolute_conversion_funnel_2024_maydec_20260821_v1 \
  --absolute-root data_perp/artifacts/strict_r3_short_p0_absolute_conversion_funnel_2025_h1_20260821_v1 \
  --absolute-root data_perp/artifacts/strict_r3_short_p0_absolute_conversion_funnel_2026_janjul_20260821_v1 \
  --policy-label-root data_perp/artifacts/strict_r3_short_p0_f90_policy_labels_2025_20260821_v1 \
  --policy-label-root data_perp/artifacts/strict_r3_short_p0_f90_policy_labels_2026_jul20260821_v1 \
  --out data_perp/artifacts/strict_r3_short_p0_m4_domain_regime_diagnostic_20260821_v1
```

The default D1 matrix is the requested 3/6/9/12-month train history crossed
with 25k/50k/100k/all support caps. Since this is an hourly top-1 population,
the requested large caps can be mathematically equivalent; the runner records
the effective cap and reuses exactly the same fit only in that case.

## Funnel and evidence

| Stage | Output | Contract |
|---|---|---|
| D0 | frozen M4 OOF population | exact stored P0/M4 rows and causal train-p80 gate |
| D1 | support-matched re-fits | same M4 target, 41 fields, model, OOF calibration and p80 rule |
| D2 | raw-score/map surfaces | descriptive deciles plus calibration error; no same-period metric becomes a rule |
| D3 | policy gross/net/exits | net + 100 bps equals gross; joins existing exit reason/minute only when supplied |
| D4 | P0 score-domain | strict-prior 90-day score geometry percentiles |
| D5 | market-state | strict-prior target-free market marker percentiles |
| D6 | predeclared interactions | breadth × volatility, direction × dispersion, funding-dispersion × direction, and P0-domain interactions |
| D7 | support/OOD | strict-train robust shrinkage Mahalanobis, p1–p99 extremeness, nearest-neighbour support and leaf support |
| D8 | population composition | per-symbol, causal listing-age and diagnostic-only stable-core/depth-proxy views |
| D9 | hierarchical shrinkage | global → M4 bucket → P0 strength → state, only if at least two markers are same-direction across eras |
| D10 | matched replay | frozen M4 p80 AND D9 expected policy net ≥ 0; demotion-only |

## Causality and limitations

- Training rows satisfy both `decision_ts < held_month` and
  `policy_label_available_at < held_month`.
- M4 input scores are its original strict-prequential P0 scores; D1 never
  consumes held outcomes for fitting or score-to-bps calibration.
- Marker percentiles are calculated from only prior target-free observations.
- D7's covariance, nearest-neighbour radius, feature ranges, and leaf support
  are fitted from the held month's prior training rows only.
- Same-era deciles are clearly marked descriptive. They cannot become a live
  admission threshold.
- Raw Geometry/K9 memberships are prohibited. The current M4 source lacks a
  stable bundle-invariant K9 support/OOD aggregate, so it is reported as
  unavailable rather than substituted.
- Existing short policy-label parts retain terminal exact-policy outcomes and
  exits, but not MFE/MAE. The runner reports those as unavailable rather than
  inventing an excursion proxy. A future exact-minute path materialisation can
  extend D3 without changing D0–D2/D4–D10 semantics.
- D9 marker screening uses the diagnostic eras; even a favourable D10 result
  is research evidence only and must be tested on a later frozen period.

## Promotion criteria

No successor advances unless it first demonstrates, on a later frozen period:

- non-negative 2024-equivalent performance with broad monthly support;
- at least 70% of frozen M4 participation in 2025 and 2026;
- net EV of at least +75 bps/trade in each later-era block;
- no negative worst month, materially better CVaR, and no concentrated single
  month responsible for more than roughly one-third of profit; and
- proof that any gate only removes existing M4 p80 candidates.
