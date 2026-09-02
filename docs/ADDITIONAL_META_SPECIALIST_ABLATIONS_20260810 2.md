# Additional frozen meta/specialist ablations (2026-08-10)

## Contract and evaluation

All runs use the frozen cross-fold specialist contract:

- seven side-specific specialist views;
- 68 causal fields per view;
- identical ordered membership across the three transport folds;
- ATR-spacing 2.0 specialist target for the new challenger arms;
- 4-hour bucket × side specialist queries unless explicitly varied below;
- pooled global ranking over July–November 2024 transport observations;
- top 1%, 5%, and 10% tails;
- 100 bps cost applied once in the resolved net labels.

The residual control is the prior frozen ATR2 stack: ordinal per-row net residual,
native LambdaRank, q4h×side, and the existing context architecture. Its reference
result is top-1 −7.30, top-5 +8.89, top-10 −37.63 bps/trade, with monthly top-5
values −51.55, −171.07, −58.25, −81.07, +11.00 for July through November.

## 1. Residual query groups longer than four hours

| Query | Top-1 net | Top-5 net | Top-10 net | Month mean | Month std | Worst month |
|---|---:|---:|---:|---:|---:|---:|
| q4h × side | −7.30 | **+8.89** | −37.63 | −70.19 | 58.95 | −171.07 |
| q8h × side | −4.02 | +8.23 | −42.35 | −69.20 | 61.04 | −172.26 |
| q12h × side | **+0.88** | +7.39 | −41.43 | −70.48 | 68.53 | −188.50 |
| q24h × side | −0.06 | +5.18 | −50.97 | −84.49 | 87.75 | −243.01 |

Side top-5 net for q4h/q8h/q12h/q24h respectively:

- Long: +17.25 / +17.78 / +20.30 / +24.26 bps.
- Short: −195.71 / −198.90 / −202.34 / −199.97 bps.

The four-hour grouping remains selected by the declared global top-5 rule. Longer
groups do not solve the short-side conversion failure and q24h materially worsens
stability.

Artifact: `data_perp/artifacts/frozen_longer_meta_query_ablation_20260810_v1/`.

## 2. Larger-feature regime-grouped specialists

The specialist feature contract was expanded to 160 causal fields. Query groups
were based on volatility, trend, transition intensity, entropy, or a composite
regime key rather than timestamp buckets. Standalone specialist scores and their
downstream residual impact were both evaluated.

| Query | Level | Top-1 net | Top-5 net | Top-10 net | Month mean | Month std | Worst month |
|---|---|---:|---:|---:|---:|---:|---:|
| volatility | standalone | +26.22 | −82.39 | −108.65 | −78.98 | 62.10 | −128.56 |
| volatility | residual | −9.50 | **+9.19** | −46.43 | −87.17 | 78.38 | −233.42 |
| trend | standalone | +30.83 | −64.84 | −108.66 | −78.25 | 71.06 | −132.76 |
| trend | residual | −9.99 | +7.45 | −46.99 | −86.85 | 78.00 | −231.79 |
| transition | standalone | +69.56 | −60.93 | −104.13 | −61.72 | 81.05 | −174.51 |
| transition | residual | −13.98 | +6.08 | −47.89 | −88.72 | 81.36 | −242.04 |
| entropy | standalone | +69.56 | −60.93 | −104.13 | −61.72 | 81.05 | −174.51 |
| entropy | residual | −13.98 | +6.08 | −47.89 | −88.72 | 81.36 | −242.04 |
| composite | standalone | +61.40 | −38.37 | −68.26 | −47.64 | 61.36 | −95.19 |
| composite | residual | −11.19 | +8.21 | −45.81 | −85.25 | 76.90 | −228.65 |

The volatility residual arm is only +0.31 bps above the q4h control and has a
much worse worst month. Composite standalone ranking is less poor in the broad
tail, but its residual impact is below control. None advances.

For the best residual volatility arm, side top-5 net is long +15.72 / short
−195.66 bps. For the composite residual arm it is long +17.57 / short −196.02.

Artifact: `data_perp/artifacts/regime_grouped_larger_specialists_20260810_v1/`.

## 3. Incremental binned-CMI meta feature addition

Candidates were restricted to fields present in the configured meta families
(`PERP_META_PRIMARY_FEATURE_KEYS`, `RESIDUAL_META_FEATURE_KEYS`,
`MARKET_CROSS_SECTIONAL_META_FEATURE_KEYS`, and `T2_FUNNEL_META_CONTEXT_FEATURE_KEYS`).
The binned conditional-MI proxy was computed on the selection half only. Features
were added greedily one at a time; the residual learner used max depth 4.

| Step | Added feature | Top-1 net | Top-5 net | Top-10 net | Month mean | Month std | Worst month |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | `rv_24h_peer_resid` | −6.50 | +8.65 | −43.00 | −79.88 | 66.95 | −198.47 |
| 2 | `mkt_oi_chg_z_24h` | −7.05 | +8.59 | −43.71 | −78.67 | 65.25 | −192.09 |
| 3 | `mkt_pct_oi_drawdown_24h_lt_minus5pct` | −8.43 | +8.15 | −43.93 | −78.96 | 63.70 | −188.73 |
| 4 | `pct_assets_oi_down_24h` | −4.80 | +8.85 | −42.33 | −78.37 | 64.57 | −190.59 |
| 5 | `mkt_median_oi_drawdown_from_peak_24h` | −9.76 | **+9.31** | −43.17 | −78.25 | 64.53 | −189.41 |
| 6 | `mkt_oi_breadth_rising_24h` | −9.30 | +7.30 | −44.63 | −82.24 | 68.26 | −201.44 |
| 7 | `mkt_funding_mean_z_30d` | −8.50 | +7.71 | −46.10 | −83.10 | 65.17 | −194.75 |
| 8 | `mkt_oi_chg_24h` | −8.21 | +7.44 | −44.03 | −78.13 | 63.02 | −188.10 |

The top-5 improvement peaks at step 5 and then falls. It is not a stable broad
improvement: step-5 side top-5 is long +17.90 / short −197.81 bps, and its
monthly top-5 values are −71.08, −189.41, −61.01, −81.51, +11.79.

Artifact: `data_perp/artifacts/meta_incremental_cmi_20260810_v1/`.

## 4. Non-residual meta target and EV-mapped asymmetric combinations

The meta learner was trained on a non-residual ordinalized exact H12 net target
using economic net-margin bins. Base and meta predictions were mapped separately
to side-local expected net bps on the later calibration half, then combined with
independent weights in `{0, .25, .5, .75, 1, 1.5}`.

The best grid point was base weight **0.75**, meta weight **0.25**:

| Combination | Top-1 net | Top-5 net | Top-10 net | Month mean | Month std | Worst month |
|---|---:|---:|---:|---:|---:|---:|
| Base 0.75 + meta 0.25 | −61.70 | **−47.65** | −60.96 | −85.11 | 49.16 | −182.79 |

Monthly top-5: July −65.74, August −62.83, September −64.16, October −50.02,
November −182.79. Side top-5 is long −96.63 / short −56.09 bps.

This is materially worse than the residual architecture; EV mapping cannot repair
the non-residual target’s ranking in this matched replay.

Artifact: `data_perp/artifacts/nonresidual_ev_combination_20260810_v1/`.

## 5. Specialist recent-error features

For each specialist head, the meta learner received prior-only rolling features:

- hit rate;
- hit-rate surprise (`actual − score percentile`);
- rolling score/outcome IC;
- lookbacks of 3, 7, and 14 days.

Test labels were hidden; test histories were seeded only from calibration rows.
The residual model used max depth 4.

| Arm | Top-1 net | Top-5 net | Top-10 net | Month mean | Month std | Worst month |
|---|---:|---:|---:|---:|---:|---:|
| No error history | −3.64 | +9.59 | −39.57 | −74.21 | 65.16 | −189.62 |
| 3-day history | −7.02 | +8.17 | −40.34 | −73.60 | 61.35 | −179.83 |
| **7-day history** | **−3.37** | **+9.71** | **−37.92** | −70.95 | 58.75 | **−172.39** |
| 14-day history | −4.92 | +8.61 | −39.91 | −73.25 | 64.35 | −187.64 |
| All 3/7/14-day fields | −4.72 | +6.84 | −41.96 | −72.38 | 62.39 | −180.22 |

The 7-day arm is the strongest new challenger: +0.83 bps top-5 versus the
matched depth-4 no-history control and a 17.23 bps improvement in worst-month
net EV versus that control. Its side top-5 values are long +18.47 / short
−195.60 bps. Thus it improves temporal trust calibration, but not the short-side
economics.

Artifact: `data_perp/artifacts/specialist_error_history_ablation_20260810_v1/`.

## Decision

1. Keep q4h×side residual queries; longer query buckets do not improve the
   primary global/stability trade-off.
2. Do not promote regime-grouped larger specialists: their residual uplift is
   negligible and their worst periods are worse.
3. The incremental CMI plateau is around five additions; the OI/peer-volatility
   fields are plausible trust context, but the gain is small and short-driven
   failure remains.
4. Reject the non-residual EV-mapped target/combination family for this contract.
5. Keep the 7-day specialist error-history features as the only new challenger
   worth carrying forward, subject to a fresh untouched-period test and explicit
   short-side admission.

No arm passes execution readiness because every arm has materially negative
short-side top-5 net EV and a negative worst month.

The requested combined regime + CMI + 7-day error-history arm and the bad-month
cause analysis are documented in
`docs/COMBINED_REGIME_CMI_ERROR_DIAGNOSIS_20260810.md`.
