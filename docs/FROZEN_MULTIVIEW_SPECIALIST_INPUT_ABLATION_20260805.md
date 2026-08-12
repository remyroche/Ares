# Frozen cross-fold specialist retrain and residual-input ablation

## Contract repair

The previous `data_view_*` identifiers were fold/side-local. This run discovers
one template from rows strictly before the earliest transport test period
(2024-07-01), then freezes the exact ordered feature membership for all three
transport folds. Specialist models are still refit inside each fold, with the
same binary target: exact H12 net outcome greater than +50 bps.

The frozen contract is side-local (`long` and `short` have separate templates)
but fold-invariant:

- 7 specialists per side;
- 68 exact fields per specialist;
- 952 specialist field slots in total;
- every `(fold, side, specialist)` has one identical feature hash;
- specialist predictions are placed on the common expected-net-bps scale before
  global ranking.

Evidence: `frozen_view_contract.json` and `frozen_view_stability.parquet`.

## Funding/OI/leverage availability

The store has 780 columns. The configured specialist context registry contains
386 fields; 233 are present in the store and 227 pass the provenance/source
allowlist. Of the stored eligible fields, 203 match funding/OI/leverage,
liquidation, basis or crowding families. The remaining configured fields are
not currently materialised in this store and are not silently imputed as
specialist inputs.

The provenance path was repaired in `packb_static_point_feature_loader.py` and
`pipeline_steps.py`: these fields remain excluded from compact production
base/meta portability contracts, but are explicitly admissible to the causal
specialist sidecar.

## Residual-input arms

All arms use the same stable specialist outputs and the same ordinalised,
per-row net-residual LambdaRank target.

1. `specialist_heads_only`: the seven stable specialist heads.
2. `specialists_plus_ae_gmm`: specialists plus the ten AE/GMM fields actually
   present in the store.
3. `specialists_plus_selected_context`: specialists plus one frozen field from
   each requested family:
   `ema20_slope_5h`, `mkt_volume_z_24h`, `funding_abs_z`,
   `mkt_oi_chg_z_24h`, `atr_percentile`,
   `distance_to_resistance_daily_vwap_atr`.
4. `current_all_inputs`: specialists plus base outputs, prequential expected
   net, train-only synergy fields, regime/trust fields, and the six selected
   context fields.

## True pooled global OOS results

These are ranked across all candidates in all three transport folds after the
common-bps mapping, not averages of side-local rankings.

| System | Top 1% net | Top 5% net | Top 10% net | Top 1% gross | Top 5% gross | Top 10% gross |
|---|---:|---:|---:|---:|---:|---:|
| Legacy 7 fold-local views | −52.37 | −73.70 | −92.18 | +47.63 | +26.30 | +7.82 |
| Stable specialists only | −44.35 | −86.27 | −108.08 | +55.65 | +13.73 | −8.08 |
| Stable + AE/GMM | **+0.23** | **−70.98** | **−103.26** | **+100.23** | **+29.02** | **−3.26** |
| Stable + six selected fields | −25.78 | −82.47 | −111.77 | +74.22 | +17.53 | −11.77 |
| Stable current all inputs | −15.14 | −74.08 | −111.05 | +84.86 | +25.92 | −11.05 |

Relative to the legacy local-view system, the best stable arm (AE/GMM)
improves top-1% by 52.59 bps and top-5% by 2.72 bps, but worsens top-10% by
11.08 bps. No arm clears costs at top-5% or top-10%.

## Month stability

Global pooled net bps/trade:

| Month | Legacy top-5 / top-10 | Stable + AE/GMM top-5 / top-10 | Stable current-all top-5 / top-10 |
|---|---:|---:|---:|
| 2024-07 | −64.01 / −75.69 | −104.57 / −136.52 | −83.46 / −118.19 |
| 2024-08 | −159.80 / −180.38 | −151.81 / −148.15 | −140.62 / −159.26 |
| 2024-09 | −56.55 / −61.09 | −72.00 / −87.43 | −72.16 / −103.66 |
| 2024-10 | −85.33 / −84.35 | −84.78 / −95.68 | −67.98 / −81.25 |
| 2024-11 | +6.29 / +15.83 | **+62.29 / −50.15** | −13.03 / −71.43 |

The AE/GMM top-5 result is driven by a strong November observation; it is not
portable enough to promote. Across months, stable AE/GMM top-5 has mean −70.31,
standard deviation 71.57, and worst month −151.81 bps.

## Side results

| System | Long top-1 / top-5 / top-10 | Short top-1 / top-5 / top-10 |
|---|---:|---:|
| Legacy local views | −33.21 / −66.20 / −75.57 | −71.28 / −112.76 / −79.72 |
| Stable specialists only | −33.46 / −65.09 / −84.10 | −131.15 / −136.06 / −151.44 |
| Stable + AE/GMM | **+5.73 / −31.84 / −67.92** | −164.09 / −137.55 / −158.23 |
| Stable + selected context | −23.06 / −45.97 / −77.70 | −183.41 / −140.83 / −157.47 |
| Stable current-all | −4.88 / −41.03 / −77.29 | −111.75 / −142.34 / −151.05 |

The stable contract does not repair the short-side conversion problem. AE/GMM
helps long top-1 but materially hurts short top-1, so it cannot be promoted as
a pooled residual input without side-specific admission or calibration.

## Specialist family audit

View IDs now have stable meaning across folds, but remain data-discovered rather
than semantic labels. The main family counts are available in
`frozen_view_family_audit.parquet`. For example, long `data_view_00` contains
22 OI/leverage, 17 volatility, 14 volume/liquidity and 9 funding fields; long
`data_view_05` contains 19 funding, 12 OI/leverage and 19 volume/liquidity
fields. The same exact fields are reused in every transport fold.

## Decision

The contract repair is accepted. The performance result is not a promotion:

- stable field semantics remove a real methodological defect;
- funding/OI/leverage context is now genuinely available to specialists;
- AE/GMM is incremental only for the extreme top-1% tail;
- no residual-input arm produces positive pooled top-5% or top-10% net EV;
- month and side dispersion remain large, especially on shorts.

Next work should focus on side-local conversion/admission and a stable short-side
mapping, rather than adding more specialist heads or more context fields to the
current residual learner.
