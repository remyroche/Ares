# TP6/SL4 canonical residual meta: granular long-only feature ablation

## Contract

- Evaluation: 2025-01 through 2025-12, long side only, 10,224 held rows.
- Training: all available earlier long rows in each chronological fold.
- Base: frozen TP6/SL4 Base+Consensus 75/25 score.
- Meta target: `exact_net_bps - train_only_isotonic(CanonicalScore)`.
- Meta grades: `[-150, -50, +50, +150]` bps.
- Queries: 4-hour UTC × side (`long` in this run).
- Final ranking: `0.75 * canonical score + 0.25 * residual rank`, then pooled global top-k.
- No GAM correction or realised outcome is an input.

## Model-layer features

Every non-control arm includes `canonical_expected_net_bps` and
`base_plus_consensus25` as anchors.  The frozen candidate blocks were:

### Uncertainty (15 candidates; 11 selected)

`consensus_head_rank_std`, `consensus_head_rank_mad`,
`consensus_head_rank_iqr`, `consensus_head_rank_min`,
`consensus_head_rank_max`, `consensus_head_raw_std`,
`consensus_head_agreement_fraction`, `base_consensus_disagreement`,
`base_score`, `r3_meta_p_clear`, `r3_meta_p_adverse`, `r3_meta_p_weak`,
`base_probability_entropy`, `base_probability_top2_margin`,
`base_conviction`.

Selected fields: `consensus_head_raw_std`, `base_consensus_disagreement`,
`consensus_head_rank_max`, `consensus_head_rank_min`, `base_conviction`,
`consensus_head_rank_mad`, `r3_meta_p_adverse`, `r3_meta_p_weak`,
`consensus_head_rank_std`, `consensus_head_rank_iqr`,
`base_probability_top2_margin`.

### Support/OOD (8 candidates; 5 selected)

`context_missing_fraction`, `context_ood_mean_abs_z`,
`context_ood_p95_abs_z`, `context_ood_outlier_fraction`,
`context_ood_tail_fraction`, `support_recent_distance`, `support_min_margin`,
`support_low_tail_fraction`.

Selected fields: `context_ood_mean_abs_z`, `context_ood_p95_abs_z`,
`support_low_tail_fraction`, `context_ood_outlier_fraction`,
`context_ood_tail_fraction`.

### Drift (8 candidates; 1 selected)

`recent_context_shift`, `recent_context_covariance_break`,
`recent_score_shift`, `recent_head_dispersion_shift`, `score_history_ood`,
`recent_volatility_shift`, `recent_breadth_shift`, `recent_liquidity_shift`.

Selected field: `score_history_ood`.

### Market state (21 candidates; 12 selected)

`median_rvol_z`, `median_volume_z`, `mkt_atr_expansion_1h`,
`mkt_atr_expansion_4h`, `q_iqr__bars_in_high_vol_state_log_norm`,
`breadth_dispersion`, `cs_dispersion_ret_24h`, `cs_dispersion_ret_4h`,
`avg_pair_corr_24h`, `corr_concentration_24h`,
`correlation_breakdown_dispersion`, `median_spread_bps`,
`ob_depth_l10_to_qv_24h`, `amihud_z_peer_resid`,
`liquidity_ratio_peer_resid`, `xs_dispersion__amihud_z`, `fund_abs_z`,
`fund_abs_z_14d`, `fund_abs_z_mkt_resid`, `mkt_abs_ret_per_oi_drop_1h`,
`oiw_intensity_entry_dist_7d_atr`.

Selected fields: `oiw_intensity_entry_dist_7d_atr`,
`ob_depth_l10_to_qv_24h`, `liquidity_ratio_peer_resid`,
`fund_abs_z_mkt_resid`, `mkt_abs_ret_per_oi_drop_1h`, `amihud_z_peer_resid`,
`median_spread_bps`, `fund_abs_z`, `breadth_dispersion`,
`xs_dispersion__amihud_z`, `avg_pair_corr_24h`, `cs_dispersion_ret_4h`.

## Archetype and cluster-layer features

The structural source is the frozen recurrent long-side contract
`tp6_sl4_archetype_cluster_vcgam_oof_20260814_v3`.  It covers 100% of the
2025 long rows.  The layer is exposure-only: no GAM score, residual, gross,
net, or future outcome is included.

### Transport and compact structure

`archetype_matched_mass`, `archetype_unmatched_mass`, `archetype_entropy`,
`archetype_top2_margin`, `archetype_abs_total`, `archetype_signed_total`,
`archetype_abs_max`, `archetype_signed_max`, `archetype_active_count`,
`archetype_active_mass`, `archetype_abs_entropy`,
`archetype_abs_top2_margin`, `structural__cluster__active_count`,
`structural__cluster__abs_entropy`, `structural__cluster__abs_top2_margin`,
`structural__cluster__abs_total`, `structural__cluster__abs_max`,
`structural__cluster__signed_total`.

The transport-only arm uses the four transport fields plus the two archetype
and two cluster entropy/margin fields.  The compact arm uses all 18 fields
above.

### Archetype exposures

For each `0000` through `0010`, the full archetype arm exposes:

- `structural__archetype_<id>__abs_contribution`;
- `structural__archetype_<id>__signed_contribution`;
- `structural__archetype_<id>__active`.

### Cluster exposures

For each `cofire_cluster_00` through `cofire_cluster_05`, the full cluster arm
exposes:

- `structural__cluster__<id>__abs_exposure`;
- `structural__cluster__<id>__signed_exposure`;
- `structural__cluster__<id>__active_mass`.

The frozen cluster membership is: 00 = archetypes 0003/0005/0008/0009;
01 = 0001/0006; 02 = 0000/0004; 03 = 0002; 04 = 0010; 05 = 0007.

## Granular ablation results

Net bps/trade, pooled global ranking:

| Arm | Top 0.5% | Top 1% | Top 5% | Top 10% | Mean monthly Top-5 | Worst month |
|---|---:|---:|---:|---:|---:|---:|
| Canonical control | +53.87 | −1.93 | **+25.88** | +3.75 | +27.06 | −116.46 |
| Model support/OOD | +71.06 | **+100.48** | **+39.93** | −12.05 | +26.23 | −85.69 |
| Model uncertainty | +34.27 | +15.37 | +31.54 | −10.62 | +22.24 | −99.48 |
| Model market/liquidity subset | −25.02 | +27.78 | +20.52 | −8.86 | **+29.71** | **−86.32** |
| Archetype absolute exposure | +74.40 | +32.95 | +25.29 | +1.38 | +18.12 | −99.78 |
| Archetype signed exposure | **+96.15** | +58.13 | +19.16 | **+4.25** | +7.28 | −134.38 |
| Cluster exposure | +45.72 | −36.64 | +9.16 | −27.66 | −6.34 | −117.21 |
| Model uncertainty + structural compact | **+137.86** | +68.55 | +13.58 | −3.87 | +10.27 | −160.99 |
| Model all + structural compact | +122.66 | +67.47 | +14.92 | −11.99 | +4.60 | −131.95 |
| Model all + structural full | +38.65 | +53.20 | +12.72 | −4.84 | +9.93 | −94.65 |

The finer channel split shows that structural signed archetype exposure is
useful at the narrowest tails, while cluster absolute/active exposure is
harmful.  The best global top-5 arm is still model support/OOD; no combined
model+structural arm beats it or the control at top-5.

## Decision

The feature-integration requirement is satisfied: both model-layer trust/state
features and frozen archetype/cluster exposure features are available under an
explicit contract and have been evaluated separately and jointly.  The
structural fields do not yet justify promotion.  Keep them as research inputs;
the next validation must use a later untouched long-side chronology before
selecting a production arm.

Artifacts: `data_perp/artifacts/tp6_sl4_canonical_residual_meta_granular_long_20260808_v3/`.
