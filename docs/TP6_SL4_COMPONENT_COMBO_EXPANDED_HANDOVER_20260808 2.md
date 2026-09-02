# TP6/SL4 Component-Combination and Expanded-Universe Replay

## Purpose

This handover records two linked experiments:

1. An exact 16-arm ablation of four residual feature blocks on the canonical long-only substrate:
   - model support/OOD;
   - archetype signed exposure;
   - uncertainty;
   - compact structural.
2. A monthly walk-forward replay of the selected challengers on the eligible 170-symbol universe, with monthly side-local base retraining, causal archetype clustering, and strict prior-month OOF residual retraining.

The central question is whether residual/context information improves the current base ranking when the universe is expanded and models are retrained in the way inference will operate.

## Executive conclusion

The canonical 2025 long-only funnel selected:

1. model support/OOD;
2. uncertainty;
3. model support/OOD + uncertainty.

Those three configurations were positive in the canonical development substrate at top-5%, but none transported to positive net EV in the expanded 170-symbol replay. The expanded base-only control was better at the long-only top 1% and 2% tails.

The strongest expanded challenger was uncertainty + compact structural at top-5%, but it remained negative:

| Key comparison (long-only control vs pooled challengers) | Net EV |
|---|---:|
| Base-only control, top 1% long-only | **+18.49 bps** |
| Base-only control, top 2% long-only | **+6.02 bps** |
| Base-only control, top 5% long-only | −7.31 bps |
| Uncertainty + compact structural, global top 5% | −18.93 bps |
| Archetype signed exposure, global top 5% | −23.41 bps |

No expanded residual challenger is execution-ready.

## 1. Canonical 16-arm funnel

### Substrate and fixed contract

- Canonical TP6/SL4 Base+Consensus score.
- Long-only rows.
- 2025 monthly OOS evaluation across 12 months.
- 10,224 underlying candidate rows; 16 arms; 163,584 prediction rows.
- Train-only isotonic mapping from canonical score to expected net bps.
- Residual target:

  `exact_net_bps - train-only isotonic(canonical score)`

- Residual grades: `[-150, -50, +50, +150]` bps.
- Native LightGBM LambdaRank.
- Query grouping: 4-hour UTC × side.
- Final score: `0.75 × canonical Base+Consensus rank + 0.25 × residual rank`.
- Evaluation: global pooled top-k ranking.

### All 16 arms

Selection was by pooled global top-5 net bps/trade, then monthly mean, worst month, and positive-month count.

| Rank | Components | Global top-5 net | Mean monthly top-5 | Worst month | Positive months |
|---:|---|---:|---:|---:|---:|
| 1 | model support/OOD | +39.93 | +26.23 | −85.69 | 6/12 |
| 2 | uncertainty | +31.54 | +22.24 | −99.48 | 6/12 |
| 3 | support/OOD + uncertainty | +29.65 | +15.67 | −126.33 | 4/12 |
| 4 | archetype signed exposure + uncertainty | +28.07 | +28.34 | −83.15 | 5/12 |
| 5 | control | +25.88 | +27.06 | −116.46 | 7/12 |
| 6 | archetype signed exposure | +18.25 | +10.05 | −93.04 | 4/12 |
| 7 | support/OOD + archetype signed exposure | +14.36 | +6.11 | −133.24 | 5/12 |
| 8 | uncertainty + compact structural | +13.58 | +10.27 | −160.99 | 6/12 |
| 9 | support/OOD + compact structural | +12.89 | +18.27 | −166.73 | 6/12 |
| 10 | archetype signed exposure + compact structural | +10.95 | +12.00 | −167.78 | 6/12 |
| 11 | support/OOD + uncertainty + compact structural | +10.67 | +9.66 | −78.45 | 7/12 |
| 12 | support/OOD + archetype signed exposure + compact structural | +10.60 | +7.76 | −148.59 | 5/12 |
| 13 | compact structural | +5.14 | −3.37 | −171.28 | 6/12 |
| 14 | archetype signed exposure + uncertainty + compact structural | +0.42 | +5.76 | −132.99 | 5/12 |
| 15 | all four blocks | −2.62 | +4.49 | −141.17 | 6/12 |
| 16 | support/OOD + archetype signed exposure + uncertainty | −5.10 | +11.08 | −119.99 | 5/12 |

### Frozen top three

| Configuration | Top 1% net | Top 2% net | Top 5% net | Top 10% net | Rank IC |
|---|---:|---:|---:|---:|---:|
| Support/OOD | +71.06 | +88.87 | +39.93 | −12.05 | 0.067 |
| Uncertainty | +34.27 | +37.57 | +31.54 | −10.62 | 0.063 |
| Support/OOD + uncertainty | +84.83 | +83.46 | +29.65 | −9.27 | 0.064 |

These results are development evidence only. The expanded replay below is the transport test.

## 2. Expanded 170-symbol monthly replay

### Source and universe

- Universe: exactly 170 symbols from the eligible-symbol list.
- Source rows after filtering: 215,210.
- Source symbols observed: 170.
- Source period: 2026-02-01 through 2026-07-10.
- Scored period: March–July 2026.
- February is warm-up because there is no earlier labelled 170-symbol history.
- Scored underlying rows: 175,281; the two-side long/short split is retained.

### Contract caveat

This source is **not** the canonical TP6/SL4 label panel. It is the available signal-close execution-margin source:

- `net_bps = exec_margin × 10,000`;
- `first_touch_gross_bps = first_touch_gross × 10,000`;
- the source execution margin already includes its fee/spread adjustment;
- no second cost subtraction was applied.

Therefore the expanded replay answers:

> Do the residual feature blocks transport to the larger source universe under a strict monthly refit?

It does not answer whether the exact canonical TP6/SL4 contract transports until exact TP6/SL4 labels and raw causal features are materialized for the 170 symbols.

### Raw-feature limitation

The prediction parquet did not contain the raw handoff feature matrix. No raw causal fields from the intended 80-field input list were available in that source.

The replay therefore used an explicit 42-field materialized causal support/context fallback, including:

- base margin and rank context;
- rank-band and margin-band prior reliability;
- support counts and support frequency;
- causal 3-day, 7-day, and 14-day hit-rate surprise fields.

Outcome-derived policy/archetype labels and post-entry path labels were excluded from the base/clustering input contract.

The exact 42-field list is recorded in `run_manifest.json` under `base_features`.

### Monthly training chronology

For each side independently:

1. Fit the base model using rows whose label availability satisfies `valid_end <= month_start`.
2. Fit a K=6 causal MiniBatchKMeans archetype representation on those prior rows.
3. Transform the held month using that month’s frozen cluster state.
4. Produce held-month base predictions.
5. Fit the residual ranker only when prior-month base OOF predictions exist.
6. Score the current month globally.
7. Store the current held-month base predictions as OOF history for the next month.

The explicit boundary rule is important: labels resolving exactly at the next month boundary are available before that month’s first decision.

Training status:

- Base models: 10 side-month refits, March–July × long/short.
- Archetype clustering: 10 side-month K=6 refits.
- Residual rankers: 8 side-month refits, April–July × long/short.
- March residual status: base-only because no prior OOF month existed.

### Base model

Side-local LightGBM multiclass classifier:

- classes: adverse `< −50 bps`, weak `−50..+50 bps`, clear `> +50 bps`;
- 140 trees;
- learning rate 0.05;
- 31 leaves;
- minimum child samples 350;
- subsample 0.8;
- feature fraction 0.8;
- L2 = 8;
- 42 fallback causal support/context inputs.

Base score:

`P(clear) − P(adverse)`

### Archetype representation

The clustering was recomputed monthly, separately by side:

- median/MAD causal normalization fitted on prior rows;
- K=6 MiniBatchKMeans;
- batch size 2,048;
- 80 maximum iterations;
- three initializations;
- soft memberships from cluster distances;
- signed exposure = membership × normalized projection on the cluster centroid;
- signed exposure clipped to `[-3, +3]`.

The exact centroid hashes and monthly support are in `cluster_audit.parquet`.

### Residual ranker

The residual ranker consumes prior-month OOF base predictions only.

- Objective: native LambdaRank.
- Query grouping: 4-hour UTC × side.
- 120 trees.
- Learning rate 0.04.
- Maximum depth 4.
- 12 leaves.
- Minimum child samples 350.
- Feature fraction 0.80.
- Bagging fraction 0.80.
- L1 = 1.
- L2 = 10.
- Max bin 63.
- Gains: `[0, 0.25, 1, 3, 7]`.
- Target: exact source net bps minus prior-OOF base-score-to-net map, ordinalized at `[-150, -50, +50, +150]` bps.

## 3. Expanded replay results

### Global long/short ranking

| Configuration | Top 1% net | Top 2% net | Top 5% net | Top 10% net | Rank IC |
|---|---:|---:|---:|---:|---:|
| Support/OOD | −17.49 | −20.02 | −26.68 | −25.57 | 0.046 |
| Uncertainty | −26.10 | −28.66 | −24.72 | −23.29 | 0.052 |
| Support/OOD + uncertainty | −51.13 | −43.62 | −30.83 | −22.47 | 0.047 |

### Monthly top-5 net bps/trade

| Configuration | Mar | Apr | May | Jun | Jul |
|---|---:|---:|---:|---:|---:|
| Support/OOD | −44.45 | +5.68 | −8.83 | −46.75 | −30.76 |
| Uncertainty | −44.45 | −15.13 | −18.23 | −30.71 | −68.71 |
| Support/OOD + uncertainty | −44.45 | +25.35 | −34.58 | −47.22 | −42.54 |

March is the shared base-only warm-up month.

### Per-side top-5 net bps/trade

| Configuration | Long | Short |
|---|---:|---:|
| Support/OOD | −14.00 | −42.17 |
| Uncertainty | −14.69 | −37.08 |
| Support/OOD + uncertainty | −17.71 | −46.22 |

## 4. Structural challenger replay

After the first expanded replay, two structural challengers were run with the same monthly refit protocol:

- uncertainty + compact structural;
- archetype signed exposure.

The monthly archetype clusters were actually fed into these residual rankers.

### Global results

| Configuration | Top 1% net | Top 2% net | Top 5% net | Top 10% net | Rank IC |
|---|---:|---:|---:|---:|---:|
| Base-only control, long-only | +18.49 | +6.02 | −7.31 | — | — |
| Archetype signed exposure | −48.74 | −30.60 | −23.41 | −16.80 | 0.048 |
| Uncertainty + compact structural | −56.70 | −40.66 | −18.93 | −15.03 | 0.050 |

The control figures above are long-only, while the challenger global figures are pooled long/short. They should not be compared as identical populations; the per-side comparison is the fairer comparison.

### Per-side challenger top-5

| Configuration | Long | Short |
|---|---:|---:|
| Archetype signed exposure | −20.95 | −20.68 |
| Uncertainty + compact structural | −16.07 | −23.22 |

### Monthly challenger top-5 net bps/trade

| Configuration | Mar | Apr | May | Jun | Jul |
|---|---:|---:|---:|---:|---:|
| Archetype signed exposure | −44.45 | −16.43 | −17.63 | −37.67 | −33.14 |
| Uncertainty + compact structural | −44.45 | −1.83 | −25.58 | −11.54 | −59.02 |

## 5. Interpretation for quant review

### What appears robust

- The canonical residual layer can improve the smaller 2025 long-only substrate.
- Support/OOD and uncertainty are the most useful individual residual blocks there.
- Monthly K=6 clustering can be refit causally and consistently per side.
- Strict prior-month OOF residual training is implementable and auditable.
- Structural inputs can improve the expanded top-5 relative to uncertainty alone, although they remain negative after costs.

### What does not transport

- No residual arm is positive on the expanded global signal-close contract.
- The base-only control beats the residual arms in the expanded long-only top 1–2% tails.
- Positive rank IC around 0.046–0.052 does not translate into positive net EV; the score-to-economics conversion remains weak.
- The best canonical arm is not the best expanded arm.
- The combined support/OOD + uncertainty arm is consistently worse than its components at the sharpest tails.

### Most likely explanations

1. **Contract mismatch:** canonical TP6/SL4 labels and expanded signal-close execution margins are not the same target.
2. **Feature-contract mismatch:** expanded replay used materialized support/context fields rather than the intended raw causal feature spine.
3. **Universe shift:** the 170-symbol population contains substantially different symbol, liquidity, and cross-sectional distributions than the 74-symbol canonical panel.
4. **Residual overfitting:** the residual learner can exploit development-period structure that does not transport.
5. **Cost-floor compression:** a rank IC of 0.05 is not enough if the selected gross edge remains below the expanded source’s effective cost floor.

## 6. Artifacts

### Canonical 16-arm funnel

- [Report](</Users/remyroche/Documents/Ares/data_perp/artifacts/tp6_sl4_component_combo_funnel_long_20260808_v1/TP6_SL4_COMPONENT_COMBO_FUNNEL_REPORT.md>)
- [Selection ranking](</Users/remyroche/Documents/Ares/data_perp/artifacts/tp6_sl4_component_combo_funnel_long_20260808_v1/selection_ranking.parquet>)
- [Top three](</Users/remyroche/Documents/Ares/data_perp/artifacts/tp6_sl4_component_combo_funnel_long_20260808_v1/top3_configs.parquet>)
- [Run manifest](</Users/remyroche/Documents/Ares/data_perp/artifacts/tp6_sl4_component_combo_funnel_long_20260808_v1/run_manifest.json>)

### Expanded top-three replay

- [Report](</Users/remyroche/Documents/Ares/data_perp/artifacts/tp6_sl4_component_combo_expanded_monthly_20260808_v1/TP6_SL4_COMPONENT_COMBO_EXPANDED_MONTHLY_REPORT.md>)
- [Predictions](</Users/remyroche/Documents/Ares/data_perp/artifacts/tp6_sl4_component_combo_expanded_monthly_20260808_v1/predictions.parquet>)
- [Global metrics](</Users/remyroche/Documents/Ares/data_perp/artifacts/tp6_sl4_component_combo_expanded_monthly_20260808_v1/metrics.parquet>)
- [Monthly metrics](</Users/remyroche/Documents/Ares/data_perp/artifacts/tp6_sl4_component_combo_expanded_monthly_20260808_v1/monthly_metrics.parquet>)
- [Per-side metrics](</Users/remyroche/Documents/Ares/data_perp/artifacts/tp6_sl4_component_combo_expanded_monthly_20260808_v1/per_side_metrics.parquet>)
- [Monthly model audit](</Users/remyroche/Documents/Ares/data_perp/artifacts/tp6_sl4_component_combo_expanded_monthly_20260808_v1/month_audit.parquet>)
- [Monthly cluster audit](</Users/remyroche/Documents/Ares/data_perp/artifacts/tp6_sl4_component_combo_expanded_monthly_20260808_v1/cluster_audit.parquet>)
- [Manifest](</Users/remyroche/Documents/Ares/data_perp/artifacts/tp6_sl4_component_combo_expanded_monthly_20260808_v1/run_manifest.json>)

### Structural challenger replay

- [Report](</Users/remyroche/Documents/Ares/data_perp/artifacts/tp6_sl4_component_combo_expanded_monthly_structural_challenge_20260808_v1/TP6_SL4_COMPONENT_COMBO_EXPANDED_MONTHLY_REPORT.md>)
- [Predictions](</Users/remyroche/Documents/Ares/data_perp/artifacts/tp6_sl4_component_combo_expanded_monthly_structural_challenge_20260808_v1/predictions.parquet>)
- [Metrics](</Users/remyroche/Documents/Ares/data_perp/artifacts/tp6_sl4_component_combo_expanded_monthly_structural_challenge_20260808_v1/metrics.parquet>)
- [Monthly audit](</Users/remyroche/Documents/Ares/data_perp/artifacts/tp6_sl4_component_combo_expanded_monthly_structural_challenge_20260808_v1/month_audit.parquet>)
- [Cluster audit](</Users/remyroche/Documents/Ares/data_perp/artifacts/tp6_sl4_component_combo_expanded_monthly_structural_challenge_20260808_v1/cluster_audit.parquet>)
- [Manifest](</Users/remyroche/Documents/Ares/data_perp/artifacts/tp6_sl4_component_combo_expanded_monthly_structural_challenge_20260808_v1/run_manifest.json>)

## 7. Reproduction scripts

- [Canonical 16-arm runner](/Users/remyroche/Documents/Ares/scripts/run_tp6_sl4_component_combo_funnel.py)
- [Expanded monthly runner](/Users/remyroche/Documents/Ares/scripts/run_tp6_sl4_component_combo_expanded_monthly.py)
- [Evidence receipt](/Users/remyroche/Documents/Ares/agents/receipts/20260808_component_combo_expanded_monthly.json)

The expanded runner accepts a frozen selector parquet containing the requested arm names. It does not perform expanded-universe HPO; all arms are frozen before the replay.

## 8. Recommended next work

1. Materialize exact TP6/SL4/H12 labels for the eligible 170 symbols using the same entry and cost convention as the canonical panel.
2. Materialize the raw causal feature spine for those same rows; do not use the 42-field fallback for the final comparison.
3. Repeat the monthly replay with the exact canonical target and identical exits.
4. Compare base-only, residual-only, and base+residual using identical rows and global top-k selection.
5. Quantify score-to-net calibration and cost-to-ATR by side, month, and symbol cohort.
6. Test whether residual features should be used as shrinkage/reliability modifiers rather than a free-standing ranker.
7. Keep structural clustering only if it improves worst-month and per-side performance without sacrificing top-1% behavior.

## 9. Questions for the quant analyst

- Is the observed positive rank IC but negative net EV explained by a cost-floor or payoff-mixture effect in the signal-close source?
- Does the residual target remain economically meaningful when the base score is already a strong ranker?
- Should residual ranks be shrinkage-weighted by prior OOF support, cluster stability, or calibration uncertainty?
- Are the materialized reliability fields sufficiently point-in-time causal for a base model, or should they remain meta-only?
- Does the 170-symbol universe require side-specific or liquidity-specific calibration before global ranking?
- Is a monthly K=6 representation stable enough, or should cluster IDs be aligned across months before using signed exposure?
- What minimum top-k gross margin over costs should be required before accepting a residual arm?
