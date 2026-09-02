# P8U Archetype Recovery — Per-head HPO, Feature Budget, and Activation-Floor Result

## Decision

`DO NOT PROMOTE — C0 RETAINS THE PRIMARY ROUTER-RECALL ADVANTAGE.`

The per-head funnel improved selected-trade mean policy net in two of three held quarters and by +6.33 bps over the three-quarter selected population.  It did **not** improve the predeclared primary metric: recall of valid `policy_net_bps > +50` candidates at the fixed 10% Router-rescue budget.  No frozen-2026 data were opened.

This is offline, long-side research only.  It modifies neither the canonical stack nor live inference.

## Frozen evaluation contract

- Outer held quarters: 2025 Q2, Q3, and Q4.
- The structural representation remained target-free, per fold: Q2 NMF K5; Q3/Q4 NMF K4.
- Specialist target candidates remained ATR-normalised policy utility or the six-bin ordinal version.
- The Router-recall/economic evaluation remains realised policy net bps.
- The comparison control is `C0_multiseed_full_universe`, evaluated only after head choices were frozen.  C0 did not enter head feature, model, HPO, membership-floor, or blend selection.
- Primary operating point: 10% selected budget with 50% Router allocation / 50% probe rescue.

## Sequential per-head funnel

For each NMF category, using pre-inner rows only:

1. Screen P0/P1, canonical target/model forms, and a category-specific weighted binned-MI feature budget.
2. Retain the best two forms; test the four fixed model configurations and membership activation floors `0.00`, `0.10`, `0.20`, `0.35`.
3. Refit the selected heads on all pre-held resolved rows.
4. Choose the probe combination on the inner window only, then compare it once with C0 on the held quarter.

The model bank was: baseline; compact regularised; shallow stable; and moderate capacity.  The exact override values are in [the hash-bound overlay](../config/strict_r3_p8u_archetype_recovery_20260830_v2_headhpo.json).

## Held-quarter result

| Held quarter | Arm | Valid selected | Mean net bps | CVaR10 net bps | Positive economic-mass recall | Recall >50 bps | Recall >100 | Recall >200 | Within-timestamp top-10% recall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2025 Q2 | Per-head HPO | 17,257 | +86.53 | -578.76 | 20.38% | 13.57% | 15.57% | 22.19% | 27.14% |
| 2025 Q2 | C0 | 17,257 | +96.03 | -568.36 | 21.04% | **13.69%** | **15.95%** | **23.35%** | **28.98%** |
| 2025 Q3 | Per-head HPO | 29,204 | +26.34 | -573.94 | 24.93% | 17.76% | 22.59% | **33.19%** | **33.33%** |
| 2025 Q3 | C0 | 29,204 | +19.67 | -590.28 | 24.76% | **18.11%** | **22.68%** | 32.01% | 32.93% |
| 2025 Q4 | Per-head HPO | 34,908 | +35.98 | -590.22 | 25.18% | 17.42% | 22.06% | **32.13%** | **32.48%** |
| 2025 Q4 | C0 | 34,914 | +22.11 | -600.00 | 24.84% | **18.22%** | **22.33%** | 30.52% | 31.83% |

## Pooled three-quarter evidence

| Metric | Per-head HPO | C0 | Difference |
|---|---:|---:|---:|
| Valid selected rows | 81,369 | 81,375 | -6 |
| Mean selected net | **+43.24 bps** | +36.91 bps | **+6.33 bps** |
| Total selected net contribution | **+3,518,536 bps** | +3,003,731 bps | **+514,805 bps** |
| Recall >50 bps | 16.46% | **16.93%** | **-0.47 pp** |
| Recall >100 bps | 20.20% | **20.45%** | **-0.25 pp** |
| Recall >200 bps | **29.02%** | 28.49% | +0.53 pp |
| Positive economic-mass recall | 23.645% | **23.659%** | -0.014 pp |

The primary +50-bps recall declines in every held quarter: Q2 -0.12 pp, Q3 -0.35 pp, and Q4 -0.80 pp.  Higher mean selected net is therefore insufficient to advance this as a Router-recall recovery architecture.

## What the head funnel selected

- All 13 selected heads used P1; none retained the smaller P0 input set.
- Feature budgets: five 32-field heads, five 56-field heads, and three 80-field heads.
- Model/target winners: seven CatBoost Huber utility heads, four CatBoost ordinal heads, and two LightGBM ordinal heads.  No LightGBM Huber head won.
- HPO winners: five shallow-stable, four baseline, two compact-regularised, and two moderate-capacity configurations.
- Activation floors: twelve heads retained `0.00`; only Q3 category 0 selected `0.20`.  Thus thresholding membership was not a general source of improvement.
- Inner-selected combinations: Q2 `gamma=1.0 / rank×membership / logsumexp`; Q3 `gamma=0.5 / rank×membership / logsumexp`; Q4 `gamma=1.0 / rank×membership / top2_mean`; all use the fixed 50/50 Router-rescue allocation.

Exact selected field lists, HPO overrides, activation floors, and inner utilities are recorded in the three `inner_selection_summary.parquet` files.

## Q4 recovery provenance

The original full-prehistory label parquet had become a zero-block sparse file after Q2/Q3 completed.  It was not overwritten.  Q4 used the immutable Aug-27 successor ledger, whose pre-append population is exactly the expected 2,564,827 rows / 2,525,498 valid rows; its August append cannot join the P8U candidate panel, which ends in July 2026.  The run is bound to this source and its SHA-256 in [the Q4 recovery overlay](../config/strict_r3_p8u_archetype_recovery_20260830_v3_headhpo_q4recovery.json).

The runner was also made memory-safe by reading only the sealed target-free panel's decision-time range from the append-only label ledger.  This is an I/O/memory bound, not an availability, label, or feature-selection change.

## Immutable outputs

- Q2: `data_perp/artifacts/strict_r3_p8u_archetype_recovery_headhpo_q2_20260830_v1`
- Q3: `data_perp/artifacts/strict_r3_p8u_archetype_recovery_headhpo_q3_20260830_v1`
- Q4: `data_perp/artifacts/strict_r3_p8u_archetype_recovery_headhpo_q4recovery_20260830_v5_memoryprofile`

Each contains `head_trial_summary.parquet`, `inner_selection_summary.parquet`, `combination_selection_summary.parquet`, `matched_budget_recall.parquet`, `controls.parquet`, `correctness_report.json`, and a run manifest.
