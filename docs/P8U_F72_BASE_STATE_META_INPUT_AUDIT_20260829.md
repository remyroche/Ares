# P8U F72 Base-state Meta-input Audit — 2026-08-29

## Decision

Do **not** promote a Base-state Meta extension.  The mixed M1 contract adds a
small amount of portfolio EV versus the exact F120 control, but gives up
worst-week performance and materially worsens maximum drawdown.  M4 and M5
are weaker downstream.  The canonical P8U F72 / Under-F120 / dual-MC1 stack
is unchanged.

This is an offline long-only research audit.  It makes no live, admission,
portfolio-production, or exchange change.

## Input contract

Every candidate preserves the immutable 120-field Under-F120 parent
contract.  The new columns were materialised target-free, by exact candidate
identity, from the strict F72 OOF ledger:

- **M2 — query geometry (13):** score gaps, IQR/MAD, local density, boundary
  density, entropy, and candidate count at the decision timestamp.
- **M3 — local calibration/support (24):** 3/7/21-day, Base-rank-band
  residual median, absolute-residual median/P90, large-residual rates, and
  EV; local values shrink through parent and global priors.
- **M4 — tree uncertainty (5):** block dispersion/MAD, positive-block
  fraction, early-vs-late shift, and concentration.
- **M5 — leaf/path state (10):** support mean/min/harmonic mean, low-support
  fraction, rarity, prequential residual prior and dispersion, prior support,
  and hashed signatures.

Calibration events activate only once `policy_label_available_ts` is strictly
before the decision.  F72 reproduces the frozen base score exactly in all 12
months (`max_abs_delta = 0.0`), leaf residual priors use train-only
prequential residuals, and no held policy/path column is part of an inference
matrix.

## Selection

The selector used only October–December 2025 outcomes, conditioning CMI and
IC on Base-rank bands for underconfidence, overconfidence, residual magnitude,
and +100-bps opportunity.  A lower-than-SHAP-core bar retained a feature when
coverage was at least 90%, it was active in at least two of three folds, and
direction consistency was at least 60%.

- **M0:** F120 control (120 fields).
- **M1:** top 20 stable mixed fields (140 total).
- **M2:** all query-geometry fields (133 total).
- **M3:** all local-calibration fields (144 total).
- **M4:** all tree-uncertainty fields (125 total).
- **M5:** all leaf/path fields (130 total).
- **M6:** predeclared M1 extension with the next four selector-qualified
  fields (144 total): `meta_cal_7d_residual_median_shrunk`,
  `meta_cal_21d_gt50_rate_shrunk`, `meta_q_gap_to_median`, and
  `meta_cal_21d_band_n`.

M6 is pre-2026 selected; it is not a union chosen after seeing the 2026
screens.

## Strict OOF Meta screen — January–July 2026

All arms use the frozen Under-F120 `rank_xendcg` trial and persist each held
score target-free before opening held outcomes.  Higher `SStableMeta`, IC,
CMI, and admission-substitution utility are better.

| Arm | Added fields | SStableMeta | Residual IC | CMI given Base | Admission utility (bps) |
|---|---:|---:|---:|---:|---:|
| M0 control | 0 | -0.09056 | 0.11638 | 0.17876 | +5.10 |
| M1 mixed stable | 20 | -0.07345 | **0.11945** | 0.18057 | +5.72 |
| M2 query geometry | 13 | -0.08577 | 0.11776 | 0.17764 | +5.24 |
| M3 local calibration | 24 | -0.10150 | 0.11646 | 0.18023 | +5.42 |
| M4 tree uncertainty | 5 | -0.06586 | 0.11722 | 0.18029 | +5.53 |
| M5 leaf/path | 10 | **-0.05526** | 0.11944 | 0.17906 | **+5.59** |
| M6 predeclared extension | 24 | -0.08831 | 0.11922 | **0.18117** | +5.54 |

M1, M4, and M5 were the only arms passed to the full downstream test.  M2,
M3, and M6 did not offer a coherent enough OOF improvement to justify a
second selection layer.

## Matched downstream test — April–July 2026

Each arm uses the same F72 Base receipts, independent strict-prequential
Current and BCF MC1 maps, both MC1 maps at the +50-bps gate, one shared
chronological portfolio, and the same normal auction constraints.  April is
the first evaluable month after the three-month MC1 warm-up.

| Arm | Accepted trades | Net EV / trade | Total net bps | Worst month | Worst week | Max drawdown |
|---|---:|---:|---:|---:|---:|---:|
| M0 control | 3,634 | +119.21 | +433,201 | +68.28 | **+50.44** | **-21.86%** |
| M1 mixed stable | 3,633 | **+119.55** | **+434,331** | **+69.68** | +46.66 | -32.75% |
| M4 tree uncertainty | 3,629 | +115.92 | +420,684 | +62.49 | +40.67 | -32.75% |
| M5 leaf/path | 3,628 | +118.56 | +430,131 | +65.30 | +42.61 | -28.30% |

M1's delta to M0 is +0.34 bps/trade and +1,130 total bps, but -3.78 bps on
the worst week and -10.89 percentage points of maximum drawdown.  That is not
an acceptable stability trade-off for a canonical Meta contract.

## SHAP lower-standard audit

The F72 SHAP audit is now closed and remains diagnostic-only:

- **Strict core (12/12):** signed SHAP balance, positive SHAP total, and
  `mark_perp_dislocation` contribution.
- **Recurrent:** entropy (12/12 aligned; 10/12 positive IC),
  `mark_trigger_risk_5h` contribution (9/12), `leverage_build` contribution
  (10/12), and `prog_eff_24` contribution (9/12).
- **Conditional only:** absolute total, top-1/top-3 shares,
  `mark_gap_vol_10h`, `mark_gap_vol_5h`, and `dist_prior_day_low`.

The only standalone OOF feature test that was directionally useful was SHAP
entropy: about +2.74 / +2.58 / +2.56 / +0.94 bps at Top-1/2/5/10, with
positive monthly deltas in 5/7, 5/7, 4/7, and 6/7 months.  It is too modest
for promotion.  Leverage, mark-trigger, and progress-efficiency variants were
mixed or tip-only and were rejected.

## Artifacts

- Target-free overlay and causality audit:
  `data_perp/artifacts/strict_r3_p8u_meta_base_state_aug25_jul26_20260829_v1`
- Pre-2026 selector and frozen M0–M5 contracts:
  `data_perp/artifacts/strict_r3_p8u_meta_base_state_selection_pre2026_20260829_v1`
- Pre-2026 M6 contract:
  `data_perp/artifacts/strict_r3_p8u_meta_base_state_selection_pre2026_m6_20260829_v1`
- Corrected v2 M0–M6 screens:
  `data_perp/artifacts/strict_r3_p8u_meta_base_state_m*_objective_janjul26_20260829_v2`
- Corrected v2 downstream comparisons:
  `data_perp/artifacts/strict_r3_p8u_meta_base_state_m{0,1,4,5}_downstream_dual_mc1_janjul26_20260829_v2`
- SHAP lower-standard diagnostic:
  `data_perp/artifacts/strict_r3_p8u_f72_shap_conditional_candidates_20260829_v4/F72_SHAP_CONDITIONAL_CANDIDATE_AUDIT.md`
