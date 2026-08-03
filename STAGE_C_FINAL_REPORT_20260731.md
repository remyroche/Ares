# Stage-C final report — 2026-07-31

## Verdict

`CURRENT_OHLCV_OI_FUNDING_CONTRACT_INSUFFICIENT_FOR_ENTRY_RETENTION`

This is an Outcome-A stop.  No feature group passed the **development-only** Stage-1 admission gate; therefore no compact Stage-2 arm, frozen Stage-B hierarchy substitution, base-score comparison, global top-10 ranking, or causal expected-net threshold test was run.  This report does not promote a target, threshold, quota, gate, sizing rule, or action policy.

Accepted inputs are the full Stage-0 panel [`stage_c_continuation_feature_panel_20260731_v2`](data_perp/artifacts/stage_c_continuation_feature_panel_20260731_v2) and the matched-seed Stage-1 run [`stage_c_conditional_retention_ablation_20260731_v3`](data_perp/artifacts/stage_c_conditional_retention_ablation_20260731_v3).  The current runner hash equals the v3 manifest hash: `155cfb60038df4e16eefb5a23de6700252e773ad357596626743648ae018e6ac`.

## 1–4. Mechanisms, reuse, causality, and coverage

| Group | Materialized mechanism | Status / reuse |
|---|---|---|
| F0 | Exact persisted v11 E15, 68 controls per side, SHA `a91c1b40…6085ed1` | Frozen inherited control; loaded, never recomputed. |
| F1 | 1/4/12h returns, acceleration, efficiency, slope/R², directional consistency, high/low distance and recency, direction changes, OHLCV candle/wick proxies, range expansion, symmetric failed breakouts/rejection, side return/slope/wick/rejection | 28 fields; short-horizon variants and named reuse/redundancy are recorded per field in `feature_source_lineage.parquet`. |
| F2 | Volume z-score/persistence, signed-volume **proxy**, range-to-volume **proxy**, churn, price-volume correlation, concentration, shock age/decay | 9 fields; no factual L2/aggressor claim. |
| F3 | RV/downside RV, vol ratio/vol-of-vol, range z, squared-return autocorrelation, ATR slope/acceleration, shock age/decay/climax, side adverse RV | 12 fields; side adverse RV is symbol-and-side local. |
| F4 | OI dynamics | Rejected: no native observed/available timestamp or bounded staleness; archived unbounded ffill is not admitted. |
| F5 | Funding/crowding | Rejected for the same native availability failure; no next payment/settlement feature. |
| F6 | Timestamp-eligible universe size, rank, breadth, dispersion, relative return, confirmation and isolated-move signals | 10 fields; membership digest persisted for each timestamp. |
| F7 | Regime/transition descriptors | Rejected: no verified candidate-level strict OOF/prequential sidecar. |
| F8 | Fixed efficiency×volume, breakout×breadth, volatility×efficiency, climax×low-efficiency, market-confirmation×volume composites | 5 transparent, unfitted products. |

The lineage map records source, formula, lookback, minimum observations, units/range, side normalization, decision-time availability, delay, missingness/staleness, live parity, factual/proxy status and reuse disposition.  Every admitted field joins the exact completed OHLCV bar at `feature_cutoff_ts`; all rolling transforms are trailing and all source fields satisfy `feature_available_ts <= decision_ts`.  Inverse PI is excluded; all materialized source symbols are linear USD perpetuals.

The immutable full compatible cohort is **252,702 / 272,686** frozen rows (SHA in the Stage-0 manifest), with **103,681** clear-first rows.  It covers 2023-04-01 through 2024-12-01 and 128 source symbols.  Exclusions are explicit by group/month/side/symbol/reason: F1 17,525 rows, F2 15,340, F3 17,799; F6/F8 have no additional exclusions.  Stage-1 uses the same complete-group cohort restricted to 2024-04..2024-11: 124,450 rows / 53,736 clear-first rows.

## 5–6. Strict OOF result, stability, calibration, and transport

All C0/C1/C2/C3/C6/C8 fits are side-local v11 LGBM models with fixed hyperparameters and arm-invariant seeds keyed only by split/fold/side.  Development OOF is expanding monthly (2024-04..07), with `decision_ts < fold_start - 12h` and `label_available_ts < fold_start`.  Incremental filters, clipping, correlation representatives and gain selection (cap 32) are train-fold-only.  Frozen selections are made from development evidence only; final OOS (2024-08..11) is diagnostic and never used for admission.

| Arm | Dev ROC / PR / Brier | Dev top-decile net | Final ROC / PR / Brier | Final ΔROC / ΔBrier vs C0 | Final top-decile net |
|---|---:|---:|---:|---:|---:|
| C0 | .5726 / .6520 / .2434 | 6.5 bps | .5560 / .6939 / .2426 | baseline | 58.5 bps |
| C1/F1 | .5741 / .6542 / .2427 | 11.1 | .5591 / .6969 / .2416 | +.0031 / −.0011 | 59.9 |
| C2/F2 | .5703 / .6503 / .2434 | 8.0 | .5568 / .6950 / .2416 | +.0008 / −.0010 | 59.0 |
| C3/F3 | .5721 / .6538 / .2434 | 11.4 | .5587 / .6965 / .2417 | +.0027 / −.0010 | 55.6 |
| C6/F6 | .5738 / .6500 / .2449 | −6.7 | .5487 / .6855 / .2451 | −.0074 / +.0024 | 45.4 |
| C8/F8 | .5709 / .6502 / .2434 | 13.9 | .5589 / .6966 / .2420 | +.0029 / −.0006 | 65.7 |

Development C0 calibration slope/intercept is .354/.278; final is .263/.463.  Final C1/C3/C8 slopes are .285/.454, .284/.454 and .281/.460.  Reliability and net for every prediction decile are in `retention_conditional_calibration.parquet`; rows/prevalence, long/short/month/fold metrics, symbol breadth, top-decile concentration, importance concentration and missingness sensitivity are in the results/stability artifacts.

The paired 200-replicate UTC-day bootstrap reinforces why final-looking deltas cannot promote a group: development ROC improvement probabilities are C1 .730, C2 .130, C3 .445, C6 .610, C8 .285; their 90% intervals cross zero except none is a stable admission result.  F1 has mean development ΔROC only +.0008 and side transport .833 (<1.0); F2/F3/F8 have negative mean development ΔROC; F6 worsens calibration and is materially worse final OOS.  Every group is `diagnostic_only`; F4/F5/F7 are `rejected`.

## 7–12. Compact, Stage B, economics, final disposition, and gap

7. No compact arm or leave-group-out ablation exists: `retention_compact_feature_manifest.json` records `STAGE2_NOT_RUN`, and LOGO is explicitly `NOT_RUN` rather than fabricated.

8. No retention head entered Stage B. `stage_b_incremental_retention_results.parquet` and its summary are `NOT_RUN`; frozen clear/adverse/net components were not altered.

9. Frozen base opportunity comparison is unavailable by design: the Stage-1 gate failed before Stage B.

10. Pooled-global top-10 and causal-threshold exact-net results are unavailable by design.  If Stage B were admissible, ranking would be one pooled global common-bps ranking after mapping—not per side, timestamp, asset, or quota—but it was not run.

11. The sole terminal disposition is Outcome A above.  Final-OOS C1/C3/C8 improvements are descriptive only and cannot override unstable development selection evidence.

12. The remaining information gap is decision-time information about whether a clear-first move will persist.  Current causal OHLCV/cross-sectional information does not provide transported, calibrated, symbol-broad retention evidence.  OI/funding need native observed/available timestamps; regime fields need a candidate-level strict OOF/prequential sidecar.  Historical L2/depth/aggressor/liquidation data remain unavailable and were not invented.

## Final requirement audit

| Ledger area | Verdict | Evidence / remaining state |
|---|---|---|
| A frozen identity and labels | PASS | Stage-0 manifest, compatible-ID ledger/hash, H0/H25/continuous diagnostics and H12 endpoint. |
| B Stage-0 F0–F8 | PASS | Full Stage-0 report, lineage, coverage, exclusions, dictionary and source ledger; F4/F5/F7 correctly rejected. |
| C strict Stage-1 | PASS | v3 predictions/results/calibration/stability/deltas/bootstrap, matched seed audit and final-OOS freeze. |
| F named Stage-0/Stage-1 tests | PASS | 30 focused tests; v3 `correctness_test_report.json` is evidence-driven and true. |
| G Stage-0/Stage-1 deliverables | PASS | All listed Stage-0 files plus all Stage-1 files are present; compact/LOGO/Stage-B files explicitly record NOT_RUN. |
| D Stage 2 / E Stage B / later F tests | Correctly not run | Outcome-A stop; no experimental result is missing or treated as a pass. |
| H final report | PASS | This artifact answers all twelve questions. |

The v3 manifest’s output hashes all verify; its inputs include feature-panel/group, E15, raw panel, alignment, post-cost, persistence and v11-result hashes, and its runner/v11/continuation source hashes verify against current source.  The Stage-0 manifest verifies 4 input hashes, 213 source-file hashes and 2 code hashes against current bytes, then seals and verifies all 14 non-manifest outputs in its `outputs` map.  `correctness_test_report.json` is included in that map; `run_manifest.json` is explicitly excluded to avoid a circular self-hash.  The only unchecked ledger items are conditional Stage-2/Stage-B tests and economics, which remain intentionally unexecuted under Outcome A.
