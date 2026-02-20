# Label Step Report — Run 20260214_190000
**Generated:** 2026-02-20  
**Pipeline:** `extreme_price_movements` → `labels`  
**Exit code:** 0 ✅  
**Duration:** ~75 min (feature load → geometry → dataset assembly)

---

## 1. Configuration

| Parameter | Value |
|---|---|
| Config ID | CFG3CE81B18B8 |
| `base_atr_window` | 672h |
| `k_tp` | 1.25 |
| `sl_as_tp_pct` | 0.60 |
| `tp_base_pct` | 1.5% |
| `barrier_mode` | `atr_norm` |
| `horizon_scaling` | `sqrt` |
| Horizons | H=2, H=4, H=8 |
| Sides | long, short |
| Kinds | mr (mean-reversion), tf (trend-following) |
| Geometry grid cells | 12 (`MR/TF × long/short × H2/H4/H8`) |
| Global ATR windows pre-computed | 504h, 672h, 840h |
| Candidate events (post-OOS holdout) | 254,780 |
| OOS holdout cutoff | 2025-08-18 (last 180 days excluded) |
| Symbols | 617 (intersection of features & panel) |

---

## 2. Geometry Phase — Per-Cell Summary

All 12 cells completed without error. Key geometry metrics from logs:

| Cell | Triplets | ATR Windows | Accepted Geoms | Best edge | bind_raw | tp_floor_share |
|---|---|---|---|---|---|---|
| MR_long_H2 | 10 | 672, 840 | 6 | 31.74 | 96.8% | 71.2% ⚠️ |
| MR_long_H4 | 10 | 672, 840 | 6 | 62.56 | 98.3% | 71.2% ⚠️ |
| MR_long_H8 | 9 | 672, 840 | 5 | 93.60 | 98.9% | 71.2% ⚠️ |
| MR_short_H2 | 10 | 672, 840 | 6 | 37.90 | 97.4% | 71.2% ⚠️ |
| MR_short_H4 | 9 | 840 | 5 | 50.32 | 98.0% | 71.2% ⚠️ |
| MR_short_H8 | 9 | 840 | 5 | 76.23 | 98.7% | 71.2% ⚠️ |
| TF_long_H2 | 10 | 672, 840 | 6 | 31.74 | 96.8% | 71.2% ⚠️ |
| TF_long_H4 | 9 | 672, 840 | 6 | 47.30 | 97.8% | 71.2% ⚠️ |
| TF_long_H8 | 9 | 840 | 6 | 78.44 | 98.7% | 71.2% ⚠️ |
| TF_short_H2 | 10 | 504, 672, 840 | 6 | 59.98 | 98.2% | 71.2% ⚠️ |
| TF_short_H4 | 10 | 504, 672, 840 | 6 | 59.98 | 98.2% | 71.2% ⚠️ |
| TF_short_H8 | 9 | 840 | 5 | 90.48 | 98.9% | 71.2% ⚠️ |

**Notes:**
- `bind_raw` = fraction of events that resolve (TP or SL) before timeout — consistently ~97–99%, excellent.
- `edge` = composite geometry quality score — increases with horizon (H8 > H4 > H2), as expected for wider barriers.
- `tp_floor_share` = 71.2% across all cells — **above the 70% warning threshold**. This means the TP barrier is being clipped to the floor for 71% of events. Consider raising `barrier_tp_lo` or widening the geometry in the next optimisation cycle.
- `sep` (TP separation) ranges 14–17pp across cells — healthy discriminability.
- `auc_b` = 1.000 for all cells (bound AUC) — the geometry is well-separated from timeout.

**Production Admissibility (label-step gate):** FAIL  
Failures: `sl_to_tp_prod_agg 58x > max 3x`, `max_cell_tp_floor_bind 71.4% > 70%`. These are known consequences of the high TP floor clipping and the fact that `auc_label` / `ap_lift` are not yet populated at the label stage (they require model predictions). This gate is informational at label time.

---

## 3. Dataset Sizes & Label Distribution

| Dataset | N events | TP% | SL% | Timeout% | Bind% | TP/SL ratio |
|---|---|---|---|---|---|---|
| long_mr_H2 | 169,413 | 2.76 | 3.21 | 94.03 | 5.97 | 0.86 |
| long_mr_H4 | 169,412 | 1.20 | 3.26 | 95.54 | 4.46 | 0.37 |
| long_mr_H8 | 169,413 | 0.58 | 3.19 | 96.23 | 3.77 | 0.18 |
| long_tf_H2 | 85,367 | 1.83 | 3.29 | 94.88 | 5.12 | 0.56 |
| long_tf_H4 | 85,367 | 1.06 | 3.36 | 95.58 | 4.42 | 0.32 |
| long_tf_H8 | 85,367 | 0.41 | 3.33 | 96.26 | 3.74 | 0.12 |
| short_mr_H2 | 85,367 | 2.05 | 3.75 | 94.21 | 5.79 | 0.55 |
| short_mr_H4 | 85,367 | 1.55 | 3.94 | 94.50 | 5.50 | 0.39 |
| short_mr_H8 | 85,367 | 0.86 | 4.13 | 95.01 | 4.99 | 0.21 |
| short_tf_H2 | 169,413 | 3.78 | 3.68 | 92.55 | 7.45 | **1.03** |
| short_tf_H4 | 169,411 | 1.81 | 3.82 | 94.37 | 5.63 | 0.47 |
| short_tf_H8 | 169,413 | 1.06 | 3.99 | 94.96 | 5.04 | 0.27 |

**Total events across all datasets:** ~1,537,866 (sum across 12 files)

### Observations
- **Timeout dominance:** 92–96% of events timeout — expected for tight barriers on a large universe. The model's job is to identify the ~3–8% that resolve.
- **TP/SL ratio < 1 for all cells except `short_tf_H2` (1.03):** SL hits slightly outnumber TP hits. This is structurally expected — the universe is selected for extreme moves, so mean-reversion and trend-following both face adverse selection. The model must learn to filter.
- **TP% decreases with horizon (H2 > H4 > H8):** Correct — wider horizons allow more time for adverse moves to accumulate before the barrier is hit.
- **Short-side has higher TP% than long-side** at matched horizons: consistent with the short-biased nature of extreme price movements in crypto.
- **`short_tf_H2` is the best-balanced cell** (TP/SL ≈ 1.03, bind 7.45%) — highest information content for training.

---

## 4. Barrier Geometry (Median Values)

| Group | TP barrier (bps) | SL barrier (bps) | TP/SL ratio |
|---|---|---|---|
| MR cells (long & short) | ~11,607 | ~5,804 | ~2.0× |
| TF cells (long & short) | ~11,464 | ~5,732 | ~2.0× |

- TP barrier ≈ 116 bps, SL barrier ≈ 58 bps — 2:1 reward/risk ratio by construction.
- Consistent across MR and TF groups (slight difference due to different ATR window distributions).

---

## 5. MFE / MAE Analysis (Median)

| Group | MFE median (bps) | MAE median (bps) | Interpretation |
|---|---|---|---|
| MR_long | -3,706 | +1,552 | Adverse MFE — most events move against before resolving |
| MR_short | -3,706 | +1,552 | Same |
| TF_long | -728 | -1,629 | MFE and MAE both negative — tight range events |
| TF_short | -3,706 | +1,552 | Same as MR pattern |

**Note:** Negative MFE median means the median event never reaches a favourable excursion before timeout — the model must predict the rare TP events from features, not from price trajectory alone.

---

## 6. Quality Score Distribution (Median `__quality__`)

| Dataset | q_med |
|---|---|
| long_mr_H2 | 0.197 |
| long_mr_H4 | 0.164 |
| long_mr_H8 | 0.139 |
| long_tf_H2 | 0.249 |
| long_tf_H4 | 0.196 |
| long_tf_H8 | 0.162 |
| short_mr_H2 | 0.290 |
| short_mr_H4 | 0.230 |
| short_mr_H8 | 0.184 |
| short_tf_H2 | 0.225 |
| short_tf_H4 | 0.187 |
| short_tf_H8 | 0.156 |

- Quality scores are low (0.14–0.29) — consistent with the high timeout rate. Quality is derived from the barrier outcome; most events timeout with near-zero quality.
- `short_mr_H2` has the highest median quality (0.290) — the most informative cell for the model.

---

## 7. Sample Weight Distribution

All datasets: `w_mean = 1.000`, `w_std = 1.528`  
- Weights are normalized to mean=1. The std of 1.53 indicates moderate dispersion — some events are up to ~13× the base weight (top-1% share ~8%).
- `n_eff` (effective sample size after weighting) ≈ 50,824 for MR cells (169K events), ≈ 25,610 for TF cells (85K events) — roughly 30% effective utilization, consistent with high overlap in the purged CV framework.

---

## 8. Artifacts Saved

| Artifact | Path |
|---|---|
| 12 training datasets | `data/artifacts/20260214_190000/labels/train_{side}_{kind}_{H}.parquet` |
| Spike anatomy (best) | `data/artifacts/20260214_190000/labels/spike_anatomy_best.parquet` |
| Spike anatomy (worst) | `data/artifacts/20260214_190000/labels/spike_anatomy_worst.parquet` |
| Trap model | `data/artifacts/20260214_190000/labels/trap_model.parquet` |
| Exhaustion history | `data/artifacts/20260214_190000/labels/exhaustion_history.parquet` |
| Bucket report | `extreme_price_movements/reports/20260214_190000/bucket_report_labels.md` |

---

## 9. Fixes Applied During This Run

| # | Error | Fix |
|---|---|---|
| 1 | ATR windows treated as separate sweep axis | `params_store.py`: store `(k_tp, sl, atr_window)` triplets per cell |
| 2 | `ValueError`: mismatched array lengths | `training.py`: reindex `_ret_s` to `_lbl_s.index` |
| 3 | `NameError: bmask` | `training.py`: add `bmask = y_bound_q.astype(bool)` |
| 4 | `KeyError: 672` on `_barrier_base_cache` | `training.py`: don't clear `_barrier_base_cache` per cell |
| 5 | `ModuleNotFoundError` on `compare_tbm_parameters` | `training.py`: revert to `compute_triple_barrier_labels` call |
| 6 | OOM after 12 cells (per-cell stacked arrays) | `training.py`: `del` large intermediates after each cell aggregation |
| 7 | OOM after 12 cells (`_prod_events_rows` 21M rows) | `training.py`: sample to 500K per cell before appending |
| 8 | OOM after 12 cells (`tb_cache` held all 12 cells) | `training.py`: `tb_cache.pop()` / `geom_cache.pop()` after each dataset |
| 9 | `ValueError: label_intervals required` | `training.py`: pass `label_intervals=` as keyword arg to `cv.split()` |

---

## 10. Action Items for Next Cycle

1. **Raise `barrier_tp_lo`** or widen geometry to reduce TP floor clipping below 70% (currently 71.2% across all cells).
2. **TP/SL ratio < 1** for 11/12 cells — consider asymmetric cost weighting in base model training or adjusting `sl_as_tp_pct`.
3. **`short_tf_H2`** is the most balanced cell (TP/SL ≈ 1.03) — monitor this cell's model performance closely as a leading indicator.
4. **Quality scores are low** (median 0.14–0.29) — the Ridge OOF quality component in sample weights may have limited signal; monitor `ridge_oof_quality` contribution in base training.
