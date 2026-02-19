# TBM compare vs label-step audit

## Key findings

1. **Denominator mismatch is real**: `compare_tbm_parameters.py` computes `tp_hit/timeout/sl_hit` after candidate prefilter, optional quantile filter, and RR filter (`events = events[events["net_rr"] >= min_rr]`). In contrast, label-step geometry scoring in `training.py` computes rates on full matrix labels (`lbl.size`) before candidate filtering. This explains low uniform `tp_hit` values in label-step logs.
2. **`rr=0` in label logs is a reject counter, not RR metric**: the printed `rr` field is `reject_counts["rr"]` (count of geometries failing min RR), so zero means no RR rejections.
3. **Floor binding computation uses tolerance (good)**: TP-floor share is computed with `<= (tp_lo_eval + 1e-9)` (not strict float equality).
4. **Order of operations differs by path**:
   - Compare path scales TP by horizon first, then caps TP/SL.
   - Label path computes TP with horizon scaling in `compute_barrier_factory`, then applies floor/ceiling via `np.clip`.
   This order is internally consistent in each path, but not fully identical between compare and label pipelines.
5. **Production TP floor override is enabled by default**: label-step re-evaluates selected geometries under production TP floor when `label_use_production_tp_floor=True`.
6. **Cache keys appear complete for geometry-affecting params in compare path**: layer1 key includes serialized barrier params + horizon + side.

## Suggested follow-ups

- Make label-step geometry diagnostics compute TP/SL/timeout on the same candidate-filtered event universe used in final training set construction.
- Rename label diagnostic `rr` field to `rr_rejects` in logs to avoid interpretation issues.
- Add an explicit denominator field (`n_eval`) wherever rates are printed.
