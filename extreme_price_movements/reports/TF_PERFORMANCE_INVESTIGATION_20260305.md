# TF Performance Investigation (Long/Short)

## Findings

1. **Trend-filter leakage on neutral bars (`trend_pct == 0`) was letting neutral events into the down-trend branch.**
   - Multiple training paths were using `<= 0` for the down filter, which includes neutral events and can dilute TF signal quality.
   - This affects both dataset generation and precomputed event selection, and is especially damaging for `short_tf` because the down branch carries more samples.

2. **Non-finite trend values were coerced to 0.0 before filtering.**
   - Previous logic used `np.nan_to_num(..., nan=0.0)` before sign checks; this silently converts unknown trend to neutral/down and introduces noisy labels/features.

3. **The reported metrics are consistent with low directional purity in TF buckets.**
   - TF models show weak IC and near-flat regime scores in the 20260214 run, despite acceptable AUC (~0.52-0.53), indicating ranking power is being diluted.

## Root Cause

Directional filtering logic was inconsistent and permissive:
- `up` branch required `> 0`, but `down` branch used `<= 0`.
- `NaN` trends were forced to 0 and retained in `down`.

This asymmetry biases TF training data composition and weakens specialist trend-following signal separation.

## Fix Applied

Implemented a strict shared helper for trend-direction masks:
- `up`: finite and `> 0`
- `down`: finite and `< 0`
- neutral (`0`) and non-finite values are excluded from both directions.

Applied this helper across training-set construction paths to keep labeling/training/event selection semantics consistent.

## Expected Impact

- Cleaner directional cohorts for TF models.
- Improved IC/lift stability for `short_tf` and `long_tf` by removing neutral/unknown trend contamination.
- More honest regime-conditioned diagnostics (less cross-regime bleed).

## Next Validation Step

Re-run `train_daily_base` for the same artifact window and compare:
- TF bucket IC, AUC, lift@k, and per-regime BSS/AUC vs previous run.
- Specifically monitor `short_tf` H2/H4 for IC uplift and reduction in calibration overconfidence.
