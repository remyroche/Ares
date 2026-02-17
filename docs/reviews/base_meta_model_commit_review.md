# Review of `base-model-enhancements-v1-6828633638613437324` and `meta-model-docs-9239251187183048585`

## Scope reviewed
- Base-model branch content effectively lands via `ce28b6ea` (the follow-up commit `fb56021a` is metadata-only and introduces no code delta).
- Meta-model/docs branch content lands via `be49ae40` (documentation) and `dba20592` (meta-model/training refactor).

---

## 1) Review of the changes made

### Base-model enhancements (`ce28b6ea`)
- Added a new 3-way triple-barrier outcome kernel with quality scoring (`SL=0`, `TIMEOUT=1`, `TP=2`) and optional API return path (`return_outcomes=True`).
- Added regime-conditioning feature block (`cusum_strength`, `move_magnitude_z`, `cusum_decay`, `vol_percentile`, `vol_of_vol`, `atr_percentile`, `liquidity_ratio`) and wired it into feature generation.
- Updated training pipeline to consume outcome+quality labels, changed target framing to binary `TP vs rest`, and introduced outcome-quality-informed sample weights.
- Updated model race scoring path to rely on calibrated OOF probabilities (isotonic) before model-selection metrics.

### Meta-model/docs enhancements (`dba20592`, `be49ae40`)
- Meta regression target moved to risk-unit normalization using ATR-like volatility proxy and monotone squashing (`arcsinh`) instead of hard winsor clipping.
- Meta classifier moved from binary framing toward multiclass (`SL/TO/TP`) with CV selection by multiclass logloss.
- Candidate set includes a robust linear option (Huber) and robust XGBoost objective (`pseudohuber`) where available.
- Added architecture/design documentation for meta models.

Overall direction is strong: more realistic label semantics, stronger robustness bias, and better calibration awareness.

---

## 2) Optimization opportunities (compute + memory)

### Compute/vectorization
1. **Avoid duplicated rolling calculations in regime features**
   - `ret1h` rolling std is computed twice for closely related features; compute once and reuse.
2. **Reduce chained `np.where` passes in weighting**
   - In training weighting logic, 3 chained `np.where` writes can be replaced by one indexed assignment or `np.select` to reduce passes over large arrays.
3. **Use incremental aggregation instead of `np.stack` on full 3D tensors**
   - `labels_stack`/`returns_stack`/`qual_stack` materialization is memory-heavy; accumulate weighted sums in a streaming loop.
4. **Top-k selection micro-optimization**
   - `np.argpartition` is already used in places; ensure all top-k metrics paths avoid full `np.argsort` where ordering inside k is not needed.

### Memory/cache/downcast/gc
1. **Downcast transient arrays aggressively**
   - Many intermediates default to float64 (`np.full`, arithmetic outputs); use float32 where numerically safe to halve memory footprint.
2. **Cache reuse for volatility/ATR-derived thresholds**
   - Risk-unit thresholds and repeated vol_proxy transforms can be cached per horizon and re-used across candidate models.
3. **Limit OOF artifacts retained per model**
   - Storing both calibrated and raw OOF arrays per candidate can be expensive; persist only winner by default and gate detailed artifacts behind a debug flag.
4. **Explicit cleanup in large loops**
   - After each horizon/model race, release large temporaries and call `gc.collect()` in long-running offline jobs.

---

## 3) Logical flaws / correctness risks

1. **Timeout class mismatch in weighting path**
   - In training, comments/spec indicate outcomes are `2=TP,1=TIMEOUT,0=SL`, but timeout detection uses `is_timeout = (lbl_vals == 0)` in MFE/MAE weighting path. That treats SL as timeout.
2. **Potential division-by-zero in quality for SL events**
   - In the new triple-barrier outcome kernel, SL quality uses division by `activation` without guarding `activation <= 0` in SL branches.
3. **Calibration fit fragility**
   - Isotonic calibration on OOF can fail or overfit when OOF valid set has low diversity (e.g., nearly constant predictions / one-class slices). A guardrail/fallback is advisable.
4. **Multiclass `predict_proba` shape instability under class-missing folds**
   - Some classifiers can emit fewer columns if a class is absent in fold training data; assignment to fixed `(N,3)` OOF should enforce class-aligned columns explicitly.
5. **No-op commit in branch (`fb56021a`)**
   - The commit message implies implementation changes but introduces no file diff; this can reduce audit clarity.

---

## 4) Financial / ML logic improvements to consider (suggestions only)

1. **Regime-conditional barrier geometry**
   - Learn `k_tp/k_sl/horizon` by volatility regime and liquidity state, not globally.
2. **Utility-aware objective for selection**
   - Move model selection from generic scoring (logloss/Brier) toward cost-aware expected utility with turnover and slippage penalties.
3. **Class-conditional calibration**
   - For multiclass meta models, calibrate per class (Dirichlet/vector scaling) and monitor calibration drift by regime.
4. **Uncertainty-aware execution gating**
   - Use entropy / margin of multiclass probabilities to suppress low-confidence trades.
5. **Time-decayed and recency-aware weighting**
   - Combine outcome quality with recency decay to adapt faster to regime shifts.
6. **Conservative same-bar TP/SL ambiguity handling variants**
   - Evaluate pessimistic vs midpoint fill assumptions and quantify downstream sensitivity.
7. **Nested purged CV for hyperparameter+model race**
   - Reduce selection bias from repeated re-use of OOF for both ranking and calibration diagnostics.
8. **Portfolio-level constraints in post-model stage**
   - Add correlation-aware sizing and drawdown-constrained allocation instead of per-trade independent ranking.

---

## Bottom line
- The two branches move the system in a robust and production-aligned direction (3-way outcomes, quality-aware weighting, risk-unit targets, multiclass meta classification).
- The highest-priority fixes are: timeout class mismatch, guard against zero/invalid activation in quality math, and robustify calibration/OOF class handling.
