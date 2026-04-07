# Layer B Policy Optimization Investigation Notes

## 10.1 Policy Scoring Note
**Objective:** Evaluate if `stable_absolute` materially improves ranking interpretability and stability vs `batch-zscore`, and if empty-fold handling meaningfully changes winner selection.

**Findings:**
1. **Interpretability & Stability:** The legacy `batch-zscore` approach standardized fold metrics (PnL, Sortino, MaxDD, Instability) relative to the specific subset of candidates in that optimization step. This caused extreme instability when candidates in the batch were very similar (small variance meant tiny differences were blown out of proportion) and made utility values totally incomparable across steps. Switching to `stable_absolute` gives each policy a deterministically calculable utility based on fixed weights. We can now compare a Step 1 candidate directly with a Step 3 candidate.
2. **Empty Fold Handling:** Previously, an empty fold resulted in arbitrary, catastrophic penalties (`maxDD = 1.0`, `timeout_rate = 1.0`), effectively "killing" candidates with tight score boundaries simply due to a small data sample. Moving to neutral zero-filling (`pnl_day = 0`, `maxDD = 0`) correctly represents no trading activity as "zero downside, zero upside" rather than a 100% loss. This safely unblocks tighter `score_quantile_fraction` settings that naturally lead to sparse validation folds.

---

## 10.2 Staged-Search Note
**Objective:** Determine if staged pruning appears to discard good joint candidates and if threshold/geometry dependence justifies later flattening.

**Findings:**
1. **Pruning Danger:** Step 1 evaluates geometries purely at `score_quantile_fraction = 0.60`. Our new instrumentation (evaluating Step 1 across `0.50`, `0.60`, and `0.80`) provides evidence that the top 2 geometries at `0.60` are not always the top 2 at `0.80`. An aggressive momentum breakout policy might be the best performer at high threshold confidences (`0.80`) but lose heavily at `0.50`, causing it to be pruned prematurely in the legacy flow.
2. **Recommendation:** Staged pruning assumes orthogonality between probability (the threshold) and path shape (the exit geometry). Because these are mathematically and economically coupled (higher predictive edge usually correlates with faster realization and less tolerance for deep drawdowns), discarding geometry candidates before exploring threshold combinations drops globally optimal solutions. **Future work should replace this staged search with a flattened search** (e.g., random, Halton, or full grid) across the joint `(geometry, threshold, time)` space.

---

## 10.3 Static-Policy Adequacy Note
**Objective:** Assess whether returning a single static policy per bucket is still acceptable or if light conditionality is needed.

**Findings:**
1. **Adequacy:** The current bucket-wide static policy applies the exact same stop loss width, take profit width, and hold duration to every trade in that bucket passing the score threshold.
2. **Limitations:** Trades at the 99th percentile of score/edge likely warrant a tighter trailing stop to lock in profits, while trades at the 70th percentile might need a wider stop to breathe. Similarly, predictions with high uncertainty (from Model 3) should intuitively be given faster time-exits to cut risk.
3. **Recommendation:** While the static search is a solid baseline (especially now that it's evaluated properly via `stable_absolute` scoring), **later work should prioritize lightly conditional policies**. Even allowing a linear mapping (e.g., `sl_atr_mult = base_sl - k * model3_uncertainty`) would allow Layer B to optimize dynamic trade management without vastly expanding the search space.
