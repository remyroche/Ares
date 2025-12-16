# Actionable Insights: Increasing Trade Frequency on ETHUSDT

## Current Performance vs Target

| Metric | Current (Layer 2) | Current (Layer 3) | Target | Gap |
|--------|------------------|-------------------|--------|-----|
| Trades/Day | 0.34 | 0.89 | 2-3 | **-85% to -70%** |
| Take Rate | 3.87% | ~10% | 15-25% | **-74% to -60%** |
| Prob Threshold | 0.70 | ~0.50 | 0.55-0.60 | Too high |
| Sharpe Ratio | 1.17 | -0.99 | >1.0 | Layer 3 failing |

---

## The Core Problem: Probability-Return Disconnect

### What Should Happen:
- High probability predictions → High returns
- Low probability predictions → Low/negative returns
- Clear separation between top and bottom predictions

### What Actually Happens:
```
Top Quartile (by probability):  mean_return = -0.00244  ❌
Bottom Quartile (by probability): mean_return = -0.00387 ❌
Correlation: 0.062 (essentially random)
```

**The model cannot distinguish profitable from unprofitable trades based on probability.**

---

## Why Thresholds Must Be So High

### Current Situation:
- **3,180 events** available
- **123 trades** taken (3.87% take rate)
- **3,057 events filtered out** (96.1%)

### Why So Much Filtering?
1. **Probability threshold = 0.70** filters out ~95% of events
2. **EV margin = 21.4%** adds additional filtering
3. **Regime barriers** filter out more events
4. **Result:** Only the absolute highest probability events pass

### The Problem:
Even at threshold 0.72 (from sweep), only **0.30 trades/day** with barely positive returns (0.000343).

**Lowering the threshold doesn't help because probabilities aren't reliable indicators of profitability.**

---

## Root Cause Chain

```
1. Model Training Issues
   ↓
2. Poor Probability Calibration (ECE: 0.14-0.23)
   ↓
3. Weak Probability-Return Correlation (0.062)
   ↓
4. Cannot Trust Probabilities for Trade Selection
   ↓
5. Must Use Extremely High Thresholds (0.70+)
   ↓
6. Low Take Rate (3.87%)
   ↓
7. Low Trade Frequency (0.34/day)
```

---

## Specific Code/Config Issues to Investigate

### 1. Probability-Return Correlation Penalty
**Location:** `src/training/steps/labeling/meta_labeling_hpo_sample_weighted.py`
- Line 13060: `layer2_prob_return_spearman_penalty_lambda` (default: 2.0)
- **Issue:** Penalty may not be strong enough or not working correctly
- **Action:** Verify penalty is actually applied and increase if needed

### 2. Calibration Loss
**Location:** Layer 3 training
- **Issue:** No explicit calibration loss in training objective
- **Action:** Add calibration loss (Brier score or ECE) to training

### 3. Threshold Search Space
**Location:** `meta_labeling_hpo_sample_weighted.py` line 10907
- Current: `"low": 0.50, "high": float(l2_prob_thr_high)` (default 0.70)
- **Issue:** Search space may be too constrained
- **Action:** But first fix the correlation issue, then expand search space

### 4. EV Margin
**Location:** Layer 2 params
- Current: `ev_margin: 0.214` (21.4%)
- **Issue:** Combined with high prob_threshold creates double-filtering
- **Action:** Reduce or make adaptive based on probability confidence

---

## Immediate Fixes (Priority Order)

### Fix #1: Add Explicit Probability-Return Correlation Loss ⚡
**Impact:** HIGH - Directly addresses root cause
**Effort:** MEDIUM

Add to training loss:
```python
prob_return_corr_loss = -spearman_corr(predictions, actual_returns)
total_loss = classification_loss + lambda_corr * prob_return_corr_loss
```

**Expected Result:** Correlation improves from 0.062 → 0.3+, enabling lower thresholds

### Fix #2: Improve Calibration ⚡
**Impact:** HIGH - Makes thresholds reliable
**Effort:** MEDIUM

Options:
1. Add calibration loss to training
2. Post-hoc calibration (Platt scaling, isotonic regression)
3. Temperature scaling

**Expected Result:** ECE improves from 0.14-0.23 → <0.10, thresholds become meaningful

### Fix #3: Regime-Specific Thresholds ⚡
**Impact:** MEDIUM - Increases trades in profitable regimes
**Effort:** LOW

Use different thresholds for different regimes:
- High trend: Lower threshold (more trades)
- Low trend: Higher threshold or skip
- Low volatility: Skip or very high threshold

**Expected Result:** Trade frequency increases by 30-50% while maintaining profitability

### Fix #4: Review Feature Selection ⚡
**Impact:** MEDIUM - May improve model quality
**Effort:** MEDIUM

- Audit which features were dropped
- Ensure regime features are included
- Test with expanded feature set

**Expected Result:** Model quality improves, better predictions

### Fix #5: Reduce EV Margin ⚡
**Impact:** LOW-MEDIUM - Quick win if probabilities improve
**Effort:** LOW

- Reduce from 21.4% to 10-15%
- Or make it adaptive based on probability confidence

**Expected Result:** 10-20% increase in trade frequency

---

## Testing Strategy

### Phase 1: Fix Probability-Return Correlation
1. Add correlation loss to training
2. Retrain Layer 3 model
3. Verify correlation improves to >0.3
4. Re-run Layer 2 HPO with improved probabilities
5. **Target:** Threshold can drop to 0.55-0.60, take rate increases to 10-15%

### Phase 2: Improve Calibration
1. Add calibration loss or post-hoc calibration
2. Verify ECE improves to <0.10
3. Re-optimize thresholds with calibrated probabilities
4. **Target:** More reliable thresholds, better trade selection

### Phase 3: Regime Optimization
1. Analyze per-regime performance
2. Implement regime-specific thresholds
3. Test and validate
4. **Target:** Trade frequency increases to 1.5-2.0/day

### Phase 4: Fine-tuning
1. Optimize EV margin
2. Review feature set
3. Final threshold optimization
4. **Target:** Achieve 2-3 trades/day with positive Sharpe

---

## Success Metrics

### Minimum Viable:
- ✅ Probability-return correlation > 0.25
- ✅ Calibration ECE < 0.15
- ✅ Trade frequency > 1.0/day
- ✅ Sharpe ratio > 0.5

### Target:
- ✅ Probability-return correlation > 0.40
- ✅ Calibration ECE < 0.10
- ✅ Trade frequency 2-3/day
- ✅ Sharpe ratio > 1.0

### Stretch:
- ✅ Probability-return correlation > 0.50
- ✅ Calibration ECE < 0.05
- ✅ Trade frequency 3-4/day
- ✅ Sharpe ratio > 1.5

---

## Key Files to Modify

1. **`src/training/steps/labeling/meta_labeling_hpo_sample_weighted.py`**
   - Add probability-return correlation loss
   - Improve calibration handling

2. **Layer 3 training code**
   - Add calibration loss
   - Improve regularization for stability

3. **Layer 2 gating logic**
   - Implement regime-specific thresholds
   - Reduce EV margin or make adaptive

4. **Feature selection code**
   - Review dropped features
   - Ensure regime features included

---

## Expected Timeline

- **Week 1:** Fix probability-return correlation → Trade frequency: 0.34 → 0.8-1.0/day
- **Week 2:** Improve calibration → Trade frequency: 1.0 → 1.2-1.5/day
- **Week 3:** Regime optimization → Trade frequency: 1.5 → 2.0-2.5/day
- **Week 4:** Fine-tuning → Trade frequency: 2.5 → 2.5-3.0/day ✅

**Total: 4 weeks to achieve target of 2-3 trades/day**
