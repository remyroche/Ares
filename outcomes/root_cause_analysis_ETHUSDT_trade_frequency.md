# Root Cause Analysis: Low Trade Frequency on ETHUSDT

## Executive Summary

**Current State:**
- Layer 2: **0.34 trades/day** (target: 2-3 trades/day)
- Layer 3: **0.89 trades/day** (target: 2-3 trades/day)
- When trade frequency increases, profitability becomes negative

**Target:** 2-3 profitable trades per day on ETHUSDT 15m timeframe

---

## Key Findings from Latest Reports

### 1. **CRITICAL: Weak Probability-Return Correlation**

**Layer 2 Report (20251215_231303):**
- `prob_return_spearman: 0.062` - **Extremely weak correlation** between predicted probabilities and actual returns
- `prob_return_kendall: 0.041` - Very weak rank correlation
- **Top quartile by probability**: mean_return = **-0.00244** (NEGATIVE!)
- **Bottom quartile by probability**: mean_return = **-0.00387** (NEGATIVE!)
- **Difference**: Only 0.00143 (minimal separation)

**Root Cause:** The model's probability predictions are NOT aligned with actual profitability. High probabilities don't correspond to profitable trades.

**Impact:** The system cannot reliably identify profitable opportunities, forcing it to use extremely high thresholds (0.70) to filter trades, which dramatically reduces frequency.

---

### 2. **CRITICAL: Poor Model Calibration**

**Layer 2:**
- `calibration_brier: 0.211` (moderate, room for improvement)
- `calibration_ece: 0.144` (moderate calibration error)

**Layer 3:**
- `calibration_brier: 0.257` (poor calibration)
- `calibration_ece: 0.231` (high calibration error)
- `calibration_mce: 0.832` (very high max calibration error)

**Root Cause:** Model probabilities are poorly calibrated - they don't accurately reflect the true probability of profitable trades.

**Impact:** Cannot trust probability thresholds. A threshold of 0.70 doesn't actually mean 70% chance of profit.

---

### 3. **Very High Probability Threshold**

**Layer 2 Best Params:**
- `prob_threshold: 0.699994` (essentially 0.70)
- `take_rate: 0.0387` (only 3.87% of events become trades)
- `valid_events: 3180` → `n_trades: 123` (96.1% filtered out)

**Probability Threshold Sweep Results:**
- Best threshold found: **0.72** with only **0.30 trades/day**
- At this threshold: mean_return = 0.000343 (barely positive)

**Root Cause:** System must use extremely high thresholds because:
1. Low probability-return correlation means probabilities aren't reliable
2. Lower thresholds include too many unprofitable trades
3. Model calibration issues make threshold selection unreliable

**Impact:** Trade frequency is artificially constrained by necessity to maintain profitability.

---

### 4. **Layer 3 Performance Issues**

**Layer 3 Report (20251215_223051):**
- `sharpe_mean: -0.989` (**NEGATIVE Sharpe ratio**)
- `mean_auc: 0.576` (barely above random 0.50)
- `trades_per_day: 0.89` (still below target)

**Per-Fold Performance:**
- Fold 0: sharpe = -0.107, mean_return = -0.000050
- Fold 1: sharpe = 3.368, mean_return = 0.002843 ✅
- Fold 2: sharpe = -1.686, mean_return = -0.001457 ❌
- Fold 3: sharpe = 0.706, mean_return = 0.000656 ✅
- Fold 4: sharpe = -7.227, mean_return = -0.004467 ❌

**Root Cause:** Extreme fold-to-fold instability. Model performance is inconsistent across time periods.

**Impact:** Cannot rely on Layer 3 predictions. Some periods are profitable, others are highly unprofitable.

---

### 5. **Regime-Specific Problems**

**Layer 3 Per-Regime Analysis:**

**Volatility Regimes:**
- Low volatility: sharpe = **-1.524**, mean_return = **-0.001143** ❌
- Medium volatility: sharpe = 0.676, mean_return = 0.000931 ✅
- High volatility: sharpe = 0.037, mean_return = 0.000048 (barely positive)

**Trend Regimes:**
- Low trend: sharpe = **-1.548**, mean_return = **-0.001109** ❌
- Medium trend: sharpe = -0.413, mean_return = -0.000661 ❌
- High trend: sharpe = 0.758, mean_return = 0.001153 ✅

**Combined Regimes:**
- `vol_low__trend_low`: sharpe = **-1.052**, mean_return = **-0.000962** ❌
- `vol_low__trend_medium`: sharpe = **-1.841**, mean_return = **-0.002857** ❌
- `vol_high__trend_low`: sharpe = **-1.550**, mean_return = **-0.002045** ❌

**Root Cause:** Model fails in specific market regimes (low volatility, low trend, or combinations). These regimes represent a significant portion of market conditions.

**Impact:** System cannot trade profitably in many market conditions, further reducing viable trade opportunities.

---

### 6. **Label Quality vs Prediction Quality Mismatch**

**Layer 2 Label-Return Alignment:**
- `label_return_spearman: 0.607` - **Good correlation** between labels and returns
- `winrate_when_label_1: 0.738` - Labels are predictive
- `winrate_when_label_0: 0.088` - Labels correctly identify losers

**BUT:**
- `prob_return_spearman: 0.062` - **Poor correlation** between predictions and returns

**Root Cause:** The labels are good quality, but the model is not learning to predict them effectively. There's a disconnect between:
- What the labels indicate (profitable opportunities)
- What the model predicts (probabilities that don't correlate with returns)

**Impact:** Even though the training data is good, the model fails to generalize the signal.

---

## Upstream Root Causes

### A. **Model Training Issues**

1. **Feature Quality:**
   - Layer 3 uses 186 selected features from 1131 original features
   - Feature selection may have removed important signals
   - Regime-specific features may not be capturing regime dynamics effectively

2. **Training Instability:**
   - High fold-to-fold variance in Layer 3 (sharpe ranges from -7.23 to 3.37)
   - Suggests overfitting or insufficient regularization
   - Model may be learning spurious patterns that don't generalize

3. **Calibration Not Optimized:**
   - Calibration errors are high but may not be penalized enough in training
   - Model focuses on AUC rather than calibration quality

### B. **Labeling Pipeline Issues**

1. **Label-Return Alignment is Good BUT:**
   - Labels may be too sparse (only 26.7% positive labels)
   - May be missing profitable opportunities that don't match label criteria
   - Label definition may be too restrictive

2. **Temporal Instability:**
   - Early vs late fold performance gap (early_mean: 1.63, late_mean: -2.74)
   - Suggests regime shift or concept drift
   - Model trained on early data doesn't work on recent data

### C. **Layer 2 Gating Logic Issues**

1. **Overly Conservative Thresholds:**
   - Must use 0.70+ threshold to maintain profitability
   - This filters out 96% of events
   - May be filtering out profitable trades that have lower probabilities

2. **EV Margin Too High:**
   - `ev_margin: 0.214` (21.4% margin required)
   - Combined with high prob_threshold, creates double-filtering
   - May be rejecting trades that are actually profitable

3. **Regime Barriers Too Restrictive:**
   - `barrier_regime_strength: 0.122` and `barrier_regime_power: 0.804`
   - May be filtering out trades in certain regimes unnecessarily

---

## Recommendations

### Immediate Actions (High Priority)

1. **Fix Probability-Return Correlation:**
   - Investigate why model probabilities don't correlate with returns
   - Add explicit loss terms to penalize poor probability-return correlation
   - Consider using return-weighted training or ranking losses

2. **Improve Model Calibration:**
   - Add calibration loss to training objective
   - Use Platt scaling or isotonic regression for post-hoc calibration
   - Monitor calibration metrics during training

3. **Reduce Threshold Conservatism:**
   - If probabilities were better calibrated, could use lower thresholds
   - Test threshold optimization with better-calibrated probabilities
   - Consider adaptive thresholds based on regime

### Medium-Term Actions

4. **Address Regime-Specific Failures:**
   - Investigate why low volatility/low trend regimes fail
   - May need regime-specific models or features
   - Consider excluding unprofitable regimes rather than trading them

5. **Improve Training Stability:**
   - Increase regularization to reduce fold-to-fold variance
   - Use ensemble methods to stabilize predictions
   - Consider temporal validation strategies

6. **Review Feature Engineering:**
   - Audit feature selection process
   - Ensure regime features capture regime dynamics
   - Consider adding features specific to profitable regimes

### Long-Term Actions

7. **Label Definition Review:**
   - Evaluate if labels are too restrictive
   - Consider expanding label definition to capture more opportunities
   - Test alternative labeling strategies

8. **Layer 2 Optimization:**
   - Review EV margin and regime barrier logic
   - Consider regime-adaptive thresholds
   - Test less conservative gating strategies

---

## Expected Impact

If probability-return correlation improves from 0.062 to 0.3+:
- Can use lower thresholds (0.55-0.60 instead of 0.70)
- Take rate could increase from 3.87% to 10-15%
- Trade frequency could increase from 0.34 to 1.0-1.5 trades/day

If calibration improves:
- Thresholds become more reliable
- Can optimize thresholds more effectively
- Better trade selection

If regime-specific issues are addressed:
- Can trade in more market conditions
- Further increase in trade frequency
- More consistent profitability

**Combined, these improvements could achieve the target of 2-3 profitable trades per day.**
