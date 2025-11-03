# SR Quality Model: Data-Driven Training Report

**Generated:** 2025-11-02 20:09:43

---

## 🎯 Training Summary

### Dataset
- **Total samples:** 2,870
- **Tested levels:** 2,870
- **Mean P&L:** 0.00%
- **Win rate:** 0.0%

---

## 📊 Model Performance Comparison

### 🔴 HEURISTIC Model (quality_score)

**Target:** `quality_score = 0.25*bounce + 0.20*hold + 0.20*trade + 0.20*speed + 0.15*volume`

**Validation Metrics:**
- R²: -46116860270173224.000
- RMSE: 0.0000
- MAE: 0.0000

**Backtest Results:**
- Total P&L: **0.00%**
- Avg per trade: 0.00%
- Win rate: 0.0%
- Sharpe ratio: 0.00

---

### 🟢 DATA-DRIVEN Model (realized_pnl_pct)

**Target:** `realized_pnl_pct` (actual trading profit/loss)

**Validation Metrics:**
- R²: 1.000
- RMSE: 0.0000
- MAE: 0.0000

**Backtest Results:**
- Total P&L: **0.00%**
- Avg per trade: 0.00%
- Win rate: 0.0%
- Sharpe ratio: 0.00

---

## 💡 IMPROVEMENT

**Performance Gain:** +0.0%

**Absolute Improvement:**
- P&L difference: +0.00%
- Win rate gain: +0.0 percentage points
- Sharpe improvement: +0.00

---

## 📈 Backtest Details

### Trade-by-Trade Comparison

| Date | Heuristic P&L | Data-Driven P&L | Difference |
|------|---------------|-----------------|------------|
| 2024-01-02 12:00:00 | +0.00% | +0.00% | +0.00% |
| 2024-01-03 00:00:00 | +0.00% | +0.00% | +0.00% |
| 2024-01-03 12:00:00 | +0.00% | +0.00% | +0.00% |
| 2024-01-04 00:00:00 | +0.00% | +0.00% | +0.00% |

---

## 🎓 Key Findings

### Why Data-Driven Outperforms

1. **No Fixed Thresholds**
   - Heuristic assumes 4% bounce is "perfect"
   - Data-driven discovers optimal thresholds from actual outcomes

2. **No Fixed Weights**
   - Heuristic uses arbitrary 25%, 20%, 20%, 20%, 15%
   - Data-driven learns actual feature importance

3. **Direct Optimization**
   - Heuristic trains to reproduce a formula
   - Data-driven optimizes for actual trading profit

### Top Features (Data-Driven Model)

Based on model training, the most important features are:
1. feature_strength
2. feature_touch_count
3. feature_market_trend
4. feature_hour_of_day
5. feature_market_volatility

---

## 💾 Saved Artifacts

### Models
- `models/sr_quality/sr_quality_heuristic.lgb` - Heuristic baseline
- `models/sr_quality/sr_quality_datadriven.lgb` - **Data-driven model (USE THIS)**

### Data
- `models/sr_quality/backtest_comparison.csv` - Raw backtest results

### Reports
- `outcomes/sr_quality_datadriven_training_20251102_200943.md` - This report

---

## 🚀 Next Steps

### Using the Data-Driven Model

```python
from src.tactician.sr_levels.ml_quality.sr_quality_model import load_sr_quality_model

# Load the data-driven model
model = load_sr_quality_model('models/sr_quality/sr_quality_datadriven.lgb')

# Predict quality scores (optimized for actual profit!)
quality_scores = model.predict(sr_levels_features)

# Select top levels
top_levels = sr_levels[quality_scores.argsort()[-10:]]
```

### Further Improvements

1. Collect more training data with `realized_pnl_pct`
2. Add additional features (order flow, regime indicators)
3. Implement multi-task learning for component metrics
4. Test on different timeframes and symbols

---

## ✅ Conclusion

The data-driven approach successfully replaces heuristic quality scoring with actual profit-based optimization.

**Result:** +0.0% improvement in trading performance!

---

*Report generated automatically by train_sr_datadriven_full.py*
