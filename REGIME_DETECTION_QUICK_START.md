# Regime Detection Quick Start Guide
**Emergency Action Plan**

## 🚨 Current Status: CRITICAL FAILURE

- **Ensemble Accuracy:** 0.0% (Should be >60%)
- **Average Model F1:** 0.13 (Should be >0.60)
- **Undetectable Regimes:** 2 out of 6 (Regimes 0, 3)

## ⚡ Immediate Actions (Start Here)

### 1. Run Analysis Script (5 minutes)
```bash
cd /Users/remyroche/Documents/Ares
python scripts/analyze_regime_reports.py
```
This will generate visual dashboards showing all issues.

### 2. Debug Ensemble Training (Priority #1)
**File:** `src/training/steps/market_analysis/components/regime_ensemble_training.py`

**Add logging around line 587-650:**
```python
tprint("🔍 [DEBUG] Starting ensemble training...", color="yellow")
tprint(f"   Meta-features shape: {meta_features.shape}", color="yellow")
tprint(f"   Labels shape: {y_train.shape}", color="yellow")
tprint(f"   Unique labels: {np.unique(y_train)}", color="yellow")

# Before calibration
tprint(f"🔍 [DEBUG] Pre-calibration model type: {type(meta_learner)}", color="yellow")

# After calibration
tprint(f"🔍 [DEBUG] Post-calibration model type: {type(calibrated_model)}", color="yellow")
tprint(f"   Calibration method: {calibration_method}", color="yellow")

# Test prediction
test_pred = calibrated_model.predict(meta_features[:10])
tprint(f"🔍 [DEBUG] Test predictions: {test_pred}", color="yellow")
```

### 3. Filter Invalid Regimes (Priority #2)
**File:** `src/training/steps/market_analysis/rolling_hmm_clustering/clustering.py`

**Add validation before saving regimes:**
```python
def validate_regime(regime_data):
    """Validate regime is learnable and economically meaningful."""
    
    # Reject if too few samples
    if regime_data['n_samples'] < 50:
        return False, f"Too few samples: {regime_data['n_samples']}"
    
    # Reject if extreme drawdown (likely data artifact)
    if regime_data['max_drawdown'] < -0.80:
        return False, f"Extreme drawdown: {regime_data['max_drawdown']:.2%}"
    
    # Reject if unrealistic Sharpe
    if abs(regime_data['sharpe_ratio']) > 10:
        return False, f"Unrealistic Sharpe: {regime_data['sharpe_ratio']:.2f}"
    
    return True, "Valid"

# Apply before regime assignment
valid_regimes = []
for regime_id, regime_data in regimes.items():
    is_valid, reason = validate_regime(regime_data)
    if is_valid:
        valid_regimes.append(regime_id)
        tprint(f"✅ Regime {regime_id}: {reason}", color="green")
    else:
        tprint(f"❌ Regime {regime_id} rejected: {reason}", color="red")
```

### 4. Fix Class Imbalance (Priority #3)
**File:** `src/training/steps/market_analysis/components/regime_models_training.py`

**Add adaptive SMOTE around line 2000:**
```python
from imblearn.over_sampling import SMOTE
from collections import Counter

# Check class distribution
class_counts = Counter(y_train)
tprint(f"📊 Original class distribution: {class_counts}", color="blue")

# Apply adaptive SMOTE only to small classes
min_samples = min(class_counts.values())
if min_samples < 50:
    tprint("⚠️ Class imbalance detected, applying adaptive SMOTE", color="yellow")
    
    # Calculate sampling strategy
    target_samples = 100  # Minimum samples per class
    sampling_strategy = {
        cls: max(target_samples, count) 
        for cls, count in class_counts.items()
    }
    
    smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)
    
    tprint(f"✅ Resampled distribution: {Counter(y_train)}", color="green")
```

## 📊 Expected Results After Fixes

| Metric | Before | After (Target) | Status |
|--------|--------|----------------|--------|
| Ensemble Accuracy | 0.00 | >0.60 | 🎯 |
| Average Model F1 | 0.13 | >0.45 | 🎯 |
| Worst Regime F1 | 0.00 | >0.30 | 🎯 |
| Valid Regimes | 6 | 4-5 | 🎯 |

## 📁 Key Files

**Training:**
- `src/training/steps/market_analysis/components/regime_ensemble_training.py` (Ensemble)
- `src/training/steps/market_analysis/components/regime_models_training.py` (Base models)

**Clustering:**
- `src/training/steps/market_analysis/rolling_hmm_clustering/clustering.py`
- `src/training/steps/market_analysis/sticky_finite_hmm_clustering/clustering.py`

**Validation:**
- `src/utils/ml_common/validation/regime_walk_forward_validator.py`

**Reports:**
- `outcomes/regime_ensemble_training_metrics_ETHUSDT_20251111_024656.csv`
- `outcomes/regime_performance_by_model_ETHUSDT_20251111_022208.csv`
- `outcomes/temporal_regime_analysis_ETHUSDT_20251111_024656.csv`

## 🔄 Testing After Changes

```bash
# 1. Re-run regime clustering with validation
python -m src.training.steps.market_analysis.rolling_hmm_clustering.clustering \
    --symbol ETHUSDT \
    --validate-regimes

# 2. Re-train base models with balanced classes
python -m src.training.steps.market_analysis.components.regime_models_training \
    --symbol ETHUSDT \
    --apply-smote

# 3. Re-train ensemble with fixes
python -m src.training.steps.market_analysis.components.regime_ensemble_training \
    --symbol ETHUSDT \
    --debug-mode

# 4. Verify improvements
python scripts/analyze_regime_reports.py
```

## 📚 Full Documentation

See **REGIME_DETECTION_IMPROVEMENT_PLAN.md** for:
- Complete root cause analysis
- 4-phase improvement strategy
- Advanced techniques (hierarchical models, focal loss, etc.)
- Timeline and success metrics

## ❓ Quick Wins (< 1 Day)

1. ✅ **Add debug logging** to ensemble training (30 min)
2. ✅ **Filter invalid regimes** based on economics (1 hour)
3. ✅ **Apply adaptive SMOTE** to balance classes (2 hours)
4. ✅ **Try different meta-learner** (XGBoost instead of LightGBM) (1 hour)
5. ✅ **Increase min_regime_samples** to 50 in config (5 min)

## 🆘 If Still Failing

1. **Check data quality:**
   - Verify no NaN/Inf values in features
   - Check temporal ordering
   - Look for data leakage

2. **Simplify problem:**
   - Start with 3 regimes instead of 6
   - Use only top 20 features
   - Train on subset of data first

3. **Alternative approach:**
   - Use simple K-means clustering as baseline
   - Compare with HMM-based regimes
   - Validate regime assignments manually

## 📞 Support

Questions? Check:
- Logs in `logs/regime_detection.log`
- Training artifacts in `artifacts/`
- Previous successful runs for comparison

---

**Status:** Ready to implement  
**Estimated time to fix:** 1-2 days  
**Expected improvement:** 0% → 60%+ accuracy
