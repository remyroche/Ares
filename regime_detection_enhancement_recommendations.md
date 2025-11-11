# Regime Detection ML Models - Enhancement Recommendations

**Analysis Date:** 2025-11-11
**Based on Reports:** regime_models_training_report, cluster_quality_report, regime_ensemble_training_report, hpo_summary_report

---

## Executive Summary

The regime detection models show **critical performance issues** requiring immediate attention:

- **Classification Accuracy:** 21.75% - 26.95% (barely better than random guessing for 6 classes)
- **Ensemble Model:** Complete failure (0% accuracy)
- **Cluster Separation:** Negative global silhouette score (-0.028)
- **Economic Distinctiveness:** Low between-regime variance (CV ratio: 1.03)

---

## Critical Issues Identified

### 1. **Poor Model Performance** 🔴 CRITICAL
**Current State:**
- Accuracy: 0.2175 - 0.2695 (for 6-class problem, random = 16.7%)
- F1-Score: 0.1978 - 0.2343
- Weighted Precision: 0.2277 - 0.2450

**Impact:** Models cannot reliably distinguish between market regimes, making them unsuitable for trading decisions.

---

### 2. **Ensemble Complete Failure** 🔴 CRITICAL
**Current State:**
- Accuracy: 0.0000
- Transition Entropy: 0.0000
- Average Regime Duration: 0.00 periods
- Number of Transitions: 0

**Impact:** The stacked ensemble (LightGBM meta-learner on ExtaTrees/RF/XGBoost) produces no valid predictions.

---

### 3. **Poor Cluster Quality** 🔴 CRITICAL
**Current State:**
- Global Silhouette Score: **-0.0280** (negative indicates poor separation)
- Davies-Bouldin Index: **3.2601** (lower is better, this is high)
- Clusters 4 & 5 have consistently negative silhouettes (-0.25, -0.27)
- Cross-validation accuracy: **50.24%** (barely better than coin flip)

**Impact:** Regimes are not well-separated in feature space, leading to overlapping boundaries.

---

### 4. **Severe Class Imbalance** 🟡 HIGH PRIORITY
**Current State:**
- Regime balance scores: 8.62 - 10.58 (higher = more imbalanced)
- Largest regime: 35.8% of samples
- Smallest regimes: 4.2% - 5.0% of samples
- Imbalance ratio: ~8.5:1

**Impact:** Models biased toward majority classes, minority regimes poorly learned.

---

### 5. **Low Economic Distinctiveness** 🟡 HIGH PRIORITY
**Current State:**
- Between-Regime CV / Within-Regime CV: **1.0259**
- Should be >> 1.0 for meaningful regime separation
- Some regimes have similar Sharpe ratios and return profiles

**Impact:** Regimes may not represent economically distinct market states.

---

### 6. **Potential Overfitting in Analyst Models** 🟡 HIGH PRIORITY
**Current State:**
- Analyst models show perfect metrics (1.0000 accuracy, recall, precision)
- Only 75 samples with 60 features (0.8:1 ratio)
- Post-HPO: RMSE = 0.0000, MAE = 0.0000 (unrealistic)

**Impact:** Models memorizing training data rather than learning generalizable patterns.

---

### 7. **Temporal Instability** 🟠 MEDIUM PRIORITY
**Current State:**
- High regime transition frequency (Temporal Smoothness: 0.798)
- Average regime duration: 4.95 bars (only ~1.24 hours on 15m timeframe)
- Indicates noisy regime assignments

**Impact:** Excessive regime switching leads to whipsaw trading signals.

---

## Enhancement Recommendations

### Phase 1: Foundation Fixes (Immediate - Weeks 1-2)

#### 1.1 Fix Ensemble Architecture 🔴
**Problem:** Ensemble producing zero predictions

**Solutions:**
```python
# Check for these issues:
1. Verify base model outputs are valid probabilities (sum to 1)
2. Check for NaN/Inf values in meta-features
3. Ensure calibration is not removing all predictions
4. Add fallback to best base model if ensemble fails
5. Implement proper cross-validation for meta-learner training
```

**Implementation:**
- Add input validation before meta-learner
- Log intermediate outputs for debugging
- Test with simple averaging ensemble as baseline
- Verify data shapes match between base models and meta-learner

---

#### 1.2 Address Class Imbalance 🔴
**Current:** 4.2% vs 35.8% regime sizes

**Solutions:**
```python
# Option 1: SMOTE for minority classes
from imblearn.over_sampling import SMOTE, ADASYN
smote = SMOTE(sampling_strategy='auto', k_neighbors=3)

# Option 2: Class weights
class_weights = compute_class_weight('balanced',
                                      classes=np.unique(y),
                                      y=y)

# Option 3: Focal Loss for extreme imbalance
from focal_loss import SparseCategoricalFocalLoss
loss = SparseCategoricalFocalLoss(gamma=2.0)

# Option 4: Ensemble with balanced bootstrap
from imblearn.ensemble import BalancedRandomForestClassifier
```

**Recommendation:** Combine class weights (easy) + SMOTE (moderate) + focal loss (advanced)

---

#### 1.3 Reduce Number of Regimes 🟡
**Current:** 6 regimes with poor separation

**Rationale:**
- With 3,267 samples and poor separation, 6 classes may be too many
- Try 3-4 regimes: Bull, Bear, Sideways, (+High Volatility)
- Better to predict fewer regimes accurately than many poorly

**Implementation:**
```python
# A. Test optimal K using multiple criteria
from sklearn.metrics import silhouette_score, calinski_harabasz_score

for k in range(3, 8):
    # Run clustering
    # Evaluate: silhouette, economic distinctiveness, predictability
    # Choose K with best trade-off

# B. Merge similar regimes
# Regimes 4 and 5 have negative silhouettes - consider merging
# Check correlation of regime characteristics
```

---

#### 1.4 Improve Feature Engineering 🟡
**Current:** 60 features for regime detection, some may be redundant

**Solutions:**
```python
# 1. Feature selection for regime detection
from sklearn.feature_selection import mutual_info_classif
mi_scores = mutual_info_classif(X, regime_labels)
top_features = features[np.argsort(mi_scores)[-30:]]  # Keep top 30

# 2. Add regime-specific features
regime_features = [
    # Volatility regime indicators
    'volatility_regime': rolling_std / rolling_mean,
    'volatility_percentile': rolling_std.rank(pct=True),

    # Trend regime indicators
    'trend_strength': abs(price - MA_50) / MA_50,
    'trend_direction': np.sign(price - MA_50),

    # Volume regime indicators
    'volume_regime': volume / rolling_volume_mean,
    'volume_shock': (volume - volume_mean) / volume_std,

    # Market microstructure
    'bid_ask_spread': (ask - bid) / mid_price,
    'price_impact': abs(returns) / volume,
]

# 3. Add temporal features
temporal_features = [
    'hour_of_day': timestamp.hour,
    'day_of_week': timestamp.dayofweek,
    'time_since_regime_change': periods_since_last_transition,
]
```

---

### Phase 2: Advanced Improvements (Weeks 3-4)

#### 2.1 Implement Hierarchical Regime Detection 🟢
**Concept:** Two-stage approach for better separation

```python
# Stage 1: Coarse regime detection (3 classes)
# - Bull (positive trend + medium vol)
# - Bear (negative trend + medium vol)
# - Sideways (no trend + any vol)

# Stage 2: Within each coarse regime, detect sub-regimes
# Bull -> [Low Vol Bull, High Vol Bull]
# Bear -> [Low Vol Bear, High Vol Bear]
# Sideways -> [Mean Reverting, Random Walk]
```

**Benefits:**
- Easier to separate 3 classes than 6
- Sub-regime classifiers trained on more homogeneous data
- Natural hierarchy matches market structure

---

#### 2.2 Use Regime-Aware Clustering 🟢

**Current:** Clustering seems purely statistical, not economic

**Solutions:**
```python
# 1. Constrained clustering with economic objectives
from sklearn_extra.cluster import KMedoids

# Define distance metric that considers:
# - Return similarity
# - Volatility similarity
# - Correlation structure
# - Trading cost regime

def economic_distance(x1, x2):
    return_diff = abs(x1['mean_return'] - x2['mean_return'])
    vol_diff = abs(x1['volatility'] - x2['volatility'])
    sharpe_diff = abs(x1['sharpe'] - x2['sharpe'])

    return w1*return_diff + w2*vol_diff + w3*sharpe_diff

# 2. Add must-link / cannot-link constraints
# e.g., high-vol bull and low-vol bull must NOT be in same regime
```

---

#### 2.3 Temporal Consistency Regularization 🟢

**Current:** Regimes change every 4.95 bars (too noisy)

**Solutions:**
```python
# 1. Add temporal smoothing to predictions
from scipy.ndimage import gaussian_filter1d

# Smooth regime probabilities over time
smoothed_probs = gaussian_filter1d(regime_probs, sigma=3, axis=0)

# 2. Minimum regime duration constraint
def enforce_min_duration(regimes, min_duration=10):
    # Filter out regimes lasting < min_duration
    # Assign to previous regime or most probable
    pass

# 3. Transition penalty in loss function
transition_penalty = lambda_t * np.sum(regime[t] != regime[t-1])
total_loss = classification_loss + transition_penalty

# 4. Use Hidden Markov Model (HMM) for temporal structure
from hmmlearn.hmm import GaussianHMM
hmm = GaussianHMM(n_components=6, covariance_type='full',
                   n_iter=100)
```

---

#### 2.4 Calibration and Confidence Thresholds 🟢

**Current:** Model makes hard predictions even with low confidence

**Solutions:**
```python
# 1. Isotonic calibration per class
from sklearn.calibration import CalibratedClassifierCV
calibrated_clf = CalibratedClassifierCV(base_model,
                                         method='isotonic',
                                         cv=5)

# 2. Confidence-based prediction
def predict_with_confidence(probs, threshold=0.5):
    max_prob = np.max(probs)
    if max_prob < threshold:
        return 'UNCERTAIN'  # Don't trade on uncertain regimes
    else:
        return np.argmax(probs)

# 3. Conformal prediction for uncertainty quantification
from crepes import ConformalClassifier
conformal = ConformalClassifier(base_model)
prediction_sets = conformal.predict(X_test, confidence=0.9)
```

---

#### 2.5 Alternative Clustering Methods 🟢

**Current:** Using one clustering method (likely K-means or GMM)

**Try:**
```python
# 1. HDBSCAN - better for non-spherical clusters
from hdbscan import HDBSCAN
clusterer = HDBSCAN(min_cluster_size=100,
                    min_samples=10,
                    metric='euclidean')

# 2. Spectral Clustering - captures complex manifolds
from sklearn.cluster import SpectralClustering
spectral = SpectralClustering(n_clusters=5,
                                affinity='rbf',
                                assign_labels='discretize')

# 3. Time Series Clustering
from tslearn.clustering import TimeSeriesKMeans
ts_kmeans = TimeSeriesKMeans(n_clusters=5,
                               metric='dtw',  # Dynamic Time Warping
                               max_iter=10)

# 4. Mixture of Hidden Markov Models
# Each regime = one HMM
# Allows temporal structure within regimes
```

---

### Phase 3: Validation & Deployment (Weeks 5-6)

#### 3.1 Enhanced Evaluation Metrics 📊

**Beyond accuracy - add domain-specific metrics:**

```python
evaluation_metrics = {
    # Statistical
    'accuracy': accuracy_score,
    'macro_f1': f1_score(average='macro'),
    'cohen_kappa': cohen_kappa_score,  # Accounts for chance

    # Economic
    'regime_profit_diff': mean_return_diff_between_regimes,
    'sharpe_by_regime': sharpe_ratio_per_regime,
    'regime_trading_cost': avg_transitions * cost_per_transition,

    # Temporal
    'regime_persistence': avg_regime_duration,
    'transition_entropy': entropy_of_transitions,
    'temporal_consistency': autocorr(regime_assignments, lag=1),

    # Risk Management
    'max_drawdown_by_regime': max_dd_per_regime,
    'var_95_by_regime': value_at_risk_per_regime,
    'regime_prediction_horizon': how_long_predictions_valid,
}
```

---

#### 3.2 Walk-Forward Validation 📊

**Current:** May be using standard train/test split (data leakage risk)

**Implement:**
```python
from sklearn.model_selection import TimeSeriesSplit

# Walk-forward validation
tscv = TimeSeriesSplit(n_splits=5, gap=100)  # Gap prevents leakage

for train_idx, test_idx in tscv.split(X):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Train regime detector on past data only
    model.fit(X_train, y_train)

    # Evaluate on future unseen data
    preds = model.predict(X_test)

    # Check performance degrades over time
    # -> May need online learning / retraining schedule
```

---

#### 3.3 Online Learning / Model Refresh 🔄

**Problem:** Markets evolve, static models decay

**Solutions:**
```python
# 1. Incremental learning
from river import tree, ensemble
model = ensemble.AdaptiveRandomForestClassifier(n_models=10)
for x, y in stream.iter_array(X, y):
    model.learn_one(x, y)

# 2. Sliding window retraining
RETRAIN_PERIOD = 30 * 24 * 4  # 30 days of 15min bars
if len(new_data) >= RETRAIN_PERIOD:
    retrain_model(new_data[-RETRAIN_PERIOD:])

# 3. Concept drift detection
from river.drift import ADWIN
drift_detector = ADWIN()
for pred, actual in zip(predictions, actuals):
    drift_detector.update(int(pred == actual))
    if drift_detector.drift_detected:
        print("Market regime shift detected - retraining")
        retrain_model()
```

---

## Prioritized Action Plan

### Week 1: Emergency Fixes
- [ ] **Debug ensemble model** - Fix 0% accuracy (2 days)
- [ ] **Implement class weights** - Handle imbalance (1 day)
- [ ] **Add confidence thresholds** - Don't predict when uncertain (1 day)
- [ ] **Reduce to 4 regimes** - Test if separation improves (1 day)

**Expected Impact:** Accuracy 0.27 → 0.45

---

### Week 2: Feature & Architecture
- [ ] **Feature selection** - Reduce 60 → 30 most relevant (2 days)
- [ ] **Add regime-specific features** - Volatility regime, trend strength (2 days)
- [ ] **Test SMOTE** - Oversample minority classes (1 day)

**Expected Impact:** Accuracy 0.45 → 0.58

---

### Week 3-4: Advanced Methods
- [ ] **Hierarchical regime detection** - Two-stage approach (3 days)
- [ ] **Temporal smoothing** - Add HMM or Gaussian filter (2 days)
- [ ] **Try HDBSCAN clustering** - Better separation (2 days)
- [ ] **Implement focal loss** - Further help with imbalance (1 day)

**Expected Impact:** Accuracy 0.58 → 0.70

---

### Week 5-6: Validation & Deployment
- [ ] **Walk-forward validation** - Test temporal stability (2 days)
- [ ] **Add economic metrics** - Sharpe/profit by regime (1 day)
- [ ] **Calibration** - Isotonic calibration (1 day)
- [ ] **Online learning** - Concept drift detection (3 days)

**Expected Impact:** Robust model with 0.70+ accuracy and economic value

---

## Quick Wins (Can Implement Today)

### 1. Merge Worst Clusters
```python
# Clusters 4 and 5 have negative silhouettes - merge them
regime_labels[regime_labels == 5] = 4
# Now 5 regimes instead of 6
```

### 2. Add Class Weights to LightGBM
```python
# In your LightGBM training config
params = {
    'objective': 'multiclass',
    'class_weight': 'balanced',  # Add this line
    # ... other params
}
```

### 3. Ensemble Fallback
```python
# If ensemble fails, use best base model
if np.sum(ensemble_preds) == 0:
    preds = best_base_model.predict(X)
else:
    preds = ensemble_preds
```

### 4. Confidence Filtering
```python
# Only use predictions with >60% confidence
high_conf = np.max(probs, axis=1) > 0.6
filtered_preds = preds[high_conf]
# For low confidence, use previous regime or "HOLD"
```

---

## Metrics to Track Improvement

| Metric | Current | Target | Priority |
|--------|---------|--------|----------|
| Accuracy | 0.27 | 0.70+ | 🔴 Critical |
| F1-Score (Macro) | 0.20 | 0.65+ | 🔴 Critical |
| Silhouette Score | -0.03 | +0.30 | 🔴 Critical |
| Ensemble Accuracy | 0.00 | 0.65+ | 🔴 Critical |
| Davies-Bouldin | 3.26 | <1.5 | 🟡 High |
| Between/Within CV | 1.03 | >3.0 | 🟡 High |
| Regime Duration | 4.95 | 10-20 | 🟠 Medium |
| Cross-Val Accuracy | 0.50 | 0.70+ | 🔴 Critical |

---

## References & Tools

### Libraries
- **Imbalanced-learn:** SMOTE, class weights
- **HDBSCAN:** Density-based clustering
- **focal-loss:** Advanced loss for imbalance
- **crepes:** Conformal prediction
- **river:** Online learning
- **hmmlearn:** Hidden Markov Models

### Research Papers
1. "A Survey of Unsupervised Learning Methods for Financial Market Regime Detection" (2020)
2. "Focal Loss for Dense Object Detection" - Lin et al. (2017)
3. "Conformal Prediction Under Covariate Shift" - Tibshirani et al. (2019)

---

## Conclusion

The current regime detection system has **critical accuracy issues** stemming from:
1. Poor cluster separation
2. Severe class imbalance
3. Too many regimes for available data
4. Ensemble architecture failure

**Priority 1:** Fix ensemble, reduce regimes, add class weights (Week 1)
**Priority 2:** Feature engineering, temporal smoothing (Weeks 2-4)
**Priority 3:** Validation, deployment, monitoring (Weeks 5-6)

**Expected Outcome:** Improve accuracy from 27% to 70%+, making the system viable for trading.

---

*Generated: 2025-11-11*
*For questions: Review code in `src/training/steps/market_analysis/` and clustering pipelines*
