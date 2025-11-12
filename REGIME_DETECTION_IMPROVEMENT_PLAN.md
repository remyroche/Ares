# Regime Detection Improvement Plan
**Date:** November 11, 2025  
**Analysis Based On:** ETHUSDT Reports (2025-11-11)

---

## Executive Summary

The regime detection system is experiencing **catastrophic failure** with the ensemble model showing **0.0% accuracy**. Individual base models also perform very poorly (F1 scores: 0.11-0.13). Multiple critical issues have been identified across class imbalance, model architecture, and regime definitions.

---

## 🚨 Critical Issues Identified

### 1. **CRITICAL: Ensemble Complete Failure**
- **Accuracy: 0.0000** (catastrophic)
- **Status:** System is unusable in production
- **Impact:** No regime predictions possible with current ensemble

**Root Causes:**
- Ensemble meta-learner (LightGBM) not learning effectively
- No calibration applied (method='none' despite config saying 'isotonic')
- Possible feature mismatch or data leakage issues

### 2. **CRITICAL: Severe Class Imbalance**
- **Regime 0:** 28 samples (4.3%) - **undetectable** (F1=0.0048)
- **Regime 3:** 19 samples (2.9%) - **undetectable** (F1=0.0000)
- **Regime 5:** 168 samples (25.8%) - best detection (F1=0.2541)
- **Support Range:** 19-196 samples across 6 regimes

**Implications:**
- Models cannot learn minority regimes (0, 3)
- Severe bias toward majority regimes
- SMOTE/oversampling not effectively addressing imbalance

### 3. **CRITICAL: Questionable Regime Definitions**
Several regimes show extreme characteristics that question their validity:

- **Regime 2:** -100% max drawdown, negative Sharpe (-7.15)
- **Regime 3:** -99.3% max drawdown, negative Sharpe (-6.71)
- **Regime 5:** -100% max drawdown, negative Sharpe (-5.51)

**Implications:**
- These may represent data artifacts or market crashes
- Models trying to learn from invalid/extreme regimes
- Need regime validation and filtering

### 4. **HIGH: Poor Individual Model Performance**
All base models perform poorly:
- **ExtraTrees:** F1=0.1292 (best)
- **LightGBM:** F1=0.1323
- **CatBoost:** F1=0.1299
- **XGBoost:** F1=0.1169
- **Random Forest:** F1=0.1073 (worst)

**Expected F1 for good regime detection:** 0.6-0.8+  
**Current F1:** 0.11-0.13 (85% below target)

### 5. **MEDIUM: Temporal Characteristics**
Positive indicators:
- Good persistence scores (0.43-0.58)
- Reasonable mean durations (3.7-5.9 periods)
- Regimes 1 & 4 show positive economics (Sharpe 5.46, 7.16)

---

## 📊 Performance Breakdown

### Per-Regime Analysis

| Regime | F1 Score | Precision | Recall | Support | Avg Return | Sharpe | Max DD | Status |
|--------|----------|-----------|--------|---------|------------|--------|--------|--------|
| 0 | 0.0048 | 0.0036 | 0.0071 | 28 | +0.08% | 4.57 | -1.4% | ❌ Undetectable |
| 1 | 0.1344 | 0.2595 | 0.0939 | 196 | +0.40% | 5.46 | -6.6% | ⚠️ Poor |
| 2 | 0.2051 | 0.1682 | 0.2819 | 127 | -0.59% | -7.15 | **-100%** | ⚠️ Invalid? |
| 3 | 0.0000 | 0.0000 | 0.0000 | 19 | -0.09% | -6.71 | **-99%** | ❌ Undetectable |
| 4 | 0.1403 | 0.1585 | 0.1404 | 114 | +0.25% | 7.16 | -2.0% | ⚠️ Poor |
| 5 | 0.2541 | 0.2275 | 0.3167 | 168 | -0.19% | -5.51 | **-100%** | ⚠️ Invalid? |

**Key Observations:**
- Only Regime 5 has somewhat acceptable detection (F1=0.25)
- Regimes 0 & 3 are essentially invisible to models
- Three regimes (2, 3, 5) have complete or near-complete drawdowns
- High precision, very low recall pattern suggests model uncertainty

---

## 🎯 Improvement Strategy

### Phase 1: Emergency Fixes (Week 1) - **PRIORITY: CRITICAL**

#### 1.1 Fix Ensemble Training
**Objective:** Restore ensemble to functional state

**Actions:**
```python
# In regime_ensemble_training.py
- [ ] Verify calibration is actually applied (currently shows 'none')
- [ ] Add extensive logging before/after ensemble training
- [ ] Check feature alignment between base models and meta-learner
- [ ] Validate no data leakage in meta-feature generation
- [ ] Add ensemble training validation checks
- [ ] Try alternative meta-learners (XGBoost, CatBoost as backup)
```

**Validation Metrics:**
- Target ensemble accuracy: >0.60 (vs current 0.00)
- Should exceed best base model by at least 5%

#### 1.2 Implement Intelligent Class Balancing
**Objective:** Make minority regimes learnable

**Strategy: Multi-tier Approach**

```python
# 1. Regime Filtering (remove invalid regimes)
REGIME_VALIDITY_CRITERIA = {
    'min_samples': 50,  # Reject if <50 samples
    'max_drawdown_threshold': -0.80,  # Reject if worse than -80%
    'min_sharpe': -3.0,  # Reject if Sharpe < -3
}

# 2. Adaptive Sampling Strategy
SAMPLING_STRATEGY = {
    'very_small': (lambda n: n < 50),     # Regime 0, 3 -> Combine or remove
    'small': (lambda n: 50 <= n < 100),   # -> SMOTE to 100
    'medium': (lambda n: 100 <= n < 150), # -> SMOTE to 150  
    'large': (lambda n: n >= 150),        # -> No change
}

# 3. Class Weights
from sklearn.utils.class_weight import compute_sample_weight
sample_weights = compute_sample_weight('balanced', y_train)

# 4. Focal Loss for Hard Examples
def focal_loss(y_true, y_pred, gamma=2.0, alpha=0.25):
    # Penalize misclassified minority classes more heavily
    pass
```

**Implementation Priority:**
1. **Immediate:** Remove regimes with <50 samples (0, 3)
2. **Short-term:** Apply adaptive SMOTE
3. **Medium-term:** Implement focal loss

#### 1.3 Regime Validation & Consolidation
**Objective:** Remove invalid/extreme regimes

**Actions:**
```python
# In rolling_hmm_clustering or regime_clustering step
def validate_regime_quality(regime_data, regime_id):
    """Validate if regime is learnable and economically meaningful."""
    
    # Economic validity
    if regime_data['max_drawdown'] < -0.80:
        return False, "Extreme drawdown (likely data artifact)"
    
    if abs(regime_data['sharpe_ratio']) > 10:
        return False, "Unrealistic Sharpe ratio"
    
    # Statistical validity
    if regime_data['n_samples'] < MIN_REGIME_SAMPLES:
        return False, f"Insufficient samples ({regime_data['n_samples']})"
    
    # Temporal validity
    if regime_data['persistence_score'] < 0.3:
        return False, "Too unstable (poor persistence)"
    
    return True, "Valid"

# Consolidation strategy
def consolidate_small_regimes(regimes, min_samples=50):
    """Merge regimes with similar characteristics and low support."""
    
    small_regimes = [r for r in regimes if r['n_samples'] < min_samples]
    
    # Cluster small regimes by feature similarity
    # Merge into larger regimes or create "other" category
    pass
```

**Expected Outcome:**
- Reduce from 6 regimes to 4-5 meaningful ones
- Remove regimes 2, 3, 5 (complete drawdowns)
- Merge regime 0 into similar regime if possible

---

### Phase 2: Model Architecture Improvements (Week 2-3)

#### 2.1 Enhanced Feature Engineering
**Current Issue:** Models may lack discriminative features

**Improvements:**
```python
# Add regime-specific discriminative features
DISCRIMINATIVE_FEATURES = [
    # Volatility regime features
    'volatility_regime_score',      # Custom metric for vol classification
    'volatility_percentile_rank',    # Historical vol percentile
    'volatility_acceleration',       # Rate of vol change
    
    # Trend regime features  
    'trend_strength_composite',      # Combine ADX, slope, R²
    'trend_consistency_score',       # Measure of trend stability
    'trend_regime_probability',      # Bayesian trend classification
    
    # Market structure features
    'market_microstructure_score',   # Order flow, spread patterns
    'regime_transition_probability', # Likelihood of regime change
    'regime_persistence_signal',     # How stable is current regime
]

# Feature interaction terms (critical for regime detection)
def create_regime_interaction_features(df):
    """Create polynomial and interaction features for regime detection."""
    
    # Key interactions
    df['vol_trend_interaction'] = df['volatility'] * df['trend_strength']
    df['vol_momentum_interaction'] = df['volatility'] * df['momentum']
    df['volume_vol_interaction'] = df['volume'] * df['volatility']
    
    # Quadratic terms for non-linear boundaries
    df['volatility_squared'] = df['volatility'] ** 2
    df['momentum_squared'] = df['momentum'] ** 2
    
    return df
```

#### 2.2 Advanced Model Architectures
**Current:** Using standard tree models  
**Improvement:** Try specialized architectures

**Option A: Deep Learning Approach**
```python
# Temporal CNN for regime classification
class RegimeCNN(nn.Module):
    def __init__(self, n_features, n_regimes, sequence_length=20):
        super().__init__()
        self.conv1 = nn.Conv1d(n_features, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, n_regimes)
    
    def forward(self, x):
        # x: [batch, sequence_length, n_features]
        x = x.transpose(1, 2)  # [batch, n_features, sequence_length]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x).squeeze(-1)
        return self.fc(x)
```

**Option B: LSTM with Attention**
```python
# For capturing temporal regime patterns
class RegimeLSTMAttention(nn.Module):
    def __init__(self, n_features, n_regimes, hidden_size=128):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden_size, batch_first=True, num_layers=2)
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=4)
        self.fc = nn.Linear(hidden_size, n_regimes)
```

**Option C: Hierarchical Model**
```python
# Stage 1: Coarse regime classification (high-vol vs low-vol)
# Stage 2: Fine-grained classification within each coarse regime
class HierarchicalRegimeDetector:
    def __init__(self):
        self.coarse_classifier = LGBMClassifier()  # 2-3 macro regimes
        self.fine_classifiers = {}  # One per macro regime
    
    def fit(self, X, y):
        # Train coarse classifier first
        coarse_labels = self._map_to_coarse_regimes(y)
        self.coarse_classifier.fit(X, coarse_labels)
        
        # Train fine classifiers
        for coarse_regime in np.unique(coarse_labels):
            mask = (coarse_labels == coarse_regime)
            self.fine_classifiers[coarse_regime] = LGBMClassifier()
            self.fine_classifiers[coarse_regime].fit(X[mask], y[mask])
```

#### 2.3 Ensemble Architecture Redesign
**Current:** Simple stacking with LightGBM  
**Improvement:** Advanced ensemble techniques

```python
# Multi-level stacking with diverse base models
ENSEMBLE_CONFIG = {
    'level_1_models': [
        'lightgbm',
        'xgboost', 
        'catboost',
        'extratrees',
        'random_forest',
        'gradient_boosting',  # Add more diversity
    ],
    'level_2_models': [
        'lightgbm',       # Main meta-learner
        'xgboost',        # Backup meta-learner
        'neural_net',     # For non-linear meta-features
    ],
    'voting_strategy': 'soft',  # Use probability voting
    'meta_features': [
        'base_predictions',
        'base_probabilities', 
        'prediction_confidence',
        'model_agreement',
        'uncertainty_metrics',
    ]
}

# Implement model diversity penalty
def diversity_penalty(predictions):
    """Penalize ensemble if models are too similar."""
    correlation_matrix = np.corrcoef(predictions.T)
    avg_correlation = correlation_matrix[np.triu_indices_from(correlation_matrix, k=1)].mean()
    return avg_correlation  # Want this < 0.7
```

---

### Phase 3: Training Pipeline Optimization (Week 3-4)

#### 3.1 Advanced Cross-Validation
**Current:** Standard temporal CV  
**Improvement:** Regime-aware purged CV

```python
from src.utils.ml_common.validation.regime_walk_forward_validator import (
    RegimeWalkForwardValidator,
    RegimeValidationConfig
)

# Enhanced CV configuration
cv_config = RegimeValidationConfig(
    n_outer_folds=5,
    n_inner_folds=3,
    embargo_pct=0.05,  # Prevent leakage
    min_train_samples=200,  # Increase for stability
    min_val_samples=50,
    min_regime_samples=20,  # Stricter minimum
    purge_window=10,  # Purge around regime transitions
)

# Add regime transition purging
def purge_regime_transitions(X, y, window=5):
    """Remove samples near regime transitions to prevent leakage."""
    regime_changes = np.where(np.diff(y) != 0)[0]
    
    mask = np.ones(len(y), dtype=bool)
    for change_idx in regime_changes:
        start = max(0, change_idx - window)
        end = min(len(y), change_idx + window + 1)
        mask[start:end] = False
    
    return X[mask], y[mask]
```

#### 3.2 Hyperparameter Optimization
**Current:** Fixed hyperparameters  
**Improvement:** Multi-objective HPO

```python
# Optimize for multiple objectives
OPTIMIZATION_OBJECTIVES = {
    'primary': 'f1_macro',        # Main metric
    'secondary': 'balanced_accuracy',  # Ensure all regimes learned
    'tertiary': 'regime_persistence',  # Temporal stability
}

# Use Optuna for multi-objective optimization
import optuna

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
    }
    
    model = LGBMClassifier(**params, class_weight='balanced')
    
    # Cross-validation with multiple metrics
    cv_results = cross_validate(
        model, X_train, y_train,
        cv=regime_cv,
        scoring=OPTIMIZATION_OBJECTIVES,
        return_train_score=True
    )
    
    return (
        cv_results['test_f1_macro'].mean(),
        cv_results['test_balanced_accuracy'].mean(),
        cv_results['test_regime_persistence'].mean()
    )

# Multi-objective study
study = optuna.create_study(
    directions=['maximize', 'maximize', 'maximize'],
    sampler=optuna.samplers.NSGAIISampler()
)
study.optimize(objective, n_trials=200)
```

#### 3.3 Training Data Quality
**Issues:** Potential data quality problems

**Improvements:**
```python
# Data quality checks
def validate_training_data(X, y, market_data):
    """Comprehensive data quality validation."""
    
    issues = []
    
    # 1. Check for data leakage
    future_features = detect_lookahead_bias(X, market_data)
    if future_features:
        issues.append(f"Lookahead bias in features: {future_features}")
    
    # 2. Check for outliers
    outlier_mask = detect_outliers(X, method='isolation_forest')
    if outlier_mask.sum() > len(X) * 0.05:
        issues.append(f"High outlier rate: {outlier_mask.sum() / len(X):.2%}")
    
    # 3. Check feature quality
    low_variance_features = check_feature_variance(X, threshold=0.01)
    if low_variance_features:
        issues.append(f"Low variance features: {len(low_variance_features)}")
    
    # 4. Check regime label quality
    regime_quality = check_regime_consistency(y, market_data)
    if not regime_quality['is_valid']:
        issues.append(f"Regime quality issues: {regime_quality['issues']}")
    
    # 5. Check temporal consistency
    if not check_temporal_order(market_data.index):
        issues.append("Temporal order violation detected")
    
    return issues

# Clean data before training
X_clean, y_clean = clean_training_data(X, y, validation_report)
```

---

### Phase 4: Evaluation & Monitoring (Week 4-5)

#### 4.1 Enhanced Evaluation Metrics
**Beyond accuracy/F1:** Add regime-specific metrics

```python
# Custom metrics for regime detection
REGIME_METRICS = {
    'regime_purity': lambda y_true, y_pred: measure_regime_purity(y_true, y_pred),
    'transition_accuracy': lambda y_true, y_pred: accuracy_at_transitions(y_true, y_pred),
    'persistence_score': lambda y_true, y_pred: measure_persistence(y_pred),
    'economic_alignment': lambda y_true, y_pred: economic_performance_alignment(y_true, y_pred, returns),
    'regime_coverage': lambda y_true, y_pred: measure_regime_coverage(y_pred),
}

def economic_performance_alignment(y_true, y_pred, returns):
    """Check if predicted regimes align with actual market performance."""
    
    alignment_scores = []
    for regime in np.unique(y_true):
        true_mask = (y_true == regime)
        pred_mask = (y_pred == regime)
        
        true_returns = returns[true_mask]
        pred_returns = returns[pred_mask]
        
        # Compare distributions
        from scipy.stats import ks_2samp
        stat, pval = ks_2samp(true_returns, pred_returns)
        alignment_scores.append(1 - stat)  # Higher is better
    
    return np.mean(alignment_scores)
```

#### 4.2 Model Monitoring Dashboard
**Create real-time monitoring:**

```python
# Monitoring metrics to track
MONITORING_METRICS = {
    'model_performance': [
        'accuracy',
        'f1_macro',
        'f1_per_regime',
        'confusion_matrix',
    ],
    'prediction_quality': [
        'average_confidence',
        'prediction_distribution',
        'regime_transition_rate',
        'uncertainty_metrics',
    ],
    'data_drift': [
        'feature_distribution_shift',
        'regime_distribution_shift',
        'concept_drift_score',
    ],
    'operational': [
        'prediction_latency',
        'memory_usage',
        'error_rate',
    ]
}

# Alert conditions
ALERT_THRESHOLDS = {
    'accuracy_drop': 0.10,  # Alert if accuracy drops >10%
    'high_uncertainty': 0.40,  # Alert if avg confidence <60%
    'regime_imbalance': 0.80,  # Alert if one regime >80%
    'feature_drift': 0.30,  # Alert if feature dist shifts >30%
}
```

---

## 📈 Success Metrics & Timeline

### Phase 1 (Week 1) - Emergency Fixes
**Target Metrics:**
- [ ] Ensemble accuracy: 0.00 → **>0.60** ✅
- [ ] Worst regime F1: 0.00 → **>0.30** ✅
- [ ] Average F1: 0.13 → **>0.45** ✅
- [ ] Number of valid regimes: 6 → **4-5** ✅

### Phase 2 (Week 2-3) - Architecture Improvements
**Target Metrics:**
- [ ] Ensemble accuracy: **>0.70** ✅
- [ ] Worst regime F1: **>0.45** ✅
- [ ] Average F1: **>0.60** ✅
- [ ] All regimes economically meaningful ✅

### Phase 3 (Week 3-4) - Pipeline Optimization
**Target Metrics:**
- [ ] Ensemble accuracy: **>0.75** ✅
- [ ] Worst regime F1: **>0.55** ✅
- [ ] Average F1: **>0.65** ✅
- [ ] Training time: <2 hours ✅

### Phase 4 (Week 4-5) - Production Readiness
**Target Metrics:**
- [ ] Ensemble accuracy: **>0.78** ✅
- [ ] Worst regime F1: **>0.60** ✅
- [ ] Average F1: **>0.70** ✅
- [ ] Prediction latency: <100ms ✅
- [ ] Monitoring dashboard live ✅

---

## 🔧 Implementation Checklist

### Immediate Actions (This Week)
- [ ] **CRITICAL:** Debug ensemble training pipeline
  - [ ] Add extensive logging in `regime_ensemble_training.py:587-3922`
  - [ ] Verify calibration is applied
  - [ ] Check feature alignment
  - [ ] Test with subset of data first

- [ ] **CRITICAL:** Implement regime validation
  - [ ] Add `validate_regime_quality()` function
  - [ ] Filter regimes with <50 samples or extreme drawdowns
  - [ ] Re-run regime clustering with validation

- [ ] **HIGH:** Implement adaptive class balancing
  - [ ] Remove/merge regimes 0, 3
  - [ ] Apply adaptive SMOTE to remaining regimes
  - [ ] Add sample weights to model training

### Short-term Actions (Week 2-3)
- [ ] Enhance feature engineering
  - [ ] Add regime interaction features
  - [ ] Compute regime-specific discriminative features
  - [ ] Feature selection on new features

- [ ] Try alternative model architectures
  - [ ] Test hierarchical regime detector
  - [ ] Experiment with LSTM/CNN for temporal patterns
  - [ ] Compare with current tree-based models

- [ ] Improve ensemble architecture
  - [ ] Add more diverse base models
  - [ ] Implement multi-level stacking
  - [ ] Test different meta-learners

### Medium-term Actions (Week 3-4)
- [ ] Optimize hyperparameters
  - [ ] Multi-objective Optuna study
  - [ ] Per-regime optimal configurations
  - [ ] Ensemble meta-learner tuning

- [ ] Enhance cross-validation
  - [ ] Implement purged regime-aware CV
  - [ ] Add transition purging
  - [ ] Increase fold count for stability

- [ ] Data quality improvements
  - [ ] Automated data validation pipeline
  - [ ] Outlier detection and handling
  - [ ] Lookahead bias verification

### Long-term Actions (Week 4-5+)
- [ ] Build monitoring infrastructure
  - [ ] Real-time performance dashboard
  - [ ] Automated alerting system
  - [ ] Data drift detection

- [ ] Comprehensive evaluation
  - [ ] Economic alignment testing
  - [ ] Regime transition analysis
  - [ ] Walk-forward backtesting

- [ ] Documentation and productionization
  - [ ] Model cards for each regime detector
  - [ ] API documentation
  - [ ] Deployment guide

---

## 🎬 Next Steps

### Recommended Starting Point:
1. **Debug ensemble training** (1-2 days)
   - File: `src/training/steps/market_analysis/components/regime_ensemble_training.py`
   - Add logging, verify calibration, test on subset

2. **Implement regime validation** (1 day)
   - Filter invalid regimes (drawdown, sample count)
   - Re-analyze with valid regimes only

3. **Fix class imbalance** (2-3 days)
   - Adaptive SMOTE implementation
   - Sample weights
   - Test on validation set

4. **Quick wins:**
   - Try different meta-learners (XGBoost, CatBoost)
   - Increase min_regime_samples to 50
   - Use focal loss for minority classes

### Weekly Goals:
- **Week 1:** Restore ensemble to >60% accuracy
- **Week 2:** Achieve >70% accuracy with architecture improvements
- **Week 3:** Optimize to >75% accuracy
- **Week 4:** Production-ready system with monitoring

---

## 📚 References

**Key Files to Modify:**
1. `src/training/steps/market_analysis/components/regime_ensemble_training.py` (ensemble training)
2. `src/training/steps/market_analysis/components/regime_models_training.py` (base models)
3. `src/training/steps/market_analysis/rolling_hmm_clustering/clustering.py` (regime discovery)
4. `src/utils/ml_common/validation/regime_walk_forward_validator.py` (validation)

**Documentation:**
- Current reports: `outcomes/regime_*_ETHUSDT_20251111_*.csv`
- Hanging process fixes: `HANGING_PROCESS_FIXES_SUMMARY.md`

---

## 💬 Questions for Discussion

1. **Should we reduce the number of regimes from 6 to 4?**
   - Pros: Better sample balance, easier to learn
   - Cons: Less granular market state detection

2. **Deep learning vs tree-based models?**
   - Trees: Faster, interpretable, current infrastructure
   - DL: Better temporal patterns, needs more data/compute

3. **Hierarchical vs flat classification?**
   - Hierarchical: More robust to imbalance
   - Flat: Simpler, current implementation

4. **What's the minimum acceptable F1 score per regime?**
   - Suggestion: 0.55-0.60 (vs current 0.00-0.25)

---

**Status:** Ready for implementation  
**Last Updated:** November 11, 2025  
**Next Review:** After Phase 1 completion
