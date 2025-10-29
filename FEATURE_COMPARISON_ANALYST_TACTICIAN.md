# Feature Comparison: Analyst & Tactician Models
## Training vs Signal Generation

### Overview
This document compares the features used during training versus the features used during signal generation for both Analyst and Tactician models.

---

## 1. ANALYST MODELS

### 1.1 Training Features
**Location:** `src/training/steps/models_training/core/model_trainer.py`
**Method:** `_engineer_analyst_features()` (lines 397-425)

#### Base Features (from X_train):
- Market data features from training DataFrame
- Features provided by the training pipeline (exact features depend on feature generation pipeline)

#### Engineered Features (added during training):
1. **Regime-based Features:**
   - `regime_strength`: Absolute value of regime probability
   - `regime_confidence`: Confidence measure based on regime probability (>0.5 or 1-probability)

2. **Market Condition Features:**
   - `volume_price_trend`: Volume multiplied by price percentage change
   - `volume_momentum`: Ratio of 5-period to 20-period volume moving averages

3. **Volatility Features:**
   - `volatility_5d`: 5-period rolling standard deviation of close prices
   - `volatility_20d`: 20-period rolling standard deviation of close prices
   - `volatility_ratio`: Ratio of 5d to 20d volatility

#### Feature Selection:
- Feature selection may be applied via `_select_features()` method (lines 448-476 in `base_trainer.py`)
- Methods: correlation, variance, mutual information
- Max features controlled by config: `config.max_features`

#### Training Data Preparation:
- Data comes from `X_train` prepared by the training pipeline
- Features are engineered AFTER preprocessing
- Final feature set: Original features + engineered features

---

### 1.2 Signal Generation Features
**Location:** `src/trading/signal_generation/signal_pipeline.py`
**Method:** `_run_analyst_base_models()` (lines 959-1087) and `_run_analyst_ensemble()` (lines 1093-1224)

#### Analyst Base Models Input:
1. **Market Data Features:**
   - All numeric columns from `market_data` DataFrame
   - Extracted from last row: `numeric_data.iloc[-1].values`

2. **Regime Probabilities:**
   - Array of regime probabilities for all `RegimeType` values
   - Format: `[regime_prob[RT1], regime_prob[RT2], ...]`

**Combined Input Format:**
```python
combined_features = [market_features, regime_probs_values]
```

#### Analyst Ensemble Model Input:
1. **Market Features:** Same as base models
2. **Regime Probabilities:** Same as base models
3. **Base Model Outputs:**
   - Array of confidences from all analyst base models
   - Format: `[base1_confidence, base2_confidence, ...]`

**Combined Input Format:**
```python
ensemble_input = [market_features, regime_probs_values, base_predictions_array]
```

#### Fallback Behavior:
- If model fails with combined features, falls back to `market_data` only
- Model may handle regime probabilities internally

---

### 1.3 Comparison: Analyst Training vs Signal Generation

| Aspect | Training | Signal Generation |
|--------|----------|-------------------|
| **Market Data** | All features from X_train (from feature generation pipeline) | All numeric columns from market_data DataFrame |
| **Regime Data** | Added as engineered features (regime_strength, regime_confidence) | Added as raw probabilities array |
| **Engineered Features** | Added during training (volume_price_trend, volume_momentum, volatility features) | NOT added - uses raw market data |
| **Feature Selection** | May apply correlation/variance/mutual_info selection | No feature selection |
| **Base Model Outputs** | N/A (base models are being trained) | Included as features for ensemble |
| **Input Format** | DataFrame with named columns | NumPy array (flattened) |

#### Key Differences:
1. **Engineered Features:** Training adds engineered features (volume_price_trend, volume_momentum, volatility_ratio), but signal generation does NOT add these. This is a **MISMATCH**.
2. **Regime Features:** Training uses engineered regime features (regime_strength, regime_confidence), signal generation uses raw probabilities.
3. **Feature Format:** Training uses DataFrame with named columns, signal generation flattens to NumPy arrays.

---

## 2. TACTICIAN MODELS

### 2.1 Training Features
**Location:** `src/training/steps/models_training/core/model_trainer.py`
**Method:** `_engineer_tactician_features()` (lines 427-452)

#### Base Features (from X_train):
- Market data features from training DataFrame
- Analyst ensemble outputs (when available)
- Features provided by the training pipeline

#### Engineered Features (added during training):
1. **Timing Features:**
   - `hour`: Hour of day (from timestamp)
   - `day_of_week`: Day of week (0=Monday, 6=Sunday)
   - `is_weekend`: Binary flag (1 if Saturday/Sunday)

2. **Analyst Signal Features:**
   - `analyst_signal_strength`: Mean of all analyst-related columns
   - `analyst_signal_consistency`: Standard deviation of analyst-related columns

3. **Risk Features:**
   - `price_momentum`: 5-period price percentage change
   - `risk_adjusted_return`: Price momentum divided by 20-period rolling std

#### Feature Selection:
- Same as Analyst (correlation, variance, mutual information)
- Max features controlled by config

#### Training Data Preparation:
- Data comes from `X_train` + analyst predictions (if available)
- Features are engineered AFTER preprocessing
- Final feature set: Original features + engineered features

---

### 2.2 Signal Generation Features
**Location:** `src/trading/signal_generation/signal_pipeline.py`
**Method:** `_run_tactician_base_models()` (lines 1226-1339)

#### Tactician Base Models Input:
1. **Market Data Features:**
   - All numeric columns from `market_data` DataFrame
   - Extracted from last row: `numeric_data.iloc[-1].values`

2. **Regime Probabilities:**
   - Array of regime probabilities for all `RegimeType` values

3. **Analyst Ensemble Outputs:**
   - `analyst_confidence`: Confidence from analyst ensemble
   - `market_health_score`: Market health score from analyst
   - `regime_adjusted_confidence`: Regime-adjusted confidence from analyst

**Combined Input Format:**
```python
combined_features = [market_features, regime_probs_values, analyst_ensemble_inputs]
```

#### Tactician Ensemble Model Input:
1. **Market Features:** Same as base models
2. **Regime Probabilities:** Same as base models
3. **Analyst Ensemble Outputs:** Same as base models
4. **Base Model Outputs:**
   - Array of confidences from all tactician base models

**Combined Input Format:**
```python
ensemble_input = [market_features, regime_probs_values, analyst_outputs, base_predictions_array]
```

---

### 2.3 Comparison: Tactician Training vs Signal Generation

| Aspect | Training | Signal Generation |
|--------|----------|-------------------|
| **Market Data** | All features from X_train | All numeric columns from market_data DataFrame |
| **Regime Data** | Not explicitly shown (may be in X_train) | Raw probabilities array |
| **Analyst Outputs** | Provided via analyst predictions in X_train | Analyst ensemble outputs (3 values: confidence, health, adjusted_confidence) |
| **Timing Features** | Added during training (hour, day_of_week, is_weekend) | **NOT added** |
| **Risk Features** | Added during training (price_momentum, risk_adjusted_return) | **NOT added** |
| **Analyst Signal Features** | Added during training (signal_strength, signal_consistency) | **NOT added** (uses raw analyst outputs instead) |
| **Base Model Outputs** | N/A (base models are being trained) | Included as features for ensemble |
| **Input Format** | DataFrame with named columns | NumPy array (flattened) |

#### Key Differences:
1. **Timing Features:** Training adds timing features (hour, day_of_week, is_weekend), but signal generation does NOT. This is a **MISMATCH**.
2. **Risk Features:** Training adds risk features (price_momentum, risk_adjusted_return), but signal generation does NOT. This is a **MISMATCH**.
3. **Analyst Signal Features:** Training adds engineered analyst signal features, signal generation uses raw analyst outputs (different format).
4. **Feature Format:** Training uses DataFrame with named columns, signal generation flattens to NumPy arrays.

---

## 3. SUMMARY OF MISMATCHES

### Critical Issues:

#### 3.1 Analyst Models
1. **Missing Engineered Features in Signal Generation:**
   - `volume_price_trend`
   - `volume_momentum`
   - `volatility_5d`, `volatility_20d`, `volatility_ratio`
   - `regime_strength`, `regime_confidence` (uses raw probabilities instead)

#### 3.2 Tactician Models
1. **Missing Engineered Features in Signal Generation:**
   - `hour`, `day_of_week`, `is_weekend` (timing features)
   - `price_momentum`, `risk_adjusted_return` (risk features)
   - `analyst_signal_strength`, `analyst_signal_consistency` (uses raw analyst outputs instead)

#### 3.3 Data Format Mismatches
- **Training:** DataFrame with named columns, engineered features added
- **Signal Generation:** NumPy arrays (flattened), raw features only

---

## 4. RECOMMENDATIONS

### High Priority:
1. **Add engineered features to signal generation pipeline:**
   - Extract or recompute all engineered features that are added during training
   - Ensure feature order matches training order

2. **Standardize feature engineering:**
   - Create a shared feature engineering module
   - Use the same feature engineering logic in both training and inference

3. **Feature name mapping:**
   - Map DataFrame column names to array positions consistently
   - Ensure signal generation uses the same feature set as training

### Medium Priority:
4. **Feature validation:**
   - Validate that signal generation features match training features
   - Add logging to compare feature sets

5. **Documentation:**
   - Document exact feature list used in training
   - Document exact feature list used in signal generation

### Implementation Suggestions:
- Create a `FeatureEngineer` class that can be used in both training and inference
- Store feature metadata (names, order, transformations) with trained models
- Add feature validation/comparison utilities

---

## 5. CODE REFERENCES

### Training Code:
- Analyst feature engineering: `src/training/steps/models_training/core/model_trainer.py:397-425`
- Tactician feature engineering: `src/training/steps/models_training/core/model_trainer.py:427-452`
- Feature selection: `src/training/steps/models_training/core/base_trainer.py:448-476`

### Signal Generation Code:
- Analyst base models: `src/trading/signal_generation/signal_pipeline.py:959-1087`
- Analyst ensemble: `src/trading/signal_generation/signal_pipeline.py:1093-1224`
- Tactician base models: `src/trading/signal_generation/signal_pipeline.py:1226-1339`
- Tactician ensemble: `src/trading/signal_generation/signal_pipeline.py` (similar structure)

---

## 6. DETAILED FEATURE LISTS

### 6.1 Analyst Training Features (Complete)
```
Base Features (from X_train):
- [All features from feature generation pipeline]

Engineered Features:
- regime_strength
- regime_confidence
- volume_price_trend
- volume_momentum
- volatility_5d
- volatility_20d
- volatility_ratio
```

### 6.2 Analyst Signal Generation Features (Complete)
```
Input Features:
- [All numeric columns from market_data DataFrame]
- [Regime probabilities array]
- [Base model outputs array] (for ensemble only)
```

### 6.3 Tactician Training Features (Complete)
```
Base Features (from X_train):
- [All features from feature generation pipeline]
- [Analyst predictions/features] (when available)

Engineered Features:
- hour
- day_of_week
- is_weekend
- analyst_signal_strength
- analyst_signal_consistency
- price_momentum
- risk_adjusted_return
```

### 6.4 Tactician Signal Generation Features (Complete)
```
Input Features:
- [All numeric columns from market_data DataFrame]
- [Regime probabilities array]
- [Analyst ensemble outputs: confidence, market_health_score, regime_adjusted_confidence]
- [Base model outputs array] (for ensemble only)
```

---

## Conclusion

**There are significant mismatches between training and signal generation features**, particularly:
1. Missing engineered features in signal generation (volume, volatility, timing, risk features)
2. Different format for regime features (engineered vs raw probabilities)
3. Different input format (DataFrame vs NumPy arrays)

These mismatches could lead to:
- Model performance degradation in production
- Unexpected predictions
- Feature importance discrepancies
- Difficult debugging

**Immediate action required:** Align feature engineering between training and signal generation.
