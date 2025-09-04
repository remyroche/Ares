# ML Model Training for S/R Detection - Complete Guide

## Overview

The ML model training for S/R detection is a comprehensive process that combines multiple machine learning models to enhance Support/Resistance level detection, quality scoring, and breakout prediction. The system uses **230+ features** from both S/R-specific characteristics and step06 feature engineering.

## Training Architecture

### 1. **Main Training Entry Point**

```python
async def train_models(
    self,
    market_data: pd.DataFrame,
    sr_levels: List[Dict[str, Any]],
    historical_performance: Optional[Dict[str, Any]] = None
) -> bool:
```

**Purpose**: Orchestrates the training of all ML models
**Input**: Market data, S/R levels, historical performance data
**Output**: Boolean indicating training success

### 2. **Three Main ML Models Trained**

#### **A. S/R Quality Prediction Model**
- **Type**: Gradient Boosting Regressor (with regularization)
- **Purpose**: Predicts the quality/strength of S/R levels
- **Target**: Quality score (0-1) based on level characteristics

#### **B. Breakout Prediction Model**
- **Type**: Random Forest Classifier
- **Purpose**: Predicts probability of S/R level breakouts
- **Target**: Binary classification (breakout/no breakout)

#### **C. Market Regime Classification Model**
- **Type**: Uses step03's LGBM model (not trained separately)
- **Purpose**: Classifies market regime (trending/ranging/transitional)
- **Target**: Regime classification

## Training Process Flow

### **Step 1: Data Preparation**

```python
async def _prepare_training_data(
    self,
    market_data: pd.DataFrame,
    sr_levels: List[Dict[str, Any]],
    historical_performance: Optional[Dict[str, Any]]
) -> Optional[MLFeatureSet]:
```

#### **Feature Extraction Process:**

1. **S/R Specific Features (31 features)**:
   ```python
   # Basic S/R features (9)
   - touch_count: Number of times level was tested
   - strength: Calculated level strength score
   - age_bars: Age of level in bars
   - avg_bounce_ratio: Average bounce strength
   - max_bounce_ratio: Maximum bounce strength
   - volume_confirmation_score: Volume confirmation strength
   - consistency_score: Level consistency over time
   - failure_count: Number of times level failed
   - proximity_to_level: Current proximity to level
   
   # Technical indicators (15)
   - rsi_14: Relative Strength Index
   - macd_line: MACD line value
   - macd_signal: MACD signal line
   - bollinger_position: Position within Bollinger Bands
   - atr_14: Average True Range
   - volume_ratio: Current vs average volume
   - price_momentum: Price momentum over periods
   - stoch_k, stoch_d: Stochastic oscillators
   - williams_r: Williams %R oscillator
   - cci: Commodity Channel Index
   - adx: Average Directional Index
   - obv: On-Balance Volume
   - doji_pattern, hammer_pattern: Candlestick patterns
   - volatility_proxy: Volatility proxy (simplified VIX)
   
   # Advanced features (6)
   - level_density: Density of nearby S/R levels
   - confluence_score: Confluence with other levels
   - time_since_touch: Time since last touch
   - volume_at_touch: Volume during last touch
   - price_action_score: Price action pattern score
   - microstructure_score: Market microstructure score
   ```

2. **Step06 Features (200+ features)**:
   ```python
   # Complete step06 feature integration
   step06_features = await self._extract_step06_features(market_data)
   
   # Categories include:
   - price_features (25+): OHLCV transformations, returns, volatility
   - volume_features (25+): Volume profiles, ratios, momentum
   - microstructure_features (25+): Order flow, bid-ask spreads
   - technical_features (25+): Moving averages, oscillators
   - regime_features (25+): Market regime classifications
   - wavelet_features (25+): Multi-resolution analysis
   - cross_timeframe_features (25+): Multi-timeframe analysis
   - interaction_features (25+): Feature combinations
   ```

3. **Feature Combination**:
   ```python
   # Combine S/R features with step06 features
   combined_features = sr_features + step06_features  # 230+ total features
   ```

#### **Target Creation**:

```python
async def _create_target_for_level(
    self,
    level: Dict[str, Any],
    historical_performance: Optional[Dict[str, Any]]
) -> float:
```

**Target Calculation Logic**:
```python
# Use historical performance if available
if historical_performance and level.get('id') in historical_performance:
    return historical_performance[level['id']]['quality_score']

# Create target based on level characteristics
target = 0.0
target += strength * 0.3                    # 30% weight
target += touch_score * 0.2                 # 20% weight  
target += bounce_score * 0.2                # 20% weight
target += volume_score * 0.15               # 15% weight
target += consistency_score * 0.15          # 15% weight

return min(max(target, 0.0), 1.0)  # Clamp to [0, 1]
```

### **Step 2: Advanced Feature Selection**

```python
# Advanced feature selection with S/R prioritization
if len(X) > 50:  # Need sufficient samples
    # Step 1: Random Forest feature importance
    rf_importance = rf_selector.feature_importances_
    
    # Step 2: Permutation importance
    perm_scores = permutation_importance(rf_selector, X, y, n_repeats=10)
    
    # Step 3: Correlation analysis
    correlation_scores = self._calculate_feature_correlations(X, y)
    
    # Step 4: SHAP analysis
    shap_scores = await self._calculate_shap_importance(rf_selector, X, feature_names)
    
    # Step 5: Combined scoring
    combined_scores = (
        rf_importance * 0.3 +      # 30% Random Forest
        perm_scores * 0.3 +        # 30% Permutation
        correlation_scores * 0.2 + # 20% Correlation
        shap_scores * 0.2          # 20% SHAP
    )
    
    # Step 6: S/R Prioritized selection
    top_features = self._select_top_features_with_sr_priority(
        combined_scores, feature_names, top_k=50
    )
    # 60% S/R features, 40% step06 features
```

### **Step 3: Model Training**

#### **A. S/R Quality Model Training**

```python
async def _train_sr_quality_model(self, training_data: MLFeatureSet) -> None:
    # Create Gradient Boosting model with regularization
    self.sr_quality_model = GradientBoostingRegressor(
        n_estimators=200,           # More trees for better performance
        max_depth=4,                # Reduced depth to prevent overfitting
        learning_rate=0.05,         # Lower learning rate for stability
        subsample=0.8,              # Bootstrap sampling
        max_features='sqrt',        # Feature subsampling
        min_samples_split=10,       # Prevent overfitting on small splits
        min_samples_leaf=5,         # Prevent overfitting on small leaves
        validation_fraction=0.2,    # Early stopping validation
        n_iter_no_change=10,        # Early stopping patience
        random_state=42
    )
    
    # Train the model
    self.sr_quality_model.fit(X_scaled, y)
```

#### **B. Breakout Prediction Model Training**

```python
async def _train_breakout_prediction_model(self, training_data: MLFeatureSet) -> None:
    # Create Random Forest model
    self.breakout_prediction_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=6,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42
    )
    
    # Train the model
    self.breakout_prediction_model.fit(X_scaled, y)
```

#### **C. Regime Classification Model**

```python
async def _train_regime_classification_model(self, market_data: pd.DataFrame) -> None:
    # Use step03's existing LGBM model instead of training new one
    from src.training.steps.vectorized_advanced_feature_engineering import (
        VectorizedAdvancedFeatureEngineeringRefactored
    )
    
    self.step03_engineer = VectorizedAdvancedFeatureEngineeringRefactored()
    # No training needed - uses existing step03 model
```

### **Step 4: Model Validation and Performance Tracking**

```python
# Cross-validation for model evaluation
cv_scores = cross_val_score(
    self.sr_quality_model, X_scaled, y, 
    cv=5, scoring='neg_mean_squared_error'
)

# Store performance metrics
self.model_performance = {
    "sr_quality": {
        "accuracy": cv_scores.mean(),
        "last_update": datetime.now()
    },
    "breakout_prediction": {
        "accuracy": breakout_accuracy,
        "last_update": datetime.now()
    },
    "regime_classification": {
        "accuracy": regime_accuracy,
        "last_update": datetime.now()
    }
}
```

## Training Data Requirements

### **Minimum Data Requirements**:
- **Samples**: At least 50 S/R levels for robust training
- **Features**: 230+ features (31 S/R + 200+ step06)
- **Market Data**: Sufficient historical data for feature calculation
- **Historical Performance**: Optional but recommended for better targets

### **Data Quality Checks**:
```python
# Validate training data
if len(features) < 50:
    self.logger.warning("Insufficient training data")
    return False

if len(feature_names) < 100:
    self.logger.warning("Insufficient features")
    return False
```

## Model Performance Monitoring

### **Performance Metrics Tracked**:
1. **S/R Quality Model**:
   - Mean Squared Error (MSE)
   - R² Score
   - Feature Importance

2. **Breakout Prediction Model**:
   - Accuracy
   - Precision
   - Recall
   - F1-Score

3. **Regime Classification Model**:
   - Accuracy
   - Confusion Matrix
   - Classification Report

### **Model Retraining**:
```python
# Retrain models if performance degrades
if self._should_retrain_model(model_name, current_performance):
    await self._retrain_model_if_needed(model_name, X, y)
```

## Training Configuration

### **YAML Configuration**:
```yaml
ml_enhancement:
  models:
    sr_quality_model:
      type: "gradient_boosting"
      parameters:
        n_estimators: 200
        max_depth: 4
        learning_rate: 0.05
        subsample: 0.8
        max_features: "sqrt"
        min_samples_split: 10
        min_samples_leaf: 5
        validation_fraction: 0.2
        n_iter_no_change: 10
    
    breakout_prediction_model:
      type: "random_forest"
      parameters:
        n_estimators: 100
        max_depth: 6
        min_samples_split: 10
        min_samples_leaf: 5
  
  training:
    min_samples: 50
    cv_folds: 5
    retrain_frequency: 100  # Retrain every 100 new samples
    feature_selection:
      top_k: 50
      sr_priority_ratio: 0.6  # 60% S/R features
```

## Training Output and Logging

### **Comprehensive Logging**:
```python
self.logger.info("🤖 Starting ML model training...")
self.logger.info(f"📊 Training data prepared: {len(features)} samples, {len(feature_names)} features")
self.logger.info(f"   - S/R specific features: {len(sr_features)}")
self.logger.info(f"   - Step06 features: {len(step06_features)}")

# Feature selection logging
self.logger.info("🔍 Comprehensive Feature Analysis Results:")
self.logger.info(f"📊 Total features analyzed: {len(combined_scores)}")
self.logger.info(f"🎯 Selected features: {len(selected_features)}")

# Top features logging
self.logger.info("🏆 Top 25 Most Important Features:")
for i, (feature, score) in enumerate(sorted_features[:25]):
    feature_type = "🎯 S/R" if is_sr_feature else "📊 STEP06"
    self.logger.info(f"  {i+1:2d}. {feature:<30} {score:.4f} {feature_type} {status}")

self.logger.info("✅ ML model training completed")
```

## Key Benefits of This Training Approach

1. **Comprehensive Feature Set**: 230+ features provide complete market analysis
2. **S/R Prioritization**: Ensures S/R-specific features are not overlooked
3. **Advanced Feature Selection**: Multiple methods ensure robust feature selection
4. **Proper Regularization**: Prevents overfitting with early stopping and subsampling
5. **Performance Monitoring**: Continuous tracking of model performance
6. **Step06 Integration**: Leverages existing feature engineering infrastructure
7. **Flexible Configuration**: YAML-based configuration for easy tuning

This training approach creates a robust, comprehensive ML system for S/R detection that combines the best of traditional technical analysis with modern machine learning techniques.