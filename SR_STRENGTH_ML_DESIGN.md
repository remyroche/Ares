# Data-Driven SR Strength Calculation - Design Document

## Current State (Hardcoded)

```python
strength = (
    base_strength +
    touch_boost * 0.1 +           # HARDCODED
    volume_boost * 0.2 +          # HARDCODED
    consistency_boost * 0.2 +     # HARDCODED
    confluence_boost * 0.1 +      # HARDCODED
    special_boost - 
    failure_penalty * 0.2         # HARDCODED
)
```

**Problem**: Weights are arbitrary, not optimized for actual SR performance.

---

## Proposed Solutions

### Option 1: HPO-Optimized Weights (Recommended - Simpler)

**Concept**: Treat strength formula weights as hyperparameters to optimize.

#### Implementation

```python
@dataclass
class StrengthWeights:
    """Optimizable weights for strength calculation."""
    touch_weight: float = 0.1          # Range: [0.05, 0.3]
    volume_weight: float = 0.2         # Range: [0.1, 0.4]
    consistency_weight: float = 0.2    # Range: [0.1, 0.4]
    confluence_weight: float = 0.1     # Range: [0.05, 0.2]
    failure_penalty_weight: float = 0.2  # Range: [0.1, 0.5]
    pivot_boost: float = 0.1           # Range: [0.05, 0.2]
    psychological_boost: float = 0.05  # Range: [0.02, 0.1]
    hvn_boost: float = 0.1             # Range: [0.05, 0.2]
```

#### HPO Search Space

```python
strength_weight_space = {
    'touch_weight': {
        'type': 'float',
        'low': 0.05,
        'high': 0.3,
        'step': 0.05
    },
    'volume_weight': {
        'type': 'float',
        'low': 0.1,
        'high': 0.4,
        'step': 0.05
    },
    'consistency_weight': {
        'type': 'float',
        'low': 0.1,
        'high': 0.4,
        'step': 0.05
    },
    'confluence_weight': {
        'type': 'float',
        'low': 0.05,
        'high': 0.2,
        'step': 0.025
    },
    'failure_penalty_weight': {
        'type': 'float',
        'low': 0.1,
        'high': 0.5,
        'step': 0.05
    },
    'pivot_boost': {
        'type': 'float',
        'low': 0.05,
        'high': 0.2,
        'step': 0.025
    },
    'psychological_boost': {
        'type': 'float',
        'low': 0.02,
        'high': 0.1,
        'step': 0.01
    },
    'hvn_boost': {
        'type': 'float',
        'low': 0.05,
        'high': 0.2,
        'step': 0.025
    }
}
```

#### Objective Function

```python
def evaluate_strength_weights(weights: StrengthWeights, 
                             historical_levels: List[SRLevel],
                             forward_performance: List[float]) -> float:
    """
    Evaluate strength weights by correlation with actual SR performance.
    
    Args:
        weights: Candidate weights to test
        historical_levels: Past SR levels with all features
        forward_performance: Actual performance (bounce %, hold time, etc.)
    
    Returns:
        Score (higher = better weights)
    """
    # Recalculate strength with candidate weights
    predicted_strengths = []
    for level in historical_levels:
        strength = calculate_strength_with_weights(level, weights)
        predicted_strengths.append(strength)
    
    # Correlation with actual performance
    correlation = np.corrcoef(predicted_strengths, forward_performance)[0, 1]
    
    # Also check ranking accuracy (do stronger levels actually perform better?)
    ranking_accuracy = spearmanr(predicted_strengths, forward_performance)[0]
    
    # Combined score
    score = 0.6 * correlation + 0.4 * ranking_accuracy
    
    return score
```

#### Integration with Existing HPO

Add to `SRParameterOptimizationStep`:

```python
class EnhancedSRConfig:
    # Existing params...
    enable_strength_weight_optimization: bool = True
    strength_weight_space: Dict[str, Any] = field(default_factory=lambda: strength_weight_space)
```

**Benefits**:
- ✅ Simple to implement (extends existing HPO)
- ✅ Fast optimization (8 params, ~50 trials)
- ✅ Interpretable results (see which weights matter)
- ✅ No training data needed (uses detection results directly)

**Drawbacks**:
- Linear combination only
- Assumes current formula structure is correct

---

### Option 2: End-to-End ML Model (More Complex)

**Concept**: Replace entire strength formula with ML model.

#### Architecture

```python
class SRStrengthModel:
    """ML model to predict SR level strength."""
    
    def __init__(self):
        self.model = lgb.LGBMRegressor(
            objective='regression',
            metric='rmse',
            n_estimators=100,
            max_depth=5,
            learning_rate=0.05
        )
    
    def train(self, X: pd.DataFrame, y: pd.Series):
        """
        Train strength prediction model.
        
        Features (X):
        - touch_count
        - avg_bounce_ratio
        - max_bounce_ratio
        - volume_confirmation_score
        - consistency_score
        - confluence_score
        - failure_count
        - pivot_level (binary)
        - psychological_level (binary)
        - volume_at_level
        - age_bars
        - time_since_last_touch
        - ... (all SRLevel fields)
        
        Target (y):
        - Actual forward performance (quality_score)
        """
        self.model.fit(X, y)
    
    def predict_strength(self, level: SRLevel) -> float:
        """Predict strength score for a level."""
        features = self._extract_features(level)
        strength = self.model.predict(features)[0]
        return np.clip(strength, 0.0, 1.0)
```

#### Data Collection

```python
async def collect_strength_training_data(
    symbol: str,
    exchange: str,
    timeframe: str,
    start_date: str,
    end_date: str
) -> pd.DataFrame:
    """
    Collect historical SR levels with forward performance.
    
    Process:
    1. Walk forward through time
    2. At each date:
       - Detect SR levels
       - Calculate current strength (using any formula)
       - Measure forward performance (next 5-10 days)
    3. Create training samples
    
    Returns:
        DataFrame with [level features, forward_performance]
    """
    # Similar to existing SRQualityDataCollector
    # But focus on strength prediction instead of quality
```

#### Integration

```python
class EnhancedSRDetector:
    def __init__(self, strength_model_path: Optional[str] = None):
        # Existing init...
        
        if strength_model_path and Path(strength_model_path).exists():
            self.strength_model = SRStrengthModel.load(strength_model_path)
            self.use_ml_strength = True
        else:
            self.strength_model = None
            self.use_ml_strength = False
    
    def _calculate_enhanced_strength(self, level: SRLevel) -> float:
        if self.use_ml_strength:
            # ML-based strength
            return self.strength_model.predict_strength(level)
        else:
            # Formula-based strength (current approach)
            return self._calculate_formula_strength(level)
```

**Benefits**:
- ✅ Learns non-linear relationships
- ✅ No hardcoded assumptions
- ✅ Can discover new important features
- ✅ Automatically adapts to market changes

**Drawbacks**:
- Requires significant training data
- Less interpretable (black box)
- More complex to maintain
- Potential overfitting risk

---

## Recommended Implementation Plan

### Phase 1: HPO-Optimized Weights (Week 1)

**Goal**: Optimize existing formula weights via HPO.

1. **Add strength weights to HPO search space**
   - Extend `EnhancedSRConfig` with weight parameters
   - Define search ranges for each weight

2. **Implement objective function**
   - Collect historical SR levels
   - Measure correlation with forward performance
   - Optimize for prediction accuracy

3. **Run optimization**
   - Use hierarchical HPO (coarse → fine → TPE)
   - Save optimized weights to config file
   - Compare hardcoded vs optimized performance

4. **Integration**
   - Load optimized weights from config
   - Fall back to hardcoded if not found
   - Add `--optimize-strength-weights` flag

**Deliverables**:
- `optimized_strength_weights.json` (per symbol/timeframe)
- Performance comparison report
- Updated `EnhancedSRDetector` with configurable weights

### Phase 2: ML Strength Model (Week 2-3)

**Goal**: Replace formula with end-to-end ML model.

1. **Data collection**
   - Extend `SRQualityDataCollector` for strength
   - Collect 6+ months of historical SR levels
   - Label with forward performance

2. **Model training**
   - Train LightGBM regressor
   - 5-fold cross-validation
   - Feature importance analysis (SHAP)

3. **Integration**
   - Add `--use-ml-strength` flag
   - Dual mode: formula or ML
   - Performance monitoring

4. **Evaluation**
   - Compare formula vs ML on test set
   - Analyze feature importance
   - Measure inference speed

**Deliverables**:
- `models/sr_strength_model.lgb`
- Training/validation reports
- SHAP plots for interpretability
- Performance benchmarks

---

## Code Structure

```
src/tactician/sr_levels/
├── enhanced_sr_detection.py          # Main detector (uses strength)
├── strength/                          # NEW: Strength calculation module
│   ├── __init__.py
│   ├── formula_strength.py           # Formula-based (current + optimized weights)
│   ├── ml_strength.py                # ML-based strength model
│   ├── strength_optimizer.py         # HPO for weights
│   └── strength_data_collector.py    # Training data collection
└── ml_quality/                        # Existing quality model
    ├── sr_quality_data_collector.py
    └── sr_quality_model.py
```

---

## Configuration

```yaml
# config/sr_strength_config.yaml
strength_calculation:
  mode: "optimized_formula"  # Options: "hardcoded", "optimized_formula", "ml_model"
  
  # Formula mode (with optimized weights)
  formula:
    weights:
      touch: 0.15              # Optimized via HPO (was 0.1)
      volume: 0.25             # Optimized via HPO (was 0.2)
      consistency: 0.18        # Optimized via HPO (was 0.2)
      confluence: 0.12         # Optimized via HPO (was 0.1)
      failure_penalty: 0.22    # Optimized via HPO (was 0.2)
      pivot_boost: 0.12        # Optimized via HPO (was 0.1)
      psychological_boost: 0.06 # Optimized via HPO (was 0.05)
      hvn_boost: 0.15          # Optimized via HPO (was 0.1)
  
  # ML mode
  ml_model:
    model_path: "models/sr_strength_model.lgb"
    fallback_to_formula: true
    min_confidence: 0.7

  # HPO settings
  optimization:
    enable: true
    n_trials: 50
    optimization_metric: "spearman_correlation"
    validation_split: 0.2
```

---

## Usage Examples

### Run with HPO Weight Optimization

```bash
# Optimize strength weights for ETHUSDT 15m
python scripts/run_sr_workflow.py \
  --symbol ETHUSDT \
  --timeframe 15m \
  --optimize-strength-weights \
  --strength-opt-trials 50
```

### Run with Optimized Weights

```bash
# Use previously optimized weights
python scripts/run_sr_workflow.py \
  --symbol ETHUSDT \
  --timeframe 15m \
  --strength-config config/sr_strength_config.yaml
```

### Run with ML Strength Model

```bash
# Use ML model for strength prediction
python scripts/run_sr_workflow.py \
  --symbol ETHUSDT \
  --timeframe 15m \
  --use-ml-strength \
  --strength-model models/sr_strength_model.lgb
```

---

## Performance Comparison

### Expected Results

| Method | Correlation | Speed | Interpretability | Maintenance |
|--------|------------|-------|------------------|-------------|
| Hardcoded | 0.45-0.55 | Fast | High | Low |
| HPO Weights | 0.60-0.70 | Fast | High | Medium |
| ML Model | 0.70-0.85 | Medium | Low | High |

---

## Next Steps

1. ✅ **Implement Phase 1 (HPO Weights)** - Recommended first step
   - Low risk, high reward
   - Quick to implement
   - Maintains interpretability

2. **Evaluate Results**
   - If Phase 1 gives 10%+ improvement → Deploy
   - If Phase 1 plateaus → Consider Phase 2

3. **Optional: Phase 2 (ML Model)**
   - Only if HPO weights insufficient
   - Requires more data and validation
   - Consider hybrid approach (ensemble)

---

## Risk Mitigation

### For HPO Approach
- ✅ Validate on out-of-sample data
- ✅ Check weight stability across different periods
- ✅ Fall back to hardcoded if optimized performs worse

### For ML Approach
- ✅ Collect sufficient training data (6+ months)
- ✅ Monitor prediction drift over time
- ✅ Keep formula as fallback
- ✅ Regular retraining (monthly)

---

## Questions to Answer

1. **Which features matter most for strength?**
   - HPO will show via weight magnitudes
   - ML will show via SHAP/feature importance

2. **Are current weights directionally correct?**
   - HPO optimization will reveal
   - Large changes = poor initial intuition

3. **Is non-linearity important?**
   - Compare HPO vs ML performance
   - If ML >> HPO, non-linearity matters

4. **Market regime dependency?**
   - Optimize weights per regime
   - Compare trending vs ranging markets

---

**Recommendation**: Start with **Phase 1 (HPO Weights)**. It's:
- Quick to implement (1 week)
- Low risk (interpretable, reversible)
- High value (likely 10-20% improvement)
- Foundation for Phase 2 if needed

