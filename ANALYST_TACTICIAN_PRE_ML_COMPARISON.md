# Analyst vs Tactician Pre-ML Orchestration Comparison

## Overview

Both `analyst_pre_ml_orchestration` and `tactician_pre_ml_orchestration` orchestrate the complete pre-training feature engineering pipeline, but with key differences in **timeframe**, **labeling strategy**, and **optimization focus**.

---

## Common Pipeline Steps

Both orchestrations use the **same three core feature engineering steps** from the pre-training sub-pipeline:

### 1. **Feature Lookback Optimization** (`feature_lookback_optimization`)
- **Purpose**: Optimize feature lookback periods per regime/cluster
- **Analyst**: Uses **60m timeframe**
- **Tactician**: Uses **15m timeframe**
- **Configuration**: Both enable per-cluster optimization

### 2. **Interactive Feature Generation** (`interactive_feature_generation`)
- **Purpose**: Generate interaction, polynomial, and cross-timeframe features
- **Implementation**: Same for both (end-to-end interactive feature generation)
- **Configuration**: Identical approach

### 3. **Final Feature Selection** (`final_feature_selection`)
- **Purpose**: Multi-stage feature selection (120→100→80→60)
- **Implementation**: Same for both
- **Configuration**: Identical approach

---

## Key Differences

### 📊 Timeframe Configuration

| Aspect | Analyst | Tactician |
|--------|---------|-----------|
| **Timeframe** | 60m | 15m |
| **Role** | Strategic IF-to-trade decisions | Tactical entry timing optimization |
| **Data Filtering** | None (uses ALL market data) | None (filtering happens in training step) |

**Location in code:**
```python
# Analyst (analyst_pre_ml_orchestration.py:64)
timeframe: str = "60m"  # ANALYST USES 60m TIMEFRAME

# Tactician (tactician_pre_ml_orchestration.py:139)
timeframe: str = "15m"  # TACTICIAN PRE-ML USES 15m TIMEFRAME
```

---

### 🎯 Labeling Strategy (Step 1)

This is the **critical difference** between the two orchestrations:

#### **Analyst: Multi-Horizon Profit Labeling**
- **Purpose**: Identify profitable targets with 0.5-1% price change horizons
- **Method**: Uses `multi_horizon_profit_labeler` → `VolatilityAwareMultiHorizonLabeler`
- **Target**: Multi-horizon profit opportunities (strategic level)
- **Implementation**: `create_enhanced_analyst_labeler()` with differentiated horizons
- **Focus**: Profitable price movements over longer timeframes

**Code location:**
```python
# analyst_pre_ml_orchestration.py:197
horizon_result = await self.pre_training_pipeline._execute_multi_horizon_profit_labeler(sub_config)
```

**Key characteristics:**
- Volatility-aware labeling with adaptive thresholds
- Multiple horizon targets (0.5%, 0.75%, 1.0% profit targets)
- Quality scoring based on predictability, stability, and balance

#### **Tactician: Entry Timing Labeling**
- **Purpose**: Identify optimal entry points (local maxima/minima)
- **Method**: Uses specialized entry timing labelers with **three strategies**:
  1. **Rule-Based** (`TacticianDifferentiatedLabeler`): Peak detection using `scipy.signal.find_peaks`
  2. **ML-Iterative** (`MLEntryTimingLabeler`): ML models trained on rule-based labels
  3. **ML-Corrected** (`CorrectedMLEntryTimingLabeler`): Peak/bottom detection with ML refinement
- **Target**: Best entry points for position execution (tactical level)
- **Implementation**: Custom entry timing labelers targeting local extrema
- **Focus**: Minimal adverse movement, maximal timing precision

**Code location:**
```python
# tactician_pre_ml_orchestration.py:1527
horizon_result = await self.pre_training_pipeline._execute_multi_horizon_profit_labeler(
    sub_config,
    run_metadata or {},
)
```

**Key characteristics:**
- Targets local maxima/minima within Analyst green light periods
- Scores entries based on:
  - **Risk-reward ratio**: `favorable_move / (adverse_move + ε)`
  - **Timing score**: Earliness of entry within green period
  - **Volatility score**: Stability around entry point
- Quality metrics: entry_quality, labeling_coverage, entry_point_density

**Entry Timing Implementation:**
```python
# tactician_pre_ml_orchestration.py:287-336
def _find_optimal_entries_in_period(self, period_data, regime_assignments):
    """Score potential entries inside a green light period."""
    # Uses scipy.signal.find_peaks to identify local maxima/minima
    peaks, properties = find_peaks(
        scores_array,
        height=self.config.entry_quality_threshold,
        distance=max(1, self.config.min_entry_window_minutes)
    )
```

---

### 🏷️ Per-Regime/Cluster Optimization

| Aspect | Analyst | Tactician |
|--------|---------|-----------|
| **Per-Regime Optimization** | ❌ Disabled (line 68) | ✅ Enabled (line 157) |
| **Per-Cluster Optimization** | ✅ Enabled (line 69) | ✅ Enabled (line 158) |
| **Rationale** | Uses regime probabilities as features instead | Full regime-aware optimization |

**Code locations:**
```python
# Analyst (analyst_pre_ml_orchestration.py:68-69)
enable_per_regime_optimization: bool = False  # Disabled - using regime probabilities as features instead
enable_per_cluster_optimization: bool = True

# Tactician (tactician_pre_ml_orchestration.py:157-158)
enable_per_regime_optimization: bool = True
enable_per_cluster_optimization: bool = True
```

---

## Labeling Strategy Details

### Analyst: Multi-Horizon Profit Labeling
**File**: `src/training/steps/pre_training/multi_horizon_profit_labeler.py`

**Workflow**:
1. Load `VolatilityAwareMultiHorizonLabeler`
2. Create enhanced analyst labeler with differentiated horizons:
   - Short horizon: 0.5% profit target
   - Medium horizon: 0.75% profit target
   - Long horizon: 1.0% profit target
3. Generate labels with confidence scores and eligibility masks
4. Normalize using sigma scaling

**Quality Metrics**:
- `predictability`: How well labels can be predicted
- `stability`: Consistency across regimes
- `balance`: Distribution of positive/negative labels
- `auc_mean`: Average AUC across horizons

### Tactician: Entry Timing Labeling
**File**: `src/training/steps/models_training/tactician_pre_ml_orchestration.py`

**Workflow** (Rule-Based Strategy):
1. Extract Analyst green light periods
2. For each green period, scan for potential entry points
3. Calculate entry quality score for each point:
   ```
   quality_score = risk_reward_ratio * 0.4 + timing_score * 0.3 + volatility_score * 0.3
   ```
4. Use `find_peaks` to identify local maxima in quality scores
5. Return peak indices as optimal entry labels

**Entry Quality Criteria**:
- `max_adverse_movement_pct`: 0.5% (max drawdown allowed)
- `min_favorable_movement_pct`: 0.2% (min profit expected)
- `max_entry_window_minutes`: 60 (time to find entry)

**Alternative Strategies**:
- **ML-Iterative**: Trains ML models on rule-based labels to refine predictions
- **ML-Corrected**: Uses peak/bottom detection with `scipy.signal.argrelextrema` for more accurate extrema identification

---

## Configuration Differences

### Analyst Configuration (`AnalystPreMLConfig`)
```python
@dataclass
class AnalystPreMLConfig:
    symbol: str = "ETHUSDT"
    exchange: str = "binance"
    timeframe: str = "60m"  # Higher timeframe for strategic decisions
    
    enable_per_regime_optimization: bool = False  # Using regime probabilities as features
    enable_per_cluster_optimization: bool = True
    
    output_directory: str = "generated/analyst_pre_ml"
```

### Tactician Configuration (`TacticianPreMLConfig`)
```python
@dataclass
class TacticianPreMLConfig:
    symbol: str = "ETHUSDT"
    exchange: str = "binance"
    timeframe: str = "15m"  # Lower timeframe for tactical timing
    
    # Analyst signal filtering
    analyst_confidence_threshold: float = 0.004  # 0.4% threshold for "green" signals
    require_analyst_signals: bool = True
    
    # Entry labeling configuration
    entry_labeling_strategy: EntryLabelingStrategy = EntryLabelingStrategy.RULE_BASED
    labeling_config: TacticianLabelingConfig = field(default_factory=TacticianLabelingConfig)
    
    enable_per_regime_optimization: bool = True
    enable_per_cluster_optimization: bool = True
    
    output_directory: str = "generated/tactician_pre_ml"
```

---

## Execution Flow Comparison

### Analyst Execution Flow
```
1. Multi-Horizon Profit Labeling (60m)
   └─> VolatilityAwareMultiHorizonLabeler
       └─> Differentiated horizons: 0.5%, 0.75%, 1.0%
       
2. Feature Lookback Optimization (60m)
   └─> Per-cluster optimization
   └─> Optimize lookback periods for 60m features
   
3. Interactive Feature Generation
   └─> Interaction, polynomial, cross-timeframe features
   
4. Final Feature Selection
   └─> Multi-stage selection: 120→100→80→60 features
```

### Tactician Execution Flow
```
1. Entry Timing Labeling (15m)
   └─> TacticianDifferentiatedLabeler (Rule-Based)
       └─> find_peaks() for local maxima/minima
       └─> Quality scoring: risk-reward + timing + volatility
   
2. Feature Lookback Optimization (15m)
   └─> Per-regime + per-cluster optimization
   └─> Optimize lookback periods for 15m features
   
3. Interactive Feature Generation
   └─> Interaction, polynomial, cross-timeframe features
   
4. Final Feature Selection
   └─> Multi-stage selection: 120→100→80→60 features
   
5. Tactician 5m Entry Optimization (Optional)
   └─> ML-based entry refinement on 5m timeframe
```

---

## Summary: What Each Orchestration Targets

### Analyst Pre-ML Orchestration
**Purpose**: Generate features for predicting **profitable trade opportunities**

**Target Definition**:
- Multi-horizon profit targets (0.5-1% price changes)
- Strategic level: "Should I enter a trade?"
- Focus: Identifying profitable market conditions

**Output**:
- Features optimized for profit prediction
- Labels indicating probability of profitable moves
- Timeframe: 60m (strategic perspective)

### Tactician Pre-ML Orchestration
**Purpose**: Generate features for predicting **optimal entry timing**

**Target Definition**:
- Local maxima/minima (best entry points)
- Tactical level: "When should I enter the trade?"
- Focus: Minimizing adverse movement, maximizing entry quality

**Output**:
- Features optimized for entry timing
- Labels indicating optimal entry points (peaks in quality score)
- Timeframe: 15m (tactical perspective) + optional 5m refinement

---

## Verification Checklist

✅ **Both use `feature_lookback_optimization`** with different timeframes (60m vs 15m)
✅ **Both use `interactive_feature_generation`** (same implementation)
✅ **Both use `final_feature_selection`** (same implementation)
✅ **Analyst uses `multi_horizon_profit_labeler`** for 0.5-1% profit targets
✅ **Tactician uses entry timing labelers** for local maxima/minima detection
✅ **Timeframes are correctly differentiated**: 60m (Analyst) vs 15m (Tactician)
✅ **Per-regime optimization**: Disabled for Analyst, Enabled for Tactician

---

## Recommendations

### Current Implementation Status: ✅ CORRECT

The current implementation correctly implements the requirements:

1. ✅ Both orchestrations use the three core feature engineering steps
2. ✅ Different timeframes: 60m (Analyst) vs 15m (Tactician)
3. ✅ Different labeling strategies:
   - Analyst: Multi-horizon profit labeling (0.5-1% targets)
   - Tactician: Entry timing labeling (local maxima/minima via `find_peaks`)
4. ✅ Proper integration with pre-training sub-pipeline

### Potential Enhancements

1. **Tactician ML Strategy**: Consider enabling `EntryLabelingStrategy.ML_CORRECTED` for more accurate peak/bottom detection using `argrelextrema`
   
2. **Analyst Per-Regime Optimization**: If regime probabilities as features don't provide sufficient signal, consider re-enabling per-regime optimization

3. **Documentation**: Add inline comments in both orchestrations clarifying the labeling strategy differences

4. **Testing**: Create integration tests to verify:
   - Analyst labels target profitable moves (0.5-1% horizons)
   - Tactician labels target local extrema (peaks/bottoms)
   - Feature timeframes match expectations (60m vs 15m)

---

## Code References

### Key Files
- **Analyst Orchestration**: `src/training/steps/models_training/analyst_pre_ml_orchestration.py`
- **Tactician Orchestration**: `src/training/steps/models_training/tactician_pre_ml_orchestration.py`
- **Pre-Training Sub-Pipeline**: `src/training/steps/pre_training/sub_pipeline.py`
- **Multi-Horizon Profit Labeler**: `src/training/steps/pre_training/multi_horizon_profit_labeler.py`
- **Entry Timing Labelers**:
  - Rule-based: Lines 204-432 in `tactician_pre_ml_orchestration.py`
  - ML-iterative: `src/training/steps/models_training/ml_based_entry_timing_labeler.py`
  - ML-corrected: `src/training/steps/models_training/corrected_ml_entry_timing_labeler.py`

### Key Line References
- Analyst timeframe config: `analyst_pre_ml_orchestration.py:64`
- Tactician timeframe config: `tactician_pre_ml_orchestration.py:139`
- Analyst labeling call: `analyst_pre_ml_orchestration.py:197`
- Tactician labeling call: `tactician_pre_ml_orchestration.py:1527`
- Peak detection implementation: `tactician_pre_ml_orchestration.py:321-330`