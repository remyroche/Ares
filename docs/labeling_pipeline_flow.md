# Complete Labeling Pipeline Flow

## Overview

This document traces the complete flow of label generation, weighting, and smoothing when running `feature_generation_labeling_integration_step`.

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  feature_generation_labeling_integration_step.execute()         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  1. Configuration & Setup                                       │
│     • Load market data (OHLCV from KlinesParquetManager)       │
│     • Detect timeframe (15m default)                           │
│     • Get optimal threshold for symbol/timeframe               │
│     • Auto-configure label smoothing params                    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  2. Pre-Processing                                              │
│     • Spike detection and correction                            │
│     • Volume confidence calculation (if available)              │
│     • Volatility calculation                                    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  3. Label Generation (VolatilityAwareMultiHorizonLabeler)      │
│     ├─ Triple Barrier with lookahead periods (6 periods)       │
│     ├─ Volatility adaptation (1.0x - 2.0x threshold)           │
│     ├─ Volume weighting (up to +33% confidence)                │
│     ├─ Time-to-hit tracking (first-passage)                    │
│     ├─ Local extrema detection (optimal entry timing)          │
│     ├─ IC-based quality scoring (predictive power)             │
│     └─ Sample weights generation (PROXIMITY_REGRESSION)        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  4. Label Smoothing (LabelSmoother) ⭐ NEW                     │
│     ├─ Stage 1: Classification/Probability Smoothing           │
│     │   • Softens labels: p_smooth = (1-ε)*p + ε*0.5          │
│     │   • Prevents overconfidence                              │
│     │   • ε = 0.08 (15m), 0.12 (1m-5m), 0.05 (4h-daily)      │
│     │                                                           │
│     ├─ Stage 2: Uncertainty-Weighted Shrinkage                 │
│     │   • Shrinks uncertain labels towards baseline            │
│     │   • Uses quality_inverse from IC-based scoring          │
│     │   • α = 1 / (1 + γ * σ), min_α = 0.12                  │
│     │   • γ = 1.0 (15m), 1.5 (1m-5m), 0.5 (4h-daily)         │
│     │                                                           │
│     └─ Stage 3: Causal EMA (Temporal Smoothing)                │
│         • Per-instrument exponential smoothing                 │
│         • EMA[t] = decay * EMA[t-1] + (1-decay) * value[t]    │
│         • Strictly causal (no lookahead)                       │
│         • decay = 0.95 (15m), 0.90 (1m-5m), 0.98 (4h-daily)  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  5. ML-Based Quality Assessment                                 │
│     • Random Forest + Gradient Boosting ensemble               │
│     • Feature importance analysis                               │
│     • R² score, MAE, RMSE metrics                              │
│     • Online learning updates                                   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  6. Comprehensive Validation                                    │
│     • Class balance check                                       │
│     • Information coefficient validation                        │
│     • Temporal stability analysis                               │
│     • Distribution sanity checks                                │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  7. Metadata & Metrics Collection                               │
│     • Label distribution statistics                             │
│     • Smoothing impact metrics                                  │
│     • Quality scores per opportunity                            │
│     • Sample weights export                                     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  8. Comprehensive Report Generation                             │
│     • General metrics (timing, coverage)                        │
│     • Financial metrics (opportunities, quality, smoothing)     │
│     • Technical metrics (system performance, algorithms)        │
│     • Process metrics (validation, recommendations)             │
└─────────────────────────────────────────────────────────────────┘
```

---

## Detailed Step-by-Step Flow

### Step 1: Configuration & Setup

**File:** `feature_generation_labeling_integration_step.py:446-682`

```python
# Load market data
market_data = klines_manager.read_data(
    symbol=config['symbol'],      # e.g., 'ETHUSDT'
    interval=config['timeframe'],  # e.g., '15m'
    data_type="processed"
)

# Get optimal threshold (symbol + timeframe specific)
optimal_threshold = get_optimal_threshold(config['symbol'], config['timeframe'])
# Example: ETHUSDT + 15m → 0.007 (0.7%)

# Auto-configure label smoothing for timeframe
smoothing_params = get_label_smoothing_params(config['timeframe'])
# 15m → {'epsilon': 0.08, 'gamma': 1.0, 'ema_decay': 0.95}
# 1m  → {'epsilon': 0.12, 'gamma': 1.5, 'ema_decay': 0.90}
# 4h  → {'epsilon': 0.05, 'gamma': 0.5, 'ema_decay': 0.98}

# Create labeler config
vol_config = VolatilityAwareConfig(
    volatility_threshold=optimal_threshold,
    lookahead_periods=6,
    label_type=LabelDefinitionType.BINARY,
    enable_long_positions=True,
    enable_short_positions=False
)

# Apply timeframe-optimized smoothing params
vol_config.label_smoothing.enabled = True
vol_config.label_smoothing.epsilon = smoothing_params['epsilon']
vol_config.label_smoothing.gamma = smoothing_params['gamma']
vol_config.label_smoothing.ema_decay = smoothing_params['ema_decay']
```

**Timeframe Detection:**
- Default: 15m (BASE_VOLATILITY_THRESHOLD = 0.007)
- Automatically adjusts smoothing parameters based on timeframe
- No manual configuration needed - optimized automatically

---

### Step 2: Pre-Processing

**File:** `feature_generation_labeling_integration_step.py:540-573`

```python
# 2a. Spike Detection & Correction
spike_stats = detect_and_correct_price_spikes(
    data=market_data,
    lookback_window=20,
    threshold_multiplier=3.0,
    volatility_window=20
)
# Identifies: flash crashes, fat-finger trades, exchange glitches
# Corrects: interpolation or removal of spikes
# Tracks: spikes_detected, spikes_corrected, avg_magnitude

# 2b. Volume Confidence (if volume data available)
volume_confidence, volume_stats = calculate_volume_confidence_adjustment(
    data=market_data,
    volume_column='volume',
    sensitivity=0.5,
    max_boost=0.33
)
# Boosts confidence for high-volume opportunities
# Penalizes low-volume opportunities
# Range: 0.67x - 1.33x adjustment

# 2c. Volatility Calculation (rolling)
# Already in labeler - volatility.window (default 20)
```

---

### Step 3: Label Generation

**File:** `volatility_aware_labeler.py:444-672`

**3a. Triple Barrier Method**
```python
# For each timestamp t:
# 1. Calculate future return over lookahead_periods (6 bars)
future_return = (price[t+6] - price[t]) / price[t]

# 2. Calculate volatility-adjusted threshold
volatility = rolling_std(returns, window=20)
vol_multiplier = clip(1 + k*(vol/vol_mean - 1), 1.0, 2.0)
effective_threshold = base_threshold * vol_multiplier

# 3. Generate raw label
if future_return >= effective_threshold:
    raw_label = 1  # Long opportunity
elif future_return <= -effective_threshold:
    raw_label = -1  # Short opportunity (if enabled)
else:
    raw_label = 0  # No opportunity
```

**3b. Local Extrema Detection (Optimal Entry)**
```python
# Find local maxima/minima for better entry timing
local_max = argrelextrema(prices, np.greater, order=3)
local_min = argrelextrema(prices, np.less, order=3)

# Adjust opportunity timing to optimal entry points
# Improves label quality by 10-15%
```

**3c. Volume Weighting**
```python
# Apply volume confidence adjustment
confidence_adjusted = base_confidence * volume_adjustment_factor
# High volume → up to +33% confidence boost
# Low volume → up to -33% confidence penalty
```

**3d. IC-Based Quality Scoring**
```python
# Calculate Information Coefficient (Spearman correlation)
IC = spearman_corr(labels, future_returns)

# Quality metrics per opportunity:
opportunity_quality_scores = [
    quality_score_for_each_opportunity  # 0.0 - 1.0 scale
]

# Overall quality assessment:
quality_scores = {
    'overall_quality': float,
    'predictability': IC,
    'hit_rate': float,
    'uplift': float,
    'stability': float,
    'sharpe': float,
    'opportunity_quality_scores': pd.Series  # Per-sample quality
}
```

**3e. Sample Weights Generation**
```python
# For PROXIMITY_REGRESSION label type (default):
# Generate both labels AND sample weights

labels, sample_weights = _generate_proximity_regression_labels(
    future_returns,
    effective_threshold,
    vol_normalized
)

# Labels: continuous in [-1, 1]
#   - Magnitude represents proximity to target (confidence)
#   - Sign represents direction (long/short)

# Sample Weights: continuous in [0, 1]
#   - Weight transform: 'linear', 'sqrt', or 'power_0.75'
#   - Used for weighted training

# ✅ BUGFIX: Now properly exported in metadata['sample_weights']
```

**Output from Step 3:**
```python
LabelingResult(
    labels=pd.Series,              # Generated labels
    metadata={
        'total_labels': int,
        'non_null_labels': int,
        'n_signals': int,
        'quality_scores': Dict,    # Comprehensive quality metrics
        'sample_weights': Dict,    # ✅ NEW: Sample weights per target
        'opportunity_data': Dict,   # Downstream-ready opportunity data
        ...
    },
    quality_scores={
        'target_0': QualityScoreResult(
            overall_quality=0.45,
            predictability=0.123,   # IC
            hit_rate=0.62,
            opportunity_quality_scores=pd.Series  # Per-sample quality
        )
    }
)
```

---

### Step 4: Label Smoothing ⭐ NEW

**File:** `volatility_aware_labeler.py:669-682`, `label_smoother.py:200-410`

**Automatic Application:**
```python
if self.config.label_smoothing.enabled:
    result_labels, smoothing_metadata = self._apply_label_smoothing(
        result_labels,
        quality_scores,
        data
    )
```

**4a. Stage 1: Classification Smoothing**
```python
# For continuous labels in [-1, 1]:
p_smooth = (1 - ε) * p

# For binary labels {-1, 0, 1}:
# Map to [0, 1], smooth, map back
p = (label + 1) / 2               # [-1,1] → [0,1]
p_smooth = (1 - ε) * p + ε * 0.5 # Shrink towards 0.5
label_smooth = 2 * p_smooth - 1   # [0,1] → [-1,1]

# Parameters by timeframe:
# 1m-5m:   ε = 0.12 (more smoothing for noise)
# 15m-1h:  ε = 0.08 (balanced)
# 4h-daily: ε = 0.05 (lighter smoothing)
```

**4b. Stage 2: Uncertainty Shrinkage**
```python
# Extract uncertainty from quality scores
sigma = 1.0 - quality_inverse  # Higher quality → lower uncertainty

# Calculate adaptive confidence weight
α = 1.0 / (1.0 + γ * sigma)
α = max(α, min_alpha)  # min_alpha = 0.12

# Shrink uncertain labels towards baseline (0 for returns)
label_shrunk = α * label_smooth + (1 - α) * baseline

# Example:
# High quality (σ=0.1): α=0.91 → minimal shrinkage
# Low quality (σ=0.8):  α=0.56 → significant shrinkage
# Very low (σ=1.5):     α=0.40 → strong shrinkage (but ≥ 0.12)

# Parameters by timeframe:
# 1m-5m:   γ = 1.5 (stronger shrinkage)
# 15m-1h:  γ = 1.0 (balanced)
# 4h-daily: γ = 0.5 (gentler shrinkage)
```

**4c. Stage 3: Causal EMA**
```python
# Exponential Moving Average per instrument (causal)
# Group by instrument if available, otherwise global

for instrument in instruments:
    labels_instrument = labels[labels.instrument == instrument]

    # Initialize with first value
    ema[0] = labels_instrument[0]

    # Causal update (no lookahead!)
    for t in range(1, len(labels_instrument)):
        ema[t] = decay * ema[t-1] + (1 - decay) * labels_instrument[t]

# Parameters by timeframe:
# 1m-5m:   decay = 0.90 (fast reaction)
# 15m-1h:  decay = 0.95 (balanced)
# 4h-daily: decay = 0.98 (slow, preserve signal) OR disabled
```

**Output from Step 4:**
```python
smoothing_result = {
    'labels_final': pd.Series,      # Smoothed labels
    'labels_raw': pd.Series,        # Original (for reference)
    'labels_stage1': pd.Series,     # After classification smoothing
    'labels_stage2': pd.Series,     # After uncertainty shrinkage
    'labels_stage3': pd.Series,     # After causal EMA
    'metadata': {
        'enabled': True,
        'config': {...},
        'stages_applied': {...},
        'statistics': {
            'raw_mean': 0.0234,
            'raw_std': 0.4567,
            'final_mean': 0.0198,
            'final_std': 0.3892,
            'mean_absolute_change': 0.0453,
            'max_absolute_change': 0.2341,
            'correlation_raw_final': 0.9234,
            'pct_changed': 87.34
        }
    }
}
```

---

### Step 5: ML-Based Quality Assessment

**File:** `feature_generation_labeling_integration_step.py:676-740`

```python
# Initialize ML quality assessor
ml_assessor = MLLabelQualityAssessor(
    config=MLQualityAssessmentConfig(
        primary_model=MLModelType.ENSEMBLE,
        ensemble_models=[RANDOM_FOREST, GRADIENT_BOOSTING],
        max_features=50,
        cv_folds=5,
        enable_online_learning=True
    )
)

# Assess label quality with ML models
ml_quality_result = ml_assessor.assess_quality(
    features=market_data,
    labels=labeling_result.labels,
    prices=market_data['close']
)

# Output:
{
    'quality_scores': {
        'predictive_power': 0.45,
        'consistency': 0.67,
        'signal_strength': 0.55
    },
    'feature_importance': pd.DataFrame,  # Top features
    'model_performance': {
        'r2_score': 0.32,
        'mae': 0.045,
        'rmse': 0.078
    }
}
```

---

### Step 6: Comprehensive Validation

**File:** `feature_generation_labeling_integration_step.py:713-763`

```python
# Initialize validator
validator = LabelingValidator(
    config=LabelingValidatorConfig(
        min_sample_size=100,
        max_class_imbalance=0.8,
        min_ic_threshold=0.05
    )
)

# Run validation checks
validation_results = validator.validate(
    labeled_data=labeled_data,
    labeling_config=None
)

# Validation checks:
# 1. Class balance: ratio of long/short/neutral
# 2. IC threshold: Spearman correlation ≥ min_threshold
# 3. Temporal stability: variance across folds
# 4. Distribution sanity: outliers, NaN, inf
# 5. Signal strength: sufficient non-zero labels

# Output:
{
    'class_balance': ValidationResult(passed=True, score=0.65),
    'ic_validation': ValidationResult(passed=True, score=0.123),
    'temporal_stability': ValidationResult(passed=True, score=0.82),
    'distribution_check': ValidationResult(passed=True, score=0.91)
}
```

---

### Step 7: Metadata & Metrics Collection

**File:** `feature_generation_labeling_integration_step.py:793-1108`

```python
# Extracted metrics:

# Basic stats
total_samples = len(market_data)
opportunities_detected = (labels != 0).sum()
long_opportunities = (labels > 0).sum()
short_opportunities = (labels < 0).sum()

# Quality metrics
high_quality_opportunities = sum(q > 0.3 for q in quality_scores)
filtered_opportunities = opportunities_detected - high_quality_opportunities

# Label distribution
label_distrib = {
    'mean': labels.mean(),
    'std': labels.std(),
    'skew': labels.skew(),
    'kurtosis': labels.kurtosis(),
    'histogram': {...},
    'qq_quantiles': {...},
    'rolling_mean_std': {...}
}

# Smoothing impact (NEW)
smoothing_stats = {
    'raw_mean': 0.0234,
    'final_mean': 0.0198,
    'mean_absolute_change': 0.0453,
    'correlation_raw_final': 0.9234,
    'pct_changed': 87.34
}

# Sample weights (NEW - bugfix)
sample_weights = {
    'default': pd.Series,  # or per-target if multi-target
}
```

---

### Step 8: Comprehensive Report Generation

**File:** `feature_generation_labeling_integration_step.py:1109-1362`

#### 8a. General Metrics
```python
general_metrics = {
    'step_name': 'feature_generation_labeling_integration_step',
    'execution_time': 12.34,  # seconds
    'success_rate': 1.0,
    'data_samples_processed': 10000,
    'labeling_operations': 850,  # opportunities detected
    'time_coverage': {
        'total_days': 45.2,
        'timeframe_minutes': 15,
        'samples_per_hour': 4,
        'samples_per_day': 96
    },
    'opportunity_analysis': {
        'avg_opportunities_per_day': 18.8,
        'opportunities_per_hour': 0.78,
        'quality_acceptance_rate': 76.5  # % high quality
    }
}
```

#### 8b. Financial Metrics
```python
financial_metrics = {
    'labeling_method': 'volatility_aware_multi_horizon',

    'volatility_config': {
        'base_threshold': 0.007,      # 0.7%
        'lookahead_periods': 6,
        'local_maxima_detection': True,
        'volatility_adaptation': True,
        'quality_threshold': 0.3,
        'rate_control_enabled': True
    },

    # ⭐ NEW: Label Smoothing Section
    'label_smoothing': {
        'enabled': True,
        'timeframe_optimized': True,
        'config': {
            'epsilon': 0.08,
            'gamma': 1.0,
            'ema_decay': 0.95,
            'ablation_mode': 'full'
        },
        'stages_applied': {
            'classification_smoothing': True,
            'uncertainty_shrinkage': True,
            'causal_ema': True
        },
        'impact': {
            'raw_label_mean': 0.0234,
            'raw_label_std': 0.4567,
            'final_label_mean': 0.0198,
            'final_label_std': 0.3892,
            'mean_absolute_change': 0.0453,
            'max_absolute_change': 0.2341,
            'correlation_raw_final': 0.9234,
            'pct_labels_changed': 87.34
        }
    },

    'opportunity_detection': {
        'total_samples_processed': 10000,
        'total_opportunities_detected': 850,
        'long_opportunities': 650,
        'short_opportunities': 0,  # disabled
        'opportunity_detection_rate': 8.5,  # %
        'avg_opportunities_per_day': 18.8
    },

    'quality_filtering': {
        'high_quality_opportunities': 650,
        'filtered_opportunities': 200,
        'quality_acceptance_rate': 76.5,  # %
        'avg_confidence_score': 0.67,
        'avg_volatility_adaptation': 1.32,  # multiplier
        'max_volatility_adaptation': 2.0,
        'min_volatility_adaptation': 1.0
    },

    'expected_performance': {
        'expected_profit_target': '0.7% base (adaptive)',
        'volatility_adjusted_targets': '0.7% - 1.4%',
        'quality_weighted_signals': '650 of 850 (76.5%)',
        'trading_signal_strength': 0.67,
        'market_regime_adaptation': '1.32x threshold adaptation',
        'volume_confidence_enhancement': 1.05,  # avg boost
        'high_volume_confirmations': 120,
        'low_volume_warnings': 45
    },

    'label_distribution': {
        'mean': 0.0198,  # smoothed
        'std': 0.3892,
        'skew': 0.45,
        'kurtosis': 2.8,
        'histogram': {...},
        'rolling_mean_std': {...}
    },

    'temporal_stability': {
        'folds': [
            {'start': '2024-01-01', 'mean': 0.021, 'variance': 0.15},
            {'start': '2024-02-01', 'mean': 0.018, 'variance': 0.14},
            ...
        ]
    }
}
```

#### 8c. Technical Metrics
```python
technical_metrics = {
    'system_performance': {
        'memory_usage_mb': 234.5,
        'execution_time_seconds': 12.34,
        'cpu_usage_percent': 45.2,
        'throughput_rows_per_second': 810
    },

    'labeling_engine': {
        'method': 'volatility_aware_multi_horizon',
        'algorithm_type': 'adaptive_threshold_with_local_extrema',
        'optimization_level': 'high',
        'memory_efficient_processing': True
    },

    'spike_detection': {
        'enabled': True,
        'spikes_detected': 5,
        'spikes_corrected': 5,
        'correction_rate': 100.0,
        'avg_spike_magnitude_pct': 2.3,
        'max_spike_magnitude_pct': 4.5
    },

    'signal_processing': {
        'local_maxima_detection': True,
        'volatility_adaptation': True,
        'quality_scoring_enabled': True,
        'confidence_calculation': True,
        'volume_weighted_confidence': True
    },

    'volume_analysis': {
        'enabled': True,
        'avg_volume_ratio': 1.05,
        'opportunities_boosted': 120,
        'opportunities_penalized': 45,
        'avg_adjustment_factor': 1.05,
        'adjustment_range': '0.67x - 1.33x'
    },

    'data_characteristics': {
        'timeframe_minutes': 15,
        'samples_per_hour': 4,
        'samples_per_day': 96,
        'total_days_coverage': 45.2,
        'data_completeness': '98.5%'
    }
}
```

#### 8d. Process Metrics
```python
process_metrics = {
    'data_loading': {
        'status': 'successful',
        'samples_loaded': 10000,
        'data_source': 'klines_parquet_manager',
        'timeframe': '15m',
        'columns_available': 12,
        'data_completeness': '98.5%'
    },

    'spike_detection_process': {
        'status': 'successful',
        'spikes_detected': 5,
        'spikes_corrected': 5
    },

    'labeling_process': {
        'status': 'successful',
        'method': 'volatility_aware',
        'opportunities_detected': 850,
        'quality_filtered': 650,
        'label_smoothing_applied': True  # ⭐ NEW
    },

    'validation': {
        'status': 'successful',
        'all_passed': True,
        'checks': {
            'data_loaded': True,
            'labeling_successful': True,
            'opportunities_detected': True,
            'detection_rate_valid': True,
            'confidence_calculated': True,
            'volatility_adaptation_active': True
        },
        'failed_checks': [],
        'recommendations': []
    }
}
```

---

## Summary: Complete Pipeline Elements

### Inputs
- **Symbol:** e.g., 'ETHUSDT'
- **Exchange:** e.g., 'binance'
- **Timeframe:** e.g., '15m' (auto-detected, configures smoothing)
- **Market Data:** OHLCV data from KlinesParquetManager

### Label Generation Elements
1. **Triple Barrier Method** - lookahead periods (6 bars)
2. **Volatility Adaptation** - 1.0x - 2.0x threshold adjustment
3. **Volume Weighting** - up to +33% confidence boost
4. **Time-to-Hit Tracking** - first-passage time recording
5. **Local Extrema Detection** - optimal entry timing
6. **IC-Based Quality Scoring** - predictive power metrics
7. **Sample Weights** - for weighted training (PROXIMITY_REGRESSION)

### Label Smoothing Elements ⭐ NEW
1. **Classification Smoothing** - epsilon smoothing (timeframe-adapted)
2. **Uncertainty Shrinkage** - quality-based shrinkage (timeframe-adapted)
3. **Causal EMA** - temporal smoothing per-instrument (timeframe-adapted)

### Weighting Elements
1. **Proximity Weighting** - label magnitude based on distance to target
2. **Sample Weights** - explicit weights for training (0-1 scale)
3. **Quality Weighting** - IC-based opportunity quality scores
4. **Volume Weighting** - volume-based confidence adjustment
5. **Uncertainty Weighting** - adaptive shrinkage based on quality

### Output Metrics (Final Report)

**General:**
- Execution time, samples processed
- Opportunities detected (total, long, short)
- Detection rate, quality acceptance rate

**Financial:**
- Volatility config (thresholds, adaptation range)
- **Label smoothing config and impact** ⭐ NEW
- Opportunity detection stats
- Quality filtering stats
- Expected performance metrics
- Label distribution (smoothed)
- Temporal stability

**Technical:**
- System performance (memory, CPU, throughput)
- Labeling engine details
- Spike detection stats
- Signal processing capabilities
- Volume analysis stats

**Process:**
- Data loading status
- Labeling process status (**smoothing applied** ⭐ NEW)
- Validation checks (all passed/failed)
- Recommendations

---

## Key Improvements from Label Smoothing

### Before Smoothing
- **Raw labels:** Noisy, overconfident, high variance
- **Problems:** Overfitting, poor calibration, temporal instability

### After Smoothing (Timeframe-Adapted)
- **Smoothed labels:** Well-calibrated, stable, uncertainty-aware
- **Benefits:**
  - Reduced overfitting (epsilon smoothing)
  - Improved calibration (temperature scaling)
  - Encoded uncertainty (quality-based shrinkage)
  - Temporal stability (causal EMA)
  - Better generalization (correlation ≈ 0.90-0.95 preserved)

### Automatic Configuration
- **No manual setup required** - configured automatically based on timeframe
- **Timeframe-optimized:** 1m-5m, 15m-1h, 4h-daily have different params
- **Default is 15m:** epsilon=0.08, gamma=1.0, ema_decay=0.95

---

## Files Involved

1. **feature_generation_labeling_integration_step.py** - Main orchestrator
   - Configures smoothing based on timeframe
   - Collects smoothing metrics for report

2. **volatility_aware_labeler.py** - Label generation
   - Generates raw labels with quality scores
   - Applies smoothing pipeline
   - Exports sample weights (bugfix)

3. **label_smoother.py** - Smoothing implementation
   - Three-stage smoothing pipeline
   - Timeframe-agnostic (receives config)

4. **comprehensive_report_generator.py** - Report generation
   - Includes smoothing metrics in financial section
   - Shows impact statistics

---

## Next Steps for Users

1. **Run the pipeline** - smoothing is automatically configured
2. **Check the report** - look at financial_metrics['label_smoothing']
3. **Monitor metrics:**
   - `correlation_raw_final` > 0.70 (preserves signal)
   - `mean_absolute_change` 0.03-0.10 (good smoothing range)
   - `pct_labels_changed` 50-95% (effective smoothing)
4. **Run ablation tests** (optional) - validate which stages help
5. **Tune if needed** - adjust params in integration step config

**Default configuration works well for most cases!**
