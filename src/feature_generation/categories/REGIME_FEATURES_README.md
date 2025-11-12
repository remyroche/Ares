# Enhanced Regime Features

This document describes the new regime-specific feature categories added to the feature generation system.

## Overview

Five new feature categories have been added to enhance regime modeling:

1. **REGIME_TRANSITIONS** - Transition probabilities, switch patterns, dynamics
2. **REGIME_PERSISTENCE** - Duration metrics, survival probabilities, exhaustion indicators
3. **MARKET_STRUCTURE** - Support/resistance within regimes, swing structure, fractals
4. **REGIME_PROBABILITY** - HMM probabilities as features, confidence metrics, patterns
5. **REGIME_UNCERTAINTY** - Classification entropy, confusion scores, ambiguity indices

These features are designed to work with HMM regime models from `training/steps/market_analysis/rolling_hmm_clustering`.

## Feature Categories

### 1. REGIME_TRANSITIONS

**File**: `regime_transitions.py`

Features related to regime transitions and switching behavior.

#### Transition Probability Features (from HMM transition matrix)

- `regime_transition_prob_to_0`, `regime_transition_prob_to_1`, etc. - Probability of transitioning to each regime
- `regime_self_transition_prob` - Probability of staying in current regime
- `regime_max_transition_prob` - Highest transition probability
- `regime_transition_entropy` - Uncertainty in next regime

#### Historical Transition Patterns

- `regime_switch_count_5`, `regime_switch_count_10`, `regime_switch_count_20` - Regime changes in windows
- `regime_switch_rate_5`, `regime_switch_rate_10`, `regime_switch_rate_20` - Normalized switch rates
- `regime_switch_acceleration` - Change in switch rate
- `regime_last_switch_distance` - Periods since last regime change

#### Transition Dynamics

- `regime_transition_volatility_5` - Volatility of regime probabilities
- `regime_transition_trend` - Increasing/decreasing transition likelihood
- `regime_boundary_proximity` - Distance from regime boundary (0-1)
- `regime_stability_score` - Inverse of transition entropy
- `regime_flip_flop_count_10` - A→B→A pattern count
- `regime_directional_bias` - Tendency to move to higher/lower regime IDs

**Usage:**
```python
from src.feature_generation.categories import create_regime_transition_generators

generators = create_regime_transition_generators()
for gen in generators:
    features = gen.generate_features(
        data=market_data,
        regime_labels=regime_labels,
        regime_probabilities=regime_probabilities,
        transition_matrix=transition_matrix
    )
```

---

### 2. REGIME_PERSISTENCE

**File**: `regime_persistence.py`

Features related to regime duration and persistence.

#### Duration Features

- `regime_duration_current` - Periods in current regime
- `regime_duration_previous` - Duration of previous regime
- `regime_duration_ratio` - Current / previous duration
- `regime_duration_percentile` - Current duration vs historical distribution
- `regime_avg_duration_5` - Average regime duration (last 5 regimes)
- `regime_max_duration_10` - Longest regime in last 10 switches
- `regime_min_duration_10` - Shortest regime in last 10 switches

#### Persistence Metrics

- `regime_persistence_score` - Duration / expected duration
- `regime_exhaustion_indicator` - Duration >> average (binary)
- `regime_premature_indicator` - Duration << average (binary)
- `regime_half_life` - Expected remaining duration
- `regime_survival_probability` - P(regime continues next period)
- `regime_age_normalized` - Duration / max observed duration

**Usage:**
```python
from src.feature_generation.categories import create_regime_persistence_generators

generators = create_regime_persistence_generators()
for gen in generators:
    features = gen.generate_features(
        data=market_data,
        regime_labels=regime_labels
    )
```

---

### 3. MARKET_STRUCTURE

**File**: `market_structure.py`

Regime-specific market structure and price patterns.

#### Price Structure

- `higher_highs_count_regime` - Count of higher highs in current regime
- `lower_lows_count_regime` - Count of lower lows in current regime
- `swing_structure_regime` - Trend indicator: 1=uptrend, -1=downtrend, 0=ranging
- `fractal_dimension_regime` - Market complexity (Higuchi fractal dimension)
- `price_range_regime_normalized` - Price range / ATR within regime
- `price_efficiency_regime` - Net displacement / total path length

#### Volume Structure

- `volume_profile_regime_std` - Volume volatility within regime
- `volume_trend_regime` - Increasing/decreasing volume
- `volume_spike_count_regime` - Number of volume spikes
- `volume_exhaustion_regime` - Declining volume indicator

**Usage:**
```python
from src.feature_generation.categories import create_market_structure_generators

generators = create_market_structure_generators()
for gen in generators:
    features = gen.generate_features(
        data=market_data,  # Must have high, low, close, volume
        regime_labels=regime_labels
    )
```

---

### 4. REGIME_PROBABILITY

**File**: `regime_probability.py`

Features derived directly from HMM regime probabilities.

#### Direct Probability Features

- `regime_prob_0`, `regime_prob_1`, etc. - Individual regime probabilities
- `regime_prob_max` - Highest regime probability
- `regime_prob_second_max` - Second highest probability
- `regime_prob_gap` - Max - second_max (confidence)

#### Probability Dynamics

- `regime_prob_entropy` - Shannon entropy across regimes
- `regime_prob_concentration` - Gini coefficient of probabilities
- `regime_prob_trend_5` - Change in max prob over 5 periods
- `regime_prob_volatility_5` - Std of max prob
- `regime_prob_acceleration` - Change in probability trend
- `regime_prob_momentum` - Rate of probability increase

#### Probability Patterns

- `regime_prob_divergence` - Current prob vs smoothed prob
- `regime_prob_crossover_count_5` - Regime probability rank changes
- `regime_prob_stability_score` - Inverse of volatility
- `regime_prob_confidence_trend` - Increasing/decreasing confidence

**Usage:**
```python
from src.feature_generation.categories import create_regime_probability_generators

generators = create_regime_probability_generators()
for gen in generators:
    features = gen.generate_features(
        data=market_data,
        regime_probabilities=regime_probabilities  # Shape: (n_samples, n_regimes)
    )
```

---

### 5. REGIME_UNCERTAINTY

**File**: `regime_uncertainty.py`

Features quantifying regime classification uncertainty.

#### Uncertainty Metrics

- `regime_classification_entropy` - Shannon entropy of regime probabilities
- `regime_confusion_score` - 1 - max_prob (how confused the model is)
- `regime_ambiguity_index` - Number of regimes with prob > 0.2
- `regime_certainty_trend` - Change in entropy over time
- `regime_decision_boundary_dist` - Distance from 50/50 decision

#### Additional Metrics

- `regime_normalized_entropy` - Entropy / max_entropy
- `regime_confidence_ratio` - Max_prob / second_max_prob
- `regime_effective_n_regimes` - exp(entropy)
- `regime_probability_spread` - Std of probabilities
- `regime_dominant_stability` - Volatility of max probability
- `regime_uncertainty_change_rate` - Rate of change in entropy

**Usage:**
```python
from src.feature_generation.categories import create_regime_uncertainty_generators

generators = create_regime_uncertainty_generators()
for gen in generators:
    features = gen.generate_features(
        data=market_data,
        regime_probabilities=regime_probabilities  # Shape: (n_samples, n_regimes)
    )
```

---

## Integration with Regime Models Training

### Step 1: Run HMM Regime Discovery

First, run the HMM regime discovery step to generate regime labels and probabilities:

```python
# This happens in training/steps/market_analysis/rolling_hmm_clustering
# Outputs:
# - regime_labels.parquet
# - regime_probabilities.h5
# - transition_matrix (in model artifact)
```

### Step 2: Load Regime Data

```python
import pandas as pd
import h5py
import pickle

# Load regime labels
regime_labels = pd.read_parquet('artifacts/regime_labels.parquet')

# Load regime probabilities
with h5py.File('artifacts/regime_probabilities.h5', 'r') as f:
    regime_probabilities = f['probabilities'][:]

# Load transition matrix from model
with open('artifacts/hmm_model.pkl', 'rb') as f:
    hmm_model = pickle.load(f)
    transition_matrix = hmm_model.transition_matrix
```

### Step 3: Generate All Regime Features

```python
from src.feature_generation.categories import (
    create_regime_transition_generators,
    create_regime_persistence_generators,
    create_market_structure_generators,
    create_regime_probability_generators,
    create_regime_uncertainty_generators
)

# Create all generators
all_generators = []
all_generators.extend(create_regime_transition_generators())
all_generators.extend(create_regime_persistence_generators())
all_generators.extend(create_market_structure_generators())
all_generators.extend(create_regime_probability_generators())
all_generators.extend(create_regime_uncertainty_generators())

# Generate features
all_features = {}
for generator in all_generators:
    features = generator.generate_features(
        data=market_data,
        regime_labels=regime_labels,
        regime_probabilities=regime_probabilities,
        transition_matrix=transition_matrix
    )
    all_features.update(features)

# Convert to DataFrame
feature_df = pd.DataFrame(all_features, index=market_data.index)
```

### Step 4: Add to Feature Matrix

```python
# Combine with other features
final_features = pd.concat([
    existing_features,
    feature_df
], axis=1)

# Use in training
X_train = final_features.loc[train_idx]
y_train = targets.loc[train_idx]
```

---

## Configuration

Each feature category has a configuration class:

```python
from src.feature_generation.categories.regime_transitions import RegimeTransitionConfig

# Customize configuration
config = RegimeTransitionConfig(
    short_window=5,
    medium_window=10,
    long_window=20,
    min_periods=5
)

# Pass to factory
generators = create_regime_transition_generators(config)
```

---

## Feature Summary

Total features added: **~60+ features**

- **REGIME_TRANSITIONS**: ~19 features
- **REGIME_PERSISTENCE**: ~13 features
- **MARKET_STRUCTURE**: ~10 features
- **REGIME_PROBABILITY**: ~15+ features (varies with n_regimes)
- **REGIME_UNCERTAINTY**: ~11 features

---

## Implementation Notes

### Data Requirements

1. **Market Data**: Must include `high`, `low`, `close`, `volume` columns
2. **Regime Labels**: Pandas Series with regime assignments (0, 1, 2, ...)
3. **Regime Probabilities**: NumPy array with shape `(n_samples, n_regimes)`
4. **Transition Matrix**: NumPy array with shape `(n_regimes, n_regimes)` (optional for some features)

### Performance Considerations

- Features use vectorized operations where possible
- Rolling operations are optimized with NumPy
- NaN handling is built-in
- Features gracefully handle missing regime data

### Data Leakage Prevention

All features are calculated using only information available up to the current time step:
- No future-looking operations
- Proper handling of rolling windows
- Regime-based calculations respect temporal boundaries

---

## Testing

Run the demo script to test all features:

```bash
python -m src.feature_generation.categories.regime_features_demo
```

This will:
1. Generate synthetic market data and regime outputs
2. Create all feature generators
3. Generate all features
4. Display feature statistics
5. Show integration example

---

## Future Enhancements

Potential additions:
- Regime-specific technical indicators
- Regime change prediction features
- Multi-regime interaction features
- Regime clustering quality metrics
- Regime-conditioned volatility forecasts

---

## References

- HMM Implementation: `src/training/steps/market_analysis/rolling_hmm_clustering/`
- Feature Generation Core: `src/feature_generation/core/`
- Feature Bank: `src/feature_generation/core/feature_bank.py`

---

## Support

For questions or issues:
1. Check the demo script: `regime_features_demo.py`
2. Review existing regime features: `regime_features.py`, `advanced_regime_features.py`
3. Consult feature integration: `regime_feature_integration.py`
