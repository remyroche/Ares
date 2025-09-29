# Regime Training Enhancements

This document describes the enhanced configurations for regime base training and regime meta-model training, incorporating advanced meta-features and optimized hyperparameters.

## Overview

The regime training system has been enhanced with:

1. **Enhanced CatBoost Configuration** - Optimized for multiclass softprob with stable sweet spot parameters
2. **ExtraTrees Configuration** - Extremely randomized trees with improved stability
3. **Bayesian Rule Lists** - New interpretable model for regime detection
4. **LightGBM Meta-Model** - Shallow, efficient meta-learner for regime prediction
5. **Advanced Meta-Features** - Comprehensive feature engineering for improved predictions

## Configuration Files

### 1. Regime Base Training Configuration (`regime_base_training_config.yaml`)

Contains configurations for base models used in regime detection:

#### CatBoost (Multiclass Softprob)
```yaml
catboost:
  task_type: CPU  # or GPU if available
  loss_function: MultiClass
  depth: [4, 5, 6]
  learning_rate: [0.03, 0.04, 0.05, 0.06]
  l2_leaf_reg: [6, 8, 10, 12]
  iterations: [500, 800, 1200]
  subsample: 0.7  # random row sampling
  colsample_bylevel: 0.7  # random feature sampling
  grow_policy: SymmetricTree
  bootstrap_type: Bayesian  # more stable
  eval_metric: MultiClass
  # Stable sweet spot: depth=5, lr=0.05, l2=8, iterations≈800
```

#### ExtraTrees (Extremely Randomized Trees)
```yaml
extratrees:
  n_estimators: [300, 500, 800]  # more trees = stabler OOS
  max_depth: [None, 10, 15]  # None = fully grown
  min_samples_split: [5, 10, 20]
  min_samples_leaf: [2, 5, 10]
  max_features: ["sqrt", 0.3, 0.5]
  bootstrap: false
  criterion: "gini"  # or "entropy"
  random_state: 42
  # Stable sweet spot: n_estimators=500, max_depth=None, min_samples_leaf=5, max_features="sqrt"
```

#### Bayesian Rule Lists
```yaml
bayesian_rule_lists:
  max_rules: 30  # cap total rules
  max_rule_length: 3  # "if cond1 & cond2 & cond3"
  n_chains: 3  # MCMC chains
  n_iter: 10000  # iterations per chain
  min_support: 0.02  # rules must cover at least 2% of samples
  alpha: 1.0  # prior for rule selection (higher → shorter lists)
  beta: 1.0  # prior for list length
```

### 2. Regime Meta-Model Training Configuration (`regime_metamodel_training_config.yaml`)

Contains configurations for meta-models that combine base model predictions:

#### LightGBM Meta (Multiclass) - Very Shallow
```yaml
lightgbm_meta:
  objective: multiclass
  num_class: null  # set to number of regimes (K)
  num_leaves: [15, 23, 31]
  max_depth: [3, 4, 5]
  learning_rate: [0.03, 0.04, 0.05]
  min_data_in_leaf: [50, 100, 150]
  feature_fraction: [0.6, 0.75, 0.9]
  bagging_fraction: 0.8
  bagging_freq: 1
  lambda_l1: [0, 1e-2, 1e-1]
  lambda_l2: [0, 1e-2, 1e-1]
  n_estimators: [200, 400, 600]
  boosting: gbdt
  metric: multi_logloss
  # Stable sweet spot: num_leaves=23, max_depth=4, lr=0.04, min_data_in_leaf=100, n_estimators≈400
```

## Meta-Features Configuration

The system includes comprehensive meta-features that go beyond simple "max-min" approaches:

### Disagreement & Uncertainty Features

1. **Margin**: `margin = max_k p̄_k - max_{j≠k*} p̄_j` where p̄ is the mean over base models
2. **Entropy**: `H(p̄) = -∑_k p̄_k log p̄_k` (higher ⇒ more uncertain)
3. **Gini Impurity**: `1 - ∑_k p̄_k²` (equivalent signal to entropy, cheaper)
4. **Pairwise Variance**: `Var_m(p_{k*}^{(m)})` for the top class k*
5. **Disagreement Rate**: Fraction of base models not equal to ensemble argmax
6. **JS Divergence Spread**: Mean pairwise Jensen-Shannon divergence between p^(m) distributions

### Temporal Dynamics Features (Short Windows: 3–8 bars)

1. **Probability Slope**: `Δp̄_{k*} = p̄_{k*}(t) - p̄_{k*}(t-1)`
2. **Momentum of Confidence**: EWMA of margin (half-life 3–8 bars)
3. **Flip Pressure**: Rolling count of argmax changes over last W bars
4. **Duration Prior**: Time since last regime change (or EWMA of persistence)

### Calibration & Reliability Features

1. **Brier Components**: Per-class `(p̄_k - 𝟙[y=k])²` on shadow validation stream
2. **Temperature Proxy**: Optimize single temperature on rolling window

### Diversity & Specialist Detection Features

1. **Specialist Gating Cues**: Include each model's own top-class prob and calibration error
2. **Cohen's κ / Q-statistic**: Between hard predictions on rolling window
3. **Diversity Metrics**: Higher diversity tends to help stacking

## Usage Examples

### Basic Usage

```python
from src.config.regime_training_integration_example import RegimeTrainingIntegration

# Initialize integration
integration = RegimeTrainingIntegration()

# Get base training configuration
base_config = integration.get_training_configuration("base")

# Get meta-model training configuration
meta_config = integration.get_training_configuration("meta")

# Create meta-features from base model predictions
meta_features = integration.create_meta_features(
    base_predictions, base_probabilities, timestamps
)
```

### Advanced Usage

```python
# Get specific model configurations
catboost_config = integration.get_catboost_config()
extratrees_config = integration.get_extratrees_config()
lightgbm_meta_config = integration.get_lightgbm_meta_config()

# Get meta-features configuration
meta_features_config = integration.get_meta_features_config()

# Validate configurations
if integration.validate_configuration(base_config):
    print("Configuration is valid")
```

## Integration with Existing Pipeline

The new configurations are integrated into the existing training pipeline through:

1. **Updated Base Training Config**: Enhanced `PerRegimeTrainingConfig` with new model types and HPO spaces
2. **New Meta-Model Config**: `RegimeMetaModelTrainingConfig` for advanced meta-learning
3. **Enhanced Ensemble Config**: Updated `EnsembleTrainingConfig` with LightGBM meta-learner

### Model Types Added

- `ExtraTreesRegressor` - Extremely randomized trees
- `BayesianRuleLists` - Interpretable rule-based model
- `LightGBMClassifier` - Shallow meta-learner

### HPO Search Spaces Updated

All model-specific hyperparameter optimization search spaces have been updated with the new parameter ranges and stable sweet spot configurations.

## Performance Considerations

### Memory Management
- Maximum memory usage: 80% of available memory
- Memory optimization enabled by default
- Caching enabled for intermediate results

### Parallel Processing
- Automatic job detection (`n_jobs: -1`)
- Parallel processing enabled by default
- Configurable worker limits

### Caching
- Results cached in `cache/regime_base_training/` and `cache/regime_metamodel_training/`
- Configurable cache directories
- Automatic cache cleanup

## Best Practices

1. **Use Stable Sweet Spots**: The configurations include pre-optimized parameter combinations that work well in practice
2. **Meta-Feature Selection**: Use 2-5 of the most important meta-features to avoid overfitting
3. **Temporal Windows**: Use short windows (3-8 bars) for temporal dynamics features
4. **Validation**: Always validate configurations before training
5. **Monitoring**: Enable detailed logging for debugging and performance monitoring

## Troubleshooting

### Common Issues

1. **Memory Issues**: Reduce batch size or enable memory optimization
2. **Slow Training**: Enable parallel processing and caching
3. **Poor Performance**: Check meta-feature selection and model configuration
4. **Configuration Errors**: Use the validation functions to check configurations

### Debug Mode

Enable detailed logging:
```python
import logging
logging.basicConfig(level=logging.INFO)
```

## Future Enhancements

1. **GPU Acceleration**: Support for GPU-accelerated training
2. **AutoML Integration**: Automatic model selection and hyperparameter optimization
3. **Real-time Adaptation**: Dynamic model updates based on performance
4. **Advanced Meta-Features**: Additional features for improved predictions

## References

- [CatBoost Documentation](https://catboost.ai/docs/)
- [LightGBM Documentation](https://lightgbm.readthedocs.io/)
- [ExtraTrees Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesRegressor.html)
- [Bayesian Rule Lists](https://github.com/Hongyuy/brl)