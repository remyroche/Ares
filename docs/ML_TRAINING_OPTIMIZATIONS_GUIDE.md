# ML Training Optimizations Integration Guide

This guide documents the new ML training optimizations implemented to improve efficiency and prevent lookahead bias.

## Overview of Changes

### 1. Reduced Burn-In Period
- **Old**: 1/6 of data (6 months for 3-year dataset)
- **New**: 1/12 of data (3 months for 3-year dataset)
- **Impact**: More data available for training while still allowing indicators to stabilize

### 2. Retraining Schedules
- **HMM Models**: Retrain every 15 days with warm start
- **GMM Models**: Retrain every 15 days with warm start and semantic sorting
- **XGB Models**: Retrain every 5 days with adaptive local search HPO
- **Analyst Base**: Retrain every 5 days with 3-month burn-in (1/20 of data)
- **Analyst Ensemble**: Train on all OOF predictions from base models

### 3. Out-of-Fold (OOF) Predictions
All models now generate predictions using only data available up to time t, preventing lookahead bias.

## New Modules

### 1. Retraining Scheduler (`src/utils/ml_common/retraining_scheduler.py`)

Manages retraining schedules and generates OOF predictions.

#### Key Classes

**`RetrainingSchedule`**: Defines retraining configuration
```python
from src.utils.ml_common.retraining_scheduler import RetrainingSchedule

# Create schedules for different model types
hmm_schedule = RetrainingSchedule.for_hmm()      # 15-day retraining
gmm_schedule = RetrainingSchedule.for_gmm()      # 15-day retraining
xgb_schedule = RetrainingSchedule.for_xgb()      # 5-day retraining
base_schedule = RetrainingSchedule.for_analyst_base()  # 5-day, 1/20 burn-in
ensemble_schedule = RetrainingSchedule.for_analyst_ensemble()
```

**`OOFPredictionGenerator`**: Generates out-of-fold predictions
```python
from src.utils.ml_common.retraining_scheduler import OOFPredictionGenerator

# Create generator
generator = OOFPredictionGenerator(
    schedule=xgb_schedule,
    data_start=data.index.min(),
    data_end=data.index.max()
)

# Define training and prediction functions
def train_model(train_data):
    # Train your model on train_data
    model = XGBClassifier(**params)
    model.fit(train_data[features], train_data[target])
    return model

def make_predictions(model, pred_data):
    # Make predictions on pred_data
    predictions = model.predict_proba(pred_data[features])
    return pd.DataFrame({
        'probability': predictions[:, 1]
    }, index=pred_data.index)

# Generate OOF predictions
oof_predictions, models, metadata = generator.generate_oof_predictions(
    data=data,
    training_func=train_model,
    prediction_func=make_predictions,
    show_progress=True
)
```

**`create_sample_weights`**: Create exponential weights for recent samples
```python
from src.utils.ml_common.retraining_scheduler import create_sample_weights

# For analyst base (18-month half-life)
weights = create_sample_weights(
    timestamps=data.index,
    half_life_months=18.0
)

# For analyst ensemble (2-month half-life)
weights = create_sample_weights(
    timestamps=data.index,
    half_life_months=2.0
)
```

### 2. Adaptive Local Search HPO (`src/utils/ml_common/optimization/local_search_hpo.py`)

Efficient hyperparameter optimization for XGBoost models.

#### Key Classes

**`AdaptiveGrid`**: Manages adaptive hyperparameter search
```python
from src.utils.ml_common.optimization.local_search_hpo import AdaptiveGrid, HPOConfig

# Create adaptive grid
config = HPOConfig(
    local_n_trials=10,          # Small local search
    global_n_trials=30,         # Wider global search
    global_every_n_runs=6,      # Global search every 6 runs
    enable_early_stopping=True,
    early_stopping_rounds=50
)

grid = AdaptiveGrid(config=config)

# Define objective function
def objective(params):
    model = XGBClassifier(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=50)
    score = model.score(X_val, y_val)
    return score

# Optimize (automatically chooses local or global search)
best_params, best_score = grid.optimize(
    model_id=f"xgb_{symbol}_{timeframe}",
    objective_func=objective
)
```

### 3. GMM Semantic Sorting (`src/utils/ml_common/gmm_semantic_sorting.py`)

Prevents label switching problem in GMM models.

#### Key Classes

**`GMMSemanticSorter`**: Sorts GMM components consistently
```python
from src.utils.ml_common.gmm_semantic_sorting import GMMSemanticSorter, create_warm_started_gmm
from sklearn.mixture import GaussianMixture

# Create sorter
sorter = GMMSemanticSorter(sort_by='mean_magnitude')

# Fit GMM and sort components
gmm = GaussianMixture(n_components=3, covariance_type='full')
sorted_gmm, component_order = sorter.fit_and_sort(gmm, X_train)

# Predict with sorted components
labels = sorted_gmm.predict(X_test)
probabilities = sorted_gmm.predict_proba(X_test)

# For warm start in next retraining
previous_gmm = sorted_gmm
new_gmm = create_warm_started_gmm(
    n_components=3,
    previous_gmm=previous_gmm,
    covariance_type='full'
)
```

### 4. HMM Warm Start (`src/utils/ml_common/hmm_warm_start.py`)

Accelerates HMM training with warm start.

#### Key Classes

**`HMMWarmStarter`**: Manages HMM warm starting
```python
from src.utils.ml_common.hmm_warm_start import HMMWarmStarter
from hmmlearn.hmm import GaussianHMM

# Create warm starter
warm_starter = HMMWarmStarter()

# Train initial HMM
hmm = GaussianHMM(n_components=3, covariance_type='full', n_iter=100)
hmm.fit(X_train)

# Extract parameters
params = warm_starter.extract_params(hmm)

# Create warm-started HMM for retraining
new_hmm = warm_starter.create_warm_started_hmm(
    n_components=3,
    n_features=X_train.shape[1],
    previous_hmm=hmm,
    covariance_type='full',
    n_iter=100
)

# Validate convergence
metrics = warm_starter.validate_hmm_convergence(new_hmm, X_train)
print(f"Converged: {metrics['converged']}, Iterations: {metrics['n_iter']}")
```

### 5. Training Optimizations (`src/utils/ml_common/training_optimizations.py`)

Performance optimizations for analyst base models.

#### Key Functions

**Histogram Binning** (XGBoost)
```python
from src.utils.ml_common.training_optimizations import HistogramBinner

# Bin features for faster training
binner = HistogramBinner(n_bins=256, strategy='quantile')
X_train_binned = binner.fit_transform(X_train)
X_val_binned = binner.transform(X_val)
```

**LightGBM GOSS**
```python
from src.utils.ml_common.training_optimizations import configure_lightgbm_optimizations

# Get optimized LightGBM parameters
lgb_params = configure_lightgbm_optimizations(
    enable_goss=True,
    top_rate=0.2,      # Keep 20% large gradient samples
    other_rate=0.1,    # Sample 10% small gradient samples
    max_bin=255
)

# Use with LightGBM
import lightgbm as lgb
model = lgb.LGBMRegressor(**lgb_params)
```

**Memory Optimization**
```python
from src.utils.ml_common.training_optimizations import optimize_dataframe_memory

# Reduce memory usage
data_optimized = optimize_dataframe_memory(data)
# Automatically converts float64→float32, reduces integer precision
```

**Precision Reduction**
```python
from src.utils.ml_common.training_optimizations import PrecisionReducer

reducer = PrecisionReducer(target_dtype='float32')
data_reduced = reducer.reduce_precision(data, exclude_columns=['target'])
```

## Integration Examples

### Example 1: XGB Specialist Model with Adaptive HPO

```python
from src.utils.ml_common.retraining_scheduler import (
    RetrainingSchedule, OOFPredictionGenerator
)
from src.utils.ml_common.optimization.local_search_hpo import AdaptiveGrid, HPOConfig
import xgboost as xgb

# Setup
schedule = RetrainingSchedule.for_xgb()  # 5-day retraining
generator = OOFPredictionGenerator(schedule, data.index.min(), data.index.max())

# Adaptive HPO setup
hpo_config = HPOConfig(local_n_trials=10, global_n_trials=30, global_every_n_runs=6)
adaptive_grid = AdaptiveGrid(hpo_config)

# Training function with HPO
def train_with_hpo(train_data):
    X_train = train_data[features]
    y_train = train_data[target]

    # Split for validation
    split_idx = int(len(X_train) * 0.8)
    X_tr, X_val = X_train[:split_idx], X_train[split_idx:]
    y_tr, y_val = y_train[:split_idx], y_train[split_idx:]

    # HPO objective
    def objective(params):
        model = xgb.XGBClassifier(**params)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], early_stopping_rounds=50, verbose=False)
        return model.score(X_val, y_val)

    # Optimize
    best_params, _ = adaptive_grid.optimize(
        model_id=f"xgb_{symbol}_{timeframe}",
        objective_func=objective
    )

    # Train final model with best params
    model = xgb.XGBClassifier(**best_params)
    model.fit(X_train, y_train)
    return model

# Prediction function
def predict(model, pred_data):
    X_pred = pred_data[features]
    proba = model.predict_proba(X_pred)
    return pd.DataFrame({'probability': proba[:, 1]}, index=pred_data.index)

# Generate OOF predictions
oof_preds, models, metadata = generator.generate_oof_predictions(
    data=data,
    training_func=train_with_hpo,
    prediction_func=predict
)
```

### Example 2: GMM Model with Semantic Sorting

```python
from src.utils.ml_common.gmm_semantic_sorting import (
    GMMSemanticSorter, create_warm_started_gmm, measure_gmm_quality
)
from src.utils.ml_common.retraining_scheduler import RetrainingSchedule, OOFPredictionGenerator

schedule = RetrainingSchedule.for_gmm()  # 15-day retraining
generator = OOFPredictionGenerator(schedule, data.index.min(), data.index.max())

sorter = GMMSemanticSorter(sort_by='mean_magnitude')
previous_gmm = None

def train_gmm(train_data):
    global previous_gmm

    X_train = train_data[features]

    # Create GMM with warm start
    gmm = create_warm_started_gmm(
        n_components=3,
        previous_gmm=previous_gmm,
        covariance_type='full'
    )

    # Fit and sort
    sorted_gmm, order = sorter.fit_and_sort(gmm, X_train)

    # Measure quality
    quality = measure_gmm_quality(sorted_gmm, X_train)
    logger.info(f"GMM quality: BIC={quality['bic']:.2f}, balance={quality['weight_balance']:.2f}")

    previous_gmm = sorted_gmm
    return sorted_gmm

def predict_gmm(model, pred_data):
    X_pred = pred_data[features]
    proba = model.predict_proba(X_pred)
    # Probabilities are already in sorted order
    return pd.DataFrame(proba, index=pred_data.index, columns=[f'regime_{i}' for i in range(3)])

oof_preds, models, metadata = generator.generate_oof_predictions(
    data=data,
    training_func=train_gmm,
    prediction_func=predict_gmm
)
```

### Example 3: HMM Model with Warm Start

```python
from src.utils.ml_common.hmm_warm_start import HMMWarmStarter, create_hmm_with_gmm_init
from src.utils.ml_common.retraining_scheduler import RetrainingSchedule, OOFPredictionGenerator
from hmmlearn.hmm import GaussianHMM
from sklearn.mixture import GaussianMixture

schedule = RetrainingSchedule.for_hmm()  # 15-day retraining
generator = OOFPredictionGenerator(schedule, data.index.min(), data.index.max())

warm_starter = HMMWarmStarter()
previous_hmm = None

def train_hmm(train_data):
    global previous_hmm

    X_train = train_data[features].values

    if previous_hmm is None:
        # First training: initialize from GMM
        gmm = GaussianMixture(n_components=3, covariance_type='full')
        gmm.fit(X_train)

        hmm = create_hmm_with_gmm_init(
            n_components=3,
            n_features=X_train.shape[1],
            gmm_means=gmm.means_,
            gmm_covariances=gmm.covariances_,
            covariance_type='full',
            n_iter=100
        )
    else:
        # Subsequent training: warm start from previous
        hmm = warm_starter.create_warm_started_hmm(
            n_components=3,
            n_features=X_train.shape[1],
            previous_hmm=previous_hmm,
            covariance_type='full',
            n_iter=100
        )

    # Fit HMM
    hmm.fit(X_train)

    # Validate convergence
    metrics = warm_starter.validate_hmm_convergence(hmm, X_train)
    logger.info(f"HMM converged: {metrics['converged']}, iterations: {metrics['n_iter']}")

    if previous_hmm is not None:
        # Compare with previous
        comparison = warm_starter.compare_hmms(previous_hmm, hmm)
        logger.info(f"HMM change: transition_diff={comparison['transition_matrix_diff']:.4f}")

    previous_hmm = hmm
    return hmm

def predict_hmm(model, pred_data):
    X_pred = pred_data[features].values
    states = model.predict(X_pred)
    return pd.DataFrame({'state': states}, index=pred_data.index)

oof_preds, models, metadata = generator.generate_oof_predictions(
    data=data,
    training_func=train_hmm,
    prediction_func=predict_hmm
)
```

### Example 4: Analyst Base with Optimizations

```python
from src.utils.ml_common.retraining_scheduler import (
    RetrainingSchedule, OOFPredictionGenerator, create_sample_weights
)
from src.utils.ml_common.training_optimizations import (
    configure_xgboost_optimizations,
    configure_lightgbm_optimizations,
    optimize_dataframe_memory
)
import xgboost as xgb
import lightgbm as lgb

# Setup with analyst base schedule (1/20 burn-in, 5-day retraining)
schedule = RetrainingSchedule.for_analyst_base()
generator = OOFPredictionGenerator(schedule, data.index.min(), data.index.max())

# Optimize memory first
data_opt = optimize_dataframe_memory(data)

# Training function with optimizations
def train_analyst_base(train_data):
    X_train = train_data[features]
    y_train = train_data[target]

    # Create sample weights (18-month half-life)
    weights = create_sample_weights(
        timestamps=train_data.index,
        half_life_months=18.0
    )

    # Configure optimized parameters
    xgb_params = configure_xgboost_optimizations(
        enable_histogram=True,
        max_bin=256,
        tree_method='hist'
    )

    # Train with sample weights
    model = xgb.XGBRegressor(**xgb_params, **base_params)
    model.fit(X_train, y_train, sample_weight=weights)

    return model

# OR use LightGBM with GOSS
def train_analyst_base_lgb(train_data):
    X_train = train_data[features]
    y_train = train_data[target]

    # Create sample weights
    weights = create_sample_weights(
        timestamps=train_data.index,
        half_life_months=18.0
    )

    # Configure LightGBM with GOSS
    lgb_params = configure_lightgbm_optimizations(
        enable_goss=True,
        top_rate=0.2,
        other_rate=0.1,
        max_bin=255
    )

    model = lgb.LGBMRegressor(**lgb_params, **base_params)
    model.fit(X_train, y_train, sample_weight=weights)

    return model

def predict_base(model, pred_data):
    X_pred = pred_data[features]
    preds = model.predict(X_pred)
    return pd.DataFrame({'prediction': preds}, index=pred_data.index)

# Generate OOF predictions
oof_preds, models, metadata = generator.generate_oof_predictions(
    data=data_opt,
    training_func=train_analyst_base,
    prediction_func=predict_base
)
```

### Example 5: Analyst Ensemble with Sample Weighting

```python
from src.utils.ml_common.retraining_scheduler import create_sample_weights
import xgboost as xgb

# Assume we have all OOF predictions from base models
# base_predictions_df contains columns: pred_model1, pred_model2, pred_model3, target

# Create sample weights with 2-month half-life
weights = create_sample_weights(
    timestamps=base_predictions_df.index,
    half_life_months=2.0
)

# Train ensemble
X_ensemble = base_predictions_df[[col for col in base_predictions_df.columns if col != 'target']]
y_ensemble = base_predictions_df['target']

ensemble_model = xgb.XGBRegressor(**ensemble_params)
ensemble_model.fit(X_ensemble, y_ensemble, sample_weight=weights)

# Make predictions
ensemble_predictions = ensemble_model.predict(X_ensemble)
```

## Migration Checklist

### For Specialist Models (HMM, GMM, XGB)

- [ ] Remove hardcoded `burnin_pct=1/6` (now defaults to 1/12)
- [ ] Import retraining scheduler modules
- [ ] Replace single training with OOF prediction generation
- [ ] For HMM: Add warm start support
- [ ] For GMM: Add semantic sorting
- [ ] For XGB: Add adaptive local search HPO
- [ ] Update artifact saving to handle OOF predictions
- [ ] Update reporting to show OOF metrics

### For Analyst Base Models

- [ ] Change burn-in to 1/20 of data (3 months after specialist burn-in)
- [ ] Implement OOF prediction generation
- [ ] Add sample weighting with 18-month half-life
- [ ] Add optimization techniques:
  - [ ] Histogram binning (XGBoost)
  - [ ] GOSS (LightGBM)
  - [ ] Precision reduction
  - [ ] Memory optimization
- [ ] Update reporting to indicate OOF predictions

### For Analyst Ensemble Models

- [ ] Train on ALL OOF predictions from base models
- [ ] Add sample weighting with 2-month half-life
- [ ] No separate burn-in needed (uses base model predictions)

## Performance Impact

### Expected Improvements

1. **Training Speed**:
   - Adaptive HPO: 2-3x faster for XGB models (small local searches)
   - GOSS: 1.5-2x faster for LightGBM models
   - Warm start: 2-5x faster convergence for HMM/GMM models

2. **Memory Usage**:
   - Precision reduction: 50% reduction (float64→float32)
   - Memory optimization: 30-50% reduction

3. **Retraining Efficiency**:
   - XGB: 5-day retraining vs. static models
   - HMM/GMM: 15-day retraining vs. static models
   - Adaptive to market regime changes

4. **Prediction Quality**:
   - OOF predictions eliminate lookahead bias
   - Sample weighting improves recent prediction accuracy
   - Semantic sorting prevents GMM label switching

## Testing

### Validation Steps

1. **Check OOF Coverage**: Ensure all timestamps have predictions
2. **Verify No Lookahead**: Confirm predictions only use past data
3. **Monitor Training Time**: Track improvement in training speed
4. **Compare Predictions**: Validate consistency across retraining windows
5. **Check Memory Usage**: Monitor memory consumption improvements

### Example Validation

```python
# Check OOF coverage
assert len(oof_predictions) == len(data_after_burnin)
assert oof_predictions.index.equals(data_after_burnin.index)

# Verify chronological order
assert oof_predictions.index.is_monotonic_increasing

# Check for gaps
time_diffs = oof_predictions.index.to_series().diff()
max_gap = time_diffs.max()
print(f"Maximum gap in predictions: {max_gap}")
```

## Troubleshooting

### Common Issues

1. **Memory Errors**: Use memory optimization and precision reduction
2. **Slow Training**: Reduce HPO trials or enable GOSS/histogram binning
3. **Label Switching in GMM**: Ensure semantic sorting is applied
4. **HMM Not Converging**: Check warm start initialization or increase n_iter
5. **OOF Gaps**: Check minimum samples requirement in schedule

## References

- Retraining Scheduler: `/home/user/Ares/src/utils/ml_common/retraining_scheduler.py`
- Adaptive HPO: `/home/user/Ares/src/utils/ml_common/optimization/local_search_hpo.py`
- GMM Semantic Sorting: `/home/user/Ares/src/utils/ml_common/gmm_semantic_sorting.py`
- HMM Warm Start: `/home/user/Ares/src/utils/ml_common/hmm_warm_start.py`
- Training Optimizations: `/home/user/Ares/src/utils/ml_common/training_optimizations.py`
