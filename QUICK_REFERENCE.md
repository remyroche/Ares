# Ares ML Pipeline - Quick Reference Guide

## File Location Quick Lookup

### Feature Generation Code
- **Base Feature Generator**: `/src/feature_generation/core/feature_generator.py`
- **Feature Engineer**: `/src/feature_generation/shared/feature_engineer.py`
- **Feature Bank (Caching)**: `/src/feature_generation/core/feature_bank.py`
- **Feature Categories**: `/src/feature_generation/categories/` (100+ feature types)
- **Feature Gen Step**: `/src/training/steps/pre_training/feature_generation_feature_generation_step.py`

### Meta-Labeling Code
- **Core Library**: `/src/utils/ml_common/labeling/meta_labeling.py`
  - `triple_barrier_labels()` - Main function
  - `purged_kfold_splits()` - Cross-validation
- **Meta-Label Step**: `/src/training/steps/market_analysis/feature_generation_meta_labeling_step.py`
  - Ensemble voting, K-fold cross-fitting, Kalman filtering
- **HPO for Meta-Labeling**: `/src/training/steps/market_analysis/meta_labeling_hpo_experiment_step.py`
  - Pareto front optimization

### Report Generation
- **Reporting System**: `/src/utils/ml_common/reporting/enhanced_reporting_system.py`
- **Report Generator**: `/src/training/steps/pre_training/utils/comprehensive_report_generator.py`
- **Metrics Calculator**: `/src/training/steps/market_analysis/shared_utils/metrics.py`
- **Metrics Sink**: `/src/training/steps/pre_training/metrics_sink.py`
- **Output Directory**: `/outcomes/` (CSV and Markdown files)

### Model Training
- **Base Step Class**: `/src/training/steps/base_step.py` (Abstract base)
- **Analyst Config**: `/src/training/steps/model_training/analyst_base_config.yaml`
- **Tactician Config**: `/src/training/steps/model_training/tactician_base_config.yaml`
- **HPO Backups**: `/src/training/steps/model_training/hpo_backups/`
- **Training Manager**: `/src/training/core/training_manager.py`

### Configuration Files
- **Model Configs**: `/src/training/steps/model_training/` (*.yaml)
- **Main Configs**: `/src/config/` (50+ YAML files)
- **Feature Configs**: `/src/config/features/` (regime, selection, etc.)

### Launcher & Orchestration
- **Main Launcher**: `/src/launcher/ares_launcher.py`
- **Step Registry**: `step_registry` (imported from base_step.py)
- **Step Modules** (auto-register on import):
  - `/src/training/steps/data_collection/__init__.py`
  - `/src/training/steps/pre_training/__init__.py`
  - `/src/training/steps/market_analysis/__init__.py`
  - `/src/training/steps/model_training/__init__.py`

### Data & Artifacts
- **Artifact Manager**: `/src/utils/artifact_manager.py`
- **Versioned Artifacts**: `/src/utils/versioned_artifacts.py`
- **Artifacts Directory**: `/artifacts/` (Parquet/HDF5 files)
- **Outcomes Directory**: `/outcomes/` (CSV/Markdown reports)

---

## Feature Generation Pipeline Steps (In Order)

1. **Data Validation**
   - File: `/src/training/steps/pre_training/feature_generation_data_validation_step.py`
   - Purpose: Validate raw OHLCV data

2. **Feature Generation**
   - File: `/src/training/steps/pre_training/feature_generation_feature_generation_step.py`
   - Output: 300+ raw features

3. **Feature Selection**
   - File: `/src/training/steps/pre_training/feature_generation_feature_selection_step.py`
   - Output: Reduced to ~100 features (remove correlations)

4. **Interaction Generation**
   - File: `/src/training/steps/pre_training/feature_generation_interaction_generation_step.py`
   - Output: Feature pair interactions

5. **Final Feature Selection**
   - File: `/src/training/steps/pre_training/feature_generation_final_feature_selection_step.py`
   - Output: 40, 50, 60 feature sets with SHAP values
   - Reports: CSV files with selection metadata

6. **Meta-Labeling**
   - File: `/src/training/steps/market_analysis/feature_generation_meta_labeling_step.py`
   - Output: Binary labels {0,1} with ensemble voting

---

## Key Metrics & Calculations

### Meta-Labeling Metrics
- Profit-take hit rate (%)
- Stop-loss hit rate (%)
- Timeout rate (%)
- Label balance ratio
- F1 score
- ROC-AUC score

### Trading Metrics
- Sharpe Ratio
- Max Drawdown
- Calmar Ratio
- Sortino Ratio
- Win Rate

### Where Metrics Are Computed
- `/src/training/steps/market_analysis/shared_utils/metrics.py` - Main metrics
- `/src/training/steps/pre_training/utils/target_quality_metrics.py` - Label quality
- Metrics recorded via `/src/training/steps/pre_training/metrics_sink.py`

---

## Report Output Locations & Format

### CSV Reports
- **Location**: `/outcomes/` directory
- **Examples**:
  - `meta_labeling_hpo_pareto_front_<symbol>_<timeframe>_<timestamp>.csv`
  - `feature_quality_<timestamp>.csv`
  - `*_interaction_summary.csv`

### Markdown Reports
- **Location**: `/outcomes/` directory
- **Format**: Auto-generated outcome files for each step
- **Generator**: `comprehensive_report_generator.py`

### Artifact Outputs
- **Location**: `/artifacts/` directory
- **Format**: Parquet or HDF5 files
- **Naming**: Descriptive names with optional versioning

---

## Configuration Key Sections (YAML)

### Model Configuration Template
```yaml
model_config:
  model_name: "name"
  target: "target_column"
  base_timeframe: "15m"
  
  base_models:
    - model_name: "ModelType"
      params: {...}
      hpo:
        enabled: true
        search_space: {...}
        optimal_params: {}
  
  training:
    cv_folds: 3
    validation_split: 0.2
    test_split: 0.1
  
  feature_engineering:
    primary_features:
      source: "feature_generation_final_feature_selection_step"
      target_count: 100
  
  evaluation:
    metrics:
      classification: ["accuracy", "precision", "recall", "f1"]
      trading: ["sharpe_ratio", "max_drawdown"]
```

---

## How to Add a New Step

1. Create step file in appropriate directory:
   ```python
   # /src/training/steps/category/my_step.py
   from src.training.steps.base_step import BaseStep
   
   class MyStep(BaseStep):
       async def execute(self, context):
           # Load data
           # Process
           # Save results
           return True
   ```

2. Register in module __init__.py:
   ```python
   from src.training.steps.base_step import step_registry
   from .my_step import MyStep
   
   step_registry.register('my_step', MyStep)
   ```

3. Call from launcher:
   ```python
   await launcher.run_step('my_step', config_dict)
   ```

---

## Common Code Patterns

### Load Artifact
```python
from src.utils.artifact_manager import ArtifactManager

mgr = ArtifactManager()
df = mgr.load_parquet('artifact_name')
```

### Save Artifact
```python
mgr.save_parquet(df, 'artifact_name')
```

### Record Metrics
```python
metrics_sink.record_metric('metric_name', value)
metrics_sink.record_metrics({'metric1': val1, 'metric2': val2})
```

### Generate Report
```python
from src.utils.tprint import tprint, tprint_success
tprint("Processing...")
tprint_success("Completed!")
```

---

## Testing & Debugging

### Run a Specific Step
```bash
cd /home/user/Ares
python -m src.launcher.ares_launcher --step feature_generation_feature_generation_step --mode light
```

### View Artifacts
```bash
ls -lh /artifacts/          # All artifacts
ls -lh /outcomes/           # Report outputs
ls -lh /versioned_artifacts/  # Versioned backups
```

### Check Logs
```bash
tail -f logs/*.log         # If logging enabled
```

---

## Dependencies & Imports

### Key Libraries
- pandas, numpy - Data manipulation
- scikit-learn - ML utilities
- lightgbm, xgboost, catboost - Gradient boosting
- tensorflow/keras - Neural networks
- optuna - Hyperparameter optimization
- vectorbt - Vectorized backtesting
- hdbscan, statsmodels - Advanced algorithms

### Key Internal Modules
```python
from src.training.steps.base_step import BaseStep, step_registry
from src.utils.artifact_manager import ArtifactManager
from src.utils.tprint import tprint, tprint_success, tprint_warning, tprint_error
from src.feature_generation.core.feature_generator import FeatureGenerator
from src.utils.ml_common.labeling.meta_labeling import triple_barrier_labels
```

---

## Lookback Configuration

In `/src/launcher/ares_launcher.py`:
```python
MODE_LOOKBACK_DAYS = {
    "light": 30,         # 30 days
    "blank": 360,        # 1 year
    "full": 365 * 3      # 3 years
}
```

---

## Symbol & Timeframe Support

### Supported Timeframes (in BaseStep)
- Minutes: 1m, 3m, 5m, 15m, 30m, 45m
- Hours: 1h, 2h, 4h, 6h, 8h, 12h
- Days: 1d, 3d, 1w, 2w
- Months: 1mo, 3mo, 6mo, 1y

### Default Assets
- ETHUSDT, BTCUSDT (Binance)
- Custom symbols via configuration

---

## Version Control & History

- Git repository: `/home/user/Ares/.git/`
- Branch (current): `claude/add-snr-diagnostics-*`
- Recent commits include winsorization and regime optimization features
- HPO configurations auto-backed up with timestamps

