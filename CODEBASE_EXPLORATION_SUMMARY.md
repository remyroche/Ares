# Ares ML Pipeline - Codebase Exploration Summary

## Overview
Ares is a sophisticated ML-based trading system with a feature-rich pre-training pipeline, meta-labeling, regime detection, and ensemble model training. The system is organized around autonomous pipeline steps using a registry pattern with artifact management.

---

## 1. KEY DIRECTORIES & STRUCTURE

### Main Source Structure (/home/user/Ares/src/)
```
src/
├── feature_generation/          # Core feature engineering module
├── training/                    # Main training pipeline
├── utils/                       # Shared utilities
├── analyst/                     # Analyst model components
├── models/                      # ML model definitions
├── trading/                     # Trading execution components
├── tactician/                   # Tactician model components
├── launcher/                    # Pipeline orchestration
├── monitoring/                  # Monitoring and diagnostics
└── core/                        # Core system components
```

---

## 2. FEATURE GENERATION (/src/feature_generation/)

### Directory Structure
```
feature_generation/
├── core/                        # Core feature generation engine
│   ├── feature_generator.py    # Base feature generator class
│   ├── feature_bank.py         # Feature storage/caching system
│   ├── auto_optimized_feature_generator.py
│   └── optimization_strategies.py
├── shared/                     # Shared feature engineering
│   ├── feature_engineer.py     # Base engineer class
│   ├── regime_feature_engineer.py
│   └── feature_validator.py
├── categories/                 # Feature types
│   ├── technical_indicators.py
│   ├── trend.py
│   ├── volatility_indicators.py
│   ├── support_resistance.py
│   ├── spectral_features.py
│   └── time.py
├── utils/                      # Feature utilities (100+ files)
│   ├── optimized_feature_pipeline.py
│   ├── cross_timeframe_analysis_pipeline.py
│   ├── vectorbt_*.py           # VectorBT optimization modules
│   ├── enhanced_sr_feature_extractor.py
│   └── [Many more utilities...]
└── integration/               # Integration with clustering/training
    ├── enhanced_hdp_hmm_clustering_integration.py
    ├── enhanced_ms_dr_clustering_integration.py
    └── enhanced_ensemble_training_integration.py
```

### Key Classes & Functions
- **FeatureGenerator**: Base class for all feature generation
- **FeatureBank**: Caches and manages feature computations
- **AutoOptimizedFeatureGenerator**: Automatic optimization for feature generation
- **RegimeFeatureEngineer**: Generates regime-specific features
- **Cross-timeframe Analysis**: Analyzes features across multiple timeframes

---

## 3. META-LABELING CODE

### Primary Meta-Labeling Implementation
**Location**: `/src/utils/ml_common/labeling/meta_labeling.py`

**Key Functions**:
- `triple_barrier_labels()`: Implements Lopez de Prado's triple-barrier method
  - Computes profit-take (TP) and stop-loss (SL) barriers
  - Returns labels {0,1} for meta-labeling success/failure
  - Supports volatility-adaptive barriers
  - Handles long/short sides

- `purged_kfold_splits()`: Time-series aware cross-validation
  - Prevents lookahead bias
  - Removes overlapping training samples
  - Implements purging and embargo buffer

- `compute_volatility()`: Rolling volatility proxy using EWMA

### Meta-Labeling Steps
**Location**: `/src/training/steps/market_analysis/`

1. **feature_generation_meta_labeling_step.py**
   - Production meta-labeling with ensemble models (LGBM + XGBoost + RF)
   - K-fold cross-fitting to prevent leakage
   - Volatility-adaptive labeling with Kalman filtering
   - RobustScaler feature engineering
   - Vectorized operations for performance
   - TPSL parameters: 1% profit, 0.5% stop, 0.15% fee

2. **meta_labeling_hpo_experiment_step.py**
   - HPO system for label quality discovery
   - Learnability scoring and entropy constraints
   - Pareto optimization for label quality vs coverage
   - Outputs CSV files with Pareto front solutions

### Configuration Files
- `/src/config/...`: Various meta-labeling configs
- `/src/training/steps/pre_training/profit_labeling_aligned_config.yaml`

---

## 4. TRAINING PIPELINE STRUCTURE

### Training Steps Hierarchy (/src/training/steps/)

```
training/steps/
├── base_step.py                    # Abstract BaseStep class
│
├── data_collection/                # Data collection & preparation (10+ components)
│   ├── data_preparation/
│   └── data_quality_components/
│
├── pre_training/                   # Feature generation pipeline
│   ├── feature_generation_data_validation_step.py
│   ├── feature_generation_labeling_integration_step.py
│   ├── feature_generation_feature_generation_step.py       (RAW FEATURES)
│   ├── feature_generation_period_lookback_optimization_step.py
│   ├── feature_generation_feature_selection_step.py
│   ├── feature_generation_interaction_generation_step.py   (INTERACTION FEATURES)
│   ├── regime_aware_feature_interaction_generation_step.py
│   ├── feature_generation_gate_feature_step.py
│   ├── feature_generation_final_feature_selection_step.py  (FINAL 40/50/60 FEATURES)
│   ├── feature_generation_final_validation_step.py
│   ├── metrics_sink.py
│   ├── analyst_profit_labeler.py
│   └── components/
│       ├── final_feature_selection.py
│       └── [Other components]
│
├── market_analysis/                # Regime detection & meta-labeling
│   ├── feature_generation_meta_labeling_step.py            (META-LABELING)
│   ├── meta_labeling_hpo_experiment_step.py                (META-LABELING HPO)
│   ├── labeling_components.py
│   ├── shared_utils/
│   │   └── metrics.py               # Consensus/economic metrics
│   └── [HMM, MS-DR, HDBSCAN clustering]
│
├── model_training/                 # Model training & ensemble
│   ├── unified_models_training_step.py
│   ├── *_config.yaml               # Model configurations
│   └── [Model implementations]
│
├── models_training/                # Alternative training framework
│   └── unified_training_pipeline.py
│
└── model_validation/               # Model validation
    └── tactician_validator.py
```

### Pipeline Flow (Logical Order)
1. **Data Collection** → Load OHLCV data from Binance
2. **Feature Generation** (7-8 steps)
   - Raw features from OHLCV
   - Interaction features
   - Cross-timeframe features
   - Regime-specific features
   - Final selection (40/50/60 features)
3. **Meta-Labeling** → Apply triple-barrier labels
4. **Model Training** → Train ensemble models (Analyst + Tactician)
5. **Model Validation** → Evaluate performance

---

## 5. REPORTS & METRICS GENERATION

### Report Generation System
**Location**: `/src/utils/ml_common/reporting/`

**Files**:
- `enhanced_reporting_system.py` - Comprehensive reporting framework
  - ReportType enum: Training, HPO, Validation, Error, Performance, System Health
  - AlertLevel enum: Info, Warning, Error, Critical
  - Real-time monitoring and alerting
  - Report persistence and history

### Metrics Calculation
**Locations**:
- `/src/training/steps/market_analysis/shared_utils/metrics.py`
  - MetricsCalculator class
  - Consensus metrics between regime detectors
  - Economic significance evaluation
  - Trading viability metrics
  - Stability evaluation

- `/src/training/steps/pre_training/metrics_sink.py`
  - MetricsSink base class for metric recording
  - Pluggable metrics infrastructure

- `/src/training/steps/pre_training/utils/target_quality_metrics.py`
  - Target/label quality metrics
  - Learnability scoring

### Report File Generation

**CSV Reports** (to outcomes/ directory):
- `meta_labeling_hpo_pareto_front_<symbol>_<timeframe>_<timestamp>.csv`
  - Pareto front solutions from HPO
  - Includes: TP threshold, SL threshold, profit target, stop target, F1 score, etc.

- `feature_quality_<timestamp>.csv`
  - Feature selection and quality metrics

- `*_interaction_summary.csv`
  - Interaction feature statistics

**Markdown Reports**:
- Generated via comprehensive_report_generator.py
- Outcome files for each step (auto-generated)
- Location: outcomes/ directory

### Key Metrics Tracked

**Classification Metrics**:
- ROC-AUC Score
- Precision, Recall, F1
- Log Loss, Brier Score
- Average Precision

**Trading Metrics**:
- Sharpe Ratio
- Max Drawdown
- Calmar Ratio
- Sortino Ratio
- Win Rate

**Meta-Labeling Specific**:
- Profit-take hit rate (TP%)
- Stop-loss hit rate (SL%)
- Timeout rate
- Label imbalance ratio
- Learnability score
- Entropy measures

---

## 6. CONFIGURATION STRUCTURE

### Configuration Files Location: `/src/training/steps/model_training/`

**Example: tactician_base_config.yaml** (400+ lines)

```yaml
tactician_config:
  model_name: "tactician_base"
  model_type: "separate_models"
  target: "entry_timing"
  base_timeframe: "15m"
  execution_frequency: "3m"
  price_change_target: 0.005  # 0.5%
  
  base_models:           # Define 5 model types
    - name: "StandaloneGRU"
    - name: "LGBM"
    - name: "CatBoost"
    - name: "ExtraTrees"
    - name: "XGBoost"
  
  # HPO configuration per model
  hpo:
    enabled: true
    n_rounds: 2
    enable_final_refinement: true
    final_refinement_trials: 50
    search_space: {...}
    optimal_params: {}  # Updated by HPO
  
  training:
    enable_cross_validation: true
    cv_folds: 3
    enable_early_stopping: true
    validation_split: 0.2
    test_split: 0.1
  
  feature_engineering:
    exclude_raw_ohlcv:
      enabled: true
    primary_features:
      source: "feature_generation_final_feature_selection_step"
      initial_count: 300
      target_count: 100
    cross_timeframe:
      enable: true
      base_timeframe: "15m"
      target_timeframes: ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]
    regime_features:
      enable: true
      source: "regime_ml_models"
    analyst_ensemble_outputs:
      enable: true
  
  evaluation:
    metrics:
      regression: ["mse", "mae", "r2", "mape"]
      classification: ["accuracy", "precision", "recall", "f1"]
      trading: ["sharpe_ratio", "max_drawdown", "calmar_ratio"]
    cross_validation:
      method: "time_series_split"
      n_splits: 5
    comparison:
      enable_model_ranking: true
      ranking_metric: "sharpe_ratio"
```

### Configuration Organization

**Config Directories**:
- `/src/config/` - Main configuration files (50+ YAML files)
- `/src/config/features/` - Feature-specific configs
- `/src/training/steps/model_training/` - Model training configs
- `/src/training/steps/market_analysis/` - Market analysis configs
- `/src/analyst/` - Analyst-specific configs
- `/src/feature_generation/configs/` - Feature generation configs

**HPO Backups** (Auto-saved):
- `/src/training/steps/model_training/hpo_backups/` - 30+ backup configs with timestamps

---

## 7. PIPELINE ORCHESTRATION

### Launcher System
**Location**: `/src/launcher/ares_launcher.py`

**Key Features**:
- Step Registry Pattern: All steps registered in a central registry
- Autonomous Execution: Each step has `execute()` method
- Artifact Management: Uses ArtifactManager for data I/O
- Outcome Files: Auto-generates markdown reports for each step

**Execution Flow**:
```python
# Register all steps
step_registry.register('feature_generation_labeling_integration_step', ...)
step_registry.register('feature_generation_feature_generation_step', ...)
...

# Execute step
await launcher.run_step(step_name, config_dict)
```

### Step Registration
**Location**: `/src/training/steps/__init__.py` files

Each step module registers itself on import:
- data_collection/__init__.py
- pre_training/__init__.py
- market_analysis/__init__.py
- model_training/__init__.py

### Lookback Configuration (in ares_launcher.py)
```python
MODE_LOOKBACK_DAYS = {
    "light": 30,         # 30 days
    "blank": 360,        # 1 year
    "full": 365 * 3      # 3 years
}
```

---

## 8. ARTIFACT & VERSIONING SYSTEM

### Artifact Management
**Location**: `/src/utils/artifact_manager.py`

**Features**:
- Centralized artifact storage in `/artifacts/` directory
- Parquet file storage for data
- HDF5 support for large datasets
- Version control with timestamps

### Versioned Artifacts
**Location**: `/src/utils/versioned_artifacts.py`

- Automatic versioning of artifacts
- Directory: `/versioned_artifacts/`
- Timestamped artifact storage

---

## 9. KEY ML PIPELINE COMPONENTS

### Feature Engineering Steps (In Sequence)

1. **Data Validation** (feature_generation_data_validation_step.py)
   - Input: Raw OHLCV data
   - Output: Validated data

2. **Raw Feature Generation** (feature_generation_feature_generation_step.py)
   - Generates 300+ raw features
   - Technical indicators, trend, volatility, momentum
   - Cross-timeframe features

3. **Feature Selection** (feature_generation_feature_selection_step.py)
   - Removes highly correlated features
   - Stability analysis
   - Quality metrics

4. **Interaction Generation** (feature_generation_interaction_generation_step.py)
   - Creates interaction features between pairs
   - Significant interaction discovery
   - Data-driven approach

5. **Final Feature Selection** (feature_generation_final_feature_selection_step.py)
   - **Key Output**: 40, 50, 60 feature sets
   - SHAP values for interpretability
   - Selection metadata
   - Location: `/outcomes/` directory

6. **Meta-Labeling** (feature_generation_meta_labeling_step.py)
   - Triple-barrier labeling
   - Ensemble voting (LGBM + XGBoost + RF)
   - K-fold cross-fitting
   - Output: Binary labels {0,1}

### Model Training Components

1. **Analyst Model**
   - Role: WHAT to trade (directional prediction)
   - Input: Selected features + regime confidence
   - Models: Ensemble (GRU, LGBM, XGBoost, CatBoost, RF)
   - Output: Probability + Confidence + Meta-learner score

2. **Tactician Model**
   - Role: WHEN to trade (timing optimization)
   - Input: Same features as Analyst
   - Target: Timing for 0.5% price change
   - Execution: Every 3 minutes on 15m timeframe

3. **Regime Models**
   - Role: Market regime detection
   - Methods: HMM, MS-DR, HDBSCAN, Sticky Finite HMM
   - Output: 4 regime probabilities per timeframe

---

## 10. KEY FILES & THEIR LOCATIONS

### Feature Generation
- `/src/feature_generation/core/feature_generator.py` - Base generator
- `/src/feature_generation/shared/feature_engineer.py` - Feature engineer
- `/src/feature_generation/core/feature_bank.py` - Feature caching

### Meta-Labeling
- `/src/utils/ml_common/labeling/meta_labeling.py` - Core meta-labeling
- `/src/training/steps/market_analysis/feature_generation_meta_labeling_step.py` - Step implementation
- `/src/training/steps/market_analysis/meta_labeling_hpo_experiment_step.py` - HPO

### Reporting & Metrics
- `/src/utils/ml_common/reporting/enhanced_reporting_system.py` - Report system
- `/src/training/steps/market_analysis/shared_utils/metrics.py` - Metrics calculator
- `/src/training/steps/pre_training/metrics_sink.py` - Metrics sink base class

### Training Pipeline
- `/src/training/steps/base_step.py` - BaseStep abstract class
- `/src/training/core/training_manager.py` - Training manager
- `/src/launcher/ares_launcher.py` - Main launcher

### Configuration
- `/src/training/steps/model_training/tactician_base_config.yaml` - Model config
- `/src/training/steps/model_training/analyst_base_config.yaml` - Analyst config
- `/src/config/` - Additional configs (50+ files)

### Utilities
- `/src/utils/artifact_manager.py` - Artifact persistence
- `/src/utils/hardware/unified_hardware_manager.py` - Hardware optimization
- `/src/utils/tprint.py` - Logging utility

---

## 11. OUTPUT DIRECTORIES

### Artifacts Directory
- **Location**: `/home/user/Ares/artifacts/`
- **Contents**: Parquet/HDF5 files, model checkpoints, feature matrices
- **Size**: ~122 MB (1000+ files)

### Outcomes Directory
- **Location**: `/home/user/Ares/outcomes/`
- **Contents**: CSV/Markdown reports from each step
- **Examples**:
  - `meta_labeling_hpo_pareto_front_*.csv`
  - `feature_quality_*.csv`
  - `*_interaction_summary.csv`
  - Step outcome markdown files

### Versioned Artifacts
- **Location**: `/home/user/Ares/versioned_artifacts/`
- **Purpose**: Timestamped versions of artifacts for rollback

### Analysis Output
- **Location**: `/home/user/Ares/analysis_output/`
- **Contents**: Analysis and diagnostic reports

---

## 12. DATA FLOW OVERVIEW

```
Binance API
    ↓
Data Collection Step
    ↓
Raw OHLCV Data (Artifacts)
    ↓
Feature Generation (300+ features)
    ├─ Technical Indicators
    ├─ Cross-timeframe Features
    ├─ Regime Features
    └─ Volatility-adjusted Features
    ↓
Feature Selection (300 → 100)
    ├─ Correlation filtering
    ├─ Stability analysis
    └─ Quality metrics
    ↓
Interaction Generation
    ├─ Feature pair combinations
    └─ Significant interaction discovery
    ↓
Final Feature Selection (100 → 40/50/60)
    ├─ SHAP values
    ├─ Feature importance
    └─ Selection metadata (CSV output)
    ↓
Meta-Labeling (Triple Barrier)
    ├─ Ensemble voting
    ├─ K-fold cross-fitting
    └─ Binary labels {0,1} (CSV output)
    ↓
Model Training
    ├─ Analyst: Direction prediction
    ├─ Tactician: Timing optimization
    └─ Regime: Market mode detection
    ↓
Evaluation Metrics
    ├─ Trading metrics (Sharpe, Drawdown)
    ├─ Classification metrics (ROC-AUC, F1)
    └─ CSV/Markdown reports
```

---

## 13. CONFIGURATION FLOW

### How Configuration Works

1. **YAML Configs** (define model structure and hyperparameters)
   - Location: `/src/training/steps/model_training/*.yaml`
   - Example: `tactician_base_config.yaml`

2. **HPO System** (hyperparameter optimization)
   - Uses Optuna for optimization
   - Searches over `search_space` defined in config
   - Saves optimal params back to `optimal_params` in config
   - Backups: `/src/training/steps/model_training/hpo_backups/`

3. **Runtime Config** (percentage-based allocation)
   - Training: 70% of samples (overrides `training_samples`)
   - Validation: 15% of samples (overrides `validation_samples`)
   - Test: 15% of samples (overrides `test_samples`)

4. **Feature Engineering Config** (in same YAML)
   - Primary features source
   - Feature selection method
   - Scaling strategy
   - Outlier handling

5. **Evaluation Config**
   - Metrics to compute
   - Cross-validation strategy
   - Model comparison parameters

---

## 14. KEY PATTERNS & DESIGN

### Step-Based Architecture
```python
class FeatureGenerationMetaLabelingStep(BaseStep):
    async def execute(self, context):
        # Load data via artifact_manager
        # Process data
        # Save results
        # Generate outcome markdown
        return True
```

### Registry Pattern
```python
# In step module
step_registry.register('step_name', StepClass)

# In launcher
step = step_registry.get('step_name')
await step.execute(config)
```

### Artifact Management
```python
# Get artifact manager
artifact_mgr = ArtifactManager()

# Load artifact
df = artifact_mgr.load_parquet('feature_set_v1')

# Save artifact
artifact_mgr.save_parquet(df, 'feature_set_v1_processed')
```

---

## 15. KEY RECENT CHANGES (Nov 2025)

From examining recent commits and code:
- **Winsorization features** added for robustness
- **Time features** to improve model robustness
- **Quantile regime optimization** with winsorized metrics
- **Meta-labeling improvements**:
  - Data starvation fixes
  - Removed vol_expansion filter → continuous feature
  - Dynamic threshold tuning
  - Wider volatility-adjusted horizons
  - Sequential bootstrapping for sample weights

---

## SUMMARY OF KEY FINDINGS

### 1. Feature Generation
- **Location**: `/src/feature_generation/`
- **Pipeline**: 300+ raw features → 100 selected → 40/50/60 final
- **Methods**: Technical, cross-timeframe, interaction, regime-specific

### 2. Meta-Labeling
- **Location**: `/src/utils/ml_common/labeling/` + `/src/training/steps/market_analysis/`
- **Method**: Triple-barrier (Lopez de Prado)
- **Outputs**: Binary labels + CSV Pareto front (HPO)

### 3. Reports & Metrics
- **Locations**: 
  - Reporting: `/src/utils/ml_common/reporting/`
  - Metrics: `/src/training/steps/market_analysis/shared_utils/`
  - Sink: `/src/training/steps/pre_training/`
- **Outputs**: CSV files to `/outcomes/` + Markdown reports

### 4. ML Pipeline
- **Architecture**: Feature Gen → Feature Sel → Meta-label → Model Train → Eval
- **Steps**: 8+ autonomous steps via registry pattern
- **Models**: Analyst (direction) + Tactician (timing) + Regime

### 5. Configuration
- **YAML-based**: All in `/src/training/steps/model_training/`
- **HPO**: Optuna-based with automatic backup
- **Flexible**: Percentage-based sample allocation at runtime

