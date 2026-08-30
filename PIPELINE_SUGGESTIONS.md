# Pipeline Refactoring Suggestions

Current structure of `run_pipeline.py` supports `download`, `train`, and `run`. The `train` step is currently monolithic, handling data loading, feature engineering, labeling, model training, and risk optimization in a single pass.

To improve modularity, traceability, and iteration speed, we suggest decomposing the pipeline into the following granular steps:

## Proposed New Steps

### 1. `features`
**Purpose:** Compute and persist technical features from raw OHLCV data.
**Why:** Features are computationally expensive and stable. They should not be recomputed for every model training iteration.
**Input:** Raw OHLCV data (`PartitionedOHLCVStore`).
**Output:** Feature Parquet files (`data_store/features/`).
**Implementation:**
- Wrap `compute_features_hourly`.
- Ensure features are saved using `save_features`.
- Add a `--force` flag to overwrite existing features.

### 2. `labels` (or `opportunities`)
**Purpose:** Generate training labels and identify trade opportunities.
**Why:** Differentiating "opportunity detection" (identifying high-volatility events) and "labeling" (determining outcomes) from model training allows for:
- Analysis of label distribution and class imbalance independent of the model.
- Reuse of the same labeled dataset across different model architectures (MR vs TF).
- Generation of "Exhaustion History" as a standalone artifact.
**Input:** Features (`data_store/features/`) + OHLCV.
**Output:**
- `training_set_H{horizon}.parquet`: X, y, weights, and metadata (ts, symbol) for specific horizons.
- `exhaustion_history.parquet`: Probabilistic exhaustion signals.
**Implementation:**
- Extract logic from `select_best_horizon` and `build_hourly_training_set_and_weights`.
- Save the resulting DataFrames to an `artifacts/{run_id}/labels/` directory.

### 3. `train`
**Purpose:** Train Machine Learning models using pre-computed labels.
**Why:** Focuses purely on model selection, hyperparameter tuning, and meta-modeling.
**Input:** Labeled Datasets (`training_set_H*.parquet`).
**Output:** `model_state.pkl` (or `model_bundle.pkl`).
**Implementation:**
- Load the labeled datasets.
- Run `ModelRace` for MR and TF models.
- Train `MetaModel`.
- Save the trained model bundle.

### 4. `risk`
**Purpose:** Optimize risk parameters based on trained model performance.
**Why:** Risk optimization (stop-loss, trailing stop) is a distinct optimization problem that operates on the output of the alpha models.
**Input:** Trained Model Bundle + Historical Data.
**Output:** Updated `model_state.pkl` with `risk_params`.
**Implementation:**
- Load `model_state.pkl`.
- Run `optimize_risk_params`.
- Update and save the state.

### 5. `backtest` (New)
**Purpose:** Verify model performance on a holdout set or recent history.
**Why:** Ensures that the trained system performs as expected before live deployment.
**Input:** `model_state.pkl` + Recent Data.
**Output:** Performance Report (Metrics, Plots).
**Implementation:**
- Use `engine.simulate_trade_hourly` over a defined test period (e.g., last 3 months).
- Generate a report using `metrics.py`.

### 6. `report` (New)
**Purpose:** Generate a human-readable report of the training run.
**Why:** To inspect feature importance, learning curves, and backtest results.
**Output:** HTML/PDF report.

## Data Flow Diagram

```mermaid
graph TD
    RawData[OHLCV Data] --> FeaturesStep[Step: features]
    FeaturesStep --> Features[Features Parquet]

    Features --> LabelsStep[Step: labels]
    RawData --> LabelsStep
    LabelsStep --> LabeledData[Labeled Datasets]
    LabelsStep --> ExhHistory[Exhaustion History]

    LabeledData --> TrainStep[Step: train]
    ExhHistory --> TrainStep
    TrainStep --> ModelBundle[Model Bundle]

    ModelBundle --> RiskStep[Step: risk]
    RawData --> RiskStep
    RiskStep --> ModelState[Model State (with Risk)]

    ModelState --> BacktestStep[Step: backtest]
    BacktestStep --> Report[Performance Report]
```

## Additional Recommendations

1.  **Artifact Versioning:**
    - Store artifacts in timestamped directories (e.g., `artifacts/20241025_1200/`).
    - Use a `run_id` to link features, labels, and models.

2.  **Configuration Management:**
    - Allow overriding `CFG` parameters via a YAML config file passed as an argument (e.g., `--config config.yaml`).
    - This allows experimenting with different horizons or feature sets without code changes.

3.  **Data Health Checks:**
    - Add a `validate` step to check for data gaps, NaNs, or anomalies in the raw data before starting the pipeline.
    - Leverage `utils.check_inf_nan` and existing checks in `data_store.py`.

4.  **Traceability:**
    - Ensure every step logs its input artifact IDs (or paths) and output artifact IDs.
    - Write a `manifest.json` in the artifact directory describing the entire pipeline run.

5.  **Exhaustion Model Separation:**
    - The `ExhaustionModel` is currently trained inside `train_daily` (specifically `generate_exhaustion_history`). This should be part of the `labels` step, as it provides a signal used as a feature for downstream models.
