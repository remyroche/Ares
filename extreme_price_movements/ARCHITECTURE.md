# Architecture Blueprint

## Purpose
This document outlines the architectural philosophy and system-level design for the ML-for-trading pipeline. The system is designed to discover, validate, and execute trading strategies by rigorously identifying highly predictable market regimes and modeling extreme price movements. The architecture bridges the gap between raw market data and economically viable execution, prioritizing mathematical rigor, computational efficiency, and strict out-of-sample validation.

## System Goals
- **Robust Discovery:** Systematically search for and validate tradable market regimes and event configurations.
- **Statistical rigor:** Prevent data leakage, enforce temporal cross-validation, and adhere to Advances in Financial Machine Learning (AFML) principles (e.g., Marcos Lopez de Prado).
- **Computational Efficiency:** Optimize for memory-constrained environments (specifically Apple Silicon / Mac M1 architectures) through strict downcasting, vectorized operations, and intelligent caching.
- **Separation of Concerns:** Maintain clear boundaries between data ingestion, predictive modeling, and economic evaluation.

## High-Level Pipeline
The end-to-end research and modeling pipeline is divided into distinct, loosely coupled stages:

1. **Data Ingestion & Preprocessing:** Fetching raw market data, handling missing values, and standardizing inputs into contiguous memory structures.
2. **Feature Generation:** Computing mathematically dense, stationary feature matrices (e.g., fractional differentiation, microstructural signals) without lookahead bias.
3. **Event / Regime Generation:** Defining directional impulses, volatility states, and candidate masks that trigger model evaluation. This is where continuous time is downsampled into discrete, tradable events.
4. **Model Training:** Fitting machine learning estimators to event-filtered data using strict out-of-fold temporal cross-validation.
5. **Economic Evaluation:** Translating statistical predictions into economic reality via path-dependent simulations (e.g., Triple-Barrier Method) to calculate risk-adjusted metrics (Sortino, MaxDD, Win Rate).
6. **Reporting / Exports:** Summarizing model metrics, regime diagnostics, and strategy weights into actionable artifacts.

## Runtime Module Map (Current Implementation)

The architecture above is implemented by the following concrete runtime components.

### Pipeline entrypoint
- `extreme_price_movements/run_pipeline.py` is the orchestrator for batch/offline flows.
- Canonical CLI modes are:
  - `download`
  - `labels`
  - `features`
  - `train`
  - `train_base`
  - `train_meta`
  - `sizer`
  - `optimise`
  - `backtest`
  - `run`
  - `breakdown_diagnostics`

### Stage mapping to code
1. **Labels (`labels`)**
   - `run_pipeline.py: run_labels(...)` delegates to `pipeline_steps.run_label_generation_step_v2(...)`.
   - Per-bucket report emitted via `reports.bucket_report.report_labels(...)`.

2. **Features (`features`)**
   - `run_pipeline.py: run_features(...)` delegates to `pipeline_steps.run_feature_generation_step(...)`.

3. **Base training (`train`)**
   - `run_pipeline.py: run_train(...)` delegates to `pipeline_steps.run_training_step(...)`.
   - Detailed per-bucket/per-model reporting is emitted through `reports.bucket_report.report_base_training(...)`.

4. **Meta training (`train_meta`)**
   - `run_pipeline.py: run_train_meta(...)` delegates to `train_daily_meta(...)`.
   - Detailed reporting is emitted through `reports.bucket_report.report_meta_training(...)`.

5. **Position sizing (`sizer`)**
   - The active CLI mode is `sizer` (not `position_sizer_v2`).
   - `run_pipeline.py: run_sizer(...)` delegates to `pipeline_steps.run_sizer_step(...)` -> `run_ridge_sizer_step(...)`.
   - Detailed per-bucket reporting is emitted through `reports.bucket_report.report_ridge_sizer(...)`.
   - `position_sizer_v2.py` exists and is used by the broader sizing stack, but is not exposed as a direct `run_pipeline.py` mode string.

6. **Optimization (`optimise`)**
   - `run_pipeline.py: run_optimise(...)` drives optimization from backtest outputs or ridge OOF mode.
   - Per-bucket detailed reporting is emitted through `reports.bucket_report.report_optimise(...)`.

### Offline optimizers and candidate-mask stack
- `extreme_price_movements/mask_optimiser.py` is an active module in the candidate-mask optimization workflow.
- `extreme_price_movements/offline_optimisers/compare_tbm_parameters.py` is an active TBM optimizer/compare entrypoint.
  - Typical invocation:
    - `python3 extreme_price_movements/offline_optimisers/compare_tbm_parameters.py --data-root data --output reports/tbm_comparison.csv`

### Inference runtime
- Inference is implemented under `extreme_price_movements/inference/`.
- Primary entrypoint: `extreme_price_movements/inference/run_inference.py`.
- Inference is intentionally separate from `run_pipeline.py` CLI modes.

## Core Architectural Principles

### Statistical Validity vs. Economic Validity
A core tenet of this architecture is the strict separation between statistical validity and economic validity:
- **Statistical Validity** measures a model's ability to learn and generalize underlying patterns (e.g., Log Loss, R², AUC). It is evaluated purely on the prediction targets.
- **Economic Validity** measures the practical utility of those predictions in a simulated market environment (e.g., trading costs, bid-ask spread, path dependency, stop-loss/take-profit boundaries).
A model must prove statistically valid before it is economically evaluated, but high statistical validity does not guarantee economic viability.

### Event-Based Modeling and Temporal Validation
Following AFML principles, the system does not predict every chronological bar. Instead, it filters the timeline down to discrete structural *events* (e.g., structural breaks, volatility spikes, extreme movements).
- All models operate on these event-driven subsets.
- Temporal validation is enforced strictly: training data must strictly precede validation data, and embargo periods must be applied to prevent overlap leakage between overlapping event horizons.

## Data and Compute Flow: Shared Caches
To avoid Out-of-Memory (OOM) errors and redundant computations, the architecture relies heavily on shared caches and intelligent preprocessing:
- **Precomputed State:** Heavy, global market tensors (like rolling volatility, ATR, or structural screening arrays) are computed exactly once and stored in shared memory caches.
- **Cache Injection:** These caches are passed by reference to downstream candidate evaluators and optimization loops.
- **Lifecycle:** Caches exist only for the duration of the optimization run and are aggressively garbage-collected or explicitly cleared when moving between major pipeline phases.

## Stage-Based Regime Discovery
Finding tradable regimes relies on a funnel approach, eliminating poor candidates early to save compute:

1. **Structural Filtering:** Rapidly discard candidate events that lack basic structural viability (e.g., too few events, insufficient market dispersion, or low volatility).
2. **Predictive Scoring:** Apply fast, lightweight proxy models to estimate the baseline predictability (e.g., zero-shot statistical dispersion) of the remaining candidates.
3. **Feature Learnability:** Run rigorous, out-of-fold ML evaluations to verify that the specific feature matrices contain actual alpha (signal) for the candidate events.
4. **Economic Tradability:** Subject the surviving, highly learnable candidates to path-dependent economic simulations (like TBM) to calculate real-world trading viability.

## Final Ranking Strategy
When the pipeline selects a final model or regime, the ranking conceptually combines three distinct pillars:
- **Predictive Signal:** The base classification or regression metric (e.g., out-of-fold AUC).
- **Feature Learnability:** The delta or "gain" showing that the model explicitly learned from the features rather than exploiting an imbalanced dataset or structural shortcut.
- **Economic Tradability:** The risk-adjusted return metric normalized by market volatility (e.g., ATR-normalized profit, Sortino ratio).

## Validation Philosophy
- **No Leakage:** Preprocessing (like imputation or scaling) must strictly occur *within* the cross-validation fold. Global `NaN` filling or scaling before splitting is strictly forbidden.
- **Temporal CV:** All cross-validation must respect the arrow of time. Purged/Embargoed K-Fold or strict forward-chaining splits are mandatory.
- **Fold Stability:** A model is only as good as its worst fold. Optimization targets should penalize high variance across folds.
- **Out-of-Sample Discipline:** A pristine, hold-out dataset must be preserved until the final strategy candidate is entirely locked.

## Performance Architecture
The project is optimized for consumer hardware (e.g., Mac M1) through deliberate engineering choices:
- **NumPy / Numba First:** Hot loops, rolling calculations, and simulations must be executed in compiled Numba kernels (`@njit`) or vectorized NumPy operations. Pandas is reserved for orchestration and final reporting, never for row-by-row iteration.
- **Float32 / Downcasting:** All numeric working arrays (OHLC, features, returns) must be explicitly downcast to `float32` or `int8`/`int32` wherever numerically safe to halve memory consumption.
- **Memory Safety:** To prevent OOMs, avoid deeply nested DataFrame copies. Operate on underlying numpy arrays (`.to_numpy()`) and clear intermediate variables dynamically.
- **Observability:** Long-running loops should utilize `tprint()` (or equivalent timestamped printing functions) to provide standard progress logging and troubleshooting hooks without heavy logging framework overhead.
- **Type Hints:** Use standard Python type hinting across all function signatures to aid static analysis and developer comprehension.

## Extension Rules

When extending the architecture, follow these conceptual boundaries:
- **New Features:** Must be implemented as stateless, vectorized transformations. They belong in the Feature Generation conceptual block and must inherently avoid lookahead bias.
- **New Regime Metrics:** Metrics defining "what is an event" (e.g., an alternative to standard deviation spikes) belong in the Event / Regime Generation stage. They should be evaluated purely on their ability to isolate structural market states.
- **New Economic Metrics:** Metrics defining profitability (e.g., custom drawdown penalties, Sharpe variations) belong in the Economic Evaluation / Policy stage. They should operate on raw path returns, independent of how the ML model generated the prediction.

## What This Document Is Not
- **It is not a coding style guide:** Refer to `AGENTS.md` for specific rules regarding syntax, linting, formatting, and required PR structures.
- **It is not a frozen repository manifest:** module names and folders may evolve, but the stage boundaries and validation principles in this document must remain stable.
