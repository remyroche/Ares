# Mean Reversion: Data Leakage and Retraining Schedule Analysis

**Date:** 2025-11-27
**Analyst:** Claude
**Files Analyzed:**
- `src/training/steps/market_analysis/ml_reversion_regime_step.py`
- `src/utils/ml_common/retraining_scheduler.py`
- `src/monitoring/retrain_monitoring.py`
- `src/trading/integration/live_regime_outputs.py`

---

## Question 1: Do we only add probabilities to samples on data that the model has been untrained on?

### TL;DR: ⚠️ **NO - There is currently DATA LEAKAGE**

### Current Implementation

The `ml_mean_reversion_step` currently generates predictions for **ALL data** including the training set:

```python
# In _train_xgb_student (lines 1404-1446):

# 1. Train model on X_train
model.fit(X_train_np, y_train_np, ...)

# 2. Generate predictions for ALL splits
raw_train = model.predict_proba(X_train_np)[:, 1]  # ❌ IN-SAMPLE (data leakage!)
raw_val = model.predict_proba(X_val_np)[:, 1]      # ✅ Out-of-sample
raw_test = model.predict_proba(X_test_np)[:, 1]    # ✅ Out-of-sample

# 3. Calibrate on validation set
calibrated_model.fit(X_val_np, y_val_np)

# 4. Generate calibrated predictions for ALL splits
calib_train = calibrated_model.predict_proba(X_train_np)[:, 1]  # ❌ IN-SAMPLE (data leakage!)
calib_val = calibrated_model.predict_proba(X_val_np)[:, 1]      # ✅ Out-of-sample
calib_test = calibrated_model.predict_proba(X_test_np)[:, 1]    # ✅ Out-of-sample

# 5. Concatenate ALL predictions (line 1505)
raw_proba_full = np.concatenate([raw_train, raw_val, raw_test])
calibrated_proba_full = np.concatenate([calib_train, calib_val, calib_test])
```

Then in `execute()` (lines 470-471):
```python
# Save ALL predictions to output dataframe
output_df.loc[X_all.index, "mr_raw_score"] = raw_scores
output_df.loc[X_all.index, "mr_probability"] = calibrated_scores
```

This artifact is saved and later loaded for live trading via `load_mean_reversion_regime_outputs()`.

### Impact of Data Leakage

**For Training/Validation:**
- Training set metrics are **inflated** (overly optimistic)
- Cannot trust in-sample performance metrics
- Grid backtest results on training period are **unrealistic**

**For Live Trading (Forward-Fill from Artifact):**
- The artifact is loaded and forward-filled to live timestamps
- Live trading only uses **future timestamps** beyond the training period
- **Live trading is NOT affected by the training set leakage** ✅
- However, any analysis/backtesting on historical training periods would be affected

### Temporal Split Structure

The step DOES use proper temporal splits:

```
Burn-in Period: [Not used for training or prediction]
├─ Start: 2024-01-01
└─ End:   2024-03-31  (3 months)

Training Period: [Used to train model]
├─ Start: 2024-04-01
└─ End:   2024-08-31

Validation Period: [Used for calibration & HPO]
├─ Start: 2024-09-01
└─ End:   2024-10-31

Test Period: [Out-of-sample evaluation]
├─ Start: 2024-11-01
└─ End:   2024-11-27
```

**The problem:** Predictions are generated for the ENTIRE dataset including training period.

### Walk-Forward Validation Does It Right

The walk-forward validation (lines 1384-1498) DOES generate proper out-of-fold predictions:

```python
def _run_walkforward_validation(...):
    for fold in range(n_folds):
        # 1. Train on data BEFORE prediction period
        X_tr = X[:val_start]  # Only past data
        model.fit(X_tr, y_tr)

        # 2. Predict on FUTURE period
        X_te = X[test_start:test_end]  # Future data only
        y_pred = model.predict(X_te)
```

However, these OOF predictions are **only stored in metrics**, not in the main output dataframe.

---

## Question 2: Do we re-train models every 10 days, with larger HPO training every 30 days?

### TL;DR: **Retraining is every 5 days for XGB models (not 10), and HPO is optional via config flag**

### Retraining Schedule (from `retraining_scheduler.py`)

#### Default Schedules by Model Type

```python
@dataclass
class RetrainingSchedule:
    model_type: str
    retrain_interval_days: int
    burnin_pct: float
    min_samples_for_training: int

# XGB Models (including mean-reversion):
RetrainingSchedule.for_xgb()
├─ retrain_interval_days: 5  ⬅️ EVERY 5 DAYS (not 10!)
├─ burnin_pct: 1/12  (3 months)
├─ min_samples_for_training: 1000
└─ enable_warm_start: False

# HMM/GMM Models:
RetrainingSchedule.for_hmm() / .for_gmm()
├─ retrain_interval_days: 15  ⬅️ EVERY 15 DAYS
├─ burnin_pct: 1/12  (3 months)
├─ min_samples_for_training: 1000
└─ enable_warm_start: True

# Analyst Base Models:
RetrainingSchedule.for_analyst_base()
├─ retrain_interval_days: 5
├─ burnin_pct: 1/20  (3 months after specialist burn-in)
├─ min_samples_for_training: 2000
└─ enable_warm_start: False
```

### HPO Schedule

**There is NO separate "every 30 days with HPO" schedule.**

HPO is controlled by a config flag:
```yaml
mr_enable_hpo: false  # Default: disabled
```

When enabled, HPO runs as part of the retraining process, but **every time the model retrains**, not on a separate schedule.

**In practice:**
- Regular retraining: Every 5 days, use cached hyperparameters
- HPO retraining: Manually triggered via config flag when you want to re-optimize
- Recommended: Enable HPO every 1-2 months or when performance degrades

### Out-of-Fold (OOF) Prediction Generation

The `OOFPredictionGenerator` class ensures proper temporal separation:

```python
def generate_oof_predictions(...):
    """
    Generate out-of-fold predictions using retraining windows.

    For each window:
    1. Train on data from data_start to window_start
    2. Predict on window_start to window_end
    3. Move window forward by retrain_interval_days
    """

    windows = []
    current_prediction_start = burnin_end

    while current_prediction_start < data_end:
        window = TrainingWindow(
            training_start=data_start,
            training_end=current_prediction_start,  # Train up to here
            prediction_start=current_prediction_start,  # Predict from here
            prediction_end=current_prediction_start + timedelta(days=retrain_interval_days)
        )
        windows.append(window)
        current_prediction_start = prediction_end  # Move forward
```

**Example with 5-day retraining:**

```
Day 0-90:   Burn-in (no predictions)
Day 90-95:  Train on days 0-90, predict days 90-95
Day 95-100: Train on days 0-95, predict days 95-100
Day 100-105: Train on days 0-100, predict days 100-105
Day 105-110: Train on days 0-105, predict days 105-110
... (continues)
```

This ensures **each prediction is made with a model that has never seen that data**.

### Monitoring-Based Retraining (from `retrain_monitoring.py`)

In addition to scheduled retraining, there's an intelligent system that monitors:

1. **Calibration Loss** (2σ threshold)
   - Monitors MSE drift from historical baseline
   - Triggers retrain if calibration degrades

2. **PSI (Population Stability Index)** (0.3 threshold)
   - Detects feature distribution drift
   - Monitors key features like σ_EW, vwap_dist

3. **Correlation Drift** (0.5 threshold)
   - Monitors correlation matrix changes
   - Detects structural shifts in feature relationships

4. **Latency Breach** (50ms p99 threshold)
   - If model becomes too slow, triggers retrain
   - May enable fallback to simpler model

5. **Scheduled Retrain** (2:00 AM ET daily check)
   - Falls back to scheduled if no triggers

**Retrain Decision Tree:**

```
Monitor Metrics → Check Thresholds → Determine Urgency → Retrain Decision

Urgency Levels:
- CRITICAL: Latency breach → Immediate retrain + fallback
- HIGH: Calibration loss → Retrain within hours
- MEDIUM: PSI/Correlation drift → Retrain within day
- LOW: Scheduled → Retrain at next scheduled time
```

---

## Current State vs. Best Practice

### What's Working ✅

1. **Temporal splits** are properly configured (train/val/test)
2. **Walk-forward validation** generates proper OOF predictions (but only for metrics)
3. **Retraining infrastructure** exists with proper OOF generation
4. **Live trading** is not affected by training set leakage (uses forward-fill beyond training period)
5. **Monitoring system** can trigger adaptive retraining

### What Needs Improvement ⚠️

1. **Training set predictions** are included in saved artifact (data leakage)
2. **OOF predictions from walk-forward** are not saved to artifact
3. **Retraining scheduler** is not integrated with mean-reversion step
4. **HPO schedule** is manual via config flag (no automatic periodic HPO)

---

## Recommendations

### 1. Fix Data Leakage in Training Set (HIGH PRIORITY)

**Option A: Only save test set predictions**
```python
# In _train_xgb_student, return only test set predictions
# Do NOT concatenate training set
raw_proba_full = raw_test  # Only test set
calibrated_proba_full = calib_test  # Only test set
```

**Option B: Use walk-forward OOF predictions for entire dataset**
```python
# Replace main training with walk-forward OOF generation
oof_generator = OOFPredictionGenerator(
    schedule=RetrainingSchedule.for_xgb(),
    data_start=market_data.index.min(),
    data_end=market_data.index.max()
)

oof_predictions, models, metadata = oof_generator.generate_oof_predictions(
    data=X_all,
    training_func=lambda train_data: train_xgb_model(train_data),
    prediction_func=lambda model, pred_data: generate_predictions(model, pred_data)
)

# Save OOF predictions (all out-of-sample)
output_df.loc[oof_predictions.index, "mr_probability"] = oof_predictions["probability"]
```

**Recommended: Option B** - Provides OOF predictions for entire history

### 2. Integrate Retraining Scheduler

**Add to mean-reversion step:**
```python
from src.utils.ml_common.retraining_scheduler import (
    RetrainingSchedule,
    OOFPredictionGenerator,
    RetrainingManager
)

# Check if retraining is needed
retrain_manager = RetrainingManager()
schedule = RetrainingSchedule.for_xgb()
model_id = f"{symbol}_{exchange}_{timeframe}_mean_reversion"

if retrain_manager.should_retrain(model_id, schedule):
    # Retrain and record
    # ... training code ...
    retrain_manager.record_training(model_id, schedule)
else:
    # Load existing model
    # ... skip training ...
```

### 3. Implement Periodic HPO Schedule

**Add HPO schedule to config:**
```yaml
# Retraining configuration
mr_retrain_interval_days: 5  # Regular retraining
mr_hpo_interval_days: 30  # HPO retraining every 30 days
mr_enable_scheduled_hpo: true  # Enable automatic HPO
```

**Implementation:**
```python
retrain_manager = RetrainingManager()

# Check regular retraining (every 5 days)
if retrain_manager.should_retrain(model_id, schedule):
    # Check HPO retraining (every 30 days)
    last_hpo = retrain_manager.get_last_training_time(f"{model_id}_hpo")
    days_since_hpo = (datetime.now() - last_hpo).days if last_hpo else 999

    if days_since_hpo >= config.get("mr_hpo_interval_days", 30):
        config["mr_enable_hpo"] = True  # Enable HPO
        retrain_manager.record_training(f"{model_id}_hpo", schedule)  # Record HPO time
    else:
        config["mr_enable_hpo"] = False  # Regular training

    # ... train model ...
```

### 4. Add Monitoring Integration

**Trigger retraining based on metrics:**
```python
from src.monitoring.retrain_monitoring import MonitoringSystem, MonitoringConfig

monitoring = MonitoringSystem(MonitoringConfig())

# Update metrics
current_metrics = monitoring.update_metrics(
    features=student_df,
    predictions=calibrated_scores,
    actual=y_target_all
)

# Check retrain decision
retrain_decision = monitoring.get_retrain_decision(current_metrics)

if retrain_decision.should_retrain:
    tprint_warning(
        f"⚠️  Retrain triggered: {retrain_decision.reason} "
        f"(urgency={retrain_decision.urgency})"
    )
    # Trigger retraining...
```

### 5. Metadata Enhancement

**Add to artifact metadata:**
```python
metadata = {
    "symbol": symbol,
    "exchange": exchange,
    "timeframe": timeframe,
    "training_start": str(split_config.training.start),
    "training_end": str(split_config.training.effective_end),
    "prediction_method": "oof",  # or "in_sample" or "test_only"
    "retrain_schedule": "5_days",
    "last_hpo_date": str(last_hpo_date),
    "oof_windows": len(oof_windows),  # Number of OOF windows
}
```

---

## Implementation Priority

### Phase 1: Fix Data Leakage (Week 1)
1. ✅ Document current leakage issue
2. ⬜ Implement OOF prediction generation for main training
3. ⬜ Save only OOF predictions to artifact
4. ⬜ Add metadata indicating prediction method
5. ⬜ Test with historical backtest (verify no lookahead)

### Phase 2: Retraining Schedule (Week 2)
1. ⬜ Integrate `RetrainingManager` with mean-reversion step
2. ⬜ Add config for retraining interval
3. ⬜ Implement scheduled HPO (every 30 days)
4. ⬜ Add retraining metadata to artifacts

### Phase 3: Monitoring Integration (Week 3)
1. ⬜ Integrate `MonitoringSystem` for calibration/drift tracking
2. ⬜ Add trigger-based retraining (not just scheduled)
3. ⬜ Implement graceful degradation (fallback to simpler model)
4. ⬜ Add monitoring dashboard

### Phase 4: Testing & Validation (Week 4)
1. ⬜ Validate no lookahead bias in OOF predictions
2. ⬜ Backtest with OOF predictions vs. in-sample
3. ⬜ Verify retraining schedule works in production
4. ⬜ Monitor first cycle of scheduled retraining

---

## Summary

**Current State:**
- ❌ Training set predictions are saved (data leakage)
- ✅ Test set predictions are properly out-of-sample
- ✅ Walk-forward validation generates OOF predictions (but not used in artifact)
- ✅ Live trading is not affected (uses timestamps beyond training)
- ⚠️ Retraining is every **5 days** (not 10)
- ⚠️ HPO is **manual** via config flag (not automatic every 30 days)
- ✅ Infrastructure exists for proper OOF and scheduled retraining

**Recommended Path Forward:**
1. **Immediate:** Implement OOF prediction generation for main artifact (fix leakage)
2. **Short-term:** Integrate retraining scheduler (automatic 5-day retraining)
3. **Medium-term:** Add periodic HPO schedule (every 30 days)
4. **Long-term:** Add monitoring-based adaptive retraining

This ensures all predictions are truly out-of-sample and models stay fresh with market conditions.
