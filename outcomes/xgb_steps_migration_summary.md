# XGBoost Steps Migration Summary

**Date:** 2025-11-27
**Status:** In Progress

## Clarifications

### 1. Historical Data Schedule (NOT Real-Time)

The 10-day/30-day retraining schedule refers to **HISTORICAL DATA windows**, not real-time retraining:

**Historical Training (OOF Windows):**
```
Day 0-90:   Burn-in (3 months, no predictions)
Day 90-100: Window 1 - Train on [0-90]   → Predict [90-100]   (NO HPO)
Day 100-110: Window 2 - Train on [0-100]  → Predict [100-110]  (NO HPO)
Day 110-120: Window 3 - Train on [0-110]  → Predict [110-120]  (WITH HPO) ← 30 days of data
Day 120-130: Window 4 - Train on [0-120]  → Predict [120-130]  (NO HPO)
...
```

Each window is 10 days of **historical data**, HPO every 3rd window (30 days of historical data).

For **live/production retraining**, use RetrainingManager separately to check if actual elapsed time requires retraining.

### 2. Warm Start Storage (Per-Step)

Warm start parameters are saved separately for each step:

```
cache/xgb_hpo/
├── ETHUSDT_binance_15m_mean_reversion_warm_start.json
├── ETHUSDT_binance_15m_smc_warm_start.json
├── ETHUSDT_binance_15m_breakout_bounce_warm_start.json
└── ETHUSDT_binance_15m_path_warm_start.json
```

Each step has its own HPO history and warm start parameters.

## Steps to Migrate

### ✅ Verified: 4 XGBoost Steps Found

| # | Step | File | Lines | XGB Refs | Status |
|---|------|------|-------|----------|--------|
| 1 | `ml_mean_reversion_step` | `ml_reversion_regime_step.py` | 2,097 | 29 | 🔄 In Progress |
| 2 | `ml_smc_regime_step` | `ml_smc_regime_step.py` | 1,865 | 62 | ⬜ Pending |
| 3 | `ml_breakout_bounce_regime_step` | `ml_breakout_bounce_regime_step.py` | 9,825 | 56 | ⬜ Pending |
| 4 | `ml_path_regime_step` | `ml_path_regime_step.py` | 8,708 | 170 | ⬜ Pending |

**Note:** Originally requested 6 steps, but verified that `hmm_ml_alpha_step` and `hmm_macro_regime` do NOT use XGBoost. They only prepare data from HMM outputs. Only 4 steps need migration.

---

## Migration Strategy

Due to large file sizes (2K-10K lines), migration will be done in phases:

### Phase 1: Pilot (ml_mean_reversion_step)
- ✅ Already has optimizations from earlier work
- ✅ Smallest file (2,097 lines)
- ✅ Well-understood structure
- Creates reusable migration pattern

### Phase 2: Medium Steps (ml_smc_regime_step)
- Similar size to mean_reversion (1,865 lines)
- Apply lessons from Phase 1

### Phase 3: Large Steps (ml_breakout_bounce, ml_path)
- Larger files (8K-10K lines)
- More complex, requires careful review
- May need additional customization

---

## Migration Pattern (Standard Template)

For each step, the migration follows this pattern:

### Step 1: Add Import
```python
from src.utils.ml_common.standardized_xgb_trainer import (
    StandardizedXGBTrainer,
    XGBTrainingConfig,
    XGBTrainingResults
)
```

### Step 2: Create OOF Training Method
```python
def _train_xgb_oof(self, X, y, config, market_data, direction="long"):
    """Train XGBoost with OOF predictions using standardized trainer."""

    # Create model ID
    symbol = config.get("symbol", "ETHUSDT")
    exchange = config.get("exchange", "binance")
    timeframe = config.get("timeframe", "15m")
    model_id = f"{symbol}_{exchange}_{timeframe}_{self.step_name}_{direction}"

    # Create trainer
    trainer = StandardizedXGBTrainer(model_id=model_id)

    # Train and get OOF predictions
    results = trainer.train_and_predict(
        X=X,
        y=y,
        data_start=market_data.index.min(),
        data_end=market_data.index.max(),
        eval_metric="logloss",
        verbose=True
    )

    return results
```

### Step 3: Replace Training Call in execute()
```python
# OLD (with data leakage)
model, metrics, raw_scores, calibrated_scores = self._train_xgb_student(X, y, ...)

# NEW (OOF only)
results = self._train_xgb_oof(X, y, config, market_data)
oof_predictions = results.oof_predictions
```

### Step 4: Update Output DataFrame
```python
# Join OOF predictions (only OOF, no training set!)
output_df = output_df.join(oof_predictions, how='left')

# Optionally mark OOF vs. filled
output_df['is_oof'] = ~oof_predictions['probability'].isna()
```

### Step 5: Update Artifact Metadata
```python
metadata = {
    "symbol": symbol,
    "exchange": exchange,
    "timeframe": timeframe,
    "prediction_method": "oof",  # IMPORTANT
    "oof_windows": len(results.metadata),
    "hpo_runs": sum(1 for m in results.metadata if m.get('used_hpo', False)),
    "retrain_interval_days": 10,
    "hpo_interval_days": 30,
}
```

---

## Migration Progress

### ml_mean_reversion_step (PILOT)

**File:** `src/training/steps/market_analysis/ml_reversion_regime_step.py`

**Changes Required:**
1. [ ] Add import for StandardizedXGBTrainer
2. [ ] Create `_train_xgb_oof()` method
3. [ ] Replace `_train_xgb_student()` call in execute()
4. [ ] Update output_df to use OOF predictions only
5. [ ] Update artifact metadata
6. [ ] Remove old training code (optional, for cleanup)

**Estimated Impact:**
- ~50-100 lines changed
- No change to external interface
- OOF predictions instead of all predictions

---

### ml_smc_regime_step

**File:** `src/training/steps/market_analysis/ml_smc_regime_step.py`

**Status:** Pending pilot completion

---

### ml_breakout_bounce_regime_step

**File:** `src/training/steps/market_analysis/ml_breakout_bounce_regime_step.py`

**Status:** Pending pilot completion
**Note:** Largest file (9,825 lines) - may need extra review

---

### ml_path_regime_step

**File:** `src/training/steps/market_analysis/ml_path_regime_step.py`

**Status:** Pending pilot completion
**Note:** Second largest (8,708 lines) - 170 XGB references

---

## Testing Plan

After each migration:

1. **Unit Test:** Run step in isolation
   ```bash
   python -m pytest tests/training/steps/test_<step_name>.py
   ```

2. **Integration Test:** Run full pipeline
   ```bash
   python scripts/run_pipeline.py --steps <step_name>
   ```

3. **Validation Checks:**
   - [ ] No training set predictions in output
   - [ ] OOF predictions align correctly
   - [ ] HPO runs on schedule (every 30 days of data)
   - [ ] Warm start parameters saved
   - [ ] Sparse matrices used when applicable
   - [ ] Memory usage acceptable

4. **Metric Comparison:**
   - Compare OOF metrics vs. old implementation
   - Expect realistic (not inflated) metrics
   - Verify no performance degradation

---

## Expected Timeline

- **ml_mean_reversion_step:** 30-60 minutes (pilot)
- **ml_smc_regime_step:** 20-30 minutes (pattern established)
- **ml_breakout_bounce_regime_step:** 45-60 minutes (large file)
- **ml_path_regime_step:** 45-60 minutes (large file)

**Total:** ~2.5-4 hours for all 4 steps

---

## Rollback Plan

If issues arise, each step can be independently rolled back:

```bash
# Rollback specific file
git checkout HEAD~1 src/training/steps/market_analysis/<step_file>.py

# Or rollback entire migration
git revert <commit_hash>
```

Each step is committed separately for easy rollback.

---

## Next Actions

1. ✅ Clarify historical vs. real-time schedule
2. ✅ Verify warm start is per-step
3. ✅ Identify 4 XGBoost steps (not 6)
4. 🔄 Migrate ml_mean_reversion_step (PILOT)
5. ⬜ Migrate ml_smc_regime_step
6. ⬜ Migrate ml_breakout_bounce_regime_step
7. ⬜ Migrate ml_path_regime_step
8. ⬜ Integration testing
9. ⬜ Production deployment
