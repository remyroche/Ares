# ML Training Optimizations - Clarifications and Integration Plan

## Important Clarifications

### 1. Retraining Scheduler - Training Only, NOT Live Trading

**The `OOFPredictionGenerator` is ONLY for the initial training phase, NOT for live trading.**

**What it does during training:**
- Simulates how the model would perform if retrained regularly
- Creates predictions where each timestamp only uses data available up to that time
- Example: For 5-day XGB retraining schedule:
  - Train on data from start to Day 100 → predict Days 100-105
  - Train on data from start to Day 105 → predict Days 105-110
  - Train on data from start to Day 110 → predict Days 110-115
  - etc.

**Result:** One trained model with OOF predictions that eliminate lookahead bias

**NOT used for:** Live trading model retraining (that's a separate deployment concern)

### 2. The 7 Specialist Models

**All 7 specialists are now covered:**

1. ✅ `hmm_ml_alpha_step.py` - HMM model
2. ✅ `ml_smc_regime_step.py` - XGB model
3. ✅ `ml_breakout_bounce_regime_step.py` - HMM + GMM
4. ✅ `ml_reversion_regime_step.py` - GMM (teacher) + XGB (student)
5. ✅ `ml_liquidity_regime_step.py` - XGB model
6. ✅ `ml_risk_regime_step.py` - HMM with GMM initialization
7. ✅ `ml_path_regime_step.py` - Uses default burn-in (no hardcoded value)

### 3. Which Models Use GMM/HMM

**GMM Models:**
- `ml_reversion_regime_step.py` - GMM teacher model for mean reversion
- `ml_risk_regime_step.py` - GMM for HMM initialization
- `ml_breakout_bounce_regime_step.py` - GMM components
- `ml_path_regime_step.py` - GMM for regime clustering

**HMM Models:**
- `hmm_ml_alpha_step.py` - Main HMM for alpha signals
- `ml_risk_regime_step.py` - Main HMM with GMM warm-start
- `ml_breakout_bounce_regime_step.py` - HMM for regime detection
- `ml_path_regime_step.py` - HMM for path regimes

## Integration Issues to Address

### Issue 1: Memory Optimizations Should Be in Unified Training Pipeline

**Current:** Standalone `training_optimizations.py` file
**Should be:** Integrated into `unified_models_training_step.py`

**Action needed:**
- Add memory optimization to data loading in unified training
- Apply precision reduction before training starts
- Integrate into existing data preprocessing flow

### Issue 2: Histogram/GOSS Should Be in Per-Model Training Setup

**Current:** Standalone functions in `training_optimizations.py`
**Should be:** Integrated into existing HPO/model configuration

**Action needed:**
- Add histogram parameters to XGBoost model configs
- Add GOSS parameters to LightGBM model configs
- Integrate into `hpo_config.py` or per-model parameter definitions

## Proposed Integration Plan

### Phase 1: Keep Infrastructure Modules ✅
These are reusable utilities (keep as-is):
- ✅ `retraining_scheduler.py` - OOF generation framework
- ✅ `optimization/local_search_hpo.py` - Adaptive HPO
- ✅ `gmm_semantic_sorting.py` - GMM utilities
- ✅ `hmm_warm_start.py` - HMM utilities

### Phase 2: Integrate Memory Optimizations into Unified Training
**File:** `unified_models_training_step.py`

**Location:** After data loading, before training

```python
# In execute() method, after loading training_data
if training_data is not None:
    # Optimize memory usage
    from src.utils.ml_common.training_optimizations import optimize_dataframe_memory
    training_data = optimize_dataframe_memory(training_data)
    tprint_info(f"Memory optimized: {training_data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
```

### Phase 3: Integrate Histogram/GOSS into Model Configs
**File:** `hpo_config.py` or per-model training logic

**For XGBoost models:**
```python
# Add to base XGBoost parameters
xgb_base_params = {
    'tree_method': 'hist',  # Enable histogram binning
    'max_bin': 256,
    ...existing params...
}
```

**For LightGBM models:**
```python
# Add to base LightGBM parameters
lgb_base_params = {
    'boosting_type': 'goss',  # Enable GOSS
    'top_rate': 0.2,
    'other_rate': 0.1,
    'max_bin': 255,
    ...existing params...
}
```

## What Can Stay as Standalone Modules

**Reusable Utilities (Good as standalone):**
1. `retraining_scheduler.py` - Used by multiple models
2. `optimization/local_search_hpo.py` - Used by XGB models
3. `gmm_semantic_sorting.py` - Used by GMM models
4. `hmm_warm_start.py` - Used by HMM models

**Should be integrated:**
1. Memory optimization → `unified_models_training_step.py`
2. Histogram binning → XGBoost model configs
3. GOSS → LightGBM model configs

## Revised File Structure

```
src/utils/ml_common/
├── retraining_scheduler.py          ✅ Keep - reusable OOF framework
├── gmm_semantic_sorting.py          ✅ Keep - GMM utilities
├── hmm_warm_start.py                ✅ Keep - HMM utilities
├── optimization/
│   └── local_search_hpo.py          ✅ Keep - adaptive HPO
└── training_optimizations.py        ⚠️  Extract and integrate:
                                        - Memory opts → unified training
                                        - Histogram/GOSS → model configs
```

## Updated Summary of Changes

### ✅ Completed (Keep As-Is)
1. Reduced burn-in period (1/6 → 1/12) in all 7 specialists
2. Retraining scheduler framework (for OOF training)
3. Adaptive local search HPO (for XGB models)
4. GMM semantic sorting (for GMM models)
5. HMM warm start (for HMM models)
6. Documentation

### 🔧 Needs Integration
1. Memory optimizations → Integrate into `unified_models_training_step.py`
2. Histogram binning → Add to XGBoost model base parameters
3. GOSS → Add to LightGBM model base parameters

### 📋 Next Actions

**Option A: Keep current structure**
- Users manually import and use `training_optimizations.py` functions
- Provides flexibility but requires manual integration

**Option B: Integrate into existing pipeline (Recommended)**
- Memory optimization in unified training data loading
- Histogram/GOSS in default model parameters
- Transparent to users, automatic optimization

**Which approach do you prefer?**

## Model-Specific Integration Targets

### XGB Models (Use Histogram + Adaptive HPO)
- `ml_smc_regime_step.py`
- `ml_reversion_regime_step.py` (student model)
- `ml_liquidity_regime_step.py`
- `ml_breakout_bounce_regime_step.py`

### GMM Models (Use Semantic Sorting + Warm Start)
- `ml_reversion_regime_step.py` (teacher model)
- `ml_risk_regime_step.py` (for initialization)
- `ml_breakout_bounce_regime_step.py`
- `ml_path_regime_step.py`

### HMM Models (Use Warm Start)
- `hmm_ml_alpha_step.py`
- `ml_risk_regime_step.py` (main model)
- `ml_breakout_bounce_regime_step.py`
- `ml_path_regime_step.py`

### Analyst Models (Use All Optimizations)
- Analyst Base: Memory + Histogram/GOSS + Sample Weighting
- Analyst Ensemble: Sample Weighting (2-month half-life)
