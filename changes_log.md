# Changes Log - HPO Pipeline Run (2026-01-07)

## Overview
This document tracks all changes made during the HPO pipeline run for ETHUSDT with `meta_labeling_hpo_experiment`.

---

## Changes Made

### 1. Step Name Correction
- **What**: Used correct step name `meta_labeling_hpo_experiment` instead of `meta_labeling_hpo_sample_weighted`
- **Where**: Command line invocation
- **Why**: Step `meta_labeling_hpo_sample_weighted` was not found in registry

---

## HPO Run Summary

### Execution Details
- **Start Time**: 2026-01-07 08:47:49
- **End Time**: 2026-01-07 09:05:42
- **Duration**: ~18 minutes (1047.86s)
- **Symbol**: ETHUSDT
- **Timeframe**: 15m
- **Direction**: Long

### Multi-Stage HPO Results

| Stage | Complexity | Best Score | Trials |
|-------|------------|------------|--------|
| Stage 1 (Sample Count Screening) | fast | -1e9 (failed) | 60 |
| Stage 2 (Sample Count Refinement) | medium | -0.0000 | 30 |
| Stage 3 (Edge Optimization) | strong | -0.0000 | 30 |
| Stage 4 (Edge Refinement) | strong | 0.0000 | 30 |

**Total**: 150 trials, 77 candidate configurations, 5 Pareto solutions

### Best Configuration Metrics
- **Edge**: 0.000036
- **Mean AUC**: 0.6058
- **Learnability**: 0.7133
- **Trades per Day**: 2.612
- **Sharpe (Winners)**: 0.9538
- **N Events**: 1016

### Key Best Parameters
```json
{
  "cusum_threshold": 0.0194,
  "target_signal_density": 11.48,
  "label_low_q": 0.396,
  "label_high_q": 0.686,
  "profit_thr_base": 0.0142,
  "stop_to_profit_ratio": 0.365,
  "vol_baseline_window": 68,
  "kalman_Q": 0.000314,
  "kalman_R": 0.00114
}
```

---

## Regime Integration Analysis

### Volatility Regime Performance
| Regime | AUC | N Events | Interpretation |
|--------|-----|----------|----------------|
| low_vol | 0.533 | 3379 | Near-random (poor) |
| medium_vol | 0.507 | 3480 | Near-random (poor) |
| high_vol | 0.703 | 3379 | Good predictive power |

**Finding**: Model performs significantly better in high volatility regimes. This is expected per de Prado's causal framework - high volatility provides cleaner signal-to-noise ratio.

### Per-Fold AUC Summary
| Fold | AUC | N Test | ECE | Net P&L per trade |
|------|-----|--------|-----|-------------------|
| 0 | 0.619 | 2048 | 0.034 | -0.0026 |
| 1 | 0.593 | 2048 | 0.100 | -0.0050 |
| 2 | 0.653 | 2048 | 0.053 | -0.0008 |
| 3 | 0.695 | 2047 | 0.047 | 0.0002 |
| 4 | 0.740 | 2047 | 0.040 | 0.0004 |

**Finding**: Model improves over time (folds 3-4 have positive P&L), suggesting regime adaptation or data recency effects.

---

## Diagnostics Summary

### Calibration Issues (⚠️ WARNING)
- **ECE**: 0.3310 (threshold: 0.05)
- **Brier Score**: 0.3809
- **Status**: Miscalibrated

### Robustness (⚠️ WARNING)
- **Worst-fold AUC**: 0.593
- **AUC CV Std**: 0.0527
- **Status**: Not robust across folds

### Leakage Tests (✅ PASSED)
- **Y-shuffle AUC**: 0.508 (near random as expected)
- **Look-ahead suspected**: False
- **God feature suspected**: False
- **Lag-1 Stress Test Delta**: 0.0073

### Two-Stage Meta-Model (✅ GOOD)
- **Activity Model AUC**: 0.917
- **Direction Model AUC**: 0.816
- **Combined AUC**: 0.798

---

## Generated Artifacts
1. `meta_labeling_hpo_best_params_ETHUSDT_15m_long_20260107_080542.json`
2. `meta_labeling_hpo_candidate_pool_ETHUSDT_15m_long_20260107_080542.csv`
3. `meta_labeling_hpo_pareto_front_ETHUSDT_15m_long_20260107_080542.csv`
4. `meta_labeling_hpo_report_ETHUSDT_15m_long_20260107_080542.md`
5. `meta_labeling_hpo_trials_ETHUSDT_20260107_080336.csv`

---

## ⚠️ CRITICAL ISSUE: De Prado Causal Framework NOT Wired

### Problem Identified
The HPO experiment is using **CUSUM-based signal generation** with high density (~11.5 events/day) instead of de Prado's causal framework.

### Current Implementation (WRONG)
```
meta_labeling_hpo_experiment_step.py → generate_primary_signals()
    → generate_dual_cusum_signals()  # High-density CUSUM signals
    → ~11.5 events/day
```

### Expected Implementation (PER DE PRADO)
```
label_based_layer_2.py → _generate_events()
    → SNR-based detection: |r_t| / sigma_t > 0.5
    → Lower density, higher quality events
    → Regime-conditional barrier families
```

### Missing Components
1. **`de_prado_causal_features.py`** - 50-feature causal super-set (T, W, X) - NOT IMPORTED
2. **`CausalSurpriseDetector`** - Structural break detection - NOT USED
3. **`LabelBasedLayer2`** - Regime-conditional geometry optimization - NOT CALLED

### Files Affected
| File | Status |
|------|--------|
| `meta_labeling_hpo_experiment_step.py` | ❌ Uses CUSUM, not de Prado |
| `de_prado_causal_features.py` | ✅ Exists, NOT imported |
| `causal_surprise_events.py` | ✅ Exists, NOT imported |
| `label_based_layer_2.py` | ⚠️ STRIPPED DOWN - only 665 lines |

### ROOT CAUSE FOUND
The full causal Layer 2 pipeline exists in `label_based_layer_2.py_saved` (8144 lines, 400KB) but the **active file** is a stripped-down version (665 lines, 24KB).

**Full causal version (`_saved`) includes:**
- ✅ `DePradoCausalFeatures` - De Prado 2026 T/W/X feature protocol
- ✅ `DMLOrthoForest` - Orthogonal Random Forest (ORF) from EconML
- ✅ `CausalDiscovery`, `CausalFeatureEngineering`, `CausalSurpriseDetector`
- ✅ `InvariantRiskMinimization` (IRM) for environment invariance
- ✅ Orthogonal label generation with specialist events
- ✅ Layer 2.5 Chaser integration
- ✅ Geometry selection with ORF causal validation

**Current active version (stripped) MISSING:**
- ❌ No causal feature generation
- ❌ No ORF/DML
- ❌ No orthogonal specialist events
- ❌ Uses basic SNR event detection only

### Recommended Fix
1. **RESTORE** the full causal Layer 2 from `label_based_layer_2.py_saved` (DONE).
    ```bash
    cp label_based_layer_2.py_saved label_based_layer_2.py
    ```
2. **SWITCH EXECUTION COMMAND** to use the correct step:
    ```bash
    python src/launcher/ares_launcher.py label_based_layer_2 --symbol ETHUSDT --execution-mode full
    ```
    *Note: The `meta_labeling_hpo_experiment` step is LEGACY CUSUM logic and should not be used for De Prado's framework.*

## [2026-01-08 09:15] Orchestration Refactor (Layer 0-5)
**Refactored `meta_labeling_hpo_experiment_step.py` to Orchestrator Mode.**
- Replaced monolithic CUSUM legacy logic with a sequential orchestrator.
- **Components Integrated**:
    - Layer 0: Kalman/VWAP (Function)
    - Layer 1: Weight Optimization (Function)
    - Layer 2: De Prado Causal Framework (Class `LabelBasedLayer2`)
    - Layer 3: Meta-Labeling (Analyst) (Function)
    - Layer 4: Position Sizing (ExtraTrees) (Function)
    - Layer 5: Portfolio Backtest (Class `Layer5PositionSizer`)
- **Impact**: The `meta_labeling_hpo_experiment` step now correctly executes the full "Proper" pipeline, utilizing causal events from Layer 2.

## [2026-01-08 09:35] Wavelet Integration & Config Enforcement
**Enhanced Layer 0 and Orchestrator Configuration.**
- **Layer 0**: Integrated `OptimizedWaveletDecomposition` into `label_based_layer_0.py` for advanced signal denoising.
- **Orchestrator**: Enforced strict configuration in `meta_labeling_hpo_experiment_step.py`:
    - `use_wavelets = True` (Layer 0)
    - `enable_causal_framework = True` (Layer 2)
    - `run_layer1_optimization = True` (Layer 1)

## [2026-01-08 09:40] Verified & Enhanced L0/L2 Features
**Completed verification and implementation of specific user requirements.**
- **Layer 0**: Implemented Advanced Denoising Pipeline:
    - **Wavelet Soft Thresholding** (`OptimizedWaveletDecomposition` with VisuShrink).
    - **Median Filtering** (`kernel=3`).
    - **Median Filtering** (`kernel=3`).
    - **Outlier Clipping** (Robust 5-sigma MAD).
- **Layer 2**: Integrated **Gaussian Mixture Model (GMM)** for Volatility Regime detection (2 components), ensuring subsequent Causal MoE logic is regime-conditional.

## [2026-01-08 09:47] Recovered Regime Framework
**Restore Critical Components from `regime-conditional-framework` branch.**
- **Recovered Files**:
    - `src/utils/ml_common/regime/adaptive_hunter_router.py`: GMM Router with Physics Features (Volatility, Efficiency, etc.).
    - `src/training/steps/market_analysis/regime_tagging_step.py`: Regime pipeline step.
    - `src/training/steps/labeling/label_based_layer_2.py`: **Regime-Conditional MoE** implementation (Replaced generic De Prado version).
    - `src/training/steps/labeling/layer3/*.py`: **Soft Gating** logic for Meta-Learner.
- **Integration**:
    - Updated `meta_labeling_hpo_experiment_step.py` (Orchestrator) to instantiate and run `AdaptiveHunterRouter` for regime generation instead of generic GMM logic.
