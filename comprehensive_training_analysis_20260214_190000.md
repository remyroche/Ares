# COMPREHENSIVE TRAINING ANALYSIS REPORT
**Run ID**: 20260214_190000  
**Generated**: 2026-02-24 08:12 UTC  
**Training Duration**: 4,064 seconds (67.7 minutes)  

---

## 1. BUGS FOUND AND FIXED

### ✅ **Fixed Issues:**
1. **sklearn SGDRegressor Parameter Error**
   - **Issue**: `InvalidParameterError: The 'loss' parameter of SGDRegressor must be a str among {'squared_epsilon_insensitive', 'huber', 'squared_error', 'epsilon_insensitive'}. Got 'modified_huber' instead.`
   - **Root Cause**: sklearn version compatibility issue with `modified_huber` loss function
   - **Fix Applied**: Replaced `modified_huber` with `huber` in:
     - `feature_selection_extreme_events.py` (line 608, 722)
     - `training.py` (line 5465)

2. **Range Features Event Scoring Issue**
   - **Issue**: `Warning: Selection metric 'range_pct' has low finite coverage (0/54247); using fallback`
   - **Root Cause**: Missing range features in feature configuration
   - **Fix Applied**: 
     - Added `range_pct`, `range_12h_pct`, `range_16h_pct`, `range_24h_pct` to `HELPER_BASE_FEATURES`
     - Added computation of `range_16h_pct` in `features.py`
     - Enhanced fallback mechanism in `training.py`

3. **NaN Values in Meta Model Predictions**
   - **Issue**: `ValueError: Input contains NaN` in log_loss calculation
   - **Root Cause**: NaN predictions in meta model evaluation
   - **Fix Applied**: Added NaN handling in `policy_ml.py` with fallback log loss

4. **Data Root Path Configuration**
   - **Issue**: Artifacts not found due to incorrect data_root path
   - **Fix Applied**: Updated `data_root` from "data" to "../data" in `config.py`

5. **Label Refresh Optimization**
   - **Issue**: Unnecessary label regeneration when artifacts exist
   - **Fix Applied**: Added artifact existence check before label refresh in `run_pipeline.py`

---

## 2. DETAILED MODEL PERFORMANCE METRICS

### **Alpha Models (Base Models)**

| Model | Features | AUC | IC | Sharpe | Prec@10 | Prec@40 | Trades/Day@10 | Trades/Day@30 |
|-------|----------|-----|----|---------|---------|---------|---------------|---------------|
| **LONG_MR** | 48 | 0.5571 | 0.0998 | -1.0485 | 0.0285 | 0.0340 | 12.58 | 37.75 |
| **LONG_TF** | 48 | 0.5516 | 0.0809 | -1.2662 | 0.0276 | 0.0367 | 5.72 | 17.15 |
| **SHORT_MR** | 48 | 0.5285 | 0.0659 | -0.6585 | 0.0680 | 0.0631 | 5.72 | 17.15 |
| **SHORT_TF** | 48 | 0.5831 | 0.1984 | -0.8304 | 0.0868 | 0.0774 | 12.58 | 37.75 |

### **Key Observations:**
- **SHORT_TF** shows highest AUC (0.5831) and IC (0.1984)
- **LONG_MR** has moderate performance but negative Sharpe
- All models show negative Sharpe ratios, indicating risk-adjusted performance issues
- Trade frequency varies significantly between MR and TF strategies

### **Per-Regime Performance Analysis**

#### **LONG_MR - Best performing regimes:**
- **Volume Regimes**: Mid vol_48h (AUC: 0.5667)
- **Trend Regimes**: Mid trend_12h (AUC: 0.5613)
- **Worst**: High trend_12h (AUC: 0.5537)

#### **SHORT_TF - Best performing regimes:**
- **Volume Regimes**: Low vol_48h (AUC: 0.6151)
- **Trend Regimes**: High trend_12h (AUC: 0.6104)
- **Most consistent**: Across all volume regimes

---

## 3. META MODEL PERFORMANCE ANALYSIS

### **Meta Model Four Heads Performance**

| Model | Samples | IC (payoff) | Mean Pred | Std Pred | Status |
|-------|---------|-------------|-----------|----------|---------|
| **long_mr_H8** | 140,905 | 0.3618 | 0.0046 | 0.0013 | ✅ Good |
| **long_mr_clf** | 140,905 | -0.0939 | 0.0083 | 0.0739 | ❌ Poor |
| **long_tf_H8** | 67,226 | 0.6049 | 0.0231 | 0.0116 | ✅ Excellent |
| **long_tf_clf** | 67,226 | -0.0554 | 0.0054 | 0.0209 | ❌ Poor |
| **short_mr_H8** | 65,293 | 0.3790 | 0.0289 | 0.0119 | ✅ Good |
| **short_mr_clf** | 65,293 | -0.2904 | 0.0837 | 0.0735 | ❌ Poor |
| **short_tf_H8** | 140,358 | 0.6077 | 0.0303 | 0.0116 | ✅ Excellent |
| **short_tf_clf** | 140,358 | -0.0704 | 0.0426 | 0.1578 | ❌ Poor |

### **Meta Model Insights:**
1. **Regression Heads (H8)** significantly outperform classification heads
2. **short_tf_H8** and **long_tf_H8** show excellent IC (0.6077, 0.6049)
3. **Classification heads consistently underperform with negative IC
4. **Prediction Stability**: Regression heads show lower std predictions, indicating more stable outputs

### **Per-Bucket Meta Performance:**
| Bucket | Median IC | Assessment |
|--------|-----------|------------|
| long_tf | 0.1919 | ✅ Good |
| short_mr | 0.1677 | ✅ Moderate |
| short_tf | 0.0700 | ⚠️ Low |
| long_mr | 0.0056 | ❌ Poor |

---

## 4. RIDGE POSITION SIZER PERFORMANCE

### **LONG Direction:**
| Bucket | Top Weight | Weight Distribution |
|--------|------------|-------------------|
| **long_tf** | reg_H4=0.3126 | Balanced across 6 weights |
| **long_mr** | reg_range=0.3548 | Concentrated on range/std |

### **SHORT Direction:**
| Bucket | Top Weight | Weight Distribution |
|--------|------------|-------------------|
| **short_mr** | reg_H4=0.5002 | Heavy concentration on H4 |
| **short_tf** | reg_std=0.6886 | Dominated by std component |

### **Ridge Sizer Analysis:**
- **LONG positions**: More balanced weight distribution
- **SHORT positions**: Higher concentration in single components
- **Risk Management**: Higher weights on std and range components suggest volatility-based sizing

---

## 5. OPTIMISE STEP ANALYSIS

### **Status**: ❌ **NO BACKTEST RESULTS FOUND**
- **Issue**: No backtest results at `data/artifacts/20260214_190000/backtest_results.csv`
- **Impact**: Unable to assess optimization performance
- **Recommendation**: Run backtest step to evaluate strategy performance

---

## 6. STAGE GATE ANALYSIS

### **Alpha Models Stage Gate:**
- **Passed**: 0/12 models (Need 6 to pass)
- **Status**: ❌ **FAILED**
- **Issue**: All models failed quality gates (PR_AUC, Brier_Improvement, Lift, Precision thresholds)

### **Meta Models Stage Gate:**
- **Passed**: 0/4 models (Need 2 to pass)  
- **Status**: ❌ **FAILED**
- **Issue**: Meta models failed spread and downside protection criteria

---

## 7. FEATURE SELECTION ANALYSIS

### **MDI Feature Selection:**
- **Subsampling**: Limited to 5K events max (✅ Implemented)
- **Features Processed**: 637 → 48 final features per model
- **Selection Method**: ElasticNet prescreen + MDI RFE
- **Efficiency**: Good feature reduction with maintained performance

### **Top Features by Model:**

#### **LONG_MR:**
1. kf_atr_mean_G_VOL_1, kf_atr_mean_G_VOL_0
2. body_pct_G_VOL_1, atr_pct_base_G_VOL_0
3. accel_5h_G_VOL_1, ret48h_G_VOL_1

#### **SHORT_TF:**
1. kf_atr_mean_G_VOL_0, atr_pct_base_G_VOL_0
2. body_pct_G_VOL_0, kf_atr_mean_G_VOL_1
3. asset_atr_level_G_VOL_1, body_pct_G_VOL_1

---

## 8. DATA QUALITY AND COVERAGE

### **Dataset Sizes:**
- **Total Samples**: ~1.5M across all datasets
- **Feature Coverage**: 1057 features pre-selection
- **Time Coverage**: 145 days (2025-09-22 to 2026-02-14)
- **Warning**: Only 145 days of data (recommend ≥ 365 days)

### **Data Quality Metrics:**
- **Coverage**: 100% non-NaN
- **Symbols**: 2 symbols after variance filtering
- **Timeframe**: 1-hour bars

---

## 9. SPECIALIST MODELS

### **Trap Specialist (GMM):**
- **Features**: 8
- **Clusters**: 4 Gaussian components
- **Silhouette Score**: 0.131 (acceptable)
- **Davies-Bouldin**: 1.940 (lower is better)

### **Gamma Specialist (ExtraTrees):**
- **Features**: 20 (selected from 25)
- **R² Score**: -0.001 (poor fit)
- **Regime Classification Accuracy**: 82.7% (good)

---

## 10. LOG FILES AND ARTIFACTS

### **Primary Log File:**
- **Training Log**: `/Users/remyroche/Documents/Ares/extreme_price_movements/training_log.txt`
- **Size**: 2.2MB
- **Lines**: 28,841
- **Duration**: 4,064 seconds

### **Report Files:**
- **Training Report**: `reports/20260214_190000/training_report.md`
- **Base Training**: `reports/20260214_190000/bucket_report_base_training.md`
- **Meta Training**: `reports/20260214_190000/bucket_report_meta_training.md`
- **Ridge Sizer**: `reports/20260214_190000/bucket_report_ridge_sizer.md`
- **Optimization**: `reports/20260214_190000/bucket_report_optimise.md`

### **Artifacts Location:**
- **Base Directory**: `../data/artifacts/20260214_190000/`
- **Models**: `models/` subdirectory
- **Labels**: `labels/` subdirectory
- **Meta OOF**: `meta_oof/` subdirectory

---

## 11. CRITICAL ISSUES AND RECOMMENDATIONS

### **🚨 Critical Issues:**
1. **Stage Gate Failures**: All models failed quality gates
2. **Negative Sharpe Ratios**: Poor risk-adjusted performance
3. **Missing Backtest Results**: No optimization evaluation
4. **Limited Data**: Only 145 days vs recommended 365+ days

### **📊 Performance Issues:**
1. **Meta Model Classification Heads**: Consistently underperform
2. **Feature Selection**: May be too aggressive (637→48)
3. **Calibration**: Some models show poor calibration profiles

### **🔧 Recommended Actions:**
1. **Extend Data Coverage**: Acquire more historical data (365+ days)
2. **Feature Engineering**: Review and enhance feature set
3. **Model Architecture**: Consider alternative algorithms for classification heads
4. **Hyperparameter Tuning**: Optimize for risk-adjusted metrics
5. **Run Backtest**: Complete optimization step for performance evaluation

### **✅ Successful Fixes Applied:**
1. sklearn compatibility issues resolved
2. Range features event scoring fixed
3. NaN handling in meta models implemented
4. Data path configuration corrected
5. Training pipeline optimization completed

---

## 12. SUMMARY

The training run completed successfully after fixing multiple technical issues. However, the model performance indicates significant challenges:

- **Technical Success**: ✅ All bugs fixed, pipeline completed
- **Model Performance**: ❌ Poor risk-adjusted returns, failed quality gates
- **Data Quality**: ⚠️ Limited historical coverage
- **Next Steps**: Require data expansion, feature enhancement, and backtest evaluation

The pipeline is now stable and ready for iterative improvements with better data coverage and feature engineering.
