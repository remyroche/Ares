# Feature Generation Period + Lookback Optimization Report

**Generated:** 2025-10-18T15:52:27.247308
**Artifact Storage Path:** `artifacts/pre_training/artifact_store/period_lookback_optimization/optimization_report.pkl`

## 📊 Execution Summary

- **Status:** completed
- **Data Rows:** 1,162,368
- **Data Columns:** 26
- **Memory Usage:** 383.55 MB

## 🎯 Optimization Results

- **Optimized Periods:** 30
- **Optimized Lookbacks:** 20
- **Method:** consolidated_pipeline

## ⚙️ Configuration

- **Symbol:** ETHUSDT
- **Timeframe:** 15m
- **Direction:** DirectionType.LONGS
- **Min Periods:** 2
- **Correlation Threshold:** 0.85
- **No Recency Bias:** True
- **Top 1 Trading:** True
- **Top 3 Interactions:** True

## 🔧 Step-by-Step Analysis

### ✅ Data Preparation & Validation

**Description:** Data loading, cleaning, and validation for optimization

**Status:** completed | **Duration:** ~0.5s

**Details:**
- **Data Source:** Consolidated parquet files
- **Data Rows:** 1162368
- **Data Columns:** 26
- **Memory Usage Mb:** 383.55
- **Data Quality Checks:**
  - Non-finite value detection and correction
  - Data completeness validation
  - Memory usage optimization
- **Validation Rules:**
  - Min Rows: 100
  - Required Columns: ['open', 'high', 'low', 'close']
  - Data Types: pandas.DataFrame

---

### ✅ Period Optimization

**Description:** Optimization of feature generation periods for maximum historical context

**Status:** completed | **Duration:** ~0.8s

**Details:**
- **Optimized Value:** 30
- **Optimization Method:** consolidated_pipeline
- **Constraints:**
  - Min Periods: 2
  - Correlation Threshold: 0.85
  - No Recency Bias: True
- **Optimization Criteria:**
  - Sufficient historical context
  - Feature stability across periods
  - Correlation threshold compliance
  - Recency bias prevention
- **Result Analysis:** Period length of 30 provides excellent historical context

---

### ✅ Lookback Window Optimization

**Description:** Optimization of lookback windows for feature computation stability

**Status:** completed | **Duration:** ~0.5s

**Details:**
- **Optimized Value:** 20
- **Optimization Method:** consolidated_pipeline
- **Constraints:**
  - Min Lookback: 5
  - Max Lookback: 252
  - Stability Requirement: True
- **Optimization Criteria:**
  - Feature computation stability
  - Sufficient data for rolling calculations
  - Memory efficiency
  - Computational performance
- **Result Analysis:** Lookback window of 20 provides excellent computation stability

---

### ✅ Feature Selection Analysis

**Description:** Analysis of feature selection criteria and constraints

**Status:** completed | **Duration:** ~0.2s

**Details:**
- **Selection Criteria:**
  - Top 1 Trading: True
  - Top 3 Interactions: True
  - Correlation Threshold: 0.85
- **Feature Diversity:**
  - Correlation Threshold: 0.85 (prevents highly correlated features)
  - Interaction Features: Top 3 interactions enabled
  - Trading Features: Top 1 trading features prioritized
- **Quality Metrics:**
  - Feature diversity maintenance
  - Correlation reduction
  - Interaction feature inclusion
  - Trading signal prioritization

---

### ✅ Artifact Storage & Persistence

**Description:** Storage of optimization results and metadata for future use

**Status:** completed | **Duration:** ~0.1s

**Details:**
- **Storage Path:** artifacts/pre_training/artifact_store/period_lookback_optimization
- **Stored Artifacts:**
  - optimized_periods.pkl
  - optimized_lookbacks.pkl
  - optimization_metadata.pkl
  - optimization_report.pkl
  - metadata.json
- **Persistence Method:** Disk + Memory (hybrid storage)
- **Retrieval Method:** Automatic fallback (memory → disk)
- **Metadata Included:**
  - Optimization parameters
  - Configuration settings
  - CMI diagnostics
  - Execution timestamps
  - Data quality metrics

---



## 🧩 Feature-Level Optimization

- **Source:** data_frame
- **Analyzed Features:** 16
- **Global Period (default):** 30
- **Global Lookback (default):** 20

### Top Features by |Pearson Corr| vs returns

| Feature | Lookback | |Pearson Corr| | Non-Null % | Mean | Std | AC(1) |
|---|---:|---:|---:|---:|---:|
| close_return | -1 | 1.0000 | 100.00 | 0.0000 | 0.0009 | -0.0220 |
| close_log_return | -1 | 1.0000 | 100.00 | 0.0000 | 0.0009 | -0.0226 |
| price_range_pct | -1 | 0.0676 | 100.00 | 0.0010 | 0.0010 | 0.6570 |
| body_size_pct | -1 | 0.0642 | 100.00 | 0.0006 | 0.0007 | 0.3587 |
| price_range | -1 | 0.0570 | 100.00 | 2.7241 | 2.6769 | 0.6581 |
| body_size | -1 | 0.0515 | 100.00 | 1.6218 | 1.9796 | 0.3538 |
| quote_volume | -1 | 0.0488 | 100.00 | 733821.1062 | 1326453.7382 | 0.6471 |
| trades | -1 | 0.0244 | 100.00 | 1895.4162 | 2241.8431 | 0.8538 |
| volume_log_return | -1 | 0.0216 | 100.00 | -0.0000 | 0.8481 | -0.4140 |
| volume_return | -1 | 0.0212 | 100.00 | 0.4286 | 1.5070 | -0.2244 |
| open_time | -1 | 0.0038 | 100.00 | 1726924830000.4148 | 3464101615.4783 | 1.0000 |
| close_time | -1 | 0.0038 | 100.00 | 1726924889983.5496 | 3464101615.7267 | 1.0000 |
| hour | -1 | 0.0021 | 100.00 | 11.5085 | 6.9184 | 0.9960 |
| day | -1 | 0.0013 | 100.00 | 16.5273 | 8.6866 | 0.9999 |
| is_weekend | -1 | 0.0007 | 100.00 | 0.2800 | 0.4490 | 0.9995 |
| day_of_week | -1 | 0.0005 | 100.00 | 2.9832 | 1.9989 | 0.9995 |

## 🧠 CMI Analysis

- **CMI Enabled:** ❌ No
- **Reason:** Not in Tactician mode or CMI unavailable

## 💡 Recommendations

- ✅ Good period length - provides sufficient historical context
- ✅ Adequate lookback window for feature computation
- 📊 Standard optimization used (CMI complementarity not available)
- ✅ Recency bias prevention enabled
- ✅ Appropriate correlation threshold for feature diversity

## 🚀 Next Steps

- Use optimized periods and lookbacks in feature generation
- Validate results with cross-validation
- Monitor performance in production
- Consider re-optimization if market conditions change significantly
