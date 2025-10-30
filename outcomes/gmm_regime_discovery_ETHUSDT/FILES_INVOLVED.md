# Files Involved in GMM Regime Discovery Run

**Date**: 2025-10-30  
**Final Run**: `gmm_regime_discovery_ETHUSDT_20251030_212536`

---

## 1. Modified/Updated Files

### Primary Implementation
**Location**: `src/training/steps/market_analysis/gmm_clustering/`

1. **`gmm_regime_discovery_step.py`** ⭐ (MODIFIED)
   - Main GMM regime discovery implementation
   - **Changes Made**:
     - Added integration with `clustering_optimization_goals.py`
     - Added integration with `cluster_quality_assessor.py`
     - Limited PCA to 20 components (was 50)
     - Added post-PCA normalization
     - Added temporal metrics support (timestamps)
     - Enhanced report generation with optimization targets
   - **Lines**: 1,033 total

---

## 2. Dependencies - Core Modules

### Market Analysis / Clustering Components

2. **`src/training/steps/market_analysis/clusters/cluster_quality_assessor.py`** ⭐
   - Unified cluster quality assessment
   - **Used For**:
     - Silhouette score calculation
     - Davies-Bouldin Index
     - Calinski-Harabasz Index
     - Within/Between regime CV metrics
     - Temporal smoothness calculation
     - Regime persistence calculation
     - Per-regime metrics
     - Economic interpretation
   - **Lines**: ~2,200 (large file)

3. **`src/training/steps/market_analysis/clusters/clustering_optimization_goals.py`** ⭐
   - Optimization goals and targets configuration
   - **Used For**:
     - Target cluster count (4-6)
     - Quality thresholds (silhouette, CV, temporal)
     - Economic targets (Sharpe, drawdown)
     - Penalty configurations
     - Cross-validation configs
   - **Lines**: ~2,000

### Base Infrastructure

4. **`src/training/steps/base_step.py`**
   - Base class for all training steps
   - Provides artifact manager, logging, error handling
   - **Used For**: Step initialization and lifecycle

5. **`src/utils/data/klines_parquet.py`**
   - Parquet-based klines data manager
   - **Used For**: Loading market data (ETHUSDT 1h)
   - Loaded: 26,277 records (full history)

6. **`src/utils/artifact_manager.py`**
   - Artifact storage and retrieval system
   - **Used For**: 
     - Loading generated features
     - Saving regime labels (attempted)
     - Context management

### Utility Modules

7. **`src/utils/tprint.py`**
   - Terminal printing utilities
   - **Functions Used**: `tprint`, `tprint_info`, `tprint_timer`, `tprint_success`, `tprint_warning`, `tprint_error`

8. **`src/utils/logger.py`**
   - System logging infrastructure
   - **Used For**: Error logging, debug logging

---

## 3. Data Files - Input

### Market Data
9. **`historical_data/binance/ETHUSDT/1h/processed/*.parquet`**
   - 40 parquet files loaded
   - Combined: 26,277 records
   - Date range: 1970-01-01 to 2025-09-13
   - **Aligned**: 480 records with features

### Feature Data
10. **`artifacts/feature_generation_feature_generation_step_generated_features_1h_long_Analyst_20251026_131017.parquet`**
    - Generated features from step 2
    - 300 features per sample
    - 480 samples (aligned with market data)
    - **Used For**: Input to GMM clustering

---

## 4. Output Files - Generated

### Reports (Markdown)

11. **`outcomes/gmm_regime_discovery_ETHUSDT/gmm_regime_discovery_report_ETHUSDT_20251030_200015.md`**
    - First run (50 PCs, no temporal metrics)
    - Quality score: 0.800

12. **`outcomes/gmm_regime_discovery_ETHUSDT/gmm_regime_discovery_report_ETHUSDT_20251030_201945.md`**
    - Second run (50 PCs, with temporal metrics)
    - Quality score: 0.840

13. **`outcomes/gmm_regime_discovery_ETHUSDT/gmm_regime_discovery_report_ETHUSDT_20251030_212536.md`** ⭐ **FINAL**
    - Third run (20 PCs, normalized, with temporal metrics)
    - Quality score: 0.811
    - **Best cohesion**: Within-CV = 11.66

### Analysis Documents

14. **`outcomes/gmm_regime_discovery_ETHUSDT/GMM_REGIME_ANALYSIS_DETAILED.md`**
    - Detailed analysis of initial results
    - Addressed all user questions
    - Identified issues and solutions

15. **`outcomes/gmm_regime_discovery_ETHUSDT/HIGH_FEATURE_CV_ANALYSIS.md`**
    - Deep dive into high feature CV problem
    - Explained why large regimes had high variance
    - Recommended solutions

16. **`outcomes/gmm_regime_discovery_ETHUSDT/20PC_IMPROVEMENT_ANALYSIS.md`** ⭐
    - Comparison of 50 PCs vs 20 PCs
    - Shows 69% reduction in Within-CV
    - Final recommendations

### Temporary Files (Deleted)

17. **`run_gmm_regime_discovery.py`** (DELETED)
    - Simple runner script for testing
    - Successfully executed 3 times
    - Removed after completion

---

## 5. External Libraries Used

### Machine Learning
- **scikit-learn**: 
  - `GaussianMixture` (GMM algorithm)
  - `StandardScaler` (feature normalization)
  - `PCA` (dimensionality reduction)
  - `silhouette_score`, `davies_bouldin_score`, `calinski_harabasz_score` (quality metrics)

### Data Processing
- **pandas**: DataFrames, time series handling
- **numpy**: Numerical operations, array processing

### Python Standard Library
- `logging`, `time`, `datetime`, `typing`, `pathlib`, `warnings`

---

## 6. Configuration Files

### Optimization Goals
**Source**: `clustering_optimization_goals.py`

**Targets Used**:
- Target cluster count: 4-6
- Min silhouette: 0.10
- Min temporal smoothness: 0.60
- Min CV score: 1.20
- Min cluster size: 2.0%
- Max cluster size: 20.0%

### Quality Assessment
**Source**: `cluster_quality_assessor.py`

**Weights Used**:
- CV Ratio: 30%
- Temporal Smoothness: 30%
- Silhouette: 20%
- Balance: 10%
- Noise Ratio: 10%

---

## 7. File Modification Summary

### Files Modified (1)
- ✅ `src/training/steps/market_analysis/gmm_clustering/gmm_regime_discovery_step.py`

### Files Created (4)
- ✅ `outcomes/gmm_regime_discovery_ETHUSDT/gmm_regime_discovery_report_ETHUSDT_20251030_200015.md`
- ✅ `outcomes/gmm_regime_discovery_ETHUSDT/gmm_regime_discovery_report_ETHUSDT_20251030_201945.md`
- ✅ `outcomes/gmm_regime_discovery_ETHUSDT/gmm_regime_discovery_report_ETHUSDT_20251030_212536.md`
- ✅ `outcomes/gmm_regime_discovery_ETHUSDT/GMM_REGIME_ANALYSIS_DETAILED.md`
- ✅ `outcomes/gmm_regime_discovery_ETHUSDT/HIGH_FEATURE_CV_ANALYSIS.md`
- ✅ `outcomes/gmm_regime_discovery_ETHUSDT/20PC_IMPROVEMENT_ANALYSIS.md`

### Files Deleted (1)
- ✅ `run_gmm_regime_discovery.py` (temporary test script)

---

## 8. Execution Flow

```
run_gmm_regime_discovery.py (temporary)
    └── gmm_regime_discovery_step.py
            ├── IMPORTS
            │   ├── clustering_optimization_goals.py (goals/targets)
            │   ├── cluster_quality_assessor.py (quality metrics)
            │   ├── base_step.py (step infrastructure)
            │   └── tprint.py, logger.py (utilities)
            │
            ├── DATA LOADING
            │   ├── klines_parquet.py
            │   │   └── historical_data/binance/ETHUSDT/1h/processed/*.parquet (40 files)
            │   └── artifact_manager.py
            │       └── artifacts/feature_generation_*_20251026_131017.parquet
            │
            ├── PROCESSING
            │   ├── CorrelationBasedFeatureSelector (300 → 171 features)
            │   ├── StandardScaler (normalize to mean=0, std=1)
            │   ├── PCA (171 → 20 components, 62.1% variance)
            │   ├── StandardScaler (re-normalize PCs to mean=0, std=1)
            │   └── GaussianMixture (fit with k=6 components)
            │
            ├── QUALITY ASSESSMENT
            │   └── cluster_quality_assessor.assess_quality()
            │       ├── Calculate silhouette, DBI, CH scores
            │       ├── Calculate within/between CV
            │       ├── Calculate temporal smoothness & persistence
            │       └── Calculate per-regime metrics
            │
            └── OUTPUT
                ├── outcomes/gmm_regime_discovery_ETHUSDT/gmm_regime_discovery_report_*.md
                └── Console logs (via tprint)
```

---

## 9. Key Dependencies by Function

### Clustering Algorithm
- **`sklearn.mixture.GaussianMixture`**: Core GMM algorithm

### Feature Engineering
- **`sklearn.preprocessing.StandardScaler`**: Normalization (2 instances)
- **`sklearn.decomposition.PCA`**: Dimensionality reduction
- **`pandas.DataFrame.corr()`**: Correlation analysis for feature reduction

### Quality Assessment
- **`sklearn.metrics.silhouette_score`**: Cluster cohesion
- **`sklearn.metrics.davies_bouldin_score`**: Cluster separation
- **`sklearn.metrics.calinski_harabasz_score`**: Variance ratio
- **`cluster_quality_assessor.py`**: Comprehensive quality calculation

### Data Access
- **`src.utils.data.klines_parquet.get_klines_manager()`**: Market data loader
- **`src.training.steps.base_step.artifact_manager`**: Feature artifact access

---

## 10. Directory Structure

```
Ares/
├── src/
│   ├── training/
│   │   └── steps/
│   │       ├── base_step.py
│   │       └── market_analysis/
│   │           ├── clusters/
│   │           │   ├── cluster_quality_assessor.py ⭐
│   │           │   └── clustering_optimization_goals.py ⭐
│   │           └── gmm_clustering/
│   │               ├── __init__.py
│   │               └── gmm_regime_discovery_step.py ⭐ (MODIFIED)
│   └── utils/
│       ├── data/
│       │   └── klines_parquet.py
│       ├── tprint.py
│       └── logger.py
│
├── historical_data/
│   └── binance/
│       └── ETHUSDT/
│           └── 1h/
│               └── processed/
│                   └── *.parquet (40 files) 📊
│
├── artifacts/
│   └── feature_generation_feature_generation_step_generated_features_1h_long_Analyst_20251026_131017.parquet 📊
│
└── outcomes/
    └── gmm_regime_discovery_ETHUSDT/
        ├── gmm_regime_discovery_report_ETHUSDT_20251030_200015.md
        ├── gmm_regime_discovery_report_ETHUSDT_20251030_201945.md
        ├── gmm_regime_discovery_report_ETHUSDT_20251030_212536.md ⭐ (FINAL)
        ├── GMM_REGIME_ANALYSIS_DETAILED.md
        ├── HIGH_FEATURE_CV_ANALYSIS.md
        └── 20PC_IMPROVEMENT_ANALYSIS.md ⭐
```

---

## 11. File Roles Summary

### Input Files (Data)
| File | Role | Records |
|------|------|---------|
| `historical_data/.../ETHUSDT/1h/processed/*.parquet` | Market OHLCV data | 26,277 |
| `artifacts/...generated_features_1h...parquet` | Feature matrix | 480 × 300 |

### Processing Files (Code)
| File | Role | Modified? |
|------|------|-----------|
| `gmm_regime_discovery_step.py` | Main implementation | ✅ YES |
| `cluster_quality_assessor.py` | Quality metrics | No (used as-is) |
| `clustering_optimization_goals.py` | Optimization targets | No (used as-is) |
| `base_step.py` | Base infrastructure | No |
| `klines_parquet.py` | Data loading | No |

### Output Files (Reports)
| File | Description | Status |
|------|-------------|--------|
| `gmm_regime_discovery_report_*_200015.md` | Run 1 (50 PCs, no temporal) | Superseded |
| `gmm_regime_discovery_report_*_201945.md` | Run 2 (50 PCs, with temporal) | Superseded |
| `gmm_regime_discovery_report_*_212536.md` | Run 3 (20 PCs, normalized) | ⭐ **FINAL** |
| `GMM_REGIME_ANALYSIS_DETAILED.md` | Detailed analysis | Reference |
| `HIGH_FEATURE_CV_ANALYSIS.md` | CV investigation | Reference |
| `20PC_IMPROVEMENT_ANALYSIS.md` | Improvement comparison | ⭐ **KEY** |

---

## 12. Execution Summary

### Total Files Involved: **~20 files**

**Categories**:
- **1 file modified** (gmm_regime_discovery_step.py)
- **2 core dependencies** (cluster_quality_assessor.py, clustering_optimization_goals.py)
- **40 data files** (historical parquet files)
- **1 feature file** (generated features artifact)
- **6 output files** (3 reports + 3 analysis docs)
- **~10 utility modules** (base_step, tprint, logger, etc.)

### Key Integrations
1. ✅ `clustering_optimization_goals.py` → Defines targets and constraints
2. ✅ `cluster_quality_assessor.py` → Calculates comprehensive metrics
3. ✅ `klines_parquet.py` → Loads market data
4. ✅ `artifact_manager` → Retrieves feature data

---

## 13. For Reproducibility

### To Run This Exact Configuration:

```python
from src.training.steps.market_analysis.gmm_clustering.gmm_regime_discovery_step import (
    create_gmm_regime_discovery_step
)

# Configuration
config = {
    'symbol': 'ETHUSDT',
    'exchange': 'binance',
    'timeframe': '1h',
    'execution_mode': 'light'
}

# Create step with optimized parameters
gmm_step = create_gmm_regime_discovery_step(
    n_components_range=(4, 6),        # From optimization_targets.target_clusters
    correlation_threshold=0.85,       # Remove correlated features
    random_state=42                   # Reproducibility
)

# Execute
results = await gmm_step.execute(config)
```

### Key Settings in Code:
- **Line 834**: `max_pcs = 20` (limit to 20 PCs)
- **Line 841**: Post-PCA normalization with `StandardScaler()`
- **Line 161-163**: Timestamp extraction for temporal metrics
- **Line 822-830**: Feature normalization verification

---

## 14. Files NOT Involved

### Not Used (but exist in project):
- ❌ `iterative_optimization.py` - Not used (this uses GMM, not HDBSCAN iterative)
- ❌ Any HDBSCAN-related files
- ❌ `regime_clustering_step.py` (different from GMM)
- ❌ Model training files (only regime discovery, no model training)
- ❌ Backtesting files (no economic validation yet)

---

## 15. File Dependency Graph

```
gmm_regime_discovery_step.py (MODIFIED)
│
├─→ clustering_optimization_goals.py
│   ├── DEFAULT_CLUSTERING_GOALS
│   └── DEFAULT_OPTIMIZATION_TARGETS
│
├─→ cluster_quality_assessor.py
│   ├── ClusterQualityAssessor
│   ├── ClusterQualityMetrics
│   └── create_cluster_quality_assessor()
│
├─→ base_step.py
│   ├── BaseStep (inheritance)
│   └── artifact_manager
│
├─→ klines_parquet.py
│   └── get_klines_manager()
│       └── Load: historical_data/binance/ETHUSDT/1h/processed/*.parquet
│
├─→ sklearn
│   ├── GaussianMixture
│   ├── StandardScaler (×2 instances)
│   ├── PCA
│   └── Quality metrics (silhouette, DBI, CH)
│
└─→ Utils
    ├── tprint.py (logging)
    └── logger.py (error handling)
```

---

## Summary Table

| Category | Count | Key Files |
|----------|-------|-----------|
| **Modified** | 1 | gmm_regime_discovery_step.py |
| **Core Dependencies** | 2 | cluster_quality_assessor.py, clustering_optimization_goals.py |
| **Data Files (Input)** | 41 | 40 parquet + 1 features artifact |
| **Output Reports** | 6 | 3 regime reports + 3 analysis docs |
| **Utility Modules** | ~10 | base_step, tprint, logger, klines_parquet, etc. |
| **External Libraries** | 3 | scikit-learn, pandas, numpy |

**Total Unique Files: ~20**

---

*This document tracks all files involved in the GMM regime discovery optimization run.*

