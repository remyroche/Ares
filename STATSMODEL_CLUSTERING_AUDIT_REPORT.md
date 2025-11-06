# Statsmodel Clustering Pipeline Audit Report

**Date:** 2025-11-06
**Scope:** Complete audit of statsmodel clustering pipeline from ares_launcher to MarkovRegressionAdapter and ClusterQualityAssessor

---

## Executive Summary

This audit identified **5 critical issues** that prevent the statsmodel clustering pipeline from functioning correctly:

1. ✅ **FIXED:** Configuration values were swapped
2. ❌ **CRITICAL:** Step not included in MARKET_ANALYSIS stage
3. ❌ **CRITICAL:** Result type mismatch causing runtime errors
4. ❌ **MISSING:** CSV report generation not implemented
5. ❌ **INCOMPLETE:** Regime probabilities not properly passed through pipeline

---

## Pipeline Flow

### 1. Entry Point: ares_launcher.py

**Location:** `src/launcher/ares_launcher.py`

**Expected usage:**
```bash
python ares_launcher.py --step statsmodel_clustering_pipeline --symbol ETHUSDT --timeframe 1h --execution-mode light
```

**Issue #1:** Step is registered but **NOT included in MARKET_ANALYSIS stage** (lines 170-176)

```python
'MARKET_ANALYSIS': [
    'sr_detection', 'sr_clustering', 'sr_parameter_optimization',
    'hdbscan_regime_discovery',
    'gmm_regime_discovery',
    'regime_feature_selection',
    'regime_models_training', 'regime_ensemble_training'
    # ❌ MISSING: 'statsmodel_clustering_pipeline'
],
```

---

### 2. Configuration: cluster_features.config

**Location:** `src/training/steps/market_analysis/statsmodel_clustering/cluster_features.config`

**Issue #2:** ✅ **FIXED** - Period values were swapped

| Mode | Expected | Was | Fixed |
|------|----------|-----|-------|
| blank | 180 days | 20 days | ✅ 180 days |
| light | 20 days | 180 days | ✅ 20 days |
| full | Use ares_launcher period | Use ares_launcher period | ✅ Correct |

---

### 3. Pipeline Step: statsmodel_clustering_pipeline_step.py

**Location:** `src/training/steps/market_analysis/statsmodel_clustering_pipeline_step.py`

#### Issue #3: Result Type Mismatch (CRITICAL)

**Line 705-708:**
```python
result = self.markov_adapter.fit(X)  # Returns MarkovRegressionResult dataclass

# ❌ WRONG: Treats dataclass as dictionary
regime_labels = result.get('regime_labels', result.get('filtered_marginal_probabilities', None))
```

**Problem:**
- `MarkovRegressionAdapter.fit()` returns a `MarkovRegressionResult` **dataclass**
- Dataclass has attribute `cluster_labels`, not `regime_labels`
- Code tries to use `.get()` method which doesn't exist on dataclass
- **This will cause AttributeError at runtime**

**Expected fix:**
```python
result = self.markov_adapter.fit(X)

# ✅ CORRECT: Access dataclass attributes directly
cluster_labels = result.cluster_labels

if cluster_labels is None or len(cluster_labels) == 0:
    raise ValueError("Markov adapter did not return cluster labels")

# If we got probabilities, convert to hard labels
if len(cluster_labels.shape) > 1:
    cluster_labels = cluster_labels.argmax(axis=1)

n_regimes = len(np.unique(cluster_labels))

return {
    'regime_labels': cluster_labels,  # Rename for consistency downstream
    'n_regimes': n_regimes,
    'model': result.fitted_model,
    'regime_probabilities': result.cluster_probabilities,
    'transition_matrix': result.transition_matrix,
    'regime_means': result.regime_params,
    'regime_covariances': result.regime_params,
    'aic': result.aic,
    'bic': result.bic,
    'log_likelihood': result.log_likelihood
}
```

#### Issue #4: Missing CSV Report Generation (CRITICAL)

**Line 231-240:**
```python
# Step 5: Assess cluster quality
quality_metrics = await self._assess_cluster_quality(
    features=transformed_features,
    labels=clustering_result['regime_labels'],
    market_data=market_data,
    config=config
)

tprint(f"✅ Quality assessment completed", "SUCCESS")

# ❌ MISSING: No CSV report generation!
# ClusterQualityAssessor has generate_comprehensive_csv_report() method
# but it's NEVER called
```

**Expected fix:**
```python
# Step 5: Assess cluster quality
quality_metrics = await self._assess_cluster_quality(
    features=transformed_features,
    labels=clustering_result['regime_labels'],
    market_data=market_data,
    config=config
)

# ✅ ADD: Generate CSV report
symbol = config.get('symbol', 'ETHUSDT')
assessor = create_cluster_quality_assessor(artifact_manager=self.artifact_manager)
csv_quality_path, csv_trials_path = assessor.generate_comprehensive_csv_report(
    metrics=quality_metrics,
    all_trials=None,  # Could pass if we have multiple trials
    symbol=symbol,
    output_dir='outcomes',
    method_specific_config={
        'k_regimes': clustering_result['n_regimes'],
        'aic': clustering_result.get('aic'),
        'bic': clustering_result.get('bic')
    }
)

if csv_quality_path:
    tprint(f"✅ Quality report saved to: {csv_quality_path}", "SUCCESS")
    artifacts.append(csv_quality_path)
```

#### Issue #5: Incomplete Regime Probabilities Handling

**Line 761-767:**
```python
quality_metrics = assessor.assess_clustering_quality(
    features=features.values,
    cluster_labels=labels,
    market_data=market_data,
    regime_probabilities=None,  # ❌ Always None!
    config=config
)
```

**Problem:**
- Regime probabilities are available from MarkovRegressionAdapter
- But they're not passed to quality assessment
- This reduces quality of the assessment

**Expected fix:**
```python
quality_metrics = assessor.assess_clustering_quality(
    features=features.values,
    cluster_labels=labels,
    market_data=market_data,
    regime_probabilities=clustering_result.get('regime_probabilities'),  # ✅ Pass probabilities
    config=config
)
```

---

### 4. Clustering Engine: MarkovRegressionAdapter

**Location:** `src/training/steps/market_analysis/statsmodel_clustering/core/markov_regression_adapter.py`

#### Verification: ✅ Correctly Clusters All Samples

**Lines 874-1028: `fit()` method**

The adapter correctly:
1. ✅ Validates input (lines 897-898, 1110-1131)
2. ✅ Preprocesses all data (lines 909-911)
3. ✅ Fits model on ALL samples (lines 931-941)
4. ✅ Returns labels for ALL samples (lines 945-946, 1179-1186)

**Return value structure:**
```python
MarkovRegressionResult(
    fitted_model=self.model,
    cluster_labels=labels,              # ✅ All samples
    cluster_probabilities=probabilities, # ✅ All samples
    n_regimes=self.config.k_regimes,
    transition_matrix=transition_matrix,
    regime_params=regime_params,
    # ... more fields
    success=True
)
```

**Batch processing for large datasets:**
- Lines 900-906: Detects large datasets (>10k samples)
- Lines 1309-1440: Implements batch processing
- ✅ Properly combines results from all batches

---

### 5. Quality Assessment: ClusterQualityAssessor

**Location:** `src/training/steps/market_analysis/clusters/cluster_quality_assessor.py`

#### Verification: ✅ Has CSV Generation Capability

**Method:** `generate_comprehensive_csv_report()` (lines 2758-2803)

The assessor provides:
1. ✅ Quality metrics CSV (line 2785)
2. ✅ All trials CSV (line 2790)
3. ✅ Comprehensive metrics including:
   - Core quality scores
   - Feature distribution (CV metrics)
   - Economic validation
   - Temporal analysis
   - Predictive power

**CSV format:**
```csv
Metric Category,Metric Name,Value,Description,Interpretation
Core Quality,Composite Quality Score,0.842,Overall clustering quality,Excellent >0.8
Core Quality,Silhouette Score,0.653,Cluster separation and cohesion,Good >0.5
...
```

**Issue:** Method exists but is **NEVER CALLED** in the pipeline!

---

## Summary of Issues

| # | Issue | Severity | Status | Impact |
|---|-------|----------|--------|--------|
| 1 | Config values swapped | High | ✅ FIXED | Wrong data periods used |
| 2 | Step not in MARKET_ANALYSIS stage | **CRITICAL** | ❌ OPEN | Can't run via stage command |
| 3 | Result type mismatch | **CRITICAL** | ❌ OPEN | Runtime AttributeError |
| 4 | CSV report not generated | **CRITICAL** | ❌ OPEN | No quality report output |
| 5 | Regime probabilities not passed | Medium | ❌ OPEN | Reduced quality assessment accuracy |

---

## Required Fixes

### Priority 1: Critical Fixes (Runtime Blockers)

1. **Add step to MARKET_ANALYSIS stage** (`src/launcher/ares_launcher.py:170-176`)
2. **Fix result access** (`src/training/steps/market_analysis/statsmodel_clustering_pipeline_step.py:705-730`)
3. **Add CSV report generation** (`src/training/steps/market_analysis/statsmodel_clustering_pipeline_step.py:240-250`)

### Priority 2: Enhancement Fixes

4. **Pass regime probabilities** (`src/training/steps/market_analysis/statsmodel_clustering_pipeline_step.py:761-767`)

---

## Expected Outcomes After Fixes

### Outcome 1: Clustered Artifact ✅ VERIFIED

The pipeline will generate an artifact containing:
- All market data samples
- Regime labels for each sample (via MarkovRegressionAdapter)
- Saved via `_save_artifact()` (line 802-814)

**Artifact name:** `statsmodel_clustered_data`

### Outcome 2: Quality Report CSV ❌ MISSING (Need to add)

After fix, the pipeline will generate:
- CSV report with comprehensive quality metrics
- Saved to `outcomes/cluster_quality_metrics_{symbol}_{timestamp}.csv`
- Contains 30+ quality metrics with interpretations

---

## Testing Recommendations

### Test 1: Single Step Execution
```bash
python ares_launcher.py \
  --step statsmodel_clustering_pipeline \
  --symbol ETHUSDT \
  --timeframe 1h \
  --execution-mode light
```

**Expected:**
1. ✅ Loads 20 days of data (light mode)
2. ✅ Generates features
3. ✅ Clusters all samples via MarkovRegressionAdapter
4. ✅ Generates clustered data artifact
5. ✅ Generates quality metrics CSV report

### Test 2: Stage Execution (After Fix)
```bash
python ares_launcher.py \
  --stage MARKET_ANALYSIS \
  --symbol ETHUSDT \
  --execution-mode blank
```

**Expected:**
1. ✅ Runs all MARKET_ANALYSIS steps including statsmodel_clustering_pipeline
2. ✅ Uses 180 days of data (blank mode)

### Test 3: Full Mode
```bash
python ares_launcher.py \
  --step statsmodel_clustering_pipeline \
  --symbol ETHUSDT \
  --execution-mode full
```

**Expected:**
1. ✅ Uses ALL available data (as determined by ares_launcher)
2. ✅ May trigger batch processing for large datasets (>10k samples)

---

## Code Quality Notes

### Strengths ✅
1. Well-structured pipeline with clear steps
2. Comprehensive MarkovRegressionAdapter with hardware optimization
3. Excellent quality assessor with 30+ metrics
4. Proper artifact management integration
5. Good error handling and logging

### Gaps ❌
1. Result type mismatch indicates insufficient integration testing
2. Missing CSV report call suggests incomplete implementation
3. Stage registration oversight indicates missing documentation

---

## Recommendations

### Immediate
1. Apply all Priority 1 fixes
2. Add integration tests for complete pipeline flow
3. Add documentation for running the pipeline

### Short-term
1. Apply Priority 2 fixes
2. Add unit tests for result conversion
3. Add validation for MarkovRegressionResult → dict conversion

### Long-term
1. Consider adding hyperparameter optimization for k_regimes
2. Add model comparison (different k_regimes values)
3. Add regime interpretation reports (what each regime represents)

---

## Conclusion

The statsmodel clustering pipeline has a solid foundation with excellent components (MarkovRegressionAdapter, ClusterQualityAssessor), but **critical integration issues** prevent it from functioning correctly. The main issues are:

1. **Type mismatch** between MarkovRegressionAdapter output and pipeline expectations
2. **Missing CSV report generation** despite having the capability
3. **Configuration oversight** in stage registration

All issues are straightforward to fix and the fixes are detailed in this report.

**Estimated fix time:** 30-45 minutes
**Risk level:** Low (fixes are isolated to single file per issue)
