# Final Feature Selection Report

**Generated:** 2025-11-23 17:50:01
**Step:** feature_generation_final_feature_selection_step

## Configuration

- **Symbol:** ETHUSDT
- **Exchange:** binance
- **Timeframe:** 15m
- **Execution Mode:** full
- **Feature Count Targets:** [60, 50, 40]
- **Selection Method:** permutation ✅
- **Importance Type:** Permutation (captures feature interactions, not just Gini splits) 📊
- **Optimization Enabled:** True

## Top IC Features (Meta-Label Overview)

**Top 5 features by IC vs binary_label (overall):**

1. day (IC = 0.3169)\n2. candlestick_engulfing_pattern_vwap_27x_ratio (IC = 0.0234)\n3. candlestick_piercing_line_pattern_vwap_27x_ratio (IC = -0.0148)\n
**Top 5 features by IC vs realized_return (overall):**

1. candlestick_engulfing_pattern_vwap_27x_ratio (IC = 0.0738)\n2. day (IC = 0.0262)\n3. candlestick_piercing_line_pattern_vwap_27x_ratio (IC = -0.0193)\n

## Feature Selection Methodology

✅ **Using Permutation Importance**
- Captures how features work together (feature interactions)
- More reliable than standard Gini importance for complex trading strategies
- Measures true impact on model predictions
- Better for identifying genuinely predictive features

## Feature Selection Results

- **60 Features Set:** 3 features selected
- **50 Features Set:** 3 features selected
- **40 Features Set:** 3 features selected

- **Total Feature Sets:** 3

## Selected Features by Set

### 60 Features Set (3 features)

1. candlestick_piercing_line_pattern_vwap_27x_ratio
2. candlestick_engulfing_pattern_vwap_27x_ratio
3. day

### 50 Features Set (3 features)

1. candlestick_piercing_line_pattern_vwap_27x_ratio
2. candlestick_engulfing_pattern_vwap_27x_ratio
3. day

### 40 Features Set (3 features)

1. candlestick_piercing_line_pattern_vwap_27x_ratio
2. candlestick_engulfing_pattern_vwap_27x_ratio
3. day


## Enhanced Feature Analysis

### Correlation Analysis

- **Average Correlation:** 0.1225  — average pairwise |ρ| between features; lower is better and values <0.2 indicate low redundancy.
- **Max Correlation:** 1.0000  — highest |ρ| observed; very high values may indicate near-duplicate signals.
- **Min Correlation:** 0.0251  — lowest |ρ|; values near 0 show some features are nearly independent.
- **High Correlation Pairs:** 0  — number of feature pairs above the threshold; 0 is ideal.
- **Correlation Threshold:** 0.8  — pairs above this are considered redundant for clustering.

### Redundancy Detection

- **Status:** Skipped (Performance optimization - correlation analysis provides sufficient information)

### Stability Analysis

- **Average Stability:** 0.6396  — 0–1 score of importance consistency across time windows; higher is better and >0.5 is strong.
- **Stable Features:** 1  — features above the stability threshold; more indicates a more robust set.
- **Stability Threshold:** 0.6815129829799506  — adaptive cutoff used to classify features as stable.
- **Time Windows:** 5  — number of rolling windows used for stability estimation.

### Cross-Validation Analysis

- **Average Consistency:** 0.0000  — average selection frequency across folds (0–1); higher means features reappear more often.
- **Consistent Features:** 0  — features with consistency above the threshold; more is better.
- **Consistency Threshold:** 0.6  — minimum fold frequency to be considered consistent.
- **CV Folds:** 10  — number of time-series splits used; more folds give a stricter stability test.

### Baseline Comparison

- **Improvement Ratio:** 0.98x  — selected set score / baseline score; values <1.0 mean the selection outperforms baseline.
- **Selected Features Avg Score:** 0.145742  — mean importance of selected features; higher is better.
- **Baseline Avg Score:** 0.149077  — mean importance over all features; acts as a reference level.
- **Baseline Trials:** 10  — number of random baseline draws; more gives a more stable baseline estimate.
- **Features Compared:** 3  — size of the selected feature set used for the comparison.

### Selection Frequency Distribution

- **Distribution Mode:** bimodal
- **Interpretation:** ✅ Clear separation between stable and unstable features
- **Highly Stable Features (>80%):** 0
- **Highly Unstable Features (<20%):** 3
- **Unstable Features Ratio:** 100.0%

**Frequency Breakdown:**
- 0-20%: 3 features (100.0%)
- 100%: 0 features (0.0%)
- 20-40%: 0 features (0.0%)
- 40-60%: 0 features (0.0%)
- 60-80%: 0 features (0.0%)
- 80-100%: 0 features (0.0%)

**⚠️ Warnings:**
- 🚨 >60% of features are highly unstable (selected <40% of time)
- ⚠️ <20% of features are highly stable (selected >80% of time)

### Mutual Information Stability (Correlation Proxy)

- **Stable Features (CV < 0.3):** 0
- **High MI Features (>0.1):** 1
- **Mean MI Stability:** 0.302
- **Method:** correlation_proxy
- **Execution Time:** 0.0s

🚨 Low MI stability - features may not generalize well

### Data Leakage Detection (Phase 3)

- **Perfect Correlations (>0.99):** 0
- **Suspicious Correlations (>0.95):** 0
- **Execution Time:** 0.0s

✅ No data leakage detected

### Feature Information Content (Phase 3)

- **Low Variance Features (<0.01):** 0
- **Quasi-Constant Features (>99%):** 0
- **Execution Time:** 0.0s

✅ All features have sufficient information content

### Meta-Label Diagnostics (IC/AUC vs Targets)

These diagnostics summarize how the final selected features relate to the meta-label targets: binary_label (classification) and realized_return (economic P&L). Scores are reported as Information Coefficient (Pearson correlation) and AUC where applicable.

#### Overall (Full Sample)

| Rank | Feature | IC (binary_label) | AUC (binary_label) | N (binary) | IC (realized_return) | N (ret) |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | day | 0.3169 | 0.6929 | 194 | 0.0262 | 390 |
| 2 | candlestick_engulfing_pattern_vwap_27x_ratio | 0.0234 | 0.3908 | 194 | 0.0738 | 390 |
| 3 | candlestick_piercing_line_pattern_vwap_27x_ratio | -0.0148 | 0.6360 | 194 | -0.0193 | 390 |


## Baseline Learnability of Selected Features

This baseline fits simple models (linear regression and small LightGBM baselines) using only the final selected features. It provides an upper bound on how much of the target variance is explainable by this feature set alone, before any complex downstream modeling.

## Baseline Predictive Check

**Dataset:** 194 samples, 3 features

### Top Single-Feature Signals

| Rank | Feature | Test R² | Pearson | Quality Score |
|------|---------|---------|---------|---------------|
| 1 | `day` | -0.008 | 0.317 | 0.127 |
| 2 | `candlestick_engulfing_pattern_vwap_27x_ratio` | -0.010 | 0.023 | 0.009 |
| 3 | `candlestick_piercing_line_pattern_vwap_27x_ratio` | -0.007 | -0.015 | 0.006 |

### Small Multivariate LGBM Baseline

| Type | Features | Test R² |
|------|----------|---------|
| Pair | `day`, `candlestick_piercing_line_pattern_vwap_27x_ratio` | 0.805 |
| Triplet | `day`, `candlestick_engulfing_pattern_vwap_27x_ratio`, `candlestick_piercing_line_pattern_vwap_27x_ratio` | 0.833 |

### Interpretation

**Quality Score:** 0.13/1.0

**Summary:** ⚠️ Weak predictive signals

**Insights:**
- Best feature `day` achieved Test R² = -0.008
- Positive Test R² features: 0 (0.0%)
- Median Test R² across evaluated features: -0.008
- LGBM best feature `day` achieved Test R² = 0.702

**Recommendations:**
- Consider revisiting labeling/target definitions; very few features carry signal
- Even the best single feature underperforms; investigate data leakage or excessive noise


**Baseline learnability CSV:** `outcomes/baseline_check_final_feature_selection_20251123_175001.csv`

### How to Read These Learnability Metrics

- **Test R²** rows show, for each selected feature, how much of the target variance it explains out-of-sample in a simple regression. Values near 0 mean weak signal; values above roughly 0.3–0.4 indicate strong linear signal; negative values indicate that even a simple model fails to generalize.
- The **quality score** aggregates how many features achieve positive Test R², how strong the best feature(s) are, and how consistent performance is across evaluated features. Scores close to 1.0 mean that many features contain robust, learnable signal; scores near 0 indicate that this feature set behaves mostly like noise.
If the selected-feature quality score is low, or if most Test R² values are negative, it suggests that the final selection may be too aggressive or misaligned with the target. In that case, consider revisiting labeling, feature generation, or selection thresholds before relying on this set in production models.

## Performance Metrics

- **Execution Time:** N/A seconds
- **Optimization Enabled:** Yes
- **Hardware Optimization:** No

## Optimization Details

- **VectorBT Optimization:** Enabled
- **Rolling Optimizer:** Available
- **Hardware Manager:** Available

## Generated Artifacts

- **Feature Sets:** 3
- **Feature DataFrames:** 3
- **SHAP Analyses:** 0
- **Metadata Files:** 2
- **Total Artifacts:** 10

## Summary

Final feature selection completed successfully. Generated 3 optimized feature sets with comprehensive SHAP analysis and metadata. All artifacts saved in both pickle and markdown formats.

---
*Generated by Feature Generation Final Feature Selection Step at 2025-11-23 17:50:01*
