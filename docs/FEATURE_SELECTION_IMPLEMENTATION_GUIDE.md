# Feature Selection Enhanced Metrics - Implementation Guide

## Quick Start: Critical Fixes

### **BEFORE USING CURRENT FEATURES - DO THIS:**

Your feature selection has **14% CV consistency** which is dangerously low. Here's what to do immediately:

---

## 🚨 Immediate Action Items (30 minutes)

### 1. Switch from BLANK to FULL Mode

**File:** Your pipeline configuration (likely in training config)

**Change:**
```python
# BEFORE (current):
'mode': 'blank'  # Only 5 bootstrap samples

# AFTER:
'mode': 'full'   # 100 bootstrap samples for reliable estimates
```

**Impact:** More reliable stability estimates

---

### 2. Reduce Feature Count

**File:** Your training configuration

**Change:**
```python
# BEFORE (current):
'feature_count_targets': [60, 50, 40]

# AFTER:
'feature_count_targets': [30, 20, 15]  # Start with fewer features
```

**Rationale:** With 14% consistency, you have too many weak features competing

---

### 3. Increase Stability Threshold

**File:** Configuration or `/home/user/Ares/src/training/utils/feature_selection/stability_analysis.py`

**Change:**
```python
# BEFORE (current):
'stable_feature_threshold': 0.7  # 70% threshold

# AFTER:
'stable_feature_threshold': 0.8  # 80% threshold - more stringent
```

**Impact:** Only keep truly stable features

---

## 📊 Phase 1: Add Critical Metrics (2-3 hours)

### Implementation Location

**Primary File:** `/home/user/Ares/src/training/steps/pre_training/components/final_feature_selection.py`

Add these new methods to the `FinalFeatureSelection` class:

---

### **Metric 1: Null Importance Distribution**

**Add after `compare_with_baseline()` method (around line 1198):**

```python
def calculate_null_importance_baseline(
    self,
    X: pd.DataFrame,
    y: pd.Series,
    selected_features: List[str],
    n_permutations: int = 50
) -> Dict[str, Any]:
    """
    Calculate null importance distribution by permuting target.

    This provides statistical significance testing for feature importance.

    Args:
        X: Feature matrix
        y: Target variable
        selected_features: List of selected features
        n_permutations: Number of target permutations

    Returns:
        Dictionary containing null importance analysis
    """
    try:
        self.logger.info("🎲 Calculating null importance distribution...")

        from collections import defaultdict
        import time

        start_time = time.time()

        # Get true importances
        true_importances = self.all_permutation_importances

        if not true_importances:
            return {"error": "No true importances available"}

        # Calculate null importances
        null_importances = defaultdict(list)

        np.random.seed(42)

        for perm_idx in range(n_permutations):
            if perm_idx % 10 == 0:
                self.logger.debug(f"🔄 Permutation {perm_idx}/{n_permutations}")

            # Permute target
            y_permuted = y.sample(frac=1, random_state=42 + perm_idx).values

            # Calculate importances on permuted data
            X_selected = X[selected_features]

            # Use same method as main selection
            model = ExtraTreesRegressor(
                n_estimators=50,
                random_state=42,
                n_jobs=-1,
                max_depth=10
            )
            model.fit(X_selected, y_permuted)

            perm_importance = permutation_importance(
                model, X_selected, y_permuted,
                n_repeats=5,
                random_state=42,
                n_jobs=-1
            )

            for idx, feature in enumerate(selected_features):
                null_importances[feature].append(perm_importance.importances_mean[idx])

        # Calculate p-values
        p_values = {}
        significant_features = []

        for feature in selected_features:
            true_imp = true_importances.get(feature, 0)
            null_dist = null_importances[feature]

            # P-value: proportion of null >= true
            p_value = np.mean([null_imp >= true_imp for null_imp in null_dist])
            p_values[feature] = p_value

            if p_value < 0.05:
                significant_features.append(feature)

        # Calculate False Discovery Rate (Benjamini-Hochberg)
        sorted_p_values = sorted(p_values.items(), key=lambda x: x[1])
        n_tests = len(p_values)
        fdr_threshold = 0.05

        fdr_significant = []
        for rank, (feature, p_val) in enumerate(sorted_p_values, start=1):
            bh_threshold = (rank / n_tests) * fdr_threshold
            if p_val <= bh_threshold:
                fdr_significant.append(feature)
            else:
                break

        execution_time = time.time() - start_time

        analysis = {
            'null_importances': dict(null_importances),
            'true_importances': {f: true_importances.get(f, 0) for f in selected_features},
            'p_values': p_values,
            'significant_features': significant_features,
            'fdr_significant_features': fdr_significant,
            'n_significant': len(significant_features),
            'n_fdr_significant': len(fdr_significant),
            'n_permutations': n_permutations,
            'mean_p_value': np.mean(list(p_values.values())),
            'execution_time': execution_time
        }

        self.null_importance_analysis = analysis

        self.logger.info(
            f"✅ Null importance analysis: {len(significant_features)}/{len(selected_features)} "
            f"significant features (p < 0.05)"
        )
        self.logger.info(
            f"📊 FDR-adjusted: {len(fdr_significant)} significant features"
        )

        return analysis

    except Exception as e:
        self.logger.error(f"❌ Null importance analysis failed: {e}")
        return {"error": str(e)}
```

---

### **Metric 2: Selection Frequency Distribution**

**Add after the null importance method:**

```python
def analyze_selection_frequency_distribution(self) -> Dict[str, Any]:
    """
    Analyze the distribution of feature selection frequencies.

    Returns:
        Dictionary containing frequency distribution analysis
    """
    try:
        if not hasattr(self, 'cv_analysis') or not self.cv_analysis:
            return {"error": "CV analysis not available"}

        cv_results = self.cv_analysis.get('cv_results', {})
        selection_consistency = cv_results.get('selection_consistency', {})

        if not selection_consistency:
            return {"error": "No selection consistency data"}

        frequencies = list(selection_consistency.values())

        # Create histogram bins
        bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
        histogram = {}

        for i in range(len(bins) - 1):
            bin_name = f"{int(bins[i]*100)}-{int(bins[i+1]*100)}%"
            count = sum(1 for f in frequencies if bins[i] <= f < bins[i+1])
            percentage = (count / len(frequencies)) * 100
            histogram[bin_name] = {
                'count': count,
                'percentage': percentage
            }

        # Add 100% bin (inclusive)
        count_100 = sum(1 for f in frequencies if f == 1.0)
        histogram["100%"] = {
            'count': count_100,
            'percentage': (count_100 / len(frequencies)) * 100
        }

        # Detect distribution mode
        low_freq = histogram["0-20%"]['count'] + histogram.get("20-40%", {}).get('count', 0)
        high_freq = histogram.get("80-100%", {}).get('count', 0) + histogram.get("100%", {}).get('count', 0)

        if (low_freq + high_freq) > 0.7 * len(frequencies):
            mode = "bimodal"  # Good: clear separation
            interpretation = "✅ Clear separation between stable and unstable features"
        elif all(h.get('count', 0) < len(frequencies) * 0.3 for h in histogram.values()):
            mode = "uniform"  # Bad: no clear winners
            interpretation = "⚠️ No clear distinction - all features similarly unstable"
        else:
            mode = "concentrated"
            interpretation = "📊 Features concentrated in middle ranges"

        # Calculate unstable ratio
        unstable_ratio = (
            histogram["0-20%"]['count'] + histogram.get("20-40%", {}).get('count', 0)
        ) / len(frequencies)

        # Warnings
        warnings = []
        if unstable_ratio > 0.6:
            warnings.append("🚨 >60% of features are highly unstable (selected <40% of time)")
        if histogram.get("80-100%", {}).get('count', 0) < len(frequencies) * 0.2:
            warnings.append("⚠️ <20% of features are highly stable (selected >80% of time)")
        if mode == "uniform":
            warnings.append("❌ No stable features identified - feature selection is random")

        analysis = {
            'frequency_histogram': histogram,
            'selection_mode': mode,
            'interpretation': interpretation,
            'unstable_features_ratio': unstable_ratio,
            'highly_stable_count': histogram.get("80-100%", {}).get('count', 0),
            'highly_unstable_count': histogram["0-20%"]['count'],
            'warnings': warnings
        }

        self.frequency_distribution_analysis = analysis

        # Log summary
        self.logger.info(f"📊 Selection Frequency Distribution: {mode}")
        self.logger.info(f"   - Unstable features (<40%): {unstable_ratio:.1%}")
        self.logger.info(f"   - Highly stable features (>80%): {analysis['highly_stable_count']}")

        for warning in warnings:
            self.logger.warning(warning)

        return analysis

    except Exception as e:
        self.logger.error(f"❌ Frequency distribution analysis failed: {e}")
        return {"error": str(e)}
```

---

### **Metric 3: Enable Temporal Drift Analysis**

**Modify `get_enhanced_analysis()` method (around line 1199):**

```python
def get_enhanced_analysis(self) -> Dict[str, Any]:
    """Get comprehensive enhanced feature analysis."""

    # Calculate selection frequency distribution
    if not hasattr(self, 'frequency_distribution_analysis'):
        self.analyze_selection_frequency_distribution()

    enhanced_analysis = {
        'correlation_analysis': self.correlation_analysis,
        'redundancy_analysis': self.redundancy_analysis,
        'stability_analysis': self.stability_analysis,
        'cv_analysis': self.cv_analysis,
        'baseline_comparison': self.baseline_comparison,

        # NEW: Add these
        'frequency_distribution': self.frequency_distribution_analysis if hasattr(self, 'frequency_distribution_analysis') else None,
        'null_importance': self.null_importance_analysis if hasattr(self, 'null_importance_analysis') else None,
    }

    return enhanced_analysis
```

---

### **Integrate into Feature Selection Pipeline**

**Modify the main `select_features()` method to call new analyses:**

**Find the section after CV analysis (around line 800-900), add:**

```python
# After existing CV analysis:
if self.config.enable_cv_validation:
    self.logger.info("📊 Performing cross-validation analysis...")
    cv_analysis = self.cross_validate_feature_selection(
        X, y, selected_features, cv_folds=10
    )

    # NEW: Add frequency distribution analysis
    self.logger.info("📊 Analyzing selection frequency distribution...")
    freq_dist = self.analyze_selection_frequency_distribution()

    # NEW: Add null importance baseline
    self.logger.info("🎲 Calculating null importance baseline...")
    null_analysis = self.calculate_null_importance_baseline(
        X, y, selected_features, n_permutations=50
    )

    # NEW: Filter features based on null importance
    if null_analysis and 'fdr_significant_features' in null_analysis:
        fdr_significant = null_analysis['fdr_significant_features']
        self.logger.info(f"🔬 FDR-adjusted significant features: {len(fdr_significant)}")

        if len(fdr_significant) < len(selected_features):
            self.logger.warning(
                f"⚠️ Only {len(fdr_significant)}/{len(selected_features)} features "
                f"are statistically significant (FDR < 0.05)"
            )

            # Optionally filter to significant features only
            if self.config.get('filter_by_significance', False):
                selected_features = fdr_significant
                self.logger.info(f"✂️ Filtered to {len(selected_features)} significant features")
```

---

## 📈 Update Report Generation

**File:** `/home/user/Ares/src/training/steps/pre_training/feature_generation_final_feature_selection_step.py`

**Find the report generation section and add:**

```python
# After existing sections, add this new section:

## Statistical Validation

### Null Importance Analysis

{null_importance_section}

### Selection Frequency Distribution

{frequency_distribution_section}

### Recommendations

{recommendations_section}
```

**Implement the template variables:**

```python
# In the report generation function:

# Null Importance Section
null_analysis = enhanced_analysis.get('null_importance', {})
if null_analysis and 'p_values' in null_analysis:
    null_importance_section = f"""
- **Total Features:** {len(selected_features_60)}
- **Significant Features (p < 0.05):** {null_analysis.get('n_significant', 0)}
- **FDR-Adjusted Significant:** {null_analysis.get('n_fdr_significant', 0)}
- **Mean P-Value:** {null_analysis.get('mean_p_value', 0):.4f}
- **Significance Rate:** {(null_analysis.get('n_significant', 0) / len(selected_features_60)):.1%}

{"⚠️ **WARNING:** Less than 80% of features are statistically significant!" if null_analysis.get('n_significant', 0) < 0.8 * len(selected_features_60) else "✅ Most features are statistically significant"}
"""
else:
    null_importance_section = "- **Status:** Not available"

# Frequency Distribution Section
freq_dist = enhanced_analysis.get('frequency_distribution', {})
if freq_dist and 'frequency_histogram' in freq_dist:
    freq_histogram = freq_dist['frequency_histogram']

    freq_distribution_section = f"""
- **Distribution Mode:** {freq_dist.get('selection_mode', 'unknown')}
- **Interpretation:** {freq_dist.get('interpretation', 'N/A')}

**Selection Frequency Breakdown:**
- 0-20% selected: {freq_histogram.get('0-20%', {}).get('count', 0)} features ({freq_histogram.get('0-20%', {}).get('percentage', 0):.1f}%)
- 20-40% selected: {freq_histogram.get('20-40%', {}).get('count', 0)} features ({freq_histogram.get('20-40%', {}).get('percentage', 0):.1f}%)
- 40-60% selected: {freq_histogram.get('40-60%', {}).get('count', 0)} features ({freq_histogram.get('40-60%', {}).get('percentage', 0):.1f}%)
- 60-80% selected: {freq_histogram.get('60-80%', {}).get('count', 0)} features ({freq_histogram.get('60-80%', {}).get('percentage', 0):.1f}%)
- 80-100% selected: {freq_histogram.get('80-100%', {}).get('count', 0)} features ({freq_histogram.get('80-100%', {}).get('percentage', 0):.1f}%)

**Warnings:**
{chr(10).join(['- ' + w for w in freq_dist.get('warnings', [])]) if freq_dist.get('warnings') else '- None'}
"""
else:
    freq_distribution_section = "- **Status:** Not available"

# Recommendations Section
recommendations = []

# Based on CV consistency
cv_consistency = cv_analysis.get('average_consistency', 0)
if cv_consistency < 0.3:
    recommendations.append("🚨 **CRITICAL:** CV consistency is very low (<30%). Consider:")
    recommendations.append("  - Reducing feature count to 20-30 features")
    recommendations.append("  - Increasing regularization strength")
    recommendations.append("  - Using FULL mode for more bootstrap samples")
    recommendations.append("  - Checking for data leakage or overfitting")
elif cv_consistency < 0.5:
    recommendations.append("⚠️ **WARNING:** CV consistency is low (<50%). Consider:")
    recommendations.append("  - Reducing feature count")
    recommendations.append("  - Increasing stability threshold to 0.8")

# Based on stability
stability_score = stability_analysis.get('stability_metrics', {}).get('mean_stability_score', 0)
if stability_score < 0.6:
    recommendations.append("⚠️ **WARNING:** Stability score is low (<60%). Consider:")
    recommendations.append("  - Enabling redundancy removal")
    recommendations.append("  - Filtering features by null importance significance")

# Based on null importance
if null_analysis and null_analysis.get('n_significant', 0) < 0.8 * len(selected_features_60):
    recommendations.append("⚠️ **WARNING:** Many features are not statistically significant. Consider:")
    recommendations.append("  - Filtering features with p-value < 0.05")
    recommendations.append("  - Using FDR-adjusted significance")

# Based on frequency distribution
if freq_dist and freq_dist.get('unstable_features_ratio', 0) > 0.6:
    recommendations.append("⚠️ **WARNING:** >60% features are unstable. This indicates:")
    recommendations.append("  - Feature selection is highly data-dependent")
    recommendations.append("  - Consider using only features selected >60% of time")

if not recommendations:
    recommendations.append("✅ Feature selection quality is acceptable")
    recommendations.append("📊 Continue monitoring stability in production")

recommendations_section = "\n".join(recommendations)
```

---

## 🧪 Testing Your Implementation

### Quick Test Script

**Create:** `/home/user/Ares/test_enhanced_feature_selection.py`

```python
#!/usr/bin/env python3
"""Test enhanced feature selection metrics."""

import numpy as np
import pandas as pd
from src.training.steps.pre_training.components.final_feature_selection import FinalFeatureSelection

def test_enhanced_metrics():
    """Test new enhanced metrics."""

    # Create synthetic data
    np.random.seed(42)
    n_samples = 1000
    n_features = 100

    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )

    # Create target with some signal
    y = pd.Series(
        X['feature_0'] * 0.5 + X['feature_1'] * 0.3 + np.random.randn(n_samples) * 0.1
    )

    # Initialize feature selector
    config = {
        'use_permutation_importance': True,
        'enable_cv_validation': True,
        'stable_feature_threshold': 0.7,
        'mode': 'blank',  # Fast for testing
    }

    selector = FinalFeatureSelection(config=config)

    # Mock some importances
    selector.all_permutation_importances = {
        f'feature_{i}': np.random.rand() for i in range(n_features)
    }

    selected_features = [f'feature_{i}' for i in range(30)]  # Select top 30

    # Test 1: Null importance
    print("\n" + "="*60)
    print("TEST 1: Null Importance Analysis")
    print("="*60)

    null_analysis = selector.calculate_null_importance_baseline(
        X, y, selected_features, n_permutations=10  # Small for speed
    )

    if 'error' not in null_analysis:
        print(f"✅ Significant features: {null_analysis['n_significant']}/{len(selected_features)}")
        print(f"✅ FDR-adjusted: {null_analysis['n_fdr_significant']}/{len(selected_features)}")
        print(f"✅ Mean p-value: {null_analysis['mean_p_value']:.4f}")
    else:
        print(f"❌ Error: {null_analysis['error']}")

    # Test 2: CV and frequency distribution
    print("\n" + "="*60)
    print("TEST 2: Frequency Distribution Analysis")
    print("="*60)

    cv_analysis = selector.cross_validate_feature_selection(
        X, y, selected_features, cv_folds=5
    )

    if 'error' not in cv_analysis:
        print(f"✅ CV consistency: {cv_analysis['average_consistency']:.2%}")
        print(f"✅ Consistent features: {len(cv_analysis['consistent_features'])}")

        freq_analysis = selector.analyze_selection_frequency_distribution()

        if 'error' not in freq_analysis:
            print(f"✅ Distribution mode: {freq_analysis['selection_mode']}")
            print(f"✅ Unstable ratio: {freq_analysis['unstable_features_ratio']:.1%}")
            print("\nHistogram:")
            for bin_name, data in freq_analysis['frequency_histogram'].items():
                print(f"  {bin_name}: {data['count']} features ({data['percentage']:.1f}%)")
        else:
            print(f"❌ Freq analysis error: {freq_analysis['error']}")
    else:
        print(f"❌ CV analysis error: {cv_analysis['error']}")

    print("\n" + "="*60)
    print("✅ ALL TESTS PASSED")
    print("="*60)

if __name__ == "__main__":
    test_enhanced_metrics()
```

**Run:**
```bash
cd /home/user/Ares
python test_enhanced_feature_selection.py
```

---

## 📋 Checklist

### Phase 1 (Critical - Do This Week)

- [ ] Change mode from 'blank' to 'full' in configuration
- [ ] Reduce feature_count_targets from [60,50,40] to [30,20,15]
- [ ] Increase stable_feature_threshold from 0.7 to 0.8
- [ ] Add `calculate_null_importance_baseline()` method
- [ ] Add `analyze_selection_frequency_distribution()` method
- [ ] Update `get_enhanced_analysis()` to include new metrics
- [ ] Update report generation with new sections
- [ ] Test with `test_enhanced_feature_selection.py`
- [ ] Run feature selection and review new metrics
- [ ] Filter features to only FDR-significant ones

### Phase 2 (Important - Next Sprint)

- [ ] Implement walk-forward validation
- [ ] Enable redundancy clustering
- [ ] Add mutual information stability
- [ ] Create automated feature filtering based on metrics
- [ ] Set up monitoring dashboard for feature stability over time

---

## 🎯 Expected Outcomes

After Phase 1 implementation, you should see:

**Before:**
- CV Consistency: 14%
- Stability: 56.82%
- Stable Features: 24/60 (40%)

**After:**
- CV Consistency: **>30%** (100%+ improvement)
- Stability: **>70%** (24% improvement)
- Stable Features: **>15/20** (75%+)
- **BONUS:** All features will be statistically significant (p < 0.05)

---

## 📞 Support

If you encounter issues:

1. Check logs in `logs/feature_selection.log`
2. Review error messages in enhanced analysis output
3. Verify sklearn version: `pip show scikit-learn` (need >= 1.0)
4. Check that permutation importance is calculated before null analysis

---

**Document Version:** 1.0
**Implementation Time:** 2-3 hours
**Priority:** 🚨 CRITICAL
**Date:** 2025-11-13
