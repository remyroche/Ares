# Feature Selection - Remaining Critical Metrics

**Status:** Phase 1 & 2 Complete | Phase 3 Recommendations

---

## Current Coverage: 8/10 ✅

Your feature selection now has **excellent** coverage of core ML validation metrics. However, there are **2-3 critical gaps** that should be addressed.

---

## 🚨 **CRITICAL: Data Leakage Detection**

### Why It's Critical

Data leakage is the **#1 cause of overfitting** in ML pipelines:
- Features calculated using future information
- Target proxies that shouldn't exist
- Preprocessing errors

**Signs of leakage:**
- Correlation with target > 0.95 (suspiciously high)
- Perfect prediction on training data, terrible on OOS
- Feature that "shouldn't" be that predictive

### Implementation

```python
def detect_potential_leakage(
    self,
    X: pd.DataFrame,
    y: pd.Series,
    selected_features: List[str],
    suspicious_threshold: float = 0.95,
    perfect_threshold: float = 0.99
) -> Dict[str, Any]:
    """
    Detect potential data leakage through suspiciously high correlations.

    Args:
        X: Feature matrix
        y: Target variable
        selected_features: List of selected features
        suspicious_threshold: Correlation threshold for warnings (default: 0.95)
        perfect_threshold: Correlation threshold for critical alerts (default: 0.99)

    Returns:
        Dictionary containing leakage detection results
    """
    try:
        self.logger.info("🔍 Detecting potential data leakage...")

        import time
        start_time = time.time()

        suspicious_features = []
        perfect_features = []
        feature_correlations = {}

        for feature in selected_features:
            try:
                # Calculate absolute correlation with target
                corr = abs(X[feature].corr(y))

                if np.isnan(corr):
                    continue

                feature_correlations[feature] = corr

                # Check thresholds
                if corr >= perfect_threshold:
                    perfect_features.append((feature, corr))
                elif corr >= suspicious_threshold:
                    suspicious_features.append((feature, corr))

            except Exception as e:
                self.logger.warning(f"Could not calculate correlation for {feature}: {e}")
                continue

        # Sort by correlation (descending)
        suspicious_features.sort(key=lambda x: x[1], reverse=True)
        perfect_features.sort(key=lambda x: x[1], reverse=True)

        execution_time = time.time() - start_time

        # Generate warnings
        warnings = []
        if perfect_features:
            warnings.append(
                f"🚨 CRITICAL: {len(perfect_features)} features have near-perfect correlation (>{perfect_threshold}) - "
                "likely data leakage!"
            )
        if suspicious_features:
            warnings.append(
                f"⚠️ WARNING: {len(suspicious_features)} features have very high correlation (>{suspicious_threshold}) - "
                "investigate for potential leakage"
            )

        analysis = {
            'perfect_features': perfect_features,
            'suspicious_features': suspicious_features,
            'feature_correlations': feature_correlations,
            'n_perfect': len(perfect_features),
            'n_suspicious': len(suspicious_features),
            'warnings': warnings,
            'perfect_threshold': perfect_threshold,
            'suspicious_threshold': suspicious_threshold,
            'execution_time': execution_time
        }

        self.leakage_detection = analysis

        # Log findings
        if perfect_features:
            self.logger.error(f"🚨 POTENTIAL LEAKAGE: {len(perfect_features)} features with r > {perfect_threshold}")
            for feature, corr in perfect_features[:5]:  # Show top 5
                self.logger.error(f"   - {feature}: r = {corr:.4f}")

        if suspicious_features:
            self.logger.warning(f"⚠️ SUSPICIOUS: {len(suspicious_features)} features with r > {suspicious_threshold}")
            for feature, corr in suspicious_features[:5]:  # Show top 5
                self.logger.warning(f"   - {feature}: r = {corr:.4f}")

        if not perfect_features and not suspicious_features:
            self.logger.info("✅ No data leakage detected")

        return analysis

    except Exception as e:
        self.logger.error(f"❌ Leakage detection failed: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}
```

### Report Section

```python
# NEW: Data Leakage Detection
if 'leakage_detection' in enhanced_analysis and enhanced_analysis['leakage_detection'] and 'error' not in enhanced_analysis['leakage_detection']:
    leakage = enhanced_analysis['leakage_detection']
    report += f"### Data Leakage Detection\n\n"
    report += f"- **Perfect Correlations (>0.99):** {leakage.get('n_perfect', 0)}\n"
    report += f"- **Suspicious Correlations (>0.95):** {leakage.get('n_suspicious', 0)}\n"
    report += f"- **Execution Time:** {leakage.get('execution_time', 'N/A'):.1f}s\n"

    # Show perfect features (critical)
    perfect_features = leakage.get('perfect_features', [])
    if perfect_features:
        report += f"\n🚨 **CRITICAL - Potential Data Leakage:**\n"
        for feature, corr in perfect_features[:10]:
            report += f"- {feature}: r = {corr:.4f}\n"
        report += f"\n**ACTION REQUIRED:** Investigate these features for data leakage!\n"

    # Show suspicious features (warning)
    suspicious_features = leakage.get('suspicious_features', [])
    if suspicious_features and not perfect_features:
        report += f"\n⚠️ **Suspicious Features:**\n"
        for feature, corr in suspicious_features[:5]:
            report += f"- {feature}: r = {corr:.4f}\n"
        report += f"\n**RECOMMENDED:** Review these features to ensure no leakage\n"

    # All clear
    if not perfect_features and not suspicious_features:
        report += f"\n✅ No data leakage detected\n"

    report += f"\n"
```

### Integration

```python
# In analyze_enhanced_features(), add:

# NEW: Data Leakage Detection (CRITICAL)
tprint_info("🔍 Detecting potential data leakage...")
leakage_detection = temp_component.detect_potential_leakage(X, y, selected_features)
analysis_results['leakage_detection'] = leakage_detection
```

---

## ⚠️ **IMPORTANT: Feature Information Content**

### Why It's Important

Features without sufficient information content are useless for ML:
- **Near-constant features**: All values the same or nearly the same
- **Low variance**: Insufficient variation to be predictive
- **Quasi-constant**: 99% of values are the same

### Implementation

```python
def check_feature_information_content(
    self,
    X: pd.DataFrame,
    selected_features: List[str],
    variance_threshold: float = 0.01,
    quasi_constant_threshold: float = 0.99
) -> Dict[str, Any]:
    """
    Check if features have sufficient information content for ML.

    Args:
        X: Feature matrix
        selected_features: List of selected features
        variance_threshold: Minimum variance required
        quasi_constant_threshold: Maximum proportion of most frequent value

    Returns:
        Dictionary containing information content analysis
    """
    try:
        self.logger.info("📊 Checking feature information content...")

        import time
        start_time = time.time()

        low_variance_features = []
        quasi_constant_features = []
        feature_stats = {}

        for feature in selected_features:
            try:
                values = X[feature]

                # Calculate variance
                variance = values.var()

                # Calculate most frequent value proportion
                value_counts = values.value_counts(normalize=True)
                max_proportion = value_counts.iloc[0] if len(value_counts) > 0 else 1.0

                # Calculate number of unique values
                n_unique = values.nunique()

                feature_stats[feature] = {
                    'variance': variance,
                    'max_value_proportion': max_proportion,
                    'n_unique': n_unique,
                    'mean': values.mean(),
                    'std': values.std()
                }

                # Check thresholds
                if variance < variance_threshold:
                    low_variance_features.append((feature, variance))

                if max_proportion >= quasi_constant_threshold:
                    quasi_constant_features.append((feature, max_proportion))

            except Exception as e:
                self.logger.warning(f"Could not analyze {feature}: {e}")
                continue

        execution_time = time.time() - start_time

        # Generate warnings
        warnings = []
        if low_variance_features:
            warnings.append(
                f"⚠️ {len(low_variance_features)} features have very low variance (<{variance_threshold})"
            )
        if quasi_constant_features:
            warnings.append(
                f"⚠️ {len(quasi_constant_features)} features are quasi-constant (>{quasi_constant_threshold*100}% same value)"
            )

        analysis = {
            'low_variance_features': low_variance_features,
            'quasi_constant_features': quasi_constant_features,
            'feature_stats': feature_stats,
            'n_low_variance': len(low_variance_features),
            'n_quasi_constant': len(quasi_constant_features),
            'warnings': warnings,
            'variance_threshold': variance_threshold,
            'quasi_constant_threshold': quasi_constant_threshold,
            'execution_time': execution_time
        }

        self.information_content_analysis = analysis

        # Log findings
        if low_variance_features:
            self.logger.warning(f"⚠️ {len(low_variance_features)} low variance features")
            for feature, var in low_variance_features[:5]:
                self.logger.warning(f"   - {feature}: variance = {var:.6f}")

        if quasi_constant_features:
            self.logger.warning(f"⚠️ {len(quasi_constant_features)} quasi-constant features")
            for feature, prop in quasi_constant_features[:5]:
                self.logger.warning(f"   - {feature}: {prop*100:.1f}% same value")

        if not low_variance_features and not quasi_constant_features:
            self.logger.info("✅ All features have sufficient information content")

        return analysis

    except Exception as e:
        self.logger.error(f"❌ Information content check failed: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}
```

---

## ⚠️ **RECOMMENDED: Feature-Target Relationship Strength**

### Why It's Recommended

We already calculate this (`mi_mean`, `high_mi_features`) but don't prominently display it in reports.

### Enhancement

**Just add a dedicated report section:**

```python
# Enhanced MI Stability Report Section
if 'mi_stability' in enhanced_analysis and enhanced_analysis['mi_stability'] and 'error' not in enhanced_analysis['mi_stability']:
    mi_stab = enhanced_analysis['mi_stability']
    report += f"### Feature-Target Relationship Analysis\n\n"

    # Overall stats
    report += f"- **Stable Features (CV < 0.3):** {mi_stab.get('n_stable', 'N/A')}\n"
    report += f"- **High MI Features (>0.1):** {mi_stab.get('n_high_mi', 'N/A')}\n"
    report += f"- **Mean MI Stability:** {mi_stab.get('mean_mi_stability', 'N/A'):.3f}\n\n"

    # NEW: Show top features by relationship strength
    mi_mean = mi_stab.get('mi_mean', {})
    if mi_mean:
        sorted_features = sorted(mi_mean.items(), key=lambda x: x[1], reverse=True)

        report += f"**Top 10 Features by Correlation Strength:**\n"
        for i, (feature, corr) in enumerate(sorted_features[:10], 1):
            report += f"{i}. {feature}: r = {corr:.3f}\n"

        report += f"\n"

        # Weak features
        weak_features = [(f, c) for f, c in sorted_features if c < 0.05]
        if weak_features:
            report += f"⚠️ **{len(weak_features)} features have weak correlation (<0.05)**\n"
            report += f"Consider removing: {', '.join([f for f, c in weak_features[:5]])}\n\n"

    # Stability interpretation
    mean_stability = mi_stab.get('mean_mi_stability', 0)
    if mean_stability >= 0.7:
        report += f"✅ High MI stability across folds\n"
    elif mean_stability >= 0.5:
        report += f"⚠️ Moderate MI stability\n"
    else:
        report += f"🚨 Low MI stability - features may not generalize well\n"

    report += f"\n"
```

---

## 📋 **Implementation Priority**

### Phase 3: Critical Gaps

1. **CRITICAL** 🚨 **Data Leakage Detection**
   - Implementation time: 20 minutes
   - Impact: Prevents catastrophic overfitting
   - **DO THIS FIRST**

2. **IMPORTANT** ⚠️ **Feature Information Content**
   - Implementation time: 20 minutes
   - Impact: Removes useless features early
   - **DO THIS SECOND**

3. **RECOMMENDED** ✅ **Enhanced Relationship Strength Reporting**
   - Implementation time: 10 minutes
   - Impact: Better visibility into feature quality
   - **Nice to have**

### Total Time: ~50 minutes

---

## 🎯 **Final Coverage After Phase 3**

| Category | Coverage | Critical? | Status |
|----------|----------|-----------|--------|
| Statistical Significance | ✅✅✅ | 🚨 Yes | ✅ Done |
| Stability | ✅✅✅ | 🚨 Yes | ✅ Done |
| OOS Performance | ✅✅✅ | 🚨 Yes | ✅ Done |
| Redundancy | ✅✅ | ⚠️ Important | ✅ Done |
| Relationship Strength | ✅✅ | ⚠️ Important | 🔄 Enhanced |
| **Data Leakage Detection** | ✅✅ | 🚨 **Yes** | 🆕 **Phase 3** |
| **Information Content** | ✅✅ | ⚠️ Important | 🆕 **Phase 3** |
| Target Range Coverage | ❌ | ⚠️ Nice-to-have | ⏭️ Future |
| Outlier Sensitivity | ❌ | ⚠️ Nice-to-have | ⏭️ Future |

**Final Score: 9.5/10** ✅✅✅

---

## 🚀 **Quick Implementation (Copy-Paste Ready)**

Add these two methods to `final_feature_selection.py` after the MI stability method:

```python
# Add to FinalFeatureSelectionComponent class

def detect_potential_leakage(self, X, y, selected_features, suspicious_threshold=0.95, perfect_threshold=0.99):
    """Detect potential data leakage through suspiciously high correlations."""
    # [Copy full implementation from above]
    ...

def check_feature_information_content(self, X, selected_features, variance_threshold=0.01, quasi_constant_threshold=0.99):
    """Check if features have sufficient information content for ML."""
    # [Copy full implementation from above]
    ...
```

Add to pipeline in `feature_generation_final_feature_selection_step.py`:

```python
# After MI stability analysis:

# Phase 3: Data Leakage Detection (CRITICAL)
tprint_info("🔍 Detecting potential data leakage...")
leakage_detection = temp_component.detect_potential_leakage(X, y, selected_features)
analysis_results['leakage_detection'] = leakage_detection

# Phase 3: Information Content Check
tprint_info("📊 Checking feature information content...")
info_content = temp_component.check_feature_information_content(X, selected_features)
analysis_results['information_content'] = info_content
```

Add to `get_enhanced_analysis()`:

```python
'leakage_detection': getattr(self, 'leakage_detection', None),
'information_content': getattr(self, 'information_content_analysis', None),
```

---

## ✅ **Conclusion**

**Current state:** You have **excellent** ML validation coverage (8/10)

**With Phase 3:** You'll have **comprehensive** coverage (9.5/10)

**Missing items are "nice-to-have"** not critical - you can implement them later as needed.

**Recommendation:** Implement Phase 3 (50 minutes) for production-grade feature selection.

