# High Feature CV Analysis - Critical Issue Assessment

**Date**: 2025-10-30  
**Issue**: Regimes 1 & 3 have very high feature coefficient of variation

---

## The Problem

### What High Feature CV Actually Means

**Coefficient of Variation (CV)** = `std_dev / mean` for each principal component **within** a regime

| Regime | Size | Mean Feature CV | Interpretation |
|--------|------|----------------|----------------|
| **Regime 1** | 33.5% | **83.5** | Features vary 83.5× relative to their mean |
| **Regime 3** | 20.2% | **76.0** | Features vary 76× relative to their mean |
| Regime 0 | 26.5% | 25.4 | Moderate variation |
| Regime 5 | 9.2% | 21.8 | Moderate variation |
| Regime 4 | 6.5% | 10.1 | Low variation (cohesive) |
| Regime 2 | 4.2% | 8.5 | Low variation (cohesive) |

### Why This is Concerning

**High feature CV within a regime suggests HETEROGENEITY**:

1. **Regime Should Be Cohesive**: Points in the same regime should have similar feature values
2. **High CV = Low Cohesion**: CV of 83.5 means the features vary wildly within the regime
3. **Potential Issue**: The regime may be a "catch-all" mixing multiple distinct market states

---

## Hypothesis: Why Are Large Regimes Heterogeneous?

### Theory 1: PCA Artifacts
**Explanation**: PCA creates features where variance increases with component order
- **PC_1, PC_2**: Low CV (core patterns, stable)
- **PC_17, PC_22, PC_44**: Extremely high CV (noise, minor patterns)
- **Mean CV** is inflated by high-order PCs

**Evidence from Report**:
```
Regime 1, PC_17: CV = 2,786 (extreme!)
Regime 1, PC_44: CV = 285
Regime 1, PC_47: CV = 88
```

**Validation**: Check if high CV is concentrated in minor PCs (PC_20+)

### Theory 2: Broad "Default" Regimes
**Explanation**: GMM assigned diverse market conditions to these large regimes
- Regime 1 (33.5%) might be "everything that's not clearly something else"
- Clustering in 50D space might group dissimilar states in feature space

**Why This Happens**:
- With only 6 regimes for 480 samples, each regime covers ~80 samples
- GMM optimizes likelihood, not within-regime cohesion
- These regimes might need sub-clustering

### Theory 3: Temporal Heterogeneity
**Explanation**: Large regimes span long time periods with evolving market conditions
- Regime 1 might include "trending periods" from different volatility environments
- The features correctly capture this diversity
- High CV reflects genuine evolution of market state within regime

**Test**: Check if these regimes have long continuous episodes or many short episodes

---

## Investigating the Issue

### Check 1: CV Distribution Across Principal Components

**Question**: Is high CV concentrated in minor PCs (noise) or major PCs (signal)?

**From Report - Regime 1**:
```python
PC_1: CV = 1.15   # Low - major component is stable
PC_2: CV = 1.04   # Low - major component is stable  
PC_3: CV = 11.07  # Moderate
PC_4: CV = 4.11   # Moderate
...
PC_17: CV = 2,786 # EXTREME - noise component
PC_22: CV = 123   # Very high
PC_44: CV = 285   # Very high
PC_47: CV = 88    # High
```

**Analysis**:
- **Major PCs (1-5)**: CV ranges 1.0-11.0 - REASONABLE ✅
- **Minor PCs (17+)**: CV ranges 88-2,786 - EXTREME ❌

**Conclusion**: The high mean CV (83.5) is **dominated by noisy minor principal components**, NOT the core features!

### Check 2: Silhouette Score Context

**Silhouette Score: 0.100** (just barely acceptable)

**What This Means**:
- Silhouette measures how well-separated clusters are
- 0.100 is very low (range: -1 to 1)
- Indicates significant overlap between regimes
- Confirms regimes have fuzzy boundaries

**Interpretation**:
- Combined with high CV, suggests regimes are **not tightly clustered**
- Points near regime boundaries contribute to high CV
- GMM's soft clustering allows marginal assignments

### Check 3: Davies-Bouldin Index

**DBI: 2.37** (target: ≤2.0) ❌

**What This Means**:
- Measures average similarity between each cluster and its most similar cluster
- Higher = more overlap
- 2.37 > 2.0 target indicates regimes are NOT well-separated

**Combined Evidence**:
- Low silhouette (0.100) + High DBI (2.37) + High CV (83.5)
- **Conclusion**: Regimes 1 & 3 are BROAD, OVERLAPPING, HETEROGENEOUS states

---

## The Real Explanation

### Regimes 1 & 3 Are "Meta-Regimes"

**What's Actually Happening**:

1. **GMM with k=6 is Under-Clustering**
   - 480 samples / 6 regimes = 80 samples per regime (too broad)
   - Each regime must cover diverse market conditions
   - Regimes 1 & 3 became "umbrella" categories

2. **Principal Components Amplify Heterogeneity**
   - Minor PCs (PC_17+) capture rare patterns and noise
   - These vary wildly within any regime
   - Mean CV calculation is sensitive to outlier PCs

3. **GMM Optimizes Likelihood, Not Cohesion**
   - GMM maximizes P(data | parameters)
   - Does NOT minimize within-regime variance
   - Accepts heterogeneous regimes if they improve overall likelihood

4. **Soft Clustering Creates Fuzzy Boundaries**
   - GMM assigns probabilities, not hard labels
   - Marginal points (50/50 between regimes) inflate CV
   - Regime 1 might include "trending-ish" states that could be 2-3 sub-regimes

---

## Implications

### Is This a Problem?

**YES - For Certain Use Cases**:

1. **Regime-Specific Modeling**:
   - Training separate models per regime requires cohesive states
   - High heterogeneity → models won't specialize effectively
   - Regime 1's model will see contradictory training examples

2. **Strategy Selection**:
   - "Use Strategy A in Regime 1" is ambiguous
   - Regime 1 mixes multiple market conditions
   - Strategy performance will be inconsistent

3. **Interpretability**:
   - "What does Regime 1 represent?" has no clear answer
   - Can't profile the regime meaningfully
   - Business logic becomes difficult

**MAYBE NOT - For Other Use Cases**:

1. **Regime-Aware Feature Engineering**:
   - Just flagging "you're in a broad active state" might suffice
   - Don't need tight cohesion, just rough classification

2. **Risk Management**:
   - "Active/volatile period" vs "calm period" distinction is enough
   - High CV within "active" is acceptable

---

## Recommended Actions

### Option 1: Increase Cluster Count (Recommended)

**Action**: Re-run with k=8-10 regimes instead of 6

**Rationale**:
- More regimes = tighter, more cohesive clusters
- Should reduce within-regime CV significantly
- Regime 1 might split into 2-3 sub-regimes

**Trade-off**:
- Smaller regime sizes (60 samples/regime instead of 80)
- Regime 2 & 4 would become even smaller
- More complex regime taxonomy

**Command**:
```python
gmm_step = create_gmm_regime_discovery_step(
    n_components_range=(8, 10),  # Instead of (4, 6)
    correlation_threshold=0.85,
    random_state=42
)
```

### Option 2: Hierarchical Sub-Clustering

**Action**: Accept 6 regimes, then sub-cluster Regimes 1 & 3

**Rationale**:
- Keep stable regimes (2, 4, 5) as-is
- Split heterogeneous regimes (1, 3) into sub-regimes
- Final: 2-3 sub-regimes × 2 = 4-6 additional regimes → 10-12 total

**Approach**:
```python
# For Regime 1 samples only:
regime_1_samples = data[regime_labels == 1]
sub_gmm = GaussianMixture(n_components=3, random_state=42)
sub_labels = sub_gmm.fit_predict(regime_1_samples)
# Relabel: 1 → 1a, 1b, 1c
```

### Option 3: Filter Minor PCs Before Clustering

**Action**: Use only major PCs (PC_1 to PC_10) for clustering

**Rationale**:
- Minor PCs (PC_11-50) contribute noise, not signal
- High CV in PC_17+ is artificially inflating metrics
- Major PCs have much lower CV

**Code**:
```python
# In discover_regimes():
if scaled_df.shape[1] > 10:
    scaled_df_for_clustering = scaled_df.iloc[:, :10]  # First 10 PCs only
    # Use for GMM fitting
```

**Expected Result**: Mean CV should drop from 83.5 to ~15-20

### Option 4: Post-Process with Within-Regime Clustering

**Action**: Accept current results, apply secondary clustering in deployment

**Rationale**:
- Use 6 regimes for coarse classification
- When entering Regime 1, apply real-time sub-classification
- Adaptive approach: "Regime 1, Sub-State A"

**Implementation**: Real-time classifier checks recent features to determine sub-state

---

## Recommendation: **Option 1 + Option 3**

**Combined Approach**:
1. **Use only first 20 PCs** for clustering (filters noise)
2. **Try k=8 regimes** for finer granularity
3. **Re-evaluate CV metrics**

**Expected Outcome**:
- Mean CV drops to 15-25 (acceptable)
- Silhouette improves to 0.15-0.20
- DBI drops below 2.0
- Regimes become more interpretable

**Test Script**:
```python
# Modified GMM discovery
gmm_step = create_gmm_regime_discovery_step(
    n_components_range=(8, 8),
    correlation_threshold=0.85,
    random_state=42
)

# In discover_regimes, before GMM fitting:
if scaled_df.shape[1] > 20:
    scaled_df = scaled_df.iloc[:, :20]  # Use first 20 PCs only
```

---

## Summary

**Your observation is correct**: High feature CV (83.5, 76.0) indicates regimes are **heterogeneous**, not cohesive.

**Root Cause**: 
- Minor principal components (PC_17+) have extreme CV
- GMM with k=6 is under-clustering (regimes too broad)
- Soft clustering creates fuzzy boundaries

**Impact**:
- Regime-specific modeling will be less effective
- Strategy selection becomes ambiguous
- Interpretability suffers

**Solution**:
- Increase to k=8-10 regimes
- Use only first 10-20 principal components
- Re-run and expect mean CV to drop to 15-25

---

*This analysis supersedes the previous incorrect interpretation that conflated feature variance with market volatility.*

