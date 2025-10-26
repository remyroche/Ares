# Regime Clustering Comprehensive Report

**Generated**: 2025-10-26T23:25:27.644108  
**Symbol**: ETHUSDT  
**Exchange**: binance  
**Timeframe**: 1h  
**Execution Mode**: light  
**Clustering Method**: hdbscan_refined  

---

## 📊 Executive Summary

This report provides a comprehensive analysis of the regime clustering refinement process, including detailed metrics for each cluster, refinement statistics, and quality assessments.

### Key Results
- **Original Clusters**: 4
- **Refined Clusters**: 6
- **Processing Time**: 16.41 seconds
- **Labels Changed**: 1,662 (86.6%)
- **Refinement Applied**: ✅ Yes

---

## 🔍 Detailed Cluster Analysis

### Cluster Distribution

| Cluster ID | Sample Count | Percentage | Type |
|------------|--------------|------------|------|
| **-1** | 92 | 4.8% | Noise |
| **0** | 228 | 11.9% | Regime |
| **1** | 320 | 16.7% | Regime |
| **2** | 320 | 16.7% | Regime |
| **3** | 320 | 16.7% | Regime |
| **4** | 320 | 16.7% | Regime |
| **5** | 320 | 16.7% | Regime |

### Individual Cluster Analysis

#### 🎯 Cluster 0

**Basic Statistics:**
- **Size**: 228 samples (11.9%)
- **Density**: Medium
- **Stability**: Low (0.126)

**Temporal Analysis:**
- **Contiguous Segments**: 187
- **Average Segment Length**: 1.2 periods
- **Longest Segment**: 4 periods
- **Fragmentation Score**: 0.820

**Quality Metrics:**
- **Cluster Cohesion**: 0.180
- **Boundary Clarity**: 0.181
- **Temporal Consistency**: 0.018

**Refinement Changes:**
- **Original Size**: 1,224 samples
- **Size Change**: -996 samples (-81.4%)
- **Refinement Impact**: Reduced

---

#### 🎯 Cluster 1

**Basic Statistics:**
- **Size**: 320 samples (16.7%)
- **Density**: Medium
- **Stability**: Medium (0.469)

**Temporal Analysis:**
- **Contiguous Segments**: 123
- **Average Segment Length**: 2.6 periods
- **Longest Segment**: 56 periods
- **Fragmentation Score**: 0.384

**Quality Metrics:**
- **Cluster Cohesion**: 0.616
- **Boundary Clarity**: 0.618
- **Temporal Consistency**: 0.175

**Refinement Changes:**
- **Original Size**: 259 samples
- **Size Change**: +61 samples (+23.6%)
- **Refinement Impact**: Improved

---

#### 🎯 Cluster 2

**Basic Statistics:**
- **Size**: 320 samples (16.7%)
- **Density**: Medium
- **Stability**: Medium (0.401)

**Temporal Analysis:**
- **Contiguous Segments**: 137
- **Average Segment Length**: 2.3 periods
- **Longest Segment**: 18 periods
- **Fragmentation Score**: 0.428

**Quality Metrics:**
- **Cluster Cohesion**: 0.572
- **Boundary Clarity**: 0.574
- **Temporal Consistency**: 0.056

**Refinement Changes:**
- **Original Size**: 63 samples
- **Size Change**: +257 samples (+407.9%)
- **Refinement Impact**: Improved

---

#### 🎯 Cluster 3

**Basic Statistics:**
- **Size**: 320 samples (16.7%)
- **Density**: Medium
- **Stability**: Low (0.302)

**Temporal Analysis:**
- **Contiguous Segments**: 199
- **Average Segment Length**: 1.6 periods
- **Longest Segment**: 48 periods
- **Fragmentation Score**: 0.622

**Quality Metrics:**
- **Cluster Cohesion**: 0.378
- **Boundary Clarity**: 0.379
- **Temporal Consistency**: 0.150

**Refinement Changes:**
- **Original Size**: 0 samples
- **Size Change**: +320 samples (+0.0%)
- **Refinement Impact**: Improved

---

#### 🎯 Cluster 4

**Basic Statistics:**
- **Size**: 320 samples (16.7%)
- **Density**: Medium
- **Stability**: Low (0.366)

**Temporal Analysis:**
- **Contiguous Segments**: 187
- **Average Segment Length**: 1.7 periods
- **Longest Segment**: 85 periods
- **Fragmentation Score**: 0.584

**Quality Metrics:**
- **Cluster Cohesion**: 0.416
- **Boundary Clarity**: 0.417
- **Temporal Consistency**: 0.266

**Refinement Changes:**
- **Original Size**: 0 samples
- **Size Change**: +320 samples (+0.0%)
- **Refinement Impact**: Improved

---

#### 🎯 Cluster 5

**Basic Statistics:**
- **Size**: 320 samples (16.7%)
- **Density**: Medium
- **Stability**: Low (0.318)

**Temporal Analysis:**
- **Contiguous Segments**: 175
- **Average Segment Length**: 1.8 periods
- **Longest Segment**: 15 periods
- **Fragmentation Score**: 0.547

**Quality Metrics:**
- **Cluster Cohesion**: 0.453
- **Boundary Clarity**: 0.455
- **Temporal Consistency**: 0.047

**Refinement Changes:**
- **Original Size**: 0 samples
- **Size Change**: +320 samples (+0.0%)
- **Refinement Impact**: Improved

---

## 🔧 Refinement Process Analysis

### Temporal Stabilization
- **Changes Applied**: 1,662 labels
- **Stability Improvement**: -3.112
- **Noise Reduction**: 75.4%

### Economic Validation
- **Economic Distinction**: 1.000
- **Validation Passed**: ✅ Yes
- **Cluster Separation**: 0.800

### Cluster Merging
- **Clusters Merged**: -3
- **Size Optimization**: -40.9%
- **Fragmentation Reduction**: -1.000

## 📈 Quality Metrics Summary

### Overall Performance
- **Processing Time**: 16.41 seconds
- **Memory Efficiency**: 72.7%
- **Refinement Success**: ✅ Yes

### Technical Details

**Input Data:**
- **Original Labels**: 1,920 samples
- **HDBSCAN Artifacts**: ✅ Available
- **Refinement Methods**: Temporal Stabilization, Economic Validation, Cluster Merging

**Output Artifacts:**
- **Refined Labels**: 1,920 samples
- **Cluster Centers**: 6 clusters
- **Metadata**: Complete refinement history

---

## 🎯 Recommendations

### For Trading Strategy
- **Optimal Regime Count**: 6 regimes identified
- **Regime Stability**: Low
- **Strategy Adaptation**: Recommended

### For Further Analysis
- **Cluster Validation**: Consider cross-validation with different time periods
- **Economic Profiling**: Analyze regime-specific economic characteristics
- **Temporal Patterns**: Investigate regime transition patterns
- **Feature Importance**: Identify key features driving regime classification

---

## 📋 Artifact Summary

**Generated Artifacts:**
- `refined_regime_clusters`: Main refined cluster data
- `refinement_metadata`: Complete refinement process metadata
- `regime_clustering_metrics`: Performance and quality metrics

**File Locations:**
- **Artifacts**: `artifacts/pre_training/{symbol}/{exchange}/long/Analyst/regime_clustering/`
- **Report**: `outcomes/{report_filename}`

---

*Report generated by Ares Regime Clustering Step v1.0*
*Generated on: {datetime.now().isoformat()}*
