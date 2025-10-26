# Regime Clustering Comprehensive Report

**Generated**: 2025-10-26T23:28:18.484383  
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
- **Processing Time**: 29.98 seconds
- **Labels Changed**: 1,658 (86.4%)
- **Refinement Applied**: ✅ Yes

---

## 🔍 Detailed Cluster Analysis

### Cluster Distribution

| Cluster ID | Sample Count | Percentage | Type |
|------------|--------------|------------|------|
| **-1** | 90 | 4.7% | Noise |
| **0** | 230 | 12.0% | Regime |
| **1** | 320 | 16.7% | Regime |
| **2** | 320 | 16.7% | Regime |
| **3** | 320 | 16.7% | Regime |
| **4** | 320 | 16.7% | Regime |
| **5** | 320 | 16.7% | Regime |

### Individual Cluster Analysis

#### 🎯 Cluster 0

**Basic Statistics:**
- **Size**: 230 samples (12.0%)
- **Density**: Medium
- **Stability**: Low (0.163)

**Temporal Analysis:**
- **Contiguous Segments**: 179
- **Average Segment Length**: 1.3 periods
- **Longest Segment**: 10 periods
- **Fragmentation Score**: 0.778

**Quality Metrics:**
- **Cluster Cohesion**: 0.222
- **Boundary Clarity**: 0.223
- **Temporal Consistency**: 0.043

**Refinement Changes:**
- **Original Size**: 1,224 samples
- **Size Change**: -994 samples (-81.2%)
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
- **Stability**: Low (0.340)

**Temporal Analysis:**
- **Contiguous Segments**: 183
- **Average Segment Length**: 1.7 periods
- **Longest Segment**: 52 periods
- **Fragmentation Score**: 0.572

**Quality Metrics:**
- **Cluster Cohesion**: 0.428
- **Boundary Clarity**: 0.429
- **Temporal Consistency**: 0.163

**Refinement Changes:**
- **Original Size**: 0 samples
- **Size Change**: +320 samples (+0.0%)
- **Refinement Impact**: Improved

---

#### 🎯 Cluster 4

**Basic Statistics:**
- **Size**: 320 samples (16.7%)
- **Density**: Medium
- **Stability**: Low (0.396)

**Temporal Analysis:**
- **Contiguous Segments**: 178
- **Average Segment Length**: 1.8 periods
- **Longest Segment**: 96 periods
- **Fragmentation Score**: 0.556

**Quality Metrics:**
- **Cluster Cohesion**: 0.444
- **Boundary Clarity**: 0.445
- **Temporal Consistency**: 0.300

**Refinement Changes:**
- **Original Size**: 0 samples
- **Size Change**: +320 samples (+0.0%)
- **Refinement Impact**: Improved

---

#### 🎯 Cluster 5

**Basic Statistics:**
- **Size**: 320 samples (16.7%)
- **Density**: Medium
- **Stability**: Low (0.262)

**Temporal Analysis:**
- **Contiguous Segments**: 201
- **Average Segment Length**: 1.6 periods
- **Longest Segment**: 13 periods
- **Fragmentation Score**: 0.628

**Quality Metrics:**
- **Cluster Cohesion**: 0.372
- **Boundary Clarity**: 0.373
- **Temporal Consistency**: 0.041

**Refinement Changes:**
- **Original Size**: 0 samples
- **Size Change**: +320 samples (+0.0%)
- **Refinement Impact**: Improved

---

## 🔧 Refinement Process Analysis

### Temporal Stabilization
- **Changes Applied**: 1,658 labels
- **Stability Improvement**: -3.077
- **Noise Reduction**: 75.9%

### Economic Validation
- **Economic Distinction**: 1.000
- **Validation Passed**: ✅ Yes
- **Cluster Separation**: 0.800

### Cluster Merging
- **Clusters Merged**: -3
- **Size Optimization**: -40.8%
- **Fragmentation Reduction**: -1.000

## 📈 Quality Metrics Summary

### Overall Performance
- **Processing Time**: 29.98 seconds
- **Memory Efficiency**: 50.0%
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
