# Regime Clustering Comprehensive Report

**Generated**: 2025-10-26T23:29:50.520360  
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
- **Processing Time**: 15.34 seconds
- **Labels Changed**: 1,652 (86.0%)
- **Refinement Applied**: ✅ Yes

---

## 🔍 Detailed Cluster Analysis

### Cluster Distribution

| Cluster ID | Sample Count | Percentage | Type |
|------------|--------------|------------|------|
| **-1** | 82 | 4.3% | Noise |
| **0** | 238 | 12.4% | Regime |
| **1** | 320 | 16.7% | Regime |
| **2** | 320 | 16.7% | Regime |
| **3** | 320 | 16.7% | Regime |
| **4** | 320 | 16.7% | Regime |
| **5** | 320 | 16.7% | Regime |

### Individual Cluster Analysis

#### 🎯 Cluster 0

**Basic Statistics:**
- **Size**: 238 samples (12.4%)
- **Density**: Medium
- **Stability**: Low (0.161)

**Temporal Analysis:**
- **Contiguous Segments**: 183
- **Average Segment Length**: 1.3 periods
- **Longest Segment**: 5 periods
- **Fragmentation Score**: 0.769

**Quality Metrics:**
- **Cluster Cohesion**: 0.231
- **Boundary Clarity**: 0.232
- **Temporal Consistency**: 0.021

**Refinement Changes:**
- **Original Size**: 1,224 samples
- **Size Change**: -986 samples (-80.6%)
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
- **Stability**: Medium (0.403)

**Temporal Analysis:**
- **Contiguous Segments**: 136
- **Average Segment Length**: 2.4 periods
- **Longest Segment**: 18 periods
- **Fragmentation Score**: 0.425

**Quality Metrics:**
- **Cluster Cohesion**: 0.575
- **Boundary Clarity**: 0.577
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
- **Stability**: Low (0.296)

**Temporal Analysis:**
- **Contiguous Segments**: 201
- **Average Segment Length**: 1.6 periods
- **Longest Segment**: 46 periods
- **Fragmentation Score**: 0.628

**Quality Metrics:**
- **Cluster Cohesion**: 0.372
- **Boundary Clarity**: 0.373
- **Temporal Consistency**: 0.144

**Refinement Changes:**
- **Original Size**: 0 samples
- **Size Change**: +320 samples (+0.0%)
- **Refinement Impact**: Improved

---

#### 🎯 Cluster 4

**Basic Statistics:**
- **Size**: 320 samples (16.7%)
- **Density**: Medium
- **Stability**: Low (0.374)

**Temporal Analysis:**
- **Contiguous Segments**: 186
- **Average Segment Length**: 1.7 periods
- **Longest Segment**: 91 periods
- **Fragmentation Score**: 0.581

**Quality Metrics:**
- **Cluster Cohesion**: 0.419
- **Boundary Clarity**: 0.420
- **Temporal Consistency**: 0.284

**Refinement Changes:**
- **Original Size**: 0 samples
- **Size Change**: +320 samples (+0.0%)
- **Refinement Impact**: Improved

---

#### 🎯 Cluster 5

**Basic Statistics:**
- **Size**: 320 samples (16.7%)
- **Density**: Medium
- **Stability**: Low (0.283)

**Temporal Analysis:**
- **Contiguous Segments**: 192
- **Average Segment Length**: 1.7 periods
- **Longest Segment**: 15 periods
- **Fragmentation Score**: 0.600

**Quality Metrics:**
- **Cluster Cohesion**: 0.400
- **Boundary Clarity**: 0.401
- **Temporal Consistency**: 0.047

**Refinement Changes:**
- **Original Size**: 0 samples
- **Size Change**: +320 samples (+0.0%)
- **Refinement Impact**: Improved

---

## 🔧 Refinement Process Analysis

### Temporal Stabilization
- **Changes Applied**: 1,652 labels
- **Stability Improvement**: -3.135
- **Noise Reduction**: 78.1%

### Economic Validation
- **Economic Distinction**: 1.000
- **Validation Passed**: ✅ Yes
- **Cluster Separation**: 0.800

### Cluster Merging
- **Clusters Merged**: -3
- **Size Optimization**: -40.6%
- **Fragmentation Reduction**: -1.000

## 📈 Quality Metrics Summary

### Overall Performance
- **Processing Time**: 15.34 seconds
- **Memory Efficiency**: 74.4%
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
