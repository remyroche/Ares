# Regime Clustering Comprehensive Report

**Generated**: 2025-10-26T23:24:12.906262  
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
- **Processing Time**: 30.00 seconds
- **Labels Changed**: 1,649 (85.9%)
- **Refinement Applied**: ✅ Yes

---

## 🔍 Detailed Cluster Analysis

### Cluster Distribution

| Cluster ID | Sample Count | Percentage | Type |
|------------|--------------|------------|------|
| **-1** | 100 | 5.2% | Noise |
| **0** | 220 | 11.5% | Regime |
| **1** | 320 | 16.7% | Regime |
| **2** | 320 | 16.7% | Regime |
| **3** | 320 | 16.7% | Regime |
| **4** | 320 | 16.7% | Regime |
| **5** | 320 | 16.7% | Regime |

### Individual Cluster Analysis

#### 🎯 Cluster 0

**Basic Statistics:**
- **Size**: 220 samples (11.5%)
- **Density**: Medium
- **Stability**: Low (0.094)

**Temporal Analysis:**
- **Contiguous Segments**: 191
- **Average Segment Length**: 1.2 periods
- **Longest Segment**: 4 periods
- **Fragmentation Score**: 0.868

**Quality Metrics:**
- **Cluster Cohesion**: 0.132
- **Boundary Clarity**: 0.132
- **Temporal Consistency**: 0.018

**Refinement Changes:**
- **Original Size**: 1,224 samples
- **Size Change**: -1,004 samples (-82.0%)
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
- **Stability**: Low (0.399)

**Temporal Analysis:**
- **Contiguous Segments**: 138
- **Average Segment Length**: 2.3 periods
- **Longest Segment**: 18 periods
- **Fragmentation Score**: 0.431

**Quality Metrics:**
- **Cluster Cohesion**: 0.569
- **Boundary Clarity**: 0.571
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
- **Contiguous Segments**: 195
- **Average Segment Length**: 1.6 periods
- **Longest Segment**: 40 periods
- **Fragmentation Score**: 0.609

**Quality Metrics:**
- **Cluster Cohesion**: 0.391
- **Boundary Clarity**: 0.392
- **Temporal Consistency**: 0.125

**Refinement Changes:**
- **Original Size**: 0 samples
- **Size Change**: +320 samples (+0.0%)
- **Refinement Impact**: Improved

---

#### 🎯 Cluster 4

**Basic Statistics:**
- **Size**: 320 samples (16.7%)
- **Density**: Medium
- **Stability**: Low (0.372)

**Temporal Analysis:**
- **Contiguous Segments**: 180
- **Average Segment Length**: 1.8 periods
- **Longest Segment**: 77 periods
- **Fragmentation Score**: 0.562

**Quality Metrics:**
- **Cluster Cohesion**: 0.438
- **Boundary Clarity**: 0.439
- **Temporal Consistency**: 0.241

**Refinement Changes:**
- **Original Size**: 0 samples
- **Size Change**: +320 samples (+0.0%)
- **Refinement Impact**: Improved

---

#### 🎯 Cluster 5

**Basic Statistics:**
- **Size**: 320 samples (16.7%)
- **Density**: Medium
- **Stability**: Low (0.292)

**Temporal Analysis:**
- **Contiguous Segments**: 190
- **Average Segment Length**: 1.7 periods
- **Longest Segment**: 20 periods
- **Fragmentation Score**: 0.594

**Quality Metrics:**
- **Cluster Cohesion**: 0.406
- **Boundary Clarity**: 0.408
- **Temporal Consistency**: 0.062

**Refinement Changes:**
- **Original Size**: 0 samples
- **Size Change**: +320 samples (+0.0%)
- **Refinement Impact**: Improved

---

## 🔧 Refinement Process Analysis

### Temporal Stabilization
- **Changes Applied**: 1,649 labels
- **Stability Improvement**: -3.162
- **Noise Reduction**: 73.3%

### Economic Validation
- **Economic Distinction**: 1.000
- **Validation Passed**: ✅ Yes
- **Cluster Separation**: 0.800

### Cluster Merging
- **Clusters Merged**: -3
- **Size Optimization**: -41.1%
- **Fragmentation Reduction**: -1.000

## 📈 Quality Metrics Summary

### Overall Performance
- **Processing Time**: 30.00 seconds
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
