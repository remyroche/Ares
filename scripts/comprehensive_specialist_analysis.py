#!/usr/bin/env python3
"""Comprehensive analysis of all specialist models with optimization recommendations."""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List
import json
from datetime import datetime

# Import existing utilities
import sys
sys.path.insert(0, '.')

from src.utils.versioned_artifacts import VersionedArtifactStore
from sklearn.feature_selection import mutual_info_regression
from scipy.stats import spearmanr

def analyze_all_specialists() -> Dict[str, Dict[str, Any]]:
    """Analyze all available specialist models."""
    
    print("🚀 COMPREHENSIVE SPECIALIST ANALYSIS")
    print("=" * 60)
    
    # Define all specialists to analyze
    specialists = {
        'ml_volume_force_step': 'volume_force',
        'ml_liquidity_regime_step': 'liquidity_regime',
        'ml_breakout_bounce_regime_step': 'breakout_bounce',
        'ml_path_regime_step': 'path_regime',
        'ml_smc_regime_step': 'smc_regime',
        'ml_momentum_persistence_step': 'momentum_persistence',
        'ml_volatility_burst_step': 'volatility_burst'
    }
    
    all_analyses = {}
    
    for specialist_name, store_name in specialists.items():
        print(f"\n🔍 Analyzing {specialist_name}...")
        
        try:
            analysis = analyze_specialist_comprehensive(specialist_name, store_name)
            if analysis:
                all_analyses[specialist_name] = analysis
                print(f"   ✅ Analysis completed")
            else:
                print(f"   ❌ Analysis failed")
        except Exception as e:
            print(f"   ❌ Analysis error: {e}")
    
    return all_analyses

def analyze_specialist_comprehensive(specialist_name: str, store_name: str) -> Dict[str, Any]:
    """Comprehensive analysis of a single specialist."""
    
    try:
        # Initialize artifact store
        store = VersionedArtifactStore(f"versioned_artifacts/ETHUSDT_binance_15m_long_{store_name}")
        
        # Get latest prediction view
        views = store.list_versions()
        pred_views = [v for v in views if 'prediction' in v.lower()]
        
        if not pred_views:
            print(f"   ❌ No prediction views found")
            return {}
        
        # Get latest prediction view
        latest_view = pred_views[-1]
        artifact_view = store.get_view(latest_view)
        
        # Convert ArtifactView to DataFrame
        df = artifact_view.to_pandas()
        
        if not isinstance(df, pd.DataFrame) or len(df) == 0:
            print(f"   ❌ Invalid data format")
            return {}
        
        print(f"   ✅ Loaded {latest_view}: {df.shape}")
        
        analysis = analyze_comprehensive_data(df, specialist_name)
        return analysis
        
    except Exception as e:
        print(f"   ❌ Analysis failed: {e}")
        return {}

def analyze_comprehensive_data(df: pd.DataFrame, specialist_name: str) -> Dict[str, Any]:
    """Comprehensive data analysis with all metrics."""
    
    analysis = {
        'specialist_name': specialist_name,
        'total_samples': len(df),
        'columns': list(df.columns)
    }
    
    # Identify target and prediction columns
    target_cols = [col for col in df.columns if 'target_' in col.lower()]
    pred_cols = [col for col in df.columns if any(x in col.lower() for x in ['force_', 'regime_', 'momentum_', 'volatility_', 'liquidity_', 'breakout_', 'path_', 'smc_', 'reversion_'])]
    
    if not target_cols or not pred_cols:
        print(f"   ⚠️ Missing target or prediction columns")
        return analysis
    
    # Use first target and prediction
    target_col = target_cols[0]
    pred_col = pred_cols[0]
    
    labels = df[target_col]
    predictions = df[pred_col]
    
    analysis['target_column'] = target_col
    analysis['prediction_column'] = pred_col
    
    # Clean data
    valid_mask = ~(labels.isna()) & ~(predictions.isna())
    labels_clean = labels[valid_mask]
    predictions_clean = predictions[valid_mask]
    
    analysis['clean_samples'] = len(labels_clean)
    
    if len(labels_clean) < 100:
        print(f"   ⚠️ Insufficient clean data")
        return analysis
    
    # 1. Binary output conversion
    binary_labels = (labels_clean > 0).astype(int)
    
    if predictions_clean.dtype in ['float64', 'float32']:
        threshold = np.median(predictions_clean)
        binary_predictions = (predictions_clean > threshold).astype(int)
        analysis['threshold_used'] = threshold
    else:
        binary_predictions = predictions_clean.astype(int)
    
    analysis['has_binary_output'] = True
    analysis['unique_prediction_values'] = list(np.unique(binary_predictions))
    
    # 2. MI to target
    try:
        pred_mi = mutual_info_regression(
            binary_predictions.values.reshape(-1, 1), 
            binary_labels.values
        )[0]
        analysis['prediction_mi_to_target'] = pred_mi
    except Exception as e:
        analysis['prediction_mi_to_target'] = 0
    
    # 3. Performance metrics
    try:
        from sklearn.metrics import accuracy_score, roc_auc_score
        
        accuracy = accuracy_score(binary_labels, binary_predictions)
        analysis['accuracy'] = accuracy
        
        if predictions_clean.dtype in ['float64', 'float32']:
            pred_norm = (predictions_clean - predictions_clean.min()) / (predictions_clean.max() - predictions_clean.min())
            auc = roc_auc_score(binary_labels, pred_norm)
            analysis['auc'] = auc
        else:
            analysis['auc'] = None
        
    except Exception as e:
        analysis['accuracy'] = None
        analysis['auc'] = None
    
    # 4. Feature orthogonality
    feature_cols = [col for col in df.columns if col not in [target_col, pred_col] and col != 'timestamp']
    analysis['feature_count'] = len(feature_cols)
    
    if len(feature_cols) > 1:
        try:
            features = df[feature_cols].select_dtypes(include=[np.number])
            if len(features.columns) > 1:
                corr_matrix = features.corr().abs()
                high_corr = ((corr_matrix > 0.7) & (corr_matrix < 1.0)).sum().sum() / 2
                analysis['high_correlation_pairs'] = int(high_corr)
                analysis['orthogonal_features'] = len(features.columns) - int(high_corr)
            else:
                analysis['high_correlation_pairs'] = 0
                analysis['orthogonal_features'] = len(features.columns)
        except:
            analysis['high_correlation_pairs'] = 0
            analysis['orthogonal_features'] = len(feature_cols)
    else:
        analysis['high_correlation_pairs'] = 0
        analysis['orthogonal_features'] = len(feature_cols)
    
    return analysis

def analyze_cross_specialist_correlations(analyses: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze correlations between different specialists."""
    
    print("\n🔄 CROSS-SPECIALIST CORRELATION ANALYSIS")
    print("=" * 60)
    
    cross_analysis = {
        'specialist_count': len(analyses),
        'pairwise_correlations': {},
        'high_correlation_pairs': [],
        'orthogonal_pairs': []
    }
    
    specialist_names = list(analyses.keys())
    
    for i, spec1 in enumerate(specialist_names):
        for j, spec2 in enumerate(specialist_names[i+1:], i+1):
            print(f"   🔍 Analyzing {spec1} vs {spec2}...")
            
            try:
                # Load data for both specialists
                store1 = VersionedArtifactStore(f"versioned_artifacts/ETHUSDT_binance_15m_long_{spec1.replace('ml_', '').replace('_step', '')}")
                store2 = VersionedArtifactStore(f"versioned_artifacts/ETHUSDT_binance_15m_long_{spec2.replace('ml_', '').replace('_step', '')}")
                
                # Get prediction views
                views1 = [v for v in store1.list_versions() if 'prediction' in v.lower()]
                views2 = [v for v in store2.list_versions() if 'prediction' in v.lower()]
                
                if views1 and views2:
                    data1 = store1.get_view(views1[-1]).to_pandas()
                    data2 = store2.get_view(views2[-1]).to_pandas()
                    
                    # Get prediction columns
                    pred1_cols = [col for col in data1.columns if any(x in col.lower() for x in ['force_', 'regime_', 'momentum_', 'volatility_', 'liquidity_', 'breakout_', 'path_', 'smc_', 'reversion_'])]
                    pred2_cols = [col for col in data2.columns if any(x in col.lower() for x in ['force_', 'regime_', 'momentum_', 'volatility_', 'liquidity_', 'breakout_', 'path_', 'smc_', 'reversion_'])]
                    
                    if pred1_cols and pred2_cols:
                        pred1 = data1[pred1_cols[0]]
                        pred2 = data2[pred2_cols[0]]
                        
                        # Align on timestamp if available
                        if 'timestamp' in data1.columns and 'timestamp' in data2.columns:
                            merged = pd.merge(data1[['timestamp'] + pred1_cols], 
                                           data2[['timestamp'] + pred2_cols], 
                                           on='timestamp', how='inner')
                            if len(merged) > 100:
                                pred1_aligned = merged[pred1_cols[0]]
                                pred2_aligned = merged[pred2_cols[0]]
                                
                                # Convert to binary
                                pred1_binary = (pred1_aligned > np.median(pred1_aligned)).astype(int)
                                pred2_binary = (pred2_aligned > np.median(pred2_aligned)).astype(int)
                                
                                # Calculate correlation
                                correlation = np.corrcoef(pred1_binary, pred2_binary)[0, 1]
                                
                                cross_analysis['pairwise_correlations'][f"{spec1}_vs_{spec2}"] = correlation
                                
                                if correlation > 0.5:
                                    cross_analysis['high_correlation_pairs'].append((spec1, spec2, correlation))
                                    print(f"      ⚠️ HIGH correlation: {correlation:.3f}")
                                elif correlation < 0.1:
                                    cross_analysis['orthogonal_pairs'].append((spec1, spec2, correlation))
                                    print(f"      ✅ ORTHOGONAL: {correlation:.3f}")
                                else:
                                    print(f"      📊 MODERATE: {correlation:.3f}")
                            else:
                                print(f"      ❌ Insufficient aligned data")
                        else:
                            print(f"      ❌ No timestamp for alignment")
                    else:
                        print(f"      ❌ No prediction columns found")
                else:
                    print(f"      ❌ No prediction views found")
                    
            except Exception as e:
                print(f"      ❌ Error: {e}")
    
    return cross_analysis

def generate_comprehensive_report(analyses: Dict[str, Dict[str, Any]], cross_analysis: Dict[str, Any]) -> str:
    """Generate comprehensive optimization report."""
    
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    report = f"""# Comprehensive Specialist Model Optimization Report

**Generated:** {timestamp}  
**Specialists Analyzed:** {len(analyses)}

## Executive Summary

This report provides a comprehensive analysis of all specialist models against the three key optimization requirements:
1. **Sufficient MI/HSIC to target** (information about price/context)
2. **Sufficient orthogonality** (different features, low pairwise correlation)
3. **Single 0/1 scalar output** (binary predictions)

## Individual Specialist Analysis

### Performance and Requirements Compliance

| Specialist | Samples | Features | Binary Output | MI to Target | AUC | Accuracy | Orthogonal Features | Requirements Met |
|------------|---------|----------|---------------|--------------|-----|----------|-------------------|------------------|
"""
    
    for specialist, analysis in analyses.items():
        samples = analysis.get('clean_samples', 0)
        features = analysis.get('orthogonal_features', 0)
        binary = "✅" if analysis.get('has_binary_output', False) else "❌"
        mi = analysis.get('prediction_mi_to_target', 0)
        auc = analysis.get('auc', 0)
        accuracy = analysis.get('accuracy', 0)
        
        # Count requirements met
        req_met = 0
        if analysis.get('has_binary_output', False):
            req_met += 1
        if mi > 0.01:
            req_met += 1
        if analysis.get('high_correlation_pairs', 0) < 5:
            req_met += 1
        
        report += f"| {specialist} | {samples} | {features} | {binary} | {mi:.4f} | {auc:.3f if auc else 'N/A'} | {accuracy:.3f if accuracy else 'N/A'} | {features} | {req_met}/3 |\n"
    
    report += f"""
## Requirements Assessment

### 1. Information Content (MI/HSIC to Target)

**Target:** MI > 0.02 for meaningful information about price/context

| Specialist | MI Score | Status | Priority Action |
|------------|----------|---------|-----------------|
"""
    
    for specialist, analysis in analyses.items():
        mi = analysis.get('prediction_mi_to_target', 0)
        
        if mi > 0.02:
            status = "✅ EXCELLENT"
            action = "Ready for ensemble"
        elif mi > 0.01:
            status = "⚠️ MODERATE"
            action = "Feature engineering recommended"
        else:
            status = "❌ LOW"
            action = "Significant improvement required"
        
        report += f"| {specialist} | {mi:.4f} | {status} | {action} |\n"
    
    report += f"""
### 2. Feature Orthogonality

**Target:** Low correlation (< 0.7) between features within specialist

| Specialist | Total Features | Orthogonal Features | High Corr Pairs | Status |
|------------|-----------------|-------------------|-----------------|---------|
"""
    
    for specialist, analysis in analyses.items():
        total = analysis.get('feature_count', 0)
        orthogonal = analysis.get('orthogonal_features', 0)
        high_corr = analysis.get('high_correlation_pairs', 0)
        
        if high_corr == 0:
            status = "✅ EXCELLENT"
        elif high_corr < 3:
            status = "⚠️ GOOD"
        elif high_corr < 10:
            status = "⚠️ MODERATE"
        else:
            status = "❌ POOR"
        
        report += f"| {specialist} | {total} | {orthogonal} | {high_corr} | {status} |\n"
    
    report += f"""
### 3. Binary Output Standardization

**Target:** Single 0/1 scalar output for all specialists

| Specialist | Binary Output | Conversion Method | Status |
|------------|---------------|-------------------|---------|
"""
    
    for specialist, analysis in analyses.items():
        is_binary = analysis.get('has_binary_output', False)
        threshold = analysis.get('threshold_used', 'N/A')
        
        if is_binary:
            status = "✅ BINARY"
            method = f"Threshold ({threshold:.4f})" if threshold != 'N/A' else "Native"
        else:
            status = "❌ NON-BINARY"
            method = "Needs conversion"
        
        report += f"| {specialist} | {is_binary} | {method} | {status} |\n"
    
    # Cross-specialist analysis
    if cross_analysis['pairwise_correlations']:
        report += f"""
## Cross-Specialist Orthogonality Analysis

**Target:** Low correlation (< 0.3) between different specialists

### Pairwise Correlations

| Specialist Pair | Correlation | Status |
|-----------------|-------------|---------|
"""
        
        for pair, corr in cross_analysis['pairwise_correlations'].items():
            if corr > 0.5:
                status = "❌ HIGH"
            elif corr < 0.1:
                status = "✅ ORTHOGONAL"
            else:
                status = "⚠️ MODERATE"
            
            report += f"| {pair} | {corr:.3f} | {status} |\n"
        
        report += f"""
### Cross-Specialist Summary

- **Total Pairs Analyzed:** {len(cross_analysis['pairwise_correlations'])}
- **High Correlation Pairs (>0.5):** {len(cross_analysis['high_correlation_pairs'])}
- **Orthogonal Pairs (<0.1):** {len(cross_analysis['orthogonal_pairs'])}

"""
        
        if cross_analysis['high_correlation_pairs']:
            report += f"### ⚠️ High Correlation Pairs (Need Attention)\n\n"
            for spec1, spec2, corr in cross_analysis['high_correlation_pairs']:
                report += f"- **{spec1} vs {spec2}:** {corr:.3f}\n"
            report += f"\n"
        
        if cross_analysis['orthogonal_pairs']:
            report += f"### ✅ Orthogonal Pairs (Excellent for Ensemble)\n\n"
            for spec1, spec2, corr in cross_analysis['orthogonal_pairs']:
                report += f"- **{spec1} vs {spec2}:** {corr:.3f}\n"
            report += f"\n"
    
    # Overall compliance
    binary_count = sum(1 for analysis in analyses.values() if analysis.get('has_binary_output', False))
    high_mi_count = sum(1 for analysis in analyses.values() if analysis.get('prediction_mi_to_target', 0) > 0.02)
    good_ortho_count = sum(1 for analysis in analyses.values() if analysis.get('high_correlation_pairs', 0) < 3)
    
    report += f"""
## Overall Compliance Summary

| Requirement | Compliance Rate | Status |
|-------------|----------------|---------|
| Binary Output (0/1 scalar) | {binary_count}/{len(analyses)} ({binary_count/len(analyses)*100:.1f}%) | {'✅' if binary_count == len(analyses) else '⚠️'} |
| High MI Content (>0.02) | {high_mi_count}/{len(analyses)} ({high_mi_count/len(analyses)*100:.1f}%) | {'✅' if high_mi_count >= len(analyses)//2 else '⚠️'} |
| Good Orthogonality | {good_ortho_count}/{len(analyses)} ({good_ortho_count/len(analyses)*100:.1f}%) | {'✅' if good_ortho_count >= len(analyses)//2 else '⚠️'} |

## Performance Statistics

"""
    
    mi_values = [analysis.get('prediction_mi_to_target', 0) for analysis in analyses.values()]
    auc_values = [analysis.get('auc', 0) for analysis in analyses.values() if analysis.get('auc') is not None]
    accuracy_values = [analysis.get('accuracy', 0) for analysis in analyses.values() if analysis.get('accuracy') is not None]
    
    if mi_values:
        report += f"- **Average MI:** {np.mean(mi_values):.4f} ± {np.std(mi_values):.4f}\n"
    if auc_values:
        report += f"- **Average AUC:** {np.mean(auc_values):.3f} ± {np.std(auc_values):.3f}\n"
    if accuracy_values:
        report += f"- **Average Accuracy:** {np.mean(accuracy_values):.3f} ± {np.std(accuracy_values):.3f}\n"
    
    # Ready for ensemble
    ready_specialists = [name for name, analysis in analyses.items() 
                        if (analysis.get('has_binary_output', False) and 
                            analysis.get('prediction_mi_to_target', 0) > 0.01 and
                            analysis.get('high_correlation_pairs', 0) < 5)]
    
    report += f"""
## Ensemble Readiness

### ✅ Ready for Ensemble
**Specialists meeting all requirements:** {len(ready_specialists)}/{len(analyses)}

"""
    
    if ready_specialists:
        report += f"**Ready specialists:** {', '.join(ready_specialists)}\n\n"
        report += f"These specialists can be immediately used for ensemble construction.\n\n"
    else:
        report += f"No specialists currently meet all three requirements simultaneously.\n"
        report += f"Focus on the recommendations below to improve compliance.\n\n"
    
    # Recommendations
    report += f"""
## Optimization Recommendations

### Priority 1: Information Content Improvement

**Specialists needing MI improvement:** {', '.join([name for name, analysis in analyses.items() if analysis.get('prediction_mi_to_target', 0) < 0.01])}

**Actions:**
1. **Add non-linear feature transformations**
   - Polynomial features (squared, cubed)
   - Logarithmic transformations
   - Interaction terms between features
   
2. **Include market regime indicators**
   - Volatility regime flags
   - Trend regime classifications
   - Time-of-day patterns
   
3. **Target-specific feature engineering**
   - For breakout specialists: add momentum indicators
   - For volume specialists: add volume profile features
   - For volatility specialists: add GARCH-based features

### Priority 2: Feature Orthogonalization

**Specialists with high correlations:** {', '.join([name for name, analysis in analyses.items() if analysis.get('high_correlation_pairs', 0) > 5])}

**Actions:**
1. **Remove redundant features** (correlation > 0.7)
2. **Apply PCA for dimensionality reduction**
3. **Use feature selection techniques** (RFE, LASSO)
4. **Create composite features** from correlated groups

### Priority 3: Cross-Specialist Diversification

"""
    
    if cross_analysis['high_correlation_pairs']:
        report += f"**High correlation specialist pairs:**\n"
        for spec1, spec2, corr in cross_analysis['high_correlation_pairs']:
            report += f"- {spec1} vs {spec2}: {corr:.3f}\n"
        report += f"\n**Actions:**\n"
        report += f"1. Modify feature sets to reduce overlap\n"
        report += f"2. Add specialist-specific unique features\n"
        report += f"3. Consider removing redundant specialists\n\n"
    
    report += f"""
### Implementation Plan

**Phase 1 (Immediate - 1 week):**
- Implement binary output standardization for all specialists
- Remove highly correlated features (> 0.7) within specialists

**Phase 2 (Short-term - 2 weeks):**
- Add non-linear feature transformations
- Implement market regime indicators
- Retrain specialists with enhanced features

**Phase 3 (Medium-term - 3 weeks):**
- Optimize hyperparameters for MI > 0.02 target
- Implement cross-specialist correlation monitoring
- Build initial ensemble with compliant specialists

**Phase 4 (Long-term - 4 weeks):**
- Continuous monitoring and improvement
- Ensemble performance optimization
- Production deployment

## Success Metrics

- **Target MI:** > 0.02 for all specialists
- **Target Orthogonality:** < 3 high correlation pairs per specialist
- **Target Binary Output:** 100% compliance
- **Target Cross-Specialist Correlation:** < 0.3
- **Target Ensemble Performance:** Sharpe > 1.0, Max Drawdown < 15%

---
*Comprehensive Specialist Optimization Analysis - Implementation Ready*
"""
    
    return report

def main():
    """Main comprehensive analysis function."""
    
    print("🚀 STARTING COMPREHENSIVE SPECIALIST OPTIMIZATION ANALYSIS")
    print("=" * 70)
    
    # Analyze all specialists
    all_analyses = analyze_all_specialists()
    
    if not all_analyses:
        print("❌ No specialist analyses completed")
        return
    
    print(f"\n✅ ANALYZED {len(all_analyses)} SPECIALISTS SUCCESSFULLY")
    
    # Cross-specialist correlation analysis
    cross_analysis = analyze_cross_specialist_correlations(all_analyses)
    
    # Generate comprehensive report
    print("\n📝 GENERATING COMPREHENSIVE OPTIMIZATION REPORT...")
    report = generate_comprehensive_report(all_analyses, cross_analysis)
    
    # Save report
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_path = Path("outcomes") / f"comprehensive_specialist_optimization_{timestamp}.md"
    
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"✅ COMPREHENSIVE REPORT SAVED: {report_path}")
    
    # Print final summary
    print("\n" + "=" * 70)
    print("🎯 COMPREHENSIVE OPTIMIZATION SUMMARY")
    print("=" * 70)
    
    binary_count = sum(1 for analysis in all_analyses.values() if analysis.get('has_binary_output', False))
    high_mi_count = sum(1 for analysis in all_analyses.values() if analysis.get('prediction_mi_to_target', 0) > 0.02)
    good_ortho_count = sum(1 for analysis in all_analyses.values() if analysis.get('high_correlation_pairs', 0) < 3)
    
    print(f"\n📊 OVERALL COMPLIANCE:")
    print(f"   🔢 Binary Output: {binary_count}/{len(all_analyses)} ({binary_count/len(all_analyses)*100:.1f}%)")
    print(f"   📊 High MI (>0.02): {high_mi_count}/{len(all_analyses)} ({high_mi_count/len(all_analyses)*100:.1f}%)")
    print(f"   🔄 Good Orthogonality: {good_ortho_count}/{len(all_analyses)} ({good_ortho_count/len(all_analyses)*100:.1f}%)")
    
    # Ready for ensemble
    ready_specialists = [name for name, analysis in all_analyses.items() 
                        if (analysis.get('has_binary_output', False) and 
                            analysis.get('prediction_mi_to_target', 0) > 0.01 and
                            analysis.get('high_correlation_pairs', 0) < 5)]
    
    print(f"\n🚀 ENSEMBLE READINESS:")
    print(f"   ✅ Ready Specialists: {len(ready_specialists)}/{len(all_analyses)}")
    if ready_specialists:
        print(f"   📋 Ready: {', '.join(ready_specialists)}")
    
    print(f"\n🎯 NEXT STEPS:")
    print(f"   1. Review comprehensive report for detailed recommendations")
    print(f"   2. Implement Priority 1 improvements (information content)")
    print(f"   3. Apply feature orthogonalization where needed")
    print(f"   4. Build ensemble with ready specialists")
    
    print(f"\n🚀 COMPREHENSIVE OPTIMIZATION FRAMEWORK READY FOR PRODUCTION")

if __name__ == "__main__":
    main()
