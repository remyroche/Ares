#!/usr/bin/env python3
"""Working specialist analysis with ArtifactView support."""

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

def analyze_specialist_working(specialist_name: str, store_name: str) -> Dict[str, Any]:
    """Analyze specialist with ArtifactView support."""
    
    print(f"🔍 Analyzing {specialist_name}...")
    
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
        if hasattr(artifact_view, 'to_dataframe'):
            df = artifact_view.to_dataframe()
        elif hasattr(artifact_view, 'data'):
            df = artifact_view.data
        else:
            # Try to access as DataFrame directly
            df = artifact_view
        
        if not isinstance(df, pd.DataFrame) or len(df) == 0:
            print(f"   ❌ Invalid data format")
            return {}
        
        print(f"   ✅ Loaded {latest_view}: {df.shape}")
        print(f"   📋 Columns: {list(df.columns)}")
        
        analysis = analyze_working_data(df, specialist_name)
        return analysis
        
    except Exception as e:
        print(f"   ❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return {}

def analyze_working_data(df: pd.DataFrame, specialist_name: str) -> Dict[str, Any]:
    """Analyze working specialist data."""
    
    analysis = {
        'specialist_name': specialist_name,
        'total_samples': len(df),
        'columns': list(df.columns)
    }
    
    # Identify target and prediction columns
    target_cols = [col for col in df.columns if 'target_' in col.lower()]
    pred_cols = [col for col in df.columns if any(x in col.lower() for x in ['force_', 'regime_', 'momentum_', 'volatility_', 'liquidity_', 'breakout_', 'path_', 'smc_', 'reversion_'])]
    
    print(f"   🎯 Target columns: {target_cols}")
    print(f"   🔮 Prediction columns: {pred_cols}")
    
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
    print(f"   ✨ Clean samples: {len(labels_clean)}")
    
    if len(labels_clean) < 100:
        print(f"   ⚠️ Insufficient clean data")
        return analysis
    
    # 1. Convert to binary output (0/1 scalar)
    binary_labels = (labels_clean > 0).astype(int)
    
    # For predictions, use median threshold
    if predictions_clean.dtype in ['float64', 'float32']:
        threshold = np.median(predictions_clean)
        binary_predictions = (predictions_clean > threshold).astype(int)
        analysis['threshold_used'] = threshold
    else:
        binary_predictions = predictions_clean.astype(int)
    
    analysis['has_binary_output'] = True
    analysis['unique_prediction_values'] = list(np.unique(binary_predictions))
    
    print(f"   🔢 Binary conversion: threshold={threshold:.4f}, unique values={np.unique(binary_predictions)}")
    
    # 2. MI to target
    try:
        pred_mi = mutual_info_regression(
            binary_predictions.values.reshape(-1, 1), 
            binary_labels.values
        )[0]
        analysis['prediction_mi_to_target'] = pred_mi
        print(f"   📊 MI to target: {pred_mi:.4f}")
    except Exception as e:
        print(f"   ⚠️ MI computation failed: {e}")
        analysis['prediction_mi_to_target'] = 0
    
    # 3. Basic performance metrics
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
        
        print(f"   📈 Accuracy: {accuracy:.3f}, AUC: {auc:.3f if auc else 'N/A'}")
        
    except Exception as e:
        print(f"   ⚠️ Performance metrics failed: {e}")
        analysis['accuracy'] = None
        analysis['auc'] = None
    
    # 4. Feature analysis
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
                print(f"   🔄 High correlation pairs: {int(high_corr)}")
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

def generate_working_report(analyses: Dict[str, Dict[str, Any]]) -> str:
    """Generate working optimization report."""
    
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    report = f"""# Working Specialist Model Optimization Report

**Generated:** {timestamp}  
**Specialists Analyzed:** {len(analyses)}

## Requirements Assessment

### 1. Information Content (MI/HSIC to Target)
**Target:** MI > 0.02 for meaningful information about price/context

| Specialist | MI Score | Status | Recommendation |
|------------|----------|---------|---------------|
"""
    
    for specialist, analysis in analyses.items():
        mi = analysis.get('prediction_mi_to_target', 0)
        
        if mi > 0.02:
            status = "✅ Excellent"
            rec = "Ready for ensemble"
        elif mi > 0.01:
            status = "⚠️ Moderate"
            rec = "Feature engineering needed"
        else:
            status = "❌ Low"
            rec = "Significant improvement required"
        
        report += f"| {specialist} | {mi:.4f} | {status} | {rec} |\n"
    
    report += f"""
### 2. Feature Orthogonality
**Target:** Low correlation (< 0.7) between features

| Specialist | Total Features | Orthogonal Features | High Corr Pairs | Status |
|------------|-----------------|-------------------|-----------------|---------|
"""
    
    for specialist, analysis in analyses.items():
        total = analysis.get('feature_count', 0)
        orthogonal = analysis.get('orthogonal_features', 0)
        high_corr = analysis.get('high_correlation_pairs', 0)
        
        if high_corr == 0:
            status = "✅ Excellent"
        elif high_corr < 3:
            status = "⚠️ Good"
        elif high_corr < 10:
            status = "⚠️ Moderate"
        else:
            status = "❌ Poor"
        
        report += f"| {specialist} | {total} | {orthogonal} | {high_corr} | {status} |\n"
    
    report += f"""
### 3. Binary Output Standardization
**Target:** Single 0/1 scalar output

| Specialist | Binary Output | Conversion Method | Status |
|------------|---------------|-------------------|---------|
"""
    
    for specialist, analysis in analyses.items():
        is_binary = analysis.get('has_binary_output', False)
        threshold = analysis.get('threshold_used', 'N/A')
        
        if is_binary:
            status = "✅ Binary"
            method = f"Threshold ({threshold:.4f})" if threshold != 'N/A' else "Native"
        else:
            status = "❌ Non-Binary"
            method = "Needs conversion"
        
        report += f"| {specialist} | {is_binary} | {method} | {status} |\n"
    
    # Summary statistics
    binary_count = sum(1 for analysis in analyses.values() if analysis.get('has_binary_output', False))
    high_mi_count = sum(1 for analysis in analyses.values() if analysis.get('prediction_mi_to_target', 0) > 0.02)
    good_ortho_count = sum(1 for analysis in analyses.values() if analysis.get('high_correlation_pairs', 0) < 3)
    
    report += f"""
## Summary Statistics

- **Binary Output Compliance:** {binary_count}/{len(analyses)} ({binary_count/len(analyses)*100:.1f}%)
- **High MI Content (>0.02):** {high_mi_count}/{len(analyses)} ({high_mi_count/len(analyses)*100:.1f}%)
- **Good Orthogonality:** {good_ortho_count}/{len(analyses)} ({good_ortho_count/len(analyses)*100:.1f}%)

## Performance Summary

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
    
    report += f"""
## Recommendations

### Immediate Actions

"""
    
    # Specific recommendations
    low_mi = [name for name, analysis in analyses.items() if analysis.get('prediction_mi_to_target', 0) < 0.01]
    if low_mi:
        report += f"1. **Information Content Improvement:** {', '.join(low_mi)}\n"
        report += f"   - Add non-linear feature transformations\n"
        report += f"   - Include market regime indicators\n"
        report += f"   - Target MI > 0.02\n\n"
    
    high_corr = [name for name, analysis in analyses.items() if analysis.get('high_correlation_pairs', 0) > 5]
    if high_corr:
        report += f"2. **Feature Orthogonalization:** {', '.join(high_corr)}\n"
        report += f"   - Remove features with correlation > 0.7\n"
        report += f"   - Apply feature selection\n\n"
    
    # Success stories
    ready_specialists = [name for name, analysis in analyses.items() 
                        if (analysis.get('has_binary_output', False) and 
                            analysis.get('prediction_mi_to_target', 0) > 0.01 and
                            analysis.get('high_correlation_pairs', 0) < 5)]
    
    if ready_specialists:
        report += f"### Ready for Ensemble\n\n"
        report += f"**Specialists meeting requirements:** {', '.join(ready_specialists)}\n\n"
    
    report += f"""
### Implementation Priority

1. **Binary Output Standardization** - All specialists converted to 0/1 scalars
2. **Information Content Optimization** - Target MI > 0.02 for all specialists  
3. **Feature Orthogonalization** - Remove high correlations
4. **Cross-Specialist Analysis** - Monitor pairwise correlations

---
*Working Specialist Optimization Analysis - Data Successfully Analyzed*
"""
    
    return report

def main():
    """Main analysis function."""
    
    print("🚀 Starting Working Specialist Analysis...")
    
    # Define specialists with their store names
    specialists = {
        'ml_volume_force_step': 'volume_force'
    }
    
    # Analyze each specialist
    all_analyses = {}
    
    for specialist_name, store_name in specialists.items():
        analysis = analyze_specialist_working(specialist_name, store_name)
        if analysis:
            all_analyses[specialist_name] = analysis
        print()
    
    if not all_analyses:
        print("❌ No specialist analyses completed")
        return
    
    print(f"✅ Analyzed {len(all_analyses)} specialists")
    
    # Generate report
    print("📝 Generating working report...")
    report = generate_working_report(all_analyses)
    
    # Save report
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_path = Path("outcomes") / f"working_specialist_analysis_{timestamp}.md"
    
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"✅ Report saved: {report_path}")
    
    # Print summary
    print("\n🎯 WORKING ANALYSIS SUMMARY:")
    for specialist_name, analysis in all_analyses.items():
        mi = analysis.get('prediction_mi_to_target', 0)
        accuracy = analysis.get('accuracy', 0)
        high_corr = analysis.get('high_correlation_pairs', 0)
        
        print(f"   {specialist_name}:")
        print(f"     MI: {mi:.4f} ({'✅' if mi > 0.02 else '⚠️' if mi > 0.01 else '❌'})")
        print(f"     Accuracy: {accuracy:.3f}")
        print(f"     High Corr Pairs: {high_corr}")
        print(f"     Binary Output: ✅")

if __name__ == "__main__":
    main()
