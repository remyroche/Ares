#!/usr/bin/env python3
"""Simple analysis of specialist artifacts for optimization requirements."""

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

def analyze_specialist_store(specialist_name: str) -> Dict[str, Any]:
    """Analyze specialist using direct artifact store access."""
    
    print(f"🔍 Analyzing {specialist_name}...")
    
    try:
        # Initialize artifact store for specialist
        store_path = f"versioned_artifacts/ETHUSDT_binance_15m_long_{specialist_name.replace('ml_', '').replace('_step', '')}"
        store = VersionedArtifactStore(store_path)
        
        # Get available views
        try:
            view_names = store.list_versions()
            print(f"   Available views: {view_names}")
        except:
            print(f"   No views found in {store_path}")
            return {}
        
        # Try to get the latest data view
        data_view = None
        for view_name in view_names:
            if '15m' in view_name or 'prediction' in view_name or 'probability' in view_name:
                try:
                    data_view = store.get_view(view_name)
                    if isinstance(data_view, pd.DataFrame) and len(data_view) > 0:
                        print(f"   ✅ Loaded view: {view_name} ({len(data_view)} rows)")
                        break
                except:
                    continue
        
        if data_view is None or not isinstance(data_view, pd.DataFrame):
            print(f"   ❌ No valid data view found")
            return {}
        
        # Analyze the data
        analysis = analyze_dataframe(data_view, specialist_name)
        return analysis
        
    except Exception as e:
        print(f"   ❌ Analysis failed: {e}")
        return {}

def analyze_dataframe(df: pd.DataFrame, specialist_name: str) -> Dict[str, Any]:
    """Analyze a DataFrame for optimization metrics."""
    
    print(f"   📊 Analyzing DataFrame: {df.shape}")
    
    analysis = {
        'specialist_name': specialist_name,
        'total_samples': len(df),
        'columns': list(df.columns)
    }
    
    # Find labels, predictions, probabilities
    label_col = None
    pred_col = None
    prob_col = None
    feature_cols = []
    
    for col in df.columns:
        if 'label' in col.lower():
            label_col = col
        elif 'prediction' in col.lower():
            pred_col = col
        elif 'probability' in col.lower() or 'prob' in col.lower():
            prob_col = col
        elif 'feature' in col.lower() or col in ['close', 'volume', 'high', 'low', 'open']:
            feature_cols.append(col)
    
    analysis['label_column'] = label_col
    analysis['prediction_column'] = pred_col
    analysis['probability_column'] = prob_col
    analysis['feature_columns'] = feature_cols
    
    print(f"   📋 Found: label={label_col}, pred={pred_col}, prob={prob_col}, features={len(feature_cols)}")
    
    # Extract data
    if label_col is None:
        print(f"   ⚠️ No label column found")
        return analysis
    
    labels = df[label_col]
    features = df[feature_cols].copy() if feature_cols else pd.DataFrame()
    
    # Handle predictions
    predictions = None
    if pred_col is not None:
        predictions = df[pred_col]
    elif prob_col is not None:
        probabilities = df[prob_col]
        predictions = (probabilities >= 0.5).astype(int)
        analysis['converted_from_probability'] = True
    
    if predictions is None:
        print(f"   ⚠️ No prediction column found")
        return analysis
    
    analysis['predictions_available'] = True
    
    # Clean data
    valid_mask = ~(features.isna().any(axis=1) if len(features.columns) > 0 else pd.Series([False]*len(df)))
    valid_mask &= ~(labels.isna()) & ~(pd.isna(predictions))
    
    features_clean = features[valid_mask] if len(features.columns) > 0 else pd.DataFrame(index=features.index[valid_mask])
    labels_clean = labels[valid_mask]
    predictions_clean = predictions[valid_mask]
    
    analysis['clean_samples'] = len(features_clean)
    print(f"   ✨ Clean samples: {len(features_clean)}")
    
    if len(features_clean) < 100:
        print(f"   ⚠️ Insufficient clean data")
        return analysis
    
    # 1. Binary output verification
    unique_preds = np.unique(predictions_clean)
    is_binary = len(unique_preds) == 2 and set(unique_preds) == {0, 1}
    analysis['has_binary_output'] = is_binary
    analysis['unique_prediction_values'] = list(unique_preds)
    
    print(f"   🔢 Binary output: {is_binary} (values: {unique_preds})")
    
    # 2. MI to target
    try:
        pred_mi = mutual_info_regression(
            predictions_clean.values.reshape(-1, 1), 
            labels_clean.values
        )[0]
        analysis['prediction_mi_to_target'] = pred_mi
        print(f"   📊 MI to target: {pred_mi:.4f}")
    except Exception as e:
        print(f"   ⚠️ MI computation failed: {e}")
        analysis['prediction_mi_to_target'] = 0
    
    # 3. Feature orthogonality
    if len(features_clean.columns) > 1:
        try:
            corr_matrix = features_clean.corr().abs()
            high_corr = (corr_matrix > 0.7).sum().sum() - len(features_clean.columns)
            analysis['high_correlation_pairs'] = high_corr // 2
            analysis['orthogonal_features'] = len(features_clean.columns) - (high_corr // 2)
            print(f"   🔄 High correlation pairs: {high_corr // 2}")
        except:
            analysis['high_correlation_pairs'] = 0
            analysis['orthogonal_features'] = len(features_clean.columns)
    else:
        analysis['high_correlation_pairs'] = 0
        analysis['orthogonal_features'] = len(features_clean.columns)
    
    # 4. Basic performance
    try:
        from sklearn.metrics import accuracy_score, roc_auc_score
        
        accuracy = accuracy_score(labels_clean, predictions_clean)
        analysis['accuracy'] = accuracy
        
        if prob_col is not None:
            probabilities = df.loc[valid_mask, prob_col]
            auc = roc_auc_score(labels_clean, probabilities)
            analysis['auc'] = auc
        else:
            analysis['auc'] = None
        
        print(f"   📈 Accuracy: {accuracy:.3f}, AUC: {auc:.3f if auc else 'N/A'}")
        
    except Exception as e:
        print(f"   ⚠️ Performance metrics failed: {e}")
        analysis['accuracy'] = None
        analysis['auc'] = None
    
    return analysis

def generate_summary_report(analyses: Dict[str, Dict[str, Any]]) -> str:
    """Generate summary report."""
    
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    report = f"""# Specialist Model Optimization Summary

**Generated:** {timestamp}  
**Specialists Analyzed:** {len(analyses)}

## Requirements Assessment

### 1. MI/HSIC to Target (Information Content)

| Specialist | Binary Output | Clean Samples | MI to Target | Status |
|------------|---------------|---------------|-------------|---------|
"""
    
    for specialist, analysis in analyses.items():
        binary = "✅" if analysis.get('has_binary_output', False) else "❌"
        samples = analysis.get('clean_samples', 0)
        mi = analysis.get('prediction_mi_to_target', 0)
        status = "✅ Good" if mi > 0.02 else "⚠️ Low" if mi > 0.005 else "❌ Poor"
        
        report += f"| {specialist} | {binary} | {samples} | {mi:.4f} | {status} |\n"
    
    report += f"""
### 2. Feature Orthogonality

| Specialist | Total Features | Orthogonal Features | High Corr Pairs | Status |
|------------|-----------------|-------------------|-----------------|---------|
"""
    
    for specialist, analysis in analyses.items():
        total_features = len(analysis.get('feature_columns', []))
        orthogonal = analysis.get('orthogonal_features', 0)
        high_corr = analysis.get('high_correlation_pairs', 0)
        
        status = "✅ Good" if high_corr < 2 else "⚠️ Some" if high_corr < 5 else "❌ Many"
        
        report += f"| {specialist} | {total_features} | {orthogonal} | {high_corr} | {status} |\n"
    
    report += f"""
### 3. Binary Output Verification

| Specialist | Binary Output | Unique Values | Status |
|------------|---------------|---------------|---------|
"""
    
    for specialist, analysis in analyses.items():
        is_binary = analysis.get('has_binary_output', False)
        unique_vals = analysis.get('unique_prediction_values', [])
        
        status = "✅ Binary" if is_binary else "❌ Non-Binary"
        unique_str = str(unique_vals) if len(unique_vals) <= 5 else f"{len(unique_vals)} values"
        
        report += f"| {specialist} | {is_binary} | {unique_str} | {status} |\n"
    
    # Summary statistics
    binary_count = sum(1 for analysis in analyses.values() if analysis.get('has_binary_output', False))
    high_mi_count = sum(1 for analysis in analyses.values() if analysis.get('prediction_mi_to_target', 0) > 0.02)
    good_ortho_count = sum(1 for analysis in analyses.values() if analysis.get('high_correlation_pairs', 0) < 2)
    
    report += f"""
## Summary Statistics

- **Specialists with binary output:** {binary_count}/{len(analyses)} ({binary_count/len(analyses)*100:.1f}%)
- **Specialists with high MI (>0.02):** {high_mi_count}/{len(analyses)} ({high_mi_count/len(analyses)*100:.1f}%)
- **Specialists with good orthogonality:** {good_ortho_count}/{len(analyses)} ({good_ortho_count/len(analyses)*100:.1f}%)

## Recommendations

### Immediate Actions

"""
    
    # Binary output issues
    non_binary = [name for name, analysis in analyses.items() if not analysis.get('has_binary_output', False)]
    if non_binary:
        report += f"- ❌ Convert to binary output: {', '.join(non_binary)}\n"
    
    # Low MI issues
    low_mi = [name for name, analysis in analyses.items() if analysis.get('prediction_mi_to_target', 0) < 0.01]
    if low_mi:
        report += f"- ⚠️ Improve MI through feature engineering: {', '.join(low_mi)}\n"
    
    # High correlation issues
    high_corr = [name for name, analysis in analyses.items() if analysis.get('high_correlation_pairs', 0) > 5]
    if high_corr:
        report += f"- 🔄 Apply feature orthogonalization: {', '.join(high_corr)}\n"
    
    # Success stories
    success = [name for name, analysis in analyses.items() 
              if (analysis.get('has_binary_output', False) and 
                  analysis.get('prediction_mi_to_target', 0) > 0.01 and
                  analysis.get('high_correlation_pairs', 0) < 5)]
    
    if success:
        report += f"- ✅ Ready for ensemble: {', '.join(success)}\n"
    
    report += f"""
### Implementation Priority

1. **Fix Binary Output** - Convert all specialists to 0/1 scalars
2. **Feature Orthogonalization** - Remove correlations > 0.7  
3. **MI Optimization** - Target MI > 0.02 for all specialists
4. **Cross-Specialist Analysis** - Ensure low pairwise correlations

---
*Specialist Optimization Analysis - Focus on Binary Output, MI Content, and Orthogonality*
"""
    
    return report

def main():
    """Main analysis function."""
    
    print("🚀 Starting Simple Specialist Analysis...")
    
    # Define specialist store names
    specialist_stores = {
        'ml_liquidity_regime_step': 'liquidity_regime',
        'ml_breakout_bounce_regime_step': 'breakout_bounce', 
        'ml_path_regime_step': 'path_regime',
        'ml_smc_regime_step': 'smc_regime',
        'ml_volume_force_step': 'volume_force'
    }
    
    # Analyze each specialist
    all_analyses = {}
    
    for specialist_name, store_name in specialist_stores.items():
        analysis = analyze_specialist_store(specialist_name)
        if analysis:
            all_analyses[specialist_name] = analysis
        print()
    
    if not all_analyses:
        print("❌ No specialist analyses completed")
        return
    
    print(f"✅ Analyzed {len(all_analyses)} specialists")
    
    # Generate report
    print("📝 Generating summary report...")
    report = generate_summary_report(all_analyses)
    
    # Save report
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_path = Path("outcomes") / f"specialist_optimization_summary_{timestamp}.md"
    
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"✅ Report saved: {report_path}")
    
    # Print summary
    print("\n🎯 Quick Summary:")
    binary_count = sum(1 for analysis in all_analyses.values() if analysis.get('has_binary_output', False))
    high_mi_count = sum(1 for analysis in all_analyses.values() if analysis.get('prediction_mi_to_target', 0) > 0.02)
    good_ortho_count = sum(1 for analysis in all_analyses.values() if analysis.get('high_correlation_pairs', 0) < 2)
    
    print(f"   Binary output: {binary_count}/{len(all_analyses)} ✅")
    print(f"   High MI (>0.02): {high_mi_count}/{len(all_analyses)} ✅") 
    print(f"   Good orthogonality: {good_ortho_count}/{len(all_analyses)} ✅")

if __name__ == "__main__":
    main()
