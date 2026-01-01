#!/usr/bin/env python3
"""Analyze existing specialist artifacts for optimization requirements."""

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
from scipy.spatial.distance import pdist, squareform

def compute_hsic(X: np.ndarray, Y: np.ndarray, sigma: float = 1.0) -> float:
    """Compute Hilbert-Schmidt Independence Criterion (HSIC)."""
    X = X.reshape(-1, 1) if X.ndim == 1 else X
    Y = Y.reshape(-1, 1) if Y.ndim == 1 else Y
    
    n = X.shape[0]
    
    def rbf_kernel(X, Y=None, sigma=sigma):
        if Y is None:
            Y = X
        pairwise_dists = pdist(X, 'sqeuclidean')
        K = np.exp(-pairwise_dists / (2 * sigma ** 2))
        return squareform(K)
    
    K = rbf_kernel(X)
    L = rbf_kernel(Y)
    
    H = np.eye(n) - np.ones((n, n)) / n
    K_centered = H @ K @ H
    L_centered = H @ L @ H
    
    hsic = np.trace(K_centered @ L_centered) / (n ** 2)
    return hsic

def analyze_specialist_artifacts(specialist_name: str, symbol: str = "ETHUSDT", 
                               exchange: str = "binance", timeframe: str = "15m", 
                               direction: str = "long") -> Dict[str, Any]:
    """Analyze a single specialist's artifacts for optimization metrics."""
    
    print(f"🔍 Analyzing {specialist_name}...")
    
    try:
        # Initialize artifact store
        artifact_store = VersionedArtifactStore("versioned_artifacts")
        
        # Load predictions
        artifact_name = f"{specialist_name}_{timeframe}"
        predictions_data = artifact_store.load_latest(
            artifact_name=artifact_name,
            symbol=symbol,
            exchange=exchange,
            timeframe=timeframe,
            direction=direction
        )
        
        if predictions_data is None or not isinstance(predictions_data, pd.DataFrame):
            print(f"❌ No valid predictions data found for {specialist_name}")
            return {}
        
        print(f"✅ Loaded {len(predictions_data)} rows for {specialist_name}")
        
        # Extract components
        feature_cols = [col for col in predictions_data.columns if col.endswith('_feature')]
        if not feature_cols:
            # Use all columns except predictions and labels
            exclude_cols = [col for col in predictions_data.columns 
                          if 'prediction' in col or 'label' in col or 'probability' in col]
            feature_cols = [col for col in predictions_data.columns if col not in exclude_cols]
        
        features = predictions_data[feature_cols].copy()
        
        # Find labels and predictions
        label_col = None
        pred_col = None
        prob_col = None
        
        for col in predictions_data.columns:
            if 'label' in col:
                label_col = col
            elif 'prediction' in col:
                pred_col = col
            elif 'probability' in col:
                prob_col = col
        
        if label_col is None:
            print(f"❌ No label column found for {specialist_name}")
            return {}
        
        labels = predictions_data[label_col]
        
        # Handle predictions
        if pred_col is not None:
            predictions = predictions_data[pred_col]
        elif prob_col is not None:
            # Convert probabilities to binary predictions
            probabilities = predictions_data[prob_col]
            predictions = (probabilities >= 0.5).astype(int)
        else:
            print(f"❌ No prediction or probability column found for {specialist_name}")
            return {}
        
        # Clean data
        valid_mask = ~(features.isna().any(axis=1) | labels.isna() | pd.isna(predictions))
        features_clean = features[valid_mask]
        labels_clean = labels[valid_mask]
        predictions_clean = predictions[valid_mask]
        
        if len(features_clean) < 100:
            print(f"⚠️ Insufficient clean data for {specialist_name}: {len(features_clean)}")
            return {}
        
        print(f"📊 Clean data: {len(features_clean)} samples, {len(features_clean.columns)} features")
        
        analysis = {
            'specialist_name': specialist_name,
            'total_samples': len(predictions_data),
            'clean_samples': len(features_clean),
            'feature_count': len(features_clean.columns),
            'label_column': label_col,
            'prediction_column': pred_col,
            'probability_column': prob_col
        }
        
        # 1. MI/HSIC to target analysis
        print(f"🔬 Computing MI/HSIC for {specialist_name}...")
        
        try:
            # Prediction MI to target
            pred_mi = mutual_info_regression(
                predictions_clean.values.reshape(-1, 1), 
                labels_clean.values
            )[0]
            analysis['prediction_mi_to_target'] = pred_mi
            
            # Prediction HSIC to target
            pred_hsic = compute_hsic(
                predictions_clean.values, 
                labels_clean.values,
                sigma=np.std(predictions_clean.values)
            )
            analysis['prediction_hsic_to_target'] = pred_hsic
            
            print(f"   MI: {pred_mi:.4f}, HSIC: {pred_hsic:.4f}")
            
        except Exception as e:
            print(f"⚠️ MI/HSIC computation failed: {e}")
            analysis['prediction_mi_to_target'] = 0
            analysis['prediction_hsic_to_target'] = 0
        
        # Feature MI/HSIC
        feature_mi_scores = {}
        feature_hsic_scores = {}
        
        for col in features_clean.columns:
            try:
                # MI
                mi = mutual_info_regression(
                    features_clean[[col]], 
                    labels_clean
                )[0]
                feature_mi_scores[col] = mi
                
                # HSIC
                hsic = compute_hsic(
                    features_clean[col].values, 
                    labels_clean.values,
                    sigma=np.std(features_clean[col].values)
                )
                feature_hsic_scores[col] = hsic
                
            except:
                feature_mi_scores[col] = 0
                feature_hsic_scores[col] = 0
        
        analysis['feature_mi_scores'] = feature_mi_scores
        analysis['feature_hsic_scores'] = feature_hsic_scores
        analysis['avg_feature_mi'] = np.mean(list(feature_mi_scores.values()))
        analysis['avg_feature_hsic'] = np.mean(list(feature_hsic_scores.values()))
        
        # 2. Feature orthogonality analysis
        print(f"🔄 Computing feature orthogonality for {specialist_name}...")
        
        if len(features_clean.columns) > 1:
            corr_matrix = features_clean.corr().abs()
            upper_tri = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )
            
            high_corr_pairs = []
            for col1, col2 in upper_tri.stack().index:
                corr_val = upper_tri.loc[col1, col2]
                if not pd.isna(corr_val) and corr_val > 0.7:
                    high_corr_pairs.append((col1, col2, corr_val))
            
            analysis['high_correlation_pairs'] = high_corr_pairs
            analysis['orthogonal_feature_count'] = len(features_clean.columns) - len(high_corr_pairs)
            analysis['dropped_for_orthogonality'] = len(high_corr_pairs)
            
            print(f"   Found {len(high_corr_pairs)} highly correlated pairs")
        else:
            analysis['high_correlation_pairs'] = []
            analysis['orthogonal_feature_count'] = len(features_clean.columns)
            analysis['dropped_for_orthogonality'] = 0
        
        # 3. Binary output verification
        print(f"🔢 Verifying binary output for {specialist_name}...")
        
        unique_preds = np.unique(predictions_clean)
        is_binary = len(unique_preds) == 2 and set(unique_preds) == {0, 1}
        analysis['has_binary_output'] = is_binary
        analysis['unique_prediction_values'] = list(unique_preds)
        
        if is_binary:
            print(f"   ✅ Binary output confirmed: {unique_preds}")
        else:
            print(f"   ⚠️ Non-binary output: {unique_preds}")
        
        # 4. Basic performance metrics
        try:
            from sklearn.metrics import accuracy_score, roc_auc_score
            
            if prob_col is not None:
                probabilities = predictions_data.loc[valid_mask, prob_col]
                auc = roc_auc_score(labels_clean, probabilities)
                analysis['auc'] = auc
            else:
                analysis['auc'] = None
            
            accuracy = accuracy_score(labels_clean, predictions_clean)
            analysis['accuracy'] = accuracy
            
            print(f"   Accuracy: {accuracy:.3f}, AUC: {auc:.3f if auc else 'N/A'}")
            
        except Exception as e:
            print(f"⚠️ Performance metrics failed: {e}")
            analysis['accuracy'] = None
            analysis['auc'] = None
        
        return analysis
        
    except Exception as e:
        print(f"❌ Analysis failed for {specialist_name}: {e}")
        return {}

def analyze_cross_specialist_orthogonality(all_analyses: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze orthogonality between specialists."""
    
    print("🔍 Analyzing cross-specialist orthogonality...")
    
    specialist_names = list(all_analyses.keys())
    if len(specialist_names) < 2:
        return {'orthogonality_matrix': {}, 'recommendations': ['Insufficient specialists for analysis']}
    
    # Load predictions for correlation analysis
    artifact_store = VersionedArtifactStore("versioned_artifacts")
    all_predictions = {}
    
    for specialist_name in specialist_names:
        try:
            artifact_name = f"{specialist_name}_15m"
            predictions_data = artifact_store.load_latest(
                artifact_name=artifact_name,
                symbol="ETHUSDT",
                exchange="binance", 
                timeframe="15m",
                direction="long"
            )
            
            if predictions_data is not None:
                # Find prediction column
                pred_col = None
                for col in predictions_data.columns:
                    if 'prediction' in col:
                        pred_col = col
                        break
                
                if pred_col is not None:
                    all_predictions[specialist_name] = predictions_data[pred_col]
                    
        except Exception as e:
            print(f"⚠️ Could not load predictions for {specialist_name}: {e}")
    
    # Compute correlations
    orthogonality_matrix = {}
    
    for i, name1 in enumerate(specialist_names):
        for j, name2 in enumerate(specialist_names):
            if i < j and name1 in all_predictions and name2 in all_predictions:
                try:
                    # Align predictions
                    common_idx = all_predictions[name1].index.intersection(all_predictions[name2].index)
                    if len(common_idx) > 100:
                        pred1 = all_predictions[name1].loc[common_idx]
                        pred2 = all_predictions[name2].loc[common_idx]
                        
                        correlation, p_value = spearmanr(pred1, pred2)
                        orthogonality_matrix[f"{name1}_vs_{name2}"] = {
                            'correlation': correlation,
                            'p_value': p_value,
                            'samples': len(common_idx)
                        }
                        
                except Exception as e:
                    print(f"⚠️ Correlation failed for {name1} vs {name2}: {e}")
    
    # Generate recommendations
    recommendations = []
    
    high_corr_pairs = [
        pair for pair, stats in orthogonality_matrix.items()
        if abs(stats['correlation']) > 0.7
    ]
    
    if high_corr_pairs:
        recommendations.append(f"⚠️ High correlation detected: {', '.join(high_corr_pairs)}")
    
    good_orthogonality_pairs = [
        pair for pair, stats in orthogonality_matrix.items()
        if abs(stats['correlation']) < 0.3
    ]
    
    if good_orthogonality_pairs:
        recommendations.append(f"✅ Good orthogonality: {', '.join(good_orthogonality_pairs)}")
    
    return {
        'orthogonality_matrix': orthogonality_matrix,
        'recommendations': recommendations,
        'specialist_count': len(specialist_names)
    }

def generate_optimization_report(all_analyses: Dict[str, Dict[str, Any]], 
                              orthogonality_analysis: Dict[str, Any]) -> str:
    """Generate comprehensive optimization report."""
    
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    report = f"""# Specialist Model Optimization Analysis

**Generated:** {timestamp}  
**Specialists Analyzed:** {len(all_analyses)}

## Executive Summary

This report analyzes specialist models against the three key requirements:
1. **Sufficient MI/HSIC to target** (information content)
2. **Sufficient orthogonality** (low pairwise correlation)  
3. **Single 0/1 scalar output** (binary predictions)

## Individual Specialist Analysis

### MI/HSIC and Performance Metrics

| Specialist | Samples | Features | Binary Output | MI to Target | HSIC to Target | AUC | Accuracy | Orthogonal Features |
|------------|---------|----------|---------------|--------------|----------------|-----|----------|-------------------|
"""
    
    # Add individual specialist analysis
    for specialist_name, analysis in all_analyses.items():
        samples = analysis.get('clean_samples', 0)
        features = analysis.get('orthogonal_feature_count', 0)
        binary = "✅" if analysis.get('has_binary_output', False) else "❌"
        mi = analysis.get('prediction_mi_to_target', 0)
        hsic = analysis.get('prediction_hsic_to_target', 0)
        auc = analysis.get('auc', 0)
        accuracy = analysis.get('accuracy', 0)
        
        report += f"| {specialist_name} | {samples} | {features} | {binary} | {mi:.4f} | {hsic:.4f} | {auc:.3f if auc else 'N/A'} | {accuracy:.3f if accuracy else 'N/A'} | {features} |\n"
    
    report += f"""
## Requirement 1: MI/HSIC to Target Analysis

### Information Content Assessment

**MI Threshold:** > 0.02 (meaningful information)  
**HSIC Threshold:** > 0.02 (non-linear dependence)

| Specialist | MI Score | Status | HSIC Score | Status |
|------------|----------|---------|------------|---------|
"""
    
    # Add MI/HSIC status
    for specialist_name, analysis in all_analyses.items():
        mi = analysis.get('prediction_mi_to_target', 0)
        hsic = analysis.get('prediction_hsic_to_target', 0)
        
        mi_status = "✅ Good" if mi > 0.02 else "⚠️ Low" if mi > 0.005 else "❌ Poor"
        hsic_status = "✅ Good" if hsic > 0.02 else "⚠️ Low" if hsic > 0.005 else "❌ Poor"
        
        report += f"| {specialist_name} | {mi:.4f} | {mi_status} | {hsic:.4f} | {hsic_status} |\n"
    
    # Summary statistics
    mi_values = [analysis.get('prediction_mi_to_target', 0) for analysis in all_analyses.values()]
    hsic_values = [analysis.get('prediction_hsic_to_target', 0) for analysis in all_analyses.values()]
    
    report += f"""
**Summary:**
- Average MI across specialists: {np.mean(mi_values):.4f} ± {np.std(mi_values):.4f}
- Average HSIC across specialists: {np.mean(hsic_values):.4f} ± {np.std(hsic_values):.4f}
- High MI specialists (>0.02): {len([v for v in mi_values if v > 0.02])}
- High HSIC specialists (>0.02): {len([v for v in hsic_values if v > 0.02])}

## Requirement 2: Orthogonality Analysis

### Cross-Specialist Correlations

| Specialist Pair | Correlation | P-Value | Samples | Status |
|------------------|-------------|---------|---------|---------|
"""
    
    # Add orthogonality matrix
    for pair, stats in orthogonality_analysis.get('orthogonality_matrix', {}).items():
        corr = stats['correlation']
        p_val = stats['p_value']
        samples = stats['samples']
        
        status = "✅ Orthogonal" if abs(corr) < 0.3 else "⚠️ Moderate" if abs(corr) < 0.7 else "❌ Highly Correlated"
        
        report += f"| {pair} | {corr:.3f} | {p_val:.4f} | {samples} | {status} |\n"
    
    report += f"""
### Feature Orthogonality Within Specialists

| Specialist | Original Features | Orthogonal Features | Dropped | Status |
|------------|-------------------|-------------------|---------|---------|
"""
    
    # Add feature orthogonality
    for specialist_name, analysis in all_analyses.items():
        original = analysis.get('feature_count', 0)
        orthogonal = analysis.get('orthogonal_feature_count', 0)
        dropped = analysis.get('dropped_for_orthogonality', 0)
        
        status = "✅ Good" if dropped < original * 0.2 else "⚠️ Many Dropped" if dropped < original * 0.5 else "❌ High Overlap"
        
        report += f"| {specialist_name} | {original} | {orthogonal} | {dropped} | {status} |\n"
    
    report += f"""
## Requirement 3: Binary Output Verification

| Specialist | Binary Output | Unique Values | Status |
|------------|---------------|---------------|---------|
"""
    
    # Add binary output verification
    for specialist_name, analysis in all_analyses.items():
        is_binary = analysis.get('has_binary_output', False)
        unique_vals = analysis.get('unique_prediction_values', [])
        
        status = "✅ Binary" if is_binary else "❌ Non-Binary"
        unique_str = str(unique_vals) if len(unique_vals) <= 5 else f"{len(unique_vals)} unique values"
        
        report += f"| {specialist_name} | {is_binary} | {unique_str} | {status} |\n"
    
    report += f"""
## Optimization Recommendations

### Priority Actions

"""
    
    # Add recommendations from orthogonality analysis
    for rec in orthogonality_analysis.get('recommendations', []):
        report += f"- {rec}\n"
    
    # MI/HSIC recommendations
    low_mi_specialists = [name for name, analysis in all_analyses.items() 
                         if analysis.get('prediction_mi_to_target', 0) < 0.01]
    
    if low_mi_specialists:
        report += f"- ⚠️ Low MI specialists: {', '.join(low_mi_specialists)} - consider feature engineering\n"
    
    low_hsic_specialists = [name for name, analysis in all_analyses.items() 
                           if analysis.get('prediction_hsic_to_target', 0) < 0.01]
    
    if low_hsic_specialists:
        report += f"- ⚠️ Low HSIC specialists: {', '.join(low_hsic_specialists)} - consider non-linear transformations\n"
    
    # Binary output recommendations
    non_binary_specialists = [name for name, analysis in all_analyses.items() 
                             if not analysis.get('has_binary_output', False)]
    
    if non_binary_specialists:
        report += f"- ❌ Non-binary output specialists: {', '.join(non_binary_specialists)} - need binary conversion\n"
    
    # Feature correlation recommendations
    high_corr_specialists = [name for name, analysis in all_analyses.items() 
                           if analysis.get('dropped_for_orthogonality', 0) > 5]
    
    if high_corr_specialists:
        report += f"- ⚠️ High feature correlation specialists: {', '.join(high_corr_specialists)} - need orthogonalization\n"
    
    report += f"""
### Model Selection Strategy

1. **Primary Specialists** (Meeting all requirements):
   - Binary output ✅
   - MI > 0.02 ✅  
   - HSIC > 0.02 ✅
   - Low cross-correlation ✅

2. **Secondary Specialists** (Partial compliance):
   - Meet 2/3 requirements
   - Can be improved with feature engineering

3. **Ensemble Construction**:
   - Prioritize orthogonal specialists
   - Weight by MI/HSIC scores
   - Ensure binary output standardization

### Next Implementation Steps

1. **Convert non-binary outputs** to 0/1 scalars
2. **Apply feature orthogonality** during training  
3. **Optimize for MI/HSIC** in hyperparameter tuning
4. **Monitor cross-specialist correlations** in real-time

---
*Specialist Optimization Analysis - Generated for Enhanced Model Performance*
"""
    
    return report

def main():
    """Main analysis function."""
    
    print("🚀 Starting Specialist Model Optimization Analysis...")
    
    # Define specialists to analyze
    specialists = [
        'ml_liquidity_regime_step',
        'ml_breakout_bounce_regime_step', 
        'ml_path_regime_step',
        'ml_smc_regime_step',
        'ml_volume_force_step'
    ]
    
    # Analyze each specialist
    all_analyses = {}
    
    for specialist in specialists:
        analysis = analyze_specialist_artifacts(specialist)
        if analysis:
            all_analyses[specialist] = analysis
        print()
    
    if not all_analyses:
        print("❌ No specialist analyses completed")
        return
    
    print(f"✅ Analyzed {len(all_analyses)} specialists")
    
    # Cross-specialist orthogonality
    orthogonality_analysis = analyze_cross_specialist_orthogonality(all_analyses)
    
    # Generate report
    print("📝 Generating optimization report...")
    report = generate_optimization_report(all_analyses, orthogonality_analysis)
    
    # Save report
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_path = Path("outcomes") / f"specialist_optimization_analysis_{timestamp}.md"
    
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"✅ Report saved: {report_path}")
    
    # Print summary
    print("\n🎯 Summary:")
    print(f"   Specialists analyzed: {len(all_analyses)}")
    
    binary_count = sum(1 for analysis in all_analyses.values() if analysis.get('has_binary_output', False))
    print(f"   Binary output: {binary_count}/{len(all_analyses)}")
    
    high_mi_count = sum(1 for analysis in all_analyses.values() if analysis.get('prediction_mi_to_target', 0) > 0.02)
    print(f"   High MI (>0.02): {high_mi_count}/{len(all_analyses)}")
    
    high_hsic_count = sum(1 for analysis in all_analyses.values() if analysis.get('prediction_hsic_to_target', 0) > 0.02)
    print(f"   High HSIC (>0.02): {high_hsic_count}/{len(all_analyses)}")
    
    orthogonality_pairs = len(orthogonality_analysis.get('orthogonality_matrix', {}))
    print(f"   Orthogonality pairs analyzed: {orthogonality_pairs}")

if __name__ == "__main__":
    main()
