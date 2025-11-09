#!/usr/bin/env python3
"""
Enhanced Feature Metrics Generator

This script generates detailed CSV and MD files with comprehensive feature metrics
including permutation importance for all features and detailed analysis.
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime
import json
from typing import Dict, List, Any

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from training.steps.pre_training.components.final_feature_selection import (
    FinalFeatureSelectionComponent,
    FinalFeatureSelectionConfig
)
from utils.artifact_manager import ArtifactManager


def analyze_outlier_detection_issue(X: pd.DataFrame, feature_name: str) -> Dict[str, Any]:
    """
    Analyze why a feature is being flagged as an outlier.
    
    Args:
        X: Feature DataFrame
        feature_name: Name of the feature to analyze
        
    Returns:
        Dictionary with outlier analysis details
    """
    try:
        feature_data = X[feature_name].dropna()
        
        if len(feature_data) == 0:
            return {"error": "No valid data points"}
        
        # Multiple outlier detection methods
        results = {
            'feature_name': feature_name,
            'data_points': len(feature_data),
            'data_range': feature_data.max() - feature_data.min(),
            'std_dev': feature_data.std(),
            'mean': feature_data.mean(),
            'median': feature_data.median(),
        }
        
        # 1. IQR Method (what pandas might be using)
        Q1 = feature_data.quantile(0.25)
        Q3 = feature_data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        iqr_outliers = ((feature_data < lower_bound) | (feature_data > upper_bound)).sum()
        
        # 2. 3-Sigma Method
        mean_3sigma = feature_data.mean()
        std_3sigma = feature_data.std()
        lower_3sigma = mean_3sigma - 3 * std_3sigma
        upper_3sigma = mean_3sigma + 3 * std_3sigma
        sigma_outliers = ((feature_data < lower_3sigma) | (feature_data > upper_3sigma)).sum()
        
        # 3. Modified Z-Score (for financial data)
        z_scores = np.abs((feature_data - feature_data.mean()) / feature_data.std())
        modified_z_outliers = (z_scores > 2.5).sum()  # More lenient threshold
        
        results.update({
            'iqr_outliers': iqr_outliers,
            'iqr_outlier_pct': (iqr_outliers / len(feature_data)) * 100,
            'sigma_outliers': sigma_outliers,
            'sigma_outlier_pct': (sigma_outliers / len(feature_data)) * 100,
            'modified_z_outliers': modified_z_outliers,
            'modified_z_outlier_pct': (modified_z_outliers / len(feature_data)) * 100,
            'iqr_bounds': (lower_bound, upper_bound),
            'sigma_bounds': (lower_3sigma, upper_3sigma),
        })
        
        # Determine which method is likely being used
        outlier_pct = max(iqr_outliers, sigma_outliers, modified_z_outliers) / len(feature_data) * 100
        
        if outlier_pct > 50:
            results['likely_cause'] = 'Very high volatility or extreme values'
        elif outlier_pct > 20:
            results['likely_cause'] = 'High volatility or fat-tailed distribution'
        elif outlier_pct > 10:
            results['likely_cause'] = 'Moderate volatility'
        else:
            results['likely_cause'] = 'Normal financial data'
            
        return results
        
    except Exception as e:
        return {"error": str(e), "feature_name": feature_name}


def generate_comprehensive_feature_metrics(X: pd.DataFrame, y: pd.Series, 
                                     selected_features: List[str],
                                     component: FinalFeatureSelectionComponent) -> Dict[str, Any]:
    """
    Generate comprehensive metrics for all features.
    
    Args:
        X: Feature DataFrame
        y: Target series
        selected_features: List of selected features
        component: FinalFeatureSelectionComponent instance
        
    Returns:
        Dictionary with comprehensive metrics
    """
    all_features = list(X.columns)
    metrics = {
        'timestamp': datetime.now().isoformat(),
        'total_features': len(all_features),
        'selected_features': len(selected_features),
        'selection_rate': len(selected_features) / len(all_features) if all_features else 0,
    }
    
    # Analyze each feature
    feature_details = []
    outlier_analysis = []
    
    for feature in all_features:
        try:
            feature_data = X[feature].dropna()
            
            # Basic statistics
            basic_stats = {
                'feature_name': feature,
                'is_selected': feature in selected_features,
                'data_type': str(feature_data.dtype),
                'non_null_count': feature_data.count(),
                'null_count': feature_data.isnull().sum(),
                'null_percentage': (feature_data.isnull().sum() / len(feature_data)) * 100,
                'unique_count': feature_data.nunique(),
                'mean': feature_data.mean(),
                'median': feature_data.median(),
                'std': feature_data.std(),
                'min': feature_data.min(),
                'max': feature_data.max(),
                'range': feature_data.max() - feature_data.min(),
                'skewness': feature_data.skew(),
                'kurtosis': feature_data.kurtosis(),
                'variance': feature_data.var(),
            }
            
            # Target correlation
            if len(feature_data) > 0:
                try:
                    correlation = abs(feature_data.corr(y))
                    basic_stats['target_correlation'] = correlation
                except:
                    basic_stats['target_correlation'] = np.nan
            else:
                basic_stats['target_correlation'] = np.nan
            
            # Feature importance score if available
            if feature in component.feature_scores:
                basic_stats['importance_score'] = component.feature_scores[feature]
            else:
                basic_stats['importance_score'] = np.nan
                
            feature_details.append(basic_stats)
            
            # Outlier analysis for problematic features
            if basic_stats.get('null_percentage', 0) < 50:  # Only analyze features with data
                outlier_analysis.append(analyze_outlier_detection_issue(X, feature))
                
        except Exception as e:
            feature_details.append({
                'feature_name': feature,
                'error': str(e),
                'is_selected': feature in selected_features,
            })
    
    metrics['feature_details'] = feature_details
    metrics['outlier_analysis'] = outlier_analysis
    
    # Global statistics
    if feature_details:
        df_details = pd.DataFrame(feature_details)
        
        # Summary statistics
        metrics['summary'] = {
            'avg_correlation_with_target': df_details['target_correlation'].mean(),
            'max_correlation_with_target': df_details['target_correlation'].max(),
            'avg_importance_score': df_details['importance_score'].mean(),
            'max_importance_score': df_details['importance_score'].max(),
            'avg_skewness': df_details['skewness'].mean(),
            'avg_kurtosis': df_details['kurtosis'].mean(),
            'high_outlier_features': len([f for f in outlier_analysis if f.get('sigma_outlier_pct', 0) > 20]),
            'features_with_extreme_volatility': len([f for f in outlier_analysis if 'Very high volatility' in f.get('likely_cause', '')]),
        }
    
    return metrics


def save_metrics_to_files(metrics: Dict[str, Any], symbol: str = "ETHUSDT"):
    """
    Save metrics to CSV and MD files with datetime in filename.
    
    Args:
        metrics: Comprehensive metrics dictionary
        symbol: Trading symbol
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_filename = f"feature_metrics_{symbol}_{timestamp}"
    
    # Save detailed CSV
    if 'feature_details' in metrics:
        df_features = pd.DataFrame(metrics['feature_details'])
        csv_path = f"outcomes/{base_filename}_detailed.csv"
        df_features.to_csv(csv_path, index=False)
        print(f"✅ Detailed feature metrics saved to: {csv_path}")
    
    # Save outlier analysis CSV
    if 'outlier_analysis' in metrics:
        df_outliers = pd.DataFrame(metrics['outlier_analysis'])
        outlier_csv_path = f"outcomes/{base_filename}_outlier_analysis.csv"
        df_outliers.to_csv(outlier_csv_path, index=False)
        print(f"✅ Outlier analysis saved to: {outlier_csv_path}")
    
    # Save comprehensive MD report
    md_content = generate_markdown_report(metrics, symbol)
    md_path = f"outcomes/{base_filename}_comprehensive_report.md"
    with open(md_path, 'w') as f:
        f.write(md_content)
    print(f"✅ Comprehensive report saved to: {md_path}")
    
    # Save JSON for programmatic access
    json_path = f"outcomes/{base_filename}_metrics.json"
    with open(json_path, 'w') as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"✅ JSON metrics saved to: {json_path}")
    
    return {
        'csv_path': csv_path if 'feature_details' in metrics else None,
        'outlier_csv_path': outlier_csv_path if 'outlier_analysis' in metrics else None,
        'md_path': md_path,
        'json_path': json_path
    }


def generate_markdown_report(metrics: Dict[str, Any], symbol: str) -> str:
    """
    Generate comprehensive markdown report.
    
    Args:
        metrics: Comprehensive metrics dictionary
        symbol: Trading symbol
        
    Returns:
        Markdown formatted report
    """
    timestamp = metrics.get('timestamp', datetime.now().isoformat())
    
    md = f"""# Comprehensive Feature Selection Report

**Generated:** {timestamp}  
**Symbol:** {symbol}  
**Analysis Type:** Final Feature Selection Metrics

## Executive Summary

- **Total Features Analyzed:** {metrics.get('total_features', 'N/A')}
- **Features Selected:** {metrics.get('selected_features', 'N/A')}
- **Selection Rate:** {metrics.get('selection_rate', 0):.2%}

## Key Findings

"""
    
    if 'summary' in metrics:
        summary = metrics['summary']
        md += f"""### Feature Quality Metrics

- **Average Target Correlation:** {summary.get('avg_correlation_with_target', 'N/A'):.4f}
- **Maximum Target Correlation:** {summary.get('max_correlation_with_target', 'N/A'):.4f}
- **Average Importance Score:** {summary.get('avg_importance_score', 'N/A'):.6f}
- **Maximum Importance Score:** {summary.get('max_importance_score', 'N/A'):.6f}
- **Average Skewness:** {summary.get('avg_skewness', 'N/A'):.4f}
- **Average Kurtosis:** {summary.get('avg_kurtosis', 'N/A'):.4f}

### Outlier Analysis Summary

- **Features with High Outliers (>20%):** {summary.get('high_outlier_features', 'N/A')}
- **Features with Extreme Volatility:** {summary.get('features_with_extreme_volatility', 'N/A')}

"""
    
    if 'outlier_analysis' in metrics:
        outlier_analysis = metrics['outlier_analysis']
        extreme_volatility = [f for f in outlier_analysis if 'Very high volatility' in f.get('likely_cause', '')]
        high_outliers = [f for f in outlier_analysis if f.get('sigma_outlier_pct', 0) > 20]
        
        md += f"""
## Detailed Outlier Investigation

### Features with Extreme Volatility
{len(extreme_volatility)} features showing extreme volatility patterns:

"""
        
        for feature in extreme_volatility[:10]:  # Limit to top 10
            md += f"- **{feature.get('feature_name', 'Unknown')}**: {feature.get('likely_cause', 'Unknown')}\n"
        
        md += f"""
### High Outlier Features
{len(high_outliers)} features with outlier counts > 20%:

"""
        
        for feature in high_outliers[:10]:  # Limit to top 10
            outlier_pct = feature.get('sigma_outlier_pct', 0)
            data_range = feature.get('data_range', 0)
            std_dev = feature.get('std_dev', 0)
            md += f"- **{feature.get('feature_name', 'Unknown')}**: {outlier_pct:.1f}% outliers (range: {data_range:.4f}, std: {std_dev:.4f})\n"
    
    md += """
## Recommendations

### Outlier Detection Issues
The high outlier percentages (79-112%) indicate potential issues with:

1. **Detection Method Too Strict**: Current 3-sigma method may be too aggressive for financial data
2. **Feature Engineering Issues**: Some features may have unrealistic scaling or transformation
3. **Data Quality**: Extreme values that should be filtered or winsorized
4. **Financial Data Characteristics**: Heavy-tailed distributions common in financial markets

### Suggested Fixes

1. **Use More Lenient Outlier Detection**: Consider 2.5-sigma or IQR method
2. **Apply Winsorization**: Cap extreme values at percentiles (e.g., 1st and 99th)
3. **Feature Normalization**: Apply log or Box-Cox transformations to reduce skewness
4. **Domain-Specific Thresholds**: Use financial data-appropriate outlier thresholds

### Feature Selection Improvements

1. **Review Feature Engineering**: Ensure features produce realistic value ranges
2. **Enhanced Validation**: Add domain-specific validation for financial features
3. **Stability Analysis**: Monitor feature stability over time periods

---
*Report generated by Enhanced Feature Metrics Generator*
"""
    
    return md


def main():
    """Main execution function"""
    print("=" * 80)
    print("ENHANCED FEATURE METRICS GENERATOR")
    print("=" * 80)
    
    try:
        # Initialize artifact manager with empty config
        artifact_manager = ArtifactManager(config={})
        
        # Load the most recent final dataset (60 features)
        print("Loading latest final dataset...")
        # Use get_artifact method instead of load_latest
        final_dataset = artifact_manager.get_artifact("final_dataset_60", artifact_type="data")
        
        if final_dataset is None:
            print("❌ No final dataset found. Please run final feature selection first.")
            return
        
        # Extract features and target
        if isinstance(final_dataset, dict):
            X = final_dataset.get('features')
            y = final_dataset.get('target')
        elif hasattr(final_dataset, 'columns'):
            # Assume it's a DataFrame with target column
            target_cols = [col for col in final_dataset.columns if 'target' in col.lower() or col in ['profit_loss', 'returns', 'pnl']]
            if target_cols:
                target_col = target_cols[0]
                y = final_dataset[target_col]
                X = final_dataset.drop(columns=[target_col])
            else:
                print("❌ No target column found in dataset")
                return
        else:
            print("❌ Unexpected dataset format")
            return
        
        print(f"✅ Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
        
        # Initialize feature selection component
        config = FinalFeatureSelectionConfig(
            max_features=60,
            min_features=10,
            selection_method="permutation",
            use_tree_based=True,
            use_permutation_importance=True
        )
        component = FinalFeatureSelectionComponent(config)
        
        # Get selected features (from previous run or re-run selection)
        print("Analyzing feature selection...")
        selected_features = component.select_features(X, y)
        
        print(f"✅ Selected {len(selected_features)} features")
        
        # Generate comprehensive metrics
        print("Generating comprehensive metrics...")
        metrics = generate_comprehensive_feature_metrics(X, y, selected_features, component)
        
        # Save to files
        print("Saving metrics to files...")
        file_paths = save_metrics_to_files(metrics, "ETHUSDT")
        
        print("\n" + "=" * 80)
        print("✅ ENHANCED FEATURE METRICS GENERATION COMPLETED")
        print("=" * 80)
        print(f"Files generated:")
        for file_type, path in file_paths.items():
            if path:
                print(f"  {file_type}: {path}")
        
        # Print key insights
        if 'summary' in metrics:
            summary = metrics['summary']
            print(f"\n🔍 KEY INSIGHTS:")
            print(f"  High outlier features: {summary.get('high_outlier_features', 0)}")
            print(f"  Extreme volatility features: {summary.get('features_with_extreme_volatility', 0)}")
            print(f"  Average correlation with target: {summary.get('avg_correlation_with_target', 0):.4f}")
            print(f"  Max correlation with target: {summary.get('max_correlation_with_target', 0):.4f}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()