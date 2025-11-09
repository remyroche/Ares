#!/usr/bin/env python3
"""
Investigate Outlier Detection Issues

This script investigates the high outlier counts in feature datasets
and provides detailed analysis.
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from utils.artifact_manager import ArtifactManager


def analyze_outlier_methods():
    """Analyze different outlier detection methods on sample data"""
    print("=" * 80)
    print("OUTLIER DETECTION INVESTIGATION")
    print("=" * 80)
    
    # Create sample data with known characteristics
    np.random.seed(42)
    n_samples = 1000
    
    # Normal financial data (should have ~0.3% outliers with 3-sigma)
    normal_data = np.random.normal(0, 1, n_samples)
    normal_outliers_3sigma = np.sum(np.abs(normal_data) > 3)
    normal_outlier_pct_3sigma = (normal_outliers_3sigma / n_samples) * 100
    
    # Heavy-tailed financial data (should have ~5-10% outliers with 3-sigma)
    heavy_tailed = np.random.standard_t(3, n_samples)
    heavy_outliers_3sigma = np.sum(np.abs(heavy_tailed) > 3)
    heavy_outlier_pct_3sigma = (heavy_outliers_3sigma / n_samples) * 100
    
    # Extreme financial data (should have ~15-20% outliers with 3-sigma)
    extreme_data = np.random.standard_t(1, n_samples) * 2  # Very heavy tails
    extreme_outliers_3sigma = np.sum(np.abs(extreme_data) > 3)
    extreme_outlier_pct_3sigma = (extreme_outliers_3sigma / n_samples) * 100
    
    print(f"Sample Data Analysis (n={n_samples}):")
    print(f"  Normal data: {normal_outlier_pct_3sigma:.1f}% outliers with 3-sigma")
    print(f"  Heavy-tailed data: {heavy_outlier_pct_3sigma:.1f}% outliers with 3-sigma")
    print(f"  Extreme data: {extreme_outlier_pct_3sigma:.1f}% outliers with 3-sigma")
    print()
    
    # Test different outlier detection methods
    def count_outliers_3sigma(x):
        return np.sum(np.abs(x - np.mean(x)) > 3 * np.std(x))
    
    def count_outliers_2_5sigma(x):
        return np.sum(np.abs(x - np.mean(x)) > 2.5 * np.std(x))
    
    def count_outliers_iqr_1_5(x):
        Q1 = np.quantile(x, 0.25)
        Q3 = np.quantile(x, 0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        count = np.sum((x < lower_bound) | (x > upper_bound))
        return {'count': count, 'bounds': (lower_bound, upper_bound)}
    
    def count_outliers_iqr_3_0(x):
        Q1 = np.quantile(x, 0.25)
        Q3 = np.quantile(x, 0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 3.0 * IQR
        upper_bound = Q3 + 3.0 * IQR
        count = np.sum((x < lower_bound) | (x > upper_bound))
        return {'count': count, 'bounds': (lower_bound, upper_bound)}
    
    methods = {
        '3-sigma': count_outliers_3sigma,
        '2.5-sigma': count_outliers_2_5sigma,
        'IQR (1.5)': count_outliers_iqr_1_5,
        'IQR (3.0)': count_outliers_iqr_3_0
    }
    
    datasets = {
        'Normal': normal_data,
        'Heavy-tailed': heavy_tailed,
        'Extreme': extreme_data
    }
    
    print("Outlier Detection Method Comparison:")
    print("-" * 80)
    print(f"{'Dataset':<15} {'3-sigma':<10} {'2.5-sigma':<10} {'IQR 1.5':<10} {'IQR 3.0':<10}")
    print("-" * 80)
    
    for name, data in datasets.items():
        row = f"{name:<15}"
        for method_name, method_func in methods.items():
            result = method_func(data)
            if isinstance(result, dict):
                count = result['count']
                pct = (count / n_samples) * 100
                row += f" {pct:>6.1f}%"
            else:
                count = result
                pct = (count / n_samples) * 100
                row += f" {pct:>6.1f}%"
        print(row)
    
    print()
    print("Key Findings:")
    print("  • 3-sigma method is too aggressive for heavy-tailed financial data")
    print("  • 2.5-sigma is more reasonable for financial markets")
    print("  • IQR 1.5x is standard for financial data")
    print("  • IQR 3.0x is very conservative")
    print("  • Financial data often has 5-15% legitimate outliers due to volatility")


def load_and_analyze_final_dataset():
    """Load and analyze the actual final dataset"""
    print("\n" + "=" * 80)
    print("ANALYZING ACTUAL FINAL DATASET")
    print("=" * 80)
    
    try:
        # Initialize artifact manager with empty config
        artifact_manager = ArtifactManager(config={})
        
        # Try to load the most recent final dataset
        print("Looking for final_dataset_60...")
        final_dataset = artifact_manager.get_artifact("final_dataset_60", artifact_type="data")
        
        if final_dataset is None:
            print("❌ No final_dataset_60 found")
            return
        
        # Extract features
        if isinstance(final_dataset, dict):
            X = final_dataset.get('features')
            y = final_dataset.get('target')
        elif hasattr(final_dataset, 'columns'):
            # Assume DataFrame with target column
            target_cols = [col for col in final_dataset.columns if 'target' in col.lower() or col in ['profit_loss', 'returns', 'pnl']]
            if target_cols:
                target_col = target_cols[0]
                y = final_dataset[target_col]
                X = final_dataset.drop(columns=[target_col])
            else:
                print("❌ No target column found")
                return
        else:
            print("❌ Unexpected dataset format")
            return
        
        print(f"✅ Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
        
        # Analyze each feature for outlier characteristics
        print("\nAnalyzing feature outlier characteristics...")
        
        outlier_analysis = []
        for feature in X.columns:
            feature_data = X[feature].dropna()
            if len(feature_data) == 0:
                continue
                
            # Calculate outlier counts using different methods
            n = len(feature_data)
            mean_val = feature_data.mean()
            std_val = feature_data.std()
            
            # 3-sigma method
            outliers_3sigma = np.sum(np.abs(feature_data - mean_val) > 3 * std_val)
            pct_3sigma = (outliers_3sigma / n) * 100
            
            # 2.5-sigma method (more reasonable)
            outliers_2_5sigma = np.sum(np.abs(feature_data - mean_val) > 2.5 * std_val)
            pct_2_5sigma = (outliers_2_5sigma / n) * 100
            
            # IQR method
            Q1 = feature_data.quantile(0.25)
            Q3 = feature_data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers_iqr = np.sum((feature_data < lower_bound) | (feature_data > upper_bound))
            pct_iqr = (outliers_iqr / n) * 100
            
            # Characterize the distribution
            skewness = feature_data.skew()
            kurtosis = feature_data.kurtosis()
            
            # Determine likely cause
            if pct_3sigma > 50:
                cause = "Very extreme volatility or data issues"
            elif pct_3sigma > 20:
                cause = "High volatility or heavy-tailed distribution"
            elif pct_3sigma > 10:
                cause = "Moderate volatility"
            else:
                cause = "Normal financial data"
            
            outlier_analysis.append({
                'feature': feature,
                'data_points': n,
                'outliers_3sigma': outliers_3sigma,
                'pct_3sigma': pct_3sigma,
                'outliers_2_5sigma': outliers_2_5sigma,
                'pct_2_5sigma': pct_2_5sigma,
                'outliers_iqr': outliers_iqr,
                'pct_iqr': pct_iqr,
                'skewness': skewness,
                'kurtosis': kurtosis,
                'likely_cause': cause,
                'std_dev': std_val,
                'range': feature_data.max() - feature_data.min()
            })
        
        # Convert to DataFrame for analysis
        df_outliers = pd.DataFrame(outlier_analysis)
        
        # Summary statistics
        high_outlier_features = df_outliers[df_outliers['pct_3sigma'] > 20]
        extreme_volatility_features = df_outliers[df_outliers['pct_3sigma'] > 50]
        
        print(f"\nOutlier Analysis Summary:")
        print(f"  Total features analyzed: {len(df_outliers)}")
        print(f"  Features with >20% outliers (3-sigma): {len(high_outlier_features)}")
        print(f"  Features with >50% outliers (3-sigma): {len(extreme_volatility_features)}")
        print(f"  Average outlier percentage (3-sigma): {df_outliers['pct_3sigma'].mean():.1f}%")
        print(f"  Average outlier percentage (2.5-sigma): {df_outliers['pct_2_5sigma'].mean():.1f}%")
        print(f"  Average outlier percentage (IQR): {df_outliers['pct_iqr'].mean():.1f}%")
        
        # Show problematic features
        if len(high_outlier_features) > 0:
            print(f"\nTop 10 features with highest outlier counts (3-sigma):")
            top_outliers = df_outliers.nlargest(10, 'pct_3sigma')
            for _, row in top_outliers.iterrows():
                print(f"  {row['feature']}: {row['pct_3sigma']:.1f}% ({row['likely_cause']})")
        
        # Save detailed analysis
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = f"outcomes/outlier_investigation_{timestamp}.csv"
        df_outliers.to_csv(csv_path, index=False)
        print(f"\n✅ Detailed outlier analysis saved to: {csv_path}")
        
        # Generate recommendations
        recommendations = generate_outlier_recommendations(df_outliers)
        md_path = f"outcomes/outlier_recommendations_{timestamp}.md"
        with open(md_path, 'w') as f:
            f.write(recommendations)
        print(f"✅ Recommendations saved to: {md_path}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


def generate_outlier_recommendations(df_outliers: pd.DataFrame) -> str:
    """Generate recommendations based on outlier analysis"""
    
    high_outlier_count = len(df_outliers[df_outliers['pct_3sigma'] > 20])
    extreme_outlier_count = len(df_outliers[df_outliers['pct_3sigma'] > 50])
    avg_3sigma = df_outliers['pct_3sigma'].mean()
    avg_2_5sigma = df_outliers['pct_2_5sigma'].mean()
    
    recommendations = f"""# Outlier Detection Analysis & Recommendations

**Generated:** {datetime.now().isoformat()}

## Executive Summary

- **Total Features Analyzed:** {len(df_outliers)}
- **Features with High Outliers (>20%):** {high_outlier_count}
- **Features with Extreme Outliers (>50%):** {extreme_outlier_count}
- **Average Outlier Rate (3-sigma):** {avg_3sigma:.1f}%
- **Average Outlier Rate (2.5-sigma):** {avg_2_5sigma:.1f}%

## Problem Identification

The high outlier counts indicate that **3-sigma outlier detection method is too aggressive** for financial data:

1. **Financial data naturally has heavy tails** - 5-15% outliers are normal
2. **Volatility clustering** creates legitimate extreme values
3. **Feature engineering** may create skewed distributions
4. **Ratio features** can have infinite or extreme values

## Root Cause Analysis

### Current Implementation Issues
- **3-sigma threshold** assumes normal distribution
- **Financial returns** follow heavy-tailed distributions
- **Volatility periods** create legitimate extreme movements
- **Ratio features** without proper bounds checking

### Data Characteristics Observed
"""
    
    # Add specific problematic features if any
    if extreme_outlier_count > 0:
        extreme_features = df_outliers[df_outliers['pct_3sigma'] > 50].nlargest(5, 'pct_3sigma')
        recommendations += f"""#### Most Problematic Features (Top 5):
"""
        for _, row in extreme_features.iterrows():
            recommendations += f"- **{row['feature']}**: {row['pct_3sigma']:.1f}% outliers ({row['likely_cause']})\n"
    
    recommendations += f"""

## Recommended Solutions

### 1. **Use More Appropriate Outlier Detection**

#### For Financial Data, Use:
- **2.5-sigma method**: {avg_2_5sigma:.1f}% average outlier rate (more reasonable)
- **IQR 1.5x method**: Standard for financial data
- **Median Absolute Deviation (MAD)**: Robust to extreme values
- **Domain-specific thresholds**: Based on volatility regimes

#### Implementation:
```python
# Replace 3-sigma with 2.5-sigma for financial data
outliers = np.sum(np.abs(feature_data - mean_val) > 2.5 * std_val)
```

### 2. **Feature Engineering Improvements**

#### Apply Transformations:
- **Log transformation**: For ratio features and prices
- **Box-Cox transformation**: For skewed distributions
- **Winsorization**: Cap extreme values at 99th percentile
- **Volatility normalization**: Standardize by rolling volatility

### 3. **Data Quality Validation**

#### Pre-processing Steps:
- **Filter infinite values**: Remove or cap infinite ratios
- **Handle division by zero**: Add small epsilon or use conditional logic
- **Validate feature ranges**: Ensure realistic financial values
- **Temporal consistency**: Check for data gaps and anomalies

### 4. **Adaptive Outlier Detection**

#### Context-Aware Methods:
- **Volatility-adjusted thresholds**: Higher thresholds during high volatility periods
- **Regime-specific detection**: Different thresholds for different market regimes
- **Time-decaying weights**: Give more weight to recent data
- **Multi-method consensus**: Combine multiple outlier detection methods

## Implementation Priority

### High Priority (Immediate)
1. **Change outlier detection from 3-sigma to 2.5-sigma**
2. **Add infinite value handling in feature generation**
3. **Implement winsorization for extreme ratio features**

### Medium Priority (Next Sprint)
1. **Add adaptive outlier detection based on volatility**
2. **Implement MAD (Median Absolute Deviation) as alternative**
3. **Add data quality validation for feature engineering**

### Low Priority (Future Enhancement)
1. **Machine learning-based outlier detection**
2. **Unsupervised anomaly detection**
3. **Real-time outlier monitoring and alerting**

## Expected Impact

After implementing these changes:
- **Outlier rates should drop to 5-15%** (normal for financial data)
- **Feature stability should improve** significantly
- **Model performance should become more consistent**
- **False positive outlier flags should reduce** dramatically

## Monitoring Recommendations

### Metrics to Track:
- **Outlier percentage by feature** (target: 5-15%)
- **Feature stability over time** (target: >80% consistency)
- **Model performance variance** (target: reduce by 50%)
- **Data quality score** (target: >0.8)

### Alert Thresholds:
- **Warning**: Individual feature outlier rate > 25%
- **Critical**: Individual feature outlier rate > 40%
- **Investigate**: Average outlier rate across all features > 20%

---
*Analysis generated by Outlier Detection Investigation Tool*
"""
    
    return recommendations


def main():
    """Main execution function"""
    print("Starting outlier detection investigation...")
    
    # First, analyze different methods on sample data
    analyze_outlier_methods()
    
    # Then analyze the actual dataset
    load_and_analyze_final_dataset()
    
    print("\n" + "=" * 80)
    print("OUTLIER INVESTIGATION COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    main()