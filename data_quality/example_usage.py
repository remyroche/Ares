#!/usr/bin/env python3
"""
Example Usage of the Unified Data Quality Orchestrator

This script demonstrates how to use the UnifiedQualityOrchestrator
to perform comprehensive data quality analysis on sample data.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from pathlib import Path

# Add the parent directory to the path to import the orchestrator
import sys
sys.path.append(str(Path(__file__).parent))

from unified_quality_orchestrator import UnifiedQualityOrchestrator, QualityThresholds


def create_sample_data():
    """Create sample data for demonstration purposes."""
    print("🔧 Creating sample data...")
    
    # Create timestamp range
    start_time = datetime(2024, 1, 1)
    timestamps = [start_time + timedelta(hours=i) for i in range(1000)]
    
    # Create sample OHLCV data (klines)
    np.random.seed(42)
    base_price = 100.0
    
    klines_data = {
        'timestamp': [int(ts.timestamp()) for ts in timestamps],
        'open': [base_price + np.random.normal(0, 1) for _ in range(1000)],
        'high': [base_price + np.random.normal(0, 1.5) for _ in range(1000)],
        'low': [base_price + np.random.normal(0, 1.5) for _ in range(1000)],
        'close': [base_price + np.random.normal(0, 1) for _ in range(1000)],
        'volume': [np.random.uniform(1000, 10000) for _ in range(1000)]
    }
    
    # Ensure high >= max(open, close) and low <= min(open, close)
    for i in range(1000):
        open_price = klines_data['open'][i]
        close_price = klines_data['close'][i]
        klines_data['high'][i] = max(open_price, close_price) + abs(np.random.normal(0, 0.5))
        klines_data['low'][i] = min(open_price, close_price) - abs(np.random.normal(0, 0.5))
    
    klines_df = pd.DataFrame(klines_data)
    
    # Create sample features data
    features_data = {
        'timestamp': klines_data['timestamp'],
        'sma_20': klines_df['close'].rolling(20).mean(),
        'sma_50': klines_df['close'].rolling(50).mean(),
        'rsi': [50 + np.random.normal(0, 15) for _ in range(1000)],
        'macd': [np.random.normal(0, 0.5) for _ in range(1000)],
        'bollinger_upper': klines_df['close'] + klines_df['close'].rolling(20).std() * 2,
        'bollinger_lower': klines_df['close'] - klines_df['close'].rolling(20).std() * 2,
        'volume_sma': klines_df['volume'].rolling(20).mean(),
        'price_change': klines_df['close'].pct_change(),
        'volatility': klines_df['close'].rolling(20).std(),
        'momentum': klines_df['close'] - klines_df['close'].shift(10)
    }
    
    features_df = pd.DataFrame(features_data)
    
    # Create sample labels data
    labels_data = {
        'timestamp': klines_data['timestamp'][:-1],  # Remove last timestamp for labels
        'label': [np.random.choice([0, 1], p=[0.7, 0.3]) for _ in range(999)]  # Imbalanced labels
    }
    
    labels_df = pd.DataFrame(labels_data)
    
    # Add some quality issues for demonstration
    # Add some NaN values
    features_df.loc[100:110, 'rsi'] = np.nan
    features_df.loc[200:205, 'macd'] = np.nan
    
    # Add some infinite values
    features_df.loc[300, 'momentum'] = np.inf
    features_df.loc[301, 'momentum'] = -np.inf
    
    # Add some constant columns
    features_df['constant_feature'] = 42
    
    # Add some highly correlated features
    features_df['highly_correlated'] = features_df['sma_20'] * 1.1 + np.random.normal(0, 0.01)
    
    print(f"✅ Created sample data:")
    print(f"   - Klines: {klines_df.shape}")
    print(f"   - Features: {features_df.shape}")
    print(f"   - Labels: {labels_df.shape}")
    
    return klines_df, features_df, labels_df


def demonstrate_basic_quality_validation():
    """Demonstrate basic data quality validation."""
    print("\n" + "="*60)
    print("🔍 DEMONSTRATING BASIC DATA QUALITY VALIDATION")
    print("="*60)
    
    # Create sample data
    klines_df, features_df, labels_df = create_sample_data()
    
    # Initialize orchestrator
    orchestrator = UnifiedQualityOrchestrator()
    
    # Validate klines data
    print("\n📊 Validating klines data...")
    klines_quality = orchestrator.validate_dataframe_quality(klines_df, "Sample OHLCV Data")
    
    print(f"   Quality check passed: {klines_quality.passed}")
    print(f"   Issues found: {len(klines_quality.issues)}")
    print(f"   Warnings: {len(klines_quality.warnings)}")
    print(f"   Memory usage: {klines_quality.metrics.get('memory_mb', 0):.2f} MB")
    
    if klines_quality.issues:
        print("   Issues:")
        for issue in klines_quality.issues[:3]:  # Show first 3 issues
            print(f"     - {issue}")
    
    # Validate features data
    print("\n📊 Validating features data...")
    features_quality = orchestrator.validate_dataframe_quality(features_df, "Sample Features Data")
    
    print(f"   Quality check passed: {features_quality.passed}")
    print(f"   Issues found: {len(features_quality.issues)}")
    print(f"   Warnings: {len(features_quality.warnings)}")
    
    if features_quality.issues:
        print("   Issues:")
        for issue in features_quality.issues[:3]:
            print(f"     - {issue}")
    
    if features_quality.warnings:
        print("   Warnings:")
        for warning in features_quality.warnings[:3]:
            print(f"     - {warning}")


def demonstrate_advanced_analysis():
    """Demonstrate advanced analysis capabilities."""
    print("\n" + "="*60)
    print("🔍 DEMONSTRATING ADVANCED ANALYSIS")
    print("="*60)
    
    # Create sample data
    klines_df, features_df, labels_df = create_sample_data()
    
    # Initialize orchestrator
    orchestrator = UnifiedQualityOrchestrator()
    
    # Multicollinearity analysis
    print("\n📊 Analyzing multicollinearity...")
    try:
        multicollinearity = orchestrator.analyze_multicollinearity(features_df)
        
        print(f"   High VIF features: {len(multicollinearity['high_vif_features'])}")
        if multicollinearity['high_vif_features']:
            print(f"     - {multicollinearity['high_vif_features']}")
        
        print(f"   High correlation pairs: {len(multicollinearity['high_correlation_pairs'])}")
        if multicollinearity['high_correlation_pairs']:
            for pair in multicollinearity['high_correlation_pairs'][:3]:
                print(f"     - {pair[0]} ↔ {pair[1]} (r={pair[2]:.3f})")
    
    except Exception as e:
        print(f"   ❌ Multicollinearity analysis failed: {e}")
    
    # Feature redundancy analysis
    print("\n📊 Analyzing feature redundancy...")
    try:
        feature_redundancy = orchestrator.analyze_feature_redundancy(features_df)
        
        print(f"   Redundancy ratio: {feature_redundancy['redundancy_ratio']:.2%}")
        print(f"   Redundant features: {len(feature_redundancy['redundant_features'])}")
        
        if feature_redundancy['recommendations']:
            print("   Recommendations:")
            for rec in feature_redundancy['recommendations']:
                print(f"     - {rec}")
    
    except Exception as e:
        print(f"   ❌ Feature redundancy analysis failed: {e}")
    
    # Label imbalance analysis
    print("\n📊 Analyzing label imbalance...")
    try:
        label_imbalance = orchestrator.analyze_label_imbalance(labels_df['label'])
        
        print(f"   Imbalance level: {label_imbalance['imbalance_level']}")
        print(f"   Imbalance ratio: {label_imbalance['imbalance_ratio']:.2f}")
        print(f"   Total samples: {label_imbalance['total_samples']}")
        
        if label_imbalance['recommendations']:
            print("   Recommendations:")
            for rec in label_imbalance['recommendations']:
                print(f"     - {rec}")
    
    except Exception as e:
        print(f"   ❌ Label imbalance analysis failed: {e}")


def demonstrate_temporal_validation():
    """Demonstrate temporal data validation."""
    print("\n" + "="*60)
    print("🔍 DEMONSTRATING TEMPORAL VALIDATION")
    print("="*60)
    
    # Create sample data
    klines_df, features_df, labels_df = create_sample_data()
    
    # Initialize orchestrator
    orchestrator = UnifiedQualityOrchestrator()
    
    # Validate temporal aspects of klines data
    print("\n📊 Validating temporal aspects of klines data...")
    try:
        temporal_validation = orchestrator.validate_temporal_data(klines_df, 'timestamp')
        
        print(f"   Temporal validation passed: {temporal_validation.passed}")
        print(f"   Issues found: {len(temporal_validation.issues)}")
        print(f"   Warnings: {len(temporal_validation.warnings)}")
        
        if temporal_validation.metrics:
            print("   Temporal metrics:")
            for key, value in temporal_validation.metrics.items():
                if key in ['max_gap', 'min_gap', 'mean_gap']:
                    print(f"     - {key}: {value}")
        
        if temporal_validation.issues:
            print("   Issues:")
            for issue in temporal_validation.issues:
                print(f"     - {issue}")
    
    except Exception as e:
        print(f"   ❌ Temporal validation failed: {e}")


def demonstrate_comprehensive_report():
    """Demonstrate comprehensive report generation."""
    print("\n" + "="*60)
    print("🔍 DEMONSTRATING COMPREHENSIVE REPORT GENERATION")
    print("="*60)
    
    # Create sample data
    klines_df, features_df, labels_df = create_sample_data()
    
    # Initialize orchestrator
    orchestrator = UnifiedQualityOrchestrator()
    
    # Generate comprehensive report for features data
    print("\n📊 Generating comprehensive quality report...")
    try:
        report = orchestrator.generate_comprehensive_report(features_df, "Sample Features Dataset")
        
        # Display summary
        summary = report.get("summary", {})
        print(f"   Overall Quality: {summary.get('overall_quality', 'unknown').upper()}")
        print(f"   Critical Issues: {summary.get('critical_issues', 0)}")
        print(f"   Recommendations: {len(summary.get('recommendations', []))}")
        
        if summary.get('recommendations'):
            print("   Top Recommendations:")
            for rec in summary['recommendations'][:3]:
                print(f"     - {rec}")
        
        # Save report
        output_file = orchestrator.save_report(report, "example_quality_report.json")
        print(f"   ✅ Report saved to: {output_file}")
        
        # Display some detailed metrics
        if report.get("quality_validation"):
            quality_val = report["quality_validation"]
            print(f"   Quality Metrics:")
            print(f"     - Rows: {quality_val.get('metrics', {}).get('rows', 'N/A')}")
            print(f"     - Columns: {quality_val.get('metrics', {}).get('columns', 'N/A')}")
            print(f"     - Memory: {quality_val.get('metrics', {}).get('memory_mb', 0):.2f} MB")
        
    except Exception as e:
        print(f"   ❌ Comprehensive report generation failed: {e}")


def demonstrate_custom_thresholds():
    """Demonstrate custom threshold usage."""
    print("\n" + "="*60)
    print("🔍 DEMONSTRATING CUSTOM THRESHOLDS")
    print("="*60)
    
    # Create sample data
    klines_df, features_df, labels_df = create_sample_data()
    
    # Create custom thresholds
    custom_thresholds = QualityThresholds(
        max_nan_ratio=0.1,  # Allow up to 10% NaN values
        max_infinite_count=5,  # Allow up to 5 infinite values
        max_correlation_threshold=0.8,  # Lower correlation threshold
        vif_threshold=3.0  # Lower VIF threshold
    )
    
    # Initialize orchestrator with custom thresholds
    orchestrator = UnifiedQualityOrchestrator(custom_thresholds)
    
    print(f"   Custom thresholds applied:")
    print(f"     - Max NaN ratio: {custom_thresholds.max_nan_ratio:.1%}")
    print(f"     - Max infinite count: {custom_thresholds.max_infinite_count}")
    print(f"     - Max correlation threshold: {custom_thresholds.max_correlation_threshold}")
    print(f"     - VIF threshold: {custom_thresholds.vif_threshold}")
    
    # Validate with custom thresholds
    print("\n📊 Validating with custom thresholds...")
    quality_result = orchestrator.validate_dataframe_quality(features_df, "Features with Custom Thresholds")
    
    print(f"   Quality check passed: {quality_result.passed}")
    print(f"   Issues found: {len(quality_result.issues)}")
    print(f"   Warnings: {len(quality_result.warnings)}")


def main():
    """Main demonstration function."""
    print("🚀 UNIFIED DATA QUALITY ORCHESTRATOR DEMONSTRATION")
    print("="*60)
    
    try:
        # Demonstrate basic quality validation
        demonstrate_basic_quality_validation()
        
        # Demonstrate advanced analysis
        demonstrate_advanced_analysis()
        
        # Demonstrate temporal validation
        demonstrate_temporal_validation()
        
        # Demonstrate comprehensive report generation
        demonstrate_comprehensive_report()
        
        # Demonstrate custom thresholds
        demonstrate_custom_thresholds()
        
        print("\n" + "="*60)
        print("🎉 DEMONSTRATION COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\n💡 Next steps:")
        print("   1. Review the generated quality report: example_quality_report.json")
        print("   2. Try running the orchestrator on your own data")
        print("   3. Customize thresholds for your specific use case")
        print("   4. Integrate the orchestrator into your data pipeline")
        
    except Exception as e:
        print(f"\n❌ Demonstration failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()