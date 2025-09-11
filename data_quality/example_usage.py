#!/usr/bin/env python3
from src.utils.tprint import tprint

"""
Example Usage of the Unified Data Quality Orchestrator

This script demonstrates how to use the UnifiedQualityOrchestrator
to perform comprehensive data quality analysis on sample data.
"""


# Add the parent directory to the path to import the orchestrator
import sys
from datetime import datetime, timedelta
from pathlib import Path


sys.path.append(str(Path(__file__).parent))

from unified_quality_orchestrator import QualityThresholds, UnifiedQualityOrchestrator
import numpy as np
import pandas as pd
import os


def create_sample_data():
    """Create sample data for demonstration purposes."""
    tprint("🔧 Creating sample data...")

    # Create timestamp range
    start_time = datetime(2024, 1, 1)
    timestamps = [start_time + timedelta(hours=i) for i in range(1000)]

    # Create sample OHLCV data (klines)
    np.random.seed(42)
    base_price = 100.0

    klines_data = {
        "timestamp": [int(ts.timestamp()) for ts in timestamps],
        "open": [base_price + np.random.normal(0, 1) for _ in range(1000)],
        "high": [base_price + np.random.normal(0, 1.5) for _ in range(1000)],
        "low": [base_price + np.random.normal(0, 1.5) for _ in range(1000)],
        "close": [base_price + np.random.normal(0, 1) for _ in range(1000)],
        "volume": [np.random.uniform(1000, 10000) for _ in range(1000)],
    }

    # Ensure high >= max(open, close) and low <= min(open, close)
    for i in range(1000):
        open_price = klines_data["open"][i]
        close_price = klines_data["close"][i]
        klines_data["high"][i] = max(open_price, close_price) + abs(np.random.normal(0, 0.5))
        klines_data["low"][i] = min(open_price, close_price) - abs(np.random.normal(0, 0.5))

    klines_df = pd.DataFrame(klines_data)

    # Create sample features data
    features_data = {
        "timestamp": klines_data["timestamp"],
        "sma_20": klines_df["close"].rolling(20).mean(),
        "sma_50": klines_df["close"].rolling(50).mean(),
        "rsi": [50 + np.random.normal(0, 15) for _ in range(1000)],
        "macd": [np.random.normal(0, 0.5) for _ in range(1000)],
        "bollinger_upper": klines_df["close"] + klines_df["close"].rolling(20).std() * 2,
        "bollinger_lower": klines_df["close"] - klines_df["close"].rolling(20).std() * 2,
        "volume_sma": klines_df["volume"].rolling(20).mean(),
        "price_change": klines_df["close"].pct_change(),
        "volatility": klines_df["close"].rolling(20).std(),
        "momentum": klines_df["close"] - klines_df["close"].shift(10),
    }

    features_df = pd.DataFrame(features_data)

    # Create sample labels data
    labels_data = {
        "timestamp": klines_data["timestamp"][:-1],  # Remove last timestamp for labels
        "label": [np.random.choice([0, 1], p=[0.7, 0.3]) for _ in range(999)],  # Imbalanced labels
    }

    labels_df = pd.DataFrame(labels_data)

    # Add some quality issues for demonstration
    # Add some NaN values
    features_df.loc[100:110, "rsi"] = np.nan
    features_df.loc[200:205, "macd"] = np.nan

    # Add some infinite values
    features_df.loc[300, "momentum"] = np.inf
    features_df.loc[301, "momentum"] = -np.inf

    # Add some constant columns
    features_df["constant_feature"] = 42

    # Add some highly correlated features
    features_df["highly_correlated"] = features_df["sma_20"] * 1.1 + np.random.normal(0, 0.01)

    tprint("✅ Created sample data:")
    tprint(f"   - Klines: {klines_df.shape}")
    tprint(f"   - Features: {features_df.shape}")
    tprint(f"   - Labels: {labels_df.shape}")

    return klines_df, features_df, labels_df


def demonstrate_basic_quality_validation():
    """Demonstrate basic data quality validation."""
    tprint("\n" + "="*60)
    tprint("🔍 DEMONSTRATING BASIC DATA QUALITY VALIDATION")
    tprint("="*60)

    # Create sample data
    klines_df, features_df, labels_df = create_sample_data()

    # Initialize orchestrator
    orchestrator = UnifiedQualityOrchestrator()

    # Validate klines data
    tprint("\n📊 Validating klines data...")
    klines_quality = orchestrator.validate_dataframe_quality(klines_df, "Sample OHLCV Data")

    tprint(f"   Quality check passed: {klines_quality.passed}")
    tprint(f"   Issues found: {len(klines_quality.issues)}")
    tprint(f"   Warnings: {len(klines_quality.warnings)}")
    tprint(f"   Memory usage: {klines_quality.metrics.get('memory_mb', 0):.2f} MB")

    if klines_quality.issues:
        tprint("   Issues:")
        for issue in klines_quality.issues[:3]:  # Show first 3 issues
            tprint(f"     - {issue}")

    # Validate features data
    tprint("\n📊 Validating features data...")
    features_quality = orchestrator.validate_dataframe_quality(features_df, "Sample Features Data")

    tprint(f"   Quality check passed: {features_quality.passed}")
    tprint(f"   Issues found: {len(features_quality.issues)}")
    tprint(f"   Warnings: {len(features_quality.warnings)}")

    if features_quality.issues:
        tprint("   Issues:")
        for issue in features_quality.issues[:3]:
            tprint(f"     - {issue}")

    if features_quality.warnings:
        tprint("   Warnings:")
        for warning in features_quality.warnings[:3]:
            tprint(f"     - {warning}")


def demonstrate_advanced_analysis():
    """Demonstrate advanced analysis capabilities."""
    tprint("\n" + "="*60)
    tprint("🔍 DEMONSTRATING ADVANCED ANALYSIS")
    tprint("="*60)

    # Create sample data
    klines_df, features_df, labels_df = create_sample_data()

    # Initialize orchestrator
    orchestrator = UnifiedQualityOrchestrator()

    # Multicollinearity analysis
    tprint("\n📊 Analyzing multicollinearity...")
    try:
        multicollinearity = orchestrator.analyze_multicollinearity(features_df)

        tprint(f"   High VIF features: {len(multicollinearity['high_vif_features'])}")
        if multicollinearity["high_vif_features"]:
            tprint(f"     - {multicollinearity['high_vif_features']}")

        tprint(f"   High correlation pairs: {len(multicollinearity['high_correlation_pairs'])}")
        if multicollinearity["high_correlation_pairs"]:
            for pair in multicollinearity["high_correlation_pairs"][:3]:
                tprint(f"     - {pair[0]} ↔ {pair[1]} (r={pair[2]:.3f})")

    except Exception as e:
        tprint(f"   ❌ Multicollinearity analysis failed: {e}")

    # Feature redundancy analysis
    tprint("\n📊 Analyzing feature redundancy...")
    try:
        feature_redundancy = orchestrator.analyze_feature_redundancy(features_df)

        tprint(f"   Redundancy ratio: {feature_redundancy['redundancy_ratio']:.2%}")
        tprint(f"   Redundant features: {len(feature_redundancy['redundant_features'])}")

        if feature_redundancy["recommendations"]:
            tprint("   Recommendations:")
            for rec in feature_redundancy["recommendations"]:
                tprint(f"     - {rec}")

    except Exception as e:
        tprint(f"   ❌ Feature redundancy analysis failed: {e}")

    # Label imbalance analysis
    tprint("\n📊 Analyzing label imbalance...")
    try:
        label_imbalance = orchestrator.analyze_label_imbalance(labels_df["label"])

        tprint(f"   Imbalance level: {label_imbalance['imbalance_level']}")
        tprint(f"   Imbalance ratio: {label_imbalance['imbalance_ratio']:.2f}")
        tprint(f"   Total samples: {label_imbalance['total_samples']}")

        if label_imbalance["recommendations"]:
            tprint("   Recommendations:")
            for rec in label_imbalance["recommendations"]:
                tprint(f"     - {rec}")

    except Exception as e:
        tprint(f"   ❌ Label imbalance analysis failed: {e}")


def demonstrate_temporal_validation():
    """Demonstrate temporal data validation."""
    tprint("\n" + "="*60)
    tprint("🔍 DEMONSTRATING TEMPORAL VALIDATION")
    tprint("="*60)

    # Create sample data
    klines_df, features_df, labels_df = create_sample_data()

    # Initialize orchestrator
    orchestrator = UnifiedQualityOrchestrator()

    # Validate temporal aspects of klines data
    tprint("\n📊 Validating temporal aspects of klines data...")
    try:
        temporal_validation = orchestrator.validate_temporal_data(klines_df, "timestamp")

        tprint(f"   Temporal validation passed: {temporal_validation.passed}")
        tprint(f"   Issues found: {len(temporal_validation.issues)}")
        tprint(f"   Warnings: {len(temporal_validation.warnings)}")

        if temporal_validation.metrics:
            tprint("   Temporal metrics:")
            for key, value in temporal_validation.metrics.items():
                if key in ["max_gap", "min_gap", "mean_gap"]:
                    tprint(f"     - {key}: {value}")

        if temporal_validation.issues:
            tprint("   Issues:")
            for issue in temporal_validation.issues:
                tprint(f"     - {issue}")

    except Exception as e:
        tprint(f"   ❌ Temporal validation failed: {e}")


def demonstrate_comprehensive_report():
    """Demonstrate comprehensive report generation."""
    tprint("\n" + "="*60)
    tprint("🔍 DEMONSTRATING COMPREHENSIVE REPORT GENERATION")
    tprint("="*60)

    # Create sample data
    klines_df, features_df, labels_df = create_sample_data()

    # Initialize orchestrator
    orchestrator = UnifiedQualityOrchestrator()

    # Generate comprehensive report for features data
    tprint("\n📊 Generating comprehensive quality report...")
    try:
        report = orchestrator.generate_comprehensive_report(features_df, "Sample Features Dataset")

        # Display summary
        summary = report.get("summary", {})
        tprint(f"   Overall Quality: {summary.get('overall_quality', 'unknown').upper()}")
        tprint(f"   Critical Issues: {summary.get('critical_issues', 0)}")
        tprint(f"   Recommendations: {len(summary.get('recommendations', []))}")

        if summary.get("recommendations"):
            tprint("   Top Recommendations:")
            for rec in summary["recommendations"][:3]:
                tprint(f"     - {rec}")

        # Save report
        output_file = orchestrator.save_report(report, "example_quality_report.json")
        tprint(f"   ✅ Report saved to: {output_file}")

        # Display some detailed metrics
        if report.get("quality_validation"):
            quality_val = report["quality_validation"]
            tprint("   Quality Metrics:")
            tprint(f"     - Rows: {quality_val.get('metrics', {}).get('rows', 'N/A')}")
            tprint(f"     - Columns: {quality_val.get('metrics', {}).get('columns', 'N/A')}")
            tprint(f"     - Memory: {quality_val.get('metrics', {}).get('memory_mb', 0):.2f} MB")

    except Exception as e:
        tprint(f"   ❌ Comprehensive report generation failed: {e}")


def demonstrate_custom_thresholds():
    """Demonstrate custom threshold usage."""
    tprint("\n" + "="*60)
    tprint("🔍 DEMONSTRATING CUSTOM THRESHOLDS")
    tprint("="*60)

    # Create sample data
    klines_df, features_df, labels_df = create_sample_data()

    # Create custom thresholds
    custom_thresholds = QualityThresholds(
        max_nan_ratio=0.1,  # Allow up to 10% NaN values
        max_infinite_count=5,  # Allow up to 5 infinite values
        max_correlation_threshold=0.8,  # Lower correlation threshold
        vif_threshold=3.0,  # Lower VIF threshold
    )

    # Initialize orchestrator with custom thresholds
    orchestrator = UnifiedQualityOrchestrator(custom_thresholds)

    tprint("   Custom thresholds applied:")
    tprint(f"     - Max NaN ratio: {custom_thresholds.max_nan_ratio:.1%}")
    tprint(f"     - Max infinite count: {custom_thresholds.max_infinite_count}")
    tprint(f"     - Max correlation threshold: {custom_thresholds.max_correlation_threshold}")
    tprint(f"     - VIF threshold: {custom_thresholds.vif_threshold}")

    # Validate with custom thresholds
    tprint("\n📊 Validating with custom thresholds...")
    quality_result = orchestrator.validate_dataframe_quality(features_df, "Features with Custom Thresholds")

    tprint(f"   Quality check passed: {quality_result.passed}")
    tprint(f"   Issues found: {len(quality_result.issues)}")
    tprint(f"   Warnings: {len(quality_result.warnings)}")


def demonstrate_directory_analysis():
    """Demonstrate directory analysis capabilities."""
    tprint("\n" + "="*60)
    tprint("🔍 DEMONSTRATING DIRECTORY ANALYSIS")
    tprint("="*60)

    # Create sample data files in a temporary directory
    import tempfile

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create sample data files
        tprint(f"🔧 Creating sample data files in temporary directory: {temp_path}")

        # Create sample klines data
        klines_df, features_df, labels_df = create_sample_data()

        # Save files
        klines_file = temp_path / "klines_sample.csv"
        features_file = temp_path / "features_sample.csv"
        labels_file = temp_path / "labels_sample.json"

        klines_df.to_csv(klines_file, index=False)
        features_df.to_csv(features_file, index=False)
        labels_df.to_json(labels_file, orient="records")

        tprint("✅ Created sample files:")
        tprint(f"   - {klines_file.name}")
        tprint(f"   - {features_file.name}")
        tprint(f"   - {labels_file.name}")

        # Initialize orchestrator
        orchestrator = UnifiedQualityOrchestrator()

        # Quick directory scan
        tprint("\n📊 Performing quick directory scan...")
        try:
            scan_summary = orchestrator.get_directory_summary(str(temp_path), recursive=True)

            tprint(f"   Directory: {scan_summary['directory_path']}")
            tprint(f"   Total data files: {scan_summary['total_files']}")
            tprint(f"   Total size: {scan_summary['total_size_mb']:.2f} MB")
            tprint("   File types:")
            for file_type, info in scan_summary["file_types"].items():
                tprint(f"     - {file_type}: {info['count']} files ({info['total_size'] / (1024*1024):.2f} MB)")

        except Exception as e:
            tprint(f"   ❌ Quick scan failed: {e}")

        # Full directory analysis
        tprint("\n📊 Performing full directory analysis...")
        try:
            directory_report = orchestrator.analyze_directory(str(temp_path), recursive=True)

            if "error" in directory_report:
                tprint(f"   ❌ Directory analysis failed: {directory_report['error']}")
            else:
                summary = directory_report.get("summary", {})
                tprint(f"   Total files: {summary['total_files']}")
                tprint(f"   Successful analyses: {summary['successful_analyses']}")
                tprint(f"   Failed analyses: {summary['failed_analyses']}")
                tprint(f"   Success rate: {summary['success_rate']:.1%}")
                tprint(f"   Overall Quality: {summary['overall_quality'].upper()}")
                tprint(f"   Critical Issues Total: {summary['critical_issues_total']}")

                if summary.get("quality_distribution"):
                    tprint("   Quality Distribution:")
                    for quality, count in summary["quality_distribution"].items():
                        tprint(f"     - {quality.capitalize()}: {count} files")

                # Save directory report
                output_file = orchestrator.save_report(directory_report, "example_directory_report.json")
                tprint(f"   ✅ Directory report saved to: {output_file}")

        except Exception as e:
            tprint(f"   ❌ Full directory analysis failed: {e}")


def demonstrate_batch_analysis():
    """Demonstrate batch file analysis capabilities."""
    tprint("\n" + "="*60)
    tprint("🔍 DEMONSTRATING BATCH FILE ANALYSIS")
    tprint("="*60)

    # Create sample data files
    klines_df, features_df, labels_df = create_sample_data()

    # Save files to current directory for demonstration
    klines_file = "temp_klines_sample.csv"
    features_file = "temp_features_sample.csv"
    labels_file = "temp_labels_sample.json"

    klines_df.to_csv(klines_file, index=False)
    features_df.to_csv(features_file, index=False)
    labels_df.to_json(labels_file, orient="records")

    tprint("🔧 Created temporary sample files:")
    tprint(f"   - {klines_file}")
    tprint(f"   - {features_file}")
    tprint(f"   - {labels_file}")

    # Initialize orchestrator
    orchestrator = UnifiedQualityOrchestrator()

    # Batch analysis
    tprint("\n📊 Performing batch file analysis...")
    try:
        file_paths = [klines_file, features_file, labels_file]
        batch_report = orchestrator.analyze_file_batch(file_paths)

        summary = batch_report.get("summary", {})
        tprint(f"   Total files: {summary['total_files']}")
        tprint(f"   Successful analyses: {summary['successful_analyses']}")
        tprint(f"   Failed analyses: {summary['failed_analyses']}")
        tprint(f"   Success rate: {summary['success_rate']:.1%}")
        tprint(f"   Overall Quality: {summary['overall_quality'].upper()}")
        tprint(f"   Critical Issues Total: {summary['critical_issues_total']}")

        # Show individual file results
        tprint("\n   Individual file results:")
        for file_path, result in batch_report["file_results"].items():
            if "error" in result:
                tprint(f"     ❌ {Path(file_path).name}: {result['error']}")
            else:
                file_summary = result.get("summary", {})
                quality = file_summary.get("overall_quality", "unknown")
                issues = file_summary.get("critical_issues", 0)
                tprint(f"     ✅ {Path(file_path).name}: {quality.upper()} ({issues} critical issues)")

        # Save batch report
        output_file = orchestrator.save_report(batch_report, "example_batch_report.json")
        tprint(f"\n   ✅ Batch report saved to: {output_file}")

    except Exception as e:
        tprint(f"   ❌ Batch analysis failed: {e}")

    # Clean up temporary files
    try:
        os.remove(klines_file)
        os.remove(features_file)
        os.remove(labels_file)
        tprint("   🧹 Cleaned up temporary files")
    except:
        pass


def main():
    """Main demonstration function."""
    tprint("🚀 UNIFIED DATA QUALITY ORCHESTRATOR DEMONSTRATION")
    tprint("="*60)

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

        # Demonstrate directory analysis
        demonstrate_directory_analysis()

        # Demonstrate batch analysis
        demonstrate_batch_analysis()

        tprint("\n" + "="*60)
        tprint("🎉 DEMONSTRATION COMPLETED SUCCESSFULLY!")
        tprint("="*60)
        tprint("\n💡 Next steps:")
        tprint("   1. Review the generated quality report: example_quality_report.json")
        tprint("   2. Try running the orchestrator on your own data")
        tprint("   3. Customize thresholds for your specific use case")
        tprint("   4. Integrate the orchestrator into your data pipeline")

    except Exception as e:
        tprint(f"\n❌ Demonstration failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
