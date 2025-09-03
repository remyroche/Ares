#!/usr/bin/env python3
"""
Debug script to analyze low variance features in the autoencoder feature generator.
This script helps identify which features have low variance and whether this is normal.
"""

import logging
from typing import Any

import numpy as np
import pandas as pd

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger=logging.getLogger(__name__)


def analyze_feature_variance(
    features_df: pd.DataFrame,
    std_threshold: float = 1e-6,
) -> dict[str, Any]:
    """
    Analyze feature variance and identify low variance features.

    Args:
        features_df: DataFrame with features
        std_threshold: Standard deviation threshold for low variance detection

    Returns:
        Dictionary with analysis results
    """
    results={
        "total_features": len(features_df.columns),
        "std_threshold": std_threshold, "low_variance_features": [],
        "feature_std_values": {},
        "analysis_summary": {},
    }

    # Calculate standard deviation for each feature
    per_feature_std=features_df.std(axis=0, skipna=True)

    # Identify low variance features
    low_std_cols=per_feature_std.index[per_feature_std <= std_threshold].tolist()

    results["low_variance_features"] = low_std_cols
    results["feature_std_values"] = per_feature_std.to_dict()

    # Analysis summary
    results["analysis_summary"] = {
        "total_features": len(features_df.columns),
        "low_variance_count": len(low_std_cols),
        "low_variance_percentage": (len(low_std_cols) / len(features_df.columns)) * 100,
        "min_std": per_feature_std.min(),
        "max_std": per_feature_std.max(),
        "mean_std": per_feature_std.mean(),
        "median_std": per_feature_std.median(),
    }

    return results


def check_feature_types(features_df: pd.DataFrame) -> dict[str, Any]:
    """
    Analyze feature types and patterns to understand the data better.
    """
    analysis={
        "feature_patterns": {},
        "data_types": features_df.dtypes.value_counts().to_dict(),
        "null_counts": features_df.isnull().sum().to_dict(),
        "unique_counts": {},
        "constant_features": [],
    }

    for col in features_df.columns:
        # Check for constant features
        if features_df[col].nunique() == 1:
            analysis["constant_features"].append(col)

        # Count unique values
        analysis["unique_counts"][col] = features_df[col].nunique()

        # Identify feature patterns
        if "cluster" in col.lower():
            if "intensity" in col.lower():
                pattern="intensity_cluster"
            elif "hmm" in col.lower():
                pattern="hmm_cluster"
            else:
                pattern = "cluster"
        elif "id" in col.lower():
            pattern="id_feature"
        else:
            pattern = "other"

        if pattern not in analysis["feature_patterns"]:
            analysis["feature_patterns"][pattern] = []
        analysis["feature_patterns"][pattern].append(col)

    return analysis


def generate_debug_report(
    features_df: pd.DataFrame,
    std_threshold: float = 1e-6,
) -> str:
    """
    Generate a comprehensive debug report for low variance features.
    """
    logger.info("🔍 Analyzing feature variance...")
    variance_analysis = analyze_feature_variance(features_df=features_df, std_threshold=std_threshold)

    logger.info("🔍 Analyzing feature types and patterns...")
    type_analysis = check_feature_types(features_df)

    # Generate report
    report=[]
    report.append("=" * 80)
    report.append("LOW VARIANCE FEATURES DEBUG REPORT")
    report.append("=" * 80)

    # Summary
    summary=variance_analysis["analysis_summary"]
    report.append("\n📊 SUMMARY:")
    report.append(f"   Total features: {summary['total_features']}")
    report.append(
        f"   Low variance features: {summary['low_variance_count']} ({summary['low_variance_percentage']:.1f}%)",
    )
    report.append(f"   Standard deviation threshold: {std_threshold}")
    report.append(f"   Min std: {summary['min_std']:.2e}")
    report.append(f"   Max std: {summary['max_std']:.2e}")
    report.append(f"   Mean std: {summary['mean_std']:.2e}")
    report.append(f"   Median std: {summary['median_std']:.2e}")

    # Low variance features list
    if variance_analysis["low_variance_features"]:
        report.append(f"\n⚠️ LOW VARIANCE FEATURES (std <= {std_threshold}):")
        for i, feature in enumerate(variance_analysis["low_variance_features"], 1):
            std_val=variance_analysis["feature_std_values"][feature]
            unique_count = type_analysis["unique_counts"][feature]
            report.append(f"   {i:2d}. {feature}")
            report.append(f"       std: {std_val:.2e}, unique values: {unique_count}")

    # Feature patterns
    report.append("\n🔍 FEATURE PATTERNS:")
    for pattern, features in type_analysis["feature_patterns"].items():
        report.append(f"   {pattern}: {len(features)} features")
        if len(features) <= 10:
            report.append(f"       {', '.join(features)}")

    # Constant features
    if type_analysis["constant_features"]:
        report.append("\n🚨 CONSTANT FEATURES (no variance):")
        for feature in type_analysis["constant_features"]:
            report.append(f"   - {feature}")

    # Assessment
    report.append("\n📋 ASSESSMENT:")
    low_var_pct=summary["low_variance_percentage"]

    if low_var_pct == 100:
        report.append("   🚨 CRITICAL: ALL features have low variance!")
        report.append("   This indicates a serious data pipeline issue.")
        report.append("   Possible causes:")
        report.append("   - Features are not being calculated correctly")
        report.append("   - Data preprocessing is removing all variance")
        report.append("   - Features are constant/static")
    elif low_var_pct > 50:
        report.append("   ⚠️ WARNING: More than 50% of features have low variance")
        report.append("   This suggests potential issues with feature engineering")
    elif low_var_pct > 20:
        report.append("   ⚠️ CAUTION: More than 20% of features have low variance")
        report.append(
            "   This might be normal for some feature types (e.g., cluster IDs)",
        )
    else:
        report.append("   ✅ Normal: Low percentage of low variance features")

    # Recommendations
    report.append("\n💡 RECOMMENDATIONS:")
    if low_var_pct > 50:
        report.append("   1. Check feature engineering pipeline")
        report.append("   2. Verify data preprocessing steps")
        report.append("   3. Investigate if features are being calculated correctly")
        report.append("   4. Consider increasing std_threshold if appropriate")
    else:
        report.append("   1. This appears to be normal behavior")
        report.append(
            "   2. Low variance features are likely cluster IDs or categorical features",
        )
        report.append(
            "   3. Consider if these features should be excluded from autoencoder training",
        )

    report.append("\n" + "=" * 80)

    return "\n".join(report)


def main():
    """
    Main function to demonstrate usage.
    """
    logger.info("🚀 Starting low variance features debug analysis...")

    # Example usage - you would replace this with your actual data
    logger.info("📝 To use this script with your data:")
    logger.info("   1. Load your features DataFrame")
    logger.info("   2. Call generate_debug_report(features_df)")
    logger.info("   3. Review the output to understand the issue")

    # Example with dummy data to show the structure
    logger.info("\n📊 Example analysis structure:")

    # Create dummy data similar to what we saw in the logs
    dummy_features=[
        "hmm_combination_id",
        "hmm_composite_cluster_id",
        "intensity_cluster_0",
        "intensity_cluster_1",
        "intensity_cluster_2",
        "intensity_cluster_3",
        "intensity_cluster_4",
        "intensity_cluster_5",
        "intensity_cluster_6",
        "intensity_cluster_7",
        "intensity_cluster_8",
        "intensity_cluster_9",
        "intensity_cluster_10",
        "intensity_cluster_11",
        "intensity_cluster_12",
        "intensity_cluster_13",
        "intensity_cluster_14",
        "intensity_cluster_15",
        "intensity_cluster_16",
        "intensity_cluster_17",
        "intensity_cluster_18",
        "intensity_cluster_19",
    ]

    # Create dummy DataFrame with very low variance (simulating the issue)
    np.random.seed(42)
    n_rows=1000
    dummy_df = pd.DataFrame()

    for feature in dummy_features:
        if "cluster" in feature:
            # Simulate cluster features with very low variance
            dummy_df[feature] = np.random.choice([0, 1], size=n_rows, p=[0.99, 0.01])
        else:
            # Simulate ID features with some variance
            dummy_df[feature] = np.random.randint(0, 5, size=n_rows)

    # Generate report
    report=generate_debug_report(dummy_df)
    print(report)

    logger.info("✅ Debug analysis complete!")


if __name__== "__main__":
    main()
