#!/usr/bin/env python3
"""
Enhanced Data Quality Assessment Script

This script demonstrates how to use the VectorizedLabellingOrchestrator's
data quality assessment functionality to analyze NaN/missing data in your datasets.
Now includes comprehensive multicollinearity analysis and label imbalance detection.

Usage:
    python scripts/assess_data_quality.py --data_path /path/to/your/data --symbol ETHUSDT --exchange binance
"""

from pathlib import Path
from sklearn.linear_model import LinearRegression
from typing import Any
from utils.logger import system_logger
import argparse
import asyncio
import sys

from sklearn.impute import SimpleImputer
import numpy as np
import pandas as pd

# Add the src directory to the Python path
current_dir = Path(__file__).parent
src_dir = current_dir.parent / "src"
sys.path.insert(0, str(src_dir))

from src.utils.advanced_decorators import performance_monitor, PerformanceLevel

class EnhancedDataQualityAnalyzer:
    """
    Enhanced data quality analyzer that addresses critical issues:
    1. Multicollinearity detection and resolution
    2. Label imbalance analysis and recommendations
    3. Feature redundancy identification
    """

    def __init__(self):
        self.logger = system_logger.getChild("EnhancedDataQualityAnalyzer")

    @performance_monitor(level=PerformanceLevel.DETAILED)
    def analyze_multicollinearity(self, data: pd.DataFrame, vif_threshold: float = 5.0) -> dict[str, Any]:
        """
        Analyze multicollinearity using VIF and correlation analysis.

        Args:
            data: Input DataFrame
            vif_threshold: VIF threshold for flagging multicollinearity

        Returns:
            Dictionary with multicollinearity analysis results
        """
        self.logger.info("🔍 Analyzing multicollinearity...")

        # Remove non-numeric columns
        numeric_data = data.select_dtypes(include=[np.number])

        # Remove potential label columns
        potential_label_columns = [
            "label",
            "target",
            "y",
            "class",
            "Label",
            "Target",
            "Y",
            "Class",
        ]
        actual_label_columns = [
            col for col in numeric_data.columns if col in potential_label_columns
        ]
        if actual_label_columns:
            self.logger.warning(
                f"⚠️ Removing label columns from multicollinearity analysis: {actual_label_columns}",
            )
            numeric_data = numeric_data.drop(columns=actual_label_columns)

        # Handle NaN values
        imputer = SimpleImputer(strategy="median")
        data_imputed = pd.DataFrame(
            imputer.fit_transform(numeric_data),
            columns=numeric_data.columns,
            index=numeric_data.index,
        )

        # Calculate VIF scores
        vif_scores: dict[str, float] = {}
        high_vif_features: list[str] = []

        for i, col in enumerate(data_imputed.columns):
            other_cols = [c for c in data_imputed.columns if c != col]
            if len(other_cols) > 0:
                X = data_imputed[other_cols]
                y = data_imputed[col]

                reg = LinearRegression()
                reg.fit(X, y)

                # Calculate R-squared
                y_pred = reg.predict(X)
                ss_res = np.sum((y - y_pred) ** 2)
                ss_tot = np.sum((y - np.mean(y)) ** 2)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

                # Calculate VIF
                vif = 1 / (1 - r_squared) if r_squared < 1 else float("inf")
                vif_scores[col] = vif

                if vif > vif_threshold:
                    high_vif_features.append(col)

        # Calculate correlation matrix
        correlation_matrix = data_imputed.corr()

        # Find highly correlated pairs
        high_correlation_pairs = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i + 1, len(correlation_matrix.columns)):
                corr_value = correlation_matrix.iloc[i, j]
                if abs(corr_value) > 0.8:
                    high_correlation_pairs.append(
                        (correlation_matrix.columns[i], correlation_matrix.columns[j], corr_value)
                    )

        return {
            "vif_scores": vif_scores,
            "high_vif_features": high_vif_features,
            "correlation_matrix": correlation_matrix,
            "high_correlation_pairs": high_correlation_pairs,
            "vif_threshold": vif_threshold,
            "total_features": len(data_imputed.columns),
            "features_with_high_vif": len(high_vif_features),
            "high_correlation_pairs_count": len(high_correlation_pairs),
        }

    @performance_monitor(level=PerformanceLevel.DETAILED)
    def analyze_label_distribution(self, data: pd.DataFrame) -> dict[str, Any]:
        """
        Analyze label distribution and identify imbalance issues.

        Args:
            data: Input DataFrame with label column

        Returns:
            Dictionary with label analysis results
        """
        self.logger.info("🔍 Analyzing label distribution...")

        # Find label column
        label_columns = [
            "label",
            "target",
            "y",
            "class",
            "Label",
            "Target",
            "Y",
        ]
        label_col = None
        for col in label_columns:
            if col in data.columns:
                label_col = col
                break

        if label_col is None:
            return {"error": "No label column found"}

        labels = data[label_col]

        # Analyze label distribution
        unique_labels, counts = np.unique(labels, return_counts=True)
        label_distribution = dict(zip(unique_labels, counts, strict=False))

        # Calculate imbalance metrics
        total_samples = len(labels)
        class_ratios = {
            label: count / total_samples
            for label, count in label_distribution.items()
        }

        # Identify issues
        issues = []
        recommendations = []

        # Check for extreme imbalance
        min_class_count = min(counts)
        max_class_count = max(counts)
        imbalance_ratio = (
            max_class_count / min_class_count
            if min_class_count > 0
            else float("inf")
        )

        if min_class_count < 10:
            issues.append(f"CRITICAL: Class with only {min_class_count} samples")
            recommendations.append(
                "Consider binary classification (remove HOLD class)",
            )

        if imbalance_ratio > 100:
            issues.append(f"SEVERE: Class imbalance ratio of {imbalance_ratio:.1f}")
            recommendations.append("Use class weights or resampling techniques")

        # Check for single-class dominance
        dominant_class_ratio = max(class_ratios.values())
        if dominant_class_ratio > 0.9:
            issues.append(
                f"DOMINANT: One class represents {dominant_class_ratio:.1%} of data",
            )
            recommendations.append("Consider different labeling strategy")

        # Check for HOLD class issues
        hold_labels = [0, "HOLD", "hold"]
        hold_count = sum(label_distribution.get(label, 0) for label in hold_labels)
        if hold_count < 100:
            issues.append(f"HOLD_CLASS: Only {hold_count} HOLD samples")
            recommendations.append("Switch to binary classification (BUY vs SELL)")

        return {
            "label_distribution": label_distribution,
            "class_ratios": class_ratios,
            "total_samples": total_samples,
            "unique_classes": len(unique_labels),
            "imbalance_ratio": imbalance_ratio,
            "min_class_count": min_class_count,
            "max_class_count": max_class_count,
            "dominant_class_ratio": dominant_class_ratio,
            "issues": issues,
            "recommendations": recommendations,
        }

    @performance_monitor(level=PerformanceLevel.DETAILED)
    def analyze_feature_redundancy(self, data: pd.DataFrame) -> dict[str, Any]:
        """
        Analyze feature redundancy and identify redundant features.

        Args:
            data: Input DataFrame

        Returns:
            Dictionary with redundancy analysis results
        """
        self.logger.info("🔍 Analyzing feature redundancy...")

        # Remove non-numeric columns
        numeric_data = data.select_dtypes(include=[np.number])

        # Calculate correlation matrix
        correlation_matrix = numeric_data.corr()

        # Find redundant features (correlation > 0.95)
        redundant_features = []
        feature_groups = []

        for i in range(len(correlation_matrix.columns)):
            for j in range(i + 1, len(correlation_matrix.columns)):
                corr_value = correlation_matrix.iloc[i, j]
                if abs(corr_value) > 0.95:
                    feature1 = correlation_matrix.columns[i]
                    feature2 = correlation_matrix.columns[j]
                    redundant_features.append((feature1, feature2, corr_value))

                    # Group redundant features
                    found_group = False
                    for group in feature_groups:
                        if feature1 in group or feature2 in group:
                            if feature1 not in group:
                                group.append(feature1)
                            if feature2 not in group:
                                group.append(feature2)
                            found_group = True
                            break
                    if not found_group:
                        feature_groups.append([feature1, feature2])

        return {
            "redundant_features": redundant_features,
            "feature_groups": feature_groups,
            "total_redundant_pairs": len(redundant_features),
            "redundant_feature_groups": len(feature_groups),
        }

    @performance_monitor(level=PerformanceLevel.DETAILED)
    def comprehensive_analysis(self, data: pd.DataFrame) -> dict[str, Any]:
        """
        Perform comprehensive data quality analysis.

        Args:
            data: Input DataFrame

        Returns:
            Dictionary with comprehensive analysis results
        """
        self.logger.info("🚀 Starting comprehensive data quality analysis...")

        results = {
            "multicollinearity": self.analyze_multicollinearity(data),
            "label_distribution": self.analyze_label_distribution(data),
            "feature_redundancy": self.analyze_feature_redundancy(data),
        }

        # Generate summary
        summary = {
            "total_features": len(data.columns),
            "total_samples": len(data),
            "critical_issues": [],
            "warnings": [],
            "recommendations": [],
        }

        # Check for critical issues
        if results["multicollinearity"]["features_with_high_vif"] > 10:
            summary["critical_issues"].append(
                f"High multicollinearity: {results['multicollinearity']['features_with_high_vif']} features with VIF > 5"
            )

        if "issues" in results["label_distribution"]:
            summary["critical_issues"].extend(results["label_distribution"]["issues"])

        if results["feature_redundancy"]["total_redundant_pairs"] > 20:
            summary["warnings"].append(
                f"Feature redundancy: {results['feature_redundancy']['total_redundant_pairs']} highly correlated feature pairs"
            )

        # Add recommendations
        if "recommendations" in results["label_distribution"]:
            summary["recommendations"].extend(results["label_distribution"]["recommendations"])

        if results["multicollinearity"]["features_with_high_vif"] > 0:
            summary["recommendations"].append(
                "Consider removing or combining highly correlated features"
            )

        results["summary"] = summary
        return results


async def main():
    """Main function for data quality assessment."""
    parser = argparse.ArgumentParser(
        description="Enhanced Data Quality Assessment Tool"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to the data file (CSV, Parquet, or PKL)",
    )
    parser.add_argument(
        "--symbol",
        type=str,
        required=True,
        help="Trading symbol (e.g., ETHUSDT)",
    )
    parser.add_argument(
        "--exchange",
        type=str,
        required=True,
        help="Exchange name (e.g., binance)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data_quality_reports",
        help="Output directory for reports",
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Load data
    data_path = Path(args.data_path)
    if data_path.suffix == ".csv":
        data = pd.read_csv(data_path)
    elif data_path.suffix == ".parquet":
        data = pd.read_parquet(data_path)
    elif data_path.suffix == ".pkl":
        data = pd.read_pickle(data_path)
    else:
        print(f"Unsupported file format: {data_path.suffix}")
        return

    # Initialize analyzer
    analyzer = EnhancedDataQualityAnalyzer()

    # Perform comprehensive analysis
    results = analyzer.comprehensive_analysis(data)

    # Save results
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"data_quality_report_{args.symbol}_{args.exchange}_{timestamp}.json"

    # Convert numpy types to native Python types for JSON serialization
    def convert_numpy_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict()
        elif isinstance(obj, pd.Series):
            return obj.to_dict()
        return obj

    import json
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=convert_numpy_types)

    print(f"✅ Data quality report saved to: {output_file}")

    # Print summary
    summary = results["summary"]
    print(f"\n📊 Data Quality Summary:")
    print(f"Total features: {summary['total_features']}")
    print(f"Total samples: {summary['total_samples']}")
    print(f"Critical issues: {len(summary['critical_issues'])}")
    print(f"Warnings: {len(summary['warnings'])}")

    if summary["critical_issues"]:
        print(f"\n🚨 Critical Issues:")
        for issue in summary["critical_issues"]:
            print(f"  - {issue}")

    if summary["warnings"]:
        print(f"\n⚠️ Warnings:")
        for warning in summary["warnings"]:
            print(f"  - {warning}")

    if summary["recommendations"]:
        print(f"\n💡 Recommendations:")
        for rec in summary["recommendations"]:
            print(f"  - {rec}")


if __name__ == "__main__":
    asyncio.run(main())
