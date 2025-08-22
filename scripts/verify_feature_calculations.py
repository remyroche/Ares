#!/usr/bin/env python3
"""
Feature Calculation Verification Script
Verifies feature calculations and investigates specific issues from the logs.
"""

from pathlib import Path
from src.utils.logger import system_logger
from typing import Any
import sys
import traceback
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))


class FeatureCalculationVerifier:
    """Verifies feature calculations and identifies specific issues."""

    def __init__(self):
        self.logger = system_logger.getChild("FeatureCalculationVerifier")

    def verify_momentum_features(self, data: pd.DataFrame) -> dict[str, Any]:
        """Verify momentum feature calculations."""
        self.logger.info("🔍 Verifying momentum features...")

        momentum_features = [
            "momentum_5",
            "momentum_10",
            "momentum_20",
            "momentum_strength",
            "1m_volume_momentum",
            "5m_volume_momentum",
            "15m_volume_momentum",
            "30m_volume_momentum",
            "1m_momentum_1m",
            "5m_momentum_5m",
            "15m_momentum_15m",
            "30m_momentum_30m",
        ]

        results = {
            "available_features": [],
            "missing_features": [],
            "calculation_issues": [],
            "correlation_analysis": {},
            "variance_analysis": {},
        }

        # Check availability
        for feature in momentum_features:
            if feature in data.columns:
                results["available_features"].append(feature)
            else:
                results["missing_features"].append(feature)

        if not results["available_features"]:
            return results

        momentum_data = data[results["available_features"]]

        # Verify calculations
        issues = self._verify_momentum_calculations(momentum_data)
        results["calculation_issues"] = issues

        # Analyze correlations
        results["correlation_analysis"] = self._analyze_correlations(momentum_data)

        # Analyze variance
        results["variance_analysis"] = self._analyze_variance(momentum_data)

        return results

    def verify_volatility_features(self, data: pd.DataFrame) -> dict[str, Any]:
        """Verify volatility feature calculations."""
        self.logger.info("🔍 Verifying volatility features...")

        volatility_features = [
            "realized_volatility",
            "parkinson_volatility",
            "garman_klass_volatility",
            "volatility_regime",
            "volatility_percentile",
            "volume_volatility",
            "adaptive_atr",
            "1m_price_volatility",
            "1m_volume_volatility",
            "5m_price_volatility",
            "5m_volume_volatility",
            "15m_price_volatility",
            "15m_volume_volatility",
            "30m_price_volatility",
            "30m_volume_volatility",
        ]

        results = {
            "available_features": [],
            "missing_features": [],
            "calculation_issues": [],
            "correlation_analysis": {},
            "variance_analysis": {},
        }

        # Check availability
        for feature in volatility_features:
            if feature in data.columns:
                results["available_features"].append(feature)
            else:
                results["missing_features"].append(feature)

        if not results["available_features"]:
            return results

        volatility_data = data[results["available_features"]]

        # Verify calculations
        issues = self._verify_volatility_calculations(volatility_data)
        results["calculation_issues"] = issues

        # Analyze correlations
        results["correlation_analysis"] = self._analyze_correlations(volatility_data)

        # Analyze variance
        results["variance_analysis"] = self._analyze_variance(volatility_data)

        return results

    def verify_liquidity_features(self, data: pd.DataFrame) -> dict[str, Any]:
        """Verify liquidity feature calculations."""
        self.logger.info("🔍 Verifying liquidity features...")

        liquidity_features = [
            "amihud_illiquidity",
            "bid_ask_spread",
            "volume_profile",
            "trade_size_distribution",
            "market_impact",
            "liquidity_score",
            "1m_liquidity_score",
            "5m_liquidity_score",
            "15m_liquidity_score",
            "30m_liquidity_score",
        ]

        results = {
            "available_features": [],
            "missing_features": [],
            "calculation_issues": [],
            "correlation_analysis": {},
            "variance_analysis": {},
        }

        # Check availability
        for feature in liquidity_features:
            if feature in data.columns:
                results["available_features"].append(feature)
            else:
                results["missing_features"].append(feature)

        if not results["available_features"]:
            return results

        liquidity_data = data[results["available_features"]]

        # Verify calculations
        issues = self._verify_liquidity_calculations(liquidity_data)
        results["calculation_issues"] = issues

        # Analyze correlations
        results["correlation_analysis"] = self._analyze_correlations(liquidity_data)

        # Analyze variance
        results["variance_analysis"] = self._analyze_variance(liquidity_data)

        return results

    def verify_technical_features(self, data: pd.DataFrame) -> dict[str, Any]:
        """Verify technical indicator calculations."""
        self.logger.info("🔍 Verifying technical features...")

        technical_features = [
            "rsi",
            "macd",
            "bollinger_bands_upper",
            "bollinger_bands_lower",
            "stochastic_k",
            "stochastic_d",
            "williams_r",
            "cci",
            "adx",
            "1m_rsi",
            "5m_rsi",
            "15m_rsi",
            "30m_rsi",
        ]

        results = {
            "available_features": [],
            "missing_features": [],
            "calculation_issues": [],
            "correlation_analysis": {},
            "variance_analysis": {},
        }

        # Check availability
        for feature in technical_features:
            if feature in data.columns:
                results["available_features"].append(feature)
            else:
                results["missing_features"].append(feature)

        if not results["available_features"]:
            return results

        technical_data = data[results["available_features"]]

        # Verify calculations
        issues = self._verify_technical_calculations(technical_data)
        results["calculation_issues"] = issues

        # Analyze correlations
        results["correlation_analysis"] = self._analyze_correlations(technical_data)

        # Analyze variance
        results["variance_analysis"] = self._analyze_variance(technical_data)

        return results

    def _verify_momentum_calculations(self, data: pd.DataFrame) -> list[str]:
        """Verify momentum feature calculations."""
        issues = []

        # Check for NaN values
        nan_counts = data.isna().sum()
        for col, count in nan_counts.items():
            if count > 0:
                issues.append(f"NaN values in {col}: {count}")

        # Check for infinite values
        inf_counts = np.isinf(data.select_dtypes(include=[np.number])).sum()
        for col, count in inf_counts.items():
            if count > 0:
                issues.append(f"Infinite values in {col}: {count}")

        # Check for constant values
        for col in data.columns:
            if data[col].nunique() == 1:
                issues.append(f"Constant values in {col}")

        return issues

    def _verify_volatility_calculations(self, data: pd.DataFrame) -> list[str]:
        """Verify volatility feature calculations."""
        issues = []

        # Check for NaN values
        nan_counts = data.isna().sum()
        for col, count in nan_counts.items():
            if count > 0:
                issues.append(f"NaN values in {col}: {count}")

        # Check for infinite values
        inf_counts = np.isinf(data.select_dtypes(include=[np.number])).sum()
        for col, count in inf_counts.items():
            if count > 0:
                issues.append(f"Infinite values in {col}: {count}")

        # Check for negative volatility (should be positive)
        for col in data.columns:
            if "volatility" in col.lower():
                negative_count = (data[col] < 0).sum()
                if negative_count > 0:
                    issues.append(f"Negative values in {col}: {negative_count}")

        return issues

    def _verify_liquidity_calculations(self, data: pd.DataFrame) -> list[str]:
        """Verify liquidity feature calculations."""
        issues = []

        # Check for NaN values
        nan_counts = data.isna().sum()
        for col, count in nan_counts.items():
            if count > 0:
                issues.append(f"NaN values in {col}: {count}")

        # Check for infinite values
        inf_counts = np.isinf(data.select_dtypes(include=[np.number])).sum()
        for col, count in inf_counts.items():
            if count > 0:
                issues.append(f"Infinite values in {col}: {count}")

        return issues

    def _verify_technical_calculations(self, data: pd.DataFrame) -> list[str]:
        """Verify technical indicator calculations."""
        issues = []

        # Check for NaN values
        nan_counts = data.isna().sum()
        for col, count in nan_counts.items():
            if count > 0:
                issues.append(f"NaN values in {col}: {count}")

        # Check for infinite values
        inf_counts = np.isinf(data.select_dtypes(include=[np.number])).sum()
        for col, count in inf_counts.items():
            if count > 0:
                issues.append(f"Infinite values in {col}: {count}")

        # Check RSI bounds (should be 0-100)
        rsi_columns = [col for col in data.columns if "rsi" in col.lower()]
        for col in rsi_columns:
            out_of_bounds = ((data[col] < 0) | (data[col] > 100)).sum()
            if out_of_bounds > 0:
                issues.append(f"RSI out of bounds in {col}: {out_of_bounds}")

        return issues

    def _analyze_correlations(self, data: pd.DataFrame) -> dict[str, Any]:
        """Analyze correlations between features."""
        if len(data.columns) < 2:
            return {}

        # Calculate correlation matrix
        corr_matrix = data.corr()

        # Find highly correlated pairs
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                corr_value = corr_matrix.iloc[i, j]
                if abs(corr_value) > 0.8:
                    high_corr_pairs.append({
                        "feature1": corr_matrix.columns[i],
                        "feature2": corr_matrix.columns[j],
                        "correlation": corr_value
                    })

        return {
            "correlation_matrix": corr_matrix.to_dict(),
            "high_correlation_pairs": high_corr_pairs,
            "max_correlation": corr_matrix.max().max(),
            "min_correlation": corr_matrix.min().min(),
        }

    def _analyze_variance(self, data: pd.DataFrame) -> dict[str, Any]:
        """Analyze variance of features."""
        variance_stats = {}
        for col in data.columns:
            if data[col].dtype in [np.number]:
                variance_stats[col] = {
                    "variance": float(data[col].var()),
                    "std": float(data[col].std()),
                    "mean": float(data[col].mean()),
                    "min": float(data[col].min()),
                    "max": float(data[col].max()),
                    "zero_variance": bool(data[col].var() == 0),
                }

        return variance_stats

    def comprehensive_verification(self, data: pd.DataFrame) -> dict[str, Any]:
        """Perform comprehensive feature verification."""
        self.logger.info("🚀 Starting comprehensive feature verification...")

        results = {
            "momentum": self.verify_momentum_features(data),
            "volatility": self.verify_volatility_features(data),
            "liquidity": self.verify_liquidity_features(data),
            "technical": self.verify_technical_features(data),
            "summary": {},
        }

        # Generate summary
        total_features = len(data.columns)
        total_issues = 0
        feature_categories = {}

        for category, category_results in results.items():
            if category == "summary":
                continue

            available = len(category_results["available_features"])
            missing = len(category_results["missing_features"])
            issues = len(category_results["calculation_issues"])

            feature_categories[category] = {
                "available": available,
                "missing": missing,
                "issues": issues,
            }

            total_issues += issues

        results["summary"] = {
            "total_features": total_features,
            "total_issues": total_issues,
            "feature_categories": feature_categories,
            "verification_score": max(0, 100 - (total_issues * 10)),
        }

        return results


def main():
    """Main function for feature verification."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Feature Calculation Verification Tool"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to the data file (CSV, Parquet, or PKL)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="feature_verification_reports",
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

    # Initialize verifier
    verifier = FeatureCalculationVerifier()

    # Perform comprehensive verification
    results = verifier.comprehensive_verification(data)

    # Save results
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"feature_verification_report_{timestamp}.json"

    import json
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"✅ Feature verification report saved to: {output_file}")

    # Print summary
    summary = results["summary"]
    print(f"\n📊 Feature Verification Summary:")
    print(f"Total features: {summary['total_features']}")
    print(f"Total issues: {summary['total_issues']}")
    print(f"Verification score: {summary['verification_score']}/100")

    # Print category breakdown
    print(f"\n📋 Feature Categories:")
    for category, stats in summary["feature_categories"].items():
        print(f"  {category.title()}:")
        print(f"    Available: {stats['available']}")
        print(f"    Missing: {stats['missing']}")
        print(f"    Issues: {stats['issues']}")

    # Print critical issues
    if summary["total_issues"] > 0:
        print(f"\n🚨 Critical Issues Found:")
        for category, category_results in results.items():
            if category == "summary":
                continue
            if category_results["calculation_issues"]:
                print(f"  {category.title()}:")
                for issue in category_results["calculation_issues"][:3]:  # Show top 3
                    print(f"    - {issue}")


if __name__ == "__main__":
    main()
