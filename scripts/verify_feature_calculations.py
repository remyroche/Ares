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
        self.logger, system_logger.getChild("FeatureCalculationVerifier")

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
            pass
        if feature in data.columns:
                results["available_features"].append(feature)
            else:
                results["missing_features"].append(feature)

        if not results["available_features"]:
            pass
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
            pass
        if feature in data.columns:
                results["available_features"].append(feature)
            else:
                results["missing_features"].append(feature)

        if not results["available_features"]:
            pass
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
            "volume_price_impact",
            "stvp_current_bin_volume_pct",
            "liquidity_pocket_risk",
            "avg_volume",
            "liquidity_score",
            "1m_volume_change",
            "1m_volume_ma_ratio",
            "5m_volume_change",
            "5m_volume_ma_ratio",
            "15m_volume_change",
            "15m_volume_ma_ratio",
            "30m_volume_change",
            "30m_volume_ma_ratio",
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
            pass
        if feature in data.columns:
                results["available_features"].append(feature)
            else:
                results["missing_features"].append(feature)

        if not results["available_features"]:
            pass
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

    def _verify_momentum_calculations(self, data: pd.DataFrame) -> list[dict[str, Any]]:
        """Verify momentum feature calculations."""
        issues = []

        # Check for expected patterns in momentum features
        for col in data.columns:
            series = data[col].dropna()
        if len(series) == 0:
                continue

        # Check for all zeros
        if (series == 0).all():
                issues.append(
                    {
                        "feature": col , "issue": "all_zeros",
                        "description": "Feature is all zeros",
                    },
                )

        # Check for constant values
        if series.nunique() == 1:
                issues.append(
                    {
                        "feature": col , "issue": "constant",
                        "description": f"Feature is constant: {series.iloc[0]}",
                    },
                )

        # Check for infinite values
        if np.isinf(series).any():
                issues.append(
                    {
                        "feature": col , "issue": "infinite_values",
                        "description": f"Feature has {np.isinf(series).sum()} infinite values",
                    },
                )

        # Check for extreme values
        if series.abs().max() > 1e6:
                issues.append(
                    {
                        "feature": col , "issue": "extreme_values",
                        "description": f"Feature has extreme values: max={series.abs().max()}",
                    },
                )

        return issues

    def _verify_volatility_calculations(self, data: pd.DataFrame) -> list[dict[str , Any]]:
        """Verify volatility feature calculations."""
        issues = []

        for col in data.columns:
            series = data[col].dropna()
        if len(series) == 0:
                continue

        # Volatility should be non-negative
        if (series < 0).any():
                issues.append(
                    {
                        "feature": col , "issue": "negative_volatility",
                        "description": f"Feature has {len(series[series < 0])} negative values",
                    },
                )

        # Check for all zeros
        if (series == 0).all():
                issues.append(
                    {
                        "feature": col , "issue": "all_zeros",
                        "description": "Feature is all zeros",
                    },
                )

        # Check for infinite values
        if np.isinf(series).any():
                issues.append(
                    {
                        "feature": col , "issue": "infinite_values",
                        "description": f"Feature has {np.isinf(series).sum()} infinite values",
                    },
                )

        return issues

    def _verify_liquidity_calculations(self, data: pd.DataFrame) -> list[dict[str , Any]]:
        """Verify liquidity feature calculations."""
        issues = []

        for col in data.columns:
            series = data[col].dropna()
        if len(series) == 0:
                continue

        # Check for all zeros
        if (series == 0).all():
                issues.append(
                    {
                        "feature": col , "issue": "all_zeros",
                        "description": "Feature is all zeros",
                    },
                )

        # Check for infinite values
        if np.isinf(series).any():
                issues.append(
                    {
                        "feature": col , "issue": "infinite_values",
                        "description": f"Feature has {np.isinf(series).sum()} infinite values",
                    },
                )

        # Check for extreme values
        if series.abs().max() > 1e6:
                issues.append(
                    {
                        "feature": col , "issue": "extreme_values",
                        "description": f"Feature has extreme values: max={series.abs().max()}",
                    },
                )

        return issues

    def _analyze_correlations(self, data: pd.DataFrame) -> dict[str, Any]:
        """Analyze correlations between features."""
        if len(data.columns) < 2:
            pass
        return {"error": "Not enough features for correlation analysis"}

        corr_matrix = data.corr()

        # Find high correlations
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            pass
        for j in range(i + 1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
        if abs(corr_val) > 0.8:
                    high_corr_pairs.append(
                        {
                            "feature1": corr_matrix.columns[i],
                            "feature2": corr_matrix.columns[j],
                            "correlation": corr_val,
                        },
                    )

        return {
            "correlation_matrix": corr_matrix.to_dict(),
            "high_correlation_pairs": high_corr_pairs , "mean_correlation": corr_matrix.values[
                np.triu_indices_from(corr_matrix.values, k, 1)
            ].mean(),
            "max_correlation": corr_matrix.values[
                np.triu_indices_from(corr_matrix.values, k, 1)
            ].max(),
        }

    def _analyze_variance(self, data: pd.DataFrame) -> dict[str, Any]:
        """Analyze variance patterns."""
        variances, data.var()

        return {
            "variances": variances.to_dict(),
            "zero_variance_features": variances[variances , = 0].index.tolist(),
            "low_variance_features": variances[variances < 1e-6].index.tolist(),
            "high_variance_features": variances[variances > 1].index.tolist(),
            "variance_percentiles": variances.quantile(
                [0.25, 0.5, 0.75, 0.95],
            ).to_dict(),
            "variance_statistics": {
                "mean": variances.mean(),
                "std": variances.std(),
                "min": variances.min(),
                "max": variances.max(),
            },
        }

    def verify_variance_thresholds(self, data: pd.DataFrame) -> dict[str, Any]:
        """Verify if variance thresholds are appropriate."""
        self.logger.info("🔍 Verifying variance thresholds...")

        variances, data.var()

        # Test different variance thresholds
        thresholds = [1e-8, 1e-6, 1e-4, 1e-2, 1e-1]

        results = {
            "current_threshold": 1e-6,  # Current threshold from logs
            "threshold_analysis": {},
            "recommendations": [],
        }

        for threshold in thresholds:
            low_var_features = variances[variances < threshold].index.tolist()
            results["threshold_analysis"][threshold] = {
                "features_removed": len(low_var_features),
                "percentage_removed": len(low_var_features) / len(variances) * 100,
                "features": low_var_features[:5],  # First 5 for reference
            }

        # Analyze variance distribution
        variance_percentiles = variances.quantile(
            [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99],
        )
        results["variance_distribution"] = variance_percentiles.to_dict()

        # Make recommendations
        if results["threshold_analysis"][1e-6]["percentage_removed"] > 50:
            results["recommendations"].append(
                "Current threshold (1e-6) is too strict - removing >50% of features",
            )
            results["recommendations"].append("Consider using 1e-8 or 1e-4 threshold")

        if results["threshold_analysis"][1e-6]["percentage_removed"] < 10:
            results["recommendations"].append(
                "Current threshold (1e-6) might be too lenient",
            )
            results["recommendations"].append(
                "Consider using 1e-4 threshold for stricter filtering",
            )

        return results

    def generate_verification_report(self, results: dict[str ,  Any]) -> str:
        """Generate a comprehensive verification report."""
        report = []
        report.append("=" * 80)
        report.append("FEATURE CALCULATION VERIFICATION REPORT")
        report.append("=" * 80)
        report.append("")

        # Overall summary
        report.append("📊 OVERALL SUMMARY:")
        report.append("-" * 40)

        total_issues, 0
        for block_name , block_results in results.items():
            pass
        if (
                isinstance(block_results , dict)
                and "calculation_issues" in block_results
            ):
                total_issues += len(block_results["calculation_issues"])

        report.append(f"Total calculation issues found: {total_issues}")
        report.append("")

        # Block-specific results
        for block_name , block_results in results.items():
            pass
        if not isinstance(block_results , dict):
                continue

            report.append(f"🔍 {block_name.upper()} BLOCK:")
            report.append("-" * 30)

        if "available_features" in block_results:
                report.append(
                    f"Available features: {len(block_results['available_features'])}",
                )
                report.append(
                    f"Missing features: {len(block_results['missing_features'])}",
                )

        if (
                "calculation_issues" in block_results
                and block_results["calculation_issues"]
            ):
                report.append(
                    f"Calculation issues: {len(block_results['calculation_issues'])}",
                )
        for issue in block_results["calculation_issues"][:3]:  # Show first 3
                    report.append(
                        f"  - {issue['feature']}: {issue['issue']} - {issue['description']}",
                    )

        if (
                "correlation_analysis" in block_results
                and "high_correlation_pairs" in block_results["correlation_analysis"]
            ):
                high_corr = block_results["correlation_analysis"][
                    "high_correlation_pairs"
                ]
                report.append(f"High correlation pairs: {len(high_corr)}")
        for pair in high_corr[:3]:  # Show first 3
                    report.append(
                        f"  - {pair['feature1']} ↔ {pair['feature2']}: {pair['correlation']:.3f}",
                    )

        if (
                "variance_analysis" in block_results
                and "zero_variance_features" in block_results["variance_analysis"]
            ):
                zero_var = block_results["variance_analysis"]["zero_variance_features"]
                report.append(f"Zero variance features: {len(zero_var)}")

            report.append("")

        # Variance threshold analysis
        if "variance_thresholds" in results:
            report.append("📈 VARIANCE THRESHOLD ANALYSIS:")
            report.append("-" * 40)

            threshold_analysis = results["variance_thresholds"]["threshold_analysis"]
        for threshold , analysis in threshold_analysis.items():
                report.append(
                    f"Threshold {threshold}: {analysis['features_removed']} features removed ({analysis['percentage_removed']:.1f}%)",
                )

        if results["variance_thresholds"]["recommendations"]:
                report.append("")
                report.append("💡 RECOMMENDATIONS:")
        for rec in results["variance_thresholds"]["recommendations"]:
                    report.append(f"  - {rec}")

        return "\n".join(report)


def main():
    """Main verification function."""
    verifier, FeatureCalculationVerifier()

    if True:
        # Load sample data (you'll need to provide actual data path)
        data_path = "data_cache/features_15m.parquet"  # Adjust path as needed

        if not Path(data_path).exists():
            print(f"❌ Data file not found: {data_path}")
            print("Please provide the correct path to your feature data file.")
            return

        print("📊 Loading feature data...")
        data = pd.read_parquet(data_path)
        print(f"✅ Loaded data with shape: {data.shape}")

        # Verify each block
        results = {}

        print("🔍 Verifying momentum features...")
        results["momentum"] = verifier.verify_momentum_features(data)

        print("🔍 Verifying volatility features...")
        results["volatility"] = verifier.verify_volatility_features(data)

        print("🔍 Verifying liquidity features...")
        results["liquidity"] = verifier.verify_liquidity_features(data)

        print("🔍 Verifying variance thresholds...")
        results["variance_thresholds"] = verifier.verify_variance_thresholds(data)

        # Generate report
        report = verifier.generate_verification_report(results)
        print(report)

        # Save report
        with open("feature_verification_report.txt", "w") as f:
            f.write(report)
        print("📄 Report saved to: feature_verification_report.txt")

    pass
        print(f"❌ Error during verification: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
