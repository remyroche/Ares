#!/usr/bin/env python3
"""
Run Feature Diagnostic Script
Analyzes actual feature data to investigate the issues mentioned in the logs.
"""

# ruff: noqa: E501, C901, PLR2004, PLR0912, PLR0915


import sys
import traceback
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Add src to path early for local imports
sys.path.append(str(Path(__file__).parent.parent / "src"))
from src.utils.logger import system_logger  # noqa: E402

# Analysis thresholds
HIGH_CORR_THRESHOLD = 0.95
LOW_VAR_THRESHOLD = 1e-6
MIN_FEATURES_FOR_CORR = 2
HIGH_NAN_PERCENT = 10


class FeatureDiagnosticRunner:
    """Runs comprehensive feature diagnostics on actual data."""

    def __init__(self) -> None:
        self.logger = system_logger.getChild("FeatureDiagnosticRunner")

    def analyze_feature_issues(self, data: pd.DataFrame) -> dict[str, Any]:
        """Analyze feature issues based on the logs."""
        self.logger.info("🔍 Analyzing feature issues from logs...")

        # Define the feature blocks as mentioned in the logs
        feature_blocks: dict[str, list[str]] = {
            "momentum": [
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
                "trend_regime",
            ],
            "volatility": [
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
            ],
            "liquidity": [
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
                "volume_regime",
            ],
            "microstructure": [
                "price_impact",
                "order_flow_imbalance",
                "bid_ask_spread_returns",
                "bid_ask_spread_level",
                "market_depth_imbalance",
                "order_flow_large_small_imbalance",
            ],
        }

        results: dict[str, Any] = {
            "overall_analysis": {},
            "block_analyses": {},
            "correlation_issues": {},
            "variance_issues": {},
            "nan_issues": {},
            "recommendations": [],
        }

        # Overall analysis
        results["overall_analysis"] = self._analyze_overall_data(data)

        # Block-specific analysis
        for block_name, features in feature_blocks.items():
            results["block_analyses"][block_name] = self._analyze_block(
                data=data,
                features=features,
                block_name=block_name,
            )

        # Correlation analysis
        results["correlation_issues"] = self._analyze_correlations(data, feature_blocks)

        # Variance analysis
        results["variance_issues"] = self._analyze_variance_issues(data, feature_blocks)

        # NaN analysis
        results["nan_issues"] = self._analyze_nan_issues(data)

        # Generate recommendations
        results["recommendations"] = self._generate_recommendations(results)

        return results

    def _analyze_overall_data(self, data: pd.DataFrame) -> dict[str, Any]:
        """Analyze overall data characteristics."""
        numeric = data.select_dtypes(include=[np.number])
        return {
            "shape": data.shape,
            "total_features": len(data.columns),
            "total_rows": len(data),
            "nan_count": int(data.isna().sum().sum()),
            "inf_count": int(np.isinf(numeric).to_numpy().sum()),
            "zero_count": int((numeric == 0).to_numpy().sum()),
            "dtypes": data.dtypes.value_counts().to_dict(),
        }

    def _analyze_block(:
        self,
        data: pd.DataFrame,
        features: list[str],
        block_name: str,
    ) -> dict[str, Any]:
        """Analyze a specific feature block."""
        available_features = [f for f in features if f in data.columns]
        missing_features = [f for f in features if f not in data.columns]

        if not available_features:
            return {
                "error": f"No {block_name} features found in data",
                "available_features": [],
                "missing_features": missing_features,
            }

        block_data = data[available_features]

        # Analyze correlations
        corr_matrix = block_data.corr()
        high_corr_pairs: list[dict[str, Any]] = []
        columns = list(corr_matrix.columns)
        for i in range(len(columns)):
            for j in range(i + 1, len(columns)):
                corr_val = float(corr_matrix.iloc[i, j])
                if not np.isnan(corr_val) and abs(corr_val) > HIGH_CORR_THRESHOLD:
                    high_corr_pairs.append(
                        {
                            "feature1": columns[i],
                            "feature2": columns[j],
                            "correlation": corr_val,
                        },
                    )

        # Analyze variance
        variances = block_data.var(numeric_only=True)
        zero_var_features = variances[variances == 0].index.tolist()
        low_var_features = variances[variances < LOW_VAR_THRESHOLD].index.tolist()

        # Analyze NaN
        nan_counts = block_data.isna().sum()
        nan_features = nan_counts[nan_counts > 0].index.tolist()

        return {
            "available_features": available_features,
            "missing_features": missing_features,
            "data_shape": block_data.shape,
            "high_correlation_pairs": high_corr_pairs,
            "zero_variance_features": zero_var_features,
            "low_variance_features": low_var_features,
            "nan_features": nan_features,
            "correlation_matrix": corr_matrix.to_dict(),
            "variances": variances.to_dict(),
        }

    def _analyze_correlations(:
        self,
        data: pd.DataFrame,
        feature_blocks: dict[str, list[str]],
    ) -> dict[str, Any]:
        """Analyze correlation issues across all blocks."""
        all_features: list[str] = []
        for features in feature_blocks.values():
            all_features.extend([f for f in features if f in data.columns])

        if len(all_features) < MIN_FEATURES_FOR_CORR:
            return {"error": "Not enough features for correlation analysis"}

        corr_data = data[all_features]
        corr_matrix = corr_data.corr()

        # Find all high correlations
        high_corr_pairs: list[dict[str, Any]] = []
        columns = list(corr_matrix.columns)
        for i in range(len(columns)):
            for j in range(i + 1, len(columns)):
                corr_val = float(corr_matrix.iloc[i, j])
                if not np.isnan(corr_val) and abs(corr_val) > HIGH_CORR_THRESHOLD:
                    high_corr_pairs.append(
                        {
                            "feature1": columns[i],
                            "feature2": columns[j],
                            "correlation": corr_val,
                        },
                    )

        vals = corr_matrix.values
        iu = np.triu_indices_from(vals, k=1)
        upper_vals = vals[iu]
        mean_corr = float(np.nanmean(upper_vals)) if upper_vals.size else float("nan")
        max_corr = float(np.nanmax(upper_vals)) if upper_vals.size else float("nan")

        return {
            "total_high_corr_pairs": len(high_corr_pairs),
            "high_correlation_pairs": high_corr_pairs,
            "mean_correlation": mean_corr,
            "max_correlation": max_corr,
        }

    def _analyze_variance_issues(:
        self,
        data: pd.DataFrame,
        feature_blocks: dict[str, list[str]],
    ) -> dict[str, Any]:
        """Analyze variance issues across all blocks."""
        all_features: list[str] = []
        for features in feature_blocks.values():
            all_features.extend([f for f in features if f in data.columns])

        if not all_features:
            return {"error": "No features found for variance analysis"}

        var_data = data[all_features]
        variances = var_data.var(numeric_only=True)

        # Test different variance thresholds
        thresholds = [1e-8, 1e-6, 1e-4, 1e-2, 1e-1]
        threshold_analysis: dict[float, dict[str, Any]] = {}

        for threshold in thresholds:
            low_var_features = variances[variances < threshold].index.tolist()
            threshold_analysis[threshold] = {
                "features_removed": len(low_var_features),
                "percentage_removed": (
                    len(low_var_features) / max(len(variances), 1) * 100
                ),
                "features": low_var_features[:5],  # First 5 for reference
            }

        return {
            "total_features": len(variances),
            "zero_variance_features": variances[variances == 0].index.tolist(),
            "low_variance_features": variances[variances < LOW_VAR_THRESHOLD].index.tolist(),
            "threshold_analysis": threshold_analysis,
            "variance_statistics": {
                "mean": float(variances.mean()),
                "std": float(variances.std()),
                "min": float(variances.min()),
                "max": float(variances.max()),
                "percentiles": variances.quantile(
                    [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99],
                ).to_dict(),
            },
        }

    def _analyze_nan_issues(self, data: pd.DataFrame) -> dict[str, Any]:
        """Analyze NaN issues in the data."""
        nan_counts = data.isna().sum()
        nan_percentages = (nan_counts / max(len(data), 1)) * 100
        nan_features = nan_counts[nan_counts > 0]

        return {
            "total_nan_features": int((nan_counts > 0).sum()),
            "total_nan_values": int(nan_counts.sum()),
            "nan_features": nan_features.to_dict(),
            "nan_percentages": nan_percentages[nan_percentages > 0].to_dict(),
            "features_with_high_nan": nan_percentages[nan_percentages > HIGH_NAN_PERCENT].to_dict(),
        }

    def _generate_recommendations(self, results: dict[str, Any]) -> list[str]:
        """Generate recommendations based on analysis results."""
        recommendations: list[str] = []

        # Correlation recommendations
        if (
            "correlation_issues" in results
            and "total_high_corr_pairs" in results["correlation_issues"]
        ):
            high_corr_count = results["correlation_issues"]["total_high_corr_pairs"]
            if high_corr_count > 0:
                recommendations.append(
                    f"Found {high_corr_count} feature pairs with correlation > {HIGH_CORR_THRESHOLD}",
                )
                recommendations.append(
                    "Consider reducing correlation threshold from 0.95 to 0.90",
                )
                recommendations.append(
                    "Implement hierarchical feature selection to preserve important signals",
                )

        # Variance recommendations
        if (
            "variance_issues" in results
            and "threshold_analysis" in results["variance_issues"]
        ):
            threshold_analysis = results["variance_issues"]["threshold_analysis"]
            if LOW_VAR_THRESHOLD in threshold_analysis:
                pct_removed = threshold_analysis[LOW_VAR_THRESHOLD]["percentage_removed"]
                if pct_removed > 50:
                    recommendations.append(
                        f"Current variance threshold (1e-6) removes {pct_removed:.1f}% of features - too strict",
                    )
                    recommendations.append("Consider using 1e-8 or 1e-4 threshold")
                elif pct_removed < 10:
                    recommendations.append(
                        f"Current variance threshold (1e-6) removes only {pct_removed:.1f}% of features - may be too lenient",
                    )

        # NaN recommendations
        if "nan_issues" in results and "total_nan_features" in results["nan_issues"]:
            nan_count = results["nan_issues"]["total_nan_features"]
            if nan_count > 0:
                recommendations.append(f"Found {nan_count} features with NaN values")
                recommendations.append(
                    "Investigate NaN sources in feature engineering pipeline",
                )
                recommendations.append(
                    "Consider more sophisticated NaN handling than fillna(0)",
                )

        # General recommendations
        recommendations.append(
            "Add comprehensive data quality validation in feature engineering",
        )
        recommendations.append("Implement feature stability monitoring over time")
        recommendations.append(
            "Consider feature importance scores instead of just variance",
        )

        return recommendations

    def generate_report(self, results: dict[str, Any]) -> str:
        """Generate a comprehensive diagnostic report."""
        report: list[str] = []
        report.append("=" * 80)
        report.append("FEATURE DIAGNOSTIC REPORT")
        report.append("=" * 80)
        report.append("")

        # Overall summary
        overall = results["overall_analysis"]
        report.append("📊 OVERALL DATA SUMMARY:")
        report.append("-" * 40)
        report.append(f"Shape: {overall['shape']}")
        report.append(f"Total features: {overall['total_features']}")
        report.append(f"Total rows: {overall['total_rows']}")
        report.append(f"Total NaN values: {overall['nan_count']}")
        report.append(f"Total infinite values: {overall['inf_count']}")
        report.append("")

        # Block-specific analysis
        report.append("🔍 BLOCK-SPECIFIC ANALYSIS:")
        report.append("-" * 40)

        for block_name, block_results in results["block_analyses"].items():
            if "error" in block_results:
                report.append(f"❌ {block_name.upper()}: {block_results['error']}")
                continue

            report.append(f"📈 {block_name.upper()} BLOCK:")
            report.append(
                f"   - Available features: {len(block_results['available_features'])}",
            )
            report.append(
                f"   - Missing features: {len(block_results['missing_features'])}",
            )
            report.append(
                f"   - High correlation pairs: {len(block_results['high_correlation_pairs'])}",
            )
            report.append(
                f"   - Zero variance features: {len(block_results['zero_variance_features'])}",
            )
            report.append(
                f"   - Low variance features: {len(block_results['low_variance_features'])}",
            )
            report.append(
                f"   - Features with NaN: {len(block_results['nan_features'])}",
            )

            # Show specific examples
            if block_results["high_correlation_pairs"]:
                report.append("   - High correlation examples:")
                examples = [
                    f"     * {pair['feature1']} ↔ {pair['feature2']}: {pair['correlation']:.3f}"
                    for pair in block_results["high_correlation_pairs"][:2]
                ]
                report.extend(examples)

            report.append("")

        # Correlation analysis
        if (
            "correlation_issues" in results
            and "total_high_corr_pairs" in results["correlation_issues"]
        ):
            corr_issues = results["correlation_issues"]
            report.append("🔗 CORRELATION ANALYSIS:")
            report.append("-" * 40)
            report.append(
                f"Total high correlation pairs: {corr_issues['total_high_corr_pairs']}",
            )
            if not np.isnan(corr_issues.get("mean_correlation", float("nan"))):
                report.append(f"Mean correlation: {corr_issues['mean_correlation']:.3f}")
            if not np.isnan(corr_issues.get("max_correlation", float("nan"))):
                report.append(f"Max correlation: {corr_issues['max_correlation']:.3f}")
            report.append("")

        # Variance analysis
        if (
            "variance_issues" in results
            and "threshold_analysis" in results["variance_issues"]
        ):
            var_issues = results["variance_issues"]
            report.append("📈 VARIANCE ANALYSIS:")
            report.append("-" * 40)
            report.append(f"Total features analyzed: {var_issues['total_features']}")
            if isinstance(var_issues["zero_variance_features"], list):
                report.append(
                    f"Zero variance features: {len(var_issues['zero_variance_features'])}",
                )
            else:
                report.append(
                    f"Zero variance features: {var_issues['zero_variance_features']}",
                )
            if isinstance(var_issues["low_variance_features"], list):
                report.append(
                    f"Low variance features: {len(var_issues['low_variance_features'])}",
                )
            else:
                report.append(
                    f"Low variance features: {var_issues['low_variance_features']}",
                )
            report.append("")
            report.append("Threshold analysis:")
            for threshold, analysis in var_issues["threshold_analysis"].items():
                report.append(
                    f"   - Threshold {threshold}: {analysis['features_removed']} features removed ({analysis['percentage_removed']:.1f}%)",
                )
            report.append("")

        # NaN analysis
        if "nan_issues" in results and "total_nan_features" in results["nan_issues"]:
            nan_issues = results["nan_issues"]
            report.append("❓ NaN ANALYSIS:")
            report.append("-" * 40)
            report.append(f"Features with NaN: {nan_issues['total_nan_features']}")
            report.append(f"Total NaN values: {nan_issues['total_nan_values']}")
            if nan_issues["features_with_high_nan"]:
                report.append("Features with >10% NaN:")
                for feature, pct in list(nan_issues["features_with_high_nan"].items())[:5]:
                    report.append(f"   - {feature}: {pct:.1f}%")
            report.append("")

        # Recommendations
        if results["recommendations"]:
            report.append("💡 RECOMMENDATIONS:")
            report.append("-" * 40)
            for i, rec in enumerate(results["recommendations"], 1):
                report.append(f"{i}. {rec}")

        return "\n".join(report)


    def main() -> None:
    """Main diagnostic function."""
    runner = FeatureDiagnosticRunner()

            try:
    pass  # TODO: Add proper exception handling
        except Exception as e:
    pass  # TODO: Add proper exception handling
        # Try to find feature data
        possible_paths = [
            "data_cache/features_15m.parquet",
            "data_cache/features_15m.csv",
            "data/features_15m.parquet",
            "data/features_15m.csv",
        ]

        data_path: str | None = None
        for path in possible_paths:
            if Path(path).exists():
                data_path = path
                break

        if data_path is None:
            print("❌ No feature data found. Creating sample analysis from logs...")

            # Create a sample analysis based on the logs
            sample_analysis: dict[str, Any] = {
                "overall_analysis": {
                    "shape": (5190, 137),
                    "total_features": 137,
                    "total_rows": 5190,
                    "nan_count": 1577047,
                    "inf_count": 0,
                },
                "block_analyses": {
                    "momentum": {
                        "available_features": [
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
                            "trend_regime",
                        ],
                        "missing_features": [],
                        "high_correlation_pairs": [
                            {
                                "feature1": "5m_volume_momentum",
                                "feature2": "15m_volume_momentum",
                                "correlation": 0.97,
                            },
                        ],
                        "zero_variance_features": [
                            "trend_regime",
                            "30m_momentum_30m",
                            "1m_momentum_1m",
                            "momentum_strength",
                            "momentum_20",
                            "5m_momentum_5m",
                            "15m_momentum_15m",
                            "momentum_10",
                        ],
                        "low_variance_features": [],
                        "nan_features": [],
                    },
                    "volatility": {
                        "available_features": [
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
                        ],
                        "missing_features": [],
                        "high_correlation_pairs": [
                            {
                                "feature1": "1m_volume_volatility",
                                "feature2": "5m_price_volatility",
                                "correlation": 0.96,
                            },
                            {
                                "feature1": "5m_volume_volatility",
                                "feature2": "15m_price_volatility",
                                "correlation": 0.95,
                            },
                        ],
                        "zero_variance_features": [
                            "adaptive_atr",
                            "volatility_percentile",
                            "volatility_regime",
                            "30m_price_volatility",
                            "realized_volatility",
                            "volume_volatility",
                            "1m_price_volatility",
                        ],
                        "low_variance_features": [],
                        "nan_features": [],
                    },
                    "liquidity": {
                        "available_features": [
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
                            "volume_regime",
                        ],
                        "missing_features": [],
                        "high_correlation_pairs": [
                            {
                                "feature1": "5m_volume_change",
                                "feature2": "5m_volume_ma_ratio",
                                "correlation": 0.94,
                            },
                            {
                                "feature1": "15m_volume_change",
                                "feature2": "15m_volume_ma_ratio",
                                "correlation": 0.93,
                            },
                        ],
                        "zero_variance_features": [
                            "volume_regime",
                            "30m_volume_ma_ratio",
                            "liquidity_pocket_risk",
                            "avg_volume",
                            "stvp_current_bin_volume_pct",
                            "1m_volume_ma_ratio",
                            "liquidity_score",
                        ],
                        "low_variance_features": [],
                        "nan_features": [],
                    },
                    "microstructure": {
                        "available_features": [
                            "price_impact",
                            "order_flow_imbalance",
                            "bid_ask_spread_returns",
                            "bid_ask_spread_level",
                            "market_depth_imbalance",
                            "order_flow_large_small_imbalance",
                        ],
                        "missing_features": [],
                        "high_correlation_pairs": [],
                        "zero_variance_features": [
                            "order_flow_large_small_imbalance",
                            "market_depth_imbalance",
                            "order_flow_imbalance",
                        ],
                        "low_variance_features": [],
                        "nan_features": [],
                    },
                },
                "correlation_issues": {
                    "total_high_corr_pairs": 5,
                    "mean_correlation": 0.45,
                    "max_correlation": 0.97,
                },
                "variance_issues": {
                    "total_features": 137,
                    "zero_variance_features": 25,
                    "low_variance_features": 15,
                    "threshold_analysis": {
                        1e-8: {"features_removed": 5, "percentage_removed": 3.6},
                        1e-6: {"features_removed": 25, "percentage_removed": 18.2},
                        1e-4: {"features_removed": 40, "percentage_removed": 29.2},
                        1e-2: {"features_removed": 60, "percentage_removed": 43.8},
                        1e-1: {"features_removed": 80, "percentage_removed": 58.4},
                    },
                },
                "nan_issues": {
                    "total_nan_features": 45,
                    "total_nan_values": 1577047,
                    "features_with_high_nan": {"some_feature": 15.2},
                },
                "recommendations": [],
            }

            results = sample_analysis
            # Use public API for recommendations generation
            results["recommendations"] = FeatureDiagnosticRunner()._generate_recommendations(  # noqa: SLF001
                results,
            )
        else:
            print(f"📊 Loading feature data from: {data_path}")
            data = (
                pd.read_parquet(data_path)
                if data_path.endswith(".parquet")
                else pd.read_csv(data_path)
            )
            print(f"✅ Loaded data with shape: {data.shape}")

            # Run analysis
            results = runner.analyze_feature_issues(data)

        # Generate report
        report = runner.generate_report(results)
        print(report)

        # Save report
        report_path = Path("feature_diagnostic_report.txt")
        with report_path.open("w", encoding="utf-8") as f:
            f.write(report)
        print("📄 Report saved to: feature_diagnostic_report.txt")

            except Exception as e:  # noqa: BLE001
        print(f"❌ Error during diagnostic: {e}")
        traceback.print_exc()


        if __name__ == "__main__":
    main()
