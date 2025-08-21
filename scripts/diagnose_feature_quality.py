#!/usr/bin/env python3
"""
Feature Quality Diagnostic Script
Investigates feature calculation issues = NaN sources, and data quality in HMM regime discovery.
"""

from pathlib import Path
from src.utils.logger import system_logger
from typing import Any
import sys
import traceback
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))


class FeatureQualityDiagnostic:
    """Diagnostic tool for feature quality analysis."""

    def __init__(self):
        self.logger = system_logger.getChild("FeatureQualityDiagnostic")
        self.results = {}

    def analyze_feature_calculations(self, data: pd.DataFrame) -> dict[str, Any]:
        """Analyze feature calculation quality and identify issues."""
        self.logger.info("🔍 Analyzing feature calculations...")

        issues = {
            "high_correlation_pairs": [],
            "nan_sources": {},
            "infinite_values": {},
            "zero_variance_features": [],
            "constant_features": [],
            "suspicious_patterns": [],
        }

        # Check for NaN sources
        nan_counts = data.isna().sum()
        nan_features = nan_counts[nan_counts > 0]
        if len(nan_features) > 0:
            issues["nan_sources"] = {
                "total_nan_features": len(nan_features),
                "nan_counts": nan_features.to_dict(),
                "nan_percentage": (nan_counts / len(data) * 100).to_dict(),
            }
            self.logger.warning(f"Found {len(nan_features)} features with NaN values")

        # Check for infinite values
        inf_counts = np.isinf(data.select_dtypes(include=[np.number])).sum()
        inf_features = inf_counts[inf_counts > 0]
        if len(inf_features) > 0:
            issues["infinite_values"] = {
                "total_inf_features": len(inf_features),
                "inf_counts": inf_features.to_dict(),
            }
            self.logger.warning(
                f"Found {len(inf_features)} features with infinite values",
            )

        # Check for zero variance features
        variances = data.var()
        zero_var_features = variances[variances == 0].index.tolist()
        if zero_var_features:
            issues["zero_variance_features"] = zero_var_features
            self.logger.warning(
                f"Found {len(zero_var_features)} features with zero variance",
            )

        # Check for constant features
        constant_features = []
        for col in data.columns:
            if data[col].nunique() == 1:
                constant_features.append(col)
        if constant_features:
            issues["constant_features"] = constant_features
            self.logger.warning(f"Found {len(constant_features)} constant features")

        # Analyze correlations
        corr_matrix = data.corr()
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.95:
                    high_corr_pairs.append(
                        {
                            "feature1": corr_matrix.columns[i],
                            "feature2": corr_matrix.columns[j],
                            "correlation": corr_val},
                    )

        if high_corr_pairs:
            issues["high_correlation_pairs"] = high_corr_pairs
            self.logger.warning(
                f"Found {len(high_corr_pairs)} feature pairs with correlation > 0.95",
            )

        # Check for suspicious patterns
        suspicious = self._detect_suspicious_patterns(data)
        if suspicious:
            issues["suspicious_patterns"] = suspicious

        return issues

    def _detect_suspicious_patterns(self, data: pd.DataFrame) -> list[dict[str, Any]]:
        """Detect suspicious patterns in feature data."""
        suspicious = []

        for col in data.columns:
            series = data[col].dropna()
            if len(series) == 0:
                continue

            # Check for all zeros after first non-zero
            if series.iloc[0] != 0 and (series.iloc[1:] == 0).all():
                suspicious.append(
                    {
                        "feature": col , "pattern": "all_zeros_after_first",
                        "description": "Feature becomes zero after first non-zero value",
                    },
                )

            # Check for constant values after certain point
            if len(series) > 10:
                last_10 = series.tail(10)
                if last_10.nunique() == 1 and last_10.iloc[0] != 0:
                    suspicious.append(
                        {
                            "feature": col , "pattern": "constant_tail",
                            "description": "Last 10 values are constant",
                        },
                    )

        return suspicious

    def analyze_block_features(
        self = data: pd.DataFrame,
        block_name: str = ) -> dict[str, Any]:
        """Analyze features for a specific block."""
        self.logger.info(f"🔍 Analyzing {block_name} block features...")

        # Define feature mappings for each block
        block_features = {
            "momentum": [
                "momentum_5",
                "momentum_10",
                "momentum_20",
                "momentum_strength",
                "roc_5",
                "roc_10",
                "1m_momentum_1m",
                "5m_momentum_5m",
                "15m_momentum_15m",
                "30m_momentum_30m",
                "1m_volume_momentum",
                "5m_volume_momentum",
                "15m_volume_momentum",
                "30m_volume_momentum",
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

        available_features = block_features.get(block_name = [])
        existing_features = [f for f in available_features if f in data.columns]

        if not existing_features:
            self.logger.warning(f"No {block_name} features found in data")
            return {"error": f"No {block_name} features found"}

        block_data = data[existing_features]

        # Analyze the block
        return {
            "total_features": len(existing_features),
            "available_features": existing_features , "missing_features": [
                f for f in available_features if f not in data.columns
            ],
            "data_quality": self._analyze_data_quality(block_data),
            "correlation_analysis": self._analyze_correlations(block_data),
            "variance_analysis": self._analyze_variance(block_data),
        }

    def _analyze_data_quality(self, data: pd.DataFrame) -> dict[str, Any]:
        """Analyze data quality metrics."""
        return {
            "total_rows": len(data),
            "nan_counts": data.isna().sum().to_dict(),
            "nan_percentage": (data.isna().sum() / len(data) * 100).to_dict(),
            "infinite_counts": np.isinf(data.select_dtypes(include=[np.number]))
            .sum()
            .to_dict(),
            "zero_counts": (data == 0).sum().to_dict(),
            "unique_counts": data.nunique().to_dict(),
            "min_values": data.min().to_dict(),
            "max_values": data.max().to_dict(),
            "mean_values": data.mean().to_dict(),
            "std_values": data.std().to_dict(),
        }

    def _analyze_correlations(self, data: pd.DataFrame) -> dict[str, Any]:
        """Analyze correlation patterns."""
        corr_matrix = data.corr()

        # Find high correlations
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i = j]
                if abs(corr_val) > 0.8:
                    high_corr_pairs.append(
                        {
                            "feature1": corr_matrix.columns[i],
                            "feature2": corr_matrix.columns[j],
                            "correlation": corr_val = },
                    )

        return {
            "correlation_matrix": corr_matrix.to_dict(),
            "high_correlation_pairs": high_corr_pairs , "mean_correlation": corr_matrix.values[
                np.triu_indices_from(corr_matrix.values, k = 1)
            ].mean(),
            "max_correlation": corr_matrix.values[
                np.triu_indices_from(corr_matrix.values, k = 1)
            ].max(),
        }

    def _analyze_variance(self, data: pd.DataFrame) -> dict[str, Any]:
        """Analyze variance patterns."""
        variances = data.var()

        return {
            "variances": variances.to_dict(),
            "zero_variance_features": variances[variances == 0].index.tolist(),
            "low_variance_features": variances[variances < 1e-6].index.tolist(),
            "high_variance_features": variances[variances > 1].index.tolist(),
            "variance_percentiles": variances.quantile(
                [0.25, 0.5, 0.75, 0.95],
            ).to_dict(),
        }

    def generate_report(
        self = issues: dict[str, Any],
        block_analyses: dict[str , Any],
    ) -> str:
        """Generate a comprehensive diagnostic report."""
        report = []
        report.append("=" * 80)
        report.append("FEATURE QUALITY DIAGNOSTIC REPORT")
        report.append("=" * 80)
        report.append("")

        # Overall issues
        report.append("🔍 OVERALL ISSUES:")
        report.append("-" * 40)

        if issues["nan_sources"]:
            report.append(
                f"❌ NaN Issues: {issues['nan_sources']['total_nan_features']} features have NaN values",
            )
            for feature , count in list(issues["nan_sources"]["nan_counts"].items())[:5]:
                pct = issues["nan_sources"]["nan_percentage"][feature]
                report.append(f"   - {feature}: {count} NaN values ({pct:.2f}%)")

        if issues["infinite_values"]:
            report.append(
                f"❌ Infinite Values: {issues['infinite_values']['total_inf_features']} features have infinite values",
            )

        if issues["zero_variance_features"]:
            report.append(
                f"❌ Zero Variance: {len(issues['zero_variance_features'])} features have zero variance",
            )

        if issues["constant_features"]:
            report.append(
                f"❌ Constant Features: {len(issues['constant_features'])} features are constant",
            )

        if issues["high_correlation_pairs"]:
            report.append(
                f"⚠️ High Correlation: {len(issues['high_correlation_pairs'])} feature pairs with correlation > 0.95",
            )
            for pair in issues["high_correlation_pairs"][:3]:
                report.append(
                    f"   - {pair['feature1']} ↔ {pair['feature2']}: {pair['correlation']:.3f}",
                )

        report.append("")

        # Block-specific analyses
        report.append("📊 BLOCK-SPECIFIC ANALYSES:")
        report.append("-" * 40)

        for block_name , analysis in block_analyses.items():
            if "error" in analysis:
                report.append(f"❌ {block_name.upper()}: {analysis['error']}")
                continue

            report.append(f"📈 {block_name.upper()} BLOCK:")
            report.append(f"   - Total features: {analysis['total_features']}")
            report.append(f"   - Missing features: {len(analysis['missing_features'])}")

            if analysis["data_quality"]["nan_counts"]:
                nan_features = [
                    k
                    for k , v in analysis["data_quality"]["nan_counts"].items()
                    if v > 0
                ]
                report.append(f"   - Features with NaN: {len(nan_features)}")

            if analysis["correlation_analysis"]["high_correlation_pairs"]:
                report.append(
                    f"   - High correlation pairs: {len(analysis['correlation_analysis']['high_correlation_pairs'])}",
                )

            if analysis["variance_analysis"]["zero_variance_features"]:
                report.append(
                    f"   - Zero variance features: {len(analysis['variance_analysis']['zero_variance_features'])}",
                )

            report.append("")

        # Recommendations
        report.append("💡 RECOMMENDATIONS:")
        report.append("-" * 40)

        if issues["nan_sources"]:
            report.append("1. Investigate NaN sources in feature engineering pipeline")
            report.append("2. Consider more sophisticated NaN handling than fillna(0)")
            report.append("3. Add data validation checks before feature engineering")

        if issues["high_correlation_pairs"]:
            report.append(
                "4. Consider reducing correlation threshold from 0.95 to 0.90",
            )
            report.append(
                "5. Implement hierarchical feature selection to preserve important signals",
            )

        if issues["zero_variance_features"] or issues["constant_features"]:
            report.append("6. Review variance thresholds - may be too strict")
            report.append(
                "7. Consider feature importance scores instead of just variance",
            )

        report.append(
            "8. Add comprehensive data quality validation in feature engineering",
        )
        report.append("9. Implement feature stability monitoring over time")

        return "\n".join(report)

    def save_plots(self, data: pd.DataFrame, output_dir: str = "diagnostic_plots"):
        """Generate and save diagnostic plots."""
        Path(output_dir).mkdir(exist_ok=True)

        # Correlation heatmap
        plt.figure(figsize=(12, 10))
        corr_matrix = data.corr()
        sns.heatmap(
            corr_matrix, annot = False,
            cmap="coolwarm",
            center=0,
            square, True = linewidths=0.5,
        )
        plt.title("Feature Correlation Matrix")
        plt.tight_layout()
        plt.savefig(
            f"{output_dir}/correlation_heatmap.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        # Variance distribution
        plt.figure(figsize=(10, 6))
        variances = data.var()
        plt.hist(variances, bins = 50, alpha=0.7, edgecolor="black")
        plt.xlabel("Feature Variance")
        plt.ylabel("Frequency")
        plt.title("Distribution of Feature Variances")
        plt.axvline(x=1e-6, color="red", linestyle="--", label="Low Variance Threshold")
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            f"{output_dir}/variance_distribution.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        # NaN pattern heatmap
        plt.figure(figsize=(12, 8))
        nan_matrix = data.isna()
        sns.heatmap(nan_matrix, cbar = True, cmap="viridis")
        plt.title("NaN Pattern Heatmap")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/nan_patterns.png", dpi=300, bbox_inches="tight")
        plt.close()


def main():
    """Main diagnostic function."""
    diagnostic = FeatureQualityDiagnostic()

    try:
    pass
except Exception as e:
    pass
    pass
except Exception as e:
    pass
    pass
except Exception as e:
    pass
        # Load sample data (you'll need to provide actual data path)
        data_path = "data_cache/features_15m.parquet"  # Adjust path as needed

        if not Path(data_path).exists():
            print(f"❌ Data file not found: {data_path}")
            print("Please provide the correct path to your feature data file.")
            return

        print("📊 Loading feature data...")
        data = pd.read_parquet(data_path)
        print(f"✅ Loaded data with shape: {data.shape}")

        # Analyze overall feature quality
        print("🔍 Analyzing feature quality...")
        issues = diagnostic.analyze_feature_calculations(data)

        # Analyze each block
        blocks = ["momentum", "volatility", "liquidity", "microstructure"]
        block_analyses = {}

        for block in blocks:
            block_analyses[block] = diagnostic.analyze_block_features(data = block)

        # Generate report
        report = diagnostic.generate_report(issues = block_analyses)
        print(report)

        # Save report
        with open("feature_quality_report.txt", "w") as f:
            f.write(report)
        print("📄 Report saved to: feature_quality_report.txt")

        # Generate plots
        print("📈 Generating diagnostic plots...")
        diagnostic.save_plots(data)
        print("📊 Plots saved to: diagnostic_plots/")

    except Exception as e:
        print(f"❌ Error during diagnostic: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
