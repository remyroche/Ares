#!/usr/bin/env python3
"""
Feature Quality Diagnostic Script
Investigates feature calculation issues, NaN sources, and data quality in HMM regime discovery.
"""


from pathlib import Path
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

from src.utils.logger import system_logger  # noqa: E402


def _handle_errors(default: Optional[Any] = None) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator to log exceptions and optionally return a default value.

    This keeps the script robust for long-running diagnostics.
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            logger = system_logger.getChild(func.__name__)
            try:
                return func(*args, **kwargs)
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.error("Error in %s: %s", func.__name__, exc, exc_info=True)
                return default

        return wrapper

    return decorator


def _require_dataframe(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator to validate the first argument is a pandas DataFrame."""

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        if not args:
            raise ValueError("Function requires a DataFrame as first argument")
        df = args[1] if hasattr(args[0], "__class__") else args[0]
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Expected a pandas DataFrame as input")
        return func(*args, **kwargs)

    return wrapper


class FeatureQualityDiagnostic:
    """Diagnostic tool for feature quality analysis."""

    def __init__(self) -> None:
        self.logger = system_logger.getChild("FeatureQualityDiagnostic")
        self.results: Dict[str, Any] = {}

    @_handle_errors(default={})
    @_require_dataframe
    def analyze_feature_calculations(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze feature calculation quality and identify issues."""
        self.logger.info("🔍 Analyzing feature calculations...")

        numeric_df = data.select_dtypes(include=[np.number])

        issues: Dict[str, Any] = {
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
                "total_nan_features": int(len(nan_features)),
                "nan_counts": nan_features.to_dict(),
                "nan_percentage": ((nan_counts / max(len(data), 1)) * 100).to_dict(),
            }
            self.logger.warning("Found %d features with NaN values", len(nan_features))

        # Check for infinite values
        if not numeric_df.empty:
            inf_counts = np.isinf(numeric_df).sum()
            inf_features = inf_counts[inf_counts > 0]
            if len(inf_features) > 0:
                issues["infinite_values"] = {
                    "total_inf_features": int(len(inf_features)),
                    "inf_counts": inf_features.to_dict(),
                }
                self.logger.warning(
                    "Found %d features with infinite values", len(inf_features)
                )

        # Check for zero variance features
        variances = numeric_df.var(numeric_only=True)
        zero_var_features = variances[variances == 0].index.tolist()
        if zero_var_features:
            issues["zero_variance_features"] = zero_var_features
            self.logger.warning(
                "Found %d features with zero variance", len(zero_var_features)
            )

        # Check for constant features
        constant_features: List[str] = []
        for col in data.columns:
            if data[col].nunique(dropna=True) == 1:
                constant_features.append(col)
        if constant_features:
            issues["constant_features"] = constant_features
            self.logger.warning("Found %d constant features", len(constant_features))

        # Analyze correlations
        high_corr_pairs: List[Dict[str, Any]] = []
        if not numeric_df.empty:
            corr_matrix = numeric_df.corr(numeric_only=True)
            cols = list(corr_matrix.columns)
            for i in range(len(cols)):
                for j in range(i + 1, len(cols)):
                    corr_val = float(corr_matrix.iloc[i, j])
                    if abs(corr_val) > 0.95:
                        high_corr_pairs.append(
                            {
                                "feature1": cols[i],
                                "feature2": cols[j],
                                "correlation": corr_val,
                            }
                        )
        if high_corr_pairs:
            issues["high_correlation_pairs"] = high_corr_pairs
            self.logger.warning(
                "Found %d feature pairs with correlation > 0.95",
                len(high_corr_pairs),
            )

        # Check for suspicious patterns
        suspicious = self._detect_suspicious_patterns(data)
        if suspicious:
            issues["suspicious_patterns"] = suspicious

        return issues

    @_handle_errors(default=[])
    @_require_dataframe
    def _detect_suspicious_patterns(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect suspicious patterns in feature data."""
        suspicious: List[Dict[str, Any]] = []

        for col in data.columns:
            series = data[col].dropna()
            if len(series) == 0:
                continue

            # Check for all zeros after first non-zero
            if series.iloc[0] != 0 and (series.iloc[1:] == 0).all():
                suspicious.append(
                    {
                        "feature": col,
                        "pattern": "all_zeros_after_first",
                        "description": "Feature becomes zero after first non-zero value",
                    }
                )

            # Check for constant values at the tail
            if len(series) > 10:
                last_10 = series.tail(10)
                if last_10.nunique() == 1 and last_10.iloc[0] != 0:
                    suspicious.append(
                        {
                            "feature": col,
                            "pattern": "constant_tail",
                            "description": "Last 10 values are constant",
                        }
                    )

        return suspicious

    @_handle_errors(default={})
    @_require_dataframe
    def analyze_block_features(self, data: pd.DataFrame, block_name: str) -> Dict[str, Any]:
        """Analyze features for a specific block."""
        self.logger.info("🔍 Analyzing %s block features...", block_name)

        # Define feature mappings for each block
        block_features: Dict[str, List[str]] = {
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

        available_features = block_features.get(block_name, [])
        existing_features = [f for f in available_features if f in data.columns]

        if not existing_features:
            self.logger.warning("No %s features found in data", block_name)
            return {"error": f"No {block_name} features found"}

        block_data = data[existing_features]

        # Analyze the block
        return {
            "total_features": len(existing_features),
            "available_features": existing_features,
            "missing_features": [
                f for f in available_features if f not in data.columns
            ],
            "data_quality": self._analyze_data_quality(block_data),
            "correlation_analysis": self._analyze_correlations(block_data),
            "variance_analysis": self._analyze_variance(block_data),
        }

    @_handle_errors(default={})
    @_require_dataframe
    def _analyze_data_quality(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze data quality metrics."""
        numeric_df = data.select_dtypes(include=[np.number])
        return {
            "total_rows": int(len(data)),
            "nan_counts": data.isna().sum().to_dict(),
            "nan_percentage": (
                (data.isna().sum() / max(len(data), 1) * 100).to_dict()
            ),
            "infinite_counts": (
                np.isinf(numeric_df).sum().to_dict() if not numeric_df.empty else {}
            ),
            "zero_counts": (data == 0).sum().to_dict(),
            "unique_counts": data.nunique().to_dict(),
            "min_values": data.min(numeric_only=True).to_dict(),
            "max_values": data.max(numeric_only=True).to_dict(),
            "mean_values": data.mean(numeric_only=True).to_dict(),
            "std_values": data.std(numeric_only=True).to_dict(),
        }

    @_handle_errors(default={})
    @_require_dataframe
    def _analyze_correlations(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze correlation patterns."""
        numeric_df = data.select_dtypes(include=[np.number])
        if numeric_df.empty:
            return {
                "correlation_matrix": {},
                "high_correlation_pairs": [],
                "mean_correlation": 0.0,
                "max_correlation": 0.0,
            }

        corr_matrix = numeric_df.corr(numeric_only=True)

        # Find high correlations
        high_corr_pairs: List[Dict[str, Any]] = []
        cols = list(corr_matrix.columns)
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                corr_val = float(corr_matrix.iloc[i, j])
                if abs(corr_val) > 0.8:
                    high_corr_pairs.append(
                        {
                            "feature1": cols[i],
                            "feature2": cols[j],
                            "correlation": corr_val,
                        }
                    )

        # Upper triangle statistics
        tri = np.triu_indices_from(corr_matrix.values, k=1)
        upper_vals = corr_matrix.values[tri]
        mean_corr = float(upper_vals.mean()) if upper_vals.size else 0.0
        max_corr = float(upper_vals.max()) if upper_vals.size else 0.0

        return {
            "correlation_matrix": corr_matrix.to_dict(),
            "high_correlation_pairs": high_corr_pairs,
            "mean_correlation": mean_corr,
            "max_correlation": max_corr,
        }

    @_handle_errors(default={})
    @_require_dataframe
    def _analyze_variance(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze variance patterns."""
        numeric_df = data.select_dtypes(include=[np.number])
        variances = numeric_df.var(numeric_only=True)

        return {
            "variances": variances.to_dict(),
            "zero_variance_features": variances[variances == 0].index.tolist(),
            "low_variance_features": variances[variances < 1e-6].index.tolist(),
            "high_variance_features": variances[variances > 1].index.tolist(),
            "variance_percentiles": variances.quantile([0.25, 0.5, 0.75, 0.95]).to_dict(),
        }

    @_handle_errors(default="")
    def generate_report(
        self, issues: Dict[str, Any], block_analyses: Dict[str, Any]
    ) -> str:
        """Generate a comprehensive diagnostic report."""
        report: List[str] = []
        report.append("=" * 80)
        report.append("FEATURE QUALITY DIAGNOSTIC REPORT")
        report.append("=" * 80)
        report.append("")

        # Overall issues
        report.append("🔍 OVERALL ISSUES:")
        report.append("-" * 40)

        if issues.get("nan_sources"):
            report.append(
                f"❌ NaN Issues: {issues['nan_sources']['total_nan_features']} features have NaN values"
            )
            for feature, count in list(issues["nan_sources"]["nan_counts"].items())[:5]:
                pct = issues["nan_sources"]["nan_percentage"][feature]
                report.append(f"   - {feature}: {count} NaN values ({pct:.2f}%)")

        if issues.get("infinite_values"):
            report.append(
                f"❌ Infinite Values: {issues['infinite_values']['total_inf_features']} features have infinite values"
            )

        if issues.get("zero_variance_features"):
            report.append(
                f"❌ Zero Variance: {len(issues['zero_variance_features'])} features have zero variance"
            )

        if issues.get("constant_features"):
            report.append(
                f"❌ Constant Features: {len(issues['constant_features'])} features are constant"
            )

        if issues.get("high_correlation_pairs"):
            report.append(
                f"⚠️ High Correlation: {len(issues['high_correlation_pairs'])} feature pairs with correlation > 0.95"
            )
            for pair in issues["high_correlation_pairs"][:3]:
                report.append(
                    f"   - {pair['feature1']} ↔ {pair['feature2']}: {pair['correlation']:.3f}"
                )

        report.append("")

        # Block-specific analyses
        report.append("📊 BLOCK-SPECIFIC ANALYSES:")
        report.append("-" * 40)

        for block_name, analysis in block_analyses.items():
            if "error" in analysis:
                report.append(f"❌ {block_name.upper()}: {analysis['error']}")
                continue

            report.append(f"📈 {block_name.upper()} BLOCK:")
            report.append(f"   - Total features: {analysis['total_features']}")
            report.append(f"   - Missing features: {len(analysis['missing_features'])}")

            if analysis["data_quality"]["nan_counts"]:
                nan_features = [
                    k
                    for k, v in analysis["data_quality"]["nan_counts"].items()
                    if v > 0
                ]
                report.append(f"   - Features with NaN: {len(nan_features)}")

            if analysis["correlation_analysis"]["high_correlation_pairs"]:
                report.append(
                    f"   - High correlation pairs: {len(analysis['correlation_analysis']['high_correlation_pairs'])}"
                )

            if analysis["variance_analysis"]["zero_variance_features"]:
                report.append(
                    f"   - Zero variance features: {len(analysis['variance_analysis']['zero_variance_features'])}"
                )

            report.append("")

        # Recommendations
        report.append("💡 RECOMMENDATIONS:")
        report.append("-" * 40)

        if issues.get("nan_sources"):
            report.append("1. Investigate NaN sources in feature engineering pipeline")
            report.append("2. Consider more sophisticated NaN handling than fillna(0)")
            report.append("3. Add data validation checks before feature engineering")

        if issues.get("high_correlation_pairs"):
            report.append("4. Consider reducing correlation threshold from 0.95 to 0.90")
            report.append(
                "5. Implement hierarchical feature selection to preserve important signals"
            )

        if issues.get("zero_variance_features") or issues.get("constant_features"):
            report.append("6. Review variance thresholds - may be too strict")
            report.append(
                "7. Consider feature importance scores instead of just variance"
            )

        report.append(
            "8. Add comprehensive data quality validation in feature engineering"
        )
        report.append("9. Implement feature stability monitoring over time")

        return "\n".join(report)

    @_handle_errors(default=None)
    @_require_dataframe
    def save_plots(self, data: pd.DataFrame, output_dir: str = "diagnostic_plots") -> None:
        """Generate and save diagnostic plots."""
        Path(output_dir).mkdir(exist_ok=True)

        numeric_df = data.select_dtypes(include=[np.number])

        # Correlation heatmap
        if not numeric_df.empty:
            plt.figure(figsize=(12, 10))
            corr_matrix = numeric_df.corr(numeric_only=True)
            sns.heatmap(
                corr_matrix,
                annot=False,
                cmap="coolwarm",
                center=0,
                square=True,
                linewidths=0.5,
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
        if not numeric_df.empty:
            plt.figure(figsize=(10, 6))
            variances = numeric_df.var(numeric_only=True)
            plt.hist(variances, bins=50, alpha=0.7, edgecolor="black")
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
        sns.heatmap(nan_matrix, cbar=True, cmap="viridis")
        plt.title("NaN Pattern Heatmap")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/nan_patterns.png", dpi=300, bbox_inches="tight")
        plt.close()


def main() -> None:
    """Main diagnostic function."""
    diagnostic = FeatureQualityDiagnostic()

    try:
    pass  # TODO: Add proper exception handling
except Exception as e:
    pass  # TODO: Add proper exception handling
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
        block_analyses: Dict[str, Any] = {}

        for block in blocks:
            block_analyses[block] = diagnostic.analyze_block_features(data, block)

        # Generate report
        report = diagnostic.generate_report(issues, block_analyses)
        print(report)

        # Save report
        with open("feature_quality_report.txt", "w", encoding="utf-8") as f:
            f.write(report)
        print("📄 Report saved to: feature_quality_report.txt")

        # Generate plots
        print("📈 Generating diagnostic plots...")
        diagnostic.save_plots(data)
        print("📊 Plots saved to: diagnostic_plots/")

    except Exception as e:  # pragma: no cover - defensive CLI wrapper
        print(f"❌ Error during diagnostic: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
