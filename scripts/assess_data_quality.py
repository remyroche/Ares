#!/usr/bin/env python3
"""
Enhanced Data Quality Assessment Script

This script demonstrates how to use the VectorizedLabellingOrchestrator's
data quality assessment functionality to analyze NaN/missing data in your datasets.
Now includes comprehensive multicollinearity analysis and label imbalance detection.

Usage:
    python scripts/assess_data_quality.py --data_path /path/to/your/data --symbol ETHUSDT --exchange binance
"""

import asyncio
import argparse
import sys
import os
from pathlib import Path

# Add the src directory to the Python path
current_dir = Path(__file__).parent
src_dir = current_dir.parent / "src"
sys.path.insert(0, str(src_dir))

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer

# Now import from the correct path
from training.steps.vectorized_labelling_orchestrator import (
    VectorizedLabellingOrchestrator,
)
from utils.logger import system_logger


class EnhancedDataQualityAnalyzer:
    """
    Enhanced data quality analyzer that addresses critical issues:
    1. Multicollinearity detection and resolution
    2. Label imbalance analysis and recommendations
    3. Feature redundancy identification
    """

    def __init__(self):
        self.logger = system_logger.getChild("EnhancedDataQualityAnalyzer")

    def analyze_multicollinearity(
        self, data: pd.DataFrame, vif_threshold: float = 5.0
    ) -> Dict[str, Any]:
        """
        Analyze multicollinearity using VIF and correlation analysis.

        Args:
            data: Input DataFrame
            vif_threshold: VIF threshold for flagging multicollinearity

        Returns:
            Dictionary with multicollinearity analysis results
        """
        try:
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
                    f"⚠️ Removing label columns from multicollinearity analysis: {actual_label_columns}"
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
            vif_scores = {}
            high_vif_features = []

            for i, col in enumerate(data_imputed.columns):
                other_cols = [c for c in data_imputed.columns if c != col]
                if len(other_cols) > 0:
                    X = data_imputed[other_cols]
                    y = data_imputed[col]

                    reg = LinearRegression()
                    reg.fit(X, y)

                    y_pred = reg.predict(X)
                    ss_res = np.sum((y - y_pred) ** 2)
                    ss_tot = np.sum((y - np.mean(y)) ** 2)
                    r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

                    vif = 1 / (1 - r_squared) if r_squared != 1 else np.inf
                    vif_scores[col] = vif

                    if vif > vif_threshold:
                        high_vif_features.append(col)

            # Calculate correlation matrix
            correlation_matrix = data_imputed.corr()
            high_correlation_pairs = []

            for i in range(len(correlation_matrix.columns)):
                for j in range(i + 1, len(correlation_matrix.columns)):
                    corr_val = abs(correlation_matrix.iloc[i, j])
                    if corr_val > 0.95:  # High correlation threshold
                        high_correlation_pairs.append(
                            {
                                "feature1": correlation_matrix.columns[i],
                                "feature2": correlation_matrix.columns[j],
                                "correlation": corr_val,
                            }
                        )

            # Identify redundant price features
            price_features = [
                "open",
                "high",
                "low",
                "close",
                "avg_price",
                "min_price",
                "max_price",
            ]
            redundant_price_features = [
                col
                for col in data_imputed.columns
                if any(price_feat in col.lower() for price_feat in price_features)
            ]

            # Sort high VIF features by VIF value
            high_vif_features_sorted = sorted(
                high_vif_features, key=lambda x: vif_scores[x], reverse=True
            )

            analysis_result = {
                "vif_scores": vif_scores,
                "high_vif_features": high_vif_features_sorted,
                "high_vif_count": len(high_vif_features),
                "max_vif": max(vif_scores.values()) if vif_scores else 0,
                "mean_vif": np.mean(list(vif_scores.values())) if vif_scores else 0,
                "high_correlation_pairs": high_correlation_pairs,
                "redundant_price_features": redundant_price_features,
                "total_features": len(data_imputed.columns),
                "severity": self._assess_multicollinearity_severity(
                    vif_scores, vif_threshold
                ),
            }

            return analysis_result

        except Exception as e:
            self.logger.error(f"❌ Error analyzing multicollinearity: {e}")
            return {"error": str(e)}

    def _assess_multicollinearity_severity(
        self, vif_scores: Dict[str, float], threshold: float
    ) -> str:
        """Assess the severity of multicollinearity issues."""
        if not vif_scores:
            return "UNKNOWN"

        max_vif = max(vif_scores.values())
        high_vif_count = sum(1 for vif in vif_scores.values() if vif > threshold)

        if max_vif > 1000:
            return "CRITICAL"
        elif max_vif > 100:
            return "HIGH"
        elif max_vif > 10:
            return "MODERATE"
        elif high_vif_count > 0:
            return "LOW"
        else:
            return "NONE"

    def analyze_label_distribution(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze label distribution and identify imbalance issues.

        Args:
            data: Input DataFrame with label column

        Returns:
            Dictionary with label analysis results
        """
        try:
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
                "Class",
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
            label_distribution = dict(zip(unique_labels, counts))

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
                    "Consider binary classification (remove HOLD class)"
                )

            if imbalance_ratio > 100:
                issues.append(f"SEVERE: Class imbalance ratio of {imbalance_ratio:.1f}")
                recommendations.append("Use class weights or resampling techniques")

            # Check for single-class dominance
            dominant_class_ratio = max(class_ratios.values())
            if dominant_class_ratio > 0.9:
                issues.append(
                    f"DOMINANT: One class represents {dominant_class_ratio:.1%} of data"
                )
                recommendations.append("Consider different labeling strategy")

            # Check for HOLD class issues
            hold_labels = [0, "HOLD", "hold"]
            hold_count = sum(label_distribution.get(label, 0) for label in hold_labels)
            if hold_count < 100:
                issues.append(f"HOLD_CLASS: Only {hold_count} HOLD samples")
                recommendations.append("Switch to binary classification (BUY vs SELL)")

            analysis_result = {
                "label_distribution": label_distribution,
                "class_ratios": class_ratios,
                "total_samples": total_samples,
                "unique_classes": len(unique_labels),
                "imbalance_ratio": imbalance_ratio,
                "min_class_count": min_class_count,
                "max_class_count": max_class_count,
                "dominant_class_ratio": dominant_class_ratio,
                "hold_class_count": hold_count,
                "issues": issues,
                "recommendations": recommendations,
                "severity": self._assess_label_imbalance_severity(issues),
            }

            return analysis_result

        except Exception as e:
            self.logger.error(f"❌ Error analyzing label distribution: {e}")
            return {"error": str(e)}

    def _assess_label_imbalance_severity(self, issues: List[str]) -> str:
        """Assess the severity of label imbalance issues."""
        if any("CRITICAL" in issue for issue in issues):
            return "CRITICAL"
        elif any("SEVERE" in issue for issue in issues):
            return "HIGH"
        elif any("DOMINANT" in issue for issue in issues):
            return "MODERATE"
        elif any("HOLD_CLASS" in issue for issue in issues):
            return "LOW"
        else:
            return "NONE"

    def generate_feature_recommendations(
        self, multicollinearity_analysis: Dict[str, Any]
    ) -> List[str]:
        """
        Generate specific feature engineering recommendations.

        Args:
            multicollinearity_analysis: Results from multicollinearity analysis

        Returns:
            List of recommendations
        """
        recommendations = []

        # Check for extreme VIF
        max_vif = multicollinearity_analysis.get("max_vif", 0)
        if max_vif > 1000:
            recommendations.append("🚨 CRITICAL: Extreme multicollinearity detected")
            recommendations.append(
                "   → Remove redundant price features (open, high, low, avg_price, min_price, max_price)"
            )
            recommendations.append(
                "   → Keep only 'close' and 'volume' as base features"
            )
            recommendations.append(
                "   → Engineer all other features from this minimal base set"
            )

        # Check for redundant price features
        redundant_price_features = multicollinearity_analysis.get(
            "redundant_price_features", []
        )
        if len(redundant_price_features) > 3:
            recommendations.append(
                "📊 HIGH: Multiple redundant price features detected"
            )
            recommendations.append("   → Consolidate to minimal price feature set")
            recommendations.append("   → Use only: close, volume")
            recommendations.append(
                "   → Remove: open, high, low, avg_price, min_price, max_price"
            )

        # Check for high correlation pairs
        high_correlation_pairs = multicollinearity_analysis.get(
            "high_correlation_pairs", []
        )
        if len(high_correlation_pairs) > 10:
            recommendations.append("🔗 MODERATE: Many highly correlated feature pairs")
            recommendations.append("   → Review and remove redundant features")
            recommendations.append("   → Consider feature selection techniques")

        return recommendations

    def generate_label_recommendations(
        self, label_analysis: Dict[str, Any]
    ) -> List[str]:
        """
        Generate specific labeling strategy recommendations.

        Args:
            label_analysis: Results from label distribution analysis

        Returns:
            List of recommendations
        """
        recommendations = []

        severity = label_analysis.get("severity", "NONE")
        issues = label_analysis.get("issues", [])

        if severity == "CRITICAL":
            recommendations.append(
                "🚨 CRITICAL: Label imbalance requires immediate action"
            )
            recommendations.append("   → Switch to binary classification (BUY vs SELL)")
            recommendations.append("   → Remove HOLD class entirely")
            recommendations.append(
                "   → Adjust profit_take_multiplier and stop_loss_multiplier"
            )

        elif severity == "HIGH":
            recommendations.append("📊 HIGH: Significant label imbalance detected")
            recommendations.append("   → Consider binary classification")
            recommendations.append("   → Use class weights in model training")
            recommendations.append(
                "   → Implement SMOTE or other resampling techniques"
            )

        # Specific recommendations based on issues
        for issue in issues:
            if "HOLD_CLASS" in issue:
                recommendations.append(
                    "🎯 SPECIFIC: HOLD class has insufficient samples"
                )
                recommendations.append(
                    "   → Set binary_classification=True in OptimizedTripleBarrierLabeling"
                )
                recommendations.append(
                    "   → This will filter out HOLD samples automatically"
                )

            if "imbalance ratio" in issue.lower():
                recommendations.append("⚖️ SPECIFIC: Extreme class imbalance")
                recommendations.append(
                    "   → Adjust barrier multipliers for more balanced labels"
                )
                recommendations.append("   → Consider different time horizons")

        return recommendations


def load_sample_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load sample data for demonstration purposes.
    In practice, you would load your actual data here.
    """
    # Create sample data with some NaN values to demonstrate the functionality
    dates = pd.date_range("2024-01-01", periods=1000, freq="1min")

    # Create price data with some NaN values
    np.random.seed(42)
    base_price = 100 + np.cumsum(np.random.randn(1000) * 0.1)

    price_data = pd.DataFrame(
        {
            "timestamp": dates,
            "open": base_price + np.random.randn(1000) * 0.5,
            "high": base_price + np.random.randn(1000) * 0.8,
            "low": base_price - np.random.randn(1000) * 0.8,
            "close": base_price + np.random.randn(1000) * 0.5,
            "volume": np.random.randint(100, 1000, 1000),
        }
    )

    # Introduce some NaN values for demonstration
    price_data.loc[100:105, "close"] = np.nan  # 6 NaN values
    price_data.loc[200:210, "volume"] = np.nan  # 11 NaN values
    price_data.loc[300:305, "high"] = np.nan  # 6 NaN values

    # Create volume data
    volume_data = pd.DataFrame(
        {
            "timestamp": dates,
            "volume": price_data["volume"].copy(),
            "trade_count": np.random.randint(10, 100, 1000),
        }
    )

    # Introduce some NaN values in volume data
    volume_data.loc[150:155, "trade_count"] = np.nan  # 6 NaN values

    # Set timestamp as index
    price_data.set_index("timestamp", inplace=True)
    volume_data.set_index("timestamp", inplace=True)

    return price_data, volume_data


def load_data_from_file(
    data_path: str, symbol: str, exchange: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load data from file. This is a placeholder - implement based on your data format.
    """
    try:
        # This is a placeholder implementation
        # You would implement this based on your actual data format (CSV, Parquet, etc.)

        # Example for CSV files:
        # price_file = Path(data_path) / f"{symbol}_{exchange}_price.csv"
        # volume_file = Path(data_path) / f"{symbol}_{exchange}_volume.csv"

        # if price_file.exists() and volume_file.exists():
        #     price_data = pd.read_csv(price_file, index_col='timestamp', parse_dates=True)
        #     volume_data = pd.read_csv(volume_file, index_col='timestamp', parse_dates=True)
        #     return price_data, volume_data

        print(f"Loading data for {symbol} from {exchange}...")
        print(
            "Note: This is a placeholder. Implement actual data loading based on your format."
        )

        # For now, return sample data
        return load_sample_data()

    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None


async def assess_data_quality_demo():
    """Demonstrate the enhanced data quality assessment functionality."""

    # Initialize the orchestrator with a basic config
    config = {
        "vectorized_labelling_orchestrator": {
            "enable_stationary_checks": True,
            "enable_data_normalization": True,
            "enable_lookahead_bias_handling": True,
            "enable_feature_selection": True,
            "enable_memory_efficient_types": True,
            "enable_parquet_saving": True,
            "profit_take_multiplier": 0.002,
            "stop_loss_multiplier": 0.001,
            "time_barrier_minutes": 30,
            "max_lookahead": 100,
        }
    }

    orchestrator = VectorizedLabellingOrchestrator(config)

    # Initialize the orchestrator
    print("🚀 Initializing VectorizedLabellingOrchestrator...")
    success = await orchestrator.initialize()

    if not success:
        print("❌ Failed to initialize orchestrator")
        return

    print("✅ Orchestrator initialized successfully")

    # Load sample data
    print("\n📊 Loading sample data...")
    price_data, volume_data = load_sample_data()

    if price_data is None or volume_data is None:
        print("❌ Failed to load data")
        return

    print(f"✅ Loaded price data: {price_data.shape}")
    print(f"✅ Loaded volume data: {volume_data.shape}")

    # Perform data quality assessment
    print("\n🔍 Performing data quality assessment...")
    quality_report = await orchestrator.assess_input_data_quality(
        price_data, volume_data
    )

    if "error" in quality_report:
        print(f"❌ Error in data quality assessment: {quality_report['error']}")
        return

    # Initialize enhanced analyzer
    enhanced_analyzer = EnhancedDataQualityAnalyzer()

    # Create a sample labeled dataset for demonstration
    print("\n🔍 Creating sample labeled dataset for enhanced analysis...")

    # Simulate the labeling process to create a sample dataset with labels
    sample_labeled_data = price_data.copy()

    # Add some engineered features to simulate the multicollinearity issue
    sample_labeled_data["avg_price"] = (
        sample_labeled_data["high"] + sample_labeled_data["low"]
    ) / 2
    sample_labeled_data["min_price"] = sample_labeled_data["low"]
    sample_labeled_data["max_price"] = sample_labeled_data["high"]
    sample_labeled_data["price_change"] = sample_labeled_data["close"].pct_change()
    sample_labeled_data["high_price_change"] = sample_labeled_data["high"].pct_change()
    sample_labeled_data["low_price_change"] = sample_labeled_data["low"].pct_change()
    sample_labeled_data["open_price_change"] = sample_labeled_data["open"].pct_change()

    # Create labels with imbalance to simulate the label issue
    np.random.seed(42)
    labels = np.random.choice(
        [1, -1, 0], size=len(sample_labeled_data), p=[0.45, 0.45, 0.1]
    )
    sample_labeled_data["label"] = labels

    # Analyze multicollinearity
    print("\n🔍 Analyzing multicollinearity...")
    multicollinearity_analysis = enhanced_analyzer.analyze_multicollinearity(
        sample_labeled_data
    )

    # Analyze label distribution
    print("\n🔍 Analyzing label distribution...")
    label_analysis = enhanced_analyzer.analyze_label_distribution(sample_labeled_data)

    # Generate recommendations
    feature_recommendations = enhanced_analyzer.generate_feature_recommendations(
        multicollinearity_analysis
    )
    label_recommendations = enhanced_analyzer.generate_label_recommendations(
        label_analysis
    )

    # Display the results
    print("\n" + "=" * 80)
    print("📋 ENHANCED DATA QUALITY ASSESSMENT RESULTS")
    print("=" * 80)

    # Summary
    summary = quality_report.get("summary", {})
    print(f"\n📈 BASIC DATA QUALITY SUMMARY:")
    print(f"   Total rows: {summary.get('total_rows', 'N/A')}")
    print(f"   Total cells: {summary.get('total_cells', 'N/A')}")
    print(f"   Total NaN values: {summary.get('total_nan_values', 'N/A')}")
    print(f"   NaN percentage: {summary.get('nan_percentage', 'N/A'):.2f}%")
    print(f"   Data quality score: {summary.get('data_quality_score', 'N/A'):.1f}/100")
    print(f"   Severity: {summary.get('severity', 'N/A')}")

    # Multicollinearity Analysis
    print(f"\n🔗 MULTICOLLINEARITY ANALYSIS:")
    if "error" not in multicollinearity_analysis:
        print(f"   Severity: {multicollinearity_analysis.get('severity', 'N/A')}")
        print(f"   Max VIF: {multicollinearity_analysis.get('max_vif', 'N/A'):.2f}")
        print(f"   Mean VIF: {multicollinearity_analysis.get('mean_vif', 'N/A'):.2f}")
        print(
            f"   High VIF features: {multicollinearity_analysis.get('high_vif_count', 'N/A')}"
        )
        print(
            f"   Total features: {multicollinearity_analysis.get('total_features', 'N/A')}"
        )

        if multicollinearity_analysis.get("high_vif_features"):
            print(f"   Top 5 high VIF features:")
            for i, feature in enumerate(
                multicollinearity_analysis["high_vif_features"][:5]
            ):
                vif_score = multicollinearity_analysis["vif_scores"][feature]
                print(f"     {i+1}. {feature}: VIF={vif_score:.2f}")
    else:
        print(f"   ❌ Error: {multicollinearity_analysis['error']}")

    # Label Distribution Analysis
    print(f"\n🎯 LABEL DISTRIBUTION ANALYSIS:")
    if "error" not in label_analysis:
        print(f"   Severity: {label_analysis.get('severity', 'N/A')}")
        print(f"   Total samples: {label_analysis.get('total_samples', 'N/A')}")
        print(f"   Unique classes: {label_analysis.get('unique_classes', 'N/A')}")
        print(f"   Imbalance ratio: {label_analysis.get('imbalance_ratio', 'N/A'):.2f}")
        print(f"   Min class count: {label_analysis.get('min_class_count', 'N/A')}")
        print(f"   Max class count: {label_analysis.get('max_class_count', 'N/A')}")
        print(f"   HOLD class count: {label_analysis.get('hold_class_count', 'N/A')}")

        print(f"   Label distribution:")
        for label, count in label_analysis.get("label_distribution", {}).items():
            ratio = label_analysis.get("class_ratios", {}).get(label, 0)
            print(f"     {label}: {count} samples ({ratio:.1%})")

        if label_analysis.get("issues"):
            print(f"   Issues detected:")
            for issue in label_analysis["issues"]:
                print(f"     ⚠️ {issue}")
    else:
        print(f"   ❌ Error: {label_analysis['error']}")

    # Recommendations
    print(f"\n💡 CRITICAL RECOMMENDATIONS:")

    if feature_recommendations:
        print(f"\n🔧 FEATURE ENGINEERING FIXES:")
        for rec in feature_recommendations:
            print(f"   {rec}")

    if label_recommendations:
        print(f"\n🎯 LABELING STRATEGY FIXES:")
        for rec in label_recommendations:
            print(f"   {rec}")

    # Action Plan
    print(f"\n🎯 IMMEDIATE ACTION PLAN:")
    print(f"   1. 🚨 FIX LABEL IMBALANCE FIRST (Highest Priority):")
    print(f"      → Set binary_classification=True in OptimizedTripleBarrierLabeling")
    print(f"      → This will automatically filter out HOLD samples")
    print(
        f"      → Adjust profit_take_multiplier and stop_loss_multiplier for better balance"
    )
    print(f"   2. 🔧 FIX MULTICOLLINEARITY (Second Priority):")
    print(
        f"      → Remove redundant price features: open, high, low, avg_price, min_price, max_price"
    )
    print(f"      → Keep only: close, volume as base features")
    print(f"      → Engineer all other features from this minimal base set")
    print(f"   3. 🔍 MONITOR RESULTS:")
    print(f"      → Re-run this analysis after implementing fixes")
    print(f"      → Ensure VIF < 10 and balanced label distribution")

    # Original recommendations
    recommendations = quality_report.get("recommendations", [])
    if recommendations:
        print(f"\n📋 ORIGINAL DATA QUALITY RECOMMENDATIONS:")
        for i, rec in enumerate(recommendations, 1):
            print(f"   {i}. {rec}")

    print("\n" + "=" * 80)
    print("✅ Enhanced data quality assessment completed!")
    print("=" * 80)


async def main():
    """Main function to run the enhanced data quality assessment."""
    parser = argparse.ArgumentParser(
        description="Assess data quality for trading datasets"
    )
    parser.add_argument("--data_path", type=str, help="Path to data directory")
    parser.add_argument("--symbol", type=str, default="ETHUSDT", help="Trading symbol")
    parser.add_argument("--exchange", type=str, default="binance", help="Exchange name")
    parser.add_argument(
        "--demo", action="store_true", help="Run with sample data for demonstration"
    )

    args = parser.parse_args()

    if args.demo:
        print("🎯 Running enhanced data quality assessment with sample data...")
        await assess_data_quality_demo()
    elif args.data_path:
        print(
            f"🎯 Running enhanced data quality assessment for {args.symbol} from {args.exchange}..."
        )
        # Load actual data and run assessment
        price_data, volume_data = load_data_from_file(
            args.data_path, args.symbol, args.exchange
        )
        if price_data is not None and volume_data is not None:
            # Initialize orchestrator and run assessment
            config = {
                "vectorized_labelling_orchestrator": {
                    "enable_stationary_checks": True,
                    "enable_data_normalization": True,
                    "enable_lookahead_bias_handling": True,
                    "enable_feature_selection": True,
                }
            }

            orchestrator = VectorizedLabellingOrchestrator(config)
            await orchestrator.initialize()

            quality_report = await orchestrator.assess_input_data_quality(
                price_data, volume_data
            )

            # Print results
            summary = quality_report.get("summary", {})
            print(f"\n📊 Data Quality Summary:")
            print(f"   Total NaN values: {summary.get('total_nan_values', 'N/A')}")
            print(f"   NaN percentage: {summary.get('nan_percentage', 'N/A'):.2f}%")
            print(f"   Severity: {summary.get('severity', 'N/A')}")
        else:
            print("❌ Failed to load data")
    else:
        print("Usage:")
        print("  python scripts/assess_data_quality.py --demo")
        print(
            "  python scripts/assess_data_quality.py --data_path /path/to/data --symbol ETHUSDT --exchange binance"
        )


if __name__ == "__main__":
    asyncio.run(main())
