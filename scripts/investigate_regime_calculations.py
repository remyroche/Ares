#!/usr/bin/env python3
"""
Investigate Regime Calculation Issues
Analyzes why regime features have zero variance and provides fixes.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import traceback
from typing import Dict, List, Tuple, Any
import warnings

warnings.filterwarnings("ignore")

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.utils.logger import system_logger


class RegimeCalculationInvestigator:
    """Investigates regime calculation issues causing zero variance."""

    def __init__(self):
        self.logger = system_logger.getChild("RegimeCalculationInvestigator")

    def investigate_regime_calculations(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Investigate regime calculation issues."""
        self.logger.info("🔍 Investigating regime calculation issues...")

        results = {
            "regime_features": {},
            "calculation_issues": [],
            "recommendations": [],
        }

        # Check regime features specifically
        regime_features = ["trend_regime", "volatility_regime", "volume_regime"]

        for feature in regime_features:
            if feature in data.columns:
                results["regime_features"][feature] = self._analyze_regime_feature(
                    data[feature], feature
                )

        # Analyze the underlying calculations
        results["calculation_issues"] = self._identify_calculation_issues(data)
        results["recommendations"] = self._generate_recommendations(results)

        return results

    def _analyze_regime_feature(
        self, series: pd.Series, feature_name: str
    ) -> Dict[str, Any]:
        """Analyze a specific regime feature."""
        analysis = {
            "feature_name": feature_name,
            "total_values": len(series),
            "unique_values": series.nunique(),
            "value_counts": series.value_counts().to_dict(),
            "nan_count": series.isna().sum(),
            "zero_count": (series == 0).sum(),
            "variance": series.var(),
            "issues": [],
        }

        # Check for issues
        if analysis["unique_values"] == 1:
            analysis["issues"].append("Only one unique value - zero variance")

        if analysis["unique_values"] < 3:
            analysis["issues"].append(
                f"Low variability: only {analysis['unique_values']} unique values"
            )

        if analysis["zero_count"] > len(series) * 0.8:
            analysis["issues"].append(
                f"Too many zeros: {analysis['zero_count']} out of {len(series)}"
            )

        if analysis["nan_count"] > 0:
            analysis["issues"].append(f"Has {analysis['nan_count']} NaN values")

        return analysis

    def _identify_calculation_issues(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Identify specific calculation issues."""
        issues = []

        # Check if we have the raw data needed for regime calculations
        required_columns = ["close", "volume"]
        missing_columns = [col for col in required_columns if col not in data.columns]

        if missing_columns:
            issues.append(
                {
                    "type": "missing_raw_data",
                    "description": f"Missing required columns for regime calculation: {missing_columns}",
                    "impact": "Cannot verify regime calculations",
                }
            )

        # Check for potential issues in regime calculations
        if "trend_regime" in data.columns:
            trend_analysis = self._analyze_trend_calculation(data)
            if trend_analysis["issues"]:
                issues.append(
                    {
                        "type": "trend_regime_issue",
                        "description": "Trend regime calculation issues",
                        "details": trend_analysis,
                    }
                )

        if "volatility_regime" in data.columns:
            vol_analysis = self._analyze_volatility_calculation(data)
            if vol_analysis["issues"]:
                issues.append(
                    {
                        "type": "volatility_regime_issue",
                        "description": "Volatility regime calculation issues",
                        "details": vol_analysis,
                    }
                )

        if "volume_regime" in data.columns:
            volume_analysis = self._analyze_volume_calculation(data)
            if volume_analysis["issues"]:
                issues.append(
                    {
                        "type": "volume_regime_issue",
                        "description": "Volume regime calculation issues",
                        "details": volume_analysis,
                    }
                )

        return issues

    def _analyze_trend_calculation(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze trend regime calculation."""
        analysis = {"issues": []}

        if "close" not in data.columns:
            analysis["issues"].append("Missing 'close' column for trend calculation")
            return analysis

        close = data["close"]

        # Simulate the trend calculation
        sma_short = close.rolling(window=10).mean()
        sma_long = close.rolling(window=30).mean()
        trend_strength = sma_short - sma_long

        # Check for issues
        if trend_strength.isna().all():
            analysis["issues"].append("Trend strength is all NaN")

        if trend_strength.nunique() < 5:
            analysis["issues"].append(
                f"Trend strength has only {trend_strength.nunique()} unique values"
            )

        # Check for constant values
        if trend_strength.std() == 0:
            analysis["issues"].append("Trend strength has zero variance")

        analysis["trend_strength_stats"] = {
            "mean": trend_strength.mean(),
            "std": trend_strength.std(),
            "min": trend_strength.min(),
            "max": trend_strength.max(),
            "unique_count": trend_strength.nunique(),
        }

        return analysis

    def _analyze_volatility_calculation(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze volatility regime calculation."""
        analysis = {"issues": []}

        if "close" not in data.columns:
            analysis["issues"].append(
                "Missing 'close' column for volatility calculation"
            )
            return analysis

        close = data["close"]

        # Simulate the volatility calculation
        vol = close.pct_change().rolling(window=20).std()

        # Check for issues
        if vol.isna().all():
            analysis["issues"].append("Volatility is all NaN")

        if vol.nunique() < 5:
            analysis["issues"].append(
                f"Volatility has only {vol.nunique()} unique values"
            )

        # Check for constant values
        if vol.std() == 0:
            analysis["issues"].append("Volatility has zero variance")

        analysis["volatility_stats"] = {
            "mean": vol.mean(),
            "std": vol.std(),
            "min": vol.min(),
            "max": vol.max(),
            "unique_count": vol.nunique(),
        }

        return analysis

    def _analyze_volume_calculation(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze volume regime calculation."""
        analysis = {"issues": []}

        if "volume" not in data.columns:
            analysis["issues"].append("Missing 'volume' column for volume calculation")
            return analysis

        volume = data["volume"]

        # Simulate the volume calculation
        vol_ma = volume.rolling(window=20).mean()
        volume_ratio = (volume / vol_ma.replace(0, np.nan)).fillna(0)

        # Check for issues
        if volume_ratio.isna().all():
            analysis["issues"].append("Volume ratio is all NaN")

        if volume_ratio.nunique() < 5:
            analysis["issues"].append(
                f"Volume ratio has only {volume_ratio.nunique()} unique values"
            )

        # Check for constant values
        if volume_ratio.std() == 0:
            analysis["issues"].append("Volume ratio has zero variance")

        analysis["volume_ratio_stats"] = {
            "mean": volume_ratio.mean(),
            "std": volume_ratio.std(),
            "min": volume_ratio.min(),
            "max": volume_ratio.max(),
            "unique_count": volume_ratio.nunique(),
        }

        return analysis

    def _generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on findings."""
        recommendations = []

        # Analyze regime features
        for feature_name, analysis in results["regime_features"].items():
            if analysis["issues"]:
                recommendations.append(
                    f"Fix {feature_name}: {', '.join(analysis['issues'])}"
                )

        # General recommendations
        recommendations.append("Improve regime calculation logic to handle edge cases")
        recommendations.append(
            "Add validation to ensure regime features have sufficient variability"
        )
        recommendations.append(
            "Consider using different binning strategies for regime classification"
        )
        recommendations.append(
            "Add fallback mechanisms when qcut fails due to duplicates"
        )

        return recommendations

    def generate_fixed_regime_calculations(self) -> Dict[str, str]:
        """Generate fixed regime calculation code."""
        return {
            "trend_regime": '''
def calculate_trend_regime_fixed(price_data: pd.DataFrame) -> pd.Series:
    """Calculate trend regime with improved logic."""
    close = price_data["close"]
    sma_short = close.rolling(window=10).mean()
    sma_long = close.rolling(window=30).mean()
    trend_strength = (sma_short - sma_long)
    
    # Handle edge cases
    if trend_strength.isna().all() or trend_strength.std() == 0:
        return pd.Series(np.zeros(len(close)), index=close.index)
    
    # Use more robust binning
    try:
        # Try qcut first
        trend_bins = pd.qcut(trend_strength.fillna(0), q=5, labels=False, duplicates="drop")
        if trend_bins.nunique() < 3:
            # Fallback to cut if qcut produces too few bins
            trend_bins = pd.cut(trend_strength.fillna(0), bins=5, labels=False, include_lowest=True)
    except Exception:
        # Final fallback
        trend_bins = pd.Series(np.zeros(len(close)), index=close.index)
    
    return trend_bins.fillna(0).astype(int)
''',
            "volatility_regime": '''
def calculate_volatility_regime_fixed(price_data: pd.DataFrame) -> pd.Series:
    """Calculate volatility regime with improved logic."""
    close = price_data["close"]
    vol = close.pct_change().rolling(window=20).std()
    
    # Handle edge cases
    if vol.isna().all() or vol.std() == 0:
        return pd.Series(np.zeros(len(close)), index=close.index)
    
    # Use more robust binning
    try:
        # Try qcut first
        vol_bins = pd.qcut(vol.fillna(0), q=5, labels=False, duplicates="drop")
        if vol_bins.nunique() < 3:
            # Fallback to cut if qcut produces too few bins
            vol_bins = pd.cut(vol.fillna(0), bins=5, labels=False, include_lowest=True)
    except Exception:
        # Final fallback
        vol_bins = pd.Series(np.zeros(len(close)), index=close.index)
    
    return vol_bins.fillna(0).astype(int)
''',
            "volume_regime": '''
def calculate_volume_regime_fixed(volume_data: pd.DataFrame) -> pd.Series:
    """Calculate volume regime with improved logic."""
    volume = volume_data["volume"]
    vol_ma = volume.rolling(window=20).mean()
    volume_ratio = (volume / vol_ma.replace(0, np.nan)).fillna(0)
    
    # Handle edge cases
    if volume_ratio.isna().all() or volume_ratio.std() == 0:
        return pd.Series(np.zeros(len(volume)), index=volume.index)
    
    # Use more robust binning
    try:
        # Try qcut first
        volreg_bins = pd.qcut(volume_ratio, q=5, labels=False, duplicates="drop")
        if volreg_bins.nunique() < 3:
            # Fallback to cut if qcut produces too few bins
            volreg_bins = pd.cut(volume_ratio, bins=5, labels=False, include_lowest=True)
    except Exception:
        # Final fallback
        volreg_bins = pd.Series(np.zeros(len(volume)), index=volume.index)
    
    return volreg_bins.fillna(0).astype(int)
''',
        }

    def generate_report(self, results: Dict[str, Any]) -> str:
        """Generate a comprehensive investigation report."""
        report = []
        report.append("=" * 80)
        report.append("REGIME CALCULATION INVESTIGATION REPORT")
        report.append("=" * 80)
        report.append("")

        # Regime feature analysis
        report.append("🔍 REGIME FEATURE ANALYSIS:")
        report.append("-" * 40)

        for feature_name, analysis in results["regime_features"].items():
            report.append(f"📊 {feature_name.upper()}:")
            report.append(f"   - Unique values: {analysis['unique_values']}")
            report.append(f"   - Variance: {analysis['variance']:.6f}")
            report.append(f"   - Value counts: {analysis['value_counts']}")

            if analysis["issues"]:
                report.append(f"   - Issues: {', '.join(analysis['issues'])}")
            else:
                report.append("   - ✅ No issues detected")
            report.append("")

        # Calculation issues
        if results["calculation_issues"]:
            report.append("🚨 CALCULATION ISSUES:")
            report.append("-" * 40)
            for issue in results["calculation_issues"]:
                report.append(f"❌ {issue['type']}: {issue['description']}")
                if "details" in issue:
                    details = issue["details"]
                    if "trend_strength_stats" in details:
                        stats = details["trend_strength_stats"]
                        report.append(
                            f"   - Trend strength: mean={stats['mean']:.6f}, std={stats['std']:.6f}, unique={stats['unique_count']}"
                        )
                    if "volatility_stats" in details:
                        stats = details["volatility_stats"]
                        report.append(
                            f"   - Volatility: mean={stats['mean']:.6f}, std={stats['std']:.6f}, unique={stats['unique_count']}"
                        )
                    if "volume_ratio_stats" in details:
                        stats = details["volume_ratio_stats"]
                        report.append(
                            f"   - Volume ratio: mean={stats['mean']:.6f}, std={stats['std']:.6f}, unique={stats['unique_count']}"
                        )
            report.append("")

        # Recommendations
        if results["recommendations"]:
            report.append("💡 RECOMMENDATIONS:")
            report.append("-" * 40)
            for i, rec in enumerate(results["recommendations"], 1):
                report.append(f"{i}. {rec}")
            report.append("")

        # Fixed code
        report.append("🔧 FIXED CALCULATION CODE:")
        report.append("-" * 40)
        fixed_code = self.generate_fixed_regime_calculations()
        for regime_type, code in fixed_code.items():
            report.append(f"# {regime_type.upper()} FIX:")
            report.append(code)
            report.append("")

        return "\n".join(report)


def main():
    """Main investigation function."""
    investigator = RegimeCalculationInvestigator()

    try:
        # Try to find feature data
        possible_paths = [
            "data_cache/features_15m.parquet",
            "data_cache/features_15m.csv",
            "data/features_15m.parquet",
            "data/features_15m.csv",
        ]

        data_path = None
        for path in possible_paths:
            if Path(path).exists():
                data_path = path
                break

        if data_path is None:
            print("❌ No feature data found. Creating sample analysis...")

            # Create sample data for demonstration
            np.random.seed(42)
            n_samples = 1000

            # Simulate price data
            price_data = pd.DataFrame(
                {
                    "close": np.cumsum(np.random.randn(n_samples) * 0.01) + 100,
                    "volume": np.random.lognormal(10, 1, n_samples),
                }
            )

            # Simulate regime features with issues
            data = pd.DataFrame(
                {
                    "trend_regime": np.zeros(n_samples),  # All zeros - zero variance
                    "volatility_regime": np.random.choice(
                        [0, 1], n_samples, p=[0.8, 0.2]
                    ),  # Low variability
                    "volume_regime": np.random.choice(
                        [0, 1, 2], n_samples, p=[0.7, 0.2, 0.1]
                    ),  # Low variability
                    "close": price_data["close"],
                    "volume": price_data["volume"],
                }
            )

            print("📊 Using simulated data for demonstration")
        else:
            print(f"📊 Loading feature data from: {data_path}")
            data = (
                pd.read_parquet(data_path)
                if data_path.endswith(".parquet")
                else pd.read_csv(data_path)
            )
            print(f"✅ Loaded data with shape: {data.shape}")

        # Run investigation
        results = investigator.investigate_regime_calculations(data)

        # Generate report
        report = investigator.generate_report(results)
        print(report)

        # Save report
        with open("regime_calculation_investigation_report.txt", "w") as f:
            f.write(report)
        print("📄 Report saved to: regime_calculation_investigation_report.txt")

    except Exception as e:
        print(f"❌ Error during investigation: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
