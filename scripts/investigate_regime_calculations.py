#!/usr/bin/env python3
"""
Investigate Regime Calculation Issues
Analyzes why regime features may have zero variance and provides fixes.
"""


from pathlib import Path
from typing import Any, Dict, List
import sys
import traceback
import warnings

import numpy as np
import pandas as pd

from src.utils.logger import system_logger
from src.utils.error_handler import handle_errors

warnings.filterwarnings("ignore")

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))


class RegimeCalculationInvestigator:
    """Investigates regime calculation issues causing zero variance."""

    def __init__(self) -> None:
        self.logger = system_logger.getChild("RegimeCalculationInvestigator")

    def investigate_regime_calculations(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Investigate regime calculation issues."""
        self.logger.info("Investigating regime calculation issues...")

        results: Dict[str, Any] = {
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
            else:
                results.setdefault("calculation_issues", []).append(
                    {
                        "type": "missing_feature",
                        "description": f"Missing regime feature: {feature}",
                    }
                )

        # Analyze the underlying calculations
        results["calculation_issues"].extend(self._identify_calculation_issues(data))
        results["recommendations"] = self._generate_recommendations(results)

        return results

    def _analyze_regime_feature(self, series: pd.Series, feature_name: str) -> Dict[str, Any]:
        """Analyze a specific regime feature."""
        series = pd.to_numeric(series, errors="coerce")
        analysis: Dict[str, Any] = {
            "feature_name": feature_name,
            "total_values": int(len(series)),
            "unique_values": int(series.nunique(dropna=True)),
            "value_counts": {int(k): int(v) for k, v in series.value_counts(dropna=False).to_dict().items()},
            "nan_count": int(series.isna().sum()),
            "zero_count": int((series == 0).sum(skipna=True)),
            "variance": float(series.var(skipna=True)) if len(series) > 0 else 0.0,
            "issues": [],
        }

        # Check for issues
        if analysis["unique_values"] <= 1:
            analysis["issues"].append("Only one unique value - zero variance")

        if analysis["unique_values"] < 3:
            analysis["issues"].append(
                f"Low variability: only {analysis['unique_values']} unique values",
            )

        if analysis["zero_count"] > len(series) * 0.8:
            analysis["issues"].append(
                f"Too many zeros: {analysis['zero_count']} out of {len(series)}",
            )

        if analysis["nan_count"] > 0:
            analysis["issues"].append(f"Has {analysis['nan_count']} NaN values")

        return analysis

    def _identify_calculation_issues(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Identify specific calculation issues."""
        issues: List[Dict[str, Any]] = []

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

        # Trend
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

        # Volatility
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

        # Volume
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
        analysis: Dict[str, Any] = {"issues": []}

        if "close" not in data.columns:
            analysis["issues"].append("Missing 'close' column for trend calculation")
            return analysis

        close = pd.to_numeric(data["close"], errors="coerce")

        # Simulate the trend calculation
        sma_short = close.rolling(window=10, min_periods=10).mean()
        sma_long = close.rolling(window=30, min_periods=30).mean()
        trend_strength = (sma_short - sma_long).fillna(0)

        # Checks
        if trend_strength.isna().all():
            analysis["issues"].append("Trend strength is all NaN")

        if trend_strength.nunique() < 5:
            analysis["issues"].append(
                f"Trend strength has only {int(trend_strength.nunique())} unique values",
            )

        if float(trend_strength.std()) == 0.0:
            analysis["issues"].append("Trend strength has zero variance")

        analysis["trend_strength_stats"] = {
            "mean": float(trend_strength.mean()),
            "std": float(trend_strength.std()),
            "min": float(trend_strength.min()),
            "max": float(trend_strength.max()),
            "unique_count": int(trend_strength.nunique()),
        }

        return analysis

    def _analyze_volatility_calculation(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze volatility regime calculation."""
        analysis: Dict[str, Any] = {"issues": []}

        if "close" not in data.columns:
            analysis["issues"].append(
                "Missing 'close' column for volatility calculation",
            )
            return analysis

        close = pd.to_numeric(data["close"], errors="coerce")

        # Simulate the volatility calculation
        vol = close.pct_change().rolling(window=20, min_periods=20).std().fillna(0)

        # Checks
        if vol.isna().all():
            analysis["issues"].append("Volatility is all NaN")

        if vol.nunique() < 5:
            analysis["issues"].append(
                f"Volatility has only {int(vol.nunique())} unique values",
            )

        if float(vol.std()) == 0.0:
            analysis["issues"].append("Volatility has zero variance")

        analysis["volatility_stats"] = {
            "mean": float(vol.mean()),
            "std": float(vol.std()),
            "min": float(vol.min()),
            "max": float(vol.max()),
            "unique_count": int(vol.nunique()),
        }

        return analysis

    def _analyze_volume_calculation(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze volume regime calculation."""
        analysis: Dict[str, Any] = {"issues": []}

        if "volume" not in data.columns:
            analysis["issues"].append("Missing 'volume' column for volume calculation")
            return analysis

        volume = pd.to_numeric(data["volume"], errors="coerce")

        # Simulate the volume calculation
        vol_ma = volume.rolling(window=20, min_periods=20).mean()
        with np.errstate(divide="ignore", invalid="ignore"):
            volume_ratio = (volume / vol_ma.replace(0, np.nan)).fillna(0)

        # Checks
        if volume_ratio.isna().all():
            analysis["issues"].append("Volume ratio is all NaN")

        if volume_ratio.nunique() < 5:
            analysis["issues"].append(
                f"Volume ratio has only {int(volume_ratio.nunique())} unique values",
            )

        if float(volume_ratio.std()) == 0.0:
            analysis["issues"].append("Volume ratio has zero variance")

        analysis["volume_ratio_stats"] = {
            "mean": float(volume_ratio.mean()),
            "std": float(volume_ratio.std()),
            "min": float(volume_ratio.min()),
            "max": float(volume_ratio.max()),
            "unique_count": int(volume_ratio.nunique()),
        }

        return analysis

    def _generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on findings."""
        recommendations: List[str] = []

        # Analyze regime features
        for feature_name, analysis in results.get("regime_features", {}).items():
            if analysis.get("issues"):
                recommendations.append(
                    f"Fix {feature_name}: {', '.join(analysis['issues'])}",
                )

        # General recommendations
        recommendations.append("Improve regime calculation logic to handle edge cases")
        recommendations.append(
            "Add validation to ensure regime features have sufficient variability",
        )
        recommendations.append(
            "Consider using different binning strategies for regime classification",
        )
        recommendations.append(
            "Add fallback mechanisms when qcut fails due to duplicates",
        )

        return recommendations

    def generate_fixed_regime_calculations(self) -> Dict[str, str]:
        """Generate fixed regime calculation code as reference (strings)."""
        return {
            "trend_regime": (
                '''
import numpy as np
import pandas as pd

def calculate_trend_regime_fixed(price_data: pd.DataFrame) -> pd.Series:
    """Calculate trend regime with improved logic."""
    close = pd.to_numeric(price_data["close"], errors="coerce")
    sma_short = close.rolling(window=10, min_periods=10).mean()
    sma_long = close.rolling(window=30, min_periods=30).mean()
    trend_strength = (sma_short - sma_long).fillna(0)

    if trend_strength.isna().all() or float(trend_strength.std()) == 0.0:
        return pd.Series(np.zeros(len(close), dtype=int), index=price_data.index)

    try:
        trend_bins = pd.qcut(trend_strength, q=5, labels=False, duplicates="drop")
        if int(pd.Series(trend_bins).nunique()) < 3:
            trend_bins = pd.cut(trend_strength, bins=5, labels=False, include_lowest=True)
    except Exception:
        trend_bins = pd.cut(trend_strength, bins=5, labels=False, include_lowest=True)

    return pd.Series(pd.to_numeric(trend_bins, errors="coerce").fillna(0).astype(int), index=price_data.index)
                '''
            ).strip(),
            "volatility_regime": (
                '''
import numpy as np
import pandas as pd

def calculate_volatility_regime_fixed(price_data: pd.DataFrame) -> pd.Series:
    """Calculate volatility regime with improved logic."""
    close = pd.to_numeric(price_data["close"], errors="coerce")
    vol = close.pct_change().rolling(window=20, min_periods=20).std().fillna(0)

    if vol.isna().all() or float(vol.std()) == 0.0:
        return pd.Series(np.zeros(len(close), dtype=int), index=price_data.index)

    try:
        vol_bins = pd.qcut(vol, q=5, labels=False, duplicates="drop")
        if int(pd.Series(vol_bins).nunique()) < 3:
            vol_bins = pd.cut(vol, bins=5, labels=False, include_lowest=True)
    except Exception:
        vol_bins = pd.cut(vol, bins=5, labels=False, include_lowest=True)

    return pd.Series(pd.to_numeric(vol_bins, errors="coerce").fillna(0).astype(int), index=price_data.index)
                '''
            ).strip(),
            "volume_regime": (
                '''
import numpy as np
import pandas as pd

def calculate_volume_regime_fixed(volume_data: pd.DataFrame) -> pd.Series:
    """Calculate volume regime with improved logic."""
    volume = pd.to_numeric(volume_data["volume"], errors="coerce")
    vol_ma = volume.rolling(window=20, min_periods=20).mean()
    with np.errstate(divide="ignore", invalid="ignore"):
        volume_ratio = (volume / vol_ma.replace(0, np.nan)).fillna(0)

    if volume_ratio.isna().all() or float(volume_ratio.std()) == 0.0:
        return pd.Series(np.zeros(len(volume), dtype=int), index=volume_data.index)

    try:
        volreg_bins = pd.qcut(volume_ratio, q=5, labels=False, duplicates="drop")
        if int(pd.Series(volreg_bins).nunique()) < 3:
            volreg_bins = pd.cut(volume_ratio, bins=5, labels=False, include_lowest=True)
    except Exception:
        volreg_bins = pd.cut(volume_ratio, bins=5, labels=False, include_lowest=True)

    return pd.Series(pd.to_numeric(volreg_bins, errors="coerce").fillna(0).astype(int), index=volume_data.index)
                '''
            ).strip(),
        }

    def generate_report(self, results: Dict[str, Any]) -> str:
        """Generate a comprehensive investigation report."""
        report: List[str] = []
        report.append("=" * 80)
        report.append("REGIME CALCULATION INVESTIGATION REPORT")
        report.append("=" * 80)
        report.append("")

        # Regime feature analysis
        report.append("Regime Feature Analysis:")
        report.append("-" * 40)
        for feature_name, analysis in results.get("regime_features", {}).items():
            report.append(f"{feature_name.upper()}:")
            report.append(f"   - Unique values: {analysis['unique_values']}")
            report.append(f"   - Variance: {analysis['variance']:.6f}")
            report.append(f"   - Value counts: {analysis['value_counts']}")
            if analysis.get("issues"):
                report.append(f"   - Issues: {', '.join(analysis['issues'])}")
            else:
                report.append("   - No issues detected")
            report.append("")

        # Calculation issues
        if results.get("calculation_issues"):
            report.append("Calculation Issues:")
            report.append("-" * 40)
            for issue in results["calculation_issues"]:
                report.append(f"- {issue['type']}: {issue['description']}")
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
        if results.get("recommendations"):
            report.append("Recommendations:")
            report.append("-" * 40)
            for i, rec in enumerate(results["recommendations"], 1):
                report.append(f"{i}. {rec}")
            report.append("")

        # Fixed code
        report.append("Fixed Calculation Code (reference):")
        report.append("-" * 40)
        fixed_code = self.generate_fixed_regime_calculations()
        for regime_type, code in fixed_code.items():
            report.append(f"# {regime_type.upper()} FIX:")
            report.append(code)
            report.append("")

        return "\n".join(report)


@handle_errors(default_return=None, context="investigate_regime_main")
def main() -> None:
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

        data_path: str | None = None
        for path in possible_paths:
            if Path(path).exists():
                data_path = path
                break

        if data_path is None:
            print("No feature data found. Creating sample analysis...")

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
                    "trend_regime": np.zeros(n_samples),
                    "volatility_regime": np.random.choice([0, 1], size=n_samples, p=[0.8, 0.2]),
                    "volume_regime": np.random.choice([0, 1, 2], size=n_samples, p=[0.7, 0.2, 0.1]),
                    "close": price_data["close"],
                    "volume": price_data["volume"],
                }
            )
            print("Using simulated data for demonstration")
        else:
            print(f"Loading feature data from: {data_path}")
            data = (
                pd.read_parquet(data_path)
                if data_path.endswith(".parquet")
                else pd.read_csv(data_path)
            )
            print(f"Loaded data with shape: {data.shape}")

        # Run investigation
        results = investigator.investigate_regime_calculations(data)

        # Generate report
        report = investigator.generate_report(results)
        print(report)

        # Save report
        with open("regime_calculation_investigation_report.txt", "w", encoding="utf-8") as f:
            f.write(report)
        print("Report saved to: regime_calculation_investigation_report.txt")

    except Exception as e:  # noqa: BLE001 - surface full error context for debugging script
        print(f"Error during investigation: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
