#!/usr/bin/env python3
"""
Analyze Impact of Strict Validation Thresholds
Analyzes the impact of new thresholds: WARNING >0.1%, ERROR >1% missing values
"""

import argparse
from collections import defaultdict
from typing import Any


def analyze_threshold_impact(log_file_path: str) -> dict[str, Any]:
    """Analyze the impact of strict thresholds on validation results"""

    # New thresholds
    NEW_WARNING_THRESHOLD=0.001  # 0.1%
    NEW_ERROR_THRESHOLD = 0.01  # 1%

    # Old thresholds (from the report)
    OLD_WARNING_THRESHOLD=0.1  # 10%
    OLD_ERROR_THRESHOLD = 0.5  # 50%

    analysis = {
        "threshold_comparison": {
            "old": {"warning": OLD_WARNING_THRESHOLD, "error": OLD_ERROR_THRESHOLD},
            "new": {"warning": NEW_WARNING_THRESHOLD, "error": NEW_ERROR_THRESHOLD},
            "impact": {
                "warning_multiplier": OLD_WARNING_THRESHOLD / NEW_WARNING_THRESHOLD,
                "error_multiplier": OLD_ERROR_THRESHOLD / NEW_ERROR_THRESHOLD,
            },
        },
        "recommendations": [],
        "affected_features": defaultdict(list),
        "timeframe_impact": {},
    }

    # Calculate impact
    warning_impact=OLD_WARNING_THRESHOLD / NEW_WARNING_THRESHOLD  # 100x more strict
    error_impact = OLD_ERROR_THRESHOLD / NEW_ERROR_THRESHOLD  # 50x more strict

    analysis["recommendations"] = [
        f"⚠️  WARNING threshold is now {warning_impact:.0f}x more strict (0.1% vs 10%)",
        f"❌  ERROR threshold is now {error_impact:.0f}x more strict (1% vs 50%)",
        "",
        "🔍 EXPECTED IMPACT:",
        "   - Features with 0.1-10% missing values: Now WARNINGS (was OK)",
        "   - Features with 1-50% missing values: Now ERRORS (was WARNINGS)",
        "   - Total validation issues will increase significantly",
        "",
        "🎯 SPECIFIC RECOMMENDATIONS:",
        "   1. Wavelet features will likely trigger many warnings",
        "   2. Multi-timeframe features may have alignment gaps",
        "   3. Technical indicators with edge effects will be flagged",
        "   4. Consider implementing feature-specific thresholds",
        "",
        "🔧 MITIGATION STRATEGIES:",
        "   1. Pre-filter features before validation",
        "   2. Use different thresholds for different feature types",
        "   3. Implement feature importance-based filtering",
        "   4. Add data quality preprocessing steps",
    ]

    return analysis


def create_feature_specific_thresholds():
    """Create feature-specific threshold recommendations"""

    return {
        "wavelet_features": {
            "missing_warning": 0.05,  # 5% - more lenient for wavelets
            "missing_error": 0.20,  # 20% - more lenient for wavelets
            "variance_threshold": 1e-12,
            "reason": "Wavelet features naturally have edge effects and low variance",
        },
        "multi_timeframe_features": {
            "missing_warning": 0.02,  # 2% - moderate tolerance
            "missing_error": 0.10,  # 10% - moderate tolerance
            "reason": "Alignment issues between timeframes can cause gaps",
        },
        "technical_indicators": {
            "missing_warning": 0.01,  # 1% - standard tolerance
            "missing_error": 0.05,  # 5% - standard tolerance
            "reason": "Technical indicators should be mostly complete",
        },
        "price_features": {
            "missing_warning": 0.001,  # 0.1% - very strict
            "missing_error": 0.01,  # 1% - very strict
            "reason": "Price data should be nearly complete",
        },
    }


def main():
    parser=argparse.ArgumentParser(description="Analyze strict validation thresholds")
    parser.add_argument("log_file", help="Path to the log file")
    parser.add_argument(
        "--output",
        default="strict_threshold_analysis.txt",
        help="Output file for analysis",
    )

    args=parser.parse_args()

    # Analyze threshold impact
    analysis=analyze_threshold_impact(args.log_file)

    # Get feature-specific thresholds
    feature_thresholds=create_feature_specific_thresholds()

    # Write analysis to file
    with open(args.output, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("STRICT VALIDATION THRESHOLD ANALYSIS\n")
        f.write("=" * 80 + "\n\n")

        # Threshold comparison
        f.write("THRESHOLD COMPARISON:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Old WARNING threshold: {analysis['threshold_comparison']['old']['warning']:.1%}\n")
        f.write(f"New WARNING threshold: {analysis['threshold_comparison']['new']['warning']:.1%}\n")
        f.write(f"Old ERROR threshold: {analysis['threshold_comparison']['old']['error']:.1%}\n")
        f.write(f"New ERROR threshold: {analysis['threshold_comparison']['new']['error']:.1%}\n")
        f.write(f"WARNING impact: {analysis['threshold_comparison']['impact']['warning_multiplier']:.0f}x more strict\n")
        f.write(f"ERROR impact: {analysis['threshold_comparison']['impact']['error_multiplier']:.0f}x more strict\n\n")

        # Recommendations
        f.write("RECOMMENDATIONS:\n")
        f.write("-" * 40 + "\n")
        f.writelines(rec + "\n" for rec in analysis["recommendations"])
        f.write("\n")

        # Feature-specific thresholds
        f.write("FEATURE-SPECIFIC THRESHOLDS:\n")
        f.write("-" * 40 + "\n")
        for feature_type, thresholds in feature_thresholds.items():
            f.write(f"\n{feature_type.upper()}:\n")
            f.write(f"  Warning threshold: {thresholds['missing_warning']:.1%}\n")
            f.write(f"  Error threshold: {thresholds['missing_error']:.1%}\n")
            f.write(f"  Reason: {thresholds['reason']}\n")
            if "variance_threshold" in thresholds:
                f.write(f"  Variance threshold: {thresholds['variance_threshold']}\n")

    print(f"Analysis written to {args.output}")


if __name__== "__main__":
    main()
