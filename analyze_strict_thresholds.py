#!/usr/bin/env python3
"""
Analyze Impact of Strict Validation Thresholds
Analyzes the impact of new thresholds: WARNING >0.1%, ERROR >1% missing values
"""

import json
import re
import pandas as pd
from collections import defaultdict
from typing import Dict, List, Any
import argparse
from pathlib import Path


def analyze_threshold_impact(log_file_path: str) -> Dict[str, Any]:
    """Analyze the impact of strict thresholds on validation results"""

    # New thresholds
    NEW_WARNING_THRESHOLD = 0.001  # 0.1%
    NEW_ERROR_THRESHOLD = 0.01  # 1%

    # Old thresholds (from the report)
    OLD_WARNING_THRESHOLD = 0.1  # 10%
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
    warning_impact = OLD_WARNING_THRESHOLD / NEW_WARNING_THRESHOLD  # 100x more strict
    error_impact = OLD_ERROR_THRESHOLD / NEW_ERROR_THRESHOLD  # 50x more strict

    analysis["recommendations"] = [
        f"⚠️  WARNING threshold is now {warning_impact:.0f}x more strict (0.1% vs 10%)",
        f"❌  ERROR threshold is now {error_impact:.0f}x more strict (1% vs 50%)",
        "",
        "🔍 EXPECTED IMPACT:",
        f"   - Features with 0.1-10% missing values: Now WARNINGS (was OK)",
        f"   - Features with 1-50% missing values: Now ERRORS (was WARNINGS)",
        f"   - Total validation issues will increase significantly",
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

    thresholds = {
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

    return thresholds


def main():
    parser = argparse.ArgumentParser(description="Analyze strict validation thresholds")
    parser.add_argument("log_file", help="Path to the log file")
    parser.add_argument(
        "--output",
        default="strict_threshold_analysis.txt",
        help="Output file for analysis",
    )

    args = parser.parse_args()

    # Analyze threshold impact
    analysis = analyze_threshold_impact(args.log_file)
    feature_thresholds = create_feature_specific_thresholds()

    # Write analysis report
    with open(args.output, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("STRICT THRESHOLD IMPACT ANALYSIS\n")
        f.write("=" * 80 + "\n\n")

        f.write("📊 THRESHOLD COMPARISON:\n")
        f.write("-" * 40 + "\n")
        f.write(
            f"OLD WARNING: {analysis['threshold_comparison']['old']['warning']*100:.1f}%\n"
        )
        f.write(
            f"NEW WARNING: {analysis['threshold_comparison']['new']['warning']*100:.1f}%\n"
        )
        f.write(
            f"OLD ERROR:   {analysis['threshold_comparison']['old']['error']*100:.1f}%\n"
        )
        f.write(
            f"NEW ERROR:   {analysis['threshold_comparison']['new']['error']*100:.1f}%\n\n"
        )

        f.write("🚨 IMPACT MULTIPLIERS:\n")
        f.write("-" * 40 + "\n")
        f.write(
            f"WARNING: {analysis['threshold_comparison']['impact']['warning_multiplier']:.0f}x more strict\n"
        )
        f.write(
            f"ERROR:   {analysis['threshold_comparison']['impact']['error_multiplier']:.0f}x more strict\n\n"
        )

        f.write("💡 RECOMMENDATIONS:\n")
        f.write("-" * 40 + "\n")
        for rec in analysis["recommendations"]:
            f.write(f"{rec}\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("FEATURE-SPECIFIC THRESHOLD RECOMMENDATIONS\n")
        f.write("=" * 80 + "\n\n")

        for feature_type, config in feature_thresholds.items():
            f.write(f"🔧 {feature_type.upper().replace('_', ' ')}:\n")
            f.write(f"   Warning: {config['missing_warning']*100:.1f}%\n")
            f.write(f"   Error:   {config['missing_error']*100:.1f}%\n")
            if "variance_threshold" in config:
                f.write(f"   Variance: {config['variance_threshold']}\n")
            f.write(f"   Reason:  {config['reason']}\n\n")

        f.write("🎯 IMPLEMENTATION STRATEGY:\n")
        f.write("-" * 40 + "\n")
        f.write("1. Implement feature-type detection in validation\n")
        f.write("2. Apply different thresholds based on feature type\n")
        f.write("3. Log which threshold was applied to each feature\n")
        f.write("4. Consider feature importance in final filtering\n")
        f.write("5. Monitor false positive rates with new thresholds\n")

    print(f"✅ Strict threshold analysis written to: {args.output}")
    print(
        f"📊 Key finding: WARNING threshold is now {analysis['threshold_comparison']['impact']['warning_multiplier']:.0f}x more strict"
    )
    print(
        f"📊 Key finding: ERROR threshold is now {analysis['threshold_comparison']['impact']['error_multiplier']:.0f}x more strict"
    )


if __name__ == "__main__":
    main()
