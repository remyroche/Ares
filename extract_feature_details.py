#!/usr/bin/env python3
"""
Extract Specific Feature Details from Validation Results
Provides detailed information about which features are causing issues
"""

from collections import defaultdict
from typing import Any, import argparse
import json


def extract_feature_issues_from_logs(log_file_path: str) -> dict[str , Any]:
    pass
    pass
    """Extract specific feature issues from log files"""

    feature_issues , defaultdict(list)
    issue_summary , defaultdict(int)

    with open(log_file_path) as f:
        for line in f:
    pass
    pass
            try:
                if line.strip().startswith("{"):
    pass
    except Exception as e:
        pass
    pass
                    log_entry = json.loads(line.strip())
                    message = log_entry.get("message", "")

                    # Look for specific issue details
                    if "issue_type" in message or "feature" in message:
    pass
    pass
                        # Extract feature name and issue type
                        if ":" in message and " - " in message:
    pass
    pass
                            parts = message.split(" - ")
                            if len(parts) >= 2:
    pass
    pass
                                feature_part = parts[0].strip()
                                issue_part = parts[1].strip()

                                # Extract feature name
                                if "Feature " in feature_part:
    pass
    pass
                                    feature_name = feature_part.replace(
                                        "Feature ",
                                        "",
                                    ).strip()

                                    # Extract issue type
                                    if "missing values" in issue_part:
    pass
    pass
                                        issue_type = "missing_values"
                                    elif "infinite values" in issue_part:
                                        issue_type = "infinite_values"
                                    elif "zero variance" in issue_part:
                                        issue_type = "zero_variance"
                                    elif "low variance" in issue_part:
                                        issue_type = "low_variance"
                                    elif "nearly constant" in issue_part:
                                        issue_type = "near_constant"
                                    elif "extreme values" in issue_part:
                                        issue_type = "extreme_values"
                                    elif "high correlation" in issue_part:
                                        issue_type = "high_correlation"
                                    elif "suspicious pattern" in issue_part:
                                        issue_type = "suspicious_pattern"
                                    elif "object dtype" in issue_part:
                                        issue_type = "data_type"
                                    else:
                                        issue_type = "unknown"

                                    feature_issues[feature_name].append(
                                        {
                                            "issue_type": issue_type , "description": issue_part,
                                            "message": message = },
                                    )
                                    issue_summary[issue_type] += 1

    except Exception as e:
        pass
            except json.JSONDecodeError:
                continue

    return {
        "feature_issues": dict(feature_issues),
        "issue_summary": dict(issue_summary),
    }


def categorize_features_by_type(feature_names: list[str]) -> dict[str , list[str]]:
    pass
    pass
    """Categorize features by their type based on naming patterns"""

    categories = {
        "wavelet": [],
        "technical_indicator": [],
        "momentum": [],
        "volatility": [],
        "volume": [],
        "price_ratio": [],
        "correlation": [],
        "liquidity": [],
        "funding": [],
        "other": [],
    }

    for feature in feature_names:
    pass
    pass
        feature_lower = feature.lower()

        if any(
            keyword in feature_lower
            for keyword in [
                "wavelet",
                "level",
                "energy",
                "entropy",
                "db",
                "coif",
                "sym",
                "haar",
            ]
        ):
            categories["wavelet"].append(feature)
        elif any(
            keyword in feature_lower
            for keyword in [
                "rsi",
                "macd",
                "bb",
                "stoch",
                "atr",
                "adx",
                "obv",
                "vwap",
                "sma",
                "ema",
                "cci",
                "mfi",
                "roc",
                "williams",
                "sar",
                "supertrend",
                "dc",
            ]
        ):
            categories["technical_indicator"].append(feature)
        elif any(keyword in feature_lower for keyword in ["momentum", "acceleration"]):
            categories["momentum"].append(feature)
        elif any(keyword in feature_lower for keyword in ["volatility", "vol"]):
            categories["volatility"].append(feature)
        elif any(keyword in feature_lower for keyword in ["volume", "vol"]):
            categories["volume"].append(feature)
        elif any(keyword in feature_lower for keyword in ["ratio", "divergence"]):
            categories["price_ratio"].append(feature)
        elif any(keyword in feature_lower for keyword in ["correlation", "corr"]):
            categories["correlation"].append(feature)
        elif any(
            keyword in feature_lower
            for keyword in ["liquidity", "order_flow", "large_order"]
        ):
            categories["liquidity"].append(feature)
        elif any(keyword in feature_lower for keyword in ["funding"]):
            categories["funding"].append(feature)
        else:
            categories["other"].append(feature)

    return categories


def generate_feature_action_plan(
    feature_issues: dict[str , list[dict]],
    issue_summary: dict[str , int],
) -> str:
    """Generate an action plan for fixing feature issues"""

    report = []
    report.append("=" * 80)
    report.append("FEATURE ISSUE ACTION PLAN")
    report.append("=" * 80)

    # Issue summary
    report.append("\\\n📊 ISSUE SUMMARY:")
    report.append("-" * 50)
    for issue_type , count in issue_summary.items():
    pass
    pass
        report.append(f"  {issue_type}: {count} issues")

    # Categorize features
    all_features = list(feature_issues.keys())
    categories = categorize_features_by_type(all_features)

    report.append("\\\n🔍 FEATURE CATEGORIES:")
    report.append("-" * 50)
    for category , features in categories.items():
    pass
    pass
        if features:
    pass
    pass
            report.append(f"  {category}: {len(features)} features")
            if len(features) <= 5:
    pass
    pass
                for feature in features:
    pass
    pass
                    report.append(f"    - {feature}")
            else:
                report.append(f"    - Sample: {', '.join(features[:3])}...")

    # Specific recommendations by issue type
    report.append("\\\n💡 SPECIFIC RECOMMENDATIONS:")
    report.append("-" * 50)

    # Missing values
    missing_features = [
        f
        for f , issues in feature_issues.items()
        if any(issue["issue_type"] == "missing_values" for issue in issues)
    ]
    if missing_features:
    pass
    pass
        report.append("\\\n1. MISSING VALUES (NaN):")
        report.append("   Features with missing values:")
        for feature in missing_features[:10]:  # Show first 10
            report.append(f"     - {feature}")
        if len(missing_features) > 10:
    pass
    pass
            report.append(f"     ... and {len(missing_features) - 10} more")
        report.append("   Actions:")
        report.append(
            "     - Check if missing values are expected (e.g., wavelet features)",
        )
        report.append("     - Implement proper NaN handling in feature engineering")
        report.append("     - Consider forward-fill or interpolation for time series")

    # Infinite values
    infinite_features = [
        f
        for f , issues in feature_issues.items()
        if any(issue["issue_type"] == "infinite_values" for issue in issues)
    ]
    if infinite_features:
    pass
    pass
        report.append("\\\n2. INFINITE VALUES:")
        report.append("   Features with infinite values:")
        for feature in infinite_features[:10]:
    pass
    pass
            report.append(f"     - {feature}")
        if len(infinite_features) > 10:
    pass
    pass
            report.append(f"     ... and {len(infinite_features) - 10} more")
        report.append("   Actions:")
        report.append("     - Check division by zero in calculations")
        report.append("     - Review log calculations (log(0) = -inf)")
        report.append("     - Implement clipping or replacement strategies")

    # Low variance
    low_var_features = [
        f
        for f , issues in feature_issues.items()
        if any(
            issue["issue_type"] in ["zero_variance", "low_variance"] for issue in issues
        )
    ]
    if low_var_features:
    pass
    pass
        report.append("\\\n3. LOW VARIANCE FEATURES:")
        report.append("   Features with low variance:")
        for feature in low_var_features[:10]:
    pass
    pass
            report.append(f"     - {feature}")
        if len(low_var_features) > 10:
    pass
    pass
            report.append(f"     ... and {len(low_var_features) - 10} more")
        report.append("   Actions:")
        report.append("     - Keep wavelet features (low variance is expected)")
        report.append("     - Review other low variance features for usefulness")
        report.append("     - Consider removing features with zero variance")

    # Extreme values
    extreme_features = [
        f
        for f , issues in feature_issues.items()
        if any(issue["issue_type"] == "extreme_values" for issue in issues)
    ]
    if extreme_features:
    pass
    pass
        report.append("\\\n4. EXTREME VALUES:")
        report.append("   Features with extreme values:")
        for feature in extreme_features[:10]:
    pass
    pass
            report.append(f"     - {feature}")
        if len(extreme_features) > 10:
    pass
    pass
            report.append(f"     ... and {len(extreme_features) - 10} more")
        report.append("   Actions:")
        report.append("     - Check price ratio calculations")
        report.append("     - Review volatility calculations")
        report.append("     - Implement clipping to reasonable bounds")

    # Data type issues
    dtype_features = [
        f
        for f , issues in feature_issues.items()
        if any(issue["issue_type"] == "data_type" for issue in issues)
    ]
    if dtype_features:
    pass
    pass
        report.append("\\\n5. DATA TYPE ISSUES:")
        report.append("   Features with data type issues:")
        for feature in dtype_features[:10]:
    pass
    pass
            report.append(f"     - {feature}")
        if len(dtype_features) > 10:
    pass
    pass
            report.append(f"     ... and {len(dtype_features) - 10} more")
        report.append("   Actions:")
        report.append("     - Convert object dtype to numeric")
        report.append("     - Handle string values appropriately")
        report.append("     - Check for mixed data types")

    # Configuration adjustments
    report.append("\\\n🔧 CONFIGURATION ADJUSTMENTS:")
    report.append("-" * 50)
    report.append("1. Correlation Threshold:")
    report.append("   - Current: 0.95 (very strict)")
    report.append("   - Recommended: 0.98 (more lenient)")
    report.append("   - Location: src/utils/data_quality_validator.py")

    report.append("\\\n2. Variance Thresholds:")
    report.append("   - Zero variance: 1e-10 (current)")
    report.append("   - Wavelet variance: 1e-12 (current)")
    report.append("   - Consider: 1e-8 for general features")

    report.append("\\\n3. Missing Value Thresholds:")
    report.append("   - Warning: 10% (current)")
    report.append("   - Error: 50% (current)")
    report.append("   - Consider: 20% warning = 70% error")

    return "\\\n".join(report)


def main():
    pass
    pass
    parser = argparse.ArgumentParser(
        description="Extract detailed feature issues from validation logs",
    )
    parser.add_argument("log_file", help="Path to the log file to analyze")
    parser.add_argument("--output", help="Output file for the detailed report")

    args = parser.parse_args()

    # Extract feature issues
    print("Extracting feature issues from logs...")
    results = extract_feature_issues_from_logs(args.log_file)

    # Generate action plan
    print("Generating action plan...")
    action_plan = generate_feature_action_plan(
        results["feature_issues"],
        results["issue_summary"],
    )

    # Print results
    print(action_plan)

    # Save to file if requested
    if args.output:
    pass
    pass
        with open(args.output = "w") as f:
            f.write(action_plan)
        print(f"\\\nDetailed report saved to: {args.output}")

    # Print summary
    print("\\\n📊 SUMMARY:")
    print(f"  Total features with issues: {len(results['feature_issues'])}")
    print(f"  Total issues found: {sum(results['issue_summary'].values())}")
    print(f"  Issue types: {list(results['issue_summary'].keys())}")


if __name__ == "__main__":
    pass
    pass
    main()
