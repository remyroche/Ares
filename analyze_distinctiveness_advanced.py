#!/usr/bin/env python3
"""
Advanced Feature Distinctiveness Validation for Liquidity Regimes

This script provides comprehensive analysis showing:
1. Within-Regime CoV (consistency of feature within regime)
2. Between/Within Ratio (distinctiveness from other regimes)
3. Validation against expected regime characteristics
4. Top distinguishing features for each regime pair
"""

import sys
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd

# Expected feature characteristics for each regime
REGIME_EXPECTATIONS = {
    3: {  # Ghost regime
        "name": "Ghost",
        "high_mean": ["whipsaw_count", "range_momentum_divergence", "ghost_ratio"],
        "low_mean": ["vol_momentum_sync", "trend_confirmation_6h", "volume_direction_conviction"],
        "description": "Whipsaws and false moves without momentum backing"
    },
    1: {  # Valid Trend regime
        "name": "Valid Trend",
        "high_mean": ["volume_direction_conviction", "trend_confirmation_6h", "momentum_persistence_3h"],
        "low_mean": ["whipsaw_count", "reversal_intensity", "ghost_ratio"],
        "description": "Strong directional flow with trend persistence"
    },
    2: {  # Absorption regime
        "name": "Absorption",
        "high_mean": ["reversal_intensity", "pressure_ratio", "absorption_ratio"],
        "low_mean": ["vol_momentum_sync", "ghost_ratio"],
        "description": "High participation with limited follow-through"
    },
    0: {  # Apathy regime
        "name": "Apathy",
        "high_mean": ["ghost_ratio", "intraday_close_ratio"],
        "low_mean": ["volume_direction_conviction", "momentum_persistence_3h"],
        "description": "Low signal, noisy, random-like behavior"
    }
}


def parse_per_regime_metrics(filepath: Path) -> Dict[int, Dict[str, float]]:
    """Extract per-regime metrics from markdown report."""
    per_regime = {}

    with open(filepath, 'r') as f:
        content = f.read()

    # Split by regime sections
    regime_pattern = r'### Regime (\d+)\n(.*?)(?=### Regime|\Z)'
    matches = re.finditer(regime_pattern, content, re.DOTALL)

    for match in matches:
        regime_id = int(match.group(1))
        regime_content = match.group(2)

        metrics = {}
        # Parse metric lines: "- metric_name: value"
        for line in regime_content.split('\n'):
            if line.startswith('- '):
                parts = line[2:].split(': ', 1)
                if len(parts) == 2:
                    metric_name = parts[0].strip()
                    try:
                        metric_value = float(parts[1])
                        metrics[metric_name] = metric_value
                    except ValueError:
                        pass

        if metrics:
            per_regime[regime_id] = metrics

    return per_regime


def analyze_feature_distinctiveness(per_regime: Dict[int, Dict[str, float]]) -> Dict[str, Any]:
    """
    Analyze feature distinctiveness:
    - Within-regime CoV (consistency)
    - Top features for each regime
    - Regime-pair separation
    """
    analysis = {}

    # Extract all feature names (base names without suffixes)
    all_features = set()
    for regime_metrics in per_regime.values():
        for metric_name in regime_metrics.keys():
            if metric_name.endswith(('_mean', '_std', '_cov')):
                base_name = metric_name.rsplit('_', 1)[0]
                all_features.add(base_name)

    # For each feature, analyze across regimes
    feature_analysis = {}

    for feature in sorted(all_features):
        regime_means = []
        regime_cov_within = []
        regime_data = {}

        for regime_id in sorted(per_regime.keys()):
            metrics = per_regime[regime_id]
            mean_key = f"{feature}_mean"
            cov_key = f"{feature}_cov"

            if mean_key in metrics and cov_key in metrics:
                mean_val = metrics[mean_key]
                cov_val = metrics[cov_key]
                regime_means.append(mean_val)
                regime_cov_within.append(cov_val)
                regime_data[regime_id] = {
                    'mean': mean_val,
                    'cov': cov_val
                }

        if len(regime_means) == len(per_regime):  # Feature present in all regimes
            # Compute between-regime statistics
            regime_means_arr = np.array(regime_means)
            between_std = float(np.std(regime_means_arr))
            between_mean = float(np.mean(regime_means_arr))
            between_cov = between_std / (abs(between_mean) + 1e-9)

            within_cov_avg = float(np.mean(regime_cov_within))

            # Distinctiveness = between/within
            distinctiveness = between_cov / (within_cov_avg + 1e-9)

            feature_analysis[feature] = {
                'between_cov': between_cov,
                'within_cov': within_cov_avg,
                'distinctiveness': distinctiveness,
                'regime_data': regime_data,
                'between_mean': between_mean,
                'between_std': between_std
            }

    # Sort by distinctiveness
    sorted_features = sorted(
        feature_analysis.items(),
        key=lambda x: x[1]['distinctiveness'],
        reverse=True
    )

    return {
        'all_features': sorted_features,
        'feature_analysis': feature_analysis
    }


def generate_detailed_report(
    per_regime: Dict[int, Dict[str, float]],
    distinctiveness: Dict[str, Any]
) -> str:
    """Generate comprehensive distinctiveness and validation report."""
    lines = []

    lines.append("\n" + "=" * 120)
    lines.append("ADVANCED FEATURE DISTINCTIVENESS VALIDATION")
    lines.append("ETHUSDT Liquidity Regimes - Within/Between CoV Analysis")
    lines.append("=" * 120)
    lines.append("")
    lines.append("This report shows:")
    lines.append("  1. Within-Regime CoV: How consistent each feature is within a regime")
    lines.append("  2. Between-Regime CoV: How much the feature varies across regimes")
    lines.append("  3. Distinctiveness Score: Between/Within ratio (higher = better separation)")
    lines.append("")

    # Part 1: Top Overall Features by Distinctiveness
    lines.append("\n" + "-" * 120)
    lines.append("PART 1: TOP FEATURES FOR OVERALL REGIME DISTINCTION")
    lines.append("-" * 120)
    lines.append("")
    lines.append(f"{'Rank':<4} {'Feature':<45} {'Between-CoV':<15} {'Within-CoV':<15} {'Distinctiveness':<15}")
    lines.append(f"{'-'*4} {'-'*45} {'-'*15} {'-'*15} {'-'*15}")

    for rank, (feature, analysis) in enumerate(distinctiveness['all_features'][:20], 1):
        lines.append(
            f"{rank:<4} {feature:<45} {analysis['between_cov']:<15.4f} "
            f"{analysis['within_cov']:<15.4f} {analysis['distinctiveness']:<15.4f}"
        )

    # Part 2: Per-Regime Analysis
    lines.append("\n" + "-" * 120)
    lines.append("PART 2: PER-REGIME ANALYSIS - FEATURE CONSISTENCY & DISTINCTIVENESS")
    lines.append("-" * 120)

    regime_order = [1, 3, 2, 0]  # Trend, Ghost, Absorption, Apathy

    for regime_id in regime_order:
        if regime_id not in per_regime:
            continue

        expectation = REGIME_EXPECTATIONS[regime_id]
        regime_name = expectation['name']

        lines.append("")
        lines.append(f"\n📊 REGIME {regime_id}: {regime_name.upper()}")
        lines.append(f"   Description: {expectation['description']}")
        lines.append(f"   Expected high features: {', '.join(expectation['high_mean'][:3])}")
        lines.append(f"   Expected low features: {', '.join(expectation['low_mean'][:3])}")
        lines.append("")

        # Features with lowest within-CoV (most consistent)
        lines.append(f"   ✓ Most Consistent Features (Low Within-CoV = Regime-defining):")
        lines.append(f"     {'Feature':<45} {'Within-CoV':<15} {'Mean':<15} {'Distinctiveness':<15}")
        lines.append(f"     {'-'*45} {'-'*15} {'-'*15} {'-'*15}")

        consistent_features = []
        for feature, analysis in distinctiveness['all_features']:
            if feature in distinctiveness['feature_analysis']:
                feat_data = distinctiveness['feature_analysis'][feature]
                if regime_id in feat_data['regime_data']:
                    regime_data = feat_data['regime_data'][regime_id]
                    consistent_features.append((
                        feature,
                        regime_data['cov'],
                        regime_data['mean'],
                        feat_data['distinctiveness']
                    ))

        # Sort by within-CoV (lowest first)
        consistent_features.sort(key=lambda x: x[1])

        for feature, within_cov, mean_val, distinctiveness_score in consistent_features[:10]:
            # Check if matches expectations
            is_high_expected = feature in expectation['high_mean']
            is_low_expected = feature in expectation['low_mean']

            if is_high_expected or is_low_expected:
                match_icon = "✅"
            else:
                match_icon = "  "

            lines.append(
                f"     {match_icon} {feature:<43} {within_cov:<15.4f} {mean_val:<15.6f} {distinctiveness_score:<15.4f}"
            )

        # Features with highest distinctiveness for this regime
        lines.append(f"\n   ✓ Most Distinctive Features (Best separate this regime from others):")
        lines.append(f"     {'Feature':<45} {'Distinctiveness':<15} {'Within-CoV':<15} {'Mean':<15}")
        lines.append(f"     {'-'*45} {'-'*15} {'-'*15} {'-'*15}")

        distinctive_for_regime = []
        for feature, analysis in distinctiveness['all_features']:
            if feature in distinctiveness['feature_analysis']:
                feat_data = distinctiveness['feature_analysis'][feature]
                if regime_id in feat_data['regime_data']:
                    regime_data = feat_data['regime_data'][regime_id]
                    distinctive_for_regime.append((
                        feature,
                        feat_data['distinctiveness'],
                        regime_data['cov'],
                        regime_data['mean']
                    ))

        # Sort by distinctiveness (highest first)
        distinctive_for_regime.sort(key=lambda x: x[1], reverse=True)

        for feature, distinctiveness_score, within_cov, mean_val in distinctive_for_regime[:10]:
            is_high_expected = feature in expectation['high_mean']
            is_low_expected = feature in expectation['low_mean']

            if is_high_expected or is_low_expected:
                match_icon = "✅"
            else:
                match_icon = "  "

            lines.append(
                f"     {match_icon} {feature:<43} {distinctiveness_score:<15.4f} {within_cov:<15.4f} {mean_val:<15.6f}"
            )

    # Part 3: Regime-Pair Separation
    lines.append("\n\n" + "-" * 120)
    lines.append("PART 3: REGIME-PAIR SEPARATION - WHICH FEATURES BEST DISTINGUISH PAIRS")
    lines.append("-" * 120)

    regime_order = [1, 3, 2, 0]
    for i, regime_a in enumerate(regime_order):
        for regime_b in regime_order[i+1:]:
            regime_a_name = REGIME_EXPECTATIONS[regime_a]['name']
            regime_b_name = REGIME_EXPECTATIONS[regime_b]['name']

            lines.append(f"\n{regime_a_name} vs {regime_b_name}:")
            lines.append(f"{'Feature':<45} {'Mean(A)':<15} {'Mean(B)':<15} {'Separation':<15}")
            lines.append(f"{'-'*45} {'-'*15} {'-'*15} {'-'*15}")

            pair_separations = []
            for feature, analysis in distinctiveness['all_features']:
                feat_data = distinctiveness['feature_analysis'][feature]
                if regime_a in feat_data['regime_data'] and regime_b in feat_data['regime_data']:
                    mean_a = feat_data['regime_data'][regime_a]['mean']
                    mean_b = feat_data['regime_data'][regime_b]['mean']
                    separation = abs(mean_a - mean_b)
                    pair_separations.append((feature, mean_a, mean_b, separation))

            # Sort by separation
            pair_separations.sort(key=lambda x: x[3], reverse=True)

            for feature, mean_a, mean_b, separation in pair_separations[:8]:
                lines.append(f"{feature:<45} {mean_a:<15.6f} {mean_b:<15.6f} {separation:<15.6f}")

    # Summary
    lines.append("\n\n" + "=" * 120)
    lines.append("SUMMARY & INTERPRETATION")
    lines.append("=" * 120)
    lines.append("""
Within-Regime CoV Interpretation:
  - LOW CoV (<0.3): Feature is consistent within regime → GOOD (defines regime)
  - MEDIUM CoV (0.3-0.7): Feature varies moderately → OK
  - HIGH CoV (>0.7): Feature varies widely → POOR (not regime-defining)

Between-Regime CoV Interpretation:
  - HIGH Between-CoV: Feature means differ across regimes → GOOD (distinguishes regimes)
  - LOW Between-CoV: Feature means similar across regimes → POOR (not discriminative)

Distinctiveness Score = Between-CoV / Within-CoV:
  - SCORE > 2.0: Excellent feature (consistent within, varies between)
  - SCORE 1.0-2.0: Good feature
  - SCORE < 1.0: Poor feature (varies within as much as between)

Expected Characteristics Matching (✅):
  ✅ = Feature behavior matches expectation for this regime
  ❌ = Feature behavior contradicts expectation
""")

    lines.append("")
    return "\n".join(lines)


def main():
    """Run detailed distinctiveness analysis."""
    print("=" * 120)
    print("ADVANCED FEATURE DISTINCTIVENESS VALIDATION")
    print("=" * 120)
    print()

    # Find latest quality report
    outcomes_dir = Path("outcomes")
    quality_reports = sorted(outcomes_dir.glob("liquidity_cluster_quality_ETHUSDT_*.md"))

    if not quality_reports:
        print("❌ No liquidity cluster quality report found")
        print("   Expected: outcomes/liquidity_cluster_quality_ETHUSDT_*.md")
        return

    latest_report = quality_reports[-1]
    print(f"📊 Loading quality report: {latest_report.name}")

    # Parse per-regime metrics
    print("📈 Parsing per-regime metrics...")
    per_regime = parse_per_regime_metrics(latest_report)

    if not per_regime:
        print("❌ Failed to parse per-regime metrics")
        return

    print(f"   ✓ Found {len(per_regime)} regimes with metrics")

    # Analyze distinctiveness
    print("📊 Analyzing feature distinctiveness...")
    distinctiveness = analyze_feature_distinctiveness(per_regime)
    print(f"   ✓ Analyzed {len(distinctiveness['feature_analysis'])} features")

    # Generate report
    print("📝 Generating detailed report...")
    report = generate_detailed_report(per_regime, distinctiveness)
    print(report)

    # Save report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = outcomes_dir / f"feature_distinctiveness_advanced_ETHUSDT_{timestamp}.md"
    with open(output_file, 'w') as f:
        f.write(report)

    print(f"\n✅ Detailed report saved to: {output_file}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
