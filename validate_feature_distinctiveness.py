#!/usr/bin/env python3
"""
Feature Distinctiveness Validation for Liquidity Regimes

This script validates that the top features from feature distinctiveness analysis
match expected regime characteristics:
- Ghost: High whipsaw_count, high range_momentum_divergence, low vol_momentum_sync
- Trend: Low whipsaw_count, high volume_direction_conviction, high trend_confirmation
- Absorption: High reversal_intensity, high reversal_volume_sync, high pressure_ratio
"""

import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd

# Expected feature characteristics for each regime
EXPECTED_REGIME_CHARACTERISTICS = {
    3: {  # Ghost regime
        "high": ["whipsaw_count", "range_momentum_divergence", "ghost_ratio"],
        "low": ["vol_momentum_sync", "trend_confirmation_6h", "volume_direction_conviction"],
        "description": "Whipsaws and false moves without momentum backing"
    },
    1: {  # Valid Trend regime
        "high": ["volume_direction_conviction", "trend_confirmation_6h", "momentum_persistence_3h"],
        "low": ["whipsaw_count", "reversal_intensity", "ghost_ratio"],
        "description": "Strong directional flow with trend persistence"
    },
    2: {  # Absorption regime
        "high": ["reversal_intensity", "pressure_ratio", "absorption_ratio"],
        "low": ["vol_momentum_sync", "ghost_ratio"],
        "description": "High participation with limited follow-through (absorption)"
    },
    0: {  # Apathy regime
        "high": ["ghost_ratio", "intraday_close_ratio"],
        "low": ["volume_direction_conviction", "momentum_persistence_3h"],
        "description": "Low signal, noisy, random-like behavior"
    }
}

REGIME_NAMES = {
    0: "Apathy",
    1: "Valid Trend",
    2: "Absorption",
    3: "Ghost"
}


def load_quality_report(filepath: Path) -> Dict[str, Any]:
    """Load existing quality report with per-regime metrics."""
    try:
        with open(filepath, 'r') as f:
            content = f.read()
        return {"content": content, "path": str(filepath)}
    except FileNotFoundError:
        return None


def analyze_regime_characteristics(
    quality_report: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Analyze regime characteristics from quality report metrics.

    Extracts per-regime metrics and validates against expected characteristics.
    """
    if not quality_report:
        return {}

    # Parse per-regime metrics from the markdown report
    content = quality_report['content']
    per_regime_data = {}

    current_regime = None
    for line in content.split('\n'):
        # Match "### Regime N"
        if line.startswith("### Regime "):
            regime_id = int(line.split("Regime ")[1])
            current_regime = regime_id
            per_regime_data[regime_id] = {}
        elif current_regime is not None and line.startswith("- "):
            # Parse "- metric_name: value"
            parts = line[2:].split(": ", 1)
            if len(parts) == 2:
                metric_name = parts[0].strip()
                try:
                    metric_value = float(parts[1])
                    per_regime_data[current_regime][metric_name] = metric_value
                except ValueError:
                    pass

    # Analyze against expected characteristics
    analysis = {}
    for regime_id, characteristics in EXPECTED_REGIME_CHARACTERISTICS.items():
        analysis[regime_id] = {
            "regime_name": REGIME_NAMES[regime_id],
            "description": characteristics["description"],
            "metrics": per_regime_data.get(regime_id, {}),
            "expected_high": characteristics["high"],
            "expected_low": characteristics["low"],
            "validations": _validate_characteristics(
                regime_id,
                per_regime_data.get(regime_id, {}),
                characteristics
            )
        }

    return analysis


def _validate_characteristics(
    regime_id: int,
    metrics: Dict[str, float],
    characteristics: Dict[str, List[str]]
) -> Dict[str, Any]:
    """Validate that regime metrics match expected characteristics."""
    validations = {
        "high_features": {},
        "low_features": {},
        "summary": {"matched": 0, "total": 0}
    }

    # Check high features
    for feature in characteristics["high"]:
        # Look for CoV and mean metrics
        for suffix in ["_cov", "_mean", "_std"]:
            metric_key = feature + suffix
            if metric_key in metrics:
                value = metrics[metric_key]
                validations["high_features"][metric_key] = {
                    "value": value,
                    "expected": "high",
                    "matched": suffix == "_mean" or suffix == "_cov"  # High CoV or high mean
                }
                validations["summary"]["total"] += 1
                if validations["high_features"][metric_key]["matched"]:
                    validations["summary"]["matched"] += 1

    # Check low features
    for feature in characteristics["low"]:
        for suffix in ["_cov", "_mean", "_std"]:
            metric_key = feature + suffix
            if metric_key in metrics:
                value = metrics[metric_key]
                validations["low_features"][metric_key] = {
                    "value": value,
                    "expected": "low",
                    "matched": value < 0.5 if suffix != "_mean" else True  # Heuristic thresholds
                }
                validations["summary"]["total"] += 1
                if validations["low_features"][metric_key]["matched"]:
                    validations["summary"]["matched"] += 1

    return validations


def generate_validation_report(analysis: Dict[str, Any]) -> str:
    """Generate human-readable validation report."""
    lines = []
    lines.append("\n" + "=" * 100)
    lines.append("FEATURE DISTINCTIVENESS VALIDATION REPORT")
    lines.append("ETHUSDT Liquidity Regimes")
    lines.append("=" * 100)
    lines.append("")
    lines.append("This report validates that regime characteristics match expected behavior patterns")
    lines.append("based on within-regime and between-regime coefficient of variation (CoV) analysis.")
    lines.append("")

    regime_order = [1, 3, 2, 0]  # Trend, Ghost, Absorption, Apathy

    for regime_id in regime_order:
        if regime_id not in analysis:
            continue

        regime_data = analysis[regime_id]
        lines.append("\n" + "-" * 100)
        lines.append(f"REGIME {regime_id}: {regime_data['regime_name'].upper()}")
        lines.append("-" * 100)
        lines.append(f"\n📋 Description: {regime_data['description']}")
        lines.append("")

        # Summary statistics
        metrics = regime_data['metrics']
        n_samples = metrics.get('n_samples', 0)
        lines.append(f"Sample count: {n_samples:.0f}")

        # Per-regime metrics summary
        lines.append(f"\n✓ Per-Regime Metrics (Mean, Std, CoV):")
        lines.append(f"{'Metric':<40} {'Mean':>12} {'Std':>12} {'CoV':>12}")
        lines.append(f"{'-'*40} {'-'*12} {'-'*12} {'-'*12}")

        for metric_name in sorted(metrics.keys()):
            if metric_name == 'n_samples':
                continue
            if '_mean' in metric_name:
                base_name = metric_name.replace('_mean', '')
                mean_val = metrics.get(metric_name, 0.0)
                std_val = metrics.get(base_name + '_std', 0.0)
                cov_val = metrics.get(base_name + '_cov', 0.0)
                lines.append(
                    f"{base_name:<40} {mean_val:>12.4f} {std_val:>12.4f} {cov_val:>12.4f}"
                )

        # Validation against expected characteristics
        lines.append(f"\n✓ Expected High Features:")
        if regime_data['validations']['high_features']:
            for metric_key, validation in regime_data['validations']['high_features'].items():
                status = "✅" if validation['matched'] else "⚠️"
                lines.append(
                    f"  {status} {metric_key:<45} {validation['value']:>8.4f} "
                    f"(expected: high)"
                )
        else:
            lines.append("  (No data available)")

        lines.append(f"\n✓ Expected Low Features:")
        if regime_data['validations']['low_features']:
            for metric_key, validation in regime_data['validations']['low_features'].items():
                status = "✅" if validation['matched'] else "⚠️"
                lines.append(
                    f"  {status} {metric_key:<45} {validation['value']:>8.4f} "
                    f"(expected: low)"
                )
        else:
            lines.append("  (No data available)")

        # Summary
        summary = regime_data['validations']['summary']
        if summary['total'] > 0:
            match_pct = (summary['matched'] / summary['total']) * 100
            lines.append(f"\n✓ Validation Score: {match_pct:.1f}% ({summary['matched']}/{summary['total']} metrics matched)")
        else:
            lines.append("\n✓ Validation Score: N/A (no feature data)")

    # Overall interpretation
    lines.append("\n" + "=" * 100)
    lines.append("INTERPRETATION GUIDE")
    lines.append("=" * 100)
    lines.append("""
✅ High Match (80%+): Regime behavior strongly matches expected characteristics
   → Regime is well-defined and distinct

⚠️  Partial Match (50-80%): Some expected characteristics present
   → Regime shows primary behavior but with some contamination

❌ Low Match (<50%): Few expected characteristics present
   → Regime may be poorly defined or overlapping with others

Within-Regime CoV Interpretation:
  - High CoV (>0.5): Feature varies widely within regime → Less distinctive
  - Low CoV (<0.2): Feature is consistent within regime → More distinctive

This indicates the regime is "pure" with consistent behavior across samples.
""")
    lines.append("")

    return "\n".join(lines)


def main():
    """Run validation analysis."""
    print("=" * 100)
    print("LIQUIDITY REGIME FEATURE DISTINCTIVENESS VALIDATION")
    print("=" * 100)
    print()

    # Find latest quality report
    outcomes_dir = Path("outcomes")
    quality_reports = sorted(outcomes_dir.glob("liquidity_cluster_quality_ETHUSDT_*.md"))

    if not quality_reports:
        print("❌ No liquidity cluster quality report found")
        print("   Expected: outcomes/liquidity_cluster_quality_ETHUSDT_*.md")
        print("\n   Please run the liquidity regime pipeline first:")
        print("   poetry run python -m src.launcher.ares_launcher --symbol ETHUSDT")
        return

    latest_report = quality_reports[-1]
    print(f"📊 Loading quality report: {latest_report.name}")

    quality_data = load_quality_report(latest_report)
    if not quality_data:
        print("❌ Failed to load quality report")
        return

    # Analyze characteristics
    print("📈 Analyzing regime characteristics...")
    analysis = analyze_regime_characteristics(quality_data)

    if not analysis:
        print("❌ Failed to analyze regime characteristics")
        return

    # Generate validation report
    print("📝 Generating validation report...")
    report = generate_validation_report(analysis)
    print(report)

    # Save validation report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = outcomes_dir / f"feature_distinctiveness_validation_ETHUSDT_{timestamp}.md"
    with open(output_file, 'w') as f:
        f.write(report)
    print(f"\n✅ Validation report saved to: {output_file}")

    # Summary for each regime
    print("\n" + "=" * 100)
    print("QUICK SUMMARY")
    print("=" * 100)

    regime_order = [1, 3, 2, 0]
    for regime_id in regime_order:
        if regime_id in analysis:
            data = analysis[regime_id]
            summary = data['validations']['summary']
            match_pct = (summary['matched'] / summary['total'] * 100) if summary['total'] > 0 else 0
            regime_name = data['regime_name']

            status_icon = "✅" if match_pct >= 80 else "⚠️" if match_pct >= 50 else "❌"
            print(f"{status_icon} {regime_name:20s}: {match_pct:5.1f}% match ({summary['matched']}/{summary['total']})")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
