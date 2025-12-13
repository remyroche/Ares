#!/usr/bin/env python3
"""
Pipeline Dashboard Generator

Creates a consolidated markdown dashboard aggregating metrics from:
- SNR diagnostics
- HPO reports
- Trading simulation results
- Feature selection stats

Shows deltas vs previous runs for trend analysis.
"""

import json
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
import glob
import re

from src.utils.tprint import tprint, tprint_success, tprint_error


def load_latest_report(pattern: str, outcomes_dir: Path) -> Optional[Dict[str, Any]]:
    """Load the most recent report matching a pattern."""
    candidates = sorted(outcomes_dir.glob(pattern))
    if not candidates:
        return None

    latest = candidates[-1]
    try:
        if latest.suffix == '.json':
            with open(latest, 'r') as f:
                return json.load(f)
        elif latest.suffix == '.csv':
            return pd.read_csv(latest).to_dict()
        else:
            # Try to read as markdown or text
            with open(latest, 'r') as f:
                content = f.read()
                # Extract key metrics from markdown
                return parse_markdown_metrics(content)
    except Exception as e:
        tprint(f"Warning: Failed to load {latest}: {e}", "WARNING")
        return None


def parse_markdown_metrics(content: str) -> Dict[str, Any]:
    """Extract key metrics from markdown content."""
    metrics = {}

    # Extract numeric values with labels
    patterns = [
        (r'Label coverage: (\d+\.?\d*)%', 'label_coverage'),
        (r'Positive rate: (\d+\.?\d*)%', 'pos_rate'),
        (r'SNR.*: (\d+\.?\d*)', 'snr'),
        (r'Cohen\'s d: (\d+\.?\d*)', 'cohens_d'),
        (r'AUC: (\d+\.?\d*)', 'auc'),
        (r'Learnability.*AUC: (\d+\.?\d*)', 'learnability_auc'),
        (r'Combined score: (\d+\.?\d*)', 'combined_score'),
        (r'Probe model.*AUC: (\d+\.?\d*)', 'probe_auc'),
        (r'Stability score: (\d+\.?\d*)', 'stability_score'),
        (r'Mean Return/Trade: ([\-\+]?\d+\.?\d*)%', 'mean_return_per_trade'),
        (r'Win Rate: (\d+\.?\d*)%', 'win_rate'),
        (r'Sharpe Ratio: ([\-\+]?\d+\.?\d*)', 'sharpe_ratio'),
        (r'Max Drawdown: ([\-\+]?\d+\.?\d*)%', 'max_drawdown'),
        (r'Trades/Day: (\d+\.?\d*)', 'trades_per_day'),
    ]

    for pattern, key in patterns:
        match = re.search(pattern, content, re.IGNORECASE)
        if match:
            try:
                value = float(match.group(1))
                # Convert percentages to decimals where appropriate
                if '%' in match.group(0) and key not in ['mean_return_per_trade', 'max_drawdown']:
                    value /= 100
                elif key in ['mean_return_per_trade', 'max_drawdown']:
                    value /= 100  # Convert percentage to decimal
                metrics[key] = value
            except ValueError:
                pass

    return metrics


def load_previous_run_metrics(outcomes_dir: Path, current_timestamp: str) -> Optional[Dict[str, Any]]:
    """Load metrics from the previous run for comparison."""
    # Find all dashboard files and get the second most recent
    dashboard_pattern = "pipeline_dashboard_*.md"
    dashboards = sorted(outcomes_dir.glob(dashboard_pattern))

    if len(dashboards) < 2:
        return None

    # Get second most recent (previous run)
    prev_dashboard = dashboards[-2]

    try:
        with open(prev_dashboard, 'r') as f:
            content = f.read()

        # Extract metrics from previous dashboard
        prev_metrics = {}
        metric_patterns = [
            r'Current: ([\-\+]?\d+\.?\d*)',
            r'Previous: ([\-\+]?\d+\.?\d*)',
            r'Delta: ([\-\+]?[\-\+]?\d+\.?\d*)'
        ]

        # This is a simplified extraction - in practice you'd want more robust parsing
        return parse_markdown_metrics(content)

    except Exception as e:
        tprint(f"Warning: Failed to load previous metrics: {e}", "WARNING")
        return None


def generate_dashboard_markdown(
    snr_metrics: Optional[Dict[str, Any]],
    hpo_metrics: Optional[Dict[str, Any]],
    trading_metrics: Optional[Dict[str, Any]],
    feature_metrics: Optional[Dict[str, Any]],
    prev_metrics: Optional[Dict[str, Any]]
) -> str:
    """Generate consolidated markdown dashboard."""

    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

    def format_metric(current: Optional[float], prev: Optional[float], fmt: str = ".3f", is_percent: bool = False) -> str:
        """Format metric with delta comparison."""
        if current is None:
            return "N/A"

        current_str = f"{current:{fmt}}"
        if is_percent:
            current_str += "%"

        if prev is not None and prev != 0:
            delta = current - prev
            delta_str = f"{delta:+{fmt}}"
            if is_percent:
                delta_str += "%"
            return f"{current_str} (Δ {delta_str})"
        else:
            return current_str

    dashboard = f"""# Pipeline Dashboard

**Generated:** {timestamp}

## Executive Summary

| Metric | Current | Previous | Delta |
|--------|---------|----------|-------|
"""

    # SNR Section
    if snr_metrics:
        dashboard += "\n## Signal Quality (SNR)\n\n"
        dashboard += "| Metric | Value | Status |\n"
        dashboard += "|--------|-------|--------|\n"

        snr_checks = [
            ("Label Coverage", snr_metrics.get('label_coverage'), lambda x: "✅ Good" if x and x >= 0.15 else "⚠️ Low"),
            ("Positive Rate", snr_metrics.get('pos_rate'), lambda x: "✅ Balanced" if x and 0.3 <= x <= 0.5 else "⚠️ Imbalanced"),
            ("SNR", snr_metrics.get('snr'), lambda x: "✅ Strong" if x and x >= 3.0 else "⚠️ Weak"),
            ("Cohen's d", snr_metrics.get('cohens_d'), lambda x: "✅ Large" if x and x >= 3.0 else "⚠️ Small"),
            ("Learnability AUC", snr_metrics.get('learnability_auc'), lambda x: "✅ Good" if x and x >= 0.55 else "⚠️ Poor"),
        ]

        for name, value, status_func in snr_checks:
            status = status_func(value) if value is not None else "❓ Unknown"
            value_str = f"{value:.3f}" if value is not None else "N/A"
            dashboard += f"| {name} | {value_str} | {status} |\n"

    # HPO Section
    if hpo_metrics:
        dashboard += "\n## Hyperparameter Optimization\n\n"
        dashboard += "| Stage | Best Score | Parameters |\n"
        dashboard += "|-------|------------|------------|\n"

        # Extract HPO results
        if 'layer_scores' in hpo_metrics:
            for stage, score in hpo_metrics['layer_scores'].items():
                params = hpo_metrics.get('best_params', {}).get(stage, 'N/A')
                dashboard += f"| {stage} | {score:.4f} | {str(params)[:50]}... |\n"

    # Trading Performance
    if trading_metrics:
        dashboard += "\n## Trading Performance\n\n"
        dashboard += "| Metric | Value | Target |\n"
        dashboard += "|--------|-------|--------|\n"

        trading_checks = [
            ("Trades/Day", trading_metrics.get('trades_per_day'), lambda x: "✅ Good" if x and x >= 20 else "⚠️ Low"),
            ("Mean Return/Trade", trading_metrics.get('mean_return_per_trade'), lambda x: "✅ Positive" if x and x > 0 else "❌ Negative"),
            ("Win Rate", trading_metrics.get('win_rate'), lambda x: "✅ Good" if x and x >= 0.5 else "⚠️ Low"),
            ("Sharpe Ratio", trading_metrics.get('sharpe_ratio'), lambda x: "✅ Good" if x and x >= 0.5 else "⚠️ Poor"),
            ("Max Drawdown", trading_metrics.get('max_drawdown'), lambda x: "✅ Acceptable" if x and abs(x) <= 0.30 else "⚠️ High"),
        ]

        for name, value, status_func in trading_checks:
            status = status_func(value) if value is not None else "❓ Unknown"
            if name == "Max Drawdown" and value is not None:
                value_str = f"{abs(value):.1%}"
            elif name == "Mean Return/Trade" and value is not None:
                value_str = f"{value:.2%}"
            else:
                value_str = f"{value:.3f}" if value is not None else "N/A"
            dashboard += f"| {name} | {value_str} | {status} |\n"

    # Feature Selection
    if feature_metrics:
        dashboard += "\n## Feature Selection\n\n"
        dashboard += "| Metric | Value |\n"
        dashboard += "|--------|-------|\n"

        if 'n_features_selected' in feature_metrics:
            dashboard += f"| Features Selected | {feature_metrics['n_features_selected']} |\n"
        if 'quality_score' in feature_metrics:
            dashboard += f"| Average Quality | {feature_metrics['quality_score']:.3f} |\n"
        if 'redundancy_rate' in feature_metrics:
            dashboard += f"| Redundancy Rate | {feature_metrics['redundancy_rate']:.1%} |\n"

    # Recommendations
    dashboard += "\n## Recommendations\n\n"

    recommendations = []

    if snr_metrics:
        if snr_metrics.get('snr', 0) < 3.0:
            recommendations.append("⚠️ **Improve SNR**: Consider adjusting labeling thresholds or expanding feature set")
        if snr_metrics.get('pos_rate', 0) < 0.3 or snr_metrics.get('pos_rate', 0) > 0.5:
            recommendations.append("⚠️ **Balance Labels**: Current positive rate may cause training bias")
        if snr_metrics.get('learnability_auc', 0) < 0.55:
            recommendations.append("⚠️ **Enhance Features**: Low learnability suggests missing predictive signals")

    if trading_metrics:
        if trading_metrics.get('trades_per_day', 0) < 20:
            recommendations.append("⚠️ **Increase Frequency**: Consider relaxing probability thresholds")
        if trading_metrics.get('sharpe_ratio', 0) < 0.5:
            recommendations.append("⚠️ **Reduce Risk**: Current strategy shows poor risk-adjusted returns")
        if trading_metrics.get('mean_return_per_trade', 0) <= 0:
            recommendations.append("❌ **Fix PnL**: Strategy is losing money per trade")

    if not recommendations:
        recommendations.append("✅ **All Clear**: Pipeline metrics look good for deployment")

    for rec in recommendations:
        dashboard += f"- {rec}\n"

    dashboard += "\n---\n*Generated by create_pipeline_dashboard.py*"

    return dashboard


def main():
    """Main dashboard generation function."""
    tprint("📊 Generating Pipeline Dashboard...")

    outcomes_dir = Path("outcomes")
    if not outcomes_dir.exists():
        tprint_error("Outcomes directory not found")
        return

    # Load latest reports
    snr_metrics = load_latest_report("snr_full_diagnostics_*.md", outcomes_dir)
    hpo_metrics = load_latest_report("hpo_multi_stage_best_params_*.json", outcomes_dir)
    trading_metrics = load_latest_report("meta_gated_backtest_metrics_*.json", outcomes_dir)
    feature_metrics = load_latest_report("hpo_feature_selection_*.json", outcomes_dir)

    # Get current timestamp for comparison
    current_ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    # Load previous run metrics
    prev_metrics = load_previous_run_metrics(outcomes_dir, current_ts)

    # Generate dashboard
    dashboard_content = generate_dashboard_markdown(
        snr_metrics, hpo_metrics, trading_metrics, feature_metrics, prev_metrics
    )

    # Save dashboard
    dashboard_path = outcomes_dir / f"pipeline_dashboard_{current_ts}.md"
    with open(dashboard_path, 'w') as f:
        f.write(dashboard_content)

    tprint_success(f"✅ Dashboard saved to {dashboard_path}")

    # Print summary to console
    print("\n" + "="*50)
    print("PIPELINE DASHBOARD SUMMARY")
    print("="*50)

    if snr_metrics:
        print("\n📈 Signal Quality:")
        print(".1%")
        print(".1%")
        print(".2f")

    if trading_metrics:
        print("\n💰 Trading Performance:")
        print(".1f")
        print(".2%")
        print(".1%")
        print(".2f")

    print(f"\n📄 Full dashboard: {dashboard_path}")
    print("="*50)


if __name__ == "__main__":
    main()




