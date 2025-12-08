"""
Winning Feature Set Selector.

This module determines the winning feature set based on three key metrics:
1. Learnability (from feature_generation_meta_labeling_step) - compute_learnability_with_calibration
2. Generalization Gap (from snr_diagnostics) - avoiding overfitting
3. Risk-Adjusted Returns (from meta_gated_backtest) - Gated Sharpe Ratio (MOST IMPORTANT)

The winning set is persisted for use by Analyst Base models as feature_set B.

Created: 2025-12-08
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.utils.tprint import tprint, tprint_info, tprint_success, tprint_warning, tprint_error

logger = logging.getLogger(__name__)


# Winning Metrics Weights
# These weights determine how much each metric contributes to the final score
METRIC_WEIGHTS = {
    # Risk-adjusted returns (Gated Sharpe) - MOST IMPORTANT (50%)
    "gated_sharpe": 0.50,
    # Learnability (AUC-based) - Important for model quality (25%)
    "learnability": 0.25,
    # Generalization gap penalty - Lower is better, penalize overfitting (25%)
    "generalization_gap_penalty": 0.25,
}

# Metric normalization ranges (for scaling to 0-1)
METRIC_RANGES = {
    "gated_sharpe": {"min": -1.0, "max": 3.0},  # Typical Sharpe range
    "learnability": {"min": 0.5, "max": 0.75},  # AUC range (0.5 = random, 0.75+ = good)
    "generalization_gap": {"min": 0.0, "max": 0.15},  # Gap as fraction
}


def normalize_metric(value: float, metric_name: str, higher_is_better: bool = True) -> float:
    """
    Normalize a metric value to 0-1 range.
    
    Args:
        value: Raw metric value
        metric_name: Name of the metric (for range lookup)
        higher_is_better: If True, higher values get higher scores
        
    Returns:
        Normalized score between 0 and 1
    """
    if metric_name not in METRIC_RANGES:
        return value  # Return as-is if no range defined
    
    range_info = METRIC_RANGES[metric_name]
    min_val = range_info["min"]
    max_val = range_info["max"]
    
    # Clamp to range
    clamped = max(min_val, min(max_val, value))
    
    # Normalize to 0-1
    if max_val == min_val:
        normalized = 0.5
    else:
        normalized = (clamped - min_val) / (max_val - min_val)
    
    # Invert if lower is better
    if not higher_is_better:
        normalized = 1.0 - normalized
    
    return normalized


def compute_composite_score(
    gated_sharpe: float,
    learnability: float,
    generalization_gap: float,
    mean_return: Optional[float] = None,
    hit_rate: Optional[float] = None,
    trade_frequency: Optional[float] = None,
) -> Tuple[float, Dict[str, float]]:
    """
    Compute composite score for a feature set based on winning metrics.
    
    The score is a weighted combination of:
    - Gated Sharpe Ratio (50%) - Risk-adjusted returns, MOST IMPORTANT
    - Learnability (25%) - Model's ability to learn patterns
    - Generalization Gap Penalty (25%) - Penalize overfitting
    
    Optional secondary factors (used as tie-breakers):
    - Mean Return - Higher is better
    - Hit Rate - Higher is better
    - Trade Frequency - Reasonable frequency preferred
    
    Args:
        gated_sharpe: Gated Sharpe ratio from meta_gated_backtest
        learnability: Learnability score (AUC-based) from meta_labeling
        generalization_gap: Train-test AUC gap from snr_diagnostics
        mean_return: Optional mean return per trade
        hit_rate: Optional win rate
        trade_frequency: Optional trades per day
        
    Returns:
        Tuple of (composite_score, score_breakdown)
    """
    # Normalize metrics
    sharpe_score = normalize_metric(gated_sharpe, "gated_sharpe", higher_is_better=True)
    learn_score = normalize_metric(learnability, "learnability", higher_is_better=True)
    gap_score = normalize_metric(generalization_gap, "generalization_gap", higher_is_better=False)
    
    # Compute weighted score
    composite = (
        METRIC_WEIGHTS["gated_sharpe"] * sharpe_score +
        METRIC_WEIGHTS["learnability"] * learn_score +
        METRIC_WEIGHTS["generalization_gap_penalty"] * gap_score
    )
    
    # Build breakdown
    breakdown = {
        "gated_sharpe_raw": gated_sharpe,
        "gated_sharpe_normalized": sharpe_score,
        "gated_sharpe_weighted": METRIC_WEIGHTS["gated_sharpe"] * sharpe_score,
        "learnability_raw": learnability,
        "learnability_normalized": learn_score,
        "learnability_weighted": METRIC_WEIGHTS["learnability"] * learn_score,
        "generalization_gap_raw": generalization_gap,
        "generalization_gap_normalized": gap_score,
        "generalization_gap_weighted": METRIC_WEIGHTS["generalization_gap_penalty"] * gap_score,
        "composite_score": composite,
    }
    
    # Add optional metrics if provided
    if mean_return is not None:
        breakdown["mean_return"] = mean_return
    if hit_rate is not None:
        breakdown["hit_rate"] = hit_rate
    if trade_frequency is not None:
        breakdown["trade_frequency"] = trade_frequency
    
    return composite, breakdown


def load_feature_set_metrics(
    exchange: str,
    asset: str,
    timeframe: str,
    feature_set_size: int,
) -> Dict[str, Any]:
    """
    Load metrics for a specific feature set from saved results.
    
    This function looks for metrics in:
    1. multi_feature_set_results (from feature_generation_meta_labeling_step)
    2. feature_set_comparison (from snr_diagnostics)
    3. meta_gated_backtest reports
    
    Args:
        exchange: Exchange name
        asset: Asset symbol
        timeframe: Timeframe string
        feature_set_size: Feature set size (50, 60, 70, 80)
        
    Returns:
        Dictionary with metrics for the feature set
    """
    import glob
    
    metrics = {
        "feature_set_size": feature_set_size,
        "gated_sharpe": None,
        "learnability": None,
        "generalization_gap": None,
        "mean_return": None,
        "hit_rate": None,
        "trade_frequency": None,
    }
    
    outcomes_dir = Path("outcomes")
    
    # Load from feature_set_comparison (snr_diagnostics)
    try:
        pattern = f"feature_set_comparison_{asset}_{timeframe}_*.json"
        files = sorted(glob.glob(str(outcomes_dir / pattern)), reverse=True)
        if files:
            with open(files[0], 'r') as f:
                data = json.load(f)
            
            comparison_results = data.get("comparison_results", {})
            size_key = str(feature_set_size)
            if size_key in comparison_results:
                fs_metrics = comparison_results[size_key]
                if "mean_auc" in fs_metrics:
                    metrics["learnability"] = fs_metrics["mean_auc"]
                if "generalization_gap" in fs_metrics:
                    metrics["generalization_gap"] = fs_metrics["generalization_gap"]
    except Exception as e:
        logger.warning(f"Failed to load snr_diagnostics metrics: {e}")
    
    # Load from multi_feature_set_results (feature_generation_meta_labeling_step)
    try:
        pattern = f"multi_feature_set_results_{asset}_{timeframe}_*.json"
        files = sorted(glob.glob(str(outcomes_dir / pattern)), reverse=True)
        if files:
            with open(files[0], 'r') as f:
                data = json.load(f)
            
            results = data.get("results", {})
            size_key = str(feature_set_size)
            if size_key in results:
                fs_data = results[size_key]
                # Learnability might be stored here too
                if metrics["learnability"] is None and "learnability" in fs_data:
                    metrics["learnability"] = fs_data["learnability"]
    except Exception as e:
        logger.warning(f"Failed to load meta_labeling results: {e}")
    
    # Load from meta_gated_backtest reports
    try:
        pattern = f"meta_gated_backtest_{asset}_{timeframe}_*.json"
        files = sorted(glob.glob(str(outcomes_dir / pattern)), reverse=True)
        if files:
            with open(files[0], 'r') as f:
                data = json.load(f)
            
            backtest_metrics = data.get("metrics", {})
            if "sharpe_trade" in backtest_metrics:
                metrics["gated_sharpe"] = backtest_metrics["sharpe_trade"]
            if "mean_return_gated" in backtest_metrics:
                metrics["mean_return"] = backtest_metrics["mean_return_gated"]
            if "hit_rate_gated" in backtest_metrics:
                metrics["hit_rate"] = backtest_metrics["hit_rate_gated"]
            if "trades_per_day" in backtest_metrics:
                metrics["trade_frequency"] = backtest_metrics["trades_per_day"]
    except Exception as e:
        logger.warning(f"Failed to load meta_gated_backtest metrics: {e}")
    
    return metrics


def determine_winning_feature_set(
    exchange: str,
    asset: str,
    timeframe: str,
    feature_set_sizes: List[int] = [50, 60, 70, 80],
    persist: bool = True,
) -> Tuple[int, Dict[str, Any]]:
    """
    Determine the winning feature set based on all available metrics.
    
    This function:
    1. Loads metrics for each feature set size
    2. Computes composite scores
    3. Selects the winner
    4. Optionally persists the winning set
    
    Args:
        exchange: Exchange name
        asset: Asset symbol
        timeframe: Timeframe string
        feature_set_sizes: List of feature set sizes to compare
        persist: Whether to persist the winning set
        
    Returns:
        Tuple of (winning_size, comparison_results)
    """
    tprint_info(f"🏆 Determining winning feature set for {asset}/{exchange} [{timeframe}]")
    
    comparison_results = {}
    scores = {}
    
    for size in feature_set_sizes:
        tprint_info(f"  📊 Loading metrics for {size}-feature set...")
        
        metrics = load_feature_set_metrics(exchange, asset, timeframe, size)
        
        # Skip if missing required metrics
        if metrics["gated_sharpe"] is None:
            tprint_warning(f"  ⚠️ Missing gated_sharpe for {size}-feature set, using default")
            metrics["gated_sharpe"] = 0.0
        if metrics["learnability"] is None:
            tprint_warning(f"  ⚠️ Missing learnability for {size}-feature set, using default")
            metrics["learnability"] = 0.5
        if metrics["generalization_gap"] is None:
            tprint_warning(f"  ⚠️ Missing generalization_gap for {size}-feature set, using default")
            metrics["generalization_gap"] = 0.05
        
        # Compute composite score
        score, breakdown = compute_composite_score(
            gated_sharpe=metrics["gated_sharpe"],
            learnability=metrics["learnability"],
            generalization_gap=metrics["generalization_gap"],
            mean_return=metrics.get("mean_return"),
            hit_rate=metrics.get("hit_rate"),
            trade_frequency=metrics.get("trade_frequency"),
        )
        
        scores[size] = score
        comparison_results[size] = {
            "metrics": metrics,
            "score_breakdown": breakdown,
            "composite_score": score,
        }
        
        tprint_info(
            f"    ↪ Score: {score:.4f} "
            f"(Sharpe={metrics['gated_sharpe']:.3f}, "
            f"Learn={metrics['learnability']:.3f}, "
            f"Gap={metrics['generalization_gap']:.4f})"
        )
    
    # Determine winner
    if not scores:
        tprint_error("❌ No valid scores computed, cannot determine winner")
        return 60, {"error": "No valid scores"}
    
    winning_size = max(scores, key=scores.get)
    winning_score = scores[winning_size]
    
    tprint_success(f"🏆 Winning feature set: {winning_size} features (score: {winning_score:.4f})")
    
    # Persist if requested
    if persist:
        try:
            from .lgbm_feature_selection import FeatureSetPersistence
            
            persistence = FeatureSetPersistence()
            
            # Load existing feature sets
            existing_data = persistence.load_feature_sets(exchange, asset)
            if existing_data:
                # Update with winning set info
                winning_metrics = comparison_results[winning_size]["metrics"]
                winning_breakdown = comparison_results[winning_size]["score_breakdown"]
                
                persistence.save_feature_sets(
                    feature_sets={
                        int(k): v.get("features", []) 
                        for k, v in existing_data.get("feature_sets", {}).items()
                    },
                    exchange=exchange,
                    asset=asset,
                    pipeline_log=existing_data.get("pipeline_log"),
                    winning_set_size=winning_size,
                    winning_metrics={
                        **winning_metrics,
                        "score_breakdown": winning_breakdown,
                        "composite_score": winning_score,
                        "determined_at": datetime.now().isoformat(),
                    },
                )
                tprint_success(f"💾 Persisted winning feature set ({winning_size}) for {asset}/{exchange}")
            else:
                tprint_warning(f"⚠️ No existing feature sets found to update for {asset}/{exchange}")
        except Exception as e:
            tprint_error(f"❌ Failed to persist winning feature set: {e}")
    
    # Generate comparison report
    _generate_winning_set_report(
        exchange, asset, timeframe, 
        winning_size, winning_score,
        comparison_results
    )
    
    return winning_size, comparison_results


def _generate_winning_set_report(
    exchange: str,
    asset: str,
    timeframe: str,
    winning_size: int,
    winning_score: float,
    comparison_results: Dict[int, Dict[str, Any]],
) -> Path:
    """Generate a markdown report for the winning feature set determination."""
    
    outcomes_dir = Path("outcomes")
    outcomes_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"winning_feature_set_{asset}_{timeframe}_{timestamp}.md"
    filepath = outcomes_dir / filename
    
    lines = [
        "# Winning Feature Set Determination Report",
        "",
        f"**Asset**: {asset}",
        f"**Exchange**: {exchange}",
        f"**Timeframe**: {timeframe}",
        f"**Generated**: {timestamp}",
        "",
        "## Winning Metrics Criteria",
        "",
        "The winning feature set is determined by a weighted composite score:",
        "",
        f"- **Gated Sharpe Ratio** ({METRIC_WEIGHTS['gated_sharpe']*100:.0f}%): Risk-adjusted returns - MOST IMPORTANT",
        f"- **Learnability** ({METRIC_WEIGHTS['learnability']*100:.0f}%): Model's ability to learn patterns (AUC-based)",
        f"- **Generalization Gap** ({METRIC_WEIGHTS['generalization_gap_penalty']*100:.0f}%): Penalty for overfitting (lower is better)",
        "",
        "## Results",
        "",
        f"### 🏆 Winner: {winning_size}-Feature Set",
        f"**Composite Score: {winning_score:.4f}**",
        "",
        "### Comparison Table",
        "",
        "| Feature Set | Composite Score | Gated Sharpe | Learnability | Gen. Gap | Mean Return | Hit Rate |",
        "|-------------|-----------------|--------------|--------------|----------|-------------|----------|",
    ]
    
    for size in sorted(comparison_results.keys(), reverse=True):
        result = comparison_results[size]
        metrics = result.get("metrics", {})
        score = result.get("composite_score", 0)
        
        winner_marker = " 🏆" if size == winning_size else ""
        
        sharpe = metrics.get("gated_sharpe", 0) or 0
        learn = metrics.get("learnability", 0) or 0
        gap = metrics.get("generalization_gap", 0) or 0
        mean_ret = metrics.get("mean_return", 0) or 0
        hit = metrics.get("hit_rate", 0) or 0
        
        lines.append(
            f"| {size}{winner_marker} | {score:.4f} | {sharpe:.3f} | {learn:.3f} | {gap:.4f} | {mean_ret:.4%} | {hit:.2%} |"
        )
    
    lines.extend([
        "",
        "### Score Breakdown (Winner)",
        "",
    ])
    
    if winning_size in comparison_results:
        breakdown = comparison_results[winning_size].get("score_breakdown", {})
        lines.extend([
            f"- Gated Sharpe: {breakdown.get('gated_sharpe_raw', 0):.3f} → normalized: {breakdown.get('gated_sharpe_normalized', 0):.3f} → weighted: {breakdown.get('gated_sharpe_weighted', 0):.4f}",
            f"- Learnability: {breakdown.get('learnability_raw', 0):.3f} → normalized: {breakdown.get('learnability_normalized', 0):.3f} → weighted: {breakdown.get('learnability_weighted', 0):.4f}",
            f"- Gen. Gap: {breakdown.get('generalization_gap_raw', 0):.4f} → normalized: {breakdown.get('generalization_gap_normalized', 0):.3f} → weighted: {breakdown.get('generalization_gap_weighted', 0):.4f}",
        ])
    
    lines.extend([
        "",
        "## Recommendation",
        "",
        f"Use the **{winning_size}-feature set** for Analyst Base models (feature_set B).",
        "",
        "This feature set achieves the best balance of:",
        "- Risk-adjusted returns (maximizing PnL with controlled risk)",
        "- Model learnability (patterns can be learned effectively)",
        "- Generalization (avoiding overfitting to training data)",
        "",
        "## Configuration",
        "",
        "To use this winning feature set in Analyst Base training:",
        "",
        "```yaml",
        "analyst_config:",
        "  feature_set: 'B'",
        "  feature_set_b_use_winning: true  # Will automatically use this winner",
        "```",
        "",
        f"Or to explicitly use the {winning_size}-feature set:",
        "",
        "```yaml",
        "analyst_config:",
        "  feature_set: 'B'",
        f"  feature_set_b_size: {winning_size}",
        "```",
    ])
    
    with open(filepath, 'w') as f:
        f.write('\n'.join(lines))
    
    tprint_info(f"📝 Winning feature set report saved to: {filepath}")
    return filepath


def run_winning_feature_set_selection(
    symbol: str,
    exchange: str,
    timeframe: str,
    direction: str = "long",
) -> Dict[str, Any]:
    """
    Main entry point to run winning feature set selection.
    
    This should be called after:
    1. feature_generation_meta_labeling_step (with use_lgbm_feature_selection=True)
    2. snr_diagnostics (with feature-set-comparison)
    3. meta_gated_backtest
    
    Args:
        symbol: Trading symbol
        exchange: Exchange name
        timeframe: Timeframe string
        direction: Trading direction
        
    Returns:
        Dictionary with winning set info
    """
    tprint_info(f"🔄 Running winning feature set selection for {symbol}/{exchange} [{timeframe}]")
    
    winning_size, comparison_results = determine_winning_feature_set(
        exchange=exchange,
        asset=symbol,
        timeframe=timeframe,
        persist=True,
    )
    
    return {
        "winning_size": winning_size,
        "comparison_results": comparison_results,
        "symbol": symbol,
        "exchange": exchange,
        "timeframe": timeframe,
        "direction": direction,
    }
