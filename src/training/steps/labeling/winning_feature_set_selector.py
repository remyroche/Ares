"""
Winning Feature Set Selector.

This module determines the winning feature set based on a simple performance formula:

    Score = (Mean Return % per Trade × Trades per Day) - (Max Drawdown % / 100)

Higher score is better. This formula captures:
- Expected daily profit (mean return * frequency)
- Risk penalty (drawdown)

All metrics are computed assuming 0.55 confidence threshold in the backtester.

Created: 2025-12-08
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.utils.tprint import tprint, tprint_info, tprint_success, tprint_warning, tprint_error

logger = logging.getLogger(__name__)


def compute_winning_score(
    mean_return_pct: float,
    trades_per_day: float,
    max_drawdown_pct: float,
) -> float:
    """
    Compute the winning score for a feature set.
    
    Formula:
        Score = (Mean Return % per Trade × Trades per Day) - (Max Drawdown % / 100)
    
    Higher is better.
    
    Args:
        mean_return_pct: Mean return per trade in percent (e.g., 0.15 for 0.15%)
        trades_per_day: Average number of trades per day
        max_drawdown_pct: Maximum drawdown in percent (e.g., 5.0 for 5%)
        
    Returns:
        Winning score (higher is better)
        
    Example:
        >>> compute_winning_score(mean_return_pct=0.12, trades_per_day=3.5, max_drawdown_pct=8.0)
        0.34  # (0.12 * 3.5) - (8.0 / 100) = 0.42 - 0.08 = 0.34
    """
    expected_daily_return = mean_return_pct * trades_per_day
    drawdown_penalty = max_drawdown_pct / 100.0
    score = expected_daily_return - drawdown_penalty
    return score


def load_feature_set_metrics(
    exchange: str,
    asset: str,
    timeframe: str,
    feature_set_size: int,
) -> Dict[str, Any]:
    """
    Load metrics for a specific feature set from saved backtest results.
    
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
        "mean_return_pct": None,
        "trades_per_day": None,
        "max_drawdown_pct": None,
    }
    
    outcomes_dir = Path("outcomes")
    
    # Load from meta_gated_backtest reports
    try:
        pattern = f"meta_gated_backtest_{asset}_{timeframe}_*.json"
        files = sorted(glob.glob(str(outcomes_dir / pattern)), reverse=True)
        if files:
            with open(files[0], 'r') as f:
                data = json.load(f)
            
            backtest_metrics = data.get("metrics", {})
            
            # Mean return per trade (%)
            if "mean_return_gated" in backtest_metrics:
                # Convert from decimal to percent if needed
                mr = backtest_metrics["mean_return_gated"]
                metrics["mean_return_pct"] = mr * 100 if abs(mr) < 1 else mr
            elif "mean_return" in backtest_metrics:
                mr = backtest_metrics["mean_return"]
                metrics["mean_return_pct"] = mr * 100 if abs(mr) < 1 else mr
            
            # Trades per day
            if "trades_per_day" in backtest_metrics:
                metrics["trades_per_day"] = backtest_metrics["trades_per_day"]
            elif "n_trades_gated" in backtest_metrics and "trading_days" in backtest_metrics:
                n_trades = backtest_metrics["n_trades_gated"]
                days = backtest_metrics["trading_days"]
                if days > 0:
                    metrics["trades_per_day"] = n_trades / days
            
            # Max drawdown (%)
            if "max_drawdown_gated" in backtest_metrics:
                dd = backtest_metrics["max_drawdown_gated"]
                metrics["max_drawdown_pct"] = abs(dd) * 100 if abs(dd) < 1 else abs(dd)
            elif "max_drawdown" in backtest_metrics:
                dd = backtest_metrics["max_drawdown"]
                metrics["max_drawdown_pct"] = abs(dd) * 100 if abs(dd) < 1 else abs(dd)
                
    except Exception as e:
        logger.warning(f"Failed to load meta_gated_backtest metrics: {e}")
    
    # Try loading from feature set specific results
    try:
        pattern = f"feature_set_backtest_{asset}_{timeframe}_{feature_set_size}_*.json"
        files = sorted(glob.glob(str(outcomes_dir / pattern)), reverse=True)
        if files:
            with open(files[0], 'r') as f:
                data = json.load(f)
            
            # Override with feature-set-specific metrics if available
            if data.get("mean_return_pct") is not None:
                metrics["mean_return_pct"] = data["mean_return_pct"]
            if data.get("trades_per_day") is not None:
                metrics["trades_per_day"] = data["trades_per_day"]
            if data.get("max_drawdown_pct") is not None:
                metrics["max_drawdown_pct"] = data["max_drawdown_pct"]
    except Exception as e:
        logger.debug(f"No feature-set-specific backtest results: {e}")
    
    return metrics


def determine_winning_feature_set(
    exchange: str,
    asset: str,
    timeframe: str,
    feature_set_sizes: List[int] = [50, 60, 70, 80],
    persist: bool = True,
    metrics_override: Optional[Dict[int, Dict[str, float]]] = None,
) -> Tuple[int, Dict[str, Any]]:
    """
    Determine the winning feature set based on backtest performance.
    
    Formula:
        Score = (Mean Return % × Trades/Day) - (Max Drawdown % / 100)
    
    Args:
        exchange: Exchange name
        asset: Asset symbol
        timeframe: Timeframe string
        feature_set_sizes: List of feature set sizes to compare
        persist: Whether to persist the winning set
        metrics_override: Optional dict of {size: {mean_return_pct, trades_per_day, max_drawdown_pct}}
                         to use instead of loading from files
        
    Returns:
        Tuple of (winning_size, comparison_results)
    """
    tprint_info(f"🏆 Determining winning feature set for {asset}/{exchange} [{timeframe}]")
    tprint_info(f"   Formula: Score = (Mean Return % × Trades/Day) - (Max Drawdown % / 100)")
    
    comparison_results = {}
    scores = {}
    
    for size in feature_set_sizes:
        tprint_info(f"  📊 Evaluating {size}-feature set...")
        
        # Use override metrics if provided, otherwise load from files
        if metrics_override and size in metrics_override:
            metrics = {
                "feature_set_size": size,
                **metrics_override[size]
            }
        else:
            metrics = load_feature_set_metrics(exchange, asset, timeframe, size)
        
        # Check for missing metrics
        mean_ret = metrics.get("mean_return_pct")
        trades = metrics.get("trades_per_day")
        drawdown = metrics.get("max_drawdown_pct")
        
        if mean_ret is None or trades is None or drawdown is None:
            tprint_warning(f"  ⚠️ Missing metrics for {size}-feature set:")
            if mean_ret is None:
                tprint_warning(f"     - mean_return_pct: MISSING")
            if trades is None:
                tprint_warning(f"     - trades_per_day: MISSING")
            if drawdown is None:
                tprint_warning(f"     - max_drawdown_pct: MISSING")
            
            # Use defaults for missing values
            mean_ret = mean_ret if mean_ret is not None else 0.0
            trades = trades if trades is not None else 1.0
            drawdown = drawdown if drawdown is not None else 10.0
        
        # Compute score
        score = compute_winning_score(
            mean_return_pct=mean_ret,
            trades_per_day=trades,
            max_drawdown_pct=drawdown,
        )
        
        scores[size] = score
        comparison_results[size] = {
            "metrics": {
                "mean_return_pct": mean_ret,
                "trades_per_day": trades,
                "max_drawdown_pct": drawdown,
            },
            "score": score,
            "score_breakdown": {
                "expected_daily_return": mean_ret * trades,
                "drawdown_penalty": drawdown / 100.0,
            }
        }
        
        tprint_info(
            f"    ↪ Score: {score:.4f} "
            f"(MeanRet={mean_ret:.3f}% × Trades={trades:.2f}/day - DD={drawdown:.2f}%/100)"
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
                        "score": winning_score,
                        "formula": "(mean_return_pct * trades_per_day) - (max_drawdown_pct / 100)",
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
        f"**Confidence Threshold**: 0.55",
        "",
        "## Scoring Formula",
        "",
        "```",
        "Score = (Mean Return % per Trade × Trades per Day) - (Max Drawdown % / 100)",
        "```",
        "",
        "**Higher score is better.** This formula captures:",
        "- **Expected daily profit**: mean return × trade frequency",
        "- **Risk penalty**: max drawdown (scaled by /100)",
        "",
        "## Results",
        "",
        f"### 🏆 Winner: {winning_size}-Feature Set",
        f"**Score: {winning_score:.4f}**",
        "",
        "### Comparison Table",
        "",
        "| Feature Set | Score | Mean Return % | Trades/Day | Max DD % | Expected Daily | DD Penalty |",
        "|-------------|-------|---------------|------------|----------|----------------|------------|",
    ]
    
    for size in sorted(comparison_results.keys(), reverse=True):
        result = comparison_results[size]
        metrics = result.get("metrics", {})
        score = result.get("score", 0)
        breakdown = result.get("score_breakdown", {})
        
        winner_marker = " 🏆" if size == winning_size else ""
        
        mean_ret = metrics.get("mean_return_pct", 0)
        trades = metrics.get("trades_per_day", 0)
        drawdown = metrics.get("max_drawdown_pct", 0)
        expected = breakdown.get("expected_daily_return", 0)
        penalty = breakdown.get("drawdown_penalty", 0)
        
        lines.append(
            f"| {size}{winner_marker} | {score:.4f} | {mean_ret:.3f}% | {trades:.2f} | {drawdown:.2f}% | {expected:.4f} | {penalty:.4f} |"
        )
    
    lines.extend([
        "",
        "### Score Calculation (Winner)",
        "",
    ])
    
    if winning_size in comparison_results:
        result = comparison_results[winning_size]
        metrics = result.get("metrics", {})
        breakdown = result.get("score_breakdown", {})
        
        mean_ret = metrics.get("mean_return_pct", 0)
        trades = metrics.get("trades_per_day", 0)
        drawdown = metrics.get("max_drawdown_pct", 0)
        expected = breakdown.get("expected_daily_return", 0)
        penalty = breakdown.get("drawdown_penalty", 0)
        
        lines.extend([
            f"```",
            f"Score = ({mean_ret:.3f}% × {trades:.2f}) - ({drawdown:.2f}% / 100)",
            f"     = {expected:.4f} - {penalty:.4f}",
            f"     = {winning_score:.4f}",
            f"```",
        ])
    
    lines.extend([
        "",
        "## Recommendation",
        "",
        f"Use the **{winning_size}-feature set** for Analyst Base models (feature_set B).",
        "",
        "This feature set achieves the best balance of:",
        "- Expected daily returns (mean return × trade frequency)",
        "- Risk control (drawdown penalty)",
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
    
    # Also save JSON version for programmatic access
    json_path = outcomes_dir / f"winning_feature_set_{asset}_{timeframe}_{timestamp}.json"
    json_data = {
        "asset": asset,
        "exchange": exchange,
        "timeframe": timeframe,
        "timestamp": timestamp,
        "confidence_threshold": 0.55,
        "formula": "(mean_return_pct * trades_per_day) - (max_drawdown_pct / 100)",
        "winning_size": winning_size,
        "winning_score": winning_score,
        "comparison_results": comparison_results,
    }
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2, default=str)
    
    tprint_info(f"📝 Winning feature set report saved to: {filepath}")
    return filepath


def run_winning_feature_set_selection(
    symbol: str,
    exchange: str,
    timeframe: str,
    direction: str = "long",
    metrics_override: Optional[Dict[int, Dict[str, float]]] = None,
) -> Dict[str, Any]:
    """
    Main entry point to run winning feature set selection.
    
    This should be called after:
    1. feature_generation_meta_labeling_step (with use_lgbm_feature_selection=True)
    2. meta_gated_backtest (for each feature set)
    
    Args:
        symbol: Trading symbol
        exchange: Exchange name
        timeframe: Timeframe string
        direction: Trading direction
        metrics_override: Optional metrics to use instead of loading from files
                         Format: {50: {mean_return_pct, trades_per_day, max_drawdown_pct}, ...}
        
    Returns:
        Dictionary with winning set info
    """
    tprint_info(f"🔄 Running winning feature set selection for {symbol}/{exchange} [{timeframe}]")
    
    winning_size, comparison_results = determine_winning_feature_set(
        exchange=exchange,
        asset=symbol,
        timeframe=timeframe,
        persist=True,
        metrics_override=metrics_override,
    )
    
    return {
        "winning_size": winning_size,
        "winning_score": comparison_results.get(winning_size, {}).get("score"),
        "comparison_results": comparison_results,
        "symbol": symbol,
        "exchange": exchange,
        "timeframe": timeframe,
        "direction": direction,
        "formula": "(mean_return_pct * trades_per_day) - (max_drawdown_pct / 100)",
    }


# Keep old names for backward compatibility but mark as simplified
METRIC_WEIGHTS = {
    "note": "Simplified formula now used - see compute_winning_score()",
    "formula": "(mean_return_pct * trades_per_day) - (max_drawdown_pct / 100)",
}


# Alias for backward compatibility
def compute_composite_score(
    mean_return_pct: float = 0.0,
    trades_per_day: float = 1.0,
    max_drawdown_pct: float = 10.0,
    **kwargs,  # Ignore old arguments like gated_sharpe, learnability, etc.
) -> Tuple[float, Dict[str, float]]:
    """
    Backward-compatible wrapper for compute_winning_score.
    
    Note: Old arguments (gated_sharpe, learnability, generalization_gap) are ignored.
    """
    score = compute_winning_score(mean_return_pct, trades_per_day, max_drawdown_pct)
    breakdown = {
        "mean_return_pct": mean_return_pct,
        "trades_per_day": trades_per_day,
        "max_drawdown_pct": max_drawdown_pct,
        "expected_daily_return": mean_return_pct * trades_per_day,
        "drawdown_penalty": max_drawdown_pct / 100.0,
        "score": score,
    }
    return score, breakdown
