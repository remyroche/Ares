from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Dict, Optional
import numpy as np
import pandas as pd
from datetime import datetime

@dataclass
class OptimizationConfig:
    candidate_periods: Optional[list[int]] = None  # if None, derive from data
    score_metric: str = "sharpe"  # placeholder
    direction: str = "longs"

def _score_series(returns: pd.Series, metric: str = "sharpe") -> float:
    if returns.empty or returns.std() == 0:
        return -np.inf
    if metric == "sharpe":
        # naive daily-like sharpe (no risk-free, unit variance scale)
        return returns.mean() / returns.std()
    elif metric == "sum":
        return returns.sum()
    return returns.mean()

def _rolling_signal(df: pd.DataFrame, period: int, direction: str) -> pd.Series:
    # Example: use close momentum as a proxy signal
    mom = df["close"].pct_change(periods=period)
    signal = np.sign(mom).fillna(0.0)
    if direction == "shorts":
        signal *= -1
    return signal.clip(-1, 1)

async def run_period_optimization_step(
    data: pd.DataFrame,
    symbol: str = "ETHUSDT",
    timeframe: str = "15m",
    direction: str = "longs",
    intensity: str = "blank",
    lookback_days: Optional[int] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    exchange: str = "binance",
    custom_overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Naive, fully self-contained 'period optimization' that scores a set of rolling periods
    on a toy momentum strategy and returns the best period.
    """
    # 0) defensive copies + slicing
    df = data.copy()
    if not {"open","high","low","close","volume"}.issubset(df.columns):
        raise ValueError("DataFrame must contain columns: open, high, low, close, volume")

    if start_date:
        df = df[df.index >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df.index <= pd.to_datetime(end_date)]
    if lookback_days:
        cutoff = df.index.max() - pd.Timedelta(days=lookback_days)
        df = df[df.index >= cutoff]

    if df.empty:
        return dict(success=False, optimized_periods=0, optimization_metadata={}, artifacts={}, error_message="No data after filtering")

    # 1) derive candidate periods if not provided
    overrides = custom_overrides or {}
    cfg = OptimizationConfig(
        candidate_periods=overrides.get("candidate_periods") or [3, 5, 8, 10, 12, 15, 20, 30, 50],
        score_metric=overrides.get("score_metric", "sharpe"),
        direction=direction,
    )

    # 2) price returns at native timeframe
    ret = df["close"].pct_change().fillna(0.0)

    # 3) brute force scan
    scores: Dict[int, float] = {}
    for p in cfg.candidate_periods:
        sig = _rolling_signal(df, p, cfg.direction)
        strat_ret = ret.shift(1) * sig  # enter on next bar
        scores[p] = _score_series(strat_ret, cfg.score_metric)

    # 4) pick best (max score)
    best_period = max(scores, key=scores.get)
    best_score = scores[best_period]

    # 5) artifacts + metadata for downstream
    artifacts = {
        "scores_by_period": scores,
        "symbol": symbol,
        "timeframe": timeframe,
        "direction": direction,
        "exchange": exchange,
    }
    metadata = {
        "optimized_periods": best_period,
        "best_score": best_score,
        "score_metric": cfg.score_metric,
        "evaluated_at": datetime.utcnow().isoformat() + "Z",
        "n_candidates": len(cfg.candidate_periods),
    }

    # Simulate async workload parity with real pipeline
    await asyncio.sleep(0)

    return dict(
        success=True,
        optimized_periods=best_period,
        optimization_metadata=metadata,
        artifacts=artifacts,
        metadata=metadata,
        error_message=None,
    )

async def run_lookback_optimization_step(
    data: pd.DataFrame,
    symbol: str = "ETHUSDT",
    timeframe: str = "15m",
    direction: str = "longs",
    intensity: str = "blank",
    lookback_days: Optional[int] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    exchange: str = "binance",
    custom_overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Naive, fully self-contained 'lookback optimization' that scores different lookback windows
    on a toy momentum strategy and returns the best lookback.
    """
    # 0) defensive copies + slicing
    df = data.copy()
    if not {"open","high","low","close","volume"}.issubset(df.columns):
        raise ValueError("DataFrame must contain columns: open, high, low, close, volume")

    if start_date:
        df = df[df.index >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df.index <= pd.to_datetime(end_date)]
    if lookback_days:
        cutoff = df.index.max() - pd.Timedelta(days=lookback_days)
        df = df[df.index >= cutoff]

    if df.empty:
        return dict(success=False, optimized_lookbacks=0, optimization_metadata={}, artifacts={}, error_message="No data after filtering")

    # 1) derive candidate lookbacks if not provided
    overrides = custom_overrides or {}
    cfg = OptimizationConfig(
        candidate_periods=overrides.get("candidate_lookbacks") or [10, 20, 30, 50, 100, 200],
        score_metric=overrides.get("score_metric", "sharpe"),
        direction=direction,
    )

    # 2) price returns at native timeframe
    ret = df["close"].pct_change().fillna(0.0)

    # 3) brute force scan
    scores: Dict[int, float] = {}
    for p in cfg.candidate_periods:
        sig = _rolling_signal(df, p, cfg.direction)
        strat_ret = ret.shift(1) * sig  # enter on next bar
        scores[p] = _score_series(strat_ret, cfg.score_metric)

    # 4) pick best (max score)
    best_lookback = max(scores, key=scores.get)
    best_score = scores[best_lookback]

    # 5) artifacts + metadata for downstream
    artifacts = {
        "scores_by_lookback": scores,
        "symbol": symbol,
        "timeframe": timeframe,
        "direction": direction,
        "exchange": exchange,
    }
    metadata = {
        "optimized_lookbacks": best_lookback,
        "best_score": best_score,
        "score_metric": cfg.score_metric,
        "evaluated_at": datetime.utcnow().isoformat() + "Z",
        "n_candidates": len(cfg.candidate_periods),
    }

    # Simulate async workload parity with real pipeline
    await asyncio.sleep(0)

    return dict(
        success=True,
        optimized_lookbacks=best_lookback,
        optimization_metadata=metadata,
        artifacts=artifacts,
        metadata=metadata,
        error_message=None,
    )

async def run_period_lookback_optimization_step(
    data: pd.DataFrame,
    symbol: str = "ETHUSDT",
    timeframe: str = "15m",
    direction: str = "longs",
    intensity: str = "blank",
    lookback_days: Optional[int] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    exchange: str = "binance",
    custom_overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Combined period and lookback optimization step.
    """
    # Run period optimization
    period_result = await run_period_optimization_step(
        data=data,
        symbol=symbol,
        timeframe=timeframe,
        direction=direction,
        intensity=intensity,
        lookback_days=lookback_days,
        start_date=start_date,
        end_date=end_date,
        exchange=exchange,
        custom_overrides=custom_overrides
    )
    
    # Run lookback optimization
    lookback_result = await run_lookback_optimization_step(
        data=data,
        symbol=symbol,
        timeframe=timeframe,
        direction=direction,
        intensity=intensity,
        lookback_days=lookback_days,
        start_date=start_date,
        end_date=end_date,
        exchange=exchange,
        custom_overrides=custom_overrides
    )
    
    # Combine results
    combined_artifacts = {}
    combined_artifacts.update(period_result.get('artifacts', {}))
    combined_artifacts.update(lookback_result.get('artifacts', {}))
    
    combined_metadata = {}
    combined_metadata.update(period_result.get('metadata', {}))
    combined_metadata.update(lookback_result.get('metadata', {}))
    
    # Generate report
    await _generate_period_lookback_optimization_report({
        'period_result': period_result,
        'lookback_result': lookback_result,
        'combined_artifacts': combined_artifacts,
        'combined_metadata': combined_metadata
    }, data)
    
    return dict(
        success=period_result.get('success', False) and lookback_result.get('success', False),
        optimized_periods=period_result.get('optimized_periods', 0),
        optimized_lookbacks=lookback_result.get('optimized_lookbacks', 0),
        optimization_metadata=combined_metadata,
        artifacts=combined_artifacts,
        metadata=combined_metadata,
        error_message=None if (period_result.get('success', False) and lookback_result.get('success', False)) else "One or both optimization steps failed",
    )

async def _generate_period_lookback_optimization_report(result: Dict[str, Any], data: pd.DataFrame) -> None:
    """Generate human-readable report for period + lookback optimization step."""
    from pathlib import Path
    
    # Create outcomes directory
    outcomes_dir = Path("outcomes")
    outcomes_dir.mkdir(exist_ok=True)

    # Generate timestamp for filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    report_filename = f"period_lookback_optimization_report_{timestamp}.md"
    report_path = outcomes_dir / report_filename

    # Generate report content
    report_content = f"""# Period + Lookback Optimization Report
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Executive Summary
- **Status**: {'✅ SUCCESS' if result.get('success', False) else '❌ FAILED'}
- **Data Shape**: {data.shape[0]} rows x {data.shape[1]} columns
- **Optimized Periods**: {result.get('optimized_periods', 'N/A')}
- **Optimized Lookbacks**: {result.get('optimized_lookbacks', 'N/A')}

## Period Optimization Results
- **Success**: {result.get('period_result', {}).get('success', False)}
- **Best Period**: {result.get('period_result', {}).get('optimized_periods', 'N/A')}
- **Best Score**: {result.get('period_result', {}).get('metadata', {}).get('best_score', 'N/A')}

## Lookback Optimization Results
- **Success**: {result.get('lookback_result', {}).get('success', False)}
- **Best Lookback**: {result.get('lookback_result', {}).get('optimized_lookbacks', 'N/A')}
- **Best Score**: {result.get('lookback_result', {}).get('metadata', {}).get('best_score', 'N/A')}

## Combined Results
- **Artifacts Generated**: {len(result.get('combined_artifacts', {}))}
- **Metadata Fields**: {len(result.get('combined_metadata', {}))}

## Next Steps
1. Review optimization results
2. Proceed to next pipeline step

---
*Report generated by Consolidated Pipeline Runner*
"""

    # Write report
    with open(report_path, 'w') as f:
        f.write(report_content)

    # Add report to artifacts
    result['combined_artifacts']['human_readable_report'] = str(report_path)

    print(f"📊 Human-readable report saved: {report_path}")