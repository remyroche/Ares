#!/usr/bin/env python3
"""Labeling Triple-Barrier Geometry Sweep

This script sweeps triple-barrier geometry parameters used during meta-labeling
on top of raw market data (no re-training of the meta-models).

Goal (per configuration):
- Use the same primary signal generation and volatility logic as the
  FeatureGenerationMetaLabelingStep.
- Apply a triple-barrier engine with:
    * profit_thr_base (base TP in fractional return space)
    * stop_to_profit_ratio (SL as a fraction of TP)
    * horizon_bars (maximum holding period in bars)
    * min_event_spacing (minimum spacing between events in bars)
- Compute realized returns with transaction costs and adaptive thresholds.
- Evaluate event-level PnL on a holdout segment and select configurations that
  maximize daily PnL subject to a minimum trades/day constraint.

This is intentionally lighter-weight than a full re-run of the meta-labeling
pipeline: it focuses purely on the event geometry and realized returns, which
are the main drivers of trade frequency and raw economic edge.

Usage (from project root):

    python3 scripts/labeling_geometry_sweep.py \
        --symbol ETHUSDT \
        --exchange binance \
        --timeframe 15m \
        --direction long \
        --holdout-fraction 0.30 \
        --min-trades-per-day 1.0 \
        --outcomes-dir outcomes

You can override the geometry grids, for example:

    --profit-bases 0.0075 0.0100 0.0125 \
    --stop-ratios 0.4 0.5 0.6 \
    --horizons 4 8 12 16 \
    --min-event-spacings 1 2 4

All profit/stop inputs are in fraction space (e.g. 0.01 = 1%).
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

# Ensure project root is on sys.path so that `src.*` imports work
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.steps.labeling import (  # type: ignore
    FeatureGenerationMetaLabelingStep,
    compute_realized_returns,
    generate_primary_signals,
    DEFAULT_PROFIT_THRESHOLD,
    DEFAULT_STOP_THRESHOLD,
    DEFAULT_TRANSACTION_COST,
)


PROFIT_FLOOR = 0.005  # 0.5% floor used in FeatureGenerationMetaLabelingStep


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep triple-barrier geometry and summarize economic metrics",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--symbol", type=str, default="ETHUSDT", help="Trading symbol")
    parser.add_argument("--exchange", type=str, default="binance", help="Exchange name")
    parser.add_argument("--timeframe", type=str, default="15m", help="Base timeframe")
    parser.add_argument("--direction", type=str, default="long", help="Trading direction context")
    parser.add_argument(
        "--execution-mode",
        type=str,
        default="full",
        help="Execution mode context for data loading (full/light/blank)",
    )
    parser.add_argument(
        "--holdout-fraction",
        type=float,
        default=0.30,
        help="Fraction of events to reserve as evaluation holdout (last fraction)",
    )
    parser.add_argument(
        "--min-trades-per-day",
        type=float,
        default=1.0,
        help="Minimum average trades/day constraint for selecting best config",
    )
    parser.add_argument(
        "--outcomes-dir",
        type=str,
        default="outcomes",
        help="Directory to save sweep results",
    )
    parser.add_argument(
        "--transaction-cost",
        type=float,
        default=DEFAULT_TRANSACTION_COST,
        help="Per-leg transaction cost in fractional return space (e.g. 0.0015 = 0.15%%)",
    )

    # Geometry grids (fractions / integers)
    parser.add_argument(
        "--profit-bases",
        type=float,
        nargs="+",
        default=[0.0150, 0.0175, 0.0200, 0.0225, 0.0250],
        help="Grid of base profit thresholds (fraction, e.g. 0.01 = 1%%)",
    )
    parser.add_argument(
        "--stop-ratios",
        type=float,
        nargs="+",
        default=[0.4, 0.5, 0.6],
        help="Grid of stop-to-profit ratios (SL = ratio * TP)",
    )
    parser.add_argument(
        "--horizons",
        type=int,
        nargs="+",
        default=[4, 8, 12, 16],
        help="Grid of horizon bars for the triple barrier",
    )
    parser.add_argument(
        "--min-event-spacings",
        type=int,
        nargs="+",
        default=[1, 2, 4],
        help="Grid of minimum event spacing values in bars",
    )

    parser.add_argument(
        "--vol-baseline-window",
        type=int,
        default=96,
        help="Baseline volatility window used to compute the vol_factor",
    )
    parser.add_argument(
        "--profit-mult-min",
        type=float,
        default=0.9,
        help="Lower multiplier for adaptive profit thresholds (around base TP)",
    )
    parser.add_argument(
        "--profit-mult-max",
        type=float,
        default=1.4,
        help="Upper multiplier for adaptive profit thresholds (around base TP)",
    )
    parser.add_argument(
        "--stop-mult-min",
        type=float,
        default=0.7,
        help="Lower multiplier for adaptive stop thresholds (around base SL)",
    )
    parser.add_argument(
        "--stop-mult-max",
        type=float,
        default=1.15,
        help="Upper multiplier for adaptive stop thresholds (around base SL)",
    )

    return parser.parse_args()


def maybe_load_hpo_defaults(symbol: str, timeframe: str) -> Dict[str, Any]:
    """Best-effort load of labeling HPO defaults to center volatility settings.

    This reads the latest meta_labeling_hpo_best_params JSON for the given
    symbol/timeframe (if present) and extracts knee_params / best_params.
    """

    outcomes_dir = PROJECT_ROOT / "outcomes"
    if not outcomes_dir.exists():
        return {}

    pattern = f"meta_labeling_hpo_best_params_{symbol}_{timeframe}_*.json"
    candidates = sorted(outcomes_dir.glob(pattern))
    if not candidates:
        return {}

    latest_path = candidates[-1]
    try:
        with latest_path.open("r") as f:
            data = json.load(f)
    except Exception:
        return {}

    params = data.get("knee_params") or data.get("best_params") or {}
    return params


def load_market_data(args: argparse.Namespace) -> pd.DataFrame:
    """Load market data via FeatureGenerationMetaLabelingStep's helper.

    This keeps behavior aligned with the production meta-labeling step.
    """

    step = FeatureGenerationMetaLabelingStep()
    step.set_context(
        symbol=args.symbol,
        exchange=args.exchange,
        timeframe=args.timeframe,
        direction=args.direction,
        model="analyst",
    )

    base_cfg: Dict[str, Any] = {
        "symbol": args.symbol,
        "exchange": args.exchange,
        "timeframe": args.timeframe,
        "direction": args.direction,
        "execution_mode": args.execution_mode,
    }

    pipeline_state: Dict[str, Any] = {}
    market_data, source = step.load_market_data_or_fail(  # type: ignore[attr-defined]
        base_cfg,
        pipeline_state=pipeline_state,
        allow_config_override=True,
    )

    if not isinstance(market_data, pd.DataFrame) or market_data.empty:
        raise ValueError(f"No market data available for {args.symbol} {args.timeframe}")

    idx = market_data.index
    if isinstance(idx, pd.DatetimeIndex) and idx.tz is not None:
        try:
            market_data = market_data.copy()
            market_data.index = market_data.index.tz_convert("UTC").tz_localize(None)
        except Exception:
            # Fallback: keep original index
            pass

    if "close" not in market_data.columns:
        raise ValueError("Missing required 'close' column in market data")

    print(f"Loaded market_data for {args.symbol} {args.timeframe}: rows={len(market_data)} from {source}")
    return market_data


def build_volatility_series(market_data: pd.DataFrame, vol_baseline_window: int) -> Tuple[pd.Series, pd.Series]:
    """Compute volatility_1d and vol_factor as in the meta-labeling step."""

    log_ret = np.log(market_data["close"]).diff()
    volatility_1d = log_ret.rolling(96).std()

    vol_baseline = volatility_1d.rolling(vol_baseline_window).mean()
    vol_factor = volatility_1d / (vol_baseline + 1e-8)

    return volatility_1d, vol_factor


def evaluate_geometry_config(
    market_data: pd.DataFrame,
    primary_signals: pd.DataFrame,
    volatility_1d: pd.Series,
    vol_factor: pd.Series,
    profit_base: float,
    stop_ratio: float,
    horizon: int,
    min_event_spacing: int,
    profit_mult_min: float,
    profit_mult_max: float,
    stop_mult_min: float,
    stop_mult_max: float,
    transaction_cost: float,
    holdout_fraction: float,
) -> Dict[str, Any]:
    """Apply triple-barrier geometry and compute economic metrics on holdout events."""

    profit_threshold = float(profit_base) if profit_base > 0.0 else float(DEFAULT_PROFIT_THRESHOLD)
    stop_threshold = max(0.0005, profit_threshold * float(stop_ratio)) if stop_ratio > 0.0 else float(
        DEFAULT_STOP_THRESHOLD
    )

    # Adaptive thresholds based on volatility and HPO multipliers
    adaptive_profit_threshold = profit_threshold * vol_factor
    adaptive_stop_threshold = stop_threshold * vol_factor

    # Enforce hard floor based on transaction costs (0.5% = 50 bps)
    adaptive_profit_threshold = adaptive_profit_threshold.clip(
        lower=max(profit_threshold * profit_mult_min, PROFIT_FLOOR),
        upper=profit_threshold * profit_mult_max,
    )
    adaptive_stop_threshold = adaptive_stop_threshold.clip(
        lower=stop_threshold * stop_mult_min,
        upper=stop_threshold * stop_mult_max,
    )

    (
        realized_returns,
        _binary_labels,
        _exit_reasons,
        event_durations,
        _mfe_series,
        _mae_series,
        _binary_labels_long,
        _binary_labels_short,
    ) = compute_realized_returns(
        market_data,
        primary_signals,
        profit_threshold=adaptive_profit_threshold,
        stop_threshold=adaptive_stop_threshold,
        horizon=int(horizon),
        transaction_cost=float(transaction_cost),
        min_event_spacing=int(min_event_spacing),
        volatility_series=volatility_1d,
    )

    # Build event-level evaluation on a holdout segment
    event_mask = ~realized_returns.isna()
    n_events_total = int(event_mask.sum())

    if n_events_total == 0:
        return {
            "profit_base": float(profit_threshold),
            "stop_to_profit_ratio": float(stop_ratio),
            "stop_threshold": float(stop_threshold),
            "horizon_bars": int(horizon),
            "min_event_spacing": int(min_event_spacing),
            "n_events_total": 0,
            "n_events_eval": 0,
            "n_trades": 0,
            "eval_days": 0,
            "trades_per_day": 0.0,
            "mean_return": 0.0,
            "std_return": 0.0,
            "hit_rate": 0.0,
            "final_equity": 1.0,
            "total_return": 0.0,
            "max_drawdown": 0.0,
            "sharpe_trade": 0.0,
            "pnl_per_day": 0.0,
            "eval_start_date": None,
            "eval_end_date": None,
            "avg_event_duration_bars": float(event_durations.mean()) if len(event_durations) > 0 else 0.0,
        }

    # Determine holdout segment by event order
    event_index = realized_returns.index[event_mask]

    if holdout_fraction <= 0.0 or holdout_fraction >= 1.0:
        eval_mask = event_mask.copy()
    else:
        n_holdout = max(1, int(round(n_events_total * holdout_fraction)))
        holdout_idx = event_index[-n_holdout:]
        eval_mask = realized_returns.index.isin(holdout_idx)

    n_events_eval = int(eval_mask.sum())

    # Evaluation days based on events in eval segment
    if isinstance(realized_returns.index, pd.DatetimeIndex) and n_events_eval > 0:
        eval_times = realized_returns.index[eval_mask]
        eval_start_date = eval_times[0].date()
        eval_end_date = eval_times[-1].date()
        eval_days = int((eval_end_date - eval_start_date).days) + 1
        if eval_days <= 0:
            eval_days = 1
    else:
        eval_start_date = None
        eval_end_date = None
        eval_days = 1

    eval_rets = realized_returns[eval_mask].astype(float).dropna()
    n_trades = int(eval_rets.size)

    trades_per_day = float(n_trades) / float(eval_days) if eval_days > 0 else 0.0

    if n_trades > 0:
        mean_ret = float(eval_rets.mean())
        std_ret = float(eval_rets.std(ddof=1)) if n_trades > 1 else 0.0
        hit_rate = float((eval_rets > 0).mean())

        equity = (1.0 + eval_rets).cumprod()
        final_equity = float(equity.iloc[-1])
        total_return = final_equity - 1.0

        running_max = equity.cummax()
        drawdown = equity / running_max - 1.0
        max_drawdown = float(drawdown.min()) if drawdown.size > 0 else 0.0

        sharpe_trade = (
            float(mean_ret / std_ret) * float(np.sqrt(n_trades)) if std_ret > 0.0 else 0.0
        )

        pnl_per_day = (
            float(final_equity ** (1.0 / eval_days) - 1.0) if eval_days > 0 else 0.0
        )
    else:
        mean_ret = 0.0
        std_ret = 0.0
        hit_rate = 0.0
        final_equity = 1.0
        total_return = 0.0
        max_drawdown = 0.0
        sharpe_trade = 0.0
        pnl_per_day = 0.0

    return {
        "profit_base": float(profit_threshold),
        "stop_to_profit_ratio": float(stop_ratio),
        "stop_threshold": float(stop_threshold),
        "horizon_bars": int(horizon),
        "min_event_spacing": int(min_event_spacing),
        "n_events_total": int(n_events_total),
        "n_events_eval": int(n_events_eval),
        "n_trades": int(n_trades),
        "eval_days": int(eval_days),
        "trades_per_day": float(trades_per_day),
        "mean_return": float(mean_ret),
        "std_return": float(std_ret),
        "hit_rate": float(hit_rate),
        "final_equity": float(final_equity),
        "total_return": float(total_return),
        "max_drawdown": float(max_drawdown),
        "sharpe_trade": float(sharpe_trade),
        "pnl_per_day": float(pnl_per_day),
        "eval_start_date": str(eval_start_date) if eval_start_date is not None else None,
        "eval_end_date": str(eval_end_date) if eval_end_date is not None else None,
        "avg_event_duration_bars": float(event_durations.mean()) if len(event_durations) > 0 else 0.0,
    }


def save_sweep_results(
    results_df: pd.DataFrame,
    analysis: Dict[str, Any],
    symbol: str,
    timeframe: str,
    direction: str,
    outcomes_dir: str,
) -> Tuple[Path, Path]:
    """Persist sweep results to CSV and JSON analysis files."""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(outcomes_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / f"labeling_geometry_sweep_{symbol}_{timeframe}_{direction}_{timestamp}.csv"
    analysis_path = out_dir / f"labeling_geometry_sweep_{symbol}_{timeframe}_{direction}_{timestamp}_analysis.json"

    results_df.to_csv(csv_path, index=False)

    with analysis_path.open("w") as f:
        json.dump(analysis, f, indent=2, default=str)

    print(f"\nSaved sweep results to: {csv_path}")
    print(f"Saved analysis summary to: {analysis_path}")

    return csv_path, analysis_path


def main() -> None:
    args = parse_args()

    print("🚀 Labeling Triple-Barrier Geometry Sweep")
    print("=" * 60)
    print(f"Symbol: {args.symbol}")
    print(f"Exchange: {args.exchange}")
    print(f"Timeframe: {args.timeframe}")
    print(f"Direction: {args.direction}")
    print(f"Holdout fraction: {args.holdout_fraction:.2f}")
    print(f"Min trades/day (constraint): {args.min_trades_per_day:.2f}")
    print(f"Outcomes dir: {args.outcomes_dir}")

    # Try to center volatility parameters around labeling HPO knee params if available
    hpo_defaults = maybe_load_hpo_defaults(args.symbol, args.timeframe)
    vol_baseline_window = int(hpo_defaults.get("vol_baseline_window", args.vol_baseline_window))
    profit_mult_min = float(hpo_defaults.get("profit_mult_min", args.profit_mult_min))
    profit_mult_max = float(hpo_defaults.get("profit_mult_max", args.profit_mult_max))
    stop_mult_min = float(hpo_defaults.get("stop_mult_min", args.stop_mult_min))
    stop_mult_max = float(hpo_defaults.get("stop_mult_max", args.stop_mult_max))

    print("=" * 60)
    print("Volatility / threshold settings:")
    print(f"  vol_baseline_window = {vol_baseline_window}")
    print(f"  profit_mult_min/max = {profit_mult_min:.3f} / {profit_mult_max:.3f}")
    print(f"  stop_mult_min/max   = {stop_mult_min:.3f} / {stop_mult_max:.3f}")
    print(f"  transaction_cost    = {args.transaction_cost:.4f}")
    print("=" * 60)

    market_data = load_market_data(args)
    primary_signals = generate_primary_signals(market_data)
    volatility_1d, vol_factor = build_volatility_series(market_data, vol_baseline_window)

    rows: List[Dict[str, Any]] = []

    for profit_base in args.profit_bases:
        for stop_ratio in args.stop_ratios:
            for horizon in args.horizons:
                for spacing in args.min_event_spacings:
                    metrics = evaluate_geometry_config(
                        market_data=market_data,
                        primary_signals=primary_signals,
                        volatility_1d=volatility_1d,
                        vol_factor=vol_factor,
                        profit_base=float(profit_base),
                        stop_ratio=float(stop_ratio),
                        horizon=int(horizon),
                        min_event_spacing=int(spacing),
                        profit_mult_min=profit_mult_min,
                        profit_mult_max=profit_mult_max,
                        stop_mult_min=stop_mult_min,
                        stop_mult_max=stop_mult_max,
                        transaction_cost=float(args.transaction_cost),
                        holdout_fraction=float(args.holdout_fraction),
                    )

                    rows.append(metrics)

                    print(
                        "tested TP={tp:.3%}, SL={sl:.3%} (ratio={ratio:.2f}), H={H:2d}, spacing={sp:2d}: "
                        "events={neval}, tpday={tpd:.2f}, pnl/day={pnl:.4%}, Sharpe={sh:.2f}".format(
                            tp=metrics["profit_base"],
                            sl=metrics["stop_threshold"],
                            ratio=metrics["stop_to_profit_ratio"],
                            H=metrics["horizon_bars"],
                            sp=metrics["min_event_spacing"],
                            neval=metrics["n_events_eval"],
                            tpd=metrics["trades_per_day"],
                            pnl=metrics["pnl_per_day"],
                            sh=metrics["sharpe_trade"],
                        )
                    )

    results_df = pd.DataFrame(rows)
    if results_df.empty:
        print("\n❌ No results; triple-barrier engine produced zero events for all configurations.")
        sys.exit(1)

    constrained = results_df[results_df["trades_per_day"] >= args.min_trades_per_day].copy()

    analysis: Dict[str, Any] = {
        "symbol": args.symbol,
        "exchange": args.exchange,
        "timeframe": args.timeframe,
        "direction": args.direction,
        "holdout_fraction": args.holdout_fraction,
        "min_trades_per_day": args.min_trades_per_day,
        "n_configs": int(len(results_df)),
        "n_configs_constraint_satisfied": int(len(constrained)),
        "vol_baseline_window": vol_baseline_window,
        "profit_mult_min": profit_mult_min,
        "profit_mult_max": profit_mult_max,
        "stop_mult_min": stop_mult_min,
        "stop_mult_max": stop_mult_max,
    }

    if constrained.empty:
        print(
            "\n⚠️ No configuration satisfied the trades/day constraint. "
            "Best overall configuration (by pnl_per_day) will be reported without the constraint."
        )
        best_all = results_df.sort_values("pnl_per_day", ascending=False).iloc[0].to_dict()
        analysis["best_overall"] = best_all
    else:
        best = constrained.sort_values("pnl_per_day", ascending=False).iloc[0].to_dict()
        analysis["best_under_constraint"] = best

    save_sweep_results(
        results_df=results_df,
        analysis=analysis,
        symbol=args.symbol,
        timeframe=args.timeframe,
        direction=args.direction,
        outcomes_dir=args.outcomes_dir,
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⏹️ Sweep interrupted by user")
        sys.exit(1)
