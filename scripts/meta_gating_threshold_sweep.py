#!/usr/bin/env python3
"""Meta-Gating Threshold Sweep

This script sweeps meta-gating thresholds on top of existing meta-labeled
`labeled_data` and isotonic calibration artifacts.

Goal:
- Maximize net PnL (using `realized_return`, already cost-aware under the
  transaction cost configured in `meta_gating_config.json`).
- Enforce a minimum average trade rate (default: >= 1 trade/day on the
  evaluation window).

It mirrors the core gating logic of `MetaGatedBacktestStep` but allows a
parameter grid over:
- Probability threshold (`prob_threshold`)
- Expected-return threshold (`expected_return_threshold` via iso regressor)

Usage (from project root):

    python3 scripts/meta_gating_threshold_sweep.py \
        --symbol ETHUSDT \
        --exchange binance \
        --timeframe 15m \
        --direction long \
        --holdout-fraction 0.30 \
        --min-trades-per-day 1.0 \
        --outcomes-dir outcomes

You can also override the default grids:

    --prob-thresholds 0.50 0.55 0.60 0.65 0.70 \
    --er-thresholds 0.0000 0.0025 0.0045

`er_threshold == 0.0` is treated as disabling expected-return gating.
"""

import argparse
import json
import sys
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

# Ensure project root is on sys.path so that `src.*` imports work
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.steps.labeling.meta_gated_backtest_step import MetaGatedBacktestStep  # type: ignore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep meta-gating thresholds and summarize economic metrics",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--symbol", type=str, default="ETHUSDT", help="Trading symbol")
    parser.add_argument("--exchange", type=str, default="binance", help="Exchange name")
    parser.add_argument("--timeframe", type=str, default="15m", help="Base timeframe")
    parser.add_argument("--direction", type=str, default="long", help="Trading direction")
    parser.add_argument(
        "--execution-mode",
        type=str,
        default="full",
        help="Execution mode context for artifact loading",
    )
    parser.add_argument(
        "--holdout-fraction",
        type=float,
        default=0.30,
        help="Fraction of labeled events to reserve as evaluation holdout (last fraction)",
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
        "--prob-thresholds",
        type=float,
        nargs="+",
        default=[0.50, 0.55, 0.60, 0.65, 0.70],
        help="Grid of probability thresholds to evaluate",
    )
    parser.add_argument(
        "--er-thresholds",
        type=float,
        nargs="+",
        default=[0.0, 0.0025, 0.0045],
        help=(
            "Grid of expected-return thresholds (fraction). "
            "A value of 0.0 disables expected-return gating."
        ),
    )
    parser.add_argument(
        "--vol-quantiles",
        type=float,
        nargs="+",
        default=[0.30, 0.40, 0.50],
        help=(
            "Grid of volatility quantiles for the volatility_1d filter. "
            "Events with volatility_1d above the given quantile are eligible."
        ),
    )

    return parser.parse_args()


def load_labeled_data(
    symbol: str,
    exchange: str,
    timeframe: str,
    direction: str,
    execution_mode: str,
) -> Tuple[pd.DataFrame, MetaGatedBacktestStep]:
    """Load labeled_data artifact via MetaGatedBacktestStep's artifact router.

    Returns the labeled DataFrame and the configured step instance so that the
    loading semantics match `MetaGatedBacktestStep`.
    """
    step = MetaGatedBacktestStep()
    step.set_context(
        symbol=symbol,
        exchange=exchange,
        timeframe=timeframe,
        direction=direction,
        model="analyst",
        execution_mode=execution_mode,
    )

    artifact_name = f"labeled_data_{symbol}_{timeframe}"
    df = step._get_artifact(  # type: ignore[attr-defined]
        artifact_name=artifact_name,
        artifact_type="data",
        data_category="features",
    )

    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError(f"Labeled data artifact '{artifact_name}' not found or empty")

    df = step._normalize_datetime_index(df, "labeled_data")  # type: ignore[attr-defined]
    df = df.sort_index()
    return df, step


def load_meta_gating_artifacts(
    symbol: str,
    exchange: str,
    timeframe: str,
    direction: str,
) -> Tuple[Dict[str, Any], Any]:
    """Load meta_gating_config and isotonic regressor (if available)."""
    va_dir = Path("versioned_artifacts") / f"{symbol}_{exchange}_{timeframe}_{direction}_analyst"
    gating_path = va_dir / "meta_gating_config.json"

    if not gating_path.exists():
        raise FileNotFoundError(
            f"meta_gating_config.json not found at {gating_path}; run feature_generation_meta_labeling_step first"
        )

    with gating_path.open("r") as f_cfg:
        cfg = json.load(f_cfg)

    meta_gating = cfg.get("meta_gating", {})
    calibration_cfg = meta_gating.get("calibration", {})

    iso_rel_path = calibration_cfg.get("iso_regressor_artifact")
    iso_model = None
    if iso_rel_path:
        iso_path = va_dir / iso_rel_path
        if iso_path.exists():
            with iso_path.open("rb") as f_iso:
                iso_model = pickle.load(f_iso)

    return meta_gating, iso_model


def build_holdout_mask(
    df: pd.DataFrame,
    realized_returns: pd.Series,
    holdout_fraction: float,
) -> Tuple[pd.Series, int, int, int]:
    """Replicate MetaGatedBacktestStep's holdout selection (by fraction)."""
    event_mask = ~realized_returns.isna()
    n_events_total = int(event_mask.sum())

    eval_mask = event_mask.copy()

    if holdout_fraction <= 0.0 or holdout_fraction >= 1.0:
        # Use all labeled events
        n_events = n_events_total
    else:
        event_idx = df.index[event_mask]
        n_events = int(event_idx.size)
        n_holdout = max(1, int(round(n_events * holdout_fraction)))
        holdout_idx = event_idx[-n_holdout:]
        time_mask = df.index.isin(holdout_idx)
        eval_mask &= time_mask
        n_events = int(eval_mask.sum())

    if n_events == 0:
        raise ValueError(
            "Hold-out selection produced zero events; adjust holdout_fraction"
        )

    eval_start_date = None
    eval_end_date = None
    eval_num_days = None
    if isinstance(df.index, pd.DatetimeIndex):
        eval_index = df.index[eval_mask]
        if len(eval_index) > 0:
            eval_start_date = eval_index[0].date()
            eval_end_date = eval_index[-1].date()
            eval_num_days = int((eval_end_date - eval_start_date).days) + 1
            if eval_num_days <= 0:
                eval_num_days = 1

    return eval_mask, n_events_total, eval_num_days or 1, n_events


def evaluate_gate(
    df: pd.DataFrame,
    meta_prob: pd.Series,
    realized_returns: pd.Series,
    iso_model: Any,
    holdout_fraction: float,
    prob_threshold: float,
    er_threshold: float,
    vol_quantile: float,
) -> Dict[str, Any]:
    """Apply meta gate and compute economic metrics on the holdout segment."""
    eval_mask, n_events_total, eval_num_days, n_events_eval = build_holdout_mask(
        df, realized_returns, holdout_fraction
    )

    event_probs = meta_prob.loc[eval_mask].astype(float)
    event_returns = realized_returns.loc[eval_mask].astype(float)

    gate_mask = event_probs >= prob_threshold

    # Optional volatility filter, mirroring MetaGatedBacktestStep
    vol_threshold = None
    if "volatility_1d" in df.columns and 0.0 <= vol_quantile <= 1.0:
        v_all = df["volatility_1d"].astype(float)
        v_eval = v_all.loc[eval_mask]
        try:
            vol_threshold = float(v_eval.quantile(vol_quantile))
        except Exception:
            # Fallback to 0.40 similar to the backtest step
            vol_threshold = float(v_eval.quantile(0.40))

        if np.isfinite(vol_threshold):
            vol_mask = v_eval >= vol_threshold
            gate_mask &= vol_mask

    use_er = er_threshold > 0.0 and iso_model is not None
    expected_returns = None
    if use_er:
        prob_array = event_probs.to_numpy(dtype=float)
        er_array = iso_model.predict(prob_array)
        expected_returns = pd.Series(er_array, index=event_probs.index)
        gate_mask &= expected_returns >= er_threshold

    gated_returns = event_returns[gate_mask]
    n_trades = int(gated_returns.size)

    if isinstance(gated_returns.index, pd.DatetimeIndex) and n_trades > 0:
        trade_index = gated_returns.index.sort_values()
        gated_start_date = trade_index[0].date()
        gated_end_date = trade_index[-1].date()
        gated_num_days = int((gated_end_date - gated_start_date).days) + 1
        if gated_num_days <= 0:
            gated_num_days = 1
    else:
        gated_start_date = None
        gated_end_date = None
        gated_num_days = eval_num_days

    trades_per_day = float(n_trades) / float(gated_num_days) if gated_num_days > 0 else 0.0

    if n_trades > 0:
        mean_ret = float(gated_returns.mean())
        std_ret = float(gated_returns.std(ddof=1)) if n_trades > 1 else 0.0
        hit_rate = float((gated_returns > 0).mean())

        equity = (1.0 + gated_returns).cumprod()
        final_equity = float(equity.iloc[-1])
        total_return = final_equity - 1.0

        running_max = equity.cummax()
        drawdown = equity / running_max - 1.0
        max_drawdown = float(drawdown.min()) if drawdown.size > 0 else 0.0

        sharpe_trade = (
            float(mean_ret / std_ret) * float(np.sqrt(n_trades))
            if std_ret > 0.0
            else 0.0
        )

        # Approximate per-day return from equity over gated_num_days
        pnl_per_day = (
            float(final_equity ** (1.0 / gated_num_days) - 1.0)
            if gated_num_days > 0
            else 0.0
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
        "prob_threshold": float(prob_threshold),
        "expected_return_threshold": float(er_threshold),
        "use_expected_return": bool(er_threshold > 0.0 and iso_model is not None),
        "vol_quantile": float(vol_quantile),
        "vol_threshold": float(vol_threshold) if vol_threshold is not None else None,
        "n_events_total": int(n_events_total),
        "n_events_eval": int(n_events_eval),
        "n_trades": int(n_trades),
        "eval_days": int(eval_num_days),
        "gated_days": int(gated_num_days),
        "trades_per_day": float(trades_per_day),
        "mean_return": float(mean_ret),
        "std_return": float(std_ret),
        "hit_rate": float(hit_rate),
        "final_equity": float(final_equity),
        "total_return": float(total_return),
        "max_drawdown": float(max_drawdown),
        "sharpe_trade": float(sharpe_trade),
        "pnl_per_day": float(pnl_per_day),
        "gated_start_date": str(gated_start_date) if gated_start_date is not None else None,
        "gated_end_date": str(gated_end_date) if gated_end_date is not None else None,
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

    csv_path = out_dir / f"meta_gating_threshold_sweep_{symbol}_{timeframe}_{direction}_{timestamp}.csv"
    analysis_path = out_dir / f"meta_gating_threshold_sweep_{symbol}_{timeframe}_{direction}_{timestamp}_analysis.json"

    results_df.to_csv(csv_path, index=False)

    with analysis_path.open("w") as f:
        json.dump(analysis, f, indent=2, default=str)

    print(f"\n💾 Saved sweep results to: {csv_path}")
    print(f"💾 Saved analysis summary to: {analysis_path}")

    return csv_path, analysis_path


def main() -> None:
    args = parse_args()

    print("🚀 Meta-Gating Threshold Sweep")
    print("=" * 60)
    print(f"Symbol: {args.symbol}")
    print(f"Exchange: {args.exchange}")
    print(f"Timeframe: {args.timeframe}")
    print(f"Direction: {args.direction}")
    print(f"Holdout fraction: {args.holdout_fraction:.2f}")
    print(f"Min trades/day (constraint): {args.min_trades_per_day:.2f}")
    print(f"Outcomes dir: {args.outcomes_dir}")
    print("=" * 60)

    # Load labeled data and meta-gating artifacts
    df, _ = load_labeled_data(
        symbol=args.symbol,
        exchange=args.exchange,
        timeframe=args.timeframe,
        direction=args.direction,
        execution_mode=args.execution_mode,
    )

    if "realized_return" not in df.columns or "meta_probability" not in df.columns:
        raise ValueError(
            "labeled_data must contain 'realized_return' and 'meta_probability' columns"
        )

    meta_gating, iso_model = load_meta_gating_artifacts(
        symbol=args.symbol,
        exchange=args.exchange,
        timeframe=args.timeframe,
        direction=args.direction,
    )

    tx_cost = float(meta_gating.get("transaction_cost", 0.0))
    print(f"Configured transaction_cost (per step config): {tx_cost:.4f}")
    print("Assuming realized_return is already net of this cost (≈0.3% round trip in your setup).")

    realized_returns = df["realized_return"].astype(float)
    meta_prob = df["meta_probability"].astype(float)

    rows: List[Dict[str, Any]] = []

    for vol_q in args.vol_quantiles:
        for prob_thr in args.prob_thresholds:
            for er_thr in args.er_thresholds:
                metrics = evaluate_gate(
                    df=df,
                    meta_prob=meta_prob,
                    realized_returns=realized_returns,
                    iso_model=iso_model,
                    holdout_fraction=args.holdout_fraction,
                    prob_threshold=float(prob_thr),
                    er_threshold=float(er_thr),
                    vol_quantile=float(vol_q),
                )
                rows.append(metrics)
                print(
                    f"tested vol_q>={vol_q:.2f}, prob>={prob_thr:.3f}, er>={er_thr:.4f}: "
                    f"trades={metrics['n_trades']}, tpday={metrics['trades_per_day']:.2f}, "
                    f"pnl/day={metrics['pnl_per_day']:.4%}, Sharpe={metrics['sharpe_trade']:.2f}"
                )

    results_df = pd.DataFrame(rows)
    if results_df.empty:
        print("\n❌ No results; gating produced zero trades for all configurations.")
        sys.exit(1)

    # Apply trade-rate constraint and select best by pnl_per_day
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
