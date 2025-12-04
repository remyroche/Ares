#!/usr/bin/env python3
"""Volume Force Threshold Sweep

This script performs a focused sweep over the ML Volume Force model configuration.
It optimizes the target definition (lookahead, ATR threshold, volatility percentile, trend beta)
and XGBoost hyperparameters to minimize OOF Log Loss across the 3 models (Breakout, Volatility, Trend).

It varies:
- `volume_force_target_threshold_atr`: Threshold for Breakout logic.
- `volume_force_lookahead`: Forecast horizon in bars (15m).
- `volume_force_normalization_window`: Rolling window size for feature z-scoring.
- `volume_force_volatility_percentile`: Percentile for high volatility regime.
- `volume_force_trend_beta`: Threshold for trend start.
- `volume_force_xgb_max_depth`: XGBoost Max Depth.
- `volume_force_xgb_learning_rate`: XGBoost Learning Rate.

Usage (from project root):

    python3 scripts/volume_force_threshold_sweep.py \
        --symbol ETHUSDT \
        --exchange binance \
        --timeframe 15m \
        --outcomes-dir outcomes

"""

import argparse
import asyncio
import json
import yaml
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple
import itertools

import numpy as np
import pandas as pd

# Ensure project root is on sys.path so that `src.*` imports work
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.steps.market_analysis.ml_volume_force_step import MLVolumeForceStep


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep Volume Force model thresholds and summarize quality metrics",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--symbol", type=str, default="ETHUSDT", help="Trading symbol")
    parser.add_argument("--exchange", type=str, default="binance", help="Exchange name")
    parser.add_argument("--timeframe", type=str, default="15m", help="Base timeframe")
    parser.add_argument("--direction", type=str, default="long", help="Trading direction (mostly for context)")
    parser.add_argument("--execution-mode", type=str, default="blank", help="Execution mode for the step")
    parser.add_argument("--outcomes-dir", type=str, default="outcomes", help="Directory to save sweep results")

    parser.add_argument(
        "--mode",
        type=str,
        default="sweep",
        help="Execution mode: 'sweep' for grid search, 'single' to run best config from YAML",
    )
    parser.add_argument(
        "--config-file",
        type=str,
        default="config/volume_force_config.yaml",
        help="YAML file containing best Volume Force configurations",
    )

    return parser.parse_args()


def build_base_config(args: argparse.Namespace) -> Dict[str, Any]:
    """Construct a base configuration for the volume force step."""

    base_config: Dict[str, Any] = {
        "symbol": args.symbol,
        "exchange": args.exchange,
        "timeframe": args.timeframe,
        "direction": args.direction,
        "execution_mode": args.execution_mode,

        # Standard trainer defaults (can be overridden if needed, but keeping fixed for sweep)
        "xgb_model_use_gpu": False,
    }

    return base_config


def build_sweep_configs(base_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Generate configs for the sweep."""

    # Define sweep ranges
    # Focused ATR band around lower thresholds
    atr_thresholds = [0.8, 1.0, 1.2]
    lookaheads = [8, 12, 16]  # 2h, 3h, 4h
    # Feature norm window
    norm_windows = [96, 192] # approx 1 day, 2 days

    # Target Definitions
    vol_percentiles = [70, 75, 80]
    trend_betas = [0.5, 0.75]

    # XGB Params
    max_depths = [4, 6]
    learning_rates = [0.03] # Keeping simple for now, can add 0.01

    configs: List[Dict[str, Any]] = []

    # Generate Cartesian product
    # Note: This can generate a lot of configs. Be mindful of execution time.
    # Current count: 3 * 3 * 2 * 3 * 2 * 2 * 1 = 216 configs.
    # That might be too many for a quick run.
    # Let's reduce lookaheads and norm windows for the full grid, or create a 'focused' grid.

    # Reduced Grid
    combinations = itertools.product(
        atr_thresholds,
        lookaheads,
        norm_windows,
        vol_percentiles,
        trend_betas,
        max_depths
    )

    for atr, lookahead, norm, vol_pct, trend_beta, depth in combinations:
        cfg = dict(base_config)
        cfg["volume_force_target_threshold_atr"] = atr
        cfg["volume_force_lookahead"] = lookahead
        cfg["volume_force_normalization_window"] = norm
        cfg["volume_force_volatility_percentile"] = vol_pct
        cfg["volume_force_trend_beta"] = trend_beta
        cfg["volume_force_xgb_max_depth"] = depth
        cfg["volume_force_xgb_learning_rate"] = 0.03 # Fixed for now

        # Tag for easy identification
        cfg["sweep_tag"] = f"atr{atr}_lah{lookahead}_vol{vol_pct}_tr{trend_beta}_d{depth}"

        configs.append(cfg)

    return configs


def build_single_config_from_yaml(
    args: argparse.Namespace,
    base_config: Dict[str, Any],
) -> Dict[str, Any]:
    """Load the best Volume Force configuration from YAML and merge into the base config."""

    config_path = Path(args.config_file)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        raw = yaml.safe_load(f) or {}

    vf_cfg = raw.get("volume_force", {})
    best = vf_cfg.get("global_best", {})

    cfg = dict(base_config)
    for key, value in best.items():
        if key.startswith("volume_force_"):
            cfg[key] = value

    return cfg


def save_sweep_results(
    results_df: pd.DataFrame,
    analysis: Dict[str, Any],
    symbol: str,
    outcomes_dir: str,
) -> Tuple[Path, Path]:
    """Persist sweep results to CSV and JSON."""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(outcomes_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / f"volume_force_sweep_{symbol}_{timestamp}.csv"
    analysis_path = out_dir / f"volume_force_sweep_{symbol}_{timestamp}_analysis.json"

    results_df.to_csv(csv_path, index=False)

    with open(analysis_path, "w") as f:
        json.dump(analysis, f, indent=2, default=str)

    print(f"\n💾 Saved sweep results to: {csv_path}")
    print(f"💾 Saved analysis summary to: {analysis_path}")

    return csv_path, analysis_path


def analyze_probability_distributions(df: pd.DataFrame) -> None:
    """Print summary stats for vol_force_* probabilities and target label balance."""

    prob_cols = [c for c in df.columns if c.startswith("vol_force_")]
    if prob_cols:
        print("\n📈 Probability distribution summary (vol_force_* columns)")
    for col in prob_cols:
        s = df[col].astype(float).dropna()
        if s.empty:
            continue
        print(f"\n== {col} ==")
        print(f"min={s.min():.4f}, max={s.max():.4f}, mean={s.mean():.4f}, std={s.std():.4f}")
        qs = s.quantile([0.5, 0.8, 0.9, 0.95, 0.99])
        print("quantiles:")
        for q, v in qs.items():
            print(f"  q{int(q*100):>2d}: {v:.4f}")

    label_cols = [c for c in df.columns if c.startswith("target_")]
    if label_cols:
        print("\n📊 Label balance (target_* columns)")
    for col in label_cols:
        y = df[col].dropna()
        if y.empty:
            continue
        vc = y.value_counts(normalize=True).sort_index()
        print(f"\n== {col} ==")
        for val, frac in vc.items():
            print(f"  {val}: {frac:.3f}")


def run_simple_backtest(df: pd.DataFrame, direction: str = "long") -> None:
    """Run a minimal per-target backtest using future_return_H.

    Metrics per target and threshold:
      - coverage, hit rate (if labels available), avg return, Sharpe, max drawdown.
    """

    if "future_return_H" not in df.columns:
        print("\n⚠️ future_return_H not available; skipping backtest.")
        return

    ret = df["future_return_H"].astype(float)
    ret = ret.dropna()
    if ret.empty:
        print("\n⚠️ No forward returns available; skipping backtest.")
        return

    print("\n📈 Simple H-bar backtest metrics (per target)")

    n_total = len(df)
    for target in ["breakout", "volatility", "trend"]:
        prob_col = f"vol_force_{target}"
        label_col = f"target_{target}"
        if prob_col not in df.columns:
            continue

        probs = df[prob_col].astype(float)
        rets = df["future_return_H"].astype(float)
        labels = df[label_col].astype(float) if label_col in df.columns else None

        print(f"\n=== {target.upper()} ===")

        # Fixed thresholds
        fixed_thresholds = [0.2, 0.3, 0.4]
        for thresh in fixed_thresholds:
            mask = probs >= thresh
            if not mask.any():
                continue
            r = rets[mask].dropna()
            if r.empty:
                continue
            coverage = float(mask.mean()) if n_total > 0 else 0.0
            avg_ret = float(r.mean())
            vol = float(r.std())
            sharpe = float(avg_ret / vol) if vol > 0 else 0.0
            eq = (1.0 + r).cumprod()
            dd = float((eq / eq.cummax() - 1.0).min()) if len(eq) > 0 else 0.0

            hit_rate = None
            if labels is not None:
                lab = labels[mask].dropna()
                hit_rate = float(lab.mean()) if len(lab) > 0 else None

            parts = [f"coverage={coverage:.3f}", f"avg_ret={avg_ret:.4f}", f"sharpe={sharpe:.2f}", f"max_dd={dd:.3f}"]
            if hit_rate is not None:
                parts.insert(1, f"hit_rate={hit_rate:.3f}")
            print(f"th={thresh:.2f}: " + ", ".join(parts))

        # Quantile-based thresholds: top 10%, 5%
        for q, label in ((0.9, "top10"), (0.95, "top5")):
            if probs.dropna().empty:
                continue
            q_thresh = float(probs.quantile(q))
            mask = probs >= q_thresh
            if not mask.any():
                continue
            r = rets[mask].dropna()
            if r.empty:
                continue
            coverage = float(mask.mean()) if n_total > 0 else 0.0
            avg_ret = float(r.mean())
            vol = float(r.std())
            sharpe = float(avg_ret / vol) if vol > 0 else 0.0
            eq = (1.0 + r).cumprod()
            dd = float((eq / eq.cummax() - 1.0).min()) if len(eq) > 0 else 0.0

            hit_rate = None
            if labels is not None:
                lab = labels[mask].dropna()
                hit_rate = float(lab.mean()) if len(lab) > 0 else None

            parts = [f"coverage={coverage:.3f}", f"avg_ret={avg_ret:.4f}", f"sharpe={sharpe:.2f}", f"max_dd={dd:.3f}"]
            if hit_rate is not None:
                parts.insert(1, f"hit_rate={hit_rate:.3f}")
            print(f"{label}: " + ", ".join(parts))


async def main_async() -> None:
    args = parse_args()

    print("🚀 Volume Force Multi-Task Threshold Sweep (Independent Optimization)")
    print("=" * 60)
    print(f"Symbol: {args.symbol}")
    print(f"Exchange: {args.exchange}")
    print(f"Timeframe: {args.timeframe}")
    print(f"Outcomes dir: {args.outcomes_dir}")
    print("=" * 60)

    # Build configs
    base_config = build_base_config(args)

    # Single-run mode: run best config from YAML and print detailed metrics
    mode = getattr(args, "mode", "sweep")
    if mode != "sweep":
        single_config = build_single_config_from_yaml(args, base_config)

        print("\n🔧 Running single Volume Force configuration from YAML (global_best)")
        config_subset = {k: v for k, v in single_config.items() if k.startswith("volume_force_")}
        print(json.dumps(config_subset, indent=2, default=str))

        step = MLVolumeForceStep()
        result = await step.execute(single_config)

        if not result.get("success", False):
            print("\n❌ Volume Force run failed:")
            print(result.get("error", "Unknown error"))
            return

        metrics = result.get("metrics", {})
        print("\n📊 Volume Force OOF Metrics (per target)")

        for target in ["breakout", "volatility", "trend"]:
            prefix = target
            print(f"\n=== {target.upper()} ===")
            ll = metrics.get(f"{prefix}_log_loss")
            acc = metrics.get(f"{prefix}_accuracy")
            roc = metrics.get(f"{prefix}_roc_auc")
            pr = metrics.get(f"{prefix}_pr_auc")
            brier = metrics.get(f"{prefix}_brier_score")

            if ll is not None:
                print(f"LogLoss: {ll:.4f}")
            if acc is not None:
                print(f"Accuracy: {acc:.4f}")
            if roc is not None:
                print(f"ROC AUC: {roc:.4f}")
            if pr is not None:
                print(f"PR AUC: {pr:.4f}")
            if brier is not None:
                print(f"Brier score: {brier:.4f}")

            # Fixed probability thresholds
            for thresh in (20, 30, 40, 50, 70, 90):
                key_prefix = f"{prefix}_th_{thresh}"
                cov = metrics.get(f"{key_prefix}_coverage")
                prec = metrics.get(f"{key_prefix}_precision")
                lift = metrics.get(f"{key_prefix}_lift")
                if cov is None and prec is None and lift is None:
                    continue

                parts = []
                if cov is not None:
                    parts.append(f"coverage={cov:.3f}")
                if prec is not None:
                    parts.append(f"precision={prec:.3f}")
                if lift is not None:
                    parts.append(f"lift={lift:.2f}")
                print(f"th={thresh/100:.2f}: " + ", ".join(parts))

            # Quantile-based thresholds (top20/top10/top5 by probability)
            for label in ("top20", "top10", "top5"):
                key_prefix = f"{prefix}_{label}"
                cov = metrics.get(f"{key_prefix}_coverage")
                prec = metrics.get(f"{key_prefix}_precision")
                lift = metrics.get(f"{key_prefix}_lift")
                if cov is None and prec is None and lift is None:
                    continue

                parts = []
                if cov is not None:
                    parts.append(f"coverage={cov:.3f}")
                if prec is not None:
                    parts.append(f"precision={prec:.3f}")
                if lift is not None:
                    parts.append(f"lift={lift:.2f}")
                print(f"{label}: " + ", ".join(parts))

        artifacts = result.get("artifacts", [])
        if artifacts:
            print("\n📦 Artifacts:")
            for path in artifacts:
                print(f"- {path}")

        # Load predictions artifact for distribution + backtest analysis
        preds_path = None
        for path in artifacts:
            if "ml_volume_force_predictions" in str(path):
                preds_path = path
                break

        if preds_path is not None:
            try:
                print("\n📂 Loading predictions artifact for detailed analysis:")
                print(f"- {preds_path}")
                df_preds = pd.read_hdf(preds_path)
                analyze_probability_distributions(df_preds)
                run_simple_backtest(df_preds, direction=args.direction)
            except Exception as e:
                print("\n⚠️ Failed to analyze predictions artifact:", e)

        return

    sweep_configs = build_sweep_configs(base_config)

    # Limit for safety during initial testing if list is huge
    if len(sweep_configs) > 500:
        print(f"⚠️ Warning: {len(sweep_configs)} configs generated. Truncating to 500.")
        sweep_configs = sweep_configs[:500]

    print(f"\n🔧 Generated {len(sweep_configs)} sweep configurations")

    # Initialize step
    step = MLVolumeForceStep()

    # Run batch
    results = await step.run_config_batch(sweep_configs, args.symbol, args.exchange)

    # Analyze results
    results_df, base_analysis = step.analyze_and_rank_results(results)

    if results_df.empty:
        print("\n❌ No results to save; all configurations appear to have failed.")
        return

    # Add Independent Analysis
    successful = results_df[results_df.get("success", False) == True].copy()

    analysis = base_analysis.copy()

    if not successful.empty:
        # Best Breakout
        best_breakout = successful.sort_values("breakout_log_loss", ascending=True).iloc[0].to_dict()
        analysis["best_config_breakout"] = {k: v for k, v in best_breakout.items() if k.startswith("config_")}
        analysis["best_loss_breakout"] = best_breakout.get("breakout_log_loss")

        # Best Volatility
        best_volatility = successful.sort_values("volatility_log_loss", ascending=True).iloc[0].to_dict()
        analysis["best_config_volatility"] = {k: v for k, v in best_volatility.items() if k.startswith("config_")}
        analysis["best_loss_volatility"] = best_volatility.get("volatility_log_loss")

        # Best Trend
        best_trend = successful.sort_values("trend_log_loss", ascending=True).iloc[0].to_dict()
        analysis["best_config_trend"] = {k: v for k, v in best_trend.items() if k.startswith("config_")}
        analysis["best_loss_trend"] = best_trend.get("trend_log_loss")

    # Persist outputs
    save_sweep_results(results_df, analysis, args.symbol, args.outcomes_dir)

    if successful.empty:
        print("\n⚠️ No successful configurations in sweep.")
        return

    # Print differentiated summaries
    cols = [
        "config_id",
        "breakout_log_loss",
        "volatility_log_loss",
        "trend_log_loss",
        "config_volume_force_target_threshold_atr",
        "config_volume_force_lookahead",
        "config_volume_force_volatility_percentile",
        "config_volume_force_trend_beta",
        "config_volume_force_xgb_max_depth"
    ]
    available_cols = [c for c in cols if c in successful.columns]

    print("\n🏆 Top 3 Configs for BREAKOUT (min LogLoss):")
    print(successful.sort_values("breakout_log_loss").head(3)[available_cols].to_string(index=False))

    print("\n🏆 Top 3 Configs for VOLATILITY (min LogLoss):")
    print(successful.sort_values("volatility_log_loss").head(3)[available_cols].to_string(index=False))

    print("\n🏆 Top 3 Configs for TREND (min LogLoss):")
    print(successful.sort_values("trend_log_loss").head(3)[available_cols].to_string(index=False))


def main() -> None:
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        print("\n⏹️ Sweep interrupted by user")
        sys.exit(1)


if __name__ == "__main__":
    main()
