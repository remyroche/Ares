#!/usr/bin/env python3
"""XGB Meso Trend Sweep

This script performs a focused sweep over feature parameters for the
XGB Meso Trend step.

It generates variations for:
- Meso trend target volatility window
- EWMA short and long spans
- HTF feature lookbacks (RSI, ATR, MACD)

It runs `XGBMesoTrendStep.run_config_batch` to execute the sweep and
analyzes results based on OOF RMSE.

Usage:
    python3 scripts/xgb_meso_trend_sweep.py \
        --symbol ETHUSDT \
        --timeframe 15m \
        --max-configs 50
"""

import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import numpy as np

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.versioned_artifacts import VersionedArtifactStore
from src.training.steps.market_analysis.xgb_meso_regime_step import XGBMesoTrendStep


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep feature parameters for XGB Meso Trend",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--symbol", type=str, default="ETHUSDT", help="Trading symbol")
    parser.add_argument("--exchange", type=str, default="binance", help="Exchange name")
    parser.add_argument("--timeframe", type=str, default="15m", help="Regime timeframe (e.g. 15m)")
    parser.add_argument("--direction", type=str, default="long", help="Trading direction")
    parser.add_argument("--max-configs", type=int, default=30, help="Max number of configs to test")
    parser.add_argument("--outcomes-dir", type=str, default="outcomes", help="Directory to save sweep results")
    parser.add_argument("--execution-mode", type=str, default="blank", choices=["full", "light", "blank"], help="Execution mode (full/light/blank)")
    parser.add_argument("--mode", type=str, default="sweep", choices=["sweep", "best"], help="Whether to run a full sweep or only the best-known config")
    return parser.parse_args()


def build_base_config(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "symbol": args.symbol,
        "exchange": args.exchange,
        "regime_timeframe": args.timeframe,
        "direction": args.direction,
        "execution_mode": args.execution_mode,
        "meso_sweep_max_configs": args.max_configs,
    }


def save_sweep_results(
    results_df: pd.DataFrame,
    analysis: Dict[str, Any],
    symbol: str,
    outcomes_dir: str,
) -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(outcomes_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / f"meso_trend_sweep_{symbol}_{timestamp}.csv"
    analysis_path = out_dir / f"meso_trend_sweep_{symbol}_{timestamp}_analysis.json"

    results_df.to_csv(csv_path, index=False)

    with open(analysis_path, "w") as f:
        json.dump(analysis, f, indent=2, default=str)

    print(f"\n💾 Saved sweep results to: {csv_path}")
    print(f"💾 Saved analysis summary to: {analysis_path}")


def evaluate_trading_effectiveness(args: argparse.Namespace) -> None:
    """Evaluate simple trading rules based on meso scores on the test window.

    This loads the latest xgb_meso_trend_training_data_15m version from the
    appropriate VersionedArtifactStore, slices it to the test period defined in
    the temporal split config, and computes basic Sharpe and hit-rate metrics
    using the vol-normalized meso_trend_target as a proxy for PnL.
    """

    try:
        # Load temporal split config to get test window
        split_path = (
            Path("config/temporal_splits")
            / f"{args.symbol}_{args.exchange}_{args.timeframe}_burnin.json"
        )
        if not split_path.exists():
            print(
                f"\n⚠️ Temporal split config not found at {split_path}, "
                "skipping trading effectiveness evaluation."
            )
            return

        with split_path.open("r") as f:
            split_cfg = json.load(f)

        test_cfg = split_cfg.get("test", {})
        test_start = pd.to_datetime(test_cfg.get("start"))
        test_end = pd.to_datetime(test_cfg.get("end"))

        if pd.isna(test_start) or pd.isna(test_end):
            print(
                "\n⚠️ Invalid test window in temporal split config; "
                "skipping trading effectiveness evaluation."
            )
            return

        # Locate VersionedArtifactStore for this symbol/exchange/timeframe/direction
        store_name = (
            f"{args.symbol}_{args.exchange}_{args.timeframe}_{args.direction}_regime_alpha"
        )
        store_path = Path("versioned_artifacts") / store_name

        if not store_path.exists():
            print(
                f"\n⚠️ VersionedArtifactStore not found at {store_path}, "
                "skipping trading effectiveness evaluation."
            )
            return

        store = VersionedArtifactStore(store_path)

        # Load latest meso training data and filter on explicit timestamp column.
        # The VersionedArtifactStore uses an integer index for this artifact, so
        # we slice by the saved timestamp column instead of using
        # query_by_index_range on the index.
        view = store.get_view()
        df_full = view.to_pandas()

        if df_full is None or df_full.empty:
            print(
                "\n⚠️ Meso trend training data in VersionedArtifactStore is empty; "
                "skipping trading effectiveness evaluation."
            )
            return

        timestamp_col = "timestamp"
        if timestamp_col not in df_full.columns:
            # Fallback to a few common alternatives if schema changes.
            for alt in ("ts", "datetime", "index"):
                if alt in df_full.columns:
                    timestamp_col = alt
                    break
            else:
                print(
                    "\n⚠️ No timestamp-like column found in meso trend data; "
                    "skipping trading effectiveness evaluation."
                )
                return

        df_full[timestamp_col] = pd.to_datetime(df_full[timestamp_col])

        window_mask = (df_full[timestamp_col] >= test_start) & (
            df_full[timestamp_col] <= test_end
        )
        # Attempt to load target vol if available, for fee estimation
        has_vol = "meso_trend_target_vol" in df_full.columns
        cols = ["meso_trend_target", "meso_trend_score_continuous"]
        if has_vol:
            cols.append("meso_trend_target_vol")

        df_test = df_full.loc[window_mask, cols]

        if df_test is None or df_test.empty:
            print(
                "\n⚠️ No meso trend data found in test window; "
                "skipping trading effectiveness evaluation."
            )
            return

        df_test = df_test.dropna(subset=cols)
        if df_test.empty:
            print(
                "\n⚠️ Test window meso trend data is empty after dropping NaNs; "
                "skipping trading effectiveness evaluation."
            )
            return

        target = df_test["meso_trend_target"].astype(float)
        score = df_test["meso_trend_score_continuous"].astype(float)
        vol = df_test["meso_trend_target_vol"].astype(float) if has_vol else None

        # Calculate duration in days for trades/day metric
        total_days = (test_end - test_start).total_seconds() / 86400.0

        print("\n🔎 Trading effectiveness on test window (vol-normalized target):")
        print(
            f"   Test window: {test_start} → {test_end} "
            f"(n={len(df_test)}, days={total_days:.1f})"
        )

        def compute_stats(returns: pd.Series, label: str, vol_col: pd.Series = None) -> None:
            if returns is None or returns.empty:
                print(f"\n⚠️ {label}: no returns to evaluate.")
                return

            mean_ret = float(returns.mean())
            std_ret = float(returns.std(ddof=0))
            sharpe = mean_ret / std_ret if std_ret > 0 else float("nan")
            hit_rate = float((returns > 0).mean())
            trades_per_day = len(returns) / total_days

            print(f"\n📈 {label}:")
            print(f"   Samples (trades): {len(returns)}")
            print(f"   Avg trades/day: {trades_per_day:.1f}")
            print(f"   Mean (per-sample): {mean_ret:.4f}")
            print(f"   Std  (per-sample): {std_ret:.4f}")
            print(f"   Sharpe (per-sample, unannualized): {sharpe:.3f}")
            print(f"   Hit rate (>0): {hit_rate:.3%}")

            if vol_col is not None:
                # 0.3% fee per round trade.
                # Since target is vol-normalized (target_raw / vol),
                # we must subtract (fee / vol) to get net vol-normalized return.
                # Fee is always positive cost, so we subtract from the PnL vector.
                # Align vol to returns index
                vol_subset = vol_col.loc[returns.index]
                cost_penalty = 0.003 / (vol_subset + 1e-8)
                net_returns = returns - cost_penalty

                mean_net = float(net_returns.mean())
                std_net = float(net_returns.std(ddof=0))
                sharpe_net = mean_net / std_net if std_net > 0 else float("nan")
                hit_rate_net = float((net_returns > 0).mean())

                print(f"   -- With 0.3% fees --")
                print(f"   Mean (Net): {mean_net:.4f}")
                print(f"   Sharpe (Net): {sharpe_net:.3f}")
                print(f"   Hit rate (Net): {hit_rate_net:.3%}")

        # Strategy 1: Long/short by sign of meso score (uses all samples with non-zero score)
        pos_ls = pd.Series(np.sign(score.values), index=score.index)
        mask_ls = pos_ls != 0
        if mask_ls.any():
            ret_ls = pos_ls[mask_ls] * target[mask_ls]
            vol_ls = vol[mask_ls] if vol is not None else None
            compute_stats(
                ret_ls,
                "Meso score long/short on vol-normalized target (non-zero scores)",
                vol_ls
            )
        else:
            print("\n⚠️ Long/short strategy: all meso scores are zero; no trades.")

        # Strategy 2: Long/flat on top 20% of meso scores (evaluate trades only)
        try:
            threshold = score.quantile(0.8)
            pos_lf = (score > threshold).astype(float)
            mask_lf = pos_lf > 0
            if mask_lf.any():
                ret_lf = pos_lf[mask_lf] * target[mask_lf]
                vol_lf = vol[mask_lf] if vol is not None else None
                compute_stats(
                    ret_lf,
                    "Meso score long/flat (top 20%) on vol-normalized target",
                    vol_lf
                )
            else:
                print(
                    "\n⚠️ Long/flat strategy: no samples exceed top-20% "
                    "threshold; no trades."
                )
        except Exception as quantile_exc:
            print(
                f"\n⚠️ Failed to compute long/flat strategy threshold: {quantile_exc}"
            )

    except Exception as e:
        print(f"\n⚠️ Trading effectiveness evaluation failed: {e}")


async def main_async() -> None:
    args = parse_args()

    print("🚀 XGB Meso Trend Feature Sweep")
    print("=" * 60)
    print(f"Symbol: {args.symbol}")
    print(f"Timeframe: {args.timeframe}")
    print(f"Max Configs: {args.max_configs}")
    print("=" * 60)

    base_config = build_base_config(args)

    step = XGBMesoTrendStep()

    if getattr(args, "mode", "sweep") == "best":
        best_config_overrides: Dict[str, Any] = {
            "meso_trend_target_vol_window": 480,
            "meso_ewma_short": 12,
            "meso_ewma_long": 16,
            "meso_htf_rsi_period": 14,
            "meso_htf_atr_period": 21,
            "meso_htf_macd_fast": 12,
            "meso_htf_macd_slow": 21,
        }
        best_config: Dict[str, Any] = {**base_config, **best_config_overrides}

        print("\n🎯 Running single best-config training (no sweep)...")
        print("Best-config overrides:", best_config_overrides)

        result = await step.execute(best_config)
        if not isinstance(result, dict) or not result.get("success", False):
            print("\n❌ Best-config run failed:", result.get("error") if isinstance(result, dict) else result)
            return

        evaluate_trading_effectiveness(args)
        return

    configs = step.generate_config_variations(base_config)

    if not configs:
        print("❌ No configurations generated.")
        return

    results = await step.run_config_batch(configs, args.symbol, args.exchange)

    results_df, analysis = step.analyze_and_rank_results(results)

    if results_df.empty:
        print("\n❌ No results to save.")
        return

    save_sweep_results(results_df, analysis, args.symbol, args.outcomes_dir)

    successful = results_df[results_df.get("success", False) == True].copy()
    if not successful.empty:
        print("\n🏆 Top 5 Configurations (by RMSE):")
        cols = ["config_id", "rmse", "n_samples", "config_signature"]
        print(successful[cols].head(5).to_string(index=False))
    else:
        print("\n⚠️ No successful configurations.")

    evaluate_trading_effectiveness(args)


def main() -> None:
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        print("\n⏹️ Sweep interrupted by user")
        sys.exit(1)


if __name__ == "__main__":
    main()
