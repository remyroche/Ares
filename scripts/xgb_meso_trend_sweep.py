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

        has_target = "meso_trend_target" in df_full.columns
        has_vol = "meso_trend_target_vol" in df_full.columns

        window_mask = (df_full[timestamp_col] >= test_start) & (
            df_full[timestamp_col] <= test_end
        )

        base_cols = ["meso_trend_forward_return_2h", "meso_trend_score_continuous"]
        cols = list(base_cols)
        if has_target:
            cols.append("meso_trend_target")
        if has_vol:
            cols.append("meso_trend_target_vol")

        missing_cols = [c for c in base_cols if c not in df_full.columns]
        if missing_cols:
            print(
                "\n⚠️ Required columns for 2h trading evaluation are missing from "
                f"meso trend data: {missing_cols}; skipping trading effectiveness "
                "evaluation."
            )
            return

        df_test = df_full.loc[window_mask, cols]

        if df_test is None or df_test.empty:
            print(
                "\n⚠️ No meso trend data found in test window; "
                "skipping trading effectiveness evaluation."
            )
            return

        df_test = df_test.dropna(subset=base_cols)
        if df_test.empty:
            print(
                "\n⚠️ Test window meso trend data is empty after dropping NaNs; "
                "skipping trading effectiveness evaluation."
            )
            return

        target = df_test["meso_trend_forward_return_2h"].astype(float)
        score = df_test["meso_trend_score_continuous"].astype(float)
        target_vol_norm = (
            df_test["meso_trend_target"].astype(float) if has_target else None
        )

        print("\n🔎 Trading effectiveness on test window (2h forward returns):")
        print(
            f"   Test window: {test_start} → {test_end} "
            f"(n={len(df_test)})"
        )

        def compute_stats(returns: pd.Series, label: str) -> None:
            if returns is None or returns.empty:
                print(f"\n⚠️ {label}: no returns to evaluate.")
                return

            mean_ret = float(returns.mean())
            std_ret = float(returns.std(ddof=0))
            sharpe = mean_ret / std_ret if std_ret > 0 else float("nan")
            hit_rate = float((returns > 0).mean())
            downside = returns[returns < 0]
            downside_std = float(downside.std(ddof=0)) if not downside.empty else float("nan")
            sortino = mean_ret / downside_std if downside_std > 0 else float("nan")

            p05 = float(returns.quantile(0.05))
            p95 = float(returns.quantile(0.95))

            pos = returns[returns > 0]
            neg = returns[returns < 0]
            avg_gain = float(pos.mean()) if not pos.empty else float("nan")
            avg_loss = float(neg.mean()) if not neg.empty else float("nan")
            profit_factor = (
                abs(avg_gain / avg_loss)
                if avg_gain > 0 and avg_loss < 0
                else float("nan")
            )

            cum = returns.cumsum()
            drawdown = cum - cum.cummax()
            max_dd = float(drawdown.min()) if not drawdown.empty else float("nan")

            print(f"\n📈 {label}:")
            print(f"   Samples (trades): {len(returns)}")
            print(f"   Mean (per-sample): {mean_ret:.4f}")
            print(f"   Std  (per-sample): {std_ret:.4f}")
            print(f"   Sharpe (per-sample, unannualized): {sharpe:.3f}")
            print(f"   Hit rate (>0): {hit_rate:.3%}")

            print(f"   Sortino (downside): {sortino:.3f}")
            print(f"   5th/95th pct: {p05:.4f} / {p95:.4f}")
            print(f"   Avg gain / loss: {avg_gain:.4f} / {avg_loss:.4f}")
            print(f"   Profit factor: {profit_factor:.3f}")
            print(f"   Max drawdown (cum): {max_dd:.4f}")

        def compute_ic(target_series: pd.Series, label: str) -> None:
            if target_series is None or target_series.empty:
                print(f"\n⚠️ {label}: no data for IC.")
                return

            common = target_series.index.intersection(score.index)
            if common.empty:
                print(f"\n⚠️ {label}: no overlapping index for IC.")
                return

            t = target_series.loc[common].astype(float)
            s = score.loc[common].astype(float)

            if t.nunique() <= 1 or s.nunique() <= 1:
                print(f"\n⚠️ {label}: insufficient variation for IC.")
                return

            pearson = float(t.corr(s, method="pearson"))
            spearman = float(t.corr(s, method="spearman"))

            print(f"\n📊 {label} vs meso score IC:")
            print(f"   Pearson IC:  {pearson:.4f}")
            print(f"   Spearman IC: {spearman:.4f}")

        def compute_deciles(target_series: pd.Series, label: str) -> None:
            if target_series is None or target_series.empty:
                return

            common = target_series.index.intersection(score.index)
            if len(common) < 20:
                return

            s = score.loc[common].astype(float)
            t = target_series.loc[common].astype(float)

            try:
                ranks = s.rank(method="first")
                deciles = pd.qcut(ranks, 10, labels=False, duplicates="drop")
            except Exception:
                print(f"\n⚠️ {label}: failed to compute score deciles.")
                return

            df_dec = pd.DataFrame({"decile": deciles, "target": t})
            dec_means = df_dec.groupby("decile")["target"].mean()

            print(
                f"\n📊 {label}: mean target by score decile (0=lowest, 9=highest):"
            )
            print(dec_means.to_string())

        def compute_top_bucket_calibration(
            target_series: pd.Series,
            label: str,
            top_pct: float = 0.2,
        ) -> None:
            """Calibration-style summary for top-pct scores vs the rest."""
            if target_series is None or target_series.empty:
                return

            common = target_series.index.intersection(score.index)
            if len(common) < 20:
                return

            t = target_series.loc[common].astype(float)
            s = score.loc[common].astype(float)

            try:
                threshold = s.quantile(1.0 - top_pct)
            except Exception:
                return

            mask_top = s >= threshold
            mask_rest = ~mask_top

            if not mask_top.any() or not mask_rest.any():
                return

            top = t[mask_top]
            rest = t[mask_rest]

            def _summary(x: pd.Series) -> Tuple[float, float, float, float]:
                mean_ret = float(x.mean())
                std_ret = float(x.std(ddof=0))
                sharpe = mean_ret / std_ret if std_ret > 0 else float("nan")
                hit_rate = float((x > 0).mean())
                return mean_ret, std_ret, sharpe, hit_rate

            top_mean, top_std, top_sharpe, top_hit = _summary(top)
            rest_mean, rest_std, rest_sharpe, rest_hit = _summary(rest)

            var_s = float(s.var(ddof=0))
            slope_global = (
                float(np.cov(s, t)[0, 1]) / var_s if var_s > 0 else float("nan")
            )

            s_top = s[mask_top]
            var_s_top = float(s_top.var(ddof=0))
            slope_top = (
                float(np.cov(s_top, top)[0, 1]) / var_s_top
                if var_s_top > 0
                else float("nan")
            )

            print(
                f"\n📊 Calibration for {label} (top {int(top_pct * 100)}% scores vs rest):"
            )
            print(
                f"   Top bucket:   mean={top_mean:.4f}, std={top_std:.4f}, "
                f"Sharpe={top_sharpe:.3f}, hit={top_hit:.3%}"
            )
            print(
                f"   Rest bucket:  mean={rest_mean:.4f}, std={rest_std:.4f}, "
                f"Sharpe={rest_sharpe:.3f}, hit={rest_hit:.3%}"
            )
            print(
                f"   Lift (top - rest): mean={top_mean - rest_mean:.4f}, "
                f"hit={top_hit - rest_hit:.3%}"
            )
            print(
                f"   Slope target~score (global/top): {slope_global:.4f} / {slope_top:.4f}"
            )

        # Strategy 1: Long/short by sign of meso score (uses all samples with non-zero score)
        pos_ls = pd.Series(np.sign(score.values), index=score.index)
        mask_ls = pos_ls != 0
        if mask_ls.any():
            ret_ls = pos_ls[mask_ls] * target[mask_ls]
            compute_stats(
                ret_ls,
                "Meso score long/short on 2h forward returns (non-zero scores)",
            )
            if target_vol_norm is not None:
                ret_ls_vol = pos_ls[mask_ls] * target_vol_norm[mask_ls]
                compute_stats(
                    ret_ls_vol,
                    "Meso score long/short on vol-normalized target (non-zero scores)",
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
                compute_stats(
                    ret_lf,
                    "Meso score long/flat (top 20%) on 2h forward returns",
                )
                if target_vol_norm is not None:
                    ret_lf_vol = pos_lf[mask_lf] * target_vol_norm[mask_lf]
                    compute_stats(
                        ret_lf_vol,
                        "Meso score long/flat (top 20%) on vol-normalized target",
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

        compute_ic(target, "2h forward returns")
        if target_vol_norm is not None:
            compute_ic(target_vol_norm, "Vol-normalized target")

        compute_deciles(target, "2h forward returns")
        if target_vol_norm is not None:
            compute_deciles(target_vol_norm, "Vol-normalized target")

        compute_top_bucket_calibration(target, "2h forward returns", top_pct=0.2)
        if target_vol_norm is not None:
            compute_top_bucket_calibration(
                target_vol_norm,
                "Vol-normalized target",
                top_pct=0.2,
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
