from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Dict, Any

import numpy as np
import pandas as pd

from extreme_price_movements.tpsl_optimiser import load_step_module
from extreme_price_movements.utils import tprint


@dataclass(frozen=True)
class Policy:
    mode: Literal["train_baseline", "inference"] = "train_baseline"
    params_path: str | None = None

    def baseline_params(self) -> dict:
        return {
            "tp_mult": 3.0,
            "sl_mult": 1.0,
            "atr_scale_lo": 0.6,
            "atr_scale_hi": 2.5,
            "risk_cut_mode": "TIMES",
            "theta0": 1.2,
            "theta_mae_min": 0.3,
            "lambda_rv": 0.5,
            "lambda_rng": 0.25,
            "sizing": {"k": 8.0, "c0": 0.70, "s_min": 0.03, "s_max": 0.15},
        }

    def resolve_params(self, bucket: str) -> dict:
        if self.mode == "train_baseline":
            return self.baseline_params()
        if not self.params_path:
            return self.baseline_params()
        p = Path(self.params_path)
        if not p.exists():
            return self.baseline_params()
        payload = json.loads(p.read_text())
        return payload.get("buckets", {}).get(str(bucket), self.baseline_params())


def _adapt_backtest_columns(trades: pd.DataFrame) -> pd.DataFrame:
    """Map backtest_results.csv columns to tpsl_optimiser expected schema."""
    df = trades.copy()
    # timestamp
    if "timestamp" not in df.columns and "entry_ts" in df.columns:
        df["timestamp"] = pd.to_datetime(df["entry_ts"], utc=True)
    # confidence
    if "confidence" not in df.columns and "score" in df.columns:
        df["confidence"] = df["score"].abs().clip(0, 1)
    # entry_price
    if "entry_price" not in df.columns and "entry_px" in df.columns:
        df["entry_price"] = df["entry_px"]
    # exit_price — reconstruct from entry_px + gross_ret
    if "exit_price" not in df.columns:
        if "entry_px" in df.columns and "gross_ret" in df.columns and "side" in df.columns:
            is_long = (df["side"] == "long").astype(int)
            df["exit_price"] = np.where(
                is_long == 1,
                df["entry_px"] * (1.0 + df["gross_ret"]),
                df["entry_px"] * (1.0 - df["gross_ret"]),
            )
        else:
            df["exit_price"] = df.get("entry_price", df.get("entry_px", 1.0))
    # is_long
    if "is_long" not in df.columns and "side" in df.columns:
        df["is_long"] = (df["side"] == "long").astype(int)
    # bucket
    if "bucket" not in df.columns and "side" in df.columns and "dom" in df.columns:
        df["bucket"] = df["side"].str.upper() + "_" + df["dom"].str.upper()
    return df


def run_optimise_step(trades: pd.DataFrame, atr_15m: pd.Series, output_path: str, policy: Policy | None = None) -> dict:
    policy = policy or Policy(mode="train_baseline")

    # Adapt column names from backtest output to tpsl_optimiser schema
    trades = _adapt_backtest_columns(trades)

    m00 = load_step_module("00_load_trades.py")
    m10 = load_step_module("10_tp_sl_calibration.py")
    m20 = load_step_module("20_loss_limiter_opt.py")
    m30 = load_step_module("30_profit_exit_opt.py")
    m40 = load_step_module("40_position_sizing_opt.py")
    m50 = load_step_module("50_eval_holdout_report.py")
    mw = load_step_module("write_params_json.py")

    buckets = list(pd.Series(trades["bucket"].astype(str).unique()).sort_values())[:4]
    all_out = {}

    # List to collect all trials across all buckets and steps
    all_trials_log = []

    for bucket in buckets:
        bucket_df = m00.load_trades_for_bucket(trades, bucket)
        if bucket_df.empty:
            continue

        # Determine test split index (same as m50 uses)
        n = len(bucket_df)
        split_idx = max(1, int(n * 0.30)) if n > 0 else 0

        atr_scale = m10.compute_atr_scale(atr_15m.reindex(bucket_df.index).ffill().fillna(atr_15m.median()))

        # Step 10: TP/SL Calibration
        tp_sl, trials_10 = m10.calibrate_tp_sl(bucket_df, atr_scale, test_split_idx=split_idx)
        trials_10["bucket"] = bucket
        trials_10["step"] = "10_tp_sl"
        all_trials_log.append(trials_10)

        sl_pct = tp_sl["sl_mult"] * atr_scale.to_numpy()

        # Step 20: Loss Limiter Optimization
        risk_cut, trials_20 = m20.optimise_loss_limiter(bucket_df, sl_pct=sl_pct, test_split_idx=split_idx)
        trials_20["bucket"] = bucket
        trials_20["step"] = "20_risk_cut"
        all_trials_log.append(trials_20)

        raw_returns = np.where(bucket_df["is_long"].astype(int).to_numpy() == 1,
                               (bucket_df["exit_price"] - bucket_df["entry_price"]) / bucket_df["entry_price"],
                               (bucket_df["entry_price"] - bucket_df["exit_price"]) / bucket_df["entry_price"])

        tp_pct_entry = tp_sl["tp_mult"] * atr_scale.to_numpy()

        # Step 30: Profit Exit Optimization
        profit, trials_30 = m30.optimise_profit_exit(bucket_df, raw_returns, tp_pct_entry=tp_pct_entry, fee_pct=0.005, test_split_idx=split_idx)
        trials_30["bucket"] = bucket
        trials_30["step"] = "30_profit_exit"
        all_trials_log.append(trials_30)

        # Step 40: Position Sizing Optimization
        # Pass raw exit/entry/is_long as original code did, but metrics will use them
        sizing, trials_40 = m40.optimise_position_sizing(
            bucket_df,
            bucket_df["exit_price"].to_numpy(dtype=float),
            bucket_df["entry_price"].to_numpy(dtype=float),
            bucket_df["is_long"].to_numpy(dtype=int),
            bucket_df["confidence"].to_numpy(dtype=float),
            test_split_idx=split_idx
        )
        trials_40["bucket"] = bucket
        trials_40["step"] = "40_sizing"
        all_trials_log.append(trials_40)

        pos_size = m40.sigmoid_sizing(bucket_df["confidence"].to_numpy(dtype=float), sizing["k"], sizing["c0"], sizing["s_min"], sizing["s_max"])
        net_returns = raw_returns * pos_size - 0.005
        report = m50.evaluate_holdout(bucket_df, net_returns)

        combined = {
            "policy_mode": policy.mode,
            "baseline_seed": policy.resolve_params(bucket),
            "tp_sl": tp_sl,
            "loss_limiter": risk_cut,
            "profit_exit": profit,
            "position_sizing": sizing,
            "evaluation": report,
        }
        all_out[bucket] = combined
        mw.merge_and_write_params(output_path, bucket, combined)
        tprint(f"optimise: bucket={bucket} trades={len(bucket_df)} saved={output_path}")

    # Concatenate and save consolidated report CSV
    if all_trials_log:
        consolidated_df = pd.concat(all_trials_log, ignore_index=True)
        # Reorder columns to put context first
        cols = ["bucket", "step"] + [c for c in consolidated_df.columns if c not in ["bucket", "step"]]
        consolidated_df = consolidated_df[cols]

        # Save CSV alongside JSON output (same directory, change extension)
        csv_path = Path(output_path).with_suffix(".csv")
        consolidated_df.to_csv(csv_path, index=False)
        tprint(f"optimise: detailed report saved={csv_path}")

    return all_out
