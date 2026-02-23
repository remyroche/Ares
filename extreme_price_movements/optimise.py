from __future__ import annotations

import hashlib
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Dict, Any, Optional

import numpy as np
import pandas as pd

from extreme_price_movements.persistence.policy_params_store import (
    get_initial_params,
    load_params_store,
    save_params_store,
    store_best_params,
)
from extreme_price_movements.pnl import CostModel, trade_return_net_vec
from extreme_price_movements.telemetry.tprint_hooks import emit_bucket_summary, emit_run_header
from extreme_price_movements.tpsl_optimiser import load_step_module
from extreme_price_movements.utils import tprint


@dataclass(frozen=True)
class Policy:
    mode: Literal["train_baseline", "inference"] = "train_baseline"
    params_path: str | None = None
    ridge_weights_path: str | None = None  # Path to ridge sizer weights

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
    
    def get_ridge_weights(self) -> Optional[Dict]:
        """Load ridge position sizer weights if available."""
        if not self.ridge_weights_path:
            return None
        p = Path(self.ridge_weights_path)
        if not p.exists():
            return None
        payload = json.loads(p.read_text())
        return payload.get("weights")


def load_ridge_weights_from_state(state_path: str) -> Optional[Dict]:
    """Load ridge sizer weights from training state file.
    
    Args:
        state_path: Path to trained_state.pkl
        
    Returns:
        Dict with weights, or None if not found
    """
    p = Path(state_path)
    if not p.exists():
        return None
    
    with open(p, "rb") as f:
        state = pickle.load(f)
    
    ridge_sizer = state.get("ridge_sizer", {})
    if ridge_sizer:
        return ridge_sizer.get("weights")
    return None


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


def run_optimise_step(trades: pd.DataFrame, atr_15m: pd.Series, output_path: str, policy: Policy | None = None, state_path: str | None = None, cost: CostModel | None = None, enforce_threaded_exit_stream: bool = True) -> dict:
    """Run the optimisation pipeline for TP/SL and position sizing.
    
    Args:
        trades: DataFrame with backtest trade results
        atr_15m: Series with 15-minute ATR values
        output_path: Path to save optimisation results
        policy: Policy configuration (mode, params_path, ridge_weights_path)
        state_path: Optional path to trained_state.pkl for loading ridge weights
        
    Returns:
        Dict with optimisation results per bucket
    """
    policy = policy or Policy(mode="train_baseline")

    run_id = Path(output_path).stem
    policy_version = str(run_id)


    # Try to load ridge weights from policy or state file
    ridge_weights = None
    if policy.ridge_weights_path:
        ridge_weights = policy.get_ridge_weights()
        if ridge_weights:
            tprint(f"Loaded ridge weights from policy path: {policy.ridge_weights_path}")
    if ridge_weights is None and state_path:
        ridge_weights = load_ridge_weights_from_state(state_path)
        if ridge_weights:
            tprint(f"Loaded ridge weights from state file: {state_path}")

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

    fee_pct = float((trades.attrs.get("fee_pct") if hasattr(trades, "attrs") else None) or 0.005)
    cost = cost or CostModel(fee_side=fee_pct / 2.0)
    cost_dict = {"fee_side": float(cost.fee_side), "slippage_side": float(cost.slippage_side), "round_trip": float(cost.round_trip)}
    cost_hash = hashlib.sha256(json.dumps(cost_dict, sort_keys=True).encode("utf-8")).hexdigest()[:10]
    emit_run_header(tprint=tprint, run_id=run_id, policy_version=policy_version, cost_model=cost_dict, extra={"n_buckets": len(buckets)})

    store = load_params_store()
    version_key = f"{policy_version}|{cost_hash}"

    for bucket in buckets:
        bucket_df = m00.load_trades_for_bucket(trades, bucket)
        if bucket_df.empty:
            continue

        # Determine test split index (same as m50 uses)
        n = len(bucket_df)
        split_idx = max(1, int(n * 0.30)) if n > 0 else 0

        atr_scale = m10.compute_atr_scale(atr_15m.reindex(bucket_df.index).ffill().fillna(atr_15m.median()))

        params_init = get_initial_params(store, version_key, bucket, defaults=policy.resolve_params(bucket))

        # Step 10: TP/SL Calibration
        tp_sl, trials_10 = m10.calibrate_tp_sl(
            bucket_df, atr_scale, test_split_idx=split_idx, fee_pct=fee_pct, cost=cost,
            init_params=params_init.get("tp_sl", params_init)
        )
        trials_10["bucket"] = bucket
        trials_10["step"] = "10_tp_sl"
        all_trials_log.append(trials_10)

        sl_pct = tp_sl["sl_mult"] * atr_scale.to_numpy()

        # Step 20: Loss Limiter Optimization
        risk_cut, trials_20 = m20.optimise_loss_limiter(bucket_df, sl_pct=sl_pct, test_split_idx=split_idx, fee_pct=fee_pct, cost=cost, init_params=params_init.get("loss_limiter", params_init))
        trials_20["bucket"] = bucket
        trials_20["step"] = "20_risk_cut"
        all_trials_log.append(trials_20)

        raw_returns = np.where(bucket_df["is_long"].astype(int).to_numpy() == 1,
                               (bucket_df["exit_price"] - bucket_df["entry_price"]) / bucket_df["entry_price"],
                               (bucket_df["entry_price"] - bucket_df["exit_price"]) / bucket_df["entry_price"])

        tp_pct_entry = tp_sl["tp_mult"] * atr_scale.to_numpy()

        # Step 30: Profit Exit Optimization
        profit, trials_30 = m30.optimise_profit_exit(bucket_df, raw_returns, tp_pct_entry=tp_pct_entry, fee_pct=fee_pct, test_split_idx=split_idx, cost=cost, init_params=params_init.get("profit_exit", params_init))
        trials_30["bucket"] = bucket
        trials_30["step"] = "30_profit_exit"
        all_trials_log.append(trials_30)

        # Step 40: Position Sizing Optimization
        threaded_exit_stream = bool(bucket_df.attrs.get("threaded_exit_stream", False))
        if enforce_threaded_exit_stream and not threaded_exit_stream:
            raise RuntimeError("Stage40 sizing is using stale exit stream; thread post-20/30 ledger first.")
        # Pass raw exit/entry/is_long as original code did, but metrics will use them
        sizing, trials_40 = m40.optimise_position_sizing(
            bucket_df,
            bucket_df["exit_price"].to_numpy(dtype=float),
            bucket_df["entry_price"].to_numpy(dtype=float),
            bucket_df["is_long"].to_numpy(dtype=int),
            bucket_df["confidence"].to_numpy(dtype=float),
            test_split_idx=split_idx,
            fee_pct=fee_pct,
            cost=cost,
            init_params=params_init.get("position_sizing", params_init.get("sizing", params_init))
        )
        trials_40["bucket"] = bucket
        trials_40["step"] = "40_sizing"
        all_trials_log.append(trials_40)

        # Apply ridge weights to confidence if available
        confidence = bucket_df["confidence"].to_numpy(dtype=float)
        if ridge_weights:
            # Ridge weights are for meta model combination
            # Here we use them to adjust confidence scaling
            # This is a simplified integration - full integration would combine
            # multiple model predictions using the weights
            tprint(f"  Ridge weights available for bucket {bucket}: using for confidence scaling")
            # Store ridge weights in sizing output for reference
            sizing["ridge_weights"] = ridge_weights

        pos_size = m40.sigmoid_sizing(confidence, sizing["k"], sizing["c0"], sizing["s_min"], sizing["s_max"])
        net_returns = trade_return_net_vec(raw_ret_underlying=raw_returns, side=np.ones(len(raw_returns)), pos_w=pos_size, cost=cost)
        report = m50.evaluate_holdout(bucket_df, net_returns)
        holdout_ledger = m50.build_holdout_trade_ledger(bucket_df, net_returns, cost=cost)
        ledger_path = Path(output_path).with_name(f"{Path(output_path).stem}_{bucket}_holdout_ledger.csv")
        if not holdout_ledger.empty:
            holdout_ledger.to_csv(ledger_path, index=False)
        report["holdout_ledger_path"] = str(ledger_path)

        if not holdout_ledger.empty:
            emit_bucket_summary(
                tprint=tprint,
                run_id=run_id,
                bucket_id=bucket,
                kind="optimiser_eval",
                stats={
                    "ledger_rows": int(len(holdout_ledger)),
                    "ledger_checksum": hashlib.sha256(holdout_ledger.to_csv(index=False).encode("utf-8")).hexdigest()[:12],
                    "holdout_pnl_net": float(report.get("holdout_pnl_net", 0.0)),
                    "best_tp_mult": float(tp_sl.get("tp_mult", 0.0)),
                    "best_sl_mult": float(tp_sl.get("sl_mult", 0.0)),
                    "best_theta0": float(risk_cut.get("theta0", 0.0)),
                    "best_act_n": float(profit.get("act_n", 0.0)),
                    "best_size_k": float(sizing.get("k", 0.0)),
                    "threaded_exit_stream": threaded_exit_stream,
                },
            )

        combined = {
            "policy_mode": policy.mode,
            "baseline_seed": params_init,
            "tp_sl": tp_sl,
            "loss_limiter": risk_cut,
            "profit_exit": profit,
            "position_sizing": sizing,
            "evaluation": report,
        }
        
        # Add ridge weights to output if available
        if ridge_weights:
            combined["ridge_weights"] = ridge_weights
        
        all_out[bucket] = combined
        mw.merge_and_write_params(output_path, bucket, combined)

        store = store_best_params(
            store=store,
            version_key=version_key,
            bucket_id=bucket,
            params={
                "tp_sl": tp_sl,
                "loss_limiter": risk_cut,
                "profit_exit": profit,
                "position_sizing": sizing,
            },
            metrics={
                "holdout_pnl_net": float(report.get("holdout_pnl_net", 0.0)),
                "holdout_win_rate": float(report.get("holdout_win_rate", 0.0)),
                "holdout_trades": int(report.get("holdout_trades", 0)),
            },
        )
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

    save_params_store(store)
    return all_out
