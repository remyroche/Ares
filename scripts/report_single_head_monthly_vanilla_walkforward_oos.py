#!/usr/bin/env python3
"""Report fixed-geometry vanilla OOS metrics for the single-head monthly walk-forward."""

from __future__ import annotations

import json
import os
import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import scripts.run_single_head_monthly_walkforward_oos as wf
from extreme_price_movements.inference.parity import calibrated_score_and_threshold
from extreme_price_movements.simple_position_sizer import load_calibration_curves
from extreme_price_movements import simple_policy_optimiser as spo


RANK_SLICES = [
    ("top_30", 0.70),
    ("top_20", 0.80),
    ("top_15", 0.85),
    ("top_10", 0.90),
    ("top_5", 0.95),
    ("top_1", 0.99),
]


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        out = float(value)
        return out if np.isfinite(out) else None
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if pd.isna(value) if not isinstance(value, (dict, list, tuple, np.ndarray)) else False:
        return None
    return value


def _load_policy_oos_frame(run_id: str, strategy_id: str) -> pd.DataFrame:
    core = wf._strategy_core(strategy_id)
    path = (
        wf.DATA_ROOT
        / "artifacts"
        / run_id
        / "policy_oos_predictions"
        / f"policy_oos_{core}_clf.parquet"
    )
    if not path.exists():
        path = path.with_name(f"policy_oos_{strategy_id}_clf.parquet")
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_parquet(path)
    df = spo._filter_policy_quote_rows(df, "perps")
    if "clf" not in df.columns and "oof_p_tp" in df.columns:
        df["clf"] = df["oof_p_tp"]
    elif "clf" not in df.columns and "oof_pred" in df.columns:
        df["clf"] = df["oof_pred"]
    if "clf" not in df.columns:
        raise RuntimeError(f"{path} has no clf/oof score column")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df["strategy_id"] = strategy_id
    if "side" not in df.columns:
        df["side"] = -1 if strategy_id.startswith("short") else 1
    return df.dropna(subset=["timestamp", "symbol", "clf"]).copy()


def _prepare_policy_frame(run_id: str, strategy_id: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    df = _load_policy_oos_frame(run_id, strategy_id)
    calibration_data = load_calibration_curves(str(wf.DATA_ROOT), run_id)
    df["raw_meta_prediction"] = pd.to_numeric(df["clf"], errors="coerce")
    df["calibrated_score"] = df["raw_meta_prediction"].map(
        lambda raw_score: (
            calibrated_score_and_threshold(
                raw_score=float(raw_score),
                strategy_id=strategy_id,
                calibration_data=calibration_data,
                default_threshold=1.0,
            )[0]
            if pd.notna(raw_score)
            else np.nan
        )
    )
    df = df.dropna(subset=["calibrated_score"]).sort_values("timestamp").reset_index(drop=True)
    df["rank_pct"] = df["calibrated_score"].rank(method="max", pct=True)
    slice_plan_path = wf.DATA_ROOT / "artifacts" / run_id / "slices" / "slice_plan.json"
    stage_view, stage_name = spo._load_policy_stage_view(slice_plan_path)
    if stage_name != "policy_optimiser":
        raise RuntimeError(f"Unexpected stage {stage_name} for {slice_plan_path}")
    opt_mask, validation_mask, split = spo._policy_optimisation_validation_masks(df, stage_view)
    if split.get("using_recent_policy_outer_split"):
        df["rank_pct"] = spo._rank_pct_against_reference(
            df["calibrated_score"],
            df.loc[opt_mask, "calibrated_score"],
        )
        df = df.dropna(subset=["rank_pct", "calibrated_score"]).copy()
        opt_mask, validation_mask, split = spo._policy_optimisation_validation_masks(
            df,
            stage_view,
        )
    return df.reset_index(drop=True), {
        "stage_view": stage_view,
        "split": split,
        "optimisation_rows": int(opt_mask.sum()),
        "validation_rows": int(validation_mask.sum()),
        "validation_mask": validation_mask.to_numpy(dtype=bool),
    }


def _score_slice(
    subset_df: pd.DataFrame,
    paths: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    rank_threshold: float,
) -> dict[str, Any]:
    mask = subset_df["rank_pct"].to_numpy(dtype=np.float32) >= float(rank_threshold)
    candidate_rows = int(mask.sum())
    if candidate_rows == 0:
        return {
            "candidate_rows": 0,
            "n_trades": 0,
            "net_pnl": 0.0,
            "mean_net_trade": 0.0,
            "hit_rate": 0.0,
            "max_drawdown": 0.0,
            "sortino": 0.0,
        }
    idx = np.flatnonzero(mask)
    rows = subset_df.iloc[idx].copy().reset_index(drop=True)
    sub_paths = spo._path_take(paths, idx)
    metrics = spo.simulate_and_score(
        rows,
        *sub_paths,
        cost_pct=spo.DEFAULT_POLICY_PER_SIDE_COST_PCT,
        size_power=1.0,
        market_mode="perps",
        max_concurrent_trades=spo.MAX_CONCURRENT_TRADES,
        max_concurrent_per_asset=spo.DEPLOYMENT_MAX_CONCURRENT_PER_ASSET,
    )
    adv = spo.calculate_advanced_metrics(
        rows,
        metrics.get("raw_gains", np.array([])),
        metrics.get("sizes", np.array([])),
        metrics.get("selected_mask"),
        metrics.get("gross_gains"),
        metrics.get("exit_reason"),
        metrics.get("exit_bars"),
    )
    return {
        "candidate_rows": candidate_rows,
        "n_trades": int(metrics.get("total_trades", 0) or 0),
        "net_pnl": float(metrics.get("net_pnl", 0.0) or 0.0),
        "mean_net_trade": float(metrics.get("mean_net_trade", 0.0) or 0.0),
        "hit_rate": float(metrics.get("win_rate", 0.0) or 0.0),
        "max_drawdown": float(adv.get("max_drawdown", adv.get("max_dd", 0.0)) or 0.0),
        "sortino": float(adv.get("m_sortino", 0.0) or 0.0),
        "avg_holding_bars": float(metrics.get("avg_holding_bars", 0.0) or 0.0),
        "trailing_exit_rate": float(
            metrics.get("trailing_exit_count", 0) / max(int(metrics.get("total_trades", 0) or 0), 1)
        ),
        "full_sl_exit_rate": float(
            metrics.get("full_sl_exit_count", 0) / max(int(metrics.get("total_trades", 0) or 0), 1)
        ),
        "timeout_exit_rate": float(metrics.get("timeout_exit_rate", 0.0) or 0.0),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment-id",
        default=os.environ.get("EPM_MONTHLY_WF_ID", wf.DEFAULT_EXPERIMENT_ID),
    )
    parser.add_argument(
        "--source-run-id",
        default=os.environ.get("EPM_SOURCE_RUN_ID", wf.DEFAULT_SOURCE_RUN_ID),
    )
    parser.add_argument("--strategy-id", default="")
    args = parser.parse_args()

    os.environ.setdefault("EPM_EXCHANGE", "krakenfutures")
    os.environ.setdefault("EPM_SIMPLE_POLICY_15M_DOWNLOAD", "1")
    os.environ.setdefault("MPLCONFIGDIR", str(wf.ROOT / ".mplconfig"))
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

    source_run_id = str(args.source_run_id).strip()
    experiment_id = str(args.experiment_id).strip()
    strategy_id = str(args.strategy_id or wf._select_june_best_strategy(source_run_id)["strategy_id"]).strip()
    ds = spo._make_policy_replay_store(str(wf.DATA_ROOT), "perps")
    rows: list[dict[str, Any]] = []
    details: dict[str, Any] = {
        "experiment_id": experiment_id,
        "strategy_id": strategy_id,
        "definition": {
            "policy": "vanilla_fixed_simulate_and_score_defaults",
            "rank_slices": RANK_SLICES,
            "geometry": {
                "sl_mult": 1.0,
                "trailing_activation_mult": 1.0,
                "trailing_power": 1.5,
                "trailing_squash_divisor": 2.0,
                "giveback_beta": 0.5,
                "capital_protect_mfe_mult": 0.0,
                "adverse_exit_enabled": False,
                "atr_power": 1.0,
                "atr_multiplier": 1.0,
                "hard_tp_abs_pct": 0.0,
                "exit_pressure_enabled": False,
                "size_power": 1.0,
            },
            "cost_pct_per_side": float(spo.DEFAULT_POLICY_PER_SIDE_COST_PCT),
            "portfolio_replay": False,
            "optuna_policy_params": False,
            "stage_a_threshold_grid": False,
        },
        "folds": {},
    }
    for fold in wf._folds(experiment_id):
        df_all, split_info = _prepare_policy_frame(fold.run_id, strategy_id)
        all_paths = spo._fetch_policy_paths(df_all, ds)
        df_all, all_paths = spo._apply_delayed_entry_execution_model(
            df_all,
            all_paths,
            data_root=str(wf.DATA_ROOT),
            market_mode="perps",
        )
        validation_mask = split_info.pop("validation_mask")
        validation_idx = np.flatnonzero(validation_mask)
        validation_df = df_all.iloc[validation_idx].copy().reset_index(drop=True)
        validation_paths = spo._path_take(all_paths, validation_idx)
        fold_detail = {
            "run_id": fold.run_id,
            "train_end": fold.train_end.isoformat(),
            "policy_start": fold.policy_start.isoformat(),
            "policy_split": fold.policy_split.isoformat(),
            "policy_end": fold.policy_end.isoformat(),
            **split_info,
            "validation_timestamp_min": (
                validation_df["timestamp"].min().isoformat() if not validation_df.empty else None
            ),
            "validation_timestamp_max": (
                validation_df["timestamp"].max().isoformat() if not validation_df.empty else None
            ),
            "metrics_by_rank_slice": {},
        }
        for label, threshold in RANK_SLICES:
            metrics = _score_slice(validation_df, validation_paths, threshold)
            fold_detail["metrics_by_rank_slice"][label] = metrics
            rows.append(
                {
                    "fold": fold.name,
                    "run_id": fold.run_id,
                    "train_end": fold.train_end.isoformat(),
                    "validation_start": fold.policy_split.isoformat(),
                    "validation_end": fold.policy_end.isoformat(),
                    "rank_slice": label,
                    "rank_threshold": threshold,
                    **metrics,
                }
            )
        details["folds"][fold.name] = fold_detail

    out_dir = wf.DATA_ROOT / "reports" / experiment_id
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_dir / "vanilla_walkforward_oos_summary.csv", index=False)
    (out_dir / "vanilla_walkforward_oos_summary.json").write_text(
        json.dumps(_json_safe(details), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(out_dir / "vanilla_walkforward_oos_summary.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
