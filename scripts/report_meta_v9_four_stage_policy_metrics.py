#!/usr/bin/env python3
"""Report the canonical V9 -> MLP -> admission -> auction policy chain."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from extreme_price_movements.inference.threshold_basis_policy import (
    apply_threshold_basis_policy_to_decisions,
)
from extreme_price_movements.portfolio_policy_replay import (
    fit_hierarchical_ev_curves,
    load_portfolio_policy_params,
    normalise_candidate_table,
    replay_candidates,
)


POLICY_ID = "ev_target_side_archetype_global_top10_before_mlp_28d_flat_v1"
POLICY_NAME = "s52_v9_tail95_mlp_hierev_evtarget28d_prempl_top10_v1"
PREDECESSOR_ID = (
    "meta_residual_extreme_local_champion_overlay_ooftrain_tieaware_downonly_"
    "20260712_v9::forced_local_tail_0.950"
)
MLP_ID = "meta_residual_v9_tail95_market_state_mlp_hier_ev_v1"
PORTFOLIO_ID = "global_auction_v1"
KEYS = ["__ts__", "__symbol__", "side_name", "archetype_policy_key"]


def _top_n_mask(values: pd.Series, count: int) -> np.ndarray:
    score = pd.to_numeric(values, errors="coerce").to_numpy(dtype=np.float64)
    mask = np.zeros(len(score), dtype=bool)
    finite = np.flatnonzero(np.isfinite(score))
    if count <= 0 or not len(finite):
        return mask
    count = min(int(count), len(finite))
    chosen = finite[np.argpartition(score[finite], -count)[-count:]]
    mask[chosen] = True
    return mask


def _fixed_monthly_activity_masks(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    month = frame["__ts__"].dt.strftime("%Y-%m")
    parent = np.zeros(len(frame), dtype=bool)
    mlp = np.zeros(len(frame), dtype=bool)
    for month_id in sorted(month.unique()):
        idx = np.flatnonzero(month.eq(month_id).to_numpy())
        budget = int(
            pd.to_numeric(frame.iloc[idx]["policy_parent_rank"], errors="coerce")
            .ge(0.90)
            .sum()
        )
        parent[idx] = _top_n_mask(frame.iloc[idx]["policy_parent_rank"], budget)
        mlp[idx] = _top_n_mask(frame.iloc[idx]["expected_ev_rank_score"], budget)
    return parent, mlp


def _reference_rows(history_path: Path, output: Path) -> pd.DataFrame:
    rows = pd.read_parquet(
        history_path,
        columns=[
            "__ts__",
            "__symbol__",
            "side_name",
            "archetype_policy_key",
            "policy_parent_rank",
            "rank_mlp_direct",
            "ev_after_1pct",
            "__fold__",
        ],
    ).rename(columns={"__ts__": "timestamp", "__symbol__": "symbol"})
    rows["timestamp"] = pd.to_datetime(rows["timestamp"], utc=True, errors="coerce")
    rows["policy_archetype"] = rows["archetype_policy_key"].astype(str)
    rows = rows.dropna(
        subset=[
            "timestamp",
            "side_name",
            "policy_archetype",
            "policy_parent_rank",
            "rank_mlp_direct",
            "ev_after_1pct",
        ]
    ).sort_values("timestamp", kind="stable")
    rows = rows.loc[
        :,
        [
            "timestamp",
            "symbol",
            "side_name",
            "policy_archetype",
            "policy_parent_rank",
            "rank_mlp_direct",
            "ev_after_1pct",
            "__fold__",
        ],
    ]
    rows.to_parquet(output, index=False, compression="zstd")
    return rows


def _threshold_policy(reference_path: Path, rows: pd.DataFrame) -> dict[str, Any]:
    return {
        "schema_version": "threshold_basis_policy_v2",
        "policy_id": POLICY_ID,
        "policy_name": POLICY_NAME,
        "enabled": True,
        "live_compatible_selection": True,
        "family": "ev_target_side_archetype_multiplier_before_mlp",
        "window_days": 28,
        "smoothing": "flat",
        "top_fraction": 0.10,
        "min_reference_rows": 40,
        "local_support_target": 160.0,
        "multiplier_min": 0.50,
        "multiplier_max": 1.50,
        "calibration_reference_score_col": "policy_parent_rank",
        "apply_reference_score_col": "rank_mlp_direct",
        "live_score_col": "expected_ev_rank_score",
        "return_col": "ev_after_1pct",
        "selection_group": "timestamp",
        "reference_candidates_path": reference_path.name,
        "reference_columns": list(rows.columns),
        "reference_rows": int(len(rows)),
        "cost_contract": (
            "ev_after_1pct contains the sole 1% round-trip cost; no extra fee "
            "or spread is subtracted in this layer-comparison replay."
        ),
    }


def _apply_admission(frame: pd.DataFrame, policy: dict[str, Any]) -> pd.DataFrame:
    decisions: list[dict[str, Any]] = []
    columns = frame.loc[
        :,
        [
            "__ts__",
            "__symbol__",
            "side_name",
            "archetype_policy_key",
            "expected_ev_rank_score",
        ],
    ]
    for timestamp, symbol, side, archetype, score in columns.itertuples(
        index=False, name=None
    ):
        decisions.append(
            {
                "timestamp": timestamp,
                "symbol": symbol,
                "side_name": side,
                "policy_archetype": archetype,
                "strategy_id": f"{side}_s52_meta_threshold_handoff",
                "expected_ev_rank_score": float(score),
                "policy_rank_pct": float(score),
            }
        )
    apply_threshold_basis_policy_to_decisions(decisions, policy=policy)
    result = pd.DataFrame(
        {
            "threshold_basis_selected": [
                bool(row.get("threshold_basis_selected", False)) for row in decisions
            ],
            "threshold_basis_rank_score": [
                float(row.get("threshold_basis_rank_score", 0.0)) for row in decisions
            ],
            "threshold_basis_multiplier": [
                float(row.get("threshold_basis_ev_target_multiplier", 1.0))
                for row in decisions
            ],
            "threshold_basis_local_support": [
                int(row.get("threshold_basis_ev_target_local_support", 0))
                for row in decisions
            ],
            "threshold_basis_global_fallback": [
                bool(row.get("threshold_basis_ev_target_global_fallback", False))
                for row in decisions
            ],
        }
    )
    return result


def _read_label_paths(labels_dir: Path, frame: pd.DataFrame) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    months = sorted(frame["__ts__"].dt.strftime("%Y-%m").unique())
    columns = [
        "__ts__",
        "__symbol__",
        "__first_touch_bar__",
        "__first_touch_stop__",
        "__first_touch_timeout__",
    ]
    for side in ("long", "short"):
        for month in months:
            year, month_no = month.split("-")
            path = labels_dir / f"train_global_{side}_5_{year}_{month_no}.parquet"
            if not path.exists():
                continue
            part = pd.read_parquet(path, columns=columns)
            part["__ts__"] = pd.to_datetime(part["__ts__"], utc=True, errors="coerce")
            part["side_name"] = side
            parts.append(part)
    if not parts:
        raise FileNotFoundError(f"No matching monthly labels under {labels_dir}")
    return pd.concat(parts, ignore_index=True, copy=False).drop_duplicates(
        ["__ts__", "__symbol__", "side_name"], keep="last"
    )


def _portfolio_candidates(
    selected: pd.DataFrame, labels_dir: Path
) -> tuple[pd.DataFrame, dict[str, Any]]:
    labels = _read_label_paths(labels_dir, selected)
    rows = selected.merge(
        labels,
        on=["__ts__", "__symbol__", "side_name"],
        how="left",
        validate="one_to_one",
    )
    valid = pd.to_numeric(rows["__first_touch_bar__"], errors="coerce").notna()
    coverage = {
        "admitted_rows": int(len(rows)),
        "replay_rows": int(valid.sum()),
        "replay_coverage": float(valid.mean()) if len(valid) else 0.0,
    }
    rows = rows.loc[valid].copy()
    holding = pd.to_numeric(rows["__first_touch_bar__"], errors="coerce").clip(lower=1)
    net = pd.to_numeric(rows["ev_after_1pct"], errors="coerce")
    side_sign = np.where(rows["side_name"].eq("long"), 1.0, -1.0)
    rank = pd.to_numeric(rows["threshold_basis_rank_score"], errors="coerce")
    stop = pd.to_numeric(rows["__first_touch_stop__"], errors="coerce").fillna(0.0)
    timeout = pd.to_numeric(rows["__first_touch_timeout__"], errors="coerce").fillna(0.0)
    reason = np.select([stop.gt(0.5), timeout.gt(0.5)], ["stop", "timeout"], default="trailing")
    policy_archetype = rows["side_name"] + "__" + rows["archetype_policy_key"].astype(str)
    candidates = pd.DataFrame(
        {
            "timestamp": rows["__ts__"],
            "symbol": rows["__symbol__"].astype(str),
            "side": side_sign,
            "side_name": rows["side_name"].astype(str),
            "strategy_id": rows["side_name"] + "_s52_meta_threshold_handoff",
            "policy_archetype": policy_archetype,
            "local_side_archetype": policy_archetype,
            "normalized_rank_score": rank,
            "strategy_rank_pct": rank,
            "base_strategy_threshold": 0.90,
            "calibrated_score": rank,
            "entry_price": 1.0,
            "exit_timestamp": rows["__ts__"] + pd.to_timedelta(holding * 15.0, unit="m"),
            "exit_price": 1.0 + side_sign * (net + 0.01),
            "net_return": net,
            "gross_return": net + 0.01,
            "holding_bars": holding,
            "simple_policy_exit_reason": reason,
            "fees_bps": 100.0,
            "slippage_bps": 0.0,
            "expected_friction_bps": 100.0,
            "price_gap_bps": 0.0,
            "liquidity_capacity_weight": 1.0,
            "ev_after_1pct": net,
            "clean_exec": rows["clean_exec"].to_numpy(),
            "dirty_positive": rows["dirty_positive"].to_numpy(),
            "full_path_bad_mae_1r": rows["full_path_bad_mae_1r"].to_numpy(),
            "timeout": rows["timeout"].to_numpy(),
            "archetype_policy_key": rows["archetype_policy_key"].to_numpy(),
        }
    )
    return normalise_candidate_table(candidates), coverage


def _ev_curve(reference_path: Path) -> dict[str, Any]:
    ref = pd.read_parquet(reference_path)
    ref["timestamp"] = pd.to_datetime(ref["timestamp"], utc=True, errors="coerce")
    ref["normalized_rank_score"] = pd.to_numeric(ref["rank_pct"], errors="coerce")
    ref["net_return"] = pd.to_numeric(ref["ret_net_notional"], errors="coerce")
    ref["base_strategy_threshold"] = 0.90
    ref["entry_price"] = 1.0
    ref["exit_price"] = 1.0
    ref["exit_timestamp"] = ref["timestamp"] + pd.Timedelta(minutes=15)
    ref["gross_return"] = ref["net_return"] + 0.01
    ref["holding_bars"] = 1.0
    ref["simple_policy_exit_reason"] = "historical_reference"
    return fit_hierarchical_ev_curves(ref)


def _metric_rows(frame: pd.DataFrame, stage: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    groups: list[tuple[str, pd.DataFrame]] = [("ALL", frame)]
    groups.extend(
        (str(month), group)
        for month, group in frame.groupby(frame["__ts__"].dt.strftime("%Y-%m"), sort=True)
    )
    for month, group in groups:
        ev = pd.to_numeric(group["ev_after_1pct"], errors="coerce")
        days = int(group["__ts__"].dt.floor("D").nunique())
        rows.append(
            {
                "stage": stage,
                "month": month,
                "selected_rows": int(len(group)),
                "days": days,
                "trades_per_day": float(len(group) / max(days, 1)),
                "mean_net_ev_after_1pct": float(ev.mean()) if len(group) else np.nan,
                "sum_net_ev_after_1pct": float(ev.sum()) if len(group) else 0.0,
                "positive_ev_rate": float(ev.gt(0).mean()) if len(group) else np.nan,
                "clean_exec_rate": float(pd.to_numeric(group["clean_exec"], errors="coerce").mean()) if len(group) else np.nan,
                "dirty_positive_rate": float(pd.to_numeric(group["dirty_positive"], errors="coerce").mean()) if len(group) else np.nan,
                "full_path_bad_mae_rate": float(pd.to_numeric(group["full_path_bad_mae_1r"], errors="coerce").mean()) if len(group) else np.nan,
                "timeout_rate": float(pd.to_numeric(group["timeout"], errors="coerce").mean()) if len(group) else np.nan,
                "bankroll_pnl": float(pd.to_numeric(group.get("bankroll_pnl"), errors="coerce").sum()) if "bankroll_pnl" in group else np.nan,
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mlp-dir", type=Path, required=True)
    parser.add_argument("--labels-dir", type=Path, required=True)
    parser.add_argument("--portfolio-config", type=Path, required=True)
    parser.add_argument("--portfolio-ev-reference", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_parquet(args.mlp_dir / "oos_predictions.parquet")
    frame["__ts__"] = pd.to_datetime(frame["__ts__"], utc=True, errors="coerce")
    frame = frame.sort_values(KEYS, kind="stable").reset_index(drop=True)

    parent_mask, mlp_mask = _fixed_monthly_activity_masks(frame)
    reference_path = args.output_dir / "threshold_basis_reference_candidates.parquet"
    reference = _reference_rows(args.mlp_dir / "walkforward_rank_history.parquet", reference_path)
    policy = _threshold_policy(reference_path, reference)
    policy_path = args.output_dir / "threshold_basis_policy.json"
    policy_path.write_text(json.dumps(policy, indent=2, sort_keys=True) + "\n")
    policy["_artifact_path"] = str(policy_path)
    admission = _apply_admission(frame, policy)
    frame = pd.concat([frame, admission], axis=1)
    frame.to_parquet(args.output_dir / "four_stage_oos_rows.parquet", index=False, compression="zstd")

    admitted = frame.loc[frame["threshold_basis_selected"]].copy()
    candidates, coverage = _portfolio_candidates(admitted, args.labels_dir)
    params = load_portfolio_policy_params(args.portfolio_config)
    decisions, equity, portfolio_summary = replay_candidates(
        candidates,
        params,
        mode="global_auction",
        ev_curve=_ev_curve(args.portfolio_ev_reference),
        market_mode="perps",
    )
    accepted = decisions.loc[decisions["accepted"]].copy()
    accepted_source = candidates.iloc[
        accepted["candidate_index"].to_numpy(dtype=np.int64)
    ].reset_index(drop=True)
    accepted_source["bankroll_pnl"] = (
        accepted["position_size"].to_numpy(dtype=np.float64)
        * accepted["position_net_return"].to_numpy(dtype=np.float64)
    )
    accepted_source = accepted_source.rename(
        columns={"timestamp": "__ts__", "symbol": "__symbol__"}
    )

    metrics: list[dict[str, Any]] = []
    metrics.extend(_metric_rows(frame.loc[parent_mask].copy(), PREDECESSOR_ID))
    metrics.extend(_metric_rows(frame.loc[mlp_mask].copy(), MLP_ID))
    metrics.extend(_metric_rows(admitted, POLICY_ID))
    metrics.extend(_metric_rows(accepted_source, PORTFOLIO_ID))
    metric_frame = pd.DataFrame(metrics)
    metric_frame.to_csv(args.output_dir / "four_stage_metrics_global_month.csv", index=False)
    decisions.to_parquet(args.output_dir / "portfolio_decisions.parquet", index=False, compression="zstd")
    equity.to_parquet(args.output_dir / "portfolio_equity.parquet", index=False, compression="zstd")
    manifest = {
        "schema": "meta_v9_four_stage_policy_metrics_v1",
        "policy_name": POLICY_NAME,
        "stages": [PREDECESSOR_ID, MLP_ID, POLICY_ID, PORTFOLIO_ID],
        "mlp_artifact": str(args.mlp_dir),
        "selection_contract": {
            "stage_1_and_2": "same parent top10-equivalent activity within each OOS month",
            "stage_3": "causal 28-day side x archetype reachable-EV admission",
            "stage_4": "saved global_auction_v1 concurrency and position sizing",
        },
        "cost_contract": "ev_after_1pct includes exactly one 1% round-trip cost",
        "portfolio_path_coverage": coverage,
        "portfolio_summary": portfolio_summary,
        "evaluation_start": frame["__ts__"].min().isoformat(),
        "evaluation_end": frame["__ts__"].max().isoformat(),
        "evidence": "monthly expanding-window model OOS; policy-validation OOS",
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str) + "\n"
    )
    print(metric_frame.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
