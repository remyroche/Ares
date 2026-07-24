#!/usr/bin/env python3
"""Repair frozen backcast outcomes without recomputing model predictions."""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd

from scripts.backfill_complete_july_meta_predictions import (
    _hourly_close_proxy_outcomes,
)

CONTRACT = "hourly_close_policy_proxy_v2_activation_deadline"
OUTCOME_COLUMNS = (
    "exec_margin",
    "ev_after_1pct",
    "first_touch_bad_mae_1r",
    "full_path_bad_mae_1r",
    "timeout",
    "clean_exec",
    "dirty_positive",
)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _repair_month(
    directory: Path,
    *,
    feature_root: Path,
    policy_manifest: dict[str, Any],
    horizon_hours: int,
    policy_bar_minutes: int,
    round_trip_cost: float,
) -> dict[str, Any]:
    manifest_path = directory / "manifest.json"
    parquet_path = directory / "frozen_predictions.parquet"
    manifest = _load_json(manifest_path)
    if (
        manifest.get("outcome_contract_version") == CONTRACT
        and int(manifest.get("policy_bar_minutes", 0)) == int(policy_bar_minutes)
        and int(manifest.get("proxy_horizon_hours", 0)) == int(horizon_hours)
        and parquet_path.exists()
    ):
        return {"month": directory.name, "status": "reused", **manifest}
    frame = pd.read_parquet(parquet_path)
    frame["__ts__"] = pd.to_datetime(frame["__ts__"], utc=True)
    outcome_parts: list[pd.DataFrame] = []
    path_stats: dict[str, Any] = {}
    for side in ("long", "short"):
        side_rows = frame.loc[frame["side_name"].astype(str).eq(side)]
        if side_rows.empty:
            continue
        outcomes, stats = _hourly_close_proxy_outcomes(
            side_rows.reset_index(drop=True),
            feature_root=feature_root,
            policy_manifest=policy_manifest,
            horizon_hours=int(horizon_hours),
            policy_bar_minutes=int(policy_bar_minutes),
            round_trip_cost=float(round_trip_cost),
        )
        outcomes.index = side_rows.index
        outcome_parts.append(outcomes)
        path_stats[side] = stats
    if not outcome_parts:
        raise RuntimeError(f"No outcomes repaired for {directory}")
    outcomes = pd.concat(outcome_parts).sort_index()
    if outcomes[list(OUTCOME_COLUMNS)].notna().all(axis=1).mean() < 0.90:
        raise RuntimeError(f"Repaired outcome coverage below 90% for {directory}")
    for name in OUTCOME_COLUMNS:
        frame.loc[outcomes.index, name] = outcomes[name]
    temporary = parquet_path.with_suffix(".repairing.parquet")
    frame.to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, parquet_path)
    selected = frame["selected_for_monitor"].fillna(False).astype(bool)
    selected_daily = (
        frame.loc[selected]
        .assign(day=lambda value: value["__ts__"].dt.floor("D"))
        .groupby("day", observed=True)["ev_after_1pct"]
        .sum()
    )
    manifest.update(
        {
            "outcome_contract_version": CONTRACT,
            "policy_bar_minutes": int(policy_bar_minutes),
            "proxy_horizon_hours": int(horizon_hours),
            "round_trip_cost": float(round_trip_cost),
            "cost_counted_once": True,
            "path_stats": path_stats,
            "negative_selected_monitor_days": int(selected_daily.lt(0.0).sum()),
            "outcome_repair": "predictions and observable features unchanged",
        }
    )
    manifest_path.write_text(
        json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8"
    )
    return {"month": directory.name, "status": "repaired", **manifest}


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.backcast_root)
    monthly = root / "monthly"
    policy_manifest = _load_json(Path(args.policy_manifest))
    directories = sorted(
        path.parent
        for path in monthly.glob("*/manifest.json")
        if (path.parent / "frozen_predictions.parquet").exists()
    )
    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, int(args.max_workers))) as pool:
        futures = {
            pool.submit(
                _repair_month,
                directory,
                feature_root=Path(args.feature_root),
                policy_manifest=policy_manifest,
                horizon_hours=int(args.horizon_hours),
                policy_bar_minutes=int(args.policy_bar_minutes),
                round_trip_cost=float(args.round_trip_cost),
            ): directory
            for directory in directories
        }
        for future in as_completed(futures):
            result = future.result()
            rows.append(result)
            print(
                json.dumps(
                    {
                        "month": result["month"],
                        "status": result["status"],
                        "rows": result.get("rows"),
                    }
                ),
                flush=True,
            )
    rows.sort(key=lambda value: value["month"])
    summary = {
        "schema": "three_year_backcast_outcome_repair_v1",
        "outcome_contract_version": CONTRACT,
        "months": len(rows),
        "repaired": sum(value["status"] == "repaired" for value in rows),
        "reused": sum(value["status"] == "reused" for value in rows),
        "policy_bar_minutes": int(args.policy_bar_minutes),
        "horizon_hours": int(args.horizon_hours),
        "round_trip_cost": float(args.round_trip_cost),
        "cost_counted_once": True,
    }
    (root / "outcome_repair_manifest.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backcast-root", type=Path, required=True)
    parser.add_argument(
        "--feature-root",
        type=Path,
        default=Path("data_perp/features/20260711_070000"),
    )
    parser.add_argument("--policy-manifest", type=Path, required=True)
    parser.add_argument("--horizon-hours", type=int, default=24)
    parser.add_argument("--policy-bar-minutes", type=int, default=15)
    parser.add_argument("--round-trip-cost", type=float, default=0.01)
    parser.add_argument("--max-workers", type=int, default=2)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
