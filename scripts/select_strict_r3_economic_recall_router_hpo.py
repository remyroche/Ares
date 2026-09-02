#!/usr/bin/env python3
"""Select strict-R3 router HPO only from equal-timestamp validation metrics.

Candidate rows and selected trades are deliberately never pooled to determine
the winner.  Each exact decision timestamp has one unit of evaluation mass;
per-month values are retained solely as a stability tie-break after the
predeclared timestamp-average economic-recall objective.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROUTES = (0.30, 0.40, 0.50)
ROUTE_WEIGHTS = {0.30: 0.25, 0.40: 0.35, 0.50: 0.40}
PRIMARY_SCORE = "router_primary_only_rank"


def _candidate(value: str) -> tuple[str, Path]:
    try:
        name, path = value.split("=", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("candidate must be NAME=/absolute/artifact/path") from exc
    if not name or not path:
        raise argparse.ArgumentTypeError("candidate name and path are required")
    return name, Path(path)


def _read(name: str, root: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    contract_path = root / "run_contract.json"
    timestamp_path = root / "router_timestamp_metrics.parquet"
    if not contract_path.exists() or not timestamp_path.exists():
        raise FileNotFoundError(f"{name}: missing strict-router contract or timestamp receipt")
    contract = json.loads(contract_path.read_text())
    if "ranker" not in contract:
        raise AssertionError(f"{name}: missing ranker contract")
    frame = pd.read_parquet(timestamp_path)
    frame = frame.loc[
        frame["score"].eq(PRIMARY_SCORE)
        & frame["route_fraction"].round(2).isin(ROUTES)
    ].copy()
    if frame.empty:
        raise AssertionError(f"{name}: no primary-only timestamp rows")
    expected = set(ROUTES)
    actual = set(frame["route_fraction"].round(2).unique())
    if actual != expected:
        raise AssertionError(f"{name}: expected routes {sorted(expected)}, found {sorted(actual)}")
    frame["candidate"] = name
    return frame, contract


def _mean(values: pd.Series) -> float:
    values = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return float(values.mean()) if len(values) else float("nan")


def _summary(name: str, timestamp: pd.DataFrame, contract: dict[str, object]) -> dict[str, object]:
    item: dict[str, object] = {
        "candidate": name,
        "primary_target": contract.get("primary_target"),
        "ranker": json.dumps(contract.get("ranker"), sort_keys=True),
        "timestamps": int(timestamp["__decision_ts__"].nunique()),
    }
    objective = 0.0
    ev_values: list[float] = []
    min_month_er: list[float] = []
    min_month_ev: list[float] = []
    for route in ROUTES:
        work = timestamp.loc[timestamp["route_fraction"].round(2).eq(route)].copy()
        er = _mean(work["timestamp_er50"])
        recall = _mean(work["timestamp_recall_50bps"])
        ev = _mean(work["timestamp_selected_net_ev_bps"])
        item[f"timestamp_er50_{int(route * 100)}"] = er
        item[f"timestamp_recall50_{int(route * 100)}"] = recall
        item[f"timestamp_ev_{int(route * 100)}_bps"] = ev
        objective += ROUTE_WEIGHTS[route] * er
        ev_values.append(ev)
        by_month = work.groupby("held_month", sort=False)
        month_er = by_month["timestamp_er50"].mean()
        month_ev = by_month["timestamp_selected_net_ev_bps"].mean()
        min_month_er.append(float(month_er.min()))
        min_month_ev.append(float(month_ev.min()))
    item["timestamp_primary_selection_score"] = objective
    item["timestamp_mean_ev_30_40_50_bps"] = float(np.nanmean(ev_values))
    item["worst_month_timestamp_er50"] = float(np.nanmin(min_month_er))
    item["worst_month_timestamp_ev_bps"] = float(np.nanmin(min_month_ev))
    return item


def run(args: argparse.Namespace) -> None:
    if args.out.exists():
        raise FileExistsError(args.out)
    parts: list[pd.DataFrame] = []
    summary: list[dict[str, object]] = []
    contracts: dict[str, object] = {}
    for name, root in args.candidate:
        timestamp, contract = _read(name, root)
        parts.append(timestamp)
        summary.append(_summary(name, timestamp, contract))
        contracts[name] = {"root": str(root), "ranker": contract["ranker"], "primary_target": contract["primary_target"]}
    result = pd.DataFrame(summary).sort_values(
        ["timestamp_primary_selection_score", "worst_month_timestamp_er50", "timestamp_mean_ev_30_40_50_bps", "worst_month_timestamp_ev_bps"],
        ascending=False, kind="stable",
    ).reset_index(drop=True)
    result.insert(0, "selection_rank", np.arange(1, len(result) + 1, dtype=np.int32))
    args.out.mkdir(parents=True)
    result.to_parquet(args.out / "timestamp_hpo_summary.parquet", index=False, compression="zstd")
    pd.concat(parts, ignore_index=True).to_parquet(args.out / "timestamp_hpo_inputs.parquet", index=False, compression="zstd")
    (args.out / "selection_manifest.json").write_text(json.dumps({
        "schema": "strict_r3_economic_recall_router_timestamp_hpo_v1",
        "selection_unit": "one exact decision timestamp; candidate/trade pooled quantities prohibited from selection",
        "primary_score": "0.25*mean_timestamp_ER50@30 + 0.35*mean_timestamp_ER50@40 + 0.40*mean_timestamp_ER50@50",
        "tie_breaks": ["worst month timestamp-average ER50", "mean timestamp EV across 30/40/50", "worst month timestamp-average EV"],
        "score_contract": PRIMARY_SCORE,
        "candidates": contracts,
        "winner": result.iloc[0]["candidate"],
        "summary": result.to_dict(orient="records"),
    }, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=_candidate, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
