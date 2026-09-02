#!/usr/bin/env python3
"""Post-score A--E group-removal gate for the strict-OOF recall router.

It consumes a completed immutable router score artifact.  All scores were
already emitted before any policy outcome join, so recombining their
train-reference ranks does not refit, select, or leak into the router.  Policy
labels are joined only to evaluate the predeclared gate:

    0.25 * ER50@30% + 0.35 * ER50@40% + 0.40 * ER50@50%.

Only if complete A--E beats primary-only on the aggregate score do we evaluate
each leave-one-group-out combination.  A group is retained when removing it
reduces the selection score; it is dropped when removal improves or preserves
the score.  Results remain development evidence, not untouched promotion.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


SCHEMA = "strict_r3_economic_recall_router_group_ablation_v1"
ROUTE_WEIGHTS = {0.30: 0.25, 0.40: 0.35, 0.50: 0.40}


def _write_json_exclusive(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(fd, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)


def _route(frame: pd.DataFrame, score: str, fraction: float) -> np.ndarray:
    work = frame.loc[:, ["candidate_id", "__decision_ts__", score]].copy()
    work["__row__"] = np.arange(len(work), dtype=np.int64)
    work["__score__"] = pd.to_numeric(work[score], errors="coerce").fillna(-np.inf)
    work = work.sort_values(["__decision_ts__", "__score__", "candidate_id"], ascending=[True, False, True], kind="stable")
    work["__ordinal__"] = work.groupby("__decision_ts__", sort=False).cumcount() + 1
    count = work.groupby("__decision_ts__", sort=False)["candidate_id"].transform("size").to_numpy(float)
    selected = (work["__ordinal__"].to_numpy() <= np.ceil(fraction * count)) & np.isfinite(work["__score__"].to_numpy(float))
    return pd.Series(selected, index=work["__row__"].to_numpy()).reindex(np.arange(len(frame))).to_numpy(bool)


def _timestamp_metrics(frame: pd.DataFrame, score: str, month: str, arm: str) -> pd.DataFrame:
    """Timestamp-equal evaluation; route before excluding unresolved paths."""
    base = frame.loc[:, ["candidate_id", "__decision_ts__", score, "policy_path_valid", "policy_net_bps"]].copy()
    base["__net__"] = pd.to_numeric(base["policy_net_bps"], errors="coerce")
    base["__valid__"] = base["policy_path_valid"].fillna(False).astype(bool) & np.isfinite(base["__net__"])
    output: list[pd.DataFrame] = []
    for fraction in ROUTE_WEIGHTS:
        work = base.copy()
        work["__selected__"] = _route(work, score, fraction)
        work["__selected_valid__"] = work["__selected__"] & work["__valid__"]
        work["__selected_net__"] = np.where(work["__selected_valid__"], work["__net__"], 0.0)
        work["__excess__"] = np.where(work["__valid__"], np.maximum(work["__net__"].to_numpy(float) - 50.0, 0.0), 0.0)
        work["__selected_excess__"] = np.where(work["__selected_valid__"], work["__excess__"], 0.0)
        work["__winner50__"] = work["__valid__"] & work["__net__"].gt(50.0)
        work["__winner100__"] = work["__valid__"] & work["__net__"].gt(100.0)
        work["__selected_winner50__"] = work["__selected__"] & work["__winner50__"]
        work["__selected_winner100__"] = work["__selected__"] & work["__winner100__"]
        grouped = work.groupby("__decision_ts__", sort=False).agg(
            candidate_rows=("candidate_id", "size"), selected_candidate_rows=("__selected__", "sum"),
            valid_rows=("__valid__", "sum"), selected_valid_rows=("__selected_valid__", "sum"),
            selected_net_sum_bps=("__selected_net__", "sum"), excess_sum=("__excess__", "sum"),
            selected_excess_sum=("__selected_excess__", "sum"), winners50=("__winner50__", "sum"),
            selected_winners50=("__selected_winner50__", "sum"), winners100=("__winner100__", "sum"),
            selected_winners100=("__selected_winner100__", "sum"),
        ).reset_index()
        n = grouped["selected_valid_rows"].to_numpy(float)
        grouped["timestamp_net_ev_bps"] = np.divide(grouped["selected_net_sum_bps"], n, out=np.full(len(grouped), np.nan), where=n > 0)
        denom = grouped["excess_sum"].to_numpy(float)
        grouped["timestamp_er50"] = np.divide(grouped["selected_excess_sum"], denom, out=np.full(len(grouped), np.nan), where=denom > 0)
        w50 = grouped["winners50"].to_numpy(float); w100 = grouped["winners100"].to_numpy(float)
        grouped["timestamp_recall50"] = np.divide(grouped["selected_winners50"], w50, out=np.full(len(grouped), np.nan), where=w50 > 0)
        grouped["timestamp_recall100"] = np.divide(grouped["selected_winners100"], w100, out=np.full(len(grouped), np.nan), where=w100 > 0)
        grouped["month"] = month; grouped["arm"] = arm; grouped["route_fraction"] = fraction
        output.append(grouped)
    return pd.concat(output, ignore_index=True)


def _metrics(frame: pd.DataFrame, score: str, month: str, arm: str) -> tuple[list[dict[str, object]], pd.DataFrame]:
    timestamp = _timestamp_metrics(frame, score, month, arm)
    rows: list[dict[str, object]] = []
    for fraction, work in timestamp.groupby("route_fraction", sort=False):
        selected = float(work["selected_valid_rows"].sum())
        net_sum = float(work["selected_net_sum_bps"].sum())
        rows.append({
            "month": month, "arm": arm, "route_fraction": float(fraction), "timestamps": int(len(work)),
            "er50_timestamps": int(work["timestamp_er50"].notna().sum()),
            "recall50_timestamps": int(work["timestamp_recall50"].notna().sum()),
            "recall100_timestamps": int(work["timestamp_recall100"].notna().sum()),
            "ev_timestamps": int(work["timestamp_net_ev_bps"].notna().sum()),
            "selected_rows": int(selected), "net_sum_bps": net_sum,
            "net_ev_bps_per_trade": float(work["timestamp_net_ev_bps"].mean()),
            "trade_weighted_net_ev_bps_per_trade": net_sum / selected if selected else np.nan,
            "er50": float(work["timestamp_er50"].mean()), "recall50": float(work["timestamp_recall50"].mean()),
            "recall100": float(work["timestamp_recall100"].mean()),
        })
    return rows, timestamp


def _selection(metrics: pd.DataFrame) -> pd.DataFrame:
    grouped_rows: list[dict[str, object]] = []
    for (arm, fraction), work in metrics.groupby(["arm", "route_fraction"], sort=False):
        ew = work["er50_timestamps"].to_numpy(float); vw = work["ev_timestamps"].to_numpy(float)
        grouped_rows.append({
            "arm": arm, "route_fraction": float(fraction), "folds": int(work["month"].nunique()),
            "mean_er50": float(np.average(work["er50"], weights=ew)) if ew.sum() else np.nan, "min_er50": float(work["er50"].min()),
            "mean_ev": float(np.average(work["net_ev_bps_per_trade"], weights=vw)) if vw.sum() else np.nan, "min_ev": float(work["net_ev_bps_per_trade"].min()),
            "total_net_bps": float(work["net_sum_bps"].sum()), "total_selected": int(work["selected_rows"].sum()),
        })
    grouped = pd.DataFrame(grouped_rows)
    weighted = grouped.assign(weight=grouped["route_fraction"].map(ROUTE_WEIGHTS)).groupby("arm", as_index=False).apply(
        lambda value: pd.Series({
            "selection_score": float((value["mean_er50"] * value["weight"]).sum()),
            "worst_er50": float(value["min_er50"].min()),
            "mean_ev_across_routes": float((value["mean_ev"] * value["weight"]).sum()),
            "total_net_bps_across_routes": float(value["total_net_bps"].sum()),
            "total_selected_across_routes": int(value["total_selected"].sum()),
        }),
        include_groups=False,
    ).reset_index()
    return grouped, weighted


def _combined_score(frame: pd.DataFrame, groups: tuple[str, ...]) -> np.ndarray:
    primary = pd.to_numeric(frame["router_primary_rank"], errors="coerce").to_numpy(float)
    if not groups:
        return primary.astype(np.float32)
    output = 0.5 * primary
    weight = 0.5 / len(groups)
    for group in groups:
        output += weight * pd.to_numeric(frame[f"router_group_{group}_rank"], errors="coerce").to_numpy(float)
    return output.astype(np.float32)


def run(source: Path, out: Path) -> None:
    contract_path = source / "run_contract.json"
    if not contract_path.exists():
        raise FileNotFoundError(contract_path)
    contract = json.loads(contract_path.read_text())
    if contract.get("schema") != "strict_r3_economic_recall_router_ae_v1":
        raise AssertionError("source is not a strict-R3 A-E router artifact")
    groups = tuple(str(value) for value in contract.get("aux_groups", ()))
    if not groups or len(set(groups)) != len(groups):
        raise AssertionError("source auxiliary contract is malformed")
    if out.exists():
        raise FileExistsError(out)
    score_paths = sorted((source / "target_free_scores").glob("month=*.parquet"))
    if not score_paths:
        raise AssertionError("source has no target-free score receipts")
    policy_path = Path(contract["policy_path"])
    policy = pd.read_parquet(policy_path, columns=["candidate_id", "policy_path_valid", "policy_net_bps"])
    if policy["candidate_id"].duplicated().any():
        raise AssertionError("policy ledger has duplicate candidate identity")
    out.mkdir(parents=True)
    _write_json_exclusive(out / "run_contract.json", {
        "schema": SCHEMA, "source": str(source), "source_contract": contract_path.read_text(),
        "scope": "post-score development gate only; no model/refit/live mutation",
    })
    rows: list[dict[str, object]] = []
    # Keep the per-timestamp receipts alongside the monthly summaries.  This
    # makes the post-score gate auditable as an equal-timestamp evaluation,
    # rather than an accidental candidate- or trade-weighted aggregate.
    timestamp_rows: list[pd.DataFrame] = []
    for path in score_paths:
        month = path.stem.split("=")[-1]
        score = pd.read_parquet(path)
        required = {"candidate_id", "__decision_ts__", "router_primary_rank", *(f"router_group_{group}_rank" for group in groups)}
        missing = required - set(score.columns)
        if missing:
            raise AssertionError(f"{path}: missing frozen group score columns {sorted(missing)}")
        # Strict target-free schema guard again: values may be named after a
        # target, but only router outputs are permitted in this input receipt.
        invalid = [column for column in score if column not in {"candidate_id", "__decision_ts__", "side_name"} and not (column.startswith("router_") or column.startswith("router__"))]
        if invalid:
            raise AssertionError(f"{path}: non-router column in target-free receipt: {invalid}")
        score["primary_only"] = _combined_score(score, ())
        score["full_ae"] = _combined_score(score, groups)
        for removed in groups:
            kept = tuple(group for group in groups if group != removed)
            score[f"loo_without_{removed}"] = _combined_score(score, kept)
        joined = score.merge(policy, on="candidate_id", how="left", validate="one_to_one")
        # First evaluate exactly the predeclared primary-only versus complete
        # contract gate.  Leave-one-group-out metrics must not be inspected if
        # this gate fails.
        for arm in ["primary_only", "full_ae"]:
            metric_rows, metric_timestamp = _metrics(joined, arm, month, arm)
            rows.extend(metric_rows)
            timestamp_rows.append(metric_timestamp)
    metrics = pd.DataFrame(rows)
    grouped, selection = _selection(metrics)
    score_by_arm = selection.set_index("arm")["selection_score"]
    primary_score = float(score_by_arm.loc["primary_only"])
    full_score = float(score_by_arm.loc["full_ae"])
    full_wins = bool(full_score > primary_score)
    retained: list[str] = []
    decision_rows: list[dict[str, object]] = []
    if full_wins:
        # Re-read the immutable target-free receipts only after the full
        # contract passes its gate.  No refit occurs and no result is written
        # back to the source artifact.
        loo_rows: list[dict[str, object]] = []
        loo_timestamp_rows: list[pd.DataFrame] = []
        for path in score_paths:
            month = path.stem.split("=")[-1]
            score = pd.read_parquet(path)
            for removed in groups:
                kept = tuple(group for group in groups if group != removed)
                score[f"loo_without_{removed}"] = _combined_score(score, kept)
            joined = score.merge(policy, on="candidate_id", how="left", validate="one_to_one")
            for removed in groups:
                arm = f"loo_without_{removed}"
                metric_rows, metric_timestamp = _metrics(joined, arm, month, arm)
                loo_rows.extend(metric_rows)
                loo_timestamp_rows.append(metric_timestamp)
        metrics = pd.concat([metrics, pd.DataFrame(loo_rows)], ignore_index=True)
        timestamp_rows.extend(loo_timestamp_rows)
        grouped, selection = _selection(metrics)
        score_by_arm = selection.set_index("arm")["selection_score"]
        for group in groups:
            loo_arm = f"loo_without_{group}"
            loo_score = float(score_by_arm.loc[loo_arm])
            retain = bool(loo_score < full_score)
            if retain:
                retained.append(group)
            decision_rows.append({"group": group, "full_score": full_score, "without_group_score": loo_score, "delta_without_group": loo_score - full_score, "decision": "retain" if retain else "drop"})
    else:
        decision_rows = [{"group": group, "full_score": full_score, "without_group_score": np.nan, "delta_without_group": np.nan, "decision": "not_run_full_did_not_win"} for group in groups]
    metrics.to_parquet(out / "group_ablation_fold_metrics.parquet", index=False, compression="zstd")
    pd.concat(timestamp_rows, ignore_index=True).to_parquet(out / "group_ablation_timestamp_metrics.parquet", index=False, compression="zstd")
    grouped.to_parquet(out / "group_ablation_route_summary.parquet", index=False, compression="zstd")
    selection.to_parquet(out / "group_ablation_selection_summary.parquet", index=False, compression="zstd")
    pd.DataFrame(decision_rows).to_parquet(out / "group_ablation_decisions.parquet", index=False, compression="zstd")
    _write_json_exclusive(out / "run_manifest.json", {
        "schema": SCHEMA, "source": str(source), "months": [path.stem.split("=")[-1] for path in score_paths],
        "aux_groups": list(groups), "primary_score": primary_score, "full_score": full_score,
        "full_wins_primary_gate": full_wins, "retained_groups": retained if full_wins else None,
        "metric_aggregation": "ER50, recall, and selected EV are averaged equally across decision timestamps; candidate rows are never globally pooled for selection",
        "rule": "group retained only when removing it decreases full-suite timestamp-averaged selection score; otherwise dropped",
        "status": "complete",
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(args.source, args.out)


if __name__ == "__main__":
    main()
