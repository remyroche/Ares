#!/usr/bin/env python3
"""Compare strict-OOF router candidates with the parity-audited B0 control.

Every score is routed on the complete target-free candidate population.  The
canonical policy outcome is joined only afterwards, and each decision timestamp
is one evaluation unit.  Pooled rows, selected counts, and total bps are
disclosure values only and cannot choose a winning router arm.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "strict_r3_router_b0_timestamp_comparison_v1"
FRACTIONS = (0.30, 0.40, 0.50)
DEFAULT_CONTROL = ROOT / "data_perp/artifacts/strict_r3_long_base_recall_funnel_2025dev_holdout_2026oos_20260822_v1/b0_target_free_reconstruction.parquet"
DEFAULT_POLICY = ROOT / "data_perp/artifacts/strict_r3_enhanced_base_rich_policy_labels_reconciled_20260823_v1/canonical_reconciled_policy_labels.parquet"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json_exclusive(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(fd, "w") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, default=str)


def _route(frame: pd.DataFrame, score: str, fraction: float) -> np.ndarray:
    work = frame.loc[:, ["candidate_id", "__decision_ts__", score]].copy()
    work["__row__"] = np.arange(len(work), dtype=np.int64)
    work["__score__"] = pd.to_numeric(work[score], errors="coerce").fillna(-np.inf)
    work = work.sort_values(["__decision_ts__", "__score__", "candidate_id"], ascending=[True, False, True], kind="stable")
    work["__ordinal__"] = work.groupby("__decision_ts__", sort=False).cumcount() + 1
    count = work.groupby("__decision_ts__", sort=False)["candidate_id"].transform("size").to_numpy(float)
    selected = (work["__ordinal__"].to_numpy() <= np.ceil(float(fraction) * count)) & np.isfinite(work["__score__"].to_numpy(float))
    return pd.Series(selected, index=work["__row__"].to_numpy()).reindex(np.arange(len(frame))).to_numpy(bool)


def _timestamp_metrics(frame: pd.DataFrame, score: str, arm: str) -> pd.DataFrame:
    """Route before the outcome-validity filter; return one row/timestamp."""
    base = frame.loc[:, ["candidate_id", "__decision_ts__", score, "policy_path_valid", "policy_net_bps"]].copy()
    base["__net__"] = pd.to_numeric(base["policy_net_bps"], errors="coerce")
    base["__valid__"] = base["policy_path_valid"].fillna(False).astype(bool) & np.isfinite(base["__net__"])
    outputs: list[pd.DataFrame] = []
    for fraction in FRACTIONS:
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
            candidate_rows=("candidate_id", "size"), selected_rows=("__selected__", "sum"),
            valid_rows=("__valid__", "sum"), selected_valid_rows=("__selected_valid__", "sum"),
            selected_net_sum_bps=("__selected_net__", "sum"), excess_sum=("__excess__", "sum"),
            selected_excess_sum=("__selected_excess__", "sum"), winners50=("__winner50__", "sum"),
            selected_winners50=("__selected_winner50__", "sum"), winners100=("__winner100__", "sum"),
            selected_winners100=("__selected_winner100__", "sum"),
        ).reset_index()
        denom = grouped["selected_valid_rows"].to_numpy(float)
        grouped["timestamp_net_ev_bps"] = np.divide(grouped["selected_net_sum_bps"], denom, out=np.full(len(grouped), np.nan), where=denom > 0)
        denom = grouped["excess_sum"].to_numpy(float)
        grouped["timestamp_er50"] = np.divide(grouped["selected_excess_sum"], denom, out=np.full(len(grouped), np.nan), where=denom > 0)
        winners50 = grouped["winners50"].to_numpy(float)
        winners100 = grouped["winners100"].to_numpy(float)
        grouped["timestamp_recall50"] = np.divide(grouped["selected_winners50"], winners50, out=np.full(len(grouped), np.nan), where=winners50 > 0)
        grouped["timestamp_recall100"] = np.divide(grouped["selected_winners100"], winners100, out=np.full(len(grouped), np.nan), where=winners100 > 0)
        grouped["arm"] = arm
        grouped["route_fraction"] = fraction
        outputs.append(grouped)
    return pd.concat(outputs, ignore_index=True)


def _summary(timestamp: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (arm, fraction), work in timestamp.groupby(["arm", "route_fraction"], sort=False):
        selected = float(work["selected_valid_rows"].sum())
        total = float(work["selected_net_sum_bps"].sum())
        rows.append({
            "arm": arm, "route_fraction": float(fraction), "timestamps": int(len(work)),
            "er50_timestamps": int(work["timestamp_er50"].notna().sum()),
            "recall50_timestamps": int(work["timestamp_recall50"].notna().sum()),
            "recall100_timestamps": int(work["timestamp_recall100"].notna().sum()),
            "ev_timestamps": int(work["timestamp_net_ev_bps"].notna().sum()),
            "timestamp_mean_er50": float(work["timestamp_er50"].mean()),
            "timestamp_mean_recall50": float(work["timestamp_recall50"].mean()),
            "timestamp_mean_recall100": float(work["timestamp_recall100"].mean()),
            "timestamp_mean_net_ev_bps": float(work["timestamp_net_ev_bps"].mean()),
            "worst_timestamp_net_ev_bps": float(work["timestamp_net_ev_bps"].min()),
            "selected_valid_rows": int(selected), "net_sum_bps": total,
            "trade_weighted_net_ev_bps": total / selected if selected else np.nan,
        })
    return pd.DataFrame(rows)


def _parse_arm(value: str) -> tuple[str, Path]:
    name, separator, text = value.partition("=")
    if not separator or not name or not text:
        raise argparse.ArgumentTypeError("--router-arm must be NAME=PATH")
    return name, Path(text)


def run(args: argparse.Namespace) -> None:
    if args.out.exists():
        raise FileExistsError(args.out)
    arms = dict(args.router_arm)
    if not arms or len(arms) != len(args.router_arm):
        raise ValueError("at least one unique --router-arm is required")
    identity = ["candidate_id", "__decision_ts__", "side_name"]
    panels: list[pd.DataFrame] = []
    metadata: dict[str, object] = {}
    canonical_keys: pd.DataFrame | None = None
    for name, root in arms.items():
        contract_path = root / "run_contract.json"
        manifest_path = root / "run_manifest.json"
        if not contract_path.exists() or not manifest_path.exists():
            raise FileNotFoundError(f"{name}: incomplete router artifact")
        contract = json.loads(contract_path.read_text())
        manifest = json.loads(manifest_path.read_text())
        if contract.get("schema") != "strict_r3_economic_recall_router_ae_v1" or manifest.get("status") != "complete":
            raise AssertionError(f"{name}: not a completed strict router source")
        parts = [pd.read_parquet(path, columns=[*identity, "router_primary_only_rank"])
                 for path in sorted((root / "target_free_scores").glob("month=*.parquet"))]
        panel = pd.concat(parts, ignore_index=True)
        panel["__decision_ts__"] = pd.to_datetime(panel["__decision_ts__"], utc=True, errors="raise")
        if panel["candidate_id"].duplicated().any():
            raise AssertionError(f"{name}: duplicate target-free candidate identity")
        if canonical_keys is None:
            canonical_keys = panel.loc[:, identity].copy()
        else:
            check = canonical_keys.merge(panel.loc[:, identity], on=identity, how="outer", indicator=True)
            if len(check) != len(canonical_keys) or not check["_merge"].eq("both").all():
                raise AssertionError(f"{name}: candidate population differs from the first router arm")
        panels.append(panel.rename(columns={"router_primary_only_rank": name}))
        metadata[name] = {"root": str(root), "contract_sha256": _sha256(contract_path), "manifest_sha256": _sha256(manifest_path)}
    assert canonical_keys is not None
    control = pd.read_parquet(args.control, columns=["candidate_id", "__decision_ts__", "side_name", "base_score"])
    control["__decision_ts__"] = pd.to_datetime(control["__decision_ts__"], utc=True, errors="raise")
    if control["candidate_id"].duplicated().any():
        raise AssertionError("B0 control has duplicate candidate identity")
    scores = canonical_keys.merge(control, on=identity, how="left", validate="one_to_one")
    if scores["base_score"].isna().any():
        raise AssertionError("B0 control does not cover every router candidate")
    scores = scores.rename(columns={"base_score": "B0"})
    for panel in panels:
        scores = scores.merge(panel, on=identity, how="left", validate="one_to_one")
    if scores.isna().any().any():
        raise AssertionError("router score merge created a missing value")
    policy = pd.read_parquet(args.policy, columns=["candidate_id", "policy_path_valid", "policy_net_bps"])
    if policy["candidate_id"].duplicated().any():
        raise AssertionError("policy ledger has duplicate candidate identity")
    joined = scores.merge(policy, on="candidate_id", how="left", validate="one_to_one")
    output = [_timestamp_metrics(joined, field, field) for field in ["B0", *arms]]
    timestamp = pd.concat(output, ignore_index=True)
    summary = _summary(timestamp)
    args.out.mkdir(parents=True)
    timestamp.to_parquet(args.out / "timestamp_metrics.parquet", index=False, compression="zstd")
    summary.to_parquet(args.out / "summary.parquet", index=False, compression="zstd")
    _write_json_exclusive(args.out / "run_manifest.json", {
        "schema": SCHEMA, "status": "complete", "router_arms": metadata,
        "control": str(args.control), "control_sha256": _sha256(args.control),
        "policy": str(args.policy), "policy_sha256": _sha256(args.policy),
        "candidate_rows": int(len(scores)), "timestamps": int(scores["__decision_ts__"].nunique()),
        "metric_aggregation": "route before label exclusion; ER/recall/EV are equal decision-timestamp means; pooled totals are disclosure only",
        "summary": summary.to_dict(orient="records"),
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--router-arm", action="append", type=_parse_arm, required=True)
    parser.add_argument("--control", type=Path, default=DEFAULT_CONTROL)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--out", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
