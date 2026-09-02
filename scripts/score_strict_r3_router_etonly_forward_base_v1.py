#!/usr/bin/env python3
"""Produce one target-free routed 50/50 E/T base score month.

This offline bridge completes the strict sequence for a forward month:

    complete point-in-time candidate grid
      -> frozen primary router's exact timestamp-local top-50% membership
      -> E/T models fitted only on prior routed and resolved rows
      -> target-free 50/50 E/T score receipt for every routed held row.

The router rank is used solely to determine membership.  It is neither a base
model feature nor persisted in the output.  No held label, path, policy, or
candidate eligibility field is opened while scoring the forward month.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
for _item in (ROOT, ROOT / "scripts"):
    if str(_item) not in sys.path:
        sys.path.insert(0, str(_item))

import run_strict_r3_o3v2_target_funnel as target_contract  # noqa: E402
import run_strict_r3_router_routed_base_stack as routed  # noqa: E402


SCHEMA = "strict_r3_router_etonly_forward_base_v1"
IDENTITY = ("candidate_id", "__decision_ts__", "side_name")
ROUTER_FIELD = "router_primary_rank"


def _exclusive_json(path: Path, payload: object) -> None:
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _month(text: str) -> pd.Timestamp:
    value = pd.Timestamp(f"{text}-01", tz="UTC")
    if value.strftime("%Y-%m") != text:
        raise ValueError("--held-month must be YYYY-MM")
    return value


def _month_end(month: pd.Timestamp) -> pd.Timestamp:
    return month + pd.offsets.MonthBegin(1)


def _base_fields(root: Path) -> tuple[str, ...]:
    sample = root / "month=2026-07" / "scores_features.parquet"
    names = pq.ParquetFile(sample).schema_arrow.names
    prefix = (
        "candidate_id", "__decision_ts__", "base_bps", "efficiency_bps",
        "timing_bps", "enhanced_base_bps", "base_rank_ts",
        "enhanced_base_routed", "e_minus_t", "e_minus_b0", "t_minus_b0",
        "base_component_std", "side_name",
    )
    if tuple(names[:len(prefix)]) != prefix:
        raise AssertionError(f"{sample}: unexpected frozen source handoff")
    fields = tuple(names[len(prefix):])
    if len(fields) != 120 or len(set(fields)) != len(fields):
        raise AssertionError(f"{sample}: expected ordered 120-field base contract")
    return fields


def _source_for_month(
    month: pd.Timestamp,
    *, historical_root: Path,
    forward_root: Path,
    held_month: pd.Timestamp,
) -> Path:
    if month == held_month:
        return forward_root / f"month={month:%Y-%m}" / "causal_feature_universe.parquet"
    return historical_root / f"month={month:%Y-%m}" / "scores_features.parquet"


def _router_for_month(
    month: pd.Timestamp,
    *, historical_root: Path,
    forward_root: Path,
    held_month: pd.Timestamp,
) -> Path:
    root = forward_root if month == held_month else historical_root
    return root / "target_free_scores" / f"month={month:%Y-%m}.parquet"


def _read_features(
    *, start: pd.Timestamp, end: pd.Timestamp, held_month: pd.Timestamp,
    historical_base_root: Path, forward_base_root: Path, fields: tuple[str, ...],
) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    months = pd.date_range(start.normalize().replace(day=1), (end - pd.Timedelta(nanoseconds=1)).normalize().replace(day=1), freq="MS", tz="UTC")
    for month in months:
        path = _source_for_month(month, historical_root=historical_base_root, forward_root=forward_base_root, held_month=held_month)
        if not path.exists():
            raise FileNotFoundError(path)
        names = set(pq.ParquetFile(path).schema_arrow.names)
        leaked = sorted(set(target_contract.PROHIBITED_SCORE_COLUMNS).intersection(names))
        if leaked:
            raise AssertionError(f"{path}: target-free source leaks {leaked}")
        missing = sorted(set((*IDENTITY, *fields)).difference(names))
        if missing:
            raise AssertionError(f"{path}: missing feature identity/contract {missing[:10]}")
        part = pd.read_parquet(path, columns=[*IDENTITY, *fields])
        part["__decision_ts__"] = pd.to_datetime(part["__decision_ts__"], utc=True, errors="raise")
        pieces.append(part.loc[part.__decision_ts__.ge(start) & part.__decision_ts__.lt(end)].copy())
    result = pd.concat(pieces, ignore_index=True)
    if result.empty or result.duplicated(IDENTITY).any() or not result.side_name.eq("long").all():
        raise AssertionError("invalid target-free E/T feature identity window")
    return result


def _read_router(
    *, start: pd.Timestamp, end: pd.Timestamp, held_month: pd.Timestamp,
    historical_router_root: Path, forward_router_root: Path,
) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    months = pd.date_range(start.normalize().replace(day=1), (end - pd.Timedelta(nanoseconds=1)).normalize().replace(day=1), freq="MS", tz="UTC")
    for month in months:
        path = _router_for_month(month, historical_root=historical_router_root, forward_root=forward_router_root, held_month=held_month)
        if not path.exists():
            raise FileNotFoundError(path)
        part = pd.read_parquet(path, columns=[*IDENTITY, ROUTER_FIELD])
        part["__decision_ts__"] = pd.to_datetime(part["__decision_ts__"], utc=True, errors="raise")
        pieces.append(part.loc[part.__decision_ts__.ge(start) & part.__decision_ts__.lt(end)].copy())
    result = pd.concat(pieces, ignore_index=True)
    values = pd.to_numeric(result[ROUTER_FIELD], errors="coerce")
    if result.empty or result.duplicated(IDENTITY).any() or not np.isfinite(values).all():
        raise AssertionError("invalid strict-prequential router score window")
    return result


def _route(frame: pd.DataFrame, fraction: float) -> np.ndarray:
    return routed.parent._exact_timestamp_top_fraction(frame, ROUTER_FIELD, fraction).to_numpy(bool)


def run(args: argparse.Namespace) -> None:
    if args.out.exists():
        raise FileExistsError(f"{args.out}: immutable output already exists")
    month = _month(args.held_month)
    if not 0.0 < args.route_fraction <= 1.0:
        raise ValueError("route fraction must lie in (0, 1]")
    fields = _base_fields(args.historical_base_root)
    reserve = month - pd.Timedelta(days=args.reserve_days)
    train_start = reserve - pd.DateOffset(months=args.train_months)
    held_end = _month_end(month)
    args.out.mkdir(parents=True)
    _exclusive_json(args.out / "run_manifest.json", {
        "schema": SCHEMA,
        "scope": "offline target-free forward E/T base scoring; no MC1/admission/portfolio/inference/live/exchange mutation",
        "held_month": f"{month:%Y-%m}", "train_start": train_start.isoformat(), "reserve_start": reserve.isoformat(),
        "base_contract": "0.50 * strict-prequential efficiency_bps + 0.50 * strict-prequential timing_bps; no R3 authority",
        "router_contract": "exact timestamp-local top-50 membership only; numeric router rank excluded from base and output",
        "historical_base_root": str(args.historical_base_root), "forward_base_root": str(args.forward_base_root),
        "historical_router_root": str(args.historical_router_root), "forward_router_root": str(args.forward_router_root),
        "labels_root": str(args.labels_root), "policy_path": str(args.policy_path),
        "feature_contract": list(fields), "feature_contract_sha256": hashlib.sha256("\n".join(fields).encode()).hexdigest(),
        "route_fraction": float(args.route_fraction), "train_months": int(args.train_months), "reserve_days": int(args.reserve_days),
        "source_hashes": {
            "forward_base": _sha_file(args.forward_base_root / f"month={month:%Y-%m}" / "causal_feature_universe.parquet"),
            "forward_router": _sha_file(args.forward_router_root / "target_free_scores" / f"month={month:%Y-%m}.parquet"),
        },
    })
    all_start = train_start
    all_end = held_end
    features = _read_features(start=all_start, end=all_end, held_month=month, historical_base_root=args.historical_base_root, forward_base_root=args.forward_base_root, fields=fields)
    router = _read_router(start=all_start, end=all_end, held_month=month, historical_router_root=args.historical_router_root, forward_router_root=args.forward_router_root)
    merged = features.merge(router, on=list(IDENTITY), how="inner", validate="one_to_one")
    if len(merged) != len(features) or len(merged) != len(router):
        raise AssertionError("router/base target-free identity mismatch")
    merged["router_routed"] = _route(merged, args.route_fraction)
    policy = routed._load_policy(args.policy_path)
    labels = routed._read_supportive_window(args.labels_root, start=train_start, end=reserve)
    train_raw = merged.loc[merged.__decision_ts__.lt(reserve)].merge(labels, on=list(IDENTITY), how="left", validate="one_to_one")
    train_raw = train_raw.merge(policy, on="candidate_id", how="left", validate="one_to_one")
    if train_raw[["supportive_path_valid", "policy_path_valid"]].isna().any().any():
        raise AssertionError("prior resolved label identity coverage failed")
    train = routed._strict_train(train_raw, reserve, include_b0=False)
    held = merged.loc[merged.__decision_ts__.ge(month)].copy()
    held = held.loc[held.router_routed].copy()
    if len(train) < routed.MIN_BASE_TRAIN_ROWS or len(held) < 1000:
        raise AssertionError(f"insufficient routed support: train={len(train)} held={len(held)}")
    efficiency, efficiency_audit = routed._fit_direct(train, held, fields, "supportive_path_efficiency_h12", 1.0, args.n_jobs, routed.SEED + 1000)
    timing, timing_audit = routed._fit_direct(train, held, fields, "supportive_time_to_meaningful_mfe_h12", -1.0, args.n_jobs, routed.SEED + 2000)
    enhanced = (.5 * efficiency + .5 * timing).astype(np.float32)
    result = held.loc[:, [*IDENTITY, *fields]].copy()
    result.insert(3, "base_bps", enhanced)
    result.insert(4, "efficiency_bps", efficiency)
    result.insert(5, "timing_bps", timing)
    result.insert(6, "enhanced_base_bps", enhanced)
    result.insert(7, "base_rank_ts", routed.parent._rank_pct(result, "enhanced_base_bps").to_numpy(np.float32))
    result.insert(8, "enhanced_base_routed", True)
    result.insert(9, "e_minus_t", (efficiency - timing).astype(np.float32))
    result.insert(10, "e_minus_b0", np.zeros(len(result), dtype=np.float32))
    result.insert(11, "t_minus_b0", (timing - enhanced).astype(np.float32))
    result.insert(12, "base_component_std", np.nanstd(np.column_stack([enhanced, efficiency, timing]), axis=1).astype(np.float32))
    output = args.out / "target_free_monthly" / f"month={month:%Y-%m}"
    output.mkdir(parents=True)
    result.to_parquet(output / "scores_features.parquet", index=False, compression="zstd")
    audit = {
        "held_month": f"{month:%Y-%m}", "held_full_population_rows": int((merged.__decision_ts__.ge(month)).sum()),
        "held_routed_rows": int(len(result)), "route_fraction": float(args.route_fraction),
        "train_rows": int(len(train)), "base_feature_complete_fraction": float(result.loc[:, list(fields)].notna().all(axis=1).mean()),
        "all_base_training_rows_router_selected": True, "r3_direct_ranking_authority": False,
        "router_numeric_input": False, "held_labels_opened": False,
        "base_semantics": "0.50 * efficiency_bps + 0.50 * timing_bps",
        "efficiency_target": "supportive_path_efficiency_h12", "timing_target": "negative_supportive_time_to_meaningful_mfe_h12",
        "efficiency_map": efficiency_audit, "timing_map": timing_audit,
    }
    pd.DataFrame([audit]).to_parquet(args.out / "routed_et_forward_fold_audit.parquet", index=False, compression="zstd")
    prohibited = sorted(set(target_contract.PROHIBITED_SCORE_COLUMNS).intersection(result.columns))
    if prohibited:
        raise AssertionError(f"target-free forward E/T output leaks {prohibited}")
    (args.out / "correctness_report.json").write_text(json.dumps({
        "target_free_output": True, "held_labels_opened": False,
        "all_base_training_rows_router_selected": True, "r3_direct_ranking_authority": False,
        "router_has_numeric_downstream_authority": False,
        "et_identity_max_error": float(np.abs(result.enhanced_base_bps - .5 * (result.efficiency_bps + result.timing_bps)).max()),
        "output_rows": int(len(result)),
    }, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical-base-root", type=Path, required=True)
    parser.add_argument("--forward-base-root", type=Path, required=True)
    parser.add_argument("--historical-router-root", type=Path, required=True)
    parser.add_argument("--forward-router-root", type=Path, required=True)
    parser.add_argument("--labels-root", type=Path, required=True)
    parser.add_argument("--policy-path", type=Path, required=True)
    parser.add_argument("--held-month", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--route-fraction", type=float, default=.50)
    parser.add_argument("--train-months", type=int, default=2)
    parser.add_argument("--reserve-days", type=int, default=28)
    parser.add_argument("--n-jobs", type=int, default=4)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
