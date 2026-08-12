#!/usr/bin/env python3
"""Backfill unavailable strict-R3 policy paths from the canonical hourly cache.

Existing exact/15-minute outcomes always win.  The one-hour replay is an
explicitly labelled, conservative proxy used only where the finer path is
unavailable.  The immutable output includes overlap diagnostics so economic
results can be split by outcome-source quality.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.strict_r3_frozen_policy_labels import (  # noqa: E402
    COST_BPS,
    replay_policy_hourly_proxy,
)


DEFAULT_INPUT = ROOT / (
    "data_perp/artifacts/"
    "strict_r3_schema_v2_optimised_policy_replay_targetfree_long_2025_aug7_2026_20260809_v1/"
    "candidate_policy_outcomes.parquet"
)
DEFAULT_HOURLY = ROOT / "data_perp/artifacts/canonical_hourly_primitive_cache_v1/hourly"
DEFAULT_POLICY = ROOT / (
    "data_perp/artifacts/"
    "strict_r3_schema_v2_simple_policy_targetfree_long_pre2025_20260809_v3/"
    "winner.json"
)
IDENTITY = [
    "candidate_id", "__ts__", "__decision_ts__", "__symbol__", "side_name",
]
POLICY_COLUMNS = [
    "atr_1h", "policy_path_valid", "policy_gross_bps", "policy_net_bps",
    "policy_exit_bar_15m", "policy_exit_reason", "policy_entry_price",
    "policy_exit_price", "policy_atr_source", "policy_atr",
    "policy_label_available_ts", "policy_market_data_source",
    "policy_market_data_quality",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _hourly_path(root: Path, symbol: str) -> Path:
    return root / f"symbol={str(symbol).replace('/', '_')}" / "part.parquet"


def _load_hourly(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path, columns=["open", "high", "low", "close"])
    if not isinstance(frame.index, pd.DatetimeIndex):
        if "ts" not in frame:
            raise ValueError(f"hourly cache lacks a timestamp index: {path}")
        frame = frame.set_index("ts")
    frame.index = pd.to_datetime(frame.index, utc=True, errors="raise")
    return frame.sort_index()


def _finite_mean(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce")
    return float(numeric.mean()) if numeric.notna().any() else np.nan


def _overlap_metrics(frame: pd.DataFrame) -> dict[str, object]:
    valid = frame.loc[
        frame["original_valid"].fillna(False).astype(bool)
        & frame["hourly_valid"].fillna(False).astype(bool)
        & np.isfinite(pd.to_numeric(frame["original_net_bps"], errors="coerce"))
        & np.isfinite(pd.to_numeric(frame["hourly_net_bps"], errors="coerce"))
    ].copy()
    if valid.empty:
        return {
            "overlap_rows": 0, "net_spearman": np.nan,
            "net_mae_bps": np.nan, "net_bias_bps": np.nan,
            "net_sign_agreement": np.nan,
        }
    original = valid["original_net_bps"].to_numpy(float)
    hourly = valid["hourly_net_bps"].to_numpy(float)
    correlation = spearmanr(original, hourly).statistic
    return {
        "overlap_rows": int(len(valid)),
        "net_spearman": float(correlation) if np.isfinite(correlation) else np.nan,
        "net_mae_bps": float(np.mean(np.abs(hourly - original))),
        "net_median_ae_bps": float(np.median(np.abs(hourly - original))),
        "net_bias_bps": float(np.mean(hourly - original)),
        "net_sign_agreement": float(np.mean((hourly > 0.0) == (original > 0.0))),
        "original_net_bps": float(np.mean(original)),
        "hourly_net_bps": float(np.mean(hourly)),
    }


def _source_name(frame: pd.DataFrame) -> pd.Series:
    if "policy_market_data_source" not in frame:
        return pd.Series("existing_15m_or_exact", index=frame.index)
    source = frame["policy_market_data_source"].fillna("").astype(str)
    return source.where(source.ne(""), "existing_15m_or_exact")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-outcomes", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--hourly-root", type=Path, default=DEFAULT_HOURLY)
    parser.add_argument("--policy-json", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--start", default="2025-01-01")
    parser.add_argument("--end", default="2026-08-01", help="exclusive UTC bound")
    parser.add_argument("--overlap-per-symbol", type=int, default=2_000)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(f"immutable output exists: {args.out_dir}")
    start = pd.Timestamp(args.start, tz="UTC")
    end = pd.Timestamp(args.end, tz="UTC")
    policy_document = json.loads(args.policy_json.read_text())
    policy = {
        key: float(policy_document["winner"][key])
        for key in (
            "sl_mult", "trailing_activation_mult",
            "fixed_trailing_gap_mult",
        )
    }
    available_columns = set(
        __import__("pyarrow.parquet", fromlist=["ParquetFile"])
        .ParquetFile(args.input_outcomes).schema.names
    )
    columns = [*IDENTITY, *[name for name in POLICY_COLUMNS if name in available_columns]]
    frame = pd.read_parquet(
        args.input_outcomes,
        columns=columns,
        filters=[("__decision_ts__", ">=", start), ("__decision_ts__", "<", end)],
    )
    for column in ("__ts__", "__decision_ts__", "policy_label_available_ts"):
        if column in frame:
            frame[column] = pd.to_datetime(frame[column], utc=True, errors="raise")
    if frame.empty or frame["candidate_id"].duplicated().any():
        raise ValueError("policy source is empty or has duplicate identities")
    if not frame["side_name"].astype(str).str.lower().eq("long").all():
        raise ValueError("this repair is intentionally long-only")
    if "policy_path_valid" not in frame:
        raise ValueError("policy source lacks path validity")
    if "atr_1h" not in frame:
        frame["atr_1h"] = np.nan
    original_valid = (
        frame["policy_path_valid"].fillna(False).astype(bool)
        & np.isfinite(pd.to_numeric(frame["policy_net_bps"], errors="coerce"))
    )
    frame["policy_outcome_source"] = np.where(
        original_valid, _source_name(frame), "unavailable",
    )
    frame["policy_cost_bps"] = np.where(original_valid, COST_BPS, np.nan)
    frame["policy_proxy_resolution_minutes"] = np.where(original_valid, 15, 0)

    repairs: list[pd.DataFrame] = []
    overlaps: list[pd.DataFrame] = []
    coverage_rows: list[dict[str, object]] = []
    symbols = sorted(frame["__symbol__"].astype(str).unique())
    for number, (symbol, block) in enumerate(frame.groupby("__symbol__", sort=True), 1):
        path = _hourly_path(args.hourly_root, str(symbol))
        if not path.exists():
            coverage_rows.append({
                "symbol": str(symbol), "rows": int(len(block)),
                "original_valid_rows": int(original_valid.loc[block.index].sum()),
                "hourly_source_exists": False, "hourly_proxy_valid_rows": 0,
                "backfilled_rows": 0,
            })
            continue
        bars = _load_hourly(path)
        invalid = block.loc[~original_valid.loc[block.index]].copy()
        existing = block.loc[original_valid.loc[block.index]].copy()
        if len(existing) > args.overlap_per_symbol:
            existing = existing.sample(
                n=args.overlap_per_symbol, random_state=20260810,
            ).sort_index()
        replay_input = pd.concat([invalid, existing], ignore_index=False)
        proxy = replay_policy_hourly_proxy(
            replay_input,
            bars,
            stop_loss_atr=policy["sl_mult"],
            trailing_activation_atr=policy["trailing_activation_mult"],
            trailing_giveback_atr=policy["fixed_trailing_gap_mult"],
            timeout_hours=12,
            cost_bps=COST_BPS,
        )
        proxy.index = replay_input.index
        proxy_valid = proxy["policy_path_valid"].fillna(False).astype(bool)
        invalid_proxy = proxy.loc[proxy.index.isin(invalid.index) & proxy_valid].copy()
        if not invalid_proxy.empty:
            repairs.append(invalid_proxy)
        existing_proxy = proxy.loc[proxy.index.isin(existing.index)].copy()
        if not existing_proxy.empty:
            overlaps.append(pd.DataFrame({
                "candidate_id": existing_proxy["candidate_id"].astype(str),
                "__decision_ts__": existing_proxy["__decision_ts__"],
                "__symbol__": str(symbol),
                "original_valid": True,
                "hourly_valid": existing_proxy["policy_path_valid"].to_numpy(bool),
                "original_net_bps": frame.loc[existing_proxy.index, "policy_net_bps"].to_numpy(float),
                "hourly_net_bps": existing_proxy["policy_net_bps"].to_numpy(float),
            }))
        coverage_rows.append({
            "symbol": str(symbol), "rows": int(len(block)),
            "original_valid_rows": int(original_valid.loc[block.index].sum()),
            "hourly_source_exists": True,
            "hourly_proxy_valid_rows": int(proxy_valid.sum()),
            "backfilled_rows": int(len(invalid_proxy)),
        })
        if number % 20 == 0 or number == len(symbols):
            print(json.dumps({
                "event": "hourly_proxy_progress", "symbols_complete": number,
                "symbols_total": len(symbols),
            }), flush=True)

    repaired = frame.copy()
    if repairs:
        repair = pd.concat(repairs, ignore_index=False)
        policy_fields = [
            "policy_path_valid", "policy_gross_bps", "policy_net_bps",
            "policy_exit_bar_15m", "policy_exit_reason", "policy_entry_price",
            "policy_exit_price", "policy_atr_source", "policy_atr",
            "policy_label_available_ts", "policy_cost_bps",
            "policy_outcome_source", "policy_market_data_source",
            "policy_market_data_quality",
        ]
        object_fields = {
            "policy_exit_reason", "policy_atr_source", "policy_outcome_source",
            "policy_market_data_source", "policy_market_data_quality",
        }
        for column in policy_fields:
            if column not in repaired:
                repaired[column] = (
                    pd.Series(pd.NA, index=repaired.index, dtype="object")
                    if column in object_fields else np.nan
                )
            elif column in object_fields and repaired[column].dtype != object:
                repaired[column] = repaired[column].astype("object")
            repaired.loc[repair.index, column] = repair[column]
        repaired.loc[repair.index, "policy_proxy_resolution_minutes"] = 60
    repaired_valid = (
        repaired["policy_path_valid"].fillna(False).astype(bool)
        & np.isfinite(pd.to_numeric(repaired["policy_net_bps"], errors="coerce"))
    )
    if not repaired.loc[original_valid, "policy_net_bps"].equals(
        frame.loc[original_valid, "policy_net_bps"],
    ):
        raise AssertionError("hourly backfill overwrote an existing valid outcome")
    if repaired["candidate_id"].duplicated().any() or len(repaired) != len(frame):
        raise AssertionError("hourly backfill changed identity/cardinality")
    valid_cost = repaired_valid & np.isfinite(
        pd.to_numeric(repaired["policy_gross_bps"], errors="coerce")
    )
    if valid_cost.any() and not np.allclose(
        repaired.loc[valid_cost, "policy_net_bps"],
        repaired.loc[valid_cost, "policy_gross_bps"] - COST_BPS,
        rtol=0.0, atol=1e-9,
    ):
        raise AssertionError("policy cost is not applied exactly once")

    repaired["month"] = repaired["__decision_ts__"].dt.strftime("%Y-%m")
    coverage = repaired.groupby(["month", "policy_outcome_source"], as_index=False).agg(
        rows=("candidate_id", "size"),
        valid_rows=("policy_path_valid", "sum"),
        mean_net_bps=("policy_net_bps", _finite_mean),
    )
    symbol_coverage = pd.DataFrame(coverage_rows)
    overlap = pd.concat(overlaps, ignore_index=True) if overlaps else pd.DataFrame()
    overlap_summary = _overlap_metrics(overlap)
    zero_before = (
        frame.assign(_valid=original_valid)
        .groupby("__symbol__")["_valid"].sum().eq(0)
    )
    zero_after = (
        repaired.assign(_valid=repaired_valid)
        .groupby("__symbol__")["_valid"].sum().eq(0)
    )
    audit_start = max(start, pd.Timestamp("2025-01-01", tz="UTC"))
    audit_end = min(end, pd.Timestamp("2025-08-01", tz="UTC"))
    audit_mask = (
        frame["__decision_ts__"].ge(audit_start)
        & frame["__decision_ts__"].lt(audit_end)
    )
    audit_before = (
        frame.loc[audit_mask].assign(_valid=original_valid.loc[audit_mask])
        .groupby("__symbol__")["_valid"].sum().eq(0)
    )
    audit_after = (
        repaired.loc[audit_mask].assign(_valid=repaired_valid.loc[audit_mask])
        .groupby("__symbol__")["_valid"].sum().eq(0)
    )
    manifest = {
        "schema": "strict_r3_policy_outcome_hourly_backfill_v1",
        "side": "long",
        "input": str(args.input_outcomes),
        "input_sha256": _sha256(args.input_outcomes),
        "hourly_root": str(args.hourly_root),
        "policy_json": str(args.policy_json),
        "policy_json_sha256": _sha256(args.policy_json),
        "period_start": start.isoformat(),
        "period_end_exclusive": end.isoformat(),
        "policy": policy,
        "timeout_hours": 12,
        "cost_bps_once": COST_BPS,
        "precedence": "existing exact_or_15m outcome then hourly OHLC proxy",
        "hourly_ordering": "stop first; prior-hour trailing state; current-hour MFE update",
        "rows": int(len(repaired)),
        "valid_rows_before": int(original_valid.sum()),
        "valid_rows_after": int(repaired_valid.sum()),
        "coverage_before": float(original_valid.mean()),
        "coverage_after": float(repaired_valid.mean()),
        "backfilled_rows": int(repaired_valid.sum() - original_valid.sum()),
        "zero_valid_symbols_before": int(zero_before.sum()),
        "zero_valid_symbols_after": int(zero_after.sum()),
        "jan_jul_2025_audit": {
            "period_start": audit_start.isoformat(),
            "period_end_exclusive": audit_end.isoformat(),
            "rows": int(audit_mask.sum()),
            "valid_rows_before": int(original_valid.loc[audit_mask].sum()),
            "valid_rows_after": int(repaired_valid.loc[audit_mask].sum()),
            "zero_valid_symbols_before": int(audit_before.sum()),
            "zero_valid_symbols_after": int(audit_after.sum()),
        },
        "overlap_validation": overlap_summary,
        "status": "complete",
    }
    args.out_dir.mkdir(parents=True)
    repaired.sort_values(
        ["__decision_ts__", "__symbol__", "candidate_id"], kind="stable",
    ).to_parquet(
        args.out_dir / "candidate_policy_outcomes.parquet",
        index=False, compression="zstd",
    )
    coverage.to_parquet(args.out_dir / "policy_coverage_by_month_source.parquet", index=False)
    symbol_coverage.to_parquet(args.out_dir / "policy_coverage_by_symbol.parquet", index=False)
    if not overlap.empty:
        overlap.to_parquet(args.out_dir / "hourly_proxy_overlap.parquet", index=False)
    (args.out_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str) + "\n",
    )
    print(json.dumps({"event": "complete", **manifest}, default=str), flush=True)


if __name__ == "__main__":
    main()
