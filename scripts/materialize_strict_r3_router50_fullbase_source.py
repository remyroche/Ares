#!/usr/bin/env python3
"""Materialise a target-free top-50% router view of frozen full-base scores.

Research-only adapter for the matched meta comparison.  It does *not* refit
or alter the full-trained enhanced three-way base.  For each decision
timestamp it joins strict-OOF router outputs, keeps exactly the top 50% by
``router_primary_rank``, overwrites only the route flag, and optionally
persists the three router outputs for the router-feature meta arm.
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
for item in (ROOT, ROOT / "scripts"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

import run_strict_r3_enhanced_base_live_stack_challenger as parent  # noqa: E402
import run_strict_r3_router_routed_base_stack as routed  # noqa: E402


MONTHS = tuple(pd.date_range("2025-10-01", "2026-07-01", freq="MS", tz="UTC"))
PROHIBITED = frozenset({
    "policy_path_valid", "policy_gross_bps", "policy_net_bps", "policy_exit_bar_15m",
    "policy_entry_price", "policy_exit_price", "policy_exit_reason", "policy_label_available_ts",
    "policy_cost_bps", "supportive_path_valid", "supportive_target_invalid",
    "supportive_label_available_ts", "h12_tp6_sl4_net_bps",
})


def _write_json_exclusive(path: Path, payload: object) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)


def _hash_coordinates(frame: pd.DataFrame) -> str:
    columns = ["candidate_id", "base_bps", "efficiency_bps", "timing_bps", "enhanced_base_bps"]
    packed = pd.util.hash_pandas_object(frame.loc[:, columns], index=False).to_numpy(np.uint64)
    return hashlib.sha256(packed.tobytes()).hexdigest()


def run(*, source_root: Path, router_root: Path, out: Path, route_fraction: float) -> None:
    if out.exists():
        raise FileExistsError(out)
    if route_fraction != .50:
        raise ValueError("this frozen matched source is deliberately top-50% only")
    out.mkdir(parents=True)
    target = out / "target_free_monthly"
    target.mkdir()
    fields = routed._source_fields(source_root)
    audits: list[dict[str, object]] = []
    for month in MONTHS:
        token = f"{month:%Y-%m}"
        source_path = source_root / "target_free_monthly" / f"month={token}" / "scores_features.parquet"
        router_path = router_root / "target_free_scores" / f"month={token}.parquet"
        if not source_path.exists() or not router_path.exists():
            raise FileNotFoundError(f"{token}: source={source_path.exists()} router={router_path.exists()}")
        source_names = set(pq.ParquetFile(source_path).schema_arrow.names)
        leaked = sorted(PROHIBITED.intersection(source_names))
        if leaked:
            raise AssertionError(f"{token}: source contains outcome columns {leaked}")
        source = pd.read_parquet(source_path)
        router = routed._read_router_month(router_root, token)
        source["__decision_ts__"] = pd.to_datetime(source["__decision_ts__"], utc=True, errors="raise")
        if source["candidate_id"].duplicated().any() or not source["side_name"].astype(str).str.lower().eq("long").all():
            raise AssertionError(f"{token}: full-base identity/side contract failed")
        merged = source.merge(
            router, on=["candidate_id", "__decision_ts__", "side_name"], how="inner", validate="one_to_one",
        )
        if len(merged) != len(source) or len(merged) != len(router):
            raise AssertionError(f"{token}: full-base/router universe mismatch {len(source)}/{len(router)}/{len(merged)}")
        expected = parent._exact_timestamp_top_fraction(merged, "router_primary_rank", route_fraction).to_numpy(bool)
        result = merged.loc[expected].copy()
        if result.empty or not np.isfinite(result.loc[:, list(routed.ROUTER_OUTPUT_FIELDS)].to_numpy(float)).all():
            raise AssertionError(f"{token}: invalid top-50 router source")
        result["enhanced_base_routed"] = True
        # Keep the exact frozen full-base coordinate values.  Router ranks are
        # appended solely for the predeclared meta-router-input ablation.
        ordered = [name for name in source.columns if name != "enhanced_base_routed"]
        route_index = ordered.index("base_rank_ts") + 1
        ordered.insert(route_index, "enhanced_base_routed")
        ordered.extend(routed.ROUTER_OUTPUT_FIELDS)
        result = result.loc[:, ordered]
        destination = target / f"month={token}"
        destination.mkdir()
        result.to_parquet(destination / "scores_features.parquet", index=False, compression="zstd")
        audits.append({
            "month": token,
            "source_rows": int(len(source)),
            "routed_rows": int(len(result)),
            "route_fraction": route_fraction,
            "base_feature_complete_fraction": float(result.loc[:, list(fields)].notna().all(axis=1).mean()),
            "source_base_coordinate_hash": _hash_coordinates(source.loc[expected].copy()),
            "routed_base_coordinate_hash": _hash_coordinates(result),
            "router_outputs_persisted": True,
            "target_free": True,
        })
    audit = pd.DataFrame(audits)
    if audit["base_feature_complete_fraction"].lt(.90).any() or not audit["source_base_coordinate_hash"].eq(audit["routed_base_coordinate_hash"]).all():
        raise AssertionError("router view coverage or full-base coordinate preservation failed")
    audit.to_parquet(out / "fullbase_router50_materialization_audit.parquet", index=False, compression="zstd")
    _write_json_exclusive(out / "run_manifest.json", {
        "schema": "strict_r3_fullbase_router50_targetfree_v1",
        "scope": "offline research only; no model or live mutation",
        "source_root": str(source_root),
        "router_root": str(router_root),
        "route": "strict-OOF router_primary_rank exact timestamp-local top 50%",
        "months": [f"{m:%Y-%m}" for m in MONTHS],
        "base_contract": "frozen full-trained enhanced three-way coordinates preserved exactly",
        "router_outputs": list(routed.ROUTER_OUTPUT_FIELDS),
        "rows": int(audit["routed_rows"].sum()),
        "audit": audit.to_dict(orient="records"),
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--router-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(source_root=args.source_root, router_root=args.router_root, out=args.out, route_fraction=.50)


if __name__ == "__main__":
    main()
