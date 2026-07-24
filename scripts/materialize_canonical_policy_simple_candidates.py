#!/usr/bin/env python3
"""Materialize fee-clean simple-policy candidates from the canonical chain."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


LABEL_COLUMNS = ("__ts__", "__symbol__", "side_name", "__barrier_pct__")
FORBIDDEN_COST_COLUMNS = {
    "ret_net",
    "net_return",
    "fees_bps",
    "round_trip_cost_floor",
}


def _load_barriers(labels_dir: Path, months: set[str]) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for path in sorted(labels_dir.glob("train_global_*_5_*.parquet")):
        month = "-".join(path.stem.rsplit("_", 2)[-2:])
        if month not in months:
            continue
        part = pd.read_parquet(path, columns=list(LABEL_COLUMNS))
        part["__ts__"] = pd.to_datetime(part["__ts__"], utc=True, errors="coerce")
        part["side_name"] = part["side_name"].astype(str).str.lower()
        parts.append(part)
    if not parts:
        raise FileNotFoundError(f"No matching label shards under {labels_dir}")
    out = pd.concat(parts, ignore_index=True)
    keys = ["__ts__", "__symbol__", "side_name"]
    duplicate_count = int(out.duplicated(keys).sum())
    if duplicate_count:
        raise ValueError(f"Label barrier table has {duplicate_count} duplicate keys")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chain-rows", type=Path, required=True)
    parser.add_argument("--labels-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = pd.read_parquet(args.chain_rows)
    required = {
        "__ts__",
        "__symbol__",
        "side_name",
        "archetype_policy_key",
        "threshold_basis_selected",
        "threshold_basis_rank_score",
    }
    missing = sorted(required.difference(rows.columns))
    if missing:
        raise ValueError(f"Canonical chain rows are missing {missing}")

    rows["__ts__"] = pd.to_datetime(rows["__ts__"], utc=True, errors="coerce")
    rows["side_name"] = rows["side_name"].astype(str).str.lower()
    rows = rows.loc[rows["threshold_basis_selected"].fillna(False)].copy()
    months = set(rows["__ts__"].dt.strftime("%Y-%m").dropna().unique())
    rows["__source_barrier_pct__"] = pd.to_numeric(
        rows.get("__barrier_pct__", pd.Series(np.nan, index=rows.index)),
        errors="coerce",
    )
    rows = rows.drop(columns=["__barrier_pct__"], errors="ignore")
    barriers = _load_barriers(args.labels_dir, months)
    keys = ["__ts__", "__symbol__", "side_name"]
    rows = rows.merge(barriers, on=keys, how="left", validate="one_to_one")
    rows["__barrier_pct__"] = pd.to_numeric(
        rows["__barrier_pct__"], errors="coerce"
    ).fillna(rows.pop("__source_barrier_pct__"))

    rows["timestamp"] = rows["__ts__"]
    rows["symbol"] = rows["__symbol__"].astype(str)
    rows["side"] = np.where(rows["side_name"].eq("short"), -1.0, 1.0).astype(
        np.float32
    )
    rows["strategy_id"] = rows["side_name"].map(
        {"long": "long_canonical_meta_policy", "short": "short_canonical_meta_policy"}
    )
    rows["rank_pct"] = pd.to_numeric(
        rows["threshold_basis_rank_score"], errors="coerce"
    ).astype(np.float32)
    rows["calibrated_score"] = pd.to_numeric(
        rows.get("expected_net_ev_after_1pct_mlp_direct"), errors="coerce"
    ).astype(np.float32)
    rows["barrier_pct"] = pd.to_numeric(
        rows["__barrier_pct__"], errors="coerce"
    ).astype(np.float32)
    rows["policy_archetype"] = rows["archetype_policy_key"].astype(str)
    rows["local_side_archetype"] = rows["policy_archetype"]
    rows["base_strategy_threshold"] = np.float32(0.0)

    output_columns = [
        "timestamp",
        "symbol",
        "side",
        "side_name",
        "strategy_id",
        "rank_pct",
        "calibrated_score",
        "barrier_pct",
        "base_strategy_threshold",
        "archetype_policy_key",
        "policy_archetype",
        "local_side_archetype",
        "policy_parent_rank",
        "rank_mlp_direct",
        "expected_ev_rank_score",
        "threshold_basis_multiplier",
        "threshold_basis_local_support",
        "threshold_basis_global_fallback",
    ]
    output_columns = [column for column in output_columns if column in rows.columns]
    out = rows[output_columns].dropna(
        subset=["timestamp", "symbol", "strategy_id", "rank_pct", "barrier_pct"]
    )
    if int(out.duplicated(["timestamp", "symbol", "strategy_id"]).sum()):
        raise ValueError("Canonical candidates are not unique per decision")
    leaked_cost_columns = sorted(FORBIDDEN_COST_COLUMNS.intersection(out.columns))
    if leaked_cost_columns:
        raise ValueError(f"Precomputed cost columns leaked into candidates: {leaked_cost_columns}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.output, index=False)
    coverage = float(rows["__barrier_pct__"].notna().mean()) if len(rows) else 0.0
    manifest = {
        "schema": "canonical_policy_simple_candidates_v1",
        "source": str(args.chain_rows),
        "labels": str(args.labels_dir),
        "output": str(args.output),
        "selected_source_rows": int(len(rows)),
        "output_rows": int(len(out)),
        "barrier_coverage": coverage,
        "barrier_source": "monthly labels with frozen-scoring-row fallback",
        "months": sorted(months),
        "strategies": out["strategy_id"].value_counts().to_dict(),
        "archetypes": out["policy_archetype"].value_counts().to_dict(),
        "fee_contract": "No precomputed net/fee fields; optimiser applies 1% once.",
    }
    args.output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
