#!/usr/bin/env python3
"""Materialise the extended long-only TP6/SL4 cluster base/meta inputs.

The canonical 2025 cluster replay starts at February because its base panel is
restricted to 2025.  This bridge reuses the strict fold-evaluation artifacts
from the extended meta-path materializer and joins the frozen R3 probabilities
from the original OOF source.  It therefore gives the cluster runner genuine
pre-2025 path/base rows without refitting or changing the base contract.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data_perp/artifacts/r3_tp6_sl4_meta_target_ablation_20260803_v1/r3_meta_target_oof_predictions.parquet"
PATHS = ROOT / "data_perp/artifacts/tp6_sl4_canonical_meta_paths_20260811_extended_v1"
BASE_OUT = ROOT / "data_perp/artifacts/tp6_sl4_extended_cluster_base_20260811_v1.parquet"
META_OUT = ROOT / "data_perp/artifacts/tp6_sl4_extended_cluster_meta_pool_20260811_v1.parquet"

LABEL_FIELDS = {"exact_net_bps", "exact_gross_bps", "label_valid", "label_available_ts"}
IDENTITY_FIELDS = {"candidate_id", "__ts__", "__symbol__", "side_name", "decision_ts"}
OUTPUT_FIELDS = {
    "r3_meta_p_adverse", "r3_meta_p_weak", "r3_meta_p_clear",
    "r3_meta_opportunity_score", "r3_meta2_p_adverse", "r3_meta2_p_weak",
    "r3_meta2_p_clear", "r3_meta2_opportunity_score",
}
DROP_TOKENS = ("target", "label", "gross_bps", "net_bps", "outcome", "event", "payoff", "pnl", "return")


def run(*, base_out: Path = BASE_OUT, meta_out: Path = META_OUT) -> tuple[Path, Path]:
    eval_files = sorted((PATHS / "fold_evaluations").glob("month=*.parquet"))
    if not eval_files:
        raise FileNotFoundError(f"no strict fold evaluations under {PATHS}")
    evaluations = pd.concat([pd.read_parquet(path) for path in eval_files], ignore_index=True)
    evaluations["__ts__"] = pd.to_datetime(evaluations["__ts__"], utc=True, errors="raise")
    evaluations = evaluations.loc[evaluations.side_name.astype(str).str.lower().eq("long")].copy()
    source_schema = pq.ParquetFile(SOURCE).schema.names
    source = pd.read_parquet(
        SOURCE,
        columns=["candidate_id", "__ts__", "side_name", "label_available_ts", "r3_meta_p_adverse", "r3_meta_p_weak", "r3_meta_p_clear"],
    )
    source["__ts__"] = pd.to_datetime(source["__ts__"], utc=True, errors="raise")
    source = source.loc[source.side_name.astype(str).str.lower().eq("long")].drop_duplicates("candidate_id", keep="last")
    base = evaluations.merge(
        source.loc[:, ["candidate_id", "__ts__", "label_available_ts", "r3_meta_p_adverse", "r3_meta_p_weak", "r3_meta_p_clear"]],
        on=["candidate_id", "__ts__"], how="left", validate="one_to_one", suffixes=("", "_source"),
    )
    if base[["r3_meta_p_adverse", "r3_meta_p_weak", "r3_meta_p_clear"]].isna().any().any():
        raise ValueError("historical strict path rows are missing frozen R3 probabilities")
    base["base_score"] = base["r3_meta_p_clear"].astype(float) - 0.5 * base["r3_meta_p_adverse"].astype(float)
    base["month"] = base["__ts__"].dt.strftime("%Y-%m")
    base["label_available_ts"] = pd.to_datetime(base["label_available_ts"], utc=True, errors="raise")
    base = base.loc[np.isfinite(base[["base_score", "base_expected_bps", "exact_net_bps", "exact_gross_bps"]].to_numpy(float)).all(axis=1)].copy()
    if base.candidate_id.duplicated().any():
        raise ValueError("extended base panel contains duplicate candidate IDs")
    base = base.loc[:, [
        "candidate_id", "__ts__", "label_available_ts", "side_name", "month",
        "exact_net_bps", "exact_gross_bps", "base_score", "r3_meta_p_clear",
        "r3_meta_p_adverse", "r3_meta_p_weak", "base_expected_bps", "meta_raw",
        "fold",
    ]].sort_values(["__ts__", "candidate_id"], kind="stable")
    base_out.parent.mkdir(parents=True, exist_ok=True)
    base.to_parquet(base_out, index=False, compression="zstd")

    wanted = set(base.candidate_id.astype(str))
    # Materialise every available numeric source field except labels, outcome
    # aliases, identity and frozen R3 output columns.  The cluster runner then
    # intersects this complete pool with config.py meta families train-only.
    source_schema = pq.ParquetFile(SOURCE).schema.names
    fields = []
    for name in source_schema:
        lower = str(name).lower()
        if name in IDENTITY_FIELDS or name in LABEL_FIELDS or name in OUTPUT_FIELDS:
            continue
        if any(token in lower for token in DROP_TOKENS):
            continue
        fields.append(name)
    source_cols = ["candidate_id", "__ts__", "side_name", *fields]
    pool = pd.read_parquet(SOURCE, columns=source_cols)
    pool = pool.loc[pool.candidate_id.astype(str).isin(wanted) & pool.side_name.astype(str).str.lower().eq("long")].copy()
    pool["__ts__"] = pd.to_datetime(pool["__ts__"], utc=True, errors="raise")
    pool = pool.drop_duplicates("candidate_id", keep="last")
    numeric = [f for f in fields if pd.api.types.is_numeric_dtype(pool[f])]
    pool = pool.loc[:, ["candidate_id", "__ts__", "side_name", *numeric]].sort_values(["__ts__", "candidate_id"], kind="stable")
    if len(pool) != len(base):
        raise ValueError(f"meta pool/base row mismatch: {len(pool)} vs {len(base)}")
    pool.to_parquet(meta_out, index=False, compression="zstd")
    manifest = {
        "schema": "tp6_sl4_extended_cluster_panel_v1",
        "source": str(SOURCE),
        "path_materializer": str(PATHS),
        "base_output": str(base_out),
        "meta_output": str(meta_out),
        "base_rows": int(len(base)),
        "meta_rows": int(len(pool)),
        "months": sorted(base.month.unique()),
        "meta_numeric_fields": int(len(numeric)),
        "base_contract": "strict TP6/SL4/H12 path-materializer folds; frozen R3 p_clear - 0.5 p_adverse; train-only bps anchor",
        "label_contract": "exact net/gross from authoritative source; labels are never meta features",
    }
    base_out.with_suffix(base_out.suffix + ".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    return base_out, meta_out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-out", type=Path, default=BASE_OUT)
    parser.add_argument("--meta-out", type=Path, default=META_OUT)
    args = parser.parse_args()
    run(base_out=args.base_out, meta_out=args.meta_out)
