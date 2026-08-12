#!/usr/bin/env python3
"""Materialise the complete available causal meta pool for canonical 2025 rows.

This is a schema handoff only: no selection, scaling, labels, or model fitting
is performed here.  It preserves every numeric field present in the source
TP6/SL4 panel after removing identity, labels, predictions, and provenance
metadata.  The cluster runner performs the train-only CMI reduction later.
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
CANONICAL = ROOT / "data_perp/artifacts/tp6_sl4_downstream_retrain_2025_20260807_v1/canonical_cluster_input_2025.parquet"
DEFAULT_OUT = ROOT / "data_perp/artifacts/tp6_sl4_canonical_meta_pool_20260808_v1.parquet"

IDENTITY = {"candidate_id", "__ts__", "__symbol__", "side_name", "decision_ts", "label_available_ts"}
METADATA = {"source_month", "population_segment", "selector_month", "selector_economic_bin", "fold"}
OUTPUTS = {
    "r3_meta_p_adverse", "r3_meta_p_weak", "r3_meta_p_clear", "r3_meta_opportunity_score",
    "r3_meta2_p_adverse", "r3_meta2_p_weak", "r3_meta2_p_clear", "r3_meta2_opportunity_score",
    "base_score", "base_raw", "base_expected_bps", "base_anchor_bps", "base_rank",
    "consensus_rank", "residual_rank", "consensus_only", "residual_only",
    "r3_class", "robust_clear_soft_b25_t50", "t2_tp6_sl4_event",
}
LABEL_TOKENS = ("label", "target", "gross_bps", "net_bps", "outcome", "event", "payoff", "pnl", "return")


def run(out: Path = DEFAULT_OUT) -> Path:
    canonical_ids = pd.read_parquet(CANONICAL, columns=["candidate_id"])["candidate_id"].astype(str)
    wanted = set(canonical_ids)
    source_schema = pq.ParquetFile(SOURCE).schema.names
    fields: list[str] = []
    for name in source_schema:
        if name in IDENTITY or name in METADATA or name in OUTPUTS:
            continue
        lower = name.lower()
        if any(token in lower for token in LABEL_TOKENS):
            continue
        fields.append(name)
    cols = ["candidate_id", "__ts__", "side_name", *fields]
    source = pd.read_parquet(SOURCE, columns=cols)
    source = source.loc[source.candidate_id.astype(str).isin(wanted) & source.side_name.astype(str).str.lower().eq("long")].copy()
    source["__ts__"] = pd.to_datetime(source["__ts__"], utc=True)
    source = source.drop_duplicates("candidate_id", keep="last")
    numeric = [f for f in fields if pd.api.types.is_numeric_dtype(source[f])]
    source = source.loc[:, ["candidate_id", "__ts__", "side_name", *numeric]].copy()
    source.to_parquet(out, index=False, compression="zstd")
    manifest = {
        "schema": "tp6_sl4_canonical_meta_pool_v1",
        "source": str(SOURCE),
        "canonical_population": str(CANONICAL),
        "side": "long",
        "rows": int(len(source)),
        "canonical_rows": int(len(canonical_ids)),
        "coverage": float(len(source) / max(len(canonical_ids), 1)),
        "field_count": int(len(numeric)),
        "fields": numeric,
        "selection": "none; cluster runner performs train-only CMI selection",
        "excluded": {"identity": sorted(IDENTITY), "metadata": sorted(METADATA), "outputs": sorted(OUTPUTS), "label_tokens": list(LABEL_TOKENS)},
        "finite_fraction": {f: float(np.isfinite(pd.to_numeric(source[f], errors="coerce")).mean()) for f in numeric},
    }
    out.with_suffix(out.suffix + ".manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    print(json.dumps(manifest, indent=2))
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    run(args.out)
