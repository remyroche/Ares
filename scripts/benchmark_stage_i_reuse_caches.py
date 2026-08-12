#!/usr/bin/env python3
"""Bounded synthetic benchmark for Stage-I target-neutral cache reuse."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.stage_i_target_neutral_cache import (  # noqa: E402
    load_relief_geometry_cache,
    load_target_neutral_cache_for_contract,
    materialize_relief_geometry_cache,
    materialize_target_neutral_cache,
    relief_scores_from_geometry,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=2_000)
    parser.add_argument("--features", type=int, default=80)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rng = np.random.default_rng(42)
    signal = pd.date_range("2024-01-01", periods=args.rows, freq="h", tz="UTC")
    identity = pd.DataFrame({
        "candidate_id": [f"bench-{i}" for i in range(args.rows)],
        "__ts__": signal,
        "__symbol__": np.where(np.arange(args.rows) % 2, "BBB", "AAA"),
        "decision_ts": signal + pd.Timedelta(hours=1),
    })
    feature_names = [f"f{i}" for i in range(args.features)]
    feature_frame = pd.DataFrame(
        rng.normal(size=(args.rows, args.features)).astype(np.float32),
        columns=feature_names,
    )
    with tempfile.TemporaryDirectory(prefix="stage_i_cache_benchmark_") as temp:
        root = Path(temp)
        start = time.perf_counter()
        neutral = materialize_target_neutral_cache(
            root / "neutral", identity=identity, features=feature_frame,
            feature_names=feature_names, selector_manifest_sha256="a" * 64,
            selector_feature_contract_sha256="b" * 64,
            selector_features_sha256="c" * 64,
            correlation_rows=min(args.rows, 2_500),
        )
        cold_seconds = time.perf_counter() - start
        start = time.perf_counter()
        load_target_neutral_cache_for_contract(
            root / "neutral", identity=identity, feature_names=feature_names,
            selector_manifest_sha256="a" * 64,
            selector_feature_contract_sha256="b" * 64,
            selector_features_sha256="c" * 64,
        )
        hot_seconds = time.perf_counter() - start
        work_rows = np.arange(min(args.rows, 2_000), dtype=np.int32)
        start = time.perf_counter()
        relief = materialize_relief_geometry_cache(
            root / "relief", matrix=neutral.matrix,
            feature_names=feature_names, work_row_ids=work_rows,
            training_candidate_ids=identity.candidate_id,
            fold_lineage_sha256="f" * 64,
            random_state=17, anchor_max_rows=min(768, len(work_rows)),
            neighbor_candidate_rows=min(2_048, len(work_rows)),
        )
        relief_cold_seconds = time.perf_counter() - start
        labels = np.arange(len(work_rows), dtype=np.int16) % 3
        start = time.perf_counter()
        relief_scores_from_geometry(relief, labels, neighbors=8)
        score_seconds = time.perf_counter() - start
        start = time.perf_counter()
        hot_relief = load_relief_geometry_cache(root / "relief")
        relief_scores_from_geometry(hot_relief, 2 - labels, neighbors=8)
        relief_hot_target_seconds = time.perf_counter() - start
        payload = {
            "schema": "stage_i_reuse_cache_benchmark_v1",
            "rows": args.rows,
            "features": args.features,
            "neutral_cold_materialization_seconds": cold_seconds,
            "neutral_hot_load_seconds": hot_seconds,
            "neutral_hot_speedup_ratio": cold_seconds / max(hot_seconds, 1e-9),
            "relief_geometry_cold_seconds": relief_cold_seconds,
            "relief_target_score_seconds": score_seconds,
            "relief_hot_load_plus_new_target_score_seconds": relief_hot_target_seconds,
            "cache_bytes": sum(
                path.stat().st_size for path in root.rglob("*") if path.is_file()
            ),
            "interpretation": (
                "synthetic infrastructure benchmark only; no model/economic result"
            ),
        }
    text = json.dumps(payload, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
