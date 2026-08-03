from __future__ import annotations

import json

import numpy as np
import pandas as pd

from scripts.materialize_febapr2025_path_head_context import (
    IDENTITY,
    SCHEMA,
    _finalize_index,
    _identity_hash,
    _load_raw_features,
    _shard_paths,
)


def test_loader_output_is_reordered_to_the_exact_input_ledger(monkeypatch, tmp_path) -> None:
    ledger = pd.DataFrame({"candidate_id": ["a", "b"], "__symbol__": ["A", "B"], "__ts__": pd.to_datetime(["2025-02-01T00:00:00Z", "2025-02-01T01:00:00Z"])})

    class Batch:
        def __init__(self) -> None:
            self.features = pd.DataFrame({"f": [2.0, 1.0]})
            self.ledger_row_positions = np.array([1, 0])
            self.matched_exact_keys = np.array([True, True])

    monkeypatch.setattr("scripts.materialize_febapr2025_path_head_context.iter_point_in_time_feature_batches", lambda *args, **kwargs: iter([Batch()]))
    raw, matched = _load_raw_features(ledger, feature_store=tmp_path, feature_contract={})
    assert matched == 2
    assert raw["f"].tolist() == [1.0, 2.0]


def test_final_index_reads_only_identity_columns_and_validates_partition_coverage(tmp_path, monkeypatch) -> None:
    population = pd.DataFrame(
        {
            "candidate_id": ["A|t|long", "A|t|short", "B|t|long", "B|t|short"],
            "side_name": ["long", "short", "long", "short"],
            "__symbol__": ["A", "A", "B", "B"],
            "__ts__": pd.to_datetime(["2025-02-01T00:00:00Z"] * 4),
        }
    )
    symbols = ["A", "B"]
    for symbol in symbols:
        data, manifest = _shard_paths(tmp_path, symbol)
        shard = population.loc[population["__symbol__"].eq(symbol)].copy()
        shard["base_oof_score"] = 0.5
        shard["base_rank_timestamp_side"] = 1
        shard["base_group_rows"] = 2
        shard["base_rank_pct_timestamp_side"] = 0.5
        shard["__decision_ts__"] = shard["__ts__"] + pd.Timedelta(hours=1)
        shard["hour_sin"] = 0.0
        shard["hour_cos"] = 1.0
        shard["raw_feature"] = 1.0
        data.parent.mkdir(parents=True, exist_ok=True)
        shard.to_parquet(data, index=False)
        manifest.write_text(json.dumps({"schema": SCHEMA, "input_identity_sha256": _identity_hash(shard), "rows": len(shard), "output_sha256": __import__("hashlib").sha256(data.read_bytes()).hexdigest(), "unique_symbol_signal_keys": 1, "exact_key_rows": 1}))
    seen: list[tuple[str, ...] | None] = []
    real_read = pd.read_parquet

    def read_identity_only(path, *args, **kwargs):
        seen.append(tuple(kwargs.get("columns", ())) if kwargs.get("columns") else None)
        return real_read(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", read_identity_only)
    index_path, coverage = _finalize_index(output_dir=tmp_path, population=population, symbols=symbols, raw_features=["raw_feature"])
    assert index_path.is_file()
    assert coverage["rows"] == 4
    assert all(columns == IDENTITY for columns in seen)
