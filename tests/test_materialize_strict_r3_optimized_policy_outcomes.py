from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "scripts" / "materialize_strict_r3_optimized_policy_outcomes.py"
SPEC = importlib.util.spec_from_file_location("strict_r3_optimized_policy_outcomes", PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_source_loader_preserves_target_free_identity_and_next_bar_entry(tmp_path: Path) -> None:
    source = tmp_path / "source.parquet"
    pd.DataFrame({
        "candidate_id": ["a", "b"],
        "__ts__": pd.to_datetime(["2024-07-01T00:00:00Z", "2024-07-01T01:00:00Z"]),
        "__decision_ts__": pd.to_datetime(["2024-07-01T01:00:00Z", "2024-07-01T02:00:00Z"]),
        "__symbol__": ["BTC/USD:USD", "ETH/USD:USD"],
        "side_name": ["long", "long"],
    }).to_parquet(source, index=False)
    result = MODULE._load_candidates(
        source,
        start=pd.Timestamp("2024-07-01T00:00:00Z"),
        end=pd.Timestamp("2024-07-02T00:00:00Z"),
    )
    assert set(result["candidate_id"]) == {"a", "b"}
    assert result["atr_1h"].isna().all()
    assert result["__decision_ts__"].eq(result["__ts__"] + pd.Timedelta(hours=1)).all()


def test_source_loader_rejects_non_next_bar_identity(tmp_path: Path) -> None:
    source = tmp_path / "source.parquet"
    pd.DataFrame({
        "candidate_id": ["a"],
        "__ts__": pd.to_datetime(["2024-07-01T00:00:00Z"]),
        "__decision_ts__": pd.to_datetime(["2024-07-01T02:00:00Z"]),
        "__symbol__": ["BTC/USD:USD"],
        "side_name": ["long"],
    }).to_parquet(source, index=False)
    with pytest.raises(ValueError, match=r"signal timestamp \+ one hour"):
        MODULE._load_candidates(
            source,
            start=pd.Timestamp("2024-07-01T00:00:00Z"),
            end=pd.Timestamp("2024-07-02T00:00:00Z"),
        )
