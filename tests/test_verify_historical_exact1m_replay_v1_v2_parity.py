from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.verify_historical_exact1m_replay_v1_v2_parity import (
    ParityError,
    compare_parquet_files,
    parser,
    sha256,
    verify_aggregate_seal,
)


def _path(high: float = 101.0) -> str:
    return json.dumps(
        {
            "timestamp": pd.date_range("2024-01-01T01:00:00Z", periods=720, freq="min").astype("int64").tolist(),
            "open": np.full(720, 100.0).tolist(),
            "high": np.full(720, high).tolist(),
            "low": np.full(720, 99.0).tolist(),
            "close": np.full(720, 100.5).tolist(),
        }
    )


def _frame(path: str, value: float = 0.1, category: str = "timeout") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "__ts__": [pd.Timestamp("2024-01-01T00:00:00Z")],
            "__symbol__": ["BTC/USD:USD"],
            "side_name": ["long"],
            "candidate_id": ["source-native"],
            "execution_future_path": [path],
            "execution_net_ev_12h": [value],
            "execution_exit_reason": [category],
            "__meaningful_mfe_reached_12h__": [1],
        }
    )


def test_comparator_requires_exact_identity_categoricals_and_decoded_path(tmp_path: Path) -> None:
    left, right = tmp_path / "left.parquet", tmp_path / "right.parquet"
    _frame(_path()).to_parquet(left, index=False)
    _frame(_path()).to_parquet(right, index=False)
    report = compare_parquet_files(left, right, float_atol=1e-10, path_atol=0.0)
    assert report["pass"]
    _frame(_path(high=101.001)).to_parquet(right, index=False)
    report = compare_parquet_files(left, right, float_atol=1e-10, path_atol=0.0)
    assert not report["pass"]
    assert report["columns"]["execution_future_path"]["mismatch_rows"] == 1
    _frame(_path(), category="trailing").to_parquet(right, index=False)
    report = compare_parquet_files(left, right, float_atol=1e-10, path_atol=0.0)
    assert not report["pass"]
    assert report["columns"]["execution_exit_reason"]["kind"] == "categorical_exact"


def test_float_tolerance_is_explicit_and_identity_is_fail_closed(tmp_path: Path) -> None:
    left, right = tmp_path / "left.parquet", tmp_path / "right.parquet"
    _frame(_path(), value=0.1).to_parquet(left, index=False)
    _frame(_path(), value=0.10000000005).to_parquet(right, index=False)
    assert compare_parquet_files(left, right, float_atol=1e-10, path_atol=0.0)["pass"]
    assert not compare_parquet_files(left, right, float_atol=1e-12, path_atol=0.0)["pass"]
    changed = _frame(_path())
    changed.loc[0, "candidate_id"] = "different"
    changed.to_parquet(right, index=False)
    with pytest.raises(ParityError, match="four-key"):
        compare_parquet_files(left, right, float_atol=1e-10, path_atol=0.0)


def test_aggregate_seal_requires_four_partitions_and_stage_binding(tmp_path: Path) -> None:
    stage = tmp_path / "stage.json"
    request = tmp_path / "requests.parquet"
    pd.DataFrame({"x": [1]}).to_parquet(request, index=False)
    stage.write_text(json.dumps({"outputs": {"download_candidates": {"sha256": sha256(request), "rows": 1}}}))
    manifest = {
        "schema": "failure_2024_exact1m_download_verification_v1",
        "status": "SEALED_COMPLETE",
        "partition_count": 4,
        "partitions": {str(index): {} for index in range(4)},
        "required_minutes": 4,
        "covered_minutes": 4,
        "coverage_fraction": 1.0,
        "failed_symbols": 0,
        "incomplete_symbols": 0,
        "request_manifest": {"path": str(stage), "sha256": sha256(stage)},
        "candidate_request": {"sha256": sha256(request), "rows": 1},
    }
    seal = tmp_path / "manifest.json"
    seal.write_text(json.dumps(manifest))
    (tmp_path / "manifest.sha256").write_text(f"{sha256(seal)}  manifest.json\n")
    assert verify_aggregate_seal(seal, coverage_manifest={"stage_manifest": {"sha256": sha256(stage)}})["pass"]
    manifest["partition_count"] = 3
    seal.write_text(json.dumps(manifest))
    (tmp_path / "manifest.sha256").write_text(f"{sha256(seal)}  manifest.json\n")
    with pytest.raises(ParityError, match="incomplete"):
        verify_aggregate_seal(seal, coverage_manifest={"stage_manifest": {"sha256": sha256(stage)}})


def test_defaults_keep_v1_and_v2_coverage_bindings_distinct() -> None:
    options = parser().parse_args([])
    assert options.v1_coverage_manifest != options.v2_coverage_manifest
    assert options.v1_coverage_manifest.name == "manifest.json"
    assert options.v2_coverage_manifest.name == "manifest.json"
    assert "candidate_coverage_20260730_v1" in str(options.v1_coverage_manifest)
    assert "candidate_coverage_20260730_v2" in str(options.v2_coverage_manifest)
