from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import pandas as pd
import pytest

from extreme_price_movements.stage_i_selector_timestamp_upgrade import (
    StageISelectorTimestampUpgradeError,
    upgrade_stage_i_selector_timestamp_contract,
)


def _sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _source(tmp_path: Path) -> Path:
    source = tmp_path / "source"
    source.mkdir()
    signal = pd.date_range("2023-01-01", periods=4, freq="h", tz="UTC")
    ledger = pd.DataFrame(
        {
            "candidate_id": range(4),
            "__ts__": signal,
            "__symbol__": ["BTC"] * 4,
            "label_available_ts": signal + pd.Timedelta(hours=13),
        }
    )
    ledger.to_parquet(source / "selector_ledger.parquet", index=False)
    pd.DataFrame({"feature": [1.0, 2.0, 3.0, 4.0]}).to_parquet(
        source / "selector_features.parquet", index=False
    )
    (source / "selector_feature_contract.json").write_text("{}\n")
    pd.DataFrame({"x": [1]}).to_parquet(
        source / "selector_exact_feature_coverage_audit.parquet", index=False
    )
    pd.DataFrame({"x": [1]}).to_parquet(
        source / "selector_exact_feature_month_side_coverage.parquet", index=False
    )
    pd.DataFrame({"x": [1]}).to_parquet(
        source / "population_summary.parquet", index=False
    )
    (source / "manifest.json").write_text(
        json.dumps(
            {
                "schema": "stage_i_selector_sample_v1",
                "status": "complete",
                "rows": 4,
                "feature_columns": 1,
            }
        )
    )
    return source


def test_upgrade_preserves_source_and_feature_payloads(tmp_path: Path) -> None:
    source = _source(tmp_path)
    source_hashes = {
        path.name: _sha(path) for path in source.iterdir() if path.is_file()
    }
    destination = tmp_path / "destination"
    manifest = upgrade_stage_i_selector_timestamp_contract(source, destination)
    upgraded = pd.read_parquet(destination / "selector_ledger.parquet")
    assert upgraded["decision_ts"].equals(
        upgraded["__ts__"] + pd.Timedelta(hours=1)
    )
    assert manifest["timestamp_contract"]["decision_to_label_available_hours"] == 12
    assert manifest["timestamp_upgrade"]["source_artifact_preserved"] is True
    for name in (
        "selector_features.parquet",
        "selector_feature_contract.json",
        "selector_exact_feature_coverage_audit.parquet",
        "selector_exact_feature_month_side_coverage.parquet",
        "population_summary.parquet",
    ):
        assert _sha(source / name) == _sha(destination / name)
    assert source_hashes == {
        path.name: _sha(path) for path in source.iterdir() if path.is_file()
    }
    assert upgrade_stage_i_selector_timestamp_contract(
        source, destination, resume=True
    )["timestamp_contract"] == manifest["timestamp_contract"]


def test_upgrade_fails_closed_on_incorrect_label_offset(tmp_path: Path) -> None:
    source = _source(tmp_path)
    ledger = pd.read_parquet(source / "selector_ledger.parquet")
    ledger.loc[0, "label_available_ts"] += pd.Timedelta(hours=1)
    ledger.to_parquet(source / "selector_ledger.parquet", index=False)
    with pytest.raises(ValueError, match=r"decision_ts \+ 12h"):
        upgrade_stage_i_selector_timestamp_contract(source, tmp_path / "bad")


def test_upgrade_refuses_in_place_mutation(tmp_path: Path) -> None:
    source = _source(tmp_path)
    with pytest.raises(StageISelectorTimestampUpgradeError, match="must differ"):
        upgrade_stage_i_selector_timestamp_contract(source, source)
