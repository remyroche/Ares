from __future__ import annotations

import json

import pandas as pd
import pytest

from extreme_price_movements.stage_i_target_adapter import (
    SOFT_SCALAR_S,
    StageITargetContract,
    file_sha256,
)
from scripts.run_stage_i_adapter_meta_feature_selection import _load_base


def _contract() -> StageITargetContract:
    return StageITargetContract(
        family=SOFT_SCALAR_S,
        layer="base",
        target_name="S__sl2_tp7",
        geometry="sl2_tp7",
        identity_sha256="1" * 64,
        target_sha256="2" * 64,
        economics_sha256="3" * 64,
        validity_sha256="4" * 64,
        weight_sha256="5" * 64,
        rows=2,
        target_columns=("target_value",),
    )


def _write_base(root, *, schema: str) -> StageITargetContract:
    side = root / "long"
    side.mkdir(parents=True)
    frame = pd.DataFrame({
        "candidate_id": ["a", "b"],
        "__ts__": pd.date_range("2024-01-01", periods=2, freq="h", tz="UTC"),
        "__symbol__": ["BTC", "ETH"],
        "side_name": ["long", "long"],
        "decision_ts": pd.date_range("2024-01-01 01:00", periods=2, freq="h", tz="UTC"),
        "label_available_ts": pd.date_range("2024-01-01 13:00", periods=2, freq="h", tz="UTC"),
        "exact_net_bps": [10.0, -20.0],
        "exact_gross_bps": [110.0, 80.0],
        "base_raw_score": [0.7, 0.3],
    })
    oof = side / "selector_base_oof.parquet"
    frame.to_parquet(oof, index=False)
    contract = _contract()
    manifest = {
        "schema": schema,
        "status": "complete",
        "side": "long",
        "target_contract_sha256": contract.sha256,
        "selector_sample_manifest_sha256": "a" * 64,
        "selector_base_oof_sha256": file_sha256(oof),
    }
    (side / "manifest.json").write_text(json.dumps(manifest))
    return contract


def test_target_specific_meta_requires_v2_base_manifest(tmp_path) -> None:
    contract = _write_base(tmp_path, schema="stage_i_base_feature_selection_v2")
    frame, manifest, _ = _load_base(
        tmp_path,
        side="long",
        selector_manifest_sha="a" * 64,
        winner_contract=contract,
    )
    assert len(frame) == 2
    assert manifest["schema"] == "stage_i_base_feature_selection_v2"


def test_target_specific_meta_rejects_legacy_r3_base_manifest(tmp_path) -> None:
    contract = _write_base(tmp_path, schema="stage_i_base_feature_selection_v1")
    with pytest.raises(ValueError, match="lineage drift"):
        _load_base(
            tmp_path,
            side="long",
            selector_manifest_sha="a" * 64,
            winner_contract=contract,
        )
