from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from extreme_price_movements.packb_static_point_feature_loader import FrozenFeatureContract
from extreme_price_movements.stage_i_production_data_adapter import MonthlyReferencePartition
from extreme_price_movements.tail_base_input_contract import (
    PooledP90SpreadMap,
    TailBaseInputContractError,
    materialize_tail_base_input_contract,
)


def _contract() -> FrozenFeatureContract:
    # The injected PIT loader means only ordered feature semantics matter here.
    return FrozenFeatureContract(
        feature_columns=("f1", "f2"), candidate_universe_sha256="a" * 64,
        source_schema_sha256="b" * 64, raw_allowlist_sha256="c" * 64,
        generator_registry_sha256="d" * 64, store_scan_manifest_sha256="e" * 64,
        coverage_profile_sha256=None, min_exact_key_coverage=0.99,
        min_non_null_feature_coverage=0.90, max_feature_columns=None,
        coverage_admission_rejections=(), feature_contract_sha256="f" * 64,
    )


def _partition(path: Path) -> MonthlyReferencePartition:
    ts = pd.date_range("2025-01-01", periods=5, freq="h", tz="UTC")
    pd.DataFrame({
        "candidate_id": [f"c{i}" for i in range(5)], "__ts__": ts,
        "__symbol__": ["BTC_USD:USD", "ETH", "BTC_USD:USD", "ETH", "BTC_USD:USD"],
        "side_name": ["long", "short", "long", "short", "long"],
        "label_valid": [True, True, False, True, True],
        "exact_gross_bps": [0., 180., 220., 500., 110.],
        "exact_net_bps": [-100., 80., 120., 400., 10.],
        "label_available_ts": ts + pd.Timedelta(hours=13), "atr_bps": [100., 100., 100., 80., 40.],
        "t2_tp6_sl4_event": [0, 2, 2, 1, 2], "robust_clear_event_b25": [0, 1, 1, 1, 0],
        "robust_clear_soft_b25_t50": [0., .8, .9, 1., .2],
    }).to_parquet(path, index=False)
    return MonthlyReferencePartition(path, "2025-01", "common30_2025_2026")


def _loader(ledger: pd.DataFrame, fields: tuple[str, ...]) -> pd.DataFrame:
    out = ledger.loc[:, ["candidate_id", "__ts__", "__symbol__"]].copy()
    for idx, field in enumerate(fields):
        out[str(field)] = np.arange(len(out), dtype=np.float32) + idx + 1.0
    return out


def _transform(source: pd.DataFrame, _state: object) -> pd.DataFrame:
    assert "side" in source.columns
    return pd.DataFrame({"posterior": source["f1"].to_numpy(dtype=np.float32), "entropy": 0.5}, index=source.index)


def test_p90_map_rejects_average_spread_substitute() -> None:
    with pytest.raises(TailBaseInputContractError, match="average spread"):
        PooledP90SpreadMap.from_frame(pd.DataFrame({"__symbol__": ["BTC"], "average_spread_bps": [20.]}))


def test_streaming_contract_keeps_t1_t2_and_reports_t3_requirement(tmp_path: Path) -> None:
    partition = _partition(tmp_path / "labels.parquet")
    manifest = materialize_tail_base_input_contract(
        partitions=[partition], raw_feature_contract=_contract(),
        p90_spread_map=PooledP90SpreadMap.from_frame(pd.DataFrame({"__symbol__": ["BTC/USD:USD", "ETH"], "p90_spread_bps": [89., 95.]})),
        aegmm_state={"enabled": True, "feature_columns": ["side", "f1", "f2"]},
        output_dir=tmp_path / "out", batch_rows=2, pit_feature_loader=_loader,
        aegmm_transformer=_transform, side_raw_features={"long": ["f1"], "short": ["f2"]},
    )
    assert manifest["rows"] == 2  # valid BTC rows only; invalid ETH and invalid label excluded
    assert manifest["tail_targets"]["t3_first_touch_tbm"].startswith("not_materialised")
    assert manifest["side_raw_feature_contracts"] == {"long": ["f1"], "short": ["f2"]}
    output = pd.concat([pd.read_parquet(tmp_path / "out" / item) for item in manifest["parts"]], ignore_index=True)
    assert output["tail_target_net_grade_0_5"].tolist() == [0, 1]
    assert output["tail_target_atr_grade_0_5"].tolist() == [0, 2]
    assert {"f1", "f2", "aegmm_posterior", "aegmm_entropy", "label_available_ts"}.issubset(output.columns)
    audit = pd.read_parquet(tmp_path / "out" / "label_spread_audit.parquet")
    assert audit.iloc[0].source_label_invalid_rows == 1
    assert audit.iloc[0].p90_eligible_rows == 2
