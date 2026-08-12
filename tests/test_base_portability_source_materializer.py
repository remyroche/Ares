from __future__ import annotations

import json

import pandas as pd
import pytest

from extreme_price_movements.base_portability_source_materializer import (
    BasePortabilitySourceContract,
    BasePortabilitySourceError,
    BasePortabilitySourceMaterializer,
    TransportScope,
    parse_f0_feature_lineage,
)


def _write_part(root, name: str, frame: pd.DataFrame) -> None:
    (root / "parts").mkdir(parents=True, exist_ok=True)
    frame.to_parquet(root / "parts" / name, index=False)


def _lineage(path) -> None:
    entries = []
    for run in ("transport_a", "transport_b"):
        for side, features in (("long", ["long_a", "shared"]), ("short", ["short_b", "shared"])):
            entries.append({
                "arm": "F0_current_frozen", "run": run, "side": side, "features": features,
                "target": "R3 robust-clear/adverse/weak; clear requires cost +25bps before lower barrier",
            })
    path.write_text(json.dumps(entries), encoding="utf-8")


def _contract(tmp_path):
    panel, winner, robust = (tmp_path / name for name in ("panel", "winner", "robust"))
    ts = pd.Timestamp("2024-01-10T00:00:00Z")
    candidates = pd.DataFrame({
        "candidate_id": ["BTC|a|long", "ETH|a|short", "SOL|bad|long", "OUT|x|long"],
        "__ts__": [ts, ts, ts, ts + pd.Timedelta(days=40)],
        "side_name": ["long", "short", "long", "long"],
        "long_a": [1.0, 2.0, 3.0, 4.0], "short_b": [4.0, 3.0, 2.0, 1.0], "shared": [3.0, 3.0, 3.0, 3.0],
    })
    _write_part(panel, "one.parquet", candidates)
    _write_part(winner, "one.parquet", pd.DataFrame({
        "candidate_id": candidates.candidate_id,
        "t4_tp6_sl4_gross_bps": [150.0, 50.0, 100.0, 120.0],
        "t4_tp6_sl4_net_bps": [50.0, -50.0, 0.0, 20.0],
        "__label_available_at__": candidates.__ts__ + pd.Timedelta(hours=13),
    }))
    _write_part(robust, "one.parquet", pd.DataFrame({
        "candidate_id": candidates.candidate_id, "label_valid": [True, True, False, True],
        "lower_touch_minute": [-1.0, 1.0, -1.0, -1.0], "robust_clear_event_b25": [1, 0, 1, 0],
    }))
    lineage = tmp_path / "base_feature_arm_lineage.json"
    _lineage(lineage)
    return BasePortabilitySourceContract(panel=panel, winner=winner, robust=robust, lineage=lineage)


def test_loads_union_but_declares_exact_side_run_contract_and_r3_semantics(tmp_path) -> None:
    materializer = BasePortabilitySourceMaterializer(_contract(tmp_path))
    result = materializer.load(scope=TransportScope("transport_a", "2024-01-01", "2024-02-01"), side="long")
    assert result.selected_features == ("long_a", "shared")
    assert result.union_features == ("long_a", "shared", "short_b")
    assert set(result.union_features).issubset(result.frame.columns)
    assert result.frame.candidate_id.tolist() == ["BTC|a|long"]
    assert result.frame.r3_class.tolist() == [2]
    assert result.frame.label_available_ts.iloc[0] - result.frame.decision_ts.iloc[0] == pd.Timedelta(hours=13)
    assert result.frame.asset.astype(str).tolist() == ["BTC"]


def test_r3_adverse_precedence_and_end_exclusive_scope(tmp_path) -> None:
    materializer = BasePortabilitySourceMaterializer(_contract(tmp_path))
    result = materializer.load(scope=TransportScope("transport_b", "2024-01-01", "2024-02-01"), side="short")
    assert result.frame.r3_class.tolist() == [0]
    with pytest.raises(BasePortabilitySourceError, match="no valid source rows"):
        materializer.load(scope=TransportScope("transport_b", "2024-02-01", "2024-02-02"), side="long")


def test_rejects_missing_label_identity_and_bad_label_availability(tmp_path) -> None:
    contract = _contract(tmp_path)
    winner_path = contract.winner / "parts" / "one.parquet"
    winner = pd.read_parquet(winner_path)
    winner.loc[0, "__label_available_at__"] = pd.Timestamp("2024-01-10T12:00:00Z")
    winner.to_parquet(winner_path, index=False)
    with pytest.raises(BasePortabilitySourceError, match="resolve exactly 13h"):
        BasePortabilitySourceMaterializer(contract).load(
            scope=TransportScope("transport_a", "2024-01-01", "2024-02-01"), side="long"
        )


def test_rejects_incomplete_or_ambiguous_f0_lineage(tmp_path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps([{
        "arm": "F0_current_frozen", "run": "x", "side": "long", "features": ["x"],
        "target": "R3 robust-clear/adverse/weak; clear requires cost +25bps before lower barrier",
    }]), encoding="utf-8")
    with pytest.raises(BasePortabilitySourceError, match="lacks canonical side"):
        parse_f0_feature_lineage(bad)
