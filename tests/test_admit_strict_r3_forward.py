from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

from extreme_price_movements.strict_r3_ev_bridge import (
    fit_strict_r3_ev_bridge,
    persist_strict_r3_ev_bridge,
)


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "scripts" / "admit_strict_r3_forward.py"
SPEC = importlib.util.spec_from_file_location("admit_strict_r3_forward", PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _reserve_ledger(*, activation: pd.Timestamp) -> pd.DataFrame:
    decision = pd.date_range("2025-01-01", periods=160, freq="h", tz="UTC")
    score = pd.Series(range(160), dtype=float) / 159.0
    return pd.DataFrame({
        "candidate_id": [f"reserve-{index}" for index in range(160)],
        "__decision_ts__": decision,
        "side_name": "long",
        "final_score": score,
        "policy_net_bps": -100.0 + 400.0 * score,
        "policy_label_available_ts": decision + pd.Timedelta(hours=12),
        "policy_path_valid": True,
        "ev_score_family_id": "family",
        "geometry_bundle_sha256": "geometry",
        "conversion_bundle_sha256": "conversion",
        "upstream_bundle_sha256": "upstream",
        "stack_is_prequential": True,
        "calibration_activation_ts": activation,
    })


def _current(*, activation: pd.Timestamp) -> pd.DataFrame:
    return pd.DataFrame({
        "candidate_id": ["live"],
        "__decision_ts__": pd.to_datetime(["2025-02-01T00:00:00Z"], utc=True),
        "side_name": ["long"],
        "final_score": [0.9],
        "ev_score_family_id": ["family"],
        "geometry_bundle_sha256": ["geometry"],
        "conversion_bundle_sha256": ["conversion"],
        "upstream_bundle_sha256": ["upstream"],
        "calibration_activation_ts": [activation],
    })


def test_exact_reserve_resolver_requires_one_matching_lockstep_producer(tmp_path: Path) -> None:
    activation = pd.Timestamp("2025-02-01T00:00:00Z")
    reserve = _reserve_ledger(activation=activation)
    bundle = fit_strict_r3_ev_bridge(
        reserve,
        fit_cutoff=activation,
        producer_lineage={
            "conversion_bundle_sha256": "conversion",
            "upstream_bundle_sha256": "upstream",
        },
    )
    artifact = tmp_path / "reserve_map"
    manifest = persist_strict_r3_ev_bridge(bundle, artifact)
    index = pd.DataFrame([{
        "ev_score_family_id": "family",
        "geometry_bundle_sha256": "geometry",
        "conversion_bundle_sha256": "conversion",
        "upstream_bundle_sha256": "upstream",
        "calibration_activation_ts": activation,
        "producer_bundle_id": "exact-producer",
        "status": "fitted_immediate_exact_producer_calibration",
        "ev_bridge_bundle": str(artifact),
        "ev_bridge_bundle_sha256": manifest["bundle_sha256"],
        "reference_min_decision_ts": activation - pd.Timedelta(days=28),
        "reference_max_decision_ts": activation - pd.Timedelta(hours=1),
        "reference_max_label_available_ts": activation - pd.Timedelta(hours=1),
    }])
    index_path = tmp_path / "immediate_calibration_index.parquet"
    index.to_parquet(index_path, index=False)
    score_manifest = {"producer_topology": "exact_lockstep_shared_cutoff"}

    loaded, audit = MODULE._load_immediate_exact_reserve_calibrator(
        index_path=index_path,
        current=_current(activation=activation),
        score_manifest=score_manifest,
        decision_ts=pd.Timestamp("2025-02-01T01:00:00Z"),
    )

    assert loaded.producer_lineage == {
        "conversion_bundle_sha256": "conversion",
        "upstream_bundle_sha256": "upstream",
    }
    assert audit["immediate_calibration_producer_bundle_id"] == "exact-producer"


def test_exact_reserve_resolver_fails_closed_for_a_mismatched_producer(tmp_path: Path) -> None:
    activation = pd.Timestamp("2025-02-01T00:00:00Z")
    index = pd.DataFrame([{
        "ev_score_family_id": "family",
        "geometry_bundle_sha256": "geometry",
        "conversion_bundle_sha256": "conversion",
        "upstream_bundle_sha256": "other-upstream",
        "calibration_activation_ts": activation,
        "producer_bundle_id": "other-producer",
        "status": "fitted_immediate_exact_producer_calibration",
        "ev_bridge_bundle": str(tmp_path / "unused"),
        "ev_bridge_bundle_sha256": "unused",
        "reference_min_decision_ts": activation - pd.Timedelta(days=28),
        "reference_max_decision_ts": activation - pd.Timedelta(hours=1),
        "reference_max_label_available_ts": activation - pd.Timedelta(hours=1),
    }])
    index_path = tmp_path / "immediate_calibration_index.parquet"
    index.to_parquet(index_path, index=False)

    with pytest.raises(ValueError, match="no exact same-producer immediate calibration reserve"):
        MODULE._load_immediate_exact_reserve_calibrator(
            index_path=index_path,
            current=_current(activation=activation),
            score_manifest={"producer_topology": "exact_lockstep_shared_cutoff"},
            decision_ts=pd.Timestamp("2025-02-01T01:00:00Z"),
        )
