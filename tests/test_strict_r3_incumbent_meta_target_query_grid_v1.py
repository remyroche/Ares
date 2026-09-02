from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "run_strict_r3_incumbent_meta_target_query_grid_v1.py"
SPEC = importlib.util.spec_from_file_location("incumbent_meta_grid", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_grid_has_all_requested_family_query_shapes() -> None:
    arms = MODULE.arms()
    families = {arm.family for arm in arms}
    assert families == {"magnitude", "under", "over", "state"}
    assert {arm.query for arm in arms if arm.family == "magnitude"} == {"base_band", "base_band_block28"}
    assert {arm.query for arm in arms if arm.family in {"under", "over"}} == {"timestamp"}
    assert {arm.query for arm in arms if arm.family == "state"} == {"base_band", "timestamp", "base_band_block28"}


def test_query_ids_are_target_free_and_timestamp_local() -> None:
    frame = pd.DataFrame({
        "candidate_id": ["a", "b", "c"],
        "__decision_ts__": pd.to_datetime(["2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z", "2026-01-02T00:00:00Z"]),
        "inc_base_rank_ts": [.99, .80, .99],
    })
    timestamp = MODULE._query_ids(frame, "timestamp")
    assert timestamp[0] == timestamp[1]
    assert timestamp[0] != timestamp[2]
    bands = MODULE._query_ids(frame, "base_band")
    assert bands[0] == bands[2]


def test_base_geometry_uses_persisted_canonical_route() -> None:
    frame = pd.DataFrame({
        "candidate_id": ["a", "b"],
        "__decision_ts__": pd.to_datetime(["2026-01-01T00:00:00Z"] * 2),
        "enhanced_base_bps": [10.0, 9.0],
        "efficiency_bps": [11.0, 10.0],
        "timing_bps": [9.0, 8.0],
        "enhanced_base_routed": [False, True],
        "base_component_std": [1.0, 1.0],
    })
    result = MODULE._add_base_geometry(frame)
    # The score ordering would select a, but the source route is the
    # canonical inference contract and must be retained without rebuilding.
    assert result.inc_routed.tolist() == [False, True]


def test_custom_arm_specs_validate_target_bin_contract(tmp_path: Path) -> None:
    path = tmp_path / "arms.json"
    path.write_text(
        '{"arms":[{"name":"m","family":"magnitude","scale":"sqrt_atr",'
        '"query":"base_band_block28","classes":5,"gain_schedule":"high","truncation_level":10},'
        '{"name":"s","family":"state","scale":"atr","query":"timestamp",'
        '"state_edges":[-1.0,-0.2,0.2,1.0]}]}'
    )
    arms = MODULE._custom_arms(path)
    assert [(arm.name, arm.classes, arm.state_edges, arm.gain_schedule, arm.truncation_level) for arm in arms] == [
        ("m", 5, None, "high", 10), ("s", 7, (-1.0, -0.2, 0.2, 1.0), "medium", None),
    ]


def test_gain_schedules_are_monotone_and_label_complete() -> None:
    labels = np.array([0, 1, 2, 3, 4, 5, 6], dtype=np.int16)
    for name in ("small", "medium", "high"):
        gain = MODULE._gain(labels, name)
        assert len(gain) == 7
        assert gain == sorted(gain)
    with pytest.raises(ValueError):
        MODULE._gain(labels, "not-a-schedule")


def test_model_receives_declared_optional_truncation() -> None:
    model = MODULE._model(seed=1729, gain=[0.0, 1.0, 2.0], truncation_level=5)
    assert model.get_params()["lambdarank_truncation_level"] == 5
    default = MODULE._model(seed=1729, gain=[0.0, 1.0, 2.0], truncation_level=None)
    assert default.get_params().get("lambdarank_truncation_level") is None


def test_full_feature_roots_require_unique_month_ownership(tmp_path: Path) -> None:
    root_a, root_b = tmp_path / "a", tmp_path / "b"
    for root in (root_a, root_b):
        (root / "month=2026-01").mkdir(parents=True)
        (root / "month=2026-01" / "causal_feature_universe.parquet").touch()
    with pytest.raises(AssertionError):
        MODULE._full_feature_path((root_a, root_b), pd.Timestamp("2026-01-01", tz="UTC"))
    assert MODULE._full_feature_path((root_a,), pd.Timestamp("2026-01-01", tz="UTC")).name == "causal_feature_universe.parquet"


def test_over_target_is_inverted_after_prediction_direction() -> None:
    values = np.array([-200.0, -50.0, 50.0], dtype=np.float32)
    atr = np.array([100.0, 100.0, 100.0], dtype=np.float32)
    assert np.allclose(MODULE._normalised_residual(values, atr, "bps"), values)
    assert np.allclose(MODULE._normalised_residual(values, atr, "atr"), values / 100.0)
    with pytest.raises(ValueError):
        MODULE._query_ids(pd.DataFrame({"inc_base_rank_ts": [.9]}), "unknown")
