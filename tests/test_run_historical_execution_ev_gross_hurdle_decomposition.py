import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "gross_hurdle_decomposition",
    ROOT / "scripts" / "run_historical_execution_ev_gross_hurdle_decomposition.py",
)
assert SPEC and SPEC.loader
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


def test_purged_training_excludes_label_unresolved_at_cutoff():
    cutoff = pd.Timestamp("2025-03-10", tz="UTC")
    frame = pd.DataFrame(
        {
            "__ts__": pd.to_datetime(["2025-03-08T00:00Z", "2025-03-08T01:00Z"]),
            "execution_label_end_utc": pd.to_datetime(["2025-03-09T00:00Z", "2025-03-11T00:00Z"]),
        }
    )
    result = MOD._purged_before(frame, cutoff)
    assert len(result) == 1
    assert (result.execution_label_end_utc < cutoff).all()


def test_soft_hurdle_label_is_monotonic_around_realized_cost_margin():
    label = MOD._soft_label(np.array([-0.01, 0.0, 0.01]), 0.005)
    assert label[0] < 0.5
    assert label[1] == 0.5
    assert label[2] > 0.5


def test_turnover_uses_selected_assets_not_timestamp_unique_candidates():
    rows = pd.DataFrame(
        {
            "__ts__": pd.to_datetime(["2025-04-01T00:00Z", "2025-04-01T01:00Z"]),
            "__symbol__": ["BTC", "BTC"],
        }
    )
    result = MOD._asset_turnover(rows, pd.Series([True, True]), "h")
    assert result["selected_asset_turnover"] == 0.0
    assert result["selected_asset_jaccard_mean"] == 1.0


def _mapping_rows() -> pd.DataFrame:
    rows = pd.DataFrame(
        {
            "__ts__": pd.date_range("2025-03-19", periods=360, freq="h", tz="UTC"),
            "execution_label_end_utc": pd.date_range("2025-03-19 12:00", periods=360, freq="h", tz="UTC"),
            "common_unit_score": np.tile([0.0, 1.0], 180),
            "execution_net_ev_12h": np.zeros(360),
        }
    )
    return rows


def test_online_mapping_excludes_outcome_resolving_after_current_decision():
    inner = _mapping_rows()
    decision = pd.Timestamp("2025-04-02", tz="UTC")
    current = pd.DataFrame({"__ts__": [decision], "execution_label_end_utc": [decision + pd.Timedelta(hours=12)], "common_unit_score": [1.0], "execution_net_ev_12h": [0.0]})
    poison = pd.DataFrame({"__ts__": [decision - pd.Timedelta(days=1)], "execution_label_end_utc": [decision + pd.Timedelta(hours=2)], "common_unit_score": [1.0], "execution_net_ev_12h": [10.0]})
    baseline = MOD._online_causal_21d_map(inner, current)[0]
    with_poison = MOD._online_causal_21d_map(inner, pd.concat([poison, current], ignore_index=True))[1]
    assert with_poison == baseline


def test_online_mapping_admits_earlier_april_outcome_only_after_resolution():
    inner = _mapping_rows()
    early = pd.DataFrame({"__ts__": [pd.Timestamp("2025-04-01", tz="UTC")], "execution_label_end_utc": [pd.Timestamp("2025-04-01 12:00", tz="UTC")], "common_unit_score": [1.0], "execution_net_ev_12h": [0.10]})
    later = pd.DataFrame({"__ts__": [pd.Timestamp("2025-04-02", tz="UTC")], "execution_label_end_utc": [pd.Timestamp("2025-04-02 12:00", tz="UTC")], "common_unit_score": [1.0], "execution_net_ev_12h": [0.0]})
    without_early = MOD._online_causal_21d_map(inner, later)[0]
    with_early = MOD._online_causal_21d_map(inner, pd.concat([early, later], ignore_index=True))[1]
    assert with_early > without_early


def test_raw_economic_unit_selection_can_differ_from_side_z_common_unit_selection():
    rows = pd.DataFrame(
        {
            "candidate_id": [f"short-{i}" for i in range(5)] + [f"long-{i}" for i in range(5)],
            "side_name": ["short"] * 5 + ["long"] * 5,
            "__symbol__": [f"s{i}" for i in range(10)],
            "__ts__": pd.date_range("2025-04-01", periods=10, freq="h", tz="UTC"),
            "execution_gross_ev_12h": 0.02,
            "execution_cost_return": 0.01,
            "execution_net_ev_12h": 0.01,
        }
    )
    raw = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 10.0, 9.0, 8.0, 7.0, 6.0])
    common = np.array([5.0, 4.0, 3.0, 2.0, 1.0, 1.0, 0.5, 0.0, -0.5, -1.0])
    raw_capacity = MOD._economics(rows, raw)["side_capacity"]
    common_capacity = MOD._economics(rows, common)["side_capacity"]
    assert raw_capacity == [{"side": "long", "rows": 1}]
    assert common_capacity == [{"side": "short", "rows": 1}]
