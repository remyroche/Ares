from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "historical_execution_ev_add_drop_gate",
    ROOT / "scripts" / "run_historical_execution_ev_add_drop_gate.py",
)
assert SPEC and SPEC.loader
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


def test_execution_target_requires_exact_gross_minus_cost() -> None:
    valid = pd.DataFrame(
        {
            "execution_gross_ev_12h": [0.02, -0.01],
            "execution_cost_return": [0.01, 0.01],
            "execution_net_ev_12h": [0.01, -0.02],
        }
    )
    MOD.validate_execution_target(valid)
    invalid = valid.copy()
    invalid.loc[0, "execution_net_ev_12h"] += 1e-5
    with pytest.raises(ValueError, match="gross EV minus realized cost"):
        MOD.validate_execution_target(invalid)


def _inner_frame() -> pd.DataFrame:
    ts = pd.date_range("2025-03-01", "2025-03-30", freq="12h", tz="UTC")
    gross = np.linspace(-0.02, 0.03, len(ts))
    frame = pd.DataFrame(
        {
            "candidate_id": [f"c{i}" for i in range(len(ts))],
            "side_name": "long",
            "__symbol__": "BTCUSDT",
            "__ts__": ts,
            "execution_label_end_utc": ts + pd.Timedelta(hours=12),
            "execution_gross_ev_12h": gross,
            "execution_cost_return": 0.01,
            "execution_net_ev_12h": gross - 0.01,
            "feature": np.linspace(-1.0, 1.0, len(ts)),
        }
    )
    return frame


def test_inner_oof_training_purges_unresolved_labels() -> None:
    base = _inner_frame()
    poison = base.iloc[[0]].copy()
    poison["candidate_id"] = "unresolved-poison"
    poison["__ts__"] = pd.Timestamp("2025-03-02", tz="UTC")
    poison["execution_label_end_utc"] = pd.Timestamp("2025-03-15", tz="UTC")
    poison["execution_gross_ev_12h"] = 100.0
    poison["execution_net_ev_12h"] = 99.99
    poison["feature"] = 100.0

    without_poison = MOD.inner_scores(base, ["feature"], 1.0, ["feature"])
    with_poison = MOD.inner_scores(
        pd.concat([base, poison], ignore_index=True), ["feature"], 1.0, ["feature"]
    )
    first_fold = with_poison["__ts__"] < pd.Timestamp("2025-03-21", tz="UTC")
    left = with_poison.loc[first_fold].sort_values("candidate_id")
    right = without_poison.loc[
        without_poison["__ts__"] < pd.Timestamp("2025-03-21", tz="UTC")
    ].sort_values("candidate_id")
    np.testing.assert_allclose(left["score"], right["score"], rtol=0.0, atol=1e-12)


def test_turnover_uses_selected_assets_not_unique_candidate_ids() -> None:
    ts = pd.to_datetime(
        ["2025-04-01 00:00Z", "2025-04-01 00:00Z", "2025-04-01 01:00Z", "2025-04-01 01:00Z"]
    )
    allq = pd.DataFrame(
        {
            "candidate_id": ["a0", "b0", "a1", "c1"],
            "__symbol__": ["A", "B", "A", "C"],
            "__ts__": ts,
            "side_name": "long",
            "m": "2025-04",
            "execution_gross_ev_12h": [0.03, 0.02, 0.03, 0.01],
            "execution_cost_return": 0.01,
            "execution_net_ev_12h": [0.02, 0.01, 0.02, 0.0],
            "mapped_score": [4.0, 3.0, 2.0, 1.0],
        }
    )
    result = MOD.diagnostic(allq, allq)
    assert result["adjacent_hour_selected_asset_turnover"] == pytest.approx(2 / 3)
    assert "daily_top10_turnover" not in result
