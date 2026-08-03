from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.audit_cross_era_direct_head_transition_attribution import (
    ACTIVE_BANDS,
    add_causal_states,
    assert_identity_equal,
    fixed_z_band,
    head_metrics,
    tail_composition,
    _join_current,
)


def _frame(rows: int = 20) -> pd.DataFrame:
    timestamp = pd.date_range("2026-07-20", periods=rows, freq="h", tz="UTC")
    return pd.DataFrame({
        "candidate_id": [f"c{i}" for i in range(rows)], "__ts__": timestamp,
        "__symbol__": ["X/USD:USD"] * rows, "side_name": ["long" if i % 2 else "short" for i in range(rows)],
        "era": ["test"] * rows, "execution_net_ev_12h": np.linspace(-.04, .04, rows),
        "q25_net_bps": np.linspace(-200, 200, rows), "q50_net_bps": np.linspace(-100, 300, rows),
        "p_loss_le_100": np.linspace(.8, .1, rows), "p_loss_le_200": np.linspace(.7, .05, rows), "p_loss_le_400": np.linspace(.5, .01, rows),
        "mapped_q25_bps": np.linspace(-100, 100, rows),
        "regime_transition_entropy_12h": np.linspace(-2, 2, rows), "regime_transition_entropy_48h": np.linspace(-1, 1, rows),
        "regime_stability_24h": np.linspace(1, -1, rows), "volatility_of_volatility_48": np.linspace(-.5, .5, rows), "vov_interaction": np.linspace(-.2, .2, rows),
        "is_high_vol_regime": [1] * rows, "is_low_vol_regime": [0] * rows, "is_ranging": [0] * rows,
    })


def test_fixed_z_bands_are_fixed_and_do_not_depend_on_outcomes():
    result = fixed_z_band(pd.Series([-2., -.1, .5, 2., np.nan]))
    assert result.tolist() == ["<-.75", "[-.75,0)", "[0,.75)", ">=.75", "missing"]


def test_missing_active_probability_is_retained_while_raw_states_cover_current():
    frame = _frame(4)
    active = pd.DataFrame({"source_utc": [frame.loc[0, "__ts__"]], "prediction": [.8]})
    result = add_causal_states(frame, active)
    assert result["active_transition_band"].tolist()[0] == ">=0.75"
    assert result["active_transition_band"].tolist()[1:] == ["missing"] * 3
    assert result.filter(like="state__").notna().all().all()


def test_head_metrics_include_quantile_and_all_severe_heads_by_state():
    frame = add_causal_states(_frame(), pd.DataFrame({"source_utc": [], "prediction": []}))
    result = head_metrics(frame, "current")
    assert set(result["head"]) == {"q25_net_bps", "q50_net_bps", "p_loss_le_100", "p_loss_le_200", "p_loss_le_400"}
    assert set(result.loc[result["state_dimension"].eq("active_transition_band"), "state"]) == {"missing"}
    assert {"transition_pressure_z", "entropy_acceleration_z", "entropy_x_vov_z"}.issubset(set(result["state_dimension"]))
    assert np.isfinite(result.loc[result["head_family"].eq("quantile"), "pinball_loss_bps"]).all()


def test_tail_has_one_global_book_and_local_state_tail_is_explicitly_descriptive():
    frame = add_causal_states(_frame(), pd.DataFrame({"source_utc": [], "prediction": []}))
    composition, diagnostic = tail_composition(frame, "current")
    assert composition.iloc[0]["selection_scope"] == "one_global_top10"
    assert composition.iloc[0]["rows"] == 2
    assert set(composition.loc[composition["selection_scope"].eq("one_global_top10"), "score_scope"]) == {"raw_q25", "frozen_mapped_q25"}
    assert diagnostic.iloc[0]["selection_scope"] == "state_local_top10_descriptive_only"
    assert diagnostic["descriptive_only"].all()


def test_identity_equality_fails_closed_on_missing_or_duplicate_rows():
    left = _frame(3).loc[:, ["candidate_id", "__ts__", "__symbol__", "side_name"]]
    assert assert_identity_equal(left, left.copy(), label="test")["identity_complete_one_to_one"]
    with pytest.raises(ValueError, match="coverage mismatch"):
        assert_identity_equal(left, left.iloc[:-1], label="test")
    with pytest.raises(ValueError, match="duplicate"):
        assert_identity_equal(left, pd.concat([left, left.iloc[[0]]]), label="test")


def test_current_feature_availability_fails_closed_after_the_decision_time():
    raw = _frame(3).drop(columns=["execution_net_ev_12h"]).copy()
    raw["feature_available_at"] = raw["__ts__"] + pd.Timedelta(hours=2)
    raw["execution_decision_utc"] = raw["__ts__"] + pd.Timedelta(hours=1)
    predictions = raw.loc[:, ["candidate_id", "__ts__", "__symbol__", "side_name", "q25_net_bps", "q50_net_bps", "p_loss_le_100", "p_loss_le_200", "p_loss_le_400", "mapped_q25_bps"]]
    labels = raw.loc[:, ["candidate_id", "__ts__", "__symbol__", "side_name"]].copy()
    labels["execution_net_ev_12h"] = 0.0
    with pytest.raises(ValueError, match="availability"):
        _join_current(predictions, raw, labels)
