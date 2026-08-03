from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.materialize_clean_competing_risk_probability_oof import (
    ARCHITECTURES,
    MAPPED_SCORE,
    MAPPING_AVAILABLE,
    SCORE,
    TARGETS,
    architecture_feature_sets,
    fit_oof_probability_heads,
    forbid_action_features,
)


def _frame(rows_per_hour: int = 8) -> pd.DataFrame:
    hours = pd.date_range("2024-01-01", "2024-06-01", freq="h", inclusive="left", tz="UTC")
    n = len(hours) * rows_per_hour
    position = np.arange(n)
    timestamp = np.repeat(hours.to_numpy(), rows_per_hour)
    side = np.where(position % 2, "short", "long")
    signal = np.sin(position / 37.0) + np.where(side == "long", .10, -.10)
    result = pd.DataFrame({
        "candidate_id": [f"c-{i:08d}" for i in position], "__ts__": timestamp,
        "__symbol__": np.where(position % 3, "ETH/USD:USD", "BTC/USD:USD"), "side_name": side,
        SCORE: signal, MAPPED_SCORE: .01 * signal, MAPPING_AVAILABLE: 1,
        "regime_state_p__0": np.clip(.5 + .2 * signal, .01, .99),
        "regime_state_p__1": np.clip(.3 - .1 * signal, .01, .99),
        "regime_state_entropy": .5 + .1 * np.cos(position / 17.0),
        "regime_state_margin": np.abs(signal), "regime_state_uncertainty": .2, "regime_state_ood_score": .1,
        "transition_state_p__stable": np.clip(.6 - .2 * signal, .01, .99),
        "transition_state_p__transition": np.clip(.2 + .2 * signal, .01, .99),
        "transition_active_probability": np.clip(.2 + .3 * signal, .01, .99),
        "transition_state_entropy": .5, "transition_state_margin": np.abs(signal), "transition_state_uncertainty": .3, "transition_state_ood_score": .2,
    })
    result[TARGETS["clean_opportunity"]] = (signal > .15).astype(int)
    result[TARGETS["adverse_competing_risk"]] = (signal < -.20).astype(int)
    result["__label_available_at__"] = result["__ts__"] + pd.Timedelta(hours=12)
    result["execution_gross_ev_12h"] = np.where(signal > .15, .025, -.005)
    result["execution_cost_return"] = .01
    result["execution_net_ev_12h"] = result["execution_gross_ev_12h"] - result["execution_cost_return"]
    return result


def test_four_architectures_are_nested_and_action_fields_fail_closed() -> None:
    pools = architecture_feature_sets(_frame(1).columns)
    assert tuple(pools) == ARCHITECTURES
    assert set(pools["baseline"]).issubset(pools["regime_only"])
    assert set(pools["baseline"]).issubset(pools["transition_only"])
    assert set(pools["regime_only"]).union(pools["transition_only"]).issubset(pools["regime_plus_transition"])
    with pytest.raises(ValueError, match="timing/MAE/target-price/wait"):
        forbid_action_features(("entry_timing_p",))


def test_probability_heads_are_side_local_chronological_and_emit_joint_diagnostics() -> None:
    frame = _frame()
    predictions, selections, audit, packed = fit_oof_probability_heads(
        frame, first_evaluation="2024-03-01", last_evaluation="2024-05-01", frequency="MS", minimum_train_months=2,
    )

    probability_columns = [name for name in predictions if "_p__" in name]
    assert len(probability_columns) == len(ARCHITECTURES) * 2
    assert np.isfinite(predictions[probability_columns].to_numpy(float)).all()
    assert ((predictions[probability_columns] >= 0) & (predictions[probability_columns] <= 1)).all().all()
    assert (audit["train_label_available_max"] < audit["evaluation_start_utc"]).all()
    assert set(selections["architecture"]) == set(ARCHITECTURES)
    assert set(selections["head"]) == set(TARGETS)
    assert set(packed["kind"]) == {"metrics", "economics", "calibration"}
    assert (packed.loc[packed.kind.eq("economics"), "score"].astype(str).str.contains("joint_clean_net_of_adverse")).any()
