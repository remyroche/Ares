from __future__ import annotations

import pandas as pd

from scripts.ablate_mapped_ev_historical_tie_repair import (
    RANK_SPECS,
    _apply_recipe,
    _select_top_k,
    select_recipe_from_history,
)


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "candidate_id": ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"],
            "primary_mapped_ev": [0.02, 0.01, 0.01, 0.01, 0.0, 0.0, -0.01, -0.01, -0.01, -0.01],
            "direct": [1.0, 1.0, 4.0, 3.0, 0.0, 0.0, -1.0, -2.0, -3.0, -4.0],
            "capture": [0.1, 0.2, 0.9, 0.8, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0],
            "base": list(range(10)),
            "residual": list(range(10)),
            "execution_net_ev_12h": [0.0, 0.0, 0.04, 0.03, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "utc_date": ["2026-07-20"] * 10,
            "side_name": ["long", "short"] * 5,
            "execution_decision_utc": pd.date_range("2026-07-20", periods=10, freq="h", tz="UTC"),
        }
    )


def test_secondary_key_only_changes_order_inside_frozen_primary_level() -> None:
    result = _select_top_k(_frame(), primary="primary_mapped_ev", secondary=("direct",), top_k_fraction=0.30)
    selected = result.loc[result["tie_repair_top_k"]]
    assert set(selected["candidate_id"]) == {"a", "c", "d"}
    # a is strictly above the mapped cutoff and cannot be displaced.
    assert bool(selected.loc[selected["candidate_id"].eq("a")].shape[0])
    assert selected["primary_mapped_ev"].min() == 0.01


def test_history_recipe_selection_is_predeclared_and_deterministic() -> None:
    history = _frame().copy()
    history["calendar_month"] = ["2026-05"] * 5 + ["2026-06"] * 5
    winner, comparison = select_recipe_from_history(history, top_k_fraction=0.30)
    assert winner in RANK_SPECS
    assert comparison["historical_selection_rank"].tolist() == list(range(1, len(comparison) + 1))


def test_apply_recipe_keeps_strict_positive_floor_contract() -> None:
    result = _apply_recipe(_frame(), recipe="raw_direct_ev", top_k_fraction=0.30)
    assert result["global_top10_capacity_member"].sum() == 3
    assert result["globally_admitted_floor_0bps"].sum() == 3
    assert not result.loc[result["primary_mapped_ev"].eq(0.0), "globally_admitted_floor_0bps"].any()


def test_current_cohort_day_is_signal_time_not_next_hour_decision() -> None:
    frame = _frame()
    frame["__ts__"] = frame["execution_decision_utc"] - pd.Timedelta(hours=1)
    # This checks the contract in the helper through the public evaluator path.
    from scripts.ablate_mapped_ev_historical_tie_repair import _current_recipes

    result = _current_recipes(frame.rename(columns={"primary_mapped_ev": "mapped_execution_ev", "direct": "final_direct_net_raw", "capture": "final_capture_probability", "base": "base_oof_score", "residual": "existing_alpha_ev"}))
    assert result.loc[result["candidate_id"].eq("a"), "utc_date"].iloc[0] == "2026-07-19"
