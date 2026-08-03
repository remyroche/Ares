from __future__ import annotations

import numpy as np
import pandas as pd

from scripts import repair_canonical_full_base_opportunity_ablation as repair


def test_raw_column_round_trip() -> None:
    column = repair.raw_column("hard25", "S1+B", "compact_d4")
    assert column == "raw__hard25__S1+B__compact_d4"
    assert repair.parse_raw_column(column) == ("hard25", "S1+B", "compact_d4")


def test_expected_tail_uses_random_tie_expectation() -> None:
    frame = pd.DataFrame(
        {
            "candidate_id": ["a", "b", "c", "d"],
            "side_name": ["long", "short", "long", "short"],
            "execution_net_ev_12h": [0.04, -0.02, 0.02, -0.01],
            "opportunity_gross_above_cost_0bps": [1, 0, 1, 0],
            "opportunity_gross_above_cost_25bps": [1, 0, 0, 0],
        }
    )
    result = repair.expected_tail(frame, np.ones(4), 0.50)
    assert result["rows"] == 2
    assert result["cutoff_tie_fraction_of_book"] == 2.0
    assert np.isclose(result["random_tie_expected_net_bps"], 75.0)
    assert np.isclose(result["random_tie_expected_hard0_precision"], 0.5)


def test_raw_selection_ignores_mapped_columns_and_chooses_raw_economics() -> None:
    rows = 40
    frame = pd.DataFrame(
        {
            "candidate_id": [f"c{i:03d}" for i in range(rows)],
            "side_name": np.where(np.arange(rows) % 2, "short", "long"),
            "execution_net_ev_12h": np.linspace(-0.02, 0.02, rows),
            "opportunity_gross_above_cost_0bps": np.arange(rows) >= 20,
            "opportunity_gross_above_cost_25bps": np.arange(rows) >= 30,
        }
    )
    predictions = pd.DataFrame()
    for target in repair.base.TARGETS:
        for arm_index, arm in enumerate(repair.base.PRIMARY_ARMS):
            for geometry in ("fixed_d5", "compact_d4", "deep_d6"):
                score = np.arange(rows, dtype=float)
                if arm_index >= 2:
                    score = -score
                predictions[
                    repair.raw_column(target, arm, geometry)
                ] = score
    predictions["mapped_net__bad"] = -np.arange(rows, dtype=float)
    _, selected_arms, winners = repair.raw_selection(predictions, frame)
    assert all(arms == ["S0", "S1"] for arms in selected_arms.values())
    assert len(winners) == 2 * len(repair.base.TARGETS)


def test_promotion_is_always_false_for_reused_april() -> None:
    metrics = pd.DataFrame(
        {
            "config": ["x"],
            "mapping_kind": ["pooled"],
            "top_fraction": [0.10],
            "random_tie_expected_net_bps": [10.0],
            "latest_week_net_bps": [5.0],
            "cutoff_tie_fraction_of_book": [0.0],
        }
    )
    sides = pd.DataFrame(
        {
            "config": ["x", "x"],
            "mapping_kind": ["pooled", "pooled"],
            "side_name": ["long", "short"],
            "share": [0.5, 0.5],
            "net_bps": [1.0, 1.0],
        }
    )
    controls = pd.DataFrame(
        {
            "control": ["base"],
            "mapping_kind": ["pooled"],
            "top_fraction": [0.10],
            "random_tie_expected_net_bps": [0.0],
        }
    )
    gate = repair.promotion_gates(metrics, sides, controls, ["x"]).iloc[0]
    assert not gate.promotion_eligible
    assert not gate.portfolio_replay_authorized
