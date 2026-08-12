"""Focused contract tests for the ten-head conditional-usefulness runner."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_ten_head_conditional_usefulness_funnel.py"
SPEC = importlib.util.spec_from_file_location("ten_head_conditional_usefulness", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_source_feature_contract_requires_exactly_120_non_fixed_columns():
    columns = list(MODULE.SOURCE_FIXED_COLUMNS) + [f"f_{index}" for index in range(120)]
    assert MODULE.source_feature_columns(columns) == [f"f_{index}" for index in range(120)]
    try:
        MODULE.source_feature_columns(columns[:-1])
    except ValueError as error:
        assert "120" in str(error)
    else:
        raise AssertionError("short source contract should fail")


def test_residual_grade_preserves_declared_inclusive_boundaries():
    values = np.asarray([-151.0, -150.0, -149.0, -50.0, 0.0, 50.0, 150.0, 151.0])
    result = MODULE.residual_grade(values, (-150.0, -50.0, 50.0, 150.0))
    assert result.tolist() == [0, 0, 1, 1, 2, 2, 3, 4]


def test_head_rank_uses_training_reference_not_held_distribution():
    rank = MODULE._rank_against_reference([0.0, 1.0, 2.0, 3.0], [0.0, 1.5, 3.0])
    assert np.allclose(rank, [.25, .5, 1.0])


def test_stack_uses_all_ten_heads_and_global_base_blend():
    population = pd.DataFrame({
        "candidate_id": ["a", "b"],
        "__ts__": pd.to_datetime(["2025-05-01", "2025-05-01"], utc=True),
        "month": ["2025-05", "2025-05"],
        "side_name": ["long", "long"],
        "__symbol__": ["A", "B"],
        "net_bps": [200.0, -100.0],
        "gross_bps": [300.0, 0.0],
        "base_rank": [.8, .2],
        "base_anchor_bps": [0.0, 0.0],
        "base_score": [.1, -.1],
    })
    ranks = pd.DataFrame({"candidate_id": ["a", "b"]})
    for index, spec in enumerate(MODULE.HEAD_SPECS):
        ranks[spec.name] = [.9 if index % 2 else .8, .1 if index % 2 else .2]
    out = MODULE.stack_scores(population, ranks)
    assert np.isclose(out.loc[out.candidate_id.eq("a"), "consensus_rank"].iloc[0], .85)
    assert np.isclose(out.loc[out.candidate_id.eq("a"), "score"].iloc[0], .75 * .8 + .25 * .85)


def test_conditional_comparison_keeps_same_rows_and_exposes_requested_tails():
    rows = 200
    base = pd.DataFrame({
        "candidate_id": [f"c{index}" for index in range(rows)],
        "__ts__": pd.to_datetime(["2025-05-01"] * 100 + ["2025-06-01"] * 100, utc=True),
        "net_bps": np.r_[np.linspace(-100, 100, 100), np.linspace(-80, 80, 100)],
        "gross_bps": np.r_[np.linspace(0, 200, 100), np.linspace(20, 180, 100)],
        "score_incumbent": np.linspace(1, 0, rows),
        "score_candidate": np.r_[np.linspace(0, 1, 100), np.linspace(0, 1, 100)],
    })
    summary = MODULE.conditional_downstream_summary(
        base,
        candidate_score_column="score_candidate",
        incumbent_score_column="score_incumbent",
    )
    assert summary["conditional_rows"] == rows
    assert "delta_top1_net_bps" in summary
    assert "delta_top2_net_bps" in summary
    assert "delta_top5_month_worst_net_bps" in summary


def test_target_query_shortlist_contains_both_target_and_query_changes():
    target_screen = pd.DataFrame({"target": [
        "resid_wide_200_75", "resid_tight_100_50", "resid_default_150_50",
    ]})
    candidates = MODULE._target_query_candidates(
        target_screen, ("q1_cycle_4h_side", "q1_cycle_6h_side", "q1_cycle_8h_side"), limit=5,
    )
    assert candidates[0] == ("resid_default_150_50", "q1_cycle_4h_side")
    assert len({target for target, _ in candidates}) > 1
    assert len({query for _, query in candidates}) > 1


def test_target_screen_preserves_filtered_source_index_for_rank_correlation():
    development = pd.DataFrame(
        {
            "net_bps": [-320.0, -120.0, -20.0, 80.0, 220.0],
            "base_anchor_bps": [0.0] * 5,
        },
        index=[101, 104, 110, 119, 127],
    )
    screen = MODULE._screen_targets(development)
    assert screen["grade_net_spearman"].notna().all()
    assert (screen["grade_net_spearman"] > 0).all()
