from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.run_base_r3_recall_ablation import _month_stability_summary

from extreme_price_movements.base_r3_recall_ablation import (
    R3_CLEAR_DEFINITIONS,
    R3_SCORE_DEFINITIONS,
    R3_WEIGHT_DEFINITIONS,
    build_r3_sample_weight,
    materialize_r3_classes,
    query_group_sizes,
    query_r3_ranking_metrics,
    ranker_may_advance,
    score_r3_simplex,
)


def _labels() -> pd.DataFrame:
    return pd.DataFrame({
        "pre_adverse_mfe_bps": [126.0, 151.0, 199.0, 205.0, 310.0, 100.0],
        "atr_bps": [80.0, 200.0, 200.0, 100.0, 250.0, 100.0],
        "lower_touch_minute": [-1, -1, -1, 4, -1, 2],
        "robust_clear_event_b0": [1, 1, 1, 1, 1, 0],
        "robust_clear_event_b25": [1, 1, 1, 1, 1, 0],
        "robust_clear_event_b50": [0, 1, 1, 1, 1, 0],
    })


def test_bps_and_atr_target_definitions_use_exact_pre_adverse_primitives() -> None:
    frame = _labels()
    b25 = materialize_r3_classes(frame, R3_CLEAR_DEFINITIONS[0])
    b50 = materialize_r3_classes(frame, R3_CLEAR_DEFINITIONS[1])
    max_atr = materialize_r3_classes(frame, R3_CLEAR_DEFINITIONS[-2])
    additive = materialize_r3_classes(frame, R3_CLEAR_DEFINITIONS[-1])
    assert b25.r3_class.tolist() == [2, 2, 2, 2, 2, 0]
    assert b50.r3_class.tolist() == [1, 2, 2, 2, 2, 0]
    # max(150 bps, ATR): row 2 needs 200 bps and is not clear at 151.
    assert max_atr.r3_class.tolist() == [1, 1, 1, 2, 2, 0]
    # 150 bps + 0.5 ATR: row 3 needs 250 bps and is not clear at 199.
    assert additive.r3_class.tolist() == [1, 1, 1, 2, 2, 0]


def test_weights_are_training_only_bounded_and_mean_one() -> None:
    frame = _labels()
    # ensure each class has support for every class-balanced arm
    classes = np.array([0, 1, 2, 0, 1, 2], dtype=np.int8)
    for definition in R3_WEIGHT_DEFINITIONS:
        value = build_r3_sample_weight(frame, classes, definition)
        assert np.isfinite(value).all()
        assert (value > 0.0).all()
        assert np.isclose(value.mean(), 1.0)
    assert np.allclose(build_r3_sample_weight(frame, classes, R3_WEIGHT_DEFINITIONS[0]), 1.0)


def test_score_sweep_and_query_metrics_preserve_query_local_scope() -> None:
    frame = pd.DataFrame({
        "candidate_id": list("abcdef"),
        "decision_ts": pd.to_datetime(["2024-01-01"] * 3 + ["2024-01-02"] * 3, utc=True),
        "side_name": ["long"] * 6,
        "r3_class": [2, 1, 0, 0, 2, 1],
        "net_bps": [200.0, 1.0, -100.0, -100.0, 220.0, 0.0],
    })
    probability = np.array([
        [.1, .1, .8], [.1, .7, .2], [.8, .1, .1],
        [.8, .1, .1], [.1, .1, .8], [.1, .7, .2],
    ])
    for definition in R3_SCORE_DEFINITIONS:
        frame["score"] = score_r3_simplex(probability, definition)
        metrics = query_r3_ranking_metrics(frame, score_column="score")
        assert metrics["within_query_rank_ic"] > 0.0
        assert metrics["top30_winner_recall"] == 1.0
        assert metrics["top40_winner_recall"] == 1.0
        assert metrics["target_decile_adjacent_violations"] == 0.0
    assert query_group_sizes(frame).tolist() == [3, 3]


def test_non_linear_clear_adverse_scores_are_finite_and_follow_clear_adverse_ordering() -> None:
    probability = np.array([
        [.05, .20, .75],  # clean clear
        [.35, .20, .45],  # less-clear and more adverse
        [.75, .20, .05],  # adverse
    ])
    definitions = {item.name: item for item in R3_SCORE_DEFINITIONS}
    for name in ("clear_x_no_adverse", "clear_vs_adverse_ratio"):
        score = score_r3_simplex(probability, definitions[name])
        assert np.isfinite(score).all()
        assert score[0] > score[1] > score[2]


def test_ranker_gate_requires_all_three_base_ranking_metrics() -> None:
    control = {"within_query_rank_ic": .10, "top30_winner_recall": .35, "top40_winner_recall": .46}
    assert ranker_may_advance(control, {"within_query_rank_ic": .10, "top30_winner_recall": .35, "top40_winner_recall": .46})
    assert not ranker_may_advance(control, {"within_query_rank_ic": .11, "top30_winner_recall": .34, "top40_winner_recall": .50})


def test_month_stability_summary_exposes_worst_month_and_quality_gates() -> None:
    frame = pd.DataFrame({
        "phase": ["score", "score", "score"],
        "score_definition": ["contrast_l0p5"] * 3,
        "target_metric": ["own"] * 3,
        "scope": ["month:2024-01", "month:2024-02", "pooled"],
        "within_query_rank_ic": [.12, .08, .10],
        "top30_winner_recall": [.42, .39, .41],
        "top40_winner_recall": [.52, .49, .51],
        "top5_clear_uplift": [.10, .05, .07],
        "top5_net_uplift_bps": [2.0, -1.0, .5],
        "target_decile_adjacent_violations": [0.0, 0.0, 0.0],
    })
    summary = _month_stability_summary(frame).iloc[0]
    assert summary["months"] == 2
    assert summary["min_month_ic"] == .08
    assert np.isclose(summary["top30_winner_recall_range"], .03)
    assert not summary["all_month_top30_recall_ge_40pct"]
    assert not summary["all_month_top40_recall_ge_50pct"]
    assert summary["all_month_deciles_monotonic"]
