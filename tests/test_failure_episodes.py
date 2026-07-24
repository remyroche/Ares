from __future__ import annotations

import pandas as pd
import pytest

from extreme_price_movements.unsupervised_regime_learning.failure_episodes import (
    EX_POST_PREFIX,
    build_failure_episodes,
    episode_feature_columns,
    validate_inference_feature_columns,
)


def _rows() -> pd.DataFrame:
    rows = []
    for day, ev, hit in (
        ("2025-01-01", 0.02, 1.0),
        ("2025-01-02", -0.03, 0.0),
        ("2025-01-03", -0.01, 0.0),
        ("2025-01-06", 0.01, 1.0),
    ):
        for side, archetype in (("long", "trend"), ("short", "reversion")):
            rows.append(
                {
                    "__ts__": pd.Timestamp(day, tz="UTC") + pd.Timedelta(hours=1),
                    "__symbol__": f"{side}_asset",
                    "side_name": side,
                    "archetype_policy_key": archetype,
                    "hit_probability": 0.8,
                    "clean_exec": hit,
                    "ev_after_1pct": ev,
                    "exec_margin": ev + 0.005,
                    "dirty_positive": float(ev > 0.0 and hit == 0.0),
                    "historical_rank": 0.9,
                    "base_score": 0.7,
                    "score_meta_base_soft_label": 0.8,
                    "full_path_bad_mae_1r": float(hit == 0.0),
                    "first_touch_bad_mae_1r": float(hit == 0.0),
                    "timeout": 0.0,
                }
            )
    return pd.DataFrame(rows)


def test_every_negative_pnl_day_is_covered_by_parent_episode() -> None:
    result = build_failure_episodes(_rows())
    assert len(result.coverage) == 2
    assert result.coverage["covered_by_parent_episode"].all()
    assert result.manifest["negative_day_coverage_pass"] is True
    assert result.parent_episodes["negative_pnl_days"].sum() == 2


def test_ex_post_columns_are_rejected_from_inference_contract() -> None:
    with pytest.raises(ValueError, match="Ex-post failure features"):
        validate_inference_feature_columns([f"{EX_POST_PREFIX}brier"])
    validate_inference_feature_columns(["state__volatility", "state__gmm_entropy"])
    with pytest.raises(ValueError, match="Ex-post failure features"):
        validate_inference_feature_columns(["availability__target__next3d__failure"])


def test_parent_episode_keeps_local_side_and_archetype_membership() -> None:
    result = build_failure_episodes(_rows())
    negative = result.parent_episodes.loc[
        result.parent_episodes["negative_pnl_days"].gt(0)
    ]
    assert len(negative) == 1
    assert negative.iloc[0]["affected_sides"] == "long|short"
    assert negative.iloc[0]["affected_archetypes"] == "reversion|trend"


def test_episode_retains_base_meta_error_shapes_and_failure_trajectory() -> None:
    result = build_failure_episodes(_rows())
    daily = result.daily_global.set_index("day")
    episode = result.parent_episodes.iloc[0]

    assert f"{EX_POST_PREFIX}base_signed_residual" in daily
    assert f"{EX_POST_PREFIX}meta_signed_residual" in daily
    assert f"{EX_POST_PREFIX}dirty_positive_rate" in daily
    assert f"{EX_POST_PREFIX}mean_exec_margin" in daily
    assert (
        daily.loc[
            pd.Timestamp("2025-01-02", tz="UTC"),
            f"{EX_POST_PREFIX}base_signed_residual",
        ]
        < 0
    )
    assert (
        daily.loc[
            pd.Timestamp("2025-01-02", tz="UTC"),
            f"{EX_POST_PREFIX}meta_confidence_high_false_positive_composition",
        ]
        == 1.0
    )
    assert episode[f"{EX_POST_PREFIX}transition_day"] == pd.Timestamp(
        "2025-01-02", tz="UTC"
    )
    assert episode[f"{EX_POST_PREFIX}peak_day"] == pd.Timestamp("2025-01-02", tz="UTC")
    assert episode[f"{EX_POST_PREFIX}recovery_day"] == pd.Timestamp(
        "2025-01-06", tz="UTC"
    )
    assert episode[f"{EX_POST_PREFIX}recovery_lag_days"] == 3
    assert f"{EX_POST_PREFIX}peak_base_signed_residual" in result.parent_episodes
    assert f"{EX_POST_PREFIX}peak_base_signed_residual" not in episode_feature_columns(
        result.parent_episodes,
        include_ex_post=False,
    )
    with pytest.raises(ValueError, match="Ex-post failure features"):
        validate_inference_feature_columns(
            [f"{EX_POST_PREFIX}peak_base_signed_residual"]
        )


def test_optional_base_meta_residuals_are_not_synthesized_when_scores_absent() -> None:
    source = _rows().drop(columns=["base_score", "score_meta_base_soft_label"])
    result = build_failure_episodes(source)

    assert f"{EX_POST_PREFIX}primary_signed_residual" in result.daily_global
    assert f"{EX_POST_PREFIX}base_signed_residual" not in result.daily_global
    assert f"{EX_POST_PREFIX}meta_signed_residual" not in result.daily_global
