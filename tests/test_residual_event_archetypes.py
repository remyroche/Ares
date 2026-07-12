from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from extreme_price_movements.residual_event_archetypes import (
    GlobalEVThresholdState,
    ResidualEventArchetypeConfig,
    ResidualEventArchetypeState,
    ResidualEventBaselineState,
    ScoreExpectationState,
    add_residual_event_targets,
    causal_eight_day_hit_rate_overlay,
    screen_local_residual_features,
)
from scripts.run_residual_event_archetype_discovery import _surprise_autocorrelation


def _frame(rows: int = 240) -> pd.DataFrame:
    timestamps = pd.date_range(
        "2025-01-01", periods=rows // 6, freq="D", tz="UTC"
    ).repeat(6)
    score = np.tile(np.linspace(0.05, 0.95, 6, dtype=np.float32), rows // 6)
    hit = (score >= 0.55).astype(np.float32)
    # A persistent adverse local stream gives the event annotator non-trivial
    # outcomes while preserving a non-RangeIndex test surface.
    adverse = (np.arange(rows) >= rows // 2) & (np.arange(rows) % 6 < 2)
    hit[adverse] = 0.0
    return pd.DataFrame(
        {
            "__ts__": timestamps,
            "__symbol__": np.where(np.arange(rows) % 2, "A", "B"),
            "side_name": np.where(np.arange(rows) % 2, "long", "short"),
            "archetype_policy_key": np.where(np.arange(rows) % 3, "arch_a", "arch_b"),
            "score_meta_base_soft_label": score,
            "clean_exec": hit,
            "ev_after_1pct": hit * 0.02 - (1.0 - hit) * 0.01,
            "dirty_positive": (1.0 - hit),
            "full_path_bad_mae_1r": (1.0 - hit),
            "timeout": 0.0,
            "stop_or_adverse": 0.0,
            "directional_feature": score + np.sin(np.arange(rows)) * 0.01,
        },
        index=pd.Index(np.arange(10_000, 10_000 + rows)),
    )


def _config() -> ResidualEventArchetypeConfig:
    return ResidualEventArchetypeConfig(
        min_global_threshold_rows=60,
        min_local_threshold_rows=30,
        min_local_state_rows=10_000,
        min_side_state_rows=10_000,
        min_event_class_rows=5,
        timestamp_min_peers=2,
    )


def test_global_ev_threshold_is_train_fitted_and_index_safe() -> None:
    train = _frame()
    cfg = _config()
    state = GlobalEVThresholdState(cfg).fit(train)
    out = state.transform(train.drop(columns=["clean_exec", "ev_after_1pct"]))

    assert out.index.equals(train.index)
    assert out["resid_event_top10_population"].sum() > 0
    assert (
        out["resid_event_top20_population"].sum()
        >= out["resid_event_top10_population"].sum()
    )
    assert state.global_targets["top10"] >= state.global_targets["top20"]


def test_timestamp_neutral_surprise_uses_leave_one_out_peers() -> None:
    train = _frame()
    cfg = _config()
    thresholds = GlobalEVThresholdState(cfg).fit(train)
    expectation = ScoreExpectationState(cfg).fit(train)
    labelled = add_residual_event_targets(
        train, threshold_state=thresholds, expectation_state=expectation
    )

    first_ts = labelled["__ts__"].iloc[0]
    group = labelled.loc[labelled["__ts__"].eq(first_ts)]
    residual = group["resid_event_global_surprise"].to_numpy(dtype=float)
    neutral = group["resid_event_timestamp_neutral_surprise"].to_numpy(dtype=float)
    expected = residual - (residual.sum() - residual) / (len(residual) - 1)
    np.testing.assert_allclose(neutral, expected, atol=1e-6)
    np.testing.assert_allclose(
        group["resid_event_market_peer_surprise"].to_numpy(dtype=float),
        (residual.sum() - residual) / (len(residual) - 1),
        atol=1e-6,
    )


def test_frozen_hit_probability_is_the_residual_expectation() -> None:
    train = _frame()
    train["hit_probability"] = np.linspace(0.2, 0.8, len(train), dtype=np.float32)
    cfg = _config()
    thresholds = GlobalEVThresholdState(cfg).fit(train)
    expectation = ScoreExpectationState(cfg).fit(train)
    labelled = add_residual_event_targets(
        train, threshold_state=thresholds, expectation_state=expectation
    )
    np.testing.assert_allclose(
        labelled["resid_event_expected_hit"], train["hit_probability"], atol=1e-7
    )


def test_oos_transform_rejects_outcomes_and_allows_preentry_rows() -> None:
    train = _frame()
    cfg = _config()
    state = ResidualEventArchetypeState(cfg).fit(
        train, candidate_features=["directional_feature"]
    )
    with pytest.raises(ValueError, match="outcome columns"):
        state.transform_oos(train)

    safe = train.drop(
        columns=[
            "clean_exec",
            "ev_after_1pct",
            "dirty_positive",
            "full_path_bad_mae_1r",
            "timeout",
            "stop_or_adverse",
        ]
    )
    out = state.transform_oos(safe)
    assert out.index.equals(train.index)
    assert out.columns.str.startswith(
        ("resid_event_aegmm_", "resid_event_market_aegmm_")
    ).all()


def test_assessment_smoother_excludes_current_day_outcomes() -> None:
    frame = _frame(120)
    cfg = _config()
    frame["resid_event_top10_population"] = 1
    overlay = causal_eight_day_hit_rate_overlay(frame, config=cfg, embargo_hours=0.0)
    first_day = frame["__ts__"].iloc[0].floor("D")
    assert (
        overlay.loc[
            frame["__ts__"].dt.floor("D").eq(first_day), "assessment_hr8_surprise"
        ]
        .isna()
        .all()
    )
    second_day = first_day + pd.Timedelta(days=1)
    assert (
        overlay.loc[
            frame["__ts__"].dt.floor("D").eq(second_day), "assessment_hr8_effective_n"
        ]
        .gt(0.0)
        .all()
    )


def test_event_baseline_marks_rows_without_cross_archetype_merge() -> None:
    train = _frame()
    cfg = _config()
    thresholds = GlobalEVThresholdState(cfg).fit(train)
    expectation = ScoreExpectationState(cfg).fit(train)
    raw = add_residual_event_targets(
        train, threshold_state=thresholds, expectation_state=expectation
    )
    baseline = ResidualEventBaselineState(cfg).fit(raw)
    labelled = add_residual_event_targets(
        train,
        threshold_state=thresholds,
        expectation_state=expectation,
        baseline_state=baseline,
    )
    assert len(labelled) == len(train)
    assert set(labelled["resid_event_class"].astype(str)).issubset(
        {
            "normal",
            "negative_residual_event",
            "adverse_path_event",
            "positive_residual_event",
            "favorable_near_miss_event",
            "high_variance_event",
        }
    )


def test_extreme_single_day_surprise_is_retained_without_prior_week() -> None:
    train = _frame(120)
    cfg = _config()
    thresholds = GlobalEVThresholdState(cfg).fit(train)
    expectation = ScoreExpectationState(cfg).fit(train)
    raw = add_residual_event_targets(
        train, threshold_state=thresholds, expectation_state=expectation
    )
    baseline = ResidualEventBaselineState(cfg).fit(raw.iloc[:60])
    # Force an acute local miss on a later day. The event must not require a
    # seven/eight-day outcome history to survive target construction.
    shocked = train.iloc[60:].copy()
    shocked["hit_probability"] = shocked["clean_exec"].astype(np.float32)
    local = shocked["side_name"].eq("short") & shocked["archetype_policy_key"].eq(
        "arch_a"
    )
    shocked.loc[local, "clean_exec"] = 0.0
    shocked.loc[local, "hit_probability"] = 1.0
    labelled = add_residual_event_targets(
        shocked,
        threshold_state=thresholds,
        expectation_state=expectation,
        baseline_state=baseline,
    )
    assert labelled["resid_event_large_event_strength"].max() > 0.0
    assert labelled["resid_event_persistent"].max() > 0.0


def test_lgbm_screen_can_be_explicitly_disabled() -> None:
    frame = _frame(1_800)
    cfg = _config()
    thresholds = GlobalEVThresholdState(cfg).fit(frame)
    expectation = ScoreExpectationState(cfg).fit(frame)
    raw = add_residual_event_targets(
        frame, threshold_state=thresholds, expectation_state=expectation
    )
    baseline = ResidualEventBaselineState(cfg).fit(raw)
    labelled = add_residual_event_targets(
        frame,
        threshold_state=thresholds,
        expectation_state=expectation,
        baseline_state=baseline,
    )
    labels = pd.Categorical(labelled["resid_event_class"]).codes.astype(np.int32)
    selected, metrics, meta = screen_local_residual_features(
        labelled,
        labels,
        ["directional_feature"],
        config=ResidualEventArchetypeConfig(
            **{
                **_config().__dict__,
                "lgbm_enabled": False,
                "max_features_after_mi": 1,
                "max_features_after_lgbm": 1,
            }
        ),
        seed=7,
    )
    assert selected == ["directional_feature"]
    assert float(metrics["lgbm_validation_gain"].max()) == 0.0
    assert meta["disabled"] == 1.0


def test_signed_autocorrelation_does_not_compress_calendar_gaps() -> None:
    frame = pd.DataFrame(
        {
            "__ts__": pd.to_datetime(
                ["2026-01-01", "2026-01-02", "2026-01-04"], utc=True
            ),
            "side_name": "long",
            "archetype_policy_key": "mixed",
            "resid_event_timestamp_neutral_surprise": [-0.8, -0.6, -0.9],
        }
    )
    result = _surprise_autocorrelation(
        frame,
        ["side_name", "archetype_policy_key"],
        surprise_col="resid_event_timestamp_neutral_surprise",
        population="top10",
    )
    assert int(result.loc[0, "consecutive_pairs"]) == 1
    assert np.isclose(result.loc[0, "adverse_lag1_product_mean"], 0.48)
    assert np.isclose(result.loc[0, "favorable_lag1_product_mean"], 0.0)
