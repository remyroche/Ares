from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from extreme_price_movements.residual_event_block_taxonomy import (
    BlockTaxonomyConfig,
    annotate_onset_mechanism_profiles,
    attach_detector_block_coverage,
    attach_event_blocks,
    build_block_taxonomy,
    daily_observable_state,
    detector_recognized_missed_contrasts,
    matched_benign_block_controls,
)


def test_onset_mechanism_annotation_uses_only_onset_and_active_state() -> None:
    blocks = pd.DataFrame(
        {
            "family__liquidation_pressure__active_mean_z": [-2.0],
            "family__liquidation_pressure__onset_abs_delta": [3.0],
            "family__recovery_short_covering__active_mean_z": [0.1],
            "family__recovery_short_covering__onset_abs_delta": [0.2],
            "family__funding_transition__active_mean_z": [0.0],
            "family__funding_transition__onset_abs_delta": [0.0],
            "family__correlation_fragmentation__active_mean_z": [0.0],
            "family__correlation_fragmentation__onset_abs_delta": [0.0],
            "family__volatility_compression_transition__active_mean_z": [0.0],
            "family__volatility_compression_transition__onset_abs_delta": [0.0],
            "family__asset_market_divergence__active_mean_z": [0.0],
            "family__asset_market_divergence__onset_abs_delta": [0.0],
            # A deliberately extreme recovery value must not change a causal
            # onset annotation.
            "family__recovery_short_covering__recovery_abs_delta": [99.0],
        }
    )
    result = annotate_onset_mechanism_profiles(blocks)
    assert result.loc[0, "onset_primary_mechanism"] == "liquidation_pressure"
    assert result.loc[0, "onset_mechanism_confident"]
    assert result.loc[0, "onset_mechanism_available_count"] == 6


def test_onset_mechanism_annotation_does_not_assign_missing_families() -> None:
    result = annotate_onset_mechanism_profiles(
        pd.DataFrame({"event_block": ["event_001"]})
    )
    assert result.loc[0, "onset_primary_mechanism"] == "unavailable"
    assert not result.loc[0, "onset_mechanism_confident"]


from extreme_price_movements import residual_event_block_taxonomy as taxonomy
from scripts.report_residual_event_block_taxonomy import (
    _daily_state_part_streaming,
    _overlay_event_calendar,
)


def _calendar() -> pd.DataFrame:
    days = pd.date_range("2026-01-01", periods=48, freq="D", tz="UTC")
    result = pd.DataFrame(
        {
            "day": days,
            "side_name": "short",
            "archetype_policy_key": "short_default_clean_path",
            "adverse_calendar_cell": 0,
            "selected_rows": 10,
            "mean_ev_after_1pct": 0.01,
            "signed_surprise": 0.0,
        }
    )
    result.loc[[35, 36, 41], "adverse_calendar_cell"] = 1
    result.loc[[35, 36, 41], "mean_ev_after_1pct"] = -0.01
    return result


def _states() -> pd.DataFrame:
    days = pd.date_range("2026-01-01", periods=48, freq="D", tz="UTC")
    timestamps = np.repeat(days.to_numpy(), 2)
    values = np.repeat(np.arange(len(days), dtype=np.float32), 2)
    return pd.DataFrame(
        {
            "__ts__": timestamps,
            "side_name": "short",
            "archetype_policy_key": "short_default_clean_path",
            "selected_top30": True,
            "mkt_median_oi_chg_4h_rz": values,
            "market_breadth_recovery_from_24h_min": values * -0.5,
        }
    )


def test_event_blocks_are_contiguous_and_local() -> None:
    blocks = attach_event_blocks(_calendar())
    events = blocks.loc[blocks["adverse_event"]]
    assert events["event_block"].tolist() == ["event_001", "event_001", "event_002"]
    assert blocks.loc[~blocks["adverse_event"], "event_block"].eq("normal").all()


def test_event_blocks_can_cap_long_market_failure_runs() -> None:
    calendar = pd.DataFrame(
        {
            "day": pd.date_range("2024-01-01", periods=10, freq="D", tz="UTC"),
            "side_name": "global",
            "archetype_policy_key": "global_market",
            "adverse_calendar_cell": True,
        }
    )

    blocks = attach_event_blocks(calendar, max_event_days=4)

    assert blocks["event_block"].tolist() == [
        "event_001",
        "event_001",
        "event_001",
        "event_001",
        "event_002",
        "event_002",
        "event_002",
        "event_002",
        "event_003",
        "event_003",
    ]


def test_daily_state_uses_candidate_rows_and_not_outcome_columns() -> None:
    states = _states()
    states["ev_after_1pct"] = -1.0
    daily = daily_observable_state(
        states,
        features=["mkt_median_oi_chg_4h_rz", "market_breadth_recovery_from_24h_min"],
    )
    assert "ev_after_1pct" not in daily.columns
    assert len(daily) == 48
    assert daily["mkt_median_oi_chg_4h_rz"].iloc[-1] == 47.0


def test_daily_state_rejects_outcome_features_even_when_requested() -> None:
    states = _states()
    states["stop_or_adverse"] = 1.0

    with pytest.raises(ValueError, match="outcome columns"):
        daily_observable_state(
            states,
            features=["mkt_median_oi_chg_4h_rz", "stop_or_adverse"],
        )


def test_daily_state_uses_first_timestamp_and_is_immune_to_later_day_values() -> None:
    states = pd.DataFrame(
        {
            "__ts__": pd.to_datetime(
                ["2026-01-01 00:00", "2026-01-01 12:00", "2026-01-02 00:00"], utc=True
            ),
            "side_name": "short",
            "archetype_policy_key": "short_default_clean_path",
            "selected_top30": [False, True, True],
            "mkt_median_oi_chg_4h_rz": [1.0, 999.0, 2.0],
        }
    )
    daily = daily_observable_state(
        states, features=["mkt_median_oi_chg_4h_rz"], selected_only=False
    )
    assert daily["mkt_median_oi_chg_4h_rz"].tolist() == [1.0, 2.0]


def test_streaming_daily_state_matches_daily_open_contract(tmp_path) -> None:
    path = tmp_path / "state.parquet"
    pd.DataFrame(
        {
            "__ts__": pd.to_datetime(
                [
                    "2026-01-01 00:00",
                    "2026-01-01 00:00",
                    "2026-01-01 12:00",
                    "2026-01-02 00:00",
                ],
                utc=True,
            ),
            "side_name": ["short"] * 4,
            "archetype_policy_key": ["short_default_clean_path"] * 4,
            "mkt_median_oi_chg_4h_rz": [1.0, 3.0, 999.0, 2.0],
        }
    ).to_parquet(path)
    result = _daily_state_part_streaming(path, ["mkt_median_oi_chg_4h_rz"])
    assert result["mkt_median_oi_chg_4h_rz"].tolist() == [2.0, 2.0]


def test_taxonomy_and_controls_operate_on_full_blocks() -> None:
    calendar = _calendar()
    daily = daily_observable_state(
        _states(),
        features=["mkt_median_oi_chg_4h_rz", "market_breadth_recovery_from_24h_min"],
    )
    config = BlockTaxonomyConfig(min_reference_days=10, controls_per_block=2)
    blocks, trajectories = build_block_taxonomy(calendar, daily, config=config)
    assert len(blocks) == 2
    assert blocks["event_days"].tolist() == [2.0, 1.0]
    assert {
        "pre__mkt_median_oi_chg_4h_rz",
        "onset_delta__mkt_median_oi_chg_4h_rz",
    }.issubset(trajectories.columns)
    controls = matched_benign_block_controls(calendar, daily, blocks, config=config)
    assert not controls.empty
    assert (controls["control_end"] < controls["event_start"]).all()


def test_taxonomy_includes_model_attribution_shift_mechanism() -> None:
    states = _states()
    states["base_attr_signed__latent_state"] = np.repeat(
        np.linspace(-1.0, 1.0, 48, dtype=np.float32), 2
    )
    daily = daily_observable_state(
        states,
        features=[
            "mkt_median_oi_chg_4h_rz",
            "base_attr_signed__latent_state",
        ],
    )

    blocks, _ = build_block_taxonomy(
        _calendar(), daily, config=BlockTaxonomyConfig(min_reference_days=10)
    )

    assert "family__model_attribution_shift__active_mean_z" in blocks.columns
    assert "family__model_attribution_shift__onset_abs_delta" in blocks.columns
    assert (
        "family__attribution_vector__active__signed__latent_state" in blocks.columns
    )
    assert (
        "family__attribution_vector__sign_flip__signed__latent_state"
        in blocks.columns
    )


def test_taxonomy_keeps_error_shape_as_separate_expost_view() -> None:
    calendar = _calendar()
    calendar["expost__loss_rate"] = np.linspace(0.1, 0.9, len(calendar))
    calendar["expost__signed_hit_surprise"] = np.linspace(0.2, -0.2, len(calendar))
    daily = daily_observable_state(
        _states(),
        features=[
            "mkt_median_oi_chg_4h_rz",
            "market_breadth_recovery_from_24h_min",
        ],
    )

    blocks, _ = build_block_taxonomy(
        calendar, daily, config=BlockTaxonomyConfig(min_reference_days=10)
    )

    assert "family__error_shape__active_mean_z" in blocks.columns
    assert "family__error_shape__onset_abs_delta" in blocks.columns
    assert "family__error_vector__active__loss_rate" in blocks.columns
    assert "family__error_vector__onset_delta__signed_hit_surprise" in blocks.columns
    assert "calendar_error__loss_rate" in blocks.columns
    assert "calendar_error__signed_hit_surprise" in blocks.columns


def test_detector_coverage_is_reported_by_whole_event_block() -> None:
    calendar = _calendar()
    daily = daily_observable_state(
        _states(),
        features=["mkt_median_oi_chg_4h_rz", "market_breadth_recovery_from_24h_min"],
    )
    blocks, _ = build_block_taxonomy(
        calendar, daily, config=BlockTaxonomyConfig(min_reference_days=10)
    )
    detector = pd.DataFrame(
        {
            "day": pd.date_range("2026-01-01", periods=48, freq="D", tz="UTC"),
            "side_name": "short",
            "archetype_policy_key": "short_default_clean_path",
            "risk": 0.1,
        }
    )
    detector.loc[detector["day"].eq(pd.Timestamp("2026-02-05", tz="UTC")), "risk"] = (
        0.95
    )
    result = attach_detector_block_coverage(
        blocks, detector, risk_column="risk", threshold=0.90
    )
    assert result["detector_recognized"].tolist() == [True, False]
    assert result["detector_assessable"].all()
    contrasts = detector_recognized_missed_contrasts(result)
    assert not contrasts.empty
    assert set(contrasts["recognized_blocks"]) == {1}
    assert contrasts["robust_standardized_difference"].abs().le(8.0).all()


def test_detector_contrasts_skip_features_missing_in_either_population() -> None:
    blocks = pd.DataFrame(
        {
            "side_name": ["long", "long"],
            "archetype_policy_key": ["long_test", "long_test"],
            "detector_assessable": [True, True],
            "detector_recognized": [True, False],
            "family__available": [2.0, 1.0],
            "family__missing": [np.nan, np.nan],
        }
    )

    contrasts = detector_recognized_missed_contrasts(blocks)

    assert contrasts["feature"].tolist() == ["family__available"]


def test_sparse_event_calendar_overlays_full_daily_calendar(tmp_path) -> None:
    daily = _calendar()
    daily["adverse_calendar_cell"] = 0
    events = daily.iloc[[35, 36]][["day", "side_name", "archetype_policy_key"]].copy()
    events["adverse_event_rows"] = [3, 4]
    path = tmp_path / "events.csv"
    events.to_csv(path, index=False)
    result = _overlay_event_calendar(daily, [path])
    assert result["adverse_calendar_cell"].sum() == 2
    assert (
        result.loc[
            result["day"].eq(events.iloc[0]["day"]), "adverse_calendar_cell"
        ].iloc[0]
        == 1
    )


def test_sparse_event_calendar_keeps_event_metrics(tmp_path) -> None:
    daily = _calendar()
    daily["adverse_calendar_cell"] = 0
    daily["mean_ev_after_1pct"] = np.nan
    events = daily.iloc[[35]][["day", "side_name", "archetype_policy_key"]].copy()
    events["adverse_event_rows"] = 3
    events["mean_ev_after_1pct"] = -0.031
    path = tmp_path / "events.csv"
    events.to_csv(path, index=False)
    result = _overlay_event_calendar(daily, [path])
    value = result.loc[
        result["adverse_calendar_cell"].eq(1), "mean_ev_after_1pct"
    ].iloc[0]
    assert value == -0.031


def test_cluster_count_rejects_singleton_family() -> None:
    matrix = np.array([[0.0], [0.1], [0.2], [0.3], [0.4], [9.0]], dtype=np.float32)
    assert (
        taxonomy._choose_cluster_count(matrix, maximum=2, minimum_cluster_blocks=3) == 0
    )


def test_robust_z_keeps_unavailable_historical_feature_missing() -> None:
    reference = np.array([[1.0, np.nan], [2.0, np.nan]], dtype=np.float32)
    values = np.array([[1.5, 8.0]], dtype=np.float32)

    result = taxonomy._robust_z(values, reference)

    assert np.isfinite(result[0, 0])
    assert np.isnan(result[0, 1])
