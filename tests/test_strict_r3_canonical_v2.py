from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from extreme_price_movements.strict_r3_canonical_v2 import (
    GEOMETRY_END,
    GEOMETRY_START,
    GEOMETRY_TARGET_POLICY_RESIDUAL,
    POLICY_RESIDUAL_GEOMETRY_SCHEMA,
    CandidateSpec,
    ScoreReference,
    assert_scoring_frame_is_target_free,
    build_point_in_time_candidates,
    fit_frozen_geometry_k9,
    fit_policy_net_map,
    load_geometry_bundle,
    persist_geometry_bundle,
    require_single_geometry_hash,
    residual_grades,
)


def _market() -> pd.DataFrame:
    timestamp = pd.Timestamp("2026-08-01 00:00", tz="UTC")
    return pd.DataFrame({
        "__ts__": [timestamp, timestamp],
        "__decision_ts__": [timestamp + pd.Timedelta(hours=1)] * 2,
        "__symbol__": ["A/USD", "B/USD"],
        "instrument_available": [True, True],
        "spread_bps": [10.0, 500.0],
        "entry_executable": [True, True],
        "feature_a": [1.0, 9.0],
        "market_source": [1.0, 9.0],
        "future_return_12h": [1000.0, -1000.0],
    })


def test_future_path_mutation_cannot_change_candidates_or_features() -> None:
    first = _market()
    second = first.copy()
    second["future_return_12h"] *= -100.0
    kwargs = dict(
        universe=["A/USD", "B/USD"], feature_fields=["feature_a"],
        cross_sectional_sources=["market_source"],
        spec=CandidateSpec(spread_limit_bps=100.0, side_names=("long",)),
    )
    population_a, eligible_a, rejected_a = build_point_in_time_candidates(first, **kwargs)
    population_b, eligible_b, rejected_b = build_point_in_time_candidates(second, **kwargs)
    pd.testing.assert_frame_equal(population_a, population_b)
    pd.testing.assert_frame_equal(eligible_a, eligible_b)
    pd.testing.assert_frame_equal(rejected_a, rejected_b)


def test_cross_sectional_features_precede_candidate_filtering() -> None:
    population, eligible, rejected = build_point_in_time_candidates(
        _market(), universe=["A/USD", "B/USD"], feature_fields=["feature_a"],
        cross_sectional_sources=["market_source"],
        spec=CandidateSpec(spread_limit_bps=100.0, side_names=("long",)),
    )
    assert len(population) == 2
    assert eligible["__symbol__"].tolist() == ["A/USD"]
    assert rejected["eligibility_reason"].tolist() == ["spread_above_frozen_limit"]
    # A is rank 0.5 only if the rejected B row participated in the complete
    # point-in-time cross-section.
    assert eligible["xs__market_source__rank"].iloc[0] == pytest.approx(0.5)


def test_scoring_contract_rejects_future_or_outcome_fields() -> None:
    with pytest.raises(ValueError, match="outcome/future"):
        assert_scoring_frame_is_target_free(pd.DataFrame({"candidate_id": ["x"], "policy_net_bps": [1.0]}))
    assert_scoring_frame_is_target_free(pd.DataFrame({"candidate_id": ["x"], "causal_feature": [1.0]}))


def test_empirical_reference_is_held_population_invariant() -> None:
    reference = ScoreReference.fit([0.0, 1.0, 2.0, 3.0], source="prior42")
    first = reference.cdf([1.5])
    second = reference.cdf([1.5, -1e9, 1e9])[:1]
    np.testing.assert_allclose(first, second)


def test_pooled_replay_requires_one_frozen_geometry_hash() -> None:
    frame = pd.DataFrame({"geometry_bundle_sha256": ["frozen", "frozen"]})
    assert require_single_geometry_hash(frame) == "frozen"
    with pytest.raises(ValueError, match="one identical frozen geometry"):
        require_single_geometry_hash(
            pd.DataFrame({"geometry_bundle_sha256": ["oct_dec", "monthly_refit"]})
        )
    with pytest.raises(ValueError, match="is null"):
        require_single_geometry_hash(
            pd.DataFrame({"geometry_bundle_sha256": ["frozen", None]})
        )


def test_policy_map_is_monotone_and_residual_bands_are_frozen() -> None:
    rank = np.linspace(0.0, 1.0, 400)
    target = -50.0 + 200.0 * rank + 5.0 * np.sin(rank * 20.0)
    mapping = fit_policy_net_map(rank, target)
    prediction = mapping.predict(rank)
    assert np.all(np.diff(prediction) >= -1e-12)
    assert residual_grades([-151, -150, -149, -50, 0, 50, 51, 150, 151]).tolist() == [0, 0, 1, 1, 2, 2, 3, 3, 4]


def _geometry_warmup() -> pd.DataFrame:
    rng = np.random.default_rng(91)
    rows: list[pd.DataFrame] = []
    for month, start in enumerate(("2024-10-01", "2024-11-01", "2024-12-01")):
        count = 420
        timestamp = pd.date_range(start, periods=count, freq="h", tz="UTC")
        x1 = rng.normal(loc=month * 0.1, size=count)
        x2 = rng.normal(size=count)
        anchor = rng.normal(0.0, 10.0, count)
        net = anchor + np.where(x1 + 0.2 * x2 > 0.0, 150.0, -150.0)
        rows.append(pd.DataFrame({
            "candidate_id": [f"{month}-{index}" for index in range(count)],
            "__decision_ts__": timestamp,
            "h12_label_valid": True,
            "h12_label_available_ts": timestamp + pd.Timedelta(hours=12),
            "h12_tp6_sl4_net_bps": net,
            "policy_label_available_ts": timestamp + pd.Timedelta(hours=12),
            "policy_net_bps": net + np.where(x2 > 0.0, 80.0, -80.0),
            "prequential_base_anchor_bps": anchor,
            "stack_is_prequential": True,
            "geometry_definition_population_complete": True,
            "f1": x1, "f2": x2,
        }))
    return pd.concat(rows, ignore_index=True)


def test_geometry_is_one_frozen_multimonth_bundle_not_a_monthly_refit(tmp_path) -> None:
    warmup = _geometry_warmup()
    geometry = fit_frozen_geometry_k9(warmup, encoder_fields=["f1", "f2"])
    assert geometry.fit_audit["definition_start"] == GEOMETRY_START.isoformat()
    assert geometry.fit_audit["definition_end_exclusive"] == GEOMETRY_END.isoformat()
    assert set(geometry.fit_audit["month_rows"]) == {"2024-10", "2024-11", "2024-12"}
    directory = tmp_path / "geometry_v2"
    manifest = persist_geometry_bundle(geometry, directory)
    loaded = load_geometry_bundle(directory)
    assert loaded.bundle_sha256 == manifest["bundle_sha256"]
    assert loaded.fit_audit == geometry.fit_audit
    # Monthly trainers require this persisted hash and have no warm-up/refit
    # argument; all consuming bundles therefore share the same semantics.
    assert loaded.bundle_sha256
    assert len(loaded.structural_fields) == 60
    assert len(loaded.severe_structural_fields) == 45
    assert loaded.cluster_fit_support is not None
    assert loaded.cluster_distance_mean is not None
    assert loaded.cluster_membership_covariance is not None
    assert loaded.cluster_structural_covariance is not None
    assert loaded.fit_audit["within_cluster_geometry_break"]["fit_population"] == "complete_oct_dec_2024_definition"


def test_k9_weighted_structural_state_is_causal_and_contains_no_raw_cluster_slot() -> None:
    geometry = fit_frozen_geometry_k9(_geometry_warmup(), encoder_fields=["f1", "f2"])
    current = pd.DataFrame(
        {
            "__decision_ts__": pd.to_datetime(
                ["2025-01-02T00:00:00Z", "2025-01-02T00:00:00Z"],
            ),
            "f1": [0.1, -0.2], "f2": [0.3, -0.4],
        }
    )
    future = pd.DataFrame(
        {
            "__decision_ts__": pd.to_datetime(["2025-02-02T00:00:00Z"]),
            "f1": [99.0], "f2": [-99.0],
        }
    )
    first = geometry.transform(current)
    with_future = geometry.transform(pd.concat([current, future], ignore_index=True)).iloc[: len(current)]
    weighted = [column for column in first if column.startswith("k9_cluster_")]
    assert len(weighted) == 15
    assert {
        "k9_cluster_activation_weighted_within_cov_break_train",
        "k9_cluster_activation_weighted_within_corr_break_train",
        "k9_cluster_activation_weighted_within_support_train",
    }.issubset(weighted)
    assert not any("k09__cluster_" in column for column in weighted)
    assert np.isfinite(first.loc[:, weighted].to_numpy(float)).all()
    pd.testing.assert_frame_equal(first, with_future)
    # Timestamp cross-sectional state is identical for the contemporaneous
    # candidates; it is not a per-leaf value.
    assert first["k9_cluster_timestamp_cov_break_train"].nunique() == 1


def test_geometry_rejects_labels_not_available_before_definition_end() -> None:
    warmup = _geometry_warmup()
    december = warmup["__decision_ts__"].dt.month.eq(12)
    warmup.loc[december, "h12_label_available_ts"] = pd.Timestamp("2025-02-01", tz="UTC")
    with pytest.raises(ValueError, match="all three declared months"):
        fit_frozen_geometry_k9(warmup, encoder_fields=["f1", "f2"])


def test_policy_residual_geometry_is_prequential_and_explicitly_noncanonical(tmp_path) -> None:
    warmup = _geometry_warmup()
    december = warmup["__decision_ts__"].dt.month.eq(12)
    # The policy mode must reject labels not resolved by the end of its frozen
    # Oct--Dec definition window, independently of H12 label availability.
    warmup.loc[december, "policy_label_available_ts"] = pd.Timestamp("2025-02-01", tz="UTC")
    with pytest.raises(ValueError, match="all three declared months"):
        fit_frozen_geometry_k9(
            warmup, encoder_fields=["f1", "f2"],
            target_mode=GEOMETRY_TARGET_POLICY_RESIDUAL,
        )
    warmup = _geometry_warmup()
    geometry = fit_frozen_geometry_k9(
        warmup, encoder_fields=["f1", "f2"],
        target_mode=GEOMETRY_TARGET_POLICY_RESIDUAL,
        policy_residual_hurdle_bps=50.0,
    )
    assert geometry.fit_audit["geometry_target_mode"] == GEOMETRY_TARGET_POLICY_RESIDUAL
    assert geometry.fit_audit["policy_residual_hurdle_bps"] == 50.0
    directory = tmp_path / "policy_residual_geometry"
    persist_geometry_bundle(geometry, directory, schema=POLICY_RESIDUAL_GEOMETRY_SCHEMA)
    with pytest.raises(ValueError, match="strict_r3_geometry_k9_oct_dec_2024_v2"):
        load_geometry_bundle(directory)
    loaded = load_geometry_bundle(directory, expected_schema=POLICY_RESIDUAL_GEOMETRY_SCHEMA)
    assert loaded.fit_audit["geometry_target_mode"] == GEOMETRY_TARGET_POLICY_RESIDUAL


def test_episodic_geometry_has_a_distinct_explicit_loader_contract(tmp_path) -> None:
    """A periodic research fit can never masquerade as the canonical parent."""

    warmup = _geometry_warmup().copy()
    warmup["__decision_ts__"] = warmup["__decision_ts__"] + pd.DateOffset(months=3)
    warmup["h12_label_available_ts"] = (
        warmup["h12_label_available_ts"] + pd.DateOffset(months=3)
    )
    geometry = fit_frozen_geometry_k9(
        warmup,
        encoder_fields=["f1", "f2"],
        definition_start="2025-01-01 00:00:00+00:00",
        definition_end_exclusive="2025-04-01 00:00:00+00:00",
    )
    directory = tmp_path / "episode_geometry"
    manifest = persist_geometry_bundle(
        geometry, directory, schema="strict_r3_geometry_k9_episode_isolated_v1",
    )
    with pytest.raises(ValueError, match="strict_r3_geometry_k9_oct_dec_2024_v2"):
        load_geometry_bundle(directory)
    loaded = load_geometry_bundle(
        directory,
        expected_schema="strict_r3_geometry_k9_episode_isolated_v1",
        definition_start="2025-01-01 00:00:00+00:00",
        definition_end_exclusive="2025-04-01 00:00:00+00:00",
    )
    assert loaded.bundle_sha256 == manifest["bundle_sha256"]
