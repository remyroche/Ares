from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import extreme_price_movements.strict_r3_canonical_current as canonical_current
from extreme_price_movements.strict_r3_canonical_current import (
    BUNDLE_SCHEMA,
    CONVERSION_BUNDLE_SCHEMA,
    CONSENSUS_BLEND_WEIGHT,
    BASE_BLEND_WEIGHT,
    CanonicalCurrentBundle,
    ConsensusHeadSpec,
    CorrectnessHead,
    FittedConsensusHead,
    FrozenGeometryK9View,
    FourWeekConversionBundle,
    CORRECTNESS_TRAIN_FRACTION,
    K9_TEMPERATURE_SCALE,
    OptimizedPolicyContract,
    LeafTrustBundle,
    SevereDiagnostic,
    load_conditional_consensus_contract,
    score_current_bundle,
    score_four_week_conversion_bundle,
    score_four_week_conversion_by_upstream_vintage,
)
from extreme_price_movements.strict_r3_canonical_v2 import ScoreReference


def test_frozen_geometry_history_is_strictly_pre_score() -> None:
    """A frozen K9 definition must not seed a historical score with its future state."""
    geometry = SimpleNamespace(parent=SimpleNamespace(state_history=pd.DataFrame({
        "__decision_ts__": pd.to_datetime([
            "2024-10-31T23:00:00Z", "2024-11-20T00:00:00Z",
            "2024-12-31T23:00:00Z",
        ], utc=True),
        **{f"k{index}": [1.0, 2.0, 3.0] for index in range(9)},
    })))
    history = canonical_current._initial_frozen_geometry_history(
        geometry, first_timestamp="2024-11-20T00:00:00Z",
    )
    assert history["__decision_ts__"].tolist() == [
        pd.Timestamp("2024-10-31T23:00:00Z"),
    ]


def test_geometry_history_rejects_replayed_score_hour() -> None:
    """The next target-free state append must not silently reuse an hour."""
    history = pd.DataFrame({
        "__decision_ts__": pd.to_datetime(["2025-01-01T00:00:00Z"], utc=True),
        **{f"k{index}": [1.0] for index in range(9)},
    })
    frame = pd.DataFrame({
        "__decision_ts__": pd.to_datetime(["2025-01-01T00:00:00Z"], utc=True),
    })
    state = pd.DataFrame({
        **{f"k09__cluster_{index:02d}__membership": [1.0 / 9.0] for index in range(9)},
    })
    with pytest.raises(AssertionError, match="repeated a scored timestamp"):
        canonical_current._append_geometry_history(
            history, frame=frame, geometry_state=state,
        )


class _BaseModel:
    classes_ = np.asarray([0, 1, 2])

    def predict_proba(self, matrix: np.ndarray) -> np.ndarray:
        clear = np.clip(0.2 + 0.1 * matrix[:, 0], 0.05, 0.85)
        adverse = np.clip(0.45 - 0.05 * matrix[:, 0], 0.05, 0.80)
        weak = 1.0 - clear - adverse
        return np.column_stack([adverse, weak, clear])


class _Map:
    def predict(self, values: np.ndarray) -> np.ndarray:
        return 200.0 * np.asarray(values, dtype=float) - 100.0


class _RankModel:
    def __init__(self, sign: float = 1.0) -> None:
        self.sign = sign

    def predict(self, matrix: np.ndarray) -> np.ndarray:
        return self.sign * matrix[:, 0]


class _LeafModel:
    def predict(self, matrix: np.ndarray, pred_leaf: bool = False) -> np.ndarray:
        assert pred_leaf
        return np.asarray([[0, 1], [1, 0]], dtype=np.int32)[: len(matrix)]


def test_conversion_cdf_reference_uses_same_upstream_vintage(monkeypatch) -> None:
    """A monthly producer must never CDF-normalise against another vintage."""
    class _Conversion:
        cutoff = pd.Timestamp("2026-07-16T00:00:00Z")

    class _Upstream:
        manifest = {"bundle_sha256": "august-upstream"}

    calls: list[tuple[bool, object]] = []

    def _score_monthly(bundle, frame, *, allow_prior_reference=False, prior_reference_start=None):
        assert bundle is upstream
        calls.append((allow_prior_reference, prior_reference_start))
        output = frame[["candidate_id", "__decision_ts__", "side_name"]].copy()
        output["upstream_bundle_sha256"] = "august-upstream"
        return output

    def _score_conversion(bundle, *, reference, held, precomputed_state=None):
        assert bundle is conversion
        assert set(reference["upstream_bundle_sha256"]) == {"august-upstream"}
        assert set(held["upstream_bundle_sha256"]) == {"august-upstream"}
        assert precomputed_state is not None
        assert set(precomputed_state["candidate_id"]) == {"ref", "held"}
        return pd.concat([
            reference.assign(__score_role__="reference"),
            held.assign(__score_role__="held"),
        ], ignore_index=True), pd.DataFrame([{}])

    monkeypatch.setattr(canonical_current, "score_monthly_upstream_bundle", _score_monthly)
    monkeypatch.setattr(canonical_current, "score_four_week_conversion_bundle", _score_conversion)
    conversion, upstream = _Conversion(), _Upstream()
    reference = pd.DataFrame({
        "candidate_id": ["ref"],
        "__decision_ts__": pd.to_datetime(["2026-06-20T00:00:00Z"], utc=True),
        "side_name": ["long"],
    })
    held = pd.DataFrame({
        "candidate_id": ["held"],
        "__decision_ts__": pd.to_datetime(["2026-08-01T00:00:00Z"], utc=True),
        "side_name": ["long"],
    })
    monkeypatch.setattr(
        canonical_current,
        "_precompute_conversion_state",
        lambda _bundle, *, reference, held: pd.concat([
            reference[["candidate_id", "__decision_ts__", "side_name"]],
            held[["candidate_id", "__decision_ts__", "side_name"]],
        ], ignore_index=True).assign(state=0.0),
    )

    scored, audit = score_four_week_conversion_by_upstream_vintage(
        conversion, reference=reference, held=held,
        upstream_bundles={"2026-08": upstream},
    )

    assert calls == [
        (False, None),
        (True, pd.Timestamp("2026-06-18T00:00:00Z")),
    ]
    assert set(scored["cdf_reference_upstream_bundle_sha256"]) == {"august-upstream"}
    assert audit["same_upstream_bundle_for_reference_and_held"].tolist() == [True]


def test_correctness_head_fit_is_bit_reproducible() -> None:
    """Identical causal inputs must not alter a persisted conversion score."""
    rows = 4_000
    timestamp = pd.date_range("2025-01-01", periods=rows, freq="h", tz="UTC")
    frame = pd.DataFrame(
        {
            "candidate_id": [f"row-{index:04d}" for index in range(rows)],
            "__decision_ts__": timestamp,
            "upstream": np.linspace(0.0, 1.0, rows),
            "base_anchor_bps": np.zeros(rows),
            "policy_net_bps": np.where(np.arange(rows) % 2, 200.0, 0.0),
            "feature": np.sin(np.arange(rows) / 17.0),
        }
    )
    left = canonical_current._fit_correctness(frame, ("feature",))
    right = canonical_current._fit_correctness(frame, ("feature",))
    matrix = canonical_current._numeric_matrix(
        frame, ("feature",), left.medians,
    )
    np.testing.assert_array_equal(left.model.predict(matrix), right.model.predict(matrix))


def test_severe_diagnostic_excludes_geometry_definition_rows(monkeypatch) -> None:
    """Oct--Dec geometry-fit rows must never train supervised Severe-200."""
    captured: dict[str, np.ndarray] = {}

    class _Classifier:
        def __init__(self, **_kwargs) -> None:
            pass

        def fit(self, matrix: np.ndarray, target: np.ndarray):
            captured["matrix"] = np.asarray(matrix)
            captured["target"] = np.asarray(target)
            return self

    monkeypatch.setattr(canonical_current, "LGBMClassifier", _Classifier)
    definition_rows = 1_100
    older_rows = 1_100
    frame = pd.DataFrame({
        "__decision_ts__": pd.to_datetime(
            ["2024-11-01T00:00:00Z"] * definition_rows
            + ["2024-09-01T00:00:00Z"] * older_rows,
            utc=True,
        ),
        "h12_label_available_ts": pd.to_datetime(
            ["2024-11-01T12:00:00Z"] * definition_rows
            + ["2024-09-01T12:00:00Z"] * older_rows,
            utc=True,
        ),
        "h12_label_valid": True,
        "h12_tp6_sl4_net_bps": np.tile([-250.0, 25.0], (definition_rows + older_rows) // 2),
        "feature": np.r_[np.full(definition_rows, 9_999.0), np.arange(older_rows)],
    })
    diagnostic = canonical_current._fit_severe_diagnostic(
        frame, ("feature",), cutoff=pd.Timestamp("2025-01-01T00:00:00Z"),
    )
    assert diagnostic.model is not None
    assert len(captured["target"]) == older_rows
    assert not np.isclose(captured["matrix"], 9_999.0).all(axis=1).any()


def test_four_week_manifest_declares_geometry_definition_exclusion() -> None:
    source = Path(canonical_current.__file__).read_text()
    marker = '"geometry_definition_rows_excluded_from_severe": True'
    # Both the single-bundle research path and exact four-week producer must
    # persist the same supervised-exclusion receipt.
    assert source.count(marker) == 2


def test_compact_consensus_sampling_preserves_canonical_query_choice(monkeypatch) -> None:
    """Identity-first sampling must select exactly the legacy capped queries."""
    monkeypatch.setattr(canonical_current, "MODEL_CAP", 40)
    rows = 120
    timestamp = pd.date_range("2025-01-01", periods=rows // 3, freq="h", tz="UTC")
    frame = pd.DataFrame({
        "candidate_id": [f"row-{index:04d}" for index in range(rows)],
        "__decision_ts__": np.repeat(timestamp.to_numpy(), 3),
        "side_name": "long",
        "field": np.linspace(-1.0, 1.0, rows),
    })
    compact = frame.loc[:, ["candidate_id", "__decision_ts__", "side_name"]].copy()
    compact["__ledger_position__"] = np.arange(rows, dtype=np.int64)
    spec = ConsensusHeadSpec(
        name="test", cap=1, weight_mode="ordinary", query="exact_timestamp_side",
        fields=("field",), target_edges_bps=(-100.0, -25.0, 25.0, 100.0), params={},
    )
    grade = (np.arange(rows) % 5).astype(np.int32)
    wide_sample, wide_target, wide_groups = canonical_current._sample_complete_consensus_queries(
        frame, grade, spec, seed=73,
    )
    compact_sample, compact_target, compact_groups = canonical_current._sample_complete_consensus_queries(
        compact, grade, spec, seed=73,
    )
    assert compact_sample["candidate_id"].tolist() == wide_sample["candidate_id"].tolist()
    np.testing.assert_array_equal(compact_target, wide_target)
    np.testing.assert_array_equal(compact_groups, wide_groups)


def test_leaf_support_is_genuinely_contribution_weighted_and_joint_ood_is_distinct() -> None:
    bundle = LeafTrustBundle(
        fields=("f",), medians=np.asarray([0.0]), model=_LeafModel(),
        support_counts=(np.asarray([10.0, 100.0]), np.asarray([20.0, 200.0])),
        train_rows=1_000,
        leaf_values=(np.asarray([1.0, 3.0]), np.asarray([2.0, 4.0])),
    )
    state = bundle.transform(pd.DataFrame({"f": [0.0, 1.0]}))
    assert state.loc[0, "leaf_support_contribution_weighted"] == pytest.approx(
        (10.0 * 1.0 + 200.0 * 4.0) / 5.0
    )
    assert not np.allclose(state["leaf_ood_joint"], state["leaf_ood_marginal"])


def test_leaf_trust_exposes_stable_latest_fit_active_rule_state() -> None:
    """Latest-fit leaf semantics are aggregated, never raw leaf identities."""
    bundle = LeafTrustBundle(
        fields=("f",), medians=np.asarray([0.0]), model=_LeafModel(),
        support_counts=(np.asarray([10.0, 100.0]), np.asarray([20.0, 200.0])),
        train_rows=1_000,
        leaf_values=(np.asarray([1.0, 3.0]), np.asarray([2.0, 4.0])),
        leaf_feature_paths=(
            np.asarray([[1.0], [0.0]], dtype=np.float32),
            np.asarray([[0.0], [1.0]], dtype=np.float32),
        ),
        rule_activation_mean=np.asarray([0.5]),
        rule_activation_covariance=np.asarray([[0.25]]),
        rule_activation_correlation=np.asarray([[1.0]]),
    )
    state = bundle.transform(pd.DataFrame({
        "f": [0.0, 1.0],
        "__decision_ts__": pd.to_datetime([
            "2025-01-01T00:00:00Z", "2025-01-01T00:00:00Z",
        ], utc=True),
    }))
    expected = {
        "active_rule_candidate_mahalanobis_train",
        "active_rule_timestamp_cov_break_train",
        "active_rule_timestamp_corr_break_train",
        "active_rule_timestamp_mahalanobis_train",
        "active_rule_timestamp_support_weighted",
        "active_rule_timestamp_ood_weighted",
    }
    assert expected.issubset(state.columns)
    assert not any("leaf_index" in column for column in state.columns)
    assert np.isfinite(state.loc[:, list(expected)].to_numpy(float)).all()


class _Geometry(FrozenGeometryK9View):
    """Minimal immutable-geometry view fixture; no rolling stand-in is valid."""

    def __init__(self) -> None:
        pass

    @property
    def bundle_sha256(self) -> str:
        return "geometry-view-hash"

    @property
    def parent_bundle_sha256(self) -> str:
        return "geometry-parent-hash"

    @property
    def definition_start(self) -> str:
        return "2024-10-01T00:00:00+00:00"

    @property
    def definition_end_exclusive(self) -> str:
        return "2025-01-01T00:00:00+00:00"

    temperature_scale = K9_TEMPERATURE_SCALE

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "k09__cluster_00__membership": np.full(len(frame), 0.5),
                "k09__cluster_01__membership": np.full(len(frame), 0.5),
                "k09__cluster_00__negative_distance": -np.ones(len(frame)),
                "k09__cluster_01__negative_distance": -2.0 * np.ones(len(frame)),
                "path_support_effective_28d": np.full(len(frame), 20.0),
                "model_ood_marginal": np.full(len(frame), 0.2),
                "model_drift_prototype_psi": np.full(len(frame), 0.3),
                "geometry_cov_break_train": np.linspace(0.0, 1.0, len(frame)),
            },
            index=frame.index,
        )


class _Leaf:
    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {"leaf_support_effective": np.arange(len(frame), dtype=float) + 1.0},
            index=frame.index,
        )


def _base_fields() -> list[str]:
    payload = json.loads(
        open("config/strict_r3_canonical_v2_feature_contract.json").read()
    )
    return payload["base_fields_by_side"]["long"]


def test_checked_in_conditional_consensus_is_exact_frozen_contract() -> None:
    fields = _base_fields()
    specs = load_conditional_consensus_contract(fields)
    assert len(specs) == 10
    assert sum(spec.query == "exact_timestamp_side" for spec in specs) == 6
    assert sum(spec.query == "cycle_4h_side" for spec in specs) == 4
    assert [len(spec.fields) for spec in specs] == [40, 40, 15, 30, 80, 80, 100, 100, 120, 51]
    assert all(spec.target_edges_bps == (-150.0, -50.0, 50.0, 150.0) for spec in specs)
    assert all(set(spec.fields).issubset(fields) for spec in specs)


@pytest.mark.parametrize("weight_mode", ["ordinary", "equal_month"])
def test_consensus_cap_keeps_complete_queries(
    monkeypatch: pytest.MonkeyPatch, weight_mode: str,
) -> None:
    monkeypatch.setattr(canonical_current, "MODEL_CAP", 20)
    timestamps = np.repeat(
        pd.date_range("2025-01-01", periods=15, freq="h", tz="UTC").to_numpy(),
        3,
    )
    frame = pd.DataFrame({
        "candidate_id": [f"candidate-{index}" for index in range(len(timestamps))],
        "__decision_ts__": timestamps,
        "side_name": "long",
        "f0": np.arange(len(timestamps), dtype=float),
    })
    grade = np.resize(np.asarray([0, 2, 4], dtype=np.int32), len(frame))
    spec = ConsensusHeadSpec(
        name=f"test_{weight_mode}", cap=40, weight_mode=weight_mode,
        query="exact_timestamp_side", fields=("f0",),
        target_edges_bps=(-150.0, -50.0, 50.0, 150.0), params={},
    )
    sampled, target, groups = canonical_current._sample_complete_consensus_queries(
        frame, grade, spec, seed=123,
    )
    assert len(sampled) <= 20
    assert len(sampled) == len(target) == int(groups.sum())
    assert set(groups.tolist()) == {3}
    assert sampled.groupby("__query__").size().eq(3).all()


def test_optimized_policy_is_the_selected_simple_policy_winner() -> None:
    policy = OptimizedPolicyContract()
    assert policy.stop_loss_atr == pytest.approx(4.1520006)
    assert policy.trailing_activation_atr == pytest.approx(2.3262249)
    assert policy.trailing_giveback_atr == pytest.approx(0.1023720)
    assert policy.timeout_hours == 12
    assert policy.cost_bps_once == 100.0


def test_episode_isolated_admission_never_mixes_geometry_support(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[set[str]] = []

    def _stub(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        calls.append(set(frame["geometry_bundle_sha256"].astype(str)))
        output = frame.copy()
        output["causal_21d_side_expected_net_bps"] = 75.0
        return output, pd.DataFrame({"rows": [len(frame)]})

    monkeypatch.setattr(canonical_current, "apply_current_admission", _stub)
    source = pd.DataFrame({
        "candidate_id": ["a", "b", "c"],
        "geometry_bundle_sha256": ["episode-a", "episode-b", "episode-a"],
    })
    output, audit = canonical_current.apply_current_admission_by_geometry(
        source, geometry_mode="episode-isolated",
    )
    assert calls == [{"episode-a"}, {"episode-b"}]
    assert output["candidate_id"].tolist() == ["a", "b", "c"]
    assert sorted(audit["geometry_bundle_sha256"].tolist()) == ["episode-a", "episode-b"]


def _fitted_heads(fields: tuple[str, ...], *, shadow: bool) -> tuple[FittedConsensusHead, ...]:
    output = []
    for index in range(10):
        spec = ConsensusHeadSpec(
            name=("shadow" if shadow else "conditional") + f"_{index}",
            cap=40,
            weight_mode="ordinary",
            query="cycle_4h_side",
            fields=(fields[0],),
            target_edges_bps=(-150.0, -50.0, 50.0, 150.0),
            params={},
        )
        output.append(
            FittedConsensusHead(
                spec,
                np.asarray([0.0], dtype=np.float32),
                _RankModel(-1.0 if shadow else 1.0),
                ScoreReference.fit([-3.0, -1.0, 0.0, 1.0, 3.0], source=spec.name),
            )
        )
    return tuple(output)


def _bundle() -> CanonicalCurrentBundle:
    fields = tuple(f"f{index}" for index in range(120))
    correctness = CorrectnessHead(
        fields=("upstream", "k9_entropy", "leaf_support_effective"),
        medians=np.zeros(3, dtype=np.float32),
        model=_RankModel(1.0),
        score_reference=ScoreReference.fit([-1.0, 0.0, 0.5, 1.0], source="correctness"),
    )
    return CanonicalCurrentBundle(
        cutoff=pd.Timestamp("2026-08-01", tz="UTC"),
        held_end_exclusive=pd.Timestamp("2026-08-29", tz="UTC"),
        base_fields=fields,
        base_medians=np.zeros(120, dtype=np.float32),
        base_model=_BaseModel(),
        policy_net_map=_Map(),
        conditional_heads=_fitted_heads(fields, shadow=False),
        ordinary_shadow_heads=_fitted_heads(fields, shadow=True),
        geometry=_Geometry(),
        leaf_trust=_Leaf(),
        correctness=correctness,
        severe_diagnostic=SevereDiagnostic(correctness.fields, np.zeros(3), None),
        schema=BUNDLE_SCHEMA,
        manifest={"bundle_sha256": "bundle-hash"},
    )


def test_lockstep_upstream_bundle_requires_an_exact_28_day_held_window() -> None:
    """Explicit lock-step producers cannot silently become partial/monthly fits."""
    fields = tuple(f"f{index}" for index in range(120))
    common = dict(
        cutoff=pd.Timestamp("2026-08-01", tz="UTC"),
        base_fields=fields,
        base_medians=np.zeros(120, dtype=np.float32),
        base_model=_BaseModel(),
        base_score_reference=ScoreReference.fit([0.0, 1.0], source="test"),
        policy_net_map=_Map(),
        conditional_heads=_fitted_heads(fields, shadow=False),
        ordinary_shadow_heads=_fitted_heads(fields, shadow=True),
        manifest={"refit_cadence": "explicit_lockstep_window"},
    )
    canonical_current.MonthlyUpstreamBundle(
        end_exclusive=pd.Timestamp("2026-08-29", tz="UTC"),
        **common,
    )
    with pytest.raises(ValueError, match="exactly one 28-day window"):
        canonical_current.MonthlyUpstreamBundle(
            end_exclusive=pd.Timestamp("2026-08-28", tz="UTC"),
            **common,
        )


def _score_frame(start: str, rows: int) -> pd.DataFrame:
    timestamp = pd.date_range(start, periods=rows, freq="h", tz="UTC")
    output = pd.DataFrame(
        {
            "candidate_id": [f"{start}-{index}" for index in range(rows)],
            "__decision_ts__": timestamp,
            "side_name": "long",
        }
    )
    for index in range(120):
        output[f"f{index}"] = np.linspace(-1.0, 1.0, rows) + index * 0.001
    return output


def test_current_score_uses_conditional_consensus_and_correctness_not_severe() -> None:
    bundle = _bundle()
    reference = _score_frame("2026-07-04", 4)
    held = _score_frame("2026-08-01", 3)
    scored, audit = score_current_bundle(bundle, reference=reference, held=held)
    assert len(scored) == 7
    assert scored["final_score"].notna().all()
    assert scored["severe200_probability_shadow"].isna().all()
    assert not scored["severe_affects_final_score"].any()
    assert not np.allclose(
        scored["conditional_consensus_rank"],
        scored["ordinary_shadow_consensus_rank"],
    )
    expected = (
        BASE_BLEND_WEIGHT * scored["base_rank42"]
        + CONSENSUS_BLEND_WEIGHT * scored["conditional_consensus_rank"]
    )
    np.testing.assert_allclose(scored["upstream"], expected)
    assert audit.loc[0, "canonical_consensus"] == "conditional_usefulness_ten_head_v1"
    assert audit.loc[0, "ordinary_consensus"] == "shadow_rollback_only"
    assert not bool(audit.loc[0, "raw_k9_in_consensus"])
    assert not bool(audit.loc[0, "raw_k9_in_correctness"])
    assert not bool(audit.loc[0, "severe_affects_final_score"])
    assert audit.loc[0, "reference_start"] == pd.Timestamp("2026-07-04", tz="UTC")
    assert audit.loc[0, "final_reference"] == "same_bundle_prior28_correctness_score"
    assert "geometry_cov_break_train" in scored


def test_current_score_rejects_a_reference_row_outside_prior_28_days() -> None:
    bundle = _bundle()
    reference = _score_frame("2026-07-03 23:00", 1)
    held = _score_frame("2026-08-01", 1)
    with pytest.raises(ValueError, match="preceding 28-day"):
        score_current_bundle(bundle, reference=reference, held=held)


def test_current_bundle_rejects_raw_k9_in_correctness() -> None:
    bundle = _bundle()
    bundle.correctness.fields = ("k09__cluster_00__membership",)
    with pytest.raises(ValueError, match="raw K9"):
        bundle.__post_init__()


def _conversion_bundle() -> FourWeekConversionBundle:
    fields = tuple(f"f{index}" for index in range(120))
    correctness = CorrectnessHead(
        fields=("upstream", "k9_entropy", "leaf_support_effective"),
        medians=np.zeros(3, dtype=np.float32),
        model=_RankModel(1.0),
        score_reference=ScoreReference.fit([-1.0, 0.0, 0.5, 1.0], source="correctness"),
    )
    return FourWeekConversionBundle(
        cutoff=pd.Timestamp("2026-08-01", tz="UTC"),
        end_exclusive=pd.Timestamp("2026-08-29", tz="UTC"),
        base_fields=fields,
        geometry=_Geometry(),
        leaf_trust=_Leaf(),
        correctness=correctness,
        severe_diagnostic=SevereDiagnostic(correctness.fields, np.zeros(3), None),
        schema=CONVERSION_BUNDLE_SCHEMA,
        manifest={"bundle_sha256": "conversion-hash"},
    )


def _upstream_score_frame(start: str, rows: int) -> pd.DataFrame:
    output = _score_frame(start, rows)
    rank = np.linspace(0.1, 0.9, rows)
    output["base_score"] = rank - 0.5
    output["base_rank42"] = rank
    output["base_anchor_bps"] = 200.0 * rank - 100.0
    output["conditional_consensus_rank"] = rank[::-1]
    output["upstream"] = 0.75 * rank + 0.25 * rank[::-1]
    output["ordinary_shadow_consensus_rank"] = 0.5
    output["ordinary_shadow_upstream"] = 0.75 * rank + 0.125
    return output


def test_exact_production_conversion_uses_monthly_upstream_handoff() -> None:
    bundle = _conversion_bundle()
    reference = _upstream_score_frame("2026-07-04", 4)
    held = _upstream_score_frame("2026-08-01", 3)
    scored, audit = score_four_week_conversion_bundle(
        bundle, reference=reference, held=held,
    )
    assert scored["final_score"].notna().all()
    assert scored["severe200_probability_shadow"].isna().all()
    assert not scored["severe_affects_final_score"].any()
    assert bool(audit.loc[0, "upstream_scores_are_prequential_monthly"])
    assert bool(audit.loc[0, "same_conversion_model_reference_and_held"])
    assert not bool(audit.loc[0, "raw_k9_in_correctness"])
    assert audit.loc[0, "correctness_training_fraction"] == pytest.approx(0.30)
    assert audit.loc[0, "k9_temperature_scale"] == pytest.approx(0.25)
    # The downstream LDF receives only these stable geometry/leaf summaries;
    # raw K9 slots never leave the conversion scorer.
    assert {
        "k9_entropy", "k9_top2_margin", "k9_ood_distance",
        "k9_path_support_effective_28d", "k9_model_ood_marginal",
        "k9_model_drift_psi", "leaf_support_effective",
    }.issubset(scored.columns)
    assert not any(column.startswith("k09__cluster_") for column in scored.columns)


def test_conversion_preserves_target_free_symbol_identity_when_present() -> None:
    bundle = _conversion_bundle()
    reference = _upstream_score_frame("2026-07-04", 4)
    held = _upstream_score_frame("2026-08-01", 3)
    reference["__symbol__"] = ["BTC", "ETH", "SOL", "XRP"]
    held["__symbol__"] = ["ADA", "DOGE", "AVAX"]

    scored, _ = score_four_week_conversion_bundle(
        bundle, reference=reference, held=held,
    )

    assert "__symbol__" in scored
    assert scored.loc[scored["__score_role__"].eq("held"), "__symbol__"].tolist() == [
        "ADA", "DOGE", "AVAX",
    ]


def test_canonical_correctness_demotes_only_the_frozen_top30_domain() -> None:
    bundle = _conversion_bundle()
    bundle.correctness.training_score_floor = 0.55
    reference = _upstream_score_frame("2026-07-04", 4)
    held = _upstream_score_frame("2026-08-01", 4)
    scored, _ = score_four_week_conversion_bundle(
        bundle, reference=reference, held=held,
    )
    inactive = ~scored["correctness_gate_active"]
    active = scored["correctness_gate_active"]
    assert inactive.any() and active.any()
    np.testing.assert_allclose(
        scored.loc[inactive, "raw_correctness_demote"],
        scored.loc[inactive, "upstream"],
    )
    assert (
        scored.loc[active, "raw_correctness_demote"]
        <= scored.loc[active, "upstream"] + 1e-12
    ).all()
    assert bundle.correctness.training_fraction == CORRECTNESS_TRAIN_FRACTION
    assert bundle.geometry.temperature_scale == K9_TEMPERATURE_SCALE


def test_conversion_bundle_rejects_raw_k9_membership_input() -> None:
    bundle = _conversion_bundle()
    bundle.correctness.fields = ("k09__cluster_00__membership",)
    with pytest.raises(ValueError, match="raw K9"):
        bundle.__post_init__()


def test_current_admission_snapshot_never_supplies_a_current_outcome(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resolved = pd.DataFrame({
        "candidate_id": ["resolved"],
        "__decision_ts__": pd.to_datetime(["2026-07-01T00:00:00Z"], utc=True),
        "side_name": ["long"],
        "final_score": [0.8],
        "policy_net_bps": [125.0],
        "policy_label_available_ts": pd.to_datetime(["2026-07-01T12:00:00Z"], utc=True),
        "conversion_bundle_sha256": ["conversion-current"],
        "upstream_bundle_sha256": ["upstream-current"],
        "geometry_bundle_sha256": ["geometry-frozen"],
        "ev_score_family_id": ["family-frozen"],
        "stack_is_prequential": [True],
    })
    current = pd.DataFrame({
        "candidate_id": ["current"],
        "__decision_ts__": pd.to_datetime(["2026-08-01T00:00:00Z"], utc=True),
        "side_name": ["long"],
        "final_score": [0.9],
        "conversion_bundle_sha256": ["conversion-current"],
        "upstream_bundle_sha256": ["upstream-current"],
        "geometry_bundle_sha256": ["geometry-frozen"],
        "ev_score_family_id": ["family-frozen"],
        "stack_is_prequential": [True],
    })

    def _admit(frame: pd.DataFrame):
        current_rows = frame[frame["__current_admission_snapshot__"].fillna(False)]
        assert len(current_rows) == 1
        assert current_rows["policy_net_bps"].isna().all()
        assert (
            current_rows["policy_label_available_ts"]
            == current_rows["__decision_ts__"] + pd.Timedelta(hours=12)
        ).all()
        output = frame.copy()
        output["causal_21d_side_expected_net_bps"] = 75.0
        output["causal_21d_side_admitted_ge_50bps"] = True
        return output, pd.DataFrame({"snapshot_utc": [pd.Timestamp("2026-08-01", tz="UTC")]})

    monkeypatch.setattr(
        canonical_current, "_apply_current_admission_by_score_vintage", _admit,
    )
    output, audit = canonical_current.apply_current_admission_snapshot(
        resolved_score_ledger=resolved,
        current_scores=current,
    )
    assert output["candidate_id"].tolist() == ["current"]
    assert output["policy_net_bps"].isna().all()
    assert output["causal_21d_side_admitted_ge_50bps"].all()
    assert len(audit) == 1


def test_vintage_aware_admission_never_pools_upstream_refits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Same CDF scale cannot make a monthly base/meta refit poolable."""
    frame = pd.DataFrame({
        "candidate_id": ["old", "new"],
        "__decision_ts__": pd.to_datetime(
            ["2026-07-01T00:00:00Z", "2026-08-01T00:00:00Z"], utc=True,
        ),
        "side_name": ["long", "long"],
        "final_score": [0.95, 0.95],
        "policy_net_bps": [150.0, 10.0],
        "policy_label_available_ts": pd.to_datetime(
            ["2026-07-01T12:00:00Z", "2026-08-01T12:00:00Z"], utc=True,
        ),
        "conversion_bundle_sha256": ["same-conversion", "same-conversion"],
        "upstream_bundle_sha256": ["old-upstream", "new-upstream"],
        "geometry_bundle_sha256": ["one-geometry", "one-geometry"],
        "ev_score_family_id": ["one-family", "one-family"],
        "stack_is_prequential": [True, True],
    })
    calls: list[set[tuple[str, str]]] = []

    def _admit(subset: pd.DataFrame):
        calls.append(set(zip(
            subset["conversion_bundle_sha256"].astype(str),
            subset["upstream_bundle_sha256"].astype(str),
            strict=True,
        )))
        output = subset.copy()
        output["causal_21d_side_expected_net_bps"] = 75.0
        output["causal_21d_side_admitted_ge_50bps"] = True
        return output, pd.DataFrame({"rows": [len(subset)]})

    monkeypatch.setattr(canonical_current, "apply_current_admission", _admit)
    output, audit = canonical_current._apply_current_admission_by_score_vintage(frame)

    assert {frozenset(call) for call in calls} == {
        frozenset({("same-conversion", "old-upstream")}),
        frozenset({("same-conversion", "new-upstream")}),
    }
    assert output["candidate_id"].tolist() == ["old", "new"]
    assert output["ev_mapping_conversion_vintage"].tolist() == [
        "same-conversion", "same-conversion",
    ]
    assert output["ev_mapping_upstream_vintage"].tolist() == [
        "old-upstream", "new-upstream",
    ]
    assert set(audit["ev_mapping_vintage_mode"]) == {
        "strict_full_producer_vintage_fail_closed_v2",
    }


def test_snapshot_filters_to_current_conversion_vintage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resolved = pd.DataFrame({
        "candidate_id": ["old", "same"],
        "__decision_ts__": pd.to_datetime(
            ["2026-07-01T00:00:00Z", "2026-07-31T00:00:00Z"], utc=True,
        ),
        "side_name": ["long", "long"],
        "final_score": [0.95, 0.95],
        "policy_net_bps": [250.0, 25.0],
        "policy_label_available_ts": pd.to_datetime(
            ["2026-07-01T12:00:00Z", "2026-07-31T12:00:00Z"], utc=True,
        ),
        "conversion_bundle_sha256": ["old", "current"],
        "upstream_bundle_sha256": ["old-upstream", "current-upstream"],
        "geometry_bundle_sha256": ["geometry", "geometry"],
        "ev_score_family_id": ["family", "family"],
        "stack_is_prequential": [True, True],
    })
    current = pd.DataFrame({
        "candidate_id": ["live"],
        "__decision_ts__": pd.to_datetime(["2026-08-01T00:00:00Z"], utc=True),
        "side_name": ["long"],
        "final_score": [0.95],
        "conversion_bundle_sha256": ["current"],
        "upstream_bundle_sha256": ["current-upstream"],
        "geometry_bundle_sha256": ["geometry"],
        "ev_score_family_id": ["family"],
        "stack_is_prequential": [True],
    })

    def _admit(subset: pd.DataFrame):
        assert set(subset["conversion_bundle_sha256"].astype(str)) == {"current"}
        assert set(subset["upstream_bundle_sha256"].astype(str)) == {"current-upstream"}
        assert set(subset["candidate_id"]) == {"same", "live"}
        output = subset.copy()
        output["causal_21d_side_expected_net_bps"] = 75.0
        output["causal_21d_side_admitted_ge_50bps"] = True
        return output, pd.DataFrame({"rows": [len(subset)]})

    monkeypatch.setattr(
        canonical_current, "_apply_current_admission_by_score_vintage", _admit,
    )
    output, _ = canonical_current.apply_current_admission_snapshot(
        resolved_score_ledger=resolved, current_scores=current,
    )
    assert output["candidate_id"].tolist() == ["live"]
