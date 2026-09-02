from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from extreme_price_movements.inference.p8u_production_contract import exact_timestamp_route
from extreme_price_movements.inference.p8u_router_first_vectorized import (
    P8URouterFirstFeaturePlan,
    P8URouterFirstVectorizedStage,
)


def _candidates(symbols: tuple[str, ...]) -> pd.DataFrame:
    source = pd.Timestamp("2026-08-14T11:00:00Z")
    frame = pd.DataFrame(
        {
            "candidate_id": [f"id-{position}" for position in range(len(symbols))],
            "__decision_ts__": [source + pd.Timedelta(hours=1)] * len(symbols),
            "side_name": ["long"] * len(symbols),
            "__symbol__": symbols,
            "__ts__": [source] * len(symbols),
            "router_score": np.linspace(1.0, 0.1, len(symbols), dtype=np.float32),
        }
    )
    return exact_timestamp_route(frame, fraction=0.50)


def test_router_fields_use_full_universe_and_routed_fields_project_after_route(monkeypatch) -> None:
    symbols = ("A/USD:USD", "B/USD:USD", "C/USD:USD", "D/USD:USD")
    candidates = _candidates(symbols)
    calls: list[tuple[tuple[str, ...], tuple[str, ...]]] = []

    def fake_vectorized(panel, *, universe_symbols, requested_features, **_kwargs):
        calls.append((tuple(universe_symbols), tuple(requested_features)))
        stamp = pd.Timestamp("2026-08-14T11:00:00Z")
        return {
            field: pd.DataFrame(
                [np.arange(len(universe_symbols), dtype=np.float32) + (10.0 * ordinal)],
                index=[stamp],
                columns=universe_symbols,
            )
            for ordinal, field in enumerate(requested_features, start=1)
        }

    monkeypatch.setattr(
        "extreme_price_movements.inference.p8u_router_first_vectorized.canonical_features_from_saved_panel",
        fake_vectorized,
    )
    plan = P8URouterFirstFeaturePlan(
        router_features=("direct_router", "cheap_router"),
        base_features=("direct_base", "cheap_base"),
        under_features=("cheap_under",),
    )
    stage = P8URouterFirstVectorizedStage(
        universe_symbols=symbols,
        plan=plan,
        direct_fields=("direct_router", "direct_base"),
    )
    direct = {
        "direct_router": np.asarray([11.0, 12.0, 13.0, 14.0], dtype=np.float32),
        "direct_base": np.asarray([21.0, 22.0, 23.0, 24.0], dtype=np.float32),
    }

    router, router_vectorized = stage.build_router_matrix(
        candidates=candidates,
        panel={"opaque": object()},
        direct_output=direct,
    )
    assert len(router) == 4
    assert router_vectorized == ("cheap_router",)
    np.testing.assert_allclose(router["direct_router"], [11.0, 12.0, 13.0, 14.0])
    # The canonical vector graph is evaluated once on the full universe with
    # the complete non-direct field union.  Router still receives only its
    # own declared field before the Router50 handoff.
    assert calls == [(symbols, ("cheap_router", "cheap_base", "cheap_under"))]

    routed = candidates.loc[candidates["router50_eligible"]].copy()
    matrix, routed_vectorized = stage.build_routed_matrix(
        routed_candidates=routed,
        panel={"opaque": object()},
        direct_output=direct,
    )
    assert len(matrix) == 2
    assert routed_vectorized == ("cheap_base", "cheap_under")
    np.testing.assert_allclose(matrix["direct_base"], [21.0, 22.0])
    # Router50 changes only the returned model matrix identities: the routed
    # projection reuses the one full-universe timestamp cache and never
    # invokes the broad canonical graph again.
    assert len(calls) == 1
    np.testing.assert_allclose(matrix["cheap_base"], [20.0, 21.0])
    np.testing.assert_allclose(matrix["cheap_under"], [30.0, 31.0])


def test_routed_matrix_rejects_a_non_router50_identity() -> None:
    symbols = ("A/USD:USD", "B/USD:USD")
    candidates = _candidates(symbols)
    plan = P8URouterFirstFeaturePlan(
        router_features=("router",), base_features=("base",), under_features=()
    )
    stage = P8URouterFirstVectorizedStage(
        universe_symbols=symbols, plan=plan, direct_fields=("router", "base")
    )
    with pytest.raises(ValueError, match="Router50"):
        stage.build_routed_matrix(
            routed_candidates=candidates,
            panel={},
            direct_output={
                "router": np.zeros(2, dtype=np.float32),
                "base": np.zeros(2, dtype=np.float32),
            },
        )


def test_vectorised_projection_never_receives_future_source_rows(monkeypatch) -> None:
    symbols = ("A/USD:USD", "B/USD:USD")
    candidates = _candidates(symbols)
    observed_ends: list[pd.Timestamp] = []

    def fake_vectorized(panel, *, universe_symbols, requested_features, **_kwargs):
        observed_ends.append(pd.DatetimeIndex(panel["close"].index).max())
        stamp = pd.Timestamp("2026-08-14T11:00:00Z")
        return {
            field: pd.DataFrame([[1.0, 2.0]], index=[stamp], columns=universe_symbols)
            for field in requested_features
        }

    monkeypatch.setattr(
        "extreme_price_movements.inference.p8u_router_first_vectorized.canonical_features_from_saved_panel",
        fake_vectorized,
    )
    stage = P8URouterFirstVectorizedStage(
        universe_symbols=symbols,
        plan=P8URouterFirstFeaturePlan(
            router_features=("cheap_router",), base_features=(), under_features=()
        ),
        direct_fields=(),
    )
    index = pd.date_range("2026-08-14T10:00:00Z", periods=3, freq="h")
    stage.build_router_matrix(
        candidates=candidates,
        panel={"close": pd.DataFrame(np.ones((3, 2)), index=index, columns=symbols)},
        direct_output={},
    )
    assert observed_ends == [pd.Timestamp("2026-08-14T11:00:00Z")]


def test_regular_feature_state_snapshot_bypasses_batch_vector_graph(monkeypatch) -> None:
    """The scoring path must consume one persisted timestamp, not a tail panel."""

    symbols = ("A/USD:USD", "B/USD:USD", "C/USD:USD", "D/USD:USD")
    candidates = _candidates(symbols)
    plan = P8URouterFirstFeaturePlan(
        router_features=("direct_router", "regular_router"),
        base_features=("regular_base",),
        under_features=("regular_under",),
    )
    stage = P8URouterFirstVectorizedStage(
        universe_symbols=symbols,
        plan=plan,
        direct_fields=("direct_router",),
    )

    def forbidden_batch_graph(*_args, **_kwargs):
        raise AssertionError("regular feature-state mode must not call the batch graph")

    monkeypatch.setattr(
        "extreme_price_movements.inference.p8u_router_first_vectorized.canonical_features_from_saved_panel",
        forbidden_batch_graph,
    )
    snapshot = candidates.loc[:, [
        "candidate_id", "__decision_ts__", "side_name", "__symbol__", "__ts__",
    ]].copy()
    snapshot["regular_router"] = [10.0, 11.0, 12.0, 13.0]
    snapshot["regular_base"] = [20.0, 21.0, 22.0, 23.0]
    snapshot["regular_under"] = [30.0, 31.0, 32.0, 33.0]
    direct = {"direct_router": np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float32)}

    router, fields = stage.build_router_matrix(
        candidates=candidates,
        panel={},
        direct_output=direct,
        vector_snapshot=snapshot,
    )
    assert fields == ("regular_router",)
    np.testing.assert_allclose(router["regular_router"], [10.0, 11.0, 12.0, 13.0])

    routed, routed_fields = stage.build_routed_matrix(
        routed_candidates=candidates.loc[candidates["router50_eligible"]].copy(),
        panel={},
        direct_output=direct,
        vector_snapshot=snapshot,
    )
    assert routed_fields == ("regular_base", "regular_under")
    np.testing.assert_allclose(routed["regular_base"], [20.0, 21.0])
    np.testing.assert_allclose(routed["regular_under"], [30.0, 31.0])


def test_regular_feature_state_producer_emits_only_non_direct_frozen_fields(monkeypatch) -> None:
    symbols = ("A/USD:USD", "B/USD:USD", "C/USD:USD", "D/USD:USD")
    candidates = _candidates(symbols)
    observed_requests: list[tuple[str, ...]] = []

    def fake_vectorized(_panel, *, universe_symbols, requested_features, **_kwargs):
        observed_requests.append(tuple(requested_features))
        stamp = pd.Timestamp("2026-08-14T11:00:00Z")
        return {
            field: pd.DataFrame(
                [np.arange(len(universe_symbols), dtype=np.float32)],
                index=[stamp], columns=universe_symbols,
            )
            for field in requested_features
        }

    monkeypatch.setattr(
        "extreme_price_movements.inference.p8u_router_first_vectorized.canonical_features_from_saved_panel",
        fake_vectorized,
    )
    stage = P8URouterFirstVectorizedStage(
        universe_symbols=symbols,
        plan=P8URouterFirstFeaturePlan(
            router_features=("direct", "regular_router"),
            base_features=("regular_base",),
            under_features=("regular_under",),
        ),
        direct_fields=("direct",),
    )
    snapshot = stage.materialize_regular_feature_state_snapshot(
        candidates=candidates,
        panel={"opaque": object()},
    )
    assert observed_requests == [("regular_router", "regular_base", "regular_under")]
    assert snapshot.columns.tolist() == [
        "candidate_id", "__decision_ts__", "side_name", "__symbol__", "__ts__",
        "regular_router", "regular_base", "regular_under",
    ]
    assert "direct" not in snapshot
