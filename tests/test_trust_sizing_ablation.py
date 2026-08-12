from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from extreme_price_movements.trust_sizing_ablation import (
    ParentExpectation,
    TrustModelSpec,
    assert_geometry_semantics,
    causal_size_multiplier,
    discover_cmi_edges,
    fit_trust_model,
    independent_experience_support,
)
from scripts.run_strict_r3_trust_sizing_ablation import _blocks


def _frame(rows: int = 2_000) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    ts = pd.date_range("2024-10-01", periods=rows, freq="h", tz="UTC")
    support = rng.normal(size=rows)
    ood = rng.normal(size=rows)
    score = rng.uniform(size=rows)
    expected = 100.0 * score - 30.0
    realised = expected + 35.0 * support - 45.0 * ood + rng.normal(0, 80, rows)
    result = pd.DataFrame(
        {
            "candidate_id": [f"c-{idx}" for idx in range(rows)],
            "__decision_ts__": ts,
            "final_score": score,
            "raw_expected_bps": expected,
            "policy_net_bps": realised,
            "support_feature": support,
            "ood_feature": ood,
            "geometry_bundle_sha256": "bundle-a",
        }
    )
    parent = ParentExpectation.fit(result["final_score"], result["policy_net_bps"])
    result["parent_expected_bps"] = parent.predict(result["final_score"])
    return result


def test_raw_k9_memberships_cannot_cross_bundle_identities() -> None:
    frame = _frame(200)
    frame.loc[100:, "geometry_bundle_sha256"] = "bundle-b"
    frame["k09__cluster_00__membership"] = 0.5
    with pytest.raises(ValueError, match="one identical Geometry/K9 bundle"):
        assert_geometry_semantics(frame, ["k09__cluster_00__membership"])
    assert_geometry_semantics(frame, ["support_feature", "ood_feature"])


def test_three_month_blocks_are_frozen_quarters_plus_partial_july() -> None:
    blocks = _blocks(2026)
    assert blocks == [
        (pd.Timestamp("2026-01-01", tz="UTC"), pd.Timestamp("2026-04-01", tz="UTC")),
        (pd.Timestamp("2026-04-01", tz="UTC"), pd.Timestamp("2026-07-01", tz="UTC")),
        (pd.Timestamp("2026-07-01", tz="UTC"), pd.Timestamp("2026-08-01", tz="UTC")),
    ]


def test_causal_size_mapping_uses_only_train_reference() -> None:
    train = np.linspace(0.0, 1.0, 1_001)
    score = np.asarray([0.1, 0.5, 0.9])
    first = causal_size_multiplier(train, score)
    second = causal_size_multiplier(train, np.r_[score, 1e9])[:3]
    np.testing.assert_allclose(first, second)
    assert np.all((first >= 0.25) & (first <= 1.75))


def test_stable_cmi_edges_are_cross_family_and_train_only() -> None:
    frame = _frame()
    edges, _bins = discover_cmi_edges(
        frame,
        ["support_feature", "ood_feature"],
        mode="rank",
        stable=True,
        max_edges=4,
    )
    assert edges
    assert all(edge.family_left != edge.family_right for edge in edges)


def test_empirical_bayes_produces_bounded_authority() -> None:
    frame = _frame()
    train, held = frame.iloc[:1_500].copy(), frame.iloc[1_500:].copy()
    spec = TrustModelSpec(
        name="test",
        pipeline="bayesian",
        model_family="empirical_bayes",
        interactions="stable_cmi",
        cmi_weighting="rank",
        lambda_max=1.10,
        risk_mode="stable_cmi",
        sizing_mode="mean_risk",
    )
    _train_prediction, held_prediction, audit = fit_trust_model(
        train, held, ["support_feature", "ood_feature"], spec,
    )
    assert np.isfinite(held_prediction.expected_bps).all()
    assert np.isfinite(held_prediction.predictive_sd_bps).all()
    assert held_prediction.shrinkage_lambda.min() >= 0.0
    assert held_prediction.shrinkage_lambda.max() <= 1.10 + 1e-9
    assert audit["raw_k9_memberships_used"] is False


def test_cell_day_residual_forest_outputs_corroborated_risk_fields() -> None:
    frame = _frame(2_400)
    train, held = frame.iloc[:1_800].copy(), frame.iloc[1_800:].copy()
    spec = TrustModelSpec(
        name="residual-test",
        pipeline="nonlinear",
        model_family="cell_day_residual_forest",
        interactions="stable_cmi",
        cmi_weighting="rank_loss_false_positive",
        lambda_max=1.10,
        risk_mode="stable_cmi",
        sizing_mode="mean_risk",
        probability_mode="shrunk",
        target_mode="cell_day_residual_clip300",
    )
    _train_prediction, held_prediction, audit = fit_trust_model(
        train, held, ["support_feature", "ood_feature"], spec,
    )
    assert audit["target_mode"] == "cell_day_residual_clip300"
    assert held_prediction.residual_mean_bps is not None
    assert held_prediction.residual_q25_bps is not None
    assert held_prediction.p_map_overestimate_100bps is not None
    assert np.isfinite(held_prediction.residual_mean_bps).all()
    assert np.isfinite(held_prediction.residual_q25_bps).all()
    assert np.all(
        (held_prediction.p_map_overestimate_100bps >= 0.0)
        & (held_prediction.p_map_overestimate_100bps <= 1.0)
    )
    np.testing.assert_allclose(
        held_prediction.expected_bps,
        held["raw_expected_bps"].to_numpy(float)
        + held_prediction.residual_mean_bps,
    )


def test_independent_experience_support_is_bounded_and_rewards_breadth() -> None:
    narrow = _frame(500)
    narrow["__symbol__"] = "BTC"
    narrow["__decision_ts__"] = pd.Timestamp("2025-01-01", tz="UTC")
    broad = _frame(500)
    broad["__symbol__"] = [f"asset-{idx % 50}" for idx in range(len(broad))]
    weight = np.ones(500)
    mask = np.ones(500, dtype=bool)
    narrow_support = independent_experience_support(narrow, mask, weight)
    broad_support = independent_experience_support(broad, mask, weight)
    assert 0 < narrow_support < broad_support <= 500


def test_residual_forest_separates_neutral_mean_and_independent_support() -> None:
    frame = _frame(2_400)
    frame["__symbol__"] = [f"asset-{idx % 40}" for idx in range(len(frame))]
    train, held = frame.iloc[:1_800].copy(), frame.iloc[1_800:].copy()
    spec = TrustModelSpec(
        name="neutral-mean-test", pipeline="nonlinear",
        model_family="cell_day_residual_forest", interactions="stable_cmi",
        cmi_weighting="rank_loss_false_positive", lambda_max=1.10,
        risk_mode="stable_cmi", sizing_mode="mean_risk",
        probability_mode="shrunk", target_mode="cell_day_residual_clip500",
        mean_weighting="uniform", support_mode="independent_experience",
        uncertainty_mode="local_leaf",
    )
    _train_prediction, held_prediction, audit = fit_trust_model(
        train, held, ["support_feature", "ood_feature"], spec,
    )
    assert audit["mean_weighting"] == "uniform"
    assert audit["support_mode"] == "independent_experience"
    assert audit["uncertainty_mode"] == "local_leaf"
    assert np.isfinite(held_prediction.expected_bps).all()
    assert np.isfinite(held_prediction.predictive_sd_bps).all()
    assert (held_prediction.predictive_sd_bps > 0.0).all()
