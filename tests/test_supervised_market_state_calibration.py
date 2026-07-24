from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from extreme_price_movements.supervised_market_state_calibration import (
    fit_hierarchical_ev_calibrator,
    hierarchical_ev_calibrator_payload,
    fit_local_state_encoder,
    mlp_hidden_embedding,
    predict_hierarchical_ev,
    predict_local_state_encoder,
)


@pytest.mark.parametrize("arm", ["ae_gmm", "mlp_gmm", "mlp_direct", "ae_mlp_gmm"])
def test_local_encoder_transforms_oos_without_targets(arm: str) -> None:
    rng = np.random.default_rng(12)
    n = 1_400
    train = pd.DataFrame(
        {
            "market_a": rng.normal(size=n),
            "market_b": rng.normal(size=n),
            "ae_0": rng.normal(size=n),
            "ae_1": rng.normal(size=n),
        }
    )
    ev = (0.01 * train["market_a"] - 0.005 * train["market_b"] + rng.normal(0, 0.01, n)).to_numpy(np.float32)
    model = fit_local_state_encoder(
        train,
        side="long",
        archetype="breakout",
        arm=arm,
        features=["market_a", "market_b"],
        ae_features=["ae_0", "ae_1"],
        ev_residual=ev,
        hit_residual=(ev > 0).astype(np.float32) - 0.5,
        sample_weight=np.ones(n, dtype=np.float32),
        seed=3,
    )
    oos = train.iloc[-25:].copy()
    result = predict_local_state_encoder(model, oos, ae_features=["ae_0", "ae_1"])
    assert set(result) == {"ev_correction", "posterior_confidence", "posterior_entropy"}
    assert all(len(value) == len(oos) for value in result.values())
    assert all(np.isfinite(value).all() for value in result.values())
    assert np.all((result["posterior_confidence"] >= 0) & (result["posterior_confidence"] <= 1))


def test_mlp_embedding_is_final_hidden_layer() -> None:
    rng = np.random.default_rng(7)
    frame = pd.DataFrame({"x": rng.normal(size=1_200), "z": rng.normal(size=1_200)})
    ev = (frame["x"] * 0.01).to_numpy(np.float32)
    model = fit_local_state_encoder(
        frame,
        side="short", archetype="mixed", arm="mlp_direct",
        features=["x", "z"], ae_features=[], ev_residual=ev,
        hit_residual=(ev > 0).astype(np.float32) - 0.5,
        sample_weight=np.ones(len(frame), dtype=np.float32), seed=4,
    )
    x = (frame[["x", "z"]].to_numpy(np.float32) - model.medians) / model.scales
    latent = mlp_hidden_embedding(model.mlp, np.clip(x, -8, 8))
    assert latent.shape == (len(frame), 8)
    assert np.isfinite(latent).all()


def test_hierarchical_ev_calibrator_emits_common_ev_units() -> None:
    rng = np.random.default_rng(19)
    n = 2_400
    score = np.linspace(0.5, 1.0, n, dtype=np.float32)
    side = np.where(np.arange(n) % 2, "long", "short")
    archetype = np.where(np.arange(n) % 3, "breakout", "mixed")
    local_shift = np.where((side == "long") & (archetype == "breakout"), 0.008, 0.0)
    ev = -0.01 + 0.03 * score + local_shift + rng.normal(0, 0.003, n)
    frame = pd.DataFrame(
        {"side_name": side, "archetype_policy_key": archetype}
    )
    calibrator = fit_hierarchical_ev_calibrator(
        frame, score, ev, shrink_rows=300.0, min_local_rows=200
    )
    oos = pd.DataFrame(
        {
            "side_name": ["long", "short", "unknown"],
            "archetype_policy_key": ["breakout", "mixed", "new"],
        }
    )
    pred = predict_hierarchical_ev(
        calibrator, oos, np.array([0.95, 0.95, 0.95], dtype=np.float32)
    )
    assert pred.dtype == np.float32
    assert np.isfinite(pred).all()
    assert pred[0] > pred[1]
    assert -0.25 <= pred[2] <= 0.25


def test_sparse_local_ev_calibration_falls_back_to_side_parent() -> None:
    score = np.linspace(0.5, 1.0, 800, dtype=np.float32)
    frame = pd.DataFrame(
        {
            "side_name": ["long"] * 400 + ["short"] * 390 + ["short"] * 10,
            "archetype_policy_key": ["main"] * 400 + ["main"] * 390 + ["sparse"] * 10,
        }
    )
    ev = -0.01 + 0.02 * score + np.where(frame["side_name"].eq("long"), 0.04, -0.04)
    calibrator = fit_hierarchical_ev_calibrator(
        frame, score, ev, shrink_rows=1.0, min_local_rows=100, local_weight_cap=1.0
    )
    assert ("short", "sparse") not in calibrator.local_models
    assert "short" in calibrator.side_models
    sparse = frame.iloc[-1:].copy()
    pred = predict_hierarchical_ev(calibrator, sparse, np.array([0.9]))
    raw = np.array([0.9])
    global_expected = calibrator.global_model.predict(raw)
    side_expected = calibrator.side_models["short"].predict(raw)
    expected = (
        (1.0 - calibrator.side_weights["short"]) * global_expected
        + calibrator.side_weights["short"] * side_expected
    )
    score_rank = (raw - calibrator.refinement_score_min) / max(
        calibrator.refinement_score_max - calibrator.refinement_score_min, 1e-8
    )
    expected += calibrator.monotonic_refinement_slope * (score_rank - 0.5)
    np.testing.assert_allclose(pred, expected, atol=1e-7)
    assert pred[0] < global_expected[0]

    payload = hierarchical_ev_calibrator_payload(calibrator)
    assert payload["schema"] == "hierarchical_monotonic_expected_ev_v3"
    assert payload["mapping_scope"] == "global_to_side_to_side_x_archetype"
    assert payload["side"]["short"]["support"] == 400

    legacy = copy.copy(calibrator)
    del legacy.side_models
    del legacy.side_weights
    del legacy.side_support
    legacy_pred = predict_hierarchical_ev(legacy, sparse, raw)
    legacy_expected = global_expected + calibrator.monotonic_refinement_slope * (
        score_rank - 0.5
    )
    np.testing.assert_allclose(legacy_pred, legacy_expected, atol=1e-7)

    legacy_local = frame.iloc[400:401].copy()
    legacy_local_pred = predict_hierarchical_ev(legacy, legacy_local, raw)
    local_key = ("short", "main")
    legacy_local_expected = (
        (1.0 - calibrator.local_weights[local_key]) * global_expected
        + calibrator.local_weights[local_key]
        * calibrator.local_models[local_key].predict(raw)
        + calibrator.monotonic_refinement_slope * (score_rank - 0.5)
    )
    np.testing.assert_allclose(legacy_local_pred, legacy_local_expected, atol=1e-7)
    assert hierarchical_ev_calibrator_payload(legacy)["mapping_scope"] == (
        "side_x_archetype_shrunk_to_global"
    )


def test_hierarchical_ev_refinement_is_strictly_monotonic_and_granular() -> None:
    score = np.linspace(0.0, 1.0, 1_000, dtype=np.float32)
    frame = pd.DataFrame(
        {
            "side_name": ["long"] * len(score),
            "archetype_policy_key": ["state"] * len(score),
        }
    )
    realized = np.where(score >= 0.8, 0.01, -0.01)
    calibrator = fit_hierarchical_ev_calibrator(
        frame,
        score,
        realized,
        min_local_rows=100,
        monotonic_refinement_slope=0.00025,
    )

    mapped = predict_hierarchical_ev(calibrator, frame, score)

    assert np.all(np.diff(mapped) > 0.0)
    assert np.unique(mapped).size == len(mapped)


def test_quantile_tail_weighting_is_score_scale_invariant() -> None:
    score = np.linspace(0.2, 0.65, 1_200, dtype=np.float32)
    transformed = 3.0 + 7.0 * score
    frame = pd.DataFrame(
        {
            "side_name": np.where(np.arange(len(score)) % 2, "long", "short"),
            "archetype_policy_key": "main",
        }
    )
    ev = -0.01 + 0.03 * score
    left = fit_hierarchical_ev_calibrator(
        frame,
        score,
        ev,
        tail_weight_by_score_quantile=True,
        min_local_rows=100,
    )
    right = fit_hierarchical_ev_calibrator(
        frame,
        transformed,
        ev,
        tail_weight_by_score_quantile=True,
        min_local_rows=100,
    )
    np.testing.assert_allclose(
        predict_hierarchical_ev(left, frame, score),
        predict_hierarchical_ev(right, frame, transformed),
        atol=1e-6,
    )
