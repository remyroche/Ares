from __future__ import annotations

import numpy as np

from extreme_price_movements.residual_rule_models import (
    ContrastiveSubgroupArm,
    EpisodeLGBMArm,
    EpisodeMLPArm,
    ModelBasedRecursivePartitionArm,
    RobustMatrixTransform,
    matched_benign_controls,
    matched_benign_period_controls,
    build_rule_arm,
)


def _sample() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(7)
    x = rng.normal(size=(240, 4)).astype(np.float32)
    y = ((x[:, 0] > 0.8) & (x[:, 1] < -0.2)).astype(np.int8)
    weights = np.ones(len(y), dtype=np.float32)
    return x, y, weights


def test_matched_controls_are_benign_and_deterministic() -> None:
    x, y, _ = _sample()
    blocks = np.full(len(y), -1, dtype=np.int32)
    blocks[y > 0] = np.arange(int(y.sum())) // 3
    selected, report = matched_benign_controls(x, y, blocks, controls_per_event=2)
    assert report
    assert not np.any(selected & (y > 0))


def test_period_controls_match_full_benign_windows() -> None:
    rng = np.random.default_rng(11)
    x = rng.normal(size=(12, 3)).astype(np.float32)
    y = np.zeros(12, dtype=np.int8)
    # Two consecutive adverse calendar days, two state rows per day.
    y[4:8] = 1
    blocks = np.full(12, -1, dtype=np.int32)
    blocks[4:8] = 0
    days = np.repeat(
        np.arange("2026-01-01", "2026-01-07", dtype="datetime64[D]"), 2
    )
    selected, report = matched_benign_period_controls(
        x, y, blocks, days, controls_per_event=2
    )
    assert report
    assert not np.any(selected & (y > 0))
    assert report[0]["adverse_days"] == 2
    assert report[0]["control_windows"] >= 1
    assert report[0]["control_rows"] >= 4


def test_contrastive_subgroup_and_partition_are_inference_only() -> None:
    x, y, weights = _sample()
    names = [f"f{index}" for index in range(x.shape[1])]
    for model in (ContrastiveSubgroupArm(), ModelBasedRecursivePartitionArm(min_leaf=20)):
        model.fit(x, y, weights, names)
        score = model.predict_proba(x)
        assert score.shape == (len(x),)
        assert np.isfinite(score).all()
        assert model.describe()


def test_robust_transform_fills_missing_and_clips() -> None:
    x, _, _ = _sample()
    x[0, 0] = np.nan
    x[1, 1] = np.inf
    transformed = RobustMatrixTransform().fit(x).transform(x)
    assert np.isfinite(transformed).all()
    assert np.abs(transformed).max() <= 8.0


def test_episode_lgbm_and_mlp_produce_inference_scores() -> None:
    x, y, weights = _sample()
    names = [f"f{index}" for index in range(x.shape[1])]
    for model in (EpisodeLGBMArm(seed=11), EpisodeMLPArm(seed=11, max_rows=400)):
        model.fit(x, y, weights, names)
        score = model.predict_proba(x)
        assert score.shape == (len(x),)
        assert np.isfinite(score).all()
        assert model.describe()


def test_contrastive_lgbm_alias_uses_the_constrained_episode_estimator() -> None:
    model = build_rule_arm("episode_lgbm_contrastive", seed=11)
    assert isinstance(model, EpisodeLGBMArm)
