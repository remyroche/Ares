import numpy as np
import pandas as pd

from extreme_price_movements.performance_regimes.archetype_experts import (
    ArchetypeExpertConfig,
    score_frozen_archetype_experts,
    train_archetype_experts,
)
from extreme_price_movements.performance_regimes.first_stage_models import (
    FirstStageLGBMConfig,
    TimeSeriesSplitSpec,
    train_first_stage_bad_good_models,
    walk_forward_splits,
)
from extreme_price_movements.performance_regimes.labels import (
    build_strategy_performance_labels,
)


def test_first_stage_lgbm_config_sets_min_child_samples_fraction():
    ts = pd.date_range("2026-01-01", periods=36, freq="h", tz="UTC")
    X = pd.DataFrame({"x": np.linspace(0, 1, len(ts)), "z": np.sin(np.arange(len(ts)))}, index=ts)
    trades = pd.DataFrame(
        {
            "timestamp": ts,
            "strategy": ["s1"] * len(ts),
            "performance": np.sin(np.linspace(0, 4, len(ts))),
        }
    )
    labels = build_strategy_performance_labels(
        trades,
        strategy_col="strategy",
        timestamp_col="timestamp",
        performance_col="performance",
        strategies=["s1"],
        ewma_halflife=2,
        anchor_mode="minmax",
    )
    bundle = train_first_stage_bad_good_models(
        X,
        labels,
        strategies=["s1"],
        cv=TimeSeriesSplitSpec(n_splits=2, min_train_size=12),
        lgbm_config=FirstStageLGBMConfig(n_estimators=20, min_child_samples_fraction=0.10),
    )

    assert not bundle.diagnostics.empty
    assert (bundle.diagnostics["min_child_samples"] >= 2).all()


def test_walk_forward_splits_supports_time_based_purge_hours():
    ts = pd.date_range("2026-01-01", periods=36, freq="h", tz="UTC")
    splits = walk_forward_splits(
        ts,
        TimeSeriesSplitSpec(n_splits=2, min_train_size=8, purge_hours=4.0),
    )

    assert splits
    for train_idx, valid_idx in splits:
        assert train_idx.max() < valid_idx.min()
        assert ts[train_idx.max()] < ts[valid_idx.min()] - pd.Timedelta(hours=4)


def test_archetype_expert_prevents_identity_leakage_and_centers_activity_score():
    ts = pd.date_range("2026-01-01", periods=48, freq="h", tz="UTC")
    archetype_id = "strategy_s1_bad_archetype_1"
    y = pd.Series((np.sin(np.linspace(0, 5, len(ts))) > 0).astype(float), index=ts)
    X = pd.DataFrame(
        {
            "market_x": np.linspace(0, 1, len(ts)),
            archetype_id: y,
            f"{archetype_id}__activity": y,
        },
        index=ts,
    )

    bundle = train_archetype_experts(
        X,
        {archetype_id: y},
        {archetype_id: pd.Series(1.0, index=ts)},
        cv=TimeSeriesSplitSpec(n_splits=2, min_train_size=12),
        config=ArchetypeExpertConfig(n_estimators=20),
    )
    result = bundle.by_archetype[archetype_id]

    assert archetype_id in result.excluded_identity_columns
    assert f"{archetype_id}__activity" in result.excluded_identity_columns
    finite_scores = result.activity_score.dropna()
    assert finite_scores.between(-1.0, 1.0).all()

    scored = score_frozen_archetype_experts(X[["market_x"]], bundle)
    assert archetype_id in scored.p_active.columns
    assert scored.activity_scores[archetype_id].dropna().between(-1.0, 1.0).all()
