from __future__ import annotations

import numpy as np
import pandas as pd

from extreme_price_movements.n5_forest_support_sizing import (
    N5ForestParams,
    fit_n5_forest,
    n5_hpo_candidates,
)


def _frame(rows: int = 900) -> pd.DataFrame:
    rng = np.random.default_rng(17)
    score = rng.uniform(size=rows)
    support = rng.normal(size=rows)
    ood = rng.normal(size=rows)
    parent = 80.0 * score - 20.0
    expected = parent + 25.0 * support
    realised = expected + 35.0 * support - 45.0 * ood + rng.normal(0.0, 70.0, rows)
    return pd.DataFrame(
        {
            "candidate_id": [f"n5-{index}" for index in range(rows)],
            "__decision_ts__": pd.date_range("2024-10-01", periods=rows, freq="h", tz="UTC"),
            "final_score": score,
            "raw_expected_bps": expected,
            "parent_expected_bps": parent,
            "policy_net_bps": realised,
            "support_feature": support,
            "ood_feature": ood,
            "geometry_bundle_sha256": "bundle-a",
        }
    )


def test_n5_uses_oob_train_reference_and_bounded_sizes() -> None:
    frame = _frame()
    train, held = frame.iloc[:700].copy(), frame.iloc[700:].copy()
    params = N5ForestParams(
        n_estimators=32,
        max_depth=5,
        min_samples_leaf=40,
        max_features=1.0,
        max_samples=0.8,
    )
    bundle, train_prediction = fit_n5_forest(
        train,
        ["support_feature", "ood_feature"],
        [],
        params=params,
    )
    prediction, multiplier = bundle.size_multiplier(held)
    assert bundle.target_audit["oob_mean_rmse_bps"] > 0.0
    assert len(bundle.train_quality_reference) == len(train)
    assert np.isfinite(train_prediction.expected_bps).all()
    assert np.isfinite(prediction.predictive_sd_bps).all()
    assert np.all((multiplier >= params.size_floor) & (multiplier <= params.size_cap))


def test_n5_target_hpo_surface_covers_mean_and_risk_targets() -> None:
    candidates = n5_hpo_candidates()
    assert {candidate.mean_target for candidate in candidates} == {
        "policy_net", "parent_residual", "winsorized_net",
    }
    assert {candidate.risk_target for candidate in candidates} == {
        "oob_squared", "oob_downside", "oob_absolute",
    }
