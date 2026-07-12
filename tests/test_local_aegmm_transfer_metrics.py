from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.run_train_meta_residual_archetype_enhancement import (
    local_aegmm_state_transfer_metrics,
)


def test_local_state_transfer_metrics_compare_train_priors_to_oos_tails() -> None:
    timestamps = pd.date_range("2026-04-01", periods=3, freq="15min", tz="UTC")
    rows: list[dict[str, object]] = []
    for timestamp in timestamps:
        rows.extend(
            [
                {
                    "__ts__": timestamp,
                    "side_name": "long",
                    "archetype_policy_key": "long_mixed",
                    "ev_after_1pct": 0.012,
                    "clean_exec": 1.0,
                    "dirty_positive": 0.0,
                    "first_touch_bad_mae_1r": 0.0,
                    "timeout": 0.0,
                    "score_current_reference": 0.91,
                    "score_alternative": 0.93,
                    "local_econ_aegmm_market_state_enabled": 1.0,
                    "local_econ_aegmm_market_state_local_model": 1.0,
                    "local_econ_aegmm_market_state_gmm_cluster_id": 0,
                    "local_econ_aegmm_market_state_expected_ev": 0.010,
                    "local_econ_aegmm_market_state_expected_clean_positive": 0.80,
                    "local_econ_aegmm_market_state_expected_dirty_positive": 0.10,
                    "local_econ_aegmm_market_state_expected_bad_mae": 0.08,
                    "local_econ_aegmm_market_state_expected_timeout": 0.01,
                },
                {
                    "__ts__": timestamp,
                    "side_name": "long",
                    "archetype_policy_key": "long_mixed",
                    "ev_after_1pct": -0.010,
                    "clean_exec": 0.0,
                    "dirty_positive": 1.0,
                    "first_touch_bad_mae_1r": 1.0,
                    "timeout": 0.0,
                    "score_current_reference": 0.10,
                    "score_alternative": 0.20,
                    "local_econ_aegmm_market_state_enabled": 1.0,
                    "local_econ_aegmm_market_state_local_model": 1.0,
                    "local_econ_aegmm_market_state_gmm_cluster_id": 1,
                    "local_econ_aegmm_market_state_expected_ev": -0.008,
                    "local_econ_aegmm_market_state_expected_clean_positive": 0.20,
                    "local_econ_aegmm_market_state_expected_dirty_positive": 0.75,
                    "local_econ_aegmm_market_state_expected_bad_mae": 0.70,
                    "local_econ_aegmm_market_state_expected_timeout": 0.02,
                },
            ]
        )
    predictions = pd.DataFrame(rows)
    catalog = pd.DataFrame(
        {
            "model_key": [
                "local::long::long_mixed::market_state",
                "local::long::long_mixed::market_state",
            ],
            "cluster": [0, 1],
            "semantic": ["clean_high_confidence", "acute_adverse_false_positive"],
            "posterior_support": [600.0, 400.0],
            "ev": [0.010, -0.008],
            "clean_positive": [0.80, 0.20],
            "dirty_positive": [0.10, 0.75],
            "bad_mae": [0.08, 0.70],
            "timeout": [0.01, 0.02],
        }
    )

    result = local_aegmm_state_transfer_metrics(
        predictions, "local_aegmm_market", catalog
    )

    assert not result.empty
    all_rows = result.loc[result["scope"].eq("all_candidates")]
    assert set(all_rows["train_semantic"]) == {
        "clean_high_confidence",
        "acute_adverse_false_positive",
    }
    clean = all_rows.loc[all_rows["state_cluster"].eq(0)].iloc[0]
    adverse = all_rows.loc[all_rows["state_cluster"].eq(1)].iloc[0]
    assert clean["mean_ev_after_1pct"] > 0.0
    assert adverse["mean_ev_after_1pct"] < 0.0
    assert clean["train_prior_ev"] > 0.0
    assert adverse["train_prior_ev"] < 0.0
    assert clean["posterior_prior_ev_lift_sign_agrees"] == np.float32(1.0)
    top10 = result.loc[result["scope"].eq("local_aegmm_market_top10")]
    assert len(top10) == 1
    assert int(top10.iloc[0]["state_cluster"]) == 0
