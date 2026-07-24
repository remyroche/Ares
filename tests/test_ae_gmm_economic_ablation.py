from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from extreme_price_movements.ae_gmm_economic_ablation import (
    AEGMMArm,
    arm_model_features,
    base_selection_ranking,
    economic_metrics,
    model_ae_gmm_features,
    split_months,
    strip_ae_gmm_features,
)
from scripts.run_materialized_trailing_label_topk_lgbm_hpo import (
    _validation_windows,
)


def test_strip_and_replace_ae_gmm_feature_contract() -> None:
    production = [
        "score_context",
        "gmm_entropy",
        "base_lgbm_AE_reconstruction_error",
        "raw_trend",
    ]
    core = strip_ae_gmm_features(production)
    assert core == ["score_context", "raw_trend"]

    arm = AEGMMArm(
        arm_id="candidate_k5_diag",
        mode="fit",
        input_features=("raw_trend", "raw_vol"),
        cluster_candidates=(5,),
    )
    candidate = arm_model_features(
        arm,
        production_features=production,
        core_features=core,
    )
    assert candidate[:2] == core
    assert "gmm_entropy" in candidate
    assert "AE_reconstruction_error" in candidate
    assert "gmm_cluster_id" not in model_ae_gmm_features()
    assert "cluster_t" not in model_ae_gmm_features()


def test_five_month_contract_reserves_last_two_for_meta() -> None:
    periods = split_months(
        ["2026-02", "2026-03", "2026-04", "2026-05", "2026-06"]
    )
    assert periods["base_selection"] == ["2026-02", "2026-03", "2026-04"]
    assert periods["meta_train"] == periods["base_selection"]
    assert periods["meta_oos"] == ["2026-05", "2026-06"]
    with pytest.raises(ValueError):
        split_months(["2026-02", "2026-04", "2026-05", "2026-06", "2026-07"])


def test_single_fit_base_window_spans_contiguous_months() -> None:
    windows = _validation_windows(
        ["2026-02", "2026-03", "2026-04", "2026-05", "2026-06"],
        max_oos_model_age_days=0,
        single_fit_oos_window=True,
    )
    assert len(windows) == 1
    assert windows[0]["valid_start"] == pd.Timestamp("2026-02-01", tz="UTC")
    assert windows[0]["valid_end"] == pd.Timestamp("2026-07-01", tz="UTC")
    with pytest.raises(ValueError, match="contiguous"):
        _validation_windows(
            ["2026-02", "2026-04"],
            max_oos_model_age_days=0,
            single_fit_oos_window=True,
        )


def test_economic_metrics_use_global_topk_and_signed_daily_diagnostics() -> None:
    rows = 100
    frame = pd.DataFrame(
        {
            "__ts__": pd.date_range("2026-02-01", periods=rows, freq="h", tz="UTC"),
            "side_name": np.where(np.arange(rows) % 2 == 0, "long", "short"),
            "__archetype_label_family__": np.where(
                np.arange(rows) % 3 == 0, "breakout", "compression"
            ),
            "score": np.linspace(0.0, 1.0, rows, dtype=np.float32),
            "__u_policy_net__": np.linspace(-0.01, 0.02, rows, dtype=np.float32),
            "__first_touch_target_soft__": np.linspace(0.1, 0.9, rows, dtype=np.float32),
            "__first_touch_net_positive__": (np.arange(rows) >= 50).astype(np.float32),
            "__path_full_bad_mae_1r__": (np.arange(rows) < 20).astype(np.float32),
            "__first_touch_timeout__": np.zeros(rows, dtype=np.float32),
            "__first_touch_stop__": (np.arange(rows) < 10).astype(np.float32),
        }
    )
    metrics = economic_metrics(frame, arm="arm_a", months=["2026-02"])
    overall_top10 = metrics.loc[
        metrics["scope"].eq("overall") & metrics["top_frac"].eq(0.10)
    ].iloc[0]
    assert int(overall_top10["selected_rows"]) == 10
    assert float(overall_top10["mean_ev_after_1pct"]) > 0.018
    assert "signed_residual_autocorr" in metrics.columns
    assert set(metrics["selection_basis"]) == {"global_topk"}

    ranking = base_selection_ranking(metrics, months=["2026-02"])
    assert ranking.iloc[0]["arm"] == "arm_a"


def test_economic_metrics_supports_public_handoff_archetype_key() -> None:
    frame = pd.DataFrame(
        {
            "__ts__": pd.date_range("2026-02-01", periods=20, freq="h", tz="UTC"),
            "side_name": ["short"] * 20,
            "archetype_policy_key": ["short_default_clean_path"] * 20,
            "score": np.linspace(0.0, 1.0, 20, dtype=np.float32),
            "__u_policy_net__": np.linspace(-0.01, 0.02, 20, dtype=np.float32),
            "__first_touch_target_soft__": np.linspace(0.1, 0.9, 20, dtype=np.float32),
            "__first_touch_net_positive__": (np.arange(20) >= 10).astype(np.float32),
            "__path_full_bad_mae_1r__": np.zeros(20, dtype=np.float32),
            "__first_touch_timeout__": np.zeros(20, dtype=np.float32),
        }
    )

    metrics = economic_metrics(frame, arm="arm_a", months=["2026-02"])
    archetype = metrics.loc[metrics["scope"].eq("archetype")].iloc[0]
    assert archetype["archetype_label_family"] == "short_default_clean_path"


def test_economic_metrics_can_rank_each_side_independently() -> None:
    frame = pd.DataFrame(
        {
            "__ts__": pd.date_range("2026-02-01", periods=20, freq="h", tz="UTC"),
            "side_name": ["long"] * 10 + ["short"] * 10,
            # A global top-10% takes the two long rows. A side-local top-10%
            # must preserve one row from each calibrated probability scale.
            "score": np.r_[np.linspace(0.80, 0.99, 10), np.linspace(0.10, 0.29, 10)],
            "__u_policy_net__": np.linspace(-0.01, 0.02, 20, dtype=np.float32),
            "__first_touch_target_soft__": np.linspace(0.1, 0.9, 20, dtype=np.float32),
            "__first_touch_net_positive__": (np.arange(20) >= 10).astype(np.float32),
            "__path_full_bad_mae_1r__": np.zeros(20, dtype=np.float32),
            "__first_touch_timeout__": np.zeros(20, dtype=np.float32),
        }
    )
    metrics = economic_metrics(
        frame,
        arm="arm_a",
        months=["2026-02"],
        selection_basis="per_side",
    )
    overall_top10 = metrics.loc[
        metrics["scope"].eq("overall") & metrics["top_frac"].eq(0.10)
    ].iloc[0]
    side_top10 = metrics.loc[
        metrics["scope"].eq("side") & metrics["top_frac"].eq(0.10)
    ]
    assert int(overall_top10["selected_rows"]) == 2
    assert set(side_top10["side_name"]) == {"long", "short"}
    assert set(metrics["selection_basis"]) == {"per_side_topk"}
