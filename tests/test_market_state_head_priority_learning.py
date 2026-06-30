from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from scripts.run_market_state_head_priority_learning import (
    _frontier_action_metric_row,
    _frontier_candidate_utilities,
    _filter_state_feature_columns,
    _load_active_state_heads,
    _prepare_staged_output_dir,
    _publish_staged_output_dir,
    _priority_grid,
    add_head_only_incremental_validation,
    build_head_priority_targets,
    build_score_head_frame,
    make_design_matrix,
    priority_starvation_attribution,
    replay_selection_gate_passed,
    replay_selection_score,
    score_priority_schedule,
    selection_gate_passed,
    selection_objective,
    static_baseline_candidate_parity,
    validate_priority_model_by_fold,
)


def test_staged_output_dir_publishes_atomically(tmp_path) -> None:
    final_dir = tmp_path / "priority_report"
    final, staging, report = _prepare_staged_output_dir(final_dir)

    assert final == final_dir
    assert staging.exists()
    assert staging.name.startswith(".priority_report.staging-")
    assert report["final_output_dir"] == str(final_dir)
    assert report["publish_mode"] == "atomic_replace_after_manifest"

    (staging / "manifest.json").write_text("{}", encoding="utf-8")
    _publish_staged_output_dir(staging, final)

    assert final.exists()
    assert not staging.exists()
    assert (final / "manifest.json").exists()


def test_staged_output_dir_rejects_non_empty_final_dir(tmp_path) -> None:
    final_dir = tmp_path / "priority_report"
    final_dir.mkdir()
    (final_dir / "existing.txt").write_text("keep", encoding="utf-8")

    try:
        _prepare_staged_output_dir(final_dir)
    except FileExistsError as exc:
        assert str(final_dir) in str(exc)
    else:
        raise AssertionError("expected non-empty final output directory to be rejected")

    assert (final_dir / "existing.txt").read_text(encoding="utf-8") == "keep"


def test_priority_grid_searches_action_amplitude() -> None:
    class Args:
        grid_target_modes = ""
        grid_min_ranks = "0.50"
        grid_frontier_gammas = "1.5"
        grid_frontier_bandwidths = "0.04"
        grid_sl_penalties = "0.01"
        grid_timeout_penalties = "0.002"
        grid_max_adjustments = "0.2,0.5"
        grid_max_priority_multipliers = "1.0,2.0"
        grid_priority_actions = "multiplier,both"
        min_rank = 0.50
        frontier_gamma = 1.5
        frontier_bandwidth = 0.04
        sl_penalty = 0.01
        timeout_penalty = 0.002
        max_adjustment = 0.2
        max_priority_multiplier = 1.0
        priority_action = "both"
        min_candidates_per_head_timestamp = 3
        target_clip = 0.08
        target_mode = "frontier_weighted_mean"

    grid = _priority_grid(Args(), ["lgbm"])

    assert [
        (row["max_adjustment"], row["max_priority_multiplier"], row["priority_action"])
        for row in grid
    ] == [
        (0.2, 1.0, "multiplier"),
        (0.2, 1.0, "both"),
        (0.2, 2.0, "multiplier"),
        (0.2, 2.0, "both"),
        (0.5, 1.0, "multiplier"),
        (0.5, 1.0, "both"),
        (0.5, 2.0, "multiplier"),
        (0.5, 2.0, "both"),
    ]


def test_priority_grid_can_search_target_modes() -> None:
    class Args:
        grid_target_modes = "frontier_weighted_mean,threshold_admission_mean"
        grid_min_ranks = "0.50"
        grid_frontier_gammas = "1.5"
        grid_frontier_bandwidths = "0.04"
        grid_sl_penalties = "0.01"
        grid_timeout_penalties = "0.002"
        grid_max_adjustments = "0.2"
        grid_max_priority_multipliers = "1.0"
        grid_max_rank_adjustments = "0.0"
        grid_priority_actions = "adjustment"
        min_rank = 0.50
        frontier_gamma = 1.5
        frontier_bandwidth = 0.04
        sl_penalty = 0.01
        timeout_penalty = 0.002
        max_adjustment = 0.2
        max_priority_multiplier = 1.0
        max_rank_adjustment = 0.0
        priority_action = "adjustment"
        min_candidates_per_head_timestamp = 3
        target_clip = 0.08
        target_mode = "frontier_weighted_mean"

    grid = _priority_grid(Args(), ["lgbm"])

    assert [row["target_mode"] for row in grid] == [
        "frontier_weighted_mean",
        "threshold_admission_mean",
    ]


def _residual_ledger() -> pd.DataFrame:
    rows = []
    for fold in [1]:
        for ts in pd.to_datetime(["2026-05-01T00:00:00Z", "2026-05-01T01:00:00Z"], utc=True):
            for head, resid in [("short_asset", -0.01), ("short_boll", 0.02)]:
                for i in range(3):
                    rows.append(
                        {
                            "fold": fold,
                            "arm": "S1_observed_axes_shared_response",
                            "timestamp": ts,
                            "head": head,
                            "strategy_id": f"{head}_s1",
                            "symbol": f"S{i}",
                            "_rank": 0.68 + 0.02 * i,
                            "_threshold": 0.70,
                            "_net_return": resid + 0.001 * i,
                            "_is_full_sl": 0.0,
                            "_is_timeout": 0.0,
                            "resid_utility": resid + 0.001 * i,
                            "resid_full_sl": 0.0,
                            "resid_timeout": 0.0,
                        }
                    )
    return pd.DataFrame(rows)


def _state_panel() -> pd.DataFrame:
    rows = []
    for ts, shock in zip(
        pd.to_datetime(["2026-05-01T00:00:00Z", "2026-05-01T01:00:00Z"], utc=True),
        [0.1, 0.2],
    ):
        rows.append(
            {
                "fold": 1,
                "split": "train",
                "state_arm": "S1_observed_axes_shared_response",
                "timestamp": ts,
                "state_shock": shock,
                "state_realized_vol": shock + 0.5,
                "prediction_contract": "train",
            }
        )
    return pd.DataFrame(rows)


def test_build_head_priority_targets_are_centered_by_timestamp() -> None:
    targets, features = build_head_priority_targets(
        _residual_ledger(),
        _state_panel(),
        min_rank=0.50,
        min_candidates_per_head_timestamp=3,
        sl_penalty=0.0,
        timeout_penalty=0.0,
    )

    assert set(features) == {"state_shock", "state_realized_vol"}
    assert set(targets["head"]) == {"short_asset", "short_boll"}
    centered = targets.groupby("timestamp")["priority_target"].mean()
    assert np.allclose(centered.to_numpy(dtype=float), 0.0)
    by_head = targets.groupby("head")["priority_target"].mean()
    assert by_head["short_boll"] > 0.0
    assert by_head["short_asset"] < 0.0


def test_build_head_priority_targets_filters_by_active_state_heads() -> None:
    targets, features = build_head_priority_targets(
        _residual_ledger(),
        _state_panel(),
        allowed_state_heads={"state_shock"},
        min_rank=0.50,
        min_candidates_per_head_timestamp=3,
        sl_penalty=0.0,
        timeout_penalty=0.0,
    )

    assert features == ["state_shock"]
    assert "state_realized_vol" not in targets.columns
    assert set(targets["head"]) == {"short_asset", "short_boll"}


def test_load_active_state_heads_from_activation_registry(tmp_path) -> None:
    path = tmp_path / "market_state_activation_registry.csv"
    pd.DataFrame(
        [
            {"state_head": "forecast_h6_shock_up", "recommended_status": "active_candidate"},
            {"state_head": "state_realized_vol", "recommended_status": "disabled_candidate"},
            {"state_head": "state_drift_score", "recommended_status": "shadow"},
        ]
    ).to_csv(path, index=False)

    heads, report = _load_active_state_heads(path)

    assert heads == {"forecast_h6_shock_up"}
    assert report["enabled"] is True
    assert report["allowed_state_heads"] == ["forecast_h6_shock_up"]
    assert report["status_counts"] == {
        "active_candidate": 1,
        "disabled_candidate": 1,
        "shadow": 1,
    }


def test_filter_state_feature_columns_keeps_only_allowed_heads() -> None:
    assert _filter_state_feature_columns(
        ["state_shock", "forecast_h6_shock_up", "state_realized_vol"],
        {"forecast_h6_shock_up"},
    ) == ["forecast_h6_shock_up"]
    assert _filter_state_feature_columns(["state_shock"], None) == ["state_shock"]


def test_static_baseline_candidate_parity_accepts_deployable_scope() -> None:
    candidates = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-06-15T00:00:00Z", "2026-06-15T01:00:00Z"], utc=True),
            "head": ["short_asset", "short_boll"],
        }
    )
    report = static_baseline_candidate_parity(
        candidates,
        candidates_path=Path("deployable.parquet"),
        static_baseline_info={
            "manifest_deployable_rows": 2,
            "manifest_candidate_rows": 3,
            "candidates_deployable_path": "deployable.parquet",
            "candidates_broad_path": "broad.parquet",
        },
    )

    assert report["candidate_scope"] == "deployable_static_baseline"
    assert report["promotion_grade_scope"] is True
    assert report["failures"] == []


def test_static_baseline_candidate_parity_flags_broad_non_deployable_scope() -> None:
    candidates = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2026-06-15T00:00:00Z", "2026-06-15T01:00:00Z", "2026-06-15T02:00:00Z"],
                utc=True,
            ),
            "head": ["short_asset", "short_boll", "short_boll"],
        }
    )
    report = static_baseline_candidate_parity(
        candidates,
        candidates_path=Path("broad.parquet"),
        static_baseline_info={
            "manifest_deployable_rows": 2,
            "manifest_candidate_rows": 3,
            "candidates_deployable_path": "deployable.parquet",
            "candidates_broad_path": "broad.parquet",
        },
    )

    assert report["candidate_scope"] == "broad_non_deployable_diagnostic"
    assert report["promotion_grade_scope"] is False
    assert report["failures"] == ["candidate_universe_not_deployable_static_baseline_scope"]


def test_head_top_candidate_target_uses_best_ranked_candidate_per_head() -> None:
    ts = pd.Timestamp("2026-05-01T00:00:00Z")
    residual = pd.DataFrame(
        [
            {
                "fold": 1,
                "arm": "S1_observed_axes_shared_response",
                "timestamp": ts,
                "head": "short_asset",
                "strategy_id": "short_asset_bad_top",
                "symbol": "A",
                "_rank": 0.95,
                "_threshold": 0.70,
                "_net_return": -0.03,
                "_is_full_sl": 1.0,
                "_is_timeout": 0.0,
                "resid_utility": -0.03,
                "resid_full_sl": 1.0,
                "resid_timeout": 0.0,
            },
            {
                "fold": 1,
                "arm": "S1_observed_axes_shared_response",
                "timestamp": ts,
                "head": "short_asset",
                "strategy_id": "short_asset_good_lower",
                "symbol": "B",
                "_rank": 0.71,
                "_threshold": 0.70,
                "_net_return": 0.04,
                "_is_full_sl": 0.0,
                "_is_timeout": 0.0,
                "resid_utility": 0.04,
                "resid_full_sl": 0.0,
                "resid_timeout": 0.0,
            },
            {
                "fold": 1,
                "arm": "S1_observed_axes_shared_response",
                "timestamp": ts,
                "head": "short_boll",
                "strategy_id": "short_boll_good_top",
                "symbol": "C",
                "_rank": 0.90,
                "_threshold": 0.70,
                "_net_return": 0.02,
                "_is_full_sl": 0.0,
                "_is_timeout": 0.0,
                "resid_utility": 0.02,
                "resid_full_sl": 0.0,
                "resid_timeout": 0.0,
            },
            {
                "fold": 1,
                "arm": "S1_observed_axes_shared_response",
                "timestamp": ts,
                "head": "short_boll",
                "strategy_id": "short_boll_bad_lower",
                "symbol": "D",
                "_rank": 0.70,
                "_threshold": 0.70,
                "_net_return": -0.04,
                "_is_full_sl": 1.0,
                "_is_timeout": 0.0,
                "resid_utility": -0.04,
                "resid_full_sl": 1.0,
                "resid_timeout": 0.0,
            },
        ]
    )

    targets, _features = build_head_priority_targets(
        residual,
        _state_panel(),
        target_mode="head_top_candidate",
        min_rank=0.50,
        min_candidates_per_head_timestamp=2,
        sl_penalty=0.0,
        timeout_penalty=0.0,
    )

    by_head = targets.set_index("head")["priority_target"]
    assert by_head["short_boll"] > 0.0
    assert by_head["short_asset"] < 0.0
    assert targets.set_index("head").loc["short_asset", "top_candidate_rank"] == 0.95


def test_threshold_admission_target_focuses_on_marginal_rows() -> None:
    ts = pd.Timestamp("2026-05-01T00:00:00Z")
    residual = pd.DataFrame(
        [
            {
                "fold": 1,
                "arm": "S1_observed_axes_shared_response",
                "timestamp": ts,
                "head": "short_asset",
                "strategy_id": "short_asset_good_top",
                "symbol": "A",
                "_rank": 0.99,
                "_threshold": 0.70,
                "_net_return": 0.08,
                "_is_full_sl": 0.0,
                "_is_timeout": 0.0,
                "resid_utility": 0.08,
                "resid_full_sl": 0.0,
                "resid_timeout": 0.0,
            },
            {
                "fold": 1,
                "arm": "S1_observed_axes_shared_response",
                "timestamp": ts,
                "head": "short_asset",
                "strategy_id": "short_asset_bad_boundary",
                "symbol": "B",
                "_rank": 0.695,
                "_threshold": 0.70,
                "_net_return": -0.04,
                "_is_full_sl": 1.0,
                "_is_timeout": 0.0,
                "resid_utility": -0.04,
                "resid_full_sl": 1.0,
                "resid_timeout": 0.0,
            },
            {
                "fold": 1,
                "arm": "S1_observed_axes_shared_response",
                "timestamp": ts,
                "head": "short_asset",
                "strategy_id": "short_asset_bad_boundary_2",
                "symbol": "C",
                "_rank": 0.705,
                "_threshold": 0.70,
                "_net_return": -0.03,
                "_is_full_sl": 1.0,
                "_is_timeout": 0.0,
                "resid_utility": -0.03,
                "resid_full_sl": 1.0,
                "resid_timeout": 0.0,
            },
            {
                "fold": 1,
                "arm": "S1_observed_axes_shared_response",
                "timestamp": ts,
                "head": "short_boll",
                "strategy_id": "short_boll_bad_top",
                "symbol": "D",
                "_rank": 0.99,
                "_threshold": 0.70,
                "_net_return": -0.08,
                "_is_full_sl": 1.0,
                "_is_timeout": 0.0,
                "resid_utility": -0.08,
                "resid_full_sl": 1.0,
                "resid_timeout": 0.0,
            },
            {
                "fold": 1,
                "arm": "S1_observed_axes_shared_response",
                "timestamp": ts,
                "head": "short_boll",
                "strategy_id": "short_boll_good_boundary",
                "symbol": "E",
                "_rank": 0.695,
                "_threshold": 0.70,
                "_net_return": 0.04,
                "_is_full_sl": 0.0,
                "_is_timeout": 0.0,
                "resid_utility": 0.04,
                "resid_full_sl": 0.0,
                "resid_timeout": 0.0,
            },
            {
                "fold": 1,
                "arm": "S1_observed_axes_shared_response",
                "timestamp": ts,
                "head": "short_boll",
                "strategy_id": "short_boll_good_boundary_2",
                "symbol": "F",
                "_rank": 0.705,
                "_threshold": 0.70,
                "_net_return": 0.03,
                "_is_full_sl": 0.0,
                "_is_timeout": 0.0,
                "resid_utility": 0.03,
                "resid_full_sl": 0.0,
                "resid_timeout": 0.0,
            },
        ]
    )

    targets, _features = build_head_priority_targets(
        residual,
        _state_panel(),
        target_mode="threshold_admission_mean",
        min_rank=0.50,
        frontier_bandwidth=0.02,
        min_candidates_per_head_timestamp=3,
        sl_penalty=0.0,
        timeout_penalty=0.0,
    )

    by_head = targets.set_index("head")["priority_target"]
    assert by_head["short_boll"] > 0.0
    assert by_head["short_asset"] < 0.0
    assert targets["threshold_below_weighted_share"].between(0.0, 1.0).all()


def test_rank_residual_target_rewards_underprioritized_head_utility() -> None:
    timestamps = pd.to_datetime(
        [
            "2026-05-01T00:00:00Z",
            "2026-05-01T01:00:00Z",
            "2026-05-01T02:00:00Z",
        ],
        utc=True,
    )
    rows = []
    # First two timestamps establish the usual rank-positive relationship.
    for ts in timestamps[:2]:
        rows.extend(
            [
                {
                    "fold": 1,
                    "arm": "S1_observed_axes_shared_response",
                    "timestamp": ts,
                    "head": "short_asset",
                    "strategy_id": "short_asset_ranked_high",
                    "symbol": "A",
                    "_rank": 0.95,
                    "_threshold": 0.70,
                    "_net_return": 0.02,
                    "_is_full_sl": 0.0,
                    "_is_timeout": 0.0,
                    "resid_utility": 0.02,
                    "resid_full_sl": 0.0,
                    "resid_timeout": 0.0,
                },
                {
                    "fold": 1,
                    "arm": "S1_observed_axes_shared_response",
                    "timestamp": ts,
                    "head": "short_boll",
                    "strategy_id": "short_boll_ranked_low",
                    "symbol": "B",
                    "_rank": 0.70,
                    "_threshold": 0.70,
                    "_net_return": -0.01,
                    "_is_full_sl": 0.0,
                    "_is_timeout": 0.0,
                    "resid_utility": -0.01,
                    "resid_full_sl": 0.0,
                    "resid_timeout": 0.0,
                },
            ]
        )
    # Last timestamp is the June-like case: short_boll is lower-ranked but better.
    rows.extend(
        [
            {
                "fold": 1,
                "arm": "S1_observed_axes_shared_response",
                "timestamp": timestamps[2],
                "head": "short_asset",
                "strategy_id": "short_asset_ranked_high",
                "symbol": "A",
                "_rank": 0.95,
                "_threshold": 0.70,
                "_net_return": 0.00,
                "_is_full_sl": 0.0,
                "_is_timeout": 0.0,
                "resid_utility": 0.00,
                "resid_full_sl": 0.0,
                "resid_timeout": 0.0,
            },
            {
                "fold": 1,
                "arm": "S1_observed_axes_shared_response",
                "timestamp": timestamps[2],
                "head": "short_boll",
                "strategy_id": "short_boll_ranked_low",
                "symbol": "B",
                "_rank": 0.70,
                "_threshold": 0.70,
                "_net_return": 0.03,
                "_is_full_sl": 0.0,
                "_is_timeout": 0.0,
                "resid_utility": 0.03,
                "resid_full_sl": 0.0,
                "resid_timeout": 0.0,
            },
        ]
    )
    state = pd.DataFrame(
        {
            "fold": [1, 1, 1],
            "split": ["train", "train", "train"],
            "state_arm": ["S1_observed_axes_shared_response"] * 3,
            "timestamp": timestamps,
            "state_shock": [0.1, 0.2, 0.9],
        }
    )

    frontier_targets, _ = build_head_priority_targets(
        pd.DataFrame(rows),
        state,
        target_mode="frontier_weighted_mean",
        min_rank=0.50,
        min_candidates_per_head_timestamp=1,
        sl_penalty=0.0,
        timeout_penalty=0.0,
    )
    residual_targets, _ = build_head_priority_targets(
        pd.DataFrame(rows),
        state,
        target_mode="rank_residual_frontier",
        min_rank=0.50,
        min_candidates_per_head_timestamp=1,
        sl_penalty=0.0,
        timeout_penalty=0.0,
        rank_residual_weight=1.0,
    )

    key = residual_targets["timestamp"].eq(timestamps[2]) & residual_targets["head"].eq("short_boll")
    residual_value = float(residual_targets.loc[key, "priority_target"].iloc[0])
    frontier_value = float(frontier_targets.loc[key, "priority_target"].iloc[0])
    assert residual_value > frontier_value
    assert float(residual_targets["rank_residual_beta"].iloc[0]) > 0.0


def test_build_score_head_frame_cross_joins_market_state_with_heads() -> None:
    candidates = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-06-01T00:00:00Z", "2026-06-01T00:00:00Z"], utc=True),
            "head": ["short_asset", "short_boll"],
        }
    )
    score_state = pd.DataFrame(
        {
            "split": ["score"],
            "state_level": ["observed"],
            "timestamp": pd.to_datetime(["2026-06-01T00:00:00Z"], utc=True),
            "state_shock": [0.3],
            "state_realized_vol": [0.9],
        }
    )

    frame = build_score_head_frame(score_state, candidates, ["state_shock", "state_realized_vol"])

    assert len(frame) == 2
    assert set(frame["head"]) == {"short_asset", "short_boll"}
    assert frame.groupby("timestamp")["state_shock"].nunique().iloc[0] == 1


def test_build_score_head_frame_uses_forecast_rows_for_forecast_features() -> None:
    candidates = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-06-01T00:00:00Z", "2026-06-01T00:00:00Z"], utc=True),
            "head": ["short_asset", "short_boll"],
        }
    )
    score_state = pd.DataFrame(
        {
            "split": ["score", "score"],
            "state_level": ["observed", "forecast"],
            "timestamp": pd.to_datetime(["2026-06-01T00:00:00Z", "2026-06-01T00:00:00Z"], utc=True),
            "state_shock": [0.3, 0.4],
            "forecast_h6_shock_up": [np.nan, 0.8],
        }
    )

    frame = build_score_head_frame(
        score_state,
        candidates,
        ["state_shock", "forecast_h6_shock_up"],
    )

    assert len(frame) == 2
    assert set(frame["head"]) == {"short_asset", "short_boll"}
    assert frame["forecast_h6_shock_up"].eq(0.8).all()
    assert frame["state_shock"].eq(0.4).all()


class _DummyModel:
    def predict(self, x: np.ndarray) -> np.ndarray:
        # Last two columns are head one-hot in sorted order.
        return x[:, -1] - x[:, -2]


def test_score_priority_schedule_is_centered_and_bounded() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-06-01T00:00:00Z", "2026-06-01T00:00:00Z"], utc=True),
            "head": ["short_asset", "short_boll"],
            "state_shock": [0.1, 0.1],
        }
    )
    heads = ["short_asset", "short_boll"]
    _x, medians, _cols = make_design_matrix(frame, feature_cols=["state_shock"], heads=heads)

    schedule = score_priority_schedule(
        _DummyModel(),
        frame,
        feature_cols=["state_shock"],
        heads=heads,
        medians=medians,
        pred_scale=1.0,
        max_adjustment=0.2,
        arm="test",
    )

    assert schedule["portfolio_priority_adjustment"].abs().max() <= 0.2 + 1e-12
    assert np.isclose(schedule["centered_head_score"].mean(), 0.0)
    assert schedule.loc[schedule["head"].eq("short_boll"), "portfolio_priority_adjustment"].iloc[0] > 0.0


def test_make_design_matrix_fills_missing_score_columns_with_training_medians() -> None:
    train = pd.DataFrame(
        {
            "head": ["short_asset", "short_boll"],
            "state_shock": [1.0, 3.0],
            "forecast_h6_missing_live": [2.0, 4.0],
        }
    )
    _x_train, medians, cols = make_design_matrix(
        train,
        feature_cols=["state_shock", "forecast_h6_missing_live"],
        heads=["short_asset", "short_boll"],
    )
    live = pd.DataFrame({"head": ["short_asset"], "state_shock": [5.0]})

    x_live, _medians, _cols = make_design_matrix(
        live,
        feature_cols=["state_shock", "forecast_h6_missing_live"],
        heads=["short_asset", "short_boll"],
        medians=medians,
    )

    assert cols[:2] == ["state_shock", "forecast_h6_missing_live"]
    assert x_live[0, 0] == 5.0
    assert x_live[0, 1] == 3.0


def test_selection_objective_prefers_better_scaled_validation_metrics() -> None:
    weak = {
        "validation_spearman": -0.2,
        "validation_directional_accuracy": 0.45,
        "validation_mae": 0.07,
    }
    strong = {
        "validation_spearman": 0.3,
        "validation_directional_accuracy": 0.60,
        "validation_mae": 0.02,
    }

    assert selection_objective(strong, target_clip=0.08) > selection_objective(
        weak,
        target_clip=0.08,
    )
    assert 0.0 <= selection_objective(strong, target_clip=0.08) <= 1.0


def test_selection_gate_requires_positive_orientation_and_support() -> None:
    assert selection_gate_passed(
        {
            "validation_rows": 34,
            "validation_spearman": 0.05,
            "validation_directional_accuracy": 0.53,
        }
    )
    assert not selection_gate_passed(
        {
            "validation_rows": 34,
            "validation_spearman": 0.05,
            "validation_directional_accuracy": 0.47,
        }
    )
    assert not selection_gate_passed(
        {
            "validation_rows": 34,
            "validation_spearman": -0.01,
            "validation_directional_accuracy": 0.53,
        }
    )


def test_fold_aware_selection_metrics_drive_objective_and_gate() -> None:
    fold_row = {
        "fold_count": 3,
        "fold_validation_rows": 90,
        "fold_mean_spearman": 0.20,
        "fold_positive_spearman_share": 2 / 3,
        "fold_mean_directional_accuracy": 0.56,
        "fold_directional_ge_50_share": 2 / 3,
        "fold_mean_mae": 0.03,
        "fold_incremental_objective": 0.10,
        "fold_incremental_spearman": 0.08,
        "fold_incremental_mae_reduction": 0.01,
        "validation_rows": 20,
        "validation_spearman": -0.10,
        "validation_directional_accuracy": 0.55,
        "validation_mae": 0.07,
    }

    assert selection_gate_passed(fold_row)
    assert selection_objective(fold_row, target_clip=0.08) > 0.0

    failing = dict(fold_row)
    failing["fold_positive_spearman_share"] = 1 / 3
    assert not selection_gate_passed(failing)

    failing_incremental = dict(fold_row)
    failing_incremental["fold_incremental_objective"] = 0.0
    assert not selection_gate_passed(failing_incremental)


def test_opportunity_selection_gate_does_not_require_tiny_trailing_split_pass() -> None:
    row = {
        "fold_count": 3,
        "fold_validation_rows": 120,
        "fold_mean_spearman": 0.25,
        "fold_positive_spearman_share": 2 / 3,
        "fold_mean_directional_accuracy": 0.58,
        "fold_directional_ge_50_share": 2 / 3,
        "fold_mean_mae": 0.03,
        "fold_incremental_objective": 0.12,
        "fold_incremental_spearman": 0.10,
        "fold_incremental_mae_reduction": 0.01,
        "fold_action_timestamps": 40,
        "fold_mean_action_utility_delta": 0.01,
        "fold_action_positive_delta_share": 0.70,
        "fold_mean_action_full_sl_delta": 0.025,
        "validation_rows": 12,
        "validation_spearman": -0.15,
        "validation_directional_accuracy": 0.45,
        "validation_mae": 0.07,
    }

    assert not selection_gate_passed(row, gate_mode="defensive")
    assert selection_gate_passed(row, gate_mode="opportunity")


def test_replay_selection_gate_requires_portfolio_swap_quality() -> None:
    good = {
        "replay_baseline_trade_count": 100,
        "replay_trade_count_delta": 0,
        "replay_net_pnl_delta": 12.0,
        "replay_full_sl_delta": 0.0,
        "replay_timeout_delta": 0.0,
        "replay_accepted_jaccard": 0.98,
        "replay_entrants": 1,
        "replay_removed": 1,
        "replay_entrant_net_pnl": 8.0,
        "replay_removed_net_pnl": 1.0,
        "replay_net_replacement_pnl": 7.0,
        "replay_net_action_pnl_delta": 12.0,
    }

    assert replay_selection_gate_passed(good)

    no_swap = dict(good, replay_entrants=0, replay_removed=0)
    assert not replay_selection_gate_passed(no_swap)

    worse_replacement = dict(good, replay_entrant_net_pnl=0.5)
    assert not replay_selection_gate_passed(worse_replacement)

    riskier = dict(good, replay_full_sl_delta=0.01)
    assert not replay_selection_gate_passed(riskier)


def test_opportunity_replay_gate_allows_small_timeout_drift_for_better_replacements() -> None:
    opportunity = {
        "replay_baseline_trade_count": 100,
        "replay_trade_count_delta": 4,
        "replay_net_pnl_delta": 20.0,
        "replay_full_sl_delta": 0.0,
        "replay_timeout_delta": 0.006,
        "replay_accepted_jaccard": 0.96,
        "replay_entrants": 5,
        "replay_removed": 1,
        "replay_entrant_net_pnl": 14.0,
        "replay_removed_net_pnl": 1.0,
        "replay_net_replacement_pnl": 13.0,
        "replay_net_action_pnl_delta": 20.0,
    }

    assert not replay_selection_gate_passed(opportunity)
    assert replay_selection_gate_passed(opportunity, gate_mode="opportunity")
    assert not replay_selection_gate_passed(
        opportunity,
        gate_mode="opportunity",
        relax_opportunity_risk_gates=False,
    )


def test_replay_selection_score_prefers_portfolio_delta_with_risk_control() -> None:
    clean = {
        "replay_net_pnl_delta": 10.0,
        "replay_net_action_pnl_delta": 8.0,
        "replay_net_replacement_pnl": 4.0,
        "replay_full_sl_delta": 0.0,
        "replay_timeout_delta": 0.0,
        "replay_accepted_jaccard": 0.98,
    }
    riskier = dict(clean, replay_net_pnl_delta=11.0, replay_full_sl_delta=0.05)

    assert replay_selection_score(clean) > replay_selection_score(riskier)


def test_head_only_incremental_validation_prefers_state_signal() -> None:
    full = {
        "fold_mean_spearman": 0.40,
        "fold_mean_directional_accuracy": 0.60,
        "fold_mean_mae": 0.02,
    }
    head_only = {
        "fold_mean_spearman": 0.10,
        "fold_mean_directional_accuracy": 0.55,
        "fold_mean_mae": 0.03,
    }

    out = add_head_only_incremental_validation(full, head_only, target_clip=0.08)

    assert out["fold_incremental_objective"] > 0.0
    assert np.isclose(out["fold_incremental_spearman"], 0.30)
    assert out["fold_incremental_directional_accuracy"] > 0.0
    assert out["fold_incremental_mae_reduction"] > 0.0


def test_validate_priority_model_by_fold_returns_recurrent_metrics() -> None:
    rows = []
    state_rows = []
    for fold in [1, 2, 3]:
        for hour in range(4):
            ts = pd.Timestamp("2026-05-01T00:00:00Z") + pd.Timedelta(days=fold, hours=hour)
            state_rows.append(
                {
                    "fold": fold,
                    "split": "train",
                    "state_arm": "S1_observed_axes_shared_response",
                    "timestamp": ts,
                    "state_shock": float(hour),
                    "state_realized_vol": float(fold),
                }
            )
            for head, base in [("short_asset", -0.02), ("short_boll", 0.02)]:
                for i in range(3):
                    rows.append(
                        {
                            "fold": fold,
                            "arm": "S1_observed_axes_shared_response",
                            "timestamp": ts,
                            "head": head,
                            "strategy_id": f"{head}_{i}",
                            "symbol": f"S{i}",
                            "_rank": 0.70 + 0.01 * i,
                            "_threshold": 0.70,
                            "_net_return": base,
                            "_is_full_sl": 0.0,
                            "_is_timeout": 0.0,
                            "resid_utility": base + 0.002 * hour,
                            "resid_full_sl": 0.0,
                            "resid_timeout": 0.0,
                        }
                    )
    targets, features = build_head_priority_targets(
        pd.DataFrame(rows),
        pd.DataFrame(state_rows),
        min_rank=0.50,
        min_candidates_per_head_timestamp=3,
        sl_penalty=0.0,
        timeout_penalty=0.0,
    )

    summary, fold_df = validate_priority_model_by_fold(
        targets,
        feature_cols=features,
        backend="lgbm",
        target_clip=0.08,
        seed=1,
        min_train_rows=8,
        min_valid_rows=4,
    )

    assert summary["fold_count"] == 3
    assert summary["fold_validation_rows"] == len(targets)
    assert len(fold_df) == 3
    assert set(fold_df["validation_fold"]) == {1, 2, 3}


def test_frontier_action_metrics_reward_useful_short_boll_lift() -> None:
    rows = []
    ts = pd.Timestamp("2026-05-01T00:00:00Z")
    for head, rank, utility in [
        ("short_asset", 0.95, -0.010),
        ("short_boll", 0.80, 0.025),
    ]:
        for i in range(3):
            rows.append(
                {
                    "fold": 1,
                    "arm": "S1_observed_axes_shared_response",
                    "timestamp": ts,
                    "head": head,
                    "strategy_id": f"{head}_{i}",
                    "symbol": f"S{i}",
                    "_rank": rank,
                    "_threshold": 0.70,
                    "_net_return": utility,
                    "_is_full_sl": 0.0,
                    "_is_timeout": 0.0,
                    "resid_utility": utility,
                    "resid_full_sl": 0.0,
                    "resid_timeout": 0.0,
                }
            )
    frontier = _frontier_candidate_utilities(
        pd.DataFrame(rows),
        state_arm="S1_observed_axes_shared_response",
        min_rank=0.50,
        frontier_gamma=1.0,
        frontier_bandwidth=0.08,
        sl_penalty=0.0,
        timeout_penalty=0.0,
        min_candidates_per_head_timestamp=3,
    )
    frontier["predicted_priority_adjustment"] = np.where(
        frontier["head"].eq("short_boll"),
        0.25,
        0.0,
    )

    metrics = _frontier_action_metric_row(frontier)

    assert metrics["action_timestamps"] == 1
    assert metrics["action_mean_utility_delta"] > 0.0
    assert metrics["action_selected_head_switch_share"] == 1.0
    assert metrics["action_baseline_selected_head_max_share"] == 1.0
    assert metrics["action_model_selected_head_max_share"] == 1.0
    assert metrics["action_selected_head_share_l1_shift"] == 2.0
    assert metrics["action_selected_head_share_by_head"]["short_asset"]["delta"] == -1.0
    assert metrics["action_selected_head_share_by_head"]["short_boll"]["delta"] == 1.0
    assert metrics["action_baseline_short_boll_share"] == 0.0
    assert metrics["action_model_short_boll_share"] == 1.0


def _priority_starvation_candidates() -> pd.DataFrame:
    ts = pd.Timestamp("2026-06-20T00:00:00Z")
    return pd.DataFrame(
        {
            "timestamp": [ts, ts],
            "symbol": ["AAA/USD:USD", "BBB/USD:USD"],
            "side": ["short", "short"],
            "strategy_id": ["short_asset_s1", "short_boll_s1"],
            "head": ["short_asset", "short_boll"],
            "normalized_rank_score": [0.95, 0.90],
            "base_strategy_threshold": [0.70, 0.70],
            "calibrated_score": [0.8, 0.7],
            "entry_price": [100.0, 100.0],
            "exit_timestamp": [ts + pd.Timedelta(hours=1), ts + pd.Timedelta(hours=1)],
            "exit_price": [99.0, 99.0],
            "net_return": [-0.01, 0.03],
            "gross_return": [-0.009, 0.031],
            "holding_bars": [4, 4],
            "simple_policy_exit_reason": ["full_sl", "tp"],
            "fees_bps": [1.0, 1.0],
            "slippage_bps": [0.0, 0.0],
            "price_gap_bps": [0.0, 0.0],
            "expected_friction_bps": [0.0, 0.0],
        }
    )


def _priority_starvation_decisions(*, accept_short_boll: bool) -> pd.DataFrame:
    ts = pd.Timestamp("2026-06-20T00:00:00Z")
    if accept_short_boll:
        accepted = [False, True]
        reasons = ["max_new_entries_per_bar_reached", "accepted"]
        priorities = [0.2, 0.5]
    else:
        accepted = [True, False]
        reasons = ["accepted", "max_new_entries_per_bar_reached"]
        priorities = [0.5, 0.2]
    return pd.DataFrame(
        {
            "candidate_index": [0, 1],
            "timestamp": [ts, ts],
            "symbol": ["AAA/USD:USD", "BBB/USD:USD"],
            "side": ["short", "short"],
            "strategy_id": ["short_asset_s1", "short_boll_s1"],
            "normalized_rank_score": [0.95, 0.90],
            "base_threshold": [0.70, 0.70],
            "dynamic_threshold": [0.70, 0.70],
            "portfolio_priority": priorities,
            "accepted": accepted,
            "rejection_reason": reasons,
        }
    )


def test_priority_starvation_attribution_measures_head_capacity_starvation() -> None:
    candidates = _priority_starvation_candidates()

    attr = priority_starvation_attribution(
        candidates_by_arm={"P0_static_priority": candidates},
        decisions_by_arm={
            "P0_static_priority": _priority_starvation_decisions(accept_short_boll=False)
        },
    )

    short_boll = attr.loc[attr["head"].eq("short_boll")].iloc[0]
    assert int(short_boll["threshold_pass_rows"]) == 1
    assert int(short_boll["accepted_rows"]) == 0
    assert int(short_boll["routing_rejected_rows"]) == 1
    assert int(short_boll["routing_rejected_positive_rows"]) == 1
    np.testing.assert_allclose(
        float(short_boll["routing_rejected_positive_net_return_sum"]),
        0.03,
    )


def test_priority_starvation_attribution_reports_delta_vs_baseline() -> None:
    candidates = _priority_starvation_candidates()

    attr = priority_starvation_attribution(
        candidates_by_arm={
            "P0_static_priority": candidates,
            "L1_lgbm_learned_priority": candidates,
        },
        decisions_by_arm={
            "P0_static_priority": _priority_starvation_decisions(accept_short_boll=False),
            "L1_lgbm_learned_priority": _priority_starvation_decisions(accept_short_boll=True),
        },
    )

    short_boll = attr.loc[
        attr["arm"].eq("L1_lgbm_learned_priority") & attr["head"].eq("short_boll")
    ].iloc[0]
    assert int(short_boll["accepted_rows"]) == 1
    assert int(short_boll["routing_rejected_rows"]) == 0
    np.testing.assert_allclose(float(short_boll["delta_vs_baseline_accepted_rows"]), 1.0)
    np.testing.assert_allclose(float(short_boll["delta_vs_baseline_routing_rejected_rows"]), -1.0)
    np.testing.assert_allclose(
        float(short_boll["delta_vs_baseline_routing_rejected_positive_net_return_sum"]),
        -0.03,
    )
