from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import pytest

from scripts import run_one_head_contextual_meta_ablation as mod


def test_meta_target_uses_only_current_y_bin() -> None:
    panel = pd.DataFrame(
        {
            "y_bin": [1.0, 0.0, np.nan, 0.8],
            "high_conf_miss": [0, 1, 0, 1],
            "high_conf_tail_loss": [1, 0, 1, 0],
        }
    )

    y = mod._meta_target(panel)

    assert y.tolist() == [1, 0, -1, 1]


def test_forbidden_training_targets_are_explicitly_blocked(tmp_path) -> None:
    args = argparse.Namespace(
        target_name="high_conf_tail_loss",
        output_dir=str(tmp_path),
    )

    with pytest.raises(RuntimeError, match="Forbidden"):
        mod.run(args)


def test_targeted_interactions_are_ordinary_inputs() -> None:
    panel = pd.DataFrame(
        {
            "oof_pred": [0.1, 0.2, 0.3],
            "oof_base_clf": [0.2, 0.3, 0.4],
            "oof_score_path_std": [0.4, 0.5, 0.6],
        }
    )
    canonical = pd.DataFrame(
        {
            "prediction_support_quality": [0.7, 0.8, 0.9],
            "leverage_funding_crowding": [0.1, 0.2, 0.3],
            "liquidity_participation_stress": [0.2, 0.3, 0.4],
            "prediction_path_instability": [0.3, 0.4, 0.5],
            "tail_volatility_stress": [0.4, 0.5, 0.6],
            "prediction_reconstruction_anomaly": [0.1, 0.2, 0.3],
            "relative_value_dislocation": [0.2, 0.3, 0.4],
            "regime_similarity_or_novelty": [0.3, 0.4, 0.5],
        }
    )

    out = mod._interaction_features(panel, canonical)

    assert "prediction_support_quality__x__leverage_funding_crowding" in out
    assert "oof_pred__x__prediction_support_quality" in out
    assert "oof_score_path_std__x__market_state_stress" in out
    assert out.shape[0] == 3


def test_requirement_audit_requires_same_label_and_cells() -> None:
    arms = (
        list(mod.FEATURE_ARMS)
        + [f"F_{mod.ARM_D}__{v}" for v in mod.DISTILLATION_VARIANTS]
        + list(mod.CONTEXTUAL_SCORE_ARMS)
    )
    variants = (
        ["unchanged_current_meta"]
        + ["hard_label_context_arm"] * 4
        + list(mod.DISTILLATION_VARIANTS)
        + ["rank_preserving_timestamp_logit_shift", "timestamp_shift_plus_regularized_model_state_delta"]
    )
    improvements = {
        mod.ARM_A: 0.0,
        mod.ARM_B: 0.01,
        mod.ARM_C: 0.02,
        mod.ARM_D: 0.05,
        mod.ARM_E: 0.03,
    }
    for variant in mod.DISTILLATION_VARIANTS:
        improvements[f"F_{mod.ARM_D}__{variant}"] = 0.05 if variant == "hard_label_only" else 0.01
    improvements[mod.ARM_G] = 0.02
    improvements[mod.ARM_H] = 0.025
    summary = pd.DataFrame(
        {
            "head": ["short_asset"] * len(arms),
            "arm": arms,
            "distillation_variant": variants,
            "training_target": ["y_bin"] * len(arms),
            "forbidden_targets_used": [False] * len(arms),
            "rows": [1000] * len(arms),
            "auc": [0.7 + improvements[a] for a in arms],
            "log_loss": [0.6 - improvements[a] for a in arms],
            "brier": [0.2 - improvements[a] * 0.5 for a in arms],
            "delta_log_loss_improvement": [improvements[a] for a in arms],
            "delta_brier_improvement": [improvements[a] * 0.5 for a in arms],
        }
    )
    folds = pd.DataFrame(
        {
            "head": ["short_asset"] * (len(arms) * 2),
            "arm": np.repeat(summary["arm"].to_numpy(), 2),
            "distillation_variant": np.repeat(summary["distillation_variant"].to_numpy(), 2),
            "fold": [1, 2] * len(arms),
        }
    )
    cells = pd.DataFrame(
        {
            "cell_family": [
                "period_type",
                "support_quality_decile",
                "market_state_decile",
                "base_score_decile",
                "support_x_market_state_cell",
            ],
        }
    )
    context = pd.DataFrame({"valid_output_feature_count": [len(mod.CANONICAL_CONTEXT)]})
    leave_one = pd.DataFrame(
        {
            "head": ["short_asset"],
            "arm": [mod.ARM_D],
            "distillation_variant": ["hard_label_context_arm"],
            "heldout_episode": ["2026-05-25"],
            "delta_log_loss_improvement": [0.01],
            "delta_brier_improvement": [0.005],
        }
    )
    gradient = pd.DataFrame(
        {
            "leaf": ["score_decile_1"],
            "regime": ["normal_period"],
            "support": [100],
            "gradient_sum": [1.0],
            "hessian_sum": [20.0],
            "optimal_update": [-0.05],
            "update_sign": [-1],
            "cancellation_score": [0.25],
        }
    )
    episode_ci = pd.DataFrame(
        {
            "metric": ["delta_log_loss_improvement"],
            "episode_count": [1],
            "mean": [0.01],
            "median": [0.01],
            "ci05": [0.01],
            "ci95": [0.01],
            "ci_method": ["episode_block_bootstrap"],
        }
    )
    oracle = pd.DataFrame(
        {
            "heldout_episode": ["2026-05-25"],
            "benchmark_model": [mod.ARM_E],
            "status": ["evaluated"],
            "diagnostic_only": [True],
        }
    )
    promotion = pd.DataFrame(
        {
            "head": ["short_asset"],
            "arm": [mod.ARM_D],
            "delta_log_loss_improvement": [0.05],
            "delta_brier_improvement": [0.025],
            "timestamp_weighted_hr_top30": [0.70],
            "delta_timestamp_weighted_hr_top30": [0.05],
            "ndcg_top30": [0.80],
            "delta_ndcg_top30": [0.02],
            "passes_directional_pooled_constraints": [True],
            "passes_directional_episode_constraints": [True],
            "directional_promotion_candidate": [True],
            "normal_period_delta_log_loss_improvement": [0.04],
            "bad_period_delta_log_loss_improvement": [0.03],
            "top10_delta_mean_return": [0.001],
            "top10_delta_winner_magnitude": [0.001],
            "top10_delta_lower_tail_return": [0.0],
            "gradient_conflict_weighted": [0.2],
            "gradient_conflict_high_row_fraction": [0.0],
            "promotion_candidate": [True],
        }
    )
    directional = pd.DataFrame(
        {
            "head": ["short_asset"],
            "arm": [mod.ARM_D],
            "distillation_variant": ["hard_label_context_arm"],
            "timestamp_weighted_hr_top10": [0.75],
            "timestamp_weighted_hr_top20": [0.72],
            "timestamp_weighted_hr_top30": [0.70],
            "trade_weighted_hr_top30": [0.69],
            "delta_timestamp_weighted_hr_top30": [0.05],
            "ndcg_top30": [0.80],
            "delta_ndcg_top30": [0.02],
            "average_precision_top30": [0.76],
            "top30_jaccard": [0.85],
            "net_correct_trades_gained": [5.0],
        }
    )
    directional_timestamp = pd.DataFrame(
        {
            "head": ["short_asset"],
            "arm": [mod.ARM_D],
            "distillation_variant": ["hard_label_context_arm"],
            "timestamp": ["2026-05-25T00:00:00+00:00"],
            "eligible_rows": [10],
            "hr_top30": [0.67],
            "baseline_hr_top30": [0.33],
            "delta_hr_top30": [0.34],
            "ndcg_top30": [0.7],
            "top30_jaccard": [0.5],
        }
    )
    directional_episode = pd.DataFrame(
        {
            "head": ["short_asset"],
            "arm": [mod.ARM_D],
            "distillation_variant": ["hard_label_context_arm"],
            "heldout_episode": ["2026-05-25"],
            "period_type": ["bad_episode"],
            "delta_timestamp_weighted_hr_top30": [0.05],
        }
    )
    directional_episode_ci = pd.DataFrame(
        {
            "head": ["short_asset"],
            "arm": [mod.ARM_D],
            "distillation_variant": ["hard_label_context_arm"],
            "metric": ["delta_timestamp_weighted_hr_top30"],
            "episode_count": [1],
            "mean": [0.05],
            "median": [0.05],
            "positive_episode_rate": [1.0],
            "ci05": [0.05],
            "ci95": [0.05],
        }
    )
    bad_context = pd.DataFrame(
        {
            "head": ["short_asset"],
            "heldout_episode": ["2026-05-25"],
            "arm": [mod.ARM_D],
            "classification_hint": ["no_material_failure"],
            "context_missing_fraction": [0.0],
        }
    )
    args = argparse.Namespace(outer_folds=2, embargo_hours=24, skip_leave_one=False, skip_oracle_specialist=False)

    audit = mod._requirement_audit(
        summary,
        folds,
        cells,
        context,
        leave_one,
        args,
        gradient_conflict=gradient,
        episode_ci=episode_ci,
        oracle_specialist=oracle,
        promotion=promotion,
        bad_episode_context=bad_context,
        directional=directional,
        directional_timestamp=directional_timestamp,
        directional_episode=directional_episode,
        directional_episode_ci=directional_episode_ci,
    )

    assert audit["status"] == "passed"
    assert all(item["status"] == "passed" for item in audit["items"])


def test_directional_metrics_are_timestamp_local_and_use_fixed_eligible_set() -> None:
    panel = pd.DataFrame(
        {
            "timestamp": [
                "2026-05-25T00:00:00Z",
                "2026-05-25T00:00:00Z",
                "2026-05-25T00:00:00Z",
                "2026-05-25T00:00:00Z",
                "2026-05-25T01:00:00Z",
                "2026-05-25T01:00:00Z",
                "2026-05-25T01:00:00Z",
                "2026-05-25T01:00:00Z",
            ],
            "oof_rank_pct": [0.8] * 8,
        }
    )
    y = np.array([1, 0, 0, 0, 0, 1, 1, 1], dtype=np.int8)
    pred = np.array([0.9, 0.8, 0.7, 0.6, 0.9, 0.8, 0.7, 0.6], dtype=np.float32)
    baseline = np.array([0.6, 0.7, 0.8, 0.9, 0.6, 0.7, 0.8, 0.9], dtype=np.float32)
    returns = np.zeros(8, dtype=np.float32)

    ts_metrics = mod._directional_timestamp_metrics(
        head="short_asset",
        arm=mod.ARM_D,
        variant="hard_label_context_arm",
        panel=panel,
        y=y,
        pred=pred,
        baseline_pred=baseline,
        returns=returns,
        bad_episodes={"2026-05-25"},
        rank_threshold=0.70,
        min_timestamp_rows=4,
    )
    agg = mod._directional_aggregate(ts_metrics)

    assert len(ts_metrics) == 2
    assert ts_metrics["selected_count_top30"].tolist() == [2, 2]
    # Timestamp 00: model top2 has one hit, baseline top2 has zero; timestamp 01 is the reverse.
    assert ts_metrics["delta_hr_top30"].round(6).tolist() == [0.5, -0.5]
    assert agg.iloc[0]["timestamp_weighted_hr_top30"] == pytest.approx(0.5)
    assert agg.iloc[0]["delta_timestamp_weighted_hr_top30"] == pytest.approx(0.0)
    assert agg.iloc[0]["top30_entrant_count"] == 4


def test_gradient_conflict_reports_required_leaf_region_fields() -> None:
    panel = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-04-13", periods=80, freq="h", tz="UTC"),
            "oof_rank_pct": np.linspace(0.0, 1.0, 80),
        }
    )
    y = np.array([0, 1] * 40, dtype=np.int8)
    pred = np.linspace(0.05, 0.95, 80).astype(np.float32)

    out = mod._gradient_conflict_diagnostics(
        head="short_asset",
        panel=panel,
        y=y,
        predictions={mod.ARM_E: pred},
        bad_episodes={"2026-04-13"},
        n_score_bins=2,
    )

    assert {"leaf", "regime", "support", "gradient_sum", "hessian_sum", "optimal_update", "update_sign", "cancellation_score"} <= set(out.columns)
    assert not out.empty


def test_episode_block_confidence_intervals_are_episode_level() -> None:
    leave_one = pd.DataFrame(
        {
            "head": ["short_asset", "short_asset", "short_asset"],
            "arm": [mod.ARM_E, mod.ARM_E, mod.ARM_E],
            "distillation_variant": ["hard_label_context_arm"] * 3,
            "heldout_episode": ["2026-04-13", "2026-04-20", "2026-05-04"],
            "delta_log_loss_improvement": [0.01, -0.02, 0.03],
            "delta_brier_improvement": [0.005, -0.001, 0.004],
            "top10_delta_mean_return": [0.001, -0.002, 0.003],
        }
    )

    out = mod._episode_block_confidence_intervals(leave_one, seed=1, bootstrap_rounds=20)

    assert "delta_log_loss_improvement" in set(out["metric"])
    row = out.loc[out["metric"].eq("delta_log_loss_improvement")].iloc[0]
    assert row["episode_count"] == 3


def test_episode_registry_loader_filters_by_head(tmp_path) -> None:
    registry = tmp_path / "registry.csv"
    pd.DataFrame(
        {
            "episode_id": ["2026-04-13", "2026-04-20", "2026-05-25"],
            "definition": ["diagnostic"] * 3,
            "target": ["diagnostic"] * 3,
            "start": ["2026-04-13T00:00:00Z", "2026-04-20T00:00:00Z", "2026-05-25T00:00:00Z"],
            "end": ["2026-04-20T00:00:00Z", "2026-04-27T00:00:00Z", "2026-06-01T00:00:00Z"],
            "severity": [1.0, 1.0, 1.0],
            "eligible_heads": ["short_asset", "long_dist", "all"],
            "reason_for_inclusion": ["unit"] * 3,
            "reason_for_exclusion": ["", "", ""],
        }
    ).to_csv(registry, index=False)

    episodes, meta = mod._load_episode_registry(registry, head="short_asset", target_name="y_bin")

    assert episodes == {"2026-04-13", "2026-05-25"}
    assert meta["reason"] == "frozen_episode_registry"
