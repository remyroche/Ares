from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from scripts import run_canonical_context_retrain_experiment as mod


EXPECTED_ARMS = [
    "baseline_current_meta_unchanged",
    "canonical_model_state_context",
    "canonical_market_state_context",
    "model_state_x_market_state_interactions",
    "auxiliary_failure_head",
]


def test_canonical_definitions_and_frame_keep_stable_order(tmp_path: Path) -> None:
    reduction = tmp_path / "canonical.csv"
    pd.DataFrame(
        [
            {
                "canonical_variable": "prediction_support_quality",
                "top_parent_features": "support_gap",
                "mechanism_channel": "model",
                "state_family": "model_state",
            },
            {
                "canonical_variable": "leverage_funding_crowding",
                "top_parent_features": "funding_z",
                "mechanism_channel": "market",
                "state_family": "market_state",
            },
            {
                "canonical_variable": "trend_range_breakout",
                "top_parent_features": "trend_t",
                "mechanism_channel": "market",
                "state_family": "market_state",
            },
        ]
    ).to_csv(reduction, index=False)

    definitions = mod._load_canonical_definitions(reduction)
    assert set(definitions) == {"prediction_support_quality", "leverage_funding_crowding"}

    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=8, freq="h", tz="UTC"),
            "symbol": ["BTC"] * 8,
            "support_gap": np.arange(8, dtype=np.float32),
            "funding_z": np.linspace(-1.0, 1.0, 8, dtype=np.float32),
        }
    )
    canonical, diagnostics = mod._build_canonical_frame(
        frame,
        definitions,
        trailing_window=3,
        min_periods=1,
        min_resolved_features=1,
    )

    assert list(canonical.columns) == list(mod.CANONICAL_CONTEXT)
    assert diagnostics["status"] == "completed"
    assert canonical["prediction_support_quality"].notna().sum() > 0
    assert canonical["leverage_funding_crowding"].notna().sum() > 0


def test_chrono_folds_apply_embargo_before_validation_window() -> None:
    timestamps = pd.Series(pd.date_range("2026-01-01", periods=48, freq="h", tz="UTC"))
    folds = mod._make_chrono_folds(timestamps, 3, embargo_hours=4)

    assert len(folds) == 3
    ts = pd.to_datetime(timestamps, utc=True)
    for fold in folds:
        valid_start = ts.iloc[fold.valid_idx].min()
        train_end = ts.iloc[fold.train_idx].max()
        assert train_end < valid_start - pd.Timedelta(hours=4)


def test_fresh_oos_indices_reserve_cutoff_and_embargo_gap() -> None:
    timestamps = pd.Series(pd.date_range("2026-01-01", periods=72, freq="h", tz="UTC"))
    fresh_start = pd.Timestamp("2026-01-03 00:00:00", tz="UTC")
    split = mod._fresh_oos_indices(timestamps, fresh_start, embargo_hours=6)

    ts = pd.to_datetime(timestamps, utc=True)
    assert len(split["train_idx"]) > 0
    assert len(split["test_idx"]) > 0
    assert ts.iloc[split["train_idx"]].max() < fresh_start - pd.Timedelta(hours=6)
    assert ts.iloc[split["test_idx"]].min() >= fresh_start


def test_go_no_go_requires_leave_one_and_fresh_oos() -> None:
    passing = pd.Series(
        {
            "arm": "canonical_model_state_context",
            "delta_log_loss_improvement": 0.01,
            "delta_pr_auc": 0.02,
            "median_bad_episode_logloss_improvement": 0.01,
            "episodes_improved_logloss": 2,
            "bad_episode_count": 3,
            "median_leave_one_logloss_improvement": 0.01,
            "leave_one_episodes_improved_logloss": 2,
            "leave_one_episode_count": 3,
            "delta_tail_loss_10pct": 0.0,
            "delta_winner_rejection_cost_10pct": 0.0,
            "normal_episode_median_logloss_improvement": 0.0,
        }
    )
    decision, reason = mod._go_no_go(passing, fresh_oos_evaluated=False)
    assert decision == "research_candidate_pending_fresh_oos"
    assert "fresh OOS" in reason

    decision, _reason = mod._go_no_go(passing, fresh_oos_evaluated=True)
    assert decision == "reject"

    passing_with_fresh = passing.copy()
    passing_with_fresh["fresh_oos_evaluated"] = True
    passing_with_fresh["fresh_oos_delta_log_loss_improvement"] = 0.01
    passing_with_fresh["fresh_oos_delta_pr_auc"] = 0.0
    passing_with_fresh["fresh_oos_delta_tail_loss_10pct"] = 0.0
    passing_with_fresh["fresh_oos_delta_winner_rejection_cost_10pct"] = 0.0
    decision, _reason = mod._go_no_go(passing_with_fresh, fresh_oos_evaluated=True)
    assert decision == "candidate"

    no_leave_one = passing.copy()
    no_leave_one["leave_one_episode_count"] = 0
    decision, reason = mod._go_no_go(no_leave_one, fresh_oos_evaluated=True)
    assert decision == "reject"
    assert "leave-one" in reason


def test_requirement_audit_blocks_without_fresh_oos() -> None:
    rows = []
    fold_rows = []
    loo_rows = []
    ctx_rows = []
    for head, targets in mod.CANDIDATES.items():
        for fold in (1, 2):
            ctx_rows.append(
                {
                    "head": head,
                    "fold": fold,
                    "train_rows": 1000,
                    "valid_rows": 500,
                    "valid_output_feature_count": len(mod.CANONICAL_CONTEXT),
                }
            )
        for target in targets:
            for arm in EXPECTED_ARMS:
                rows.append(
                    {
                        "head": head,
                        "target": target,
                        "arm": arm,
                        "recommendation": "baseline" if arm == "baseline_current_meta_unchanged" else "reject",
                        "roc_auc": 0.55,
                        "pr_auc": 0.20,
                        "log_loss": 0.30,
                        "brier": 0.10,
                        "calibration_slope": 1.0,
                        "calibration_intercept": 0.0,
                        "top_reliable_hit_rate_10pct": 0.7,
                        "top_reliable_net_return_mean_10pct": 0.01,
                        "top_reliable_tail_loss_10pct": -0.02,
                        "delta_tail_loss_10pct": 0.0,
                        "delta_winner_rejection_cost_10pct": 0.0,
                        "weekly_auc_std": 0.1,
                        "scored_coverage": 0.8,
                        "weekly_rejection_turnover_10pct": 0.3,
                        "normal_episode_median_logloss_improvement": 0.0,
                        "median_bad_episode_logloss_improvement": 0.0,
                        "worst_bad_episode_logloss_improvement": 0.0,
                        "episodes_improved_logloss": 1,
                        "fold_fitted": True,
                        "causal_trailing": True,
                        "live_equivalent": True,
                        "raw_alias_outputs_used": False,
                        "bad_contract_feature_count": 0,
                        "fresh_oos_evaluated": False,
                    }
                )
                for fold in (1, 2):
                    fold_rows.append({"head": head, "target": target, "arm": arm, "fold": fold})
                loo_rows.append({"head": head, "target": target, "arm": arm, "heldout_episode": "2026-01-01"})

    args = type(
        "Args",
        (),
        {"only_head": [], "outer_folds": 2, "embargo_hours": 24, "fresh_oos_start": "", "assume_oof_final": False},
    )()
    audit = mod._build_requirement_audit(
        summary=pd.DataFrame(rows),
        fold_metrics=pd.DataFrame(fold_rows),
        leave_one=pd.DataFrame(loo_rows),
        fresh_oos=pd.DataFrame(
            [{"status": "not_evaluated", "fresh_oos_start": "", "reason": "no untouched later period"}]
        ),
        context_diagnostics=pd.DataFrame(ctx_rows),
        args=args,
    )

    assert audit["status"] == "blocked"
    assert audit["failed_requirements"] == []
    assert audit["blocked_requirements"] == ["fresh_chronological_oos_confirmation"]
    assert audit["outcomes"]["summary_rows"] == len(rows)
    assert audit["outcomes"]["candidate_rows"] == 0
    assert all("metrics" in item for item in audit["items"])

    args.assume_oof_final = True
    audit = mod._build_requirement_audit(
        summary=pd.DataFrame(rows),
        fold_metrics=pd.DataFrame(fold_rows),
        leave_one=pd.DataFrame(loo_rows),
        fresh_oos=pd.DataFrame(
            [{"status": "not_evaluated", "fresh_oos_start": "", "reason": "no untouched later period"}]
        ),
        context_diagnostics=pd.DataFrame(ctx_rows),
        args=args,
    )

    assert audit["status"] == "passed_with_waiver"
    assert audit["blocked_requirements"] == []
    assert audit["failed_requirements"] == []
    assert audit["waived_requirements"] == ["fresh_chronological_oos_confirmation"]
