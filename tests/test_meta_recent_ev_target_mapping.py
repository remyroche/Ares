from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.ablate_meta_recent_ev_target_mapping import (
    Arm,
    _recent_weights,
    _safe_multiplier,
    _threshold_for_target_ev,
)
from extreme_price_movements.inference.threshold_basis_policy import (
    apply_threshold_basis_policy_to_decisions,
    load_threshold_basis_policy,
)
from extreme_price_movements.inference.prediction_ledger import (
    PREDICTION_LEDGER_DIAGNOSTIC_COLUMNS,
)


def test_target_threshold_prefers_a_tail_that_reaches_the_ev_target() -> None:
    score = np.linspace(0.0, 1.0, 200)
    ev = np.where(score >= 0.8, 0.02, -0.01)
    threshold = _threshold_for_target_ev(
        score,
        ev,
        np.ones(len(score)),
        target_ev=0.015,
        min_rows=20,
    )

    assert threshold >= 0.75
    assert ev[score >= threshold].mean() >= 0.015


def test_multiplier_is_shrunk_when_local_support_is_small() -> None:
    low_support = _safe_multiplier(0.90, 0.60, support=0.0)
    high_support = _safe_multiplier(0.90, 0.60, support=160.0)

    assert low_support == 1.0
    assert high_support == 1.5


def test_ewma_recent_weights_favor_newer_observations() -> None:
    arm = Arm("test", "after_mlp", 8, "ewma", 5)
    weights = _recent_weights(
        pd.to_datetime(["2026-04-01", "2026-04-06"], utc=True).to_numpy(),
        pd.Timestamp("2026-04-06", tz="UTC"),
        arm,
    )

    assert weights[1] > weights[0]
    np.testing.assert_allclose(weights[0], 0.5)


def test_pre_mlp_policy_uses_parent_reference_and_final_mlp_rank(tmp_path) -> None:
    reference = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-06-01", periods=80, freq="6h", tz="UTC"),
            "side_name": ["short"] * 80,
            "policy_archetype": ["short_default_clean_path"] * 80,
            "policy_parent_rank": np.linspace(0.10, 0.99, 80),
            "rank_mlp_direct": np.linspace(0.10, 0.99, 80),
            "ev_after_1pct": np.where(np.arange(80) >= 64, 0.02, -0.01),
        }
    )
    reference_path = tmp_path / "reference.parquet"
    reference.to_parquet(reference_path, index=False)
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(
        __import__("json").dumps(
            {
                "enabled": True,
                "policy_id": "test_pre_mlp",
                "family": "ev_target_side_archetype_multiplier_before_mlp",
                "window_days": 28,
                "min_reference_rows": 40,
                "local_support_target": 40,
                "multiplier_min": 0.5,
                "multiplier_max": 1.5,
                "top_fraction": 0.10,
                "calibration_reference_score_col": "policy_parent_rank",
                "apply_reference_score_col": "rank_mlp_direct",
                "live_score_col": "expected_ev_rank_score",
                "return_col": "ev_after_1pct",
                "reference_candidates_path": str(reference_path),
            }
        )
    )
    decisions = [
        {
            "timestamp": "2026-06-22T00:00:00Z",
            "side_name": "short",
            "policy_archetype": "short_default_clean_path",
            "expected_ev_rank_score": 0.99,
        },
        {
            "timestamp": "2026-06-22T00:00:00Z",
            "side_name": "short",
            "policy_archetype": "short_default_clean_path",
            "expected_ev_rank_score": 0.20,
        },
    ]
    apply_threshold_basis_policy_to_decisions(
        decisions, policy=load_threshold_basis_policy(policy_path)
    )

    assert decisions[0]["threshold_basis_selected"] is True
    assert decisions[1]["threshold_basis_selected"] is False
    assert decisions[0]["threshold_basis_policy_id"] == "test_pre_mlp"
    assert decisions[0]["threshold_basis_ev_target_local_support"] >= 40
    assert decisions[1]["threshold_basis_ev_target_local_support"] >= 40
    assert (
        decisions[1]["threshold_basis_ev_target_global_fallback"]
        == decisions[0]["threshold_basis_ev_target_global_fallback"]
    )
    assert (
        decisions[1]["threshold_basis_ev_target_multiplier"]
        == decisions[0]["threshold_basis_ev_target_multiplier"]
    )


def test_pre_mlp_policy_matches_live_side_prefixed_archetype_to_reference(
    tmp_path,
) -> None:
    reference = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-06-01", periods=80, freq="6h", tz="UTC"),
            "side_name": ["long"] * 80,
            "policy_archetype": ["long_breakout_diagnostic_candidate"] * 80,
            "policy_parent_rank": np.linspace(0.10, 0.99, 80),
            "rank_mlp_direct": np.linspace(0.10, 0.99, 80),
            "ev_after_1pct": np.where(np.arange(80) >= 64, 0.02, -0.01),
        }
    )
    reference_path = tmp_path / "reference.parquet"
    reference.to_parquet(reference_path, index=False)
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(
        __import__("json").dumps(
            {
                "enabled": True,
                "policy_id": "test_prefixed_archetype",
                "family": "ev_target_side_archetype_multiplier_before_mlp",
                "window_days": 28,
                "min_reference_rows": 40,
                "local_support_target": 40,
                "multiplier_min": 0.5,
                "multiplier_max": 1.5,
                "top_fraction": 0.10,
                "calibration_reference_score_col": "policy_parent_rank",
                "apply_reference_score_col": "rank_mlp_direct",
                "live_score_col": "expected_ev_rank_score",
                "return_col": "ev_after_1pct",
                "reference_candidates_path": str(reference_path),
            }
        )
    )
    decisions = [
        {
            "timestamp": "2026-06-22T00:00:00Z",
            "side_name": "long",
            "policy_archetype": "long__long_breakout_diagnostic_candidate",
            "expected_ev_rank_score": 0.99,
        }
    ]

    apply_threshold_basis_policy_to_decisions(
        decisions, policy=load_threshold_basis_policy(policy_path)
    )

    assert decisions[0]["threshold_basis_selected"] is True
    assert decisions[0]["threshold_basis_ev_target_local_support"] >= 40


def test_expected_ev_policy_corrects_side_archetype_ev_in_ev_units(tmp_path) -> None:
    timestamps = pd.date_range("2026-06-01", periods=160, freq="3h", tz="UTC")
    archetypes = np.where(np.arange(160) % 2 == 0, "clean_state", "dirty_state")
    reference = pd.DataFrame(
        {
            "timestamp": timestamps,
            "side_name": ["long"] * 160,
            "policy_archetype": archetypes,
            "mapped_expected_ev": np.full(160, 0.01),
            "ev_after_1pct": np.where(archetypes == "clean_state", 0.03, -0.01),
        }
    )
    reference_path = tmp_path / "reference.parquet"
    reference.to_parquet(reference_path, index=False)
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(
        __import__("json").dumps(
            {
                "enabled": True,
                "policy_id": "test_ev_unit_correction",
                "family": "side_archetype_expected_ev_recent_correction",
                "window_days": 28,
                "min_reference_rows": 20,
                "side_support_target": 20,
                "local_support_target": 20,
                "recent_ev_correction_cap": 0.05,
                "top_fraction": 0.10,
                "mapped_expected_ev_col": "expected_net_ev_after_1pct_side_archetype",
                "reference_mapped_expected_ev_col": "mapped_expected_ev",
                "return_col": "ev_after_1pct",
                "reference_candidates_path": str(reference_path),
            }
        )
    )
    decisions = [
        {
            "timestamp": "2026-06-22T00:00:00Z",
            "side_name": "long",
            "policy_archetype": "long__clean_state",
            "expected_net_ev_after_1pct_side_archetype": 0.01,
        },
        {
            "timestamp": "2026-06-22T00:00:00Z",
            "side_name": "long",
            "policy_archetype": "long__dirty_state",
            "expected_net_ev_after_1pct_side_archetype": 0.01,
        },
    ]

    apply_threshold_basis_policy_to_decisions(
        decisions, policy=load_threshold_basis_policy(policy_path)
    )

    assert decisions[0]["threshold_basis_mapped_expected_ev_side_archetype"] == 0.01
    assert decisions[1]["threshold_basis_mapped_expected_ev_side_archetype"] == 0.01
    assert decisions[0]["threshold_basis_side_archetype_recent_ev_correction"] > 0.0
    assert decisions[1]["threshold_basis_side_archetype_recent_ev_correction"] < 0.0
    assert (
        decisions[0]["threshold_basis_corrected_expected_ev"]
        > decisions[1]["threshold_basis_corrected_expected_ev"]
    )
    assert decisions[0]["threshold_basis_selected"] is True
    assert decisions[1]["threshold_basis_selected"] is False


def test_expected_ev_policy_audit_fields_are_persisted_in_prediction_ledger() -> None:
    required = {
        "threshold_basis_mapped_expected_ev_side_archetype",
        "threshold_basis_side_archetype_recent_ev_correction",
        "threshold_basis_corrected_expected_ev",
        "threshold_basis_corrected_expected_ev_rank",
        "threshold_basis_parent_rank",
        "threshold_basis_blended_rank",
        "threshold_basis_ev_rank_blend_weight",
        "threshold_basis_expected_ev_correction_scope",
    }

    assert required.issubset(PREDICTION_LEDGER_DIAGNOSTIC_COLUMNS)


def test_expected_ev_policy_blends_parent_and_per_archetype_ev_ranks(tmp_path) -> None:
    timestamps = pd.date_range("2026-06-01", periods=100, freq="3h", tz="UTC")
    reference = pd.DataFrame(
        {
            "timestamp": timestamps,
            "side_name": ["long"] * 100,
            "policy_archetype": ["state"] * 100,
            "mapped_expected_ev": np.linspace(-0.02, 0.03, 100),
            "ev_after_1pct": np.linspace(-0.02, 0.03, 100),
        }
    )
    reference_path = tmp_path / "reference.parquet"
    reference.to_parquet(reference_path, index=False)
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(
        __import__("json").dumps(
            {
                "enabled": True,
                "policy_id": "test_ev_rank_blend",
                "family": "side_archetype_expected_ev_recent_correction",
                "window_days": 28,
                "min_reference_rows": 20,
                "side_support_target": 20,
                "local_support_target": 20,
                "recent_ev_correction_cap": 0.05,
                "top_fraction": 0.10,
                "ev_rank_blend_weight": 0.30,
                "rank_blend_parent_col": "v9_tail95_predecessor_rank",
                "mapped_expected_ev_col": "expected_net_ev_after_1pct_side_archetype",
                "reference_mapped_expected_ev_col": "mapped_expected_ev",
                "return_col": "ev_after_1pct",
                "reference_candidates_path": str(reference_path),
            }
        )
    )
    decisions = [
        {
            "timestamp": "2026-06-14T00:00:00Z",
            "side_name": "long",
            "policy_archetype": "long__state",
            "expected_net_ev_after_1pct_side_archetype": 0.03,
            "v9_tail95_predecessor_rank": 0.99,
        }
    ]

    apply_threshold_basis_policy_to_decisions(
        decisions, policy=load_threshold_basis_policy(policy_path)
    )

    row = decisions[0]
    expected = 0.70 * row["threshold_basis_parent_rank"] + 0.30 * row[
        "threshold_basis_corrected_expected_ev_rank"
    ]
    np.testing.assert_allclose(row["threshold_basis_blended_rank"], expected)
    np.testing.assert_allclose(row["threshold_basis_dynamic_score_threshold"], 0.90)


def test_expected_ev_policy_can_use_parent_only_to_break_ev_ties(tmp_path) -> None:
    timestamps = pd.date_range("2026-06-01", periods=100, freq="3h", tz="UTC")
    reference = pd.DataFrame(
        {
            "timestamp": timestamps,
            "side_name": ["long"] * 100,
            "policy_archetype": ["state"] * 100,
            "mapped_expected_ev": [0.01] * 100,
            "ev_after_1pct": [0.01] * 100,
            "rank_mlp_direct": np.linspace(0.01, 1.0, 100),
        }
    )
    reference_path = tmp_path / "reference.parquet"
    reference.to_parquet(reference_path, index=False)
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(
        __import__("json").dumps(
            {
                "enabled": True,
                "policy_id": "test_ev_rank_tie_break",
                "family": "side_archetype_expected_ev_recent_correction",
                "window_days": 28,
                "min_reference_rows": 20,
                "side_support_target": 20,
                "local_support_target": 20,
                "top_fraction": 0.10,
                "ev_rank_blend_weight": 1.0,
                "corrected_ev_tie_break_parent": True,
                "reference_parent_rank_col": "rank_mlp_direct",
                "rank_blend_parent_col": "v9_tail95_predecessor_rank",
                "mapped_expected_ev_col": "expected_net_ev_after_1pct_side_archetype",
                "reference_mapped_expected_ev_col": "mapped_expected_ev",
                "return_col": "ev_after_1pct",
                "reference_columns": list(reference.columns),
                "reference_candidates_path": str(reference_path),
            }
        )
    )
    decisions = [
        {
            "timestamp": "2026-06-14T00:00:00Z",
            "side_name": "long",
            "policy_archetype": "long__state",
            "expected_net_ev_after_1pct_side_archetype": 0.01,
            "v9_tail95_predecessor_rank": 0.95,
        },
        {
            "timestamp": "2026-06-14T00:00:00Z",
            "side_name": "long",
            "policy_archetype": "long__state",
            "expected_net_ev_after_1pct_side_archetype": 0.01,
            "v9_tail95_predecessor_rank": 0.50,
        },
    ]

    apply_threshold_basis_policy_to_decisions(
        decisions, policy=load_threshold_basis_policy(policy_path)
    )

    assert decisions[0]["threshold_basis_corrected_expected_ev_rank"] > 0.90
    assert decisions[0]["threshold_basis_selected"] is True
    assert decisions[1]["threshold_basis_corrected_expected_ev_rank"] < 0.90
    assert decisions[1]["threshold_basis_selected"] is False


def test_expected_ev_policy_supports_fixed_net_ev_target_without_topk_quota(
    tmp_path,
) -> None:
    timestamps = pd.date_range("2026-06-01", periods=120, freq="3h", tz="UTC")
    reference = pd.DataFrame(
        {
            "timestamp": timestamps,
            "side_name": ["long"] * 120,
            "policy_archetype": ["state"] * 120,
            "mapped_expected_ev": np.linspace(-0.01, 0.02, 120),
            "ev_after_1pct": np.linspace(-0.01, 0.02, 120),
        }
    )
    reference_path = tmp_path / "reference.parquet"
    reference.to_parquet(reference_path, index=False)
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(
        __import__("json").dumps(
            {
                "enabled": True,
                "policy_id": "test_fixed_ev_target",
                "family": "side_archetype_expected_ev_recent_correction",
                "selection_mode": "fixed_corrected_ev_threshold",
                "fixed_target_net_ev": 0.007,
                "window_days": 28,
                "recalibration_frequency": "1d_at_00_utc",
                "min_reference_rows": 20,
                "side_support_target": 20,
                "local_support_target": 20,
                "recent_ev_correction_cap": 0.05,
                "top_fraction": 0.10,
                "mapped_expected_ev_col": (
                    "expected_net_ev_after_1pct_side_archetype"
                ),
                "reference_mapped_expected_ev_col": "mapped_expected_ev",
                "return_col": "ev_after_1pct",
                "reference_candidates_path": str(reference_path),
            }
        )
    )
    decisions = [
        {
            "timestamp": "2026-06-16T00:00:00Z",
            "side_name": "long",
            "policy_archetype": "long__state",
            "expected_net_ev_after_1pct_side_archetype": 0.008,
        },
        {
            "timestamp": "2026-06-16T00:00:00Z",
            "side_name": "long",
            "policy_archetype": "long__state",
            "expected_net_ev_after_1pct_side_archetype": 0.006,
        },
    ]

    apply_threshold_basis_policy_to_decisions(
        decisions, policy=load_threshold_basis_policy(policy_path)
    )

    assert decisions[0]["threshold_basis_selected"] is True
    assert decisions[1]["threshold_basis_selected"] is False
    assert decisions[0]["threshold_basis_rank_score"] >= 0.90
    assert decisions[0]["threshold_basis_dynamic_ev_target"] == 0.007
    assert (
        decisions[0]["threshold_basis_recalibration_frequency"]
        == "1d_at_00_utc"
    )
    assert decisions[0]["threshold_basis_reference_asof"].startswith(
        "2026-06-16T00:00:00"
    )


def test_expected_ev_policy_daily_trim_matches_portfolio_ablation(tmp_path) -> None:
    from scripts.ablate_side_archetype_ev_portfolio_matrix import _trimmed_correction

    outcome_days = pd.date_range("2026-06-01", periods=21, freq="D", tz="UTC")
    daily_residuals = np.asarray(
        [-0.08, *np.linspace(-0.004, 0.006, 19), 0.10], dtype=np.float64
    )
    reference = pd.DataFrame(
        {
            "timestamp": outcome_days - pd.Timedelta(hours=12),
            "outcome_resolved_at": outcome_days,
            "side_name": "long",
            "policy_archetype": "state",
            "mapped_expected_ev": 0.01,
            "ev_after_1pct": 0.01 + daily_residuals,
        }
    )
    reference_path = tmp_path / "reference.parquet"
    reference.to_parquet(reference_path, index=False)
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(
        __import__("json").dumps(
            {
                "enabled": True,
                "policy_id": "test_robust_fixed_ev_target",
                "family": "side_archetype_expected_ev_recent_correction",
                "selection_mode": "fixed_corrected_ev_threshold",
                "fixed_target_net_ev": 0.007,
                "window_days": 21,
                "recalibration_frequency": "1d_at_00_utc",
                "robust_daily_residual_trim_fraction": 0.10,
                "robust_daily_residual_normalization": "median_iqr",
                "min_reference_rows": 5,
                "side_support_target": 1,
                "local_support_target": 1,
                "recent_ev_correction_cap": 0.20,
                "mapped_expected_ev_col": (
                    "expected_net_ev_after_1pct_side_archetype"
                ),
                "reference_mapped_expected_ev_col": "mapped_expected_ev",
                "return_col": "ev_after_1pct",
                "reference_candidates_path": str(reference_path),
            }
        )
    )
    decisions = [
        {
            "timestamp": "2026-06-22T00:00:00Z",
            "side_name": "long",
            "policy_archetype": "long__state",
            "expected_net_ev_after_1pct_side_archetype": 0.01,
        }
    ]

    apply_threshold_basis_policy_to_decisions(
        decisions, policy=load_threshold_basis_policy(policy_path)
    )

    stats = pd.DataFrame(
        {
            "sum": daily_residuals,
            "count": np.ones(len(daily_residuals)),
            "mean": daily_residuals,
        }
    )
    expected, support, retained_days, _ = _trimmed_correction(stats, 0.10)
    assert decisions[0]["threshold_basis_side_archetype_recent_ev_correction"] == pytest.approx(
        expected
    )
    assert decisions[0]["threshold_basis_ev_target_local_support"] == support
    assert decisions[0]["threshold_basis_global_days_retained"] == retained_days
    assert decisions[0]["threshold_basis_robust_daily_residual_trim_fraction"] == 0.10
