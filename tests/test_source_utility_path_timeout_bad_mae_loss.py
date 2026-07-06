from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_source_utility_path_timeout_risk_diagnostic import (
    RiskHeadSpec,
    _add_utility_only_deltas,
    _build_risk_head_target,
    _path_summary,
    _selection_specs_by_name,
)


def test_bad_mae_loss_target_ignores_recovered_bad_mae() -> None:
    metrics = pd.DataFrame(
        {
            "u_policy_net": [0.020, -0.012, 0.035, -0.004],
            "mae_norm": [1.20, 1.10, 2.00, 0.80],
            "barrier": [0.010, 0.010, 0.040, 0.010],
            "is_timeout": [False, False, False, False],
            "bars_policy": [2.0, 3.0, 8.0, 2.0],
        }
    )

    target, weights, report = _build_risk_head_target(
        metrics=metrics,
        train_mask=pd.Series([True, True, True, False]),
        spec=RiskHeadSpec("bad_mae_loss_risk_v1", "bad_mae_loss", "test"),
    )

    assert target["target_hard"].tolist() == [0.0, 1.0, 0.0, 0.0]
    assert target.loc[1, "target_soft"] > target.loc[0, "target_soft"]
    assert float(weights.min()) > 0.0
    assert report["risk_head"] == "bad_mae_loss_risk_v1"


def test_fast_bad_mae_loss_target_requires_fast_negative_bad_mae() -> None:
    metrics = pd.DataFrame(
        {
            "u_policy_net": [-0.012, -0.012, 0.020, -0.020],
            "mae_norm": [1.20, 1.20, 1.20, 0.70],
            "barrier": [0.010, 0.010, 0.010, 0.010],
            "is_timeout": [False, False, False, False],
            "bars_policy": [3.0, 12.0, 3.0, 3.0],
        }
    )

    target, weights, report = _build_risk_head_target(
        metrics=metrics,
        train_mask=pd.Series([True, True, True, False]),
        spec=RiskHeadSpec("fast_bad_mae_loss_risk_v1", "fast_bad_mae_loss", "test"),
    )

    assert target["target_hard"].tolist() == [1.0, 0.0, 0.0, 0.0]
    assert target.loc[0, "target_soft"] > target.loc[1, "target_soft"]
    assert float(weights.min()) > 0.0
    assert report["risk_head"] == "fast_bad_mae_loss_risk_v1"


def test_bad_mae_recovery_failure_target_emphasizes_negative_vs_recovered_bad_mae() -> None:
    metrics = pd.DataFrame(
        {
            "u_policy_net": [-0.012, 0.020, -0.003, 0.015, -0.020],
            "mae_norm": [1.20, 1.30, 0.80, 0.40, 1.50],
            "barrier": [0.010, 0.010, 0.010, 0.010, 0.010],
            "is_timeout": [False, False, False, False, False],
            "bars_policy": [3.0, 8.0, 3.0, 2.0, 12.0],
        }
    )

    target, weights, report = _build_risk_head_target(
        metrics=metrics,
        train_mask=pd.Series([True, True, True, True, False]),
        spec=RiskHeadSpec(
            "bad_mae_recovery_failure_risk_v1",
            "bad_mae_recovery_failure",
            "test",
        ),
    )

    assert target["target_hard"].tolist() == [1.0, 0.0, 0.0, 0.0, 1.0]
    assert target.loc[0, "target_soft"] > target.loc[1, "target_soft"]
    assert target.loc[1, "target_soft"] == 0.0
    assert weights.loc[0] > weights.loc[3]
    assert weights.loc[1] > weights.loc[3]
    assert report["risk_head"] == "bad_mae_recovery_failure_risk_v1"
    assert report["contrastive_bad_mae_failure_rows"] == 2
    assert report["contrastive_bad_mae_recovered_rows"] == 1


def test_bad_mae_loss_selectors_register_required_heads() -> None:
    specs = _selection_specs_by_name(
        [
            "utility_minus_bad_mae_loss_timeout_0p50",
            "stage1_fast_bad_mae_loss_q40_then_timeout_0p50",
            "utility_minus_recovery_failure_timeout_0p50",
            "stage1_recovery_failure_q40_then_timeout_0p50",
        ]
    )

    assert specs[0].required_heads == ("timeout_risk_v1", "bad_mae_loss_risk_v1")
    assert specs[1].required_heads == ("timeout_risk_v1", "fast_bad_mae_loss_risk_v1")
    assert specs[2].required_heads == ("timeout_risk_v1", "bad_mae_recovery_failure_risk_v1")
    assert specs[3].required_heads == ("timeout_risk_v1", "bad_mae_recovery_failure_risk_v1")


def test_path_summary_splits_raw_bad_mae_into_negative_and_recovered() -> None:
    metrics = pd.DataFrame(
        {
            "u_policy_net": [-0.012, 0.020, -0.030, 0.015, 0.010],
            "mae_norm": [1.20, 1.30, 1.40, 1.10, 0.40],
            "barrier": [0.010, 0.010, 0.030, 0.010, 0.010],
            "is_timeout": [False, False, True, False, False],
            "bars_policy": [3.0, 2.0, 10.0, 12.0, 2.0],
        }
    )

    summary = _path_summary(metrics)

    assert summary["bad_mae_1r_rate"] == 0.8
    assert summary["bad_mae_negative_rate"] == 0.4
    assert summary["bad_mae_recovered_rate"] == 0.4
    assert summary["fast_bad_mae_negative_rate"] == 0.2
    assert summary["fast_bad_mae_recovered_rate"] == 0.2
    assert summary["late_bad_mae_negative_rate"] == 0.2
    assert summary["late_bad_mae_recovered_rate"] == 0.2
    assert summary["bad_mae_negative_share_of_bad_mae"] == 0.5
    assert summary["bad_mae_recovered_share_of_bad_mae"] == 0.5
    assert summary["bad_mae_negative_mean_u"] == pytest.approx(-0.021)
    assert summary["bad_mae_recovered_mean_u"] == pytest.approx(0.0175)


def test_utility_only_deltas_include_bad_mae_recovery_splits() -> None:
    monthly = pd.DataFrame(
        {
            "period": ["2026-04", "2026-04"],
            "label": ["utility", "utility"],
            "risk_heads": ["heads", "heads"],
            "feature_set": ["base", "base"],
            "source_bucket": ["all_rows", "all_rows"],
            "causal_gate": ["no_gate", "no_gate"],
            "top_frac": [0.1, 0.1],
            "selection": ["utility_only", "utility_minus_bad_mae_loss_timeout_0p50"],
            "selected_rows": [100, 100],
            "mean_u": [0.010, 0.011],
            "hit_u": [0.55, 0.58],
            "q10_u": [-0.020, -0.018],
            "bad_mae_1r_rate": [0.70, 0.65],
            "bad_mae_negative_rate": [0.35, 0.22],
            "bad_mae_recovered_rate": [0.35, 0.43],
            "fast_bad_mae_negative_rate": [0.20, 0.10],
            "fast_bad_mae_recovered_rate": [0.15, 0.25],
            "p90_mae_norm": [7.0, 6.5],
            "timeout_rate": [0.08, 0.04],
            "wide_barrier_25bps_rate": [0.12, 0.08],
            "mean_bars_policy": [8.0, 7.0],
        }
    )

    out = _add_utility_only_deltas(monthly)
    selected = out[out["selection"].eq("utility_minus_bad_mae_loss_timeout_0p50")].iloc[0]

    assert selected["delta_bad_mae_negative_rate_vs_utility_only"] == pytest.approx(-0.13)
    assert selected["delta_bad_mae_recovered_rate_vs_utility_only"] == pytest.approx(0.08)
    assert selected["delta_fast_bad_mae_negative_rate_vs_utility_only"] == pytest.approx(-0.10)
    assert selected["delta_fast_bad_mae_recovered_rate_vs_utility_only"] == pytest.approx(0.10)
