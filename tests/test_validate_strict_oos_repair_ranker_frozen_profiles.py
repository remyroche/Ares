import pandas as pd

from scripts.validate_strict_oos_repair_ranker_frozen_profiles import evaluate_profile


PROFILE = {
    "name": "candidate",
    "source_bucket": "compression_capture_dirty_excluded",
    "proxy_col": "oof_pred",
    "top_frac": 0.1,
    "feature_mode": "frozen_features",
    "selection_method": "repair_proxy_blend_70_30",
}

GUARDS = {
    "min_months": 1,
    "min_selected_rows": 5,
    "require_positive_each_period": True,
    "require_delta_positive_each_period": True,
    "min_mean_repair_u": 0.0,
    "min_worst_month_repair_u": 0.0,
    "min_mean_delta_u_vs_proxy": 0.0,
    "min_worst_month_delta_u_vs_proxy": 0.0,
    "min_oracle_capture": 0.05,
    "min_oracle_capture_delta": 0.0,
    "max_bad_mae_excess": 0.15,
    "max_timeout_excess": 0.15,
    "max_repair_bad_mae_rate": 0.85,
    "max_repair_timeout_or_slow_holding_rate": 0.50,
}


def _ledger_row(period="2026-07", **overrides):
    row = {
        "period": period,
        "source_bucket": "compression_capture_dirty_excluded",
        "proxy_col": "oof_pred",
        "top_frac": 0.1,
        "feature_mode": "frozen_features",
        "selection_method": "repair_proxy_blend_70_30",
        "selected_rows": 13,
        "repair_mean_u": 0.024,
        "proxy_mean_u": -0.010,
        "repair_delta_mean_u_vs_proxy": 0.034,
        "repair_oracle_capture_at_k": 0.15,
        "proxy_oracle_capture_at_k": 0.0,
        "repair_delta_oracle_capture_at_k": 0.15,
        "repair_bad_mae_1r_rate": 0.70,
        "proxy_bad_mae_1r_rate": 0.80,
        "repair_timeout_or_slow_holding_rate": 0.10,
        "proxy_timeout_or_slow_holding_rate": 0.12,
    }
    row.update(overrides)
    return row


def test_profile_passes_future_validation_when_all_guards_hold():
    monthly, aggregate = evaluate_profile(
        pd.DataFrame([_ledger_row()]),
        PROFILE,
        validation_periods=["2026-07"],
        guards=GUARDS,
        non_promotion_periods={"2026-04", "2026-05", "2026-06"},
    )

    assert monthly.loc[0, "period_status"] == "passes_period_guards"
    assert aggregate["validation_status"] == "passes_frozen_validation"
    assert aggregate["promotion_allowed"] is True


def test_profile_is_not_promotion_evidence_on_retrospective_period():
    _, aggregate = evaluate_profile(
        pd.DataFrame([_ledger_row(period="2026-06")]),
        PROFILE,
        validation_periods=["2026-06"],
        guards=GUARDS,
        non_promotion_periods={"2026-04", "2026-05", "2026-06"},
    )

    assert aggregate["validation_status"] == "passes_guards_but_retrospective_only"
    assert aggregate["promotion_allowed"] is False


def test_profile_fails_when_repair_is_negative_even_if_delta_beats_proxy():
    monthly, aggregate = evaluate_profile(
        pd.DataFrame([_ledger_row(repair_mean_u=-0.004, proxy_mean_u=-0.010, repair_delta_mean_u_vs_proxy=0.006)]),
        PROFILE,
        validation_periods=["2026-07"],
        guards=GUARDS,
        non_promotion_periods=set(),
    )

    assert monthly.loc[0, "period_status"] == "fails_period_guards"
    assert "non_positive_repair_mean" in monthly.loc[0, "period_failure_reasons"]
    assert aggregate["validation_status"] == "fails_frozen_validation"
    assert "non_positive_repair_month" in aggregate["failure_reasons"]


def test_profile_reports_missing_validation_period():
    monthly, aggregate = evaluate_profile(
        pd.DataFrame([_ledger_row(period="2026-06")]),
        PROFILE,
        validation_periods=["2026-07"],
        guards=GUARDS,
        non_promotion_periods=set(),
    )

    assert monthly.empty
    assert "period" in monthly.columns
    assert "period_status" in monthly.columns
    assert aggregate["validation_status"] == "fails_frozen_validation"
    assert aggregate["missing_periods"] == "2026-07"
    assert "missing_validation_periods" in aggregate["failure_reasons"]


def test_profile_fails_on_path_risk_excess():
    monthly, aggregate = evaluate_profile(
        pd.DataFrame(
            [
                _ledger_row(
                    repair_bad_mae_1r_rate=0.90,
                    proxy_bad_mae_1r_rate=0.60,
                    repair_timeout_or_slow_holding_rate=0.40,
                    proxy_timeout_or_slow_holding_rate=0.10,
                )
            ]
        ),
        PROFILE,
        validation_periods=["2026-07"],
        guards=GUARDS,
        non_promotion_periods=set(),
    )

    reasons = monthly.loc[0, "period_failure_reasons"]
    assert "bad_mae_excess" in reasons
    assert "timeout_excess" in reasons
    assert "repair_bad_mae_rate_too_high" in reasons
    assert aggregate["validation_status"] == "fails_frozen_validation"
