import pandas as pd

from scripts.run_strict_oos_repair_ranker_frozen_validation import (
    _combine_csv,
    build_reference_consistency_audit,
    planned_months,
)


def test_planned_months_uses_frozen_history_then_validation_periods():
    manifest = {
        "non_promotion_periods": ["2026-04", "2026-05", "2026-06"],
        "validation_periods": ["2026-07"],
    }

    assert planned_months(profile_manifest=manifest) == ["2026-04", "2026-05", "2026-06", "2026-07"]


def test_planned_months_accepts_cli_validation_override():
    manifest = {
        "non_promotion_periods": ["2026-04", "2026-05", "2026-06"],
        "validation_periods": ["2026-07"],
    }

    assert planned_months(profile_manifest=manifest, validation_periods=["2026-08"]) == [
        "2026-04",
        "2026-05",
        "2026-06",
        "2026-08",
    ]


def test_planned_months_accepts_cli_history_override_and_dedupes():
    manifest = {
        "non_promotion_periods": ["2026-04", "2026-05", "2026-06"],
        "validation_periods": ["2026-07"],
    }

    assert planned_months(
        profile_manifest=manifest,
        history_periods=["2026-05", "2026-06", "2026-07"],
        validation_periods=["2026-07", "2026-08"],
    ) == ["2026-05", "2026-06", "2026-07", "2026-08"]


def test_combine_csv_skips_empty_inputs(tmp_path):
    empty = tmp_path / "empty.csv"
    non_empty = tmp_path / "data.csv"
    out = tmp_path / "combined.csv"
    empty.write_text("")
    pd.DataFrame([{"a": 1}]).to_csv(non_empty, index=False)

    rows = _combine_csv([empty, non_empty], out)

    assert rows == 1
    assert pd.read_csv(out).to_dict("records") == [{"a": 1}]


def _monthly_row(**overrides):
    row = {
        "period": "2026-06",
        "source_bucket": "compression_capture_dirty_excluded",
        "proxy_col": "oof_pred",
        "top_frac": 0.1,
        "feature_mode": "frozen_features",
        "selection_method": "repair_proxy_blend_70_30",
        "selected_rows": 13,
        "repair_mean_u": 0.024629,
        "proxy_mean_u": -0.014757,
    }
    row.update(overrides)
    return row


def test_reference_consistency_passes_for_matching_rows():
    frozen = pd.DataFrame([_monthly_row()])
    reference = pd.DataFrame([_monthly_row(), _monthly_row(period="2026-05")])

    audit, summary = build_reference_consistency_audit(frozen, reference)

    assert audit.loc[0, "consistency_status"] == "matches_reference"
    assert summary["passes"] is True


def test_reference_consistency_fails_on_missing_reference_row():
    frozen = pd.DataFrame([_monthly_row(period="2026-07")])
    reference = pd.DataFrame([_monthly_row(period="2026-06")])

    audit, summary = build_reference_consistency_audit(frozen, reference)

    assert audit.loc[0, "consistency_status"] == "missing_reference_row"
    assert summary["passes"] is False


def test_reference_consistency_fails_on_metric_difference():
    frozen = pd.DataFrame([_monthly_row(repair_mean_u=0.024629)])
    reference = pd.DataFrame([_monthly_row(repair_mean_u=0.010000)])

    audit, summary = build_reference_consistency_audit(frozen, reference)

    assert audit.loc[0, "consistency_status"] == "differs_from_reference"
    assert "repair_mean_u" in audit.loc[0, "mismatch_columns"]
    assert summary["passes"] is False
