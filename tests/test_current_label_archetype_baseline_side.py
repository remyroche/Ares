from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_current_label_archetype_baseline import (
    _candidate_key_from_row,
    _render_side_coverage_section,
    _side_coverage_blocker_sentence,
    _summarize_candidates,
    _summary_rows,
    _timeout_stage1_rows,
)


def _base_frames() -> dict[str, pd.DataFrame]:
    path_row = {
        "label": "utility_linear_source_q80_v1",
        "risk_target": "bad_mae_risk_v1",
        "feature_set": "base_plus_source",
        "source_bucket": "risk_adjusted_capture_candidate",
        "causal_gate": "low_barrier_pressure_q50",
        "selection": "utility_minus_risk",
        "top_frac": 0.03,
        "q25_week_u": 0.002,
        "positive_weeks": 3,
        "weeks": 4,
        "mean_u": 0.01,
        "min_selected_rows": 10,
        "max_top_symbol_share": 0.7,
        "max_week_side_top_share": 0.8,
        "decision": "diagnostic_only",
    }
    joint_row = {
        "label": "utility_linear_source_q80_v1",
        "risk_heads": "timeout_risk_v1",
        "feature_set": "base_plus_source",
        "source_bucket": "risk_adjusted_capture_candidate",
        "causal_gate": "no_gate",
        "selection": "utility_minus_timeout_0p50",
        "top_frac": 0.01,
        "q25_week_u": 0.003,
        "positive_weeks": 4,
        "weeks": 4,
        "mean_u": 0.02,
        "min_selected_rows": 10,
        "max_week_top_symbol_share": 0.6,
        "overall_side_top_share": 0.75,
        "decision": "diagnostic_only",
    }
    return {
        "source_quality_aggregate": pd.DataFrame(
            [{"mean_u": 0.0, "worst_month_mean_u": -0.1}]
        ),
        "utility_label_rework": pd.DataFrame(
            [{"mean_model_u": 0.0, "worst_model_month_u": -0.1}]
        ),
        "utility_risk_gate": pd.DataFrame(
            [
                {
                    "decision": "candidate_gate_within_economic_limits",
                    "label": "utility_linear_source_q80_v1",
                    "source_bucket": "all_rows",
                    "risk_gate": "high_barrier_relief_q50",
                    "top_frac": 0.01,
                    "mean_u": 0.01,
                    "worst_month_u": 0.001,
                }
            ]
        ),
        "candidate_weekly_aggregate": pd.DataFrame(
            [
                {
                    "candidate": "weekly_c",
                    "label": "utility_linear_source_q80_v1",
                    "source_bucket": "risk_adjusted_capture_candidate",
                    "risk_gate": "low_barrier_pressure_q50",
                    "top_frac": 0.1,
                    "q25_week_u": 0.001,
                    "positive_weeks": 2,
                    "weeks": 3,
                    "mean_u": 0.01,
                    "max_top_symbol_share": 0.9,
                    "max_side_top_share": 0.85,
                }
            ]
        ),
        "path_risk_aggregate": pd.DataFrame([path_row]),
        "joint_path_timeout_aggregate": pd.DataFrame([joint_row]),
        "candidate_selected_rows": pd.DataFrame(
            {
                "candidate": ["weekly_c", "weekly_c", "weekly_c"],
                "week_start": ["2026-04-06"] * 3,
                "primary_source_tag": ["risk", "risk", "compression"],
                "side_name": ["short", "short", "long"],
            }
        ),
        "path_risk_selected_rows": pd.DataFrame(
            {
                "candidate": [_candidate_key_from_row(pd.Series(path_row))] * 3,
                "week_start": ["2026-04-06"] * 3,
                "primary_source_tag": ["risk", "compression", "risk"],
                "side": [1, -1, -1],
            }
        ),
        "joint_path_timeout_selected_rows": pd.DataFrame(
            {
                "candidate": [_candidate_key_from_row(pd.Series(joint_row))] * 4,
                "week_start": ["2026-04-06"] * 4,
                "primary_source_tag": ["risk", "risk", "risk", "compression"],
                "side_name": ["long", "long", "short", "short"],
            }
        ),
    }


def test_baseline_candidate_summary_includes_side_concentration() -> None:
    candidates = _summarize_candidates(_base_frames())
    rows = {row["scope"]: row for row in _summary_rows(candidates)}

    weekly = rows["Best weekly candidate"]
    assert weekly["max_side"] == 0.85
    assert weekly["top_side"] == "short"
    assert weekly["top_side_share"] == 2 / 3

    path = rows["Best path-risk candidate"]
    assert path["max_side"] == 0.8
    assert path["top_side"] == "short"
    assert path["top_side_share"] == 2 / 3

    joint = rows["Best joint path/timeout candidate"]
    assert joint["max_side"] == 0.75
    assert joint["top_side_share"] == 0.5


def test_timeout_stage1_rows_include_side_concentration_when_available() -> None:
    rows = _timeout_stage1_rows(
        {
            "timeout_stage1_weekaware_aggregate": pd.DataFrame(
                [
                    {
                        "decision": "candidate_timeout_filter",
                        "label": "timeout_risk_v1",
                        "label_kind": "timeout",
                        "feature_set": "base",
                        "selector": "low_risk_keep_weekly",
                        "fraction": 0.5,
                        "timeout_reduction_frac_vs_valid": 0.25,
                        "score_ic_timeout": 0.2,
                        "target_auc": 0.7,
                        "target_brier_score": 0.18,
                        "top_risk_decile_timeout_lift": 2.0,
                        "timeout_rate": 0.08,
                        "valid_timeout_rate": 0.12,
                        "delta_mean_u_vs_valid": -0.0001,
                        "q25_week_u_delta_vs_valid": 0.001,
                        "positive_weeks": 4,
                        "valid_positive_weeks": 4,
                        "min_week_selected_rows": 8,
                        "max_week_side_top_share": 0.67,
                    }
                ]
            )
        }
    )

    assert len(rows) == 1
    assert rows[0]["gate"] == "pass"
    assert rows[0]["max_side"] == 0.67


def test_side_coverage_audit_renders_bidirectional_blocker() -> None:
    audit = {
        "decision": "long_only_or_missing_short_evidence",
        "bidirectional_evidence_ready": False,
        "registry_summary": {
            "registries": 4,
            "registries_with_long": 4,
            "registries_with_short": 0,
            "total_long_rows": 4,
            "total_short_rows": 0,
            "bidirectional_registries": 0,
        },
        "artifacts": [
            {
                "role": "label_ledger",
                "status": "long_only_or_single_side",
                "rows": 10,
                "side_counts": {"long": 10, "short": 0},
                "top_side_share": 1.0,
                "bidirectional": False,
                "failures": ["missing_short_rows"],
            }
        ],
    }

    blocker = _side_coverage_blocker_sentence(audit)
    section = "\n".join(_render_side_coverage_section(audit))

    assert "4 long rows and 0 short rows" in blocker
    assert "short-side performance must be materialized from real short candidates" in blocker
    assert "## Side Coverage Audit" in section
    assert "long_only_or_missing_short_evidence" in section
    assert "label_ledger" in section
    assert "long: 10, short: 0" in section


def test_side_coverage_blocker_distinguishes_registry_scaffold_from_ledgers() -> None:
    audit = {
        "decision": "long_only_or_missing_short_evidence",
        "bidirectional_evidence_ready": False,
        "registry_summary": {
            "registries": 80,
            "registries_with_long": 80,
            "registries_with_short": 4,
            "total_long_rows": 80,
            "total_short_rows": 4,
            "bidirectional_registries": 4,
        },
        "blocking_artifacts": [
            {"role": "label_ledger"},
            {"role": "weekly_candidate_selected_rows"},
            {"role": "path_risk_selected_rows"},
            {"role": "joint_path_timeout_selected_rows"},
            {"role": "archetype_materialized_rows"},
            {"role": "timeout_stage1_weekaware_selected_rows"},
        ],
        "artifacts": [],
    }

    blocker = _side_coverage_blocker_sentence(audit)

    assert "4 bidirectional registries" in blocker
    assert "80 long rows, 4 short rows" in blocker
    assert "required diagnostic ledgers remain long-only" in blocker
    assert "`label_ledger`" in blocker
    assert "plus 1 more" in blocker
