from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.report_source_quality_current_state import (  # noqa: E402
    _alignment_summary,
    _label_distribution,
    _source_coverage,
    build_current_state_report,
)


def test_alignment_summary_fails_low_match_rates(tmp_path: Path) -> None:
    pd.DataFrame(
        [
            {
                "feature_input_rows": 100,
                "candidate_source_tags_rows": 100,
                "quality_label_candidates_rows": 100,
                "outcome_rows_matched": 7,
                "outcome_match_rate": 0.07,
                "prediction_rows": 20,
                "prediction_match_rate": 0.10,
                "duplicate_candidate_id_rows": 0,
                "duplicate_timestamp_symbol_rows": 0,
                "duplicate_timestamp_symbol_side_rows": 0,
                "rows_with_multiple_outcomes_joined": 0,
                "rows_with_multiple_predictions_joined": 0,
                "label_duplicate_keys": 0,
                "prediction_duplicate_keys": 0,
                "metadata_columns_preserved": 1,
                "alignment_quality": "warning",
                "alignment_warnings": "prediction_alignment_warning",
            }
        ]
    ).to_csv(tmp_path / "row_alignment_audit.csv", index=False)

    _, report = _alignment_summary(
        tmp_path,
        min_outcome_match_rate=0.80,
        min_prediction_match_rate=0.80,
    )

    assert report["status"] == "fail"
    assert any("outcome_match_rate" in item for item in report["failures"])
    assert any("prediction_match_rate" in item for item in report["failures"])


def test_source_coverage_and_label_distribution() -> None:
    source = pd.DataFrame(
        {
            "__ts__": pd.date_range("2026-04-01", periods=4, freq="h", tz="UTC"),
            "__symbol__": ["BTC", "ETH", "BTC", "SOL"],
            "tag_quiet_continuation": [True, False, True, False],
            "tag_dirty_shock_avoid": [False, True, False, False],
            "primary_source_tag": ["quiet_continuation", "dirty_shock_avoid", "quiet_continuation", "ambiguous_none"],
        }
    )
    labels = pd.DataFrame(
        {
            "quality_label_v0": [1, 0, -1, 0],
            "quality_label_source_wf_v1": [1, -1, -1, 0],
        }
    )

    coverage, summary = _source_coverage(source)
    distribution = _label_distribution(labels)

    quiet = coverage[(coverage["scope"].eq("multi_tag")) & (coverage["source"].eq("quiet_continuation"))].iloc[0]
    assert summary["rows"] == 4
    assert summary["symbols"] == 3
    assert quiet["rows"] == 2
    assert quiet["coverage_pct"] == 0.5
    v0 = distribution[distribution["label_col"].eq("quality_label_v0")].set_index("label_value")
    assert v0.loc["1", "rows"] == 1
    assert v0.loc["0", "rows"] == 2
    assert v0.loc["-1", "rows"] == 1


def test_build_current_state_report_writes_outputs_and_marks_alignment_fail(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    v2_dir = tmp_path / "v2"
    timeout_dir = tmp_path / "timeout"
    support_dir = tmp_path / "support"
    failure_dir = tmp_path / "failure"
    gap_dir = tmp_path / "gap"
    output_dir = tmp_path / "out"
    for directory in [source_dir, v2_dir, timeout_dir, support_dir, failure_dir, gap_dir]:
        directory.mkdir(parents=True)

    source = pd.DataFrame(
        {
            "__ts__": pd.date_range("2026-04-01", periods=4, freq="h", tz="UTC"),
            "__symbol__": ["BTC", "ETH", "BTC", "SOL"],
            "tag_quiet_continuation": [True, False, True, False],
            "tag_dirty_shock_avoid": [False, True, False, False],
            "primary_source_tag": ["quiet_continuation", "dirty_shock_avoid", "quiet_continuation", "ambiguous_none"],
        }
    )
    labels = source.copy()
    labels["quality_label_v0"] = [1, 0, -1, 0]
    labels["quality_label_source_rank_v1"] = [1, 0, -1, 0]
    labels["quality_label_source_wf_v1"] = [1, -1, -1, 0]
    labels["quality_label_clean_path_v2"] = [1, 0, -1, 0]
    labels["quality_label_recoverable_opportunity_v2"] = [1, 1, -1, 0]
    labels["quality_label_opportunity_capture_v3"] = [-1, 1, -1, 0]
    labels["quality_label_economic_capture_v4"] = [-1, 0, -1, 1]
    source.to_parquet(source_dir / "candidate_source_tags.parquet", index=False)
    labels.to_parquet(source_dir / "quality_label_candidates.parquet", index=False)
    pd.DataFrame(
        [
            {
                "feature_input_rows": 4,
                "candidate_source_tags_rows": 4,
                "quality_label_candidates_rows": 4,
                "outcome_rows_matched": 1,
                "outcome_match_rate": 0.25,
                "prediction_rows": 2,
                "prediction_match_rate": 0.50,
                "duplicate_candidate_id_rows": 0,
                "duplicate_timestamp_symbol_rows": 0,
                "duplicate_timestamp_symbol_side_rows": 0,
                "rows_with_multiple_outcomes_joined": 0,
                "rows_with_multiple_predictions_joined": 0,
                "label_duplicate_keys": 0,
                "prediction_duplicate_keys": 0,
                "metadata_columns_preserved": 1,
                "alignment_quality": "warning",
                "alignment_warnings": "prediction_alignment_warning",
            }
        ]
    ).to_csv(source_dir / "row_alignment_audit.csv", index=False)
    (source_dir / "label_ablation_manifest.json").write_text('{"experiments": [{"name": "baseline"}]}\n')
    pd.DataFrame({"bucket": ["quiet"], "rows": [2], "mean_net_utility": [0.01]}).to_csv(
        source_dir / "failure_mode_by_source.csv", index=False
    )
    pd.DataFrame({"bucket": ["quiet"], "outcome_rows": [2], "mean_net_utility": [0.02]}).to_csv(
        source_dir / "opportunity_capture_by_source.csv", index=False
    )
    (v2_dir / "manifest.json").write_text(
        '{"rows": 4, "join_report": {"join_match_rate_vs_quality": 1.0, "join_match_rate_vs_labels": 1.0}}\n'
    )
    pd.DataFrame({"archetype": ["source_evidence_archetype"], "rows": [2], "mean_utility": [0.01]}).to_csv(
        v2_dir / "source_archetypes_v2_scorecard.csv", index=False
    )
    pd.DataFrame({"target_auc": [0.7], "label": ["timeout"], "mean_u": [0.0]}).to_csv(
        timeout_dir / "timeout_holding_risk_label_aggregate.csv", index=False
    )
    for directory in [support_dir, failure_dir]:
        pd.DataFrame(
            {
                "decision": ["diagnostic_only"],
                "mean_u": [0.01],
                "bad_mae_negative_rate": [0.2],
                "selection": ["utility_only"],
            }
        ).to_csv(directory / "source_utility_path_timeout_risk_aggregate.csv", index=False)
    pd.DataFrame({"top_best_auc": [0.75], "top_feature": ["impulse"]}).to_csv(
        gap_dir / "bad_mae_recovery_feature_gap_summary.csv", index=False
    )

    manifest = build_current_state_report(
        source_dir=source_dir,
        v2_dir=v2_dir,
        timeout_dir=timeout_dir,
        recovery_support_dir=support_dir,
        recovery_failure_dir=failure_dir,
        bad_mae_gap_dir=gap_dir,
        clean_subset_dir=None,
        output_dir=output_dir,
        min_outcome_match_rate=0.80,
        min_prediction_match_rate=0.80,
    )

    assert manifest["status"] == "fail"
    assert manifest["artifact_status"] == "pass"
    assert Path(manifest["outputs"]["report"]).exists()
    assert Path(manifest["outputs"]["label_distribution"]).exists()
