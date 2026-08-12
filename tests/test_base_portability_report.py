from __future__ import annotations

import json

import pandas as pd
import pytest

from extreme_price_movements.base_portability_report import (
    BasePortabilityReportError,
    build_base_portability_scorecard,
    dataframe_sha256,
    verify_immutable_base_portability_report,
    write_immutable_base_portability_report,
)


def _relationship() -> pd.DataFrame:
    return pd.DataFrame({
        "month": ["2025-01", "2025-02", "2025-03"],
        "scope": ["pooled"] * 3,
        "scope_value": ["all"] * 3,
        "n_rows": [100, 100, 100],
        "within_query_rank_ic": [.03, .04, .025],
        "top5_uplift": [.02, .01, .03],
        "top30_winner_recall": [.50, .46, .48],
        "top40_winner_recall": [.60, .56, .58],
        "decile_adjacent_violations": [0, 0, 0],
    })


def _input() -> pd.DataFrame:
    # Multiple feature rows per month exercise worst-feature aggregation.
    return pd.DataFrame({
        "month": ["2025-01", "2025-01", "2025-02", "2025-02", "2025-03", "2025-03"],
        "max_feature_psi": [.01, .02, .01, .03, .01, .02],
        "max_feature_extrapolation_rate": [.01, .02, .01, .01, .03, .01],
    })


def _stability() -> pd.DataFrame:
    return pd.DataFrame({
        "month": ["2025-01", "2025-02", "2025-03"],
        "frozen_rank_ic": [.03, .04, .025],
        "refit_rank_ic": [.031, .041, .026],
        "score_spearman": [.99, .98, .99],
        "top_05_overlap_fraction": [.99, .98, .99],
        "train_rank_ic": [.031, .041, .026],
        "calibration_slope_shift": [.01, .01, .01],
    })


def test_build_scorecard_merges_precomputed_tables_and_applies_goals() -> None:
    scorecard, summary = build_base_portability_scorecard(_relationship(), _input(), _stability())
    assert list(scorecard["period"]) == ["2025-01", "2025-02", "2025-03"]
    assert scorecard.loc[1, "max_feature_psi"] == pytest.approx(.03)
    assert set(scorecard["drift_attribution"]) == {"NO_DOMINANT_DRIFT"}
    assert summary["pooled_within_query_rank_ic"] == pytest.approx(.0316666667)
    assert summary["portable"] is True
    assert all(summary["gates"].values())


def test_scorecard_fails_closed_when_a_period_lacks_one_diagnostic_surface() -> None:
    with pytest.raises(BasePortabilityReportError, match="every relationship period"):
        build_base_portability_scorecard(_relationship(), _input().query("month != '2025-03'"), _stability())


def test_frame_hash_is_schema_and_column_order_stable_but_value_sensitive() -> None:
    frame = pd.DataFrame({"b": [1.0, 2.0], "a": ["x", "y"]})
    assert dataframe_sha256(frame) == dataframe_sha256(frame.loc[:, ["a", "b"]])
    changed = frame.copy()
    changed.loc[1, "b"] = 3.0
    assert dataframe_sha256(frame) != dataframe_sha256(changed)


def test_immutable_writer_hash_binds_outputs_and_refuses_overwrite(tmp_path) -> None:
    scorecard, summary = build_base_portability_scorecard(_relationship(), _input(), _stability())
    output = tmp_path / "sealed"
    result = write_immutable_base_portability_report(
        scorecard, summary, output, provenance={"source": "precomputed-only", "run": "unit-test"}
    )
    assert result.portable is True
    manifest = verify_immutable_base_portability_report(output)
    assert manifest["refit_or_data_loading_performed"] is False
    assert json.loads(result.summary_path.read_text(encoding="utf-8"))["portable"] is True
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_immutable_base_portability_report(scorecard, summary, output, provenance={"source": "again"})
    result.report_path.write_text("tampered", encoding="utf-8")
    with pytest.raises(BasePortabilityReportError, match="hash mismatch"):
        verify_immutable_base_portability_report(output)
