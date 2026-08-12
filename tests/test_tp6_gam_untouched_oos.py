from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "data_perp/artifacts/tp6_sl4_gam_untouched_oos_2026_20260815_v3"


def test_untouched_oos_contract_is_single_field_and_prequential() -> None:
    manifest = json.loads((ARTIFACT / "run_manifest.json").read_text())
    correctness = json.loads((ARTIFACT / "correctness_test_report.json").read_text())
    assert manifest["status"] == "COMPLETE"
    assert manifest["canonical_field"] == "gam_delta_bps"
    assert manifest["gamma"] == 0.25
    assert manifest["base_ev_modulation"] is False
    assert manifest["target_month_outcomes_used_in_any_fit"] is False
    assert correctness["training_months_for_target_gam"] == ["2026-06"]
    assert correctness["target_month_outcomes_used_in_gam_fit"] is False
    assert correctness["target_month_outcomes_used_in_meta_fit"] is False


def test_untouched_oos_rows_and_scores_are_reproducible_contract() -> None:
    predictions = pd.read_parquet(ARTIFACT / "untouched_oos_predictions.parquet")
    assert len(predictions) == 852
    assert predictions["candidate_id"].is_unique
    assert predictions["month"].astype(str).eq("2026-07").all()
    assert predictions["gam_delta_bps"].notna().all()
    assert np.isfinite(predictions[["control_score", "gated_gam_score"]].to_numpy(float)).all()
    invalid = ~predictions["rolling_transport_valid"].astype(bool)
    if invalid.any():
        np.testing.assert_allclose(
            predictions.loc[invalid, "gated_gam_score"].to_numpy(float),
            predictions.loc[invalid, "control_score"].to_numpy(float),
        )


def test_untouched_oos_metrics_include_matched_global_tails() -> None:
    metrics = pd.read_parquet(ARTIFACT / "untouched_oos_metrics.parquet")
    assert set(metrics["arm"]) == {"control_score", "gated_gam_score"}
    assert set(metrics["tail"]) == {0.005, 0.01, 0.02, 0.05, 0.10}
    assert metrics.groupby("arm").size().to_dict() == {"control_score": 5, "gated_gam_score": 5}
