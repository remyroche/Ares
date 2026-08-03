import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "data_perp" / "artifacts"


def test_hourly_cadence_supplement_is_sealed_and_explicit():
    p = ART / "pre2026_oof_model_failure_incremental_value_20260730_v4"
    manifest = json.loads((p / "manifest.json").read_text())
    assert hashlib.sha256((p / "manifest.json").read_bytes()).hexdigest() == (p / "manifest.sha256").read_text().split()[0]
    assert manifest["contract"]["decision_cadence"] == "1h"
    assert manifest["counts"]["all_timestamps_hour_aligned"] is True
    assert manifest["counts"]["all_labels_end_before_2026"] is True
    assert "1m data is nested" in manifest["contract"]["rule"]


def test_arm_local_coverage_preserves_trajectory_2023_only_where_supported():
    p = ART / "pre2026_oof_model_failure_incremental_value_20260730_v3"
    manifest = json.loads((p / "manifest.json").read_text())
    coverage = manifest["counts"]["arm_local_feature_coverage"]
    assert "2023_apr_dec" in coverage["trajectory"]["eras"]
    assert "2023_apr_dec" not in coverage["regime"]["eras"]
    assert "2023_apr_dec" not in coverage["transition"]["eras"]
    assert "2023_apr_dec" not in coverage["combined"]["eras"]
