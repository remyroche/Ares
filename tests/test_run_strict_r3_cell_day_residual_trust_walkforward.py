from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_strict_r3_cell_day_residual_trust_walkforward.py"
SPEC = importlib.util.spec_from_file_location("r5_walkforward", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _score() -> pd.DataFrame:
    ts = pd.to_datetime(["2026-01-01T00:00:00Z", "2026-02-01T00:00:00Z"])
    return pd.DataFrame({
        "candidate_id": ["a", "b"],
        "__decision_ts__": ts,
        "calibration_activation_ts": ts,
        "policy_label_available_ts": ts + pd.Timedelta(hours=12),
        "policy_path_valid": [True, True],
        "policy_net_bps": [10.0, 20.0],
        "geometry_bundle_sha256": ["frozen", "frozen"],
        "stack_is_prequential": [True, True],
        "final_score": [0.1, 0.2],
    })


def _mapped() -> pd.DataFrame:
    return pd.DataFrame({
        "candidate_id": ["a", "b"],
        "__decision_ts__": pd.to_datetime([
            "2026-01-01T00:00:00Z", "2026-02-01T00:00:00Z",
        ]),
        MODULE.CANONICAL_MAP_FIELD: [50.0, 60.0],
        MODULE.CANONICAL_ADMITTED_FIELD: [True, True],
    })


def test_join_rejects_nonprequential_source() -> None:
    score = _score()
    score.loc[1, "stack_is_prequential"] = False
    with pytest.raises(ValueError, match="non-prequential"):
        MODULE._join_inputs(
            score, _mapped(),
            expected_field=MODULE.CANONICAL_MAP_FIELD,
            admitted_field=MODULE.CANONICAL_ADMITTED_FIELD,
        )


def test_join_rejects_mixed_geometry_bundles() -> None:
    score = _score()
    score.loc[1, "geometry_bundle_sha256"] = "refit"
    with pytest.raises(ValueError, match="one frozen geometry"):
        MODULE._join_inputs(
            score, _mapped(),
            expected_field=MODULE.CANONICAL_MAP_FIELD,
            admitted_field=MODULE.CANONICAL_ADMITTED_FIELD,
        )


def test_join_requires_exact_timestamp_identity() -> None:
    mapped = _mapped()
    mapped.loc[0, "__decision_ts__"] += pd.Timedelta(hours=1)
    with pytest.raises(ValueError, match="identity/timestamp mismatch"):
        MODULE._join_inputs(
            _score(), mapped,
            expected_field=MODULE.CANONICAL_MAP_FIELD,
            admitted_field=MODULE.CANONICAL_ADMITTED_FIELD,
        )


def test_map_contract_requires_canonical_28d_cell_day_trim(tmp_path: Path) -> None:
    sidecar = tmp_path / "cell_day_bayesian_selection.parquet"
    sidecar.write_bytes(b"placeholder")
    manifest = {
        "schema": "strict_r3_cell_day_bayesian_ev_map_ablation_v1",
        "rolling_window_days": 28,
        "arms": ["cell_day_trim_15pct"],
        "period_weighting": "one observation per UTC day x fixed score cell",
    }
    (tmp_path / "run_manifest.json").write_text(json.dumps(manifest))
    loaded = MODULE._load_map_contract(
        sidecar, expected_field=MODULE.CANONICAL_MAP_FIELD,
    )
    assert loaded["rolling_window_days"] == 28
    manifest["rolling_window_days"] = 42
    (tmp_path / "run_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="28-calendar-day"):
        MODULE._load_map_contract(sidecar, expected_field=MODULE.CANONICAL_MAP_FIELD)


def test_runner_has_no_held_outcome_input_to_bundle_score() -> None:
    text = SCRIPT.read_text()
    assert 'inputs = held.loc[:, ["candidate_id", *bundle.fields]].copy()' in text
    assert 'inputs["raw_expected_bps"]' in text
    assert '"policy_net_bps"' not in text.split(
        'inputs = held.loc[:, ["candidate_id", *bundle.fields]].copy()', 1
    )[1].split("prediction = bundle.score(inputs)", 1)[0]


def test_runner_physically_limits_each_fit_source_to_nine_months() -> None:
    text = SCRIPT.read_text()
    assert "training_start = cutoff - pd.DateOffset(months=9)" in text
    assert 'joined["__decision_ts__"].ge(training_start)' in text
    assert 'joined["__decision_ts__"].lt(cutoff)' in text
    assert "train_cell_day_residual_trust_bundle(\n                fit_source," in text


def test_complete_nine_month_history_gate() -> None:
    required = pd.Timestamp("2025-01-08T00:00:00Z")
    assert MODULE._has_full_training_window(
        pd.Timestamp("2025-01-01T00:00:00Z"), required_start=required,
    )
    assert not MODULE._has_full_training_window(
        pd.Timestamp("2025-01-09T00:00:00Z"), required_start=required,
    )
    assert not MODULE._has_full_training_window(pd.NaT, required_start=required)
