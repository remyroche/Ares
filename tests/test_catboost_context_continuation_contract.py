import importlib.util
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = (
    ROOT
    / "data_perp/artifacts/historical_execution_ev_catboost_context_continuation_20260729_v1"
)
SPEC = importlib.util.spec_from_file_location(
    "ctx_cont",
    ROOT / "scripts" / "run_historical_execution_ev_catboost_context_continuation.py",
)
M = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(M)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_purge_excludes_unresolved_label():
    rows = pd.DataFrame(
        {
            "__ts__": pd.to_datetime(
                ["2025-03-01T00:00Z", "2025-03-01T01:00Z"]
            ),
            "execution_label_end_utc": pd.to_datetime(
                ["2025-03-02T00:00Z", "2025-03-11T00:00Z"]
            ),
        }
    )
    assert len(M._purged_before(rows, pd.Timestamp("2025-03-10", tz="UTC"))) == 1


def test_frozen_artifact_binds_runner_outputs_and_no_context_controls():
    manifest = json.loads((ARTIFACT / "manifest.json").read_text())
    report = json.loads((ARTIFACT / "report.json").read_text())
    assert _sha(ROOT / "scripts/run_historical_execution_ev_catboost_context_continuation.py") == manifest["runner_sha256"]
    for relative, expected in manifest["output_sha256"].items():
        assert _sha(ARTIFACT / relative) == expected
    assert report["fit_plan"]["planned_model_fits"] == 360
    assert report["portfolio_eligibility"]["actual_completed_model_fits"] == 360
    assert report["portfolio_eligibility"]["eligible_arms"] == []
    assert "risk_peak_direct_net__no_context" in report["arms"]
    assert "risk_direct_gross__no_context" in report["arms"]


def test_april_ledger_reconciles_and_primary_tail_is_one_global_book():
    name = "risk_direct_gross__no_context"
    ledger = pd.read_parquet(ARTIFACT / name / "april_predictions.parquet")
    report = json.loads((ARTIFACT / "report.json").read_text())
    assert len(ledger) == 69_258
    np.testing.assert_allclose(
        ledger["execution_gross_ev_12h"] - ledger["execution_cost_return"],
        ledger["execution_net_ev_12h"],
        atol=1e-12,
        rtol=0,
    )
    count = int(np.ceil(0.10 * len(ledger)))
    selected = ledger.nlargest(count, "common_unit_score")
    assert len(selected) == 6_926
    # Selection is performed once across both sides, not as two side quotas.
    assert selected["side_name"].nunique() == 2
    expected = report["arms"][name]["april"]["common_unit"]["net_bps"]
    assert float(selected["execution_net_ev_12h"].mean() * 1e4) == expected
