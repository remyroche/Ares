import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "verify_live_ledger_feature_json_parity.py"
SPEC = importlib.util.spec_from_file_location("verify_live_ledger_feature_json_parity", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
PARITY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PARITY)


def test_coverage_gate_rejects_zero_evidence_even_when_cli_minimums_are_zero():
    failures = PARITY._coverage_gate_failures(
        {"compared_rows": 0, "common_cells": 0},
        min_compared_rows=0,
        min_common_cells=0,
        require_complete_sidecar_coverage=False,
    )

    assert failures == ["no_compared_rows", "no_common_cells"]
