from __future__ import annotations

import importlib.util
from pathlib import Path


RUNNER = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "audit_execution_utility_workstream_completion.py"
)
SPEC = importlib.util.spec_from_file_location("completion_audit", RUNNER)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_audit_stays_open_when_prospective_support_is_missing() -> None:
    rows = [
        {"requirement": "done", "status": "PROVED_COMPLETE"},
        {
            "requirement": "support",
            "status": "INCOMPLETE_EXTERNAL_SUPPORT",
        },
    ]
    assert (
        MODULE.audit_status(rows)
        == "IMPLEMENTATION_COMPLETE_PROSPECTIVE_ACCUMULATION_OPEN"
    )


def test_authoritative_requirements_include_negative_results_and_support_gate() -> None:
    rows = MODULE.requirement_rows(MODULE.load_sources())
    by_name = {row["requirement"]: row for row in rows}
    assert (
        by_name["direct_model_outperform_residual_on_untouched_oos"]["status"]
        == "COMPLETED_NEGATIVE_RESULT"
    )
    assert (
        by_name["accumulate_60_to_100_compatible_strict_incidents"]["status"]
        == "INCOMPLETE_EXTERNAL_SUPPORT"
    )
    assert (
        by_name["train_or_promote_failure_router"]["status"]
        == "CORRECTLY_NOT_AUTHORIZED"
    )
