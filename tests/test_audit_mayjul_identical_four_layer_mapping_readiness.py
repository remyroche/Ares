from __future__ import annotations

import importlib.util
from pathlib import Path


RUNNER = Path(__file__).resolve().parents[1] / "scripts" / "audit_mayjul_identical_four_layer_mapping_readiness.py"
SPEC = importlib.util.spec_from_file_location("mayjul_identical_mapping_readiness", RUNNER)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_exact_direct_score_identity_is_ready_but_availability_fails_closed() -> None:
    readiness, coverage, extra = MODULE.audit(MODULE.DEFAULT_WATERFALL, MODULE.DEFAULT_DIRECT)
    bounds = extra["bounds"]
    assert bounds["exact_identity_rows"] == 127777
    assert bounds["exact_direct_score_rows"] == 127777
    assert bounds["direct_score_bit_identical_to_waterfall"]
    assert readiness.loc[readiness.requirement.str.contains("availability"), "available"].eq(False).all()
    assert readiness.loc[readiness.requirement.str.contains("lineage"), "available"].eq(False).all()
    assert coverage["rows"].sum() == 127777


def test_causal_support_is_strictly_resolved_and_preserves_warmup() -> None:
    _, _, extra = MODULE.audit(MODULE.DEFAULT_WATERFALL, MODULE.DEFAULT_DIRECT)
    support = extra["support"]
    assert support["strictly_resolved_before_snapshot"].all()
    assert support["map_support_available"].any()
    assert (~support["map_support_available"]).any()
