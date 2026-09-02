from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "materialize_strict_r3_p8u_meta_gateproxy_confirmation_plan_v1.py"
SPEC = importlib.util.spec_from_file_location("strict_r3_p8u_meta_gateproxy_plan", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_targetquery_descriptor_root_is_a_supported_provenance_alias() -> None:
    roots = MODULE._descriptor_score_roots({"target_query_roots": ["/tmp/score_a"]})
    assert set(roots) == {"score_a"}


def test_descriptor_requires_a_nonempty_score_root_list() -> None:
    with pytest.raises(AssertionError, match="score-root provenance"):
        MODULE._descriptor_score_roots({})
