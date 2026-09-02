from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pandas as pd


SCRIPT = Path(__file__).parents[1] / "scripts" / "run_strict_r3_incumbent_meta_mc1_combinations_v1.py"
SPEC = importlib.util.spec_from_file_location("incumbent_meta_mc1_combos", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_top_substitution_is_post_selection_and_uses_bcf_priority() -> None:
    timestamp = pd.Timestamp("2026-04-01T00:00:00Z")
    admissions = pd.DataFrame({
        "combo": ["R__U", "R__U", "U"],
        "candidate_id": ["a", "b", "b"],
        "__decision_ts__": [timestamp, timestamp, timestamp],
        "bcf_mc1_expected_bps": [100.0, 80.0, 90.0],
        "policy_net_bps": [50.0, 10.0, 10.0],
    })
    result = MODULE._top_substitution(admissions, combo="R__U", comparator="U", k=1)
    assert result["combo_only_selected"] == 1
    assert result["comparator_only_selected"] == 1
    assert result["substitution_delta_policy_net_bps"] == 40.0
    assert result["post_selection_policy_diagnostic_only"] is True
