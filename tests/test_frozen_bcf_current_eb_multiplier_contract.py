from pathlib import Path
import importlib.util
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("eb_multiplier", ROOT / "scripts/ablate_frozen_bcf_current_eb_multiplier.py")
assert SPEC and SPEC.loader
mod = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = mod
SPEC.loader.exec_module(mod)


def test_multiplier_variants_are_bounded_and_bcf_primary():
    assert any(v.name == "B1_frozen_dual_gate" and v.hard_current_gate for v in mod.VARIANTS)
    for variant in mod.VARIANTS:
        assert 0.0 < variant.lower <= variant.upper
        assert variant.weight >= 0.0


def test_insufficient_support_is_neutral():
    frame = pd.DataFrame({
        "__decision_ts__": pd.to_datetime(["2026-06-01T00:00:00Z"]),
        "policy_label_available_ts": pd.to_datetime(["2026-06-01T13:00:00Z"]),
        "policy_path_valid": [True],
        "policy_net_bps": [100.0],
        "bcf_mc1_expected_bps": [100.0],
        "current_mc1_expected_bps": [100.0],
        "candidate_id": ["x"],
    })
    alpha, beta, rows, days, status = mod._posterior(frame, pd.Timestamp("2026-06-02T00:00:00Z"))
    assert (alpha, beta, status) == (0.0, 0.0, "insufficient_prior_resolved_support")
    assert rows == 1 and days == 1


def test_target_free_panel_contains_no_policy_columns():
    # The runner writes target-free admission panels before outcome attachment;
    # this guards the forbidden-column contract used by the source adapter.
    assert "policy_net_bps" in mod.base.POLICY_FORBIDDEN
    assert "policy_label_available_ts" in mod.base.POLICY_FORBIDDEN
