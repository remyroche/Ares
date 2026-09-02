from pathlib import Path
import importlib.util
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("sr_router", ROOT / "scripts/ablate_causal_sr_selective_router.py")
assert SPEC and SPEC.loader
mod = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = mod
SPEC.loader.exec_module(mod)


def _panel():
    return pd.DataFrame({
        "candidate_id": ["available", "missing"],
        "__decision_ts__": pd.to_datetime(["2026-08-01T00:00:00Z", "2026-08-01T00:00:00Z"]),
        "bcf_mc1_expected_bps_c0": [60.0, 60.0], "current_mc1_expected_bps_c0": [60.0, 60.0],
        "bcf_mc1_expected_bps_c1": [70.0, 10.0], "current_mc1_expected_bps_c1": [70.0, 10.0],
        "sr_snapshot_available": [True, False],
    })


def test_selective_router_preserves_c0_when_snapshot_missing():
    routed = mod._route(_panel(), "R1_C1_when_sr_available_else_C0")
    assert routed.loc[routed.candidate_id.eq("available"), "route_used_c1"].item()
    assert not routed.loc[routed.candidate_id.eq("missing"), "route_used_c1"].item()
    assert routed.dual_admitted.tolist() == [True, True]


def test_additive_router_never_removes_c0_admission():
    routed = mod._route(_panel(), "R2_C1_additive_on_sr_available")
    assert routed.dual_admitted.tolist() == [True, True]


def test_router_names_are_explicitly_supported():
    for arm in ("C0_core", "C1_all", "R1_C1_when_sr_available_else_C0", "R2_C1_additive_on_sr_available"):
        assert len(mod._route(_panel(), arm)) == 2
