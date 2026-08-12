from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "combine_strict_r3_ev_map_fallbacks.py"
SPEC = importlib.util.spec_from_file_location("ev_map_fallbacks", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_fallback_only_enters_at_timestamp_with_no_exact_admission() -> None:
    frame = pd.DataFrame({
        "candidate_id": ["a", "b", "c", "d"],
        "__decision_ts__": pd.to_datetime([
            "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z",
            "2026-01-01T01:00:00Z", "2026-01-01T01:00:00Z",
        ], utc=True),
        "exact_reserve_control__admitted": [True, False, False, False],
        "exact_reserve_control__expected_net_bps": [80.0, 20.0, 10.0, 10.0],
        "cell_day_trim_15pct__admitted": [True, True, True, False],
        "cell_day_trim_15pct__expected_net_bps": [90.0, 70.0, 60.0, 20.0],
        "cell_day_equal_weight__admitted": [True, True, True, False],
        "cell_day_equal_weight__expected_net_bps": [85.0, 65.0, 55.0, 15.0],
        "bayes_k07_p90__admitted": [True, True, True, True],
        "bayes_k07_p90__expected_net_bps": [75.0, 75.0, 65.0, 55.0],
    })
    out = MODULE.build_fallbacks(frame)
    arm = "exact_primary_trim15_timestamp_fallback"
    assert out.loc[out.candidate_id.eq("a"), f"{arm}__admitted"].item()
    assert not out.loc[out.candidate_id.eq("b"), f"{arm}__admitted"].item()
    assert out.loc[out.candidate_id.eq("c"), f"{arm}__admitted"].item()
    assert not out.loc[out.candidate_id.eq("d"), f"{arm}__admitted"].item()
