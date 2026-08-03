import importlib.util
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("ic_ev_divergence", ROOT / "scripts" / "diagnose_historical_base_ic_ev_divergence.py")
assert SPEC and SPEC.loader
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


def test_top10_economics_is_one_global_pool_not_per_timestamp():
    frame = pd.DataFrame(
        {
            "candidate_id": ["a", "b", "c", "d"], "side_name": ["long", "long", "short", "short"],
            "__symbol__": ["A", "B", "C", "D"], "__ts__": pd.to_datetime(["2025-03-01T00:00Z"] * 2 + ["2025-03-01T01:00Z"] * 2),
            "historical_base_soft_oof": [4.0, 3.0, 2.0, 1.0],
            "execution_gross_ev_12h": [0.02, 0.01, 0.0, -0.01], "execution_cost_return": [0.01] * 4,
            "execution_net_ev_12h": [0.01, 0.0, -0.01, -0.02], "execution_exit_reason": ["target"] * 4,
        }
    )
    result = MOD._economics(frame)
    assert result["rows"] == 1
    assert result["side_capacity"] == [{"side": "long", "rows": 1}]
