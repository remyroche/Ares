import importlib.util
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "div2", ROOT / "scripts" / "diagnose_historical_base_ic_ev_divergence_v2.py"
)
MOD = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MOD)


def _rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "candidate_id": ["a", "b", "c", "d"],
            "side_name": ["long", "long", "short", "short"],
            "__symbol__": ["x", "x", "y", "z"],
            "__ts__": pd.to_datetime(
                ["2025-02-01T00:00Z"] * 2 + ["2025-02-01T01:00Z"] * 2
            ),
            "old_score": [4.0, 3.0, 2.0, 1.0],
            "target24": [1.0, 0.8, 0.2, 0.0],
            "execution_gross_ev_12h": [0.02, 0.01, 0.0, -0.01],
            "execution_cost_return": [0.01] * 4,
            "execution_net_ev_12h": [0.01, 0.0, -0.01, -0.02],
            "execution_exit_reason": ["trailing", "full_sl", "timeout", "timeout"],
            "execution_mfe_return_12h": [0.03, 0.01, 0.0, 0.0],
            "execution_mae_return_12h": [0.01, 0.04, 0.02, 0.03],
            "execution_exit_minute": [120, 300, 720, 720],
            "exit_time_bucket": ["early", "mid", "late", "late"],
        }
    )


def test_top_is_pooled_global_not_per_timestamp():
    result = MOD.top(_rows())
    assert result["rows"] == 1
    assert result["side_capacity"] == {"long": 1}


def test_top_persists_concentration_and_exit_conditional_payoffs():
    result = MOD.top(_rows(), frac=0.5)
    assert result["asset_count"] == 1
    assert result["asset_top_share"] == 1.0
    assert result["asset_hhi"] == 1.0
    by_exit = {row["exit_reason"]: row for row in result["exit_conditional_path"]}
    assert by_exit["trailing"]["share"] == 0.5
    assert by_exit["trailing"]["net_bps"] == 100.0
    assert by_exit["full_sl"]["mae_bps"] == 400.0


def test_deciles_preserve_score_order_and_economic_scale():
    rows = pd.concat([_rows()] * 3, ignore_index=True).iloc[:10].copy()
    rows["candidate_id"] = [f"c{i}" for i in range(10)]
    rows["old_score"] = range(10)
    rows["target24"] = rows["old_score"] / 9
    rows["execution_gross_ev_12h"] = rows["old_score"] / 100
    rows["execution_net_ev_12h"] = (
        rows["execution_gross_ev_12h"] - rows["execution_cost_return"]
    )
    deciles = MOD.deciles(rows)
    assert len(deciles) == 10
    assert deciles[0]["native_target"] < deciles[-1]["native_target"]
    assert deciles[0]["net"] < deciles[-1]["net"]
