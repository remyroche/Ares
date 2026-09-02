from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from extreme_price_movements.strict_r3_rich_policy import RichPolicyParams


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "orthogonal_meta_semantics",
    ROOT / "scripts" / "materialize_strict_r3_orthogonal_meta_semantics.py",
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _frame() -> pd.DataFrame:
    return pd.DataFrame({
        "candidate_id": ["valid", "invalid"],
        "__decision_ts__": pd.to_datetime(["2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z"]),
        "__symbol__": ["TEST/USD:USD", "TEST/USD:USD"],
        "entry_price": [100.0, np.nan],
        "path_arch_atr_fraction": [0.01, np.nan],
        "supportive_path_valid": [1, 0],
        "supportive_label_available_ts": pd.to_datetime(["2026-01-01T12:00:00Z", "2026-01-01T12:00:00Z"]),
        "policy_path_valid": [True, False],
        "policy_net_bps": [80.0, np.nan],
        "policy_exit_reason": ["trailing", "invalid_path"],
        "path_arch_peak_mfe_atr": [2.0, np.nan],
        "path_arch_mae_before_meaningful_mfe_r": [0.001, np.nan],
        "path_arch_peak_retention_ratio": [0.8, np.nan],
        "path_arch_final_return_r": [0.02, np.nan],
    })


def test_policy_tbm_uses_frozen_upper_cost_floor_and_stop(tmp_path: Path) -> None:
    index = pd.date_range("2026-01-01T00:00:00Z", periods=48, freq="15min")
    bars = pd.DataFrame({"high": 100.0, "low": 100.0}, index=index)
    # The first complete post-decision bar reaches the initial upper barrier.
    bars.iloc[0, bars.columns.get_loc("high")] = 101.2
    bars.to_parquet(tmp_path / "testusd:usd_15m.parquet")
    params = RichPolicyParams(
        sl_mult=1.0,
        trailing_activation_mult=1.0,
        sl_abs_floor_pct=0.0,
        sl_abs_cap_pct=0.0,
    )
    tbm = MODULE._policy_tbm(
        _frame(), bars_root=tmp_path, params=params, median_atr_fraction=0.01,
    )
    row = tbm.loc[tbm.candidate_id.eq("valid")].iloc[0]
    assert bool(row.semantic_tbm_path_complete)
    assert row.semantic_tbm_event == "upper_first"
    assert row.semantic_upper_bar == 1
    # With entry=100 and ATR=1, the cost floor and activation are each 1 ATR.
    assert np.isclose(row.semantic_upper_distance_atr, 1.0)
    assert np.isclose(row.semantic_lower_distance_atr, 1.0)


def test_invalid_path_remains_unlabelled_not_an_economic_failure(tmp_path: Path) -> None:
    index = pd.date_range("2026-01-01T00:00:00Z", periods=48, freq="15min")
    pd.DataFrame({"high": 100.0, "low": 100.0}, index=index).to_parquet(
        tmp_path / "testusd:usd_15m.parquet"
    )
    params = RichPolicyParams(sl_mult=1.0, trailing_activation_mult=1.0)
    frame = _frame()
    tbm = MODULE._policy_tbm(frame, bars_root=tmp_path, params=params, median_atr_fraction=0.01)
    semantic = MODULE._path_axes(frame, tbm)
    invalid = semantic.loc[semantic.candidate_id.eq("invalid")].iloc[0]
    assert not bool(invalid.semantic_path_valid)
    assert pd.isna(invalid.semantic_composite)
    assert pd.isna(invalid.semantic_tbm_event)
