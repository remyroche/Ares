from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_monthly_base_portability_diagnosis.py"
_SPEC = importlib.util.spec_from_file_location("monthly_base_portability_diagnosis", _PATH)
assert _SPEC and _SPEC.loader
_MOD = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MOD)


def _oof() -> pd.DataFrame:
    decision = pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC")
    return pd.DataFrame({
        "candidate_id": ["a", "b", "c"], "decision_ts": decision,
        "label_available_ts": decision + pd.Timedelta(hours=13),
        "base_fit_cutoff_ts": decision - pd.Timedelta(days=1), "side_name": "long",
        "r3_class": [0, 1, 2], "net_bps": [-10.0, 0.0, 10.0],
        "p_adverse": [0.7, 0.2, 0.1], "p_weak": [0.2, 0.6, 0.2], "p_clear": [0.1, 0.2, 0.7],
        "base_raw": [-0.6, 0.0, 0.6],
    })


def test_direct_strict_oof_contract_and_prior_resolved_boundary() -> None:
    frame = _MOD.validate_direct_strict_oof(_oof(), side="long")
    assert frame.r3_clear.tolist() == [0, 0, 1]
    support = _MOD.prior_resolved_oof_support(frame, decision_ts="2024-01-01T14:00:00Z")
    assert support.candidate_id.tolist() == ["a"]


def test_rejects_non_direct_converted_or_non_strict_scores() -> None:
    frame = _oof()
    frame.loc[0, "base_raw"] = 1.0
    with pytest.raises(_MOD.MonthlyBasePortabilityError, match="without conversion"):
        _MOD.validate_direct_strict_oof(frame, side="long")
    frame = _oof()
    frame.loc[0, "base_fit_cutoff_ts"] = frame.loc[0, "decision_ts"]
    with pytest.raises(_MOD.MonthlyBasePortabilityError, match="fit cutoff"):
        _MOD.validate_direct_strict_oof(frame, side="long")
