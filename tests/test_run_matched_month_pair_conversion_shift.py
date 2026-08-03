from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


RUNNER = Path(__file__).resolve().parents[1] / "scripts" / "run_matched_month_pair_conversion_shift.py"
SPEC = importlib.util.spec_from_file_location("matched_month_pair_conversion_shift", RUNNER)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _frame(month: str, *, shift: float = 0.0, rows: int = 600) -> pd.DataFrame:
    values = np.arange(rows, dtype=float)
    timestamp = pd.Timestamp(f"{month}-01T00:00:00Z") + pd.to_timedelta(values, unit="h")
    gross = (values / rows) * .03 + shift
    cost = np.full(rows, .01)
    return pd.DataFrame({
        "candidate_id": [f"{month}-{index}" for index in range(rows)], "side_name": np.where(values % 2, "long", "short"),
        "__symbol__": np.where(values % 3, "BTC/USDT", "ETH/USDT"), "__ts__": timestamp,
        "candidate_month": month, "score": values, "candidate_group_rows": 100.0 + values % 5,
        "score_ventile": np.where(values < rows / 2, "v18", "v19"), "candidate_group_size_bin": np.where(values % 2, "q1", "q2"),
        "transition_state": np.where(values % 2, "q0|q1|q2", "q1|q1|q1"),
        "liq": values / rows, "vol": (values % 17) / 17, "trend": np.sin(values / 10),
        "execution_gross_ev_12h": gross, "execution_cost_return": cost, "execution_net_ev_12h": gross - cost,
        "execution_exit_reason": np.where(values % 5 == 0, "full_sl", np.where(values % 7 == 0, "timeout", "trailing")),
    })


def test_stable_top_is_global_and_candidate_id_tie_stable() -> None:
    frame = _frame("2025-02", rows=10)
    frame["score"] = 1.0
    assert MODULE.stable_top(frame, "score", .2).candidate_id.tolist() == ["2025-02-0", "2025-02-1"]


def test_response_metrics_use_explicit_cost_and_exit_mix() -> None:
    frame = _frame("2025-02", rows=300)
    frame["exit_class"] = MODULE._exit_class(frame)
    metrics = MODULE.response_metrics(frame)
    assert metrics["cost_bps"] == pytest.approx(100.0)
    assert 0.0 <= metrics["full_stop_rate"] <= 1.0
    assert "favorable_gross_bps_given_opportunity" in metrics


def test_propensity_diagnosis_reconciles_and_reports_overlap() -> None:
    source = _frame("2025-02", rows=600)
    target = _frame("2025-03", shift=-.002, rows=600)
    for local in (source, target):
        local["exit_class"] = MODULE._exit_class(local)
    response, coverage, balance = MODULE.diagnose_pair(
        source, target, family="test", from_month="2025-02", to_month="2025-03",
        continuous=("liq", "vol", "trend"),
        categorical=("side_name", "__symbol__", "score_ventile", "candidate_group_size_bin", "transition_state"),
    )
    assert coverage.common_support_pass.iloc[0]
    assert not balance.empty
    assert np.abs(response.reconciliation_error).max() < 1e-10
    net = response.loc[response.metric.eq("net_ev_bps")].iloc[0]
    assert net.raw_all_delta == pytest.approx(-20.0)


def test_weak_tail_fails_closed_without_response_claim() -> None:
    source = _frame("2025-02", rows=20); target = _frame("2025-03", rows=20)
    for local in (source, target):
        local["exit_class"] = MODULE._exit_class(local)
    response, coverage, balance = MODULE.diagnose_pair(
        source, target, family="test", from_month="2025-02", to_month="2025-03",
        continuous=("liq", "vol", "trend"),
        categorical=("side_name", "__symbol__", "score_ventile", "candidate_group_size_bin", "transition_state"),
    )
    assert response.empty and balance.empty
    assert not coverage.common_support_pass.iloc[0]
