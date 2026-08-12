from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "report_strict_r3_r5_canonical_waterfall.py"
SPEC = importlib.util.spec_from_file_location("r5_waterfall", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _frame() -> pd.DataFrame:
    rows = 100
    return pd.DataFrame({
        "candidate_id": [f"c-{index:03d}" for index in range(rows)],
        "__decision_ts__": pd.date_range("2026-01-01", periods=rows, freq="h", tz="UTC"),
        "base_rank42": np.linspace(0.0, 1.0, rows),
        "conditional_consensus_rank": np.linspace(0.0, 1.0, rows),
        "upstream": np.linspace(0.0, 1.0, rows),
        "final_score": np.linspace(0.0, 1.0, rows),
        "causal_21d_side_expected_net_bps": np.linspace(-100.0, 200.0, rows),
        "trust_posterior_expected_bps": np.linspace(-120.0, 220.0, rows),
        "causal_21d_side_admitted_ge_50bps": [False] * 50 + [True] * 50,
        "trust_posterior_admitted_ge_50bps": [False] * 60 + [True] * 40,
        "policy_path_valid": True,
        "policy_gross_bps": np.linspace(-50.0, 250.0, rows),
        "policy_net_bps": np.linspace(-150.0, 150.0, rows),
        "h12_label_valid": True,
        "h12_tp6_sl4_gross_bps": np.linspace(-60.0, 240.0, rows),
        "h12_tp6_sl4_net_bps": np.linspace(-160.0, 140.0, rows),
    })


def test_waterfall_reports_both_outcome_contracts() -> None:
    metrics = MODULE._waterfall(_frame())
    assert set(metrics["outcome"]) == {
        "optimized_trailing_policy", "exact_h12_tp6_sl4",
    }


def test_waterfall_separates_diagnostic_tail_and_executable_admission() -> None:
    metrics = MODULE._waterfall(_frame())
    assert "retrospective_global_tail_diagnostic" in set(metrics["selection"])
    assert "executable_all_admitted" in set(metrics["selection"])
    posterior = metrics.loc[
        metrics["stage"].eq("r5_posterior_expected_net")
        & metrics["selection"].eq("executable_all_admitted")
        & metrics["period_scope"].eq("all")
        & metrics["outcome"].eq("optimized_trailing_policy")
    ]
    assert posterior.iloc[0]["selected_rows"] == 40


def test_top_tail_is_one_pooled_global_ranking() -> None:
    selected = MODULE._select_tail(_frame(), "final_score", 0.01)
    assert selected["candidate_id"].tolist() == ["c-099"]


def test_report_declares_outcomes_joined_after_selection() -> None:
    text = SCRIPT.read_text()
    assert '"outcomes_joined_after_selection": True' in text
    assert '"tail_interpretation": "retrospective global ranking diagnostic only"' in text

