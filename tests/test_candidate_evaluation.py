import numpy as np
import pandas as pd

from extreme_price_movements.candidate_evaluation import TailGate, evaluate_global_book, paired_day_block_bootstrap, stable_global_top_k, tail_gates


def _frame() -> pd.DataFrame:
    return pd.DataFrame({
        "candidate_id": ["z", "a", "b", "c", "d", "e", "f", "g", "h", "i"],
        "__ts__": pd.date_range("2025-01-01", periods=10, freq="D", tz="UTC"),
        "side_name": ["long"] * 5 + ["short"] * 5,
        "score": [10., 10., 8., 7., 6., 5., 4., 3., 2., 1.],
        "execution_net_ev_12h": [.03, -.01, .02, -.01, -.01, -.01, -.01, -.01, -.01, -.01],
        "execution_gross_ev_12h": [.04, 0., .03, 0., 0., 0., 0., 0., 0., 0.],
        "execution_cost_return": [.01] * 10,
        "regime": ["trend"] * 5 + ["range"] * 5,
        "liquidity_score": np.arange(10, dtype=float), "hurdle_probability": np.linspace(.1, .9, 10),
    })


def test_global_tails_are_pooled_deterministic_and_have_optional_attribution() -> None:
    frame = _frame()
    assert stable_global_top_k(frame, "score", .10).candidate_id.tolist() == ["a"]
    tails, diagnostics = evaluate_global_book(frame, score_column="score", net_column="execution_net_ev_12h", net_unit="return", gross_column="execution_gross_ev_12h", gross_unit="return", cost_column="execution_cost_return", cost_unit="return", regime_column="regime", liquidity_column="liquidity_score", hurdle_column="hurdle_probability")
    assert tails.top_fraction.tolist() == [.01, .05, .10, .20]
    assert tails.loc[tails.top_fraction.eq(.10), "net_bps"].iloc[0] == -100.0
    assert set(diagnostics) >= {"side", "month", "regime", "cost", "liquidity", "hurdle"}


def test_tail_reversal_and_side_gate_are_diagnostic_not_selection_changes() -> None:
    tails, diag = evaluate_global_book(_frame(), score_column="score", net_column="execution_net_ev_12h", net_unit="return")
    gates = tail_gates(tails, diag["side"], gate=TailGate(min_side_rows=1))
    assert bool(gates.promotion_authorized.iloc[0]) is False
    assert bool(gates.top_tail_reversal_detected.iloc[0]) is True
    assert bool(gates.side_gate_pass.iloc[0]) is False


def test_paired_day_block_bootstrap_is_deterministic_and_preserves_frozen_books() -> None:
    baseline = _frame().iloc[:5].copy(); challenger = baseline.copy()
    challenger.execution_net_ev_12h += .01
    first = paired_day_block_bootstrap(baseline, challenger, net_column="execution_net_ev_12h", net_unit="return", block_days=2, replicates=100, seed=7)
    second = paired_day_block_bootstrap(baseline, challenger, net_column="execution_net_ev_12h", net_unit="return", block_days=2, replicates=100, seed=7)
    assert first == second
    assert first["mean_daily_net_delta_bps"] > 0
