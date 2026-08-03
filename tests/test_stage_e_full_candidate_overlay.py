from pathlib import Path

import pandas as pd


ROOT = Path("data_perp/artifacts/stage_e_full_candidate_overlay_20260731_v1")


def _frame() -> pd.DataFrame:
    return pd.read_parquet(ROOT / "stage_e_full_candidate_overlay.parquet")


def test_full_overlay_keeps_entry_population_identical() -> None:
    frame = _frame()
    assert len(frame) == frame.candidate_id.nunique()
    assert frame.p0_net_bps.notna().all() and frame.p1_net_bps.notna().all()


def test_non_clear_candidates_follow_frozen_policy() -> None:
    frame = _frame(); rows = frame.loc[~frame.overlay_action_eligible]
    assert (rows.p0_gross_bps == rows.p1_gross_bps).all()
    assert (rows.p0_cost_bps == rows.p1_cost_bps).all()
    assert (rows.p0_net_bps == rows.p1_net_bps).all()


def test_only_first_clear_action_is_changed() -> None:
    frame = _frame(); acted = frame.loc[frame.overlay_action_eligible]
    assert acted.postcost_h0_event.eq("clear_cost_first").all()
    assert acted.action_decision_ts.notna().all()
    assert frame.loc[~frame.overlay_action_eligible, "action"].isna().all()


def test_no_portfolio_or_sizing_logic_is_invoked() -> None:
    source = Path("scripts/run_stage_e_full_candidate_overlay.py").read_text()
    assert all(token not in source for token in ("simple_policy_optimiser(", "position_size", "concurrency_limit", "portfolio_limit"))
