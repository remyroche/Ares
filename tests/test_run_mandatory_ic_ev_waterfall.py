from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


RUNNER = Path(__file__).resolve().parents[1] / "scripts" / "run_mandatory_ic_ev_waterfall.py"
SPEC = importlib.util.spec_from_file_location("mandatory_ic_ev_waterfall", RUNNER)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _frame(months: tuple[str, ...] = ("2025-02", "2025-03")) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for month_number, month in enumerate(months):
        for index in range(80):
            gross = (index - 36 - month_number * 4) / 10_000
            cost = .01
            timestamp = pd.Timestamp(f"{month}-01T00:00:00Z") + pd.Timedelta(hours=index)
            rows.append({
                "candidate_id": f"{month}-{index}", "side_name": "long" if index % 2 else "short",
                "__symbol__": f"ASSET{index % 3}", "__ts__": timestamp, "candidate_month": month,
                "__first_touch_target_soft__": index / 80, "execution_mfe_return_12h": gross + .02,
                "execution_mae_return_12h": -.01, "execution_gross_ev_12h": gross,
                "execution_cost_return": cost, "execution_net_ev_12h": gross - cost,
                "execution_exit_class": ("timeout" if index % 4 == 0 else "adverse_exit"),
                "score_base_alpha": index / 80, "score_residual_expected_ev": index / 90,
                "score_direct_q25_bps": index * 1.0,
            })
    return pd.DataFrame(rows)


def test_stable_top_is_one_global_book_with_deterministic_ties() -> None:
    frame = _frame().iloc[:10].copy()
    frame["score_base_alpha"] = 1.0
    selected = MODULE.stable_top(frame, "score_base_alpha", .2)
    assert selected.candidate_id.tolist() == sorted(frame.candidate_id)[:2]


def test_tail_and_response_include_capture_exit_and_global_tail_ic() -> None:
    frame = _frame()
    tails = MODULE.tail_metrics(frame, "test", "score_base_alpha")
    pooled = tails.query("scope == 'pooled_global' and fraction == .1")
    assert set(pooled.candidate_month) == {"2025-02", "2025-03"}
    assert {"tail_net_rank_ic", "mfe_to_gross_capture_ratio", "exit_timeout_rate", "cvar05_net_bps"}.issubset(tails.columns)
    cells = MODULE.response_cells(frame, "test", "score_base_alpha")
    assert cells.query("scope == 'pooled_global'").score_ventile.nunique() == 20
    assert "conditional_positive_gross_bps" in cells


def test_rank_level_waterfall_reconciles_on_complete_side_cells() -> None:
    frame = _frame()
    result = MODULE.rank_level_waterfall(frame, "test", "score_base_alpha")
    assert not result.empty
    grouped = result.groupby(["from_month", "to_month"])
    for _, local in grouped:
        assert local.contribution_bps.sum() == pytest.approx(local.actual_net_delta_bps.iloc[0], abs=1e-10)
        assert local.common_rank_cell_mass_source.iloc[0] == pytest.approx(1.0)
        assert local.common_rank_cell_mass_target.iloc[0] == pytest.approx(1.0)


def test_day_block_bootstrap_is_frozen_selection_and_repeatable() -> None:
    frame = _frame()
    first = MODULE.bootstrap_tail_ci(frame, "test", "score_base_alpha", reps=20, seed=7)
    second = MODULE.bootstrap_tail_ci(frame, "test", "score_base_alpha", reps=20, seed=7)
    pd.testing.assert_frame_equal(first, second)
    assert set(first.kind) == {"monthly_tail_net", "adjacent_month_delta"}


def test_prepare_input_rejects_mapped_score_stream() -> None:
    frame = _frame()
    frame["score_mapped_bad"] = 0.0
    with pytest.raises(ValueError, match="mapped score"):
        MODULE.prepare_inputs(frame, _frame(), _frame())


def test_prepare_input_checks_exact_economics_identity() -> None:
    frame = _frame()
    frame.loc[0, "execution_net_ev_12h"] += .01
    with pytest.raises(ValueError, match="gross - explicit cost"):
        MODULE.prepare_inputs(frame, _frame(), _frame())
