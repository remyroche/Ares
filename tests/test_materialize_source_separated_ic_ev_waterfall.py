from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


RUNNER = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "materialize_source_separated_ic_ev_waterfall.py"
)
SPEC = importlib.util.spec_from_file_location("source_separated_waterfall", RUNNER)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _frame(*, months: tuple[str, ...] = ("2025-02", "2025-03")) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for month_number, month in enumerate(months):
        for index in range(100):
            gross = (index - 45 - 10 * month_number) / 10_000.0
            cost = 0.01 + index / 1_000_000.0
            rows.append(
                {
                    "candidate_id": f"{month}-{index:03d}",
                    "__ts__": pd.Timestamp(f"{month}-01", tz="UTC")
                    + pd.Timedelta(hours=index),
                    "candidate_month": month,
                    "side_name": "long" if index % 2 == 0 else "short",
                    "__symbol__": f"asset-{index % 4}",
                    "source_family": "synthetic",
                    "score_base_alpha": index / 100.0,
                    "__first_touch_target_soft__": index / 100.0,
                    "execution_mfe_return_12h": gross + 0.02,
                    "execution_gross_ev_12h": gross,
                    "execution_cost_return": cost,
                    "execution_net_ev_12h": gross - cost,
                    "opportunity_gross_above_cost_0bps": gross > cost,
                }
            )
    return pd.DataFrame(rows)


def test_score_columns_reject_mapping_fields() -> None:
    frame = _frame()
    assert MODULE.score_columns(frame) == ["score_base_alpha"]
    frame["score_mapped_direct"] = frame.score_base_alpha
    with pytest.raises(ValueError, match="mapped"):
        MODULE.score_columns(frame)
    assert MODULE.score_role("score_base_alpha") == "raw_base_alpha"
    assert (
        MODULE.score_role("score_residual_expected_ev")
        == "upstream_expected_ev_stream"
    )


def test_full_ic_is_month_and_side_scoped_and_omits_absent_base_target() -> None:
    result = MODULE.full_ic(_frame(), source_family="synthetic", score="score_base_alpha")
    assert set(result.scope) == {"pooled_global", "side_long", "side_short"}
    assert set(result.target) == set(MODULE.TARGETS)
    assert np.allclose(
        result.loc[result.target.eq("legacy_native24_base_target"), "rank_ic"].to_numpy(float),
        1.0,
    )
    no_target = _frame().drop(columns="__first_touch_target_soft__")
    result = MODULE.full_ic(no_target, source_family="synthetic", score="score_base_alpha")
    assert "legacy_native24_base_target" not in set(result.target)


def test_tail_metrics_uses_deterministic_global_book_and_explicit_waterfall_gaps() -> None:
    frame = _frame(months=("2025-02",))
    frame.loc[frame.index[-2:], "score_base_alpha"] = 0.99
    selected = MODULE.stable_top(frame, "score_base_alpha", 0.01)
    assert selected.candidate_id.iloc[0] == "2025-02-098"
    tails = MODULE.tail_metrics(frame, source_family="synthetic", score="score_base_alpha")
    top1 = tails.loc[(tails.scope == "pooled_global") & np.isclose(tails.fraction, .01)].iloc[0]
    assert top1.selected_rows == 1
    assert top1.gross_to_net_explicit_cost_gap_bps == pytest.approx(top1.mean_cost_bps)
    assert top1.mfe_ceiling_to_gross_gap_bps == pytest.approx(200.0)
    assert np.isfinite(top1.cvar05_net_bps)


def test_response_and_cutoff_diagnostics_make_non_monotonicity_and_ties_observable() -> None:
    frame = _frame(months=("2025-02",))
    # Exact tie at the top has label-dependent best/worst bounds, while the
    # deterministic candidate ID selection remains independent of labels.
    frame.loc[frame.index[-2:], "score_base_alpha"] = 1.0
    frame.loc[frame.index[-2], "execution_net_ev_12h"] = 0.10
    frame.loc[frame.index[-1], "execution_net_ev_12h"] = -0.30
    cells, summary = MODULE.response_20bin(frame, source_family="synthetic", score="score_base_alpha")
    assert cells.score_rank_bin.nunique() == 20
    assert "net_monotonicity_violations" in summary
    ties = MODULE.cutoff_ties(frame, source_family="synthetic", score="score_base_alpha")
    top1 = ties.loc[(ties.scope == "pooled_global") & np.isclose(ties.fraction, .01)].iloc[0]
    assert top1.candidate_id_tie_break_used
    assert top1.tie_sensitivity_bps > 0


def test_fixed_composition_is_exactly_zero_for_identical_adjacent_months() -> None:
    first = _frame(months=("2025-02",))
    second = first.copy()
    second["candidate_month"] = "2025-03"
    second["candidate_id"] = "2025-03-" + second.candidate_id.str.rsplit("-", n=1).str[-1]
    result = MODULE.fixed_composition(pd.concat([first, second], ignore_index=True), source_family="synthetic", score="score_base_alpha")
    assert set(result.fraction) == set(MODULE.TOP_FRACTIONS)
    assert np.allclose(result.composition_effect_bps.to_numpy(float), 0.0)
    assert np.allclose(result.within_cell_payoff_effect_bps.to_numpy(float), 0.0)
    for component in (
        "net_positive_alias_pp",
        "mfe_ceiling_bps",
        "deployed_gross_bps",
        "explicit_cost_bps",
        "exact_net_bps",
    ):
        assert np.allclose(
            result[f"composition_effect_{component}"].to_numpy(float),
            0.0,
        )
        assert np.allclose(
            result[f"within_cell_effect_{component}"].to_numpy(float),
            0.0,
        )


def test_validate_source_requires_one_unique_raw_source_family() -> None:
    frame = _frame(months=("2025-02",))
    record = {"source_family": "synthetic"}
    MODULE.validate_source(frame, record)
    bad = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        MODULE.validate_source(bad, record)
    bad = frame.copy()
    bad.loc[bad.index[0], "execution_net_ev_12h"] += 0.001
    with pytest.raises(ValueError, match="waterfall identity"):
        MODULE.validate_source(bad, record)
    bad = frame.copy()
    bad.loc[bad.index[0], "opportunity_gross_above_cost_0bps"] = True
    with pytest.raises(ValueError, match="net-positive alias"):
        MODULE.validate_source(bad, record)
