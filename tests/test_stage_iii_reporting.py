from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from extreme_price_movements.stage_iii_reporting import (
    StageIIIReportingConfig,
    StageIIIReportingError,
    build_stage_iii_report_tables,
)


def _frame() -> pd.DataFrame:
    timestamp = pd.to_datetime(
        ["2024-01-01", "2024-01-02", "2024-01-08", "2024-02-01",
         "2024-02-02", "2024-02-08", "2024-02-09", "2024-02-15"], utc=True,
    )
    net = np.asarray([-80, -60, -40, 20, 40, 60, 80, 100], dtype=float)
    return pd.DataFrame({
        "candidate_id": [f"c{i}" for i in range(8)],
        "symbol": ["BTC", "ETH", "SOL", "BTC", "ETH", "SOL", "BTC", "ETH"],
        "decision_ts": timestamp,
        "side_name": ["long", "short"] * 4,
        "exact_net_bps": net,
        "exact_gross_bps": net + 100.0,
        "base_score_bps": np.asarray([1, 2, 3, 4, 5, 6, 7, 8], dtype=float),
        "meta_score_bps": np.asarray([1, 0, 2, 8, 3, 7, 4, 6], dtype=float),
        "causal_21d_admitted": [True, False, True, True, False, True, True, True],
        "hit_rate_surprise_3d": np.linspace(-0.2, 0.2, 8),
        "hit_rate_surprise_7d": np.linspace(-0.1, 0.3, 8),
        "hit_rate_surprise_14d": np.linspace(-0.3, 0.1, 8),
        "hit_rate_surprise_support_3d": np.arange(1, 9),
        "hit_rate_surprise_support_7d": np.arange(2, 10),
        "hit_rate_surprise_support_14d": np.arange(3, 11),
    })


def _config() -> StageIIIReportingConfig:
    return StageIIIReportingConfig(top_fractions=(0.25, 0.50))


def test_tails_are_selected_once_globally_then_attributed_without_local_reranking() -> None:
    tables = build_stage_iii_report_tables(
        _frame(), score_columns={"base": "base_score_bps", "meta": "meta_score_bps"},
        config=_config(),
    )
    summary = tables.tail_summary
    assert set(summary.layer) == {"base", "meta"}
    assert set(summary.admission_scope) == {"without_21d", "with_21d"}
    base_top = summary.loc[
        summary.layer.eq("base") & summary.admission_scope.eq("without_21d")
        & summary.top_fraction.eq(0.25)
    ].iloc[0]
    assert base_top.selected_rows == 2
    assert base_top.net_bps_per_trade == 90.0
    assert base_top.cost_bps_per_trade == 100.0

    contribution = tables.selected_attribution.loc[
        tables.selected_attribution.layer.eq("base")
        & tables.selected_attribution.admission_scope.eq("without_21d")
        & tables.selected_attribution.top_fraction.eq(0.25)
        & tables.selected_attribution.scope.eq("month")
    ]
    # Both globally selected rows are in February. January is not locally
    # re-ranked to manufacture a contribution.
    assert contribution.month.tolist() == ["2024-02"]
    assert int(contribution.selected_rows.sum()) == 2


def test_reports_side_week_concentration_signed_residuals_and_signed_surprise() -> None:
    tables = build_stage_iii_report_tables(
        _frame(), score_columns={"base": "base_score_bps"}, config=_config(),
    )
    attribution = tables.selected_attribution
    assert {"week", "week_side", "month_side"}.issubset(set(attribution.scope))
    assert set(tables.residual_diagnostics.side) == {"__all__", "long", "short"}
    assert tables.residual_diagnostics.signed_residual_mean_bps.notna().all()
    concentration = tables.time_concentration
    assert concentration.max_day_share.between(0, 1).all()
    assert concentration.max_symbol_share.between(0, 1).all()
    assert set(tables.hit_surprise.horizon) == {"3d", "7d", "14d"}
    assert (tables.hit_surprise.effective_support_sum > 0).all()


def test_residual_autocorrelation_uses_chronological_not_score_order() -> None:
    frame = _frame()
    # Scores make the global top half appear in the score order c2, c0, c3,
    # c1, while their event-time order is c0, c1, c2, c3.  The signed errors
    # are deliberately non-symmetric so the two autocorrelations differ.
    frame["base_score_bps"] = [10.0, 7.0, 12.0, 8.0, 1.0, 2.0, 3.0, 4.0]
    frame["exact_net_bps"] = [11.0, 11.0, 15.0, 13.0, -99.0, -98.0, -97.0, -96.0]
    frame["exact_gross_bps"] = frame["exact_net_bps"] + 100.0
    tables = build_stage_iii_report_tables(
        frame, score_columns={"base": "base_score_bps"}, config=_config(),
    )
    observed = tables.residual_diagnostics.loc[
        tables.residual_diagnostics.layer.eq("base")
        & tables.residual_diagnostics.admission_scope.eq("without_21d")
        & tables.residual_diagnostics.top_fraction.eq(0.50)
        & tables.residual_diagnostics.side.eq("__all__")
    ].iloc[0]
    # c0..c3 chronological residuals are [1, 4, 3, 5].
    expected = float(np.corrcoef(np.asarray([1.0, 4.0, 3.0]), np.asarray([4.0, 3.0, 5.0]))[0, 1])
    assert observed.signed_residual_lag1_autocorrelation == pytest.approx(expected)


@pytest.mark.parametrize("bad", ["False", np.nan, 2])
def test_admission_truthiness_is_rejected(bad: object) -> None:
    frame = _frame()
    frame["causal_21d_admitted"] = bad
    with pytest.raises(StageIIIReportingError, match="bool or integer 0/1"):
        build_stage_iii_report_tables(
            frame, score_columns={"base": "base_score_bps"}, config=_config(),
        )


def test_cost_and_surprise_contracts_fail_closed() -> None:
    bad_cost = _frame()
    bad_cost.loc[0, "exact_gross_bps"] += 1.0
    with pytest.raises(StageIIIReportingError, match="declared cost"):
        build_stage_iii_report_tables(
            bad_cost, score_columns={"base": "base_score_bps"}, config=_config(),
        )
    with pytest.raises(StageIIIReportingError, match="hit-surprise"):
        build_stage_iii_report_tables(
            _frame().drop(columns="hit_rate_surprise_7d"),
            score_columns={"base": "base_score_bps"}, config=_config(),
        )
