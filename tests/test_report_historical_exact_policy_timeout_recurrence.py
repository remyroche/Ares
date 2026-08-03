from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.report_historical_exact_policy_timeout_recurrence import (
    OOF_FLAG,
    SCORE_COLUMN,
    SOURCE_SCORE_COLUMN,
    build_frozen_historical_population,
    recurrence_metrics,
)


def _labels(horizon: int) -> pd.DataFrame:
    decision = pd.date_range("2026-05-01 01:00", periods=6, freq="10D", tz="UTC")
    net_12 = [-0.03, -0.02, 0.01, -0.01, 0.02, -0.04]
    net_24 = [0.02, -0.03, 0.01, 0.01, 0.03, -0.02]
    net = net_12 if horizon == 12 else net_24
    reason = (
        ["timeout", "timeout", "trailing", "timeout", "trailing", "timeout"]
        if horizon == 12
        else ["trailing", "full_sl", "trailing", "trailing", "trailing", "timeout"]
    )
    hold = [float(horizon), 4.0, 3.0, float(horizon), 5.0, float(horizon)]
    return pd.DataFrame(
        {
            "candidate_id": [f"c{i}" for i in range(6)],
            "__ts__": decision - pd.Timedelta(hours=1),
            "__symbol__": ["A", "B", "C", "D", "E", "F"],
            "side_name": ["long", "long", "short", "short", "long", "short"],
            "execution_decision_utc": decision,
            "policy_archetype": ["trend", "risk", "trend", "range", "trend", "range"],
            "execution_geometry_key": ["parent"] * 6,
            "execution_geometry_source": ["side_parent"] * 6,
            "execution_gross_ev_12h": np.asarray(net) + 0.01,
            "execution_cost_return": [0.01] * 6,
            "execution_net_ev_12h": net,
            "execution_exit_reason": reason,
            "execution_exit_hour": hold,
            "execution_mfe_return_12h": [0.04] * 6,
            "execution_mae_return_12h": [-0.02] * 6,
            "execution_entry_price": [100.0] * 6,
            "execution_exit_price": [101.0] * 6,
            "execution_expected_spread_bps": [50.0] * 6,
            "execution_entry_half_spread_bps": [25.0] * 6,
            "execution_exit_half_spread_bps": [25.0] * 6,
            "execution_label_end_utc": decision + pd.Timedelta(hours=horizon),
            "execution_label_available_at": decision + pd.Timedelta(hours=horizon),
        }
    )


def _scores() -> pd.DataFrame:
    frame = _labels(12).loc[
        :,
        [
            "candidate_id",
            "__ts__",
            "__symbol__",
            "side_name",
            "execution_decision_utc",
            "execution_net_ev_12h",
            "execution_gross_ev_12h",
            "execution_cost_return",
        ],
    ].copy()
    frame[SOURCE_SCORE_COLUMN] = [0.05, 0.04, 0.03, 0.02, 0.01, -0.01]
    frame[OOF_FLAG] = True
    frame["execution_label_end_utc"] = (
        frame["execution_decision_utc"] + pd.Timedelta(hours=12)
    )
    frame[SCORE_COLUMN] = frame[SOURCE_SCORE_COLUMN]
    return frame


def test_historical_pair_freezes_oof_score_cohort_before_24h_join() -> None:
    paired = build_frozen_historical_population(
        _scores(), _labels(12), _labels(24), top_k_fraction=0.40
    )
    assert len(paired) == 6
    assert paired["global_top10_capacity_member"].sum() == 3
    assert set(
        paired.loc[paired["global_top10_capacity_member"], "candidate_id"]
    ) == {"c0", "c1", "c2"}
    assert paired["snx_style_late_recovery"].sum() == 2
    assert set(
        paired.loc[paired["snx_style_late_recovery"], "candidate_id"]
    ) == {"c0", "c3"}


def test_recurrence_metrics_break_out_side_regime_and_exit_transition() -> None:
    paired = build_frozen_historical_population(
        _scores(), _labels(12), _labels(24), top_k_fraction=0.40
    )
    metrics = recurrence_metrics(paired)
    assert {
        "overall",
        "month",
        "side",
        "month_side",
        "policy_regime",
        "exit_transition",
        "month_side_exit",
    } == set(metrics["scope"])
    overall = metrics.loc[
        metrics["cohort"].eq("full_population") & metrics["scope"].eq("overall")
    ].iloc[0]
    assert overall["snx_style_late_recovery_rows"] == 2
    assert overall["snx_style_late_recovery_unique_assets"] == 2
    assert overall["snx_style_late_recovery_symbol_days"] == 2
    assert overall["loss_to_win_rows"] == 2
    transition = metrics.loc[
        metrics["cohort"].eq("full_population")
        & metrics["scope"].eq("exit_transition")
        & metrics["exit_transition"].eq("timeout -> trailing")
    ].iloc[0]
    assert transition["snx_style_late_recovery_rows"] == 2


def test_non_oof_mapped_score_is_excluded() -> None:
    scores = _scores()
    scores.loc[0, OOF_FLAG] = False
    paired = build_frozen_historical_population(
        scores, _labels(12), _labels(24), top_k_fraction=0.40
    )
    assert "c0" not in set(paired["candidate_id"])
