from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.report_execution_ev_timeout_ablation import (
    _validate_shared_label_lineage,
    exit_mix_metrics,
    horizon_metrics,
    pair_horizon_labels,
    paired_delta_metrics,
)
from scripts.score_execution_ev_forward_population import apply_global_admission


def _labels(horizon_hours: int) -> pd.DataFrame:
    decision = pd.date_range("2026-07-20 01:00", periods=4, freq="h", tz="UTC")
    if horizon_hours == 12:
        net = [-0.02, 0.01, -0.01, 0.03]
        gross = [-0.01, 0.02, 0.00, 0.04]
        reason = ["timeout", "trailing", "timeout", "trailing"]
        holding = [12.0, 3.0, 12.0, 4.0]
    else:
        net = [0.01, -0.01, -0.02, 0.04]
        gross = [0.02, 0.00, -0.01, 0.05]
        reason = ["trailing", "trailing", "timeout", "trailing"]
        holding = [15.0, 3.0, 24.0, 4.0]
    return pd.DataFrame(
        {
            "candidate_id": [f"c{i}" for i in range(4)],
            "__ts__": decision - pd.Timedelta(hours=1),
            "__symbol__": ["A", "B", "C", "D"],
            "side_name": ["long", "long", "short", "short"],
            "execution_decision_utc": decision,
            "policy_archetype": ["side_parent"] * 4,
            "execution_geometry_key": ["parent"] * 4,
            "execution_geometry_source": ["side_parent"] * 4,
            "execution_gross_ev_12h": gross,
            "execution_cost_return": [0.01] * 4,
            "execution_net_ev_12h": net,
            "execution_exit_reason": reason,
            "execution_exit_hour": holding,
            "execution_mfe_return_12h": [0.03, 0.04, 0.01, 0.06],
            "execution_mae_return_12h": [-0.02, -0.01, -0.03, -0.01],
            "execution_entry_price": [100.0] * 4,
            "execution_exit_price": [101.0] * 4,
            "execution_expected_spread_bps": [50.0] * 4,
            "execution_entry_half_spread_bps": [25.0] * 4,
            "execution_exit_half_spread_bps": [25.0] * 4,
            "execution_label_end_utc": decision
            + pd.Timedelta(hours=horizon_hours),
            "execution_label_available_at": decision
            + pd.Timedelta(hours=horizon_hours),
        }
    )


def _scored_12h() -> pd.DataFrame:
    frame = _labels(12)
    frame["mapped_execution_ev"] = [0.02, 0.01, -0.01, -0.02]
    return apply_global_admission(frame, top_k_fraction=0.5)


def test_pairing_preserves_frozen_cohorts_and_reports_paired_flips() -> None:
    scored = _scored_12h()
    original_members = scored.set_index("candidate_id")[
        "global_top10_capacity_member"
    ].to_dict()
    paired = pair_horizon_labels(scored, _labels(24))
    assert (
        paired.set_index("candidate_id")["global_top10_capacity_member"].to_dict()
        == original_members
    )
    assert paired["loss_to_win_12h_to_24h"].sum() == 1
    assert paired["win_to_loss_12h_to_24h"].sum() == 1
    assert np.allclose(
        paired["paired_delta_net_24h_minus_12h"],
        [0.03, -0.02, -0.01, 0.01],
    )


def test_metrics_include_all_frozen_cohorts_sides_days_and_exit_mix() -> None:
    paired = pair_horizon_labels(_scored_12h(), _labels(24))
    horizons = horizon_metrics(paired)
    deltas = paired_delta_metrics(paired)
    exits = exit_mix_metrics(paired)
    assert {"12h", "24h"} == set(horizons["horizon"])
    assert {"overall", "side", "day", "day_side"} == set(horizons["scope"])
    assert {
        "global_top10",
        "admitted_gt_0bps",
        "admitted_gt_25bps",
        "admitted_gt_50bps",
    }.issubset(deltas["cohort"])
    overall = deltas.loc[
        deltas["cohort"].eq("full_population") & deltas["scope"].eq("overall")
    ].iloc[0]
    assert overall["loss_to_win_rows"] == 1
    assert overall["win_to_loss_rows"] == 1
    assert {"timeout", "trailing"} == set(exits["exit_reason"])


def test_shared_lineage_rejects_any_non_timeout_difference() -> None:
    common_source = {
        "candidates_sha256": "a",
        "context_sha256": "b",
        "path_targets_sha256": "c",
        "policy_sha256": "d",
    }
    common_contract = {
        "geometry_scope": "side_parent",
        "policy_pathway_id": "p",
        "replay_timeframe": "1m",
        "simulator": "s",
        "source_policy_sha256": "d",
        "trailing_activation_curve": "total_mfe",
    }
    accounting = {"same": True}
    left = {
        "source": common_source,
        "exit_policy_contract": common_contract,
        "accounting": accounting,
    }
    right = {
        "source": dict(common_source),
        "exit_policy_contract": dict(common_contract),
        "accounting": accounting,
    }
    _validate_shared_label_lineage(left, right)
    right["source"]["context_sha256"] = "different"
    try:
        _validate_shared_label_lineage(left, right)
    except ValueError as exc:
        assert "context_sha256" in str(exc)
    else:
        raise AssertionError("different context lineage was accepted")
