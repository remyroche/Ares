import pandas as pd

from extreme_price_movements.base_oof_transition_diagnostic import build_transition_diagnostic


def _scores() -> pd.DataFrame:
    rows = []
    for timestamp in pd.date_range("2025-02-01", periods=74, freq="h", tz="UTC"):
        for side in ("long", "short"):
            for number in range(5):
                rows.append({
                    "candidate_id": f"{timestamp.isoformat()}-{side}-{number}", "side_name": side,
                    "__symbol__": f"S{number}", "__ts__": timestamp, "base_oof_score": 5 - number,
                    "__first_touch_target_soft__": number / 4, "execution_net_ev_12h": (number - 2) / 100,
                })
    return pd.DataFrame(rows)


def test_event_diagnostic_has_all_phases_sides_and_top40_coverage() -> None:
    windows = pd.DataFrame({"transition_event_id": ["e1"], "transition_window_start_utc": [pd.Timestamp("2025-02-02T00:00Z")], "transition_window_end_utc": [pd.Timestamp("2025-02-02T23:00Z")], "transition_active_hours": [2]})
    active = pd.DataFrame({"source_utc": pd.date_range("2025-02-01", periods=74, freq="h", tz="UTC"), "target__event_id": [None] * 24 + ["e1"] * 24 + [None] * 26, "target__transition_active": [0] * 24 + [1, 0, 1] + [0] * 47})
    coverage, metrics, summary = build_transition_diagnostic(_scores(), windows, active)
    assert coverage.loc[0, "window_complete"]
    assert coverage.loc[0, "active_top40_covered"]
    assert set(metrics["phase"]) == {"before_24h", "during_window", "after_24h", "active_hours"}
    assert len(metrics) == 12
    assert summary["later_health_active_risk_readiness"]["sufficient_base_score_coverage"]
