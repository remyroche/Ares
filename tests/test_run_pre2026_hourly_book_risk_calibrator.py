import importlib.util
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("hourly_book", ROOT / "scripts" / "run_pre2026_hourly_book_risk_calibrator.py")
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


def test_hourly_book_is_one_row_per_hour_and_uses_selected_count_weight():
    ts = pd.Timestamp("2025-05-01T00:00:00Z")
    rows = []
    for i, selected in enumerate([True, True, False]):
        row = {"candidate_id": f"x{i}", "__ts__": ts, "era": "2025_mayjun", "side_name": "long" if i != 1 else "short",
               "execution_net_ev_12h": [0.02, -0.01, 0.03][i], "residual_selected_global_top10": selected,
               "bocpd_regime_available": True, "lgbm_transition_available": True, "trajectory_available": True,
               "base_score": 0.01 + i, "residual_score": 0.02 + i, "residual_minus_base": 0.01}
        for feature in MOD.REGIME + MOD.TRANSITION + MOD.TRAJECTORY:
            row[feature] = 0.5
        rows.append(row)
    frame = pd.DataFrame(rows)
    result = MOD.hourly_book(frame, "regime")
    assert len(result) == 1
    assert result.loc[0, "selected_count"] == 2
    assert result.loc[0, "candidate_count"] == 3
    assert result.loc[0, "book_mean_net_ev_if_selected"] == 0.005
    assert result.loc[0, "book_failure_rate_if_selected"] == 0.5
    assert result.loc[0, "book_opportunity"] == 1


def test_hourly_book_keeps_zero_opportunity_hour():
    ts = pd.Timestamp("2025-05-01T01:00:00Z")
    rows = []
    for i in range(2):
        row = {"candidate_id": f"z{i}", "__ts__": ts, "era": "2025_mayjun", "side_name": "long",
               "execution_net_ev_12h": 0.01, "residual_selected_global_top10": False,
               "bocpd_regime_available": True, "lgbm_transition_available": True, "trajectory_available": True,
               "base_score": 0.01, "residual_score": 0.02, "residual_minus_base": 0.01}
        for feature in MOD.REGIME + MOD.TRANSITION + MOD.TRAJECTORY:
            row[feature] = 0.5
        rows.append(row)
    result = MOD.hourly_book(pd.DataFrame(rows), "regime")
    assert len(result) == 1
    assert result.loc[0, "selected_count"] == 0
    assert result.loc[0, "book_opportunity"] == 0
    assert pd.isna(result.loc[0, "book_mean_net_ev_if_selected"])


def test_gamma_zero_is_exact_raw_residual_book():
    frame = pd.DataFrame({"candidate_id": ["b", "a", "c"], "era": ["e"] * 3,
                          "__ts__": pd.to_datetime(["2025-01-01T00:00:00Z"] * 3), "__symbol__": ["X", "Y", "X"],
                          "side_name": ["long", "short", "long"], "residual_score": [0.1, 0.2, 0.0],
                          "context_hour_prediction": [3.0, -2.0, 1.0], "score_only_hour_prediction": [-4.0, 2.0, 1.0],
                          "execution_net_ev_12h": [0.01, 0.02, -0.01]})
    result = MOD.policy_metrics(frame, "regime", 0.0)
    assert result.loc[0, "turnover_vs_residual_control"] == 0.0
