from __future__ import annotations

import sys

import pandas as pd

from scripts import report_strict_r3_phase_h1_native_completion as report


def test_phase_report_includes_native_map_hourly_pass_breakdown(tmp_path, monkeypatch) -> None:
    root = tmp_path / "chain"
    pooled = root / "pooled_four_phase_native"
    pooled.mkdir(parents=True)
    timestamp = pd.Timestamp("2026-05-01T00:15:00Z")
    hourly = pd.DataFrame({
        "phase_minutes": [15, 15],
        "hour": [timestamp, timestamp + pd.Timedelta(hours=1)],
        "current_routed_rows": [12, 12],
        "current_mapper_pass_rows": [3, 0],
        "bcf_mapper_pass_rows": [2, 1],
        "dual_admitted_target_free_rows": [2, 0],
        "dual_admitted_valid_outcome_rows": [2, 0],
        "portfolio_accepted_rows": [1, 0],
        "accepted_net_ev_bps": [120.0, float("nan")],
        "accepted_net_sum_bps": [120.0, 0.0],
    })
    summary = pd.DataFrame({
        "phase_minutes": [15],
        "decision_hours": [2],
        "current_routed_rows": [24],
        "current_mapper_pass_rows": [3],
        "bcf_mapper_pass_rows": [3],
        "dual_admitted_target_free_rows": [2],
        "dual_admitted_valid_outcome_rows": [2],
        "portfolio_accepted_rows": [1],
        "accepted_net_sum_bps": [120.0],
        "accepted_net_ev_bps": [120.0],
    })
    decisions = pd.DataFrame({
        "candidate_id": ["a"],
        "accepted": [True],
        "phase_minutes": [15],
        "timestamp": [timestamp],
        "position_net_return": [0.012],
    })
    portfolio = pd.DataFrame([{
        "accepted_rows": 1,
        "realised_rows": 1,
        "net_ev_bps_per_realised_trade": 120.0,
        "net_sum_bps_realised": 120.0,
        "max_drawdown": -0.02,
        "worst_month_bps": 120.0,
        "worst_week_bps": 120.0,
    }])
    hourly.to_parquet(pooled / "phase_hourly_admissions.parquet", index=False)
    summary.to_parquet(pooled / "phase_hourly_admission_summary.parquet", index=False)
    decisions.to_parquet(pooled / "portfolio_decisions.parquet", index=False)
    portfolio.to_parquet(pooled / "portfolio_metrics.parquet", index=False)
    (pooled / "run_manifest.json").write_text("{}\n")

    monkeypatch.setattr(sys, "argv", ["report", "--chain-root", str(root)])
    report.main()

    text = (root / "REPORT.md").read_text()
    assert "Hours with current pass" in text
    assert "Hours with BCF pass" in text
    assert "Current pass p90/hour" in text
    assert "BCF pass p90/hour" in text
    assert "| 15 | 2 | 24 | 3 | 1.50 | 1.50 |" in text
