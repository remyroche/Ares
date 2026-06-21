import pandas as pd

from extreme_price_movements.late_window_metrics import (
    compute_late_window_hit_rate_summary,
    flatten_late_window_summary,
)


def test_late_window_metrics_detect_bad_tail_windows():
    n = 80
    summary = compute_late_window_hit_rate_summary(
        timestamps=pd.date_range("2026-01-01", periods=n, freq="D", tz="UTC"),
        actual_hit=[1.0] * 60 + [0.0] * 20,
        expected_probability=[0.60] * n,
        pnl=[0.01] * 60 + [-0.01] * 20,
        late_days=30,
        min_rows_per_day=1,
    )

    assert summary["status"] == "ok"
    assert summary["windows"]["3d"]["bad_window_count"] > 0
    assert summary["windows"]["5d"]["bad_window_count"] > 0
    assert summary["windows"]["3d"]["worst"]["hit_rate_delta"] < 0.0
    assert summary["windows"]["3d"]["worst"]["hit_rate_surprise_z"] < 0.0

    flat = flatten_late_window_summary(summary)
    assert flat["late_window_3d_bad_window_count"] == summary["windows"]["3d"][
        "bad_window_count"
    ]
    assert "late_window_5d_worst_hit_rate_surprise_z" in flat
