import pandas as pd

from extreme_price_movements.funnel_selection import global_tail_metrics, monthly_stability, select_winner


def test_global_tail_and_monthly_stability_are_global():
    frame = pd.DataFrame({
        "candidate_id": ["a", "b", "c", "d"],
        "__ts__": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-02-01", "2024-02-02"], utc=True),
        "score": [4., 3., 2., 1.], "net_bps": [100., 50., 0., -50.], "gross_bps": [200., 150., 100., 50.],
    })
    metrics = global_tail_metrics(frame, tails=(.5,))
    assert metrics["top50_net_bps"] == 75.
    stability = monthly_stability(frame, tail=.5)
    assert stability["month_count"] == 2
    assert stability["month_worst_net_bps"] == 0.


def test_selection_applies_tie_stability_then_top1():
    table = pd.DataFrame([
        {"arm": "a", "top5_net_bps": 10., "month_std_net_bps": 20., "month_worst_net_bps": -50., "top1_net_bps": 1.},
        {"arm": "b", "top5_net_bps": 9.5, "month_std_net_bps": 5., "month_worst_net_bps": -10., "top1_net_bps": 5.},
        {"arm": "c", "top5_net_bps": 9.5, "month_std_net_bps": 5., "month_worst_net_bps": -10., "top1_net_bps": 7.},
    ])
    assert select_winner(table, tie_tolerance_bps=1.).arm == "c"
