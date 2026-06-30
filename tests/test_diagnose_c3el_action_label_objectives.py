import pandas as pd

from scripts.diagnose_c3el_action_label_objectives import load_group_panel, summarise_objectives


def _row(strategy_id: str, idx: int, *, best: float, worst: float, immediate: float = 0.0, y: float = 0.0):
    return {
        "timestamp": f"2026-06-01T{idx:02d}:00:00+00:00",
        "strategy_id": strategy_id,
        "group_can_bind": 1.0,
        "y_intervene": y,
        "best_gain": best,
        "best_margin": max(best, 0.0),
        "best_gain_per_notional": max(best, 0.0) / 10000.0,
        "best_margin_per_notional": max(best, 0.0) / 10000.0,
        "best_immediate_gain": immediate,
        "best_nonbaseline_gain": best,
        "worst_nonbaseline_gain": worst,
    }


def test_summarise_objectives_flags_sparse_low_precision_headroom():
    groups = pd.DataFrame(
        [
            _row("short_asset_a", 0, best=10.0, worst=-100.0, y=0.0),
            _row("short_asset_b", 1, best=0.0, worst=-50.0, y=0.0),
            _row("short_asset_c", 2, best=0.0, worst=-25.0, y=0.0),
            _row("short_asset_d", 3, best=0.0, worst=-25.0, y=0.0),
        ]
    )
    groups["head"] = "short_asset"

    report = summarise_objectives(groups)
    row = report.iloc[0]

    assert row["diagnosis"] == "sparse_low_precision_headroom"
    assert row["current_positive_rate"] == 0.0
    assert row["relaxed_full_positive_rate"] == 0.25
    assert row["full_gain_to_worst_abs_ratio"] == 0.05


def test_summarise_objectives_flags_sparse_but_viable_headroom():
    groups = pd.DataFrame(
        [
            _row("short_boll_a", 0, best=100.0, worst=-20.0, y=1.0),
            _row("short_boll_b", 1, best=0.0, worst=-10.0, y=0.0),
            _row("short_boll_c", 2, best=0.0, worst=-10.0, y=0.0),
            _row("short_boll_d", 3, best=0.0, worst=-10.0, y=0.0),
            _row("short_boll_e", 4, best=0.0, worst=-10.0, y=0.0),
            _row("short_boll_f", 5, best=0.0, worst=-10.0, y=0.0),
            _row("short_boll_g", 6, best=0.0, worst=-10.0, y=0.0),
            _row("short_boll_h", 7, best=0.0, worst=-10.0, y=0.0),
            _row("short_boll_i", 8, best=0.0, worst=-10.0, y=0.0),
            _row("short_boll_j", 9, best=0.0, worst=-10.0, y=0.0),
            _row("short_boll_k", 10, best=0.0, worst=-10.0, y=0.0),
            _row("short_boll_l", 11, best=0.0, worst=-10.0, y=0.0),
            _row("short_boll_m", 12, best=0.0, worst=-10.0, y=0.0),
            _row("short_boll_n", 13, best=0.0, worst=-10.0, y=0.0),
            _row("short_boll_o", 14, best=0.0, worst=-10.0, y=0.0),
            _row("short_boll_p", 15, best=0.0, worst=-10.0, y=0.0),
            _row("short_boll_q", 16, best=0.0, worst=-10.0, y=0.0),
            _row("short_boll_r", 17, best=0.0, worst=-10.0, y=0.0),
            _row("short_boll_s", 18, best=0.0, worst=-10.0, y=0.0),
            _row("short_boll_t", 19, best=0.0, worst=-10.0, y=0.0),
            _row("short_boll_u", 20, best=0.0, worst=-10.0, y=0.0),
        ]
    )
    groups["head"] = "short_boll"

    report = summarise_objectives(groups)
    row = report.iloc[0]

    assert row["diagnosis"] == "sparse_but_viable_headroom"
    assert row["current_positive_rate"] < 0.05
    assert row["full_gain_to_worst_abs_ratio"] > 0.25


def test_load_group_panel_deduplicates_action_rows(tmp_path):
    panel = tmp_path / "panel.csv"
    pd.DataFrame(
        [
            _row("short_asset_a", 0, best=10.0, worst=-5.0),
            _row("short_asset_a", 0, best=20.0, worst=-5.0),
        ]
    ).to_csv(panel, index=False)

    groups = load_group_panel([panel])

    assert len(groups) == 1
    assert groups["best_nonbaseline_gain"].iloc[0] == 20.0
    assert groups["head"].iloc[0] == "short_asset"
