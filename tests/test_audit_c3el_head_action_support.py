import pandas as pd

from scripts.audit_c3el_head_action_support import (
    _decision_table,
    _decision_table_from_groups,
    _group_panel,
    _normalise_panel,
)


def test_group_panel_marks_positive_nonbase_by_threshold() -> None:
    ts = pd.Timestamp("2026-06-08 12:00:00", tz="UTC")
    frame = pd.DataFrame(
        {
            "timestamp": [ts, ts, ts, ts + pd.Timedelta(hours=1)],
            "strategy_id": ["short_asset_v1", "short_asset_v1", "short_asset_v1", "short_boll_v1"],
            "multiplier": [1.0, 0.5, 0.0, 0.5],
            "delta_full_J": [0.0, 30.0, 125.0, -20.0],
            "delta_immediate_J": [0.0, 10.0, 80.0, -5.0],
            "action_binds": [0.0, 1.0, 1.0, 1.0],
            "affected_notional": [1000.0, 1000.0, 1000.0, 500.0],
        }
    )

    groups = _group_panel(_normalise_panel(frame))
    short_asset = groups.loc[groups["head"].eq("short_asset")].iloc[0]
    short_boll = groups.loc[groups["head"].eq("short_boll")].iloc[0]

    assert bool(short_asset["positive_nonbase_e50"])
    assert bool(short_asset["positive_nonbase_e100"])
    assert not bool(short_asset["positive_nonbase_e150"])
    assert not bool(short_boll["positive_nonbase_e50"])
    assert bool(short_boll["harmful_nonbase"])


def test_decision_table_uses_best_recurrent_panel_per_head() -> None:
    by_panel = pd.DataFrame(
        {
            "panel": ["small", "large"],
            "panel_path": ["small.csv", "large.csv"],
            "head": ["short_asset", "short_asset"],
            "groups": [20, 100],
            "weeks": [1, 4],
            "positive_nonbase_e50_groups": [8, 70],
            "positive_nonbase_e50_weeks": [1, 4],
            "positive_nonbase_e100_groups": [4, 30],
            "positive_nonbase_e150_groups": [2, 10],
            "strict_y_groups": [8, 65],
            "can_bind_groups": [10, 80],
            "harmful_nonbase_groups": [5, 40],
            "best_nonbase_delta_p90": [60.0, 120.0],
            "best_nonbase_delta_p99": [90.0, 300.0],
            "worst_nonbase_delta_p05": [-20.0, -50.0],
        }
    )

    decisions = _decision_table(by_panel, production_groups=60, production_weeks=3)

    assert decisions.loc[0, "status"] == "production_candidate"
    assert decisions.loc[0, "best_panel"] == "large"


def test_aggregate_decision_table_uses_unique_groups_across_panels() -> None:
    rows = []
    for idx in range(65):
        rows.append(
            {
                "head": "short_asset",
                "timestamp": pd.Timestamp("2026-06-01", tz="UTC") + pd.Timedelta(hours=idx),
                "strategy_id": f"short_asset_{idx}",
                "week_start": pd.Timestamp("2026-06-01", tz="UTC") + pd.Timedelta(days=(idx // 20) * 7),
                "best_nonbase_delta": 75.0,
                "worst_nonbase_delta": -10.0,
                "strict_y_intervene": 1.0,
                "can_bind": True,
                "harmful_nonbase": True,
            }
        )
    groups = pd.DataFrame(rows)
    duplicate = groups.iloc[:10].copy()
    duplicate["best_nonbase_delta"] = 25.0
    groups = pd.concat([duplicate, groups], ignore_index=True)

    decisions = _decision_table_from_groups(
        groups,
        production_groups=60,
        production_weeks=3,
        recent_start=pd.Timestamp("2026-06-01", tz="UTC"),
    )

    row = decisions.iloc[0]
    assert row["status"] == "production_candidate"
    assert row["best_panel"] == "aggregate_unique_groups"
    assert row["positive_e50_groups"] == 65
    assert row["recent_positive_e50_groups"] == 65
    assert row["support_blocker"] == "none"
