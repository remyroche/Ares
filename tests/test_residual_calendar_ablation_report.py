from argparse import Namespace
import importlib.util
from pathlib import Path

import pandas as pd
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "report_residual_calendar_ablation.py"
SPEC = importlib.util.spec_from_file_location("report_residual_calendar_ablation", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
run = MODULE.run


def test_calendar_ablation_reports_side_archetype_deltas(tmp_path):
    keys = {
        "__ts__": pd.to_datetime(
            ["2026-04-01T00:00:00Z", "2026-04-01T01:00:00Z", "2026-04-02T00:00:00Z"]
        ),
        "__symbol__": ["A", "B", "A"],
        "side_name": ["long", "long", "short"],
        "archetype_policy_key": ["breakout", "breakout", "mixed"],
    }
    outcomes = pd.DataFrame(
        {
            **keys,
            "ev_after_1pct": [-0.01, 0.03, 0.02],
            "clean_exec": [0.0, 1.0, 1.0],
            "dirty_positive": [0.0, 0.0, 0.0],
            "first_touch_bad_mae_1r": [1.0, 0.0, 0.0],
            "full_path_bad_mae_1r": [1.0, 0.0, 0.0],
            "timeout": [0.0, 0.0, 0.0],
        }
    )
    outcomes_path = tmp_path / "outcomes.parquet"
    outcomes.to_parquet(outcomes_path, index=False)
    champion = pd.DataFrame({**keys, "selected": [True, True, True]})
    challenger = pd.DataFrame({**keys, "selected": [False, True, True]})
    champion_path = tmp_path / "champion.parquet"
    challenger_path = tmp_path / "challenger.parquet"
    champion.to_parquet(champion_path, index=False)
    challenger.to_parquet(challenger_path, index=False)
    calendar = pd.DataFrame(
        {
            "day": ["2026-04-01", "2026-04-02"],
            "side_name": ["long", "short"],
            "archetype_policy_key": ["breakout", "mixed"],
            "adverse_event_rows": [2, 0],
            "favorable_event_rows": [0, 1],
            "material_extreme": [True, False],
        }
    )
    calendar_path = tmp_path / "calendar.csv"
    calendar.to_csv(calendar_path, index=False)
    output = tmp_path / "report"

    manifest = run(
        Namespace(
            outcomes=outcomes_path,
            calendar=calendar_path,
            arm=[
                ("champion", champion_path, "selected"),
                ("challenger", challenger_path, "selected"),
            ],
            baseline="champion",
            output_dir=output,
        )
    )

    assert manifest["calendar_cells_matched"] == 2
    report = pd.read_csv(output / "side_archetype_metrics.csv")
    row = report.loc[
        report["arm"].eq("challenger")
        & report["side_name"].eq("long")
        & report["archetype_policy_key"].eq("breakout")
    ].iloc[0]
    assert row["mean_ev_after_1pct"] == pytest.approx(0.03)
    assert row["baseline__mean_ev_after_1pct"] == pytest.approx(0.01)
    assert row["delta_vs_baseline__mean_ev_after_1pct"] == pytest.approx(0.02)
