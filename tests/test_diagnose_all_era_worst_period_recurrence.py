from __future__ import annotations

import pandas as pd

from scripts.diagnose_all_era_worst_period_recurrence import (
    compact_feature_columns,
    identify_worst_weeks_by_era,
    recurrence_summary,
)


def test_worst_weeks_are_defined_independently_per_lineage() -> None:
    records = []
    for era, offset in (("old", 0), ("new", 100)):
        for n in range(8):
            records.append({"period_type": "week", "complete_for_percentage": True, "lineage_id": era, "mean_net_bps": offset + n, "period_start_utc": pd.Timestamp("2024-01-01", tz="UTC") + pd.Timedelta(days=7*n)})
    output = identify_worst_weeks_by_era(pd.DataFrame(records), quantile=.25)
    assert output.groupby("era_id")["worst_week"].sum().to_dict() == {"new": 2, "old": 2}
    assert output.loc[output["era_id"].eq("new") & output["worst_week"], "mean_net_bps"].max() == 101


def test_structural_compact_selection_is_causal_and_bounded() -> None:
    columns = ["source_utc", "mv__foo__robust_z_24h", "mv__bar__delta_24h", "mv__target_leak__delta_24h", "mv__dependence__eig1_share_168h", "mv__a__delta_24h", "mv__b__delta_24h", "mv__c__delta_24h", "mv__d__delta_24h", "mv__e__delta_24h"]
    output = compact_feature_columns(columns, maximum=3)
    assert len(output) == 3
    assert "mv__foo__robust_z_24h" in output
    assert all("target" not in field for field in output)


def test_recurrence_requires_same_direction_in_two_tested_eras() -> None:
    frame = pd.DataFrame({"era_id": ["a", "b", "c"], "diagnostic_kind": ["feature_shift"]*3, "feature": ["x"]*3, "era_significant": [True, True, True], "direction": [1, 1, -1]})
    output = recurrence_summary([frame], expected_eras=["a", "b", "c", "missing"])
    row = output.iloc[0]
    assert bool(row["recurrent"])
    assert row["recurrent_direction"] == "positive"
    assert row["uncovered_calendar_eras"] == "missing"
