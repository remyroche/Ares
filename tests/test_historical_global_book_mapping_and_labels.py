from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.materialize_historical_frozen_backcast_global_book_mapping_source import (
    RAW_SCORE,
    build_mapping_source,
)
from scripts.materialize_canonical_global_book_conversion_transition_labels import (
    materialize_global_book_labels,
)


def _small_scores_and_labels() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for day in (pd.Timestamp("2023-01-01T00:00:00Z"), pd.Timestamp("2023-01-02T00:00:00Z")):
        for side, score, net in (("long", 0.8, 0.03), ("short", 0.2, -0.01)):
            rows.append({"__ts__": day - pd.Timedelta(hours=1), "__symbol__": f"{side}-asset", "side_name": side, "candidate_id": f"{day.date()}-{side}", RAW_SCORE: score})
            # Decision is exactly one hour after signal; label end is +12h.
    scores = pd.DataFrame(rows)
    labels = scores.loc[:, ["__ts__", "__symbol__", "side_name", "candidate_id"]].copy()
    labels["execution_decision_utc"] = labels["__ts__"] + pd.Timedelta(hours=1)
    labels["execution_label_end_utc"] = labels["execution_decision_utc"] + pd.Timedelta(hours=12)
    labels["execution_gross_ev_12h"] = [0.04, 0.00, 0.04, 0.00]
    labels["execution_cost_return"] = 0.01
    labels["execution_net_ev_12h"] = labels["execution_gross_ev_12h"] - labels["execution_cost_return"]
    labels["execution_exit_class"] = ["trailing", "full_stop", "trailing", "timeout"]
    return scores, labels


def test_mapping_is_prior_resolved_only_and_keeps_warmup_unmapped() -> None:
    scores, labels = _small_scores_and_labels()
    mapped, audit = build_mapping_source(scores, labels, minimum_reference_rows=2)
    first = mapped["execution_decision_utc"].dt.day.eq(1)
    second = mapped["execution_decision_utc"].dt.day.eq(2)

    assert not mapped.loc[first, "mapped_eligible"].any()
    assert mapped.loc[second, "mapped_eligible"].all()
    assert mapped.loc[second, "map_reference_rows"].eq(2).all()
    assert mapped.loc[second, "mapped_direct_net"].notna().all()
    assert audit.loc[audit["snapshot_utc"].dt.day.eq(2), "reference_label_end_max_utc"].iloc[0] < pd.Timestamp("2023-01-02T00:00:00Z")


def _global_mapping() -> pd.DataFrame:
    rows = []
    for day in (pd.Timestamp("2023-02-01T00:00:00Z"), pd.Timestamp("2023-02-02T00:00:00Z")):
        for index in range(1000):
            gross = 0.03 if index < 100 else 0.0
            rows.append({
                "candidate_id": f"{day.date()}-{index:04d}", "__symbol__": f"A{index}",
                "side_name": "long" if index % 2 == 0 else "short",
                "execution_decision_utc": day, "execution_label_end_utc": day + pd.Timedelta(hours=12),
                "candidate_month": "2023-02", "mapped_eligible": True,
                "mapped_direct_net": 1.0 - index / 1000.0,
                "map_reference_rows": 1000, "map_side_reference_rows": 500, "map_cell_reference_rows": 1000,
                "execution_gross_ev_12h": gross, "execution_cost_return": 0.01,
                "execution_net_ev_12h": gross - 0.01,
                "execution_exit_class": "trailing" if gross else "full_stop",
                "opportunity_gross_above_cost_0bps": float(gross > .01),
                "opportunity_gross_above_cost_25bps": float(gross > .0125),
            })
    return pd.DataFrame(rows)


def test_top10_slice_uses_single_global_book_with_exact_before_after_windows() -> None:
    labels = materialize_global_book_labels(_global_mapping())
    assert set(labels["horizon_hours"]) == {3, 12}
    top10 = labels.loc[labels["book_fraction"].eq(0.10)].copy()
    assert top10["selection_contract"].eq("one_pooled_global_mapped_direct_net").all()
    for horizon, group in top10.groupby("horizon_hours"):
        assert group["before_window_end_utc"].eq(group["cohort_anchor_utc"]).all()
        assert group["after_window_start_utc"].eq(group["cohort_anchor_utc"]).all()
        assert group["before_window_start_utc"].eq(group["cohort_anchor_utc"] - pd.Timedelta(hours=int(horizon))).all()
        assert group["after_window_end_utc"].eq(group["cohort_anchor_utc"] + pd.Timedelta(hours=int(horizon))).all()
        assert group["after_target_available_utc"].ge(group["after_window_end_utc"]).all()
    # The top 10% is global, so the winning raw-score rows can be all one side.
    assert np.isfinite(top10["after_direct_mean_net"].dropna()).all()
