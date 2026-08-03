import numpy as np
import pandas as pd

from scripts.materialize_canonical_economic_conversion_transition_labels import (
    _window_metrics,
    add_frozen_causal_score_deciles,
    materialize_transition_labels,
)


def _source(hours: int = 18) -> pd.DataFrame:
    records = []
    origin = pd.Timestamp("2025-02-01T00:00:00Z")
    for hour in range(hours):
        stamp = origin + pd.Timedelta(hours=hour)
        for side in ("long", "short"):
            for rank in range(10):
                exit_class = ("trailing", "timeout", "full_stop", "adverse_exit")[rank % 4]
                net = float(hour + rank + 1) / 100.0
                if rank % 3 == 0:
                    net = -net
                records.append(
                    {
                        "candidate_id": f"{side}-{hour:02d}-{rank:02d}",
                        "side_name": side,
                        "__symbol__": f"S{rank:02d}",
                        "__ts__": stamp,
                        "base_oof_score": float(100 - rank),
                        "execution_label_end_utc": stamp + pd.Timedelta(hours=13),
                        "execution_net_ev_12h": net,
                        "execution_exit_class": exit_class,
                        "opportunity_gross_above_cost_0bps": net > 0.0,
                        "opportunity_gross_above_cost_25bps": net > 0.0025,
                    }
                )
    return pd.DataFrame.from_records(records)


def _label(labels: pd.DataFrame, anchor_hour: int, *, horizon: int = 3) -> pd.Series:
    anchor = pd.Timestamp("2025-02-01T00:00:00Z") + pd.Timedelta(hours=anchor_hour)
    return labels.loc[
        labels["cohort_anchor_utc"].eq(anchor)
        & labels["side_name"].eq("long")
        & labels["frozen_base_score_decile"].eq(0)
        & labels["horizon_hours"].eq(horizon)
    ].iloc[0]


def test_half_open_windows_exclude_anchor_from_before_and_end_from_after():
    labels = materialize_transition_labels(_source())
    row = _label(labels, 5)
    # The top-score long cohort has exactly one row each hour.  Before is 2,3,4;
    # after is 5,6,7, proving [s-H,s) and [s,s+H) respectively.
    assert row["before_candidate_support"] == 3
    assert row["after_candidate_support"] == 3
    assert row["before_direct_mean_net"] == np.mean([-0.03, -0.04, -0.05])
    assert row["after_direct_mean_net"] == np.mean([-0.06, -0.07, -0.08])
    assert row["after_window_end_utc"] == pd.Timestamp("2025-02-01T08:00:00Z")


def test_availability_uses_actual_latest_execution_label_end_plus_one_hour():
    source = _source()
    # Change only the final row in the after window for long/top-decile.
    changed = source["candidate_id"].eq("long-07-00")
    source.loc[changed, "execution_label_end_utc"] = pd.Timestamp("2025-02-01T22:00:00Z")
    labels = materialize_transition_labels(source)
    row = _label(labels, 5)
    assert row["after_target_available_utc"] == pd.Timestamp("2025-02-01T23:00:00Z")
    # On the normal exact hourly source, H=3 availability is s+H+13h.
    normal = _label(materialize_transition_labels(_source()), 5)
    assert normal["after_target_available_utc"] == pd.Timestamp("2025-02-01T21:00:00Z")


def test_frozen_deciles_are_deterministic_and_do_not_use_outcomes():
    source = _source(hours=1)
    source.loc[:, "base_oof_score"] = 1.0  # force every score tie
    first = add_frozen_causal_score_deciles(source)
    changed = source.copy()
    changed["execution_net_ev_12h"] *= -1000.0
    changed["execution_exit_class"] = "full_stop"
    second = add_frozen_causal_score_deciles(changed)
    key = ["candidate_id", "frozen_base_score_decile"]
    assert first.loc[:, key].equals(second.loc[:, key])
    long = first.loc[first["side_name"].eq("long")].sort_values("candidate_id")
    assert long["frozen_base_score_decile"].tolist() == list(range(10))


def test_empty_conditional_support_is_explicit_and_never_synthesized():
    rows = _source(hours=1).iloc[:2].copy()
    rows["execution_net_ev_12h"] = [-0.01, 0.0]
    metrics = _window_metrics(rows)
    assert metrics["favorable_net_support"] == 0
    assert metrics["favorable_net_missing_support_flag"]
    assert np.isnan(metrics["conditional_favorable_net_robust_mean"])
    assert metrics["adverse_loss_support"] == 2
    assert not metrics["adverse_loss_missing_support_flag"]


def test_exit_mixture_reconciles_the_direct_mean_exactly():
    rows = _source(hours=1).iloc[:10].copy()
    metrics = _window_metrics(rows)
    assert metrics["exit_mixture_reconciles_direct_mean_flag"]
    assert metrics["exit_mixture_expected_net"] == metrics["direct_mean_net"]
    assert sum(metrics[f"p_exit_{name}"] for name in ("trailing", "timeout", "full_stop", "adverse_exit")) == 1.0
