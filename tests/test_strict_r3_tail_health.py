from __future__ import annotations

import numpy as np
import pandas as pd

from extreme_price_movements.strict_r3_tail_health import (
    TailHealthSpec,
    apply_exact_producer_tail_health,
)


def _frame() -> pd.DataFrame:
    decision = pd.to_datetime([
        "2026-06-01T00:00:00Z",
        "2026-06-01T01:00:00Z",
        "2026-06-01T02:00:00Z",
        "2026-06-01T03:00:00Z",
        "2026-06-01T04:00:00Z",
        "2026-06-01T05:00:00Z",
        "2026-06-01T06:00:00Z",
        "2026-06-01T07:00:00Z",
    ], utc=True)
    return pd.DataFrame({
        "candidate_id": [f"row-{index}" for index in range(len(decision))],
        "__decision_ts__": decision,
        "policy_label_available_ts": decision + pd.Timedelta(hours=1),
        "side_name": "long",
        "producer_bundle_id": ["p1", "p1", "p1", "p1", "p1", "p1", "p2", "p1"],
        "ev_bridge_prior_expected_net_bps": [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 20.0],
        "policy_path_valid": True,
        "policy_net_bps": [-100.0, -100.0, -100.0, -100.0, np.nan, np.nan, np.nan, np.nan],
    })


def test_tail_health_is_producer_local_and_uses_only_resolved_tail_labels() -> None:
    spec = TailHealthSpec(
        residual_windows_days=(3,), residual_shrinkage_rows=(1.0,),
        minimum_residual_rows=4, trim_fraction=0.0,
    )
    out, audit = apply_exact_producer_tail_health(_frame(), spec=spec)

    # The first decision has no earlier resolved label; its own future outcome
    # cannot influence its score.  Row five can use only the first four rows,
    # all of which resolved strictly before its decision timestamp.
    assert out.loc[0, "tail_health_recent_residual_bps"] == 0.0
    assert out.loc[1, "tail_health_recent_residual_bps"] == 0.0
    assert out.loc[5, "tail_health_recent_residual_bps"] <= -150.0
    # p2 must not borrow p1's four adverse observations.
    assert out.loc[6, "tail_health_recent_residual_bps"] == 0.0
    # A row outside the reserve-defined positive tail cannot be admitted even
    # if a positive correction would otherwise lift it over the EV threshold.
    assert not out.loc[7, "tail_health_reserve_eligible"]
    assert not out.loc[7, "tail_health_admitted_ge_50bps"]
    assert audit["strictly_prior_resolved"].all()


def test_tail_health_confidence_bound_only_demotes() -> None:
    frame = _frame()
    common = dict(
        residual_windows_days=(3,), residual_shrinkage_rows=(1.0,),
        minimum_residual_rows=4, trim_fraction=0.0,
    )
    mean, _ = apply_exact_producer_tail_health(frame, spec=TailHealthSpec(**common))
    lcb, _ = apply_exact_producer_tail_health(
        frame, spec=TailHealthSpec(**common, lower_confidence_z=1.0),
    )
    assert (lcb["tail_health_lcb_bps"] <= mean["tail_health_expected_net_bps"]).all()
