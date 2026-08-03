from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.run_long_raw_base_residual_h12_ablation import (
    _attach_pooled_tail_weights,
    _global_tail_weights,
    base_target,
    meta_target,
)


def _frame() -> pd.DataFrame:
    return pd.DataFrame({
        "execution_net_ev_12h": [-0.020, 0.010, 0.030],
        "base_expected_net": [-0.010, 0.000, 0.010],
        "__adverse_competing_risk_12h__": [1.0, 0.0, 0.0],
        "__mae_before_meaningful_mfe_atr_12h__": [6.0, 1.0, 0.2],
        "__opportunity_occurred_12h__": [0.0, 1.0, 1.0],
        "__time_to_first_meaningful_mfe_hours_12h__": [12.0, 8.0, 1.0],
    })


def test_all_base_targets_are_bounded_and_cost_directed() -> None:
    frame = _frame()
    for arm in ("net_hurdle_soft", "risk_penalised_net", "timely_clean_net"):
        values = base_target(frame, arm)
        assert np.all((values >= 0.0) & (values <= 1.0))
        assert values[2] > values[0]


def test_meta_targets_keep_economic_direction() -> None:
    frame = _frame()
    assert meta_target(frame, "net_residual")[2] > meta_target(frame, "net_residual")[0]
    soft = meta_target(frame, "policy_soft_clear")
    assert soft[2] > soft[0]
    assert np.all((soft >= 0.0) & (soft <= 1.0))


def test_tail_weights_are_one_global_book_not_timestamp_groups() -> None:
    frame = _frame().assign(base_expected_net=[0.0, 100.0, 99.0])
    weights = _global_tail_weights(frame, clean=False)
    assert weights.tolist() == [1.0, 4.0, 1.0]


def test_pooled_tail_is_attached_before_side_split_and_rejoins_exactly() -> None:
    frame = _frame().assign(
        candidate_id=["l0", "s0", "l1"],
        side_name=["long", "short", "long"],
        __ts__=pd.to_datetime(["2024-04-01", "2024-04-01", "2024-04-02"], utc=True),
        base_expected_net=[0.0, 100.0, 99.0],
    )
    attached = _attach_pooled_tail_weights(frame, np.array([True, True, True]))
    # ceil(10% of three) is one and it is the short row, proving this is not
    # a top-10 selection independently within long and short.
    assert attached.set_index("candidate_id").loc["s0", "pooled_tail_member"]
    assert int(attached.pooled_tail_member.sum()) == 1
    assert int(attached.loc[attached.side_name.eq("long"), "pooled_tail_member"].sum()) == 0
    assert int(attached.loc[attached.side_name.eq("short"), "pooled_tail_member"].sum()) == 1
    assert attached.loc[attached.pooled_tail_member, "pooled_tail_weight"].tolist() == [4.0]


def test_pooled_tail_weights_remain_fixed_when_fold_rows_are_sliced() -> None:
    frame = _frame().assign(
        candidate_id=["l0", "s0", "l1"],
        side_name=["long", "short", "long"],
        __ts__=pd.to_datetime(["2024-04-01", "2024-04-01", "2024-04-02"], utc=True),
        base_expected_net=[0.0, 100.0, 99.0],
    )
    attached = _attach_pooled_tail_weights(frame, np.array([True, True, True]))
    short_fold = attached.loc[attached.side_name.eq("short")]
    assert short_fold.pooled_tail_weight.tolist() == [4.0]
    # Recomputing on this side would also select it; checking the immutable
    # original field is what guards a later fold implementation from doing so.
    assert attached.set_index("candidate_id").loc["s0", "pooled_tail_member"]
