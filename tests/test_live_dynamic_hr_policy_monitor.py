import numpy as np
import pandas as pd

from scripts.monitor_live_dynamic_hr_policy import (
    _policy_replay_scope_mask,
    _policy_replay_summary,
)


def test_policy_replay_scope_excludes_rank_rejects_by_default() -> None:
    rows = pd.DataFrame(
        {
            "portfolio_decision": ["rank_rejected", "portfolio_rejected", "portfolio_rejected"],
            "portfolio_reject_reason": [
                "rank_below_dynamic_threshold",
                "global_auction_capacity:global_entry_cap_reached",
                "global_auction_capacity:max_new_entries_per_bar_reached",
            ],
        }
    )

    mask = _policy_replay_scope_mask(rows, "capacity_rejects")

    assert mask.tolist() == [False, True, True]


def test_policy_replay_scope_can_select_non_rank_rejects() -> None:
    rows = pd.DataFrame(
        {
            "portfolio_decision": ["rank_rejected", "portfolio_rejected", "portfolio_rejected"],
            "portfolio_reject_reason": [
                "rank_below_dynamic_threshold",
                "global_auction_capacity:global_entry_cap_reached",
                "liquidity_guard",
            ],
        }
    )

    mask = _policy_replay_scope_mask(rows, "non_rank_rejects")

    assert mask.tolist() == [False, True, True]


def test_policy_replay_summary_uses_resolved_simple_policy_returns() -> None:
    table = pd.DataFrame(
        {
            "policy_replay_status": ["resolved", "resolved", "pending_future_bars"],
            "net_return": [0.02, -0.01, np.nan],
            "dynamic_hr_surprise_head": ["long_dist", "long_dist", "short_asset"],
        }
    )

    summary = _policy_replay_summary(table)

    assert summary["mtm_proxy_used"] is False
    assert summary["resolved_rows"] == 2
    assert summary["pending_rows"] == 1
    assert summary["mean_net_return"] == 0.005
    assert summary["hit_rate"] == 0.5
    assert summary["by_head"]["long_dist"]["resolved_rows"] == 2
    assert summary["by_head"]["short_asset"]["pending_rows"] == 1
