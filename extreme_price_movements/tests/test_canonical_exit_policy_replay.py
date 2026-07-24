import pandas as pd

from scripts.materialize_canonical_exit_policy_replay import (
    _apply_policy_spread_to_returns,
)


def test_policy_spread_is_already_embedded_and_not_deducted_twice():
    rows = pd.DataFrame(
        {
            "net_return": [0.02],
            "gross_return": [0.03],
            "spread_cost_bps": [12.0],
            "exit_spread_cost_bps": [18.0],
        }
    )

    adjusted = _apply_policy_spread_to_returns(rows)
    repeated = _apply_policy_spread_to_returns(adjusted)

    assert adjusted.loc[0, "net_return"] == 0.02
    assert adjusted.loc[0, "gross_return"] == 0.03
    assert not bool(adjusted.loc[0, "policy_spread_applied_to_returns"])
    assert bool(adjusted.loc[0, "policy_spread_embedded_in_executable_prices"])
    pd.testing.assert_frame_equal(adjusted, repeated)
