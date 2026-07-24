import json

import numpy as np
import pandas as pd

from extreme_price_movements import simple_policy_optimiser as optimiser
from extreme_price_movements.simple_policy_winner import (
    WINNER_FORWARD_BARS,
    WINNER_POLICY_PATHWAY_ID,
    apply_raw_bayesian_sizing_state,
    fit_raw_bayesian_sizing_state,
)


def _rows(n: int = 600) -> pd.DataFrame:
    index = np.arange(n)
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-05-01", periods=n, freq="h", tz="UTC"),
            "side": np.where(index % 2, -1, 1),
            "policy_archetype": np.where(index % 3, "continuation", "reversal"),
            "rank_pct": np.linspace(0.70, 1.0, n),
            "expected_net_ev_after_1pct_mlp_direct": np.linspace(-1.0, 1.0, n),
            "meta_hit_probability_uncertainty_p1mp": np.linspace(1.0, 0.0, n),
            "gmm_ood_score": np.sin(index / 20.0),
            "cluster_entropy_norm": np.cos(index / 30.0),
        }
    )


def test_default_optimizer_path_is_one_minute_joint_raw_bayesian(tmp_path):
    assert optimiser.DEFAULT_POLICY_PATHWAY_ID == WINNER_POLICY_PATHWAY_ID
    assert optimiser.DEFAULT_BAR_MINUTES == 1
    assert optimiser.DEFAULT_FORWARD_BARS == WINNER_FORWARD_BARS == 1440
    store = optimiser._make_policy_replay_store(tmp_path, "perps")
    assert store.timeframe == "1m"
    assert str(store.root_dir).endswith("execution_1m")


def test_raw_bayesian_state_is_frozen_serialisable_and_bounded():
    rows = _rows()
    outcomes = np.where(
        (rows["policy_archetype"] == "continuation")
        & (rows["expected_net_ev_after_1pct_mlp_direct"] > 0.0),
        0.02,
        -0.01,
    )
    state = fit_raw_bayesian_sizing_state(
        rows,
        outcomes,
        strength=3.0,
        ood_weight=0.5,
    )
    json.dumps(state)
    multipliers = apply_raw_bayesian_sizing_state(rows, state)
    assert len(state["cells"]) > 0
    assert multipliers.min() >= 0.65
    assert multipliers.max() <= 1.20
    assert np.ptp(multipliers) > 0.0

    changed_future = rows.copy()
    changed_future.loc[len(rows) // 2 :, "expected_net_ev_after_1pct_mlp_direct"] *= -100.0
    original_prefix = apply_raw_bayesian_sizing_state(rows.iloc[:100], state)
    changed_prefix = apply_raw_bayesian_sizing_state(changed_future.iloc[:100], state)
    np.testing.assert_allclose(original_prefix, changed_prefix)


def test_winner_capital_layer_is_structurally_disabled():
    rows = _rows(20)
    long_guard = optimiser._frozen_winner_adverse_guard(rows.assign(side=1))
    short_guard = optimiser._frozen_winner_adverse_guard(rows.assign(side=-1))
    assert long_guard["adverse_exit_enabled"] is True
    assert long_guard["adverse_exit_fast_bars"] == 60
    assert short_guard["adverse_exit_enabled"] is False
