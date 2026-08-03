from __future__ import annotations

import pandas as pd

from scripts.build_canonical_exact_policy_regime_input import replace_outcomes


def test_replace_outcomes_removes_superseded_target_and_reconciles() -> None:
    identity = {
        "__ts__": [pd.Timestamp("2026-07-01", tz="UTC")],
        "__symbol__": ["BTC/USD:USD"],
        "side_name": ["long"],
        "candidate_id": ["candidate"],
    }
    features = pd.DataFrame(
        identity
        | {
            "execution_net_ev_12h": [0.20],
            "execution_gross_ev_12h": [0.21],
            "feature": [1.5],
        }
    )
    exact = pd.DataFrame(
        identity
        | {
            "execution_decision_utc": [
                pd.Timestamp("2026-07-01 01:00", tz="UTC")
            ],
            "execution_label_end_utc": [
                pd.Timestamp("2026-07-01 13:00", tz="UTC")
            ],
            "execution_net_ev_12h": [0.01],
            "execution_gross_ev_12h": [0.02],
            "execution_cost_return": [0.01],
            "execution_exit_reason": ["trailing"],
            "execution_exit_hour": [3.0],
            "execution_mfe_return_12h": [0.04],
            "execution_mae_return_12h": [0.01],
        }
    )
    atr = pd.DataFrame(identity | {"oof_entry_atr_fraction": [0.02]})
    joined, audit = replace_outcomes(features, exact, atr)
    assert joined.loc[0, "execution_net_ev_12h"] == 0.01
    assert joined.loc[0, "feature"] == 1.5
    assert joined.loc[0, "oof_entry_atr_fraction"] == 0.02
    assert audit["target_change"]["changed_rows"] == 1
    assert audit["max_gross_cost_net_delta"] == 0.0
