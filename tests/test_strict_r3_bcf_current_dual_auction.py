import json

import pandas as pd

from extreme_price_movements.strict_r3_shadow_portfolio import (
    ShadowPortfolioPolicy,
    ShadowPortfolioState,
    auction_admitted_snapshot,
)


def test_dual_bcf_current_auction_requires_both_maps_and_prioritizes_bcf_ev():
    policy = ShadowPortfolioPolicy.from_payload(json.load(open(
        "config/strict_r3_bcf_current_dual_mc1_portfolio_challenger_v1.json"
    )))
    decision = pd.Timestamp("2026-08-17T05:00:00Z")
    state = ShadowPortfolioState(decision, 1000.0, tuple())
    rows = pd.DataFrame({
        "candidate_id": ["a", "b", "c"],
        "__decision_ts__": [decision, decision, decision],
        "__symbol__": ["A/USD:USD", "B/USD:USD", "C/USD:USD"],
        "side_name": ["long", "long", "long"],
        "final_score": [0.99, 0.80, 0.97],
        "frozen_base_contract_complete": [True, True, True],
        "base_route_timestamp_top30": [True, True, True],
        "mc1_d2_expected_net_bps": [40.0, 40.0, 90.0],
        "current_mc1_admitted_ge_30bps": [True, True, True],
        "bcf_mc1_expected_net_bps": [45.0, 85.0, 70.0],
        "bcf_mc1_admitted_ge_30bps": [True, True, False],
        "causal_21d_side_expected_net_bps": [45.0, 85.0, 70.0],
        "causal_21d_side_admitted_ge_50bps": [True, True, False],
    })
    out = auction_admitted_snapshot(rows, state=state, policy=policy)
    accepted = out.loc[out["portfolio_accepted"], "candidate_id"].tolist()
    assert set(accepted) == {"a", "b"}
    ranks = out.set_index("candidate_id")["portfolio_priority_rank"].to_dict()
    assert ranks["b"] == 1.0
    assert ranks["a"] == 2.0
    assert out.loc[out["candidate_id"].eq("c"), "portfolio_rejection_reason"].item() == "ev_map_rejected"
