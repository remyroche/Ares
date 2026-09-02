from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from extreme_price_movements.inference.p8u_c0_c1_no_order_stack import P8UC0C1NoOrderStack
from extreme_price_movements.inference.p8u_sealed_inference_stack import P8UCoordinateScores


IDENTITY = ["candidate_id", "__decision_ts__", "__symbol__", "side_name"]


def _coordinates() -> P8UCoordinateScores:
    rows = pd.DataFrame({
        "candidate_id": ["both", "c0", "c1"],
        "__decision_ts__": pd.to_datetime(["2026-09-01T01:00:00Z"] * 3, utc=True),
        "__symbol__": ["A/USD:USD", "B/USD:USD", "C/USD:USD"],
        "side_name": ["long"] * 3,
        "final_score": [.9, .8, .7], "base_rank42": [.9, .8, .7],
        "conditional_consensus_rank": [.9, .8, .7], "upstream": [.9, .8, .7],
        "ordinary_shadow_consensus_rank": [.9, .8, .7], "correctness_rank": [.5] * 3,
    })
    gate = rows.loc[:, ["candidate_id", "__decision_ts__", "side_name"]].copy()
    gate["router_score"] = [.9, .8, .7]
    gate["router50_eligible"] = True
    gate["router_fraction"] = .5
    gate["router_timestamp_ordinal"] = [1, 2, 3]
    gate["router_timestamp_count"] = 6
    return P8UCoordinateScores(router_population=gate, current_coordinates=rows, bcf_coordinates=rows.copy())


class _Upstream:
    def score_staged_coordinates(self, **_kwargs):
        return _coordinates()

    def map_c0_coordinates(self, coordinates):
        result = coordinates.current_coordinates.loc[:, IDENTITY].copy()
        result["bcf_mc1_expected_bps"] = [100., 75., 10.]
        result["current_mc1_expected_bps"] = [100., 75., 10.]
        result["auction_priority_bps"] = result["bcf_mc1_expected_bps"]
        result["dual_mc1_admitted"] = result["bcf_mc1_expected_bps"].ge(50.)
        return result


class _C1:
    def score(self, *, current_coordinates, **_kwargs):
        result = current_coordinates.copy()
        result["bcf_mc1_expected_bps"] = [150., 10., 60.]
        result["current_mc1_expected_bps"] = [150., 10., 60.]
        result["auction_priority_bps"] = result["bcf_mc1_expected_bps"]
        result["dual_mc1_admitted"] = result["bcf_mc1_expected_bps"].ge(50.)
        result["c1_lva_adapter"] = "fake"
        return result


def test_no_order_stack_enforces_both_then_c0_then_c1() -> None:
    stack = P8UC0C1NoOrderStack(upstream=_Upstream(), c1_adapter=_C1())
    result = stack.score_staged(
        router_features=pd.DataFrame(), routed_features=pd.DataFrame(),
        c1_snapshots=pd.DataFrame(),
    )
    selected = result.selected_scores.set_index("candidate_id")
    assert selected.loc["both", "portfolio_tier"] == 2
    assert selected.loc["c0", "portfolio_tier"] == 1
    assert selected.loc["c1", "portfolio_tier"] == 0
    assert selected.loc["both", "auction_priority_bps"] == 100.0
    assert selected.loc["c1", "auction_priority_bps"] == 60.0
