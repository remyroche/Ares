from __future__ import annotations

import pandas as pd

from extreme_price_movements.p8u_e2_h4_giveback20_contract import (
    P8UE2H4Giveback20Contract,
    e2_selection_mask,
    h4_next_interval_modifier,
)


def test_contract_hashes_and_research_boundary_validate() -> None:
    contract = P8UE2H4Giveback20Contract.load()
    assert contract.payload["status"] == "CANONICAL_RESEARCH_NOT_LIVE"
    assert contract.payload["entry_authority"]["can_expand_entry_capacity"] is False


def test_e2_requires_both_q50_heads() -> None:
    frame = pd.DataFrame({
        "candidate_id": ["a", "b", "c", "d"],
        "h0_q50_pair_advantage_bps": [50.0, 49.99, 100.0, float("nan")],
        "h3_q50_pair_advantage_bps": [50.0, 100.0, 49.99, 100.0],
    })
    assert e2_selection_mask(frame).tolist() == [True, False, False, False]


def test_h4_authority_is_fixed_next_interval_only() -> None:
    assert h4_next_interval_modifier(0.0) == {
        "active": True,
        "activation_earlier": 0.5,
        "giveback_tighten": 0.2,
        "sl_tighten": 0.0,
        "effective_from_next_interval": True,
    }
    assert h4_next_interval_modifier(-0.1) == {
        "active": False,
        "activation_earlier": 0.0,
        "giveback_tighten": 0.0,
        "sl_tighten": 0.0,
        "effective_from_next_interval": True,
    }
    assert h4_next_interval_modifier(None)["active"] is False
