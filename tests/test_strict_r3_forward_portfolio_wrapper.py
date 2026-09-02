"""Regression coverage for the forward portfolio wrapper's local adapter."""

from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "replay_strict_r3_forward_portfolio.py"


def _module():
    spec = importlib.util.spec_from_file_location("strict_r3_forward_portfolio", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_forward_wrapper_uses_frozen_global_auction_contract() -> None:
    module = _module()
    params = module._frozen_portfolio_params(
        threshold=0.0,
        perp_leverage=7.0,
        margin_slot_wallet_fraction=0.10,
        strategy_ids=("strict_r3_short",),
    )
    assert params.capacity_mode == "pre_leverage_wallet"
    assert params.max_concurrent_positions == 8
    assert params.max_new_entries_per_bar == 2
    assert params.max_total_wallet_allocation_pct == 0.80
    assert params.perp_default_leverage == 7.0
    assert params.margin_slot_wallet_fraction == 0.10
    assert params.strategy_ids == ("strict_r3_short",)
