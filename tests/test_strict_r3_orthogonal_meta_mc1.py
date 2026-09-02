from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

from extreme_price_movements.portfolio_policy_replay import (
    PortfolioPolicyParams,
    _controlled_perps_rank_sizing,
)


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "orthogonal_meta_mc1",
    ROOT / "scripts" / "run_strict_r3_orthogonal_meta_mc1.py",
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_mc1_never_accepts_resolved_semantic_or_policy_columns() -> None:
    prohibited = MODULE.PROHIBITED_META_LABEL_COLUMNS
    assert {
        "policy_net_bps",
        "policy_label_available_ts",
        "semantic_composite",
        "semantic_tbm_event",
    }.issubset(prohibited)
    # Target-free O3 score coordinates have semantic provenance in their
    # *names*, but are model outputs and therefore are not raw labels.
    assert "om__o3_calibrated_residual_semantic__consensus_rank" not in prohibited


def test_controlled_perps_sizing_matches_frozen_rank_formula() -> None:
    params = PortfolioPolicyParams(
        perp_default_leverage=7.0,
        max_total_wallet_allocation_pct=0.80,
        max_position_wallet_pct=0.15,
        max_position_quote_notional=1_000_000_000.0,
        rank_multiplier_min=1.0,
        rank_multiplier_max=1.0,
        rank_size_power=1.0,
    )
    sizing = _controlled_perps_rank_sizing(
        wallet_value=1_000.0,
        adjusted_rank_score=1.0,
        final_threshold=0.0,
        params=params,
        available_wallet_value=800.0,
        remaining_allocated_capital=800.0,
        rank_size_power=1.0,
    )
    expected = 1_000.0 * 7.0**1.5 / 100.0
    assert np.isclose(sizing["perp_effective_leverage"], 7.0)
    assert np.isclose(sizing["size_after_liquidity"], expected)
