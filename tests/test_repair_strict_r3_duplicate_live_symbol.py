from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts/repair_strict_r3_duplicate_live_symbol.py"


def _module():
    spec = importlib.util.spec_from_file_location("duplicate_repair_test", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _state():
    return {
        "schema": "strict_r3_kraken_live_state_v1",
        "positions": [
            {"candidate_id": "old", "symbol": "PUMP/USD:USD", "exchange_symbol": "PUMP/USD:USD", "side": "long", "amount": 9700.0},
            {"candidate_id": "new", "symbol": "PUMP/USD:USD", "exchange_symbol": "PUMP/USD:USD", "side": "long", "amount": 9600.0},
        ],
    }


def test_duplicate_repair_plan_requires_the_exact_two_rows():
    module = _module()
    retained, retired = module.duplicate_symbol_repair_plan(
        state=_state(), symbol="PUMP/USD:USD", retain_candidate_id="old", retire_candidate_id="new"
    )
    assert retained["amount"] == 9700.0
    assert retired["amount"] == 9600.0


def test_duplicate_repair_plan_rejects_an_unexpected_candidate_id():
    module = _module()
    with pytest.raises(ValueError, match="candidate IDs"):
        module.duplicate_symbol_repair_plan(
            state=_state(), symbol="PUMP/USD:USD", retain_candidate_id="old", retire_candidate_id="other"
        )
