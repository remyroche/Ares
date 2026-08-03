from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "backfill_kraken_missing_ohlcv_gaps",
    ROOT / "scripts" / "backfill_kraken_missing_ohlcv_gaps.py",
)
assert SPEC and SPEC.loader
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


def test_symbol_file_is_exact_ordered_and_deduplicated(tmp_path: Path) -> None:
    path = tmp_path / "symbols.txt"
    path.write_text("# frozen cohort\nALGO/USD:USD\n\nBTC/USD:USD # context\nALGO/USD:USD\n")
    assert module._load_symbols_file(path) == ["ALGO/USD:USD", "BTC/USD:USD"]


def test_symbol_file_fails_closed_when_empty(tmp_path: Path) -> None:
    path = tmp_path / "symbols.txt"
    path.write_text("# no candidates\n\n")
    with pytest.raises(ValueError, match="contains no symbols"):
        module._load_symbols_file(path)
