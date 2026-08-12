from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.backfill_kraken_frozen_contract_inputs import _symbols_from_json


def test_symbols_loader_accepts_frozen_source_map(tmp_path: Path) -> None:
    path = tmp_path / 'manifest.json'
    path.write_text(json.dumps({'source_map': {'B/USD:USD': None, 'A/USD:USD': 'A_USD'}}))
    assert _symbols_from_json(path) == ['A/USD:USD', 'B/USD:USD']


def test_symbols_loader_accepts_inventory_rows(tmp_path: Path) -> None:
    path = tmp_path / 'manifest.json'
    path.write_text(json.dumps({'symbols': [{'perp_symbol': 'B/USD:USD'}, {'perp_symbol': 'A/USD:USD'}]}))
    assert _symbols_from_json(path) == ['A/USD:USD', 'B/USD:USD']


def test_symbols_loader_rejects_duplicates(tmp_path: Path) -> None:
    path = tmp_path / 'manifest.json'; path.write_text(json.dumps(['A', 'A']))
    with pytest.raises(ValueError, match='duplicate'):
        _symbols_from_json(path)
