from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "scripts" / "backfill_kraken_oi_funding_sidecars.py"
SPEC = importlib.util.spec_from_file_location("kraken_oi_funding_sidecars", PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_slash_and_feature_store_symbols_have_one_pf_identity() -> None:
    slash = MODULE._parse_pf_product("BTC/USD:USD")
    feature = MODULE._parse_pf_product("BTC_USD:USD")
    assert slash.product_id == feature.product_id == "PF_XBTUSD"
    assert slash.sidecar_key == feature.sidecar_key == "BTC_USD_USD"


def test_frozen_source_map_manifest_is_a_supported_symbol_registry(tmp_path) -> None:
    manifest = tmp_path / "run_manifest.json"
    manifest.write_text(json.dumps({
        "source_map": {"BTC/USD:USD": "BTC_USDT", "ETH/USD:USD": "ETH_USDT"},
    }))
    products = MODULE._read_products(tmp_path, manifest)
    assert [product.product_id for product in products] == ["PF_XBTUSD", "PF_ETHUSD"]
