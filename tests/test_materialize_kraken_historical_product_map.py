from __future__ import annotations

import json
import sys

import pandas as pd
import pytest

from scripts import materialize_kraken_historical_product_map as product_map


class _Exchange:
    def __init__(self, *, inverse: bool = False):
        self.inverse = inverse

    def load_markets(self):
        return {}

    def market(self, symbol):
        return {
            "id": "PI_XBTUSD" if self.inverse else "PF_XBTUSD",
            "quote": "USD",
            "settle": "BTC" if self.inverse else "USD",
            "linear": not self.inverse,
            "inverse": self.inverse,
            "active": True,
        }


class _CatalogueResponse:
    def raise_for_status(self):
        return None

    def json(self):
        return ["PF_XBTUSD", "PI_XBTUSD"]


class _CatalogueSession:
    def get(self, *args, **kwargs):
        return _CatalogueResponse()


def _stage(tmp_path):
    stage_dir = tmp_path / "stage"
    stage_dir.mkdir()
    staged = pd.DataFrame(
        {
            "candidate_id": ["candidate-1"],
            "decision_timestamp": [pd.Timestamp("2024-01-01T01:00:00Z")],
            "symbol": ["BTC/USD:USD"],
        }
    )
    requests = staged.rename(columns={"decision_timestamp": "timestamp"})[
        ["timestamp", "symbol"]
    ]
    staged.to_parquet(stage_dir / "staged_candidates.parquet", index=False)
    requests.to_parquet(stage_dir / "download_candidates.parquet", index=False)
    manifest = {
        "schema": "historical_backcast_exact1m_request_stage_v2",
        "outputs": {
            "staged_candidates": {
                "sha256": product_map._sha256(stage_dir / "staged_candidates.parquet")
            },
            "download_candidates": {
                "sha256": product_map._sha256(stage_dir / "download_candidates.parquet")
            },
        },
    }
    (stage_dir / "manifest.json").write_text(json.dumps(manifest))
    return stage_dir


def test_product_map_freezes_pf_id_and_binds_stage(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stage_dir = _stage(tmp_path)
    monkeypatch.setattr(product_map, "make_perp_exchange", lambda: _Exchange())
    monkeypatch.setattr(
        product_map, "_public_data_session", lambda: _CatalogueSession()
    )
    monkeypatch.setattr(
        product_map,
        "_fetch_kraken_futures_charts_ohlcv",
        lambda *args, **kwargs: pd.DataFrame({"close": [1.0]}),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "materialize_kraken_historical_product_map.py",
            "--stage-dir",
            str(stage_dir),
            "--output-dir",
            str(tmp_path / "output"),
        ],
    )

    assert product_map.main() == 0
    frozen = pd.read_parquet(tmp_path / "output" / "product_map.parquet")
    assert frozen.loc[0, "product_id"] == "PF_XBTUSD"
    requests = pd.read_parquet(
        tmp_path / "output" / "download_candidates_with_product.parquet"
    )
    assert requests.loc[0, "product_id"] == "PF_XBTUSD"


def test_product_map_rejects_inverse_pi_lineage(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stage_dir = _stage(tmp_path)
    monkeypatch.setattr(
        product_map, "make_perp_exchange", lambda: _Exchange(inverse=True)
    )
    monkeypatch.setattr(
        product_map, "_public_data_session", lambda: _CatalogueSession()
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "materialize_kraken_historical_product_map.py",
            "--stage-dir",
            str(stage_dir),
            "--output-dir",
            str(tmp_path / "output"),
        ],
    )
    with pytest.raises(ValueError, match="not an accepted USD-linear PF lineage"):
        product_map.main()
