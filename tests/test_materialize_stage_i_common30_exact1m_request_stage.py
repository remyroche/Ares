from __future__ import annotations

import json
import sys

import pandas as pd
import pytest

from scripts import materialize_stage_i_common30_exact1m_request_stage as stage
from scripts import materialize_kraken_historical_product_map as product_map


def _manifest(root) -> None:
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "schema": "exact_tp6_sl4_h12_r3_relabel_v2",
                "status": "complete",
                "complete": True,
                "candidate_source_kind": "packb",
            }
        )
    )


def _row(candidate_id: str, ts: str, side: str, symbol: str) -> dict[str, object]:
    signal = pd.Timestamp(ts)
    return {
        "candidate_id": candidate_id,
        "__ts__": signal,
        "__symbol__": symbol,
        "side_name": side,
        "__decision_ts__": signal + pd.Timedelta(hours=1),
                "kraken_minute_symbol": symbol.replace("/", "_"),
        "label_valid": False,
        "target_invalid": True,
    }


def _source(tmp_path):
    root = tmp_path / "packb"
    root.mkdir()
    _manifest(root)
    for month in stage.DEFAULT_MONTHS:
        for side in stage.SIDES:
            part = root / "parts" / f"month={month}"
            part.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(
                [
                    _row(f"{month}-{side}-btc", f"{month}-01T00:00:00Z", side, "BTC/USD:USD"),
                    _row(f"{month}-{side}-eth", f"{month}-01T01:00:00Z", side, "ETH/USD:USD"),
                ]
            ).to_parquet(part / f"side={side}.parquet", index=False)
    universe = tmp_path / "common30.txt"
    universe.write_text("# frozen universe\nBTC/USD:USD\n")
    return root, universe


def _run(monkeypatch, root, universe, output) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "materialize_stage_i_common30_exact1m_request_stage.py",
            "--packb-root", str(root),
            "--universe-path", str(universe),
            "--output-dir", str(output),
        ],
    )
    assert stage.main() == 0


def test_stage_preserves_packb_identity_and_source_times(tmp_path, monkeypatch) -> None:
    root, universe = _source(tmp_path)
    output = tmp_path / "stage"
    _run(monkeypatch, root, universe, output)
    staged = pd.read_parquet(output / "staged_candidates.parquet")
    assert len(staged) == 8
    assert staged["candidate_id"].is_unique
    assert set(staged["candidate_id"]) == {
        f"{month}-{side}-btc" for month in stage.DEFAULT_MONTHS for side in stage.SIDES
    }
    assert staged["decision_timestamp"].eq(staged["signal_timestamp"] + pd.Timedelta(hours=1)).all()
    assert staged["path_end_exclusive"].eq(staged["decision_timestamp"] + pd.Timedelta(hours=12)).all()
    download = pd.read_parquet(output / "download_candidates.parquet")
    assert set(download["symbol"]) == {"BTC/USD:USD"}
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["schema"] == stage.SCHEMA
    assert manifest["source"]["universe"]["sha256"] == stage._sha256(universe)
    assert manifest["outputs"]["staged_candidates"]["sha256"] == stage._sha256(output / "staged_candidates.parquet")


def test_stage_fails_when_source_decision_timestamp_is_not_preserved(tmp_path) -> None:
    root, universe = _source(tmp_path)
    path = root / "parts" / "month=2026-01" / "side=long.parquet"
    broken = pd.read_parquet(path)
    broken.loc[0, "__decision_ts__"] = pd.Timestamp("2026-01-01T02:00:00Z")
    broken.to_parquet(path, index=False)
    with pytest.raises(ValueError, match=r"signal-close \+1h"):
        stage.materialize(packb_root=root, universe_path=universe, output_dir=tmp_path / "stage")


class _Exchange:
    def load_markets(self):
        return {}

    def market(self, symbol):
        return {"id": "PF_XBTUSD", "quote": "USD", "settle": "USD", "linear": True, "inverse": False, "active": True}


class _Response:
    def raise_for_status(self):
        return None

    def json(self):
        return ["PF_XBTUSD"]


class _Session:
    def get(self, *args, **kwargs):
        return _Response()


def test_product_map_accepts_stage_i_hash_bound_stage(tmp_path, monkeypatch) -> None:
    root, universe = _source(tmp_path)
    stage_dir = tmp_path / "stage"
    stage.materialize(packb_root=root, universe_path=universe, output_dir=stage_dir)
    monkeypatch.setattr(product_map, "make_perp_exchange", lambda: _Exchange())
    monkeypatch.setattr(product_map, "_public_data_session", lambda: _Session())
    monkeypatch.setattr(product_map, "_fetch_kraken_futures_charts_ohlcv", lambda *args, **kwargs: pd.DataFrame({"close": [1.0]}))
    monkeypatch.setattr(sys, "argv", ["materialize_kraken_historical_product_map.py", "--stage-dir", str(stage_dir), "--output-dir", str(tmp_path / "products")])
    assert product_map.main() == 0
    requests = pd.read_parquet(tmp_path / "products" / "download_candidates_with_product.parquet")
    assert set(requests["product_id"]) == {"PF_XBTUSD"}
    manifest = json.loads((tmp_path / "products" / "manifest.json").read_text())
    assert manifest["stage_manifest"]["schema"] == stage.SCHEMA


def test_product_map_refuses_to_overwrite_stage_i_freeze(tmp_path, monkeypatch) -> None:
    root, universe = _source(tmp_path)
    stage_dir = tmp_path / "stage"
    stage.materialize(packb_root=root, universe_path=universe, output_dir=stage_dir)
    output = tmp_path / "products"
    output.mkdir()
    (output / "prior_freeze.txt").write_text("immutable")
    monkeypatch.setattr(sys, "argv", ["materialize_kraken_historical_product_map.py", "--stage-dir", str(stage_dir), "--output-dir", str(output)])
    with pytest.raises(FileExistsError, match="Stage-I product-map"):
        product_map.main()
