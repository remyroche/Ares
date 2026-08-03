#!/usr/bin/env python3
"""Freeze Kraken Futures product IDs for a historical exact-1m request stage.

Only current-catalog, USD-settled linear PF products are accepted.  Each
product must also return trade candles around the earliest and latest staged
decision dates.  Inverse PI contracts and inferred ``PF_*`` fallbacks are
rejected rather than silently mixed into the research lineage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.data_store import (
    _fetch_kraken_futures_charts_ohlcv,
    _public_data_session,
    make_perp_exchange,
)


SUPPORTED_STAGE_SCHEMAS = {
    "historical_backcast_exact1m_request_stage_v2",
    "stage_i_common30_exact1m_request_stage_v1",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--probe-hours", type=int, default=24)
    args = parser.parse_args()
    if args.probe_hours <= 0:
        raise ValueError("--probe-hours must be positive")

    stage_manifest_path = args.stage_dir / "manifest.json"
    staged_path = args.stage_dir / "staged_candidates.parquet"
    download_path = args.stage_dir / "download_candidates.parquet"
    stage_manifest = _json(stage_manifest_path)
    if stage_manifest.get("schema") not in SUPPORTED_STAGE_SCHEMAS:
        raise ValueError(
            "stage must use one of " + ", ".join(sorted(SUPPORTED_STAGE_SCHEMAS))
        )
    outputs = stage_manifest.get("outputs") or {}
    if _sha256(staged_path) != outputs.get("staged_candidates", {}).get("sha256"):
        raise ValueError("staged candidate hash does not match stage manifest")
    if _sha256(download_path) != outputs.get("download_candidates", {}).get("sha256"):
        raise ValueError("download candidate hash does not match stage manifest")
    # The Stage-I request is a frozen backfill boundary.  Refuse to replace a
    # product assignment once it has been written; a changed catalogue must be
    # represented by a new, explicitly named request/product-map stage.
    if (
        stage_manifest.get("schema") == "stage_i_common30_exact1m_request_stage_v1"
        and args.output_dir.exists()
        and any(args.output_dir.iterdir())
    ):
        raise FileExistsError(
            f"refusing to overwrite non-empty Stage-I product-map output: {args.output_dir}"
        )

    candidates = pd.read_parquet(
        staged_path, columns=["candidate_id", "decision_timestamp", "symbol"]
    )
    candidates["decision_timestamp"] = pd.to_datetime(
        candidates["decision_timestamp"], utc=True, errors="raise"
    )
    if candidates["candidate_id"].duplicated().any():
        raise ValueError("stage has duplicate candidate IDs")

    exchange = make_perp_exchange()
    exchange.load_markets()
    catalogue_response = _public_data_session().get(
        "https://futures.kraken.com/api/charts/v1/trade",
        timeout=30,
        headers={"User-Agent": "Ares historical product lineage audit"},
    )
    catalogue_response.raise_for_status()
    catalogue_payload = catalogue_response.json()
    if not isinstance(catalogue_payload, list):
        raise ValueError("Kraken Charts trade catalogue did not return a product list")
    charts_products = {
        str(value).strip() for value in catalogue_payload if str(value).strip()
    }
    probe_span = pd.Timedelta(hours=int(args.probe_hours))
    product_rows: list[dict[str, Any]] = []
    for symbol, rows in candidates.groupby("symbol", sort=True):
        try:
            market = exchange.market(str(symbol))
        except Exception:
            market = None
        if market is None:
            base = str(symbol).split("/", 1)[0].upper()
            base = "XBT" if base == "BTC" else base
            exact_catalogue_id = f"PF_{base}USD"
            if exact_catalogue_id not in charts_products:
                raise ValueError(
                    f"{symbol} is absent from both the current CCXT catalogue and "
                    "the exact Kraken Charts PF trade catalogue"
                )
            product_id = exact_catalogue_id
            settle = "USD"
            quote = "USD"
            linear = True
            inverse = False
            active_in_current_catalog = False
            mapping_source = "kraken_charts_trade_catalog_exact_pf_match"
        else:
            product_id = str(market.get("id") or "").strip()
            settle = str(market.get("settle") or "").upper()
            quote = str(market.get("quote") or "").upper()
            linear = bool(market.get("linear"))
            inverse = bool(market.get("inverse"))
            active_in_current_catalog = bool(market.get("active", True))
            mapping_source = "live_krakenfutures_ccxt_catalog_no_fallback"
        if (
            not product_id.startswith("PF_")
            or settle != "USD"
            or quote != "USD"
            or not linear
            or inverse
            or not str(symbol).endswith(":USD")
        ):
            raise ValueError(
                f"{symbol} is not an accepted USD-linear PF lineage: "
                f"id={product_id} quote={quote} settle={settle} "
                f"linear={linear} inverse={inverse}"
            )
        first = rows["decision_timestamp"].min().floor("min")
        last = rows["decision_timestamp"].max().floor("min")
        probe_counts: list[int] = []
        for anchor in (first, last):
            probe = _fetch_kraken_futures_charts_ohlcv(
                exchange,
                str(symbol),
                int(anchor.value // 10**6),
                int((anchor + probe_span).value // 10**6),
                timeframe="1m",
                tick_type="trade",
                product_id=product_id,
            )
            probe_counts.append(int(len(probe)))
        if min(probe_counts) <= 0:
            raise ValueError(
                f"{symbol}/{product_id} has no trade candles in a "
                f"{args.probe_hours}h boundary probe: {probe_counts}"
            )
        product_rows.append(
            {
                "symbol": str(symbol),
                "product_id": product_id,
                "contract_family": "PF",
                "quote": quote,
                "settle": settle,
                "linear": linear,
                "inverse": inverse,
                "active_in_current_catalog": active_in_current_catalog,
                "first_staged_decision": first,
                "last_staged_decision": last,
                "first_probe_candles": probe_counts[0],
                "last_probe_candles": probe_counts[1],
                "mapping_source": mapping_source,
            }
        )

    product_map = pd.DataFrame(product_rows).sort_values("symbol").reset_index(drop=True)
    requests = pd.read_parquet(download_path)
    requests = requests.merge(
        product_map[["symbol", "product_id", "contract_family", "settle"]],
        on="symbol",
        how="left",
        validate="many_to_one",
    )
    if requests["product_id"].isna().any():
        raise ValueError("product map does not cover every download request")

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    product_path = output / "product_map.parquet"
    requests_path = output / "download_candidates_with_product.parquet"
    product_map.to_parquet(product_path, index=False)
    requests.to_parquet(requests_path, index=False)
    manifest = {
        "schema": "kraken_historical_product_map_v1",
        "status": "frozen_and_boundary_probed",
        "allowed_contract": "USD-settled linear PF only",
        "inverse_pi_allowed": False,
        "fallback_mapping_allowed": False,
        "charts_trade_catalogue_url": "https://futures.kraken.com/api/charts/v1/trade",
        "probe_hours": int(args.probe_hours),
        "symbols": int(len(product_map)),
        "stage_manifest": {
            "path": str(stage_manifest_path.resolve()),
            "sha256": _sha256(stage_manifest_path),
            "schema": stage_manifest["schema"],
        },
        "stage_candidates": {
            "path": str(staged_path.resolve()),
            "sha256": _sha256(staged_path),
        },
        "outputs": {
            "product_map": {
                "path": str(product_path.resolve()),
                "rows": int(len(product_map)),
                "sha256": _sha256(product_path),
            },
            "download_candidates_with_product": {
                "path": str(requests_path.resolve()),
                "rows": int(len(requests)),
                "sha256": _sha256(requests_path),
            },
        },
        "evidence_scope": "historical_frozen_backcast_request_provenance",
        "promotion_eligible": False,
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
