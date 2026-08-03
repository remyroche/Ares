#!/usr/bin/env python3
"""Freeze and boundary-probe the Jan--Jul 2022 inverse PI product bindings.

The mapping is deliberately source-bound.  Unlike the USD-linear PF mapper it
does not ask the current exchange catalogue to infer a substitute product: the
only acceptable ID for each symbol is the exact PI ID frozen in the candidate
source.  This preserves a separate, non-promotable research lineage.
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

from extreme_price_movements.data_store import (  # noqa: E402
    _fetch_kraken_futures_charts_ohlcv,
    make_perp_exchange,
)
from scripts.materialize_historical_backcast_exact1m_stage import (  # noqa: E402
    INVERSE_CAUSAL_POPULATION_LINEAGE,
    INVERSE_CAUSAL_SCOPE,
    INVERSE_GRID_POPULATION_LINEAGE,
    INVERSE_GRID_PRODUCT_LINEAGE,
    INVERSE_GRID_SCOPE,
    INVERSE_PARENT_ASSIGNMENT_SOURCE,
    INVERSE_PARENT_POLICY_KEYS,
    INVERSE_PRODUCTS,
)


EXPECTED_PRODUCT_BY_SYMBOL = dict(INVERSE_PRODUCTS)
BOOTSTRAP_STAGE_EVIDENCE_SCOPE = "inverse_pi_market_grid_bootstrap_research_not_oof"
CAUSAL_STAGE_EVIDENCE_SCOPE = "inverse_pi_market_grid_causal_features_research_not_oof"
# Compatibility alias for the already materialized bootstrap request contract.
STAGE_EVIDENCE_SCOPE = BOOTSTRAP_STAGE_EVIDENCE_SCOPE


def _lineage_contract(stage_manifest: dict[str, Any]) -> dict[str, Any]:
    """Resolve one of the two explicitly isolated inverse-PI populations."""

    stage_scope = str(stage_manifest.get("evidence_scope") or "")
    if stage_scope == BOOTSTRAP_STAGE_EVIDENCE_SCOPE:
        return {
            "stage_evidence_scope": BOOTSTRAP_STAGE_EVIDENCE_SCOPE,
            "row_evidence_scope": INVERSE_GRID_SCOPE,
            "population_lineage": INVERSE_GRID_POPULATION_LINEAGE,
            "bootstrap_barrier_data_acquisition_only": True,
            "requires_parent_binding": False,
            "mapping_source": "source_frozen_exact_inverse_pi_binding_no_catalogue_fallback",
        }
    if stage_scope == CAUSAL_STAGE_EVIDENCE_SCOPE:
        return {
            "stage_evidence_scope": CAUSAL_STAGE_EVIDENCE_SCOPE,
            "row_evidence_scope": INVERSE_CAUSAL_SCOPE,
            "population_lineage": INVERSE_CAUSAL_POPULATION_LINEAGE,
            "bootstrap_barrier_data_acquisition_only": False,
            "requires_parent_binding": True,
            "mapping_source": "causal_source_exact_inverse_pi_binding_no_catalogue_fallback",
        }
    raise ValueError("stage is not an allowlisted inverse PI research lineage")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_stage(
    stage_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame, Path, Path]:
    manifest_path = stage_dir / "manifest.json"
    staged_path = stage_dir / "staged_candidates.parquet"
    download_path = stage_dir / "download_candidates.parquet"
    manifest = _json(manifest_path)
    if manifest.get("schema") != "historical_backcast_exact1m_request_stage_v2":
        raise ValueError("stage must use historical_backcast_exact1m_request_stage_v2")
    contract = _lineage_contract(manifest)
    if manifest.get("candidate_population_lineage") != contract["population_lineage"]:
        raise ValueError("stage has an unexpected inverse PI population lineage")
    if manifest.get("product_lineage") != INVERSE_GRID_PRODUCT_LINEAGE:
        raise ValueError("stage has an unexpected inverse PI product lineage")
    if (
        manifest.get("bootstrap_barrier_data_acquisition_only")
        is not contract["bootstrap_barrier_data_acquisition_only"]
    ):
        raise ValueError("stage bootstrap-barrier contract disagrees with its lineage")
    if contract["requires_parent_binding"]:
        expected_parent_binding = {
            "assignment_source": INVERSE_PARENT_ASSIGNMENT_SOURCE,
            "archetype_policy_key": "parent",
            "side_policy_keys": INVERSE_PARENT_POLICY_KEYS,
        }
        if manifest.get("parent_policy_binding") != expected_parent_binding:
            raise ValueError("causal inverse PI stage must bind explicit side-parent policy")
    outputs = manifest.get("outputs") or {}
    if _sha256(staged_path) != outputs.get("staged_candidates", {}).get("sha256"):
        raise ValueError("staged candidate hash does not match stage manifest")
    if _sha256(download_path) != outputs.get("download_candidates", {}).get("sha256"):
        raise ValueError("download candidate hash does not match stage manifest")
    required = {
        "candidate_id",
        "decision_timestamp",
        "symbol",
        "source_product_id",
        "source_contract_family",
        "source_product_lineage",
        "candidate_population_lineage",
        "bootstrap_barrier_data_acquisition_only",
        "archetype_policy_key",
        "side_name",
    }
    if contract["requires_parent_binding"]:
        required.update({"product_id", "policy_archetype_assignment_source"})
    candidates = pd.read_parquet(staged_path)
    missing = sorted(required - set(candidates.columns))
    if missing:
        raise ValueError(f"inverse PI stage is missing product-binding columns: {missing}")
    candidates["decision_timestamp"] = pd.to_datetime(
        candidates["decision_timestamp"], utc=True, errors="raise"
    )
    if candidates["candidate_id"].duplicated().any():
        raise ValueError("stage has duplicate candidate IDs")
    if set(candidates["evidence_scope"].astype(str)) != {
        contract["row_evidence_scope"]
    }:
        raise ValueError("stage candidate rows have an unexpected evidence scope")
    if set(candidates["source_contract_family"].astype(str)) != {"PI"}:
        raise ValueError("stage candidate rows must bind PI contracts")
    if set(candidates["source_product_lineage"].astype(str)) != {
        INVERSE_GRID_PRODUCT_LINEAGE
    }:
        raise ValueError("stage candidate rows have an unexpected product lineage")
    if set(candidates["candidate_population_lineage"].astype(str)) != {
        contract["population_lineage"]
    }:
        raise ValueError("stage candidate rows have an unexpected population lineage")
    bootstrap_values = candidates["bootstrap_barrier_data_acquisition_only"].fillna(
        not contract["bootstrap_barrier_data_acquisition_only"]
    ).astype(bool)
    if bootstrap_values.ne(contract["bootstrap_barrier_data_acquisition_only"]).any():
        raise ValueError("stage candidate rows disagree with the lineage bootstrap-barrier contract")
    if contract["requires_parent_binding"]:
        if set(candidates["archetype_policy_key"].dropna().astype(str)) != {"parent"}:
            raise ValueError("causal inverse PI candidates must bind the parent policy key")
        if set(candidates["policy_archetype_assignment_source"].dropna().astype(str)) != {
            INVERSE_PARENT_ASSIGNMENT_SOURCE
        }:
            raise ValueError("causal inverse PI candidates have an invalid parent policy binding")
        signed_keys = candidates["side_name"].astype(str).map(
            INVERSE_PARENT_POLICY_KEYS
        )
        if signed_keys.isna().any():
            raise ValueError("causal inverse PI candidates have an invalid policy side")
        if not (
            candidates["side_name"].astype(str)
            + "__"
            + candidates["archetype_policy_key"].astype(str)
        ).eq(signed_keys).all():
            raise ValueError("causal inverse PI candidates have an invalid signed parent policy key")
        expected_products = candidates["symbol"].astype(str).map(EXPECTED_PRODUCT_BY_SYMBOL)
        if expected_products.isna().any() or not candidates["product_id"].astype(str).eq(
            expected_products
        ).all():
            raise ValueError("causal inverse PI candidates have an invalid exact product binding")
    return manifest, contract, candidates, staged_path, download_path


def _product_rows(
    candidates: pd.DataFrame,
    *,
    exchange: Any,
    probe_hours: int,
    mapping_source: str,
) -> list[dict[str, Any]]:
    probe_span = pd.Timedelta(hours=int(probe_hours))
    # Validate the complete source binding before making any network request.
    # A malformed late-sorting symbol must never cause partial product probing
    # to look like a valid map.
    for symbol, group in candidates.groupby("symbol", sort=True):
        expected_product_id = EXPECTED_PRODUCT_BY_SYMBOL.get(str(symbol))
        source_ids = set(group["source_product_id"].dropna().astype(str))
        if expected_product_id is None:
            raise ValueError(f"{symbol} is not in the frozen inverse PI product grid")
        if source_ids != {expected_product_id}:
            raise ValueError(
                f"{symbol} does not bind its exact frozen PI product: "
                f"expected={expected_product_id} got={sorted(source_ids)}"
            )
    if set(candidates["symbol"].astype(str)) != set(EXPECTED_PRODUCT_BY_SYMBOL):
        raise ValueError("inverse PI stage must cover the complete fixed product grid")
    rows: list[dict[str, Any]] = []
    for symbol, group in candidates.groupby("symbol", sort=True):
        symbol = str(symbol)
        expected_product_id = EXPECTED_PRODUCT_BY_SYMBOL.get(symbol)
        assert expected_product_id is not None
        first = group["decision_timestamp"].min().floor("min")
        last = group["decision_timestamp"].max().floor("min")
        probe_counts: list[int] = []
        for anchor in (first, last):
            probe = _fetch_kraken_futures_charts_ohlcv(
                exchange,
                symbol,
                int(anchor.value // 10**6),
                int((anchor + probe_span).value // 10**6),
                timeframe="1m",
                tick_type="trade",
                product_id=expected_product_id,
            )
            probe_counts.append(int(len(probe)))
        if min(probe_counts) <= 0:
            raise ValueError(
                f"{symbol}/{expected_product_id} has no trade candles in a "
                f"{probe_hours}h boundary probe: {probe_counts}"
            )
        collateral = symbol.rsplit(":", 1)[-1]
        rows.append(
            {
                "symbol": symbol,
                "product_id": expected_product_id,
                "contract_family": "PI",
                "quote": "USD",
                "settle": collateral,
                "collateral": collateral,
                "linear": False,
                "inverse": True,
                "active_in_current_catalog": None,
                "first_staged_decision": first,
                "last_staged_decision": last,
                "first_probe_candles": probe_counts[0],
                "last_probe_candles": probe_counts[1],
                "mapping_source": mapping_source,
                "notional_return_comparable_to_usd_linear_pf": False,
            }
        )
    return rows


def run(stage_dir: Path, output_dir: Path, *, probe_hours: int = 24) -> dict[str, Any]:
    if probe_hours <= 0:
        raise ValueError("--probe-hours must be positive")
    stage_manifest, contract, candidates, staged_path, download_path = _validate_stage(
        stage_dir
    )
    exchange = make_perp_exchange()
    exchange.load_markets()
    product_map = pd.DataFrame(
        _product_rows(
            candidates,
            exchange=exchange,
            probe_hours=probe_hours,
            mapping_source=str(contract["mapping_source"]),
        )
    ).sort_values("symbol").reset_index(drop=True)
    requests = pd.read_parquet(download_path).merge(
        product_map[
            ["symbol", "product_id", "contract_family", "settle", "collateral", "inverse"]
        ],
        on="symbol",
        how="left",
        validate="many_to_one",
    )
    if requests["product_id"].isna().any():
        raise ValueError("product map does not cover every download request")
    output_dir.mkdir(parents=True, exist_ok=True)
    product_path = output_dir / "product_map.parquet"
    requests_path = output_dir / "download_candidates_with_product.parquet"
    product_map.to_parquet(product_path, index=False)
    requests.to_parquet(requests_path, index=False)
    manifest = {
        # Retain the established downstream schema and output names while
        # making the PI exception machine-readable rather than implicit.
        "schema": "kraken_historical_product_map_v1",
        "status": "frozen_and_boundary_probed",
        "allowed_contract": "frozen exact inverse PI product grid only",
        "inverse_pi_allowed": True,
        "fallback_mapping_allowed": False,
        "charts_trade_catalogue_url": "https://futures.kraken.com/api/charts/v1/trade",
        "probe_hours": int(probe_hours),
        "symbols": int(len(product_map)),
        "stage_manifest": {
            "path": str((stage_dir / "manifest.json").resolve()),
            "sha256": _sha256(stage_dir / "manifest.json"),
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
        "evidence_scope": contract["stage_evidence_scope"],
        "promotion_eligible": False,
        "candidate_population_lineage": contract["population_lineage"],
        "product_lineage": INVERSE_GRID_PRODUCT_LINEAGE,
        "bootstrap_barrier_data_acquisition_only": contract[
            "bootstrap_barrier_data_acquisition_only"
        ],
        "parent_policy_binding": (
            {
                "assignment_source": INVERSE_PARENT_ASSIGNMENT_SOURCE,
                "archetype_policy_key": "parent",
                "side_policy_keys": INVERSE_PARENT_POLICY_KEYS,
            }
            if contract["requires_parent_binding"]
            else None
        ),
        "inverse_contract_limitations": {
            "inverse": True,
            "collateral_varies_by_product": True,
            "notional_return_comparable_to_usd_linear_pf": False,
            "historical_spread_l2_available": False,
            "requires_separate_inverse_economic_contract_before_labels_or_replay": True,
            "must_not_be_pooled_with_usd_linear_pf_population": True,
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--probe-hours", type=int, default=24)
    args = parser.parse_args()
    manifest = run(args.stage_dir, args.output_dir, probe_hours=args.probe_hours)
    print(json.dumps(manifest, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
