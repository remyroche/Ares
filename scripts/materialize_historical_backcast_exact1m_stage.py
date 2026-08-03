#!/usr/bin/env python3
"""Freeze historical backcast candidates before targeted exact-1m collection.

This stage is intentionally research-only.  It converts frozen diagnostic
backcast rows into immutable signal/decision identities and a minimal
``timestamp,symbol`` downloader input.  It does not upgrade a future-trained
backcast to OOF evidence or claim deployed-policy geometry/spread parity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq


REQUIRED_COLUMNS = {
    "__ts__",
    "__symbol__",
    "side_name",
    "__barrier_pct__",
    "archetype_policy_key",
    "selected_for_monitor",
    "evidence_scope",
}
OPTIONAL_PROVENANCE_COLUMNS = {
    "base_score",
    "historical_rank",
    "policy_archetype_assignment_source",
    "score_meta_base_soft_label",
    "side",
    "policy_archetype_assignment_source",
    # The inverse PI research population is deliberately separate from the
    # frozen-base backcast.  Keep its source-native binding intact so a later
    # downloader cannot infer or substitute another contract family.
    "source_candidate_id",
    "source_product_id",
    "source_contract_family",
    "source_product_lineage",
    "candidate_population_lineage",
    "bootstrap_barrier_data_acquisition_only",
}

FROZEN_BACKCAST_SCOPE = "frozen_backcast_diagnostic"
INVERSE_GRID_SCOPE = "inverse_pi_market_grid_bootstrap_research"
INVERSE_GRID_POPULATION_LINEAGE = "jan_jul_2022_inverse_pi_market_grid_bootstrap_v1"
INVERSE_GRID_PRODUCT_LINEAGE = "kraken_inverse_pi_exact_product_binding_v1"
INVERSE_CAUSAL_SCOPE = "inverse_pi_market_grid_causal_features_research"
INVERSE_CAUSAL_POPULATION_LINEAGE = "jan_jul_2022_inverse_pi_market_grid_causal_features_v1"
INVERSE_PARENT_ASSIGNMENT_SOURCE = "explicit_deployed_side_parent_inverse_grid"
INVERSE_PARENT_POLICY_KEYS = {"long": "long__parent", "short": "short__parent"}
INVERSE_PRODUCTS = {
    "BTC/USD:BTC": "PI_XBTUSD",
    "ETH/USD:ETH": "PI_ETHUSD",
    "LTC/USD:LTC": "PI_LTCUSD",
    "XRP/USD:XRP": "PI_XRPUSD",
    "BCH/USD:BCH": "PI_BCHUSD",
}


def _stage_lineage_contract(
    candidates: pd.DataFrame,
    *,
    population_lineage: str = "",
) -> dict[str, Any]:
    """Return the narrowly allowed source-population contract.

    The second branch intentionally does not broaden the legacy frozen-backcast
    contract: it only admits the fixed Jan--Jul 2022 inverse PI bootstrap
    population, which is isolated from the USD-linear research lineage.
    """

    requested_population_lineage = str(population_lineage).strip() or None
    scopes = set(candidates["evidence_scope"].dropna().astype(str))
    if scopes == {FROZEN_BACKCAST_SCOPE}:
        if requested_population_lineage is not None:
            raise ValueError(
                "--population-lineage requires a source-declared alternate "
                "population; the legacy frozen backcast has none"
            )
        return {
            "lineage": "historical_frozen_backcast_exact1m_research_only",
            "stage_evidence_scope": "frozen_backcast_diagnostic_not_oof",
            "candidate_population_lineage": None,
            "product_lineage": None,
            "bootstrap_barrier_data_acquisition_only": False,
            "economics_contract": "frozen_or_current_spread_counterfactual_only",
            "return_unit": "decimal_notional_return",
            "parent_policy_binding": None,
            "preserve_feature_columns": False,
        }
    if scopes == {INVERSE_CAUSAL_SCOPE}:
        required = {
            "candidate_population_lineage",
            "source_product_lineage",
            "product_id",
            "bootstrap_barrier_data_acquisition_only",
            "policy_archetype_assignment_source",
        }
        missing = sorted(required - set(candidates.columns))
        if missing:
            raise ValueError(f"causal inverse PI source is missing lineage columns: {missing}")
        population = set(candidates["candidate_population_lineage"].dropna().astype(str))
        product = set(candidates["source_product_lineage"].dropna().astype(str))
        if population != {INVERSE_CAUSAL_POPULATION_LINEAGE}:
            raise ValueError("causal inverse PI source has an unexpected population lineage")
        if product != {INVERSE_GRID_PRODUCT_LINEAGE}:
            raise ValueError("causal inverse PI source has an unexpected product lineage")
        if (
            requested_population_lineage is not None
            and requested_population_lineage != INVERSE_CAUSAL_POPULATION_LINEAGE
        ):
            raise ValueError(
                "--population-lineage must exactly match the source-declared "
                "causal inverse PI population lineage"
            )
        if candidates["bootstrap_barrier_data_acquisition_only"].fillna(True).astype(bool).any():
            raise ValueError("causal inverse PI candidates must not carry a bootstrap barrier")
        if set(candidates["archetype_policy_key"].dropna().astype(str)) != {"parent"}:
            raise ValueError("causal inverse PI candidates must bind the parent policy key")
        if set(candidates["policy_archetype_assignment_source"].dropna().astype(str)) != {
            INVERSE_PARENT_ASSIGNMENT_SOURCE
        }:
            raise ValueError("causal inverse PI candidates have an invalid parent policy binding")
        if not candidates["side_name"].astype(str).isin(INVERSE_PARENT_POLICY_KEYS).all():
            raise ValueError("causal inverse PI candidates have an invalid policy side")
        expected_products = candidates["symbol"].astype(str).map(INVERSE_PRODUCTS)
        if expected_products.isna().any() or not candidates["product_id"].astype(str).eq(expected_products).all():
            raise ValueError("causal inverse PI candidates have an invalid exact product binding")
        return {
            "lineage": "historical_inverse_pi_market_grid_exact1m_research_only",
            "stage_evidence_scope": "inverse_pi_market_grid_causal_features_research_not_oof",
            "candidate_population_lineage": INVERSE_CAUSAL_POPULATION_LINEAGE,
            "product_lineage": INVERSE_GRID_PRODUCT_LINEAGE,
            "bootstrap_barrier_data_acquisition_only": False,
            "economics_contract": "inverse_quote_notional_current_spread_counterfactual_only",
            "return_unit": "quote_notional_price_return_not_inverse_collateral_roe",
            "parent_policy_binding": {
                "assignment_source": INVERSE_PARENT_ASSIGNMENT_SOURCE,
                "archetype_policy_key": "parent",
                "side_policy_keys": INVERSE_PARENT_POLICY_KEYS,
            },
            "preserve_feature_columns": True,
        }
    if scopes != {INVERSE_GRID_SCOPE}:
        raise ValueError(f"Unexpected evidence scopes: {sorted(scopes)}")
    required = {
        "candidate_population_lineage",
        "source_product_lineage",
        "source_product_id",
        "source_contract_family",
        "bootstrap_barrier_data_acquisition_only",
    }
    missing = sorted(required - set(candidates.columns))
    if missing:
        raise ValueError(f"inverse PI source is missing lineage columns: {missing}")
    source_population_lineages = set(
        candidates["candidate_population_lineage"].dropna().astype(str)
    )
    if source_population_lineages != {
        INVERSE_GRID_POPULATION_LINEAGE
    }:
        raise ValueError("inverse PI source has an unexpected population lineage")
    source_population_lineage = next(iter(source_population_lineages))
    if (
        requested_population_lineage is not None
        and requested_population_lineage != source_population_lineage
    ):
        raise ValueError(
            "--population-lineage must exactly match the source-declared "
            "inverse PI population lineage"
        )
    if set(candidates["source_product_lineage"].dropna().astype(str)) != {
        INVERSE_GRID_PRODUCT_LINEAGE
    }:
        raise ValueError("inverse PI source has an unexpected product lineage")
    if set(candidates["source_contract_family"].dropna().astype(str)) != {"PI"}:
        raise ValueError("inverse PI source must bind PI contracts exactly")
    if not candidates["bootstrap_barrier_data_acquisition_only"].fillna(False).astype(bool).all():
        raise ValueError("inverse PI bootstrap barrier must be acquisition-only")
    return {
        "lineage": "historical_inverse_pi_market_grid_exact1m_research_only",
        "stage_evidence_scope": "inverse_pi_market_grid_bootstrap_research_not_oof",
        # Never substitute a caller/default label for source-native lineage.
        # The optional CLI value is an equality assertion, not a relabeling API.
        "candidate_population_lineage": source_population_lineage,
        "product_lineage": INVERSE_GRID_PRODUCT_LINEAGE,
        "bootstrap_barrier_data_acquisition_only": True,
        "economics_contract": "frozen_or_current_spread_counterfactual_only",
        "return_unit": "decimal_notional_return",
        "parent_policy_binding": None,
        "preserve_feature_columns": False,
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
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--decision-delay-hours", type=int, default=1)
    parser.add_argument("--horizon-minutes", type=int, default=720)
    parser.add_argument("--signal-start", default="")
    parser.add_argument("--signal-end-exclusive", default="")
    parser.add_argument(
        "--require-symbol-suffix",
        default="",
        help="Fail-closed historical contract filter, for example ':USD'.",
    )
    parser.add_argument(
        "--population-lineage",
        default="",
        help=(
            "Optional equality assertion for a source-declared alternate "
            "candidate population. Empty preserves the legacy frozen-backcast "
            "default and never relabels source rows."
        ),
    )
    args = parser.parse_args()

    if args.decision_delay_hours < 0:
        raise ValueError("--decision-delay-hours must be non-negative")
    if args.horizon_minutes <= 0:
        raise ValueError("--horizon-minutes must be positive")

    shards = sorted(args.candidate_root.glob("*.parquet"))
    if not shards:
        raise FileNotFoundError(f"No candidate shards under {args.candidate_root}")

    signal_start = (
        pd.Timestamp(args.signal_start).tz_convert("UTC") if args.signal_start else None
    )
    signal_end = (
        pd.Timestamp(args.signal_end_exclusive).tz_convert("UTC")
        if args.signal_end_exclusive
        else None
    )

    frames: list[pd.DataFrame] = []
    source_rows_total = 0
    source_rows_effective = 0
    source_records: list[dict[str, Any]] = []
    for shard in shards:
        schema = set(pq.read_schema(shard).names)
        missing = sorted(REQUIRED_COLUMNS - schema)
        if missing:
            raise ValueError(f"{shard} missing required columns: {missing}")
        shard_rows = pq.ParquetFile(shard).metadata.num_rows
        shard_sha256 = _sha256(shard)
        source_rows_total += shard_rows
        read_columns = sorted(REQUIRED_COLUMNS | (OPTIONAL_PROVENANCE_COLUMNS & schema))
        frame = pd.read_parquet(shard, columns=read_columns)
        # The final causal inverse grid is already a complete pre-entry feature
        # population.  Preserve its columns byte-for-column through the exact
        # request stage; the source scope is later allowlisted and its lineage,
        # product binding, and parent-policy binding are validated centrally.
        if set(frame["evidence_scope"].dropna().astype(str)) == {
            INVERSE_CAUSAL_SCOPE
        }:
            frame = pd.read_parquet(shard)
        frame["source_row_number"] = range(len(frame))
        selected_before_time = int(
            frame["selected_for_monitor"].fillna(False).astype(bool).sum()
        )
        frame = frame.loc[
            frame["selected_for_monitor"].fillna(False).astype(bool)
        ].copy()
        frame["signal_timestamp"] = pd.to_datetime(
            frame.pop("__ts__"), utc=True, errors="raise"
        )
        if signal_start is not None:
            frame = frame.loc[frame["signal_timestamp"] >= signal_start]
        if signal_end is not None:
            frame = frame.loc[frame["signal_timestamp"] < signal_end]
        if args.require_symbol_suffix:
            frame = frame.loc[
                frame["__symbol__"].astype(str).str.endswith(args.require_symbol_suffix)
            ]
        selected_after_time = int(len(frame))
        if frame.empty:
            continue
        frame["source_shard_sha256"] = shard_sha256
        frame["source_shard_path"] = str(shard.resolve())
        frames.append(frame)
        source_rows_effective += shard_rows
        source_records.append(
            {
                "path": str(shard.resolve()),
                "rows": shard_rows,
                "selected_rows_before_time_filter": selected_before_time,
                "selected_rows_after_time_filter": selected_after_time,
                "sha256": shard_sha256,
            }
        )

    if not frames:
        raise ValueError("No selected candidates remain after the signal-time filter")
    candidates = pd.concat(frames, ignore_index=True)
    candidates["decision_timestamp"] = candidates["signal_timestamp"] + pd.Timedelta(
        hours=args.decision_delay_hours
    )
    candidates["symbol"] = candidates.pop("__symbol__").astype(str)
    lineage_contract = _stage_lineage_contract(
        candidates, population_lineage=args.population_lineage
    )

    logical_identity = (
        candidates["signal_timestamp"].astype(str)
        + "|"
        + candidates["symbol"]
        + "|"
        + candidates["side_name"].astype(str)
        + "|"
        + candidates["archetype_policy_key"].astype(str)
        + "|"
        + candidates["__barrier_pct__"].map(lambda value: f"{float(value):.17g}")
    )
    logical_collisions = logical_identity.duplicated(keep=False)
    if bool(logical_collisions.any()):
        collision_rows = candidates.loc[
            logical_collisions,
            [
                "signal_timestamp",
                "symbol",
                "side_name",
                "archetype_policy_key",
                "__barrier_pct__",
                "source_shard_path",
                "source_row_number",
            ],
        ]
        raise ValueError(
            "Historical candidate logical-identity collision; source rows must "
            f"be resolved explicitly:\n{collision_rows.head(20).to_string(index=False)}"
        )
    source_identity = (
        candidates["source_shard_sha256"]
        + "|"
        + candidates["source_row_number"].astype(str)
        + "|"
        + logical_identity
    )
    candidates["candidate_id"] = source_identity.map(
        lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest()
    )
    if bool(candidates["candidate_id"].duplicated().any()):
        raise ValueError("Source-native candidate_id collision")
    candidates = candidates.sort_values(
        ["decision_timestamp", "symbol", "side_name", "candidate_id"]
    )
    candidates["path_end_exclusive"] = candidates["decision_timestamp"] + pd.Timedelta(
        minutes=args.horizon_minutes
    )
    candidates["lineage"] = lineage_contract["lineage"]
    candidates["execution_parity_claim"] = False
    candidates["promotion_eligible"] = False

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    staged_path = output / "staged_candidates.parquet"
    path_map_path = output / "candidate_path_map.parquet"
    download_path = output / "download_candidates.parquet"
    candidates.to_parquet(staged_path, index=False)
    path_map = candidates.rename(columns={"decision_timestamp": "timestamp"})[
        ["candidate_id", "timestamp", "symbol", "path_end_exclusive"]
    ]
    path_map.to_parquet(path_map_path, index=False)
    path_map[["timestamp", "symbol"]].drop_duplicates().to_parquet(
        download_path, index=False
    )

    manifest = {
        "schema": "historical_backcast_exact1m_request_stage_v2",
        "status": "request_population_frozen_before_download",
        "evidence_scope": lineage_contract["stage_evidence_scope"],
        "lineage": lineage_contract["lineage"],
        "execution_parity_claim": False,
        "promotion_eligible": False,
        "candidate_population_lineage": lineage_contract[
            "candidate_population_lineage"
        ],
        "product_lineage": lineage_contract["product_lineage"],
        "bootstrap_barrier_data_acquisition_only": lineage_contract[
            "bootstrap_barrier_data_acquisition_only"
        ],
        "economics_contract": lineage_contract["economics_contract"],
        "return_unit": lineage_contract["return_unit"],
        "parent_policy_binding": lineage_contract["parent_policy_binding"],
        "feature_columns_preserved": bool(lineage_contract["preserve_feature_columns"]),
        "signal_to_decision_hours": int(args.decision_delay_hours),
        "signal_filter_start": signal_start.isoformat() if signal_start is not None else None,
        "signal_filter_end_exclusive": (
            signal_end.isoformat() if signal_end is not None else None
        ),
        "required_symbol_suffix": args.require_symbol_suffix or None,
        "path_horizon_minutes": int(args.horizon_minutes),
        "path_interval": "[decision_timestamp, path_end_exclusive)",
        "same_minute_barrier_conflict_rule_required": "conservative_adverse_first",
        "historical_spread_l2_available": False,
        "known_geometry_limitation": (
            "pre-2025 rows do not carry the bit-exact deployed-policy "
            "__path_auxiliary_atr_fraction__ contract"
        ),
        "candidate_identity_contract": (
            "sha256(source_shard_sha256|source_row_number|signal|symbol|side|"
            "archetype|barrier_pct)"
        ),
        "logical_collision_policy": "fail_closed",
        "source_rows_total": int(source_rows_total),
        "source_rows_effective_shards": int(source_rows_effective),
        "selected_rows": int(len(candidates)),
        "distinct_symbols": int(candidates["symbol"].nunique()),
        "signal_start": candidates["signal_timestamp"].min().isoformat(),
        "signal_end": candidates["signal_timestamp"].max().isoformat(),
        "decision_start": candidates["decision_timestamp"].min().isoformat(),
        "path_end": candidates["path_end_exclusive"].max().isoformat(),
        "sources": source_records,
        "outputs": {
            "staged_candidates": {
                "path": str(staged_path.resolve()),
                "rows": int(len(candidates)),
                "sha256": _sha256(staged_path),
            },
            "candidate_path_map": {
                "path": str(path_map_path.resolve()),
                "rows": int(len(path_map)),
                "sha256": _sha256(path_map_path),
            },
            "download_candidates": {
                "path": str(download_path.resolve()),
                "rows": int(pq.ParquetFile(download_path).metadata.num_rows),
                "sha256": _sha256(download_path),
            },
        },
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
