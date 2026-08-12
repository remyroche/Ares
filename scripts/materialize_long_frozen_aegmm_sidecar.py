#!/usr/bin/env python3
"""Project the user-approved frozen AE/GMM state on full long candidates.

The state is not refit.  Projection is row-independent, uses decision-time
raw store fields only, and outputs AE/GMM geometry fields alone (not any other
regime-representation surface).
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.features_gmm_ae import transform_ae_gmm_features
from extreme_price_movements.packb_static_point_feature_loader import (
    _feature_contract_digest, discover_causal_feature_universe, freeze_feature_contract,
)
from extreme_price_movements.stage_i_production_data_adapter import make_static_pit_feature_loader


SIDE = "long"


def _subset(contract, fields):
    fields = sorted(map(str, fields))
    return replace(contract, feature_columns=tuple(fields), feature_contract_sha256=_feature_contract_digest(
        feature_columns=fields, candidate_universe_sha256=contract.candidate_universe_sha256,
        source_schema_sha256=contract.source_schema_sha256, raw_allowlist_sha256=contract.raw_allowlist_sha256,
        generator_registry_sha256=contract.generator_registry_sha256,
        store_scan_manifest_sha256=contract.store_scan_manifest_sha256,
        coverage_profile_sha256=contract.coverage_profile_sha256,
        min_exact_key_coverage=contract.min_exact_key_coverage,
        min_non_null_feature_coverage=contract.min_non_null_feature_coverage,
        max_feature_columns=contract.max_feature_columns,
        coverage_admission_rejections=contract.coverage_admission_rejections,
    ))


def _ledger(ledger_path: Path, panel_glob: Path) -> pd.DataFrame:
    frame = pd.read_parquet(ledger_path, columns=["candidate_id", "__ts__", "side_name"])
    frame = frame.loc[frame.side_name.astype(str).str.lower().eq(SIDE), ["candidate_id", "__ts__"]].copy()
    symbols = duckdb.sql(
        "SELECT candidate_id, any_value(__symbol__) AS __symbol__ FROM read_parquet(?) GROUP BY candidate_id",
        params=[str(panel_glob)],
    ).df()
    frame = frame.merge(symbols, on="candidate_id", how="left", validate="one_to_one", sort=False)
    if frame["__symbol__"].isna().any() or frame.candidate_id.duplicated().any():
        raise ValueError("long identity map is incomplete or duplicate")
    frame["__symbol__"] = frame["__symbol__"].astype(str).str.replace("_USD:USD", "/USD:USD", n=1, regex=False)
    frame["__ts__"] = pd.to_datetime(frame["__ts__"], utc=True)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--panel-glob", type=Path, default=ROOT / "data_perp/artifacts/full_universe_t2_t4_panel_20260801_v3/parts/*.parquet")
    parser.add_argument("--state", type=Path, default=ROOT / "data_perp/artifacts/s59_s52_july_oosbase_fullmeta_v9tail95_mlp_hierev_20260716_v2/ae_gmm_state/ae_gmm_state.pkl")
    parser.add_argument("--feature-store", default=str(ROOT / "data_perp/features/20260711_070000"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    state = pd.read_pickle(args.state)
    if not isinstance(state, dict) or not state.get("enabled"):
        raise ValueError("frozen AE/GMM state is disabled or invalid")
    state = dict(state)
    source_temporal = str(state.get("temporal_feature_contract") or "unspecified")
    state["temporal_feature_contract"] = "row_independent_v1"
    inputs = list(map(str, state.get("feature_columns", [])))
    if not inputs:
        raise ValueError("frozen AE/GMM state has no ordered raw input contract")
    ledger = _ledger(args.ledger, args.panel_glob)
    universe = discover_causal_feature_universe(ledger, feature_store_dir=args.feature_store)
    available = set(universe.feature_columns)
    store_fields = sorted(set(inputs).intersection(available).difference({"side"}))
    if len(store_fields) / len(inputs) < 0.50:
        raise ValueError("less than 50% of frozen AE/GMM input fields are available in current causal store")
    raw_contract = _subset(freeze_feature_contract(
        universe, min_exact_key_coverage=0.0, min_non_null_feature_coverage=0.0, max_feature_columns=None,
    ), store_fields)
    loader = make_static_pit_feature_loader(
        feature_store_dir=args.feature_store, feature_contract=raw_contract,
        max_rows_per_batch=4_000, max_columns_per_read=min(256, len(store_fields)), verify_frozen_schema=True,
    )
    raw = loader(ledger, store_fields)
    source = raw.reindex(columns=inputs)
    if "side" in inputs:
        source["side"] = 1.0
    transformed = transform_ae_gmm_features(source, state, index=raw.index, prefix="aegmm_")
    transformed = transformed.replace([np.inf, -np.inf], np.nan)
    # Constants are not model inputs; they cannot carry information and often
    # arise from deliberately disabled sequential state dynamics.
    fields = [name for name in transformed if transformed[name].notna().all() and transformed[name].std() > 1e-8]
    if not fields:
        raise ValueError("frozen AE/GMM projection has no finite varying outputs")
    result = pd.concat([raw.loc[:, ["candidate_id"]], transformed.loc[:, fields]], axis=1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(args.output, index=False, compression="zstd")
    manifest = {
        "schema": "long_frozen_aegmm_sidecar_v1", "side": SIDE, "rows": len(result),
        "state": str(args.state), "source_temporal_contract": source_temporal,
        "projection_temporal_contract": "row_independent_v1",
        "state_input_count": len(inputs), "store_input_count": len(store_fields),
        "state_input_overlap": len(store_fields) / len(inputs), "output_fields": fields,
        "output_coverage_min": float(result.loc[:, fields].notna().mean().min()),
        "representation_scope": "only frozen AE/GMM outputs; no leaf, DAE/GMM-derived transition, or other regime representations",
    }
    args.output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
