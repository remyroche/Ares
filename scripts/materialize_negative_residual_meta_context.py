#!/usr/bin/env python3
"""Materialize hardened market and family context on the production meta rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from extreme_price_movements.features_negative_residuals import (
    NEGATIVE_RESIDUAL_META_FEATURE_KEYS,
    negative_residual_feature_contract,
)
from extreme_price_movements.residual_state_family_features import (
    ResidualStateFamilyContract,
)


KEYS = ["__ts__", "__symbol__", "side_name", "archetype_policy_key"]


def run(args: argparse.Namespace) -> dict[str, object]:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows = pd.read_parquet(args.prepared_dataset, columns=KEYS)
    rows["__ts__"] = pd.to_datetime(rows["__ts__"], utc=True, errors="coerce")
    market = pd.read_parquet(
        args.market_features, columns=NEGATIVE_RESIDUAL_META_FEATURE_KEYS
    )
    market.index = pd.to_datetime(market.index, utc=True, errors="coerce")
    market = market.loc[~market.index.duplicated(keep="last")].sort_index()
    market.index.name = "__ts__"
    merged = rows.merge(market.reset_index(), on="__ts__", how="left", validate="many_to_one")

    payload = json.loads(args.family_contract.read_text())
    family_contract = ResidualStateFamilyContract.from_dict(payload)
    family = family_contract.transform(
        merged,
        merged["side_name"].astype(str),
        merged["archetype_policy_key"].astype(str),
    )
    output = pd.concat(
        [merged[KEYS].reset_index(drop=True), merged[NEGATIVE_RESIDUAL_META_FEATURE_KEYS], family],
        axis=1,
    )
    for column in output.columns:
        if column not in KEYS:
            output[column] = pd.to_numeric(output[column], errors="coerce").astype(np.float32)
    output.to_parquet(args.output, index=False, compression="zstd")
    matched = output[NEGATIVE_RESIDUAL_META_FEATURE_KEYS].notna().any(axis=1)
    active_columns = [name for name in family if name.endswith("_active")]
    computable_columns = [name for name in family if name.endswith("_computable")]
    manifest = {
        "schema": "negative_residual_meta_context_v1",
        "prepared_dataset": str(args.prepared_dataset),
        "market_feature_source": str(args.market_features),
        "output": str(args.output),
        "rows": int(len(output)),
        "timestamp_min": str(output["__ts__"].min()),
        "timestamp_max": str(output["__ts__"].max()),
        "market_feature_count": int(len(NEGATIVE_RESIDUAL_META_FEATURE_KEYS)),
        "family_feature_count": int(len(family.columns)),
        "market_context_matched_rows": int(matched.sum()),
        "market_context_match_rate": float(matched.mean()),
        "family_active_rows": int(output[active_columns].max(axis=1).gt(0).sum()),
        "family_computable_rows": int(output[computable_columns].max(axis=1).gt(0).sum()),
        "source_feature_contract_hash": negative_residual_feature_contract()["contract_hash"],
        "family_contract_hash": family_contract.contract_hash,
    }
    args.output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dataset", type=Path, required=True)
    parser.add_argument("--market-features", type=Path, required=True)
    parser.add_argument("--family-contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
