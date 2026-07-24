#!/usr/bin/env python3
"""Materialize the causal market-wide negative-residual feature contract.

The negative-residual library is market-wide by construction: every generated
column is broadcast to each asset after it is computed from the point-in-time
cross-sectional panel.  This script stores one canonical timestamp-indexed
copy, suitable for overlay training without reintroducing a sparse benchmark
symbol join.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from extreme_price_movements.features_negative_residuals import (
    NEGATIVE_RESIDUAL_FEATURE_SCHEMA_VERSION,
    NEGATIVE_RESIDUAL_META_FEATURE_KEYS,
    add_negative_residual_features,
)
from scripts.backfill_negative_residual_temporal_mechanisms import _load_source_panels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--start", required=True, help="Inclusive UTC timestamp")
    parser.add_argument("--end", required=True, help="Inclusive UTC timestamp")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    start = pd.Timestamp(args.start, tz="UTC")
    end = pd.Timestamp(args.end, tz="UTC")
    # 30d causal robust normalization plus a five-day persistence window.
    panels = _load_source_panels(args.source_root, start=start - pd.Timedelta(days=45))
    generated = add_negative_residual_features(
        panels,
        requested_feature_keys=NEGATIVE_RESIDUAL_META_FEATURE_KEYS,
        cfg={"feature_bars_per_hour": 1},
    )
    missing = sorted(set(NEGATIVE_RESIDUAL_META_FEATURE_KEYS) - set(generated))
    if missing:
        raise RuntimeError(f"Residual feature generation was incomplete: {missing}")
    columns = panels[NEGATIVE_RESIDUAL_META_FEATURE_KEYS[0]].columns
    benchmark = next((name for name in columns if str(name).upper() == "BTC/USD:USD"), None)
    if benchmark is None:
        raise KeyError("Source universe has no BTC/USD:USD benchmark contract")
    result = pd.DataFrame(
        {
            name: panels[name][benchmark].astype(np.float32, copy=False)
            for name in NEGATIVE_RESIDUAL_META_FEATURE_KEYS
        },
        index=panels[NEGATIVE_RESIDUAL_META_FEATURE_KEYS[0]].index,
    )
    result = result.loc[(result.index >= start) & (result.index <= end)]
    if result.empty or result.isna().all().any():
        missing_columns = result.columns[result.isna().all()].tolist()
        raise ValueError(f"Market feature panel has empty coverage: {missing_columns}")
    # Generated columns are broadcast market states. Assert that contract
    # rather than silently selecting an arbitrary symbol column.
    check_columns = [columns[0], benchmark, columns[-1]]
    for name in NEGATIVE_RESIDUAL_META_FEATURE_KEYS:
        values = panels[name].loc[result.index, check_columns].to_numpy(np.float32)
        reference = values[:, :1]
        if not np.allclose(values, reference, equal_nan=True):
            raise ValueError(f"{name} is not market-wide/broadcast as contracted")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(args.output, index=True, compression="zstd")
    manifest = {
        "schema": "negative_residual_market_feature_panel_v1",
        "schema_version": NEGATIVE_RESIDUAL_FEATURE_SCHEMA_VERSION,
        "source_root": str(args.source_root),
        "start": str(result.index.min()),
        "end": str(result.index.max()),
        "rows": int(len(result)),
        "feature_count": int(len(result.columns)),
        "feature_schema_hash": hashlib.sha256(
            "\n".join(result.columns).encode("utf-8")
        ).hexdigest(),
        "benchmark_contract": benchmark,
        "source_history_buffer_days": 45,
        "contract": "Causal, market-wide pre-entry features only; no labels, residual outcomes, model scores, or episode calendar fields are read.",
    }
    args.output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
