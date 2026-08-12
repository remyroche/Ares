#!/usr/bin/env python3
"""Materialize the immutable compact hourly market-transition sidecar."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.market_transition_sidecar import (  # noqa: E402
    SCHEMA, TransitionSidecarConfig, build_market_transition_sidecar, resolve_spine_sources,
)

DEFAULT_PANEL = ROOT / "data_perp/artifacts/regime_multiview_panel_2022_2026_20260730_v2/multiview_regime_features.parquet"
DEFAULT_OUT = ROOT / "data_perp/artifacts/market_transition_sidecar_2022_2026_20260803_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def run(panel_path: Path = DEFAULT_PANEL, output: Path = DEFAULT_OUT) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    schema = pq.ParquetFile(panel_path).schema.names
    sources = resolve_spine_sources(schema)
    keep = ["source_utc", *sources.values(), *[name for name in schema if name.startswith("market_regime__state_p_")]]
    panel = pd.read_parquet(panel_path, columns=list(dict.fromkeys(keep)))
    sidecar, features = build_market_transition_sidecar(panel, source_columns=sources, config=TransitionSidecarConfig())
    sidecar.to_parquet(output / "market_transition_features.parquet", index=False)
    availability = pd.DataFrame({
        "feature": features,
        "coverage": [float(sidecar[name].notna().mean()) for name in features],
        "nonconstant": [bool(sidecar[name].nunique(dropna=True) > 1) for name in features],
    })
    availability["usable_90pct_nonconstant"] = availability.coverage.ge(.90) & availability.nonconstant
    availability.to_parquet(output / "feature_availability.parquet", index=False)
    (output / "manifest.json").write_text(json.dumps({
        "schema": SCHEMA,
        "status": "COMPLETED",
        "panel": str(panel_path),
        "panel_sha256": _sha256(panel_path),
        "source_columns": sources,
        "rows": len(sidecar),
        "time_start": str(sidecar.source_utc.min()),
        "time_end": str(sidecar.source_utc.max()),
        "feature_count": len(features),
        "causality": "one continuous hourly sidecar; robust references exclude current row; online BOCPD and EWMA covariance update sequentially; candidate consumers must backward-asof join source_utc <= decision timestamp",
        "diagnostic_only": "transition episodes/model failures are not feature inputs",
    }, indent=2) + "\n")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    print(run(args.panel, args.out))
