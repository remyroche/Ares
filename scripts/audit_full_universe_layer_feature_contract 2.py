#!/usr/bin/env python3
"""Audit the strict, differentiated feature pools for full-universe T2/T4.

This deliberately reports *pools*, not a model feature list.  The sequential
runner selects a smaller side-local subset from each pool using its training
window only.  Keeping those two steps separate prevents a convenient but
invalid "all configured features" fallback.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.config import (
    DAILY_SR_BASE_FEATURE_KEYS,
    LONG_HORIZON_PERP_META_FEATURE_KEYS,
    MODEL_DIRECT_BASE_FEATURE_KEYS,
    MODEL_REGIME_COMPOSITE_META_FEATURE_KEYS,
    ORDERBOOK_BASE_FEATURE_KEYS,
    ORDERBOOK_META_FEATURE_KEYS,
    RESIDUAL_BASE_FEATURE_KEYS,
    RESIDUAL_META_FEATURE_KEYS,
    VOLUME_FREE_PERP_BASE_FEATURE_KEYS,
    VOLUME_FREE_PERP_META_FEATURE_KEYS,
)


def _dedupe(values: list[str]) -> list[str]:
    return list(dict.fromkeys(map(str, values)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    parts = sorted((args.panel / "parts").glob("*.parquet"))
    if not parts:
        raise FileNotFoundError(f"no panel parts below {args.panel}")

    # These are the proper existing config-layer keys.  Do not add a generic
    # shared pool: base stays asset-local and meta stays regime/contextual.
    base_allowed = _dedupe(
        MODEL_DIRECT_BASE_FEATURE_KEYS
        + RESIDUAL_BASE_FEATURE_KEYS
        + ORDERBOOK_BASE_FEATURE_KEYS
        + VOLUME_FREE_PERP_BASE_FEATURE_KEYS
        + DAILY_SR_BASE_FEATURE_KEYS
    )
    meta_allowed = _dedupe(
        MODEL_REGIME_COMPOSITE_META_FEATURE_KEYS
        + RESIDUAL_META_FEATURE_KEYS
        + ORDERBOOK_META_FEATURE_KEYS
        + VOLUME_FREE_PERP_META_FEATURE_KEYS
        + LONG_HORIZON_PERP_META_FEATURE_KEYS
    )
    configured_overlap = sorted(set(base_allowed).intersection(meta_allowed))
    # The legacy config contains a small order-book overlap.  This experiment
    # needs differentiated *trained* layers, so assign it deterministically to
    # meta (where order-book context is permitted) and remove it from the base
    # candidate pool before any side-local selection occurs.
    base_candidates = [key for key in base_allowed if key not in configured_overlap]
    meta_candidates = meta_allowed
    overlap = sorted(set(base_candidates).intersection(meta_candidates))
    if overlap:
        raise AssertionError(f"exclusive base/meta candidates overlap: {overlap}")

    schema = set(pd.read_parquet(parts[0]).columns)
    base_available = [key for key in base_candidates if key in schema]
    meta_available = [key for key in meta_candidates if key in schema]
    # Compute coverage on every materialised row, without treating imputation
    # as coverage.  This is the >=90% gate used before train-only selection.
    candidates = base_available + meta_available
    finite = dict.fromkeys(candidates, 0)
    total = 0
    for part in parts:
        frame = pd.read_parquet(part, columns=candidates)
        total += len(frame)
        for key in candidates:
            finite[key] += int(np.isfinite(frame[key].to_numpy(dtype=float, copy=False)).sum())
    coverage = {key: finite[key] / total for key in candidates}
    report = {
        "schema": "full_universe_layer_feature_contract_audit_v1",
        "panel": str(args.panel),
        "rows_audited": total,
        "selection_rule": "per-side, training-window-only rank screen followed by MDA; select a strict subset, never all candidates",
        "configured_raw_overlap": configured_overlap,
        "exclusive_assignment": "configured overlaps belong to meta; base excludes them before selection",
        "base": {
            "config_pools": ["MODEL_DIRECT_BASE_FEATURE_KEYS", "RESIDUAL_BASE_FEATURE_KEYS", "ORDERBOOK_BASE_FEATURE_KEYS", "VOLUME_FREE_PERP_BASE_FEATURE_KEYS", "DAILY_SR_BASE_FEATURE_KEYS"],
            "configured": len(base_allowed),
            "exclusive_candidates": len(base_candidates),
            "available": base_available,
            "coverage_ge_90pct": [key for key in base_available if coverage[key] >= 0.90],
        },
        "meta": {
            "config_pools": ["MODEL_REGIME_COMPOSITE_META_FEATURE_KEYS", "RESIDUAL_META_FEATURE_KEYS", "ORDERBOOK_META_FEATURE_KEYS", "VOLUME_FREE_PERP_META_FEATURE_KEYS", "LONG_HORIZON_PERP_META_FEATURE_KEYS"],
            "configured": len(meta_allowed),
            "exclusive_candidates": len(meta_candidates),
            "available": meta_available,
            "coverage_ge_90pct": [key for key in meta_available if coverage[key] >= 0.90],
        },
        "raw_feature_overlap": overlap,
        "coverage": coverage,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"rows": total, "base_available": len(base_available), "base_90": len(report["base"]["coverage_ge_90pct"]), "meta_available": len(meta_available), "meta_90": len(report["meta"]["coverage_ge_90pct"]), "overlap": len(overlap)}))


if __name__ == "__main__":
    main()
