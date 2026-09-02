#!/usr/bin/env python3
"""Materialize one frozen side-residual feature contract into a compact handoff.

This is a compute cache only: it reads the canonical static feature endpoint
once, joins the frozen selected feature union by UTC timestamp/symbol, and
keeps the label/outcome ledger byte-identical.  It lets parameter-only
ablations reuse exact features without rehydrating the entire store per arm.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import pandas as pd

from scripts.run_meta_v9_ev_mapped_side_residual_ablation import _augment_from_feature_store


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--handoff", type=Path, required=True)
    parser.add_argument("--scored-ledger", type=Path, required=True)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.selection_manifest.read_text(encoding="utf-8"))
    selected = {
        str(side): [str(feature) for feature in values]
        for side, values in dict(manifest.get("selected_features") or {}).items()
    }
    if not {"long", "short"}.issubset(selected):
        raise ValueError("Selection manifest must contain both long and short contracts")
    # This anchor is causally recomputed from the loaded base score by the
    # residual runner for each evaluation contract; it is not a static-store
    # feature and must not be materialized from a future-inclusive cache.
    derived_anchors = {"base_score_rank_pct_train_prior"}
    features = [
        feature
        for feature in dict.fromkeys([*selected["long"], *selected["short"]])
        if feature not in derived_anchors
    ]
    frame = pd.read_parquet(args.handoff)
    hydrated = _augment_from_feature_store(frame, args.feature_dir, features)
    missing = [feature for feature in features if feature not in hydrated.columns]
    if missing:
        raise ValueError(f"Canonical store did not materialize selected features: {missing}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_handoff = args.output_dir / "train_meta_regime_handoff.parquet"
    temporary = output_handoff.with_suffix(".parquet.tmp")
    hydrated.to_parquet(temporary, index=False, compression="zstd")
    temporary.replace(output_handoff)
    output_ledger = args.output_dir / "s52_trailing_regime_scored_ledger.parquet"
    if output_ledger.exists():
        output_ledger.unlink()
    try:
        os.link(args.scored_ledger, output_ledger)
    except OSError:
        shutil.copy2(args.scored_ledger, output_ledger)
    report = {
        "schema": "meta_residual_handoff_feature_cache_v1",
        "source_handoff": str(args.handoff),
        "source_ledger": str(args.scored_ledger),
        "selection_manifest": str(args.selection_manifest),
        "feature_dir": str(args.feature_dir),
        "rows": int(len(hydrated)),
        "selected_features": selected,
        "selected_union_count": int(len(features)),
        "runner_derived_anchors": sorted(derived_anchors),
        "null_rate": {
            feature: float(pd.to_numeric(hydrated[feature], errors="coerce").isna().mean())
            for feature in features
        },
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
