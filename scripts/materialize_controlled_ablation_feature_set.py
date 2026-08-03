#!/usr/bin/env python3
"""Materialise the fail-closed causal feature list for the target ablation.

The raw panel contract is a broad source registry.  This step narrows it to
the semantic ``ELIGIBLE_RESEARCH_CAUSAL`` base inputs from the lineage audit,
then removes columns that are absent for every candidate.  No target,
realised-path, cost or model-derived field can enter the runner by accident.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.controlled_target_supportive_ablation import validate_causal_raw_features


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def materialize(*, audit: Path, raw_panel: Path, output: Path) -> dict[str, object]:
    if output.exists():
        raise FileExistsError(output)
    manifest = pd.read_parquet(audit)
    eligible = manifest.loc[
        manifest.model_layer.eq("base") & manifest.eligibility_status.eq("ELIGIBLE_RESEARCH_CAUSAL"),
        "feature_name",
    ].astype(str).drop_duplicates().tolist()
    eligible = list(validate_causal_raw_features(eligible))
    panel = pd.read_parquet(raw_panel, columns=eligible)
    finite = np.isfinite(panel.to_numpy(dtype=float)).any(axis=0)
    excluded = [name for name, has_value in zip(eligible, finite, strict=True) if not has_value]
    features = [name for name in eligible if name not in set(excluded)]
    if not features:
        raise ValueError("semantic causal audit left no usable feature")
    payload = {
        "schema": "controlled_target_supportive_causal_feature_set_v1",
        "model_layer": "base_and_execution_raw_causal_inputs",
        "feature_columns": features,
        "feature_count": len(features),
        "semantic_audit_eligible_count": len(eligible),
        "excluded_all_missing": excluded,
        "audit_path": str(audit),
        "audit_sha256": sha256(audit),
        "raw_panel_path": str(raw_panel),
        "raw_panel_sha256": sha256(raw_panel),
        "selection_rule": "base-layer ELIGIBLE_RESEARCH_CAUSAL only; all-missing columns excluded; strict future/target semantic rejection",
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--raw-panel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(materialize(audit=args.audit, raw_panel=args.raw_panel, output=args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
