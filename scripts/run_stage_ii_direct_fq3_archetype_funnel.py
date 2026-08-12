#!/usr/bin/env python3
"""Run the bounded native-score Stage-II direct-FQ3 archetype comparison.

This is a development runner.  It accepts an already materialised, immutable
direct Stage-I ledger plus its causal/path fields.  It neither refits the base
nor applies a bps map before the three-class meta correction.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.stage_ii_direct_fq3_bridge import (
    StageIIDirectFQ3Candidate,
    StageIIDirectFQ3Spec,
    run_stage_ii_direct_fq3_archetype_funnel,
)
from extreme_price_movements.stage_ii_meta_archetypes import StageIIMetaArchetypeConfig


def _load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("specification must be a JSON object")
    return value


def _spec(raw: dict) -> tuple[StageIIDirectFQ3Spec, tuple[StageIIDirectFQ3Candidate, ...]]:
    model = raw.get("model_params")
    candidates_raw = raw.get("candidates")
    if not isinstance(model, dict) or not isinstance(candidates_raw, list):
        raise ValueError("specification requires model_params and candidates")
    values = dict(raw)
    values.pop("candidates", None)
    spec = StageIIDirectFQ3Spec(**values)
    candidates = []
    for value in candidates_raw:
        if not isinstance(value, dict) or not isinstance(value.get("archetype_config"), dict):
            raise ValueError("every candidate needs archetype_config and causal_feature_cols")
        candidates.append(StageIIDirectFQ3Candidate(
            candidate_id=str(value.get("candidate_id", "")),
            archetype_config=StageIIMetaArchetypeConfig(**value["archetype_config"]),
            causal_feature_cols=tuple(map(str, value.get("causal_feature_cols", ()))),
        ))
    return spec, tuple(candidates)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direct-stage-i-ledger", type=Path, required=True)
    parser.add_argument("--spec-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        parser.error("--output-dir must be a new immutable directory")
    spec, candidates = _spec(_load(args.spec_json))
    result = run_stage_ii_direct_fq3_archetype_funnel(
        pd.read_parquet(args.direct_stage_i_ledger), spec=spec, candidates=candidates,
    )
    args.output_dir.mkdir(parents=True)
    result.candidate_audit.to_parquet(args.output_dir / "candidate_audit.parquet", index=False, compression="zstd")
    for arm in result.arms:
        root = args.output_dir / arm.candidate_id / arm.arm
        root.mkdir(parents=True)
        arm.oof_predictions.to_parquet(root / "strict_oof_predictions.parquet", index=False, compression="zstd")
        arm.metrics.to_parquet(root / "pooled_global_metrics.parquet", index=False, compression="zstd")
        arm.contributions.to_parquet(root / "selected_contributions.parquet", index=False, compression="zstd")
        arm.admission_audit.to_parquet(root / "causal_21d_map_audit.parquet", index=False, compression="zstd")
        arm.fold_audit.to_parquet(root / "fold_audit.parquet", index=False, compression="zstd")
        (root / "feature_contract.json").write_text(json.dumps({"ordered_meta_features": arm.feature_names}, indent=2) + "\n")
    if result.archetype_oof_features is not None:
        result.archetype_oof_features.to_parquet(args.output_dir / "selected_archetype_oof_features.parquet", index=False, compression="zstd")
    manifest = dict(result.manifest)
    manifest.update({"status": "complete", "specification": asdict(spec), "selected_features": list(result.selected_features)})
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    print(json.dumps({"status": "complete", "selected_candidate_id": result.selected_candidate_id, "selected_arm": result.selected_arm, "output_dir": str(args.output_dir)}, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
