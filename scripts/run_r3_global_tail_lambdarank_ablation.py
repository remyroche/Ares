#!/usr/bin/env python3
"""Run frozen broad-R3 -> global-tail -> residual LambdaRank ablation cells.

This runner accepts an explicit target set over the fixed global-handoff
x={30%,40%} funnel.  It never re-ranks per bar, month, or side.  T3 is
accepted only on a complete exact-path substrate; a declared T1/T2-only run is
valid and records T3 as deliberately unrun rather than silently dropping it.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys
import tempfile
from typing import Any, Mapping

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.config import CFG
from extreme_price_movements.stage_i_feature_selection import resolve_stage_i_feature_universe
from extreme_price_movements.stage_iv_broad_to_tail import (
    StageIVPlan,
    canonical_lambdarank_params,
    run_stage_iv_pooled_global_handoff_ablation,
)
from extreme_price_movements.stage_iv_r3_tail_fitter import (
    current_r3_broad_tail_fitter,
    current_r3_class,
    current_r3_tree_params,
)


TARGETS = {
    "t1_exact_net": ("tail_target_t1_valid", "tail_target_net_grade_0_5"),
    "t2_atr_net": ("tail_target_t2_valid", "tail_target_atr_grade_0_5"),
    "t3_exact_tbm": ("tail_target_t3_valid", "tail_target_t3_first_touch_grade_0_4"),
}
FRACTIONS = (0.30, 0.40)
SCHEMA = "r3_global_tail_lambdarank_ablation_v2"


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_frame(
    source: Path, *, selected_targets: tuple[str, ...] = tuple(TARGETS),
) -> tuple[pd.DataFrame, Mapping[str, Any]]:
    unknown = sorted(set(selected_targets).difference(TARGETS))
    if not selected_targets or unknown:
        raise ValueError(f"selected targets must be a non-empty subset of {sorted(TARGETS)}; unknown={unknown}")
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise ValueError("input substrate must have a complete immutable manifest")
    parts = [source / item for item in manifest.get("parts", [])]
    if not parts or any(not path.is_file() for path in parts):
        raise ValueError("input substrate has missing declared parts")
    frame = pd.concat((pd.read_parquet(path) for path in parts), ignore_index=True)
    required = {
        "candidate_id", "__ts__", "side_name", "label_available_ts", "label_valid", "p90_spread_bps",
        "exact_net_bps", "robust_clear_event_b25", "lower_touch_minute",
        *sum((list(TARGETS[target]) for target in selected_targets), []),
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"input substrate lacks required strict-R3/tail columns: {missing}")
    if not frame["label_valid"].astype(bool).all():
        raise ValueError("this runner expects the valid-label research substrate; invalid paths remain coverage-only")
    if frame["candidate_id"].duplicated().any():
        raise ValueError("candidate_id must be globally unique in the frozen substrate")
    if "t3_exact_tbm" in selected_targets and not frame["tail_target_t3_valid"].astype(bool).all():
        raise ValueError("T3 requires exact TP4/TP6/SL4/SL6 path fields for every comparison row")
    if not (pd.to_numeric(frame["p90_spread_bps"], errors="coerce") < 90.0).all():
        raise ValueError("input substrate contains an ineligible p90-spread row")
    frame["__ts__"] = pd.to_datetime(frame["__ts__"], utc=True)
    frame["label_available_ts"] = pd.to_datetime(frame["label_available_ts"], utc=True)
    return frame, manifest


def _read_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _features(
    spec: Mapping[str, Any], frame: pd.DataFrame, coverage: Mapping[str, float],
    aegmm_fields: set[str], *, side: str, layer: str,
) -> list[str]:
    entry = spec.get(side)
    if not isinstance(entry, Mapping) or not isinstance(entry.get(layer), list):
        raise ValueError(f"feature spec needs {side}.{layer} list")
    selected = list(dict.fromkeys(map(str, entry[layer])))
    raw_selected = [name for name in selected if not name.startswith("aegmm_")]
    if not raw_selected or len(raw_selected) > 40:
        raise ValueError(f"{side}.{layer} must contain 1..40 selected non-AE/GMM fields")
    missing_aegmm = sorted(aegmm_fields.difference(selected))
    if missing_aegmm:
        raise ValueError(f"{side}.{layer} must include every frozen AE/GMM output: {missing_aegmm}")
    allowed_layer = "meta" if layer == "meta" else "base"
    head = "shared_exact_net_residual" if allowed_layer == "meta" else None
    allowed = set(resolve_stage_i_feature_universe(CFG, layer=allowed_layer, side=side, head=head))
    # Frozen AE/GMM outputs are an explicitly disclosed representation
    # exception and are intentionally admitted to every model layer.
    invalid = [name for name in selected if name not in allowed and not name.startswith("aegmm_")]
    if invalid:
        raise ValueError(f"{side}.{layer} includes fields outside its configured layer namespace: {invalid}")
    absent = [name for name in selected if name not in frame.columns]
    if absent:
        raise ValueError(f"{side}.{layer} selected fields absent from substrate: {absent}")
    undercovered = [name for name in selected if float(coverage.get(name, 0.0)) < 0.90]
    if undercovered:
        raise ValueError(f"{side}.{layer} selected fields fail the 90% coverage gate: {undercovered}")
    constant = [name for name in selected if frame[name].nunique(dropna=True) < 2]
    if constant:
        raise ValueError(f"{side}.{layer} selected constant/null fields: {constant}")
    return selected


def _side_plan(
    frame: pd.DataFrame, *, side: str, fraction: float, target: str,
    feature_spec: Mapping[str, Any], coverage: Mapping[str, float], aegmm_fields: set[str],
    broad_params: Mapping[str, Any], ranker_params: Mapping[str, Any],
) -> StageIVPlan:
    local = frame.loc[frame.side_name.astype(str).str.lower().eq(side)].copy().reset_index(drop=True)
    valid_column, target_column = TARGETS[target]
    if not local[valid_column].astype(bool).all():
        raise ValueError(f"{side} has unavailable {target} rows; comparison population would drift")
    grade = pd.to_numeric(local[target_column], errors="coerce").to_numpy(np.float32)
    if not np.isfinite(grade).all() or (grade < 0).any() or (grade > 5).any():
        raise ValueError(f"{side} has invalid {target} grade")
    r3 = current_r3_class(
        robust_clear_event=local["robust_clear_event_b25"],
        lower_touch_minute=local["lower_touch_minute"], label_valid=local["label_valid"],
    )
    if (r3 < 0).any():
        raise AssertionError("valid R3 substrate unexpectedly yielded invalid target")
    broad = _features(feature_spec, local, coverage, aegmm_fields, side=side, layer="broad")
    tail = _features(feature_spec, local, coverage, aegmm_fields, side=side, layer="tail")
    meta = _features(feature_spec, local, coverage, aegmm_fields, side=side, layer="meta")
    return StageIVPlan(
        side=side, candidate_ids=local["candidate_id"].astype(str).tolist(), frame=local,
        base_target=r3, tail_target=grade, exact_net_bps=local["exact_net_bps"].to_numpy(np.float32),
        decision_timestamps=local["__ts__"], label_available_timestamps=local["label_available_ts"],
        broad_feature_names=broad, tail_feature_names=tail, meta_feature_names=meta,
        broad_params=current_r3_tree_params(broad_params),
        tail_params=canonical_lambdarank_params(ranker_params),
        meta_params=canonical_lambdarank_params(ranker_params),
        tail_fraction=fraction, broad_output_route="both", burn_in_months=2,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--feature-spec", type=Path, required=True)
    parser.add_argument("--long-r3-manifest", type=Path, required=True)
    parser.add_argument("--short-r3-manifest", type=Path, required=True)
    parser.add_argument("--ranker-params-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--targets", nargs="+", choices=tuple(TARGETS), default=tuple(TARGETS),
        help="tail targets to compare; omissions are explicitly recorded as unrun",
    )
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite immutable ablation directory: {args.output_dir}")
    selected_targets = tuple(args.targets)
    cell_order = tuple((fraction, target) for fraction in FRACTIONS for target in selected_targets)
    frame, input_manifest = _load_frame(args.input_dir, selected_targets=selected_targets)
    coverage_frame = pd.read_parquet(args.input_dir / "feature_coverage_audit.parquet")
    if not {"feature", "coverage"}.issubset(coverage_frame.columns):
        raise ValueError("input feature coverage audit is malformed")
    coverage = dict(zip(coverage_frame.feature.astype(str), coverage_frame.coverage.astype(float)))
    aegmm_fields = set(map(str, input_manifest.get("frozen_aegmm_output_columns", [])))
    if not aegmm_fields:
        raise ValueError("input substrate has no frozen AE/GMM outputs to feed every layer")
    feature_spec = _read_json(args.feature_spec)
    long_manifest, short_manifest = _read_json(args.long_r3_manifest), _read_json(args.short_r3_manifest)
    ranker_params = _read_json(args.ranker_params_json)
    if not isinstance(long_manifest.get("best_params"), Mapping) or not isinstance(short_manifest.get("best_params"), Mapping):
        raise ValueError("frozen R3 manifests lack best_params")
    # Write every cell to a same-filesystem staging directory.  A failed later
    # cell must never make a partial result look like an immutable six-cell
    # experiment, and final publication is an atomic directory rename.
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(tempfile.mkdtemp(prefix=f".{args.output_dir.name}.staging.", dir=args.output_dir.parent))
    cells: list[dict[str, Any]] = []
    for fraction, target in cell_order:
        cell_id = f"x{int(fraction * 100):02d}_{target}"
        plans = [
            _side_plan(frame, side="long", fraction=fraction, target=target, feature_spec=feature_spec, coverage=coverage, aegmm_fields=aegmm_fields,
                       broad_params=long_manifest["best_params"], ranker_params=ranker_params),
            _side_plan(frame, side="short", fraction=fraction, target=target, feature_spec=feature_spec, coverage=coverage, aegmm_fields=aegmm_fields,
                       broad_params=short_manifest["best_params"], ranker_params=ranker_params),
        ]
        result = run_stage_iv_pooled_global_handoff_ablation(plans, fitter=current_r3_broad_tail_fitter())
        cell_dir = staging_dir / cell_id
        cell_dir.mkdir()
        result.predictions.to_parquet(cell_dir / "strict_oof_predictions.parquet", index=False)
        result.metrics_without_admission.to_parquet(cell_dir / "pooled_global_metrics.parquet", index=False)
        for side, value in result.side_results.items():
            value.fold_provenance.to_parquet(cell_dir / f"fold_provenance_{side}.parquet", index=False)
        (cell_dir / "manifest.json").write_text(json.dumps(result.manifest, indent=2, default=str) + "\n", encoding="utf-8")
        cells.append({"cell_id": cell_id, "tail_fraction": fraction, "tail_target": target, "manifest": result.manifest})
    manifest = {
        "schema": SCHEMA, "status": "complete", "cell_order": [item["cell_id"] for item in cells],
        "targets_requested": list(selected_targets),
        "targets_unrun": [target for target in TARGETS if target not in selected_targets],
        "ranking": "one pooled-global rank after side-local predecessor-resolved bps mapping; never timestamp/month/side reranked",
        "input_dir": str(args.input_dir), "input_manifest_sha256": _sha256_file(args.input_dir / "manifest.json"),
        "feature_spec_sha256": _sha256_file(args.feature_spec),
        "long_r3_manifest_sha256": _sha256_file(args.long_r3_manifest),
        "short_r3_manifest_sha256": _sha256_file(args.short_r3_manifest),
        "ranker_params_sha256": _sha256_file(args.ranker_params_json), "cells": cells,
    }
    (staging_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8")
    staging_dir.replace(args.output_dir)
    print(json.dumps({"status": "complete", "output_dir": str(args.output_dir), "cells": [item["cell_id"] for item in cells]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
