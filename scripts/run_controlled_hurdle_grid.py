#!/usr/bin/env python3
"""Run the preregistered T4 net-hurdle grid on the frozen OOF supports."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

import pandas as pd

from extreme_price_movements.controlled_target_supportive_ablation import (
    AcceptanceGates,
    SUPPORT_STAGES,
    derive_economic_targets,
    pooled_global_top_k_metrics,
    support_columns,
)
from scripts.run_controlled_target_supportive_ablation import _arm_score


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(*, support_oof: Path, features_json: Path, output: Path, hurdle_grid: tuple[float, ...]) -> dict[str, object]:
    if output.exists():
        raise FileExistsError(output)
    payload = json.loads(features_json.read_text())
    features = payload.get("feature_columns") if isinstance(payload, dict) else payload
    features = list(features)
    frame = pd.read_parquet(support_oof)
    if "fold_order" not in frame or set(frame["fold_order"].unique()) != {1, 2}:
        raise ValueError("support OOF audit must contain exactly meta_train/meta_oos fold rows")
    train_base = frame[frame["fold_order"].eq(1)].copy()
    test_base = frame[frame["fold_order"].eq(2)].copy()
    rows: list[dict[str, object]] = []
    for hurdle_bps in hurdle_grid:
        work = derive_economic_targets(frame, hurdle_bps=hurdle_bps)
        train = work[work["fold_order"].eq(1)].copy()
        test = work[work["fold_order"].eq(2)].copy()
        for stage in SUPPORT_STAGES:
            columns = [*features, *support_columns(stage)]
            score = _arm_score("T4_hurdle_decomposition", train, test, columns, hurdle_bps=hurdle_bps)
            scored = test[["candidate_id", "__ts__", "__decision_ts__", "side_name", "__symbol__", "execution_net_ev_12h", "execution_gross_ev_12h", "execution_cost_return"]].copy()
            scored["score"] = score
            metrics = pooled_global_top_k_metrics(scored, "score", gates=AcceptanceGates())
            rows.append({"target_arm": "T4_hurdle_decomposition", "support_stage": stage, "hurdle_bps": float(hurdle_bps), **metrics})
    summary = pd.DataFrame(rows)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        summary.to_parquet(stage / "hurdle_grid_policy_summary.parquet", index=False, compression="zstd")
        manifest = {
            "schema": "controlled_hurdle_grid_v1", "status": "RESEARCH_ONLY_NOT_PROMOTION",
            "support_oof": str(support_oof), "support_oof_sha256": _sha256(support_oof),
            "features_json": str(features_json), "features_json_sha256": _sha256(features_json),
            "hurdle_grid_bps": list(hurdle_grid), "support_stages": list(SUPPORT_STAGES),
            "training_rule": "meta_train only; support predictions strict OOF; evaluate pooled global top-10% meta_oos",
            "output_sha256": _sha256(stage / "hurdle_grid_policy_summary.parquet"),
        }
        (stage / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        os.replace(stage, output)
        return manifest
    except Exception:
        import shutil
        shutil.rmtree(stage, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--support-oof", type=Path, required=True)
    parser.add_argument("--features-json", type=Path, required=True)
    parser.add_argument("--hurdle-bps", type=float, nargs="+", default=[0.0, 25.0, 50.0])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(support_oof=args.support_oof, features_json=args.features_json, output=args.output, hurdle_grid=tuple(args.hurdle_bps)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
