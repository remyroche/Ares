#!/usr/bin/env python3
"""Rescore a small U-head HPO finalist set into strict target-free OOF panels.

The input HPO has already chosen its finalists on earlier development months.
This producer only refits each finalist causally for the declared monthly
history.  It persists target-free meta scores; a separate MC1 comparison owns
all policy joins, admissions, and portfolio economics.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT, ROOT / "scripts"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import run_strict_r3_incumbent_meta_target_query_grid_v1 as grid  # noqa: E402
import run_strict_r3_incumbent_meta_under_hpo_v1 as hpo  # noqa: E402


SCHEMA = "strict_r3_incumbent_meta_under_hpo_finalist_score_v1"
DEFAULT_HPO = ROOT / "data_perp/artifacts/strict_r3_incumbent_meta_under_hpo_20260827_v3"
DEFAULT_CONTRACT = hpo.DEFAULT_CONTRACT
DEFAULT_FEATURE_ROOTS = hpo.DEFAULT_FEATURE_ROOTS


def _exclusive_json(path: Path, payload: object) -> None:
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    for target in (sorted(path.rglob("*.parquet")) if path.is_dir() else [path]):
        digest.update(str(target).encode())
        with target.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
    return digest.hexdigest()


def _parse_months(raw: str) -> tuple[pd.Timestamp, ...]:
    output = tuple(pd.Timestamp(f"{item.strip()}-01", tz="UTC") for item in raw.split(",") if item.strip())
    if not output or tuple(sorted(set(output))) != output:
        raise ValueError("--held-months must be unique chronological calendar months")
    return output


def _model_params(row: pd.Series) -> dict[str, Any]:
    depth, leaves = (int(part[1:]) for part in str(row.tree_geometry).split("_"))
    truncation = str(row.truncation)
    return {
        "n_estimators": 1_200,
        "learning_rate": float(row.learning_rate),
        "max_depth": depth,
        "num_leaves": leaves,
        "min_data_fraction": float(row.min_data_fraction),
        "subsample": float(row.bagging_fraction),
        "subsample_freq": 1,
        "colsample_bytree": float(row.feature_fraction),
        "reg_alpha": float(row.lambda_l1),
        "reg_lambda": float(row.lambda_l2),
        "min_split_gain": float(row.min_gain_to_split),
        "sigmoid": float(row.sigmoid),
        "lambdarank_truncation_level": None if truncation == "none" else int(truncation),
    }


def run(args: argparse.Namespace) -> None:
    if args.out.exists():
        raise FileExistsError(f"{args.out}: immutable score root already exists")
    trials = pd.read_parquet(args.hpo_root / "trials.parquet")
    finalists = trials.loc[trials.state.eq("complete")].sort_values("selection_score", ascending=False).head(args.top_n).copy()
    if len(finalists) != args.top_n:
        raise AssertionError("fewer completed HPO trials than requested finalists")
    fields = grid._load_feature_contract(args.feature_contract)
    roots = tuple(Path(item.strip()) for item in args.feature_roots.split(",") if item.strip())
    if not 30 <= len(fields) <= 70 or len(roots) < 2:
        raise ValueError("requires a 30..70-field contract and predecessor/current full-causal roots")
    months = _parse_months(args.held_months)
    policy = grid._read_policy(args.policy)
    folds = grid._prepare_folds(
        source_root=args.source_root,
        policy=policy,
        path_root=args.path_root,
        held_months=months,
        full_feature_roots=roots,
        full_feature_fields=fields,
    )
    args.out.mkdir(parents=True)
    candidates: list[dict[str, Any]] = []
    if args.include_default:
        candidates.append({
            "name": "under_default_f50",
            "trial": None,
            "development_selection_score": None,
            "model_params": {},
        })
    for _, row in finalists.iterrows():
        name = f"under_hpo_trial{int(row.trial):02d}"
        candidates.append({
            "name": name,
            "trial": int(row.trial),
            "development_selection_score": float(row.selection_score),
            "model_params": _model_params(row),
        })
    _exclusive_json(args.out / "run_manifest.json", {
        "schema": SCHEMA,
        "scope": "offline strict-prequential U-head HPO finalist scoring; target-free scores only; no MC1/admission/portfolio/inference/live/exchange mutation",
        "incumbent_upstream": "0.50 * efficiency_bps + 0.50 * timing_bps",
        "hpo_root": str(args.hpo_root),
        "hpo_trial_sha256": _sha(args.hpo_root / "trials.parquet"),
        "feature_contract": str(args.feature_contract),
        "feature_count": len(fields),
        "feature_roots": [str(root) for root in roots],
        "held_months": [f"{month:%Y-%m}" for month in months],
        "candidates": candidates,
        "seed_contract": "all finalists use the identical calendar-stable per-fold seed = 1729 + 12*(year-2000) + month; no candidate-specific sampling or tree-seed advantage",
        "causality": "same stored incumbent route; fit only before 28-day reserve; held target-free scores persisted before policy diagnostics",
    })
    metrics: list[dict[str, Any]] = []
    for candidate_index, candidate in enumerate(candidates):
        arm = dataclasses.replace(hpo.UNDER_ARM, name=str(candidate["name"]))
        for fold_index, fold in enumerate(folds):
            score, cache = grid._fit_score(
                fold,
                arm,
                # Use the same calendar-month seed as development HPO.  This
                # is independent of candidate order and of whether a caller
                # asks for a short or full score history.
                seed=hpo._fold_seed(fold.held_month),
                model_params=candidate["model_params"],
            )
            receipt = grid._write_scores(args.out, arm, fold, score)
            metric = grid._metrics(fold, score, cache)
            metric.update({
                "trial": candidate["trial"],
                "development_selection_score": candidate["development_selection_score"],
                "feature_contract": str(args.feature_contract),
                "feature_count": len(fields),
            })
            metrics.append(metric)
            with (args.out / "progress.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({
                    "event": "fold_complete", "candidate": candidate["name"],
                    "trial": candidate["trial"], "month": f"{fold.held_month:%Y-%m}",
                    "rows": len(score), "receipt": str(receipt),
                }, sort_keys=True) + "\n")
    pd.DataFrame(metrics).to_parquet(args.out / "target_query_metrics.parquet", index=False, compression="zstd")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--hpo-root", type=Path, default=DEFAULT_HPO)
    parser.add_argument("--feature-contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--top-n", type=int, default=3)
    parser.add_argument("--include-default", action="store_true", help="also rescore the unmodified U F50 contract as a matched control")
    parser.add_argument("--held-months", default="2025-09,2025-10,2025-11,2025-12,2026-01,2026-02,2026-03,2026-04,2026-05,2026-06,2026-07")
    parser.add_argument("--source-root", type=Path, default=grid.DEFAULT_SOURCE_ROOT)
    parser.add_argument("--policy", type=Path, default=grid.DEFAULT_POLICY)
    parser.add_argument("--path-root", type=Path, default=grid.DEFAULT_PATH_ROOT)
    parser.add_argument("--feature-roots", default=",".join(str(root) for root in DEFAULT_FEATURE_ROOTS))
    run(parser.parse_args())


if __name__ == "__main__":
    main()
