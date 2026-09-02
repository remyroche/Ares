#!/usr/bin/env python3
"""Score selected full-causal incumbent meta contracts into one OOF receipt.

The four meta families may use different explicitly selected full-feature
contracts.  The base geometry remains the immutable 50/50 E/T incumbent in
every arm.  This producer loads target-free feature panels by identity,
trains each arm strictly before its reserve, persists held target-free scores,
and only then writes label-derived diagnostics.

It intentionally does not fit MC1, admit candidates, replay a portfolio, or
change any inference/live artifact.  A subsequent MC1-combination runner is
the only consumer with admission authority in offline research.
"""

from __future__ import annotations

import argparse
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


SCHEMA = "strict_r3_incumbent_meta_selected_contract_score_v1"
DEFAULT_SOURCE_ROOT = grid.DEFAULT_SOURCE_ROOT
DEFAULT_POLICY = grid.DEFAULT_POLICY
DEFAULT_PATH_ROOT = grid.DEFAULT_PATH_ROOT
DEFAULT_FEATURE_ROOTS = (
    ROOT / "data_perp/artifacts/strict_r3_incumbent_meta_causal_features_20260827_preaug_v1",
    ROOT / "data_perp/artifacts/strict_r3_incumbent_meta_causal_features_20260827_v1",
)


def _exclusive_json(path: Path, payload: object) -> None:
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _fold_seed(month: pd.Timestamp) -> int:
    """Calendar-stable seed shared by compared HPO-derived contracts."""
    return grid.SEED + 12 * (int(month.year) - 2000) + int(month.month)


def _load_spec(path: Path) -> tuple[dict[str, grid.Arm], dict[str, Path], dict[str, dict[str, Any]]]:
    payload = json.loads(path.read_text())
    raw_arms = payload.get("arms")
    raw_contracts = payload.get("contracts")
    if not isinstance(raw_arms, list) or not isinstance(raw_contracts, dict):
        raise ValueError("contract spec requires arms[] and contracts{}")
    # Parse the same target/query/gain schema inline without creating a
    # transient configuration file alongside an immutable experiment receipt.
    allowed_family = {"magnitude", "under", "over", "state"}
    allowed_scale = {"bps", "atr", "sqrt_atr"}
    allowed_query = {"base_band", "timestamp", "base_band_block28"}
    arms: dict[str, grid.Arm] = {}
    model_params: dict[str, dict[str, Any]] = {}
    for item in raw_arms:
        if not isinstance(item, dict):
            raise ValueError("arm entries must be objects")
        name = str(item.get("name", "")); family = str(item.get("family", "")); scale = str(item.get("scale", "")); query = str(item.get("query", ""))
        if not name or family not in allowed_family or scale not in allowed_scale or query not in allowed_query:
            raise ValueError(f"invalid arm {item}")
        threshold = item.get("threshold")
        if family in {"under", "over"} and threshold is None:
            raise ValueError(f"{name}: missing threshold")
        classes = int(item.get("classes", 7))
        edges = item.get("state_edges")
        gain = str(item.get("gain_schedule", "medium"))
        if gain not in grid.GAIN_SCHEDULES:
            raise ValueError(f"{name}: unsupported gain schedule")
        truncation = item.get("truncation_level")
        arms[name] = grid.Arm(
            name=name, family=family, scale=scale, query=query,
            threshold=None if threshold is None else float(threshold), classes=classes,
            state_edges=None if edges is None else tuple(float(value) for value in edges),
            gain_schedule=gain, truncation_level=None if truncation is None else int(truncation),
        )
        raw_params = item.get("model_params", {})
        if not isinstance(raw_params, dict):
            raise ValueError(f"{name}: model_params must be an object")
        permitted = set(grid._MODEL_OVERRIDE_KEYS).union({"min_data_fraction"})
        unknown = sorted(set(raw_params).difference(permitted))
        if unknown:
            raise ValueError(f"{name}: unsupported model_params {unknown}")
        model_params[name] = dict(raw_params)
    if len(arms) != len(raw_arms):
        raise ValueError("arm names must be unique")
    contracts: dict[str, Path] = {}
    for name in arms:
        raw = raw_contracts.get(name)
        if not isinstance(raw, str) or not raw:
            raise ValueError(f"{name}: missing explicit contract path")
        candidate = Path(raw)
        contracts[name] = candidate if candidate.is_absolute() else (ROOT / candidate)
        if not contracts[name].exists():
            raise FileNotFoundError(contracts[name])
    return arms, contracts, model_params


def run(args: argparse.Namespace) -> None:
    if args.out.exists():
        raise FileExistsError(f"{args.out}: immutable score root already exists")
    arms, contract_paths, model_params = _load_spec(args.contract_spec)
    roots = tuple(Path(item.strip()) for item in args.feature_roots.split(",") if item.strip())
    if len(roots) < 2:
        raise ValueError("--feature-roots requires predecessor and current full-causal roots")
    source_roots = (
        tuple(Path(item.strip()) for item in args.source_roots.split(",") if item.strip())
        if args.source_roots else (args.source_root,)
    )
    if not source_roots or not all(root.exists() for root in source_roots):
        raise ValueError("--source-roots must name existing immutable target-free source roots")
    months = grid._parse_months(args.held_months)
    unlabelled_held_months = grid._parse_months(args.unlabelled_held_months) if args.unlabelled_held_months else ()
    if not set(unlabelled_held_months).issubset(months):
        raise ValueError("--unlabelled-held-months must be a subset of --held-months")
    policy = grid._read_policy(args.policy)
    args.out.mkdir(parents=True)
    _exclusive_json(args.out / "run_manifest.json", {
        "schema": SCHEMA,
        "scope": "offline strict-prequential selected full-causal meta scoring; no MC1/admission/portfolio/inference/live/exchange mutation",
        "incumbent_upstream": "0.50 * efficiency_bps + 0.50 * timing_bps",
        "source_root": str(args.source_root), "source_roots": [str(root) for root in source_roots], "policy": str(args.policy), "path_root": str(args.path_root),
        "feature_roots": [str(root) for root in roots], "held_months": [f"{month:%Y-%m}" for month in months],
        "unlabelled_held_months": [f"{month:%Y-%m}" for month in unlabelled_held_months],
        "arms": {name: {"arm": vars(arm), "contract": str(contract_paths[name]), "contract_sha256": _sha_file(contract_paths[name]), "model_params": model_params[name]} for name, arm in arms.items()},
        "seed_contract": "calendar-stable 1729 + 12*(year-2000) + month, shared by every arm",
        "causality": "stored incumbent route and point-in-time feature identities; model fits only before 28-day reserve; held score persisted target-free before diagnostics",
    })
    metrics: list[dict[str, Any]] = []
    for arm_index, (name, arm) in enumerate(arms.items()):
        fields = grid._load_feature_contract(contract_paths[name])
        folds = grid._prepare_folds(
            source_root=source_roots, policy=policy, path_root=args.path_root, held_months=months,
            full_feature_roots=roots, full_feature_fields=fields, unlabelled_held_months=unlabelled_held_months,
        )
        for fold_index, fold in enumerate(folds):
            scores, cache = grid._fit_score(
                fold,
                arm,
                seed=_fold_seed(fold.held_month),
                model_params=model_params[name],
            )
            receipt = grid._write_scores(args.out, arm, fold, scores)
            metric = grid._metrics(fold, scores, cache)
            metric.update({"feature_contract": str(contract_paths[name]), "feature_count": len(fields)})
            metrics.append(metric)
            with (args.out / "progress.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"event": "arm_fold_complete", "arm": name, "month": f"{fold.held_month:%Y-%m}", "rows": len(scores), "receipt": str(receipt)}, sort_keys=True) + "\n")
    report = pd.DataFrame(metrics)
    report.to_parquet(args.out / "target_query_metrics.parquet", index=False, compression="zstd")
    # `_metrics` intentionally preserves the compact family/query field names
    # from the target-query grid.  Keep this summary aligned with that emitted
    # schema so score-only receipts close successfully after every arm.
    report.groupby(["arm", "family", "query", "feature_count"], sort=True).agg(
        folds=("held_month", "nunique"), residual_ic=("residual_spearman_ic", "mean"), cmi=("conditional_mi_meta_policy_given_base", "mean"),
        sub_top1=("substitution_delta_top1_bps", "mean"), sub_top2=("substitution_delta_top2_bps", "mean"), worst_top2=("substitution_delta_top2_bps", "min"),
    ).reset_index().to_parquet(args.out / "score_summary.parquet", index=False, compression="zstd")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--contract-spec", type=Path, required=True)
    parser.add_argument("--held-months", default="2025-09,2025-10,2025-11,2025-12,2026-01,2026-02,2026-03,2026-04,2026-05,2026-06,2026-07")
    parser.add_argument(
        "--unlabelled-held-months", default="",
        help="comma-separated held months scored target-free without opening unresolved outcome diagnostics",
    )
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument(
        "--source-roots", default="",
        help="comma-separated immutable source roots; exactly one root must own each required calendar month",
    )
    parser.add_argument("--feature-roots", default=",".join(str(root) for root in DEFAULT_FEATURE_ROOTS))
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--path-root", type=Path, default=DEFAULT_PATH_ROOT)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
