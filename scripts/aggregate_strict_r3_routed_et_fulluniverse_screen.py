#!/usr/bin/env python3
"""Aggregate independent strict-OOF E/T full-universe screening receipts.

The individual fold workers are intentionally process-isolated to bound native
LightGBM memory.  This producer is deterministic and combines their OOF
evidence before selecting the actual E_SCREEN120 and T_SCREEN120 contracts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from run_strict_r3_routed_et_fulluniverse_screen import _shortlist  # noqa: E402


def _write_exclusive(path: Path, payload: object) -> None:
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)


def _receipt(root: Path, prefix: str, head: str, month: str) -> Path:
    path = root / f"{prefix}_{head.lower()}_{month}"
    required = [
        "run_manifest.json", "screen_gain.parquet", "screen_shap.parquet",
        "screen_univariate.parquet", "screen_randomized_stability.parquet",
        "screen_fold_metrics.parquet", "correlation_clusters.parquet",
    ]
    missing = [item for item in required if not (path / item).exists()]
    if missing:
        raise FileNotFoundError(f"{path}: missing {missing}")
    return path


def _family_audit(summary: pd.DataFrame) -> pd.DataFrame:
    result = summary.groupby("family", sort=True).agg(
        screened_fields=("feature", "size"),
        selected_screen120=("selected_screen120", "sum"),
        mean_screen_score=("screen_score", "mean"),
        best_screen_score=("screen_score", "max"),
        best_precision_shap=("precision_shap", "max"),
        best_univariate_top10_ev=("univariate_top10_ev", "max"),
    ).reset_index()
    result["family_survives"] = result.selected_screen120.gt(0)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=ROOT / "data_perp" / "artifacts")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--run-prefix", default="strict_r3_routed_et_fulluniverse_screen_20260826_v7")
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    args.out.mkdir(parents=True)

    months = ("202602", "202603", "202604")
    manifests: list[dict] = []
    aggregate_metrics: list[pd.DataFrame] = []
    parent_paths: dict[str, list[str]] = {}
    for head in ("E", "T"):
        paths = [_receipt(args.artifacts, args.run_prefix, head, month) for month in months]
        parent_paths[head] = [str(path) for path in paths]
        manifests.extend(json.loads((path / "run_manifest.json").read_text()) for path in paths)
        gains = pd.concat([pd.read_parquet(path / "screen_gain.parquet") for path in paths], ignore_index=True)
        shap = pd.concat([pd.read_parquet(path / "screen_shap.parquet") for path in paths], ignore_index=True)
        univariate = pd.concat([pd.read_parquet(path / "screen_univariate.parquet") for path in paths], ignore_index=True)
        stability = pd.concat([pd.read_parquet(path / "screen_randomized_stability.parquet") for path in paths], ignore_index=True)
        metrics = pd.concat([pd.read_parquet(path / "screen_fold_metrics.parquet") for path in paths], ignore_index=True)
        correlation = pd.read_parquet(paths[0] / "correlation_clusters.parquet")
        fields = correlation.loc[correlation.retained_after_redundancy, "feature"].sort_values(kind="stable").tolist()
        summary, chosen = _shortlist(
            head=head, fields=fields, correlation=correlation, gain=gains, shap=shap,
            univariate=univariate, stability=stability,
        )
        digest = hashlib.sha256("\n".join(chosen).encode()).hexdigest()
        _write_exclusive(args.out / f"{head.lower()}_screen120_contract.json", {
            "schema": "strict_r3_routed_et_crossfold_screen120_v1",
            "head": head,
            "feature_contract": chosen,
            "feature_contract_sha256": digest,
            "parents": parent_paths[head],
            "selection": "three independent strict-OOF folds: gain + general/tail TreeSHAP + univariate rescue + randomized stability + semantic-family rescue; only abs-Spearman >= .995 duplicate veto",
        })
        summary.to_parquet(args.out / f"{head.lower()}_crossfold_feature_summary.parquet", index=False, compression="zstd")
        _family_audit(summary).to_parquet(args.out / f"{head.lower()}_crossfold_family_audit.parquet", index=False, compression="zstd")
        gains.to_parquet(args.out / f"{head.lower()}_crossfold_gain.parquet", index=False, compression="zstd")
        shap.to_parquet(args.out / f"{head.lower()}_crossfold_shap.parquet", index=False, compression="zstd")
        univariate.to_parquet(args.out / f"{head.lower()}_crossfold_univariate.parquet", index=False, compression="zstd")
        stability.to_parquet(args.out / f"{head.lower()}_crossfold_stability.parquet", index=False, compression="zstd")
        aggregate_metrics.append(metrics)

    metrics = pd.concat(aggregate_metrics, ignore_index=True)
    metrics.to_parquet(args.out / "crossfold_screen_metrics.parquet", index=False, compression="zstd")
    shutil.copy2(_receipt(args.artifacts, args.run_prefix, "E", "202602") / "feature_hygiene_coverage.parquet", args.out / "feature_hygiene_coverage.parquet")
    shutil.copy2(_receipt(args.artifacts, args.run_prefix, "E", "202602") / "correlation_clusters.parquet", args.out / "correlation_clusters.parquet")
    _write_exclusive(args.out / "run_manifest.json", {
        "schema": "strict_r3_routed_et_crossfold_screen_aggregate_v1",
        "scope": "offline research only; E/T only; B0 and live artifacts unchanged",
        "parents": parent_paths,
        "folds": ["2026-02", "2026-03", "2026-04"],
        "strict_oof": True,
        "route": "frozen timestamp-local top50 exact router identity",
        "target_or_outcome_in_features": False,
    })


if __name__ == "__main__":
    main()
