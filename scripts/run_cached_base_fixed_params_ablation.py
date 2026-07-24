#!/usr/bin/env python3
"""Refit a fixed base contract from immutable cached fold matrices.

This utility is intentionally narrow: it changes model parameters/weights while
holding corrected labels, selected columns, frozen AE/GMM outputs, and OOS fold
boundaries fixed. It avoids repeating feature generation or representation fit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

from scripts.run_materialized_trailing_label_topk_lgbm_hpo import (
    _json_safe,
    _load_fixed_params,
    _score_best_oos_ledger,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cached_folds(cache_dir: Path) -> list[dict[str, Any]]:
    folds: list[dict[str, Any]] = []
    required = (
        "x_train",
        "train",
        "train_metrics",
        "x_valid",
        "valid",
        "valid_metrics",
    )
    for fold_dir in cache_dir.iterdir():
        manifest_path = fold_dir / "fold_manifest.json"
        if not fold_dir.is_dir() or not manifest_path.is_file():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        payload_paths = {
            name: str(fold_dir / f"{name}.parquet") for name in required
        }
        missing = [name for name, path in payload_paths.items() if not Path(path).is_file()]
        if missing:
            raise FileNotFoundError(f"Incomplete cached fold {fold_dir}: {missing}")
        context_path = fold_dir / "ae_gmm_context_valid.parquet"
        if context_path.is_file():
            payload_paths["ae_gmm_context_valid"] = str(context_path)
        manifest.update(
            {
                "payload_paths": payload_paths,
                "compact_fixed_training_payload": False,
                "valid_start": pd.Timestamp(manifest["valid_start"]),
                "valid_end": pd.Timestamp(manifest["valid_end"]),
                "train_rows": int(manifest.get("train_rows_payload", 0)),
                "valid_rows": int(manifest.get("valid_rows_raw", 0)),
            }
        )
        folds.append(manifest)
    if not folds:
        raise FileNotFoundError(f"No complete cached folds under {cache_dir}")
    return sorted(folds, key=lambda fold: pd.Timestamp(fold["valid_start"]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--params-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--trial-number", type=int, default=135)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-rows", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_run = args.source_run.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    folds = _cached_folds(source_run / "_fold_cache")
    params = _load_fixed_params(args.params_json)
    params.pop("_fixed_trial_number", None)
    params["loss_function"] = "regression"
    manifest = {
        "schema": "cached_base_fixed_params_ablation_v1",
        "status": "running",
        "source_run": str(source_run),
        "source_manifest_sha256": _sha256(source_run / "manifest.json"),
        "source_fold_cache": str(source_run / "_fold_cache"),
        "params_json": str(args.params_json.resolve()),
        "params_json_sha256": _sha256(args.params_json.resolve()),
        "params": params,
        "trial_number": int(args.trial_number),
        "seed": int(args.seed),
        "max_train_rows": int(args.max_train_rows),
        "fold_count": int(len(folds)),
        "valid_start": min(pd.Timestamp(fold["valid_start"]) for fold in folds),
        "valid_end": max(pd.Timestamp(fold["valid_end"]) for fold in folds),
        "contract": (
            "corrected causal labels and frozen 150-feature/AE-GMM fold payloads; "
            "only benchmark L2 parameters, target mode and weighting are changed"
        ),
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(_json_safe(manifest), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    ledger_path = output_dir / "best_oos_scored_ledger.parquet"
    result = _score_best_oos_ledger(
        folds=folds,
        params=params,
        trial_number=int(args.trial_number),
        max_train_rows=int(args.max_train_rows),
        seed=int(args.seed),
        save_fold_models_dir=output_dir / "models",
        output_path=ledger_path,
    )
    manifest.update(
        {
            "status": "complete",
            "oos_ledger": str(ledger_path),
            "oos_rows": int(pq.ParquetFile(ledger_path).metadata.num_rows),
            "saved_fold_models": _json_safe(result.attrs.get("saved_fold_models", [])),
        }
    )
    manifest_path.write_text(
        json.dumps(_json_safe(manifest), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(_json_safe(manifest), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
