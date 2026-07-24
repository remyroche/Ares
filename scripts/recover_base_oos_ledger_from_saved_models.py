#!/usr/bin/env python3
"""Recover a base OOS ledger from completed fold caches and saved models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from scripts.run_materialized_trailing_label_topk_lgbm_hpo import (
    TOP_FRACS,
    _ae_gmm_context_columns,
    _rank_top_indices,
)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _combine_shards(shard_paths: list[Path], output: Path) -> None:
    schemas = [pq.read_schema(path) for path in shard_paths]
    union = pa.unify_schemas(schemas)
    writer = pq.ParquetWriter(output, union, compression="zstd")
    try:
        for path in shard_paths:
            table = pq.read_table(path)
            missing = [field for field in union if field.name not in table.column_names]
            for field in missing:
                table = table.append_column(
                    field.name, pa.nulls(table.num_rows, type=field.type)
                )
            table = table.select(union.names).cast(union)
            writer.write_table(table)
    finally:
        writer.close()


def recover(run_dir: Path, *, combine_only: bool = False) -> Path:
    cache_root = run_dir / "_fold_cache"
    model_root = run_dir / "models"
    shard_root = run_dir / "recovered_oos_shards"
    shard_root.mkdir(parents=True, exist_ok=True)
    model_manifests = sorted(model_root.glob("*/manifest.json"))
    if not model_manifests:
        raise FileNotFoundError(f"no saved fold models under {model_root}")

    output = run_dir / "best_oos_scored_ledger.parquet"
    if combine_only:
        shard_paths = sorted(shard_root.glob("*.parquet"))
        if not shard_paths:
            raise FileNotFoundError(f"no recovered shards under {shard_root}")
        _combine_shards(shard_paths, output)
        return output

    rows = 0
    fold_rows: list[dict] = []
    for manifest_path in model_manifests:
            manifest = _load_json(manifest_path)
            fold = str(manifest["fold"])
            cache_dir = cache_root / fold
            valid = pd.read_parquet(cache_dir / "valid.parquet")
            x_valid = pd.read_parquet(cache_dir / "x_valid.parquet")
            model = joblib.load(Path(manifest["model_path"]))
            pred = pd.Series(model.predict(x_valid).astype(np.float32, copy=False))
            scored = valid.copy()
            scored["score"] = pred.to_numpy(copy=False)
            scored["oos_fold"] = fold
            ts = pd.to_datetime(scored["__ts__"], errors="coerce", utc=True)
            scored["fold_window"] = str(manifest.get("calendar_month", fold))
            scored["calendar_month"] = ts.dt.strftime("%Y-%m")
            scored["month"] = scored["calendar_month"]
            scored["valid_start"] = pd.Timestamp(manifest["valid_start"])
            scored["valid_end"] = pd.Timestamp(manifest["valid_end"])
            scored["max_oos_model_age_days"] = int(
                manifest.get("max_oos_model_age_days", 0)
            )
            scored["base_model_trial_number"] = int(manifest["trial_number"])
            scored["base_model_target_mode"] = str(manifest["target_mode"])
            scored["base_model_weight_arm"] = str(manifest["weight_arm"])

            context_path = cache_dir / "ae_gmm_context_valid.parquet"
            if context_path.exists():
                context = pd.read_parquet(context_path)
                if len(context) == len(scored):
                    for col in _ae_gmm_context_columns(context.columns):
                        if col not in scored.columns:
                            scored[col] = context[col].to_numpy(copy=False)
                del context
            side = pd.to_numeric(
                scored.get("__side__", scored.get("side", np.nan)), errors="coerce"
            )
            if "side_name" not in scored.columns:
                scored["side_name"] = np.where(side.to_numpy(copy=False) < 0, "short", "long")
            for frac in TOP_FRACS:
                mask = np.zeros(len(scored), dtype=bool)
                selected = _rank_top_indices(pred, float(frac))
                mask[selected] = True
                scored[f"selected_top{int(round(frac * 100))}"] = mask
            scored["__ts__"] = pd.to_datetime(scored["__ts__"], errors="coerce", utc=True)
            scored = scored.sort_values(
                ["__ts__", "__symbol__", "side_name"], kind="mergesort"
            ).reset_index(drop=True)
            shard_path = shard_root / f"{fold}.parquet"
            scored.to_parquet(shard_path, index=False)
            rows += len(scored)
            fold_rows.append(
                {
                    "fold": fold,
                    "rows": int(len(scored)),
                    "valid_start": manifest["valid_start"],
                    "valid_end": manifest["valid_end"],
                    "model_path": manifest["model_path"],
                    "shard_path": str(shard_path),
                }
            )
            print(f"[recover] fold={fold} rows={len(scored)} total={rows}", flush=True)
            del valid, x_valid, model, pred, scored

    _combine_shards(sorted(shard_root.glob("*.parquet")), output)

    recovery = {
        "schema": "base_oos_ledger_recovery_v1",
        "rows": int(rows),
        "folds": fold_rows,
        "output": str(output),
        "contract": "saved walk-forward fold models score only their cached validation rows",
    }
    (run_dir / "recovery_manifest.json").write_text(
        json.dumps(recovery, indent=2) + "\n", encoding="utf-8"
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--combine-only", action="store_true")
    args = parser.parse_args()
    print(recover(args.run_dir, combine_only=bool(args.combine_only)))


if __name__ == "__main__":
    main()
