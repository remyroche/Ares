#!/usr/bin/env python3
"""Materialize clean first-touch dual-proxy label candidates."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_clean_first_touch_label_ablation import _build_arms  # noqa: E402
from scripts.run_label_quality_proxy_diagnostics import (  # noqa: E402
    _json_safe,
    _safe_mean,
    _safe_numeric,
    _safe_quantile,
    _spearman,
)
from scripts.run_materialized_label_column_proxy_diagnostics import _execution_metrics  # noqa: E402


DEFAULT_RECIPE_PATH = Path("docs/clean_first_touch_dual_proxy_label_recipe_stage166.json")
DEFAULT_OUTPUT_RUN_ID = "20260703_180000_clean_first_touch_dual_proxy_stage166_labels"
DEFAULT_SUMMARY_FILE = "stage166_clean_first_touch_dual_proxy_materialization_summary.json"


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _load_manifest(labels_dir: Path) -> dict[str, Any]:
    path = labels_dir / "labels_manifest.json"
    if not path.exists():
        return {}
    return _load_json(path)


def _parquet_files(labels_dir: Path) -> list[Path]:
    files = sorted(path for path in labels_dir.glob("*.parquet") if path.is_file())
    if not files:
        raise FileNotFoundError(f"No parquet files found under {labels_dir}")
    return files


def _column_stats(frame: pd.DataFrame, column: str) -> dict[str, Any]:
    values = _safe_numeric(frame[column])
    return {
        "column": column,
        "finite_frac": float(values.notna().mean()) if len(values) else float("nan"),
        "mean": _safe_mean(values),
        "p10": _safe_quantile(values, 0.10),
        "p50": _safe_quantile(values, 0.50),
        "p90": _safe_quantile(values, 0.90),
        "min": float(values.min(skipna=True)) if values.notna().any() else float("nan"),
        "max": float(values.max(skipna=True)) if values.notna().any() else float("nan"),
    }


def _materialize_file(
    *,
    source_file: Path,
    output_file: Path,
    recipe: dict[str, Any],
) -> dict[str, Any]:
    frame = pd.read_parquet(source_file).reset_index(drop=True)
    metric_prefix = str(recipe["metric_prefix"])
    primary_prefix = str(recipe["primary_prefix"])
    challenger_prefix = str(recipe["challenger_prefix"])
    output_prefix = str(recipe["materialized_prefix"])
    utility_arm = str(recipe["utility_arm"])
    risk_arm = str(recipe["risk_arm"])
    metrics = _execution_metrics(frame, metric_prefix)
    arms = {
        arm.name: arm
        for arm in _build_arms(
            frame=frame,
            metrics=metrics,
            primary_prefix=primary_prefix,
            challenger_prefix=challenger_prefix,
        )
    }
    missing = [arm for arm in (utility_arm, risk_arm) if arm not in arms]
    if missing:
        raise ValueError(f"Unknown recipe arm(s): {missing}")
    utility = arms[utility_arm].target
    risk = arms[risk_arm].target
    added = {
        f"{output_prefix}_utility_target_soft__": utility["target_soft"],
        f"{output_prefix}_utility_target_hard__": utility["target_hard"],
        f"{output_prefix}_risk_target_soft__": risk["target_soft"],
        f"{output_prefix}_risk_target_hard__": risk["target_hard"],
        f"{output_prefix}_first_touch_net__": metrics["first_touch_net"],
        f"{output_prefix}_clean_first_touch_exec__": metrics["clean_first_touch_exec"],
        f"{output_prefix}_first_touch_timeout__": metrics["first_touch_timeout"].astype(float),
        f"{output_prefix}_first_touch_mae_to_sl__": metrics["first_touch_mae_to_sl"],
    }
    out = frame.copy()
    for column, values in added.items():
        out[column] = _safe_numeric(values).astype(np.float32)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output_file, index=False)
    stats = [_column_stats(out, column) for column in added]
    return {
        "file": source_file.name,
        "rows": int(len(out)),
        "timestamp_min": pd.to_datetime(out["__ts__"], errors="coerce").min(),
        "timestamp_max": pd.to_datetime(out["__ts__"], errors="coerce").max(),
        "symbols": int(out["__symbol__"].nunique(dropna=True)),
        "columns_added": list(added.keys()),
        "column_stats": stats,
        "utility_ic_first_touch_net": _spearman(utility["target_soft"], metrics["first_touch_net"]),
        "risk_ic_clean_exec": _spearman(risk["target_soft"], metrics["clean_first_touch_exec"]),
        "risk_ic_mae_to_sl": _spearman(risk["target_soft"], metrics["first_touch_mae_to_sl"]),
    }


def run_materialization(
    *,
    recipe_path: Path,
    output_run_id: str,
    summary_file: str,
) -> dict[str, Any]:
    recipe = _load_json(recipe_path)
    source_labels_dir = Path(str(recipe["source_labels_dir"]))
    output_labels_dir = Path("data_perp/artifacts") / output_run_id / "labels"
    output_labels_dir.mkdir(parents=True, exist_ok=True)
    source_manifest = _load_manifest(source_labels_dir)
    datasets: dict[str, Any] = {}
    summaries: list[dict[str, Any]] = []
    for source_file in _parquet_files(source_labels_dir):
        output_file = output_labels_dir / source_file.name
        summary = _materialize_file(
            source_file=source_file,
            output_file=output_file,
            recipe=recipe,
        )
        summaries.append(summary)
        dataset_key = source_file.stem
        datasets[dataset_key] = {
            "file": source_file.name,
            "rows": int(summary["rows"]),
            "timestamp_min": summary["timestamp_min"],
            "timestamp_max": summary["timestamp_max"],
            "symbols": int(summary["symbols"]),
        }

    total_rows = int(sum(int(item["rows"]) for item in summaries))
    all_mins = [item["timestamp_min"] for item in summaries if pd.notna(item["timestamp_min"])]
    all_maxs = [item["timestamp_max"] for item in summaries if pd.notna(item["timestamp_max"])]
    payload = {
        "scope": "clean_first_touch_dual_proxy_label_materialization",
        "recipe": recipe,
        "recipe_path": str(recipe_path),
        "source_labels_dir": str(source_labels_dir),
        "output_run_id": str(output_run_id),
        "output_labels_dir": str(output_labels_dir),
        "source_manifest_run_id": source_manifest.get("run_id"),
        "rows": total_rows,
        "timestamp_min": min(all_mins) if all_mins else None,
        "timestamp_max": max(all_maxs) if all_maxs else None,
        "datasets": datasets,
        "file_summaries": summaries,
    }
    labels_manifest = {
        **{key: value for key, value in source_manifest.items() if key != "datasets"},
        "run_id": output_run_id,
        "source_labels_dir": str(source_labels_dir),
        "stage166_clean_first_touch_dual_proxy_recipe": recipe,
        "datasets": datasets,
    }
    (output_labels_dir / "labels_manifest.json").write_text(
        json.dumps(_json_safe(labels_manifest), indent=2),
        encoding="utf-8",
    )
    (output_labels_dir / summary_file).write_text(
        json.dumps(_json_safe(payload), indent=2),
        encoding="utf-8",
    )
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe-path", type=Path, default=DEFAULT_RECIPE_PATH)
    parser.add_argument("--output-run-id", default=DEFAULT_OUTPUT_RUN_ID)
    parser.add_argument("--summary-file", default=DEFAULT_SUMMARY_FILE)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = run_materialization(
        recipe_path=args.recipe_path,
        output_run_id=str(args.output_run_id),
        summary_file=str(args.summary_file),
    )
    print(json.dumps(_json_safe(payload), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
