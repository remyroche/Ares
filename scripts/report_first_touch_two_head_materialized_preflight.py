#!/usr/bin/env python3
"""Preflight materialized first-touch two-head label recipe artifacts."""

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

from scripts.run_first_touch_label_training_smoke import (  # noqa: E402
    _first_touch_eval_metrics,
    _target_from_frame,
)
from scripts.run_label_quality_proxy_diagnostics import (  # noqa: E402
    _json_safe,
    _path_metrics,
    _safe_mean,
    _safe_quantile,
)


DEFAULT_LABELS_DIR = Path("data_perp/artifacts/20260703_170000_first_touch_two_head_stage164_labels/labels")
DEFAULT_OUTPUT_DIR = Path("data_perp/reports/first_touch_two_head_stage164_materialized_preflight_v1")


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _safe_numeric(values: Any) -> pd.Series:
    if isinstance(values, pd.Series):
        return pd.to_numeric(values, errors="coerce")
    return pd.to_numeric(pd.Series(values), errors="coerce")


def _finite(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def _column_stats(frame: pd.DataFrame, column: str) -> dict[str, Any]:
    values = _safe_numeric(frame[column])
    return {
        "column": str(column),
        "rows": int(len(values)),
        "finite_frac": float(values.notna().mean()) if len(values) else float("nan"),
        "mean": _safe_mean(values),
        "p10": _safe_quantile(values, 0.10),
        "p90": _safe_quantile(values, 0.90),
        "min": _finite(values.min(skipna=True)),
        "max": _finite(values.max(skipna=True)),
    }


def _max_abs_diff(left: pd.Series, right: pd.Series) -> float:
    diff = (_safe_numeric(left).reset_index(drop=True) - _safe_numeric(right).reset_index(drop=True)).abs()
    return _finite(diff.max(skipna=True))


def _target_checks(
    *,
    frame: pd.DataFrame,
    candidate: dict[str, Any],
    prefix: str,
    metrics: pd.DataFrame,
    role: str,
) -> list[dict[str, Any]]:
    utility = _target_from_frame(frame, metrics, target_mode=str(candidate["utility_target_mode"]))
    support = _target_from_frame(frame, metrics, target_mode=str(candidate["support_target_mode"]))
    checks: list[dict[str, Any]] = []
    pairs = [
        (f"{prefix}_utility_target_soft__", utility["target_soft"], f"{role}_utility_soft_matches"),
        (f"{prefix}_utility_target_hard__", utility["target_hard"], f"{role}_utility_hard_matches"),
        (f"{prefix}_support_target_soft__", support["target_soft"], f"{role}_support_soft_matches"),
        (f"{prefix}_support_target_hard__", support["target_hard"], f"{role}_support_hard_matches"),
        (f"{prefix}_first_touch_net__", metrics["first_touch_net"], f"{role}_first_touch_net_matches"),
        (
            f"{prefix}_clean_first_touch_exec__",
            metrics["clean_first_touch_exec"],
            f"{role}_clean_first_touch_exec_matches",
        ),
        (f"{prefix}_first_touch_timeout__", metrics["first_touch_timeout"], f"{role}_timeout_matches"),
        (f"{prefix}_first_touch_mae_to_sl__", metrics["first_touch_mae_to_sl"], f"{role}_mae_to_sl_matches"),
    ]
    for column, expected, check in pairs:
        if column not in frame.columns:
            checks.append({"check": check, "pass": False, "detail": {"missing_column": column}})
            continue
        max_abs_diff = _max_abs_diff(frame[column], expected)
        checks.append(
            {
                "check": check,
                "pass": bool(math.isfinite(max_abs_diff) and max_abs_diff <= 1e-6),
                "detail": {"column": column, "max_abs_diff": max_abs_diff},
            }
        )
    return checks


def run_preflight(
    *,
    labels_dir: Path,
    output_dir: Path,
    recipe_manifest_key: str,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = labels_dir / "labels_manifest.json"
    manifest = _load_json(manifest_path)
    recipe = manifest.get(recipe_manifest_key)
    if not isinstance(recipe, dict):
        raise ValueError(f"Manifest missing recipe key: {recipe_manifest_key}")
    primary = recipe.get("selected_by_fit")
    challenger = recipe.get("balanced_challenger")
    if not isinstance(primary, dict):
        raise ValueError("Recipe missing selected_by_fit")
    primary_prefix = str(recipe.get("primary_column_prefix") or "")
    challenger_prefix = str(recipe.get("balanced_column_prefix") or "")
    if not primary_prefix:
        raise ValueError("Recipe missing primary_column_prefix")

    checks: list[dict[str, Any]] = []
    column_rows: list[dict[str, Any]] = []
    dataset_summaries: list[dict[str, Any]] = []
    datasets = manifest.get("datasets")
    if not isinstance(datasets, dict) or not datasets:
        raise ValueError(f"No datasets in {manifest_path}")

    for dataset, meta in datasets.items():
        file_name = str(meta.get("file") or "")
        path = labels_dir / file_name
        frame = pd.read_parquet(path).reset_index(drop=True)
        metrics = _first_touch_eval_metrics(frame, _path_metrics(frame))
        expected_rows = int(meta.get("rows", -1))
        checks.append(
            {
                "dataset": str(dataset),
                "check": "row_count_manifest_match",
                "pass": bool(expected_rows == len(frame)),
                "detail": {"manifest_rows": expected_rows, "rows": int(len(frame))},
            }
        )
        required = [
            f"{primary_prefix}_utility_target_soft__",
            f"{primary_prefix}_utility_target_hard__",
            f"{primary_prefix}_support_target_soft__",
            f"{primary_prefix}_support_target_hard__",
            f"{primary_prefix}_first_touch_net__",
            f"{primary_prefix}_clean_first_touch_exec__",
            f"{primary_prefix}_first_touch_timeout__",
            f"{primary_prefix}_first_touch_mae_to_sl__",
        ]
        if isinstance(challenger, dict) and challenger_prefix:
            required.extend(
                [
                    f"{challenger_prefix}_utility_target_soft__",
                    f"{challenger_prefix}_utility_target_hard__",
                    f"{challenger_prefix}_support_target_soft__",
                    f"{challenger_prefix}_support_target_hard__",
                    f"{challenger_prefix}_first_touch_net__",
                    f"{challenger_prefix}_clean_first_touch_exec__",
                    f"{challenger_prefix}_first_touch_timeout__",
                    f"{challenger_prefix}_first_touch_mae_to_sl__",
                ]
            )
        missing = [column for column in required if column not in frame.columns]
        checks.append(
            {
                "dataset": str(dataset),
                "check": "required_columns_present",
                "pass": not missing,
                "detail": {"missing": missing},
            }
        )
        for column in required:
            if column in frame.columns:
                row = {"dataset": str(dataset), **_column_stats(frame, column)}
                column_rows.append(row)
        for check in _target_checks(
            frame=frame,
            candidate=primary,
            prefix=primary_prefix,
            metrics=metrics,
            role="primary",
        ):
            checks.append({"dataset": str(dataset), **check})
        if isinstance(challenger, dict) and challenger_prefix:
            for check in _target_checks(
                frame=frame,
                candidate=challenger,
                prefix=challenger_prefix,
                metrics=metrics,
                role="challenger",
            ):
                checks.append({"dataset": str(dataset), **check})
        dataset_summaries.append(
            {
                "dataset": str(dataset),
                "rows": int(len(frame)),
                "timestamp_min": pd.to_datetime(frame["__ts__"], errors="coerce").min(),
                "timestamp_max": pd.to_datetime(frame["__ts__"], errors="coerce").max(),
                "symbols": int(frame["__symbol__"].nunique(dropna=True)),
            }
        )

    checks_df = pd.DataFrame(checks)
    columns_df = pd.DataFrame(column_rows)
    checks_path = output_dir / "first_touch_two_head_materialized_checks.csv"
    columns_path = output_dir / "first_touch_two_head_materialized_column_preflight.csv"
    json_path = output_dir / "first_touch_two_head_materialized_preflight.json"
    markdown_path = output_dir / "first_touch_two_head_materialized_preflight.md"
    checks_df.to_csv(checks_path, index=False)
    columns_df.to_csv(columns_path, index=False)
    all_pass = bool(checks_df["pass"].astype(bool).all()) if not checks_df.empty else False
    payload = {
        "labels_dir": str(labels_dir),
        "manifest": str(manifest_path),
        "recipe_manifest_key": str(recipe_manifest_key),
        "recipe_id": str(recipe.get("recipe_id", "")),
        "primary_column_prefix": primary_prefix,
        "challenger_column_prefix": challenger_prefix or None,
        "datasets": dataset_summaries,
        "checks": int(len(checks_df)),
        "failed_checks": int((~checks_df["pass"].astype(bool)).sum()) if not checks_df.empty else 0,
        "all_pass": all_pass,
        "outputs": {
            "checks": str(checks_path),
            "columns": str(columns_path),
            "json": str(json_path),
            "markdown": str(markdown_path),
        },
    }
    json_path.write_text(json.dumps(_json_safe(payload), indent=2), encoding="utf-8")
    lines = [
        "# First-Touch Two-Head Materialized Preflight",
        "",
        f"Labels: `{labels_dir}`",
        f"Recipe: `{payload['recipe_id']}`",
        f"All pass: `{all_pass}`",
        f"Failed checks: `{payload['failed_checks']}` / `{payload['checks']}`",
        "",
        "## Outputs",
        "",
        f"- JSON: `{json_path}`",
        f"- Checks CSV: `{checks_path}`",
        f"- Column CSV: `{columns_path}`",
    ]
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-dir", type=Path, default=DEFAULT_LABELS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--recipe-manifest-key", default="stage164_two_head_label_recipe")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = run_preflight(
        labels_dir=args.labels_dir,
        output_dir=args.output_dir,
        recipe_manifest_key=str(args.recipe_manifest_key),
    )
    print(json.dumps(_json_safe(payload), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
