#!/usr/bin/env python3
"""Materialize selected first-touch two-head label recipe columns.

The output artifact keeps the original first-touch labels intact and appends
explicit utility/support target columns for the selected two-head recipe. Sample
weight arms are recorded in the manifest as fit-time recipes rather than
materialized globally, because arms such as W14 use training-window quantiles.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
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
    _spearman,
)


DEFAULT_SOURCE_LABELS_DIR = Path(
    "data_perp/artifacts/20260702_120500_first_touch_c0_fast6_s10_policy_net_labels_exitaligned/labels"
)
DEFAULT_RECIPE_PATH = Path("docs/first_touch_two_head_label_recipe_stage134.json")
DEFAULT_OUTPUT_RUN_ID = "20260703_120000_first_touch_two_head_stage134_labels"
PRIMARY_PREFIX = "__stage134_primary"
BALANCED_PREFIX = "__stage134_balanced"
DEFAULT_RECIPE_MANIFEST_KEY = "stage134_two_head_label_recipe"
DEFAULT_SUMMARY_FILE = "stage134_two_head_label_materialization_summary.json"


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_numeric(values: Any) -> pd.Series:
    if isinstance(values, pd.Series):
        return pd.to_numeric(values, errors="coerce")
    return pd.to_numeric(pd.Series(values), errors="coerce")


def _month_series(frame: pd.DataFrame) -> pd.Series:
    return pd.to_datetime(frame["__ts__"], errors="coerce").dt.to_period("M").astype(str)


def _target_summary(
    *,
    frame: pd.DataFrame,
    metrics: pd.DataFrame,
    target: pd.DataFrame,
    prefix: str,
) -> dict[str, Any]:
    target_soft = _safe_numeric(target["target_soft"])
    target_hard = _safe_numeric(target["target_hard"])
    first_touch_net = _safe_numeric(metrics["first_touch_net"])
    clean_exec = _safe_numeric(metrics["clean_first_touch_exec"])
    summary: dict[str, Any] = {
        f"{prefix}_target_soft_mean": _safe_mean(target_soft),
        f"{prefix}_target_soft_p10": _safe_quantile(target_soft, 0.10),
        f"{prefix}_target_soft_p90": _safe_quantile(target_soft, 0.90),
        f"{prefix}_target_hard_rate": _safe_mean(target_hard),
        f"{prefix}_ic_first_touch_net": _spearman(target_soft, first_touch_net),
        f"{prefix}_ic_clean_exec": _spearman(target_soft, clean_exec),
    }
    monthly: list[dict[str, Any]] = []
    months = _month_series(frame)
    for period, idx in pd.Series(np.arange(len(frame))).groupby(months, dropna=False):
        pos = idx.to_numpy(dtype=np.int64)
        monthly.append(
            {
                "period": str(period),
                "rows": int(len(pos)),
                "target_soft_mean": _safe_mean(target_soft.iloc[pos]),
                "target_hard_rate": _safe_mean(target_hard.iloc[pos]),
                "target_ic_first_touch_net": _spearman(target_soft.iloc[pos], first_touch_net.iloc[pos]),
                "target_ic_clean_exec": _spearman(target_soft.iloc[pos], clean_exec.iloc[pos]),
            }
        )
    summary[f"{prefix}_monthly"] = monthly
    return summary


def _candidate(recipe: dict[str, Any], key: str) -> dict[str, Any]:
    candidate = recipe.get(key)
    if not isinstance(candidate, dict):
        raise ValueError(f"Recipe missing candidate '{key}'")
    required = {
        "utility_target_mode",
        "support_target_mode",
        "utility_weight_arm",
        "support_weight_arm",
        "score_rule",
        "support_gate_frac",
        "top_frac",
    }
    missing = sorted(required - set(candidate))
    if missing:
        raise ValueError(f"Recipe candidate '{key}' missing fields: {missing}")
    return candidate


def _append_candidate_columns(
    *,
    out: pd.DataFrame,
    utility_target: pd.DataFrame,
    support_target: pd.DataFrame,
    metrics: pd.DataFrame,
    prefix: str,
) -> None:
    out[f"{prefix}_utility_target_soft__"] = _safe_numeric(utility_target["target_soft"]).to_numpy(dtype=np.float32)
    out[f"{prefix}_utility_target_hard__"] = _safe_numeric(utility_target["target_hard"]).to_numpy(dtype=np.float32)
    out[f"{prefix}_support_target_soft__"] = _safe_numeric(support_target["target_soft"]).to_numpy(dtype=np.float32)
    out[f"{prefix}_support_target_hard__"] = _safe_numeric(support_target["target_hard"]).to_numpy(dtype=np.float32)
    out[f"{prefix}_first_touch_net__"] = _safe_numeric(metrics["first_touch_net"]).to_numpy(dtype=np.float32)
    out[f"{prefix}_clean_first_touch_exec__"] = _safe_numeric(metrics["clean_first_touch_exec"]).to_numpy(dtype=np.float32)
    out[f"{prefix}_first_touch_timeout__"] = _safe_numeric(metrics["first_touch_timeout"]).to_numpy(dtype=np.float32)
    out[f"{prefix}_first_touch_mae_to_sl__"] = _safe_numeric(metrics["first_touch_mae_to_sl"]).to_numpy(dtype=np.float32)


def _materialize_file(
    *,
    source_path: Path,
    output_path: Path,
    dataset_name: str,
    recipe: dict[str, Any],
    primary: dict[str, Any],
    balanced: dict[str, Any] | None,
    primary_prefix: str,
    balanced_prefix: str,
) -> dict[str, Any]:
    frame = pd.read_parquet(source_path).reset_index(drop=True)
    required = {"__ts__", "__symbol__", "__u_policy_net__", "__first_touch_policy_soft__"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"{source_path}: missing required columns {missing}")
    metrics = _first_touch_eval_metrics(frame, _path_metrics(frame))

    utility_target = _target_from_frame(
        frame,
        metrics,
        target_mode=str(primary["utility_target_mode"]),
    )
    primary_support = _target_from_frame(
        frame,
        metrics,
        target_mode=str(primary["support_target_mode"]),
    )
    out = frame.copy()
    _append_candidate_columns(
        out=out,
        utility_target=utility_target,
        support_target=primary_support,
        metrics=metrics,
        prefix=primary_prefix,
    )

    summary = {
        "dataset": str(dataset_name),
        "source_file": str(source_path),
        "output_file": str(output_path),
        "rows": int(len(out)),
        "timestamp_min": pd.to_datetime(out["__ts__"], errors="coerce").min(),
        "timestamp_max": pd.to_datetime(out["__ts__"], errors="coerce").max(),
        "symbols": int(out["__symbol__"].nunique(dropna=True)),
        "first_touch_net_mean": _safe_mean(metrics["first_touch_net"]),
        "clean_first_touch_exec_rate": _safe_mean(metrics["clean_first_touch_exec"]),
        "first_touch_timeout_rate": _safe_mean(metrics["first_touch_timeout"]),
        "first_touch_bad_mae_to_sl_rate": _safe_mean(_safe_numeric(metrics["first_touch_mae_to_sl"]) >= 1.0),
    }
    summary.update(
        _target_summary(
            frame=frame,
            metrics=metrics,
            target=utility_target,
            prefix="primary_utility",
        )
    )
    summary.update(
        _target_summary(
            frame=frame,
            metrics=metrics,
            target=primary_support,
            prefix="primary_support",
        )
    )

    if balanced is not None:
        balanced_utility = (
            utility_target
            if str(balanced["utility_target_mode"]) == str(primary["utility_target_mode"])
            else _target_from_frame(frame, metrics, target_mode=str(balanced["utility_target_mode"]))
        )
        balanced_support = _target_from_frame(
            frame,
            metrics,
            target_mode=str(balanced["support_target_mode"]),
        )
        _append_candidate_columns(
            out=out,
            utility_target=balanced_utility,
            support_target=balanced_support,
            metrics=metrics,
            prefix=balanced_prefix,
        )
        summary.update(
            _target_summary(
                frame=frame,
                metrics=metrics,
                target=balanced_support,
                prefix="balanced_support",
            )
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output_path, index=False)
    summary["columns_added"] = [
        col
        for col in out.columns
        if col.startswith(primary_prefix) or col.startswith(balanced_prefix)
    ]
    summary["recipe_id"] = str(recipe.get("recipe_id", ""))
    return summary


def _updated_dataset_meta(meta: dict[str, Any], added_columns: list[str]) -> dict[str, Any]:
    out = dict(meta)
    columns = list(out.get("columns", []))
    for col in added_columns:
        if col not in columns:
            columns.append(col)
    out["columns"] = columns
    return out


def run_materialization(
    *,
    source_labels_dir: Path,
    output_labels_dir: Path,
    output_run_id: str,
    recipe_path: Path,
    include_balanced_challenger: bool,
    primary_prefix: str = PRIMARY_PREFIX,
    balanced_prefix: str = BALANCED_PREFIX,
    recipe_manifest_key: str = DEFAULT_RECIPE_MANIFEST_KEY,
    summary_file: str = DEFAULT_SUMMARY_FILE,
) -> dict[str, Any]:
    source_manifest_path = source_labels_dir / "labels_manifest.json"
    source_manifest = _read_json(source_manifest_path)
    recipe = _read_json(recipe_path)
    primary = _candidate(recipe, "selected_by_fit")
    balanced = _candidate(recipe, "balanced_challenger") if include_balanced_challenger else None
    datasets = source_manifest.get("datasets")
    if not isinstance(datasets, dict) or not datasets:
        raise RuntimeError(f"No datasets in {source_manifest_path}")

    output_labels_dir.mkdir(parents=True, exist_ok=True)
    out_manifest = {
        "run_id": str(output_run_id),
        "source_labels_dir": str(source_labels_dir),
        "source_manifest": str(source_manifest_path),
        "datasets": {},
        str(recipe_manifest_key): {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "recipe_path": str(recipe_path),
            "recipe_id": str(recipe.get("recipe_id", "")),
            "primary_column_prefix": str(primary_prefix),
            "balanced_column_prefix": str(balanced_prefix) if balanced is not None else None,
            "selected_by_fit": primary,
            "balanced_challenger": balanced,
            "sample_weight_contract": (
                "weight arms are fit-time recipes; do not use globally precomputed weights for W14/W16"
            ),
        },
    }
    summaries: list[dict[str, Any]] = []
    for dataset_name, meta in datasets.items():
        if not isinstance(meta, dict):
            continue
        file_name = str(meta.get("file") or "")
        if not file_name:
            continue
        source_path = source_labels_dir / file_name
        output_path = output_labels_dir / file_name
        summary = _materialize_file(
            source_path=source_path,
            output_path=output_path,
            dataset_name=str(dataset_name),
            recipe=recipe,
            primary=primary,
            balanced=balanced,
            primary_prefix=str(primary_prefix),
            balanced_prefix=str(balanced_prefix),
        )
        summaries.append(summary)
        out_meta = _updated_dataset_meta(meta, list(summary.get("columns_added", [])))
        out_meta["rows"] = int(summary["rows"])
        out_manifest["datasets"][dataset_name] = out_meta
    if not summaries:
        raise RuntimeError(f"No datasets materialized from {source_labels_dir}")

    manifest_path = output_labels_dir / "labels_manifest.json"
    summary_path = output_labels_dir / str(summary_file)
    manifest_path.write_text(json.dumps(_json_safe(out_manifest), indent=2, sort_keys=True), encoding="utf-8")
    summary_doc = {
        "output_labels_dir": str(output_labels_dir),
        "datasets": summaries,
        "recipe": out_manifest[str(recipe_manifest_key)],
    }
    summary_path.write_text(json.dumps(_json_safe(summary_doc), indent=2, sort_keys=True), encoding="utf-8")
    return {
        "output_labels_dir": str(output_labels_dir),
        "manifest": str(manifest_path),
        "summary": str(summary_path),
        "datasets": summaries,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-labels-dir", type=Path, default=DEFAULT_SOURCE_LABELS_DIR)
    parser.add_argument("--output-run-id", default=DEFAULT_OUTPUT_RUN_ID)
    parser.add_argument("--output-labels-dir", type=Path, default=None)
    parser.add_argument("--recipe-path", type=Path, default=DEFAULT_RECIPE_PATH)
    parser.add_argument("--no-balanced-challenger", action="store_true")
    parser.add_argument("--primary-prefix", default=PRIMARY_PREFIX)
    parser.add_argument("--balanced-prefix", default=BALANCED_PREFIX)
    parser.add_argument("--recipe-manifest-key", default=DEFAULT_RECIPE_MANIFEST_KEY)
    parser.add_argument("--summary-file", default=DEFAULT_SUMMARY_FILE)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_labels_dir = (
        args.output_labels_dir
        if args.output_labels_dir is not None
        else Path("data_perp/artifacts") / str(args.output_run_id) / "labels"
    )
    result = run_materialization(
        source_labels_dir=args.source_labels_dir,
        output_labels_dir=output_labels_dir,
        output_run_id=str(args.output_run_id),
        recipe_path=args.recipe_path,
        include_balanced_challenger=not bool(args.no_balanced_challenger),
        primary_prefix=str(args.primary_prefix),
        balanced_prefix=str(args.balanced_prefix),
        recipe_manifest_key=str(args.recipe_manifest_key),
        summary_file=str(args.summary_file),
    )
    print(json.dumps(_json_safe(result), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
