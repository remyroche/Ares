#!/usr/bin/env python3
"""Run bounded common-geometry historical/current transition classifiers.

All classifier inputs are drawn from the panel's strict 90-field whitelist.
Source/domain/provenance fields are used only for source-balanced fitting and
reporting.  The 2022--2023 source remains non-OOF diagnostic evidence and the
current source contributes only strict mapped-OOF rows to fitted diagnostics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

try:
    from scripts.run_pooled_transition_lifecycle_diagnostic import (
        MIN_POSITIVES,
        _metrics,
        _nested_shrunk_weighted_prediction,
        grouped_oof,
    )
except ModuleNotFoundError:
    from run_pooled_transition_lifecycle_diagnostic import MIN_POSITIVES, _metrics, _nested_shrunk_weighted_prediction, grouped_oof


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PANEL = ROOT / "data_perp/artifacts/pooled_historical_current_transition_panel_20260730_v1"
DEFAULT_OUTPUT = ROOT / "data_perp/artifacts/pooled_historical_current_transition_classifier_20260730_v1"
SCHEMA = "pooled_historical_current_transition_classifier_v1"
CURRENT_SOURCE = "current_exact_spread_mayjul2026"
TARGETS = (
    "target__active_adverse",
    "target__adverse_onset_within_3h",
    "target__lifecycle_recovery_within_3h",
    "target__lifecycle_reversal_after_recovery_within_3h",
)
TRANSFER_TARGETS = TARGETS[:2]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe(value: Any) -> Any:
    if isinstance(value, (Path, pd.Timestamp)):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_safe(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(_safe(payload), indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def eligible_rows(panel: pd.DataFrame) -> pd.DataFrame:
    required = {"source_family", "mapping_provenance_role", "context_available", "cv_group_id", *TARGETS}
    missing = sorted(required.difference(panel.columns))
    if missing:
        raise ValueError(f"pooled panel lacks classifier fields: {missing}")
    result = panel.loc[panel["context_available"].astype(bool)].copy()
    result = result.loc[
        ~(
            result["source_family"].eq(CURRENT_SOURCE)
            & result["mapping_provenance_role"].ne("strict_oof")
        )
    ].copy()
    if result.empty:
        raise ValueError("no strict eligible transition rows")
    return result.reset_index(drop=True)


def _coverage(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for target in TARGETS:
        availability = f"{target}_available_utc"
        if availability not in frame:
            raise ValueError(f"target lacks exact availability: {target}")
        for source, local in frame.groupby("source_family", sort=True):
            valid = local[target].notna() & local[availability].notna()
            values = pd.to_numeric(local.loc[valid, target], errors="raise")
            rows.append({
                "target": target, "source_family": source, "rows": int(len(local)),
                "resolved_rows": int(valid.sum()), "positive_rows": int(values.sum()),
                "prevalence": float(values.mean()) if len(values) else float("nan"),
                "groups": int(local.loc[valid, "cv_group_id"].nunique()),
                "first_anchor": local.loc[valid, "cohort_anchor_utc"].min() if valid.any() else pd.NaT,
                "last_anchor": local.loc[valid, "cohort_anchor_utc"].max() if valid.any() else pd.NaT,
            })
    return pd.DataFrame(rows)


def pooled_grouped_diagnostic(frame: pd.DataFrame, features: Sequence[str], *, targets: Sequence[str] = TARGETS, models: Sequence[str] = ("logistic", "extra_trees")) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metrics: list[dict[str, Any]] = []
    predictions: list[pd.DataFrame] = []
    skipped: list[dict[str, Any]] = []
    for target in targets:
        for model_name in models:
            try:
                prediction = grouped_oof(frame, target=target, columns=features, model_name=model_name)
            except ValueError as error:
                skipped.append({"target": target, "model": model_name, "reason": str(error)})
                continue
            prediction["target_name"] = target
            prediction["model"] = model_name
            prediction["feature_count"] = len(features)
            predictions.append(prediction)
            metrics.append(_metrics(prediction, target=target, model=model_name, scope="pooled_source_balanced"))
            for source, local in prediction.groupby("source_family", sort=True):
                metrics.append(_metrics(local, target=target, model=model_name, scope=f"source::{source}"))
    return pd.DataFrame(metrics), (pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()), pd.DataFrame(skipped)


def _top10(score: np.ndarray, frame: pd.DataFrame) -> np.ndarray:
    selected = np.zeros(len(frame), dtype=bool)
    count = max(1, int(math.ceil(0.10 * len(frame))))
    order = np.lexsort((pd.to_datetime(frame["cohort_anchor_utc"], utc=True).astype("int64").to_numpy(), -score))
    selected[order[:count]] = True
    return selected


def source_transfer(frame: pd.DataFrame, features: Sequence[str], *, targets: Sequence[str] = TRANSFER_TARGETS, train_sources: Sequence[str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metrics: list[dict[str, Any]] = []
    predictions: list[pd.DataFrame] = []
    skipped: list[dict[str, Any]] = []
    sources = sorted(frame["source_family"].unique())
    selected_train_sources = list(train_sources) if train_sources is not None else sources
    unknown = sorted(set(selected_train_sources).difference(sources))
    if unknown:
        raise ValueError(f"unknown transfer train sources: {unknown}")
    for target in targets:
        availability = f"{target}_available_utc"
        for train_source in selected_train_sources:
            train = frame.loc[frame["source_family"].eq(train_source) & frame[target].notna() & frame[availability].notna()].reset_index(drop=True)
            train_y = pd.to_numeric(train[target], errors="raise").astype(int)
            if len(train) < 60 or int(train_y.sum()) < MIN_POSITIVES or train_y.nunique() < 2 or train["cv_group_id"].nunique() < 3:
                skipped.append({"target": target, "train_source": train_source, "reason": "insufficient_training_support"})
                continue
            for evaluation_source in sources:
                if evaluation_source == train_source:
                    continue
                evaluation = frame.loc[frame["source_family"].eq(evaluation_source) & frame[target].notna() & frame[availability].notna()].reset_index(drop=True)
                evaluation_y = pd.to_numeric(evaluation[target], errors="raise").astype(int)
                if evaluation.empty:
                    skipped.append({"target": target, "train_source": train_source, "evaluation_source": evaluation_source, "reason": "no_evaluation_support"})
                    continue
                try:
                    score, shrink = _nested_shrunk_weighted_prediction(train, train_y, evaluation, features, "logistic")
                except ValueError as error:
                    skipped.append({"target": target, "train_source": train_source, "evaluation_source": evaluation_source, "reason": str(error)})
                    continue
                prediction = evaluation.loc[:, ["cohort_anchor_utc", "source_family", "economics_tier", "mapping_provenance_role", "cv_group_id"]].copy()
                prediction["target"] = evaluation_y.to_numpy(float)
                prediction["prediction"] = score
                prediction["selected_top10"] = _top10(score, evaluation)
                prediction["target_available_utc"] = pd.to_datetime(evaluation[availability], utc=True)
                prediction["target_name"] = target
                prediction["model"] = "logistic_shrunk"
                prediction["train_source"] = train_source
                prediction["evaluation_source"] = evaluation_source
                prediction["calibration_shrinkage_weight"] = shrink
                predictions.append(prediction)
                row = _metrics(prediction, target=target, model="logistic_shrunk", scope="source_transfer")
                row.update({"train_source": train_source, "evaluation_source": evaluation_source, "train_rows": int(len(train)), "train_positive_rows": int(train_y.sum()), "calibration_shrinkage_weight": shrink})
                metrics.append(row)
    return pd.DataFrame(metrics), (pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()), pd.DataFrame(skipped)


def run(args: argparse.Namespace) -> dict[str, Any]:
    panel_root = Path(args.panel)
    panel_path, manifest_path, sidecar = panel_root / "transition_panel.parquet", panel_root / "manifest.json", panel_root / "manifest.sha256"
    if not all(path.is_file() for path in (panel_path, manifest_path, sidecar)):
        raise FileNotFoundError("pooled common transition panel is incomplete")
    if sidecar.read_text(encoding="utf-8").split()[0] != sha256(manifest_path):
        raise ValueError("pooled common transition panel manifest checksum fails")
    upstream = json.loads(manifest_path.read_text(encoding="utf-8"))
    if upstream.get("feature_count") != 90 or len(upstream.get("feature_columns", [])) != 90:
        raise ValueError("pooled panel is not the strict 90-field contract")
    if upstream.get("outputs_sha256", {}).get(panel_path.name) != sha256(panel_path):
        raise ValueError("pooled panel parquet checksum fails")
    features = list(upstream["feature_columns"])
    panel = eligible_rows(pd.read_parquet(panel_path))
    selected_targets = tuple(args.targets or TARGETS)
    selected_models = tuple(args.models or ("logistic", "extra_trees"))
    selected_transfer_targets = tuple(target for target in selected_targets if target in TRANSFER_TARGETS)
    if args.skip_pooled:
        metrics, predictions, skipped = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    else:
        metrics, predictions, skipped = pooled_grouped_diagnostic(panel, features, targets=selected_targets, models=selected_models)
    if args.skip_transfer or not selected_transfer_targets:
        transfer_metrics, transfer_predictions, transfer_skipped = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    else:
        transfer_metrics, transfer_predictions, transfer_skipped = source_transfer(panel, features, targets=selected_transfer_targets, train_sources=args.transfer_train_sources or None)
    coverage = _coverage(panel)
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite immutable output {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(dir=output.parent, prefix=f".{output.name}."))
    paths = {
        "metrics.csv": metrics, "grouped_oof_predictions.parquet": predictions,
        "skipped_arms.csv": skipped, "coverage.csv": coverage,
        "source_transfer_metrics.csv": transfer_metrics,
        "source_transfer_predictions.parquet": transfer_predictions,
        "source_transfer_skipped.csv": transfer_skipped,
    }
    for name, value in paths.items():
        path = temporary / name
        if name.endswith(".parquet"):
            value.to_parquet(path, index=False, compression="zstd")
        else:
            value.to_csv(path, index=False)
    manifest = {
        "schema": SCHEMA, "status": "GROUPED_SOURCE_BALANCED_COMMON_GEOMETRY_DIAGNOSTIC_COMPLETE",
        "eligible_rows": int(len(panel)), "feature_count": len(features), "targets": list(TARGETS),
        "run_filter": {"targets": list(selected_targets), "models": list(selected_models), "skip_pooled": bool(args.skip_pooled), "skip_transfer": bool(args.skip_transfer), "transfer_train_sources": list(args.transfer_train_sources)},
        "contracts": {
            "features": "strict upstream 90-field decision-time whitelist; no source/domain/calendar/provenance/outcome feature",
            "historical": "2022-23 non-OOF backcast remains diagnostic-only and is source-separated in every metric",
            "current": "current fitting/reporting excludes frozen_forward_oos and retains strict_oof only",
            "pooled_cv": "five-fold shuffled seven-day StratifiedGroupKFold, two-sided 36h purge, fold-local preprocessing, source-balanced sample weights",
            "calibration": "nested grouped/36h-purged Brier selection shrinks toward source-balanced training prevalence",
            "transfer": "logistic-only bounded active/onset source-to-source matrix; calibration is selected solely within the training source; reverse-time cells diagnostic-only",
            "promotion": "no walk-forward requirement for diagnosis; no production, admission, timing, policy or portfolio promotion",
        },
        "metric_rows": int(len(metrics)), "prediction_rows": int(len(predictions)),
        "transfer_metric_rows": int(len(transfer_metrics)), "transfer_prediction_rows": int(len(transfer_predictions)),
        "source": {"panel": str(panel_path), "sha256": sha256(panel_path), "manifest_sha256": sha256(manifest_path)},
        "outputs_sha256": {name: sha256(temporary / name) for name in paths},
        "promotion_eligible": False,
    }
    _write_json(temporary / "manifest.json", manifest)
    (temporary / "manifest.sha256").write_text(f"{sha256(temporary / 'manifest.json')}  manifest.json\n")
    os.replace(temporary, output)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--targets", nargs="*", choices=TARGETS, default=[])
    parser.add_argument("--models", nargs="*", choices=("logistic", "extra_trees"), default=[])
    parser.add_argument("--skip-pooled", action="store_true")
    parser.add_argument("--skip-transfer", action="store_true")
    parser.add_argument("--transfer-train-sources", nargs="*", default=[])
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(_safe(run(parse_args())), indent=2, sort_keys=True))
