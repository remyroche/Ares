#!/usr/bin/env python3
"""Fixed sparse-mechanism screen on the strict common 90-field geometry.

This is deliberately not feature selection or HPO.  The eight feature arms
below are named before reading outcomes: all90 is a control; the remaining
arms are compact, interpretable transition mechanisms.  Every score is a
diagnostic probability, never a veto or production routing signal.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

try:
    from scripts.audit_pooled_historical_current_transition_score_ties import dispersion, tie_aware_top10
    from scripts.run_pooled_historical_current_transition_classifier import (
        CURRENT_SOURCE,
        eligible_rows,
        pooled_grouped_diagnostic,
        source_transfer,
    )
    from scripts.run_pooled_transition_lifecycle_diagnostic import _metrics
except ModuleNotFoundError:  # direct execution from scripts/
    from audit_pooled_historical_current_transition_score_ties import dispersion, tie_aware_top10
    from run_pooled_historical_current_transition_classifier import CURRENT_SOURCE, eligible_rows, pooled_grouped_diagnostic, source_transfer
    from run_pooled_transition_lifecycle_diagnostic import _metrics


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PANEL = ROOT / "data_perp/artifacts/pooled_historical_current_transition_panel_20260730_v1"
DEFAULT_OUTPUT = ROOT / "data_perp/artifacts/pooled_historical_current_sparse_transition_mechanism_ablation_20260730_v1"
SCHEMA = "pooled_historical_current_sparse_transition_mechanism_ablation_v1"
TARGETS = ("target__active_adverse", "target__adverse_onset_within_3h")
HISTORICAL_SOURCE = "historical_backcast_2022_2023_non_oof"
RECONSTRUCTED_SOURCE = "reconstructed_exact1m_janapr2025"
TRANSFER_SOURCES = (RECONSTRUCTED_SOURCE, HISTORICAL_SOURCE)
MODEL = "logistic"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def safe(value: Any) -> Any:
    if isinstance(value, (Path, pd.Timestamp)):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(safe(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _fields(features: Sequence[str], *, raw_names: Sequence[str] = (), prefix: str | None = None, suffixes: Sequence[str] = ()) -> list[str]:
    selected: list[str] = []
    for field in features:
        raw_match = any(field.endswith(f"__{name}") for name in raw_names)
        prefix_match = prefix is not None and field.startswith(prefix)
        suffix_match = bool(suffixes) and any(field.startswith("context__") and field.endswith(suffix) for suffix in suffixes)
        if raw_match or prefix_match or suffix_match:
            selected.append(field)
    return selected


def feature_arms(features: Sequence[str]) -> dict[str, list[str]]:
    """Return only predefined, named semantic groups from the 90-field contract."""

    features = list(features)
    if len(features) != 90 or len(set(features)) != 90:
        raise ValueError("requires exact strict 90-field common geometry")
    compression = _fields(features, raw_names=("atr_compression_ratio",))
    trend = _fields(features, raw_names=("ema20_slope_5h", "trend_acceleration"))
    leverage = _fields(features, raw_names=("leverage_build_score",))
    memory_range = _fields(features, raw_names=(
        "log_bars_since_above_1atr", "log_bars_since_above_2atr",
        "memory_asymmetry_1ATR", "memory_asymmetry_2ATR", "memory_asymmetry_3ATR",
    ))
    state_levels = [field for field in features if field.startswith("context__state_mean__median__")]
    short_deltas = [field for field in features if "__past_delta_1h__" in field or "__past_delta_3h__" in field]
    compact_raw = (
        "atr_compression_ratio", "ema20_slope_5h", "trend_acceleration", "leverage_build_score",
        "log_bars_since_above_1atr", "log_bars_since_above_2atr",
        "memory_asymmetry_1ATR", "memory_asymmetry_2ATR", "memory_asymmetry_3ATR",
    )
    compact = [field for field in features if (
        (field.startswith("context__state_mean__median__") and any(field.endswith(f"__{raw}") for raw in compact_raw))
        or (("__past_delta_1h__" in field or "__past_delta_3h__" in field) and any(field.endswith(f"__{raw}") for raw in compact_raw))
    )]
    arms = {
        "all90_control": features,
        "compression_release": compression,
        "trend_ema_acceleration": trend,
        "leverage_build": leverage,
        "memory_range_recurrence": memory_range,
        "sparse_state_levels": state_levels,
        "short_1h_3h_deltas": short_deltas,
        "compact_union": compact,
    }
    expected = {
        "compression_release": 10, "trend_ema_acceleration": 20, "leverage_build": 10,
        "memory_range_recurrence": 50, "sparse_state_levels": 9, "short_1h_3h_deltas": 36,
        "compact_union": 45,
    }
    if {name: len(values) for name, values in arms.items() if name != "all90_control"} != expected:
        raise ValueError(f"semantic feature-group construction changed: { {name: len(values) for name, values in arms.items()} }")
    return arms


def _tie_record(prediction: pd.DataFrame, *, arm: str, evaluation_kind: str, train_source: str) -> dict[str, Any]:
    record = {"arm": arm, "evaluation_kind": evaluation_kind, "train_source": train_source}
    record.update(dispersion(prediction)); record.update(tie_aware_top10(prediction))
    record["interpretation"] = "NON_RANKING_CONSTANT_OR_ZERO_SHRINK" if not record["ranking_informative"] else (
        "CUTOFF_TIE_AMBIGUOUS_DIAGNOSTIC_ONLY" if record["cutoff_is_ambiguous"] else "TIE_AWARE_RANKING_DIAGNOSTIC_ONLY"
    )
    return record


def _metrics_and_ties(prediction: pd.DataFrame, *, arm: str, evaluation_kind: str, train_source: str) -> tuple[dict[str, Any], dict[str, Any]]:
    target = str(prediction.target_name.iloc[0])
    metric = _metrics(prediction, target=target, model=MODEL, scope=evaluation_kind)
    metric.update({"arm": arm, "evaluation_kind": evaluation_kind, "train_source": train_source, "feature_count": int(prediction.feature_count.iloc[0])})
    tie = _tie_record(prediction, arm=arm, evaluation_kind=evaluation_kind, train_source=train_source)
    tie.update({"target": target, "model": MODEL, "feature_count": int(prediction.feature_count.iloc[0])})
    return metric, tie


def run_ablation(panel: pd.DataFrame, features: Sequence[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    arms = feature_arms(features)
    metrics: list[dict[str, Any]] = []
    ties: list[dict[str, Any]] = []
    predictions: list[pd.DataFrame] = []
    skipped: list[dict[str, Any]] = []
    current = panel.loc[panel.source_family.eq(CURRENT_SOURCE)].copy()
    if current.empty or current.mapping_provenance_role.ne("strict_oof").any():
        raise ValueError("current strict-OOF cohort is unavailable or contaminated")
    for arm, columns in arms.items():
        # Pooled grouped/purged OOF: source-balanced fitting is inherited from
        # the original classifier.  It is the common-geometry control track.
        pooled_metrics, pooled_predictions, pooled_skipped = pooled_grouped_diagnostic(
            panel, columns, targets=TARGETS, models=(MODEL,)
        )
        for _, item in pooled_predictions.groupby("target_name", sort=True):
            item = item.copy(); item["arm"] = arm; item["evaluation_kind"] = "pooled_grouped_oof"; item["train_source"] = "POOLED"
            metric, tie = _metrics_and_ties(item, arm=arm, evaluation_kind="pooled_grouped_oof", train_source="POOLED")
            metrics.append(metric); ties.append(tie); predictions.append(item)
        for _, item in pooled_skipped.iterrows():
            skipped.append({"arm": arm, "evaluation_kind": "pooled_grouped_oof", **item.to_dict()})

        # This is a separate within-current fit/evaluation.  It uses only the
        # frozen strict-OOF current rows and the same grouped/purged protocol.
        current_metrics, current_predictions, current_skipped = pooled_grouped_diagnostic(
            current, columns, targets=TARGETS, models=(MODEL,)
        )
        for _, item in current_predictions.groupby("target_name", sort=True):
            item = item.copy(); item["arm"] = arm; item["evaluation_kind"] = "current_strict_oof_within_source"; item["train_source"] = CURRENT_SOURCE
            metric, tie = _metrics_and_ties(item, arm=arm, evaluation_kind="current_strict_oof_within_source", train_source=CURRENT_SOURCE)
            metrics.append(metric); ties.append(tie); predictions.append(item)
        for _, item in current_skipped.iterrows():
            skipped.append({"arm": arm, "evaluation_kind": "current_strict_oof_within_source", **item.to_dict()})

        # Transfer runs intentionally test only the two requested sources into
        # current, excluding the canonical source and avoiding a transfer grid.
        for train_source in TRANSFER_SOURCES:
            transfer_frame = panel.loc[panel.source_family.isin((train_source, CURRENT_SOURCE))].copy()
            transfer_metrics, transfer_predictions, transfer_skipped = source_transfer(
                transfer_frame, columns, targets=TARGETS, train_sources=(train_source,)
            )
            for _, item in transfer_predictions.groupby("target_name", sort=True):
                item = item.copy(); item["feature_count"] = len(columns); item["arm"] = arm; item["evaluation_kind"] = "transfer_into_current"; item["train_source"] = train_source
                metric, tie = _metrics_and_ties(item, arm=arm, evaluation_kind="transfer_into_current", train_source=train_source)
                metrics.append(metric); ties.append(tie); predictions.append(item)
            for _, item in transfer_skipped.iterrows():
                skipped.append({"arm": arm, "evaluation_kind": "transfer_into_current", "train_source": train_source, **item.to_dict()})
    catalog = pd.DataFrame([{"arm": arm, "feature_count": len(columns), "features": json.dumps(columns)} for arm, columns in arms.items()])
    return pd.DataFrame(metrics), (pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()), pd.DataFrame(ties), pd.DataFrame(skipped), catalog


def run(*, panel_root: Path, output_dir: Path) -> dict[str, Any]:
    panel_path, manifest_path, seal_path = panel_root / "transition_panel.parquet", panel_root / "manifest.json", panel_root / "manifest.sha256"
    if not all(path.is_file() for path in (panel_path, manifest_path, seal_path)):
        raise FileNotFoundError("strict common transition panel is incomplete")
    if seal_path.read_text(encoding="utf-8").split()[0] != sha256(manifest_path):
        raise ValueError("strict common transition panel manifest checksum fails")
    upstream = json.loads(manifest_path.read_text(encoding="utf-8"))
    features = list(upstream.get("feature_columns", []))
    if len(features) != 90 or upstream.get("outputs_sha256", {}).get(panel_path.name) != sha256(panel_path):
        raise ValueError("strict 90-field panel contract/checksum fails")
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite immutable output {output_dir}")
    panel = eligible_rows(pd.read_parquet(panel_path))
    metrics, predictions, ties, skipped, catalog = run_ablation(panel, features)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(dir=output_dir.parent, prefix=f".{output_dir.name}."))
    outputs = {
        "metrics.csv": metrics, "predictions.parquet": predictions, "tie_aware_top10.csv": ties,
        "skipped_arms.csv": skipped, "feature_arm_catalog.csv": catalog,
    }
    for name, value in outputs.items():
        path = temporary / name
        if name.endswith(".parquet"):
            value.to_parquet(path, index=False, compression="zstd")
        else:
            value.to_csv(path, index=False)
    manifest = {
        "schema": SCHEMA, "status": "FIXED_SPARSE_MECHANISM_ABLATION_DIAGNOSTIC_COMPLETE", "promotion_eligible": False,
        "feature_arms": {item["arm"]: int(item["feature_count"]) for item in catalog.to_dict("records")},
        "targets": list(TARGETS), "model": MODEL,
        "contracts": {
            "features": "all90 control plus seven fixed semantic groups from exact decision-time common 90-field geometry; no data-driven feature selection or HPO",
            "pooled_oof": "five-fold shuffled seven-day grouped CV, two-sided 36h purge, fold-local preprocessing and source-balanced fit/calibration weights inherited from the frozen classifier contract",
            "current": "separate within-current grouped/purged diagnostic uses strict_oof current rows only; frozen_forward_oos is excluded before every fit",
            "transfer": "only reconstructed Jan-Apr 2025 -> current and historical 2022-23 -> current, logistic-only; historical remains non-OOF diagnostic evidence",
            "tie_aware_top10": "top10 is a single pooled-global ranking within each prediction arm. Constant/zero-shrink scores are non-ranking; cutoff ties report expected precision/lift and exact precision bounds rather than timestamp tie-break lift",
            "promotion": "no veto, policy, timing, admission, portfolio or production use; no promotion inference",
        },
        "eligible_rows": int(len(panel)), "metric_rows": int(len(metrics)), "prediction_rows": int(len(predictions)), "tie_rows": int(len(ties)), "skipped_rows": int(len(skipped)),
        "source": {"panel": str(panel_path), "panel_sha256": sha256(panel_path), "panel_manifest_sha256": sha256(manifest_path)},
        "outputs_sha256": {name: sha256(temporary / name) for name in outputs},
        "runner": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__).resolve())},
    }
    write_json(temporary / "manifest.json", manifest)
    (temporary / "manifest.sha256").write_text(f"{sha256(temporary / 'manifest.json')}  manifest.json\n", encoding="utf-8")
    os.replace(temporary, output_dir)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    print(json.dumps(safe(run(panel_root=arguments.panel, output_dir=arguments.output_dir)), sort_keys=True))
