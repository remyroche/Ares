#!/usr/bin/env python3
"""Fail-closed readiness audit for historical execution-EV reconstruction.

This separates two evidence tiers that must never be merged:

* January--February 2025 can be reconstructed with the exact one-minute,
  12-hour execution path.  January needs a resolved-label warm-up, while
  February has January as its strictly prior training period.
* Late 2024 has hourly OHLCV and point-in-time covariates, but no one-minute
  execution archive.  It is therefore an hourly-path comparator only, never
  evidence of one-minute policy or timing parity.

The old 55-column meta score is deliberately not reconstructed.  Its OOD and
reliability fields are model-derived, and some are outcome/recent-performance
derived.  A historical reconstruction must create a fresh side-local base
candidate score from raw/PIT fields, fitting every learned transform inside the
prior fold.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
AUDIT_SCHEMA = "late2024_execution_ev_reconstruction_readiness_v2"
JANUARY_START = pd.Timestamp("2025-01-01T00:00:00Z")
JANUARY_WARMUP_END = pd.Timestamp("2025-01-15T00:00:00Z")
JANUARY_CANDIDATE_END = pd.Timestamp("2025-01-31T23:00:00Z")
JANUARY_END_RESOLVED = pd.Timestamp("2025-02-01T12:00:00Z")
FEBRUARY_START = pd.Timestamp("2025-02-01T00:00:00Z")
FEBRUARY_CANDIDATE_END = pd.Timestamp("2025-02-28T23:00:00Z")
FEBRUARY_END_RESOLVED = pd.Timestamp("2025-03-01T12:00:00Z")
LATE_2024_START = pd.Timestamp("2024-10-01T00:00:00Z")
LATE_2024_END = pd.Timestamp("2025-01-01T00:00:00Z")
EXACT_HORIZON_MINUTES = 12 * 60

# These are not raw observables.  Reusing their later/frozen values would make
# a historical panel appear to have had a model that did not exist yet.
OLD55_NON_RAW_FIELDS = (
    "meta_sel_ood_abs_z_max",
    "meta_sel_ood_abs_z_mean",
    "meta_sel_ood_abs_z_p95",
    "meta_sel_ood_centroid_l2",
    "meta_sel_ood_iqr_exceed_frac",
    "meta_sel_ood_missing_frac",
    "rel_marginband_clean_rate",
    "rel_marginband_exec_margin_mean",
    "rel_marginband_timeout_rate",
    "rel_rankband_bad_mae_rate",
    "rel_rankband_edge",
    "rel_rankband_exec_margin_mean",
    "rel_rankband_timeout_rate",
)


class ReconstructionReadinessError(ValueError):
    """Raised when an inventory cannot support a safe readiness statement."""


def _safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _utc(value: Any) -> pd.Timestamp | None:
    if value is None or pd.isna(value):
        return None
    result = pd.Timestamp(value)
    return result.tz_localize("UTC") if result.tzinfo is None else result.tz_convert("UTC")


def parquet_timestamp_bounds(
    paths: Iterable[Path], candidates: tuple[str, ...]
) -> dict[str, Any]:
    """Return UTC bounds from a narrow Parquet projection.

    A file lacking the requested timestamp is not silently treated as useful:
    it is reported as unreadable and cannot establish an exact-path tier.
    """

    minimum: pd.Timestamp | None = None
    maximum: pd.Timestamp | None = None
    rows = 0
    files = 0
    unreadable: list[str] = []
    for path in sorted({Path(path) for path in paths}):
        try:
            parquet = pq.ParquetFile(path)
            names = parquet.schema_arrow.names
            column = next((name for name in candidates if name in names), None)
            if column is None:
                unreadable.append(str(path))
                continue
            table = parquet.read(columns=[column])
            local_min = _utc(pc.min(table[column]).as_py())
            local_max = _utc(pc.max(table[column]).as_py())
            if local_min is None or local_max is None:
                unreadable.append(str(path))
                continue
            minimum = local_min if minimum is None else min(minimum, local_min)
            maximum = local_max if maximum is None else max(maximum, local_max)
            rows += int(table.num_rows)
            files += 1
        except Exception:
            unreadable.append(str(path))
    return {
        "files": files,
        "rows": rows,
        "minimum_utc": minimum,
        "maximum_utc": maximum,
        "unreadable_files": unreadable,
    }


def parquet_schema_inventory(paths: Iterable[Path]) -> dict[str, Any]:
    """Report common/union feature columns without reading feature values."""

    common: set[str] | None = None
    union: set[str] = set()
    readable = 0
    unreadable: list[str] = []
    for path in sorted({Path(path) for path in paths}):
        try:
            cols = set(pq.ParquetFile(path).schema_arrow.names)
        except Exception:
            unreadable.append(str(path))
            continue
        readable += 1
        union.update(cols)
        common = cols if common is None else common.intersection(cols)
    return {
        "files": readable,
        "common_columns": sorted(common or set()),
        "union_columns": sorted(union),
        "unreadable_files": unreadable,
    }


def _covers(source: Mapping[str, Any], start: pd.Timestamp, end: pd.Timestamp) -> bool:
    minimum, maximum = _utc(source.get("minimum_utc")), _utc(source.get("maximum_utc"))
    return bool(minimum is not None and maximum is not None and minimum <= start and maximum >= end)


def _has_pit_feature_history(source: Mapping[str, Any], at_or_before: pd.Timestamp) -> bool:
    return bool(
        int(source.get("files", 0)) > 0
        and _utc(source.get("minimum_utc")) is not None
        and _utc(source.get("minimum_utc")) <= at_or_before
    )


def classify_readiness(
    *,
    execution_one_minute: Mapping[str, Any],
    source_labels: Mapping[str, Any],
    pit_features: Mapping[str, Any],
    hourly_ohlcv: Mapping[str, Any],
    archived_candidates: Mapping[str, Any],
    old55_feature_columns: Iterable[str] = (),
) -> dict[str, Any]:
    """Classify the only permissible reconstruction tiers.

    This is intentionally stricter than a source-exists check.  Exact 1m
    labels require complete source coverage through the last label resolution,
    and a forward OOF tier requires both features and prior resolved labels.
    """

    # The source-label table provides pre-entry candidates.  The actual
    # execution target must have a full further 12-hour minute path after the
    # final candidate; checking only the warm-up date would falsely certify a
    # January score window that cannot be evaluated to month end.
    minute_jan = _covers(execution_one_minute, JANUARY_START, JANUARY_END_RESOLVED)
    labels_jan = _covers(source_labels, JANUARY_START, JANUARY_CANDIDATE_END)
    feature_jan = _has_pit_feature_history(pit_features, JANUARY_START)
    january_partial = minute_jan and labels_jan and feature_jan

    minute_feb = _covers(execution_one_minute, JANUARY_START, FEBRUARY_END_RESOLVED)
    labels_feb = _covers(source_labels, JANUARY_START, FEBRUARY_CANDIDATE_END)
    feature_feb = _has_pit_feature_history(pit_features, JANUARY_START)
    february_forward = minute_feb and labels_feb and feature_feb

    hourly_late2024 = _covers(hourly_ohlcv, LATE_2024_START, LATE_2024_END)
    feature_late2024 = _has_pit_feature_history(pit_features, LATE_2024_START)
    minute_bounds_overlap_late2024 = _covers(
        execution_one_minute, LATE_2024_START, LATE_2024_END
    )
    old55_fields = {str(value) for value in old55_feature_columns}
    static_old55_missing = sorted(set(OLD55_NON_RAW_FIELDS) - old55_fields)
    archived_start = _utc(archived_candidates.get("minimum_utc"))

    return {
        "january_2025_exact_1m_12h_expanding_oof": {
            "status": (
                "reconstructible_strict_exact_1m_12h_oof_partial_after_warmup"
                if january_partial
                else "unavailable_missing_exact_1m_labels_or_pit_features"
            ),
            "scope": "January scores begin only after resolved-label warm-up; no full-January OOF claim",
            "warmup_required": {
                "minimum_prior_resolved_labels": "base warm-up January 1-7 plus inner base OOF January 8-14; at each fit use only rows whose 12-hour labels resolved by the fold start",
                "recommended_first_scored_signal_utc": JANUARY_WARMUP_END,
                "purge_and_embargo": "at least the 12-hour execution-label horizon",
            },
            "requirements_met": {
                "execution_one_minute": minute_jan,
                "candidate_label_source_through_january_end": labels_jan,
                "pit_feature_history": feature_jan,
            },
        },
        "february_2025_exact_1m_12h_forward_oof": {
            "status": (
                "reconstructible_strict_exact_1m_12h_forward_oof"
                if february_forward
                else "unavailable_missing_january_prior_or_exact_february_paths"
            ),
            "scope": "train side-local models only on resolved January rows; score February forward",
            "requirements_met": {
                "execution_one_minute_through_february_resolution": minute_feb,
                "candidate_source_through_february_end": labels_feb,
                "pit_feature_history": feature_feb,
            },
        },
        "late_2024_hourly_comparator": {
            "status": (
                "reconstructible_hourly_comparator_only_no_1m_policy_or_timing_parity"
                if hourly_late2024 and feature_late2024
                else "unavailable_missing_hourly_ohlcv_or_pit_features"
            ),
            "scope": "hourly 12h simulation with fee-only historical cost; report separately from exact-1m evidence",
            "requirements_met": {
                "hourly_ohlcv": hourly_late2024,
                "pit_feature_history": feature_late2024,
                "global_one_minute_bounds_overlap": minute_bounds_overlap_late2024,
                "candidate_level_complete_one_minute_universe_certified": False,
            },
            "one_minute_note": "Global store bounds can be set by a few deep-history symbols and never certify the point-in-time candidate universe.",
            "forbidden_claims": [
                "one-minute exit-geometry parity",
                "timing/wait-action parity",
                "combined aggregate metric with exact-1m tier",
                "historical L2/spread parity when source is an OHLCV proxy",
            ],
        },
        "old55_exact_score_contract": {
            "status": "unavailable_must_use_fold_local_raw_pit_base_candidate_score",
            "non_raw_or_model_derived_columns": list(OLD55_NON_RAW_FIELDS),
            "static_feature_columns_missing": static_old55_missing,
            "required_replacement": [
                "fresh per-side base/direct candidate model from raw/PIT features",
                "fold-local imputation, feature selection, HPO and AE/GMM if used",
                "base OOF predictions before execution-EV meta training",
                "no frozen later score, OOD or reliability values",
            ],
        },
        "march_2025_onward": {
            "status": (
                "archived_candidate_stream_available"
                if archived_start is not None and archived_start <= pd.Timestamp("2025-03-01T00:00:00Z")
                else "audit_required"
            ),
            "first_archived_candidate_utc": archived_start,
        },
        "prohibited_substitutions": [
            "frozen future-trained backcast presented as OOF",
            "random within-month folds presented as forward OOS",
            "all-symbol hourly population presented as the base candidate population",
            "96-bar first-touch target substituted for current 12h execution EV",
            "old55 OOD/reliability fields copied backward from a later model",
            "historical OHLCV-derived orderbook proxy treated as historical L2 spread",
        ],
    }


def _files(root: Path, *, recursive: bool = False) -> list[Path]:
    return sorted(root.rglob("*.parquet") if recursive else root.glob("*.parquet"))


def run(args: argparse.Namespace) -> dict[str, Any]:
    roots = {
        "execution_one_minute": (args.execution_one_minute_root, True, ("ts", "timestamp", "__ts__")),
        "source_labels": (args.source_labels_root, False, ("__ts__", "timestamp")),
        "pit_features": (args.pit_features_root, False, ("ts", "__index_level_0__", "__ts__")),
        "hourly_ohlcv": (args.hourly_ohlcv_root, True, ("ts", "timestamp", "__ts__")),
        "archived_candidates": (args.archived_candidates_root, False, ("__ts__", "timestamp")),
    }
    sources: dict[str, dict[str, Any]] = {}
    feature_files: list[Path] = []
    for name, (root, recursive, timestamp_columns) in roots.items():
        root = Path(root)
        paths = _files(root, recursive=recursive) if root.is_dir() else []
        sources[name] = {"root": str(root), **parquet_timestamp_bounds(paths, timestamp_columns)}
        if name == "pit_features":
            feature_files = paths
    feature_schema = parquet_schema_inventory(feature_files)
    report = {
        "schema": AUDIT_SCHEMA,
        "sources": sources,
        "pit_feature_schema": feature_schema,
        "label_contract": {
            "execution_horizon_minutes": EXACT_HORIZON_MINUTES,
            "first_executable_path_bar": "first canonical execution_1m bar at/after decision, matching the immutable policy-label materializer",
            "cost": "current frozen policy fee exactly once; historical L2/spread unavailable",
        },
    }
    report["readiness"] = classify_readiness(
        execution_one_minute=sources["execution_one_minute"],
        source_labels=sources["source_labels"],
        pit_features=sources["pit_features"],
        hourly_ohlcv=sources["hourly_ohlcv"],
        archived_candidates=sources["archived_candidates"],
        old55_feature_columns=feature_schema["union_columns"],
    )
    report["evidence_hashes"] = {
        name: {str(path): _sha256(path) for path in _files(root, recursive=recursive)}
        for name, (root, recursive, _timestamp_columns) in roots.items()
        if name in {"source_labels", "archived_candidates"}
        and Path(root).is_dir()
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "readiness.json").write_text(
        json.dumps(_safe(report), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--execution-one-minute-root",
        type=Path,
        default=ROOT / "data_perp/exchanges/krakenfutures/execution_1m/ohlcv",
    )
    result.add_argument(
        "--source-labels-root",
        type=Path,
        default=ROOT / "data_perp/artifacts/20260720_s59_h5_signalclose_causal_trailing_cost100bps_labels_v2/labels",
    )
    result.add_argument(
        "--pit-features-root", type=Path, default=ROOT / "data_perp/features/20260711_070000"
    )
    result.add_argument(
        "--hourly-ohlcv-root", type=Path, default=ROOT / "data_perp/exchanges/krakenfutures/ohlcv"
    )
    result.add_argument(
        "--archived-candidates-root",
        type=Path,
        default=ROOT / "data_perp/artifacts/20260713_meta_fullhistory_old55_expandedpool/prediction_shards",
    )
    result.add_argument("--output-dir", type=Path, required=True)
    return result


if __name__ == "__main__":
    print(json.dumps(_safe(run(parser().parse_args())), indent=2, sort_keys=True))
