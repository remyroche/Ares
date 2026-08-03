#!/usr/bin/env python3
"""Materialize the strict nine-field historical/current transition geometry.

This is a deliberately narrow bridge between the 2022--2023 frozen candidate
backcast and the current ``cross_era_global_book_transition_research_panel_v4``.
It uses only the nine raw observable fields with exact semantic overlap.  The
historical rows are recovered from the exact-stage source shard and row-number
binding, aggregated at the signal timestamp, and attached to the exact label
candidate identity at its decision time (signal + one hour).  No label,
outcome, representation feature, as-of join, resampling, or fill is allowed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "historical_current_common_transition_geometry_v1"
IDENTITY = ("__ts__", "__symbol__", "side_name", "candidate_id")
DECISION_DELAY = pd.Timedelta(hours=1)
LAGS = (1, 3, 12)
RAW_FIELDS = (
    "atr_compression_ratio",
    "ema20_slope_5h",
    "leverage_build_score",
    "log_bars_since_above_1atr",
    "log_bars_since_above_2atr",
    "memory_asymmetry_1ATR",
    "memory_asymmetry_2ATR",
    "memory_asymmetry_3ATR",
    "trend_acceleration",
)
STATISTICS = ("median", "iqr")

DEFAULT_STAGE = ROOT / "data_perp/artifacts/failure_2022_2023_pf_exact1m_request_stage_20260730_v1/staged_candidates.parquet"
DEFAULT_STAGE_MANIFEST = DEFAULT_STAGE.with_name("manifest.json")
DEFAULT_LABEL_CANDIDATES = ROOT / "data_perp/artifacts/failure_2022_2023_pf_exact1m_label_inputs_20260730_v2/candidates.parquet"
DEFAULT_LABEL_MANIFEST = DEFAULT_LABEL_CANDIDATES.with_name("manifest.json")
DEFAULT_CURRENT_PANEL = ROOT / "data_perp/artifacts/cross_era_global_book_transition_research_panel_20260730_v4/transition_research_panel.parquet"
DEFAULT_CURRENT_MANIFEST = DEFAULT_CURRENT_PANEL.with_name("manifest.json")
DEFAULT_CURRENT_MANIFEST_SIDECAR = DEFAULT_CURRENT_PANEL.with_name("manifest.sha256")
DEFAULT_OUTPUT = ROOT / "data_perp/artifacts/historical_current_common_transition_geometry_20260730_v1"


class CommonGeometryError(RuntimeError):
    """Raised when the semantic common geometry cannot be proven exactly."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_safe(item) for item in value]
    if isinstance(value, (Path, pd.Timestamp, datetime)):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(_safe(dict(payload)), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def canonical_feature_columns() -> tuple[str, ...]:
    """Return the 90 canonical v4 feature names in deterministic order."""

    fields: list[str] = []
    for raw in RAW_FIELDS:
        for statistic in STATISTICS:
            base = f"{statistic}__{raw}"
            fields.extend((
                f"context__state_mean__{base}",
                f"context__state_long_short_gap__{base}",
            ))
            fields.extend(
                f"context__past_delta_{lag}h__{base}" for lag in LAGS
            )
    return tuple(fields)


CANONICAL_FEATURES = canonical_feature_columns()


def semantic_mapping() -> dict[str, Any]:
    """The versioned raw-to-canonical semantic mapping; no inferred aliases."""

    return {
        "schema": "historical_current_common_transition_semantic_mapping_v1",
        "raw_field_overlap": list(RAW_FIELDS),
        "historical_source": {
            "selection": "exact stage source_shard_path + source_row_number, then exact stage/label four-key identity",
            "timestamp": "signal_timestamp",
            "side": "side_name",
            "statistics": {"median": "p50", "iqr": "p75 - p25"},
        },
        "canonical_state": {
            "state_mean": "mean of the long and short per-side statistic, ignoring a missing side only for mean",
            "long_short_gap": "long statistic - short statistic; null if either side is unavailable",
            "past_deltas": "current state mean minus the exact timestamp at 1/3/12 hours earlier; no nearest/asof/resample/fill",
        },
        "signal_to_decision": "historical __decision_ts__ == __ts__ + 1h; features are evaluated at the signal timestamp, matching v4 signal_context_utc == cohort_anchor_utc - 1h",
        "canonical_feature_columns": list(CANONICAL_FEATURES),
        "prohibited": "targets, outcomes, economic fields, DAE, GMM, state IDs, availability-derived model features, resampling, interpolation, ffill, bfill",
    }


def _load_manifest(path: Path, *, name: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{name} manifest is absent: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise CommonGeometryError(f"{name} manifest must be an object")
    return value


def _verify_output_hash(manifest: Mapping[str, Any], path: Path, *, keys: Sequence[str], source: str) -> None:
    current: Any = manifest
    for key in keys:
        if not isinstance(current, Mapping):
            raise CommonGeometryError(f"{source} manifest has no hash binding for {path.name}")
        current = current.get(key)
    if not isinstance(current, str) or current != _sha256(path):
        raise CommonGeometryError(f"{source} manifest hash does not bind {path.name}")


def _canonical_stage(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "signal_timestamp", "decision_timestamp", "symbol", "side_name", "candidate_id",
        "source_shard_path", "source_shard_sha256", "source_row_number",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise CommonGeometryError(f"exact stage lacks required fields: {missing}")
    result = frame.loc[:, sorted(required)].copy()
    result["signal_timestamp"] = pd.to_datetime(result["signal_timestamp"], utc=True, errors="raise")
    result["decision_timestamp"] = pd.to_datetime(result["decision_timestamp"], utc=True, errors="raise")
    result["symbol"] = result["symbol"].astype(str)
    result["side_name"] = result["side_name"].astype(str).str.strip().str.lower()
    result["candidate_id"] = result["candidate_id"].astype(str)
    result["source_shard_path"] = result["source_shard_path"].astype(str)
    result["source_shard_sha256"] = result["source_shard_sha256"].astype(str)
    result["source_row_number"] = pd.to_numeric(result["source_row_number"], errors="raise").astype(np.int64)
    if not result["side_name"].isin(("long", "short")).all():
        raise CommonGeometryError("exact stage has a noncanonical side")
    if result["source_row_number"].lt(0).any():
        raise CommonGeometryError("exact stage has a negative source row number")
    if not result["decision_timestamp"].eq(result["signal_timestamp"] + DECISION_DELAY).all():
        raise CommonGeometryError("exact stage decision timestamp is not signal + 1h")
    keys = ["signal_timestamp", "symbol", "side_name", "candidate_id"]
    if result[keys].isna().any().any() or result.duplicated(keys).any():
        raise CommonGeometryError("exact stage identity is null or duplicated")
    return result


def _canonical_labels(frame: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(set(IDENTITY).difference(frame.columns))
    if missing:
        raise CommonGeometryError(f"exact labels lack identity fields: {missing}")
    result = frame.loc[:, list(IDENTITY)].copy()
    result["__ts__"] = pd.to_datetime(result["__ts__"], utc=True, errors="raise")
    result["__symbol__"] = result["__symbol__"].astype(str)
    result["side_name"] = result["side_name"].astype(str).str.strip().str.lower()
    result["candidate_id"] = result["candidate_id"].astype(str)
    if not result["side_name"].isin(("long", "short")).all():
        raise CommonGeometryError("exact labels have a noncanonical side")
    if result[list(IDENTITY)].isna().any().any() or result.duplicated(list(IDENTITY)).any():
        raise CommonGeometryError("exact labels identity is null or duplicated")
    return result


def _quantile_iqr(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    return float(numeric.quantile(0.75) - numeric.quantile(0.25)) if len(numeric) else np.nan


def reconstruct_selected_raw_rows(stage: pd.DataFrame) -> pd.DataFrame:
    """Recover just the exact staged source rows and prove their identity binding."""

    canonical = _canonical_stage(stage)
    pieces: list[pd.DataFrame] = []
    requested_columns = ["__ts__", "__symbol__", "side_name", *RAW_FIELDS]
    for source_path, wanted in canonical.groupby("source_shard_path", sort=True):
        path = Path(source_path)
        if not path.is_file():
            raise FileNotFoundError(f"exact stage source shard is absent: {path}")
        declared_hashes = set(wanted["source_shard_sha256"])
        if len(declared_hashes) != 1 or next(iter(declared_hashes)) != _sha256(path):
            raise CommonGeometryError(f"exact stage source shard hash fails: {path}")
        schema = set(pq.ParquetFile(path).schema_arrow.names)
        missing = sorted(set(requested_columns).difference(schema))
        if missing:
            raise CommonGeometryError(f"source shard lacks common raw fields: {missing}")
        raw = pq.read_table(path, columns=requested_columns).to_pandas()
        row_ids = wanted["source_row_number"].to_numpy(dtype=np.int64)
        if int(row_ids.max()) >= len(raw):
            raise CommonGeometryError(f"exact stage source row is beyond shard length: {path}")
        selected = raw.iloc[row_ids].reset_index(drop=True).copy()
        selected["__ts__"] = pd.to_datetime(selected["__ts__"], utc=True, errors="raise")
        selected["__symbol__"] = selected["__symbol__"].astype(str)
        selected["side_name"] = selected["side_name"].astype(str).str.strip().str.lower()
        expected = wanted.reset_index(drop=True)
        if not selected["__ts__"].equals(expected["signal_timestamp"]):
            raise CommonGeometryError(f"source timestamp does not match exact stage: {path}")
        if not selected["__symbol__"].equals(expected["symbol"]) or not selected["side_name"].equals(expected["side_name"]):
            raise CommonGeometryError(f"source identity does not match exact stage: {path}")
        selected["candidate_id"] = expected["candidate_id"].to_numpy()
        pieces.append(selected.loc[:, ["__ts__", "__symbol__", "side_name", "candidate_id", *RAW_FIELDS]])
    result = pd.concat(pieces, ignore_index=True)
    result = result.merge(
        canonical.loc[:, ["signal_timestamp", "symbol", "side_name", "candidate_id"]],
        left_on=["__ts__", "__symbol__", "side_name", "candidate_id"],
        right_on=["signal_timestamp", "symbol", "side_name", "candidate_id"],
        how="inner", validate="one_to_one",
    ).drop(columns=["signal_timestamp", "symbol"])
    if len(result) != len(canonical):
        raise CommonGeometryError("source-row recovery did not retain every exact stage row")
    return result


def build_historical_hourly_state(selected_raw: pd.DataFrame) -> pd.DataFrame:
    """Aggregate strict per-side medians/IQRs and canonical state geometry."""

    required = {"__ts__", "side_name", *RAW_FIELDS}
    missing = sorted(required.difference(selected_raw.columns))
    if missing:
        raise CommonGeometryError(f"selected raw rows lack required fields: {missing}")
    work = selected_raw.loc[:, ["__ts__", "side_name", *RAW_FIELDS]].copy()
    work["__ts__"] = pd.to_datetime(work["__ts__"], utc=True, errors="raise")
    work["side_name"] = work["side_name"].astype(str).str.lower()
    if not work["side_name"].isin(("long", "short")).all():
        raise CommonGeometryError("selected raw rows have a noncanonical side")
    for field in RAW_FIELDS:
        work[field] = pd.to_numeric(work[field], errors="coerce")
    grouped = work.groupby(["__ts__", "side_name"], sort=True, observed=True)
    median = grouped[list(RAW_FIELDS)].median().rename(columns=lambda field: f"median__{field}")
    iqr = grouped[list(RAW_FIELDS)].agg(_quantile_iqr).rename(columns=lambda field: f"iqr__{field}")
    per_side = median.join(iqr).reset_index()
    primitive = [f"{statistic}__{field}" for field in RAW_FIELDS for statistic in STATISTICS]
    wide_parts: dict[str, pd.DataFrame] = {}
    for side in ("long", "short"):
        local = per_side.loc[per_side["side_name"].eq(side), ["__ts__", *primitive]].copy()
        wide_parts[side] = local.rename(columns={column: f"{column}__{side}" for column in primitive})
    wide = wide_parts["long"].merge(wide_parts["short"], on="__ts__", how="outer", validate="one_to_one").sort_values("__ts__", kind="stable").reset_index(drop=True)
    output = pd.DataFrame({"signal_context_utc": wide["__ts__"]})
    for field in RAW_FIELDS:
        for statistic in STATISTICS:
            base = f"{statistic}__{field}"
            long_values = pd.to_numeric(wide[f"{base}__long"], errors="coerce")
            short_values = pd.to_numeric(wide[f"{base}__short"], errors="coerce")
            mean_name = f"context__state_mean__{base}"
            gap_name = f"context__state_long_short_gap__{base}"
            output[mean_name] = pd.concat([long_values, short_values], axis=1).mean(axis=1)
            output[gap_name] = long_values - short_values
    indexed = output.set_index("signal_context_utc", drop=False)
    mean_features = [name for name in indexed.columns if name.startswith("context__state_mean__")]
    for lag in LAGS:
        prior = indexed.reindex(indexed.index - pd.Timedelta(hours=lag)).set_axis(indexed.index)
        for mean_name in mean_features:
            base = mean_name.removeprefix("context__state_mean__")
            indexed[f"context__past_delta_{lag}h__{base}"] = indexed[mean_name] - prior[mean_name]
    result = indexed.reset_index(drop=True)
    missing_canonical = sorted(set(CANONICAL_FEATURES).difference(result.columns))
    if missing_canonical:
        raise CommonGeometryError(f"historical geometry did not emit canonical fields: {missing_canonical}")
    return result.loc[:, ["signal_context_utc", *CANONICAL_FEATURES]]


def attach_historical_candidates(labels: pd.DataFrame, stage: pd.DataFrame, hourly: pd.DataFrame) -> pd.DataFrame:
    """Attach signal-time state geometry to exact label identity without filling."""

    exact_labels = _canonical_labels(labels)
    exact_stage = _canonical_stage(stage)
    stage_identity = exact_stage.rename(columns={"signal_timestamp": "__ts__", "symbol": "__symbol__"})
    stage_identity = stage_identity.loc[:, [*IDENTITY, "decision_timestamp"]]
    checked = exact_labels.merge(stage_identity, on=list(IDENTITY), how="left", validate="one_to_one", indicator=True)
    if not checked["_merge"].eq("both").all() or len(checked) != len(exact_labels):
        raise CommonGeometryError("exact label candidates do not exactly match stage identity")
    checked = checked.drop(columns="_merge")
    checked["__decision_ts__"] = checked["__ts__"] + DECISION_DELAY
    if not checked["__decision_ts__"].eq(checked["decision_timestamp"]).all():
        raise CommonGeometryError("label signal-to-decision timing disagrees with exact stage")
    source = hourly.copy()
    source["signal_context_utc"] = pd.to_datetime(source["signal_context_utc"], utc=True, errors="raise")
    output = checked.merge(source, left_on="__ts__", right_on="signal_context_utc", how="left", validate="many_to_one", sort=False)
    if len(output) != len(exact_labels) or not output.loc[:, list(IDENTITY)].equals(exact_labels.loc[:, list(IDENTITY)]):
        raise CommonGeometryError("historical candidate attachment changed exact identity order")
    output["common_transition_context_available"] = output[list(CANONICAL_FEATURES)].notna().any(axis=1)
    return output.loc[:, [*IDENTITY, "__decision_ts__", "common_transition_context_available", *CANONICAL_FEATURES]].reset_index(drop=True)


def project_current_v4_context(panel: pd.DataFrame, manifest_features: Sequence[str]) -> pd.DataFrame:
    """Project the current v4 panel onto this common geometry, never labels."""

    missing_manifest = sorted(set(CANONICAL_FEATURES).difference(manifest_features))
    if missing_manifest:
        raise CommonGeometryError(f"current v4 feature contract lacks common canonical fields: {missing_manifest}")
    required = {"signal_context_utc", "context_available", *CANONICAL_FEATURES}
    missing = sorted(required.difference(panel.columns))
    if missing:
        raise CommonGeometryError(f"current v4 panel lacks common canonical fields: {missing}")
    work = panel.loc[:, ["signal_context_utc", "context_available", *CANONICAL_FEATURES]].copy()
    work["signal_context_utc"] = pd.to_datetime(work["signal_context_utc"], utc=True, errors="raise")
    work["context_available"] = work["context_available"].astype(bool)
    for field in CANONICAL_FEATURES:
        work[field] = pd.to_numeric(work[field], errors="coerce")
    rows: list[pd.Series] = []
    for timestamp, group in work.groupby("signal_context_utc", sort=True, observed=True):
        if group["context_available"].nunique(dropna=False) != 1:
            raise CommonGeometryError(f"current v4 availability is inconsistent at {timestamp}")
        record: dict[str, Any] = {"signal_context_utc": timestamp, "common_transition_context_available": bool(group["context_available"].iloc[0])}
        for field in CANONICAL_FEATURES:
            values = group[field]
            non_null = values.dropna()
            if non_null.nunique() > 1:
                raise CommonGeometryError(f"current v4 canonical field is inconsistent at {timestamp}: {field}")
            record[field] = non_null.iloc[0] if len(non_null) else np.nan
        rows.append(pd.Series(record))
    result = pd.DataFrame(rows)
    if result.empty:
        raise CommonGeometryError("current v4 projection is empty")
    if result.loc[~result["common_transition_context_available"], list(CANONICAL_FEATURES)].notna().any().any():
        raise CommonGeometryError("unavailable current v4 context cannot have common feature values")
    return result.loc[:, ["signal_context_utc", "common_transition_context_available", *CANONICAL_FEATURES]]


def _coverage(frame: pd.DataFrame, *, timestamp: str, availability: str) -> list[dict[str, Any]]:
    work = frame.loc[:, [timestamp, availability]].copy()
    value = pd.to_datetime(work[timestamp], utc=True, errors="raise")
    work["year_month"] = value.dt.strftime("%Y-%m")
    result = work.groupby("year_month", sort=True)[availability].agg(rows="size", available_rows="sum").reset_index()
    result["unavailable_rows"] = result["rows"] - result["available_rows"]
    result["available_rate"] = result["available_rows"] / result["rows"]
    return [_safe(record) for record in result.to_dict(orient="records")]


def _identity_hash(frame: pd.DataFrame) -> str:
    values = pd.util.hash_pandas_object(frame.loc[:, list(IDENTITY)], index=False)
    return hashlib.sha256(values.to_numpy(dtype="uint64").tobytes()).hexdigest()


def run(
    *,
    stage_path: Path = DEFAULT_STAGE,
    stage_manifest_path: Path = DEFAULT_STAGE_MANIFEST,
    label_candidates_path: Path = DEFAULT_LABEL_CANDIDATES,
    label_manifest_path: Path = DEFAULT_LABEL_MANIFEST,
    current_panel_path: Path = DEFAULT_CURRENT_PANEL,
    current_manifest_path: Path = DEFAULT_CURRENT_MANIFEST,
    current_manifest_sidecar_path: Path = DEFAULT_CURRENT_MANIFEST_SIDECAR,
    destination: Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    """Atomically materialize the strict common geometry and its audit."""

    if destination.exists():
        raise FileExistsError(f"refusing to overwrite immutable common geometry: {destination}")
    for path in (stage_path, stage_manifest_path, label_candidates_path, label_manifest_path, current_panel_path, current_manifest_path, current_manifest_sidecar_path):
        if not Path(path).is_file():
            raise FileNotFoundError(path)
    stage_manifest = _load_manifest(stage_manifest_path, name="exact stage")
    label_manifest = _load_manifest(label_manifest_path, name="exact labels")
    _verify_output_hash(stage_manifest, stage_path, keys=("outputs", "staged_candidates", "sha256"), source="exact stage")
    _verify_output_hash(label_manifest, label_candidates_path, keys=("outputs", "candidates", "sha256"), source="exact labels")
    current_manifest = _load_manifest(current_manifest_path, name="current v4 panel")
    signed = current_manifest_sidecar_path.read_text(encoding="utf-8").split()
    if not signed or signed[0] != _sha256(current_manifest_path):
        raise CommonGeometryError("current v4 manifest sidecar checksum fails")
    _verify_output_hash(current_manifest, current_panel_path, keys=("outputs", "panel", "sha256"), source="current v4 panel")
    manifest_features = current_manifest.get("feature_columns")
    if not isinstance(manifest_features, list) or len(manifest_features) != len(set(manifest_features)):
        raise CommonGeometryError("current v4 manifest has an invalid feature whitelist")

    stage = pd.read_parquet(stage_path)
    labels = pd.read_parquet(label_candidates_path)
    selected_raw = reconstruct_selected_raw_rows(stage)
    historical_hourly = build_historical_hourly_state(selected_raw)
    historical_candidates = attach_historical_candidates(labels, stage, historical_hourly)
    panel = pd.read_parquet(current_panel_path, columns=["signal_context_utc", "context_available", *CANONICAL_FEATURES])
    current_context = project_current_v4_context(panel, manifest_features)
    if set(historical_hourly.columns).difference({"signal_context_utc", *CANONICAL_FEATURES}):
        raise CommonGeometryError("historical output contains nonsemantic geometry fields")

    stage_canonical = _canonical_stage(stage)
    raw_overlap = set(RAW_FIELDS)
    # The current raw geometry source is recorded in the v4 manifest.  The v4
    # feature names prove the canonical projection rather than silently relying
    # on an independently changing source schema.
    audit = {
        "schema": "historical_current_common_transition_geometry_audit_v1",
        "status": "MATERIALIZED_STRICT_SEMANTIC_COMMON_GEOMETRY",
        "raw_name_overlap": {"count": len(raw_overlap), "fields": list(RAW_FIELDS), "exact_expected_nine": True},
        "canonical_parity": {
            "feature_count": len(CANONICAL_FEATURES),
            "all_common_features_declared_by_current_v4": set(CANONICAL_FEATURES).issubset(set(manifest_features)),
            "historical_columns_equal_contract": list(historical_hourly.columns[1:]) == list(CANONICAL_FEATURES),
            "current_columns_equal_contract": list(current_context.columns[2:]) == list(CANONICAL_FEATURES),
        },
        "timing": semantic_mapping()["signal_to_decision"],
        "no_fill": "historical lags use exact timestamp reindex; candidate attachment is exact signal key; no asof/resample/interpolation/ffill/bfill",
        "excluded": semantic_mapping()["prohibited"],
        "historical_source_recovery": {
            "stage_rows": int(len(stage_canonical)),
            "selected_raw_rows": int(len(selected_raw)),
            "source_shards": int(stage_canonical["source_shard_path"].nunique()),
            "stage_to_label_identity_exact": True,
        },
        "coverage": {
            "historical_candidates": _coverage(historical_candidates, timestamp="__decision_ts__", availability="common_transition_context_available"),
            "current_v4_unique_signal_context": _coverage(current_context, timestamp="signal_context_utc", availability="common_transition_context_available"),
        },
    }
    stage_dir = destination.parent / f".{destination.name}.staging-{uuid.uuid4().hex}"
    try:
        stage_dir.mkdir(parents=True)
        outputs = {
            "historical_candidate_context": stage_dir / "historical_candidate_context.parquet",
            "historical_hourly_state_geometry": stage_dir / "historical_hourly_state_geometry.parquet",
            "current_v4_semantic_context": stage_dir / "current_v4_semantic_context.parquet",
        }
        historical_candidates.to_parquet(outputs["historical_candidate_context"], index=False, compression="zstd", compression_level=5)
        historical_hourly.to_parquet(outputs["historical_hourly_state_geometry"], index=False, compression="zstd", compression_level=5)
        current_context.to_parquet(outputs["current_v4_semantic_context"], index=False, compression="zstd", compression_level=5)
        _write_json(stage_dir / "semantic_mapping.json", semantic_mapping())
        _write_json(stage_dir / "audit.json", audit)
        output_manifest = {
            key: {
                "path": str(destination / path.name), "sha256": _sha256(path),
                "rows": int(pq.ParquetFile(path).metadata.num_rows), "columns": int(len(pq.ParquetFile(path).schema_arrow.names)),
            }
            for key, path in outputs.items()
        }
        output_manifest["historical_candidate_context"]["candidate_identity_sha256"] = _identity_hash(historical_candidates)
        report = {
            "schema": SCHEMA,
            "status": audit["status"],
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "sources": {
                "exact_stage": {"path": str(stage_path), "sha256": _sha256(stage_path), "manifest_sha256": _sha256(stage_manifest_path)},
                "exact_label_candidates": {"path": str(label_candidates_path), "sha256": _sha256(label_candidates_path), "manifest_sha256": _sha256(label_manifest_path)},
                "current_v4_panel": {"path": str(current_panel_path), "sha256": _sha256(current_panel_path), "manifest_sha256": _sha256(current_manifest_path)},
            },
            "outputs": output_manifest,
            "semantic_mapping": {"path": str(destination / "semantic_mapping.json"), "sha256": _sha256(stage_dir / "semantic_mapping.json")},
            "audit": {"path": str(destination / "audit.json"), "sha256": _sha256(stage_dir / "audit.json")},
            "research_only": True,
            "promotion_eligible": False,
        }
        _write_json(stage_dir / "report.json", report)
        manifest = {
            "schema": SCHEMA,
            "status": report["status"],
            "report": {"path": "report.json", "sha256": _sha256(stage_dir / "report.json")},
            "semantic_mapping": {"path": "semantic_mapping.json", "sha256": _sha256(stage_dir / "semantic_mapping.json")},
            "audit": {"path": "audit.json", "sha256": _sha256(stage_dir / "audit.json")},
            "outputs": output_manifest,
            "sources": report["sources"],
            "atomic_publication": "all files are staged, hash-bound, then directory-renamed once",
        }
        _write_json(stage_dir / "manifest.json", manifest)
        (stage_dir / "manifest.sha256").write_text(f"{_sha256(stage_dir / 'manifest.json')}  manifest.json\n", encoding="utf-8")
        os.replace(stage_dir, destination)
        return report
    except BaseException:
        shutil.rmtree(stage_dir, ignore_errors=True)
        raise


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--stage", type=Path, default=DEFAULT_STAGE)
    value.add_argument("--stage-manifest", type=Path, default=DEFAULT_STAGE_MANIFEST)
    value.add_argument("--label-candidates", type=Path, default=DEFAULT_LABEL_CANDIDATES)
    value.add_argument("--label-manifest", type=Path, default=DEFAULT_LABEL_MANIFEST)
    value.add_argument("--current-panel", type=Path, default=DEFAULT_CURRENT_PANEL)
    value.add_argument("--current-manifest", type=Path, default=DEFAULT_CURRENT_MANIFEST)
    value.add_argument("--current-manifest-sidecar", type=Path, default=DEFAULT_CURRENT_MANIFEST_SIDECAR)
    value.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return value


if __name__ == "__main__":
    arguments = parser().parse_args()
    print(json.dumps(_safe(run(
        stage_path=arguments.stage,
        stage_manifest_path=arguments.stage_manifest,
        label_candidates_path=arguments.label_candidates,
        label_manifest_path=arguments.label_manifest,
        current_panel_path=arguments.current_panel,
        current_manifest_path=arguments.current_manifest,
        current_manifest_sidecar_path=arguments.current_manifest_sidecar,
        destination=arguments.output_dir,
    )), indent=2, sort_keys=True))
