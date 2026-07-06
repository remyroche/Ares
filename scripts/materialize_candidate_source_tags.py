#!/usr/bin/env python3
"""Materialize causal candidate source tags and diagnostic quality labels.

This script is diagnostic-only. Source archetype scores and tags are computed
only from prediction-time observable columns. Realized outcomes are used later,
only for source quality diagnostics and candidate supervised labels/weights.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/source_quality_labels.yaml"
DEFAULT_OUTDIR = ROOT / "artifacts/source_tags"

OUTCOME_LIKE_RE = re.compile(
    r"(future|fwd|mfe|mae|pnl|profit|utility|target|label|oracle|timeout|realized|outcome|barrier_result|__y_|__r_policy|__u_policy)",
    re.IGNORECASE,
)
FORBIDDEN_PROXY_RE = re.compile(
    r"(future|fwd|mfe|mae|pnl|profit|target|label|oracle|hit|timeout|realized|outcome|barrier_result|__y_|__r_policy|__u_policy)",
    re.IGNORECASE,
)
RAW_PORTABILITY_RE = re.compile(r"(^|_)(open|high|low|close|price|volume)$", re.IGNORECASE)
SOURCE_ROW_IDX_COL = "__source_row_idx__"
SOURCE_KEY_COL = "__source_key__"
DEFAULT_PROXY_SCORE_CANDIDATES = [
    "proxy_score",
    "model_score",
    "pred_score",
    "p_hat",
    "y_hat",
    "ev",
    "expected_value",
    "expected_utility",
    "policy_score",
    "rank_score",
    "score",
    "base_score",
    "meta_score",
    "__score__",
]

COMPONENT_COLS = [
    "trend_path_score",
    "shock_impulse_score",
    "execution_quality_score",
    "execution_risk_score",
    "oi_agreement_score",
    "location_quality_score",
    "pullback_retest_score",
    "compression_score",
    "volume_confirmation_score",
    "barrier_pressure_score",
]
ARCHETYPE_COLS = [
    "quiet_continuation_score",
    "loud_breakout_impulse_score",
    "dirty_shock_avoid_score",
    "retest_reversal_score",
    "compression_release_score",
    "base_positive_source_score",
    "prior_recent_source_strength",
    "run_entry_score",
    "late_run_continuation_score",
    "not_dirty_shock_score",
    "loud_clean_source_score",
    "barrier_relief_score",
    "clean_execution_context_score",
    "calm_positive_source_score",
    "loud_clean_execution_score",
    "clean_run_entry_score",
    "compression_capture_candidate_score",
    "risk_adjusted_capture_candidate_score",
    "clean_economic_capture_candidate_score",
    "misleading_location_risk_score",
]
SOURCE_SCORE_EVAL_COLS = [
    "quiet_continuation_score",
    "loud_breakout_impulse_score",
    "dirty_shock_avoid_score",
    "clean_execution_context_score",
    "calm_positive_source_score",
    "loud_clean_execution_score",
    "clean_run_entry_score",
    "compression_capture_candidate_score",
    "risk_adjusted_capture_candidate_score",
    "clean_economic_capture_candidate_score",
    "misleading_location_risk_score",
    "retest_reversal_score",
    "compression_release_score",
    "base_positive_source_score",
    "run_entry_score",
    "late_run_continuation_score",
    "not_dirty_shock_score",
    "loud_clean_source_score",
]
SOURCE_SCORE_TARGET_COLS = COMPONENT_COLS + SOURCE_SCORE_EVAL_COLS
TAG_COLS = [
    "tag_quiet_continuation",
    "tag_loud_breakout_impulse",
    "tag_dirty_shock_avoid",
    "tag_clean_execution_context",
    "tag_calm_positive_source",
    "tag_loud_clean_execution",
    "tag_clean_run_entry",
    "tag_compression_capture_candidate",
    "tag_risk_adjusted_capture_candidate",
    "tag_clean_economic_capture_candidate",
    "tag_misleading_location_risk",
    "tag_retest_reversal",
    "tag_compression_release",
    "tag_run_entry",
    "tag_late_run_continuation",
    "tag_ambiguous_none",
]
TAG_TO_SCORE = {
    "tag_quiet_continuation": "quiet_continuation_score",
    "tag_loud_breakout_impulse": "loud_breakout_impulse_score",
    "tag_dirty_shock_avoid": "dirty_shock_avoid_score",
    "tag_clean_execution_context": "clean_execution_context_score",
    "tag_calm_positive_source": "calm_positive_source_score",
    "tag_loud_clean_execution": "loud_clean_execution_score",
    "tag_clean_run_entry": "clean_run_entry_score",
    "tag_compression_capture_candidate": "compression_capture_candidate_score",
    "tag_risk_adjusted_capture_candidate": "risk_adjusted_capture_candidate_score",
    "tag_clean_economic_capture_candidate": "clean_economic_capture_candidate_score",
    "tag_misleading_location_risk": "misleading_location_risk_score",
    "tag_retest_reversal": "retest_reversal_score",
    "tag_compression_release": "compression_release_score",
    "tag_run_entry": "run_entry_score",
    "tag_late_run_continuation": "late_run_continuation_score",
}
QUALITY_LABEL_VARIANTS = [
    "quality_label_v0",
    "quality_label_source_rank_v1",
    "quality_label_source_wf_v1",
    "quality_label_clean_path_v2",
    "quality_label_recoverable_opportunity_v2",
    "quality_label_opportunity_capture_v3",
    "quality_label_economic_capture_v4",
]
FAILURE_MODE_COLS = [
    "outcome_positive_utility_flag",
    "outcome_clean_win_flag",
    "outcome_dirty_win_flag",
    "outcome_path_failure_flag",
    "outcome_timeout_failure_flag",
    "outcome_recoverable_opportunity_flag",
    "outcome_missed_opportunity_flag",
    "outcome_reversal_trap_flag",
    "outcome_no_edge_flag",
    "outcome_high_recovery_ratio_flag",
    "outcome_opportunity_captured_flag",
    "outcome_opportunity_capture_loss_flag",
    "outcome_economic_capture_flag",
    "outcome_expensive_capture_flag",
    "outcome_economic_capture_loss_flag",
]
SOURCE_SCORE_TARGETS = [
    "realized_net_utility",
    "outcome_recoverable_opportunity_flag",
    "outcome_opportunity_captured_flag",
    "outcome_economic_capture_flag",
    "outcome_opportunity_capture_loss_flag",
    "outcome_economic_capture_loss_flag",
    "opportunity_capture_efficiency",
    "outcome_clean_win_flag",
    "outcome_path_failure_flag",
    "outcome_no_edge_flag",
]
PRIMARY_PRIORITY = [
    ("tag_dirty_shock_avoid", "dirty_shock_avoid"),
    ("tag_misleading_location_risk", "misleading_location_risk"),
    ("tag_risk_adjusted_capture_candidate", "risk_adjusted_capture_candidate"),
    ("tag_compression_capture_candidate", "compression_capture_candidate"),
    ("tag_clean_run_entry", "clean_run_entry"),
    ("tag_run_entry", "run_entry"),
    ("tag_late_run_continuation", "late_run_continuation"),
    ("tag_calm_positive_source", "calm_positive_source"),
    ("tag_quiet_continuation", "quiet_continuation"),
    ("tag_compression_release", "compression_release"),
    ("tag_retest_reversal", "retest_reversal"),
    ("tag_loud_clean_execution", "loud_clean_execution"),
    ("tag_loud_breakout_impulse", "loud_breakout_impulse"),
    ("tag_ambiguous_none", "ambiguous_none"),
]


def clip01(x: Any) -> pd.Series:
    return pd.to_numeric(x, errors="coerce").clip(0.0, 1.0)


def _safe_numeric(values: Any) -> pd.Series:
    if isinstance(values, pd.Series):
        return pd.to_numeric(values, errors="coerce")
    return pd.to_numeric(pd.Series(values), errors="coerce")


def _safe_mean(values: Any) -> float:
    ser = _safe_numeric(values).dropna()
    return float(ser.mean()) if len(ser) else float("nan")


def _safe_quantile(values: Any, q: float) -> float:
    ser = _safe_numeric(values).dropna()
    return float(ser.quantile(q)) if len(ser) else float("nan")


def _spearman(x: Any, y: Any) -> float:
    xs = _safe_numeric(x)
    ys = _safe_numeric(y)
    mask = xs.notna() & ys.notna()
    if int(mask.sum()) < 5:
        return float("nan")
    xr = xs.loc[mask].rank(method="average")
    yr = ys.loc[mask].rank(method="average")
    if xr.nunique(dropna=True) < 2 or yr.nunique(dropna=True) < 2:
        return float("nan")
    return float(xr.corr(yr))


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(v) for v in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        out = float(value)
        return out if math.isfinite(out) else None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, pd.Series):
        return {"series_name": str(value.name), "series_len": int(len(value))}
    if isinstance(value, pd.DataFrame):
        return {"dataframe_shape": [int(value.shape[0]), int(value.shape[1])]}
    if pd.isna(value) if not isinstance(value, (list, dict, tuple, np.ndarray)) else False:
        return None
    return value


def load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing config: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return data


def _parse_csv_list(value: str | list[str] | tuple[str, ...] | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        raw = value
    else:
        raw = str(value).split(",")
    return [str(item).strip() for item in raw if str(item).strip()]


def _configured_proxy_score_columns(config: dict[str, Any], extra: list[str] | None = None) -> list[str]:
    return list(
        dict.fromkeys(
            [str(col) for col in (config.get("proxy_score_columns") or DEFAULT_PROXY_SCORE_CANDIDATES)]
            + [str(col) for col in (extra or [])]
        )
    )


def _configured_metadata_columns(config: dict[str, Any], frame: pd.DataFrame) -> list[str]:
    cols: list[str] = [SOURCE_ROW_IDX_COL, SOURCE_KEY_COL]
    for col in [
        config.get("candidate_id_col"),
        config.get("timestamp_col") or "__ts__",
        config.get("symbol_col") or "__symbol__",
        config.get("side_col"),
    ]:
        if col:
            cols.append(str(col))
    cols.extend(str(col) for col in config.get("regime_head_columns") or [])
    cols.extend(_configured_proxy_score_columns(config))
    return [col for col in dict.fromkeys(cols) if col in frame.columns]


def _outcome_column_names(config: dict[str, Any]) -> set[str]:
    return {
        str(col)
        for candidates in (config.get("outcome_columns") or {}).values()
        for col in candidates or []
    }


def is_forbidden_proxy_column(col: str, config: dict[str, Any]) -> bool:
    name = str(col)
    if name in _outcome_column_names(config):
        return True
    if name in {"utility", "net_utility", "realized_net_utility", "pnl_net", "target", "label"}:
        return True
    return bool(FORBIDDEN_PROXY_RE.search(name))


def _validate_proxy_score_columns(proxy_cols: list[str], config: dict[str, Any]) -> None:
    forbidden = [col for col in proxy_cols if is_forbidden_proxy_column(col, config)]
    if forbidden:
        raise ValueError(
            "Proxy/model score columns cannot be realized outcome-like columns: "
            + ", ".join(sorted(set(forbidden)))
        )


def add_source_identity(frame: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    out = frame.copy()
    timestamp_col = str(config.get("timestamp_col") or "__ts__")
    symbol_col = str(config.get("symbol_col") or "__symbol__")
    side_col = config.get("side_col")
    out[SOURCE_ROW_IDX_COL] = np.arange(len(out), dtype=np.int64)
    key_cols = [timestamp_col, symbol_col]
    if side_col and side_col in out.columns:
        key_cols.append(str(side_col))
    key_frame = out[key_cols].copy()
    key_frame[SOURCE_ROW_IDX_COL] = out[SOURCE_ROW_IDX_COL]
    key_frame = key_frame.astype(str).fillna("")
    hashes = pd.util.hash_pandas_object(key_frame, index=False).astype("uint64")
    out[SOURCE_KEY_COL] = hashes.map(lambda value: f"{int(value):016x}")
    return out


def _flatten_config_columns(config: dict[str, Any]) -> set[str]:
    cols: set[str] = set()
    for group_cols in (config.get("allowed_causal_feature_groups") or {}).values():
        cols.update(str(col) for col in group_cols or [])
    cols.update(str(col) for col in config.get("regime_head_columns") or [])
    cols.update(_configured_proxy_score_columns(config))
    for candidates in (config.get("outcome_columns") or {}).values():
        cols.update(str(col) for col in candidates or [])
    for key in ["timestamp_col", "symbol_col", "candidate_id_col", "side_col"]:
        value = config.get(key)
        if value:
            cols.add(str(value))
    return cols


def _schema_names(path: Path) -> set[str]:
    try:
        import pyarrow.parquet as pq

        return set(str(v) for v in pq.read_schema(path).names)
    except Exception:
        try:
            return set(str(v) for v in pd.read_parquet(path).columns)
        except Exception:
            return set()


def _normalize_timestamps(frame: pd.DataFrame, timestamp_col: str) -> pd.DataFrame:
    if timestamp_col not in frame.columns:
        index_name = frame.index.name or "index"
        maybe = frame.reset_index()
        if index_name in maybe.columns and pd.api.types.is_datetime64_any_dtype(maybe[index_name]):
            frame = maybe.rename(columns={index_name: timestamp_col})
        elif "index" in maybe.columns and pd.api.types.is_datetime64_any_dtype(maybe["index"]):
            frame = maybe.rename(columns={"index": timestamp_col})
        else:
            raise ValueError(f"Missing timestamp column {timestamp_col!r}")
    frame[timestamp_col] = pd.to_datetime(frame[timestamp_col], utc=True, errors="coerce").dt.tz_convert(None)
    return frame


def _derive_symbol_from_path(path: Path) -> str:
    name = path.name
    if name.startswith("symbol="):
        raw = name[len("symbol=") :].replace(".parquet", "")
        if "_" in raw and "/" not in raw:
            left, _, right = raw.partition("_")
            return f"{left}/{right}"
        return raw
    return path.stem


def load_frame(path: Path, config: dict[str, Any], *, columns: set[str] | None = None) -> pd.DataFrame:
    timestamp_col = str(config.get("timestamp_col") or "__ts__")
    symbol_col = str(config.get("symbol_col") or "__symbol__")
    if path.is_dir():
        files = sorted(path.glob("*.parquet"))
        if not files:
            files = sorted(path.glob("**/*.parquet"))
        if not files:
            raise FileNotFoundError(f"No parquet files found under {path}")
        frames: list[pd.DataFrame] = []
        requested = set(columns or _flatten_config_columns(config))
        requested.add(symbol_col)
        requested.add(timestamp_col)
        for file in files:
            names = _schema_names(file)
            read_cols = [col for col in requested if col in names]
            try:
                part = pd.read_parquet(file, columns=read_cols or None)
            except Exception:
                part = pd.read_parquet(file)
                if read_cols:
                    part = part[[col for col in read_cols if col in part.columns]].copy()
            part = _normalize_timestamps(part, timestamp_col)
            if symbol_col not in part.columns:
                part[symbol_col] = _derive_symbol_from_path(file)
            frames.append(part)
        out = pd.concat(frames, ignore_index=True)
    elif path.suffix.lower() in {".parquet", ".pq"}:
        read_cols = None
        if columns:
            names = _schema_names(path)
            read_cols = [col for col in columns if col in names]
        out = pd.read_parquet(path, columns=read_cols)
        out = _normalize_timestamps(out, timestamp_col)
    elif path.suffix.lower() in {".csv", ".gz"}:
        out = pd.read_csv(path)
        out = _normalize_timestamps(out, timestamp_col)
    else:
        raise ValueError(f"Unsupported input path: {path}")
    if symbol_col not in out.columns:
        raise ValueError(f"Missing symbol column {symbol_col!r}")
    out[symbol_col] = out[symbol_col].astype(str)
    return out.sort_values([timestamp_col, symbol_col], kind="mergesort").reset_index(drop=True)


def load_generic_table(path: Path) -> pd.DataFrame:
    if path.is_dir():
        files = sorted(path.glob("*.parquet")) or sorted(path.glob("**/*.parquet"))
        if not files:
            files = sorted(path.glob("*.csv")) or sorted(path.glob("**/*.csv"))
        if not files:
            raise FileNotFoundError(f"No parquet/csv files found under {path}")
        parts = [load_generic_table(file) for file in files]
        return pd.concat(parts, ignore_index=True)
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix in {".csv", ".gz"}:
        return pd.read_csv(path)
    raise ValueError(f"Unsupported table path: {path}")


def _duplicate_key_count(frame: pd.DataFrame, key_cols: list[str]) -> int:
    if not key_cols or not set(key_cols).issubset(frame.columns):
        return 0
    return int(frame.duplicated(key_cols, keep=False).sum())


def _matched_source_key_count(
    source_keys: pd.DataFrame,
    other: pd.DataFrame,
    key_cols: list[str],
) -> int:
    if not key_cols or not set(key_cols).issubset(other.columns):
        return 0
    unique_other = other[key_cols].drop_duplicates()
    matched = source_keys.merge(unique_other.assign(__matched__=True), on=key_cols, how="left")["__matched__"].eq(True)
    return int(matched.sum())


def join_labels(
    features: pd.DataFrame,
    labels_path: Path | None,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if labels_path is None:
        return features, {
            "labels_path": None,
            "label_rows": 0,
            "label_key_cols": [],
            "label_duplicate_keys": 0,
            "outcome_matched_source_rows": 0,
            "outcome_match_rate": 0.0,
            "rows_with_multiple_outcomes_joined": 0,
        }
    timestamp_col = str(config.get("timestamp_col") or "__ts__")
    symbol_col = str(config.get("symbol_col") or "__symbol__")
    labels = load_frame(labels_path, config)
    key_cols = [timestamp_col, symbol_col]
    source_keys = features[key_cols].copy()
    label_duplicate_keys = _duplicate_key_count(labels, key_cols)
    matched_rows = _matched_source_key_count(source_keys, labels, key_cols)
    label_counts = labels.groupby(key_cols, dropna=False).size().rename("__label_key_count__").reset_index()
    source_label_counts = source_keys.merge(label_counts, on=key_cols, how="left")["__label_key_count__"].fillna(0)
    rows_with_multiple_outcomes = int(source_label_counts.gt(1).sum())
    keep_cols = [timestamp_col, symbol_col]
    configured_outcome_cols = set()
    for candidates in (config.get("outcome_columns") or {}).values():
        configured_outcome_cols.update(str(col) for col in candidates or [])
    for col in labels.columns:
        if col in {timestamp_col, symbol_col}:
            continue
        if col in configured_outcome_cols or is_outcome_like_column(col, config) or col in set(config.get("regime_head_columns") or []):
            keep_cols.append(col)
    labels = labels.loc[:, list(dict.fromkeys([col for col in keep_cols if col in labels.columns]))].copy()
    duplicate_cols = [col for col in labels.columns if col in features.columns and col not in {timestamp_col, symbol_col}]
    labels = labels.drop(columns=duplicate_cols)
    merged = features.merge(labels, on=[timestamp_col, symbol_col], how="left", validate="many_to_one")
    return merged, {
        "labels_path": str(labels_path),
        "label_rows": int(len(labels)),
        "label_key_cols": key_cols,
        "label_duplicate_keys": int(label_duplicate_keys),
        "outcome_matched_source_rows": int(matched_rows),
        "outcome_match_rate": float(matched_rows / len(features)) if len(features) else 0.0,
        "rows_with_multiple_outcomes_joined": rows_with_multiple_outcomes,
    }


def join_predictions(
    features: pd.DataFrame,
    predictions_path: Path | None,
    config: dict[str, Any],
    *,
    prediction_key_cols: list[str] | None = None,
    proxy_score_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    configured_proxy_cols = _configured_proxy_score_columns(config, proxy_score_cols)
    available_feature_proxy_cols = [col for col in configured_proxy_cols if col in features.columns]
    if predictions_path is None:
        _validate_proxy_score_columns(available_feature_proxy_cols, config)
        return features, {
            "predictions_path": None,
            "prediction_rows": 0,
            "prediction_key_cols": [],
            "proxy_score_columns": available_feature_proxy_cols,
            "prediction_duplicate_keys": 0,
            "prediction_matched_source_rows": int(features[available_feature_proxy_cols].notna().any(axis=1).sum())
            if available_feature_proxy_cols
            else 0,
            "prediction_match_rate": float(features[available_feature_proxy_cols].notna().any(axis=1).mean())
            if available_feature_proxy_cols and len(features)
            else 0.0,
            "rows_with_missing_proxy_score": int((~features[available_feature_proxy_cols].notna().any(axis=1)).sum())
            if available_feature_proxy_cols
            else int(len(features)),
            "rows_with_multiple_predictions_joined": 0,
        }

    predictions = load_generic_table(predictions_path)
    timestamp_col = str(config.get("timestamp_col") or "__ts__")
    symbol_col = str(config.get("symbol_col") or "__symbol__")
    side_col = config.get("side_col")
    key_cols = prediction_key_cols or []
    if not key_cols:
        if config.get("candidate_id_col") and str(config["candidate_id_col"]) in features.columns and str(config["candidate_id_col"]) in predictions.columns:
            key_cols = [str(config["candidate_id_col"])]
        elif SOURCE_KEY_COL in features.columns and SOURCE_KEY_COL in predictions.columns:
            key_cols = [SOURCE_KEY_COL]
        else:
            key_cols = [timestamp_col, symbol_col]
            if side_col and str(side_col) in features.columns and str(side_col) in predictions.columns:
                key_cols.append(str(side_col))
    rename_keys: dict[str, str] = {}
    if timestamp_col in key_cols and timestamp_col not in predictions.columns and "timestamp" in predictions.columns:
        rename_keys["timestamp"] = timestamp_col
    if symbol_col in key_cols and symbol_col not in predictions.columns and "symbol" in predictions.columns:
        rename_keys["symbol"] = symbol_col
    if rename_keys:
        predictions = predictions.rename(columns=rename_keys)
    missing_feature_keys = [col for col in key_cols if col not in features.columns]
    missing_prediction_keys = [col for col in key_cols if col not in predictions.columns]
    if missing_feature_keys or missing_prediction_keys:
        raise ValueError(
            "Prediction join key mismatch; missing in features="
            f"{missing_feature_keys}, missing in predictions={missing_prediction_keys}"
        )
    if timestamp_col in key_cols:
        predictions = _normalize_timestamps(predictions, timestamp_col)
    if symbol_col in key_cols:
        predictions[symbol_col] = predictions[symbol_col].astype(str)

    candidate_proxy_cols = proxy_score_cols or configured_proxy_cols
    proxy_cols = [col for col in dict.fromkeys(candidate_proxy_cols) if col in predictions.columns or col in features.columns]
    _validate_proxy_score_columns(proxy_cols, config)
    if not proxy_cols:
        raise ValueError("No proxy/model score columns were found in features or predictions.")

    prediction_duplicate_keys = _duplicate_key_count(predictions, key_cols)
    prediction_counts = predictions.groupby(key_cols, dropna=False).size().rename("__prediction_key_count__").reset_index()
    source_prediction_counts = features[key_cols].merge(prediction_counts, on=key_cols, how="left")["__prediction_key_count__"].fillna(0)
    rows_with_multiple_predictions = int(source_prediction_counts.gt(1).sum())
    deduped_predictions = predictions.drop_duplicates(key_cols, keep="last")
    source_keys = features[key_cols].copy()
    matched_rows = _matched_source_key_count(source_keys, deduped_predictions, key_cols)
    keep_cols = list(dict.fromkeys(key_cols + [col for col in proxy_cols if col in deduped_predictions.columns]))
    merged = features.merge(
        deduped_predictions[keep_cols],
        on=key_cols,
        how="left",
        suffixes=("", "__pred"),
        validate="many_to_one",
    )
    for col in proxy_cols:
        pred_col = f"{col}__pred"
        if pred_col in merged.columns:
            if col in features.columns:
                merged[col] = merged[col].combine_first(merged[pred_col])
            else:
                merged[col] = merged[pred_col]
            merged = merged.drop(columns=[pred_col])
    effective_proxy_cols = [col for col in proxy_cols if col in merged.columns]
    has_proxy = merged[effective_proxy_cols].notna().any(axis=1) if effective_proxy_cols else pd.Series(False, index=merged.index)
    min_match_rate = float((config.get("diagnostics") or {}).get("prediction_min_match_rate", 0.80))
    match_rate = float(matched_rows / len(features)) if len(features) else 0.0
    status = "pass"
    if rows_with_multiple_predictions > 0 or prediction_duplicate_keys > 0:
        status = "fail"
    elif match_rate < min_match_rate:
        status = "warning"
    return merged, {
        "predictions_path": str(predictions_path),
        "prediction_rows": int(len(predictions)),
        "prediction_key_cols": key_cols,
        "proxy_score_columns": effective_proxy_cols,
        "prediction_duplicate_keys": int(prediction_duplicate_keys),
        "prediction_matched_source_rows": int(matched_rows),
        "prediction_match_rate": match_rate,
        "prediction_min_match_rate": min_match_rate,
        "rows_with_missing_proxy_score": int((~has_proxy).sum()),
        "rows_with_multiple_predictions_joined": rows_with_multiple_predictions,
        "alignment_status": status,
    }


def is_outcome_like_column(col: str, config: dict[str, Any]) -> bool:
    whitelist = set(str(v) for v in config.get("explicit_causal_whitelist") or [])
    if col in whitelist:
        return False
    return bool(OUTCOME_LIKE_RE.search(str(col)))


def build_feature_registry(frame: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    whitelist = set(str(v) for v in config.get("explicit_causal_whitelist") or [])
    groups = config.get("allowed_causal_feature_groups") or {}
    available: dict[str, list[str]] = {}
    missing: dict[str, list[str]] = {}
    excluded: list[str] = []
    portability_warnings: list[str] = []
    for group, configured in groups.items():
        group_available: list[str] = []
        group_missing: list[str] = []
        for raw_col in configured or []:
            col = str(raw_col)
            if col not in frame.columns:
                group_missing.append(col)
                continue
            if is_outcome_like_column(col, config) and col not in whitelist:
                excluded.append(col)
                continue
            if RAW_PORTABILITY_RE.search(col):
                portability_warnings.append(col)
                continue
            if _safe_numeric(frame[col]).notna().sum() < 2:
                group_missing.append(col)
                continue
            group_available.append(col)
        available[str(group)] = list(dict.fromkeys(group_available))
        missing[str(group)] = list(dict.fromkeys(group_missing))
    source_cols = sorted({col for cols in available.values() for col in cols})
    return {
        "available": available,
        "missing": missing,
        "excluded_outcome_like": sorted(set(excluded)),
        "portability_warnings": sorted(set(portability_warnings)),
        "source_columns": source_cols,
    }


def safe_pct_rank_by_timestamp(df: pd.DataFrame, col: str, timestamp_col: str) -> pd.Series:
    values = _safe_numeric(df[col])
    ranks = values.groupby(df[timestamp_col], dropna=False).rank(method="average", pct=True)
    counts = values.groupby(df[timestamp_col], dropna=False).transform("count")
    ranks = ranks.where(counts >= 2, 0.5)
    return ranks.clip(0.0, 1.0).astype(np.float32)


def safe_rolling_rank_by_symbol(
    df: pd.DataFrame,
    col: str,
    symbol_col: str,
    timestamp_col: str,
    lookback: int,
) -> pd.Series:
    out = pd.Series(np.nan, index=df.index, dtype=np.float32)
    work = df[[symbol_col, timestamp_col, col]].copy()
    work["__pos__"] = np.arange(len(work), dtype=np.int64)
    work = work.sort_values([symbol_col, timestamp_col], kind="mergesort")
    for _, group in work.groupby(symbol_col, sort=False):
        vals = _safe_numeric(group[col]).to_numpy(dtype=np.float64, copy=False)
        pos = group["__pos__"].to_numpy(dtype=np.int64, copy=False)
        ranks = np.full(len(group), np.nan, dtype=np.float32)
        for i in range(len(group)):
            left = max(0, i - int(lookback) + 1)
            window = vals[left : i + 1]
            finite = window[np.isfinite(window)]
            if len(finite) >= 2 and np.isfinite(vals[i]):
                ranks[i] = float((np.sum(finite <= vals[i])) / len(finite))
            elif np.isfinite(vals[i]):
                ranks[i] = 0.5
        out.iloc[pos] = ranks
    return out.clip(0.0, 1.0)


def weighted_mean_available(df: pd.DataFrame, columns: list[str], weights: list[float] | None = None) -> pd.Series:
    if not columns:
        return pd.Series(np.nan, index=df.index, dtype=np.float32)
    valid_cols = [col for col in columns if col in df.columns]
    if not valid_cols:
        return pd.Series(np.nan, index=df.index, dtype=np.float32)
    mat = df[valid_cols].apply(pd.to_numeric, errors="coerce").astype(np.float64)
    if weights is None:
        w = np.ones(len(valid_cols), dtype=np.float64)
    else:
        raw_weights = dict(zip(columns, weights))
        w = np.asarray([float(raw_weights.get(col, 0.0)) for col in valid_cols], dtype=np.float64)
    mask = mat.notna().to_numpy(dtype=bool)
    arr = mat.fillna(0.0).to_numpy(dtype=np.float64)
    denom = mask @ w
    numer = arr @ w
    values = np.divide(numer, denom, out=np.full(len(mat), np.nan, dtype=np.float64), where=denom > 0.0)
    return pd.Series(values, index=df.index, dtype=np.float32)


def _component_from_features(
    frame: pd.DataFrame,
    features: list[str],
    *,
    timestamp_col: str,
    lower_is_better: set[str],
    prefix: str,
    normalized: dict[str, pd.Series],
) -> pd.Series:
    norm_cols: list[str] = []
    temp = pd.DataFrame(index=frame.index)
    for feature in features:
        if feature not in normalized:
            rank = safe_pct_rank_by_timestamp(frame, feature, timestamp_col)
            if feature in lower_is_better:
                rank = 1.0 - rank
            normalized[feature] = rank.fillna(0.5).clip(0.0, 1.0).astype(np.float32)
        name = f"{prefix}__{feature}"
        temp[name] = normalized[feature]
        norm_cols.append(name)
    score = weighted_mean_available(temp, norm_cols).fillna(0.5)
    return score.clip(0.0, 1.0).astype(np.float32)


def build_component_scores(
    frame: pd.DataFrame,
    registry: dict[str, Any],
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    timestamp_col = str(config.get("timestamp_col") or "__ts__")
    groups = registry["available"]
    lower_is_better = set(str(v) for v in config.get("lower_is_better_features") or [])
    high_risk = [feature for feature in config.get("high_risk_features") or [] if feature in frame.columns]
    normalized: dict[str, pd.Series] = {}

    def component(group_name: str) -> pd.Series:
        return _component_from_features(
            frame,
            groups.get(group_name, []),
            timestamp_col=timestamp_col,
            lower_is_better=lower_is_better,
            prefix=group_name,
            normalized=normalized,
        )

    trend = component("trend_path_features")
    shock = component("shock_impulse_features")
    execution_quality = component("liquidity_execution_features")
    oi = component("oi_agreement_features")
    location = component("location_features")
    pullback = component("pullback_retest_features")
    compression = component("compression_features")
    volume = component("volume_confirmation_features")
    barrier = component("barrier_pressure_features")
    high_risk_score = _component_from_features(
        frame,
        high_risk,
        timestamp_col=timestamp_col,
        lower_is_better=set(),
        prefix="high_risk_features",
        normalized=normalized,
    )
    execution_risk = weighted_mean_available(
        pd.DataFrame(
            {
                "inverse_execution_quality": (1.0 - execution_quality).clip(0.0, 1.0),
                "high_risk_score": high_risk_score,
                "barrier_pressure_score": barrier,
            },
            index=frame.index,
        ),
        ["inverse_execution_quality", "high_risk_score", "barrier_pressure_score"],
        [0.55, 0.25, 0.20],
    ).fillna(0.5)

    components = pd.DataFrame(
        {
            "trend_path_score": trend,
            "shock_impulse_score": shock,
            "execution_quality_score": execution_quality,
            "execution_risk_score": execution_risk.clip(0.0, 1.0),
            "oi_agreement_score": oi,
            "location_quality_score": location,
            "pullback_retest_score": pullback,
            "compression_score": compression,
            "volume_confirmation_score": volume,
            "barrier_pressure_score": barrier,
        },
        index=frame.index,
    ).astype(np.float32)
    return components, {
        "normalized_feature_count": int(len(normalized)),
        "component_neutral_counts": {
            col: int(components[col].eq(0.5).sum()) for col in components.columns
        },
    }


def _subcomponent(
    frame: pd.DataFrame,
    registry: dict[str, Any],
    config: dict[str, Any],
    names: list[str],
    *,
    lower_is_better: bool = False,
) -> pd.Series:
    timestamp_col = str(config.get("timestamp_col") or "__ts__")
    present = [name for name in names if name in frame.columns and not is_outcome_like_column(name, config)]
    lower = set(present) if lower_is_better else set()
    return _component_from_features(
        frame,
        present,
        timestamp_col=timestamp_col,
        lower_is_better=lower,
        prefix="subcomponent",
        normalized={},
    )


def prior_recent_source_strength(
    frame: pd.DataFrame,
    score: pd.Series,
    *,
    symbol_col: str,
    timestamp_col: str,
    hours: float,
    rows: int,
) -> pd.Series:
    out = pd.Series(0.0, index=frame.index, dtype=np.float32)
    work = pd.DataFrame(
        {
            symbol_col: frame[symbol_col].astype(str),
            timestamp_col: pd.to_datetime(frame[timestamp_col], errors="coerce"),
            "__score__": _safe_numeric(score).fillna(0.0),
            "__pos__": np.arange(len(frame), dtype=np.int64),
        },
        index=frame.index,
    ).sort_values([symbol_col, timestamp_col], kind="mergesort")
    window = pd.Timedelta(hours=float(hours))
    for _, group in work.groupby(symbol_col, sort=False):
        timestamps = pd.to_datetime(group[timestamp_col]).to_numpy()
        scores = group["__score__"].to_numpy(dtype=np.float64, copy=False)
        positions = group["__pos__"].to_numpy(dtype=np.int64, copy=False)
        start = 0
        for i in range(len(group)):
            ts = pd.Timestamp(timestamps[i])
            while start < i and ts - pd.Timestamp(timestamps[start]) > window:
                start += 1
            row_start = max(start, i - int(rows))
            if row_start < i:
                out.iloc[positions[i]] = float(np.nanmax(scores[row_start:i]))
            else:
                out.iloc[positions[i]] = 0.0
    return out.clip(0.0, 1.0).astype(np.float32)


def build_archetype_scores(
    frame: pd.DataFrame,
    components: pd.DataFrame,
    registry: dict[str, Any],
    config: dict[str, Any],
) -> pd.DataFrame:
    timestamp_col = str(config.get("timestamp_col") or "__ts__")
    symbol_col = str(config.get("symbol_col") or "__symbol__")
    thresholds = config.get("tag_thresholds") or {}
    breakout_range = _subcomponent(
        frame,
        registry,
        config,
        ["breakout_24h", "breakout_confirmed", "breakout_soft", "pct_breakout_t", "range_24h_pct", "range_12h_pct", "range_pct"],
    )
    speed_jump = _subcomponent(
        frame,
        registry,
        config,
        ["speed", "jump_intensity", "shock_12h", "shock_vol_ratio", "second_leg_accel_1h", "second_leg_accel_2h", "impulse_ratio_24"],
    )
    rejection_risk = _subcomponent(
        frame,
        registry,
        config,
        ["rejection_proxy", "tail_fail", "trap_strength", "mr_failure"],
    )
    overextension_risk = _subcomponent(
        frame,
        registry,
        config,
        ["trend_overextension_z", "pct_extreme", "dist_rolling_7d_high", "dist_ema20_atr"],
    )

    quiet = (
        0.30 * components["trend_path_score"]
        + 0.20 * components["location_quality_score"]
        + 0.20 * components["oi_agreement_score"]
        + 0.15 * components["execution_quality_score"]
        + 0.15 * components["volume_confirmation_score"]
        - 0.30 * components["shock_impulse_score"]
        - 0.15 * components["barrier_pressure_score"]
    )
    loud = (
        0.30 * components["shock_impulse_score"]
        + 0.25 * breakout_range
        + 0.20 * speed_jump
        + 0.15 * components["volume_confirmation_score"]
        + 0.10 * components["trend_path_score"]
    )
    dirty = (
        0.35 * loud
        + 0.25 * components["execution_risk_score"]
        + 0.20 * components["barrier_pressure_score"]
        + 0.10 * rejection_risk
        + 0.10 * overextension_risk
    )
    retest = (
        0.35 * components["pullback_retest_score"]
        + 0.25 * components["location_quality_score"]
        + 0.15 * components["trend_path_score"]
        + 0.15 * components["execution_quality_score"]
        + 0.10 * components["oi_agreement_score"]
        - 0.20 * components["shock_impulse_score"]
    )
    compression_release = (
        0.35 * components["compression_score"]
        + 0.25 * components["shock_impulse_score"]
        + 0.20 * components["volume_confirmation_score"]
        + 0.10 * components["trend_path_score"]
        + 0.10 * components["execution_quality_score"]
    )
    base = pd.concat(
        [
            clip01(quiet).rename("quiet"),
            (clip01(loud) * (1.0 - clip01(dirty))).rename("loud_clean"),
            clip01(retest).rename("retest"),
            clip01(compression_release).rename("compression"),
        ],
        axis=1,
    ).max(axis=1)
    prior = prior_recent_source_strength(
        frame,
        base,
        symbol_col=symbol_col,
        timestamp_col=timestamp_col,
        hours=float(thresholds.get("run_prior_hours", 3.0)),
        rows=int(thresholds.get("run_prior_rows", 3)),
    )
    run_entry = base * (1.0 - prior)
    late = base * prior
    not_dirty = (1.0 - clip01(dirty)).clip(0.0, 1.0)
    barrier_relief = (1.0 - components["barrier_pressure_score"]).clip(0.0, 1.0)
    clean_execution = weighted_mean_available(
        pd.DataFrame(
            {
                "execution_quality": components["execution_quality_score"],
                "inverse_execution_risk": (1.0 - components["execution_risk_score"]).clip(0.0, 1.0),
                "barrier_relief": barrier_relief,
            },
            index=frame.index,
        ),
        ["execution_quality", "inverse_execution_risk", "barrier_relief"],
        [0.45, 0.25, 0.30],
    ).fillna(0.5)
    calm_positive = base * clean_execution * not_dirty * (1.0 - 0.50 * components["shock_impulse_score"].clip(0.0, 1.0))
    loud_clean_execution = clip01(loud) * clean_execution * not_dirty
    clean_run_entry = run_entry * clean_execution * not_dirty
    compression_capture = weighted_mean_available(
        pd.DataFrame(
            {
                "compression": components["compression_score"],
                "late_run": late,
                "compression_release": clip01(compression_release),
                "shock_impulse": components["shock_impulse_score"],
                "barrier_pressure": components["barrier_pressure_score"],
                "oi_agreement": components["oi_agreement_score"],
            },
            index=frame.index,
        ),
        ["compression", "late_run", "compression_release", "shock_impulse", "barrier_pressure", "oi_agreement"],
        [0.30, 0.25, 0.20, 0.10, 0.10, 0.05],
    ).fillna(0.5)
    misleading_location_risk = weighted_mean_available(
        pd.DataFrame(
            {
                "location": components["location_quality_score"],
                "retest": clip01(retest),
                "quiet": clip01(quiet),
                "trend": components["trend_path_score"],
                "inverse_compression": (1.0 - components["compression_score"]).clip(0.0, 1.0),
                "low_shock": (1.0 - components["shock_impulse_score"]).clip(0.0, 1.0),
            },
            index=frame.index,
        ),
        ["location", "retest", "quiet", "trend", "inverse_compression", "low_shock"],
        [0.30, 0.25, 0.15, 0.10, 0.10, 0.10],
    ).fillna(0.5)
    risk_adjusted_capture = weighted_mean_available(
        pd.DataFrame(
            {
                "compression_capture": compression_capture,
                "inverse_misleading_location": (1.0 - clip01(misleading_location_risk)).clip(0.0, 1.0),
                "barrier_relief": barrier_relief,
                "execution_quality": components["execution_quality_score"],
                "late_run": late,
            },
            index=frame.index,
        ),
        ["compression_capture", "inverse_misleading_location", "barrier_relief", "execution_quality", "late_run"],
        [0.45, 0.25, 0.10, 0.10, 0.10],
    ).fillna(0.5)
    clean_economic_capture = weighted_mean_available(
        pd.DataFrame(
            {
                "risk_adjusted_capture": risk_adjusted_capture,
                "compression_capture": compression_capture,
                "clean_execution": clean_execution,
                "not_dirty": not_dirty,
                "inverse_misleading_location": (1.0 - clip01(misleading_location_risk)).clip(0.0, 1.0),
                "barrier_relief": barrier_relief,
                "not_path_pressure": (1.0 - components["barrier_pressure_score"].clip(0.0, 1.0)).clip(0.0, 1.0),
            },
            index=frame.index,
        ),
        [
            "risk_adjusted_capture",
            "compression_capture",
            "clean_execution",
            "not_dirty",
            "inverse_misleading_location",
            "barrier_relief",
            "not_path_pressure",
        ],
        [0.30, 0.20, 0.18, 0.12, 0.10, 0.05, 0.05],
    ).fillna(0.5)
    return pd.DataFrame(
        {
            "quiet_continuation_score": clip01(quiet),
            "loud_breakout_impulse_score": clip01(loud),
            "dirty_shock_avoid_score": clip01(dirty),
            "retest_reversal_score": clip01(retest),
            "compression_release_score": clip01(compression_release),
            "base_positive_source_score": clip01(base),
            "prior_recent_source_strength": clip01(prior),
            "run_entry_score": clip01(run_entry),
            "late_run_continuation_score": clip01(late),
            "barrier_relief_score": clip01(barrier_relief),
            "clean_execution_context_score": clip01(clean_execution),
            "calm_positive_source_score": clip01(calm_positive),
            "loud_clean_execution_score": clip01(loud_clean_execution),
            "clean_run_entry_score": clip01(clean_run_entry),
            "compression_capture_candidate_score": clip01(compression_capture),
            "risk_adjusted_capture_candidate_score": clip01(risk_adjusted_capture),
            "clean_economic_capture_candidate_score": clip01(clean_economic_capture),
            "misleading_location_risk_score": clip01(misleading_location_risk),
        },
        index=frame.index,
    ).astype(np.float32)


def bool_from_score_rank(
    score: pd.Series,
    timestamp_col: pd.Series,
    threshold: float,
    *,
    min_count: int = 5,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    score = _safe_numeric(score)
    ts = pd.to_datetime(timestamp_col, errors="coerce")
    ranks = score.groupby(ts, dropna=False).rank(method="average", pct=True)
    counts = score.groupby(ts, dropna=False).transform("count")
    use_xs = counts >= int(min_count)
    tag = (ranks >= (1.0 - float(threshold))) & use_xs & score.notna()
    fallback = (~use_xs) & score.notna()
    if bool(fallback.any()):
        order = np.argsort(ts.astype("int64").to_numpy(dtype=np.int64, copy=False), kind="mergesort")
        prior_scores: list[float] = []
        out = tag.to_numpy(dtype=bool, copy=True)
        unique_ts = pd.Series(ts.iloc[order]).drop_duplicates().tolist()
        for current_ts in unique_ts:
            positions = np.flatnonzero((ts == current_ts).to_numpy())
            small_positions = positions[fallback.iloc[positions].to_numpy(dtype=bool)]
            if len(small_positions) and prior_scores:
                cutoff = float(np.nanquantile(np.asarray(prior_scores, dtype=np.float64), 1.0 - float(threshold)))
                out[small_positions] = score.iloc[small_positions].to_numpy(dtype=np.float64) >= cutoff
            elif len(small_positions):
                out[small_positions] = False
            current_values = score.iloc[positions].dropna().tolist()
            prior_scores.extend(float(v) for v in current_values if math.isfinite(float(v)))
        tag = pd.Series(out, index=score.index)
    return tag.fillna(False).astype(bool), ranks.astype(np.float32), fallback.fillna(False).astype(bool)


def build_source_tags(
    frame: pd.DataFrame,
    scores: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    timestamp_col = str(config.get("timestamp_col") or "__ts__")
    thresholds = config.get("tag_thresholds") or {}
    min_count = int(thresholds.get("min_timestamp_rows", 5))
    out = pd.DataFrame(index=frame.index)
    fallback_counts: dict[str, int] = {}
    for tag_col, score_col in TAG_TO_SCORE.items():
        threshold_key = tag_col.replace("tag_", "")
        tag, rank, fallback = bool_from_score_rank(
            scores[score_col],
            frame[timestamp_col],
            float(thresholds.get(threshold_key, 0.20)),
            min_count=min_count,
        )
        out[tag_col] = tag
        fallback_counts[tag_col] = int(fallback.sum())

    prior = scores["prior_recent_source_strength"].fillna(0.0)
    out["tag_run_entry"] = out["tag_run_entry"] & (prior <= float(thresholds.get("run_prior_low_threshold", 0.35)))
    out["tag_late_run_continuation"] = out["tag_late_run_continuation"] & (prior >= float(thresholds.get("run_prior_high_threshold", 0.35)))
    positive = (
        out["tag_quiet_continuation"]
        | out["tag_loud_breakout_impulse"]
        | out["tag_calm_positive_source"]
        | out["tag_loud_clean_execution"]
        | out["tag_compression_capture_candidate"]
        | out["tag_risk_adjusted_capture_candidate"]
        | out["tag_retest_reversal"]
        | out["tag_compression_release"]
        | out["tag_run_entry"]
        | out["tag_clean_run_entry"]
        | out["tag_late_run_continuation"]
    )
    out["tag_ambiguous_none"] = (~positive) & (~out["tag_dirty_shock_avoid"])
    primary = pd.Series("ambiguous_none", index=frame.index, dtype=object)
    for tag_col, label in reversed(PRIMARY_PRIORITY):
        primary = primary.where(~out[tag_col], label)
    out["primary_source_tag"] = primary
    return out, {
        "fallback_counts": fallback_counts,
    }


def _reason_codes(row: pd.Series) -> str:
    reasons: list[str] = []
    if bool(row.get("tag_quiet_continuation", False)):
        reasons.append("high_trend_low_shock")
    if bool(row.get("tag_loud_breakout_impulse", False)):
        reasons.append("loud_impulse")
    if bool(row.get("tag_dirty_shock_avoid", False)):
        reasons.append("loud_plus_bad_spread")
    if bool(row.get("tag_clean_execution_context", False)):
        reasons.append("clean_execution_context")
    if bool(row.get("tag_calm_positive_source", False)):
        reasons.append("calm_positive_source")
    if bool(row.get("tag_loud_clean_execution", False)):
        reasons.append("loud_clean_execution")
    if bool(row.get("tag_compression_capture_candidate", False)):
        reasons.append("compression_late_capture_candidate")
    if bool(row.get("tag_risk_adjusted_capture_candidate", False)):
        reasons.append("risk_adjusted_capture_candidate")
    if bool(row.get("tag_clean_economic_capture_candidate", False)):
        reasons.append("clean_economic_capture_candidate")
    if bool(row.get("tag_misleading_location_risk", False)):
        reasons.append("misleading_location_path_risk")
    if float(row.get("location_quality_score", 0.0) or 0.0) >= 0.65:
        reasons.append("clean_location")
    if bool(row.get("tag_run_entry", False)):
        reasons.append("prior_symbol_run_absent")
    if bool(row.get("tag_clean_run_entry", False)):
        reasons.append("clean_prior_symbol_run_absent")
    if bool(row.get("tag_late_run_continuation", False)):
        reasons.append("prior_symbol_run_present")
    if bool(row.get("tag_compression_release", False)):
        reasons.append("compression_plus_volume")
    if bool(row.get("tag_retest_reversal", False)):
        reasons.append("retest_or_reversal")
    if bool(row.get("tag_ambiguous_none", False)):
        reasons.append("fallback_none")
    return ";".join(reasons)


def materialize_source_tags(frame: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    timestamp_col = str(config.get("timestamp_col") or "__ts__")
    symbol_col = str(config.get("symbol_col") or "__symbol__")
    frame = _normalize_timestamps(frame.copy(), timestamp_col)
    frame[symbol_col] = frame[symbol_col].astype(str)
    registry = build_feature_registry(frame, config)
    components, component_report = build_component_scores(frame, registry, config)
    archetypes = build_archetype_scores(frame, components, registry, config)
    archetypes["not_dirty_shock_score"] = (1.0 - archetypes["dirty_shock_avoid_score"]).clip(0.0, 1.0).astype(np.float32)
    archetypes["loud_clean_source_score"] = (
        archetypes["loud_breakout_impulse_score"] * archetypes["not_dirty_shock_score"]
    ).clip(0.0, 1.0).astype(np.float32)
    tags, tag_report = build_source_tags(frame, archetypes, config)

    id_cols = _configured_metadata_columns(config, frame)
    source = frame.loc[:, list(dict.fromkeys(id_cols))].copy()
    source = pd.concat([source, components, archetypes, tags], axis=1)
    source_cols = registry["source_columns"]
    source["missing_feature_count"] = frame[source_cols].isna().sum(axis=1).astype(np.int16) if source_cols else 0
    source["source_tag_reason_codes"] = source.apply(_reason_codes, axis=1)
    report = {
        "registry": registry,
        "components": component_report,
        "tags": tag_report,
        "source_columns_used": source_cols,
        "source_output_columns": list(source.columns),
    }
    return source, report


def _first_existing(frame: pd.DataFrame, candidates: list[str] | None) -> str | None:
    for col in candidates or []:
        if col in frame.columns:
            return str(col)
    return None


def _outcome_series(
    frame: pd.DataFrame,
    config: dict[str, Any],
    key: str,
) -> tuple[pd.Series, str | None]:
    col = _first_existing(frame, (config.get("outcome_columns") or {}).get(key))
    if col is None:
        return pd.Series(np.nan, index=frame.index, dtype=np.float32), None
    return _safe_numeric(frame[col]).astype(np.float32), col


def build_outcome_frame(frame: pd.DataFrame, source: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    timestamp_col = str(config.get("timestamp_col") or "__ts__")
    thresholds = config.get("label_thresholds") or {}
    utility, utility_col = _outcome_series(frame, config, "realized_net_utility")
    ret, ret_col = _outcome_series(frame, config, "realized_return")
    mae_raw, mae_col = _outcome_series(frame, config, "adverse_mae")
    mfe_raw, mfe_col = _outcome_series(frame, config, "favorable_mfe")
    barrier, barrier_col = _outcome_series(frame, config, "barrier_width")
    timeout, timeout_col = _outcome_series(frame, config, "timeout_flag")
    bad_flag, bad_flag_col = _outcome_series(frame, config, "bad_mae_flag")
    wide_flag, wide_flag_col = _outcome_series(frame, config, "wide_barrier_flag")
    fill_quality, fill_col = _outcome_series(frame, config, "execution_fill_quality")

    mae = mae_raw.copy()
    finite_mae = mae.dropna()
    if len(finite_mae) and float(finite_mae.median()) < 0.0:
        mae = (-mae).clip(lower=0.0)
    else:
        mae = mae.clip(lower=0.0)
    barrier_abs = barrier.abs().replace(0.0, np.nan)
    mae_norm = mae / barrier_abs if barrier_col else mae
    mfe = mfe_raw.clip(lower=0.0)
    mfe_norm = mfe / barrier_abs if barrier_col else mfe

    if bad_flag_col is None:
        threshold = float(thresholds.get("bad_mae_norm_threshold", 1.0))
        if mae_norm.notna().sum() < 5:
            threshold = _training_quantile(
                mae_norm,
                frame[timestamp_col],
                float(thresholds.get("fallback_bad_mae_quantile", 0.75)),
                float(thresholds.get("calibration_fraction", 0.50)),
            )
        bad_mae = mae_norm >= threshold
    else:
        bad_mae = bad_flag > 0.5
    if wide_flag_col is None:
        if barrier_col is not None:
            wide_threshold = float(thresholds.get("wide_barrier_abs_threshold", 0.025))
            wide_barrier = barrier_abs >= wide_threshold
        else:
            wide_barrier = pd.Series(False, index=frame.index)
    else:
        wide_barrier = wide_flag > 0.5

    outcome_available = utility.notna()
    bad_mae_bool = bad_mae.fillna(False)
    timeout_bool = (timeout > 0.5).fillna(False)
    wide_barrier_bool = wide_barrier.fillna(False)
    mae_norm_for_rules = mae_norm.where(mae_norm.notna(), np.inf)
    mfe_norm_for_rules = mfe_norm.where(mfe_norm.notna(), -np.inf)
    opportunity_mfe_threshold = float(thresholds.get("opportunity_mfe_norm_threshold", 1.0))
    opportunity_max_mae = float(thresholds.get("opportunity_max_mae_norm", 1.0))
    clean_path_max_mae = float(thresholds.get("clean_path_max_mae_norm", 0.75))
    recovery_ratio_threshold = float(thresholds.get("recovery_ratio_threshold", 1.25))
    recovery_ratio_cap = float(thresholds.get("recovery_ratio_cap", 20.0))
    capture_eff_threshold = float(thresholds.get("opportunity_capture_efficiency_threshold", 0.10))
    capture_loss_eff_threshold = float(thresholds.get("opportunity_capture_loss_efficiency_threshold", -0.05))
    capture_eff_cap = float(thresholds.get("opportunity_capture_efficiency_cap", 3.0))
    economic_capture_min_utility = float(thresholds.get("economic_capture_min_utility", 0.0))
    economic_capture_min_efficiency = float(thresholds.get("economic_capture_min_efficiency", capture_eff_threshold))
    economic_capture_max_mae = float(thresholds.get("economic_capture_max_mae_norm", clean_path_max_mae))
    recovery_ratio = mfe_norm / (mae_norm.replace(0.0, np.nan))
    recovery_ratio = recovery_ratio.replace([np.inf, -np.inf], np.nan).clip(upper=recovery_ratio_cap)
    capture_return = ret.where(ret.notna(), utility)
    capture_efficiency = capture_return / mfe.replace(0.0, np.nan)
    capture_efficiency = capture_efficiency.replace([np.inf, -np.inf], np.nan).clip(
        lower=-capture_eff_cap,
        upper=capture_eff_cap,
    )
    opportunity = (
        outcome_available
        & (mfe_norm_for_rules >= opportunity_mfe_threshold)
        & (mae_norm_for_rules <= opportunity_max_mae)
    )
    clean_win = (
        outcome_available
        & (utility > 0.0)
        & (mae_norm_for_rules <= clean_path_max_mae)
        & (~timeout_bool)
        & (~wide_barrier_bool)
    )
    dirty_win = outcome_available & (utility > 0.0) & (~clean_win)
    path_failure = outcome_available & (bad_mae_bool | wide_barrier_bool | (mae_norm_for_rules > opportunity_max_mae))
    timeout_failure = outcome_available & timeout_bool & (utility <= 0.0)
    missed_opportunity = opportunity & (utility <= 0.0)
    reversal_trap = outcome_available & (mfe_norm_for_rules >= opportunity_mfe_threshold) & path_failure
    no_edge = outcome_available & (utility <= 0.0) & (mfe_norm_for_rules < opportunity_mfe_threshold)
    high_recovery = outcome_available & (recovery_ratio >= recovery_ratio_threshold)
    opportunity_captured = opportunity & (utility > 0.0) & (capture_efficiency >= capture_eff_threshold)
    opportunity_capture_loss = opportunity & ((utility <= 0.0) | (capture_efficiency <= capture_loss_eff_threshold))
    economic_capture = (
        opportunity
        & (utility > economic_capture_min_utility)
        & (capture_efficiency >= economic_capture_min_efficiency)
        & (mae_norm_for_rules <= economic_capture_max_mae)
        & (~timeout_bool)
        & (~wide_barrier_bool)
        & (~bad_mae_bool)
    )
    expensive_capture = opportunity_captured & (~economic_capture)
    economic_capture_loss = opportunity & (~economic_capture)

    outcomes = pd.DataFrame(
        {
            "realized_net_utility": utility,
            "realized_return": ret,
            "adverse_mae": mae,
            "adverse_mae_norm": mae_norm,
            "favorable_mfe": mfe,
            "favorable_mfe_norm": mfe_norm,
            "barrier_width": barrier_abs,
            "timeout_flag": (timeout > 0.5).astype(float),
            "bad_mae_flag": bad_mae.astype(float),
            "wide_barrier_flag": wide_barrier.astype(float),
            "execution_fill_quality": fill_quality,
            "mfe_mae_recovery_ratio": recovery_ratio.astype(np.float32),
            "opportunity_capture_efficiency": capture_efficiency.astype(np.float32),
            "outcome_positive_utility_flag": ((utility > 0.0) & outcome_available).where(outcome_available).astype(float),
            "outcome_clean_win_flag": clean_win.where(outcome_available).astype(float),
            "outcome_dirty_win_flag": dirty_win.where(outcome_available).astype(float),
            "outcome_path_failure_flag": path_failure.where(outcome_available).astype(float),
            "outcome_timeout_failure_flag": timeout_failure.where(outcome_available).astype(float),
            "outcome_recoverable_opportunity_flag": opportunity.where(outcome_available).astype(float),
            "outcome_missed_opportunity_flag": missed_opportunity.where(outcome_available).astype(float),
            "outcome_reversal_trap_flag": reversal_trap.where(outcome_available).astype(float),
            "outcome_no_edge_flag": no_edge.where(outcome_available).astype(float),
            "outcome_high_recovery_ratio_flag": high_recovery.where(outcome_available).astype(float),
            "outcome_opportunity_captured_flag": opportunity_captured.where(outcome_available).astype(float),
            "outcome_opportunity_capture_loss_flag": opportunity_capture_loss.where(outcome_available).astype(float),
            "outcome_economic_capture_flag": economic_capture.where(outcome_available).astype(float),
            "outcome_expensive_capture_flag": expensive_capture.where(outcome_available).astype(float),
            "outcome_economic_capture_loss_flag": economic_capture_loss.where(outcome_available).astype(float),
        },
        index=frame.index,
    )
    report = {
        "outcome_columns": {
            "realized_net_utility": utility_col,
            "realized_return": ret_col,
            "adverse_mae": mae_col,
            "favorable_mfe": mfe_col,
            "barrier_width": barrier_col,
            "timeout_flag": timeout_col,
            "bad_mae_flag": bad_flag_col,
            "wide_barrier_flag": wide_flag_col,
            "execution_fill_quality": fill_col,
        },
        "outcome_rows": int(outcomes["realized_net_utility"].notna().sum()),
    }
    return outcomes, report


def _training_quantile(values: pd.Series, timestamps: pd.Series, q: float, calibration_fraction: float) -> float:
    values = _safe_numeric(values)
    ts = pd.to_datetime(timestamps, errors="coerce")
    order = np.argsort(ts.astype("int64").to_numpy(dtype=np.int64, copy=False), kind="mergesort")
    cutoff = max(1, int(math.ceil(float(calibration_fraction) * len(order))))
    calibration = values.iloc[order[:cutoff]].dropna()
    return float(calibration.quantile(q)) if len(calibration) else float("nan")


def _period_series(timestamps: pd.Series, freq: str) -> pd.Series:
    return pd.to_datetime(timestamps, errors="coerce").dt.to_period(freq).astype(str)


def _top_capture_metrics(group: pd.DataFrame, proxy_col: str | None, utility_col: str, top_frac: float) -> dict[str, Any]:
    if proxy_col is None or proxy_col not in group.columns or utility_col not in group.columns:
        return {"proxy_ic": float("nan"), "proxy_topk_mean_utility": float("nan"), "capture_at_k": float("nan")}
    valid = group[[proxy_col, utility_col]].dropna()
    if len(valid) < 5:
        return {"proxy_ic": float("nan"), "proxy_topk_mean_utility": float("nan"), "capture_at_k": float("nan")}
    k = max(1, int(math.ceil(float(top_frac) * len(valid))))
    proxy_top = valid.sort_values(proxy_col, ascending=False).head(k)
    oracle_top_idx = set(valid.sort_values(utility_col, ascending=False).head(k).index)
    proxy_top_idx = set(proxy_top.index)
    return {
        "proxy_ic": _spearman(valid[proxy_col], valid[utility_col]),
        "proxy_topk_mean_utility": _safe_mean(proxy_top[utility_col]),
        "capture_at_k": float(len(oracle_top_idx & proxy_top_idx) / max(1, len(oracle_top_idx))),
    }


def _quality_metric_row(
    group: pd.DataFrame,
    *,
    scope: str,
    bucket: str,
    value: str,
    total_rows: int,
    proxy_col: str | None,
    proxy_top_frac: float,
) -> dict[str, Any]:
    utility = group["realized_net_utility"]
    weekly = group.groupby("week", dropna=False)["realized_net_utility"].mean() if "week" in group.columns else pd.Series(dtype=float)
    row = {
        "scope": scope,
        "bucket": bucket,
        "value": value,
        "rows": int(len(group)),
        "coverage_pct": float(len(group) / total_rows) if total_rows else 0.0,
        "mean_net_utility": _safe_mean(utility),
        "median_net_utility": _safe_quantile(utility, 0.50),
        "p25_net_utility": _safe_quantile(utility, 0.25),
        "weekly_mean_lower_tail_p10": _safe_quantile(weekly, 0.10),
        "weekly_mean_lower_tail_p25": _safe_quantile(weekly, 0.25),
        "bad_mae_rate": _safe_mean(group["bad_mae_flag"] > 0.5),
        "p90_mae": _safe_quantile(group["adverse_mae_norm"], 0.90),
        "timeout_rate": _safe_mean(group["timeout_flag"] > 0.5),
        "wide_barrier_rate": _safe_mean(group["wide_barrier_flag"] > 0.5),
        "average_barrier_width": _safe_mean(group["barrier_width"]),
    }
    proxy = _top_capture_metrics(group, proxy_col, "realized_net_utility", proxy_top_frac)
    row.update(proxy)
    return row


def _score_selection_metric_row(
    group: pd.DataFrame,
    *,
    scope: str,
    bucket: str,
    score_col: str,
    top_frac: float,
    high_good: bool = True,
) -> dict[str, Any]:
    score = _safe_numeric(group[score_col])
    utility = _safe_numeric(group["realized_net_utility"])
    mask = score.notna() & utility.notna()
    valid = group.loc[mask].copy()
    if valid.empty:
        return {
            "scope": scope,
            "bucket": bucket,
            "score_col": score_col,
            "top_frac": float(top_frac),
            "rows": int(len(group)),
            "selected_rows": 0,
            "selected_coverage_pct": 0.0,
            "score_ic_utility": float("nan"),
            "mean_net_utility": float("nan"),
            "bad_mae_rate": float("nan"),
            "p90_mae": float("nan"),
            "timeout_rate": float("nan"),
            "wide_barrier_rate": float("nan"),
        }
    order_score = _safe_numeric(valid[score_col])
    order = order_score.sort_values(ascending=not high_good, kind="mergesort")
    k = max(1, int(math.ceil(float(top_frac) * len(order))))
    selected = valid.loc[order.index[:k]]
    return {
        "scope": scope,
        "bucket": bucket,
        "score_col": score_col,
        "top_frac": float(top_frac),
        "rows": int(len(valid)),
        "selected_rows": int(len(selected)),
        "selected_coverage_pct": float(len(selected) / len(valid)) if len(valid) else 0.0,
        "score_ic_utility": _spearman(valid[score_col], valid["realized_net_utility"]),
        "mean_net_utility": _safe_mean(selected["realized_net_utility"]),
        "median_net_utility": _safe_quantile(selected["realized_net_utility"], 0.50),
        "p25_net_utility": _safe_quantile(selected["realized_net_utility"], 0.25),
        "bad_mae_rate": _safe_mean(selected["bad_mae_flag"] > 0.5),
        "p90_mae": _safe_quantile(selected["adverse_mae_norm"], 0.90),
        "timeout_rate": _safe_mean(selected["timeout_flag"] > 0.5),
        "wide_barrier_rate": _safe_mean(selected["wide_barrier_flag"] > 0.5),
        "top_symbol_share": float(selected["__symbol__"].value_counts(normalize=True).iloc[0])
        if "__symbol__" in selected.columns and len(selected)
        else float("nan"),
    }


def evaluate_source_score_slices(full: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    timestamp_col = str(config.get("timestamp_col") or "__ts__")
    diagnostics = config.get("diagnostics") or {}
    top_fracs = [float(v) for v in diagnostics.get("source_score_top_fracs", [0.01, 0.03, 0.05, 0.10])]
    eval_frame = full.loc[full["realized_net_utility"].notna()].copy()
    if eval_frame.empty:
        return pd.DataFrame(), pd.DataFrame()
    eval_frame["month"] = _period_series(eval_frame[timestamp_col], "M")
    if "not_dirty_shock_score" not in eval_frame.columns and "dirty_shock_avoid_score" in eval_frame.columns:
        eval_frame["not_dirty_shock_score"] = (1.0 - _safe_numeric(eval_frame["dirty_shock_avoid_score"])).clip(0.0, 1.0)
    if "loud_clean_source_score" not in eval_frame.columns and {"loud_breakout_impulse_score", "dirty_shock_avoid_score"}.issubset(eval_frame.columns):
        eval_frame["loud_clean_source_score"] = (
            _safe_numeric(eval_frame["loud_breakout_impulse_score"])
            * (1.0 - _safe_numeric(eval_frame["dirty_shock_avoid_score"]).clip(0.0, 1.0))
        ).clip(0.0, 1.0)
    score_cols = [col for col in SOURCE_SCORE_EVAL_COLS if col in eval_frame.columns]
    rows: list[dict[str, Any]] = []
    for score_col in score_cols:
        for top_frac in top_fracs:
            rows.append(
                _score_selection_metric_row(
                    eval_frame,
                    scope="overall",
                    bucket="all",
                    score_col=score_col,
                    top_frac=top_frac,
                    high_good=True,
                )
            )
    overall = pd.DataFrame(rows)
    month_rows: list[dict[str, Any]] = []
    for month, group in eval_frame.groupby("month", dropna=False):
        for score_col in score_cols:
            for top_frac in top_fracs:
                month_rows.append(
                    _score_selection_metric_row(
                        group,
                        scope="month",
                        bucket=str(month),
                        score_col=score_col,
                        top_frac=top_frac,
                        high_good=True,
                    )
                )
    return overall, pd.DataFrame(month_rows)


def _score_target_diagnostic_row(
    group: pd.DataFrame,
    *,
    scope: str,
    bucket: str,
    score_col: str,
    target_col: str,
    top_frac: float,
) -> dict[str, Any]:
    score = _safe_numeric(group[score_col])
    target = _safe_numeric(group[target_col])
    mask = score.notna() & target.notna()
    valid = group.loc[mask].copy()
    if valid.empty:
        return {
            "scope": scope,
            "bucket": bucket,
            "score_col": score_col,
            "target_col": target_col,
            "top_frac": float(top_frac),
            "rows": int(len(group)),
            "valid_rows": 0,
            "selected_rows": 0,
            "score_ic_target": float("nan"),
            "target_mean": float("nan"),
            "selected_target_mean": float("nan"),
            "selected_target_delta": float("nan"),
            "selected_target_lift_ratio": float("nan"),
        }
    order = _safe_numeric(valid[score_col]).sort_values(ascending=False, kind="mergesort")
    k = max(1, int(math.ceil(float(top_frac) * len(order))))
    selected = valid.loc[order.index[:k]]
    target_mean = _safe_mean(valid[target_col])
    selected_target_mean = _safe_mean(selected[target_col])
    lift_ratio = (
        float(selected_target_mean / target_mean)
        if target_mean is not None and math.isfinite(float(target_mean)) and abs(float(target_mean)) > 1e-12
        else float("nan")
    )
    opportunity = selected.get("outcome_recoverable_opportunity_flag")
    captured = selected.get("outcome_opportunity_captured_flag")
    selected_opportunity_rate = _safe_mean(opportunity > 0.5) if opportunity is not None else float("nan")
    selected_capture_rate = _safe_mean(captured > 0.5) if captured is not None else float("nan")
    if opportunity is not None and captured is not None:
        opportunity_rows = selected[opportunity.fillna(0.0).gt(0.5)]
        selected_capture_among_opportunity = _safe_mean(opportunity_rows["outcome_opportunity_captured_flag"] > 0.5)
    else:
        selected_capture_among_opportunity = float("nan")
    return {
        "scope": scope,
        "bucket": bucket,
        "score_col": score_col,
        "target_col": target_col,
        "top_frac": float(top_frac),
        "rows": int(len(group)),
        "valid_rows": int(len(valid)),
        "selected_rows": int(len(selected)),
        "score_ic_target": _spearman(valid[score_col], valid[target_col]),
        "target_mean": target_mean,
        "selected_target_mean": selected_target_mean,
        "selected_target_delta": float(selected_target_mean - target_mean)
        if math.isfinite(float(selected_target_mean)) and math.isfinite(float(target_mean))
        else float("nan"),
        "selected_target_lift_ratio": lift_ratio,
        "selected_mean_net_utility": _safe_mean(selected["realized_net_utility"]) if "realized_net_utility" in selected.columns else float("nan"),
        "selected_positive_utility_rate": _safe_mean(selected["realized_net_utility"] > 0.0) if "realized_net_utility" in selected.columns else float("nan"),
        "selected_opportunity_rate": selected_opportunity_rate,
        "selected_capture_rate": selected_capture_rate,
        "selected_capture_among_opportunity_rate": selected_capture_among_opportunity,
        "selected_capture_efficiency_mean": _safe_mean(selected["opportunity_capture_efficiency"]) if "opportunity_capture_efficiency" in selected.columns else float("nan"),
        "selected_path_failure_rate": _safe_mean(selected["outcome_path_failure_flag"] > 0.5) if "outcome_path_failure_flag" in selected.columns else float("nan"),
        "selected_no_edge_rate": _safe_mean(selected["outcome_no_edge_flag"] > 0.5) if "outcome_no_edge_flag" in selected.columns else float("nan"),
    }


def evaluate_source_score_target_diagnostics(full: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    timestamp_col = str(config.get("timestamp_col") or "__ts__")
    diagnostics = config.get("diagnostics") or {}
    top_fracs = [float(v) for v in diagnostics.get("source_score_top_fracs", [0.01, 0.03, 0.05, 0.10])]
    eval_frame = full.loc[full["realized_net_utility"].notna()].copy()
    if eval_frame.empty:
        return pd.DataFrame(), pd.DataFrame()
    eval_frame["month"] = _period_series(eval_frame[timestamp_col], "M")
    score_cols = [col for col in dict.fromkeys(SOURCE_SCORE_TARGET_COLS) if col in eval_frame.columns]
    target_cols = [col for col in SOURCE_SCORE_TARGETS if col in eval_frame.columns]
    rows: list[dict[str, Any]] = []
    for score_col in score_cols:
        for target_col in target_cols:
            for top_frac in top_fracs:
                rows.append(
                    _score_target_diagnostic_row(
                        eval_frame,
                        scope="overall",
                        bucket="all",
                        score_col=score_col,
                        target_col=target_col,
                        top_frac=top_frac,
                    )
                )
    month_rows: list[dict[str, Any]] = []
    for month, group in eval_frame.groupby("month", dropna=False):
        for score_col in score_cols:
            for target_col in target_cols:
                for top_frac in top_fracs:
                    month_rows.append(
                        _score_target_diagnostic_row(
                            group,
                            scope="month",
                            bucket=str(month),
                            score_col=score_col,
                            target_col=target_col,
                            top_frac=top_frac,
                        )
                    )
    return pd.DataFrame(rows), pd.DataFrame(month_rows)


def evaluate_source_quality(
    full: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    timestamp_col = str(config.get("timestamp_col") or "__ts__")
    symbol_col = str(config.get("symbol_col") or "__symbol__")
    diagnostics = config.get("diagnostics") or {}
    proxy_col = _first_existing(full, config.get("proxy_score_columns") or [])
    proxy_top_frac = float(diagnostics.get("proxy_top_frac", 0.10))
    outcome_mask = full["realized_net_utility"].notna()
    eval_frame = full.loc[outcome_mask].copy()
    if eval_frame.empty:
        empty = pd.DataFrame()
        return empty, empty, empty, empty, {"proxy_score_column": proxy_col, "outcome_eval_rows": 0}
    eval_frame["month"] = _period_series(eval_frame[timestamp_col], "M")
    eval_frame["week"] = _period_series(eval_frame[timestamp_col], "W-SUN")
    total = int(len(eval_frame))

    def rows_for_period(period_col: str) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for period, period_group in eval_frame.groupby(period_col, dropna=False):
            period_total = int(len(period_group))
            for tag in TAG_COLS:
                if tag not in period_group.columns:
                    continue
                selected = period_group[period_group[tag].astype(bool)]
                if len(selected):
                    rows.append(
                        _quality_metric_row(
                            selected,
                            scope="multi_tag",
                            bucket=str(period),
                            value=tag.replace("tag_", ""),
                            total_rows=period_total,
                            proxy_col=proxy_col,
                            proxy_top_frac=proxy_top_frac,
                        )
                    )
            for primary, selected in period_group.groupby("primary_source_tag", dropna=False):
                rows.append(
                    _quality_metric_row(
                        selected,
                        scope="primary_source_tag",
                        bucket=str(period),
                        value=str(primary),
                        total_rows=period_total,
                        proxy_col=proxy_col,
                        proxy_top_frac=proxy_top_frac,
                    )
                )
        return pd.DataFrame(rows)

    by_month = rows_for_period("month")
    by_week = rows_for_period("week")

    symbol_rows: list[dict[str, Any]] = []
    for symbol, symbol_group in eval_frame.groupby(symbol_col, dropna=False):
        symbol_total = int(len(symbol_group))
        for primary, selected in symbol_group.groupby("primary_source_tag", dropna=False):
            symbol_rows.append(
                _quality_metric_row(
                    selected,
                    scope="primary_source_tag",
                    bucket=str(symbol),
                    value=str(primary),
                    total_rows=symbol_total,
                    proxy_col=proxy_col,
                    proxy_top_frac=proxy_top_frac,
                )
            )
    by_symbol = pd.DataFrame(symbol_rows)

    regime_rows: list[dict[str, Any]] = []
    regime_cols = [col for col in config.get("regime_head_columns") or [] if col in eval_frame.columns]
    for regime_col in regime_cols:
        for regime_value, regime_group in eval_frame.groupby(regime_col, dropna=False):
            regime_total = int(len(regime_group))
            for tag in TAG_COLS:
                selected = regime_group[regime_group[tag].astype(bool)]
                if len(selected):
                    row = _quality_metric_row(
                        selected,
                        scope=str(regime_col),
                        bucket=str(regime_value),
                        value=tag.replace("tag_", ""),
                        total_rows=regime_total,
                        proxy_col=proxy_col,
                        proxy_top_frac=proxy_top_frac,
                    )
                    row["regime_col"] = str(regime_col)
                    row["regime_value"] = str(regime_value)
                    row["source_tag"] = tag.replace("tag_", "")
                    regime_rows.append(row)
    source_x_regime = pd.DataFrame(regime_rows)
    return by_month, by_week, by_symbol, source_x_regime, {
        "proxy_score_column": proxy_col,
        "outcome_eval_rows": total,
        "regime_columns_present": regime_cols,
    }


def _outcome_pct_rank_by_timestamp(values: pd.Series, timestamps: pd.Series, *, high_good: bool = True) -> pd.Series:
    ranks = _safe_numeric(values).groupby(timestamps, dropna=False).rank(method="average", pct=True)
    if not high_good:
        ranks = 1.0 - ranks
    return ranks.fillna(0.5).clip(0.0, 1.0).astype(np.float32)


def _outcome_pct_rank_by_timestamp_and_group(
    values: pd.Series,
    timestamps: pd.Series,
    groups: pd.Series,
    *,
    high_good: bool = True,
    min_count: int = 5,
) -> pd.Series:
    values = _safe_numeric(values)
    ts = pd.to_datetime(timestamps, errors="coerce")
    group_key = groups.astype(str).fillna("missing")
    grouped = values.groupby([ts, group_key], dropna=False)
    ranks = grouped.rank(method="average", pct=True)
    counts = grouped.transform("count")
    fallback = _outcome_pct_rank_by_timestamp(values, timestamps, high_good=True)
    ranks = ranks.where(counts >= int(min_count), fallback)
    if not high_good:
        ranks = 1.0 - ranks
    return ranks.fillna(0.5).clip(0.0, 1.0).astype(np.float32)


def _expanding_group_quantile_thresholds(
    values: pd.Series,
    timestamps: pd.Series,
    groups: pd.Series,
    *,
    low_q: float,
    high_q: float,
    min_group_prior: int,
    min_global_prior: int,
) -> tuple[pd.Series, pd.Series]:
    values = _safe_numeric(values)
    ts = pd.to_datetime(timestamps, errors="coerce")
    group_key = groups.astype(str).fillna("missing")
    work = pd.DataFrame(
        {
            "__ts__": ts,
            "__group__": group_key,
            "__value__": values,
            "__pos__": np.arange(len(values), dtype=np.int64),
        }
    ).sort_values(["__ts__", "__group__", "__pos__"], kind="mergesort")
    low_out = np.full(len(values), np.nan, dtype=np.float32)
    high_out = np.full(len(values), np.nan, dtype=np.float32)
    prior_by_group: dict[str, list[float]] = {}
    prior_global: list[float] = []

    for _, ts_group in work.groupby("__ts__", sort=False):
        for group_value, current in ts_group.groupby("__group__", sort=False):
            group_prior = prior_by_group.get(str(group_value), [])
            if len(group_prior) >= int(min_group_prior):
                reference = group_prior
            elif len(prior_global) >= int(min_global_prior):
                reference = prior_global
            else:
                reference = []
            if reference:
                ref = np.asarray(reference, dtype=np.float64)
                positions = current["__pos__"].to_numpy(dtype=np.int64, copy=False)
                low_out[positions] = float(np.nanquantile(ref, float(low_q)))
                high_out[positions] = float(np.nanquantile(ref, float(high_q)))
        finite_current = ts_group[np.isfinite(ts_group["__value__"].to_numpy(dtype=np.float64, copy=False))]
        for group_value, current in finite_current.groupby("__group__", sort=False):
            vals = [float(v) for v in current["__value__"].tolist() if math.isfinite(float(v))]
            if not vals:
                continue
            prior_by_group.setdefault(str(group_value), []).extend(vals)
            prior_global.extend(vals)

    return pd.Series(low_out, index=values.index), pd.Series(high_out, index=values.index)


def build_quality_label_candidates(
    source: pd.DataFrame,
    outcomes: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    timestamp_col = str(config.get("timestamp_col") or "__ts__")
    thresholds = config.get("label_thresholds") or {}
    quality_parts = pd.DataFrame(index=source.index)
    weights: dict[str, float] = {}
    if outcomes["realized_net_utility"].notna().any():
        quality_parts["utility_rank"] = _outcome_pct_rank_by_timestamp(
            outcomes["realized_net_utility"],
            source[timestamp_col],
            high_good=True,
        )
        weights["utility_rank"] = 0.60
    if outcomes["adverse_mae_norm"].notna().any():
        quality_parts["mae_quality_rank"] = _outcome_pct_rank_by_timestamp(
            outcomes["adverse_mae_norm"],
            source[timestamp_col],
            high_good=False,
        )
        weights["mae_quality_rank"] = 0.15
    if outcomes["timeout_flag"].notna().any():
        quality_parts["timeout_quality"] = (1.0 - outcomes["timeout_flag"].fillna(1.0)).clip(0.0, 1.0)
        weights["timeout_quality"] = 0.10
    if outcomes["wide_barrier_flag"].notna().any():
        quality_parts["barrier_quality"] = (1.0 - outcomes["wide_barrier_flag"].fillna(1.0)).clip(0.0, 1.0)
        weights["barrier_quality"] = 0.10
    if outcomes["execution_fill_quality"].notna().any():
        quality_parts["fill_quality"] = clip01(outcomes["execution_fill_quality"]).fillna(0.5)
        weights["fill_quality"] = 0.05
    score = weighted_mean_available(quality_parts, list(weights.keys()), list(weights.values())).fillna(np.nan)
    outcome_available = outcomes["realized_net_utility"].notna()
    score = score.where(outcome_available)
    score_rank = _outcome_pct_rank_by_timestamp(score, source[timestamp_col], high_good=True)
    good_threshold = 1.0 - float(thresholds.get("good_top_frac", 0.30))
    bad_threshold = float(thresholds.get("bad_bottom_frac", 0.40))
    utility = outcomes["realized_net_utility"]
    bad_mae = outcomes["bad_mae_flag"] > 0.5
    labels = pd.Series(-1, index=source.index, dtype=np.int8)
    labels[(score_rank >= good_threshold) & (utility > 0.0) & (~bad_mae)] = 1
    labels[(score_rank <= bad_threshold) | (utility < 0.0) | bad_mae] = 0
    labels[utility.isna()] = -1

    source_rank = _outcome_pct_rank_by_timestamp_and_group(
        score,
        source[timestamp_col],
        source["primary_source_tag"] if "primary_source_tag" in source.columns else pd.Series("all", index=source.index),
        high_good=True,
        min_count=int(thresholds.get("source_conditioned_min_timestamp_rows", 5)),
    )
    source_rank_labels = pd.Series(-1, index=source.index, dtype=np.int8)
    source_rank_labels[(source_rank >= good_threshold) & (utility > 0.0) & (~bad_mae)] = 1
    source_rank_labels[(source_rank <= bad_threshold) | (utility < 0.0) | bad_mae] = 0
    source_rank_labels[utility.isna()] = -1

    source_bad_cutoff, source_good_cutoff = _expanding_group_quantile_thresholds(
        score,
        source[timestamp_col],
        source["primary_source_tag"] if "primary_source_tag" in source.columns else pd.Series("all", index=source.index),
        low_q=bad_threshold,
        high_q=good_threshold,
        min_group_prior=int(thresholds.get("source_conditioned_min_prior_rows", 200)),
        min_global_prior=int(thresholds.get("source_conditioned_global_min_prior_rows", 1000)),
    )
    source_wf_labels = pd.Series(-1, index=source.index, dtype=np.int8)
    wf_calibrated = source_bad_cutoff.notna() & source_good_cutoff.notna() & utility.notna()
    source_wf_labels[wf_calibrated & (score >= source_good_cutoff) & (utility > 0.0) & (~bad_mae)] = 1
    source_wf_labels[wf_calibrated & ((score <= source_bad_cutoff) | (utility < 0.0) | bad_mae)] = 0

    clean_win = outcomes["outcome_clean_win_flag"] > 0.5
    path_failure = outcomes["outcome_path_failure_flag"] > 0.5
    timeout_failure = outcomes["outcome_timeout_failure_flag"] > 0.5
    recoverable_opportunity = outcomes["outcome_recoverable_opportunity_flag"] > 0.5
    no_edge = outcomes["outcome_no_edge_flag"] > 0.5
    clean_path_labels = pd.Series(-1, index=source.index, dtype=np.int8)
    clean_path_labels[clean_win] = 1
    clean_path_labels[(utility < 0.0) | path_failure | timeout_failure | no_edge] = 0
    clean_path_labels[utility.isna()] = -1
    recoverable_labels = pd.Series(-1, index=source.index, dtype=np.int8)
    recoverable_labels[recoverable_opportunity] = 1
    recoverable_labels[path_failure | no_edge] = 0
    recoverable_labels[utility.isna()] = -1
    opportunity_captured = outcomes["outcome_opportunity_captured_flag"] > 0.5
    opportunity_capture_loss = outcomes["outcome_opportunity_capture_loss_flag"] > 0.5
    capture_labels = pd.Series(-1, index=source.index, dtype=np.int8)
    capture_labels[opportunity_captured] = 1
    capture_labels[opportunity_capture_loss] = 0
    capture_labels[utility.isna()] = -1
    economic_capture = outcomes["outcome_economic_capture_flag"] > 0.5
    economic_capture_loss = outcomes["outcome_economic_capture_loss_flag"] > 0.5
    economic_capture_labels = pd.Series(-1, index=source.index, dtype=np.int8)
    economic_capture_labels[economic_capture] = 1
    economic_capture_labels[economic_capture_loss] = 0
    economic_capture_labels[utility.isna()] = -1

    out_cols = _configured_metadata_columns(config, source)
    out_cols.extend([col for col in COMPONENT_COLS + ARCHETYPE_COLS + TAG_COLS + ["primary_source_tag"] if col in source.columns])
    out = source.loc[:, list(dict.fromkeys(out_cols))].copy()
    out["realized_quality_score_v0"] = score.astype(np.float32)
    out["realized_quality_rank_v0"] = score_rank.astype(np.float32)
    out["quality_label_v0"] = labels
    out["realized_quality_source_rank_v1"] = source_rank.astype(np.float32)
    out["source_wf_bad_cutoff_v1"] = source_bad_cutoff.astype(np.float32)
    out["source_wf_good_cutoff_v1"] = source_good_cutoff.astype(np.float32)
    out["quality_label_source_rank_v1"] = source_rank_labels
    out["quality_label_source_wf_v1"] = source_wf_labels
    for outcome_col in [
        "mfe_mae_recovery_ratio",
        "opportunity_capture_efficiency",
        "outcome_positive_utility_flag",
        "outcome_clean_win_flag",
        "outcome_dirty_win_flag",
        "outcome_path_failure_flag",
        "outcome_timeout_failure_flag",
        "outcome_recoverable_opportunity_flag",
        "outcome_missed_opportunity_flag",
        "outcome_reversal_trap_flag",
        "outcome_no_edge_flag",
        "outcome_high_recovery_ratio_flag",
        "outcome_opportunity_captured_flag",
        "outcome_opportunity_capture_loss_flag",
        "outcome_economic_capture_flag",
        "outcome_expensive_capture_flag",
        "outcome_economic_capture_loss_flag",
    ]:
        if outcome_col in outcomes.columns:
            out[outcome_col] = outcomes[outcome_col].astype(np.float32)
    out["quality_label_clean_path_v2"] = clean_path_labels
    out["quality_label_recoverable_opportunity_v2"] = recoverable_labels
    out["quality_label_opportunity_capture_v3"] = capture_labels
    out["quality_label_economic_capture_v4"] = economic_capture_labels
    out["train_include_all_rows_v0"] = labels.ne(-1)
    out["train_include_non_neutral_v0"] = labels.ne(-1)
    out["train_include_source_rank_non_neutral_v1"] = source_rank_labels.ne(-1)
    out["train_include_source_wf_non_neutral_v1"] = source_wf_labels.ne(-1)
    out["train_include_clean_path_non_neutral_v2"] = clean_path_labels.ne(-1)
    out["train_include_recoverable_opportunity_non_neutral_v2"] = recoverable_labels.ne(-1)
    out["train_include_opportunity_capture_non_neutral_v3"] = capture_labels.ne(-1)
    out["train_include_economic_capture_non_neutral_v4"] = economic_capture_labels.ne(-1)
    out["train_include_missed_opportunity_review_v2"] = outcomes["outcome_missed_opportunity_flag"].fillna(0.0).gt(0.5)
    out["train_include_no_edge_review_v2"] = outcomes["outcome_no_edge_flag"].fillna(0.0).gt(0.5)
    out["train_include_opportunity_capture_loss_review_v3"] = outcomes["outcome_opportunity_capture_loss_flag"].fillna(0.0).gt(0.5)
    out["train_include_economic_capture_loss_review_v4"] = outcomes["outcome_economic_capture_loss_flag"].fillna(0.0).gt(0.5)
    out["train_include_quiet_only_v0"] = out["tag_quiet_continuation"] & labels.ne(-1)
    out["train_include_loud_only_v0"] = out["tag_loud_breakout_impulse"] & labels.ne(-1)
    out["train_include_loud_clean_v0"] = out["tag_loud_breakout_impulse"] & (~out["tag_dirty_shock_avoid"]) & labels.ne(-1)
    out["train_include_dirty_excluded_v0"] = (~out["tag_dirty_shock_avoid"]) & labels.ne(-1)
    out["train_include_run_entry_only_v0"] = out["tag_run_entry"] & labels.ne(-1)
    out["train_include_retest_only_v0"] = out["tag_retest_reversal"] & labels.ne(-1)
    out["train_include_compression_only_v0"] = out["tag_compression_release"] & labels.ne(-1)
    out["train_include_clean_execution_context_v1"] = out.get("tag_clean_execution_context", False) & labels.ne(-1)
    out["train_include_calm_positive_source_v1"] = out.get("tag_calm_positive_source", False) & labels.ne(-1)
    out["train_include_loud_clean_execution_v1"] = out.get("tag_loud_clean_execution", False) & labels.ne(-1)
    out["train_include_clean_run_entry_v1"] = out.get("tag_clean_run_entry", False) & labels.ne(-1)
    out["train_include_compression_capture_candidate_v3"] = out.get("tag_compression_capture_candidate", False) & labels.ne(-1)
    out["train_include_risk_adjusted_capture_candidate_v4"] = out.get("tag_risk_adjusted_capture_candidate", False) & labels.ne(-1)
    out["train_include_clean_economic_capture_candidate_v5"] = (
        out.get("tag_clean_economic_capture_candidate", False) & economic_capture_labels.ne(-1)
    )
    out["train_include_misleading_location_risk_excluded_v3"] = (~out.get("tag_misleading_location_risk", False)) & labels.ne(-1)
    score_thresholds = {
        "train_include_quiet_score_top10_v1": ("quiet_continuation_score", 0.10),
        "train_include_base_positive_score_top10_v1": ("base_positive_source_score", 0.10),
        "train_include_loud_clean_score_top10_v1": ("loud_clean_source_score", 0.10),
        "train_include_not_dirty_score_top70_v1": ("not_dirty_shock_score", 0.70),
        "train_include_run_entry_score_top10_v1": ("run_entry_score", 0.10),
        "train_include_clean_execution_score_top20_v1": ("clean_execution_context_score", 0.20),
        "train_include_calm_positive_score_top10_v1": ("calm_positive_source_score", 0.10),
        "train_include_loud_clean_execution_score_top05_v1": ("loud_clean_execution_score", 0.05),
        "train_include_clean_run_entry_score_top10_v1": ("clean_run_entry_score", 0.10),
        "train_include_compression_capture_score_top10_v3": ("compression_capture_candidate_score", 0.10),
        "train_include_risk_adjusted_capture_score_top10_v4": ("risk_adjusted_capture_candidate_score", 0.10),
        "train_include_clean_economic_capture_score_top10_v5": ("clean_economic_capture_candidate_score", 0.10),
    }
    for flag_col, (score_col, top_frac) in score_thresholds.items():
        if score_col in source.columns:
            flag, _, _ = bool_from_score_rank(
                source[score_col],
                source[timestamp_col],
                float(top_frac),
                min_count=int((config.get("tag_thresholds") or {}).get("min_timestamp_rows", 5)),
            )
            out[flag_col] = flag & labels.ne(-1)
        else:
            out[flag_col] = False
    out["train_include_compression_economic_capture_score_top10_v4"] = (
        out.get("train_include_compression_capture_score_top10_v3", False) & economic_capture_labels.ne(-1)
    )
    out["train_include_risk_adjusted_economic_capture_score_top10_v4"] = (
        out.get("train_include_risk_adjusted_capture_score_top10_v4", False) & economic_capture_labels.ne(-1)
    )
    out["train_include_clean_economic_capture_score_top10_v5"] = (
        out.get("train_include_clean_economic_capture_score_top10_v5", False) & economic_capture_labels.ne(-1)
    )
    if "misleading_location_risk_score" in source.columns:
        high_risk_flag, _, _ = bool_from_score_rank(
            source["misleading_location_risk_score"],
            source[timestamp_col],
            0.30,
            min_count=int((config.get("tag_thresholds") or {}).get("min_timestamp_rows", 5)),
        )
        out["train_include_misleading_location_risk_bottom70_v3"] = (~high_risk_flag) & labels.ne(-1)
    else:
        out["train_include_misleading_location_risk_bottom70_v3"] = False

    neutral_weight = float(thresholds.get("neutral_weight", 0.0))
    base_weight = pd.Series(neutral_weight, index=source.index, dtype=np.float32)
    base_weight[labels.isin([0, 1])] = 1.0
    multipliers = config.get("sample_weight_multipliers") or {}
    source_weight = base_weight.copy()
    source_weight *= np.where(out["tag_dirty_shock_avoid"], float(multipliers.get("dirty_shock_avoid", 0.25)), 1.0)
    source_weight *= np.where(out["tag_run_entry"], float(multipliers.get("run_entry", 1.25)), 1.0)
    source_weight *= np.where(out["tag_late_run_continuation"], float(multipliers.get("late_run_continuation", 0.75)), 1.0)
    source_weight *= np.where(out["tag_ambiguous_none"], float(multipliers.get("ambiguous_none", 0.50)), 1.0)
    out["sample_weight_base_v0"] = base_weight.astype(np.float32)
    out["sample_weight_source_v1"] = source_weight.astype(np.float32)
    base_score = _safe_numeric(source.get("base_positive_source_score", pd.Series(0.5, index=source.index))).fillna(0.5).clip(0.0, 1.0)
    dirty_score = _safe_numeric(source.get("dirty_shock_avoid_score", pd.Series(0.5, index=source.index))).fillna(0.5).clip(0.0, 1.0)
    run_entry_score = _safe_numeric(source.get("run_entry_score", pd.Series(0.5, index=source.index))).fillna(0.5).clip(0.0, 1.0)
    source_weight_v2 = (
        base_weight
        * (0.50 + base_score)
        * (1.0 - 0.75 * dirty_score)
        * (1.0 + 0.25 * run_entry_score)
    ).clip(0.0, 3.0)
    positive_mean = float(source_weight_v2[labels.isin([0, 1])].mean()) if bool(labels.isin([0, 1]).any()) else 1.0
    if positive_mean and math.isfinite(positive_mean):
        source_weight_v2 = source_weight_v2 / positive_mean
    out["sample_weight_source_v2"] = source_weight_v2.astype(np.float32)
    clean_execution_score = _safe_numeric(source.get("clean_execution_context_score", pd.Series(0.5, index=source.index))).fillna(0.5).clip(0.0, 1.0)
    clean_run_entry_score = _safe_numeric(source.get("clean_run_entry_score", pd.Series(0.0, index=source.index))).fillna(0.0).clip(0.0, 1.0)
    source_wf_base_weight = pd.Series(neutral_weight, index=source.index, dtype=np.float32)
    source_wf_base_weight[source_wf_labels.isin([0, 1])] = 1.0
    source_wf_weight = (
        source_wf_base_weight
        * (0.75 + 0.75 * clean_execution_score)
        * (0.75 + 0.75 * base_score)
        * (1.0 - 0.60 * dirty_score)
        * (1.0 + 0.35 * clean_run_entry_score)
    ).clip(0.0, 3.0)
    wf_mean = float(source_wf_weight[source_wf_labels.isin([0, 1])].mean()) if bool(source_wf_labels.isin([0, 1]).any()) else 1.0
    if wf_mean and math.isfinite(wf_mean):
        source_wf_weight = source_wf_weight / wf_mean
    out["sample_weight_source_wf_v1"] = source_wf_weight.astype(np.float32)
    clean_path_weight = pd.Series(neutral_weight, index=source.index, dtype=np.float32)
    clean_path_weight[clean_path_labels.isin([0, 1])] = 1.0
    clean_path_weight *= np.where(out["quality_label_clean_path_v2"].eq(1), 1.25, 1.0)
    clean_path_weight *= np.where(out["outcome_reversal_trap_flag"].fillna(0.0).gt(0.5), 1.15, 1.0)
    out["sample_weight_clean_path_v2"] = clean_path_weight.astype(np.float32)
    opportunity_weight = pd.Series(neutral_weight, index=source.index, dtype=np.float32)
    opportunity_weight[recoverable_labels.isin([0, 1])] = 1.0
    opportunity_weight *= np.where(out["outcome_missed_opportunity_flag"].fillna(0.0).gt(0.5), 1.20, 1.0)
    opportunity_weight *= np.where(out["outcome_high_recovery_ratio_flag"].fillna(0.0).gt(0.5), 1.10, 1.0)
    out["sample_weight_opportunity_v2"] = opportunity_weight.astype(np.float32)
    capture_weight = pd.Series(neutral_weight, index=source.index, dtype=np.float32)
    capture_weight[capture_labels.isin([0, 1])] = 1.0
    capture_weight *= np.where(out["outcome_opportunity_capture_loss_flag"].fillna(0.0).gt(0.5), 1.20, 1.0)
    capture_weight *= np.where(out["outcome_high_recovery_ratio_flag"].fillna(0.0).gt(0.5), 1.10, 1.0)
    out["sample_weight_capture_v3"] = capture_weight.astype(np.float32)
    economic_capture_weight = pd.Series(neutral_weight, index=source.index, dtype=np.float32)
    economic_capture_weight[economic_capture_labels.isin([0, 1])] = 1.0
    economic_capture_weight *= np.where(out["outcome_economic_capture_loss_flag"].fillna(0.0).gt(0.5), 1.25, 1.0)
    economic_capture_weight *= np.where(out["outcome_expensive_capture_flag"].fillna(0.0).gt(0.5), 1.15, 1.0)
    out["sample_weight_economic_capture_v4"] = economic_capture_weight.astype(np.float32)
    expected_metadata = _configured_metadata_columns(config, source)
    missing_metadata = [col for col in expected_metadata if col not in out.columns]
    if missing_metadata:
        raise ValueError(f"quality_label_candidates unexpectedly dropped metadata columns: {missing_metadata}")
    return out, {
        "quality_score_components": weights,
        "metadata_columns_preserved": expected_metadata,
        "quality_label_rows": int(len(out)),
        "label_counts": {str(k): int(v) for k, v in labels.value_counts(dropna=False).sort_index().items()},
        "source_rank_label_counts": {str(k): int(v) for k, v in source_rank_labels.value_counts(dropna=False).sort_index().items()},
        "source_wf_label_counts": {str(k): int(v) for k, v in source_wf_labels.value_counts(dropna=False).sort_index().items()},
        "clean_path_label_counts": {str(k): int(v) for k, v in clean_path_labels.value_counts(dropna=False).sort_index().items()},
        "recoverable_opportunity_label_counts": {str(k): int(v) for k, v in recoverable_labels.value_counts(dropna=False).sort_index().items()},
        "opportunity_capture_label_counts": {str(k): int(v) for k, v in capture_labels.value_counts(dropna=False).sort_index().items()},
        "economic_capture_label_counts": {str(k): int(v) for k, v in economic_capture_labels.value_counts(dropna=False).sort_index().items()},
        "source_wf_calibrated_rows": int(wf_calibrated.sum()),
        "failure_mode_counts": {
            "clean_win": int(out["outcome_clean_win_flag"].fillna(0.0).gt(0.5).sum()),
            "dirty_win": int(out["outcome_dirty_win_flag"].fillna(0.0).gt(0.5).sum()),
            "path_failure": int(out["outcome_path_failure_flag"].fillna(0.0).gt(0.5).sum()),
            "timeout_failure": int(out["outcome_timeout_failure_flag"].fillna(0.0).gt(0.5).sum()),
            "recoverable_opportunity": int(out["outcome_recoverable_opportunity_flag"].fillna(0.0).gt(0.5).sum()),
            "missed_opportunity": int(out["outcome_missed_opportunity_flag"].fillna(0.0).gt(0.5).sum()),
            "no_edge": int(out["outcome_no_edge_flag"].fillna(0.0).gt(0.5).sum()),
            "opportunity_captured": int(out["outcome_opportunity_captured_flag"].fillna(0.0).gt(0.5).sum()),
            "opportunity_capture_loss": int(out["outcome_opportunity_capture_loss_flag"].fillna(0.0).gt(0.5).sum()),
            "economic_capture": int(out["outcome_economic_capture_flag"].fillna(0.0).gt(0.5).sum()),
            "expensive_capture": int(out["outcome_expensive_capture_flag"].fillna(0.0).gt(0.5).sum()),
            "economic_capture_loss": int(out["outcome_economic_capture_loss_flag"].fillna(0.0).gt(0.5).sum()),
        },
        "score_flag_counts": {
            col: int(out[col].sum())
            for col in list(score_thresholds)
            + [
                "train_include_misleading_location_risk_bottom70_v3",
                "train_include_clean_economic_capture_candidate_v5",
                "train_include_compression_economic_capture_score_top10_v4",
                "train_include_risk_adjusted_economic_capture_score_top10_v4",
                "train_include_clean_economic_capture_score_top10_v5",
            ]
            if col in out.columns
        },
    }


def _label_variant_metric_row(
    frame: pd.DataFrame,
    *,
    label_col: str,
    scope: str,
    bucket: str,
) -> dict[str, Any]:
    labels = frame[label_col]
    labeled = frame[labels.isin([0, 1]) & frame["realized_net_utility"].notna()]
    good = labeled[labeled[label_col].eq(1)]
    bad = labeled[labeled[label_col].eq(0)]
    return {
        "label_col": label_col,
        "scope": scope,
        "bucket": bucket,
        "outcome_rows": int(frame["realized_net_utility"].notna().sum()),
        "labeled_rows": int(len(labeled)),
        "positive_rows": int(len(good)),
        "negative_rows": int(len(bad)),
        "positive_rate": float(len(good) / len(labeled)) if len(labeled) else float("nan"),
        "positive_mean_utility": _safe_mean(good["realized_net_utility"]),
        "negative_mean_utility": _safe_mean(bad["realized_net_utility"]),
        "positive_bad_mae_rate": _safe_mean(good["bad_mae_flag"] > 0.5),
        "negative_bad_mae_rate": _safe_mean(bad["bad_mae_flag"] > 0.5),
        "positive_p90_mae": _safe_quantile(good["adverse_mae_norm"], 0.90),
        "negative_p90_mae": _safe_quantile(bad["adverse_mae_norm"], 0.90),
        "positive_timeout_rate": _safe_mean(good["timeout_flag"] > 0.5),
        "negative_timeout_rate": _safe_mean(bad["timeout_flag"] > 0.5),
    }


def evaluate_quality_label_variants(
    label_candidates: pd.DataFrame,
    outcomes: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    timestamp_col = str(config.get("timestamp_col") or "__ts__")
    cols = [timestamp_col, "primary_source_tag"] + [col for col in QUALITY_LABEL_VARIANTS if col in label_candidates.columns]
    frame = pd.concat([label_candidates.loc[:, list(dict.fromkeys(cols))], outcomes], axis=1)
    frame["month"] = _period_series(frame[timestamp_col], "M")
    frame["week"] = _period_series(frame[timestamp_col], "W-SUN")
    label_cols = [col for col in QUALITY_LABEL_VARIANTS if col in frame.columns]
    overall_rows: list[dict[str, Any]] = []
    month_rows: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []
    for label_col in label_cols:
        overall_rows.append(_label_variant_metric_row(frame, label_col=label_col, scope="overall", bucket="all"))
        for month, group in frame.groupby("month", dropna=False):
            month_rows.append(_label_variant_metric_row(group, label_col=label_col, scope="month", bucket=str(month)))
        for source_tag, group in frame.groupby("primary_source_tag", dropna=False):
            source_rows.append(_label_variant_metric_row(group, label_col=label_col, scope="primary_source_tag", bucket=str(source_tag)))
    return pd.DataFrame(overall_rows), pd.DataFrame(month_rows), pd.DataFrame(source_rows)


def build_row_alignment_audit(
    *,
    frame: pd.DataFrame,
    source: pd.DataFrame,
    label_candidates: pd.DataFrame,
    outcomes: pd.DataFrame,
    config: dict[str, Any],
    label_join_report: dict[str, Any],
    prediction_report: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    timestamp_col = str(config.get("timestamp_col") or "__ts__")
    symbol_col = str(config.get("symbol_col") or "__symbol__")
    candidate_id_col = config.get("candidate_id_col")
    side_col = config.get("side_col")
    proxy_cols = [col for col in _configured_proxy_score_columns(config) if col in source.columns]
    regime_cols = [col for col in config.get("regime_head_columns") or [] if col in source.columns]
    rows = {
        "feature_input_rows": int(len(frame)),
        "candidate_source_tags_rows": int(len(source)),
        "quality_label_candidates_rows": int(len(label_candidates)),
        "outcome_rows_matched": int(outcomes["realized_net_utility"].notna().sum())
        if "realized_net_utility" in outcomes.columns
        else 0,
        "outcome_match_rate": float(outcomes["realized_net_utility"].notna().mean())
        if "realized_net_utility" in outcomes.columns and len(outcomes)
        else 0.0,
        "prediction_rows": int(prediction_report.get("prediction_rows", 0)),
        "prediction_match_rate": float(prediction_report.get("prediction_match_rate", 0.0)),
        "duplicate_candidate_id_rows": _duplicate_key_count(source, [str(candidate_id_col)])
        if candidate_id_col and str(candidate_id_col) in source.columns
        else 0,
        "duplicate_timestamp_symbol_rows": _duplicate_key_count(source, [timestamp_col, symbol_col]),
        "duplicate_timestamp_symbol_side_rows": _duplicate_key_count(source, [timestamp_col, symbol_col, str(side_col)])
        if side_col and str(side_col) in source.columns
        else 0,
        "rows_with_missing_proxy_score": int((~source[proxy_cols].notna().any(axis=1)).sum()) if proxy_cols else int(len(source)),
        "rows_with_missing_regime": int(source[regime_cols].isna().any(axis=1).sum()) if regime_cols else 0,
        "rows_with_missing_outcome": int(outcomes["realized_net_utility"].isna().sum())
        if "realized_net_utility" in outcomes.columns
        else int(len(outcomes)),
        "rows_with_multiple_outcomes_joined": int(label_join_report.get("rows_with_multiple_outcomes_joined", 0)),
        "rows_with_multiple_predictions_joined": int(prediction_report.get("rows_with_multiple_predictions_joined", 0)),
        "label_duplicate_keys": int(label_join_report.get("label_duplicate_keys", 0)),
        "prediction_duplicate_keys": int(prediction_report.get("prediction_duplicate_keys", 0)),
        "metadata_columns_preserved": int(
            all(col in label_candidates.columns for col in _configured_metadata_columns(config, source))
        ),
    }
    status = "pass"
    warnings: list[str] = []
    if candidate_id_col and rows["duplicate_candidate_id_rows"] > 0:
        status = "fail"
        warnings.append("duplicate_candidate_id")
    if rows["rows_with_multiple_outcomes_joined"] > 0 or rows["label_duplicate_keys"] > 0:
        status = "fail"
        warnings.append("multiple_outcomes")
    if rows["rows_with_multiple_predictions_joined"] > 0 or rows["prediction_duplicate_keys"] > 0:
        status = "fail"
        warnings.append("multiple_predictions")
    if len(label_candidates) != len(source) or len(source) != len(frame):
        status = "fail"
        warnings.append("row_count_changed")
    if not rows["metadata_columns_preserved"]:
        status = "fail"
        warnings.append("metadata_dropped")
    pred_status = str(prediction_report.get("alignment_status", "pass"))
    if pred_status == "fail":
        status = "fail"
        warnings.append("prediction_alignment_fail")
    elif pred_status == "warning" and status == "pass":
        status = "warning"
        warnings.append("prediction_alignment_warning")
    rows["alignment_quality"] = status
    rows["alignment_warnings"] = ",".join(warnings)
    audit = pd.DataFrame([rows])
    return audit, rows


def _pearson(x: Any, y: Any) -> float:
    xs = _safe_numeric(x)
    ys = _safe_numeric(y)
    mask = xs.notna() & ys.notna()
    if int(mask.sum()) < 5:
        return float("nan")
    if xs.loc[mask].nunique(dropna=True) < 2 or ys.loc[mask].nunique(dropna=True) < 2:
        return float("nan")
    return float(xs.loc[mask].corr(ys.loc[mask]))


def _proxy_learnability_row(
    group: pd.DataFrame,
    *,
    scope: str,
    bucket: str,
    source_tag: str,
    source_scope: str,
    proxy_col: str,
    top_frac: float,
    total_rows: int,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    required = [proxy_col, "realized_net_utility"]
    valid = group.dropna(subset=[col for col in required if col in group.columns]).copy()
    row = {
        "scope": scope,
        "bucket": bucket,
        "source_tag": source_tag,
        "source_scope": source_scope,
        "proxy_col": proxy_col,
        "top_frac": float(top_frac),
        "rows": int(len(group)),
        "coverage_pct": float(len(group) / total_rows) if total_rows else 0.0,
        "valid_rows": int(len(valid)),
    }
    if extra:
        row.update(extra)
    if len(valid) < 5:
        row.update(
            {
                "proxy_ic_spearman": float("nan"),
                "proxy_ic_pearson": float("nan"),
                "proxy_topk_rows": 0,
                "proxy_topk_mean_utility": float("nan"),
                "proxy_topk_bad_mae_rate": float("nan"),
                "proxy_topk_timeout_rate": float("nan"),
                "proxy_topk_p90_mae": float("nan"),
                "proxy_topk_economic_capture_good_rate": float("nan"),
                "oracle_topk_mean_utility": float("nan"),
                "proxy_oracle_capture_ratio": float("nan"),
                "capture_at_k": float("nan"),
                "bad_mae_contamination_rate": float("nan"),
            }
        )
        for label_col in QUALITY_LABEL_VARIANTS:
            row[f"precision_at_k_{label_col}"] = float("nan")
            row[f"recall_at_k_{label_col}"] = float("nan")
        return row
    k = max(1, int(math.ceil(float(top_frac) * len(valid))))
    proxy_top = valid.sort_values(proxy_col, ascending=False, kind="mergesort").head(k)
    oracle_top = valid.sort_values("realized_net_utility", ascending=False, kind="mergesort").head(k)
    oracle_mean = _safe_mean(oracle_top["realized_net_utility"])
    proxy_mean = _safe_mean(proxy_top["realized_net_utility"])
    oracle_top_idx = set(oracle_top.index)
    proxy_top_idx = set(proxy_top.index)
    row.update(
        {
            "proxy_ic_spearman": _spearman(valid[proxy_col], valid["realized_net_utility"]),
            "proxy_ic_pearson": _pearson(valid[proxy_col], valid["realized_net_utility"]),
            "proxy_topk_rows": int(len(proxy_top)),
            "proxy_topk_mean_utility": proxy_mean,
            "proxy_topk_bad_mae_rate": _safe_mean(proxy_top["bad_mae_flag"] > 0.5)
            if "bad_mae_flag" in proxy_top.columns
            else float("nan"),
            "proxy_topk_timeout_rate": _safe_mean(proxy_top["timeout_flag"] > 0.5)
            if "timeout_flag" in proxy_top.columns
            else float("nan"),
            "proxy_topk_p90_mae": _safe_quantile(proxy_top["adverse_mae_norm"], 0.90)
            if "adverse_mae_norm" in proxy_top.columns
            else float("nan"),
            "proxy_topk_economic_capture_good_rate": _safe_mean(proxy_top["outcome_economic_capture_flag"] > 0.5)
            if "outcome_economic_capture_flag" in proxy_top.columns
            else float("nan"),
            "oracle_topk_mean_utility": oracle_mean,
            "proxy_oracle_capture_ratio": float(proxy_mean / oracle_mean)
            if math.isfinite(float(proxy_mean)) and math.isfinite(float(oracle_mean)) and abs(float(oracle_mean)) > 1e-12
            else float("nan"),
            "capture_at_k": float(len(oracle_top_idx & proxy_top_idx) / max(1, len(oracle_top_idx))),
            "bad_mae_contamination_rate": _safe_mean(proxy_top["bad_mae_flag"] > 0.5)
            if "bad_mae_flag" in proxy_top.columns
            else float("nan"),
        }
    )
    for label_col in QUALITY_LABEL_VARIANTS:
        if label_col in valid.columns:
            positives = valid[valid[label_col].eq(1)]
            row[f"precision_at_k_{label_col}"] = _safe_mean(proxy_top[label_col].eq(1))
            row[f"recall_at_k_{label_col}"] = (
                float(proxy_top[label_col].eq(1).sum() / len(positives)) if len(positives) else float("nan")
            )
        else:
            row[f"precision_at_k_{label_col}"] = float("nan")
            row[f"recall_at_k_{label_col}"] = float("nan")
    return row


def evaluate_proxy_learnability(
    frame: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    timestamp_col = str(config.get("timestamp_col") or "__ts__")
    diagnostics = config.get("diagnostics") or {}
    top_fracs = [float(v) for v in diagnostics.get("proxy_top_fracs", [0.01, 0.05, 0.10])]
    proxy_cols = [col for col in _configured_proxy_score_columns(config) if col in frame.columns]
    proxy_cols = [col for col in proxy_cols if not is_forbidden_proxy_column(col, config)]
    eval_frame = frame.loc[frame["realized_net_utility"].notna()].copy() if "realized_net_utility" in frame.columns else pd.DataFrame()
    if eval_frame.empty or not proxy_cols:
        empty = pd.DataFrame()
        return empty, empty, empty, empty, {
            "proxy_score_columns": proxy_cols,
            "proxy_available": bool(proxy_cols),
            "outcome_eval_rows": int(len(eval_frame)),
        }
    eval_frame["month"] = _period_series(eval_frame[timestamp_col], "M")
    eval_frame["week"] = _period_series(eval_frame[timestamp_col], "W-SUN")
    total = int(len(eval_frame))
    rows: list[dict[str, Any]] = []

    def add_source_rows(group: pd.DataFrame, scope: str, bucket: str, *, total_rows: int, extra: dict[str, Any] | None = None) -> None:
        for proxy_col in proxy_cols:
            for top_frac in top_fracs:
                for tag_col in TAG_COLS:
                    if tag_col in group.columns:
                        selected = group[group[tag_col].astype(bool)]
                        if len(selected):
                            rows.append(
                                _proxy_learnability_row(
                                    selected,
                                    scope=scope,
                                    bucket=bucket,
                                    source_tag=tag_col.replace("tag_", ""),
                                    source_scope="multi_tag",
                                    proxy_col=proxy_col,
                                    top_frac=top_frac,
                                    total_rows=total_rows,
                                    extra=extra,
                                )
                            )
                if "primary_source_tag" in group.columns:
                    for primary, selected in group.groupby("primary_source_tag", dropna=False):
                        rows.append(
                            _proxy_learnability_row(
                                selected,
                                scope=scope,
                                bucket=bucket,
                                source_tag=str(primary),
                                source_scope="primary_source_tag",
                                proxy_col=proxy_col,
                                top_frac=top_frac,
                                total_rows=total_rows,
                                extra=extra,
                            )
                        )

    add_source_rows(eval_frame, "overall", "all", total_rows=total)
    overall = pd.DataFrame(rows)
    month_rows: list[dict[str, Any]] = []
    rows = month_rows
    for month, group in eval_frame.groupby("month", dropna=False):
        add_source_rows(group, "month", str(month), total_rows=int(len(group)))
    by_month = pd.DataFrame(month_rows)
    week_rows: list[dict[str, Any]] = []
    rows = week_rows
    for week, group in eval_frame.groupby("week", dropna=False):
        add_source_rows(group, "week", str(week), total_rows=int(len(group)))
    by_week = pd.DataFrame(week_rows)

    regime_rows: list[dict[str, Any]] = []
    rows = regime_rows
    regime_cols = [col for col in config.get("regime_head_columns") or [] if col in eval_frame.columns]
    overall_concentration: dict[str, float] = {}
    for tag_col in TAG_COLS:
        if tag_col in eval_frame.columns:
            overall_concentration[tag_col.replace("tag_", "")] = float(eval_frame[tag_col].astype(bool).mean())
    for regime_col in regime_cols:
        for regime_value, regime_group in eval_frame.groupby(regime_col, dropna=False):
            regime_total = int(len(regime_group))
            for proxy_col in proxy_cols:
                for top_frac in top_fracs:
                    for tag_col in TAG_COLS:
                        if tag_col not in regime_group.columns:
                            continue
                        selected = regime_group[regime_group[tag_col].astype(bool)]
                        if selected.empty:
                            continue
                        source_tag = tag_col.replace("tag_", "")
                        concentration = float(len(selected) / regime_total) if regime_total else 0.0
                        overall_rate = overall_concentration.get(source_tag, float("nan"))
                        extra = {
                            "regime_col": str(regime_col),
                            "regime_value": str(regime_value),
                            "source_concentration_within_regime": concentration,
                            "source_lift_vs_overall": float(concentration / overall_rate)
                            if math.isfinite(float(overall_rate)) and float(overall_rate) > 0.0
                            else float("nan"),
                        }
                        rows.append(
                            _proxy_learnability_row(
                                selected,
                                scope="source_x_regime",
                                bucket=f"{regime_col}={regime_value}",
                                source_tag=source_tag,
                                source_scope="multi_tag",
                                proxy_col=proxy_col,
                                top_frac=top_frac,
                                total_rows=regime_total,
                                extra=extra,
                            )
                        )
    source_x_regime = pd.DataFrame(regime_rows)
    return overall, by_month, by_week, source_x_regime, {
        "proxy_score_columns": proxy_cols,
        "proxy_available": True,
        "outcome_eval_rows": total,
        "top_fracs": top_fracs,
        "regime_columns_present": regime_cols,
    }


def _failure_mode_metric_row(
    group: pd.DataFrame,
    *,
    scope: str,
    bucket: str,
    total_rows: int,
) -> dict[str, Any]:
    outcome_rows = group[group["realized_net_utility"].notna()]
    row = {
        "scope": scope,
        "bucket": bucket,
        "rows": int(len(outcome_rows)),
        "coverage_pct": float(len(outcome_rows) / total_rows) if total_rows else 0.0,
        "mean_net_utility": _safe_mean(outcome_rows["realized_net_utility"]),
        "median_net_utility": _safe_quantile(outcome_rows["realized_net_utility"], 0.50),
        "p25_net_utility": _safe_quantile(outcome_rows["realized_net_utility"], 0.25),
        "p90_mae": _safe_quantile(outcome_rows["adverse_mae_norm"], 0.90),
        "p90_mfe": _safe_quantile(outcome_rows["favorable_mfe_norm"], 0.90),
        "mean_recovery_ratio": _safe_mean(outcome_rows["mfe_mae_recovery_ratio"]),
    }
    for col in FAILURE_MODE_COLS:
        if col in outcome_rows.columns:
            row[col.replace("outcome_", "").replace("_flag", "_rate")] = _safe_mean(outcome_rows[col] > 0.5)
    return row


def evaluate_failure_modes(full: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    timestamp_col = str(config.get("timestamp_col") or "__ts__")
    eval_frame = full.loc[full["realized_net_utility"].notna()].copy()
    if eval_frame.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    eval_frame["month"] = _period_series(eval_frame[timestamp_col], "M")
    total = int(len(eval_frame))
    by_month = pd.DataFrame(
        [
            _failure_mode_metric_row(group, scope="month", bucket=str(month), total_rows=total)
            for month, group in eval_frame.groupby("month", dropna=False)
        ]
    )
    by_source = pd.DataFrame(
        [
            _failure_mode_metric_row(group, scope="primary_source_tag", bucket=str(source_tag), total_rows=total)
            for source_tag, group in eval_frame.groupby("primary_source_tag", dropna=False)
        ]
    )
    by_source_month = pd.DataFrame(
        [
            _failure_mode_metric_row(group, scope="primary_source_tag_month", bucket=f"{source_tag}|{month}", total_rows=total)
            for (source_tag, month), group in eval_frame.groupby(["primary_source_tag", "month"], dropna=False)
        ]
    )
    return by_source, by_month, by_source_month


def _opportunity_capture_metric_row(
    group: pd.DataFrame,
    *,
    scope: str,
    bucket: str,
    total_rows: int,
) -> dict[str, Any]:
    outcome_rows = group[group["realized_net_utility"].notna()]
    opportunities = outcome_rows[outcome_rows["outcome_recoverable_opportunity_flag"].fillna(0.0).gt(0.5)]
    captured = opportunities[opportunities["outcome_opportunity_captured_flag"].fillna(0.0).gt(0.5)]
    economic_capture = opportunities[opportunities["outcome_economic_capture_flag"].fillna(0.0).gt(0.5)]
    expensive_capture = opportunities[opportunities["outcome_expensive_capture_flag"].fillna(0.0).gt(0.5)]
    capture_loss = opportunities[opportunities["outcome_opportunity_capture_loss_flag"].fillna(0.0).gt(0.5)]
    return {
        "scope": scope,
        "bucket": bucket,
        "outcome_rows": int(len(outcome_rows)),
        "opportunity_rows": int(len(opportunities)),
        "opportunity_coverage_pct": float(len(opportunities) / total_rows) if total_rows else 0.0,
        "capture_rate": float(len(captured) / len(opportunities)) if len(opportunities) else float("nan"),
        "economic_capture_rate": float(len(economic_capture) / len(opportunities)) if len(opportunities) else float("nan"),
        "expensive_capture_rate": float(len(expensive_capture) / len(opportunities)) if len(opportunities) else float("nan"),
        "capture_loss_rate": float(len(capture_loss) / len(opportunities)) if len(opportunities) else float("nan"),
        "missed_opportunity_rate": _safe_mean(opportunities["outcome_missed_opportunity_flag"] > 0.5),
        "clean_win_rate": _safe_mean(opportunities["outcome_clean_win_flag"] > 0.5),
        "positive_utility_rate": _safe_mean(opportunities["outcome_positive_utility_flag"] > 0.5),
        "mean_net_utility": _safe_mean(opportunities["realized_net_utility"]),
        "captured_mean_utility": _safe_mean(captured["realized_net_utility"]),
        "economic_capture_mean_utility": _safe_mean(economic_capture["realized_net_utility"]),
        "expensive_capture_mean_utility": _safe_mean(expensive_capture["realized_net_utility"]),
        "loss_mean_utility": _safe_mean(capture_loss["realized_net_utility"]),
        "mean_capture_efficiency": _safe_mean(opportunities["opportunity_capture_efficiency"]),
        "p25_capture_efficiency": _safe_quantile(opportunities["opportunity_capture_efficiency"], 0.25),
        "p75_capture_efficiency": _safe_quantile(opportunities["opportunity_capture_efficiency"], 0.75),
        "p90_mae": _safe_quantile(opportunities["adverse_mae_norm"], 0.90),
        "p90_mfe": _safe_quantile(opportunities["favorable_mfe_norm"], 0.90),
    }


def evaluate_opportunity_capture(full: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    timestamp_col = str(config.get("timestamp_col") or "__ts__")
    eval_frame = full.loc[full["realized_net_utility"].notna()].copy()
    if eval_frame.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    eval_frame["month"] = _period_series(eval_frame[timestamp_col], "M")
    total = int(len(eval_frame))
    by_source = pd.DataFrame(
        [
            _opportunity_capture_metric_row(group, scope="primary_source_tag", bucket=str(source_tag), total_rows=total)
            for source_tag, group in eval_frame.groupby("primary_source_tag", dropna=False)
        ]
    )
    by_month = pd.DataFrame(
        [
            _opportunity_capture_metric_row(group, scope="month", bucket=str(month), total_rows=total)
            for month, group in eval_frame.groupby("month", dropna=False)
        ]
    )
    by_source_month = pd.DataFrame(
        [
            _opportunity_capture_metric_row(group, scope="primary_source_tag_month", bucket=f"{source_tag}|{month}", total_rows=total)
            for (source_tag, month), group in eval_frame.groupby(["primary_source_tag", "month"], dropna=False)
        ]
    )
    return by_source, by_month, by_source_month


def _capture_utility_gap_row(
    group: pd.DataFrame,
    *,
    scope: str,
    bucket: str,
    score_col: str,
    top_frac: float,
) -> dict[str, Any]:
    score = _safe_numeric(group[score_col])
    mask = score.notna() & group["realized_net_utility"].notna()
    valid = group.loc[mask].copy()
    if valid.empty:
        return {
            "scope": scope,
            "bucket": bucket,
            "score_col": score_col,
            "top_frac": float(top_frac),
            "valid_rows": 0,
            "selected_rows": 0,
        }
    order = _safe_numeric(valid[score_col]).sort_values(ascending=False, kind="mergesort")
    k = max(1, int(math.ceil(float(top_frac) * len(order))))
    selected = valid.loc[order.index[:k]]
    opportunities = selected[selected["outcome_recoverable_opportunity_flag"].fillna(0.0).gt(0.5)]
    captured = opportunities[opportunities["outcome_opportunity_captured_flag"].fillna(0.0).gt(0.5)]
    economic_capture = opportunities[opportunities["outcome_economic_capture_flag"].fillna(0.0).gt(0.5)]
    expensive_capture = opportunities[opportunities["outcome_expensive_capture_flag"].fillna(0.0).gt(0.5)]
    capture_loss = opportunities[opportunities["outcome_economic_capture_loss_flag"].fillna(0.0).gt(0.5)]
    no_edge = selected[selected["outcome_no_edge_flag"].fillna(0.0).gt(0.5)]
    path_failure = selected[selected["outcome_path_failure_flag"].fillna(0.0).gt(0.5)]
    return {
        "scope": scope,
        "bucket": bucket,
        "score_col": score_col,
        "top_frac": float(top_frac),
        "valid_rows": int(len(valid)),
        "selected_rows": int(len(selected)),
        "selected_mean_utility": _safe_mean(selected["realized_net_utility"]),
        "selected_positive_utility_rate": _safe_mean(selected["realized_net_utility"] > 0.0),
        "selected_bad_mae_rate": _safe_mean(selected["bad_mae_flag"] > 0.5),
        "selected_p90_mae": _safe_quantile(selected["adverse_mae_norm"], 0.90),
        "selected_opportunity_rate": _safe_mean(selected["outcome_recoverable_opportunity_flag"] > 0.5),
        "selected_capture_rate": _safe_mean(selected["outcome_opportunity_captured_flag"] > 0.5),
        "selected_economic_capture_rate": _safe_mean(selected["outcome_economic_capture_flag"] > 0.5),
        "selected_expensive_capture_rate": _safe_mean(selected["outcome_expensive_capture_flag"] > 0.5),
        "selected_economic_capture_loss_rate": _safe_mean(selected["outcome_economic_capture_loss_flag"] > 0.5),
        "capture_among_opportunity_rate": float(len(captured) / len(opportunities)) if len(opportunities) else float("nan"),
        "economic_capture_among_opportunity_rate": float(len(economic_capture) / len(opportunities)) if len(opportunities) else float("nan"),
        "expensive_capture_among_opportunity_rate": float(len(expensive_capture) / len(opportunities)) if len(opportunities) else float("nan"),
        "economic_capture_loss_among_opportunity_rate": float(len(capture_loss) / len(opportunities)) if len(opportunities) else float("nan"),
        "captured_mean_utility": _safe_mean(captured["realized_net_utility"]),
        "economic_capture_mean_utility": _safe_mean(economic_capture["realized_net_utility"]),
        "expensive_capture_mean_utility": _safe_mean(expensive_capture["realized_net_utility"]),
        "economic_capture_loss_mean_utility": _safe_mean(capture_loss["realized_net_utility"]),
        "no_edge_rate": _safe_mean(selected["outcome_no_edge_flag"] > 0.5),
        "path_failure_rate": _safe_mean(selected["outcome_path_failure_flag"] > 0.5),
        "no_edge_mean_utility": _safe_mean(no_edge["realized_net_utility"]),
        "path_failure_mean_utility": _safe_mean(path_failure["realized_net_utility"]),
        "mean_capture_efficiency": _safe_mean(opportunities["opportunity_capture_efficiency"]),
    }


def evaluate_capture_utility_gap(full: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    timestamp_col = str(config.get("timestamp_col") or "__ts__")
    diagnostics = config.get("diagnostics") or {}
    top_fracs = [float(v) for v in diagnostics.get("source_score_top_fracs", [0.01, 0.03, 0.05, 0.10])]
    eval_frame = full.loc[full["realized_net_utility"].notna()].copy()
    if eval_frame.empty:
        return pd.DataFrame(), pd.DataFrame()
    eval_frame["month"] = _period_series(eval_frame[timestamp_col], "M")
    score_cols = [col for col in SOURCE_SCORE_EVAL_COLS if col in eval_frame.columns]
    rows: list[dict[str, Any]] = []
    for score_col in score_cols:
        for top_frac in top_fracs:
            rows.append(
                _capture_utility_gap_row(
                    eval_frame,
                    scope="overall",
                    bucket="all",
                    score_col=score_col,
                    top_frac=top_frac,
                )
            )
    month_rows: list[dict[str, Any]] = []
    for month, group in eval_frame.groupby("month", dropna=False):
        for score_col in score_cols:
            for top_frac in top_fracs:
                month_rows.append(
                    _capture_utility_gap_row(
                        group,
                        scope="month",
                        bucket=str(month),
                        score_col=score_col,
                        top_frac=top_frac,
                    )
                )
    return pd.DataFrame(rows), pd.DataFrame(month_rows)


def _metric_from_target_diag(
    diagnostics: pd.DataFrame,
    *,
    score_col: str,
    target_col: str,
    top_frac: float,
    metric_col: str,
) -> float:
    if diagnostics.empty or metric_col not in diagnostics.columns:
        return float("nan")
    subset = diagnostics[
        diagnostics["score_col"].eq(score_col)
        & diagnostics["target_col"].eq(target_col)
        & diagnostics["scope"].eq("overall")
    ].copy()
    if subset.empty:
        return float("nan")
    subset["_frac_distance"] = (_safe_numeric(subset["top_frac"]) - float(top_frac)).abs()
    row = subset.sort_values(["_frac_distance", "top_frac"], ascending=[True, False]).iloc[0]
    return float(row[metric_col]) if pd.notna(row[metric_col]) else float("nan")


def _metric_from_score_slices(
    score_slices: pd.DataFrame,
    *,
    score_col: str,
    top_frac: float,
    metric_col: str,
) -> float:
    if score_slices.empty or metric_col not in score_slices.columns:
        return float("nan")
    subset = score_slices[score_slices["score_col"].eq(score_col) & score_slices["scope"].eq("overall")].copy()
    if subset.empty:
        return float("nan")
    subset["_frac_distance"] = (_safe_numeric(subset["top_frac"]) - float(top_frac)).abs()
    row = subset.sort_values(["_frac_distance", "top_frac"], ascending=[True, False]).iloc[0]
    return float(row[metric_col]) if pd.notna(row[metric_col]) else float("nan")


def _monthly_metric_stats(
    diagnostics: pd.DataFrame | None,
    *,
    score_col: str,
    top_frac: float,
    metric_col: str,
    target_col: str | None = None,
) -> dict[str, float]:
    if diagnostics is None or diagnostics.empty or metric_col not in diagnostics.columns:
        return {
            "count": 0.0,
            "mean": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "positive_rate": float("nan"),
            "nonpositive_rate": float("nan"),
            "negative_rate": float("nan"),
        }
    subset = diagnostics[
        diagnostics["score_col"].eq(score_col)
        & diagnostics["scope"].eq("month")
    ].copy()
    if target_col is not None:
        subset = subset[subset["target_col"].eq(target_col)]
    if subset.empty:
        return {
            "count": 0.0,
            "mean": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "positive_rate": float("nan"),
            "nonpositive_rate": float("nan"),
            "negative_rate": float("nan"),
        }
    subset["_frac_distance"] = (_safe_numeric(subset["top_frac"]) - float(top_frac)).abs()
    subset = (
        subset.sort_values(["bucket", "_frac_distance", "top_frac"], ascending=[True, True, False])
        .groupby("bucket", sort=False, dropna=False)
        .head(1)
    )
    values = _safe_numeric(subset[metric_col]).dropna()
    if values.empty:
        return {
            "count": 0.0,
            "mean": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "positive_rate": float("nan"),
            "nonpositive_rate": float("nan"),
            "negative_rate": float("nan"),
        }
    return {
        "count": float(len(values)),
        "mean": _safe_mean(values),
        "min": _safe_quantile(values, 0.0),
        "max": _safe_quantile(values, 1.0),
        "positive_rate": _safe_mean(values > 0.0),
        "nonpositive_rate": _safe_mean(values <= 0.0),
        "negative_rate": _safe_mean(values < 0.0),
    }


def _promotion_stability(
    *,
    score_kind: str,
    bucket: str,
    capture_month: dict[str, float],
    recoverable_month: dict[str, float],
    path_failure_month: dict[str, float],
    no_edge_month: dict[str, float],
    capture_loss_month: dict[str, float],
    utility_month: dict[str, float],
    stable_rate: float,
    mixed_rate: float,
    high_confidence_utility_rate: float,
) -> tuple[str, str]:
    if score_kind == "component":
        return "component_input", "diagnostic_only"
    utility_positive_rate = float(utility_month.get("positive_rate", float("nan")))
    if bucket == "risk_avoidance":
        risk_rates = [
            float(path_failure_month.get("positive_rate", float("nan"))),
            float(no_edge_month.get("positive_rate", float("nan"))),
            float(capture_loss_month.get("positive_rate", float("nan"))),
        ]
        finite_risk_rates = [rate for rate in risk_rates if math.isfinite(rate)]
        risk_rate = max(finite_risk_rates) if finite_risk_rates else float("nan")
        if math.isfinite(risk_rate) and risk_rate >= stable_rate:
            return "stable_risk_signal", "high"
        if math.isfinite(risk_rate) and risk_rate >= mixed_rate:
            return "mixed_risk_signal", "medium"
        return "weak_risk_signal", "low"
    if bucket in {"safer_opportunity", "capture_maximizer", "opportunity_finder", "utility_positive_slice"}:
        if bucket == "safer_opportunity":
            signal_rates = [
                float(capture_month.get("positive_rate", float("nan"))),
                float(recoverable_month.get("positive_rate", float("nan"))),
            ]
            safety_rates = [
                float(path_failure_month.get("nonpositive_rate", float("nan"))),
                float(no_edge_month.get("nonpositive_rate", float("nan"))),
            ]
            finite_signal_rates = [rate for rate in signal_rates if math.isfinite(rate)]
            finite_safety_rates = [rate for rate in safety_rates if math.isfinite(rate)]
            signal_rate = min(finite_signal_rates) if finite_signal_rates else float("nan")
            safety_rate = min(finite_safety_rates) if finite_safety_rates else float("nan")
            stable = (
                math.isfinite(signal_rate)
                and signal_rate >= stable_rate
                and math.isfinite(safety_rate)
                and safety_rate >= stable_rate
            )
            mixed = (
                math.isfinite(signal_rate)
                and signal_rate >= mixed_rate
                and (not math.isfinite(safety_rate) or safety_rate >= mixed_rate)
            )
        elif bucket == "capture_maximizer":
            signal_rate = float(capture_month.get("positive_rate", float("nan")))
            stable = math.isfinite(signal_rate) and signal_rate >= stable_rate
            mixed = math.isfinite(signal_rate) and signal_rate >= mixed_rate
        elif bucket == "opportunity_finder":
            signal_rate = float(recoverable_month.get("positive_rate", float("nan")))
            stable = math.isfinite(signal_rate) and signal_rate >= stable_rate
            mixed = math.isfinite(signal_rate) and signal_rate >= mixed_rate
        else:
            signal_rate = utility_positive_rate
            stable = math.isfinite(signal_rate) and signal_rate >= stable_rate
            mixed = math.isfinite(signal_rate) and signal_rate >= mixed_rate
        if stable:
            confidence = "high" if math.isfinite(utility_positive_rate) and utility_positive_rate >= high_confidence_utility_rate else "medium"
            return "stable_promoted_signal", confidence
        if mixed:
            return "mixed_promoted_signal", "medium"
        return "unstable_promoted_signal", "low"
    return "not_promoted", "diagnostic_only"


def _classify_promotion_from_metrics(
    *,
    score_kind: str,
    utility_top: float,
    utility_ic: float,
    recoverable_delta: float,
    recoverable_ic: float,
    capture_delta: float,
    capture_ic: float,
    capture_efficiency_delta: float,
    path_failure_delta: float,
    no_edge_delta: float,
    capture_loss_delta: float,
) -> tuple[str, str, str]:
    capture_signal = (
        math.isfinite(capture_delta)
        and capture_delta >= 0.03
        and (not math.isfinite(capture_ic) or capture_ic >= 0.02)
    )
    recoverable_signal = (
        math.isfinite(recoverable_delta)
        and recoverable_delta >= 0.04
        and (not math.isfinite(recoverable_ic) or recoverable_ic >= 0.02)
    )
    efficiency_signal = math.isfinite(capture_efficiency_delta) and capture_efficiency_delta >= 0.03
    path_risk_signal = (
        (math.isfinite(path_failure_delta) and path_failure_delta >= 0.05)
        or (math.isfinite(no_edge_delta) and no_edge_delta >= 0.04)
        or (math.isfinite(capture_loss_delta) and capture_loss_delta >= 0.04)
    )
    safer_signal = (
        (capture_signal or recoverable_signal)
        and math.isfinite(path_failure_delta)
        and path_failure_delta <= -0.02
        and (not math.isfinite(no_edge_delta) or no_edge_delta <= 0.02)
    )

    if score_kind == "component":
        if path_risk_signal:
            return (
                "risk_component",
                "use_as_guard_or_archetype_input_not_standalone_label",
                "Raw component score isolates risk; use it to refine causal archetypes or guards, not as a standalone promoted source.",
            )
        if capture_signal or efficiency_signal or recoverable_signal:
            return (
                "component_signal",
                "use_to_refine_archetype_not_standalone_ablation",
                "Raw component score separates opportunity/capture; fold it into an archetype before model-training ablation.",
            )
        return (
            "component_diagnostic",
            "keep_as_archetype_input_only",
            "Raw component score is useful for source construction diagnostics but is not a promoted standalone source.",
        )
    if path_risk_signal and not capture_signal:
        return (
            "risk_avoidance",
            "use_as_exclusion_or_downweight_flag",
            "High scores isolate path-risk/no-edge rows more than profitable opportunity.",
        )
    if safer_signal:
        return (
            "safer_opportunity",
            "promote_as_clean_opportunity_filter_ablation",
            "High scores increase opportunity exposure while reducing path failure.",
        )
    if capture_signal or efficiency_signal:
        return (
            "capture_maximizer",
            "promote_for_capture_label_ablation",
            "High scores improve opportunity capture or capture efficiency, but may not be safest.",
        )
    if recoverable_signal:
        return (
            "opportunity_finder",
            "test_against_recoverable_opportunity_label",
            "High scores find recoverable opportunities, before checking policy capture quality.",
        )
    if path_risk_signal:
        return (
            "risk_avoidance",
            "use_as_exclusion_or_downweight_flag",
            "High scores isolate path-risk/no-edge rows.",
        )
    if math.isfinite(utility_top) and utility_top > 0.0 and (not math.isfinite(utility_ic) or utility_ic > 0.0):
        return (
            "utility_positive_slice",
            "diagnostic_training_candidate",
            "Top score slice has positive mean utility, but target-side reason is not yet clear.",
        )
    return (
        "not_promoted",
        "keep_diagnostic_only",
        "Current diagnostics do not show enough economic or learnability distinction.",
    )


def build_source_archetype_promotion_scorecard(
    *,
    source_score_target_overall: pd.DataFrame,
    source_score_quality_overall: pd.DataFrame,
    config: dict[str, Any],
    source_score_target_by_month: pd.DataFrame | None = None,
    source_score_quality_by_month: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Classify causal source scores for later ablations without changing training."""
    if source_score_target_overall.empty and source_score_quality_overall.empty:
        return pd.DataFrame()
    diagnostics = config.get("diagnostics") or {}
    scorecard_top_frac = float(diagnostics.get("promotion_scorecard_top_frac", 0.10))
    stable_month_rate = float(diagnostics.get("promotion_stable_month_rate", 0.75))
    mixed_month_rate = float(diagnostics.get("promotion_mixed_month_rate", 0.50))
    high_confidence_utility_rate = float(diagnostics.get("promotion_high_confidence_utility_positive_month_rate", 0.50))
    score_cols = list(
        dict.fromkeys(
            list(source_score_target_overall.get("score_col", pd.Series(dtype=str)).dropna().astype(str))
            + list(source_score_quality_overall.get("score_col", pd.Series(dtype=str)).dropna().astype(str))
        )
    )
    rows: list[dict[str, Any]] = []
    for score_col in score_cols:
        score_kind = "component" if score_col in COMPONENT_COLS else "archetype"
        utility_top = _metric_from_score_slices(
            source_score_quality_overall,
            score_col=score_col,
            top_frac=scorecard_top_frac,
            metric_col="mean_net_utility",
        )
        utility_ic = _metric_from_score_slices(
            source_score_quality_overall,
            score_col=score_col,
            top_frac=scorecard_top_frac,
            metric_col="score_ic_utility",
        )
        bad_mae_top = _metric_from_score_slices(
            source_score_quality_overall,
            score_col=score_col,
            top_frac=scorecard_top_frac,
            metric_col="bad_mae_rate",
        )
        p90_mae_top = _metric_from_score_slices(
            source_score_quality_overall,
            score_col=score_col,
            top_frac=scorecard_top_frac,
            metric_col="p90_mae",
        )
        timeout_top = _metric_from_score_slices(
            source_score_quality_overall,
            score_col=score_col,
            top_frac=scorecard_top_frac,
            metric_col="timeout_rate",
        )
        recoverable_delta = _metric_from_target_diag(
            source_score_target_overall,
            score_col=score_col,
            target_col="outcome_recoverable_opportunity_flag",
            top_frac=scorecard_top_frac,
            metric_col="selected_target_delta",
        )
        recoverable_ic = _metric_from_target_diag(
            source_score_target_overall,
            score_col=score_col,
            target_col="outcome_recoverable_opportunity_flag",
            top_frac=scorecard_top_frac,
            metric_col="score_ic_target",
        )
        capture_delta = _metric_from_target_diag(
            source_score_target_overall,
            score_col=score_col,
            target_col="outcome_opportunity_captured_flag",
            top_frac=scorecard_top_frac,
            metric_col="selected_target_delta",
        )
        capture_ic = _metric_from_target_diag(
            source_score_target_overall,
            score_col=score_col,
            target_col="outcome_opportunity_captured_flag",
            top_frac=scorecard_top_frac,
            metric_col="score_ic_target",
        )
        capture_among_opp = _metric_from_target_diag(
            source_score_target_overall,
            score_col=score_col,
            target_col="outcome_opportunity_captured_flag",
            top_frac=scorecard_top_frac,
            metric_col="selected_capture_among_opportunity_rate",
        )
        capture_efficiency_delta = _metric_from_target_diag(
            source_score_target_overall,
            score_col=score_col,
            target_col="opportunity_capture_efficiency",
            top_frac=scorecard_top_frac,
            metric_col="selected_target_delta",
        )
        capture_efficiency_top = _metric_from_target_diag(
            source_score_target_overall,
            score_col=score_col,
            target_col="opportunity_capture_efficiency",
            top_frac=scorecard_top_frac,
            metric_col="selected_target_mean",
        )
        path_failure_delta = _metric_from_target_diag(
            source_score_target_overall,
            score_col=score_col,
            target_col="outcome_path_failure_flag",
            top_frac=scorecard_top_frac,
            metric_col="selected_target_delta",
        )
        path_failure_ic = _metric_from_target_diag(
            source_score_target_overall,
            score_col=score_col,
            target_col="outcome_path_failure_flag",
            top_frac=scorecard_top_frac,
            metric_col="score_ic_target",
        )
        no_edge_delta = _metric_from_target_diag(
            source_score_target_overall,
            score_col=score_col,
            target_col="outcome_no_edge_flag",
            top_frac=scorecard_top_frac,
            metric_col="selected_target_delta",
        )
        capture_loss_delta = _metric_from_target_diag(
            source_score_target_overall,
            score_col=score_col,
            target_col="outcome_opportunity_capture_loss_flag",
            top_frac=scorecard_top_frac,
            metric_col="selected_target_delta",
        )
        utility_month = _monthly_metric_stats(
            source_score_quality_by_month,
            score_col=score_col,
            top_frac=scorecard_top_frac,
            metric_col="mean_net_utility",
        )
        capture_month = _monthly_metric_stats(
            source_score_target_by_month,
            score_col=score_col,
            target_col="outcome_opportunity_captured_flag",
            top_frac=scorecard_top_frac,
            metric_col="selected_target_delta",
        )
        recoverable_month = _monthly_metric_stats(
            source_score_target_by_month,
            score_col=score_col,
            target_col="outcome_recoverable_opportunity_flag",
            top_frac=scorecard_top_frac,
            metric_col="selected_target_delta",
        )
        capture_efficiency_month = _monthly_metric_stats(
            source_score_target_by_month,
            score_col=score_col,
            target_col="opportunity_capture_efficiency",
            top_frac=scorecard_top_frac,
            metric_col="selected_target_delta",
        )
        path_failure_month = _monthly_metric_stats(
            source_score_target_by_month,
            score_col=score_col,
            target_col="outcome_path_failure_flag",
            top_frac=scorecard_top_frac,
            metric_col="selected_target_delta",
        )
        no_edge_month = _monthly_metric_stats(
            source_score_target_by_month,
            score_col=score_col,
            target_col="outcome_no_edge_flag",
            top_frac=scorecard_top_frac,
            metric_col="selected_target_delta",
        )
        capture_loss_month = _monthly_metric_stats(
            source_score_target_by_month,
            score_col=score_col,
            target_col="outcome_opportunity_capture_loss_flag",
            top_frac=scorecard_top_frac,
            metric_col="selected_target_delta",
        )

        bucket, action, hypothesis = _classify_promotion_from_metrics(
            score_kind=score_kind,
            utility_top=utility_top,
            utility_ic=utility_ic,
            recoverable_delta=recoverable_delta,
            recoverable_ic=recoverable_ic,
            capture_delta=capture_delta,
            capture_ic=capture_ic,
            capture_efficiency_delta=capture_efficiency_delta,
            path_failure_delta=path_failure_delta,
            no_edge_delta=no_edge_delta,
            capture_loss_delta=capture_loss_delta,
        )
        stability_bucket, promotion_confidence = _promotion_stability(
            score_kind=score_kind,
            bucket=bucket,
            capture_month=capture_month,
            recoverable_month=recoverable_month,
            path_failure_month=path_failure_month,
            no_edge_month=no_edge_month,
            capture_loss_month=capture_loss_month,
            utility_month=utility_month,
            stable_rate=stable_month_rate,
            mixed_rate=mixed_month_rate,
            high_confidence_utility_rate=high_confidence_utility_rate,
        )

        rows.append(
            {
                "score_col": score_col,
                "score_kind": score_kind,
                "top_frac": scorecard_top_frac,
                "promotion_bucket": bucket,
                "stability_bucket": stability_bucket,
                "promotion_confidence": promotion_confidence,
                "training_action": action,
                "hypothesis": hypothesis,
                "utility_top_mean": utility_top,
                "utility_ic": utility_ic,
                "monthly_utility_positive_rate": utility_month["positive_rate"],
                "monthly_utility_min": utility_month["min"],
                "bad_mae_top_rate": bad_mae_top,
                "p90_mae_top": p90_mae_top,
                "timeout_top_rate": timeout_top,
                "recoverable_delta": recoverable_delta,
                "recoverable_ic": recoverable_ic,
                "monthly_recoverable_delta_positive_rate": recoverable_month["positive_rate"],
                "monthly_recoverable_delta_min": recoverable_month["min"],
                "capture_delta": capture_delta,
                "capture_ic": capture_ic,
                "monthly_capture_delta_positive_rate": capture_month["positive_rate"],
                "monthly_capture_delta_min": capture_month["min"],
                "capture_among_opportunity_rate": capture_among_opp,
                "capture_efficiency_delta": capture_efficiency_delta,
                "monthly_capture_efficiency_delta_positive_rate": capture_efficiency_month["positive_rate"],
                "monthly_capture_efficiency_delta_min": capture_efficiency_month["min"],
                "capture_efficiency_top_mean": capture_efficiency_top,
                "path_failure_delta": path_failure_delta,
                "path_failure_ic": path_failure_ic,
                "monthly_path_failure_delta_nonpositive_rate": path_failure_month["nonpositive_rate"],
                "monthly_path_failure_delta_max": path_failure_month["max"],
                "no_edge_delta": no_edge_delta,
                "monthly_no_edge_delta_nonpositive_rate": no_edge_month["nonpositive_rate"],
                "monthly_no_edge_delta_max": no_edge_month["max"],
                "capture_loss_delta": capture_loss_delta,
                "monthly_capture_loss_delta_nonpositive_rate": capture_loss_month["nonpositive_rate"],
                "monthly_capture_loss_delta_max": capture_loss_month["max"],
                "monthly_evidence_count": max(
                    utility_month["count"],
                    capture_month["count"],
                    recoverable_month["count"],
                    path_failure_month["count"],
                    no_edge_month["count"],
                    capture_loss_month["count"],
                ),
            }
        )
    priority = {
        "safer_opportunity": 0,
        "capture_maximizer": 1,
        "opportunity_finder": 2,
        "utility_positive_slice": 3,
        "risk_avoidance": 4,
        "component_signal": 5,
        "risk_component": 6,
        "component_diagnostic": 7,
        "not_promoted": 8,
    }
    out = pd.DataFrame(rows)
    out["_priority"] = out["promotion_bucket"].map(priority).fillna(99)
    return out.sort_values(
        ["_priority", "capture_delta", "recoverable_delta", "utility_top_mean"],
        ascending=[True, False, False, False],
    ).drop(columns=["_priority"])


def _month_metric_value(
    frame: pd.DataFrame | None,
    *,
    score_col: str,
    month: str,
    top_frac: float,
    metric_col: str,
    target_col: str | None = None,
) -> float:
    if frame is None or frame.empty or metric_col not in frame.columns:
        return float("nan")
    subset = frame[
        frame["score_col"].eq(score_col)
        & frame["scope"].eq("month")
        & frame["bucket"].astype(str).eq(str(month))
    ].copy()
    if target_col is not None:
        subset = subset[subset["target_col"].eq(target_col)]
    if subset.empty:
        return float("nan")
    subset["_frac_distance"] = (_safe_numeric(subset["top_frac"]) - float(top_frac)).abs()
    row = subset.sort_values(["_frac_distance", "top_frac"], ascending=[True, False]).iloc[0]
    return float(row[metric_col]) if pd.notna(row[metric_col]) else float("nan")


def _mean_finite(values: list[float]) -> float:
    finite = [float(v) for v in values if math.isfinite(float(v))]
    return float(np.mean(finite)) if finite else float("nan")


def _stats_from_values(values: list[float]) -> dict[str, float]:
    finite = [float(v) for v in values if math.isfinite(float(v))]
    if not finite:
        return {
            "count": 0.0,
            "mean": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "positive_rate": float("nan"),
            "nonpositive_rate": float("nan"),
            "negative_rate": float("nan"),
        }
    series = pd.Series(finite, dtype=float)
    return {
        "count": float(len(series)),
        "mean": _safe_mean(series),
        "min": _safe_quantile(series, 0.0),
        "max": _safe_quantile(series, 1.0),
        "positive_rate": _safe_mean(series > 0.0),
        "nonpositive_rate": _safe_mean(series <= 0.0),
        "negative_rate": _safe_mean(series < 0.0),
    }


def _walkforward_eval_success(
    *,
    promotion_bucket: str,
    eval_capture_delta: float,
    eval_recoverable_delta: float,
    eval_capture_efficiency_delta: float,
    eval_path_failure_delta: float,
    eval_no_edge_delta: float,
    eval_capture_loss_delta: float,
    eval_utility_top_mean: float,
) -> bool:
    if promotion_bucket == "safer_opportunity":
        return (
            math.isfinite(eval_capture_delta)
            and eval_capture_delta > 0.0
            and math.isfinite(eval_recoverable_delta)
            and eval_recoverable_delta > 0.0
            and (not math.isfinite(eval_path_failure_delta) or eval_path_failure_delta <= 0.0)
            and (not math.isfinite(eval_no_edge_delta) or eval_no_edge_delta <= 0.0)
        )
    if promotion_bucket == "capture_maximizer":
        return (
            (math.isfinite(eval_capture_delta) and eval_capture_delta > 0.0)
            or (math.isfinite(eval_capture_efficiency_delta) and eval_capture_efficiency_delta > 0.0)
        )
    if promotion_bucket == "opportunity_finder":
        return math.isfinite(eval_recoverable_delta) and eval_recoverable_delta > 0.0
    if promotion_bucket == "utility_positive_slice":
        return math.isfinite(eval_utility_top_mean) and eval_utility_top_mean > 0.0
    if promotion_bucket == "risk_avoidance":
        return (
            (math.isfinite(eval_path_failure_delta) and eval_path_failure_delta > 0.0)
            or (math.isfinite(eval_no_edge_delta) and eval_no_edge_delta > 0.0)
            or (math.isfinite(eval_capture_loss_delta) and eval_capture_loss_delta > 0.0)
        )
    return False


def build_source_archetype_walkforward_readiness(
    *,
    source_score_target_by_month: pd.DataFrame,
    source_score_quality_by_month: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Select source-score actions from prior months and evaluate the next month."""
    if source_score_target_by_month.empty and source_score_quality_by_month.empty:
        return pd.DataFrame()
    diagnostics = config.get("diagnostics") or {}
    top_frac = float(diagnostics.get("promotion_scorecard_top_frac", 0.10))
    min_history_months = int(diagnostics.get("promotion_walkforward_min_history_months", 2))
    stable_month_rate = float(diagnostics.get("promotion_stable_month_rate", 0.75))
    mixed_month_rate = float(diagnostics.get("promotion_mixed_month_rate", 0.50))
    high_confidence_utility_rate = float(diagnostics.get("promotion_high_confidence_utility_positive_month_rate", 0.50))
    score_cols = list(
        dict.fromkeys(
            list(source_score_target_by_month.get("score_col", pd.Series(dtype=str)).dropna().astype(str))
            + list(source_score_quality_by_month.get("score_col", pd.Series(dtype=str)).dropna().astype(str))
        )
    )
    months = sorted(
        set(source_score_target_by_month.get("bucket", pd.Series(dtype=str)).dropna().astype(str))
        | set(source_score_quality_by_month.get("bucket", pd.Series(dtype=str)).dropna().astype(str))
    )
    rows: list[dict[str, Any]] = []
    for score_col in score_cols:
        score_kind = "component" if score_col in COMPONENT_COLS else "archetype"
        metrics_by_month: dict[str, dict[str, float]] = {}
        for month in months:
            metrics_by_month[month] = {
                "utility_top_mean": _month_metric_value(
                    source_score_quality_by_month,
                    score_col=score_col,
                    month=month,
                    top_frac=top_frac,
                    metric_col="mean_net_utility",
                ),
                "utility_ic": _month_metric_value(
                    source_score_quality_by_month,
                    score_col=score_col,
                    month=month,
                    top_frac=top_frac,
                    metric_col="score_ic_utility",
                ),
                "recoverable_delta": _month_metric_value(
                    source_score_target_by_month,
                    score_col=score_col,
                    month=month,
                    top_frac=top_frac,
                    metric_col="selected_target_delta",
                    target_col="outcome_recoverable_opportunity_flag",
                ),
                "recoverable_ic": _month_metric_value(
                    source_score_target_by_month,
                    score_col=score_col,
                    month=month,
                    top_frac=top_frac,
                    metric_col="score_ic_target",
                    target_col="outcome_recoverable_opportunity_flag",
                ),
                "capture_delta": _month_metric_value(
                    source_score_target_by_month,
                    score_col=score_col,
                    month=month,
                    top_frac=top_frac,
                    metric_col="selected_target_delta",
                    target_col="outcome_opportunity_captured_flag",
                ),
                "capture_ic": _month_metric_value(
                    source_score_target_by_month,
                    score_col=score_col,
                    month=month,
                    top_frac=top_frac,
                    metric_col="score_ic_target",
                    target_col="outcome_opportunity_captured_flag",
                ),
                "capture_efficiency_delta": _month_metric_value(
                    source_score_target_by_month,
                    score_col=score_col,
                    month=month,
                    top_frac=top_frac,
                    metric_col="selected_target_delta",
                    target_col="opportunity_capture_efficiency",
                ),
                "path_failure_delta": _month_metric_value(
                    source_score_target_by_month,
                    score_col=score_col,
                    month=month,
                    top_frac=top_frac,
                    metric_col="selected_target_delta",
                    target_col="outcome_path_failure_flag",
                ),
                "no_edge_delta": _month_metric_value(
                    source_score_target_by_month,
                    score_col=score_col,
                    month=month,
                    top_frac=top_frac,
                    metric_col="selected_target_delta",
                    target_col="outcome_no_edge_flag",
                ),
                "capture_loss_delta": _month_metric_value(
                    source_score_target_by_month,
                    score_col=score_col,
                    month=month,
                    top_frac=top_frac,
                    metric_col="selected_target_delta",
                    target_col="outcome_opportunity_capture_loss_flag",
                ),
            }
        for idx, eval_month in enumerate(months):
            history_months = months[:idx]
            if len(history_months) < min_history_months:
                continue
            history = [metrics_by_month[month] for month in history_months]
            hist_utility = _stats_from_values([row["utility_top_mean"] for row in history])
            hist_capture = _stats_from_values([row["capture_delta"] for row in history])
            hist_recoverable = _stats_from_values([row["recoverable_delta"] for row in history])
            hist_path_failure = _stats_from_values([row["path_failure_delta"] for row in history])
            hist_no_edge = _stats_from_values([row["no_edge_delta"] for row in history])
            hist_capture_loss = _stats_from_values([row["capture_loss_delta"] for row in history])
            hist_capture_efficiency = _stats_from_values([row["capture_efficiency_delta"] for row in history])
            hist_metrics = {
                "utility_top": hist_utility["mean"],
                "utility_ic": _mean_finite([row["utility_ic"] for row in history]),
                "recoverable_delta": hist_recoverable["mean"],
                "recoverable_ic": _mean_finite([row["recoverable_ic"] for row in history]),
                "capture_delta": hist_capture["mean"],
                "capture_ic": _mean_finite([row["capture_ic"] for row in history]),
                "capture_efficiency_delta": hist_capture_efficiency["mean"],
                "path_failure_delta": hist_path_failure["mean"],
                "no_edge_delta": hist_no_edge["mean"],
                "capture_loss_delta": hist_capture_loss["mean"],
            }
            bucket, action, hypothesis = _classify_promotion_from_metrics(score_kind=score_kind, **hist_metrics)
            stability_bucket, promotion_confidence = _promotion_stability(
                score_kind=score_kind,
                bucket=bucket,
                capture_month=hist_capture,
                recoverable_month=hist_recoverable,
                path_failure_month=hist_path_failure,
                no_edge_month=hist_no_edge,
                capture_loss_month=hist_capture_loss,
                utility_month=hist_utility,
                stable_rate=stable_month_rate,
                mixed_rate=mixed_month_rate,
                high_confidence_utility_rate=high_confidence_utility_rate,
            )
            eval_metrics = metrics_by_month[eval_month]
            evaluated_bucket = bucket in {
                "safer_opportunity",
                "capture_maximizer",
                "opportunity_finder",
                "utility_positive_slice",
                "risk_avoidance",
            }
            eval_success = _walkforward_eval_success(
                promotion_bucket=bucket,
                eval_capture_delta=eval_metrics["capture_delta"],
                eval_recoverable_delta=eval_metrics["recoverable_delta"],
                eval_capture_efficiency_delta=eval_metrics["capture_efficiency_delta"],
                eval_path_failure_delta=eval_metrics["path_failure_delta"],
                eval_no_edge_delta=eval_metrics["no_edge_delta"],
                eval_capture_loss_delta=eval_metrics["capture_loss_delta"],
                eval_utility_top_mean=eval_metrics["utility_top_mean"],
            ) if evaluated_bucket else False
            walkforward_status = (
                "confirmed_next_month"
                if eval_success
                else "failed_next_month"
                if evaluated_bucket
                else "diagnostic_only_not_evaluated"
            )
            rows.append(
                {
                    "score_col": score_col,
                    "score_kind": score_kind,
                    "top_frac": top_frac,
                    "history_months": int(len(history_months)),
                    "history_start_month": history_months[0],
                    "history_end_month": history_months[-1],
                    "eval_month": eval_month,
                    "history_promotion_bucket": bucket,
                    "history_stability_bucket": stability_bucket,
                    "history_promotion_confidence": promotion_confidence,
                    "history_training_action": action,
                    "history_hypothesis": hypothesis,
                    "history_utility_top_mean": hist_metrics["utility_top"],
                    "history_utility_positive_rate": hist_utility["positive_rate"],
                    "history_recoverable_delta": hist_metrics["recoverable_delta"],
                    "history_recoverable_positive_rate": hist_recoverable["positive_rate"],
                    "history_capture_delta": hist_metrics["capture_delta"],
                    "history_capture_positive_rate": hist_capture["positive_rate"],
                    "history_capture_efficiency_delta": hist_metrics["capture_efficiency_delta"],
                    "history_path_failure_delta": hist_metrics["path_failure_delta"],
                    "history_path_failure_positive_rate": hist_path_failure["positive_rate"],
                    "history_no_edge_delta": hist_metrics["no_edge_delta"],
                    "history_capture_loss_delta": hist_metrics["capture_loss_delta"],
                    "eval_utility_top_mean": eval_metrics["utility_top_mean"],
                    "eval_recoverable_delta": eval_metrics["recoverable_delta"],
                    "eval_capture_delta": eval_metrics["capture_delta"],
                    "eval_capture_efficiency_delta": eval_metrics["capture_efficiency_delta"],
                    "eval_path_failure_delta": eval_metrics["path_failure_delta"],
                    "eval_no_edge_delta": eval_metrics["no_edge_delta"],
                    "eval_capture_loss_delta": eval_metrics["capture_loss_delta"],
                    "eval_signal_success": bool(eval_success),
                    "walkforward_status": walkforward_status,
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    priority = {
        "safer_opportunity": 0,
        "capture_maximizer": 1,
        "opportunity_finder": 2,
        "utility_positive_slice": 3,
        "risk_avoidance": 4,
        "component_signal": 5,
        "risk_component": 6,
        "component_diagnostic": 7,
        "not_promoted": 8,
    }
    out["_priority"] = out["history_promotion_bucket"].map(priority).fillna(99)
    return out.sort_values(
        ["eval_month", "_priority", "history_capture_delta", "history_recoverable_delta"],
        ascending=[True, True, False, False],
    ).drop(columns=["_priority"])


def build_label_ablation_manifest() -> dict[str, Any]:
    required = [
        "timestamp_balanced_HR@30",
        "NDCG@30",
        "weekly_Q10_HR@30",
        "net_PnL_after_costs",
        "bad_MAE_rate",
        "p90_MAE",
        "timeout_rate",
        "wide_barrier_rate",
        "economic_capture_rate",
        "expensive_capture_rate",
        "proxy_IC",
        "proxy_topk_realized_utility",
    ]
    entries = [
        {
            "name": "baseline_all_rows",
            "row_filter_expression": "train_include_non_neutral_v0 == true",
            "label_column": "quality_label_v0",
            "sample_weight_column": "sample_weight_base_v0",
            "intended_hypothesis": "Baseline source-aware quality label on all non-neutral rows.",
        },
        {
            "name": "quiet_continuation_only",
            "row_filter_expression": "train_include_quiet_only_v0 == true",
            "label_column": "quality_label_v0",
            "sample_weight_column": "sample_weight_base_v0",
            "intended_hypothesis": "Test whether quiet continuation is the learnable clean source.",
        },
        {
            "name": "loud_breakout_impulse_only",
            "row_filter_expression": "train_include_loud_only_v0 == true",
            "label_column": "quality_label_v0",
            "sample_weight_column": "sample_weight_base_v0",
            "intended_hypothesis": "Measure whether loud breakouts are trainable or mostly dirty.",
        },
        {
            "name": "loud_clean_only",
            "row_filter_expression": "train_include_loud_clean_v0 == true",
            "label_column": "quality_label_v0",
            "sample_weight_column": "sample_weight_base_v0",
            "intended_hypothesis": "Keep loud breakouts only when dirty-shock avoid is false.",
        },
        {
            "name": "dirty_excluded",
            "row_filter_expression": "train_include_dirty_excluded_v0 == true",
            "label_column": "quality_label_v0",
            "sample_weight_column": "sample_weight_base_v0",
            "intended_hypothesis": "Exclude dirty-shock rows and test lower path risk.",
        },
        {
            "name": "run_entry_boosted_late_downweighted",
            "row_filter_expression": "train_include_non_neutral_v0 == true",
            "label_column": "quality_label_v0",
            "sample_weight_column": "sample_weight_source_v1",
            "intended_hypothesis": "Boost early run entries and downweight late repeated rows.",
        },
        {
            "name": "source_score_weighted_v2",
            "row_filter_expression": "train_include_non_neutral_v0 == true",
            "label_column": "quality_label_v0",
            "sample_weight_column": "sample_weight_source_v2",
            "intended_hypothesis": "Use continuous causal source scores to boost positive-source rows and discount dirty-shock rows.",
        },
        {
            "name": "source_rank_balanced_v1",
            "row_filter_expression": "train_include_source_rank_non_neutral_v1 == true",
            "label_column": "quality_label_source_rank_v1",
            "sample_weight_column": "sample_weight_base_v0",
            "intended_hypothesis": "Make good/bad labels relative to the row's source archetype instead of one global quality distribution.",
        },
        {
            "name": "source_conditioned_wf_v1",
            "row_filter_expression": "train_include_source_wf_non_neutral_v1 == true",
            "label_column": "quality_label_source_wf_v1",
            "sample_weight_column": "sample_weight_source_wf_v1",
            "intended_hypothesis": "Use expanding prior source-specific thresholds so label cutoffs do not depend on future outcome distribution.",
        },
        {
            "name": "clean_path_label_v2",
            "row_filter_expression": "train_include_clean_path_non_neutral_v2 == true",
            "label_column": "quality_label_clean_path_v2",
            "sample_weight_column": "sample_weight_clean_path_v2",
            "intended_hypothesis": "Train only against clean profitable paths versus explicit path/no-edge failures.",
        },
        {
            "name": "recoverable_opportunity_label_v2",
            "row_filter_expression": "train_include_recoverable_opportunity_non_neutral_v2 == true",
            "label_column": "quality_label_recoverable_opportunity_v2",
            "sample_weight_column": "sample_weight_opportunity_v2",
            "intended_hypothesis": "Recover rows with enough future MFE and bounded MAE even when the current policy failed to monetize them.",
        },
        {
            "name": "missed_opportunity_review_v2",
            "row_filter_expression": "train_include_missed_opportunity_review_v2 == true",
            "label_column": "quality_label_recoverable_opportunity_v2",
            "sample_weight_column": "sample_weight_opportunity_v2",
            "intended_hypothesis": "Diagnostic review set for rows where MFE/MAE says opportunity existed but net utility was non-positive.",
        },
        {
            "name": "opportunity_capture_label_v3",
            "row_filter_expression": "train_include_opportunity_capture_non_neutral_v3 == true",
            "label_column": "quality_label_opportunity_capture_v3",
            "sample_weight_column": "sample_weight_capture_v3",
            "intended_hypothesis": "Among recoverable opportunities, learn whether the current policy captured or lost the opportunity.",
        },
        {
            "name": "opportunity_capture_loss_review_v3",
            "row_filter_expression": "train_include_opportunity_capture_loss_review_v3 == true",
            "label_column": "quality_label_opportunity_capture_v3",
            "sample_weight_column": "sample_weight_capture_v3",
            "intended_hypothesis": "Diagnostic review set for recoverable opportunities that the current policy failed to capture.",
        },
        {
            "name": "economic_capture_label_v4",
            "row_filter_expression": "train_include_economic_capture_non_neutral_v4 == true",
            "label_column": "quality_label_economic_capture_v4",
            "sample_weight_column": "sample_weight_economic_capture_v4",
            "intended_hypothesis": "Among recoverable opportunities, learn economically clean capture rather than any positive capture.",
        },
        {
            "name": "economic_capture_loss_review_v4",
            "row_filter_expression": "train_include_economic_capture_loss_review_v4 == true",
            "label_column": "quality_label_economic_capture_v4",
            "sample_weight_column": "sample_weight_economic_capture_v4",
            "intended_hypothesis": "Diagnostic review set for recoverable opportunities whose policy outcome was not economically clean.",
        },
        {
            "name": "no_edge_review_v2",
            "row_filter_expression": "train_include_no_edge_review_v2 == true",
            "label_column": "quality_label_recoverable_opportunity_v2",
            "sample_weight_column": "sample_weight_opportunity_v2",
            "intended_hypothesis": "Diagnostic review set for rows where the future path did not produce enough MFE within economic limits.",
        },
        {
            "name": "clean_execution_context_only",
            "row_filter_expression": "train_include_clean_execution_context_v1 == true",
            "label_column": "quality_label_v0",
            "sample_weight_column": "sample_weight_base_v0",
            "intended_hypothesis": "Check whether causal execution geometry alone isolates economically cleaner rows.",
        },
        {
            "name": "calm_positive_source_only",
            "row_filter_expression": "train_include_calm_positive_source_v1 == true",
            "label_column": "quality_label_v0",
            "sample_weight_column": "sample_weight_base_v0",
            "intended_hypothesis": "Test positive source strength after discounting shock, dirty execution, and barrier pressure.",
        },
        {
            "name": "loud_clean_execution_only",
            "row_filter_expression": "train_include_loud_clean_execution_v1 == true",
            "label_column": "quality_label_v0",
            "sample_weight_column": "sample_weight_base_v0",
            "intended_hypothesis": "Keep loud impulses only when execution and barrier geometry are clean.",
        },
        {
            "name": "clean_run_entry_only",
            "row_filter_expression": "train_include_clean_run_entry_v1 == true",
            "label_column": "quality_label_v0",
            "sample_weight_column": "sample_weight_base_v0",
            "intended_hypothesis": "Test whether early symbol-run rows are useful only when execution geometry is clean.",
        },
        {
            "name": "compression_capture_candidate_only",
            "row_filter_expression": "train_include_compression_capture_candidate_v3 == true",
            "label_column": "quality_label_opportunity_capture_v3",
            "sample_weight_column": "sample_weight_capture_v3",
            "intended_hypothesis": "Test the causal compression/late-run source archetype that best separated opportunity capture in diagnostics.",
        },
        {
            "name": "compression_capture_score_top10",
            "row_filter_expression": "train_include_compression_capture_score_top10_v3 == true",
            "label_column": "quality_label_opportunity_capture_v3",
            "sample_weight_column": "sample_weight_capture_v3",
            "intended_hypothesis": "Evaluate the strictest compression/late-run capture candidate score slice.",
        },
        {
            "name": "risk_adjusted_capture_candidate_only",
            "row_filter_expression": "train_include_risk_adjusted_capture_candidate_v4 == true",
            "label_column": "quality_label_opportunity_capture_v3",
            "sample_weight_column": "sample_weight_capture_v3",
            "intended_hypothesis": "Test compression/late-run capture candidates after discounting misleading-location path risk.",
        },
        {
            "name": "risk_adjusted_capture_score_top10",
            "row_filter_expression": "train_include_risk_adjusted_capture_score_top10_v4 == true",
            "label_column": "quality_label_opportunity_capture_v3",
            "sample_weight_column": "sample_weight_capture_v3",
            "intended_hypothesis": "Evaluate the strictest risk-adjusted capture candidate score slice.",
        },
        {
            "name": "compression_economic_capture_score_top10",
            "row_filter_expression": "train_include_compression_economic_capture_score_top10_v4 == true",
            "label_column": "quality_label_economic_capture_v4",
            "sample_weight_column": "sample_weight_economic_capture_v4",
            "intended_hypothesis": "Test whether the strongest compression capture slice can learn economically clean capture, not just opportunity capture.",
        },
        {
            "name": "risk_adjusted_economic_capture_score_top10",
            "row_filter_expression": "train_include_risk_adjusted_economic_capture_score_top10_v4 == true",
            "label_column": "quality_label_economic_capture_v4",
            "sample_weight_column": "sample_weight_economic_capture_v4",
            "intended_hypothesis": "Test whether the risk-adjusted capture slice improves economic capture quality versus raw capture.",
        },
        {
            "name": "clean_economic_capture_candidate_only",
            "row_filter_expression": "train_include_clean_economic_capture_candidate_v5 == true",
            "label_column": "quality_label_economic_capture_v4",
            "sample_weight_column": "sample_weight_economic_capture_v4",
            "intended_hypothesis": "Test a stricter causal capture archetype that requires compression/capture structure plus clean execution and low path-risk proxies.",
        },
        {
            "name": "clean_economic_capture_score_top10",
            "row_filter_expression": "train_include_clean_economic_capture_score_top10_v5 == true",
            "label_column": "quality_label_economic_capture_v4",
            "sample_weight_column": "sample_weight_economic_capture_v4",
            "intended_hypothesis": "Evaluate the strictest clean-economic capture slice against the economic capture target.",
        },
        {
            "name": "misleading_location_risk_excluded",
            "row_filter_expression": "train_include_misleading_location_risk_excluded_v3 == true",
            "label_column": "quality_label_v0",
            "sample_weight_column": "sample_weight_base_v0",
            "intended_hypothesis": "Exclude location/retest/quiet patterns that diagnostics associated with path failure and low opportunity.",
        },
        {
            "name": "misleading_location_risk_bottom70",
            "row_filter_expression": "train_include_misleading_location_risk_bottom70_v3 == true",
            "label_column": "quality_label_v0",
            "sample_weight_column": "sample_weight_base_v0",
            "intended_hypothesis": "Keep rows outside the highest misleading-location path-risk scores.",
        },
        {
            "name": "quiet_score_top10",
            "row_filter_expression": "train_include_quiet_score_top10_v1 == true",
            "label_column": "quality_label_v0",
            "sample_weight_column": "sample_weight_base_v0",
            "intended_hypothesis": "Test a stricter quiet-continuation score slice than the default tag threshold.",
        },
        {
            "name": "base_positive_score_top10",
            "row_filter_expression": "train_include_base_positive_score_top10_v1 == true",
            "label_column": "quality_label_v0",
            "sample_weight_column": "sample_weight_base_v0",
            "intended_hypothesis": "Test the strongest causal positive-source rows regardless of archetype.",
        },
        {
            "name": "loud_clean_score_top10",
            "row_filter_expression": "train_include_loud_clean_score_top10_v1 == true",
            "label_column": "quality_label_v0",
            "sample_weight_column": "sample_weight_base_v0",
            "intended_hypothesis": "Test loud-breakout candidates only when the dirty-shock score is low.",
        },
        {
            "name": "not_dirty_score_top70",
            "row_filter_expression": "train_include_not_dirty_score_top70_v1 == true",
            "label_column": "quality_label_v0",
            "sample_weight_column": "sample_weight_base_v0",
            "intended_hypothesis": "Remove the dirtiest shock/execution candidates using continuous not-dirty score.",
        },
        {
            "name": "calm_positive_score_top10",
            "row_filter_expression": "train_include_calm_positive_score_top10_v1 == true",
            "label_column": "quality_label_v0",
            "sample_weight_column": "sample_weight_base_v0",
            "intended_hypothesis": "Evaluate the strictest calm positive-source rows before using the broader binary tag.",
        },
        {
            "name": "loud_clean_execution_score_top05",
            "row_filter_expression": "train_include_loud_clean_execution_score_top05_v1 == true",
            "label_column": "quality_label_v0",
            "sample_weight_column": "sample_weight_base_v0",
            "intended_hypothesis": "Evaluate the narrow loud impulse slice with strongest clean execution geometry.",
        },
        {
            "name": "clean_run_entry_score_top10",
            "row_filter_expression": "train_include_clean_run_entry_score_top10_v1 == true",
            "label_column": "quality_label_v0",
            "sample_weight_column": "sample_weight_base_v0",
            "intended_hypothesis": "Evaluate the strongest early clean-run rows.",
        },
        {
            "name": "source_multilabel_as_features",
            "row_filter_expression": "train_include_non_neutral_v0 == true",
            "label_column": "quality_label_v0",
            "sample_weight_column": "sample_weight_base_v0",
            "intended_hypothesis": "Expose source score/tag columns as candidate features.",
        },
        {
            "name": "source_primary_heads",
            "row_filter_expression": "group by primary_source_tag; do not train yet",
            "label_column": "quality_label_v0",
            "sample_weight_column": "sample_weight_base_v0",
            "intended_hypothesis": "Define separate source-head reports before any model training.",
        },
    ]
    return {
        "experiments": [
            {
                **entry,
                "required_diagnostic_metrics": required,
            }
            for entry in entries
        ]
    }


def _coverage_stability(source: pd.DataFrame, timestamp_col: str, config: dict[str, Any]) -> pd.DataFrame:
    month = _period_series(source[timestamp_col], "M")
    criteria = config.get("minimum_coverage_stability") or {}
    trainable_min = float(criteria.get("trainable_min_coverage", 0.05))
    preferred_min = float(criteria.get("preferred_min_coverage", 0.10))
    rows: list[dict[str, Any]] = []
    for tag in TAG_COLS:
        rates = source[tag].astype(float).groupby(month).mean()
        coverage = _safe_mean(source[tag].astype(float))
        if coverage >= preferred_min:
            status = "preferred_material"
        elif coverage >= trainable_min:
            status = "trainable_material"
        elif tag == "tag_dirty_shock_avoid" and coverage > 0.0:
            status = "abstention_candidate"
        else:
            status = "diagnostic_only"
        rows.append(
            {
                "source_tag": tag.replace("tag_", ""),
                "coverage": coverage,
                "monthly_min_coverage": _safe_quantile(rates, 0.00),
                "monthly_max_coverage": _safe_quantile(rates, 1.00),
                "monthly_coverage_cv": float(rates.std(ddof=0) / rates.mean()) if len(rates) and rates.mean() else float("nan"),
                "materiality_status": status,
            }
        )
    return pd.DataFrame(rows)


def _interpret_source_quality(overall_quality: pd.DataFrame, coverage: pd.DataFrame) -> list[str]:
    if overall_quality.empty:
        return ["- No realized outcome rows were available, so source quality could not be evaluated."]
    multi = overall_quality[overall_quality["scope"].eq("multi_tag")].copy()
    if multi.empty:
        return ["- No multi-tag quality rows were available."]
    lines: list[str] = []
    positive = multi[_safe_numeric(multi["mean_net_utility"]) > 0.0]
    if positive.empty:
        lines.append("- No source tag has positive mean realized net utility in this run; none should be promoted directly as a profitable bucket yet.")
    else:
        best = positive.sort_values("mean_net_utility", ascending=False).iloc[0]
        lines.append(f"- Best positive source by mean utility: `{best['value']}` with mean `{float(best['mean_net_utility']):.4f}`.")
    dirty = multi.sort_values("bad_mae_rate", ascending=False).iloc[0]
    clean = multi.sort_values("bad_mae_rate", ascending=True).iloc[0]
    lines.append(
        f"- Highest bad-MAE source: `{dirty['value']}` at `{float(dirty['bad_mae_rate']):.3f}`; lowest bad-MAE source: `{clean['value']}` at `{float(clean['bad_mae_rate']):.3f}`."
    )
    sparse = coverage[coverage["materiality_status"].eq("diagnostic_only")]["source_tag"].astype(str).tolist()
    if sparse:
        lines.append(f"- Diagnostic-only sparse tags: `{', '.join(sparse)}`.")
    if "proxy_ic" in multi.columns and multi["proxy_ic"].notna().sum() == 0:
        lines.append("- No existing proxy/model score column was present, so proxy IC and proxy top-k learnability are not measured in this run.")
    return lines


def write_report(
    *,
    output_dir: Path,
    frame: pd.DataFrame,
    source: pd.DataFrame,
    quality_by_month: pd.DataFrame,
    quality_by_week: pd.DataFrame,
    quality_by_symbol: pd.DataFrame,
    source_x_regime: pd.DataFrame,
    row_alignment_audit: pd.DataFrame,
    source_proxy_learnability_overall: pd.DataFrame,
    source_proxy_learnability_by_month: pd.DataFrame,
    source_proxy_learnability_by_week: pd.DataFrame,
    source_x_regime_proxy_learnability: pd.DataFrame,
    source_score_quality_overall: pd.DataFrame,
    source_score_quality_by_month: pd.DataFrame,
    source_score_target_overall: pd.DataFrame,
    source_archetype_promotion_scorecard: pd.DataFrame,
    source_archetype_walkforward_readiness: pd.DataFrame,
    failure_mode_by_source: pd.DataFrame,
    failure_mode_by_month: pd.DataFrame,
    opportunity_capture_by_source: pd.DataFrame,
    opportunity_capture_by_month: pd.DataFrame,
    capture_utility_gap_overall: pd.DataFrame,
    capture_utility_gap_by_month: pd.DataFrame,
    label_variant_summary: pd.DataFrame,
    manifest: dict[str, Any],
    config: dict[str, Any],
) -> Path:
    timestamp_col = str(config.get("timestamp_col") or "__ts__")
    symbol_col = str(config.get("symbol_col") or "__symbol__")
    path = output_dir / "source_tag_report.md"

    def table(df: pd.DataFrame, cols: list[str], limit: int | None = None) -> str:
        if df.empty:
            return "No rows."
        view = df[[col for col in cols if col in df.columns]].copy()
        if limit is not None:
            view = view.head(limit)
        for col in view.columns:
            if pd.api.types.is_float_dtype(view[col]):
                view[col] = view[col].map(lambda x: f"{float(x):.4f}" if pd.notna(x) else "")
        return view.to_markdown(index=False)

    coverage = _coverage_stability(source, timestamp_col, config)
    overall_quality = []
    if not quality_by_month.empty:
        overall_quality = (
            quality_by_month.groupby(["scope", "value"], dropna=False)
            .agg(
                rows=("rows", "sum"),
                mean_net_utility=("mean_net_utility", "mean"),
                bad_mae_rate=("bad_mae_rate", "mean"),
                p90_mae=("p90_mae", "mean"),
                timeout_rate=("timeout_rate", "mean"),
                weekly_mean_lower_tail_p10=("weekly_mean_lower_tail_p10", "mean"),
                proxy_ic=("proxy_ic", "mean"),
            )
            .reset_index()
            .sort_values("mean_net_utility", ascending=False)
        )
    registry = manifest.get("source_report", {}).get("registry", {})
    available_groups = {
        group: len(cols) for group, cols in (registry.get("available") or {}).items()
    }
    missing_groups = {
        group: len(cols) for group, cols in (registry.get("missing") or {}).items()
    }
    lines = [
        "# Candidate Source Tags And Quality Labels",
        "",
        "Diagnostic-only pipeline. Source tags are causal prediction-time archetypes; realized outcomes are used only for diagnostics and target-side label candidates.",
        "",
        "## 1. Data Summary",
        "",
        f"- Rows: `{len(frame)}`",
        f"- Date range: `{pd.to_datetime(frame[timestamp_col]).min()}` to `{pd.to_datetime(frame[timestamp_col]).max()}`",
        f"- Symbols: `{frame[symbol_col].nunique(dropna=True)}`",
        f"- Regime/head columns present: `{', '.join(manifest.get('regime_columns_present', [])) or 'none'}`",
        f"- Proxy/model score columns present: `{', '.join(manifest.get('proxy_learnability_report', {}).get('proxy_score_columns', [])) or 'none'}`",
        f"- Available causal feature groups: `{available_groups}`",
        f"- Missing feature counts by group: `{missing_groups}`",
        f"- Outcome columns: `{manifest.get('outcome_report', {}).get('outcome_columns', {})}`",
        "",
        "## 2. Leakage Audit",
        "",
        f"- Excluded outcome-like source columns: `{registry.get('excluded_outcome_like', [])}`",
        f"- Source columns used: `{len(manifest.get('source_report', {}).get('source_columns_used', []))}`",
        "- Source normalization: cross-sectional percentile ranks within prediction timestamp; lower-is-better features are inverted.",
        "- Run-entry/late-run scores: lagged same-symbol source strength only.",
        f"- Timestamp-rank fallback counts: `{manifest.get('source_report', {}).get('tags', {}).get('fallback_counts', {})}`",
        f"- Smoke tests: `{manifest.get('smoke_tests', {})}`",
        "",
        "## V17. Alignment And Proxy Learnability",
        "",
        "### Alignment Summary",
        "",
        table(
            row_alignment_audit,
            [
                "alignment_quality",
                "feature_input_rows",
                "candidate_source_tags_rows",
                "quality_label_candidates_rows",
                "outcome_rows_matched",
                "outcome_match_rate",
                "prediction_rows",
                "prediction_match_rate",
                "rows_with_missing_proxy_score",
                "rows_with_missing_regime",
                "rows_with_multiple_outcomes_joined",
                "rows_with_multiple_predictions_joined",
                "duplicate_timestamp_symbol_rows",
                "alignment_warnings",
            ],
        ),
        "",
        "### Proxy Score Availability",
        "",
        f"- Prediction report: `{manifest.get('prediction_report', {})}`",
        f"- Proxy learnability report: `{manifest.get('proxy_learnability_report', {})}`",
        "",
        "### Learnability By Source",
        "",
        table(
            source_proxy_learnability_overall.sort_values("proxy_topk_mean_utility", ascending=False)
            if not source_proxy_learnability_overall.empty and "proxy_topk_mean_utility" in source_proxy_learnability_overall.columns
            else source_proxy_learnability_overall,
            [
                "source_scope",
                "source_tag",
                "proxy_col",
                "top_frac",
                "rows",
                "valid_rows",
                "proxy_ic_spearman",
                "proxy_topk_mean_utility",
                "proxy_topk_bad_mae_rate",
                "proxy_topk_timeout_rate",
                "proxy_topk_p90_mae",
                "proxy_topk_economic_capture_good_rate",
                "oracle_topk_mean_utility",
                "proxy_oracle_capture_ratio",
                "capture_at_k",
                "precision_at_k_quality_label_economic_capture_v4",
            ],
            limit=40,
        ),
        "",
        "### Learnability By Source X Regime",
        "",
        table(
            source_x_regime_proxy_learnability.sort_values("proxy_topk_mean_utility", ascending=False)
            if not source_x_regime_proxy_learnability.empty and "proxy_topk_mean_utility" in source_x_regime_proxy_learnability.columns
            else source_x_regime_proxy_learnability,
            [
                "regime_col",
                "regime_value",
                "source_tag",
                "proxy_col",
                "top_frac",
                "rows",
                "source_concentration_within_regime",
                "source_lift_vs_overall",
                "proxy_ic_spearman",
                "proxy_topk_mean_utility",
                "proxy_topk_bad_mae_rate",
                "proxy_topk_economic_capture_good_rate",
                "oracle_topk_mean_utility",
                "proxy_oracle_capture_ratio",
            ],
            limit=50,
        ),
        "",
        "### Recommended Ablations",
        "",
        "- `baseline_all_rows`",
        "- `economic_capture_label_v4`",
        "- `dirty_excluded`",
        "- `risk_adjusted_capture_candidate_only`",
        "- `compression_capture_candidate_only`",
        "- `risk_adjusted_economic_capture_score_top10`",
        "- `compression_economic_capture_score_top10`",
        "- `source_multilabel_as_features`",
        "",
        "### Do-Not-Promote Warnings",
        "",
        "- Do not promote source tags directly on mean utility; require walk-forward delta versus baseline.",
        "- Do not redesign regimes unless source x regime proxy metrics show regime adds information after controlling for source.",
        "- Do not treat `primary_source_tag` as the only training signal; it is a lossy reporting collapse.",
        "",
        "## 3. Source Tag Coverage",
        "",
        table(
            coverage,
            ["source_tag", "coverage", "monthly_min_coverage", "monthly_max_coverage", "monthly_coverage_cv", "materiality_status"],
        ),
        "",
        "## 4. Source Quality Summary",
        "",
        *(_interpret_source_quality(overall_quality, coverage)),
        "",
        table(
            overall_quality,
            ["scope", "value", "rows", "mean_net_utility", "bad_mae_rate", "p90_mae", "timeout_rate", "weekly_mean_lower_tail_p10", "proxy_ic"],
            limit=30,
        ),
        "",
        "## 5. Source Score Slices",
        "",
        table(
            source_score_quality_overall.sort_values("mean_net_utility", ascending=False)
            if not source_score_quality_overall.empty
            else source_score_quality_overall,
            [
                "score_col",
                "top_frac",
                "selected_rows",
                "mean_net_utility",
                "bad_mae_rate",
                "p90_mae",
                "timeout_rate",
                "score_ic_utility",
                "top_symbol_share",
            ],
            limit=40,
        ),
        "",
        "## 6. Source Score Target Diagnostics",
        "",
        table(
            source_score_target_overall.assign(
                abs_score_ic_target=source_score_target_overall["score_ic_target"].abs()
            ).sort_values("abs_score_ic_target", ascending=False)
            if not source_score_target_overall.empty and "score_ic_target" in source_score_target_overall.columns
            else source_score_target_overall,
            [
                "score_col",
                "target_col",
                "top_frac",
                "valid_rows",
                "selected_rows",
                "score_ic_target",
                "target_mean",
                "selected_target_mean",
                "selected_target_delta",
                "selected_target_lift_ratio",
                "selected_mean_net_utility",
                "selected_opportunity_rate",
                "selected_capture_among_opportunity_rate",
                "selected_capture_efficiency_mean",
                "selected_path_failure_rate",
                "selected_no_edge_rate",
            ],
            limit=50,
        ),
        "",
        "## 7. Archetype Promotion Scorecard",
        "",
        table(
            source_archetype_promotion_scorecard,
            [
                "score_col",
                "score_kind",
                "promotion_bucket",
                "stability_bucket",
                "promotion_confidence",
                "training_action",
                "utility_top_mean",
                "monthly_utility_positive_rate",
                "recoverable_delta",
                "monthly_recoverable_delta_positive_rate",
                "capture_delta",
                "monthly_capture_delta_positive_rate",
                "capture_among_opportunity_rate",
                "capture_efficiency_delta",
                "path_failure_delta",
                "monthly_path_failure_delta_nonpositive_rate",
                "no_edge_delta",
                "capture_loss_delta",
            ],
            limit=40,
        ),
        "",
        "## 8. Walk-Forward Promotion Readiness",
        "",
        table(
            source_archetype_walkforward_readiness,
            [
                "score_col",
                "score_kind",
                "history_end_month",
                "eval_month",
                "history_promotion_bucket",
                "history_stability_bucket",
                "history_promotion_confidence",
                "eval_signal_success",
                "walkforward_status",
                "history_capture_delta",
                "eval_capture_delta",
                "history_recoverable_delta",
                "eval_recoverable_delta",
                "history_path_failure_delta",
                "eval_path_failure_delta",
                "history_no_edge_delta",
                "eval_no_edge_delta",
                "history_utility_top_mean",
                "eval_utility_top_mean",
            ],
            limit=60,
        ),
        "",
        "## 9. Regime Interaction",
        "",
        table(
            source_x_regime.sort_values("mean_net_utility", ascending=False) if not source_x_regime.empty else source_x_regime,
            ["regime_col", "regime_value", "source_tag", "rows", "coverage_pct", "mean_net_utility", "bad_mae_rate", "p90_mae", "timeout_rate", "proxy_ic"],
            limit=40,
        ),
        "",
        "## 10. Recommended Next Ablations",
        "",
        "- `baseline_all_rows`",
        "- `quiet_continuation_only`",
        "- `loud_clean_only`",
        "- `dirty_excluded`",
        "- `run_entry_boosted_late_downweighted`",
        "- `source_score_weighted_v2`",
        "- `source_rank_balanced_v1`",
        "- `source_conditioned_wf_v1`",
        "- `calm_positive_source_only`",
        "- `loud_clean_execution_only`",
        "- `clean_run_entry_only`",
        "- `compression_capture_candidate_only`",
        "- `compression_capture_score_top10`",
        "- `risk_adjusted_capture_candidate_only`",
        "- `risk_adjusted_capture_score_top10`",
        "- `economic_capture_label_v4`",
        "- `compression_economic_capture_score_top10`",
        "- `risk_adjusted_economic_capture_score_top10`",
        "- `misleading_location_risk_excluded`",
        "- `base_positive_score_top10`",
        "- `loud_clean_score_top10`",
        "",
        "## 11. Failure Mode Summary",
        "",
        f"- Failure-mode counts: `{manifest.get('label_report', {}).get('failure_mode_counts', {})}`",
        "",
        table(
            failure_mode_by_source.sort_values("recoverable_opportunity_rate", ascending=False)
            if not failure_mode_by_source.empty and "recoverable_opportunity_rate" in failure_mode_by_source.columns
            else failure_mode_by_source,
            [
                "bucket",
                "rows",
                "mean_net_utility",
                "positive_utility_rate",
                "clean_win_rate",
                "dirty_win_rate",
                "recoverable_opportunity_rate",
                "missed_opportunity_rate",
                "path_failure_rate",
                "no_edge_rate",
                "timeout_failure_rate",
                "p90_mae",
                "p90_mfe",
            ],
            limit=30,
        ),
        "",
        table(
            failure_mode_by_month,
            [
                "bucket",
                "rows",
                "mean_net_utility",
                "positive_utility_rate",
                "clean_win_rate",
                "recoverable_opportunity_rate",
                "missed_opportunity_rate",
                "path_failure_rate",
                "no_edge_rate",
                "timeout_failure_rate",
            ],
        ),
        "",
        "## 12. Opportunity Capture Summary",
        "",
        table(
            opportunity_capture_by_source.sort_values("capture_rate", ascending=False)
            if not opportunity_capture_by_source.empty and "capture_rate" in opportunity_capture_by_source.columns
            else opportunity_capture_by_source,
            [
                "bucket",
                "opportunity_rows",
                "capture_rate",
                "economic_capture_rate",
                "expensive_capture_rate",
                "capture_loss_rate",
                "missed_opportunity_rate",
                "clean_win_rate",
                "positive_utility_rate",
                "mean_net_utility",
                "captured_mean_utility",
                "economic_capture_mean_utility",
                "expensive_capture_mean_utility",
                "loss_mean_utility",
                "mean_capture_efficiency",
                "p25_capture_efficiency",
                "p75_capture_efficiency",
                "p90_mae",
                "p90_mfe",
            ],
            limit=30,
        ),
        "",
        table(
            opportunity_capture_by_month,
            [
                "bucket",
                "opportunity_rows",
                "capture_rate",
                "economic_capture_rate",
                "expensive_capture_rate",
                "capture_loss_rate",
                "missed_opportunity_rate",
                "clean_win_rate",
                "positive_utility_rate",
                "mean_net_utility",
                "mean_capture_efficiency",
                "p90_mae",
                "p90_mfe",
            ],
        ),
        "",
        "## 13. Capture Utility Gap",
        "",
        table(
            capture_utility_gap_overall.sort_values("selected_mean_utility", ascending=False)
            if not capture_utility_gap_overall.empty and "selected_mean_utility" in capture_utility_gap_overall.columns
            else capture_utility_gap_overall,
            [
                "score_col",
                "top_frac",
                "selected_rows",
                "selected_mean_utility",
                "selected_positive_utility_rate",
                "selected_opportunity_rate",
                "selected_capture_rate",
                "selected_economic_capture_rate",
                "selected_expensive_capture_rate",
                "selected_economic_capture_loss_rate",
                "capture_among_opportunity_rate",
                "economic_capture_among_opportunity_rate",
                "expensive_capture_among_opportunity_rate",
                "path_failure_rate",
                "no_edge_rate",
                "mean_capture_efficiency",
            ],
            limit=50,
        ),
        "",
        table(
            capture_utility_gap_by_month,
            [
                "bucket",
                "score_col",
                "top_frac",
                "selected_mean_utility",
                "selected_economic_capture_rate",
                "selected_expensive_capture_rate",
                "path_failure_rate",
                "no_edge_rate",
            ],
            limit=60,
        ),
        "",
        "## 14. Label Candidate Summary",
        "",
        f"- `quality_label_v0` counts: `{manifest.get('label_report', {}).get('label_counts', {})}`",
        f"- `quality_label_source_rank_v1` counts: `{manifest.get('label_report', {}).get('source_rank_label_counts', {})}`",
        f"- `quality_label_source_wf_v1` counts: `{manifest.get('label_report', {}).get('source_wf_label_counts', {})}`",
        f"- `quality_label_clean_path_v2` counts: `{manifest.get('label_report', {}).get('clean_path_label_counts', {})}`",
        f"- `quality_label_recoverable_opportunity_v2` counts: `{manifest.get('label_report', {}).get('recoverable_opportunity_label_counts', {})}`",
        f"- `quality_label_opportunity_capture_v3` counts: `{manifest.get('label_report', {}).get('opportunity_capture_label_counts', {})}`",
        f"- `quality_label_economic_capture_v4` counts: `{manifest.get('label_report', {}).get('economic_capture_label_counts', {})}`",
        f"- Source-WF calibrated rows: `{manifest.get('label_report', {}).get('source_wf_calibrated_rows', 0)}`",
        f"- Score/top-slice include counts: `{manifest.get('label_report', {}).get('score_flag_counts', {})}`",
        "",
        table(
            label_variant_summary,
            [
                "label_col",
                "labeled_rows",
                "positive_rows",
                "negative_rows",
                "positive_rate",
                "positive_mean_utility",
                "negative_mean_utility",
                "positive_bad_mae_rate",
                "negative_bad_mae_rate",
                "positive_p90_mae",
                "negative_p90_mae",
            ],
        ),
        "",
        "## Outputs",
        "",
    ]
    for key, value in manifest.get("outputs", {}).items():
        lines.append(f"- {key}: `{value}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _source_signature(source: pd.DataFrame) -> pd.DataFrame:
    cols = COMPONENT_COLS + ARCHETYPE_COLS + TAG_COLS + ["primary_source_tag"]
    return source[[col for col in cols if col in source.columns]].copy()


def run_smoke_tests(frame: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    source, _ = materialize_source_tags(frame, config)
    baseline = _source_signature(source)
    outcome_cols = [
        col
        for candidates in (config.get("outcome_columns") or {}).values()
        for col in candidates or []
        if col in frame.columns
    ]
    shuffled = frame.copy()
    rng = np.random.default_rng(42)
    for col in outcome_cols:
        shuffled[col] = rng.permutation(shuffled[col].to_numpy(copy=True))
    shuffled_sig = _source_signature(materialize_source_tags(shuffled, config)[0])

    dropped = frame.drop(columns=outcome_cols, errors="ignore")
    dropped_sig = _source_signature(materialize_source_tags(dropped, config)[0])

    timestamp_col = str(config.get("timestamp_col") or "__ts__")
    symbol_col = str(config.get("symbol_col") or "__symbol__")
    future = frame.copy()
    future[timestamp_col] = pd.to_datetime(future[timestamp_col], errors="coerce") + pd.Timedelta(days=365)
    appended = pd.concat([frame, future], ignore_index=True)
    appended_sig = _source_signature(materialize_source_tags(appended, config)[0]).iloc[: len(frame)].reset_index(drop=True)
    baseline_reset = baseline.reset_index(drop=True)

    def equal_frame(left: pd.DataFrame, right: pd.DataFrame) -> bool:
        for col in left.columns:
            if pd.api.types.is_numeric_dtype(left[col]):
                if not np.allclose(
                    pd.to_numeric(left[col], errors="coerce").fillna(-9999),
                    pd.to_numeric(right[col], errors="coerce").fillna(-9999),
                    atol=1e-7,
                    equal_nan=True,
                ):
                    return False
            elif not left[col].astype(str).equals(right[col].astype(str)):
                return False
        return True

    return {
        "rows_tested": int(len(frame)),
        "source_unchanged_after_outcome_shuffle": equal_frame(baseline_reset, shuffled_sig.reset_index(drop=True)),
        "source_unchanged_after_outcome_drop": equal_frame(baseline_reset, dropped_sig.reset_index(drop=True)),
        "past_source_unchanged_after_future_append": equal_frame(baseline_reset, appended_sig),
    }


def _smoke_sample_frame(frame: pd.DataFrame, config: dict[str, Any], *, max_rows: int = 50_000) -> pd.DataFrame:
    if len(frame) <= int(max_rows):
        return frame.copy()
    timestamp_col = str(config.get("timestamp_col") or "__ts__")
    timestamps = pd.to_datetime(frame[timestamp_col], errors="coerce")
    month = timestamps.dt.to_period("M").astype(str)
    parts: list[pd.DataFrame] = []
    per_month = max(1, int(max_rows) // max(1, month.nunique(dropna=True)))
    for _, group in frame.groupby(month, sort=True, dropna=False):
        if len(group) <= per_month:
            parts.append(group)
        else:
            # Deterministic time-spread sample: beginning, middle, end by row order.
            positions = np.linspace(0, len(group) - 1, per_month, dtype=np.int64)
            parts.append(group.iloc[np.unique(positions)])
    return pd.concat(parts, ignore_index=True).head(int(max_rows))


def materialize_pipeline(
    *,
    input_path: Path,
    outdir: Path,
    config_path: Path,
    labels_path: Path | None = None,
    predictions_path: Path | None = None,
    prediction_key_cols: list[str] | None = None,
    proxy_score_cols: list[str] | None = None,
    run_smokes: bool = True,
) -> dict[str, Any]:
    config = load_config(config_path)
    if proxy_score_cols:
        config["proxy_score_columns"] = _configured_proxy_score_columns(config, proxy_score_cols)
    needed = _flatten_config_columns(config)
    needed.update(prediction_key_cols or [])
    frame = load_frame(input_path, config, columns=needed)
    frame = add_source_identity(frame, config)
    frame, label_join_report = join_labels(frame, labels_path, config)
    frame, prediction_report = join_predictions(
        frame,
        predictions_path,
        config,
        prediction_key_cols=prediction_key_cols,
        proxy_score_cols=proxy_score_cols,
    )
    outdir.mkdir(parents=True, exist_ok=True)
    source, source_report = materialize_source_tags(frame, config)
    outcomes, outcome_report = build_outcome_frame(frame, source, config)
    full = pd.concat([source, outcomes], axis=1)
    by_month, by_week, by_symbol, source_x_regime, quality_report = evaluate_source_quality(full, config)
    source_score_quality_overall, source_score_quality_by_month = evaluate_source_score_slices(full, config)
    source_score_target_overall, source_score_target_by_month = evaluate_source_score_target_diagnostics(full, config)
    source_archetype_promotion_scorecard = build_source_archetype_promotion_scorecard(
        source_score_target_overall=source_score_target_overall,
        source_score_quality_overall=source_score_quality_overall,
        source_score_target_by_month=source_score_target_by_month,
        source_score_quality_by_month=source_score_quality_by_month,
        config=config,
    )
    source_archetype_walkforward_readiness = build_source_archetype_walkforward_readiness(
        source_score_target_by_month=source_score_target_by_month,
        source_score_quality_by_month=source_score_quality_by_month,
        config=config,
    )
    failure_mode_by_source, failure_mode_by_month, failure_mode_by_source_month = evaluate_failure_modes(full, config)
    opportunity_capture_by_source, opportunity_capture_by_month, opportunity_capture_by_source_month = evaluate_opportunity_capture(full, config)
    capture_utility_gap_overall, capture_utility_gap_by_month = evaluate_capture_utility_gap(full, config)
    label_candidates, label_report = build_quality_label_candidates(source, outcomes, config)
    label_cols_for_learnability = [col for col in QUALITY_LABEL_VARIANTS if col in label_candidates.columns]
    learnability_frame = pd.concat([full, label_candidates[label_cols_for_learnability]], axis=1)
    (
        source_proxy_learnability_overall,
        source_proxy_learnability_by_month,
        source_proxy_learnability_by_week,
        source_x_regime_proxy_learnability,
        proxy_learnability_report,
    ) = evaluate_proxy_learnability(learnability_frame, config)
    row_alignment_audit, row_alignment_report = build_row_alignment_audit(
        frame=frame,
        source=source,
        label_candidates=label_candidates,
        outcomes=outcomes,
        config=config,
        label_join_report=label_join_report,
        prediction_report=prediction_report,
    )
    label_variant_summary, label_variant_by_month, label_variant_by_source = evaluate_quality_label_variants(
        label_candidates,
        outcomes,
        config,
    )
    label_manifest = build_label_ablation_manifest()
    smoke_frame = _smoke_sample_frame(frame, config) if run_smokes else frame.iloc[:0].copy()
    smoke_tests = run_smoke_tests(smoke_frame, config) if run_smokes else {}

    paths = {
        "candidate_source_tags_parquet": outdir / "candidate_source_tags.parquet",
        "candidate_source_tags_csv": outdir / "candidate_source_tags.csv",
        "source_tag_quality_by_month": outdir / "source_tag_quality_by_month.csv",
        "source_tag_quality_by_week": outdir / "source_tag_quality_by_week.csv",
        "source_tag_quality_by_symbol": outdir / "source_tag_quality_by_symbol.csv",
        "source_x_regime_matrix": outdir / "source_x_regime_matrix.csv",
        "row_alignment_audit": outdir / "row_alignment_audit.csv",
        "source_proxy_learnability_overall": outdir / "source_proxy_learnability_overall.csv",
        "source_proxy_learnability_by_month": outdir / "source_proxy_learnability_by_month.csv",
        "source_proxy_learnability_by_week": outdir / "source_proxy_learnability_by_week.csv",
        "source_x_regime_proxy_learnability": outdir / "source_x_regime_proxy_learnability.csv",
        "source_score_quality_overall": outdir / "source_score_quality_overall.csv",
        "source_score_quality_by_month": outdir / "source_score_quality_by_month.csv",
        "source_score_target_diagnostics": outdir / "source_score_target_diagnostics.csv",
        "source_score_target_diagnostics_by_month": outdir / "source_score_target_diagnostics_by_month.csv",
        "source_archetype_promotion_scorecard": outdir / "source_archetype_promotion_scorecard.csv",
        "source_archetype_walkforward_readiness": outdir / "source_archetype_walkforward_readiness.csv",
        "failure_mode_by_source": outdir / "failure_mode_by_source.csv",
        "failure_mode_by_month": outdir / "failure_mode_by_month.csv",
        "failure_mode_by_source_month": outdir / "failure_mode_by_source_month.csv",
        "opportunity_capture_by_source": outdir / "opportunity_capture_by_source.csv",
        "opportunity_capture_by_month": outdir / "opportunity_capture_by_month.csv",
        "opportunity_capture_by_source_month": outdir / "opportunity_capture_by_source_month.csv",
        "capture_utility_gap_overall": outdir / "capture_utility_gap_overall.csv",
        "capture_utility_gap_by_month": outdir / "capture_utility_gap_by_month.csv",
        "quality_label_candidates_parquet": outdir / "quality_label_candidates.parquet",
        "quality_label_candidates_csv": outdir / "quality_label_candidates.csv",
        "quality_label_variant_summary": outdir / "quality_label_variant_summary.csv",
        "quality_label_variant_by_month": outdir / "quality_label_variant_by_month.csv",
        "quality_label_variant_by_source": outdir / "quality_label_variant_by_source.csv",
        "label_ablation_manifest": outdir / "label_ablation_manifest.json",
        "pipeline_manifest": outdir / "source_tag_pipeline_manifest.json",
    }
    source.to_parquet(paths["candidate_source_tags_parquet"], index=False)
    source.to_csv(paths["candidate_source_tags_csv"], index=False)
    by_month.to_csv(paths["source_tag_quality_by_month"], index=False)
    by_week.to_csv(paths["source_tag_quality_by_week"], index=False)
    by_symbol.to_csv(paths["source_tag_quality_by_symbol"], index=False)
    source_x_regime.to_csv(paths["source_x_regime_matrix"], index=False)
    row_alignment_audit.to_csv(paths["row_alignment_audit"], index=False)
    source_proxy_learnability_overall.to_csv(paths["source_proxy_learnability_overall"], index=False)
    source_proxy_learnability_by_month.to_csv(paths["source_proxy_learnability_by_month"], index=False)
    source_proxy_learnability_by_week.to_csv(paths["source_proxy_learnability_by_week"], index=False)
    source_x_regime_proxy_learnability.to_csv(paths["source_x_regime_proxy_learnability"], index=False)
    source_score_quality_overall.to_csv(paths["source_score_quality_overall"], index=False)
    source_score_quality_by_month.to_csv(paths["source_score_quality_by_month"], index=False)
    source_score_target_overall.to_csv(paths["source_score_target_diagnostics"], index=False)
    source_score_target_by_month.to_csv(paths["source_score_target_diagnostics_by_month"], index=False)
    source_archetype_promotion_scorecard.to_csv(paths["source_archetype_promotion_scorecard"], index=False)
    source_archetype_walkforward_readiness.to_csv(paths["source_archetype_walkforward_readiness"], index=False)
    failure_mode_by_source.to_csv(paths["failure_mode_by_source"], index=False)
    failure_mode_by_month.to_csv(paths["failure_mode_by_month"], index=False)
    failure_mode_by_source_month.to_csv(paths["failure_mode_by_source_month"], index=False)
    opportunity_capture_by_source.to_csv(paths["opportunity_capture_by_source"], index=False)
    opportunity_capture_by_month.to_csv(paths["opportunity_capture_by_month"], index=False)
    opportunity_capture_by_source_month.to_csv(paths["opportunity_capture_by_source_month"], index=False)
    capture_utility_gap_overall.to_csv(paths["capture_utility_gap_overall"], index=False)
    capture_utility_gap_by_month.to_csv(paths["capture_utility_gap_by_month"], index=False)
    label_candidates.to_parquet(paths["quality_label_candidates_parquet"], index=False)
    label_candidates.to_csv(paths["quality_label_candidates_csv"], index=False)
    label_variant_summary.to_csv(paths["quality_label_variant_summary"], index=False)
    label_variant_by_month.to_csv(paths["quality_label_variant_by_month"], index=False)
    label_variant_by_source.to_csv(paths["quality_label_variant_by_source"], index=False)
    paths["label_ablation_manifest"].write_text(json.dumps(_json_safe(label_manifest), indent=2), encoding="utf-8")

    manifest = {
        "input": str(input_path),
        "labels": str(labels_path) if labels_path else None,
        "predictions": str(predictions_path) if predictions_path else None,
        "outdir": str(outdir),
        "config": str(config_path),
        "rows": int(len(frame)),
        "timestamp_min": frame[str(config.get("timestamp_col") or "__ts__")].min(),
        "timestamp_max": frame[str(config.get("timestamp_col") or "__ts__")].max(),
        "symbols": int(frame[str(config.get("symbol_col") or "__symbol__")].nunique(dropna=True)),
        "source_report": source_report,
        "outcome_report": outcome_report,
        "label_join_report": label_join_report,
        "prediction_report": prediction_report,
        "row_alignment_report": row_alignment_report,
        "quality_report": quality_report,
        "proxy_learnability_report": proxy_learnability_report,
        "proxy_learnability_rows": {
            "overall": int(len(source_proxy_learnability_overall)),
            "by_month": int(len(source_proxy_learnability_by_month)),
            "by_week": int(len(source_proxy_learnability_by_week)),
            "source_x_regime": int(len(source_x_regime_proxy_learnability)),
        },
        "source_score_quality_rows": {
            "overall": int(len(source_score_quality_overall)),
            "by_month": int(len(source_score_quality_by_month)),
        },
        "source_score_target_rows": {
            "overall": int(len(source_score_target_overall)),
            "by_month": int(len(source_score_target_by_month)),
        },
        "source_archetype_promotion_scorecard_rows": int(len(source_archetype_promotion_scorecard)),
        "source_archetype_walkforward_readiness_rows": int(len(source_archetype_walkforward_readiness)),
        "failure_mode_rows": {
            "by_source": int(len(failure_mode_by_source)),
            "by_month": int(len(failure_mode_by_month)),
            "by_source_month": int(len(failure_mode_by_source_month)),
        },
        "opportunity_capture_rows": {
            "by_source": int(len(opportunity_capture_by_source)),
            "by_month": int(len(opportunity_capture_by_month)),
            "by_source_month": int(len(opportunity_capture_by_source_month)),
        },
        "capture_utility_gap_rows": {
            "overall": int(len(capture_utility_gap_overall)),
            "by_month": int(len(capture_utility_gap_by_month)),
        },
        "label_report": label_report,
        "label_variant_summary_rows": {
            "overall": int(len(label_variant_summary)),
            "by_month": int(len(label_variant_by_month)),
            "by_source": int(len(label_variant_by_source)),
        },
        "regime_columns_present": quality_report.get("regime_columns_present", []),
        "smoke_tests": smoke_tests,
        "outputs": {key: str(value) for key, value in paths.items()},
    }
    report = write_report(
        output_dir=outdir,
        frame=frame,
        source=source,
        quality_by_month=by_month,
        quality_by_week=by_week,
        quality_by_symbol=by_symbol,
        source_x_regime=source_x_regime,
        row_alignment_audit=row_alignment_audit,
        source_proxy_learnability_overall=source_proxy_learnability_overall,
        source_proxy_learnability_by_month=source_proxy_learnability_by_month,
        source_proxy_learnability_by_week=source_proxy_learnability_by_week,
        source_x_regime_proxy_learnability=source_x_regime_proxy_learnability,
        source_score_quality_overall=source_score_quality_overall,
        source_score_quality_by_month=source_score_quality_by_month,
        source_score_target_overall=source_score_target_overall,
        source_archetype_promotion_scorecard=source_archetype_promotion_scorecard,
        source_archetype_walkforward_readiness=source_archetype_walkforward_readiness,
        failure_mode_by_source=failure_mode_by_source,
        failure_mode_by_month=failure_mode_by_month,
        opportunity_capture_by_source=opportunity_capture_by_source,
        opportunity_capture_by_month=opportunity_capture_by_month,
        capture_utility_gap_overall=capture_utility_gap_overall,
        capture_utility_gap_by_month=capture_utility_gap_by_month,
        label_variant_summary=label_variant_summary,
        manifest=manifest,
        config=config,
    )
    manifest["outputs"]["source_tag_report"] = str(report)
    paths["pipeline_manifest"].write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--labels", type=Path, default=None, help="Optional realized label/outcome file or directory to join on timestamp+symbol.")
    parser.add_argument("--predictions", type=Path, default=None, help="Optional prediction/proxy score file or directory to join before diagnostics.")
    parser.add_argument("--prediction-key-cols", type=str, default=None, help="Comma-separated prediction join key columns.")
    parser.add_argument("--proxy-score-cols", type=str, default=None, help="Comma-separated proxy/model score columns to preserve and evaluate.")
    parser.add_argument("--skip-smoke-tests", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = materialize_pipeline(
        input_path=args.input,
        outdir=args.outdir,
        config_path=args.config,
        labels_path=args.labels,
        predictions_path=args.predictions,
        prediction_key_cols=_parse_csv_list(args.prediction_key_cols),
        proxy_score_cols=_parse_csv_list(args.proxy_score_cols),
        run_smokes=not bool(args.skip_smoke_tests),
    )
    print(
        json.dumps(
            _json_safe(
                {
                    "rows": manifest["rows"],
                    "symbols": manifest["symbols"],
                    "outdir": manifest["outdir"],
                    "outcome_rows": manifest["outcome_report"].get("outcome_rows"),
                    "source_columns_used": len(manifest["source_report"].get("source_columns_used", [])),
                    "smoke_tests": manifest.get("smoke_tests", {}),
                    "outputs": manifest["outputs"],
                }
            ),
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
