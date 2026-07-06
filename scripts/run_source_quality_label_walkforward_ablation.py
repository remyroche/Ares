#!/usr/bin/env python3
"""Month-forward smoke ablations for v17 source quality labels.

This is a diagnostic bridge between the source-tag label materialization and
full train_base/train_meta integration. It trains a small fixed ExtraTrees
model on prior months and scores Apr/May/Jun, then reports each source-label
variant as a delta versus a vanilla S10 policy-net label baseline.

It intentionally does not run Optuna, feature selection, LightGBM, policy
geometry optimization, or production training.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_label_feature_store_model_smoke import (  # noqa: E402
    _baseline_row,
    _fit_predict,
    _month_model_frame,
    _timestamp_ranking_metrics,
)
from scripts.run_label_quality_proxy_diagnostics import (  # noqa: E402
    DEFAULT_FEATURE_DIR,
    DEFAULT_FEATURE_LIST_CSV,
    DEFAULT_LABELS_DIR,
    _decile_diagnostics,
    _json_safe,
    _load_feature_store_columns,
    _load_labels,
    _make_targets,
    _path_metrics,
    _read_feature_list,
    _safe_mean,
    _safe_quantile,
    _selection_metrics,
    _spearman,
)


DEFAULT_SOURCE_DIR = Path("data_perp/reports/source_tags_s10_policy_net_v17_proxy_alignment_diagnostic")
DEFAULT_QUALITY_LABELS = DEFAULT_SOURCE_DIR / "quality_label_candidates.parquet"
DEFAULT_MANIFEST = DEFAULT_SOURCE_DIR / "label_ablation_manifest.json"
DEFAULT_OUTPUT_DIR = Path("data_perp/reports/source_quality_label_walkforward_ablation_v1_sideaware_20260702")
DEFAULT_ABLATIONS = (
    "baseline_all_rows",
    "economic_capture_label_v4",
    "dirty_excluded",
    "risk_adjusted_capture_candidate_only",
    "compression_capture_candidate_only",
    "source_multilabel_as_features",
)
DEFAULT_MONTHS = ("2026-04", "2026-05", "2026-06")
DEFAULT_TOP_FRACS = (0.01, 0.03, 0.05, 0.10)
DEFAULT_SEEDS = (42, 1729)
VANILLA_NAME = "vanilla_s10_policy_net_soft"
VANILLA_LABEL_ARM = "S10_policy_net_soft"


SOURCE_FEATURE_PREFIXES = (
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
)


CAUSAL_SOURCE_GATES = {
    "dirty_excluded": "train_include_dirty_excluded_v0",
    "risk_adjusted_capture_candidate_only": "train_include_risk_adjusted_capture_candidate_v4",
    "compression_capture_candidate_only": "train_include_compression_capture_candidate_v3",
}


@dataclass(frozen=True)
class AblationSpec:
    name: str
    row_filter_expression: str
    label_column: str
    sample_weight_column: str
    source_gate_column: str | None = None
    add_source_features: bool = False
    is_vanilla: bool = False


def _parse_csv(value: str | None, default: tuple[str, ...]) -> list[str]:
    if value is None or not str(value).strip():
        return list(default)
    return [part.strip() for part in str(value).split(",") if part.strip()]


def _parse_float_csv(value: str | None, default: tuple[float, ...]) -> list[float]:
    if value is None or not str(value).strip():
        return list(default)
    return [float(part.strip()) for part in str(value).split(",") if part.strip()]


def _parse_int_csv(value: str | None, default: tuple[int, ...]) -> list[int]:
    if value is None or not str(value).strip():
        return list(default)
    return [int(part.strip()) for part in str(value).split(",") if part.strip()]


def _safe_numeric(values: Any) -> pd.Series:
    if isinstance(values, pd.Series):
        return pd.to_numeric(values, errors="coerce")
    return pd.to_numeric(pd.Series(values), errors="coerce")


def _bool_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(False, index=frame.index)
    values = frame[column]
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False).astype(bool)
    lowered = values.astype(str).str.lower()
    return lowered.isin({"1", "true", "t", "yes", "y"})


def _row_filter_mask(frame: pd.DataFrame, expression: str) -> pd.Series:
    expr = str(expression or "").strip()
    if not expr:
        return pd.Series(True, index=frame.index)
    lower = expr.lower().replace(" ", "")
    if lower.endswith("==true"):
        return _bool_series(frame, expr.split("==", 1)[0].strip())
    if lower.endswith("==false"):
        return ~_bool_series(frame, expr.split("==", 1)[0].strip())
    if lower.startswith("not"):
        return ~_bool_series(frame, expr[3:].strip())
    if expr in frame.columns:
        return _bool_series(frame, expr)
    raise ValueError(f"Unsupported row_filter_expression: {expression!r}")


def _quality_target(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    if column not in frame.columns:
        raise ValueError(f"Missing quality label column: {column}")
    label = _safe_numeric(frame[column])
    target_soft = label.where(label >= 0.0, 0.5).clip(0.0, 1.0)
    target_hard = label.eq(1.0).astype(float)
    return pd.DataFrame({"target_soft": target_soft, "target_hard": target_hard}, index=frame.index)


def _quality_train_label_mask(frame: pd.DataFrame, column: str) -> pd.Series:
    label = _safe_numeric(frame[column])
    return label.isin([0.0, 1.0])


def _weight_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(1.0, index=frame.index, dtype=np.float32)
    weights = _safe_numeric(frame[column]).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return weights.clip(lower=0.0, upper=10.0).astype(np.float32)


def _load_manifest_specs(path: Path, names: list[str]) -> list[AblationSpec]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    experiments = {str(row["name"]): row for row in payload.get("experiments", [])}
    missing = sorted(set(names) - set(experiments))
    if missing:
        raise ValueError(f"Missing ablation(s) in manifest {path}: {missing}")

    specs = [
        AblationSpec(
            name=VANILLA_NAME,
            row_filter_expression="",
            label_column=VANILLA_LABEL_ARM,
            sample_weight_column="",
            is_vanilla=True,
        )
    ]
    for name in names:
        row = experiments[name]
        specs.append(
            AblationSpec(
                name=name,
                row_filter_expression=str(row.get("row_filter_expression", "")),
                label_column=str(row.get("label_column", "")),
                sample_weight_column=str(row.get("sample_weight_column", "")),
                source_gate_column=CAUSAL_SOURCE_GATES.get(name),
                add_source_features=name == "source_multilabel_as_features",
            )
        )
    return specs


def _normalise_side_join_column(frame: pd.DataFrame) -> bool:
    """Normalize any side-like column to the shared numeric join contract."""

    if "side" in frame.columns:
        raw = frame["side"]
    elif "__side__" in frame.columns:
        raw = frame["__side__"]
    elif "side_name" in frame.columns:
        raw = frame["side_name"]
    else:
        return False
    text = raw.astype(str).str.strip().str.lower()
    numeric = pd.to_numeric(raw, errors="coerce")
    side = pd.Series(np.nan, index=frame.index, dtype=np.float32)
    side.loc[text.isin({"long", "buy", "+1", "1"})] = 1.0
    side.loc[text.isin({"short", "sell", "-1"})] = -1.0
    side = side.fillna(numeric)
    valid = side.notna() & side.ne(0.0)
    if not bool(valid.any()):
        return False
    frame["side"] = np.where(side.fillna(1.0) < 0.0, -1, 1).astype(np.int8)
    frame["side_name"] = np.where(frame["side"].to_numpy(dtype=np.int8) < 0, "short", "long")
    if "__side__" not in frame.columns:
        frame["__side__"] = frame["side"]
    return True


def _normalise_string_key(frame: pd.DataFrame, column: str) -> None:
    if column in frame.columns:
        frame[column] = frame[column].astype(str)


def _side_aware_join_plan(quality: pd.DataFrame, labels: pd.DataFrame) -> tuple[list[str], str, str]:
    q_side = _normalise_side_join_column(quality)
    l_side = _normalise_side_join_column(labels)
    q_candidate = "candidate_id" in quality.columns and quality["candidate_id"].notna().any()
    l_candidate = "candidate_id" in labels.columns and labels["candidate_id"].notna().any()
    q_timeframe = "timeframe" in quality.columns and quality["timeframe"].notna().any()
    l_timeframe = "timeframe" in labels.columns and labels["timeframe"].notna().any()

    if q_candidate and l_candidate:
        _normalise_string_key(quality, "candidate_id")
        _normalise_string_key(labels, "candidate_id")
        return ["candidate_id"], "candidate_id", "one_to_one"
    if q_side and l_side and q_timeframe and l_timeframe:
        _normalise_string_key(quality, "timeframe")
        _normalise_string_key(labels, "timeframe")
        return ["__ts__", "__symbol__", "timeframe", "side"], "timestamp_symbol_timeframe_side", "one_to_one"
    if q_side and l_side:
        return ["__ts__", "__symbol__", "side"], "timestamp_symbol_side", "one_to_one"
    if q_side and not l_side:
        return ["__ts__", "__symbol__"], "timestamp_symbol_broadcast_quality_side", "many_to_one"
    if l_side and not q_side:
        return ["__ts__", "__symbol__"], "timestamp_symbol_broadcast_label_side", "one_to_many"
    return ["__ts__", "__symbol__"], "timestamp_symbol", "one_to_one"


def _label_merge_columns(labels: pd.DataFrame, quality: pd.DataFrame, key_cols: list[str]) -> list[str]:
    contract_cols = ["side", "side_name", "__side__", "timeframe", "candidate_id"]
    raw_cols = [
        "__ts__",
        "__symbol__",
        "__y_lbl__",
        "__mfe__",
        "__mae__",
        "__tp__",
        "__sl__",
        "__is_timeout__",
        "__quality__",
        "__mae_ret__",
        "__mfe_ret__",
        "__bars_to_mfe__",
        "__bars_policy__",
        "__barrier_pct__",
        "__n_tp__",
        "__n_sl__",
        "__w_consensus__",
        "__y_bin__",
        "__y_ret__",
        "__y_outcome__",
        "__w__",
        "__u_policy_net__",
        "__r_policy_net__",
    ]
    cols: list[str] = []
    for col in [*key_cols, *contract_cols, *raw_cols]:
        if col not in labels.columns or col in cols:
            continue
        if col not in key_cols and col in quality.columns:
            continue
        cols.append(col)
    return cols


def _dedupe_join_side(
    frame: pd.DataFrame,
    key_cols: list[str],
    *,
    require_unique: bool,
) -> tuple[pd.DataFrame, int]:
    dupes = int(frame.duplicated(key_cols).sum())
    if require_unique and dupes:
        frame = frame.sort_values(key_cols, kind="mergesort").drop_duplicates(key_cols, keep="last")
    return frame, dupes


def _load_joined_frame(
    *,
    quality_labels_path: Path,
    labels_path: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    quality = pd.read_parquet(quality_labels_path)
    labels = _load_labels(labels_path)
    if "__ts__" not in quality.columns or "__symbol__" not in quality.columns:
        raise ValueError(f"{quality_labels_path} must include __ts__ and __symbol__")
    quality["__ts__"] = pd.to_datetime(quality["__ts__"], utc=True, errors="coerce")
    labels["__ts__"] = pd.to_datetime(labels["__ts__"], utc=True, errors="coerce")

    key_cols, join_mode, validate = _side_aware_join_plan(quality, labels)
    raw_cols = _label_merge_columns(labels, quality, key_cols)
    label_ts_symbol_dupes = int(labels.duplicated(["__ts__", "__symbol__"]).sum())
    quality_ts_symbol_dupes = int(quality.duplicated(["__ts__", "__symbol__"]).sum())
    quality, quality_dupes = _dedupe_join_side(
        quality,
        key_cols,
        require_unique=validate in {"one_to_one", "one_to_many"},
    )
    labels, label_dupes = _dedupe_join_side(
        labels,
        key_cols,
        require_unique=validate in {"one_to_one", "many_to_one"},
    )

    joined = quality.merge(
        labels[raw_cols],
        on=key_cols,
        how="inner",
        validate=validate,
    )
    joined = joined.sort_values(key_cols, kind="mergesort").reset_index(drop=True)
    report = {
        "quality_rows": int(len(quality)),
        "label_rows": int(len(labels)),
        "joined_rows": int(len(joined)),
        "join_key": key_cols,
        "join_mode": join_mode,
        "merge_validate": validate,
        "side_join_used": "side" in key_cols or join_mode.startswith("timestamp_symbol_broadcast"),
        "quality_duplicate_timestamp_symbol_rows": quality_ts_symbol_dupes,
        "label_duplicate_timestamp_symbol_rows": label_ts_symbol_dupes,
        "quality_duplicate_join_key_rows": quality_dupes,
        "label_duplicate_join_key_rows": label_dupes,
        "join_match_rate_vs_quality": float(len(joined) / len(quality)) if len(quality) else 0.0,
        "join_match_rate_vs_labels": float(len(joined) / len(labels)) if len(labels) else 0.0,
    }
    return joined, report


def _load_ablation_frame(
    *,
    quality_labels_path: Path,
    labels_path: Path,
    joined_subset_path: Path | None = None,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    if joined_subset_path is None:
        frame, join_report = _load_joined_frame(
            quality_labels_path=quality_labels_path,
            labels_path=labels_path,
        )
        return frame, join_report, {
            "mode": "quality_labels_joined_to_labels",
            "quality_labels_path": str(quality_labels_path),
            "labels_path": str(labels_path),
            "joined_subset_path": None,
        }

    if not joined_subset_path.exists():
        raise FileNotFoundError(joined_subset_path)
    frame = pd.read_parquet(joined_subset_path)
    if "__ts__" not in frame.columns or "__symbol__" not in frame.columns:
        raise ValueError(f"{joined_subset_path} must include __ts__ and __symbol__")
    required_outcome_cols = ["__barrier_pct__", "__mfe_ret__", "__mae_ret__", "__u_policy_net__"]
    missing_outcome_cols = [col for col in required_outcome_cols if col not in frame.columns]
    if missing_outcome_cols:
        raise ValueError(
            f"{joined_subset_path} is not a joined label subset; missing outcome columns: {missing_outcome_cols}"
        )
    frame["__ts__"] = pd.to_datetime(frame["__ts__"], utc=True, errors="coerce")
    manifest_path = joined_subset_path.parent / "manifest.json"
    subset_manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    stored_join_report = subset_manifest.get("join_report", {}) if isinstance(subset_manifest, dict) else {}
    join_report = {
        "quality_rows": int(stored_join_report.get("quality_rows", len(frame))),
        "label_rows": int(stored_join_report.get("label_rows", len(frame))),
        "joined_rows": int(len(frame)),
        "join_key": stored_join_report.get("join_key", ["prejoined_subset"]),
        "join_mode": "prejoined_subset",
        "merge_validate": "prejoined_subset",
        "side_join_used": bool(stored_join_report.get("side_join_used", "side" in frame.columns)),
        "quality_duplicate_timestamp_symbol_rows": int(
            stored_join_report.get("quality_duplicate_timestamp_symbol_rows", 0)
        ),
        "label_duplicate_timestamp_symbol_rows": int(
            stored_join_report.get("label_duplicate_timestamp_symbol_rows", 0)
        ),
        "quality_duplicate_join_key_rows": int(stored_join_report.get("quality_duplicate_join_key_rows", 0)),
        "label_duplicate_join_key_rows": int(stored_join_report.get("label_duplicate_join_key_rows", 0)),
        "join_match_rate_vs_quality": float(stored_join_report.get("join_match_rate_vs_quality", 1.0)),
        "join_match_rate_vs_labels": float(stored_join_report.get("join_match_rate_vs_labels", 1.0)),
    }
    if "candidate_id" in frame.columns:
        join_report["duplicate_candidate_id_rows"] = int(frame["candidate_id"].astype(str).duplicated().sum())
    input_report = {
        "mode": "prejoined_clean_subset",
        "quality_labels_path": str(quality_labels_path),
        "labels_path": str(labels_path),
        "joined_subset_path": str(joined_subset_path),
        "joined_subset_manifest": str(manifest_path) if manifest_path.exists() else None,
        "joined_subset_status": subset_manifest.get("subset_status") if isinstance(subset_manifest, dict) else None,
        "joined_subset_overall_status": subset_manifest.get("overall_status") if isinstance(subset_manifest, dict) else None,
        "joined_subset_warnings": subset_manifest.get("warnings", []) if isinstance(subset_manifest, dict) else [],
    }
    return frame, join_report, input_report


def _source_feature_columns(frame: pd.DataFrame) -> list[str]:
    cols: list[str] = []
    for col in frame.columns:
        if col.startswith("tag_"):
            cols.append(col)
        elif col in SOURCE_FEATURE_PREFIXES:
            cols.append(col)
        elif col.startswith("__regime_") or col == "G_VOL":
            cols.append(col)
    return cols


def _target_for_spec(
    *,
    spec: AblationSpec,
    frame: pd.DataFrame,
    vanilla_targets: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    if spec.is_vanilla:
        return vanilla_targets[VANILLA_LABEL_ARM].reindex(frame.index)
    return _quality_target(frame, spec.label_column)


def _training_mask_for_spec(
    *,
    spec: AblationSpec,
    frame: pd.DataFrame,
    month: str,
    train_lookback_months: int | None,
) -> pd.Series:
    month_period = frame["__ts__"].dt.to_period("M").astype(str)
    train = month_period < str(month)
    if train_lookback_months is not None and int(train_lookback_months) > 0:
        prior_months = sorted(month_period[train].dropna().unique())
        keep = set(prior_months[-int(train_lookback_months) :])
        train = train & month_period.isin(keep)
    if spec.is_vanilla:
        return train
    return (
        train
        & _row_filter_mask(frame, spec.row_filter_expression)
        & _quality_train_label_mask(frame, spec.label_column)
        & _weight_series(frame, spec.sample_weight_column).gt(0.0)
    )


def _eval_scopes_for_spec(
    *,
    spec: AblationSpec,
    source_gate_columns: dict[str, str],
) -> list[tuple[str, str, pd.Series | None]]:
    scopes: list[tuple[str, str, pd.Series | None]] = [("all_rows", "", None)]
    if spec.is_vanilla:
        for gate_name, gate_col in source_gate_columns.items():
            scopes.append((f"source_gate::{gate_name}", gate_col, None))
    elif spec.source_gate_column:
        scopes.append((f"source_gate::{spec.name}", spec.source_gate_column, None))
    return scopes


def _selected_source_buckets(
    *,
    selected_frame: pd.DataFrame,
    selected_metrics: pd.DataFrame,
    row_context: dict[str, Any],
) -> list[dict[str, Any]]:
    if selected_frame.empty:
        return []
    rows: list[dict[str, Any]] = []
    bucket_cols = ["primary_source_tag"] + [col for col in selected_frame.columns if col.startswith("tag_")]
    for col in bucket_cols:
        if col not in selected_frame.columns:
            continue
        if col == "primary_source_tag":
            groups = selected_frame[col].astype(str)
        else:
            groups = _bool_series(selected_frame, col).map({True: col.replace("tag_", ""), False: "__not_selected__"})
        for bucket, idx in groups.groupby(groups, dropna=False).groups.items():
            if bucket == "__not_selected__":
                continue
            metrics = selected_metrics.loc[idx]
            rows.append(
                {
                    **row_context,
                    "bucket_col": col,
                    "bucket": str(bucket),
                    "bucket_rows": int(len(metrics)),
                    "bucket_mean_u": _safe_mean(metrics["u_policy_net"]),
                    "bucket_hit_u": _safe_mean(metrics["u_policy_net"] > 0.0),
                    "bucket_bad_mae_1r_rate": _safe_mean(metrics["mae_norm"] >= 1.0),
                    "bucket_p90_mae_norm": _safe_quantile(metrics["mae_norm"], 0.90),
                    "bucket_timeout_rate": _safe_mean(metrics["is_timeout"].astype(float)),
                    "bucket_wide_barrier_25bps_rate": _safe_mean(metrics["barrier"] > 0.025),
                }
            )
    return rows


def _score_month(
    *,
    spec: AblationSpec,
    frame: pd.DataFrame,
    metrics: pd.DataFrame,
    vanilla_targets: dict[str, pd.DataFrame],
    base_features: list[str],
    source_features: list[str],
    month: str,
    top_fracs: list[float],
    seeds: list[int],
    train_lookback_months: int | None,
    min_train_rows: int,
    min_valid_rows: int,
    source_gate_columns: dict[str, str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    month_period = frame["__ts__"].dt.to_period("M").astype(str)
    valid_mask = month_period == str(month)
    train_mask = _training_mask_for_spec(
        spec=spec,
        frame=frame,
        month=month,
        train_lookback_months=train_lookback_months,
    )
    if int(train_mask.sum()) < int(min_train_rows) or int(valid_mask.sum()) < int(min_valid_rows):
        return [], [
            {
                "ablation": spec.name,
                "period": month,
                "skipped": True,
                "reason": "too_few_rows",
                "train_rows": int(train_mask.sum()),
                "valid_rows": int(valid_mask.sum()),
            }
        ], []

    features = list(base_features)
    if spec.add_source_features:
        features = list(dict.fromkeys(features + source_features))
    x_train, x_valid = _month_model_frame(
        frame,
        train_mask=train_mask,
        valid_mask=valid_mask,
        features=features,
    )
    target = _target_for_spec(spec=spec, frame=frame, vanilla_targets=vanilla_targets)
    target_train = target.loc[train_mask].copy()
    if spec.is_vanilla:
        weights = _safe_numeric(frame.loc[train_mask, "__w__"] if "__w__" in frame.columns else 1.0).fillna(1.0)
    else:
        weights = _weight_series(frame, spec.sample_weight_column).loc[train_mask]
    pred_matrix = np.vstack(
        [
            _fit_predict(
                x_train=x_train,
                y_train=target_train["target_soft"],
                w_train=weights,
                x_valid=x_valid,
                seed=seed,
            )
            for seed in seeds
        ]
    )
    score_all = pd.Series(np.mean(pred_matrix, axis=0).astype(np.float32), index=frame.loc[valid_mask].index)
    pred_seed_std = float(np.std(pred_matrix, axis=0).mean()) if pred_matrix.size else float("nan")

    valid_frame_all = frame.loc[valid_mask].copy()
    valid_metrics_all = metrics.loc[valid_mask].copy()
    valid_target_all = target.loc[valid_mask].copy()
    diagnostics = [
        {
            "ablation": spec.name,
            "period": month,
            "skipped": False,
            "train_rows": int(train_mask.sum()),
            "valid_rows": int(valid_mask.sum()),
            "train_positive_rate": _safe_mean(target_train["target_hard"] > 0.5),
            "train_target_soft_mean": _safe_mean(target_train["target_soft"]),
            "train_weight_mean": _safe_mean(weights),
            "model_feature_count": int(len(features)),
            "source_features_added": bool(spec.add_source_features),
            "prediction_seed_std_mean": pred_seed_std,
            "score_ic_u_all_valid": _spearman(score_all, valid_metrics_all["u_policy_net"]),
            "score_ic_label_all_valid": _spearman(score_all, valid_target_all["target_soft"]),
            "row_filter_expression": spec.row_filter_expression,
            "label_column": spec.label_column,
            "sample_weight_column": spec.sample_weight_column,
            "source_gate_column": spec.source_gate_column or "",
        }
    ]

    monthly_rows: list[dict[str, Any]] = []
    bucket_rows: list[dict[str, Any]] = []
    for eval_scope, gate_col, _unused in _eval_scopes_for_spec(spec=spec, source_gate_columns=source_gate_columns):
        if gate_col:
            scope_mask = _bool_series(valid_frame_all, gate_col)
            valid_frame = valid_frame_all.loc[scope_mask].reset_index(drop=True)
            valid_metrics = valid_metrics_all.loc[scope_mask].reset_index(drop=True)
            valid_target = valid_target_all.loc[scope_mask].reset_index(drop=True)
            score = score_all.loc[scope_mask[scope_mask].index].reset_index(drop=True)
        else:
            valid_frame = valid_frame_all.reset_index(drop=True)
            valid_metrics = valid_metrics_all.reset_index(drop=True)
            valid_target = valid_target_all.reset_index(drop=True)
            score = score_all.reset_index(drop=True)
        if int(len(valid_frame)) < int(min_valid_rows):
            continue
        period_baseline = _baseline_row(valid_metrics)
        decile = _decile_diagnostics(score, valid_metrics["u_policy_net"])
        ts_rank = _timestamp_ranking_metrics(
            frame=valid_frame,
            metrics=valid_metrics,
            target=valid_target,
            score=score,
        )
        for top_frac in top_fracs:
            row = _selection_metrics(
                frame=valid_frame,
                metrics=valid_metrics,
                target=valid_target,
                score=score,
                arm=spec.name,
                selector="source_quality_label_extra_trees_month_forward",
                period=month,
                top_frac=top_frac,
            )
            row.update(period_baseline)
            row["delta_mean_u_vs_period"] = row["mean_u"] - period_baseline["period_baseline_mean_u"]
            row["delta_hit_u_vs_period"] = row["hit_u"] - period_baseline["period_baseline_hit_u"]
            row["delta_q10_u_vs_period"] = row["q10_u"] - period_baseline["period_baseline_q10_u"]
            row.update(
                {
                    "ablation": spec.name,
                    "label_column": spec.label_column,
                    "sample_weight_column": spec.sample_weight_column,
                    "eval_scope": eval_scope,
                    "gate_column": gate_col,
                    "train_rows": int(train_mask.sum()),
                    "valid_scope_rows": int(len(valid_frame)),
                    "model_feature_count": int(len(features)),
                    "source_features_added": bool(spec.add_source_features),
                    "score_ic_u": _spearman(score, valid_metrics["u_policy_net"]),
                    "score_ic_label": _spearman(score, valid_target["target_soft"]),
                    **decile,
                    **ts_rank,
                }
            )
            monthly_rows.append(row)

            selected_idx = np.asarray(
                _rank_top_indices_local(score, top_frac),
                dtype=np.int64,
            )
            selected_frame = valid_frame.iloc[selected_idx].copy() if len(selected_idx) else valid_frame.iloc[:0].copy()
            selected_metrics = valid_metrics.iloc[selected_idx].copy() if len(selected_idx) else valid_metrics.iloc[:0].copy()
            bucket_rows.extend(
                _selected_source_buckets(
                    selected_frame=selected_frame,
                    selected_metrics=selected_metrics,
                    row_context={
                        "ablation": spec.name,
                        "period": month,
                        "top_frac": float(top_frac),
                        "eval_scope": eval_scope,
                        "gate_column": gate_col,
                    },
                )
            )
    return monthly_rows, diagnostics, bucket_rows


def _rank_top_indices_local(score: Any, frac: float) -> np.ndarray:
    score_ser = _safe_numeric(score)
    valid = score_ser.notna().to_numpy()
    if not bool(valid.any()):
        return np.array([], dtype=np.int64)
    valid_idx = np.flatnonzero(valid)
    k = max(1, int(math.ceil(float(frac) * len(valid_idx))))
    order = np.argsort(-score_ser.iloc[valid_idx].to_numpy(dtype=np.float64), kind="mergesort")
    return valid_idx[order[:k]].astype(np.int64, copy=False)


def _add_vanilla_deltas(monthly: pd.DataFrame) -> pd.DataFrame:
    if monthly.empty:
        return monthly
    keys = ["period", "top_frac", "eval_scope", "gate_column"]
    vanilla = monthly[monthly["ablation"].eq(VANILLA_NAME)].copy()
    baseline_cols = [
        "mean_u",
        "hit_u",
        "q10_u",
        "bad_mae_1r_rate",
        "p90_mae_norm",
        "timeout_rate",
        "wide_barrier_25bps_rate",
        "ts_rank_hr10_u",
        "ts_rank_hr30_u",
        "ts_rank_ndcg30_u",
        "ts_rank_week_hr30_q10",
        "ts_rank_week_hr30_q25",
        "ts_rank_top30_bad_mae_1r_rate",
        "ts_rank_top30_timeout_rate",
    ]
    vanilla = vanilla[keys + baseline_cols].rename(
        columns={col: f"vanilla_{col}" for col in baseline_cols}
    )
    out = monthly.merge(vanilla, on=keys, how="left", validate="many_to_one")
    for col in baseline_cols:
        out[f"delta_{col}_vs_vanilla"] = (
            pd.to_numeric(out[col], errors="coerce")
            - pd.to_numeric(out[f"vanilla_{col}"], errors="coerce")
        )
    return out


def _aggregate(monthly: pd.DataFrame, *, expected_months: int) -> pd.DataFrame:
    if monthly.empty:
        return monthly
    rows: list[dict[str, Any]] = []
    group_cols = ["ablation", "eval_scope", "gate_column", "top_frac"]
    for key, group in monthly.groupby(group_cols, dropna=False, observed=True):
        ablation, eval_scope, gate_column, top_frac = key
        mean_u = _safe_numeric(group["mean_u"])
        delta = _safe_numeric(group.get("delta_mean_u_vs_vanilla"))
        month_count = int(group["period"].nunique())
        positive_delta_months = int((delta > 0.0).sum())
        worst_month_mean_u = float(mean_u.min()) if len(mean_u.dropna()) else float("nan")
        mean_delta = _safe_mean(group.get("delta_mean_u_vs_vanilla"))
        mean_bad_mae = _safe_mean(group["bad_mae_1r_rate"])
        mean_vanilla_bad_mae = _safe_mean(group["vanilla_bad_mae_1r_rate"])
        mean_timeout = _safe_mean(group["timeout_rate"])
        mean_vanilla_timeout = _safe_mean(group["vanilla_timeout_rate"])
        mean_wide_barrier = _safe_mean(group["wide_barrier_25bps_rate"])
        mean_vanilla_wide_barrier = _safe_mean(group["vanilla_wide_barrier_25bps_rate"])
        promote = (
            ablation != VANILLA_NAME
            and month_count >= int(expected_months)
            and positive_delta_months >= int(expected_months)
            and math.isfinite(mean_delta)
            and mean_delta > 0.0
            and math.isfinite(worst_month_mean_u)
            and worst_month_mean_u > 0.0
            and mean_bad_mae <= mean_vanilla_bad_mae
            and mean_timeout <= mean_vanilla_timeout
            and mean_wide_barrier <= mean_vanilla_wide_barrier
        )
        rows.append(
            {
                "ablation": ablation,
                "eval_scope": eval_scope,
                "gate_column": gate_column,
                "top_frac": float(top_frac),
                "months": month_count,
                "positive_months": int((mean_u > 0.0).sum()),
                "positive_delta_months_vs_vanilla": positive_delta_months,
                "mean_u": _safe_mean(mean_u),
                "worst_month_mean_u": worst_month_mean_u,
                "hit_u": _safe_mean(group["hit_u"]),
                "q10_u": _safe_mean(group["q10_u"]),
                "delta_mean_u_vs_vanilla": mean_delta,
                "delta_hit_u_vs_vanilla": _safe_mean(group.get("delta_hit_u_vs_vanilla")),
                "delta_q10_u_vs_vanilla": _safe_mean(group.get("delta_q10_u_vs_vanilla")),
                "score_ic_u": _safe_mean(group["score_ic_u"]),
                "score_ic_label": _safe_mean(group["score_ic_label"]),
                "ts_rank_hr10_u": _safe_mean(group["ts_rank_hr10_u"]),
                "ts_rank_hr30_u": _safe_mean(group["ts_rank_hr30_u"]),
                "delta_ts_rank_hr30_u_vs_vanilla": _safe_mean(group.get("delta_ts_rank_hr30_u_vs_vanilla")),
                "ts_rank_ndcg30_u": _safe_mean(group["ts_rank_ndcg30_u"]),
                "delta_ts_rank_ndcg30_u_vs_vanilla": _safe_mean(group.get("delta_ts_rank_ndcg30_u_vs_vanilla")),
                "weekly_q10_hr30": _safe_mean(group["ts_rank_week_hr30_q10"]),
                "weekly_q25_hr30": _safe_mean(group["ts_rank_week_hr30_q25"]),
                "bad_mae_1r_rate": _safe_mean(group["bad_mae_1r_rate"]),
                "delta_bad_mae_1r_rate_vs_vanilla": _safe_mean(group.get("delta_bad_mae_1r_rate_vs_vanilla")),
                "p90_mae_norm": _safe_mean(group["p90_mae_norm"]),
                "timeout_rate": _safe_mean(group["timeout_rate"]),
                "delta_timeout_rate_vs_vanilla": _safe_mean(group.get("delta_timeout_rate_vs_vanilla")),
                "wide_barrier_25bps_rate": _safe_mean(group["wide_barrier_25bps_rate"]),
                "delta_wide_barrier_25bps_rate_vs_vanilla": _safe_mean(group.get("delta_wide_barrier_25bps_rate_vs_vanilla")),
                "mean_selected_rows": _safe_mean(group["selected_rows"]),
                "min_selected_rows": int(pd.to_numeric(group["selected_rows"], errors="coerce").min()),
                "decision": "candidate_for_full_train_ablation" if promote else "diagnostic_only",
            }
        )
    out = pd.DataFrame(rows)
    return out.sort_values(
        ["top_frac", "delta_mean_u_vs_vanilla", "mean_u"],
        ascending=[True, False, False],
        na_position="last",
    )


def _write_markdown(output_dir: Path, aggregate: pd.DataFrame, manifest: dict[str, Any]) -> Path:
    path = output_dir / "source_quality_label_walkforward_ablation.md"

    def table(frame: pd.DataFrame, cols: list[str], limit: int | None = None) -> str:
        if frame.empty:
            return "No rows."
        view = frame[[col for col in cols if col in frame.columns]].copy()
        if limit is not None:
            view = view.head(limit)
        for col in view.columns:
            if pd.api.types.is_float_dtype(view[col]):
                view[col] = view[col].map(lambda value: f"{float(value):.4f}" if pd.notna(value) else "")
        return view.to_markdown(index=False)

    cols = [
        "decision",
        "ablation",
        "eval_scope",
        "top_frac",
        "months",
        "positive_months",
        "positive_delta_months_vs_vanilla",
        "mean_u",
        "delta_mean_u_vs_vanilla",
        "worst_month_mean_u",
        "hit_u",
        "ts_rank_hr30_u",
        "delta_ts_rank_hr30_u_vs_vanilla",
        "ts_rank_ndcg30_u",
        "weekly_q10_hr30",
        "bad_mae_1r_rate",
        "delta_bad_mae_1r_rate_vs_vanilla",
        "p90_mae_norm",
        "timeout_rate",
        "wide_barrier_25bps_rate",
        "mean_selected_rows",
    ]
    promoted = aggregate[aggregate["decision"].eq("candidate_for_full_train_ablation")]
    lines = [
        "# Source Quality Label Walk-Forward Ablation",
        "",
        "Scope: diagnostic ExtraTrees month-forward smoke over v17 source-quality labels. This is not production LightGBM training.",
        "",
        f"Rows joined to outcome labels: `{manifest['rows']}`",
        f"Utility source: `{manifest.get('utility_source', '')}`",
        f"Months: `{', '.join(manifest['months'])}`",
        f"Vanilla baseline: `{VANILLA_NAME}`",
        "",
        "## Candidate For Full Train Ablation",
        "",
        table(promoted, cols, limit=50),
        "",
        "## All Results",
        "",
        table(aggregate, cols, limit=120),
        "",
        "## Alignment",
        "",
        f"- Quality rows: `{manifest['join_report']['quality_rows']}`",
        f"- Label rows: `{manifest['join_report']['label_rows']}`",
        f"- Joined rows: `{manifest['join_report']['joined_rows']}`",
        f"- Join match vs labels: `{manifest['join_report']['join_match_rate_vs_labels']:.4f}`",
        "",
        "## Outputs",
        "",
        f"- Monthly: `{manifest['outputs']['monthly']}`",
        f"- Aggregate: `{manifest['outputs']['aggregate']}`",
        f"- Diagnostics: `{manifest['outputs']['diagnostics']}`",
        f"- Failure buckets: `{manifest['outputs']['failure_buckets']}`",
        f"- Manifest: `{manifest['outputs']['manifest']}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_ablation(
    *,
    quality_labels_path: Path,
    labels_path: Path,
    joined_subset_path: Path | None,
    manifest_path: Path,
    output_dir: Path,
    feature_dir: Path,
    feature_list_csv: Path,
    max_feature_store_features: int | None,
    ablations: list[str],
    months: list[str],
    top_fracs: list[float],
    seeds: list[int],
    train_lookback_months: int | None,
    min_train_rows: int,
    min_valid_rows: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    specs = _load_manifest_specs(manifest_path, ablations)
    frame, join_report, input_report = _load_ablation_frame(
        quality_labels_path=quality_labels_path,
        labels_path=labels_path,
        joined_subset_path=joined_subset_path,
    )
    selected_features = _read_feature_list(feature_list_csv, max_features=max_feature_store_features)
    feature_matrix, feature_report = _load_feature_store_columns(
        frame,
        feature_dir=feature_dir,
        selected_features=selected_features,
    )
    for col in feature_matrix.columns:
        frame[col] = feature_matrix[col].to_numpy(dtype=np.float32, copy=False)
    metrics = _path_metrics(frame)
    vanilla_targets = _make_targets(frame, metrics)
    base_features = list(feature_matrix.columns)
    source_features = _source_feature_columns(frame)
    source_gate_columns = {name: col for name, col in CAUSAL_SOURCE_GATES.items() if name in ablations and col in frame.columns}

    monthly_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    bucket_rows: list[dict[str, Any]] = []
    for month in months:
        for spec in specs:
            rows, diagnostics, buckets = _score_month(
                spec=spec,
                frame=frame,
                metrics=metrics,
                vanilla_targets=vanilla_targets,
                base_features=base_features,
                source_features=source_features,
                month=month,
                top_fracs=top_fracs,
                seeds=seeds,
                train_lookback_months=train_lookback_months,
                min_train_rows=min_train_rows,
                min_valid_rows=min_valid_rows,
                source_gate_columns=source_gate_columns,
            )
            monthly_rows.extend(rows)
            diagnostic_rows.extend(diagnostics)
            bucket_rows.extend(buckets)

    monthly = _add_vanilla_deltas(pd.DataFrame(monthly_rows))
    aggregate = _aggregate(monthly, expected_months=len(months))
    diagnostics = pd.DataFrame(diagnostic_rows)
    failure_buckets = pd.DataFrame(bucket_rows)

    paths = {
        "monthly": output_dir / "source_quality_label_walkforward_monthly.csv",
        "aggregate": output_dir / "source_quality_label_walkforward_aggregate.csv",
        "diagnostics": output_dir / "source_quality_label_walkforward_diagnostics.csv",
        "failure_buckets": output_dir / "source_quality_label_failure_buckets_by_source.csv",
        "manifest": output_dir / "manifest.json",
    }
    monthly.to_csv(paths["monthly"], index=False)
    aggregate.to_csv(paths["aggregate"], index=False)
    diagnostics.to_csv(paths["diagnostics"], index=False)
    failure_buckets.to_csv(paths["failure_buckets"], index=False)
    manifest = {
        "scope": "diagnostic_source_quality_label_month_forward_smoke_not_full_training",
        "input_report": input_report,
        "quality_labels_path": str(quality_labels_path),
        "labels_path": str(labels_path),
        "joined_subset_path": str(joined_subset_path) if joined_subset_path else None,
        "label_ablation_manifest": str(manifest_path),
        "output_dir": str(output_dir),
        "rows": int(len(frame)),
        "utility_source": metrics.attrs.get("utility_source"),
        "timestamp_min": frame["__ts__"].min(),
        "timestamp_max": frame["__ts__"].max(),
        "symbols": int(frame["__symbol__"].nunique(dropna=True)),
        "months": months,
        "ablations": [spec.name for spec in specs],
        "source_gate_columns": source_gate_columns,
        "top_fracs": [float(v) for v in top_fracs],
        "join_report": join_report,
        "feature_store": feature_report,
        "base_feature_count": int(len(base_features)),
        "source_feature_count": int(len(source_features)),
        "source_features": source_features,
        "model": {
            "type": "ExtraTreesRegressor",
            "seeds": [int(seed) for seed in seeds],
            "seed_count": int(len(seeds)),
            "train_lookback_months": int(train_lookback_months)
            if train_lookback_months is not None
            else None,
            "fixed_hpo": True,
            "feature_selection": "frozen_feature_list_csv_plus_optional_v17_source_features",
        },
        "outputs": {key: str(value) for key, value in paths.items()},
    }
    paths["manifest"].write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")
    markdown = _write_markdown(output_dir, aggregate, manifest)
    manifest["outputs"]["markdown"] = str(markdown)
    paths["manifest"].write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quality-labels-path", type=Path, default=DEFAULT_QUALITY_LABELS)
    parser.add_argument("--labels-path", type=Path, default=DEFAULT_LABELS_DIR)
    parser.add_argument(
        "--joined-subset-path",
        type=Path,
        default=None,
        help="Optional prejoined clean subset parquet. When set, skips joining quality labels to labels_path.",
    )
    parser.add_argument("--label-ablation-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--feature-list-csv", type=Path, default=DEFAULT_FEATURE_LIST_CSV)
    parser.add_argument("--max-feature-store-features", type=int, default=96)
    parser.add_argument("--ablations", type=str, default=",".join(DEFAULT_ABLATIONS))
    parser.add_argument("--months", type=str, default=",".join(DEFAULT_MONTHS))
    parser.add_argument("--top-fracs", type=str, default=",".join(str(v) for v in DEFAULT_TOP_FRACS))
    parser.add_argument("--seeds", type=str, default=",".join(str(v) for v in DEFAULT_SEEDS))
    parser.add_argument("--train-lookback-months", type=int, default=None)
    parser.add_argument("--min-train-rows", type=int, default=500)
    parser.add_argument("--min-valid-rows", type=int, default=100)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = run_ablation(
        quality_labels_path=args.quality_labels_path,
        labels_path=args.labels_path,
        joined_subset_path=args.joined_subset_path,
        manifest_path=args.label_ablation_manifest,
        output_dir=args.output_dir,
        feature_dir=args.feature_dir,
        feature_list_csv=args.feature_list_csv,
        max_feature_store_features=args.max_feature_store_features,
        ablations=_parse_csv(args.ablations, DEFAULT_ABLATIONS),
        months=_parse_csv(args.months, DEFAULT_MONTHS),
        top_fracs=_parse_float_csv(args.top_fracs, DEFAULT_TOP_FRACS),
        seeds=_parse_int_csv(args.seeds, DEFAULT_SEEDS),
        train_lookback_months=args.train_lookback_months,
        min_train_rows=int(args.min_train_rows),
        min_valid_rows=int(args.min_valid_rows),
    )
    print(json.dumps(_json_safe(manifest), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
