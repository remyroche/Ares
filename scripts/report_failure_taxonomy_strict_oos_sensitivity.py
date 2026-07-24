#!/usr/bin/env python3
"""Leakage-honest sensitivity report for the failure-taxonomy research path.

This is deliberately a *read-only* report.  It keeps the genuine expanding
base-OOS ledger and genuine base+meta-OOS handoff separate, and treats the
three-year failure taxonomy as descriptive context only.  In particular, the
taxonomy was built from a frozen diagnostic backcast, so it is never presented
as a strict-OOS classifier or a source of strict-OOS targets here.

The report uses timestamp/symbol/side/archetype keys normalized to UTC.  It
does not train, calibrate, threshold, or mutate any production artifact.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import shutil
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import average_precision_score, brier_score_loss


DEFAULT_BASE_LEDGER = Path(
    "data_perp/reports/"
    "s59_h5_benchmark66_matchedaegmm_refit_wf30_20260716_v1/"
    "best_oos_scored_ledger.parquet"
)
DEFAULT_META_LEDGER = Path(
    "data_perp/reports/meta_v9_recovery_20260717/"
    "residual_state_mda95_hier_newaegmm_hpo150_v1/oos_predictions.parquet"
)
DEFAULT_TAXONOMY = Path(
    "data_perp/reports/failure_episode_taxonomy_20260719_v17_three_year_taxonomy"
)
DEFAULT_DETECTOR = Path(
    "data_perp/reports/prospective_failure_mode_detection_20260719_v7_three_year"
    "/local_oos_predictions.parquet"
)
DEFAULT_OUTPUT = Path(
    "data_perp/reports/failure_taxonomy_strict_oos_sensitivity_20260719_v1"
)
KEYS = ("__ts__", "__symbol__", "side_name", "archetype_policy_key")
TOP_K = (0.10, 0.20, 0.30)
BASE_PROJECTION = (
    "__ts__",
    "__symbol__",
    "side_name",
    "__archetype_policy_key__",
    "score",
    "__u_policy_net__",
    "__long_path_clean_exec_label__",
    "__long_path_dirty_positive_label__",
    "__path_full_bad_mae_1r__",
    "__first_touch_timeout__",
    "__first_touch_stop__",
    "oos_fold",
)
META_PROJECTION = (
    "__ts__",
    "__symbol__",
    "side_name",
    "archetype_policy_key",
    "score_base",
    "score",
    "ev_after_1pct",
    "clean_exec",
    "dirty_positive",
    "full_path_bad_mae_1r",
    "timeout",
    "score_base_ev_mapped",
    "score_base_ev_residual_expert",
    "score_base_ev_residual_expert_hier_mapped",
    "meta_residual_expert_delta_ev",
    "score_base_rank",
    "score_base_ev_rank_train_reference",
    "score_base_residual_ev_rank_train_reference",
)


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Unsupported JSON value: {type(value).__name__}")


def _utc(values: pd.Series) -> pd.Series:
    """Interpret legacy naive timestamps as UTC and preserve real UTC instants."""

    return pd.to_datetime(values, utc=True, errors="coerce")


def _available_columns(path: Path, requested: Iterable[str]) -> list[str]:
    names = set(pq.ParquetFile(path).schema.names)
    return [name for name in requested if name in names]


def _read_projection(path: Path, requested: Iterable[str]) -> pd.DataFrame:
    columns = _available_columns(path, requested)
    if "__ts__" not in columns or "__symbol__" not in columns:
        raise ValueError(f"{path} must contain __ts__ and __symbol__")
    return pd.read_parquet(path, columns=columns)


def _read_projection_window(
    path: Path,
    requested: Iterable[str],
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    """Read a bounded UTC interval without retaining the full base ledger."""

    columns = _available_columns(path, requested)
    source = pq.ParquetFile(path)
    pieces: list[pd.DataFrame] = []
    for batch in source.iter_batches(columns=columns, batch_size=100_000):
        part = batch.to_pandas()
        timestamp = _utc(part["__ts__"])
        keep = timestamp.ge(start) & timestamp.le(end)
        if keep.any():
            pieces.append(part.loc[keep].copy())
    if not pieces:
        return pd.DataFrame(columns=columns)
    return pd.concat(pieces, ignore_index=True, copy=False)


def _as_bool(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False).astype(bool)
    return pd.to_numeric(values, errors="coerce").fillna(0.0).ne(0.0)


def _finite(values: Any) -> np.ndarray:
    return np.isfinite(pd.to_numeric(values, errors="coerce").to_numpy(np.float64))


def _safe_mean(frame: pd.DataFrame, column: str) -> float:
    if column not in frame:
        return np.nan
    values = pd.to_numeric(frame[column], errors="coerce")
    return float(values.mean()) if values.notna().any() else np.nan


def _safe_rate(frame: pd.DataFrame, column: str) -> float:
    if column not in frame:
        return np.nan
    values = pd.to_numeric(frame[column], errors="coerce")
    return float(values.mean()) if values.notna().any() else np.nan


def _safe_spearman(left: pd.Series, right: pd.Series) -> float:
    pair = pd.DataFrame({"left": left, "right": right}).dropna()
    return float(pair.corr(method="spearman").iloc[0, 1]) if len(pair) >= 3 else np.nan


def _rank_within_timestamp(frame: pd.DataFrame, score_col: str) -> pd.Series:
    score = pd.to_numeric(frame[score_col], errors="coerce")
    return score.groupby(frame["__ts__"], observed=True).rank(
        method="average", pct=True, ascending=True
    )


def _normalize_scope(frame: pd.DataFrame, *, scope: str) -> pd.DataFrame:
    """Canonicalize a narrow, score/outcome-only strict OOS frame."""

    # This is intentionally in-place.  The base source has more than four
    # million rows, and retaining the raw plus a renamed copy is needless.
    result = frame
    result["__ts__"] = _utc(result["__ts__"])
    result["__symbol__"] = result["__symbol__"].astype(str)
    result["side_name"] = result["side_name"].astype(str).str.lower()
    if scope == "base_oos":
        result.rename(
            columns={
                "__archetype_policy_key__": "archetype_policy_key",
                "score": "model_score",
                "__u_policy_net__": "ev_after_1pct",
                "__long_path_clean_exec_label__": "clean_exec",
                "__long_path_dirty_positive_label__": "dirty_positive",
                "__path_full_bad_mae_1r__": "full_path_bad_mae_1r",
                "__first_touch_timeout__": "timeout",
                "__first_touch_stop__": "stop_or_adverse",
                "oos_fold": "oos_fold",
            },
            inplace=True,
        )
        result["base_score"] = result["model_score"]
        result["meta_score"] = np.nan
    elif scope == "base_meta_oos":
        # ``score`` is the unchanged base backbone in this artifact. The
        # residual expert's actual ordering and its matching base comparator
        # are the two train-reference rank columns below.
        result["backbone_score"] = pd.to_numeric(
            result.get("score"), errors="coerce"
        )
        result["model_score"] = pd.to_numeric(
            result.get(
                "score_base_residual_ev_rank_train_reference",
                result.get("score"),
            ),
            errors="coerce",
        )
        result["base_score"] = pd.to_numeric(
            result.get(
                "score_base_ev_rank_train_reference",
                result.get("score_base"),
            ),
            errors="coerce",
        )
        result["meta_score"] = result["model_score"]
        result["stop_or_adverse"] = np.nan
        result["oos_fold"] = pd.NA
    else:
        raise ValueError(f"Unknown scope: {scope}")
    if "archetype_policy_key" not in result:
        raise ValueError(f"{scope} source has no archetype policy key")
    result["archetype_policy_key"] = result["archetype_policy_key"].astype(str)
    for name in (
        "model_score",
        "base_score",
        "meta_score",
        "ev_after_1pct",
        "clean_exec",
        "dirty_positive",
        "full_path_bad_mae_1r",
        "timeout",
        "stop_or_adverse",
    ):
        if name not in result:
            result[name] = np.nan
        result[name] = pd.to_numeric(result[name], errors="coerce").astype(np.float32)
    result = result.loc[result["__ts__"].notna()]
    if result.duplicated(list(KEYS)).any():
        result = result.drop_duplicates(list(KEYS), keep="last")
    result.sort_values(list(KEYS), kind="stable", inplace=True)
    result["day"] = result["__ts__"].dt.floor("D")
    result["month"] = result["day"].dt.strftime("%Y-%m")
    result["week_start"] = result["day"] - pd.to_timedelta(
        result["day"].dt.weekday, unit="D"
    )
    result["score_rank_pct"] = _rank_within_timestamp(result, "model_score").astype(np.float32)
    for fraction in TOP_K:
        name = f"top{int(fraction * 100):02d}"
        result[name] = result["score_rank_pct"].ge(1.0 - fraction).fillna(False)
    return result.reset_index(drop=True)


def _load_mode_context(taxonomy: Path) -> pd.DataFrame:
    """Attach frozen ex-post mode labels as *descriptive* context only."""

    calendar = pd.read_parquet(
        taxonomy / "local_adverse_calendar.parquet",
        columns=["day", "side_name", "archetype_policy_key", "event_block", "adverse_event"],
    )
    assignments = pd.read_parquet(
        taxonomy / "local_frozen_failure_mode_semantic_assignments.parquet",
        columns=["side_name", "archetype_policy_key", "event_block", "semantic_label", "failure_mode_id"],
    )
    calendar["day"] = _utc(calendar["day"]).dt.floor("D")
    calendar["side_name"] = calendar["side_name"].astype(str).str.lower()
    assignments["side_name"] = assignments["side_name"].astype(str).str.lower()
    context = calendar.merge(
        assignments.drop_duplicates(["side_name", "archetype_policy_key", "event_block"]),
        on=["side_name", "archetype_policy_key", "event_block"],
        how="left",
        validate="many_to_one",
    )
    context["frozen_failure_mode"] = context["semantic_label"].where(
        _as_bool(context["adverse_event"]) & context["semantic_label"].notna(),
        "benign_or_unassigned",
    )
    return context.loc[
        :, ["day", "side_name", "archetype_policy_key", "event_block", "adverse_event", "frozen_failure_mode", "failure_mode_id"]
    ].drop_duplicates(["day", "side_name", "archetype_policy_key"])


def _attach_context(frame: pd.DataFrame, context: pd.DataFrame) -> pd.DataFrame:
    # A merge makes a second full copy of the 4.3M-row base ledger.  Indexed
    # lookup preserves the exact same many-to-one calendar contract.
    context_index = pd.MultiIndex.from_frame(
        context.loc[:, ["day", "side_name", "archetype_policy_key"]]
    )
    if context_index.has_duplicates:
        raise ValueError("Taxonomy context must be unique by day/side/archetype")
    lookup = context.set_index(["day", "side_name", "archetype_policy_key"])
    row_index = pd.MultiIndex.from_frame(
        frame.loc[:, ["day", "side_name", "archetype_policy_key"]]
    )
    matched = lookup.reindex(row_index)
    for column in ("event_block", "adverse_event", "frozen_failure_mode", "failure_mode_id"):
        frame[column] = matched[column].to_numpy(copy=False)
    frame["frozen_failure_mode"] = frame["frozen_failure_mode"].fillna("taxonomy_unavailable")
    frame["taxonomy_context_available"] = frame["frozen_failure_mode"].ne("taxonomy_unavailable")
    return frame


def _metric_row(frame: pd.DataFrame) -> dict[str, Any]:
    selected = len(frame)
    score = pd.to_numeric(frame["model_score"], errors="coerce")
    clean = pd.to_numeric(frame["clean_exec"], errors="coerce")
    ev = pd.to_numeric(frame["ev_after_1pct"], errors="coerce")
    residual = clean - score
    valid_probability = clean.notna() & score.between(0.0, 1.0)
    return {
        "rows": int(selected),
        "timestamps": int(frame["__ts__"].nunique()),
        "days": int(frame["day"].nunique()),
        "symbols": int(frame["__symbol__"].nunique()),
        "mean_ev_after_1pct": _safe_mean(frame, "ev_after_1pct"),
        "sum_ev_after_1pct": float(ev.sum()) if ev.notna().any() else np.nan,
        "positive_ev_rate": float(ev.gt(0.0).mean()) if ev.notna().any() else np.nan,
        "clean_exec_rate": _safe_rate(frame, "clean_exec"),
        "dirty_positive_rate": _safe_rate(frame, "dirty_positive"),
        "full_path_bad_mae_rate": _safe_rate(frame, "full_path_bad_mae_1r"),
        "timeout_rate": _safe_rate(frame, "timeout"),
        "stop_or_adverse_rate": _safe_rate(frame, "stop_or_adverse"),
        "mean_model_score": float(score.mean()) if score.notna().any() else np.nan,
        "mean_clean_score_residual": float(residual.mean()) if residual.notna().any() else np.nan,
        "mean_abs_clean_score_residual": float(residual.abs().mean()) if residual.notna().any() else np.nan,
        "clean_score_brier": (
            float(brier_score_loss(clean.loc[valid_probability], score.loc[valid_probability]))
            if valid_probability.any()
            else np.nan
        ),
        "score_ev_spearman": _safe_spearman(score, ev),
    }


def _metric_table(frame: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    groups: Iterable[tuple[Any, pd.DataFrame]]
    if group_columns:
        groups = frame.groupby(group_columns, observed=True, sort=True, dropna=False)
    else:
        groups = [((), frame)]
    for values, part in groups:
        if not isinstance(values, tuple):
            values = (values,)
        rows.append({**dict(zip(group_columns, values)), **_metric_row(part)})
    return pd.DataFrame(rows)


def _selection_tables(frame: pd.DataFrame, *, scope: str) -> dict[str, pd.DataFrame]:
    output: dict[str, pd.DataFrame] = {}
    for fraction in TOP_K:
        tail = f"top{int(fraction * 100):02d}"
        selected = frame.loc[frame[tail]].copy()
        for name, groups in {
            "overall": [],
            "month": ["month"],
            "week": ["week_start"],
            "side": ["side_name"],
            "archetype": ["side_name", "archetype_policy_key"],
            "mode": ["frozen_failure_mode"],
            "month_side_archetype_mode": ["month", "side_name", "archetype_policy_key", "frozen_failure_mode"],
        }.items():
            table = _metric_table(selected, groups)
            table.insert(0, "scope", scope)
            table.insert(1, "tail", tail)
            output[f"{tail}_{name}"] = table
    return output


def _daily_health(frame: pd.DataFrame, *, scope: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for values, part in frame.loc[frame["top10"]].groupby(
        ["day", "side_name", "archetype_policy_key", "frozen_failure_mode"],
        observed=True,
        sort=True,
        dropna=False,
    ):
        day, side, archetype, mode = values
        metric = _metric_row(part)
        rows.append(
            {
                "scope": scope,
                "day": day,
                "side_name": side,
                "archetype_policy_key": archetype,
                "frozen_failure_mode": mode,
                "negative_ev_day": bool(metric["mean_ev_after_1pct"] < 0.0)
                if np.isfinite(metric["mean_ev_after_1pct"])
                else False,
                **metric,
            }
        )
    return pd.DataFrame(rows)


def _disagreement_table(frame: pd.DataFrame, *, scope: str) -> pd.DataFrame:
    work = frame.loc[:, [
        "month", "side_name", "archetype_policy_key", "frozen_failure_mode",
        "base_score", "meta_score", "ev_after_1pct",
        *(["score_base_ev_residual_expert_hier_mapped"]
          if "score_base_ev_residual_expert_hier_mapped" in frame else []),
    ]].copy()
    base = pd.to_numeric(work["base_score"], errors="coerce")
    meta = pd.to_numeric(work["meta_score"], errors="coerce")
    work["meta_minus_base_score"] = meta - base
    work["base_meta_abs_disagreement"] = (meta - base).abs()
    if "score_base_ev_residual_expert_hier_mapped" in work:
        work["meta_expected_ev_minus_realized"] = (
            pd.to_numeric(work["score_base_ev_residual_expert_hier_mapped"], errors="coerce")
            - pd.to_numeric(work["ev_after_1pct"], errors="coerce")
        )
    rows: list[dict[str, Any]] = []
    for values, part in work.groupby(
        ["month", "side_name", "archetype_policy_key", "frozen_failure_mode"],
        observed=True,
        sort=True,
        dropna=False,
    ):
        row = dict(zip(["month", "side_name", "archetype_policy_key", "frozen_failure_mode"], values))
        row.update(
            {
                "scope": scope,
                "rows": int(len(part)),
                "base_score_coverage": float(pd.to_numeric(part["base_score"], errors="coerce").notna().mean()),
                "meta_score_coverage": float(pd.to_numeric(part["meta_score"], errors="coerce").notna().mean()),
                "mean_meta_minus_base_score": _safe_mean(part, "meta_minus_base_score"),
                "mean_abs_base_meta_disagreement": _safe_mean(part, "base_meta_abs_disagreement"),
                "base_meta_score_spearman": _safe_spearman(part["base_score"], part["meta_score"]),
                "mean_meta_expected_ev_minus_realized": _safe_mean(part, "meta_expected_ev_minus_realized"),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _detector_alignment(
    health: pd.DataFrame,
    detector_path: Path | None,
    *,
    scope: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Join detector probabilities only as a cross-source diagnostic.

    Detector targets were derived from the diagnostic backcast.  Therefore the
    relationship to strict OOS negative days is explicitly a consistency check,
    not a prospective accuracy claim.
    """

    empty_columns = [
        "scope", "failure_mode", "side_name", "archetype_policy_key",
        "frozen_failure_mode", "strict_oos_day_cells",
        "detector_probability_coverage", "mean_detector_risk",
        "detector_alert_days", "strict_negative_ev_days",
        "strict_negative_ev_prevalence", "alert_precision_against_strict_negative_ev",
        "alert_lift_against_strict_negative_ev", "risk_ap_against_strict_negative_ev",
        "cross_source_diagnostic_only",
    ]
    if detector_path is None or not detector_path.exists():
        return pd.DataFrame(columns=empty_columns), {
            "detector_available": False,
            "detector_path": str(detector_path) if detector_path else "",
            "reason": "No detector prediction artifact exists at the requested path.",
        }
    requested = [
        "day", "side_name", "archetype_policy_key", "failure_mode", "risk", "alert",
        "threshold", "fold_index", "train_end", "eval_end", "target_horizon_days",
    ]
    detector = pd.read_parquet(detector_path, columns=_available_columns(detector_path, requested))
    required = {"day", "side_name", "archetype_policy_key", "failure_mode", "risk"}
    missing = required.difference(detector.columns)
    if missing:
        raise ValueError(f"Detector artifact missing required fields: {sorted(missing)}")
    detector["day"] = _utc(detector["day"]).dt.floor("D")
    detector["side_name"] = detector["side_name"].astype(str).str.lower()
    detector["risk"] = pd.to_numeric(detector["risk"], errors="coerce")
    if "alert" not in detector:
        detector["alert"] = False
    detector["alert"] = _as_bool(detector["alert"])
    aligned = health.merge(
        detector,
        on=["day", "side_name", "archetype_policy_key"],
        how="left",
        validate="one_to_many",
        suffixes=("", "_detector"),
    )
    rows: list[dict[str, Any]] = []
    group_columns = ["failure_mode", "side_name", "archetype_policy_key", "frozen_failure_mode"]
    for values, part in aligned.groupby(group_columns, observed=True, sort=True, dropna=False):
        risk = pd.to_numeric(part["risk"], errors="coerce")
        target = part["negative_ev_day"].astype(bool)
        valid = risk.notna()
        alerts = part["alert"].astype("boolean").fillna(False).astype(bool) & valid
        prevalence = float(target.loc[valid].mean()) if valid.any() else np.nan
        precision = float(target.loc[alerts].mean()) if alerts.any() else np.nan
        rows.append(
            {
                "scope": scope,
                **dict(zip(group_columns, values)),
                "strict_oos_day_cells": int(part[["day", "side_name", "archetype_policy_key"]].drop_duplicates().shape[0]),
                "detector_probability_coverage": float(valid.mean()),
                "mean_detector_risk": float(risk.mean()) if valid.any() else np.nan,
                "detector_alert_days": int(alerts.sum()),
                "strict_negative_ev_days": int(target.loc[valid].sum()),
                "strict_negative_ev_prevalence": prevalence,
                "alert_precision_against_strict_negative_ev": precision,
                "alert_lift_against_strict_negative_ev": precision / prevalence if np.isfinite(precision) and prevalence else np.nan,
                "risk_ap_against_strict_negative_ev": (
                    float(average_precision_score(target.loc[valid].astype(np.int8), risk.loc[valid]))
                    if valid.sum() >= 2 and target.loc[valid].nunique() == 2
                    else np.nan
                ),
                "cross_source_diagnostic_only": True,
            }
        )
    return pd.DataFrame(rows), {
        "detector_available": True,
        "detector_path": str(detector_path.resolve()),
        "detector_rows": int(len(detector)),
        "contract": (
            "Detector targets were constructed from frozen diagnostic-backcast outcomes. "
            "This alignment against genuine strict-OOS negative days is descriptive only, "
            "not a new OOS detector-performance estimate."
        ),
    }


def _coverage_row(frame: pd.DataFrame, *, scope: str, source_rows: int) -> dict[str, Any]:
    return {
        "scope": scope,
        "source_rows": int(source_rows),
        "strict_rows": int(len(frame)),
        "row_retention": float(len(frame) / max(source_rows, 1)),
        "start": frame["__ts__"].min(),
        "end": frame["__ts__"].max(),
        "days": int(frame["day"].nunique()),
        "timestamps": int(frame["__ts__"].nunique()),
        "symbols": int(frame["__symbol__"].nunique()),
        "side_archetype_cells": int(frame[["side_name", "archetype_policy_key"]].drop_duplicates().shape[0]),
        "taxonomy_context_coverage": float(frame["taxonomy_context_available"].mean()),
        "score_coverage": float(pd.to_numeric(frame["model_score"], errors="coerce").notna().mean()),
        "outcome_coverage": float(pd.to_numeric(frame["ev_after_1pct"], errors="coerce").notna().mean()),
    }


def _intersection_tail_metrics(frame: pd.DataFrame, score_column: str) -> dict[str, Any]:
    score = pd.to_numeric(frame[score_column], errors="coerce")
    result: dict[str, Any] = {}
    ranked = score.groupby(frame["__ts__"], observed=True).rank(
        method="average", pct=True, ascending=True
    )
    for fraction in TOP_K:
        selected = frame.loc[ranked.ge(1.0 - fraction)]
        prefix = f"top{int(fraction * 100):02d}"
        result[f"{prefix}_rows"] = int(len(selected))
        result[f"{prefix}_mean_ev_after_1pct"] = _safe_mean(selected, "ev_after_1pct")
        result[f"{prefix}_clean_exec_rate"] = _safe_rate(selected, "clean_exec")
        result[f"{prefix}_dirty_positive_rate"] = _safe_rate(selected, "dirty_positive")
        result[f"{prefix}_full_path_bad_mae_rate"] = _safe_rate(selected, "full_path_bad_mae_1r")
        result[f"{prefix}_timeout_rate"] = _safe_rate(selected, "timeout")
    return result


def _base_meta_intersection(
    base: pd.DataFrame,
    meta: pd.DataFrame,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """Compare base and meta ranking only on their exact UTC-key intersection."""

    base_columns = list(KEYS) + [
        "day", "month", "week_start", "model_score", "ev_after_1pct", "clean_exec",
        "dirty_positive", "full_path_bad_mae_1r", "timeout", "frozen_failure_mode",
    ]
    meta_columns = list(KEYS) + [
        "base_score", "model_score", "ev_after_1pct", "clean_exec", "dirty_positive",
        "full_path_bad_mae_1r", "timeout", "frozen_failure_mode",
        "score_base_ev_residual_expert_hier_mapped",
    ]
    left = base.loc[:, [name for name in base_columns if name in base]].rename(
        columns={
            "model_score": "base_ledger_raw_score",
            "ev_after_1pct": "base_oos_ev_after_1pct",
            "clean_exec": "base_oos_clean_exec",
            "dirty_positive": "base_oos_dirty_positive",
            "full_path_bad_mae_1r": "base_oos_full_path_bad_mae_1r",
            "timeout": "base_oos_timeout",
        }
    )
    right = meta.loc[:, [name for name in meta_columns if name in meta]].rename(
        columns={
            "base_score": "base_comparator_score",
            "model_score": "meta_oos_score",
            "ev_after_1pct": "ev_after_1pct",
            "clean_exec": "clean_exec",
            "dirty_positive": "dirty_positive",
            "full_path_bad_mae_1r": "full_path_bad_mae_1r",
            "timeout": "timeout",
        }
    )
    overlap = left.merge(right, on=list(KEYS), how="inner", validate="one_to_one")
    label_columns = (
        ("base_oos_ev_after_1pct", "ev_after_1pct"),
        ("base_oos_clean_exec", "clean_exec"),
        ("base_oos_dirty_positive", "dirty_positive"),
        ("base_oos_full_path_bad_mae_1r", "full_path_bad_mae_1r"),
        ("base_oos_timeout", "timeout"),
    )
    mismatch: dict[str, int] = {}
    for left_name, right_name in label_columns:
        paired = overlap[[left_name, right_name]].dropna()
        mismatch[f"{left_name}_cross_ledger_mismatch_rows"] = int(
            (~np.isclose(
                pd.to_numeric(paired[left_name], errors="coerce"),
                pd.to_numeric(paired[right_name], errors="coerce"),
                equal_nan=True,
            )).sum()
        )
    overall = pd.DataFrame(
        [
            {
                "comparison_scope": "exact_base_meta_oos_intersection",
                "rows": int(len(overlap)),
                "timestamps": int(overlap["__ts__"].nunique()),
                "days": int(overlap["day"].nunique()),
                "symbols": int(overlap["__symbol__"].nunique()),
                "score_spearman": _safe_spearman(overlap["base_comparator_score"], overlap["meta_oos_score"]),
                "mean_abs_score_delta": float(
                    (pd.to_numeric(overlap["meta_oos_score"], errors="coerce") - pd.to_numeric(overlap["base_comparator_score"], errors="coerce")).abs().mean()
                ),
                **{f"base_{key}": value for key, value in _intersection_tail_metrics(overlap, "base_comparator_score").items()},
                **{f"meta_{key}": value for key, value in _intersection_tail_metrics(overlap, "meta_oos_score").items()},
                **mismatch,
            }
        ]
    )
    monthly_rows: list[dict[str, Any]] = []
    for month, part in overlap.groupby("month", observed=True, sort=True):
        monthly_rows.append(
            {
                "month": month,
                "rows": int(len(part)),
                "score_spearman": _safe_spearman(part["base_comparator_score"], part["meta_oos_score"]),
                "mean_abs_score_delta": float(
                    (pd.to_numeric(part["meta_oos_score"], errors="coerce") - pd.to_numeric(part["base_comparator_score"], errors="coerce")).abs().mean()
                ),
                **{f"base_{key}": value for key, value in _intersection_tail_metrics(part, "base_comparator_score").items()},
                **{f"meta_{key}": value for key, value in _intersection_tail_metrics(part, "meta_oos_score").items()},
            }
        )
    local_rows: list[dict[str, Any]] = []
    for values, part in overlap.groupby(
        ["side_name", "archetype_policy_key", "frozen_failure_mode"],
        observed=True,
        sort=True,
        dropna=False,
    ):
        local_rows.append(
            {
                "side_name": values[0],
                "archetype_policy_key": values[1],
                "frozen_failure_mode": values[2],
                "rows": int(len(part)),
                "score_spearman": _safe_spearman(part["base_comparator_score"], part["meta_oos_score"]),
                **{f"base_{key}": value for key, value in _intersection_tail_metrics(part, "base_comparator_score").items()},
                **{f"meta_{key}": value for key, value in _intersection_tail_metrics(part, "meta_oos_score").items()},
            }
        )
    source_coverage = {
        "base_rows": int(len(base)),
        "meta_rows": int(len(meta)),
        "overlap_rows": int(len(overlap)),
        "meta_of_base_coverage": float(len(overlap) / max(len(base), 1)),
        "base_of_meta_coverage": float(len(overlap) / max(len(meta), 1)),
        "timestamp_contract": "exact UTC timestamp + symbol + side + archetype key",
        "base_comparator_score": "score_base_ev_rank_train_reference",
        "meta_score": "score_base_residual_ev_rank_train_reference",
        "outcome_contract": (
            "Both rankers are evaluated against ev_after_1pct and path labels "
            "from the genuine meta-OOS handoff. The base ledger is joined only "
            "to prove exact OOS row provenance."
        ),
        "cross_ledger_outcome_contract_mismatches": mismatch,
    }
    return {
        "coverage": pd.DataFrame([source_coverage]),
        "overall": overall,
        "monthly": pd.DataFrame(monthly_rows),
        "side_archetype_mode": pd.DataFrame(local_rows),
    }, source_coverage


def _load_base_intersection(
    *,
    source: Path,
    meta: pd.DataFrame,
) -> pd.DataFrame:
    """Stream only exact base/meta overlap rows from the large base ledger.

    The frozen failure mode is already attached to the meta side of the exact
    key join.  Avoiding a taxonomy join on every base row in the April--July
    temporal slice keeps this read bounded to the actual meta candidate set.
    """

    wanted = pd.MultiIndex.from_frame(meta.loc[:, list(KEYS)])
    columns = _available_columns(source, BASE_PROJECTION)
    pieces: list[pd.DataFrame] = []
    start, end = meta["__ts__"].min(), meta["__ts__"].max()
    for batch in pq.ParquetFile(source).iter_batches(columns=columns, batch_size=100_000):
        part = batch.to_pandas()
        timestamp = _utc(part["__ts__"])
        temporal = timestamp.ge(start) & timestamp.le(end)
        if not temporal.any():
            continue
        candidate = part.loc[temporal]
        candidate_keys = pd.MultiIndex.from_arrays(
            [
                _utc(candidate["__ts__"]),
                candidate["__symbol__"].astype(str),
                candidate["side_name"].astype(str).str.lower(),
                candidate["__archetype_policy_key__"].astype(str),
            ],
            names=KEYS,
        )
        matched = candidate.loc[candidate_keys.isin(wanted)]
        if not matched.empty:
            pieces.append(matched.copy())
    if not pieces:
        return pd.DataFrame(columns=[*KEYS, "model_score"])
    raw = pd.concat(pieces, ignore_index=True, copy=False)
    return _normalize_scope(raw, scope="base_oos")


def _write_csvs(output: Path, prefix: str, tables: Mapping[str, pd.DataFrame]) -> list[str]:
    names: list[str] = []
    for name, table in tables.items():
        path = output / f"{prefix}_{name}.csv"
        table.to_csv(path, index=False)
        names.append(path.name)
    return names


def _reuse_base_scope(
    *,
    source_report: Path,
    base_ledger: Path,
    taxonomy: Path,
    detector_path: Path | None,
    output: Path,
) -> tuple[list[str], dict[str, Any], pd.DataFrame]:
    """Reuse a verified strict base-OOS report without recalculating its 4M rows."""

    manifest_path = source_report / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    source_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    base_scope = source_manifest.get("scopes", {}).get("base_oos")
    if not isinstance(base_scope, dict):
        raise ValueError(f"{manifest_path} has no reusable base_oos scope")
    if Path(base_scope.get("source", "")).resolve() != base_ledger.resolve():
        raise ValueError("Reusable base report was generated from a different base ledger")
    if Path(source_manifest.get("taxonomy", "")).resolve() != taxonomy.resolve():
        raise ValueError("Reusable base report was generated from a different taxonomy")
    copied: list[str] = []
    for path in sorted(source_report.glob("strict_base_oos_*.csv")):
        destination = output / path.name
        shutil.copy2(path, destination)
        copied.append(destination.name)
    coverage_path = output / "strict_base_oos_coverage.csv"
    if not coverage_path.exists():
        raise ValueError("Reusable base report lacks strict_base_oos_coverage.csv")
    daily_health_path = output / "strict_base_oos_daily_health.csv"
    if not daily_health_path.exists():
        raise ValueError("Reusable base report lacks strict_base_oos_daily_health.csv")

    # Reuse the expensive strict-base metrics, but never inherit a detector
    # version from the source report. Detector alignment is a lightweight
    # descriptive join on saved daily health and must honor this invocation's
    # explicit detector contract.
    base_health = pd.read_csv(daily_health_path)
    base_health["day"] = _utc(base_health["day"]).dt.floor("D")
    base_health["side_name"] = base_health["side_name"].astype(str).str.lower()
    base_health["archetype_policy_key"] = base_health["archetype_policy_key"].astype(str)
    base_detector, detector_manifest = _detector_alignment(
        base_health, detector_path, scope="base_oos"
    )
    detector_destination = output / "strict_base_oos_detector_alignment.csv"
    base_detector.to_csv(detector_destination, index=False)
    if detector_destination.name not in copied:
        copied.append(detector_destination.name)
    base_scope = {
        **base_scope,
        "detector": detector_manifest,
        "reused_from_report": str(source_report.resolve()),
        "reuse_contract": (
            "Copied read-only CSV outputs after verifying the exact base ledger and taxonomy paths. "
            "No model, taxonomy, score, outcome, threshold, or calculation was changed."
        ),
    }
    return copied, base_scope, pd.read_csv(coverage_path)


def _scope(
    *,
    source: Path,
    scope: str,
    taxonomy_context: pd.DataFrame,
    detector_path: Path | None,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[str, Any]]:
    projection = BASE_PROJECTION if scope == "base_oos" else META_PROJECTION
    raw = _read_projection(source, projection)
    source_rows = len(raw)
    strict = _attach_context(_normalize_scope(raw, scope=scope), taxonomy_context)
    del raw
    health = _daily_health(strict, scope=scope)
    detector, detector_manifest = _detector_alignment(health, detector_path, scope=scope)
    tables = _selection_tables(strict, scope=scope)
    tables["daily_health"] = health
    tables["negative_days"] = health.loc[health["negative_ev_day"]].copy()
    tables["error_residual_disagreement"] = _disagreement_table(strict, scope=scope)
    tables["detector_alignment"] = detector
    tables["coverage"] = pd.DataFrame([_coverage_row(strict, scope=scope, source_rows=source_rows)])
    scope_manifest = {
        "scope": scope,
        "source": str(source.resolve()),
        "source_rows": int(source_rows),
        "strict_rows": int(len(strict)),
        "utc_join_contract": "timestamps normalized to UTC before all joins; naive legacy timestamps interpreted as UTC",
        "strict_oos_contract": (
            "This scope consumes only persisted growing-window OOS predictions and their realized outcomes. "
            "No taxonomy fit, detector fit, feature selection, calibration, or threshold was performed here."
        ),
        "taxonomy_contract": (
            "V17 frozen modes are joined as descriptive frozen-backcast context only. "
            "The three-year taxonomy source is a frozen diagnostic backcast, not full OOS."
        ),
        "detector": detector_manifest,
    }
    return strict, tables, scope_manifest


def run(
    *,
    base_ledger: Path = DEFAULT_BASE_LEDGER,
    meta_ledger: Path = DEFAULT_META_LEDGER,
    taxonomy: Path = DEFAULT_TAXONOMY,
    detector: Path | None = DEFAULT_DETECTOR,
    output: Path = DEFAULT_OUTPUT,
    reuse_base_report: Path | None = None,
) -> dict[str, Any]:
    """Build strict base-OOS and strict base+meta-OOS sensitivity artifacts."""

    base_ledger, meta_ledger, taxonomy, output = map(Path, (base_ledger, meta_ledger, taxonomy, output))
    detector = Path(detector) if detector else None
    reuse_base_report = Path(reuse_base_report) if reuse_base_report else None
    for path in (base_ledger, meta_ledger):
        if not path.exists():
            raise FileNotFoundError(path)
    if not (taxonomy / "manifest.json").exists():
        raise FileNotFoundError(taxonomy / "manifest.json")
    output.mkdir(parents=True, exist_ok=True)
    taxonomy_manifest = json.loads((taxonomy / "manifest.json").read_text(encoding="utf-8"))
    context = _load_mode_context(taxonomy)
    if reuse_base_report is not None:
        files, base_manifest, base_coverage = _reuse_base_scope(
            source_report=reuse_base_report,
            base_ledger=base_ledger,
            taxonomy=taxonomy,
            detector_path=detector,
            output=output,
        )
    else:
        base_strict, base_tables, base_manifest = _scope(
            source=base_ledger,
            scope="base_oos",
            taxonomy_context=context,
            detector_path=detector,
        )
        files = _write_csvs(output, "strict_base_oos", base_tables)
        base_coverage = base_tables["coverage"]
        del base_tables, base_strict
        gc.collect()
    meta_strict, meta_tables, meta_manifest = _scope(
        source=meta_ledger,
        scope="base_meta_oos",
        taxonomy_context=context,
        detector_path=detector,
    )
    files += _write_csvs(output, "strict_base_meta_oos", meta_tables)
    base_intersection = _load_base_intersection(
        source=base_ledger,
        meta=meta_strict,
    )
    overlap_tables, overlap_manifest = _base_meta_intersection(base_intersection, meta_strict)
    files += _write_csvs(output, "strict_base_meta_intersection", overlap_tables)
    coverage = pd.concat([base_coverage, meta_tables["coverage"]], ignore_index=True)
    coverage.to_csv(output / "strict_oos_coverage.csv", index=False)
    files.append("strict_oos_coverage.csv")
    manifest: dict[str, Any] = {
        "schema": "failure_taxonomy_strict_oos_sensitivity_v1",
        "purpose": "read-only strict-OOS sensitivity for failure-taxonomy research",
        "taxonomy": str(taxonomy.resolve()),
        "taxonomy_source_provenance": taxonomy_manifest.get("source", {}).get("provenance"),
        "taxonomy_not_full_oos": True,
        "taxonomy_disclosure": (
            "The three-year V17 taxonomy is a frozen diagnostic backcast. It provides descriptive mode context, "
            "not full-OOS taxonomy discovery or detector labels for the strict OOS ledgers."
        ),
        "scopes": {"base_oos": base_manifest, "base_meta_oos": meta_manifest},
        "base_meta_intersection": overlap_manifest,
        "cost_contract": "Uses persisted ev_after_1pct / __u_policy_net__ values as supplied; no additional cost is subtracted.",
        "top_k_contract": "Top10/20/30 recomputed per UTC timestamp from the persisted score available in each source scope.",
        "files": sorted(files),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, default=_json_default), flush=True)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-ledger", type=Path, default=DEFAULT_BASE_LEDGER)
    parser.add_argument("--meta-ledger", type=Path, default=DEFAULT_META_LEDGER)
    parser.add_argument("--taxonomy", type=Path, default=DEFAULT_TAXONOMY)
    parser.add_argument("--detector", type=Path, default=DEFAULT_DETECTOR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--reuse-base-report",
        type=Path,
        default=None,
        help="Verified prior strict-OOS report from the exact same base ledger and taxonomy.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        base_ledger=args.base_ledger,
        meta_ledger=args.meta_ledger,
        taxonomy=args.taxonomy,
        detector=args.detector,
        output=args.output,
        reuse_base_report=args.reuse_base_report,
    )
