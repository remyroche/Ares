"""Diagnostics for archetype usefulness and portfolio modulation."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import logging
import time
from pathlib import Path
from typing import Any, Iterator, Mapping

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def _report_scalar(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return json.dumps(value.tolist(), default=str)
    if isinstance(value, Mapping):
        return json.dumps({str(k): _report_scalar(v) for k, v in value.items()}, default=str)
    if isinstance(value, (list, tuple, set)):
        return json.dumps([_report_scalar(v) for v in value], default=str)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
        return value if np.isfinite(value) else np.nan
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if np.isfinite(value) else np.nan
    return str(value)


@dataclass
class PipelineStageReporter:
    """Structured stage telemetry for fold-local pipeline runs."""

    logger: logging.Logger | None = None
    rows: list[dict[str, Any]] = field(default_factory=list)
    _sequence: int = 0

    def event(
        self,
        stage: str,
        status: str,
        *,
        fold: int | str | None = None,
        **metrics: Any,
    ) -> dict[str, Any]:
        self._sequence += 1
        row: dict[str, Any] = {
            "sequence": self._sequence,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "stage": str(stage),
            "status": str(status),
        }
        if fold is not None:
            row["fold"] = fold
        for key, value in metrics.items():
            row[str(key)] = _report_scalar(value)
        self.rows.append(row)
        if self.logger is not None:
            log_metrics = {
                key: value
                for key, value in row.items()
                if key not in {"sequence", "timestamp_utc", "stage", "status"}
            }
            message = "pipeline_stage stage=%s status=%s"
            args: tuple[Any, ...] = (stage, status)
            if status == "fail":
                self.logger.error(message + " metrics=%s", *args, log_metrics)
            else:
                self.logger.info(message + " metrics=%s", *args, log_metrics)
        return row

    @contextmanager
    def stage(
        self,
        stage: str,
        *,
        fold: int | str | None = None,
        **metadata: Any,
    ) -> Iterator[dict[str, Any]]:
        metrics = dict(metadata)
        start = time.perf_counter()
        start_metadata = dict(metadata)
        start_metadata.pop("fold", None)
        self.event(stage, "start", fold=fold, **start_metadata)
        try:
            yield metrics
        except Exception as exc:
            metrics.update(
                {
                    "duration_seconds": time.perf_counter() - start,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
            )
            fail_metrics = dict(metrics)
            fail_metrics.pop("fold", None)
            self.event(stage, "fail", fold=fold, **fail_metrics)
            raise
        metrics["duration_seconds"] = time.perf_counter() - start
        end_metrics = dict(metrics)
        end_metrics.pop("fold", None)
        self.event(stage, "end", fold=fold, **end_metrics)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)

    def summary_frame(self) -> pd.DataFrame:
        frame = self.to_frame()
        if frame.empty:
            return pd.DataFrame(
                columns=[
                    "fold",
                    "stage",
                    "status",
                    "event_count",
                    "duration_seconds_sum",
                    "duration_seconds_mean",
                    "duration_seconds_max",
                ]
            )
        completed = frame.loc[frame["status"].isin(["end", "fail"])].copy()
        if completed.empty:
            return pd.DataFrame()
        if "duration_seconds" not in completed.columns:
            completed["duration_seconds"] = np.nan
        if "fold" not in completed.columns:
            completed["fold"] = "global"
        return (
            completed.groupby(["fold", "stage", "status"], dropna=False)["duration_seconds"]
            .agg(
                event_count="count",
                duration_seconds_sum="sum",
                duration_seconds_mean="mean",
                duration_seconds_max="max",
            )
            .reset_index()
        )


@dataclass(frozen=True)
class ArchetypeUsefulnessReport:
    metrics: pd.DataFrame
    predictions: pd.DataFrame


def _r2(y: pd.Series, pred: pd.Series, weight: pd.Series | None = None) -> float:
    yv = pd.to_numeric(y, errors="coerce")
    pv = pd.to_numeric(pred, errors="coerce").reindex(yv.index)
    ok = yv.notna() & pv.notna()
    if not bool(ok.any()):
        return np.nan
    if weight is None:
        mean = float(yv.loc[ok].mean())
        denom = float(((yv.loc[ok] - mean) ** 2).sum())
        return 0.0 if denom <= 1e-12 else float(1.0 - ((yv.loc[ok] - pv.loc[ok]) ** 2).sum() / denom)
    w = pd.to_numeric(weight, errors="coerce").reindex(yv.index).fillna(1.0).loc[ok]
    mean = float(np.average(yv.loc[ok], weights=np.maximum(w, 1e-12)))
    denom = float(np.sum(np.maximum(w, 1e-12) * (yv.loc[ok] - mean) ** 2))
    return 0.0 if denom <= 1e-12 else float(1.0 - np.sum(np.maximum(w, 1e-12) * (yv.loc[ok] - pv.loc[ok]) ** 2) / denom)


def _linear_oof_like(X: pd.DataFrame, y: pd.Series) -> pd.Series:
    X = X.reindex(y.index).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if X.empty:
        return pd.Series(float(y.mean()), index=y.index)
    design = np.column_stack([np.ones(len(X)), X.to_numpy(dtype=float)])
    try:
        coef, *_ = np.linalg.lstsq(design, y.to_numpy(dtype=float), rcond=None)
        return pd.Series(design @ coef, index=y.index)
    except Exception:
        return pd.Series(float(y.mean()), index=y.index)


def evaluate_archetype_usefulness(
    archetype_scores: pd.DataFrame,
    strategy_performance: pd.DataFrame,
    unreliability_targets: pd.DataFrame,
    base_market_features: pd.DataFrame,
) -> ArchetypeUsefulnessReport:
    rows: list[dict[str, object]] = []
    preds: dict[str, pd.Series] = {}
    for target_name in unreliability_targets.columns:
        y = pd.to_numeric(unreliability_targets[target_name], errors="coerce").fillna(0.0)
        arch_pred = _linear_oof_like(archetype_scores, y)
        base_pred = _linear_oof_like(base_market_features, y)
        combined_pred = _linear_oof_like(pd.concat([base_market_features, archetype_scores], axis=1), y)
        preds[f"unreliability__{target_name}__archetype"] = arch_pred
        q90 = float(arch_pred.quantile(0.90)) if len(arch_pred) else np.nan
        top = arch_pred >= q90
        failure_rate = float(y.mean())
        top_rate = float(y.loc[top].mean()) if bool(top.any()) else np.nan
        try:
            auc_arch = float(roc_auc_score((y > 0.5).astype(int), arch_pred))
            auc_base = float(roc_auc_score((y > 0.5).astype(int), base_pred))
        except Exception:
            auc_arch = np.nan
            auc_base = np.nan
        rows.append(
            {
                "target_type": "unreliability",
                "target": target_name,
                "archetype_r2": _r2(y, arch_pred),
                "base_market_r2": _r2(y, base_pred),
                "combined_r2": _r2(y, combined_pred),
                "incremental_r2_over_base": _r2(y, combined_pred) - _r2(y, base_pred),
                "incremental_unreliability_auc": auc_arch - auc_base if np.isfinite(auc_arch) and np.isfinite(auc_base) else np.nan,
                "precision_at_top_decile": top_rate,
                "failure_lift_top_decile": top_rate / max(failure_rate, 1e-12) if np.isfinite(top_rate) else np.nan,
            }
        )
    for target_name in strategy_performance.columns:
        y = pd.to_numeric(strategy_performance[target_name], errors="coerce").fillna(0.0)
        arch_pred = _linear_oof_like(archetype_scores, y)
        base_pred = _linear_oof_like(base_market_features, y)
        combined_pred = _linear_oof_like(pd.concat([base_market_features, archetype_scores], axis=1), y)
        reference_pred = _linear_oof_like(pd.concat([base_market_features, archetype_scores], axis=1), y)
        inc_arch = _r2(y, arch_pred)
        inc_ref = _r2(y, reference_pred)
        rows.append(
            {
                "target_type": "strategy_performance",
                "target": target_name,
                "archetype_r2": inc_arch,
                "base_market_r2": _r2(y, base_pred),
                "combined_r2": _r2(y, combined_pred),
                "incremental_r2_over_base": _r2(y, combined_pred) - _r2(y, base_pred),
                "rank_ic": float(pd.Series(arch_pred).corr(y, method="spearman")),
                "archetype_explainable_share": inc_arch / max(inc_ref, 1e-12),
            }
        )
    return ArchetypeUsefulnessReport(pd.DataFrame(rows), pd.DataFrame(preds))
