#!/usr/bin/env python3
"""Report execution-EV diagnostics from an OOF-only ledger.

The execution-EV trainer writes a wide ledger where every prediction arm is a
column (for example ``direct__all_features``).  This script also accepts long
ledgers through ``--arm-col`` and several inputs through repeated ``--input``.
It is evaluation-only: no threshold, arm, or policy is selected here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_TIMESTAMP_COL = "__ts__"
DEFAULT_NET_EV_COL = "execution_net_ev_12h"
DEFAULT_GROSS_EV_COL = "execution_gross_ev_12h"
DEFAULT_SIDE_COL = "side_name"
DEFAULT_ARCHETYPE_COL = "catboost_archetype"
DEFAULT_OOF_FOLD_COL = "execution_ev_oof_fold"
TOP_FRACTIONS: tuple[tuple[str, float], ...] = (
    ("1", 0.01), ("5", 0.05), ("10", 0.10), ("20", 0.20), ("30", 0.30)
)
PREDICTION_PREFIXES = ("direct__", "residual__", "baseline__", "execution_ev_")
PNL_CANDIDATES = ("bankroll_pnl", "portfolio_pnl", "realized_pnl", "pnl")
SIZE_CANDIDATES = ("position_size", "sizing_notional", "notional", "allocated_notional")
SURPRISE_CANDIDATES = ("signed_hit_rate_surprise", "hit_rate_surprise")
ACTUAL_HIT_RATE_CANDIDATES = ("recent_resolved_hit_rate", "actual_hit_rate")
EXPECTED_HIT_RATE_CANDIDATES = (
    "train_derived_expected_hit_rate",
    "expected_hit_rate",
)
CLEAN_CANDIDATES = ("execution_clean", "is_clean", "clean", "clean_path")
DIRTY_CANDIDATES = ("execution_dirty", "is_dirty", "dirty", "dirty_path")


def _json_safe(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, (pd.Timestamp, pd.Timedelta)):
        return value.isoformat()
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _source_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _first_present(columns: Iterable[str], candidates: Sequence[str]) -> str | None:
    available = set(columns)
    return next((candidate for candidate in candidates if candidate in available), None)


def _read_frame(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".pkl", ".pickle"}:
        return pd.read_pickle(path)
    raise ValueError(f"unsupported OOF ledger extension {path.suffix!r}")


def _utc(values: pd.Series, *, column: str) -> pd.Series:
    converted = pd.to_datetime(values, utc=True, errors="coerce")
    if converted.isna().any():
        raise ValueError(f"{column!r} contains null or invalid timestamps")
    return converted


def _prediction_columns(frame: pd.DataFrame, requested: Sequence[str] | None) -> list[str]:
    if requested:
        missing = [column for column in requested if column not in frame]
        if missing:
            raise ValueError("prediction columns missing from ledger: " + ", ".join(missing))
        return list(requested)
    output = [
        column
        for column in frame.columns
        if column.startswith(PREDICTION_PREFIXES)
        and not column.endswith("__is_oof")
        and pd.api.types.is_numeric_dtype(frame[column])
    ]
    if not output:
        raise ValueError(
            "no prediction arms found; pass --prediction-cols or use --arm-col with --prediction-col"
        )
    return output


def _oof_mask(
    frame: pd.DataFrame,
    *,
    prediction_col: str,
    oof_col: str | None,
    fold_col: str,
) -> tuple[np.ndarray, str]:
    indicator = oof_col or f"{prediction_col}__is_oof"
    if indicator in frame:
        values = frame[indicator]
        if pd.api.types.is_bool_dtype(values):
            return values.fillna(False).to_numpy(dtype=bool), indicator
        numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
        return np.isfinite(numeric) & (numeric != 0.0), indicator
    if fold_col in frame:
        folds = pd.to_numeric(frame[fold_col], errors="coerce").to_numpy(dtype=float)
        return np.isfinite(folds) & (folds >= 0.0) & (folds == np.floor(folds)), fold_col
    raise ValueError(
        f"{prediction_col!r} has no explicit OOF indicator; expected {indicator!r} or {fold_col!r}"
    )


def _build_arms(
    inputs: Sequence[Path],
    arm_names: Sequence[str] | None,
    *,
    timestamp_col: str,
    net_ev_col: str,
    prediction_cols: Sequence[str] | None,
    prediction_col: str | None,
    arm_col: str | None,
    rank_col: str | None,
    oof_col: str | None,
    fold_col: str,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    if arm_names and len(arm_names) != len(inputs):
        raise ValueError("--arm-name must be supplied once for every --input")
    records: list[pd.DataFrame] = []
    sources: list[dict[str, Any]] = []
    for source_index, path in enumerate(inputs):
        if not path.is_file():
            raise ValueError(f"OOF input does not exist: {path}")
        frame = _read_frame(path)
        required = [timestamp_col, net_ev_col]
        missing = [column for column in required if column not in frame]
        if missing:
            raise ValueError(f"{path} is missing required columns: {', '.join(missing)}")
        frame = frame.copy()
        frame[timestamp_col] = _utc(frame[timestamp_col], column=timestamp_col)
        if arm_col:
            if arm_col not in frame:
                raise ValueError(f"{path} is missing arm column {arm_col!r}")
            if not prediction_col:
                raise ValueError("--prediction-col is required with --arm-col")
            columns = [prediction_col]
        else:
            columns = _prediction_columns(frame, prediction_cols or ([prediction_col] if prediction_col else None))
        for column in columns:
            if column not in frame:
                raise ValueError(f"{path} is missing prediction column {column!r}")
            arm_values = frame[arm_col].astype(str) if arm_col else pd.Series(
                arm_names[source_index] if arm_names else column, index=frame.index, dtype="object"
            )
            oof, oof_source = _oof_mask(
                frame, prediction_col=column, oof_col=oof_col, fold_col=fold_col
            )
            part = frame.copy()
            part["__arm__"] = arm_values.to_numpy()
            part["__prediction__"] = pd.to_numeric(part[column], errors="coerce")
            if rank_col:
                if rank_col not in part:
                    raise ValueError(f"{path} is missing rank column {rank_col!r}")
                part["__rank__"] = pd.to_numeric(part[rank_col], errors="coerce")
            else:
                part["__rank__"] = np.nan
            part["__oof__"] = oof
            part["__source__"] = str(path)
            part["__input_order__"] = np.arange(len(part), dtype=np.int64)
            part = part.loc[part["__oof__"]].copy()
            part = part.loc[np.isfinite(part["__prediction__"].to_numpy(dtype=float))].copy()
            if arm_col:
                # A long ledger has one source prediction column, so retain all arms.
                records.append(part)
            else:
                records.append(part)
            sources.append(
                {
                    "path": str(path),
                    "sha256": _source_hash(path),
                    "prediction_column": column,
                    "oof_indicator": oof_source,
                    "rows_read": int(len(frame)),
                    "oof_rows_before_prediction_filter": int(oof.sum()),
                }
            )
    if not records:
        raise ValueError("no OOF prediction rows were found")
    output = pd.concat(records, ignore_index=True, sort=False)
    if output.empty:
        raise ValueError("no finite OOF prediction rows were found")
    return output, sources


def _rate(values: pd.Series | np.ndarray) -> float:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(numeric)
    return float(np.mean(numeric[valid])) if valid.any() else float("nan")


def _bool_rate(frame: pd.DataFrame, column: str | None) -> float:
    if not column:
        return float("nan")
    values = frame[column]
    if pd.api.types.is_bool_dtype(values):
        return float(values.fillna(False).mean())
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(numeric)
    return float(np.mean(numeric[valid] != 0.0)) if valid.any() else float("nan")


def _lag1_autocorrelation(frame: pd.DataFrame, residual: np.ndarray, timestamp_col: str) -> float:
    order = np.lexsort((frame["__input_order__"].to_numpy(), frame[timestamp_col].astype("int64").to_numpy()))
    return _ordered_lag1_autocorrelation(residual[order])


def _ordered_lag1_autocorrelation(values: np.ndarray) -> float:
    if len(values) < 2 or np.std(values[:-1]) == 0.0 or np.std(values[1:]) == 0.0:
        return float("nan")
    return float(np.corrcoef(values[:-1], values[1:])[0, 1])


def _time_ordered_component(
    frame: pd.DataFrame,
    values: np.ndarray,
    *,
    timestamp_col: str,
    component_mask: np.ndarray,
) -> np.ndarray:
    valid = np.isfinite(values) & component_mask
    part = frame.iloc[np.flatnonzero(valid)]
    order = np.lexsort(
        (part["__input_order__"].to_numpy(), part[timestamp_col].astype("int64").to_numpy())
    )
    return values[valid][order]


def _selection(frame: pd.DataFrame, fraction: float, *, rank_col: str | None) -> pd.DataFrame:
    count = max(1, int(np.ceil(len(frame) * fraction)))
    if rank_col:
        valid = frame.loc[np.isfinite(frame["__rank__"].to_numpy(dtype=float))]
        return valid.nsmallest(min(count, len(valid)), "__rank__", keep="first")
    return frame.nlargest(count, "__prediction__", keep="first")


def _bankroll_metrics(frame: pd.DataFrame, *, timestamp_col: str, pnl_col: str | None, size_col: str | None) -> dict[str, Any]:
    if not pnl_col or not size_col:
        return {"available": False, "reason": "requires both a PnL and sizing column"}
    pnl = pd.to_numeric(frame[pnl_col], errors="coerce").to_numpy(dtype=float)
    size = pd.to_numeric(frame[size_col], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(pnl) & np.isfinite(size) & (size > 0.0)
    if not valid.any():
        return {"available": False, "reason": "no rows have finite positive sizing and finite PnL"}
    part = frame.iloc[np.flatnonzero(valid)].copy()
    part["__pnl__"] = pnl[valid]
    part = part.sort_values([timestamp_col, "__input_order__"], kind="stable")
    curve = part["__pnl__"].cumsum().to_numpy(dtype=float)
    running_peak = np.maximum.accumulate(np.r_[0.0, curve])[:-1]
    drawdown = curve - running_peak
    return {
        "available": True,
        "pnl_column": pnl_col,
        "sizing_column": size_col,
        "sized_rows": int(valid.sum()),
        "sum_pnl": float(curve[-1]),
        "mean_pnl_per_trade": float(np.mean(part["__pnl__"])),
        "max_drawdown_pnl": float(np.min(drawdown)),
    }


def _hit_rate_metrics(
    frame: pd.DataFrame,
    *,
    timestamp_col: str,
    surprise_col: str | None,
    actual_hit_rate_col: str | None,
    expected_hit_rate_col: str | None,
) -> dict[str, Any]:
    if surprise_col:
        surprise = pd.to_numeric(frame[surprise_col], errors="coerce").to_numpy(dtype=float)
        source = surprise_col
    elif actual_hit_rate_col and expected_hit_rate_col:
        surprise = (
            pd.to_numeric(frame[actual_hit_rate_col], errors="coerce").to_numpy(dtype=float)
            - pd.to_numeric(frame[expected_hit_rate_col], errors="coerce").to_numpy(dtype=float)
        )
        source = f"{actual_hit_rate_col} - {expected_hit_rate_col}"
    else:
        return {"available": False, "reason": "no signed surprise or actual/expected hit-rate columns"}
    signed = _time_ordered_component(
        frame,
        surprise,
        timestamp_col=timestamp_col,
        component_mask=np.ones(len(frame), dtype=bool),
    )
    if not len(signed):
        return {"available": False, "reason": "signed hit-rate surprise has no finite values", "source": source}
    positive = np.maximum(signed, 0.0)
    negative = np.minimum(signed, 0.0)
    return {
        "available": True,
        "source": source,
        "lag_contract": "lag-1, ascending UTC timestamp, stable input-order tiebreak; positive/negative components preserve the full timeline with inactive observations equal to zero",
        "signed_rows": int(len(signed)),
        "signed_mean": float(np.mean(signed)),
        "signed_lag1_autocorrelation": _ordered_lag1_autocorrelation(signed),
        "positive_component_rows": int(np.count_nonzero(positive > 0.0)),
        "positive_component_support_rows": int(len(positive)),
        "positive_component_mean": float(np.mean(positive)),
        "positive_component_lag1_autocorrelation": _ordered_lag1_autocorrelation(positive),
        "negative_component_rows": int(np.count_nonzero(negative < 0.0)),
        "negative_component_support_rows": int(len(negative)),
        "negative_component_mean": float(np.mean(negative)),
        "negative_component_lag1_autocorrelation": _ordered_lag1_autocorrelation(negative),
    }


def _metrics(
    frame: pd.DataFrame,
    *,
    timestamp_col: str,
    net_ev_col: str,
    gross_ev_col: str,
    rank_col: str | None,
    pnl_col: str | None,
    size_col: str | None,
    surprise_col: str | None,
    actual_hit_rate_col: str | None,
    expected_hit_rate_col: str | None,
    clean_col: str | None,
    dirty_col: str | None,
) -> dict[str, Any]:
    net = pd.to_numeric(frame[net_ev_col], errors="coerce").to_numpy(dtype=float)
    prediction = frame["__prediction__"].to_numpy(dtype=float)
    valid = np.isfinite(net) & np.isfinite(prediction)
    part = frame.iloc[np.flatnonzero(valid)].copy()
    net = net[valid]
    prediction = prediction[valid]
    if not len(part):
        return {"rows": 0, "status": "unavailable_no_finite_target_and_prediction"}
    gross = (
        pd.to_numeric(part[gross_ev_col], errors="coerce").to_numpy(dtype=float)
        if gross_ev_col in part else np.full(len(part), np.nan)
    )
    residual = net - prediction
    exit_reason = part["execution_exit_reason"].fillna("").astype(str).str.lower() if "execution_exit_reason" in part else None
    output: dict[str, Any] = {
        "rows": int(len(part)),
        "days": int(part[timestamp_col].dt.floor("D").nunique()),
        "trades_per_day": float(len(part) / part[timestamp_col].dt.floor("D").nunique()),
        "mean_net_ev_per_trade": float(np.mean(net)),
        "sum_net_ev": float(np.sum(net)),
        "mean_gross_ev_per_trade": float(np.nanmean(gross)) if np.isfinite(gross).any() else float("nan"),
        "sum_gross_ev": float(np.nansum(gross)) if np.isfinite(gross).any() else float("nan"),
        "positive_net_ev_rate": float(np.mean(net > 0.0)),
        "signed_residual_definition": "realized_net_ev - prediction",
        "signed_residual_mean": float(np.mean(residual)),
        "signed_residual_lag1_autocorrelation": _lag1_autocorrelation(part, residual, timestamp_col),
        "residual_lag_contract": "lag-1, ascending UTC timestamp, stable input-order tiebreak",
        "stop_rate": float(exit_reason.str.contains("stop", regex=False).mean()) if exit_reason is not None else float("nan"),
        "timeout_rate": float(exit_reason.str.contains("timeout", regex=False).mean()) if exit_reason is not None else float("nan"),
        "positive_rate": float(np.mean(net > 0.0)),
        "clean_rate": _bool_rate(part, clean_col),
        "dirty_rate": _bool_rate(part, dirty_col),
        "hit_rate_surprise": _hit_rate_metrics(
            part,
            timestamp_col=timestamp_col,
            surprise_col=surprise_col,
            actual_hit_rate_col=actual_hit_rate_col,
            expected_hit_rate_col=expected_hit_rate_col,
        ),
        "bankroll": _bankroll_metrics(part, timestamp_col=timestamp_col, pnl_col=pnl_col, size_col=size_col),
    }
    for label, fraction in TOP_FRACTIONS:
        selected = _selection(part, fraction, rank_col=rank_col)
        selected_net = pd.to_numeric(selected[net_ev_col], errors="coerce").to_numpy(dtype=float)
        selected_gross = pd.to_numeric(selected[gross_ev_col], errors="coerce").to_numpy(dtype=float) if gross_ev_col in selected else np.full(len(selected), np.nan)
        prefix = f"top_{label}pct"
        output.update(
            {
                f"{prefix}_rows": int(len(selected)),
                f"{prefix}_mean_net_ev_per_trade": float(np.mean(selected_net)),
                f"{prefix}_sum_net_ev": float(np.sum(selected_net)),
                f"{prefix}_mean_gross_ev_per_trade": float(np.nanmean(selected_gross)) if np.isfinite(selected_gross).any() else float("nan"),
                f"{prefix}_sum_gross_ev": float(np.nansum(selected_gross)) if np.isfinite(selected_gross).any() else float("nan"),
                f"{prefix}_trades_per_day": float(len(selected) / part[timestamp_col].dt.floor("D").nunique()),
            }
        )
    return output


def _calendar_stability(
    frame: pd.DataFrame,
    *,
    calendar_col: str,
    label: str,
    net_ev_col: str,
    rank_col: str | None,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for top_label, fraction in TOP_FRACTIONS:
        values: list[float] = []
        for _, part in frame.groupby(calendar_col, observed=True, sort=True):
            selected = _selection(part, fraction, rank_col=rank_col)
            values.append(float(pd.to_numeric(selected[net_ev_col], errors="coerce").mean()))
        array = np.asarray(values, dtype=float)
        prefix = f"top_{top_label}pct"
        output[f"{prefix}_{label}_count"] = int(len(array))
        output[f"worst_{label}_{prefix}_mean_net_ev"] = float(np.nanmin(array)) if len(array) else float("nan")
        output[f"q05_{label}_{prefix}_mean_net_ev"] = float(np.nanquantile(array, 0.05)) if len(array) else float("nan")
        output[f"median_{label}_{prefix}_mean_net_ev"] = float(np.nanmedian(array)) if len(array) else float("nan")
    return output


def _selected_summary(
    frame: pd.DataFrame,
    *,
    timestamp_col: str,
    net_ev_col: str,
    gross_ev_col: str,
) -> dict[str, Any]:
    net = pd.to_numeric(frame[net_ev_col], errors="coerce")
    gross = (
        pd.to_numeric(frame[gross_ev_col], errors="coerce")
        if gross_ev_col in frame
        else pd.Series(np.nan, index=frame.index, dtype="float64")
    )
    days = frame[timestamp_col].dt.floor("D").nunique()
    return {
        "selected_rows": int(len(frame)),
        "selected_days": int(days),
        "selected_trades_per_day": float(len(frame) / days) if days else float("nan"),
        "selected_mean_net_ev_per_trade": float(net.mean()),
        "selected_sum_net_ev": float(net.sum()),
        "selected_mean_gross_ev_per_trade": float(gross.mean()),
        "selected_sum_gross_ev": float(gross.sum(min_count=1)),
    }


def _selected_grouped_summary(
    frame: pd.DataFrame,
    *,
    group_col: str,
    timestamp_col: str,
    net_ev_col: str,
    gross_ev_col: str,
) -> dict[str, dict[str, Any]]:
    if group_col not in frame:
        return {"unavailable": {"reason": f"ledger lacks {group_col!r}"}}
    work = pd.DataFrame(
        {
            "__group__": frame[group_col].fillna("missing").astype(str),
            "__day__": frame[timestamp_col].dt.floor("D"),
            "__net__": pd.to_numeric(frame[net_ev_col], errors="coerce"),
            "__gross__": (
                pd.to_numeric(frame[gross_ev_col], errors="coerce")
                if gross_ev_col in frame
                else np.nan
            ),
        }
    )
    grouped = work.groupby("__group__", observed=True, sort=True)
    summary = pd.DataFrame(
        {
            "selected_rows": grouped.size(),
            "selected_days": grouped["__day__"].nunique(),
            "selected_mean_net_ev_per_trade": grouped["__net__"].mean(),
            "selected_sum_net_ev": grouped["__net__"].sum(),
            "selected_mean_gross_ev_per_trade": grouped["__gross__"].mean(),
            "selected_sum_gross_ev": grouped["__gross__"].sum(min_count=1),
        }
    )
    summary["selected_trades_per_day"] = summary["selected_rows"] / summary["selected_days"]
    return {
        str(group): {key: _json_safe(value) for key, value in row.items()}
        for group, row in summary.to_dict(orient="index").items()
    }


def _global_tail_breakdowns(
    frame: pd.DataFrame,
    *,
    timestamp_col: str,
    side_col: str,
    archetype_col: str,
    net_ev_col: str,
    gross_ev_col: str,
    rank_col: str | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    valid = np.isfinite(pd.to_numeric(frame[net_ev_col], errors="coerce").to_numpy(dtype=float))
    valid &= np.isfinite(frame["__prediction__"].to_numpy(dtype=float))
    eligible = frame.iloc[np.flatnonzero(valid)]
    output: dict[str, Any] = {}
    flat_rows: list[dict[str, Any]] = []
    scopes = {
        "month": "__month__",
        "week": "__week__",
        "side": side_col,
        "base_archetype": archetype_col,
    }
    ordering = "rank_ascending" if rank_col else "prediction_descending"
    for label, fraction in TOP_FRACTIONS:
        selected = _selection(eligible, fraction, rank_col=rank_col)
        selection_basis = {
            "selection_scope": "global_oof_rows_within_arm",
            "fraction": fraction,
            "ordering": ordering,
            "grouping_contract": "selected globally once; groups are a breakdown with no within-group reselection",
        }
        item: dict[str, Any] = {
            "selection_basis": selection_basis,
            "overall": _selected_summary(
                selected,
                timestamp_col=timestamp_col,
                net_ev_col=net_ev_col,
                gross_ev_col=gross_ev_col,
            ),
        }
        for scope, group_col in scopes.items():
            grouped = _selected_grouped_summary(
                selected,
                group_col=group_col,
                timestamp_col=timestamp_col,
                net_ev_col=net_ev_col,
                gross_ev_col=gross_ev_col,
            )
            item[scope] = grouped
            for group, summary in grouped.items():
                if group == "unavailable":
                    continue
                flat_rows.append(
                    {
                        "scope": f"global_top_{label}pct_{scope}",
                        "group": group,
                        "selection_basis": "global_oof_rows_within_arm; " + ordering,
                        **summary,
                    }
                )
        output[f"top_{label}pct"] = item
    return output, flat_rows


def _scoped_metrics(arms: pd.DataFrame, *, timestamp_col: str, side_col: str, archetype_col: str, **kwargs: Any) -> tuple[dict[str, Any], pd.DataFrame]:
    work = arms.copy()
    work["__month__"] = work[timestamp_col].dt.strftime("%Y-%m")
    work["__week__"] = work[timestamp_col].dt.strftime("%G-W%V")
    scopes = {
        "overall": [],
        "month": ["__month__"],
        "week": ["__week__"],
        "side": [side_col] if side_col in work else [],
        "base_archetype": [archetype_col] if archetype_col in work else [],
    }
    nested: dict[str, Any] = {}
    flat_rows: list[dict[str, Any]] = []
    for arm, arm_frame in work.groupby("__arm__", sort=True, observed=True):
        arm_out: dict[str, Any] = {}
        for scope, keys in scopes.items():
            if scope == "overall":
                metric = _metrics(arm_frame, timestamp_col=timestamp_col, **kwargs)
                metric.update(_calendar_stability(arm_frame, calendar_col="__week__", label="week", net_ev_col=kwargs["net_ev_col"], rank_col=kwargs["rank_col"]))
                metric.update(_calendar_stability(arm_frame, calendar_col="__month__", label="month", net_ev_col=kwargs["net_ev_col"], rank_col=kwargs["rank_col"]))
                arm_out[scope] = {"all": metric}
                flat_rows.append({"arm": arm, "scope": scope, "group": "all", **_flat_metric(metric)})
            elif not keys:
                arm_out[scope] = {"unavailable": {"reason": f"ledger lacks {side_col if scope == 'side' else archetype_col!r}"}}
            else:
                groups: dict[str, Any] = {}
                for key, part in arm_frame.groupby(keys, observed=True, sort=True):
                    group = str(key)
                    metric = _metrics(part, timestamp_col=timestamp_col, **kwargs)
                    groups[group] = metric
                    flat_rows.append({"arm": arm, "scope": scope, "group": group, **_flat_metric(metric)})
                arm_out[scope] = groups
        global_breakdown, global_rows = _global_tail_breakdowns(
            arm_frame,
            timestamp_col=timestamp_col,
            side_col=side_col,
            archetype_col=archetype_col,
            net_ev_col=kwargs["net_ev_col"],
            gross_ev_col=kwargs["gross_ev_col"],
            rank_col=kwargs["rank_col"],
        )
        arm_out["global_top_tail_breakdown"] = global_breakdown
        flat_rows.extend({"arm": arm, **row} for row in global_rows)
        nested[str(arm)] = arm_out
    return nested, pd.DataFrame(flat_rows)


def _flat_metric(metric: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in metric.items() if not isinstance(value, dict)}


def _arm_comparisons(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for left, right in combinations(sorted(metrics), 2):
        a = metrics[left]["overall"]["all"]
        b = metrics[right]["overall"]["all"]
        row = {"left_arm": left, "right_arm": right}
        for key in ("mean_net_ev_per_trade", "sum_net_ev", "trades_per_day", "signed_residual_mean"):
            if isinstance(a.get(key), (int, float)) and isinstance(b.get(key), (int, float)):
                row[f"{right}_minus_{left}__{key}"] = float(b[key] - a[key])
        for label, _ in TOP_FRACTIONS:
            key = f"top_{label}pct_mean_net_ev_per_trade"
            if isinstance(a.get(key), (int, float)) and isinstance(b.get(key), (int, float)):
                row[f"{right}_minus_{left}__{key}"] = float(b[key] - a[key])
        rows.append(row)
    return rows


def run_report(
    inputs: Sequence[Path] | Path,
    output_dir: Path,
    *,
    arm_names: Sequence[str] | None = None,
    timestamp_col: str = DEFAULT_TIMESTAMP_COL,
    net_ev_col: str = DEFAULT_NET_EV_COL,
    gross_ev_col: str = DEFAULT_GROSS_EV_COL,
    side_col: str = DEFAULT_SIDE_COL,
    archetype_col: str = DEFAULT_ARCHETYPE_COL,
    prediction_cols: Sequence[str] | None = None,
    prediction_col: str | None = None,
    arm_col: str | None = None,
    rank_col: str | None = None,
    oof_col: str | None = None,
    fold_col: str = DEFAULT_OOF_FOLD_COL,
    pnl_col: str | None = None,
    size_col: str | None = None,
    hit_rate_surprise_col: str | None = None,
    actual_hit_rate_col: str | None = None,
    expected_hit_rate_col: str | None = None,
    clean_col: str | None = None,
    dirty_col: str | None = None,
) -> dict[str, Any]:
    if isinstance(inputs, Path):
        inputs = [inputs]
    if not inputs:
        raise ValueError("at least one --input is required")
    arms, sources = _build_arms(
        inputs, arm_names, timestamp_col=timestamp_col, net_ev_col=net_ev_col,
        prediction_cols=prediction_cols, prediction_col=prediction_col, arm_col=arm_col,
        rank_col=rank_col, oof_col=oof_col, fold_col=fold_col,
    )
    pnl_col = pnl_col or _first_present(arms.columns, PNL_CANDIDATES)
    size_col = size_col or _first_present(arms.columns, SIZE_CANDIDATES)
    surprise_col = hit_rate_surprise_col or _first_present(arms.columns, SURPRISE_CANDIDATES)
    actual_hit_rate_col = actual_hit_rate_col or _first_present(arms.columns, ACTUAL_HIT_RATE_CANDIDATES)
    expected_hit_rate_col = expected_hit_rate_col or _first_present(arms.columns, EXPECTED_HIT_RATE_CANDIDATES)
    clean_col = clean_col or _first_present(arms.columns, CLEAN_CANDIDATES)
    dirty_col = dirty_col or _first_present(arms.columns, DIRTY_CANDIDATES)
    metric_args = {
        "net_ev_col": net_ev_col, "gross_ev_col": gross_ev_col, "rank_col": rank_col,
        "pnl_col": pnl_col, "size_col": size_col, "surprise_col": surprise_col,
        "actual_hit_rate_col": actual_hit_rate_col, "expected_hit_rate_col": expected_hit_rate_col,
        "clean_col": clean_col, "dirty_col": dirty_col,
    }
    metrics, csv = _scoped_metrics(
        arms, timestamp_col=timestamp_col, side_col=side_col, archetype_col=archetype_col,
        **metric_args,
    )
    comparisons = _arm_comparisons(metrics)
    manifest = {
        "schema": "report_execution_ev_meta_oof_v1",
        "sources": sources,
        "reported_rows": int(len(arms)),
        "arms": sorted(metrics),
        "columns": {
            "timestamp": timestamp_col, "net_ev": net_ev_col, "gross_ev": gross_ev_col,
            "side": side_col, "base_archetype": archetype_col, "rank": rank_col,
            "pnl": pnl_col, "sizing": size_col, "signed_hit_rate_surprise": surprise_col,
        },
        "oof_contract": "Rows require a per-prediction OOF indicator or a non-negative integer OOF fold.",
        "residual_contract": "Economic residual is realized net EV minus prediction; lag-1 is time-ordered UTC with stable input-order tiebreak.",
        "selection_contract": "Top tails are sorted descending by prediction, or ascending by --rank-col when supplied. Calendar stability selects within each calendar group; global_top_tail_breakdown selects once across all OOF rows for an arm, then only groups those selected rows.",
        "status": "evaluation_only_not_policy_selection",
    }
    payload = {"manifest": manifest, "metrics": metrics, "arm_comparisons": comparisons}
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "execution_ev_oof_metrics.json"
    csv_path = output_dir / "execution_ev_oof_metrics.csv"
    manifest_path = output_dir / "execution_ev_oof_manifest.json"
    json_path.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    csv.to_csv(csv_path, index=False)
    manifest_path.write_text(json.dumps(_json_safe(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"manifest": manifest, "metrics": metrics, "arm_comparisons": comparisons, "paths": {"json": json_path, "csv": csv_path, "manifest": manifest_path}}


def _parse_columns(value: str) -> list[str]:
    result = [item.strip() for item in value.split(",") if item.strip()]
    if not result:
        raise argparse.ArgumentTypeError("at least one column is required")
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True, help="OOF parquet, CSV, or pickle ledger; repeat for auxiliary arms.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--arm-name", action="append", help="Arm label for each repeated --input.")
    parser.add_argument("--timestamp-col", default=DEFAULT_TIMESTAMP_COL)
    parser.add_argument("--net-ev-col", default=DEFAULT_NET_EV_COL)
    parser.add_argument("--gross-ev-col", default=DEFAULT_GROSS_EV_COL)
    parser.add_argument("--side-col", default=DEFAULT_SIDE_COL)
    parser.add_argument("--archetype-col", default=DEFAULT_ARCHETYPE_COL)
    parser.add_argument("--prediction-cols", type=_parse_columns, help="Comma-separated wide prediction arms; otherwise discover direct/residual/baseline arms.")
    parser.add_argument("--prediction-col", help="Single prediction column, required with --arm-col.")
    parser.add_argument("--arm-col", help="Long-format arm column.")
    parser.add_argument("--rank-col", help="Optional rank, where lower is selected first instead of higher prediction.")
    parser.add_argument("--oof-col", help="Explicit OOF boolean/numeric indicator, shared by all arms.")
    parser.add_argument("--fold-col", default=DEFAULT_OOF_FOLD_COL)
    parser.add_argument("--pnl-col")
    parser.add_argument("--size-col")
    parser.add_argument("--hit-rate-surprise-col")
    parser.add_argument("--actual-hit-rate-col")
    parser.add_argument("--expected-hit-rate-col")
    parser.add_argument("--clean-col")
    parser.add_argument("--dirty-col")
    return parser


def main() -> None:
    args = _parser().parse_args()
    try:
        result = run_report(
            args.input, args.output_dir, arm_names=args.arm_name, timestamp_col=args.timestamp_col,
            net_ev_col=args.net_ev_col, gross_ev_col=args.gross_ev_col, side_col=args.side_col,
            archetype_col=args.archetype_col, prediction_cols=args.prediction_cols,
            prediction_col=args.prediction_col, arm_col=args.arm_col, rank_col=args.rank_col,
            oof_col=args.oof_col, fold_col=args.fold_col, pnl_col=args.pnl_col, size_col=args.size_col,
            hit_rate_surprise_col=args.hit_rate_surprise_col, actual_hit_rate_col=args.actual_hit_rate_col,
            expected_hit_rate_col=args.expected_hit_rate_col, clean_col=args.clean_col, dirty_col=args.dirty_col,
        )
    except ValueError as exc:
        raise SystemExit(f"execution-EV OOF report failed: {exc}") from exc
    for name, path in result["paths"].items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
