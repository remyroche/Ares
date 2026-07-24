#!/usr/bin/env python3
"""Report top-k train_meta ablation metrics by side, archetype, and month.

The S52 train_meta smoke writes one prediction parquet per ablation arm.  This
reporter keeps the evaluation aligned with the deployment use case: rank by the
arm's meta score, inspect top 10/20/30%, and treat EV/path quality as primary
metrics.  A base-score comparator is included for each arm when available.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PREDICTION_NAME = "s52_train_meta_regime_handoff_smoke_predictions.parquet"
SUMMARY_NAME = "s52_train_meta_regime_handoff_smoke_summary.csv"
THRESHOLD_NAME = "s52_train_meta_regime_handoff_threshold_policy_summary.csv"
TOP_FRACS = (0.10, 0.20, 0.30)
BASELINE_ARM = "baseline_current_full_context"
DELTA_METRICS = (
    "selected_rows",
    "trades_per_day",
    "mean_ev_after_1pct",
    "sum_ev_after_1pct",
    "mean_exec_margin",
    "sum_exec_margin",
    "positive_ev_rate",
    "positive_exec_margin_rate",
    "clean_exec_precision",
    "ev_weighted_clean_precision",
    "dirty_positive_rate",
    "full_path_bad_mae_rate",
    "first_touch_bad_mae_rate",
    "timeout_rate",
    "stop_or_adverse_rate",
    "mfe_before_mae_rate",
    "mae_before_mfe_rate",
    "signed_hit_surprise_mean",
    "signed_hit_surprise_autocorr_lag1",
    "positive_hit_surprise_autocorr_lag1",
    "negative_hit_surprise_autocorr_lag1",
    "worst_day_mean_ev_after_1pct",
    "worst_week_mean_ev_after_1pct",
    "worst_month_mean_ev_after_1pct",
)
DELTA_KEY_COLUMNS = (
    "selector",
    "score_col",
    "top_frac",
    "scope",
    "selection_basis",
    "month",
    "week_start",
    "side_name",
    "archetype_label_family",
    "archetype_policy_key",
)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if pd.isna(value):
        return None
    return value


def _num(frame: pd.DataFrame, col: str, default: float = np.nan) -> pd.Series:
    if col not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=np.float32)
    return pd.to_numeric(frame[col], errors="coerce")


def _arm_name(path: Path) -> str:
    name = path.name
    if name.startswith("trial_"):
        parts = name.split("_", 2)
        if len(parts) == 3:
            return parts[2]
    return name


def _score_columns(frame: pd.DataFrame) -> list[str]:
    cols = ["score_base"] if "score_base" in frame.columns else []
    for col in ("score_meta_base_soft_label", "score_meta_context_hint_blend"):
        if col in frame.columns:
            cols.append(col)
    cols.extend(
        col
        for col in frame.columns
        if col.startswith("score_meta_") and col not in set(cols)
    )
    return cols


def _safe_autocorr(values: pd.Series) -> float:
    clean = (
        pd.to_numeric(values, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    if len(clean) < 3:
        return float("nan")
    left = clean.iloc[:-1].to_numpy(dtype=np.float64)
    right = clean.iloc[1:].to_numpy(dtype=np.float64)
    if len(left) < 2 or float(np.std(left)) <= 1e-12 or float(np.std(right)) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def _selected_temporal_metrics(
    selected: pd.DataFrame,
    *,
    score_col: str,
    clean: pd.Series,
    ev: pd.Series,
) -> dict[str, Any]:
    """Report realized surprise persistence on the selected OOS rows only.

    The expected clean probability is the selector's own score, clipped to the
    binary-label range. This is intentionally an evaluation statistic, not a
    recent-performance feature or a policy input.
    """
    ts_col = "__ts__" if "__ts__" in selected.columns else "timestamp"
    if ts_col not in selected.columns:
        return {}
    ts = pd.to_datetime(selected[ts_col], utc=True, errors="coerce")
    score = _num(selected, score_col).clip(0.0, 1.0)
    usable = ts.notna() & clean.notna() & score.notna() & ev.notna()
    if int(usable.sum()) == 0:
        return {}
    local = pd.DataFrame(
        {
            "day": ts.loc[usable].dt.floor("D").dt.tz_localize(None),
            "signed_surprise": clean.loc[usable].astype(float).to_numpy()
            - score.loc[usable].astype(float).to_numpy(),
            "ev": ev.loc[usable].astype(float).to_numpy(),
        }
    )
    daily = local.groupby("day", observed=True, sort=True).mean(numeric_only=True)
    weekly = (
        local.assign(week_start=local["day"].dt.to_period("W-SUN").dt.start_time)
        .groupby("week_start", observed=True, sort=True)["ev"]
        .mean()
    )
    monthly = (
        local.assign(month=local["day"].dt.to_period("M").astype(str))
        .groupby("month", observed=True, sort=True)["ev"]
        .mean()
    )
    signed = daily["signed_surprise"]
    positive = signed.clip(lower=0.0)
    negative = (-signed).clip(lower=0.0)
    return {
        "daily_surprise_days": int(len(daily)),
        "signed_hit_surprise_mean": float(signed.mean()),
        "signed_hit_surprise_autocorr_lag1": _safe_autocorr(signed),
        "positive_hit_surprise_autocorr_lag1": _safe_autocorr(positive),
        "negative_hit_surprise_autocorr_lag1": _safe_autocorr(negative),
        "worst_day_mean_ev_after_1pct": float(daily["ev"].min()),
        "worst_week_mean_ev_after_1pct": float(weekly.min())
        if len(weekly)
        else float("nan"),
        "worst_month_mean_ev_after_1pct": float(monthly.min())
        if len(monthly)
        else float("nan"),
    }


def _metric_row(
    rows: pd.DataFrame,
    *,
    arm: str,
    selector: str,
    score_col: str,
    top_frac: float,
    scope: str,
    group_values: dict[str, Any],
    preselected: bool = False,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "arm": arm,
        "selector": selector,
        "score_col": score_col,
        "top_frac": float(top_frac),
        "scope": scope,
        "selection_basis": "global_topk" if preselected else "within_scope_topk",
        **group_values,
    }
    out["candidate_rows"] = int(len(rows))
    score = _num(rows, score_col)
    valid = rows.loc[score.notna()].copy()
    valid["_score"] = score.loc[valid.index].astype(np.float32)
    out["scored_rows"] = int(len(valid))
    if valid.empty:
        out["selected_rows"] = 0
        return out
    if preselected:
        selected = valid
    else:
        n = max(1, int(math.ceil(len(valid) * float(top_frac))))
        selected = valid.sort_values("_score", ascending=False).head(n)
    out["selected_rows"] = int(len(selected))
    out["selected_share"] = float(len(selected) / len(valid))
    ts_col = "__ts__" if "__ts__" in selected.columns else "timestamp"
    selected_ts = (
        pd.to_datetime(selected.get(ts_col), utc=True, errors="coerce")
        if ts_col in selected.columns
        else pd.Series([], dtype="datetime64[ns]")
    )
    selected_days = int(selected_ts.dt.date.nunique()) if len(selected_ts) else 0
    out["selected_days"] = selected_days
    out["trades_per_day"] = (
        float(len(selected) / selected_days) if selected_days > 0 else float("nan")
    )
    ev = _num(selected, "ev_after_1pct")
    exec_margin = _num(selected, "exec_margin")
    clean = _num(selected, "clean_exec")
    clean_label = _num(selected, "clean_exec_label")
    if clean.isna().all() and not clean_label.isna().all():
        clean = clean_label
    dirty = _num(selected, "dirty_positive")
    bad_mae = _num(selected, "full_path_bad_mae_1r")
    first_bad_mae = _num(selected, "first_touch_bad_mae_1r")
    timeout = _num(selected, "timeout")
    stop_or_adverse = pd.Series(np.nan, index=selected.index, dtype=np.float32)
    for col in (
        "full_stop_loss",
        "full_path_stop_loss",
        "stop_loss",
        "stop_hit",
        "first_touch_stop",
        "adverse_path",
        "full_path_adverse",
    ):
        vals = _num(selected, col)
        if vals.notna().any():
            stop_or_adverse = vals.fillna(0.0).clip(0.0, 1.0)
            break
    mfe_first = _num(selected, "mfe_before_mae_1r")
    mae_first = _num(selected, "mae_before_mfe_1r")
    positive_ev = ev.gt(0.0)
    positive_exec = exec_margin.gt(0.0)
    ev_pos_weight = ev.clip(lower=0.0).fillna(0.0)
    weight_sum = float(ev_pos_weight.sum())
    out.update(
        {
            "mean_ev_after_1pct": float(ev.mean()),
            "sum_ev_after_1pct": float(ev.sum()),
            "mean_exec_margin": float(exec_margin.mean()),
            "sum_exec_margin": float(exec_margin.sum()),
            "positive_ev_rate": float(positive_ev.mean()),
            "positive_exec_margin_rate": float(positive_exec.mean()),
            "clean_exec_precision": float(clean.mean()),
            "ev_weighted_clean_precision": float(
                (ev_pos_weight * clean.fillna(0.0)).sum() / weight_sum
            )
            if weight_sum > 0.0
            else float("nan"),
            "dirty_positive_rate": float(dirty.mean()),
            "full_path_bad_mae_rate": float(bad_mae.mean()),
            "first_touch_bad_mae_rate": float(first_bad_mae.mean()),
            "timeout_rate": float(timeout.mean()),
            "stop_or_adverse_rate": float(stop_or_adverse.mean()),
            "mfe_before_mae_rate": float(mfe_first.mean()),
            "mae_before_mfe_rate": float(mae_first.mean()),
            "mean_score": float(selected["_score"].mean()),
            "min_score": float(selected["_score"].min()),
        }
    )
    out.update(
        _selected_temporal_metrics(selected, score_col=score_col, clean=clean, ev=ev)
    )
    for col in (
        "long_path_clean_exec_label",
        "long_path_dirty_positive_label",
        "long_path_slow_profit",
        "long_path_post_mfe_bad_drawdown",
        "long_bad_path_label",
    ):
        values = _num(selected, col)
        if values.notna().any():
            out[f"{col}_rate"] = float(values.mean())
    return out


def _group_metrics(
    frame: pd.DataFrame,
    *,
    arm: str,
    selector: str,
    score_col: str,
    scope: str,
    group_cols: list[str],
    min_group_rows: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if group_cols:
        grouped = frame.groupby(group_cols, dropna=False, sort=True)
        iterator = grouped
    else:
        iterator = [((), frame)]
    for keys, group in iterator:
        if len(group) < int(min_group_rows):
            continue
        if not isinstance(keys, tuple):
            keys = (keys,)
        group_values = {col: key for col, key in zip(group_cols, keys)}
        for frac in TOP_FRACS:
            rows.append(
                _metric_row(
                    group,
                    arm=arm,
                    selector=selector,
                    score_col=score_col,
                    top_frac=frac,
                    scope=scope,
                    group_values=group_values,
                )
            )
    return pd.DataFrame(rows)


def _global_topk_breakdown_metrics(
    frame: pd.DataFrame,
    *,
    arm: str,
    selector: str,
    score_col: str,
    group_specs: list[tuple[str, list[str]]],
    min_group_rows: int,
) -> pd.DataFrame:
    """Decompose the actual global top-k book without locally re-ranking it."""
    score = _num(frame, score_col)
    valid = frame.loc[score.notna()].copy()
    if valid.empty:
        return pd.DataFrame()
    valid["_score"] = score.loc[valid.index].astype(np.float32)
    rows: list[dict[str, Any]] = []
    for frac in TOP_FRACS:
        n = max(1, int(math.ceil(len(valid) * float(frac))))
        selected_global = valid.sort_values("_score", ascending=False).head(n)
        for scope, group_cols in group_specs:
            if not group_cols:
                continue
            missing = [col for col in group_cols if col not in selected_global.columns]
            if missing:
                continue
            for keys, group in selected_global.groupby(
                group_cols, dropna=False, sort=True
            ):
                if len(group) < int(min_group_rows):
                    continue
                if not isinstance(keys, tuple):
                    keys = (keys,)
                group_values = {col: key for col, key in zip(group_cols, keys)}
                rows.append(
                    _metric_row(
                        group,
                        arm=arm,
                        selector=selector,
                        score_col=score_col,
                        top_frac=frac,
                        scope=f"global_topk_{scope}",
                        group_values=group_values,
                        preselected=True,
                    )
                )
    return pd.DataFrame(rows)


def _topk_selected(frame: pd.DataFrame, score_col: str, top_frac: float = 0.10) -> pd.DataFrame:
    score = _num(frame, score_col)
    valid = frame.loc[score.notna()].copy()
    if valid.empty:
        return valid
    valid["_score"] = score.loc[valid.index].astype(np.float32)
    n = max(1, int(math.ceil(len(valid) * float(top_frac))))
    return valid.sort_values("_score", ascending=False).head(n)


def _high_surprise_event_metrics(
    frame: pd.DataFrame,
    *,
    arm: str,
    selector: str,
    score_col: str,
    top_frac: float = 0.10,
    min_event_rows: int = 3,
) -> pd.DataFrame:
    """Daily side x archetype surprise events on the actual selected top tail."""

    selected = _topk_selected(frame, score_col, top_frac)
    if selected.empty or "__ts__" not in selected.columns:
        return pd.DataFrame()
    ts = pd.to_datetime(selected["__ts__"], utc=True, errors="coerce")
    clean = _num(selected, "clean_exec")
    clean_label = _num(selected, "clean_exec_label")
    if clean.isna().all() and clean_label.notna().any():
        clean = clean_label
    score = _num(selected, score_col).clip(0.0, 1.0)
    ev = _num(selected, "ev_after_1pct")
    dirty = _num(selected, "dirty_positive")
    bad = _num(selected, "full_path_bad_mae_1r")
    timeout = _num(selected, "timeout")
    arch = (
        selected["archetype_policy_key"].astype(str)
        if "archetype_policy_key" in selected.columns
        else selected.get(
            "archetype_label_family",
            pd.Series("missing", index=selected.index),
        ).astype(str)
    )
    work = pd.DataFrame(
        {
            "event_day": ts.dt.floor("D").dt.tz_localize(None),
            "side_name": selected.get(
                "side_name", pd.Series("missing", index=selected.index)
            )
            .astype(str)
            .str.lower(),
            "archetype_policy_key": arch,
            "signed_hit_surprise": clean.astype(float) - score.astype(float),
            "positive_hit_surprise": (clean.astype(float) - score.astype(float)).clip(
                lower=0.0
            ),
            "negative_hit_surprise": (
                -(clean.astype(float) - score.astype(float))
            ).clip(lower=0.0),
            "ev_after_1pct": ev,
            "clean_exec": clean,
            "dirty_positive": dirty,
            "full_path_bad_mae_1r": bad,
            "timeout": timeout,
            "score": score,
        }
    )
    work = work.loc[work["event_day"].notna()]
    if work.empty:
        return pd.DataFrame()
    grouped = work.groupby(
        ["event_day", "side_name", "archetype_policy_key"],
        sort=True,
        dropna=False,
    ).agg(
        selected_rows=("signed_hit_surprise", "size"),
        mean_score=("score", "mean"),
        signed_hit_surprise_mean=("signed_hit_surprise", "mean"),
        positive_hit_surprise_mean=("positive_hit_surprise", "mean"),
        negative_hit_surprise_mean=("negative_hit_surprise", "mean"),
        mean_ev_after_1pct=("ev_after_1pct", "mean"),
        clean_exec_precision=("clean_exec", "mean"),
        dirty_positive_rate=("dirty_positive", "mean"),
        full_path_bad_mae_rate=("full_path_bad_mae_1r", "mean"),
        timeout_rate=("timeout", "mean"),
    )
    grouped = grouped.loc[grouped["selected_rows"].ge(int(min_event_rows))].reset_index()
    if grouped.empty:
        return grouped
    grouped.insert(0, "top_frac", float(top_frac))
    grouped.insert(0, "score_col", score_col)
    grouped.insert(0, "selector", selector)
    grouped.insert(0, "arm", arm)
    grouped["abs_hit_surprise_mean"] = grouped["signed_hit_surprise_mean"].abs()
    grouped["surprise_direction"] = np.where(
        grouped["signed_hit_surprise_mean"].lt(0.0), "negative", "positive"
    )
    return grouped


def _high_surprise_event_deltas(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty or "arm" not in events.columns:
        return pd.DataFrame()
    key_cols = [
        "selector",
        "score_col",
        "top_frac",
        "event_day",
        "side_name",
        "archetype_policy_key",
    ]
    metric_cols = [
        "selected_rows",
        "signed_hit_surprise_mean",
        "positive_hit_surprise_mean",
        "negative_hit_surprise_mean",
        "abs_hit_surprise_mean",
        "mean_ev_after_1pct",
        "clean_exec_precision",
        "dirty_positive_rate",
        "full_path_bad_mae_rate",
        "timeout_rate",
    ]
    baseline = events.loc[
        events["arm"].eq(BASELINE_ARM), key_cols + metric_cols
    ].copy()
    if baseline.empty:
        return pd.DataFrame()
    quantile_rows = []
    for keys, group in baseline.groupby(["selector", "score_col", "top_frac"], sort=False):
        cut = float(group["abs_hit_surprise_mean"].quantile(0.90))
        for _, row in group.iterrows():
            quantile_rows.append(
                {
                    "selector": keys[0],
                    "score_col": keys[1],
                    "top_frac": keys[2],
                    "event_day": row["event_day"],
                    "side_name": row["side_name"],
                    "archetype_policy_key": row["archetype_policy_key"],
                    "baseline_high_surprise_cut": cut,
                    "baseline_high_surprise_event": bool(
                        float(row["abs_hit_surprise_mean"]) >= cut
                    ),
                }
            )
    high_flags = pd.DataFrame(quantile_rows)
    baseline = baseline.rename(columns={col: f"baseline_{col}" for col in metric_cols})
    merged = events.merge(baseline, on=key_cols, how="left").merge(
        high_flags, on=key_cols, how="left"
    )
    for col in metric_cols:
        merged[f"delta_vs_{BASELINE_ARM}__{col}"] = pd.to_numeric(
            merged[col], errors="coerce"
        ) - pd.to_numeric(merged[f"baseline_{col}"], errors="coerce")
    baseline_abs = pd.to_numeric(
        merged["baseline_abs_hit_surprise_mean"], errors="coerce"
    )
    current_abs = pd.to_numeric(merged["abs_hit_surprise_mean"], errors="coerce")
    ev_delta = pd.to_numeric(
        merged[f"delta_vs_{BASELINE_ARM}__mean_ev_after_1pct"], errors="coerce"
    )
    merged["surprise_abs_contraction"] = baseline_abs - current_abs
    merged["surprise_abs_contraction_frac"] = (
        merged["surprise_abs_contraction"] / baseline_abs.replace(0.0, np.nan)
    )
    merged["high_surprise_significantly_improved"] = (
        merged["baseline_high_surprise_event"].fillna(False).astype(bool)
        & (
            merged["surprise_abs_contraction_frac"].ge(0.20)
            | ev_delta.ge(0.0)
        )
    )
    return merged


def _load_arm_predictions(arm_dir: Path) -> pd.DataFrame | None:
    path = arm_dir / PREDICTION_NAME
    if path.exists():
        frame = pd.read_parquet(path)
    else:
        shard_dir = arm_dir / "prediction_shards"
        shard_paths = sorted(shard_dir.glob("*.parquet")) if shard_dir.exists() else []
        if not shard_paths:
            return None
        frame = pd.concat((pd.read_parquet(p) for p in shard_paths), ignore_index=True)
    if "month" not in frame.columns and "__ts__" in frame.columns:
        frame["month"] = (
            pd.to_datetime(frame["__ts__"], errors="coerce")
            .dt.to_period("M")
            .astype(str)
        )
    if "week_start" not in frame.columns and "__ts__" in frame.columns:
        ts = pd.to_datetime(frame["__ts__"], utc=True, errors="coerce")
        # W-SUN produces Monday-starting weeks, unlike W-MON (Tuesday start).
        frame["week_start"] = (
            ts.dt.tz_localize(None).dt.to_period("W-SUN").dt.start_time.astype(str)
        )
    if "archetype_label_family" not in frame.columns:
        for col in (
            "__archetype_label_family__",
            "policy_archetype",
            "archetype_policy_key",
            "__archetype_policy_key__",
            "local_side_archetype",
            "source_family",
            "source_archetype",
            "source_semantic_family",
        ):
            if col in frame.columns:
                frame["archetype_label_family"] = frame[col].astype(str)
                break
    if (
        "archetype_policy_key" not in frame.columns
        and "__archetype_policy_key__" in frame.columns
    ):
        frame["archetype_policy_key"] = frame["__archetype_policy_key__"].astype(str)
    return frame


def _add_baseline_deltas(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty or "arm" not in metrics.columns:
        return pd.DataFrame()
    key_cols = [col for col in DELTA_KEY_COLUMNS if col in metrics.columns]
    metric_cols = [col for col in DELTA_METRICS if col in metrics.columns]
    if not key_cols or not metric_cols:
        return pd.DataFrame()
    baseline = metrics.loc[
        metrics["arm"].eq(BASELINE_ARM), key_cols + metric_cols
    ].copy()
    if baseline.empty:
        return pd.DataFrame()
    baseline = baseline.rename(columns={col: f"baseline_{col}" for col in metric_cols})
    merged = metrics.merge(baseline, on=key_cols, how="left")
    for col in metric_cols:
        merged[f"delta_vs_{BASELINE_ARM}__{col}"] = pd.to_numeric(
            merged[col], errors="coerce"
        ) - pd.to_numeric(merged[f"baseline_{col}"], errors="coerce")
    return merged


def _add_base_score_deltas(metrics: pd.DataFrame) -> pd.DataFrame:
    """Compare every meta selector to the base score on identical OOS scopes."""
    if metrics.empty or "selector" not in metrics.columns:
        return pd.DataFrame()
    metric_cols = [col for col in DELTA_METRICS if col in metrics.columns]
    if not metric_cols:
        return pd.DataFrame()
    key_candidates = (
        "arm",
        "top_frac",
        "scope",
        "selection_basis",
        "month",
        "week_start",
        "side_name",
        "archetype_label_family",
        "archetype_policy_key",
    )
    key_cols = [col for col in key_candidates if col in metrics.columns]
    base = metrics.loc[
        metrics["selector"].eq("base_score"), key_cols + metric_cols
    ].copy()
    if base.empty:
        return pd.DataFrame()
    base = base.rename(columns={col: f"base_score_{col}" for col in metric_cols})
    compared = metrics.loc[~metrics["selector"].eq("base_score")].merge(
        base, on=key_cols, how="left"
    )
    for col in metric_cols:
        compared[f"delta_vs_base_score__{col}"] = pd.to_numeric(
            compared[col], errors="coerce"
        ) - pd.to_numeric(compared[f"base_score_{col}"], errors="coerce")
    return compared


def build_report(
    *, root_dir: Path, out_dir: Path, min_group_rows: int
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    arm_dirs = sorted(
        path
        for path in root_dir.iterdir()
        if path.is_dir()
        and (
            (path / PREDICTION_NAME).exists()
            or any((path / "prediction_shards").glob("*.parquet"))
        )
    )
    all_rows: list[pd.DataFrame] = []
    event_rows: list[pd.DataFrame] = []
    arm_summary_rows: list[dict[str, Any]] = []
    group_specs = [
        ("overall", []),
        ("month", ["month"]),
        ("week", ["week_start"]),
        ("side", ["side_name"]),
        ("archetype_family", ["archetype_label_family"]),
        ("archetype_policy", ["archetype_policy_key"]),
        ("month_side", ["month", "side_name"]),
        ("side_archetype_family", ["side_name", "archetype_label_family"]),
        (
            "month_side_archetype_family",
            ["month", "side_name", "archetype_label_family"],
        ),
        (
            "week_side_archetype_family",
            ["week_start", "side_name", "archetype_label_family"],
        ),
    ]
    for arm_dir in arm_dirs:
        frame = _load_arm_predictions(arm_dir)
        if frame is None or frame.empty:
            continue
        arm = _arm_name(arm_dir)
        score_cols = _score_columns(frame)
        manifest_path = arm_dir / "manifest.json"
        manifest = (
            json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        )
        arm_summary_rows.append(
            {
                "arm": arm,
                "arm_dir": str(arm_dir),
                "rows": int(len(frame)),
                "months": int(frame["month"].nunique(dropna=True))
                if "month" in frame.columns
                else 0,
                "sides": int(frame["side_name"].nunique(dropna=True))
                if "side_name" in frame.columns
                else 0,
                "archetype_families": int(
                    frame["archetype_label_family"].nunique(dropna=True)
                )
                if "archetype_label_family" in frame.columns
                else 0,
                "best_selector": (manifest.get("best_selector") or {}).get("selector"),
                "best_status": (manifest.get("best_selector") or {}).get(
                    "meta_smoke_status"
                ),
                "selected_feature_union_count": manifest.get(
                    "selected_feature_union_count"
                ),
            }
        )
        for score_col in score_cols:
            selector = (
                "base_score"
                if score_col == "score_base"
                else score_col.removeprefix("score_")
            )
            events = _high_surprise_event_metrics(
                frame,
                arm=arm,
                selector=selector,
                score_col=score_col,
                top_frac=0.10,
            )
            if not events.empty:
                event_rows.append(events)
            for scope, group_cols in group_specs:
                missing = [col for col in group_cols if col not in frame.columns]
                if missing:
                    continue
                metrics = _group_metrics(
                    frame,
                    arm=arm,
                    selector=selector,
                    score_col=score_col,
                    scope=scope,
                    group_cols=group_cols,
                    min_group_rows=1 if scope == "overall" else int(min_group_rows),
                )
                if not metrics.empty:
                    all_rows.append(metrics)
            global_breakdown = _global_topk_breakdown_metrics(
                frame,
                arm=arm,
                selector=selector,
                score_col=score_col,
                group_specs=group_specs,
                min_group_rows=int(min_group_rows),
            )
            if not global_breakdown.empty:
                all_rows.append(global_breakdown)
    metrics_df = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    delta_df = _add_baseline_deltas(metrics_df)
    base_delta_df = _add_base_score_deltas(metrics_df)
    events_df = (
        pd.concat(event_rows, ignore_index=True) if event_rows else pd.DataFrame()
    )
    event_delta_df = _high_surprise_event_deltas(events_df)
    arm_summary = pd.DataFrame(arm_summary_rows)
    metrics_path = out_dir / "train_meta_extended_pool_ablation_topk_metrics.csv"
    delta_path = out_dir / "train_meta_extended_pool_ablation_delta_vs_baseline.csv"
    base_delta_path = (
        out_dir / "train_meta_extended_pool_ablation_delta_vs_base_score.csv"
    )
    summary_path = out_dir / "train_meta_extended_pool_ablation_arm_summary.csv"
    events_path = out_dir / "train_meta_extended_pool_ablation_high_surprise_events.csv"
    event_delta_path = (
        out_dir / "train_meta_extended_pool_ablation_high_surprise_event_deltas.csv"
    )
    metrics_df.to_csv(metrics_path, index=False)
    delta_df.to_csv(delta_path, index=False)
    base_delta_df.to_csv(base_delta_path, index=False)
    events_df.to_csv(events_path, index=False)
    event_delta_df.to_csv(event_delta_path, index=False)
    arm_summary.to_csv(summary_path, index=False)
    outputs: dict[str, str] = {
        "all_metrics": str(metrics_path),
        "delta_vs_baseline": str(delta_path),
        "delta_vs_base_score": str(base_delta_path),
        "high_surprise_events": str(events_path),
        "high_surprise_event_deltas": str(event_delta_path),
        "arm_summary": str(summary_path),
    }
    for scope in (
        sorted(metrics_df["scope"].dropna().unique()) if not metrics_df.empty else []
    ):
        scoped = metrics_df.loc[metrics_df["scope"].eq(scope)].copy()
        path = out_dir / f"train_meta_extended_pool_ablation_{scope}_metrics.csv"
        scoped.to_csv(path, index=False)
        outputs[f"{scope}_metrics"] = str(path)
    manifest = {
        "generated_by": "report_train_meta_extended_pool_ablation_metrics",
        "root_dir": str(root_dir),
        "out_dir": str(out_dir),
        "arms": arm_summary_rows,
        "top_fracs": list(TOP_FRACS),
        "baseline_arm": BASELINE_ARM,
        "min_group_rows": int(min_group_rows),
        "high_surprise_event_contract": (
            "Daily side x archetype cells are built from selected top10 rows. "
            "Baseline high-surprise events are baseline-current cells above the "
            "90th percentile of absolute mean hit surprise for each selector. "
            "An arm significantly improves an event when absolute surprise "
            "contracts by at least 20% or mean EV is no worse."
        ),
        "outputs": outputs,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(_json_safe(manifest), indent=2, sort_keys=True)
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--min-group-rows", type=int, default=25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_report(
        root_dir=args.root_dir,
        out_dir=args.out_dir,
        min_group_rows=int(args.min_group_rows),
    )
    print(
        json.dumps(
            _json_safe(
                {"event": "train_meta_extended_pool_ablation_report_done", **manifest}
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
