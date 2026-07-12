"""Leakage-safe diagnostics for shared-state, local-response AE/GMM research.

The helpers in this module do not train a trading policy and do not consume
recent realized performance as an input.  They assess whether a frozen state
representation separates future outcomes in the two decision-relevant parts
of the base candidate book:

* incumbent top-10% candidates that may need demotion;
* the near-miss 10-20% band that may contain promotion candidates.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .local_economic_aegmm import LOCAL_ECONOMIC_AEGMM_PREFIX, _safe_token

VALIDATION_ZONES: tuple[str, ...] = (
    "incumbent_top10",
    "near_miss_top10_20",
)


def _numeric(frame: pd.DataFrame, name: str, default: float = 0.0) -> np.ndarray:
    if name not in frame.columns:
        return np.full(len(frame), np.float32(default), dtype=np.float32)
    values = pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=np.float32)
    return np.nan_to_num(
        values,
        nan=np.float32(default),
        posinf=np.float32(default),
        neginf=np.float32(default),
    )


def _autocorrelation(values: np.ndarray) -> float:
    array = np.asarray(values, dtype=np.float64)
    if len(array) < 5 or not np.isfinite(array).all():
        return float("nan")
    left = array[:-1]
    right = array[1:]
    if float(np.std(left)) <= 1e-10 or float(np.std(right)) <= 1e-10:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def _finite_mean(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return float(finite.mean()) if finite.size else float("nan")


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    valid = np.isfinite(left) & np.isfinite(right)
    if int(valid.sum()) < 5:
        return float("nan")
    lhs = pd.Series(left[valid]).rank(method="average").to_numpy(dtype=np.float64)
    rhs = pd.Series(right[valid]).rank(method="average").to_numpy(dtype=np.float64)
    if float(np.std(lhs)) <= 1e-10 or float(np.std(rhs)) <= 1e-10:
        return float("nan")
    return float(np.corrcoef(lhs, rhs)[0, 1])


def annotate_base_decision_zones(
    frame: pd.DataFrame,
    *,
    score_col: str = "score_base",
    timestamp_col: str = "__ts__",
) -> pd.DataFrame:
    """Attach a global candidate-book rank and the two state-validation zones."""

    if timestamp_col not in frame.columns:
        raise ValueError(f"State validation requires {timestamp_col!r}")
    if score_col not in frame.columns:
        raise ValueError(f"State validation requires observable {score_col!r}")
    result = frame.copy(deep=False)
    timestamp = pd.to_datetime(result[timestamp_col], utc=True, errors="coerce")
    score = pd.to_numeric(result[score_col], errors="coerce").fillna(0.0)
    rank = score.groupby(timestamp, sort=False).rank(method="average", pct=True)
    result["state_validation_base_rank_pct"] = rank.fillna(0.0).astype(np.float32)
    zone = np.full(len(result), "outside", dtype=object)
    rank_values = result["state_validation_base_rank_pct"].to_numpy(dtype=np.float32)
    zone[rank_values >= 0.90] = "incumbent_top10"
    zone[(rank_values >= 0.80) & (rank_values < 0.90)] = "near_miss_top10_20"
    result["state_validation_zone"] = pd.Categorical(
        zone,
        categories=["outside", *VALIDATION_ZONES],
        ordered=True,
    )
    return result


def _week_start(timestamp: pd.Series) -> pd.Series:
    normalized = pd.to_datetime(timestamp, utc=True, errors="coerce").dt.normalize()
    return normalized - pd.to_timedelta(normalized.dt.dayofweek, unit="D")


def _top_bottom_delta(
    score: np.ndarray, value: np.ndarray
) -> tuple[float, float, float]:
    finite = np.isfinite(score) & np.isfinite(value)
    if int(finite.sum()) < 12 or float(np.nanstd(score[finite])) <= 1e-8:
        return float("nan"), float("nan"), float("nan")
    order = np.argsort(score[finite], kind="stable")
    ordered = value[finite][order]
    count = max(1, int(np.ceil(ordered.size / 3.0)))
    lower = float(np.mean(ordered[:count]))
    upper = float(np.mean(ordered[-count:]))
    return upper, lower, upper - lower


def _metric_record(
    group: pd.DataFrame,
    *,
    fold: str,
    state_block: str,
    zone: str,
    scope: str,
) -> dict[str, Any]:
    ev = _numeric(group, "ev_after_1pct", np.nan)
    clean = _numeric(group, "clean_exec")
    dirty = _numeric(group, "dirty_positive")
    first_touch_bad_mae = _numeric(group, "first_touch_bad_mae_1r")
    timeout = _numeric(group, "timeout")
    base_score = _numeric(group, "score_base", 0.5)
    expected_ev = _numeric(group, "__state_expected_ev__", np.nan)
    expected_bad = _numeric(group, "__state_expected_bad_mae__", np.nan)
    hit_surprise = clean - base_score
    top_ev, bottom_ev, ev_lift = _top_bottom_delta(expected_ev, ev)
    top_clean, bottom_clean, clean_lift = _top_bottom_delta(expected_ev, clean)
    # A low predicted bad-MAE score should correspond to lower realized risk.
    low_bad, high_bad, bad_lift = _top_bottom_delta(expected_bad, first_touch_bad_mae)
    ts = pd.to_datetime(group["__ts__"], utc=True, errors="coerce")
    week = _week_start(ts)
    week_ev = pd.Series(ev, index=group.index).groupby(week, sort=True).mean()
    month_key = ts.dt.strftime("%Y-%m")
    month_ev = pd.Series(ev, index=group.index).groupby(month_key, sort=True).mean()
    finite_ev = np.isfinite(ev)
    return {
        "fold": str(fold),
        "state_block": str(state_block),
        "zone": str(zone),
        "scope": str(scope),
        "selected_rows": int(len(group)),
        "days": int(ts.dt.normalize().nunique()),
        "weeks": int(week.nunique()),
        "months": int(month_key.nunique()),
        "mean_ev_after_1pct": float(np.nanmean(ev))
        if finite_ev.any()
        else float("nan"),
        "sum_ev_after_1pct": float(np.nansum(ev)) if finite_ev.any() else float("nan"),
        "positive_ev_rate": float(np.mean(ev > 0.0))
        if finite_ev.any()
        else float("nan"),
        "clean_exec_precision": float(np.mean(clean >= 0.5)),
        "dirty_positive_rate": float(np.mean(dirty >= 0.5)),
        "first_touch_bad_mae_rate": float(np.mean(first_touch_bad_mae >= 0.5)),
        "timeout_rate": float(np.mean(timeout >= 0.5)),
        "mean_hit_surprise": float(np.mean(hit_surprise)),
        "negative_hit_surprise_mean": float(np.mean(np.minimum(hit_surprise, 0.0))),
        "positive_hit_surprise_mean": float(np.mean(np.maximum(hit_surprise, 0.0))),
        "state_expected_ev_mean": _finite_mean(expected_ev),
        "state_expected_bad_mae_mean": _finite_mean(expected_bad),
        "state_ev_calibration_error": _finite_mean(ev - expected_ev),
        "state_bad_mae_calibration_error": _finite_mean(
            first_touch_bad_mae - expected_bad
        ),
        "state_expected_ev_spearman": _spearman(expected_ev, ev),
        "state_expected_bad_mae_spearman": _spearman(expected_bad, first_touch_bad_mae),
        "expected_ev_top_tercile": top_ev,
        "expected_ev_bottom_tercile": bottom_ev,
        "expected_ev_top_minus_bottom": ev_lift,
        "expected_clean_top_tercile": top_clean,
        "expected_clean_bottom_tercile": bottom_clean,
        "expected_clean_top_minus_bottom": clean_lift,
        "pred_bad_mae_high_tercile": high_bad,
        "pred_bad_mae_low_tercile": low_bad,
        "pred_bad_mae_high_minus_low": bad_lift,
        "worst_week_mean_ev": float(week_ev.min())
        if not week_ev.empty
        else float("nan"),
        "worst_month_mean_ev": float(month_ev.min())
        if not month_ev.empty
        else float("nan"),
        "state_cluster_count": int(
            pd.to_numeric(group["__state_cluster__"], errors="coerce").nunique()
        ),
        "mean_state_support_log1p": _finite_mean(
            _numeric(group, "__state_support_log1p__", np.nan)
        ),
    }


def state_validation_metrics(
    predictions: pd.DataFrame,
    *,
    fold: str,
    state_block: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return OOS zone, state, daily-surprise, and autocorrelation diagnostics."""

    prefix = f"{LOCAL_ECONOMIC_AEGMM_PREFIX}{_safe_token(state_block)}_"
    required = {
        "__ts__",
        "side_name",
        "archetype_policy_key",
        "score_base",
        "ev_after_1pct",
        "clean_exec",
        "dirty_positive",
        "first_touch_bad_mae_1r",
        "timeout",
        f"{prefix}gmm_cluster_id",
    }
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(f"State validation missing required columns: {missing}")
    work = annotate_base_decision_zones(predictions)
    work = work.loc[work["state_validation_zone"].isin(VALIDATION_ZONES)].copy()
    work["__state_cluster__"] = (
        pd.to_numeric(work[f"{prefix}gmm_cluster_id"], errors="coerce")
        .round()
        .astype("Int16")
    )
    expected_top10 = f"{prefix}expected_top10_ev"
    expected_ev = f"{prefix}expected_ev"
    expected_bad = f"{prefix}expected_top10_bad_mae"
    fallback_bad = f"{prefix}expected_bad_mae"
    work["__state_expected_ev__"] = _numeric(
        work, expected_top10 if expected_top10 in work else expected_ev, np.nan
    )
    work["__state_expected_bad_mae__"] = _numeric(
        work, expected_bad if expected_bad in work else fallback_bad, np.nan
    )
    work["__state_support_log1p__"] = _numeric(work, f"{prefix}support_log1p", np.nan)
    work["__date__"] = pd.to_datetime(
        work["__ts__"], utc=True, errors="coerce"
    ).dt.normalize()
    work["__week_start__"] = _week_start(work["__ts__"])
    work["__hit_surprise__"] = _numeric(work, "clean_exec") - _numeric(
        work, "score_base", 0.5
    )

    summary_rows: list[dict[str, Any]] = []
    state_rows: list[dict[str, Any]] = []
    for zone, zone_frame in work.groupby(
        "state_validation_zone", observed=True, sort=True
    ):
        scope_groups = {
            "overall": [],
            "side_archetype": ["side_name", "archetype_policy_key"],
        }
        for scope, group_cols in scope_groups.items():
            grouped = (
                [((), zone_frame)]
                if not group_cols
                else zone_frame.groupby(group_cols, observed=True, sort=True)
            )
            for keys, group in grouped:
                if not isinstance(keys, tuple):
                    keys = (keys,)
                row = _metric_record(
                    group,
                    fold=fold,
                    state_block=state_block,
                    zone=str(zone),
                    scope=scope,
                )
                for name, value in zip(group_cols, keys, strict=False):
                    row[name] = value
                summary_rows.append(row)
        for keys, group in zone_frame.groupby(
            ["side_name", "archetype_policy_key", "__state_cluster__"],
            observed=True,
            dropna=False,
            sort=True,
        ):
            side, archetype, cluster = keys
            row = _metric_record(
                group,
                fold=fold,
                state_block=state_block,
                zone=str(zone),
                scope="side_archetype_state",
            )
            row.update(
                {
                    "side_name": side,
                    "archetype_policy_key": archetype,
                    "state_cluster": cluster,
                }
            )
            state_rows.append(row)

    daily = (
        work.groupby(
            ["state_validation_zone", "side_name", "archetype_policy_key", "__date__"],
            observed=True,
            sort=True,
        )
        .agg(
            selected_rows=("__hit_surprise__", "size"),
            mean_hit_surprise=("__hit_surprise__", "mean"),
            negative_hit_surprise=(
                "__hit_surprise__",
                lambda values: float(
                    np.minimum(values.to_numpy(dtype=np.float32), 0.0).mean()
                ),
            ),
            positive_hit_surprise=(
                "__hit_surprise__",
                lambda values: float(
                    np.maximum(values.to_numpy(dtype=np.float32), 0.0).mean()
                ),
            ),
            mean_ev_after_1pct=("ev_after_1pct", "mean"),
            clean_exec_precision=("clean_exec", "mean"),
            first_touch_bad_mae_rate=("first_touch_bad_mae_1r", "mean"),
            timeout_rate=("timeout", "mean"),
        )
        .reset_index()
        .rename(columns={"state_validation_zone": "zone", "__date__": "date"})
    )
    daily["fold"] = str(fold)
    daily["state_block"] = str(state_block)
    autocorr_rows: list[dict[str, Any]] = []
    for keys, group in daily.groupby(
        ["zone", "side_name", "archetype_policy_key"], observed=True, sort=True
    ):
        zone, side, archetype = keys
        ordered = group.sort_values("date", kind="stable")
        autocorr_rows.append(
            {
                "fold": str(fold),
                "state_block": str(state_block),
                "zone": str(zone),
                "side_name": side,
                "archetype_policy_key": archetype,
                "days": int(len(ordered)),
                "signed_hit_surprise_autocorr_lag1": _autocorrelation(
                    ordered["mean_hit_surprise"].to_numpy(dtype=np.float32)
                ),
                "negative_hit_surprise_autocorr_lag1": _autocorrelation(
                    ordered["negative_hit_surprise"].to_numpy(dtype=np.float32)
                ),
                "positive_hit_surprise_autocorr_lag1": _autocorrelation(
                    ordered["positive_hit_surprise"].to_numpy(dtype=np.float32)
                ),
                "worst_day_mean_ev": float(ordered["mean_ev_after_1pct"].min()),
                "mean_daily_ev": float(ordered["mean_ev_after_1pct"].mean()),
            }
        )
    return (
        pd.DataFrame(summary_rows),
        pd.DataFrame(state_rows),
        daily,
        pd.DataFrame(autocorr_rows),
    )


__all__ = [
    "VALIDATION_ZONES",
    "annotate_base_decision_zones",
    "state_validation_metrics",
]
