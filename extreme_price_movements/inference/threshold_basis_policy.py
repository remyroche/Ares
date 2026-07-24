"""Live threshold-basis policy transforms for portfolio inference.

The promoted policy is intentionally separated from the historical hit-rate
surprise layer. It turns a batch of live decisions into the same top-band rank
space used by the offline threshold-basis replay: historical prior rows define
reachable EV thresholds, while current rows are selected per timestamp batch.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Sequence

import numpy as np
import pandas as pd


_REFERENCE_CACHE: dict[str, pd.DataFrame] = {}
_INVALID_EXPECTED_EV_SENTINELS = (-1.0,)
_EXPECTED_EV_SENTINEL_ATOL = 1e-12
_EMAIL_ARCHETYPE_BASELINE_WINDOW_DAYS = 28


def _first_available_column(
    frame: pd.DataFrame,
    candidates: Sequence[str],
) -> str | None:
    for candidate in candidates:
        if candidate in frame.columns:
            return str(candidate)
    return None


def _binary_rate(frame: pd.DataFrame, column: str | None) -> float:
    if not column:
        return float("nan")
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(
        dtype=np.float64, copy=False
    )
    values = values[np.isfinite(values)]
    return float(np.mean(values > 0.5)) if values.size else float("nan")


def _robust_residual_retained_rows(
    reference: pd.DataFrame,
    *,
    trim_fraction: float,
    return_col: str,
    mapped_ev_col: str,
) -> tuple[pd.DataFrame, int, int, float]:
    """Keep the same outcome days retained by admission residual trimming.

    The diagnostic baseline must describe the information available at entry,
    not a later retrospective summary.  Its day-level trimming therefore uses
    the identical realized-minus-mapped-EV residual and median/IQR procedure as
    the admission correction.
    """

    if reference.empty:
        return reference, 0, 0, float("nan")
    if return_col not in reference.columns or mapped_ev_col not in reference.columns:
        return reference.iloc[0:0], 0, 0, float("nan")
    realized = pd.to_numeric(reference[return_col], errors="coerce")
    mapped = pd.to_numeric(reference[mapped_ev_col], errors="coerce")
    residual = (realized - mapped).replace([np.inf, -np.inf], np.nan)
    valid = residual.notna()
    work = reference.loc[valid]
    if work.empty:
        return work, 0, 0, float("nan")
    if trim_fraction <= 0.0:
        return work, int(work["__email_outcome_day__"].nunique()), 0, float("nan")

    daily = (
        pd.DataFrame(
            {
                "outcome_day": work["__email_outcome_day__"].to_numpy(copy=False),
                "residual": residual.loc[work.index].to_numpy(dtype=np.float64, copy=False),
            },
            index=work.index,
        )
        .dropna(subset=["outcome_day", "residual"])
        .groupby("outcome_day", sort=False, observed=True)["residual"]
        .mean()
    )
    if daily.empty:
        return work.iloc[0:0], 0, 0, float("nan")
    values = daily.to_numpy(dtype=np.float64, copy=False)
    median = float(np.median(values))
    q25, q75 = np.quantile(values, [0.25, 0.75])
    iqr = max(float(q75 - q25), 1e-8)
    keep = np.ones(len(daily), dtype=bool)
    if len(daily) >= 4:
        robust_z = (values - median) / iqr
        low, high = np.quantile(robust_z, [trim_fraction, 1.0 - trim_fraction])
        keep = (robust_z >= low) & (robust_z <= high)
    kept_days = daily.index[keep]
    retained = work.loc[work["__email_outcome_day__"].isin(kept_days)]
    return retained, int(keep.sum()), int((~keep).sum()), iqr


def _email_baseline_statistics(
    reference: pd.DataFrame,
    *,
    trim_fraction: float,
    return_col: str,
    mapped_ev_col: str,
    current_mapped_ev: float,
) -> dict[str, Any]:
    """Summarize a robust side/archetype reference for the close-email audit."""

    retained, retained_days, trimmed_days, daily_iqr = _robust_residual_retained_rows(
        reference,
        trim_fraction=trim_fraction,
        return_col=return_col,
        mapped_ev_col=mapped_ev_col,
    )
    result: dict[str, Any] = {
        "support": int(len(retained)),
        "retained_days": int(retained_days),
        "trimmed_days": int(trimmed_days),
        "daily_residual_iqr": daily_iqr,
    }
    if retained.empty:
        return result
    realized = pd.to_numeric(retained[return_col], errors="coerce").to_numpy(
        dtype=np.float64, copy=False
    )
    realized = realized[np.isfinite(realized)]
    if realized.size:
        q25, q75 = np.quantile(realized, [0.25, 0.75])
        result.update(
            {
                "ev_mean": float(np.mean(realized)),
                "ev_median": float(np.median(realized)),
                "ev_iqr": float(q75 - q25),
                "positive_ev_rate": float(np.mean(realized > 0.0)),
            }
        )
    mae_col = _first_available_column(
        retained,
        ("first_touch_mae_to_sl", "__first_touch_mae_to_sl__"),
    )
    if mae_col:
        returns = pd.to_numeric(retained[return_col], errors="coerce").to_numpy(
            dtype=np.float64, copy=False
        )
        mae = pd.to_numeric(retained[mae_col], errors="coerce").to_numpy(
            dtype=np.float64, copy=False
        )
        successful_mae = mae[(returns > 0.0) & np.isfinite(mae) & (mae >= 0.0)]
        result["successful_trade_mae_to_sl_support"] = int(successful_mae.size)
        if successful_mae.size:
            result["successful_trade_mae_to_sl_mean"] = float(np.mean(successful_mae))
    for destination, candidates in (
        ("clean_rate", ("clean_exec", "clean_positive")),
        ("dirty_positive_rate", ("dirty_positive",)),
        (
            "take_profit_rate",
            ("first_touch_tp_hit", "__first_touch_hit__", "first_touch_hit", "tp_hit"),
        ),
        ("bad_mae_rate", ("first_touch_bad_mae_1r", "full_path_bad_mae_1r", "bad_mae_1r")),
        (
            "timeout_rate",
            ("first_touch_timeout", "__first_touch_timeout__", "timeout", "timed_out"),
        ),
        (
            "stop_rate",
            ("first_touch_stop", "__first_touch_stop__", "full_stop_loss", "stop_loss", "stop_hit"),
        ),
    ):
        result[destination] = _binary_rate(
            retained, _first_available_column(retained, candidates)
        )

    mapped = pd.to_numeric(retained[mapped_ev_col], errors="coerce").to_numpy(
        dtype=np.float64, copy=False
    )
    finite = np.isfinite(mapped)
    if finite.any() and np.isfinite(current_mapped_ev):
        mapped_valid = mapped[finite]
        percentile = float(np.mean(mapped_valid <= float(current_mapped_ev)))
        decile = int(np.clip(np.ceil(percentile * 10.0), 1.0, 10.0))
        lower = float(np.quantile(mapped_valid, (decile - 1) / 10.0))
        upper = float(np.quantile(mapped_valid, decile / 10.0))
        if decile == 10:
            decile_mask = finite & (mapped >= lower) & (mapped <= upper)
        else:
            decile_mask = finite & (mapped >= lower) & (mapped < upper)
        residual = (
            pd.to_numeric(retained[return_col], errors="coerce").to_numpy(
                dtype=np.float64, copy=False
            )
            - mapped
        )
        decile_residual = residual[decile_mask & np.isfinite(residual)]
        result["mapped_ev_decile"] = decile
        result["mapped_ev_decile_support"] = int(decile_residual.size)
        if decile_residual.size:
            result["mapped_ev_decile_calibration_residual"] = float(
                np.mean(decile_residual)
            )
    return result


def _canonical_gmm_cluster(value: Any) -> str:
    numeric = _safe_float(value, np.nan)
    if np.isfinite(numeric):
        return str(int(round(numeric)))
    return str(value or "").strip()


def _email_baseline_scope(
    reference: pd.DataFrame,
    *,
    side: str,
    archetype: str,
    min_rows: int,
) -> tuple[pd.DataFrame, str]:
    """Choose the standard side/archetype fallback hierarchy for email metrics."""

    local = reference.loc[
        reference["side_name"].astype(str).eq(side)
        & reference["policy_archetype"].astype(str).eq(archetype)
    ]
    if len(local) >= min_rows:
        return local, "side_x_archetype"
    local = reference.loc[reference["side_name"].astype(str).eq(side)]
    if len(local) >= min_rows:
        return local, "side_fallback"
    return reference, "global_fallback"


def _email_archetype_baselines_for_batch(
    batch_rows: pd.DataFrame,
    *,
    all_prior: pd.DataFrame,
    reference_asof: pd.Timestamp,
    policy: Mapping[str, Any],
) -> dict[int, dict[str, Any]]:
    """Build fixed 28-day robust baselines without changing selection logic."""

    if not bool(policy.get("email_archetype_baseline_enabled", True)):
        return {}
    if batch_rows.empty or all_prior.empty:
        return {}
    return_col = str(policy.get("return_col") or "ev_after_1pct")
    mapped_ev_col = str(
        policy.get("reference_mapped_expected_ev_col") or "mapped_expected_ev"
    )
    if return_col not in all_prior.columns or mapped_ev_col not in all_prior.columns:
        return {}
    window_days = max(
        1,
        int(
            _safe_float(
                policy.get("email_archetype_baseline_window_days"),
                _EMAIL_ARCHETYPE_BASELINE_WINDOW_DAYS,
            )
        ),
    )
    min_rows = max(
        1,
        int(
            _safe_float(
                policy.get("email_archetype_baseline_min_rows"),
                _safe_float(policy.get("min_reference_rows"), 40.0),
            )
        ),
    )
    trim_fraction = float(
        np.clip(policy.get("robust_daily_residual_trim_fraction") or 0.0, 0.0, 0.49)
    )
    start = reference_asof - pd.Timedelta(days=window_days)
    time_col = "outcome_resolved_at" if "outcome_resolved_at" in all_prior.columns else "timestamp"
    dates = pd.to_datetime(all_prior[time_col], utc=True, errors="coerce")
    reference = all_prior.loc[dates.ge(start) & dates.lt(reference_asof)].copy()
    if reference.empty:
        return {}
    reference["__email_outcome_day__"] = dates.loc[reference.index].dt.floor("D")
    historical = all_prior.loc[dates.lt(start)].copy()
    if not historical.empty:
        historical["__email_outcome_day__"] = dates.loc[historical.index].dt.floor(
            "D"
        )

    cache: dict[tuple[str, str, str, float], dict[str, Any]] = {}
    results: dict[int, dict[str, Any]] = {}
    for _, row in batch_rows.iterrows():
        side = str(row.get("side_name") or "")
        archetype = str(row.get("policy_archetype") or "missing")
        cluster = _canonical_gmm_cluster(row.get("gmm_cluster_id"))
        mapped_ev = _safe_float(row.get("mapped_expected_ev"), np.nan)
        mapped_ev_key = round(mapped_ev, 12) if np.isfinite(mapped_ev) else float("nan")
        cache_key = (side, archetype, cluster, mapped_ev_key)
        baseline = cache.get(cache_key)
        if baseline is None:
            local, scope = _email_baseline_scope(
                reference,
                side=side,
                archetype=archetype,
                min_rows=min_rows,
            )
            stats = _email_baseline_statistics(
                local,
                trim_fraction=trim_fraction,
                return_col=return_col,
                mapped_ev_col=mapped_ev_col,
                current_mapped_ev=mapped_ev,
            )
            baseline = {
                "threshold_basis_archetype_baseline_window_days": int(window_days),
                "threshold_basis_archetype_baseline_scope": scope,
                "threshold_basis_archetype_baseline_trim_fraction": trim_fraction,
                "threshold_basis_archetype_baseline_robust_method": "daily_residual_median_iqr_symmetric_trim",
                **{
                    f"threshold_basis_archetype_baseline_{key}": value
                    for key, value in stats.items()
                },
            }
            if not historical.empty:
                historical_local, historical_scope = _email_baseline_scope(
                    historical,
                    side=side,
                    archetype=archetype,
                    min_rows=min_rows,
                )
                historical_stats = _email_baseline_statistics(
                    historical_local,
                    trim_fraction=trim_fraction,
                    return_col=return_col,
                    mapped_ev_col=mapped_ev_col,
                    current_mapped_ev=mapped_ev,
                )
                recent_positive_ev_rate = _safe_float(
                    stats.get("positive_ev_rate"), np.nan
                )
                historical_positive_ev_rate = _safe_float(
                    historical_stats.get("positive_ev_rate"), np.nan
                )
                baseline.update(
                    {
                        "threshold_basis_archetype_baseline_historical_scope": historical_scope,
                        "threshold_basis_archetype_baseline_historical_support": int(
                            historical_stats.get("support", 0)
                        ),
                        "threshold_basis_archetype_baseline_historical_positive_ev_rate": historical_positive_ev_rate,
                        "threshold_basis_archetype_baseline_recent_vs_historical_positive_ev_rate": (
                            recent_positive_ev_rate - historical_positive_ev_rate
                            if np.isfinite(recent_positive_ev_rate)
                            and np.isfinite(historical_positive_ev_rate)
                            else float("nan")
                        ),
                    }
                )
            gmm_col = _first_available_column(
                local, ("gmm_cluster_id", "aegmm_cluster", "side_aegmm_cluster")
            )
            if gmm_col and cluster:
                same_state = local.loc[
                    local[gmm_col].map(_canonical_gmm_cluster).eq(cluster)
                ]
                state_stats = _email_baseline_statistics(
                    same_state,
                    trim_fraction=trim_fraction,
                    return_col=return_col,
                    mapped_ev_col=mapped_ev_col,
                    current_mapped_ev=mapped_ev,
                )
                if int(state_stats.get("support", 0)) >= max(12, min_rows // 2):
                    baseline.update(
                        {
                            "threshold_basis_archetype_baseline_gmm_cluster_id": cluster,
                            "threshold_basis_archetype_baseline_gmm_state_ev_mean": state_stats.get("ev_mean"),
                            "threshold_basis_archetype_baseline_gmm_state_support": state_stats.get("support"),
                        }
                    )
            cache[cache_key] = baseline
        results[int(row["_decision_idx"])] = dict(baseline)
    return results


def load_threshold_basis_policy(path: str | Path | None) -> dict[str, Any]:
    """Load a threshold-basis policy artifact."""
    if path is None:
        return {}
    policy_path = Path(path)
    if not policy_path.exists():
        return {}
    payload = json.loads(policy_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return {}
    payload["_artifact_path"] = str(policy_path)
    return payload


def _safe_float(value: Any, default: float = np.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if np.isfinite(out) else float(default)


def _first_non_empty(mapping: Mapping[str, Any], names: Sequence[str]) -> str:
    for name in names:
        value = mapping.get(name)
        if value is None:
            continue
        text = str(value).strip()
        if text and text.lower() not in {"nan", "none", "null", "<na>"}:
            return text
    return ""


def _canonical_policy_archetype(side: Any, value: Any) -> str:
    """Match live ``side__label`` keys to frozen side + label references.

    Live keeps the explicit side prefix for auditability, while threshold
    reference artifacts store side in ``side_name`` and the unprefixed label in
    ``policy_archetype``.  Strip exactly one matching ``side__`` prefix for
    local calibration lookups; labels such as ``long_breakout_*`` remain
    unchanged.
    """
    side_name = str(side or "").strip().lower()
    text = str(value or "").strip()
    prefix = f"{side_name}__" if side_name else ""
    return text[len(prefix) :] if prefix and text.startswith(prefix) else text


def _policy_path(payload: Mapping[str, Any], key: str) -> Path | None:
    value = str(payload.get(key) or "").strip()
    if not value:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    artifact_path = str(payload.get("_artifact_path") or "").strip()
    if artifact_path:
        candidate = Path(artifact_path).parent / path
        if candidate.exists():
            return candidate
    return path


def _load_reference(payload: Mapping[str, Any]) -> pd.DataFrame:
    path = _policy_path(payload, "reference_candidates_path")
    if path is None or not path.exists():
        return pd.DataFrame()
    key = str(path.resolve())
    cached = _REFERENCE_CACHE.get(key)
    if cached is not None:
        return cached
    columns = payload.get("reference_columns")
    kwargs: dict[str, Any] = {}
    if isinstance(columns, Iterable) and not isinstance(columns, (str, bytes)):
        kwargs["columns"] = list(columns)
    ref = pd.read_parquet(path, **kwargs)
    ref = _normalise_rows(ref)
    if "outcome_resolved_at" in ref.columns:
        ref["outcome_resolved_at"] = pd.to_datetime(
            ref["outcome_resolved_at"], utc=True, errors="coerce"
        )
    ref = ref.sort_values("timestamp", kind="stable").reset_index(drop=True)
    _REFERENCE_CACHE[key] = ref
    return ref


def _normalise_rows(rows: pd.DataFrame) -> pd.DataFrame:
    out = rows.copy()
    if "timestamp" in out.columns:
        out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    canonical_selected = pd.Series(False, index=out.index)
    for canonical_col in ("archetype_policy_key", "__archetype_policy_key__"):
        if canonical_col not in out.columns:
            continue
        canonical = out[canonical_col].fillna("").astype(str).str.strip()
        valid = (~canonical_selected) & ~canonical.str.lower().isin(
            {"", "nan", "none", "null", "<na>"}
        )
        if "policy_archetype" not in out.columns:
            out["policy_archetype"] = "missing"
        out.loc[valid, "policy_archetype"] = canonical.loc[valid]
        canonical_selected.loc[valid] = True
    if "policy_archetype" not in out.columns:
        for col in (
            "archetype_policy_key",
            "local_side_archetype",
            "archetype_label_family",
            "source_archetype",
        ):
            if col in out.columns:
                out["policy_archetype"] = out[col].astype(str)
                break
    if "policy_archetype" not in out.columns:
        out["policy_archetype"] = "missing"
    out["policy_archetype"] = out["policy_archetype"].fillna("missing").astype(str)
    if "side_name" in out.columns:
        out["side_name"] = out["side_name"].fillna("").astype(str)
        out["policy_archetype"] = [
            _canonical_policy_archetype(side, archetype)
            for side, archetype in zip(
                out["side_name"].to_numpy(copy=False),
                out["policy_archetype"].to_numpy(copy=False),
            )
        ]
    return out


def _multiplier_for_thresholds(
    global_threshold: float,
    local_threshold: float,
    *,
    support: int,
    support_target: float,
    multiplier_min: float,
    multiplier_max: float,
) -> float:
    """Return a support-shrunk local/global threshold multiplier.

    The calibration score can differ from the final rank score.  This is the
    intended contract for the pre-MLP mapping: use the parent rank to estimate
    state quality, then nudge the final MLP rank rather than replacing it.
    """
    if not np.isfinite(global_threshold) or not np.isfinite(local_threshold):
        return 1.0
    raw = float(
        np.clip(
            global_threshold / max(float(local_threshold), 1e-8),
            float(multiplier_min),
            float(multiplier_max),
        )
    )
    confidence = float(np.clip(float(support) / max(float(support_target), 1.0), 0.0, 1.0))
    return float(1.0 + confidence * (raw - 1.0))


def _rank_against_ref(score: pd.Series, ref_score: pd.Series) -> pd.Series:
    ref = (
        pd.to_numeric(ref_score, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    ref_arr = np.sort(ref.to_numpy(dtype=np.float64, copy=False))
    out = np.full(len(score), np.nan, dtype=np.float64)
    if ref_arr.size == 0:
        return pd.Series(out, index=score.index)
    values = pd.to_numeric(score, errors="coerce").to_numpy(dtype=np.float64, copy=False)
    finite = np.isfinite(values)
    out[finite] = (
        np.searchsorted(ref_arr, values[finite], side="right") / float(ref_arr.size)
    )
    return pd.Series(out, index=score.index).clip(0.0, 1.0)


def _rank_ev_with_parent_tiebreak(
    current_ev: np.ndarray,
    current_parent: np.ndarray,
    reference_ev: np.ndarray,
    reference_parent: np.ndarray,
) -> np.ndarray:
    """Rank EV primarily and use parent rank only within exact EV ties."""
    out = np.full(len(current_ev), np.nan, dtype=np.float64)
    ref_finite = np.isfinite(reference_ev)
    ref_ev = np.asarray(reference_ev[ref_finite], dtype=np.float64)
    ref_parent = np.asarray(reference_parent[ref_finite], dtype=np.float64)
    if ref_ev.size == 0:
        return out
    ref_parent = np.where(np.isfinite(ref_parent), ref_parent, 0.5)
    order = np.argsort(ref_ev, kind="stable")
    ref_ev = ref_ev[order]
    ref_parent = ref_parent[order]
    finite_current = np.isfinite(current_ev)
    for idx in np.flatnonzero(finite_current):
        value = float(current_ev[idx])
        left = int(np.searchsorted(ref_ev, value, side="left"))
        right = int(np.searchsorted(ref_ev, value, side="right"))
        if right <= left or not np.isfinite(current_parent[idx]):
            out[idx] = right / float(ref_ev.size)
            continue
        tied_parent = np.sort(ref_parent[left:right], kind="stable")
        parent_count = int(
            np.searchsorted(tied_parent, float(current_parent[idx]), side="right")
        )
        out[idx] = (left + parent_count) / float(ref_ev.size)
    return np.clip(out, 0.0, 1.0)


def _threshold_for_target_ev(
    ref: pd.DataFrame,
    *,
    score_col: str,
    return_col: str,
    target_ev: float,
    min_rows: int,
) -> float:
    if ref.empty or score_col not in ref.columns or return_col not in ref.columns:
        return float("nan")
    score = pd.to_numeric(ref[score_col], errors="coerce").to_numpy(
        dtype=np.float64, copy=False
    )
    ev = pd.to_numeric(ref[return_col], errors="coerce").to_numpy(
        dtype=np.float64, copy=False
    )
    finite = np.isfinite(score) & np.isfinite(ev)
    score = score[finite]
    ev = ev[finite]
    if score.size < int(min_rows) or not np.isfinite(float(target_ev)):
        return float("nan")
    grid = np.unique(np.quantile(score, np.linspace(0.70, 0.99, 60)))
    order = np.argsort(score, kind="stable")
    score_sorted = score[order]
    ev_sorted = ev[order]
    reverse_sum = np.cumsum(ev_sorted[::-1], dtype=np.float64)[::-1]
    starts = np.searchsorted(score_sorted, grid, side="left")
    counts = score_sorted.size - starts
    eligible = counts >= int(min_rows)
    means = np.full(grid.size, np.nan, dtype=np.float64)
    means[eligible] = reverse_sum[starts[eligible]] / counts[eligible]
    passes = eligible & (means >= float(target_ev))
    if np.any(passes):
        gaps = np.where(passes, np.abs(means - float(target_ev)), np.inf)
        return float(grid[int(np.argmin(gaps))])
    return float(np.quantile(score, 0.99))


def _historical_top10_ev(
    ref: pd.DataFrame,
    *,
    score_col: str,
    return_col: str,
    min_rows: int,
) -> float:
    if ref.empty or score_col not in ref.columns or return_col not in ref.columns:
        return float("nan")
    score = pd.to_numeric(ref[score_col], errors="coerce").to_numpy(
        dtype=np.float64, copy=False
    )
    ev = pd.to_numeric(ref[return_col], errors="coerce").to_numpy(
        dtype=np.float64, copy=False
    )
    finite = np.isfinite(score) & np.isfinite(ev)
    score = score[finite]
    ev = ev[finite]
    if score.size < int(min_rows):
        return float("nan")
    threshold = float(np.quantile(score, 0.90))
    chosen = ev[score >= threshold]
    if chosen.size < int(min_rows):
        return float("nan")
    return float(np.mean(chosen))


def _historical_current_top10_ev(
    ref: pd.DataFrame,
    *,
    return_col: str,
    rank_col: str,
    min_rows: int,
) -> float:
    if ref.empty or return_col not in ref.columns or rank_col not in ref.columns:
        return float("nan")
    rank = pd.to_numeric(ref[rank_col], errors="coerce").to_numpy(
        dtype=np.float64, copy=False
    )
    ev = pd.to_numeric(ref[return_col], errors="coerce").to_numpy(
        dtype=np.float64, copy=False
    )
    finite = np.isfinite(rank) & np.isfinite(ev)
    rank = rank[finite]
    ev = ev[finite]
    if rank.size < int(min_rows):
        return float("nan")
    chosen = ev[rank >= 0.90]
    if chosen.size < max(1, int(min_rows) // 2):
        return float("nan")
    return float(np.mean(chosen))


def _top_n_by_score(rows: pd.DataFrame, score: pd.Series, n: int) -> pd.DataFrame:
    if rows.empty or int(n) <= 0:
        out = rows.iloc[0:0].copy()
        out["selection_score"] = pd.Series(dtype="float64")
        return out
    work = rows.copy()
    work["_activity_score"] = pd.to_numeric(score.reindex(work.index), errors="coerce")
    work = work.loc[work["_activity_score"].notna()].copy()
    if work.empty:
        out = rows.iloc[0:0].copy()
        out["selection_score"] = pd.Series(dtype="float64")
        return out
    sort_cols = [col for col in ("_activity_score", "timestamp", "symbol") if col in work.columns]
    asc = [False] + [True] * (len(sort_cols) - 1)
    work = work.sort_values(sort_cols, ascending=asc).head(int(n))
    work["selection_score"] = pd.to_numeric(work["_activity_score"], errors="coerce")
    return work.drop(columns=["_activity_score"])


def _rescale_selection_score_to_top_band(
    rows: pd.DataFrame,
    *,
    floor: float,
) -> pd.DataFrame:
    if rows.empty:
        return rows
    out = rows.copy()
    score = pd.to_numeric(out.get("selection_score"), errors="coerce")
    if score.notna().sum() == 0:
        out["selection_score"] = float(floor)
        return out
    rank = score.rank(method="first", pct=True)
    out["selection_score"] = float(floor) + (1.0 - float(floor)) * rank
    return out


def _match_baseline_activity(
    batch_rows: pd.DataFrame,
    chosen: pd.DataFrame,
    *,
    score: pd.Series,
    target_count: int,
    top_band_floor: float,
) -> pd.DataFrame:
    target = int(target_count)
    if target <= 0:
        out = batch_rows.iloc[0:0].copy()
        out["selection_score"] = pd.Series(dtype="float64")
        return out
    if len(chosen) >= target:
        return _rescale_selection_score_to_top_band(
            _top_n_by_score(chosen, score, target),
            floor=top_band_floor,
        )
    return _rescale_selection_score_to_top_band(
        _top_n_by_score(batch_rows, score, target),
        floor=top_band_floor,
    )


def _score_value(decision: Mapping[str, Any], score_col: str) -> float:
    chain = decision.get("chain_results")
    if not isinstance(chain, Mapping):
        chain = {}
    aliases = [score_col]
    aliases.extend(
        [
            "calibrated_score_regime_ev",
            "score_regime_calibrated",
            "calibrated_score",
            "meta_score_oof",
            "meta_pred",
        ]
    )
    for name in aliases:
        if not name:
            continue
        value = decision.get(name, chain.get(name))
        out = _safe_float(value)
        if np.isfinite(out):
            return out
    return float("nan")


def _is_invalid_expected_ev(value: Any) -> bool:
    numeric = _safe_float(value, np.nan)
    return bool(
        np.isfinite(numeric)
        and any(
            np.isclose(
                numeric,
                sentinel,
                rtol=0.0,
                atol=_EXPECTED_EV_SENTINEL_ATOL,
            )
            for sentinel in _INVALID_EXPECTED_EV_SENTINELS
        )
    )


def _expected_ev_lookup(
    decision: Mapping[str, Any], expected_ev_col: str
) -> tuple[float, bool]:
    chain = decision.get("chain_results")
    if not isinstance(chain, Mapping):
        chain = {}
    saw_invalid_sentinel = False
    for name in (
        expected_ev_col,
        "expected_net_ev_after_1pct_side_archetype",
        "expected_net_ev_after_1pct",
        "market_state_mlp_expected_net_ev_after_1pct",
    ):
        value = _safe_float(decision.get(name, chain.get(name)))
        if not np.isfinite(value):
            continue
        if _is_invalid_expected_ev(value):
            saw_invalid_sentinel = True
            continue
        return value, False
    return float("nan"), saw_invalid_sentinel


def _expected_ev_value(decision: Mapping[str, Any], expected_ev_col: str) -> float:
    return _expected_ev_lookup(decision, expected_ev_col)[0]


def _decision_timestamp(decision: Mapping[str, Any]) -> pd.Timestamp:
    for name in ("signal_bar_ts", "timestamp", "decision_ts"):
        value = decision.get(name)
        if value is None:
            continue
        ts = pd.Timestamp(value)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        return ts
    return pd.Timestamp.now(tz="UTC")


def _baseline_rank(
    decision: Mapping[str, Any],
    *,
    store: Any,
) -> float:
    chain = decision.get("chain_results")
    if not isinstance(chain, Mapping):
        chain = {}
    for name in ("policy_rank_pct", "rank_pct"):
        value = _safe_float(decision.get(name, chain.get(name)))
        if np.isfinite(value):
            return float(np.clip(value, 0.0, 1.0))
    if store is None:
        return float("nan")
    score = _safe_float(decision.get("calibrated_score", chain.get("calibrated_score")))
    if not np.isfinite(score):
        return float("nan")
    try:
        result = store.lookup(
            strategy_id=str(decision.get("strategy_id") or ""),
            side=str(decision.get("side") or decision.get("side_name") or ""),
            calibrated_score=float(score),
        )
        value = _safe_float(getattr(result, "policy_rank_pct", np.nan))
        return float(np.clip(value, 0.0, 1.0)) if np.isfinite(value) else float("nan")
    except Exception:
        return float("nan")


def _decision_rows(
    decisions: Sequence[MutableMapping[str, Any]],
    *,
    policy: Mapping[str, Any],
    store: Any,
) -> pd.DataFrame:
    live_score_col = str(
        policy.get("live_score_col")
        or policy.get("score_col")
        or "calibrated_score"
    )
    expected_ev_col = str(
        policy.get("mapped_expected_ev_col")
        or "expected_net_ev_after_1pct_side_archetype"
    )
    parent_rank_col = str(
        policy.get("rank_blend_parent_col")
        or "v9_tail95_predecessor_rank"
    )
    rows: list[dict[str, Any]] = []
    for idx, decision in enumerate(decisions):
        chain = decision.get("chain_results")
        if not isinstance(chain, Mapping):
            chain = {}
        side = _first_non_empty(decision, ("side_name", "side")) or _first_non_empty(chain, ("side_name", "side"))
        policy_archetype = "missing"
        for archetype_field in (
            "archetype_policy_key",
            "__archetype_policy_key__",
            "policy_archetype",
            "local_side_archetype",
            "archetype_label_family",
        ):
            policy_archetype = _first_non_empty(
                decision, (archetype_field,)
            ) or _first_non_empty(chain, (archetype_field,))
            if policy_archetype:
                break
        mapped_expected_ev, invalid_ev_sentinel = _expected_ev_lookup(
            decision, expected_ev_col
        )
        rows.append(
            {
                "_decision_idx": idx,
                "timestamp": _decision_timestamp(decision),
                "symbol": str(decision.get("symbol") or chain.get("symbol") or ""),
                "strategy_id": str(decision.get("strategy_id") or chain.get("strategy_id") or ""),
                "side_name": str(side or ""),
                "policy_archetype": str(policy_archetype or "missing"),
                "score": _score_value(decision, live_score_col),
                "parent_rank": _score_value(decision, parent_rank_col),
                "mapped_expected_ev": mapped_expected_ev,
                "mapped_expected_ev_invalid_sentinel": invalid_ev_sentinel,
                "rank_pct": _baseline_rank(decision, store=store),
                "gmm_cluster_id": _safe_float(
                    decision.get("gmm_cluster_id", chain.get("gmm_cluster_id")),
                    np.nan,
                ),
                "gmm_posterior_max": _safe_float(
                    decision.get(
                        "gmm_posterior_max", chain.get("gmm_posterior_max")
                    ),
                    np.nan,
                ),
            }
        )
    return _normalise_rows(pd.DataFrame(rows))


def _select_batch(
    batch_rows: pd.DataFrame,
    *,
    recent_ref: pd.DataFrame,
    all_prior: pd.DataFrame,
    policy: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    reference_score_col = str(
        policy.get("reference_score_col")
        or policy.get("score_col")
        or "calibrated_score_regime_ev"
    )
    return_col = str(policy.get("return_col") or "ret_net_notional")
    rank_col = str(policy.get("rank_col") or "rank_pct")
    min_rows = int(policy.get("min_reference_rows") or 40)
    arch_min_rows = int(policy.get("arch_min_reference_rows") or max(8, min_rows // 4))
    base_rank_threshold = float(policy.get("base_rank_threshold") or 0.90)
    top_band_floor = float(policy.get("top_band_floor") or base_rank_threshold)

    baseline_count = int(
        pd.to_numeric(batch_rows.get("rank_pct"), errors="coerce")
        .ge(base_rank_threshold)
        .sum()
    )
    global_target = _historical_current_top10_ev(
        all_prior,
        return_col=return_col,
        rank_col=rank_col,
        min_rows=min_rows,
    )
    if not np.isfinite(global_target):
        global_target = _historical_top10_ev(
            all_prior,
            score_col=reference_score_col,
            return_col=return_col,
            min_rows=min_rows,
        )
    global_threshold = _threshold_for_target_ev(
        recent_ref,
        score_col=reference_score_col,
        return_col=return_col,
        target_ev=global_target,
        min_rows=min_rows,
    )
    chosen_parts: list[pd.DataFrame] = []
    score_rank = pd.Series(np.nan, index=batch_rows.index, dtype="float64")
    dynamic_targets = pd.Series(np.nan, index=batch_rows.index, dtype="float64")
    dynamic_thresholds = pd.Series(np.nan, index=batch_rows.index, dtype="float64")
    for archetype, sub in batch_rows.groupby("policy_archetype", dropna=False):
        archetype_key = str(archetype)
        ref_arch = recent_ref.loc[recent_ref["policy_archetype"].eq(archetype_key)]
        prior_arch = all_prior.loc[all_prior["policy_archetype"].eq(archetype_key)]
        target = _historical_current_top10_ev(
            prior_arch,
            return_col=return_col,
            rank_col=rank_col,
            min_rows=arch_min_rows,
        )
        threshold = _threshold_for_target_ev(
            ref_arch,
            score_col=reference_score_col,
            return_col=return_col,
            target_ev=target,
            min_rows=arch_min_rows,
        )
        if not np.isfinite(threshold):
            threshold = global_threshold
            target = global_target
        dynamic_targets.loc[sub.index] = (
            float(target) if np.isfinite(float(target)) else np.nan
        )
        dynamic_thresholds.loc[sub.index] = (
            float(threshold) if np.isfinite(float(threshold)) else np.nan
        )
        ref_score = (
            ref_arch[reference_score_col]
            if len(ref_arch) >= arch_min_rows and reference_score_col in ref_arch.columns
            else recent_ref.get(reference_score_col, pd.Series(dtype="float64"))
        )
        local_rank = _rank_against_ref(sub["score"], ref_score)
        score_rank.loc[sub.index] = local_rank
        part = sub.loc[pd.to_numeric(sub["score"], errors="coerce").ge(threshold)].copy()
        part["selection_score"] = local_rank.reindex(part.index)
        chosen_parts.append(part)
    chosen = (
        pd.concat(chosen_parts, ignore_index=False)
        if chosen_parts
        else batch_rows.iloc[0:0].copy()
    )
    global_rank = _rank_against_ref(
        batch_rows["score"],
        recent_ref.get(reference_score_col, pd.Series(dtype="float64")),
    )
    selected = _match_baseline_activity(
        batch_rows,
        chosen,
        score=score_rank.fillna(global_rank),
        target_count=baseline_count,
        top_band_floor=top_band_floor,
    )
    if not selected.empty:
        selected["dynamic_ev_target"] = dynamic_targets.reindex(selected.index)
        selected["dynamic_score_threshold"] = dynamic_thresholds.reindex(selected.index)
    meta = {
        "baseline_activity_count": int(baseline_count),
        "recent_reference_rows": int(len(recent_ref)),
        "reference_rows": int(len(all_prior)),
        "global_dynamic_ev_target": float(global_target)
        if np.isfinite(global_target)
        else np.nan,
        "global_dynamic_score_threshold": float(global_threshold)
        if np.isfinite(global_threshold)
        else np.nan,
    }
    return selected, meta


def _select_pre_mlp_multiplier_batch(
    batch_rows: pd.DataFrame,
    *,
    recent_ref: pd.DataFrame,
    all_prior: pd.DataFrame,
    policy: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply the causal side x archetype EV-target mapping selected by HPO.

    Thresholds are learned from the parent rank in the historical reference
    stream.  The resulting, support-shrunk multiplier is then applied to the
    final MLP rank.  This deliberately preserves the MLP ordering within a
    local state instead of replacing it with a threshold-only selector.
    """
    calibration_col = str(policy.get("calibration_reference_score_col") or "policy_parent_rank")
    apply_reference_col = str(policy.get("apply_reference_score_col") or "rank_mlp_direct")
    return_col = str(policy.get("return_col") or "ev_after_1pct")
    min_rows = int(policy.get("min_reference_rows") or 40)
    support_target = float(policy.get("local_support_target") or 160.0)
    multiplier_min = float(policy.get("multiplier_min") or 0.50)
    multiplier_max = float(policy.get("multiplier_max") or 1.50)
    top_fraction = float(policy.get("top_fraction") or 0.10)

    global_target = _historical_top10_ev(
        all_prior,
        score_col=calibration_col,
        return_col=return_col,
        min_rows=min_rows,
    )
    global_threshold = _threshold_for_target_ev(
        recent_ref,
        score_col=calibration_col,
        return_col=return_col,
        target_ev=global_target,
        min_rows=min_rows,
    )
    if not np.isfinite(global_threshold):
        global_threshold = _threshold_for_target_ev(
            all_prior,
            score_col=calibration_col,
            return_col=return_col,
            target_ev=global_target,
            min_rows=min_rows,
        )

    apply_ref = (
        pd.to_numeric(all_prior[apply_reference_col], errors="coerce").dropna()
        if apply_reference_col in all_prior.columns
        else pd.Series(dtype="float64")
    )
    apply_cutoff = (
        float(np.nanquantile(apply_ref.to_numpy(dtype=float), 1.0 - top_fraction))
        if len(apply_ref) >= min_rows
        else float(policy.get("base_rank_threshold") or 0.90)
    )
    local_multiplier = pd.Series(1.0, index=batch_rows.index, dtype="float64")
    local_threshold = pd.Series(np.nan, index=batch_rows.index, dtype="float64")
    local_support = pd.Series(0, index=batch_rows.index, dtype="int32")
    fallback = pd.Series(True, index=batch_rows.index, dtype="bool")

    group_cols = ["side_name", "policy_archetype"]
    for key, sub in batch_rows.groupby(group_cols, dropna=False, sort=False):
        side, archetype = (str(key[0]), str(key[1]))
        ref_local = recent_ref.loc[
            recent_ref["side_name"].eq(side)
            & recent_ref["policy_archetype"].eq(archetype)
        ]
        support = int(
            pd.to_numeric(ref_local.get(calibration_col), errors="coerce").notna().sum()
        )
        threshold = _threshold_for_target_ev(
            ref_local,
            score_col=calibration_col,
            return_col=return_col,
            target_ev=global_target,
            min_rows=min_rows,
        )
        local_support.loc[sub.index] = support
        if np.isfinite(threshold):
            local_threshold.loc[sub.index] = float(threshold)
            local_multiplier.loc[sub.index] = _multiplier_for_thresholds(
                global_threshold,
                float(threshold),
                support=support,
                support_target=support_target,
                multiplier_min=multiplier_min,
                multiplier_max=multiplier_max,
            )
            fallback.loc[sub.index] = False
        else:
            local_threshold.loc[sub.index] = global_threshold

    score = pd.to_numeric(batch_rows["score"], errors="coerce")
    mapped = (score * local_multiplier).clip(0.0, 1.0)
    selected = batch_rows.loc[mapped.ge(apply_cutoff)].copy()
    selected["selection_score"] = mapped.reindex(selected.index)
    selected["dynamic_ev_target"] = float(global_target) if np.isfinite(global_target) else np.nan
    selected["dynamic_score_threshold"] = local_threshold.reindex(selected.index)
    selected["ev_target_multiplier"] = local_multiplier.reindex(selected.index)
    selected["ev_target_local_support"] = local_support.reindex(selected.index)
    selected["ev_target_global_fallback"] = fallback.reindex(selected.index)
    return selected, {
        "baseline_activity_count": int(mapped.ge(apply_cutoff).sum()),
        "recent_reference_rows": int(len(recent_ref)),
        "reference_rows": int(len(all_prior)),
        "global_dynamic_ev_target": float(global_target) if np.isfinite(global_target) else np.nan,
        "global_dynamic_score_threshold": float(global_threshold) if np.isfinite(global_threshold) else np.nan,
        "apply_cutoff": float(apply_cutoff),
        "local_support": local_support,
        "fallback": fallback,
        "multiplier": local_multiplier,
    }


def _select_side_archetype_expected_ev_batch(
    batch_rows: pd.DataFrame,
    *,
    recent_ref: pd.DataFrame,
    all_prior: pd.DataFrame,
    policy: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Admit on a common EV unit with causal side x archetype corrections.

    The frozen postprocessor first maps every row to a side x archetype expected
    EV.  The admission layer then estimates only the recent realized-minus-
    mapped residual.  Corrections shrink local -> side -> global and therefore
    cannot replace the archetype-specific base EV curve.
    """
    reference_ev_col = str(
        policy.get("reference_mapped_expected_ev_col")
        or "mapped_expected_ev"
    )
    return_col = str(policy.get("return_col") or "ev_after_1pct")
    min_rows = int(policy.get("min_reference_rows") or 40)
    side_support_target = float(policy.get("side_support_target") or 320.0)
    local_support_target = float(policy.get("local_support_target") or 160.0)
    correction_cap = max(float(policy.get("recent_ev_correction_cap") or 0.03), 0.0)
    top_fraction = float(np.clip(policy.get("top_fraction") or 0.10, 1e-4, 0.50))
    ev_rank_blend_weight = float(
        np.clip(policy.get("ev_rank_blend_weight", 1.0), 0.0, 1.0)
    )
    selection_mode = str(policy.get("selection_mode") or "top_fraction_rank").strip()
    fixed_target_net_ev = _safe_float(policy.get("fixed_target_net_ev"), np.nan)
    trim_fraction = float(
        np.clip(policy.get("robust_daily_residual_trim_fraction") or 0.0, 0.0, 0.49)
    )

    reference = recent_ref.copy(deep=False)
    if reference_ev_col not in reference.columns:
        fallback_col = str(
            policy.get("reference_mapped_expected_ev_fallback_col")
            or "expected_net_ev_after_1pct_mlp_direct"
        )
        if fallback_col in reference.columns:
            reference = reference.rename(columns={fallback_col: reference_ev_col})
    if reference_ev_col not in reference.columns:
        return batch_rows.iloc[:0].copy(), {
            "baseline_activity_count": 0,
            "recent_reference_rows": int(len(recent_ref)),
            "reference_rows": int(len(all_prior)),
            "global_dynamic_ev_target": np.nan,
            "global_dynamic_score_threshold": np.nan,
            "apply_cutoff": np.nan,
            "recent_ev_correction": pd.Series(0.0, index=batch_rows.index),
            "local_support": pd.Series(0, index=batch_rows.index, dtype="int32"),
            "fallback": pd.Series(True, index=batch_rows.index, dtype="bool"),
            "mapped_expected_ev": pd.Series(np.nan, index=batch_rows.index),
            "corrected_expected_ev": pd.Series(np.nan, index=batch_rows.index),
            "mapping_scope": pd.Series("missing", index=batch_rows.index),
        }

    mapped_ref = pd.to_numeric(reference[reference_ev_col], errors="coerce")
    invalid_ref_ev = np.zeros(len(mapped_ref), dtype=bool)
    for sentinel in _INVALID_EXPECTED_EV_SENTINELS:
        invalid_ref_ev |= np.isclose(
            mapped_ref.to_numpy(dtype=np.float64, copy=False),
            sentinel,
            rtol=0.0,
            atol=_EXPECTED_EV_SENTINEL_ATOL,
            equal_nan=False,
        )
    mapped_ref = mapped_ref.mask(invalid_ref_ev)
    realized_ref = pd.to_numeric(reference.get(return_col), errors="coerce")
    residual = (realized_ref - mapped_ref).replace([np.inf, -np.inf], np.nan)
    finite = residual.notna() & mapped_ref.notna()

    def robust_correction(positions: Iterable[Any]) -> tuple[float, int, int, float]:
        """Compute the ablation's row-weighted correction after daily trimming."""
        position_index = pd.Index(positions)
        values = residual.loc[position_index]
        valid = values.notna()
        if not valid.any():
            return np.nan, 0, 0, np.nan
        values = values.loc[valid]
        if trim_fraction <= 0.0 or "outcome_resolved_at" not in reference.columns:
            return float(values.mean()), int(len(values)), 0, np.nan
        resolved = pd.to_datetime(
            reference.loc[values.index, "outcome_resolved_at"],
            utc=True,
            errors="coerce",
        )
        daily = (
            pd.DataFrame(
                {
                    "outcome_day": resolved.dt.floor("D"),
                    "residual": values.to_numpy(dtype=np.float64, copy=False),
                },
                index=values.index,
            )
            .dropna(subset=["outcome_day", "residual"])
            .groupby("outcome_day", sort=False, observed=True)["residual"]
            .agg(["sum", "count", "mean"])
        )
        if daily.empty:
            return np.nan, 0, 0, np.nan
        means = daily["mean"].to_numpy(dtype=np.float64, copy=False)
        sums = daily["sum"].to_numpy(dtype=np.float64, copy=False)
        counts = daily["count"].to_numpy(dtype=np.float64, copy=False)
        median = float(np.median(means))
        q25, q75 = np.quantile(means, [0.25, 0.75])
        iqr = max(float(q75 - q25), 1e-8)
        keep = np.ones(len(daily), dtype=bool)
        if len(daily) >= 4:
            robust_z = (means - median) / iqr
            low, high = np.quantile(
                robust_z, [trim_fraction, 1.0 - trim_fraction]
            )
            keep = (robust_z >= low) & (robust_z <= high)
        if not keep.any():
            return np.nan, 0, int(len(daily)), iqr
        support = int(np.sum(counts[keep]))
        correction = float(np.sum(sums[keep]) / max(np.sum(counts[keep]), 1.0))
        return correction, support, int(keep.sum()), iqr

    global_value, global_support, global_days_retained, global_iqr = robust_correction(
        reference.index[finite]
    )
    global_correction = global_value if np.isfinite(global_value) else 0.0
    global_correction = float(np.clip(global_correction, -correction_cap, correction_cap))

    side_corrections: dict[str, tuple[float, int]] = {}
    local_corrections: dict[tuple[str, str], tuple[float, int, bool]] = {}
    for side, positions in reference.loc[finite].groupby("side_name", sort=False).groups.items():
        local_value, support, _, _ = robust_correction(positions)
        alpha = float(np.clip(support / max(side_support_target, 1.0), 0.0, 1.0))
        local_mean = local_value if np.isfinite(local_value) else global_correction
        correction = (1.0 - alpha) * global_correction + alpha * local_mean
        side_corrections[str(side)] = (
            float(np.clip(correction, -correction_cap, correction_cap)),
            support,
        )
    for key, positions in reference.loc[finite].groupby(
        ["side_name", "policy_archetype"], sort=False
    ).groups.items():
        side_key, archetype_key = str(key[0]), str(key[1])
        parent, _ = side_corrections.get(
            side_key, (global_correction, global_support)
        )
        local_value, support, _, _ = robust_correction(positions)
        alpha = float(np.clip(support / max(local_support_target, 1.0), 0.0, 1.0))
        local_mean = local_value if np.isfinite(local_value) else parent
        correction = (1.0 - alpha) * parent + alpha * local_mean
        local_corrections[(side_key, archetype_key)] = (
            float(np.clip(correction, -correction_cap, correction_cap)),
            support,
            bool(support < min_rows),
        )

    def correction_arrays(rows: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        corrections = np.empty(len(rows), dtype=np.float64)
        supports = np.empty(len(rows), dtype=np.int32)
        scopes = np.empty(len(rows), dtype=object)
        sides = rows["side_name"].astype(str).to_numpy(copy=False)
        arches = rows["policy_archetype"].astype(str).to_numpy(copy=False)
        for idx, (side, archetype) in enumerate(zip(sides, arches)):
            local = local_corrections.get((side, archetype))
            if local is not None:
                corrections[idx], supports[idx] = local[0], local[1]
                scopes[idx] = "side_x_archetype"
                continue
            parent = side_corrections.get(side)
            if parent is not None:
                corrections[idx], supports[idx] = parent
                scopes[idx] = "side_fallback"
            else:
                corrections[idx], supports[idx] = global_correction, global_support
                scopes[idx] = "global_fallback"
        return corrections, supports, scopes

    ref_correction, _, _ = correction_arrays(reference)
    corrected_ref_full = (
        mapped_ref.to_numpy(dtype=np.float64, copy=False) + ref_correction
    )
    corrected_ref = corrected_ref_full[np.isfinite(corrected_ref_full)]
    apply_cutoff = (
        float(np.quantile(corrected_ref, 1.0 - top_fraction))
        if corrected_ref.size >= min_rows
        else float("nan")
    )

    mapped_current = pd.to_numeric(
        batch_rows.get("mapped_expected_ev"), errors="coerce"
    ).to_numpy(dtype=np.float64, copy=True)
    for sentinel in _INVALID_EXPECTED_EV_SENTINELS:
        mapped_current[np.isclose(
            mapped_current,
            sentinel,
            rtol=0.0,
            atol=_EXPECTED_EV_SENTINEL_ATOL,
            equal_nan=False,
        )] = np.nan
    current_correction, current_support, current_scope = correction_arrays(batch_rows)
    current_correction[~np.isfinite(mapped_current)] = 0.0
    corrected_current = mapped_current + current_correction
    corrected_ev_rank = np.full(len(batch_rows), np.nan, dtype=np.float64)
    ref_sorted = np.sort(corrected_ref)
    finite_current = np.isfinite(corrected_current)
    if ref_sorted.size:
        corrected_ev_rank[finite_current] = np.searchsorted(
            ref_sorted, corrected_current[finite_current], side="right"
        ) / float(ref_sorted.size)
    parent_source = batch_rows.get(
        "parent_rank", pd.Series(np.nan, index=batch_rows.index)
    )
    parent_rank = pd.to_numeric(parent_source, errors="coerce").to_numpy(
        dtype=np.float64, copy=False
    )
    tie_break_parent = bool(policy.get("corrected_ev_tie_break_parent", False))
    if tie_break_parent:
        reference_parent_col = str(
            policy.get("reference_parent_rank_col") or "rank_mlp_direct"
        )
        reference_parent = pd.to_numeric(
            reference.get(reference_parent_col), errors="coerce"
        ).to_numpy(dtype=np.float64, copy=False)
        corrected_ev_rank = _rank_ev_with_parent_tiebreak(
            corrected_current,
            parent_rank,
            corrected_ref_full,
            reference_parent,
        )
    parent_finite = np.isfinite(parent_rank)
    blended_rank = corrected_ev_rank.copy()
    both_finite = np.isfinite(corrected_ev_rank) & parent_finite
    blended_rank[both_finite] = (
        (1.0 - ev_rank_blend_weight) * parent_rank[both_finite]
        + ev_rank_blend_weight * corrected_ev_rank[both_finite]
    )
    rank_cutoff = 1.0 - top_fraction
    if selection_mode == "fixed_corrected_ev_threshold":
        if not np.isfinite(fixed_target_net_ev):
            raise ValueError(
                "fixed_corrected_ev_threshold requires finite fixed_target_net_ev"
            )
        selected_mask = (
            np.isfinite(corrected_current)
            & (corrected_current >= fixed_target_net_ev)
        )
        # Selected rows still enter the regular policy/portfolio chain through
        # its [0.90, 1.00] admission band. Preserve corrected-EV ordering
        # inside that band without imposing an additional top-k quota.
        eligible_ref = np.sort(
            corrected_ref[corrected_ref >= fixed_target_net_ev]
        )
        selection_rank = np.zeros(len(batch_rows), dtype=np.float64)
        if eligible_ref.size:
            conditional_rank = np.searchsorted(
                eligible_ref, corrected_current[selected_mask], side="right"
            ) / float(eligible_ref.size)
            selection_rank[selected_mask] = rank_cutoff + top_fraction * conditional_rank
        else:
            selection_rank[selected_mask] = rank_cutoff
    else:
        selected_mask = np.isfinite(blended_rank) & (blended_rank >= rank_cutoff)
        selection_rank = blended_rank
    minimum_ev = _safe_float(policy.get("minimum_corrected_ev"), np.nan)
    if np.isfinite(minimum_ev):
        selected_mask &= corrected_current >= minimum_ev
    selected = batch_rows.loc[selected_mask].copy()
    selected["selection_score"] = selection_rank[selected_mask]
    selected["dynamic_ev_target"] = (
        fixed_target_net_ev
        if selection_mode == "fixed_corrected_ev_threshold"
        else apply_cutoff
    )
    selected["dynamic_score_threshold"] = (
        fixed_target_net_ev
        if selection_mode == "fixed_corrected_ev_threshold"
        else rank_cutoff
    )
    selected["corrected_expected_ev_reference_cutoff"] = apply_cutoff
    selected["recent_ev_correction"] = current_correction[selected_mask]
    selected["ev_target_local_support"] = current_support[selected_mask]
    selected["ev_target_global_fallback"] = current_scope[selected_mask] != "side_x_archetype"
    selected["mapped_expected_ev"] = mapped_current[selected_mask]
    selected["corrected_expected_ev"] = corrected_current[selected_mask]
    selected["expected_ev_correction_scope"] = current_scope[selected_mask]
    realized_values = realized_ref.to_numpy(dtype=np.float64, copy=False)
    realized_valid = np.isfinite(corrected_ref_full) & np.isfinite(realized_values)
    reference_top = realized_valid & (corrected_ref_full >= apply_cutoff)
    global_target = (
        float(realized_values[reference_top].mean())
        if reference_top.any()
        else np.nan
    )
    index = batch_rows.index
    return selected, {
        "baseline_activity_count": int(selected_mask.sum()),
        "recent_reference_rows": int(len(recent_ref)),
        "reference_rows": int(len(all_prior)),
        "global_dynamic_ev_target": (
            fixed_target_net_ev
            if selection_mode == "fixed_corrected_ev_threshold"
            else global_target
        ),
        "global_dynamic_score_threshold": (
            fixed_target_net_ev
            if selection_mode == "fixed_corrected_ev_threshold"
            else rank_cutoff
        ),
        "apply_cutoff": apply_cutoff,
        "corrected_expected_ev_reference_cutoff": apply_cutoff,
        "recent_ev_correction": pd.Series(current_correction, index=index),
        "local_support": pd.Series(current_support, index=index, dtype="int32"),
        "fallback": pd.Series(current_scope != "side_x_archetype", index=index),
        "mapped_expected_ev": pd.Series(mapped_current, index=index),
        "corrected_expected_ev": pd.Series(corrected_current, index=index),
        "corrected_expected_ev_rank": pd.Series(corrected_ev_rank, index=index),
        "parent_rank": pd.Series(parent_rank, index=index),
        "blended_rank": pd.Series(blended_rank, index=index),
        "ev_rank_blend_weight": ev_rank_blend_weight,
        "corrected_ev_tie_break_parent": tie_break_parent,
        "selection_mode": selection_mode,
        "fixed_target_net_ev": fixed_target_net_ev,
        "robust_daily_residual_trim_fraction": trim_fraction,
        "robust_daily_residual_normalization": (
            "median_iqr" if trim_fraction > 0.0 else "disabled"
        ),
        "global_days_retained": int(global_days_retained),
        "global_daily_residual_iqr": global_iqr,
        "mapping_scope": pd.Series(current_scope, index=index, dtype="object"),
    }


def apply_threshold_basis_policy_to_decisions(
    decisions: Sequence[MutableMapping[str, Any]],
    *,
    policy: Mapping[str, Any] | None,
    store: Any = None,
) -> Sequence[MutableMapping[str, Any]]:
    """Attach threshold-basis rank scores to live decisions.

    Non-selected rows receive a score of 0.0 so the regular rank threshold gate
    rejects them. Selected rows receive scores rescaled into the configured
    top band, matching the replay candidate table semantics.
    """
    if not decisions or not isinstance(policy, Mapping):
        return decisions
    enabled = bool(policy.get("enabled", True))
    if not enabled:
        return decisions
    reference = _load_reference(policy)
    if reference.empty:
        for decision in decisions:
            decision["threshold_basis_reason"] = "missing_reference_candidates"
        return decisions
    min_rows = int(policy.get("min_reference_rows") or 40)
    window_days = int(policy.get("window_days") or 8)
    family = str(policy.get("family") or "ev_target_archetype_reachable_matched_activity")
    policy_id = str(policy.get("policy_id") or policy.get("name") or "threshold_basis_policy")
    recalibration_frequency = str(
        policy.get("recalibration_frequency") or "each_decision_timestamp"
    ).strip().lower()
    rows = _decision_rows(decisions, policy=policy, store=store)
    if rows.empty:
        return decisions
    reference = reference.sort_values("timestamp", kind="stable").reset_index(drop=True)
    reference_ts = reference["timestamp"].astype("int64").to_numpy(
        dtype=np.int64, copy=False
    )
    # Fixed-EV admission with daily recalibration is row-independent within a
    # UTC day: the reference window, residual correction, and EV threshold are
    # identical for every timestamp in that day.  Batch the whole day so we do
    # not rebuild the same side/archetype residual tables up to 24 times.
    daily_fixed_ev = (
        recalibration_frequency in {"1d", "daily", "1d_at_00_utc"}
        and family == "side_archetype_expected_ev_recent_correction"
        and str(policy.get("selection_mode") or "").strip()
        == "fixed_corrected_ev_threshold"
    )
    period_key = rows["timestamp"].dt.floor("D") if daily_fixed_ev else rows["timestamp"]
    for period, batch_rows in rows.groupby(period_key, sort=True):
        period_ts = pd.Timestamp(period)
        reference_asof = (
            period_ts.floor("D")
            if recalibration_frequency in {"1d", "daily", "1d_at_00_utc"}
            else period_ts
        )
        ref_start = reference_asof - pd.Timedelta(days=window_days)
        period_ns = int(reference_asof.value)
        start_ns = int(ref_start.value)
        prior_end = int(np.searchsorted(reference_ts, period_ns, side="left"))
        recent_start = int(np.searchsorted(reference_ts, start_ns, side="left"))
        recent_ref = reference.iloc[recent_start:prior_end]
        all_prior = reference.iloc[:prior_end]
        if "outcome_resolved_at" in reference.columns:
            recent_ref = recent_ref.loc[
                recent_ref["outcome_resolved_at"].lt(reference_asof)
            ]
            all_prior = all_prior.loc[
                all_prior["outcome_resolved_at"].lt(reference_asof)
            ]
            trim_fraction = float(
                np.clip(
                    policy.get("robust_daily_residual_trim_fraction") or 0.0,
                    0.0,
                    0.49,
                )
            )
            if trim_fraction > 0.0:
                # The promoted robust policy is calibrated on outcome-resolution
                # days, matching the portfolio ablation and avoiding a boundary
                # mismatch for entries whose 12h path crosses midnight.
                recent_ref = all_prior.loc[
                    all_prior["outcome_resolved_at"].ge(ref_start)
                ]
        fixed_ev_fallback_allowed = (
            family == "side_archetype_expected_ev_recent_correction"
            and str(policy.get("selection_mode") or "").strip()
            == "fixed_corrected_ev_threshold"
        )
        if len(recent_ref) < min_rows and not fixed_ev_fallback_allowed:
            for idx in batch_rows["_decision_idx"].astype(int).tolist():
                decision = decisions[idx]
                decision["threshold_basis_policy_id"] = policy_id
                decision["threshold_basis_family"] = family
                decision["threshold_basis_rank_score"] = 0.0
                decision["threshold_basis_selected"] = False
                decision["threshold_basis_reason"] = "insufficient_recent_reference"
                decision["threshold_basis_recent_reference_rows"] = int(len(recent_ref))
                decision["threshold_basis_reference_rows"] = int(len(all_prior))
                decision["threshold_basis_recalibration_frequency"] = (
                    recalibration_frequency
                )
                decision["threshold_basis_reference_asof"] = (
                    reference_asof.isoformat()
                )
            continue
        if family == "ev_target_side_archetype_multiplier_before_mlp":
            selected, meta = _select_pre_mlp_multiplier_batch(
                batch_rows,
                recent_ref=recent_ref,
                all_prior=all_prior,
                policy=policy,
            )
        elif family == "side_archetype_expected_ev_recent_correction":
            selected, meta = _select_side_archetype_expected_ev_batch(
                batch_rows,
                recent_ref=recent_ref,
                all_prior=all_prior,
                policy=policy,
            )
        else:
            selected, meta = _select_batch(
                batch_rows,
                recent_ref=recent_ref,
                all_prior=all_prior,
                policy=policy,
            )
        selected_scores = {
            int(row["_decision_idx"]): _safe_float(row.get("selection_score"), 0.0)
            for _, row in selected.iterrows()
        }
        selected_thresholds = {
            int(row["_decision_idx"]): _safe_float(row.get("dynamic_score_threshold"))
            for _, row in selected.iterrows()
        }
        selected_targets = {
            int(row["_decision_idx"]): _safe_float(row.get("dynamic_ev_target"))
            for _, row in selected.iterrows()
        }
        selected_multipliers = {
            int(row["_decision_idx"]): _safe_float(row.get("ev_target_multiplier"), 1.0)
            for _, row in selected.iterrows()
        }
        selected_support = {
            int(row["_decision_idx"]): int(_safe_float(row.get("ev_target_local_support"), 0.0))
            for _, row in selected.iterrows()
        }
        selected_fallback = {
            int(row["_decision_idx"]): bool(row.get("ev_target_global_fallback", False))
            for _, row in selected.iterrows()
        }
        selected_ev_corrections = {
            int(row["_decision_idx"]): _safe_float(row.get("recent_ev_correction"), 0.0)
            for _, row in selected.iterrows()
        }
        selected_mapped_ev = {
            int(row["_decision_idx"]): _safe_float(row.get("mapped_expected_ev"))
            for _, row in selected.iterrows()
        }
        selected_corrected_ev = {
            int(row["_decision_idx"]): _safe_float(row.get("corrected_expected_ev"))
            for _, row in selected.iterrows()
        }
        selected_ev_scope = {
            int(row["_decision_idx"]): str(row.get("expected_ev_correction_scope") or "")
            for _, row in selected.iterrows()
        }
        row_indices = batch_rows.index.to_series().groupby(
            batch_rows["_decision_idx"].astype(int), sort=False
        ).first()
        all_multipliers = {
            int(decision_idx): _safe_float(meta.get("multiplier", {}).get(row_idx), 1.0)
            for decision_idx, row_idx in row_indices.items()
        }
        all_support = {
            int(decision_idx): int(
                _safe_float(meta.get("local_support", {}).get(row_idx), 0.0)
            )
            for decision_idx, row_idx in row_indices.items()
        }
        all_fallback = {
            int(decision_idx): bool(meta.get("fallback", {}).get(row_idx, False))
            for decision_idx, row_idx in row_indices.items()
        }
        all_ev_corrections = {
            int(decision_idx): _safe_float(
                meta.get("recent_ev_correction", {}).get(row_idx), 0.0
            )
            for decision_idx, row_idx in row_indices.items()
        }
        all_mapped_ev = {
            int(decision_idx): _safe_float(
                meta.get("mapped_expected_ev", {}).get(row_idx), np.nan
            )
            for decision_idx, row_idx in row_indices.items()
        }
        all_corrected_ev = {
            int(decision_idx): _safe_float(
                meta.get("corrected_expected_ev", {}).get(row_idx), np.nan
            )
            for decision_idx, row_idx in row_indices.items()
        }
        all_corrected_ev_rank = {
            int(decision_idx): _safe_float(
                meta.get("corrected_expected_ev_rank", {}).get(row_idx), np.nan
            )
            for decision_idx, row_idx in row_indices.items()
        }
        all_parent_rank = {
            int(decision_idx): _safe_float(
                meta.get("parent_rank", {}).get(row_idx), np.nan
            )
            for decision_idx, row_idx in row_indices.items()
        }
        all_blended_rank = {
            int(decision_idx): _safe_float(
                meta.get("blended_rank", {}).get(row_idx), np.nan
            )
            for decision_idx, row_idx in row_indices.items()
        }
        all_ev_scope = {
            int(decision_idx): str(
                meta.get("mapping_scope", {}).get(row_idx, "")
            )
            for decision_idx, row_idx in row_indices.items()
        }
        all_policy_archetypes = {
            int(decision_idx): str(batch_rows.loc[row_idx, "policy_archetype"])
            for decision_idx, row_idx in row_indices.items()
        }
        all_invalid_mapped_ev = {
            int(decision_idx): bool(
                batch_rows.loc[row_idx, "mapped_expected_ev_invalid_sentinel"]
            )
            for decision_idx, row_idx in row_indices.items()
        }
        email_baselines = _email_archetype_baselines_for_batch(
            batch_rows,
            all_prior=all_prior,
            reference_asof=reference_asof,
            policy=policy,
        )
        for idx in batch_rows["_decision_idx"].astype(int).tolist():
            decision = decisions[idx]
            selected_flag = idx in selected_scores
            score = selected_scores.get(idx, 0.0)
            invalid_mapped_ev_sentinel = all_invalid_mapped_ev.get(idx, False)
            mapped_expected_ev = all_mapped_ev.get(
                idx, selected_mapped_ev.get(idx, np.nan)
            )
            chain = dict(decision.get("chain_results") or {})
            payload = {
                "threshold_basis_policy_id": policy_id,
                "threshold_basis_family": family,
                "threshold_basis_policy_archetype": all_policy_archetypes.get(
                    idx, "missing"
                ),
                "threshold_basis_window_days": int(window_days),
                "threshold_basis_recalibration_frequency": recalibration_frequency,
                "threshold_basis_reference_asof": reference_asof.isoformat(),
                "threshold_basis_robust_daily_residual_trim_fraction": _safe_float(
                    meta.get("robust_daily_residual_trim_fraction"),
                    _safe_float(
                        policy.get("robust_daily_residual_trim_fraction"), 0.0
                    ),
                ),
                "threshold_basis_robust_daily_residual_normalization": str(
                    meta.get("robust_daily_residual_normalization")
                    or policy.get("robust_daily_residual_normalization")
                    or "disabled"
                ),
                "threshold_basis_global_days_retained": int(
                    _safe_float(meta.get("global_days_retained"), 0.0)
                ),
                "threshold_basis_rank_score": float(np.clip(score, 0.0, 1.0)),
                "threshold_basis_rank_score_source": f"threshold_basis:{policy_id}",
                "threshold_basis_selected": bool(selected_flag),
                "threshold_basis_reason": (
                    "selected"
                    if selected_flag
                    else "invalid_mapped_expected_ev_sentinel"
                    if invalid_mapped_ev_sentinel
                    else "not_selected"
                ),
                "threshold_basis_dynamic_ev_target": selected_targets.get(
                    idx, _safe_float(meta.get("global_dynamic_ev_target"), np.nan)
                ),
                "threshold_basis_dynamic_score_threshold": selected_thresholds.get(
                    idx,
                    _safe_float(meta.get("global_dynamic_score_threshold"), np.nan),
                ),
                "threshold_basis_recent_reference_rows": int(meta["recent_reference_rows"]),
                "threshold_basis_reference_rows": int(meta["reference_rows"]),
                "threshold_basis_baseline_activity_count": int(meta["baseline_activity_count"]),
                "threshold_basis_global_dynamic_ev_target": meta["global_dynamic_ev_target"],
                "threshold_basis_global_dynamic_score_threshold": meta[
                    "global_dynamic_score_threshold"
                ],
                "threshold_basis_apply_cutoff": _safe_float(
                    meta.get("apply_cutoff"), np.nan
                ),
                "threshold_basis_ev_target_multiplier": all_multipliers.get(
                    idx, selected_multipliers.get(idx, 1.0)
                ),
                "threshold_basis_ev_target_local_support": all_support.get(
                    idx, selected_support.get(idx, 0)
                ),
                "threshold_basis_ev_target_global_fallback": all_fallback.get(
                    idx, selected_fallback.get(idx, False)
                ),
                "threshold_basis_mapped_expected_ev_side_archetype": mapped_expected_ev,
                "threshold_basis_mapped_expected_ev_valid": bool(
                    np.isfinite(mapped_expected_ev)
                ),
                "threshold_basis_invalid_mapped_expected_ev_sentinel": bool(
                    invalid_mapped_ev_sentinel
                ),
                "threshold_basis_side_archetype_recent_ev_correction": all_ev_corrections.get(
                    idx, selected_ev_corrections.get(idx, 0.0)
                ),
                "threshold_basis_corrected_expected_ev": all_corrected_ev.get(
                    idx, selected_corrected_ev.get(idx, np.nan)
                ),
                "threshold_basis_corrected_expected_ev_rank": all_corrected_ev_rank.get(
                    idx, np.nan
                ),
                "threshold_basis_parent_rank": all_parent_rank.get(idx, np.nan),
                "threshold_basis_blended_rank": all_blended_rank.get(idx, np.nan),
                "threshold_basis_ev_rank_blend_weight": _safe_float(
                    meta.get("ev_rank_blend_weight"), 1.0
                ),
                "threshold_basis_expected_ev_correction_scope": all_ev_scope.get(
                    idx, selected_ev_scope.get(idx, "")
                ),
                **email_baselines.get(idx, {}),
            }
            decision.update(payload)
            chain.update(payload)
            decision["chain_results"] = chain
    return decisions
