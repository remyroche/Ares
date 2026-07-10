"""Market-state overlays for side x archetype trading precision.

This module fits small frozen ``regime_ev_calibration`` artifacts from replay
or OOS candidate ledgers.  The fitted effects are keyed by the inference
surface we actually trade, usually ``side_name x policy_archetype``.  Positive
effects are unfavorable states and lower the score through
``apply_regime_ev_calibration``; negative effects are favorable states and
raise the score.

The implementation is intentionally simple and memory-conscious: it scores one
feature at a time, stores effects as bucketed curves, downcasts numerics, and
uses Numba for the inner bucket-stat loop when available.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

try:  # pragma: no cover - exercised when numba is installed locally.
    from numba import njit
except Exception:  # pragma: no cover - fallback is tested.
    njit = None


DEFAULT_MARKET_STATE_PREFIXES: tuple[str, ...] = (
    "canonical__",
    "badregime__",
    "market_state__",
    "regime_",
    "ctx_",
    "gmm_",
    "aegmm_",
    "ae_",
    "AE_",
    "mahalanobis",
    "reconstruction",
    "archetype_hit_surprise_",
    "hit_surprise_",
    "support_drift_",
    "leaf_drift_",
    "source_drift_",
    "exec_opportunity_pressure",
    "exec_adverse_path_pressure",
    "__derived_",
)

DEFAULT_EXCLUDED_SUBSTRINGS: tuple[str, ...] = (
    "target",
    "label",
    "future",
    "oracle",
    "realized",
    "ret_net",
    "net_return",
    "gross_return",
    "pnl",
    "profit",
    "loss",
    "first_touch",
    "full_path",
    "timeout",
    "stop",
    "bad_mae",
    "bad_MAE",
    "exec_margin",
    "hit_rate",
    "clean_positive",
    "dirty_positive",
)

DEFAULT_OUTCOME_COLUMNS: tuple[str, ...] = (
    "ret_net_notional",
    "net_return_notional",
    "net_return",
    "ret_net",
    "net_pnl_pct",
    "mean_net_return_per_trade",
)

DEFAULT_STOP_COLUMNS: tuple[str, ...] = (
    "full_sl",
    "full_stop_loss",
    "full_sl_hit",
    "stop_or_adverse",
    "stop_or_adverse_rate",
)

DEFAULT_TIMEOUT_COLUMNS: tuple[str, ...] = (
    "timeout",
    "timeout_hit",
    "timeout_rate",
)


@dataclass(frozen=True)
class MarketStateOverlayConfig:
    side_col: str = "side_name"
    archetype_col: str = "policy_archetype"
    timestamp_col: str = "timestamp"
    source_score_col: str = "rank_pct"
    adjusted_score_col: str = "score_market_state_calibrated"
    risk_score_col: str = "market_state_risk_score"
    effect_count_col: str = "market_state_effect_count"
    n_buckets: int = 5
    min_group_rows: int = 80
    min_bucket_rows: int = 15
    min_feature_coverage: float = 0.20
    max_features_per_group: int = 8
    min_abs_effect: float = 0.0025
    max_abs_effect: float = 0.040
    risk_cap: float = 0.080
    effect_scale: float = 1.75
    bad_rate_penalty_scale: float = 0.035
    hit_rate_bonus_scale: float = 0.015
    shrinkage_prior_rows: float = 120.0
    stability_prior: float = 0.0025
    quantile_low: float = 0.05
    quantile_high: float = 0.95
    random_state: int = 42


@dataclass(frozen=True)
class MarketStateOverlayFitResult:
    artifact: dict[str, Any]
    effect_metrics: pd.DataFrame
    group_metrics: pd.DataFrame


def resolve_archetype_column(frame: pd.DataFrame, requested: str = "policy_archetype") -> str:
    """Return the best available inference archetype column."""

    candidates = (
        requested,
        "archetype_policy_key",
        "policy_archetype",
        "local_side_archetype",
        "source_archetype",
        "archetype_label_family",
    )
    for col in candidates:
        if col in frame.columns:
            return col
    raise ValueError("No inference archetype column found")


def resolve_outcome_column(frame: pd.DataFrame, requested: str | None = None) -> str:
    if requested and requested in frame.columns:
        return requested
    for col in DEFAULT_OUTCOME_COLUMNS:
        if col in frame.columns:
            return col
    raise ValueError(f"No net outcome column found. Tried: {DEFAULT_OUTCOME_COLUMNS}")


def resolve_optional_binary_column(frame: pd.DataFrame, candidates: Sequence[str]) -> str | None:
    for col in candidates:
        if col in frame.columns:
            return col
    return None


def select_market_state_columns(
    frame: pd.DataFrame,
    *,
    include_prefixes: Sequence[str] = DEFAULT_MARKET_STATE_PREFIXES,
    excluded_substrings: Sequence[str] = DEFAULT_EXCLUDED_SUBSTRINGS,
    required_columns: Iterable[str] = (),
    max_columns: int = 0,
) -> list[str]:
    """Pick inference-compatible state/context columns from a candidate ledger."""

    required = {str(c) for c in required_columns}
    excluded = tuple(str(x).lower() for x in excluded_substrings)
    prefixes = tuple(str(x) for x in include_prefixes)
    cols: list[str] = []
    available = set(map(str, frame.columns))
    for col in frame.columns:
        text = str(col)
        lower = text.lower()
        if text in required:
            continue
        if text.startswith("gmm_prob_"):
            suffix = text.removeprefix("gmm_prob_")
            if f"gmm_cluster_posterior_{suffix}" in available:
                continue
        if any(token in lower for token in excluded):
            continue
        if not prefixes or text.startswith(prefixes):
            if pd.api.types.is_numeric_dtype(frame[col]):
                cols.append(text)
    if max_columns and len(cols) > int(max_columns):
        finite_share = []
        for col in cols:
            values = pd.to_numeric(frame[col], errors="coerce")
            finite_share.append(float(np.isfinite(values.to_numpy(dtype=np.float64, copy=False)).mean()))
        order = np.argsort(np.asarray(finite_share, dtype=np.float32))[::-1]
        cols = [cols[int(i)] for i in order[: int(max_columns)]]
    return cols


if njit is not None:

    @njit(cache=True)
    def _bucket_stats_numba(
        buckets: np.ndarray,
        returns: np.ndarray,
        bad: np.ndarray,
        n_buckets: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        counts = np.zeros(n_buckets, dtype=np.int64)
        sums = np.zeros(n_buckets, dtype=np.float64)
        sumsq = np.zeros(n_buckets, dtype=np.float64)
        bad_counts = np.zeros(n_buckets, dtype=np.int64)
        for i in range(buckets.shape[0]):
            b = int(buckets[i])
            r = float(returns[i])
            if b < 0 or b >= n_buckets or not np.isfinite(r):
                continue
            counts[b] += 1
            sums[b] += r
            sumsq[b] += r * r
            if bad[i] > 0:
                bad_counts[b] += 1
        return counts, sums, sumsq, bad_counts

else:
    _bucket_stats_numba = None


def _bucket_stats(
    buckets: np.ndarray,
    returns: np.ndarray,
    bad: np.ndarray,
    n_buckets: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    b = np.asarray(buckets, dtype=np.int16)
    r = np.asarray(returns, dtype=np.float32)
    bad_arr = np.asarray(bad, dtype=np.int8)
    valid = (b >= 0) & (b < int(n_buckets)) & np.isfinite(r)
    if _bucket_stats_numba is not None:
        return _bucket_stats_numba(b, r, bad_arr, int(n_buckets))
    counts = np.bincount(b[valid], minlength=int(n_buckets)).astype(np.int64, copy=False)
    sums = np.bincount(b[valid], weights=r[valid], minlength=int(n_buckets)).astype(np.float64, copy=False)
    sumsq = np.bincount(b[valid], weights=(r[valid] * r[valid]), minlength=int(n_buckets)).astype(
        np.float64,
        copy=False,
    )
    bad_counts = np.bincount(b[valid], weights=bad_arr[valid], minlength=int(n_buckets)).astype(
        np.int64,
        copy=False,
    )
    return counts, sums, sumsq, bad_counts


def _safe_numeric_array(frame: pd.DataFrame, column: str, dtype: Any = np.float32) -> np.ndarray:
    return pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=dtype, copy=False)


def _group_labels(side: pd.Series, archetype: pd.Series) -> tuple[np.ndarray, list[tuple[str, str]]]:
    labels = side.astype(str).to_numpy(dtype=object, copy=False)
    arch = archetype.astype(str).to_numpy(dtype=object, copy=False)
    combined = np.asarray([f"{s}\0{a}" for s, a in zip(labels, arch)], dtype=object)
    codes, uniques = pd.factorize(combined, sort=True)
    groups: list[tuple[str, str]] = []
    for raw in uniques:
        s, a = str(raw).split("\0", 1)
        groups.append((s, a))
    return codes.astype(np.int32, copy=False), groups


def _build_bad_array(frame: pd.DataFrame, returns: np.ndarray) -> np.ndarray:
    stop_col = resolve_optional_binary_column(frame, DEFAULT_STOP_COLUMNS)
    timeout_col = resolve_optional_binary_column(frame, DEFAULT_TIMEOUT_COLUMNS)
    bad = ~np.isfinite(returns) | (returns <= 0.0)
    if stop_col is not None:
        bad |= _safe_numeric_array(frame, stop_col) > 0.5
    if timeout_col is not None:
        bad |= _safe_numeric_array(frame, timeout_col) > 0.5
    return bad.astype(np.int8, copy=False)


def _unique_quantiles(values: np.ndarray, cfg: MarketStateOverlayConfig) -> np.ndarray:
    finite = values[np.isfinite(values)]
    if finite.size < max(int(cfg.min_group_rows), int(cfg.n_buckets) * int(cfg.min_bucket_rows)):
        return np.asarray([], dtype=np.float32)
    qs = np.linspace(float(cfg.quantile_low), float(cfg.quantile_high), int(cfg.n_buckets) + 1)[1:-1]
    raw = np.nanquantile(finite.astype(np.float64, copy=False), qs)
    unique = np.unique(raw[np.isfinite(raw)])
    return unique.astype(np.float32, copy=False)


def _bucket_effects(
    *,
    values: np.ndarray,
    returns: np.ndarray,
    bad: np.ndarray,
    baseline_return: float,
    baseline_bad_rate: float,
    quantiles: np.ndarray,
    cfg: MarketStateOverlayConfig,
) -> tuple[dict[int, float], dict[str, Any]]:
    n_buckets = int(len(quantiles) + 1)
    buckets = np.digitize(values.astype(np.float32, copy=False), quantiles, right=True).astype(np.int16, copy=False)
    buckets[~np.isfinite(values)] = -1
    counts, sums, sumsq, bad_counts = _bucket_stats(buckets, returns, bad, n_buckets)
    effects: dict[int, float] = {}
    bucket_rows: list[dict[str, Any]] = []
    objective = 0.0
    max_abs = 0.0
    for bucket in range(n_buckets):
        count = int(counts[bucket])
        if count < int(cfg.min_bucket_rows):
            effects[bucket] = 0.0
            continue
        mean_ret = float(sums[bucket] / max(count, 1))
        var = max(float(sumsq[bucket] / max(count, 1) - mean_ret * mean_ret), 0.0)
        std = float(np.sqrt(var))
        bad_rate = float(bad_counts[bucket] / max(count, 1))
        support = float(count / (count + float(cfg.shrinkage_prior_rows)))
        lift = mean_ret - float(baseline_return)
        stability = abs(lift) / (abs(lift) + std + float(cfg.stability_prior))
        bad_lift = bad_rate - float(baseline_bad_rate)
        raw_effect = -float(cfg.effect_scale) * lift
        raw_effect += float(cfg.bad_rate_penalty_scale) * bad_lift
        raw_effect -= float(cfg.hit_rate_bonus_scale) * (-bad_lift)
        effect = float(np.clip(raw_effect * support * stability, -float(cfg.max_abs_effect), float(cfg.max_abs_effect)))
        if abs(effect) < float(cfg.min_abs_effect):
            effect = 0.0
        effects[bucket] = effect
        max_abs = max(max_abs, abs(effect))
        objective += abs(effect) * count
        bucket_rows.append(
            {
                "bucket": int(bucket),
                "rows": count,
                "mean_return": mean_ret,
                "baseline_return": float(baseline_return),
                "return_lift": lift,
                "bad_rate": bad_rate,
                "baseline_bad_rate": float(baseline_bad_rate),
                "bad_rate_lift": bad_lift,
                "support_shrink": support,
                "stability_shrink": stability,
                "effect": effect,
            }
        )
    return effects, {
        "objective": float(objective / max(float(np.sum(counts)), 1.0)),
        "max_abs_effect": float(max_abs),
        "bucket_rows": bucket_rows,
        "n_buckets": n_buckets,
    }


def fit_market_state_archetype_overlay(
    frame: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    outcome_col: str | None = None,
    config: MarketStateOverlayConfig = MarketStateOverlayConfig(),
    valid_from: str | None = None,
    valid_to: str | None = None,
) -> MarketStateOverlayFitResult:
    """Fit a side x archetype market-state calibration artifact."""

    if frame.empty:
        raise ValueError("Cannot fit market-state overlay on an empty frame")
    arch_col = resolve_archetype_column(frame, config.archetype_col)
    outcome = resolve_outcome_column(frame, outcome_col)
    if config.side_col not in frame.columns:
        raise ValueError(f"side column missing: {config.side_col}")
    required = {config.side_col, arch_col, outcome}
    feature_cols = [str(c) for c in feature_columns if str(c) in frame.columns and str(c) not in required]
    if not feature_cols:
        raise ValueError("No market-state feature columns available")

    returns = _safe_numeric_array(frame, outcome)
    bad = _build_bad_array(frame, returns)
    group_codes, groups = _group_labels(frame[config.side_col], frame[arch_col])
    effects: list[dict[str, Any]] = []
    effect_rows: list[dict[str, Any]] = []
    group_rows: list[dict[str, Any]] = []
    for group_code, (side, archetype) in enumerate(groups):
        group_mask = group_codes == int(group_code)
        group_valid = group_mask & np.isfinite(returns)
        group_n = int(np.sum(group_valid))
        if group_n < int(config.min_group_rows):
            continue
        group_ret = returns[group_valid]
        group_bad = bad[group_valid]
        baseline_return = float(np.nanmean(group_ret)) if group_ret.size else 0.0
        baseline_bad_rate = float(np.mean(group_bad > 0)) if group_bad.size else 0.0
        group_rows.append(
            {
                "side_name": side,
                "archetype_policy_key": archetype,
                "rows": group_n,
                "baseline_return": baseline_return,
                "baseline_bad_rate": baseline_bad_rate,
            }
        )
        candidates: list[tuple[float, dict[str, Any], list[dict[str, Any]]]] = []
        for col in feature_cols:
            values_all = _safe_numeric_array(frame, col)
            values = values_all[group_valid]
            coverage = float(np.isfinite(values).mean()) if values.size else 0.0
            if coverage < float(config.min_feature_coverage):
                continue
            quantiles = _unique_quantiles(values, config)
            if quantiles.size == 0:
                continue
            effect_map, meta = _bucket_effects(
                values=values,
                returns=group_ret,
                bad=group_bad,
                baseline_return=baseline_return,
                baseline_bad_rate=baseline_bad_rate,
                quantiles=quantiles,
                cfg=config,
            )
            if float(meta["max_abs_effect"]) <= 0.0:
                continue
            effect = {
                "side_name": side,
                "archetype_policy_key": archetype,
                "feature_col": col,
                "shape": "bucketed",
                "params": {
                    "quantiles": [float(x) for x in quantiles.tolist()],
                    "effects": {str(k): float(v) for k, v in effect_map.items()},
                },
                "rows": group_n,
                "coverage": coverage,
                "baseline_return": baseline_return,
                "baseline_bad_rate": baseline_bad_rate,
                "objective": float(meta["objective"]),
                "max_abs_effect": float(meta["max_abs_effect"]),
                "valid_from": valid_from or "",
                "valid_to": valid_to or "",
            }
            bucket_rows = []
            for bucket_row in meta["bucket_rows"]:
                row = {
                    "side_name": side,
                    "archetype_policy_key": archetype,
                    "feature_col": col,
                    "coverage": coverage,
                    **bucket_row,
                }
                bucket_rows.append(row)
            candidates.append((float(meta["objective"]), effect, bucket_rows))
        candidates.sort(key=lambda item: item[0], reverse=True)
        for _, effect, bucket_rows in candidates[: int(config.max_features_per_group)]:
            effects.append(effect)
            effect_rows.extend(bucket_rows)

    artifact = {
        "policy_id": "market_state_archetype_overlay_v1",
        "artifact_type": "regime_ev_calibration",
        "generated_by": "fit_market_state_archetype_overlay",
        "side_col": config.side_col,
        "archetype_col": arch_col,
        "source_score_col": config.source_score_col,
        "adjusted_score_col": config.adjusted_score_col,
        "risk_score_col": config.risk_score_col,
        "effect_count_col": config.effect_count_col,
        "risk_cap": float(config.risk_cap),
        "risk_cap_positive": float(config.risk_cap),
        "risk_cap_negative": float(config.risk_cap),
        "effects": effects,
        "fit_config": {
            "n_buckets": int(config.n_buckets),
            "min_group_rows": int(config.min_group_rows),
            "min_bucket_rows": int(config.min_bucket_rows),
            "max_features_per_group": int(config.max_features_per_group),
            "min_abs_effect": float(config.min_abs_effect),
            "max_abs_effect": float(config.max_abs_effect),
            "effect_scale": float(config.effect_scale),
            "bad_rate_penalty_scale": float(config.bad_rate_penalty_scale),
            "hit_rate_bonus_scale": float(config.hit_rate_bonus_scale),
            "shrinkage_prior_rows": float(config.shrinkage_prior_rows),
        },
        "feature_columns": feature_cols,
        "outcome_col": outcome,
        "valid_from": valid_from or "",
        "valid_to": valid_to or "",
        "numba_bucket_stats": bool(_bucket_stats_numba is not None),
    }
    effect_metrics = pd.DataFrame(effect_rows)
    group_metrics = pd.DataFrame(group_rows)
    for df in (effect_metrics, group_metrics):
        for col in df.select_dtypes(include=["float64"]).columns:
            df[col] = df[col].astype(np.float32, copy=False)
        for col in df.select_dtypes(include=["int64"]).columns:
            df[col] = pd.to_numeric(df[col], downcast="integer")
    return MarketStateOverlayFitResult(
        artifact=artifact,
        effect_metrics=effect_metrics,
        group_metrics=group_metrics,
    )


def topk_precision_metrics(
    frame: pd.DataFrame,
    *,
    score_col: str,
    outcome_col: str,
    group_cols: Sequence[str] = (),
    top_fracs: Sequence[float] = (0.10, 0.20, 0.30),
) -> pd.DataFrame:
    """Compute net-return precision metrics for top-k score slices."""

    needed = [score_col, outcome_col, *group_cols]
    work = frame.loc[:, [c for c in needed if c in frame.columns]]
    score = pd.to_numeric(work[score_col], errors="coerce").to_numpy(dtype=np.float32, copy=False)
    ret = pd.to_numeric(work[outcome_col], errors="coerce").to_numpy(dtype=np.float32, copy=False)
    valid = np.isfinite(score) & np.isfinite(ret)
    if not group_cols:
        groups = [("__all__", np.flatnonzero(valid))]
    else:
        key = work.loc[:, list(group_cols)].astype(str).agg("\0".join, axis=1)
        codes, uniques = pd.factorize(key, sort=True)
        groups = [(str(u), np.flatnonzero(valid & (codes == i))) for i, u in enumerate(uniques)]
    rows: list[dict[str, Any]] = []
    for group_key, idx in groups:
        if idx.size == 0:
            continue
        order = idx[np.argsort(score[idx])[::-1]]
        parts = group_key.split("\0") if group_cols else ["__all__"]
        base = {col: parts[pos] for pos, col in enumerate(group_cols)}
        for frac in top_fracs:
            n = max(1, int(np.ceil(float(frac) * order.size)))
            chosen = order[:n]
            rr = ret[chosen]
            rows.append(
                {
                    **base,
                    "top_frac": float(frac),
                    "rows": int(chosen.size),
                    "mean_return": float(np.nanmean(rr)) if rr.size else np.nan,
                    "positive_return_rate": float(np.mean(rr > 0.0)) if rr.size else np.nan,
                    "sum_return": float(np.nansum(rr)) if rr.size else np.nan,
                }
            )
    return pd.DataFrame(rows)
