"""Economic relevance scoring for regime features and composites.

The existing unsupervised regime-learning stack can generate many plausible
state descriptors.  This module scores those descriptors against the two trading
questions that matter for the current EV/EBM calibration layer:

* demotion in the global top10%: rows the policy would actually trade but
  should not have traded;
* promotion in the global top20% excluding top10%: rows that are near the
  threshold, not currently traded, and favorable for a specific side x
  archetype.

All scoring is local to ``side x archetype``.  The top-k denominator is global
within the selected period/month so local groups do not get artificially easy
thresholds.  Wider top15/top20/top30 slices are kept as diagnostics, not as the
primary EBM candidate source.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import importlib.util
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from extreme_price_movements.evm_latent_state_discovery import is_market_context_shock_entropy_feature


_LIGHTGBM_AVAILABLE = importlib.util.find_spec("lightgbm") is not None


def _lightgbm_module() -> Any:
    if not _LIGHTGBM_AVAILABLE:
        return None
    try:
        import lightgbm as lgb  # type: ignore

        return lgb
    except Exception:
        return None


@dataclass(frozen=True)
class EconomicRegimeRelevanceConfig:
    side_col: str = "side_name"
    archetype_col: str = "archetype_policy_key"
    score_col: str = "score_meta_base_soft_label"
    month_col: str = "month"
    timestamp_col: str = "__ts__"
    ev_col: str = "ev_after_1pct"
    clean_col: str = "clean_exec"
    dirty_col: str = "dirty_positive"
    bad_mae_col: str = "full_path_bad_mae_1r"
    timeout_col: str = "timeout"
    stop_col: str = "stop_or_adverse"
    week_col: str = "week_start"
    trade_top_fraction: float = 0.10
    negative_top_fraction: float = 0.10
    negative_diagnostic_top_fractions: tuple[float, float] = (0.15, 0.20)
    positive_outer_top_fraction: float = 0.20
    positive_diagnostic_outer_fraction: float = 0.30
    positive_excluded_top_fraction: float = 0.10
    candidate_tasks: tuple[str, str] = ("demote_top10", "promote_top20_not_top10")
    diagnostic_tasks: tuple[str, str, str] = (
        "diagnostic_negative_top15",
        "diagnostic_negative_top20",
        "diagnostic_positive_top30_not_top10",
    )
    min_group_rows: int = 300
    min_population_rows: int = 60
    min_state_rows: int = 20
    min_feature_coverage: float = 0.35
    min_unique_values: int = 8
    max_features_per_group: int = 80
    max_features_for_composites: int = 10
    max_composites_per_group_task: int = 80
    quantiles: tuple[float, float] = (1.0 / 3.0, 2.0 / 3.0)
    nonlinear_bins: int = 7
    nonlinear_min_bin_rows: int = 20
    negative_ev_weight: float = 20.0
    positive_ev_weight: float = 20.0
    temporal_tail_fraction: float = 0.25
    temporal_score_weight: float = 0.25
    min_candidate_score: float = 0.03
    materialize_composite_intensity: bool = True
    lgbm_enabled: bool = True
    lgbm_min_rows: int = 250
    lgbm_validation_fraction: float = 0.30
    lgbm_max_features: int = 80
    lgbm_n_estimators: int = 160
    lgbm_learning_rate: float = 0.04
    lgbm_num_leaves: int = 31
    lgbm_max_depth: int = 5
    lgbm_min_child_samples: int = 40
    random_state: int = 42


@dataclass(frozen=True)
class EconomicRegimeRelevanceResult:
    feature_metrics: pd.DataFrame
    composite_metrics: pd.DataFrame
    lgbm_feature_metrics: pd.DataFrame
    lgbm_model_metrics: pd.DataFrame
    selected_candidates: pd.DataFrame
    composite_definitions: list[dict[str, Any]] = field(default_factory=list)
    ebm_candidate_manifest: dict[str, Any] = field(default_factory=dict)


def _safe_numeric(frame: pd.DataFrame, col: str, default: float = np.nan) -> pd.Series:
    if col in frame.columns:
        return pd.to_numeric(frame[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    return pd.Series(default, index=frame.index, dtype="float32")


def _binary(frame: pd.DataFrame, col: str) -> pd.Series:
    return _safe_numeric(frame, col, 0.0).fillna(0.0).gt(0.5)


def add_global_topk_surprise_targets(
    frame: pd.DataFrame,
    *,
    config: EconomicRegimeRelevanceConfig = EconomicRegimeRelevanceConfig(),
) -> pd.DataFrame:
    """Return a shallow copy with global top-k surprise populations/targets."""

    out = frame.copy()
    score = _safe_numeric(out, config.score_col)
    if config.month_col in out.columns:
        rank = score.groupby(out[config.month_col], sort=False).rank(pct=True, method="first")
    else:
        rank = score.rank(pct=True, method="first")
    out["url_global_score_rank_pct"] = rank.astype("float32")
    trade_top = rank.ge(1.0 - float(config.trade_top_fraction))
    negative_top = rank.ge(1.0 - float(config.negative_top_fraction))
    top15 = rank.ge(1.0 - float(config.negative_diagnostic_top_fractions[0]))
    top20 = rank.ge(1.0 - float(config.negative_diagnostic_top_fractions[1]))
    positive_outer = rank.ge(1.0 - float(config.positive_outer_top_fraction))
    positive_diag_outer = rank.ge(1.0 - float(config.positive_diagnostic_outer_fraction))
    excluded_top = rank.ge(1.0 - float(config.positive_excluded_top_fraction))
    ev = _safe_numeric(out, config.ev_col, 0.0).fillna(0.0)
    clean = _binary(out, config.clean_col)
    dirty = _binary(out, config.dirty_col)
    bad = _binary(out, config.bad_mae_col)
    timeout = _binary(out, config.timeout_col)
    stop = _binary(out, config.stop_col)
    bad_trade = ev.le(0.0) | ~clean | dirty | bad | timeout | stop
    good_trade = ev.gt(0.0) & clean & ~timeout & ~stop
    out["url_trade_top10_population"] = trade_top.astype("int8")
    out["url_demote_top10_population"] = negative_top.astype("int8")
    out["url_demote_top10_target"] = (negative_top & bad_trade).astype("int8")
    out["url_promote_top20_not_top10_population"] = (positive_outer & ~excluded_top).astype("int8")
    out["url_promote_top20_not_top10_target"] = (positive_outer & ~excluded_top & good_trade).astype("int8")
    out["url_negative_top15_population"] = top15.astype("int8")
    out["url_negative_top15_target"] = (top15 & bad_trade).astype("int8")
    out["url_negative_top20_population"] = top20.astype("int8")
    out["url_negative_top20_target"] = (top20 & bad_trade).astype("int8")
    out["url_positive_mid30_population"] = (positive_diag_outer & ~excluded_top).astype("int8")
    out["url_positive_mid30_target"] = (positive_diag_outer & ~excluded_top & good_trade).astype("int8")
    return out


def _feature_thresholds(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    config: EconomicRegimeRelevanceConfig,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for feature in feature_columns:
        if feature not in frame.columns or not pd.api.types.is_numeric_dtype(frame[feature]):
            continue
        values = _safe_numeric(frame, feature)
        coverage = float(values.notna().mean())
        if coverage < float(config.min_feature_coverage):
            continue
        unique = int(values.nunique(dropna=True))
        if unique < int(config.min_unique_values):
            continue
        q_low, q_high = values.quantile(list(config.quantiles)).to_numpy(dtype=float)
        if not np.isfinite(q_low) or not np.isfinite(q_high) or q_low >= q_high:
            continue
        rows.append(
            {
                "feature": str(feature),
                "q_low": float(q_low),
                "q_high": float(q_high),
                "coverage": coverage,
                "n_unique": unique,
            }
        )
    return rows


def _assign_bin(values: pd.Series, threshold: Mapping[str, Any]) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)
    out = pd.Series("mid", index=values.index, dtype=object)
    out.loc[numeric.le(float(threshold["q_low"]))] = "low"
    out.loc[numeric.gt(float(threshold["q_high"]))] = "high"
    out.loc[numeric.isna()] = "missing"
    return out


def _bin_depth(values: pd.Series, threshold: Mapping[str, Any], bin_name: str) -> pd.Series:
    """Continuous depth inside a frozen low/mid/high state.

    The binary composite says whether a row is in a state.  This depth says how
    strongly the row is inside that state, preserving intensity for EBM/GAM
    calibrators without making the feature non-causal.
    """

    numeric = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)
    q_low = float(threshold.get("q_low", np.nan))
    q_high = float(threshold.get("q_high", np.nan))
    width = max(q_high - q_low, 1e-6)
    if not np.isfinite(q_low) or not np.isfinite(q_high) or q_low >= q_high:
        return pd.Series(0.0, index=values.index, dtype="float32")
    if bin_name == "low":
        depth = (q_low - numeric) / width
    elif bin_name == "high":
        depth = (numeric - q_high) / width
    elif bin_name == "mid":
        depth = np.minimum(numeric - q_low, q_high - numeric) / width
    else:
        depth = pd.Series(0.0, index=values.index, dtype="float32")
    return pd.Series(depth, index=values.index).clip(lower=0.0, upper=5.0).fillna(0.0).astype("float32")


def _nonlinear_feature_summary(
    *,
    task: str,
    values: pd.Series,
    population: pd.Series,
    target: pd.Series,
    ev: pd.Series,
    baseline_rate: float,
    baseline_ev: float,
    config: EconomicRegimeRelevanceConfig,
) -> dict[str, Any]:
    mask = population & values.notna()
    if int(mask.sum()) < max(int(config.min_population_rows), int(config.nonlinear_min_bin_rows) * 2):
        return {
            "nonlinear_relevance_score": 0.0,
            "nonlinear_target_rate_span": 0.0,
            "nonlinear_ev_span": 0.0,
            "nonlinear_best_bin": "",
            "nonlinear_best_bin_rows": 0,
            "nonlinear_best_bin_target_rate": np.nan,
            "nonlinear_best_bin_ev": np.nan,
        }
    x = pd.to_numeric(values.loc[mask], errors="coerce").to_numpy(dtype=np.float64, copy=False)
    y = target.loc[mask].astype("float32").to_numpy(dtype=np.float32, copy=False)
    e = ev.loc[mask].astype("float32").to_numpy(dtype=np.float32, copy=False)
    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(e)
    x = x[finite]
    y = y[finite]
    e = e[finite]
    if x.size < max(int(config.min_population_rows), int(config.nonlinear_min_bin_rows) * 2):
        return {
            "nonlinear_relevance_score": 0.0,
            "nonlinear_target_rate_span": 0.0,
            "nonlinear_ev_span": 0.0,
            "nonlinear_best_bin": "",
            "nonlinear_best_bin_rows": 0,
            "nonlinear_best_bin_target_rate": np.nan,
            "nonlinear_best_bin_ev": np.nan,
        }
    q = np.linspace(0.0, 1.0, max(int(config.nonlinear_bins), 3) + 1)
    edges = np.unique(np.nanquantile(x, q))
    if edges.size < 3:
        return {
            "nonlinear_relevance_score": 0.0,
            "nonlinear_target_rate_span": 0.0,
            "nonlinear_ev_span": 0.0,
            "nonlinear_best_bin": "",
            "nonlinear_best_bin_rows": 0,
            "nonlinear_best_bin_target_rate": np.nan,
            "nonlinear_best_bin_ev": np.nan,
        }
    bin_id = np.searchsorted(edges[1:-1], x, side="right")
    rows: list[tuple[int, int, float, float, float]] = []
    for i in range(edges.size - 1):
        local = bin_id == i
        n = int(local.sum())
        if n < int(config.nonlinear_min_bin_rows):
            continue
        rate = float(np.mean(y[local]))
        mean_ev = float(np.mean(e[local]))
        if _is_negative_task(task):
            score = (rate - float(baseline_rate)) + max(float(baseline_ev) - mean_ev, 0.0) * float(config.negative_ev_weight)
        else:
            score = (rate - float(baseline_rate)) + max(mean_ev - float(baseline_ev), 0.0) * float(config.positive_ev_weight)
        rows.append((i, n, rate, mean_ev, float(score)))
    if not rows:
        return {
            "nonlinear_relevance_score": 0.0,
            "nonlinear_target_rate_span": 0.0,
            "nonlinear_ev_span": 0.0,
            "nonlinear_best_bin": "",
            "nonlinear_best_bin_rows": 0,
            "nonlinear_best_bin_target_rate": np.nan,
            "nonlinear_best_bin_ev": np.nan,
        }
    best = max(rows, key=lambda item: item[4])
    rates = [item[2] for item in rows]
    evs = [item[3] for item in rows]
    return {
        "nonlinear_relevance_score": float(max(best[4], 0.0)),
        "nonlinear_target_rate_span": float(max(rates) - min(rates)),
        "nonlinear_ev_span": float(max(evs) - min(evs)),
        "nonlinear_best_bin": f"q{best[0]}",
        "nonlinear_best_bin_rows": int(best[1]),
        "nonlinear_best_bin_target_rate": float(best[2]),
        "nonlinear_best_bin_ev": float(best[3]),
    }


def _population_task(frame: pd.DataFrame, task: str) -> tuple[pd.Series, pd.Series]:
    if task == "demote_top10":
        return (
            _safe_numeric(frame, "url_demote_top10_population", 0.0).fillna(0.0).gt(0.5),
            _safe_numeric(frame, "url_demote_top10_target", 0.0).fillna(0.0).gt(0.5),
        )
    if task == "promote_top20_not_top10":
        return (
            _safe_numeric(frame, "url_promote_top20_not_top10_population", 0.0).fillna(0.0).gt(0.5),
            _safe_numeric(frame, "url_promote_top20_not_top10_target", 0.0).fillna(0.0).gt(0.5),
        )
    if task in {"negative_top15", "diagnostic_negative_top15"}:
        return (
            _safe_numeric(frame, "url_negative_top15_population", 0.0).fillna(0.0).gt(0.5),
            _safe_numeric(frame, "url_negative_top15_target", 0.0).fillna(0.0).gt(0.5),
        )
    if task == "diagnostic_negative_top20":
        return (
            _safe_numeric(frame, "url_negative_top20_population", 0.0).fillna(0.0).gt(0.5),
            _safe_numeric(frame, "url_negative_top20_target", 0.0).fillna(0.0).gt(0.5),
        )
    if task in {"positive_mid30_not_top10", "diagnostic_positive_top30_not_top10"}:
        return (
            _safe_numeric(frame, "url_positive_mid30_population", 0.0).fillna(0.0).gt(0.5),
            _safe_numeric(frame, "url_positive_mid30_target", 0.0).fillna(0.0).gt(0.5),
        )
    raise ValueError(f"unknown economic relevance task: {task}")


def _is_negative_task(task: str) -> bool:
    return str(task).startswith("demote") or "negative" in str(task)


def _has_shock_entropy(name: str) -> bool:
    lower = str(name).lower()
    return "shock" in lower or "entropy" in lower


def _is_market_or_cross_sectional_context_feature(name: str) -> bool:
    lower = str(name).lower()
    if any(token in lower for token in ("symbol_minus_mkt", "asset_minus_mkt", "peer_resid", "mkt_resid", "ts_resid")):
        return False
    if _has_shock_entropy(lower):
        return is_market_context_shock_entropy_feature(lower)
    return any(
        token in lower
        for token in (
            "market",
            "mkt_",
            "xs_",
            "xs_dispersion",
            "cs_",
            "cross_asset",
            "crossasset",
            "cross_section",
            "xasset",
            "breadth",
            "dispersion",
            "state_spectral",
            "eig_",
            "basket",
            "factor",
            "funding_rate_cross_asset",
            "gmm",
            "reconstruction",
        )
    )


def _valid_pair_for_shock_entropy_context(feature_a: str, feature_b: str) -> bool:
    if not (_has_shock_entropy(feature_a) or _has_shock_entropy(feature_b)):
        return True
    return _is_market_or_cross_sectional_context_feature(feature_a) and _is_market_or_cross_sectional_context_feature(feature_b)


def _scoring_tasks(config: EconomicRegimeRelevanceConfig) -> tuple[str, ...]:
    return tuple(dict.fromkeys([*config.candidate_tasks, *config.diagnostic_tasks]))


def _state_score(
    *,
    task: str,
    state_rate: float,
    baseline_rate: float,
    state_ev: float,
    baseline_ev: float,
    temporal_score: float,
    config: EconomicRegimeRelevanceConfig,
) -> float:
    event_lift = state_rate - baseline_rate
    ev_delta = state_ev - baseline_ev
    if _is_negative_task(task):
        base = event_lift + float(max(-ev_delta, 0.0)) * float(config.negative_ev_weight)
    else:
        base = event_lift + float(max(ev_delta, 0.0)) * float(config.positive_ev_weight)
    return float(base + float(config.temporal_score_weight) * float(max(temporal_score, 0.0)))


def _longest_true_run(values: Sequence[bool]) -> int:
    best = 0
    current = 0
    for value in values:
        if bool(value):
            current += 1
            best = max(best, current)
        else:
            current = 0
    return int(best)


def _period_keys(frame: pd.DataFrame, config: EconomicRegimeRelevanceConfig) -> tuple[pd.Series, pd.Series]:
    if config.timestamp_col in frame.columns:
        ts = pd.to_datetime(frame[config.timestamp_col], utc=True, errors="coerce")
        day = ts.dt.floor("D").astype(str)
    else:
        ts = None
        day = pd.Series("", index=frame.index, dtype=object)
    if config.week_col in frame.columns:
        week = frame[config.week_col].astype(str)
    elif ts is not None:
        week = ts.dt.tz_convert(None).dt.to_period("W-MON").astype(str)
    else:
        week = pd.Series("", index=frame.index, dtype=object)
    return day, week


def _temporal_state_metrics(
    *,
    task: str,
    state_mask: pd.Series,
    population: pd.Series,
    target: pd.Series,
    ev: pd.Series,
    period_key: pd.Series,
    prefix: str,
    config: EconomicRegimeRelevanceConfig,
) -> dict[str, float]:
    mask = population & period_key.notna() & period_key.astype(str).ne("")
    if int(mask.sum()) < int(config.min_state_rows):
        return {
            f"{prefix}_aligned_periods": 0.0,
            f"{prefix}_state_share_lift": 0.0,
            f"{prefix}_state_score_corr": 0.0,
            f"{prefix}_aligned_streak": 0.0,
            f"{prefix}_temporal_score": 0.0,
        }
    tmp = pd.DataFrame(
        {
            "period": period_key.loc[mask].astype(str).to_numpy(),
            "state": state_mask.loc[mask].astype("float32").to_numpy(),
            "target": target.loc[mask].astype("float32").to_numpy(),
            "ev": ev.loc[mask].astype("float32").to_numpy(),
        }
    )
    period = tmp.groupby("period", sort=True).agg(
        rows=("target", "size"),
        state_share=("state", "mean"),
        target_rate=("target", "mean"),
        mean_ev=("ev", "mean"),
    )
    if len(period) < 3 or float(period["state_share"].max()) <= 0.0:
        return {
            f"{prefix}_aligned_periods": float(len(period)),
            f"{prefix}_state_share_lift": 0.0,
            f"{prefix}_state_score_corr": 0.0,
            f"{prefix}_aligned_streak": 0.0,
            f"{prefix}_temporal_score": 0.0,
        }
    if _is_negative_task(task):
        period_score = period["target_rate"] - period["mean_ev"] * float(config.negative_ev_weight)
    else:
        period_score = period["target_rate"] + period["mean_ev"] * float(config.positive_ev_weight)
    threshold = float(period_score.quantile(1.0 - float(config.temporal_tail_fraction)))
    aligned = period_score.ge(threshold)
    weights = period["rows"].astype("float64")
    base_share = float(np.average(period["state_share"], weights=weights))
    aligned_rows = int(aligned.sum())
    aligned_share = (
        float(np.average(period.loc[aligned, "state_share"], weights=weights.loc[aligned]))
        if aligned_rows
        else base_share
    )
    lift = aligned_share - base_share
    if period["state_share"].nunique(dropna=True) < 2 or period_score.nunique(dropna=True) < 2:
        corr = 0.0
    else:
        corr = float(period["state_share"].corr(period_score, method="spearman"))
        if not np.isfinite(corr):
            corr = 0.0
    streak = float(_longest_true_run((aligned & period["state_share"].ge(base_share)).tolist()))
    temporal_score = max(lift, 0.0) + max(corr, 0.0) * 0.25 + min(streak, 5.0) * 0.02
    return {
        f"{prefix}_aligned_periods": float(aligned_rows),
        f"{prefix}_state_share_lift": float(lift),
        f"{prefix}_state_score_corr": float(corr),
        f"{prefix}_aligned_streak": float(streak),
        f"{prefix}_temporal_score": float(temporal_score),
    }


def _metrics_row(
    *,
    task: str,
    side: str,
    archetype: str,
    state_kind: str,
    state_name: str,
    feature: str,
    feature_b: str,
    feature_bin: str,
    feature_b_bin: str,
    state_mask: pd.Series,
    population: pd.Series,
    target: pd.Series,
    ev: pd.Series,
    baseline_rate: float,
    baseline_ev: float,
    day_key: pd.Series,
    week_key: pd.Series,
    threshold: Mapping[str, Any] | None,
    threshold_b: Mapping[str, Any] | None,
    nonlinear_summary: Mapping[str, Any] | None,
    config: EconomicRegimeRelevanceConfig,
) -> dict[str, Any] | None:
    mask = population & state_mask
    rows = int(mask.sum())
    if rows < int(config.min_state_rows):
        return None
    target_rate = float(target.loc[mask].mean())
    mean_ev = float(ev.loc[mask].mean())
    day_metrics = _temporal_state_metrics(
        task=task,
        state_mask=state_mask,
        population=population,
        target=target,
        ev=ev,
        period_key=day_key,
        prefix="day",
        config=config,
    )
    week_metrics = _temporal_state_metrics(
        task=task,
        state_mask=state_mask,
        population=population,
        target=target,
        ev=ev,
        period_key=week_key,
        prefix="week",
        config=config,
    )
    temporal_score = float(day_metrics.get("day_temporal_score", 0.0)) + float(
        week_metrics.get("week_temporal_score", 0.0)
    )
    score = _state_score(
        task=task,
        state_rate=target_rate,
        baseline_rate=float(baseline_rate),
        state_ev=mean_ev,
        baseline_ev=float(baseline_ev),
        temporal_score=temporal_score,
        config=config,
    )
    row = {
        "task": task,
        "side_name": side,
        "archetype_policy_key": archetype,
        "state_kind": state_kind,
        "state_name": state_name,
        "feature": feature,
        "feature_b": feature_b,
        "feature_bin": feature_bin,
        "feature_b_bin": feature_b_bin,
        "rows": rows,
        "population_rows": int(population.sum()),
        "target_rate": target_rate,
        "baseline_target_rate": float(baseline_rate),
        "target_rate_lift": target_rate - float(baseline_rate),
        "mean_ev_after_1pct": mean_ev,
        "baseline_mean_ev_after_1pct": float(baseline_ev),
        "ev_delta": mean_ev - float(baseline_ev),
        "temporal_relevance_score": temporal_score,
        "economic_relevance_score": score,
        "q_low": float(threshold["q_low"]) if threshold else np.nan,
        "q_high": float(threshold["q_high"]) if threshold else np.nan,
        "q_low_b": float(threshold_b["q_low"]) if threshold_b else np.nan,
        "q_high_b": float(threshold_b["q_high"]) if threshold_b else np.nan,
    }
    row.update(
        nonlinear_summary
        or {
            "nonlinear_relevance_score": 0.0,
            "nonlinear_target_rate_span": 0.0,
            "nonlinear_ev_span": 0.0,
            "nonlinear_best_bin": "",
            "nonlinear_best_bin_rows": 0,
            "nonlinear_best_bin_target_rate": np.nan,
            "nonlinear_best_bin_ev": np.nan,
        }
    )
    row.update(day_metrics)
    row.update(week_metrics)
    return row


def score_side_archetype_economic_relevance(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    *,
    config: EconomicRegimeRelevanceConfig = EconomicRegimeRelevanceConfig(),
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    """Score univariate and pairwise state relevance per side x archetype."""

    work = add_global_topk_surprise_targets(frame, config=config)
    feature_columns = [str(c) for c in dict.fromkeys(feature_columns) if str(c) in work.columns]
    feature_rows: list[dict[str, Any]] = []
    composite_rows: list[dict[str, Any]] = []
    composite_defs: list[dict[str, Any]] = []
    ev = _safe_numeric(work, config.ev_col, 0.0).fillna(0.0)
    for (side, arch), group_idx in work.groupby([config.side_col, config.archetype_col], observed=True).groups.items():
        group = work.loc[group_idx]
        if len(group) < int(config.min_group_rows):
            continue
        thresholds = _feature_thresholds(group, feature_columns, config)
        if not thresholds:
            continue
        thresholds = thresholds[: int(config.max_features_per_group)]
        threshold_by_feature = {str(t["feature"]): t for t in thresholds}
        local_ev = ev.loc[group.index]
        day_key, week_key = _period_keys(group, config)
        for task in _scoring_tasks(config):
            population, target = _population_task(group, task)
            pop_rows = int(population.sum())
            if pop_rows < int(config.min_population_rows) or target.nunique() < 2:
                continue
            baseline_rate = float(target.loc[population].mean())
            baseline_ev = float(local_ev.loc[population].mean())
            nonlinear_by_feature = {
                str(threshold["feature"]): _nonlinear_feature_summary(
                    task=task,
                    values=group[str(threshold["feature"])],
                    population=population,
                    target=target,
                    ev=local_ev,
                    baseline_rate=baseline_rate,
                    baseline_ev=baseline_ev,
                    config=config,
                )
                for threshold in thresholds
            }
            for threshold in thresholds:
                feature = str(threshold["feature"])
                bins = _assign_bin(group[feature], threshold)
                for bin_name in ("low", "mid", "high", "missing"):
                    state_mask = bins.eq(bin_name)
                    row = _metrics_row(
                        task=task,
                        side=str(side),
                        archetype=str(arch),
                        state_kind="feature_bin",
                        state_name=f"{feature}={bin_name}",
                        feature=feature,
                        feature_b="",
                        feature_bin=bin_name,
                        feature_b_bin="",
                        state_mask=state_mask,
                        population=population,
                        target=target,
                        ev=local_ev,
                        baseline_rate=baseline_rate,
                        baseline_ev=baseline_ev,
                        day_key=day_key,
                        week_key=week_key,
                        threshold=threshold,
                        threshold_b=None,
                        nonlinear_summary=nonlinear_by_feature.get(feature),
                        config=config,
                    )
                    if row is not None:
                        feature_rows.append(row)

            local_feature_metrics = pd.DataFrame(
                [r for r in feature_rows if r["side_name"] == str(side) and r["archetype_policy_key"] == str(arch) and r["task"] == task]
            )
            if local_feature_metrics.empty:
                continue
            top_features = (
                local_feature_metrics.sort_values("economic_relevance_score", ascending=False)["feature"]
                .drop_duplicates()
                .head(int(config.max_features_for_composites))
                .tolist()
            )
            composite_count = 0
            for i, feature_a in enumerate(top_features):
                for feature_b in top_features[i + 1 :]:
                    if not _valid_pair_for_shock_entropy_context(str(feature_a), str(feature_b)):
                        continue
                    t_a = threshold_by_feature.get(str(feature_a))
                    t_b = threshold_by_feature.get(str(feature_b))
                    if t_a is None or t_b is None:
                        continue
                    bins_a = _assign_bin(group[str(feature_a)], t_a)
                    bins_b = _assign_bin(group[str(feature_b)], t_b)
                    for bin_a in ("low", "mid", "high"):
                        for bin_b in ("low", "mid", "high"):
                            state_mask = bins_a.eq(bin_a) & bins_b.eq(bin_b)
                            row = _metrics_row(
                                task=task,
                                side=str(side),
                                archetype=str(arch),
                                state_kind="pair_bin",
                                state_name=f"{feature_a}={bin_a} & {feature_b}={bin_b}",
                                feature=str(feature_a),
                                feature_b=str(feature_b),
                                feature_bin=bin_a,
                                feature_b_bin=bin_b,
                                state_mask=state_mask,
                                population=population,
                                target=target,
                                ev=local_ev,
                                baseline_rate=baseline_rate,
                                baseline_ev=baseline_ev,
                                day_key=day_key,
                                week_key=week_key,
                                threshold=t_a,
                                threshold_b=t_b,
                                nonlinear_summary=None,
                                config=config,
                            )
                            if row is not None:
                                composite_rows.append(row)
                                if row["economic_relevance_score"] >= float(config.min_candidate_score):
                                    composite_name = _composite_feature_name(row)
                                    composite_defs.append(
                                        {
                                            "name": composite_name,
                                            "task": task,
                                            "side_name": str(side),
                                            "archetype_policy_key": str(arch),
                                            "feature": str(feature_a),
                                            "feature_bin": bin_a,
                                            "q_low": float(t_a["q_low"]),
                                            "q_high": float(t_a["q_high"]),
                                            "feature_b": str(feature_b),
                                            "feature_b_bin": bin_b,
                                            "q_low_b": float(t_b["q_low"]),
                                            "q_high_b": float(t_b["q_high"]),
                                            "economic_relevance_score": float(row["economic_relevance_score"]),
                                        }
                                    )
                            composite_count += 1
                            if composite_count >= int(config.max_composites_per_group_task):
                                break
                        if composite_count >= int(config.max_composites_per_group_task):
                            break
                    if composite_count >= int(config.max_composites_per_group_task):
                        break
                if composite_count >= int(config.max_composites_per_group_task):
                    break
    return pd.DataFrame(feature_rows), pd.DataFrame(composite_rows), composite_defs


def _composite_feature_name(row: Mapping[str, Any]) -> str:
    def safe(value: Any) -> str:
        text = str(value).replace("/", "_").replace(":", "").replace(" ", "_")
        for ch in "=|&(),[]{}":
            text = text.replace(ch, "_")
        return "_".join(part for part in text.split("_") if part)

    return (
        f"url_cmp__{safe(row.get('task'))}__{safe(row.get('side_name'))}__"
        f"{safe(row.get('archetype_policy_key'))}__{safe(row.get('feature'))}_{safe(row.get('feature_bin'))}__"
        f"{safe(row.get('feature_b'))}_{safe(row.get('feature_b_bin'))}"
    )[:220]


def materialize_composite_features(
    frame: pd.DataFrame,
    composite_definitions: Sequence[Mapping[str, Any]],
    *,
    include_intensity: bool = True,
) -> pd.DataFrame:
    """Materialize binary and optional intensity composite features.

    The binary mask is useful for crisp state membership.  The intensity column
    preserves how deeply the row sits inside the same frozen two-feature state,
    which gives EBM/GAM calibrators more than a hard yes/no signal.
    """

    columns: dict[str, np.ndarray] = {}
    for definition in composite_definitions:
        name = str(definition.get("name") or "")
        feature = str(definition.get("feature") or "")
        feature_b = str(definition.get("feature_b") or "")
        if not name or feature not in frame.columns or feature_b not in frame.columns:
            continue
        bins_a = _assign_bin(frame[feature], definition)
        bins_b = _assign_bin(
            frame[feature_b],
            {
                "q_low": float(definition.get("q_low_b", np.nan)),
                "q_high": float(definition.get("q_high_b", np.nan)),
            },
        )
        bin_a = str(definition.get("feature_bin") or "")
        bin_b = str(definition.get("feature_b_bin") or "")
        match = bins_a.eq(bin_a) & bins_b.eq(bin_b)
        columns[name] = match.to_numpy(dtype=np.int8, copy=False)
        if include_intensity:
            depth_a = _bin_depth(frame[feature], definition, bin_a).to_numpy(dtype=np.float32, copy=False)
            depth_b = _bin_depth(
                frame[feature_b],
                {
                    "q_low": float(definition.get("q_low_b", np.nan)),
                    "q_high": float(definition.get("q_high_b", np.nan)),
                },
                bin_b,
            ).to_numpy(dtype=np.float32, copy=False)
            intensity = np.minimum(depth_a, depth_b)
            intensity = np.where(match.to_numpy(dtype=bool, copy=False), intensity, 0.0).astype(np.float32, copy=False)
            columns[f"{name}__intensity"] = intensity
    if not columns:
        return pd.DataFrame(index=frame.index)
    return pd.DataFrame(columns, index=frame.index)


def _matrix(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    cols = [str(c) for c in dict.fromkeys(columns) if str(c) in frame.columns]
    if not cols:
        return pd.DataFrame(index=frame.index)
    x = frame[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    med = x.median(axis=0, skipna=True).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return x.fillna(med).astype("float32", copy=False)


def _binary_auc(y: np.ndarray, score: np.ndarray) -> float:
    yy = np.asarray(y, dtype=np.int8).reshape(-1)
    ss = np.asarray(score, dtype=np.float64).reshape(-1)
    mask = np.isfinite(ss)
    yy = yy[mask]
    ss = ss[mask]
    pos = yy > 0
    n_pos = int(pos.sum())
    n_neg = int(yy.size - n_pos)
    if yy.size < 3 or n_pos == 0 or n_neg == 0:
        return 0.5
    order = np.argsort(ss, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, yy.size + 1, dtype=np.float64)
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / max(n_pos * n_neg, 1))


def train_local_lgbm_relevance_models(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    *,
    config: EconomicRegimeRelevanceConfig = EconomicRegimeRelevanceConfig(),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train separate LGBM relevance models per side x archetype x task."""

    if not bool(config.lgbm_enabled):
        return pd.DataFrame(), pd.DataFrame()
    lgb = _lightgbm_module()
    if lgb is None:
        return pd.DataFrame(), pd.DataFrame(
            [{"status": "lightgbm_unavailable", "task": "", "side_name": "", "archetype_policy_key": ""}]
        )
    work = add_global_topk_surprise_targets(frame, config=config)
    feature_columns = [str(c) for c in dict.fromkeys(feature_columns) if str(c) in work.columns]
    feature_rows: list[dict[str, Any]] = []
    model_rows: list[dict[str, Any]] = []
    for (side, arch), group_idx in work.groupby([config.side_col, config.archetype_col], observed=True).groups.items():
        group = work.loc[group_idx].sort_values(config.timestamp_col if config.timestamp_col in work.columns else config.score_col)
        if len(group) < int(config.min_group_rows):
            continue
        cols = feature_columns[: int(config.lgbm_max_features)]
        for task in _scoring_tasks(config):
            population, target = _population_task(group, task)
            sub = group.loc[population].copy()
            y = target.loc[population].astype("int8").to_numpy()
            if len(sub) < int(config.lgbm_min_rows) or np.unique(y).size < 2:
                continue
            split = int(max(10, round(len(sub) * (1.0 - float(config.lgbm_validation_fraction)))))
            split = min(max(split, 10), len(sub) - 5)
            x = _matrix(sub, cols)
            x_train = x.iloc[:split]
            y_train = y[:split]
            x_val = x.iloc[split:]
            y_val = y[split:]
            if np.unique(y_train).size < 2 or np.unique(y_val).size < 2:
                continue
            model = lgb.LGBMClassifier(
                objective="binary",
                n_estimators=int(config.lgbm_n_estimators),
                learning_rate=float(config.lgbm_learning_rate),
                num_leaves=int(config.lgbm_num_leaves),
                max_depth=int(config.lgbm_max_depth),
                min_child_samples=int(config.lgbm_min_child_samples),
                subsample=0.85,
                colsample_bytree=0.85,
                random_state=int(config.random_state),
                n_jobs=1,
                verbosity=-1,
            )
            model.fit(x_train, y_train)
            proba = model.predict_proba(x_val)[:, 1]
            top = proba >= np.nanquantile(proba, 0.85)
            baseline = float(np.mean(y_val))
            precision = float(np.mean(y_val[top])) if int(top.sum()) else np.nan
            auc = _binary_auc(y_val, proba)
            model_rows.append(
                {
                    "task": task,
                    "side_name": str(side),
                    "archetype_policy_key": str(arch),
                    "rows": int(len(sub)),
                    "train_rows": int(len(x_train)),
                    "validation_rows": int(len(x_val)),
                    "baseline_target_rate": baseline,
                    "top15_model_precision": precision,
                    "top15_model_lift": precision - baseline if np.isfinite(precision) else np.nan,
                    "auc": auc,
                }
            )
            booster = model.booster_
            gain = booster.feature_importance(importance_type="gain")
            split_imp = booster.feature_importance(importance_type="split")
            for feature, gain_value, split_value in zip(x.columns, gain, split_imp):
                if float(gain_value) <= 0.0 and int(split_value) <= 0:
                    continue
                feature_rows.append(
                    {
                        "task": task,
                        "side_name": str(side),
                        "archetype_policy_key": str(arch),
                        "feature": str(feature),
                        "lgbm_gain": float(gain_value),
                        "lgbm_split": int(split_value),
                        "model_auc": auc,
                        "model_top15_lift": precision - baseline if np.isfinite(precision) else np.nan,
                    }
                )
    return pd.DataFrame(feature_rows), pd.DataFrame(model_rows)


def build_ebm_candidate_manifest(
    *,
    feature_metrics: pd.DataFrame,
    composite_metrics: pd.DataFrame,
    lgbm_feature_metrics: pd.DataFrame,
    composite_definitions: Sequence[Mapping[str, Any]],
    config: EconomicRegimeRelevanceConfig = EconomicRegimeRelevanceConfig(),
    max_features_per_side_archetype: int = 12,
) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    for source_name, df in (
        ("feature_bin_relevance", feature_metrics),
        ("pair_composite_relevance", composite_metrics),
    ):
        if df is None or df.empty:
            continue
        cur = df.copy()
        cur["effective_candidate_score"] = pd.to_numeric(cur["economic_relevance_score"], errors="coerce")
        if source_name == "feature_bin_relevance" and "nonlinear_relevance_score" in cur.columns:
            cur["effective_candidate_score"] = np.maximum(
                cur["effective_candidate_score"].fillna(0.0),
                pd.to_numeric(cur["nonlinear_relevance_score"], errors="coerce").fillna(0.0),
            )
        cur = cur.loc[cur["effective_candidate_score"].ge(float(config.min_candidate_score))]
        cur = cur.loc[cur["task"].astype(str).isin(set(config.candidate_tasks))]
        for _, row in cur.iterrows():
            feature_name = str(row.get("feature") or "")
            source = source_name
            if source_name == "pair_composite_relevance":
                base_name = _composite_feature_name(row)
                feature_names = [f"{base_name}__intensity", base_name] if bool(config.materialize_composite_intensity) else [base_name]
            else:
                feature_names = [feature_name]
            for feature_name in feature_names:
                if source_name == "pair_composite_relevance" and feature_name.endswith("__intensity"):
                    source = "pair_composite_intensity"
                else:
                    source = source_name
                score = float(row.get("effective_candidate_score", row.get("economic_relevance_score", np.nan)))
                if source == "pair_composite_intensity":
                    score += 1e-9
                candidates.append(
                    {
                        "source": source,
                        "task": str(row.get("task") or ""),
                        "side_name": str(row.get("side_name") or ""),
                        "archetype_policy_key": str(row.get("archetype_policy_key") or ""),
                        "feature": feature_name,
                        "economic_relevance_score": score,
                        "bin_economic_relevance_score": float(row.get("economic_relevance_score", np.nan)),
                        "nonlinear_relevance_score": float(row.get("nonlinear_relevance_score", np.nan)),
                        "target_rate_lift": float(row.get("target_rate_lift", np.nan)),
                        "ev_delta": float(row.get("ev_delta", np.nan)),
                    }
                )
    if lgbm_feature_metrics is not None and not lgbm_feature_metrics.empty:
        ranked = lgbm_feature_metrics.sort_values(["model_top15_lift", "lgbm_gain"], ascending=False)
        for _, row in ranked.iterrows():
            if str(row.get("task") or "") not in set(config.candidate_tasks):
                continue
            candidates.append(
                {
                    "source": "local_lgbm",
                    "task": str(row.get("task") or ""),
                    "side_name": str(row.get("side_name") or ""),
                    "archetype_policy_key": str(row.get("archetype_policy_key") or ""),
                    "feature": str(row.get("feature") or ""),
                    "economic_relevance_score": float(row.get("model_top15_lift", np.nan)),
                    "lgbm_gain": float(row.get("lgbm_gain", np.nan)),
                    "model_auc": float(row.get("model_auc", np.nan)),
                }
            )
    candidates_df = pd.DataFrame(candidates)
    feature_map: dict[str, list[str]] = {}
    if not candidates_df.empty:
        candidates_df = candidates_df.sort_values(
            ["side_name", "archetype_policy_key", "economic_relevance_score"],
            ascending=[True, True, False],
            kind="mergesort",
        )
        for (side, arch), group in candidates_df.groupby(["side_name", "archetype_policy_key"], observed=True):
            features = [str(f) for f in group["feature"].dropna().drop_duplicates().head(int(max_features_per_side_archetype))]
            feature_map[f"{side}|{arch}"] = features
    return {
        "artifact_type": "unsupervised_regime_learning_economic_ebm_candidate_features",
        "consumer": "scripts/run_regime_calibration_model_ablation.py",
        "calibration_policy_id": "per_regime_archetype_calibration_v1",
        "intended_policy_reference": "ev_target_archetype_reachable_match_current_activity_8d_hr_off_regimecal_v1",
        "topk_contract": {
            "demote": "global top10% by score; target is bad/dirty/non-clean/negative EV top10",
            "promote": "global top20% excluding global top10%; target is clean positive EV near-threshold",
            "diagnostics": "global top15/top20 demotion and top30-not-top10 promotion are reported but not selected by default",
            "locality": "feature scoring and LGBM models are per side x archetype",
            "temporal_alignment": "feature states are additionally scored by best/worst day/week alignment and aligned streaks per side x archetype",
        },
        "feature_map": feature_map,
        "composite_definitions": list(composite_definitions),
        "config": asdict(config),
    }


def run_economic_regime_relevance(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    *,
    config: EconomicRegimeRelevanceConfig = EconomicRegimeRelevanceConfig(),
) -> EconomicRegimeRelevanceResult:
    feature_metrics, composite_metrics, composite_definitions = score_side_archetype_economic_relevance(
        frame,
        feature_columns,
        config=config,
    )
    composite_frame = materialize_composite_features(
        frame,
        composite_definitions,
        include_intensity=bool(config.materialize_composite_intensity),
    )
    model_frame = pd.concat([frame.reset_index(drop=True), composite_frame.reset_index(drop=True)], axis=1)
    lgbm_features = [*feature_columns, *list(composite_frame.columns)]
    lgbm_feature_metrics, lgbm_model_metrics = train_local_lgbm_relevance_models(
        model_frame,
        lgbm_features,
        config=config,
    )
    manifest = build_ebm_candidate_manifest(
        feature_metrics=feature_metrics,
        composite_metrics=composite_metrics,
        lgbm_feature_metrics=lgbm_feature_metrics,
        composite_definitions=composite_definitions,
        config=config,
    )
    selected_rows: list[dict[str, Any]] = []
    for key, features in (manifest.get("feature_map") or {}).items():
        side, arch = str(key).split("|", 1)
        for feature in features:
            selected_rows.append({"side_name": side, "archetype_policy_key": arch, "feature": feature})
    return EconomicRegimeRelevanceResult(
        feature_metrics=feature_metrics,
        composite_metrics=composite_metrics,
        lgbm_feature_metrics=lgbm_feature_metrics,
        lgbm_model_metrics=lgbm_model_metrics,
        selected_candidates=pd.DataFrame(selected_rows),
        composite_definitions=composite_definitions,
        ebm_candidate_manifest=manifest,
    )
