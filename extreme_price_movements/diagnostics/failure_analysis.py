"""Retrospective failure-day and failure-episode diagnostics.

This module deliberately consumes realized outcomes.  Its tables describe what
happened in an already completed ledger; they are not signals, gates, model
features, or evidence that removing a condition would have caused a different
outcome.  In particular, the counterfactual table is a leave-one-condition-out
accounting decomposition, not a policy recommendation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FailureAnalysisConfig:
    """Column mapping and thresholds for retrospective failure diagnostics.

    ``percentile_reference`` selects the flag used in ``failure_day``.  Both
    pooled and prior-only causal percentile flags remain in the daily output so
    a report can disclose the choice.  A drawdown threshold is expressed in the
    same units as cumulative ``bankroll_pnl``; leave it as ``None`` to disable
    that failure condition.
    """

    timestamp_col: str = "timestamp"
    expected_pnl_col: str = "expected_pnl"
    realized_pnl_col: str = "realized_pnl"
    bankroll_pnl_col: str = "bankroll_pnl"
    base_model_col: str = "base_model"
    side_col: str = "side"
    setup_col: str = "setup"
    horizon_col: str = "horizon"
    symbol_col: str = "symbol"
    regime_col: str = "regime"
    market_state_cols: tuple[str, ...] = ()
    residual_z_window_days: int = 20
    residual_z_min_periods: int = 10
    residual_z_threshold: float = -2.0
    residual_percentile: float = 0.05
    residual_percentile_min_days: int = 20
    percentile_reference: Literal["pooled", "causal"] = "causal"
    bankroll_drawdown_threshold: float | None = None
    sequence_window_trades: int = 30
    sequence_group_cols: tuple[str, ...] = ()
    comparison_hours: tuple[int, ...] = (6, 12, 24)
    similarity_min_history_episodes: int = 1


@dataclass(frozen=True)
class FailureAnalysisResult:
    """All descriptive tables produced by :func:`analyze_failure_diagnostics`."""

    daily: pd.DataFrame
    episodes: pd.DataFrame
    episode_comparisons: pd.DataFrame
    monthly_performance: pd.DataFrame
    counterfactual_removals: pd.DataFrame
    sequence_metrics: pd.DataFrame
    episode_similarity: pd.DataFrame
    manifest: dict[str, Any]


_TIMESTAMP_FALLBACKS = ("__ts__", "ts", "datetime")
_EXPECTED_FALLBACKS = ("ev_after_1pct", "expected_value", "expected_ev", "ev")
_REALIZED_FALLBACKS = ("pnl", "net_pnl", "realized_return", "return")
_BANKROLL_FALLBACKS = ("portfolio_pnl", "sized_pnl", "pnl")


def _resolve_column(frame: pd.DataFrame, requested: str, fallbacks: Sequence[str]) -> str:
    for name in (requested, *fallbacks):
        if name in frame:
            return name
    raise KeyError(
        f"Failure analysis requires {requested!r}; tried {list((requested, *fallbacks))}"
    )


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan)


def _dimension(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series("__all__", index=frame.index, dtype="object")
    return frame[column].where(frame[column].notna(), "__missing__").astype(str)


def _empty(columns: Sequence[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=list(columns))


def _prepare_ledger(
    ledger: pd.DataFrame, config: FailureAnalysisConfig
) -> tuple[pd.DataFrame, dict[str, str]]:
    timestamp_col = _resolve_column(ledger, config.timestamp_col, _TIMESTAMP_FALLBACKS)
    expected_col = _resolve_column(ledger, config.expected_pnl_col, _EXPECTED_FALLBACKS)
    realized_col = _resolve_column(ledger, config.realized_pnl_col, _REALIZED_FALLBACKS)
    bankroll_col = _resolve_column(
        ledger, config.bankroll_pnl_col, _BANKROLL_FALLBACKS
    ) if config.bankroll_pnl_col in ledger or any(
        name in ledger for name in _BANKROLL_FALLBACKS
    ) else realized_col

    work = ledger.copy()
    work["_timestamp"] = pd.to_datetime(work[timestamp_col], utc=True, errors="coerce")
    work = work.loc[work["_timestamp"].notna()].copy()
    work["_source_order"] = np.arange(len(work), dtype=np.int64)
    work["_expected_pnl"] = _numeric(work, expected_col).fillna(0.0)
    work["_realized_pnl"] = _numeric(work, realized_col).fillna(0.0)
    work["_bankroll_pnl"] = _numeric(work, bankroll_col).fillna(work["_realized_pnl"])
    work["_residual"] = work["_realized_pnl"] - work["_expected_pnl"]
    work["_day"] = work["_timestamp"].dt.floor("D")
    return work.sort_values(["_timestamp", "_source_order"], kind="stable"), {
        "timestamp": timestamp_col,
        "expected_pnl": expected_col,
        "realized_pnl": realized_col,
        "bankroll_pnl": bankroll_col,
    }


def _daily_failure_flags(work: pd.DataFrame, config: FailureAnalysisConfig) -> pd.DataFrame:
    if work.empty:
        return _empty(
            (
                "day", "support", "expected_pnl", "realized_pnl", "residual",
                "residual_z", "pooled_residual_p05", "causal_residual_p05",
                "failure_day",
            )
        )
    daily = (
        work.groupby("_day", observed=True, sort=True)
        .agg(
            support=("_residual", "size"),
            expected_pnl=("_expected_pnl", "sum"),
            realized_pnl=("_realized_pnl", "sum"),
            bankroll_pnl=("_bankroll_pnl", "sum"),
        )
        .rename_axis("day")
        .reset_index()
    )
    daily["residual"] = daily["realized_pnl"] - daily["expected_pnl"]
    prior = daily["residual"].shift(1)
    rolling = prior.rolling(
        window=max(1, int(config.residual_z_window_days)),
        min_periods=max(1, int(config.residual_z_min_periods)),
    )
    mean = rolling.mean()
    std = rolling.std(ddof=0).replace(0.0, np.nan)
    daily["residual_z"] = (daily["residual"] - mean) / std
    daily["residual_z_failure"] = daily["residual_z"].le(config.residual_z_threshold).fillna(False)

    pooled_p05 = float(daily["residual"].quantile(config.residual_percentile))
    daily["pooled_residual_p05"] = pooled_p05
    daily["pooled_residual_p05_failure"] = daily["residual"].le(pooled_p05)
    causal_p05 = np.full(len(daily), np.nan, dtype=np.float64)
    values = daily["residual"].to_numpy(dtype=np.float64)
    for position in range(len(values)):
        history = values[:position]
        history = history[np.isfinite(history)]
        if len(history) >= int(config.residual_percentile_min_days):
            causal_p05[position] = float(np.quantile(history, config.residual_percentile))
    daily["causal_residual_p05"] = causal_p05
    daily["causal_residual_p05_failure"] = (
        daily["residual"].le(daily["causal_residual_p05"]).fillna(False)
    )

    bankroll_curve = daily["bankroll_pnl"].cumsum()
    daily["bankroll_cumulative_pnl"] = bankroll_curve
    daily["bankroll_drawdown"] = bankroll_curve.cummax() - bankroll_curve
    if config.bankroll_drawdown_threshold is None:
        daily["bankroll_drawdown_failure"] = False
    else:
        daily["bankroll_drawdown_failure"] = daily["bankroll_drawdown"].ge(
            float(config.bankroll_drawdown_threshold)
        )
    selected_percentile = f"{config.percentile_reference}_residual_p05_failure"
    daily["residual_percentile_failure"] = daily[selected_percentile]
    daily["failure_day"] = (
        daily["residual_z_failure"]
        | daily["residual_percentile_failure"]
        | daily["bankroll_drawdown_failure"]
    )
    daily["failure_reasons"] = daily.apply(
        lambda row: "|".join(
            name
            for name, active in (
                ("residual_z", row["residual_z_failure"]),
                (f"{config.percentile_reference}_residual_p05", row["residual_percentile_failure"]),
                ("bankroll_drawdown", row["bankroll_drawdown_failure"]),
            )
            if bool(active)
        ),
        axis=1,
    )
    return daily


def _episodes_from_daily(daily: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    columns = (
        "episode_id", "start_day", "end_day", "duration_days", "failure_days",
        "support", "expected_pnl", "realized_pnl", "residual", "worst_residual",
        "max_bankroll_drawdown", "failure_reasons",
    )
    membership_columns = ("day", "episode_id")
    active = daily.loc[daily["failure_day"]].copy()
    if active.empty:
        return _empty(columns), _empty(membership_columns)
    active["_new_episode"] = active["day"].diff().gt(pd.Timedelta(days=1)).fillna(True)
    active["_episode_number"] = active["_new_episode"].cumsum().astype(int)
    membership = active[["day", "_episode_number"]].rename(columns={"_episode_number": "episode_id"})
    rows: list[dict[str, Any]] = []
    for episode_id, group in active.groupby("_episode_number", sort=True):
        reasons = sorted(
            {
                reason
                for value in group["failure_reasons"]
                for reason in str(value).split("|")
                if reason
            }
        )
        rows.append(
            {
                "episode_id": int(episode_id),
                "start_day": group["day"].iloc[0],
                "end_day": group["day"].iloc[-1],
                "duration_days": int((group["day"].iloc[-1] - group["day"].iloc[0]).days + 1),
                "failure_days": int(len(group)),
                "support": int(group["support"].sum()),
                "expected_pnl": float(group["expected_pnl"].sum()),
                "realized_pnl": float(group["realized_pnl"].sum()),
                "residual": float(group["residual"].sum()),
                "worst_residual": float(group["residual"].min()),
                "max_bankroll_drawdown": float(group["bankroll_drawdown"].max()),
                "failure_reasons": "|".join(reasons),
            }
        )
    return pd.DataFrame(rows, columns=columns), membership


def _episode_comparisons(
    work: pd.DataFrame, episodes: pd.DataFrame, config: FailureAnalysisConfig
) -> pd.DataFrame:
    columns = (
        "episode_id", "lookback_hours", "pre_support", "during_support",
        "pre_expected_pnl", "during_expected_pnl", "expected_pnl_change",
        "pre_realized_pnl", "during_realized_pnl", "realized_pnl_change",
        "pre_residual", "during_residual", "residual_change",
    )
    rows: list[dict[str, Any]] = []
    for episode in episodes.itertuples(index=False):
        during_start = pd.Timestamp(episode.start_day)
        during_end = pd.Timestamp(episode.end_day) + pd.Timedelta(days=1)
        during = work.loc[work["_timestamp"].ge(during_start) & work["_timestamp"].lt(during_end)]
        for hours in config.comparison_hours:
            pre = work.loc[
                work["_timestamp"].ge(during_start - pd.Timedelta(hours=int(hours)))
                & work["_timestamp"].lt(during_start)
            ]
            def average(frame: pd.DataFrame, column: str) -> float:
                return float(frame[column].mean()) if len(frame) else np.nan
            pre_expected, during_expected = average(pre, "_expected_pnl"), average(during, "_expected_pnl")
            pre_realized, during_realized = average(pre, "_realized_pnl"), average(during, "_realized_pnl")
            pre_residual, during_residual = average(pre, "_residual"), average(during, "_residual")
            rows.append(
                {
                    "episode_id": int(episode.episode_id), "lookback_hours": int(hours),
                    "pre_support": int(len(pre)), "during_support": int(len(during)),
                    "pre_expected_pnl": pre_expected, "during_expected_pnl": during_expected,
                    "expected_pnl_change": during_expected - pre_expected,
                    "pre_realized_pnl": pre_realized, "during_realized_pnl": during_realized,
                    "realized_pnl_change": during_realized - pre_realized,
                    "pre_residual": pre_residual, "during_residual": during_residual,
                    "residual_change": during_residual - pre_residual,
                }
            )
    return pd.DataFrame(rows, columns=columns)


def _monthly_performance(work: pd.DataFrame, config: FailureAnalysisConfig) -> pd.DataFrame:
    dimensions = (config.base_model_col, config.side_col, config.setup_col, config.horizon_col)
    source = work.copy()
    for dimension in dimensions:
        source[dimension] = _dimension(source, dimension)
    source["month"] = source["_timestamp"].dt.strftime("%Y-%m")
    return (
        source.groupby(["month", *dimensions], observed=True, dropna=False)
        .agg(
            support=("_realized_pnl", "size"), expected_pnl=("_expected_pnl", "sum"),
            realized_pnl=("_realized_pnl", "sum"), residual=("_residual", "sum"),
            mean_realized_pnl=("_realized_pnl", "mean"), mean_residual=("_residual", "mean"),
            positive_realized_rate=("_realized_pnl", lambda value: float(value.gt(0.0).mean())),
        )
        .reset_index()
        .sort_values(["month", *dimensions], kind="stable")
        .reset_index(drop=True)
    )


def _counterfactual_removals(work: pd.DataFrame, config: FailureAnalysisConfig) -> pd.DataFrame:
    """Return descriptive one-condition removals with no causal interpretation."""

    dimensions = (
        config.side_col, config.setup_col, config.base_model_col,
        "hour_utc", config.symbol_col, config.regime_col,
    )
    source = work.copy()
    source["hour_utc"] = source["_timestamp"].dt.hour.astype(int)
    baseline_support = int(len(source))
    baseline_pnl = float(source["_bankroll_pnl"].sum())
    rows: list[dict[str, Any]] = []
    for dimension in dimensions:
        source[dimension] = _dimension(source, dimension)
        for value, group in source.groupby(dimension, observed=True, dropna=False, sort=True):
            removed_pnl = float(group["_bankroll_pnl"].sum())
            removed_support = int(len(group))
            rows.append(
                {
                    "condition": dimension, "condition_value": str(value),
                    "baseline_pnl": baseline_pnl, "baseline_support": baseline_support,
                    "removed_pnl": removed_pnl, "removed_support": removed_support,
                    "remaining_pnl": baseline_pnl - removed_pnl,
                    "remaining_support": baseline_support - removed_support,
                    "delta_pnl": -removed_pnl,
                    "diagnostic_only": True,
                }
            )
    return pd.DataFrame(rows)


def _sequence_metrics(work: pd.DataFrame, config: FailureAnalysisConfig) -> pd.DataFrame:
    """Attach outcome metrics computed from strictly earlier trades only."""

    source = work.copy()
    group_columns = [column for column in config.sequence_group_cols if column in source]
    if group_columns:
        grouped: Any = source.groupby(group_columns, observed=True, dropna=False, sort=False)
        pieces = [
            _sequence_metrics_one(group, config.sequence_window_trades)
            for _, group in grouped
        ]
        return pd.concat(pieces, axis=0).sort_values("_source_order", kind="stable")
    return _sequence_metrics_one(source, config.sequence_window_trades)


def _sequence_metrics_one(source: pd.DataFrame, window: int) -> pd.DataFrame:
    out = source.sort_values(["_timestamp", "_source_order"], kind="stable").copy()
    prior_pnl = out["_realized_pnl"].shift(1)
    prior_residual = out["_residual"].shift(1)
    rolling = prior_pnl.rolling(max(1, int(window)), min_periods=1)
    previous_count = (
        prior_pnl.notna()
        .rolling(max(1, int(window)), min_periods=1)
        .sum()
        .fillna(0)
        .astype(int)
    )
    out["previous_trade_count"] = previous_count
    out["previous_trade_pnl_sum"] = rolling.sum()
    out["previous_trade_pnl_mean"] = rolling.mean()
    out["previous_trade_win_rate"] = (
        prior_pnl.gt(0.0)
        .where(prior_pnl.notna())
        .rolling(max(1, int(window)), min_periods=1)
        .mean()
        .where(previous_count.gt(0))
    )
    out["previous_trade_residual_mean"] = prior_residual.rolling(max(1, int(window)), min_periods=1).mean()
    streak = 0
    previous_loss_streak: list[int] = []
    for value in out["_realized_pnl"].to_numpy(dtype=float):
        previous_loss_streak.append(streak)
        streak = streak + 1 if np.isfinite(value) and value < 0.0 else 0
    out["previous_trade_loss_streak"] = previous_loss_streak
    return out


def _episode_similarity(
    work: pd.DataFrame,
    episodes: pd.DataFrame,
    config: FailureAnalysisConfig,
) -> pd.DataFrame:
    columns = (
        "episode_id", "historical_similarity_available", "historical_support_episodes",
        "historical_nearest_episode_id", "historical_max_similarity",
    )
    state_columns = [column for column in config.market_state_cols if column in work]
    if episodes.empty or not state_columns:
        return _empty([*columns, *state_columns])
    summaries: list[dict[str, Any]] = []
    for episode in episodes.sort_values("start_day", kind="stable").itertuples(index=False):
        start, end = pd.Timestamp(episode.start_day), pd.Timestamp(episode.end_day) + pd.Timedelta(days=1)
        during = work.loc[work["_timestamp"].ge(start) & work["_timestamp"].lt(end)]
        row: dict[str, Any] = {"episode_id": int(episode.episode_id)}
        for column in state_columns:
            row[column] = float(pd.to_numeric(during[column], errors="coerce").mean())
        summaries.append(row)
    states = pd.DataFrame(summaries).merge(
        episodes[["episode_id", "start_day"]], on="episode_id", how="left"
    ).sort_values("start_day", kind="stable")
    output: list[dict[str, Any]] = []
    for position, current in states.reset_index(drop=True).iterrows():
        historical = states.iloc[:position]
        current_values = current[state_columns].to_numpy(dtype=float)
        scores: list[tuple[int, float]] = []
        for _, candidate in historical.iterrows():
            candidate_values = candidate[state_columns].to_numpy(dtype=float)
            valid = np.isfinite(current_values) & np.isfinite(candidate_values)
            if not valid.any():
                continue
            left, right = current_values[valid], candidate_values[valid]
            denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
            if denominator > 0.0:
                scores.append((int(candidate["episode_id"]), float(np.dot(left, right) / denominator)))
        available = len(historical) >= int(config.similarity_min_history_episodes) and bool(scores)
        nearest_id, similarity = (max(scores, key=lambda item: item[1]) if available else (np.nan, np.nan))
        row = {
            "episode_id": int(current["episode_id"]),
            "historical_similarity_available": bool(available),
            "historical_support_episodes": int(len(historical)),
            "historical_nearest_episode_id": nearest_id,
            "historical_max_similarity": similarity,
        }
        row.update({column: current[column] for column in state_columns})
        output.append(row)
    return pd.DataFrame(output, columns=[*columns, *state_columns])


def analyze_failure_diagnostics(
    ledger: pd.DataFrame,
    config: FailureAnalysisConfig | None = None,
) -> FailureAnalysisResult:
    """Build descriptive failure diagnostics from a completed trade ledger.

    The first two daily failure tests are residual-based: a daily realized minus
    expected PnL residual is compared with *previous* daily residual history,
    and with a fifth-percentile threshold.  The pooled threshold is intentionally
    ex-post; the causal version only sees earlier days.  Daily failures are
    merged when their UTC dates overlap or are adjacent.
    """

    cfg = config or FailureAnalysisConfig()
    if cfg.percentile_reference not in {"pooled", "causal"}:
        raise ValueError("percentile_reference must be 'pooled' or 'causal'")
    work, resolved = _prepare_ledger(ledger, cfg)
    daily = _daily_failure_flags(work, cfg)
    episodes, membership = _episodes_from_daily(daily)
    if not membership.empty:
        daily = daily.merge(membership, on="day", how="left")
    else:
        daily["episode_id"] = np.nan
    comparisons = _episode_comparisons(work, episodes, cfg)
    monthly = _monthly_performance(work, cfg)
    removals = _counterfactual_removals(work, cfg)
    sequence = _sequence_metrics(work, cfg)
    similarity = _episode_similarity(work, episodes, cfg)
    manifest = {
        "schema": "failure_analysis_diagnostics_v1",
        "descriptive_only": True,
        "noncausal": True,
        "inference_eligible": False,
        "resolved_columns": resolved,
        "config": asdict(cfg),
        "failure_day_rule": (
            "residual_z <= threshold OR selected residual fifth-percentile flag "
            "OR configured bankroll drawdown threshold"
        ),
        "sequence_rule": "Each sequence metric excludes the current trade outcome.",
        "similarity_rule": "Each episode compares only with earlier episodes.",
        "counterfactual_rule": "One condition is removed at a time for accounting only; no causal claim.",
    }
    return FailureAnalysisResult(
        daily=daily,
        episodes=episodes,
        episode_comparisons=comparisons,
        monthly_performance=monthly,
        counterfactual_removals=removals,
        sequence_metrics=sequence,
        episode_similarity=similarity,
        manifest=manifest,
    )


# A compact alias for callers that use the module name as the operation name.
build_failure_analysis = analyze_failure_diagnostics
