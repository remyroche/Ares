"""Pure, read-only diagnostics for normalized trade and prediction frames.

No function in this module fits a model, chooses a threshold, or mutates its
input.  Returns must already include costs exactly once.  By default return
columns are assumed side-relative; pass ``returns_are_side_relative=False``
when a return column is an underlying-price return and a ``side`` column is
available.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
import pandas as pd

_EPSILON = 1e-12
_RETURN_ALIASES = {
    "gross": ("gross_return", "gross_ret", "return_gross"),
    "net": ("net_return", "net_ret", "return_net", "net_ev", "pnl_net"),
    "bankroll": ("bankroll_pnl", "portfolio_pnl", "sized_pnl"),
    "mfe": ("mfe", "max_favorable_return", "max_favorable_return_until_exit"),
    "mae": ("mae", "max_adverse_return", "max_adverse_return_until_exit"),
    "holding": ("holding_time", "holding_hours", "exit_hours", "exit_bars"),
}


def _empty_decomposition() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "date",
            "trade_count",
            "gross_return_sum",
            "gross_return_mean",
            "net_return_sum",
            "net_return_mean",
            "bankroll_pnl_sum",
            "win_rate",
            "avg_win",
            "avg_loss",
            "payoff",
            "profit_factor",
            "mfe_mean",
            "mfe_median",
            "mae_mean",
            "mae_median",
            "holding_mean",
            "holding_median",
            "daily_sharpe",
            "annualized_sharpe",
            "daily_sortino",
            "annualized_sortino",
        ]
    )


def _resolve_column(
    frame: pd.DataFrame,
    requested: str | None,
    aliases: Iterable[str] = (),
    *,
    required: bool = False,
) -> str | None:
    candidates = [requested] if requested else []
    candidates.extend(alias for alias in aliases if alias not in candidates)
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    if required:
        wanted = requested or next(iter(aliases), "column")
        raise KeyError(f"Missing required column: {wanted}")
    return None


def _numeric(frame: pd.DataFrame, column: str | None) -> np.ndarray:
    if column is None:
        return np.full(len(frame), np.nan, dtype=np.float64)
    return pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=np.float64, copy=False)


def _finite(values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=np.float64)[np.isfinite(values)]


def _mean(values: np.ndarray) -> float:
    finite = _finite(values)
    return float(np.mean(finite)) if finite.size else float("nan")


def _sum(values: np.ndarray) -> float:
    finite = _finite(values)
    return float(np.sum(finite)) if finite.size else float("nan")


def _median(values: np.ndarray) -> float:
    finite = _finite(values)
    return float(np.median(finite)) if finite.size else float("nan")


def _safe_ratio(numerator: float, denominator: float) -> float:
    if not np.isfinite(numerator) or not np.isfinite(denominator) or abs(denominator) <= _EPSILON:
        return float("nan")
    return float(numerator / denominator)


def _risk_ratios(daily_returns: np.ndarray, annualization_days: float) -> dict[str, float]:
    values = _finite(daily_returns)
    if values.size < 2:
        return {
            "daily_sharpe": float("nan"),
            "annualized_sharpe": float("nan"),
            "daily_sortino": float("nan"),
            "annualized_sortino": float("nan"),
        }
    mean_return = float(np.mean(values))
    standard_deviation = float(np.std(values, ddof=1))
    daily_sharpe = _safe_ratio(mean_return, standard_deviation)
    downside = np.minimum(values, 0.0)
    downside_deviation = float(np.sqrt(np.mean(np.square(downside))))
    daily_sortino = _safe_ratio(mean_return, downside_deviation)
    scale = float(np.sqrt(annualization_days)) if annualization_days > 0.0 else float("nan")
    return {
        "daily_sharpe": daily_sharpe,
        "annualized_sharpe": daily_sharpe * scale if np.isfinite(daily_sharpe) else float("nan"),
        "daily_sortino": daily_sortino,
        "annualized_sortino": daily_sortino * scale if np.isfinite(daily_sortino) else float("nan"),
    }


def side_relative_returns(
    returns: Sequence[float] | np.ndarray | pd.Series,
    sides: Sequence[Any] | np.ndarray | pd.Series,
) -> np.ndarray:
    """Convert underlying-price returns into long/short economic returns.

    Known long values are ``long``, ``buy``, ``1`` and truthy booleans; known
    short values are ``short``, ``sell``, ``-1`` and false booleans.  Unknown
    side values produce ``NaN`` rather than silently assuming long.
    """

    values = pd.to_numeric(pd.Series(returns), errors="coerce").to_numpy(dtype=np.float64)
    side_values = pd.Series(sides)
    if len(values) != len(side_values):
        raise ValueError("returns and sides must have the same length")

    normalized = side_values.astype(str).str.strip().str.lower()
    signs = np.full(len(values), np.nan, dtype=np.float64)
    long_values = {"long", "buy", "1", "+1", "true", "t"}
    short_values = {"short", "sell", "-1", "false", "f"}
    signs[normalized.isin(long_values).to_numpy()] = 1.0
    signs[normalized.isin(short_values).to_numpy()] = -1.0
    signs[side_values.eq(True).to_numpy()] = 1.0
    signs[side_values.eq(False).to_numpy()] = -1.0
    return values * signs


def _trade_columns(
    trades: pd.DataFrame,
    *,
    timestamp_col: str,
    gross_return_col: str | None,
    net_return_col: str | None,
    bankroll_pnl_col: str | None,
    mfe_col: str | None,
    mae_col: str | None,
    holding_col: str | None,
    side_col: str | None,
    returns_are_side_relative: bool,
) -> pd.DataFrame:
    timestamp_name = _resolve_column(trades, timestamp_col, ("timestamp", "decision_ts", "entry_timestamp"), required=True)
    columns = pd.DataFrame(index=trades.index)
    columns["_source_row"] = np.arange(len(trades), dtype=np.int64)
    columns["timestamp"] = pd.to_datetime(trades[timestamp_name], utc=True, errors="coerce")
    columns["gross_return"] = _numeric(trades, _resolve_column(trades, gross_return_col, _RETURN_ALIASES["gross"]))
    columns["net_return"] = _numeric(trades, _resolve_column(trades, net_return_col, _RETURN_ALIASES["net"]))
    columns["bankroll_pnl"] = _numeric(trades, _resolve_column(trades, bankroll_pnl_col, _RETURN_ALIASES["bankroll"]))
    columns["mfe"] = _numeric(trades, _resolve_column(trades, mfe_col, _RETURN_ALIASES["mfe"]))
    columns["mae"] = _numeric(trades, _resolve_column(trades, mae_col, _RETURN_ALIASES["mae"]))
    columns["holding"] = _numeric(trades, _resolve_column(trades, holding_col, _RETURN_ALIASES["holding"]))

    if not returns_are_side_relative:
        resolved_side = _resolve_column(trades, side_col, ("side", "is_long"), required=True)
        for name in ("gross_return", "net_return", "mfe", "mae"):
            columns[name] = side_relative_returns(columns[name], trades[resolved_side])
    return columns


def _exit_share_columns(group: pd.DataFrame, exit_reason: pd.Series | None) -> dict[str, float]:
    if exit_reason is None or group.empty:
        return {}
    values = exit_reason.loc[group.index].dropna().astype(str).str.strip().str.lower()
    values = values[values.ne("")]
    if values.empty:
        return {}
    counts = values.value_counts(sort=False)
    out: dict[str, float] = {}
    for reason, count in counts.items():
        safe_reason = "".join(char if char.isalnum() else "_" for char in reason).strip("_") or "unknown"
        out[f"exit_{safe_reason}_share"] = float(count / len(group))
    return out


def _performance_row(group: pd.DataFrame, period_name: str, exit_reason: pd.Series | None) -> dict[str, Any]:
    net = group["net_return"].to_numpy(dtype=np.float64)
    finite_net = _finite(net)
    wins = finite_net[finite_net > 0.0]
    losses = finite_net[finite_net < 0.0]
    avg_win = _mean(wins)
    avg_loss = _mean(losses)
    row: dict[str, Any] = {
        "date": period_name,
        "trade_count": int(len(group)),
        "gross_return_sum": _sum(group["gross_return"].to_numpy(dtype=np.float64)),
        "gross_return_mean": _mean(group["gross_return"].to_numpy(dtype=np.float64)),
        "net_return_sum": _sum(net),
        "net_return_mean": _mean(net),
        "bankroll_pnl_sum": _sum(group["bankroll_pnl"].to_numpy(dtype=np.float64)),
        "win_rate": float(np.mean(finite_net > 0.0)) if finite_net.size else float("nan"),
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "payoff": _safe_ratio(avg_win, abs(avg_loss)),
        "profit_factor": _safe_ratio(_sum(wins), abs(_sum(losses))),
        "mfe_mean": _mean(group["mfe"].to_numpy(dtype=np.float64)),
        "mfe_median": _median(group["mfe"].to_numpy(dtype=np.float64)),
        "mae_mean": _mean(group["mae"].to_numpy(dtype=np.float64)),
        "mae_median": _median(group["mae"].to_numpy(dtype=np.float64)),
        "holding_mean": _mean(group["holding"].to_numpy(dtype=np.float64)),
        "holding_median": _median(group["holding"].to_numpy(dtype=np.float64)),
    }
    row.update(_exit_share_columns(group, exit_reason))
    return row


def daily_performance_decomposition(
    trades: pd.DataFrame,
    *,
    timestamp_col: str = "timestamp",
    gross_return_col: str | None = "gross_return",
    net_return_col: str | None = "net_return",
    bankroll_pnl_col: str | None = "bankroll_pnl",
    mfe_col: str | None = "mfe",
    mae_col: str | None = "mae",
    holding_col: str | None = "holding_hours",
    exit_reason_col: str | None = "exit_reason",
    side_col: str | None = "side",
    returns_are_side_relative: bool = True,
    annualization_days: float = 365.0,
) -> pd.DataFrame:
    """Return daily execution and economics diagnostics.

    The risk ratios use the series of daily net-return sums over the complete
    supplied frame.  They are repeated on every row so a CSV remains
    self-contained; they are ``NaN`` when support or dispersion is insufficient.
    """

    columns = _trade_columns(
        trades,
        timestamp_col=timestamp_col,
        gross_return_col=gross_return_col,
        net_return_col=net_return_col,
        bankroll_pnl_col=bankroll_pnl_col,
        mfe_col=mfe_col,
        mae_col=mae_col,
        holding_col=holding_col,
        side_col=side_col,
        returns_are_side_relative=returns_are_side_relative,
    )
    columns = columns.loc[columns["timestamp"].notna()].copy().reset_index(drop=True)
    if columns.empty:
        return _empty_decomposition()
    columns["date"] = columns["timestamp"].dt.floor("D")
    exit_column = _resolve_column(trades, exit_reason_col, ("exit_reason", "exit_type"))
    exit_reason = trades[exit_column].iloc[columns["_source_row"].to_numpy(dtype=np.int64)].reset_index(drop=True) if exit_column else None

    rows: list[dict[str, Any]] = []
    for date, group in columns.groupby("date", sort=True):
        rows.append(_performance_row(group, str(date), exit_reason))
    result = pd.DataFrame(rows)
    result["date"] = pd.to_datetime(result["date"], utc=True)
    ratios = _risk_ratios(result["net_return_sum"].to_numpy(dtype=np.float64), annualization_days)
    for name, value in ratios.items():
        result[name] = value
    return result


def monthly_performance_comparison(
    trades: pd.DataFrame,
    **kwargs: Any,
) -> pd.DataFrame:
    """Return the same execution/economic fields as daily diagnostics by month.

    Monthly rows calculate risk ratios from the daily net-return sums inside
    that month, so short months with sparse support remain explicitly undefined.
    """

    timestamp_col = kwargs.get("timestamp_col", "timestamp")
    columns = _trade_columns(
        trades,
        timestamp_col=timestamp_col,
        gross_return_col=kwargs.get("gross_return_col", "gross_return"),
        net_return_col=kwargs.get("net_return_col", "net_return"),
        bankroll_pnl_col=kwargs.get("bankroll_pnl_col", "bankroll_pnl"),
        mfe_col=kwargs.get("mfe_col", "mfe"),
        mae_col=kwargs.get("mae_col", "mae"),
        holding_col=kwargs.get("holding_col", "holding_hours"),
        side_col=kwargs.get("side_col", "side"),
        returns_are_side_relative=kwargs.get("returns_are_side_relative", True),
    )
    columns = columns.loc[columns["timestamp"].notna()].copy().reset_index(drop=True)
    if columns.empty:
        return _empty_decomposition().rename(columns={"date": "month"})
    columns["month"] = columns["timestamp"].dt.tz_localize(None).dt.to_period("M").astype(str)
    columns["day"] = columns["timestamp"].dt.floor("D")
    exit_column = _resolve_column(trades, kwargs.get("exit_reason_col", "exit_reason"), ("exit_reason", "exit_type"))
    exit_reason = trades[exit_column].iloc[columns["_source_row"].to_numpy(dtype=np.int64)].reset_index(drop=True) if exit_column else None
    annualization_days = float(kwargs.get("annualization_days", 365.0))

    rows: list[dict[str, Any]] = []
    for month, group in columns.groupby("month", sort=True):
        row = _performance_row(group, str(month), exit_reason)
        daily_returns = group.groupby("day", sort=True)["net_return"].sum(min_count=1).to_numpy(dtype=np.float64)
        row.update(_risk_ratios(daily_returns, annualization_days))
        rows.append(row)
    return pd.DataFrame(rows).rename(columns={"date": "month"})


def _probability_arrays(
    target: Sequence[float] | np.ndarray | pd.Series,
    score: Sequence[float] | np.ndarray | pd.Series,
) -> tuple[np.ndarray, np.ndarray]:
    y = pd.to_numeric(pd.Series(target), errors="coerce").to_numpy(dtype=np.float64)
    p = pd.to_numeric(pd.Series(score), errors="coerce").to_numpy(dtype=np.float64)
    if y.size != p.size:
        raise ValueError("target and score must have the same length")
    valid = np.isfinite(y) & np.isfinite(p) & (y >= 0.0) & (y <= 1.0)
    return y[valid], np.clip(p[valid], _EPSILON, 1.0 - _EPSILON)


def _binary_auc(y_binary: np.ndarray, scores: np.ndarray) -> float:
    positive = y_binary > 0
    n_positive = int(np.count_nonzero(positive))
    n_negative = int(len(y_binary) - n_positive)
    if not n_positive or not n_negative:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    return float((np.sum(ranks[positive]) - n_positive * (n_positive + 1) / 2.0) / (n_positive * n_negative))


def reliability_table(
    target: Sequence[float] | np.ndarray | pd.Series,
    score: Sequence[float] | np.ndarray | pd.Series,
    *,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Return equal-width reliability bins for hard or soft targets."""

    if n_bins < 1:
        raise ValueError("n_bins must be positive")
    y, p = _probability_arrays(target, score)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_index = np.minimum((p * n_bins).astype(int), n_bins - 1) if p.size else np.array([], dtype=int)
    rows: list[dict[str, float | int]] = []
    total = len(p)
    for index in range(n_bins):
        mask = bin_index == index
        support = int(np.count_nonzero(mask))
        predicted_mean = float(np.mean(p[mask])) if support else float("nan")
        observed_mean = float(np.mean(y[mask])) if support else float("nan")
        gap = observed_mean - predicted_mean if support else float("nan")
        rows.append(
            {
                "bin": index,
                "bin_left": float(edges[index]),
                "bin_right": float(edges[index + 1]),
                "support": support,
                "predicted_mean": predicted_mean,
                "observed_mean": observed_mean,
                "calibration_gap": gap,
                "ece_contribution": float((support / total) * abs(gap)) if support and total else 0.0,
            }
        )
    return pd.DataFrame(rows)


def calibration_metrics(
    target: Sequence[float] | np.ndarray | pd.Series,
    score: Sequence[float] | np.ndarray | pd.Series,
    *,
    n_bins: int = 10,
    hard_target_threshold: float = 0.5,
) -> dict[str, float | int]:
    """Calculate calibration and ranking metrics without fitting a calibrator."""

    y, p = _probability_arrays(target, score)
    metrics: dict[str, float | int] = {"support": int(len(y))}
    if not len(y):
        return {
            **metrics,
            "ece": float("nan"),
            "brier": float("nan"),
            "log_loss": float("nan"),
            "auc": float("nan"),
            "top_1_precision": float("nan"),
            "top_5_precision": float("nan"),
            "top_10_precision": float("nan"),
        }
    y_binary = (y >= hard_target_threshold).astype(np.int8)
    metrics["ece"] = float(reliability_table(y, p, n_bins=n_bins)["ece_contribution"].sum())
    metrics["brier"] = float(np.mean(np.square(p - y)))
    metrics["log_loss"] = float(-np.mean(y * np.log(p) + (1.0 - y) * np.log1p(-p)))
    metrics["auc"] = _binary_auc(y_binary, p)
    for percentage in (1, 5, 10):
        count = max(1, int(np.ceil(len(y) * percentage / 100.0)))
        top_indices = np.argsort(-p, kind="mergesort")[:count]
        metrics[f"top_{percentage}_precision"] = float(np.mean(y_binary[top_indices]))
    return metrics


def calibration_diagnostics(
    trades: pd.DataFrame,
    *,
    target_col: str = "target",
    score_col: str = "score",
    group_cols: str | Sequence[str] | None = None,
    n_bins: int = 10,
    hard_target_threshold: float = 0.5,
) -> dict[str, pd.DataFrame]:
    """Return overall and optional grouped calibration metric/reliability tables."""

    target_name = _resolve_column(trades, target_col, ("target", "label", "y_true"), required=True)
    score_name = _resolve_column(trades, score_col, ("score", "probability", "meta_score"), required=True)
    group_names = [group_cols] if isinstance(group_cols, str) else list(group_cols or [])
    missing_groups = [column for column in group_names if column not in trades.columns]
    if missing_groups:
        raise KeyError(f"Missing group columns: {missing_groups}")

    metric_rows: list[dict[str, Any]] = []
    reliability_rows: list[pd.DataFrame] = []

    def add_scope(scope: str, group: pd.DataFrame, group_values: tuple[Any, ...] = ()) -> None:
        base = {"scope": scope}
        base.update(dict(zip(group_names, group_values)))
        metric_rows.append({**base, **calibration_metrics(group[target_name], group[score_name], n_bins=n_bins, hard_target_threshold=hard_target_threshold)})
        table = reliability_table(group[target_name], group[score_name], n_bins=n_bins)
        for name, value in base.items():
            table[name] = value
        reliability_rows.append(table)

    add_scope("overall", trades)
    if group_names:
        grouped = trades.groupby(group_names, sort=True, dropna=False)
        for key, group in grouped:
            values = key if isinstance(key, tuple) else (key,)
            add_scope("grouped", group, values)
    reliability = pd.concat(reliability_rows, ignore_index=True) if reliability_rows else pd.DataFrame()
    return {"metrics": pd.DataFrame(metric_rows), "reliability": reliability}


def meta_score_tail_diagnostics(
    trades: pd.DataFrame,
    *,
    score_col: str = "meta_score",
    ev_col: str | None = "net_ev",
    net_return_col: str | None = "net_return",
    mfe_col: str | None = "mfe",
    mae_col: str | None = "mae",
    target_col: str | None = None,
    side_col: str | None = "side",
    returns_are_side_relative: bool = True,
    hard_target_threshold: float = 0.5,
) -> pd.DataFrame:
    """Report disjoint top-1, 1-2, 2-5 and 5-10 percent meta-score tails."""

    score_name = _resolve_column(trades, score_col, ("meta_score", "score", "probability"), required=True)
    ev_name = _resolve_column(trades, ev_col, ("net_ev", "net_return", "return_net"))
    if ev_name is None:
        ev_name = _resolve_column(trades, net_return_col, _RETURN_ALIASES["net"])
    mfe_name = _resolve_column(trades, mfe_col, _RETURN_ALIASES["mfe"])
    mae_name = _resolve_column(trades, mae_col, _RETURN_ALIASES["mae"])
    target_name = _resolve_column(trades, target_col, ("target", "label", "y_true")) if target_col else None

    frame = pd.DataFrame(
        {
            "score": _numeric(trades, score_name),
            "ev": _numeric(trades, ev_name),
            "mfe": _numeric(trades, mfe_name),
            "mae": _numeric(trades, mae_name),
        }
    )
    if not returns_are_side_relative:
        resolved_side = _resolve_column(trades, side_col, ("side", "is_long"), required=True)
        for name in ("ev", "mfe", "mae"):
            frame[name] = side_relative_returns(frame[name], trades[resolved_side])
    if target_name:
        frame["target"] = _numeric(trades, target_name)
    frame = frame.loc[np.isfinite(frame["score"])].sort_values("score", ascending=False, kind="mergesort").reset_index(drop=True)
    boundaries = (("top_1", 0.0, 0.01), ("top_1_2", 0.01, 0.02), ("top_2_5", 0.02, 0.05), ("top_5_10", 0.05, 0.10))
    rows: list[dict[str, Any]] = []
    total = len(frame)
    for name, lower, upper in boundaries:
        start = int(np.ceil(total * lower))
        end = int(np.ceil(total * upper))
        subset = frame.iloc[start:end]
        ev = subset["ev"].to_numpy(dtype=np.float64)
        finite_ev = _finite(ev)
        wins = finite_ev[finite_ev > 0.0]
        losses = finite_ev[finite_ev < 0.0]
        row: dict[str, Any] = {
            "tail": name,
            "start_pct": lower * 100.0,
            "end_pct": upper * 100.0,
            "trade_count": int(len(subset)),
            "ev_mean": _mean(ev),
            "ev_sum": _sum(ev),
            "win_rate": float(np.mean(finite_ev > 0.0)) if finite_ev.size else float("nan"),
            "profit_factor": _safe_ratio(_sum(wins), abs(_sum(losses))),
            "mfe_mean": _mean(subset["mfe"].to_numpy(dtype=np.float64)),
            "mfe_median": _median(subset["mfe"].to_numpy(dtype=np.float64)),
            "mae_mean": _mean(subset["mae"].to_numpy(dtype=np.float64)),
            "mae_median": _median(subset["mae"].to_numpy(dtype=np.float64)),
        }
        if "target" in subset:
            target = _finite(subset["target"].to_numpy(dtype=np.float64))
            row["precision"] = float(np.mean(target >= hard_target_threshold)) if target.size else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)
