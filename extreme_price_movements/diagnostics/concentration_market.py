"""Pure concentration and market-context diagnostics for candidate/trade rows.

The functions in this module summarize already-produced decisions.  They do
not fit, score, calibrate, or otherwise alter a model or policy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

DEFAULT_MARKET_STATE_COLUMNS: Mapping[str, tuple[str, ...]] = {
    "atr": ("atr", "atr_pct", "market_atr", "market_atr_pct"),
    "rv": ("rv", "realized_volatility", "realized_vol", "market_rv"),
    "dispersion": ("dispersion", "cross_sectional_dispersion", "market_dispersion"),
    "correlation": (
        "correlation",
        "avg_pairwise_corr",
        "average_pairwise_correlation",
        "market_correlation",
    ),
    "pc1_share": ("pc1_share", "pc1_variance_share", "market_pc1_share"),
    "trend_efficiency": ("trend_efficiency", "market_trend_efficiency"),
    "wick": ("wick", "wick_ratio", "market_wick"),
    "volume_z": ("volume_z", "volume_zscore", "market_volume_z"),
    "btc_dominance": ("btc_dominance", "bitcoin_dominance"),
    "btc_return": ("btc_return", "btc_ret", "bitcoin_return"),
    "eth_return": ("eth_return", "eth_ret", "ethereum_return"),
}


@dataclass(frozen=True)
class ConcentrationMarketDiagnostics:
    """Result tables produced by :func:`build_concentration_market_diagnostics`."""

    daily: pd.DataFrame
    structural_breaks: pd.DataFrame
    monthly_comparisons: pd.DataFrame


def _empty_daily_frame() -> pd.DataFrame:
    columns = [
        "date",
        "n_rows",
        "n_symbols",
        "symbol_hhi",
        "symbol_effective_count",
        "symbol_top1_share",
        "symbol_top3_share",
        "same_side_share",
        "same_setup_share",
        "same_model_share",
        "same_entry_hour_share",
        "average_prediction_similarity",
        "average_feature_cosine",
        "average_embedding_cosine",
        "average_pairwise_trade_return_correlation",
        "average_pairwise_asset_return_correlation",
        "n_asset_return_pairs",
    ]
    columns.extend(f"market_{name}" for name in DEFAULT_MARKET_STATE_COLUMNS)
    return pd.DataFrame(columns=columns)


def _utc_timestamp(values: pd.Series | pd.Index) -> pd.Series:
    return pd.Series(pd.to_datetime(values, utc=True, errors="coerce"), index=getattr(values, "index", None))


def _with_timestamp_column(frame: pd.DataFrame, timestamp_col: str) -> pd.DataFrame:
    out = frame.copy()
    if timestamp_col not in out.columns:
        if isinstance(out.index, pd.DatetimeIndex):
            out[timestamp_col] = out.index
        else:
            raise ValueError(
                f"Expected '{timestamp_col}' column or a DatetimeIndex for timestamps"
            )
    out[timestamp_col] = _utc_timestamp(out[timestamp_col])
    return out.loc[out[timestamp_col].notna()].copy()


def _mode_share(values: pd.Series) -> float:
    present = values.dropna()
    if present.empty:
        return np.nan
    return float(present.value_counts(dropna=True).iloc[0] / len(present))


def _mean_pairwise_cosine(values: pd.DataFrame) -> float:
    if len(values) < 2 or values.shape[1] == 0:
        return np.nan
    array = values.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(array).all(axis=1)
    array = array[valid]
    if len(array) < 2:
        return np.nan
    norms = np.linalg.norm(array, axis=1)
    array = array[norms > 0.0]
    norms = norms[norms > 0.0]
    if len(array) < 2:
        return np.nan
    normalized = array / norms[:, None]
    similarities = normalized @ normalized.T
    return float(similarities[np.triu_indices(len(similarities), k=1)].mean())


def _mean_prediction_similarity(values: pd.DataFrame) -> float:
    """Return vector cosine similarity, or bounded scalar-score closeness."""
    if len(values) < 2 or values.shape[1] == 0:
        return np.nan
    if values.shape[1] > 1:
        return _mean_pairwise_cosine(values)
    score = pd.to_numeric(values.iloc[:, 0], errors="coerce").dropna().to_numpy(dtype=float)
    if len(score) < 2:
        return np.nan
    spread = float(score.max() - score.min())
    if spread <= 0.0:
        return 1.0
    differences = np.abs(score[:, None] - score[None, :]) / spread
    return float((1.0 - differences)[np.triu_indices(len(score), k=1)].mean())


def _market_returns_wide(
    market_returns: pd.DataFrame | None,
    *,
    timestamp_col: str,
    symbol_col: str,
    return_col: str,
) -> pd.DataFrame:
    if market_returns is None or market_returns.empty:
        return pd.DataFrame()
    returns = _with_timestamp_column(market_returns, timestamp_col)
    if symbol_col in returns.columns and return_col in returns.columns:
        long = returns[[timestamp_col, symbol_col, return_col]].copy()
        long[return_col] = pd.to_numeric(long[return_col], errors="coerce")
        return long.pivot_table(
            index=timestamp_col, columns=symbol_col, values=return_col, aggfunc="last"
        ).sort_index()
    value_columns = [column for column in returns.columns if column != timestamp_col]
    if not value_columns:
        return pd.DataFrame()
    wide = returns.set_index(timestamp_col)[value_columns]
    return wide.apply(pd.to_numeric, errors="coerce").sort_index()


def _mean_pairwise_return_correlation(
    returns: pd.DataFrame,
    day: pd.Timestamp,
    symbols: Iterable[object],
) -> tuple[float, int]:
    if returns.empty:
        return np.nan, 0
    names = [symbol for symbol in pd.unique(pd.Series(list(symbols)).dropna()) if symbol in returns.columns]
    if len(names) < 2:
        return np.nan, 0
    same_day = returns.loc[returns.index.normalize() == day.normalize(), names]
    if len(same_day) < 2:
        return np.nan, 0
    correlation = same_day.corr()
    values = correlation.to_numpy(dtype=float)[np.triu_indices(len(names), k=1)]
    values = values[np.isfinite(values)]
    return (float(values.mean()), int(len(values))) if len(values) else (np.nan, 0)


def _daily_market_state_summary(
    market_state: pd.DataFrame | None,
    *,
    timestamp_col: str,
    column_aliases: Mapping[str, Sequence[str]],
) -> pd.DataFrame:
    expected = [f"market_{name}" for name in column_aliases]
    if market_state is None or market_state.empty:
        return pd.DataFrame(columns=["date", *expected])
    state = _with_timestamp_column(market_state, timestamp_col)
    state["date"] = state[timestamp_col].dt.normalize()
    output = pd.DataFrame({"date": pd.Index(state["date"].unique()).sort_values()})
    for name, aliases in column_aliases.items():
        source = next((column for column in aliases if column in state.columns), None)
        if source is None:
            output[f"market_{name}"] = np.nan
            continue
        values = pd.to_numeric(state[source], errors="coerce")
        summary = values.groupby(state["date"]).mean()
        output[f"market_{name}"] = output["date"].map(summary)
    return output


def build_daily_concentration_metrics(
    rows: pd.DataFrame,
    *,
    market_returns: pd.DataFrame | None = None,
    timestamp_col: str = "timestamp",
    symbol_col: str = "symbol",
    side_col: str = "side",
    setup_col: str = "setup",
    model_col: str = "model",
    entry_hour_col: str | None = None,
    prediction_columns: Sequence[str] = ("prediction",),
    feature_columns: Sequence[str] = (),
    embedding_columns: Sequence[str] = (),
    market_return_timestamp_col: str = "timestamp",
    market_return_symbol_col: str = "symbol",
    market_return_col: str = "return",
) -> pd.DataFrame:
    """Build one concentration row per UTC day from candidate/trade rows.

    Symbol and categorical shares use row counts.  Return correlations use the
    supplied intraday return panel restricted to symbols active that day; a
    single supplied daily return per symbol is therefore intentionally not
    enough to infer a correlation.
    """
    if symbol_col not in rows.columns:
        raise ValueError(f"rows must include '{symbol_col}'")
    data = _with_timestamp_column(rows, timestamp_col)
    if data.empty:
        return _empty_daily_frame()
    data["date"] = data[timestamp_col].dt.normalize()
    returns = _market_returns_wide(
        market_returns,
        timestamp_col=market_return_timestamp_col,
        symbol_col=market_return_symbol_col,
        return_col=market_return_col,
    )
    present_predictions = [column for column in prediction_columns if column in data]
    present_features = [column for column in feature_columns if column in data]
    present_embeddings = [column for column in embedding_columns if column in data]
    records: list[dict[str, object]] = []

    for day, group in data.groupby("date", sort=True):
        symbol_counts = group[symbol_col].dropna().value_counts()
        shares = symbol_counts / symbol_counts.sum() if not symbol_counts.empty else pd.Series(dtype=float)
        hhi = float((shares**2).sum()) if not shares.empty else np.nan
        entry_hours = (
            group[entry_hour_col]
            if entry_hour_col is not None and entry_hour_col in group.columns
            else group[timestamp_col].dt.hour
        )
        return_corr, return_pairs = _mean_pairwise_return_correlation(
            returns, day, symbol_counts.index
        )
        records.append(
            {
                "date": day,
                "n_rows": int(len(group)),
                "n_symbols": int(len(symbol_counts)),
                "symbol_hhi": hhi,
                "symbol_effective_count": float(1.0 / hhi) if hhi and hhi > 0.0 else np.nan,
                "symbol_top1_share": float(shares.iloc[0]) if not shares.empty else np.nan,
                "symbol_top3_share": float(shares.iloc[:3].sum()) if not shares.empty else np.nan,
                "same_side_share": _mode_share(group[side_col]) if side_col in group else np.nan,
                "same_setup_share": _mode_share(group[setup_col]) if setup_col in group else np.nan,
                "same_model_share": _mode_share(group[model_col]) if model_col in group else np.nan,
                "same_entry_hour_share": _mode_share(entry_hours),
                "average_prediction_similarity": _mean_prediction_similarity(group[present_predictions]),
                "average_feature_cosine": _mean_pairwise_cosine(group[present_features]),
                "average_embedding_cosine": _mean_pairwise_cosine(group[present_embeddings]),
                "average_pairwise_trade_return_correlation": return_corr,
                "average_pairwise_asset_return_correlation": return_corr,
                "n_asset_return_pairs": return_pairs,
            }
        )
    return pd.DataFrame.from_records(records, columns=_empty_daily_frame().columns)


def build_structural_break_diagnostics(
    daily: pd.DataFrame,
    *,
    date_col: str = "date",
    metrics: Sequence[str] | None = None,
    robust_z_threshold: float = 3.5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare daily values and month medians to pooled robust baselines.

    The pooled median and IQR are descriptive references across the supplied
    sample, not fitted parameters or predictive estimates.  A zero IQR yields
    a zero robust score for values equal to the median and an infinite score for
    distinct values, making the exceptional value explicit.
    """
    daily = daily.copy()
    if daily.empty:
        return (
            pd.DataFrame(columns=["date", "metric", "value", "pooled_median", "pooled_iqr", "robust_z", "is_structural_break"]),
            pd.DataFrame(columns=["month", "metric", "n_days", "month_median", "pooled_median", "pooled_iqr", "month_robust_z", "previous_month_median", "month_to_month_delta", "month_to_month_robust_z", "is_structural_break"]),
        )
    if date_col not in daily.columns:
        raise ValueError(f"daily must include '{date_col}'")
    daily[date_col] = _utc_timestamp(daily[date_col])
    daily = daily.loc[daily[date_col].notna()].copy()
    if metrics is None:
        metrics = [
            column
            for column in daily.columns
            if column != date_col and pd.api.types.is_numeric_dtype(daily[column])
        ]
    break_rows: list[dict[str, object]] = []
    month_rows: list[dict[str, object]] = []
    for metric in metrics:
        if metric not in daily.columns:
            continue
        values = pd.to_numeric(daily[metric], errors="coerce")
        finite = values[np.isfinite(values)]
        if finite.empty:
            continue
        median = float(finite.median())
        iqr = float(finite.quantile(0.75) - finite.quantile(0.25))
        scale = iqr / 1.349 if iqr > 0.0 else 0.0
        for date, value in zip(daily[date_col], values):
            if not np.isfinite(value):
                z = np.nan
            elif scale > 0.0:
                z = float((value - median) / scale)
            elif value == median:
                z = 0.0
            else:
                z = float(np.copysign(np.inf, value - median))
            break_rows.append(
                {
                    "date": date,
                    "metric": metric,
                    "value": value,
                    "pooled_median": median,
                    "pooled_iqr": iqr,
                    "robust_z": z,
                    "is_structural_break": bool(np.isfinite(z) and abs(z) >= robust_z_threshold) or bool(np.isinf(z)),
                }
            )
        months = (
            daily[date_col]
            .dt.tz_localize(None)
            .dt.to_period("M")
            .dt.to_timestamp()
            .dt.tz_localize("UTC")
        )
        monthly = pd.DataFrame({"month": months, "value": values}).dropna()
        summary = monthly.groupby("month", sort=True)["value"].agg(n_days="count", month_median="median").reset_index()
        previous = summary["month_median"].shift(1)
        for row, previous_median in zip(summary.itertuples(index=False), previous):
            if scale > 0.0:
                month_z = float((row.month_median - median) / scale)
                delta_z = float((row.month_median - previous_median) / scale) if np.isfinite(previous_median) else np.nan
            elif row.month_median == median:
                month_z = 0.0
                delta_z = 0.0 if np.isfinite(previous_median) and row.month_median == previous_median else np.nan
            else:
                month_z = float(np.copysign(np.inf, row.month_median - median))
                delta_z = float(np.copysign(np.inf, row.month_median - previous_median)) if np.isfinite(previous_median) and row.month_median != previous_median else np.nan
            month_rows.append(
                {
                    "month": row.month,
                    "metric": metric,
                    "n_days": int(row.n_days),
                    "month_median": float(row.month_median),
                    "pooled_median": median,
                    "pooled_iqr": iqr,
                    "month_robust_z": month_z,
                    "previous_month_median": previous_median,
                    "month_to_month_delta": float(row.month_median - previous_median) if np.isfinite(previous_median) else np.nan,
                    "month_to_month_robust_z": delta_z,
                    "is_structural_break": bool(np.isfinite(month_z) and abs(month_z) >= robust_z_threshold) or bool(np.isinf(month_z)),
                }
            )
    return pd.DataFrame(break_rows), pd.DataFrame(month_rows)


def build_concentration_market_diagnostics(
    rows: pd.DataFrame,
    *,
    market_returns: pd.DataFrame | None = None,
    market_state: pd.DataFrame | None = None,
    timestamp_col: str = "timestamp",
    symbol_col: str = "symbol",
    side_col: str = "side",
    setup_col: str = "setup",
    model_col: str = "model",
    entry_hour_col: str | None = None,
    prediction_columns: Sequence[str] = ("prediction",),
    feature_columns: Sequence[str] = (),
    embedding_columns: Sequence[str] = (),
    market_return_timestamp_col: str = "timestamp",
    market_return_symbol_col: str = "symbol",
    market_return_col: str = "return",
    market_state_timestamp_col: str = "timestamp",
    market_state_columns: Mapping[str, Sequence[str]] = DEFAULT_MARKET_STATE_COLUMNS,
    robust_z_threshold: float = 3.5,
) -> ConcentrationMarketDiagnostics:
    """Build concentration, market-state, and robust structural-break tables."""
    daily = build_daily_concentration_metrics(
        rows,
        market_returns=market_returns,
        timestamp_col=timestamp_col,
        symbol_col=symbol_col,
        side_col=side_col,
        setup_col=setup_col,
        model_col=model_col,
        entry_hour_col=entry_hour_col,
        prediction_columns=prediction_columns,
        feature_columns=feature_columns,
        embedding_columns=embedding_columns,
        market_return_timestamp_col=market_return_timestamp_col,
        market_return_symbol_col=market_return_symbol_col,
        market_return_col=market_return_col,
    )
    market_daily = _daily_market_state_summary(
        market_state,
        timestamp_col=market_state_timestamp_col,
        column_aliases=market_state_columns,
    )
    if not market_daily.empty:
        daily = daily.merge(market_daily, on="date", how="left", suffixes=("", "_state"))
        for name in market_state_columns:
            state_column = f"market_{name}_state"
            if state_column in daily.columns:
                daily[f"market_{name}"] = daily[state_column]
                daily = daily.drop(columns=state_column)
    for name in market_state_columns:
        column = f"market_{name}"
        if column not in daily.columns:
            daily[column] = np.nan
    daily = daily.loc[:, _empty_daily_frame().columns]
    breaks, months = build_structural_break_diagnostics(
        daily, robust_z_threshold=robust_z_threshold
    )
    return ConcentrationMarketDiagnostics(
        daily=daily, structural_breaks=breaks, monthly_comparisons=months
    )


run_concentration_market_diagnostics = build_concentration_market_diagnostics
