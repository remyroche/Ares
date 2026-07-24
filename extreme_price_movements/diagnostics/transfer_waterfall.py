"""Read-only diagnostics for comparing staged trade ledgers on identical rows.

The functions here are intentionally descriptive.  They never fit a model,
choose an admission threshold, or apply a cost.  ``net_return`` is treated as
the final post-cost return supplied by each ledger; the reported cost drag is
only the observed ``gross_return - net_return`` reconciliation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd


_DEFAULT_KEY_COLS = ("timestamp", "symbol", "side")
_ALIASES = {
    "timestamp": ("timestamp", "__ts__", "decision_ts", "entry_timestamp"),
    "side": ("side", "side_name"),
    "archetype": ("archetype", "policy_archetype", "archetype_policy_key"),
    "score": (
        "score",
        "ranking_score",
        "meta_score",
        "calibrated_score",
        "expected_ev_rank_score",
    ),
    "gross_return": ("gross_return", "gross_ret", "return_gross"),
    "net_return": ("net_return", "net_ret", "net_ev", "return_net", "pnl_net"),
    "mfe": ("mfe", "max_favorable_return", "max_favorable_return_until_exit"),
    "mae": ("mae", "max_adverse_return", "max_adverse_return_until_exit"),
    "exit_reason": ("exit_reason", "simple_policy_exit_reason"),
    "target": ("target", "target_hit", "is_target", "take_profit_hit"),
    "stop": ("stop", "stop_hit", "is_stop", "stop_loss_hit"),
    "timeout": ("timeout", "is_timeout", "timed_out"),
    "baseline_side": (
        "baseline_side",
        "stored_baseline_side",
        "baseline_direction",
        "baseline_signal_side",
    ),
}
_TARGET_REASONS = frozenset(("target", "tp", "hard_tp", "take_profit"))
_STOP_REASONS = frozenset(("stop", "sl", "full_sl", "stop_loss", "adverse_exit"))
_TIMEOUT_REASONS = frozenset(("timeout", "time_out"))
_METRIC_COLUMNS = (
    "trade_count",
    "ev_per_trade",
    "net_return_sum",
    "gross_per_trade",
    "gross_return_sum",
    "cost_drag_per_trade",
    "cost_drag_sum",
    "gross_to_net_ratio",
    "win_rate",
    "mfe_mean",
    "mae_mean",
    "target_rate",
    "stop_rate",
    "timeout_rate",
    "flip_vs_previous_count",
    "flip_vs_previous_support",
    "flip_vs_previous_rate",
    "flip_vs_baseline_count",
    "flip_vs_baseline_support",
    "flip_vs_baseline_rate",
)


def _resolve_column(frame: pd.DataFrame, requested: str | None, name: str, *, required: bool = False) -> str | None:
    candidates = (requested,) if requested else ()
    candidates += tuple(column for column in _ALIASES[name] if column not in candidates)
    for column in candidates:
        if column in frame.columns:
            return column
    if required:
        raise KeyError(f"Missing required {name!r} column; tried {list(candidates)}")
    return None


def _numeric(frame: pd.DataFrame, column: str | None) -> np.ndarray:
    if column is None:
        return np.full(len(frame), np.nan, dtype=np.float64)
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=np.float64, copy=False)
    return np.where(np.isfinite(values), values, np.nan)


def _side_code(values: pd.Series | Sequence[Any]) -> np.ndarray:
    series = pd.Series(values, copy=False)
    normalized = series.astype("string").str.strip().str.lower()
    result = np.full(len(series), np.nan, dtype=np.float64)
    result[normalized.isin(("long", "buy", "1", "+1", "true", "t")).to_numpy()] = 1.0
    result[normalized.isin(("short", "sell", "-1", "false", "f")).to_numpy()] = -1.0
    result[series.eq(True).to_numpy()] = 1.0
    result[series.eq(False).to_numpy()] = -1.0
    return result


def _reason_code(values: pd.Series | Sequence[Any]) -> pd.Series:
    return pd.Series(values, copy=False).astype("string").str.strip().str.lower()


def _staged_items(staged_ledgers: Mapping[str, pd.DataFrame] | Sequence[tuple[str, pd.DataFrame]]) -> list[tuple[str, pd.DataFrame]]:
    items = list(staged_ledgers.items()) if isinstance(staged_ledgers, Mapping) else list(staged_ledgers)
    if not items:
        raise ValueError("At least one staged ledger is required")
    names = [str(name) for name, _ in items]
    if len(names) != len(set(names)):
        raise ValueError("Stage names must be unique")
    for name, ledger in items:
        if not isinstance(ledger, pd.DataFrame):
            raise TypeError(f"Stage {name!r} must be a pandas DataFrame")
    return [(str(name), ledger) for name, ledger in items]


def _validate_keys(frame: pd.DataFrame, key_cols: Sequence[str], stage: str) -> None:
    missing = [column for column in key_cols if column not in frame.columns]
    if missing:
        raise KeyError(f"Stage {stage!r} is missing key columns: {missing}")
    if frame.loc[:, key_cols].isna().any(axis=None):
        raise ValueError(f"Stage {stage!r} has null values in key columns")
    duplicate_count = int(frame.duplicated(list(key_cols)).sum())
    if duplicate_count:
        raise ValueError(f"Stage {stage!r} has {duplicate_count} duplicate key rows")


def normalize_stage_ledger(
    ledger: pd.DataFrame,
    *,
    stage: str,
    key_cols: Sequence[str] = _DEFAULT_KEY_COLS,
    timestamp_col: str | None = None,
    side_col: str | None = None,
    archetype_col: str | None = None,
    score_col: str | None = None,
    gross_return_col: str | None = None,
    net_return_col: str | None = None,
    mfe_col: str | None = None,
    mae_col: str | None = None,
    exit_reason_col: str | None = None,
    target_col: str | None = None,
    stop_col: str | None = None,
    timeout_col: str | None = None,
    baseline_side_col: str | None = None,
) -> pd.DataFrame:
    """Return a canonical, non-mutating view of one stage ledger.

    ``key_cols`` define the identical-row contract and must be complete and
    unique in every stage.  Gross return is optional, but net return is
    required because it is the supplied post-cost EV used by all metrics.
    """

    keys = tuple(key_cols)
    if not keys:
        raise ValueError("key_cols must not be empty")
    _validate_keys(ledger, keys, stage)
    resolved = {
        "timestamp": _resolve_column(ledger, timestamp_col, "timestamp"),
        "side": _resolve_column(ledger, side_col, "side"),
        "archetype": _resolve_column(ledger, archetype_col, "archetype"),
        "score": _resolve_column(ledger, score_col, "score"),
        "gross_return": _resolve_column(ledger, gross_return_col, "gross_return"),
        "net_return": _resolve_column(ledger, net_return_col, "net_return", required=True),
        "mfe": _resolve_column(ledger, mfe_col, "mfe"),
        "mae": _resolve_column(ledger, mae_col, "mae"),
        "exit_reason": _resolve_column(ledger, exit_reason_col, "exit_reason"),
        "target": _resolve_column(ledger, target_col, "target"),
        "stop": _resolve_column(ledger, stop_col, "stop"),
        "timeout": _resolve_column(ledger, timeout_col, "timeout"),
        "baseline_side": _resolve_column(ledger, baseline_side_col, "baseline_side"),
    }
    output = ledger.loc[:, list(keys)].copy()
    output["stage"] = str(stage)
    output["timestamp"] = (
        pd.to_datetime(ledger[resolved["timestamp"]], utc=True, errors="coerce")
        if resolved["timestamp"] is not None
        else pd.NaT
    )
    output["side"] = ledger[resolved["side"]].to_numpy() if resolved["side"] is not None else pd.NA
    output["archetype"] = ledger[resolved["archetype"]].to_numpy() if resolved["archetype"] is not None else pd.NA
    output["score"] = _numeric(ledger, resolved["score"])
    output["gross_return"] = _numeric(ledger, resolved["gross_return"])
    output["net_return"] = _numeric(ledger, resolved["net_return"])
    output["mfe"] = _numeric(ledger, resolved["mfe"])
    output["mae"] = _numeric(ledger, resolved["mae"])
    output["exit_reason"] = (
        _reason_code(ledger[resolved["exit_reason"]]).to_numpy()
        if resolved["exit_reason"] is not None
        else pd.NA
    )
    for name in ("target", "stop", "timeout"):
        values = _numeric(ledger, resolved[name])
        output[f"_tw_{name}_flag"] = np.where(np.isfinite(values), values > 0.0, np.nan)
    output["baseline_side"] = (
        ledger[resolved["baseline_side"]].to_numpy()
        if resolved["baseline_side"] is not None
        else pd.NA
    )
    output["_tw_side_code"] = _side_code(output["side"])
    output["_tw_baseline_side_code"] = _side_code(output["baseline_side"])
    return output


def common_key_intersection(
    staged_ledgers: Mapping[str, pd.DataFrame] | Sequence[tuple[str, pd.DataFrame]],
    *,
    key_cols: Sequence[str] = _DEFAULT_KEY_COLS,
) -> pd.DataFrame:
    """Return the unique key rows present in every supplied stage ledger."""

    items = _staged_items(staged_ledgers)
    keys = tuple(key_cols)
    common: pd.DataFrame | None = None
    for stage, frame in items:
        _validate_keys(frame, keys, stage)
        stage_keys = frame.loc[:, list(keys)]
        common = stage_keys.copy() if common is None else common.merge(
            stage_keys, on=list(keys), how="inner", sort=False, validate="one_to_one"
        )
    return common.reset_index(drop=True) if common is not None else pd.DataFrame(columns=keys)


def enforce_common_key_intersection(
    staged_ledgers: Mapping[str, pd.DataFrame] | Sequence[tuple[str, pd.DataFrame]],
    *,
    key_cols: Sequence[str] = _DEFAULT_KEY_COLS,
) -> dict[str, pd.DataFrame]:
    """Filter every stage to the common keys, preserving each ledger's order."""

    items = _staged_items(staged_ledgers)
    keys = tuple(key_cols)
    common = common_key_intersection(items, key_cols=keys)
    common_index = pd.MultiIndex.from_frame(common.loc[:, list(keys)])
    aligned: dict[str, pd.DataFrame] = {}
    for stage, frame in items:
        row_index = pd.MultiIndex.from_frame(frame.loc[:, list(keys)])
        aligned[stage] = frame.loc[row_index.isin(common_index)].copy().reset_index(drop=True)
    return aligned


def normalize_staged_ledgers(
    staged_ledgers: Mapping[str, pd.DataFrame] | Sequence[tuple[str, pd.DataFrame]],
    *,
    key_cols: Sequence[str] = _DEFAULT_KEY_COLS,
    **column_options: str | None,
) -> dict[str, pd.DataFrame]:
    """Normalize stages and enforce their common identical-row key intersection."""

    items = _staged_items(staged_ledgers)
    normalized = {
        stage: normalize_stage_ledger(frame, stage=stage, key_cols=key_cols, **column_options)
        for stage, frame in items
    }
    return enforce_common_key_intersection(normalized, key_cols=key_cols)


def _finite_mean(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(finite.mean()) if finite.size else float("nan")


def _finite_sum(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(finite.sum()) if finite.size else float("nan")


def _rate(mask: np.ndarray, valid: np.ndarray) -> tuple[int, int, float]:
    support = int(valid.sum())
    count = int(np.count_nonzero(mask & valid))
    return count, support, float(count / support) if support else float("nan")


def _outcome_rate(flags: np.ndarray, exit_reason: np.ndarray, reasons: frozenset[str]) -> float:
    finite_flags = pd.notna(flags)
    if finite_flags.any():
        return float(np.mean(np.asarray(flags[finite_flags], dtype=bool)))
    return float(np.mean(np.isin(exit_reason, tuple(reasons)))) if len(exit_reason) else float("nan")


def _metrics(frame: pd.DataFrame) -> dict[str, float | int]:
    net = frame["net_return"].to_numpy(dtype=np.float64, copy=False)
    gross = frame["gross_return"].to_numpy(dtype=np.float64, copy=False)
    cost_drag = gross - net
    finite_net = np.isfinite(net)
    exit_reason = frame["exit_reason"].astype("string").fillna("").to_numpy(dtype=object)
    target_flag = frame["_tw_target_flag"].to_numpy(dtype=object)
    stop_flag = frame["_tw_stop_flag"].to_numpy(dtype=object)
    timeout_flag = frame["_tw_timeout_flag"].to_numpy(dtype=object)
    positive = np.where(finite_net, net > 0.0, np.nan)
    prior_positive = frame.get(
        "_tw_previous_positive", pd.Series(np.nan, index=frame.index)
    ).to_numpy(dtype=np.float64)
    baseline_positive = frame.get(
        "_tw_baseline_positive", pd.Series(np.nan, index=frame.index)
    ).to_numpy(dtype=np.float64)
    previous_valid = np.isfinite(positive) & np.isfinite(prior_positive)
    baseline_valid = np.isfinite(positive) & np.isfinite(baseline_positive)
    previous_count, previous_support, previous_rate = _rate(
        positive != prior_positive, previous_valid
    )
    baseline_count, baseline_support, baseline_rate = _rate(
        positive != baseline_positive, baseline_valid
    )
    gross_sum = _finite_sum(gross)
    net_sum = _finite_sum(net)
    return {
        "trade_count": int(len(frame)),
        "ev_per_trade": _finite_mean(net),
        "net_return_sum": net_sum,
        "gross_per_trade": _finite_mean(gross),
        "gross_return_sum": gross_sum,
        "cost_drag_per_trade": _finite_mean(cost_drag),
        "cost_drag_sum": _finite_sum(cost_drag),
        "gross_to_net_ratio": float(net_sum / gross_sum)
        if np.isfinite(net_sum) and np.isfinite(gross_sum) and abs(gross_sum) > 1e-12
        else float("nan"),
        "win_rate": float(np.mean(net[finite_net] > 0.0)) if finite_net.any() else float("nan"),
        "mfe_mean": _finite_mean(frame["mfe"].to_numpy(dtype=np.float64, copy=False)),
        "mae_mean": _finite_mean(frame["mae"].to_numpy(dtype=np.float64, copy=False)),
        "target_rate": _outcome_rate(target_flag, exit_reason, _TARGET_REASONS),
        "stop_rate": _outcome_rate(stop_flag, exit_reason, _STOP_REASONS),
        "timeout_rate": _outcome_rate(timeout_flag, exit_reason, _TIMEOUT_REASONS),
        "flip_vs_previous_count": previous_count,
        "flip_vs_previous_support": previous_support,
        "flip_vs_previous_rate": previous_rate,
        "flip_vs_baseline_count": baseline_count,
        "flip_vs_baseline_support": baseline_support,
        "flip_vs_baseline_rate": baseline_rate,
    }


def _with_outcome_references(
    normalized_ledgers: Mapping[str, pd.DataFrame], key_cols: Sequence[str]
) -> dict[str, pd.DataFrame]:
    enriched: dict[str, pd.DataFrame] = {}
    previous: pd.DataFrame | None = None
    baseline: pd.DataFrame | None = None
    for stage, frame in normalized_ledgers.items():
        current = frame.copy()
        current["_tw_positive"] = np.where(
            np.isfinite(current["net_return"]), current["net_return"] > 0.0, np.nan
        )
        if baseline is None:
            baseline = current.loc[:, [*key_cols, "_tw_positive"]].rename(
                columns={"_tw_positive": "_tw_baseline_positive"}
            )
        current = current.merge(
            baseline, on=list(key_cols), how="left", sort=False, validate="one_to_one"
        )
        if previous is None:
            current["_tw_previous_positive"] = np.nan
        else:
            previous_outcome = previous.loc[:, [*key_cols, "_tw_positive"]].rename(
                columns={"_tw_positive": "_tw_previous_positive"}
            )
            current = current.merge(
                previous_outcome,
                on=list(key_cols),
                how="left",
                sort=False,
                validate="one_to_one",
            )
        enriched[stage] = current
        previous = current
    return enriched


def stage_waterfall_metrics(
    staged_ledgers: Mapping[str, pd.DataFrame] | Sequence[tuple[str, pd.DataFrame]],
    *,
    key_cols: Sequence[str] = _DEFAULT_KEY_COLS,
    **column_options: str | None,
) -> pd.DataFrame:
    """Compute comparable per-stage metrics after enforcing the common key set."""

    ledgers = normalize_staged_ledgers(staged_ledgers, key_cols=key_cols, **column_options)
    enriched = _with_outcome_references(ledgers, tuple(key_cols))
    return pd.DataFrame([{"stage": stage, **_metrics(frame)} for stage, frame in enriched.items()])


def _grouped_metrics(
    ledgers: Mapping[str, pd.DataFrame], group_col: str, *, include_empty_tails: Sequence[str] = ()
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for stage, frame in ledgers.items():
        values = frame[group_col].astype("string").fillna("__missing__")
        groups = [(value, frame.loc[values.eq(value)]) for value in sorted(values.unique())]
        if include_empty_tails:
            present = {str(value) for value, _ in groups}
            groups.extend((tail, frame.iloc[0:0]) for tail in include_empty_tails if tail not in present)
        for value, group in groups:
            rows.append({"stage": stage, group_col: value, **_metrics(group)})
    return pd.DataFrame(rows, columns=["stage", group_col, *_METRIC_COLUMNS])


def _score_tails(frame: pd.DataFrame, tail_bounds: Sequence[tuple[str, float, float]]) -> pd.DataFrame:
    output = frame.copy()
    output["score_tail"] = pd.NA
    score = output["score"].to_numpy(dtype=np.float64, copy=False)
    finite_positions = np.flatnonzero(np.isfinite(score))
    if not len(finite_positions):
        return output
    ordered = finite_positions[np.argsort(-score[finite_positions], kind="stable")]
    for name, lower, upper in tail_bounds:
        if not 0.0 <= lower < upper <= 1.0:
            raise ValueError("score tail bounds must satisfy 0 <= lower < upper <= 1")
        start = int(np.ceil(lower * len(ordered)))
        end = int(np.ceil(upper * len(ordered)))
        output.loc[output.index[ordered[start:end]], "score_tail"] = name
    return output


def grouped_transfer_reports(
    staged_ledgers: Mapping[str, pd.DataFrame] | Sequence[tuple[str, pd.DataFrame]],
    *,
    key_cols: Sequence[str] = _DEFAULT_KEY_COLS,
    score_tail_bounds: Sequence[tuple[str, float, float]] = (
        ("top_1", 0.00, 0.01),
        ("top_1_2", 0.01, 0.02),
        ("top_2_5", 0.02, 0.05),
        ("top_5_10", 0.05, 0.10),
    ),
    **column_options: str | None,
) -> dict[str, pd.DataFrame]:
    """Return overall and side/month/archetype/global-score-tail waterfall tables.

    Score tails are ranked globally within each stage's common-row ledger, not
    within side, month, or archetype groups.  The supplied bounds are disjoint
    percentile intervals and therefore avoid overlapping tail accounting.
    """

    normalized = normalize_staged_ledgers(staged_ledgers, key_cols=key_cols, **column_options)
    ledgers = _with_outcome_references(normalized, tuple(key_cols))
    overall = pd.DataFrame([{"stage": stage, **_metrics(frame)} for stage, frame in ledgers.items()])
    month_ledgers: dict[str, pd.DataFrame] = {}
    tail_ledgers: dict[str, pd.DataFrame] = {}
    for stage, frame in ledgers.items():
        month_frame = frame.copy()
        month_frame["month"] = month_frame["timestamp"].dt.strftime("%Y-%m").fillna("__missing__")
        month_ledgers[stage] = month_frame
        tail_ledgers[stage] = _score_tails(frame, score_tail_bounds)
    return {
        "overall": overall,
        "side": _grouped_metrics(ledgers, "side"),
        "month": _grouped_metrics(month_ledgers, "month"),
        "archetype": _grouped_metrics(ledgers, "archetype"),
        "global_score_tail": _grouped_metrics(
            tail_ledgers, "score_tail", include_empty_tails=[name for name, _, _ in score_tail_bounds]
        ),
    }


# A short name for callers that want the complete report rather than a single table.
transfer_waterfall_report = grouped_transfer_reports


__all__ = [
    "common_key_intersection",
    "enforce_common_key_intersection",
    "grouped_transfer_reports",
    "normalize_stage_ledger",
    "normalize_staged_ledgers",
    "stage_waterfall_metrics",
    "transfer_waterfall_report",
]
