"""Portable exact-global-book evaluation for OOF candidate predictions.

All tail metrics are formed once per configuration from the pooled candidate
population.  Side, month, regime, cost, liquidity and hurdle tables are
descriptive attribution of that fixed book; none can modify selection.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


FRACTIONS = (0.01, 0.05, 0.10, 0.20)


class EvaluationError(ValueError):
    pass


@dataclass(frozen=True)
class TailGate:
    min_side_share: float = 0.05
    min_side_rows: int = 20
    require_nonnegative_side_net: bool = True

    def manifest(self) -> dict[str, object]:
        return asdict(self)


def stable_global_top_k(frame: pd.DataFrame, score_column: str, fraction: float) -> pd.DataFrame:
    if not 0.0 < float(fraction) <= 1.0:
        raise EvaluationError("fraction must be in (0, 1]")
    required = {"candidate_id", score_column}
    missing = required - set(frame.columns)
    if missing:
        raise EvaluationError(f"global selection missing columns: {sorted(missing)}")
    score = pd.to_numeric(frame[score_column], errors="coerce")
    if score.isna().any():
        raise EvaluationError("global selection score must be finite")
    selected_rows = max(1, int(np.ceil(len(frame) * float(fraction))))
    return frame.assign(__candidate_score__=score).sort_values(
        ["__candidate_score__", "candidate_id"], ascending=[False, True], kind="mergesort"
    ).head(selected_rows).drop(columns="__candidate_score__")


def _outcome_bps(frame: pd.DataFrame, column: str, unit: str) -> np.ndarray:
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(float)
    if not np.isfinite(values).all():
        raise EvaluationError(f"outcome {column} must be finite")
    if unit not in {"return", "bps"}:
        raise EvaluationError("outcome unit must be return or bps")
    return values * (10_000.0 if unit == "return" else 1.0)


def _summary(selected: pd.DataFrame, *, net_column: str, net_unit: str, gross_column: str | None, gross_unit: str | None, cost_column: str | None, cost_unit: str | None) -> dict[str, float | int]:
    net = _outcome_bps(selected, net_column, net_unit)
    result: dict[str, float | int] = {
        "selected_rows": int(len(selected)),
        "net_bps": float(net.mean()),
        "net_median_bps": float(np.median(net)),
        "net_positive_rate": float((net > 0.0).mean()),
        "net_cvar05_bps": float(np.sort(net)[: max(1, int(np.ceil(.05 * len(net))))].mean()),
    }
    if gross_column is not None:
        result["gross_bps"] = float(_outcome_bps(selected, gross_column, str(gross_unit)).mean())
    if cost_column is not None:
        result["cost_bps"] = float(_outcome_bps(selected, cost_column, str(cost_unit)).mean())
    return result


def _attribution(selected: pd.DataFrame, *, dimension: str, net_column: str, net_unit: str, gross_column: str | None, gross_unit: str | None, cost_column: str | None, cost_unit: str | None) -> pd.DataFrame:
    if dimension not in selected:
        return pd.DataFrame()
    result: list[dict[str, object]] = []
    for value, group in selected.groupby(dimension, dropna=False, observed=True, sort=True):
        result.append({"dimension": dimension, "dimension_value": "<missing>" if pd.isna(value) else str(value), "selected_share": float(len(group) / len(selected)), **_summary(group, net_column=net_column, net_unit=net_unit, gross_column=gross_column, gross_unit=gross_unit, cost_column=cost_column, cost_unit=cost_unit)})
    return pd.DataFrame(result)


def _bucket_column(frame: pd.DataFrame, column: str, name: str) -> pd.Series:
    value = pd.to_numeric(frame[column], errors="coerce")
    if value.notna().sum() == 0:
        return pd.Series("<missing>", index=frame.index)
    if value.nunique(dropna=True) <= 8:
        return value.round(8).astype("string").fillna("<missing>")
    return pd.qcut(value, q=4, duplicates="drop").astype("string").fillna("<missing>").rename(name)


def evaluate_global_book(
    frame: pd.DataFrame,
    *,
    score_column: str,
    net_column: str,
    net_unit: str,
    gross_column: str | None = None,
    gross_unit: str | None = None,
    cost_column: str | None = None,
    cost_unit: str | None = None,
    fractions: Sequence[float] = FRACTIONS,
    regime_column: str | None = None,
    liquidity_column: str | None = None,
    hurdle_column: str | None = None,
) -> tuple[pd.DataFrame, Mapping[str, pd.DataFrame]]:
    """Evaluate a single score configuration with no quota/replacement path."""
    if frame.empty:
        raise EvaluationError("cannot evaluate an empty candidate population")
    work = frame.copy()
    work["__ts__"] = pd.to_datetime(work["__ts__"], utc=True, errors="raise")
    work["__month__"] = work["__ts__"].dt.to_period("M").astype(str)
    optional: list[tuple[str, str | None]] = [("side", "side_name"), ("month", "__month__"), ("regime", regime_column)]
    for label, column in (("liquidity", liquidity_column), ("hurdle", hurdle_column), ("cost", cost_column)):
        if column is not None and column in work:
            bucket = f"__{label}_bucket__"
            work[bucket] = _bucket_column(work, column, bucket)
            optional.append((label, bucket))
    summary_rows: list[dict[str, object]] = []
    attribution: dict[str, list[pd.DataFrame]] = {name: [] for name, _ in optional}
    selected_books: dict[float, pd.DataFrame] = {}
    for fraction in fractions:
        selected = stable_global_top_k(work, score_column, float(fraction))
        selected_books[float(fraction)] = selected
        summary_rows.append({"selection_scope": "one_pooled_global_post_score_top_k", "top_fraction": float(fraction), "population_rows": int(len(work)), **_summary(selected, net_column=net_column, net_unit=net_unit, gross_column=gross_column, gross_unit=gross_unit, cost_column=cost_column, cost_unit=cost_unit)})
        for name, column in optional:
            if column is None:
                continue
            local = _attribution(selected, dimension=column, net_column=net_column, net_unit=net_unit, gross_column=gross_column, gross_unit=gross_unit, cost_column=cost_column, cost_unit=cost_unit)
            if not local.empty:
                local.insert(0, "top_fraction", float(fraction))
                attribution[name].append(local)
    return pd.DataFrame(summary_rows), {name: pd.concat(parts, ignore_index=True) if parts else pd.DataFrame() for name, parts in attribution.items()}


def tail_gates(tails: pd.DataFrame, side_attribution: pd.DataFrame, *, gate: TailGate = TailGate()) -> pd.DataFrame:
    """Diagnostic top-tail reversal and side-exposure gates at the 10% book."""
    by_fraction = tails.set_index("top_fraction")
    required = set(FRACTIONS) & set(by_fraction.index)
    if not required:
        raise EvaluationError("tail summary contains no standard fractions")
    net = {fraction: float(by_fraction.loc[fraction, "net_bps"]) for fraction in required}
    ordered = [fraction for fraction in FRACTIONS if fraction in net]
    reversals = [net[left] < net[right] for left, right in zip(ordered, ordered[1:], strict=False)]
    top = side_attribution.loc[side_attribution.top_fraction.eq(.10)].copy() if not side_attribution.empty else pd.DataFrame()
    if top.empty:
        side_rows = 0
        side_gate = False
        min_share = np.nan
        min_net = np.nan
    else:
        side_rows = int(len(top))
        min_share = float(top.selected_share.min())
        min_net = float(top.net_bps.min())
        side_gate = bool((top.selected_rows >= gate.min_side_rows).all() and min_share >= gate.min_side_share and (not gate.require_nonnegative_side_net or min_net >= 0.0))
    return pd.DataFrame([{
        "gate_scope": "diagnostic_only_no_selection_change", "top_tail_reversal_detected": bool(any(reversals)),
        "top_tail_no_reversal_gate": bool(not any(reversals)), "side_attribution_groups": side_rows,
        "min_side_share": min_share, "min_side_net_bps": min_net, "side_gate_pass": side_gate,
        "promotion_authorized": False, **gate.manifest(),
    }])


def paired_day_block_bootstrap(
    baseline_selected: pd.DataFrame,
    challenger_selected: pd.DataFrame,
    *,
    net_column: str,
    net_unit: str,
    block_days: int = 5,
    replicates: int = 2000,
    seed: int = 20260801,
) -> Mapping[str, float | int | str]:
    """Paired circular UTC-day-block bootstrap on two frozen selected books.

    Each arm is selected once before resampling.  Calendar days with no selected
    candidates carry zero contribution, so the paired estimand respects a
    globally-selected book's differing day concentration rather than silently
    discarding non-trading days.
    """
    if block_days < 1 or replicates < 1:
        raise EvaluationError("bootstrap block_days and replicates must be positive")
    def daily(frame: pd.DataFrame) -> pd.Series:
        timestamp = pd.to_datetime(frame["__ts__"], utc=True, errors="raise")
        values = pd.Series(_outcome_bps(frame, net_column, net_unit), index=timestamp.dt.floor("D"))
        return values.groupby(level=0).sum()
    left, right = daily(baseline_selected), daily(challenger_selected)
    start, end = min(left.index.min(), right.index.min()), max(left.index.max(), right.index.max())
    days = pd.date_range(start, end, freq="D", tz="UTC")
    delta = (right.reindex(days, fill_value=0.0) - left.reindex(days, fill_value=0.0)).to_numpy(float)
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, len(delta), size=(replicates, int(np.ceil(len(delta) / block_days))))
    offsets = np.arange(block_days)
    sampled = (starts[:, :, None] + offsets).reshape(replicates, -1) % len(delta)
    draw = delta[sampled[:, : len(delta)]].mean(axis=1)
    return {
        "bootstrap_contract": "frozen independently pooled-global selected books; paired circular UTC-day-block resampling; zero contribution on non-selected calendar days",
        "calendar_days": int(len(days)), "block_days": int(block_days), "replicates": int(replicates), "seed": int(seed),
        "mean_daily_net_delta_bps": float(delta.mean()), "ci95_low_bps": float(np.quantile(draw, .025)), "ci95_high_bps": float(np.quantile(draw, .975)), "p_delta_le_zero": float((draw <= 0.0).mean()),
    }
