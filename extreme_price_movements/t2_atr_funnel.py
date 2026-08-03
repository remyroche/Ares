"""Exact H12, ATR-normalised soft triple-barrier labels for the T2 funnel.

This module is deliberately separate from the original G0 screen.  Changed
barriers must be reconstructed from ordered one-minute OHLC paths: endpoint
summaries cannot establish first-touch order.  All functions here produce
*training labels only*; no path-derived value is an inference feature.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Iterable

import numpy as np
import pandas as pd


class T2FunnelError(ValueError):
    """Raised when the exact path or causal contract is not satisfied."""


@dataclass(frozen=True)
class BarrierGeometry:
    tp_atr: float
    sl_atr: float

    @property
    def name(self) -> str:
        return f"TP{self.tp_atr:g}_SL{self.sl_atr:g}"


GEOMETRIES = tuple(BarrierGeometry(tp, sl) for tp in (2.0, 3.0) for sl in (1.0, 2.0))
STATE_COLUMNS = ("upper_first", "lower_first", "timeout")


def geometry_by_name(name: str) -> BarrierGeometry:
    for geometry in GEOMETRIES:
        if geometry.name == name:
            return geometry
    raise T2FunnelError(f"unknown fixed T2 geometry: {name}")


def _path_arrays(serialised: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = json.loads(serialised)
    try:
        high = np.asarray(path["high"], dtype=float)
        low = np.asarray(path["low"], dtype=float)
        close = np.asarray(path["close"], dtype=float)
    except (KeyError, TypeError, ValueError) as exc:
        raise T2FunnelError("invalid H12 minute path") from exc
    if len(high) != 720 or len(low) != 720 or len(close) != 720:
        raise T2FunnelError("the T2 contract requires exactly 720 one-minute bars")
    if not (np.isfinite(high).all() and np.isfinite(low).all() and np.isfinite(close).all()):
        raise T2FunnelError("H12 minute path contains non-finite prices")
    return high, low, close


def _first_index(mask: np.ndarray) -> int:
    hits = np.flatnonzero(mask)
    return int(hits[0]) if len(hits) else -1


def materialize_geometry_events_bulk(
    paths: pd.DataFrame,
    geometries: Iterable[BarrierGeometry] = GEOMETRIES,
) -> dict[str, pd.DataFrame]:
    """Decode each H12 path once and emit event dictionaries for all grids."""
    geometries = tuple(geometries)
    if not geometries:
        raise T2FunnelError("at least one geometry is required")
    required = {"candidate_id", "side_name", "execution_future_path", "atr_1h", "decision_price"}
    missing = sorted(required - set(paths.columns))
    if missing:
        raise T2FunnelError(f"path surface lacks columns: {missing}")
    if paths.candidate_id.duplicated().any():
        raise T2FunnelError("path candidate IDs must be unique")
    records: dict[str, list[dict[str, object]]] = {geometry.name: [] for geometry in geometries}
    ordered_columns = ["candidate_id", "atr_1h", "decision_price", "execution_future_path", "side_name"]
    for row in paths.loc[:, ordered_columns].itertuples(index=False):
        candidate_id, atr, price, serialised, side = row
        atr = float(atr)
        price = float(price)
        if not np.isfinite(atr) or atr <= 0.0 or not np.isfinite(price) or price <= 0.0:
            raise T2FunnelError("entry ATR fraction and decision price must be positive finite values")
        side_text = str(side).lower()
        if side_text not in {"long", "short"}:
            raise T2FunnelError(f"unsupported side: {side}")
        high, low, close = _path_arrays(serialised)
        if side_text == "long":
            favourable = high / price - 1.0
            adverse = low / price - 1.0
            terminal_atr = (close[-1] / price - 1.0) / atr
        else:
            # Exact stated contract: side_sign * (price / entry - 1), not
            # a reciprocal return.  The latter would silently alter both
            # barrier locations and terminal softness for short candidates.
            favourable = 1.0 - low / price
            adverse = 1.0 - high / price
            terminal_atr = (1.0 - close[-1] / price) / atr
        for geometry in geometries:
            upper_minute = _first_index(favourable >= geometry.tp_atr * atr)
            lower_minute = _first_index(adverse <= -geometry.sl_atr * atr)
            # Tie is adverse/lower by the project-wide minute-bar convention.
            upper_first = upper_minute >= 0 and (lower_minute < 0 or upper_minute < lower_minute)
            lower_first = lower_minute >= 0 and (upper_minute < 0 or lower_minute <= upper_minute)
            records[geometry.name].append({
                "candidate_id": candidate_id,
                "geometry": geometry.name,
                "upper_first": float(upper_first),
                "lower_first": float(lower_first),
                "timeout": float(not upper_first and not lower_first),
                "upper_first_minute": upper_minute,
                "lower_first_minute": lower_minute,
                "same_minute_conflict": bool(upper_minute >= 0 and upper_minute == lower_minute),
                "terminal_atr": float(terminal_atr),
                "entry_atr_fraction": atr,
            })
    results: dict[str, pd.DataFrame] = {}
    for geometry in geometries:
        result = pd.DataFrame.from_records(records[geometry.name])
        state_sum = result.loc[:, list(STATE_COLUMNS)].sum(axis=1)
        if not np.array_equal(state_sum.to_numpy(int), np.ones(len(result), dtype=int)):
            raise T2FunnelError("geometry events are not mutually exclusive and exhaustive")
        results[geometry.name] = result
    return results


def materialize_geometry_events(paths: pd.DataFrame, geometry: BarrierGeometry) -> pd.DataFrame:
    """Materialise one geometry; prefer the bulk API for a grid."""
    return materialize_geometry_events_bulk(paths, (geometry,))[geometry.name]


def materialize_contract_events_bulk(
    paths: pd.DataFrame,
    contracts: Iterable[tuple[str, BarrierGeometry, int]],
) -> dict[str, pd.DataFrame]:
    """One-pass label materialisation for nearby barrier/horizon contracts."""
    contracts = tuple(contracts)
    if not contracts or any(minutes not in (480, 720) for _, _, minutes in contracts):
        raise T2FunnelError("certainty contracts require H8 or H12 frozen paths")
    required = {"candidate_id", "side_name", "execution_future_path", "atr_1h", "decision_price"}
    if required - set(paths.columns):
        raise T2FunnelError("certainty path contract is incomplete")
    records = {name: [] for name, _, _ in contracts}
    cols = ["candidate_id", "atr_1h", "decision_price", "execution_future_path", "side_name"]
    for candidate_id, atr, price, serialised, side in paths.loc[:, cols].itertuples(index=False):
        atr, price = float(atr), float(price)
        high, low, close = _path_arrays(serialised)
        long = str(side).lower() == "long"
        if not long and str(side).lower() != "short":
            raise T2FunnelError("unsupported side in certainty contract")
        favourable = high / price - 1.0 if long else 1.0 - low / price
        adverse = low / price - 1.0 if long else 1.0 - high / price
        for name, geometry, minutes in contracts:
            fav, adv = favourable[:minutes], adverse[:minutes]
            upper_minute = _first_index(fav >= geometry.tp_atr * atr)
            lower_minute = _first_index(adv <= -geometry.sl_atr * atr)
            upper = upper_minute >= 0 and (lower_minute < 0 or upper_minute < lower_minute)
            lower = lower_minute >= 0 and (upper_minute < 0 or lower_minute <= upper_minute)
            terminal = (close[minutes - 1] / price - 1.0 if long else 1.0 - close[minutes - 1] / price) / atr
            records[name].append({"candidate_id": candidate_id, "contract": name, "upper_first": float(upper), "lower_first": float(lower), "timeout": float(not upper and not lower), "upper_first_minute": upper_minute, "lower_first_minute": lower_minute, "same_minute_conflict": bool(upper_minute >= 0 and upper_minute == lower_minute), "terminal_atr": float(terminal), "path_completeness": 1.0})
    return {name: pd.DataFrame.from_records(values) for name, values in records.items()}


def soft_event_targets(events: pd.DataFrame, geometry: BarrierGeometry, *, temperature_atr: float) -> np.ndarray:
    """Return soft P(upper, lower, timeout) labels for a fixed geometry.

    First touches remain exact.  Only H12 timeouts are softened from their
    final ATR-normalised position; distances are expressed in *each row's*
    entry ATR unit, making the geometry invariant to average raw volatility.
    """
    if temperature_atr <= 0.0 or not np.isfinite(temperature_atr):
        raise T2FunnelError("temperature_atr must be positive and finite")
    missing = sorted((set(STATE_COLUMNS) | {"terminal_atr"}) - set(events.columns))
    if missing:
        raise T2FunnelError(f"event surface lacks columns: {missing}")
    hard = events.loc[:, list(STATE_COLUMNS)].to_numpy(float)
    if not np.allclose(hard.sum(axis=1), 1.0, rtol=0.0, atol=1e-8):
        raise T2FunnelError("event states must sum to one")
    result = hard.copy()
    timeout = hard[:, 2] > 0.5
    if timeout.any():
        terminal = events.loc[:, "terminal_atr"].to_numpy(float)
        upper_distance = np.maximum(geometry.tp_atr - terminal, 0.0)
        lower_distance = np.maximum(terminal + geometry.sl_atr, 0.0)
        logits = np.column_stack((
            -upper_distance / temperature_atr,
            -lower_distance / temperature_atr,
            -np.abs(upper_distance - lower_distance) / temperature_atr,
        ))
        logits -= logits.max(axis=1, keepdims=True)
        weights = np.exp(np.clip(logits, -60.0, 60.0))
        result[timeout] = weights[timeout] / weights[timeout].sum(axis=1, keepdims=True)
    if not np.allclose(result.sum(axis=1), 1.0, rtol=0.0, atol=1e-7):
        raise T2FunnelError("soft barrier probabilities do not sum to one")
    return result.astype(np.float32)


def top_book_metrics(frame: pd.DataFrame, *, score_column: str, group_columns: Iterable[str] = ()) -> pd.DataFrame:
    """Evaluate one pooled-global common-bps book, optionally by attribution."""
    groups = list(group_columns)
    frames = [((), frame)] if not groups else list(frame.groupby(groups, observed=True, sort=True))
    rows: list[dict[str, object]] = []
    for key, group in frames:
        key = key if isinstance(key, tuple) else (key,)
        ordered = group.sort_values([score_column, "candidate_id"], ascending=[False, True], kind="mergesort")
        for fraction in (0.01, 0.05, 0.10, 0.20):
            selected = ordered.head(max(1, int(np.ceil(len(ordered) * fraction))))
            row = {
                "top_fraction": fraction,
                "population_rows": len(ordered),
                "selected_rows": len(selected),
                "gross_bps_per_trade": selected.execution_gross_ev_12h.mean() * 10_000.0,
                "cost_bps_per_trade": selected.execution_cost_return.mean() * 10_000.0,
                "net_bps_per_trade": selected.execution_net_ev_12h.mean() * 10_000.0,
            }
            row.update(dict(zip(groups, key)))
            rows.append(row)
    return pd.DataFrame(rows)
