"""Causal, size-dependent L2 execution confirmation for historical replays.

This module is deliberately a supplementary confirmation layer.  It never
fills missing books from another symbol or time and it leaves uncovered trades
without an L2 cost adjustment.  Book walking delegates to the live inference
implementation so research and inference use identical level semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd

from extreme_price_movements.inference.liquidity_precheck import _walk_levels


@dataclass(frozen=True)
class L2ConfirmationConfig:
    """Settings for causal snapshot selection and executable depth walking."""

    max_snapshot_age: pd.Timedelta = pd.Timedelta(minutes=75)
    max_walk_slippage_bps: float = 100.0


_OUTPUT_COLUMNS = (
    "l2_entry_snapshot_covered",
    "l2_exit_snapshot_covered",
    "l2_roundtrip_snapshot_covered",
    "l2_entry_covered",
    "l2_exit_covered",
    "l2_roundtrip_covered",
    "l2_entry_snapshot_observed_ts",
    "l2_exit_snapshot_observed_ts",
    "l2_entry_snapshot_age_minutes",
    "l2_exit_snapshot_age_minutes",
    "l2_admitted_quote_notional",
    "l2_admitted_quantity",
    "l2_entry_capacity_quote",
    "l2_exit_capacity_quote",
    "l2_entry_capacity_ratio",
    "l2_exit_capacity_ratio",
    "l2_entry_fill_price",
    "l2_exit_fill_price",
    "l2_entry_depth_slippage_bps",
    "l2_exit_depth_slippage_bps",
    "l2_roundtrip_depth_slippage_bps",
    "l2_confirmation_reason",
)


def _utc(value: Any) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if pd.isna(ts):
        return pd.NaT
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def _normalise_books(levels: pd.DataFrame) -> pd.DataFrame:
    work = levels.reset_index(drop=False).copy()
    if "observed_ts" not in work:
        raise ValueError("L2 levels must contain observed_ts")
    required = {"symbol", "side", "price", "qty"}
    missing = sorted(required.difference(work.columns))
    if missing:
        raise ValueError(f"L2 levels missing columns: {missing}")
    work["observed_ts"] = pd.to_datetime(work["observed_ts"], utc=True, errors="coerce")
    work["price"] = pd.to_numeric(work["price"], errors="coerce")
    work["qty"] = pd.to_numeric(work["qty"], errors="coerce")
    work["side"] = work["side"].astype(str).str.lower()
    work = work[
        work["observed_ts"].notna()
        & work["price"].gt(0)
        & work["qty"].gt(0)
        & work["side"].isin(("bid", "bids", "ask", "asks"))
    ].copy()
    work["book_side"] = work["side"].str.rstrip("s")
    return work.sort_values(["symbol", "observed_ts"], kind="mergesort")


def _latest_causal_snapshot(
    books: pd.DataFrame,
    *,
    symbol: str,
    execution_ts: pd.Timestamp,
    max_age: pd.Timedelta,
) -> tuple[Optional[pd.DataFrame], str, Optional[pd.Timestamp], float]:
    if pd.isna(execution_ts):
        return None, "invalid_execution_timestamp", None, float("nan")
    candidates = books[
        (books["symbol"].astype(str) == str(symbol))
        & (books["observed_ts"] <= execution_ts)
    ]
    if candidates.empty:
        return None, "no_causally_prior_snapshot", None, float("nan")
    observed_ts = candidates["observed_ts"].max()
    age = execution_ts - observed_ts
    age_minutes = float(age.total_seconds() / 60.0)
    if age < pd.Timedelta(0):  # Defensive; the filter above is the causal contract.
        return None, "future_snapshot", observed_ts, age_minutes
    if age > max_age:
        return None, "stale_snapshot", observed_ts, age_minutes
    snapshot = candidates[candidates["observed_ts"] == observed_ts]
    if not {"bid", "ask"}.issubset(set(snapshot["book_side"])):
        return None, "incomplete_snapshot", observed_ts, age_minutes
    return snapshot, "covered", observed_ts, age_minutes


def _walk_leg(
    snapshot: pd.DataFrame,
    *,
    action: str,
    intended_quote: float,
    max_slippage_bps: float,
) -> dict[str, float]:
    buy = action == "buy"
    book_side = "ask" if buy else "bid"
    side = "long" if buy else "short"
    selected = snapshot[snapshot["book_side"] == book_side].copy()
    selected = selected.sort_values("price", ascending=buy, kind="mergesort")
    levels = list(selected[["price", "qty"]].itertuples(index=False, name=None))
    best_touch = float(selected.iloc[0]["price"])
    max_walk_price = best_touch * (
        1.0 + max_slippage_bps / 10000.0
        if buy
        else 1.0 - max_slippage_bps / 10000.0
    )

    # Infinite demand measures all capacity inside the configured price band.
    capacity_quote, _capacity_base, _ = _walk_levels(
        levels,
        side=side,
        best_touch=best_touch,
        max_walk_price=max_walk_price,
        intended_quote_size=float("inf"),
    )
    filled_quote, filled_base, fill_price = _walk_levels(
        levels,
        side=side,
        best_touch=best_touch,
        max_walk_price=max_walk_price,
        intended_quote_size=float(intended_quote),
    )
    if np.isfinite(fill_price) and fill_price > 0:
        slippage_bps = (
            (fill_price / best_touch - 1.0) * 10000.0
            if buy
            else (1.0 - fill_price / best_touch) * 10000.0
        )
    else:
        slippage_bps = float("nan")
    return {
        "capacity_quote": float(capacity_quote),
        "capacity_ratio": float(capacity_quote / max(intended_quote, 1e-12)),
        "filled_quote": float(filled_quote),
        "filled_base": float(filled_base),
        "fill_price": float(fill_price),
        "slippage_bps": float(slippage_bps),
    }


def confirm_l2_execution(
    trades: pd.DataFrame,
    orderbook_levels: pd.DataFrame,
    *,
    config: L2ConfirmationConfig = L2ConfirmationConfig(),
    symbol_col: str = "symbol",
    side_col: str = "side",
    entry_ts_col: str = "entry_ts",
    exit_ts_col: str = "exit_ts",
    admitted_quote_col: str = "admitted_quote_notional",
    admitted_quantity_col: str = "admitted_quantity",
) -> pd.DataFrame:
    """Return per-trade causal L2 entry/exit confirmation diagnostics.

    When both sizing fields exist, entry uses the admitted quote notional while
    exit marks the held base quantity at the exit touch. If only one field is
    available it is used for both legs. A round-trip depth cost is emitted only
    when both legs are covered and fully executable.
    """

    required = {symbol_col, side_col, entry_ts_col, exit_ts_col}
    missing = sorted(required.difference(trades.columns))
    if missing:
        raise ValueError(f"Trades missing columns: {missing}")
    if admitted_quote_col not in trades and admitted_quantity_col not in trades:
        raise ValueError("Trades need admitted quote notional or admitted quantity")
    if config.max_snapshot_age < pd.Timedelta(0):
        raise ValueError("max_snapshot_age must be non-negative")
    if not np.isfinite(config.max_walk_slippage_bps) or config.max_walk_slippage_bps < 0:
        raise ValueError("max_walk_slippage_bps must be finite and non-negative")

    books = _normalise_books(orderbook_levels)
    records: list[dict[str, Any]] = []
    for _, trade in trades.iterrows():
        symbol = str(trade[symbol_col])
        position_side = str(trade[side_col]).lower()
        if position_side not in {"long", "short"}:
            raise ValueError(f"Unsupported position side: {position_side!r}")
        entry_ts, exit_ts = _utc(trade[entry_ts_col]), _utc(trade[exit_ts_col])
        quote = pd.to_numeric(trade.get(admitted_quote_col), errors="coerce")
        quantity = pd.to_numeric(trade.get(admitted_quantity_col), errors="coerce")
        quote = float(quote) if np.isfinite(quote) and quote > 0 else float("nan")
        quantity = (
            float(quantity) if np.isfinite(quantity) and quantity > 0 else float("nan")
        )

        entry_book, entry_reason, entry_obs, entry_age = _latest_causal_snapshot(
            books, symbol=symbol, execution_ts=entry_ts, max_age=config.max_snapshot_age
        )
        exit_book, exit_reason, exit_obs, exit_age = _latest_causal_snapshot(
            books, symbol=symbol, execution_ts=exit_ts, max_age=config.max_snapshot_age
        )
        row: dict[str, Any] = {name: np.nan for name in _OUTPUT_COLUMNS}
        row.update(
            {
                "l2_entry_snapshot_covered": entry_book is not None,
                "l2_exit_snapshot_covered": exit_book is not None,
                "l2_roundtrip_snapshot_covered": (
                    entry_book is not None and exit_book is not None
                ),
                "l2_entry_covered": False,
                "l2_exit_covered": False,
                "l2_roundtrip_covered": False,
                "l2_entry_snapshot_observed_ts": entry_obs,
                "l2_exit_snapshot_observed_ts": exit_obs,
                "l2_entry_snapshot_age_minutes": entry_age,
                "l2_exit_snapshot_age_minutes": exit_age,
                "l2_admitted_quote_notional": quote,
                "l2_admitted_quantity": quantity,
                "l2_confirmation_reason": f"entry:{entry_reason};exit:{exit_reason}",
            }
        )

        leg_results: dict[str, dict[str, float]] = {}
        for leg, snapshot, action in (
            ("entry", entry_book, "buy" if position_side == "long" else "sell"),
            ("exit", exit_book, "sell" if position_side == "long" else "buy"),
        ):
            if snapshot is None:
                continue
            book_side = "ask" if action == "buy" else "bid"
            touches = snapshot[snapshot["book_side"] == book_side]["price"]
            best_touch = float(touches.min() if action == "buy" else touches.max())
            if leg == "entry" and np.isfinite(quote):
                intended_quote = quote
            elif np.isfinite(quantity):
                intended_quote = quantity * best_touch
            else:
                intended_quote = quote
            result = _walk_leg(
                snapshot,
                action=action,
                intended_quote=float(intended_quote),
                max_slippage_bps=float(config.max_walk_slippage_bps),
            )
            fully_executable = result["filled_quote"] >= intended_quote * (1.0 - 1e-12)
            row[f"l2_{leg}_covered"] = bool(fully_executable)
            row[f"l2_{leg}_capacity_quote"] = result["capacity_quote"]
            row[f"l2_{leg}_capacity_ratio"] = result["capacity_ratio"]
            row[f"l2_{leg}_fill_price"] = result["fill_price"]
            row[f"l2_{leg}_depth_slippage_bps"] = result["slippage_bps"]
            leg_results[leg] = result

        roundtrip = bool(row["l2_entry_covered"] and row["l2_exit_covered"])
        row["l2_roundtrip_covered"] = roundtrip
        if roundtrip:
            row["l2_roundtrip_depth_slippage_bps"] = float(
                leg_results["entry"]["slippage_bps"]
                + leg_results["exit"]["slippage_bps"]
            )
            row["l2_confirmation_reason"] = "covered"
        elif entry_book is not None and exit_book is not None:
            failed_legs = [
                leg for leg in ("entry", "exit") if not row[f"l2_{leg}_covered"]
            ]
            row["l2_confirmation_reason"] = (
                "insufficient_capacity:" + ",".join(failed_legs)
            )
        records.append(row)

    return pd.DataFrame(records, index=trades.index, columns=_OUTPUT_COLUMNS)


def summarise_l2_confirmation(diagnostics: pd.DataFrame) -> dict[str, float]:
    """Aggregate coverage, capacity, and depth-cost diagnostics."""

    n = len(diagnostics)
    covered = diagnostics.get("l2_roundtrip_covered", pd.Series(False, index=diagnostics.index)).fillna(False)
    confirmed = diagnostics.loc[covered]
    return {
        "trade_count": float(n),
        "entry_snapshot_coverage_rate": float(diagnostics["l2_entry_snapshot_covered"].mean()) if n else 0.0,
        "exit_snapshot_coverage_rate": float(diagnostics["l2_exit_snapshot_covered"].mean()) if n else 0.0,
        "roundtrip_snapshot_coverage_rate": float(diagnostics["l2_roundtrip_snapshot_covered"].mean()) if n else 0.0,
        "entry_coverage_rate": float(diagnostics["l2_entry_covered"].mean()) if n else 0.0,
        "exit_coverage_rate": float(diagnostics["l2_exit_covered"].mean()) if n else 0.0,
        "roundtrip_coverage_rate": float(covered.mean()) if n else 0.0,
        "confirmed_trade_count": float(covered.sum()),
        "median_entry_capacity_ratio": float(confirmed["l2_entry_capacity_ratio"].median()) if len(confirmed) else float("nan"),
        "median_exit_capacity_ratio": float(confirmed["l2_exit_capacity_ratio"].median()) if len(confirmed) else float("nan"),
        "mean_entry_depth_slippage_bps": float(confirmed["l2_entry_depth_slippage_bps"].mean()) if len(confirmed) else float("nan"),
        "mean_exit_depth_slippage_bps": float(confirmed["l2_exit_depth_slippage_bps"].mean()) if len(confirmed) else float("nan"),
        "mean_roundtrip_depth_slippage_bps": float(confirmed["l2_roundtrip_depth_slippage_bps"].mean()) if len(confirmed) else float("nan"),
    }


def apply_confirmed_l2_cost(
    baseline_net_return: pd.Series,
    diagnostics: pd.DataFrame,
) -> pd.DataFrame:
    """Subtract observed round-trip L2 depth cost only for confirmed trades.

    Returns both the adjusted return and an explicit application mask.  Returns
    are decimal fractions; L2 diagnostics are basis points.  Uncovered rows are
    copied byte-for-byte from the baseline rather than imputed or extrapolated.
    """

    baseline = pd.to_numeric(baseline_net_return, errors="coerce").reindex(
        diagnostics.index
    )
    covered = diagnostics["l2_roundtrip_covered"].fillna(False).astype(bool)
    cost = pd.to_numeric(
        diagnostics["l2_roundtrip_depth_slippage_bps"], errors="coerce"
    )
    applicable = covered & cost.notna()
    adjusted = baseline.copy()
    adjusted.loc[applicable] = baseline.loc[applicable] - cost.loc[applicable] / 10000.0
    return pd.DataFrame(
        {
            "l2_adjusted_net_return": adjusted,
            "l2_cost_applied": applicable,
            "l2_incremental_cost_bps": cost.where(applicable),
        },
        index=diagnostics.index,
    )
