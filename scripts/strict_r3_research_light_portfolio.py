"""Dependency-light fixed auction used only by Strict-R3 offline research.

This module deliberately implements the *fixed* controlled-portfolio contract
used by the enhanced-base research: long only, 7x leverage, 10% wallet margin
slots, 80% total margin capacity, eight concurrent positions, two new entries
per decision timestamp and one open position per symbol.  It is not a general
portfolio engine and it has no live or exchange path.

The narrow implementation exists because importing the broad production
portfolio replay also loads optional model/parity machinery.  Research callers
use this only after candidate scores, MC1 mapping and policy outcomes have
already been materialised.  Keeping this module dependency-light makes the
terminal constrained replay deterministic and prevents optional imports from
blocking an otherwise complete OOS experiment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


INITIAL_WALLET = 1_000.0
LEVERAGE = 7.0
MARGIN_SLOT_FRACTION = 0.10
TOTAL_MARGIN_FRACTION = 0.80
MAX_CONCURRENT = 8
MAX_NEW_PER_TIMESTAMP = 2


@dataclass
class _Position:
    symbol: str
    entry_timestamp: pd.Timestamp
    exit_timestamp: pd.Timestamp
    size: float
    net_return: float
    gross_return: float
    exit_reason: str
    exit_price: float


def _progress(position: _Position, timestamp: pd.Timestamp) -> float:
    total_seconds = max(
        float((position.exit_timestamp - position.entry_timestamp).total_seconds()),
        1.0,
    )
    elapsed = float((timestamp - position.entry_timestamp).total_seconds())
    return float(np.clip(elapsed / total_seconds, 0.0, 1.0))


def _marked_notional(positions: list[_Position], timestamp: pd.Timestamp) -> float:
    return float(sum(
        position.size * max(1.0 + position.gross_return * _progress(position, timestamp), 0.0)
        for position in positions
    ))


def _allocated_margin(positions: list[_Position], timestamp: pd.Timestamp) -> float:
    return float(sum(
        position.size
        * max(1.0 + position.gross_return * _progress(position, timestamp), 0.0)
        / LEVERAGE
        for position in positions
    ))


def _unrealised_pnl(positions: list[_Position], timestamp: pd.Timestamp) -> float:
    return float(sum(
        position.size * position.net_return * _progress(position, timestamp)
        for position in positions
    ))


def _require_candidates(candidates: pd.DataFrame) -> pd.DataFrame:
    required = {
        "timestamp", "candidate_id", "symbol", "normalized_rank_score",
        "calibrated_score", "exit_timestamp", "net_return", "gross_return",
        "exit_reason", "exit_price",
    }
    missing = sorted(required.difference(candidates.columns))
    if missing:
        raise ValueError(f"light research auction missing columns: {missing}")
    frame = candidates.copy()
    for field in ("timestamp", "exit_timestamp"):
        frame[field] = pd.to_datetime(frame[field], utc=True, errors="coerce")
    for field in ("normalized_rank_score", "calibrated_score", "net_return", "gross_return", "exit_price"):
        frame[field] = pd.to_numeric(frame[field], errors="coerce")
    frame["symbol"] = frame["symbol"].astype(str)
    frame["candidate_id"] = frame["candidate_id"].astype(str)
    frame["exit_reason"] = frame["exit_reason"].astype(str)
    frame = frame.dropna(subset=["timestamp", "exit_timestamp", "normalized_rank_score", "net_return", "gross_return"])
    key = ["timestamp", "symbol", "candidate_id"]
    if frame.duplicated(key).any():
        raise ValueError("light research auction received duplicate candidate identities")
    return frame.sort_values(["timestamp", "symbol", "candidate_id"], kind="stable").reset_index(drop=True)


def replay_fixed_controlled_auction(
    candidates: pd.DataFrame,
    *,
    initial_wallet: float = INITIAL_WALLET,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Replay the frozen controlled 7x / 10%-slot research portfolio.

    Ordering is exactly the fixed auction ordering in the production replay
    under its controlled parameters: rank, then calibrated score, then the
    normalised lexical candidate order.  The caller is responsible for the
    causal admission mask; this function only applies portfolio constraints.
    """

    frame = _require_candidates(candidates)
    wallet = float(initial_wallet)
    positions: list[_Position] = []
    decisions: list[dict[str, Any]] = []
    equity_rows: list[dict[str, Any]] = []

    for timestamp, group in frame.groupby("timestamp", sort=True):
        timestamp = pd.Timestamp(timestamp)
        due = [position for position in positions if position.exit_timestamp <= timestamp]
        positions = [position for position in positions if position.exit_timestamp > timestamp]
        wallet += float(sum(position.size * position.net_return for position in due))
        entries = 0
        group = group.sort_values(
            ["normalized_rank_score", "calibrated_score", "symbol", "candidate_id"],
            ascending=[False, False, True, True],
            kind="stable",
        )
        for _, row in group.iterrows():
            rank = float(row.normalized_rank_score)
            capital_limit = TOTAL_MARGIN_FRACTION * max(wallet, 0.0)
            open_margin = _allocated_margin(positions, timestamp)
            remaining_margin = max(capital_limit - open_margin, 0.0)
            open_notional = _marked_notional(positions, timestamp)
            reason = "accepted"
            accepted = False
            position_size = 0.0
            if not np.isfinite(rank) or rank < 0.0:
                reason = "below_dynamic_threshold"
            elif any(position.symbol == str(row.symbol) for position in positions):
                reason = "symbol_already_open"
            elif len(positions) >= MAX_CONCURRENT:
                reason = "max_concurrent_positions_reached"
            elif entries >= MAX_NEW_PER_TIMESTAMP:
                reason = "max_new_entries_per_strategy_per_bar_reached"
            elif remaining_margin < 1.0:
                reason = "max_capital_allocation_reached"
            else:
                margin = min(MARGIN_SLOT_FRACTION * max(wallet, 0.0), remaining_margin)
                position_size = margin * LEVERAGE
                if position_size < 1.0:
                    reason = "position_size_too_small"
                else:
                    accepted = True
                    entries += 1
                    position = _Position(
                        symbol=str(row.symbol), entry_timestamp=timestamp,
                        exit_timestamp=pd.Timestamp(row.exit_timestamp),
                        size=float(position_size), net_return=float(row.net_return),
                        gross_return=float(row.gross_return), exit_reason=str(row.exit_reason),
                        exit_price=float(row.exit_price),
                    )
                    positions.append(position)
            open_margin_after = _allocated_margin(positions, timestamp)
            decisions.append({
                "candidate_id": str(row.candidate_id), "timestamp": timestamp,
                "symbol": str(row.symbol), "side": "long",
                "strategy_id": "strict_r3_enhanced_live_stack_long",
                "normalized_rank_score": rank, "effective_rank_score": rank,
                "base_threshold": 0.0, "dynamic_threshold": 0.0,
                "portfolio_priority": rank, "accepted": accepted,
                "rejection_reason": reason, "position_size": position_size if accepted else 0.0,
                "open_positions_before": int(len(positions) - (1 if accepted else 0)),
                "open_positions_after": int(len(positions)), "side_count_before": int(len(positions) - (1 if accepted else 0)),
                "strategy_count_before": int(len(positions) - (1 if accepted else 0)),
                "strategy_entries_this_bar_before": int(entries - (1 if accepted else 0)),
                "wallet_before": wallet, "wallet_after": wallet,
                "open_notional_before": open_notional,
                "open_notional_after": _marked_notional(positions, timestamp),
                "capital_limit_at_entry": capital_limit,
                "committed_initial_capital_after_entry": float(sum(position.size / LEVERAGE for position in positions)),
                "marked_allocated_capital_after_entry": open_margin_after,
                "position_initial_margin": position_size / LEVERAGE if accepted else 0.0,
                "position_exit_timestamp": pd.Timestamp(row.exit_timestamp) if accepted else pd.NaT,
                "position_net_return": float(row.net_return) if accepted else np.nan,
                "position_gross_return": float(row.gross_return) if accepted else np.nan,
                "position_exit_reason": str(row.exit_reason) if accepted else "",
                "position_exit_price": float(row.exit_price) if accepted else np.nan,
                "policy_outcome_available": True,
            })
        equity_rows.append({
            "timestamp": timestamp, "wallet": wallet,
            "mtm_equity": wallet + _unrealised_pnl(positions, timestamp),
            "unrealized_pnl": _unrealised_pnl(positions, timestamp),
            "open_notional": _marked_notional(positions, timestamp),
            "open_allocated_capital": _allocated_margin(positions, timestamp),
            "committed_initial_capital": float(sum(position.size / LEVERAGE for position in positions)),
            "open_positions": len(positions), "entries_this_bar": entries,
            "realized_pnl": float(sum(position.size * position.net_return for position in due)),
        })
    if positions:
        terminal = max(position.exit_timestamp for position in positions)
        wallet += float(sum(position.size * position.net_return for position in positions))
        equity_rows.append({
            "timestamp": terminal, "wallet": wallet, "mtm_equity": wallet,
            "unrealized_pnl": 0.0, "open_notional": 0.0, "open_allocated_capital": 0.0,
            "committed_initial_capital": 0.0, "open_positions": 0,
            "entries_this_bar": 0,
            "realized_pnl": float(sum(position.size * position.net_return for position in positions)),
        })
    return pd.DataFrame(decisions), pd.DataFrame(equity_rows)
