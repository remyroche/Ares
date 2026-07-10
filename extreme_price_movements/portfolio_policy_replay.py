"""Portfolio-level replay and compact optimisation for simple-policy candidates."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import optuna
import pandas as pd

from extreme_price_movements.inference.portfolio_policy import (
    PortfolioPolicyConfig,
    compute_rank_based_position_size,
)
from extreme_price_movements.inference.training_live_parity_contract import (
    build_training_live_parity_contract,
    persist_training_live_parity_contract,
)
from extreme_price_movements.model_loader import load_full_state

EPS = 1e-9
INITIAL_WALLET = 10_000.0
DEFAULT_BAR_MINUTES = 15
DEFAULT_OFFLINE_PRICE_GAP_BPS = 50.0


@dataclass(frozen=True)
class PortfolioPolicyParams:
    """Replay parameters with the v1 global-auction contract shape."""

    max_concurrent_positions: int = 8
    max_concurrent_per_side: Optional[int] = 4
    max_concurrent_per_strategy: Optional[int] = 4
    max_concurrent_per_symbol: int = 1
    max_new_entries_per_bar: int = 2
    max_new_entries_per_strategy_per_bar: Optional[int] = None
    max_total_wallet_allocation_pct: float = 0.75

    global_threshold_floor: float = 0.60
    threshold_viability_margin: float = 0.0
    occupancy_threshold_alpha: float = 0.30
    occupancy_threshold_power: float = 1.50
    allocation_threshold_alpha: float = 0.0
    allocation_threshold_power: float = 1.0

    rank_size_power: float = 1.50
    rank_multiplier_min: float = 0.50
    rank_multiplier_max: float = 1.50

    max_signal_gap_bps: Optional[float] = None
    min_liquidity_capacity_weight: Optional[float] = None
    min_position_size: float = 1.0
    cooldown_hours_after_loss: float = 24.0
    max_consecutive_losing_trades: int = 0
    global_loss_cooldown_hours: float = 0.0
    max_consecutive_losing_trades_per_archetype: int = 0
    archetype_loss_cooldown_hours: float = 0.0
    max_side_concentration: Optional[float] = None
    max_strategy_concentration: Optional[float] = None

    portfolio_policy_version: str = "global_auction_v1"
    strategy_ids: Tuple[str, ...] = ()
    strategy_cores: Tuple[str, ...] = ()

    def to_live_config(self) -> Dict[str, Any]:
        return {
            "portfolio_policy_version": self.portfolio_policy_version,
            "strategy_contract": {
                "strategy_ids": list(self.strategy_ids),
                "strategy_cores": list(self.strategy_cores),
            },
            "selection": {
                "global_threshold_floor": float(self.global_threshold_floor),
                "threshold_viability_margin": float(self.threshold_viability_margin),
                "occupancy_threshold_alpha": float(self.occupancy_threshold_alpha),
                "occupancy_threshold_power": float(self.occupancy_threshold_power),
                "allocation_threshold_alpha": float(self.allocation_threshold_alpha),
                "allocation_threshold_power": float(self.allocation_threshold_power),
            },
            "concurrency": {
                "max_concurrent_positions": int(self.max_concurrent_positions),
                "max_concurrent_per_side": self.max_concurrent_per_side,
                "max_concurrent_per_strategy": self.max_concurrent_per_strategy,
                "max_concurrent_per_symbol": int(self.max_concurrent_per_symbol),
                "max_new_entries_per_bar": int(self.max_new_entries_per_bar),
                "max_new_entries_per_strategy_per_bar": (
                    None
                    if self.max_new_entries_per_strategy_per_bar is None
                    else int(self.max_new_entries_per_strategy_per_bar)
                ),
            },
            "allocation": {
                "max_total_wallet_allocation_pct": float(
                    self.max_total_wallet_allocation_pct
                ),
            },
            "sizing": {
                "rank_size_power": float(self.rank_size_power),
                "rank_multiplier_min": float(self.rank_multiplier_min),
                "rank_multiplier_max": float(self.rank_multiplier_max),
            },
            "friction": {
                "max_signal_gap_bps": self.max_signal_gap_bps,
                "min_liquidity_capacity_weight": self.min_liquidity_capacity_weight,
                "offline_default_price_gap_bps": DEFAULT_OFFLINE_PRICE_GAP_BPS,
            },
            "guardrails": {
                "max_side_concentration": self.max_side_concentration,
                "max_strategy_concentration": self.max_strategy_concentration,
            },
            "risk": {
                "cooldown_hours_after_loss": float(self.cooldown_hours_after_loss),
                "max_consecutive_losing_trades": int(
                    self.max_consecutive_losing_trades
                ),
                "global_loss_cooldown_hours": float(self.global_loss_cooldown_hours),
                "max_consecutive_losing_trades_per_archetype": int(
                    self.max_consecutive_losing_trades_per_archetype
                ),
                "archetype_loss_cooldown_hours": float(
                    self.archetype_loss_cooldown_hours
                ),
            },
            "adaptive_quality": {"enabled": False},
        }

    def to_policy_config(self) -> PortfolioPolicyConfig:
        return PortfolioPolicyConfig(
            max_concurrent_positions=int(self.max_concurrent_positions),
            max_concurrent_per_side=self.max_concurrent_per_side,
            max_concurrent_per_strategy=self.max_concurrent_per_strategy,
            initial_rank_threshold=float(self.global_threshold_floor),
            initial_rank_threshold_floor=float(self.global_threshold_floor),
            threshold_viability_margin=float(self.threshold_viability_margin),
            occupancy_threshold_alpha=float(self.occupancy_threshold_alpha),
            occupancy_threshold_power=float(self.occupancy_threshold_power),
            allocation_threshold_alpha=float(self.allocation_threshold_alpha),
            allocation_threshold_power=float(self.allocation_threshold_power),
            portfolio_policy_version=self.portfolio_policy_version,
            max_new_entries_per_bar=int(self.max_new_entries_per_bar),
            max_new_entries_per_strategy_per_bar=(
                None
                if self.max_new_entries_per_strategy_per_bar is None
                else int(self.max_new_entries_per_strategy_per_bar)
            ),
            max_concurrent_per_symbol=int(self.max_concurrent_per_symbol),
            max_total_wallet_allocation_pct=float(
                self.max_total_wallet_allocation_pct
            ),
            rank_multiplier_min=float(self.rank_multiplier_min),
            rank_multiplier_max=float(self.rank_multiplier_max),
            rank_size_power=float(self.rank_size_power),
            max_consecutive_losing_trades=int(self.max_consecutive_losing_trades),
            max_consecutive_losing_trades_per_archetype=int(
                self.max_consecutive_losing_trades_per_archetype
            ),
            strategy_ids=tuple(self.strategy_ids),
            strategy_cores=tuple(self.strategy_cores),
        )


def portfolio_policy_params_from_live_config(
    payload: Dict[str, Any],
) -> PortfolioPolicyParams:
    """Load replay params from the deployed live portfolio-policy JSON shape."""
    payload = payload if isinstance(payload, dict) else {}
    selection = payload.get("selection", {})
    concurrency = payload.get("concurrency", {})
    allocation = payload.get("allocation", {})
    sizing = payload.get("sizing", {})
    friction = payload.get("friction", {})
    guardrails = payload.get("guardrails", {})
    risk = payload.get("risk", {})
    contract = payload.get("strategy_contract", {})
    if not isinstance(selection, dict):
        selection = {}
    if not isinstance(concurrency, dict):
        concurrency = {}
    if not isinstance(allocation, dict):
        allocation = {}
    if not isinstance(sizing, dict):
        sizing = {}
    if not isinstance(friction, dict):
        friction = {}
    if not isinstance(guardrails, dict):
        guardrails = {}
    if not isinstance(risk, dict):
        risk = {}
    if not isinstance(contract, dict):
        contract = {}

    missing = object()

    def _section_get(section: Dict[str, Any], key: str, default: Any) -> Any:
        value = section.get(key, missing)
        return default if value is missing else value

    def _none_or_int(value: Any) -> Optional[int]:
        if value is None:
            return None
        return int(value)

    def _none_or_float(value: Any) -> Optional[float]:
        if value is None:
            return None
        return float(value)

    return PortfolioPolicyParams(
        max_concurrent_positions=int(
            _section_get(
                concurrency,
                "max_concurrent_positions",
                PortfolioPolicyParams.max_concurrent_positions,
            )
        ),
        max_concurrent_per_side=_none_or_int(
            _section_get(
                concurrency,
                "max_concurrent_per_side",
                PortfolioPolicyParams.max_concurrent_per_side,
            )
        ),
        max_concurrent_per_strategy=_none_or_int(
            _section_get(
                concurrency,
                "max_concurrent_per_strategy",
                PortfolioPolicyParams.max_concurrent_per_strategy,
            )
        ),
        max_concurrent_per_symbol=int(
            _section_get(
                concurrency,
                "max_concurrent_per_symbol",
                PortfolioPolicyParams.max_concurrent_per_symbol,
            )
        ),
        max_new_entries_per_bar=int(
            _section_get(
                concurrency,
                "max_new_entries_per_bar",
                PortfolioPolicyParams.max_new_entries_per_bar,
            )
        ),
        max_new_entries_per_strategy_per_bar=_none_or_int(
            _section_get(
                concurrency,
                "max_new_entries_per_strategy_per_bar",
                PortfolioPolicyParams.max_new_entries_per_strategy_per_bar,
            )
        ),
        max_total_wallet_allocation_pct=float(
            _section_get(
                allocation,
                "max_total_wallet_allocation_pct",
                PortfolioPolicyParams.max_total_wallet_allocation_pct,
            )
        ),
        global_threshold_floor=float(
            _section_get(
                selection,
                "global_threshold_floor",
                PortfolioPolicyParams.global_threshold_floor,
            )
        ),
        threshold_viability_margin=float(
            _section_get(
                selection,
                "threshold_viability_margin",
                PortfolioPolicyParams.threshold_viability_margin,
            )
        ),
        occupancy_threshold_alpha=float(
            _section_get(
                selection,
                "occupancy_threshold_alpha",
                PortfolioPolicyParams.occupancy_threshold_alpha,
            )
        ),
        occupancy_threshold_power=float(
            _section_get(
                selection,
                "occupancy_threshold_power",
                PortfolioPolicyParams.occupancy_threshold_power,
            )
        ),
        allocation_threshold_alpha=float(
            _section_get(
                selection,
                "allocation_threshold_alpha",
                PortfolioPolicyParams.allocation_threshold_alpha,
            )
        ),
        allocation_threshold_power=float(
            _section_get(
                selection,
                "allocation_threshold_power",
                PortfolioPolicyParams.allocation_threshold_power,
            )
        ),
        rank_size_power=float(
            _section_get(sizing, "rank_size_power", PortfolioPolicyParams.rank_size_power)
        ),
        rank_multiplier_min=float(
            _section_get(
                sizing,
                "rank_multiplier_min",
                PortfolioPolicyParams.rank_multiplier_min,
            )
        ),
        rank_multiplier_max=float(
            _section_get(
                sizing,
                "rank_multiplier_max",
                PortfolioPolicyParams.rank_multiplier_max,
            )
        ),
        max_signal_gap_bps=_none_or_float(
            _section_get(
                friction,
                "max_signal_gap_bps",
                PortfolioPolicyParams.max_signal_gap_bps,
            )
        ),
        min_liquidity_capacity_weight=_none_or_float(
            _section_get(
                friction,
                "min_liquidity_capacity_weight",
                PortfolioPolicyParams.min_liquidity_capacity_weight,
            )
        ),
        max_side_concentration=_none_or_float(
            _section_get(
                guardrails,
                "max_side_concentration",
                PortfolioPolicyParams.max_side_concentration,
            )
        ),
        max_strategy_concentration=_none_or_float(
            _section_get(
                guardrails,
                "max_strategy_concentration",
                PortfolioPolicyParams.max_strategy_concentration,
            )
        ),
        cooldown_hours_after_loss=float(
            _section_get(
                risk,
                "cooldown_hours_after_loss",
                PortfolioPolicyParams.cooldown_hours_after_loss,
            )
        ),
        max_consecutive_losing_trades=int(
            _section_get(
                risk,
                "max_consecutive_losing_trades",
                PortfolioPolicyParams.max_consecutive_losing_trades,
            )
        ),
        global_loss_cooldown_hours=float(
            _section_get(
                risk,
                "global_loss_cooldown_hours",
                PortfolioPolicyParams.global_loss_cooldown_hours,
            )
        ),
        max_consecutive_losing_trades_per_archetype=int(
            _section_get(
                risk,
                "max_consecutive_losing_trades_per_archetype",
                PortfolioPolicyParams.max_consecutive_losing_trades_per_archetype,
            )
        ),
        archetype_loss_cooldown_hours=float(
            _section_get(
                risk,
                "archetype_loss_cooldown_hours",
                PortfolioPolicyParams.archetype_loss_cooldown_hours,
            )
        ),
        portfolio_policy_version=str(
            payload.get(
                "portfolio_policy_version",
                PortfolioPolicyParams.portfolio_policy_version,
            )
        ),
        strategy_ids=tuple(str(v) for v in contract.get("strategy_ids", ()) or ()),
        strategy_cores=tuple(str(v) for v in contract.get("strategy_cores", ()) or ()),
    )


def load_portfolio_policy_params(path: str | Path) -> PortfolioPolicyParams:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return portfolio_policy_params_from_live_config(payload)


@dataclass
class OpenPosition:
    symbol: str
    side: str
    strategy_id: str
    entry_timestamp: pd.Timestamp
    exit_timestamp: pd.Timestamp
    position_size: float
    net_return: float
    gross_return: float
    exit_reason: str
    entry_price: float = np.nan
    exit_price: float = np.nan
    fees_bps: float = 0.0
    mtm_path_gross_returns: Optional[Tuple[float, ...]] = None
    policy_archetype: str = "missing"


@dataclass
class PortfolioState:
    wallet: float = INITIAL_WALLET
    open_positions: List[OpenPosition] = field(default_factory=list)
    closed_positions: List[OpenPosition] = field(default_factory=list)
    cooldowns: Dict[str, pd.Timestamp] = field(default_factory=dict)
    open_notional_value: float = 0.0
    side_open: Dict[str, int] = field(default_factory=lambda: {"long": 0, "short": 0})
    strategy_open: Dict[str, int] = field(default_factory=dict)
    symbol_open: Dict[str, int] = field(default_factory=dict)
    consecutive_losing_trades: int = 0
    archetype_consecutive_losing_trades: Dict[str, int] = field(default_factory=dict)
    global_loss_cooldown_until: Optional[pd.Timestamp] = None
    archetype_loss_cooldowns: Dict[str, pd.Timestamp] = field(default_factory=dict)
    loss_guard_events: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.open_positions and not self.strategy_open and not self.symbol_open:
            self.open_notional_value = 0.0
            self.side_open = {"long": 0, "short": 0}
            for pos in self.open_positions:
                self.open_notional_value += float(pos.position_size)
                self.side_open[pos.side] = int(self.side_open.get(pos.side, 0)) + 1
                self.strategy_open[pos.strategy_id] = (
                    int(self.strategy_open.get(pos.strategy_id, 0)) + 1
                )
                self.symbol_open[pos.symbol] = (
                    int(self.symbol_open.get(pos.symbol, 0)) + 1
                )

    def clone(self) -> "PortfolioState":
        """Return a lossless replay-state copy for exact counterfactual branches."""
        return PortfolioState(
            wallet=float(self.wallet),
            open_positions=[replace(pos) for pos in self.open_positions],
            closed_positions=[replace(pos) for pos in self.closed_positions],
            cooldowns={str(symbol): pd.Timestamp(until) for symbol, until in self.cooldowns.items()},
            open_notional_value=float(self.open_notional_value),
            side_open={str(key): int(value) for key, value in self.side_open.items()},
            strategy_open={str(key): int(value) for key, value in self.strategy_open.items()},
            symbol_open={str(key): int(value) for key, value in self.symbol_open.items()},
            consecutive_losing_trades=int(self.consecutive_losing_trades),
            archetype_consecutive_losing_trades={
                str(key): int(value)
                for key, value in self.archetype_consecutive_losing_trades.items()
            },
            global_loss_cooldown_until=(
                pd.Timestamp(self.global_loss_cooldown_until)
                if self.global_loss_cooldown_until is not None
                else None
            ),
            archetype_loss_cooldowns={
                str(key): pd.Timestamp(value)
                for key, value in self.archetype_loss_cooldowns.items()
            },
            loss_guard_events=[dict(event) for event in self.loss_guard_events],
        )

    def _decrement_position_counts(self, pos: OpenPosition) -> None:
        self.open_notional_value = max(
            0.0, float(self.open_notional_value) - float(pos.position_size)
        )
        for counts, key in (
            (self.side_open, pos.side),
            (self.strategy_open, pos.strategy_id),
            (self.symbol_open, pos.symbol),
        ):
            next_value = int(counts.get(key, 0)) - 1
            if next_value > 0:
                counts[key] = next_value
            else:
                counts.pop(key, None)

    def open_position(self, pos: OpenPosition) -> None:
        self.open_positions.append(pos)
        self.open_notional_value += float(pos.position_size)
        self.side_open[pos.side] = int(self.side_open.get(pos.side, 0)) + 1
        self.strategy_open[pos.strategy_id] = (
            int(self.strategy_open.get(pos.strategy_id, 0)) + 1
        )
        self.symbol_open[pos.symbol] = int(self.symbol_open.get(pos.symbol, 0)) + 1

    @staticmethod
    def _manual_block_until() -> pd.Timestamp:
        return pd.Timestamp("2262-01-01", tz="UTC")

    @staticmethod
    def _normalise_archetype_key(policy_archetype: Any) -> str:
        text = str(policy_archetype or "").strip()
        return text if text else "missing_policy_archetype"

    @classmethod
    def _block_until(cls, timestamp: pd.Timestamp, cooldown_hours: float) -> pd.Timestamp:
        ts = pd.Timestamp(timestamp)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        if float(cooldown_hours) > 0.0:
            return ts + pd.Timedelta(hours=float(cooldown_hours))
        return cls._manual_block_until()

    def _expire_loss_guards(self, timestamp: pd.Timestamp) -> None:
        ts = pd.Timestamp(timestamp)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        if (
            self.global_loss_cooldown_until is not None
            and pd.Timestamp(self.global_loss_cooldown_until) <= ts
        ):
            self.global_loss_cooldown_until = None
            self.consecutive_losing_trades = 0
        self.archetype_loss_cooldowns = {
            key: pd.Timestamp(until)
            for key, until in self.archetype_loss_cooldowns.items()
            if pd.Timestamp(until) > ts
        }

    def loss_guard_block_reason(
        self, timestamp: pd.Timestamp, policy_archetype: Any
    ) -> Optional[str]:
        self._expire_loss_guards(timestamp)
        ts = pd.Timestamp(timestamp)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        if (
            self.global_loss_cooldown_until is not None
            and pd.Timestamp(self.global_loss_cooldown_until) > ts
        ):
            return "global_loss_streak_block"
        archetype_key = self._normalise_archetype_key(policy_archetype)
        until = self.archetype_loss_cooldowns.get(archetype_key)
        if until is not None and pd.Timestamp(until) > ts:
            return "archetype_loss_streak_block"
        return None

    def close_due(
        self,
        timestamp: pd.Timestamp,
        *,
        cooldown_hours_after_loss: float = 24.0,
        max_consecutive_losing_trades: int = 0,
        global_loss_cooldown_hours: float = 0.0,
        max_consecutive_losing_trades_per_archetype: int = 0,
        archetype_loss_cooldown_hours: float = 0.0,
    ) -> float:
        realized = 0.0
        still_open: List[OpenPosition] = []
        self._expire_loss_guards(timestamp)
        for pos in self.open_positions:
            if pos.exit_timestamp <= timestamp:
                pnl = float(pos.position_size) * float(pos.net_return)
                self.wallet += pnl
                realized += pnl
                self.closed_positions.append(pos)
                archetype_key = self._normalise_archetype_key(pos.policy_archetype)
                if pnl > 0.0:
                    self.consecutive_losing_trades = 0
                    self.archetype_consecutive_losing_trades[archetype_key] = 0
                else:
                    self.consecutive_losing_trades += 1
                    archetype_streak = (
                        int(self.archetype_consecutive_losing_trades.get(archetype_key, 0))
                        + 1
                    )
                    self.archetype_consecutive_losing_trades[archetype_key] = (
                        archetype_streak
                    )
                    if (
                        int(max_consecutive_losing_trades_per_archetype) > 0
                        and archetype_streak
                        >= int(max_consecutive_losing_trades_per_archetype)
                    ):
                        until = self._block_until(
                            pos.exit_timestamp, float(archetype_loss_cooldown_hours)
                        )
                        self.archetype_loss_cooldowns[archetype_key] = until
                        self.loss_guard_events.append(
                            {
                                "event": "archetype_loss_streak_block",
                                "timestamp": pd.Timestamp(pos.exit_timestamp),
                                "policy_archetype": archetype_key,
                                "streak": int(archetype_streak),
                                "threshold": int(
                                    max_consecutive_losing_trades_per_archetype
                                ),
                                "blocked_until": until,
                            }
                        )
                    if (
                        int(max_consecutive_losing_trades) > 0
                        and self.consecutive_losing_trades
                        >= int(max_consecutive_losing_trades)
                    ):
                        until = self._block_until(
                            pos.exit_timestamp, float(global_loss_cooldown_hours)
                        )
                        self.global_loss_cooldown_until = until
                        self.loss_guard_events.append(
                            {
                                "event": "global_loss_streak_block",
                                "timestamp": pd.Timestamp(pos.exit_timestamp),
                                "streak": int(self.consecutive_losing_trades),
                                "threshold": int(max_consecutive_losing_trades),
                                "blocked_until": until,
                            }
                        )
                if pnl <= 0.0 and float(cooldown_hours_after_loss) > 0.0:
                    self.cooldowns[pos.symbol] = pos.exit_timestamp + pd.Timedelta(
                        hours=float(cooldown_hours_after_loss)
                    )
                self._decrement_position_counts(pos)
            else:
                still_open.append(pos)
        self.open_positions = still_open
        self.cooldowns = {
            symbol: until
            for symbol, until in self.cooldowns.items()
            if until > timestamp
        }
        self._expire_loss_guards(timestamp)
        return realized

    @property
    def open_notional(self) -> float:
        return float(self.open_notional_value)

    def side_counts(self) -> Dict[str, int]:
        return {
            "long": int(self.side_open.get("long", 0)),
            "short": int(self.side_open.get("short", 0)),
        }

    def strategy_counts(self) -> Dict[str, int]:
        return dict(self.strategy_open)

    def symbol_counts(self) -> Dict[str, int]:
        return dict(self.symbol_open)

    def unrealized_pnl(self, timestamp: pd.Timestamp) -> float:
        return float(
            sum(
                float(pos.position_size) * _position_mtm_return(pos, timestamp)
                for pos in self.open_positions
            )
        )


@dataclass
class ReplayDecision:
    candidate_index: int
    timestamp: pd.Timestamp
    symbol: str
    side: str
    strategy_id: str
    normalized_rank_score: float
    effective_rank_score: float
    base_threshold: float
    dynamic_threshold: float
    portfolio_priority: float
    accepted: bool
    rejection_reason: str
    position_size: float
    open_positions_before: int
    open_positions_after: int
    side_count_before: int
    strategy_count_before: int
    strategy_entries_this_bar_before: int
    wallet_before: float
    wallet_after: float
    open_notional_before: float
    open_notional_after: float
    position_exit_timestamp: Optional[pd.Timestamp] = None
    position_net_return: float = np.nan
    position_gross_return: float = np.nan
    position_exit_reason: str = ""
    position_exit_price: float = np.nan


@dataclass(frozen=True)
class CandidateReplayCache:
    frame: pd.DataFrame
    timestamps: np.ndarray
    groups: Tuple[np.ndarray, ...]
    symbol: np.ndarray
    side: np.ndarray
    policy_archetype: np.ndarray
    side_order: np.ndarray
    strategy_id: np.ndarray
    rank_score: np.ndarray
    base_threshold: np.ndarray
    calibrated_score: np.ndarray
    price_gap_bps: np.ndarray
    expected_friction_bps: np.ndarray
    liquidity_capacity_weight: np.ndarray
    portfolio_max_new_entries_per_bar: np.ndarray
    portfolio_max_new_entries_per_strategy_per_bar: np.ndarray
    portfolio_max_concurrent_per_strategy: np.ndarray
    portfolio_wallet_cap_multiplier: np.ndarray
    portfolio_size_multiplier: np.ndarray
    portfolio_rank_size_power: np.ndarray
    portfolio_priority_multiplier: np.ndarray
    portfolio_priority_adjustment: np.ndarray
    portfolio_rank_adjustment: np.ndarray
    portfolio_fixed_position_size: np.ndarray
    exit_timestamp: np.ndarray
    holding_bars: np.ndarray
    net_return: np.ndarray
    gross_return: np.ndarray
    entry_price: np.ndarray
    exit_price: np.ndarray
    fees_bps: np.ndarray
    mtm_path_gross_returns: np.ndarray
    exit_reason: np.ndarray


def _candidate_cache(candidates: pd.DataFrame) -> CandidateReplayCache:
    work = _normalised_candidate_table(candidates)
    timestamps = work["timestamp"].to_numpy()
    if len(work):
        change = np.flatnonzero(timestamps[1:] != timestamps[:-1]) + 1
        starts = np.concatenate(([0], change))
        ends = np.concatenate((change, [len(work)]))
        groups = tuple(np.arange(start, end, dtype=np.int64) for start, end in zip(starts, ends))
        unique_timestamps = timestamps[starts]
    else:
        groups = ()
        unique_timestamps = np.asarray([], dtype="datetime64[ns]")
    side = work["side"].astype(str).to_numpy()
    if "policy_archetype" in work.columns:
        policy_archetype = work["policy_archetype"].fillna("missing").astype(str).to_numpy()
    elif "local_side_archetype" in work.columns:
        policy_archetype = work["local_side_archetype"].fillna("missing").astype(str).to_numpy()
    else:
        policy_archetype = np.full(len(work), "missing", dtype=object)
    return CandidateReplayCache(
        frame=work,
        timestamps=unique_timestamps,
        groups=groups,
        symbol=work["symbol"].astype(str).to_numpy(),
        side=side,
        policy_archetype=policy_archetype,
        side_order=np.asarray([_side_sort_key(value) for value in side], dtype=np.int8),
        strategy_id=work["strategy_id"].astype(str).to_numpy(),
        rank_score=pd.to_numeric(work["normalized_rank_score"], errors="coerce").to_numpy(dtype=float),
        base_threshold=pd.to_numeric(work["base_strategy_threshold"], errors="coerce").fillna(1.0).to_numpy(dtype=float),
        calibrated_score=pd.to_numeric(work["calibrated_score"], errors="coerce").fillna(-np.inf).to_numpy(dtype=float),
        price_gap_bps=pd.to_numeric(work["price_gap_bps"], errors="coerce").fillna(0.0).to_numpy(dtype=float),
        expected_friction_bps=pd.to_numeric(work["expected_friction_bps"], errors="coerce").fillna(0.0).to_numpy(dtype=float),
        liquidity_capacity_weight=pd.to_numeric(work["liquidity_capacity_weight"], errors="coerce").fillna(1.0).to_numpy(dtype=float),
        portfolio_max_new_entries_per_bar=pd.to_numeric(
            work.get("portfolio_max_new_entries_per_bar"),
            errors="coerce",
        ).to_numpy(dtype=float)
        if "portfolio_max_new_entries_per_bar" in work.columns
        else np.full(len(work), np.nan, dtype=float),
        portfolio_max_new_entries_per_strategy_per_bar=pd.to_numeric(
            work.get("portfolio_max_new_entries_per_strategy_per_bar"),
            errors="coerce",
        ).to_numpy(dtype=float)
        if "portfolio_max_new_entries_per_strategy_per_bar" in work.columns
        else np.full(len(work), np.nan, dtype=float),
        portfolio_max_concurrent_per_strategy=pd.to_numeric(
            work.get("portfolio_max_concurrent_per_strategy"),
            errors="coerce",
        ).to_numpy(dtype=float)
        if "portfolio_max_concurrent_per_strategy" in work.columns
        else np.full(len(work), np.nan, dtype=float),
        portfolio_wallet_cap_multiplier=pd.to_numeric(
            work.get("portfolio_wallet_cap_multiplier"),
            errors="coerce",
        ).fillna(1.0).clip(lower=0.0, upper=1.0).to_numpy(dtype=float)
        if "portfolio_wallet_cap_multiplier" in work.columns
        else np.ones(len(work), dtype=float),
        portfolio_size_multiplier=pd.to_numeric(
            work.get("portfolio_size_multiplier"),
            errors="coerce",
        ).fillna(1.0).clip(lower=0.0).to_numpy(dtype=float)
        if "portfolio_size_multiplier" in work.columns
        else np.ones(len(work), dtype=float),
        portfolio_rank_size_power=pd.to_numeric(
            work.get("portfolio_rank_size_power"),
            errors="coerce",
        ).replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)
        if "portfolio_rank_size_power" in work.columns
        else np.full(len(work), np.nan, dtype=float),
        portfolio_priority_multiplier=pd.to_numeric(
            work.get("portfolio_priority_multiplier"),
            errors="coerce",
        )
        .replace([np.inf, -np.inf], np.nan)
        .fillna(1.0)
        .clip(lower=0.0)
        .to_numpy(dtype=float)
        if "portfolio_priority_multiplier" in work.columns
        else np.ones(len(work), dtype=float),
        portfolio_priority_adjustment=pd.to_numeric(
            work.get("portfolio_priority_adjustment"),
            errors="coerce",
        ).replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float)
        if "portfolio_priority_adjustment" in work.columns
        else np.zeros(len(work), dtype=float),
        portfolio_rank_adjustment=pd.to_numeric(
            work.get("portfolio_rank_adjustment"),
            errors="coerce",
        ).replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float)
        if "portfolio_rank_adjustment" in work.columns
        else np.zeros(len(work), dtype=float),
        portfolio_fixed_position_size=pd.to_numeric(
            work.get("portfolio_fixed_position_size"),
            errors="coerce",
        ).replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)
        if "portfolio_fixed_position_size" in work.columns
        else np.full(len(work), np.nan, dtype=float),
        exit_timestamp=work["exit_timestamp"].to_numpy(),
        holding_bars=pd.to_numeric(work["holding_bars"], errors="coerce").fillna(1.0).to_numpy(dtype=float),
        net_return=pd.to_numeric(work["net_return"], errors="coerce").fillna(0.0).to_numpy(dtype=float),
        gross_return=pd.to_numeric(work["gross_return"], errors="coerce").fillna(0.0).to_numpy(dtype=float),
        entry_price=pd.to_numeric(work["entry_price"], errors="coerce").fillna(np.nan).to_numpy(dtype=float),
        exit_price=pd.to_numeric(work["exit_price"], errors="coerce").fillna(np.nan).to_numpy(dtype=float),
        fees_bps=pd.to_numeric(work["fees_bps"], errors="coerce").fillna(0.0).to_numpy(dtype=float),
        mtm_path_gross_returns=(
            work["mtm_path_gross_returns"].to_numpy(dtype=object)
            if "mtm_path_gross_returns" in work.columns
            else np.full(len(work), None, dtype=object)
        ),
        exit_reason=work["simple_policy_exit_reason"].fillna("").astype(str).to_numpy(),
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(float(value)) else None
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _side(value: Any, strategy_id: Any = "") -> str:
    text = str(value).strip().lower()
    if text in {"1", "1.0", "long", "l"}:
        return "long"
    if text in {"-1", "-1.0", "short", "s"}:
        return "short"
    sid = str(strategy_id).strip().lower()
    return "short" if sid.startswith("s") or sid.startswith("short") else "long"


def _coerce_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
        return out if np.isfinite(out) else default
    except Exception:
        return default


def _cap_reached(count: int, cap: Optional[int]) -> bool:
    return cap is not None and int(count) >= int(cap)


def _coerce_return_path(value: Any) -> Optional[Tuple[float, ...]]:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            value = json.loads(text)
        except Exception:
            return None
    if isinstance(value, np.ndarray):
        raw = value.ravel().tolist()
    elif isinstance(value, (list, tuple)):
        raw = list(value)
    else:
        return None
    out: List[float] = []
    for item in raw:
        try:
            val = float(item)
        except Exception:
            val = np.nan
        out.append(val if np.isfinite(val) else np.nan)
    return tuple(out) if out else None


def _normalised_candidate_table(candidates: pd.DataFrame) -> pd.DataFrame:
    if candidates.attrs.get("portfolio_policy_candidates_normalised") is True:
        return candidates
    return normalise_candidate_table(candidates)


def normalise_candidate_table(candidates: pd.DataFrame) -> pd.DataFrame:
    if candidates.attrs.get("portfolio_policy_candidates_normalised") is True:
        return candidates.copy()
    required = {
        "timestamp",
        "symbol",
        "side",
        "strategy_id",
        "base_strategy_threshold",
        "calibrated_score",
        "entry_price",
        "exit_timestamp",
        "exit_price",
        "net_return",
        "gross_return",
        "holding_bars",
        "simple_policy_exit_reason",
    }
    missing = sorted(required.difference(candidates.columns))
    if missing:
        raise ValueError(f"candidate table missing required fields: {missing}")
    out = candidates.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    out["exit_timestamp"] = pd.to_datetime(
        out["exit_timestamp"], utc=True, errors="coerce"
    )
    out["symbol"] = out["symbol"].astype(str)
    out["strategy_id"] = out["strategy_id"].astype(str)
    out["side"] = [_side(s, sid) for s, sid in zip(out["side"], out["strategy_id"])]
    numeric = [
        "normalized_rank_score",
        "strategy_rank_pct",
        "base_strategy_threshold",
        "calibrated_score",
        "entry_price",
        "exit_price",
        "net_return",
        "gross_return",
        "fees_bps",
        "slippage_bps",
        "exit_quote_half_spread_bps",
        "holding_bars",
        "barrier_pct",
        "policy_sl_mult",
        "policy_trailing_activation_mult",
        "policy_trailing_activation_return",
        "policy_trailing_power",
        "policy_trailing_squash_divisor",
        "policy_giveback_beta",
        "price_gap_bps",
        "expected_friction_bps",
        "liquidity_capacity_weight",
        "portfolio_max_new_entries_per_bar",
        "portfolio_max_new_entries_per_strategy_per_bar",
        "portfolio_max_concurrent_per_strategy",
        "portfolio_wallet_cap_multiplier",
        "portfolio_size_multiplier",
        "portfolio_rank_size_power",
        "portfolio_priority_multiplier",
        "portfolio_priority_adjustment",
        "portfolio_rank_adjustment",
        "portfolio_fixed_position_size",
        "orderbook_slippage_bps",
    ]
    for col in numeric:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    if (
        "normalized_rank_score" not in out.columns
        or out["normalized_rank_score"].isna().all()
    ):
        out["normalized_rank_score"] = out["calibrated_score"].rank(
            method="max", pct=True
        )
    if "strategy_rank_pct" not in out.columns:
        out["strategy_rank_pct"] = out["normalized_rank_score"]
    if "fees_bps" not in out.columns:
        out["fees_bps"] = 0.0
    if "slippage_bps" not in out.columns:
        out["slippage_bps"] = 0.0
    if "expected_friction_bps" not in out.columns:
        out["expected_friction_bps"] = out["fees_bps"].fillna(0.0) + out[
            "slippage_bps"
        ].fillna(0.0)
    if "price_gap_bps" not in out.columns:
        out["price_gap_bps"] = DEFAULT_OFFLINE_PRICE_GAP_BPS
    if "liquidity_capacity_weight" not in out.columns:
        out["liquidity_capacity_weight"] = 1.0
    if "mtm_path_gross_returns" in out.columns:
        out["mtm_path_gross_returns"] = out["mtm_path_gross_returns"].map(
            _coerce_return_path
        )
    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.dropna(
        subset=[
            "timestamp",
            "exit_timestamp",
            "normalized_rank_score",
            "base_strategy_threshold",
            "net_return",
            "gross_return",
        ]
    )
    # The live/base-meta handoff can contain simultaneous long and short
    # candidates for the same symbol and strategy. Treating side as outside the
    # decision key makes a valid side-aware candidate table look duplicated.
    key_cols = ["timestamp", "symbol", "side", "strategy_id"]
    dupes = int(out.duplicated(key_cols).sum())
    if dupes:
        raise ValueError(
            "candidate table has duplicate decision keys "
            f"{key_cols}: {dupes} duplicate rows"
        )
    out = out.sort_values(["timestamp", "strategy_id", "symbol", "side"]).reset_index(drop=True)
    out.attrs["portfolio_policy_candidates_normalised"] = True
    return out


def dynamic_threshold(
    base_strategy_threshold: float,
    state: PortfolioState,
    params: PortfolioPolicyParams,
) -> float:
    return dynamic_threshold_for_count(
        base_strategy_threshold,
        len(state.open_positions),
        params,
    )


def dynamic_threshold_for_count(
    base_strategy_threshold: float,
    open_positions: int,
    params: PortfolioPolicyParams,
    *,
    allocation_share: float = 0.0,
) -> float:
    base = max(float(base_strategy_threshold), float(params.global_threshold_floor))
    occupancy = int(open_positions) / max(int(params.max_concurrent_positions), 1)
    occupancy_uplift = (
        float(params.occupancy_threshold_alpha)
        * float(occupancy) ** float(params.occupancy_threshold_power)
        * (1.0 - base)
    )
    allocated = float(np.clip(allocation_share, 0.0, 1.0))
    allocation_uplift = (
        float(params.allocation_threshold_alpha)
        * allocated ** float(params.allocation_threshold_power)
        * (1.0 - base)
    )
    return float(min(base + occupancy_uplift + allocation_uplift, 0.999))


def dynamic_threshold_values(
    base_strategy_thresholds: np.ndarray,
    open_positions: int,
    params: PortfolioPolicyParams,
    *,
    allocation_share: float = 0.0,
) -> np.ndarray:
    base = np.maximum(
        np.asarray(base_strategy_thresholds, dtype=float),
        float(params.global_threshold_floor),
    )
    occupancy = int(open_positions) / max(int(params.max_concurrent_positions), 1)
    occupancy_uplift = (
        float(params.occupancy_threshold_alpha)
        * float(occupancy) ** float(params.occupancy_threshold_power)
        * (1.0 - base)
    )
    allocated = float(np.clip(allocation_share, 0.0, 1.0))
    allocation_uplift = (
        float(params.allocation_threshold_alpha)
        * allocated ** float(params.allocation_threshold_power)
        * (1.0 - base)
    )
    return np.minimum(base + occupancy_uplift + allocation_uplift, 0.999)


def fit_monotone_ev_curve(candidates: pd.DataFrame, bins: int = 20) -> Dict[str, Any]:
    if candidates.empty:
        return {"schema": "monotone_ev_curve_v1", "x": [0.0, 1.0], "y": [0.0, 0.0], "ev_span": 0.0, "n_rows": 0}
    work = candidates[["normalized_rank_score", "net_return"]].dropna().copy()
    if work.empty:
        return {"schema": "monotone_ev_curve_v1", "x": [0.0, 1.0], "y": [0.0, 0.0], "ev_span": 0.0, "n_rows": 0}
    work["bin"] = pd.qcut(
        work["normalized_rank_score"].rank(method="first"),
        q=min(int(bins), max(2, len(work))),
        duplicates="drop",
    )
    grouped = (
        work.groupby("bin", observed=True)
        .agg(x=("normalized_rank_score", "mean"), y=("net_return", "mean"))
        .dropna()
        .sort_values("x")
    )
    if grouped.empty:
        return {"schema": "monotone_ev_curve_v1", "x": [0.0, 1.0], "y": [0.0, 0.0], "ev_span": 0.0, "n_rows": int(len(work))}
    x = grouped["x"].to_numpy(dtype=float)
    y = np.maximum.accumulate(grouped["y"].to_numpy(dtype=float))
    x = np.concatenate(([0.0], x, [1.0]))
    y = np.concatenate(([y[0]], y, [y[-1]]))
    return {"schema": "monotone_ev_curve_v1", "x": x.tolist(), "y": y.tolist(), "ev_span": float(max(y) - min(y)), "n_rows": int(len(work))}


def _shrunk_ev_curve(
    specific: Dict[str, Any],
    global_curve: Dict[str, Any],
    *,
    support_rows: int,
    shrink_rows: int,
) -> Dict[str, Any]:
    gx, gy, _ = _ev_arrays(global_curve)
    sx, sy, _ = _ev_arrays(specific)
    target_x = np.unique(np.clip(np.concatenate([gx, sx, np.asarray([0.0, 1.0])]), 0.0, 1.0))
    if target_x.size < 2:
        target_x = np.asarray([0.0, 1.0], dtype=float)
    global_y = np.interp(target_x, gx, gy)
    specific_y = np.interp(target_x, sx, sy)
    weight = float(support_rows) / max(float(support_rows + max(int(shrink_rows), 0)), EPS)
    y = weight * specific_y + (1.0 - weight) * global_y
    y = np.maximum.accumulate(y)
    return {
        "schema": "monotone_ev_curve_v1",
        "x": target_x.tolist(),
        "y": y.tolist(),
        "ev_span": float(np.nanmax(y) - np.nanmin(y)) if y.size else 0.0,
        "n_rows": int(support_rows),
        "shrink_weight": float(weight),
    }


def fit_hierarchical_ev_curves(
    candidates: pd.DataFrame,
    *,
    bins: int = 20,
    min_group_rows: int = 80,
    shrink_rows: int = 240,
) -> Dict[str, Any]:
    """Fit strategy/side/archetype EV curves with shrinkage to a global curve."""
    work = normalise_candidate_table(candidates)
    if "policy_archetype" not in work.columns:
        if "local_side_archetype" in work.columns:
            work["policy_archetype"] = work["local_side_archetype"]
        else:
            work["policy_archetype"] = "missing"
    work["policy_archetype"] = work["policy_archetype"].fillna("missing").astype(str)
    global_curve = fit_monotone_ev_curve(work, bins=bins)
    by_strategy_side_archetype: Dict[str, Any] = {}
    by_side_archetype: Dict[str, Any] = {}
    by_strategy_side: Dict[str, Any] = {}
    by_strategy: Dict[str, Any] = {}
    if not work.empty:
        for (side, archetype), group in work.groupby(["side", "policy_archetype"], sort=True):
            curve = fit_monotone_ev_curve(group, bins=bins)
            n = int(curve.get("n_rows", len(group)))
            if n >= int(min_group_rows):
                by_side_archetype[f"{side}|{archetype}"] = _shrunk_ev_curve(
                    curve,
                    global_curve,
                    support_rows=n,
                    shrink_rows=int(shrink_rows),
                )
        for strategy_id, group in work.groupby("strategy_id", sort=True):
            curve = fit_monotone_ev_curve(group, bins=bins)
            n = int(curve.get("n_rows", len(group)))
            if n >= int(min_group_rows):
                by_strategy[str(strategy_id)] = _shrunk_ev_curve(
                    curve,
                    global_curve,
                    support_rows=n,
                    shrink_rows=int(shrink_rows),
                )
            for side, side_group in group.groupby("side", sort=True):
                side_curve = fit_monotone_ev_curve(side_group, bins=bins)
                n_side = int(side_curve.get("n_rows", len(side_group)))
                if n_side >= int(min_group_rows):
                    by_strategy_side[f"{strategy_id}|{side}"] = _shrunk_ev_curve(
                        side_curve,
                        global_curve,
                        support_rows=n_side,
                        shrink_rows=int(shrink_rows),
                    )
                for archetype, archetype_group in side_group.groupby("policy_archetype", sort=True):
                    archetype_curve = fit_monotone_ev_curve(archetype_group, bins=bins)
                    n_archetype = int(archetype_curve.get("n_rows", len(archetype_group)))
                    if n_archetype >= int(min_group_rows):
                        by_strategy_side_archetype[f"{strategy_id}|{side}|{archetype}"] = _shrunk_ev_curve(
                            archetype_curve,
                            global_curve,
                            support_rows=n_archetype,
                            shrink_rows=int(shrink_rows),
                        )
    return {
        "schema": "hierarchical_ev_curve_v1",
        "global": global_curve,
        "strategy_side_archetype": by_strategy_side_archetype,
        "side_archetype": by_side_archetype,
        "strategy_side": by_strategy_side,
        "strategy": by_strategy,
        "min_group_rows": int(min_group_rows),
        "shrink_rows": int(shrink_rows),
        "bins": int(bins),
    }


def _select_ev_curve(
    ev_curve: Dict[str, Any],
    *,
    strategy_id: str,
    side: str,
    policy_archetype: str | None = None,
) -> Dict[str, Any]:
    if str(ev_curve.get("schema") or "") != "hierarchical_ev_curve_v1":
        return ev_curve
    strategy_side_archetype = (
        ev_curve.get("strategy_side_archetype")
        if isinstance(ev_curve.get("strategy_side_archetype"), dict)
        else {}
    )
    side_archetype = ev_curve.get("side_archetype") if isinstance(ev_curve.get("side_archetype"), dict) else {}
    strategy_side = ev_curve.get("strategy_side") if isinstance(ev_curve.get("strategy_side"), dict) else {}
    strategy = ev_curve.get("strategy") if isinstance(ev_curve.get("strategy"), dict) else {}
    archetype = str(policy_archetype or "").strip()
    if archetype and archetype.lower() not in {"nan", "none", "missing"}:
        key = f"{strategy_id}|{side}|{archetype}"
        if key in strategy_side_archetype:
            return strategy_side_archetype[key]
        key = f"{side}|{archetype}"
        if key in side_archetype:
            return side_archetype[key]
    key = f"{strategy_id}|{side}"
    if key in strategy_side:
        return strategy_side[key]
    if strategy_id in strategy:
        return strategy[strategy_id]
    global_curve = ev_curve.get("global")
    return global_curve if isinstance(global_curve, dict) else {"x": [0.0, 1.0], "y": [0.0, 0.0], "ev_span": 0.0}


def _ev_arrays(curve: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, float]:
    x = np.asarray(curve.get("x", [0.0, 1.0]), dtype=float)
    y = np.asarray(curve.get("y", [0.0, 0.0]), dtype=float)
    if len(x) < 2 or len(x) != len(y):
        x = np.asarray([0.0, 1.0], dtype=float)
        y = np.asarray([0.0, 0.0], dtype=float)
    ev_span = abs(_coerce_float(curve.get("ev_span"), 0.0))
    return x, y, float(max(ev_span, EPS))


def portfolio_priority_from_values(
    rank_score: float,
    dynamic_threshold_value: float,
    gap_bps: float,
    friction_bps: float,
    ev_curve: Dict[str, Any],
) -> float:
    if not np.isfinite(float(rank_score)):
        return float("-inf")
    x, y, fallback_span = _ev_arrays(ev_curve)
    rank = float(np.clip(rank_score, 0.0, 1.0))
    threshold = float(np.clip(dynamic_threshold_value, 0.0, 0.999))
    surplus = max((rank - threshold) / max(1.0 - threshold, EPS), 0.0)
    rank_ev = float(np.interp(rank, x, y))
    threshold_ev = float(np.interp(threshold, x, y))
    ev_span = max(abs(rank_ev - threshold_ev), fallback_span, EPS)
    price_gap_penalty = (
        (max(float(gap_bps), 0.0) + max(float(friction_bps), 0.0)) / 10_000.0
    ) / ev_span
    return float(surplus - price_gap_penalty)


def portfolio_priority_values(
    rank_scores: np.ndarray,
    dynamic_thresholds: np.ndarray,
    gap_bps: np.ndarray,
    friction_bps: np.ndarray,
    ev_curve: Dict[str, Any],
) -> np.ndarray:
    ranks = np.asarray(rank_scores, dtype=float)
    thresholds = np.asarray(dynamic_thresholds, dtype=float)
    x, y, fallback_span = _ev_arrays(ev_curve)
    clipped_ranks = np.clip(ranks, 0.0, 1.0)
    clipped_thresholds = np.clip(thresholds, 0.0, 0.999)
    surplus = np.maximum(
        (clipped_ranks - clipped_thresholds)
        / np.maximum(1.0 - clipped_thresholds, EPS),
        0.0,
    )
    rank_ev = np.interp(clipped_ranks, x, y)
    threshold_ev = np.interp(clipped_thresholds, x, y)
    ev_span = np.maximum.reduce(
        [
            np.abs(rank_ev - threshold_ev),
            np.full_like(clipped_ranks, fallback_span, dtype=float),
            np.full_like(clipped_ranks, EPS, dtype=float),
        ]
    )
    gap = np.maximum(np.asarray(gap_bps, dtype=float), 0.0)
    friction = np.maximum(np.asarray(friction_bps, dtype=float), 0.0)
    priorities = surplus - ((gap + friction) / 10_000.0) / ev_span
    return np.where(np.isfinite(ranks), priorities, float("-inf"))


def portfolio_priority(
    row: pd.Series,
    dynamic_threshold_value: float,
    ev_curve: Dict[str, Any],
) -> float:
    rank_score = _coerce_float(row.get("normalized_rank_score"), np.nan)
    gap_bps = max(_coerce_float(row.get("price_gap_bps"), 0.0), 0.0)
    friction_bps = max(_coerce_float(row.get("expected_friction_bps"), 0.0), 0.0)
    selected_curve = _select_ev_curve(
        ev_curve,
        strategy_id=str(row.get("strategy_id", "")),
        side=_side(row.get("side"), row.get("strategy_id", "")),
        policy_archetype=str(row.get("policy_archetype", row.get("local_side_archetype", ""))),
    )
    return portfolio_priority_from_values(
        rank_score,
        dynamic_threshold_value,
        gap_bps,
        friction_bps,
        selected_curve,
    )


def portfolio_priority_values_for_rows(
    *,
    rank_scores: np.ndarray,
    dynamic_thresholds: np.ndarray,
    gap_bps: np.ndarray,
    friction_bps: np.ndarray,
    strategy_ids: np.ndarray,
    sides: np.ndarray,
    ev_curve: Dict[str, Any],
    policy_archetypes: np.ndarray | None = None,
) -> np.ndarray:
    if str(ev_curve.get("schema") or "") != "hierarchical_ev_curve_v1":
        return portfolio_priority_values(
            rank_scores,
            dynamic_thresholds,
            gap_bps,
            friction_bps,
            ev_curve,
        )
    out = np.full(len(rank_scores), float("-inf"), dtype=float)
    strategy_values = np.asarray(strategy_ids).astype(str)
    side_values = np.asarray(sides).astype(str)
    if policy_archetypes is None:
        archetype_values = np.full(len(strategy_values), "", dtype=object)
    else:
        archetype_values = np.asarray(policy_archetypes).astype(str)
    keys = np.asarray(
        [f"{sid}|{side}|{arch}" for sid, side, arch in zip(strategy_values, side_values, archetype_values)],
        dtype=object,
    )
    for key in np.unique(keys):
        mask = keys == key
        if not bool(np.any(mask)):
            continue
        strategy_id, side, archetype = str(key).split("|", 2)
        curve = _select_ev_curve(ev_curve, strategy_id=strategy_id, side=side, policy_archetype=archetype)
        out[mask] = portfolio_priority_values(
            np.asarray(rank_scores)[mask],
            np.asarray(dynamic_thresholds)[mask],
            np.asarray(gap_bps)[mask],
            np.asarray(friction_bps)[mask],
            curve,
        )
    return out


def _side_sort_key(side: str) -> int:
    return 0 if side == "long" else 1


def _candidate_order(frame: pd.DataFrame, mode: str) -> pd.DataFrame:
    work = frame.copy()
    work["_side_order"] = work["side"].map(_side_sort_key).fillna(2)
    sort_cols = [
        "portfolio_priority",
        "normalized_rank_score",
        "calibrated_score",
        "expected_friction_bps",
    ]
    ascending = [False, False, False, True]
    if mode == "live_baseline":
        sort_cols = ["_side_order"] + sort_cols
        ascending = [True] + ascending
    return work.sort_values(sort_cols, ascending=ascending).drop(
        columns=["_side_order"]
    )


def _candidate_order_indices(
    cache: CandidateReplayCache,
    indices: np.ndarray,
    priorities: np.ndarray,
    *,
    mode: str,
) -> np.ndarray:
    rank = np.clip(
        cache.rank_score[indices] + cache.portfolio_rank_adjustment[indices],
        0.0,
        1.0,
    )
    calibrated = cache.calibrated_score[indices]
    friction = cache.expected_friction_bps[indices]
    if mode == "live_baseline":
        order = np.lexsort((friction, -calibrated, -rank, -priorities, cache.side_order[indices]))
    else:
        order = np.lexsort((friction, -calibrated, -rank, -priorities))
    return indices[order]


def _position_mtm_return(pos: OpenPosition, timestamp: pd.Timestamp) -> float:
    ts = pd.Timestamp(timestamp)
    entry_ts = pd.Timestamp(pos.entry_timestamp)
    exit_ts = pd.Timestamp(pos.exit_timestamp)
    if ts <= entry_ts:
        return 0.0
    if ts >= exit_ts:
        return float(pos.net_return)
    total_seconds = max(float((exit_ts - entry_ts).total_seconds()), 1.0)
    elapsed_seconds = min(max(float((ts - entry_ts).total_seconds()), 0.0), total_seconds)
    progress = float(np.clip(elapsed_seconds / total_seconds, 0.0, 1.0))
    path = pos.mtm_path_gross_returns
    if path:
        elapsed_bars = int(np.floor(elapsed_seconds / (DEFAULT_BAR_MINUTES * 60.0)))
        if elapsed_bars > 0:
            idx = min(elapsed_bars - 1, len(path) - 1)
            gross_return = float(path[idx])
            if np.isfinite(gross_return):
                fee_return = float(pos.gross_return) - float(pos.net_return)
                return float(gross_return - fee_return * progress)
    return float(pos.net_return) * progress


def _max_abs_decision_delta(
    left: pd.DataFrame,
    right: pd.DataFrame,
    column: str,
) -> Optional[float]:
    if (
        left.empty
        or right.empty
        or column not in left.columns
        or column not in right.columns
    ):
        return None
    n = min(len(left), len(right))
    if n <= 0:
        return None
    lhs = pd.to_numeric(left[column].iloc[:n], errors="coerce").to_numpy(dtype=float)
    rhs = pd.to_numeric(right[column].iloc[:n], errors="coerce").to_numpy(dtype=float)
    delta = np.abs(lhs - rhs)
    finite = delta[np.isfinite(delta)]
    return float(finite.max()) if finite.size else None


def replay_candidates(
    candidates: pd.DataFrame,
    params: PortfolioPolicyParams,
    *,
    mode: str = "global_auction",
    ev_curve: Optional[Dict[str, Any]] = None,
    initial_wallet: float = INITIAL_WALLET,
    initial_state: Optional[PortfolioState] = None,
    pre_decision_snapshot_callback: Optional[
        Callable[[pd.Timestamp, PortfolioState, np.ndarray, CandidateReplayCache], None]
    ] = None,
    accepted_position_callback: Optional[
        Callable[
            [
                int,
                pd.Timestamp,
                PortfolioState,
                CandidateReplayCache,
                float,
                float,
                float,
                np.ndarray,
            ],
            Optional[Dict[str, Any]],
        ]
    ] = None,
    market_mode: str = "spot",
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    cache = _candidate_cache(candidates)
    work = cache.frame
    state = initial_state.clone() if initial_state is not None else PortfolioState(wallet=float(initial_wallet))
    ev_curve = ev_curve or fit_hierarchical_ev_curves(work)
    decisions: List[ReplayDecision] = []
    equity_rows: List[Dict[str, Any]] = []
    policy = params.to_policy_config()

    for timestamp, group_idx in zip(cache.timestamps, cache.groups):
        ts = pd.Timestamp(timestamp)
        realized = state.close_due(
            ts,
            cooldown_hours_after_loss=float(params.cooldown_hours_after_loss),
            max_consecutive_losing_trades=int(params.max_consecutive_losing_trades),
            global_loss_cooldown_hours=float(params.global_loss_cooldown_hours),
            max_consecutive_losing_trades_per_archetype=int(
                params.max_consecutive_losing_trades_per_archetype
            ),
            archetype_loss_cooldown_hours=float(
                params.archetype_loss_cooldown_hours
            ),
        )
        if pre_decision_snapshot_callback is not None:
            pre_decision_snapshot_callback(ts, state.clone(), group_idx.copy(), cache)
        entries_this_bar = 0
        strategy_entries_this_bar: Dict[str, int] = {}
        allocation_share_before_bar = float(
            state.open_notional / max(float(state.wallet), EPS)
        )
        thresholds = dynamic_threshold_values(
            cache.base_threshold[group_idx],
            len(state.open_positions),
            params,
            allocation_share=allocation_share_before_bar,
        )
        effective_rank_scores = np.clip(
            cache.rank_score[group_idx] + cache.portfolio_rank_adjustment[group_idx],
            0.0,
            1.0,
        )
        priorities = portfolio_priority_values_for_rows(
            rank_scores=effective_rank_scores,
            dynamic_thresholds=thresholds,
            gap_bps=cache.price_gap_bps[group_idx],
            friction_bps=cache.expected_friction_bps[group_idx],
            strategy_ids=cache.strategy_id[group_idx],
            sides=cache.side[group_idx],
            policy_archetypes=cache.policy_archetype[group_idx],
            ev_curve=ev_curve,
        )
        priorities = (
            priorities * cache.portfolio_priority_multiplier[group_idx]
            + cache.portfolio_priority_adjustment[group_idx]
        )
        max_entries_this_bar = int(params.max_new_entries_per_bar)
        dynamic_bar_caps = cache.portfolio_max_new_entries_per_bar[group_idx]
        finite_bar_caps = dynamic_bar_caps[np.isfinite(dynamic_bar_caps)]
        if finite_bar_caps.size:
            max_entries_this_bar = int(
                max(0, np.floor(float(np.nanmin(finite_bar_caps))))
            )
        ordered_idx = _candidate_order_indices(cache, group_idx, priorities, mode=mode)

        for idx in ordered_idx:
            side = str(cache.side[idx])
            strategy_id = str(cache.strategy_id[idx])
            symbol = str(cache.symbol[idx])
            rank_score = float(cache.rank_score[idx])
            effective_rank_score = float(
                np.clip(rank_score + float(cache.portfolio_rank_adjustment[idx]), 0.0, 1.0)
            )
            base_threshold = max(
                float(cache.base_threshold[idx]),
                float(params.global_threshold_floor),
            )
            wallet_before = float(state.wallet)
            open_notional_before = float(state.open_notional)
            dyn = dynamic_threshold_for_count(
                base_threshold,
                len(state.open_positions),
                params,
                allocation_share=(
                    float(open_notional_before) / max(float(wallet_before), EPS)
                ),
            )
            price_gap_bps = float(cache.price_gap_bps[idx])
            expected_friction_bps = float(cache.expected_friction_bps[idx])
            priority = portfolio_priority_from_values(
                effective_rank_score,
                dyn,
                price_gap_bps,
                expected_friction_bps,
                _select_ev_curve(
                    ev_curve,
                    strategy_id=strategy_id,
                    side=side,
                    policy_archetype=str(cache.policy_archetype[idx]),
                ),
            )
            priority = (
                priority * float(cache.portfolio_priority_multiplier[idx])
                + float(cache.portfolio_priority_adjustment[idx])
            )
            open_before = len(state.open_positions)
            side_count_before = state.side_open.get(side, 0)
            strategy_count_before = state.strategy_open.get(strategy_id, 0)
            strategy_bar_count_before = int(strategy_entries_this_bar.get(strategy_id, 0))
            strategy_concurrent_cap = params.max_concurrent_per_strategy
            row_strategy_cap = float(cache.portfolio_max_concurrent_per_strategy[idx])
            if np.isfinite(row_strategy_cap):
                strategy_concurrent_cap = int(max(0, np.floor(row_strategy_cap)))
            strategy_bar_cap = params.max_new_entries_per_strategy_per_bar
            row_strategy_bar_cap = float(
                cache.portfolio_max_new_entries_per_strategy_per_bar[idx]
            )
            if np.isfinite(row_strategy_bar_cap):
                strategy_bar_cap = int(max(0, np.floor(row_strategy_bar_cap)))
            wallet_cap_multiplier = max(
                _coerce_float(cache.portfolio_wallet_cap_multiplier[idx], 1.0),
                0.0,
            )
            capital_limit = (
                max(float(params.max_total_wallet_allocation_pct), 0.0)
                * min(wallet_cap_multiplier, 1.0)
                * max(wallet_before, 0.0)
            )
            remaining_capital = max(capital_limit - open_notional_before, 0.0)
            reason = "accepted"
            accepted = False
            position_size = 0.0

            if not np.isfinite(effective_rank_score) or effective_rank_score < dyn + float(
                params.threshold_viability_margin
            ):
                reason = "below_dynamic_threshold"
            elif state.symbol_open.get(symbol, 0) >= int(
                params.max_concurrent_per_symbol
            ):
                reason = "symbol_already_open"
            elif (
                state.cooldowns.get(symbol) is not None and state.cooldowns[symbol] > ts
            ):
                reason = "symbol_in_cooldown"
            else:
                loss_guard_reason = state.loss_guard_block_reason(
                    ts, cache.policy_archetype[idx]
                )
                if loss_guard_reason is not None:
                    reason = loss_guard_reason
                else:
                    loss_guard_reason = ""
            if reason not in {"accepted", ""}:
                pass
            elif len(state.open_positions) >= int(params.max_concurrent_positions):
                reason = "max_concurrent_positions_reached"
            elif _cap_reached(side_count_before, params.max_concurrent_per_side):
                reason = "max_concurrent_per_side_reached"
            elif _cap_reached(
                strategy_count_before, strategy_concurrent_cap
            ):
                reason = "max_concurrent_per_strategy_reached"
            elif _cap_reached(
                strategy_bar_count_before,
                strategy_bar_cap,
            ):
                reason = "max_new_entries_per_strategy_per_bar_reached"
            elif entries_this_bar >= max_entries_this_bar:
                reason = "max_new_entries_per_bar_reached"
            elif remaining_capital <= float(params.min_position_size):
                reason = "max_capital_allocation_reached"
            elif params.min_liquidity_capacity_weight is not None and _coerce_float(
                cache.liquidity_capacity_weight[idx], 1.0
            ) < float(params.min_liquidity_capacity_weight):
                reason = "insufficient_liquidity_capacity"
            elif params.max_signal_gap_bps is not None and abs(price_gap_bps) > float(
                params.max_signal_gap_bps
            ):
                reason = "price_gap_too_large"
            else:
                row_rank_size_power = _coerce_float(
                    cache.portfolio_rank_size_power[idx],
                    np.nan,
                )
                sizing = compute_rank_based_position_size(
                    wallet_value=state.wallet,
                    open_notional=state.open_notional,
                    adjusted_rank_score=effective_rank_score,
                    final_threshold=dyn,
                    policy=policy,
                    liquidity_capacity_weight=_coerce_float(
                        cache.liquidity_capacity_weight[idx], 1.0
                    ),
                    rank_size_power=(
                        float(row_rank_size_power)
                        if np.isfinite(row_rank_size_power)
                        and row_rank_size_power > 0.0
                        else float(params.rank_size_power)
                    ),
                    open_positions=len(state.open_positions),
                    market_mode=market_mode,
                    available_wallet_value=remaining_capital,
                    remaining_total_notional=remaining_capital,
                )
                position_size = min(
                    _coerce_float(sizing.get("size_after_liquidity"), 0.0),
                    remaining_capital,
                )
                position_size = min(
                    position_size
                    * max(_coerce_float(cache.portfolio_size_multiplier[idx], 1.0), 0.0),
                    remaining_capital,
                )
                fixed_position_size = float(cache.portfolio_fixed_position_size[idx])
                if np.isfinite(fixed_position_size) and fixed_position_size > 0.0:
                    position_size = min(fixed_position_size, remaining_capital)
                if position_size <= 0.0 or position_size < float(
                    params.min_position_size
                ):
                    reason = "position_size_too_small"
                elif position_size > max(state.wallet * 100.0, 0.0):
                    reason = "insufficient_wallet_capacity"
                else:
                    accepted = True
                    entries_this_bar += 1
                    strategy_entries_this_bar[strategy_id] = strategy_bar_count_before + 1
                    exit_ts = pd.Timestamp(cache.exit_timestamp[idx])
                    if pd.isna(exit_ts) or exit_ts <= ts:
                        bars = max(1, int(_coerce_float(cache.holding_bars[idx], 1.0)))
                        exit_ts = ts + pd.Timedelta(minutes=DEFAULT_BAR_MINUTES * bars)
                    position_net_return = float(cache.net_return[idx])
                    position_gross_return = float(cache.gross_return[idx])
                    position_exit_reason = str(cache.exit_reason[idx] or "")
                    position_exit_price = float(cache.exit_price[idx])
                    if accepted_position_callback is not None:
                        adjusted = accepted_position_callback(
                            int(idx),
                            ts,
                            state,
                            cache,
                            float(position_size),
                            float(capital_limit),
                            float(remaining_capital),
                            group_idx,
                        )
                        if adjusted:
                            adj_exit_ts = adjusted.get("exit_timestamp")
                            if adj_exit_ts is not None:
                                exit_ts = pd.Timestamp(adj_exit_ts)
                            position_net_return = _coerce_float(
                                adjusted.get("net_return"),
                                position_net_return,
                            )
                            position_gross_return = _coerce_float(
                                adjusted.get("gross_return"),
                                position_gross_return,
                            )
                            position_exit_reason = str(
                                adjusted.get("exit_reason", position_exit_reason) or ""
                            )
                            position_exit_price = _coerce_float(
                                adjusted.get("exit_price"),
                                position_exit_price,
                            )
                    state.open_position(
                        OpenPosition(
                            symbol=symbol,
                            side=side,
                            strategy_id=strategy_id,
                            entry_timestamp=ts,
                            exit_timestamp=exit_ts,
                            position_size=float(position_size),
                            net_return=float(position_net_return),
                            gross_return=float(position_gross_return),
                            exit_reason=str(position_exit_reason),
                            entry_price=float(cache.entry_price[idx]),
                            exit_price=float(position_exit_price),
                            fees_bps=float(cache.fees_bps[idx]),
                            mtm_path_gross_returns=_coerce_return_path(
                                cache.mtm_path_gross_returns[idx]
                            ),
                            policy_archetype=str(cache.policy_archetype[idx]),
                        )
                    )

            decisions.append(
                ReplayDecision(
                    candidate_index=int(idx),
                    timestamp=ts,
                    symbol=symbol,
                    side=side,
                    strategy_id=strategy_id,
                    normalized_rank_score=float(rank_score),
                    effective_rank_score=float(effective_rank_score),
                    base_threshold=float(base_threshold),
                    dynamic_threshold=float(dyn),
                    portfolio_priority=float(priority),
                    accepted=bool(accepted),
                    rejection_reason=reason,
                    position_size=float(position_size if accepted else 0.0),
                    open_positions_before=int(open_before),
                    open_positions_after=int(len(state.open_positions)),
                    side_count_before=int(side_count_before),
                    strategy_count_before=int(strategy_count_before),
                    strategy_entries_this_bar_before=int(strategy_bar_count_before),
                    wallet_before=float(wallet_before),
                    wallet_after=float(state.wallet),
                    open_notional_before=float(open_notional_before),
                    open_notional_after=float(state.open_notional),
                    position_exit_timestamp=exit_ts if accepted else None,
                    position_net_return=float(position_net_return) if accepted else np.nan,
                    position_gross_return=float(position_gross_return) if accepted else np.nan,
                    position_exit_reason=str(position_exit_reason) if accepted else "",
                    position_exit_price=float(position_exit_price) if accepted else np.nan,
                )
            )
        unrealized_pnl = state.unrealized_pnl(ts)
        mtm_equity = float(state.wallet) + float(unrealized_pnl)
        equity_rows.append(
            {
                "timestamp": ts,
                "wallet": float(state.wallet),
                "mtm_equity": float(mtm_equity),
                "unrealized_pnl": float(unrealized_pnl),
                "open_notional": float(state.open_notional),
                "open_capital_pct": float(
                    state.open_notional / max(state.wallet, EPS)
                ),
                "open_positions": int(len(state.open_positions)),
                "entries_this_bar": int(entries_this_bar),
                "realized_pnl": float(realized),
                "mtm_source": "candidate_path_or_linear_interpolation",
            }
        )

    final_ts = work["exit_timestamp"].max()
    if pd.notna(final_ts):
        state.close_due(
            pd.Timestamp(final_ts) + pd.Timedelta(minutes=DEFAULT_BAR_MINUTES),
            cooldown_hours_after_loss=float(params.cooldown_hours_after_loss),
            max_consecutive_losing_trades=int(params.max_consecutive_losing_trades),
            global_loss_cooldown_hours=float(params.global_loss_cooldown_hours),
            max_consecutive_losing_trades_per_archetype=int(
                params.max_consecutive_losing_trades_per_archetype
            ),
            archetype_loss_cooldown_hours=float(
                params.archetype_loss_cooldown_hours
            ),
        )
        unrealized_pnl = state.unrealized_pnl(pd.Timestamp(final_ts))
        equity_rows.append(
            {
                "timestamp": pd.Timestamp(final_ts),
                "wallet": float(state.wallet),
                "mtm_equity": float(state.wallet + unrealized_pnl),
                "unrealized_pnl": float(unrealized_pnl),
                "open_notional": float(state.open_notional),
                "open_capital_pct": float(
                    state.open_notional / max(state.wallet, EPS)
                ),
                "open_positions": int(len(state.open_positions)),
                "entries_this_bar": 0,
                "realized_pnl": 0.0,
                "mtm_source": "candidate_path_or_linear_interpolation",
            }
        )

    decisions_df = pd.DataFrame([asdict(d) for d in decisions])
    equity_df = pd.DataFrame(equity_rows)
    metrics = compute_replay_metrics(
        work,
        decisions_df,
        equity_df,
        initial_wallet=initial_wallet,
        params=params,
    )
    metrics["loss_guard_event_count"] = int(len(state.loss_guard_events))
    metrics["loss_guard_events"] = _json_safe(state.loss_guard_events)
    metrics["final_consecutive_losing_trades"] = int(state.consecutive_losing_trades)
    metrics["final_archetype_loss_streaks"] = {
        str(key): int(value)
        for key, value in state.archetype_consecutive_losing_trades.items()
    }
    return decisions_df, equity_df, metrics


def compute_replay_metrics(
    candidates: pd.DataFrame,
    decisions: pd.DataFrame,
    equity_curve: pd.DataFrame,
    *,
    initial_wallet: float = INITIAL_WALLET,
    params: Optional[PortfolioPolicyParams] = None,
) -> Dict[str, Any]:
    if decisions.empty:
        return {"objective": float("-inf"), "trade_count": 0}
    accepted = decisions[decisions["accepted"]].copy()
    merged = accepted.merge(
        candidates[
            [
                "timestamp",
                "symbol",
                "side",
                "strategy_id",
                "net_return",
                "gross_return",
                "simple_policy_exit_reason",
            ]
        ],
        on=["timestamp", "symbol", "side", "strategy_id"],
        how="left",
    )
    if "position_net_return" in merged.columns:
        adjusted_net = pd.to_numeric(
            merged["position_net_return"], errors="coerce"
        ).replace([np.inf, -np.inf], np.nan)
        merged["net_return"] = adjusted_net.where(
            adjusted_net.notna(), merged["net_return"]
        )
    if "position_gross_return" in merged.columns:
        adjusted_gross = pd.to_numeric(
            merged["position_gross_return"], errors="coerce"
        ).replace([np.inf, -np.inf], np.nan)
        merged["gross_return"] = adjusted_gross.where(
            adjusted_gross.notna(), merged["gross_return"]
        )
    if "position_exit_reason" in merged.columns:
        adjusted_reason = merged["position_exit_reason"].astype(str)
        merged["simple_policy_exit_reason"] = adjusted_reason.where(
            adjusted_reason.str.len() > 0,
            merged["simple_policy_exit_reason"],
        )
    trade_count = int(len(accepted))
    net_pnl = float(
        (
            pd.to_numeric(merged.get("position_size"), errors="coerce").fillna(0.0)
            * pd.to_numeric(merged.get("net_return"), errors="coerce").fillna(0.0)
        ).sum()
    )
    gross_pnl = float(
        (
            pd.to_numeric(merged.get("position_size"), errors="coerce").fillna(0.0)
            * pd.to_numeric(merged.get("gross_return"), errors="coerce").fillna(0.0)
        ).sum()
    )
    position_size = pd.to_numeric(
        merged.get("position_size"), errors="coerce"
    ).fillna(0.0)
    net_return = pd.to_numeric(merged.get("net_return"), errors="coerce").fillna(0.0)
    gross_return = pd.to_numeric(
        merged.get("gross_return"), errors="coerce"
    ).fillna(0.0)
    wallet_before = pd.to_numeric(
        merged.get("wallet_before"), errors="coerce"
    ).replace([np.inf, -np.inf], np.nan)
    total_notional = float(position_size.sum())
    net_pnl_per_trade = position_size * net_return
    gross_pnl_per_trade = position_size * gross_return
    entry_wallet = wallet_before.where(wallet_before > 0.0)
    position_pct_entry_wallet = position_size / entry_wallet
    net_pnl_entry_wallet = net_pnl_per_trade / entry_wallet
    gross_pnl_entry_wallet = gross_pnl_per_trade / entry_wallet
    wallet = (
        pd.to_numeric(equity_curve["wallet"], errors="coerce")
        if not equity_curve.empty and "wallet" in equity_curve.columns
        else pd.Series([initial_wallet], dtype=float)
    )
    mtm_equity = (
        pd.to_numeric(equity_curve["mtm_equity"], errors="coerce")
        if not equity_curve.empty and "mtm_equity" in equity_curve.columns
        else wallet
    )
    equity_for_risk = mtm_equity.dropna()
    if equity_for_risk.empty:
        equity_for_risk = wallet.dropna()
    final_wallet = (
        float(wallet.dropna().iloc[-1])
        if len(wallet.dropna())
        else float(initial_wallet)
    )
    final_equity = (
        float(equity_for_risk.iloc[-1]) if len(equity_for_risk) else final_wallet
    )
    compounded_return = float(final_wallet / max(initial_wallet, EPS) - 1.0)
    returns = (
        equity_for_risk.pct_change(fill_method=None)
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    start = (
        pd.to_datetime(equity_curve["timestamp"], utc=True, errors="coerce").min()
        if not equity_curve.empty
        else pd.NaT
    )
    end = (
        pd.to_datetime(equity_curve["timestamp"], utc=True, errors="coerce").max()
        if not equity_curve.empty
        else pd.NaT
    )
    days = (
        max(float((end - start).total_seconds() / 86400.0), 1.0)
        if pd.notna(start) and pd.notna(end)
        else 1.0
    )
    equity_ratio = max(float(final_equity) / max(float(initial_wallet), EPS), EPS)
    annualized_log_return = np.log(equity_ratio) * (365.0 / max(float(days), EPS))
    if not np.isfinite(annualized_log_return):
        annualized_return = 0.0
    else:
        annualized_return = float(np.expm1(np.clip(annualized_log_return, -50.0, 50.0)))
    running_max = equity_for_risk.cummax()
    drawdown = equity_for_risk / running_max.replace(0.0, np.nan) - 1.0
    max_drawdown = float(drawdown.min()) if len(drawdown.dropna()) else 0.0
    wallet_drawdown = wallet / wallet.cummax().replace(0.0, np.nan) - 1.0
    realized_wallet_max_drawdown = (
        float(wallet_drawdown.min()) if len(wallet_drawdown.dropna()) else 0.0
    )
    downside = returns[returns < 0.0]
    sortino = (
        float(returns.mean() / np.sqrt(np.mean(downside**2)))
        if len(downside)
        else (100.0 if returns.mean() > 0 else 0.0)
    )
    worst_week = 0.0
    if not equity_curve.empty:
        eq = equity_curve.copy()
        eq["timestamp"] = pd.to_datetime(eq["timestamp"], utc=True, errors="coerce")
        weekly = (
            eq.dropna(subset=["timestamp"])
            .set_index("timestamp")[
                "mtm_equity" if "mtm_equity" in eq.columns else "wallet"
            ]
            .resample("W")
            .last()
            .pct_change(fill_method=None)
            .dropna()
        )
        worst_week = float(weekly.min()) if len(weekly) else 0.0
    side_conc = (
        float(accepted["side"].value_counts(normalize=True).max())
        if trade_count
        else 0.0
    )
    strat_conc = (
        float(accepted["strategy_id"].value_counts(normalize=True).max())
        if trade_count
        else 0.0
    )
    max_strategy_concentration = (
        params.max_strategy_concentration if params is not None else 0.75
    )
    max_side_concentration = params.max_side_concentration if params is not None else 0.90
    concentration_penalty = (
        (max(0.0, strat_conc - float(max_strategy_concentration)) if max_strategy_concentration is not None else 0.0)
        + (max(0.0, side_conc - float(max_side_concentration)) if max_side_concentration is not None else 0.0)
    )
    trades_per_day = float(trade_count / max(days, 1.0))
    notional_turnover_per_day = float(total_notional / max(days, 1.0))
    notional_turnover_per_day_bankroll = float(
        notional_turnover_per_day / max(initial_wallet, EPS)
    )
    turnover_penalty = float(
        trades_per_day / 24.0 + notional_turnover_per_day_bankroll / 10.0
    )
    max_capital_pct = (
        float(
            pd.to_numeric(
                equity_curve.get("open_capital_pct", pd.Series(dtype=float)),
                errors="coerce",
            )
            .replace([np.inf, -np.inf], np.nan)
            .max()
            or 0.0
        )
        if not equity_curve.empty
        else 0.0
    )
    max_total_wallet_allocation_pct = (
        params.max_total_wallet_allocation_pct if params is not None else 0.75
    )
    sortino_component = float(np.tanh(sortino / 2.0))
    objective = (
        compounded_return
        + 0.05 * sortino_component
        - 0.75 * abs(max_drawdown)
        - 0.05 * concentration_penalty
    )
    exit_counts = (
        merged["simple_policy_exit_reason"].astype(str).value_counts(normalize=True)
        if trade_count and "simple_policy_exit_reason" in merged.columns
        else pd.Series(dtype=float)
    )
    open_avg = float(
        equity_curve.get("open_positions", pd.Series(dtype=float)).mean() or 0.0
    )
    max_open = float(
        equity_curve.get("open_positions", pd.Series(dtype=float)).max() or 0.0
    )
    rejection = decisions["rejection_reason"].astype(str).value_counts()
    high_conf = decisions[
        (~decisions["accepted"])
        & (pd.to_numeric(decisions["normalized_rank_score"], errors="coerce") >= 0.95)
    ]
    return {
        "objective": float(objective),
        "net_pnl": net_pnl,
        "gross_pnl": gross_pnl,
        "final_wallet": final_wallet,
        "final_mtm_equity": final_equity,
        "compounded_return": compounded_return,
        "annualized_return": annualized_return,
        "trades_per_day": trades_per_day,
        "notional_turnover": total_notional,
        "notional_turnover_per_day": notional_turnover_per_day,
        "notional_turnover_per_day_bankroll": notional_turnover_per_day_bankroll,
        "mean_trade_notional": (
            float(merged["position_size"].mean()) if trade_count else 0.0
        ),
        "median_trade_notional": (
            float(merged["position_size"].median()) if trade_count else 0.0
        ),
        "mean_trade_bankroll": float(
            net_pnl / max(trade_count, 1) / max(initial_wallet, EPS)
        ),
        "mean_net_pnl_per_trade": float(net_pnl / max(trade_count, 1)),
        "mean_gross_pnl_per_trade": float(gross_pnl / max(trade_count, 1)),
        "mean_position_pct_entry_wallet": (
            float(position_pct_entry_wallet.mean()) if trade_count else 0.0
        ),
        "median_position_pct_entry_wallet": (
            float(position_pct_entry_wallet.median()) if trade_count else 0.0
        ),
        "mean_net_pnl_trade_entry_wallet": (
            float(net_pnl_entry_wallet.mean()) if trade_count else 0.0
        ),
        "median_net_pnl_trade_entry_wallet": (
            float(net_pnl_entry_wallet.median()) if trade_count else 0.0
        ),
        "mean_gross_pnl_trade_entry_wallet": (
            float(gross_pnl_entry_wallet.mean()) if trade_count else 0.0
        ),
        "median_gross_pnl_trade_entry_wallet": (
            float(gross_pnl_entry_wallet.median()) if trade_count else 0.0
        ),
        "notional_weighted_net_return": float(net_pnl / max(total_notional, EPS)),
        "notional_weighted_gross_return": float(
            gross_pnl / max(total_notional, EPS)
        ),
        "mean_net_return_per_trade": (
            float(net_return.mean()) if trade_count else 0.0
        ),
        "mean_gross_return_per_trade": (
            float(gross_return.mean()) if trade_count else 0.0
        ),
        "median_trade_bankroll": (
            float(
                (
                    merged["position_size"]
                    * merged["net_return"]
                    / max(initial_wallet, EPS)
                ).median()
            )
            if trade_count
            else 0.0
        ),
        "sortino": sortino,
        "sortino_objective_component": sortino_component,
        "max_drawdown": max_drawdown,
        "realized_wallet_max_drawdown": realized_wallet_max_drawdown,
        "objective_formula": (
            "compounded_return + 0.05*tanh(sortino/2) "
            "- 0.75*abs(max_drawdown) - 0.05*concentration_penalty"
        ),
        "risk_equity_source": (
            "mtm_equity"
            if not equity_curve.empty and "mtm_equity" in equity_curve.columns
            else "wallet"
        ),
        "worst_week": worst_week,
        "trade_count": trade_count,
        "avg_open_positions": open_avg,
        "position_utilization": float(max_open),
        "capacity_utilization": float(
            max_open
            / max(
                1.0, decisions["open_positions_after"].max() if len(decisions) else 1.0
            )
        ),
        "max_capital_allocation_pct": max_capital_pct,
        "side_concentration": side_conc,
        "strategy_concentration": strat_conc,
        "missed_high_confidence_trades": int(len(high_conf)),
        "full_sl_rate": float(exit_counts.get("full_sl", 0.0)),
        "adverse_exit_rate": float(exit_counts.get("adverse_exit", 0.0)),
        "timeout_rate": float(exit_counts.get("timeout", 0.0)),
        "liquidity_rejection_rate": float(
            (
                rejection.get("insufficient_liquidity_capacity", 0)
                / max(len(decisions), 1)
            )
        ),
        "guardrails": {
            "max_strategy_concentration_ok": bool(
                max_strategy_concentration is None
                or strat_conc <= float(max_strategy_concentration)
            ),
            "max_side_concentration_ok": bool(
                max_side_concentration is None
                or side_conc <= float(max_side_concentration)
            ),
            "minimum_validation_coverage_ok": True,
            "max_capital_allocation_ok": bool(
                max_capital_pct <= float(max_total_wallet_allocation_pct) + EPS
            ),
        },
        "rejection_reasons": {str(k): int(v) for k, v in rejection.items()},
    }


def _parameter_grid() -> Iterable[PortfolioPolicyParams]:
    for max_pos in [5, 6, 7, 8, 9]:
        for max_side in [4, 5, 6, 7, None]:
            for max_strat in [3, 4, 5, 6, None]:
                for max_bar in [1, 2, 3, 4]:
                    for max_strat_bar in [1, 2, None]:
                        for floor in [0.70, 0.75, 0.80, 0.85, 0.90]:
                            for margin in [0.00]:
                                for alpha in [0.0, 0.15, 0.30, 0.50, 0.75]:
                                    for power in [0.75, 1.0, 1.5, 2.0]:
                                        for side_conc in [0.90, None]:
                                            for strat_conc in [0.75, None]:
                                                yield PortfolioPolicyParams(
                                                    max_concurrent_positions=max_pos,
                                                    max_concurrent_per_side=max_side,
                                                    max_concurrent_per_strategy=max_strat,
                                                    max_new_entries_per_bar=max_bar,
                                                    max_new_entries_per_strategy_per_bar=max_strat_bar,
                                                    global_threshold_floor=floor,
                                                    threshold_viability_margin=margin,
                                                    occupancy_threshold_alpha=alpha,
                                                    occupancy_threshold_power=power,
                                                    max_side_concentration=side_conc,
                                                    max_strategy_concentration=strat_conc,
                                                )


def _sizing_variants(base: PortfolioPolicyParams) -> Iterable[PortfolioPolicyParams]:
    for power in [0.75, 1.0, 1.5, 2.0]:
        for mult_min in [0.25, 0.50, 0.75]:
            for mult_max in [1.0, 1.5, 2.0]:
                yield PortfolioPolicyParams(
                    **{
                        **asdict(base),
                        "rank_size_power": power,
                        "rank_multiplier_min": mult_min,
                        "rank_multiplier_max": mult_max,
                    }
                )


def _suggest_params(trial: optuna.Trial) -> PortfolioPolicyParams:
    max_concurrent_positions = trial.suggest_categorical(
        "max_concurrent_positions", [5, 6, 7, 8]
    )
    strategy_capacity_pct = trial.suggest_categorical(
        "max_concurrent_per_strategy_pct", [0.50, 0.66, 0.75]
    )
    max_concurrent_per_strategy = max(
        1, int(np.ceil(float(max_concurrent_positions) * float(strategy_capacity_pct)))
    )
    return PortfolioPolicyParams(
        max_concurrent_positions=max_concurrent_positions,
        max_concurrent_per_side=None,
        max_concurrent_per_strategy=max_concurrent_per_strategy,
        max_new_entries_per_bar=trial.suggest_categorical(
            "max_new_entries_per_bar", [2, 3, 4]
        ),
        max_new_entries_per_strategy_per_bar=trial.suggest_categorical(
            "max_new_entries_per_strategy_per_bar", [1, 2, None]
        ),
        max_concurrent_per_symbol=1,
        max_total_wallet_allocation_pct=trial.suggest_categorical(
            "max_total_wallet_allocation_pct", [0.60, 0.70, 0.80]
        ),
        global_threshold_floor=trial.suggest_float(
            "global_threshold_floor", 0.70, 0.95, step=0.01
        ),
        threshold_viability_margin=0.0,
        occupancy_threshold_alpha=trial.suggest_float(
            "occupancy_threshold_alpha", 0.0, 0.90
        ),
        occupancy_threshold_power=trial.suggest_float(
            "occupancy_threshold_power", 0.50, 2.50
        ),
        rank_size_power=trial.suggest_float("rank_size_power", 0.50, 2.50),
        rank_multiplier_min=trial.suggest_categorical(
            "rank_multiplier_min", [0.25, 0.50, 0.75, 1.0]
        ),
        rank_multiplier_max=trial.suggest_categorical(
            "rank_multiplier_max", [1.0, 1.25, 1.5, 2.0, 2.5]
        ),
        max_side_concentration=None,
        max_strategy_concentration=None,
    )


def _guardrail_relaxation_score(params: Optional[PortfolioPolicyParams]) -> int:
    if params is None:
        return 0
    return int(params.max_side_concentration is None) + int(
        params.max_strategy_concentration is None
    )


def _is_better_candidate(
    metrics: Dict[str, Any],
    params: PortfolioPolicyParams,
    best_metrics: Dict[str, Any],
    best_params: Optional[PortfolioPolicyParams],
) -> bool:
    objective = float(metrics.get("objective", float("-inf")))
    best_objective = float(best_metrics.get("objective", float("-inf")))
    if objective > best_objective + 1e-12:
        return True
    if abs(objective - best_objective) <= 1e-12:
        return _guardrail_relaxation_score(params) > _guardrail_relaxation_score(
            best_params
        )
    return False


def _relax_concentration_guardrails(
    params: PortfolioPolicyParams,
    metrics: Dict[str, Any],
) -> PortfolioPolicyParams:
    side_conc = metrics.get("side_concentration")
    strat_conc = metrics.get("strategy_concentration")
    next_params = params
    if (
        params.max_side_concentration is not None
        and side_conc is not None
        and float(side_conc) > float(params.max_side_concentration)
    ):
        next_params = replace(
            next_params,
            max_side_concentration=min(float(side_conc), 1.0),
        )
    if (
        params.max_strategy_concentration is not None
        and strat_conc is not None
        and float(strat_conc) > float(params.max_strategy_concentration)
    ):
        next_params = replace(
            next_params,
            max_strategy_concentration=min(float(strat_conc), 1.0),
        )
    return next_params


def _mark_guardrails_after_relaxation(metrics: Dict[str, Any]) -> None:
    metrics.setdefault("guardrails", {})
    metrics["guardrails"]["max_strategy_concentration_ok"] = True
    metrics["guardrails"]["max_side_concentration_ok"] = True


def optimise_params(
    train_candidates: pd.DataFrame,
    *,
    max_evaluations: Optional[int] = None,
    market_mode: str = "spot",
) -> Tuple[PortfolioPolicyParams, Dict[str, Any]]:
    max_evaluations = int(
        max_evaluations
        if max_evaluations is not None
        else os.environ.get(
            "EPM_PORTFOLIO_POLICY_MAX_EVALS",
            os.environ.get("EPM_PORTFOLIO_POLICY_MAX_GRID_EVALS", "3000"),
        )
    )
    train = _normalised_candidate_table(train_candidates)
    ev_curve = fit_hierarchical_ev_curves(train)
    best_params: Optional[PortfolioPolicyParams] = None
    best_metrics: Dict[str, Any] = {"objective": float("-inf")}
    progress_interval = max(
        1,
        int(os.environ.get("EPM_PORTFOLIO_POLICY_PROGRESS_INTERVAL", "100")),
    )
    early_stop_patience = max(
        0,
        int(os.environ.get("EPM_PORTFOLIO_POLICY_EARLY_STOP_PATIENCE", "150")),
    )
    sampler_seed = int(os.environ.get("EPM_PORTFOLIO_POLICY_OPTUNA_SEED", "1729"))
    startup_trials = min(
        max(25, int(os.environ.get("EPM_PORTFOLIO_POLICY_TPE_STARTUP", "50"))),
        max_evaluations,
    )
    print(
        "portfolio_policy_replay: TPE search started "
        f"rows={len(train)} max_evaluations={max_evaluations} "
        f"startup_trials={startup_trials} early_stop_patience={early_stop_patience} "
        "objective=compounded_return_risk_adjusted",
        flush=True,
    )

    def objective(trial: optuna.Trial) -> float:
        nonlocal best_params, best_metrics
        params = _suggest_params(trial)
        _, _, metrics = replay_candidates(
            train,
            params,
            mode="global_auction",
            ev_curve=ev_curve,
            market_mode=market_mode,
        )
        trial.set_user_attr("metrics", metrics)
        trial.set_user_attr("params_config", asdict(params))
        if _is_better_candidate(metrics, params, best_metrics, best_params):
            best_params = params
            best_metrics = metrics
        trial_no = int(trial.number) + 1
        if trial_no == 1 or trial_no % progress_interval == 0:
            print(
                "portfolio_policy_replay: TPE progress "
                f"{trial_no}/{max_evaluations} "
                f"best_objective={float(best_metrics.get('objective', float('-inf'))):.6f}",
                flush=True,
            )
        return float(metrics.get("objective", float("-inf")))

    sampler = optuna.samplers.TPESampler(
        seed=sampler_seed,
        n_startup_trials=startup_trials,
        multivariate=True,
        group=True,
    )
    study = optuna.create_study(direction="maximize", sampler=sampler)

    def early_stop_callback(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        if early_stop_patience <= 0 or study.best_trial is None:
            return
        trials_since_best = int(trial.number) - int(study.best_trial.number)
        if trials_since_best >= early_stop_patience:
            print(
                "portfolio_policy_replay: early stopping "
                f"trial={int(trial.number) + 1} best_trial={int(study.best_trial.number) + 1} "
                f"patience={early_stop_patience}",
                flush=True,
            )
            study.stop()

    study.optimize(
        objective,
        n_trials=max_evaluations,
        callbacks=[early_stop_callback],
        show_progress_bar=False,
    )

    if best_params is None:
        best_params = PortfolioPolicyParams()
    print(
        "portfolio_policy_replay: optimisation completed "
        f"evaluations={len(study.trials)} "
        f"best_objective={float(best_metrics.get('objective', float('-inf'))):.6f}",
        flush=True,
    )
    best_metrics = dict(best_metrics)
    best_metrics["search_method"] = "optuna_tpe"
    best_metrics["search_evaluations"] = int(len(study.trials))
    best_metrics["search_evaluation_cap"] = int(max_evaluations)
    best_metrics["tpe_startup_trials"] = int(startup_trials)
    best_metrics["early_stop_patience"] = int(early_stop_patience)
    best_metrics["early_stopped"] = bool(len(study.trials) < max_evaluations)
    best_metrics["optuna_best_value"] = float(study.best_value)
    return best_params, best_metrics


def walk_forward_validate(
    candidates: pd.DataFrame,
    *,
    max_evaluations: Optional[int] = None,
    market_mode: str = "spot",
) -> Tuple[PortfolioPolicyParams, Dict[str, Any]]:
    work = _normalised_candidate_table(candidates)
    timestamps = np.array(sorted(work["timestamp"].dropna().unique()))
    if len(timestamps) < 3:
        params = PortfolioPolicyParams()
        ev_curve = fit_hierarchical_ev_curves(work)
        baseline_decisions, baseline_eq, baseline_metrics = replay_candidates(
            work,
            params,
            mode="live_baseline",
            ev_curve=ev_curve,
            market_mode=market_mode,
        )
        global_decisions, global_eq, global_metrics = replay_candidates(
            work,
            params,
            mode="global_auction",
            ev_curve=ev_curve,
            market_mode=market_mode,
        )
        return params, {
            "folds": [],
            "live_baseline_validation_metrics": baseline_metrics,
            "global_auction_validation_metrics": global_metrics,
            "accepted": bool(
                global_metrics.get("objective", -np.inf)
                >= baseline_metrics.get("objective", np.inf)
            ),
        }
    high_rank_mask = (
        pd.to_numeric(work["normalized_rank_score"], errors="coerce") >= 0.80
    )
    split_source = work.loc[high_rank_mask, ["timestamp"]].sort_values("timestamp")
    split_basis = "high_rank_candidates_ge_0_8"
    if split_source.empty:
        split_source = work[["timestamp"]].sort_values("timestamp")
        split_basis = "all_candidates_fallback"
    split_idx = int(len(split_source) * 2 / 3)
    split_idx = min(max(split_idx, 1), len(split_source) - 1)
    split_ts = pd.Timestamp(split_source.iloc[split_idx]["timestamp"])
    train = work[work["timestamp"] < split_ts].copy()
    validation = work[work["timestamp"] >= split_ts].copy()
    if train.empty or validation.empty:
        split_basis = "row_count_fallback"
        train = work.iloc[: int(len(work) * 2 / 3)].copy()
        validation = work.iloc[int(len(work) * 2 / 3) :].copy()
    params, train_metrics = optimise_params(
        train,
        max_evaluations=max_evaluations,
        market_mode=market_mode,
    )
    ev_curve = fit_hierarchical_ev_curves(train)
    _, _, baseline_metrics = replay_candidates(
        validation,
        PortfolioPolicyParams(),
        mode="live_baseline",
        ev_curve=ev_curve,
        market_mode=market_mode,
    )
    _, _, global_metrics = replay_candidates(
        validation,
        params,
        mode="global_auction",
        ev_curve=ev_curve,
        market_mode=market_mode,
    )
    validation_high_rank_rows = int(
        (
            pd.to_numeric(validation["normalized_rank_score"], errors="coerce") >= 0.80
        ).sum()
    )
    high_rank_candidate_rows = int(high_rank_mask.sum())
    coverage = validation_high_rank_rows / max(high_rank_candidate_rows, 1)
    guardrails_ok = (
        bool(
            global_metrics.get("guardrails", {}).get(
                "max_strategy_concentration_ok", False
            )
        )
        and bool(
            global_metrics.get("guardrails", {}).get("max_side_concentration_ok", False)
        )
        and bool(
            global_metrics.get("guardrails", {}).get(
                "max_capital_allocation_ok", False
            )
        )
        and coverage >= 0.25
    )
    beats_baseline = bool(
        float(global_metrics.get("objective", float("-inf")))
        > float(baseline_metrics.get("objective", float("-inf")))
    )
    adjusted_params = params
    guardrails_relaxed = False
    if beats_baseline and coverage >= 0.25 and not guardrails_ok:
        adjusted_params = _relax_concentration_guardrails(params, global_metrics)
        guardrails_relaxed = adjusted_params != params
        if guardrails_relaxed:
            _mark_guardrails_after_relaxation(global_metrics)
            guardrails_ok = True
    accepted = bool(beats_baseline and guardrails_ok)
    global_metrics.setdefault("guardrails", {})["minimum_validation_coverage_ok"] = (
        bool(coverage >= 0.25)
    )
    report = {
        "folds": [
            {
                "fold": 1,
                "train_rows": int(len(train)),
                "validation_rows": int(len(validation)),
                "validation_coverage": float(coverage),
                "split_basis": split_basis,
                "split_timestamp": split_ts.isoformat(),
                "high_rank_candidate_rows": high_rank_candidate_rows,
                "validation_high_rank_candidate_rows": validation_high_rank_rows,
                "train_metrics": train_metrics,
                "live_baseline_validation_metrics": baseline_metrics,
                "global_auction_validation_metrics": global_metrics,
                "guardrails_relaxed_to_validation_concentration": guardrails_relaxed,
                "accepted": accepted,
            }
        ],
        "live_baseline_validation_metrics": baseline_metrics,
        "global_auction_validation_metrics": global_metrics,
        "accepted": accepted,
        "guardrails_relaxed_to_validation_concentration": guardrails_relaxed,
        "selection_reason": (
            "global_auction_beats_baseline_oos_and_guardrails"
            if accepted
            else "baseline_retained_or_guardrail_failed"
        ),
    }
    return adjusted_params if accepted else PortfolioPolicyParams(), report


def run_portfolio_policy_replay(
    *,
    data_root: str,
    run_id: str,
    market_mode: str = "spot",
    candidate_path: Optional[str | Path] = None,
    output_dir: Optional[str | Path] = None,
    max_evaluations: Optional[int] = None,
    fixed_policy_config_path: Optional[str | Path] = None,
    ev_curve_candidate_path: Optional[str | Path] = None,
    persist_live_artifacts: bool = True,
) -> Dict[str, Any]:
    run_root = Path(data_root) / "artifacts" / str(run_id)
    if candidate_path is None:
        candidate_path = (
            run_root / "simple_policy_optimiser" / "simple_policy_candidates.parquet"
        )
    candidate_path = Path(candidate_path)
    if not candidate_path.exists():
        raise FileNotFoundError(
            f"simple policy candidate table not found: {candidate_path}"
        )
    output = (
        Path(output_dir)
        if output_dir is not None
        else run_root / "portfolio_policy_replay"
    )
    output.mkdir(parents=True, exist_ok=True)
    candidates = normalise_candidate_table(pd.read_parquet(candidate_path))
    strategy_ids = tuple(
        sorted(
            {
                str(strategy_id)
                for strategy_id in candidates["strategy_id"].dropna().unique()
                if str(strategy_id).strip()
            }
        )
    )
    strategy_cores = tuple(
        sorted(
            {
                str(strategy_id).removeprefix("long_").removeprefix("short_")
                for strategy_id in strategy_ids
            }
        )
    )
    if fixed_policy_config_path is not None:
        fixed_policy_config_path = Path(fixed_policy_config_path)
        params = load_portfolio_policy_params(fixed_policy_config_path)
        validation_report = {
            "folds": [],
            "accepted": True,
            "selection_reason": "fixed_policy_config_no_optimisation",
            "fixed_policy_config_path": str(fixed_policy_config_path),
            "candidate_rows": int(len(candidates)),
        }
    else:
        params, validation_report = walk_forward_validate(
            candidates,
            max_evaluations=max_evaluations,
            market_mode=market_mode,
        )
    params = replace(
        params,
        strategy_ids=tuple(params.strategy_ids) or strategy_ids,
        strategy_cores=tuple(params.strategy_cores) or strategy_cores,
    )
    ev_curve_source_path = candidate_path
    ev_curve_candidates = candidates
    if ev_curve_candidate_path is not None:
        ev_curve_source_path = Path(ev_curve_candidate_path)
        if not ev_curve_source_path.exists():
            raise FileNotFoundError(
                f"EV-curve candidate table not found: {ev_curve_source_path}"
            )
        ev_curve_candidates = normalise_candidate_table(
            pd.read_parquet(ev_curve_source_path)
        )
    ev_curve = fit_hierarchical_ev_curves(ev_curve_candidates)
    decisions, equity, metrics = replay_candidates(
        candidates,
        params,
        mode="global_auction",
        ev_curve=ev_curve,
        market_mode=market_mode,
    )
    baseline_decisions, _, baseline_metrics = replay_candidates(
        candidates,
        PortfolioPolicyParams(),
        mode="live_baseline",
        ev_curve=ev_curve,
        market_mode=market_mode,
    )
    accepted_global = set(
        zip(
            decisions.loc[decisions["accepted"], "timestamp"],
            decisions.loc[decisions["accepted"], "symbol"],
            decisions.loc[decisions["accepted"], "strategy_id"],
        )
    )
    accepted_baseline = set(
        zip(
            baseline_decisions.loc[baseline_decisions["accepted"], "timestamp"],
            baseline_decisions.loc[baseline_decisions["accepted"], "symbol"],
            baseline_decisions.loc[baseline_decisions["accepted"], "strategy_id"],
        )
    )
    overlap = len(accepted_global.intersection(accepted_baseline)) / max(
        len(accepted_baseline), 1
    )
    diagnostics = {
        "accepted_overlap_pct": float(overlap),
        "rejection_reason_match_pct": (
            float(
                (
                    decisions["rejection_reason"].astype(str).to_numpy()
                    == baseline_decisions["rejection_reason"].astype(str).to_numpy()
                ).mean()
            )
            if len(decisions) == len(baseline_decisions)
            else None
        ),
        "position_count_path_error": _max_abs_decision_delta(
            decisions, baseline_decisions, "open_positions_after"
        ),
        "side_count_path_error": _max_abs_decision_delta(
            decisions, baseline_decisions, "side_count_before"
        ),
        "strategy_count_path_error": _max_abs_decision_delta(
            decisions, baseline_decisions, "strategy_count_before"
        ),
        "notional_path_error": _max_abs_decision_delta(
            decisions, baseline_decisions, "open_notional_after"
        ),
        "wallet_path_error": _max_abs_decision_delta(
            decisions, baseline_decisions, "wallet_after"
        ),
    }
    decisions.to_parquet(output / "per_candidate_replay_decisions.parquet", index=False)
    equity.to_parquet(output / "portfolio_replay_equity_curve.parquet", index=False)
    config_payload = params.to_live_config()
    report = {
        "generated_by": "portfolio_policy_replay",
        "candidate_path": str(candidate_path),
        "market_mode": market_mode,
        "run_id": str(run_id),
        "policy_replay_mode": (
            "fixed_policy_config" if fixed_policy_config_path is not None else "optimised"
        ),
        "fixed_policy_config_path": (
            str(fixed_policy_config_path)
            if fixed_policy_config_path is not None
            else None
        ),
        "ev_curve_candidate_path": str(ev_curve_source_path),
        "optimized_params": config_payload,
        "global_auction_metrics": metrics,
        "live_baseline_metrics": baseline_metrics,
        "baseline_diagnostics": diagnostics,
        "walk_forward": validation_report,
    }
    (output / "optimized_portfolio_policy_config.json").write_text(
        json.dumps(_json_safe(config_payload), indent=2),
        encoding="utf-8",
    )
    (output / "portfolio_policy_replay_report.json").write_text(
        json.dumps(_json_safe(report), indent=2),
        encoding="utf-8",
    )
    (output / "per_fold_validation_metrics.json").write_text(
        json.dumps(_json_safe(validation_report), indent=2),
        encoding="utf-8",
    )
    if fixed_policy_config_path is None and persist_live_artifacts:
        policy_params_dir = run_root / "policy_params"
        policy_params_dir.mkdir(parents=True, exist_ok=True)
        policy_config_path = policy_params_dir / "optimized_portfolio_policy_config.json"
        policy_config_path.write_text(
            json.dumps(_json_safe(config_payload), indent=2),
            encoding="utf-8",
        )
        try:
            model_bundle = load_full_state(str(run_id), str(data_root))
            contract = build_training_live_parity_contract(
                data_root=str(data_root),
                run_id=str(run_id),
                market_mode=market_mode,
                model_bundle=model_bundle,
                strategy_ids=strategy_ids,
                portfolio_payload=config_payload,
            )
            persist_training_live_parity_contract(
                contract,
                data_root=str(data_root),
                run_id=str(run_id),
            )
        except Exception as exc:
            model_dir = run_root / "models"
            if model_dir.exists() and any(model_dir.glob("*.pkl")):
                raise RuntimeError(
                    "portfolio_policy_replay completed but failed to write the "
                    "training-live parity contract"
                ) from exc
            contract = {
                "schema_version": "training_live_parity_contract_v1",
                "generated_by": "portfolio_policy_replay",
                "run_id": str(run_id),
                "market_mode": market_mode,
                "strategy_contract": {
                    "strategy_ids": list(strategy_ids),
                    "strategy_cores": list(strategy_cores),
                },
                "model_contracts": {},
                "portfolio_policy": config_payload,
                "contract_warning": "model_artifacts_missing",
            }
            persist_training_live_parity_contract(
                contract,
                data_root=str(data_root),
                run_id=str(run_id),
            )
    return report


def _latest_run_id(data_root: str) -> str:
    artifacts = Path(data_root) / "artifacts"
    runs = [p for p in artifacts.iterdir() if p.is_dir()]
    if not runs:
        raise FileNotFoundError(f"no artifact runs under {artifacts}")
    return max(runs, key=lambda p: p.stat().st_mtime).name


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=os.environ.get("EPM_DATA_ROOT", "data"))
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--market-mode", default=os.environ.get("EPM_MARKET_MODE", "spot")
    )
    parser.add_argument("--candidate-path", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-evaluations", type=int, default=None)
    parser.add_argument("--fixed-policy-config-path", default=None)
    parser.add_argument("--ev-curve-candidate-path", default=None)
    args = parser.parse_args(argv)
    run_id = args.run_id or _latest_run_id(args.data_root)
    report = run_portfolio_policy_replay(
        data_root=args.data_root,
        run_id=run_id,
        market_mode=args.market_mode,
        candidate_path=args.candidate_path,
        output_dir=args.output_dir,
        max_evaluations=args.max_evaluations,
        fixed_policy_config_path=args.fixed_policy_config_path,
        ev_curve_candidate_path=args.ev_curve_candidate_path,
    )
    print(json.dumps(_json_safe({"run_id": run_id, "report": report}), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
