"""Unified trailing-stop and profit management for live trading."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional

from src.utils.logger import system_logger
from src.utils.tprint import tprint_info, tprint_warning, tprint_error, tprint_success

logger = system_logger.getChild("UnifiedTrailingManager")

class TrailingAction(Enum):
    """Possible trailing management actions."""

    NONE = "none"
    MOVE_STOP = "move_stop"
    PARTIAL_EXIT = "partial_exit"
    FULL_EXIT = "full_exit"

@dataclass
class TrailingDecision:
    """Result of evaluating a position."""

    action: TrailingAction
    stop_price: Optional[float] = None
    exit_fraction: float = 0.0
    reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TrailingState:
    """Internal state for a monitored position."""

    position_id: str
    side: str
    entry_price: float
    entry_time: datetime
    quantity: float
    entry_atr: float
    entry_sigma: float
    trailing_stop: float
    profit_target: float
    profit_buffer: float
    regime: str
    bar_duration: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    best_price: float = field(init=False)
    worst_price: float = field(init=False)
    tp_trail_active: bool = False
    partial_taken: bool = False
    breakeven_set: bool = False
    tightened: bool = False
    last_update: Optional[datetime] = None

    def __post_init__(self) -> None:
        self.best_price = self.entry_price
        self.worst_price = self.entry_price

    @property
    def side_multiplier(self) -> float:
        return 1.0 if self.side.lower() == "long" else -1.0

class UnifiedTrailingManager:
    """Unified trailing-stop management aligned with optimizer parameters."""

    DEFAULT_CONFIG: Dict[str, Any] = {
        "profit_buffer_atr_multiplier": 0.6,
        "profit_buffer_min_fraction": 0.001,
        "trail_base_atr_multiplier": 0.8,
        "breakeven_activation_atr": 1.0,
        "trail_activation_atr": 1.0,
        "tp_trail_activation_atr": 2.0,
        "tp_trail_trigger_atr": 2.5,
        "partial_take_fraction": 0.5,
        "drawdown_tighten_atr": 0.8,
        "drawdown_exit_atr": 1.2,
        "tighten_trail_atr": 0.5,
        "volatility_tighten_threshold": 0.7,
        "volatility_tighten_adjustment": 0.3,
        "volatility_loosen_threshold": 1.3,
        "volatility_loosen_adjustment": 0.2,
        "time_decay_bars": 8,
        "time_decay_threshold_atr": 0.3,
        "bar_duration_seconds": 900,
        "ml_confidence_tighten_threshold": 0.3,
        "ml_confidence_tighten_atr": 0.3,
        "ml_regime_partial_fraction": 0.5,
        "max_trailing_backstep_atr": 4.0,
        "regime_bands": {
            "low": {"sl_atr": 1.2, "tp_atr": 2.0, "trail_atr": 0.8, "tp_trail": 2.2},
            "normal": {"sl_atr": 1.5, "tp_atr": 2.5, "trail_atr": 1.0, "tp_trail": 2.5},
            "high": {"sl_atr": 1.8, "tp_atr": 3.0, "trail_atr": 1.2, "tp_trail": 3.0},
        },
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = self._merge_config(config or {})
        self.positions: Dict[str, TrailingState] = {}

    def register_position(
        self,
        position_id: str,
        *,
        side: str,
        entry_price: float,
        entry_time: datetime,
        quantity: float,
        entry_atr: float,
        entry_sigma: float,
        bar_duration: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> TrailingState:
        """Register a new position for monitoring."""

        if entry_atr <= 0:
            raise ValueError("entry_atr must be positive for trailing management")

        regime = self._determine_regime(entry_atr, entry_price)
        regime_params = self.config["regime_bands"].get(regime, {})

        stop_atr = regime_params.get("sl_atr", 1.5)
        tp_atr = regime_params.get("tp_atr", 2.5)
        trail_atr = regime_params.get("trail_atr", self.config["trail_base_atr_multiplier"])
        tp_trail = regime_params.get("tp_trail", self.config["tp_trail_trigger_atr"])

        profit_buffer = max(
            self.config["profit_buffer_atr_multiplier"] * entry_atr,
            self.config["profit_buffer_min_fraction"] * entry_price,
        )

        side_multiplier = 1.0 if side.lower() == "long" else -1.0
        initial_stop = entry_price - side_multiplier * stop_atr * entry_atr
        profit_target = entry_price + side_multiplier * tp_atr * entry_atr

        state = TrailingState(
            position_id=position_id,
            side=side,
            entry_price=entry_price,
            entry_time=entry_time,
            quantity=quantity,
            entry_atr=entry_atr,
            entry_sigma=entry_sigma,
            trailing_stop=initial_stop,
            profit_target=profit_target,
            profit_buffer=profit_buffer,
            regime=regime,
            bar_duration=bar_duration or self.config["bar_duration_seconds"],
            metadata=metadata or {},
        )

        state.metadata.setdefault("trail_atr_multiplier", trail_atr)
        state.metadata.setdefault("tp_trail_trigger", tp_trail)

        self.positions[position_id] = state
        logger.info(
            "📌 Registered position %s side=%s regime=%s stop=%.4f target=%.4f",
            position_id,
            side,
            regime,
            state.trailing_stop,
            state.profit_target,
        )
        tprint_info(f"📌 Registered position {position_id} ({side}) - regime={regime}, stop={state.trailing_stop:.4f}, target={state.profit_target:.4f}")
        return state

    def remove_position(self, position_id: str) -> None:
        """Remove a position from tracking."""
        removed = self.positions.pop(position_id, None)
        if removed:
            tprint_info(f"🗑️ Removed position {position_id} from tracking")
        else:
            tprint_warning(f"⚠️ Position {position_id} not found for removal")

    # ------------------------------------------------------------------
    # Evaluation logic
    # ------------------------------------------------------------------

    def evaluate_position(
        self,
        position_id: str,
        *,
        price: float,
        atr: float,
        sigma: float,
        momentum: float,
        rsi: float,
        vol_slope: float,
        timestamp: datetime,
        ml_context: Optional[Dict[str, Any]] = None,
    ) -> TrailingDecision:
        """Evaluate trailing actions for a position."""

        state = self.positions.get(position_id)
        if state is None:
            tprint_warning(f"⚠️ Position {position_id} not found for evaluation")
            return TrailingDecision(TrailingAction.NONE, reason="position_not_found")

        if atr <= 0 or price <= 0:
            tprint_error(f"❌ Invalid inputs for position {position_id}: atr={atr}, price={price}")
            return TrailingDecision(TrailingAction.NONE, reason="invalid_inputs")

        state.last_update = timestamp
        self._update_extrema(state, price)

        side_mult = state.side_multiplier
        delta_price = (price - state.entry_price) * side_mult
        atr_move = delta_price / atr if atr > 0 else 0.0

        # Determine baseline trailing distance
        base_multiplier = state.metadata.get(
            "trail_atr_multiplier", self.config["trail_base_atr_multiplier"]
        )
        trail_distance = max(base_multiplier * atr, state.profit_buffer)

        trail_distance = self._apply_volatility_adjustment(
            trail_distance, atr, sigma, state.entry_sigma
        )

        trail_distance = self._apply_ml_adjustments(
            trail_distance, atr, ml_context, state
        )

        decision = self._check_breakeven(state, atr_move)
        if decision.action != TrailingAction.NONE:
            tprint_info(f"📊 Position {position_id}: Breakeven decision - {decision.reason}")
            return decision

        decision = self._update_trailing_stop(
            state, price, trail_distance, atr_move
        )
        if decision.action != TrailingAction.NONE:
            tprint_info(f"📊 Position {position_id}: Trailing stop updated - {decision.reason}, stop={decision.stop_price:.4f}")
            return decision

        decision = self._check_tp_trail(
            state,
            atr_move,
            momentum,
            rsi,
            vol_slope,
            ml_context,
        )
        if decision.action != TrailingAction.NONE:
            tprint_info(f"📊 Position {position_id}: TP trail decision - {decision.reason}, exit_fraction={decision.exit_fraction:.2%}")
            return decision

        decision = self._check_drawdown(state, price, atr)
        if decision.action != TrailingAction.NONE:
            tprint_warning(f"⚠️ Position {position_id}: Drawdown decision - {decision.reason}")
            return decision

        decision = self._check_time_decay(state, timestamp, atr_move)
        if decision.action != TrailingAction.NONE:
            tprint_warning(f"⚠️ Position {position_id}: Time decay decision - {decision.reason}")
            return decision

        return TrailingDecision(TrailingAction.NONE, reason="hold")

    # ------------------------------------------------------------------
    # Internal utilities
    # ------------------------------------------------------------------

    def _merge_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        merged = {**self.DEFAULT_CONFIG}
        merged.update({k: v for k, v in config.items() if k != "regime_bands"})
        regime_config = self.DEFAULT_CONFIG["regime_bands"].copy()
        regime_config.update(config.get("regime_bands", {}))
        merged["regime_bands"] = regime_config
        return merged

    def _determine_regime(self, atr: float, price: float) -> str:
        if price <= 0:
            return "normal"

        atr_pct = atr / price
        if atr_pct < 0.0025:
            return "low"
        if atr_pct > 0.006:
            return "high"
        return "normal"

    def _update_extrema(self, state: TrailingState, price: float) -> None:
        if state.side.lower() == "long":
            state.best_price = max(state.best_price, price)
            state.worst_price = min(state.worst_price, price)
        else:
            state.best_price = min(state.best_price, price)
            state.worst_price = max(state.worst_price, price)

    def _apply_volatility_adjustment(
        self, distance: float, atr: float, sigma: float, entry_sigma: float
    ) -> float:
        if sigma <= 0 or entry_sigma <= 0:
            return distance

        if sigma < entry_sigma * self.config["volatility_tighten_threshold"]:
            distance -= self.config["volatility_tighten_adjustment"] * atr
        elif sigma > entry_sigma * self.config["volatility_loosen_threshold"]:
            distance += self.config["volatility_loosen_adjustment"] * atr

        return max(distance, 0.1 * atr)

    def _apply_ml_adjustments(
        self,
        distance: float,
        atr: float,
        ml_context: Optional[Dict[str, Any]],
        state: TrailingState,
    ) -> float:
        if not ml_context:
            return distance

        entry_conf = ml_context.get("entry", {}).get("analyst_confidence")
        current_conf = ml_context.get("analyst_confidence")
        if (
            entry_conf is not None
            and current_conf is not None
            and entry_conf - current_conf >= self.config["ml_confidence_tighten_threshold"]
        ):
            distance -= self.config["ml_confidence_tighten_atr"] * atr

        tact_momentum = ml_context.get("tactician_momentum")
        if tact_momentum is not None:
            if state.side.lower() == "long" and tact_momentum < 0:
                state.tp_trail_active = True
            elif state.side.lower() == "short" and tact_momentum > 0:
                state.tp_trail_active = True

        if ml_context.get("regime_changed"):
            if not state.metadata.get("regime_partial_requested"):
                state.metadata["regime_partial_requested"] = True
                logger.info(
                    "⚠️ Regime change detected for %s – scheduling partial exit",
                    state.position_id,
                )
                tprint_warning(f"⚠️ Regime change detected for {state.position_id} – scheduling partial exit")

        return max(distance, state.profit_buffer)

    def _check_breakeven(self, state: TrailingState, atr_move: float) -> TrailingDecision:
        if state.breakeven_set:
            return TrailingDecision(TrailingAction.NONE)

        if atr_move >= self.config["breakeven_activation_atr"]:
            state.breakeven_set = True
            side_mult = state.side_multiplier
            new_stop = state.entry_price
            updated = self._set_trailing_stop(state, new_stop, force=True)
            if updated:
                tprint_success(f"✅ Breakeven activated for {state.position_id}, stop moved to entry price")
                return TrailingDecision(
                    TrailingAction.MOVE_STOP,
                    stop_price=state.trailing_stop,
                    reason="breakeven",
                )

        return TrailingDecision(TrailingAction.NONE)

    def _update_trailing_stop(
        self,
        state: TrailingState,
        price: float,
        trail_distance: float,
        atr_move: float,
    ) -> TrailingDecision:
        if atr_move < self.config["trail_activation_atr"]:
            return TrailingDecision(TrailingAction.NONE)

        candidate = price - state.side_multiplier * trail_distance
        updated = self._set_trailing_stop(state, candidate)
        if updated:
            tprint_info(f"📈 Trailing stop updated for {state.position_id}: {state.trailing_stop:.4f}")
            return TrailingDecision(
                TrailingAction.MOVE_STOP,
                stop_price=state.trailing_stop,
                reason="trail_update",
            )
        return TrailingDecision(TrailingAction.NONE)

    def _set_trailing_stop(
        self, state: TrailingState, candidate: float, force: bool = False
    ) -> bool:
        side = state.side.lower()
        if side == "long":
            min_allowed = state.entry_price - self.config["max_trailing_backstep_atr"] * state.entry_atr
            candidate = max(candidate, min_allowed)
            if force or candidate > state.trailing_stop:
                state.trailing_stop = candidate
                return True
        else:
            max_allowed = state.entry_price + self.config["max_trailing_backstep_atr"] * state.entry_atr
            candidate = min(candidate, max_allowed)
            if force or candidate < state.trailing_stop:
                state.trailing_stop = candidate
                return True
        return False

    def _check_tp_trail(
        self,
        state: TrailingState,
        atr_move: float,
        momentum: float,
        rsi: float,
        vol_slope: float,
        ml_context: Optional[Dict[str, Any]],
    ) -> TrailingDecision:
        if state.metadata.get("regime_partial_requested") and not state.partial_taken:
            state.partial_taken = True
            state.metadata["regime_partial_requested"] = False
            fraction = self.config["ml_regime_partial_fraction"]
            return TrailingDecision(
                TrailingAction.PARTIAL_EXIT,
                exit_fraction=fraction,
                reason="regime_partial",
            )

        trigger = state.metadata.get("tp_trail_trigger", self.config["tp_trail_trigger_atr"])

        if atr_move >= self.config["tp_trail_activation_atr"]:
            state.tp_trail_active = True

        if not state.tp_trail_active or state.partial_taken:
            return TrailingDecision(TrailingAction.NONE)

        momentum_condition = (
            momentum < 0 if state.side.lower() == "long" else momentum > 0
        )

        rsi_condition = False
        if state.side.lower() == "long":
            rsi_condition = rsi < 45
        else:
            rsi_condition = rsi > 55

        slope_condition = vol_slope < 0

        ml_trigger = False
        if ml_context:
            current_conf = ml_context.get("analyst_confidence")
            entry_conf = ml_context.get("entry", {}).get("analyst_confidence")
            if (
                current_conf is not None
                and entry_conf is not None
                and current_conf < entry_conf
            ):
                ml_trigger = True

        if atr_move >= trigger and (momentum_condition or rsi_condition or slope_condition or ml_trigger):
            state.partial_taken = True
            fraction = self.config["partial_take_fraction"]
            tprint_info(f"💰 Partial exit triggered for {state.position_id}: {fraction:.2%} at {atr_move:.2f} ATR")
            return TrailingDecision(
                TrailingAction.PARTIAL_EXIT,
                exit_fraction=fraction,
                reason="tp_trail_partial",
            )

        return TrailingDecision(TrailingAction.NONE)

    def _check_drawdown(
        self, state: TrailingState, price: float, atr: float
    ) -> TrailingDecision:
        if atr <= 0:
            return TrailingDecision(TrailingAction.NONE)

        if state.side.lower() == "long":
            retrace = state.best_price - price
        else:
            retrace = price - state.best_price

        retrace_atr = retrace / atr if atr > 0 else 0.0

        if retrace_atr >= self.config["drawdown_exit_atr"]:
            tprint_warning(f"⚠️ Drawdown exit triggered for {state.position_id}: {retrace_atr:.2f} ATR retrace")
            return TrailingDecision(
                TrailingAction.FULL_EXIT, reason="drawdown_exit"
            )

        if retrace_atr >= self.config["drawdown_tighten_atr"] and not state.tightened:
            state.tightened = True
            tighten_distance = max(
                state.profit_buffer, self.config["tighten_trail_atr"] * atr
            )
            candidate = price - state.side_multiplier * tighten_distance
            updated = self._set_trailing_stop(state, candidate, force=True)
            if updated:
                tprint_warning(f"⚠️ Trailing stop tightened for {state.position_id} due to drawdown: {state.trailing_stop:.4f}")
                return TrailingDecision(
                    TrailingAction.MOVE_STOP,
                    stop_price=state.trailing_stop,
                    reason="drawdown_tighten",
                )

        return TrailingDecision(TrailingAction.NONE)

    def _check_time_decay(
        self, state: TrailingState, timestamp: datetime, atr_move: float
    ) -> TrailingDecision:
        if state.bar_duration <= 0:
            return TrailingDecision(TrailingAction.NONE)

        held_seconds = (timestamp - state.entry_time).total_seconds()
        bars_held = held_seconds / state.bar_duration

        if (
            bars_held >= self.config["time_decay_bars"]
            and atr_move < self.config["time_decay_threshold_atr"]
        ):
            tprint_warning(f"⚠️ Time decay exit triggered for {state.position_id}: held {bars_held:.1f} bars with {atr_move:.2f} ATR move")
            return TrailingDecision(
                TrailingAction.FULL_EXIT, reason="time_decay"
            )

        return TrailingDecision(TrailingAction.NONE)
