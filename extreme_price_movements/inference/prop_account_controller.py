"""Fail-closed prop-account risk overlay for current inference outputs.

The controller does not rescore models or place orders.  It converts an already
causal inference candidate plus the current marked account/position state into
an auditable entry, resize, reject, pause, or flatten decision.  The caller is
responsible for passing approved entries to the existing TradeExecutor.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import pandas as pd


def _utc(value: Any) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def _finite(value: Any, default: float = math.nan) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


@dataclass(frozen=True)
class PropAccountPolicy:
    schema_version: str = "prop_account_overlay_v2"
    max_total_wallet_allocation_fraction: float = 0.40
    max_marked_notional_fraction: float = 0.45
    operational_stop_risk_fraction: float = 0.0075
    max_stop_risk_fraction: float = 0.01
    max_position_stop_risk_fraction: float = 0.002
    convex_loss_budget_enabled: bool = False
    convex_loss_budget_fraction: float = 0.75
    convex_loss_budget_power: float = 1.50
    internal_concurrent_loss_limit_fraction: float = 0.015
    firm_max_daily_loss_fraction: float = 0.03
    firm_max_drawdown_fraction: float = 0.06
    drawdown_reference_mode: str = "starting_balance"
    stop_loss_risk_margin_multiplier: float = 1.0
    stop_loss_risk_margin_bps: float = 0.0
    max_single_opportunity_risk_share: float = 0.25
    max_archetype_batch_risk_share: float = 0.40
    max_side_batch_risk_share: float = 0.65
    archetype_risk_increment_per_position: float = 0.25
    max_archetype_risk_multiplier: float = 1.75
    require_policy_archetype_for_risk: bool = True
    base_min_rank: float = 0.95
    base_min_net_ev: float = 0.007
    base_entry_notional_fraction: float = 0.10
    max_position_notional_fraction: float = 0.10
    rank_multiplier_min: float = 0.80
    rank_multiplier_max: float = 1.60
    rank_size_power: float = 2.00
    rank_sizing_cap: float = 0.99
    min_bayesian_size_multiplier: float = 0.65
    max_bayesian_size_multiplier: float = 1.20
    require_raw_bayesian_sizing: bool = True
    min_entry_quote: float = 3.0
    max_leverage: float = 2.0
    max_gross_marked_notional_fraction: float = 1.0
    stop_entries_daily_return: float = -0.0075
    flatten_daily_return: float = -0.0125
    stop_entries_peak_drawdown: float = -0.03
    flatten_peak_drawdown: float = -0.04
    entry_pause_hours: float = 12.0
    flatten_pause_hours: float = 24.0
    max_signal_age_minutes: float = 20.0
    require_whitelist: bool = True
    require_stop: bool = True
    require_l2: bool = True
    min_l2_capacity_weight: float = 0.25
    max_l2_slippage_bps: float = 50.0
    drawdown_tiers: tuple[tuple[float, float, float], ...] = (
        (-0.0025, 0.965, 0.35),
        (-0.0050, 0.980, 0.25),
    )

    def validate(self) -> None:
        if not 0.40 <= self.max_marked_notional_fraction <= 0.50:
            raise ValueError("max_marked_notional_fraction must be within [0.40, 0.50]")
        if not (
            0.0
            < self.max_total_wallet_allocation_fraction
            <= self.max_marked_notional_fraction
        ):
            raise ValueError(
                "max_total_wallet_allocation_fraction must be positive and no "
                "greater than max_marked_notional_fraction"
            )
        if not 0 < self.max_stop_risk_fraction <= 0.015:
            raise ValueError("max_stop_risk_fraction must be within (0, 0.015]")
        if not (
            0.0
            < self.max_position_stop_risk_fraction
            <= self.operational_stop_risk_fraction
            <= self.max_stop_risk_fraction
        ):
            raise ValueError(
                "stop-risk fractions must satisfy position <= operational <= hard"
            )
        if self.archetype_risk_increment_per_position < 0.0:
            raise ValueError("archetype risk increment must be non-negative")
        if self.max_archetype_risk_multiplier < 1.0:
            raise ValueError("max archetype risk multiplier must be at least 1")
        if self.max_leverage > 2.0 or self.max_leverage <= 0:
            raise ValueError("max_leverage must be within (0, 2]")
        if not 0 < self.max_gross_marked_notional_fraction <= 2.0:
            raise ValueError("max_gross_marked_notional_fraction must be within (0, 2]")
        if not 0 < self.convex_loss_budget_fraction < 1.0:
            raise ValueError("convex_loss_budget_fraction must be within (0, 1)")
        if self.convex_loss_budget_power < 1.0:
            raise ValueError("convex_loss_budget_power must be at least 1")
        if not 0 < self.internal_concurrent_loss_limit_fraction < 0.03:
            raise ValueError(
                "internal_concurrent_loss_limit_fraction must be within (0, 0.03)"
            )
        if not 0 < self.firm_max_daily_loss_fraction <= 0.03:
            raise ValueError("firm_max_daily_loss_fraction must be within (0, 0.03]")
        if not 0 < self.firm_max_drawdown_fraction <= 0.06:
            raise ValueError("firm_max_drawdown_fraction must be within (0, 0.06]")
        if self.drawdown_reference_mode not in {"high_water", "starting_balance"}:
            raise ValueError(
                "drawdown_reference_mode must be high_water or starting_balance"
            )
        if self.stop_loss_risk_margin_multiplier < 1.0:
            raise ValueError("stop_loss_risk_margin_multiplier must be at least 1")
        if self.stop_loss_risk_margin_bps < 0.0:
            raise ValueError("stop_loss_risk_margin_bps must be non-negative")
        for name, value in (
            (
                "max_single_opportunity_risk_share",
                self.max_single_opportunity_risk_share,
            ),
            ("max_archetype_batch_risk_share", self.max_archetype_batch_risk_share),
            ("max_side_batch_risk_share", self.max_side_batch_risk_share),
        ):
            if not 0 < value <= 1.0:
                raise ValueError(f"{name} must be within (0, 1]")
        if not 0 < self.max_position_notional_fraction <= 0.15:
            raise ValueError("max_position_notional_fraction must be within (0, 0.15]")
        if not self.base_min_rank < self.rank_sizing_cap <= 0.99:
            raise ValueError(
                "rank_sizing_cap must be above base_min_rank and at most 0.99"
            )
        if self.rank_size_power <= 1.0:
            raise ValueError("rank_size_power must exceed 1 for convex sizing")
        if not (
            0.0
            <= self.min_bayesian_size_multiplier
            <= self.max_bayesian_size_multiplier
        ):
            raise ValueError("invalid Bayesian sizing multiplier bounds")
        if self.flatten_daily_return >= self.stop_entries_daily_return:
            raise ValueError(
                "flatten_daily_return must be below stop_entries_daily_return"
            )

    @classmethod
    def from_json(cls, path: Path) -> "PropAccountPolicy":
        raw = json.loads(path.read_text())
        if "drawdown_tiers" in raw:
            raw["drawdown_tiers"] = tuple(tuple(x) for x in raw["drawdown_tiers"])
        out = cls(**raw)
        out.validate()
        return out


@dataclass(frozen=True)
class MarkedPosition:
    symbol: str
    side: str
    marked_notional: float
    mark_price: float
    stop_price: float
    policy_archetype: Optional[str] = None
    leverage: float = 1.0

    @property
    def stop_distance_fraction(self) -> float:
        if self.mark_price <= 0:
            return math.inf
        if self.side.lower() == "long":
            return max((self.mark_price - self.stop_price) / self.mark_price, 0.0)
        return max((self.stop_price - self.mark_price) / self.mark_price, 0.0)

    @property
    def stop_risk(self) -> float:
        return abs(self.marked_notional) * self.stop_distance_fraction

    @property
    def invested_amount(self) -> float:
        leverage = self.leverage if math.isfinite(self.leverage) else 1.0
        return abs(self.marked_notional) / max(leverage, 1e-12)


@dataclass(frozen=True)
class AccountSnapshot:
    timestamp: pd.Timestamp
    equity: float
    positions: tuple[MarkedPosition, ...] = ()
    day_start_equity: Optional[float] = None
    high_water_equity: Optional[float] = None
    starting_equity: Optional[float] = None

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "AccountSnapshot":
        positions = tuple(MarkedPosition(**p) for p in raw.get("positions", []))
        return cls(
            timestamp=_utc(raw["timestamp"]),
            equity=float(raw["equity"]),
            positions=positions,
            day_start_equity=(
                float(raw["day_start_equity"])
                if raw.get("day_start_equity") is not None
                else None
            ),
            high_water_equity=(
                float(raw["high_water_equity"])
                if raw.get("high_water_equity") is not None
                else None
            ),
            starting_equity=(
                float(raw["starting_equity"])
                if raw.get("starting_equity") is not None
                else None
            ),
        )

    @property
    def marked_notional(self) -> float:
        return sum(abs(p.marked_notional) for p in self.positions)

    @property
    def marked_invested_amount(self) -> float:
        """Actual wallet committed, excluding effective leverage."""
        return sum(p.invested_amount for p in self.positions)

    @property
    def stop_risk(self) -> float:
        return sum(p.stop_risk for p in self.positions)


@dataclass(frozen=True)
class L2Capacity:
    capacity_quote: float
    capacity_weight: float
    expected_slippage_bps: Optional[float] = None
    reject_reason: Optional[str] = None


@dataclass
class ControllerState:
    utc_day: str
    day_start_equity: float
    high_water_equity: float
    starting_equity: float
    cooldown_until: Optional[str] = None
    flatten_latched: bool = False
    processed_signal_keys: list[str] = field(default_factory=list)

    @classmethod
    def initialise(cls, snapshot: AccountSnapshot) -> "ControllerState":
        return cls(
            snapshot.timestamp.date().isoformat(),
            snapshot.day_start_equity or snapshot.equity,
            snapshot.high_water_equity or snapshot.equity,
            snapshot.starting_equity or snapshot.equity,
        )

    @classmethod
    def load(cls, path: Path, snapshot: AccountSnapshot) -> "ControllerState":
        if not path.exists():
            return cls.initialise(snapshot)
        raw = json.loads(path.read_text())
        # State files written before the static-drawdown contract did not
        # persist the evaluation starting balance. Prefer the explicit account
        # snapshot; the old day-start value is the conservative fallback.
        raw.setdefault(
            "starting_equity",
            snapshot.starting_equity or raw.get("day_start_equity") or snapshot.equity,
        )
        return cls(**raw)

    def roll(self, snapshot: AccountSnapshot) -> None:
        day = snapshot.timestamp.date().isoformat()
        if day != self.utc_day:
            self.utc_day = day
            self.day_start_equity = snapshot.day_start_equity or snapshot.equity
            self.flatten_latched = False
        self.high_water_equity = max(
            self.high_water_equity,
            snapshot.high_water_equity or snapshot.equity,
        )
        self.processed_signal_keys = self.processed_signal_keys[-5000:]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(asdict(self), indent=2, sort_keys=True) + "\n")
        tmp.replace(path)


@dataclass(frozen=True)
class RiskDecision:
    action: str
    reason: str
    symbol: Optional[str]
    approved_notional: float
    requested_notional: float
    effective_min_rank: float
    daily_return: float
    peak_drawdown: float
    marked_notional_before: float
    stop_risk_before: float
    reserved_stop_risk_before: float = 0.0
    l2: Optional[dict[str, Any]] = None
    sizing: Optional[dict[str, Any]] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def canonical_symbol(value: Any) -> str:
    return str(value or "").strip().upper()


class PropAccountController:
    def __init__(self, policy: PropAccountPolicy, whitelist: Iterable[str]):
        policy.validate()
        self.policy = policy
        self.whitelist = {canonical_symbol(v) for v in whitelist if str(v).strip()}

    def account_action(
        self, snapshot: AccountSnapshot, state: ControllerState
    ) -> RiskDecision:
        state.roll(snapshot)
        daily = snapshot.equity / max(state.day_start_equity, 1e-12) - 1.0
        drawdown = self._contract_drawdown(snapshot, state)
        now = snapshot.timestamp
        flatten = (
            daily <= self.policy.flatten_daily_return + 1e-12
            or drawdown <= self.policy.flatten_peak_drawdown + 1e-12
        )
        if flatten:
            state.flatten_latched = True
            state.cooldown_until = (
                now + timedelta(hours=self.policy.flatten_pause_hours)
            ).isoformat()
            return self._decision(
                "flatten", "hard_drawdown_limit", snapshot, daily, drawdown
            )
        stopped = (
            daily <= self.policy.stop_entries_daily_return + 1e-12
            or drawdown <= self.policy.stop_entries_peak_drawdown + 1e-12
        )
        if stopped:
            until = now + timedelta(hours=self.policy.entry_pause_hours)
            previous = _utc(state.cooldown_until) if state.cooldown_until else now
            state.cooldown_until = max(previous, until).isoformat()
            return self._decision(
                "pause", "entry_drawdown_limit", snapshot, daily, drawdown
            )
        if state.flatten_latched:
            return self._decision(
                "pause", "flatten_latched_for_utc_day", snapshot, daily, drawdown
            )
        if state.cooldown_until and now < _utc(state.cooldown_until):
            return self._decision("pause", "cooldown_active", snapshot, daily, drawdown)
        return self._decision(
            "allow", "account_limits_clear", snapshot, daily, drawdown
        )

    def evaluate_entry(
        self,
        candidate: Mapping[str, Any],
        snapshot: AccountSnapshot,
        state: ControllerState,
        l2: Optional[L2Capacity],
    ) -> RiskDecision:
        gate = self.account_action(snapshot, state)
        if gate.action != "allow":
            return gate
        symbol = canonical_symbol(candidate.get("symbol"))
        daily, peak = gate.daily_return, gate.peak_drawdown
        min_rank, max_alloc = self._dynamic_limits(daily, peak)
        sizing = self.sizing_breakdown(candidate, snapshot, min_rank)
        requested = float(sizing["requested_notional"])
        base = dict(
            symbol=symbol or None,
            requested_notional=requested,
            effective_min_rank=min_rank,
            daily_return=daily,
            peak_drawdown=peak,
            marked_notional_before=snapshot.marked_notional,
            stop_risk_before=snapshot.stop_risk,
            reserved_stop_risk_before=self.reserved_stop_risk(snapshot.positions),
            sizing=sizing,
        )
        if self.policy.require_whitelist and symbol not in self.whitelist:
            return RiskDecision(
                "reject",
                "symbol_not_whitelisted",
                approved_notional=0.0,
                l2=None,
                **base,
            )
        if symbol in {canonical_symbol(p.symbol) for p in snapshot.positions}:
            return RiskDecision(
                "reject",
                "symbol_already_open",
                approved_notional=0.0,
                l2=None,
                **base,
            )
        if candidate.get("passed_rank_gate") is False:
            return RiskDecision(
                "reject",
                "upstream_policy_gate_failed",
                approved_notional=0.0,
                l2=None,
                **base,
            )
        leverage = self._first(
            candidate,
            ("requested_entry_leverage", "configured_entry_leverage"),
            1.0,
        )
        if leverage <= 0.0 or leverage > self.policy.max_leverage + 1e-12:
            return RiskDecision(
                "reject",
                "requested_leverage_above_cap",
                approved_notional=0.0,
                l2=None,
                **base,
            )
        signal_ts = _utc(candidate.get("timestamp", snapshot.timestamp))
        age = (snapshot.timestamp - signal_ts).total_seconds() / 60.0
        if age < -0.1 or age > self.policy.max_signal_age_minutes:
            return RiskDecision(
                "reject",
                "stale_or_future_signal",
                approved_notional=0.0,
                l2=None,
                **base,
            )
        key = f"{signal_ts.isoformat()}|{symbol}|{candidate.get('side', '')}"
        if key in state.processed_signal_keys:
            return RiskDecision(
                "reject", "duplicate_signal", approved_notional=0.0, l2=None, **base
            )
        rank = self._first(
            candidate,
            (
                "threshold_basis_corrected_expected_ev_rank",
                "final_gate_rank_score",
                "portfolio_gate_rank_score",
            ),
        )
        ev = self._first(
            candidate,
            (
                "threshold_basis_corrected_expected_ev",
                "ev_adjusted_net_return_after_friction",
                "estimated_ev_net_return",
            ),
        )
        if not math.isfinite(rank) or rank < min_rank:
            return RiskDecision(
                "reject",
                "rank_below_prop_threshold",
                approved_notional=0.0,
                l2=None,
                **base,
            )
        if not math.isfinite(ev) or ev < self.policy.base_min_net_ev:
            return RiskDecision(
                "reject",
                "net_ev_below_prop_threshold",
                approved_notional=0.0,
                l2=None,
                **base,
            )
        if (
            self.policy.require_raw_bayesian_sizing
            and not sizing["raw_bayesian_multiplier_available"]
        ):
            return RiskDecision(
                "reject",
                "missing_raw_bayesian_size_multiplier",
                approved_notional=0.0,
                l2=None,
                **base,
            )
        policy_archetype = str(
            candidate.get("policy_archetype")
            or candidate.get("archetype_policy_key")
            or ""
        ).strip()
        if self.policy.require_policy_archetype_for_risk and not policy_archetype:
            return RiskDecision(
                "reject",
                "missing_policy_archetype_for_risk",
                approved_notional=0.0,
                l2=None,
                **base,
            )
        signal_price = self._first(
            candidate, ("expected_entry_price", "signal_price", "policy_entry_price")
        )
        stop_price = self._first(candidate, ("policy_stop_price", "stop_price"))
        side = str(candidate.get("side", "")).lower()
        stop_distance = self._stop_distance(side, signal_price, stop_price)
        if self.policy.require_stop and not math.isfinite(stop_distance):
            return RiskDecision(
                "reject",
                "missing_or_nonprotective_stop",
                approved_notional=0.0,
                l2=None,
                **base,
            )
        stressed_stop_distance = self._stressed_stop_distance(stop_distance)
        operational_invested = max_alloc * snapshot.equity
        hard_invested = self.policy.max_marked_notional_fraction * snapshot.equity
        gross_notional = (
            self.policy.max_gross_marked_notional_fraction * snapshot.equity
        )
        capacity_room = min(
            leverage * max(operational_invested - snapshot.marked_invested_amount, 0.0),
            leverage * max(hard_invested - snapshot.marked_invested_amount, 0.0),
            max(gross_notional - snapshot.marked_notional, 0.0),
        )
        loss_budget = self.loss_capacity_budget(snapshot, state, daily, peak)
        archetype_risk = self._prospective_archetype_risk_room(
            snapshot,
            policy_archetype=policy_archetype,
            reserved_budget=float(loss_budget["reserved_risk_budget"]),
        )
        position_stop_room = (
            self.policy.max_position_stop_risk_fraction * snapshot.equity
        )
        stop_room = min(
            float(archetype_risk["candidate_base_stop_risk_room"]),
            position_stop_room,
        )
        risk_share = 1.0
        if self.policy.convex_loss_budget_enabled:
            risk_share = self._first(
                candidate,
                ("portfolio_risk_budget_share",),
                self.policy.max_single_opportunity_risk_share,
            )
            risk_share = min(
                max(risk_share, 0.0), self.policy.max_single_opportunity_risk_share
            )
        prospective_multiplier = float(
            archetype_risk["prospective_archetype_multiplier"]
        )
        existing_uplift = float(archetype_risk["existing_group_uplift_reserved"])
        if self.policy.convex_loss_budget_enabled:
            batch_reserved_room = max(
                float(loss_budget["reserved_risk_budget"]) * risk_share
                - existing_uplift,
                0.0,
            )
            batch_base_room = batch_reserved_room / max(prospective_multiplier, 1e-12)
            stop_room = min(stop_room, batch_base_room)
        sizing["archetype_risk"] = archetype_risk
        sizing["loss_capacity_budget"] = loss_budget
        sizing["portfolio_risk_budget_share"] = risk_share
        sizing["raw_stop_distance_fraction"] = stop_distance
        sizing["stressed_stop_distance_fraction"] = stressed_stop_distance
        sizing["marked_invested_amount_before"] = snapshot.marked_invested_amount
        sizing["requested_leverage"] = leverage
        risk_size = (
            stop_room / stressed_stop_distance if stressed_stop_distance > 0 else 0.0
        )
        approved = min(requested, capacity_room, risk_size)
        l2_dict = asdict(l2) if l2 is not None else None
        if self.policy.require_l2 and l2 is None:
            return RiskDecision(
                "reject",
                "missing_l2_capacity_check",
                approved_notional=0.0,
                l2=None,
                **base,
            )
        if l2 is not None:
            if l2.reject_reason:
                return RiskDecision(
                    "reject",
                    f"l2:{l2.reject_reason}",
                    approved_notional=0.0,
                    l2=l2_dict,
                    **base,
                )
            if l2.capacity_weight < self.policy.min_l2_capacity_weight:
                return RiskDecision(
                    "reject",
                    "l2_capacity_weight_below_min",
                    approved_notional=0.0,
                    l2=l2_dict,
                    **base,
                )
            if (
                l2.expected_slippage_bps is not None
                and l2.expected_slippage_bps > self.policy.max_l2_slippage_bps
            ):
                return RiskDecision(
                    "reject",
                    "l2_slippage_above_cap",
                    approved_notional=0.0,
                    l2=l2_dict,
                    **base,
                )
            approved = min(approved, max(l2.capacity_quote, 0.0))
        if approved < self.policy.min_entry_quote:
            return RiskDecision(
                "reject",
                "insufficient_notional_or_stop_risk_capacity",
                approved_notional=0.0,
                l2=l2_dict,
                **base,
            )
        state.processed_signal_keys.append(key)
        reason = (
            "approved"
            if approved >= requested - 1e-9
            else "approved_reduced_by_risk_or_l2"
        )
        return RiskDecision(
            "enter", reason, approved_notional=approved, l2=l2_dict, **base
        )

    def requested_notional(
        self,
        candidate: Mapping[str, Any],
        snapshot: AccountSnapshot,
        effective_min_rank: Optional[float] = None,
    ) -> float:
        """Rescale an upstream intent to this account without leverage."""
        return float(
            self.sizing_breakdown(
                candidate,
                snapshot,
                effective_min_rank,
            )["requested_notional"]
        )

    def sizing_breakdown(
        self,
        candidate: Mapping[str, Any],
        snapshot: AccountSnapshot,
        effective_min_rank: Optional[float] = None,
    ) -> dict[str, Any]:
        """Return the bounded rank/Bayesian sizing calculation and audit.

        Rank is admission context, not an unlimited confidence proxy. Its
        contribution is convex from the effective admission threshold through
        0.99 and then flat. The frozen raw-Bayesian multiplier supplies the
        combined calibrated-EV, posterior-uncertainty and OOD adjustment. Stop
        risk, portfolio capacity and L2 remain downstream hard caps.
        """
        rank = self._first(
            candidate,
            (
                "threshold_basis_corrected_expected_ev_rank",
                "final_gate_rank_score",
                "portfolio_gate_rank_score",
            ),
        )
        threshold = effective_min_rank or self.policy.base_min_rank
        calibrated_ev = self._first(
            candidate,
            (
                "threshold_basis_corrected_expected_ev",
                "ev_adjusted_net_return_after_friction",
                "estimated_ev_net_return",
            ),
        )
        bayesian_raw = self._first(
            candidate,
            ("raw_bayesian_size_multiplier",),
        )
        bayesian_available = math.isfinite(bayesian_raw)
        bayesian = (
            min(
                max(bayesian_raw, self.policy.min_bayesian_size_multiplier),
                self.policy.max_bayesian_size_multiplier,
            )
            if bayesian_available
            else 1.0
        )
        if not math.isfinite(rank) or rank < threshold:
            return self._sizing_audit(
                rank=None if not math.isfinite(rank) else rank,
                capped_rank=None,
                threshold=threshold,
                strength=0.0,
                curved=0.0,
                multiplier=self.policy.rank_multiplier_min,
                slot_fraction=0.0,
                calibrated_ev=calibrated_ev,
                bayesian_available=bayesian_available,
                bayesian=bayesian,
                requested_fraction=0.0,
                requested=0.0,
            )
        capped_rank = min(rank, self.policy.rank_sizing_cap)
        excess = min(
            max(
                (capped_rank - threshold)
                / max(self.policy.rank_sizing_cap - threshold, 1e-12),
                0.0,
            ),
            1.0,
        )
        curved = excess ** max(self.policy.rank_size_power, 1.000001)
        multiplier = (
            self.policy.rank_multiplier_min
            + (self.policy.rank_multiplier_max - self.policy.rank_multiplier_min)
            * curved
        )
        slot_fraction = min(
            max(multiplier / max(self.policy.rank_multiplier_max, 1e-12), 0.0),
            1.0,
        )
        requested_fraction = min(
            self.policy.base_entry_notional_fraction * slot_fraction * bayesian,
            self.policy.max_position_notional_fraction,
        )
        requested = snapshot.equity * requested_fraction
        return self._sizing_audit(
            rank=rank,
            capped_rank=capped_rank,
            threshold=threshold,
            strength=excess,
            curved=curved,
            multiplier=multiplier,
            slot_fraction=slot_fraction,
            calibrated_ev=calibrated_ev,
            bayesian_available=bayesian_available,
            bayesian=bayesian,
            requested_fraction=requested_fraction,
            requested=requested,
        )

    def _sizing_audit(
        self,
        *,
        rank: Optional[float],
        capped_rank: Optional[float],
        threshold: float,
        strength: float,
        curved: float,
        multiplier: float,
        slot_fraction: float,
        calibrated_ev: float,
        bayesian_available: bool,
        bayesian: float,
        requested_fraction: float,
        requested: float,
    ) -> dict[str, Any]:
        return {
            "rank": rank,
            "rank_sizing_cap": self.policy.rank_sizing_cap,
            "capped_rank": capped_rank,
            "effective_min_rank": threshold,
            "rank_strength": strength,
            "curved_rank_strength": curved,
            "rank_multiplier": multiplier,
            "rank_slot_fraction": slot_fraction,
            "calibrated_net_ev": (
                calibrated_ev if math.isfinite(calibrated_ev) else None
            ),
            "raw_bayesian_multiplier_available": bayesian_available,
            "raw_bayesian_multiplier": bayesian,
            "bayesian_components": [
                "calibrated_ev",
                "posterior_uncertainty",
                "gmm_ood",
                "archetype_support",
            ],
            "requested_notional_fraction": requested_fraction,
            "requested_notional": requested,
        }

    def _archetype_risk_multiplier(self, position_count: int) -> float:
        if position_count <= 1:
            return 1.0
        return min(
            1.0
            + self.policy.archetype_risk_increment_per_position
            * float(position_count - 1),
            self.policy.max_archetype_risk_multiplier,
        )

    def reserved_stop_risk(self, positions: Sequence[MarkedPosition]) -> float:
        """Return stressed stop risk plus same-archetype concentration reserve."""
        grouped: dict[str, list[MarkedPosition]] = {}
        for position in positions:
            # Missing legacy provenance is conservatively treated as one shared
            # unknown archetype rather than assumed independent.
            archetype = (
                str(position.policy_archetype or "").strip().lower() or "__unknown__"
            )
            grouped.setdefault(archetype, []).append(position)
        reserved = 0.0
        for group in grouped.values():
            raw_risk = sum(self._position_stressed_stop_risk(p) for p in group)
            reserved += raw_risk * self._archetype_risk_multiplier(len(group))
        return float(reserved)

    def _prospective_archetype_risk_room(
        self,
        snapshot: AccountSnapshot,
        *,
        policy_archetype: str,
        reserved_budget: Optional[float] = None,
    ) -> dict[str, Any]:
        """Solve the candidate's base-risk room after group-risk uplift.

        Adding another position increases the multiplier on the whole existing
        archetype group. The uplift on existing positions is reserved before
        any risk is made available to the candidate.
        """
        archetype = str(policy_archetype or "").strip().lower()
        same = [
            position
            for position in snapshot.positions
            if str(position.policy_archetype or "").strip().lower() == archetype
            and archetype
        ]
        same_raw_risk = sum(self._position_stressed_stop_risk(p) for p in same)
        current_multiplier = self._archetype_risk_multiplier(len(same))
        prospective_multiplier = self._archetype_risk_multiplier(len(same) + 1)
        existing_group_uplift = max(
            (prospective_multiplier - current_multiplier) * same_raw_risk,
            0.0,
        )
        current_reserved = self.reserved_stop_risk(snapshot.positions)
        operational_budget = (
            float(reserved_budget)
            if reserved_budget is not None
            else self.policy.operational_stop_risk_fraction * snapshot.equity
        )
        operational_room = max(operational_budget - current_reserved, 0.0)
        hard_room = max(
            self.policy.max_stop_risk_fraction * snapshot.equity - current_reserved,
            0.0,
        )
        incremental_reserved_room = min(operational_room, hard_room)
        candidate_reserved_room = max(
            incremental_reserved_room - existing_group_uplift,
            0.0,
        )
        candidate_base_room = candidate_reserved_room / max(
            prospective_multiplier, 1e-12
        )
        return {
            "policy_archetype": policy_archetype,
            "same_archetype_positions_before": len(same),
            "same_archetype_raw_stop_risk_before": same_raw_risk,
            "current_archetype_multiplier": current_multiplier,
            "prospective_archetype_multiplier": prospective_multiplier,
            "existing_group_uplift_reserved": existing_group_uplift,
            "raw_stop_risk_before": snapshot.stop_risk,
            "reserved_stop_risk_before": current_reserved,
            "operational_reserved_stop_risk_room": operational_room,
            "operational_reserved_stop_risk_budget": operational_budget,
            "hard_reserved_stop_risk_room": hard_room,
            "candidate_base_stop_risk_room": candidate_base_room,
        }

    def loss_capacity_budget(
        self,
        snapshot: AccountSnapshot,
        state: ControllerState,
        daily_return: Optional[float] = None,
        peak_drawdown: Optional[float] = None,
    ) -> dict[str, float]:
        """Return the concurrent stressed-loss budget at the current mark.

        The budget is 75% (configurable) of convex remaining headroom to each
        applicable loss boundary. The minimum boundary wins. Existing open
        positions are charged separately by ``reserved_stop_risk``.
        """
        daily = (
            float(daily_return)
            if daily_return is not None
            else snapshot.equity / max(state.day_start_equity, 1e-12) - 1.0
        )
        drawdown = (
            float(peak_drawdown)
            if peak_drawdown is not None
            else self._contract_drawdown(snapshot, state)
        )
        if not self.policy.convex_loss_budget_enabled:
            budget = min(
                self.policy.operational_stop_risk_fraction * snapshot.equity,
                self.policy.max_stop_risk_fraction * snapshot.equity,
            )
            return {
                "enabled": 0.0,
                "daily_return": daily,
                "peak_drawdown": drawdown,
                "daily_budget": budget,
                "drawdown_budget": budget,
                "internal_budget": budget,
                "reserved_risk_budget": budget,
                "reserved_risk_before": self.reserved_stop_risk(snapshot.positions),
            }

        fraction = self.policy.convex_loss_budget_fraction
        power = self.policy.convex_loss_budget_power

        def boundary_budget(limit: float, loss: float) -> float:
            if limit <= 0.0:
                return 0.0
            headroom_ratio = min(max((limit - loss) / limit, 0.0), 1.0)
            return fraction * limit * headroom_ratio**power

        day_limit = self.policy.firm_max_daily_loss_fraction * max(
            state.day_start_equity, 0.0
        )
        drawdown_reference = self._drawdown_reference_equity(state)
        drawdown_limit = self.policy.firm_max_drawdown_fraction * drawdown_reference
        internal_reference = min(
            max(state.day_start_equity, 0.0),
            drawdown_reference,
        )
        internal_limit = (
            self.policy.internal_concurrent_loss_limit_fraction * internal_reference
        )
        daily_loss = max(-daily * state.day_start_equity, 0.0)
        drawdown_loss = max(-drawdown * drawdown_reference, 0.0)
        internal_loss = max(daily_loss, drawdown_loss)
        daily_budget = boundary_budget(day_limit, daily_loss)
        drawdown_budget = boundary_budget(drawdown_limit, drawdown_loss)
        internal_budget = boundary_budget(internal_limit, internal_loss)
        hard_budget = self.policy.max_stop_risk_fraction * snapshot.equity
        budget = min(daily_budget, drawdown_budget, internal_budget, hard_budget)
        return {
            "enabled": 1.0,
            "daily_return": daily,
            "peak_drawdown": drawdown,
            "drawdown_reference_equity": drawdown_reference,
            "daily_loss": daily_loss,
            "drawdown_loss": drawdown_loss,
            "internal_loss": internal_loss,
            "daily_headroom": max(day_limit - daily_loss, 0.0),
            "drawdown_headroom": max(drawdown_limit - drawdown_loss, 0.0),
            "internal_headroom": max(internal_limit - internal_loss, 0.0),
            "daily_budget": daily_budget,
            "drawdown_budget": drawdown_budget,
            "internal_budget": internal_budget,
            "hard_budget": hard_budget,
            "reserved_risk_budget": max(budget, 0.0),
            "reserved_risk_before": self.reserved_stop_risk(snapshot.positions),
        }

    def _drawdown_reference_equity(self, state: ControllerState) -> float:
        if self.policy.drawdown_reference_mode == "starting_balance":
            return max(state.starting_equity, 0.0)
        return max(state.high_water_equity, 0.0)

    def _contract_drawdown(
        self, snapshot: AccountSnapshot, state: ControllerState
    ) -> float:
        reference = self._drawdown_reference_equity(state)
        return snapshot.equity / max(reference, 1e-12) - 1.0

    def opportunity_risk_shares(
        self, candidates: Sequence[Mapping[str, Any]]
    ) -> list[float]:
        """Allocate batch risk across side/archetype-diverse opportunities.

        Missing diversity leaves budget unused: a lone trade cannot consume
        more than the single-opportunity cap merely because no alternative is
        present. This prevents arrival order from monopolising loss capacity.
        """
        if not candidates:
            return []
        weights: list[float] = []
        sides: list[str] = []
        archetypes: list[str] = []
        for row in candidates:
            rank = self._first(
                row,
                (
                    "threshold_basis_corrected_expected_ev_rank",
                    "final_gate_rank_score",
                    "portfolio_gate_rank_score",
                ),
                self.policy.base_min_rank,
            )
            strength = min(
                max(
                    (min(rank, self.policy.rank_sizing_cap) - self.policy.base_min_rank)
                    / max(
                        self.policy.rank_sizing_cap - self.policy.base_min_rank, 1e-12
                    ),
                    0.0,
                ),
                1.0,
            )
            ev = max(
                self._first(
                    row,
                    (
                        "threshold_basis_corrected_expected_ev",
                        "ev_adjusted_net_return_after_friction",
                        "estimated_ev_net_return",
                    ),
                    self.policy.base_min_net_ev,
                ),
                0.0,
            )
            bayesian = min(
                max(
                    self._first(row, ("raw_bayesian_size_multiplier",), 1.0),
                    self.policy.min_bayesian_size_multiplier,
                ),
                self.policy.max_bayesian_size_multiplier,
            )
            ev_quality = math.sqrt(
                min(ev / max(self.policy.base_min_net_ev, 1e-12), 3.0)
            )
            weights.append(
                max((0.25 + 0.75 * strength**2) * bayesian * ev_quality, 1e-9)
            )
            sides.append(
                str(row.get("side") or row.get("side_name") or "unknown").lower()
            )
            archetypes.append(
                str(
                    row.get("policy_archetype")
                    or row.get("archetype_policy_key")
                    or "unknown"
                ).lower()
            )

        shares = [0.0] * len(candidates)
        side_groups: dict[str, list[int]] = {}
        for idx, side in enumerate(sides):
            side_groups.setdefault(side, []).append(idx)
        side_weights = [sum(weights[i] for i in idxs) for idxs in side_groups.values()]
        side_alloc = self._capped_weighted_allocation(
            side_weights, 1.0, self.policy.max_side_batch_risk_share
        )
        for side_budget, idxs in zip(side_alloc, side_groups.values()):
            archetype_groups: dict[str, list[int]] = {}
            for idx in idxs:
                archetype_groups.setdefault(archetypes[idx], []).append(idx)
            archetype_weights = [
                sum(weights[i] for i in group) for group in archetype_groups.values()
            ]
            archetype_alloc = self._capped_weighted_allocation(
                archetype_weights,
                side_budget,
                self.policy.max_archetype_batch_risk_share,
            )
            for archetype_budget, group in zip(
                archetype_alloc, archetype_groups.values()
            ):
                trade_weights = [weights[i] for i in group]
                trade_alloc = self._capped_weighted_allocation(
                    trade_weights,
                    archetype_budget,
                    self.policy.max_single_opportunity_risk_share,
                )
                for idx, allocation in zip(group, trade_alloc):
                    shares[idx] = allocation
        return shares

    @staticmethod
    def _capped_weighted_allocation(
        weights: Sequence[float], total: float, cap: float
    ) -> list[float]:
        """Deterministic capped water-filling; unavailable diversity stays cash."""
        out = [0.0] * len(weights)
        active = {i for i, weight in enumerate(weights) if weight > 0.0}
        remaining = max(total, 0.0)
        while active and remaining > 1e-12:
            weight_sum = sum(weights[i] for i in active)
            if weight_sum <= 0.0:
                break
            proposed = {i: remaining * weights[i] / weight_sum for i in active}
            capped = [i for i in active if out[i] + proposed[i] > cap + 1e-12]
            if not capped:
                for i, value in proposed.items():
                    out[i] += value
                remaining = 0.0
                break
            for i in capped:
                addition = max(cap - out[i], 0.0)
                out[i] += addition
                remaining -= addition
                active.remove(i)
        return out

    def _stressed_stop_distance(self, raw_distance: float) -> float:
        if not math.isfinite(raw_distance) or raw_distance <= 0.0:
            return math.nan
        return (
            raw_distance * self.policy.stop_loss_risk_margin_multiplier
            + self.policy.stop_loss_risk_margin_bps / 10_000.0
        )

    def _position_stressed_stop_risk(self, position: MarkedPosition) -> float:
        distance = self._stressed_stop_distance(position.stop_distance_fraction)
        return abs(position.marked_notional) * distance

    def _dynamic_limits(self, daily: float, peak: float) -> tuple[float, float]:
        loss = min(daily, peak)
        rank, alloc = (
            self.policy.base_min_rank,
            self.policy.max_total_wallet_allocation_fraction,
        )
        for trigger, tier_rank, tier_alloc in self.policy.drawdown_tiers:
            if loss <= trigger:
                rank, alloc = max(rank, tier_rank), min(alloc, tier_alloc)
        return rank, alloc

    def _decision(
        self,
        action: str,
        reason: str,
        snapshot: AccountSnapshot,
        daily: float,
        peak: float,
    ) -> RiskDecision:
        rank, _ = self._dynamic_limits(daily, peak)
        return RiskDecision(
            action,
            reason,
            None,
            0.0,
            0.0,
            rank,
            daily,
            peak,
            snapshot.marked_notional,
            snapshot.stop_risk,
            self.reserved_stop_risk(snapshot.positions),
        )

    @staticmethod
    def _first(
        row: Mapping[str, Any], names: Sequence[str], default: float = math.nan
    ) -> float:
        for name in names:
            value = _finite(row.get(name))
            if math.isfinite(value):
                return value
        return default

    @staticmethod
    def _stop_distance(side: str, price: float, stop: float) -> float:
        if not math.isfinite(price) or not math.isfinite(stop) or price <= 0:
            return math.nan
        distance = (price - stop) / price if side == "long" else (stop - price) / price
        return distance if distance > 0 else math.nan


def load_whitelist(path: Path) -> set[str]:
    if not path.exists():
        return set()
    raw = json.loads(path.read_text())
    values = raw.get("symbols", []) if isinstance(raw, dict) else raw
    return {canonical_symbol(v) for v in values}
