"""Market and strategy kill-switch primitives for live inference."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SafetySwitchDecision:
    allow_new_entries: bool
    active: bool
    reason: str
    details: Dict[str, Any]


class MarketKillSwitch:
    """Self-reversible market shock/depeg switch that blocks only new entries."""

    _DEPRECATED_REASONS = {
        "BTC_1H_MOVE_GT_7PCT",
        "ETH_1H_MOVE_GT_7PCT",
        "BTC_4H_MOVE_GT_10PCT",
        "ETH_4H_MOVE_GT_10PCT",
        "MARKET_MEDIAN_1H_MOVE_GT_5PCT",
    }

    def __init__(
        self,
        path: str | Path = "data/live_state/market_kill_switch.json",
        *,
        halt_hours: float = 12.0,
        min_market_basket_assets: int = 20,
    ):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.halt_hours = float(halt_hours)
        self.min_market_basket_assets = int(max(min_market_basket_assets, 1))

    def _load_state(self) -> Dict[str, Any]:
        if not self.path.exists():
            return {}
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            return payload if isinstance(payload, dict) else {}
        except Exception:
            return {}

    def _write_state(self, state: Dict[str, Any]) -> None:
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(self.path)

    @staticmethod
    def _ret(series: pd.Series, periods: int) -> float:
        s = pd.Series(series).dropna()
        if len(s) <= periods:
            return 0.0
        return float(s.iloc[-1] / max(float(s.iloc[-1 - periods]), 1e-12) - 1.0)

    @staticmethod
    def _market_basket_close(basket_close: pd.DataFrame) -> pd.DataFrame:
        """Use broad non-BTC/ETH context for market-wide shock detection."""
        if not isinstance(basket_close, pd.DataFrame) or basket_close.empty:
            return pd.DataFrame()
        keep_cols = []
        for col in basket_close.columns:
            text = str(col).upper()
            base = text.split("/", 1)[0].split("_", 1)[0]
            if base in {"BTC", "ETH"}:
                continue
            keep_cols.append(col)
        if not keep_cols:
            return pd.DataFrame(index=basket_close.index)
        return basket_close.loc[:, keep_cols]

    @staticmethod
    def _basket_abs_move(basket_close: pd.DataFrame, periods: int) -> float:
        if not isinstance(basket_close, pd.DataFrame) or len(basket_close) <= periods:
            return 0.0
        latest = basket_close.iloc[-1].astype(float)
        prior = basket_close.iloc[-1 - periods].astype(float).replace(0.0, np.nan)
        move = (latest / prior - 1.0).abs().replace([np.inf, -np.inf], np.nan)
        mean_move = float(move.mean(skipna=True))
        return mean_move if np.isfinite(mean_move) else 0.0

    def evaluate(
        self,
        *,
        now: pd.Timestamp,
        usdc_usdt_ticker: Optional[Dict[str, Any]],
        btc_close: pd.Series,
        eth_close: pd.Series,
        basket_close: pd.DataFrame,
    ) -> SafetySwitchDecision:
        now_ts = pd.Timestamp(now)
        if now_ts.tzinfo is None:
            now_ts = now_ts.tz_localize("UTC")
        else:
            now_ts = now_ts.tz_convert("UTC")

        details: Dict[str, Any] = {}
        reason = ""
        if isinstance(usdc_usdt_ticker, dict):
            bid = float(usdc_usdt_ticker.get("bid", np.nan))
            ask = float(usdc_usdt_ticker.get("ask", np.nan))
            last = float(usdc_usdt_ticker.get("last", np.nan))
            if np.isfinite(bid) and np.isfinite(ask):
                mid = (bid + ask) / 2.0
            else:
                mid = last
            if np.isfinite(mid):
                details["usdc_usdt_mid"] = float(mid)
                if mid < 0.98 or mid > 1.02:
                    reason = "USDC_USDT_DEPEG"

        if (
            not reason
            and isinstance(basket_close, pd.DataFrame)
            and len(basket_close) > 1
        ):
            market_basket_close = self._market_basket_close(basket_close)
            market_basket_assets = int(market_basket_close.shape[1])
            details["market_basket_assets"] = market_basket_assets
            details["market_basket_min_assets"] = int(self.min_market_basket_assets)
            if market_basket_assets < self.min_market_basket_assets:
                details["market_basket_status"] = "insufficient_breadth"
                avg_1h = 0.0
                avg_4h = 0.0
            else:
                details["market_basket_status"] = "broad_non_btc_eth"
                avg_1h = self._basket_abs_move(market_basket_close, 1)
                avg_4h = self._basket_abs_move(market_basket_close, 4)
            details["market_avg_1h_abs_move"] = avg_1h
            details["market_avg_4h_abs_move"] = avg_4h
            if avg_1h > 0.05:
                reason = "MARKET_AVG_1H_MOVE_GT_5PCT"
            elif avg_4h > 0.10:
                reason = "MARKET_AVG_4H_MOVE_GT_10PCT"

        state = self._load_state()
        if reason:
            halt_until = now_ts + pd.Timedelta(hours=self.halt_hours)
            state = {
                "active": True,
                "self_reversible": True,
                "manual_reset_required": False,
                "triggered_at": now_ts.isoformat(),
                "halt_until": halt_until.isoformat(),
                "reason": reason,
                "details": details,
            }
            self._write_state(state)
            return SafetySwitchDecision(False, True, reason, details)

        if state.get("active"):
            if state.get("reason") in self._DEPRECATED_REASONS:
                state["active"] = False
                state["recovered_at"] = now_ts.isoformat()
                state["recovery_reason"] = "deprecated_market_kill_switch_reason"
                self._write_state(state)
                return SafetySwitchDecision(True, False, "allowed", details)
            halt_until = pd.to_datetime(
                state.get("halt_until"), utc=True, errors="coerce"
            )
            if pd.notna(halt_until) and now_ts < halt_until:
                active_details = dict(details)
                active_details.update(
                    {
                        "halt_source": "stored_state",
                        "stored_halt_reason": str(
                            state.get("reason") or "market_kill_switch_active"
                        ),
                        "stored_halt_triggered_at": state.get("triggered_at"),
                        "stored_halt_until": halt_until.isoformat(),
                        "stored_halt_details": dict(state.get("details") or {}),
                    }
                )
                return SafetySwitchDecision(
                    False,
                    True,
                    str(state.get("reason") or "market_kill_switch_active"),
                    active_details,
                )
            state["active"] = False
            state["recovered_at"] = now_ts.isoformat()
            self._write_state(state)

        return SafetySwitchDecision(True, False, "allowed", details)


class StrategyKillSwitch:
    """Manual-reset per-strategy switch with observe-only support."""

    def __init__(
        self,
        path: str | Path = "data/live_state/strategy_kill_switches.json",
        *,
        observe_only: bool = True,
    ):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.observe_only = bool(observe_only)

    def load(self) -> Dict[str, Any]:
        if not self.path.exists():
            return {}
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            return payload if isinstance(payload, dict) else {}
        except Exception:
            return {}

    def set_state(
        self,
        strategy_id: str,
        *,
        active: bool,
        reason: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        state = self.load()
        state[str(strategy_id)] = {
            "active": bool(active),
            "manual_reset_required": bool(active),
            "reason": str(reason),
            "details": dict(details or {}),
        }
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(self.path)

    def is_blocked(self, strategy_id: str) -> SafetySwitchDecision:
        state = self.load().get(str(strategy_id), {})
        active = bool(isinstance(state, dict) and state.get("active"))
        if active and not self.observe_only:
            return SafetySwitchDecision(
                False,
                True,
                "strategy_kill_switch_active",
                dict(state),
            )
        return SafetySwitchDecision(
            True,
            active,
            "observe_only" if active else "allowed",
            dict(state) if isinstance(state, dict) else {},
        )
