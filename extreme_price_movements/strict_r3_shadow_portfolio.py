"""Causal, exchange-free strict-R3 portfolio auction for one live snapshot.

This module deliberately does not know how to fetch balances, positions, or
orders.  It consumes an explicit point-in-time account snapshot and applies
the frozen long-only capacity contract to candidates that already passed the
canonical EV-admission map.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd


SCHEMA = "strict_r3_shadow_portfolio_snapshot_v1"
LEGACY_POLICY_SCHEMA = "strict_r3_cell_day_trim15_portfolio_v1"
POLICY_SCHEMA = "strict_r3_cell_day_trim15_portfolio_28d_trust_v1"
POSTERIOR_POLICY_SCHEMA = "strict_r3_cell_day_trim15_portfolio_28d_r5_9m_posterior_v1"
A5_POLICY_SCHEMA = "strict_r3_cell_day_trim15_portfolio_28d_a5_b10_v1"


@dataclass(frozen=True)
class ShadowPortfolioPolicy:
    max_concurrent_positions: int = 8
    max_concurrent_per_symbol: int = 1
    max_new_entries_per_bar: int = 2
    max_total_margin_fraction: float = 0.80
    margin_slot_fraction: float = 0.10
    leverage: float = 7.0
    minimum_gross_notional: float = 1.0
    admission_mode: str = "cell_day_demotion_only"
    policy_schema: str = POLICY_SCHEMA

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "ShadowPortfolioPolicy":
        schema = str(payload.get("schema") or "")
        if schema not in {
            LEGACY_POLICY_SCHEMA, POLICY_SCHEMA, POSTERIOR_POLICY_SCHEMA,
            A5_POLICY_SCHEMA,
        }:
            raise ValueError(
                "portfolio policy has an unknown strict-R3 schema"
            )
        posterior = schema == POSTERIOR_POLICY_SCHEMA
        a5 = schema == A5_POLICY_SCHEMA
        if posterior:
            if payload.get("missing_posterior") != "fail_closed":
                raise ValueError("posterior admission must fail closed")
            if payload.get("admission_expected_net_field") != "trust_posterior_expected_bps":
                raise ValueError("posterior policy has the wrong admission value")
            if not np.isclose(float(payload.get("admission_threshold_bps", np.nan)), 50.0):
                raise ValueError("posterior policy requires a +50-bps admission floor")
        if a5:
            if payload.get("missing_component") != "fail_closed":
                raise ValueError("bounded-A5 admission must fail closed")
            if payload.get("admission_expected_net_field") != "a5_bounded10_expected_bps":
                raise ValueError("bounded-A5 policy has the wrong auction value")
            if payload.get("admission_boolean_field") != "a5_bounded10_admitted":
                raise ValueError("bounded-A5 policy has the wrong admission field")
            if payload.get("anchor_expected_net_field") != "trust_posterior_expected_bps":
                raise ValueError("bounded-A5 policy has the wrong A0 anchor")
            if not np.isclose(float(payload.get("anchor_threshold_bps", np.nan)), 50.0):
                raise ValueError("bounded-A5 policy requires an A0 +50-bps floor")
            if payload.get("domain") != "timestamp_local_top15_by_pretrust_final_score":
                raise ValueError("bounded-A5 policy requires the frozen top-15 domain")
        policy = cls(
            max_concurrent_positions=int(payload["max_concurrent_positions"]),
            max_concurrent_per_symbol=int(payload["max_concurrent_per_symbol"]),
            max_new_entries_per_bar=int(payload["max_new_entries_per_bar"]),
            max_total_margin_fraction=float(payload["max_total_margin_fraction"]),
            margin_slot_fraction=float(payload["margin_slot_fraction"]),
            leverage=float(payload["leverage"]),
            minimum_gross_notional=float(payload.get("minimum_gross_notional", 1.0)),
            admission_mode=(
                "a5_bounded10" if a5 else (
                    "r5_9m_posterior" if posterior else "cell_day_demotion_only"
                )
            ),
            policy_schema=schema,
        )
        if policy.max_concurrent_positions != 8:
            raise ValueError("canonical strict-R3 portfolio requires eight slots")
        if policy.max_concurrent_per_symbol != 1:
            raise ValueError("canonical strict-R3 portfolio requires one position per asset")
        if policy.max_new_entries_per_bar != 2:
            raise ValueError("canonical strict-R3 portfolio requires two new entries per bar")
        if not np.isclose(policy.max_total_margin_fraction, 0.80):
            raise ValueError("canonical strict-R3 portfolio requires an 80% margin cap")
        if not np.isclose(policy.margin_slot_fraction, 0.10):
            raise ValueError("canonical strict-R3 portfolio requires 10% margin slots")
        if not np.isclose(policy.leverage, 7.0):
            raise ValueError("canonical strict-R3 portfolio requires 7x leverage")
        if policy.minimum_gross_notional <= 0.0:
            raise ValueError("minimum gross notional must be positive")
        return policy


@dataclass(frozen=True)
class ShadowOpenPosition:
    symbol: str
    side: str
    gross_notional: float
    effective_leverage: float

    @property
    def committed_margin(self) -> float:
        return float(self.gross_notional / self.effective_leverage)


@dataclass(frozen=True)
class ShadowPortfolioState:
    as_of_ts: pd.Timestamp
    wallet: float
    open_positions: tuple[ShadowOpenPosition, ...]

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any],
        *,
        expected_as_of_ts: pd.Timestamp,
    ) -> "ShadowPortfolioState":
        if str(payload.get("schema") or "") != SCHEMA:
            raise ValueError(f"portfolio state must use {SCHEMA}")
        as_of = pd.Timestamp(payload["as_of_ts"])
        as_of = as_of.tz_localize("UTC") if as_of.tzinfo is None else as_of.tz_convert("UTC")
        expected = pd.Timestamp(expected_as_of_ts)
        expected = expected.tz_localize("UTC") if expected.tzinfo is None else expected.tz_convert("UTC")
        if as_of != expected:
            raise ValueError("portfolio state must be captured at the exact decision timestamp")
        wallet = float(payload["wallet"])
        if not np.isfinite(wallet) or wallet <= 0.0:
            raise ValueError("portfolio wallet must be finite and positive")
        positions: list[ShadowOpenPosition] = []
        for raw in payload.get("open_positions", []):
            position = ShadowOpenPosition(
                symbol=str(raw["symbol"]),
                side=str(raw.get("side", "long")).lower(),
                gross_notional=float(raw["gross_notional"]),
                effective_leverage=float(raw["effective_leverage"]),
            )
            if position.side != "long":
                raise ValueError("canonical strict-R3 portfolio is long-only")
            if (
                not position.symbol
                or not np.isfinite(position.gross_notional)
                or position.gross_notional <= 0.0
                or not np.isfinite(position.effective_leverage)
                or position.effective_leverage <= 0.0
            ):
                raise ValueError("open positions require finite positive notional and leverage")
            positions.append(position)
        symbols = [position.symbol for position in positions]
        if len(symbols) != len(set(symbols)):
            raise ValueError("portfolio state contains duplicate open assets")
        return cls(as_of_ts=as_of, wallet=wallet, open_positions=tuple(positions))


def auction_admitted_snapshot(
    candidates: pd.DataFrame,
    *,
    state: ShadowPortfolioState,
    policy: ShadowPortfolioPolicy,
) -> pd.DataFrame:
    """Apply the frozen auction without outcomes, future paths, or exchange IO."""
    required = {
        "candidate_id", "__decision_ts__", "__symbol__", "side_name", "final_score",
        "causal_21d_side_expected_net_bps", "causal_21d_side_admitted_ge_50bps",
    }
    missing = sorted(required.difference(candidates.columns))
    if missing:
        raise ValueError(f"shadow auction candidates lack {missing}")
    if candidates["candidate_id"].duplicated().any():
        raise ValueError("shadow auction requires unique candidate identities")
    work = candidates.copy()
    timestamps = pd.to_datetime(work["__decision_ts__"], utc=True, errors="raise")
    if not timestamps.eq(state.as_of_ts).all():
        raise ValueError("shadow candidates and portfolio state are not contemporaneous")
    if not work["side_name"].astype(str).str.lower().eq("long").all():
        raise ValueError("canonical strict-R3 auction is long-only")
    cell_day = pd.to_numeric(work["causal_21d_side_expected_net_bps"], errors="coerce")
    if policy.admission_mode == "a5_bounded10":
        a0 = pd.to_numeric(
            work.get(
                "trust_posterior_expected_bps",
                pd.Series(np.nan, index=work.index),
            ), errors="coerce",
        )
        bounded = pd.to_numeric(
            work.get(
                "a5_bounded10_expected_bps",
                pd.Series(np.nan, index=work.index),
            ), errors="coerce",
        )
        declared = work.get(
            "a5_bounded10_admitted", pd.Series(False, index=work.index),
        ).fillna(False).astype(bool)
        top15 = work.get(
            "a5_timestamp_top15", pd.Series(False, index=work.index),
        ).fillna(False).astype(bool)
        available = np.isfinite(a0) & np.isfinite(bounded)
        admitted = declared & available & a0.ge(50.0) & top15
        if (declared & ~admitted).any():
            raise ValueError("bounded-A5 admission violates its A0/top-15 contract")
        work["__mapped__"] = bounded
    elif policy.admission_mode == "r5_9m_posterior":
        posterior = pd.to_numeric(
            work.get(
                "trust_posterior_expected_bps",
                pd.Series(np.nan, index=work.index),
            ),
            errors="coerce",
        )
        admitted = np.isfinite(posterior) & posterior.ge(50.0)
        work["trust_posterior_admitted_ge_50bps"] = admitted
        work["__mapped__"] = posterior
    else:
        admitted = work["causal_21d_side_admitted_ge_50bps"].fillna(False).astype(bool)
        if (admitted & (~np.isfinite(cell_day) | cell_day.lt(50.0))).any():
            raise ValueError("EV-admitted candidate violates the canonical +50-bps floor")
        corrected = pd.to_numeric(
            work.get("trust_corrected_expected_net_bps", cell_day), errors="coerce",
        )
        if (~np.isfinite(corrected)).any() or corrected.gt(cell_day + 1e-8).any():
            raise ValueError("R5 trust score must be finite and demotion-only")
        work["__mapped__"] = corrected
    work["__score__"] = pd.to_numeric(work["final_score"], errors="coerce")
    work["portfolio_accepted"] = False
    work["portfolio_rejection_reason"] = "ev_map_rejected"
    work["portfolio_priority_rank"] = np.nan
    work["portfolio_initial_margin"] = 0.0
    work["portfolio_gross_notional"] = 0.0

    initial_committed_margin = float(
        sum(position.committed_margin for position in state.open_positions)
    )
    margin_cap = float(policy.max_total_margin_fraction * state.wallet)
    work["portfolio_wallet"] = state.wallet
    work["portfolio_open_positions_before"] = len(state.open_positions)
    work["portfolio_committed_margin_before"] = initial_committed_margin
    work["portfolio_margin_cap"] = margin_cap
    work["portfolio_policy_schema"] = policy.policy_schema
    work["portfolio_state_schema"] = SCHEMA

    eligible = work.loc[admitted].sort_values(
        ["__mapped__", "__score__", "candidate_id"],
        ascending=[False, False, True],
        kind="stable",
    )
    if eligible.empty:
        return work.drop(columns=["__mapped__", "__score__"])
    work.loc[eligible.index, "portfolio_priority_rank"] = np.arange(1, len(eligible) + 1)
    open_symbols = {position.symbol for position in state.open_positions}
    open_count = len(state.open_positions)
    committed_margin = initial_committed_margin
    new_entries = 0
    for idx, row in eligible.iterrows():
        symbol = str(row["__symbol__"])
        reason = "accepted"
        if symbol in open_symbols:
            reason = "symbol_already_open"
        elif open_count >= policy.max_concurrent_positions:
            reason = "max_concurrent_positions_reached"
        elif new_entries >= policy.max_new_entries_per_bar:
            reason = "max_new_entries_per_bar_reached"
        else:
            remaining_margin = max(margin_cap - committed_margin, 0.0)
            initial_margin = min(policy.margin_slot_fraction * state.wallet, remaining_margin)
            gross_notional = initial_margin * policy.leverage
            if gross_notional < policy.minimum_gross_notional:
                reason = "max_capital_allocation_reached"
            else:
                work.at[idx, "portfolio_accepted"] = True
                work.at[idx, "portfolio_initial_margin"] = initial_margin
                work.at[idx, "portfolio_gross_notional"] = gross_notional
                open_symbols.add(symbol)
                open_count += 1
                new_entries += 1
                committed_margin += initial_margin
        work.at[idx, "portfolio_rejection_reason"] = reason
    return work.drop(columns=["__mapped__", "__score__"])


__all__ = [
    "A5_POLICY_SCHEMA", "LEGACY_POLICY_SCHEMA", "POLICY_SCHEMA",
    "POSTERIOR_POLICY_SCHEMA", "SCHEMA",
    "ShadowPortfolioPolicy", "ShadowPortfolioState",
    "auction_admitted_snapshot",
]
