#!/usr/bin/env python3
"""Matched July replay of static versus convex prop-account loss budgets.

The input frontier is the clean, non-zero-entry-volume Joint trailing/raw
Bayesian selected ledger. Outcomes, costs, timestamps, and causal entry ATR are
identical across arms; only the prop-account portfolio overlay changes.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from extreme_price_movements.inference.prop_account_controller import (
    AccountSnapshot,
    ControllerState,
    L2Capacity,
    MarkedPosition,
    PropAccountController,
    PropAccountPolicy,
)

ROOT = Path(
    "data_perp/reports/"
    "simple_policy_1m_joint_trailing_raw_bayesian_champion_20260718_v1"
)
OUT = Path("data_perp/reports/prop_account_convex_loss_budget_replay_20260718_v1")
INITIAL_EQUITY = 5_000.0
PAYOUT_FRACTION = 0.80
PURCHASE_COST = 45.0


def _inputs() -> tuple[pd.DataFrame, set[str]]:
    selected = pd.read_parquet(
        ROOT
        / "daily_replay_july01_17_nonzero_entry_volume_v1/selected_trade_ledger.parquet"
    )
    selected = selected.loc[
        (selected["policy"] == "joint_trailing_plus_bayesian_raw")
        & (selected["timestamp"] < pd.Timestamp("2026-07-17", tz="UTC"))
        & (selected["rank_pct"] >= 0.95)
    ].copy()
    atr = pd.concat(
        [
            pd.read_parquet(ROOT / "replay/causal_entry_atr_audit.parquet"),
            pd.read_parquet(
                ROOT / "forward_replay_jul11_17_v1/causal_entry_atr_audit.parquet"
            ),
        ],
        ignore_index=True,
    )
    atr["side_name"] = atr["side"].str.lower()
    atr = atr.drop_duplicates(["timestamp", "symbol", "side_name"], keep="last")
    selected = selected.merge(
        atr[["timestamp", "symbol", "side_name", "effective_atr_fraction"]],
        on=["timestamp", "symbol", "side_name"],
        how="left",
        validate="many_to_one",
    )
    if selected["effective_atr_fraction"].isna().any():
        raise RuntimeError("causal entry ATR coverage is incomplete")

    candidates = pd.read_parquet(
        ROOT
        / "daily_replay_july01_17_nonzero_entry_volume_v1/candidate_liquidity_audit.parquet"
    )
    candidates = candidates.loc[
        (candidates["entry_minute_volume"] > 0)
        & (candidates["timestamp"] < pd.Timestamp("2026-07-17", tz="UTC"))
        & (candidates["rank_pct"] >= 0.95)
    ]
    symbols = sorted(candidates["symbol"].unique())
    rng = np.random.default_rng(20260718)
    samples: list[tuple[int, float, set[str]]] = []
    for _ in range(3_000):
        whitelist = set(rng.choice(symbols, size=min(60, len(symbols)), replace=False))
        rows = selected.loc[selected["symbol"].isin(whitelist)]
        samples.append((len(rows), float(rows["net_return"].mean()), whitelist))
    median_count = float(np.median([row[0] for row in samples]))
    median_ev = float(np.nanmedian([row[1] for row in samples]))
    representative = min(
        samples,
        key=lambda row: abs(row[0] - median_count) + 50.0 * abs(row[1] - median_ev),
    )[2]
    return selected, representative


def _policy(arm: str) -> tuple[PropAccountPolicy, float]:
    common: dict[str, Any] = {
        "require_l2": True,
        "max_gross_marked_notional_fraction": 1.0,
    }
    if arm == "previous_static":
        return PropAccountPolicy(**common), 1.0
    convex = dict(
        common,
        schema_version="prop_account_overlay_v3_convex_loss_budget",
        convex_loss_budget_enabled=True,
        convex_loss_budget_fraction=0.75,
        convex_loss_budget_power=1.5,
        internal_concurrent_loss_limit_fraction=0.015,
        firm_max_daily_loss_fraction=0.03,
        firm_max_drawdown_fraction=0.03,
        drawdown_reference_mode="starting_balance",
        stop_loss_risk_margin_multiplier=1.25,
        max_stop_risk_fraction=0.0125,
        max_position_stop_risk_fraction=0.0025,
        base_entry_notional_fraction=0.11,
        max_position_notional_fraction=0.11,
        max_single_opportunity_risk_share=0.30,
        max_archetype_batch_risk_share=0.60,
        max_side_batch_risk_share=0.75,
        drawdown_tiers=(),
        stop_entries_peak_drawdown=-0.015,
        flatten_peak_drawdown=-0.02,
    )
    if arm == "convex_loss_budget_1x":
        return PropAccountPolicy(**convex), 1.0
    if arm == "convex_loss_budget_margin_2x":
        return PropAccountPolicy(**convex), 2.0
    raise ValueError(arm)


def _replay(
    source: pd.DataFrame,
    whitelist: set[str],
    arm: str,
    universe: str,
) -> tuple[dict[str, Any], pd.DataFrame]:
    policy, leverage = _policy(arm)
    controller = PropAccountController(policy, whitelist)
    rows = source.loc[source["symbol"].isin(whitelist)].copy()
    rows = rows.sort_values(["timestamp", "rank_pct"], ascending=[True, False])
    equity = INITIAL_EQUITY
    day_start_equity = INITIAL_EQUITY
    high_water = INITIAL_EQUITY
    current_day = rows["timestamp"].min().floor("D")
    snapshot = AccountSnapshot(
        rows["timestamp"].min(),
        equity,
        day_start_equity=day_start_equity,
        high_water_equity=high_water,
    )
    state = ControllerState.initialise(snapshot)
    open_rows: list[dict[str, Any]] = []
    trades: list[dict[str, Any]] = []
    rejects: Counter[str] = Counter()
    equity_points: list[tuple[pd.Timestamp, float]] = [(snapshot.timestamp, equity)]
    daily_pnl: Counter[pd.Timestamp] = Counter()
    peak_gross = peak_invested = peak_reserved = peak_raw = 0.0
    batch_count = diverse_batch_count = 0

    def positions() -> tuple[MarkedPosition, ...]:
        return tuple(row["position"] for row in open_rows)

    def close_through(timestamp: pd.Timestamp) -> None:
        nonlocal equity, high_water, open_rows
        closing = sorted(
            [row for row in open_rows if row["exit_timestamp"] <= timestamp],
            key=lambda row: row["exit_timestamp"],
        )
        open_rows = [row for row in open_rows if row["exit_timestamp"] > timestamp]
        for row in closing:
            pnl = row["notional"] * row["net_return"]
            equity += pnl
            high_water = max(high_water, equity)
            daily_pnl[row["exit_timestamp"].floor("D")] += pnl
            equity_points.append((row["exit_timestamp"], equity))
            row["pnl"] = pnl
            row["equity_after_exit"] = equity
            trades.append(row)

    for timestamp, batch in rows.groupby("timestamp", sort=True):
        close_through(timestamp)
        day = timestamp.floor("D")
        if day != current_day:
            current_day = day
            day_start_equity = equity
        candidate_rows: list[dict[str, Any]] = []
        source_rows: list[pd.Series] = []
        for _, row in batch.iterrows():
            stop_distance = 4.0 * float(row["effective_atr_fraction"])
            stop_price = (
                1.0 - stop_distance
                if row["side_name"] == "long"
                else 1.0 + stop_distance
            )
            candidate_rows.append(
                {
                    "timestamp": timestamp,
                    "symbol": row["symbol"],
                    "side": row["side_name"],
                    "threshold_basis_corrected_expected_ev_rank": row["rank_pct"],
                    # The input is already the fixed EV-admitted selected frontier.
                    "threshold_basis_corrected_expected_ev": policy.base_min_net_ev,
                    "raw_bayesian_size_multiplier": row["size_multiplier"],
                    "policy_archetype": row["policy_archetype"],
                    "signal_price": 1.0,
                    "policy_stop_price": stop_price,
                    "requested_entry_leverage": leverage,
                    "passed_rank_gate": True,
                }
            )
            source_rows.append(row)
        shares = (
            controller.opportunity_risk_shares(candidate_rows)
            if policy.convex_loss_budget_enabled
            else [1.0] * len(candidate_rows)
        )
        batch_count += 1
        if len({row["policy_archetype"] for row in candidate_rows}) > 1:
            diverse_batch_count += 1
        evaluation_order = sorted(
            range(len(candidate_rows)),
            key=lambda idx: (
                shares[idx],
                candidate_rows[idx]["threshold_basis_corrected_expected_ev_rank"],
            ),
            reverse=True,
        )
        for idx in evaluation_order:
            candidate = candidate_rows[idx]
            candidate["portfolio_risk_budget_share"] = shares[idx]
            row = source_rows[idx]
            snapshot = AccountSnapshot(
                timestamp,
                equity,
                positions(),
                day_start_equity,
                high_water,
            )
            decision = controller.evaluate_entry(
                candidate,
                snapshot,
                state,
                L2Capacity(1_000_000.0, 1.0, 0.0),
            )
            if decision.action != "enter":
                rejects[decision.reason] += 1
                continue
            position = MarkedPosition(
                candidate["symbol"],
                candidate["side"],
                decision.approved_notional,
                1.0,
                candidate["policy_stop_price"],
                candidate["policy_archetype"],
                leverage,
            )
            open_rows.append(
                {
                    "arm": arm,
                    "universe": universe,
                    "entry_timestamp": timestamp,
                    "exit_timestamp": timestamp
                    + pd.Timedelta(minutes=int(row["exit_bars"])),
                    "symbol": row["symbol"],
                    "side": row["side_name"],
                    "policy_archetype": row["policy_archetype"],
                    "rank_pct": row["rank_pct"],
                    "net_return": row["net_return"],
                    "notional": decision.approved_notional,
                    "invested_amount": decision.approved_notional / leverage,
                    "raw_stop_distance": 4.0 * row["effective_atr_fraction"],
                    "stressed_stop_distance": decision.sizing[
                        "stressed_stop_distance_fraction"
                    ],
                    "risk_share": shares[idx],
                    "loss_budget": decision.sizing["loss_capacity_budget"][
                        "reserved_risk_budget"
                    ],
                    "position": position,
                }
            )
            current_positions = positions()
            peak_gross = max(
                peak_gross,
                sum(abs(p.marked_notional) for p in current_positions) / equity,
            )
            peak_invested = max(
                peak_invested,
                sum(p.invested_amount for p in current_positions) / equity,
            )
            peak_raw = max(
                peak_raw,
                sum(p.stop_risk for p in current_positions) / equity,
            )
            peak_reserved = max(
                peak_reserved,
                controller.reserved_stop_risk(current_positions) / equity,
            )

    close_through(pd.Timestamp.max.tz_localize("UTC"))
    ledger = pd.DataFrame(trades)
    if not ledger.empty:
        ledger = ledger.drop(columns=["position"])
    curve = pd.DataFrame(equity_points, columns=["timestamp", "equity"]).sort_values(
        "timestamp"
    )
    curve["peak"] = curve["equity"].cummax()
    max_drawdown = float((curve["equity"] / curve["peak"] - 1.0).min())
    days = pd.date_range("2026-07-01", "2026-07-16", freq="D", tz="UTC")
    day_values = pd.Series([daily_pnl[day] for day in days], index=days)
    day_start = INITIAL_EQUITY + day_values.cumsum().shift(fill_value=0.0)
    daily_returns = day_values / day_start
    profit = equity - INITIAL_EQUITY
    payout = max(profit, 0.0) * PAYOUT_FRACTION
    summary = {
        "arm": arm,
        "universe": universe,
        "candidate_rows": int(len(rows)),
        "trades": int(len(ledger)),
        "trades_per_day": float(len(ledger) / 16.0),
        "account_profit_usd": float(profit),
        "account_return": float(profit / INITIAL_EQUITY),
        "payout_usd": float(payout),
        "net_after_purchase_usd": float(payout - PURCHASE_COST),
        "roi_on_purchase": float((payout - PURCHASE_COST) / PURCHASE_COST),
        "mean_notional_usd": float(ledger["notional"].mean()),
        "mean_invested_usd": float(ledger["invested_amount"].mean()),
        "peak_gross_notional_fraction": peak_gross,
        "peak_invested_wallet_fraction": peak_invested,
        "peak_raw_stop_risk_fraction": peak_raw,
        "peak_reserved_stressed_stop_risk_fraction": peak_reserved,
        "realized_max_drawdown": max_drawdown,
        "worst_realized_day": float(daily_returns.min()),
        "batch_count": batch_count,
        "diverse_batch_count": diverse_batch_count,
        "rejects": json.dumps(dict(rejects), sort_keys=True),
    }
    return summary, ledger


def main() -> int:
    source, representative = _inputs()
    universes = {
        "representative_60": representative,
        "all_eligible_symbols": set(source["symbol"].unique()),
    }
    summaries: list[dict[str, Any]] = []
    ledgers: list[pd.DataFrame] = []
    for universe, whitelist in universes.items():
        for arm in (
            "previous_static",
            "convex_loss_budget_1x",
            "convex_loss_budget_margin_2x",
        ):
            summary, ledger = _replay(source, whitelist, arm, universe)
            summaries.append(summary)
            ledgers.append(ledger)
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summaries).to_csv(OUT / "summary.csv", index=False)
    pd.concat(ledgers, ignore_index=True).to_parquet(OUT / "trade_ledger.parquet")
    manifest = {
        "status": "complete",
        "evidence_class": "post-selection OOS diagnostic",
        "period": "2026-07-01 through 2026-07-16 UTC",
        "source": str(
            ROOT
            / "daily_replay_july01_17_nonzero_entry_volume_v1/selected_trade_ledger.parquet"
        ),
        "frontier": "Joint trailing/raw Bayesian, rank >= 0.95, positive entry-minute volume",
        "costs": "source baseline: 1% round trip plus policy spread, applied once",
        "atr": "causal entry ATR; production Joint trailing initial stop = 4x ATR",
        "loss_budget": {
            "fraction": 0.75,
            "power": 1.5,
            "internal_boundary_fraction": 0.015,
            "stop_margin_multiplier": 1.25,
            "drawdown_reference": "static evaluation starting balance",
        },
        "diversity": {
            "single_trade_share_cap": 0.30,
            "archetype_share_cap": 0.60,
            "side_share_cap": 0.75,
        },
        "limitations": [
            "actual 60-symbol whitelist is not yet available; representative whitelist is deterministic",
            "marked intratrade equity is unavailable; drawdown gates use realized exit equity",
            "L2 is held non-binding for the matched historical comparison because full-history L2 is unavailable",
            "selected frontier already passed EV admission; replay uses the fixed 0.70% threshold value",
        ],
        "representative_whitelist": sorted(representative),
        "policies": {
            arm: asdict(_policy(arm)[0])
            for arm in (
                "previous_static",
                "convex_loss_budget_1x",
                "convex_loss_budget_margin_2x",
            )
        },
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(pd.DataFrame(summaries).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
