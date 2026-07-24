#!/usr/bin/env python3
"""Causal entry-halt ablation using realized policy-replay trades.

The overlay suspends *new* already-selected portfolio entries when completed
trades in a trailing window underperform their decision-time expected PnL. It
does not close positions, alter exits, or backfill skipped entries. This keeps
the test causal and isolates the value of a risk-off admission pause.
"""

from __future__ import annotations

import argparse
import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


KEYS = ["timestamp", "symbol", "side_name"]


@dataclass(frozen=True)
class Rule:
    hours: int
    deficit: float | None

    @property
    def name(self) -> str:
        if self.deficit is None:
            return f"halt_if_{self.hours}h_realized_pnl_negative"
        return f"halt_if_{self.hours}h_realized_lt_{int(self.deficit * 100)}pct_below_expected"


def _timestamp(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_datetime(frame[column], utc=True, errors="raise")


def _read_selected(path: Path) -> pd.DataFrame:
    columns = [
        "timestamp", "symbol", "side_name", "policy", "selected", "exit_bars",
        "net_pnl_bankroll", "position_size", "net_return",
    ]
    df = pd.read_parquet(path, columns=columns)
    df = df.loc[
        df["selected"].fillna(False).astype(bool)
        & df["policy"].eq("joint_trailing_plus_bayesian_raw")
    ].copy()
    df["timestamp"] = _timestamp(df, "timestamp")
    df["symbol"] = df["symbol"].astype(str)
    df["side_name"] = df["side_name"].astype(str)
    for col in ("exit_bars", "net_pnl_bankroll", "position_size", "net_return"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if df["net_pnl_bankroll"].isna().any() or df["exit_bars"].isna().any():
        raise ValueError("selected ledger has unresolved PnL or exit bars")
    df["exit_ts"] = df["timestamp"] + pd.to_timedelta(df["exit_bars"].astype(int) + 1, unit="min")
    return df.sort_values(KEYS, kind="stable").reset_index(drop=True)


def _expected_from_primary(path: Path) -> pd.DataFrame:
    columns = KEYS + ["threshold_basis_corrected_expected_ev"]
    df = pd.read_parquet(path, columns=columns).copy()
    df["timestamp"] = _timestamp(df, "timestamp")
    df["symbol"] = df["symbol"].astype(str)
    df["side_name"] = df["side_name"].astype(str)
    df = df.rename(columns={"threshold_basis_corrected_expected_ev": "expected_return"})
    df["expected_return"] = pd.to_numeric(df["expected_return"], errors="coerce")
    df["expected_source"] = "corrected_side_archetype_ev"
    return df.drop_duplicates(KEYS, keep="last")


def _expected_from_forward_context(paths: list[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for index, path in enumerate(paths):
        df = pd.read_parquet(
            path,
            columns=["__ts__", "__symbol__", "side_name", "expected_net_ev_after_1pct"],
        ).rename(columns={"__ts__": "timestamp", "__symbol__": "symbol", "expected_net_ev_after_1pct": "expected_return"})
        df["timestamp"] = _timestamp(df, "timestamp")
        df["symbol"] = df["symbol"].astype(str)
        df["side_name"] = df["side_name"].astype(str)
        df["expected_return"] = pd.to_numeric(df["expected_return"], errors="coerce")
        # These source files intentionally overlap at the July cutover. The
        # newer source wins there, matching the forward-replay report.
        df["expected_source"] = f"forward_mlp_ev_context_{index + 1}"
        frames.append(df[KEYS + ["expected_return", "expected_source"]])
    return pd.concat(frames, ignore_index=True).drop_duplicates(KEYS, keep="last")


def _join_expected(
    selected: pd.DataFrame,
    primary: pd.DataFrame,
    forward: pd.DataFrame,
    *,
    allow_zero_expected_fallback: bool,
) -> pd.DataFrame:
    expected = pd.concat([forward, primary], ignore_index=True).drop_duplicates(KEYS, keep="last")
    joined = selected.merge(expected, on=KEYS, how="left", validate="one_to_one")
    missing = joined["expected_return"].isna()
    if missing.any() and not allow_zero_expected_fallback:
        examples = joined.loc[missing, KEYS].head(5).to_dict("records")
        raise ValueError(f"missing decision-time expected EV for {int(missing.sum())} selected rows: {examples}")
    if missing.any():
        # The two forward-replay rows that lack a persisted decision-time EV
        # remain in the realized PnL stream. A zero expected contribution is
        # conservative for underperformance halts and avoids inventing a
        # future-derived expectation.
        joined.loc[missing, "expected_return"] = 0.0
        joined.loc[missing, "expected_source"] = "zero_expected_fallback_missing_forward_context"
    joined["expected_pnl_bankroll"] = joined["expected_return"] * joined["position_size"]
    return joined.sort_values(KEYS, kind="stable").reset_index(drop=True)


def _simulate(entries: pd.DataFrame, rule: Rule) -> tuple[pd.DataFrame, pd.DataFrame]:
    window = pd.Timedelta(hours=rule.hours)
    by_entry = {ts: part for ts, part in entries.groupby("timestamp", sort=True)}
    exits = entries.sort_values("exit_ts", kind="stable")
    exit_index = 0
    history: deque[tuple[pd.Timestamp, float, float]] = deque()
    realized = 0.0
    expected = 0.0
    accepted_rows: list[pd.DataFrame] = []
    trace_rows: list[dict[str, object]] = []

    for timestamp, candidates in by_entry.items():
        while exit_index < len(exits) and exits.iloc[exit_index]["exit_ts"] <= timestamp:
            exit_row = exits.iloc[exit_index]
            event = (exit_row["exit_ts"], float(exit_row["net_pnl_bankroll"]), float(exit_row["expected_pnl_bankroll"]))
            history.append(event)
            realized += event[1]
            expected += event[2]
            exit_index += 1
        cutoff = timestamp - window
        while history and history[0][0] <= cutoff:
            _, pnl, exp = history.popleft()
            realized -= pnl
            expected -= exp

        known_exits = len(history)
        if rule.deficit is None:
            halted = known_exits > 0 and realized < 0.0
        else:
            halted = known_exits > 0 and expected > 0.0 and realized < (1.0 - rule.deficit) * expected
        decision = candidates.copy()
        decision["halt_rule"] = rule.name
        decision["halted_at_entry"] = halted
        decision["rolling_realized_pnl"] = realized
        decision["rolling_expected_pnl"] = expected
        decision["rolling_completed_trades"] = known_exits
        decision["accepted_after_halt"] = bool(not halted)
        accepted_rows.append(decision)
        trace_rows.append({
            "timestamp": timestamp, "halt_rule": rule.name, "halted": halted,
            "rolling_realized_pnl": realized, "rolling_expected_pnl": expected,
            "rolling_completed_trades": known_exits, "candidate_entries": len(candidates),
        })

    decisions = pd.concat(accepted_rows, ignore_index=True)
    return decisions, pd.DataFrame(trace_rows)


def _metrics(decisions: pd.DataFrame, rule: Rule) -> dict[str, object]:
    kept = decisions.loc[decisions["accepted_after_halt"].astype(bool)].copy()
    # Equity is recognized when the path has actually resolved, which is also
    # the information available to the causal halt condition.
    exits = kept.groupby("exit_ts", as_index=False)["net_pnl_bankroll"].sum().sort_values("exit_ts")
    equity = exits["net_pnl_bankroll"].cumsum() if len(exits) else pd.Series(dtype=float)
    drawdown = equity - equity.cummax() if len(equity) else pd.Series(dtype=float)
    exit_weeks = kept.assign(week=kept["exit_ts"].dt.to_period("W-MON").dt.start_time)
    weekly = exit_weeks.groupby("week")["net_pnl_bankroll"].sum()
    elapsed_days = max(1.0, (decisions["timestamp"].max() - decisions["timestamp"].min()).total_seconds() / 86_400.0 + 1.0 / 24.0)
    return {
        "rule": rule.name,
        "rolling_hours": rule.hours,
        "underperformance_threshold": "negative" if rule.deficit is None else f"{int(rule.deficit * 100)}pct",
        "accepted_trades": int(len(kept)),
        "skipped_trades": int((~decisions["accepted_after_halt"]).sum()),
        "skip_rate": float((~decisions["accepted_after_halt"]).mean()),
        "trades_per_day": float(len(kept) / elapsed_days),
        "net_pnl_bankroll": float(kept["net_pnl_bankroll"].sum()),
        "net_return_per_trade": float(kept["net_return"].mean()) if len(kept) else np.nan,
        "max_drawdown_bankroll": float(drawdown.min()) if len(drawdown) else 0.0,
        "worst_week_pnl_bankroll": float(weekly.min()) if len(weekly) else 0.0,
        "halted_entry_bars": int(decisions.drop_duplicates("timestamp")["halted_at_entry"].sum()),
        "halted_entry_bar_rate": float(decisions.drop_duplicates("timestamp")["halted_at_entry"].mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selected-ledger", type=Path, required=True)
    parser.add_argument("--expected-ledger", type=Path, required=True)
    parser.add_argument("--forward-context", type=Path, action="append", default=[])
    parser.add_argument("--allow-zero-expected-fallback", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    selected = _read_selected(args.selected_ledger)
    joined = _join_expected(
        selected,
        _expected_from_primary(args.expected_ledger),
        _expected_from_forward_context(args.forward_context),
        allow_zero_expected_fallback=args.allow_zero_expected_fallback,
    )
    rules = [Rule(hours, None) for hours in (6, 9, 12, 15, 18, 21, 24)]
    rules += [Rule(hours, deficit) for hours in (6, 9, 12, 15, 18, 21, 24) for deficit in (0.25, 0.50, 0.75)]

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    all_metrics: list[dict[str, object]] = []
    all_traces: list[pd.DataFrame] = []
    baseline = joined.copy()
    baseline["accepted_after_halt"] = True
    baseline["halted_at_entry"] = False
    all_metrics.append(_metrics(baseline, Rule(0, None)) | {"rule": "baseline_no_halt", "rolling_hours": 0, "underperformance_threshold": "none"})
    for rule in rules:
        decisions, trace = _simulate(joined, rule)
        all_metrics.append(_metrics(decisions, rule))
        all_traces.append(trace)
    metrics = pd.DataFrame(all_metrics).sort_values(["rolling_hours", "underperformance_threshold"], kind="stable")
    metrics.to_csv(out / "rolling_pnl_entry_halt_ablation.csv", index=False)
    pd.concat(all_traces, ignore_index=True).to_parquet(out / "rolling_pnl_entry_halt_trace.parquet", index=False)
    joined.to_parquet(out / "matched_selected_entries_with_expected_ev.parquet", index=False)
    (out / "manifest.json").write_text(json.dumps({
        "selected_ledger": str(args.selected_ledger), "expected_ledger": str(args.expected_ledger),
        "forward_context": [str(path) for path in args.forward_context],
        "rows": int(len(joined)), "start": str(joined.timestamp.min()), "end": str(joined.timestamp.max()),
        "cost_contract": "net_pnl_bankroll is reused from the 1m replay; no fee or spread is subtracted again",
        "overlay": "entry-only halt; open paths and their exits are retained; skipped portfolio slots are not backfilled",
        "expected_ev_sources": joined.expected_source.value_counts().to_dict(),
    }, indent=2))
    print(metrics.to_string(index=False, float_format=lambda value: f"{value:.6f}"))


if __name__ == "__main__":
    main()
