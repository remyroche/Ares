#!/usr/bin/env python3
"""Render a compact, reproducible execution-risk research report.

This reporter consumes immutable receipts produced by the separate manifest,
surface, oracle, and training steps.  It makes no model or policy decision and
does not write under any live-trading directory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _read_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    return json.loads(path.read_text())


def _fmt(value: object, digits: int = 2) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "n/a"
    if isinstance(value, (int, float, np.number)):
        return f"{float(value):,.{digits}f}"
    return str(value)


def _oracle_table(oracle: pd.DataFrame) -> list[str]:
    if oracle.empty or "preemption_gain_bps" not in oracle.columns:
        return ["No execution-oracle rows were supplied."]
    valid = oracle.loc[~oracle.get("insufficient_depth", pd.Series(True, index=oracle.index)).fillna(True)].copy()
    if valid.empty:
        return ["No oracle rows had sufficient executable book depth."]
    lines = ["| Earlier exit | Rows | Median gain (bps) | P(gain > 0) | P(gain > 50) |", "|---:|---:|---:|---:|---:|"]
    for minutes, group in valid.groupby("preempt_minutes", sort=True):
        gains = pd.to_numeric(group["preemption_gain_bps"], errors="coerce").dropna()
        if gains.empty:
            continue
        lines.append(
            f"| {int(minutes)}m | {len(gains):,} | {_fmt(gains.median())} | {_fmt(gains.gt(0).mean() * 100)}% | {_fmt(gains.gt(50).mean() * 100)}% |"
        )
    return lines


def render_report(
    *,
    state_receipt: dict[str, Any],
    oracle: pd.DataFrame,
    training: dict[str, Any],
) -> str:
    """Produce the required evidence headings without fabricating metrics."""
    lines = [
        "# Kraken L2 Execution-Risk Research Report",
        "",
        "## Scope and invariant",
        "",
        "This is an offline, bounded pre-emption study. The canonical close-price policy remains the authority. No result from this report changes stops, targets, trailing logic, or live execution without a separately approved contract.",
        "",
        "## Data coverage and reconstruction",
        "",
        f"- Surface schema: `{state_receipt.get('schema', 'not supplied')}`",
        f"- Processed raw L2 files: {_fmt(state_receipt.get('processed', 0), 0)}",
        f"- Notional grid (quote units): `{state_receipt.get('notional_grid_quote', [])}`",
        "- Raw files are immutable; state features begin only after a valid snapshot and source rows sharing a local timestamp are applied atomically.",
        "",
        "## Cost distribution and book quality",
        "",
        "Read `kraken_execution_state_build_audit.parquet` alongside this report for per-symbol/day coverage, pre-snapshot, crossed-book, and missing-depth rates. Do not forward-fill a broken book.",
        "",
        "## Fixed-policy exit oracle",
        "",
        *_oracle_table(oracle),
        "",
        "Oracle gain is the earlier close-price PnL change plus the difference in contemporaneous executable book cost. It is not an exit-policy optimisation target.",
        "",
        "## Chronological prediction controls",
        "",
        f"- Arm: `{training.get('arm', 'not run')}`",
        f"- Task: `{training.get('task', 'not run')}`",
        f"- Training months: `{training.get('training_months', [])}`",
        f"- Held months: `{training.get('validation_months', [])}`",
        f"- Features: `{training.get('feature_columns', [])}`",
        f"- Metrics: `{training.get('metrics', {})}`",
        "",
        "The runner compares empirical shrinkage, OHLCV-only, and L2-aware shallow controls using chronological month partitions only. It excludes future deterioration fields from inference features and has no route into the live policy.",
        "",
        "## Decision gate",
        "",
        "Advance only if reconstruction coverage is adequate, causal held-month evidence beats the empirical/OHLCV controls, L2 adds stable incremental value, and the fixed-policy oracle shows a robust economically useful earlier-exit tail. Otherwise retain the current policy unchanged.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--state-receipt", type=Path, help="JSON emitted by build_kraken_execution_states.py")
    parser.add_argument("--oracle", type=Path, help="Execution-oracle parquet")
    parser.add_argument("--training-manifest", type=Path, help="run_manifest.json from train_execution_risk.py")
    args = parser.parse_args()

    oracle = pd.read_parquet(args.oracle) if args.oracle and args.oracle.exists() else pd.DataFrame()
    text = render_report(
        state_receipt=_read_json(args.state_receipt), oracle=oracle, training=_read_json(args.training_manifest),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text)
    print(str(args.out))


if __name__ == "__main__":
    main()
