#!/usr/bin/env python3
"""Replay causal long-entry delay-gap treatments on the frozen exact +5m panel.

This is an admission/auction ablation, not an exit-policy or mapper retune.  It
holds fixed the frozen dual BCF/current route, observed Kraken one-minute fill
at decision +5 minutes, exact one-minute rich exits, canonical policy cost,
and global portfolio constraints.  The only change is the delay component in
the live executable-EV check:

    BCF mapped net + 100 policy-cost bps - 10 bps buffer - delay_penalty >= 50

The input panel predates historical order-book snapshots, so microstructure
friction is deliberately held at zero in every arm.  Thus this isolates the
effect of delay treatment; it is not an estimate of live all-in friction.

For a long position, ``gap_bps = fill / decision_close - 1``.  The primary
production-consistent arm is ``adverse_100``: penalise only a higher delayed
fill.  ``absolute_100`` charges movement in either direction, while
``signed_100`` credits a lower fill and is deliberately a permissive control.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_strict_r3_exact_1m_rich_matched_attribution import (  # noqa: E402
    _json_safe,
    _portfolio_candidates,
    _sha256,
    _write_arm,
)


DEFAULT_INPUT = ROOT / "data_perp/artifacts/strict_r3_exact1m_state_anchor_20260817_v4_matched_ladder"
DEFAULT_OUT = ROOT / "data_perp/artifacts/strict_r3_adverse_delay_gap_20260817_v1"
POLICY_COST_BPS = 100.0
BUFFER_BPS = 10.0
ADMISSION_FLOOR_BPS = 50.0


def _assert_new(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"immutable output already exists: {path}")
    path.mkdir(parents=True, exist_ok=False)


def _penalty(gap_bps: pd.Series, kind: str) -> pd.Series:
    gap = pd.to_numeric(gap_bps, errors="raise").astype(float)
    adverse = gap.clip(lower=0.0)
    if kind == "none":
        return pd.Series(0.0, index=gap.index)
    if kind.startswith("adverse_"):
        return adverse * float(kind.rsplit("_", 1)[1]) / 100.0
    if kind == "absolute_100":
        return gap.abs()
    if kind == "signed_100":
        return gap
    raise ValueError(f"unknown delay treatment {kind}")


def _summary(
    arm: str,
    metrics: dict[str, Any],
    source_rows: int,
    admitted_rows: int,
    gap: pd.Series,
    penalty: pd.Series,
) -> dict[str, Any]:
    return {
        **metrics,
        "source_common_route_rows": int(source_rows),
        "execution_admitted_rows": int(admitted_rows),
        "admission_fraction": float(admitted_rows / max(source_rows, 1)),
        "delay_gap_mean_bps": float(gap.mean()),
        "delay_gap_p95_bps": float(gap.quantile(0.95)),
        "delay_gap_adverse_fraction": float((gap.gt(0.0)).mean()),
        "delay_penalty_mean_bps": float(penalty.mean()),
        "delay_penalty_p95_bps": float(penalty.quantile(0.95)),
    }


def run(args: argparse.Namespace) -> Path:
    source = Path(args.input_dir).resolve()
    output = Path(args.out_dir).resolve()
    _assert_new(output)
    coverage = pd.read_parquet(source / "fill_anchor_routed_outcome_coverage.parquet").copy()
    outcomes = pd.read_parquet(source / "fill_anchor_outcomes.parquet").copy()
    candidates = pd.read_parquet(source / "fill_anchor_portfolio_candidates.parquet").copy()
    required_coverage = {
        "candidate_id", "timestamp", "symbol", "priority_bps", "bcf_mc1_expected_bps",
        "current_v5_mc1_expected_bps", "decision_timestamp", "entry_timestamp", "outcome_available",
    }
    if not required_coverage.issubset(coverage.columns):
        raise ValueError(f"coverage missing {sorted(required_coverage - set(coverage.columns))}")
    if not coverage["outcome_available"].fillna(False).astype(bool).all():
        raise AssertionError("exact common panel must contain only complete outcomes")
    coverage["candidate_id"] = coverage["candidate_id"].astype(str)
    outcomes["candidate_id"] = outcomes["candidate_id"].astype(str)
    candidates["candidate_id"] = candidates["candidate_id"].astype(str)
    if coverage["candidate_id"].duplicated().any() or outcomes["candidate_id"].duplicated().any():
        raise AssertionError("input common panel has duplicate IDs")
    if set(coverage["candidate_id"]) != set(outcomes["candidate_id"]):
        raise AssertionError("coverage/outcome IDs differ")
    # The decision-close state arm contains the contemporaneously observable
    # completed-candle reference price.  Its anchor gap is exactly fill vs
    # decision close for a long, with no future path input.
    decision = pd.read_parquet(source / "decision_close_anchor_outcomes.parquet", columns=[
        "candidate_id", "state_anchor_price", "anchor_gap_bps",
    ]).copy()
    decision["candidate_id"] = decision["candidate_id"].astype(str)
    panel = coverage.merge(
        decision, on="candidate_id", how="inner", validate="one_to_one"
    ).merge(
        outcomes, on="candidate_id", how="inner", validate="one_to_one", suffixes=("", "_outcome")
    )
    if len(panel) != len(coverage):
        raise AssertionError("delay panel lost common candidate identities")
    if not np.isclose(
        pd.to_numeric(panel["priority_bps"], errors="raise"),
        pd.to_numeric(panel["bcf_mc1_expected_bps"], errors="raise"),
        rtol=0.0, atol=1e-12,
    ).all():
        raise AssertionError("common route is not BCF-priority")
    raw_expected_gross = pd.to_numeric(panel["bcf_mc1_expected_bps"], errors="raise") + POLICY_COST_BPS
    gap = pd.to_numeric(panel["anchor_gap_bps"], errors="raise")
    treatment_kinds = ["none", "adverse_50", "adverse_75", "adverse_100", "adverse_125", "absolute_100", "signed_100"]
    metric_rows: list[dict[str, Any]] = []
    month_rows: list[pd.DataFrame] = []
    exit_rows: list[pd.DataFrame] = []
    for kind in treatment_kinds:
        penalty = _penalty(gap, kind)
        execution_adjusted = raw_expected_gross - BUFFER_BPS - penalty
        admitted = execution_adjusted.ge(ADMISSION_FLOOR_BPS)
        routed = panel.loc[admitted, [
            "candidate_id", "timestamp", "symbol",
            "bcf_mc1_expected_bps", "current_v5_mc1_expected_bps",
        ]].copy()
        routed["priority_bps"] = execution_adjusted.loc[admitted].to_numpy(float)
        if routed.empty:
            raise AssertionError(f"{kind}: no entries passed executable-EV floor")
        outcome = panel.loc[admitted, [
            "candidate_id", "decision_timestamp_outcome", "entry_timestamp_outcome", "entry_price",
            "exit_timestamp", "exit_price", "gross_bps", "net_bps", "exit_reason", "outcome_available",
            "outcome_invalid_reason", "outcome_source",
        ]].rename(columns={
            "decision_timestamp_outcome": "decision_timestamp",
            "entry_timestamp_outcome": "entry_timestamp",
        }).copy()
        if outcome.columns.duplicated().any():
            raise AssertionError(f"{kind}: duplicate outcome columns")
        candidate, population = _portfolio_candidates(routed, outcome, arm=kind)
        metrics, month, exits, _ = _write_arm(output, kind, candidate, population)
        metric_rows.append(_summary(kind, metrics, len(panel), int(admitted.sum()), gap, penalty))
        month_rows.append(month)
        exit_rows.append(exits)
        audit = panel.loc[:, [
            "candidate_id", "timestamp", "symbol", "bcf_mc1_expected_bps", "current_v5_mc1_expected_bps",
            "anchor_gap_bps", "state_anchor_price", "entry_price",
        ]].copy()
        audit["raw_expected_gross_bps"] = raw_expected_gross
        audit["delay_penalty_bps"] = penalty
        audit["execution_adjusted_expected_gross_bps"] = execution_adjusted
        audit["execution_admitted"] = admitted
        audit.to_parquet(output / f"{kind}_admission_audit.parquet", index=False, compression="zstd")
    summary = pd.DataFrame(metric_rows)
    baseline = summary.set_index("arm").loc["adverse_100"]
    for col in (
        "portfolio_accepted_trades", "net_ev_bps_per_trade", "net_sum_bps", "portfolio_net_pnl_quote",
        "portfolio_max_drawdown", "portfolio_sortino", "portfolio_worst_week_return", "execution_admitted_rows",
    ):
        summary[f"delta_vs_adverse_100_{col}"] = pd.to_numeric(summary[col], errors="coerce") - float(baseline[col])
    summary.to_csv(output / "summary_metrics.csv", index=False)
    summary.to_parquet(output / "summary_metrics.parquet", index=False, compression="zstd")
    pd.concat(month_rows, ignore_index=True).to_parquet(output / "monthly_metrics.parquet", index=False, compression="zstd")
    pd.concat(exit_rows, ignore_index=True).to_parquet(output / "exit_reason_metrics.parquet", index=False, compression="zstd")
    (output / "run_manifest.json").write_text(json.dumps(_json_safe({
        "schema": "strict_r3_adverse_delay_gap_ablation_v1",
        "status": "complete",
        "purpose": "causal entry-time delay-gap treatment; no live contract change",
        "input_dir": str(source),
        "input_hashes": {name: _sha256(source / name) for name in [
            "fill_anchor_routed_outcome_coverage.parquet", "fill_anchor_outcomes.parquet",
            "decision_close_anchor_outcomes.parquet", "fill_anchor_portfolio_candidates.parquet",
        ]},
        "source_rows": int(len(panel)),
        "entry": "observed Kraken one-minute open at decision +5 minutes",
        "delay_gap": "long fill / completed decision-candle close - 1, in bps",
        "formula": "BCF mapped net + 100 policy-cost bps - 10 bps buffer - delay penalty >= 50 bps; historical microstructure held zero across all arms",
        "treatments": treatment_kinds,
        "portfolio": "same BCF-MC1-priority global auction: 7x, 80% margin, 10% slots, two entries per decision",
        "cost": "100 bps already present once in realised policy net; no second debit in auction",
        "code_sha256": _sha256(Path(__file__).resolve()),
    }), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    print(run(args))


if __name__ == "__main__":
    main()
