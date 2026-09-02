#!/usr/bin/env python3
"""Publish the immutable no-order report for a completed stateful recovery.

The recovery scorer deliberately never requests a time-travel live order book.
Consequently this report derives only a *spread-only* execution-adjusted EV
proxy from the archived decision-time bid/ask.  It never pretends that missing
historical depth or a later real-world delay was known.  The production entry
executor still performs the complete VWAP/impact/delay recheck immediately
before any fresh live order.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "data_perp" / "artifacts"
SCHEMA = "strict_r3_stateful_recovery_report_v1"


def _path(value: Path) -> Path:
    return value if value.is_absolute() else ROOT / value


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


def _spread_bps(frame: pd.DataFrame) -> pd.Series:
    direct = pd.to_numeric(frame.get("spread_bps"), errors="coerce")
    bid = pd.to_numeric(frame.get("decision_book_bid"), errors="coerce")
    ask = pd.to_numeric(frame.get("decision_book_ask"), errors="coerce")
    midpoint = 0.5 * (bid + ask)
    derived = (ask - bid) / midpoint * 10_000.0
    return direct.where(np.isfinite(direct), derived)


def _one_hour(hour_dir: Path) -> tuple[pd.DataFrame, dict]:
    receipt = _read(hour_dir / "recovery_hour_manifest.json")
    run_dir = _path(Path(receipt["run"]))
    decisions = pd.read_parquet(run_dir / "cycle" / "shadow_decisions.parquet")
    candidates = pd.read_parquet(
        run_dir / "candidate_grid" / "target_free_candidate_population.parquet",
        columns=[
            "candidate_id", "decision_book_bid", "decision_book_ask", "spread_bps",
            "decision_open", "decision_open_source",
        ],
    )
    rows = decisions.loc[
        decisions["dual_bcf_current_admitted_ge_30bps"].fillna(False)
        | decisions["portfolio_accepted"].fillna(False)
    ].copy()
    selected = {
        "candidate_id", "__decision_ts__", "__symbol__", "decision_open",
        "bcf_mc1_expected_net_bps", "mc1_d2_expected_net_bps",
        "dual_auction_priority_bps", "portfolio_accepted",
        "portfolio_rejection_reason", "portfolio_priority_rank",
        "portfolio_initial_margin", "portfolio_gross_notional", "policy_cost_bps_once",
    }
    missing = selected.difference(rows.columns)
    if missing:
        raise KeyError(f"shadow decision schema missing: {sorted(missing)}")
    rows = rows.loc[:, sorted(selected)].merge(
        candidates, on=["candidate_id", "decision_open"], how="left", validate="one_to_one"
    )
    rows["decision_timestamp"] = pd.to_datetime(rows["__decision_ts__"], utc=True)
    # In the no-order replay the simulated entry is the executable decision
    # boundary, not a later actual fill.  Zero delay and unavailable VWAP
    # impact are explicit rather than silently treated as observed values.
    rows["simulated_entry_timestamp"] = rows["decision_timestamp"]
    rows["simulated_entry_price"] = pd.to_numeric(rows["decision_open"], errors="coerce")
    rows["source_spread_bps"] = _spread_bps(rows)
    rows["source_execution_delay_gap_bps"] = 0.0
    rows["source_entry_impact_bps"] = np.nan
    rows["source_execution_buffer_bps"] = 10.0
    rows["source_execution_adjusted_ev_bps"] = (
        pd.to_numeric(rows["bcf_mc1_expected_net_bps"], errors="coerce")
        + pd.to_numeric(rows["policy_cost_bps_once"], errors="coerce")
        - 1.2 * rows["source_spread_bps"]
        - rows["source_execution_buffer_bps"]
    )
    rows["source_execution_adjusted_ev_method"] = (
        "decision-time bid/ask spread only; zero simulated delay; "
        "depth/VWAP impact unavailable in archived source"
    )
    rows["status"] = np.where(
        rows["portfolio_accepted"].fillna(False),
        "portfolio_accepted_simulated",
        "dual_admitted_not_selected",
    )
    rows["rejection_reason"] = np.where(
        rows["portfolio_accepted"].fillna(False),
        "portfolio_accepted",
        rows["portfolio_rejection_reason"].fillna("portfolio_rejected_without_reason"),
    )
    summary = {
        "decision_timestamp": receipt["decision_ts"],
        "feature_complete_rows": int(receipt["feature_complete_rows"]),
        "eligible_rows": int(receipt["eligible_rows"]),
        "dual_admitted_rows": int(
            decisions["dual_bcf_current_admitted_ge_30bps"].fillna(False).sum()
        ),
        "portfolio_accepted_rows": int(receipt["portfolio_accepted_rows"]),
        "exchange_calls": int(receipt["exchange_calls"]),
        "order_submission_enabled": bool(receipt["order_submission_enabled"]),
        "geometry_bundle_sha256": str(receipt["geometry_bundle_sha256"]),
        "run": str(run_dir.relative_to(ROOT)),
    }
    source = dict(receipt.get("source") or {})
    fifteen = dict(source.get("fifteen_minute") or {})
    coverage = dict(fifteen.get("coverage_after_retry") or {})
    summary.update({
        "source_refresh_seconds": float(source.get("duration_seconds", float("nan"))),
        "source_future_bars_requested": bool(source.get("future_bars_requested")),
        "source_feature_ready_symbols": int(coverage.get("feature_source_ready", 0)),
        "source_missing_15m_bar_symbols": int(coverage.get("missing_15m_bar", 0)),
        "source_synthetic_flat_bar_symbols": int(coverage.get("synthetic_flat_bar", 0)),
        "source_missing_decision_open_symbols": int(coverage.get("missing_decision_open", 0)),
        "source_retry_attempts": int(len(fifteen.get("retry_attempts") or [])),
    })
    return rows, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recovery-root", type=Path, required=True)
    args = parser.parse_args()
    recovery_root = _path(args.recovery_root)
    root_manifest = _read(recovery_root / "run_manifest.json")
    if root_manifest.get("schema") != "strict_r3_stateful_recovery_v1" or root_manifest.get("status") != "complete":
        raise ValueError("recovery root is not a completed immutable recovery")
    hourly = sorted(recovery_root.glob("hour_*/recovery_hour_manifest.json"))
    if len(hourly) != len(root_manifest.get("hours") or []):
        raise AssertionError("hourly receipt count does not match completed recovery manifest")
    report_rows: list[pd.DataFrame] = []
    summaries: list[dict] = []
    for manifest in hourly:
        rows, summary = _one_hour(manifest.parent)
        report_rows.append(rows)
        summaries.append(summary)
    summary_frame = pd.DataFrame(summaries).sort_values("decision_timestamp", kind="stable")
    if (summary_frame["exchange_calls"] != 0).any() or summary_frame["order_submission_enabled"].any():
        raise AssertionError("recovery report found exchange activity")
    if summary_frame["geometry_bundle_sha256"].nunique() != 1:
        raise AssertionError("frozen Geometry/K9 bundle changed within recovery")
    positions = (
        pd.concat(report_rows, ignore_index=True)
        if report_rows else pd.DataFrame()
    )
    positions.to_parquet(recovery_root / "missed_hour_positions.parquet", index=False, compression="zstd")
    positions.to_csv(recovery_root / "missed_hour_positions.csv", index=False)
    summary_frame.to_parquet(recovery_root / "missed_hour_summary.parquet", index=False, compression="zstd")
    summary_frame.to_csv(recovery_root / "missed_hour_summary.csv", index=False)
    # A deliberately narrow coverage receipt makes source availability easy to
    # inspect independently from model/admission outcomes.
    coverage_columns = [
        "decision_timestamp", "source_refresh_seconds", "source_future_bars_requested",
        "source_feature_ready_symbols", "source_missing_15m_bar_symbols",
        "source_synthetic_flat_bar_symbols", "source_missing_decision_open_symbols",
        "source_retry_attempts", "feature_complete_rows", "eligible_rows",
    ]
    summary_frame.loc[:, coverage_columns].to_parquet(
        recovery_root / "per_hour_source_feature_coverage.parquet", index=False,
        compression="zstd",
    )
    summary_frame.loc[:, coverage_columns].to_csv(
        recovery_root / "per_hour_source_feature_coverage.csv", index=False,
    )
    report_manifest = {
        "schema": SCHEMA,
        "status": "complete",
        "recovery_manifest": str((recovery_root / "run_manifest.json").relative_to(ROOT)),
        "hours": int(len(summary_frame)),
        "positions_or_admitted_not_selected_rows": int(len(positions)),
        "portfolio_mode": root_manifest.get("portfolio_mode"),
        "exchange_calls": 0,
        "order_submission_enabled": False,
        "execution_adjusted_ev": (
            "spread-only decision-time proxy; full depth/VWAP impact and real delay are intentionally unavailable in no-order historical replay"
        ),
        "geometry_bundle_sha256": str(summary_frame["geometry_bundle_sha256"].iloc[0]),
    }
    (recovery_root / "missed_hour_position_report_manifest.json").write_text(
        json.dumps(report_manifest, indent=2) + "\n"
    )
    print(json.dumps(report_manifest, sort_keys=True))


if __name__ == "__main__":
    main()
