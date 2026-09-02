#!/usr/bin/env python3
"""No-order, receipt-backed dual-MC1 replay over a bounded recent window."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.replay_strict_r3_bcf_exact5m_1m import (
    _exact_labels,
    _policy,
    _portfolio_candidates,
    _run_portfolio,
)


POLICY = ROOT / "data_perp/artifacts/strict_r3_schema_v2_simple_policy_targetfree_long_pre2025_20260809_v3/winner.json"

KEEP = [
    "candidate_id", "__decision_ts__", "__symbol__", "final_score",
    "frozen_base_contract_complete", "base_route_timestamp_top30",
    "mc1_d2_expected_net_bps", "bcf_mc1_expected_net_bps",
    "bcf_mc1_available", "current_mc1_admitted_ge_30bps",
    "bcf_mc1_admitted_ge_30bps", "dual_bcf_current_admitted_ge_30bps",
    "dual_auction_priority_bps", "decision_open", "signal_atr",
    "policy_sl_atr", "policy_trailing_activation_atr",
    "policy_trailing_giveback_atr", "policy_timeout_hours",
    "policy_cost_bps_once", "portfolio_accepted",
    "portfolio_rejection_reason", "portfolio_priority_rank",
    "portfolio_open_positions_before", "portfolio_committed_margin_before",
    "portfolio_margin_cap", "shadow_action", "dual_admission_rejection_reason",
]


def _utc(value: str) -> pd.Timestamp:
    item = pd.Timestamp(value)
    return item.tz_localize("UTC") if item.tzinfo is None else item.tz_convert("UTC")


def _source_rank(path: Path, count: int) -> tuple[int, int, str]:
    text = str(path)
    if "successor_" in text and "_live_" in text:
        priority = 0
    elif "stateful_recovery" in text:
        priority = 1
    elif "feature_runtime_equivalence" in text:
        priority = 2
    elif "backfill" in text:
        priority = 3
    else:
        priority = 4
    return priority, -count, text


def _receipts(start: pd.Timestamp, end: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    chosen: dict[pd.Timestamp, tuple[tuple[int, int, str], Path]] = {}
    for path in ROOT.glob("data_perp/artifacts/**/cycle/shadow_decisions.parquet"):
        text = str(path)
        if "terminal" in text or "parity" in text:
            continue
        try:
            tiny = pd.read_parquet(path, columns=["__decision_ts__"])
            timestamp = pd.to_datetime(tiny["__decision_ts__"].iloc[0], utc=True)
        except Exception:
            continue
        if not start <= timestamp < end:
            continue
        candidate = (_source_rank(path, len(tiny)), path)
        current = chosen.get(timestamp)
        if current is None or candidate[0] < current[0]:
            chosen[timestamp] = candidate
    coverage = pd.date_range(start, end - pd.Timedelta(hours=1), freq="1h", tz="UTC")
    missing = [stamp.isoformat() for stamp in coverage if stamp not in chosen]
    frames: list[pd.DataFrame] = []
    audit: list[dict[str, object]] = []
    for timestamp, (_, path) in sorted(chosen.items()):
        frame = pd.read_parquet(path)
        missing_columns = sorted(set(KEEP).difference(frame.columns))
        if missing_columns:
            raise ValueError(f"{path} lacks {missing_columns}")
        frame = frame.loc[:, KEEP].copy()
        frame["__decision_ts__"] = pd.to_datetime(frame["__decision_ts__"], utc=True)
        if frame["__decision_ts__"].nunique() != 1 or frame["__decision_ts__"].iloc[0] != timestamp:
            raise ValueError(f"receipt timestamp mismatch: {path}")
        if frame["candidate_id"].duplicated().any():
            raise ValueError(f"duplicate candidate IDs: {path}")
        frame["source_receipt"] = str(path.relative_to(ROOT))
        frames.append(frame)
        audit.append({"decision_ts": timestamp, "rows": len(frame), "source_receipt": str(path.relative_to(ROOT))})
    return pd.concat(frames, ignore_index=True), pd.DataFrame(audit), missing


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True, help="exclusive decision timestamp")
    parser.add_argument("--as-of", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    start, end, as_of = _utc(args.start), _utc(args.end), _utc(args.as_of)
    if args.out_dir.exists():
        raise FileExistsError(f"immutable output already exists: {args.out_dir}")
    rows, coverage, missing = _receipts(start, end)
    for column in ["dual_bcf_current_admitted_ge_30bps", "base_route_timestamp_top30", "portfolio_accepted"]:
        rows[column] = rows[column].fillna(False).astype(bool)
    rows["dual_admitted"] = rows["dual_bcf_current_admitted_ge_30bps"]
    rows["resolved_by_asof"] = rows["__decision_ts__"] + pd.Timedelta(hours=12, minutes=5) <= as_of
    selected = rows.loc[rows["dual_admitted"]].copy()
    resolved_request = selected.loc[selected["resolved_by_asof"], ["candidate_id", "__decision_ts__", "__symbol__"]].copy()
    policy = _policy(POLICY)
    labels = _exact_labels(
        resolved_request,
        resolved_request,
        data_root=ROOT / "data_perp",
        policy=policy,
        atr_source="canonical_15m_aggregated",
        entry_delay_minutes=5,
    ) if len(resolved_request) else pd.DataFrame()
    if len(labels):
        labels["outcome_status"] = np.where(labels["policy_path_valid"].fillna(False).astype(bool), "resolved", "invalid_source")
        valid_labels = labels.loc[labels["outcome_status"].eq("resolved")].copy()
    else:
        valid_labels = labels
    selected_resolved = selected.merge(valid_labels, on=["candidate_id", "__decision_ts__", "__symbol__"], how="inner", validate="one_to_one") if len(valid_labels) else pd.DataFrame()
    if len(selected_resolved):
        bcf = selected_resolved.loc[:, ["candidate_id", "bcf_mc1_expected_net_bps"]].rename(columns={"bcf_mc1_expected_net_bps": "mc1_expected_bps"})
        current = selected_resolved.loc[:, ["candidate_id", "mc1_d2_expected_net_bps"]].rename(columns={"mc1_d2_expected_net_bps": "mc1_expected_bps"})
        candidates = _portfolio_candidates(valid_labels, bcf, current, threshold_bps=30.0, entry_delay_minutes=5)
        decisions, equity, engine_metrics = _run_portfolio(candidates)
    else:
        candidates, decisions, equity, engine_metrics = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {}
    accepted = decisions.loc[decisions.get("accepted", pd.Series(False, index=decisions.index)).fillna(False).astype(bool)].copy()
    if len(accepted):
        trade_columns = [
            "candidate_id", "timestamp", "symbol", "mapped_expected_net_bps", "bcf_mc1_expected_bps",
            "current_mc1_expected_bps", "entry_price", "exit_timestamp", "exit_price",
            "simple_policy_exit_reason", "net_return", "gross_return", "position_net_return",
            "position_gross_return", "holding_bars",
        ]
        trade_columns = [column for column in trade_columns if column in accepted]
        trades = accepted.loc[:, trade_columns].copy().rename(columns={"timestamp": "entry_timestamp", "simple_policy_exit_reason": "exit_reason"})
        for source, target in [("net_return", "policy_net_bps"), ("gross_return", "policy_gross_bps"), ("position_net_return", "portfolio_net_bps"), ("position_gross_return", "portfolio_gross_bps")]:
            if source in trades:
                trades[target] = pd.to_numeric(trades[source], errors="coerce") * 10_000.0
        trades = trades.merge(rows.loc[:, ["candidate_id", "final_score", "decision_open", "signal_atr"]], on="candidate_id", how="left", validate="one_to_one")
        net = pd.to_numeric(accepted["position_net_return"], errors="coerce") * 10_000.0
        aggregate = {
            "trades": int(len(accepted)), "net_bps_sum": float(net.sum()), "net_bps_mean": float(net.mean()),
            "net_bps_median": float(net.median()), "hit_rate": float((net > 0).mean()),
            "wins": int((net > 0).sum()), "losses": int((net <= 0).sum()),
            "best_net_bps": float(net.max()), "worst_net_bps": float(net.min()),
        }
    else:
        trades, aggregate = pd.DataFrame(), {"trades": 0}
    pending = selected.loc[~selected["resolved_by_asof"], [
        "candidate_id", "__decision_ts__", "__symbol__", "bcf_mc1_expected_net_bps",
        "mc1_d2_expected_net_bps", "dual_auction_priority_bps", "portfolio_accepted",
        "portfolio_rejection_reason",
    ]].copy()
    args.out_dir.mkdir(parents=True)
    rows.to_parquet(args.out_dir / "receipt_backed_funnel.parquet", index=False, compression="zstd")
    coverage.to_parquet(args.out_dir / "receipt_coverage.parquet", index=False, compression="zstd")
    labels.to_parquet(args.out_dir / "exact1m_outcomes.parquet", index=False, compression="zstd")
    candidates.to_parquet(args.out_dir / "resolved_portfolio_candidates.parquet", index=False, compression="zstd")
    decisions.to_parquet(args.out_dir / "resolved_portfolio_decisions.parquet", index=False, compression="zstd")
    equity.to_parquet(args.out_dir / "resolved_portfolio_equity.parquet", index=False, compression="zstd")
    trades.to_parquet(args.out_dir / "per_trade_metrics.parquet", index=False, compression="zstd")
    pending.to_parquet(args.out_dir / "pending_dual_admissions.parquet", index=False, compression="zstd")
    manifest = {
        "schema": "strict_r3_dual_live_last48h_receipt_backed_replay_v1",
        "exchange_calls": 0, "order_submission_enabled": False,
        "range": {"start": start.isoformat(), "end_exclusive": end.isoformat(), "as_of": as_of.isoformat()},
        "coverage": {"expected_hours": int(len(pd.date_range(start, end - pd.Timedelta(hours=1), freq="1h"))), "covered_hours": int(len(coverage)), "missing_hours": missing},
        "contract": {"admission": "BCF MC1 >= +30 bps AND current-v5 MC1 >= +30 bps", "priority": "BCF MC1 expected bps", "outcome": "exact Kraken 1m decision+5 frozen parent policy; 100 bps cost exactly once; no Adaptive Exit V1 overlay"},
        "funnel": {"target_free_rows": int(len(rows)), "base_routed": int(rows["base_route_timestamp_top30"].sum()), "dual_admitted": int(len(selected)), "receipt_portfolio_accepted": int(selected["portfolio_accepted"].sum()), "resolved_dual_requested": int(len(resolved_request)), "resolved_valid_paths": int(len(valid_labels)), "pending_dual": int(len(pending)), "replay_portfolio_accepted": int(len(accepted))},
        "aggregate": aggregate, "engine_metrics": engine_metrics,
    }
    (args.out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n")
    print(json.dumps({"event": "complete", "out_dir": str(args.out_dir), **manifest["funnel"], **aggregate}, sort_keys=True))


if __name__ == "__main__":
    main()
