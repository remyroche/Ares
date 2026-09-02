#!/usr/bin/env python3
"""Replay a sealed target-free P8U route on corrected exact 1-minute outcomes.

This is an offline research/audit utility.  It deliberately performs no model
scoring, exchange I/O, account reads, or order submission.  It first verifies
that the route was selected without outcomes, then joins the already-resolved
exact outcomes and applies the canonical fixed-slot chronological auction.
Invalid exact paths are excluded *after* routing and are reported explicitly;
they never become losses or consume historical portfolio capacity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.portfolio_policy_replay import (  # noqa: E402
    compute_replay_metrics,
    normalise_candidate_table,
    replay_candidates,
)
from scripts.report_strict_r3_mc1_d2_controlled_portfolio import (  # noqa: E402
    CAUSAL_AUCTION_CURVE,
    _params as canonical_portfolio_params,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_once(path: Path, payload: Any) -> None:
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")


def _read_route(root: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    manifest_path = root / "candidate_manifest.json"
    candidate_path = root / "candidates.parquet"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("target_free") is not True:
        raise AssertionError("candidate route must be explicitly target-free")
    if str(manifest.get("candidate_sha256")) != _sha256(candidate_path):
        raise AssertionError("candidate parquet differs from its sealed manifest")
    selection = dict(manifest.get("selection") or {})
    if float(selection.get("bcf_mc1_expected_bps_gte", np.nan)) != 50.0:
        raise AssertionError("replay only supports the frozen BCF 50-bps route")
    if float(selection.get("current_mc1_expected_bps_gte", np.nan)) != 50.0:
        raise AssertionError("replay only supports the frozen Current 50-bps route")
    if selection.get("priority") != "bcf_mc1_expected_bps":
        raise AssertionError("route must bind BCF-MC1 auction priority")
    route = pd.read_parquet(candidate_path).copy()
    route["candidate_id"] = route["candidate_id"].astype(str)
    route["timestamp"] = pd.to_datetime(route["timestamp"], utc=True, errors="raise")
    route["entry_ts"] = pd.to_datetime(route["entry_ts"], utc=True, errors="raise")
    if route["candidate_id"].duplicated().any():
        raise AssertionError("target-free route has duplicate candidate identities")
    if not route["side_name"].astype(str).str.lower().eq("long").all():
        raise AssertionError("P8U exact portfolio replay is long-only")
    if not route["entry_ts"].eq(route["timestamp"] + pd.Timedelta(minutes=5)).all():
        raise AssertionError("route does not bind the declared +5-minute entry")
    return route, manifest


def _read_outcomes(root: Path, route: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    manifest_path = root / "run_manifest.json"
    outcome_path = root / "exact_1m_policy_outcomes.parquet"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise AssertionError("exact outcome materialisation is incomplete")
    if int(manifest.get("candidate_rows", -1)) != len(route):
        raise AssertionError("exact outcomes do not cover the sealed route population")
    oracle = dict(manifest.get("oracle_equivalence") or {})
    valid_expected = int(manifest.get("candidate_rows", 0)) - int(manifest.get("invalid_outcome_rows", 0))
    if int(oracle.get("live_state_machine_equivalence_rows", -1)) != valid_expected:
        raise AssertionError("exact outcome materialisation lacks exhaustive live-state parity")
    outcome = pd.read_parquet(outcome_path).copy()
    outcome["candidate_id"] = outcome["candidate_id"].astype(str)
    if outcome["candidate_id"].duplicated().any() or set(outcome["candidate_id"]) != set(route["candidate_id"]):
        raise AssertionError("exact outcome identities differ from the sealed route")
    outcome["decision_timestamp"] = pd.to_datetime(outcome["decision_timestamp"], utc=True, errors="raise")
    outcome["entry_timestamp"] = pd.to_datetime(outcome["entry_timestamp"], utc=True, errors="raise")
    outcome["exit_timestamp"] = pd.to_datetime(outcome["exit_timestamp"], utc=True, errors="coerce")
    return outcome, manifest


def _portfolio_candidates(route: pd.DataFrame, outcome: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    merged = route.merge(outcome, on="candidate_id", how="left", validate="one_to_one")
    if merged["outcome_available"].isna().any():
        raise AssertionError("outcome frame does not account for every target-free candidate")
    population = merged.loc[:, [
        "candidate_id", "timestamp", "symbol", "priority_bps", "decision_timestamp",
        "entry_timestamp", "outcome_available", "outcome_invalid_reason", "outcome_source",
    ]].copy()
    selected = merged.loc[merged["outcome_available"].fillna(False).astype(bool)].copy()
    if selected.empty:
        raise RuntimeError("no valid exact outcomes after target-free routing")
    entry = pd.to_datetime(selected["entry_timestamp"], utc=True, errors="raise")
    exit_ = pd.to_datetime(selected["exit_timestamp"], utc=True, errors="raise")
    holding = np.maximum(1, np.ceil((exit_ - entry).dt.total_seconds().to_numpy(float) / 900.0)).astype(int)
    candidates = pd.DataFrame({
        "timestamp": entry,
        "decision_timestamp": pd.to_datetime(selected["timestamp"], utc=True, errors="raise"),
        "candidate_id": selected["candidate_id"].astype(str),
        "symbol": selected["symbol"].astype(str),
        "side": "long",
        "strategy_id": "strict_r3_exact_1m_rich_matched_long",
        "policy_archetype": "strict_r3_exact_1m_rich_matched_long",
        # Fixed slots: rank is intentionally inert. BCF expected net remains
        # the only auction ordering authority through priority adjustment.
        "normalized_rank_score": 1.0,
        "strategy_rank_pct": 1.0,
        "base_strategy_threshold": 0.0,
        "calibrated_score": 1.0,
        "portfolio_priority_adjustment": pd.to_numeric(selected["priority_bps"], errors="raise"),
        "entry_price": pd.to_numeric(selected["entry_price"], errors="raise"),
        "exit_timestamp": exit_,
        "exit_price": pd.to_numeric(selected["exit_price"], errors="raise"),
        "net_return": pd.to_numeric(selected["net_bps"], errors="raise") / 10_000.0,
        "gross_return": pd.to_numeric(selected["gross_bps"], errors="raise") / 10_000.0,
        "holding_bars": holding,
        "simple_policy_exit_reason": selected["exit_reason"].astype(str),
        # net_bps already has the one 100-bps policy cost embedded.
        "fees_bps": 100.0,
        "expected_friction_bps": 0.0,
        "price_gap_bps": 0.0,
        "liquidity_capacity_weight": 1.0,
        "policy_outcome_available": True,
    })
    return normalise_candidate_table(candidates), population


def _attach_identities(decisions: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    result = decisions.copy()
    indices = pd.to_numeric(result["candidate_index"], errors="raise").astype(int)
    if (indices < 0).any() or (indices >= len(candidates)).any():
        raise AssertionError("portfolio decision refers outside the candidate table")
    result["candidate_id"] = candidates.iloc[indices.to_numpy()]["candidate_id"].to_numpy()
    result["decision_timestamp"] = candidates.iloc[indices.to_numpy()]["decision_timestamp"].to_numpy()
    return result


def _daily(accepted: pd.DataFrame) -> pd.DataFrame:
    data = accepted.copy()
    data["day"] = pd.to_datetime(data["decision_timestamp"], utc=True).dt.strftime("%Y-%m-%d")
    data["net_bps"] = pd.to_numeric(data["position_net_return"], errors="raise") * 10_000.0
    data["gross_bps"] = pd.to_numeric(data["position_gross_return"], errors="raise") * 10_000.0
    data["portfolio_net_pnl_quote"] = (
        pd.to_numeric(data["position_size"], errors="raise")
        * pd.to_numeric(data["position_net_return"], errors="raise")
    )
    result = data.groupby("day", sort=True).agg(
        portfolio_accepted_trades=("candidate_id", "size"),
        net_ev_bps_per_trade=("net_bps", "mean"),
        gross_ev_bps_per_trade=("gross_bps", "mean"),
        net_sum_bps=("net_bps", "sum"),
        gross_sum_bps=("gross_bps", "sum"),
        win_rate=("net_bps", lambda value: float((value > 0).mean())),
        portfolio_net_pnl_quote=("portfolio_net_pnl_quote", "sum"),
    ).reset_index()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-dir", required=True, type=Path)
    parser.add_argument("--outcome-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()
    out = args.out_dir.resolve()
    if out.exists():
        raise FileExistsError(f"immutable output already exists: {out}")
    route, route_manifest = _read_route(args.candidate_dir.resolve())
    outcome, outcome_manifest = _read_outcomes(args.outcome_dir.resolve(), route)
    candidates, coverage = _portfolio_candidates(route, outcome)
    decisions, equity, _ = replay_candidates(
        candidates,
        canonical_portfolio_params(),
        mode="global_auction",
        ev_curve=CAUSAL_AUCTION_CURVE,
        market_mode="perp",
    )
    decisions = _attach_identities(decisions, candidates)
    accepted = decisions.loc[decisions["accepted"].fillna(False).astype(bool)].copy()
    daily = _daily(accepted)
    metrics = compute_replay_metrics(candidates, decisions, equity, params=canonical_portfolio_params())

    out.mkdir(parents=True, exist_ok=False)
    coverage.to_parquet(out / "routed_outcome_coverage.parquet", index=False, compression="zstd")
    candidates.to_parquet(out / "portfolio_candidates.parquet", index=False, compression="zstd")
    decisions.to_parquet(out / "portfolio_decisions.parquet", index=False, compression="zstd")
    accepted.to_parquet(out / "accepted_trades.parquet", index=False, compression="zstd")
    equity.to_parquet(out / "portfolio_equity.parquet", index=False, compression="zstd")
    daily.to_parquet(out / "daily_portfolio_metrics.parquet", index=False, compression="zstd")
    _write_once(out / "run_manifest.json", {
        "schema": "strict_r3_p8u_exact_1m_constrained_portfolio_v1",
        "status": "complete",
        "scope": "offline chronological portfolio replay; no feature scoring, exchange IO, account reads, or order submission",
        "candidate_route": str(args.candidate_dir.resolve()),
        "candidate_manifest_sha256": _sha256(args.candidate_dir.resolve() / "candidate_manifest.json"),
        "exact_outcomes": str(args.outcome_dir.resolve()),
        "exact_outcome_manifest_sha256": _sha256(args.outcome_dir.resolve() / "run_manifest.json"),
        "route_candidates_target_free": int(len(route)),
        "label_complete_candidates_after_route": int(len(candidates)),
        "excluded_invalid_outcomes_after_route": int(len(route) - len(candidates)),
        "portfolio_accepted_trades": int(len(accepted)),
        "portfolio_contract": {
            "max_concurrent_positions": 8,
            "max_new_entries_per_decision": 2,
            "margin_budget_pct": 0.80,
            "margin_slot_pct": 0.10,
            "leverage": 7.0,
            "priority": "BCF MC1 expected policy-net bps only",
        },
        "outcome_handling": "invalid exact paths excluded after target-free routing; no pseudo-trades or capacity reservation",
        "metrics": metrics,
        "route_selection": route_manifest["selection"],
        "exact_outcome_live_state_parity": outcome_manifest.get("oracle_equivalence"),
        "code_sha256": _sha256(Path(__file__).resolve()),
    })
    print(out)


if __name__ == "__main__":
    main()
