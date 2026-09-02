#!/usr/bin/env python3
"""Replay the successor C0/C1 agreement route on exact one-minute outcomes.

This is a deliberately offline, no-order audit.  It reads the already sealed
target-free agreement route first, and attaches rich-policy outcomes only
afterwards.  Exact entry and exit timestamps are preserved, so concurrency is
not approximated by a 15-minute holding-bar proxy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_causal_sr_c1_exact1m_parent_portfolio_replay import (  # noqa: E402
    _monthly,
    _portfolio,
)


SCHEMA = "p8u_successor_c0_c1_exact1m_portfolio_v2"
POLICY_FIELDS = (
    "policy_path_valid", "policy_net_bps", "policy_gross_bps",
    "policy_entry_timestamp", "policy_exit_timestamp", "policy_entry_price",
    "policy_exit_price", "policy_exit_reason", "policy_label_available_ts",
    "policy_invalid_reason", "policy_cost_bps",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(
    mapper: Path, *, route_mode: str, admission_floor_bps: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    manifest = json.loads((mapper / "run_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise AssertionError("mapper bundle is incomplete")
    route = pd.read_parquet(mapper / "agreement_tier_target_free_predictions.parquet").copy()
    policy = pd.read_parquet(mapper / "agreement_tier_policy_replay.parquet").copy()
    forbidden = {col for col in route.columns if col.startswith("policy_") or "outcome" in col.lower()}
    if forbidden:
        raise AssertionError(f"target-free route carries outcome fields: {sorted(forbidden)}")
    required_route = {
        "candidate_id", "__decision_ts__", "__symbol__", "side_name",
        "portfolio_order_priority_bps", "auction_priority_bps", "agreement_tier",
    }
    required_policy = {"candidate_id", "__decision_ts__", "__symbol__", "side_name", *POLICY_FIELDS}
    if missing := required_route.difference(route.columns):
        raise KeyError(f"target-free route missing {sorted(missing)}")
    if missing := required_policy.difference(policy.columns):
        raise KeyError(f"policy replay missing {sorted(missing)}")
    keys = ["candidate_id", "__decision_ts__", "__symbol__", "side_name"]
    for name, frame in (("target-free route", route), ("policy replay", policy)):
        frame["candidate_id"] = frame["candidate_id"].astype(str)
        frame["__decision_ts__"] = pd.to_datetime(frame["__decision_ts__"], utc=True, errors="raise")
        if frame.duplicated(keys).any():
            raise AssertionError(f"{name} duplicates candidate-time identity")
    if set(map(tuple, route[keys].to_numpy())) != set(map(tuple, policy[keys].to_numpy())):
        raise AssertionError("target-free route and policy replay identities differ")
    if not route["side_name"].astype(str).str.lower().eq("long").all():
        raise AssertionError("successor replay must be long-only")
    route = route.sort_values(keys, kind="stable")
    if route_mode == "agreement":
        selected = route
    elif route_mode == "c0_only":
        selected = route.loc[route["c0_dual_mc1_admitted"].fillna(False).astype(bool)].copy()
        selected["portfolio_order_priority_bps"] = pd.to_numeric(
            selected["c0_bcf_mc1_expected_bps"], errors="raise"
        )
    elif route_mode == "c1_only":
        selected = route.loc[route["c1_dual_mc1_admitted"].fillna(False).astype(bool)].copy()
        selected["portfolio_order_priority_bps"] = pd.to_numeric(
            selected["c1_bcf_mc1_expected_bps"], errors="raise"
        )
    else:  # pragma: no cover - argparse also protects this branch
        raise ValueError(f"unknown route mode: {route_mode}")
    if route_mode != "agreement":
        expected = (
            pd.to_numeric(selected[f"{route_mode[:2]}_bcf_mc1_expected_bps"], errors="raise")
            if route_mode in {"c0_only", "c1_only"} else pd.Series(dtype=float)
        )
        if not expected.ge(float(admission_floor_bps)).all():
            raise AssertionError(f"{route_mode} contains a candidate below its stated admission floor")
    return selected.reset_index(drop=True), policy.loc[:, [*keys, *POLICY_FIELDS]], manifest


def _policy_join(route: pd.DataFrame, policy: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = ["candidate_id", "__decision_ts__", "__symbol__", "side_name"]
    joined = route.merge(policy, on=keys, how="left", validate="one_to_one")
    valid = (
        joined["policy_path_valid"].fillna(False).astype(bool)
        & np.isfinite(pd.to_numeric(joined["policy_net_bps"], errors="coerce"))
        & np.isfinite(pd.to_numeric(joined["policy_gross_bps"], errors="coerce"))
        & pd.to_datetime(joined["policy_entry_timestamp"], utc=True, errors="coerce").notna()
        & pd.to_datetime(joined["policy_exit_timestamp"], utc=True, errors="coerce").notna()
    )
    joined["policy_outcome_available"] = valid.astype(bool)
    selected = joined.loc[valid].copy()
    entry = pd.to_datetime(selected["policy_entry_timestamp"], utc=True, errors="raise")
    exit_ = pd.to_datetime(selected["policy_exit_timestamp"], utc=True, errors="raise")
    decision = pd.to_datetime(selected["__decision_ts__"], utc=True, errors="raise")
    if not entry.eq(decision + pd.Timedelta(minutes=5)).all():
        raise AssertionError("exact policy entry is not the declared decision-plus-five-minute open")
    if not exit_.ge(entry).all():
        raise AssertionError("exact policy exit precedes entry")
    if not np.allclose(
        pd.to_numeric(selected["policy_gross_bps"], errors="raise")
        - pd.to_numeric(selected["policy_net_bps"], errors="raise"),
        pd.to_numeric(selected["policy_cost_bps"], errors="raise"), rtol=0.0, atol=1e-8,
    ):
        raise AssertionError("exact rich-policy cost is not applied exactly once")
    resolution = pd.to_datetime(selected["policy_label_available_ts"], utc=True, errors="raise")
    if not resolution.eq(decision + pd.Timedelta(hours=12, minutes=5)).all():
        raise AssertionError("policy label availability violates the frozen H12 resolution contract")
    return joined, selected


def _route_for_portfolio(selected: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    decision = pd.to_datetime(selected["__decision_ts__"], utc=True, errors="raise")
    entry = pd.to_datetime(selected["policy_entry_timestamp"], utc=True, errors="raise")
    route = pd.DataFrame({
        "candidate_id": selected["candidate_id"].astype(str),
        "timestamp": decision,
        "entry_ts": entry,
        "symbol": selected["__symbol__"].astype(str),
        "side_name": "long",
        # This is deliberately the hard agreement ordering, not a new EV
        # estimate: both-admitted > C0-only > C1-only, raw mapped EV within tier.
        "auction_priority_bps": pd.to_numeric(selected["portfolio_order_priority_bps"], errors="raise"),
        "raw_mapped_expected_bps": pd.to_numeric(selected["auction_priority_bps"], errors="raise"),
        "agreement_tier": pd.to_numeric(selected["agreement_tier"], errors="raise").astype(int),
    })
    outcomes = pd.DataFrame({
        "candidate_id": route["candidate_id"], "timestamp": route["timestamp"],
        "entry_ts": route["entry_ts"], "symbol": route["symbol"],
        "exact_entry_price": pd.to_numeric(selected["policy_entry_price"], errors="raise"),
        "exact_gross_bps": pd.to_numeric(selected["policy_gross_bps"], errors="raise"),
        "exact_net_bps": pd.to_numeric(selected["policy_net_bps"], errors="raise"),
        "exact_exit_price": pd.to_numeric(selected["policy_exit_price"], errors="raise"),
        "exact_exit_ts": pd.to_datetime(selected["policy_exit_timestamp"], utc=True, errors="raise"),
        "exact_exit_minute": np.maximum(
            ((pd.to_datetime(selected["policy_exit_timestamp"], utc=True, errors="raise") - entry).dt.total_seconds() / 60.0).round().astype(int), 0,
        ),
        "exact_exit_reason": selected["policy_exit_reason"].astype(str),
    })
    return route, outcomes


def _daily(accepted: pd.DataFrame) -> pd.DataFrame:
    result = accepted.copy()
    result["day"] = pd.to_datetime(result["decision_timestamp"], utc=True).dt.strftime("%Y-%m-%d")
    return result.groupby("day", as_index=False, sort=True).agg(
        trades=("candidate_id", "size"),
        net_bps_per_trade=("net_bps", "mean"),
        total_net_bps=("net_bps", "sum"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mapper", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--route-mode", choices=("agreement", "c0_only", "c1_only"), default="agreement")
    parser.add_argument("--admission-floor-bps", type=float, default=50.0)
    args = parser.parse_args()
    mapper, out = args.mapper.resolve(), args.out.resolve()
    if out.exists():
        raise FileExistsError("output must be immutable")
    if args.admission_floor_bps <= 0.0:
        raise ValueError("admission floor must be positive")
    route, policy, mapper_manifest = _load(
        mapper, route_mode=str(args.route_mode), admission_floor_bps=float(args.admission_floor_bps),
    )
    joined, selected = _policy_join(route, policy)
    # A forward replay often contains target-free decisions whose H12 label is
    # simply not resolved yet.  Keep that distinct from a completed but
    # invalid/missing exact path: neither may reserve portfolio capacity, but
    # conflating them makes the source-quality audit misleading.
    path_state = joined["policy_path_valid"]
    resolved = pd.to_datetime(
        joined["policy_label_available_ts"], utc=True, errors="coerce"
    ).notna()
    invalid_path = path_state.eq(False)
    unresolved = path_state.isna() & ~resolved
    malformed_resolved = (~joined["policy_outcome_available"].astype(bool)) & resolved & ~invalid_path
    exact_route, outcomes = _route_for_portfolio(selected)
    candidates, decisions, accepted, equity, metrics = _portfolio(
        exact_route, outcomes, "p8u_successor_c0_c1_agreement_exact1m"
    )
    metrics.update({
        "target_free_agreement_admitted": int(len(route)),
        "exact_path_valid_after_target_free_route": int(len(selected)),
        "resolved_policy_labels_after_target_free_route": int(resolved.sum()),
        "invalid_exact_paths_excluded_after_target_free_route": int(invalid_path.sum()),
        "unresolved_policy_labels_excluded_after_target_free_route": int(unresolved.sum()),
        "malformed_resolved_policy_rows_excluded_after_target_free_route": int(malformed_resolved.sum()),
        "portfolio_accepted": int(len(accepted)),
        "net_bps_per_trade": float(accepted["net_bps"].mean()) if len(accepted) else float("nan"),
        "total_net_bps": float(accepted["net_bps"].sum()) if len(accepted) else 0.0,
    })
    out.mkdir(parents=True, exist_ok=False)
    route.to_parquet(out / "target_free_agreement_route.parquet", index=False, compression="zstd")
    joined.to_parquet(out / "target_free_route_with_posthoc_policy_coverage.parquet", index=False, compression="zstd")
    candidates.to_parquet(out / "portfolio_candidates.parquet", index=False, compression="zstd")
    decisions.to_parquet(out / "portfolio_decisions.parquet", index=False, compression="zstd")
    accepted.to_parquet(out / "portfolio_accepted.parquet", index=False, compression="zstd")
    equity.to_parquet(out / "portfolio_equity.parquet", index=False, compression="zstd")
    _monthly(accepted).to_parquet(out / "monthly_metrics.parquet", index=False, compression="zstd")
    _daily(accepted).to_parquet(out / "daily_metrics.parquet", index=False, compression="zstd")
    (out / "run_manifest.json").write_text(json.dumps({
        "schema": SCHEMA, "status": "complete",
        "scope": "offline no-order exact-one-minute chronological constrained portfolio replay",
        "mapper": {"path": str(mapper), "manifest_sha256": _sha256(mapper / "run_manifest.json")},
        "selection": (
            "target-free C0/C1 hard agreement route: both-admitted -> C0-only -> C1-only"
            if args.route_mode == "agreement" else f"target-free {args.route_mode} MC1 admission"
        ),
        "route_mode": str(args.route_mode), "admission_floor_bps": float(args.admission_floor_bps),
        "priority": (
            "hard tier then raw mapped expected EV; no outcome-derived ordering"
            if args.route_mode == "agreement" else "selected mapper's raw expected EV; no outcome-derived ordering"
        ),
        "policy": "observed 1m decision+5m entry, exact exit timestamp, 100 bps cost exactly once",
        "outcome_handling": "invalid paths and not-yet-resolved labels are excluded only after the target-free route; neither influences selection or reserve capacity",
        "portfolio": "canonical 7x, eight concurrent, two new entries per decision, 80% margin budget, 10% slots",
        "metrics": metrics,
        "input_sha256": {
            "target_free_route": _sha256(mapper / "agreement_tier_target_free_predictions.parquet"),
            "policy_replay": _sha256(mapper / "agreement_tier_policy_replay.parquet"),
        },
    }, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
