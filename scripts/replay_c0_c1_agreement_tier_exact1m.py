#!/usr/bin/env python3
"""Causal C0/C1 agreement-tiered exact-one-minute portfolio replay.

This offline test keeps the normal global portfolio auction and exact one-minute
rich-parent outcomes.  It changes only candidate order, using target-free
admission agreement:

  tier 2: admitted by both C0 and C1;
  unpaired: admitted by C1-only or C0-only and ordered by raw BCF EV.

Within each tier, the respective BCF-MC1 mapped EV remains the priority.  C0
provides the priority for the both-admitted tier.  The tiers are formed before
any path/outcome access.  This is a research ablation, not live authority.
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

from scripts.replay_c0_primary_c1_gapfill_exact1m import _route_from_exact_candidates  # noqa: E402
from scripts.run_causal_sr_c1_exact1m_parent_portfolio_replay import (  # noqa: E402
    _direct_dual_target_free_route,
    _monthly,
    _portfolio,
)


SCHEMA = "c0_c1_agreement_tier_exact1m_v1"
TIER_OFFSET_BPS = 10_000.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def select_c0_c1_agreement_tiers(
    *,
    c0_target_free: pd.DataFrame,
    c1_target_free: pd.DataFrame,
    c0_exact_candidates: pd.DataFrame,
    c1_exact_candidates: pd.DataFrame,
    unpaired_order: str = "c0_then_c1",
) -> tuple[pd.DataFrame, dict[str, int | float]]:
    """Return a valid exact route ordered by causal C0/C1 agreement tiers."""
    c0 = _route_from_exact_candidates(c0_exact_candidates)
    c1 = _route_from_exact_candidates(c1_exact_candidates)
    c0_ids = set(c0_target_free["candidate_id"].astype(str))
    c1_ids = set(c1_target_free["candidate_id"].astype(str))
    if not set(c0["candidate_id"]).issubset(c0_ids):
        raise AssertionError("C0 exact candidates are not a subset of target-free C0 admissions")
    if not set(c1["candidate_id"]).issubset(c1_ids):
        raise AssertionError("C1 exact candidates are not a subset of target-free C1 admissions")

    # A candidate that both arms admit has one common exact outcome.  Preserve
    # the C0 score coordinate for it, matching the reported both-admitted C0
    # selection cohort.  C1 supplies the coordinate only in its own gap.
    c0["route_source"] = "C0"
    c1["route_source"] = "C1"
    route = pd.concat([c0, c1.loc[~c1["candidate_id"].isin(set(c0["candidate_id"]))]], ignore_index=True)
    if route["candidate_id"].duplicated().any():
        raise AssertionError("agreement tier route duplicates candidate identity")
    if unpaired_order not in {"highest_raw_bcf", "c0_then_c1", "c1_then_c0"}:
        raise ValueError(f"unknown unpaired C0/C1 ordering: {unpaired_order}")
    route["agreement_tier"] = 0
    route.loc[route["candidate_id"].isin(c1_ids), "agreement_tier"] = 1
    route.loc[route["candidate_id"].isin(c0_ids.intersection(c1_ids)), "agreement_tier"] = 2
    if not route["candidate_id"].isin(c0_ids.union(c1_ids)).all():
        raise AssertionError("agreement tier route contains an unadmitted candidate")
    raw_priority = pd.to_numeric(route["auction_priority_bps"], errors="raise")
    if float(raw_priority.abs().max()) >= TIER_OFFSET_BPS:
        raise AssertionError("tier offset is not greater than every mapped-EV priority")
    route["raw_bcf_mc1_priority_bps"] = raw_priority
    if unpaired_order == "highest_raw_bcf":
        route["portfolio_tier"] = np.where(route["agreement_tier"].eq(2), 1, 0)
    elif unpaired_order == "c0_then_c1":
        route["portfolio_tier"] = np.select(
            [route["agreement_tier"].eq(2), route["agreement_tier"].eq(0)],
            [2, 1], default=0,
        )
    else:
        route["portfolio_tier"] = route["agreement_tier"]
    route["auction_priority_bps"] = raw_priority + TIER_OFFSET_BPS * route["portfolio_tier"]
    route["unpaired_order"] = unpaired_order
    route = route.sort_values(["timestamp", "portfolio_tier", "raw_bcf_mc1_priority_bps", "candidate_id"], ascending=[True, False, False, True], kind="stable").reset_index(drop=True)
    counts = route["agreement_tier"].value_counts().to_dict()
    return route, {
        "target_free_c0_admitted_rows": int(len(c0_target_free)),
        "target_free_c1_admitted_rows": int(len(c1_target_free)),
        "exact_c0_evaluable_rows": int(len(c0)),
        "exact_c1_evaluable_rows": int(len(c1)),
        "tier_both_evaluable_rows": int(counts.get(2, 0)),
        "tier_c1_only_evaluable_rows": int(counts.get(1, 0)),
        "tier_c0_only_evaluable_rows": int(counts.get(0, 0)),
        "tier_offset_bps": float(TIER_OFFSET_BPS),
        "unpaired_order": unpaired_order,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exact-root", type=Path, required=True)
    parser.add_argument("--c0-target-free-panel", type=Path, required=True)
    parser.add_argument("--c1-target-free-panel", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--admission-threshold-bps", type=float, default=50.0)
    parser.add_argument("--unpaired-order", choices=("highest_raw_bcf", "c0_then_c1", "c1_then_c0"), default="c0_then_c1")
    args = parser.parse_args()
    root = args.exact_root.resolve()
    out = args.out.resolve()
    if out.exists():
        raise FileExistsError("output must be immutable")
    if args.admission_threshold_bps <= 0.0:
        raise ValueError("admission threshold must be positive")

    c0_panel = args.c0_target_free_panel.resolve()
    c1_panel = args.c1_target_free_panel.resolve()
    c0_target_free = _direct_dual_target_free_route(c0_panel, admission_threshold_bps=float(args.admission_threshold_bps))
    c1_target_free = _direct_dual_target_free_route(c1_panel, admission_threshold_bps=float(args.admission_threshold_bps))
    c0_path = root / "C0_refit_core_postfeb_portfolio_candidates.parquet"
    c1_path = root / "C1_LVA_refit_core_plus_causal_sr_portfolio_candidates.parquet"
    outcomes_path = root / "exact_1m_rich_parent_outcomes.parquet"
    route, audit = select_c0_c1_agreement_tiers(
        c0_target_free=c0_target_free,
        c1_target_free=c1_target_free,
        c0_exact_candidates=pd.read_parquet(c0_path),
        c1_exact_candidates=pd.read_parquet(c1_path),
        unpaired_order=str(args.unpaired_order),
    )
    outcomes = pd.read_parquet(outcomes_path)
    valid_ids = set(outcomes["candidate_id"].astype(str))
    if not set(route["candidate_id"]).issubset(valid_ids):
        raise AssertionError("exact route contains a candidate without a valid exact outcome")
    candidates, decisions, accepted, equity, metrics = _portfolio(route, outcomes, "C0_C1_agreement_tier")
    metrics.update(audit)
    metrics["arm"] = "C0_C1_agreement_tier"
    base_summary = pd.read_parquet(root / "portfolio_summary.parquet")
    baseline = base_summary.loc[base_summary["arm"].astype(str).eq("C0_refit_core_postfeb")]
    if len(baseline) != 1:
        raise AssertionError("exact root lacks exactly one C0 reference row")
    ref = baseline.iloc[0]
    for field in ("portfolio_accepted", "net_bps_per_trade", "total_net_bps", "max_drawdown", "sortino", "worst_week", "compounded_return"):
        metrics[f"delta_vs_C0_{field}"] = float(metrics[field]) - float(ref[field])
    summary = pd.concat([base_summary, pd.DataFrame([metrics])], ignore_index=True, sort=False)
    out.mkdir(parents=True, exist_ok=False)
    route.to_parquet(out / "target_free_agreement_tier_route.parquet", index=False, compression="zstd")
    candidates.to_parquet(out / "portfolio_candidates.parquet", index=False, compression="zstd")
    decisions.to_parquet(out / "portfolio_decisions.parquet", index=False, compression="zstd")
    accepted.to_parquet(out / "portfolio_accepted.parquet", index=False, compression="zstd")
    equity.to_parquet(out / "portfolio_equity.parquet", index=False, compression="zstd")
    _monthly(accepted).to_parquet(out / "monthly_metrics.parquet", index=False, compression="zstd")
    summary.to_parquet(out / "portfolio_summary.parquet", index=False, compression="zstd")
    (out / "run_manifest.json").write_text(json.dumps({
        "schema": SCHEMA,
        "scope": "offline research only; no refit, no exchange I/O, no order authority",
        "selection": (
            "both-admitted first; C0-only next; C1-only only as gap fill"
            if str(args.unpaired_order) == "c0_then_c1"
            else "both-admitted first; C1-only and C0-only then compete by selected raw BCF EV; C0 BCF EV within both/C0 and C1 BCF EV within C1"
        ),
        "tier_offset_bps": TIER_OFFSET_BPS,
        "admission_threshold_bps": float(args.admission_threshold_bps),
        "portfolio": "normal global chronological constrained auction; no extra strategy capacity",
        "outcome": "reused immutable exact-1m rich-parent outcomes: decision+5m entry, 100 bps once",
        "inputs": {
            "exact_root_manifest": _sha256(root / "run_manifest.json"),
            "c0_target_free_panel": _sha256(c0_panel),
            "c1_target_free_panel": _sha256(c1_panel),
            "c0_exact_candidates": _sha256(c0_path),
            "c1_exact_candidates": _sha256(c1_path),
            "outcomes": _sha256(outcomes_path),
        },
        "audit": audit,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
