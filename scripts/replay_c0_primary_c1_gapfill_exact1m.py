#!/usr/bin/env python3
"""Exact-1m C0-primary/C1-gap-fill constrained portfolio replay.

This is a deliberately bounded C0/C1 combination test.  C0 owns every
timestamp where it had at least one *target-free* dual-MC1 admission.  C1-LVA
may supply candidates only at timestamps with no C0 admission.  Thus C1 can
extend recall without replacing a C0 candidate or adding a second per-strategy
capacity channel.  Exact one-minute outcomes are attached only after the route
is formed, using the immutable outcome panel from the matched C0/C1 replay.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_causal_sr_c1_exact1m_parent_portfolio_replay import (  # noqa: E402
    _direct_dual_target_free_route,
    _monthly,
    _portfolio,
)


SCHEMA = "c0_primary_c1_gapfill_exact1m_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _route_from_exact_candidates(candidates: pd.DataFrame) -> pd.DataFrame:
    required = {
        "candidate_id", "decision_timestamp", "timestamp", "symbol",
        "side", "portfolio_priority_adjustment",
    }
    missing = sorted(required.difference(candidates.columns))
    if missing:
        raise AssertionError(f"exact candidate panel lacks {missing}")
    route = pd.DataFrame({
        "candidate_id": candidates["candidate_id"].astype(str),
        "timestamp": pd.to_datetime(candidates["decision_timestamp"], utc=True, errors="raise"),
        "entry_ts": pd.to_datetime(candidates["timestamp"], utc=True, errors="raise"),
        "symbol": candidates["symbol"].astype(str),
        "side_name": candidates["side"].astype(str).str.lower(),
        "auction_priority_bps": pd.to_numeric(
            candidates["portfolio_priority_adjustment"], errors="raise"
        ),
    })
    route["bcf_mc1_expected_bps"] = route["auction_priority_bps"]
    route["current_mc1_expected_bps"] = route["auction_priority_bps"]
    route["dual_admitted"] = True
    if route["candidate_id"].duplicated().any():
        raise AssertionError("exact candidate panel duplicates candidate identity")
    return route


def _resolve_exact_candidate_panel(root: Path, *, arm: str) -> Path:
    """Resolve an immutable per-arm candidate panel without guessing data.

    The earlier C1 producer used ``C1_LVA_refit_core_plus_causal_sr`` while
    the direct target-free producer names the same declared arm
    ``C1_refit_core_plus_causal_sr``.  Accept only these two explicit,
    versioned names and reject ambiguity; the route itself remains governed by
    the supplied target-free score panel.
    """
    names = {
        "C0": ("C0_refit_core_postfeb",),
        "C1": (
            "C1_LVA_refit_core_plus_causal_sr",
            "C1_refit_core_plus_causal_sr",
        ),
    }
    try:
        candidates = [
            root / f"{name}_portfolio_candidates.parquet"
            for name in names[str(arm)]
        ]
    except KeyError as exc:
        raise ValueError(f"unsupported exact candidate arm: {arm}") from exc
    found = [path for path in candidates if path.is_file()]
    if len(found) != 1:
        raise FileNotFoundError(
            f"expected exactly one {arm} exact candidate panel; found "
            f"{[str(path) for path in found]}"
        )
    return found[0]


def select_c0_primary_c1_gapfill(
    *,
    c0_target_free: pd.DataFrame,
    c1_target_free: pd.DataFrame,
    c0_exact_candidates: pd.DataFrame,
    c1_exact_candidates: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Return a route with C1 permitted only at C0-empty decision timestamps."""
    c0_times = set(pd.to_datetime(c0_target_free["timestamp"], utc=True, errors="raise"))
    c0 = _route_from_exact_candidates(c0_exact_candidates)
    if not set(c0["candidate_id"]).issubset(set(c0_target_free["candidate_id"])):
        raise AssertionError("C0 exact candidates are not a subset of target-free C0 admissions")
    # Validate the invariant which defines this experiment before inspecting the
    # optional C1 gap-fill source.  It keeps a malformed C0 route fail-closed
    # even if there are no C1 candidates at all.
    c1 = _route_from_exact_candidates(c1_exact_candidates)
    if not set(c1["candidate_id"]).issubset(set(c1_target_free["candidate_id"])):
        raise AssertionError("C1 exact candidates are not a subset of target-free C1 admissions")
    c1_fill = c1.loc[~c1["timestamp"].isin(c0_times)].copy()
    route = pd.concat([c0, c1_fill], ignore_index=True).sort_values(
        ["timestamp", "candidate_id"], kind="stable"
    ).reset_index(drop=True)
    if route["candidate_id"].duplicated().any():
        raise AssertionError("hybrid route duplicates candidate identity")
    if not c1_fill.empty and c1_fill["timestamp"].isin(c0_times).any():
        raise AssertionError("C1 displaced a C0-admitted timestamp")
    return route, {
        "c0_target_free_admitted_rows": int(len(c0_target_free)),
        "c1_target_free_admitted_rows": int(len(c1_target_free)),
        "c0_target_free_admitted_timestamps": int(len(c0_times)),
        "c0_exact_evaluable_rows": int(len(c0)),
        "c1_exact_evaluable_rows": int(len(c1)),
        "c1_gapfill_evaluable_rows": int(len(c1_fill)),
        "c1_gapfill_timestamps": int(c1_fill["timestamp"].nunique()),
        "hybrid_exact_evaluable_rows": int(len(route)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exact-root", type=Path, required=True)
    parser.add_argument("--c0-target-free-panel", type=Path, required=True)
    parser.add_argument("--c1-target-free-panel", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--admission-threshold-bps", type=float, default=50.0)
    parser.add_argument(
        "--c1-gapfill-dual-floor-bps", type=float, default=None,
        help="Optional stricter dual-MC1 floor for C1-only gap fill; C0 stays at --admission-threshold-bps.",
    )
    args = parser.parse_args()
    root = args.exact_root.resolve()
    out = args.out.resolve()
    if out.exists():
        raise FileExistsError("output must be immutable")
    if args.admission_threshold_bps <= 0.0:
        raise ValueError("admission threshold must be positive")
    c1_floor = float(args.c1_gapfill_dual_floor_bps or args.admission_threshold_bps)
    if c1_floor < float(args.admission_threshold_bps):
        raise ValueError("C1 gap-fill floor may not be below the C0 admission threshold")

    c0_target_free = _direct_dual_target_free_route(
        args.c0_target_free_panel.resolve(),
        admission_threshold_bps=float(args.admission_threshold_bps),
    )
    c1_target_free = _direct_dual_target_free_route(
        args.c1_target_free_panel.resolve(),
        admission_threshold_bps=c1_floor,
    )
    c0_candidates_path = _resolve_exact_candidate_panel(root, arm="C0")
    c1_candidates_path = _resolve_exact_candidate_panel(root, arm="C1")
    outcomes_path = root / "exact_1m_rich_parent_outcomes.parquet"
    route, audit = select_c0_primary_c1_gapfill(
        c0_target_free=c0_target_free,
        c1_target_free=c1_target_free,
        c0_exact_candidates=pd.read_parquet(c0_candidates_path),
        c1_exact_candidates=pd.read_parquet(c1_candidates_path).loc[
            lambda table: table["candidate_id"].astype(str).isin(
                set(c1_target_free["candidate_id"].astype(str))
            )
        ].copy(),
    )
    outcomes = pd.read_parquet(outcomes_path)
    valid_ids = set(outcomes["candidate_id"].astype(str))
    valid_route = route.loc[route["candidate_id"].isin(valid_ids)].copy()
    if len(valid_route) != len(route):
        raise AssertionError("exact candidate source and outcome panel identity mismatch")

    candidates, decisions, accepted, equity, metrics = _portfolio(
        valid_route, outcomes, "C0_primary_C1_gapfill"
    )
    metrics.update(audit)
    metrics["arm"] = "C0_primary_C1_gapfill"
    summary_path = root / "portfolio_summary.parquet"
    base_summary = pd.read_parquet(summary_path).copy()
    baseline = base_summary.loc[
        base_summary["arm"].astype(str).eq("C0_refit_core_postfeb")
    ]
    if len(baseline) != 1:
        raise AssertionError("exact root lacks one C0 reference row")
    reference = baseline.iloc[0]
    for field in ("portfolio_accepted", "net_bps_per_trade", "total_net_bps", "max_drawdown", "sortino", "worst_week", "compounded_return"):
        metrics[f"delta_vs_C0_{field}"] = float(metrics[field]) - float(reference[field])
    summary = pd.concat([base_summary, pd.DataFrame([metrics])], ignore_index=True, sort=False)
    out.mkdir(parents=True, exist_ok=False)
    route.to_parquet(out / "target_free_route_evaluable_after_source_join.parquet", index=False, compression="zstd")
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
            "C0 owns every timestamp with a target-free dual-MC1 admission; "
            "C1 may fill only a timestamp where C0 has no target-free admission"
        ),
        "admission_threshold_bps": float(args.admission_threshold_bps),
        "c1_gapfill_dual_floor_bps": c1_floor,
        "portfolio": "normal global chronological constrained auction; one strategy identity prevents extra timestamp capacity",
        "outcome": "reused immutable exact-1m rich-parent outcomes: decision+5m entry, 100 bps once",
        "inputs": {
            "exact_root_manifest": _sha256(root / "run_manifest.json"),
            "c0_target_free_panel": _sha256(args.c0_target_free_panel.resolve()),
            "c1_target_free_panel": _sha256(args.c1_target_free_panel.resolve()),
            "c0_exact_candidates": _sha256(c0_candidates_path),
            "c1_exact_candidates": _sha256(c1_candidates_path),
            "outcomes": _sha256(outcomes_path),
        },
        "audit": audit,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
