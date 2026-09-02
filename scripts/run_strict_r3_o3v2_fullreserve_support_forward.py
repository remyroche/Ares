#!/usr/bin/env python3
"""Freeze full-reserve support choices, then score a later forward block.

This is deliberately a narrow research orchestrator.  It first selects one
support/weight contract per already-selected O3-v2 target using only a
declared development block.  It then refits and scores *only* the supplied
later months with those fixed pairs.  No policy outcome is read while a held
score receipt is written; MC1, admission, portfolio, canonical, and live
artifacts are intentionally outside this script's scope.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import run_strict_r3_o3v2_support_funnel as funnel  # noqa: E402
import select_strict_r3_o3v2_support as selector  # noqa: E402


SCHEMA = "strict_r3_o3v2_fullreserve_support_forward_v1"


def _months(raw: str) -> tuple[str, ...]:
    values = tuple(value.strip() for value in raw.split(",") if value.strip())
    if not values:
        raise ValueError("at least one YYYY-MM month is required")
    return values


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _exclusive_json(path: Path, payload: object) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(fd, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def _selected_pairs(path: Path) -> tuple[tuple[str, str], ...]:
    contract = json.loads(path.read_text())
    selected = contract.get("selected", ())
    pairs = tuple((str(row["target_arm"]), str(row["support_arm"])) for row in selected)
    if not pairs:
        raise AssertionError("support selection produced no retained contracts")
    if len(set(pairs)) != len(pairs):
        raise AssertionError("support selection contains duplicate target/support pairs")
    if len({target for target, _support in pairs}) != len(pairs):
        raise AssertionError("support selection must retain at most one support contract per target")
    return pairs


def run(
    *,
    target_metrics: Path,
    development_support_metrics: tuple[Path, ...],
    selection_out: Path,
    feature_root: Path,
    semantic_root: Path,
    policy_path: Path,
    bundle_root: Path,
    forward_out: Path,
    development_months: tuple[str, ...],
    forward_months: tuple[str, ...],
    query_mode: str,
    n_jobs: int = 1,
) -> None:
    """Select using development only and produce strictly later score receipts."""
    if selection_out.exists():
        selection_path = selection_out / "selected_support_contracts.json"
        if not selection_path.exists():
            raise AssertionError(f"incomplete immutable selection artifact: {selection_out}")
    else:
        selector.run(
            target_metrics=target_metrics,
            support_metrics=development_support_metrics,
            out=selection_out,
            months=development_months,
        )
        selection_path = selection_out / "selected_support_contracts.json"

    pairs = _selected_pairs(selection_path)
    latest_development = max(pd.Timestamp(f"{month}-01", tz="UTC") for month in development_months)
    first_forward = min(pd.Timestamp(f"{month}-01", tz="UTC") for month in forward_months)
    if first_forward <= latest_development:
        raise AssertionError("forward months must begin after the development selection block")
    if forward_out.exists():
        raise FileExistsError(forward_out)

    # S0 is the already-existing uniform target-funnel receipt.  It is not a
    # weighting arm that can be re-fit by the support runner.  A weighted
    # winner, on the other hand, must be scored afresh over the frozen later
    # block.  Keeping the two source kinds explicit prevents accidental
    # substitution of a weighted score for the matching uniform control.
    weighted_pairs = tuple((target, support) for target, support in pairs if support != "S0_uniform")
    target_score_root = target_metrics.parent
    uniform_pairs = tuple((target, support) for target, support in pairs if support == "S0_uniform")
    for target, _support in uniform_pairs:
        for month in forward_months:
            source = target_score_root / "target_free_scores" / target / f"month={month}.parquet"
            if not source.exists():
                raise FileNotFoundError(f"missing frozen uniform target receipt: {source}")
    if weighted_pairs:
        funnel.run(
            feature_root=feature_root,
            semantic_root=semantic_root,
            policy_path=policy_path,
            bundle_root=bundle_root,
            out=forward_out,
            months=tuple(pd.Timestamp(f"{month}-01", tz="UTC") for month in forward_months),
            target_arms=tuple(dict.fromkeys(target for target, _support in weighted_pairs)),
            support_arms=tuple(dict.fromkeys(support for _target, support in weighted_pairs)),
            pairs=weighted_pairs,
            query_mode=query_mode,
            resume=False,
            n_jobs=n_jobs,
        )
    else:
        forward_out.mkdir(parents=True)
    receipt = {
        "schema": SCHEMA,
        "scope": "offline full-reserve support selection and later target-free scoring only; no MC1, admission, portfolio, canonical, or live mutation",
        "target_metrics": str(target_metrics),
        "target_metrics_sha256": _hash(target_metrics),
        "development_support_metrics": [str(path) for path in development_support_metrics],
        "development_support_metrics_sha256": {str(path): _hash(path) for path in development_support_metrics},
        "selection_contract": str(selection_path),
        "selection_contract_sha256": _hash(selection_path),
        "development_months": list(development_months),
        "forward_months": list(forward_months),
        "selected_pairs": [
            {"target_arm": target, "support_arm": support}
            for target, support in pairs
        ],
        "query_mode": query_mode,
        "lightgbm_n_jobs": n_jobs,
        "score_sources": {
            f"{target}__{support}": (
                str(target_score_root) if support == "S0_uniform" else str(forward_out)
            )
            for target, support in pairs
        },
        "causality": {
            "selection": "development months only; later forward metrics are never read by the selector",
            "fit": "six full preceding resolved calendar months before the 28-day reserve",
            "held_scores": "target-free receipts are persisted before policy outcomes are joined for metrics",
        },
    }
    _exclusive_json(forward_out / "selection_forward_receipt.json", receipt)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-metrics", type=Path, required=True)
    parser.add_argument("--development-support-metrics", action="append", type=Path, required=True)
    parser.add_argument("--selection-out", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--semantic-root", type=Path, required=True)
    parser.add_argument("--policy-path", type=Path, required=True)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--forward-out", type=Path, required=True)
    parser.add_argument("--development-months", default="2025-11,2025-12,2026-01")
    parser.add_argument("--forward-months", default="2026-02,2026-03,2026-04,2026-05,2026-06,2026-07")
    parser.add_argument(
        "--query-mode",
        choices=("exact_timestamp_side", "exact_timestamp_baseband_side", "cycle_4h_side"),
        default="cycle_4h_side",
    )
    parser.add_argument("--n-jobs", type=int, default=1, help="deterministic LightGBM worker count for frozen weighted forward fits")
    args = parser.parse_args()
    if args.n_jobs <= 0 or args.n_jobs > 8:
        parser.error("--n-jobs must be between 1 and 8")
    run(
        target_metrics=args.target_metrics,
        development_support_metrics=tuple(args.development_support_metrics),
        selection_out=args.selection_out,
        feature_root=args.feature_root,
        semantic_root=args.semantic_root,
        policy_path=args.policy_path,
        bundle_root=args.bundle_root,
        forward_out=args.forward_out,
        development_months=_months(args.development_months),
        forward_months=_months(args.forward_months),
        query_mode=args.query_mode,
        n_jobs=args.n_jobs,
    )


if __name__ == "__main__":
    main()
