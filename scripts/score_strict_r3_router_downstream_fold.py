#!/usr/bin/env python3
"""Bounded-memory single-fold worker for the economic-router downstream replay.

``run_strict_r3_router_downstream.py`` remains the canonical orchestration
entry point.  This helper is only its operationally equivalent fold executor:
one Python process fits and writes exactly one target-free consensus fold, then
exits.  That bounds native LightGBM memory across long historical ledgers.

It is useful when a host cannot safely retain several sequential native model
fits in one process.  No labels are written to score panels, and finalisation
merely concatenates immutable per-fold audit receipts.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import run_strict_r3_router_downstream as downstream  # noqa: E402


EXPECTED = tuple(pd.date_range("2026-01-01", "2026-07-01", freq="MS", tz="UTC"))


def _month(token: str) -> pd.Timestamp:
    value = pd.Timestamp(f"{token}-01", tz="UTC")
    if value not in EXPECTED:
        raise ValueError(f"unsupported fold month {token}; expected Jan--Jul 2026")
    return value


def _base_fields(source_root: Path) -> tuple[str, ...]:
    path = source_root / "target_free_monthly" / "month=2025-11" / "scores_features.parquet"
    return downstream._source_base_fields(path)


def score(
    *, source_root: Path, policy_path: Path, out: Path, month: pd.Timestamp,
    n_jobs: int, label: str, target_free_root: Path | None = None,
) -> None:
    """Fit one fold, optionally reading a separately sealed target-free root."""
    out.mkdir(parents=True, exist_ok=True)
    source = target_free_root if target_free_root is not None else out
    target_free = source / "target_free_monthly"
    if not target_free.exists():
        raise FileNotFoundError(f"missing materialised target-free root: {target_free}")
    policy = downstream._restrict_policy_to_source(downstream._load_policy(policy_path), target_free)
    previous = downstream.ALL_SCORE_MONTHS
    try:
        downstream.ALL_SCORE_MONTHS = (month,)
        audit = downstream._score_router_folds(
            target_free, policy, _base_fields(source_root),
            downstream.parent.POLICY_CONVERSION_LABEL_SPECS[label], out, n_jobs,
        )
    finally:
        downstream.ALL_SCORE_MONTHS = previous
    receipt = out / "fold_audits"
    receipt.mkdir(exist_ok=True)
    audit.to_parquet(receipt / f"month={month:%Y-%m}.parquet", index=False, compression="zstd")


def finalize(*, out: Path) -> None:
    root = out / "fold_audits"
    parts = []
    for month in EXPECTED:
        path = root / f"month={month:%Y-%m}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"missing completed fold receipt: {path}")
        parts.append(pd.read_parquet(path))
    audit = pd.concat(parts, ignore_index=True)
    if audit["month"].tolist() != [f"{month:%Y-%m}" for month in EXPECTED]:
        raise AssertionError("fold audit months are not exactly January--July 2026")
    if audit["held_rows"].le(0).any():
        raise AssertionError("empty held fold")
    audit.to_parquet(out / "consensus_fold_audit.parquet", index=False, compression="zstd")
    (out / "fold_finalization.json").write_text(json.dumps({
        "schema": "strict_r3_economic_router_downstream_fold_v1",
        "folds": audit["month"].tolist(), "target_free_scores": "one isolated process per fold",
        "head_contract": list(downstream.RETAINED_HEADS),
    }, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--policy-path", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--target-free-root", type=Path,
        help="sealed parent of target_free_monthly; defaults to --out",
    )
    parser.add_argument("--month")
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--policy-label", default="direct_policy_economic_200_0_50_150", choices=tuple(downstream.parent.POLICY_CONVERSION_LABEL_SPECS))
    parser.add_argument("--finalize", action="store_true")
    args = parser.parse_args()
    if args.n_jobs < 1:
        parser.error("--n-jobs must be positive")
    if args.finalize:
        if args.month:
            parser.error("--finalize does not take --month")
        finalize(out=args.out)
    else:
        if not args.month:
            parser.error("--month is required unless --finalize is supplied")
        score(source_root=args.source_root, policy_path=args.policy_path, out=args.out,
              month=_month(args.month), n_jobs=args.n_jobs, label=args.policy_label,
              target_free_root=args.target_free_root)


if __name__ == "__main__":
    main()
