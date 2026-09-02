#!/usr/bin/env python3
"""Isolated strict-prequential router-consensus fold for an external OOF label.

The external value is an already causal, strict-OOF expected-policy projection
from the repaired path/regime studies.  This worker joins it only to the
training policy sidecar.  It never reaches a target-free held score panel.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import run_strict_r3_router_downstream as downstream  # noqa: E402


EXPECTED = tuple(pd.date_range("2026-01-01", "2026-07-01", freq="MS", tz="UTC"))
EDGES = (-100.0, -30.0, 30.0, 90.0)


@dataclass(frozen=True)
class ExternalLabelSpec:
    name: str
    column: str
    description: str
    source: str = "strict_oof_external_expected_policy_bps"
    edges_bps: tuple[float, ...] = EDGES
    objective: str = "ordinal_lambdarank"
    clip_abs_bps: float | None = 500.0

    def values(self, frame: pd.DataFrame) -> np.ndarray:
        value = pd.to_numeric(frame[self.column], errors="coerce").to_numpy(float)
        if self.clip_abs_bps is not None:
            value = np.clip(value, -float(self.clip_abs_bps), float(self.clip_abs_bps))
        return value


def _month(value: str) -> pd.Timestamp:
    result = pd.Timestamp(f"{value}-01", tz="UTC")
    if result not in EXPECTED:
        raise ValueError(f"unsupported fold {value}; expected Jan--Jul 2026")
    return result


def _external_policy(policy_path: Path, labels_path: Path, column: str) -> pd.DataFrame:
    policy = downstream._load_policy(policy_path)
    labels = pd.read_parquet(labels_path, columns=["candidate_id", "__decision_ts__", column]).copy()
    labels["__decision_ts__"] = pd.to_datetime(labels["__decision_ts__"], utc=True, errors="raise")
    if labels["candidate_id"].duplicated().any():
        raise AssertionError("external-label sidecar duplicates candidate IDs")
    if not np.isfinite(pd.to_numeric(labels[column], errors="coerce")).all():
        raise AssertionError(f"external-label sidecar {column} is non-finite")
    result = policy.merge(labels.drop(columns="__decision_ts__"), on="candidate_id", how="inner", validate="one_to_one")
    if result.empty:
        raise AssertionError("external labels have no policy identity overlap")
    return result


def _base_fields(source_root: Path) -> tuple[str, ...]:
    probe = source_root / "target_free_monthly" / "month=2025-11" / "scores_features.parquet"
    return downstream._source_base_fields(probe)


def score(
    *, source_root: Path, policy_path: Path, labels_path: Path, column: str,
    out: Path, month: pd.Timestamp, n_jobs: int, target_free_root: Path | None = None,
) -> None:
    out.mkdir(parents=True, exist_ok=True)
    # The raw enhanced-base source owns the immutable 120-field contract,
    # whereas a P3-routed source owns the target-free candidate population.
    # Keeping these roots explicit prevents a router source from being
    # reinterpreted as a base model or vice versa.
    score_source = target_free_root if target_free_root is not None else source_root
    target_free = score_source / "target_free_monthly"
    if not target_free.exists():
        raise FileNotFoundError(target_free)
    policy = _external_policy(policy_path, labels_path, column)
    policy = downstream._restrict_policy_to_source(policy, target_free)
    spec = ExternalLabelSpec(
        name=f"external_{column}", column=column,
        description="strict-OOF external expected-policy-bps projection; target-only router-consensus supervision",
    )
    previous = downstream.ALL_SCORE_MONTHS
    try:
        downstream.ALL_SCORE_MONTHS = (month,)
        audit = downstream._score_router_folds(target_free, policy, _base_fields(source_root), spec, out, n_jobs)
    finally:
        downstream.ALL_SCORE_MONTHS = previous
    receipt = out / "fold_audits"
    receipt.mkdir(exist_ok=True)
    audit.to_parquet(receipt / f"month={month:%Y-%m}.parquet", index=False, compression="zstd")


def finalize(*, out: Path) -> None:
    parts = []
    for month in EXPECTED:
        path = out / "fold_audits" / f"month={month:%Y-%m}.parquet"
        if not path.exists():
            raise FileNotFoundError(path)
        parts.append(pd.read_parquet(path))
    audit = pd.concat(parts, ignore_index=True)
    if audit["month"].tolist() != [f"{item:%Y-%m}" for item in EXPECTED]:
        raise AssertionError("external label fold receipts are not Jan--Jul exactly")
    audit.to_parquet(out / "consensus_fold_audit.parquet", index=False, compression="zstd")
    (out / "fold_finalization.json").write_text(json.dumps({
        "schema": "strict_r3_router_external_label_fold_v1",
        "folds": audit["month"].tolist(),
        "target_free_scores": "one isolated external-label fold per process",
        "head_contract": list(downstream.RETAINED_HEADS),
    }, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--policy-path", type=Path, required=True)
    parser.add_argument("--labels-path", type=Path, required=True)
    parser.add_argument("--column", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--target-free-root", type=Path,
        help="sealed router-only candidate source; defaults to --source-root",
    )
    parser.add_argument("--month")
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--finalize", action="store_true")
    args = parser.parse_args()
    if args.finalize:
        if args.month:
            parser.error("--finalize does not take --month")
        finalize(out=args.out)
    else:
        if not args.month:
            parser.error("--month is required unless --finalize is supplied")
        score(
            source_root=args.source_root.resolve(), policy_path=args.policy_path.resolve(),
            labels_path=args.labels_path.resolve(), column=str(args.column), out=args.out.resolve(),
            month=_month(args.month), n_jobs=int(args.n_jobs),
            target_free_root=args.target_free_root.resolve() if args.target_free_root else None,
        )


if __name__ == "__main__":
    main()
