#!/usr/bin/env python3
"""Build an immutable target-free union of disjoint P8U Router OOS receipts.

This is a lineage adapter for offline research only.  It selects exactly one
pre-existing Router score receipt per requested calendar month, validates that
the score frame is target-free, and writes a self-contained month partition.
It never refits a model, joins an outcome, or changes any source artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


IDENTITY = ("candidate_id", "__decision_ts__", "side_name")
REQUIRED = (*IDENTITY, "router_primary_rank")
PROHIBITED_TOKENS = ("policy_", "label_", "target_", "outcome_", "path_")
SCHEMA = "strict_r3_p8u_router_targetfree_score_union_v1"


def _sha256(paths: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(str(path).encode("utf-8"))
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
    return digest.hexdigest()


def _months(text: str) -> tuple[pd.Timestamp, ...]:
    values = tuple(pd.Timestamp(f"{token.strip()}-01", tz="UTC") for token in text.split(",") if token.strip())
    if not values or tuple(sorted(values)) != values or len(set(values)) != len(values):
        raise ValueError("--months must be non-empty, unique, and chronological YYYY-MM values")
    return values


def _write_json_exclusive(path: Path, payload: object) -> None:
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)


def _source_for_month(roots: tuple[Path, ...], month: pd.Timestamp) -> Path:
    token = f"month={month:%Y-%m}.parquet"
    candidates = [root / "target_free_scores" / token for root in roots]
    found = [path for path in candidates if path.is_file()]
    if len(found) != 1:
        raise AssertionError(
            f"{month:%Y-%m}: expected exactly one source receipt across {[str(item) for item in candidates]}, found {len(found)}"
        )
    return found[0]


def _target_free_score(path: Path, month: pd.Timestamp) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    missing = sorted(set(REQUIRED).difference(frame.columns))
    if missing:
        raise AssertionError(f"{path}: missing Router score columns {missing}")
    leaked = sorted(
        column for column in frame.columns
        if column not in IDENTITY
        and any(token in column.lower() for token in PROHIBITED_TOKENS)
    )
    if leaked:
        raise AssertionError(f"{path}: non-target-free score columns {leaked}")
    result = frame.loc[:, list(REQUIRED)].copy()
    result["candidate_id"] = result["candidate_id"].astype(str)
    result["__decision_ts__"] = pd.to_datetime(result["__decision_ts__"], utc=True, errors="raise")
    if result["candidate_id"].duplicated().any():
        raise AssertionError(f"{path}: duplicate candidate identity")
    if not result["side_name"].astype(str).str.lower().eq("long").all():
        raise AssertionError(f"{path}: non-long score identity")
    start, end = month, month + pd.offsets.MonthBegin(1)
    if not result["__decision_ts__"].ge(start).all() or not result["__decision_ts__"].lt(end).all():
        raise AssertionError(f"{path}: score rows cross month boundary")
    if not np.isfinite(pd.to_numeric(result["router_primary_rank"], errors="coerce")).all():
        raise AssertionError(f"{path}: non-finite Router rank")
    return result.sort_values(["__decision_ts__", "candidate_id"], kind="stable").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-roots", required=True, help="comma-separated disjoint immutable Router score roots")
    parser.add_argument("--months", required=True, help="comma-separated chronological YYYY-MM months")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    roots = tuple(Path(token.strip()).resolve() for token in args.source_roots.split(",") if token.strip())
    if not roots or len(set(roots)) != len(roots):
        raise ValueError("--source-roots must name unique non-empty paths")
    months = _months(args.months)
    out = args.out.resolve()
    if out.exists():
        raise FileExistsError(f"immutable output exists: {out}")
    out.mkdir(parents=True)
    target = out / "target_free_scores"
    target.mkdir()
    audit: list[dict[str, object]] = []
    used: list[Path] = []
    for month in months:
        source = _source_for_month(roots, month)
        score = _target_free_score(source, month)
        destination = target / f"month={month:%Y-%m}.parquet"
        score.to_parquet(destination, index=False, compression="zstd")
        used.append(source)
        audit.append({
            "month": f"{month:%Y-%m}", "rows": int(len(score)),
            "timestamps": int(score["__decision_ts__"].nunique()),
            "source": str(source), "source_sha256": _sha256([source]),
            "target_free": True,
        })
    pd.DataFrame(audit).to_parquet(out / "coverage_audit.parquet", index=False, compression="zstd")
    _write_json_exclusive(out / "run_manifest.json", {
        "schema": SCHEMA,
        "scope": "offline target-free Router score union; no training, labels, MC1, admission, portfolio, inference, live, or exchange mutation",
        "identity_contract": "each month is copied unchanged from exactly one strict-OOF Router receipt",
        "sources": [str(root) for root in roots],
        "source_sha256": _sha256(used),
        "months": [f"{month:%Y-%m}" for month in months],
        "audit": audit,
        "correctness": {
            "single_source_per_month": True,
            "all_scores_target_free": True,
            "no_label_or_path_columns": True,
            "all_identities_long_only_unique": True,
        },
    })
    print(out)


if __name__ == "__main__":
    main()
