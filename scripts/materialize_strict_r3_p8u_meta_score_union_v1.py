#!/usr/bin/env python3
"""Create a target-free union of matching P8U Meta OOS score receipts."""

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
REQUIRED = (*IDENTITY, "base_score", "base_rank_ts", "meta_raw_score", "meta_rank_ts", "arm", "family", "scale", "query_contract", "trial", "held_month", "target_free", "fit_weight_profile")
FORBIDDEN = ("policy_", "label_", "target_", "outcome_", "path_")
SCHEMA = "strict_r3_p8u_meta_targetfree_score_union_v1"


def _sha(paths: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(str(path).encode())
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
    return digest.hexdigest()


def _once(path: Path, payload: object) -> None:
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)


def _months(value: str) -> tuple[pd.Timestamp, ...]:
    months = tuple(pd.Timestamp(f"{item.strip()}-01", tz="UTC") for item in value.split(",") if item.strip())
    if not months or tuple(sorted(months)) != months or len(months) != len(set(months)):
        raise ValueError("--months must be non-empty, chronological unique YYYY-MM values")
    return months


def _source(roots: tuple[Path, ...], arm: str, month: pd.Timestamp) -> Path:
    token = f"month={month:%Y-%m}.parquet"
    candidates = [root / "target_free_scores" / arm / token for root in roots]
    found = [path for path in candidates if path.is_file()]
    if len(found) != 1:
        raise AssertionError(f"{month:%Y-%m}: expected one source among {[str(item) for item in candidates]}, found {len(found)}")
    return found[0]


def _read(path: Path, arm: str, month: pd.Timestamp) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    missing = sorted(set(REQUIRED).difference(frame.columns))
    if missing:
        raise AssertionError(f"{path}: missing Meta fields {missing}")
    # ``target_free`` is an explicit provenance boolean, not a target value.
    # All other target/path/policy/outcome fields are forbidden.
    leaked = sorted(column for column in frame.columns if column not in REQUIRED and any(token in column.lower() for token in FORBIDDEN))
    if leaked:
        raise AssertionError(f"{path}: non-target-free Meta columns {leaked}")
    result = frame.loc[:, list(REQUIRED)].copy()
    result["candidate_id"] = result.candidate_id.astype(str)
    result["__decision_ts__"] = pd.to_datetime(result.__decision_ts__, utc=True, errors="raise")
    if result.candidate_id.duplicated().any() or not result.side_name.astype(str).str.lower().eq("long").all():
        raise AssertionError(f"{path}: invalid long identity")
    start, end = month, month + pd.offsets.MonthBegin(1)
    if not result.__decision_ts__.ge(start).all() or not result.__decision_ts__.lt(end).all():
        raise AssertionError(f"{path}: rows cross held month")
    # ``--arm`` names the score-receipt directory (the frozen trial name),
    # whereas the persisted ``arm`` column names the target/query definition.
    # Require one declared semantic arm per source and preserve it unchanged.
    if result["arm"].astype(str).nunique() != 1 or not result["target_free"].astype(bool).all():
        raise AssertionError(f"{path}: semantic-arm/target-free contract mismatch")
    for column in ("base_score", "base_rank_ts", "meta_raw_score", "meta_rank_ts"):
        if not np.isfinite(pd.to_numeric(result[column], errors="coerce")).all():
            raise AssertionError(f"{path}: non-finite {column}")
    return result.sort_values(["__decision_ts__", "candidate_id"], kind="stable").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-roots", required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--months", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    roots = tuple(Path(item.strip()).resolve() for item in args.source_roots.split(",") if item.strip())
    if not roots or len(roots) != len(set(roots)):
        raise ValueError("source roots must be non-empty and unique")
    months = _months(args.months)
    out = args.out.resolve()
    if out.exists():
        raise FileExistsError(f"immutable output exists: {out}")
    target = out / "target_free_scores" / str(args.arm)
    target.mkdir(parents=True)
    audit, used = [], []
    for month in months:
        source = _source(roots, str(args.arm), month)
        frame = _read(source, str(args.arm), month)
        destination = target / f"month={month:%Y-%m}.parquet"
        frame.to_parquet(destination, index=False, compression="zstd")
        used.append(source)
        audit.append({"month": f"{month:%Y-%m}", "rows": int(len(frame)), "timestamps": int(frame.__decision_ts__.nunique()), "source": str(source), "source_sha256": _sha([source]), "target_free": True})
    pd.DataFrame(audit).to_parquet(out / "coverage_audit.parquet", index=False, compression="zstd")
    _once(out / "run_manifest.json", {"schema": SCHEMA, "scope": "offline target-free Meta score union; no training, labels, MC1, admission, portfolio, inference, live, or exchange mutation", "arm": str(args.arm), "sources": [str(root) for root in roots], "source_sha256": _sha(used), "months": [f"{month:%Y-%m}" for month in months], "audit": audit, "correctness": {"single_source_per_month": True, "target_free": True, "long_unique_identity": True}})
    print(out)


if __name__ == "__main__":
    main()
