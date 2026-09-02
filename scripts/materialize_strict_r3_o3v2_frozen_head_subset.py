#!/usr/bin/env python3
"""Apply a development-frozen O3-v2 specialist subset without outcome access.

The head-quality selector writes an ordered list using only a declared
development block.  This adapter applies that exact list to arbitrary later
target-free specialist receipts.  It is deliberately storage-only: policy
outcomes, semantic labels, MC1, admission, portfolio state, and live artifacts
are neither read nor changed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import run_strict_r3_o3v2_target_funnel as target  # noqa: E402


SCHEMA = "strict_r3_o3v2_frozen_head_subset_v1"
SLOTS = (
    "cap100_ordinary", "cap80_ordinary", "cap120_equal_month",
    "cap40_equal_month", "cap60_equal_month",
)
PROHIBITED = set(target.PROHIBITED_SCORE_COLUMNS)


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    paths = sorted(path.rglob("*.parquet")) if path.is_dir() else [path]
    for child in paths:
        digest.update(str(child).encode())
        with child.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
    return digest.hexdigest()


def _source(roots: Sequence[Path], target_name: str, month: str) -> Path:
    candidates = [root / "target_free_scores" / target_name / f"month={month}" / "scores.parquet" for root in roots]
    found = [path for path in candidates if path.exists()]
    if len(found) != 1:
        raise FileNotFoundError(f"{target_name} {month}: expected exactly one target-free specialist source, found {found}")
    return found[0]


def _write_exclusive(path: Path, payload: object) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(fd, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def run(
    *, roots: Sequence[Path], selection_path: Path, out: Path,
    target_name: str, months: Sequence[str], subset_count: int | None,
) -> None:
    if out.exists():
        raise FileExistsError(out)
    selection = json.loads(selection_path.read_text())
    if selection.get("target") != target_name:
        raise AssertionError(f"selection target {selection.get('target')!r} does not match {target_name!r}")
    ordered = selection.get("ordered_heads")
    if not isinstance(ordered, list) or not ordered or not all(isinstance(field, str) for field in ordered):
        raise AssertionError("development selection must contain a non-empty ordered_heads list")
    count = len(ordered) if subset_count is None else int(subset_count)
    if not 1 <= count <= len(ordered):
        raise ValueError(f"subset-count must be in [1, {len(ordered)}]")
    heads = tuple(ordered[:count])
    arm = f"{target_name}__frozen_best{count}"
    out.mkdir(parents=True)
    target_root = out / "target_free_scores" / arm
    target_root.mkdir(parents=True)
    audit: list[dict[str, object]] = []
    for month in months:
        source = _source(roots, target_name, month)
        raw = pd.read_parquet(source)
        if leaked := PROHIBITED.intersection(raw.columns):
            raise AssertionError(f"{source}: outcome field in target-free specialist receipt: {sorted(leaked)}")
        required = {"candidate_id", "__decision_ts__", "side_name", "f1_base_rank_ts", *heads}
        if missing := required - set(raw.columns):
            raise KeyError(f"{source}: missing frozen selected head columns {sorted(missing)}")
        result = raw.loc[:, ["candidate_id", "__decision_ts__", "side_name", "f1_base_rank_ts", *heads]].copy()
        ranks = result.loc[:, list(heads)].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        result = result.rename(columns={"f1_base_rank_ts": "base_rank_ts"})
        result["conditional_consensus_rank"] = np.nanmedian(ranks, axis=1).astype(np.float32)
        result["ordinary_shadow_consensus_rank"] = result["conditional_consensus_rank"]
        result["head_agreement_std"] = np.nanstd(ranks, axis=1).astype(np.float32)
        result["o3v2_rank_75_25"] = (
            .75 * pd.to_numeric(result["base_rank_ts"], errors="coerce")
            + .25 * pd.to_numeric(result["conditional_consensus_rank"], errors="coerce")
        ).astype(np.float32)
        for index, slot in enumerate(SLOTS):
            result[f"head__{slot}__rank"] = ranks[:, min(index, count - 1)].astype(np.float32)
        result = result.drop(columns=list(heads))
        if result["candidate_id"].duplicated().any():
            raise AssertionError(f"{source}: duplicate candidate IDs")
        if leaked := PROHIBITED.intersection(result.columns):
            raise AssertionError(f"{source}: output retained prohibited fields {sorted(leaked)}")
        result.to_parquet(target_root / f"month={month}.parquet", index=False, compression="zstd")
        audit.append({
            "month": month, "rows": int(len(result)), "selected_heads": int(count),
            "complete_fraction": float(result.notna().all(axis=1).mean()),
        })
    pd.DataFrame(audit).to_parquet(out / "adapter_audit.parquet", index=False, compression="zstd")
    _write_exclusive(out / "run_manifest.json", {
        "schema": SCHEMA,
        "scope": "target-free frozen-head adapter only; no fit, outcome join, MC1, admission, portfolio, or live change",
        "target": target_name, "arm": arm, "months": list(months),
        "selection": "development-frozen ordered specialist heads; later sources are not reselected",
        "selected_heads": list(heads), "sources": [str(root) for root in roots],
        "source_hashes": {str(root): _hash(root) for root in roots},
        "selection_hash": _hash(selection_path),
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", action="append", type=Path, required=True)
    parser.add_argument("--selection-json", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--months", required=True)
    parser.add_argument("--subset-count", type=int)
    args = parser.parse_args()
    run(
        roots=args.source_root, selection_path=args.selection_json, out=args.out,
        target_name=args.target, months=tuple(token for token in args.months.split(",") if token),
        subset_count=args.subset_count,
    )


if __name__ == "__main__":
    main()
