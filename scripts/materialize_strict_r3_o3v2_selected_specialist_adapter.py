#!/usr/bin/env python3
"""Normalize retained T3 specialist receipts for the generic O3-v2 MC1 test.

This is a target-free storage adapter.  It does not fit a model, join an
outcome, alter a score, or touch a live artifact.  Source roots may contain a
single sealed month each; exactly one source is required for every emitted
month so partially produced runs cannot be silently mixed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

import run_strict_r3_o3v2_target_funnel as target


SCHEMA = "strict_r3_o3v2_selected_specialist_adapter_v1"
TARGET = "T3_pair_residual_lambdarank"
MODES = ("H2", "H3")
CANONICAL_HEADS = (
    "cap100_ordinary", "cap80_ordinary", "cap120_equal_month",
    "cap40_equal_month", "cap60_equal_month",
)
PROHIBITED = set(target.PROHIBITED_SCORE_COLUMNS)


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    for child in sorted(path.rglob("*.parquet")) if path.is_dir() else [path]:
        digest.update(str(child).encode())
        with child.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
    return digest.hexdigest()


def _source(roots: Sequence[Path], month: str) -> Path:
    candidates = [root / "target_free_scores" / TARGET / f"month={month}" / "scores.parquet" for root in roots]
    found = [path for path in candidates if path.exists()]
    if len(found) != 1:
        raise FileNotFoundError(f"{TARGET} {month}: expected exactly one sealed source, found {found}")
    return found[0]


def _columns(mode: str) -> tuple[str, ...]:
    prefix = TARGET.lower()
    if mode == "H2":
        return tuple(f"{prefix}__{name}__rank" for name in (
            "h2_ordinary", "h2_equal_month", "h2_equal_archetype", "h2_hard_base_error", "h2_policy_state",
        ))
    return (f"{prefix}__h3_f4_f5__rank",)


def run(*, roots: Sequence[Path], out: Path, months: Sequence[str], mode: str) -> None:
    if out.exists():
        raise FileExistsError(out)
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    score_columns = _columns(mode)
    arm = f"T3_{mode}_selected"
    out.mkdir(parents=True)
    target_root = out / "target_free_scores" / arm
    target_root.mkdir(parents=True)
    audit: list[dict[str, object]] = []
    for month in months:
        source = _source(roots, month)
        raw = pd.read_parquet(source)
        if leaked := PROHIBITED.intersection(raw.columns):
            raise AssertionError(f"{source}: outcome field in target-free receipt: {sorted(leaked)}")
        required = {"candidate_id", "__decision_ts__", "side_name", "f1_base_rank_ts", *score_columns}
        if missing := required - set(raw.columns):
            raise KeyError(f"{source}: missing {sorted(missing)}")
        result = raw.loc[:, ["candidate_id", "__decision_ts__", "side_name", "f1_base_rank_ts", *score_columns]].copy()
        head_matrix = result.loc[:, list(score_columns)].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        if mode == "H2":
            # Preserve the producer's median ensemble exactly.  Recomputing
            # it from the same five ranks is an identity check, not a new
            # combining model.
            if "specialist_ensemble_rank" not in raw.columns:
                raise KeyError(f"{source}: missing specialist_ensemble_rank")
            consensus = pd.to_numeric(raw["specialist_ensemble_rank"], errors="coerce").to_numpy(float)
            recomputed = np.nanmedian(head_matrix, axis=1)
            if not np.allclose(consensus, recomputed, rtol=0.0, atol=1e-7, equal_nan=True):
                raise AssertionError(f"{source}: H2 ensemble identity mismatch")
            disagreement = np.nanstd(head_matrix, axis=1)
            aliases = score_columns
        else:
            consensus = head_matrix[:, 0]
            disagreement = np.zeros(len(result), dtype=float)
            aliases = score_columns * len(CANONICAL_HEADS)
        result["conditional_consensus_rank"] = consensus.astype(np.float32)
        result["head_agreement_std"] = disagreement.astype(np.float32)
        result["o3v2_rank_75_25"] = (
            .75 * pd.to_numeric(result["f1_base_rank_ts"], errors="coerce")
            + .25 * pd.to_numeric(result["conditional_consensus_rank"], errors="coerce")
        ).astype(np.float32)
        for canonical, source_column in zip(CANONICAL_HEADS, aliases):
            result[f"head__{canonical}__rank"] = pd.to_numeric(result[source_column], errors="coerce").astype(np.float32)
        result = result.drop(columns=["f1_base_rank_ts", *score_columns])
        if result["candidate_id"].duplicated().any():
            raise AssertionError(f"{source}: duplicate candidate identities")
        result.to_parquet(target_root / f"month={month}.parquet", index=False, compression="zstd")
        audit.append({"month": month, "rows": int(len(result)), "mode": mode, "complete_fraction": float(result.notna().all(axis=1).mean())})
    pd.DataFrame(audit).to_parquet(out / "adapter_audit.parquet", index=False, compression="zstd")
    manifest = {
        "schema": SCHEMA,
        "scope": "target-free adapter only; no fit, outcome join, MC1, portfolio, or live change",
        "target": TARGET, "mode": mode, "arm": arm, "months": list(months),
        "sources": [str(root) for root in roots],
        "source_hashes": {str(root): _hash(root) for root in roots},
        "coordinates": {
            "consensus": "producer median H2 ensemble" if mode == "H2" else "H3 F4/F5 hybrid rank",
            "combined": "0.75 * base timestamp rank + 0.25 * retained specialist consensus rank",
            "disagreement": "standard deviation across H2's five output ranks" if mode == "H2" else "0.0; one H3 head",
        },
    }
    fd = os.open(out / "run_manifest.json", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(fd, "w") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", action="append", type=Path, required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--months", required=True)
    parser.add_argument("--mode", choices=MODES, required=True)
    args = parser.parse_args()
    run(roots=args.source_root, out=args.out, months=tuple(token for token in args.months.split(",") if token), mode=args.mode)


if __name__ == "__main__":
    main()
