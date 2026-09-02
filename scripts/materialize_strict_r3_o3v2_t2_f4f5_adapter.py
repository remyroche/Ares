#!/usr/bin/env python3
"""Create a target-free MC1 adapter for the predeclared T2 F4/F5 pair.

The adapter is a storage-normalisation layer only.  It transforms held OOF
specialist ranks into the aggregate coordinates expected by the existing
research-only O3-v2 MC1 portfolio adapter.  No outcome columns are read or
written, and no model is fitted here.
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


SCHEMA = "strict_r3_o3v2_t2_f4f5_mc1_adapter_v1"
ARM_BY_MODE = {
    "F4F5": "T2_F4_F5_selected",
    "H3": "T2_H3_F4_F5_selected",
    "H2EA": "T2_H2_EQUAL_ARCHETYPE_selected",
}
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


def run(*, specialist_root: Path, out: Path, months: Sequence[str], mode: str) -> None:
    if out.exists():
        raise FileExistsError(out)
    if mode not in ARM_BY_MODE:
        raise ValueError(f"mode must be one of {tuple(ARM_BY_MODE)}")
    arm = ARM_BY_MODE[mode]
    source_root = specialist_root / "target_free_scores" / "T2_economic_residual_ordinal"
    out.mkdir(parents=True)
    target_root = out / "target_free_scores" / arm
    target_root.mkdir(parents=True)
    audits: list[dict[str, object]] = []
    f4 = "t2_economic_residual_ordinal__f4__rank"
    f5 = "t2_economic_residual_ordinal__f5__rank"
    h3 = "t2_economic_residual_ordinal__h3_f4_f5__rank"
    h2ea = "t2_economic_residual_ordinal__h2_equal_archetype__rank"
    for month in months:
        source = source_root / f"month={month}" / "scores.parquet"
        raw = pd.read_parquet(source)
        leaked = PROHIBITED.intersection(raw.columns)
        if leaked:
            raise AssertionError(f"{source}: outcome field in target-free specialist receipt: {sorted(leaked)}")
        specialist_fields = (f4, f5) if mode == "F4F5" else ((h3,) if mode == "H3" else (h2ea,))
        required = {"candidate_id", "__decision_ts__", "side_name", "f1_base_rank_ts", *specialist_fields}
        if missing := required - set(raw.columns):
            raise KeyError(f"{source}: missing {sorted(missing)}")
        output = raw.loc[:, ["candidate_id", "__decision_ts__", "side_name", "f1_base_rank_ts", *specialist_fields]].copy()
        if mode == "F4F5":
            output["conditional_consensus_rank"] = .5 * (
                pd.to_numeric(output[f4], errors="coerce") + pd.to_numeric(output[f5], errors="coerce")
            )
            output["head_agreement_std"] = np.abs(
                pd.to_numeric(output[f4], errors="coerce") - pd.to_numeric(output[f5], errors="coerce")
            ) / np.sqrt(2.0)
            aliases = (f4, f5, f4, f5, "conditional_consensus_rank")
        else:
            single = h3 if mode == "H3" else h2ea
            output["conditional_consensus_rank"] = pd.to_numeric(output[single], errors="coerce")
            # A one-head hybrid has no internal disagreement.  The aggregate
            # MC1 mode still sees its parent score agreement coordinates.
            output["head_agreement_std"] = 0.0
            aliases = (single, single, single, single, single)
        output["o3v2_rank_75_25"] = (
            .75 * pd.to_numeric(output["f1_base_rank_ts"], errors="coerce")
            + .25 * pd.to_numeric(output["conditional_consensus_rank"], errors="coerce")
        )
        # Aggregate feature mode does not consume individual head ranks, but
        # the generic reader validates their presence.  These aliases do not
        # manufacture additional information; all five names retain the two
        # selected, development-frozen F4/F5 coordinates.
        for name, source_name in zip(("cap100_ordinary", "cap80_ordinary", "cap120_equal_month", "cap40_equal_month", "cap60_equal_month"), aliases):
            output[f"head__{name}__rank"] = pd.to_numeric(output[source_name], errors="coerce")
        output = output.drop(columns=["f1_base_rank_ts", *specialist_fields])
        if output["candidate_id"].duplicated().any():
            raise AssertionError(f"{source}: duplicate candidate IDs")
        # Match the generic O3-v2 reader's immutable flat monthly receipt
        # layout: target_free_scores/<arm>/month=YYYY-MM.parquet.
        output.to_parquet(target_root / f"month={month}.parquet", index=False, compression="zstd")
        audits.append({"month": month, "rows": int(len(output)), "f4_f5_complete_fraction": float(output[["conditional_consensus_rank", "o3v2_rank_75_25", "head_agreement_std"]].notna().all(axis=1).mean())})
    pd.DataFrame(audits).to_parquet(out / "adapter_audit.parquet", index=False, compression="zstd")
    manifest = {
        "schema": SCHEMA, "scope": "target-free research adapter only; no fitting, outcome join, MC1, live, or canonical change",
        "arm": arm, "source": str(specialist_root), "months": list(months), "mode": mode,
        "selection": (
            "development-frozen T2 F4 state-transition plus F5 parent-score-provenance heads"
            if mode == "F4F5" else (
                "predeclared H3 hybrid over the frozen T2 F4/F5 feature families"
                if mode == "H3" else "development-selected H2 equal-archetype population head"
            )
        ),
        "coordinates": {
            "consensus": (
                "0.5 * F4_rank + 0.5 * F5_rank" if mode == "F4F5"
                else ("H3(F4,F5)_rank" if mode == "H3" else "H2_equal_archetype_rank")
            ),
            "combined": "0.75 * base_rank + 0.25 * consensus",
            "disagreement": "abs(F4_rank - F5_rank) / sqrt(2)" if mode == "F4F5" else "0.0 (single hybrid head)",
        },
        "source_hash": _hash(specialist_root),
    }
    fd = os.open(out / "run_manifest.json", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(fd, "w") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--specialist-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--months", required=True, help="comma-separated YYYY-MM")
    parser.add_argument("--mode", choices=tuple(ARM_BY_MODE), default="F4F5")
    args = parser.parse_args()
    run(specialist_root=args.specialist_root, out=args.out, months=tuple(token for token in args.months.split(",") if token), mode=args.mode)


if __name__ == "__main__":
    main()
