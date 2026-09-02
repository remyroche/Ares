#!/usr/bin/env python3
"""Expose incumbent E/T raw coordinates as target-free head-score ledgers.

This is an identity-preserving adapter for the frozen three-way source.  It
allows a replacement B score to be blended with the *same* incumbent E/T
coordinates.  No label, policy outcome, candidate filter, or row ordering is
read or changed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


IDENTITY = ["candidate_id", "__decision_ts__", "side_name"]
COMPONENTS = {"E": "efficiency_bps", "T": "timing_bps"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument(
        "--identity-root", type=Path, default=None,
        help=(
            "optional target-free score ledger whose identities define a common "
            "router population; no outcome/label field is read"
        ),
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--months", required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    months = tuple(token.strip() for token in args.months.split(",") if token.strip())
    if not months or tuple(sorted(months)) != months or len(set(months)) != len(months):
        raise ValueError("months must be unique chronological YYYY-MM tokens")
    args.out.mkdir(parents=True)
    audit: list[dict[str, object]] = []
    for name, source_column in COMPONENTS.items():
        root = args.out / name
        root.mkdir()
        for token in months:
            source = args.source_root / f"month={token}" / "scores_features.parquet"
            frame = pd.read_parquet(source, columns=[*IDENTITY, source_column]).copy()
            frame["__decision_ts__"] = pd.to_datetime(frame["__decision_ts__"], utc=True, errors="raise")
            if frame.duplicated(IDENTITY).any():
                raise AssertionError(f"{source}: duplicate target-free candidate identity")
            if args.identity_root is not None:
                identity_path = args.identity_root / f"month={token}" / "target_free_scores.parquet"
                identities = pd.read_parquet(identity_path, columns=IDENTITY).copy()
                identities["__decision_ts__"] = pd.to_datetime(identities["__decision_ts__"], utc=True, errors="raise")
                if identities.duplicated(IDENTITY).any():
                    raise AssertionError(f"{identity_path}: duplicate router identity")
                frame = identities.merge(frame, on=IDENTITY, how="inner", validate="one_to_one")
                if len(frame) != len(identities):
                    raise AssertionError(f"{token}: incumbent E/T source does not cover every target-free router identity")
            score = pd.to_numeric(frame.pop(source_column), errors="coerce")
            if not np.isfinite(score).all():
                raise AssertionError(f"{source}: non-finite incumbent {name} coordinate")
            frame["head_score"] = score.astype(np.float32)
            frame["held_month"] = token
            target = root / f"month={token}"
            target.mkdir()
            frame.to_parquet(target / "target_free_scores.parquet", index=False, compression="zstd")
            audit.append({"head": name, "month": token, "rows": len(frame), "target_free": True})
    manifest = {
        "schema": "strict_r3_incumbent_component_scores_v1",
        "scope": "offline research-only target-free E/T coordinate adapter; no labels, outcomes, or candidate eligibility fields read",
        "source_root": str(args.source_root),
        "identity_root": str(args.identity_root) if args.identity_root is not None else None,
        "months": list(months),
        "components": COMPONENTS,
        "identity_contract": (
            "every source identity is retained exactly once"
            if args.identity_root is None
            else "every identity from the target-free router ledger is retained exactly once"
        ),
    }
    (args.out / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    pd.DataFrame(audit).to_parquet(args.out / "identity_audit.parquet", index=False, compression="zstd")
    print(json.dumps({"event": "complete", "months": len(months), "rows": int(sum(row["rows"] for row in audit) // len(COMPONENTS))}))


if __name__ == "__main__":
    main()
