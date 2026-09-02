#!/usr/bin/env python3
"""Create an immutable, zero-copy union of monthly target-free score files."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


IDENTITY = ("candidate_id", "__decision_ts__", "side_name")
PROHIBITED = ("label", "outcome", "future", "policy_net_bps", "path_valid", "target_invalid")


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", action="append", type=Path, required=True)
    parser.add_argument("--source-subdir", default="", help="optional identical score subdirectory below each source root")
    parser.add_argument("--output-subdir", default="target_free_scores", help="score subdirectory below the new bridge root")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    files: dict[str, Path] = {}
    for root in args.source_root:
        source = root.resolve() / str(args.source_subdir)
        for path in sorted(source.glob("month=*.parquet")):
            if path.name in files:
                raise AssertionError(f"duplicate month source {path.name}: {files[path.name]} and {path}")
            files[path.name] = path
    if not files:
        raise FileNotFoundError("no month=YYYY-MM.parquet files in declared source roots")
    schemas = {name: pq.ParquetFile(path).schema_arrow.names for name, path in files.items()}
    baseline = next(iter(schemas.values()))
    if any(names != baseline for names in schemas.values()):
        raise AssertionError("monthly score schemas differ")
    missing = sorted(set(IDENTITY).difference(baseline))
    prohibited = [name for name in baseline if any(token in name.lower() for token in PROHIBITED)]
    if missing or prohibited:
        raise AssertionError(f"target-free schema invalid; missing={missing}; prohibited={prohibited}")
    audits: list[dict[str, object]] = []
    for name, path in sorted(files.items()):
        frame = pd.read_parquet(path, columns=list(IDENTITY))
        frame["__decision_ts__"] = pd.to_datetime(frame["__decision_ts__"], utc=True, errors="raise")
        if frame.duplicated(list(IDENTITY)).any():
            raise AssertionError(f"{path}: duplicate target-free identities")
        audits.append({"month": name[6:13], "rows": int(len(frame)), "timestamps": int(frame["__decision_ts__"].nunique()), "source": str(path), "sha256": _sha(path)})
    target = args.out / str(args.output_subdir)
    target.mkdir(parents=True)
    for name, source in sorted(files.items()):
        # A hard link preserves bytes and avoids duplicating historical scores.
        os.link(source, target / name)
    manifest = {
        "schema": "strict_r3_target_free_score_ledger_bridge_v1",
        "scope": "offline immutable target-free score ledger bridge; no labels, outcomes, models, calibration, admission, portfolio, live, or execution mutation",
        "source_roots": [str(path.resolve()) for path in args.source_root],
        "source_subdir": str(args.source_subdir),
        "output_subdir": str(args.output_subdir),
        "ordered_schema": baseline,
        "target_free": True,
        "identity_overlap_rows": 0,
        "months": audits,
        "storage": "hard-link immutable source files; source bytes are not copied or modified",
    }
    (args.out / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"event": "complete", "months": len(audits), "rows": sum(int(row["rows"]) for row in audits), "out": str(args.out)}, sort_keys=True))


if __name__ == "__main__":
    main()
