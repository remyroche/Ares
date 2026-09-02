#!/usr/bin/env python3
"""Assemble an immutable, target-free F72 feature source from sealed months.

The early source and the later full-universe source have disjoint calendar
coverage.  This utility intentionally creates only directory symlinks and a
lineage manifest; it never reads, writes, filters, or augments a candidate
panel.  The downstream strict-OOF router/scorer therefore sees precisely the
already-audited target-free monthly parquet for each calendar month.

Research only.  No inference, admission, portfolio, execution, or exchange
state is read or changed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _months(root: Path) -> dict[str, Path]:
    found: dict[str, Path] = {}
    for source in root.glob("month=*"):
        token = source.name.removeprefix("month=")
        panel = source / "causal_feature_universe.parquet"
        if not panel.is_file():
            continue
        if token in found:
            raise AssertionError(f"{root}: duplicate calendar month {token}")
        found[token] = source.resolve()
    if not found:
        raise FileNotFoundError(f"{root}: no monthly causal feature panels")
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--early-root", type=Path, required=True)
    parser.add_argument("--later-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    early = _months(args.early_root)
    later = _months(args.later_root)
    overlap = sorted(set(early) & set(later))
    if overlap:
        raise AssertionError(f"source months overlap: {overlap}")
    combined = {**early, **later}
    ordered = sorted(combined)
    if not ordered:
        raise AssertionError("combined target-free feature source is empty")
    args.out.mkdir(parents=True)
    for token in ordered:
        (args.out / f"month={token}").symlink_to(combined[token], target_is_directory=True)
    manifest = {
        "schema": "strict_r3_f72_contiguous_feature_source_v1",
        "scope": "offline research-only target-free source composition; no rows were transformed, filtered, or outcome-joined",
        "month_count": len(ordered),
        "months": ordered,
        "sources": {
            "early": {"root": str(args.early_root.resolve()), "manifest_sha256": _sha256(args.early_root / "run_manifest.json"), "months": sorted(early)},
            "later": {"root": str(args.later_root.resolve()), "months": sorted(later)},
        },
        "composition": "month directories are immutable symlinks to the audited source panels",
    }
    (args.out / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"event": "complete", "months": len(ordered), "first": ordered[0], "last": ordered[-1]}))


if __name__ == "__main__":
    main()
