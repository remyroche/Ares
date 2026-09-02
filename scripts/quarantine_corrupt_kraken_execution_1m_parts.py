#!/usr/bin/env python3
"""Move explicitly named corrupted Kraken one-minute parts to quarantine.

The operation is deliberately narrow and reversible: each named file must be
inside the canonical execution store and fail a full Parquet read before it
is moved.  A hash-bound receipt records its original and quarantine paths.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.data_store import canonical_kraken_execution_1m_root


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data_perp")
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--part", action="append", type=Path)
    source_group.add_argument(
        "--coverage",
        type=Path,
        help=(
            "Exact-path coverage audit produced by a fail-closed replay.  Every "
            "non-null source_error must identify one unreadable canonical Parquet "
            "part; the parts are then validated again before their reversible move."
        ),
    )
    source_group.add_argument(
        "--download-manifest",
        action="append",
        type=Path,
        help=(
            "Append-only one-minute backfill manifest.  Every failed result must "
            "name one unreadable canonical part in its exact error; those parts "
            "are revalidated before their reversible move."
        ),
    )
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()

    root = canonical_kraken_execution_1m_root(args.data_root).resolve()
    parts = list(args.part or [])
    if args.coverage is not None:
        coverage = pd.read_parquet(args.coverage)
        if "source_error" not in coverage.columns:
            raise ValueError(f"coverage audit has no source_error column: {args.coverage}")
        paths: set[Path] = set()
        for error in coverage["source_error"].dropna().astype(str):
            match = re.search(r"execution_1m source (?P<path>/.+?\.parquet):", error)
            if match is None:
                # A coverage audit can also record a genuine public-source gap
                # (for example, no complete policy path).  That is not a local
                # corrupt shard and must remain a fail-closed source exclusion,
                # not a reason to alter the store.
                continue
            paths.add(Path(match.group("path")))
        if not paths:
            raise ValueError(f"coverage audit contains no unreadable parts: {args.coverage}")
        parts = sorted(paths)
    if args.download_manifest:
        paths: set[Path] = set()
        for manifest_path in args.download_manifest:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            for row in manifest.get("results", []):
                error = str(row.get("error") or "")
                if not error:
                    continue
                match = re.search(r"invalid execution_1m part (?P<path>.+?\.parquet):", error)
                if match is not None:
                    paths.add(Path(match.group("path")))
        if not paths:
            raise ValueError("download manifests contain no named unreadable canonical parts")
        parts = sorted(paths)
    receipts: list[dict[str, object]] = []
    for raw in parts:
        path = raw.resolve()
        try:
            relative = path.relative_to(root / "ohlcv")
        except ValueError as exc:
            raise ValueError(f"refusing to quarantine outside canonical ohlcv root: {path}") from exc
        target = root / "quarantine" / relative
        if not path.is_file():
            # An interrupted earlier recovery may already have atomically moved
            # this exact immutable object.  Treat that as an auditable,
            # idempotent state only when the expected quarantine object is
            # present and still fails a full Parquet read.  This does not
            # silently accept a missing source part or a readable replacement.
            if not target.is_file():
                raise FileNotFoundError(path)
            try:
                pd.read_parquet(target)
            except Exception as exc:
                validation = f"{type(exc).__name__}: {exc}"
                operation = "already_quarantined_revalidated"
            else:
                # Do not resurrect an object merely because it now decodes: it
                # was placed outside the canonical source by an earlier
                # integrity action, and no current receipt can prove why.  It
                # remains excluded and the public-source backfill supplies the
                # requested history instead.
                validation = "readable_preexisting_quarantine_object_untrusted"
                operation = "already_quarantined_retained_untrusted"
            receipts.append({
                "original_path": str(path),
                "quarantine_path": str(target),
                "sha256": _sha256(target),
                "bytes": int(target.stat().st_size),
                "validation_failure": validation,
                "operation": operation,
            })
            continue
        try:
            pd.read_parquet(path)
        except Exception as exc:
            failure = f"{type(exc).__name__}: {exc}"
        else:
            raise ValueError(f"refusing to quarantine readable Parquet part: {path}")
        digest = _sha256(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            raise FileExistsError(f"quarantine target already exists: {target}")
        os.replace(path, target)
        receipts.append({
            "original_path": str(path),
            "quarantine_path": str(target),
            "sha256": digest,
            "bytes": int(target.stat().st_size),
            "validation_failure": failure,
            "operation": "atomic_move_reversible",
        })
    receipt = Path(args.receipt).resolve()
    receipt.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(receipt, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump({
            "schema": "kraken_execution_1m_corrupt_part_quarantine_v1",
            "canonical_root": str(root),
            "source_coverage": str(args.coverage.resolve()) if args.coverage else None,
            "download_manifests": (
                [str(path.resolve()) for path in args.download_manifest]
                if args.download_manifest
                else None
            ),
            "parts": receipts,
            "restoration": "move each quarantine_path back to original_path after independent validation",
        }, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(receipt)


if __name__ == "__main__":
    main()
