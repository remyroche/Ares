#!/usr/bin/env python3
"""Permanently delete only hash-verified, still-unreadable 1m quarantine parts.

This is intentionally a one-purpose cleanup tool.  It consumes the immutable
receipts created by ``quarantine_corrupt_kraken_execution_1m_parts.py`` and
will delete a part only if all of these are true at deletion time:

* it is under the canonical execution-1m quarantine tree;
* the receipt records an explicit, failed full Parquet read (not a merely
  untrusted or readable object);
* its current SHA-256 still matches the receipt; and
* a new complete ``pandas.read_parquet`` call still fails.

The output receipt is created before deletion, then amended with the exact
per-file deletion status.  It retains original path, quarantine path, hash,
byte size, source receipt, and revalidation failure after the bytes are gone.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.data_store import canonical_kraken_execution_1m_root


SAFE_OPERATIONS = {
    "atomic_move_reversible",
    "already_quarantined_revalidated",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _full_read_failure(path: Path, *, timeout_seconds: float) -> str:
    """Revalidate in a child so a malformed shard cannot stall cleanup."""
    command = [
        sys.executable,
        "-c",
        (
            "import pandas as pd,sys; "
            "pd.read_parquet(sys.argv[1])"
        ),
        str(path),
    ]
    try:
        completed = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        # A timeout is not enough evidence to delete.  Retain the object for
        # separate inspection rather than treating a slow read as corruption.
        raise RuntimeError(
            f"refusing to delete read-timeout object after {timeout_seconds}s: {path}"
        ) from exc
    if completed.returncode == 0:
        raise ValueError(f"refusing to delete readable quarantine object: {path}")
    error = (completed.stderr or completed.stdout).strip().replace("\n", " ")
    return f"child_read_failed(exit={completed.returncode}): {error}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data_perp")
    parser.add_argument(
        "--receipt-dir",
        action="append",
        type=Path,
        required=True,
        help="Directory containing per-part immutable quarantine receipts.",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--read-timeout-seconds", type=float, default=20.0)
    parser.add_argument(
        "--delete",
        action="store_true",
        help="Perform deletion after every candidate passes the revalidation gate.",
    )
    args = parser.parse_args()

    execution_root = canonical_kraken_execution_1m_root(args.data_root).resolve()
    quarantine_root = (execution_root / "quarantine").resolve()
    candidates: dict[Path, dict[str, object]] = {}

    for receipt_dir in args.receipt_dir:
        receipt_dir = receipt_dir.resolve()
        for receipt_path in sorted(receipt_dir.glob("*.json")):
            payload = json.loads(receipt_path.read_text(encoding="utf-8"))
            if payload.get("schema") != "kraken_execution_1m_corrupt_part_quarantine_v1":
                # Historical-repair directories also contain immutable download
                # manifests.  They are not deletion authority and must simply
                # be ignored; only a dedicated quarantine receipt can ever
                # nominate a file for purge.
                continue
            for part in payload.get("parts", []):
                operation = str(part.get("operation") or "")
                validation = str(part.get("validation_failure") or "")
                if operation not in SAFE_OPERATIONS:
                    continue
                if not validation or validation.startswith("readable_"):
                    continue
                quarantine_path = Path(str(part["quarantine_path"])).resolve()
                try:
                    quarantine_path.relative_to(quarantine_root)
                except ValueError as exc:
                    raise ValueError(
                        f"refusing to delete outside canonical quarantine: {quarantine_path}"
                    ) from exc
                record = {
                    "source_receipt": str(receipt_path),
                    "original_path": str(part["original_path"]),
                    "quarantine_path": str(quarantine_path),
                    "receipt_sha256": str(part["sha256"]),
                    "receipt_bytes": int(part["bytes"]),
                    "receipt_validation_failure": validation,
                    "receipt_operation": operation,
                }
                prior = candidates.get(quarantine_path)
                if prior is not None and prior["receipt_sha256"] != record["receipt_sha256"]:
                    raise ValueError(f"conflicting hashes for quarantine object: {quarantine_path}")
                candidates[quarantine_path] = record

    if not candidates:
        raise ValueError("no explicitly corrupt, deletable quarantine objects found")

    verified: list[dict[str, object]] = []
    retained: list[dict[str, object]] = []
    for quarantine_path, record in sorted(candidates.items()):
        if not quarantine_path.is_file():
            raise FileNotFoundError(f"quarantine object disappeared before cleanup: {quarantine_path}")
        current_hash = _sha256(quarantine_path)
        if current_hash != record["receipt_sha256"]:
            raise ValueError(f"hash changed since quarantine receipt: {quarantine_path}")
        current_bytes = int(quarantine_path.stat().st_size)
        if current_bytes != record["receipt_bytes"]:
            raise ValueError(f"byte count changed since quarantine receipt: {quarantine_path}")
        record = dict(record)
        try:
            record["revalidated_failure"] = _full_read_failure(
                quarantine_path, timeout_seconds=args.read_timeout_seconds
            )
        except (ValueError, RuntimeError) as exc:
            # A shard which now decodes, or cannot be decisively read inside
            # the isolation timeout, is explicitly retained.  Neither case
            # meets the irreversible-deletion burden.
            record["retained_reason"] = str(exc)
            record["current_sha256"] = current_hash
            record["current_bytes"] = current_bytes
            retained.append(record)
            continue
        record["current_sha256"] = current_hash
        record["current_bytes"] = current_bytes
        verified.append(record)

    out = args.out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise FileExistsError(f"refusing to overwrite cleanup receipt: {out}")
    payload: dict[str, object] = {
        "schema": "kraken_execution_1m_verified_corrupt_cleanup_v1",
        "execution_root": str(execution_root),
        "mode": "delete" if args.delete else "audit_only",
        "parts": verified,
        "retained_after_revalidation": retained,
        "deletion_results": [],
    }
    descriptor = os.open(out, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")

    if args.delete:
        results: list[dict[str, str]] = []
        for record in verified:
            path = Path(str(record["quarantine_path"]))
            try:
                path.unlink()
            except Exception as exc:
                results.append({"quarantine_path": str(path), "status": f"failed: {type(exc).__name__}: {exc}"})
            else:
                results.append({"quarantine_path": str(path), "status": "deleted"})
        payload["deletion_results"] = results
        out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        failures = [row for row in results if row["status"] != "deleted"]
        if failures:
            raise RuntimeError(f"cleanup had {len(failures)} deletion failures; receipt retained: {out}")

    print(json.dumps({
        "receipt": str(out),
        "verified_still_corrupt": len(verified),
        "deleted": sum(row.get("status") == "deleted" for row in payload["deletion_results"]),
        "retained_after_revalidation": len(retained),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
