#!/usr/bin/env python3
"""Create an immutable strict-R3 inference-bundle successor for a source patch.

This utility does not touch model artifacts or decision-policy values.  It
copies a validated source contract, updates explicitly named runtime-code
hashes, records the reason, and refuses to overwrite an existing destination.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reason", required=True)
    parser.add_argument(
        "--runtime-source",
        action="append",
        default=[],
        metavar="RELATIVE_PATH",
        help="Repository-relative source whose current SHA-256 is resealed.",
    )
    args = parser.parse_args()
    base = (ROOT / args.base).resolve() if not args.base.is_absolute() else args.base.resolve()
    out = (ROOT / args.out).resolve() if not args.out.is_absolute() else args.out.resolve()
    if not base.is_file():
        raise FileNotFoundError(base)
    if out.exists():
        raise FileExistsError(f"immutable successor already exists: {out}")
    if ROOT not in out.parents:
        raise ValueError("output path escapes repository root")
    payload = json.loads(base.read_text())
    hashes = payload.get("runtime_code_sha256")
    if not isinstance(hashes, dict):
        raise ValueError("base inference bundle lacks runtime_code_sha256")
    changed: dict[str, str] = {}
    for relative in args.runtime_source:
        source = (ROOT / relative).resolve()
        if ROOT not in source.parents or not source.is_file():
            raise ValueError(f"invalid runtime source: {relative}")
        if relative not in hashes:
            raise ValueError(f"runtime source is not sealed by base bundle: {relative}")
        hashes[relative] = _sha(source)
        changed[relative] = hashes[relative]
    if not changed:
        raise ValueError("provide at least one --runtime-source")
    payload["version_note"] = str(args.reason)
    payload["runtime_reseal"] = {
        "parent_bundle": str(base.relative_to(ROOT)),
        "parent_sha256": _sha(base),
        "changed_runtime_sources": changed,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_name(f".{out.name}.tmp.{os.getpid()}")
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, out)
    finally:
        temporary.unlink(missing_ok=True)
    print(json.dumps({"out": str(out.relative_to(ROOT)), "sha256": _sha(out), "changed": changed}))


if __name__ == "__main__":
    main()
