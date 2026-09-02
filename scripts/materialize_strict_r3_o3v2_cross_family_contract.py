#!/usr/bin/env python3
"""Compose frozen family winners with a later frozen cross-family mixed set.

The first G3 run selects up to four additions independently within each
semantic family.  A second G3 run evaluates only the union of those winners
and writes a ``mixed`` contract.  This helper joins the two frozen decisions
for downstream specialist fitting; it does not fit, score, join outcomes, or
touch MC1/live artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


FAMILIES = ("f1", "f2", "f3", "f4", "f5", "f6")
SCHEMA = "strict_r3_o3v2_cross_family_contract_v1"


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_exclusive(path: Path, payload: object) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(fd, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def run(*, family_contract: Path, cross_contract: Path, out: Path) -> None:
    if out.exists():
        raise FileExistsError(out)
    family = json.loads(family_contract.read_text())
    cross = json.loads(cross_contract.read_text())
    target = family.get("target")
    if not isinstance(target, str) or cross.get("target") != target:
        raise AssertionError("family and cross-family G3 contracts must have the same declared target")
    family_sets = family.get("contracts")
    cross_sets = cross.get("contracts")
    if not isinstance(family_sets, dict) or not set(FAMILIES).issubset(family_sets):
        raise AssertionError("family contract must include all F1--F6 frozen selections")
    if not isinstance(cross_sets, dict) or "mixed" not in cross_sets:
        raise AssertionError("cross-family contract must include the frozen mixed selection")
    selected = {name: family_sets[name] for name in FAMILIES}
    selected["mixed"] = cross_sets["mixed"]
    if not all(isinstance(values, list) and all(isinstance(value, str) for value in values) for values in selected.values()):
        raise AssertionError("all frozen contracts must be lists of field names")
    _write_exclusive(out, {
        "schema": SCHEMA,
        "target": target,
        "contracts": selected,
        "selection": (
            "per-family strict-OOF winners followed by a strict-OOF greedy mixed pass over only their union"
        ),
        "family_contract": str(family_contract),
        "cross_contract": str(cross_contract),
        "source_hashes": {"family": _hash(family_contract), "cross": _hash(cross_contract)},
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family-contract", type=Path, required=True)
    parser.add_argument("--cross-contract", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(family_contract=args.family_contract, cross_contract=args.cross_contract, out=args.out)


if __name__ == "__main__":
    main()
