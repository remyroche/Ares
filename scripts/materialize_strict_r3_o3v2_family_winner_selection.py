#!/usr/bin/env python3
"""Extract only strict-OOF per-family G3 winners for a later mixed pass.

This emits the flat F1--F6 selection shape required by the G3 runner.  It
does not evaluate an outcome or inspect any later period: it simply removes
the shared upstream core from an already sealed per-family contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


FAMILIES = ("f1", "f2", "f3", "f4", "f5", "f6")
CORE = {
    "f1_enhanced_base_bps", "f1_base_rank_ts", "f1_base_bps",
    "f1_efficiency_bps", "f1_timing_bps", "f1_e_minus_t",
    "f1_e_minus_b0", "f1_t_minus_b0", "f1_base_component_std",
}
SCHEMA = "strict_r3_o3v2_family_winner_selection_v1"


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


def run(*, source: Path, out: Path, receipt: Path) -> None:
    raw = json.loads(source.read_text())
    target = raw.get("target")
    contracts = raw.get("contracts")
    if not isinstance(target, str) or not isinstance(contracts, dict) or not set(FAMILIES).issubset(contracts):
        raise AssertionError("source must be a sealed G3 contract with target and F1--F6 family lists")
    selected: dict[str, list[str]] = {}
    for family in FAMILIES:
        values = contracts[family]
        if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
            raise AssertionError(f"{family} is not a field-name list")
        additions = [value for value in values if value not in CORE]
        if len(additions) > 4:
            raise AssertionError(f"{family} has more than the declared four G3 additions")
        selected[family] = additions
    _write_exclusive(out, selected)
    _write_exclusive(receipt, {
        "schema": SCHEMA, "target": target,
        "selection": "only per-family strict-OOF G3 winners; shared nine-field core supplied by downstream runner",
        "source": str(source), "source_hash": _hash(source), "families": selected,
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    run(source=args.source, out=args.out, receipt=args.receipt)


if __name__ == "__main__":
    main()
