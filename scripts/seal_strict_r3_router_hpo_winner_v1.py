#!/usr/bin/env python3
"""Seal a router HPO winner as an explicit immutable feature contract."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


SCHEMA = "strict_r3_router_hpo_winner_contract_v1"


def _hash(fields: list[str]) -> str:
    return hashlib.sha256("\n".join(fields).encode("utf-8")).hexdigest()


def _write_once(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hpo-winner", type=Path, required=True)
    parser.add_argument("--finalists", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"immutable output exists: {args.out}")
    winner = json.loads(args.hpo_winner.read_text())["winner"]
    finalists = json.loads(args.finalists.read_text())["finalists"]
    match = [item for item in finalists if item["candidate"] == winner["candidate"]]
    if winner["candidate"] == "frozen30_control":
        control = json.loads(args.finalists.read_text()).get("control")
        if not isinstance(control, dict):
            raise AssertionError("finalists receipt lacks its frozen control")
        item = {"candidate": "frozen30_control", **control}
    elif len(match) == 1:
        item = match[0]
    else:
        raise AssertionError("HPO winner does not identify exactly one guarded finalist")
    fields = [str(value) for value in item["feature_contract"]]
    if len(fields) != len(set(fields)) or not fields:
        raise AssertionError("winner feature contract is empty or duplicated")
    feature_hash = _hash(fields)
    if feature_hash != winner["feature_hash"] or feature_hash != item["feature_contract_sha256"]:
        raise AssertionError("winner feature-contract hash mismatch")
    _write_once(args.out, {
        "schema": SCHEMA,
        "scope": "research-only Router frozen-forward contract; no live or exchange mutation",
        "hpo_winner_source": str(args.hpo_winner.resolve()),
        "finalist_source": str(args.finalists.resolve()),
        "candidate": winner["candidate"],
        "feature_contract": fields,
        "feature_contract_sha256": feature_hash,
        "hpo": {
            "best_trial": winner["best_trial"], "params_json": winner["params_json"],
            "s_stable": winner["s_stable"], "fold_scores_json": winner["fold_scores_json"],
        },
    })


if __name__ == "__main__":
    main()
