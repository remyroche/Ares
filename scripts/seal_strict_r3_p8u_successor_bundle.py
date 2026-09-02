#!/usr/bin/env python3
"""Create an immutable hash-bound successor of a sealed P8U bundle.

This deliberately small sealer copies no model files.  It refreshes selected
artifact identities from their exact on-disk content, records a source-lineage
note, and optionally changes the execution authority only when explicitly
requested.  It is useful when operational guard code or the sealed transform
state changes while Router/Base/Under/MC1 model artefacts remain frozen.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.inference.p8u_production_contract import artifact_hash


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _parse_replace(value: str) -> tuple[str, str, Path]:
    try:
        role, kind, raw_path = value.split(":", 2)
    except ValueError as error:
        raise argparse.ArgumentTypeError("replacement must be ROLE:TYPE:PATH") from error
    if kind not in {"file", "tree"}:
        raise argparse.ArgumentTypeError("replacement TYPE must be file or tree")
    path = Path(raw_path)
    if not role or not raw_path:
        raise argparse.ArgumentTypeError("replacement role and path must be non-empty")
    return role, kind, path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--replace", action="append", default=[], type=_parse_replace)
    parser.add_argument("--source-lineage-note", required=True)
    parser.add_argument("--activate-live", action="store_true")
    args = parser.parse_args()

    parent = args.parent.resolve()
    out = args.out.resolve()
    if not parent.is_file():
        raise FileNotFoundError(parent)
    if out.exists():
        raise FileExistsError(f"immutable successor bundle exists: {out}")
    if ROOT not in parent.parents or ROOT not in out.parents:
        raise ValueError("P8U bundle paths must remain below repository root")
    payload = json.loads(parent.read_text())
    if not isinstance(payload, dict) or payload.get("schema") != "strict_r3_p8u_preproduction_bundle_v1":
        raise ValueError("parent is not a supported P8U bundle")
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("parent P8U bundle lacks artifact mapping")

    successor = copy.deepcopy(payload)
    for role, kind, supplied in args.replace:
        path = supplied.resolve()
        if ROOT not in path.parents or not path.exists():
            raise FileNotFoundError(f"replacement {role} must exist below repository root: {path}")
        successor["artifacts"][role] = {
            "path": path.relative_to(ROOT).as_posix(),
            "sha256": artifact_hash(path, kind),
            "type": kind,
        }

    runtime = successor.setdefault("runtime", {})
    feature_execution = runtime.setdefault("feature_execution", {})
    feature_execution["feature_coverage_requirement"] = (
        "the active point-in-time panel must materialise every sealed Router/Base/Under "
        "field; wholly unavailable fields fail before scoring and row-local missingness "
        "is recorded before sealed train-time imputation"
    )
    runtime["source_lineage_note"] = str(args.source_lineage_note)
    runtime["successor_of"] = {
        "path": parent.relative_to(ROOT).as_posix(),
        "sha256": artifact_hash(parent, "file"),
    }
    if args.activate_live:
        runtime["order_submission"] = True
        runtime["promotion_status"] = "approved_live"
        runtime["blockers"] = [
            value
            for value in runtime.get("blockers", [])
            if "intentionally have no exchange" not in str(value)
            and "untouched prospective" not in str(value)
        ]
    else:
        runtime["order_submission"] = False
        runtime["promotion_status"] = "blocked_preproduction"
    _atomic_json(out, successor)
    print(json.dumps({
        "status": "sealed_p8u_successor_bundle",
        "bundle": str(out),
        "bundle_sha256": artifact_hash(out, "file"),
        "replaced_roles": [role for role, _kind, _path in args.replace],
        "order_submission": runtime["order_submission"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
