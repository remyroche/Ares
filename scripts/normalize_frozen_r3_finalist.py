#!/usr/bin/env python3
"""Adapt frozen R3 strict OOF + frozen 21-day admission for finalist comparison.

This command performs no fit, target construction, calibration-map refit, or
threshold tuning.  It only verifies immutable source hashes, joins identical
candidate identities, and writes the comparator's canonical column names.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.stage_i_target_adapter import canonical_sha256, file_sha256
from extreme_price_movements.stage_i_target_specific_oos import (
    FrozenR3FinalistInput,
    write_frozen_r3_finalist_normalizer,
)


def _json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict-oof-ledger", type=Path, required=True)
    parser.add_argument("--strict-oof-manifest", type=Path, required=True)
    parser.add_argument("--admission-ledger", type=Path, required=True)
    parser.add_argument("--admission-manifest", type=Path, required=True)
    parser.add_argument("--coverage-audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--finalist-name", default="R3")
    args = parser.parse_args(argv)
    strict_manifest = _json(args.strict_oof_manifest)
    admission_manifest = _json(args.admission_manifest)
    source = FrozenR3FinalistInput(
        strict_oof_predictions=pd.read_parquet(args.strict_oof_ledger),
        admission_predictions=pd.read_parquet(args.admission_ledger),
        coverage_audit=pd.read_parquet(args.coverage_audit),
        strict_oof_manifest=strict_manifest,
        admission_manifest=admission_manifest,
        strict_oof_file_sha256=file_sha256(args.strict_oof_ledger),
        strict_oof_manifest_sha256=file_sha256(args.strict_oof_manifest),
        admission_file_sha256=file_sha256(args.admission_ledger),
        admission_manifest_sha256=file_sha256(args.admission_manifest),
        coverage_audit_sha256=file_sha256(args.coverage_audit),
        strict_oof_artifact_path=args.strict_oof_ledger.name,
        admission_artifact_path=args.admission_ledger.name,
    )
    manifest = write_frozen_r3_finalist_normalizer(
        source=source, output_dir=args.output_dir, finalist_name=str(args.finalist_name),
    )
    print(json.dumps({
        "status": manifest["status"], "schema": manifest["schema"],
        "output_dir": str(args.output_dir.resolve()),
        "operation": manifest["normalizer"],
        "source_lineage_sha256": canonical_sha256(manifest["source_lineage"]),
    }, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
