#!/usr/bin/env python3
"""Materialise a separate supportive-target sidecar from frozen path labels.

This command never overwrites the source target pack and does not train a
model.  The result remains research-only until a later strict-OOF head runner
consumes its labels with the documented masks and censoring semantics.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

from extreme_price_movements.supportive_target_semantics import (
    DEFAULT_HAZARD_BOUNDARIES_HOURS,
    DEFAULT_HORIZON_HOURS,
    REQUIRED_SOURCE_COLUMNS,
    SCHEMA,
    materialize_supportive_target_semantics,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "data_perp/artifacts/root_cause_exact_h12_execution_target_pack_20260801_v2/supportive_labels.parquet"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_boundaries(value: str) -> tuple[float, ...]:
    try:
        return tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("hazard boundaries must be comma-separated numbers") from error


def run(
    *,
    source: Path = DEFAULT_SOURCE,
    output: Path,
    horizon_hours: float = DEFAULT_HORIZON_HOURS,
    hazard_boundaries_hours: tuple[float, ...] = DEFAULT_HAZARD_BOUNDARIES_HOURS,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite artifact: {output}")
    source_resolved = source.resolve()
    output_resolved = output.resolve()
    if source_resolved == output_resolved or source_resolved in output_resolved.parents:
        raise ValueError("output must be a distinct sibling/new artifact, never the frozen source pack")
    # The source pack has many diagnostic labels.  Read only the declared
    # columns required for this compact target sidecar.
    frame = pd.read_parquet(source, columns=list(REQUIRED_SOURCE_COLUMNS))
    labels, contract = materialize_supportive_target_semantics(
        frame,
        horizon_hours=horizon_hours,
        hazard_boundaries_hours=hazard_boundaries_hours,
    )
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        labels.to_parquet(stage / "supportive_target_semantics.parquet", index=False, compression="zstd")
        manifest = {
            **contract,
            "status": "MATERIALIZED_RESEARCH_LABEL_SIDECAR_ONLY",
            "source": str(source),
            "source_sha256": _sha256(source),
            "rows": int(len(labels)),
            "columns": list(labels.columns),
            "outputs_sha256": {"supportive_target_semantics.parquet": _sha256(stage / "supportive_target_semantics.parquet")},
            "runner": {"path": str(Path(__file__).relative_to(ROOT)), "sha256": _sha256(Path(__file__))},
        }
        (stage / "supportive_target_semantics_contract.json").write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (stage / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(stage, output)
        return manifest
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--horizon-hours", type=float, default=DEFAULT_HORIZON_HOURS)
    parser.add_argument(
        "--hazard-boundaries-hours",
        type=_parse_boundaries,
        default=DEFAULT_HAZARD_BOUNDARIES_HOURS,
        help="Comma-separated interval ends; must end at horizon (default: 1,2,4,8,12).",
    )
    args = parser.parse_args()
    print(json.dumps(run(
        source=args.source,
        output=args.output,
        horizon_hours=args.horizon_hours,
        hazard_boundaries_hours=args.hazard_boundaries_hours,
    ), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
