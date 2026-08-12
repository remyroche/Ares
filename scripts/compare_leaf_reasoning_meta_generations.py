#!/usr/bin/env python3
"""Validate the linked-spec S0/S1/S2 successor-meta comparison.

Inputs are sealed, development-only Parquet outputs emitted by
``write_immutable_meta_funnel_output``.  This utility never fits a model,
derives predecessor features, or opens final OOS; it only compares the exact
matched transport/top-tail cells and writes one atomic comparison artifact.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.leaf_reasoning_meta_funnel import (  # noqa: E402
    MetaTransportGateConfig,
    compare_successor_meta_generations,
)


COMPARISON_SCHEMA = "leaf_reasoning_meta_generation_comparison_v2"
COMPLETE_ARTIFACT_STATE = "COMPLETE"


@dataclass(frozen=True)
class ImmutableGenerationRun:
    """Verified source table plus immutable provenance for one S-generation."""

    root: Path
    metrics: pd.DataFrame
    manifest: dict[str, Any]
    manifest_sha256: str
    metrics_sha256: str
    metrics_path: Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_object(path: Path) -> dict[str, Any]:
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid immutable meta-funnel manifest: {path}") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"immutable meta-funnel manifest must be a JSON object: {path}")
    return parsed


def _read_generation(
    root: Path,
    expected_successor: str,
    *,
    allow_legacy_csv: bool = False,
) -> ImmutableGenerationRun:
    """Load one hash-verified immutable funnel result.

    Modern funnel output is Parquet-only.  The explicit legacy switch is kept
    solely for already-sealed historical research artifacts and still requires
    an immutable manifest plus a declared matching file hash; it is never an
    implicit CSV fallback.
    """

    root = Path(root)
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"{root} is not a complete immutable meta-funnel output")
    manifest = _json_object(manifest_path)
    if manifest.get("immutable_output") is not True:
        raise ValueError(f"{root} is not marked immutable")
    if manifest.get("successor") != expected_successor:
        raise ValueError(f"{root} declares successor {manifest.get('successor')!r}, expected {expected_successor}")
    modern = (
        manifest.get("artifact_state") == COMPLETE_ARTIFACT_STATE
        and manifest.get("table_format") == "parquet_zstd"
    )
    if modern:
        metrics_path = root / "metrics.parquet"
    elif allow_legacy_csv:
        metrics_path = root / "metrics.csv"
    else:
        raise ValueError(
            f"{root} is not a complete immutable Parquet meta-funnel output; "
            "pass --allow-legacy-csv only for a hash-sealed historical CSV artifact"
        )
    if not metrics_path.is_file():
        raise ValueError(f"immutable meta-funnel output lacks {metrics_path.name}: {root}")
    hashes = manifest.get("sha256")
    expected_hash = hashes.get(metrics_path.name) if isinstance(hashes, dict) else None
    if not isinstance(expected_hash, str) or len(expected_hash) != 64:
        raise ValueError(f"immutable meta-funnel manifest lacks a valid {metrics_path.name} hash: {root}")
    actual_hash = _sha256(metrics_path)
    if actual_hash != expected_hash:
        raise ValueError(f"immutable meta-funnel {metrics_path.name} hash mismatch: {root}")
    table = pd.read_parquet(metrics_path) if metrics_path.suffix == ".parquet" else pd.read_csv(metrics_path)
    return ImmutableGenerationRun(
        root=root.resolve(), metrics=table, manifest=manifest,
        manifest_sha256=_sha256(manifest_path), metrics_sha256=actual_hash,
        metrics_path=metrics_path.resolve(),
    )


def write_immutable_generation_comparison(
    table: pd.DataFrame,
    *,
    output_dir: Path,
    sources: dict[str, ImmutableGenerationRun],
    selected_arms: dict[str, str],
    required_transport_count: int,
) -> Path:
    """Atomically publish the comparison and hash-bind it to all three runs."""

    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite successor comparison: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        comparison_path = temporary / "meta_generation_comparison.parquet"
        table.to_parquet(comparison_path, index=False, compression="zstd")
        comparison_hash = _sha256(comparison_path)
        manifest = {
            "schema": COMPARISON_SCHEMA,
            "immutable_output": True,
            "artifact_state": COMPLETE_ARTIFACT_STATE,
            "table_format": "parquet_zstd",
            "selected_arms": dict(selected_arms),
            "required_transport_count": int(required_transport_count),
            "terminal_decision": str(table["terminal_decision"].iloc[0]),
            "sources": {
                generation: {
                    "root": str(source.root),
                    "manifest_sha256": source.manifest_sha256,
                    "metrics_path": str(source.metrics_path),
                    "metrics_sha256": source.metrics_sha256,
                    "successor": source.manifest.get("successor"),
                    "development_only": str(source.manifest.get("selection_status", "")).startswith("DEVELOPMENT_"),
                }
                for generation, source in sorted(sources.items())
            },
            "sha256": {comparison_path.name: comparison_hash},
        }
        manifest_path = temporary / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (temporary / "manifest.sha256").write_text(
            f"{_sha256(manifest_path)}  manifest.json\n", encoding="utf-8"
        )
        # A same-filesystem rename makes a partial comparison invisible.
        os.replace(temporary, output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for generation in ("s0", "s1", "s2"):
        parser.add_argument(f"--{generation}-run", required=True, type=Path)
        parser.add_argument(f"--{generation}-arm", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--required-transport-count", type=int, default=2)
    parser.add_argument(
        "--allow-legacy-csv", action="store_true",
        help="explicitly allow a hash-sealed historical immutable metrics.csv input; modern outputs must be Parquet",
    )
    args = parser.parse_args()

    selected_arms = {"S0": args.s0_arm, "S1": args.s1_arm, "S2": args.s2_arm}
    sources = {
        "S0": _read_generation(args.s0_run, "S0", allow_legacy_csv=args.allow_legacy_csv),
        "S1": _read_generation(args.s1_run, "S1", allow_legacy_csv=args.allow_legacy_csv),
        "S2": _read_generation(args.s2_run, "S2", allow_legacy_csv=args.allow_legacy_csv),
    }
    table = compare_successor_meta_generations(
        {generation: source.metrics for generation, source in sources.items()},
        selected_arm_by_generation=selected_arms,
        gate_config=MetaTransportGateConfig(required_transport_count=args.required_transport_count),
    )
    output = write_immutable_generation_comparison(
        table,
        output_dir=args.output_dir,
        sources=sources,
        selected_arms=selected_arms,
        required_transport_count=args.required_transport_count,
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
