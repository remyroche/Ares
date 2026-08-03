#!/usr/bin/env python3
"""Materialize a descriptive transition diagnostic from accepted base-only OOF."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.base_oof_transition_diagnostic import build_transition_diagnostic


DEFAULT_BASE = ROOT / "data_perp/artifacts/febapr2025_canonical_base_oof_20260727_v1/oof_predictions.parquet"
DEFAULT_WINDOWS = ROOT / "data_perp/artifacts/febapr2025_canonical_exact_policy_base_population_20260727_v2/transition_event_windows.parquet"
DEFAULT_ACTIVE = ROOT / "data_perp/artifacts/regime_transition_active_head_20260726_v1/grouped_oof.parquet"
DEFAULT_OUTPUT = ROOT / "data_perp/artifacts/febapr2025_base_oof_transition_diagnostic_20260727_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, (pd.Timestamp, Path)):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-oof", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--windows", type=Path, default=DEFAULT_WINDOWS)
    parser.add_argument("--active-hours", type=Path, default=DEFAULT_ACTIVE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    coverage, metrics, summary = build_transition_diagnostic(
        pd.read_parquet(args.base_oof), pd.read_parquet(args.windows), pd.read_parquet(args.active_hours)
    )
    args.output_dir.mkdir(parents=True)
    coverage.to_parquet(args.output_dir / "event_coverage.parquet", index=False)
    metrics.to_parquet(args.output_dir / "event_phase_side_metrics.parquet", index=False)
    summary["sources"] = {
        "accepted_base_oof": {"path": str(args.base_oof), "sha256": _sha256(args.base_oof)},
        "frozen_event_windows": {"path": str(args.windows), "sha256": _sha256(args.windows)},
        "expost_active_hour_ledger": {"path": str(args.active_hours), "sha256": _sha256(args.active_hours)},
    }
    summary["outputs"] = {
        "event_coverage": _sha256(args.output_dir / "event_coverage.parquet"),
        "event_phase_side_metrics": _sha256(args.output_dir / "event_phase_side_metrics.parquet"),
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(_safe(summary), indent=2, sort_keys=True) + "\n")
    print(json.dumps(_safe(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
