#!/usr/bin/env python3
"""Materialize compact, causal current-lineage health by decision hour."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from extreme_price_movements.regime_transition_current_model_health import (
    CURRENT_MODEL_HEALTH_COLUMNS,
    build_hourly_current_model_health,
)


DEFAULT_LEDGER = Path("data_perp/artifacts/failure_first_detector_current_transfer_20260726_v5/candidate_overlay.parquet")
DEFAULT_HANDOFF = Path("data_perp/artifacts/execution_ev_repaired_heads_representation_handoff_20260726_v7/joined.parquet")
DEFAULT_OUTPUT = Path("data_perp/artifacts/regime_transition_current_model_health_20260727_v1")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mapping-ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--current-handoff", type=Path, default=DEFAULT_HANDOFF)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser


def run(args: argparse.Namespace) -> dict[str, object]:
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True)
    health, report = build_hourly_current_model_health(
        pd.read_parquet(args.mapping_ledger), pd.read_parquet(args.current_handoff)
    )
    health.to_parquet(output / "hourly_model_health.parquet", index=False)
    pd.DataFrame({"feature": CURRENT_MODEL_HEALTH_COLUMNS}).to_csv(
        output / "field_catalog.csv", index=False
    )
    report["mapping_ledger"] = str(Path(args.mapping_ledger).resolve())
    report["current_handoff"] = str(Path(args.current_handoff).resolve())
    (output / "manifest.json").write_text(json.dumps(report, indent=2, default=str, sort_keys=True) + "\n")
    return report


def main() -> None:
    print(json.dumps(run(_parser().parse_args()), indent=2, default=str, sort_keys=True))


if __name__ == "__main__":
    main()
