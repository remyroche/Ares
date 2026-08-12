#!/usr/bin/env python3
"""Build the TP6 F0/F3 candidate panel and contracts for standalone F4 MDA.

This command only materialises the development panel.  It does not fit a base
model, invoke the F4 evidence callback, or evaluate November 2024 final OOS.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.feature_portability_f4_panel import (  # noqa: E402
    materialize_tp6_f4_candidate_panel,
    write_tp6_f4_candidate_panel,
)
from extreme_price_movements.tp6_portability_data import TP6PortabilityContract  # noqa: E402


def _read_dispositions(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            payload = payload.get("rows", payload.get("dispositions", payload))
        if not isinstance(payload, list):
            raise ValueError("portability disposition JSON must contain a rows/dispositions list")
        return pd.DataFrame(payload)
    raise ValueError("portability dispositions must be parquet, csv, or JSON")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Materialise the TP6 development F0/F3 panel for F4")
    parser.add_argument("--panel-dir", type=Path, required=True)
    parser.add_argument("--winner-dir", type=Path, required=True)
    parser.add_argument("--robust-dir", type=Path, required=True)
    parser.add_argument("--base-feature-manifest-dir", type=Path, required=True)
    parser.add_argument("--portability-dispositions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    result = materialize_tp6_f4_candidate_panel(
        contract=TP6PortabilityContract(
            panel=args.panel_dir, winner=args.winner_dir, robust=args.robust_dir,
            feature_manifest=args.base_feature_manifest_dir,
        ),
        portability_dispositions=_read_dispositions(args.portability_dispositions),
    )
    write_tp6_f4_candidate_panel(result, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
