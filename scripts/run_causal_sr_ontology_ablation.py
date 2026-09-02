#!/usr/bin/env python3
"""Run a bounded, causal ontology-only S/R ablation.

The materialiser is deliberately rerun before fitting any head: merge and
reset choices alter the objects and labels themselves, so reusing V1 events
would be invalid.  Each candidate has immutable source and head outputs.  The
selection metric is interaction-level 2026 OOS stability; downstream MC1 is a
later and separate confirmation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
MATERIALIZER = ROOT / "scripts/materialize_causal_sr_engine.py"
HEADS = ROOT / "scripts/run_causal_sr_heads.py"
DEFAULT_OUTPUT = ROOT / "data_perp/artifacts/causal_sr_ontology_ablation_20260831_v1"

# V1 is represented by the existing immutable source.  These are deliberately
# sparse one-factor constructions, not a combinatorial PnL search.
VARIANTS: dict[str, dict[str, object]] = {
    "S1_precise_levels": {
        "merge_radius_atr": 0.10,
        "touch_radius_atr": 0.10,
    },
    "S2_independent_retests": {
        "reset_distance_atr": 0.50,
        "reset_bars": 4,
        "reset_mode": "or",
    },
    "S3_barrier_12h": {
        "reaction_barriers": [0.50, 1.00, 1.50, 2.00],
        "penetration_barriers": [0.25, 0.50, 1.00],
        "horizon_bars": 48,
        "speed_tau_bars": None,
    },
}


def _run(command: list[str]) -> None:
    print(" ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--variant", choices=tuple(VARIANTS), action="append", help="repeatable; default all")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-symbols", type=int, help="deterministic smoke cap only")
    parser.add_argument("--held-month", action="append", default=["2026-06", "2026-07", "2026-08"], help="repeatable YYYY-MM")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    selected = args.variant or list(VARIANTS)
    records: list[dict[str, object]] = []
    for name in selected:
        ontology_path = output / f"{name}_ontology.json"
        ontology_path.write_text(json.dumps(VARIANTS[name], indent=2, sort_keys=True) + "\n", encoding="utf-8")
        engine_output = output / name / "engine"
        heads_output = output / name / "heads"
        command = [sys.executable, str(MATERIALIZER), "--output", str(engine_output), "--workers", str(args.workers), "--ontology-json", str(ontology_path), "--compact"]
        if args.max_symbols is not None:
            command.extend(("--max-symbols", str(args.max_symbols)))
        _run(command)
        heads_command = [sys.executable, str(HEADS), "--source", str(engine_output), "--output", str(heads_output)]
        for held in args.held_month:
            heads_command.extend(("--held-month", held))
        _run(heads_command)
        records.append({"variant": name, "ontology": VARIANTS[name], "engine": str(engine_output), "heads": str(heads_output)})
    manifest = {
        "schema": "causal-sr-ontology-ablation-v1",
        "scope": "offline only; select on interaction target prediction before downstream economics",
        "held_months": args.held_month,
        "variants": records,
        "causality": "each engine feature is point-in-time; each head trains only on labels resolved before held-month start",
    }
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
