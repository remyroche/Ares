#!/usr/bin/env python3
"""Materialise canonical Stage-I side-local causal 21-day EV admission.

This is a post-score component: it never refits a base/residual model, changes
the candidate population, or ranks by timestamp/side.  It maps scores with
prior-resolved side-local observations, applies the 50-bps expected-net floor,
then reports pooled-global rankings before and after admission.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.stage_i_causal_admission import (
    Causal21dAdmissionSpec,
    apply_causal_21d_side_admission,
    pooled_global_admission_comparison,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--score-column", required=True)
    parser.add_argument("--net-column", default="net_bps")
    parser.add_argument("--decision-column", default="__ts__")
    parser.add_argument("--label-available-column", default="label_available_ts")
    parser.add_argument("--identity-column", default="candidate_id")
    parser.add_argument("--min-reference-rows", type=int, default=500)
    parser.add_argument("--top-fraction", action="append", type=float, default=[])
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite admission artifact: {args.out}")
    source = pd.read_parquet(args.input)
    spec = Causal21dAdmissionSpec(min_reference_rows=args.min_reference_rows)
    mapped, audit = apply_causal_21d_side_admission(
        source, score_column=args.score_column, net_column=args.net_column,
        decision_column=args.decision_column, label_available_column=args.label_available_column,
        identity_column=args.identity_column, spec=spec,
    )
    metrics = pooled_global_admission_comparison(
        mapped, raw_score_column=args.score_column, net_column=args.net_column,
        identity_column=args.identity_column, top_fractions=tuple(args.top_fraction or (.01, .05, .10)),
    )
    if len(mapped) != len(source) or not mapped[args.identity_column].equals(source[args.identity_column]):
        raise AssertionError("admission must retain every candidate in original order")
    args.out.mkdir(parents=True)
    mapped.to_parquet(args.out / "candidates_with_causal_21d_admission.parquet", index=False, compression="zstd")
    audit.to_parquet(args.out / "causal_21d_admission_audit.parquet", index=False, compression="zstd")
    metrics.to_parquet(args.out / "pooled_global_admission_comparison.parquet", index=False, compression="zstd")
    manifest = {
        "schema": "stage_i_causal_21d_side_admission_v1",
        "status": "MATERIALIZED_DIAGNOSTIC_NO_POLICY_PROMOTION",
        "input": str(args.input),
        "contract": {
            "mapping": "side-local 5% trimmed equal-frequency-bin conditional net-bps means + increasing isotonic",
            "reference": "full trailing 21 calendar days; exact label_available_ts strictly before snapshot",
            "support": f"fail closed below {spec.min_reference_rows} rows per side; no pooled fallback",
            "admission": "mapped common expected net >= 50 bps",
            "ranking": "pooled global only after admission; no side/timestamp top-k",
        },
        "rows": {
            "input": int(len(source)), "mapped": int(mapped.causal_21d_side_expected_net_bps.notna().sum()),
            "admitted": int(mapped.causal_21d_side_admitted_ge_50bps.sum()),
        },
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
