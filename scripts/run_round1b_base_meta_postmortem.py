#!/usr/bin/env python3
"""Publish identical-row base/meta increment diagnostics for sealed Round 1."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from extreme_price_movements.candidate_evaluation import paired_day_block_bootstrap, stable_global_top_k


FRACTIONS = (.01, .05, .10, .20)


def _selected(frame: pd.DataFrame, fraction: float) -> pd.DataFrame:
    return stable_global_top_k(frame, "score_bps", fraction)


def _summary(frame: pd.DataFrame) -> dict[str, float | int]:
    return {
        "rows": int(len(frame)),
        "gross_bps": float(frame.execution_gross_ev_12h.mean() * 10_000.0),
        "cost_bps": float(frame.execution_cost_return.mean() * 10_000.0),
        "net_bps": float(frame.execution_net_ev_12h.mean() * 10_000.0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    frame = pd.read_parquet(args.predictions)
    required = {"candidate_id", "target_arm", "model_variant", "score_bps", "__ts__", "side_name", "execution_gross_ev_12h", "execution_cost_return", "execution_net_ev_12h"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"missing columns: {missing}")
    rows: list[dict[str, object]] = []
    attribution: list[dict[str, object]] = []
    boot: list[dict[str, object]] = []
    for arm, group in frame.groupby("target_arm", sort=True, observed=True):
        variants = {name: part.copy() for name, part in group.groupby("model_variant", sort=True, observed=True)}
        if set(variants) != {"base_only", "meta_only", "base_plus_meta"}:
            raise ValueError(f"{arm}: incomplete variants {sorted(variants)}")
        identities = [tuple(sorted(part.candidate_id.astype(str))) for part in variants.values()]
        if len(set(identities)) != 1:
            raise ValueError(f"{arm}: variants do not have identical candidate identities")
        base = variants["base_only"]
        combined = variants["base_plus_meta"]
        for fraction in FRACTIONS:
            selected = {name: _selected(part, fraction) for name, part in variants.items()}
            summaries = {name: _summary(part) for name, part in selected.items()}
            for variant, value in summaries.items():
                rows.append({"target_arm": arm, "top_fraction": fraction, "model_variant": variant, **value})
                for dimension, column in (("side", "side_name"), ("month", "__month__")):
                    local = selected[variant].copy()
                    if dimension == "month":
                        local[column] = pd.to_datetime(local["__ts__"], utc=True).dt.to_period("M").astype(str)
                    for key, part in local.groupby(column, observed=True, sort=True):
                        attribution.append({"target_arm": arm, "top_fraction": fraction, "model_variant": variant, "dimension": dimension, "value": str(key), **_summary(part)})
            boot.append({"target_arm": arm, "top_fraction": fraction, "comparison": "base_plus_meta_minus_base_only", **paired_day_block_bootstrap(selected["base_only"], selected["base_plus_meta"], net_column="execution_net_ev_12h", net_unit="return")})
            rows.append({"target_arm": arm, "top_fraction": fraction, "model_variant": "meta_delta_vs_base", "rows": summaries["base_plus_meta"]["rows"], "gross_bps": summaries["base_plus_meta"]["gross_bps"] - summaries["base_only"]["gross_bps"], "cost_bps": summaries["base_plus_meta"]["cost_bps"] - summaries["base_only"]["cost_bps"], "net_bps": summaries["base_plus_meta"]["net_bps"] - summaries["base_only"]["net_bps"]})
    args.output.mkdir(parents=True)
    pd.DataFrame(rows).to_parquet(args.output / "base_meta_increment.parquet", index=False)
    pd.DataFrame(attribution).to_parquet(args.output / "base_meta_attribution.parquet", index=False)
    pd.DataFrame(boot).to_parquet(args.output / "base_meta_paired_bootstrap.parquet", index=False)
    (args.output / "manifest.json").write_text(json.dumps({"schema": "round1b_base_meta_postmortem_v1", "source": str(args.predictions), "selection": "independently pooled-global top-k on identical candidate rows; paired bootstrap after frozen selection"}, indent=2) + "\n")


if __name__ == "__main__":
    main()
