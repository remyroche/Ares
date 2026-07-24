#!/usr/bin/env python3
"""Materialize frozen OOS EBM breakout-path probabilities as meta-context fields.

This is a research handoff only. It does not alter a meta model, rank, policy,
or live inference schema. Every probability comes from a model fitted before
the recorded OOS fold; reliability is entirely pre-entry.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


KEYS = ("__ts__", "__symbol__", "candidate_id", "side_name", "__archetype_policy_key__", "fold_start")
TARGETS = ("rapid_reversal", "severe_retention")


def run(args: argparse.Namespace) -> dict[str, object]:
    raw = pd.concat(
        [pd.read_parquet(path) for path in args.predictions], ignore_index=True, copy=False
    )
    required = {*KEYS, "target", "model", "prediction", "probability_reliability"}
    missing = sorted(required.difference(raw.columns))
    if missing:
        raise KeyError(f"Prediction artifact is missing fields required for context materialization: {missing}")
    rows = raw.loc[
        raw["model"].eq("ebm") & raw["target"].isin(TARGETS),
        [*KEYS, "target", "prediction", "probability_reliability", "feature_distribution_reliability",
         "score_bin_support", "model_disagreement"],
    ].copy()
    if rows.duplicated([*KEYS, "target"]).any():
        raise ValueError("EBM path probabilities are not unique on the declared prediction key")
    probability = rows.pivot(index=list(KEYS), columns="target", values="prediction")
    reliability = rows.pivot(index=list(KEYS), columns="target", values="probability_reliability")
    output = probability.reset_index()
    output["breakout_rapid_reversal_probability_ebm"] = probability["rapid_reversal"].to_numpy("float32")
    output["breakout_severe_retention_probability_ebm"] = probability["severe_retention"].to_numpy("float32")
    output["breakout_rapid_reversal_probability_reliability"] = reliability["rapid_reversal"].to_numpy("float32")
    output["breakout_severe_retention_probability_reliability"] = reliability["severe_retention"].to_numpy("float32")
    output = output.loc[:, [
        *KEYS,
        "breakout_rapid_reversal_probability_ebm",
        "breakout_rapid_reversal_probability_reliability",
        "breakout_severe_retention_probability_ebm",
        "breakout_severe_retention_probability_reliability",
    ]]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(args.output, index=False, compression="zstd")
    manifest = {
        "schema": "breakout_path_quality_context_v1",
        "status": "diagnostic_context_only_not_activated_in_meta_or_policy",
        "source_predictions": [str(path) for path in args.predictions],
        "scope": {"side": "short", "archetype_policy_key": "short_breakout_precision"},
        "rows": int(len(output)),
        "keys": list(KEYS),
        "fields": output.columns[len(KEYS):].tolist(),
        "reliability_contract": (
            "Reliability uses only train-derived score-bin support, robust feature-distribution "
            "proximity, and cross-model agreement. It contains no OOS outcomes or post-hoc "
            "calibration diagnostics."
        ),
        "leakage_contract": (
            "Each EBM probability was fitted before its outer OOS fold with an eight-hour "
            "label-horizon purge. This artifact is an OOS diagnostic handoff, not a live feature "
            "contract until a new meta model is trained with the same fields."
        ),
    }
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True, action="append")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
