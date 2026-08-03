#!/usr/bin/env python3
"""Materialize strict-OOF EV challenger scores for common policy replay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


IDENTITY = ("__ts__", "__symbol__", "side_name", "candidate_id")
DEFAULT_SCORES = (
    "direct_net_ev__hierarchical_oof__causal_recent_ev",
    "direct_primary_auxiliary_oof_blend_net_ev__hierarchical_oof__causal_recent_ev",
    "direct_primary_shared_multitask_oof_net_ev__hierarchical_oof__causal_recent_ev",
)


def materialize(frame: pd.DataFrame, score_columns: list[str]) -> pd.DataFrame:
    required = {*IDENTITY, "oof_fold", *score_columns}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"multi-task OOF table missing columns: {missing}")
    if frame.duplicated(list(IDENTITY)).any():
        raise ValueError("multi-task OOF identity must be one-to-one")
    fold = pd.to_numeric(frame["oof_fold"], errors="coerce")
    eligible = np.isfinite(fold.to_numpy(dtype=float))
    output = frame.loc[eligible, list(IDENTITY)].copy()
    output["execution_ev_model_ablation_oof_fold"] = (
        fold.loc[eligible].astype(int).to_numpy()
    )
    for column in score_columns:
        score = pd.to_numeric(frame.loc[eligible, column], errors="coerce")
        if not np.isfinite(score.to_numpy(dtype=float)).all():
            raise ValueError(f"eligible OOF score {column!r} is non-finite")
        output[column] = score.to_numpy(dtype=float)
        output[f"{column}__is_oof"] = True
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--score",
        action="append",
        dest="scores",
        help="Score column; repeat for multiple arms.",
    )
    args = parser.parse_args()
    scores = list(args.scores or DEFAULT_SCORES)
    output = materialize(pd.read_parquet(args.input), scores)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    path = args.output_dir / "portfolio_oof_scores.parquet"
    output.to_parquet(path, index=False)
    manifest = {
        "schema": "execution_ev_multitask_portfolio_scores_v1",
        "source": str(args.input),
        "rows": int(len(output)),
        "score_columns": scores,
        "selection_contract": (
            "strict outer-OOF scores only; downstream performs one pooled "
            "global top10 before portfolio constraints"
        ),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
