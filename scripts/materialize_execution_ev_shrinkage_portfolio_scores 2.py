#!/usr/bin/env python3
"""Materialize one weekly-forward shrinkage arm for exact portfolio replay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


IDENTITY = ("__ts__", "__symbol__", "side_name", "candidate_id")


def run(args: argparse.Namespace) -> dict[str, object]:
    frame = pd.read_parquet(args.predictions)
    selected = frame.loc[frame["arm"].astype(str).eq(args.arm)].copy()
    if selected.empty:
        raise ValueError(f"no predictions for arm {args.arm!r}")
    if selected.duplicated(list(IDENTITY)).any():
        raise ValueError("weekly-forward score identities are not one-to-one")
    if selected["promotion_eligible"].astype(bool).any():
        raise ValueError("research weekly OOS rows must remain non-promotable")
    score = pd.to_numeric(
        selected["prediction_canonical_recent_ev_mapping"], errors="raise"
    )
    if score.isna().any():
        raise ValueError("portfolio score contains nulls")
    weeks = {
        week: number
        for number, week in enumerate(sorted(selected["week"].astype(str).unique()))
    }
    output = selected.loc[:, list(IDENTITY)].copy()
    output[args.score_col] = score.to_numpy(dtype=float)
    output[f"{args.score_col}__is_oof"] = True
    output["execution_ev_model_ablation_oof_fold"] = (
        selected["week"].astype(str).map(weeks).to_numpy(dtype=int)
    )
    output["weekly_forward_fold"] = selected["week"].astype(str).to_numpy()
    output["promotion_eligible"] = False
    args.output_dir.mkdir(parents=True, exist_ok=False)
    output.to_parquet(args.output_dir / "portfolio_oof_scores.parquet", index=False)
    manifest = {
        "schema": "execution_ev_shrinkage_portfolio_scores_v1",
        "source": str(args.predictions),
        "arm": args.arm,
        "score_col": args.score_col,
        "rows": int(len(output)),
        "weeks": weeks,
        "selection_contract": (
            "score is canonical recent-EV mapped weekly forward OOS; downstream "
            "performs one pooled global top10, never per timestamp"
        ),
        "promotion_eligible": False,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--arm", default="shrink_0p50")
    parser.add_argument(
        "--score-col", default="weekly_forward_shrink_0p50_recent_ev_score"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    payload = run(parser.parse_args())
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
