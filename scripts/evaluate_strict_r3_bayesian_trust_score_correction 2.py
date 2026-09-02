#!/usr/bin/env python3
"""Evaluate a causal Bayesian trust posterior as a bounded score correction.

The input consists exclusively of strict-prequential posterior predictions
already emitted by ``run_strict_r3_binned_bayes_feature_compare.py``.  This
script performs no fitting and never reads a held outcome when constructing a
score.  It differs from the historical Bayesian sizing use: rather than
rescaling a position multiplier near its cap, it reorders only the top 30%% of
candidates within their current decision timestamp.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


TOP_FRACTION = 0.30
TAILS = (0.005, 0.01, 0.02, 0.05, 0.10)


def _top30_and_quality_rank(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    ordered = frame.sort_values(
        ["__decision_ts__", "final_score", "candidate_id"],
        ascending=[True, False, True], kind="stable",
    ).copy()
    position = ordered.groupby("__decision_ts__", sort=False).cumcount().to_numpy()
    count = ordered.groupby("__decision_ts__", sort=False)["candidate_id"].transform("size").to_numpy()
    top30 = position < np.maximum(1, np.ceil(count * TOP_FRACTION).astype(int))
    quality = (
        pd.to_numeric(ordered["posterior_expected_rank_train"], errors="coerce").fillna(0.5)
        - 0.5 * pd.to_numeric(ordered["posterior_adverse_rank_train"], errors="coerce").fillna(0.5)
    )
    ranked = ordered.loc[top30, ["__decision_ts__", "candidate_id"]].copy()
    ranked["__quality__"] = quality.loc[top30].to_numpy(float)
    ranked = ranked.sort_values(
        ["__decision_ts__", "__quality__", "candidate_id"],
        ascending=[True, True, True], kind="stable",
    )
    ranked["__rank__"] = ranked.groupby("__decision_ts__", sort=False).cumcount() + 1
    ranked["__count__"] = ranked.groupby("__decision_ts__", sort=False)["candidate_id"].transform("size")
    ranked["__quality_percentile__"] = ranked["__rank__"] / ranked["__count__"]
    merged = ordered.merge(
        ranked[["candidate_id", "__quality_percentile__"]], on="candidate_id", how="left", validate="one_to_one",
    )
    inverse = pd.Series(np.arange(len(ordered)), index=ordered.index)
    original_position = inverse.loc[frame.index].to_numpy()
    return top30[original_position], merged["__quality_percentile__"].fillna(0.5).to_numpy(float)[original_position]


def _metrics(frame: pd.DataFrame, *, arm: str, kind: str) -> list[dict[str, object]]:
    if kind == "global":
        groups = [("all", frame)]
    elif kind == "month":
        groups = [(str(value), part) for value, part in frame.groupby(frame["__decision_ts__"].dt.strftime("%Y-%m"), sort=True)]
    elif kind == "week":
        groups = [(str(value), part) for value, part in frame.groupby(frame["__decision_ts__"].dt.strftime("%G-W%V"), sort=True)]
    else:  # pragma: no cover
        raise ValueError(kind)
    output: list[dict[str, object]] = []
    for period, part in groups:
        ranked = part.sort_values(["corrected_score", "candidate_id"], ascending=[False, True], kind="stable")
        for tail in TAILS:
            selected = ranked.head(max(1, int(np.ceil(len(ranked) * tail))))
            valid = selected.loc[selected["policy_path_valid"].fillna(False).astype(bool)].copy()
            net = pd.to_numeric(valid["policy_net_bps"], errors="coerce")
            net = net[np.isfinite(net)]
            output.append({
                "arm": arm, "period_kind": kind, "period": period, "tail": tail,
                "selected_score_rows": int(len(selected)), "valid_outcomes": int(len(net)),
                "outcome_coverage": float(len(net) / max(len(selected), 1)),
                "net_bps_per_trade": float(net.mean()) if len(net) else np.nan,
                "positive_rate": float(net.gt(0.0).mean()) if len(net) else np.nan,
            })
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--arms", default="B1_current_trust_overlay,B3_current_trust_overlay,B5_current_trust_overlay")
    parser.add_argument("--alphas", default="0.025,0.05,0.10,0.20")
    parser.add_argument(
        "--include-final-score-control", action="store_true",
        help="Emit the unmodified final_score from the same Bayesian arm as a matched control.",
    )
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(f"immutable output exists: {args.out_dir}")
    requested = tuple(token.strip() for token in str(args.arms).split(",") if token.strip())
    alphas = tuple(float(token.strip()) for token in str(args.alphas).split(",") if token.strip())
    if not requested or not alphas or any(value <= 0.0 or value > 0.20 for value in alphas):
        raise ValueError("arms must be nonempty and alphas must be in (0, 0.20]")
    frame = pd.read_parquet(args.predictions)
    required = {
        "arm", "candidate_id", "__decision_ts__", "final_score", "policy_path_valid", "policy_net_bps",
        "posterior_expected_rank_train", "posterior_adverse_rank_train",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise KeyError(f"predictions missing {missing}")
    frame["__decision_ts__"] = pd.to_datetime(frame["__decision_ts__"], utc=True, errors="raise")
    args.out_dir.mkdir(parents=True)
    all_predictions: list[pd.DataFrame] = []
    rows: list[dict[str, object]] = []
    for arm in requested:
        part = frame.loc[frame["arm"].eq(arm)].copy()
        if part.empty:
            raise ValueError(f"requested Bayesian arm absent: {arm}")
        if part["candidate_id"].duplicated().any():
            raise AssertionError(f"{arm} duplicates candidate identities")
        top30, quality_percentile = _top30_and_quality_rank(part)
        base = pd.to_numeric(part["final_score"], errors="coerce").fillna(0.0).to_numpy(float)
        if args.include_final_score_control:
            control = part.loc[:, [
                "candidate_id", "__decision_ts__", "side_name", "policy_label_available_ts",
                "policy_path_valid", "policy_net_bps", "final_score",
                "posterior_expected_rank_train", "posterior_adverse_rank_train",
            ]].copy()
            control["timestamp_top30"] = top30
            control["bayesian_quality_percentile"] = quality_percentile.astype(np.float32)
            control["corrected_score"] = base
            control["arm"] = f"{arm}_final_score_control"
            all_predictions.append(control)
            for kind in ("global", "month", "week"):
                rows.extend(_metrics(control, arm=str(control["arm"].iloc[0]), kind=kind))
        for alpha in alphas:
            output = part.loc[:, [
                "candidate_id", "__decision_ts__", "side_name", "policy_label_available_ts",
                "policy_path_valid", "policy_net_bps", "final_score",
                "posterior_expected_rank_train", "posterior_adverse_rank_train",
            ]].copy()
            output["timestamp_top30"] = top30
            output["bayesian_quality_percentile"] = quality_percentile.astype(np.float32)
            output["corrected_score"] = base + float(alpha) * np.where(top30, quality_percentile - 0.5, 0.0)
            output["arm"] = f"{arm}_scorecorr_a{alpha:g}"
            all_predictions.append(output)
            for kind in ("global", "month", "week"):
                rows.extend(_metrics(output, arm=str(output["arm"].iloc[0]), kind=kind))
    predictions = pd.concat(all_predictions, ignore_index=True)
    metrics = pd.DataFrame(rows)
    predictions.to_parquet(args.out_dir / "predictions.parquet", index=False, compression="zstd")
    metrics.to_parquet(args.out_dir / "metrics.parquet", index=False)
    manifest = {
        "schema": "strict_r3_bayesian_trust_timestamp_score_correction_v1",
        "source_predictions": str(args.predictions),
        "arms": list(requested), "alphas": list(alphas),
        "include_final_score_control": bool(args.include_final_score_control),
        "integration": (
            "for candidates in the current timestamp's base-score top 30%, final_score + alpha * "
            "(within-timestamp percentile of [posterior_expected_rank_train - 0.5 * posterior_adverse_rank_train], minus 0.5); "
            "other candidates unchanged"
        ),
        "causality": (
            "Bayesian posterior is strict prequential from source; timestamp percentile uses only contemporaneous "
            "predictions; held outcomes are read only after score construction for evaluation"
        ),
        "position_sizing": "unchanged; this does not alter the trust multiplier, cap, admission, or policy",
    }
    (args.out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"event": "complete", "out_dir": str(args.out_dir), "rows": int(len(predictions))}))


if __name__ == "__main__":
    main()
