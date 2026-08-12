#!/usr/bin/env python3
"""Evaluate a Bayesian trust score correction *after* frozen EV admission.

This is deliberately an overlay, not an EV-map experiment.  Admission is
copied unchanged from a causal control score's prequential 21-day EV map;
the Bayesian correction only reorders the already-admitted candidates.
Outcomes are read solely after this ranking for evaluation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


TAILS = (0.005, 0.01, 0.02, 0.05, 0.10)


def _metrics(frame: pd.DataFrame, score_col: str, arm: str, kind: str) -> list[dict[str, object]]:
    if kind == "global":
        groups = [("all", frame)]
    elif kind == "month":
        groups = [(str(key), part) for key, part in frame.groupby(frame["__decision_ts__"].dt.strftime("%Y-%m"), sort=True)]
    elif kind == "week":
        groups = [(str(key), part) for key, part in frame.groupby(frame["__decision_ts__"].dt.strftime("%G-W%V"), sort=True)]
    else:  # pragma: no cover
        raise ValueError(kind)
    rows: list[dict[str, object]] = []
    for period, part in groups:
        ranked = part.sort_values([score_col, "candidate_id"], ascending=[False, True], kind="stable")
        for tail in TAILS:
            selected = ranked.head(max(1, int(np.ceil(len(ranked) * tail))))
            valid = selected.loc[selected["policy_path_valid"].fillna(False).astype(bool)].copy()
            net = pd.to_numeric(valid["policy_net_bps"], errors="coerce")
            net = net[np.isfinite(net)]
            rows.append({
                "arm": arm, "period_kind": kind, "period": period, "tail": tail,
                "admitted_score_rows": int(len(part)), "selected_score_rows": int(len(selected)),
                "valid_outcomes": int(len(net)), "outcome_coverage": float(len(net) / max(len(selected), 1)),
                "net_bps_per_trade": float(net.mean()) if len(net) else np.nan,
                "positive_rate": float(net.gt(0.0).mean()) if len(net) else np.nan,
            })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--admission-predictions", type=Path, required=True)
    parser.add_argument("--correction-predictions", type=Path, required=True)
    parser.add_argument("--control-arm", required=True)
    parser.add_argument("--correction-arm", required=True)
    parser.add_argument("--evaluation-start", required=True)
    parser.add_argument("--evaluation-end", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(f"immutable output exists: {args.out_dir}")
    start = pd.Timestamp(args.evaluation_start, tz="UTC")
    end = pd.Timestamp(args.evaluation_end, tz="UTC")
    admission = pd.read_parquet(args.admission_predictions)
    admission["__decision_ts__"] = pd.to_datetime(admission["__decision_ts__"], utc=True, errors="raise")
    control = admission.loc[admission["arm"].eq(args.control_arm)].copy()
    if control.empty:
        raise ValueError("control arm absent from admission predictions")
    if control["candidate_id"].duplicated().any():
        raise AssertionError("control admission predictions contain duplicate identities")
    control = control.loc[
        control["__decision_ts__"].ge(start) & control["__decision_ts__"].lt(end)
    ].copy()
    corrected = pd.read_parquet(args.correction_predictions, columns=["candidate_id", "arm", "corrected_score"])
    corrected = corrected.loc[corrected["arm"].eq(args.correction_arm), ["candidate_id", "corrected_score"]].copy()
    corrected = corrected.rename(columns={"corrected_score": "bayesian_overlay_score"})
    if corrected.empty or corrected["candidate_id"].duplicated().any():
        raise ValueError("correction score identities are absent or not unique")
    joined = control.merge(corrected, on="candidate_id", how="left", validate="one_to_one")
    if joined["bayesian_overlay_score"].isna().any():
        raise ValueError("correction score does not cover every frozen-admission identity")
    eligible = joined.loc[joined["causal_21d_side_admitted_ge_50bps"].fillna(False).astype(bool)].copy()
    eligible["control_score"] = pd.to_numeric(eligible["corrected_score"], errors="raise")
    eligible["bayesian_overlay_score"] = pd.to_numeric(eligible["bayesian_overlay_score"], errors="raise")
    if eligible.empty:
        raise ValueError("no frozen-admitted candidates in evaluation interval")
    rows: list[dict[str, object]] = []
    for label, score_col in (("frozen_admission_control", "control_score"), ("frozen_admission_bayesian_overlay", "bayesian_overlay_score")):
        for kind in ("global", "month", "week"):
            rows.extend(_metrics(eligible, score_col, label, kind))
    args.out_dir.mkdir(parents=True)
    eligible.to_parquet(args.out_dir / "overlay_predictions.parquet", index=False, compression="zstd")
    pd.DataFrame(rows).to_parquet(args.out_dir / "metrics.parquet", index=False)
    (args.out_dir / "run_manifest.json").write_text(json.dumps({
        "schema": "strict_r3_bayesian_post_admission_overlay_v1",
        "admission_predictions": str(args.admission_predictions),
        "correction_predictions": str(args.correction_predictions),
        "control_arm": args.control_arm, "correction_arm": args.correction_arm,
        "evaluation": [str(start), str(end)],
        "contract": "frozen causal EV admission; Bayesian score can only reorder admitted candidates",
        "causality": "admission was computed from prior-resolved labels; outcomes are used only for final evaluation",
    }, indent=2) + "\n")
    print(json.dumps({"event": "complete", "rows": int(len(eligible)), "out_dir": str(args.out_dir)}))


if __name__ == "__main__":
    main()
