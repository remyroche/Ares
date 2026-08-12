#!/usr/bin/env python3
"""No-learning control for timestamp-local strict-R3 score corrections.

It consumes the immutable OOF predictions emitted by
``run_strict_r3_timestamp_top30_reliability_rank_ablation.py`` and derives a
score correction using only the upstream final score's rank among current
timestamp top-30 candidates.  This distinguishes value from the learned
reliability feature contract from value attributable solely to local score
normalisation.  Outcomes are read only after scores are constructed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


TAILS = (0.005, 0.01, 0.02, 0.05, 0.10)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _metrics(frame: pd.DataFrame, arm: str, kind: str) -> pd.DataFrame:
    if kind == "global":
        groups = [("all", frame)]
    elif kind == "month":
        groups = frame.groupby(frame["__decision_ts__"].dt.strftime("%Y-%m"), sort=True)
    elif kind == "week":
        groups = frame.groupby(frame["__decision_ts__"].dt.strftime("%G-W%V"), sort=True)
    else:
        raise ValueError(kind)
    rows: list[dict[str, object]] = []
    for period, block in groups:
        for tail in TAILS:
            selected = block.nlargest(max(1, int(math.ceil(tail * len(block)))), "corrected_score", keep="first")
            valid = selected.loc[
                selected["policy_path_valid"].fillna(False).astype(bool)
                & np.isfinite(pd.to_numeric(selected["policy_net_bps"], errors="coerce"))
            ]
            net = pd.to_numeric(valid["policy_net_bps"], errors="coerce")
            rows.append({
                "arm": arm, "period_kind": kind, "period": str(period), "tail": tail,
                "selected_score_rows": len(selected), "valid_outcomes": len(valid),
                "outcome_coverage": float(len(valid) / max(len(selected), 1)),
                "net_bps_per_trade": float(net.mean()) if len(net) else np.nan,
                "positive_rate": float(net.gt(0.0).mean()) if len(net) else np.nan,
            })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--alphas", default="0.1,0.2")
    parser.add_argument(
        "--learned-predictions", type=Path,
        help="Optional focused-head OOF predictions.  Its rank is tested only as a residual to the timestamp base-rank control.",
    )
    parser.add_argument("--learned-arm", default=None)
    parser.add_argument("--base-alpha", type=float, default=0.20)
    parser.add_argument("--residual-authorities", default="0.025,0.05,0.1")
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(f"immutable output exists: {args.out_dir}")
    alphas = tuple(float(token.strip()) for token in args.alphas.split(",") if token.strip())
    if not alphas or any(not 0.0 < value <= 0.20 for value in alphas):
        raise ValueError("alphas must lie in (0, 0.20]")
    residual_authorities = tuple(float(token.strip()) for token in args.residual_authorities.split(",") if token.strip())
    if any(not 0.0 < value <= 0.20 for value in residual_authorities):
        raise ValueError("residual authorities must lie in (0, 0.20]")
    required = [
        "candidate_id", "__decision_ts__", "final_score", "timestamp_top30",
        "policy_path_valid", "policy_net_bps", "arm",
    ]
    source = pd.read_parquet(args.predictions, columns=required)
    # Every model arm duplicates the same identity/population.  The explicit
    # upstream control row is authoritative and avoids accidental mixing of a
    # learned score into this no-learning control.
    source = source.loc[source["arm"].eq("control_final_score")].copy()
    if source["candidate_id"].duplicated().any():
        raise AssertionError("control predictions must have one row per candidate")
    source["__decision_ts__"] = pd.to_datetime(source["__decision_ts__"], utc=True, errors="raise")
    active = source.loc[source["timestamp_top30"].astype(bool), ["__decision_ts__", "candidate_id", "final_score"]].copy()
    active = active.sort_values(["__decision_ts__", "final_score", "candidate_id"], kind="stable")
    position = active.groupby("__decision_ts__", sort=False).cumcount().to_numpy(float)
    count = active.groupby("__decision_ts__", sort=False)["candidate_id"].transform("size").to_numpy(float)
    active["timestamp_base_rank"] = np.divide(position, count - 1.0, out=np.full(len(active), 0.5), where=count > 1.0)
    source = source.merge(active.loc[:, ["candidate_id", "timestamp_base_rank"]], on="candidate_id", how="left", validate="one_to_one")
    source["timestamp_base_rank"] = source["timestamp_base_rank"].fillna(0.5)
    parts: list[pd.DataFrame] = []
    metric_parts: list[pd.DataFrame] = []
    for alpha in alphas:
        scored = source.copy()
        scored["corrected_score"] = pd.to_numeric(scored["final_score"], errors="coerce")
        mask = scored["timestamp_top30"].astype(bool).to_numpy()
        scored.loc[mask, "corrected_score"] += float(alpha) * (
            scored.loc[mask, "timestamp_base_rank"].to_numpy(float) - 0.5
        )
        arm = f"timestamp_base_rank_scorecorr_a{alpha:g}"
        scored["arm"] = arm
        parts.append(scored)
        metric_parts.extend(_metrics(scored, arm, kind) for kind in ("global", "month", "week"))
    if args.learned_predictions is not None:
        if not args.learned_predictions.exists():
            raise FileNotFoundError(args.learned_predictions)
        learned = pd.read_parquet(args.learned_predictions, columns=["candidate_id", "focused_rank", "arm"])
        if args.learned_arm is not None:
            learned = learned.loc[learned["arm"].eq(args.learned_arm)].copy()
        else:
            learned = learned.loc[~learned["arm"].eq("control_final_score")].copy()
        if learned.empty or learned["candidate_id"].duplicated().any():
            raise ValueError("learned prediction source must select exactly one non-control row per candidate")
        learned["focused_rank"] = pd.to_numeric(learned["focused_rank"], errors="coerce").fillna(0.5)
        joined = source.merge(
            learned.loc[:, ["candidate_id", "focused_rank"]], on="candidate_id", how="inner", validate="one_to_one",
        )
        if len(joined) != len(source):
            raise ValueError("learned prediction identities do not exactly cover the base-rank control")
        for authority in residual_authorities:
            scored = joined.copy()
            base = pd.to_numeric(scored["final_score"], errors="coerce").to_numpy(float)
            local = scored["timestamp_base_rank"].to_numpy(float)
            learned_rank = scored["focused_rank"].to_numpy(float)
            active = scored["timestamp_top30"].astype(bool).to_numpy()
            corrected = base.copy()
            corrected[active] += float(args.base_alpha) * (local[active] - 0.5)
            # The focused model may modify only the part unexplained by the
            # powerful no-learning local-rank correction.  This prevents it
            # from discarding timestamp comparability merely to impose a
            # different query-local ordering.
            corrected[active] += float(authority) * (learned_rank[active] - local[active])
            scored["corrected_score"] = corrected
            arm = f"timestamp_base_rank_a{args.base_alpha:g}_plus_focused_residual_a{authority:g}"
            scored["arm"] = arm
            parts.append(scored)
            metric_parts.extend(_metrics(scored, arm, kind) for kind in ("global", "month", "week"))
    control = source.copy()
    control["corrected_score"] = pd.to_numeric(control["final_score"], errors="coerce")
    control["arm"] = "control_final_score"
    parts.append(control)
    metric_parts.extend(_metrics(control, "control_final_score", kind) for kind in ("global", "month", "week"))
    args.out_dir.mkdir(parents=True)
    pd.concat(parts, ignore_index=True).to_parquet(args.out_dir / "oof_predictions.parquet", index=False, compression="zstd")
    pd.concat(metric_parts, ignore_index=True).to_parquet(args.out_dir / "metrics.parquet", index=False)
    (args.out_dir / "run_manifest.json").write_text(json.dumps({
        "schema": "strict_r3_timestamp_rank_no_learning_control_v1",
        "predictions": str(args.predictions), "predictions_sha256": _sha(args.predictions),
        "score": "final_score + alpha * (within-current-timestamp final-score rank among timestamp top-30 - 0.5)",
        "causality": "the control score uses only contemporaneous final scores and candidate identities; outcomes are evaluation-only",
        "admission": "not replayed: this is a global-tail ranking control.  An executable arm must rebuild the causal 21-day admission map prequentially on corrected_score.",
        "sizing": "unchanged",
        "alphas": list(alphas),
        "learned_predictions": None if args.learned_predictions is None else str(args.learned_predictions),
        "learned_predictions_sha256": None if args.learned_predictions is None else _sha(args.learned_predictions),
        "learned_arm": args.learned_arm,
        "residual_score_formula": None if args.learned_predictions is None else "final_score + base_alpha*(timestamp_base_rank - .5) + residual_authority*(focused_rank - timestamp_base_rank), timestamp-top30 only",
        "base_alpha": float(args.base_alpha), "residual_authorities": list(residual_authorities),
    }, indent=2) + "\n")
    print(json.dumps({"event": "complete", "rows": len(source), "out_dir": str(args.out_dir)}))


if __name__ == "__main__":
    main()
