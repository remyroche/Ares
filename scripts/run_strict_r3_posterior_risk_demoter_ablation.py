#!/usr/bin/env python3
"""No-refit causal posterior risk-demoter diagnostic for strict-R3 trust arms."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


TAILS = (0.005, 0.01, 0.02, 0.05, 0.10)
ALPHAS = (0.025, 0.05, 0.10)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _metrics(frame: pd.DataFrame, score: str, arm: str, kind: str) -> pd.DataFrame:
    groups = [("all", frame)] if kind == "global" else frame.groupby(
        frame["__decision_ts__"].dt.to_period("M").astype(str), sort=True,
    )
    rows: list[dict[str, object]] = []
    for period, block in groups:
        population = block.loc[np.isfinite(pd.to_numeric(block[score], errors="coerce"))]
        for tail in TAILS:
            count = max(1, int(math.ceil(tail * len(population))))
            selected = population.nlargest(count, score, keep="first")
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
                "positive_rate": float((net > 0.0).mean()) if len(net) else np.nan,
            })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(f"immutable output exists: {args.out_dir}")
    needed = [
        "arm", "__decision_ts__", "final_score", "posterior_expected_rank_train",
        "posterior_adverse_rank_train", "policy_path_valid", "policy_net_bps",
    ]
    frame = pd.read_parquet(args.predictions, columns=needed)
    frame["__decision_ts__"] = pd.to_datetime(frame["__decision_ts__"], utc=True, errors="raise")
    frame["posterior_expected_rank_train"] = pd.to_numeric(
        frame["posterior_expected_rank_train"], errors="coerce").fillna(0.5)
    frame["posterior_adverse_rank_train"] = pd.to_numeric(
        frame["posterior_adverse_rank_train"], errors="coerce").fillna(0.5)
    frame["final_score"] = pd.to_numeric(frame["final_score"], errors="coerce")
    grids: list[tuple[str, np.ndarray]] = [("base_score", frame["final_score"].to_numpy(float))]
    for alpha in ALPHAS:
        grids.append((
            f"risk_only_a{alpha:g}",
            frame["final_score"].to_numpy(float) - alpha * frame["posterior_adverse_rank_train"].to_numpy(float),
        ))
        grids.append((
            f"mean_minus_risk_a{alpha:g}",
            frame["final_score"].to_numpy(float) + alpha * (
                frame["posterior_expected_rank_train"].to_numpy(float)
                - frame["posterior_adverse_rank_train"].to_numpy(float)
            ),
        ))
    metric_parts: list[pd.DataFrame] = []
    for source_arm, block in frame.groupby("arm", sort=True):
        for name, values in grids:
            work = block.copy()
            work["demoted_score"] = values[block.index.to_numpy()]
            arm = f"{source_arm}__{name}"
            metric_parts.extend([
                _metrics(work, "demoted_score", arm, "global"),
                _metrics(work, "demoted_score", arm, "month"),
            ])
    metrics = pd.concat(metric_parts, ignore_index=True)
    args.out_dir.mkdir(parents=True)
    metrics.to_parquet(args.out_dir / "metrics.parquet", index=False)
    (args.out_dir / "run_manifest.json").write_text(json.dumps({
        "schema": "strict_r3_traincdf_posterior_risk_demoter_v1",
        "predictions": str(args.predictions), "predictions_sha256": _sha(args.predictions),
        "formulas": {
            "risk_only": "final_score - alpha * posterior_adverse_rank_train",
            "mean_minus_risk": "final_score + alpha * (posterior_expected_rank_train - posterior_adverse_rank_train)",
        },
        "alphas": list(ALPHAS),
        "causality": "posterior ranks are empirical CDFs against only the fitting-fold train prediction distribution; no held-window rank, outcome, or selection is used in a score",
        "integration": "diagnostic re-ranking after the frozen upstream score; causal admission remains a separate production gate",
        "selection": "all predeclared arms reported; no 2026 winner is promoted",
    }, indent=2) + "\n")
    print(json.dumps({"event": "complete", "rows": len(metrics), "out": str(args.out_dir)}))


if __name__ == "__main__":
    main()
