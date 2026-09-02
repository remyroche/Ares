#!/usr/bin/env python3
"""Test whether the best causal×path J2 score complements the direct blend.

This is deliberately an algebraic, sealed-OOF comparison.  It retrains no
model and consumes only scores already produced by strict chronological folds:
the selected direct bps blend and C1 Ward K4 soft causal×path expected EV.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


TAILS = (0.01, 0.02, 0.05, 0.10)
DIRECT_ARM = "S3_direct_efficiency_time_base_equal"
JOINT_COLUMN = "C1_ward_k4_J2_soft_base_equal_causal120"
WEIGHTS = {
    "M0_direct_control": (1.0, 0.0),
    "M1_direct75_joint25": (0.75, 0.25),
    "M2_direct50_joint50": (0.50, 0.50),
    "M3_direct25_joint75": (0.25, 0.75),
}


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    usable = np.isfinite(x) & np.isfinite(y)
    if usable.sum() < 8:
        return float("nan")
    left = pd.Series(x[usable]).rank(method="average").to_numpy(float)
    right = pd.Series(y[usable]).rank(method="average").to_numpy(float)
    if left.std(ddof=0) < 1e-12 or right.std(ddof=0) < 1e-12:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def _read_direct(root: Path, suffix: str) -> pd.DataFrame:
    path = root / "oof_prediction_parts" / f"fold={suffix}"
    for part in sorted(path.glob("*.parquet")):
        probe = pd.read_parquet(part, columns=["arm"])
        if str(probe["arm"].iloc[0]) == DIRECT_ARM:
            return pd.read_parquet(part, columns=["candidate_id", "__decision_ts__", "fold", "cohort", "arm", "predicted_policy_net_bps", "realised_policy_net_bps"])
    raise FileNotFoundError(f"{DIRECT_ARM} missing from {path}")


def _metrics(frame: pd.DataFrame, arm: str, score: np.ndarray) -> list[dict[str, Any]]:
    actual = frame["realised_policy_net_bps"].to_numpy(float)
    base = frame["direct_bps"].to_numpy(float)
    rows: list[dict[str, Any]] = []
    common = {
        "fold": str(frame["fold"].iloc[0]), "cohort": str(frame["cohort"].iloc[0]), "arm": arm,
        "held_rows": int(len(frame)), "score_policy_spearman": _spearman(score, actual),
        "score_policy_residual_spearman": _spearman(score, actual - base),
    }
    usable = np.isfinite(score) & np.isfinite(actual)
    for tail in TAILS:
        count = max(1, int(np.ceil(tail * len(frame))))
        pick = np.argsort(score[usable], kind="stable")[-count:]
        selected = actual[usable][pick]
        rows.append({**common, "metric": f"top_{tail:.0%}_net_ev_bps", "tail": tail, "selected_rows": int(len(selected)), "value": float(selected.mean()), "net_sum_bps": float(selected.sum())})
    rows.append({**common, "metric": "global_score_policy_residual_spearman", "tail": np.nan, "selected_rows": int(usable.sum()), "value": common["score_policy_residual_spearman"], "net_sum_bps": np.nan})
    return rows


def run(*, direct_root: Path, joint_root: Path, out: Path) -> Path:
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((direct_root / "run_manifest.json").read_text())
    direct_parts_root = direct_root / "oof_prediction_parts"
    joint_parts = sorted((joint_root / "causal_joint_oof_predictions").glob("fold=*.parquet"))
    prediction_root = out / "oof_prediction_parts"; prediction_root.mkdir()
    metrics: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    for joint_path in joint_parts:
        suffix = joint_path.stem.split("fold=", 1)[1]
        direct = _read_direct(direct_root, suffix).rename(columns={"predicted_policy_net_bps": "direct_bps"})
        joint = pd.read_parquet(joint_path, columns=["candidate_id", "__decision_ts__", JOINT_COLUMN, "policy_net_bps"])
        merged = direct.merge(joint, on=["candidate_id", "__decision_ts__"], how="inner", validate="one_to_one")
        if len(merged) != len(direct) or merged["candidate_id"].duplicated().any():
            raise AssertionError(f"identity mismatch in {suffix}")
        # One sealed source persisted this evaluation-only field as float32 and
        # the other as float64.  1e-3 bps is far below both storage precision
        # and any economic threshold; identity remains exact.
        if not np.isclose(merged["realised_policy_net_bps"], merged["policy_net_bps"], atol=1e-3, rtol=0.0).all():
            raise AssertionError(f"policy outcome mismatch in {suffix}")
        target = prediction_root / f"fold={suffix}.parquet"
        output = merged[["candidate_id", "__decision_ts__", "fold", "cohort", "direct_bps", JOINT_COLUMN, "realised_policy_net_bps"]].copy()
        for arm, (direct_weight, joint_weight) in WEIGHTS.items():
            score = direct_weight * merged["direct_bps"].to_numpy(float) + joint_weight * merged[JOINT_COLUMN].to_numpy(float)
            metrics.extend(_metrics(merged, arm, score))
            output[arm] = score.astype(np.float32)
        output.to_parquet(target, index=False, compression="zstd")
        audit.append({"fold": str(merged["fold"].iloc[0]), "rows": int(len(merged)), "identity_exact": True, "policy_exact": True})
    metric_frame = pd.DataFrame(metrics)
    metric_frame.to_parquet(out / "direct_joint_blend_metrics.parquet", index=False, compression="zstd")
    summary = metric_frame.loc[metric_frame["metric"].isin(("top_1%_net_ev_bps", "top_2%_net_ev_bps", "top_5%_net_ev_bps", "global_score_policy_residual_spearman"))].groupby(["arm", "cohort", "metric"], as_index=False).agg(mean_value=("value", "mean"), worst_value=("value", "min"), folds=("fold", "nunique"))
    summary.to_parquet(out / "direct_joint_blend_summary.parquet", index=False, compression="zstd")
    pd.DataFrame(audit).to_parquet(out / "identity_audit.parquet", index=False, compression="zstd")
    (out / "run_manifest.json").write_text(json.dumps({
        "schema": "strict_r3_long_direct_joint_blends_v1", "scope": "offline sealed-OOF score arithmetic only; no live mutation",
        "direct_root": str(direct_root.resolve()), "joint_root": str(joint_root.resolve()),
        "direct_arm": DIRECT_ARM, "joint_column": JOINT_COLUMN, "weights": WEIGHTS,
        "causality": "both source scores are held-fold strict-OOF; this runner reads no path coordinate or raw training outcome as an input",
    }, indent=2) + "\n")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--direct-root", type=Path, required=True)
    parser.add_argument("--joint-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps({"out": str(run(direct_root=args.direct_root.resolve(), joint_root=args.joint_root.resolve(), out=args.out.resolve()))}))


if __name__ == "__main__":
    main()
