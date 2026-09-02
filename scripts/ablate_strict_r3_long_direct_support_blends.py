#!/usr/bin/env python3
"""Small downstream blend test for the two strongest direct supportive heads.

This consumes only already sealed strict-OOF Stage-1 score panels.  It never
refits an outcome model, so every input score retains the original H12
prequential lineage.  The blends operate in expected-policy-bps space and are
therefore a compact integration test, not a new broad relearner.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


TAILS = (0.01, 0.02, 0.05, 0.10)
SOURCE = "strict_r3_long_supportive_label_funnel_stage1_20260823_v6"
ARMS = {
    "S3_direct_efficiency_time_equal": (0.5, 0.5, 0.0),
    "S3_direct_efficiency_75_base_25": (0.75, 0.0, 0.25),
    "S3_direct_time_75_base_25": (0.0, 0.75, 0.25),
    "S3_direct_efficiency_time_base_equal": (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(path.rglob("*.parquet")):
        digest.update(str(item.relative_to(path)).encode())
        with item.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
    return digest.hexdigest()


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    valid = np.isfinite(left) & np.isfinite(right)
    if int(valid.sum()) < 8:
        return float("nan")
    a = pd.Series(left[valid]).rank(method="average").to_numpy(float)
    b = pd.Series(right[valid]).rank(method="average").to_numpy(float)
    if a.std(ddof=0) <= 1e-12 or b.std(ddof=0) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _metrics(*, fold: str, cohort: str, score: np.ndarray, actual: np.ndarray, base: np.ndarray, arm: str) -> list[dict[str, Any]]:
    common = {
        "fold": fold,
        "cohort": cohort,
        "arm": arm,
        "feature_mode": "sealed_direct_oof_bps_blend",
        "held_rows": int(len(actual)),
        "score_policy_spearman": _spearman(score, actual),
        "score_policy_residual_spearman": _spearman(score, actual - base),
    }
    rows = []
    valid = np.isfinite(score) & np.isfinite(actual)
    valid_count = int(valid.sum())
    for tail in TAILS:
        # Outcome availability is deliberately not part of the candidate
        # scorer.  Tail diagnostics must consequently size their tail from
        # the resolved evaluation support, not all scored candidates; doing
        # otherwise silently changes a nominal top-k percentage whenever the
        # label substrate has incomplete rows.
        count = max(1, int(np.ceil(tail * valid_count)))
        order = np.argsort(score[valid], kind="stable")[-count:]
        selected = actual[valid][order]
        rows.append({
            **common,
            "metric": f"top_{tail:.0%}_net_ev_bps",
            "tail": tail,
            "selected_rows": int(len(selected)),
            "value": float(selected.mean()),
            "net_sum_bps": float(selected.sum()),
            "policy_ge50_fraction": float((selected >= 50.0).mean()),
        })
    rows.append({**common, "metric": "global_score_policy_spearman", "tail": np.nan, "selected_rows": int(valid.sum()), "value": common["score_policy_spearman"], "net_sum_bps": np.nan, "policy_ge50_fraction": np.nan})
    rows.append({**common, "metric": "global_score_policy_residual_spearman", "tail": np.nan, "selected_rows": int(valid.sum()), "value": common["score_policy_residual_spearman"], "net_sum_bps": np.nan, "policy_ge50_fraction": np.nan})
    return rows


def _read_arm(paths: list[Path], arm: str) -> pd.DataFrame:
    columns = ["candidate_id", "__decision_ts__", "fold", "cohort", "arm", "feature_mode", "predicted_policy_net_bps", "realised_policy_net_bps"]
    for path in paths:
        probe = pd.read_parquet(path, columns=["arm"])
        if str(probe["arm"].iloc[0]) == arm:
            return pd.read_parquet(path, columns=columns)
    raise FileNotFoundError(f"missing source arm {arm}")


def run(*, source_root: Path, out: Path) -> Path:
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True, exist_ok=False)
    prediction_manifest = json.loads((source_root / "stage1_oof_predictions_manifest.json").read_text())
    parts_root = Path(prediction_manifest["root"])
    folders = sorted({Path(item).parent for item in prediction_manifest["parts"]})
    part_root = out / "oof_prediction_parts"
    part_root.mkdir(parents=True, exist_ok=False)
    metrics: list[dict[str, Any]] = []
    fold_audit: list[dict[str, Any]] = []
    for folder in folders:
        paths = sorted(parts_root.glob(f"{folder.as_posix()}/*.parquet"))
        base = _read_arm(paths, "B0_prequential_upstream").rename(columns={"predicted_policy_net_bps": "base_bps"})
        efficiency = _read_arm(paths, "D_direct_efficiency").query("feature_mode == 'causal120'").rename(columns={"predicted_policy_net_bps": "efficiency_bps"})
        timing = _read_arm(paths, "D_direct_time_to_meaningful").query("feature_mode == 'causal120'").rename(columns={"predicted_policy_net_bps": "timing_bps"})
        keys = ["candidate_id", "__decision_ts__", "fold", "cohort"]
        merged = base[keys + ["base_bps", "realised_policy_net_bps"]].merge(
            efficiency[keys + ["efficiency_bps"]], on=keys, how="inner", validate="one_to_one"
        ).merge(timing[keys + ["timing_bps"]], on=keys, how="inner", validate="one_to_one")
        if len(merged) != len(base):
            raise AssertionError(f"identity loss in {folder}: {len(merged)} != {len(base)}")
        if merged["candidate_id"].duplicated().any():
            raise AssertionError("duplicate candidate IDs after score join")
        target_dir = part_root / folder
        target_dir.mkdir(parents=True, exist_ok=False)
        for part, (arm, weights) in enumerate(ARMS.items()):
            eff_w, time_w, base_w = weights
            score = (
                eff_w * pd.to_numeric(merged["efficiency_bps"], errors="coerce").to_numpy(float)
                + time_w * pd.to_numeric(merged["timing_bps"], errors="coerce").to_numpy(float)
                + base_w * pd.to_numeric(merged["base_bps"], errors="coerce").to_numpy(float)
            )
            actual = pd.to_numeric(merged["realised_policy_net_bps"], errors="coerce").to_numpy(float)
            base_score = pd.to_numeric(merged["base_bps"], errors="coerce").to_numpy(float)
            metrics.extend(_metrics(fold=str(merged["fold"].iloc[0]), cohort=str(merged["cohort"].iloc[0]), score=score, actual=actual, base=base_score, arm=arm))
            frame = merged[keys + ["base_bps", "efficiency_bps", "timing_bps", "realised_policy_net_bps"]].copy()
            frame.insert(4, "arm", arm)
            frame.insert(5, "feature_mode", "sealed_direct_oof_bps_blend")
            frame.insert(6, "predicted_policy_net_bps", score.astype(np.float32))
            frame.to_parquet(target_dir / f"part={part:03d}.parquet", index=False, compression="zstd")
        fold_audit.append({"partition": folder.as_posix(), "fold": str(merged["fold"].iloc[0]), "rows": int(len(merged)), "status": "ok"})
    metric_frame = pd.DataFrame(metrics)
    metric_frame.to_parquet(out / "stage3_metrics.parquet", index=False, compression="zstd")
    summary = metric_frame.loc[metric_frame["metric"].isin(("top_1%_net_ev_bps", "top_5%_net_ev_bps", "global_score_policy_residual_spearman"))].groupby(
        ["arm", "feature_mode", "cohort", "metric"], as_index=False
    ).agg(mean_value=("value", "mean"), median_value=("value", "median"), worst_value=("value", "min"), folds=("fold", "nunique"))
    summary.to_parquet(out / "stage3_summary.parquet", index=False, compression="zstd")
    pd.DataFrame(fold_audit).to_parquet(out / "stage3_fold_audit.parquet", index=False, compression="zstd")
    (out / "run_manifest.json").write_text(json.dumps({
        "schema": "strict_r3_long_direct_support_blends_v1",
        "scope": "offline only; no inference/live mutation",
        "source_stage1": str(source_root.resolve()),
        "source_stage1_parquet_sha256": _sha256(source_root / "stage1_oof_prediction_parts"),
        "inputs": {
            "base": "B0_prequential_upstream/frozen_stack",
            "direct_efficiency": "D_direct_efficiency/causal120",
            "direct_time_to_meaningful": "D_direct_time_to_meaningful/causal120",
        },
        "predeclared_bps_weights": {key: list(value) for key, value in ARMS.items()},
        "input_lineage": "all source scores are sealed Stage-1 strict-OOF predictions; no future labels are read by this runner",
    }, indent=2) + "\n")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = run(source_root=args.source_root.resolve(), out=args.out.resolve())
    print(json.dumps({"status": "ok", "out": str(result)}))


if __name__ == "__main__":
    main()
