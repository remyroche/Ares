#!/usr/bin/env python3
"""Freeze diagnostic geometry for the three S3 direct-label component scores.

This is analysis-only: it reads sealed strict-OOF component values and cannot
change S3 selection, calibration, admission, or portfolio construction.  The
realised policy net field is persisted only for retrospective diagnostics.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


COMPONENTS = ("base_bps", "efficiency_bps", "timing_bps")
KEEP_ARM = "S3_direct_efficiency_time_base_equal"
IDENTITY = ("candidate_id", "__decision_ts__", "fold", "cohort")


def _read_parts(root: Path) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for fold_root in sorted((root / "oof_prediction_parts").glob("fold=*")):
        fold_parts = [pd.read_parquet(path) for path in sorted(fold_root.glob("*.parquet"))]
        if not fold_parts:
            raise AssertionError(f"no panels in {fold_root}")
        frame = pd.concat(fold_parts, ignore_index=True)
        selected = frame.loc[frame["arm"].eq(KEEP_ARM)].copy()
        if selected.empty:
            raise AssertionError(f"{KEEP_ARM} missing from {fold_root}")
        if selected["candidate_id"].duplicated().any():
            raise AssertionError(f"duplicate S3 identity in {fold_root}")
        parts.append(selected.loc[:, [*IDENTITY, *COMPONENTS, "realised_policy_net_bps"]])
    if not parts:
        raise FileNotFoundError(root / "oof_prediction_parts")
    result = pd.concat(parts, ignore_index=True)
    if result["candidate_id"].duplicated().any():
        raise AssertionError("candidate identity appears in more than one held fold")
    return result


def _rank_by_timestamp(frame: pd.DataFrame, column: str) -> pd.Series:
    return frame.groupby("__decision_ts__", sort=False)[column].rank(pct=True, method="average")


def run(*, source_root: Path, out: Path) -> Path:
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True, exist_ok=False)
    frame = _read_parts(source_root)
    frame = frame.rename(columns={"base_bps": "B_bps", "efficiency_bps": "E_bps", "timing_bps": "T_bps"})
    for component in ("E_bps", "T_bps", "B_bps"):
        frame[f"{component[0]}_timestamp_rank"] = _rank_by_timestamp(frame, component).astype(np.float32)
    frame["E_minus_T_bps"] = frame["E_bps"] - frame["T_bps"]
    frame["E_minus_B_bps"] = frame["E_bps"] - frame["B_bps"]
    frame["T_minus_B_bps"] = frame["T_bps"] - frame["B_bps"]
    triples = frame.loc[:, ["E_bps", "T_bps", "B_bps"]].to_numpy(float)
    frame["ETB_dispersion_bps"] = np.std(triples, axis=1, ddof=0)
    frame["ETB_min_bps"] = np.min(triples, axis=1)
    frame["ETB_max_bps"] = np.max(triples, axis=1)
    frame["ETB_range_bps"] = frame["ETB_max_bps"] - frame["ETB_min_bps"]

    b, e, t = frame["B_timestamp_rank"], frame["E_timestamp_rank"], frame["T_timestamp_rank"]
    frame["diagnostic_pattern"] = np.select(
        [
            (b >= 0.95) & (e <= 0.50) & (t <= 0.50),
            b.between(0.40, 0.60) & (e >= 0.90) & (t >= 0.90),
            (e >= 0.90) & (t >= 0.90) & (b <= 0.50),
            (b >= 0.90) & (e <= 0.25) & (t <= 0.25),
        ],
        ["B_extreme_E_T_low", "B_mid_E_T_high", "B_low_E_T_high", "B_high_E_T_very_low"],
        default="other",
    )
    frame["dispersion_decile"] = pd.qcut(frame["ETB_dispersion_bps"], q=10, duplicates="drop", labels=False).astype("Int8")
    frame.to_parquet(out / "s3_score_disagreement_oof.parquet", index=False, compression="zstd")

    metric_rows: list[dict[str, object]] = []
    for group_name, group in frame.groupby(["fold", "cohort", "diagnostic_pattern"], observed=True):
        metric_rows.append({
            "fold": group_name[0], "cohort": group_name[1], "slice": group_name[2],
            "rows": int(len(group)), "policy_net_mean_bps": float(group["realised_policy_net_bps"].mean()),
            "policy_net_median_bps": float(group["realised_policy_net_bps"].median()),
            "policy_net_p05_bps": float(group["realised_policy_net_bps"].quantile(0.05)),
            "mean_dispersion_bps": float(group["ETB_dispersion_bps"].mean()),
        })
    metrics = pd.DataFrame(metric_rows)
    metrics.to_parquet(out / "diagnostic_pattern_outcomes.parquet", index=False, compression="zstd")
    by_dispersion = frame.groupby(["fold", "cohort", "dispersion_decile"], observed=True).agg(
        rows=("candidate_id", "size"), policy_net_mean_bps=("realised_policy_net_bps", "mean"),
        policy_net_median_bps=("realised_policy_net_bps", "median"),
        range_mean_bps=("ETB_range_bps", "mean"),
    ).reset_index()
    by_dispersion.to_parquet(out / "dispersion_outcomes.parquet", index=False, compression="zstd")
    correlation = frame.loc[:, ["E_bps", "T_bps", "B_bps", "realised_policy_net_bps"]].corr(method="spearman")
    correlation.to_parquet(out / "score_outcome_spearman.parquet", compression="zstd")
    (out / "run_manifest.json").write_text(json.dumps({
        "schema": "strict_r3_long_s3_score_disagreement_v1",
        "scope": "sealed strict-OOF diagnostic only; prohibited from S3 selection, admission, portfolio, or live inference",
        "source_root": str(source_root.resolve()), "source_arm": KEEP_ARM,
        "components": {"E": "efficiency_bps", "T": "timing_bps", "B": "base_bps"},
        "outputs": ["E", "T", "B", "E-T", "E-B", "T-B", "std(E,T,B)", "min(E,T,B)", "max(E,T,B)"],
        "outcome_usage": "retrospective diagnostics only",
    }, indent=2) + "\n")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(run(source_root=args.source_root.resolve(), out=args.out.resolve()))


if __name__ == "__main__":
    main()
