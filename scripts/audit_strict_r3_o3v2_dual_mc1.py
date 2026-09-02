#!/usr/bin/env python3
"""Audit the separate current and BCF MC1 expected-EV mapper heads.

The report is research-only.  It checks identity and label-time invariants,
then quantifies calibration, agreement, and the population selected by the
dual admission intersection.  It never fits, scores, admits, or trades.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED = (
    "candidate_id", "__decision_ts__", "final_score", "mc1_expected_bps",
    "static_expected_bps", "recent_shift_bps", "policy_path_valid",
    "policy_net_bps", "policy_label_available_ts",
)


def _load(path: Path, family: str) -> pd.DataFrame:
    frame = pd.read_parquet(path, columns=list(REQUIRED))
    missing = sorted(set(REQUIRED) - set(frame.columns))
    if missing:
        raise AssertionError(f"{family}: missing required fields {missing}")
    if frame["candidate_id"].duplicated().any():
        raise AssertionError(f"{family}: duplicate candidate IDs")
    frame["__decision_ts__"] = pd.to_datetime(frame["__decision_ts__"], utc=True, errors="raise")
    frame["policy_label_available_ts"] = pd.to_datetime(
        frame["policy_label_available_ts"], utc=True, errors="coerce",
    )
    frame = frame.rename(columns={
        "final_score": f"{family}_final_score",
        "mc1_expected_bps": f"{family}_expected_bps",
        "static_expected_bps": f"{family}_static_bps",
        "recent_shift_bps": f"{family}_shift_bps",
    })
    return frame


def _valid(frame: pd.DataFrame) -> pd.Series:
    return (
        frame["policy_path_valid"].fillna(False).astype(bool)
        & np.isfinite(pd.to_numeric(frame["policy_net_bps"], errors="coerce"))
    )


def _head_metrics(frame: pd.DataFrame, family: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    valid = frame.loc[_valid(frame)].copy()
    valid["month"] = valid["__decision_ts__"].dt.strftime("%Y-%m")
    for month, part in valid.groupby("month", sort=True):
        score = pd.to_numeric(part[f"{family}_expected_bps"], errors="coerce")
        target = pd.to_numeric(part["policy_net_bps"], errors="coerce")
        rows.append({
            "family": family,
            "month": month,
            "rows": int(len(part)),
            "expected_mean_bps": float(score.mean()),
            "realised_mean_bps": float(target.mean()),
            "calibration_gap_bps": float((score - target).mean()),
            "rank_ic_spearman": float(score.corr(target, method="spearman")),
            "static_mean_bps": float(pd.to_numeric(part[f"{family}_static_bps"], errors="coerce").mean()),
            "shift_mean_bps": float(pd.to_numeric(part[f"{family}_shift_bps"], errors="coerce").mean()),
        })
    return rows


def _deciles(frame: pd.DataFrame, family: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    valid = frame.loc[_valid(frame)].copy()
    valid["month"] = valid["__decision_ts__"].dt.strftime("%Y-%m")
    for month, part in valid.groupby("month", sort=True):
        score = pd.to_numeric(part[f"{family}_expected_bps"], errors="coerce")
        rank = score.rank(method="first", pct=True)
        bucket = np.minimum(10, np.maximum(1, np.ceil(10.0 * rank).astype(int)))
        part = part.assign(decile=bucket)
        for decile, cell in part.groupby("decile", sort=True):
            rows.append({
                "family": family,
                "month": month,
                "decile": int(decile),
                "rows": int(len(cell)),
                "expected_mean_bps": float(pd.to_numeric(cell[f"{family}_expected_bps"], errors="coerce").mean()),
                "realised_mean_bps": float(pd.to_numeric(cell["policy_net_bps"], errors="coerce").mean()),
                "calibration_gap_bps": float((pd.to_numeric(cell[f"{family}_expected_bps"], errors="coerce") - pd.to_numeric(cell["policy_net_bps"], errors="coerce")).mean()),
            })
    return rows


def _cohorts(frame: pd.DataFrame, thresholds: tuple[float, ...]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    valid = frame.loc[_valid(frame)].copy()
    for threshold in thresholds:
        current = pd.to_numeric(valid["current_expected_bps"], errors="coerce").ge(threshold)
        bcf = pd.to_numeric(valid["bcf_expected_bps"], errors="coerce").ge(threshold)
        groups = {
            "both_admit": current & bcf,
            "current_only": current & ~bcf,
            "bcf_only": ~current & bcf,
            "neither": ~current & ~bcf,
        }
        for name, mask in groups.items():
            cell = valid.loc[mask]
            rows.append({
                "threshold_bps": float(threshold),
                "cohort": name,
                "rows": int(len(cell)),
                "share_valid": float(len(cell) / len(valid)) if len(valid) else float("nan"),
                "expected_current_bps": float(pd.to_numeric(cell["current_expected_bps"], errors="coerce").mean()) if len(cell) else float("nan"),
                "expected_bcf_bps": float(pd.to_numeric(cell["bcf_expected_bps"], errors="coerce").mean()) if len(cell) else float("nan"),
                "realised_net_bps": float(pd.to_numeric(cell["policy_net_bps"], errors="coerce").mean()) if len(cell) else float("nan"),
            })
    return rows


def _agreement(frame: pd.DataFrame) -> pd.DataFrame:
    valid = frame.loc[_valid(frame)].copy()
    valid["month"] = valid["__decision_ts__"].dt.strftime("%Y-%m")
    rows = []
    for month, part in valid.groupby("month", sort=True):
        rows.append({
            "month": month,
            "rows": int(len(part)),
            "expected_pearson": float(part["current_expected_bps"].corr(part["bcf_expected_bps"], method="pearson")),
            "expected_spearman": float(part["current_expected_bps"].corr(part["bcf_expected_bps"], method="spearman")),
            "score_spearman": float(part["current_final_score"].corr(part["bcf_final_score"], method="spearman")),
            "absolute_expected_gap_bps": float((part["current_expected_bps"] - part["bcf_expected_bps"]).abs().mean()),
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current", required=True, type=Path)
    parser.add_argument("--bcf", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--thresholds", default="30,50")
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    args.out.mkdir(parents=True)
    thresholds = tuple(float(value) for value in args.thresholds.split(",") if value)
    current = _load(args.current, "current")
    bcf = _load(args.bcf, "bcf")
    shared = current.merge(
        bcf.drop(columns=["policy_path_valid", "policy_net_bps", "policy_label_available_ts"]),
        on=["candidate_id", "__decision_ts__"], how="inner", validate="one_to_one",
    )
    if len(shared) != len(current) or len(shared) != len(bcf):
        raise AssertionError("dual MC1 heads do not have identical candidate identities")
    label_after_decision = (
        shared["policy_label_available_ts"].notna()
        & shared["policy_label_available_ts"].gt(shared["__decision_ts__"])
    )
    if (_valid(shared) & ~label_after_decision).any():
        raise AssertionError("valid MC1 row lacks post-decision label availability")

    metrics = pd.DataFrame(_head_metrics(shared, "current") + _head_metrics(shared, "bcf"))
    deciles = pd.DataFrame(_deciles(shared, "current") + _deciles(shared, "bcf"))
    cohorts = pd.DataFrame(_cohorts(shared, thresholds))
    agreement = _agreement(shared)
    metrics.to_parquet(args.out / "head_month_metrics.parquet", index=False, compression="zstd")
    deciles.to_parquet(args.out / "head_month_deciles.parquet", index=False, compression="zstd")
    cohorts.to_parquet(args.out / "dual_admission_cohorts.parquet", index=False, compression="zstd")
    agreement.to_parquet(args.out / "head_agreement.parquet", index=False, compression="zstd")
    report = {
        "schema": "strict_r3_o3v2_dual_mc1_audit_v1",
        "scope": "offline audit only; no models, admissions, portfolio state, or live process was changed",
        "rows": int(len(shared)),
        "valid_rows": int(_valid(shared).sum()),
        "identical_current_bcf_identities": True,
        "valid_labels_available_after_decision": True,
        "thresholds_bps": list(thresholds),
        "sources": {"current": str(args.current), "bcf": str(args.bcf)},
    }
    (args.out / "correctness_report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
