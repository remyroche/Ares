#!/usr/bin/env python3
"""Diagnose why head-native C3el threshold selection falls back.

This script is intentionally artifact-only: it reads ``head_native_folds.csv``
and, when present, ``head_native_threshold_trials.csv`` from completed
head-native C3el runs.  It does not replay trades or refit models.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _read_frame(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _numeric(frame: pd.DataFrame, col: str, *, default: float = 0.0) -> pd.Series:
    if col not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[col], errors="coerce").fillna(default)


def _bool_series(frame: pd.DataFrame, col: str, *, default: bool = False) -> pd.Series:
    if col not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=bool)
    values = frame[col]
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(default).astype(bool)
    return values.astype(str).str.lower().isin({"true", "1", "yes"})


def _parse_candidate(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        path = Path(raw)
        return path.name, path
    name, value = raw.split("=", 1)
    return name.strip(), Path(value.strip())


def _status(row: dict[str, Any]) -> str:
    used = int(row.get("used_model_week_count", 0) or 0)
    if used <= 0:
        return "no_used_model_weeks"
    if not bool(row.get("threshold_trial_file_present", False)):
        return "missing_threshold_trial_artifact"
    eligible = int(row.get("threshold_trial_eligible_count", 0) or 0)
    positive = int(row.get("threshold_trial_positive_count", 0) or 0)
    best = float(row.get("threshold_trial_best_value", np.nan))
    if eligible <= 0:
        return "no_eligible_threshold_trials"
    if positive > 0:
        return "holdout_positive"
    if np.isfinite(best) and best < 0.0:
        return "holdout_selection_negative"
    return "no_positive_holdout_threshold_trial"


def _recommendation(status: str) -> str:
    if status == "holdout_positive":
        return "candidate_has_non_fallback_threshold_evidence"
    if status == "holdout_selection_negative":
        return "do_not_promote_fallback; revise target/action label or require regime-conditioned threshold evidence"
    if status == "missing_threshold_trial_artifact":
        return "rerun_or_materialize_threshold_trials_before_comparing_candidate"
    if status == "no_eligible_threshold_trials":
        return "relax min_keep/grid only as diagnostic; current grid cannot produce evaluable holdout actions"
    if status == "no_used_model_weeks":
        return "increase label support before training C3el action learner"
    return "keep_shadow_only_until_positive_holdout_threshold_evidence"


def summarise_candidate(name: str, run_dir: Path) -> pd.DataFrame:
    folds_path = run_dir / "head_native_folds.csv"
    if not folds_path.exists():
        raise FileNotFoundError(f"{run_dir} missing head_native_folds.csv")
    folds = _read_frame(folds_path)
    if "head" not in folds.columns:
        raise ValueError(f"{folds_path} is missing head")
    folds["head"] = folds["head"].astype(str)
    folds["_used_model"] = _bool_series(folds, "used_model")
    folds["_fallback_used"] = _bool_series(folds, "fallback_used")
    fold_summary = (
        folds.groupby("head", dropna=False)
        .agg(
            used_model_week_count=("_used_model", "sum"),
            total_fold_rows=("_used_model", "size"),
            fallback_used_week_count=("_fallback_used", "sum"),
        )
        .reset_index()
    )
    for col in (
        "eval_groups",
        "kept_eval_groups",
        "threshold_keep",
        "threshold_value",
        "train_groups",
        "train_positive_groups",
        "train_positive_group_rate",
    ):
        vals = folds.assign(_value=_numeric(folds, col))
        agg = vals.groupby("head", dropna=False)["_value"]
        if col == "train_positive_group_rate":
            part = agg.mean().rename(f"mean_{col}").reset_index()
        else:
            part = agg.sum().rename(f"sum_{col}").reset_index()
        fold_summary = fold_summary.merge(part, on="head", how="left")

    trials_path = run_dir / "head_native_threshold_trials.csv"
    trial_rows: list[dict[str, Any]] = []
    if trials_path.exists():
        trials = _read_frame(trials_path)
        if "head" not in trials.columns:
            raise ValueError(f"{trials_path} is missing head")
        trials["head"] = trials["head"].astype(str)
        trials["_eligible"] = _bool_series(trials, "eligible")
        trials["_value"] = _numeric(trials, "value")
        for head, group in trials.groupby("head", dropna=False):
            eligible = group.loc[group["_eligible"]].copy()
            row: dict[str, Any] = {
                "head": str(head),
                "threshold_trial_file_present": True,
                "threshold_trial_count": int(len(group)),
                "threshold_trial_eligible_count": int(len(eligible)),
                "threshold_trial_positive_count": int(eligible["_value"].gt(0.0).sum()) if not eligible.empty else 0,
                "threshold_trial_best_value": np.nan,
                "threshold_trial_best_week": "",
                "threshold_trial_best_threshold": np.nan,
                "threshold_trial_best_min_pred_delta": np.nan,
                "threshold_trial_best_keep": np.nan,
            }
            if not eligible.empty:
                best = eligible.loc[eligible["_value"].idxmax()]
                row.update(
                    {
                        "threshold_trial_best_value": float(best["_value"]),
                        "threshold_trial_best_week": str(best.get("week_start", "")),
                        "threshold_trial_best_threshold": float(best.get("threshold", np.nan)),
                        "threshold_trial_best_min_pred_delta": float(best.get("min_pred_delta", np.nan)),
                        "threshold_trial_best_keep": float(best.get("keep", np.nan)),
                    }
                )
            trial_rows.append(row)
    trial_summary = pd.DataFrame(trial_rows)
    if trial_summary.empty:
        trial_summary = pd.DataFrame(
            {
                "head": fold_summary["head"],
                "threshold_trial_file_present": False,
                "threshold_trial_count": 0,
                "threshold_trial_eligible_count": 0,
                "threshold_trial_positive_count": 0,
                "threshold_trial_best_value": np.nan,
                "threshold_trial_best_week": "",
                "threshold_trial_best_threshold": np.nan,
                "threshold_trial_best_min_pred_delta": np.nan,
                "threshold_trial_best_keep": np.nan,
            }
        )
    out = fold_summary.merge(trial_summary, on="head", how="left")
    out["candidate"] = str(name)
    out["run_dir"] = str(run_dir)
    out["fallback_used_week_rate"] = _numeric(out, "fallback_used_week_count") / _numeric(
        out, "used_model_week_count"
    ).replace(0.0, np.nan)
    out["kept_eval_share"] = _numeric(out, "sum_kept_eval_groups") / _numeric(out, "sum_eval_groups").replace(0.0, np.nan)
    out["diagnosis"] = [_status(row._asdict()) for row in out.itertuples(index=False)]
    out["recommendation"] = out["diagnosis"].map(_recommendation)
    cols = ["candidate", "run_dir", "head"] + [c for c in out.columns if c not in {"candidate", "run_dir", "head"}]
    return out[cols]


def _write_markdown(path: Path, report: pd.DataFrame) -> None:
    lines = [
        "# C3el threshold-selection diagnostics",
        "",
        "This report diagnoses whether replay improvements are supported by fold holdout threshold evidence or only by fallback action gates.",
        "",
    ]
    if report.empty:
        lines.append("No rows.")
    else:
        display_cols = [
            "candidate",
            "head",
            "diagnosis",
            "recommendation",
            "used_model_week_count",
            "mean_train_positive_group_rate",
            "fallback_used_week_rate",
            "sum_kept_eval_groups",
            "kept_eval_share",
            "threshold_trial_file_present",
            "threshold_trial_eligible_count",
            "threshold_trial_positive_count",
            "threshold_trial_best_value",
            "threshold_trial_best_week",
            "threshold_trial_best_threshold",
            "threshold_trial_best_min_pred_delta",
            "threshold_trial_best_keep",
        ]
        lines.append(report[display_cols].to_markdown(index=False, floatfmt=".4f"))
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `holdout_selection_negative`: the current threshold objective rejected interventions in fold holdout, so replay gains are fallback-only.",
            "- `missing_threshold_trial_artifact`: the run cannot prove non-fallback threshold evidence; regenerate trials before comparing it to another candidate.",
            "- `no_eligible_threshold_trials`: the grid/min-keep contract produced no evaluable holdout actions.",
            "",
            "## Next ablation hypothesis",
            "",
            "If replay is positive but threshold holdout is negative, the likely issue is action-label/threshold-objective mismatch or regime-conditional action value. Test revised labels or regime-conditioned thresholds before adding model capacity.",
        ]
    )
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", action="append", default=[], help="NAME=run_dir or run_dir")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if not args.candidate:
        raise ValueError("at least one --candidate is required")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    parts = [summarise_candidate(name, path) for name, path in map(_parse_candidate, args.candidate)]
    report = pd.concat(parts, ignore_index=True, sort=False)
    report.to_csv(args.out_dir / "threshold_selection_diagnostics.csv", index=False)
    _write_markdown(args.out_dir / "summary.md", report)
    (args.out_dir / "manifest.json").write_text(
        json.dumps(
            {
                "generated_by": "diagnose_c3el_threshold_selection",
                "candidates": [{"name": name, "run_dir": str(path)} for name, path in map(_parse_candidate, args.candidate)],
            },
            indent=2,
            sort_keys=True,
        )
    )
    print(report.to_string(index=False))


if __name__ == "__main__":
    main()
