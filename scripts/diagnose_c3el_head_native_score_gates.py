#!/usr/bin/env python3
"""Diagnose head-native C3el score gates without labels or replay.

This reads ``head_native_group_scores.csv`` and ``head_native_folds.csv`` from a
completed head-native run and reports how many groups survive each observable
gate: threshold, predicted-delta, optional feature guard, and final keep.
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


def _num(frame: pd.DataFrame, col: str, *, default: float = 0.0) -> pd.Series:
    if col not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(default)


def _bool(frame: pd.DataFrame, col: str, *, default: bool = False) -> pd.Series:
    if col not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=bool)
    values = frame[col]
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(default).astype(bool)
    return values.astype(str).str.lower().isin({"true", "1", "yes"})


def _normalise_week(frame: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_datetime(frame[col], utc=True, errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S%z")


def _diagnosis(row: pd.Series) -> str:
    rows = int(row.get("rows", 0) or 0)
    score_eligible = int(row.get("score_eligible_groups", 0) or 0)
    gate_kept = int(row.get("gate_kept_groups", 0) or 0)
    guard_min = int(row.get("guard_action_feature_min_groups", 0) or 0)
    max_keep = int(row.get("max_eval_keep", 0) or 0)
    if rows <= 0:
        return "no_eval_groups"
    if score_eligible <= 0:
        return "score_gate_empty"
    if guard_min >= score_eligible and guard_min > 0:
        return "feature_guard_blocks_score_candidates"
    if max_keep > 0 and score_eligible > max_keep and gate_kept <= max_keep:
        return "cap_limited"
    if gate_kept <= 0:
        return "post_score_gate_empty"
    return "gate_passes_some_groups"


def build_report(run_dir: Path) -> pd.DataFrame:
    scores_path = run_dir / "head_native_group_scores.csv"
    folds_path = run_dir / "head_native_folds.csv"
    if not scores_path.exists():
        raise FileNotFoundError(scores_path)
    if not folds_path.exists():
        raise FileNotFoundError(folds_path)
    scores = _read_frame(scores_path)
    folds = _read_frame(folds_path)
    required = {"head", "week_start", "p_intervene", "pred_action_delta_J", "gate_keep"}
    missing = sorted(required.difference(scores.columns))
    if missing:
        raise ValueError(f"{scores_path} missing required columns: {missing}")
    if not {"head", "week_start"}.issubset(folds.columns):
        raise ValueError(f"{folds_path} missing head/week_start")

    scores = scores.copy()
    scores["head"] = scores["head"].astype(str)
    scores["week_key"] = _normalise_week(scores, "week_start")
    scores["p_intervene"] = _num(scores, "p_intervene")
    scores["pred_action_delta_J"] = _num(scores, "pred_action_delta_J")
    scores["gate_keep"] = _bool(scores, "gate_keep")
    scores["guard_action_feature_min"] = _bool(scores, "guard_action_feature_min")

    folds = folds.copy()
    folds["head"] = folds["head"].astype(str)
    folds["week_key"] = _normalise_week(folds, "week_start")
    folds = folds.drop_duplicates(["head", "week_key"], keep="last")
    fold_cols = [
        "head",
        "week_key",
        "threshold",
        "effective_min_pred_delta",
        "max_eval_keep",
        "fallback_used",
        "action_feature_min_guarded_eval_groups",
        "kept_eval_groups",
        "eval_groups",
    ]
    fold_cols = [col for col in fold_cols if col in folds.columns]
    merged = scores.merge(folds[fold_cols], on=["head", "week_key"], how="left")
    threshold = _num(merged, "threshold")
    min_delta = _num(merged, "effective_min_pred_delta")
    merged["score_eligible"] = merged["p_intervene"].ge(threshold) & merged["pred_action_delta_J"].gt(min_delta)
    merged["delta_positive"] = merged["pred_action_delta_J"].gt(0.0)
    merged["delta_min_eligible"] = merged["pred_action_delta_J"].gt(min_delta)

    rows: list[dict[str, Any]] = []
    for (head, week_key), group in merged.groupby(["head", "week_key"], dropna=False):
        first = group.iloc[0]
        row = {
            "head": str(head),
            "week_start": str(week_key),
            "rows": int(len(group)),
            "threshold": float(first.get("threshold", np.nan)),
            "effective_min_pred_delta": float(first.get("effective_min_pred_delta", np.nan)),
            "max_eval_keep": int(first.get("max_eval_keep", 0) or 0),
            "fallback_used": bool(str(first.get("fallback_used", "False")).lower() in {"true", "1", "yes"}),
            "delta_positive_groups": int(group["delta_positive"].sum()),
            "delta_min_eligible_groups": int(group["delta_min_eligible"].sum()),
            "score_eligible_groups": int(group["score_eligible"].sum()),
            "guard_action_feature_min_groups": int(group["guard_action_feature_min"].sum()),
            "gate_kept_groups": int(group["gate_keep"].sum()),
            "p_q90": float(group["p_intervene"].quantile(0.90)),
            "p_max": float(group["p_intervene"].max()),
            "pred_delta_q90": float(group["pred_action_delta_J"].quantile(0.90)),
            "pred_delta_max": float(group["pred_action_delta_J"].max()),
        }
        row["score_eligible_share"] = float(row["score_eligible_groups"] / max(row["rows"], 1))
        row["gate_keep_share"] = float(row["gate_kept_groups"] / max(row["rows"], 1))
        row["diagnosis"] = _diagnosis(pd.Series(row))
        rows.append(row)
    out = pd.DataFrame(rows).sort_values(["head", "week_start"]).reset_index(drop=True)

    head_rows = []
    for head, group in out.groupby("head", dropna=False):
        head_rows.append(
            {
                "head": str(head),
                "week_start": "ALL",
                "rows": int(group["rows"].sum()),
                "threshold": np.nan,
                "effective_min_pred_delta": np.nan,
                "max_eval_keep": int(group["max_eval_keep"].sum()),
                "fallback_used": bool(group["fallback_used"].any()),
                "delta_positive_groups": int(group["delta_positive_groups"].sum()),
                "delta_min_eligible_groups": int(group["delta_min_eligible_groups"].sum()),
                "score_eligible_groups": int(group["score_eligible_groups"].sum()),
                "guard_action_feature_min_groups": int(group["guard_action_feature_min_groups"].sum()),
                "gate_kept_groups": int(group["gate_kept_groups"].sum()),
                "p_q90": np.nan,
                "p_max": float(scores.loc[scores["head"].eq(str(head)), "p_intervene"].max()),
                "pred_delta_q90": np.nan,
                "pred_delta_max": float(scores.loc[scores["head"].eq(str(head)), "pred_action_delta_J"].max()),
                "score_eligible_share": float(group["score_eligible_groups"].sum() / max(group["rows"].sum(), 1)),
                "gate_keep_share": float(group["gate_kept_groups"].sum() / max(group["rows"].sum(), 1)),
                "diagnosis": _diagnosis(
                    pd.Series(
                        {
                            "rows": int(group["rows"].sum()),
                            "score_eligible_groups": int(group["score_eligible_groups"].sum()),
                            "gate_kept_groups": int(group["gate_kept_groups"].sum()),
                            "guard_action_feature_min_groups": int(group["guard_action_feature_min_groups"].sum()),
                            "max_eval_keep": int(group["max_eval_keep"].sum()),
                        }
                    )
                ),
            }
        )
    return pd.concat([pd.DataFrame(head_rows), out], ignore_index=True, sort=False)


def _write_markdown(path: Path, report: pd.DataFrame) -> None:
    lines = [
        "# C3el head-native score-gate diagnostics",
        "",
        "This report checks whether C3el score/cap/guard logic leaves enough candidate groups for each head.",
        "",
    ]
    display = [
        "head",
        "week_start",
        "diagnosis",
        "rows",
        "score_eligible_groups",
        "guard_action_feature_min_groups",
        "gate_kept_groups",
        "score_eligible_share",
        "gate_keep_share",
        "threshold",
        "effective_min_pred_delta",
        "max_eval_keep",
        "p_max",
        "pred_delta_max",
    ]
    lines.append(report[display].to_markdown(index=False, floatfmt=".4f"))
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `score_gate_empty`: p/delta thresholds leave no candidates.",
            "- `feature_guard_blocks_score_candidates`: a required feature guard removes the score-eligible set.",
            "- `cap_limited`: score candidates exist, but the evaluation cap dominates the number retained.",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    report = build_report(args.run_dir)
    report.to_csv(args.out_dir / "head_native_score_gate_diagnostics.csv", index=False)
    _write_markdown(args.out_dir / "summary.md", report)
    (args.out_dir / "manifest.json").write_text(
        json.dumps(
            {
                "generated_by": "diagnose_c3el_head_native_score_gates",
                "run_dir": str(args.run_dir),
            },
            indent=2,
            sort_keys=True,
        )
    )
    print(report.to_string(index=False))


if __name__ == "__main__":
    main()
