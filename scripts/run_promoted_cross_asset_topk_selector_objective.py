#!/usr/bin/env python3
"""Leakage-safe top-k EV/path selector objective for promoted meta scores.

This is a post-prediction selector smoke.  It does not retrain train_meta.
For each validation month after the first available month, it:

* scores a small set of fixed and blended meta-score candidates;
* selects the candidate using strictly prior OOF months;
* applies the selected score to the next month;
* compares against fixed baseline/promoted selectors on the same months.

The goal is to test whether the promoted cross-asset representation is being
held back by a fixed score blend, while preserving the train/meta leakage
contract: no labels from the evaluated month are used to choose the selector.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_promoted_cross_asset_cell_effects import (  # noqa: E402
    DEFAULT_BASELINE_SMOKE_DIR,
    DEFAULT_PROMOTED_HANDOFF_DIR,
    DEFAULT_PROMOTED_SMOKE_DIR,
    PREDICTIONS_NAME,
    _best_score_column,
)
from scripts.run_s52_train_meta_regime_handoff_smoke import (  # noqa: E402
    _breakdown_rows,
    _num,
    _selector_metrics,
    _summarize,
)


DEFAULT_OUT_DIR = DEFAULT_PROMOTED_HANDOFF_DIR / "promoted_cross_asset_topk_selector_objective_v1"
OBJECTIVE_KEEP_WEIGHTS = {0.10: 0.50, 0.20: 0.30, 0.30: 0.20}


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    weights: dict[str, float]


FIXED_SCORE_COLUMNS = [
    "score_base",
    "score_meta_clean_exec",
    "score_meta_positive_margin",
    "score_meta_exec_margin",
    "score_meta_clean_minus_risk",
    "score_meta_exec_margin_risk_blend",
    "score_meta_context_hint_blend",
    "score_meta_long_aware_clean_minus_risk",
]

BLEND_CANDIDATES = [
    CandidateSpec(
        "blend_ev_path_balanced",
        {
            "score_meta_exec_margin": 1.00,
            "score_meta_clean_exec": 0.45,
            "score_meta_positive_margin": 0.25,
            "score_meta_bad_path": -0.50,
            "score_meta_timeout": -0.20,
        },
    ),
    CandidateSpec(
        "blend_clean_first",
        {
            "score_meta_clean_exec": 1.00,
            "score_meta_positive_margin": 0.35,
            "score_meta_exec_margin": 0.35,
            "score_meta_bad_path": -0.65,
            "score_meta_timeout": -0.25,
        },
    ),
    CandidateSpec(
        "blend_bad_mae_strict",
        {
            "score_meta_exec_margin": 0.70,
            "score_meta_clean_exec": 0.55,
            "score_meta_bad_path": -1.00,
            "score_meta_timeout": -0.20,
        },
    ),
    CandidateSpec(
        "blend_timeout_strict",
        {
            "score_meta_exec_margin": 0.85,
            "score_meta_clean_exec": 0.40,
            "score_meta_bad_path": -0.40,
            "score_meta_timeout": -0.80,
        },
    ),
    CandidateSpec(
        "blend_long_aware_path",
        {
            "score_meta_exec_margin": 0.85,
            "score_meta_long_aware_clean_minus_risk": 0.55,
            "score_meta_clean_exec": 0.25,
            "score_meta_bad_path": -0.35,
            "score_meta_timeout": -0.20,
        },
    ),
    CandidateSpec(
        "blend_exec_ev_only",
        {
            "score_meta_exec_margin": 1.00,
            "score_meta_positive_margin": 0.20,
            "score_meta_bad_path": -0.20,
        },
    ),
]


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return value


def _read_predictions(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_parquet(path)
    if "month" not in frame.columns:
        raise ValueError(f"Predictions missing month column: {path}")
    return frame.copy()


def _candidate_score_columns(frame: pd.DataFrame) -> list[str]:
    return [col for col in FIXED_SCORE_COLUMNS if col in frame.columns and _num(frame[col]).notna().any()]


def _robust_scaler(history: pd.DataFrame, columns: list[str]) -> dict[str, tuple[float, float]]:
    stats: dict[str, tuple[float, float]] = {}
    for col in columns:
        series = _num(history.get(col), index=history.index).replace([np.inf, -np.inf], np.nan)
        med = float(series.median()) if series.notna().any() else 0.0
        q25 = float(series.quantile(0.25)) if series.notna().any() else 0.0
        q75 = float(series.quantile(0.75)) if series.notna().any() else 1.0
        scale = q75 - q25
        if not math.isfinite(scale) or abs(scale) < 1e-12:
            std = float(series.std()) if series.notna().any() else 1.0
            scale = std if math.isfinite(std) and abs(std) >= 1e-12 else 1.0
        stats[col] = (med, scale)
    return stats


def _standardized(frame: pd.DataFrame, col: str, stats: dict[str, tuple[float, float]]) -> pd.Series:
    med, scale = stats.get(col, (0.0, 1.0))
    return ((_num(frame.get(col), index=frame.index) - med) / scale).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _materialize_blend_scores(
    frame: pd.DataFrame,
    *,
    history: pd.DataFrame,
    candidate_specs: list[CandidateSpec],
) -> tuple[pd.DataFrame, list[str]]:
    out = frame.copy()
    needed = sorted({col for spec in candidate_specs for col in spec.weights})
    stats = _robust_scaler(history, [col for col in needed if col in history.columns])
    created: list[str] = []
    for spec in candidate_specs:
        if not all(col in frame.columns for col in spec.weights):
            continue
        score = pd.Series(0.0, index=frame.index, dtype=np.float64)
        for col, weight in spec.weights.items():
            score = score + float(weight) * _standardized(frame, col, stats)
        score_col = f"score_topkobj_{spec.name}"
        out[score_col] = score.astype(np.float32)
        created.append(score_col)
    return out, created


def _objective_from_rows(rows: pd.DataFrame) -> float:
    if rows.empty:
        return float("-inf")
    values: list[float] = []
    for _, row in rows.iterrows():
        score = 0.0
        for keep_frac, weight in OBJECTIVE_KEEP_WEIGHTS.items():
            tag = f"keep{int(round(keep_frac * 100)):03d}"
            ev = float(row.get(f"{tag}_ev_after_1pct", np.nan))
            exec_margin = float(row.get(f"{tag}_exec_margin", np.nan))
            clean = float(row.get(f"{tag}_clean_exec_precision", np.nan))
            bad = float(row.get(f"{tag}_full_path_bad_mae", np.nan))
            timeout = float(row.get(f"{tag}_timeout", np.nan))
            mfe_first = float(row.get(f"{tag}_mfe_before_mae", np.nan))
            mae_first = float(row.get(f"{tag}_mae_before_mfe", np.nan))
            piece = 0.0
            piece += 1.00 * (ev if math.isfinite(ev) else 0.0)
            piece += 0.35 * (exec_margin if math.isfinite(exec_margin) else 0.0)
            piece += 0.0040 * (clean if math.isfinite(clean) else 0.0)
            piece -= 0.0040 * (bad if math.isfinite(bad) else 0.0)
            piece -= 0.0025 * (timeout if math.isfinite(timeout) else 0.0)
            piece += 0.0015 * (mfe_first if math.isfinite(mfe_first) else 0.0)
            piece -= 0.0015 * (mae_first if math.isfinite(mae_first) else 0.0)
            score += float(weight) * piece
        values.append(score)
    if not values:
        return float("-inf")
    mean_score = float(np.nanmean(values))
    worst_score = float(np.nanmin(values))
    return mean_score + 0.25 * min(0.0, worst_score)


def _monthly_selector_rows(frame: pd.DataFrame, score_col: str, selector: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for month, group in frame.groupby("month", dropna=False):
        rows.append(_selector_metrics(group, score_col, selector, str(month)))
    return pd.DataFrame(rows)


def _summarize_with_ev(folds: pd.DataFrame) -> pd.DataFrame:
    summary = _summarize(folds) if not folds.empty else pd.DataFrame()
    if summary.empty:
        return summary
    ev_cols = [col for col in folds.columns if col.startswith("keep") and col.endswith("_ev_after_1pct")]
    if not ev_cols:
        return summary
    ev = folds.groupby("selector", dropna=False)[ev_cols].mean().reset_index()
    ev = ev.rename(columns={col: f"mean_{col}" for col in ev_cols})
    existing = [col for col in ev.columns if col in summary.columns and col != "selector"]
    if existing:
        summary = summary.drop(columns=existing)
    return summary.merge(ev, on="selector", how="left")


def _select_candidate(
    history: pd.DataFrame,
    candidate_cols: list[str],
) -> tuple[str, pd.DataFrame]:
    rows: list[pd.DataFrame] = []
    for col in candidate_cols:
        metrics = _monthly_selector_rows(history, col, col.removeprefix("score_"))
        metrics["candidate_score_col"] = col
        metrics["candidate_objective"] = _objective_from_rows(metrics)
        rows.append(metrics)
    if not rows:
        raise ValueError("No candidate selector columns were available")
    all_rows = pd.concat(rows, ignore_index=True)
    ranked = (
        all_rows[["candidate_score_col", "candidate_objective"]]
        .drop_duplicates()
        .sort_values(["candidate_objective", "candidate_score_col"], ascending=[False, True])
        .reset_index(drop=True)
    )
    return str(ranked.iloc[0]["candidate_score_col"]), all_rows


def _write_markdown(out_dir: Path, manifest: dict[str, Any], summary: pd.DataFrame, selections: pd.DataFrame) -> Path:
    lines = [
        "# Promoted Cross-Asset Top-K Selector Objective",
        "",
        "## Verdict",
        "",
        f"- status: `{manifest.get('status')}`",
        f"- evaluated months: `{', '.join(manifest.get('evaluated_months') or [])}`",
        f"- first month treatment: `{manifest.get('first_month_treatment')}`",
        f"- learned selector delta vs promoted top10 EV: `{manifest.get('delta_vs_promoted', {}).get('mean_keep010_ev_after_1pct')}`",
        f"- learned selector delta vs promoted top10 bad-MAE: `{manifest.get('delta_vs_promoted', {}).get('mean_keep010_full_path_bad_mae')}`",
        "",
        "## Summary",
        "",
    ]
    if summary.empty:
        lines.append("_No selector rows produced._")
    else:
        cols = [
            "selector",
            "folds",
            "mean_keep010_ev_after_1pct",
            "mean_keep010_exec_margin",
            "mean_keep010_clean_exec_precision",
            "mean_keep010_full_path_bad_mae",
            "mean_keep010_timeout",
            "mean_keep020_ev_after_1pct",
            "mean_keep030_ev_after_1pct",
            "mean_keep030_full_path_bad_mae",
            "meta_smoke_status",
        ]
        lines.append(summary[[col for col in cols if col in summary.columns]].to_markdown(index=False))
    lines.extend(["", "## Prior-Month Selections", ""])
    if selections.empty:
        lines.append("_No learned selections._")
    else:
        display = [
            "test_month",
            "history_months",
            "selected_score_col",
            "selected_objective",
            "candidate_count",
        ]
        lines.append(selections[[col for col in display if col in selections.columns]].to_markdown(index=False))
    path = out_dir / "promoted_cross_asset_topk_selector_objective.md"
    path.write_text("\n".join(lines) + "\n")
    return path


def run_topk_selector_objective(
    *,
    promoted_predictions_path: Path,
    out_dir: Path,
    baseline_predictions_path: Path | None = None,
    baseline_score_col: str | None = None,
    promoted_score_col: str | None = None,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    promoted = _read_predictions(promoted_predictions_path)
    months = sorted(str(m) for m in promoted["month"].dropna().astype(str).unique())
    if len(months) < 2:
        raise ValueError("Need at least two months for prior-month selector objective")

    if promoted_score_col is None:
        try:
            _, promoted_score_col = _best_score_column(promoted_predictions_path.parent)
        except Exception:
            promoted_score_col = "score_meta_clean_exec" if "score_meta_clean_exec" in promoted.columns else "score_base"
    baseline: pd.DataFrame | None = None
    if baseline_predictions_path is not None and baseline_predictions_path.exists():
        baseline = _read_predictions(baseline_predictions_path)
    if baseline is not None and baseline_score_col is None:
        try:
            _, baseline_score_col = _best_score_column(baseline_predictions_path.parent)
        except Exception:
            baseline_score_col = "score_base" if "score_base" in baseline.columns else None

    fold_rows: list[dict[str, Any]] = []
    breakdown_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    scored_parts: list[pd.DataFrame] = []
    candidate_history_parts: list[pd.DataFrame] = []

    learned_score_col = "score_topk_objective_selected"
    evaluated_months = months[1:]
    for test_month in evaluated_months:
        history_months = [m for m in months if m < test_month]
        history = promoted[promoted["month"].astype(str).isin(history_months)].copy()
        test = promoted[promoted["month"].astype(str).eq(test_month)].copy()
        if history.empty or test.empty:
            continue
        combined = pd.concat([history, test], ignore_index=False)
        combined, blend_cols = _materialize_blend_scores(combined, history=history, candidate_specs=BLEND_CANDIDATES)
        history_scored = combined.loc[history.index].copy()
        test_scored = combined.loc[test.index].copy()
        candidate_cols = [*(_candidate_score_columns(test_scored)), *blend_cols]
        candidate_cols = list(dict.fromkeys([col for col in candidate_cols if col in history_scored.columns]))
        selected_col, candidate_rows = _select_candidate(history_scored, candidate_cols)
        selected_objective = float(
            candidate_rows[candidate_rows["candidate_score_col"].eq(selected_col)]["candidate_objective"].iloc[0]
        )
        test_scored[learned_score_col] = _num(test_scored[selected_col], index=test_scored.index).astype(np.float32)
        test_scored["topk_objective_selected_score_col"] = selected_col
        scored_parts.append(test_scored)
        candidate_rows.insert(0, "selector_test_month", test_month)
        candidate_rows.insert(1, "history_months", ",".join(history_months))
        candidate_history_parts.append(candidate_rows)
        selection_rows.append(
            {
                "test_month": test_month,
                "history_months": ",".join(history_months),
                "selected_score_col": selected_col,
                "selected_objective": selected_objective,
                "candidate_count": int(len(candidate_cols)),
            }
        )
        fold_rows.append(_selector_metrics(test_scored, learned_score_col, "learned_topk_ev_path_objective", test_month))
        for keep in (0.10, 0.20, 0.30):
            breakdown_rows.extend(
                _breakdown_rows(test_scored, learned_score_col, "learned_topk_ev_path_objective", test_month, keep)
            )
        if promoted_score_col in test_scored.columns:
            fold_rows.append(_selector_metrics(test_scored, promoted_score_col, f"promoted_fixed:{promoted_score_col}", test_month))
        for col in ("score_meta_clean_minus_risk", "score_meta_exec_margin_risk_blend", "score_meta_long_aware_clean_minus_risk"):
            if col in test_scored.columns and col != promoted_score_col:
                fold_rows.append(_selector_metrics(test_scored, col, f"promoted_fixed:{col}", test_month))
    if baseline is not None and baseline_score_col:
        for test_month in evaluated_months:
            base_test = baseline[baseline["month"].astype(str).eq(test_month)].copy()
            if not base_test.empty and baseline_score_col in base_test.columns:
                fold_rows.append(_selector_metrics(base_test, baseline_score_col, f"baseline_fixed:{baseline_score_col}", test_month))

    folds = pd.DataFrame(fold_rows)
    summary = _summarize_with_ev(folds)
    selections = pd.DataFrame(selection_rows)
    candidate_history = pd.concat(candidate_history_parts, ignore_index=True) if candidate_history_parts else pd.DataFrame()
    predictions = pd.concat(scored_parts, ignore_index=True) if scored_parts else pd.DataFrame()
    breakdown = pd.DataFrame(breakdown_rows)

    learned = summary[summary["selector"].eq("learned_topk_ev_path_objective")].iloc[0].to_dict() if not summary.empty and summary["selector"].eq("learned_topk_ev_path_objective").any() else {}
    promoted_fixed_name = f"promoted_fixed:{promoted_score_col}"
    promoted_fixed = summary[summary["selector"].eq(promoted_fixed_name)].iloc[0].to_dict() if not summary.empty and summary["selector"].eq(promoted_fixed_name).any() else {}
    baseline_fixed_name = f"baseline_fixed:{baseline_score_col}" if baseline_score_col else None
    baseline_fixed = summary[summary["selector"].eq(baseline_fixed_name)].iloc[0].to_dict() if baseline_fixed_name and not summary.empty and summary["selector"].eq(baseline_fixed_name).any() else {}

    def _delta(a: dict[str, Any], b: dict[str, Any], key: str) -> float | None:
        try:
            av = float(a.get(key))
            bv = float(b.get(key))
            if math.isfinite(av) and math.isfinite(bv):
                return av - bv
        except Exception:
            return None
        return None

    delta_vs_promoted = {
        key: _delta(learned, promoted_fixed, key)
        for key in (
            "mean_keep010_ev_after_1pct",
            "mean_keep010_exec_margin",
            "mean_keep010_clean_exec_precision",
            "mean_keep010_full_path_bad_mae",
            "mean_keep010_timeout",
            "mean_keep020_ev_after_1pct",
            "mean_keep030_ev_after_1pct",
            "mean_keep030_full_path_bad_mae",
        )
    }
    delta_vs_baseline = {
        key: _delta(learned, baseline_fixed, key)
        for key in (
            "mean_keep010_ev_after_1pct",
            "mean_keep010_exec_margin",
            "mean_keep010_full_path_bad_mae",
            "mean_keep030_ev_after_1pct",
            "mean_keep030_full_path_bad_mae",
        )
    }
    improved_ev = (delta_vs_promoted.get("mean_keep010_ev_after_1pct") or -1.0) > 0.0
    nonworse_risk = (delta_vs_promoted.get("mean_keep010_full_path_bad_mae") or 1.0) <= 0.01
    status = "candidate_for_deeper_meta_eval" if improved_ev and nonworse_risk else "diagnostic_or_fail"

    outputs = {
        "summary": str(out_dir / "topk_selector_objective_summary.csv"),
        "folds": str(out_dir / "topk_selector_objective_folds.csv"),
        "selections": str(out_dir / "topk_selector_objective_selections.csv"),
        "candidate_history": str(out_dir / "topk_selector_objective_candidate_history.csv"),
        "predictions": str(out_dir / "topk_selector_objective_predictions.parquet"),
        "breakdown": str(out_dir / "topk_selector_objective_breakdown.csv"),
        "manifest": str(out_dir / "manifest.json"),
    }
    summary.to_csv(outputs["summary"], index=False)
    folds.to_csv(outputs["folds"], index=False)
    selections.to_csv(outputs["selections"], index=False)
    candidate_history.to_csv(outputs["candidate_history"], index=False)
    breakdown.to_csv(outputs["breakdown"], index=False)
    if not predictions.empty:
        predictions.to_parquet(outputs["predictions"], index=False)
    else:
        pd.DataFrame().to_parquet(outputs["predictions"], index=False)

    manifest = {
        "status": status,
        "promoted_predictions_path": str(promoted_predictions_path),
        "baseline_predictions_path": str(baseline_predictions_path) if baseline_predictions_path else None,
        "promoted_score_col": promoted_score_col,
        "baseline_score_col": baseline_score_col,
        "evaluated_months": evaluated_months,
        "first_month_treatment": "skipped_as_selector_training_history",
        "objective_keep_weights": OBJECTIVE_KEEP_WEIGHTS,
        "learned_selector": learned,
        "promoted_fixed_selector": promoted_fixed,
        "baseline_fixed_selector": baseline_fixed,
        "delta_vs_promoted": delta_vs_promoted,
        "delta_vs_baseline": delta_vs_baseline,
        "outputs": outputs,
    }
    markdown = _write_markdown(out_dir, manifest, summary, selections)
    manifest["outputs"]["markdown"] = str(markdown)
    Path(outputs["manifest"]).write_text(json.dumps(_json_safe(manifest), indent=2, sort_keys=True) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--promoted-predictions-path", type=Path, default=DEFAULT_PROMOTED_SMOKE_DIR / PREDICTIONS_NAME)
    parser.add_argument("--baseline-predictions-path", type=Path, default=DEFAULT_BASELINE_SMOKE_DIR / PREDICTIONS_NAME)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--promoted-score-col", default=None)
    parser.add_argument("--baseline-score-col", default=None)
    args = parser.parse_args()
    manifest = run_topk_selector_objective(
        promoted_predictions_path=args.promoted_predictions_path,
        baseline_predictions_path=args.baseline_predictions_path,
        out_dir=args.out_dir,
        promoted_score_col=args.promoted_score_col,
        baseline_score_col=args.baseline_score_col,
    )
    print(json.dumps(_json_safe(manifest), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
