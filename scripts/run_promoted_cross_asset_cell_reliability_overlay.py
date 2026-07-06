#!/usr/bin/env python3
"""Leakage-safe cell reliability overlay for promoted cross-asset meta scores.

The promoted cross-asset features help several side x archetype cells, but they
also damage others.  This script tests a conservative meta-layer overlay:

* use only prior OOF months to classify each side x source-family cell;
* use the promoted score in cells with prior evidence of improvement;
* fall back to the baseline score in damaged or unsupported cells;
* evaluate the resulting score on the next month.

This is a smoke/proxy for a future regularized meta learner.  It is not frozen
replay evidence.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_promoted_cross_asset_cell_effects import (
    DEFAULT_BASELINE_SMOKE_DIR,
    DEFAULT_PROMOTED_HANDOFF_DIR,
    DEFAULT_PROMOTED_SMOKE_DIR,
    GROUP_COLUMNS,
    KEY_COLUMNS,
    PREDICTIONS_NAME,
    _best_score_column,
    _cell_rows,
    _classify_cells,
    _json_safe,
    _read_json,
    _num,
)
from scripts.run_s52_train_meta_regime_handoff_smoke import (
    TOP_KEEP_FRACTIONS,
    _breakdown_rows,
    _selector_metrics,
    _summarize,
)


DEFAULT_OUT_DIR = DEFAULT_PROMOTED_HANDOFF_DIR / "promoted_cross_asset_cell_reliability_overlay_v1"


def _read_predictions(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_parquet(path)


def _merge_predictions(
    baseline: pd.DataFrame,
    promoted: pd.DataFrame,
    *,
    baseline_score_col: str,
    promoted_score_col: str,
) -> pd.DataFrame:
    base_cols = list(dict.fromkeys([*KEY_COLUMNS, "month", *GROUP_COLUMNS, baseline_score_col]))
    promoted_cols = list(dict.fromkeys([*KEY_COLUMNS, promoted_score_col]))
    missing_base = [col for col in base_cols if col not in baseline.columns]
    missing_promoted = [col for col in promoted_cols if col not in promoted.columns]
    if missing_base:
        raise ValueError(f"Baseline predictions missing columns: {missing_base}")
    if missing_promoted:
        raise ValueError(f"Promoted predictions missing columns: {missing_promoted}")
    merged = baseline.copy()
    rename_base = {baseline_score_col: "score_overlay_baseline"}
    merged = merged.rename(columns=rename_base)
    promoted_score = promoted[promoted_cols].rename(columns={promoted_score_col: "score_overlay_promoted"})
    merged = merged.merge(promoted_score, on=list(KEY_COLUMNS), how="inner", validate="one_to_one")
    return merged


def _build_cell_policy(
    history_baseline: pd.DataFrame,
    history_promoted: pd.DataFrame,
    *,
    baseline_score_col: str,
    promoted_score_col: str,
    min_valid_rows: int,
    min_months: int,
    min_clean_rows: int,
    min_positive_rows: int,
    max_asset_share: float,
    max_week_share: float,
    min_promote_cell_value: float,
    require_positive_exec_delta: bool,
    max_promote_bad_mae_delta: float,
    max_promote_timeout_delta: float,
) -> tuple[dict[tuple[str, str], str], pd.DataFrame]:
    cells = _cell_rows(
        history_baseline,
        history_promoted,
        baseline_score_col=baseline_score_col,
        promoted_score_col=promoted_score_col,
    )
    cells = _classify_cells(
        cells,
        min_valid_rows=min_valid_rows,
        min_months=min_months,
        min_clean_rows=min_clean_rows,
        min_positive_rows=min_positive_rows,
        max_asset_share=max_asset_share,
        max_week_share=max_week_share,
    )
    policy: dict[tuple[str, str], str] = {}
    if cells.empty:
        return policy, cells
    keep10 = cells[cells["keep_frac"].eq(0.10)].copy()
    if keep10.empty:
        keep10 = cells.sort_values("keep_frac").drop_duplicates(list(GROUP_COLUMNS), keep="first")
    for _, row in keep10.iterrows():
        key = tuple(str(row[col]) for col in GROUP_COLUMNS)
        cell_value = float(row.get("cell_value_score", 0.0))
        exec_ok = float(row.get("delta_exec_margin", 0.0)) > 0.0 if require_positive_exec_delta else True
        risk_ok = (
            float(row.get("delta_full_path_bad_mae", 0.0)) <= float(max_promote_bad_mae_delta)
            and float(row.get("delta_timeout", 0.0)) <= float(max_promote_timeout_delta)
        )
        if bool(row.get("beneficial_supported_cell", False)) and cell_value >= float(min_promote_cell_value) and exec_ok and risk_ok:
            policy[key] = "promoted"
        elif bool(row.get("catastrophic_supported_degradation", False)):
            policy[key] = "baseline"
        elif bool(row.get("support_pass", False)) and cell_value >= max(float(min_promote_cell_value), 1.25):
            # Soft-positive cells are allowed only if path risk is not worse.
            if exec_ok and risk_ok:
                policy[key] = "promoted"
            else:
                policy[key] = "baseline"
        else:
            policy[key] = "baseline"
    return policy, cells


def _apply_policy(frame: pd.DataFrame, policy: dict[tuple[str, str], str]) -> pd.DataFrame:
    out = frame.copy()
    use_promoted = []
    for _, row in out[list(GROUP_COLUMNS)].astype(str).iterrows():
        key = tuple(row[col] for col in GROUP_COLUMNS)
        use_promoted.append(policy.get(key, "baseline") == "promoted")
    use_promoted_s = pd.Series(use_promoted, index=out.index)
    promoted = _num(out.get("score_overlay_promoted"), index=out.index)
    baseline = _num(out.get("score_overlay_baseline"), index=out.index)
    out["score_cell_reliability_overlay"] = promoted.where(use_promoted_s & promoted.notna(), baseline)
    out["cell_reliability_uses_promoted"] = use_promoted_s.astype(np.float32)
    return out


def _summarize_policy_rows(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return _summarize(out)


def _write_markdown(out_dir: Path, manifest: dict[str, Any], summary: pd.DataFrame, cell_policies: pd.DataFrame) -> Path:
    lines = [
        "# Promoted Cross-Asset Cell Reliability Overlay",
        "",
        "## Verdict",
        "",
        f"- status: `{manifest.get('status')}`",
        f"- scored months: `{', '.join(manifest.get('scored_months') or [])}`",
        f"- baseline selector: `{manifest.get('baseline_selector')}`",
        f"- promoted selector: `{manifest.get('promoted_selector')}`",
        f"- delta vs baseline top10 exec: `{manifest.get('delta_vs_baseline', {}).get('mean_keep010_exec_margin')}`",
        f"- delta vs baseline top10 bad-MAE: `{manifest.get('delta_vs_baseline', {}).get('mean_keep010_full_path_bad_mae')}`",
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
            "mean_keep010_exec_margin",
            "mean_keep010_clean_exec_precision",
            "mean_keep010_full_path_bad_mae",
            "mean_keep010_timeout",
            "mean_keep010_oracle_recall",
            "mean_keep030_exec_margin",
            "mean_keep030_full_path_bad_mae",
            "mean_keep030_timeout",
            "meta_smoke_status",
        ]
        lines.append(summary[[col for col in cols if col in summary.columns]].to_markdown(index=False))
    lines.extend(["", "## Cell Policy Rows", ""])
    if cell_policies.empty:
        lines.append("_No prior cell policy rows._")
    else:
        display_cols = [
            "test_month",
            "source_months",
            "policy_cells",
            "promoted_cells",
            "baseline_cells",
            "history_supported_cells",
            "history_beneficial_cells",
            "history_damaged_cells",
        ]
        lines.append(cell_policies[[col for col in display_cols if col in cell_policies.columns]].to_markdown(index=False))
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This overlay uses only prior OOF months to decide whether each side x source-family cell should use the promoted score.",
            "With the current two-month OOF prediction artifact, only June has a non-empty history. Treat it as a direction check, not a final gate.",
        ]
    )
    path = out_dir / "promoted_cross_asset_cell_reliability_overlay.md"
    path.write_text("\n".join(lines) + "\n")
    return path


def run_overlay(
    *,
    baseline_smoke_dir: Path,
    promoted_smoke_dir: Path,
    out_dir: Path,
    min_valid_rows: int = 30,
    min_months: int = 1,
    min_clean_rows: int = 5,
    min_positive_rows: int = 5,
    max_asset_share: float = 0.80,
    max_week_share: float = 0.80,
    min_promote_cell_value: float = 1.50,
    require_positive_exec_delta: bool = True,
    max_promote_bad_mae_delta: float = 0.02,
    max_promote_timeout_delta: float = 0.02,
) -> dict[str, Any]:
    baseline_selector, baseline_score_col = _best_score_column(baseline_smoke_dir)
    promoted_selector, promoted_score_col = _best_score_column(promoted_smoke_dir)
    baseline = _read_predictions(baseline_smoke_dir / PREDICTIONS_NAME)
    promoted = _read_predictions(promoted_smoke_dir / PREDICTIONS_NAME)
    merged = _merge_predictions(
        baseline,
        promoted,
        baseline_score_col=baseline_score_col,
        promoted_score_col=promoted_score_col,
    )
    months = sorted(str(m) for m in merged["month"].dropna().astype(str).unique())
    fold_rows: list[dict[str, Any]] = []
    breakdown_rows: list[dict[str, Any]] = []
    cell_policy_rows: list[dict[str, Any]] = []
    scored_frames: list[pd.DataFrame] = []
    for test_month in months:
        history_months = [m for m in months if m < test_month]
        valid = merged[merged["month"].astype(str).eq(test_month)].copy()
        if not history_months:
            policy: dict[tuple[str, str], str] = {}
            history_cells = pd.DataFrame()
        else:
            hist_base = baseline[baseline["month"].astype(str).isin(history_months)].copy()
            hist_prom = promoted[promoted["month"].astype(str).isin(history_months)].copy()
            policy, history_cells = _build_cell_policy(
                hist_base,
                hist_prom,
                baseline_score_col=baseline_score_col,
                promoted_score_col=promoted_score_col,
                min_valid_rows=min_valid_rows,
                min_months=min_months,
                min_clean_rows=min_clean_rows,
                min_positive_rows=min_positive_rows,
                max_asset_share=max_asset_share,
                max_week_share=max_week_share,
                min_promote_cell_value=min_promote_cell_value,
                require_positive_exec_delta=require_positive_exec_delta,
                max_promote_bad_mae_delta=max_promote_bad_mae_delta,
                max_promote_timeout_delta=max_promote_timeout_delta,
            )
        scored = _apply_policy(valid, policy)
        scored_frames.append(scored)
        selector = "cell_reliability_overlay"
        fold_rows.append(_selector_metrics(scored, "score_cell_reliability_overlay", selector, test_month))
        for keep_frac in (0.10, 0.20, 0.30):
            breakdown_rows.extend(_breakdown_rows(scored, "score_cell_reliability_overlay", selector, test_month, keep_frac))
        cell_policy_rows.append(
            {
                "test_month": str(test_month),
                "source_months": ",".join(history_months),
                "policy_cells": int(len(policy)),
                "promoted_cells": int(sum(1 for value in policy.values() if value == "promoted")),
                "baseline_cells": int(sum(1 for value in policy.values() if value == "baseline")),
                "history_cell_rows": int(len(history_cells)),
                "history_supported_cells": int(history_cells["support_pass"].sum()) if "support_pass" in history_cells.columns else 0,
                "history_beneficial_cells": int(history_cells["beneficial_supported_cell"].sum())
                if "beneficial_supported_cell" in history_cells.columns
                else 0,
                "history_damaged_cells": int(history_cells["catastrophic_supported_degradation"].sum())
                if "catastrophic_supported_degradation" in history_cells.columns
                else 0,
            }
        )
    folds = pd.DataFrame(fold_rows)
    summary = _summarize_policy_rows(fold_rows)
    breakdown = pd.DataFrame(breakdown_rows)
    cell_policies = pd.DataFrame(cell_policy_rows)
    predictions = pd.concat(scored_frames, ignore_index=True) if scored_frames else pd.DataFrame()
    out_dir.mkdir(parents=True, exist_ok=True)
    folds.to_csv(out_dir / "promoted_cross_asset_cell_reliability_overlay_folds.csv", index=False)
    summary.to_csv(out_dir / "promoted_cross_asset_cell_reliability_overlay_summary.csv", index=False)
    breakdown.to_csv(out_dir / "promoted_cross_asset_cell_reliability_overlay_breakdown.csv", index=False)
    cell_policies.to_csv(out_dir / "promoted_cross_asset_cell_reliability_overlay_cell_policies.csv", index=False)
    if not predictions.empty:
        predictions.to_parquet(out_dir / "promoted_cross_asset_cell_reliability_overlay_predictions.parquet", index=False)
    best = summary.iloc[0].to_dict() if not summary.empty else {}
    baseline_manifest = _read_json(baseline_smoke_dir / "manifest.json")
    baseline_best = baseline_manifest.get("best_selector") or {}

    def metric(row: dict[str, Any], key: str, default: float = float("nan")) -> float:
        try:
            return float(row.get(key, default))
        except Exception:
            return default

    delta_vs_baseline = {
        key: metric(best, key) - metric(baseline_best, key)
        for key in (
            "mean_keep010_exec_margin",
            "mean_keep010_clean_exec_precision",
            "mean_keep010_full_path_bad_mae",
            "mean_keep010_timeout",
            "mean_keep010_oracle_recall",
            "mean_keep030_exec_margin",
            "mean_keep030_clean_exec_precision",
            "mean_keep030_full_path_bad_mae",
            "mean_keep030_timeout",
            "mean_keep030_oracle_recall",
            "mean_auc_clean_exec",
            "mean_ap_clean_exec",
        )
    }
    status = (
        "candidate_for_deeper_meta_eval"
        if best
        and delta_vs_baseline["mean_keep010_exec_margin"] > 0.0
        and delta_vs_baseline["mean_keep010_full_path_bad_mae"] <= 0.0
        and delta_vs_baseline["mean_keep010_clean_exec_precision"] >= 0.0
        and delta_vs_baseline["mean_keep030_exec_margin"] >= -0.00025
        and delta_vs_baseline["mean_keep030_full_path_bad_mae"] <= 0.010
        else "diagnostic_or_fail"
    )
    manifest = {
        "generated_by": "run_promoted_cross_asset_cell_reliability_overlay",
        "baseline_smoke_dir": str(baseline_smoke_dir),
        "promoted_smoke_dir": str(promoted_smoke_dir),
        "baseline_selector": baseline_selector,
        "baseline_score_col": baseline_score_col,
        "promoted_selector": promoted_selector,
        "promoted_score_col": promoted_score_col,
        "months": months,
        "scored_months": months,
        "rows": int(len(merged)),
        "status": status,
        "best_selector": _json_safe(best),
        "baseline_best_selector": _json_safe(baseline_best),
        "delta_vs_baseline": _json_safe(delta_vs_baseline),
        "support_rule": {
            "min_valid_rows": int(min_valid_rows),
            "min_months": int(min_months),
            "min_clean_rows": int(min_clean_rows),
            "min_positive_rows": int(min_positive_rows),
            "max_asset_share": float(max_asset_share),
            "max_week_share": float(max_week_share),
            "min_promote_cell_value": float(min_promote_cell_value),
            "require_positive_exec_delta": bool(require_positive_exec_delta),
            "max_promote_bad_mae_delta": float(max_promote_bad_mae_delta),
            "max_promote_timeout_delta": float(max_promote_timeout_delta),
        },
        "leakage_contract": {
            "cell_policy_source": "prior OOF months only",
            "first_month_policy": "fallback_to_baseline_score_because_no_prior_oof_month_exists",
            "selection": "top-k using score_cell_reliability_overlay",
            "labels_used_for": "prior-month cell reliability and offline validation metrics only",
        },
        "outputs": {
            "folds": str(out_dir / "promoted_cross_asset_cell_reliability_overlay_folds.csv"),
            "summary": str(out_dir / "promoted_cross_asset_cell_reliability_overlay_summary.csv"),
            "breakdown": str(out_dir / "promoted_cross_asset_cell_reliability_overlay_breakdown.csv"),
            "cell_policies": str(out_dir / "promoted_cross_asset_cell_reliability_overlay_cell_policies.csv"),
            "predictions": str(out_dir / "promoted_cross_asset_cell_reliability_overlay_predictions.parquet"),
            "markdown": str(out_dir / "promoted_cross_asset_cell_reliability_overlay.md"),
            "json": str(out_dir / "manifest.json"),
        },
    }
    markdown = _write_markdown(out_dir, manifest, summary, cell_policies)
    manifest["outputs"]["markdown"] = str(markdown)
    (out_dir / "manifest.json").write_text(json.dumps(_json_safe(manifest), indent=2, sort_keys=True))
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-smoke-dir", type=Path, default=DEFAULT_BASELINE_SMOKE_DIR)
    parser.add_argument("--promoted-smoke-dir", type=Path, default=DEFAULT_PROMOTED_SMOKE_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--min-valid-rows", type=int, default=30)
    parser.add_argument("--min-months", type=int, default=1)
    parser.add_argument("--min-clean-rows", type=int, default=5)
    parser.add_argument("--min-positive-rows", type=int, default=5)
    parser.add_argument("--max-asset-share", type=float, default=0.80)
    parser.add_argument("--max-week-share", type=float, default=0.80)
    parser.add_argument("--min-promote-cell-value", type=float, default=1.50)
    parser.add_argument("--allow-negative-exec-delta", action="store_true")
    parser.add_argument("--max-promote-bad-mae-delta", type=float, default=0.02)
    parser.add_argument("--max-promote-timeout-delta", type=float, default=0.02)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = run_overlay(
        baseline_smoke_dir=args.baseline_smoke_dir,
        promoted_smoke_dir=args.promoted_smoke_dir,
        out_dir=args.out_dir,
        min_valid_rows=args.min_valid_rows,
        min_months=args.min_months,
        min_clean_rows=args.min_clean_rows,
        min_positive_rows=args.min_positive_rows,
        max_asset_share=args.max_asset_share,
        max_week_share=args.max_week_share,
        min_promote_cell_value=args.min_promote_cell_value,
        require_positive_exec_delta=not bool(args.allow_negative_exec_delta),
        max_promote_bad_mae_delta=args.max_promote_bad_mae_delta,
        max_promote_timeout_delta=args.max_promote_timeout_delta,
    )
    print(json.dumps(_json_safe({"event": "promoted_cross_asset_cell_reliability_overlay_done", **manifest}), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
