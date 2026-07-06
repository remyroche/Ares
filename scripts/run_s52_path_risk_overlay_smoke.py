#!/usr/bin/env python3
"""Leakage-safe path-risk overlay smoke for S52 ranker ledgers.

The overlay trains on fit months only and predicts whether a candidate has
unacceptable pre-MFE path pain. It then tests whether risk-adjusted reranking can
improve top-k path quality on the holdout month without changing label geometry.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from lightgbm import LGBMClassifier

    _LIGHTGBM_AVAILABLE = True
except Exception:  # pragma: no cover
    LGBMClassifier = None
    _LIGHTGBM_AVAILABLE = False

from scripts.report_s52_state_overlay_ablation import _json_safe, _metrics, _parse_csv, _prefix  # noqa: E402
from scripts.run_s52_ranker_smoke import _state_feature_columns  # noqa: E402


DEFAULT_LEDGER = Path(
    "data_perp/reports/ae_gmm_archetype_validation_status_20260704/"
    "s52_ranker_smoke_best_archetype_overlay_v1/s52_ranker_smoke_scored_ledger.parquet"
)
DEFAULT_OUTPUT_DIR = Path("data_perp/reports/s52_path_risk_overlay_smoke_v1")


def _safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def _make_dirty_label(
    frame: pd.DataFrame,
    *,
    mae_before_threshold: float,
    adverse_threshold: float,
    underwater_bars_threshold: float,
) -> pd.Series:
    mae_before = pd.to_numeric(frame.get("mae_1r_before_mfe_1r"), errors="coerce").fillna(0.0)
    adverse = pd.to_numeric(frame.get("max_adverse_before_mfe_1r"), errors="coerce").fillna(0.0)
    underwater = pd.to_numeric(frame.get("underwater_bars_before_mfe_1r"), errors="coerce").fillna(0.0)
    return (
        mae_before.gt(float(mae_before_threshold))
        | adverse.gt(float(adverse_threshold))
        | underwater.gt(float(underwater_bars_threshold))
    ).astype(int)


def _feature_columns(ledger: pd.DataFrame, *, include_score: bool = True) -> list[str]:
    cols = _state_feature_columns(ledger.columns)
    extras = ["score", "side"]
    out: list[str] = []
    for col in extras + cols if include_score else cols:
        if col in ledger.columns and col not in out:
            out.append(col)
    return out


def _prepare_x(frame: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    x = frame.reindex(columns=feature_cols).copy()
    for col in x.columns:
        x[col] = pd.to_numeric(x[col], errors="coerce")
    return x.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)


def _fit_predict_risk(
    ledger: pd.DataFrame,
    *,
    fit_months: list[str],
    feature_cols: list[str],
    label_col: str,
    side_specific: bool,
    seed: int,
) -> np.ndarray:
    if not _LIGHTGBM_AVAILABLE:
        raise RuntimeError("lightgbm is required for this smoke")
    pred = np.full(len(ledger), np.nan, dtype=np.float32)
    month = ledger["month"].astype(str)
    fit_mask = month.isin(fit_months).to_numpy()
    sides = sorted(ledger["side_name"].astype(str).unique().tolist()) if side_specific else ["__all__"]
    for side in sides:
        if side_specific:
            local_fit = fit_mask & ledger["side_name"].astype(str).eq(side).to_numpy()
            local_all = ledger["side_name"].astype(str).eq(side).to_numpy()
        else:
            local_fit = fit_mask
            local_all = np.ones(len(ledger), dtype=bool)
        y = pd.to_numeric(ledger.loc[local_fit, label_col], errors="coerce").fillna(0).astype(int)
        if len(y) < 100 or y.nunique() < 2:
            fill = float(y.mean()) if len(y) else 0.5
            pred[local_all] = fill
            continue
        model = LGBMClassifier(
            n_estimators=240,
            learning_rate=0.035,
            num_leaves=31,
            min_child_samples=80,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.05,
            reg_lambda=0.50,
            random_state=int(seed),
            n_jobs=4,
            verbose=-1,
        )
        model.fit(_prepare_x(ledger.loc[local_fit], feature_cols), y)
        proba = model.predict_proba(_prepare_x(ledger.loc[local_all], feature_cols))[:, 1]
        pred[local_all] = proba.astype(np.float32)
    fill = np.nanmean(pred) if np.isfinite(pred).any() else 0.5
    return np.where(np.isfinite(pred), pred, fill).astype(np.float32)


def _baseline_selected_count(group: pd.DataFrame, selected_col: str) -> int:
    return int(pd.to_numeric(group[selected_col], errors="coerce").fillna(0.0).gt(0.5).sum())


def _select_overlay(
    ledger: pd.DataFrame,
    *,
    selected_col: str,
    score_col: str,
    adjusted_score_col: str,
    risk_col: str,
    mode: str,
    risk_threshold: float,
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for _, group in ledger.groupby(["month", "side_name"], observed=True, dropna=False, sort=False):
        budget = _baseline_selected_count(group, selected_col)
        if budget <= 0:
            continue
        if mode == "filter":
            selected = group.loc[group[selected_col].astype(bool)]
            selected = selected[pd.to_numeric(selected[risk_col], errors="coerce").le(float(risk_threshold))]
            parts.append(selected)
            continue
        eligible = group[pd.to_numeric(group[risk_col], errors="coerce").le(float(risk_threshold))]
        if eligible.empty:
            continue
        order_col = adjusted_score_col if mode == "risk_adjusted_refill" else score_col
        parts.append(eligible.sort_values(order_col, ascending=False, kind="mergesort").head(budget))
    return pd.concat(parts, ignore_index=False) if parts else ledger.iloc[0:0].copy()


def _objective(row: dict[str, Any], prefix: str, *, min_retention: float) -> float:
    precision = _safe_float(row.get(f"{prefix}_ev_weighted_first_touch_precision"), 0.0)
    mean_u = _safe_float(row.get(f"{prefix}_mean_u"), -0.02)
    mae_before = _safe_float(row.get(f"{prefix}_mae_before_mfe_1r_rate"), 1.0)
    adverse = _safe_float(row.get(f"{prefix}_mean_max_adverse_before_mfe_1r"), 3.0)
    underwater = _safe_float(row.get(f"{prefix}_mean_underwater_bars_before_mfe"), 25.0)
    selected_rows = _safe_float(row.get(f"{prefix}_selected_rows"), 0.0)
    baseline_rows = _safe_float(row.get("fit_baseline_selected_rows"), 1.0)
    retention = selected_rows / max(baseline_rows, 1.0)
    return float(
        precision
        + 0.50 * mean_u
        - 0.40 * max(mae_before - 0.35, 0.0)
        - 0.25 * max(adverse - 1.50, 0.0)
        - 0.025 * max(underwater - 10.0, 0.0)
        - 0.75 * max(float(min_retention) - retention, 0.0)
    )


def build_report(
    *,
    ledger_path: Path,
    output_dir: Path,
    variant: str,
    selected_col: str,
    fit_months: list[str],
    holdout_month: str,
    round_trip_cost: float,
    mae_before_threshold: float,
    adverse_threshold: float,
    underwater_bars_threshold: float,
    side_specific: bool,
    min_retention: float,
    seed: int,
) -> dict[str, Any]:
    ledger = pd.read_parquet(ledger_path)
    if variant:
        ledger = ledger[ledger["variant"].astype(str).eq(str(variant))].copy()
    if ledger.empty:
        raise ValueError(f"No rows found for variant {variant!r}")
    output_dir.mkdir(parents=True, exist_ok=True)
    ledger["month"] = ledger["month"].astype(str)
    if "side_name" not in ledger.columns:
        side = pd.to_numeric(ledger.get("side", 1.0), errors="coerce").fillna(1.0)
        ledger["side_name"] = np.where(side.to_numpy(dtype=np.float64) < 0.0, "short", "long")
    feature_cols = _feature_columns(ledger)
    ledger["path_risk_dirty"] = _make_dirty_label(
        ledger,
        mae_before_threshold=float(mae_before_threshold),
        adverse_threshold=float(adverse_threshold),
        underwater_bars_threshold=float(underwater_bars_threshold),
    )
    ledger["path_risk_prob"] = _fit_predict_risk(
        ledger,
        fit_months=fit_months,
        feature_cols=feature_cols,
        label_col="path_risk_dirty",
        side_specific=bool(side_specific),
        seed=int(seed),
    )
    fit_selected = ledger[ledger["month"].isin(fit_months) & ledger[selected_col].astype(bool)].copy()
    holdout_selected = ledger[ledger["month"].eq(str(holdout_month)) & ledger[selected_col].astype(bool)].copy()
    rows: list[dict[str, Any]] = []
    thresholds = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.80, 0.90]
    penalties = [0.0, 0.25, 0.50, 0.75, 1.00, 1.50, 2.00]
    for mode in ("filter", "refill", "risk_adjusted_refill"):
        for threshold in thresholds:
            for penalty in (penalties if mode == "risk_adjusted_refill" else [0.0]):
                local = ledger.copy()
                local["risk_adjusted_score"] = pd.to_numeric(local["score"], errors="coerce").fillna(0.0) - float(
                    penalty
                ) * pd.to_numeric(local["path_risk_prob"], errors="coerce").fillna(0.5)
                selected = _select_overlay(
                    local,
                    selected_col=selected_col,
                    score_col="score",
                    adjusted_score_col="risk_adjusted_score",
                    risk_col="path_risk_prob",
                    mode=mode,
                    risk_threshold=float(threshold),
                )
                fit = selected[selected["month"].isin(fit_months)].copy()
                holdout = selected[selected["month"].eq(str(holdout_month))].copy()
                row: dict[str, Any] = {
                    "mode": mode,
                    "risk_threshold": float(threshold),
                    "risk_penalty": float(penalty),
                }
                row.update(_prefix("fit_baseline", _metrics(fit_selected, round_trip_cost=float(round_trip_cost))))
                row.update(_prefix("fit_overlay", _metrics(fit, round_trip_cost=float(round_trip_cost))))
                row.update(_prefix("holdout_baseline", _metrics(holdout_selected, round_trip_cost=float(round_trip_cost))))
                row.update(_prefix("holdout_overlay", _metrics(holdout, round_trip_cost=float(round_trip_cost))))
                row["fit_retention"] = row["fit_overlay_selected_rows"] / max(row["fit_baseline_selected_rows"], 1.0)
                row["holdout_retention"] = row["holdout_overlay_selected_rows"] / max(
                    row["holdout_baseline_selected_rows"], 1.0
                )
                row["fit_objective"] = _objective(row, "fit_overlay", min_retention=float(min_retention))
                row["holdout_objective"] = _objective(
                    {**row, "fit_baseline_selected_rows": row["holdout_baseline_selected_rows"]},
                    "holdout_overlay",
                    min_retention=float(min_retention),
                )
                rows.append(row)
    summary = pd.DataFrame(rows).sort_values(
        ["fit_objective", "holdout_objective"], ascending=[False, False]
    ).reset_index(drop=True)
    best = summary.iloc[0].to_dict() if not summary.empty else {}
    paths = {
        "summary": output_dir / "s52_path_risk_overlay_summary.csv",
        "scored_ledger": output_dir / "s52_path_risk_overlay_scored_ledger.parquet",
        "manifest": output_dir / "manifest.json",
        "report": output_dir / "s52_path_risk_overlay_smoke.md",
    }
    summary.to_csv(paths["summary"], index=False)
    ledger.to_parquet(paths["scored_ledger"], index=False)
    manifest = {
        "scope": "s52_path_risk_overlay_smoke",
        "ledger_path": str(ledger_path),
        "output_dir": str(output_dir),
        "variant": str(variant),
        "selected_col": str(selected_col),
        "fit_months": fit_months,
        "holdout_month": str(holdout_month),
        "round_trip_cost": float(round_trip_cost),
        "mae_before_threshold": float(mae_before_threshold),
        "adverse_threshold": float(adverse_threshold),
        "underwater_bars_threshold": float(underwater_bars_threshold),
        "side_specific": bool(side_specific),
        "min_retention": float(min_retention),
        "feature_count": int(len(feature_cols)),
        "feature_cols": feature_cols,
        "best_fit_row": _json_safe(best),
        "outputs": {key: str(value) for key, value in paths.items()},
    }
    paths["manifest"].write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")
    cols = [
        "mode",
        "risk_threshold",
        "risk_penalty",
        "fit_objective",
        "holdout_objective",
        "fit_retention",
        "holdout_retention",
        "holdout_overlay_selected_rows",
        "holdout_overlay_ev_weighted_first_touch_precision",
        "holdout_overlay_mean_u",
        "holdout_overlay_mae_before_mfe_1r_rate",
        "holdout_overlay_mean_max_adverse_before_mfe_1r",
        "holdout_overlay_mean_underwater_bars_before_mfe",
        "holdout_baseline_selected_rows",
        "holdout_baseline_ev_weighted_first_touch_precision",
        "holdout_baseline_mean_u",
        "holdout_baseline_mae_before_mfe_1r_rate",
        "holdout_baseline_mean_max_adverse_before_mfe_1r",
        "holdout_baseline_mean_underwater_bars_before_mfe",
    ]
    lines = [
        "# S52 Path-Risk Overlay Smoke",
        "",
        f"Ledger: `{ledger_path}`",
        f"Variant: `{variant}`",
        f"Fit months: `{', '.join(fit_months)}`",
        f"Holdout month: `{holdout_month}`",
        f"Feature count: `{len(feature_cols)}`",
        "",
        "## Top Fit-Selected Overlays",
        "",
        summary[[c for c in cols if c in summary.columns]].head(20).to_markdown(index=False)
        if not summary.empty
        else "No overlays evaluated.",
        "",
        "## Outputs",
        "",
        f"- Summary: `{paths['summary']}`",
        f"- Scored ledger: `{paths['scored_ledger']}`",
        f"- Manifest: `{paths['manifest']}`",
    ]
    paths["report"].write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger-path", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--variant", default="ranker_side_specific_timestamp")
    parser.add_argument("--selected-col", default="selected_top10")
    parser.add_argument("--fit-months", default="2026-04,2026-05")
    parser.add_argument("--holdout-month", default="2026-06")
    parser.add_argument("--round-trip-cost", type=float, default=0.0100)
    parser.add_argument("--mae-before-threshold", type=float, default=0.35)
    parser.add_argument("--adverse-threshold", type=float, default=1.50)
    parser.add_argument("--underwater-bars-threshold", type=float, default=10.0)
    parser.add_argument("--side-specific", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min-retention", type=float, default=0.80)
    parser.add_argument("--seed", type=int, default=52)
    args = parser.parse_args()
    manifest = build_report(
        ledger_path=args.ledger_path,
        output_dir=args.output_dir,
        variant=str(args.variant),
        selected_col=str(args.selected_col),
        fit_months=_parse_csv(args.fit_months, ()),
        holdout_month=str(args.holdout_month),
        round_trip_cost=float(args.round_trip_cost),
        mae_before_threshold=float(args.mae_before_threshold),
        adverse_threshold=float(args.adverse_threshold),
        underwater_bars_threshold=float(args.underwater_bars_threshold),
        side_specific=bool(args.side_specific),
        min_retention=float(args.min_retention),
        seed=int(args.seed),
    )
    print(json.dumps(_json_safe(manifest), indent=2))


if __name__ == "__main__":
    main()
