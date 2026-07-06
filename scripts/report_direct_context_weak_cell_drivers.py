#!/usr/bin/env python3
"""Diagnose context-feature drivers in weak side x archetype cells.

This report consumes the direct train_meta context handoff and the risk-aware
train_meta smoke predictions.  For cells where a selector worsens full-SL, it
reconstructs the selector's top-k rows and compares their context-feature
distribution with the full candidate pool for the same month/side/archetype.

The output is diagnostic only.  It does not create admission flags and does not
feed accepted-cell metadata back into model inputs.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_FEATURE_SET_DIR = Path(
    "data_perp/reports/contextual_tp_sl_ablation_workflow_v14_runtime_health_20260701/"
    "direct_cross_asset_meta_context_v1/train_meta_direct_context_feature_set_v1"
)
DEFAULT_RISK_DIR = DEFAULT_FEATURE_SET_DIR / "risk_aware_train_meta_smoke_v1"
DEFAULT_HANDOFF = DEFAULT_FEATURE_SET_DIR / "train_meta_direct_context_handoff.parquet"
DEFAULT_FEATURE_MANIFEST = DEFAULT_FEATURE_SET_DIR / "train_meta_direct_context_feature_manifest.json"
DEFAULT_PREDICTIONS = DEFAULT_RISK_DIR / "risk_aware_train_meta_predictions.parquet"
DEFAULT_WORST_CELLS = DEFAULT_RISK_DIR / "risk_aware_train_meta_worst_cell_tradeoffs.csv"
DEFAULT_ACCEPTED_CELLS = DEFAULT_FEATURE_SET_DIR / "train_meta_direct_context_accepted_cells.csv"
DEFAULT_OUT_DIR = DEFAULT_RISK_DIR / "weak_cell_driver_report_v1"
DEFAULT_SELECTORS = (
    "s12_ev_clean_strong_risk",
    "s13_ev_clean_fullsl_neutral_timeout",
    "s14_cell_prior_fullsl_s12",
    "s15_cell_prior_fullsl_timeout_s12",
    "s16_cell_prior_clean_risk_s12",
    "s7_clean_strong_fullsl",
)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(float(value)) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def _load_manifest(path: Path) -> tuple[list[str], dict[str, list[str]]]:
    payload = json.loads(path.read_text())
    feature_columns = [str(c) for c in payload.get("feature_columns", [])]
    families = {
        str(k): [str(c) for c in v]
        for k, v in dict(payload.get("families", {})).items()
        if isinstance(v, list)
    }
    return feature_columns, families


def _feature_family_map(families: dict[str, list[str]]) -> dict[str, str]:
    base_features = set(families.get("f00_score_only", []))
    mapping: dict[str, str] = {}
    for family, cols in families.items():
        if family == "f00_score_only":
            continue
        for col in cols:
            if col not in base_features:
                mapping.setdefault(col, family)
    for col in base_features:
        mapping.setdefault(col, "f00_score_only")
    return mapping


def _safe_corr(a: pd.Series, b: pd.Series) -> float:
    av = pd.to_numeric(a, errors="coerce")
    bv = pd.to_numeric(b, errors="coerce")
    mask = av.notna() & bv.notna()
    if int(mask.sum()) < 20:
        return float("nan")
    ax = av[mask].to_numpy(dtype="float64")
    bx = bv[mask].to_numpy(dtype="float64")
    if float(np.nanstd(ax)) <= 1e-12 or float(np.nanstd(bx)) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(ax, bx)[0, 1])


def _cell_key(row: pd.Series) -> tuple[str, str, str]:
    return str(row["month"]), str(row["side_name"]), str(row["source_archetype"])


def _select_topk(pool: pd.DataFrame, selector: str, top_frac: float) -> pd.DataFrame:
    score_col = f"score_{selector}"
    valid = pool[pd.to_numeric(pool[score_col], errors="coerce").notna()].copy()
    if valid.empty:
        return valid
    n = max(1, int(math.ceil(len(valid) * float(top_frac))))
    return valid.assign(_score=pd.to_numeric(valid[score_col], errors="coerce")).sort_values(
        "_score", ascending=False
    ).head(n)


def _feature_driver_rows(
    *,
    selector: str,
    month: str,
    side_name: str,
    source_archetype: str,
    pool: pd.DataFrame,
    selected: pd.DataFrame,
    feature_columns: list[str],
    family_by_feature: dict[str, str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if pool.empty or selected.empty:
        return rows
    for feature in feature_columns:
        if feature not in pool.columns:
            continue
        pool_values = pd.to_numeric(pool[feature], errors="coerce")
        selected_values = pd.to_numeric(selected[feature], errors="coerce")
        finite_pool = pool_values[np.isfinite(pool_values)]
        finite_selected = selected_values[np.isfinite(selected_values)]
        if len(finite_pool) < 50 or len(finite_selected) < 5:
            continue
        pool_std = float(finite_pool.std(ddof=0))
        if not np.isfinite(pool_std) or pool_std <= 1e-12:
            continue
        pool_mean = float(finite_pool.mean())
        selected_mean = float(finite_selected.mean())
        z_delta = (selected_mean - pool_mean) / pool_std
        rows.append(
            {
                "selector": selector,
                "month": month,
                "side_name": side_name,
                "source_archetype": source_archetype,
                "feature": feature,
                "family": family_by_feature.get(feature, "unknown"),
                "pool_rows": int(len(pool)),
                "selected_rows": int(len(selected)),
                "pool_finite_rate": float(pool_values.notna().mean()),
                "selected_finite_rate": float(selected_values.notna().mean()),
                "pool_mean": pool_mean,
                "selected_mean": selected_mean,
                "pool_std": pool_std,
                "selected_minus_pool": float(selected_mean - pool_mean),
                "selected_z_delta": float(z_delta),
                "abs_selected_z_delta": float(abs(z_delta)),
                "corr_feature_full_sl": _safe_corr(pool_values, pool["full_sl"]),
                "corr_feature_timeout": _safe_corr(pool_values, pool["timeout"]),
                "corr_feature_ev": _safe_corr(pool_values, pool["exec_ev_after_1pct_cost"]),
            }
        )
    return rows


def _family_summary(feature_drivers: pd.DataFrame) -> pd.DataFrame:
    if feature_drivers.empty:
        return pd.DataFrame()
    return (
        feature_drivers.groupby(["selector", "month", "side_name", "source_archetype", "family"], as_index=False)
        .agg(
            features=("feature", "nunique"),
            mean_abs_selected_z_delta=("abs_selected_z_delta", "mean"),
            max_abs_selected_z_delta=("abs_selected_z_delta", "max"),
            mean_corr_feature_full_sl=("corr_feature_full_sl", "mean"),
            mean_corr_feature_timeout=("corr_feature_timeout", "mean"),
            mean_corr_feature_ev=("corr_feature_ev", "mean"),
        )
        .sort_values(
            ["selector", "month", "side_name", "source_archetype", "mean_abs_selected_z_delta"],
            ascending=[True, True, True, True, False],
        )
    )


def _accepted_context(accepted_cells: pd.DataFrame, weak_cells: pd.DataFrame) -> pd.DataFrame:
    if accepted_cells.empty or weak_cells.empty:
        return pd.DataFrame()
    keys = ["month", "side_name", "source_archetype"]
    key_frame = weak_cells[keys].drop_duplicates()
    out = accepted_cells.merge(key_frame, on=keys, how="inner")
    if "family_rank_in_cell" in out.columns:
        out = out.sort_values(keys + ["family_rank_in_cell"])
    return out


def _write_report(
    path: Path,
    *,
    manifest: dict[str, Any],
    weak_cells: pd.DataFrame,
    feature_drivers: pd.DataFrame,
    family_summary: pd.DataFrame,
    accepted_context: pd.DataFrame,
) -> None:
    top_features = (
        feature_drivers.sort_values("abs_selected_z_delta", ascending=False).head(25)
        if not feature_drivers.empty
        else pd.DataFrame()
    )
    top_families = (
        family_summary.sort_values("mean_abs_selected_z_delta", ascending=False).head(25)
        if not family_summary.empty
        else pd.DataFrame()
    )
    lines = [
        "# Direct Context Weak-Cell Driver Report",
        "",
        "## Scope",
        "",
        f"- Rows inspected: `{manifest['rows']}`",
        f"- Weak cells inspected: `{manifest['weak_cell_count']}`",
        f"- Feature columns inspected: `{manifest['feature_count']}`",
        f"- Selectors: `{', '.join(manifest['selectors'])}`",
        "- Diagnostic only: no accepted-family or weak-cell flags are fed back as model inputs.",
        "",
        "## Weak Cells",
        "",
        weak_cells.head(30).to_markdown(index=False) if not weak_cells.empty else "No weak cells.",
        "",
        "## Largest Family Pressures",
        "",
        top_families.to_markdown(index=False) if not top_families.empty else "No family pressure rows.",
        "",
        "## Largest Feature Pressures",
        "",
        top_features.to_markdown(index=False) if not top_features.empty else "No feature driver rows.",
        "",
        "## Accepted-Family Evidence In These Cells",
        "",
        accepted_context.head(40).to_markdown(index=False) if not accepted_context.empty else "No accepted-family context rows.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(
    *,
    handoff_path: Path,
    feature_manifest_path: Path,
    predictions_path: Path,
    worst_cells_path: Path,
    accepted_cells_path: Path,
    output_dir: Path,
    selectors: tuple[str, ...],
    top_frac: float,
    max_cells: int,
    min_delta_full_sl: float,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_columns, families = _load_manifest(feature_manifest_path)
    family_by_feature = _feature_family_map(families)

    handoff = pd.read_parquet(handoff_path)
    predictions = pd.read_parquet(predictions_path)
    if len(handoff) != len(predictions):
        raise ValueError(f"handoff/prediction row mismatch: {len(handoff)} != {len(predictions)}")
    score_cols = [c for c in predictions.columns if c.startswith("score_") or c.startswith("pred_")]
    frame = pd.concat(
        [
            handoff.reset_index(drop=True),
            predictions[[c for c in score_cols if c not in handoff.columns]].reset_index(drop=True),
        ],
        axis=1,
    )

    worst_cells = pd.read_csv(worst_cells_path)
    weak = worst_cells[
        worst_cells["selector"].astype(str).isin(selectors)
        & (pd.to_numeric(worst_cells["delta_full_sl_rate"], errors="coerce") >= float(min_delta_full_sl))
    ].copy()
    weak = weak.sort_values(["delta_full_sl_rate", "delta_mean_ev_after_1pct"], ascending=[False, True]).head(
        int(max_cells)
    )

    driver_rows: list[dict[str, Any]] = []
    inspected_cells: list[dict[str, Any]] = []
    for _, weak_row in weak.iterrows():
        selector = str(weak_row["selector"])
        month, side_name, source_archetype = _cell_key(weak_row)
        score_col = f"score_{selector}"
        if score_col not in frame.columns:
            continue
        mask = (
            frame["month"].astype(str).eq(month)
            & frame["side_name"].astype(str).eq(side_name)
            & frame["source_archetype"].astype(str).eq(source_archetype)
        )
        pool = frame[mask].copy()
        selected = _select_topk(pool, selector, top_frac)
        inspected_cells.append(
            {
                "selector": selector,
                "month": month,
                "side_name": side_name,
                "source_archetype": source_archetype,
                "pool_rows": int(len(pool)),
                "selected_rows": int(len(selected)),
                "delta_full_sl_rate": float(weak_row["delta_full_sl_rate"]),
                "delta_timeout_rate": float(weak_row["delta_timeout_rate"]),
                "delta_mean_ev_after_1pct": float(weak_row["delta_mean_ev_after_1pct"]),
            }
        )
        driver_rows.extend(
            _feature_driver_rows(
                selector=selector,
                month=month,
                side_name=side_name,
                source_archetype=source_archetype,
                pool=pool,
                selected=selected,
                feature_columns=feature_columns,
                family_by_feature=family_by_feature,
            )
        )

    inspected = pd.DataFrame(inspected_cells)
    feature_drivers = pd.DataFrame(driver_rows)
    family_summary = _family_summary(feature_drivers)
    accepted_cells = pd.read_csv(accepted_cells_path) if accepted_cells_path.exists() else pd.DataFrame()
    accepted_context = _accepted_context(accepted_cells, inspected)

    outputs = {
        "inspected_cells": output_dir / "weak_cell_driver_inspected_cells.csv",
        "feature_drivers": output_dir / "weak_cell_feature_drivers.csv",
        "family_summary": output_dir / "weak_cell_family_pressure.csv",
        "accepted_context": output_dir / "weak_cell_accepted_family_context.csv",
        "manifest": output_dir / "manifest.json",
        "report": output_dir / "weak_cell_driver_report.md",
    }
    inspected.to_csv(outputs["inspected_cells"], index=False)
    feature_drivers.to_csv(outputs["feature_drivers"], index=False)
    family_summary.to_csv(outputs["family_summary"], index=False)
    accepted_context.to_csv(outputs["accepted_context"], index=False)

    manifest = {
        "scope": "direct_context_weak_cell_driver_report",
        "handoff_path": str(handoff_path),
        "feature_manifest_path": str(feature_manifest_path),
        "predictions_path": str(predictions_path),
        "worst_cells_path": str(worst_cells_path),
        "accepted_cells_path": str(accepted_cells_path),
        "output_dir": str(output_dir),
        "rows": int(len(frame)),
        "feature_count": int(len(feature_columns)),
        "weak_cell_count": int(len(inspected)),
        "selectors": list(selectors),
        "top_frac": float(top_frac),
        "min_delta_full_sl": float(min_delta_full_sl),
        "no_leakage_contract": (
            "uses saved month-forward predictions and pre-entry context features for diagnostics only; "
            "accepted-family context is report metadata, not model input"
        ),
        "outputs": {k: str(v) for k, v in outputs.items()},
    }
    outputs["manifest"].write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")
    _write_report(
        outputs["report"],
        manifest=manifest,
        weak_cells=inspected,
        feature_drivers=feature_drivers,
        family_summary=family_summary,
        accepted_context=accepted_context,
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--handoff-path", type=Path, default=DEFAULT_HANDOFF)
    parser.add_argument("--feature-manifest-path", type=Path, default=DEFAULT_FEATURE_MANIFEST)
    parser.add_argument("--predictions-path", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--worst-cells-path", type=Path, default=DEFAULT_WORST_CELLS)
    parser.add_argument("--accepted-cells-path", type=Path, default=DEFAULT_ACCEPTED_CELLS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--selectors", nargs="+", default=list(DEFAULT_SELECTORS))
    parser.add_argument("--top-frac", type=float, default=0.10)
    parser.add_argument("--max-cells", type=int, default=18)
    parser.add_argument("--min-delta-full-sl", type=float, default=0.02)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = run(
        handoff_path=args.handoff_path,
        feature_manifest_path=args.feature_manifest_path,
        predictions_path=args.predictions_path,
        worst_cells_path=args.worst_cells_path,
        accepted_cells_path=args.accepted_cells_path,
        output_dir=args.output_dir,
        selectors=tuple(str(s) for s in args.selectors),
        top_frac=float(args.top_frac),
        max_cells=int(args.max_cells),
        min_delta_full_sl=float(args.min_delta_full_sl),
    )
    print(json.dumps(_json_safe(manifest), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
