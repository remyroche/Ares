#!/usr/bin/env python3
"""Attribute bad promoted-vs-baseline flips to pre-entry context features.

For side x archetype cells where the promoted cross-asset score degraded
relative to the baseline score, this audit compares rows selected only by the
promoted top-k against rows selected only by the baseline top-k.  It then ranks
pre-entry context features whose distributions differ most between those two
sets.

This is an offline diagnostic.  It explains OOF failures and does not create
deployable gates.
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

from scripts.audit_promoted_cross_asset_cell_effects import (  # noqa: E402
    DEFAULT_BASELINE_SMOKE_DIR,
    DEFAULT_PROMOTED_HANDOFF_DIR,
    DEFAULT_PROMOTED_SMOKE_DIR,
    KEY_COLUMNS,
    PREDICTIONS_NAME,
    _best_score_column,
    _json_safe,
)
from scripts.audit_promoted_cross_asset_month_flip_attribution import DEFAULT_OUT_DIR as DEFAULT_FLIP_AUDIT_DIR  # noqa: E402
from scripts.materialize_promoted_cross_asset_meta_handoff import HANDOFF_NAME  # noqa: E402


DEFAULT_FLIPS_PATH = DEFAULT_FLIP_AUDIT_DIR / "promoted_cross_asset_month_flips.csv"
DEFAULT_HANDOFF_PATH = DEFAULT_PROMOTED_HANDOFF_DIR / HANDOFF_NAME
DEFAULT_OUT_DIR = DEFAULT_PROMOTED_HANDOFF_DIR / "cross_asset_flip_feature_attribution_v1"
BAD_FLIP_TYPES = {"positive_to_negative", "material_degradation"}
GROUP_COLUMNS = ("month", "side_name", "source_semantic_family")
OUTCOME_COLUMNS = {
    "exec_margin",
    "ev_after_1pct",
    "ret_net",
    "u_policy_net",
    "first_touch_gross",
    "first_touch_bad_mae_1r",
    "full_path_bad_mae_1r",
    "timeout",
    "mfe_before_mae_1r",
    "mae_before_mfe_1r",
    "clean_exec",
    "dirty_positive",
    "underwater_bars_before_mfe_1r",
    "long_path_clean_exec_label",
    "long_path_dirty_positive_label",
    "long_path_post_mfe_drawdown_norm",
    "long_path_time_to_profit_bars",
    "long_path_slow_profit",
    "long_path_post_mfe_bad_drawdown",
    "long_bad_path_label",
}
FEATURE_ALLOW_PREFIXES = (
    "source_",
    "base_score_",
    "aegmm_",
    "side_aegmm_",
    "reconstruction_",
    "dae_reconstruction_",
    "cluster_",
    "latent_",
    "calendar_",
    "structural_",
    "regime_",
    "gmm_",
    "AE_",
    "asym_",
    "btc_",
    "eth_",
    "cs_",
    "mkt_",
    "pct_assets_",
    "spectral_",
    "tail_",
    "trend_",
    "xasset_",
    "cross_lgbm_",
)


def _num(values: Any, *, index: pd.Index | None = None) -> pd.Series:
    if isinstance(values, pd.Series):
        return pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)
    return pd.to_numeric(pd.Series(values, index=index), errors="coerce").replace([np.inf, -np.inf], np.nan)


def _mean(values: Any) -> float:
    s = _num(values).dropna()
    return float(s.mean()) if len(s) else float("nan")


def _rate(values: Any) -> float:
    s = _num(values).dropna()
    return float(s.clip(0.0, 1.0).mean()) if len(s) else float("nan")


def _row_key_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["_row_key"] = out[list(KEY_COLUMNS)].astype(str).agg("|".join, axis=1)
    return out


def _top_keys(frame: pd.DataFrame, score_col: str, keep_frac: float) -> set[str]:
    scored = _row_key_frame(frame)
    scored["_score"] = _num(scored.get(score_col), index=scored.index)
    scored = scored[scored["_score"].notna()]
    if scored.empty:
        return set()
    n = max(1, int(math.ceil(len(scored) * float(keep_frac))))
    return set(scored.sort_values("_score", ascending=False, kind="mergesort").head(n)["_row_key"].astype(str))


def _feature_columns(handoff: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric: list[str] = []
    categorical: list[str] = []
    excluded = set(KEY_COLUMNS) | {"month", "score", "selected_top10"} | OUTCOME_COLUMNS
    for col in handoff.columns:
        if col in excluded or col.startswith("meta_action_mean_holdout_"):
            continue
        if not col.startswith(FEATURE_ALLOW_PREFIXES):
            continue
        if pd.api.types.is_numeric_dtype(handoff[col]) or handoff[col].dtype == bool:
            numeric.append(col)
        else:
            categorical.append(col)
    return sorted(numeric), sorted(categorical)


def _outcome_summary(rows: pd.DataFrame, prefix: str) -> dict[str, Any]:
    return {
        f"{prefix}_rows": int(len(rows)),
        f"{prefix}_exec_margin": _mean(rows.get("exec_margin")),
        f"{prefix}_ev_after_1pct": _mean(rows.get("ev_after_1pct")),
        f"{prefix}_clean_exec": _rate(rows.get("clean_exec")),
        f"{prefix}_bad_mae": _rate(rows.get("full_path_bad_mae_1r")),
        f"{prefix}_timeout": _rate(rows.get("timeout")),
        f"{prefix}_mfe_before_mae": _rate(rows.get("mfe_before_mae_1r")),
    }


def _numeric_attribution(
    cell: pd.DataFrame,
    promoted_only: pd.DataFrame,
    baseline_only: pd.DataFrame,
    numeric_cols: list[str],
    *,
    max_features: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if promoted_only.empty or baseline_only.empty:
        return pd.DataFrame()
    for col in numeric_cols:
        all_values = _num(cell.get(col), index=cell.index)
        if all_values.notna().sum() < 5 or all_values.nunique(dropna=True) < 2:
            continue
        mean = float(all_values.mean())
        std = float(all_values.std(ddof=0))
        if not math.isfinite(std) or std <= 1e-12:
            continue
        p = _num(promoted_only.get(col), index=promoted_only.index)
        b = _num(baseline_only.get(col), index=baseline_only.index)
        p_mean = float(p.mean()) if p.notna().any() else float("nan")
        b_mean = float(b.mean()) if b.notna().any() else float("nan")
        z_diff = ((p_mean - mean) / std) - ((b_mean - mean) / std) if math.isfinite(p_mean) and math.isfinite(b_mean) else float("nan")
        rows.append(
            {
                "feature": col,
                "feature_type": "numeric",
                "promoted_only_mean": p_mean,
                "baseline_only_mean": b_mean,
                "cell_mean": mean,
                "cell_std": std,
                "standardized_diff_promoted_minus_baseline": z_diff,
                "abs_standardized_diff": abs(z_diff) if math.isfinite(z_diff) else float("nan"),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values("abs_standardized_diff", ascending=False).head(max_features)


def _categorical_attribution(
    promoted_only: pd.DataFrame,
    baseline_only: pd.DataFrame,
    categorical_cols: list[str],
    *,
    max_features: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if promoted_only.empty or baseline_only.empty:
        return pd.DataFrame()
    for col in categorical_cols:
        p = promoted_only[col].astype(str).fillna("missing") if col in promoted_only.columns else pd.Series(dtype=str)
        b = baseline_only[col].astype(str).fillna("missing") if col in baseline_only.columns else pd.Series(dtype=str)
        cats = sorted(set(p.unique()).union(set(b.unique())))
        for cat in cats:
            p_rate = float(p.eq(cat).mean()) if len(p) else float("nan")
            b_rate = float(b.eq(cat).mean()) if len(b) else float("nan")
            diff = p_rate - b_rate
            if abs(diff) < 0.10:
                continue
            rows.append(
                {
                    "feature": col,
                    "feature_type": "categorical",
                    "category": cat,
                    "promoted_only_rate": p_rate,
                    "baseline_only_rate": b_rate,
                    "rate_diff_promoted_minus_baseline": diff,
                    "abs_rate_diff": abs(diff),
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values("abs_rate_diff", ascending=False).head(max_features)


def _prepare_joined(
    baseline_predictions: pd.DataFrame,
    promoted_predictions: pd.DataFrame,
    handoff: pd.DataFrame,
    *,
    baseline_score_col: str,
    promoted_score_col: str,
) -> pd.DataFrame:
    promoted_cols = [
        *KEY_COLUMNS,
        "month",
        "source_semantic_family",
        promoted_score_col,
        *[col for col in OUTCOME_COLUMNS if col in promoted_predictions.columns],
    ]
    baseline_cols = [*KEY_COLUMNS, baseline_score_col]
    promoted = promoted_predictions[list(dict.fromkeys(promoted_cols))].copy()
    baseline = baseline_predictions[list(dict.fromkeys(baseline_cols))].copy().rename(
        columns={baseline_score_col: "score_attrib_baseline"}
    )
    promoted = promoted.rename(columns={promoted_score_col: "score_attrib_promoted"})
    joined = promoted.merge(baseline, on=list(KEY_COLUMNS), how="inner", validate="one_to_one")
    feature_cols = [col for col in handoff.columns if col not in joined.columns or col in KEY_COLUMNS]
    joined = joined.merge(handoff[feature_cols], on=list(KEY_COLUMNS), how="left", validate="many_to_one")
    return _row_key_frame(joined)


def run_audit(
    *,
    baseline_smoke_dir: Path,
    promoted_smoke_dir: Path,
    handoff_path: Path,
    flips_path: Path,
    out_dir: Path,
    max_cells: int = 12,
    max_features_per_cell: int = 15,
) -> dict[str, Any]:
    baseline_selector, baseline_score_col = _best_score_column(baseline_smoke_dir)
    promoted_selector, promoted_score_col = _best_score_column(promoted_smoke_dir)
    baseline_predictions = pd.read_parquet(baseline_smoke_dir / PREDICTIONS_NAME)
    promoted_predictions = pd.read_parquet(promoted_smoke_dir / PREDICTIONS_NAME)
    handoff = pd.read_parquet(handoff_path)
    flips = pd.read_csv(flips_path)
    bad_flips = flips[flips["flip_type"].astype(str).isin(BAD_FLIP_TYPES)].copy()
    bad_flips = bad_flips.sort_values(["keep_frac", "effect_value_delta"], ascending=[True, True]).head(int(max_cells))
    joined = _prepare_joined(
        baseline_predictions,
        promoted_predictions,
        handoff,
        baseline_score_col=baseline_score_col,
        promoted_score_col=promoted_score_col,
    )
    numeric_cols, categorical_cols = _feature_columns(joined)

    cell_rows: list[dict[str, Any]] = []
    numeric_rows: list[pd.DataFrame] = []
    categorical_rows: list[pd.DataFrame] = []
    for _, flip in bad_flips.iterrows():
        month = str(flip["current_month"])
        side = str(flip["side_name"])
        family = str(flip["source_semantic_family"])
        keep_frac = float(flip["keep_frac"])
        mask = (
            joined["month"].astype(str).eq(month)
            & joined["side_name"].astype(str).eq(side)
            & joined["source_semantic_family"].astype(str).eq(family)
        )
        cell = joined.loc[mask].copy()
        if cell.empty:
            continue
        promoted_keys = _top_keys(cell, "score_attrib_promoted", keep_frac)
        baseline_keys = _top_keys(cell, "score_attrib_baseline", keep_frac)
        promoted_only_keys = promoted_keys.difference(baseline_keys)
        baseline_only_keys = baseline_keys.difference(promoted_keys)
        overlap_keys = promoted_keys.intersection(baseline_keys)
        promoted_only = cell[cell["_row_key"].isin(promoted_only_keys)].copy()
        baseline_only = cell[cell["_row_key"].isin(baseline_only_keys)].copy()
        overlap = cell[cell["_row_key"].isin(overlap_keys)].copy()
        record = {
            "month": month,
            "side_name": side,
            "source_semantic_family": family,
            "keep_frac": keep_frac,
            "flip_type": flip.get("flip_type"),
            "history_effect_value": flip.get("history_effect_value"),
            "current_effect_value": flip.get("current_effect_value"),
            "effect_value_delta": flip.get("effect_value_delta"),
            "cell_rows": int(len(cell)),
            "promoted_top_rows": int(len(promoted_keys)),
            "baseline_top_rows": int(len(baseline_keys)),
            "overlap_rows": int(len(overlap_keys)),
            "promoted_only_rows": int(len(promoted_only)),
            "baseline_only_rows": int(len(baseline_only)),
            "top_overlap_rate": float(len(overlap_keys) / max(len(promoted_keys), 1)),
            **_outcome_summary(promoted_only, "promoted_only"),
            **_outcome_summary(baseline_only, "baseline_only"),
            **_outcome_summary(overlap, "overlap"),
        }
        cell_rows.append(record)
        num = _numeric_attribution(cell, promoted_only, baseline_only, numeric_cols, max_features=max_features_per_cell)
        if not num.empty:
            for col, value in record.items():
                if col in {"month", "side_name", "source_semantic_family", "keep_frac", "flip_type"}:
                    num[col] = value
            numeric_rows.append(num)
        cat = _categorical_attribution(promoted_only, baseline_only, categorical_cols, max_features=max_features_per_cell)
        if not cat.empty:
            for col, value in record.items():
                if col in {"month", "side_name", "source_semantic_family", "keep_frac", "flip_type"}:
                    cat[col] = value
            categorical_rows.append(cat)

    cells_df = pd.DataFrame(cell_rows)
    numeric_df = pd.concat(numeric_rows, ignore_index=True) if numeric_rows else pd.DataFrame()
    categorical_df = pd.concat(categorical_rows, ignore_index=True) if categorical_rows else pd.DataFrame()
    out_dir.mkdir(parents=True, exist_ok=True)
    cells_path = out_dir / "cross_asset_flip_feature_attribution_cells.csv"
    numeric_path = out_dir / "cross_asset_flip_feature_attribution_numeric.csv"
    categorical_path = out_dir / "cross_asset_flip_feature_attribution_categorical.csv"
    cells_df.to_csv(cells_path, index=False)
    numeric_df.to_csv(numeric_path, index=False)
    categorical_df.to_csv(categorical_path, index=False)

    top_numeric = (
        numeric_df.sort_values("abs_standardized_diff", ascending=False).head(20).to_dict("records")
        if not numeric_df.empty
        else []
    )
    top_categorical = (
        categorical_df.sort_values("abs_rate_diff", ascending=False).head(20).to_dict("records")
        if not categorical_df.empty
        else []
    )
    manifest = {
        "generated_by": "audit_cross_asset_flip_feature_attribution",
        "baseline_smoke_dir": str(baseline_smoke_dir),
        "promoted_smoke_dir": str(promoted_smoke_dir),
        "handoff_path": str(handoff_path),
        "flips_path": str(flips_path),
        "baseline_selector": baseline_selector,
        "baseline_score_col": baseline_score_col,
        "promoted_selector": promoted_selector,
        "promoted_score_col": promoted_score_col,
        "attributed_cells": int(len(cells_df)),
        "numeric_feature_count": int(len(numeric_cols)),
        "categorical_feature_count": int(len(categorical_cols)),
        "top_numeric_attributions": _json_safe(top_numeric),
        "top_categorical_attributions": _json_safe(top_categorical),
        "leakage_contract": {
            "prediction_source": "OOF train_meta smoke predictions",
            "feature_source": "train_meta handoff pre-entry/context columns",
            "labels_used_for": "offline failure attribution only",
            "deployment_use": "diagnostic leads for meta features, not hard gates",
        },
        "outputs": {
            "cells": str(cells_path),
            "numeric": str(numeric_path),
            "categorical": str(categorical_path),
            "json": str(out_dir / "cross_asset_flip_feature_attribution.json"),
            "markdown": str(out_dir / "cross_asset_flip_feature_attribution.md"),
        },
    }
    markdown = _write_markdown(out_dir, manifest, cells_df, numeric_df, categorical_df)
    manifest["outputs"]["markdown"] = str(markdown)
    (out_dir / "cross_asset_flip_feature_attribution.json").write_text(
        json.dumps(_json_safe(manifest), indent=2, sort_keys=True) + "\n"
    )
    return manifest


def _write_markdown(
    out_dir: Path,
    manifest: dict[str, Any],
    cells: pd.DataFrame,
    numeric: pd.DataFrame,
    categorical: pd.DataFrame,
) -> Path:
    lines = [
        "# Cross-Asset Flip Feature Attribution",
        "",
        "## Verdict",
        "",
        f"- attributed cells: `{manifest.get('attributed_cells')}`",
        f"- baseline selector: `{manifest.get('baseline_selector')}`",
        f"- promoted selector: `{manifest.get('promoted_selector')}`",
        "",
        "## Cell Selection Deltas",
        "",
    ]
    cell_cols = [
        "month",
        "side_name",
        "source_semantic_family",
        "keep_frac",
        "flip_type",
        "cell_rows",
        "promoted_only_rows",
        "baseline_only_rows",
        "top_overlap_rate",
        "promoted_only_ev_after_1pct",
        "baseline_only_ev_after_1pct",
        "promoted_only_bad_mae",
        "baseline_only_bad_mae",
    ]
    lines.append(cells[[col for col in cell_cols if col in cells.columns]].to_markdown(index=False) if not cells.empty else "_No cells._")
    lines.extend(["", "## Top Numeric Differences", ""])
    num_cols = [
        "month",
        "side_name",
        "source_semantic_family",
        "keep_frac",
        "feature",
        "promoted_only_mean",
        "baseline_only_mean",
        "standardized_diff_promoted_minus_baseline",
    ]
    lines.append(numeric[[col for col in num_cols if col in numeric.columns]].head(30).to_markdown(index=False) if not numeric.empty else "_No numeric attribution._")
    lines.extend(["", "## Top Categorical Differences", ""])
    cat_cols = [
        "month",
        "side_name",
        "source_semantic_family",
        "keep_frac",
        "feature",
        "category",
        "promoted_only_rate",
        "baseline_only_rate",
        "rate_diff_promoted_minus_baseline",
    ]
    lines.append(categorical[[col for col in cat_cols if col in categorical.columns]].head(30).to_markdown(index=False) if not categorical.empty else "_No categorical attribution._")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Rows labeled promoted_only are selected by the promoted score but not by the baseline score inside the same month/side/archetype/top-k cell.",
            "Large feature differences are candidates for meta-layer context or interaction learning. They are not validated gates.",
        ]
    )
    path = out_dir / "cross_asset_flip_feature_attribution.md"
    path.write_text("\n".join(lines) + "\n")
    return path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-smoke-dir", type=Path, default=DEFAULT_BASELINE_SMOKE_DIR)
    parser.add_argument("--promoted-smoke-dir", type=Path, default=DEFAULT_PROMOTED_SMOKE_DIR)
    parser.add_argument("--handoff-path", type=Path, default=DEFAULT_HANDOFF_PATH)
    parser.add_argument("--flips-path", type=Path, default=DEFAULT_FLIPS_PATH)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--max-cells", type=int, default=12)
    parser.add_argument("--max-features-per-cell", type=int, default=15)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = run_audit(
        baseline_smoke_dir=args.baseline_smoke_dir,
        promoted_smoke_dir=args.promoted_smoke_dir,
        handoff_path=args.handoff_path,
        flips_path=args.flips_path,
        out_dir=args.out_dir,
        max_cells=args.max_cells,
        max_features_per_cell=args.max_features_per_cell,
    )
    print(json.dumps(_json_safe({"event": "cross_asset_flip_feature_attribution_done", **manifest}), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
