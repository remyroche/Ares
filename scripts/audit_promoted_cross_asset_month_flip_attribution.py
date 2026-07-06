#!/usr/bin/env python3
"""Month-flip attribution for promoted cross-asset meta scores.

The promoted cross-asset score can look useful in one OOF month and then lose
to the baseline score in the next.  This audit keeps the comparison at the
side x archetype cell level and reports where the promoted-vs-baseline effect
changes sign or degrades materially across months.

It is diagnostic only: it uses realized labels to explain the OOF artifact, not
to choose a deployable rule.
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
    KEEP_FRACS,
    PREDICTIONS_NAME,
    _best_score_column,
    _json_safe,
    _mean,
    _num,
    _rate,
)


DEFAULT_OUT_DIR = DEFAULT_PROMOTED_HANDOFF_DIR / "promoted_cross_asset_month_flip_attribution_v1"


def _support_metrics(cell: pd.DataFrame) -> dict[str, Any]:
    ts = pd.to_datetime(cell.get("__ts__"), utc=True, errors="coerce")
    weeks = ts.dt.strftime("%G-W%V").fillna("unknown")
    symbols = cell.get("__symbol__", pd.Series("unknown", index=cell.index)).astype(str)
    return {
        "rows": int(len(cell)),
        "symbol_count": int(symbols.nunique()),
        "week_count": int(weeks.nunique()),
        "clean_rows": int(_num(cell.get("clean_exec"), index=cell.index, default=0.0).fillna(0.0).gt(0.5).sum()),
        "positive_exec_rows": int(_num(cell.get("exec_margin"), index=cell.index, default=np.nan).gt(0.0).sum()),
        "max_single_asset_share": float(symbols.value_counts(normalize=True).iloc[0]) if len(symbols) else float("nan"),
        "max_single_week_share": float(weeks.value_counts(normalize=True).iloc[0]) if len(weeks) else float("nan"),
    }


def _top_metrics(cell: pd.DataFrame, score_col: str, keep_frac: float) -> dict[str, Any]:
    if score_col not in cell.columns:
        raise ValueError(f"Missing score column {score_col!r}")
    scored = cell.copy()
    scored["_score__tmp"] = _num(scored.get(score_col), index=scored.index)
    scored = scored[scored["_score__tmp"].notna()]
    if scored.empty:
        return {
            "selected_rows": 0,
            "ev_after_1pct": float("nan"),
            "exec_margin": float("nan"),
            "clean_exec_precision": float("nan"),
            "full_path_bad_mae": float("nan"),
            "timeout": float("nan"),
            "dirty_positive": float("nan"),
            "mfe_before_mae": float("nan"),
            "mae_before_mfe": float("nan"),
            "underwater_bars": float("nan"),
            "cell_oracle_overlap": float("nan"),
        }
    n = max(1, int(math.ceil(len(scored) * float(keep_frac))))
    top = scored.sort_values("_score__tmp", ascending=False, kind="mergesort").head(n)
    oracle = scored.sort_values("exec_margin", ascending=False, kind="mergesort").head(n) if "exec_margin" in scored.columns else scored.head(0)
    top_keys = set(map(tuple, top[list(KEY_COLUMNS)].astype(str).to_numpy())) if all(col in top.columns for col in KEY_COLUMNS) else set()
    oracle_keys = (
        set(map(tuple, oracle[list(KEY_COLUMNS)].astype(str).to_numpy()))
        if all(col in oracle.columns for col in KEY_COLUMNS)
        else set()
    )
    overlap = float(len(top_keys & oracle_keys) / max(1, len(oracle_keys))) if oracle_keys else float("nan")
    return {
        "selected_rows": int(len(top)),
        "ev_after_1pct": _mean(top.get("ev_after_1pct")),
        "exec_margin": _mean(top.get("exec_margin")),
        "clean_exec_precision": _rate(top.get("clean_exec")),
        "full_path_bad_mae": _rate(top.get("full_path_bad_mae_1r")),
        "timeout": _rate(top.get("timeout")),
        "dirty_positive": _rate(top.get("dirty_positive")),
        "mfe_before_mae": _rate(top.get("mfe_before_mae_1r")),
        "mae_before_mfe": _rate(top.get("mae_before_mfe_1r")),
        "underwater_bars": _mean(top.get("underwater_bars_before_mfe_1r")),
        "cell_oracle_overlap": overlap,
    }


def _value_score(record: dict[str, Any]) -> float:
    ev = float(record.get("delta_ev_after_1pct", 0.0) or 0.0)
    exec_margin = float(record.get("delta_exec_margin", 0.0) or 0.0)
    return float(
        np.clip(ev / 0.002, -3.0, 3.0)
        + 0.50 * np.clip(exec_margin / 0.002, -2.0, 2.0)
        + 0.75 * float(record.get("delta_clean_exec_precision", 0.0) or 0.0)
        - 0.75 * float(record.get("delta_full_path_bad_mae", 0.0) or 0.0)
        - 0.50 * float(record.get("delta_timeout", 0.0) or 0.0)
        + 0.35 * float(record.get("delta_mfe_before_mae", 0.0) or 0.0)
        - 0.35 * float(record.get("delta_mae_before_mfe", 0.0) or 0.0)
        + 0.25 * float(record.get("delta_cell_oracle_overlap", 0.0) or 0.0)
    )


def _month_cell_rows(
    baseline: pd.DataFrame,
    promoted: pd.DataFrame,
    *,
    baseline_score_col: str,
    promoted_score_col: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_key = ["month", *GROUP_COLUMNS]
    for key, base_cell in baseline.groupby(group_key, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        mask = pd.Series(True, index=promoted.index)
        for col, value in zip(group_key, key, strict=False):
            mask &= promoted[col].astype(str).eq(str(value))
        promoted_cell = promoted.loc[mask].copy()
        if promoted_cell.empty:
            continue
        support = _support_metrics(base_cell)
        for keep_frac in KEEP_FRACS:
            base_metrics = _top_metrics(base_cell, baseline_score_col, keep_frac)
            promoted_metrics = _top_metrics(promoted_cell, promoted_score_col, keep_frac)
            record: dict[str, Any] = {
                "keep_frac": float(keep_frac),
                **{col: value for col, value in zip(group_key, key, strict=False)},
                **support,
            }
            for metric, value in base_metrics.items():
                record[f"baseline_{metric}"] = value
            for metric, value in promoted_metrics.items():
                record[f"promoted_{metric}"] = value
            for metric in (
                "ev_after_1pct",
                "exec_margin",
                "clean_exec_precision",
                "full_path_bad_mae",
                "timeout",
                "dirty_positive",
                "mfe_before_mae",
                "mae_before_mfe",
                "underwater_bars",
                "cell_oracle_overlap",
            ):
                record[f"delta_{metric}"] = promoted_metrics[metric] - base_metrics[metric]
            record["effect_value_score"] = _value_score(record)
            rows.append(record)
    return pd.DataFrame(rows)


def _classify_month_cells(
    cells: pd.DataFrame,
    *,
    min_valid_rows: int,
    min_clean_rows: int,
    min_positive_rows: int,
    max_asset_share: float,
    max_week_share: float,
) -> pd.DataFrame:
    if cells.empty:
        return cells
    out = cells.copy()
    out["support_pass"] = (
        out["rows"].ge(int(min_valid_rows))
        & out["clean_rows"].ge(int(min_clean_rows))
        & out["positive_exec_rows"].ge(int(min_positive_rows))
        & out["max_single_asset_share"].le(float(max_asset_share))
        & out["max_single_week_share"].le(float(max_week_share))
    )
    out["promoted_beneficial"] = (
        out["support_pass"]
        & out["effect_value_score"].gt(0.35)
        & out["delta_ev_after_1pct"].ge(-0.0010)
        & out["delta_full_path_bad_mae"].le(0.05)
        & out["delta_timeout"].le(0.02)
    )
    out["promoted_damaged"] = (
        out["support_pass"]
        & (
            out["effect_value_score"].lt(-0.35)
            | out["delta_ev_after_1pct"].lt(-0.0020)
            | out["delta_full_path_bad_mae"].gt(0.08)
            | out["delta_clean_exec_precision"].lt(-0.08)
        )
    )
    return out.sort_values(["month", "keep_frac", "support_pass", "effect_value_score"], ascending=[True, True, False, False])


def _flip_rows(cells: pd.DataFrame) -> pd.DataFrame:
    if cells.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    key_cols = [*GROUP_COLUMNS, "keep_frac"]
    months = sorted(str(m) for m in cells["month"].astype(str).dropna().unique())
    if len(months) < 2:
        return pd.DataFrame()
    for current_month in months[1:]:
        history_months = [m for m in months if m < current_month]
        hist = cells[cells["month"].astype(str).isin(history_months)]
        current = cells[cells["month"].astype(str).eq(current_month)]
        if hist.empty or current.empty:
            continue
        hist_agg = (
            hist.groupby(key_cols, dropna=False)
            .agg(
                history_effect_value=("effect_value_score", "mean"),
                history_delta_ev=("delta_ev_after_1pct", "mean"),
                history_delta_bad_mae=("delta_full_path_bad_mae", "mean"),
                history_delta_timeout=("delta_timeout", "mean"),
                history_supported=("support_pass", "sum"),
                history_beneficial=("promoted_beneficial", "sum"),
                history_damaged=("promoted_damaged", "sum"),
                history_month_count=("month", "nunique"),
            )
            .reset_index()
        )
        merged = current.merge(hist_agg, on=key_cols, how="inner")
        for _, row in merged.iterrows():
            supported_history = int(row.get("history_supported", 0)) > 0
            supported_current = bool(row.get("support_pass", False))
            if not (supported_history and supported_current):
                continue
            hist_value = float(row.get("history_effect_value", np.nan))
            cur_value = float(row.get("effect_value_score", np.nan))
            flip_type = "stable_or_small_change"
            if hist_value > 0.35 and cur_value < -0.35:
                flip_type = "positive_to_negative"
            elif hist_value < -0.35 and cur_value > 0.35:
                flip_type = "negative_to_positive"
            elif hist_value > 0.35 and cur_value <= 0.05:
                flip_type = "positive_to_flat"
            elif cur_value < hist_value - 0.75:
                flip_type = "material_degradation"
            elif cur_value > hist_value + 0.75:
                flip_type = "material_improvement"
            rows.append(
                {
                    "current_month": current_month,
                    "history_months": ",".join(history_months),
                    "flip_type": flip_type,
                    **{col: row[col] for col in key_cols},
                    "history_effect_value": hist_value,
                    "current_effect_value": cur_value,
                    "effect_value_delta": cur_value - hist_value,
                    "history_delta_ev": row.get("history_delta_ev"),
                    "current_delta_ev": row.get("delta_ev_after_1pct"),
                    "history_delta_bad_mae": row.get("history_delta_bad_mae"),
                    "current_delta_bad_mae": row.get("delta_full_path_bad_mae"),
                    "history_delta_timeout": row.get("history_delta_timeout"),
                    "current_delta_timeout": row.get("delta_timeout"),
                    "history_beneficial_months": int(row.get("history_beneficial", 0)),
                    "history_damaged_months": int(row.get("history_damaged", 0)),
                    "current_promoted_beneficial": bool(row.get("promoted_beneficial", False)),
                    "current_promoted_damaged": bool(row.get("promoted_damaged", False)),
                    "current_rows": int(row.get("rows", 0)),
                    "current_clean_rows": int(row.get("clean_rows", 0)),
                    "current_positive_exec_rows": int(row.get("positive_exec_rows", 0)),
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    order = ["positive_to_negative", "material_degradation", "positive_to_flat", "negative_to_positive", "material_improvement", "stable_or_small_change"]
    out["flip_type_order"] = out["flip_type"].map({v: i for i, v in enumerate(order)}).fillna(99)
    return out.sort_values(["current_month", "keep_frac", "flip_type_order", "effect_value_delta"], ascending=[True, True, True, True]).drop(columns=["flip_type_order"])


def _summary(cells: pd.DataFrame, flips: pd.DataFrame) -> dict[str, Any]:
    if cells.empty:
        return {"status": "no_cells"}
    keep10 = cells[cells["keep_frac"].eq(0.10)]
    flip_keep10 = flips[flips["keep_frac"].eq(0.10)] if not flips.empty else pd.DataFrame()
    bad_flips = flip_keep10[flip_keep10["flip_type"].isin(["positive_to_negative", "material_degradation"])] if not flip_keep10.empty else pd.DataFrame()
    status = "month_instability_detected" if not bad_flips.empty else "no_major_month_flip_detected"
    return {
        "status": status,
        "months": sorted(str(m) for m in cells["month"].astype(str).unique()),
        "cells": int(len(cells)),
        "supported_month_cells": int(cells["support_pass"].sum()),
        "beneficial_month_cells": int(cells["promoted_beneficial"].sum()),
        "damaged_month_cells": int(cells["promoted_damaged"].sum()),
        "keep10_supported_month_cells": int(keep10["support_pass"].sum()) if not keep10.empty else 0,
        "keep10_bad_flip_cells": int(len(bad_flips)),
        "worst_keep10_flips": _json_safe(
            bad_flips.head(12)[
                [
                    "current_month",
                    "flip_type",
                    "side_name",
                    "source_semantic_family",
                    "history_effect_value",
                    "current_effect_value",
                    "effect_value_delta",
                    "history_delta_ev",
                    "current_delta_ev",
                    "history_delta_bad_mae",
                    "current_delta_bad_mae",
                    "current_rows",
                ]
            ].to_dict("records")
        )
        if not bad_flips.empty
        else [],
    }


def _write_markdown(out_dir: Path, manifest: dict[str, Any], flips: pd.DataFrame) -> Path:
    summary = manifest["summary"]
    worst = pd.DataFrame(summary.get("worst_keep10_flips") or [])
    lines = [
        "# Promoted Cross-Asset Month Flip Attribution",
        "",
        "## Verdict",
        "",
        f"- status: `{summary.get('status')}`",
        f"- months: `{', '.join(summary.get('months') or [])}`",
        f"- supported month-cells: `{summary.get('supported_month_cells')}`",
        f"- beneficial month-cells: `{summary.get('beneficial_month_cells')}`",
        f"- damaged month-cells: `{summary.get('damaged_month_cells')}`",
        f"- keep10 bad flip cells: `{summary.get('keep10_bad_flip_cells')}`",
        "",
        "## Worst Keep10 Flips",
        "",
    ]
    lines.append(worst.to_markdown(index=False) if not worst.empty else "_No major keep10 negative flips._")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This is attribution, not a deployable gate. It uses realized outcomes to identify where the promoted OOF score helped in prior months but degraded in the current month.",
            "Cells with positive-to-negative or material-degradation flips are candidates for more robust meta training, longer history, or explicit stability features; they should not become hand-coded June gates.",
        ]
    )
    path = out_dir / "promoted_cross_asset_month_flip_attribution.md"
    path.write_text("\n".join(lines) + "\n")
    return path


def run_audit(
    *,
    baseline_smoke_dir: Path,
    promoted_smoke_dir: Path,
    out_dir: Path,
    min_valid_rows: int = 20,
    min_clean_rows: int = 4,
    min_positive_rows: int = 4,
    max_asset_share: float = 0.85,
    max_week_share: float = 0.85,
) -> dict[str, Any]:
    baseline_selector, baseline_score_col = _best_score_column(baseline_smoke_dir)
    promoted_selector, promoted_score_col = _best_score_column(promoted_smoke_dir)
    baseline = pd.read_parquet(baseline_smoke_dir / PREDICTIONS_NAME)
    promoted = pd.read_parquet(promoted_smoke_dir / PREDICTIONS_NAME)
    required = ["month", *GROUP_COLUMNS, *KEY_COLUMNS]
    for frame_name, frame, score_col in (
        ("baseline", baseline, baseline_score_col),
        ("promoted", promoted, promoted_score_col),
    ):
        missing = [col for col in [*required, score_col] if col not in frame.columns]
        if missing:
            raise ValueError(f"{frame_name} predictions missing required columns: {missing}")
    cells = _month_cell_rows(
        baseline,
        promoted,
        baseline_score_col=baseline_score_col,
        promoted_score_col=promoted_score_col,
    )
    cells = _classify_month_cells(
        cells,
        min_valid_rows=min_valid_rows,
        min_clean_rows=min_clean_rows,
        min_positive_rows=min_positive_rows,
        max_asset_share=max_asset_share,
        max_week_share=max_week_share,
    )
    flips = _flip_rows(cells)
    out_dir.mkdir(parents=True, exist_ok=True)
    cells_path = out_dir / "promoted_cross_asset_month_cell_effects.csv"
    flips_path = out_dir / "promoted_cross_asset_month_flips.csv"
    cells.to_csv(cells_path, index=False)
    flips.to_csv(flips_path, index=False)
    manifest = {
        "generated_by": "audit_promoted_cross_asset_month_flip_attribution",
        "baseline_smoke_dir": str(baseline_smoke_dir),
        "promoted_smoke_dir": str(promoted_smoke_dir),
        "baseline_selector": baseline_selector,
        "baseline_score_col": baseline_score_col,
        "promoted_selector": promoted_selector,
        "promoted_score_col": promoted_score_col,
        "support_rule": {
            "min_valid_rows": int(min_valid_rows),
            "min_clean_rows": int(min_clean_rows),
            "min_positive_rows": int(min_positive_rows),
            "max_asset_share": float(max_asset_share),
            "max_week_share": float(max_week_share),
        },
        "leakage_contract": {
            "prediction_source": "OOF train_meta smoke predictions",
            "labels_used_for": "offline attribution only",
            "no_deployable_policy_selection": True,
        },
        "summary": _summary(cells, flips),
        "outputs": {
            "month_cell_effects": str(cells_path),
            "month_flips": str(flips_path),
            "json": str(out_dir / "promoted_cross_asset_month_flip_attribution.json"),
            "markdown": str(out_dir / "promoted_cross_asset_month_flip_attribution.md"),
        },
    }
    markdown = _write_markdown(out_dir, manifest, flips)
    manifest["outputs"]["markdown"] = str(markdown)
    (out_dir / "promoted_cross_asset_month_flip_attribution.json").write_text(
        json.dumps(_json_safe(manifest), indent=2, sort_keys=True) + "\n"
    )
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-smoke-dir", type=Path, default=DEFAULT_BASELINE_SMOKE_DIR)
    parser.add_argument("--promoted-smoke-dir", type=Path, default=DEFAULT_PROMOTED_SMOKE_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--min-valid-rows", type=int, default=20)
    parser.add_argument("--min-clean-rows", type=int, default=4)
    parser.add_argument("--min-positive-rows", type=int, default=4)
    parser.add_argument("--max-asset-share", type=float, default=0.85)
    parser.add_argument("--max-week-share", type=float, default=0.85)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = run_audit(
        baseline_smoke_dir=args.baseline_smoke_dir,
        promoted_smoke_dir=args.promoted_smoke_dir,
        out_dir=args.out_dir,
        min_valid_rows=args.min_valid_rows,
        min_clean_rows=args.min_clean_rows,
        min_positive_rows=args.min_positive_rows,
        max_asset_share=args.max_asset_share,
        max_week_share=args.max_week_share,
    )
    print(json.dumps(_json_safe({"event": "promoted_cross_asset_month_flip_attribution_done", **manifest}), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
