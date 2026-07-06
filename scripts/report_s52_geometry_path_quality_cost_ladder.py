#!/usr/bin/env python3
"""Cost ladder for S52 path-quality geometry selection.

The S52 plan asks for cost evaluation after path ordering. This report keeps
the first-passage/path-order metrics fixed, recomputes net from gross capture at
each cost level, and reselects geometries by the path-quality objective.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts.report_s52_geometry_breadth_selection import (
    _json_safe,
    _num,
    _parse_csv,
    _read_sweep,
)
from scripts.report_s52_geometry_path_quality_selection import (
    _apply_bars,
    select_path_quality_rows,
)


DEFAULT_OUTPUT_DIR = Path("data_perp/reports/s52_geometry_path_quality_cost_ladder_v1")
DEFAULT_COSTS_BPS = "0,10,25,50,100"
DEFAULT_TOP_FRACS = "0.10,0.20,0.30"


def _parse_float_csv(raw: str) -> list[float]:
    out: list[float] = []
    for token in str(raw or "").split(","):
        token = token.strip()
        if token:
            out.append(float(token))
    return out


def _parse_costs_bps(raw: str) -> list[float]:
    costs = []
    for value in _parse_float_csv(raw):
        if value < 0.0:
            raise ValueError(f"cost bps must be non-negative: {value}")
        costs.append(value / 10000.0)
    return sorted(set(costs))


def _load_candidates(inputs: list[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in inputs:
        if path.is_file() and path.name.endswith(".csv"):
            frame = pd.read_csv(path)
            if "source" not in frame.columns:
                frame["source"] = path.parent.name
            frame["source_path"] = str(path)
        else:
            frame = _read_sweep(path)
        frames.append(frame)
    if not frames:
        raise ValueError("At least one input sweep directory or CSV is required")
    return pd.concat(frames, ignore_index=True)


def adjust_geometry_cost(frame: pd.DataFrame, *, cost: float) -> pd.DataFrame:
    """Return a copy with period net columns recomputed from gross capture."""
    adjusted = frame.copy()
    for period in ("all", "fit", "holdout"):
        gross_col = f"{period}_mean_capture_gross"
        net_col = f"{period}_mean_capture_net"
        if gross_col in adjusted.columns:
            adjusted[net_col] = _num(adjusted, gross_col, 0.0).fillna(0.0) - float(cost)
    adjusted["cost"] = float(cost)
    adjusted["cost_bps"] = float(cost) * 10000.0
    return adjusted


def _dedupe_geometry_rows(rows: pd.DataFrame) -> pd.DataFrame:
    keys = [
        col
        for col in [
            "cost_bps",
            "top_frac",
            "arm",
            "selection_mode",
            "regime_family",
            "tp_r",
            "sl_r",
            "trail_r",
            "max_bars_to_mfe",
            "max_barrier",
        ]
        if col in rows.columns
    ]
    if not keys:
        return rows
    sort_cols = [c for c in ["fit_path_quality_score", "holdout_path_quality_score"] if c in rows.columns]
    if sort_cols:
        rows = rows.sort_values(sort_cols, ascending=[False] * len(sort_cols))
    return rows.drop_duplicates(keys, keep="first").reset_index(drop=True)


def build_cost_ladder(
    candidates: pd.DataFrame,
    *,
    costs: list[float],
    top_fracs: list[float],
    min_fit_rows: int,
    min_fit_side_rows: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    expanded_parts: list[pd.DataFrame] = []
    selected_parts: list[pd.DataFrame] = []
    for cost in costs:
        adjusted = adjust_geometry_cost(candidates, cost=float(cost))
        adjusted = _apply_bars(adjusted, min_rows=int(min_fit_rows), min_side_rows=int(min_fit_side_rows))
        expanded_parts.append(adjusted)
        selected = select_path_quality_rows(
            adjusted,
            top_fracs=top_fracs,
            min_fit_rows=int(min_fit_rows),
            min_fit_side_rows=int(min_fit_side_rows),
        )
        if not selected.empty:
            selected["cost"] = float(cost)
            selected["cost_bps"] = float(cost) * 10000.0
            selected_parts.append(selected)
    expanded = pd.concat(expanded_parts, ignore_index=True) if expanded_parts else pd.DataFrame()
    selected_all = pd.concat(selected_parts, ignore_index=True) if selected_parts else pd.DataFrame()
    return _dedupe_geometry_rows(expanded), selected_all.reset_index(drop=True)


def _table(frame: pd.DataFrame, cols: list[str], n: int = 40) -> str:
    if frame.empty:
        return "No rows."
    view = frame[[c for c in cols if c in frame.columns]].head(n).copy()
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda v: f"{float(v):.6f}" if pd.notna(v) else "")
    return view.to_markdown(index=False)


def _write_markdown(
    *,
    output_dir: Path,
    selected: pd.DataFrame,
    candidates: pd.DataFrame,
    manifest: dict[str, Any],
) -> None:
    cols = [
        "cost_bps",
        "top_frac",
        "arm",
        "selection_reason",
        "fit_relative_path_bar",
        "holdout_relative_path_bar",
        "fit_path_quality_score",
        "holdout_path_quality_score",
        "fit_gross_ev_weighted_first_touch_precision",
        "holdout_gross_ev_weighted_first_touch_precision",
        "fit_mean_capture_gross",
        "holdout_mean_capture_gross",
        "fit_mean_capture_net",
        "holdout_mean_capture_net",
        "fit_first_touch_bad_mae_1r_rate",
        "holdout_first_touch_bad_mae_1r_rate",
        "fit_first_touch_p90_mae_norm",
        "holdout_first_touch_p90_mae_norm",
        "fit_mae_1r_before_mfe_1r_rate",
        "holdout_mae_1r_before_mfe_1r_rate",
        "fit_mean_underwater_bars_before_mfe_1r",
        "holdout_mean_underwater_bars_before_mfe_1r",
        "holdout_relative_failures",
        "holdout_net_warning",
    ]
    best_by_cost = (
        selected.sort_values(["cost_bps", "top_frac"], ascending=[True, True])
        if not selected.empty
        else selected
    )
    pass_counts = (
        candidates.groupby("cost_bps", observed=True, dropna=False)
        .agg(
            rows=("arm", "size"),
            fit_relative_pass=("fit_relative_path_bar", "sum"),
            holdout_relative_pass=("holdout_relative_path_bar", "sum"),
            fit_strict_pass=("fit_strict_path_bar", "sum"),
            holdout_strict_pass=("holdout_strict_path_bar", "sum"),
        )
        .reset_index()
        .sort_values("cost_bps")
    )
    lines = [
        "# S52 Geometry Path-Quality Cost Ladder",
        "",
        "Net columns are recomputed from `mean_capture_gross - cost`; path-order metrics are unchanged.",
        "",
        f"Candidate rows per cost before de-duplication: `{manifest['candidate_rows_per_cost']}`",
        f"Costs: `{', '.join(str(v) for v in manifest['costs_bps'])}` bps",
        "",
        "## Pass Counts",
        "",
        _table(pass_counts, ["cost_bps", "rows", "fit_relative_pass", "holdout_relative_pass", "fit_strict_pass", "holdout_strict_pass"]),
        "",
        "## Selected Geometries By Cost",
        "",
        _table(best_by_cost, cols, n=200),
        "",
        "## Outputs",
        "",
        f"- Selected: `{manifest['outputs']['selected']}`",
        f"- Candidates: `{manifest['outputs']['candidates']}`",
        f"- Manifest: `{manifest['outputs']['manifest']}`",
    ]
    (output_dir / "s52_geometry_path_quality_cost_ladder.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def run_report(
    *,
    inputs: list[Path],
    output_dir: Path,
    costs: list[float],
    top_fracs: list[float],
    min_fit_rows: int,
    min_fit_side_rows: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    base = _load_candidates(inputs)
    candidates, selected = build_cost_ladder(
        base,
        costs=costs,
        top_fracs=top_fracs,
        min_fit_rows=int(min_fit_rows),
        min_fit_side_rows=int(min_fit_side_rows),
    )
    paths = {
        "candidates": output_dir / "s52_geometry_path_quality_cost_ladder_candidates.csv",
        "selected": output_dir / "s52_geometry_path_quality_cost_ladder_selected.csv",
        "manifest": output_dir / "manifest.json",
        "markdown": output_dir / "s52_geometry_path_quality_cost_ladder.md",
    }
    candidates.to_csv(paths["candidates"], index=False)
    selected.to_csv(paths["selected"], index=False)
    manifest = {
        "scope": "s52_geometry_path_quality_cost_ladder",
        "inputs": [str(path) for path in inputs],
        "output_dir": str(output_dir),
        "costs_bps": [float(cost) * 10000.0 for cost in costs],
        "top_fracs": [float(v) for v in top_fracs],
        "min_fit_rows": int(min_fit_rows),
        "min_fit_side_rows": int(min_fit_side_rows),
        "base_candidate_rows": int(len(base)),
        "candidate_rows_per_cost": int(len(base)),
        "expanded_candidate_rows": int(len(candidates)),
        "selected_rows": int(len(selected)),
        "outputs": {key: str(value) for key, value in paths.items()},
    }
    paths["manifest"].write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")
    _write_markdown(output_dir=output_dir, selected=selected, candidates=candidates, manifest=manifest)
    return {
        "output_dir": str(output_dir),
        "selected": str(paths["selected"]),
        "candidates": str(paths["candidates"]),
        "report": str(paths["markdown"]),
        "manifest": _json_safe(manifest),
        "top": _json_safe(selected.to_dict(orient="records")),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", required=True, help="Comma-separated sweep dirs or candidate CSV files.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--costs-bps", default=DEFAULT_COSTS_BPS)
    parser.add_argument("--top-fracs", default=DEFAULT_TOP_FRACS)
    parser.add_argument("--min-fit-rows", type=int, default=500)
    parser.add_argument("--min-fit-side-rows", type=int, default=100)
    args = parser.parse_args()
    result = run_report(
        inputs=[Path(value) for value in _parse_csv(args.inputs)],
        output_dir=args.output_dir,
        costs=_parse_costs_bps(args.costs_bps),
        top_fracs=_parse_float_csv(args.top_fracs),
        min_fit_rows=int(args.min_fit_rows),
        min_fit_side_rows=int(args.min_fit_side_rows),
    )
    print(json.dumps(_json_safe(result), indent=2))


if __name__ == "__main__":
    main()
