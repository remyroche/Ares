#!/usr/bin/env python3
"""Side-specific S52 path-quality geometry selection.

This is the side-aware counterpart to the global S52 geometry selector. It reads
completed bidirectional geometry sweeps, uses the side-family summaries, and
selects one geometry per side/top-k/cost using fit-month path-order metrics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from scripts.report_s52_geometry_breadth_selection import _json_safe, _num, _parse_csv
from scripts.report_s52_geometry_path_quality_cost_ladder import (
    _parse_costs_bps,
    _parse_float_csv,
    adjust_geometry_cost,
)
from scripts.report_s52_geometry_path_quality_selection import _apply_bars


DEFAULT_OUTPUT_DIR = Path("data_perp/reports/s52_side_geometry_path_quality_selection_v1")
DEFAULT_COSTS_BPS = "100"
DEFAULT_TOP_FRACS = "0.10,0.20,0.30"


def _side_summary_path(path: Path) -> Path:
    if path.is_file():
        return path
    candidate = path / "s52_bidirectional_geometry_side_family_summary.csv"
    if not candidate.exists():
        raise FileNotFoundError(f"Missing side geometry summary: {candidate}")
    return candidate


def _source_name(path: Path) -> str:
    return path.parent.name if path.is_file() else path.name


def read_side_sweeps(inputs: list[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in inputs:
        summary_path = _side_summary_path(path)
        frame = pd.read_csv(summary_path)
        frame["source"] = _source_name(path)
        frame["source_path"] = str(summary_path)
        frames.append(frame)
    if not frames:
        raise ValueError("At least one sweep directory or side summary CSV is required")
    rows = pd.concat(frames, ignore_index=True)
    if "side" not in rows.columns:
        raise ValueError("Side summary input must contain a side column")
    rows["side"] = rows["side"].astype(str).str.lower()
    return rows[rows["side"].isin(["long", "short"])].reset_index(drop=True)


def _dedupe(rows: pd.DataFrame) -> pd.DataFrame:
    keys = [
        col
        for col in [
            "cost_bps",
            "side",
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
    sort_cols = [col for col in ["fit_path_quality_score", "holdout_path_quality_score"] if col in rows.columns]
    if sort_cols:
        rows = rows.sort_values(sort_cols, ascending=[False] * len(sort_cols))
    return rows.drop_duplicates(keys, keep="first").reset_index(drop=True)


def select_side_rows(
    candidates: pd.DataFrame,
    *,
    costs: list[float],
    top_fracs: list[float],
    min_fit_rows: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    expanded_parts: list[pd.DataFrame] = []
    selected_parts: list[pd.DataFrame] = []
    for cost in costs:
        adjusted = adjust_geometry_cost(candidates, cost=float(cost))
        scored = _apply_bars(adjusted, min_rows=int(min_fit_rows), min_side_rows=0)
        expanded_parts.append(scored)
        for cost_bps, cost_frame in scored.groupby("cost_bps", observed=True, dropna=False):
            for side, side_frame in cost_frame.groupby("side", observed=True, dropna=False):
                for top_frac in top_fracs:
                    part = side_frame[
                        side_frame["top_frac"].map(lambda value: abs(float(value) - float(top_frac)) < 1e-9)
                    ].copy()
                    if part.empty:
                        continue
                    strict = part[part["fit_strict_path_bar"].astype(bool)].copy()
                    relative = part[part["fit_relative_path_bar"].astype(bool)].copy()
                    if not strict.empty:
                        pool = strict
                        reason = "fit_strict_path_bar_best"
                    elif not relative.empty:
                        pool = relative
                        reason = "fit_relative_path_bar_best"
                    else:
                        pool = part
                        reason = "fallback_fit_path_quality_score"
                    pool = pool.sort_values(
                        [
                            "fit_path_quality_score",
                            "fit_gross_ev_weighted_first_touch_precision",
                            "fit_mean_capture_net",
                            "fit_first_touch_bad_mae_1r_rate",
                            "fit_mean_underwater_bars_before_mfe_1r",
                        ],
                        ascending=[False, False, False, True, True],
                    )
                    chosen = pool.head(1).copy()
                    chosen["selection_reason"] = reason
                    chosen["selection_cost_bps"] = float(cost_bps)
                    chosen["selection_side"] = str(side)
                    selected_parts.append(chosen)
    expanded = _dedupe(pd.concat(expanded_parts, ignore_index=True)) if expanded_parts else pd.DataFrame()
    selected = pd.concat(selected_parts, ignore_index=True) if selected_parts else pd.DataFrame()
    return expanded, selected.reset_index(drop=True)


def _table(frame: pd.DataFrame, cols: list[str], n: int = 80) -> str:
    if frame.empty:
        return "No rows."
    view = frame[[col for col in cols if col in frame.columns]].head(n).copy()
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda value: f"{float(value):.6f}" if pd.notna(value) else "")
    return view.to_markdown(index=False)


def _write_report(output_dir: Path, candidates: pd.DataFrame, selected: pd.DataFrame, manifest: dict[str, Any]) -> None:
    cols = [
        "cost_bps",
        "side",
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
    counts = (
        candidates.groupby(["cost_bps", "side"], observed=True, dropna=False)
        .agg(
            rows=("arm", "size"),
            fit_relative_pass=("fit_relative_path_bar", "sum"),
            holdout_relative_pass=("holdout_relative_path_bar", "sum"),
            fit_strict_pass=("fit_strict_path_bar", "sum"),
            holdout_strict_pass=("holdout_strict_path_bar", "sum"),
        )
        .reset_index()
        .sort_values(["cost_bps", "side"])
    )
    lines = [
        "# S52 Side Geometry Path-Quality Selection",
        "",
        "Selection is fit-only and side-specific. Holdout columns are reported after selection.",
        "",
        f"Candidate rows: `{len(candidates)}`",
        f"Selected rows: `{len(selected)}`",
        "",
        "## Pass Counts",
        "",
        _table(counts, ["cost_bps", "side", "rows", "fit_relative_pass", "holdout_relative_pass", "fit_strict_pass", "holdout_strict_pass"]),
        "",
        "## Selected Side Geometries",
        "",
        _table(selected.sort_values(["cost_bps", "top_frac", "side"]), cols, n=200),
        "",
        "## Outputs",
        "",
        f"- Selected: `{manifest['outputs']['selected']}`",
        f"- Candidates: `{manifest['outputs']['candidates']}`",
        f"- Manifest: `{manifest['outputs']['manifest']}`",
    ]
    (output_dir / "s52_side_geometry_path_quality_selection.md").write_text(
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
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    base = read_side_sweeps(inputs)
    candidates, selected = select_side_rows(
        base,
        costs=costs,
        top_fracs=top_fracs,
        min_fit_rows=int(min_fit_rows),
    )
    paths = {
        "candidates": output_dir / "s52_side_geometry_path_quality_candidates.csv",
        "selected": output_dir / "s52_side_geometry_path_quality_selected.csv",
        "manifest": output_dir / "manifest.json",
        "markdown": output_dir / "s52_side_geometry_path_quality_selection.md",
    }
    candidates.to_csv(paths["candidates"], index=False)
    selected.to_csv(paths["selected"], index=False)
    manifest = {
        "scope": "s52_side_geometry_path_quality_selection",
        "inputs": [str(path) for path in inputs],
        "output_dir": str(output_dir),
        "costs_bps": [float(cost) * 10000.0 for cost in costs],
        "top_fracs": [float(v) for v in top_fracs],
        "min_fit_rows": int(min_fit_rows),
        "base_candidate_rows": int(len(base)),
        "expanded_candidate_rows": int(len(candidates)),
        "selected_rows": int(len(selected)),
        "outputs": {key: str(path) for key, path in paths.items()},
    }
    paths["manifest"].write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")
    _write_report(output_dir, candidates, selected, manifest)
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
    parser.add_argument("--inputs", required=True, help="Comma-separated sweep dirs or side summary CSV files.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--costs-bps", default=DEFAULT_COSTS_BPS)
    parser.add_argument("--top-fracs", default=DEFAULT_TOP_FRACS)
    parser.add_argument("--min-fit-rows", type=int, default=500)
    args = parser.parse_args()
    result = run_report(
        inputs=[Path(value) for value in _parse_csv(args.inputs)],
        output_dir=args.output_dir,
        costs=_parse_costs_bps(args.costs_bps),
        top_fracs=_parse_float_csv(args.top_fracs),
        min_fit_rows=int(args.min_fit_rows),
    )
    print(json.dumps(_json_safe(result), indent=2))


if __name__ == "__main__":
    main()
