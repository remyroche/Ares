#!/usr/bin/env python3
"""S52 path-quality geometry selection report.

This report reads completed S52 geometry sweeps and asks a narrower question:
which vol-normalized TP/SL geometries are plausible base-label candidates once
path order is treated as a first-class requirement?

Selection is fit-only. Holdout metrics are reported after selection so the
report can expose whether fit-selected geometries actually survive June.
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
    _parse_float_csv,
    _read_sweep,
)


DEFAULT_OUTPUT_DIR = Path("data_perp/reports/s52_geometry_path_quality_selection_v1")
DEFAULT_TOP_FRACS = (0.10, 0.20, 0.30)


def _top_frac_floor(top_frac: float) -> float:
    rounded = round(float(top_frac), 2)
    if rounded <= 0.10:
        return 0.70
    if rounded <= 0.20:
        return 0.65
    return 0.60


def _score_period(rows: pd.DataFrame, period: str) -> pd.Series:
    evw = _num(rows, f"{period}_gross_ev_weighted_first_touch_precision", 0.0).fillna(0.0)
    side_floor = _num(
        rows,
        f"{period}_min_side_gross_ev_weighted_first_touch_precision",
        np.nan,
    ).fillna(evw)
    net = _num(rows, f"{period}_mean_capture_net", 0.0).fillna(0.0).clip(lower=-0.02, upper=0.02)
    first_touch_bad = _num(rows, f"{period}_first_touch_bad_mae_1r_rate", 1.0).fillna(1.0)
    selected_bad = _num(rows, f"{period}_selected_path_bad_mae_1r_rate", 1.0).fillna(1.0)
    first_touch_p90_mae = _num(rows, f"{period}_first_touch_p90_mae_norm", 99.0).fillna(99.0)
    selected_p90_mae = _num(rows, f"{period}_selected_path_p90_mae_norm", 99.0).fillna(99.0)
    mae_before = _num(rows, f"{period}_mae_1r_before_mfe_1r_rate", 1.0).fillna(1.0)
    mfe_before = _num(rows, f"{period}_mfe_1r_before_mae_1r_rate", 0.0).fillna(0.0)
    max_adverse = _num(rows, f"{period}_mean_max_adverse_before_mfe_1r", 99.0).fillna(99.0)
    underwater_bars = _num(rows, f"{period}_mean_underwater_bars_before_mfe_1r", 99.0).fillna(99.0)
    underwater_frac = _num(rows, f"{period}_mean_underwater_fraction_before_mfe_1r", 1.0).fillna(1.0)
    timeout = _num(rows, f"{period}_timeout_rate", 1.0).fillna(1.0)
    rows_count = _num(rows, f"{period}_selected_rows", 0.0).fillna(0.0)

    score = (
        1.00 * evw
        + 0.30 * side_floor
        + 0.25 * mfe_before
        + 10.00 * net
        - 12.00 * (-net).clip(lower=0.0)
        - 0.65 * (first_touch_bad - 0.25).clip(lower=0.0)
        - 0.18 * (selected_bad - 0.65).clip(lower=0.0)
        - 0.14 * (first_touch_p90_mae - 3.0).clip(lower=0.0)
        - 0.03 * (selected_p90_mae - 9.0).clip(lower=0.0)
        - 0.45 * (mae_before - 0.35).clip(lower=0.0)
        - 0.25 * (max_adverse - 1.50).clip(lower=0.0)
        - 0.08 * (underwater_bars - 10.0).clip(lower=0.0)
        - 0.15 * (underwater_frac - 0.45).clip(lower=0.0)
        - 0.20 * (timeout - 0.12).clip(lower=0.0)
    )
    score = score.where(rows_count.gt(0.0), score - 1.0)
    return score.astype(np.float64)


def _bar_failures(
    row: pd.Series,
    *,
    period: str,
    strict: bool,
    min_rows: int,
    min_side_rows: int,
) -> list[str]:
    top_frac = float(row.get("top_frac", np.nan))
    evw_floor = _top_frac_floor(top_frac) if strict else max(0.50, _top_frac_floor(top_frac) - 0.15)
    first_touch_bad_cap = 0.25 if strict else 0.35
    first_touch_p90_cap = 3.0 if strict else 5.0
    mae_before_cap = 0.40 if strict else 0.45
    max_adverse_cap = 1.50 if strict else 2.50
    underwater_cap = 10.0 if strict else 18.0
    underwater_frac_cap = 0.45 if strict else 0.65
    timeout_cap = 0.12 if strict else 0.18

    failures: list[str] = []
    if float(row.get(f"{period}_selected_rows", 0.0) or 0.0) < float(min_rows):
        failures.append("rows")
    if f"{period}_min_side_selected_rows" in row and float(
        row.get(f"{period}_min_side_selected_rows", 0.0) or 0.0
    ) < float(min_side_rows):
        failures.append("side_rows")
    if float(row.get(f"{period}_gross_ev_weighted_first_touch_precision", -np.inf)) < evw_floor:
        failures.append("evw_precision")
    if f"{period}_min_side_gross_ev_weighted_first_touch_precision" in row and float(
        row.get(f"{period}_min_side_gross_ev_weighted_first_touch_precision", -np.inf)
    ) < max(0.0, evw_floor - 0.10):
        failures.append("side_evw_precision")
    if float(row.get(f"{period}_first_touch_bad_mae_1r_rate", np.inf)) > first_touch_bad_cap:
        failures.append("first_touch_bad_mae")
    if float(row.get(f"{period}_first_touch_p90_mae_norm", np.inf)) > first_touch_p90_cap:
        failures.append("first_touch_p90_mae")
    if float(row.get(f"{period}_mae_1r_before_mfe_1r_rate", np.inf)) > mae_before_cap:
        failures.append("mae_before_mfe")
    if float(row.get(f"{period}_mean_max_adverse_before_mfe_1r", np.inf)) > max_adverse_cap:
        failures.append("max_adverse_before_mfe")
    if float(row.get(f"{period}_mean_underwater_bars_before_mfe_1r", np.inf)) > underwater_cap:
        failures.append("underwater_bars")
    if float(row.get(f"{period}_mean_underwater_fraction_before_mfe_1r", np.inf)) > underwater_frac_cap:
        failures.append("underwater_fraction")
    if float(row.get(f"{period}_timeout_rate", np.inf)) > timeout_cap:
        failures.append("timeout")
    return failures


def _net_warning(row: pd.Series, *, period: str) -> str:
    net = float(row.get(f"{period}_mean_capture_net", np.nan))
    if not np.isfinite(net):
        return "missing_net"
    if net < -0.004:
        return "net_below_minus_40bps"
    if net < 0.0:
        return "net_negative_at_recorded_cost"
    return ""


def _full_horizon_warning_failures(row: pd.Series, *, period: str) -> list[str]:
    """Warnings for path pain after the hypothetical first-touch exit.

    These are not admission failures for first-passage geometry. They remain
    useful because a large gap between first-touch and full-horizon pain means
    the label is only coherent if the execution layer really exits on touch.
    """
    failures: list[str] = []
    if float(row.get(f"{period}_selected_path_bad_mae_1r_rate", np.inf)) > 0.65:
        failures.append("selected_path_bad_mae")
    if float(row.get(f"{period}_selected_path_p90_mae_norm", np.inf)) > 9.0:
        failures.append("selected_path_p90_mae")
    if float(row.get(f"{period}_target_full_path_bad_mae_1r_rate", np.inf)) > 0.65:
        failures.append("target_full_path_bad_mae")
    if float(row.get(f"{period}_target_full_path_mae_to_sl_p90", np.inf)) > 20.0:
        failures.append("target_full_path_mae_to_sl_p90")
    return failures


def _apply_bars(rows: pd.DataFrame, *, min_rows: int, min_side_rows: int) -> pd.DataFrame:
    out = rows.copy()
    out["fit_path_quality_score"] = _score_period(out, "fit")
    out["holdout_path_quality_score"] = _score_period(out, "holdout")
    for period in ("fit", "holdout"):
        strict_failures = []
        relative_failures = []
        for _, row in out.iterrows():
            strict_failures.append(
                _bar_failures(row, period=period, strict=True, min_rows=min_rows, min_side_rows=min_side_rows)
            )
            relative_failures.append(
                _bar_failures(
                    row,
                    period=period,
                    strict=False,
                    min_rows=min_rows,
                    min_side_rows=min_side_rows,
                )
            )
        out[f"{period}_strict_failures"] = [",".join(v) for v in strict_failures]
        out[f"{period}_relative_failures"] = [",".join(v) for v in relative_failures]
        out[f"{period}_strict_path_bar"] = [not bool(v) for v in strict_failures]
        out[f"{period}_relative_path_bar"] = [not bool(v) for v in relative_failures]
        full_warnings = [_full_horizon_warning_failures(row, period=period) for _, row in out.iterrows()]
        out[f"{period}_full_horizon_warnings"] = [",".join(v) for v in full_warnings]
        out[f"{period}_full_horizon_warning_count"] = [len(v) for v in full_warnings]
        out[f"{period}_net_warning"] = [_net_warning(row, period=period) for _, row in out.iterrows()]
    return out


def select_path_quality_rows(
    candidates: pd.DataFrame,
    *,
    top_fracs: list[float],
    min_fit_rows: int = 500,
    min_fit_side_rows: int = 100,
) -> pd.DataFrame:
    rows = _apply_bars(candidates, min_rows=min_fit_rows, min_side_rows=min_fit_side_rows)
    selected_parts: list[pd.DataFrame] = []
    for top_frac in top_fracs:
        part = rows[np.isclose(_num(rows, "top_frac", np.nan), float(top_frac))].copy()
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
                "fit_selected_path_p90_mae_norm",
                "fit_selected_path_bad_mae_1r_rate",
            ],
            ascending=[False, False, False, True, True],
        )
        chosen = pool.head(1).copy()
        chosen["selection_reason"] = reason
        selected_parts.append(chosen)
    if not selected_parts:
        return pd.DataFrame()
    return pd.concat(selected_parts, ignore_index=True).sort_values("top_frac").reset_index(drop=True)


def _load_candidates(inputs: list[Path]) -> pd.DataFrame:
    if not inputs:
        raise ValueError("At least one sweep directory or CSV is required")
    frames: list[pd.DataFrame] = []
    for path in inputs:
        if path.is_file() and path.name == "s52_geometry_breadth_candidates.csv":
            frame = pd.read_csv(path)
            if "source" not in frame.columns:
                frame["source"] = path.parent.name
            frame["source_path"] = str(path)
        else:
            frame = _read_sweep(path)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _table(frame: pd.DataFrame, cols: list[str], n: int = 20) -> str:
    if frame.empty:
        return "No rows."
    view = frame[[c for c in cols if c in frame.columns]].head(n).copy()
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda v: f"{float(v):.4f}" if pd.notna(v) else "")
    return view.to_markdown(index=False)


def _write_markdown(
    *,
    output_dir: Path,
    candidates: pd.DataFrame,
    selected: pd.DataFrame,
    manifest: dict[str, Any],
) -> None:
    cols = [
        "top_frac",
        "source",
        "arm",
        "selection_reason",
        "fit_strict_path_bar",
        "fit_relative_path_bar",
        "holdout_strict_path_bar",
        "holdout_relative_path_bar",
        "fit_path_quality_score",
        "holdout_path_quality_score",
        "fit_gross_ev_weighted_first_touch_precision",
        "holdout_gross_ev_weighted_first_touch_precision",
        "fit_mean_capture_net",
        "holdout_mean_capture_net",
        "fit_selected_path_bad_mae_1r_rate",
        "holdout_selected_path_bad_mae_1r_rate",
        "fit_first_touch_bad_mae_1r_rate",
        "holdout_first_touch_bad_mae_1r_rate",
        "fit_selected_path_p90_mae_norm",
        "holdout_selected_path_p90_mae_norm",
        "fit_first_touch_p90_mae_norm",
        "holdout_first_touch_p90_mae_norm",
        "fit_mae_1r_before_mfe_1r_rate",
        "holdout_mae_1r_before_mfe_1r_rate",
        "fit_timeout_rate",
        "holdout_timeout_rate",
        "holdout_strict_failures",
        "holdout_relative_failures",
        "holdout_full_horizon_warnings",
        "holdout_net_warning",
    ]
    top_fit = candidates.sort_values("fit_path_quality_score", ascending=False)
    strict_count = int(candidates["fit_strict_path_bar"].sum()) if "fit_strict_path_bar" in candidates else 0
    relative_count = int(candidates["fit_relative_path_bar"].sum()) if "fit_relative_path_bar" in candidates else 0
    holdout_strict_count = (
        int(candidates["holdout_strict_path_bar"].sum()) if "holdout_strict_path_bar" in candidates else 0
    )
    holdout_relative_count = (
        int(candidates["holdout_relative_path_bar"].sum()) if "holdout_relative_path_bar" in candidates else 0
    )
    lines = [
        "# S52 Geometry Path-Quality Selection",
        "",
        "Selection uses fit-month path-order metrics only. Holdout metrics are reported after selection.",
        "",
        f"Candidate rows: `{len(candidates)}`",
        f"Fit strict pass rows: `{strict_count}`",
        f"Fit relative pass rows: `{relative_count}`",
        f"Holdout strict pass rows: `{holdout_strict_count}`",
        f"Holdout relative pass rows: `{holdout_relative_count}`",
        "",
        "## Selected Geometries",
        "",
        _table(selected, cols, n=20),
        "",
        "## Top Fit Path-Quality Rows",
        "",
        _table(top_fit, cols, n=30),
        "",
        "## Outputs",
        "",
        f"- Selected: `{manifest['outputs']['selected']}`",
        f"- Candidates: `{manifest['outputs']['candidates']}`",
        f"- Manifest: `{manifest['outputs']['manifest']}`",
    ]
    (output_dir / "s52_geometry_path_quality_selection.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def run_report(
    *,
    inputs: list[Path],
    output_dir: Path,
    top_fracs: list[float],
    min_fit_rows: int,
    min_fit_side_rows: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates = _load_candidates(inputs)
    candidates = _apply_bars(candidates, min_rows=int(min_fit_rows), min_side_rows=int(min_fit_side_rows))
    selected = select_path_quality_rows(
        candidates,
        top_fracs=top_fracs,
        min_fit_rows=int(min_fit_rows),
        min_fit_side_rows=int(min_fit_side_rows),
    )
    paths = {
        "candidates": output_dir / "s52_geometry_path_quality_candidates.csv",
        "selected": output_dir / "s52_geometry_path_quality_selected.csv",
        "manifest": output_dir / "manifest.json",
        "markdown": output_dir / "s52_geometry_path_quality_selection.md",
    }
    candidates.to_csv(paths["candidates"], index=False)
    selected.to_csv(paths["selected"], index=False)
    manifest = {
        "scope": "s52_geometry_path_quality_selection",
        "inputs": [str(path) for path in inputs],
        "output_dir": str(output_dir),
        "top_fracs": [float(v) for v in top_fracs],
        "min_fit_rows": int(min_fit_rows),
        "min_fit_side_rows": int(min_fit_side_rows),
        "candidate_rows": int(len(candidates)),
        "selected_rows": int(len(selected)),
        "fit_strict_pass_rows": int(candidates["fit_strict_path_bar"].sum()),
        "fit_relative_pass_rows": int(candidates["fit_relative_path_bar"].sum()),
        "holdout_strict_pass_rows": int(candidates["holdout_strict_path_bar"].sum()),
        "holdout_relative_pass_rows": int(candidates["holdout_relative_path_bar"].sum()),
        "outputs": {key: str(value) for key, value in paths.items()},
    }
    paths["manifest"].write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")
    _write_markdown(output_dir=output_dir, candidates=candidates, selected=selected, manifest=manifest)
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
    parser.add_argument("--top-fracs", default="0.10,0.20,0.30")
    parser.add_argument("--min-fit-rows", type=int, default=500)
    parser.add_argument("--min-fit-side-rows", type=int, default=100)
    args = parser.parse_args()
    result = run_report(
        inputs=[Path(v) for v in _parse_csv(args.inputs)],
        output_dir=args.output_dir,
        top_fracs=_parse_float_csv(args.top_fracs, DEFAULT_TOP_FRACS),
        min_fit_rows=int(args.min_fit_rows),
        min_fit_side_rows=int(args.min_fit_side_rows),
    )
    print(json.dumps(_json_safe(result), indent=2))


if __name__ == "__main__":
    main()
