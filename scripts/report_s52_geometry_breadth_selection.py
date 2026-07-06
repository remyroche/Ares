#!/usr/bin/env python3
"""Fit-selected S52 geometry breadth decision report.

This report is intentionally selection-only: it reads completed S52 geometry
sweeps, chooses candidate geometries using fit-month metrics only, and then
reports holdout metrics. It is used to separate "which geometry is learnable"
from downstream model/ranker changes.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_OUTPUT_DIR = Path("data_perp/reports/s52_geometry_breadth_selection_v1")
DEFAULT_RETENTION_TOP_FRACS = (0.10, 0.20, 0.30)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return [_json_safe(v) for v in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        out = float(value)
        return out if math.isfinite(out) else None
    if pd.isna(value):
        return None
    return value


def _parse_csv(value: str | None) -> list[str]:
    text = str(value or "").strip()
    if not text:
        return []
    return [part.strip() for part in text.split(",") if part.strip()]


def _parse_float_csv(value: str | None, default: tuple[float, ...]) -> list[float]:
    parsed = _parse_csv(value)
    return [float(v) for v in parsed] if parsed else list(default)


def _num(frame: pd.DataFrame, name: str, default: float = np.nan) -> pd.Series:
    if name in frame.columns:
        return pd.to_numeric(frame[name], errors="coerce")
    return pd.Series(float(default), index=frame.index, dtype=np.float64)


def _source_name(path: Path) -> str:
    if path.is_dir():
        return path.name
    parent = path.parent.name
    return parent or path.stem


def _summary_path(path: Path) -> Path:
    if path.is_file():
        return path
    candidate = path / "s52_bidirectional_geometry_summary.csv"
    if not candidate.exists():
        raise FileNotFoundError(f"Missing geometry summary: {candidate}")
    return candidate


def _side_summary_path(path: Path) -> Path | None:
    directory = path if path.is_dir() else path.parent
    candidate = directory / "s52_bidirectional_geometry_side_family_summary.csv"
    return candidate if candidate.exists() else None


def _read_sweep(path: Path) -> pd.DataFrame:
    summary_path = _summary_path(path)
    summary = pd.read_csv(summary_path)
    summary["source"] = _source_name(path)
    summary["source_path"] = str(summary_path)
    key_cols = ["source", "arm", "selection_mode", "top_frac", "regime_family"]
    side_path = _side_summary_path(path)
    if side_path is None:
        return summary
    side = pd.read_csv(side_path)
    side["source"] = _source_name(path)
    side_key_cols = [c for c in key_cols if c in side.columns]
    if not side_key_cols:
        return summary

    side_rows: list[dict[str, Any]] = []
    for key, group in side.groupby(side_key_cols, observed=True, dropna=False):
        row = dict(zip(side_key_cols, key if isinstance(key, tuple) else (key,)))
        for period in ("fit", "holdout"):
            ev = _num(group, f"{period}_gross_ev_weighted_first_touch_precision")
            rows = _num(group, f"{period}_selected_rows", 0.0).fillna(0.0)
            valid = ev.notna() & rows.gt(0.0)
            row[f"{period}_min_side_gross_ev_weighted_first_touch_precision"] = (
                float(ev[valid].min()) if bool(valid.any()) else float("nan")
            )
            row[f"{period}_max_side_first_touch_bad_mae_1r_rate"] = float(
                _num(group, f"{period}_first_touch_bad_mae_1r_rate").max()
            )
            row[f"{period}_max_side_mae_1r_before_mfe_1r_rate"] = float(
                _num(group, f"{period}_mae_1r_before_mfe_1r_rate").max()
            )
            row[f"{period}_max_side_mean_underwater_bars_before_mfe_1r"] = float(
                _num(group, f"{period}_mean_underwater_bars_before_mfe_1r").max()
            )
            row[f"{period}_min_side_selected_rows"] = float(rows.min()) if len(rows) else float("nan")
        side_rows.append(row)
    side_summary = pd.DataFrame(side_rows)
    join_cols = [c for c in key_cols if c in summary.columns and c in side_summary.columns]
    if not join_cols:
        return summary
    return summary.merge(side_summary, on=join_cols, how="left", validate="many_to_one")


def _breadth_objective(frame: pd.DataFrame) -> pd.Series:
    fit_evw = _num(frame, "fit_gross_ev_weighted_first_touch_precision", 0.0).fillna(0.0)
    fit_side_floor = _num(
        frame,
        "fit_min_side_gross_ev_weighted_first_touch_precision",
        np.nan,
    ).fillna(fit_evw)
    fit_mfe_before = _num(frame, "fit_mfe_1r_before_mae_1r_rate", 0.0).fillna(0.0)
    fit_mae_before = _num(frame, "fit_mae_1r_before_mfe_1r_rate", 1.0).fillna(1.0)
    fit_bad_mae = _num(frame, "fit_first_touch_bad_mae_1r_rate", 1.0).fillna(1.0)
    fit_max_adverse = _num(frame, "fit_mean_max_adverse_before_mfe_1r", 99.0).fillna(99.0)
    fit_underwater = _num(frame, "fit_mean_underwater_bars_before_mfe_1r", 99.0).fillna(99.0)
    fit_underwater_frac = _num(frame, "fit_mean_underwater_fraction_before_mfe_1r", 1.0).fillna(1.0)
    fit_timeout = _num(frame, "fit_timeout_rate", 1.0).fillna(1.0)
    fit_net = _num(frame, "fit_mean_capture_net", 0.0).fillna(0.0)
    score = (
        1.00 * fit_evw
        + 0.35 * fit_side_floor
        + 0.20 * fit_mfe_before
        + 8.00 * fit_net.clip(lower=-0.02, upper=0.02)
        - 8.00 * (-fit_net).clip(lower=0.0, upper=0.02)
        - 0.55 * (fit_bad_mae - 0.25).clip(lower=0.0)
        - 0.45 * (fit_mae_before - 0.35).clip(lower=0.0)
        - 0.25 * (fit_max_adverse - 1.50).clip(lower=0.0)
        - 0.08 * (fit_underwater - 10.0).clip(lower=0.0)
        - 0.15 * (fit_underwater_frac - 0.45).clip(lower=0.0)
        - 0.20 * (fit_timeout - 0.12).clip(lower=0.0)
    )
    return score.astype(np.float64)


def _eligible_rows(
    rows: pd.DataFrame,
    *,
    min_fit_selected_rows: int,
    min_fit_side_rows: int,
    min_fit_evw: float,
    min_fit_side_evw: float,
    min_fit_net: float,
    max_fit_bad_mae: float,
    max_fit_mae_before: float,
    max_fit_max_adverse: float,
    max_fit_underwater: float,
    max_fit_underwater_fraction: float,
) -> pd.Series:
    mask = _num(rows, "fit_selected_rows", 0.0).ge(float(min_fit_selected_rows))
    if "fit_min_side_selected_rows" in rows.columns:
        mask &= _num(rows, "fit_min_side_selected_rows", 0.0).ge(float(min_fit_side_rows))
    if "fit_min_side_gross_ev_weighted_first_touch_precision" in rows.columns:
        mask &= _num(rows, "fit_min_side_gross_ev_weighted_first_touch_precision", 0.0).ge(
            float(min_fit_side_evw)
        )
    mask &= _num(rows, "fit_gross_ev_weighted_first_touch_precision", 0.0).ge(float(min_fit_evw))
    mask &= _num(rows, "fit_mean_capture_net", -np.inf).ge(float(min_fit_net))
    mask &= _num(rows, "fit_first_touch_bad_mae_1r_rate", 1.0).le(float(max_fit_bad_mae))
    mask &= _num(rows, "fit_mae_1r_before_mfe_1r_rate", 1.0).le(float(max_fit_mae_before))
    mask &= _num(rows, "fit_mean_max_adverse_before_mfe_1r", 0.0).le(float(max_fit_max_adverse))
    mask &= _num(rows, "fit_mean_underwater_bars_before_mfe_1r", 99.0).le(float(max_fit_underwater))
    mask &= _num(rows, "fit_mean_underwater_fraction_before_mfe_1r", 1.0).le(
        float(max_fit_underwater_fraction)
    )
    return mask.fillna(False)


def select_breadth_rows(
    candidates: pd.DataFrame,
    *,
    top_fracs: list[float],
    min_fit_selected_rows: int,
    min_fit_side_rows: int,
    min_fit_evw: float,
    min_fit_side_evw: float = 0.0,
    min_fit_net: float = -float("inf"),
    max_fit_bad_mae: float = 0.35,
    max_fit_mae_before: float = 0.40,
    max_fit_max_adverse: float = float("inf"),
    max_fit_underwater: float = 16.0,
    max_fit_underwater_fraction: float = 1.0,
) -> pd.DataFrame:
    rows = candidates.copy()
    rows["breadth_objective_fit"] = _breadth_objective(rows)
    rows["eligible"] = _eligible_rows(
        rows,
        min_fit_selected_rows=int(min_fit_selected_rows),
        min_fit_side_rows=int(min_fit_side_rows),
        min_fit_evw=float(min_fit_evw),
        min_fit_side_evw=float(min_fit_side_evw),
        min_fit_net=float(min_fit_net),
        max_fit_bad_mae=float(max_fit_bad_mae),
        max_fit_mae_before=float(max_fit_mae_before),
        max_fit_max_adverse=float(max_fit_max_adverse),
        max_fit_underwater=float(max_fit_underwater),
        max_fit_underwater_fraction=float(max_fit_underwater_fraction),
    )
    selected_parts: list[pd.DataFrame] = []
    for top_frac in top_fracs:
        slice_rows = rows[np.isclose(_num(rows, "top_frac", np.nan), float(top_frac))].copy()
        if slice_rows.empty:
            continue
        eligible = slice_rows[slice_rows["eligible"].astype(bool)].copy()
        pool = eligible if not eligible.empty else slice_rows
        pool = pool.sort_values(
            [
                "breadth_objective_fit",
                "fit_gross_ev_weighted_first_touch_precision",
                "fit_mean_capture_net",
                "fit_first_touch_bad_mae_1r_rate",
            ],
            ascending=[False, False, False, True],
        )
        chosen = pool.head(1).copy()
        chosen["selection_reason"] = "eligible_fit_best" if not eligible.empty else "fallback_no_eligible"
        selected_parts.append(chosen)
    if not selected_parts:
        return pd.DataFrame()
    out = pd.concat(selected_parts, ignore_index=True)
    return out.sort_values("top_frac").reset_index(drop=True)


def _gate_status(row: pd.Series) -> str:
    failures = []
    top_frac = float(row.get("top_frac", np.nan))
    ev_floor = {0.10: 0.70, 0.20: 0.65, 0.30: 0.60}.get(round(top_frac, 2), 0.60)
    if float(row.get("holdout_gross_ev_weighted_first_touch_precision", -np.inf)) < ev_floor:
        failures.append("holdout_evw")
    if float(row.get("holdout_first_touch_bad_mae_1r_rate", np.inf)) > 0.25:
        failures.append("holdout_first_touch_bad_mae")
    if float(row.get("holdout_mae_1r_before_mfe_1r_rate", np.inf)) > 0.35:
        failures.append("holdout_mae_before_mfe")
    if float(row.get("holdout_mean_max_adverse_before_mfe_1r", np.inf)) > 1.50:
        failures.append("holdout_max_adverse")
    if float(row.get("holdout_mean_underwater_bars_before_mfe_1r", np.inf)) > 10.0:
        failures.append("holdout_underwater_bars")
    if float(row.get("holdout_mean_underwater_fraction_before_mfe_1r", np.inf)) > 0.45:
        failures.append("holdout_underwater_fraction")
    if float(row.get("holdout_timeout_rate", np.inf)) > 0.12:
        failures.append("holdout_timeout")
    if float(row.get("holdout_mean_capture_net", -np.inf)) < 0.0:
        failures.append("holdout_net")
    return "pass" if not failures else "fail:" + ",".join(failures)


def _write_report(output_dir: Path, selected: pd.DataFrame, candidates: pd.DataFrame, manifest: dict[str, Any]) -> None:
    def table(frame: pd.DataFrame, cols: list[str], n: int = 20) -> str:
        if frame.empty:
            return "No rows."
        view = frame[[c for c in cols if c in frame.columns]].head(n).copy()
        for col in view.columns:
            if pd.api.types.is_float_dtype(view[col]):
                view[col] = view[col].map(lambda v: f"{float(v):.4f}" if pd.notna(v) else "")
        return view.to_markdown(index=False)

    cols = [
        "top_frac",
        "source",
        "arm",
        "selection_reason",
        "breadth_objective_fit",
        "fit_gross_ev_weighted_first_touch_precision",
        "holdout_gross_ev_weighted_first_touch_precision",
        "fit_mean_capture_net",
        "holdout_mean_capture_net",
        "fit_first_touch_bad_mae_1r_rate",
        "holdout_first_touch_bad_mae_1r_rate",
        "fit_mae_1r_before_mfe_1r_rate",
        "holdout_mae_1r_before_mfe_1r_rate",
        "fit_mean_max_adverse_before_mfe_1r",
        "holdout_mean_max_adverse_before_mfe_1r",
        "fit_mean_underwater_bars_before_mfe_1r",
        "holdout_mean_underwater_bars_before_mfe_1r",
        "fit_mean_underwater_fraction_before_mfe_1r",
        "holdout_mean_underwater_fraction_before_mfe_1r",
        "fit_min_side_gross_ev_weighted_first_touch_precision",
        "holdout_min_side_gross_ev_weighted_first_touch_precision",
        "gate_status",
    ]
    top_candidates = candidates.sort_values("breadth_objective_fit", ascending=False)
    lines = [
        "# S52 Geometry Breadth Selection",
        "",
        "Selection uses fit-month metrics only. Holdout columns are reported after selection.",
        "",
        f"Candidate rows: `{len(candidates)}`",
        f"Selected rows: `{len(selected)}`",
        f"Inputs: `{', '.join(manifest['inputs'])}`",
        "",
        "## Selected Fit-Only Candidates",
        "",
        table(selected, cols, n=20),
        "",
        "## Top Fit Candidates",
        "",
        table(top_candidates, cols, n=30),
        "",
        "## Outputs",
        "",
        f"- Selected: `{manifest['outputs']['selected']}`",
        f"- Candidates: `{manifest['outputs']['candidates']}`",
        f"- Manifest: `{manifest['outputs']['manifest']}`",
    ]
    (output_dir / "s52_geometry_breadth_selection.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_report(
    *,
    inputs: list[Path],
    output_dir: Path,
    top_fracs: list[float],
    min_fit_selected_rows: int,
    min_fit_side_rows: int,
    min_fit_evw: float,
    min_fit_side_evw: float,
    min_fit_net: float,
    max_fit_bad_mae: float,
    max_fit_mae_before: float,
    max_fit_max_adverse: float,
    max_fit_underwater: float,
    max_fit_underwater_fraction: float,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    if not inputs:
        raise ValueError("At least one input sweep directory or summary CSV is required")
    candidates = pd.concat([_read_sweep(Path(path)) for path in inputs], ignore_index=True)
    candidates["breadth_objective_fit"] = _breadth_objective(candidates)
    candidates["eligible"] = _eligible_rows(
        candidates,
        min_fit_selected_rows=int(min_fit_selected_rows),
        min_fit_side_rows=int(min_fit_side_rows),
        min_fit_evw=float(min_fit_evw),
        min_fit_side_evw=float(min_fit_side_evw),
        min_fit_net=float(min_fit_net),
        max_fit_bad_mae=float(max_fit_bad_mae),
        max_fit_mae_before=float(max_fit_mae_before),
        max_fit_max_adverse=float(max_fit_max_adverse),
        max_fit_underwater=float(max_fit_underwater),
        max_fit_underwater_fraction=float(max_fit_underwater_fraction),
    )
    selected = select_breadth_rows(
        candidates,
        top_fracs=top_fracs,
        min_fit_selected_rows=int(min_fit_selected_rows),
        min_fit_side_rows=int(min_fit_side_rows),
        min_fit_evw=float(min_fit_evw),
        min_fit_side_evw=float(min_fit_side_evw),
        min_fit_net=float(min_fit_net),
        max_fit_bad_mae=float(max_fit_bad_mae),
        max_fit_mae_before=float(max_fit_mae_before),
        max_fit_max_adverse=float(max_fit_max_adverse),
        max_fit_underwater=float(max_fit_underwater),
        max_fit_underwater_fraction=float(max_fit_underwater_fraction),
    )
    if not selected.empty:
        selected["gate_status"] = selected.apply(_gate_status, axis=1)
    paths = {
        "candidates": output_dir / "s52_geometry_breadth_candidates.csv",
        "selected": output_dir / "s52_geometry_breadth_selected.csv",
        "manifest": output_dir / "manifest.json",
        "markdown": output_dir / "s52_geometry_breadth_selection.md",
    }
    candidates.to_csv(paths["candidates"], index=False)
    selected.to_csv(paths["selected"], index=False)
    manifest = {
        "scope": "s52_geometry_breadth_selection",
        "inputs": [str(path) for path in inputs],
        "output_dir": str(output_dir),
        "top_fracs": [float(v) for v in top_fracs],
        "min_fit_selected_rows": int(min_fit_selected_rows),
        "min_fit_side_rows": int(min_fit_side_rows),
        "min_fit_evw": float(min_fit_evw),
        "min_fit_side_evw": float(min_fit_side_evw),
        "min_fit_net": float(min_fit_net),
        "max_fit_bad_mae": float(max_fit_bad_mae),
        "max_fit_mae_before": float(max_fit_mae_before),
        "max_fit_max_adverse": float(max_fit_max_adverse),
        "max_fit_underwater": float(max_fit_underwater),
        "max_fit_underwater_fraction": float(max_fit_underwater_fraction),
        "candidate_rows": int(len(candidates)),
        "selected_rows": int(len(selected)),
        "outputs": {key: str(value) for key, value in paths.items()},
    }
    paths["manifest"].write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")
    _write_report(output_dir, selected, candidates, manifest)
    return {
        "output_dir": str(output_dir),
        "selected": str(paths["selected"]),
        "candidates": str(paths["candidates"]),
        "report": str(paths["markdown"]),
        "top": _json_safe(selected.to_dict(orient="records")),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", required=True, help="Comma-separated sweep dirs or summary CSV files.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--top-fracs", default="0.10,0.20,0.30")
    parser.add_argument("--min-fit-selected-rows", type=int, default=500)
    parser.add_argument("--min-fit-side-rows", type=int, default=100)
    parser.add_argument("--min-fit-evw", type=float, default=0.35)
    parser.add_argument("--min-fit-side-evw", type=float, default=0.0)
    parser.add_argument("--min-fit-net", type=float, default=-float("inf"))
    parser.add_argument("--max-fit-bad-mae", type=float, default=0.35)
    parser.add_argument("--max-fit-mae-before", type=float, default=0.40)
    parser.add_argument("--max-fit-max-adverse", type=float, default=float("inf"))
    parser.add_argument("--max-fit-underwater", type=float, default=16.0)
    parser.add_argument("--max-fit-underwater-fraction", type=float, default=1.0)
    args = parser.parse_args()
    result = run_report(
        inputs=[Path(v) for v in _parse_csv(args.inputs)],
        output_dir=args.output_dir,
        top_fracs=_parse_float_csv(args.top_fracs, DEFAULT_RETENTION_TOP_FRACS),
        min_fit_selected_rows=int(args.min_fit_selected_rows),
        min_fit_side_rows=int(args.min_fit_side_rows),
        min_fit_evw=float(args.min_fit_evw),
        min_fit_side_evw=float(args.min_fit_side_evw),
        min_fit_net=float(args.min_fit_net),
        max_fit_bad_mae=float(args.max_fit_bad_mae),
        max_fit_mae_before=float(args.max_fit_mae_before),
        max_fit_max_adverse=float(args.max_fit_max_adverse),
        max_fit_underwater=float(args.max_fit_underwater),
        max_fit_underwater_fraction=float(args.max_fit_underwater_fraction),
    )
    print(json.dumps(_json_safe(result), indent=2))


if __name__ == "__main__":
    main()
