#!/usr/bin/env python3
"""Compare wide-stop capture proxy rows with oracle capture sorting.

This report is a no-training diagnostic. It asks whether the fixed-capture
execution definition is economically viable in the label artifact, and whether
the cheap causal proxy can recover it under the same Apr-May -> June gate.
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

from scripts.run_label_quality_proxy_diagnostics import (  # noqa: E402
    DEFAULT_LABELS_DIR,
    _json_safe,
    _load_labels,
    _path_metrics,
)
from scripts.run_label_widestop_capture_proxy import (  # noqa: E402
    CAPTURE_ARMS,
    _capture_outcome,
    _fit_holdout_summary,
    _selection_metrics,
    _weekly_rows,
)


DEFAULT_OUTPUT_DIR = Path("data_perp/reports/label_widestop_capture_oracle_gap_v1")


def _parse_csv(value: str) -> list[str]:
    return [part.strip() for part in str(value).split(",") if part.strip()]


def _parse_float_csv(value: str) -> list[float]:
    return [float(part.strip()) for part in str(value).split(",") if part.strip()]


def _safe_numeric(frame: pd.DataFrame, col: str) -> pd.Series:
    if col not in frame:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[col], errors="coerce")


def _oracle_fit_holdout(
    *,
    labels_path: Path,
    months: list[str],
    top_fracs: list[float],
    fit_months: list[str],
    holdout_month: str,
    min_week_rows: int,
) -> pd.DataFrame:
    frame = _load_labels(labels_path)
    metrics = _path_metrics(frame)
    month_series = frame["__ts__"].dt.to_period("M").astype(str)
    monthly_rows: list[dict[str, Any]] = []
    weekly_rows: list[dict[str, Any]] = []
    for month in months:
        mask = month_series.eq(str(month))
        if int(mask.sum()) < 100:
            continue
        valid = frame.loc[mask].copy().reset_index(drop=True)
        valid_metrics = metrics.loc[mask].copy().reset_index(drop=True)
        for arm in CAPTURE_ARMS:
            target = _capture_outcome(valid_metrics, arm)
            scores = {
                "oracle_soft": target["target_soft"],
                "oracle_hard": target["target_hard"] + 0.01 * target["target_soft"],
                "oracle_net": target["capture_net"],
            }
            for score_name, score in scores.items():
                oracle_arm = f"{arm.name}::{score_name}"
                for top_frac in top_fracs:
                    row = _selection_metrics(
                        frame=valid,
                        metrics=valid_metrics,
                        target=target,
                        score=score,
                        arm=oracle_arm,
                        period=str(month),
                        top_frac=float(top_frac),
                        selection_mode="global",
                    )
                    row.update(
                        {
                            "selection_mode": "oracle_global",
                            "tp_r": arm.tp_r,
                            "sl_r": arm.sl_r,
                            "max_bars_to_mfe": arm.max_bars_to_mfe,
                            "max_barrier": arm.max_barrier,
                        }
                    )
                    monthly_rows.append(row)
                    for week_row in _weekly_rows(
                        frame=valid,
                        metrics=valid_metrics,
                        target=target,
                        score=score,
                        arm=oracle_arm,
                        period=str(month),
                        top_frac=float(top_frac),
                        selection_mode="global",
                    ):
                        week_row.update(
                            {
                                "selection_mode": "oracle_global",
                                "tp_r": arm.tp_r,
                                "sl_r": arm.sl_r,
                                "max_bars_to_mfe": arm.max_bars_to_mfe,
                                "max_barrier": arm.max_barrier,
                            }
                        )
                        weekly_rows.append(week_row)
    monthly = pd.DataFrame(monthly_rows)
    weekly = pd.DataFrame(weekly_rows)
    return _fit_holdout_summary(
        monthly=monthly,
        weekly=weekly,
        fit_months=fit_months,
        holdout_month=holdout_month,
        min_week_rows=min_week_rows,
    )


def _summarize_source(name: str, frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {
            "source": name,
            "rows": 0,
            "fit_sign_pass": 0,
            "holdout_sign_pass": 0,
            "fit_bounded_pass": 0,
            "holdout_bounded_pass": 0,
            "fit_strict_pass": 0,
            "holdout_strict_pass": 0,
            "positive_dirty_holdout": 0,
            "best_arm": "",
            "best_top_frac": float("nan"),
            "best_holdout_capture_net": float("nan"),
            "best_holdout_hit_rate": float("nan"),
            "best_holdout_stop_rate": float("nan"),
        }
    best = frame.sort_values(
        ["holdout_strict_pass", "holdout_bounded_pass", "positive_dirty_holdout", "capture_proxy_score"],
        ascending=[False, False, False, False],
    ).iloc[0]
    return {
        "source": name,
        "rows": int(len(frame)),
        "fit_sign_pass": int(frame["fit_sign_pass"].sum()),
        "holdout_sign_pass": int(frame["holdout_sign_pass"].sum()),
        "fit_bounded_pass": int(frame["fit_bounded_pass"].sum()),
        "holdout_bounded_pass": int(frame["holdout_bounded_pass"].sum()),
        "fit_strict_pass": int(frame["fit_strict_pass"].sum()),
        "holdout_strict_pass": int(frame["holdout_strict_pass"].sum()),
        "positive_dirty_holdout": int(frame["positive_dirty_holdout"].sum()),
        "best_arm": str(best.get("arm", "")),
        "best_top_frac": float(best.get("top_frac", float("nan"))),
        "best_fit_capture_net": float(best.get("fit_mean_capture_net", float("nan"))),
        "best_holdout_capture_net": float(best.get("holdout_mean_capture_net", float("nan"))),
        "best_holdout_hit_rate": float(best.get("holdout_hit_rate", float("nan"))),
        "best_holdout_stop_rate": float(best.get("holdout_stop_rate", float("nan"))),
        "best_holdout_effective_sl_abs_p90": float(best.get("holdout_effective_sl_abs_p90", float("nan"))),
    }


def _format_table(frame: pd.DataFrame, cols: list[str], limit: int = 30) -> str:
    if frame.empty:
        return "No rows."
    view = frame[[col for col in cols if col in frame.columns]].head(limit).copy()
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda value: f"{float(value):.5f}" if pd.notna(value) else "")
    return view.to_markdown(index=False)


def _write_markdown(output_dir: Path, comparison: pd.DataFrame, oracle: pd.DataFrame, model_frames: dict[str, pd.DataFrame], manifest: dict[str, Any]) -> Path:
    path = output_dir / "label_widestop_capture_oracle_gap.md"
    summary_cols = [
        "source",
        "rows",
        "fit_sign_pass",
        "holdout_sign_pass",
        "fit_bounded_pass",
        "holdout_bounded_pass",
        "fit_strict_pass",
        "holdout_strict_pass",
        "positive_dirty_holdout",
        "best_arm",
        "best_top_frac",
        "best_fit_capture_net",
        "best_holdout_capture_net",
        "best_holdout_hit_rate",
        "best_holdout_stop_rate",
        "best_holdout_effective_sl_abs_p90",
    ]
    detail_cols = [
        "arm",
        "selection_mode",
        "top_frac",
        "capture_proxy_score",
        "fit_mean_capture_net",
        "fit_worst_capture_net",
        "fit_material_positive_week_rate",
        "holdout_mean_capture_net",
        "holdout_material_positive_week_rate",
        "holdout_hit_rate",
        "holdout_stop_rate",
        "holdout_effective_sl_abs_p90",
        "holdout_bounded_pass",
        "holdout_strict_pass",
    ]
    lines = [
        "# Wide-Stop Capture Oracle Gap",
        "",
        "Scope: no-training diagnostic. Oracle rows sort by realized capture labels and are not deployable; they prove whether the fixed-capture execution definition exists in the label artifact.",
        "",
        f"Labels: `{manifest['labels_path']}`",
        f"Fit months: `{','.join(manifest['fit_months'])}`",
        f"Holdout month: `{manifest['holdout_month']}`",
        f"Top fractions: `{','.join(str(v) for v in manifest['top_fracs'])}`",
        "",
        "## Summary",
        "",
        _format_table(comparison, summary_cols, limit=20),
        "",
        "## Oracle Best Rows",
        "",
        _format_table(
            oracle.sort_values(
                ["holdout_strict_pass", "holdout_bounded_pass", "capture_proxy_score"],
                ascending=[False, False, False],
            ),
            detail_cols,
            limit=30,
        ),
        "",
    ]
    for name, frame in model_frames.items():
        lines.extend(
            [
                f"## Model Proxy Best Rows: {name}",
                "",
                _format_table(frame.sort_values("capture_proxy_score", ascending=False), detail_cols, limit=20),
                "",
            ]
        )
    lines.extend(
        [
            "## Outputs",
            "",
            f"- Comparison: `{manifest['outputs']['comparison']}`",
            f"- Oracle fit/holdout: `{manifest['outputs']['oracle_fit_holdout']}`",
            f"- Manifest: `{manifest['outputs']['manifest']}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_report(
    *,
    labels_path: Path,
    model_dirs: list[Path],
    output_dir: Path,
    months: list[str],
    fit_months: list[str],
    holdout_month: str,
    top_fracs: list[float],
    min_week_rows: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    oracle = _oracle_fit_holdout(
        labels_path=labels_path,
        months=months,
        top_fracs=top_fracs,
        fit_months=fit_months,
        holdout_month=holdout_month,
        min_week_rows=min_week_rows,
    )
    model_frames: dict[str, pd.DataFrame] = {}
    summary_rows = [_summarize_source("oracle_capture_sort", oracle)]
    for model_dir in model_dirs:
        path = model_dir / "label_widestop_capture_proxy_fit_holdout.csv"
        frame = pd.read_csv(path)
        name = model_dir.name
        model_frames[name] = frame
        summary_rows.append(_summarize_source(name, frame))
    comparison = pd.DataFrame(summary_rows)
    paths = {
        "comparison": output_dir / "label_widestop_capture_model_vs_oracle.csv",
        "oracle_fit_holdout": output_dir / "label_widestop_capture_oracle_fit_holdout.csv",
        "manifest": output_dir / "manifest.json",
    }
    comparison.to_csv(paths["comparison"], index=False)
    oracle.to_csv(paths["oracle_fit_holdout"], index=False)
    manifest = {
        "labels_path": str(labels_path),
        "model_dirs": [str(path) for path in model_dirs],
        "output_dir": str(output_dir),
        "months": [str(v) for v in months],
        "fit_months": [str(v) for v in fit_months],
        "holdout_month": str(holdout_month),
        "top_fracs": [float(v) for v in top_fracs],
        "min_week_rows": int(min_week_rows),
        "oracle_rows": int(len(oracle)),
        "outputs": {key: str(value) for key, value in paths.items()},
    }
    markdown = _write_markdown(output_dir, comparison, oracle, model_frames, manifest)
    manifest["outputs"]["markdown"] = str(markdown)
    paths["manifest"].write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-path", type=Path, default=DEFAULT_LABELS_DIR)
    parser.add_argument("--model-dir", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--months", default="2026-04,2026-05,2026-06")
    parser.add_argument("--fit-months", default="2026-04,2026-05")
    parser.add_argument("--holdout-month", default="2026-06")
    parser.add_argument("--top-fracs", default="0.0025,0.005,0.01")
    parser.add_argument("--min-week-rows", type=int, default=3)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = run_report(
        labels_path=args.labels_path,
        model_dirs=args.model_dir,
        output_dir=args.output_dir,
        months=_parse_csv(args.months),
        fit_months=_parse_csv(args.fit_months),
        holdout_month=str(args.holdout_month),
        top_fracs=_parse_float_csv(args.top_fracs),
        min_week_rows=int(args.min_week_rows),
    )
    print(json.dumps(_json_safe(manifest), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
