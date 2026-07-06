#!/usr/bin/env python3
"""Fit/holdout selector for first-touch two-head label recipes.

This is a report-only guard against post-selection optimism. It consumes
existing month-forward smoke outputs, ranks recipes on fit months only, and
then reports holdout months without using them for selection.
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

from scripts.run_label_quality_proxy_diagnostics import _json_safe, _safe_mean, _safe_quantile  # noqa: E402


DEFAULT_REPORT_DIRS = (
    Path("data_perp/reports/first_touch_two_head_training_smoke_stage131_full_v1"),
    Path("data_perp/reports/first_touch_two_head_training_smoke_stage132_support_label_grid_v1"),
    Path("data_perp/reports/first_touch_two_head_training_smoke_stage133_time_decay_utility_v1"),
)
DEFAULT_OUTPUT_DIR = Path("data_perp/reports/first_touch_two_head_recipe_fit_holdout_stage134_v1")
KEY_COLS = [
    "source",
    "utility_target_mode",
    "support_target_mode",
    "utility_weight_arm",
    "support_weight_arm",
    "score_rule",
    "support_gate_frac",
    "top_frac",
]
METRIC_COLS = [
    "mean_first_touch_net",
    "hit_first_touch_net",
    "q10_first_touch_net",
    "clean_first_touch_exec_rate",
    "first_touch_timeout_rate",
    "first_touch_bad_mae_to_sl_rate",
    "p90_first_touch_mae_to_sl",
    "p90_full_path_mae_to_sl",
    "support_ic_clean_exec",
    "utility_ic_first_touch_net",
    "score_ic_first_touch_net",
    "selected_rows",
]


def _parse_path_csv(value: str | None, default: tuple[Path, ...]) -> list[Path]:
    if value is None or not str(value).strip():
        return list(default)
    return [Path(part.strip()) for part in str(value).split(",") if part.strip()]


def _parse_csv(value: str | None, default: tuple[str, ...]) -> list[str]:
    if value is None or not str(value).strip():
        return list(default)
    return [part.strip() for part in str(value).split(",") if part.strip()]


def _safe_numeric(values: Any) -> pd.Series:
    if isinstance(values, pd.Series):
        return pd.to_numeric(values, errors="coerce")
    return pd.to_numeric(pd.Series(values), errors="coerce")


def _source_name(report_dir: Path) -> str:
    name = report_dir.name
    for prefix in ("first_touch_two_head_training_smoke_",):
        if name.startswith(prefix):
            return name[len(prefix) :]
    return name


def _load_monthly(report_dirs: list[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for report_dir in report_dirs:
        path = report_dir / "first_touch_two_head_training_smoke_monthly.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path)
        frame.insert(0, "source", _source_name(report_dir))
        frame.insert(1, "report_dir", str(report_dir))
        frames.append(frame)
    if not frames:
        raise ValueError("No report directories supplied")
    out = pd.concat(frames, ignore_index=True)
    missing = sorted(set(KEY_COLS + ["period"] + METRIC_COLS) - set(out.columns))
    if missing:
        raise ValueError(f"Missing columns in monthly inputs: {missing}")
    out["period"] = out["period"].astype(str)
    return out


def _period_aggregate(group: pd.DataFrame, *, prefix: str) -> dict[str, Any]:
    rows: dict[str, Any] = {
        f"{prefix}_months": int(group["period"].nunique()) if not group.empty else 0,
        f"{prefix}_periods": ",".join(sorted(group["period"].astype(str).unique())) if not group.empty else "",
    }
    if group.empty:
        for col in METRIC_COLS:
            rows[f"{prefix}_{col}"] = float("nan")
        rows[f"{prefix}_positive_first_touch_months"] = 0
        rows[f"{prefix}_worst_month_first_touch_net"] = float("nan")
        return rows

    weights = _safe_numeric(group["selected_rows"]).fillna(0.0).clip(lower=0.0)
    use_weighted = bool(float(weights.sum()) > 0.0)
    for col in METRIC_COLS:
        values = _safe_numeric(group[col])
        if col == "selected_rows":
            rows[f"{prefix}_{col}"] = _safe_mean(values)
        elif use_weighted:
            valid = values.notna() & weights.gt(0.0)
            rows[f"{prefix}_{col}"] = (
                float(np.average(values.loc[valid], weights=weights.loc[valid]))
                if bool(valid.any())
                else float("nan")
            )
        else:
            rows[f"{prefix}_{col}"] = _safe_mean(values)
    mean_ft = _safe_numeric(group["mean_first_touch_net"])
    rows[f"{prefix}_positive_first_touch_months"] = int((mean_ft > 0.0).sum())
    rows[f"{prefix}_worst_month_first_touch_net"] = _safe_quantile(mean_ft, 0.0)
    return rows


def _fit_score(row: pd.Series) -> float:
    mean_net = float(row.get("fit_mean_first_touch_net", float("nan")))
    worst_net = float(row.get("fit_worst_month_first_touch_net", float("nan")))
    hit = float(row.get("fit_hit_first_touch_net", float("nan")))
    clean = float(row.get("fit_clean_first_touch_exec_rate", float("nan")))
    timeout = float(row.get("fit_first_touch_timeout_rate", float("nan")))
    bad = float(row.get("fit_first_touch_bad_mae_to_sl_rate", float("nan")))
    p90 = float(row.get("fit_p90_first_touch_mae_to_sl", float("nan")))
    support_ic = float(row.get("fit_support_ic_clean_exec", float("nan")))
    if not math.isfinite(mean_net) or not math.isfinite(worst_net):
        return float("-inf")
    return (
        1.00 * mean_net
        + 0.75 * worst_net
        + 0.0030 * (hit if math.isfinite(hit) else 0.0)
        + 0.0025 * (clean if math.isfinite(clean) else 0.0)
        + 0.0010 * (support_ic if math.isfinite(support_ic) else 0.0)
        - 0.0060 * (timeout if math.isfinite(timeout) else 1.0)
        - 0.0040 * (bad if math.isfinite(bad) else 1.0)
        - 0.0006 * max(0.0, (p90 if math.isfinite(p90) else 10.0) - 1.0)
    )


def _fit_pass(
    row: pd.Series,
    *,
    min_clean: float,
    max_timeout: float,
    max_bad_mae: float,
    max_p90_mae: float,
    min_worst_net: float,
    min_fit_months: int,
) -> bool:
    return bool(
        int(row.get("fit_months", 0) or 0) >= int(min_fit_months)
        and int(row.get("fit_positive_first_touch_months", 0) or 0) >= int(min_fit_months)
        and float(row.get("fit_worst_month_first_touch_net", float("-inf"))) >= float(min_worst_net)
        and float(row.get("fit_clean_first_touch_exec_rate", float("-inf"))) >= float(min_clean)
        and float(row.get("fit_first_touch_timeout_rate", float("inf"))) <= float(max_timeout)
        and float(row.get("fit_first_touch_bad_mae_to_sl_rate", float("inf"))) <= float(max_bad_mae)
        and float(row.get("fit_p90_first_touch_mae_to_sl", float("inf"))) <= float(max_p90_mae)
    )


def _holdout_pass(row: pd.Series) -> bool:
    holdout_months = int(row.get("holdout_months", 0) or 0)
    return bool(
        holdout_months > 0
        and int(row.get("holdout_positive_first_touch_months", 0) or 0) >= holdout_months
        and float(row.get("holdout_worst_month_first_touch_net", float("-inf"))) > 0.0
        and float(row.get("holdout_clean_first_touch_exec_rate", float("-inf"))) >= 0.80
        and float(row.get("holdout_first_touch_timeout_rate", float("inf"))) <= 0.05
        and float(row.get("holdout_first_touch_bad_mae_to_sl_rate", float("inf"))) <= 0.15
    )


def _fit_holdout(
    monthly: pd.DataFrame,
    *,
    fit_months: list[str],
    holdout_months: list[str],
    min_clean: float,
    max_timeout: float,
    max_bad_mae: float,
    max_p90_mae: float,
    min_worst_net: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    fit_set = set(fit_months)
    holdout_set = set(holdout_months)
    for key, group in monthly.groupby(KEY_COLS, observed=True, dropna=False):
        row = {col: value for col, value in zip(KEY_COLS, key, strict=True)}
        fit = group[group["period"].isin(fit_set)].copy()
        holdout = group[group["period"].isin(holdout_set)].copy()
        row.update(_period_aggregate(fit, prefix="fit"))
        row.update(_period_aggregate(holdout, prefix="holdout"))
        series = pd.Series(row)
        row["fit_score"] = _fit_score(series)
        row["fit_pass_strict"] = _fit_pass(
            series,
            min_clean=min_clean,
            max_timeout=max_timeout,
            max_bad_mae=max_bad_mae,
            max_p90_mae=max_p90_mae,
            min_worst_net=min_worst_net,
            min_fit_months=len(fit_months),
        )
        row["holdout_pass_execution"] = _holdout_pass(series)
        rows.append(row)
    out = pd.DataFrame(rows)
    return out.sort_values(
        ["fit_pass_strict", "fit_score", "fit_worst_month_first_touch_net"],
        ascending=[False, False, False],
    )


def _table(frame: pd.DataFrame, cols: list[str], limit: int | None = None) -> str:
    if frame.empty:
        return "No rows."
    view = frame[[col for col in cols if col in frame.columns]].copy()
    if limit is not None:
        view = view.head(limit)
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda v: f"{float(v):.4f}" if pd.notna(v) else "")
    return view.to_markdown(index=False)


def _write_markdown(
    *,
    path: Path,
    fit_holdout: pd.DataFrame,
    selected: pd.DataFrame,
    fit_months: list[str],
    holdout_months: list[str],
    outputs: dict[str, Path],
) -> None:
    cols = [
        "source",
        "utility_target_mode",
        "support_target_mode",
        "utility_weight_arm",
        "support_weight_arm",
        "score_rule",
        "support_gate_frac",
        "top_frac",
        "fit_pass_strict",
        "holdout_pass_execution",
        "fit_score",
        "fit_mean_first_touch_net",
        "fit_worst_month_first_touch_net",
        "fit_clean_first_touch_exec_rate",
        "fit_first_touch_timeout_rate",
        "fit_first_touch_bad_mae_to_sl_rate",
        "fit_p90_first_touch_mae_to_sl",
        "holdout_mean_first_touch_net",
        "holdout_clean_first_touch_exec_rate",
        "holdout_first_touch_timeout_rate",
        "holdout_first_touch_bad_mae_to_sl_rate",
        "holdout_p90_first_touch_mae_to_sl",
    ]
    passing = fit_holdout[fit_holdout["fit_pass_strict"].astype(bool)].copy()
    holdout_passing = passing[passing["holdout_pass_execution"].astype(bool)].copy()
    lines = [
        "# First-Touch Two-Head Recipe Fit/Holdout Selector",
        "",
        f"Fit months: `{','.join(fit_months)}`",
        f"Holdout months: `{','.join(holdout_months)}`",
        "",
        "Scope: report-only selector using existing month-forward smoke outputs. Holdout months are not used for ranking.",
        "",
        "## Selected By Fit",
        "",
        _table(selected, cols, limit=20),
        "",
        "## Fit-Passing And Holdout-Passing",
        "",
        _table(holdout_passing.sort_values(["holdout_mean_first_touch_net", "fit_score"], ascending=[False, False]), cols, limit=30),
        "",
        "## Top Fit-Passing Rows",
        "",
        _table(passing, cols, limit=40),
        "",
        "## Outputs",
        "",
        f"- Fit/Holdout: `{outputs['fit_holdout']}`",
        f"- Selected: `{outputs['selected']}`",
        f"- Manifest: `{outputs['manifest']}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_selector(
    *,
    report_dirs: list[Path],
    output_dir: Path,
    fit_months: list[str],
    holdout_months: list[str],
    min_clean: float,
    max_timeout: float,
    max_bad_mae: float,
    max_p90_mae: float,
    min_worst_net: float,
    select_top_n: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    monthly = _load_monthly(report_dirs)
    fit_holdout = _fit_holdout(
        monthly,
        fit_months=fit_months,
        holdout_months=holdout_months,
        min_clean=min_clean,
        max_timeout=max_timeout,
        max_bad_mae=max_bad_mae,
        max_p90_mae=max_p90_mae,
        min_worst_net=min_worst_net,
    )
    selected_pool = fit_holdout[fit_holdout["fit_pass_strict"].astype(bool)].copy()
    if selected_pool.empty:
        selected_pool = fit_holdout.copy()
    selected = selected_pool.head(int(select_top_n)).copy()
    outputs = {
        "fit_holdout": output_dir / "first_touch_two_head_recipe_fit_holdout.csv",
        "selected": output_dir / "first_touch_two_head_recipe_selected_by_fit.csv",
        "manifest": output_dir / "manifest.json",
        "markdown": output_dir / "first_touch_two_head_recipe_fit_holdout.md",
    }
    fit_holdout.to_csv(outputs["fit_holdout"], index=False)
    selected.to_csv(outputs["selected"], index=False)
    manifest = {
        "scope": "report_only_first_touch_two_head_recipe_fit_holdout_selector",
        "report_dirs": [str(path) for path in report_dirs],
        "output_dir": str(output_dir),
        "fit_months": list(fit_months),
        "holdout_months": list(holdout_months),
        "rows": int(len(fit_holdout)),
        "fit_pass_strict_rows": int(fit_holdout["fit_pass_strict"].astype(bool).sum()),
        "fit_and_holdout_pass_rows": int(
            (fit_holdout["fit_pass_strict"].astype(bool) & fit_holdout["holdout_pass_execution"].astype(bool)).sum()
        ),
        "selection_constraints": {
            "min_clean": float(min_clean),
            "max_timeout": float(max_timeout),
            "max_bad_mae": float(max_bad_mae),
            "max_p90_mae": float(max_p90_mae),
            "min_worst_net": float(min_worst_net),
        },
        "outputs": {key: str(value) for key, value in outputs.items()},
    }
    _write_markdown(
        path=outputs["markdown"],
        fit_holdout=fit_holdout,
        selected=selected,
        fit_months=fit_months,
        holdout_months=holdout_months,
        outputs=outputs,
    )
    outputs["manifest"].write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dirs", default=",".join(str(path) for path in DEFAULT_REPORT_DIRS))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--fit-months", default="2026-04,2026-05")
    parser.add_argument("--holdout-months", default="2026-06")
    parser.add_argument("--min-clean", type=float, default=0.80)
    parser.add_argument("--max-timeout", type=float, default=0.05)
    parser.add_argument("--max-bad-mae", type=float, default=0.15)
    parser.add_argument("--max-p90-mae", type=float, default=1.50)
    parser.add_argument("--min-worst-net", type=float, default=0.0)
    parser.add_argument("--select-top-n", type=int, default=20)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = run_selector(
        report_dirs=_parse_path_csv(args.report_dirs, DEFAULT_REPORT_DIRS),
        output_dir=args.output_dir,
        fit_months=_parse_csv(args.fit_months, ("2026-04", "2026-05")),
        holdout_months=_parse_csv(args.holdout_months, ("2026-06",)),
        min_clean=float(args.min_clean),
        max_timeout=float(args.max_timeout),
        max_bad_mae=float(args.max_bad_mae),
        max_p90_mae=float(args.max_p90_mae),
        min_worst_net=float(args.min_worst_net),
        select_top_n=int(args.select_top_n),
    )
    print(json.dumps(_json_safe(manifest), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
