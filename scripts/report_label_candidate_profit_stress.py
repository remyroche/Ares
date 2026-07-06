#!/usr/bin/env python3
"""Profit-after-cost stress report for locked label candidates.

This is a report-only diagnostic. It consumes an existing locked walk-forward
candidate comparison plus its underlying two-head model-smoke monthly/weekly
metrics. It does not train models, tune thresholds, or run policy geometry.
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

from scripts.report_label_candidate_locked_walkforward import (  # noqa: E402
    DEFAULT_OUTPUT_DIR as DEFAULT_LOCKED_DIR,
    DEFAULT_TWO_HEAD_DIR,
    _json_safe,
)
from scripts.run_label_quality_proxy_diagnostics import ROUND_TRIP_COST  # noqa: E402


DEFAULT_OUTPUT_DIR = Path("data_perp/reports/label_candidate_profit_stress_stage19_v1")
DEFAULT_EXTRA_COST_BPS = (0.0, 10.0, 25.0, 50.0)
DEFAULT_MATERIAL_WEEK_ROWS = 3


def _parse_float_csv(value: str | None, default: tuple[float, ...]) -> list[float]:
    if value is None or not str(value).strip():
        return list(default)
    return [float(part.strip()) for part in str(value).split(",") if part.strip()]


def _safe_numeric(values: Any) -> pd.Series:
    if isinstance(values, pd.Series):
        return pd.to_numeric(values, errors="coerce")
    return pd.to_numeric(pd.Series(values), errors="coerce")


def _finite_float(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def _safe_mean(values: Any) -> float:
    arr = _safe_numeric(values).dropna()
    return float(arr.mean()) if len(arr) else float("nan")


def _safe_quantile(values: Any, q: float) -> float:
    arr = _safe_numeric(values).dropna()
    return float(arr.quantile(q)) if len(arr) else float("nan")


def _normalize_config(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for col in ("source", "weight_arm", "score_rule", "period", "eval_month"):
        if col in out.columns:
            out[col] = out[col].astype(str)
    for col in ("bad_threshold", "top_k", "selected_rows", "mean_u", "mean_return_net"):
        if col in out.columns:
            out[col] = _safe_numeric(out[col])
    return out


def _config_filter(
    frame: pd.DataFrame,
    *,
    source: str,
    weight_arm: str,
    score_rule: str,
    bad_threshold: float,
    top_k: int,
) -> pd.Series:
    mask = (
        frame["source"].astype(str).eq(str(source))
        & frame["weight_arm"].astype(str).eq(str(weight_arm))
        & frame["score_rule"].astype(str).eq(str(score_rule))
        & np.isclose(_safe_numeric(frame["bad_threshold"]), float(bad_threshold), equal_nan=False)
        & _safe_numeric(frame["top_k"]).eq(int(top_k))
    )
    return pd.Series(mask, index=frame.index)


def _return_col(frame: pd.DataFrame) -> pd.Series:
    if "mean_return_net" in frame.columns:
        ret = _safe_numeric(frame["mean_return_net"])
        if ret.notna().any():
            return ret
    return _safe_numeric(frame["mean_u"])


def _with_profit_columns(
    frame: pd.DataFrame,
    *,
    extra_cost_bps: list[float],
    mean_return_col: str = "mean_return_net",
) -> pd.DataFrame:
    out = frame.copy()
    if mean_return_col in out.columns:
        ret = _safe_numeric(out[mean_return_col])
    else:
        ret = _return_col(out)
    rows = _safe_numeric(out["selected_rows"]).fillna(0.0)
    out["mean_return_net_used"] = ret
    out["sum_return_net"] = ret * rows
    out["positive_net"] = ret > 0.0
    for bps in extra_cost_bps:
        suffix = str(int(round(float(bps))))
        extra = float(bps) / 10000.0
        out[f"mean_return_net_extra{suffix}bps"] = ret - extra
        out[f"sum_return_net_extra{suffix}bps"] = (ret - extra) * rows
        out[f"positive_net_extra{suffix}bps"] = (ret - extra) > 0.0
    return out


def _candidate_month_rows(
    *,
    eval_comparison: pd.DataFrame,
    monthly: pd.DataFrame,
    source_baseline: pd.DataFrame,
    extra_cost_bps: list[float],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    eval_comparison = _normalize_config(eval_comparison)
    monthly = _normalize_config(monthly)
    source_baseline = _normalize_config(source_baseline)
    for _, cfg in eval_comparison.iterrows():
        label = str(cfg.get("label", ""))
        month = str(cfg.get("eval_month", ""))
        if label.startswith("source_all::"):
            subset = source_baseline[source_baseline["eval_month"].astype(str).eq(month)]
            if subset.empty:
                continue
            src = subset.iloc[0]
            source_mean_u = _finite_float(src.get("mean_u"))
            rows.append(
                {
                    "label": label,
                    "eval_month": month,
                    "fit_months": cfg.get("fit_months", ""),
                    "selection_basis": cfg.get("selection_basis", ""),
                    "source": src.get("source", cfg.get("source", "")),
                    "weight_arm": cfg.get("weight_arm", ""),
                    "score_rule": cfg.get("score_rule", ""),
                    "bad_threshold": cfg.get("bad_threshold", np.nan),
                    "top_k": cfg.get("top_k", np.nan),
                    "selected_rows": int(src.get("selected_rows", 0) or 0),
                    "mean_u": source_mean_u,
                    "mean_return_net": source_mean_u - float(ROUND_TRIP_COST),
                    "hit_u": _finite_float(src.get("hit_u")),
                    "q10_u": _finite_float(src.get("q10_u")),
                    "first_touch_bad_mae_to_sl_rate": _finite_float(src.get("first_touch_bad_mae_to_sl_rate")),
                    "p90_first_touch_mae_to_sl": _finite_float(src.get("p90_first_touch_mae_to_sl")),
                    "clean_exec_actual_rate": _finite_float(src.get("clean_exec_actual_rate")),
                    "first_touch_timeout_rate": _finite_float(src.get("first_touch_timeout_rate")),
                    "candidate_timestamp_coverage": 1.0,
                    "holdout_bounded_pass": np.nan,
                }
            )
            continue
        source = str(cfg.get("source", ""))
        weight_arm = str(cfg.get("weight_arm", ""))
        score_rule = str(cfg.get("score_rule", ""))
        bad_threshold = _finite_float(cfg.get("bad_threshold"))
        top_k = int(_finite_float(cfg.get("top_k")))
        mask = (
            monthly["period"].astype(str).eq(month)
            & _config_filter(
                monthly,
                source=source,
                weight_arm=weight_arm,
                score_rule=score_rule,
                bad_threshold=bad_threshold,
                top_k=top_k,
            )
        )
        subset = monthly[mask].copy()
        if subset.empty:
            continue
        row = subset.sort_values(["selected_rows", "mean_u"], ascending=[False, False]).iloc[0]
        out = row.to_dict()
        out.update(
            {
                "label": label,
                "eval_month": month,
                "fit_months": cfg.get("fit_months", ""),
                "selection_basis": cfg.get("selection_basis", ""),
                "holdout_bounded_pass": cfg.get("holdout_bounded_pass", np.nan),
            }
        )
        rows.append(out)
    out_frame = pd.DataFrame(rows)
    if out_frame.empty:
        return out_frame
    out_frame = _with_profit_columns(out_frame, extra_cost_bps=extra_cost_bps)
    return out_frame.sort_values(["eval_month", "label"]).reset_index(drop=True)


def _candidate_week_rows(
    *,
    month_rows: pd.DataFrame,
    weekly: pd.DataFrame,
    extra_cost_bps: list[float],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    weekly = _normalize_config(weekly)
    for _, cfg in month_rows.iterrows():
        label = str(cfg.get("label", ""))
        if label.startswith("source_all::"):
            continue
        source = str(cfg.get("source", ""))
        weight_arm = str(cfg.get("weight_arm", ""))
        score_rule = str(cfg.get("score_rule", ""))
        bad_threshold = _finite_float(cfg.get("bad_threshold"))
        top_k = int(_finite_float(cfg.get("top_k")))
        month = str(cfg.get("eval_month", cfg.get("period", "")))
        mask = (
            weekly["period"].astype(str).eq(month)
            & _config_filter(
                weekly,
                source=source,
                weight_arm=weight_arm,
                score_rule=score_rule,
                bad_threshold=bad_threshold,
                top_k=top_k,
            )
        )
        subset = weekly[mask].copy()
        if subset.empty:
            continue
        subset["label"] = label
        subset["eval_month"] = month
        subset["fit_months"] = cfg.get("fit_months", "")
        subset["selection_basis"] = cfg.get("selection_basis", "")
        rows.extend(subset.to_dict("records"))
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = _with_profit_columns(out, extra_cost_bps=extra_cost_bps)
    return out.sort_values(["eval_month", "label", "week"]).reset_index(drop=True)


def _aggregate_rows(
    *,
    month_rows: pd.DataFrame,
    week_rows: pd.DataFrame,
    extra_cost_bps: list[float],
    material_week_rows: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    groups = month_rows.groupby("label", dropna=False, observed=True)
    for label, group in groups:
        weeks = week_rows[week_rows["label"].astype(str).eq(str(label))].copy()
        material = weeks[_safe_numeric(weeks.get("selected_rows", pd.Series(dtype=float))) >= int(material_week_rows)].copy()
        selected_rows = _safe_numeric(group["selected_rows"]).fillna(0.0)
        total_rows = float(selected_rows.sum())
        sum_return = _safe_numeric(group["sum_return_net"]).fillna(0.0)
        base: dict[str, Any] = {
            "label": label,
            "months": int(group["eval_month"].nunique()),
            "selected_rows": int(total_rows),
            "sum_return_net": float(sum_return.sum()),
            "mean_return_net_weighted": float(sum_return.sum() / total_rows) if total_rows > 0 else float("nan"),
            "positive_months": int((_safe_numeric(group["mean_return_net_used"]) > 0.0).sum()),
            "worst_month_mean_return_net": _safe_quantile(group["mean_return_net_used"], 0.0),
            "q25_month_mean_return_net": _safe_quantile(group["mean_return_net_used"], 0.25),
            "material_weeks": int(len(material)),
            "positive_material_weeks": int((_safe_numeric(material.get("mean_return_net_used", pd.Series(dtype=float))) > 0.0).sum()),
            "positive_material_week_rate": _safe_mean(
                _safe_numeric(material.get("mean_return_net_used", pd.Series(dtype=float))) > 0.0
            ),
            "q25_material_week_mean_return_net": _safe_quantile(
                material.get("mean_return_net_used", pd.Series(dtype=float)), 0.25
            ),
            "worst_material_week_mean_return_net": _safe_quantile(
                material.get("mean_return_net_used", pd.Series(dtype=float)), 0.0
            ),
            "mean_bad_mae": _safe_mean(group.get("first_touch_bad_mae_to_sl_rate", pd.Series(dtype=float))),
            "mean_p90_mae": _safe_mean(group.get("p90_first_touch_mae_to_sl", pd.Series(dtype=float))),
            "mean_clean_exec": _safe_mean(group.get("clean_exec_actual_rate", pd.Series(dtype=float))),
        }
        for bps in extra_cost_bps:
            suffix = str(int(round(float(bps))))
            stress_sum = _safe_numeric(group[f"sum_return_net_extra{suffix}bps"]).fillna(0.0)
            stress_mean = _safe_numeric(group[f"mean_return_net_extra{suffix}bps"])
            base[f"sum_return_net_extra{suffix}bps"] = float(stress_sum.sum())
            base[f"mean_return_net_weighted_extra{suffix}bps"] = (
                float(stress_sum.sum() / total_rows) if total_rows > 0 else float("nan")
            )
            base[f"positive_months_extra{suffix}bps"] = int((stress_mean > 0.0).sum())
            if not material.empty:
                material_stress = _safe_numeric(material[f"mean_return_net_extra{suffix}bps"])
                base[f"positive_material_week_rate_extra{suffix}bps"] = _safe_mean(material_stress > 0.0)
                base[f"q25_material_week_mean_return_net_extra{suffix}bps"] = _safe_quantile(material_stress, 0.25)
            else:
                base[f"positive_material_week_rate_extra{suffix}bps"] = float("nan")
                base[f"q25_material_week_mean_return_net_extra{suffix}bps"] = float("nan")
        rows.append(base)
    return pd.DataFrame(rows).sort_values(["sum_return_net", "mean_return_net_weighted"], ascending=[False, False])


def _format_table(frame: pd.DataFrame, cols: list[str], limit: int = 80) -> str:
    if frame.empty:
        return "No rows."
    view = frame[[col for col in cols if col in frame.columns]].head(limit).copy()
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda value: f"{float(value):.4f}" if pd.notna(value) else "")
    return view.to_markdown(index=False)


def _write_markdown(
    *,
    output_dir: Path,
    manifest: dict[str, Any],
    aggregate: pd.DataFrame,
    monthly: pd.DataFrame,
    weekly: pd.DataFrame,
) -> Path:
    path = output_dir / "label_candidate_profit_stress.md"
    lines = [
        "# Label Candidate Profit Stress",
        "",
        "Scope: report-only profitability-after-costs stress over locked candidate selections. No training, Optuna, or policy geometry optimisation is run.",
        "",
        f"Locked comparison: `{manifest['locked_dir']}`",
        f"Two-head smoke: `{manifest['two_head_dir']}`",
        f"Extra cost bps: `{', '.join(str(v) for v in manifest['extra_cost_bps'])}`",
        "",
        f"`mean_return_net_used` is the Stage17 net-return column where present. For source baselines it is `mean_u - {float(ROUND_TRIP_COST):.4f}`. Extra-cost columns subtract additional round-trip bps per selected row on top of that embedded cost.",
        "",
        "## Aggregate",
        "",
        _format_table(
            aggregate,
            [
                "label",
                "months",
                "selected_rows",
                "sum_return_net",
                "mean_return_net_weighted",
                "positive_months",
                "material_weeks",
                "positive_material_week_rate",
                "q25_material_week_mean_return_net",
                "sum_return_net_extra25bps",
                "sum_return_net_extra50bps",
                "mean_bad_mae",
                "mean_p90_mae",
                "mean_clean_exec",
            ],
            limit=120,
        ),
        "",
        "## Monthly",
        "",
        _format_table(
            monthly,
            [
                "label",
                "eval_month",
                "fit_months",
                "weight_arm",
                "score_rule",
                "bad_threshold",
                "top_k",
                "selected_rows",
                "mean_return_net_used",
                "sum_return_net",
                "sum_return_net_extra25bps",
                "first_touch_bad_mae_to_sl_rate",
                "p90_first_touch_mae_to_sl",
                "clean_exec_actual_rate",
            ],
            limit=160,
        ),
        "",
        "## Weekly Sample",
        "",
        _format_table(
            weekly.sort_values(["eval_month", "label", "week"]),
            [
                "label",
                "eval_month",
                "week",
                "selected_rows",
                "mean_return_net_used",
                "sum_return_net",
                "sum_return_net_extra25bps",
                "first_touch_bad_mae_to_sl_rate",
                "p90_first_touch_mae_to_sl",
                "clean_exec_actual_rate",
            ],
            limit=120,
        ),
        "",
        "## Outputs",
        "",
        f"- Aggregate: `{manifest['outputs']['aggregate']}`",
        f"- Monthly: `{manifest['outputs']['monthly']}`",
        f"- Weekly: `{manifest['outputs']['weekly']}`",
        f"- Manifest: `{manifest['outputs']['manifest']}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_report(
    *,
    locked_dir: Path,
    two_head_dir: Path,
    output_dir: Path,
    extra_cost_bps: list[float],
    material_week_rows: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    eval_comparison = pd.read_csv(locked_dir / "eval_comparison.csv")
    source_baseline = pd.read_csv(locked_dir / "source_baseline_monthly.csv")
    monthly_src = pd.read_csv(two_head_dir / "source_conditioned_two_head_model_monthly.csv")
    weekly_src = pd.read_csv(two_head_dir / "source_conditioned_two_head_model_weekly.csv")
    monthly = _candidate_month_rows(
        eval_comparison=eval_comparison,
        monthly=monthly_src,
        source_baseline=source_baseline,
        extra_cost_bps=extra_cost_bps,
    )
    weekly = _candidate_week_rows(
        month_rows=monthly,
        weekly=weekly_src,
        extra_cost_bps=extra_cost_bps,
    )
    aggregate = _aggregate_rows(
        month_rows=monthly,
        week_rows=weekly,
        extra_cost_bps=extra_cost_bps,
        material_week_rows=material_week_rows,
    )
    paths = {
        "aggregate": output_dir / "profit_stress_aggregate.csv",
        "monthly": output_dir / "profit_stress_monthly.csv",
        "weekly": output_dir / "profit_stress_weekly.csv",
        "manifest": output_dir / "manifest.json",
    }
    aggregate.to_csv(paths["aggregate"], index=False)
    monthly.to_csv(paths["monthly"], index=False)
    weekly.to_csv(paths["weekly"], index=False)
    manifest = {
        "locked_dir": str(locked_dir),
        "two_head_dir": str(two_head_dir),
        "output_dir": str(output_dir),
        "extra_cost_bps": extra_cost_bps,
        "embedded_round_trip_cost": float(ROUND_TRIP_COST),
        "material_week_rows": int(material_week_rows),
        "rows": {
            "aggregate": int(len(aggregate)),
            "monthly": int(len(monthly)),
            "weekly": int(len(weekly)),
        },
        "outputs": {key: str(value) for key, value in paths.items()},
    }
    markdown = _write_markdown(
        output_dir=output_dir,
        manifest=manifest,
        aggregate=aggregate,
        monthly=monthly,
        weekly=weekly,
    )
    manifest["outputs"]["markdown"] = str(markdown)
    paths["manifest"].write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--locked-dir", type=Path, default=DEFAULT_LOCKED_DIR)
    parser.add_argument("--two-head-dir", type=Path, default=DEFAULT_TWO_HEAD_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--extra-cost-bps", default=",".join(str(v) for v in DEFAULT_EXTRA_COST_BPS))
    parser.add_argument("--material-week-rows", type=int, default=DEFAULT_MATERIAL_WEEK_ROWS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = run_report(
        locked_dir=args.locked_dir,
        two_head_dir=args.two_head_dir,
        output_dir=args.output_dir,
        extra_cost_bps=_parse_float_csv(args.extra_cost_bps, DEFAULT_EXTRA_COST_BPS),
        material_week_rows=int(args.material_week_rows),
    )
    print(json.dumps(_json_safe({k: manifest[k] for k in ("output_dir", "rows", "outputs")}), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
