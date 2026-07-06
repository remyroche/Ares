#!/usr/bin/env python3
"""Week-forward health guard replay for meta/regime OOS predictions.

This is an online diagnostic layered on top of already-generated OOS prediction
scores.  A week can only be skipped using outcomes from earlier completed
weeks, never from the week being scored.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_CONTEXT_DIR = Path(
    "data_perp/reports/ae_gmm_archetype_validation_status_20260704/"
    "meta_prefeature_regime_source_interaction_audit_v1/"
    "meta_regime_context_filter_oos_v1"
)
DEFAULT_OUTPUT_DIR = DEFAULT_CONTEXT_DIR / "week_health_guard_v1"
DEFAULT_SCORE_CONTAINS = (
    "contrast_path_rank50_clean25_prior25_side_volatility_shape_guard_bad50_timeout12_n20",
    "contrast_path_rank50_clean25_prior25_side_volatility_shape_guard_bad52_timeout12_n20",
    "contrast_path_rank50_clean25_prior25_side_source_liquidity_guard_bad52_timeout12_n20",
    "contrast_path_rank50_clean25_prior25_side_liquidity_guard_bad52_timeout12_n20",
    "contrast_path_rank50_clean25_prior25_side_source_liquidity_guard_bad55_timeout12_n20",
    "contrast_path_rank50_clean25_prior25_side_liquidity_guard_bad55_timeout12_n20",
    "contrast_path_rank50_clean25_prior25_side_market_dispersion_guard_bad55_timeout12_n20",
    "contrast_path_rank50_clean25_prior25_side_market_dispersion_modelrisk_bad40_timeout60",
    "contrast_path_rank50_clean25_prior25_side_market_dispersion_modelrisk_bad45_timeout65",
    "contrast_path_rank50_clean25_prior25_side_liquidity_modelrisk_bad45_timeout65",
    "contrast_path_rank50_clean25_prior25_side_source_liquidity_modelrisk_bad45_timeout65",
    "contrast_path_rank50_clean25_prior25_side_volatility_shape_modelrisk_bad40_timeout60",
    "contrast_path_rank50_clean25_prior25_side_volatility_shape_modelrisk_bad45_timeout65",
    "contrast_path_rank50_clean25_prior25_side_aegmm_entropy",
    "regime_candidate_codes_only_path_rank75_prior25_side_source_liquidity_guard_bad52_timeout12_n20",
    "regime_candidate_codes_only_path_rank75_prior25_side_liquidity_guard_bad52_timeout12_n20",
    "regime_candidate_codes_only_clean_exec",
    "prefit_scores_risk_only_path_rank75_prior25_side_aegmm_entropy_guard_bad50_timeout12_n20",
    "meta_prefeature_no_regime_candidates_blend_65_clean_35_contrast",
    "meta_prefeature_no_regime_candidates_blend_minus_bad50_timeout20",
)
TOP_FRACS = (0.10, 0.05)
GUARD_ARMS = (
    {
        "guard": "no_guard",
        "bad_mae_cap": 1.0,
        "timeout_cap": 1.0,
        "min_ev": -1.0,
        "min_prior_rows": 0,
        "rolling_weeks": 0,
    },
    {
        "guard": "rolling2_bad55_timeout12_evpos_n20",
        "bad_mae_cap": 0.55,
        "timeout_cap": 0.12,
        "min_ev": 0.0,
        "min_prior_rows": 20,
        "rolling_weeks": 2,
    },
    {
        "guard": "rolling2_bad50_timeout12_evpos_n20",
        "bad_mae_cap": 0.50,
        "timeout_cap": 0.12,
        "min_ev": 0.0,
        "min_prior_rows": 20,
        "rolling_weeks": 2,
    },
    {
        "guard": "expanding_bad55_timeout12_evpos_n30",
        "bad_mae_cap": 0.55,
        "timeout_cap": 0.12,
        "min_ev": 0.0,
        "min_prior_rows": 30,
        "rolling_weeks": 0,
    },
    {
        "guard": "expanding_bad50_timeout12_evpos_n30",
        "bad_mae_cap": 0.50,
        "timeout_cap": 0.12,
        "min_ev": 0.0,
        "min_prior_rows": 30,
        "rolling_weeks": 0,
    },
)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if pd.isna(value):
        return None
    return value


def _safe_num(values: Any) -> pd.Series:
    if isinstance(values, pd.Series):
        return pd.to_numeric(values, errors="coerce")
    return pd.to_numeric(pd.Series(values), errors="coerce")


def _safe_mean(values: Any) -> float:
    arr = _safe_num(values).replace([np.inf, -np.inf], np.nan).dropna()
    return float(arr.mean()) if len(arr) else float("nan")


def _rate(values: Any) -> float:
    arr = _safe_num(values).replace([np.inf, -np.inf], np.nan).dropna()
    return float(arr.clip(0.0, 1.0).mean()) if len(arr) else float("nan")


def _score_label(score_col: str) -> str:
    return str(score_col).removeprefix("score_")


def _select_top(frame: pd.DataFrame, score_col: str, top_frac: float) -> pd.DataFrame:
    valid = frame[_safe_num(frame[score_col]).notna()].copy()
    if valid.empty:
        return valid
    n = max(1, int(math.ceil(float(top_frac) * len(valid))))
    return valid.sort_values(score_col, ascending=False, kind="mergesort").head(n).copy()


def _metrics(frame: pd.DataFrame) -> dict[str, Any]:
    side = frame["side_name"].astype(str) if "side_name" in frame.columns else pd.Series(dtype=str)
    return {
        "selected_rows": int(len(frame)),
        "mean_ev": _safe_mean(frame["ev_after_cost"]) if "ev_after_cost" in frame.columns else float("nan"),
        "clean_precision": _rate(frame["clean_exec"]) if "clean_exec" in frame.columns else float("nan"),
        "bad_mae": _rate(frame["bad_mae"]) if "bad_mae" in frame.columns else float("nan"),
        "timeout": _rate(frame["timeout"]) if "timeout" in frame.columns else float("nan"),
        "dirty_positive": _rate(frame["dirty_positive"]) if "dirty_positive" in frame.columns else float("nan"),
        "mfe_before_mae_1r": _rate(frame["mfe_before_mae_1r"]) if "mfe_before_mae_1r" in frame.columns else float("nan"),
        "mae_before_mfe_1r": _rate(frame["mae_before_mfe_1r"]) if "mae_before_mfe_1r" in frame.columns else float("nan"),
        "long_share": float(side.eq("long").mean()) if len(side) else float("nan"),
        "short_share": float(side.eq("short").mean()) if len(side) else float("nan"),
    }


def _prior_window(history: list[pd.DataFrame], rolling_weeks: int) -> pd.DataFrame:
    if not history:
        return pd.DataFrame()
    if int(rolling_weeks) > 0:
        return pd.concat(history[-int(rolling_weeks) :], ignore_index=True)
    return pd.concat(history, ignore_index=True)


def _guard_pass(prior: pd.DataFrame, arm: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    min_prior_rows = int(arm["min_prior_rows"])
    if min_prior_rows <= 0:
        return True, {
            "prior_rows": int(len(prior)),
            "prior_ev": _safe_mean(prior["ev_after_cost"]) if len(prior) else float("nan"),
            "prior_bad_mae": _rate(prior["bad_mae"]) if len(prior) else float("nan"),
            "prior_timeout": _rate(prior["timeout"]) if len(prior) else float("nan"),
        }
    if len(prior) < min_prior_rows:
        return True, {
            "prior_rows": int(len(prior)),
            "prior_ev": _safe_mean(prior["ev_after_cost"]) if len(prior) else float("nan"),
            "prior_bad_mae": _rate(prior["bad_mae"]) if len(prior) else float("nan"),
            "prior_timeout": _rate(prior["timeout"]) if len(prior) else float("nan"),
            "prior_status": "insufficient_history_allow",
        }
    prior_ev = _safe_mean(prior["ev_after_cost"])
    prior_bad = _rate(prior["bad_mae"])
    prior_timeout = _rate(prior["timeout"])
    passed = (
        prior_ev >= float(arm["min_ev"])
        and prior_bad <= float(arm["bad_mae_cap"])
        and prior_timeout <= float(arm["timeout_cap"])
    )
    return passed, {
        "prior_rows": int(len(prior)),
        "prior_ev": prior_ev,
        "prior_bad_mae": prior_bad,
        "prior_timeout": prior_timeout,
        "prior_status": "pass" if passed else "block",
    }


def _choose_score_columns(predictions: pd.DataFrame, score_contains: tuple[str, ...]) -> list[str]:
    score_cols = [col for col in predictions.columns if str(col).startswith("score_")]
    if not score_contains:
        return score_cols
    selected: list[str] = []
    lowered = [token.lower() for token in score_contains]
    for col in score_cols:
        lower = str(col).lower()
        if any(token in lower for token in lowered):
            selected.append(str(col))
    return sorted(set(selected))


def run_report(
    *,
    context_dir: Path,
    output_dir: Path,
    score_contains: tuple[str, ...] = DEFAULT_SCORE_CONTAINS,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = context_dir / "meta_regime_context_filter_oos_predictions.parquet"
    predictions = pd.read_parquet(predictions_path)
    predictions["timestamp"] = pd.to_datetime(predictions["timestamp"], errors="coerce", utc=True)
    predictions = predictions[predictions["timestamp"].notna()].copy()
    timestamp_naive_utc = predictions["timestamp"].dt.tz_convert("UTC").dt.tz_localize(None)
    predictions["week_start"] = timestamp_naive_utc.dt.to_period("W-SUN").dt.start_time.astype(str)
    score_cols = _choose_score_columns(predictions, score_contains)
    if not score_cols:
        raise RuntimeError("No score columns matched the requested filters.")
    week_rows: list[dict[str, Any]] = []
    monthly_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    selected_frames: list[pd.DataFrame] = []

    for score_col in score_cols:
        score_name = _score_label(score_col)
        for top_frac in TOP_FRACS:
            for arm in GUARD_ARMS:
                history: list[pd.DataFrame] = []
                selected_all: list[pd.DataFrame] = []
                for week_start, week in predictions.sort_values("timestamp").groupby("week_start", sort=True):
                    prior = _prior_window(history, int(arm["rolling_weeks"]))
                    passed, prior_stats = _guard_pass(prior, arm)
                    selected = _select_top(week, score_col, top_frac) if passed else week.iloc[0:0].copy()
                    selected = selected.copy()
                    selected["score_col"] = score_col
                    selected["score_name"] = score_name
                    selected["top_frac"] = float(top_frac)
                    selected["guard"] = str(arm["guard"])
                    selected["week_start"] = str(week_start)
                    selected_all.append(selected)
                    # A deployable health guard only observes outcomes for weeks
                    # it actually traded.  Skipped weeks must not refill history
                    # with counterfactual selected rows.
                    history.append(selected)
                    row = {
                        "score_name": score_name,
                        "score_col": score_col,
                        "top_frac": float(top_frac),
                        "guard": str(arm["guard"]),
                        "week_start": str(week_start),
                        "month": str(week["month"].mode().iloc[0]) if "month" in week.columns and len(week) else "",
                        "week_rows": int(len(week)),
                        "week_scorable_rows": int(_safe_num(week[score_col]).notna().sum()),
                        "guard_pass": bool(passed),
                        **prior_stats,
                        **_metrics(selected),
                    }
                    week_rows.append(row)
                selected_cat = pd.concat(selected_all, ignore_index=True) if selected_all else pd.DataFrame()
                if len(selected_cat):
                    selected_frames.append(selected_cat)
                    for month, month_frame in selected_cat.groupby("month", dropna=False):
                        monthly_rows.append(
                            {
                                "score_name": score_name,
                                "score_col": score_col,
                                "top_frac": float(top_frac),
                                "guard": str(arm["guard"]),
                                "month": str(month),
                                **_metrics(month_frame),
                            }
                        )
                week_frame = pd.DataFrame([row for row in week_rows if row["score_col"] == score_col and row["top_frac"] == float(top_frac) and row["guard"] == str(arm["guard"])])
                month_frame = pd.DataFrame(
                    [
                        row
                        for row in monthly_rows
                        if row["score_col"] == score_col
                        and row["top_frac"] == float(top_frac)
                        and row["guard"] == str(arm["guard"])
                    ]
                )
                summary = {
                    "score_name": score_name,
                    "score_col": score_col,
                    "top_frac": float(top_frac),
                    "guard": str(arm["guard"]),
                    "weeks": int(len(week_frame)),
                    "traded_weeks": int(week_frame["guard_pass"].sum()) if len(week_frame) else 0,
                    "no_trade_week_rate": float((~week_frame["guard_pass"].astype(bool)).mean()) if len(week_frame) else float("nan"),
                    **_metrics(selected_cat),
                }
                if len(month_frame):
                    summary.update(
                        {
                            "months": int(month_frame["month"].nunique()),
                            "positive_months": int((_safe_num(month_frame["mean_ev"]) > 0.0).sum()),
                            "worst_month_ev": float(_safe_num(month_frame["mean_ev"]).min()),
                            "max_month_bad_mae": float(_safe_num(month_frame["bad_mae"]).max()),
                            "max_month_timeout": float(_safe_num(month_frame["timeout"]).max()),
                            "min_month_rows": int(_safe_num(month_frame["selected_rows"]).min()),
                        }
                    )
                summary["gate_status"] = (
                    "week_health_pass"
                    if summary.get("mean_ev", float("nan")) > 0.0
                    and summary.get("worst_month_ev", float("nan")) > 0.0
                    and summary.get("bad_mae", float("nan")) <= 0.50
                    and summary.get("max_month_bad_mae", float("nan")) <= 0.50
                    and summary.get("timeout", float("nan")) <= 0.12
                    and summary.get("max_month_timeout", float("nan")) <= 0.12
                    and summary.get("min_month_rows", 0) >= 10
                    else "fail_or_diagnostic"
                )
                summary_rows.append(summary)

    week_metrics = pd.DataFrame(week_rows)
    monthly_metrics = pd.DataFrame(monthly_rows)
    summary = pd.DataFrame(summary_rows).sort_values(
        ["gate_status", "bad_mae", "max_month_bad_mae", "mean_ev"],
        ascending=[True, True, True, False],
    )
    selected_ledger = pd.concat(selected_frames, ignore_index=True) if selected_frames else pd.DataFrame()
    outputs = {
        "week_metrics": output_dir / "week_health_guard_week_metrics.csv",
        "monthly_metrics": output_dir / "week_health_guard_monthly_metrics.csv",
        "summary": output_dir / "week_health_guard_summary.csv",
        "selected_ledger": output_dir / "week_health_guard_selected_ledger.parquet",
        "manifest": output_dir / "manifest.json",
        "report": output_dir / "week_health_guard.md",
    }
    week_metrics.to_csv(outputs["week_metrics"], index=False)
    monthly_metrics.to_csv(outputs["monthly_metrics"], index=False)
    summary.to_csv(outputs["summary"], index=False)
    selected_ledger.to_parquet(outputs["selected_ledger"], index=False)
    manifest = {
        "scope": "online_week_forward_health_guard",
        "context_dir": str(context_dir),
        "predictions_path": str(predictions_path),
        "output_dir": str(output_dir),
        "rows": int(len(predictions)),
        "score_columns": score_cols,
        "score_column_count": int(len(score_cols)),
        "top_fracs": list(TOP_FRACS),
        "guard_arms": list(GUARD_ARMS),
        "week_metric_rows": int(len(week_metrics)),
        "monthly_metric_rows": int(len(monthly_metrics)),
        "summary_rows": int(len(summary)),
        "pass_rows": int(summary["gate_status"].eq("week_health_pass").sum()) if len(summary) else 0,
        "leakage_contract": (
            "Online replay only: a week guard uses selected-row outcomes from earlier completed weeks for the same "
            "score/top-frac arm. It does not use current-week outcomes. This is not an untouched month-forward metric."
        ),
        "outputs": {key: str(path) for key, path in outputs.items()},
    }
    with open(outputs["manifest"], "w", encoding="utf-8") as fh:
        json.dump(_json_safe(manifest), fh, indent=2, sort_keys=True)
    display_cols = [
        "score_name",
        "top_frac",
        "guard",
        "selected_rows",
        "mean_ev",
        "worst_month_ev",
        "bad_mae",
        "max_month_bad_mae",
        "timeout",
        "max_month_timeout",
        "no_trade_week_rate",
        "gate_status",
    ]
    existing = [col for col in display_cols if col in summary.columns]
    lines = [
        "# Week Health Guard",
        "",
        manifest["leakage_contract"],
        "",
        "## Top Summary",
        "",
        summary[existing].head(25).to_markdown(index=False) if len(summary) else "_No rows_",
        "",
    ]
    outputs["report"].write_text("\n".join(lines), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context-dir", type=Path, default=DEFAULT_CONTEXT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--score-contains", type=str, default=",".join(DEFAULT_SCORE_CONTAINS))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    score_contains = tuple(part.strip() for part in str(args.score_contains).split(",") if part.strip())
    manifest = run_report(context_dir=args.context_dir, output_dir=args.output_dir, score_contains=score_contains)
    print(json.dumps(_json_safe(manifest), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
