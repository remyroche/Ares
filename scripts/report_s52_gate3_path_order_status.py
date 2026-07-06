#!/usr/bin/env python3
"""Gate 3 status report for S52 path-ordered base ranker artifacts."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError


DEFAULT_INPUT_DIR = Path(
    "data_perp/reports/ae_gmm_archetype_validation_status_20260704/"
    "s52_ranker_smoke_best_archetype_overlay_v1"
)
DEFAULT_OUTPUT_DIR = DEFAULT_INPUT_DIR / "gate3_status"
DEFAULT_LGBM_PIPELINE = Path("extreme_price_movements/lgbm_pipeline.py")


THRESHOLDS: dict[str, float] = {
    "top10_ev_weighted_first_touch_precision_min": 0.70,
    "top20_ev_weighted_first_touch_precision_min": 0.65,
    "top30_ev_weighted_first_touch_precision_min": 0.60,
    "top10_first_touch_bad_mae_1r_rate_max": 0.25,
    "top10_timeout_rate_max": 0.12,
    "top10_mean_first_touch_mae_norm_max": 1.50,
    "top10_p90_first_touch_mae_norm_max": 3.00,
    "top10_mean_max_adverse_before_mfe_1r_max": 1.50,
    "top10_mean_underwater_fraction_before_mfe_max": 0.45,
    "top10_mean_underwater_bars_before_mfe_max": 10.0,
    "top10_post_exit_full_path_bad_mae_1r_rate_warn": 0.50,
    "top10_p90_post_exit_full_path_mae_norm_warn": 3.00,
    "top10_mfe_1r_before_mae_1r_rate_min": 0.55,
    "top10_mae_1r_before_mfe_1r_rate_max": 0.35,
    "top10_mean_u_min": 0.0,
    "top10_full_horizon_bad_mae_rate_max": 0.50,
    "long_top10_mae_1r_before_mfe_1r_rate_max": 0.35,
    "short_top10_mae_1r_before_mfe_1r_rate_max": 0.35,
}


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
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


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size <= 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except EmptyDataError:
        return pd.DataFrame()


def _metric(row: pd.Series, name: str) -> float:
    value = pd.to_numeric(pd.Series([row.get(name)]), errors="coerce").iloc[0]
    return float(value) if pd.notna(value) and math.isfinite(float(value)) else float("nan")


def _check(
    rows: list[dict[str, Any]],
    *,
    name: str,
    value: float,
    threshold: float,
    op: str,
    severity: str = "fail",
    scope: str = "aggregate",
) -> None:
    if not math.isfinite(value):
        status = "missing" if severity == "fail" else f"missing_{severity}"
    elif op == ">=":
        status = "pass" if value >= threshold else severity
    elif op == "<=":
        status = "pass" if value <= threshold else severity
    else:
        raise ValueError(f"unknown operator: {op}")
    rows.append(
        {
            "scope": scope,
            "metric": name,
            "value": value,
            "threshold": threshold,
            "operator": op,
            "status": status,
        }
    )


def _status_from_checks(checks: pd.DataFrame) -> str:
    if checks.empty:
        return "missing"
    statuses = set(checks["status"].astype(str))
    if "missing" in statuses:
        return "fail_missing_evidence"
    if "fail" in statuses:
        return "fail"
    if "warn" in statuses:
        return "conditional_pass_with_warnings"
    return "pass"


def _table(frame: pd.DataFrame, cols: list[str], *, limit: int = 20) -> str:
    if frame.empty:
        return "No rows."
    view = frame[[c for c in cols if c in frame.columns]].head(int(limit)).copy()
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda v: f"{float(v):.4f}" if pd.notna(v) else "")
    return view.to_markdown(index=False)


def _production_ranker_check(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "status": "missing",
            "path": str(path),
            "has_lgbm_ranker": False,
            "has_lambdarank": False,
            "reason": "lgbm_pipeline_not_found",
        }
    text = path.read_text(encoding="utf-8", errors="ignore")
    has_lgbm_ranker = "LGBMRanker" in text
    has_lambdarank = "lambdarank" in text
    status = "pass" if has_lgbm_ranker and has_lambdarank else "blocked"
    reason = "native_ranker_path_detected" if status == "pass" else "native_timestamp_side_ranker_not_materialized"
    return {
        "status": status,
        "path": str(path),
        "has_lgbm_ranker": bool(has_lgbm_ranker),
        "has_lambdarank": bool(has_lambdarank),
        "reason": reason,
    }


def build_report(
    *,
    input_dir: Path,
    output_dir: Path,
    thresholds: dict[str, float] | None = None,
    lgbm_pipeline_path: Path = DEFAULT_LGBM_PIPELINE,
) -> dict[str, Any]:
    thresholds = dict(THRESHOLDS if thresholds is None else thresholds)
    summary = _read_csv(input_dir / "s52_ranker_smoke_summary.csv")
    folds = _read_csv(input_dir / "s52_ranker_smoke_folds.csv")
    archetypes = _read_csv(input_dir / "s52_ranker_smoke_archetype_path_diagnostics.csv")
    manifest_path = input_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    output_dir.mkdir(parents=True, exist_ok=True)

    if summary.empty:
        raise FileNotFoundError(f"No summary rows under {input_dir}")
    production_ranker = _production_ranker_check(lgbm_pipeline_path)
    best = summary.sort_values("objective", ascending=False).iloc[0]
    best_variant = str(best.get("variant", best.get("stage", "")))
    best_stage = str(best.get("stage", best_variant))
    if not folds.empty:
        fold_key = folds.get("variant", folds.get("stage"))
        if fold_key is not None:
            fold_key_str = fold_key.astype(str)
            best_folds = folds[fold_key_str.isin({best_variant, best_stage})].copy()
            if best_folds.empty:
                best_folds = folds.copy()
        else:
            best_folds = folds.copy()
    else:
        best_folds = folds.copy()
    checks: list[dict[str, Any]] = []
    _check(
        checks,
        name="top10_ev_weighted_first_touch_precision",
        value=_metric(best, "mean_top10_ev_weighted_first_touch_precision"),
        threshold=thresholds["top10_ev_weighted_first_touch_precision_min"],
        op=">=",
    )
    _check(
        checks,
        name="top20_ev_weighted_first_touch_precision",
        value=_metric(best, "mean_top20_ev_weighted_first_touch_precision"),
        threshold=thresholds["top20_ev_weighted_first_touch_precision_min"],
        op=">=",
    )
    _check(
        checks,
        name="top30_ev_weighted_first_touch_precision",
        value=_metric(best, "mean_top30_ev_weighted_first_touch_precision"),
        threshold=thresholds["top30_ev_weighted_first_touch_precision_min"],
        op=">=",
    )
    _check(
        checks,
        name="top10_first_touch_bad_mae_1r_rate",
        value=_metric(best, "mean_top10_first_touch_bad_mae_1r_rate"),
        threshold=thresholds["top10_first_touch_bad_mae_1r_rate_max"],
        op="<=",
    )
    _check(
        checks,
        name="top10_timeout_rate",
        value=_metric(best, "mean_top10_timeout_rate"),
        threshold=thresholds["top10_timeout_rate_max"],
        op="<=",
    )
    _check(
        checks,
        name="top10_mean_first_touch_mae_norm",
        value=_metric(best, "mean_top10_mean_first_touch_mae_norm"),
        threshold=thresholds["top10_mean_first_touch_mae_norm_max"],
        op="<=",
    )
    _check(
        checks,
        name="top10_p90_first_touch_mae_norm",
        value=_metric(best, "mean_top10_p90_first_touch_mae_norm"),
        threshold=thresholds["top10_p90_first_touch_mae_norm_max"],
        op="<=",
    )
    _check(
        checks,
        name="top10_mean_max_adverse_before_mfe_1r",
        value=_metric(best, "mean_top10_mean_max_adverse_before_mfe_1r"),
        threshold=thresholds["top10_mean_max_adverse_before_mfe_1r_max"],
        op="<=",
    )
    _check(
        checks,
        name="top10_mean_underwater_fraction_before_mfe",
        value=_metric(best, "mean_top10_mean_underwater_fraction_before_mfe"),
        threshold=thresholds["top10_mean_underwater_fraction_before_mfe_max"],
        op="<=",
    )
    _check(
        checks,
        name="top10_mean_underwater_bars_before_mfe",
        value=_metric(best, "mean_top10_mean_underwater_bars_before_mfe"),
        threshold=thresholds["top10_mean_underwater_bars_before_mfe_max"],
        op="<=",
    )
    _check(
        checks,
        name="top10_post_exit_full_path_bad_mae_1r_rate",
        value=_metric(best, "mean_top10_first_touch_full_path_bad_mae_1r_rate"),
        threshold=thresholds["top10_post_exit_full_path_bad_mae_1r_rate_warn"],
        op="<=",
        severity="warn",
    )
    _check(
        checks,
        name="top10_p90_post_exit_full_path_mae_norm",
        value=_metric(best, "mean_top10_p90_first_touch_full_path_mae_norm"),
        threshold=thresholds["top10_p90_post_exit_full_path_mae_norm_warn"],
        op="<=",
        severity="warn",
    )
    _check(
        checks,
        name="top10_mfe_1r_before_mae_1r_rate",
        value=_metric(best, "mean_top10_mfe_1r_before_mae_1r_rate"),
        threshold=thresholds["top10_mfe_1r_before_mae_1r_rate_min"],
        op=">=",
    )
    _check(
        checks,
        name="top10_mae_1r_before_mfe_1r_rate",
        value=_metric(best, "mean_top10_mae_1r_before_mfe_1r_rate"),
        threshold=thresholds["top10_mae_1r_before_mfe_1r_rate_max"],
        op="<=",
    )
    _check(
        checks,
        name="top10_mean_u",
        value=_metric(best, "mean_top10_mean_u"),
        threshold=thresholds["top10_mean_u_min"],
        op=">=",
    )
    _check(
        checks,
        name="top10_full_horizon_bad_mae_rate",
        value=_metric(best, "mean_top10_bad_mae_rate"),
        threshold=thresholds["top10_full_horizon_bad_mae_rate_max"],
        op="<=",
        severity="warn",
    )
    _check(
        checks,
        name="long_top10_mae_1r_before_mfe_1r_rate",
        value=_metric(best, "mean_long_top10_mae_1r_before_mfe_1r_rate"),
        threshold=thresholds["long_top10_mae_1r_before_mfe_1r_rate_max"],
        op="<=",
    )
    _check(
        checks,
        name="short_top10_mae_1r_before_mfe_1r_rate",
        value=_metric(best, "mean_short_top10_mae_1r_before_mfe_1r_rate"),
        threshold=thresholds["short_top10_mae_1r_before_mfe_1r_rate_max"],
        op="<=",
    )

    for _, fold in best_folds.iterrows():
        scope = f"fold:{fold.get('month')}"
        _check(
            checks,
            scope=scope,
            name="top10_p90_first_touch_mae_norm",
            value=_metric(fold, "top10_p90_first_touch_mae_norm"),
            threshold=thresholds["top10_p90_first_touch_mae_norm_max"],
            op="<=",
            severity="warn",
        )
        _check(
            checks,
            scope=scope,
            name="top10_mean_max_adverse_before_mfe_1r",
            value=_metric(fold, "top10_mean_max_adverse_before_mfe_1r"),
            threshold=thresholds["top10_mean_max_adverse_before_mfe_1r_max"],
            op="<=",
            severity="warn",
        )
        _check(
            checks,
            scope=scope,
            name="top10_mean_underwater_fraction_before_mfe",
            value=_metric(fold, "top10_mean_underwater_fraction_before_mfe"),
            threshold=thresholds["top10_mean_underwater_fraction_before_mfe_max"],
            op="<=",
            severity="warn",
        )
        _check(
            checks,
            scope=scope,
            name="top10_mean_underwater_bars_before_mfe",
            value=_metric(fold, "top10_mean_underwater_bars_before_mfe"),
            threshold=thresholds["top10_mean_underwater_bars_before_mfe_max"],
            op="<=",
            severity="warn",
        )
        _check(
            checks,
            scope=scope,
            name="top10_post_exit_full_path_bad_mae_1r_rate",
            value=_metric(fold, "top10_first_touch_full_path_bad_mae_1r_rate"),
            threshold=thresholds["top10_post_exit_full_path_bad_mae_1r_rate_warn"],
            op="<=",
            severity="warn",
        )
        _check(
            checks,
            scope=scope,
            name="top10_p90_post_exit_full_path_mae_norm",
            value=_metric(fold, "top10_p90_first_touch_full_path_mae_norm"),
            threshold=thresholds["top10_p90_post_exit_full_path_mae_norm_warn"],
            op="<=",
            severity="warn",
        )
        _check(
            checks,
            scope=scope,
            name="top10_mae_1r_before_mfe_1r_rate",
            value=_metric(fold, "top10_mae_1r_before_mfe_1r_rate"),
            threshold=thresholds["top10_mae_1r_before_mfe_1r_rate_max"],
            op="<=",
            severity="warn",
        )
        _check(
            checks,
            scope=scope,
            name="top10_mean_u",
            value=_metric(fold, "top10_mean_u"),
            threshold=thresholds["top10_mean_u_min"],
            op=">=",
        )

    checks_df = pd.DataFrame(checks)
    status = _status_from_checks(checks_df)
    hard_failures = checks_df[checks_df["status"].astype(str).eq("fail")].copy()
    warnings = checks_df[checks_df["status"].astype(str).eq("warn")].copy()

    best_long = pd.DataFrame()
    worst_long = pd.DataFrame()
    best_short = pd.DataFrame()
    if not archetypes.empty:
        selected = archetypes[pd.to_numeric(archetypes.get("selected_rows"), errors="coerce").fillna(0).ge(20)].copy()
        long = selected[selected["side"].astype(str).eq("long")].copy()
        short = selected[selected["side"].astype(str).eq("short")].copy()
        best_long = long.sort_values(
            ["selected_mfe_before_mae_1r_rate", "selected_mae_before_mfe_1r_rate"],
            ascending=[False, True],
        )
        worst_long = long.sort_values(
            ["selected_mae_before_mfe_1r_rate", "selected_rows"],
            ascending=[False, False],
        )
        best_short = short.sort_values(
            ["selected_mfe_before_mae_1r_rate", "selected_mae_before_mfe_1r_rate"],
            ascending=[False, True],
        )

    outputs = {
        "checks": output_dir / "s52_gate3_path_order_checks.csv",
        "json": output_dir / "s52_gate3_path_order_status.json",
        "markdown": output_dir / "s52_gate3_path_order_status.md",
    }
    checks_df.to_csv(outputs["checks"], index=False)
    decision = {
        "scope": "s52_gate3_path_order_status",
        "input_dir": str(input_dir),
        "status": status,
        "best_variant": best_variant,
        "thresholds": thresholds,
        "aggregate_metrics": {str(k): _json_safe(best.get(k)) for k in best.index},
        "hard_failure_count": int(len(hard_failures)),
        "warning_count": int(len(warnings)),
        "manifest_scope": manifest.get("scope"),
        "production_ranker_materialization": production_ranker,
        "outputs": {k: str(v) for k, v in outputs.items()},
    }
    outputs["json"].write_text(json.dumps(_json_safe(decision), indent=2), encoding="utf-8")

    cols = ["scope", "metric", "value", "operator", "threshold", "status"]
    arch_cols = [
        "state_feature",
        "bucket",
        "side",
        "selected_rows",
        "selected_mfe_before_mae_1r_rate",
        "selected_mae_before_mfe_1r_rate",
        "selected_mean_first_touch_mae_norm",
        "selected_p90_first_touch_mae_norm",
        "selected_first_touch_full_path_bad_mae_1r_rate",
        "selected_p90_first_touch_full_path_mae_norm",
        "selected_mean_u",
    ]
    lines = [
        "# S52 Gate 3 Path-Order Status",
        "",
        f"Input: `{input_dir}`",
        f"Best variant: `{decision['best_variant']}`",
        f"Status: `{status}`",
        f"Production ranker materialization: `{production_ranker['status']}` ({production_ranker['reason']})",
        "",
        "## Hard Failures",
        "",
        _table(hard_failures, cols, limit=30),
        "",
        "## Warnings",
        "",
        _table(warnings, cols, limit=30),
        "",
        "## All Checks",
        "",
        _table(checks_df, cols, limit=80),
        "",
        "## Archetype Overlay Read",
        "",
        "Best long state buckets:",
        "",
        _table(best_long, arch_cols, limit=10),
        "",
        "Worst long state buckets:",
        "",
        _table(worst_long, arch_cols, limit=10),
        "",
        "Best short state buckets:",
        "",
        _table(best_short, arch_cols, limit=10),
        "",
        "## Production Integration",
        "",
        _table(pd.DataFrame([production_ranker]), ["status", "path", "has_lgbm_ranker", "has_lambdarank", "reason"]),
        "",
        "## Outputs",
        "",
        f"- Checks: `{outputs['checks']}`",
        f"- JSON: `{outputs['json']}`",
    ]
    outputs["markdown"].write_text("\n".join(lines) + "\n", encoding="utf-8")
    return decision


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--lgbm-pipeline-path", type=Path, default=DEFAULT_LGBM_PIPELINE)
    args = parser.parse_args()
    decision = build_report(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        lgbm_pipeline_path=args.lgbm_pipeline_path,
    )
    print(json.dumps(_json_safe(decision), indent=2))


if __name__ == "__main__":
    main()
