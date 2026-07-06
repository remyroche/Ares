#!/usr/bin/env python3
"""Fit-selected utility + timeout + path-risk gate diagnostic.

This is a pre-training label/execution alignment diagnostic. It selects utility
candidates, timeout gates, and adverse-path gates from fit-month IC evidence
only, then reports whether simple rank combinations transfer to the holdout
month inside the economic envelope.
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

from scripts.report_timeout_feature_stability import (  # noqa: E402
    DEFAULT_EVENT_FEATURE_STORE_FEATURES,
    DEFAULT_FEATURE_DIR,
    DEFAULT_FEATURE_LIST_CSV,
    DEFAULT_FIT_MONTHS,
    DEFAULT_HOLDOUT_MONTH,
    DEFAULT_LABELS_DIR,
    DEFAULT_PRIOR_WINDOWS_DAYS,
    DEFAULT_STATE_PATH_PRIOR_FEATURES,
    _is_clean,
    _load_frame,
    _month_summary,
    _parse_csv,
    _parse_float_csv,
    _safe_mean,
    _safe_numeric,
    _table,
)
from scripts.report_timeout_gate_combo_diagnostic import (  # noqa: E402
    DEFAULT_STAGE51_DIR,
    _rank_pct,
    _select_utility_candidates,
)
from scripts.run_label_quality_proxy_diagnostics import _json_safe, _selection_metrics  # noqa: E402


DEFAULT_OUTPUT_DIR = Path("data_perp/reports/path_gate_combo_stage53_v1")
DEFAULT_TOP_FRACS = (0.0025, 0.005, 0.01)
DEFAULT_TIMEOUT_PENALTIES = (0.50, 1.00, 1.50)
DEFAULT_PATH_PENALTIES = (0.50, 1.00, 1.50)
DEFAULT_TIMEOUT_KEEP_FRACS = (1.0,)
DEFAULT_PATH_KEEP_FRACS = (1.0, 0.50)


def _fit_gate_summary(
    period_ic: pd.DataFrame,
    feature_summary: pd.DataFrame,
    *,
    fit_months: list[str],
    target: str,
) -> pd.DataFrame:
    if target not in {"timeout", "bad_mae"}:
        raise ValueError(f"Unsupported gate target: {target}")
    fit_periods = set(str(v) for v in fit_months)
    rows: list[dict[str, Any]] = []
    for feature, group in period_ic.groupby("feature", sort=False):
        fit = group[group["period"].astype(str).isin(fit_periods)].copy()
        if fit.empty:
            continue
        raw_target = _safe_numeric(fit[f"ic_{target}"])
        if not raw_target.notna().any():
            continue
        direction = -1.0 if _safe_mean(raw_target) > 0.0 else 1.0
        signed_target = direction * _safe_numeric(fit[f"ic_{target}"])
        signed_u = direction * _safe_numeric(fit["ic_utility"])
        signed_bad = direction * _safe_numeric(fit["ic_bad_mae"])
        signed_timeout = direction * _safe_numeric(fit["ic_timeout"])
        signed_wide = direction * _safe_numeric(fit["ic_wide_25"])
        target_max = float(signed_target.max()) if signed_target.notna().any() else float("nan")
        utility_min = float(signed_u.min()) if signed_u.notna().any() else float("nan")
        bad_max = float(signed_bad.max()) if signed_bad.notna().any() else float("nan")
        timeout_max = float(signed_timeout.max()) if signed_timeout.notna().any() else float("nan")
        wide_max = float(signed_wide.max()) if signed_wide.notna().any() else float("nan")
        gate_score = (
            -2.0 * (target_max if math.isfinite(target_max) else 1.0)
            + 0.75 * (utility_min if math.isfinite(utility_min) else -1.0)
            - 0.50 * max(0.0, bad_max if math.isfinite(bad_max) else 1.0)
            - 0.25 * max(0.0, timeout_max if math.isfinite(timeout_max) else 1.0)
            - 0.10 * max(0.0, wide_max if math.isfinite(wide_max) else 1.0)
        )
        rows.append(
            {
                "feature": str(feature),
                "gate_target": target,
                "gate_direction": "high" if direction > 0.0 else "low",
                "gate_direction_sign": direction,
                "fit_target_ic_max": target_max,
                "fit_target_ic_mean": _safe_mean(signed_target),
                "fit_utility_ic_min": utility_min,
                "fit_utility_ic_mean": _safe_mean(signed_u),
                "fit_bad_mae_ic_max": bad_max,
                "fit_timeout_ic_max": timeout_max,
                "fit_wide_ic_max": wide_max,
                "fit_gate_score": float(gate_score),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    meta_cols = ["feature", "feature_family", "mean_finite_frac", "min_nunique"]
    out = out.merge(feature_summary[meta_cols], on="feature", how="left")
    return out.sort_values(
        ["fit_gate_score", "fit_target_ic_mean", "fit_utility_ic_min"],
        ascending=[False, True, False],
    ).reset_index(drop=True)


def _select_gates(
    period_ic: pd.DataFrame,
    feature_summary: pd.DataFrame,
    *,
    fit_months: list[str],
    target: str,
    max_gates: int,
    require_utility: bool,
) -> pd.DataFrame:
    gates = _fit_gate_summary(period_ic, feature_summary, fit_months=fit_months, target=target)
    if gates.empty:
        return gates
    strict = gates[gates["fit_target_ic_max"].le(0.0)].copy()
    if target == "timeout":
        strict = strict[strict["fit_bad_mae_ic_max"].le(0.0)].copy()
    if require_utility:
        strict = strict[strict["fit_utility_ic_min"].ge(0.0)].copy()
    if len(strict) < int(max_gates):
        relaxed = gates[gates["fit_target_ic_mean"].lt(0.0)].copy()
        strict = pd.concat([strict, relaxed], ignore_index=True).drop_duplicates("feature")
    return strict.sort_values(
        ["fit_gate_score", "fit_target_ic_mean", "fit_utility_ic_min"],
        ascending=[False, True, False],
    ).head(int(max_gates)).reset_index(drop=True)


def _combo_score(
    frame: pd.DataFrame,
    *,
    utility_feature: str,
    utility_direction_sign: float,
    timeout_feature: str,
    timeout_direction_sign: float,
    path_feature: str,
    path_direction_sign: float,
    timeout_penalty: float,
    path_penalty: float,
    timeout_keep_frac: float,
    path_keep_frac: float,
) -> pd.Series:
    utility_rank = _rank_pct(float(utility_direction_sign) * _safe_numeric(frame[utility_feature]))
    timeout_rank = _rank_pct(float(timeout_direction_sign) * _safe_numeric(frame[timeout_feature]))
    path_rank = _rank_pct(float(path_direction_sign) * _safe_numeric(frame[path_feature]))
    score = utility_rank + float(timeout_penalty) * timeout_rank + float(path_penalty) * path_rank
    timeout_keep = float(timeout_keep_frac)
    if timeout_keep < 0.999:
        score = score.where(timeout_rank >= max(0.0, min(1.0, 1.0 - timeout_keep)), np.nan)
    path_keep = float(path_keep_frac)
    if path_keep < 0.999:
        score = score.where(path_rank >= max(0.0, min(1.0, 1.0 - path_keep)), np.nan)
    return score


def _aggregate(monthly: pd.DataFrame, fit_months: list[str], holdout_month: str) -> pd.DataFrame:
    if monthly.empty:
        return pd.DataFrame()
    group_cols = [
        "utility_feature",
        "utility_feature_family",
        "utility_direction",
        "timeout_feature",
        "timeout_feature_family",
        "timeout_direction",
        "path_feature",
        "path_feature_family",
        "path_direction",
        "timeout_penalty",
        "path_penalty",
        "timeout_keep_frac",
        "path_keep_frac",
        "top_frac",
    ]
    rows: list[dict[str, Any]] = []
    for key, group in monthly.groupby(group_cols, dropna=False, sort=False):
        row = dict(zip(group_cols, key))
        fit = group[group["period"].astype(str).isin(fit_months)].copy()
        holdout = group[group["period"].astype(str).eq(str(holdout_month))].copy()
        if fit.empty or holdout.empty:
            continue
        row.update(_month_summary("fit", fit))
        row.update(_month_summary("holdout", holdout))
        row["fit_clean_pass"] = _is_clean(pd.Series(row), "fit")
        row["holdout_clean_standalone_pass"] = _is_clean(pd.Series(row), "holdout")
        row["holdout_clean_pass"] = bool(row["fit_clean_pass"] and row["holdout_clean_standalone_pass"])
        row["positive_dirty_holdout"] = bool(row["holdout_mean_month_u"] > 0.0 and not row["holdout_clean_pass"])
        row["path_risk_score"] = float(
            (row["holdout_mean_month_u"] if math.isfinite(row["holdout_mean_month_u"]) else 0.0)
            + 0.25 * (row["holdout_q10_u"] if math.isfinite(row["holdout_q10_u"]) else 0.0)
            - 0.020 * (row["holdout_bad_mae_1r_rate"] if math.isfinite(row["holdout_bad_mae_1r_rate"]) else 0.0)
            - 0.003 * (row["holdout_p90_mae_norm"] if math.isfinite(row["holdout_p90_mae_norm"]) else 0.0)
            - 0.010 * (row["holdout_timeout_rate"] if math.isfinite(row["holdout_timeout_rate"]) else 0.0)
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _write_markdown(output_dir: Path, aggregate: pd.DataFrame, manifest: dict[str, Any]) -> Path:
    path = output_dir / "path_gate_combo_report.md"
    if aggregate.empty:
        best_fit = aggregate
        best_holdout = aggregate
    else:
        best_fit = aggregate.sort_values(
            ["fit_clean_pass", "fit_mean_month_u", "fit_worst_month_u", "fit_timeout_rate", "fit_p90_mae_norm"],
            ascending=[False, False, False, True, True],
        )
        best_holdout = aggregate.sort_values(
            ["holdout_clean_pass", "positive_dirty_holdout", "path_risk_score"],
            ascending=[False, False, False],
        )
    cols = [
        "utility_feature",
        "utility_feature_family",
        "timeout_feature",
        "timeout_direction",
        "path_feature",
        "path_direction",
        "timeout_penalty",
        "path_penalty",
        "path_keep_frac",
        "top_frac",
        "fit_mean_month_u",
        "fit_worst_month_u",
        "fit_bad_mae_1r_rate",
        "fit_p90_mae_norm",
        "fit_timeout_rate",
        "holdout_mean_month_u",
        "holdout_bad_mae_1r_rate",
        "holdout_p90_mae_norm",
        "holdout_timeout_rate",
        "holdout_clean_pass",
        "path_risk_score",
    ]
    lines = [
        "# Path Gate Combo Diagnostic",
        "",
        "Scope: pre-training rank-combo diagnostic. Utility, timeout, and path gates are selected from fit months only; holdout is reported after selection.",
        "",
        f"Fit months: `{','.join(manifest['fit_months'])}`. Holdout month: `{manifest['holdout_month']}`.",
        f"Utility candidates: `{manifest['utility_candidate_count']}`.",
        f"Timeout gates: `{manifest['timeout_gate_count']}`.",
        f"Path gates: `{manifest['path_gate_count']}`.",
        f"Combos evaluated: `{manifest['combo_rows']}`.",
        "",
        "## Counts",
        "",
        f"- Fit clean rows: `{manifest['fit_clean_rows']}`",
        f"- Holdout clean rows: `{manifest['holdout_clean_rows']}`",
        f"- Positive but dirty holdout rows: `{manifest['positive_dirty_holdout_rows']}`",
        "",
        "## Best Fit-Selected Rows",
        "",
        _table(best_fit, cols, limit=30),
        "",
        "## Best Holdout Rows",
        "",
        _table(best_holdout, cols, limit=30),
        "",
        "## Interpretation",
        "",
        "- A train decision requires fit-clean and holdout-clean rows under the same combo definition.",
        "- Holdout-only positives remain diagnostic until the fit envelope also passes.",
        "",
        "## Outputs",
        "",
        f"- Monthly: `{manifest['outputs']['monthly']}`",
        f"- Aggregate: `{manifest['outputs']['aggregate']}`",
        f"- Utility candidates: `{manifest['outputs']['utility_candidates']}`",
        f"- Timeout gates: `{manifest['outputs']['timeout_gates']}`",
        f"- Path gates: `{manifest['outputs']['path_gates']}`",
        f"- Manifest: `{manifest['outputs']['manifest']}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_report(
    *,
    labels_path: Path,
    feature_dir: Path,
    feature_list_csv: Path,
    stage51_dir: Path,
    output_dir: Path,
    fit_months: list[str],
    holdout_month: str,
    top_fracs: list[float],
    timeout_penalties: list[float],
    path_penalties: list[float],
    timeout_keep_fracs: list[float],
    path_keep_fracs: list[float],
    max_utility_features: int,
    max_timeout_gates: int,
    max_path_gates: int,
    prior_windows_days: list[float],
    prior_embargo_hours: float,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_summary = pd.read_csv(stage51_dir / "timeout_feature_stability.csv")
    period_ic = pd.read_csv(stage51_dir / "timeout_feature_period_ic.csv")
    utility_candidates = _select_utility_candidates(
        period_ic=period_ic,
        feature_summary=feature_summary,
        fit_months=fit_months,
        max_utility_features=max_utility_features,
    )
    timeout_gates = _select_gates(
        period_ic,
        feature_summary,
        fit_months=fit_months,
        target="timeout",
        max_gates=max_timeout_gates,
        require_utility=True,
    )
    path_gates = _select_gates(
        period_ic,
        feature_summary,
        fit_months=fit_months,
        target="bad_mae",
        max_gates=max_path_gates,
        require_utility=True,
    )
    frame, metrics, _, reports = _load_frame(
        labels_path=labels_path,
        feature_dir=feature_dir,
        feature_list_csv=feature_list_csv,
        max_feature_store_features=None,
        include_causal_outcome_priors=True,
        include_causal_state_path_priors=True,
        include_event_confirmation_features=True,
        include_slow_trade_diagnostic_features=True,
        prior_windows_days=prior_windows_days,
        prior_embargo_hours=prior_embargo_hours,
        state_path_prior_features=list(DEFAULT_STATE_PATH_PRIOR_FEATURES),
        event_feature_store_features=list(DEFAULT_EVENT_FEATURE_STORE_FEATURES),
    )
    month_series = frame["__ts__"].dt.to_period("M").astype(str)
    months = sorted(set(fit_months + [holdout_month]))
    dummy_targets = pd.DataFrame(
        {
            "target_soft": pd.Series(0.0, index=frame.index),
            "target_hard": pd.Series(0.0, index=frame.index),
        }
    )
    monthly_rows: list[dict[str, Any]] = []
    for _, urow in utility_candidates.iterrows():
        utility_feature = str(urow["feature"])
        if utility_feature not in frame.columns:
            continue
        for _, trow in timeout_gates.iterrows():
            timeout_feature = str(trow["feature"])
            if timeout_feature not in frame.columns:
                continue
            for _, prow in path_gates.iterrows():
                path_feature = str(prow["feature"])
                if path_feature not in frame.columns:
                    continue
                for timeout_penalty in timeout_penalties:
                    for path_penalty in path_penalties:
                        for timeout_keep_frac in timeout_keep_fracs:
                            for path_keep_frac in path_keep_fracs:
                                for period in months:
                                    mask = month_series.eq(period)
                                    if int(mask.sum()) < 100:
                                        continue
                                    valid_frame = frame.loc[mask].reset_index(drop=True)
                                    valid_metrics = metrics.loc[mask].reset_index(drop=True)
                                    valid_target = dummy_targets.loc[mask].reset_index(drop=True)
                                    score = _combo_score(
                                        valid_frame,
                                        utility_feature=utility_feature,
                                        utility_direction_sign=float(urow["utility_direction_sign"]),
                                        timeout_feature=timeout_feature,
                                        timeout_direction_sign=float(trow["gate_direction_sign"]),
                                        path_feature=path_feature,
                                        path_direction_sign=float(prow["gate_direction_sign"]),
                                        timeout_penalty=float(timeout_penalty),
                                        path_penalty=float(path_penalty),
                                        timeout_keep_frac=float(timeout_keep_frac),
                                        path_keep_frac=float(path_keep_frac),
                                    )
                                    for top_frac in top_fracs:
                                        metric_row = _selection_metrics(
                                            frame=valid_frame,
                                            metrics=valid_metrics,
                                            target=valid_target,
                                            score=score,
                                            arm=f"{utility_feature}::{timeout_feature}::{path_feature}",
                                            selector="utility_timeout_path_gate_rank_combo",
                                            period=str(period),
                                            top_frac=float(top_frac),
                                        )
                                        metric_row.update(
                                            {
                                                "utility_feature": utility_feature,
                                                "utility_feature_family": str(urow.get("feature_family", "")),
                                                "utility_direction": str(urow["utility_direction"]),
                                                "timeout_feature": timeout_feature,
                                                "timeout_feature_family": str(trow.get("feature_family", "")),
                                                "timeout_direction": str(trow["gate_direction"]),
                                                "path_feature": path_feature,
                                                "path_feature_family": str(prow.get("feature_family", "")),
                                                "path_direction": str(prow["gate_direction"]),
                                                "timeout_penalty": float(timeout_penalty),
                                                "path_penalty": float(path_penalty),
                                                "timeout_keep_frac": float(timeout_keep_frac),
                                                "path_keep_frac": float(path_keep_frac),
                                                "utility_fit_candidate_score": float(urow["fit_candidate_score"]),
                                                "timeout_gate_score": float(trow["fit_gate_score"]),
                                                "path_gate_score": float(prow["fit_gate_score"]),
                                            }
                                        )
                                        monthly_rows.append(metric_row)

    monthly = pd.DataFrame(monthly_rows)
    aggregate = _aggregate(monthly, fit_months=fit_months, holdout_month=holdout_month)
    paths = {
        "monthly": output_dir / "path_gate_combo_monthly.csv",
        "aggregate": output_dir / "path_gate_combo_fit_holdout.csv",
        "utility_candidates": output_dir / "path_gate_combo_utility_candidates.csv",
        "timeout_gates": output_dir / "path_gate_combo_timeout_gates.csv",
        "path_gates": output_dir / "path_gate_combo_path_gates.csv",
        "manifest": output_dir / "manifest.json",
    }
    monthly.to_csv(paths["monthly"], index=False)
    aggregate.to_csv(paths["aggregate"], index=False)
    utility_candidates.to_csv(paths["utility_candidates"], index=False)
    timeout_gates.to_csv(paths["timeout_gates"], index=False)
    path_gates.to_csv(paths["path_gates"], index=False)
    manifest = {
        "scope": "path_gate_combo_diagnostic",
        "labels_path": str(labels_path),
        "feature_dir": str(feature_dir),
        "feature_list_csv": str(feature_list_csv),
        "stage51_dir": str(stage51_dir),
        "output_dir": str(output_dir),
        "fit_months": list(fit_months),
        "holdout_month": str(holdout_month),
        "top_fracs": [float(v) for v in top_fracs],
        "timeout_penalties": [float(v) for v in timeout_penalties],
        "path_penalties": [float(v) for v in path_penalties],
        "timeout_keep_fracs": [float(v) for v in timeout_keep_fracs],
        "path_keep_fracs": [float(v) for v in path_keep_fracs],
        "utility_candidate_count": int(len(utility_candidates)),
        "timeout_gate_count": int(len(timeout_gates)),
        "path_gate_count": int(len(path_gates)),
        "combo_rows": int(len(aggregate)),
        "fit_clean_rows": int(aggregate["fit_clean_pass"].sum()) if not aggregate.empty else 0,
        "holdout_clean_rows": int(aggregate["holdout_clean_pass"].sum()) if not aggregate.empty else 0,
        "positive_dirty_holdout_rows": int(aggregate["positive_dirty_holdout"].sum()) if not aggregate.empty else 0,
        "feature_reports": reports,
        "outputs": {key: str(value) for key, value in paths.items()},
    }
    markdown = _write_markdown(output_dir, aggregate, manifest)
    manifest["outputs"]["markdown"] = str(markdown)
    paths["manifest"].write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-path", type=Path, default=DEFAULT_LABELS_DIR)
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--feature-list-csv", type=Path, default=DEFAULT_FEATURE_LIST_CSV)
    parser.add_argument("--stage51-dir", type=Path, default=DEFAULT_STAGE51_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--fit-months", type=_parse_csv, default=",".join(DEFAULT_FIT_MONTHS))
    parser.add_argument("--holdout-month", type=str, default=DEFAULT_HOLDOUT_MONTH)
    parser.add_argument("--top-fracs", type=_parse_float_csv, default=",".join(str(v) for v in DEFAULT_TOP_FRACS))
    parser.add_argument(
        "--timeout-penalties",
        type=_parse_float_csv,
        default=",".join(str(v) for v in DEFAULT_TIMEOUT_PENALTIES),
    )
    parser.add_argument(
        "--path-penalties",
        type=_parse_float_csv,
        default=",".join(str(v) for v in DEFAULT_PATH_PENALTIES),
    )
    parser.add_argument(
        "--timeout-keep-fracs",
        type=_parse_float_csv,
        default=",".join(str(v) for v in DEFAULT_TIMEOUT_KEEP_FRACS),
    )
    parser.add_argument(
        "--path-keep-fracs",
        type=_parse_float_csv,
        default=",".join(str(v) for v in DEFAULT_PATH_KEEP_FRACS),
    )
    parser.add_argument("--max-utility-features", type=int, default=30)
    parser.add_argument("--max-timeout-gates", type=int, default=2)
    parser.add_argument("--max-path-gates", type=int, default=8)
    parser.add_argument("--prior-windows-days", type=_parse_float_csv, default=",".join(str(v) for v in DEFAULT_PRIOR_WINDOWS_DAYS))
    parser.add_argument("--prior-embargo-hours", type=float, default=24.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = run_report(
        labels_path=args.labels_path,
        feature_dir=args.feature_dir,
        feature_list_csv=args.feature_list_csv,
        stage51_dir=args.stage51_dir,
        output_dir=args.output_dir,
        fit_months=list(args.fit_months),
        holdout_month=str(args.holdout_month),
        top_fracs=list(args.top_fracs),
        timeout_penalties=list(args.timeout_penalties),
        path_penalties=list(args.path_penalties),
        timeout_keep_fracs=list(args.timeout_keep_fracs),
        path_keep_fracs=list(args.path_keep_fracs),
        max_utility_features=int(args.max_utility_features),
        max_timeout_gates=int(args.max_timeout_gates),
        max_path_gates=int(args.max_path_gates),
        prior_windows_days=list(args.prior_windows_days),
        prior_embargo_hours=float(args.prior_embargo_hours),
    )
    print(json.dumps(_json_safe(manifest), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
