#!/usr/bin/env python3
"""Source-conditioned two-head label proxy test before model training.

This keeps the Stage 10 abstain-then-rank shape, but fits and evaluates it
inside causal decision-time source masks. The goal is to test whether the
global two-head failure is caused by mixing incompatible source regimes.
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

from scripts.run_label_first_touch_execution_proxy_ablation import (  # noqa: E402
    DEFAULT_FEATURE_DIR,
    DEFAULT_FEATURE_LIST_CSV,
    DEFAULT_FIT_MONTHS,
    DEFAULT_HOLDOUT_MONTH,
    DEFAULT_LABELS_DIR,
    _feature_columns,
    _first_touch_metrics,
    _json_safe,
    _load_feature_store_columns,
    _load_labels,
    _parse_csv,
    _parse_float_csv,
    _path_metrics,
    _proxy_score,
    _read_feature_list,
    _safe_mean,
    _safe_numeric,
    _spearman,
)
from scripts.run_label_first_touch_soft_recipe_proxy_ablation import DEFAULT_TOP_KS  # noqa: E402
from scripts.run_label_two_head_abstention_utility_proxy import (  # noqa: E402
    DEFAULT_BAD_THRESHOLDS,
    DEFAULT_SCORE_RULES,
    DEFAULT_UTILITY_TARGETS,
    TwoHeadSpec,
    _build_specs,
    _fit_holdout_summary,
    _format_table,
    _global_bad_soft,
    _monthly_weekly_rows,
    _score_from_spec,
    _target_for_selection,
    _utility_targets,
)
from scripts.run_soft_label_candidate_source_ablation import (  # noqa: E402
    _build_sources,
    _source_context,
    _source_summary,
)
from scripts.run_soft_label_economic_proxy_ablation import (  # noqa: E402
    DEFAULT_EVENT_FEATURE_STORE_FEATURES,
    _event_confirmation_features,
)


DEFAULT_OUTPUT_DIR = Path("data_perp/reports/label_source_conditioned_two_head_proxy_v1")
DEFAULT_SOURCES = (
    "all",
    "quiet_mid",
    "quiet_quality",
    "loud_event",
    "any_event_quality",
    "low_zscore_rebound",
    "low_atr_rebound",
    "rebound_mid",
    "rebound_event_quality",
)


def _source_fit_holdout_summary(
    *,
    monthly: pd.DataFrame,
    weekly: pd.DataFrame,
    fit_months: list[str],
    holdout_month: str,
    min_week_rows: int,
) -> pd.DataFrame:
    if monthly.empty:
        return pd.DataFrame()
    parts: list[pd.DataFrame] = []
    for source in sorted(monthly["source"].astype(str).unique()):
        monthly_source = monthly[monthly["source"].astype(str).eq(source)].copy()
        weekly_source = weekly[weekly["source"].astype(str).eq(source)].copy()
        summary = _fit_holdout_summary(
            monthly=monthly_source,
            weekly=weekly_source,
            fit_months=fit_months,
            holdout_month=holdout_month,
            min_week_rows=min_week_rows,
        )
        if summary.empty:
            continue
        summary.insert(0, "source", source)
        parts.append(summary)
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True)
    return out.sort_values(
        ["holdout_bounded_pass", "positive_dirty_holdout", "holdout_objective", "holdout_mean_month_u"],
        ascending=[False, False, False, False],
    )


def _select_by_source_fit(fit_holdout: pd.DataFrame) -> pd.DataFrame:
    if fit_holdout.empty:
        return fit_holdout
    rows: list[pd.Series] = []
    group_cols = ["source", "utility_target", "score_rule", "top_k"]
    for _, group in fit_holdout.groupby(group_cols, observed=True, dropna=False):
        candidates = group.copy()
        if bool(candidates["fit_bounded_pass"].any()):
            candidates = candidates[candidates["fit_bounded_pass"]].copy()
        elif bool(candidates["fit_sign_pass"].any()):
            candidates = candidates[candidates["fit_sign_pass"]].copy()
        chosen = candidates.sort_values(
            ["fit_bounded_pass", "fit_sign_pass", "fit_selection_objective", "fit_mean_month_u"],
            ascending=[False, False, False, False],
        ).iloc[0]
        rows.append(chosen)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(
        ["holdout_bounded_pass", "holdout_sign_pass", "holdout_objective", "fit_selection_objective"],
        ascending=[False, False, False, False],
    )


def _counts_by_source(fit_holdout: pd.DataFrame) -> pd.DataFrame:
    if fit_holdout.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for source, group in fit_holdout.groupby("source", observed=True, dropna=False):
        rows.append(
            {
                "source": str(source),
                "rows": int(len(group)),
                "fit_sign": int(group["fit_sign_pass"].sum()),
                "holdout_sign": int(group["holdout_sign_pass"].sum()),
                "fit_bounded": int(group["fit_bounded_pass"].sum()),
                "holdout_bounded": int(group["holdout_bounded_pass"].sum()),
                "standalone_holdout_bounded": int(group["holdout_bounded_standalone_pass"].sum()),
                "positive_dirty": int(group["positive_dirty_holdout"].sum()),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["holdout_bounded", "standalone_holdout_bounded", "holdout_sign", "fit_sign"],
        ascending=[False, False, False, False],
    )


def _write_markdown(
    *,
    output_dir: Path,
    source_summary: pd.DataFrame,
    fit_holdout: pd.DataFrame,
    selected_by_fit: pd.DataFrame,
    proxy_ic: pd.DataFrame,
    manifest: dict[str, Any],
) -> Path:
    path = output_dir / "source_conditioned_two_head_proxy.md"
    counts = _counts_by_source(fit_holdout)
    source_cols = [
        "source",
        "rows",
        "row_frac",
        "mean_u",
        "hit_u",
        "bad_mae_1r_rate",
        "timeout_rate",
        "rows_2026_04",
        "rows_2026_05",
        "rows_2026_06",
    ]
    fit_cols = [
        "source",
        "utility_target",
        "score_rule",
        "bad_threshold",
        "top_k",
        "fit_selection_objective",
        "holdout_objective",
        "fit_sign_pass",
        "fit_bounded_pass",
        "holdout_sign_pass",
        "holdout_bounded_standalone_pass",
        "holdout_bounded_pass",
        "fit_mean_month_u",
        "fit_material_positive_week_rate",
        "fit_p90_first_touch_mae_to_sl",
        "fit_first_touch_bad_mae_to_sl_rate",
        "fit_clean_exec_actual_rate",
        "holdout_mean_month_u",
        "holdout_material_positive_week_rate",
        "holdout_p90_first_touch_mae_to_sl",
        "holdout_first_touch_bad_mae_to_sl_rate",
        "holdout_clean_exec_actual_rate",
    ]
    ic_cols = [
        "source",
        "period",
        "utility_target",
        "train_rows",
        "valid_rows",
        "utility_proxy_ic_target",
        "utility_proxy_ic_u",
        "utility_proxy_ic_clean_exec",
        "utility_proxy_ic_bad",
        "bad_proxy_ic_bad",
        "bad_proxy_ic_clean_exec",
        "utility_proxy_features",
        "bad_proxy_features",
    ]
    lines = [
        "# Source-Conditioned Two-Head Proxy",
        "",
        "Scope: proxy tests only. No LightGBM, Optuna, policy geometry, or base/meta training is run.",
        "",
        f"Labels: `{manifest['labels_path']}`",
        f"Proxy method: `{manifest['proxy_method']}`",
        f"Sources: `{','.join(manifest['sources'])}`",
        f"Fit months: `{','.join(manifest['fit_months'])}`",
        f"Holdout month: `{manifest['holdout_month']}`",
        "",
        "Threshold selection is per source and uses Apr-May only; June is the holdout.",
        "",
        "## Source Summary",
        "",
        _format_table(source_summary, source_cols, limit=80),
        "",
        "## Counts By Source",
        "",
        _format_table(counts, ["source", "rows", "fit_sign", "holdout_sign", "fit_bounded", "holdout_bounded", "standalone_holdout_bounded", "positive_dirty"], limit=80),
        "",
        "## Selected By Source Fit",
        "",
        _format_table(selected_by_fit, fit_cols, limit=80),
        "",
        "## Best Grid Rows",
        "",
        _format_table(fit_holdout, fit_cols, limit=120),
        "",
        "## Proxy IC",
        "",
        _format_table(proxy_ic.sort_values(["source", "period", "utility_proxy_ic_u"], ascending=[True, True, False]), ic_cols, limit=120),
        "",
        "## Outputs",
        "",
        f"- Source summary: `{manifest['outputs']['source_summary']}`",
        f"- Monthly: `{manifest['outputs']['monthly']}`",
        f"- Weekly: `{manifest['outputs']['weekly']}`",
        f"- Proxy IC: `{manifest['outputs']['proxy_ic']}`",
        f"- Fit/Holdout: `{manifest['outputs']['fit_holdout']}`",
        f"- Selected by fit: `{manifest['outputs']['selected_by_fit']}`",
        f"- Manifest: `{manifest['outputs']['manifest']}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_ablation(
    *,
    labels_path: Path,
    output_dir: Path,
    feature_dir: Path,
    feature_list_csv: Path,
    max_feature_store_features: int | None,
    proxy_top_k: int,
    proxy_method: str,
    proxy_tail_frac: float,
    utility_targets: list[str],
    score_rules: list[str],
    bad_thresholds: list[float],
    top_ks: list[int],
    fit_months: list[str],
    holdout_month: str,
    min_week_rows: int,
    sources: list[str],
    run_gap_hours: float,
    min_train_source_rows: int,
    min_valid_source_rows: int,
    event_feature_store_features: list[str],
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = _load_labels(labels_path)
    selected_features = _read_feature_list(feature_list_csv, max_features=max_feature_store_features)
    selected_features = list(dict.fromkeys(list(selected_features) + list(event_feature_store_features)))
    feature_matrix, feature_store_report = _load_feature_store_columns(
        frame,
        feature_dir=feature_dir,
        selected_features=selected_features,
    )
    if not feature_matrix.empty:
        new_cols = [col for col in feature_matrix.columns if col not in frame.columns]
        frame = pd.concat([frame.reset_index(drop=True), feature_matrix.loc[:, new_cols].reset_index(drop=True)], axis=1)
    event_features, event_report = _event_confirmation_features(
        frame,
        event_features=event_feature_store_features,
    )
    if not event_features.empty:
        new_event_cols = [col for col in event_features.columns if col not in frame.columns]
        frame = pd.concat([frame.reset_index(drop=True), event_features.loc[:, new_event_cols].reset_index(drop=True)], axis=1)
    context = _source_context(frame)
    frame = pd.concat([frame.reset_index(drop=True), context.reset_index(drop=True)], axis=1)
    metrics = _path_metrics(frame)
    ft = _first_touch_metrics(frame, metrics)
    bad_soft = _global_bad_soft(ft)
    utility_map = _utility_targets(frame, ft)
    missing = sorted(set(utility_targets) - set(utility_map))
    if missing:
        raise ValueError(f"Unknown utility target(s): {missing}")
    all_sources = _build_sources(frame, context, run_gap_hours=run_gap_hours)
    if sources:
        missing_sources = sorted(set(sources) - set(all_sources))
        if missing_sources:
            raise ValueError(f"Unknown source(s): {missing_sources}")
        source_masks = {source: all_sources[source] for source in sources}
    else:
        source_masks = all_sources
        sources = list(source_masks)
    source_summary = _source_summary(frame=frame, metrics=ft, context=context, sources=source_masks)

    features = _feature_columns(frame)
    specs = _build_specs(utility_targets=utility_targets, score_rules=score_rules, bad_thresholds=bad_thresholds)
    month_series = frame["__ts__"].dt.to_period("M").astype(str)
    months = sorted(month_series.dropna().unique())
    monthly_rows: list[dict[str, Any]] = []
    weekly_rows: list[dict[str, Any]] = []
    proxy_ic_rows: list[dict[str, Any]] = []

    for source_name, source_mask in source_masks.items():
        source_mask = source_mask.reindex(frame.index, fill_value=False).astype(bool)
        for month in months[1:]:
            train_mask = month_series.lt(str(month)) & source_mask
            valid_mask = month_series.eq(str(month)) & source_mask
            train_rows = int(train_mask.sum())
            valid_rows = int(valid_mask.sum())
            if train_rows < int(min_train_source_rows) or valid_rows < int(min_valid_source_rows):
                continue
            train = frame.loc[train_mask].copy()
            valid = frame.loc[valid_mask].copy()
            valid_metrics = ft.loc[valid_mask].copy()
            bad_proxy, bad_diag = _proxy_score(
                train=train,
                valid=valid,
                features=features,
                target_train=bad_soft.loc[train_mask],
                top_k=proxy_top_k,
                method=str(proxy_method),
                tail_frac=float(proxy_tail_frac),
            )
            bad_proxy_reset = _safe_numeric(bad_proxy).reset_index(drop=True)
            valid_bad = bad_soft.loc[valid_mask].reset_index(drop=True)
            for utility_target in utility_targets:
                utility_soft = utility_map[utility_target]
                utility_proxy, utility_diag = _proxy_score(
                    train=train,
                    valid=valid,
                    features=features,
                    target_train=utility_soft.loc[train_mask],
                    top_k=proxy_top_k,
                    method=str(proxy_method),
                    tail_frac=float(proxy_tail_frac),
                )
                utility_proxy_reset = _safe_numeric(utility_proxy).reset_index(drop=True)
                valid_target = _target_for_selection(
                    valid_metrics,
                    utility_soft.loc[valid_mask].reset_index(drop=True),
                    valid_bad,
                )
                valid_target_reset = valid_target.reset_index(drop=True)
                valid_metrics_reset = valid_metrics.reset_index(drop=True)
                diag = {
                    "source": str(source_name),
                    "train_rows": train_rows,
                    "valid_rows": valid_rows,
                    "utility_proxy_ic_target": _spearman(utility_proxy_reset, valid_target_reset["target_soft"]),
                    "utility_proxy_ic_u": _spearman(utility_proxy_reset, valid_metrics_reset["u_policy_net"]),
                    "utility_proxy_ic_clean_exec": _spearman(utility_proxy_reset, valid_metrics_reset["clean_exec_actual"]),
                    "utility_proxy_ic_bad": _spearman(utility_proxy_reset, valid_bad),
                    "bad_proxy_ic_bad": _spearman(bad_proxy_reset, valid_bad),
                    "bad_proxy_ic_u": _spearman(bad_proxy_reset, valid_metrics_reset["u_policy_net"]),
                    "bad_proxy_ic_clean_exec": _spearman(bad_proxy_reset, valid_metrics_reset["clean_exec_actual"]),
                    "utility_proxy_top_abs_ic": utility_diag.get("top_abs_ic"),
                    "utility_proxy_mean_top_abs_ic": utility_diag.get("mean_top_abs_ic"),
                    "utility_proxy_features": ",".join(utility_diag.get("features", [])),
                    "bad_proxy_top_abs_ic": bad_diag.get("top_abs_ic"),
                    "bad_proxy_mean_top_abs_ic": bad_diag.get("mean_top_abs_ic"),
                    "bad_proxy_features": ",".join(bad_diag.get("features", [])),
                }
                proxy_ic_rows.append(
                    {
                        "source": str(source_name),
                        "period": str(month),
                        "utility_target": str(utility_target),
                        "proxy_method": str(proxy_method),
                        **diag,
                    }
                )
                target_specs = [spec for spec in specs if spec.utility_target == utility_target]
                for spec in target_specs:
                    source_spec = TwoHeadSpec(
                        name=f"{source_name}::{spec.name}",
                        utility_target=spec.utility_target,
                        score_rule=spec.score_rule,
                        bad_threshold=spec.bad_threshold,
                    )
                    score = _score_from_spec(spec, utility_proxy, bad_proxy)
                    m_rows, w_rows = _monthly_weekly_rows(
                        valid_frame=valid,
                        valid_metrics=valid_metrics,
                        valid_target=valid_target,
                        score=score,
                        spec=source_spec,
                        month=str(month),
                        top_ks=top_ks,
                        diag=diag,
                    )
                    for row in m_rows:
                        row["source"] = str(source_name)
                        row["source_train_rows"] = train_rows
                        row["source_valid_rows"] = valid_rows
                    for row in w_rows:
                        row["source"] = str(source_name)
                        row["source_train_rows"] = train_rows
                        row["source_valid_rows"] = valid_rows
                    monthly_rows.extend(m_rows)
                    weekly_rows.extend(w_rows)

    monthly = pd.DataFrame(monthly_rows)
    weekly = pd.DataFrame(weekly_rows)
    proxy_ic = pd.DataFrame(proxy_ic_rows)
    fit_holdout = _source_fit_holdout_summary(
        monthly=monthly,
        weekly=weekly,
        fit_months=fit_months,
        holdout_month=holdout_month,
        min_week_rows=min_week_rows,
    )
    selected_by_fit = _select_by_source_fit(fit_holdout)

    paths = {
        "source_summary": output_dir / "source_summary.csv",
        "monthly": output_dir / "source_conditioned_two_head_monthly.csv",
        "weekly": output_dir / "source_conditioned_two_head_weekly.csv",
        "proxy_ic": output_dir / "source_conditioned_two_head_proxy_ic.csv",
        "fit_holdout": output_dir / "source_conditioned_two_head_fit_holdout.csv",
        "selected_by_fit": output_dir / "source_conditioned_two_head_selected_by_fit.csv",
        "manifest": output_dir / "manifest.json",
    }
    source_summary.to_csv(paths["source_summary"], index=False)
    monthly.to_csv(paths["monthly"], index=False)
    weekly.to_csv(paths["weekly"], index=False)
    proxy_ic.to_csv(paths["proxy_ic"], index=False)
    fit_holdout.to_csv(paths["fit_holdout"], index=False)
    selected_by_fit.to_csv(paths["selected_by_fit"], index=False)

    manifest = {
        "labels_path": str(labels_path),
        "output_dir": str(output_dir),
        "rows": int(len(frame)),
        "timestamp_min": frame["__ts__"].min(),
        "timestamp_max": frame["__ts__"].max(),
        "symbols": int(frame["__symbol__"].nunique(dropna=True)),
        "feature_dir": str(feature_dir),
        "feature_list_csv": str(feature_list_csv),
        "max_feature_store_features": max_feature_store_features,
        "feature_store": feature_store_report,
        "event_feature_report": event_report,
        "feature_count": int(len(features)),
        "proxy_top_k": int(proxy_top_k),
        "proxy_method": str(proxy_method),
        "proxy_tail_frac": float(proxy_tail_frac),
        "sources": list(sources),
        "utility_targets": list(utility_targets),
        "score_rules": list(score_rules),
        "bad_thresholds": [float(v) for v in bad_thresholds],
        "top_ks": [int(v) for v in top_ks],
        "fit_months": [str(v) for v in fit_months],
        "holdout_month": str(holdout_month),
        "min_train_source_rows": int(min_train_source_rows),
        "min_valid_source_rows": int(min_valid_source_rows),
        "spec_count": int(len(specs)),
        "rows_monthly": int(len(monthly)),
        "rows_weekly": int(len(weekly)),
        "fit_sign_pass_rows": int(fit_holdout["fit_sign_pass"].sum()) if not fit_holdout.empty else 0,
        "holdout_sign_pass_rows": int(fit_holdout["holdout_sign_pass"].sum()) if not fit_holdout.empty else 0,
        "fit_bounded_pass_rows": int(fit_holdout["fit_bounded_pass"].sum()) if not fit_holdout.empty else 0,
        "holdout_bounded_pass_rows": int(fit_holdout["holdout_bounded_pass"].sum()) if not fit_holdout.empty else 0,
        "standalone_holdout_bounded_rows": int(fit_holdout["holdout_bounded_standalone_pass"].sum())
        if not fit_holdout.empty
        else 0,
        "positive_dirty_holdout_rows": int(fit_holdout["positive_dirty_holdout"].sum()) if not fit_holdout.empty else 0,
        "selected_by_fit_rows": int(len(selected_by_fit)),
        "selected_by_fit_holdout_bounded_rows": int(selected_by_fit["holdout_bounded_pass"].sum())
        if not selected_by_fit.empty
        else 0,
        "outputs": {key: str(value) for key, value in paths.items()},
    }
    markdown = _write_markdown(
        output_dir=output_dir,
        source_summary=source_summary,
        fit_holdout=fit_holdout,
        selected_by_fit=selected_by_fit,
        proxy_ic=proxy_ic,
        manifest=manifest,
    )
    manifest["outputs"]["markdown"] = str(markdown)
    paths["manifest"].write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-path", type=Path, default=DEFAULT_LABELS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--feature-list-csv", type=Path, default=DEFAULT_FEATURE_LIST_CSV)
    parser.add_argument("--max-feature-store-features", type=int, default=None)
    parser.add_argument("--proxy-top-k", type=int, default=12)
    parser.add_argument("--proxy-method", choices=["ic", "tail_lift"], default="ic")
    parser.add_argument("--proxy-tail-frac", type=float, default=0.05)
    parser.add_argument("--utility-targets", default=",".join(DEFAULT_UTILITY_TARGETS))
    parser.add_argument("--score-rules", default=",".join(DEFAULT_SCORE_RULES))
    parser.add_argument("--bad-thresholds", default=",".join(str(v) for v in DEFAULT_BAD_THRESHOLDS))
    parser.add_argument("--top-ks", default=",".join(str(v) for v in DEFAULT_TOP_KS))
    parser.add_argument("--fit-months", default=",".join(DEFAULT_FIT_MONTHS))
    parser.add_argument("--holdout-month", default=DEFAULT_HOLDOUT_MONTH)
    parser.add_argument("--min-week-rows", type=int, default=3)
    parser.add_argument("--sources", default=",".join(DEFAULT_SOURCES))
    parser.add_argument("--run-gap-hours", type=float, default=24.0)
    parser.add_argument("--min-train-source-rows", type=int, default=500)
    parser.add_argument("--min-valid-source-rows", type=int, default=100)
    parser.add_argument("--event-feature-store-features", default=",".join(DEFAULT_EVENT_FEATURE_STORE_FEATURES))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = run_ablation(
        labels_path=args.labels_path,
        output_dir=args.output_dir,
        feature_dir=args.feature_dir,
        feature_list_csv=args.feature_list_csv,
        max_feature_store_features=args.max_feature_store_features,
        proxy_top_k=int(args.proxy_top_k),
        proxy_method=str(args.proxy_method),
        proxy_tail_frac=float(args.proxy_tail_frac),
        utility_targets=_parse_csv(args.utility_targets),
        score_rules=_parse_csv(args.score_rules),
        bad_thresholds=_parse_float_csv(args.bad_thresholds),
        top_ks=[int(v) for v in _parse_csv(args.top_ks)],
        fit_months=_parse_csv(args.fit_months),
        holdout_month=str(args.holdout_month),
        min_week_rows=int(args.min_week_rows),
        sources=_parse_csv(args.sources),
        run_gap_hours=float(args.run_gap_hours),
        min_train_source_rows=int(args.min_train_source_rows),
        min_valid_source_rows=int(args.min_valid_source_rows),
        event_feature_store_features=_parse_csv(args.event_feature_store_features),
    )
    summary_keys = [
        "output_dir",
        "rows",
        "feature_count",
        "proxy_method",
        "sources",
        "spec_count",
        "rows_monthly",
        "rows_weekly",
        "fit_sign_pass_rows",
        "holdout_sign_pass_rows",
        "fit_bounded_pass_rows",
        "holdout_bounded_pass_rows",
        "standalone_holdout_bounded_rows",
        "positive_dirty_holdout_rows",
        "selected_by_fit_rows",
        "selected_by_fit_holdout_bounded_rows",
        "outputs",
    ]
    print(json.dumps(_json_safe({key: manifest.get(key) for key in summary_keys}), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
