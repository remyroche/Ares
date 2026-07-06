#!/usr/bin/env python3
"""Analyze repair-ranker profile selection drift.

This report explains why the May-only candidate freeze selected different
profiles than the May+June aggregate ranking, and tests a small set of
predefined selection-rule sensitivities. It does not fit models or write
training artifacts.

The key distinction:

* aggregate rows are descriptive and already include June;
* rule sensitivity selects profiles from May only, then reads out June.
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

from scripts.run_label_quality_proxy_diagnostics import _json_safe, _safe_mean  # noqa: E402
from scripts.select_strict_oos_repair_ranker_profiles import (  # noqa: E402
    DEFAULT_INPUT_DIR,
    MONTHLY_METRIC_COLS,
    PROFILE_KEYS,
    _build_profile_summary,
    _candidate_pool,
    _dedupe_candidates,
    _load_monthly,
    _proxy_family,
    _safe_numeric,
    _table,
)


DEFAULT_MONTHLY = DEFAULT_INPUT_DIR / "strict_oos_repair_ranker_monthly.csv"
DEFAULT_AGGREGATE = DEFAULT_INPUT_DIR / "strict_oos_repair_ranker_aggregate.csv"
DEFAULT_OUTPUT_DIR = Path(
    "data_perp/reports/source_quality_label_walkforward_ablation_v1/"
    "strict_oos_repair_ranker_selection_drift"
)


def _profile_id(frame: pd.DataFrame) -> pd.Series:
    return frame[PROFILE_KEYS].astype(str).agg("|".join, axis=1)


def _load_aggregate(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    missing = sorted(set(PROFILE_KEYS).difference(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    frame["top_frac"] = _safe_numeric(frame["top_frac"]).round(6)
    frame["proxy_family"] = frame["proxy_col"].map(_proxy_family)
    frame["profile_id"] = _profile_id(frame)
    return frame


def _base_pool(monthly: pd.DataFrame, *, selection_month: str, rule: dict[str, Any]) -> pd.DataFrame:
    pool = _candidate_pool(
        monthly,
        selection_month=selection_month,
        min_selected_rows=int(rule.get("min_selected_rows", 5)),
        min_train_class_rows=int(rule.get("min_train_class_rows", 10)),
        min_selection_delta=float(rule.get("min_selection_delta", 0.0)),
        min_selection_mean_u=float(rule.get("min_selection_mean_u", 0.0)),
        min_oracle_capture_delta=float(rule.get("min_oracle_capture_delta", 0.0)),
        max_bad_mae_rate=float(rule.get("max_bad_mae_rate", 0.75)),
        max_bad_mae_excess=float(rule.get("max_bad_mae_excess", 0.15)),
        max_timeout_excess=float(rule.get("max_timeout_excess", 0.15)),
    )
    if pool.empty:
        return pool
    include_families = set(rule.get("include_proxy_families", []) or [])
    if include_families:
        pool = pool[pool["proxy_family"].isin(include_families)].copy()
    exclude_proxy_cols = set(rule.get("exclude_proxy_cols", []) or [])
    if exclude_proxy_cols:
        pool = pool[~pool["proxy_col"].isin(exclude_proxy_cols)].copy()
    include_proxy_cols = set(rule.get("include_proxy_cols", []) or [])
    if include_proxy_cols:
        pool = pool[pool["proxy_col"].isin(include_proxy_cols)].copy()
    return pool.reset_index(drop=True)


def _default_rules() -> list[dict[str, Any]]:
    base = {
        "min_selected_rows": 5,
        "min_train_class_rows": 10,
        "min_selection_delta": 0.0,
        "min_selection_mean_u": 0.0,
        "min_oracle_capture_delta": 0.0,
        "max_bad_mae_rate": 0.75,
        "max_bad_mae_excess": 0.15,
        "max_timeout_excess": 0.15,
        "min_holdout_oracle_capture": 0.05,
    }
    return [
        {"rule_name": "default_top3", "max_profiles": 3, **base},
        {"rule_name": "default_top6", "max_profiles": 6, **base},
        {
            "rule_name": "oof_meta_family_top3",
            "max_profiles": 3,
            "include_proxy_families": ["oof_meta_pair"],
            **base,
        },
        {
            "rule_name": "exclude_base_model_score_pct_top3",
            "max_profiles": 3,
            "exclude_proxy_cols": ["base_model_score_pct"],
            **base,
        },
        {
            "rule_name": "low_bad_mae_top3",
            "max_profiles": 3,
            "max_bad_mae_rate": 0.50,
            **{key: value for key, value in base.items() if key != "max_bad_mae_rate"},
        },
        {
            "rule_name": "strict_delta_top3",
            "max_profiles": 3,
            "min_selection_delta": 0.01,
            **{key: value for key, value in base.items() if key != "min_selection_delta"},
        },
    ]


def _summarize_rule(
    monthly: pd.DataFrame,
    *,
    selection_month: str,
    holdout_month: str,
    rule: dict[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame]:
    pool = _base_pool(monthly, selection_month=selection_month, rule=rule)
    selected = _dedupe_candidates(pool, max_profiles=int(rule["max_profiles"]))
    summary = _build_profile_summary(
        selected,
        monthly,
        selection_month=selection_month,
        holdout_month=holdout_month,
        min_holdout_oracle_capture=float(rule.get("min_holdout_oracle_capture", 0.05)),
        max_bad_mae_excess=float(rule.get("max_bad_mae_excess", 0.15)),
        max_timeout_excess=float(rule.get("max_timeout_excess", 0.15)),
    )
    status_counts = (
        summary["holdout_status"].value_counts().to_dict()
        if not summary.empty and "holdout_status" in summary.columns
        else {}
    )
    row = {
        "rule_name": rule["rule_name"],
        "pool_rows": int(len(pool)),
        "selected_profiles": int(len(selected)),
        "survives_holdout": int(status_counts.get("survives_holdout", 0)),
        "beats_proxy_but_fails_guard": int(status_counts.get("beats_proxy_but_fails_guard", 0)),
        "fails_holdout_delta": int(status_counts.get("fails_holdout_delta", 0)),
        "holdout_mean_repair_u": _safe_mean(summary.get("holdout_repair_mean_u")),
        "holdout_mean_proxy_u": _safe_mean(summary.get("holdout_proxy_mean_u")),
        "holdout_mean_delta_u_vs_proxy": _safe_mean(summary.get("holdout_repair_delta_mean_u_vs_proxy")),
        "holdout_mean_oracle_capture": _safe_mean(summary.get("holdout_repair_oracle_capture_at_k")),
        "selected_profile_ids": "; ".join(_profile_id(selected).tolist()) if not selected.empty else "",
    }
    if not summary.empty:
        summary = summary.copy()
        summary.insert(0, "rule_name", rule["rule_name"])
    return row, summary


def _exclusion_reasons(row: pd.Series, rule: dict[str, Any]) -> str:
    reasons: list[str] = []
    if float(row.get("selected_rows", 0.0) or 0.0) < float(rule.get("min_selected_rows", 5)):
        reasons.append("selected_rows")
    if float(row.get("train_positive_events", 0.0) or 0.0) < float(rule.get("min_train_class_rows", 10)):
        reasons.append("train_positive_events")
    if float(row.get("train_negative_events", 0.0) or 0.0) < float(rule.get("min_train_class_rows", 10)):
        reasons.append("train_negative_events")
    if float(row.get("repair_delta_mean_u_vs_proxy", -np.inf) or -np.inf) < float(rule.get("min_selection_delta", 0.0)):
        reasons.append("selection_delta")
    if float(row.get("repair_mean_u", -np.inf) or -np.inf) < float(rule.get("min_selection_mean_u", 0.0)):
        reasons.append("selection_mean_u")
    if float(row.get("repair_delta_oracle_capture_at_k", -np.inf) or -np.inf) < float(rule.get("min_oracle_capture_delta", 0.0)):
        reasons.append("oracle_capture_delta")
    if float(row.get("repair_bad_mae_1r_rate", np.inf) or np.inf) > float(rule.get("max_bad_mae_rate", 0.75)):
        reasons.append("bad_mae_rate")
    bad_excess = float(row.get("repair_bad_mae_1r_rate", 0.0) or 0.0) - float(row.get("proxy_bad_mae_1r_rate", 0.0) or 0.0)
    if bad_excess > float(rule.get("max_bad_mae_excess", 0.15)):
        reasons.append("bad_mae_excess")
    timeout_excess = float(row.get("repair_timeout_or_slow_holding_rate", 0.0) or 0.0) - float(row.get("proxy_timeout_or_slow_holding_rate", 0.0) or 0.0)
    if timeout_excess > float(rule.get("max_timeout_excess", 0.15)):
        reasons.append("timeout_excess")
    return ",".join(reasons)


def _aggregate_profile_drift(
    *,
    monthly: pd.DataFrame,
    aggregate: pd.DataFrame,
    selection_month: str,
    holdout_month: str,
    default_rule: dict[str, Any],
    top_n: int,
) -> pd.DataFrame:
    may = monthly[monthly["period"].eq(selection_month)].copy()
    holdout = monthly[monthly["period"].eq(holdout_month)].copy()
    may["profile_id"] = _profile_id(may)
    holdout["profile_id"] = _profile_id(holdout)
    pool = _base_pool(monthly, selection_month=selection_month, rule=default_rule)
    default_selected = _dedupe_candidates(pool, max_profiles=int(default_rule["max_profiles"]))
    pool_ids = set(_profile_id(pool).tolist()) if not pool.empty else set()
    selected_ids = set(_profile_id(default_selected).tolist()) if not default_selected.empty else set()

    reviewable = aggregate[aggregate["months"].ge(2)].copy()
    reviewable = reviewable.sort_values(
        ["repair_delta_mean_u_vs_proxy", "repair_mean_u"],
        ascending=[False, False],
        kind="mergesort",
    ).head(top_n).copy()
    reviewable["profile_id"] = _profile_id(reviewable)
    rows: list[dict[str, Any]] = []
    may_map = may.set_index("profile_id")
    holdout_map = holdout.set_index("profile_id")
    for rank, row in enumerate(reviewable.to_dict("records"), start=1):
        profile_id = str(row["profile_id"])
        may_row = may_map.loc[profile_id] if profile_id in may_map.index else pd.Series(dtype=object)
        holdout_row = holdout_map.loc[profile_id] if profile_id in holdout_map.index else pd.Series(dtype=object)
        if isinstance(may_row, pd.DataFrame):
            may_row = may_row.iloc[0]
        if isinstance(holdout_row, pd.DataFrame):
            holdout_row = holdout_row.iloc[0]
        rows.append(
            {
                "aggregate_rank": rank,
                **{key: row.get(key) for key in PROFILE_KEYS},
                "proxy_family": row.get("proxy_family"),
                "aggregate_decision": row.get("decision"),
                "agg_repair_mean_u": row.get("repair_mean_u"),
                "agg_proxy_mean_u": row.get("proxy_mean_u"),
                "agg_delta_u_vs_proxy": row.get("repair_delta_mean_u_vs_proxy"),
                "agg_worst_month_u": row.get("repair_worst_month_u"),
                "may_in_default_pool": profile_id in pool_ids,
                "may_selected_default_top3": profile_id in selected_ids,
                "may_exclusion_reasons": "" if profile_id in pool_ids else _exclusion_reasons(may_row, default_rule),
                "may_repair_mean_u": may_row.get("repair_mean_u", np.nan),
                "may_proxy_mean_u": may_row.get("proxy_mean_u", np.nan),
                "may_delta_u_vs_proxy": may_row.get("repair_delta_mean_u_vs_proxy", np.nan),
                "may_oracle_capture": may_row.get("repair_oracle_capture_at_k", np.nan),
                "may_proxy_oracle_capture": may_row.get("proxy_oracle_capture_at_k", np.nan),
                "may_bad_mae": may_row.get("repair_bad_mae_1r_rate", np.nan),
                "june_repair_mean_u": holdout_row.get("repair_mean_u", np.nan),
                "june_proxy_mean_u": holdout_row.get("proxy_mean_u", np.nan),
                "june_delta_u_vs_proxy": holdout_row.get("repair_delta_mean_u_vs_proxy", np.nan),
                "june_oracle_capture": holdout_row.get("repair_oracle_capture_at_k", np.nan),
                "june_proxy_oracle_capture": holdout_row.get("proxy_oracle_capture_at_k", np.nan),
                "june_bad_mae": holdout_row.get("repair_bad_mae_1r_rate", np.nan),
            }
        )
    return pd.DataFrame(rows)


def _family_month_summary(monthly: pd.DataFrame) -> pd.DataFrame:
    work = monthly.copy()
    work["proxy_family"] = work["proxy_col"].map(_proxy_family)
    work["bad_mae_excess"] = work["repair_bad_mae_1r_rate"] - work["proxy_bad_mae_1r_rate"]
    work["timeout_excess"] = (
        work["repair_timeout_or_slow_holding_rate"] - work["proxy_timeout_or_slow_holding_rate"]
    )
    rows: list[dict[str, Any]] = []
    for key, group in work.groupby(["period", "proxy_family"], dropna=False, observed=True):
        period, proxy_family = key
        rows.append(
            {
                "period": period,
                "proxy_family": proxy_family,
                "rows": int(len(group)),
                "mean_repair_u": _safe_mean(group["repair_mean_u"]),
                "mean_proxy_u": _safe_mean(group["proxy_mean_u"]),
                "mean_delta_u_vs_proxy": _safe_mean(group["repair_delta_mean_u_vs_proxy"]),
                "positive_repair_rate": _safe_mean(group["repair_mean_u"] > 0.0),
                "positive_delta_rate": _safe_mean(group["repair_delta_mean_u_vs_proxy"] > 0.0),
                "mean_oracle_capture_delta": _safe_mean(group["repair_delta_oracle_capture_at_k"]),
                "mean_bad_mae_excess": _safe_mean(group["bad_mae_excess"]),
                "mean_timeout_excess": _safe_mean(group["timeout_excess"]),
            }
        )
    return pd.DataFrame(rows).sort_values(["period", "mean_delta_u_vs_proxy"], ascending=[True, False])


def _write_report(
    *,
    output_dir: Path,
    manifest: dict[str, Any],
    rule_summary: pd.DataFrame,
    rule_profiles: pd.DataFrame,
    aggregate_drift: pd.DataFrame,
    family_summary: pd.DataFrame,
) -> Path:
    path = output_dir / "strict_oos_repair_ranker_selection_drift_report.md"
    rule_cols = [
        "rule_name",
        "pool_rows",
        "selected_profiles",
        "survives_holdout",
        "beats_proxy_but_fails_guard",
        "fails_holdout_delta",
        "holdout_mean_repair_u",
        "holdout_mean_proxy_u",
        "holdout_mean_delta_u_vs_proxy",
        "holdout_mean_oracle_capture",
    ]
    profile_cols = [
        "rule_name",
        "holdout_status",
        "source_bucket",
        "proxy_col",
        "top_frac",
        "feature_mode",
        "selection_method",
        "selection_repair_mean_u",
        "selection_delta_mean_u_vs_proxy",
        "holdout_repair_mean_u",
        "holdout_proxy_mean_u",
        "holdout_repair_delta_mean_u_vs_proxy",
        "holdout_repair_oracle_capture_at_k",
        "holdout_failure_reasons",
    ]
    drift_cols = [
        "aggregate_rank",
        "source_bucket",
        "proxy_col",
        "top_frac",
        "feature_mode",
        "selection_method",
        "agg_delta_u_vs_proxy",
        "may_in_default_pool",
        "may_selected_default_top3",
        "may_delta_u_vs_proxy",
        "june_delta_u_vs_proxy",
        "june_repair_mean_u",
        "june_oracle_capture",
    ]
    family_cols = [
        "period",
        "proxy_family",
        "rows",
        "mean_repair_u",
        "mean_proxy_u",
        "mean_delta_u_vs_proxy",
        "positive_repair_rate",
        "positive_delta_rate",
        "mean_oracle_capture_delta",
    ]
    lines = [
        "# Strict OOS Repair Ranker Selection Drift",
        "",
        "Diagnostic report explaining May-only selection drift versus the May+June aggregate repair-ranker ranking.",
        "",
        "## Scope",
        "",
        f"- Monthly ledger: `{manifest['monthly_path']}`",
        f"- Aggregate ledger: `{manifest['aggregate_path']}`",
        f"- Selection month: `{manifest['selection_month']}`",
        f"- Holdout month: `{manifest['holdout_month']}`",
        "",
        "## Rule Sensitivity",
        "",
        _table(rule_summary, rule_cols, limit=None),
        "",
        "## Selected Profiles By Rule",
        "",
        _table(rule_profiles, profile_cols, limit=80),
        "",
        "## Aggregate Top Profiles Versus May Selection",
        "",
        _table(aggregate_drift, drift_cols, limit=40),
        "",
        "## Proxy Family Month Summary",
        "",
        _table(family_summary, family_cols, limit=40),
        "",
        "## Interpretation",
        "",
        "- Default May selection picked `base_model_score_pct` profiles because they had the highest May selection score, but those profiles remained negative in June and captured no oracle top-k rows.",
        "- Excluding `base_model_score_pct` or constraining to the OOF/meta family exposes one June-surviving profile, but this rule is post-hoc and must not be promoted without later untouched validation.",
        "- The surviving profile is useful as a hypothesis for a pre-registered next-period test, not as current deployment evidence.",
        "",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_report(
    *,
    monthly_path: Path,
    aggregate_path: Path,
    output_dir: Path,
    selection_month: str,
    holdout_month: str,
    aggregate_top_n: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    monthly = _load_monthly(monthly_path)
    aggregate = _load_aggregate(aggregate_path)
    rules = _default_rules()
    rule_rows: list[dict[str, Any]] = []
    profile_frames: list[pd.DataFrame] = []
    for rule in rules:
        row, profiles = _summarize_rule(
            monthly,
            selection_month=selection_month,
            holdout_month=holdout_month,
            rule=rule,
        )
        rule_rows.append(row)
        if not profiles.empty:
            profile_frames.append(profiles)

    rule_summary = pd.DataFrame(rule_rows)
    rule_profiles = pd.concat(profile_frames, ignore_index=True) if profile_frames else pd.DataFrame()
    aggregate_drift = _aggregate_profile_drift(
        monthly=monthly,
        aggregate=aggregate,
        selection_month=selection_month,
        holdout_month=holdout_month,
        default_rule=rules[0],
        top_n=aggregate_top_n,
    )
    family_summary = _family_month_summary(monthly)

    paths = {
        "rule_summary": output_dir / "strict_oos_repair_ranker_selection_rule_summary.csv",
        "rule_profiles": output_dir / "strict_oos_repair_ranker_selection_rule_profiles.csv",
        "aggregate_drift": output_dir / "strict_oos_repair_ranker_aggregate_selection_drift.csv",
        "family_summary": output_dir / "strict_oos_repair_ranker_proxy_family_month_summary.csv",
        "manifest": output_dir / "manifest.json",
    }
    rule_summary.to_csv(paths["rule_summary"], index=False)
    rule_profiles.to_csv(paths["rule_profiles"], index=False)
    aggregate_drift.to_csv(paths["aggregate_drift"], index=False)
    family_summary.to_csv(paths["family_summary"], index=False)

    manifest = {
        "scope": "strict_oos_repair_ranker_selection_drift",
        "monthly_path": str(monthly_path),
        "aggregate_path": str(aggregate_path),
        "output_dir": str(output_dir),
        "selection_month": selection_month,
        "holdout_month": holdout_month,
        "aggregate_top_n": int(aggregate_top_n),
        "rule_count": int(len(rules)),
        "rule_summary_rows": int(len(rule_summary)),
        "rule_profile_rows": int(len(rule_profiles)),
        "surviving_rules": rule_summary.loc[
            rule_summary["survives_holdout"].fillna(0).astype(float).gt(0),
            "rule_name",
        ].tolist()
        if not rule_summary.empty
        else [],
        "outputs": {key: str(value) for key, value in paths.items()},
    }
    paths["manifest"].write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")
    report = _write_report(
        output_dir=output_dir,
        manifest=manifest,
        rule_summary=rule_summary,
        rule_profiles=rule_profiles,
        aggregate_drift=aggregate_drift,
        family_summary=family_summary,
    )
    manifest["outputs"]["markdown"] = str(report)
    paths["manifest"].write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--monthly-path", type=Path, default=DEFAULT_MONTHLY)
    parser.add_argument("--aggregate-path", type=Path, default=DEFAULT_AGGREGATE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--selection-month", type=str, default="2026-05")
    parser.add_argument("--holdout-month", type=str, default="2026-06")
    parser.add_argument("--aggregate-top-n", type=int, default=40)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = run_report(
        monthly_path=args.monthly_path,
        aggregate_path=args.aggregate_path,
        output_dir=args.output_dir,
        selection_month=args.selection_month,
        holdout_month=args.holdout_month,
        aggregate_top_n=args.aggregate_top_n,
    )
    print(json.dumps(_json_safe(manifest), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
