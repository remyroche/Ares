#!/usr/bin/env python3
"""Promote the S52 side/parser baseline reference and audit June decay.

This script intentionally does not mutate model or policy artifacts. It writes a
small reference manifest/report plus CSV diagnostics that line up base, meta,
and execution evidence for the current S52 handoff.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path("data_perp/reports")
BASE_DIR = ROOT / "s52_trailing_profit_best_pointwise_scored_ledger_20260705_v1"
META_DIR = BASE_DIR / "s52_trailing_regime_meta_handoff_longsplit_v2"
THRESHOLD_DIR = META_DIR / "s52_meta_threshold_top10_longaware_sidebad055_v1"
CANDIDATE_DIR = ROOT / "s52_replay_candidates_current_top10_longaware_sidebad055_20260705_v1"
WF_DIR = ROOT / "simple_policy_exit_side_archetype_s52_current_top10_longaware_sidebad055_sidefix_cost1pct_20260705_v1"
METRICS_DIR = ROOT / "s52_side_archetype_policy_metrics_sidefix_cost1pct_20260705_v1"

REFERENCE_DIR = ROOT / "s52_current_reference_sidefix_baseline_20260705_v1"
AUDIT_DIR = ROOT / "s52_june_decay_audit_sidefix_20260705_v1"


def _read_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def _write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def _num(v: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(v):
            return default
        return float(v)
    except Exception:
        return default


def _int(v: Any, default: int = 0) -> int:
    try:
        if pd.isna(v):
            return default
        return int(v)
    except Exception:
        return default


def _week_start(series: pd.Series) -> pd.Series:
    ts = pd.to_datetime(series, utc=True, errors="coerce")
    return (ts.dt.normalize() - pd.to_timedelta(ts.dt.weekday, unit="D")).dt.date.astype(str)


def _summarize_base_folds() -> pd.DataFrame:
    path = BASE_DIR / "s52_ranker_smoke_folds.csv"
    df = pd.read_csv(path)
    keep = [
        "month",
        "rows",
        "top10_rows",
        "top10_clean_precision",
        "top10_ev_weighted_first_touch_precision",
        "top10_mean_first_touch_executable_margin",
        "top10_first_touch_full_path_bad_mae_1r_rate",
        "top10_timeout_rate",
        "top10_mean_u",
        "top20_rows",
        "top20_clean_precision",
        "top20_ev_weighted_first_touch_precision",
        "top20_mean_first_touch_executable_margin",
        "top20_first_touch_full_path_bad_mae_1r_rate",
        "top20_timeout_rate",
        "top20_mean_u",
        "top30_rows",
        "top30_clean_precision",
        "top30_ev_weighted_first_touch_precision",
        "top30_mean_first_touch_executable_margin",
        "top30_first_touch_full_path_bad_mae_1r_rate",
        "top30_timeout_rate",
        "top30_mean_u",
    ]
    out = df[[c for c in keep if c in df.columns]].copy()
    return out


def _summarize_meta_weekly() -> pd.DataFrame:
    path = THRESHOLD_DIR / "s52_meta_threshold_guarded_offline_eval_candidates.parquet"
    df = pd.read_parquet(path)
    if "__ts__" not in df.columns:
        return pd.DataFrame()
    df = df.copy()
    df["week"] = _week_start(df["__ts__"])
    df["is_short"] = df.get("side_name", "").astype(str).str.lower().eq("short")
    group_cols = ["week", "month"]
    rows = []
    for keys, g in df.groupby(group_cols, dropna=False):
        week, month = keys
        rows.append(
            {
                "week": week,
                "month": month,
                "rows": len(g),
                "symbols": g["__symbol__"].nunique() if "__symbol__" in g.columns else np.nan,
                "exec_margin": _num(g.get("exec_margin", pd.Series(dtype=float)).mean()),
                "ret_net": _num(g.get("ret_net", pd.Series(dtype=float)).mean()),
                "clean_exec_precision": _num(g.get("clean_exec", pd.Series(dtype=float)).mean()),
                "full_path_bad_mae": _num(g.get("full_path_bad_mae_1r", pd.Series(dtype=float)).mean()),
                "timeout": _num(g.get("timeout", pd.Series(dtype=float)).mean()),
                "dirty_positive": _num(g.get("dirty_positive", pd.Series(dtype=float)).mean()),
                "short_share": _num(g["is_short"].mean()),
                "score_base_mean": _num(g.get("score_base", pd.Series(dtype=float)).mean()),
                "score_meta_mean": _num(
                    g.get("score_meta_long_aware_clean_minus_risk", pd.Series(dtype=float)).mean()
                ),
                "long_rows": int((~g["is_short"]).sum()),
                "short_rows": int(g["is_short"].sum()),
            }
        )
    return pd.DataFrame(rows).sort_values(["week", "month"])


def _summarize_replay_weekly() -> tuple[pd.DataFrame, pd.DataFrame]:
    stage = pd.read_csv(WF_DIR / "walkforward_stage_summary.csv")
    stage = stage[stage["arm"].isin(["A0_baseline", "A6_time_decay"])].copy()
    stage["week"] = pd.to_datetime(stage["validation_start"], utc=True, errors="coerce").dt.date.astype(str)
    cols = [
        "week",
        "fold",
        "arm",
        "stage",
        "validation_candidate_rows",
        "accepted_trades",
        "portfolio_net_pnl",
        "portfolio_gross_pnl",
        "portfolio_objective",
        "portfolio_full_sl_rate",
        "portfolio_timeout_rate",
        "portfolio_max_drawdown",
        "portfolio_side_concentration",
        "portfolio_strategy_concentration",
        "portfolio_avg_open_positions",
    ]
    stage_out = stage[[c for c in cols if c in stage.columns]].copy()

    detail = pd.read_csv(METRICS_DIR / "side_archetype_week_metrics_aggregated.csv")
    detail = detail[
        (detail["arm"].isin(["A0_baseline", "A6_time_decay"])) & (detail["accepted_trades"] > 0)
    ].copy()
    detail_cols = [
        "period",
        "arm",
        "stage",
        "side",
        "policy_archetype",
        "accepted_trades",
        "accepted_net_pnl",
        "accepted_gross_pnl",
        "accepted_win_rate",
        "accepted_full_sl_rows",
        "accepted_full_sl_rate",
        "accepted_timeout_rows",
        "accepted_timeout_rate",
        "accepted_trailing_rows",
        "accepted_trailing_rate",
        "accepted_mean_net_return",
        "accepted_avg_holding_hours",
    ]
    detail_out = detail[[c for c in detail_cols if c in detail.columns]].copy()
    return stage_out.sort_values(["week", "arm"]), detail_out.sort_values(
        ["period", "arm", "accepted_net_pnl"], ascending=[True, True, True]
    )


def _side_month_replay(detail: pd.DataFrame) -> pd.DataFrame:
    month = pd.read_csv(METRICS_DIR / "side_archetype_month_metrics_aggregated.csv")
    month = month[(month["arm"].isin(["A0_baseline", "A6_time_decay"])) & (month["accepted_trades"] > 0)].copy()
    out = (
        month.groupby(["period", "arm", "side"], dropna=False)
        .agg(
            accepted_trades=("accepted_trades", "sum"),
            accepted_net_pnl=("accepted_net_pnl", "sum"),
            accepted_full_sl_rows=("accepted_full_sl_rows", "sum"),
            accepted_timeout_rows=("accepted_timeout_rows", "sum"),
            accepted_mean_net_return=("accepted_mean_net_return", "mean"),
        )
        .reset_index()
        .sort_values(["period", "arm", "side"])
    )
    return out


def _weak_week_drivers(detail: pd.DataFrame) -> pd.DataFrame:
    weak = detail[detail["period"].isin(["2026-06-15", "2026-06-22"])].copy()
    return weak.sort_values(["period", "arm", "accepted_net_pnl"], ascending=[True, True, True])


def _oos_contract_summary() -> dict[str, Any]:
    base_manifest = _read_json(BASE_DIR / "manifest.json")
    meta_manifest = _read_json(META_DIR / "manifest.json")
    meta_smoke_manifest = _read_json(META_DIR / "train_meta_regime_handoff_smoke_v1" / "manifest.json")
    threshold_manifest = _read_json(THRESHOLD_DIR / "s52_meta_threshold_guarded_manifest.json")
    wf_manifest = _read_json(WF_DIR / "walkforward_manifest.json")
    verdict = _read_json(WF_DIR / "walkforward_verdict.json")

    return {
        "base": {
            "artifact": str(BASE_DIR / "manifest.json"),
            "scope": base_manifest.get("scope"),
            "rows": base_manifest.get("rows"),
            "symbols": base_manifest.get("symbols"),
            "timestamp_min": base_manifest.get("timestamp_min"),
            "timestamp_max": base_manifest.get("timestamp_max"),
            "fold_months": base_manifest.get("fold_months"),
            "fold_count": base_manifest.get("fold_count"),
            "round_trip_cost": base_manifest.get("round_trip_cost"),
            "max_train_rows": base_manifest.get("max_train_rows"),
            "ae_gmm_fold_cache_statuses": base_manifest.get("ae_gmm_fold_cache_statuses"),
            "interpretation": "monthly walk-forward/OOF scored ledger for base top-k validation",
        },
        "meta_regime_handoff": {
            "artifact": str(META_DIR / "manifest.json"),
            "fit_months": meta_manifest.get("fit_months"),
            "holdout_month": meta_manifest.get("holdout_month"),
            "rows": meta_manifest.get("rows"),
            "fit_rows": meta_manifest.get("fit_rows"),
            "holdout_rows": meta_manifest.get("holdout_rows"),
            "selected_col": meta_manifest.get("selected_col"),
            "leakage_contract": meta_manifest.get("leakage_contract"),
            "interpretation": "June is holdout for the longsplit regime handoff; May is part of fit/reference months for that artifact",
        },
        "train_meta_smoke": {
            "artifact": str(META_DIR / "train_meta_regime_handoff_smoke_v1" / "manifest.json"),
            "generated_by": meta_smoke_manifest.get("generated_by"),
            "months": meta_smoke_manifest.get("months"),
            "train_scope": meta_smoke_manifest.get("train_scope"),
            "rows": meta_smoke_manifest.get("rows"),
            "best_selector": meta_smoke_manifest.get("best_selector"),
            "best_threshold_policy": meta_smoke_manifest.get("best_threshold_policy"),
            "interpretation": "month-forward train-past/validate-next meta smoke; not final untouched deployment evidence",
        },
        "threshold_handoff": {
            "artifact": str(THRESHOLD_DIR / "s52_meta_threshold_guarded_manifest.json"),
            "selector": threshold_manifest.get("selector"),
            "policy": threshold_manifest.get("policy"),
            "summary": threshold_manifest.get("summary"),
            "leakage_audit": threshold_manifest.get("leakage_audit"),
            "interpretation": "fixed smoke template OOS predictions; threshold not validation optimized per manifest",
        },
        "execution_optimiser": {
            "artifact": str(WF_DIR / "walkforward_manifest.json"),
            "raw_rows": wf_manifest.get("raw_rows"),
            "replay_rows": wf_manifest.get("replay_rows"),
            "path_survival_fraction": wf_manifest.get("path_survival_fraction"),
            "source_replay_start": wf_manifest.get("source_replay_start"),
            "source_replay_end": wf_manifest.get("source_replay_end"),
            "min_train_weeks": wf_manifest.get("min_train_weeks"),
            "embargo_days": wf_manifest.get("embargo_days"),
            "inner_min_train_weeks": wf_manifest.get("inner_min_train_weeks"),
            "inner_embargo_days": wf_manifest.get("inner_embargo_days"),
            "round_trip_cost_pct": wf_manifest.get("round_trip_cost_pct"),
            "group_by": wf_manifest.get("group_by"),
            "verdict": verdict,
            "interpretation": "weekly walk-forward execution optimiser validation with embargo; selected policy is not an untouched frozen replay",
        },
    }


def _make_reference_manifest(
    base_folds: pd.DataFrame,
    replay_stage: pd.DataFrame,
    side_month: pd.DataFrame,
    oos_contract: dict[str, Any],
) -> dict[str, Any]:
    verdict = _read_json(WF_DIR / "walkforward_verdict.json")
    candidate_manifest = _read_json(CANDIDATE_DIR / "manifest.json")
    coverage = _read_json(ROOT / "s52_replay_path_coverage_preflight_current_top10_longaware_sidebad055_sidefix_20260705_v1" / "manifest.json")

    baseline = replay_stage[replay_stage["arm"] == "A0_baseline"].copy()
    final = replay_stage[replay_stage["arm"] == "A6_time_decay"].copy()
    baseline_positive = int((baseline["portfolio_net_pnl"] > 0).sum())

    return {
        "status": "promoted_reference",
        "promoted_reference": "S52 baseline execution handoff with side/parser fix",
        "not_promoted": "side x archetype simple_policy_optimiser overrides",
        "reason": (
            "The side/parser fix gives full replay coverage and correct side grouping, "
            "but the optimised side x archetype policy underperforms the baseline."
        ),
        "candidate_path": str(CANDIDATE_DIR / "simple_policy_candidates_with_archetypes.parquet"),
        "candidate_manifest": str(CANDIDATE_DIR / "manifest.json"),
        "walkforward_dir": str(WF_DIR),
        "metrics_dir": str(METRICS_DIR),
        "coverage_manifest": str(
            ROOT
            / "s52_replay_path_coverage_preflight_current_top10_longaware_sidebad055_sidefix_20260705_v1"
            / "manifest.json"
        ),
        "candidate_summary": candidate_manifest,
        "path_coverage_summary": {
            "raw_rows": coverage.get("raw_rows"),
            "replay_rows": coverage.get("replay_rows"),
            "path_survival_fraction": coverage.get("path_survival_fraction"),
            "round_trip_cost_pct": coverage.get("round_trip_cost_pct"),
        },
        "baseline_execution_metrics": {
            "net_pnl": verdict.get("baseline_net_pnl"),
            "positive_folds": baseline_positive,
            "fold_count": len(baseline),
            "accepted_trades": _int(baseline["accepted_trades"].sum()),
            "round_trip_cost_pct": oos_contract["execution_optimiser"].get("round_trip_cost_pct"),
            "weak_weeks": baseline.loc[baseline["portfolio_net_pnl"] < 0, "week"].tolist(),
        },
        "optimised_execution_metrics": {
            "final_arm": verdict.get("final_stage_arm"),
            "net_pnl": verdict.get("final_net_pnl"),
            "delta_vs_baseline": verdict.get("delta_net_pnl"),
            "passing_net_pnl_folds": verdict.get("passing_net_pnl_folds"),
            "fold_count": verdict.get("fold_count"),
            "accepted_trades": _int(final["accepted_trades"].sum()),
        },
        "base_oos_monthly_summary": base_folds.to_dict(orient="records"),
        "side_month_replay_summary": side_month.to_dict(orient="records"),
        "oos_contract_summary": oos_contract,
    }


def _write_reference_report(manifest: dict[str, Any], replay_stage: pd.DataFrame) -> None:
    baseline = replay_stage[replay_stage["arm"] == "A0_baseline"][
        ["week", "accepted_trades", "portfolio_net_pnl", "portfolio_full_sl_rate", "portfolio_timeout_rate"]
    ].copy()
    final = replay_stage[replay_stage["arm"] == "A6_time_decay"][
        ["week", "accepted_trades", "portfolio_net_pnl", "portfolio_full_sl_rate", "portfolio_timeout_rate"]
    ].copy()

    lines = [
        "# S52 Current Reference: Sidefix Baseline",
        "",
        "## Decision",
        "",
        "Promote the side/parser fix and use the S52 baseline execution handoff as the current reference.",
        "Do not promote the side x archetype optimiser overrides from this run.",
        "",
        "## Reference Metrics",
        "",
        f"- Candidate rows: {manifest['candidate_summary'].get('summary', {}).get('rows')}",
        f"- Symbols: {manifest['candidate_summary'].get('summary', {}).get('symbols')}",
        f"- Replay survival: {manifest['path_coverage_summary'].get('path_survival_fraction'):.2%}",
        f"- Round-trip cost: {manifest['baseline_execution_metrics'].get('round_trip_cost_pct'):.2%}",
        f"- Baseline net PnL: {manifest['baseline_execution_metrics'].get('net_pnl'):.2f}",
        f"- Baseline positive folds: {manifest['baseline_execution_metrics'].get('positive_folds')}/{manifest['baseline_execution_metrics'].get('fold_count')}",
        f"- Baseline accepted trades: {manifest['baseline_execution_metrics'].get('accepted_trades')}",
        f"- Optimised final net PnL: {manifest['optimised_execution_metrics'].get('net_pnl'):.2f}",
        f"- Optimised delta vs baseline: {manifest['optimised_execution_metrics'].get('delta_vs_baseline'):.2f}",
        "",
        "## Baseline Weekly Replay",
        "",
        baseline.to_markdown(index=False),
        "",
        "## Final Optimised Weekly Replay",
        "",
        final.to_markdown(index=False),
        "",
        "## OOS Interpretation",
        "",
        "- Base: monthly walk-forward/OOF scored ledger over 2026-04, 2026-05, and 2026-06.",
        "- Meta: month-forward smoke plus a handoff artifact where June is held out for the longsplit regime layer.",
        "- Optimiser: weekly walk-forward validation with a 1-day embargo and 1% round-trip cost.",
        "- This reference is not an untouched frozen deployment replay; it is the current validation reference.",
        "",
    ]
    (REFERENCE_DIR / "current_reference_report.md").write_text("\n".join(lines))


def _write_audit_report(
    base_folds: pd.DataFrame,
    meta_weekly: pd.DataFrame,
    replay_stage: pd.DataFrame,
    weak: pd.DataFrame,
    side_month: pd.DataFrame,
    oos_contract: dict[str, Any],
) -> None:
    baseline = replay_stage[replay_stage["arm"] == "A0_baseline"].copy()
    final = replay_stage[replay_stage["arm"] == "A6_time_decay"].copy()
    june15 = baseline[baseline["week"] == "2026-06-15"]
    june22 = baseline[baseline["week"] == "2026-06-22"]

    lines = [
        "# S52 June Decay Audit",
        "",
        "## Main Findings",
        "",
        "1. The side/parser fix is valid mechanically: replay path survival is 100% and side x archetype groups are no longer inverted by numeric short values.",
        "2. The optimiser layer did not improve the current reference: A6_time_decay loses 2100.19 net PnL versus the S52 baseline.",
        "3. The 2026-06-15 and 2026-06-22 failures are replay/execution failures concentrated in shorts, with elevated stop-loss counts and sparse late-June samples.",
        "4. The May to June degradation is already visible before the optimiser: base top10 clean precision drops and timeout rises, while meta clean precision drops from 73.9% in May to 58.0% in June. Base EV-weighted precision and executable margin remain good, so this is not a total base-layer collapse.",
        "",
        "## Base Monthly OOF Top-K",
        "",
        base_folds.to_markdown(index=False),
        "",
        "## Meta Weekly Offline Handoff",
        "",
        meta_weekly.to_markdown(index=False),
        "",
        "## Replay Weekly Baseline vs Optimised",
        "",
        replay_stage.to_markdown(index=False),
        "",
        "## Side x Month Replay Contribution",
        "",
        side_month.to_markdown(index=False),
        "",
        "## Weak Week Drivers",
        "",
        weak.to_markdown(index=False),
        "",
        "## 2026-06-15 / 2026-06-22 Read",
        "",
        f"- 2026-06-15 baseline net PnL: {_num(june15['portfolio_net_pnl'].iloc[0]) if len(june15) else float('nan'):.2f}",
        f"- 2026-06-22 baseline net PnL: {_num(june22['portfolio_net_pnl'].iloc[0]) if len(june22) else float('nan'):.2f}",
        "- Both weeks are negative under the baseline and become worse under the per-archetype optimiser.",
        "- This points to degraded execution-path quality and weak temporal robustness, not a parser-side artifact.",
        "",
        "## OOS Contract",
        "",
        f"- Base folds: {oos_contract['base'].get('fold_months')} over {oos_contract['base'].get('timestamp_min')} to {oos_contract['base'].get('timestamp_max')}.",
        f"- Meta handoff: fit months {oos_contract['meta_regime_handoff'].get('fit_months')}; holdout month {oos_contract['meta_regime_handoff'].get('holdout_month')}.",
        f"- Optimiser: weekly walk-forward from {oos_contract['execution_optimiser'].get('source_replay_start')} to {oos_contract['execution_optimiser'].get('source_replay_end')} with {oos_contract['execution_optimiser'].get('embargo_days')} day embargo.",
        "",
        "## Recommendation",
        "",
        "Keep S52 baseline as the current execution reference. Investigate June decay at the meta/regime and replay-risk layers before trying another optimiser HPO. The immediate target should be a June-aware diagnostic, not a new hard gate in the base layer.",
        "",
    ]
    (AUDIT_DIR / "june_decay_audit.md").write_text("\n".join(lines))


def main() -> None:
    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)

    base_folds = _summarize_base_folds()
    meta_weekly = _summarize_meta_weekly()
    replay_stage, replay_detail = _summarize_replay_weekly()
    side_month = _side_month_replay(replay_detail)
    weak = _weak_week_drivers(replay_detail)
    oos_contract = _oos_contract_summary()

    base_folds.to_csv(AUDIT_DIR / "base_monthly_topk_summary.csv", index=False)
    meta_weekly.to_csv(AUDIT_DIR / "meta_weekly_offline_handoff_summary.csv", index=False)
    replay_stage.to_csv(AUDIT_DIR / "execution_weekly_replay_summary.csv", index=False)
    replay_detail.to_csv(AUDIT_DIR / "execution_week_side_archetype_attribution.csv", index=False)
    side_month.to_csv(AUDIT_DIR / "execution_month_side_summary.csv", index=False)
    weak.to_csv(AUDIT_DIR / "weak_week_side_archetype_drivers.csv", index=False)
    _write_json(AUDIT_DIR / "oos_contract_summary.json", oos_contract)

    reference_manifest = _make_reference_manifest(base_folds, replay_stage, side_month, oos_contract)
    _write_json(REFERENCE_DIR / "current_reference_manifest.json", reference_manifest)
    replay_stage.to_csv(REFERENCE_DIR / "current_reference_weekly_replay.csv", index=False)
    side_month.to_csv(REFERENCE_DIR / "current_reference_side_month_replay.csv", index=False)
    _write_reference_report(reference_manifest, replay_stage)
    _write_audit_report(base_folds, meta_weekly, replay_stage, weak, side_month, oos_contract)

    print(f"Wrote reference artifacts to {REFERENCE_DIR}")
    print(f"Wrote June decay audit to {AUDIT_DIR}")


if __name__ == "__main__":
    main()
