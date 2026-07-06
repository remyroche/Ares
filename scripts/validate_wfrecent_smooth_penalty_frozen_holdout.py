#!/usr/bin/env python3
"""Frozen holdout validation for wf_recent smooth penalty variants."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.portfolio_policy_replay import (  # noqa: E402
    PortfolioPolicyParams,
    fit_hierarchical_ev_curves,
    replay_candidates,
)
from scripts.ablate_wfrecent_smooth_rank_penalty import SmoothRule, _fit_threshold, _penalty_values  # noqa: E402
from scripts.freeze_apply_wfrecent_smooth_penalty_bundle import DEFAULT_RULES, _sha256_file  # noqa: E402
from scripts.replay_wfrecent_smooth_rank_penalty_fixed import _per_head_table  # noqa: E402
from scripts.validate_wfrecent_row_guard_walkforward import (  # noqa: E402
    _apply_risk_scores,
    _fit_percentile_reference,
    _fmt_table,
    _head_name,
    _json_safe,
    _period_tables,
    _summary,
)


DIAGNOSTIC_FAMILY_RULES: dict[str, SmoothRule] = {
    "uncertainty_long_dist": SmoothRule("uncertainty_risk", "long_dist", 0.90, 0.70, 0.01, 2.0),
    "drift_short_asset": SmoothRule("drift_risk", "short_asset", 0.95, 0.90, 0.01, 2.0),
    "ood_short_asset": SmoothRule("ood_risk", "short_asset", 0.90, 0.70, 0.01, 2.0),
    "recent_hr_long_bars": SmoothRule("recent_perf_risk", "long_bars", 0.90, 0.80, 0.01, 2.0),
    "friction_long_dist": SmoothRule("friction_risk", "long_dist", 0.95, 0.70, 0.01, 2.0),
}


def _diagnostic_family_grid_rules() -> dict[str, SmoothRule]:
    scoped_families = {
        "uncertainty": ("uncertainty_risk", ("all", "long_dist")),
        "drift": ("drift_risk", ("all", "short_asset")),
        "ood": ("ood_risk", ("all", "short_asset")),
        "recent_hr": ("recent_perf_risk", ("all", "long_bars")),
        "friction": ("friction_risk", ("all", "long_dist")),
    }
    rules: dict[str, SmoothRule] = {}
    for family_label, (score_name, scopes) in scoped_families.items():
        for scope in scopes:
            for risk_quantile in (0.85, 0.90):
                for max_penalty in (0.025, 0.05):
                    for power in (1.0, 2.0):
                        label = (
                            f"{family_label}_{scope}_q{int(risk_quantile * 100)}_"
                            f"pen{str(max_penalty).replace('.', 'p')}_pow{str(power).replace('.', 'p')}"
                        )
                        rules[label] = SmoothRule(score_name, scope, risk_quantile, 0.70, max_penalty, power)
    return rules


def _rules_for_mode(mode: str) -> dict[str, SmoothRule]:
    if mode == "default":
        return dict(DEFAULT_RULES)
    if mode == "diagnostic_families":
        return dict(DIAGNOSTIC_FAMILY_RULES)
    if mode == "default_plus_diagnostic_families":
        return {**dict(DEFAULT_RULES), **dict(DIAGNOSTIC_FAMILY_RULES)}
    if mode == "diagnostic_family_grid":
        return _diagnostic_family_grid_rules()
    if mode == "default_plus_diagnostic_family_grid":
        return {**dict(DEFAULT_RULES), **_diagnostic_family_grid_rules()}
    raise ValueError(f"Unsupported rule mode: {mode}")


def _load_candidates(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    frame = frame[frame["timestamp"].notna()].sort_values(["timestamp", "strategy_id", "symbol"]).reset_index(drop=True)
    frame["head"] = frame["strategy_id"].map(_head_name)
    if "portfolio_rank_adjustment" not in frame.columns:
        frame["portfolio_rank_adjustment"] = np.float32(0.0)
    else:
        frame["portfolio_rank_adjustment"] = pd.to_numeric(frame["portfolio_rank_adjustment"], errors="coerce").fillna(0.0).astype("float32")
    return frame


def _apply_rule(holdout_scored: pd.DataFrame, train_scored: pd.DataFrame, rule: SmoothRule) -> tuple[pd.DataFrame, dict[str, Any]]:
    threshold = _fit_threshold(train_scored, rule)
    penalty = _penalty_values(holdout_scored, rule, threshold)
    out = holdout_scored.copy()
    base_adj = pd.to_numeric(out["portfolio_rank_adjustment"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    out["portfolio_rank_adjustment"] = np.clip(base_adj + penalty, -1.0, 1.0).astype("float32")
    out["smooth_penalty_variant"] = rule.label
    out["smooth_penalty_value"] = penalty.astype("float32")
    return out, {
        "threshold": float(threshold),
        "penalized_rows": int(np.sum(penalty < 0.0)),
        "penalized_share": float(np.mean(penalty < 0.0)) if len(penalty) else 0.0,
        "mean_penalty": float(np.mean(penalty[penalty < 0.0])) if np.any(penalty < 0.0) else 0.0,
        "min_penalty": float(np.min(penalty[penalty < 0.0])) if np.any(penalty < 0.0) else 0.0,
    }


def _delta_summary(base: dict[str, Any], cur: dict[str, Any]) -> dict[str, Any]:
    row = {"label": cur["label"]}
    for key in (
        "net_pnl",
        "gross_pnl",
        "trade_count",
        "hit_rate",
        "full_sl_rate",
        "timeout_rate",
        "max_drawdown",
        "objective_week",
        "q20_week_net_pnl",
        "q35_week_net_pnl",
        "worst_week_net_pnl",
        "positive_weeks",
    ):
        row[f"baseline_{key}"] = base[key]
        row[f"challenger_{key}"] = cur[key]
        row[f"delta_{key}"] = float(cur[key] - base[key])
    return row


def _monthly_table(base_weekly: pd.DataFrame, challenger_weekly: pd.DataFrame, variant: str) -> pd.DataFrame:
    def prep(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
        cur = frame[frame["period_type"].eq("week")].copy()
        cur["week_start"] = pd.PeriodIndex(cur["week"], freq="W").start_time
        cur["month"] = cur["week_start"].dt.to_period("M").astype(str)
        out = (
            cur.groupby("month", as_index=False)
            .agg(
                net_pnl=("net_pnl", "sum"),
                trades=("trades", "sum"),
                hit_rate=("hit_rate", "mean"),
                full_sl_rate=("full_sl_rate", "mean"),
                timeout_rate=("timeout_rate", "mean"),
                worst_week_net_pnl=("net_pnl", "min"),
            )
        )
        return out.rename(columns={c: f"{prefix}_{c}" for c in out.columns if c != "month"})

    out = prep(base_weekly, "baseline").merge(prep(challenger_weekly, "challenger"), on="month", how="outer")
    out["variant"] = variant
    for key in ("net_pnl", "trades", "hit_rate", "full_sl_rate", "timeout_rate", "worst_week_net_pnl"):
        out[f"delta_{key}"] = out[f"challenger_{key}"] - out[f"baseline_{key}"]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, default=Path("data_perp/reports/contextual_tp_sl_materialized_wf_recent_q35w07_q20w03_6mo_20260701/combo_candidates.parquet"))
    parser.add_argument("--output-dir", type=Path, default=Path("data_perp/reports/contextual_tp_sl_wfrecent_smooth_penalty_frozen_holdout_may_jun_20260701"))
    parser.add_argument("--cutoff", default="2026-05-01T00:00:00+00:00")
    parser.add_argument("--end", default="2026-06-27T00:00:00+00:00")
    parser.add_argument("--q35-weight", type=float, default=0.70)
    parser.add_argument("--q20-weight", type=float, default=0.30)
    parser.add_argument(
        "--rule-mode",
        choices=(
            "default",
            "diagnostic_families",
            "default_plus_diagnostic_families",
            "diagnostic_family_grid",
            "default_plus_diagnostic_family_grid",
        ),
        default="default",
        help="Which smooth-penalty rules to validate. Default preserves the production challenger set.",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    candidates = _load_candidates(args.candidates)
    cutoff = pd.Timestamp(args.cutoff, tz="UTC")
    end = pd.Timestamp(args.end, tz="UTC")
    train = candidates[candidates["timestamp"].lt(cutoff)].copy().reset_index(drop=True)
    holdout = candidates[candidates["timestamp"].ge(cutoff) & candidates["timestamp"].lt(end)].copy().reset_index(drop=True)
    if train.empty or holdout.empty:
        raise ValueError(f"Empty train or holdout: train={len(train)} holdout={len(holdout)}")

    refs = _fit_percentile_reference(train)
    train_scored = _apply_risk_scores(train, refs)
    holdout_scored = _apply_risk_scores(holdout, refs)
    ev_curve = fit_hierarchical_ev_curves(train_scored)
    params = PortfolioPolicyParams(global_threshold_floor=0.0)

    base_decisions, _base_eq, base_metrics = replay_candidates(
        holdout_scored,
        params,
        mode="global_auction",
        ev_curve=ev_curve,
        market_mode="perps",
    )
    _base_daily, base_weekly = _period_tables(base_decisions)
    base_summary = _summary("baseline", base_decisions, base_weekly, base_metrics, args.q35_weight, args.q20_weight)

    summary_rows: list[dict[str, Any]] = []
    monthly_rows: list[pd.DataFrame] = []
    per_head_rows: list[pd.DataFrame] = []
    audit_rows: list[dict[str, Any]] = []
    rules = _rules_for_mode(str(args.rule_mode))
    for variant, rule in rules.items():
        adjusted, audit = _apply_rule(holdout_scored, train_scored, rule)
        decisions, _eq, metrics = replay_candidates(
            adjusted,
            params,
            mode="global_auction",
            ev_curve=ev_curve,
            market_mode="perps",
        )
        _daily, weekly = _period_tables(decisions)
        cur_summary = _summary(variant, decisions, weekly, metrics, args.q35_weight, args.q20_weight)
        row = _delta_summary(base_summary, cur_summary)
        row["variant"] = variant
        row.update(
            {
                "score_name": rule.score_name,
                "scope": rule.scope,
                "risk_quantile": float(rule.risk_quantile),
                "min_rank_pct": float(rule.min_rank_pct),
                "max_penalty": float(rule.max_penalty),
                "power": float(rule.power),
            }
        )
        row.update(audit)
        summary_rows.append(row)
        monthly_table = _monthly_table(base_weekly, weekly, variant)
        monthly_table["score_name"] = rule.score_name
        monthly_table["scope"] = rule.scope
        monthly_rows.append(monthly_table)
        ph = _per_head_table(base_decisions, decisions)
        ph["variant"] = variant
        per_head_rows.append(ph)
        audit_rows.append({"variant": variant, **audit})

    summary = pd.DataFrame(summary_rows).sort_values(["delta_objective_week", "delta_net_pnl"], ascending=[False, False])
    monthly = pd.concat(monthly_rows, ignore_index=True) if monthly_rows else pd.DataFrame()
    per_head = pd.concat(per_head_rows, ignore_index=True) if per_head_rows else pd.DataFrame()
    audit = pd.DataFrame(audit_rows)

    summary.to_csv(args.output_dir / "frozen_holdout_summary.csv", index=False)
    monthly.to_csv(args.output_dir / "frozen_holdout_monthly.csv", index=False)
    per_head.to_csv(args.output_dir / "frozen_holdout_per_head.csv", index=False)
    audit.to_csv(args.output_dir / "frozen_holdout_apply_audit.csv", index=False)
    base_weekly.to_csv(args.output_dir / "frozen_holdout_baseline_weekly.csv", index=False)
    manifest = {
        "generated_by": "validate_wfrecent_smooth_penalty_frozen_holdout",
        "candidate_source": str(args.candidates),
        "candidate_source_sha256": _sha256_file(args.candidates),
        "cutoff": cutoff.isoformat(),
        "end": end.isoformat(),
        "train_rows": int(len(train)),
        "holdout_rows": int(len(holdout)),
        "train_start": train["timestamp"].min().isoformat(),
        "train_end": train["timestamp"].max().isoformat(),
        "holdout_start": holdout["timestamp"].min().isoformat(),
        "holdout_end": holdout["timestamp"].max().isoformat(),
        "rule_mode": str(args.rule_mode),
        "rules": {name: rule.__dict__ for name, rule in rules.items()},
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(_json_safe(manifest), indent=2, sort_keys=True) + "\n")

    lines = [
        "# wf_recent Smooth Penalty Frozen Holdout Validation",
        "",
        "Single delayed/frozen holdout. Risk references, thresholds, and EV curve are fit only on rows before the cutoff.",
        "",
        f"Cutoff: `{cutoff.isoformat()}`",
        f"Holdout: `{holdout['timestamp'].min().isoformat()}` to `{holdout['timestamp'].max().isoformat()}`",
        "",
        "## Summary",
        "",
        _fmt_table(
            summary,
            [
                "variant",
                "score_name",
                "scope",
                "delta_net_pnl",
                "delta_objective_week",
                "delta_q35_week_net_pnl",
                "delta_worst_week_net_pnl",
                "delta_hit_rate",
                "delta_full_sl_rate",
                "delta_timeout_rate",
                "delta_trade_count",
                "penalized_rows",
                "penalized_share",
                "mean_penalty",
            ],
        ),
        "",
        "## Monthly Deltas",
        "",
        _fmt_table(
            monthly,
            [
                "variant",
                "score_name",
                "scope",
                "month",
                "delta_net_pnl",
                "delta_trades",
                "delta_hit_rate",
                "delta_full_sl_rate",
                "delta_worst_week_net_pnl",
            ],
        ),
        "",
        "## Per-Head Deltas",
        "",
        _fmt_table(
            per_head,
            [
                "variant",
                "head",
                "delta_net_pnl",
                "delta_gross_pnl",
                "delta_trades",
                "delta_hit_rate",
                "delta_full_sl_rate",
                "delta_timeout_rate",
            ],
        ),
    ]
    (args.output_dir / "frozen_holdout_report.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
