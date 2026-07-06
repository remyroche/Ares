#!/usr/bin/env python3
"""Frozen holdout validation for bounded smooth-penalty rule combinations."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
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
from scripts.validate_wfrecent_smooth_penalty_frozen_holdout import _load_candidates  # noqa: E402


RULE_LIBRARY: dict[str, SmoothRule] = {
    **dict(DEFAULT_RULES),
    "uncertainty_all_q90_pen0p025_pow2p0": SmoothRule("uncertainty_risk", "all", 0.90, 0.70, 0.025, 2.0),
    "uncertainty_long_dist_q90_pen0p05_pow2p0": SmoothRule("uncertainty_risk", "long_dist", 0.90, 0.70, 0.05, 2.0),
    "recent_hr_all_q85_pen0p025_pow2p0": SmoothRule("recent_perf_risk", "all", 0.85, 0.70, 0.025, 2.0),
    "recent_hr_all_q85_pen0p05_pow2p0": SmoothRule("recent_perf_risk", "all", 0.85, 0.70, 0.05, 2.0),
    "recent_hr_long_bars_q85_pen0p025_pow2p0": SmoothRule("recent_perf_risk", "long_bars", 0.85, 0.70, 0.025, 2.0),
    "recent_hr_long_dist_q85_pen0p025_pow2p0": SmoothRule("recent_perf_risk", "long_dist", 0.85, 0.70, 0.025, 2.0),
    "recent_hr_short_asset_q85_pen0p025_pow2p0": SmoothRule("recent_perf_risk", "short_asset", 0.85, 0.70, 0.025, 2.0),
    "recent_hr_short_bollinger_q85_pen0p025_pow2p0": SmoothRule("recent_perf_risk", "short_bollinger", 0.85, 0.70, 0.025, 2.0),
    "ood_all_q85_pen0p025_pow1p0": SmoothRule("ood_risk", "all", 0.85, 0.70, 0.025, 1.0),
    "ood_all_q90_pen0p025_pow1p0": SmoothRule("ood_risk", "all", 0.90, 0.70, 0.025, 1.0),
    "ood_long_bars_q90_pen0p025_pow1p0": SmoothRule("ood_risk", "long_bars", 0.90, 0.70, 0.025, 1.0),
    "ood_long_dist_q90_pen0p025_pow1p0": SmoothRule("ood_risk", "long_dist", 0.90, 0.70, 0.025, 1.0),
    "ood_short_asset_q90_pen0p025_pow1p0": SmoothRule("ood_risk", "short_asset", 0.90, 0.70, 0.025, 1.0),
    "ood_short_bollinger_q90_pen0p025_pow1p0": SmoothRule(
        "ood_risk", "short_bollinger", 0.90, 0.70, 0.025, 1.0
    ),
    "drift_all_q90_pen0p025_pow1p0": SmoothRule("drift_risk", "all", 0.90, 0.70, 0.025, 1.0),
    "drift_all_q95_pen0p025_pow1p0": SmoothRule("drift_risk", "all", 0.95, 0.70, 0.025, 1.0),
    "drift_long_bars_q90_pen0p025_pow1p0": SmoothRule("drift_risk", "long_bars", 0.90, 0.70, 0.025, 1.0),
    "drift_long_dist_q90_pen0p025_pow1p0": SmoothRule("drift_risk", "long_dist", 0.90, 0.70, 0.025, 1.0),
    "drift_short_asset_q90_pen0p025_pow1p0": SmoothRule("drift_risk", "short_asset", 0.90, 0.70, 0.025, 1.0),
    "drift_short_bollinger_q90_pen0p025_pow1p0": SmoothRule(
        "drift_risk", "short_bollinger", 0.90, 0.70, 0.025, 1.0
    ),
    "drift_long_bars_q95_pen0p025_pow1p0": SmoothRule("drift_risk", "long_bars", 0.95, 0.70, 0.025, 1.0),
    "drift_long_dist_q95_pen0p025_pow1p0": SmoothRule("drift_risk", "long_dist", 0.95, 0.70, 0.025, 1.0),
}


@dataclass(frozen=True)
class ComboLeg:
    rule_name: str
    weight: float


@dataclass(frozen=True)
class Combo:
    label: str
    legs: tuple[ComboLeg, ...]
    total_cap: float


def _default_combos() -> list[Combo]:
    return [
        Combo("q85_only", (ComboLeg("q85_aggressive", 1.0),), 0.05),
        Combo(
            "q85_plus_uncertainty_all_half",
            (ComboLeg("q85_aggressive", 1.0), ComboLeg("uncertainty_all_q90_pen0p025_pow2p0", 0.5)),
            0.06,
        ),
        Combo(
            "q85_plus_uncertainty_ld_half",
            (ComboLeg("q85_aggressive", 1.0), ComboLeg("uncertainty_long_dist_q90_pen0p05_pow2p0", 0.5)),
            0.06,
        ),
        Combo(
            "q85_plus_recent_hr_quarter",
            (ComboLeg("q85_aggressive", 1.0), ComboLeg("recent_hr_all_q85_pen0p025_pow2p0", 0.25)),
            0.06,
        ),
        Combo(
            "q85_plus_recent_hr_half",
            (ComboLeg("q85_aggressive", 1.0), ComboLeg("recent_hr_all_q85_pen0p025_pow2p0", 0.5)),
            0.07,
        ),
        Combo(
            "q85_plus_uncertainty_recent_quarter",
            (
                ComboLeg("q85_aggressive", 1.0),
                ComboLeg("uncertainty_all_q90_pen0p025_pow2p0", 0.5),
                ComboLeg("recent_hr_all_q85_pen0p025_pow2p0", 0.25),
            ),
            0.07,
        ),
        Combo(
            "q85_plus_ood_quarter",
            (ComboLeg("q85_aggressive", 1.0), ComboLeg("ood_all_q90_pen0p025_pow1p0", 0.25)),
            0.06,
        ),
        Combo(
            "q85_plus_drift_quarter",
            (ComboLeg("q85_aggressive", 1.0), ComboLeg("drift_all_q90_pen0p025_pow1p0", 0.25)),
            0.06,
        ),
        Combo(
            "q85_plus_drift_ood_quarter",
            (
                ComboLeg("q85_aggressive", 1.0),
                ComboLeg("drift_all_q90_pen0p025_pow1p0", 0.25),
                ComboLeg("ood_all_q90_pen0p025_pow1p0", 0.25),
            ),
            0.07,
        ),
        Combo(
            "q85_plus_drift_long_bars_quarter",
            (ComboLeg("q85_aggressive", 1.0), ComboLeg("drift_long_bars_q90_pen0p025_pow1p0", 0.25)),
            0.06,
        ),
        Combo(
            "q85_plus_drift_long_dist_quarter",
            (ComboLeg("q85_aggressive", 1.0), ComboLeg("drift_long_dist_q90_pen0p025_pow1p0", 0.25)),
            0.06,
        ),
        Combo(
            "q85_plus_drift_long_heads_quarter",
            (
                ComboLeg("q85_aggressive", 1.0),
                ComboLeg("drift_long_bars_q90_pen0p025_pow1p0", 0.25),
                ComboLeg("drift_long_dist_q90_pen0p025_pow1p0", 0.25),
            ),
            0.07,
        ),
        Combo(
            "q85_plus_drift_long_heads_q95_quarter",
            (
                ComboLeg("q85_aggressive", 1.0),
                ComboLeg("drift_long_bars_q95_pen0p025_pow1p0", 0.25),
                ComboLeg("drift_long_dist_q95_pen0p025_pow1p0", 0.25),
            ),
            0.07,
        ),
        Combo(
            "uncertainty_recent_balanced",
            (ComboLeg("uncertainty_all_q90_pen0p025_pow2p0", 0.75), ComboLeg("recent_hr_all_q85_pen0p025_pow2p0", 0.35)),
            0.06,
        ),
    ]


def _parse_combos(text: str) -> list[Combo]:
    if not text.strip():
        return _default_combos()
    combos: list[Combo] = []
    for raw_combo in text.split(";"):
        raw_combo = raw_combo.strip()
        if not raw_combo:
            continue
        label_part, legs_part = raw_combo.split("=", 1)
        label = label_part.strip()
        total_cap = 0.07
        legs_text = legs_part
        if "|cap=" in legs_part:
            legs_text, cap_text = legs_part.rsplit("|cap=", 1)
            total_cap = float(cap_text)
        legs: list[ComboLeg] = []
        for raw_leg in legs_text.split("+"):
            raw_leg = raw_leg.strip()
            if not raw_leg:
                continue
            if "*" in raw_leg:
                weight_text, rule_name = raw_leg.split("*", 1)
                legs.append(ComboLeg(rule_name.strip(), float(weight_text)))
            else:
                legs.append(ComboLeg(raw_leg, 1.0))
        combos.append(Combo(label, tuple(legs), float(total_cap)))
    return combos


def _apply_combo(
    holdout_scored: pd.DataFrame,
    train_scored: pd.DataFrame,
    combo: Combo,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    total = np.zeros(len(holdout_scored), dtype=np.float32)
    rows: list[dict[str, Any]] = []
    for leg in combo.legs:
        if leg.rule_name not in RULE_LIBRARY:
            raise ValueError(f"Unknown combo rule: {leg.rule_name}")
        rule = RULE_LIBRARY[leg.rule_name]
        threshold = _fit_threshold(train_scored, rule)
        penalty = _penalty_values(holdout_scored, rule, threshold).astype(np.float32) * float(leg.weight)
        total += penalty
        rows.append(
            {
                "combo": combo.label,
                "rule_name": leg.rule_name,
                "weight": float(leg.weight),
                "threshold": float(threshold),
                "raw_penalized_rows": int(np.sum(penalty < 0.0)),
                "raw_penalized_share": float(np.mean(penalty < 0.0)) if len(penalty) else 0.0,
                "raw_mean_penalty": float(np.mean(penalty[penalty < 0.0])) if np.any(penalty < 0.0) else 0.0,
            }
        )
    capped = np.clip(total, -float(combo.total_cap), 0.0).astype(np.float32)
    out = holdout_scored.copy()
    base_adj = pd.to_numeric(out["portfolio_rank_adjustment"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    out["portfolio_rank_adjustment"] = np.clip(base_adj + capped, -1.0, 1.0).astype("float32")
    out["smooth_penalty_variant"] = combo.label
    out["smooth_penalty_value"] = capped
    rows.append(
        {
            "combo": combo.label,
            "rule_name": "__combined__",
            "weight": 1.0,
            "threshold": np.nan,
            "raw_penalized_rows": int(np.sum(capped < 0.0)),
            "raw_penalized_share": float(np.mean(capped < 0.0)) if len(capped) else 0.0,
            "raw_mean_penalty": float(np.mean(capped[capped < 0.0])) if np.any(capped < 0.0) else 0.0,
            "total_cap": float(combo.total_cap),
            "min_penalty": float(np.min(capped[capped < 0.0])) if np.any(capped < 0.0) else 0.0,
        }
    )
    return out, pd.DataFrame(rows)


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
    parser.add_argument(
        "--candidates",
        type=Path,
        default=Path("data_perp/reports/contextual_tp_sl_materialized_wf_recent_q35w07_q20w03_6mo_20260701/combo_candidates.parquet"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data_perp/reports/contextual_tp_sl_wfrecent_smooth_penalty_combo_holdout_20260701"),
    )
    parser.add_argument("--cutoff", default="2026-04-01T00:00:00+00:00")
    parser.add_argument("--end", default="2026-06-27T00:00:00+00:00")
    parser.add_argument("--q35-weight", type=float, default=0.70)
    parser.add_argument("--q20-weight", type=float, default=0.30)
    parser.add_argument("--combos", default="")
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
    audit_rows: list[pd.DataFrame] = []
    for combo in _parse_combos(args.combos):
        adjusted, audit = _apply_combo(holdout_scored, train_scored, combo)
        decisions, _eq, metrics = replay_candidates(
            adjusted,
            params,
            mode="global_auction",
            ev_curve=ev_curve,
            market_mode="perps",
        )
        _daily, weekly = _period_tables(decisions)
        cur_summary = _summary(combo.label, decisions, weekly, metrics, args.q35_weight, args.q20_weight)
        row = _delta_summary(base_summary, cur_summary)
        row["variant"] = combo.label
        row["total_cap"] = float(combo.total_cap)
        row["legs"] = ",".join(f"{leg.weight:g}*{leg.rule_name}" for leg in combo.legs)
        combined = audit[audit["rule_name"].eq("__combined__")]
        if not combined.empty:
            row["penalized_rows"] = int(combined.iloc[0]["raw_penalized_rows"])
            row["penalized_share"] = float(combined.iloc[0]["raw_penalized_share"])
            row["mean_penalty"] = float(combined.iloc[0]["raw_mean_penalty"])
            row["min_penalty"] = float(combined.iloc[0]["min_penalty"])
        summary_rows.append(row)
        monthly_rows.append(_monthly_table(base_weekly, weekly, combo.label))
        ph = _per_head_table(base_decisions, decisions)
        ph["variant"] = combo.label
        per_head_rows.append(ph)
        audit_rows.append(audit)

    summary = pd.DataFrame(summary_rows).sort_values(["delta_objective_week", "delta_net_pnl"], ascending=[False, False])
    monthly = pd.concat(monthly_rows, ignore_index=True) if monthly_rows else pd.DataFrame()
    per_head = pd.concat(per_head_rows, ignore_index=True) if per_head_rows else pd.DataFrame()
    audit = pd.concat(audit_rows, ignore_index=True) if audit_rows else pd.DataFrame()

    summary.to_csv(args.output_dir / "combo_holdout_summary.csv", index=False)
    monthly.to_csv(args.output_dir / "combo_holdout_monthly.csv", index=False)
    per_head.to_csv(args.output_dir / "combo_holdout_per_head.csv", index=False)
    audit.to_csv(args.output_dir / "combo_holdout_apply_audit.csv", index=False)
    base_weekly.to_csv(args.output_dir / "combo_holdout_baseline_weekly.csv", index=False)

    manifest = {
        "generated_by": "validate_wfrecent_smooth_penalty_combo_holdout",
        "candidate_source": str(args.candidates),
        "candidate_source_sha256": _sha256_file(args.candidates),
        "cutoff": cutoff.isoformat(),
        "end": end.isoformat(),
        "q35_weight": float(args.q35_weight),
        "q20_weight": float(args.q20_weight),
        "rule_library": {name: rule.__dict__ for name, rule in RULE_LIBRARY.items()},
        "combos": [
            {"label": combo.label, "total_cap": combo.total_cap, "legs": [leg.__dict__ for leg in combo.legs]}
            for combo in _parse_combos(args.combos)
        ],
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(_json_safe(manifest), indent=2, sort_keys=True) + "\n")

    lines = [
        "# wf_recent Smooth-Penalty Combo Frozen Holdout",
        "",
        "Bounded combinations sum several train-fitted smooth penalties, cap the total rank adjustment, and replay the same holdout universe.",
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
                "delta_net_pnl",
                "delta_objective_week",
                "delta_q35_week_net_pnl",
                "delta_worst_week_net_pnl",
                "delta_hit_rate",
                "delta_full_sl_rate",
                "delta_trade_count",
                "penalized_rows",
                "penalized_share",
                "mean_penalty",
                "total_cap",
            ],
        ),
        "",
        "## Monthly Deltas",
        "",
        _fmt_table(
            monthly,
            [
                "variant",
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
    (args.output_dir / "combo_holdout_report.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
