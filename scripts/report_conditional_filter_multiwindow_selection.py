#!/usr/bin/env python3
"""Rank conditional filter rules across multiple replay windows.

This report consumes the outputs from
``ablate_contextual_tp_sl_conditional_head_filters.py`` and compares every
rule to a baseline rule. It is intentionally replay-artifact driven: no model
is refit and no candidate table is replayed here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple

import numpy as np
import pandas as pd


CORE_WINDOWS: Dict[str, Tuple[pd.Timestamp | None, pd.Timestamp | None]] = {
    "full": (None, None),
    "pre_may": (None, pd.Timestamp("2026-05-01", tz="UTC")),
    "may_june": (pd.Timestamp("2026-05-01", tz="UTC"), None),
    "june": (pd.Timestamp("2026-06-01", tz="UTC"), None),
}

EXTRA_WINDOWS: Dict[str, Tuple[pd.Timestamp | None, pd.Timestamp | None]] = {
    "jun23_plus": (pd.Timestamp("2026-06-23", tz="UTC"), None),
    "jun27_28": (pd.Timestamp("2026-06-27", tz="UTC"), None),
    "post_jun28": (pd.Timestamp("2026-06-29", tz="UTC"), None),
}


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(float(value)) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _head_name(strategy_id: str) -> str:
    text = str(strategy_id)
    if text.startswith("short_bollinger"):
        return "short_bollinger"
    parts = text.split("_")
    return "_".join(parts[:2]) if len(parts) >= 2 else text


def _period_start(frame: pd.DataFrame, period_col: str) -> pd.Series:
    if period_col == "day":
        return pd.to_datetime(frame[period_col], utc=True, errors="coerce")
    if period_col == "week":
        values = pd.PeriodIndex(frame[period_col].astype(str), freq="W").start_time.tz_localize("UTC")
        return pd.Series(values, index=frame.index)
    raise ValueError(f"Unknown period column `{period_col}`")


def _filter_window(
    frame: pd.DataFrame,
    period_col: str,
    bounds: Tuple[pd.Timestamp | None, pd.Timestamp | None],
) -> pd.DataFrame:
    start, end = bounds
    ts = _period_start(frame, period_col)
    mask = pd.Series(True, index=frame.index)
    if start is not None:
        mask &= ts.ge(start)
    if end is not None:
        mask &= ts.lt(end)
    return frame.loc[mask].copy()


def _delta_series(
    frame: pd.DataFrame,
    period_col: str,
    rule_id: str,
    baseline_rule: str,
) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype=float)
    piv = (
        frame.pivot_table(index=period_col, columns="rule_id", values="net_pnl", aggfunc="sum")
        .fillna(0.0)
        .sort_index()
    )
    if piv.empty:
        return pd.Series(dtype=float)
    if baseline_rule not in piv.columns:
        raise ValueError(f"Missing baseline rule `{baseline_rule}` in {period_col} table")
    if rule_id not in piv.columns:
        return pd.Series(dtype=float)
    return piv[rule_id].astype(float) - piv[baseline_rule].astype(float)


def _q(values: pd.Series, percentile: float) -> float:
    arr = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if arr.size == 0:
        return 0.0
    return float(np.nanpercentile(arr, percentile))


def _window_metrics(
    daily: pd.DataFrame,
    weekly: pd.DataFrame,
    rule_id: str,
    baseline_rule: str,
    windows: Mapping[str, Tuple[pd.Timestamp | None, pd.Timestamp | None]],
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for window, bounds in windows.items():
        daily_w = _filter_window(daily, "day", bounds)
        weekly_w = _filter_window(weekly, "week", bounds)
        day_delta = _delta_series(daily_w, "day", rule_id, baseline_rule)
        week_delta = _delta_series(weekly_w, "week", rule_id, baseline_rule)
        days = int(len(day_delta))
        weeks = int(len(week_delta))
        avg_week_delta = float(week_delta.mean()) if weeks else 0.0
        daily_q20 = _q(day_delta, 20)
        daily_q35 = _q(day_delta, 35)
        weekly_q20 = _q(week_delta, 20)
        weighted_daily_tail = 0.7 * daily_q35 + 0.3 * daily_q20
        delta_objective = avg_week_delta + weighted_daily_tail
        rows.append(
            {
                "window": window,
                "rule_id": rule_id,
                "days": days,
                "weeks": weeks,
                "delta_net_pnl": float(day_delta.sum()),
                "delta_avg_week_pnl": avg_week_delta,
                "delta_daily_q20_pnl": daily_q20,
                "delta_daily_q35_pnl": daily_q35,
                "delta_weighted_daily_tail": weighted_daily_tail,
                "delta_weekly_q20_pnl": weekly_q20,
                "delta_weekly_q35_pnl": _q(week_delta, 35),
                "positive_day_share": float((day_delta > 0.0).mean()) if days else 0.0,
                "positive_week_share": float((week_delta > 0.0).mean()) if weeks else 0.0,
                "delta_objective_avgweek_0p7dayq35_0p3dayq20": delta_objective,
                "pnl_tail_gate": bool(
                    days > 0
                    and float(day_delta.sum()) > 0.0
                    and delta_objective > 0.0
                    and weighted_daily_tail >= 0.0
                    and weekly_q20 >= 0.0
                ),
                "strict_tail_gate": bool(
                    days > 0
                    and float(day_delta.sum()) > 0.0
                    and delta_objective > 0.0
                    and daily_q20 >= 0.0
                    and daily_q35 >= 0.0
                    and weekly_q20 >= 0.0
                ),
            }
        )
    return pd.DataFrame(rows)


def _replacement_quality(accepted: pd.DataFrame, rule_id: str, baseline_rule: str) -> Dict[str, Any]:
    if accepted.empty:
        return {}
    work = accepted.copy()
    work["head"] = work["strategy_id"].astype(str).map(_head_name)
    work["net_pnl_amount"] = (
        pd.to_numeric(work.get("position_size"), errors="coerce").fillna(0.0)
        * pd.to_numeric(work.get("position_net_return"), errors="coerce").fillna(0.0)
    )
    work["hit"] = pd.to_numeric(work.get("position_net_return"), errors="coerce").fillna(0.0).gt(0.0)
    work["full_sl"] = work.get("position_exit_reason", "").astype(str).eq("full_sl")
    key_cols = ["timestamp", "symbol", "side", "strategy_id"]
    baseline = work.loc[work["rule_id"].eq(baseline_rule)].copy()
    challenger = work.loc[work["rule_id"].eq(rule_id)].copy()
    if baseline.empty or challenger.empty:
        return {}
    base_keys = set(map(tuple, baseline[key_cols].astype(str).to_numpy()))
    challenger_keys = set(map(tuple, challenger[key_cols].astype(str).to_numpy()))
    entrants = challenger.loc[[tuple(row) in challenger_keys - base_keys for row in challenger[key_cols].astype(str).to_numpy()]]
    removed = baseline.loc[[tuple(row) in base_keys - challenger_keys for row in baseline[key_cols].astype(str).to_numpy()]]

    def summarize(prefix: str, frame: pd.DataFrame) -> Dict[str, Any]:
        return {
            f"{prefix}_trades": int(len(frame)),
            f"{prefix}_net_pnl": float(frame["net_pnl_amount"].sum()),
            f"{prefix}_hit_rate": float(frame["hit"].mean()) if len(frame) else 0.0,
            f"{prefix}_full_sl_rate": float(frame["full_sl"].mean()) if len(frame) else 0.0,
        }

    out: Dict[str, Any] = {}
    out.update(summarize("entrant", entrants))
    out.update(summarize("removed", removed))
    out["entrant_minus_removed_net_pnl"] = out["entrant_net_pnl"] - out["removed_net_pnl"]
    out["entrant_minus_removed_hit_rate"] = out["entrant_hit_rate"] - out["removed_hit_rate"]
    return out


def _candidate_summary(
    window_metrics: pd.DataFrame,
    accepted: pd.DataFrame,
    baseline_rule: str,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for rule_id, group in window_metrics.groupby("rule_id", sort=False):
        core = group.loc[group["window"].isin(CORE_WINDOWS)]
        extras = group.loc[group["window"].isin(EXTRA_WINDOWS)]
        full = core.loc[core["window"].eq("full")].iloc[0]
        june = core.loc[core["window"].eq("june")].iloc[0]
        rec: Dict[str, Any] = {
            "rule_id": rule_id,
            "core_pnl_tail_gate_count": int(core["pnl_tail_gate"].sum()),
            "core_strict_tail_gate_count": int(core["strict_tail_gate"].sum()),
            "core_min_delta_objective": float(core["delta_objective_avgweek_0p7dayq35_0p3dayq20"].min()),
            "core_min_delta_net_pnl": float(core["delta_net_pnl"].min()),
            "core_min_delta_weekly_q20": float(core["delta_weekly_q20_pnl"].min()),
            "core_min_delta_weighted_daily_tail": float(core["delta_weighted_daily_tail"].min()),
            "full_delta_objective": float(full["delta_objective_avgweek_0p7dayq35_0p3dayq20"]),
            "full_delta_net_pnl": float(full["delta_net_pnl"]),
            "full_delta_weekly_q20": float(full["delta_weekly_q20_pnl"]),
            "full_delta_weighted_daily_tail": float(full["delta_weighted_daily_tail"]),
            "june_delta_objective": float(june["delta_objective_avgweek_0p7dayq35_0p3dayq20"]),
            "june_delta_net_pnl": float(june["delta_net_pnl"]),
            "june_delta_weekly_q20": float(june["delta_weekly_q20_pnl"]),
            "june_delta_weighted_daily_tail": float(june["delta_weighted_daily_tail"]),
            "extra_window_bind_count": int(extras["delta_net_pnl"].ne(0.0).sum()) if not extras.empty else 0,
        }
        rec.update(_replacement_quality(accepted, rule_id, baseline_rule))
        rows.append(rec)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    sort_cols = [
        "core_pnl_tail_gate_count",
        "core_strict_tail_gate_count",
        "core_min_delta_objective",
        "full_delta_objective",
        "full_delta_net_pnl",
    ]
    out = out.sort_values(sort_cols, ascending=[False, False, False, False, False]).reset_index(drop=True)
    out.insert(0, "selection_rank", np.arange(1, len(out) + 1, dtype=int))
    return out


def _profile_recommendations(
    summary: pd.DataFrame,
    *,
    tolerant_min_core_gates: int,
    tolerant_min_core_net_pnl: float,
    tolerant_min_core_objective: float,
) -> Dict[str, Dict[str, Any] | None]:
    if summary.empty:
        return {
            "pnl_dominant": None,
            "balanced_pnl_tail": None,
            "balanced_tolerant": None,
            "strict_tail": None,
        }

    def first_or_none(frame: pd.DataFrame) -> Dict[str, Any] | None:
        return frame.iloc[0].to_dict() if not frame.empty else None

    pnl_dominant = summary.loc[
        (pd.to_numeric(summary["core_min_delta_net_pnl"], errors="coerce") > 0.0)
        & (pd.to_numeric(summary["core_min_delta_objective"], errors="coerce") > 0.0)
    ].sort_values(["full_delta_net_pnl", "full_delta_objective"], ascending=[False, False])

    balanced = summary.loc[
        (pd.to_numeric(summary["core_pnl_tail_gate_count"], errors="coerce") >= 3)
        & (pd.to_numeric(summary["core_min_delta_net_pnl"], errors="coerce") > 0.0)
        & (pd.to_numeric(summary["core_min_delta_objective"], errors="coerce") > 0.0)
    ].sort_values(
        ["full_delta_net_pnl", "core_pnl_tail_gate_count", "core_min_delta_objective"],
        ascending=[False, False, False],
    )

    strict = summary.loc[
        (pd.to_numeric(summary["core_strict_tail_gate_count"], errors="coerce") >= len(CORE_WINDOWS))
        & (pd.to_numeric(summary["core_min_delta_net_pnl"], errors="coerce") > 0.0)
        & (pd.to_numeric(summary["core_min_delta_objective"], errors="coerce") > 0.0)
    ].sort_values(["full_delta_net_pnl", "full_delta_objective"], ascending=[False, False])

    tolerant = summary.loc[
        (pd.to_numeric(summary["core_pnl_tail_gate_count"], errors="coerce") >= int(tolerant_min_core_gates))
        & (pd.to_numeric(summary["core_min_delta_net_pnl"], errors="coerce") >= float(tolerant_min_core_net_pnl))
        & (pd.to_numeric(summary["core_min_delta_objective"], errors="coerce") >= float(tolerant_min_core_objective))
        & (pd.to_numeric(summary["full_delta_net_pnl"], errors="coerce") > 0.0)
        & (pd.to_numeric(summary["full_delta_objective"], errors="coerce") > 0.0)
        & (pd.to_numeric(summary["full_delta_weighted_daily_tail"], errors="coerce") >= 0.0)
        & (pd.to_numeric(summary["full_delta_weekly_q20"], errors="coerce") >= 0.0)
    ].sort_values(
        [
            "core_pnl_tail_gate_count",
            "core_min_delta_objective",
            "full_delta_objective",
            "full_delta_net_pnl",
        ],
        ascending=[False, False, False, False],
    )

    return {
        "pnl_dominant": first_or_none(pnl_dominant),
        "balanced_pnl_tail": first_or_none(balanced),
        "balanced_tolerant": first_or_none(tolerant),
        "strict_tail": first_or_none(strict),
    }


def _markdown_table(df: pd.DataFrame, columns: Iterable[str]) -> str:
    cols = [c for c in columns if c in df.columns]
    if df.empty or not cols:
        return "_No rows._"
    return df[cols].round(6).to_markdown(index=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attribution-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--baseline-rule", default="none")
    parser.add_argument("--tolerant-min-core-gates", type=int, default=2)
    parser.add_argument("--tolerant-min-core-net-pnl", type=float, default=0.0)
    parser.add_argument("--tolerant-min-core-objective", type=float, default=-25.0)
    args = parser.parse_args()

    daily_path = args.attribution_dir / "conditional_filter_daily.csv"
    weekly_path = args.attribution_dir / "conditional_filter_weekly.csv"
    accepted_path = args.attribution_dir / "conditional_filter_accepted_decisions.parquet"
    if not daily_path.exists() or not weekly_path.exists():
        raise FileNotFoundError("conditional_filter_daily.csv and conditional_filter_weekly.csv are required")
    daily = pd.read_csv(daily_path)
    weekly = pd.read_csv(weekly_path)
    accepted = pd.read_parquet(accepted_path) if accepted_path.exists() else pd.DataFrame()
    rules = [r for r in daily["rule_id"].dropna().astype(str).unique() if r != args.baseline_rule]
    all_windows = {**CORE_WINDOWS, **EXTRA_WINDOWS}
    frames = [_window_metrics(daily, weekly, rule, args.baseline_rule, all_windows) for rule in rules]
    window_metrics = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    summary = _candidate_summary(window_metrics, accepted, args.baseline_rule)
    profile_recs = _profile_recommendations(
        summary,
        tolerant_min_core_gates=args.tolerant_min_core_gates,
        tolerant_min_core_net_pnl=args.tolerant_min_core_net_pnl,
        tolerant_min_core_objective=args.tolerant_min_core_objective,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    window_metrics.to_csv(args.out_dir / "multiwindow_rule_deltas.csv", index=False)
    summary.to_csv(args.out_dir / "multiwindow_candidate_summary.csv", index=False)

    recommended = (
        profile_recs.get("balanced_pnl_tail")
        or profile_recs.get("balanced_tolerant")
        or profile_recs.get("pnl_dominant")
        or profile_recs.get("strict_tail")
    )
    best_by_sort_order = summary.iloc[0].to_dict() if not summary.empty else {}
    payload = {
        "attribution_dir": str(args.attribution_dir),
        "baseline_rule": args.baseline_rule,
        "core_windows": {k: [str(v[0]), str(v[1])] for k, v in CORE_WINDOWS.items()},
        "extra_windows": {k: [str(v[0]), str(v[1])] for k, v in EXTRA_WINDOWS.items()},
        "selection_order": [
            "core_pnl_tail_gate_count desc",
            "core_strict_tail_gate_count desc",
            "core_min_delta_objective desc",
            "full_delta_objective desc",
            "full_delta_net_pnl desc",
        ],
        "tolerant_profile": {
            "min_core_gates": args.tolerant_min_core_gates,
            "min_core_net_pnl": args.tolerant_min_core_net_pnl,
            "min_core_objective": args.tolerant_min_core_objective,
        },
        "recommended": recommended,
        "best_by_sort_order": best_by_sort_order,
        "profile_recommendations": profile_recs,
    }
    (args.out_dir / "multiwindow_selection.json").write_text(json.dumps(_json_safe(payload), indent=2) + "\n")

    report = [
        "# Conditional Filter Multi-Window Selection",
        "",
        f"Attribution source: `{args.attribution_dir}`",
        f"Baseline rule: `{args.baseline_rule}`",
        "",
        "Core windows: full, pre-May, May-June, June. Extra windows are diagnostic only.",
        "",
        "Selection order: core PnL/tail gate count, strict-tail gate count, worst core objective, full objective, full net PnL.",
        "",
        "The `balanced_tolerant` profile is a research-candidate profile: it still requires positive full-period PnL/objective and non-negative full-period weekly/daily tail deltas, but allows a small core-window objective near-miss.",
        "",
        "## Candidate Summary",
        "",
        _markdown_table(
            summary,
            [
                "selection_rank",
                "rule_id",
                "core_pnl_tail_gate_count",
                "core_strict_tail_gate_count",
                "core_min_delta_objective",
                "full_delta_objective",
                "full_delta_net_pnl",
                "june_delta_objective",
                "june_delta_net_pnl",
                "entrant_minus_removed_net_pnl",
                "entrant_minus_removed_hit_rate",
                "extra_window_bind_count",
            ],
        ),
        "",
        "## Profile Recommendations",
        "",
    ]
    for profile, rec in profile_recs.items():
        if rec is None:
            report.append(f"- `{profile}`: no candidate passed the profile.")
        else:
            report.append(
                f"- `{profile}`: `{rec['rule_id']}` "
                f"(full net delta `{rec['full_delta_net_pnl']:.2f}`, "
                f"core min objective `{rec['core_min_delta_objective']:.2f}`)."
            )
    report.extend(
        [
            "",
        "## Window Deltas",
        "",
        _markdown_table(
            window_metrics.sort_values(["window", "delta_objective_avgweek_0p7dayq35_0p3dayq20"], ascending=[True, False]),
            [
                "window",
                "rule_id",
                "delta_objective_avgweek_0p7dayq35_0p3dayq20",
                "delta_net_pnl",
                "delta_weekly_q20_pnl",
                "delta_weighted_daily_tail",
                "pnl_tail_gate",
                "strict_tail_gate",
            ],
        ),
        "",
        ]
    )
    if recommended:
        report.extend(
            [
                "## Recommendation",
                "",
                f"Recommended next frozen/fresh replay candidate: `{recommended['rule_id']}`.",
                "",
            ]
        )
    elif best_by_sort_order:
        report.extend(
            [
                "## Recommendation",
                "",
                "No candidate passed a recommendation profile.",
                "",
                f"Best by sort order only: `{best_by_sort_order['rule_id']}`. Treat as diagnostic, not a frozen/fresh replay candidate.",
                "",
            ]
        )
    (args.out_dir / "multiwindow_selection_report.md").write_text("\n".join(report))
    print(json.dumps(_json_safe({"out_dir": str(args.out_dir), "recommended": recommended}), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
