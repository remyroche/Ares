#!/usr/bin/env python3
"""Row-level execution-quality guard around the fixed wf_recent TP/SL combo.

This script tests whether existing candidate diagnostics can veto likely bad
``wf_recent`` candidates while preserving the portfolio auction. It first
screens simple high-risk veto rules on accepted trades, then reruns the full
portfolio replay for the best shortlisted rules.

The test is a development ablation over existing materialized artifacts, not a
fresh OOS promotion gate.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
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


HEAD_ORDER = ("long_bars", "long_dist", "short_asset", "short_bollinger")
RISK_SCORE_NAMES = (
    "uncertainty_risk",
    "drift_risk",
    "ood_risk",
    "recent_perf_risk",
    "friction_risk",
    "composite_risk",
)


@dataclass(frozen=True)
class VetoRule:
    score_name: str
    scope: str
    risk_quantile: float
    min_rank_pct: float


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
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _fmt_table(frame: pd.DataFrame, cols: list[str], max_rows: int | None = None) -> str:
    if frame.empty:
        return "_No rows._"
    view = frame[cols].head(max_rows).copy() if max_rows else frame[cols].copy()
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda x: "" if pd.isna(x) else f"{x:,.3f}")
    return view.to_markdown(index=False)


def _head_name(strategy_id: Any) -> str:
    text = str(strategy_id)
    if text.startswith("short_bollinger"):
        return "short_bollinger"
    parts = text.split("_")
    return "_".join(parts[:2]) if len(parts) >= 2 else text


def _period_tables(decisions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    accepted = decisions[decisions["accepted"].astype(bool)].copy()
    if accepted.empty:
        return pd.DataFrame(), pd.DataFrame()
    ts = pd.to_datetime(accepted["timestamp"], utc=True, errors="coerce")
    accepted["day"] = ts.dt.date.astype(str)
    accepted["week"] = ts.dt.to_period("W").astype(str)
    accepted["head"] = accepted["strategy_id"].map(_head_name)
    size = pd.to_numeric(accepted["position_size"], errors="coerce").fillna(0.0)
    net = pd.to_numeric(accepted["position_net_return"], errors="coerce").fillna(0.0)
    gross = pd.to_numeric(accepted["position_gross_return"], errors="coerce").fillna(0.0)
    accepted["net_pnl_amount"] = size * net
    accepted["gross_pnl_amount"] = size * gross
    accepted["is_win"] = net > 0.0
    reason = accepted["position_exit_reason"].astype(str) if "position_exit_reason" in accepted.columns else pd.Series("", index=accepted.index)
    accepted["is_full_sl"] = reason.str.contains("sl", case=False, na=False)
    accepted["is_timeout"] = reason.str.contains("timeout", case=False, na=False)
    frames_daily: list[pd.DataFrame] = []
    frames_weekly: list[pd.DataFrame] = []
    for cols, target in ((["day"], frames_daily), (["day", "head"], frames_daily), (["week"], frames_weekly), (["week", "head"], frames_weekly)):
        cur = (
            accepted.groupby(cols, as_index=False)
            .agg(
                net_pnl=("net_pnl_amount", "sum"),
                gross_pnl=("gross_pnl_amount", "sum"),
                trades=("accepted", "size"),
                hit_rate=("is_win", "mean"),
                full_sl_rate=("is_full_sl", "mean"),
                timeout_rate=("is_timeout", "mean"),
            )
            .sort_values(cols)
        )
        cur.insert(0, "period_type", "_".join(cols))
        target.append(cur)
    return pd.concat(frames_daily, ignore_index=True), pd.concat(frames_weekly, ignore_index=True)


def _metrics_from_decisions(decisions: pd.DataFrame) -> dict[str, float | int]:
    accepted = decisions[decisions["accepted"].astype(bool)].copy() if "accepted" in decisions.columns else pd.DataFrame()
    if accepted.empty:
        return {
            "net_pnl": 0.0,
            "gross_pnl": 0.0,
            "trade_count": 0,
            "full_sl_rate": 0.0,
            "timeout_rate": 0.0,
            "max_drawdown": np.nan,
        }
    size = pd.to_numeric(accepted.get("position_size", 0.0), errors="coerce").fillna(0.0)
    net = pd.to_numeric(accepted.get("position_net_return", 0.0), errors="coerce").fillna(0.0)
    gross = pd.to_numeric(accepted.get("position_gross_return", 0.0), errors="coerce").fillna(0.0)
    reason = accepted.get("position_exit_reason", pd.Series("", index=accepted.index)).astype(str)
    return {
        "net_pnl": float((size * net).sum()),
        "gross_pnl": float((size * gross).sum()),
        "trade_count": int(len(accepted)),
        "full_sl_rate": float(reason.str.contains("sl", case=False, na=False).mean()),
        "timeout_rate": float(reason.str.contains("timeout", case=False, na=False).mean()),
        "max_drawdown": np.nan,
    }


def _objective_from_weekly(weekly: pd.DataFrame, q35_weight: float, q20_weight: float) -> float:
    global_weekly = weekly[weekly["period_type"].eq("week")].copy()
    values = pd.to_numeric(global_weekly["net_pnl"], errors="coerce").dropna().to_numpy(dtype=np.float64)
    if values.size == 0:
        return float("nan")
    return float(np.mean(values) + q35_weight * np.quantile(values, 0.35) + q20_weight * np.quantile(values, 0.20))


def _load_inputs(candidates_path: Path, decisions_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    columns = None
    candidates = pd.read_parquet(candidates_path, columns=columns)
    decisions = pd.read_parquet(decisions_path)
    candidates["timestamp"] = pd.to_datetime(candidates["timestamp"], utc=True, errors="coerce")
    candidates = candidates[candidates["timestamp"].notna()].sort_values(["timestamp", "strategy_id", "symbol"]).reset_index(drop=True)
    candidates["head"] = candidates["strategy_id"].map(_head_name)
    return candidates, decisions


def _utc_timestamp(value: str) -> pd.Timestamp | None:
    if not value:
        return None
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _pct_rank(values: pd.Series, group: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    ranked = numeric.groupby(group).rank(method="average", pct=True)
    return ranked.astype("float32")


def _risk_mean(frame: pd.DataFrame, cols: list[str], *, invert: set[str] | None = None) -> pd.Series:
    invert = invert or set()
    if not cols:
        return pd.Series(np.nan, index=frame.index, dtype="float32")
    parts = []
    for col in cols:
        pct = _pct_rank(frame[col], frame["head"])
        if col in invert:
            pct = 1.0 - pct
        parts.append(pct)
    out = pd.concat(parts, axis=1).mean(axis=1, skipna=True)
    return out.fillna(out.median()).astype("float32")


def _add_risk_scores(candidates: pd.DataFrame) -> pd.DataFrame:
    out = candidates.copy()
    out["uncertainty_risk"] = _risk_mean(
        out,
        [
            "generated_score_uncertainty_p1mp",
            "generated_score_entropy",
            "generated_score_abs_distance_from_half",
        ],
        invert={"generated_score_abs_distance_from_half"},
    )
    out["drift_risk"] = _risk_mean(
        out,
        [
            "generated_score_abs_diff_1",
            "generated_score_abs_diff_4",
            "generated_score_abs_diff_24",
            "generated_score_abs_minus_prev24_mean",
            "generated_score_prev24_std",
            "generated_strategy_score_shift_abs_z",
        ],
    )
    out["ood_risk"] = _risk_mean(
        out,
        [
            "generated_strategy_score_ood_abs_z",
            "generated_strategy_barrier_ood_abs_z",
            "generated_strategy_friction_ood_abs_z",
        ],
    )
    out["recent_perf_risk"] = _risk_mean(
        out,
        [
            "generated_hr_surprise_24",
            "generated_hr_surprise_96",
            "generated_weighted_hr_surprise_24",
            "generated_weighted_hr_surprise_96",
            "generated_loss_rate_24",
            "generated_loss_rate_96",
        ],
        invert={
            "generated_hr_surprise_24",
            "generated_hr_surprise_96",
            "generated_weighted_hr_surprise_24",
            "generated_weighted_hr_surprise_96",
        },
    )
    out["friction_risk"] = _risk_mean(
        out,
        [
            "expected_friction_bps",
            "price_gap_bps",
            "entry_gap_bps",
            "entry_slippage_proxy_bps",
            "orderbook_slippage_bps",
            "delay_max_adverse_bps",
            "liquidity_capacity_weight",
        ],
        invert={"liquidity_capacity_weight"},
    )
    out["composite_risk"] = out[list(RISK_SCORE_NAMES[:-1])].mean(axis=1, skipna=True).astype("float32")
    return out


def _rule_mask(candidates: pd.DataFrame, rule: VetoRule) -> pd.Series:
    score = pd.to_numeric(candidates[rule.score_name], errors="coerce")
    if rule.scope == "all":
        scope_mask = pd.Series(True, index=candidates.index)
    else:
        scope_mask = candidates["head"].eq(rule.scope)
    ranks = pd.to_numeric(candidates.get("rank_pct", candidates.get("policy_rank_pct")), errors="coerce").fillna(0.0)
    threshold = score[scope_mask].quantile(rule.risk_quantile) if bool(scope_mask.any()) else np.nan
    if not np.isfinite(float(threshold)):
        return pd.Series(False, index=candidates.index)
    return scope_mask & ranks.ge(rule.min_rank_pct) & score.ge(float(threshold))


def _candidate_rules() -> list[VetoRule]:
    scopes = ("all",) + HEAD_ORDER
    rules: list[VetoRule] = []
    for score_name in RISK_SCORE_NAMES:
        for scope in scopes:
            for risk_quantile in (0.90, 0.95, 0.98):
                for min_rank_pct in (0.70, 0.80, 0.90):
                    rules.append(VetoRule(score_name, scope, risk_quantile, min_rank_pct))
    return rules


def _append_family_representatives(
    shortlist: pd.DataFrame,
    screen: pd.DataFrame,
    score_names: list[str],
) -> pd.DataFrame:
    """Ensure requested diagnostic families get at least one replayed rule."""
    if screen.empty or not score_names:
        return shortlist
    out = shortlist.copy()
    key_cols = ["score_name", "scope", "risk_quantile", "min_rank_pct"]
    existing = {
        tuple(row)
        for row in out[key_cols].itertuples(index=False, name=None)
    } if not out.empty else set()
    extras: list[pd.Series] = []
    for score_name in score_names:
        if out.get("score_name", pd.Series(dtype=object)).astype(str).eq(score_name).any():
            continue
        rows = screen.loc[screen["score_name"].astype(str).eq(score_name)]
        if rows.empty:
            continue
        row = rows.iloc[0]
        key = tuple(row[col] for col in key_cols)
        if key in existing:
            continue
        existing.add(key)
        extras.append(row)
    if extras:
        out = pd.concat([out, pd.DataFrame(extras)], ignore_index=True)
    return out.reset_index(drop=True)


def _screen_rules(candidates: pd.DataFrame, decisions: pd.DataFrame, min_removed: int) -> pd.DataFrame:
    accepted = decisions[decisions["accepted"].astype(bool)].copy()
    if accepted.empty:
        return pd.DataFrame()
    accepted["candidate_index"] = pd.to_numeric(accepted["candidate_index"], errors="coerce").astype("int64")
    accepted = accepted.merge(
        candidates[["head", *RISK_SCORE_NAMES]].reset_index(names="candidate_index"),
        on="candidate_index",
        how="left",
    )
    accepted["net_pnl_amount"] = (
        pd.to_numeric(accepted["position_size"], errors="coerce").fillna(0.0)
        * pd.to_numeric(accepted["position_net_return"], errors="coerce").fillna(0.0)
    )
    accepted["is_win"] = pd.to_numeric(accepted["position_net_return"], errors="coerce").fillna(0.0) > 0.0
    accepted["is_full_sl"] = accepted["position_exit_reason"].astype(str).str.contains("sl", case=False, na=False)
    accepted["is_timeout"] = accepted["position_exit_reason"].astype(str).str.contains("timeout", case=False, na=False)
    accepted_idx = accepted["candidate_index"].to_numpy(dtype=np.int64)
    rows: list[dict[str, object]] = []
    for rule in _candidate_rules():
        mask = _rule_mask(candidates, rule)
        removed = accepted[mask.iloc[accepted_idx].to_numpy(dtype=bool)].copy()
        n = int(len(removed))
        if n < int(min_removed):
            continue
        removed_net = float(removed["net_pnl_amount"].sum())
        removed_full_sl = int(removed["is_full_sl"].sum())
        removed_timeout = int(removed["is_timeout"].sum())
        removed_win = int(removed["is_win"].sum())
        full_sl_rate = float(removed["is_full_sl"].mean()) if n else 0.0
        timeout_rate = float(removed["is_timeout"].mean()) if n else 0.0
        hit_rate = float(removed["is_win"].mean()) if n else 0.0
        # Positive proxy means the removed accepted trades were bad enough that
        # vetoing them may help before backfill effects.
        proxy_delta_net = -removed_net
        proxy_score = proxy_delta_net + 25.0 * removed_full_sl + 10.0 * removed_timeout - 25.0 * removed_win
        rows.append(
            {
                **asdict(rule),
                "candidate_veto_count": int(mask.sum()),
                "removed_accepted_count": n,
                "removed_net_pnl": removed_net,
                "proxy_delta_net_no_backfill": proxy_delta_net,
                "proxy_score": float(proxy_score),
                "removed_full_sl_rate": full_sl_rate,
                "removed_timeout_rate": timeout_rate,
                "removed_hit_rate": hit_rate,
                "removed_full_sl_count": removed_full_sl,
                "removed_timeout_count": removed_timeout,
                "removed_win_count": removed_win,
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(
        ["proxy_score", "proxy_delta_net_no_backfill", "removed_full_sl_rate"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def _apply_veto(candidates: pd.DataFrame, rule: VetoRule) -> tuple[pd.DataFrame, int]:
    out = candidates.copy()
    if "portfolio_rank_adjustment" not in out.columns:
        out["portfolio_rank_adjustment"] = 0.0
    else:
        out["portfolio_rank_adjustment"] = pd.to_numeric(out["portfolio_rank_adjustment"], errors="coerce").fillna(0.0)
    mask = _rule_mask(out, rule)
    out.loc[mask, "portfolio_rank_adjustment"] = -1.0
    return out, int(mask.sum())


def _summarize_replay(label: str, decisions: pd.DataFrame, weekly: pd.DataFrame, metrics: dict[str, Any], q35_weight: float, q20_weight: float) -> dict[str, Any]:
    global_weekly = weekly[weekly["period_type"].eq("week")].copy()
    values = pd.to_numeric(global_weekly["net_pnl"], errors="coerce").dropna().to_numpy(dtype=np.float64)
    accepted = decisions[decisions["accepted"].astype(bool)].copy()
    return {
        "label": label,
        "net_pnl": float(metrics.get("net_pnl", np.nan)),
        "gross_pnl": float(metrics.get("gross_pnl", np.nan)),
        "trade_count": int(metrics.get("trade_count", len(accepted))),
        "full_sl_rate": float(metrics.get("full_sl_rate", np.nan)),
        "timeout_rate": float(metrics.get("timeout_rate", np.nan)),
        "max_drawdown": float(metrics.get("max_drawdown", np.nan)),
        "hit_rate": float((pd.to_numeric(accepted["position_net_return"], errors="coerce") > 0.0).mean()) if len(accepted) else np.nan,
        "objective_week": _objective_from_weekly(weekly, q35_weight, q20_weight),
        "q20_week_net_pnl": float(np.quantile(values, 0.20)) if values.size else np.nan,
        "q35_week_net_pnl": float(np.quantile(values, 0.35)) if values.size else np.nan,
        "worst_week_net_pnl": float(np.min(values)) if values.size else np.nan,
        "positive_weeks": int(np.sum(values > 0.0)) if values.size else 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("data_perp/reports/contextual_tp_sl_materialized_wf_recent_q35w07_q20w03_6mo_20260701"))
    parser.add_argument("--output-dir", type=Path, default=Path("data_perp/reports/contextual_tp_sl_wfrecent_row_execution_guard_20260701"))
    parser.add_argument("--top-rules", type=int, default=24)
    parser.add_argument("--min-removed", type=int, default=12)
    parser.add_argument("--q35-weight", type=float, default=0.70)
    parser.add_argument("--q20-weight", type=float, default=0.30)
    parser.add_argument("--eval-start", default="")
    parser.add_argument("--eval-end", default="")
    parser.add_argument(
        "--ensure-score-names",
        default="",
        help="Comma-separated risk score names whose best screened rule should be replayed even if outside top-rules.",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    candidates_path = args.input_dir / "combo_candidates.parquet"
    decisions_path = args.input_dir / "combo_replay_decisions.parquet"
    candidates_all, baseline_decisions = _load_inputs(candidates_path, decisions_path)
    manifest_path = args.input_dir / "combo_replay_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    eval_start_raw = args.eval_start or str(manifest.get("eval_start") or "")
    eval_end_raw = args.eval_end or str(manifest.get("eval_end") or "")
    eval_start = _utc_timestamp(eval_start_raw)
    eval_end = _utc_timestamp(eval_end_raw)
    eval_mask = pd.Series(True, index=candidates_all.index)
    if eval_start is not None:
        eval_mask &= candidates_all["timestamp"].ge(eval_start)
    if eval_end is not None:
        eval_mask &= candidates_all["timestamp"].le(eval_end)
    candidates = candidates_all.loc[eval_mask].copy().reset_index(drop=True)
    train_cutoff = candidates["timestamp"].min() if not candidates.empty else eval_start
    train_candidates = candidates_all.loc[candidates_all["timestamp"].lt(train_cutoff)].copy().reset_index(drop=True)
    if train_candidates.empty:
        train_candidates = candidates_all.loc[candidates_all["timestamp"].lt(candidates["timestamp"].min())].copy().reset_index(drop=True)
    if candidates.empty or train_candidates.empty:
        raise ValueError("empty train or evaluation candidate slice after applying eval bounds")
    candidates = _add_risk_scores(candidates)
    baseline_daily, baseline_weekly = _period_tables(baseline_decisions)
    baseline_metrics = dict(manifest.get("metrics") or {})
    if not baseline_metrics:
        baseline_metrics = _metrics_from_decisions(baseline_decisions)

    screen = _screen_rules(candidates, baseline_decisions, args.min_removed)
    if screen.empty:
        raise RuntimeError("No row-level guard rules passed the screening minimum")
    shortlist = screen.head(int(args.top_rules)).copy()
    ensure_score_names = [part.strip() for part in str(args.ensure_score_names).split(",") if part.strip()]
    shortlist = _append_family_representatives(shortlist, screen, ensure_score_names)

    params = PortfolioPolicyParams(global_threshold_floor=0.0)
    summaries: list[dict[str, Any]] = []
    replay_rows: list[pd.DataFrame] = []
    baseline_summary = _summarize_replay("wf_recent_baseline", baseline_decisions, baseline_weekly, baseline_metrics, args.q35_weight, args.q20_weight)
    summaries.append({**baseline_summary, "run_id": -1, "vetoed_candidate_count": 0})

    for run_id, row in shortlist.reset_index(drop=True).iterrows():
        rule = VetoRule(
            score_name=str(row["score_name"]),
            scope=str(row["scope"]),
            risk_quantile=float(row["risk_quantile"]),
            min_rank_pct=float(row["min_rank_pct"]),
        )
        guarded, vetoed_count = _apply_veto(candidates, rule)
        ev_curve = fit_hierarchical_ev_curves(train_candidates)
        decisions, equity, metrics = replay_candidates(
            guarded,
            params,
            mode="global_auction",
            ev_curve=ev_curve,
            market_mode="perps",
        )
        daily, weekly = _period_tables(decisions)
        label = f"{rule.scope}__{rule.score_name}__q{int(rule.risk_quantile * 100)}__rank{int(rule.min_rank_pct * 100)}"
        summary = _summarize_replay(label, decisions, weekly, metrics, args.q35_weight, args.q20_weight)
        summary.update(
            {
                "run_id": int(run_id),
                **asdict(rule),
                "vetoed_candidate_count": int(vetoed_count),
                "screen_proxy_score": float(row["proxy_score"]),
                "screen_proxy_delta_net_no_backfill": float(row["proxy_delta_net_no_backfill"]),
                "screen_removed_accepted_count": int(row["removed_accepted_count"]),
                "screen_removed_full_sl_rate": float(row["removed_full_sl_rate"]),
                "screen_removed_timeout_rate": float(row["removed_timeout_rate"]),
                "screen_removed_hit_rate": float(row["removed_hit_rate"]),
            }
        )
        for key in ("net_pnl", "gross_pnl", "objective_week", "q20_week_net_pnl", "q35_week_net_pnl", "worst_week_net_pnl", "full_sl_rate", "timeout_rate", "max_drawdown", "hit_rate"):
            summary[f"delta_{key}_vs_baseline"] = float(summary[key] - baseline_summary[key])
        summary["delta_trade_count_vs_baseline"] = int(summary["trade_count"] - baseline_summary["trade_count"])
        summaries.append(summary)
        weekly = weekly.copy()
        weekly["run_id"] = int(run_id)
        weekly["label"] = label
        replay_rows.append(weekly)

    summary_df = pd.DataFrame(summaries)
    guard_df = summary_df[summary_df["run_id"].ge(0)].copy()
    guard_df = guard_df.sort_values(
        ["delta_objective_week_vs_baseline", "delta_net_pnl_vs_baseline", "delta_full_sl_rate_vs_baseline"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    summary_out = pd.concat([summary_df[summary_df["run_id"].lt(0)], guard_df], ignore_index=True)
    weekly_out = pd.concat(replay_rows, ignore_index=True) if replay_rows else pd.DataFrame()

    screen.to_csv(args.output_dir / "row_execution_guard_screen.csv", index=False)
    shortlist.to_csv(args.output_dir / "row_execution_guard_shortlist.csv", index=False)
    summary_out.to_csv(args.output_dir / "row_execution_guard_replay_summary.csv", index=False)
    weekly_out.to_csv(args.output_dir / "row_execution_guard_replay_weekly.csv", index=False)
    baseline_weekly.to_csv(args.output_dir / "row_execution_guard_baseline_weekly.csv", index=False)
    manifest_out = {
        "generated_by": "ablate_wfrecent_row_execution_guard",
        "input_dir": str(args.input_dir),
        "candidate_rows": int(len(candidates_all)),
        "train_candidate_rows": int(len(train_candidates)),
        "eval_candidate_rows": int(len(candidates)),
        "baseline_decision_rows": int(len(baseline_decisions)),
        "eval_start": eval_start.isoformat() if eval_start is not None else "",
        "eval_end": eval_end.isoformat() if eval_end is not None else "",
        "screened_rules": int(len(screen)),
        "replayed_rules": int(len(shortlist)),
        "ensure_score_names": ensure_score_names,
        "q35_weight": float(args.q35_weight),
        "q20_weight": float(args.q20_weight),
        "min_removed": int(args.min_removed),
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(_json_safe(manifest_out), indent=2, sort_keys=True) + "\n")

    best = guard_df.iloc[0] if not guard_df.empty else pd.Series(dtype=object)
    best_tail = (
        guard_df.sort_values(["delta_full_sl_rate_vs_baseline", "delta_timeout_rate_vs_baseline", "delta_net_pnl_vs_baseline"], ascending=[True, True, False]).iloc[0]
        if not guard_df.empty
        else pd.Series(dtype=object)
    )
    lines = [
        "# wf_recent Row-Level Execution Guard",
        "",
        "This is a full portfolio replay ablation over shortlisted row-level veto rules. It uses existing `wf_recent` materialized candidates and sets `portfolio_rank_adjustment = -1.0` for vetoed candidates so the global auction can backfill naturally.",
        "",
        "Risk families tested: uncertainty, drift, OOD, recent hit-rate surprise/loss-rate, friction, and composite risk.",
        "",
        "## Baseline",
        "",
        _fmt_table(pd.DataFrame([baseline_summary]), ["label", "net_pnl", "gross_pnl", "trade_count", "hit_rate", "full_sl_rate", "timeout_rate", "max_drawdown", "objective_week", "q20_week_net_pnl", "worst_week_net_pnl"]),
        "",
        "## Best Replayed Guard By Objective",
        "",
        _fmt_table(pd.DataFrame([best]) if not best.empty else pd.DataFrame(), ["label", "score_name", "scope", "risk_quantile", "min_rank_pct", "vetoed_candidate_count", "net_pnl", "delta_net_pnl_vs_baseline", "objective_week", "delta_objective_week_vs_baseline", "full_sl_rate", "delta_full_sl_rate_vs_baseline", "timeout_rate", "delta_timeout_rate_vs_baseline", "worst_week_net_pnl", "delta_worst_week_net_pnl_vs_baseline"]),
        "",
        "## Best Replayed Guard By Full-SL Reduction",
        "",
        _fmt_table(pd.DataFrame([best_tail]) if not best_tail.empty else pd.DataFrame(), ["label", "score_name", "scope", "risk_quantile", "min_rank_pct", "vetoed_candidate_count", "net_pnl", "delta_net_pnl_vs_baseline", "objective_week", "delta_objective_week_vs_baseline", "full_sl_rate", "delta_full_sl_rate_vs_baseline", "timeout_rate", "delta_timeout_rate_vs_baseline", "worst_week_net_pnl", "delta_worst_week_net_pnl_vs_baseline"]),
        "",
        "## Top Replayed Guards",
        "",
        _fmt_table(guard_df, ["label", "score_name", "scope", "risk_quantile", "min_rank_pct", "vetoed_candidate_count", "delta_net_pnl_vs_baseline", "delta_objective_week_vs_baseline", "delta_full_sl_rate_vs_baseline", "delta_timeout_rate_vs_baseline", "delta_worst_week_net_pnl_vs_baseline", "delta_trade_count_vs_baseline"], max_rows=20),
        "",
        "## Proxy Screen Top Rules",
        "",
        _fmt_table(screen, ["score_name", "scope", "risk_quantile", "min_rank_pct", "candidate_veto_count", "removed_accepted_count", "proxy_delta_net_no_backfill", "proxy_score", "removed_full_sl_rate", "removed_timeout_rate", "removed_hit_rate"], max_rows=20),
    ]
    (args.output_dir / "row_execution_guard_report.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
