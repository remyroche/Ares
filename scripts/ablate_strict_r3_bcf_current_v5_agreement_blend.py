#!/usr/bin/env python3
"""Common-state BCF/current-v5 MC1 agreement and blend ablation.

This is deliberately a *post-mapper* research ablation.  It consumes the
already frozen, strictly prequential MC1_d2 predictions and canonical policy
outcomes.  A common candidate must clear the specified threshold in *both*
maps; BCF therefore acts as a conservative reliability confirmation rather
than manufacturing an admission.  The only grid degree of freedom is auction
priority: BCF MC1 EV or a predeclared BCF/current-v5 mapped-EV blend.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.portfolio_policy_replay import (  # noqa: E402
    normalise_candidate_table,
    replay_candidates,
)
from scripts.report_strict_r3_mc1_d2_controlled_portfolio import (  # noqa: E402
    CAUSAL_AUCTION_CURVE,
    _metrics,
    _params,
)


POLICY_COLUMNS = (
    "policy_path_valid", "policy_gross_bps", "policy_net_bps",
    "policy_exit_bar_15m", "policy_entry_price", "policy_exit_price",
    "policy_exit_reason", "policy_label_available_ts", "policy_outcome_source",
    "policy_cost_bps",
)
THRESHOLDS = (30.0, 40.0, 50.0, 60.0)
BCF_WEIGHTS = (0.10, 0.25, 1.0 / 3.0, 0.50)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    frame["__decision_ts__"] = pd.to_datetime(frame["__decision_ts__"], utc=True, errors="raise")
    if frame["candidate_id"].duplicated().any():
        raise ValueError(f"duplicate candidate identity in {path}")
    return frame


def _policy_equal(left: pd.DataFrame, right: pd.DataFrame) -> None:
    for column in ("__decision_ts__", *POLICY_COLUMNS):
        lhs, rhs = left[f"{column}_bcf"], right[f"{column}_current"]
        if pd.api.types.is_numeric_dtype(lhs):
            equal = np.isclose(lhs.to_numpy(float), rhs.to_numpy(float), equal_nan=True).all()
        else:
            equal = lhs.fillna("__null__").astype(str).equals(rhs.fillna("__null__").astype(str))
        if not equal:
            raise AssertionError(f"canonical policy mismatch across score families: {column}")


def _common_panel(bcf: pd.DataFrame, current: pd.DataFrame) -> pd.DataFrame:
    bcf_cols = ["candidate_id", "__decision_ts__", "__symbol__", "final_score", "mc1_expected_bps", *POLICY_COLUMNS]
    current_cols = ["candidate_id", "__decision_ts__", "final_score", "mc1_expected_bps", *POLICY_COLUMNS]
    merged = bcf.loc[:, bcf_cols].merge(
        current.loc[:, current_cols], on="candidate_id", suffixes=("_bcf", "_current"), validate="one_to_one",
    )
    _policy_equal(merged, merged)
    out = pd.DataFrame({
        "candidate_id": merged["candidate_id"].astype(str),
        "__decision_ts__": merged["__decision_ts___bcf"],
        "__symbol__": merged["__symbol__"].astype(str),
        "bcf_final_score": pd.to_numeric(merged["final_score_bcf"], errors="raise"),
        "bcf_mc1_expected_bps": pd.to_numeric(merged["mc1_expected_bps_bcf"], errors="raise"),
        "current_mc1_expected_bps": pd.to_numeric(merged["mc1_expected_bps_current"], errors="raise"),
    })
    for field in POLICY_COLUMNS:
        out[field] = merged[f"{field}_bcf"]
    return out


def _valid(panel: pd.DataFrame) -> pd.Series:
    return (
        panel["policy_path_valid"].fillna(False).astype(bool)
        & np.isfinite(pd.to_numeric(panel["policy_net_bps"], errors="coerce"))
        & np.isfinite(pd.to_numeric(panel["policy_gross_bps"], errors="coerce"))
        & np.isfinite(pd.to_numeric(panel["policy_exit_bar_15m"], errors="coerce"))
        & np.isfinite(pd.to_numeric(panel["policy_entry_price"], errors="coerce"))
        & np.isfinite(pd.to_numeric(panel["policy_exit_price"], errors="coerce"))
    )


def _to_candidates(panel: pd.DataFrame, *, admission: pd.Series, priority: pd.Series) -> pd.DataFrame:
    selected = panel.loc[_valid(panel) & admission].copy()
    selected["priority_bps"] = pd.to_numeric(priority.loc[selected.index], errors="raise")
    if selected.empty:
        return pd.DataFrame()
    selected["auction_rank"] = selected.groupby("__decision_ts__", sort=False)["priority_bps"].rank(
        pct=True, method="average",
    )
    exit_bar = pd.to_numeric(selected["policy_exit_bar_15m"], errors="raise").astype(int)
    decision = selected["__decision_ts__"]
    candidate = pd.DataFrame({
        "timestamp": decision,
        "symbol": selected["__symbol__"].astype(str),
        "side": "long",
        "strategy_id": "strict_r3_bcf_current_v5_agreement_long",
        "policy_archetype": "strict_r3_bcf_current_v5_agreement_long",
        "normalized_rank_score": selected["auction_rank"].to_numpy(float),
        "strategy_rank_pct": selected["auction_rank"].to_numpy(float),
        "base_strategy_threshold": 0.0,
        "calibrated_score": selected["auction_rank"].to_numpy(float),
        "entry_price": pd.to_numeric(selected["policy_entry_price"], errors="raise"),
        "exit_timestamp": decision + pd.to_timedelta((exit_bar + 1) * 15, unit="min"),
        "exit_price": pd.to_numeric(selected["policy_exit_price"], errors="raise"),
        "net_return": pd.to_numeric(selected["policy_net_bps"], errors="raise") / 10_000.0,
        "gross_return": pd.to_numeric(selected["policy_gross_bps"], errors="raise") / 10_000.0,
        "holding_bars": exit_bar + 1,
        "simple_policy_exit_reason": selected["policy_exit_reason"].astype(str),
        "fees_bps": pd.to_numeric(selected["policy_cost_bps"], errors="raise"),
        "slippage_bps": 0.0,
        "expected_friction_bps": pd.to_numeric(selected["policy_cost_bps"], errors="raise"),
        "price_gap_bps": 0.0,
        "liquidity_capacity_weight": 1.0,
        "source_month": decision.dt.strftime("%Y-%m"),
        "candidate_id": selected["candidate_id"].astype(str),
        "mapped_expected_net_bps": selected["priority_bps"],
        "policy_outcome_available": np.ones(len(selected), dtype=bool),
    })
    return normalise_candidate_table(candidate)


def _replay(panel: pd.DataFrame, *, arm: str, admission: pd.Series, priority: pd.Series, years: tuple[int, ...], out_dir: Path) -> tuple[list[dict[str, object]], pd.DataFrame]:
    metrics: list[dict[str, object]] = []
    decision_blocks: list[pd.DataFrame] = []
    for year in years:
        piece = panel.loc[panel["__decision_ts__"].dt.year.eq(year)].copy()
        candidates = _to_candidates(piece, admission=admission.loc[piece.index], priority=priority.loc[piece.index])
        decisions, equity, _ = replay_candidates(
            candidates, _params(), mode="global_auction", ev_curve=CAUSAL_AUCTION_CURVE,
            market_mode="perps", initial_wallet=1000.0,
        )
        if decisions.empty:
            decisions = decisions.copy()
            decisions["policy_outcome_available"] = pd.Series(dtype=bool)
        else:
            provenance = candidates.loc[:, ["candidate_id", "policy_outcome_available"]].reset_index(drop=True)
            provenance.index.name = "candidate_index"
            decisions = decisions.merge(provenance, on="candidate_index", how="left", validate="many_to_one")
            if decisions["policy_outcome_available"].isna().any():
                raise AssertionError("portfolio decision lacks canonical policy provenance")
        decisions.to_parquet(out_dir / f"{arm}_{year}_decisions.parquet", index=False, compression="zstd")
        equity.to_parquet(out_dir / f"{arm}_{year}_equity.parquet", index=False, compression="zstd")
        metric = _metrics(decisions, equity, arm, str(year))
        metric["admitted_candidates"] = int(len(candidates))
        metric["admission_threshold_bps"] = float("nan")
        metrics.append(metric)
        decisions["arm"] = arm
        decisions["year"] = year
        decision_blocks.append(decisions)
    return metrics, pd.concat(decision_blocks, ignore_index=True) if decision_blocks else pd.DataFrame()


def _daily_metrics(decisions: pd.DataFrame, *, arm: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    days = pd.date_range(start.normalize(), end.normalize() - pd.Timedelta(days=1), freq="1D", tz="UTC")
    accepted = decisions.loc[decisions.get("accepted", pd.Series(index=decisions.index, dtype=bool)).fillna(False).astype(bool)].copy()
    if not accepted.empty:
        accepted["day"] = pd.to_datetime(accepted["timestamp"], utc=True).dt.normalize()
        grouped = accepted.groupby("day", sort=True)["position_net_return"].agg(["count", "mean", "sum"])
    else:
        grouped = pd.DataFrame(columns=["count", "mean", "sum"])
    out = pd.DataFrame({"day": days}).merge(grouped, left_on="day", right_index=True, how="left")
    out["arm"] = arm
    out["trades"] = out["count"].fillna(0).astype(int)
    out["net_ev_bps_per_trade"] = out["mean"] * 10_000.0
    out["net_sum_bps"] = out["sum"].fillna(0.0) * 10_000.0
    return out.loc[:, ["arm", "day", "trades", "net_ev_bps_per_trade", "net_sum_bps"]]


def _summary(metrics: pd.DataFrame, daily: pd.DataFrame, *, baseline_arm: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for arm, piece in metrics.groupby("arm", sort=True):
        accepted = int(piece["accepted_rows"].sum())
        total = float(piece["net_sum_bps_realised"].sum())
        days = daily.loc[daily["arm"].eq(arm)]
        rows.append({
            "arm": arm,
            "portfolio_accepted_trades": accepted,
            "trades_per_calendar_day": accepted / len(days) if len(days) else float("nan"),
            "days_lt_1_trade": int(days["trades"].lt(1).sum()),
            "days_lt_5_trades": int(days["trades"].lt(5).sum()),
            "net_ev_bps_per_trade": total / accepted if accepted else float("nan"),
            "net_sum_bps": total,
            "worst_month_bps": float(piece["worst_month_bps"].min()),
            "worst_week_bps": float(piece["worst_week_bps"].min()),
            "worst_year_reset_max_drawdown": float(piece["max_drawdown"].min()),
        })
    out = pd.DataFrame(rows)
    baseline = out.loc[out["arm"].eq(baseline_arm)].iloc[0]
    for column in (
        "portfolio_accepted_trades", "trades_per_calendar_day", "days_lt_1_trade",
        "days_lt_5_trades", "net_ev_bps_per_trade", "net_sum_bps", "worst_month_bps",
        "worst_week_bps", "worst_year_reset_max_drawdown",
    ):
        out[f"delta_vs_{baseline_arm}_{column}"] = out[column] - baseline[column]
    return out


def _choose(summary: pd.DataFrame, *, baseline_arm: str) -> pd.DataFrame:
    """Predeclared balanced screen: no material EV-rate or drought regression."""
    base = summary.loc[summary["arm"].eq(baseline_arm)].iloc[0]
    candidates = summary.loc[~summary["arm"].eq(baseline_arm)].copy()
    candidates["passes_balance_gate"] = (
        candidates["net_sum_bps"].gt(base["net_sum_bps"])
        & candidates["net_ev_bps_per_trade"].ge(0.97 * base["net_ev_bps_per_trade"])
        & candidates["days_lt_1_trade"].le(base["days_lt_1_trade"] + 5)
    )
    candidates = candidates.sort_values(
        ["passes_balance_gate", "net_sum_bps", "net_ev_bps_per_trade", "days_lt_1_trade"],
        ascending=[False, False, False, True], kind="stable",
    )
    return candidates.reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bcf-predictions", required=True, type=Path)
    parser.add_argument("--current-predictions", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--years", nargs="+", type=int, default=[2026])
    parser.add_argument("--thresholds", nargs="+", type=float, default=list(THRESHOLDS))
    parser.add_argument("--bcf-weights", nargs="*", type=float, default=list(BCF_WEIGHTS))
    parser.add_argument("--include-bcf-priority", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(f"immutable output exists: {args.out_dir}")
    args.out_dir.mkdir(parents=True)
    years = tuple(sorted(set(args.years)))
    bcf, current = _read(args.bcf_predictions), _read(args.current_predictions)
    common = _common_panel(bcf, current)
    common = common.loc[common["__decision_ts__"].dt.year.isin(years)].copy().reset_index(drop=True)
    current_full = current.loc[current["__decision_ts__"].dt.year.isin(years)].copy().reset_index(drop=True)
    # Current-v5 full-route control: identical frozen map and policy; only the
    # BCF agreement gate is absent.  It is the requested challenger baseline.
    current_full = current_full.rename(columns={
        "mc1_expected_bps": "current_mc1_expected_bps",
        "__symbol__": "__symbol__",
    })
    current_full["bcf_mc1_expected_bps"] = np.nan
    current_full["bcf_final_score"] = np.nan

    metric_rows: list[dict[str, object]] = []
    daily_blocks: list[pd.DataFrame] = []
    decisions_root = args.out_dir / "decisions"
    decisions_root.mkdir()
    control_admit = current_full["current_mc1_expected_bps"].ge(50.0)
    control_metrics, control_decisions = _replay(
        current_full, arm="current_v5_full_t50", admission=control_admit,
        priority=current_full["current_mc1_expected_bps"], years=years, out_dir=decisions_root,
    )
    metric_rows.extend(control_metrics)
    start, end = common["__decision_ts__"].min(), common["__decision_ts__"].max() + pd.Timedelta(days=1)
    daily_blocks.append(_daily_metrics(control_decisions, arm="current_v5_full_t50", start=start, end=end))

    thresholds = tuple(args.thresholds)
    bcf_weights = tuple(args.bcf_weights)
    if not thresholds:
        raise ValueError("at least one agreement threshold is required")
    if any(weight < 0.0 or weight > 1.0 for weight in bcf_weights):
        raise ValueError("BCF blend weights must lie in [0, 1]")
    for threshold in thresholds:
        agreement = common["bcf_mc1_expected_bps"].ge(threshold) & common["current_mc1_expected_bps"].ge(threshold)
        controls = (
            ([("bcf_priority", 1.0)] if args.include_bcf_priority else [])
            + [(f"blend_bcf{weight:g}", weight) for weight in bcf_weights]
        )
        if not controls:
            raise ValueError("at least one priority rule is required")
        for name, weight in controls:
            priority = weight * common["bcf_mc1_expected_bps"] + (1.0 - weight) * common["current_mc1_expected_bps"]
            arm = f"both_t{threshold:g}_{name}"
            metrics, decisions = _replay(common, arm=arm, admission=agreement, priority=priority, years=years, out_dir=decisions_root)
            for metric in metrics:
                metric["agreement_threshold_bps"] = threshold
                metric["bcf_priority_weight"] = weight
                metric["priority_rule"] = name
            metric_rows.extend(metrics)
            daily_blocks.append(_daily_metrics(decisions, arm=arm, start=start, end=end))
            print(json.dumps({"event": "arm_complete", "arm": arm}), flush=True)

    metrics = pd.DataFrame(metric_rows)
    daily = pd.concat(daily_blocks, ignore_index=True)
    summary = _summary(metrics, daily, baseline_arm="current_v5_full_t50")
    ranking = _choose(summary, baseline_arm="current_v5_full_t50")
    metrics.to_parquet(args.out_dir / "portfolio_metrics.parquet", index=False)
    metrics.to_csv(args.out_dir / "portfolio_metrics.csv", index=False)
    daily.to_parquet(args.out_dir / "daily_constrained_metrics.parquet", index=False)
    daily.to_csv(args.out_dir / "daily_constrained_metrics.csv", index=False)
    summary.to_parquet(args.out_dir / "summary_metrics.parquet", index=False)
    summary.to_csv(args.out_dir / "summary_metrics.csv", index=False)
    ranking.to_parquet(args.out_dir / "selection_ranking.parquet", index=False)
    ranking.to_csv(args.out_dir / "selection_ranking.csv", index=False)
    manifest = {
        "schema": "strict_r3_bcf_current_v5_agreement_blend_v1",
        "status": "complete",
        "purpose": "research-only common-state conservative BCF agreement grid; no mapper or live-contract retraining",
        "inputs": {
            "bcf_predictions": {"path": str(args.bcf_predictions), "sha256": _sha256(args.bcf_predictions)},
            "current_v5_predictions": {"path": str(args.current_predictions), "sha256": _sha256(args.current_predictions)},
        },
        "years": list(years),
        "common_route": "both maps must clear the same threshold; invalid policy outcomes excluded before capacity allocation",
        "grid": {"thresholds_bps": list(thresholds), "bcf_weights": list(bcf_weights), "include_bcf_priority": bool(args.include_bcf_priority), "priority": "weighted BCF/current MC1 expected EV; BCF-only priority control included when requested"},
        "portfolio": "long-only, 7x, 10%-margin slots, two new entries per hour, eight concurrent, 80% wallet cap",
        "selection": "total bps uplift, <=3% EV/trade reduction, and no more than five extra zero-trade days; then maximise total bps, EV/trade, fewer drought days",
    }
    (args.out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
