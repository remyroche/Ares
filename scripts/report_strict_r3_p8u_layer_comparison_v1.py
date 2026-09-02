#!/usr/bin/env python3
"""Matched offline P8u Router/Base/Meta/MC1 research audit.

This report intentionally reads only immutable, target-free score receipts and
joins the canonical rich-policy labels *after* the score panels are assembled.
It compares the successor P8u/F72/Under-F120 stack with the pre-feature-
selection routed-only stack on exact candidate intersections, then replays the
already-fitted strict-prequential MC1 maps through the same constrained
portfolio engine at declared admission floors and timestamp capacities.

It is reporting/replay code only: no model fitting, no live-state access and
no exchange I/O.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from dataclasses import replace
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

IDENTITY = ("candidate_id", "__decision_ts__", "side_name")
FRACTIONS = (0.01, 0.02, 0.05, 0.10, 0.15, 0.20)
ROUTER_THRESHOLDS = (50.0, 100.0, 150.0, 200.0)
MC1_FLOORS = (30.0, 40.0, 50.0)


def _once(path: Path, payload: object) -> None:
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _month_paths(root: Path, pattern: str) -> list[Path]:
    paths = sorted(root.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"no score partitions under {root} matching {pattern}")
    return paths


def _load_router(root: Path) -> pd.DataFrame:
    parts = [pd.read_parquet(path, columns=[*IDENTITY, "router_primary_rank"])
             for path in _month_paths(root, "target_free_scores/month=*.parquet")]
    out = pd.concat(parts, ignore_index=True)
    return _clean_scores(out, "router_primary_rank")


def _load_base(root: Path) -> pd.DataFrame:
    parts = [pd.read_parquet(path, columns=[*IDENTITY, "base_rank_ts"])
             for path in _month_paths(root, "target_free_scores/month=*.parquet")]
    out = pd.concat(parts, ignore_index=True)
    return _clean_scores(out, "base_rank_ts")


def _load_meta(root: Path, arm: str) -> pd.DataFrame:
    parts = [pd.read_parquet(path, columns=[*IDENTITY, "meta_rank_ts"])
             for path in _month_paths(root, f"target_free_scores/{arm}/month=*.parquet")]
    out = pd.concat(parts, ignore_index=True)
    return _clean_scores(out, "meta_rank_ts")


def _clean_scores(frame: pd.DataFrame, score: str) -> pd.DataFrame:
    out = frame.copy()
    out["candidate_id"] = out["candidate_id"].astype(str)
    out["__decision_ts__"] = pd.to_datetime(out["__decision_ts__"], utc=True, errors="raise")
    out["side_name"] = out["side_name"].astype(str).str.lower()
    if out.duplicated(list(IDENTITY)).any() or not out["side_name"].eq("long").all():
        raise AssertionError(f"invalid long score identity for {score}")
    values = pd.to_numeric(out[score], errors="coerce")
    if not np.isfinite(values).all():
        raise AssertionError(f"non-finite {score}")
    out[score] = values.astype(np.float32)
    return out.sort_values(["__decision_ts__", "candidate_id"], kind="stable").reset_index(drop=True)


def _policy(path: Path) -> pd.DataFrame:
    columns = [
        "candidate_id", "policy_path_valid", "policy_net_bps", "policy_gross_bps",
        "policy_exit_bar_15m", "policy_entry_price", "policy_exit_price",
        "policy_exit_reason", "policy_label_available_ts", "policy_cost_bps",
    ]
    out = pd.read_parquet(path, columns=columns)
    out["candidate_id"] = out["candidate_id"].astype(str)
    out["policy_label_available_ts"] = pd.to_datetime(out["policy_label_available_ts"], utc=True, errors="raise")
    if out.candidate_id.duplicated().any():
        raise AssertionError("policy candidate identity is not unique")
    return out


def _join_policy(scores: pd.DataFrame, policy: pd.DataFrame) -> pd.DataFrame:
    out = scores.merge(policy, on="candidate_id", how="left", validate="one_to_one")
    if len(out) != len(scores) or out.policy_path_valid.isna().any():
        raise AssertionError("policy coverage differs from score identity")
    valid = (
        out.policy_path_valid.fillna(False).astype(bool)
        & np.isfinite(pd.to_numeric(out.policy_net_bps, errors="coerce"))
    )
    return out.loc[valid].copy()


def _top_mask(frame: pd.DataFrame, score: str, fraction: float) -> np.ndarray:
    work = frame.loc[:, ["__decision_ts__", "candidate_id", score]].copy()
    work["__row__"] = np.arange(len(work), dtype=np.int64)
    work = work.sort_values(["__decision_ts__", score, "candidate_id"], ascending=[True, False, True], kind="stable")
    ordinal = work.groupby("__decision_ts__", sort=False).cumcount().to_numpy(np.int64)
    size = work.groupby("__decision_ts__", sort=False)["candidate_id"].transform("size").to_numpy(np.int64)
    keep_sorted = ordinal < np.maximum(1, np.ceil(size * float(fraction)).astype(np.int64))
    out = np.zeros(len(frame), dtype=bool)
    out[work["__row__"].to_numpy(np.int64)] = keep_sorted
    return out


def _tail_metrics(frame: pd.DataFrame, score: str, *, label: str, scope: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    work = frame.copy()
    work["month"] = work["__decision_ts__"].dt.strftime("%Y-%m")
    for fraction in FRACTIONS:
        selected = work.loc[_top_mask(work, score, fraction)].copy()
        grouped = selected.groupby("__decision_ts__", sort=True)["policy_net_bps"]
        timestamp = grouped.agg(
            timestamp_mean_net_bps="mean",
            timestamp_hit_rate_positive=lambda s: float((s > 0.0).mean()),
            timestamp_precision_gt50=lambda s: float((s > 50.0).mean()),
        ).reset_index()
        for split, sub in (("global", timestamp), ("month", timestamp.assign(month=timestamp["__decision_ts__"].dt.strftime("%Y-%m")))):
            if split == "global":
                parts = [("all", sub)]
            else:
                parts = list(sub.groupby("month", sort=True))
            for period, metric in parts:
                rows.append({
                    "scope": scope, "label": label, "period_kind": split, "period": str(period),
                    "fraction": fraction, "timestamps": int(len(metric)), "selected_rows": int(len(selected) if split == "global" else selected[selected.month.eq(period)].shape[0]),
                    "timestamp_mean_net_bps": float(metric.timestamp_mean_net_bps.mean()),
                    "timestamp_hit_rate_positive": float(metric.timestamp_hit_rate_positive.mean()),
                    "timestamp_precision_gt50": float(metric.timestamp_precision_gt50.mean()),
                    "timestamp_worst_net_bps": float(metric.timestamp_mean_net_bps.min()),
                })
    return pd.DataFrame(rows)


def _router_recall(frame: pd.DataFrame, score: str, *, label: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    selected = frame.loc[_top_mask(frame, score, .50)].copy()
    for threshold in ROUTER_THRESHOLDS:
        denominator = frame.assign(hit=frame.policy_net_bps.gt(threshold)).groupby("__decision_ts__", sort=True).hit.sum()
        numerator = selected.assign(hit=selected.policy_net_bps.gt(threshold)).groupby("__decision_ts__", sort=True).hit.sum()
        merged = pd.DataFrame({"total": denominator, "selected": numerator}).fillna(0.0)
        eligible = merged.total.gt(0)
        metric = merged.loc[eligible].copy()
        metric["recall"] = metric.selected / metric.total
        calendar = metric.reset_index()
        for period_kind, sub in (("global", calendar), ("month", calendar.assign(month=calendar["__decision_ts__"].dt.strftime("%Y-%m")))):
            groups = [("all", sub)] if period_kind == "global" else list(sub.groupby("month", sort=True))
            for period, part in groups:
                rows.append({"label": label, "period_kind": period_kind, "period": str(period),
                             "threshold_bps": threshold, "timestamps_with_oracle": int(len(part)),
                             "top50_recall": float(part.recall.mean()),
                             "selected_hits": float(part.selected.sum()), "oracle_hits": float(part.total.sum())})
    return pd.DataFrame(rows), selected


def _cmi_binned(frame: pd.DataFrame, *, base: str, meta: str, label: str) -> pd.DataFrame:
    # Ten fixed percentile bins for score coordinates and policy outcome.  This
    # is deliberately a transparent conditional-MI proxy, not a learned model.
    work = frame.loc[:, ["__decision_ts__", "policy_net_bps", base, meta]].copy()
    work["month"] = work["__decision_ts__"].dt.strftime("%Y-%m")
    base_bin = np.minimum(9, np.floor(pd.to_numeric(work[base], errors="coerce").to_numpy(float) * 10.0).astype(int))
    meta_bin = np.minimum(9, np.floor(pd.to_numeric(work[meta], errors="coerce").to_numpy(float) * 10.0).astype(int))
    target_bin = np.digitize(pd.to_numeric(work.policy_net_bps, errors="coerce").to_numpy(float), [-200, -50, 0, 50, 100, 200, 400], right=True)
    work["base_bin"], work["meta_bin"], work["target_bin"] = base_bin, meta_bin, target_bin
    rows=[]
    for period_kind, sub in (("global", work), ("month", work)):
        groups = [("all", sub)] if period_kind == "global" else list(sub.groupby("month", sort=True))
        for period, part in groups:
            n = float(len(part)); value=0.0
            for _b, cell in part.groupby("base_bin", sort=True):
                if len(cell) < 20: continue
                joint = cell.groupby(["meta_bin", "target_bin"], sort=False).size().astype(float) / len(cell)
                pm = cell.groupby("meta_bin", sort=False).size().astype(float) / len(cell)
                py = cell.groupby("target_bin", sort=False).size().astype(float) / len(cell)
                mi = 0.0
                for (m,y), prob in joint.items():
                    mi += float(prob) * math.log(max(float(prob) / max(float(pm.loc[m])*float(py.loc[y]),1e-12),1e-12))
                value += len(cell)/n*mi
            rows.append({"label":label,"period_kind":period_kind,"period":str(period),"rows":int(len(part)),"conditional_mi_nats":value})
    return pd.DataFrame(rows)


def _oracle(frame: pd.DataFrame, *, candidate_scope: str, fractions: Sequence[float] = (0.02, .05, .10)) -> pd.DataFrame:
    rows=[]
    for fraction in fractions:
        mask=_top_mask(frame, "policy_net_bps", float(fraction))
        chosen=frame.loc[mask]
        per=chosen.groupby("__decision_ts__",sort=True).policy_net_bps.mean()
        rows.append({"scope":candidate_scope,"fraction":fraction,"timestamps":int(len(per)),"oracle_timestamp_mean_net_bps":float(per.mean()),"oracle_worst_timestamp_net_bps":float(per.min()),"oracle_rows":int(len(chosen))})
    return pd.DataFrame(rows)


def _legacy_previous(path: Path, policy: pd.DataFrame) -> pd.DataFrame:
    cols = ["candidate_id", "__decision_ts__", "side_name", "current_final_score", "bcf_final_score", "enhanced_base_routed"]
    raw = pd.read_parquet(path, columns=cols).rename(columns={
        "bcf_final_score": "legacy_base_rank",
        "current_final_score": "legacy_meta_rank",
    })
    raw = _clean_scores(raw, "legacy_base_rank")
    raw["legacy_meta_rank"] = pd.to_numeric(raw["legacy_meta_rank"], errors="coerce").astype(np.float32)
    if not np.isfinite(raw["legacy_meta_rank"].to_numpy(float)).all():
        raise AssertionError("previous routed-only Meta score is non-finite")
    raw = raw.loc[raw.enhanced_base_routed.fillna(False).astype(bool)].copy()
    return _join_policy(raw, policy)


def _current_layer_panel(base: pd.DataFrame, meta: pd.DataFrame, policy: pd.DataFrame) -> pd.DataFrame:
    merged = base.merge(meta.loc[:, [*IDENTITY, "meta_rank_ts"]], on=list(IDENTITY), how="inner", validate="one_to_one")
    merged = _join_policy(merged, policy)
    merged["current_score"] = .75 * merged.base_rank_ts + .25 * merged.meta_rank_ts
    # A weighted rank blend needs a second causal timestamp-local rank.
    values = merged.loc[:, ["__decision_ts__", "candidate_id", "current_score"]].copy()
    values["__row__"] = np.arange(len(values), dtype=np.int64)
    values=values.sort_values(["__decision_ts__","current_score","candidate_id"],ascending=[True,False,True],kind="stable")
    ordinal=values.groupby("__decision_ts__",sort=False).cumcount().to_numpy(float)+1
    count=values.groupby("__decision_ts__",sort=False).candidate_id.transform("size").to_numpy(float)
    rank=np.empty(len(values),dtype=np.float32);rank[values.__row__.to_numpy(np.int64)]=1-(ordinal-.5)/count
    merged["current_score_rank"] = rank
    return merged


def _admission_metrics(frame: pd.DataFrame, *, label: str, floors: Sequence[float]) -> pd.DataFrame:
    valid = frame.policy_path_valid.fillna(False).astype(bool) & np.isfinite(pd.to_numeric(frame.policy_net_bps, errors="coerce"))
    rows=[]
    for floor in floors:
        admitted=frame.loc[valid & frame.current_mc1_expected_bps.ge(floor) & frame.bcf_mc1_expected_bps.ge(floor)].copy()
        for period_kind, sub in (("global", admitted), ("month", admitted.assign(month=admitted.__decision_ts__.dt.strftime("%Y-%m")))):
            groups=[("all",sub)] if period_kind=="global" else list(sub.groupby("month",sort=True))
            for period, part in groups:
                rows.append({"label":label,"floor_bps":float(floor),"period_kind":period_kind,"period":str(period),"admitted_rows":int(len(part)),"timestamps":int(part.__decision_ts__.nunique()),"net_ev_bps_per_trade":float(part.policy_net_bps.mean()) if len(part) else float('nan'),"hit_rate_positive":float(part.policy_net_bps.gt(0).mean()) if len(part) else float('nan'),"precision_gt50":float(part.policy_net_bps.gt(50).mean()) if len(part) else float('nan')})
    return pd.DataFrame(rows)


def _mc1_tail_metrics(frame: pd.DataFrame, score: str, *, label: str) -> pd.DataFrame:
    """Memory-bounded MC1 timestamp-tail conversion diagnostic.

    MC1 panels are much wider and larger than the matched Base/Meta panel.
    This implementation retains only the three required columns, computes the
    timestamp rank once, and then evaluates the requested 2/5/10% cuts.  It
    intentionally does not reuse ``_tail_metrics`` because that helper makes
    full-frame copies per cut and can exhaust memory on the long MC1 ledger.
    """
    work = frame.loc[:, ["__decision_ts__", "policy_net_bps", score]].copy()
    work[score] = pd.to_numeric(work[score], errors="coerce")
    work["policy_net_bps"] = pd.to_numeric(work["policy_net_bps"], errors="coerce")
    work = work.loc[np.isfinite(work[score]) & np.isfinite(work["policy_net_bps"])].copy()
    work["__row__"] = np.arange(len(work), dtype=np.int64)
    ordered = work.sort_values(["__decision_ts__", score], ascending=[True, False], kind="stable")
    ordered["ordinal"] = ordered.groupby("__decision_ts__", sort=False).cumcount().to_numpy(np.int64)
    ordered["count"] = ordered.groupby("__decision_ts__", sort=False)[score].transform("size").to_numpy(np.int64)
    ordered["month"] = ordered["__decision_ts__"].dt.strftime("%Y-%m")
    rows: list[dict[str, object]] = []
    for fraction in (0.02, 0.05, 0.10):
        chosen = ordered.loc[ordered.ordinal.lt(np.maximum(1, np.ceil(ordered["count"] * fraction).astype(np.int64)))].copy()
        timestamp = chosen.groupby("__decision_ts__", sort=True)["policy_net_bps"].agg(
            timestamp_mean_net_bps="mean",
            timestamp_hit_rate_positive=lambda values: float((values > 0.0).mean()),
            timestamp_precision_gt50=lambda values: float((values > 50.0).mean()),
        ).reset_index()
        timestamp["month"] = timestamp["__decision_ts__"].dt.strftime("%Y-%m")
        groups: list[tuple[str, pd.DataFrame]] = [("all", timestamp)]
        groups.extend((str(month), part) for month, part in timestamp.groupby("month", sort=True))
        for period, metric in groups:
            rows.append({
                "scope": "mc1", "label": label,
                "period_kind": "global" if period == "all" else "month", "period": period,
                "fraction": fraction, "timestamps": int(len(metric)),
                "selected_rows": int(len(chosen) if period == "all" else int(chosen.month.eq(period).sum())),
                "timestamp_mean_net_bps": float(metric.timestamp_mean_net_bps.mean()),
                "timestamp_hit_rate_positive": float(metric.timestamp_hit_rate_positive.mean()),
                "timestamp_precision_gt50": float(metric.timestamp_precision_gt50.mean()),
                "timestamp_worst_net_bps": float(metric.timestamp_mean_net_bps.min()),
            })
    return pd.DataFrame(rows)


def _portfolio_candidates(frame: pd.DataFrame, floor: float) -> pd.DataFrame:
    import run_strict_r3_enhanced_base_live_stack_challenger as parent
    old=parent.MC1_THRESHOLD_BPS
    try:
        parent.MC1_THRESHOLD_BPS=float(floor)
        result=parent._portfolio_input(frame, "bcf_mc1_expected_bps")
    finally:
        parent.MC1_THRESHOLD_BPS=old
    return result


def _portfolio_metrics(decisions: pd.DataFrame, equity: pd.DataFrame) -> dict[str, float | int]:
    accepted=decisions.loc[decisions.accepted.fillna(False).astype(bool)].copy()
    net=pd.to_numeric(accepted.position_net_return,errors="coerce")*10_000.0
    ts=pd.to_datetime(accepted.timestamp,utc=True,errors="coerce")
    monthly=net.groupby(ts.dt.strftime("%Y-%m"),sort=True).mean()
    weekly=net.groupby(ts.dt.strftime("%G-W%V"),sort=True).mean()
    wallet=pd.to_numeric(equity.wallet,errors="coerce").dropna() if "wallet" in equity else pd.Series(dtype=float)
    drawdown=float((wallet/wallet.cummax()-1).min()) if len(wallet) else float('nan')
    risk = _risk_metrics(equity)
    return {"accepted_rows":int(len(accepted)),"net_ev_bps_per_trade":float(net.mean()) if len(net) else float('nan'),"net_total_bps":float(net.sum()),"trades_per_calendar_day":float(len(accepted)/max((ts.max().normalize()-ts.min().normalize()).days+1,1)) if len(ts) else 0.0,"worst_month_bps":float(monthly.min()) if len(monthly) else float('nan'),"worst_week_bps":float(weekly.min()) if len(weekly) else float('nan'),"max_drawdown":drawdown,**risk}


def _risk_metrics(equity: pd.DataFrame) -> dict[str, float | int]:
    """Return explicitly named equity-curve risk measures for one replay.

    Weekly wallet returns are measured from the first to the final equity
    snapshot inside each UTC Monday-start week.  They are not trade-weighted.
    """
    if equity.empty or "wallet" not in equity:
        return {"sortino_weekly_annualized": float("nan"), "sortino_defined": False, "negative_weeks": 0, "weekly_q05_wallet_return_pct": float("nan"), "weekly_mad_wallet_return_pct": float("nan"), "weekly_std_wallet_return_pct": float("nan"), "weeks": 0, "weeks_below_mean_minus_0_5std": 0, "weeks_below_mean_minus_1_0std": 0, "weeks_below_mean_minus_1_5std": 0, "weeks_below_mean_minus_2_0std": 0}
    eq = equity.loc[:, ["timestamp", "wallet"]].copy()
    eq["timestamp"] = pd.to_datetime(eq["timestamp"], utc=True, errors="coerce")
    eq["wallet"] = pd.to_numeric(eq["wallet"], errors="coerce")
    eq = eq.dropna().sort_values("timestamp", kind="stable")
    if len(eq) < 2:
        return {"sortino_weekly_annualized": float("nan"), "sortino_defined": False, "negative_weeks": 0, "weekly_q05_wallet_return_pct": float("nan"), "weekly_mad_wallet_return_pct": float("nan"), "weekly_std_wallet_return_pct": float("nan"), "weeks": 0, "weeks_below_mean_minus_0_5std": 0, "weeks_below_mean_minus_1_0std": 0, "weeks_below_mean_minus_1_5std": 0, "weeks_below_mean_minus_2_0std": 0}
    eq["week"] = eq.timestamp.dt.normalize() - pd.to_timedelta(eq.timestamp.dt.dayofweek, unit="D")
    weekly = eq.groupby("week", sort=True).wallet.agg(["first", "last"])
    ret = (weekly["last"] / weekly["first"] - 1.0).replace([np.inf, -np.inf], np.nan).dropna() * 100.0
    if not len(ret):
        return {"sortino_weekly_annualized": float("nan"), "sortino_defined": False, "negative_weeks": 0, "weekly_q05_wallet_return_pct": float("nan"), "weekly_mad_wallet_return_pct": float("nan"), "weekly_std_wallet_return_pct": float("nan"), "weeks": 0, "weeks_below_mean_minus_0_5std": 0, "weeks_below_mean_minus_1_0std": 0, "weeks_below_mean_minus_1_5std": 0, "weeks_below_mean_minus_2_0std": 0}
    mean, std = float(ret.mean()), float(ret.std(ddof=0))
    downside = np.minimum(ret.to_numpy(float), 0.0)
    downside_dev = float(np.sqrt(np.mean(np.square(downside))))
    negative_weeks = int(ret.lt(0.0).sum())
    # A zero downside denominator does not demonstrate infinite risk-adjusted
    # performance; it means this finite replay has no negative weekly
    # observation under the stated accounting.  Report it as undefined.
    sortino = float(mean / downside_dev * np.sqrt(52.0)) if downside_dev > 1e-12 else float("nan")
    result: dict[str, float | int] = {"sortino_weekly_annualized": sortino, "sortino_defined": bool(downside_dev > 1e-12), "negative_weeks": negative_weeks, "weekly_q05_wallet_return_pct": float(ret.quantile(.05)), "weekly_mad_wallet_return_pct": float(np.median(np.abs(ret - ret.median()))), "weekly_std_wallet_return_pct": std, "weeks": int(len(ret))}
    for multiplier in (.5, 1.0, 1.5, 2.0):
        result[f"weeks_below_mean_minus_{multiplier:.1f}std"] = int(ret.lt(mean - multiplier * std).sum())
    return result


def _monthly_drawdown(equity: pd.DataFrame) -> dict[str, float]:
    if equity.empty or "wallet" not in equity:
        return {}
    eq = equity.loc[:, ["timestamp", "wallet"]].copy()
    eq["timestamp"] = pd.to_datetime(eq["timestamp"], utc=True, errors="coerce")
    eq["wallet"] = pd.to_numeric(eq["wallet"], errors="coerce")
    eq = eq.dropna().sort_values("timestamp", kind="stable")
    eq["month"] = eq.timestamp.dt.strftime("%Y-%m")
    return {str(month): float((part.wallet / part.wallet.cummax() - 1.0).min()) for month, part in eq.groupby("month", sort=True)}


def _portfolio_sweep(frame: pd.DataFrame, *, label: str, floors: Sequence[float], out: Path) -> tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame]:
    from extreme_price_movements.portfolio_policy_replay import replay_candidates
    from scripts.report_strict_r3_mc1_d2_controlled_portfolio import CAUSAL_AUCTION_CURVE, _params
    rows=[]; month_rows=[]; daily_rows=[]
    for floor in floors:
        candidates=_portfolio_candidates(frame,float(floor))
        for capacity in (1,2,3,4):
            params=replace(_params(),max_new_entries_per_bar=capacity,max_new_entries_per_strategy_per_bar=capacity,portfolio_policy_version=f"p8u_layer_audit_cap{capacity}")
            decisions,equity,_=replay_candidates(candidates,params,mode="global_auction",ev_curve=CAUSAL_AUCTION_CURVE,market_mode="perps",initial_wallet=1000.0)
            decisions.to_parquet(out/f"{label}_floor{int(floor)}_cap{capacity}_decisions.parquet",index=False,compression="zstd")
            equity.to_parquet(out/f"{label}_floor{int(floor)}_cap{capacity}_equity.parquet",index=False,compression="zstd")
            rows.append({"label":label,"floor_bps":float(floor),"max_new_entries_per_timestamp":capacity,**_portfolio_metrics(decisions,equity)})
            accepted=decisions.loc[decisions.accepted.fillna(False).astype(bool)].copy()
            if accepted.empty: continue
            accepted["timestamp"]=pd.to_datetime(accepted.timestamp,utc=True,errors="raise")
            accepted["net_bps"]=pd.to_numeric(accepted.position_net_return,errors="coerce")*10_000.0
            accepted["month"]=accepted.timestamp.dt.strftime("%Y-%m");accepted["day"]=accepted.timestamp.dt.normalize()
            month_dd = _monthly_drawdown(equity)
            for month,part in accepted.groupby("month",sort=True):
                month_rows.append({"label":label,"floor_bps":float(floor),"max_new_entries_per_timestamp":capacity,"month":month,"trades":int(len(part)),"trades_per_day":float(len(part)/max(part.day.nunique(),1)),"net_ev_bps_per_trade":float(part.net_bps.mean()),"net_total_bps":float(part.net_bps.sum()),"max_drawdown":month_dd.get(str(month), float("nan"))})
            for day,part in accepted.groupby("day",sort=True):
                daily_rows.append({"label":label,"floor_bps":float(floor),"max_new_entries_per_timestamp":capacity,"day":day,"trades":int(len(part)),"net_ev_bps_per_trade":float(part.net_bps.mean()),"net_total_bps":float(part.net_bps.sum())})
    return pd.DataFrame(rows),pd.DataFrame(month_rows),pd.DataFrame(daily_rows)


def _markdown(router: pd.DataFrame, base: pd.DataFrame, meta: pd.DataFrame, mc1_tail: pd.DataFrame, admission: pd.DataFrame, portfolio: pd.DataFrame) -> str:
    def table(frame: pd.DataFrame, columns: Sequence[str], rows: int=20) -> list[str]:
        view=frame.loc[:,[c for c in columns if c in frame]].head(rows).copy()
        names=list(view.columns)
        result=["| " + " | ".join(names) + " |", "| " + " | ".join(["---"]*len(names)) + " |"]
        for row in view.itertuples(index=False, name=None):
            rendered=[]
            for value in row:
                if isinstance(value,(float,np.floating)):
                    rendered.append("" if not np.isfinite(value) else f"{float(value):.4f}")
                else:
                    rendered.append(str(value))
            result.append("| " + " | ".join(rendered) + " |")
        return result
    lines=["# P8u Layer Comparison Audit", "", "All score metrics use exact timestamp-local top fractions and the canonical frozen rich 15-minute policy net labels (100 bps fixed round-trip cost once).  This is offline research only.", "", "## Router top-50 recall", *table(router.loc[router.period_kind.eq('global')], ["label","threshold_bps","top50_recall","selected_hits","oracle_hits"],30), "", "## Matched Base diagnostics", *table(base.loc[base.period_kind.eq('global')], ["label","fraction","timestamp_mean_net_bps","timestamp_hit_rate_positive","timestamp_precision_gt50","timestamps"],30), "", "## Matched Meta diagnostics", *table(meta.loc[meta.period_kind.eq('global')], ["label","fraction","timestamp_mean_net_bps","timestamp_hit_rate_positive","timestamp_precision_gt50","timestamps"],40), "", "## MC1 timestamp-local conversion", *table(mc1_tail.loc[mc1_tail.period_kind.eq('global')], ["label","fraction","timestamp_mean_net_bps","timestamp_hit_rate_positive","timestamp_precision_gt50","timestamps"],30), "", "## MC1 admission", *table(admission.loc[admission.period_kind.eq('global')], ["label","floor_bps","admitted_rows","net_ev_bps_per_trade","hit_rate_positive","precision_gt50"],30), "", "## Constrained portfolio", *table(portfolio, ["label","floor_bps","max_new_entries_per_timestamp","accepted_rows","net_ev_bps_per_trade","net_total_bps","trades_per_calendar_day","worst_month_bps","worst_week_bps","max_drawdown"],50), ""]
    return "\n".join(lines)


def run(args: argparse.Namespace) -> Path:
    out=args.out.resolve()
    if out.exists(): raise FileExistsError(out)
    out.mkdir(parents=True)
    policy=_policy(args.policy.resolve())
    new_router=_join_policy(_load_router(args.new_router.resolve()),policy)
    old_router=_join_policy(_load_router(args.old_router.resolve()),policy)
    router_common=new_router.merge(old_router.loc[:,[*IDENTITY,"router_primary_rank"]],on=list(IDENTITY),suffixes=("_new","_old"),validate="one_to_one")
    router_new,_=_router_recall(router_common.rename(columns={"router_primary_rank_new":"rank"}),"rank",label="current_p8u_successor")
    router_old,_=_router_recall(router_common.rename(columns={"router_primary_rank_old":"rank"}),"rank",label="previous_p8u_router")
    router=pd.concat([router_old,router_new],ignore_index=True)
    base=_load_base(args.new_base.resolve());meta=_load_meta(args.new_meta.resolve(),args.new_meta_arm)
    base_full = _join_policy(base, policy)
    new=_current_layer_panel(base,meta,policy)
    old=_legacy_previous(args.previous_dual.resolve(),policy)
    # Strictly exact identity intersection gives all score heads the same
    # candidate/outcome population, avoiding coverage-as-performance.
    common=new.merge(old.loc[:,[*IDENTITY,"legacy_base_rank","legacy_meta_rank"]],on=list(IDENTITY),how="inner",validate="one_to_one")
    common=common.loc[common.__decision_ts__.ge(pd.Timestamp(args.match_start,tz="UTC")) & common.__decision_ts__.lt(pd.Timestamp(args.match_end,tz="UTC"))].copy()
    base_metrics=pd.concat([
        _tail_metrics(base_full,"base_rank_ts",label="current_f72_base_full_oof",scope="base_full_oof"),
        _tail_metrics(common,"base_rank_ts",label="current_f72_base_matched",scope="base_matched"),
        _tail_metrics(common,"legacy_base_rank",label="previous_routed_threeway_base_matched",scope="base_matched"),
    ],ignore_index=True)
    meta_metrics=pd.concat([
        _tail_metrics(new,"current_score_rank",label="current_f72_underf120_current_full_oof",scope="meta_full_oof"),
        _tail_metrics(common,"current_score_rank",label="current_f72_underf120_current_matched",scope="meta_matched"),
        _tail_metrics(common,"base_rank_ts",label="current_f72_base_coordinate_matched",scope="meta_matched"),
        _tail_metrics(common,"legacy_meta_rank",label="previous_t6t9_current_matched",scope="meta_matched"),
        _tail_metrics(common,"legacy_base_rank",label="previous_threeway_bcf_matched",scope="meta_matched"),
    ],ignore_index=True)
    cmi=pd.concat([
        _cmi_binned(new,base="base_rank_ts",meta="meta_rank_ts",label="current_under_given_f72_full_oof"),
        _cmi_binned(common,base="base_rank_ts",meta="meta_rank_ts",label="current_under_given_f72_matched"),
        _cmi_binned(common,base="legacy_base_rank",meta="legacy_meta_rank",label="previous_t6t9_given_threeway_matched"),
    ],ignore_index=True)
    oracle=pd.concat([_oracle(new,candidate_scope="current_router50_base_meta_common"),_oracle(common,candidate_scope="matched_current_previous_layer_population")],ignore_index=True)
    current_mc1=pd.read_parquet(args.current_mc1.resolve())
    current_mc1["__decision_ts__"]=pd.to_datetime(current_mc1["__decision_ts__"],utc=True,errors="raise")
    legacy_current=pd.read_parquet(args.legacy_current_mc1.resolve())
    legacy_bcf=pd.read_parquet(args.legacy_bcf_mc1.resolve())
    legacy_current["__decision_ts__"]=pd.to_datetime(legacy_current["__decision_ts__"],utc=True,errors="raise");legacy_bcf["__decision_ts__"]=pd.to_datetime(legacy_bcf["__decision_ts__"],utc=True,errors="raise")
    legacy=legacy_current.loc[:,["candidate_id","__decision_ts__","mc1_expected_bps","final_score"]].merge(legacy_bcf.loc[:,["candidate_id","__decision_ts__","mc1_expected_bps","final_score"]],on=["candidate_id","__decision_ts__"],suffixes=("_current","_bcf"),how="inner",validate="one_to_one")
    legacy=legacy.rename(columns={"mc1_expected_bps_current":"current_mc1_expected_bps","mc1_expected_bps_bcf":"bcf_mc1_expected_bps","final_score_current":"current_final_score","final_score_bcf":"bcf_final_score"})
    legacy=legacy.merge(policy,on="candidate_id",how="left",validate="one_to_one")
    legacy["enhanced_base_routed"]=True;legacy["side_name"]="long";legacy["__symbol__"]=legacy.candidate_id.str.split("|",n=1,expand=True)[0]
    start=pd.Timestamp(args.mc1_start,tz="UTC");end=pd.Timestamp(args.mc1_end,tz="UTC")
    current_mc1=current_mc1.loc[current_mc1.__decision_ts__.ge(start)&current_mc1.__decision_ts__.lt(end)].copy();legacy=legacy.loc[legacy.__decision_ts__.ge(start)&legacy.__decision_ts__.lt(end)].copy()
    mc1_tail = pd.concat([
        _mc1_tail_metrics(current_mc1, "bcf_mc1_expected_bps", label="current_f72_underf120_bcf_mc1"),
        _mc1_tail_metrics(current_mc1, "current_mc1_expected_bps", label="current_f72_underf120_current_mc1"),
        _mc1_tail_metrics(legacy, "bcf_mc1_expected_bps", label="legacy_live_bcf_mc1"),
        _mc1_tail_metrics(legacy, "current_mc1_expected_bps", label="legacy_live_current_mc1"),
    ], ignore_index=True)
    admission=pd.concat([_admission_metrics(current_mc1,label="current_f72_underf120",floors=MC1_FLOORS),_admission_metrics(legacy,label="legacy_live",floors=MC1_FLOORS)],ignore_index=True)
    portfolio,portfolio_monthly,portfolio_daily=_portfolio_sweep(current_mc1,label="current_f72_underf120",floors=MC1_FLOORS,out=out)
    legacy_portfolio,legacy_monthly,legacy_daily=_portfolio_sweep(legacy,label="legacy_live",floors=MC1_FLOORS,out=out)
    portfolio=pd.concat([portfolio,legacy_portfolio],ignore_index=True)
    portfolio_monthly=pd.concat([portfolio_monthly,legacy_monthly],ignore_index=True)
    portfolio_daily=pd.concat([portfolio_daily,legacy_daily],ignore_index=True)
    coverage = pd.DataFrame([
        {"layer": "router", "first_timestamp": new_router.__decision_ts__.min(), "last_timestamp": new_router.__decision_ts__.max(), "rows": int(len(new_router)), "note": "P8U successor target-free strict-OOF router scores"},
        {"layer": "base", "first_timestamp": base_full.__decision_ts__.min(), "last_timestamp": base_full.__decision_ts__.max(), "rows": int(len(base_full)), "note": "F72 target-free strict-OOF Base scores"},
        {"layer": "meta", "first_timestamp": new.__decision_ts__.min(), "last_timestamp": new.__decision_ts__.max(), "rows": int(len(new)), "note": "F72+Under target-free strict-OOF scores"},
        {"layer": "mc1", "first_timestamp": current_mc1.__decision_ts__.min(), "last_timestamp": current_mc1.__decision_ts__.max(), "rows": int(len(current_mc1)), "note": "separate strict-prequential Current/BCF MC1 maps after resolved-label warm-up"},
    ])
    for name,frame in {"coverage":coverage,"router_recall":router,"base_tail_metrics":base_metrics,"meta_tail_metrics":meta_metrics,"conditional_mi":cmi,"oracle_opportunity":oracle,"mc1_tail_metrics":mc1_tail,"mc1_admission":admission,"portfolio_capacity":portfolio,"portfolio_monthly":portfolio_monthly,"portfolio_daily":portfolio_daily}.items(): frame.to_parquet(out/f"{name}.parquet",index=False,compression="zstd")
    (out/"LAYER_COMPARISON_RECEIPT.md").write_text(_markdown(router,base_metrics,meta_metrics,mc1_tail,admission,portfolio),encoding="utf-8")
    _once(out/"run_manifest.json",{"schema":"strict_r3_p8u_layer_comparison_v1","scope":"offline matched score/replay audit only; no fitting, live state, exchange IO or mutation","policy":str(args.policy.resolve()),"policy_sha256":_sha256(args.policy.resolve()),"new": {"router":str(args.new_router.resolve()),"base":str(args.new_base.resolve()),"meta":str(args.new_meta.resolve()),"mc1":str(args.current_mc1.resolve())},"previous":{"router":str(args.old_router.resolve()),"routed_dual":str(args.previous_dual.resolve()),"legacy_current_mc1":str(args.legacy_current_mc1.resolve()),"legacy_bcf_mc1":str(args.legacy_bcf_mc1.resolve())},"matched_layer_period":[args.match_start,args.match_end],"mc1_period":[args.mc1_start,args.mc1_end],"policy_contract":"frozen rich 15-minute SimplePolicyOptimiser successor including smooth capital protection; 100-bps round-trip cost embedded once","portfolio":"global chronological 7x / 10% margin slots / eight concurrent, capacity sweep 1..4 entries per timestamp","august_coverage":"not present in current F72/Under target-free OOF score receipts; intentionally not substituted with unrelated live-model scores"})
    return out


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--new-router",type=Path,required=True);parser.add_argument("--old-router",type=Path,required=True)
    parser.add_argument("--new-base",type=Path,required=True);parser.add_argument("--new-meta",type=Path,required=True);parser.add_argument("--new-meta-arm",default="xendcg_selected_under_bps100")
    parser.add_argument("--previous-dual",type=Path,required=True);parser.add_argument("--current-mc1",type=Path,required=True)
    parser.add_argument("--legacy-current-mc1",type=Path,required=True);parser.add_argument("--legacy-bcf-mc1",type=Path,required=True);parser.add_argument("--policy",type=Path,required=True);parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--match-start",default="2026-04-01");parser.add_argument("--match-end",default="2026-08-01");parser.add_argument("--mc1-start",default="2025-11-01");parser.add_argument("--mc1-end",default="2026-08-01")
    args=parser.parse_args();print(run(args))


if __name__=="__main__": main()
