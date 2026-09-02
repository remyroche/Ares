#!/usr/bin/env python3
"""Evaluate fixed and deterministic recovery-aware strict-R3 EV-map arms."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.replay_strict_r3_forward_portfolio import (  # noqa: E402
    CAUSAL_AUCTION_CURVE, _auction_candidates, _run,
)
from scripts.run_strict_r3_recovery_detection_funnel import FIXED_ARMS  # noqa: E402


FAMILIES = {
    "level": ["fast_minus_slow", "fast_ev", "slow_ev"],
    "trend": ["fast_minus_slow", "fast_ev", "slow_ev", "slope_14d", "slope_acceleration_7_21"],
    "persistence": ["fast_minus_slow", "slope_14d", "slope_acceleration_7_21", "positive_fraction_7d", "consecutive_positive_days"],
    "downside": ["fast_minus_slow", "slope_14d", "slope_acceleration_7_21", "positive_fraction_7d", "catastrophic_fraction_7d", "negative_semideviation_7d", "cvar_7_minus_28"],
    "breadth": ["fast_minus_slow", "slope_14d", "slope_acceleration_7_21", "positive_fraction_7d", "catastrophic_fraction_7d", "negative_semideviation_7d", "positive_cell_fraction_7d", "positive_symbol_fraction_7d"],
    "performance": ["fast_minus_slow", "slope_14d", "slope_acceleration_7_21", "positive_fraction_7d", "catastrophic_fraction_7d", "negative_semideviation_7d", "positive_cell_fraction_7d", "positive_symbol_fraction_7d", "model_ic_3d", "model_ic_7d", "model_ic_14d", "model_ic_28d", "model_hit_surprise_3d", "model_hit_surprise_7d", "model_hit_surprise_14d", "model_hit_surprise_28d", "model_ic_3_minus_14", "model_hit_3_minus_14", "model_surprise_accel_3_14"],
    "entropy": ["fast_minus_slow", "slope_14d", "slope_acceleration_7_21", "positive_fraction_7d", "catastrophic_fraction_7d", "negative_semideviation_7d", "positive_cell_fraction_7d", "positive_symbol_fraction_7d", "model_ic_7d", "model_hit_surprise_7d", "state_entropy_7d", "entropy_delta_7_28"],
}


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _causal_zscores(state: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    output = pd.DataFrame(index=state.index)
    for column in columns:
        values = pd.to_numeric(state[column], errors="coerce")
        expanding = values.expanding(min_periods=28)
        center = expanding.median().shift(1)
        mad = values.expanding(min_periods=28).apply(
            lambda block: float(np.median(np.abs(block - np.median(block)))), raw=True,
        ).shift(1)
        output[column] = ((values - center) / (1.4826 * mad.clip(lower=1e-6))).clip(-5, 5)
    return output


def _authority(state: pd.DataFrame) -> pd.DataFrame:
    output = state.loc[:, ["snapshot_utc"]].copy()
    adverse = {"catastrophic_fraction_7d", "negative_semideviation_7d"}
    for family, columns in FAMILIES.items():
        z = _causal_zscores(state, columns)
        signs = np.array([-1.0 if column in adverse else 1.0 for column in columns])
        score = (z.to_numpy(float) * signs).mean(axis=1)
        output[f"recovery_score_{family}"] = score
        output[f"lambda_{family}_continuous"] = 1.0 / (1.0 + np.exp(-score))
        output[f"lambda_{family}_bounded50"] = 0.5 * output[f"lambda_{family}_continuous"]
    full = output["lambda_performance_continuous"].fillna(0.0).to_numpy(float)
    hysteresis = np.zeros(len(full), dtype=float)
    for index, target in enumerate(full):
        previous = hysteresis[index - 1] if index else 0.0
        hysteresis[index] = target if target <= previous else min(target, previous + 0.20)
    output["lambda_performance_hysteresis"] = hysteresis
    # Causal four-state controller.  Every input below is computed from trades
    # whose fixed H12 availability timestamp is no later than the snapshot.
    # Recent model performance is deliberately part of both shock detection
    # and recovery confirmation rather than being used as a hindsight label.
    states: list[str] = []
    state_name = "NORMAL"
    evidence_run = 0
    authority_by_state = {
        "DEFENSIVE": 0.0,
        "PROBE": 0.25,
        "RECOVERING": 0.75,
        "NORMAL": 0.50,
    }
    for row in state.itertuples(index=False):
        shock = (
            np.isfinite(row.fast_ev)
            and np.isfinite(row.catastrophic_fraction_7d)
            and np.isfinite(row.catastrophic_fraction_28d)
            and np.isfinite(row.model_hit_surprise_7d)
            and row.fast_ev < 0.0
            and row.catastrophic_fraction_7d > row.catastrophic_fraction_28d
            and row.model_hit_surprise_7d < 0.0
        )
        evidence = sum((
            bool(np.isfinite(row.fast_minus_slow) and row.fast_minus_slow > 0.0),
            bool(np.isfinite(row.slope_14d) and row.slope_14d > 0.0),
            bool(np.isfinite(row.model_ic_7d) and np.isfinite(row.model_ic_28d)
                 and row.model_ic_7d > row.model_ic_28d),
            bool(np.isfinite(row.model_hit_surprise_7d)
                 and np.isfinite(row.model_hit_surprise_28d)
                 and row.model_hit_surprise_7d > row.model_hit_surprise_28d),
        )) >= 3
        evidence_run = evidence_run + 1 if evidence else 0
        if shock:
            state_name = "DEFENSIVE"
            evidence_run = 0
        elif state_name == "DEFENSIVE" and evidence_run >= 2:
            state_name = "PROBE"
        elif state_name == "PROBE" and evidence_run >= 3 and row.positive_fraction_7d >= 0.50:
            state_name = "RECOVERING"
        elif (
            state_name == "RECOVERING" and evidence_run >= 3
            and row.medium_ev > 25.0 and row.model_hit_surprise_7d >= 0.0
        ):
            state_name = "NORMAL"
        states.append(state_name)
    output["recovery_state_machine"] = states
    output["lambda_state_machine"] = [authority_by_state[value] for value in states]
    return output


def _arms(state: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    authority = _authority(state)
    output = state.loc[:, ["snapshot_utc"]].merge(authority, on="snapshot_utc", validate="one_to_one")
    mapping = {name: name for name in FIXED_ARMS}
    mapping.update({
        "A11_recovery_level_continuous": "recovery_level_continuous",
        "A12_recovery_trend_continuous": "recovery_trend_continuous",
        "A13_recovery_persistence_continuous": "recovery_persistence_continuous",
        "A14_recovery_downside_continuous": "recovery_downside_continuous",
        "A15_recovery_breadth_continuous": "recovery_breadth_continuous",
        "A16_recovery_performance_continuous": "recovery_performance_continuous",
        "A17_recovery_entropy_continuous": "recovery_entropy_continuous",
        "A18_recovery_performance_hysteresis": "recovery_performance_hysteresis",
        "A19_recovery_performance_bounded50": "recovery_performance_bounded50",
        "A20_recovery_state_machine": "recovery_state_machine_map",
    })
    return output, mapping


def _cvar(values: pd.Series, fraction: float) -> float:
    values = pd.to_numeric(values, errors="coerce").dropna().sort_values()
    if values.empty:
        return np.nan
    return float(values.iloc[:max(1, int(math.ceil(len(values) * fraction)))].mean())


def _longest_zero(daily: pd.Series) -> int:
    best = current = 0
    for value in daily.astype(bool):
        current = current + 1 if value else 0
        best = max(best, current)
    return best


def _summaries(frame: pd.DataFrame, arm_columns: dict[str, str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary: list[dict[str, Any]] = []
    monthly: list[dict[str, Any]] = []
    daily_out: list[dict[str, Any]] = []
    frame = frame.copy()
    frame["year"] = frame["__decision_ts__"].dt.year
    frame["month"] = frame["__decision_ts__"].dt.strftime("%Y-%m")
    frame["day"] = frame["__decision_ts__"].dt.normalize()
    valid = frame["policy_path_valid"].fillna(False).astype(bool)
    for arm, column in arm_columns.items():
        admitted = np.isfinite(frame[column]) & frame[column].ge(50.0)
        for day, indices in frame.groupby("day", sort=True).groups.items():
            net = pd.to_numeric(frame.loc[indices][admitted.loc[indices] & valid.loc[indices]]["policy_net_bps"], errors="coerce")
            daily_out.append({"arm": arm, "day": day, "trades": len(net), "net_bps_per_trade": net.mean(), "total_net_bps": net.sum()})
        for period_kind, labels in (("year", frame["year"]), ("all", pd.Series("2025_2026", index=frame.index))):
            for period, indices in labels.groupby(labels, sort=True).groups.items():
                net = pd.to_numeric(frame.loc[indices][admitted.loc[indices] & valid.loc[indices]]["policy_net_bps"], errors="coerce")
                d = pd.DataFrame(daily_out)
                d = d.loc[d["arm"].eq(arm) & pd.to_datetime(d["day"]).dt.year.eq(int(period))] if period_kind=="year" else d.loc[d["arm"].eq(arm)]
                summary.append({"arm":arm,"period_kind":period_kind,"period":str(period),"trades":len(net),"net_bps_per_trade":net.mean(),"total_net_bps":net.sum(),"positive_rate":(net>0).mean(),"median_trade_bps":net.median(),"cvar05_bps":_cvar(net,.05),"cvar10_bps":_cvar(net,.10),"active_days":int((d.trades>0).sum()),"zero_trade_days":int((d.trades==0).sum()),"max_consecutive_zero_days":_longest_zero(d.trades.eq(0))})
        for month, indices in frame["month"].groupby(frame["month"], sort=True).groups.items():
            net = pd.to_numeric(frame.loc[indices][admitted.loc[indices] & valid.loc[indices]]["policy_net_bps"], errors="coerce")
            monthly.append({"arm":arm,"month":month,"trades":len(net),"net_bps_per_trade":net.mean(),"total_net_bps":net.sum()})
    return pd.DataFrame(summary), pd.DataFrame(monthly), pd.DataFrame(daily_out)


def _portfolio(frame: pd.DataFrame, arms: list[str]) -> pd.DataFrame:
    rows=[]
    for year in (2025, 2026):
        block=frame.loc[frame["__decision_ts__"].dt.year.eq(year)].copy()
        for arm in arms:
            work=block.copy()
            work["causal_21d_side_expected_net_bps"]=work[arm]
            work["causal_21d_side_admitted_ge_50bps"]=np.isfinite(work[arm]) & work[arm].ge(50)
            work["auction_tie_break_score"]=work["final_score"]
            candidates=_auction_candidates(work,strategy_prefix=f"recovery_{arm}")
            decisions,equity,_,summary=_run(candidates,0.0,f"{year}_{arm}",initial_wallet=1000.0,perp_leverage=7.0,margin_slot_wallet_fraction=.10,ev_curve=CAUSAL_AUCTION_CURVE)
            accepted=decisions.loc[decisions["accepted"].fillna(False)]
            net=pd.to_numeric(accepted.get("position_net_return"),errors="coerce")*10000
            rows.append({"arm":arm,"year":year,"accepted_trades":len(accepted),"net_bps_per_trade":net.mean(),"total_net_bps":net.sum(),"final_wallet":summary.get("final_wallet"),"portfolio_net_pnl":summary.get("portfolio_net_pnl")})
    return pd.DataFrame(rows)


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--funnel-root",type=Path,required=True)
    parser.add_argument("--out-dir",type=Path,required=True)
    parser.add_argument("--skip-portfolio",action="store_true")
    args=parser.parse_args()
    if args.out_dir.exists(): raise FileExistsError(args.out_dir)
    args.out_dir.mkdir(parents=True)
    state=pd.read_parquet(args.funnel_root/"recovery_state_features.parquet")
    state["snapshot_utc"]=pd.to_datetime(state["snapshot_utc"],utc=True)
    authority, arm_columns=_arms(state)
    maps=pd.read_parquet(args.funnel_root/"multiwindow_ev_maps.parquet")
    maps["__decision_ts__"]=pd.to_datetime(maps["__decision_ts__"],utc=True)
    maps["snapshot_utc"]=maps["__decision_ts__"].dt.normalize()
    frame=maps.merge(authority,on="snapshot_utc",how="left",validate="many_to_one")
    fast=.5*(frame["m14"]+frame["m21"])
    slow=frame[["m28","m35","m42"]].median(axis=1)
    for family in FAMILIES:
        lam=frame[f"lambda_{family}_continuous"]
        frame[f"recovery_{family}_continuous"]=(1-lam)*slow+lam*fast
    lam=frame["lambda_performance_hysteresis"]
    frame["recovery_performance_hysteresis"]=(1-lam)*slow+lam*fast
    lam=frame["lambda_performance_bounded50"]
    frame["recovery_performance_bounded50"]=(1-lam)*slow+lam*fast
    lam=frame["lambda_state_machine"]
    frame["recovery_state_machine_map"]=(1-lam)*slow+lam*fast
    summary,monthly,daily=_summaries(frame,arm_columns)
    summary.to_csv(args.out_dir/"ablation_summary.csv",index=False)
    monthly.to_csv(args.out_dir/"monthly_summary.csv",index=False)
    daily.to_parquet(args.out_dir/"daily_summary.parquet",index=False)
    authority.to_parquet(args.out_dir/"recovery_authority.parquet",index=False)
    # Portfolio replay the complete fixed frontier and deterministic challengers.
    portfolio = pd.DataFrame()
    if not args.skip_portfolio:
        portfolio=_portfolio(frame,[
            "A0_robust28", "A2_robust21", "A5_mean5", "A7_max5",
            "U_second_lowest", "U_bottom3_mean",
            "A18_recovery_performance_hysteresis",
            "A16_recovery_performance_continuous",
        ])
    portfolio.to_csv(args.out_dir/"portfolio_summary.csv",index=False)
    (args.out_dir/"run_manifest.json").write_text(json.dumps({"schema":"strict_r3_recovery_deterministic_ablation_v1","status":"complete","selection":"2025 development only; 2026 confirmation","fixed_arms":list(FIXED_ARMS),"recovery_families":FAMILIES,"availability":"resolved labels <= snapshot; unresolved ignored","input_maps_sha256":_sha(args.funnel_root/'multiwindow_ev_maps.parquet'),"input_state_sha256":_sha(args.funnel_root/'recovery_state_features.parquet')},indent=2)+"\n")
    print(json.dumps({"event":"deterministic_ablation_complete","arms":len(arm_columns),"out_dir":str(args.out_dir)}))


if __name__=="__main__": main()
