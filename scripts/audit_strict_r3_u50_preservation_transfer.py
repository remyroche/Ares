#!/usr/bin/env python3
"""Causal diagnostic for U50 E/T control versus preservation-Base challenger.

Scores are first sealed without policy outcomes.  The report then joins the
canonical rich-policy ledger only for post-hoc evaluation.  It has no fitting,
threshold, inference, exchange, or live-trading authority.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

MONTHS = ("2026-04", "2026-05", "2026-06", "2026-07")
ROUTE = .50


def _exclusive(path: Path, payload: object) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True, default=str)


def _top_fraction(frame: pd.DataFrame, field: str, fraction: float = ROUTE) -> pd.Series:
    work = frame.loc[:, ["candidate_id", "__decision_ts__", field]].copy()
    work["__ix__"] = work.index
    work["__score__"] = pd.to_numeric(work[field], errors="coerce").fillna(-np.inf)
    work = work.sort_values(["__decision_ts__", "__score__", "candidate_id"], ascending=[True, False, True], kind="stable")
    ordinal = work.groupby("__decision_ts__", sort=False).cumcount() + 1
    count = work.groupby("__decision_ts__", sort=False)["candidate_id"].transform("size")
    work["flag"] = np.isfinite(work["__score__"]) & ordinal.le(np.ceil(fraction * count).astype(int))
    return work.set_index("__ix__")["flag"]


def _top_pct(frame: pd.DataFrame, field: str) -> pd.Series:
    work = frame.loc[:, ["candidate_id", "__decision_ts__", field]].copy()
    work["__ix__"] = work.index
    work["__score__"] = pd.to_numeric(work[field], errors="coerce").fillna(-np.inf)
    work = work.sort_values(["__decision_ts__", "__score__", "candidate_id"], ascending=[True, False, True], kind="stable")
    ordinal = work.groupby("__decision_ts__", sort=False).cumcount().to_numpy(float)
    count = work.groupby("__decision_ts__", sort=False)["candidate_id"].transform("size").to_numpy(float)
    work["top_pct"] = 100.0 * ordinal / np.maximum(count, 1.0)
    return work.set_index("__ix__")["top_pct"]


def _utility(net: pd.Series, valid: pd.Series) -> np.ndarray:
    values = pd.to_numeric(net, errors="coerce").fillna(0.0).to_numpy(float)
    return np.sqrt(np.minimum(np.maximum(values - 50.0, 0.0), 300.0) / 300.0) * valid.to_numpy(bool)


def _router(root: Path, month: str, prefix: str) -> pd.DataFrame:
    path = root / f"target_free_scores/month={month}.parquet"
    out = pd.read_parquet(path, columns=["candidate_id", "__decision_ts__", "router_primary_rank"])
    out["__decision_ts__"] = pd.to_datetime(out["__decision_ts__"], utc=True)
    out[f"{prefix}_route"] = _top_fraction(out, "router_primary_rank").reindex(out.index).to_numpy(bool)
    return out.rename(columns={"router_primary_rank": f"{prefix}_router_rank"})


def _scores(root: Path, month: str, prefix: str) -> pd.DataFrame:
    base = pd.read_parquet(
        root / f"target_free_monthly/month={month}/scores_features.parquet",
        columns=["candidate_id", "__decision_ts__", "base_bps", "efficiency_bps", "timing_bps", "enhanced_base_bps"],
    ).rename(columns={c: f"{prefix}_{c}" for c in ("base_bps", "efficiency_bps", "timing_bps", "enhanced_base_bps")})
    base["__decision_ts__"] = pd.to_datetime(base["__decision_ts__"], utc=True)
    parts = [base]
    for family in ("current", "bcf"):
        score = pd.read_parquet(
            root / f"target_free_scores/{family}/month={month}.parquet",
            columns=["candidate_id", "__decision_ts__", "final_score"],
        ).rename(columns={"final_score": f"{prefix}_{family}_final_score"})
        score["__decision_ts__"] = pd.to_datetime(score["__decision_ts__"], utc=True)
        parts.append(score)
    out = parts[0]
    for part in parts[1:]:
        out = out.merge(part, on=["candidate_id", "__decision_ts__"], how="outer", validate="one_to_one")
    return out


def _read_mc1(root: Path, prefix: str) -> pd.DataFrame:
    out = pd.read_parquet(root / "dual_mc1_predictions.parquet", columns=["candidate_id", "__decision_ts__", "current_mc1_expected_bps", "bcf_mc1_expected_bps"])
    out["__decision_ts__"] = pd.to_datetime(out["__decision_ts__"], utc=True)
    out = out.loc[out["__decision_ts__"].dt.strftime("%Y-%m").isin(MONTHS)].copy()
    return out.rename(columns={"current_mc1_expected_bps": f"{prefix}_current_mc1_expected_bps", "bcf_mc1_expected_bps": f"{prefix}_bcf_mc1_expected_bps"})


def _rank_columns(trace: pd.DataFrame, arm: str) -> None:
    route = trace["u50_route"].fillna(False).astype(bool)
    for field, name in ((f"{arm}_enhanced_base_bps", "base"), (f"{arm}_base_bps", "preservation"), (f"{arm}_current_final_score", "current_meta"), (f"{arm}_bcf_final_score", "bcf_meta")):
        trace[f"{arm}_{name}_top_pct"] = _top_pct(trace.loc[route], field).reindex(trace.index)


def _rank_movement(frame: pd.DataFrame) -> pd.DataFrame:
    valid = frame["policy_path_valid"].fillna(False).astype(bool) & np.isfinite(frame["policy_net_bps"])
    rows = []
    for cohort in ("u50_only", "common"):
        for threshold in (50.0, 100.0, 200.0):
            part = frame.loc[valid & frame["cohort"].eq(cohort) & frame["policy_net_bps"].gt(threshold)]
            for arm in ("control", "challenger"):
                rank = part[f"{arm}_base_top_pct"]
                rows.append({
                    "cohort": cohort, "winner_threshold_bps": threshold, "arm": arm, "rows": int(len(part)),
                    "median_base_top_pct": float(rank.median()), "q25_base_top_pct": float(rank.quantile(.25)), "q10_base_top_pct": float(rank.quantile(.10)),
                    "fraction_base_top30": float(rank.lt(30).mean()), "fraction_base_top20": float(rank.lt(20).mean()),
                    "fraction_base_top10": float(rank.lt(10).mean()), "fraction_base_top5": float(rank.lt(5).mean()),
                })
    return pd.DataFrame(rows)


def _utility_transfer(frame: pd.DataFrame) -> pd.DataFrame:
    valid = frame["policy_path_valid"].fillna(False).astype(bool) & np.isfinite(frame["policy_net_bps"])
    only = frame["cohort"].eq("u50_only")
    util = _utility(frame["policy_net_bps"], valid)
    denominator = float(util[only].sum())
    rows = []
    for arm in ("control", "challenger"):
        for cutoff in (10.0, 20.0, 30.0):
            mask = only & frame[f"{arm}_base_top_pct"].lt(cutoff)
            rows.append({"arm": arm, "cutoff": cutoff, "rows": int(mask.sum()), "utility": float(util[mask].sum()), "transfer_rate": float(util[mask].sum()/denominator) if denominator else np.nan, "ev_bps": float(frame.loc[mask & valid, "policy_net_bps"].mean())})
    return pd.DataFrame(rows)


def _conditional_preservation(frame: pd.DataFrame) -> pd.DataFrame:
    valid = frame["policy_path_valid"].fillna(False).astype(bool) & np.isfinite(frame["policy_net_bps"])
    routed = frame["u50_route"].fillna(False).astype(bool) & valid
    rows = []
    for lo, hi in ((0,10),(10,20),(20,30),(30,50),(50,100)):
        part = frame.loc[routed & frame["control_base_top_pct"].ge(lo) & frame["control_base_top_pct"].lt(hi)].copy()
        p = pd.to_numeric(part["challenger_base_bps"], errors="coerce")
        part["p_high"] = p.rank(pct=True, method="first").ge(.5)
        ic = p.rank().corr(pd.to_numeric(part["policy_net_bps"], errors="coerce").rank(), method="pearson")
        util = _utility(part["policy_net_bps"], pd.Series(True, index=part.index))
        for label, sub in (("low_p", part.loc[~part["p_high"]]), ("high_p", part.loc[part["p_high"]])):
            index = sub.index
            rows.append({"et_top_pct_band": f"{lo}-{hi}", "p_group": label, "rows": int(len(sub)), "within_band_spearman_ic": float(ic), "ev_bps": float(sub["policy_net_bps"].mean()), "utility": float(util[np.isin(part.index, index)].mean())})
    return pd.DataFrame(rows)


def _progression(frame: pd.DataFrame) -> pd.DataFrame:
    valid = frame["policy_path_valid"].fillna(False).astype(bool) & np.isfinite(frame["policy_net_bps"])
    rows = []
    for arm in ("control", "challenger"):
        for threshold in (50.0,100.0,200.0):
            base = valid & frame["cohort"].eq("u50_only") & frame["policy_net_bps"].gt(threshold)
            for family in ("current","bcf"):
                for stage, mask in (("base_top30", base & frame[f"{arm}_base_top_pct"].lt(30)), ("meta_top30", base & frame[f"{arm}_{family}_meta_top_pct"].lt(30)), ("meta_top20", base & frame[f"{arm}_{family}_meta_top_pct"].lt(20)), ("meta_top10", base & frame[f"{arm}_{family}_meta_top_pct"].lt(10))):
                    util = _utility(frame.loc[mask, "policy_net_bps"], pd.Series(True, index=frame.index[mask]))
                    rows.append({"arm":arm,"family":family,"winner_threshold_bps":threshold,"stage":stage,"rows":int(mask.sum()),"ev_bps":float(frame.loc[mask,"policy_net_bps"].mean()),"utility":float(util.sum())})
    return pd.DataFrame(rows)


def _mc1(frame: pd.DataFrame) -> pd.DataFrame:
    valid = frame["policy_path_valid"].fillna(False).astype(bool) & np.isfinite(frame["policy_net_bps"])
    rows=[]
    for arm in ("control","challenger"):
        cur=pd.to_numeric(frame[f"{arm}_current_mc1_expected_bps"],errors="coerce");bcf=pd.to_numeric(frame[f"{arm}_bcf_mc1_expected_bps"],errors="coerce")
        state=np.select([cur.ge(50)&bcf.ge(50),cur.ge(50)&bcf.lt(50),cur.lt(50)&bcf.ge(50)],["both_pass","current_pass_bcf_fail","bcf_pass_current_fail"],default="both_fail_or_missing")
        for threshold in (50.0,100.0,200.0):
            work=frame.loc[valid & frame["cohort"].eq("u50_only") & frame["policy_net_bps"].gt(threshold)].copy();work["state"]=state[work.index]
            for name,part in work.groupby("state",sort=True):
                rows.append({"arm":arm,"winner_threshold_bps":threshold,"state":name,"rows":int(len(part)),"ev_bps":float(part["policy_net_bps"].mean()),"utility":float(_utility(part["policy_net_bps"],pd.Series(True,index=part.index)).sum())})
    return pd.DataFrame(rows)


def _accepted_ids(root: Path) -> set[str]:
    path=root/"routed_base_dual_50_2026_marjul_decisions.parquet"
    d=pd.read_parquet(path,columns=["timestamp","symbol","accepted"])
    d["timestamp"]=pd.to_datetime(d["timestamp"],utc=True)
    d=d.loc[d["accepted"].fillna(False).astype(bool)]
    return set(d["symbol"].astype(str)+"|long|"+(d["timestamp"]-pd.Timedelta(hours=1)).dt.strftime("%Y-%m-%dT%H:%M:%SZ"))


def _portfolio_substitution(frame: pd.DataFrame, control: Path, challenger: Path) -> pd.DataFrame:
    c=_accepted_ids(control);p=_accepted_ids(challenger)
    valid=frame["policy_path_valid"].fillna(False).astype(bool)&np.isfinite(frame["policy_net_bps"])
    state=pd.Series(np.select([frame.candidate_id.isin(c)&frame.candidate_id.isin(p),frame.candidate_id.isin(c),frame.candidate_id.isin(p)],["common","control_only","challenger_only"],default="neither"),index=frame.index)
    rows=[]
    work=frame.loc[valid].copy();work["state"]=state.loc[work.index]
    for name,part in work.groupby("state",sort=True):
        if name=="neither": continue
        rows.append({"portfolio_cohort":name,"rows":int(len(part)),"ev_bps":float(part.policy_net_bps.mean()),"total_bps":float(part.policy_net_bps.sum()),"utility":float(_utility(part.policy_net_bps,pd.Series(True,index=part.index)).sum())})
    return pd.DataFrame(rows)


def _md(table: pd.DataFrame) -> str:
    t=table.copy()
    for c in t:
        if c.endswith("bps") or c in {"utility","transfer_rate","within_band_spearman_ic"}:
            t[c]=t[c].map(lambda x:"—" if not np.isfinite(x) else f"{x:+.3f}")
    hdr=list(t.columns);lines=["| "+" | ".join(hdr)+" |","| "+" | ".join(["---"]*len(hdr))+" |"]
    for r in t.itertuples(index=False,name=None): lines.append("| "+" | ".join(str(x) for x in r)+" |")
    return "\n".join(lines)


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--p8u-router",type=Path,required=True);p.add_argument("--u50-router",type=Path,required=True)
    p.add_argument("--control",type=Path,required=True);p.add_argument("--challenger",type=Path,required=True)
    p.add_argument("--policy",type=Path,required=True);p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    if a.out.exists(): raise FileExistsError(a.out)
    a.out.mkdir(parents=True)
    parts=[]
    for month in MONTHS:
        p8=_router(a.p8u_router,month,"p8u");u=_router(a.u50_router,month,"u50")
        if set(p8.candidate_id)!=set(u.candidate_id): raise AssertionError(f"{month}: router identities differ")
        x=p8.merge(u,on=["candidate_id","__decision_ts__"],validate="one_to_one")
        x=x.merge(_scores(a.control,month,"control"),on=["candidate_id","__decision_ts__"],how="left",validate="one_to_one")
        x=x.merge(_scores(a.challenger,month,"challenger"),on=["candidate_id","__decision_ts__"],how="left",validate="one_to_one")
        parts.append(x)
    trace=pd.concat(parts,ignore_index=True)
    trace["cohort"]=np.select([trace.p8u_route&trace.u50_route,trace.p8u_route,trace.u50_route],["common","p8u_only","u50_only"],default="neither")
    trace=trace.merge(_read_mc1(a.control,"control"),on=["candidate_id","__decision_ts__"],how="left",validate="one_to_one")
    trace=trace.merge(_read_mc1(a.challenger,"challenger"),on=["candidate_id","__decision_ts__"],how="left",validate="one_to_one")
    if any(any(k in c.lower() for k in ("policy_","outcome","label","net_bps","gross_bps")) for c in trace): raise AssertionError("target-free trace contains outcome field")
    for arm in ("control","challenger"): _rank_columns(trace,arm)
    trace.to_parquet(a.out/"target_free_trace.parquet",index=False,compression="zstd")
    labels=pd.read_parquet(a.policy,columns=["candidate_id","policy_path_valid","policy_net_bps","policy_label_available_ts"])
    joined=trace.merge(labels,on="candidate_id",how="left",validate="one_to_one")
    joined.to_parquet(a.out/"outcome_joined_trace.parquet",index=False,compression="zstd")
    tables={"base_rank_movement":_rank_movement(joined),"base_utility_transfer":_utility_transfer(joined),"conditional_preservation":_conditional_preservation(joined),"meta_progression":_progression(joined),"mc1_attribution":_mc1(joined),"portfolio_substitution":_portfolio_substitution(joined,a.control,a.challenger)}
    for name,t in tables.items(): t.to_parquet(a.out/f"{name}.parquet",index=False,compression="zstd")
    report="# U50 Preservation Base Transfer Audit\n\nOffline diagnostic: all score fields were sealed target-free before policy outcomes were joined.\n\n"
    for name,t in tables.items(): report+=f"## {name.replace('_',' ').title()}\n\n"+_md(t)+"\n\n"
    (a.out/"PRESERVATION_TRANSFER_REPORT.md").write_text(report)
    _exclusive(a.out/"run_manifest.json",{"scope":"offline diagnosis only","months":MONTHS,"route":"frozen U50/P8u exact Top50","mc1":"fixed dual Current/BCF >=50","control":str(a.control),"challenger":str(a.challenger),"trace_then_policy_join":True})

if __name__=="__main__": main()
