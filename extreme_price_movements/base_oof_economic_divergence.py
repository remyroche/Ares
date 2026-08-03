"""Identical-row diagnosis of native-base-label versus execution-EV divergence."""
from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd

IDENTITY=("candidate_id","side_name","__ts__")
BASE_REQUIRED={*IDENTITY,"base_oof_score","__first_touch_target_soft__","__first_touch_capture_net__","__decision_ts__","base_label_resolution_utc"}
EXEC_REQUIRED={*IDENTITY,"execution_gross_ev_12h","execution_cost_return","execution_net_ev_12h","execution_exit_reason","execution_exit_hour","execution_mfe_return_12h","execution_mae_return_12h","execution_expected_spread_bps","execution_label_end_utc"}
POP_REQUIRED={*IDENTITY,"transition_window_member","expost_transition_active"}

def _rho(frame: pd.DataFrame, left: str, right: str) -> float:
    v=frame[[left,right]].apply(pd.to_numeric,errors="coerce").dropna()
    return float(v.corr(method="spearman").iloc[0,1]) if len(v)>=2 else float("nan")

def _within_rank_ic(frame: pd.DataFrame, left: str, right: str, group: str) -> float:
    v=frame[[left,right,group]].copy()
    v[left]=pd.to_numeric(v[left],errors="coerce")
    v[right]=pd.to_numeric(v[right],errors="coerce")
    v=v.dropna()
    if v.empty:return float("nan")
    # Rank-transform within the contemporaneous candidate set or symbol; this
    # removes group-level location shifts without using future outcomes.
    a=v.groupby(group,observed=True)[left].rank(pct=True)
    b=v.groupby(group,observed=True)[right].rank(pct=True)
    return float(a.corr(b)) if len(v)>=2 else float("nan")

def _top_mask(frame: pd.DataFrame) -> pd.Series:
    # The flagged economics are a pooled monthly/side top decile, not a
    # timestamp quota. Candidate-id tie breaking makes membership reproducible.
    ordered=frame.sort_values(["month","side_name","base_oof_score","candidate_id"],ascending=[True,True,False,True],kind="stable").copy()
    group=ordered.groupby(["month","side_name"],sort=False,observed=True)
    ordered["__rank__"]=group.cumcount()+1;ordered["__n__"]=group.candidate_id.transform("size")
    return ordered.set_index("candidate_id")["__rank__"].le(np.ceil(ordered.set_index("candidate_id")["__n__"]*.10)).reindex(frame.candidate_id).fillna(False).astype(bool).set_axis(frame.index)

def join_identical_rows(base: pd.DataFrame, execution: pd.DataFrame, population: pd.DataFrame) -> pd.DataFrame:
    for frame,required,name in ((base,BASE_REQUIRED,"base OOF"),(execution,EXEC_REQUIRED,"execution labels"),(population,POP_REQUIRED,"population")):
        missing=sorted(required.difference(frame.columns))
        if missing:raise ValueError(f"{name} lacks {missing}")
        if frame.candidate_id.duplicated().any():raise ValueError(f"{name} has duplicate candidate IDs")
    work=base.copy();work.__ts__=pd.to_datetime(work.__ts__,utc=True,errors="raise")
    execution=execution.copy();execution.__ts__=pd.to_datetime(execution.__ts__,utc=True,errors="raise")
    population=population.copy();population.__ts__=pd.to_datetime(population.__ts__,utc=True,errors="raise")
    echeck=execution[["candidate_id","side_name","__ts__"]].rename(columns={"side_name":"__execution_side","__ts__":"__execution_ts"})
    evals=execution.drop(columns=["side_name","__ts__","__symbol__"],errors="ignore")
    # The accepted base OOF already carries net EV as an economic gate.  Check
    # it agrees with the exact-label ledger instead of accepting pandas' x/y
    # suffixes as two ambiguous targets.
    if "execution_net_ev_12h" in work.columns:
        exact_net=evals.loc[:,["candidate_id","execution_net_ev_12h"]].rename(columns={"execution_net_ev_12h":"__exact_execution_net__"})
        work=work.merge(exact_net,on="candidate_id",how="left",validate="one_to_one")
        if not np.allclose(pd.to_numeric(work.execution_net_ev_12h),pd.to_numeric(work.__exact_execution_net__),rtol=0.0,atol=1e-9,equal_nan=False):
            raise ValueError("base OOF and exact execution net EV disagree")
        work=work.drop(columns="__exact_execution_net__")
        evals=evals.drop(columns="execution_net_ev_12h")
    work=work.merge(evals,on="candidate_id",how="left",validate="one_to_one").merge(echeck,on="candidate_id",how="left",validate="one_to_one")
    if not (work.side_name.eq(work.pop("__execution_side"))&work.__ts__.eq(work.pop("__execution_ts"))).all():raise ValueError("execution rows do not exactly match base OOF identities")
    pcheck=population[["candidate_id","side_name","__ts__"]].rename(columns={"side_name":"__population_side","__ts__":"__population_ts"})
    pvals=population[["candidate_id","transition_window_member","expost_transition_active"]]
    work=work.merge(pvals,on="candidate_id",how="left",validate="one_to_one").merge(pcheck,on="candidate_id",how="left",validate="one_to_one")
    if not (work.side_name.eq(work.pop("__population_side"))&work.__ts__.eq(work.pop("__population_ts"))).all():raise ValueError("population rows do not exactly match base OOF identities")
    if work[list(EXEC_REQUIRED-{"candidate_id","side_name","__ts__"})].isna().any().any():raise ValueError("an identical base OOF row lacks exact execution labels")
    work["month"]=work.__ts__.dt.strftime("%Y-%m");work["side_name"]=work.side_name.astype(str).str.lower()
    work["transition_window_member"]=work.transition_window_member.fillna(False).astype(bool);work["expost_transition_active"]=work.expost_transition_active.fillna(0).astype(bool)
    work["pooled_month_side_top_decile"]=_top_mask(work)
    return work

def _metrics(frame: pd.DataFrame) -> dict[str,Any]:
    top=frame.loc[frame.pooled_month_side_top_decile]
    result={"rows":int(len(frame)),"hours":int(frame.__ts__.nunique()),"symbols":int(frame.__symbol__.nunique()),
            "execution_gross_ev_12h_mean":float(frame.execution_gross_ev_12h.mean()),"execution_cost_return_mean":float(frame.execution_cost_return.mean()),"execution_net_ev_12h_mean":float(frame.execution_net_ev_12h.mean()),
            "top_decile_rows":int(len(top)),"top_decile_symbols":int(top.__symbol__.nunique()),"top_decile_execution_gross_ev_12h_mean":float(top.execution_gross_ev_12h.mean()) if len(top) else float("nan"),"top_decile_execution_cost_return_mean":float(top.execution_cost_return.mean()) if len(top) else float("nan"),"top_decile_execution_net_ev_12h_mean":float(top.execution_net_ev_12h.mean()) if len(top) else float("nan"),
            "top_decile_native_soft_mean":float(top.__first_touch_target_soft__.mean()) if len(top) else float("nan"),"top_decile_native_capture_mean":float(top.__first_touch_capture_net__.mean()) if len(top) else float("nan"),
            "top_decile_native_positive_execution_net_nonpositive_fraction":float(((top.__first_touch_target_soft__>0)&(top.execution_net_ev_12h<=0)).mean()) if len(top) else float("nan"),
            "top_decile_execution_net_negative_fraction":float((top.execution_net_ev_12h<=0).mean()) if len(top) else float("nan")}
    for target,name in (("__first_touch_target_soft__","native_soft"),("__first_touch_capture_net__","native_capture")):
        result[f"score_to_{name}_ic_pooled"]=_rho(frame,"base_oof_score",target)
        result[f"score_to_{name}_ic_timestamp_local"]=_within_rank_ic(frame,"base_oof_score",target,"__ts__")
        result[f"score_to_{name}_ic_symbol_neutral"]=_within_rank_ic(frame,"base_oof_score",target,"__symbol__")
        for exec_col,short in (("execution_gross_ev_12h","gross"),("execution_cost_return","cost"),("execution_net_ev_12h","net")):
            result[f"{name}_to_{short}_ic"]=_rho(frame,target,exec_col)
    for exec_col,short in (("execution_gross_ev_12h","gross"),("execution_cost_return","cost"),("execution_net_ev_12h","net")):
        result[f"score_to_{short}_ic"]=_rho(frame,"base_oof_score",exec_col)
    return result

def _turnover(frame: pd.DataFrame) -> pd.DataFrame:
    out=[]
    for (month,side),g in frame.loc[frame.pooled_month_side_top_decile].groupby(["month","side_name"],sort=True,observed=True):
        sets=[set(x.__symbol__.astype(str)) for _,x in g.groupby("__ts__",sort=True,observed=True)]
        overlaps=[len(a&b)/len(a|b) for a,b in zip(sets,sets[1:]) if a|b]
        out.append({"month":month,"side_name":side,"top_decile_hours":len(sets),"mean_top_decile_symbols_per_hour":float(np.mean([len(x) for x in sets])) if sets else float("nan"),"adjacent_hour_symbol_jaccard":float(np.mean(overlaps)) if overlaps else float("nan"),"adjacent_hour_symbol_turnover":float(1-np.mean(overlaps)) if overlaps else float("nan")})
    return pd.DataFrame(out)

def build_divergence_diagnostic(base: pd.DataFrame, execution: pd.DataFrame, population: pd.DataFrame) -> tuple[dict[str,pd.DataFrame],dict[str,Any]]:
    work=join_identical_rows(base,execution,population)
    groups=[]
    for (month,side),local in work.groupby(["month","side_name"],sort=True,observed=True):groups.append({"month":month,"side_name":side,**_metrics(local)})
    metric=pd.DataFrame(groups)
    deciles=[]
    for (month,side),g in work.groupby(["month","side_name"],sort=True,observed=True):
        rank=g.base_oof_score.rank(method="first",pct=True,ascending=True);d=g.assign(score_decile=np.minimum(10,np.ceil(rank*10).astype(int)))
        for decile,local in d.groupby("score_decile",sort=True):deciles.append({"month":month,"side_name":side,"score_decile":int(decile),**_metrics(local)})
    slice_rows=[]
    for (month,side),g in work.groupby(["month","side_name"],sort=True,observed=True):
        for name,mask in (("all",pd.Series(True,index=g.index)),("transition_window",g.transition_window_member),("active_transition",g.expost_transition_active),("non_transition",~g.transition_window_member)):
            slice_rows.append({"month":month,"side_name":side,"slice":name,**_metrics(g.loc[mask])})
    assets=work.loc[work.pooled_month_side_top_decile].groupby(["month","side_name","__symbol__"],sort=True,observed=True).agg(rows=("candidate_id","size"),native_soft_mean=("__first_touch_target_soft__","mean"),gross_ev_mean=("execution_gross_ev_12h","mean"),cost_mean=("execution_cost_return","mean"),net_ev_mean=("execution_net_ev_12h","mean"),expected_spread_bps_mean=("execution_expected_spread_bps","mean")).reset_index().sort_values(["month","side_name","rows"],ascending=[True,True,False],kind="stable")
    exits=work.loc[work.pooled_month_side_top_decile].groupby(["month","side_name","execution_exit_reason"],sort=True,observed=True).agg(rows=("candidate_id","size"),net_ev_mean=("execution_net_ev_12h","mean"),mfe_mean=("execution_mfe_return_12h","mean"),mae_mean=("execution_mae_return_12h","mean"),exit_hour_mean=("execution_exit_hour","mean"),cost_mean=("execution_cost_return","mean")).reset_index()
    horizons=pd.DataFrame([{ "native_label_horizon_hours":float((pd.to_datetime(work.base_label_resolution_utc,utc=True)-pd.to_datetime(work.__decision_ts__,utc=True)).dt.total_seconds().median()/3600),"execution_label_horizon_hours":float((pd.to_datetime(work.execution_label_end_utc,utc=True)-pd.to_datetime(work.__decision_ts__,utc=True)).dt.total_seconds().median()/3600),"rows":int(len(work))}])
    summary={"schema":"febapr2025_base_oof_native_execution_divergence_v1","research_only":True,"scope":"identical accepted base-OOF identities; diagnostic only","rows":int(len(work)),"row_join_exact":True,"horizon_contract":{"native_target":"first-touch soft target resolved 24h after decision","execution_target":"exact current-spread exit policy net EV resolved 12h after decision"},"headline":metric.loc[:,["month","side_name","score_to_native_soft_ic_pooled","top_decile_execution_net_ev_12h_mean","top_decile_execution_gross_ev_12h_mean","top_decile_execution_cost_return_mean","top_decile_native_positive_execution_net_nonpositive_fraction"]].to_dict(orient="records")}
    return {"month_side_metrics":metric,"score_decile_monotonicity":pd.DataFrame(deciles),"transition_slice_metrics":pd.DataFrame(slice_rows),"top_decile_asset_composition":assets,"top_decile_exit_path_breakdown":exits,"top_decile_turnover":_turnover(work),"horizon_audit":horizons},summary
