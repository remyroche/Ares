#!/usr/bin/env python3
"""Round-B meta target screen around a daily-prequential B2 expected-net base.

This runner is intentionally P0/all-candidates only.  It first materialises a
candidate's B2 expected value from event-payoff means known strictly before
that candidate's day, freezes the last map at the OOS boundary, then trains
one shared per-row meta target.  Later Round-C population selection and
combination rules consume these predictions; they are not hidden here.
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_full_universe_residual_meta import select  # noqa: E402

PROBS=("p_upper","p_lower","p_timeout")

def _base_predictions(root:Path, geometry:str)->pd.DataFrame:
    parts=[]
    for side in ("long","short"):
        path=root/side/"target_screen_predictions.parquet"
        if not path.exists():path=root/f"t2_{geometry}_{side}"/"target_screen_predictions.parquet"
        parts.append(pd.read_parquet(path,columns=["candidate_id","score_bps",*PROBS]))
    return pd.concat(parts,ignore_index=True)

def _payoff_vector(history:pd.DataFrame, side:str)->np.ndarray:
    """Causal conditional *gross* payoffs; fixed entry cost is applied later."""
    global_mean=history.groupby("event",observed=True).gross_bps.mean().reindex(range(3))
    if global_mean.isna().any():raise RuntimeError("prequential payoff history lacks an event state")
    local=history[history.side_name.eq(side)].groupby("event",observed=True).gross_bps.agg(["mean","count"]).reindex(range(3))
    # Fixed conservative shrinkage prevents a thin side/event cell from
    # creating an arbitrary common-unit shift.
    count=local["count"].fillna(0.).to_numpy(float); mean=local["mean"].fillna(global_mean).to_numpy(float)
    return (count*mean+2000.*global_mean.to_numpy(float))/(count+2000.)

def _attach_prequential_expected(data:pd.DataFrame, train_start:pd.Timestamp, oos_start:pd.Timestamp)->pd.DataFrame:
    """Use only resolved prior rows for every train day; freeze at OOS start."""
    result=[]
    resolved=data[data.__label_available_at__.lt(oos_start)].sort_values("__label_available_at__")
    resolved_time=resolved.__label_available_at__.to_numpy(dtype="datetime64[ns]")
    event=resolved.event.to_numpy(int); value=resolved.gross_bps.to_numpy(float); side=resolved.side_name.to_numpy(str)
    global_count=np.zeros(3);global_sum=np.zeros(3)
    side_count={name:np.zeros(3) for name in ("long","short")};side_sum={name:np.zeros(3) for name in ("long","short")}
    pointer=0
    for day, frame in data.groupby(data.__ts__.dt.floor("D"), sort=True):
        if day < train_start: continue
        cutoff=min(day,oos_start)
        end=int(np.searchsorted(resolved_time,cutoff.to_datetime64(),side="left"))
        if end>pointer:
            new_event=event[pointer:end];new_value=value[pointer:end];new_side=side[pointer:end]
            global_count+=np.bincount(new_event,minlength=3);global_sum+=np.bincount(new_event,weights=new_value,minlength=3)
            for name in ("long","short"):
                take=new_side==name;side_count[name]+=np.bincount(new_event[take],minlength=3);side_sum[name]+=np.bincount(new_event[take],weights=new_value[take],minlength=3)
            pointer=end
        if (global_count==0).any(): continue
        global_mean=global_sum/global_count
        maps={}
        for name in ("long","short"):
            maps[name]=(side_sum[name]+2000.*global_mean)/(side_count[name]+2000.)
        z=frame.copy();pay=np.empty((len(z),3));long=z.side_name.eq("long").to_numpy();pay[long]=maps["long"];pay[~long]=maps["short"];p=z.loc[:,PROBS].to_numpy(float)
        z["base_expected_gross_bps"]=np.einsum("ij,ij->i",p,pay)
        z["base_expected_net_bps"]=z.base_expected_gross_bps-100.
        z["base_cost_margin_bps"]=z.base_expected_net_bps
        z["base_payoff_mixture_sd_bps"]=np.sqrt(np.maximum((p*(pay-z.base_expected_gross_bps.to_numpy()[:,None])**2).sum(1),0.))
        z["base_payoff_map_cutoff"]=cutoff
        result.append(z)
    out=pd.concat(result,ignore_index=True)
    if out.candidate_id.duplicated().any():raise RuntimeError("prequential map duplicated candidate")
    return out

def _state_features(frame:pd.DataFrame, cols:list[str])->np.ndarray:
    p=frame.loc[:,PROBS].to_numpy(float);clipped=np.clip(p,1e-12,1.)
    ordered=np.sort(p,axis=1)
    extra=np.column_stack([
        frame.base_expected_net_bps.to_numpy(float),frame.base_expected_gross_bps.to_numpy(float),frame.base_cost_margin_bps.to_numpy(float),frame.base_payoff_mixture_sd_bps.to_numpy(float),
        -(clipped*np.log(clipped)).sum(1)/np.log(3.),(p*p).sum(1),p[:,0]-p[:,1],p[:,0]-np.maximum(p[:,1],p[:,2]),ordered[:,2]-ordered[:,1],
        frame.side_name.eq("long").to_numpy(float),
    ])
    context=frame[cols].replace([np.inf,-np.inf],np.nan).fillna(0.).to_numpy("float32")
    return np.column_stack([context,p,extra])

def _attach_prequential_population(frame:pd.DataFrame, population:float, train_start:pd.Timestamp, oos_start:pd.Timestamp)->pd.DataFrame:
    """Fold-local pooled high-base flag from prior-resolved base expectations."""
    if population <= 0:return frame.assign(high_base_eligible=True,high_base_cutoff=np.nan)
    out=[]
    for day,z in frame.groupby(frame.__ts__.dt.floor("D"),sort=True):
        if day < train_start:continue
        cutoff=min(day,oos_start);history=frame[frame.__label_available_at__.lt(cutoff)]
        if len(history)<20_000:continue
        threshold=float(history.base_expected_net_bps.quantile(1-population))
        y=z.copy();y["high_base_cutoff"]=threshold;y["high_base_eligible"]=y.base_expected_net_bps.ge(threshold);out.append(y)
    return pd.concat(out,ignore_index=True)

def _model(classification:bool):
    common=dict(n_estimators=180,learning_rate=.05,num_leaves=24,min_child_samples=400,colsample_bytree=.8,subsample=.8,reg_lambda=10.,random_state=20260803,n_jobs=1,verbosity=-1)
    return lgb.LGBMClassifier(objective="binary",**common) if classification else lgb.LGBMRegressor(objective="huber",alpha=.9,**common)

def main()->None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--panel",type=Path,required=True);p.add_argument("--audit",type=Path,required=True);p.add_argument("--base-root",type=Path,required=True);p.add_argument("--out",type=Path,required=True)
    p.add_argument("--geometry",default="tp3_sl2");p.add_argument("--train-start",default="2024-05-01");p.add_argument("--oos-start",default="2024-08-01");p.add_argument("--oos-end",default="2024-12-01");p.add_argument("--population",type=float,default=0.,choices=(0.,.5,.3,.2),help='0=all, otherwise fold-local top fraction')
    p.add_argument("--side-local",action="store_true",help="fit one meta function and feature subset per side; scores remain common units")
    p.add_argument("--target",required=True,choices=("cost_clear","base_correct","overestimate50","overestimate100","upside_surprise50","failure","failure_severity","residual"));a=p.parse_args()
    train_start=pd.Timestamp(a.train_start,tz="UTC");oos_start=pd.Timestamp(a.oos_start,tz="UTC");oos_end=pd.Timestamp(a.oos_end,tz="UTC")
    net=f"t4_{a.geometry}_net_bps";gross=f"t4_{a.geometry}_gross_bps";event=f"t2_{a.geometry}_event"
    meta=json.loads(a.audit.read_text())["meta"]["coverage_ge_90pct"]
    cols=["candidate_id","__ts__","__label_available_at__","side_name",net,gross,event]+meta
    raw=pd.concat([pd.read_parquet(x,columns=cols) for x in sorted((a.panel/"parts").glob("*.parquet"))],ignore_index=True)
    raw.__ts__=pd.to_datetime(raw.__ts__,utc=True);raw.__label_available_at__=pd.to_datetime(raw.__label_available_at__,utc=True)
    raw=raw.rename(columns={event:"event",net:"net_bps",gross:"gross_bps"})
    net="net_bps";gross="gross_bps"
    data=raw.merge(_base_predictions(a.base_root,a.geometry),on="candidate_id",validate="one_to_one")
    # Need an April resolved history to seed May's daily maps.  OOS maps freeze
    # at August 1; all training labels are resolved before that boundary.
    mapped=_attach_prequential_expected(data,pd.Timestamp("2024-04-15",tz="UTC"),oos_start)
    mapped=_attach_prequential_population(mapped,a.population,train_start,oos_start)
    train=mapped[(mapped.__ts__.ge(train_start)) & (mapped.__ts__.lt(oos_start)) & (mapped.__label_available_at__.lt(oos_start)) & mapped.high_base_eligible].copy()
    ev=mapped[(mapped.__ts__.ge(oos_start)) & (mapped.__ts__.lt(oos_end))].copy()
    ev_eligible=ev[ev.high_base_eligible].copy()
    if a.target=="cost_clear":y=train[net].gt(0).to_numpy(int);classification=True;definition="I(realised net > 0)"
    elif a.target=="base_correct":
        # Policy correctness: does the realised trade clear costs iff the
        # causally known base expectation says it should?  This is deliberately
        # distinct from a raw profitable-trade classifier.
        y=((train[net].gt(0)) == (train.base_expected_net_bps.gt(0))).to_numpy(int);classification=True;definition="I(sign(realised net) == sign(prequential expected net))"
    elif a.target.startswith("overestimate"):
        margin=float(a.target.replace("overestimate",""));y=train[net].lt(train.base_expected_net_bps-margin).to_numpy(int);classification=True;definition=f"I(realised net < prequential expected net - {margin:g} bps)"
    elif a.target=="upside_surprise50":y=train[net].gt(train.base_expected_net_bps+50.).to_numpy(int);classification=True;definition="I(realised net > prequential expected net + 50 bps)"
    elif a.target=="failure":y=train[net].le(0).to_numpy(int);classification=True;definition="I(realised net <= 0)"
    elif a.target=="failure_severity":
        train=train[train[net].le(0)].copy();y=np.log1p(np.maximum(-train[net].to_numpy(float),0.));classification=False;definition="log1p(-net), conditional on net <= 0"
    else:y=train[net].to_numpy(float)-train.base_expected_net_bps.to_numpy(float);classification=False;definition="realised net minus prequential B2 expected net"
    # The shared selector requires at least 30 entries; 30 remains within the
    # attached roadmap's 20--30 tail cap.
    feature_count=36 if a.population in (0.,.5) else 30
    if not a.side_local:
        chosen=select(train,meta,y,n=feature_count);xtr=_state_features(train,chosen);xev=_state_features(ev_eligible,chosen);m=_model(classification).fit(xtr,y)
        raw_score=m.predict_proba(xev)[:,1] if classification else m.predict(xev)
        chosen_manifest=chosen
    else:
        raw_score=np.full(len(ev_eligible),np.nan);chosen_manifest={}
        for side in ("long","short"):
            take_train=train.side_name.eq(side).to_numpy();take_ev=ev_eligible.side_name.eq(side).to_numpy()
            side_train=train.loc[take_train];side_y=y[take_train]
            chosen=select(side_train,meta,side_y,n=feature_count);chosen_manifest[side]=chosen
            model=_model(classification).fit(_state_features(side_train,chosen),side_y)
            raw_score[take_ev]=model.predict_proba(_state_features(ev_eligible.loc[take_ev],chosen))[:,1] if classification else model.predict(_state_features(ev_eligible.loc[take_ev],chosen))
        if np.isnan(raw_score).any():raise RuntimeError("side-local meta did not score all eligible rows")
    if classification:
        if a.target=="cost_clear":actual=ev_eligible[net].gt(0).to_numpy(int)
        elif a.target=="base_correct":actual=((ev_eligible[net].gt(0)) == (ev_eligible.base_expected_net_bps.gt(0))).to_numpy(int)
        elif a.target.startswith("overestimate"):
            margin=float(a.target.replace("overestimate",""));actual=ev_eligible[net].lt(ev_eligible.base_expected_net_bps-margin).to_numpy(int)
        elif a.target=="upside_surprise50":actual=ev_eligible[net].gt(ev_eligible.base_expected_net_bps+50.).to_numpy(int)
        else:actual=ev_eligible[net].le(0).to_numpy(int)
        diagnostics={"oos_auc":float(roc_auc_score(actual,raw_score)),"oos_brier":float(brier_score_loss(actual,raw_score)),"oos_prevalence":float(actual.mean())}
        final=raw_score
    else:
        if a.target=="residual":final=ev_eligible.base_expected_net_bps.to_numpy(float)+raw_score+float(np.mean(y-m.predict(xtr)))
        else:final=np.expm1(raw_score)
        diagnostics={"oos_prediction_std_bps":float(np.std(final))}
    out=ev[["candidate_id","__ts__","side_name",gross,net,"base_expected_net_bps","base_expected_gross_bps","base_payoff_mixture_sd_bps","high_base_eligible","high_base_cutoff"]].copy();out["meta_score"]=np.nan;out.loc[ev_eligible.index,"meta_score"]=raw_score;out["final_score"]=out.base_expected_net_bps;out.loc[ev_eligible.index,"final_score"]=final;out=out.sort_values(["final_score","candidate_id"],ascending=[False,True]);rows=[]
    for q in (.01,.05,.1,.2):
        z=out.head(int(len(out)*q+.999));rows.append({"top_fraction":q,"n":len(z),"gross_bps":float(z[gross].mean()),"net_bps":float(z[net].mean()),"long_n":int(z.side_name.eq("long").sum()),"short_n":int(z.side_name.eq("short").sum())})
    a.out.mkdir(parents=True,exist_ok=True);out.to_parquet(a.out/"predictions.parquet",index=False);pd.DataFrame(rows).to_parquet(a.out/"metrics.parquet",index=False)
    manifest={"schema":"full_universe_round_b_meta_target_v1","target":a.target,"target_definition":definition,"population_fraction":a.population,"side_local":a.side_local,"base_representation":"daily-prequential side-shrunk event-payoff expected net; frozen at OOS start","base_inputs":[*PROBS,"base_expected_net_bps","base_expected_gross_bps","base_cost_margin_bps","base_payoff_mixture_sd_bps"],"meta_features":chosen_manifest,"train_window":[str(train_start),str(oos_start)],"eval_window":[str(oos_start),str(oos_end)],"eligible_train_rows":len(train),"eligible_oos_rows":len(ev_eligible),"diagnostics":diagnostics,"metrics":rows}
    (a.out/"manifest.json").write_text(json.dumps(manifest,indent=2));print(json.dumps({"target":a.target,"diagnostics":diagnostics,"top10":rows[2]}))
if __name__=="__main__":main()
