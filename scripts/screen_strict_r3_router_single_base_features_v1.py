#!/usr/bin/env python3
"""Screen a 1,407-field causal universe for one Router50 single-Base finalist.

The output is a sealed <=120 feature receipt usable by the strict-OOF Base
runner.  It applies coverage/variance hygiene, a .97 Spearman representative
veto, then fold-balanced full-model gain stability.  It is offline only.
"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
import numpy as np
import pandas as pd
from lightgbm import LGBMRanker

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT / 'scripts') not in sys.path: sys.path.insert(0,str(ROOT / 'scripts'))
import run_strict_r3_router_single_base_prescreen_v1 as base

IDENTITY=set(base.IDENTITY)|{'__ts__','__symbol__'}
def once(path:Path,payload:object):
    fd=os.open(path,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o644)
    with os.fdopen(fd,'w') as h: json.dump(payload,h,indent=2,sort_keys=True,default=str)
def utc(text:str): return pd.Timestamp(text,tz='UTC')
def month_end(m): return m+pd.offsets.MonthBegin(1)

def universe(roots):
    names=None
    for root in roots:
        p=next(root.glob('month=*/causal_feature_universe.parquet'))
        cols=pd.read_parquet(p).columns.tolist()
        keep=[x for x in cols if x not in IDENTITY]
        names=keep if names is None else [x for x in names if x in set(keep)]
    return names
def hygiene(fields, roots, months):
    # Full panels are wide enough that materialising all 1,407 columns at
    # once is unnecessary and can exceed replay memory.  The statistics are
    # separable by column; sample only the bounded rows needed for correlation.
    coverage={f:[] for f in fields}; variance={f:[] for f in fields}; samples=[]
    for month in months:
        path=base._feature_path(roots,month)
        month_sample=[]
        for begin in range(0,len(fields),64):
            block=fields[begin:begin+64]
            d=pd.read_parquet(path,columns=block)
            x=d.apply(pd.to_numeric,errors='coerce').replace([np.inf,-np.inf],np.nan)
            for f in block:
                coverage[f].append(float(x[f].notna().mean())); variance[f].append(float(x[f].var(ddof=0)))
            month_sample.append(x.iloc[:min(750,len(x))].astype(np.float32).reset_index(drop=True))
        samples.append(pd.concat(month_sample,axis=1))
    c=pd.concat(samples,ignore_index=True)
    stats=pd.DataFrame({'feature':fields,'coverage':[min(coverage[f]) for f in fields],'variance':[float(np.nanmedian(variance[f])) for f in fields]})
    keep=stats.loc[(stats.coverage>=.90)&np.isfinite(stats.variance)&stats.variance.gt(1e-12),'feature'].tolist()
    # Deterministic .97 representative veto; favour higher coverage then lexical stability.
    # A bounded 2,250-row float32 reference is ample for detecting the near
    # duplicate (.97) pairs while avoiding a wide rank-correlation peak.
    x=c[keep].fillna(c[keep].median()).fillna(0.0).astype(np.float32)
    corr=x.corr(method='spearman').abs(); selected=[]
    ordered=stats.loc[stats.feature.isin(keep)].sort_values(['coverage','variance','feature'],ascending=[False,False,True]).feature.tolist()
    for f in ordered:
        if not selected or float(corr.loc[f,selected].max()) < .97: selected.append(f)
    stats['hygiene_keep']=stats.feature.isin(selected)
    return stats,selected
def fit_fold(args, fields, held):
    reserve=held-pd.Timedelta(days=args.reserve_days); start=reserve-pd.DateOffset(months=args.train_months)
    window,_=base._load_window(candidate_root=None,feature_root=args.feature_roots,label_root=args.label_root,router_root=args.router_root,start=start,end=month_end(held),fields=fields)
    train=window.loc[window.__decision_ts__.lt(reserve)].copy(); heldf=window.loc[window.__decision_ts__.ge(held)].copy()
    spec=base.TARGETS[args.target]
    valid, value = spec.valid_column, spec.value_column
    # The selected target, rather than an unrelated policy sidecar, owns the
    # resolved-label boundary.  For the policy-ordinal target these are the
    # same timestamp; for the normalized targets this avoids silently
    # tightening support while retaining the identical causal rule.
    available = pd.to_datetime(train[spec.available_column], utc=True, errors='coerce')
    train=train.loc[train[valid].fillna(False)&available.lt(reserve)&np.isfinite(pd.to_numeric(train[value],errors='coerce'))]
    train=base._sample_complete_queries(train,args.train_cap).sort_values(['__decision_ts__','candidate_id'])
    y,_=base._target_labels(train,heldf,spec)
    x,med=base._numeric_matrix(train,fields); z,_=base._numeric_matrix(heldf,fields,med)
    m=LGBMRanker(objective=args.objective,metric='ndcg',n_estimators=130,learning_rate=.05,max_depth=4,num_leaves=15,min_child_samples=260,subsample=.8,subsample_freq=1,colsample_bytree=.8,reg_alpha=.05,reg_lambda=8.,min_split_gain=.001,lambdarank_truncation_level=args.truncation,label_gain=base.GAIN_SCHEDULES[args.gain],lambdarank_norm=True,random_state=1729+held.month,n_jobs=args.n_jobs,deterministic=True,force_col_wise=True,verbosity=-1)
    m.fit(x,y,group=base._query_groups(train)); gain=m.booster_.feature_importance(importance_type='gain'); split=m.booster_.feature_importance(importance_type='split')
    return pd.DataFrame({'held_month':f'{held:%Y-%m}','feature':fields,'gain':gain,'split':split})
def main():
    p=argparse.ArgumentParser();p.add_argument('--feature-roots',required=True);p.add_argument('--label-root',type=Path,required=True);p.add_argument('--router-root',type=Path,required=True);p.add_argument('--target',choices=base.TARGETS,required=True);p.add_argument('--gain',choices=base.GAIN_SCHEDULES,required=True);p.add_argument('--objective',choices=('lambdarank','rank_xendcg'),required=True);p.add_argument('--truncation',type=int,required=True);p.add_argument('--held-months',default='2025-11,2026-01,2026-03');p.add_argument('--train-months',type=int,default=3);p.add_argument('--reserve-days',type=int,default=28);p.add_argument('--train-cap',type=int,default=60000);p.add_argument('--n-jobs',type=int,default=4);p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    if args.out.exists(): raise FileExistsError(args.out)
    args.feature_roots=tuple(Path(x).resolve() for x in args.feature_roots.split(',')); args.out.mkdir(parents=True)
    months=tuple(utc(x.strip()+'-01') for x in args.held_months.split(',')); fields=universe(args.feature_roots); stats,fields=hygiene(fields,args.feature_roots,months)
    gain=pd.concat([fit_fold(args,fields,m) for m in months],ignore_index=True); summary=gain.groupby('feature').agg(gain_median=('gain','median'),gain_mean=('gain','mean'),use_fraction=('split',lambda x:float((x>0).mean()))).reset_index().merge(stats,on='feature',how='left')
    summary['score']=summary.gain_median*summary.use_fraction; chosen=summary.sort_values(['score','feature'],ascending=[False,True]).head(120).feature.tolist()
    gain.to_parquet(args.out/'fold_gain.parquet',index=False,compression='zstd');summary.sort_values('score',ascending=False).to_parquet(args.out/'feature_summary.parquet',index=False,compression='zstd')
    once(args.out/'selection.json',{'schema':'strict_r3_router_single_base_feature_screen_v1','selected_features':chosen,'target':args.target,'gain':args.gain,'objective':args.objective,'truncation':args.truncation,'held_months':[f'{x:%Y-%m}' for x in months],'hygiene':{'coverage_min':.90,'spearman_veto':.97},'causality':'Router50 exact identities; train-only resolved labels; held scoring target-free'})
    once(args.out/'run_manifest.json',{'scope':'offline feature screen only; no live/exchange mutation','causal_universe':len(universe(args.feature_roots)),'hygiene_fields':len(fields),'selected_fields':len(chosen)})
if __name__=='__main__': main()
