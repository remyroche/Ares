#!/usr/bin/env python3
"""Stage-1 full-universe T2/T4 target screen with strict layer contracts.

The screen is intentionally narrow: four fixed ATR barrier geometries for a
soft three-state T2 model and an ATR-normalised T4 quantile model.  Features
are side-local training-only subsets from the audited base pool.  Evaluation
is one pooled global book, never timestamp-local or side-quota based.
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path
import lightgbm as lgb
from catboost import CatBoostRegressor
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

GEOMS = ("tp2_sl1", "tp2_sl2", "tp3_sl1", "tp3_sl2")
TP_SL = {"tp2_sl1": (2.,1.), "tp2_sl2": (2.,2.), "tp3_sl1": (3.,1.), "tp3_sl2": (3.,2.)}
QUANTILES = (.10,.25,.50,.75,.90)
PARAMS = dict(n_estimators=80, learning_rate=.06, num_leaves=24, min_child_samples=400,
              colsample_bytree=.80, subsample=.80, reg_lambda=8., random_state=20260801,
              n_jobs=1, verbosity=-1)
MODEL_FAMILY = 'lightgbm'

def _read(panel: Path, columns: list[str]) -> pd.DataFrame:
    return pd.concat([pd.read_parquet(p, columns=columns) for p in sorted((panel/'parts').glob('*.parquet'))], ignore_index=True)

def _family(name: str) -> str:
    for token, family in (("ob_","orderbook"),("fund","funding"),("oi","open_interest"),("volume","volume"),("vol_","volatility"),("atr","volatility"),("ret","returns"),("trend","trend"),("price","price"),("range","price"),("support","levels"),("resistance","levels"),("vwap","levels"),("donch","levels"),("mkt","cross_asset"),("xasset","cross_asset")):
        if token in name: return family
    return "other"

def _subset(train: pd.DataFrame, candidates: list[str], target: np.ndarray, n: int=36, *, mda: bool=False) -> list[str]:
    """Training-only rank screen with family caps for a diverse 30–40 set."""
    # Rank screening does not need every highly correlated row; a fixed
    # chronological-stratified sample keeps this gate fast while keeping all
    # rows in the actual side-local model fit.
    if len(train) > 250_000:
        take=np.linspace(0,len(train)-1,250_000,dtype=int)
        screen=train.iloc[take]; target=target[take]
    else: screen=train
    score=[]
    for col in candidates:
        x=pd.to_numeric(screen[col], errors='coerce').to_numpy(float)
        good=np.isfinite(x) & np.isfinite(target)
        if good.sum() < 500 or good.mean() < .90: continue
        v=spearmanr(x[good], target[good]).statistic
        score.append((abs(float(v)) if np.isfinite(v) else -1., col))
    ordered=[col for _,col in sorted(score, reverse=True)]
    if mda:
        # A small chronological permutation-importance gate follows the rank
        # screen.  It is deliberately fit only on the early portion of the
        # base training window and evaluated on its later portion, so it is
        # feature selection rather than an evaluation-period tuning leak.
        shortlist=ordered[:min(len(ordered),max(48,n+12))]
        sample=screen.iloc[:min(len(screen),100_000)].copy(); y=target[:len(sample)]
        cut=max(500,int(len(sample)*.75)); fit=sample.iloc[:cut]; valid=sample.iloc[cut:]
        xv=_x(valid,shortlist); yf=y[:cut]; yv=y[cut:]
        model=lgb.LGBMRegressor(objective='huber',alpha=.90,n_estimators=40,learning_rate=.07,num_leaves=20,min_child_samples=250,colsample_bytree=.8,subsample=.8,reg_lambda=8.,random_state=20260801,n_jobs=1,verbosity=-1).fit(_x(fit,shortlist),yf)
        base_pred=model.predict(xv); baseline=spearmanr(base_pred,yv).statistic
        rng=np.random.default_rng(20260801); importance=[]
        for j,col in enumerate(shortlist):
            altered=xv.copy(); altered[:,j]=rng.permutation(altered[:,j]); v=spearmanr(model.predict(altered),yv).statistic
            importance.append(((baseline-v) if np.isfinite(v) and np.isfinite(baseline) else -np.inf,col))
        ordered=[col for _,col in sorted(importance,reverse=True)]
    chosen=[]; count={}
    for col in ordered:
        family=_family(col)
        if count.get(family,0) >= 5: continue
        chosen.append(col); count[family]=count.get(family,0)+1
        if len(chosen)==n: break
    if len(chosen) < 30: raise RuntimeError(f"only {len(chosen)} diverse admissible base features")
    return chosen

def _x(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    return df[cols].replace([np.inf,-np.inf],np.nan).fillna(0.).to_numpy(np.float32)

def _soft(frame: pd.DataFrame, geom: str, tau: float) -> np.ndarray:
    """Fixed soft three-state label. First hit is retained via a winner logit;
    H12 MFE/MAE supplies near-miss information without entering inference."""
    tp,sl=TP_SL[geom]; event=frame[f't2_{geom}_event'].to_numpy(int)
    exitm=frame[f't2_{geom}_exit_minute'].to_numpy(float)
    mfe=frame.t2_path_mfe_atr.to_numpy(float); mae=frame.t2_path_mae_atr.to_numpy(float)
    up=(mfe-tp)/tau; down=(mae-sl)/tau
    timeout=np.minimum((tp-mfe)/tau, (sl-mae)/tau)
    # winner bonus declines gently with late hit; it resolves paths that cross
    # both barriers later while preserving the exact first-touch label.
    bonus=2.0 + .75*(1.-np.minimum(exitm,720.)/720.)
    up[event==0]+=bonus[event==0]; down[event==1]+=bonus[event==1]; timeout[event==2]+=2.0
    z=np.column_stack([up,down,timeout]); z-=z.max(1,keepdims=True)
    p=np.exp(np.clip(z,-40,0)); return p/p.sum(1,keepdims=True)

def _top_metrics(frame: pd.DataFrame) -> list[dict]:
    ordered=frame.sort_values(['score_bps','candidate_id'],ascending=[False,True],kind='mergesort')
    out=[]
    for frac in (.01,.05,.10,.20):
        x=ordered.head(int(np.ceil(len(ordered)*frac)))
        for side, y in [('all',x),('long',x[x.side_name.eq('long')]),('short',x[x.side_name.eq('short')])]:
            out.append(dict(top_fraction=frac, side=side, n=len(y), gross_bps=float(y.gross_bps.mean()), net_bps=float(y.net_bps.mean())))
    return out

def _train_t2(tr, dv, cols, geom, tau, *, catboost_iterations: int, catboost_depth: int, sample_weight: np.ndarray | None=None):
    label=_soft(tr,geom,tau); xt,xd=_x(tr,cols),_x(dv,cols); pred=[]
    for j in range(3):
        if MODEL_FAMILY == 'catboost':
            # Keep CatBoost capacity explicit in the experiment contract.  A
            # fixed hidden 200-round setting made it impossible to run a
            # bounded, reproducible model-family comparison.
            m=CatBoostRegressor(loss_function='RMSE', iterations=catboost_iterations, depth=catboost_depth, learning_rate=.05, l2_leaf_reg=8.0, random_seed=20260801, thread_count=1, verbose=False).fit(xt,label[:,j],sample_weight=sample_weight)
        else:
            m=lgb.LGBMRegressor(objective='huber',alpha=.90,**PARAMS).fit(xt,label[:,j],sample_weight=sample_weight)
        pred.append(np.maximum(m.predict(xd),0.))
    p=np.column_stack(pred); p/=np.maximum(p.sum(1,keepdims=True),1e-8)
    net=tr[f't4_{geom}_net_bps'].to_numpy(float)
    weight=np.ones(len(tr),dtype=float) if sample_weight is None else sample_weight
    means=(label*net[:,None]*weight[:,None]).sum(0)/np.maximum((label*weight[:,None]).sum(0),1.)
    return p@means, {'p_upper':p[:,0],'p_lower':p[:,1],'p_timeout':p[:,2], 'conditional_net_means_bps':means.tolist()}

def _train_t4(tr, dv, cols, geom):
    xt,xd=_x(tr,cols),_x(dv,cols); y=tr[f't4_{geom}_exit_pnl_atr'].to_numpy(float)
    qs=[]
    for q in QUANTILES:
        qs.append(lgb.LGBMRegressor(objective='quantile',alpha=q,**PARAMS).fit(xt,y).predict(xd))
    q=np.sort(np.column_stack(qs),axis=1)
    scale=dv.atr_1h.to_numpy(float)/dv.decision_price.to_numpy(float)*1e4
    qbps=q*scale[:,None]-dv.assumed_round_trip_cost_bps.to_numpy(float)[:,None]
    return qbps.mean(1), {f'q{int(100*q)}_net_bps':qbps[:,i] for i,q in enumerate(QUANTILES)}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--panel',type=Path,required=True); ap.add_argument('--audit',type=Path,required=True); ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--train-end',default='2024-04-01'); ap.add_argument('--dev-end',default='2024-08-01')
    ap.add_argument('--family', choices=('T2_soft_barrier','T4_atr_quantile'))
    ap.add_argument('--geometry', choices=GEOMS)
    ap.add_argument('--side', choices=('long','short'))
    ap.add_argument('--tau', type=float, default=.25, help='T2 softness; ignored by T4')
    ap.add_argument('--trees', type=int, default=80)
    ap.add_argument('--leaves', type=int, default=24)
    ap.add_argument('--feature-count', type=int, default=36)
    ap.add_argument('--model-family', choices=('lightgbm','catboost'), default='lightgbm')
    ap.add_argument('--catboost-iterations', type=int, default=200)
    ap.add_argument('--catboost-depth', type=int, default=6)
    ap.add_argument('--mda', action='store_true', help='apply a chronological permutation-importance gate after rank screening')
    ap.add_argument('--certainty', type=Path, help='training-only full-universe certainty ledger')
    ap.add_argument('--certainty-weighting', choices=('none','mild','strong'), default='none')
    args=ap.parse_args()
    audit=json.loads(args.audit.read_text()); base=audit['base']['coverage_ge_90pct']
    label_cols=['candidate_id','__ts__','__label_available_at__','side_name','atr_1h','decision_price','assumed_round_trip_cost_bps','t2_path_mfe_atr','t2_path_mae_atr']
    for g in GEOMS: label_cols += [f't2_{g}_event',f't2_{g}_exit_minute',f't4_{g}_exit_pnl_atr',f't4_{g}_gross_bps',f't4_{g}_net_bps']
    data=_read(args.panel,list(dict.fromkeys(label_cols+base))); data['__ts__']=pd.to_datetime(data['__ts__'],utc=True)
    if args.certainty:
        cert=pd.read_parquet(args.certainty,columns=['candidate_id','label_certainty'])
        data=data.merge(cert,on='candidate_id',how='left',validate='one_to_one')
        if data.label_certainty.isna().any(): raise RuntimeError('certainty ledger does not cover every full-universe candidate')
    elif args.certainty_weighting != 'none':
        ap.error('--certainty is required when --certainty-weighting is not none')
    train_end=pd.Timestamp(args.train_end,tz='UTC')
    # H12 outcomes are unavailable until decision + 12h.  The former
    # timestamp-only split leaked the last resolved horizon at this boundary.
    train=data[data.__ts__.lt(train_end)&(pd.to_datetime(data.__label_available_at__,utc=True)<train_end)].copy(); dev=data[(data.__ts__>=train_end)&(data.__ts__<pd.Timestamp(args.dev_end,tz='UTC'))].copy()
    if train.empty or dev.empty: raise RuntimeError('empty chronological split')
    results=[]; predictions=[]; feature_contract={}
    families=(args.family,) if args.family else ('T2_soft_barrier','T4_atr_quantile')
    geometries=(args.geometry,) if args.geometry else GEOMS
    for family in families:
      for geom in geometries:
       for side in ((args.side,) if args.side else ('long','short')):
        tr=train[train.side_name.eq(side)].copy(); dv=dev[dev.side_name.eq(side)].copy()
        proxy=tr[f't4_{geom}_net_bps'].to_numpy(float)
        cols=_subset(tr,base,proxy,args.feature_count,mda=args.mda)
        feature_contract[f'{family}|{geom}|{side}']=cols
        global PARAMS, MODEL_FAMILY
        MODEL_FAMILY=args.model_family
        old_trees=PARAMS['n_estimators']; old_leaves=PARAMS['num_leaves']; PARAMS={**PARAMS,'n_estimators':args.trees,'num_leaves':args.leaves}
        if args.certainty_weighting == 'mild': weights=.5+.5*tr.label_certainty.to_numpy(float)
        elif args.certainty_weighting == 'strong': weights=.25+.75*tr.label_certainty.to_numpy(float)
        else: weights=None
        if family.startswith('T2'): score,extra=_train_t2(tr,dv,cols,geom,args.tau,catboost_iterations=args.catboost_iterations,catboost_depth=args.catboost_depth,sample_weight=weights)
        else: score,extra=_train_t4(tr,dv,cols,geom)
        PARAMS={**PARAMS,'n_estimators':old_trees,'num_leaves':old_leaves}
        out=dv[['candidate_id','__ts__','side_name',f't4_{geom}_gross_bps',f't4_{geom}_net_bps']].copy()
        out.columns=['candidate_id','__ts__','side_name','gross_bps','net_bps']; out['score_bps']=score; out['family']=family; out['geometry']=geom; out['tau']=args.tau
        for k,v in extra.items():
            if isinstance(v,np.ndarray): out[k]=v
        predictions.append(out); results.append(dict(family=family,geometry=geom,side_model=side,features=cols,details={k:v for k,v in extra.items() if not isinstance(v,np.ndarray)}))
    pred=pd.concat(predictions,ignore_index=True)
    metrics=[]
    for family in pred.family.unique():
      for geom in pred.loc[pred.family.eq(family), 'geometry'].unique():
       x=pred[(pred.family==family)&(pred.geometry==geom)]
       for item in _top_metrics(x): metrics.append(dict(family=family,geometry=geom,tau=float(x.tau.iloc[0]),month='all',**item))
       for month, y in x.groupby(x.__ts__.dt.to_period('M').astype(str)):
        for item in _top_metrics(y): metrics.append(dict(family=family,geometry=geom,tau=float(x.tau.iloc[0]),month=month,**item))
    met=pd.DataFrame(metrics); top=met[(met.top_fraction==.10)&(met.side=='all')&(met.month=='all')].sort_values(['net_bps','gross_bps'],ascending=False)
    args.out.mkdir(parents=True,exist_ok=True); pred.to_parquet(args.out/'target_screen_predictions.parquet',index=False); met.to_parquet(args.out/'target_screen_metrics.parquet',index=False)
    (args.out/'target_family_manifest.json').write_text(json.dumps(dict(schema='full_universe_t2_t4_target_screen_v1',candidate_rows=len(data),train_window=[str(train.__ts__.min()),args.train_end],development_window=[args.train_end,args.dev_end],entry='next hourly open',exit='first TP/SL then H12 timeout',global_selection='pooled across sides and timestamps after common-bps mapping',cost='100 bps declared round-trip assumption',softness_tau=args.tau,model_family=args.model_family,catboost_iterations=args.catboost_iterations if args.model_family == 'catboost' else None,catboost_depth=args.catboost_depth if args.model_family == 'catboost' else None,feature_selection='chronological rank screen plus MDA' if args.mda else 'chronological rank screen',certainty_weighting=args.certainty_weighting,certainty_training_only=bool(args.certainty),feature_contract=feature_contract,winner=top.iloc[0].to_dict(),arms=results),indent=2,default=str))
    print(top.to_string(index=False))
if __name__=='__main__': main()
