#!/usr/bin/env python3
"""Sequential A0--A3 shared regime-aware residual ablation (no local experts).

Uses the frozen TP6/SL4/H12, fixed-100-bps base-OOF population.  Base-net and
regime residual priors are materialised day-prequentially: each row can use
only earlier fully resolved days.  The soft state sidecars are sealed causal
OOF/prequential artefacts; unavailable state remains explicit via availability
features, never silently represented as a zero transition probability.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT=Path(__file__).resolve().parents[1]
RAW=ROOT/'data_perp/artifacts/tp6_m6_hierarchy_20260809_v1_stage'
OUT=ROOT/'data_perp/artifacts/tp6_shared_residual_a0_a3_20260809_v1'
SIDE=ROOT/'data_perp/artifacts/authoritative_soft_regime_transition_sidecars_20260730_v1'
ERAS=['2023-07_08','2023-09_10','2023-11_12','2024-01_02','2024-05_06','2024-07_08','2024-09_10','2024-11']
CTX=['mkt_ret_eq_24h','regime_liquidity_score','mkt_rv_ratio_1h_24h','mkt_oi_chg_z_24h','mkt_funding_dispersion','cross_asset_corr_4h','mkt_systemic_deleveraging_score','mkt_flush_exhaustion_score','post_liquidation_rebound_score','negative_breadth_pct','btc_resilience_alt_weakness','short_covering_score_market','deleveraging_without_followthrough','short_signal_recovery_conflict']
SOFT=['bocpd__change_probability_mean','bocpd__run_length_mean','bocpd__run_length_entropy','bocpd__state_age_hours','bocpd_ood_score','lgbm_transition_probability','lgbm_entropy','lgbm_margin','lgbm_ood_score','bocpd_onset_h1_probability','bocpd_onset_h12_probability','bocpd_stable_vs_transition_probability']
BASE=['p_adverse','p_weak','p_clear','base_raw']
ARMS={'A0_current_residual':'prior_none','A1_side_centered':'prior_side','A2_side_regime_centered':'prior_hard','A3_soft_regime_centered':'prior_soft'}
TOPS=(.01,.05,.10)
LABEL_AVAILABILITY_DELAY=pd.Timedelta(hours=13)  # signal close +1h entry + H12

def reg(): return lgb.LGBMRegressor(objective='huber',alpha=.9,n_estimators=180,learning_rate=.035,num_leaves=24,min_child_samples=400,colsample_bytree=.8,subsample=.8,reg_lambda=12.,random_state=20260809,n_jobs=1,verbosity=-1)
def arr(x,c): return x[c].replace([np.inf,-np.inf],np.nan).fillna(0.).to_numpy(np.float32)

def ensure_label_available_ts(x):
 if 'label_available_ts' in x: available=pd.to_datetime(x.label_available_ts,utc=True,errors='raise')
 elif '__label_available_at__' in x: available=pd.to_datetime(x.__label_available_at__,utc=True,errors='raise')
 else: available=pd.to_datetime(x.__ts__,utc=True)+LABEL_AVAILABILITY_DELAY
 if available.isna().any() or (available < pd.to_datetime(x.__ts__,utc=True)+LABEL_AVAILABILITY_DELAY).any(): raise ValueError('H12 availability must be signal-close +13h or later')
 x=x.copy();x['label_available_ts']=available;return x

def load_raw():
    xs=[]
    for e in ERAS:
        x=pd.read_parquet(RAW/f'{e}.parquet');x['era']=e;xs.append(x)
    x=pd.concat(xs,ignore_index=True);x['__ts__']=pd.to_datetime(x.__ts__,utc=True);x=ensure_label_available_ts(x)
    if not np.allclose(x.gross_bps-x.net_bps,100.,atol=.02): raise ValueError('cost contract')
    return x.sort_values('__ts__',kind='mergesort').reset_index(drop=True)

def soft_join(x):
    r=pd.read_parquet(SIDE/'soft_regime_hourly.parquet');t=pd.read_parquet(SIDE/'soft_transition_hourly.parquet')
    r=r.rename(columns={'source_utc':'__ts__'});t=t.rename(columns={'source_utc':'__ts__'})
    # Require that any available OOF prediction was fit before its decision row.
    for z,avail,end in [(r,'bocpd_regime_available','train_end_exclusive_utc_bocpd'),(t,'lgbm_transition_available','train_end_exclusive_utc_lgbm')]:
        m=z[avail].fillna(False)
        if m.any() and not (pd.to_datetime(z.loc[m,end],utc=True)<=pd.to_datetime(z.loc[m,'__ts__'],utc=True)).all(): raise ValueError('soft-state lineage breach')
    rcols=['__ts__','bocpd_regime_available','bocpd_ood_available','bocpd__change_probability_mean','bocpd__run_length_mean','bocpd__run_length_entropy','bocpd__state_age_hours','bocpd_ood_score']
    tcols=['__ts__','lgbm_transition_available','lgbm_ood_available','lgbm_transition_probability','lgbm_entropy','lgbm_margin','lgbm_ood_score','bocpd_onset_h1_probability','bocpd_onset_h12_probability','bocpd_stable_vs_transition_probability']
    z=x.merge(r[rcols],on='__ts__',how='left',validate='many_to_one').merge(t[tcols],on='__ts__',how='left',validate='many_to_one')
    z['soft_available']=(z.bocpd_regime_available.fillna(False)|z.lgbm_transition_available.fillna(False)).astype(float)
    for c in SOFT: z[c+'_missing']=~np.isfinite(z[c]); z[c]=z[c].replace([np.inf,-np.inf],np.nan).fillna(0.)
    z['p_change']=z['bocpd__change_probability_mean'].clip(0,1)
    z['p_transition']=np.maximum(z['lgbm_transition_probability'],z['bocpd_onset_h12_probability']).clip(0,1)
    z['p_transition']*=1-z.p_change
    z['p_stable']=(1-z.p_change-z.p_transition).clip(0,1)
    z['hard_regime']=np.argmax(z[['p_stable','p_transition','p_change']].to_numpy(),axis=1)
    return z

def materialise_prequential(x):
    """Fixed bins; priors admit only rows whose exact labels are resolved."""
    x=ensure_label_available_ts(x).reset_index(drop=True); x['base_signal']=(x.p_clear-x.p_adverse).clip(-1,1);x['base_bin']=np.clip(((x.base_signal+1)*5).astype(int),0,9)
    for c in ['base_expected_bps','prior_side','prior_hard','prior_soft']:x[c]=0.
    # running side and side×hard sufficient statistics; component soft stats
    bsum=np.zeros((2,10));bn=np.zeros((2,10));ssum=np.zeros(2);sn=np.zeros(2);hsum=np.zeros((2,3));hn=np.zeros((2,3));wsum=np.zeros((2,3));wn=np.zeros((2,3))
    day=x['__ts__'].dt.floor('D');applied=np.zeros(len(x),dtype=bool)
    for start,ind in x.groupby(day,sort=True).groups.items():
        # Apply deferred outcomes only once their exact availability is before
        # this daily prediction boundary.  In particular, a late previous-day
        # signal is not allowed to update the next-day prior at midnight.
        ready=np.flatnonzero(x.label_available_ts.lt(pd.Timestamp(start)).to_numpy() & ~applied)
        if len(ready):
            ri=ready;side0=x.loc[ri,'side_name'].eq('long').to_numpy(int);b0=x.loc[ri,'base_bin'].to_numpy(int);h0=x.loc[ri,'hard_regime'].to_numpy(int);rr0=x.loc[ri,'net_bps'].to_numpy()-x.loc[ri,'base_expected_bps'].to_numpy();probs0=x.loc[ri,['p_stable','p_transition','p_change']].to_numpy()
            for s in (0,1):
                m=side0==s
                if not m.any(): continue
                np.add.at(bsum[s],b0[m],x.loc[ri[m],'net_bps'].to_numpy());np.add.at(bn[s],b0[m],1);ssum[s]+=rr0[m].sum();sn[s]+=m.sum();np.add.at(hsum[s],h0[m],rr0[m]);np.add.at(hn[s],h0[m],1);wsum[s]+=(probs0[m]*rr0[m,None]).sum(0);wn[s]+=probs0[m].sum(0)
            applied[ri]=True
        ix=np.asarray(list(ind)); side=x.loc[ix,'side_name'].eq('long').to_numpy(int);b=x.loc[ix,'base_bin'].to_numpy(int);h=x.loc[ix,'hard_regime'].to_numpy(int)
        # binned expected net, side-shrunk (pseudo-count 500)
        side_mean=np.divide(ssum,sn,out=np.zeros(2),where=sn>0);bm=np.divide(bsum[side,b],bn[side,b],out=side_mean[side].copy(),where=bn[side,b]>0)
        x.loc[ix,'base_expected_bps']=(bn[side,b]*bm+500*side_mean[side])/(bn[side,b]+500)
        residual=x.loc[ix,'net_bps'].to_numpy()-x.loc[ix,'base_expected_bps'].to_numpy()
        sp=side_mean[side]; hp=np.divide(hsum[side,h],hn[side,h],out=sp.copy(),where=hn[side,h]>0);hard=(hn[side,h]*hp+1000*sp)/(hn[side,h]+1000)
        probs=x.loc[ix,['p_stable','p_transition','p_change']].to_numpy(); comp=np.divide(wsum,wn,out=np.broadcast_to(side_mean[:,None],(2,3)).copy(),where=wn>0);comp=(wn*comp+1000*side_mean[:,None])/(wn+1000);soft=(comp[side]*probs).sum(1)
        x.loc[ix,'prior_side']=sp;x.loc[ix,'prior_hard']=hard;x.loc[ix,'prior_soft']=soft
    return x

def metrics(z,score,common):
    out=[]
    for view,q in [('global',z),('long',z[z.side_name.eq('long')]),('short',z[z.side_name.eq('short')])]:
        for top in TOPS:
            take=q.sort_values([score,'candidate_id'],ascending=[False,True],kind='mergesort').head(max(1,int(np.ceil(len(q)*top))))
            out.append({**common,'view':view,'top_fraction':top,'n':len(take),'net_bps':float(take.net_bps.mean()),'gross_bps':float(take.gross_bps.mean()),'residual_ic':float(spearmanr(q[score],q.net_bps).statistic),'selected_long_fraction':float(take.side_name.eq('long').mean())})
    return out

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,default=OUT);ap.add_argument('--materialise',action='store_true');ap.add_argument('--only-era',choices=ERAS[1:]);ap.add_argument('--only-arm',choices=list(ARMS));ap.add_argument('--finalize',action='store_true');a=ap.parse_args();stage=a.out.with_name(a.out.name+'_stage');feat=stage/'prequential_features.parquet'
 if a.materialise:
  stage.mkdir(exist_ok=True);x=materialise_prequential(soft_join(load_raw()));x.to_parquet(feat,index=False);(stage/'materialisation.json').write_text(json.dumps({'rows':len(x),'contract':'soft OOF/prequential sidecars; daily prior-resolved mappings'},indent=2));print(json.dumps({'features':str(feat),'rows':len(x)}));return
 if a.finalize:
  cs=sorted((stage/'checkpoints').glob('*.parquet'));m=pd.concat([pd.read_parquet(p) for p in cs],ignore_index=True);m=m.drop_duplicates(['arm','test_era','view','top_fraction'],keep='last'); need={(a,e) for a in ARMS for e in ERAS[1:]};got=set(zip(m.arm,m.test_era));
  if need-got: raise ValueError(f'missing checkpoints: {sorted(need-got)}')
  a.out.mkdir(exist_ok=True);m.to_parquet(a.out/'metrics.parquet',index=False);g=m[(m.view.eq('global')) & (m.top_fraction.eq(.01))];s=g.groupby('arm',as_index=False).agg(mean_top1=('net_bps','mean'),worst_top1=('net_bps','min'),positive_eras=('net_bps',lambda v:int((v>0).sum())),eras=('test_era','nunique'));s.to_parquet(a.out/'summary.parquet',index=False);table='| arm | mean top-1 net bps | worst top-1 net bps | positive eras | eras |\n|---|---:|---:|---:|---:|\n'+'\n'.join(f"| {r.arm} | {r.mean_top1:.3f} | {r.worst_top1:.3f} | {int(r.positive_eras)} | {int(r.eras)} |" for r in s.itertuples(index=False));(a.out/'REPORT.md').write_text('# A0–A3 shared residual ablation\n\n'+table+'\n\nSelection is lexicographic: worst era, then mean; diagnostic only, no promotion.\n');(a.out/'manifest.json').write_text(json.dumps({'status':'COMPLETED_DIAGNOSTIC_NO_PROMOTION','arms':ARMS,'weights':'square-root training-era, mean-normalised','model':'single shared Huber LGBM, no local experts','soft_state':'sealed causal OOF/prequential sidecars'},indent=2));print(s.to_json(orient='records'));return
 if not a.only_era or not feat.exists():raise ValueError('materialise first, then --only-era')
 x=pd.read_parquet(feat);i=ERAS.index(a.only_era);train=x[x.era.isin(ERAS[:i])].copy();test=x[x.era.eq(a.only_era)].copy();
 # square-root environment weights, fit solely from earlier environments
 n=train.groupby('era').size();w=train.era.map(np.sqrt(len(train)/(len(n)*n)));w=np.clip(w/w.mean(),.25,4.)
 f=[*BASE,'base_expected_bps',*CTX,*SOFT,*[c+'_missing' for c in SOFT], 'soft_available','p_stable','p_transition','p_change']
 rows=[]
 for arm,prior in ARMS.items():
  if a.only_arm and arm != a.only_arm: continue
  offset_train=0. if prior=='prior_none' else train[prior]; offset_test=0. if prior=='prior_none' else test[prior]
  y=train.net_bps-train.base_expected_bps-offset_train;m=reg().fit(arr(train,f),y,sample_weight=w);test2=test.copy();test2['score']=test2.base_expected_bps+offset_test+m.predict(arr(test2,f));rows+=metrics(test2,'score',{'arm':arm,'test_era':a.only_era,'train_through':ERAS[i-1],'train_rows':len(train),'test_rows':len(test)})
 ck=stage/'checkpoints';ck.mkdir(exist_ok=True);suffix=f'_{a.only_arm}' if a.only_arm else '';pd.DataFrame(rows).to_parquet(ck/f'{a.only_era}{suffix}.parquet',index=False);print(json.dumps({'checkpoint':a.only_era,'arm':a.only_arm or 'all','rows':len(rows)}))
if __name__=='__main__':main()
