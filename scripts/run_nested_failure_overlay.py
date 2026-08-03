#!/usr/bin/env python3
from __future__ import annotations
import hashlib,json,os,platform,shutil,tempfile,sys
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.linear_model import LogisticRegression,Ridge
from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss
from sklearn.preprocessing import StandardScaler
ROOT=Path(__file__).resolve().parents[1];A=ROOT/'data_perp/artifacts';SRC=A/'pre2026_oof_model_failure_incremental_value_20260730_v3';OUT=A/'pre2026_nested_residual_context_failure_overlay_20260730_v1';CAP=150000;G=.5;EPS=1e-4
CORE=['base_score','residual_score','residual_minus_base']; REG=['regime_change_probability_mean','regime_change_probability_max','regime_run_length_mean','regime_run_length_q05','regime_run_length_entropy','regime_signal_count','regime_state_age_hours','regime_is_persistent_24h','regime_is_persistent_72h']; TRA=['transition_lgbm_probability','transition_lgbm_entropy','transition_lgbm_margin','transition_bocpd_stable_probability','transition_bocpd_onset_h1_probability','transition_bocpd_onset_h3_probability','transition_bocpd_onset_h6_probability','transition_bocpd_onset_h12_probability']; TJ=['trajectory_available','trajectory_transition_probability','trajectory_probability_entropy','trajectory_top2_margin']; ARMS={'regime':REG,'transition':TRA,'trajectory':TJ,'combined':REG+TRA+TJ}
def h(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ids(x):return hashlib.sha256('|'.join(x.candidate_id.sort_values()).encode()).hexdigest()
def cap(x):return x.iloc[np.argsort(pd.util.hash_pandas_object(x.candidate_id,index=False).to_numpy(),kind='stable')[:CAP]] if len(x)>CAP else x
def corefit(tr,te):
 q=cap(tr);s=StandardScaler().fit(q[CORE]);m=LogisticRegression(C=.02,max_iter=300,class_weight='balanced',random_state=17).fit(s.transform(q[CORE]),q.y);return m.predict_proba(s.transform(te[CORE]))[:,1],len(q)
def run():
 if OUT.exists():raise RuntimeError(OUT)
 if h(SRC/'manifest.json')!=(SRC/'manifest.sha256').read_text().split()[0]:raise RuntimeError('unsealed')
 common=['candidate_id','__ts__','side_name','era','execution_net_ev_12h','residual_selected_global_top10','residual_selected_net_failure','bocpd_regime_available','lgbm_transition_available','trajectory_available']
 st=Path(tempfile.mkdtemp(dir=OUT.parent,prefix='.'+OUT.name+'.'));out=[];aud=[]
 try:
  for arm,ctx in ARMS.items():
   x=pd.read_parquet(SRC/'materialized_targets.parquet',columns=list(dict.fromkeys(common+CORE+ctx)));x.__ts__=pd.to_datetime(x.__ts__,utc=True);x['y']=x.residual_selected_net_failure.astype(int);ok=x.residual_selected_global_top10 & x[CORE+ctx].notna().all(axis=1)
   if arm in ['regime','combined']:ok&=x.bocpd_regime_available.fillna(False)
   if arm in ['transition','combined']:ok&=x.lgbm_transition_available.fillna(False)
   if arm in ['trajectory','combined']:ok&=x.trajectory_available.fillna(False)
   x=x[ok]
   for outer,te0 in x.groupby('era',sort=True):
    tr0=x[x.era.ne(outer)]
    for side,te in te0.groupby('side_name',sort=True):
     tr=tr0[tr0.side_name.eq(side)]; inner=[]
     for inn,iv in tr.groupby('era',sort=True):
      it=tr[tr.era.ne(inn)];p,_=corefit(it,iv);inner.append(iv.assign(p_inner=p))
     inn=pd.concat(inner);pout,post=corefit(tr,te);q=cap(inn);sc=StandardScaler().fit(q[ctx]);r=Ridge(alpha=30).fit(sc.transform(q[ctx]),q.y-q.p_inner);res=r.predict(sc.transform(te[ctx]));po=np.clip(pout+G*res,EPS,1-EPS)
     for kind,p in [('core',pout),('overlay',po)]:out.append(te[['candidate_id','era','side_name','execution_net_ev_12h','y']].assign(arm=arm,kind=kind,p=p))
     aud.append({'arm':arm,'outer_era':outer,'side':side,'pre_cap_train_rows':len(tr),'post_cap_train_rows':post,'train_hash':ids(tr),'test_rows':len(te),'test_hash':ids(te),'inner_eras':inn.era.nunique()})
   del x
  z=pd.concat(out);met=[]
  for (a,k,e,s),q in z.groupby(['arm','kind','era','side_name']):met.append({'arm':a,'kind':k,'era':e,'side':s,'auc':roc_auc_score(q.y,q.p),'ap':average_precision_score(q.y,q.p),'brier':brier_score_loss(q.y,q.p),'high_low_ev':q.loc[q.p>=q.p.quantile(.9),'execution_net_ev_12h'].mean()-q.loc[q.p<=q.p.quantile(.1),'execution_net_ev_12h'].mean()})
  m=pd.DataFrame(met);pool=m.groupby(['arm','kind','era']).mean(numeric_only=True).reset_index();d=pool.pivot(index=['arm','era'],columns='kind',values=['auc','brier']).reset_index();d['auc_delta']=d[('auc','overlay')]-d[('auc','core')];d['brier_delta']=d[('brier','overlay')]-d[('brier','core')];g=[]
  for a,q in d.groupby(('arm','')):
   sides=m.pivot_table(index=['arm','era','side'],columns='kind',values='auc').reset_index();ss=sides[sides.arm.eq(a)];sd=ss.overlay-ss.core;g.append({'arm':a,'median_auc_delta':q.auc_delta.median(),'min_auc_delta':q.auc_delta.min(),'positive_fraction':(q.auc_delta>0).mean(),'median_brier_delta':q.brier_delta.median(),'long_median_auc_delta':sd[ss.side.eq('long')].median(),'short_median_auc_delta':sd[ss.side.eq('short')].median(),'eligible':bool(q.auc_delta.median()>0 and (q.auc_delta>0).mean()>=.75 and q.auc_delta.min()>=-.02 and q.brier_delta.median()<=0 and sd[ss.side.eq('long')].median()>=0 and sd[ss.side.eq('short')].median()>=0)})
  z.to_parquet(st/'predictions.parquet',index=False);m.to_csv(st/'metrics.csv',index=False);d.to_csv(st/'pooled_deltas.csv',index=False);pd.DataFrame(g).to_csv(st/'eligibility.csv',index=False);pd.DataFrame(aud).to_csv(st/'cohort_audit.csv',index=False)
  c={'schema':'nested_failure_overlay_v1','status':'SEALED_PRE2026_NESTED_FAILURE_OVERLAY_NON_PROMOTION','decision_cadence':'1h','labels':'1m nested exact-12h only','gamma':G,'ridge_alpha':30,'cap':CAP,'formula':'clip(p_core + .5*context_residual,eps)','implementation_sha256':{str(Path(__file__).resolve()):h(Path(__file__))},'environment':{'python':sys.version,'platform':platform.platform()},'no_2026':True};(st/'contract.json').write_text(json.dumps(c,indent=2,sort_keys=True)+'\n');fs=[p for p in st.iterdir() if p.is_file()];man={'schema':c['schema'],'status':c['status'],'contract':c,'inputs_sha256':{str((SRC/'manifest.json').resolve()):h(SRC/'manifest.json')},'outputs_sha256':{p.name:h(p) for p in fs}};(st/'manifest.json').write_text(json.dumps(man,indent=2,sort_keys=True)+'\n');(st/'manifest.sha256').write_text(f'{h(st/"manifest.json")}  manifest.json\n');os.replace(st,OUT);print(OUT)
 except Exception:shutil.rmtree(st,ignore_errors=True);raise
if __name__=='__main__':run()
