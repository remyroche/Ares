from __future__ import annotations
import hashlib,json,os,platform,sys,tempfile
from pathlib import Path
import numpy as np,pandas as pd,sklearn
from sklearn.linear_model import LogisticRegression,Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss
R=Path(__file__).resolve().parents[1];A=R/'data_perp/artifacts';S=A/'pre2026_oof_model_failure_incremental_value_20260730_v3';V=A/'pre2026_nested_residual_context_failure_overlay_20260730_v3';O=A/'pre2026_regime_overlay_gamma_hpo_20260730_v1';C=['base_score','residual_score','residual_minus_base'];X=['regime_change_probability_mean','regime_change_probability_max','regime_run_length_mean','regime_run_length_q05','regime_run_length_entropy','regime_signal_count','regime_state_age_hours','regime_is_persistent_24h','regime_is_persistent_72h'];G=[.125,.25,.5];CAP=150000;E=1e-4
def h(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def cap(x):return x.iloc[np.argsort(pd.util.hash_pandas_object(x.candidate_id,index=False).to_numpy(),kind='stable')[:CAP]] if len(x)>CAP else x
def fit(tr,te,fs,y,log):
 q=cap(tr);s=StandardScaler().fit(q[fs]);m=(LogisticRegression(C=.02,max_iter=300,class_weight='balanced',random_state=17) if log else Ridge(alpha=30)).fit(s.transform(q[fs]),q[y]);return m.predict_proba(s.transform(te[fs]))[:,1] if log else m.predict(s.transform(te[fs]))
def run():
 if O.exists():raise RuntimeError(O)
 x=pd.read_parquet(S/'materialized_targets.parquet',columns=['candidate_id','__ts__','side_name','era','execution_net_ev_12h','residual_selected_global_top10','residual_selected_net_failure','bocpd_regime_available']+C+X);x.__ts__=pd.to_datetime(x.__ts__,utc=True);x['y']=x.residual_selected_net_failure.astype(int);x=x[x.residual_selected_global_top10&x.bocpd_regime_available.fillna(False)&x[C+X].notna().all(axis=1)]
 out=[]
 for oe,te0 in x.groupby('era'):
  tr0=x[x.era.ne(oe)]
  for side,te in te0.groupby('side_name'):
   tr=tr0[tr0.side_name.eq(side)];inn=[]
   for ie,iv in tr.groupby('era'):
    inn.append(iv.assign(pi=fit(tr[tr.era.ne(ie)],iv,C,'y',True)))
   inn=pd.concat(inn);inn['rr']=inn.y-inn.pi;pc=fit(tr,te,C,'y',True);res=fit(inn,te,X,'rr',False);out.append(te[['candidate_id','__ts__','side_name','era','execution_net_ev_12h','y']].assign(p_core=pc,raw_context_residual=res))
 z=pd.concat(out);rows=[]
 for g in G:
  z[f'p_{g}']=np.clip(z.p_core+g*z.raw_context_residual,E,1-E)
  for (e,s),q in z.groupby(['era','side_name']):rows.append({'gamma':g,'era':e,'side':s,'auc':roc_auc_score(q.y,q[f'p_{g}']),'ap':average_precision_score(q.y,q[f'p_{g}']),'brier':brier_score_loss(q.y,q[f'p_{g}']),'hl':q.loc[q[f'p_{g}']>=q[f'p_{g}'].quantile(.9),'execution_net_ev_12h'].mean()-q.loc[q[f'p_{g}']<=q[f'p_{g}'].quantile(.1),'execution_net_ev_12h'].mean()})
 m=pd.DataFrame(rows);base=[]
 for (e,s),q in z.groupby(['era','side_name']):base.append({'era':e,'side':s,'auc0':roc_auc_score(q.y,q.p_core),'ap0':average_precision_score(q.y,q.p_core),'brier0':brier_score_loss(q.y,q.p_core),'hl0':q.loc[q.p_core>=q.p_core.quantile(.9),'execution_net_ev_12h'].mean()-q.loc[q.p_core<=q.p_core.quantile(.1),'execution_net_ev_12h'].mean()})
 b=pd.DataFrame(base);d=m.merge(b,on=['era','side']);d[['ad','apd','bd','hld']]=d[['auc','ap','brier','hl']].to_numpy()-d[['auc0','ap0','brier0','hl0']].to_numpy();st=O.parent/('.'+O.name+'.tmp');st.mkdir();z.to_parquet(st/'raw_predictions.parquet',index=False);m.to_csv(st/'side_metrics.csv',index=False);d.to_csv(st/'deltas.csv',index=False);c={'schema':'regime_gamma_hpo_v1','status':'SEALED_PRE2026_DIAGNOSTIC_NON_PROMOTION','gammas':G,'implementation_sha256':{str(Path(__file__).resolve()):h(Path(__file__))},'environment':{'python':sys.version,'platform':platform.platform(),'numpy':np.__version__,'pandas':pd.__version__,'sklearn':sklearn.__version__},'no_2026':True};(st/'contract.json').write_text(json.dumps(c,indent=2)+'\n');fs=list(st.iterdir());man={'schema':c['schema'],'status':c['status'],'contract':c,'outputs_sha256':{p.name:h(p) for p in fs}};(st/'manifest.json').write_text(json.dumps(man,indent=2)+'\n');(st/'manifest.sha256').write_text(f'{h(st/"manifest.json")}  manifest.json\n');os.replace(st,O);print(O)
if __name__=='__main__':run()
