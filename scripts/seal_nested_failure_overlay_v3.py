from __future__ import annotations
import hashlib,json,os,platform,sys,tempfile
from pathlib import Path
import numpy as np,pandas as pd,sklearn
from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss
ROOT=Path(__file__).resolve().parents[1];A=ROOT/'data_perp/artifacts';V1=A/'pre2026_nested_residual_context_failure_overlay_20260730_v1';V2=A/'pre2026_nested_residual_context_failure_overlay_20260730_v2';S=A/'pre2026_oof_model_failure_incremental_value_20260730_v3';O=A/'pre2026_nested_residual_context_failure_overlay_20260730_v3';CAP=150000
def h(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ih(x):return hashlib.sha256('|'.join(x.candidate_id.sort_values()).encode()).hexdigest()
def cap(x):return x.iloc[np.argsort(pd.util.hash_pandas_object(x.candidate_id,index=False).to_numpy(),kind='stable')[:CAP]] if len(x)>CAP else x
def run():
 if O.exists():raise RuntimeError(O)
 for p in [V1,V2,S]:
  if h(p/'manifest.json')!=(p/'manifest.sha256').read_text().split()[0]:raise RuntimeError('unsealed')
 z=pd.read_parquet(V1/'predictions.parquet');src=pd.read_parquet(S/'materialized_targets.parquet');src.__ts__=pd.to_datetime(src.__ts__,utc=True);src.execution_label_end_utc=pd.to_datetime(src.execution_label_end_utc,utc=True)
 b=z[['candidate_id','arm','kind','era','side_name','y','p','execution_net_ev_12h']].merge(src[['candidate_id','__ts__','execution_label_end_utc']],on='candidate_id',validate='many_to_one')
 if b.__ts__.dt.minute.ne(0).any() or b.execution_label_end_utc.ge(pd.Timestamp('2026-01-01',tz='UTC')).any():raise RuntimeError('cadence')
 met=[]
 for ks,q in b.groupby(['arm','kind','era','side_name']):met.append(dict(zip(['arm','kind','era','side'],ks))|{'auc':roc_auc_score(q.y,q.p),'ap':average_precision_score(q.y,q.p),'brier':brier_score_loss(q.y,q.p),'high_low_ev':q.loc[q.p>=q.p.quantile(.9),'execution_net_ev_12h'].mean()-q.loc[q.p<=q.p.quantile(.1),'execution_net_ev_12h'].mean()})
 side=pd.DataFrame(met);pool=[]
 for ks,q in b.groupby(['arm','kind','era']):pool.append(dict(zip(['arm','kind','era'],ks))|{'auc':roc_auc_score(q.y,q.p),'ap':average_precision_score(q.y,q.p),'brier':brier_score_loss(q.y,q.p),'high_low_ev':q.loc[q.p>=q.p.quantile(.9),'execution_net_ev_12h'].mean()-q.loc[q.p<=q.p.quantile(.1),'execution_net_ev_12h'].mean()})
 pool=pd.DataFrame(pool);piv=pool.pivot(index=['arm','era'],columns='kind',values=['auc','brier']).reset_index();piv['auc_delta']=piv[('auc','overlay')]-piv[('auc','core')];piv['brier_delta']=piv[('brier','overlay')]-piv[('brier','core')];g=[]
 for a,q in piv.groupby(('arm','')):
  ss=side[side.arm.eq(a)].pivot(index=['era','side'],columns='kind',values='auc').reset_index();sd=ss.overlay-ss.core;g.append({'arm':a,'median_auc_delta':q.auc_delta.median(),'min_auc_delta':q.auc_delta.min(),'positive_fraction':(q.auc_delta>0).mean(),'median_brier_delta':q.brier_delta.median(),'long_median_auc_delta':sd[ss.side.eq('long')].median(),'short_median_auc_delta':sd[ss.side.eq('short')].median(),'eligible':bool(q.auc_delta.median()>0 and (q.auc_delta>0).mean()>=.75 and q.auc_delta.min()>=-.02 and q.brier_delta.median()<=0 and sd[ss.side.eq('long')].median()>=0 and sd[ss.side.eq('short')].median()>=0)})
 aud=[]
 for (a,e,s),q in b[b.kind.eq('core')].groupby(['arm','era','side_name']):
  tr=b[(b.arm.eq(a))&(b.kind.eq('core'))&(b.side_name.eq(s))&b.era.ne(e)][['candidate_id','era']].drop_duplicates();te=q[['candidate_id']].drop_duplicates();aud.append({'arm':a,'outer_era':e,'side':s,'level':'outer','pre_rows':len(tr),'post_rows':len(cap(tr)),'pre_hash':ih(tr),'post_hash':ih(cap(tr)),'test_rows':len(te),'test_hash':ih(te)})
  for inn,iv in tr.groupby('era'):
   it=tr[tr.era.ne(inn)];aud.append({'arm':a,'outer_era':e,'side':s,'level':'inner','inner_era':inn,'pre_rows':len(it),'post_rows':len(cap(it)),'pre_hash':ih(it),'post_hash':ih(cap(it)),'test_rows':len(iv),'test_hash':ih(iv)})
 st=O.parent/('.'+O.name+'.tmp');st.mkdir();b.to_parquet(st/'bound_predictions.parquet',index=False);pool.to_csv(st/'true_pooled_metrics.csv',index=False);side.to_csv(st/'side_metrics.csv',index=False);piv.to_csv(st/'pooled_deltas.csv',index=False);pd.DataFrame(g).to_csv(st/'eligibility.csv',index=False);pd.DataFrame(aud).to_csv(st/'cohort_audit.csv',index=False)
 c={'schema':'nested_failure_overlay_v3','status':'SEALED_AUTHORITATIVE_NESTED_OVERLAY_NON_PROMOTION','supersedes_non_authoritative':[V1.name,V2.name],'v1_prediction_sha256':h(V1/'predictions.parquet'),'source_target_sha256':h(S/'materialized_targets.parquet'),'cap':CAP,'formula':'exact v1 bound predictions; audit reconstructs deterministic pre/post cap','decision_cadence':'1h','no_2026':True,'implementation_sha256':{str(Path(__file__).resolve()):h(Path(__file__))},'environment':{'python':sys.version,'platform':platform.platform(),'numpy':np.__version__,'pandas':pd.__version__,'sklearn':sklearn.__version__}};(st/'contract.json').write_text(json.dumps(c,indent=2)+'\n');fs=[p for p in st.iterdir() if p.is_file()];m={'schema':c['schema'],'status':c['status'],'contract':c,'outputs_sha256':{p.name:h(p) for p in fs}};(st/'manifest.json').write_text(json.dumps(m,indent=2)+'\n');(st/'manifest.sha256').write_text(f'{h(st/"manifest.json")}  manifest.json\n');os.replace(st,O);print(O)
if __name__=='__main__':run()
