from __future__ import annotations
import hashlib,json,os,platform,sys,tempfile
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss
import sklearn
ROOT=Path(__file__).resolve().parents[1];A=ROOT/'data_perp/artifacts';V=A/'pre2026_nested_residual_context_failure_overlay_20260730_v1';S=A/'pre2026_oof_model_failure_incremental_value_20260730_v3';O=A/'pre2026_nested_residual_context_failure_overlay_20260730_v2';CAP=150000
def h(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ih(x):return hashlib.sha256('|'.join(x.candidate_id.sort_values()).encode()).hexdigest()
def cp(x):return x.iloc[np.argsort(pd.util.hash_pandas_object(x.candidate_id,index=False).to_numpy(),kind='stable')[:CAP]] if len(x)>CAP else x
def run():
 if O.exists():raise RuntimeError(O)
 for p in [V,S]:
  if h(p/'manifest.json')!=(p/'manifest.sha256').read_text().split()[0]:raise RuntimeError('unsealed')
 z=pd.read_parquet(V/'predictions.parquet');src=pd.read_parquet(S/'materialized_targets.parquet',columns=['candidate_id','__ts__','execution_label_end_utc']);src.__ts__=pd.to_datetime(src.__ts__,utc=True);src.execution_label_end_utc=pd.to_datetime(src.execution_label_end_utc,utc=True);z=z.merge(src,on='candidate_id',validate='many_to_one')
 if z.__ts__.dt.minute.ne(0).any() or z.execution_label_end_utc.ge(pd.Timestamp('2026-01-01',tz='UTC')).any():raise RuntimeError('cadence')
 rows=[]
 for (a,k,e),q in z.groupby(['arm','kind','era']):rows.append({'arm':a,'kind':k,'era':e,'auc':roc_auc_score(q.y,q.p),'ap':average_precision_score(q.y,q.p),'brier':brier_score_loss(q.y,q.p),'high_low_ev':q.loc[q.p>=q.p.quantile(.9),'execution_net_ev_12h'].mean()-q.loc[q.p<=q.p.quantile(.1),'execution_net_ev_12h'].mean()})
 m=pd.DataFrame(rows);d=m.pivot(index=['arm','era'],columns='kind',values=['auc','brier']).reset_index();d['auc_delta']=d[('auc','overlay')]-d[('auc','core')];d['brier_delta']=d[('brier','overlay')]-d[('brier','core')]
 st=O.parent/('.'+O.name+'.tmp');st.mkdir();z.to_parquet(st/'bound_predictions.parquet',index=False);m.to_csv(st/'true_pooled_metrics.csv',index=False);d.to_csv(st/'true_pooled_deltas.csv',index=False)
 c={'schema':'nested_failure_overlay_v2','status':'SEALED_AUTHORITATIVE_RECOMPUTED_METRICS_NON_PROMOTION','binds_v1_prediction_sha256':h(V/'predictions.parquet'),'implementation_sha256':{str(Path(__file__).resolve()):h(Path(__file__))},'environment':{'python':sys.version,'platform':platform.platform(),'numpy':np.__version__,'pandas':pd.__version__,'sklearn':sklearn.__version__},'cadence':'1h candidate rows; 1m nested labels only','no_2026':True};(st/'contract.json').write_text(json.dumps(c,indent=2)+'\n');fs=[p for p in st.iterdir() if p.is_file()];man={'schema':c['schema'],'status':c['status'],'contract':c,'outputs_sha256':{p.name:h(p) for p in fs}};(st/'manifest.json').write_text(json.dumps(man,indent=2)+'\n');(st/'manifest.sha256').write_text(f'{h(st/"manifest.json")}  manifest.json\n');os.replace(st,O);print(O)
if __name__=='__main__':run()
