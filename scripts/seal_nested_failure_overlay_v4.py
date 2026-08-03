import hashlib,json,os,tempfile
from pathlib import Path
import pandas as pd
R=Path(__file__).resolve().parents[1];A=R/'data_perp/artifacts';V=[A/f'pre2026_nested_residual_context_failure_overlay_20260730_v{i}' for i in [1,2,3]];S=A/'pre2026_oof_model_failure_incremental_value_20260730_v3';O=A/'pre2026_nested_residual_context_failure_overlay_20260730_v4'
def h(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def run():
 if O.exists():raise RuntimeError(O)
 for p in V+[S]:
  if h(p/'manifest.json')!=(p/'manifest.sha256').read_text().split()[0]:raise RuntimeError('unsealed')
 z=pd.read_parquet(V[0]/'predictions.parquet');src=pd.read_parquet(S/'materialized_targets.parquet',columns=['candidate_id','__ts__','execution_label_end_utc']);src.__ts__=pd.to_datetime(src.__ts__,utc=True);src.execution_label_end_utc=pd.to_datetime(src.execution_label_end_utc,utc=True);b=z.merge(src,on='candidate_id',validate='many_to_one')
 assert b.__ts__.dt.minute.eq(0).all() and b.__ts__.dt.second.eq(0).all() and b.execution_label_end_utc.gt(b.__ts__).all() and b.execution_label_end_utc.lt(pd.Timestamp('2026-01-01',tz='UTC')).all()
 eq=[]
 for k,q in b.groupby(['arm','era','side_name']):
  c=q[q.kind.eq('core')].candidate_id;d=q[q.kind.eq('overlay')].candidate_id;eq.append({'arm':k[0],'era':k[1],'side':k[2],'core_rows':len(c),'overlay_rows':len(d),'equal':set(c)==set(d),'unique_core':c.is_unique,'unique_overlay':d.is_unique})
 eq=pd.DataFrame(eq);assert eq[['equal','unique_core','unique_overlay']].all().all()
 old=pd.read_csv(V[0]/'cohort_audit.csv');new=pd.read_csv(V[2]/'cohort_audit.csv');o=old[['arm','outer_era','side','pre_cap_train_rows','train_hash','test_rows','test_hash']];n=new[new.level.eq('outer')][['arm','outer_era','side','pre_rows','pre_hash','test_rows','test_hash']];m=o.merge(n,on=['arm','outer_era','side']);assert (m.pre_cap_train_rows==m.pre_rows).all() and (m.train_hash==m.pre_hash).all() and (m.test_rows_x==m.test_rows_y).all() and (m.test_hash_x==m.test_hash_y).all()
 p=pd.read_csv(V[2]/'true_pooled_metrics.csv');w=p.pivot(index=['arm','era'],columns='kind',values='high_low_ev').reset_index();w['high_low_ev_delta']=w.overlay-w.core
 st=O.parent/('.'+O.name+'.tmp');st.mkdir();eq.to_csv(st/'candidate_set_assertions.csv',index=False);m.to_csv(st/'v1_outer_identity_assertions.csv',index=False);w.to_csv(st/'high_low_ev_deltas.csv',index=False);c={'schema':'nested_failure_overlay_v4_audit','status':'SEALED_AUTHORITATIVE_AUDIT_SUPPLEMENT_NON_PROMOTION','supersedes':['v1','v2','v3 lineage only; v3 metrics unchanged'],'assertions':{'v1_outer_rows':len(m),'core_overlay_cohorts':len(eq),'all_passed':True,'no_2026':True}};(st/'contract.json').write_text(json.dumps(c,indent=2)+'\n');fs=[x for x in st.iterdir() if x.is_file()];inp={str((x/'manifest.json').resolve()):h(x/'manifest.json') for x in V+[S]};inp[str((V[0]/'predictions.parquet').resolve())]=h(V[0]/'predictions.parquet');inp[str((S/'materialized_targets.parquet').resolve())]=h(S/'materialized_targets.parquet');man={'schema':c['schema'],'status':c['status'],'contract':c,'inputs_sha256':inp,'outputs_sha256':{x.name:h(x) for x in fs}};(st/'manifest.json').write_text(json.dumps(man,indent=2)+'\n');(st/'manifest.sha256').write_text(f'{h(st/"manifest.json")}  manifest.json\n');os.replace(st,O);print(O)
if __name__=='__main__':run()
