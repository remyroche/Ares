#!/usr/bin/env python3
"""Non-overwriting correction for v1 support-composition tie precision."""
from __future__ import annotations
import argparse,hashlib,json,math,os,tempfile
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];V1=ROOT/'data_perp/artifacts/bounded_side_local_support_composition_20260730_v1';MAE=ROOT/'data_perp/artifacts/febapr2025_historical_auxiliary_oof_20260729_v2_timing_mae';FRACS=(.01,.05,.1,.2);Y='execution_net_ev_12h'
def h(p):
 d=hashlib.sha256()
 with Path(p).open('rb') as x:
  for b in iter(lambda:x.read(1<<20),b''):d.update(b)
 return d.hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2,sort_keys=True,default=str)+'\n')
def expected_precision(above_y,tie_y,need):
 """Random tie allocation precision, unlike mean outcome's sign."""
 a=np.asarray(above_y,float);t=np.asarray(tie_y,float);n=len(a)+int(need)
 return float(((a>0).sum()+int(need)*(t>0).mean())/n) if n else float('nan')
def select(x,col,f):
 n=max(1,math.ceil(len(x)*f));return x.sort_values([col,'candidate_id','__ts__','__symbol__','side_name'],ascending=[False,True,True,True,True],kind='mergesort').iloc[:n]
def bound(x,col,f):
 q=select(x,col,f);n=len(q);cut=float(q[col].iloc[-1]);above=x[x[col]>cut];tie=x[np.isclose(x[col].to_numpy(float),cut,rtol=0,atol=1e-14)];need=n-len(above);a=above[Y].to_numpy(float);t=tie[Y].to_numpy(float);exp_net=float((a.sum()+need*t.mean())/n*1e4)
 return {'score_kind':'mapped' if col=='mapped_score' else 'raw','top_fraction':f,'rows':n,'cutoff':cut,'above_cutoff_rows':len(above),'cutoff_tie_rows':len(tie),'cutoff_tie_fraction_of_book':len(tie)/n,'required_from_tie_rows':need,'random_tie_expected_net_bps':exp_net,'random_tie_expected_precision':expected_precision(a,t,need),'best_tie_precision':float(np.r_[a,np.sort(t)[-need:]].__gt__(0).mean()),'worst_tie_precision':float(np.r_[a,np.sort(t)[:need]].__gt__(0).mean())}
def adverse_proof(root):
 m=json.loads((root/'manifest.json').read_text());roles=['mae_before_meaningful_mfe_atr.p_hit','mae_before_meaningful_mfe_atr.if_hit','mae_before_meaningful_mfe_atr.if_no_hit'];proof=[]
 assert m['status']=='STRICT_SIDE_LOCAL_MARCH_APRIL_AUXILIARY_OOF_COMPLETE'
 for role in roles:
  folds=m['roles'][role]['folds'];assert set(folds)=={'2025-03','2025-04'}
  for month,report in folds.items():
   start=pd.Timestamp(f'{month}-01T00:00:00Z');assert pd.Timestamp(report['cutoff_utc'])==start
   for side in ('long','short'):
    for f in report['side'][side]['outer_fold']:
     assert f['fold_month']==month and pd.Timestamp(f['training_label_resolved_max'])<pd.Timestamp(f['valid_start']) and pd.Timestamp(f['valid_start'])==start and f['resolution_before_valid_start_assertion']
     proof.append({'role':role,'side':side,'month':month,'cutoff_utc':report['cutoff_utc'],'training_label_resolved_max':f['training_label_resolved_max'],'validation_start_utc':f['valid_start'],'strict_resolution_assertion':True,'available_at_decision':'prediction uses frozen PIT pre-entry features and is emitted for every outer validation decision row'})
 return {'status':'STRICT_OOF_ADVERSE_SEVERITY_PROVEN','source_manifest_status':m['status'],'source_manifest_sha256':h(root/'manifest.json'),'source_predictions_sha256':h(root/'oof_predictions.parquet'),'rows':proof}
def run(a):
 if a.output_dir.exists():raise FileExistsError(a.output_dir)
 x=pd.read_parquet(a.v1/'april_confirmation_predictions.parquet');x['__ts__']=pd.to_datetime(x['__ts__'],utc=True);old=pd.read_csv(a.v1/'tie_bounds.csv');rows=[]
 for col in ('raw_score','mapped_score'):
  rows += [bound(x,col,f) for f in FRACS]
 new=pd.DataFrame(rows);old_net=old[['score_kind','top_fraction','random_tie_expected_net_bps']].merge(new[['score_kind','top_fraction','random_tie_expected_net_bps']],on=['score_kind','top_fraction'],suffixes=('_old','_new'));assert np.allclose(old_net.random_tie_expected_net_bps_old,old_net.random_tie_expected_net_bps_new)
 proof=adverse_proof(a.mae);st=Path(tempfile.mkdtemp(prefix='.'+a.output_dir.name+'.',dir=a.output_dir.parent));new.to_csv(st/'corrected_tie_bounds.csv',index=False);write(st/'adverse_strict_oof_proof.json',proof);write(st/'v1_precision_invalidation.json',{'schema':'bounded_side_local_support_composition_v1_precision_invalidation_v1','status':'V1_PRECISION_FIELDS_NONAUTHORITATIVE_NET_UNCHANGED','invalidated_manifest_sha256':h(a.v1/'manifest.json'),'reason':'v1 expected precision incorrectly tested sign of expected net rather than expected positive-rate under random ties','unchanged':'expected net, control parity, selections, and promotion result','replacement':'corrected_tie_bounds.csv'})
 out={p.name:h(p) for p in st.iterdir() if p.is_file()};man={'schema':'bounded_side_local_support_composition_tie_correction_v2','status':'SEALED_CORRECTION_NO_REPLAY','sources':{'v1_manifest':h(a.v1/'manifest.json'),'v1_predictions':h(a.v1/'april_confirmation_predictions.parquet'),'adverse_manifest':h(a.mae/'manifest.json'),'adverse_predictions':h(a.mae/'oof_predictions.parquet')},'net_parity_assertion':True,'outputs_sha256':out,'runner_sha256':h(Path(__file__))};write(st/'manifest.json',man);(st/'manifest.sha256').write_text(h(st/'manifest.json')+'  manifest.json\n');os.replace(st,a.output_dir);return man
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--v1',type=Path,default=V1);p.add_argument('--mae',type=Path,default=MAE);p.add_argument('--output-dir',type=Path,required=True);print(json.dumps(run(p.parse_args()),indent=2))
