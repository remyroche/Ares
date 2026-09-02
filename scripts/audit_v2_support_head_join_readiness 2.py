#!/usr/bin/env python3
import argparse,hashlib,json,os,tempfile
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];V2=ROOT/'data_perp/artifacts/bounded_direct_tail_repair_20260730_v2';AUX=ROOT/'data_perp/artifacts/febapr2025_historical_auxiliary_oof_20260729_v2/oof_predictions.parquet';OUT=ROOT/'data_perp/artifacts/bounded_direct_tail_repair_v2_support_head_readiness_20260730_v1';ID=['candidate_id','side_name','__symbol__','__ts__']
def h(p):
 d=hashlib.sha256();d.update(p.read_bytes());return d.hexdigest()
def run(a):
 if a.output_dir.exists():raise FileExistsError(a.output_dir)
 p=pd.read_parquet(a.v2/'confirmation_predictions.parquet');q=pd.read_parquet(a.aux);base=set(map(tuple,p[ID].drop_duplicates().to_numpy()));peak=set(map(tuple,q[ID].to_numpy()));j=len(base&peak)
 rows=pd.DataFrame([{'head':'meaningful_mfe_event_plus_peak_contribution','status':'JOINABLE_STRICT_OOF','v2_rows':len(base),'head_rows':len(peak),'exact_identity_rows':j,'availability':'__label_end_ts__ exists; predictions are historical OOF'}, {'head':'future_slope','status':'FAIL_CLOSED_MISSING_STRICT_OOF_PREDICTION_LEDGER','v2_rows':len(base),'head_rows':0,'exact_identity_rows':0,'availability':'no future_slope prediction field/ledger in established FebApr auxiliary OOF artifact'}])
 st=Path(tempfile.mkdtemp(prefix='.'+a.output_dir.name+'.',dir=a.output_dir.parent));rows.to_csv(st/'readiness.csv',index=False);m={'schema':'v2_support_head_join_readiness_v1','status':'FAIL_CLOSED_FUTURE_SLOPE_UNAVAILABLE_NO_ABLATION','contract':'no refit/substitution; peak requires p_hit times conditional magnitude','inputs':{'v2':h(a.v2/'manifest.json'),'aux':h(a.aux)},'outputs':{'readiness':h(st/'readiness.csv')}};(st/'manifest.json').write_text(json.dumps(m,indent=2)+'\n');(st/'manifest.sha256').write_text(h(st/'manifest.json')+'  manifest.json\n');os.replace(st,a.output_dir);return m
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--v2',type=Path,default=V2);p.add_argument('--aux',type=Path,default=AUX);p.add_argument('--output-dir',type=Path,required=True);print(json.dumps(run(p.parse_args()),indent=2))
