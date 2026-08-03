#!/usr/bin/env python3
"""Seal score-only OOF control and context-versus-control gate; no 2026 input."""
from __future__ import annotations
import hashlib,importlib.util,json,os,shutil,tempfile
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[1];ART=ROOT/'data_perp/artifacts';V3=ART/'pre2026_oof_model_failure_incremental_value_20260730_v3';V4=ART/'pre2026_oof_model_failure_incremental_value_20260730_v4';OUT=ART/'pre2026_oof_model_failure_incremental_value_score_control_20260730_v2'
spec=importlib.util.spec_from_file_location('fv',ROOT/'scripts/run_pre2026_model_failure_incremental_value.py');fv=importlib.util.module_from_spec(spec);spec.loader.exec_module(fv)
CORE=fv.CORE
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):
 q=Path(p).with_name('.'+Path(p).name+'.partial');q.write_text(json.dumps(x,indent=2,sort_keys=True,default=str)+'\n');os.replace(q,p)
def run(output=OUT):
 output=Path(output)
 if output.exists():raise RuntimeError(output)
 for p in [V3,V4]:
  if sha(p/'manifest.json') != (p/'manifest.sha256').read_text().split()[0]:raise RuntimeError('unsealed prerequisite')
 cols=['candidate_id','__ts__','__symbol__','side_name','source','era','execution_label_end_utc','execution_net_ev_12h','base_selected_global_top10','residual_selected_global_top10','incremental_selected_book_utility','residual_selected_net_failure']+CORE
 x=pd.read_parquet(V3/'materialized_targets.parquet',columns=cols);x.__ts__=pd.to_datetime(x.__ts__,utc=True);x.execution_label_end_utc=pd.to_datetime(x.execution_label_end_utc,utc=True)
 if x.__ts__.dt.minute.ne(0).any() or x.execution_label_end_utc.ge(pd.Timestamp('2026-01-01',tz='UTC')).any():raise RuntimeError('fail closed cadence')
 targets={'incremental_selected_book_utility':('incremental_selected_book_utility',False),'selected_net_failure':('residual_selected_net_failure',True)}
 stage=Path(tempfile.mkdtemp(dir=output.parent,prefix='.'+output.name+'.'));parts=[];aud=[]
 try:
  for name,(target,classification) in targets.items():
   use=x.dropna(subset=[target]);use=use if name=='incremental_selected_book_utility' else use[use.residual_selected_global_top10]
   out=[]
   for era,test in use.groupby('era',sort=True):
    train=use[use.era.ne(era)]
    for side,te in test.groupby('side_name',sort=True):
     tr=train[train.side_name.eq(side)]
     if len(tr)<500 or len(te)<50 or (classification and tr[target].nunique()<2):raise RuntimeError(f'unsupported {name}/{era}/{side}')
     pred,std=fv.model_predictions(tr,te,CORE,target,classification)
     out.append(te[['candidate_id','__ts__','__symbol__','side_name','source','era','execution_net_ev_12h','base_selected_global_top10','residual_selected_global_top10',target]].rename(columns={target:'actual_target'}).assign(target=name,arm='score_only_control',prediction=pred,prediction_std=std))
     aud.append({'target':name,'held_era':era,'side_name':side,'train_rows':len(tr),'test_rows':len(te),'features':'|'.join(CORE),'classification':classification})
   p=pd.concat(out,ignore_index=True);p.to_parquet(stage/f'leave_era_oof_{name}_score_only_control.parquet',index=False);parts.append(fv.metric_rows(p))
  metrics=pd.concat(parts,ignore_index=True);metrics.to_csv(stage/'score_control_fold_metrics.csv',index=False)
  cm=pd.read_csv(V3/'fold_metrics.csv');rows=[]
  for target in targets:
   b=metrics[(metrics.target.eq(target))&metrics.scope.eq('pooled')][['era','rank_metric']].rename(columns={'rank_metric':'control_rank_metric'})
   for arm,g in cm[(cm.target.eq(target))&cm.scope.eq('pooled')].groupby('arm',sort=True):
    d=g[['era','rank_metric']].merge(b,on='era',how='inner');d['incremental_rank_metric']=d.rank_metric-d.control_rank_metric
    rows.extend(d.assign(target=target,arm=arm).to_dict('records'))
  delta=pd.DataFrame(rows);delta.to_csv(stage/'context_vs_score_control_fold_deltas.csv',index=False);gate=[]
  for (target,arm),g in delta.groupby(['target','arm'],sort=True):
   v=g.incremental_rank_metric;gate.append({'target':target,'arm':arm,'matched_eras':len(v),'median_incremental_rank_metric':v.median(),'min_incremental_rank_metric':v.min(),'positive_era_fraction':(v>0).mean(),'context_incremental_gate':bool(len(v)>=6 and v.median()>0 and (v>0).mean()>=.75 and v.min()>=-.02)})
  pd.DataFrame(gate).to_csv(stage/'context_incremental_gate.csv',index=False);pd.DataFrame(aud).to_csv(stage/'fold_audit.csv',index=False)
  c={'schema':'pre2026_oof_score_only_incremental_control_v2','status':'SEALED_PRE2026_SCORE_ONLY_CONTEXT_INCREMENTAL_GATE_NON_PROMOTION','promotion_eligible':False,'decision_cadence':'1h','exact_replay_bar_cadence':'1m_labels_only','scope':'uses sealed v3/v4 rows only; no 2026 candidate/economics input','control':'same side-local leave-era low-capacity architecture and fixed 150000 candidate-hash cap as v3, but CORE scores only','gate':'context head may enter frozen-2026 preregistration only when its matched-era rank metric exceeds score-only with median >0, >=75% positive eras, min >=-.02 and >=6 matched eras','targets':list(targets),'severity':'excluded','implementation_sha256':{str(Path(__file__).resolve()):sha(Path(__file__)),str((ROOT/'scripts/run_pre2026_model_failure_incremental_value.py').resolve()):sha(ROOT/'scripts/run_pre2026_model_failure_incremental_value.py')}}
  dump(stage/'contract.json',c);files=[p for p in stage.iterdir() if p.is_file()];m={'schema':c['schema'],'status':c['status'],'promotion_eligible':False,'contract':c,'inputs_sha256':{str((V3/'manifest.json').resolve()):sha(V3/'manifest.json'),str((V4/'manifest.json').resolve()):sha(V4/'manifest.json')},'outputs_sha256':{p.name:sha(p) for p in files}};dump(stage/'manifest.json',m);(stage/'manifest.sha256').write_text(f'{sha(stage/"manifest.json")}  manifest.json\n');os.replace(stage,output);return output
 except Exception:shutil.rmtree(stage,ignore_errors=True);raise
if __name__=='__main__':print(run())
