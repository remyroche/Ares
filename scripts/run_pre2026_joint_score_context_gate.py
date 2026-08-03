#!/usr/bin/env python3
"""One implementation for matched score-only/context OOF gates; no 2026 input."""
from __future__ import annotations
import hashlib,importlib.util,json,os,shutil,tempfile
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];ART=ROOT/'data_perp/artifacts';SRC=ART/'pre2026_oof_model_failure_incremental_value_20260730_v3';OUT=ART/'pre2026_joint_score_context_incremental_gate_20260730_v2'
spec=importlib.util.spec_from_file_location('impl',ROOT/'scripts/run_pre2026_model_failure_incremental_value.py');impl=importlib.util.module_from_spec(spec);spec.loader.exec_module(impl)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):
 q=Path(p).with_name('.'+Path(p).name+'.partial');q.write_text(json.dumps(x,indent=2,sort_keys=True,default=str)+'\n');os.replace(q,p)
def run(output=OUT):
 output=Path(output)
 if output.exists():raise RuntimeError(output)
 if sha(SRC/'manifest.json')!=(SRC/'manifest.sha256').read_text().split()[0]:raise RuntimeError('unsealed source')
 arms=impl.ARMS;targets={'incremental_selected_book_utility':('incremental_selected_book_utility',False),'selected_net_failure':('residual_selected_net_failure',True)}
 common=['candidate_id','__ts__','__symbol__','side_name','source','era','execution_label_end_utc','execution_net_ev_12h','base_selected_global_top10','residual_selected_global_top10','incremental_selected_book_utility','residual_selected_net_failure','bocpd_regime_available','lgbm_transition_available','trajectory_available']
 stage=Path(tempfile.mkdtemp(dir=output.parent,prefix='.'+output.name+'.'));mrows=[];audit=[]
 try:
  for arm,fs in arms.items():
   x=pd.read_parquet(SRC/'materialized_targets.parquet',columns=list(dict.fromkeys(common+fs)));x.__ts__=pd.to_datetime(x.__ts__,utc=True);x.execution_label_end_utc=pd.to_datetime(x.execution_label_end_utc,utc=True)
   if x.candidate_id.duplicated().any() or x.__ts__.dt.minute.ne(0).any() or x.__ts__.dt.second.ne(0).any() or x.execution_label_end_utc.le(x.__ts__).any() or x.execution_label_end_utc.ge(pd.Timestamp('2026-01-01',tz='UTC')).any():raise RuntimeError('fail closed: identity/cadence/label boundary')
   ok=x[fs].notna().all(axis=1)
   if arm in ['regime','combined']:ok&=x.bocpd_regime_available.fillna(False)
   if arm in ['transition','combined']:ok&=x.lgbm_transition_available.fillna(False)
   if arm in ['trajectory','combined']:ok&=x.trajectory_available.fillna(False)
   x=x[ok]
   for name,(target,cl) in targets.items():
    u=x.dropna(subset=[target]);u=u if name.startswith('incremental') else u[u.residual_selected_global_top10]
    for kind,features in [('score_only_'+arm,impl.CORE),(arm,fs)]:
     out=[]
     for era,te0 in u.groupby('era',sort=True):
      tr0=u[u.era.ne(era)]
      for side,te in te0.groupby('side_name',sort=True):
       tr=tr0[tr0.side_name.eq(side)]
       if len(tr)<500 or len(te)<50 or (cl and tr[target].nunique()<2):raise RuntimeError('fail closed: minimum support')
       pred,std=impl.model_predictions(tr,te,features,target,cl);idh=hashlib.sha256('|'.join(te.candidate_id.sort_values()).encode()).hexdigest();trh=hashlib.sha256('|'.join(tr.candidate_id.sort_values()).encode()).hexdigest()
       out.append(te[['candidate_id','__ts__','side_name','era','execution_net_ev_12h',target]].rename(columns={target:'actual_target'}).assign(arm=kind,target=name,prediction=pred,prediction_std=std));audit.append({'arm':arm,'kind':kind,'target':name,'era':era,'side':side,'pre_cap_train_rows':len(tr),'test_rows':len(te),'pre_cap_train_candidate_sha256':trh,'test_candidate_sha256':idh,'features':'|'.join(features)})
     p=pd.concat(out,ignore_index=True);p.to_parquet(stage/f'oof_{kind}_{name}.parquet',index=False);mrows.append(impl.metric_rows(p));del p
   del x
  met=pd.concat(mrows,ignore_index=True);met.to_csv(stage/'fold_metrics.csv',index=False);ctx=[]
  for arm in arms:
   c=met[(met.arm==('score_only_'+arm))&(met.scope=='pooled')][['target','era','rank_metric']].rename(columns={'rank_metric':'score_only_metric'});q=met[(met.arm==arm)&(met.scope=='pooled')][['target','era','rank_metric']].merge(c,on=['target','era']);ctx.append(q.assign(arm=arm,delta=q.rank_metric-q.score_only_metric))
  ctx=pd.concat(ctx,ignore_index=True);ctx.to_csv(stage/'matched_fold_deltas.csv',index=False);g=[]
  for (target,arm),z in ctx.groupby(['target','arm']):
   d=z.delta;g.append({'target':target,'arm':arm,'matched_eras':len(d),'median_delta':d.median(),'min_delta':d.min(),'positive_fraction':(d>0).mean(),'eligible':bool(len(d)>=6 and d.median()>0 and (d>0).mean()>=.75 and d.min()>=-.02)})
  pd.DataFrame(g).to_csv(stage/'eligibility.csv',index=False);aa=pd.DataFrame(audit);eq=aa.groupby(['arm','target','era','side']).agg(kinds=('kind','nunique'),test_hashes=('test_candidate_sha256','nunique'),test_rows=('test_rows','nunique'),train_hashes=('pre_cap_train_candidate_sha256','nunique'),train_rows=('pre_cap_train_rows','nunique')).reset_index();eq['candidate_set_equal']=(eq.kinds==2)&(eq.test_hashes==1)&(eq.test_rows==1)&(eq.train_hashes==1)&(eq.train_rows==1)
  if not eq.candidate_set_equal.all():raise RuntimeError('fail closed: arm-matched candidate inequality')
  aa.to_csv(stage/'fold_audit.csv',index=False);eq.to_csv(stage/'candidate_set_equality.csv',index=False)
  c={'schema':'pre2026_joint_score_context_incremental_gate_v2','status':'SEALED_PRE2026_ARM_MATCHED_IMPLEMENTATION_GATE_NON_PROMOTION','promotion_eligible':False,'decision_cadence':'1h','exact_replay_bar_cadence':'1m_labels_only','implementation_sha256':{str(Path(__file__).resolve()):sha(Path(__file__)),str((ROOT/'scripts/run_pre2026_model_failure_incremental_value.py').resolve()):sha(ROOT/'scripts/run_pre2026_model_failure_incremental_value.py')},'config':{'context_arms':{k:v for k,v in arms.items()},'controls':'CORE-only per context arm, after identical arm availability mask','targets':list(targets),'side_local':True,'leave_era_out':True,'candidate_hash_train_cap':150000,'three_subsample_predictions':True,'gate':'median delta>0, >=75% positive, min>=-.02, >=6 matched eras'},'scope':'pre-2026 v3 materialized target ledger only; no 2026 input'};dump(stage/'contract.json',c);files=[p for p in stage.iterdir() if p.is_file()];man={'schema':c['schema'],'status':c['status'],'promotion_eligible':False,'contract':c,'inputs_sha256':{str((SRC/'manifest.json').resolve()):sha(SRC/'manifest.json'),str((SRC/'materialized_targets.parquet').resolve()):sha(SRC/'materialized_targets.parquet')},'outputs_sha256':{p.name:sha(p) for p in files}};dump(stage/'manifest.json',man);(stage/'manifest.sha256').write_text(f'{sha(stage/"manifest.json")}  manifest.json\n');os.replace(stage,output);return output
 except Exception:shutil.rmtree(stage,ignore_errors=True);raise
if __name__=='__main__':print(run())
