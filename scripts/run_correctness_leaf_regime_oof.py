#!/usr/bin/env python3
"""Nested chronological OOF correctness-leaf regime representation runner."""
from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Any
import numpy as np, pandas as pd, lightgbm as lgb
from pyarrow.parquet import ParquetFile
from scipy.stats import spearmanr
from sklearn.isotonic import IsotonicRegression

ROOT=Path(__file__).resolve().parents[1]
import sys
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from extreme_price_movements.performance_regimes.correctness_leaf_targets import (TARGET_FAMILIES,aggregate_correctness_periods,binary_surprise,fit_correctness_scale,probability_entropy,select_top_base_per_timestamp,soft_correctness,soft_negative_surprise,soft_positive_surprise)
from extreme_price_movements.performance_regimes.correctness_leaf_regimes import LeafRule,aggregate_membership,cluster_rules,cluster_state_dynamics,membership_dynamics,soft_rule_membership

INPUT=ROOT/'data_perp/artifacts/correctness_leaf_regime_input_20260803_v1/input.parquet'
AUDIT=ROOT/'data_perp/artifacts/correctness_leaf_regime_input_20260803_v1/feature_availability.parquet'
OUT=ROOT/'data_perp/artifacts/correctness_leaf_regime_oof_20260803_v1'
TARGETS={'row':None,'period12h':12,'period24h':24,'period72h':72}
MIN_DAYS=175.; TOP=.05; MAX_FEATURES=None; MAX_RULES=80; MAX_CLUSTERS=12
EPS=1e-8

def _paths(node:dict,names:list[str],path=()):
 if 'leaf_index' in node: yield int(node['leaf_index']),path,float(node.get('leaf_value',0.)); return
 f=names[int(node['split_feature'])];t=float(node['threshold'])
 yield from _paths(node['left_child'],names,path+((f,-1,t),))
 yield from _paths(node['right_child'],names,path+((f,1,t),))

def _screen(train:pd.DataFrame,fields:list[str],y:np.ndarray)->list[str]:
 out=[]
 for f in fields:
  x=pd.to_numeric(train[f],errors='coerce').to_numpy(float);ok=np.isfinite(x)&np.isfinite(y)
  if ok.sum()>=200 and np.std(x[ok])>1e-10:
   c=spearmanr(x[ok],y[ok]).statistic
   if np.isfinite(c):out.append((abs(float(c)),f))
 # The regime learner receives the complete causal meta contract.  The
 # univariate score only supplies a deterministic order for the model dump;
 # it is not a feature-selection gate.
 ranked=[f for _,f in sorted(out,key=lambda z:(-z[0],z[1]))]
 return ranked if MAX_FEATURES is None else ranked[:MAX_FEATURES]

def _target(frame:pd.DataFrame,train:pd.DataFrame,horizon:int|None,target_family:str='correctness')->pd.DataFrame:
 """Build one strictly available leaf-discovery target.

 ``positive``, ``negative`` and ``surprise`` depend on realised residual and
 therefore retain the exact realised-label availability. ``entropy`` is the
 normalized entropy of the frozen OOF R3 simplex, hence known at decision
 time; it is deliberately not delayed to the realised-path timestamp.
 """
 if target_family not in TARGET_FAMILIES: raise ValueError(f'unknown target family: {target_family}')
 out=frame.copy(); out['residual_bps']=out.net_bps.to_numpy(float)-out.prequential_base_expected_net_bps.to_numpy(float); train=train.copy();train['residual_bps']=train.net_bps.to_numpy(float)-train.prequential_base_expected_net_bps.to_numpy(float)
 if target_family=='correctness':
  out['leaf_target']=np.nan
  for side in ('long','short'):
   value=train.loc[train.side_name.eq(side),'residual_bps']
   if len(value)<100: continue
   scale=fit_correctness_scale(value); m=out.side_name.eq(side)
   out.loc[m,'leaf_target']=soft_correctness(out.loc[m,'residual_bps'],scale)
  available=out.label_available_ts
 elif target_family=='positive':
  out['leaf_target']=soft_positive_surprise(out.residual_bps)
  available=out.label_available_ts
 elif target_family=='negative':
  out['leaf_target']=soft_negative_surprise(out.residual_bps)
  available=out.label_available_ts
 elif target_family=='surprise':
  out['leaf_target']=binary_surprise(out.residual_bps)
  available=out.label_available_ts
 else:
  probability_columns=['r3_p_adverse','r3_p_weak','r3_p_clear']
  missing=[col for col in probability_columns if col not in out]
  if missing: raise ValueError(f'entropy target requires frozen base simplex columns: {missing}')
  out['leaf_target']=probability_entropy(out[probability_columns].to_numpy(float))
  available=out.__ts__
 out['__leaf_target_available_ts__']=pd.to_datetime(available,utc=True)
 if horizon is None:
  out['target_value']=out.leaf_target;out['target_available_ts']=out.__leaf_target_available_ts__;return out
 out=aggregate_correctness_periods(out,target_column='leaf_target',horizon_hours=horizon,label_available_column='__leaf_target_available_ts__')
 out['target_value']=out.period_correctness_target;out['target_available_ts']=out.period_label_available_ts;return out

def _rules(model,fields,reference,quality):
 dump=model.booster_.dump_model();rules=[];memberships={}
 for tree in dump['tree_info']:
  for lid,conditions,value in _paths(tree['tree_structure'],fields):
   if not conditions:continue
   rule=LeafRule(f't{tree["tree_index"]}_l{lid}',tuple(conditions),value,max(abs(value),1e-4))
   m=soft_rule_membership(reference,rule);support=(m>=.6).mean()
   if support>=.01:rules.append((support*abs(value),rule));memberships[rule.rule_id]=m
 rules=[r for _,r in sorted(rules,key=lambda z:(-z[0],z[1].rule_id))[:MAX_RULES]]
 return rules,{r.rule_id:memberships[r.rule_id] for r in rules}

def _cluster_signature(members):
 """Stable, human-readable mechanism key for cross-fold stability checks."""
 parts=[]
 for rule in members:
  # ``+1`` is the right child / ``feature >= threshold`` and ``-1`` is the
  # left child / ``feature < threshold``.  This is presentation metadata
  # only, but its direction must agree with the executable rule dump.
  parts.extend(f'{field}:{"hi" if direction > 0 else "lo"}' for field,direction,_ in rule.conditions)
 return '|'.join(sorted(set(parts)))

def _episode_count(values, timestamps, threshold=.60):
 active=pd.Series(np.asarray(values)>=threshold,index=pd.to_datetime(timestamps,utc=True)).groupby(level=0).mean().ge(.5)
 # The candidate timestamps are regular bars; a gap begins a new independent episode.
 return int((active & ~active.shift(fill_value=False)).sum())

def _represent(test,rules,reference_memberships,side,target_name,fold,minimum_similarity: float=.70):
 clusters,sim=cluster_rules(rules,reference_memberships,minimum_similarity=minimum_similarity)
 rows=[];lineage=[];result=test[['candidate_id','__ts__','side_name']].copy();used=[];canonical_memberships=[];represented=0
 for rank,cluster in enumerate(clusters):
  if len(cluster)<2:continue
  members=[next(r for r in rules if r.rule_id==rid) for rid in cluster]
  signature=_cluster_signature(members)
  values=np.vstack([soft_rule_membership(test,r) for r in members]);weights=[r.weight for r in members]
  effect=float(np.average([r.economic_effect for r in members],weights=np.maximum([r.weight for r in members],EPS)))
  ref_values=np.vstack([reference_memberships[r.rule_id] for r in members])
  ref_membership=aggregate_membership(ref_values,[r.weight for r in members],mode='G1_weighted_geometric')
  cluster_pairs=sim.loc[(sim.left.isin(cluster))&(sim.right.isin(cluster)), 'total']
  stability=float(cluster_pairs.mean()) if len(cluster_pairs) else 1.0
  support=float((ref_membership>=.60).mean())
  for mode in ('G0_geometric','G1_weighted_geometric','G2_generalized_pminus2','G3_softmin'):
   name=f'leafreg__{target_name}__f{fold}__s{side}__c{rank:02d}__{mode}'
   result[name]=aggregate_membership(values,weights,mode=mode);used.append(name)
   lineage.append({'target':target_name,'fold':fold,'side_name':side,'feature':name,'mode':mode,'cluster':rank,'cluster_signature':signature,'active_share':float((result[name]>=.60).mean()),'episodes':_episode_count(result[name],result['__ts__'])})
  # The weighted geometric activation is the canonical contribution surface.
  # The other three modes remain independent candidates for MDA, while these
  # signed and trust outputs expose *why* the cluster is active.
  canonical=f'leafreg__{target_name}__f{fold}__s{side}__c{rank:02d}__G1_weighted_geometric'
  canonical_memberships.append(canonical)
  base=f'leafreg__{target_name}__f{fold}__s{side}__c{rank:02d}'
  result[f'{base}__signed_contribution']=result[canonical].to_numpy(float)*effect
  result[f'{base}__positive_contribution']=np.maximum(result[f'{base}__signed_contribution'].to_numpy(float),0.)
  result[f'{base}__negative_contribution']=np.minimum(result[f'{base}__signed_contribution'].to_numpy(float),0.)
  result[f'{base}__absolute_contribution']=np.abs(result[f'{base}__signed_contribution'].to_numpy(float))
  result[f'{base}__historical_correctness_effect']=np.float32(effect)
  result[f'{base}__historical_support']=np.float32(support)
  result[f'{base}__structural_stability']=np.float32(stability)
  result[f'{base}__instability']=np.float32(1.0-stability)
  result[f'{base}__rule_count']=np.float32(len(members))
  used.extend([f'{base}__signed_contribution',f'{base}__positive_contribution',f'{base}__negative_contribution',f'{base}__absolute_contribution',f'{base}__historical_correctness_effect',f'{base}__historical_support',f'{base}__structural_stability',f'{base}__instability',f'{base}__rule_count'])
  rows += [{'target':target_name,'fold':fold,'side_name':side,'cluster':rank,'rule_id':r.rule_id,'economic_effect':r.economic_effect,'cluster_size':len(cluster),'cluster_signature':signature,'conditions_json':json.dumps(r.conditions)} for r in members]
  represented+=1
  if represented>=MAX_CLUSTERS:break
 if not used:return result,pd.DataFrame(rows),sim,[],pd.DataFrame(lineage)
 # Convert canonical contributions into relative cluster shares.  This must
 # happen before timestamp aggregation so a candidate receives the share of
 # the explanation actually active on that row.
 abs_columns=[c for c in used if c.endswith('__absolute_contribution')]
 abs_mass=result[abs_columns].sum(axis=1).to_numpy(float) if abs_columns else np.zeros(len(result))
 for column in abs_columns:
  share_column=f'{column[:-len("__absolute_contribution")]}__total_contribution_share'
  result[share_column]=np.divide(result[column].to_numpy(float),abs_mass,out=np.zeros(len(result)),where=abs_mass>EPS).astype(np.float32)
  used.append(share_column)
 # Dynamics belong to the canonical weighted posterior surface.  Calculating
 # velocity/age for all aggregation alternatives and static trust constants
 # creates thousands of collinear candidates without adding information.
 state=result.groupby(['side_name','__ts__'],observed=True)[canonical_memberships].mean().reset_index();dyn=membership_dynamics(state,canonical_memberships,group_columns=('side_name',))
 # ``membership_dynamics`` intentionally emits generic aggregate names.  A
 # representation is fitted separately for every target/fold/side, therefore
 # those aggregates must be namespaced before joining more than one surface.
 prefix=f'leafreg__{target_name}__f{fold}__s{side}'
 dyn=dyn.rename(columns={'activation_mass':f'{prefix}__activation_mass','activation_entropy':f'{prefix}__activation_entropy'})
 dynamic=[name for name in dyn.columns if name not in {'side_name','__ts__',*canonical_memberships}]
 result=result.merge(dyn.drop(columns=canonical_memberships),on=['side_name','__ts__'],how='left',validate='many_to_one')
 state_surface=cluster_state_dynamics(result[['candidate_id','__ts__','side_name',*canonical_memberships]],canonical_memberships,group_columns=('side_name',))
 # The relative state surface is useful; the arbitrary dominant-cluster ID is
 # not portable and must never be treated as ordinal by the meta learner.
 state_outputs=[c for c in state_surface.columns if c.startswith('cluster_state_') and not c.endswith('_dominant_id')]
 result=result.merge(state_surface[['candidate_id',*state_outputs]],on='candidate_id',how='left',validate='one_to_one')
 # Posterior, velocity, acceleration, smoothing, active-age and aggregate
 # concentration are one candidate family.  The nested MDA gate—not this
 # generator—decides which are incremental for the residual meta model.
 return result,pd.DataFrame(rows),sim,[*used,*dynamic,*state_outputs],pd.DataFrame(lineage)

def run(out:Path=OUT, input_path:Path=INPUT, audit_path:Path=AUDIT, start_test:str|None=None, history_days:float|None=None):
 out.mkdir(parents=True,exist_ok=True);(out/'manifest.json').write_text(json.dumps({'status':'RUNNING'},indent=2))
 feature=pd.read_parquet(audit_path);base_columns={'candidate_id','candidate_key','__ts__','side_name','era','gross_bps','net_bps','prequential_base_expected_net_bps'};fields=[f for f in feature.loc[feature.usable_90pct_nonconstant,'feature'].astype(str).tolist() if f not in base_columns]
 available_columns=set(ParquetFile(input_path).schema.names)
 keep=list(dict.fromkeys(['candidate_id','__ts__','side_name','era','net_bps','prequential_base_expected_net_bps',*(['label_available_ts'] if 'label_available_ts' in available_columns else []),*fields]))
 data=pd.read_parquet(input_path,columns=keep);data['__ts__']=pd.to_datetime(data['__ts__'],utc=True);data['label_available_ts']=pd.to_datetime(data['label_available_ts'],utc=True) if 'label_available_ts' in data else data['__ts__']+pd.Timedelta(hours=13);data=data[np.isfinite(data.net_bps)&np.isfinite(data.prequential_base_expected_net_bps)].copy();data['cohort']=select_top_base_per_timestamp(data,score_column='prequential_base_expected_net_bps',fraction=TOP);data=data[data.cohort].sort_values(['__ts__','candidate_id']).reset_index(drop=True)
 timestamps=pd.Index(data['__ts__'].drop_duplicates().sort_values())
 if start_test is None:
  fold_of={t:min(4,int(5*i/max(len(timestamps),1))) for i,t in enumerate(timestamps)}
 else:
  beginning=pd.Timestamp(start_test,tz='UTC');test_times=timestamps[timestamps>=beginning]
  if len(test_times)<3:raise ValueError('not enough timestamps after --start-test for three OOF folds')
  fold_of={t:-1 for t in timestamps};fold_of.update({t:2+min(2,int(3*i/max(len(test_times),1))) for i,t in enumerate(test_times)})
 data['fold']=data['__ts__'].map(fold_of).astype(int)
 reps=[];rules=[];similarities=[];lineages=[];diagnostics=[]
 for fold in (2,3,4):
  test=data[data.fold.eq(fold)].copy();start=test['__ts__'].min();history=data[data.label_available_ts.lt(start)].copy()
  if history_days is not None: history=history[history['__ts__'].ge(start-pd.Timedelta(days=float(history_days)))].copy()
  days=(history['__ts__'].max()-history['__ts__'].min()).total_seconds()/86400.
  if days<MIN_DAYS:continue
  for name,horizon in TARGETS.items():
   labelled=_target(pd.concat([history,test]),history,horizon);train=labelled[labelled.target_available_ts.lt(start)].copy();evaluate=labelled[labelled.fold.eq(fold)].copy()
   for side in ('long','short'):
    tr=train[train.side_name.eq(side)&np.isfinite(train.target_value)].copy();te=evaluate[evaluate.side_name.eq(side)&np.isfinite(evaluate.target_value)].copy()
    if len(tr)<2000 or len(te)<200:continue
    selected=_screen(tr,fields,tr.target_value.to_numpy(float));med=tr[selected].median().fillna(0.);iqr=(tr[selected].quantile(.75)-tr[selected].quantile(.25)).replace(0,1).fillna(1.)
    x=((tr[selected].fillna(med)-med)/iqr).clip(-8,8).to_numpy('float32');xt=((te[selected].fillna(med)-med)/iqr).clip(-8,8).to_numpy('float32')
    model=lgb.LGBMRegressor(objective='regression_l2',n_estimators=80,learning_rate=.04,num_leaves=16,max_depth=4,min_child_samples=max(80,int(.01*len(tr))),colsample_bytree=.8,reg_lambda=20.,random_state=20260803+fold,n_jobs=1,verbosity=-1).fit(x,tr.target_value.to_numpy(float))
    reference=tr.iloc[int(.8*len(tr)):].copy();reference.loc[:,selected]=((reference[selected].fillna(med)-med)/iqr).clip(-8,8);norm_te=te.copy();norm_te.loc[:,selected]=((norm_te[selected].fillna(med)-med)/iqr).clip(-8,8)
    r,rm=_rules(model,selected,reference,0.);rep,rule_rows,sim,used,lineage=_represent(norm_te,r,rm,side,name,fold)
    rep['target']=name;rep['fold']=fold;rep['target_value']=te.target_value.to_numpy(float);rep['net_bps']=te.net_bps.to_numpy(float);rep['residual_bps']=te.residual_bps.to_numpy(float);reps.append(rep);rules.append(rule_rows);similarities.append(sim.assign(target=name,fold=fold,side_name=side));lineages.append(lineage)
    for f in used:
     for target_col in ('target_value','net_bps','residual_bps'):
      c=spearmanr(rep[f],rep[target_col]).statistic;diagnostics.append({'target':name,'fold':fold,'side_name':side,'feature':f,'outcome':target_col,'rank_ic':float(c) if np.isfinite(c) else np.nan,'rows':len(rep)})
 pd.concat(reps,ignore_index=True).to_parquet(out/'oof_representations.parquet',index=False) if reps else None
 pd.concat(rules,ignore_index=True).to_parquet(out/'rule_clusters.parquet',index=False) if rules else None
 pd.concat(similarities,ignore_index=True).to_parquet(out/'rule_similarity.parquet',index=False) if similarities else None
 pd.concat(lineages,ignore_index=True).to_parquet(out/'representation_lineage.parquet',index=False) if lineages else None
 pd.DataFrame(diagnostics).to_parquet(out/'standalone_feature_diagnostics.parquet',index=False)
 manifest={'status':'COMPLETED','input':str(input_path),'cohort':'global top5% base expected net per timestamp','targets':TARGETS,'base_correctness':'soft side-local p05/p95 residual, .5 at exact calibration','period_rule':'equal-timestamp non-overlapping aggregation with period-end availability','min_resolved_history_days':MIN_DAYS,'history_window_days':history_days,'test_start':start_test,'shallow_lgbm':{'max_depth':4,'num_leaves':16},'feature_universe':len(fields),'feature_policy':'all available >=90%-coverage, nonconstant causal meta fields; no univariate cap','nested_folds':[2,3,4],'outputs':'representations are discovery candidates pending support/stability/redundancy/transport/meta gates'}
 (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');return out
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=OUT);p.add_argument('--input',type=Path,default=INPUT);p.add_argument('--availability',type=Path,default=AUDIT);p.add_argument('--start-test');p.add_argument('--history-days',type=float);a=p.parse_args();print(run(a.out,a.input,a.availability,a.start_test,a.history_days))
