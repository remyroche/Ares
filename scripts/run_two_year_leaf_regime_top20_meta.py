#!/usr/bin/env python3
"""Nested top-20 adaptive leaf-regime selection and ordinal-meta comparison.

Each outer fold uses strictly prior label-resolved history.  The default is a
730-day rolling history, but an expanding-history replay can explicitly use
every available row.  Its early 60% discovers the adaptive leaf dictionary;
its later 40% selects at most 20 representations per side using an inner
chronological validation segment.  The outer period stays untouched until the
final baseline-vs-augmented comparison.
"""
from __future__ import annotations
import json,sys
from collections import defaultdict
from pathlib import Path
import numpy as np,pandas as pd,lightgbm as lgb
from sklearn.metrics import log_loss

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts.run_correctness_leaf_regime_oof import _rules,_represent,_screen,_target

# v3 is the first input contract that contains the full causal five-state,
# phase, geometry and continuous relationship surface.  v2 only carried two
# explicit context fields and therefore cannot answer whether those outputs
# help the residual meta model.
INPUT=ROOT/'data_perp/artifacts/correctness_leaf_regime_two_year_input_20260803_v3/input.parquet'
AVAIL=ROOT/'data_perp/artifacts/correctness_leaf_regime_two_year_input_20260803_v3/feature_availability.parquet'
OUT=ROOT/'data_perp/artifacts/correctness_leaf_regime_two_year_top20_meta_20260803_v2'
TARGETS={'row':None,'period12h':12,'period24h':24,'period72h':72}


def _required_context_output_groups(fields: list[str]) -> dict[str, list[str]]:
 """Return the complete requested causal state surface, or fail closed.

 These fields are candidates for nested MDA, not automatically selected
 meta inputs.  The check prevents a future replay from silently falling back
 to a posterior-only or partially materialised state contract.
 """
 available=set(map(str,fields))
 required={
  'five_state_probabilities':[f'market_regime__state_p_{i}' for i in range(5)],
  'state_uncertainty_and_persistence':[
   'market_regime__entropy','market_regime__top2_margin',
   'market_regime__state_age_hours','market_regime__state_switch_probability',
  ],
  'transition_phase_probabilities':[
   'market_regime__phase_p_stable','market_regime__phase_p_onset',
   'market_regime__phase_p_active','market_regime__phase_p_settling',
  ],
  'named_transition_probabilities':[
   'transition_stable_probability','transition_onset_probability',
   'transition_active_probability','transition_settling_probability',
  ],
 }
 missing={name:[field for field in group if field not in available] for name,group in required.items()}
 missing={name:group for name,group in missing.items() if group}
 if missing:
  raise ValueError(f'incomplete causal state contract; missing={missing}')
 geometry=sorted(field for field in available if field.startswith('geometry_regime__') and '__state_p_' in field)
 continuous=sorted(field for field in available if field.startswith('continuous_regime__'))
 if not geometry or not continuous:
  raise ValueError('incomplete causal geometry/continuous contract')
 return {**required,'geometry_memberships':geometry,'continuous_and_relation_features':continuous}

def _is_context_candidate(name: str) -> bool:
 """Regime/trust surfaces must earn admission through nested MDA.

 The raw alpha/meta contract remains the control.  The richer causal state
 outputs are passed to the exact same side-local selection gate as adaptive
 leaf representations, rather than being silently promoted into every fit.
 """
 lower=str(name).lower()
 return lower.startswith((
  'market_regime__','geometry_regime__','continuous_regime__',
  'regime_p_','regime_state_','transition_state_',
  'transition_','state_age','state_switch','regime_entropy','regime_top2_',
  'regime_transition_','market_direction_',
 ))

def _candidate_family(name: str) -> str:
 return str(name).split('__')[1] if str(name).startswith('leafreg__') else 'causal_context'

_GEOMETRY_INVARIANT_SUFFIXES=frozenset({
 'entropy','top2_margin','state_age_hours','state_switch_probability',
 'ood_distance_percentile','within_state_radius_percentile',
 'state_boundary_margin','centroid_distance_velocity','phase_entropy',
 'phase_top2_margin','phase_p_stable','phase_p_onset','phase_p_active',
 'phase_p_settling',
})

def _invariant_context_group(name: str) -> str | None:
 """Return a predeclared transport-oriented group, otherwise ``None``.

 Raw state-coordinate probabilities are deliberately excluded: a local GMM's
 component number has no fixed cross-era meaning.  Entropy, confidence,
 persistence, phase and novelty summaries retain the same semantics even when
 the fitted state geometry moves.
 """
 name=str(name)
 if name.startswith('continuous_regime__relationship_break__'):
  return 'relationship_breaks'
 if name.startswith('continuous_regime__'):
  return 'continuous_context'
 if name.startswith('geometry_regime__'):
  suffix=name.rsplit('__',1)[-1]
  if suffix not in _GEOMETRY_INVARIANT_SUFFIXES:
   return None
  return f'geometry_{name.split("__")[1]}_invariants'
 if name.startswith('market_regime__'):
  suffix=name.rsplit('__',1)[-1]
  return 'primary_confidence_transition' if suffix in _GEOMETRY_INVARIANT_SUFFIXES else None
 if name in {
  'regime_entropy','regime_top2_margin','state_age_hours','state_switch_probability',
  'regime_transition_entropy_12h','regime_transition_entropy_48h',
 } or name.startswith('transition_'):
  # transition_ names use a stable public ontology (stable/onset/active/
  # settling), unlike numeric GMM component coordinates.
  return 'primary_confidence_transition'
 return None

def _invariant_leaf_group(name: str) -> str | None:
 name=str(name)
 if name.startswith('cluster_state_'):
  return 'leaf_persistence_state'
 if not name.startswith('leafreg__'):
  return None
 if any(token in name for token in (
  '__signed_contribution','__positive_contribution','__negative_contribution',
  '__absolute_contribution','__total_contribution_share','__historical_',
  '__structural_stability','__instability','__rule_count',
 )):
  return 'leaf_economic_trust'
 if name.endswith('__G1_weighted_geometric') or any(token in name for token in (
  '__velocity_','__acceleration','__smoothed_membership','__hours_active_',
  '__activation_mass','__activation_entropy',
 )):
  return 'leaf_persistence_state'
 return None

def _folds(data, start_test: str = '2024-09-24'):
 ts=pd.Index(data.__ts__.drop_duplicates().sort_values());test=ts[ts>=pd.Timestamp(start_test,tz='UTC')]
 if len(test) < 3: raise ValueError(f'not enough test timestamps after {start_test}')
 m={x:-1 for x in ts};m.update({x:2+min(2,int(3*i/max(len(test),1))) for i,x in enumerate(test)});return data.__ts__.map(m).astype('int8')
def _ord(x): return np.where(x<=-50.,0,np.where(x>=50.,2,1)).astype('int8')
def _matrix(a,b,fields):
 med=a[fields].replace([np.inf,-np.inf],np.nan).median().fillna(0.);return a[fields].replace([np.inf,-np.inf],np.nan).fillna(med).to_numpy('float32'),b[fields].replace([np.inf,-np.inf],np.nan).fillna(med).to_numpy('float32')
def _model(a,b,fields):
 r=a.net_bps.to_numpy(float)-a.prequential_base_expected_net_bps.to_numpy(float);y=_ord(r);x,xt=_matrix(a,b,fields);c=np.bincount(y,minlength=3).astype(float);w=np.sqrt(len(y)/np.maximum(3*c[y],1));w=np.clip(w/w.mean(),.5,2.)
 m=lgb.LGBMClassifier(objective='multiclass',num_class=3,n_estimators=120,learning_rate=.035,num_leaves=20,min_child_samples=max(80,int(.01*len(a))),colsample_bytree=.8,reg_lambda=20.,random_state=20260803,n_jobs=1,verbosity=-1).fit(x,y,sample_weight=w)
 med=a[fields].replace([np.inf,-np.inf],np.nan).median().fillna(0.);p=np.clip(m.predict_proba(xt),1e-6,1.);p/=p.sum(1,keepdims=True);means=np.array([r[y==k].mean() for k in range(3)]);return p,means,m,med
def _score_metrics(frame,score,fold,arm):
 out=[]
 for q in (.01,.05,.10):
  z=frame.assign(_s=score).sort_values(['_s','candidate_id'],ascending=[False,True],kind='stable').head(max(1,int(np.ceil(len(frame)*q))))
  out.append({'fold':fold,'arm':arm,'top_fraction':q,'trades':len(z),'net_bps':float(z.net_bps.mean()),'gross_bps':float(z.gross_bps.mean()),'long_share':float(z.side_name.eq('long').mean())})
 return out

def run(out:Path=OUT, similarity_threshold: float=.70, input_path:Path=INPUT, availability_path:Path=AVAIL, context_policy: str='full', history_days: float|None=730., end_ts: str|None=None, start_test: str='2024-09-24', cohort_fraction: float=1.0):
 if context_policy not in {'full','invariant','posterior'}:
  raise ValueError('context_policy must be full, invariant or posterior')
 out.mkdir(parents=True,exist_ok=True)
 a=pd.read_parquet(availability_path);raw=[x for x in a.loc[a.usable_90pct_nonconstant,'feature'].astype(str) if x not in {'candidate_id','__ts__','__symbol__','decision_ts','label_available_ts','side_name','era','gross_bps','net_bps','prequential_base_expected_net_bps'}]
 all_context=[x for x in raw if _is_context_candidate(x)]
 context_contract=_required_context_output_groups(all_context) if context_policy in {'full','invariant'} else {}
 context_group={x:_invariant_context_group(x) for x in all_context}
 context=[x for x in all_context if context_policy=='full' or (context_policy=='invariant' and context_group[x] is not None)]
 # Coordinate-dependent outputs must not silently move into the raw baseline
 # when the invariant arm excludes them.
 # Posterior-only is the original control: raw base/meta inputs are left
 # untouched and no separately materialised market-context fields compete in
 # MDA.  Full/invariant arms explicitly remove their context candidate pool
 # from raw, so those fields can only enter after the nested gate.
 if context_policy!='posterior': raw=[x for x in raw if x not in all_context]
 cols=['candidate_id','__ts__','label_available_ts','side_name','era','gross_bps','net_bps','prequential_base_expected_net_bps',*raw]
 d=pd.read_parquet(input_path,columns=[*cols,*context]);d.__ts__=pd.to_datetime(d.__ts__,utc=True);d.label_available_ts=pd.to_datetime(d.label_available_ts,utc=True)
 if end_ts is not None:
  d=d[d.__ts__<pd.Timestamp(end_ts,tz='UTC')].copy()
 # The meta/residual learner is a per-row conversion model.  The historical
 # runner took the best candidate at each timestamp, which both discarded
 # almost the entire 5% cohort in sparse hours and changed the deployed
 # decision from the required pooled-global top-k after causal bps mapping.
 # Retain every same-side OOF base row here.  The only top-k operation is the
 # final, pooled score evaluation in ``_score_metrics``.
 d=d[np.isfinite(d.net_bps)&np.isfinite(d.prequential_base_expected_net_bps)].copy();d=d.sort_values(['__ts__','candidate_id']).reset_index(drop=True)
 if not (0.0 < cohort_fraction <= 1.0): raise ValueError('cohort_fraction must be in (0, 1]')
 if cohort_fraction < 1.0:
  d['cohort']=select_top_base_per_timestamp(d,score_column='prequential_base_expected_net_bps',fraction=cohort_fraction);d=d[d.cohort].copy()
 d['fold']=_folds(d,start_test)
 rows=[];select_rows=[];selection_audit=[];target_audit=[];group_audit=[];prediction=[]
 for fold in (2,3,4):
  test=d[d.fold.eq(fold)].copy();start=test.__ts__.min();history=d[d.label_available_ts<start].copy()
  if history_days is not None: history=history[history.__ts__>=start-pd.Timedelta(days=history_days)].copy()
  cut=history.__ts__.quantile(.60);discovery=history[history.__ts__<=cut].copy();meta_train=history[history.__ts__>cut].copy();inner_cut=meta_train.__ts__.quantile(.50);inner_train=meta_train[meta_train.__ts__<=inner_cut].copy();inner_valid=meta_train[meta_train.__ts__>inner_cut].copy();scored=[]
  for side in ('long','short'):
   disc=discovery[discovery.side_name.eq(side)].copy();mt=meta_train[meta_train.side_name.eq(side)].copy();it=inner_train[inner_train.side_name.eq(side)].copy();iv=inner_valid[inner_valid.side_name.eq(side)].copy();te=test[test.side_name.eq(side)].copy()
   if min(map(len,(disc,it,iv,te)))<300:continue
   candidates=list(context);candidate_group={x:(context_group[x] if context_policy=='invariant' else 'full_context') for x in context}
   combo_raw=pd.concat([mt,te],ignore_index=True)
   for name,horizon in TARGETS.items():
    labelled=_target(pd.concat([disc,mt,te],ignore_index=True),disc,horizon)
    tr=labelled.iloc[:len(disc)].copy()
    tr=tr[np.isfinite(tr.target_value)].copy();chosen=_screen(tr,raw,tr.target_value.to_numpy(float));med=tr[chosen].median().fillna(0.);iqr=(tr[chosen].quantile(.75)-tr[chosen].quantile(.25)).replace(0,1).fillna(1.)
    x=((tr[chosen].fillna(med)-med)/iqr).clip(-8,8).to_numpy('float32');model=lgb.LGBMRegressor(objective='regression_l2',n_estimators=80,learning_rate=.04,num_leaves=16,max_depth=4,min_child_samples=max(80,int(.01*len(tr))),colsample_bytree=.8,reg_lambda=20.,random_state=20260803+fold,n_jobs=1,verbosity=-1).fit(x,tr.target_value.to_numpy(float))
    ref=combo_raw.iloc[:len(mt)].copy();ref.loc[:,chosen]=((ref[chosen].fillna(med)-med)/iqr).clip(-8,8);norm=combo_raw.copy();norm.loc[:,chosen]=((norm[chosen].fillna(med)-med)/iqr).clip(-8,8)
    rules,memberships=_rules(model,chosen,ref,0.);rep,_,_,cluster_outputs,_=_represent(norm,rules,memberships,side,name,fold,minimum_similarity=similarity_threshold)
    # ``cluster_outputs`` includes posterior alternatives *and* contribution,
    # support/stability and causal activation/state dynamics.  Formerly only
    # the four posterior variants were appended here, leaving the richer
    # surface materialised but invisible to nested MDA and the final meta fit.
    if context_policy=='invariant':
     cluster_outputs=[col for col in cluster_outputs if _invariant_leaf_group(col) is not None]
    elif context_policy=='posterior':
     cluster_outputs=[col for col in cluster_outputs if col.endswith(('__G0_geometric','__G1_weighted_geometric','__G2_generalized_pminus2','__G3_softmin'))]
    candidates.extend(cluster_outputs)
    candidate_group.update({col:(_invariant_leaf_group(col) if context_policy=='invariant' else ('posterior_leaf' if context_policy=='posterior' else 'leaf_full_context')) for col in cluster_outputs})
    for col in cluster_outputs: combo_raw[col]=rep[col].to_numpy(float)
    valid_selection=labelled.iloc[len(disc):len(disc)+len(mt)].target_value.dropna()
    target_audit.append({'fold':fold,'side_name':side,'target':name,'discovery_rows':len(tr),'selection_rows':len(valid_selection),'target_std':float(valid_selection.std()) if len(valid_selection) else np.nan,'target_iqr':float(valid_selection.quantile(.75)-valid_selection.quantile(.25)) if len(valid_selection) else np.nan})
   # Candidate columns exist on meta train plus untouched test.  Inner MDA is
   # train-only; phantoms establish a null distribution under identical model
   # capacity and preprocessing.
   mt=combo_raw.iloc[:len(mt)].copy();te=combo_raw.iloc[len(mt):].copy();it=mt[mt.__ts__<=inner_cut].copy();iv=mt[mt.__ts__>inner_cut].copy()
   candidates=[c for c in candidates if c in mt and mt[c].notna().any() and mt[c].nunique(dropna=True)>1]
   rng=np.random.default_rng(20260803+fold+(0 if side=='long' else 17));phantoms=[]
   for j in range(20):
    n=f'phantom_{j:02d}';it[n]=rng.normal(size=len(it));iv[n]=rng.normal(size=len(iv));phantoms.append(n)
   all_fields=list(dict.fromkeys([*raw,*candidates,*phantoms]));p,means,model,median=_model(it,iv,all_fields);base_loss=log_loss(_ord(iv.net_bps.to_numpy(float)-iv.prequential_base_expected_net_bps.to_numpy(float)),p,labels=[0,1,2])
   mda={}
   # Permute the exact imputed matrix seen by the fitted model.  The prior
   # implementation copied a wide DataFrame and re-imputed it for every
   # candidate, which is allocation-heavy once the full regime/leaf contract
   # is present.  In-place restoration is equivalent for model inputs and
   # keeps the nested MDA test tractable.
   validation_matrix=iv[all_fields].replace([np.inf,-np.inf],np.nan).fillna(median).to_numpy('float32')
   field_index={name:index for index,name in enumerate(all_fields)}
   for c in [*candidates,*phantoms]:
    index=field_index[c];original=validation_matrix[:,index].copy();validation_matrix[:,index]=rng.permutation(original);pp=np.clip(model.predict_proba(validation_matrix),1e-6,1.);pp/=pp.sum(1,keepdims=True);mda[c]=float(log_loss(_ord(iv.net_bps.to_numpy(float)-iv.prequential_base_expected_net_bps.to_numpy(float)),pp,labels=[0,1,2])-base_loss);validation_matrix[:,index]=original
   threshold=float(np.quantile([mda[x] for x in phantoms],.95));ordered=[c for c in sorted(candidates,key=lambda x:-mda[x]) if mda[c]>threshold]
   group_mda={};eligible_groups=None
   if context_policy=='invariant':
    grouped=defaultdict(list)
    for c in candidates: grouped[candidate_group[c]].append(c)
    # Jointly permuting one predeclared group preserves its internal geometry
    # while breaking its relationship to the residual target.  Individual
    # phantom-MDA remains the stricter second-stage feature admission gate.
    for group,fields in grouped.items():
     indexes=[field_index[field] for field in fields];original=validation_matrix[:,indexes].copy();validation_matrix[:,indexes]=original[rng.permutation(len(validation_matrix))];pp=np.clip(model.predict_proba(validation_matrix),1e-6,1.);pp/=pp.sum(1,keepdims=True);group_mda[group]=float(log_loss(_ord(iv.net_bps.to_numpy(float)-iv.prequential_base_expected_net_bps.to_numpy(float)),pp,labels=[0,1,2])-base_loss);validation_matrix[:,indexes]=original
    eligible_groups={group for group,value in group_mda.items() if value>0.0}
    for group,fields in grouped.items():
     group_audit.append({'fold':fold,'side_name':side,'group':group,'candidate_count':len(fields),'group_mda_logloss':group_mda[group],'group_accepted':group in eligible_groups})
    ordered=[c for c in ordered if candidate_group[c] in eligible_groups]
   selected=[]
   for c in ordered:
    corr=max([abs(float(pd.Series(it[c]).corr(pd.Series(it[s])))) for s in selected],default=0.)
    reason='selected'
    if len(selected)>=20: reason='slot_cap'
    elif np.isfinite(corr) and corr>.80: reason='activation_correlation_above_080'
    else: selected.append(c)
    target_name=_candidate_family(c)
    selection_audit.append({'fold':fold,'side_name':side,'feature':c,'target':target_name,'group':candidate_group[c],'group_mda_logloss':group_mda.get(candidate_group[c],np.nan),'group_accepted':eligible_groups is None or candidate_group[c] in eligible_groups,'mda_logloss':mda[c],'phantom_q95':threshold,'mda_excess_over_phantom':mda[c]-threshold,'max_activation_correlation':corr,'candidate_rank_by_mda':ordered.index(c)+1,'accepted':reason=='selected','rejection_reason':reason})
   for c in candidates:
    if c in ordered: continue
    rejected='mda_at_or_below_phantom_q95' if mda[c]<=threshold else 'group_mda_at_or_below_zero'
    selection_audit.append({'fold':fold,'side_name':side,'feature':c,'target':_candidate_family(c),'group':candidate_group[c],'group_mda_logloss':group_mda.get(candidate_group[c],np.nan),'group_accepted':eligible_groups is None or candidate_group[c] in eligible_groups,'mda_logloss':mda[c],'phantom_q95':threshold,'mda_excess_over_phantom':mda[c]-threshold,'max_activation_correlation':np.nan,'candidate_rank_by_mda':np.nan,'accepted':False,'rejection_reason':rejected})
   # Final meta fit only sees post-discovery historical rows; base and
   # augmented predictions share exactly the same side/test contract.
   pb,means,_,_=_model(mt,te,raw);pa,_,_,_=_model(mt,te,[*raw,*selected]) if selected else (pb,means,None,None)
   z=te[['candidate_id','__ts__','side_name','gross_bps','net_bps','prequential_base_expected_net_bps']].copy();z['baseline_score_bps']=z.prequential_base_expected_net_bps.to_numpy(float)+pb@means;z['augmented_score_bps']=z.prequential_base_expected_net_bps.to_numpy(float)+pa@means;scored.append(z)
   select_rows.extend({'fold':fold,'side_name':side,'feature':c,'group':candidate_group[c],'group_mda_logloss':group_mda.get(candidate_group[c],np.nan),'mda_logloss':mda[c],'phantom_q95':threshold,'rank':i+1} for i,c in enumerate(selected))
  z=pd.concat(scored,ignore_index=True);z['fold']=fold;prediction.append(z);rows.extend(_score_metrics(z,z.baseline_score_bps.to_numpy(float),fold,'baseline_all_meta'));rows.extend(_score_metrics(z,z.augmented_score_bps.to_numpy(float),fold,'top20_leaf_regime'))
 pd.DataFrame(rows).to_parquet(out/'meta_comparison_metrics.parquet',index=False);pd.DataFrame(select_rows).to_parquet(out/'top20_selection.parquet',index=False);pd.DataFrame(selection_audit).to_parquet(out/'selection_candidate_audit.parquet',index=False);pd.DataFrame(group_audit).to_parquet(out/'invariant_group_mda_audit.parquet',index=False);pd.DataFrame(target_audit).to_parquet(out/'target_horizon_audit.parquet',index=False);pd.concat(prediction,ignore_index=True).to_parquet(out/'oof_predictions.parquet',index=False)
 (out/'manifest.json').write_text(json.dumps({'status':'COMPLETED','history_window_days':history_days,'history_policy':'expanding_all_prior_resolved_rows' if history_days is None else 'rolling','end_ts_exclusive':end_ts,'similarity_threshold':similarity_threshold,'context_policy':context_policy,'input_path':str(input_path),'availability_path':str(availability_path),'raw_feature_count':len(raw),'causal_context_candidate_count':len(context),'causal_context_candidate_contract':context_contract,'cluster_output_contract':'posterior variants; signed/positive/negative/absolute contribution; total contribution share; historical support/correctness/stability; membership velocity/acceleration/smoothing/active age; relative cluster entropy/top-two margin/switch probability/state age','discovery_fraction':.60,'meta_train_fraction':.40,'candidate_handoff':'all per-row same-side strict-OOF base outputs; no per-timestamp or per-side rank gate','selection':'predeclared invariant-group joint MDA > 0, then inner chronological phantom-MDA q95 + correlation <=.80, maximum 20 per side' if context_policy=='invariant' else 'inner chronological phantom-MDA q95 + correlation <=.80, maximum 20 per side','target':'ordinal residual +/-50 bps','ranking':'one pooled global top-k after side class-to-bps map'},indent=2)+'\n')

if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=OUT);p.add_argument('--similarity-threshold',type=float,default=.70);p.add_argument('--input',type=Path,default=INPUT);p.add_argument('--availability',type=Path,default=AVAIL);p.add_argument('--context-policy',choices=('full','invariant','posterior'),default='full');p.add_argument('--history-days',type=float,default=730.);p.add_argument('--end-ts',default=None);p.add_argument('--start-test',default='2024-09-24');p.add_argument('--cohort-fraction',type=float,default=1.0);a=p.parse_args();run(a.out,a.similarity_threshold,a.input,a.availability,a.context_policy,None if a.history_days<=0 else a.history_days,a.end_ts,a.start_test,a.cohort_fraction)
