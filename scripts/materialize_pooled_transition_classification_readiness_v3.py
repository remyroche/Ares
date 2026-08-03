#!/usr/bin/env python3
"""Audit completed historical prerequisites for pooled transition research.

The audit is evidence-only.  It binds the continuous historical context,
exact candidate labels, causal global map, candidate mapping coordinates and
global-book before/after labels.  It deliberately keeps common-feature
semantic parity as a separate gate: matching economic labels cannot prove
feature semantics.
"""
from __future__ import annotations

import argparse, hashlib, json, os, sys, tempfile
from pathlib import Path
from typing import Any, Mapping
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from scripts.materialize_historical_current_common_transition_geometry import CANONICAL_FEATURES, RAW_FIELDS
CONTEXT=ROOT/'data_perp/artifacts/failure_2022_2023_pf_exact1m_transition_context_continuation_20260730_v1'
CANDIDATE_LABELS=ROOT/'data_perp/artifacts/failure_2022_2023_pf_exact1m_multitask_labels_20260730_v1'
MAPPING=ROOT/'data_perp/artifacts/failure_2022_2023_pf_exact1m_causal_global_book_mapping_20260730_v1'
BOOK_LABELS=ROOT/'data_perp/artifacts/failure_2022_2023_pf_exact1m_global_book_transition_labels_20260730_v1'
CURRENT=ROOT/'data_perp/artifacts/cross_era_global_book_transition_research_panel_20260730_v4'
COMMON_GEOMETRY=ROOT/'data_perp/artifacts/historical_current_common_transition_geometry_20260730_v1'
# v3--v6 were published from older or failed-closed audit schemas; never overwrite them.
OUTPUT=ROOT/'data_perp/artifacts/pooled_transition_classification_readiness_20260730_v7'
IDENTITY=('__ts__','__symbol__','side_name','candidate_id')
SCHEMA='pooled_transition_classification_readiness_v3'
COMMON_SCHEMA='historical_current_common_transition_geometry_v1'
COMMON_STATUS='MATERIALIZED_STRICT_SEMANTIC_COMMON_GEOMETRY'

def sha256(path:Path)->str:
 d=hashlib.sha256()
 with path.open('rb') as h:
  for b in iter(lambda:h.read(1<<20),b''): d.update(b)
 return d.hexdigest()
def safe(x:Any)->Any:
 if isinstance(x,Mapping): return {str(k):safe(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)): return [safe(v) for v in x]
 if isinstance(x,(Path,pd.Timestamp)): return str(x)
 return x
def write_json(p:Path,x:Mapping[str,Any])->None:
 t=p.with_name(f'.{p.name}.{os.getpid()}.tmp');t.write_text(json.dumps(safe(x),indent=2,sort_keys=True)+'\n');os.replace(t,p)
def manifest(root:Path)->tuple[dict[str,Any],Path]:
 p=root/'manifest.json'; seal=root/'manifest.sha256'
 if not p.is_file(): raise FileNotFoundError(root)
 # Older immutable label packets predate detached seals but bind their output
 # hashes in manifest.json; output binding is still required below.
 if seal.is_file() and seal.read_text().split()[0]!=sha256(p): raise ValueError(f'manifest checksum fails: {root}')
 return json.loads(p.read_text()),p
def bound(root:Path,name:str,expected_sha:str|None=None)->tuple[dict[str,Any],Path,Path]:
 m,mp=manifest(root); p=root/name
 if not p.is_file(): raise FileNotFoundError(p)
 actual=sha256(p)
 if expected_sha is not None and expected_sha!=actual: raise ValueError(f'{name} does not match manifest: {root}')
 return m,mp,p

def _identity(frame:pd.DataFrame)->pd.DataFrame:
 out=frame.loc[:,list(IDENTITY)].copy();out['__ts__']=pd.to_datetime(out['__ts__'],utc=True,errors='raise');out['__symbol__']=out['__symbol__'].astype(str);out['side_name']=out['side_name'].astype(str).str.lower();out['candidate_id']=out['candidate_id'].astype(str)
 if out.duplicated(list(IDENTITY)).any(): raise ValueError('common geometry identity duplicated')
 return out

def common_geometry_requirement(root:Path,candidates:pd.DataFrame,current_features:list[str],current_panel_path:Path)->dict[str,Any]:
 """Fail closed unless the semantic common-geometry artifact proves every contract."""
 try:
  gm,gmp=manifest(root)
  if gm.get('schema')!=COMMON_SCHEMA or gm.get('status')!=COMMON_STATUS: raise ValueError('schema/status mismatch')
  outputs=gm.get('outputs');
  if not isinstance(outputs,dict): raise ValueError('missing output bindings')
  paths={}
  for name in ('historical_candidate_context','historical_hourly_state_geometry','current_v4_semantic_context'):
   record=outputs.get(name)
   if not isinstance(record,dict): raise ValueError(f'missing output binding: {name}')
   path=Path(record.get('path','')).resolve()
   if path.parent!=root.resolve() or not path.is_file() or record.get('sha256')!=sha256(path): raise ValueError(f'output hash/path mismatch: {name}')
   paths[name]=path
  for name in ('semantic_mapping','audit'):
   record=gm.get(name)
   if not isinstance(record,dict): raise ValueError(f'missing {name} binding')
   path=(root/record.get('path','')).resolve()
   if path.parent!=root.resolve() or not path.is_file() or record.get('sha256')!=sha256(path): raise ValueError(f'{name} hash/path mismatch')
   paths[name]=path
  mapping=json.loads(paths['semantic_mapping'].read_text());audit=json.loads(paths['audit'].read_text())
  if mapping.get('raw_field_overlap')!=list(RAW_FIELDS) or mapping.get('canonical_feature_columns')!=list(CANONICAL_FEATURES): raise ValueError('semantic mapping field contract mismatch')
  nofill=str(audit.get('no_fill',''))
  if not all(token in nofill for token in ('exact timestamp','no asof','resample','interpolation','ffill','bfill')): raise ValueError('no-fill audit is incomplete')
  parity=audit.get('canonical_parity',{})
  if parity.get('feature_count')!=90 or not all(parity.get(k) is True for k in ('all_common_features_declared_by_current_v4','historical_columns_equal_contract','current_columns_equal_contract')): raise ValueError('current-v4 canonical parity audit fails')
  overlap=audit.get('raw_name_overlap',{})
  if overlap.get('count')!=9 or overlap.get('fields')!=list(RAW_FIELDS) or overlap.get('exact_expected_nine') is not True: raise ValueError('raw semantic overlap audit fails')
  source_current=gm.get('sources',{}).get('current_v4_panel',{})
  if source_current.get('sha256')!=sha256(current_panel_path): raise ValueError('common geometry is not bound to this current-v4 panel')
  if set(CANONICAL_FEATURES)-set(current_features): raise ValueError('current-v4 declaration lacks a canonical common feature')
  historical=pd.read_parquet(paths['historical_candidate_context'])
  expected_columns=[*IDENTITY,'__decision_ts__','common_transition_context_available',*CANONICAL_FEATURES]
  if historical.columns.tolist()!=expected_columns: raise ValueError('historical candidate common geometry has an invalid column contract')
  historical_identity=_identity(historical);candidate_identity=_identity(candidates)
  historical_set=historical_identity.sort_values(list(IDENTITY),kind='stable').reset_index(drop=True)
  candidate_set=candidate_identity.sort_values(list(IDENTITY),kind='stable').reset_index(drop=True)
  if not historical_set.equals(candidate_set): raise ValueError('historical common geometry identity set differs from exact labels')
  decision=pd.to_datetime(historical['__decision_ts__'],utc=True,errors='raise')
  if not decision.eq(historical_identity['__ts__']+pd.Timedelta(hours=1)).all(): raise ValueError('historical common geometry signal-to-decision timing fails')
  if not historical['common_transition_context_available'].astype(bool).all(): raise ValueError('historical common geometry does not cover every exact candidate')
  hourly=pd.read_parquet(paths['historical_hourly_state_geometry'])
  if hourly.columns.tolist()!=['signal_context_utc',*CANONICAL_FEATURES] or hourly.empty: raise ValueError('historical hourly common geometry contract fails')
  current=pd.read_parquet(paths['current_v4_semantic_context'])
  if current.columns.tolist()!=['signal_context_utc','common_transition_context_available',*CANONICAL_FEATURES] or current.empty: raise ValueError('current-v4 common projection contract fails')
  if not current['common_transition_context_available'].astype(bool).all(): raise ValueError('current-v4 common geometry has unavailable rows')
  return {'requirement':'common_decision_time_feature_contract','ready':True,'reason':f'hash-verified strict semantic bridge: 9 raw fields -> 90 canonical current-v4 fields; historical exact coverage={len(historical)}/{len(candidates)}; current unique contexts={len(current)}; no-fill parity verified'}
 except Exception as exc:
  return {'requirement':'common_decision_time_feature_contract','ready':False,'reason':f'strict semantic common geometry unavailable: {type(exc).__name__}: {exc}'}

def audit(context:pd.DataFrame,candidates:pd.DataFrame,mapping:pd.DataFrame,coordinates:pd.DataFrame,labels:pd.DataFrame,current_features:list[str],common_requirement:dict[str,Any])->tuple[pd.DataFrame,list[dict[str,Any]]]:
 for n,f in [('context',context),('candidate labels',candidates)]:
  if set(IDENTITY)-set(f): raise ValueError(f'{n} lacks exact identity')
  if f.duplicated(list(IDENTITY)).any(): raise ValueError(f'{n} identity duplicated')
 left=context.loc[:,list(IDENTITY)].sort_values(list(IDENTITY)).reset_index(drop=True);right=candidates.loc[:,list(IDENTITY)].sort_values(list(IDENTITY)).reset_index(drop=True)
 if not left.equals(right): raise ValueError('context/candidate label identity sets differ')
 context=context.copy();context['__decision_ts__']=pd.to_datetime(context['__decision_ts__'],utc=True,errors='raise')
 cov=context.assign(year=context.__decision_ts__.dt.year,month=context.__decision_ts__.dt.month).groupby(['year','month','side_name'],sort=True).agg(candidate_rows=('transition_context_available','size'),transition_context_rows=('transition_context_available','sum')).reset_index();cov['transition_context_coverage']=cov.transition_context_rows/cov.candidate_rows
 is23=context.__decision_ts__.dt.year.eq(2023); covered23=int(context.loc[is23,'transition_context_available'].sum());rows23=int(is23.sum())
 reqmap={'candidate_id','execution_decision_utc','execution_label_end_utc','mapped_eligible','mapped_direct_net','map_reference_rows'}
 if reqmap-set(mapping): raise ValueError('mapping lacks required causal fields')
 e=mapping.mapped_eligible.fillna(False).astype(bool); refs=pd.to_numeric(mapping.loc[e,'map_reference_rows'],errors='coerce');scores=pd.to_numeric(mapping.loc[e,'mapped_direct_net'],errors='coerce')
 mapok=bool(e.any() and refs.ge(1000).all() and scores.notna().all())
 if set(['candidate_id','causal_global_mapped_ev_band'])-set(coordinates): raise ValueError('coordinate artifact lacks required fields')
 coordok=int(coordinates.causal_global_mapped_ev_band.ne('UNAVAILABLE').sum())
 reqlabel={'cohort_anchor_utc','horizon_hours','book_fraction','before_window_start_utc','before_window_end_utc','after_window_start_utc','after_window_end_utc','before_target_available_utc','after_target_available_utc','selection_contract'}
 if reqlabel-set(labels): raise ValueError('global-book label artifact lacks required fields')
 top=labels.loc[pd.to_numeric(labels.book_fraction,errors='coerce').eq(.10)].copy(); exact=True
 for h in (3,12):
  x=top.loc[pd.to_numeric(top.horizon_hours,errors='coerce').eq(h)].copy()
  for c in ('cohort_anchor_utc','before_window_start_utc','before_window_end_utc','after_window_start_utc','after_window_end_utc','before_target_available_utc','after_target_available_utc'): x[c]=pd.to_datetime(x[c],utc=True,errors='coerce')
  exact &= bool(len(x) and x.before_window_end_utc.eq(x.cohort_anchor_utc).all() and x.after_window_start_utc.eq(x.cohort_anchor_utc).all() and x.before_window_start_utc.eq(x.cohort_anchor_utc-pd.Timedelta(hours=h)).all() and x.after_window_end_utc.eq(x.cohort_anchor_utc+pd.Timedelta(hours=h)).all() and x.after_target_available_utc.ge(x.after_window_end_utc).all())
 labelok=bool(exact and top.selection_contract.eq('one_pooled_global_mapped_direct_net').all())
 reqs=[
  {'requirement':'historical_transition_context_through_2023','ready':bool(rows23 and covered23==rows23),'reason':f'2023 exact decision context={covered23}/{rows23}; latest={context.__decision_ts__.max()}'},
  common_requirement,
  {'requirement':'causal_global_book_selection','ready':mapok,'reason':f'causal mapped candidates={int(e.sum())}/{len(mapping)}; min prior-resolved support={int(refs.min()) if len(refs) else 0}; coordinate-ready={coordok}/{len(coordinates)}'},
  {'requirement':'exact_before_after_transition_targets','ready':labelok,'reason':f'pooled-global top10 H3/H12 rows={len(top)}; exact [s-H,s)/[s,s+H) and availability={exact}'},
 ]
 return cov,reqs

def run(args:argparse.Namespace)->dict[str,Any]:
 cm,cmp,cp=bound(args.context,'context.parquet',manifest(args.context)[0].get('output',{}).get('sha256'))
 lm,lmp,lp=bound(args.candidate_labels,'joined_multitask_labels.parquet',manifest(args.candidate_labels)[0].get('outputs',{}).get('joined_multitask_labels',{}).get('sha256'))
 mm,mmp,mp=bound(args.mapping,'causal_mapped_candidates.parquet',manifest(args.mapping)[0].get('outputs',{}).get('mapped',{}).get('sha256'))
 bm,bmp,bp=bound(args.book_labels,'global_book_transition_labels.parquet',manifest(args.book_labels)[0].get('outputs_sha256',{}).get('global_book_transition_labels.parquet'))
 _,_,coord=bound(args.book_labels,'candidate_global_mapped_ev_coordinates.parquet',bm.get('outputs_sha256',{}).get('candidate_global_mapped_ev_coordinates.parquet'))
 current,currmp,currp=bound(args.current,'transition_research_panel.parquet',manifest(args.current)[0].get('outputs',{}).get('panel',{}).get('sha256'))
 # Canonical label packets store paths relative to the workspace while this
 # audit uses absolute paths.  Resolve before comparing the bound hash.
 bound_hashes={Path(k).resolve():v for k,v in bm.get('source_artifacts_sha256',{}).items()}
 if bound_hashes.get(mmp.resolve())!=sha256(mmp): raise ValueError('global-book labels do not bind mapping manifest')
 candidates=pd.read_parquet(lp)
 common_requirement=common_geometry_requirement(args.common_geometry,candidates,list(current.get('feature_columns',[])),currp)
 cov,reqs=audit(pd.read_parquet(cp),candidates,pd.read_parquet(mp),pd.read_parquet(coord),pd.read_parquet(bp),list(current.get('feature_columns',[])),common_requirement)
 out=Path(args.output_dir)
 if out.exists(): raise FileExistsError(out)
 out.parent.mkdir(parents=True,exist_ok=True); tmp=Path(tempfile.mkdtemp(dir=out.parent,prefix=f'.{out.name}.'))
 cov.to_csv(tmp/'coverage_by_month_side.csv',index=False);pd.DataFrame(reqs).to_csv(tmp/'readiness.csv',index=False);write_json(tmp/'missing_derivations.json',{'missing_derivations':[r for r in reqs if not r['ready']]})
 result={'schema':SCHEMA,'status':'READY_FOR_POOLED_TRANSITION_CLASSIFICATION' if all(r['ready'] for r in reqs) else 'INCOMPLETE_POOLED_TRANSITION_CLASSIFICATION_REQUIREMENTS','all_requirements_ready':all(r['ready'] for r in reqs),'missing_requirement_ids':[r['requirement'] for r in reqs if not r['ready']],'contracts':{'no_imputation':'audit only','causal_map':'outcomes resolve before each UTC-day snapshot','selection':'one pooled global top10 after mapping; no timestamp/side/asset quota','labels':'exact before [s-H,s), after [s,s+H), H3/H12; outcomes only','common_geometry':'hash-verified 9-field semantic bridge; 90 canonical current-v4 decision-time fields; exact signal+1h and no fill'},'sources':{n:{'path':str(p),'sha256':sha256(p),'manifest_sha256':sha256(m)} for n,p,m in [('context',cp,cmp),('candidate_labels',lp,lmp),('mapping',mp,mmp),('global_book_labels',bp,bmp),('coordinates',coord,bmp),('current_panel',currp,currmp)]}|{'common_geometry':{'path':str(args.common_geometry),'manifest_sha256':sha256(args.common_geometry/'manifest.json')}},'outputs_sha256':{p.name:sha256(p) for p in tmp.iterdir() if p.is_file()}}
 write_json(tmp/'manifest.json',result);(tmp/'manifest.sha256').write_text(f'{sha256(tmp/"manifest.json")}  manifest.json\n');os.replace(tmp,out);return result
def parser()->argparse.ArgumentParser:
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--context',type=Path,default=CONTEXT);p.add_argument('--candidate-labels',type=Path,default=CANDIDATE_LABELS);p.add_argument('--mapping',type=Path,default=MAPPING);p.add_argument('--book-labels',type=Path,default=BOOK_LABELS);p.add_argument('--current',type=Path,default=CURRENT);p.add_argument('--common-geometry',type=Path,default=COMMON_GEOMETRY);p.add_argument('--output-dir',type=Path,default=OUTPUT);return p
if __name__=='__main__': print(json.dumps(safe(run(parser().parse_args())),indent=2,sort_keys=True))
