#!/usr/bin/env python3
"""Seal the honest recurrence/support bound for fold-local transition morphology."""
from __future__ import annotations
import argparse, hashlib, json, os, shutil, uuid
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
CAT=ROOT/'data_perp/artifacts/transition_pattern_catalogue_20260730_v6'
STAB=ROOT/'data_perp/artifacts/recurring_transition_taxonomy_stability_20260730_v1'
OUT=ROOT/'data_perp/artifacts/transition_morphology_support_bound_20260730_v1'
def sha(p):
 h=hashlib.sha256();
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def sealed(r,*n):
 return r.is_dir() and all((r/x).is_file() for x in n) and ((not (r/'manifest.sha256').exists()) or (r/'manifest.sha256').read_text().split()[0]==sha(r/'manifest.json'))
def run(out:Path=OUT):
 if out.exists():raise FileExistsError(out)
 if not sealed(CAT,'morphology_oof.parquet','morphology_fold_plan.parquet','event_preonset_sequences.parquet') or not sealed(STAB,'morphology_classifier_oof_agreement.csv','gmm_fold_local_component_support.csv'):raise FileNotFoundError('sealed catalogue/stability inputs required')
 o=pd.read_parquet(CAT/'morphology_oof.parquet'); f=pd.read_parquet(CAT/'morphology_fold_plan.parquet'); o['era']=pd.to_datetime(o.anchor_source_utc,utc=True).dt.year.astype(str)
 # One OOF row per physical event is a hard anti-duplication invariant.
 if o.event_id.duplicated().any() or len(o)!=157:raise ValueError('morphology OOF is not one-row-per-event')
 test=f.loc[f.role.eq('test'),['fold','event_id']];
 # The catalogue OOF table itself is one row per event.  The fold-plan test
 # membership is only a supported subset, which is the key limiting count.
 z=o.copy(); z['fold']=z.oof_fold
 fold_audit=pd.DataFrame([{'oof_events':len(o),'unique_oof_events':o.event_id.nunique(),'fold_plan_test_rows':len(test),'unique_fold_plan_test_events':test.event_id.nunique(),'duplicate_fold_plan_test_events':int(test.event_id.duplicated().sum()),'all_events_explicitly_represented_in_fold_plan_test':bool(set(o.event_id)==set(test.event_id))}])
 support=z.groupby(['fold','morphology__component_id','era'],observed=True).size().rename('heldout_events').reset_index()
 aggregate=support.groupby(['fold','morphology__component_id'],observed=True).agg(heldout_events=('heldout_events','sum'),heldout_eras=('era','nunique')).reset_index()
 aggregate['eligible_global_type']=False;aggregate['reason']='component IDs are fold-local; train-only semantic descriptors/prototypes are absent, so cross-fold IDs cannot be matched honestly'
 calibration=z.groupby(['fold','morphology__component_id'],observed=True).agg(events=('event_id','size'),mean_entropy=('morphology__entropy','mean'),mean_top2_margin=('morphology__top2_margin','mean'),abstain_rate=('morphology__abstained','mean')).reset_index()
 agreement=pd.read_csv(STAB/'morphology_classifier_oof_agreement.csv')
 agreement['interpretation']='agreement is within existing OOF classifier outputs; it does not semantically align GMM components across folds'
 limits=pd.DataFrame([{'events':len(z),'eras':z.era.nunique(),'unique_oof_events':z.event_id.nunique(),'fold_local_components':aggregate.shape[0],'global_types_named':0,'decision':'NO_GLOBAL_MORPHOLOGY_TYPES_OR_GATES','limiting_condition':'No train-only cross-fold descriptor/prototype matching plus insufficient separated-era support per semantic type.'}])
 stage=out.parent/f'.{out.name}.{uuid.uuid4().hex}.stage';stage.mkdir()
 try:
  tables={'fold_local_support.csv':support,'component_support_bound.csv':aggregate,'calibration_abstention.csv':calibration,'agreement_context.csv':agreement,'fold_plan_coverage_audit.csv':fold_audit,'limiting_counts.csv':limits}
  outputs={}
  for n,t in tables.items():p=stage/n;t.to_csv(p,index=False);outputs[n]={'path':str(out/n),'rows':len(t),'sha256':sha(p)}
  m={'schema':'transition_morphology_support_bound_v1','status':'BOUND_INSUFFICIENT_FOR_GLOBAL_TYPES','promotion_eligible':False,'anti_leakage':'one OOF row per event; fold-plan test membership is audited separately and is a supported subset, never expanded by duplication','semantic_alignment':'fold-local components retained; no global semantic type names, gates, or later-state equivalence','inputs':{'catalogue_manifest':sha(CAT/'manifest.json'),'stability_manifest':sha(STAB/'manifest.json')},'outputs':outputs,'runner':{'path':str(Path(__file__)),'sha256':sha(Path(__file__))}}
  (stage/'manifest.json').write_text(json.dumps(m,indent=2,sort_keys=True)+'\n');(stage/'manifest.sha256').write_text(sha(stage/'manifest.json')+'  manifest.json\n');os.replace(stage,out)
 except Exception:shutil.rmtree(stage,ignore_errors=True);raise
 return m
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,default=OUT);a=p.parse_args();print(json.dumps(run(a.output_dir),sort_keys=True))
