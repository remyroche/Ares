#!/usr/bin/env python3
"""Genuine leave-one-era-out, train-only GMM morphology evaluation."""
from __future__ import annotations
import argparse, hashlib, json, os, shutil, uuid
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
ROOT=Path(__file__).resolve().parents[1]; INP=ROOT/'data_perp/artifacts/transition_pattern_catalogue_20260730_v6/event_preonset_sequences.parquet'; OUT=ROOT/'data_perp/artifacts/leave_one_era_out_transition_morphology_20260730_v1'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def run(out=OUT):
 if out.exists():raise FileExistsError(out)
 x=pd.read_parquet(INP); x['era']=pd.to_datetime(x.anchor_source_utc,utc=True).dt.year.astype(str)
 if x.event_id.duplicated().any() or len(x)!=157:raise ValueError('requires exactly 157 unique events')
 cols=[c for c in x if c.startswith('sequence__') and pd.api.types.is_numeric_dtype(x[c]) and not c.endswith('complete_1h')]
 # retain finite trainable columns; selection is outcome-free and repeated per era.
 rows=[]; skips=[]; descriptors=[]
 for era in sorted(x.era.unique()):
  tr=x[x.era.ne(era)]; te=x[x.era.eq(era)]
  keep=[c for c in cols if tr[c].notna().mean()>=.9 and tr[c].std(skipna=True)>1e-10]
  if len(tr)<32 or len(te)<4 or len(keep)<8:
   skips.append({'heldout_era':era,'train_events':len(tr),'test_events':len(te),'features':len(keep),'reason':'minimum train/test/feature support'});continue
  imp=SimpleImputer(strategy='median'); sc=StandardScaler(); a=sc.fit_transform(imp.fit_transform(tr[keep])); b=sc.transform(imp.transform(te[keep])); g=GaussianMixture(n_components=3,reg_covar=1e-5,n_init=5,random_state=1729).fit(a); p=g.predict_proba(b); comp=p.argmax(1); ent=-(np.clip(p,1e-12,1)*np.log(np.clip(p,1e-12,1))).sum(1)/np.log(3); margin=np.sort(p,axis=1)[:,-1]-np.sort(p,axis=1)[:,-2]
  # Fold-local descriptor is train-only; no cross-era semantic matching claim.
  for k in range(3): descriptors.append({'heldout_era':era,'fold_component':k,'train_events':len(tr),'train_component_events':int((g.predict(a)==k).sum()),'descriptor_rv_proxy':float(g.means_[k,keep.index('sequence__negative_breadth_pct__mean_1h')] if 'sequence__negative_breadth_pct__mean_1h' in keep else np.nan)})
  q=te[['event_id','anchor_source_utc','era']].copy();q['heldout_era']=era;q['fold_component']=comp;q['posterior']=p.max(1);q['entropy']=ent;q['top2_margin']=margin;q['abstained']=(p.max(1)<.60)|(margin<.15);rows.append(q)
 o=pd.concat(rows,ignore_index=True) if rows else pd.DataFrame(); support=o.groupby(['fold_component','heldout_era'],observed=True).size().rename('events').reset_index() if len(o) else pd.DataFrame(columns=['fold_component','heldout_era','events'])
 recurring=support.groupby('fold_component').agg(heldout_eras=('heldout_era','nunique'),events=('events','sum')).reset_index();recurring['semantic_recurrence_pass']=False;recurring['reason']='component identifiers/descriptors are fold-local; no train-only cross-fold prototype matching or predictive outcome exists'
 stage=out.parent/f'.{out.name}.{uuid.uuid4().hex}.stage';stage.mkdir()
 try:
  files={'oof_assignments.parquet':o,'support.csv':support,'train_only_descriptors.csv':pd.DataFrame(descriptors),'recurrence_gate.csv':recurring,'skip_reasons.csv':pd.DataFrame(skips)};outs={}
  for n,t in files.items():p=stage/n;(t.to_parquet(p,index=False) if n.endswith('parquet') else t.to_csv(p,index=False));outs[n]={'rows':len(t),'sha256':sha(p)}
  m={'schema':'leave_one_era_out_transition_morphology_v1','status':'EVALUATED_NO_GLOBAL_TYPES','promotion_eligible':False,'events':157,'anti_leakage':'each event assigned only by a GMM trained excluding its calendar era; no duplicated event rows','result':'fold-local recurrence support is measured, but global semantic/predictive recurrence and policy gates remain false','outputs':outs};(stage/'manifest.json').write_text(json.dumps(m,indent=2,sort_keys=True)+'\n');(stage/'manifest.sha256').write_text(sha(stage/'manifest.json')+'  manifest.json\n');os.replace(stage,out)
 except Exception:shutil.rmtree(stage,ignore_errors=True);raise
 return m
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,default=OUT);a=p.parse_args();print(json.dumps(run(a.output_dir),sort_keys=True))
