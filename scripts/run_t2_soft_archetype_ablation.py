#!/usr/bin/env python3
"""A0/A1/A2 soft-archetype meta-input gate for selected T2 C4 stack."""
from __future__ import annotations
import argparse,json,os,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import lightgbm as lgb,numpy as np,pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from extreme_price_movements.feature_provenance_gate import validate_feature_columns
from extreme_price_movements.t2_atr_funnel import top_book_metrics
from scripts.run_t2_atr_sequential_funnel import META_CANDIDATES,_add_causal_context,_huber,_score_frame
PATH=("__peak_mfe_atr_12h__","__mae_before_meaningful_mfe_atr_12h__","__time_to_first_meaningful_mfe_hours_12h__","__future_slope_atr_per_hour_12h__","__mfe_mae_path_efficiency_12h__","__mfe_persistence_path_efficiency_12h__","__peak_mfe_fraction_above_80pct_12h__","__adverse_trough_recovery_fraction_12h__","endpoint_signed_return")
def main():
 p=argparse.ArgumentParser();p.add_argument('--ledger',type=Path,required=True);p.add_argument('--features-json',type=Path,required=True);p.add_argument('--support',type=Path,required=True);p.add_argument('--c4',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.output.exists():raise FileExistsError(a.output)
 raw=[x for x in META_CANDIDATES if x in validate_feature_columns(json.loads(a.features_json.read_text())['raw_feature_columns'])]
 cols={'candidate_id','side_name','__ts__','__symbol__','execution_gross_ev_12h','execution_cost_return','execution_net_ev_12h','oof_fold'}|set(raw)
 led=_add_causal_context(pd.read_parquet(a.ledger,columns=sorted(cols)));sup=pd.read_parquet(a.support,columns=['candidate_id',*PATH]);pred=pd.read_parquet(a.c4);pred=pred[pred.variant.eq('C4_contract_ensemble')][['candidate_id','score_bps','p_upper','p_lower','p_timeout']]
 d=led.merge(sup,on='candidate_id',validate='one_to_one').merge(pred,on='candidate_id',how='left',validate='one_to_one');d['__ts__']=pd.to_datetime(d.__ts__,utc=True);base=d[d.oof_fold.eq('base_train')].copy();dev=d[d.oof_fold.eq('meta_train')].copy()
 # GMM selection uses only earlier base rows, BIC plus a 5% support floor.
 path_pipe=make_pipeline(SimpleImputer(),StandardScaler());xp=path_pipe.fit_transform(base[list(PATH)]);xc=make_pipeline(SimpleImputer(),StandardScaler(),PCA(n_components=5,random_state=20260801)).fit_transform(base[raw])
 selected={};models={}
 for name,x in {'path':xp,'multiview':np.column_stack((xp,xc))}.items():
  candidates=[]
  for k in (3,4,5,6):
   g=GaussianMixture(k,covariance_type='diag',reg_covar=.01,random_state=20260801).fit(x);w=g.predict_proba(x).mean(0)
   if w.min()>=.05:candidates.append((g.bic(x),g,w))
  _,g,w=min(candidates,key=lambda z:z[0]);selected[name]=(g,w);models[name]=g
 # fit separate causal membership predictors base->dev, which is strict OOF for dev.
 xbase=base[raw+['side_is_long','causal_entry_cost_bps']].to_numpy(np.float32);xdev=dev[raw+['side_is_long','causal_entry_cost_bps']].to_numpy(np.float32)
 probs={}
 for name,(g,_) in selected.items():
  embed=xp if name=='path' else np.column_stack((xp,xc));membership=g.predict_proba(embed);out=[]
  for j in range(membership.shape[1]):out.append(np.clip(_huber(xbase,membership[:,j],xdev),0,1))
  z=np.column_stack(out);probs[name]=z/np.maximum(z.sum(1,keepdims=True),1e-8)
 # July residual test after 14 calendar development days of causal meta input.
 cut=pd.Timestamp('2024-07-01',tz='UTC');train=dev[(dev.__ts__>=cut-pd.Timedelta(days=14))&(dev.__ts__<cut)].copy();test=dev[dev.__ts__>=cut].copy()
 def mat(x,extra=None):
  arr=[x[raw].to_numpy(np.float32),x[['side_is_long','causal_entry_cost_bps','score_bps','p_upper','p_lower','p_timeout']].to_numpy(np.float32)]
  if extra is not None:arr.append(extra)
  return np.column_stack(arr)
 def eval_variant(name,prob=None):
  extra_train=extra_test=None
  if prob is not None:
   mask=(dev.__ts__>=cut-pd.Timedelta(days=14))&(dev.__ts__<cut);extra_train=prob[mask.to_numpy()];extra_test=prob[(dev.__ts__>=cut).to_numpy()]
  target=train.execution_net_ev_12h.to_numpy()*1e4-train.score_bps.to_numpy();cor=_huber(mat(train,extra_train),target,mat(test,extra_test));book=_score_frame(test,test.score_bps.to_numpy()+cor,name,'archetype_development','TP2_SL1',.25);r=top_book_metrics(book,score_column='score_bps');r['variant']=name;return r
 result=pd.concat([eval_variant('A0_no_archetype'),eval_variant('A1_path_archetype',probs['path']),eval_variant('A2_multiview_archetype',probs['multiview'])])
 stage=Path(tempfile.mkdtemp(prefix='.'+a.output.name+'.',dir=a.output.parent))
 try:
  result.to_parquet(stage/'soft_archetype_ablation.parquet',index=False)
  pd.DataFrame({'candidate_id':dev.candidate_id,**{f'{name}_p_{j}':z[:,j] for name,z in probs.items() for j in range(z.shape[1])}}).to_parquet(stage/'soft_archetype_oof_probabilities.parquet',index=False)
  cent=[]
  for name,(g,w) in selected.items():
   for j in range(g.n_components):cent.append({'view':name,'component':j,'support':float(w[j]),'k':g.n_components})
  pd.DataFrame(cent).to_parquet(stage/'soft_archetype_centroids.parquet',index=False)
  (stage/'soft_archetype_manifest.json').write_text(json.dumps({'K_candidates':[3,4,5,6],'selection':'minimum BIC with >=5% base-train support','path_features':list(PATH),'causal_features':raw,'causal_prediction':'base_train fit, meta_train OOF probabilities only','final_oos_untouched':True},indent=2)+'\n');os.replace(stage,a.output)
 except Exception:
  import shutil;shutil.rmtree(stage,ignore_errors=True);raise
if __name__=='__main__':main()
