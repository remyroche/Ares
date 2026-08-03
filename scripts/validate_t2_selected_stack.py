#!/usr/bin/env python3
"""Configuration validation of C4 base + A1 archetype + 14d residual stack."""
from __future__ import annotations
import argparse,json,os,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import numpy as np,pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from extreme_price_movements.feature_provenance_gate import validate_feature_columns
from extreme_price_movements.t2_atr_funnel import BarrierGeometry,soft_event_targets,top_book_metrics
from scripts.run_t2_atr_sequential_funnel import META_CANDIDATES,_add_causal_context,_conditional_mean,_huber,_score_frame
from scripts.run_t2_soft_archetype_ablation import PATH
def geometry(name):
 x=name.split('_TP')[1].split('_SL');return BarrierGeometry(float(x[0]),float(x[1]))
def label(d,e):
 x=d.merge(e,on='candidate_id',how='left',validate='one_to_one');x[['t2_upper_soft','t2_lower_soft','t2_timeout_soft']]=soft_event_targets(x,geometry(e.contract.iloc[0]),temperature_atr=.25);return x
def fit_base(tr,te,raw):
 cols=raw+['side_is_long','causal_entry_cost_bps'];x,z=tr[cols].to_numpy(np.float32),te[cols].to_numpy(np.float32);y=tr[['t2_upper_soft','t2_lower_soft','t2_timeout_soft']].to_numpy(float);p=np.column_stack([np.maximum(_huber(x,y[:,j],z),0) for j in range(3)]);p/=np.maximum(p.sum(1,keepdims=True),1e-8);m=np.array([_conditional_mean(y[:,j],tr.execution_net_ev_12h.to_numpy(float)) for j in range(3)])*1e4;return p@m,p
def main():
 p=argparse.ArgumentParser();p.add_argument('--ledger',type=Path,required=True);p.add_argument('--features-json',type=Path,required=True);p.add_argument('--certainty',type=Path,required=True);p.add_argument('--support',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.output.exists():raise FileExistsError(a.output)
 raw=list(validate_feature_columns(json.loads(a.features_json.read_text())['raw_feature_columns']));meta=[x for x in META_CANDIDATES if x in raw];need={'candidate_id','side_name','__ts__','__symbol__','execution_gross_ev_12h','execution_cost_return','execution_net_ev_12h','oof_fold'}|set(raw)
 led=_add_causal_context(pd.read_parquet(a.ledger,columns=sorted(need)));sup=pd.read_parquet(a.support,columns=['candidate_id',*PATH]);d=led.merge(sup,on='candidate_id',validate='one_to_one');d['__ts__']=pd.to_datetime(d.__ts__,utc=True);base=d[d.oof_fold.eq('base_train')].copy();dev=d[d.oof_fold.eq('meta_train')].copy();oos=d[d.oof_fold.eq('meta_oos')].copy();events={f.stem:pd.read_parquet(f) for f in a.certainty.glob('H*.parquet')}
 # C4 ensemble predictions from eight predeclared contracts.
 def c4(target):
  scores=[];probs=[]
  for e in events.values():
   s,q=fit_base(label(base,e),label(target,e),raw);scores.append(s);probs.append(q)
  return np.mean(scores,0),np.mean(probs,0)
 devs,devp=c4(dev);ooss,oosp=c4(oos)
 for f,s,q in ((dev,devs,devp),(oos,ooss,oosp)):
  f['score_bps']=s;f[['p_upper','p_lower','p_timeout']]=q
 # Path GMM K selected exactly as A1 (K=6) from earlier base, causal predictor forward.
 pp=make_pipeline(SimpleImputer(),StandardScaler());xp=pp.fit_transform(base[list(PATH)]);g=GaussianMixture(6,covariance_type='diag',reg_covar=.01,random_state=20260801).fit(xp);m=g.predict_proba(xp);cols=raw+['side_is_long','causal_entry_cost_bps'];xb=base[cols].to_numpy(np.float32)
 def causal_prob(f):
  z=f[cols].to_numpy(np.float32);out=np.column_stack([np.clip(_huber(xb,m[:,j],z),0,1) for j in range(6)]);return out/np.maximum(out.sum(1,keepdims=True),1e-8)
 devp_a,oosp_a=causal_prob(dev),causal_prob(oos)
 # Residual fit: last 14 development calendar days, exact OOF base/archetype inputs.
 start=oos.__ts__.min()-pd.Timedelta(days=14);mt=dev[dev.__ts__.ge(start)].copy();mt_a=devp_a[dev.__ts__.ge(start).to_numpy()]
 def mat(f,a):return np.column_stack((f[meta].to_numpy(np.float32),f[['side_is_long','causal_entry_cost_bps','score_bps','p_upper','p_lower','p_timeout']].to_numpy(np.float32),a))
 residual=mt.execution_net_ev_12h.to_numpy()*1e4-mt.score_bps.to_numpy();cor=_huber(mat(mt,mt_a),residual,mat(oos,oosp_a));
 books=[]
 for name,score in [('base_C4',ooss),('base_C4_plus_A1_residual',ooss+cor)]:
  b=_score_frame(oos,score,name,'configuration_validation','TP2_SL1',.25);r=top_book_metrics(b,score_column='score_bps');r['variant']=name;books.append(r)
 stage=Path(tempfile.mkdtemp(prefix='.'+a.output.name+'.',dir=a.output.parent))
 try:
  pd.concat(books).to_parquet(stage/'base_meta_stack_results.parquet',index=False);pd.DataFrame({'candidate_id':oos.candidate_id,'base_c4_bps':ooss,'a1_residual_bps':cor,'final_score_bps':ooss+cor}).to_parquet(stage/'base_meta_stack_predictions.parquet',index=False)
  (stage/'run_manifest.json').write_text(json.dumps({'stack':'C4 eight-contract base ensemble + A1 path-defined causal probabilities + 14d pointwise residual','evaluation':'configuration validation on previously opened meta_oos; NOT an independent untouched final OOS','base_train_rows':len(base),'meta_train_rows':len(dev),'meta_oos_rows':len(oos),'inference_features':'causal raw/context, base outputs, causal archetype probabilities only'},indent=2)+'\n');os.replace(stage,a.output)
 except Exception:
  import shutil;shutil.rmtree(stage,ignore_errors=True);raise
if __name__=='__main__':main()
