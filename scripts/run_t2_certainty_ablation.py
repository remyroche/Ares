#!/usr/bin/env python3
"""Development-only C0--C4 certainty ablation for frozen T2 TP2/SL1."""
from __future__ import annotations
import argparse,json,os,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import numpy as np,pandas as pd
from extreme_price_movements.feature_provenance_gate import validate_feature_columns
from extreme_price_movements.t2_atr_funnel import BarrierGeometry,soft_event_targets,top_book_metrics
from scripts.run_t2_atr_sequential_funnel import _add_causal_context,_fit_base,_score_frame

CANON='H12_TP2.0_SL1.0'
def labels(base,event):
 d=base.merge(event,on='candidate_id',how='left',validate='one_to_one')
 if d.timeout.isna().any():raise ValueError('event label gap')
 d[['t2_upper_soft','t2_lower_soft','t2_timeout_soft']]=soft_event_targets(d,BarrierGeometry(float(event.contract.iloc[0].split('_TP')[1].split('_SL')[0]),float(event.contract.iloc[0].split('_SL')[1])),temperature_atr=.25)
 return d
def metrics(frame,score,variant):
 out=top_book_metrics(_score_frame(frame,score,variant,'certainty_development','TP2_SL1',.25),score_column='score_bps');out['variant']=variant;return out
def main():
 a=argparse.ArgumentParser();a.add_argument('--ledger',type=Path,required=True);a.add_argument('--features-json',type=Path,required=True);a.add_argument('--certainty',type=Path,required=True);a.add_argument('--output',type=Path,required=True);x=a.parse_args()
 if x.output.exists():raise FileExistsError(x.output)
 fs=list(validate_feature_columns(json.loads(x.features_json.read_text())['raw_feature_columns']))
 req={'candidate_id','side_name','__ts__','__symbol__','execution_gross_ev_12h','execution_cost_return','execution_net_ev_12h','oof_fold'}
 ledger=_add_causal_context(pd.read_parquet(x.ledger,columns=sorted(req|set(fs))))
 train0=ledger[ledger.oof_fold.eq('base_train')];dev0=ledger[ledger.oof_fold.eq('meta_train')]
 cert=pd.read_parquet(x.certainty/'label_certainty_diagnostics.parquet')
 event_files=sorted(x.certainty.glob('H*.parquet'));events={p.stem:pd.read_parquet(p) for p in event_files}
 train=labels(train0,events[CANON]).merge(cert[['candidate_id','label_certainty']],on='candidate_id',validate='one_to_one');dev=labels(dev0,events[CANON])
 allm=[];pred=[]
 variants={'C0_uniform':None,'C1_mild':.5+.5*train.label_certainty.to_numpy(),'C2_strong':.25+.75*train.label_certainty.to_numpy()}
 for name,w in variants.items():
  score,prob=_fit_base(train,dev,fs,w);allm.append(metrics(dev,score,name));z=_score_frame(dev,score,name,'certainty_development','TP2_SL1',.25);z[['p_upper','p_lower','p_timeout']]=prob;pred.append(z)
 # Consensus is the mean of each predeclared soft contract label, training-only.
 train_cons=train.copy();dev_cons=dev.copy()
 for target in ('t2_upper_soft','t2_lower_soft','t2_timeout_soft'):
  train_cons[target]=0.;dev_cons[target]=0.
 for event in events.values():
  et=labels(train0,event);ed=labels(dev0,event)
  for target in ('t2_upper_soft','t2_lower_soft','t2_timeout_soft'):
   train_cons[target]+=et[target].to_numpy()/len(events);dev_cons[target]+=ed[target].to_numpy()/len(events)
 score,prob=_fit_base(train_cons,dev_cons,fs);allm.append(metrics(dev_cons,score,'C3_consensus'));z=_score_frame(dev_cons,score,'C3_consensus','certainty_development','TP2_SL1',.25);z[['p_upper','p_lower','p_timeout']]=prob;pred.append(z)
 # Contract ensemble: fit separate contract-labelled bases, average outputs.
 scores=[];probs=[]
 for event in events.values():
  et,ed=labels(train0,event),labels(dev0,event);s,p=_fit_base(et,ed,fs);scores.append(s);probs.append(p)
 score=np.mean(scores,axis=0);prob=np.mean(probs,axis=0);allm.append(metrics(dev,score,'C4_contract_ensemble'));z=_score_frame(dev,score,'C4_contract_ensemble','certainty_development','TP2_SL1',.25);z[['p_upper','p_lower','p_timeout']]=prob;pred.append(z)
 stage=Path(tempfile.mkdtemp(prefix='.'+x.output.name+'.',dir=x.output.parent))
 try:
  pd.concat(allm).to_parquet(stage/'label_stability_ablation.parquet',index=False);pd.concat(pred).to_parquet(stage/'certainty_development_predictions.parquet',index=False)
  (stage/'manifest.json').write_text(json.dumps({'protocol':'development only; final OOS remains untouched','variants':list(variants)+['C3_consensus','C4_contract_ensemble'],'certainty_not_in_inference':True,'contracts':[p.stem for p in event_files]},indent=2)+'\n');os.replace(stage,x.output)
 except Exception:
  import shutil;shutil.rmtree(stage,ignore_errors=True);raise
if __name__=='__main__':main()
