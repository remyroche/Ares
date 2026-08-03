#!/usr/bin/env python3
"""R0 pointwise versus R1 context-pairwise rank correction on C4 outputs."""
from __future__ import annotations
import argparse,json,os,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import lightgbm as lgb,numpy as np,pandas as pd
from sklearn.linear_model import Ridge
from extreme_price_movements.feature_provenance_gate import validate_feature_columns
from extreme_price_movements.t2_atr_funnel import top_book_metrics
from scripts.run_t2_atr_sequential_funnel import META_CANDIDATES,_add_causal_context,_huber,_score_frame
def matrix(d,raw):return np.column_stack((d[raw].to_numpy(np.float32),d[['side_is_long','causal_entry_cost_bps','score_bps','p_upper','p_lower','p_timeout']].to_numpy(np.float32)))
def context(d,edges):
 x=d.copy();x['week']=pd.to_datetime(x.__ts__,utc=True).dt.to_period('W').astype(str);x['cost_bin']=np.digitize(x.causal_entry_cost_bps,edges[0]);x['vol_bin']=np.digitize(x.atr_percentile,edges[1]);x['opp_bin']=np.digitize(x.score_bps,edges[2]);x['group']=x.side_name.astype(str)+'|'+x.week+'|'+x.cost_bin.astype(str)+'|'+x.vol_bin.astype(str)+'|'+x.opp_bin.astype(str);return x
def metric(d,s,v):
 r=top_book_metrics(_score_frame(d,s,v,'ranking_development','TP2_SL1',.25),score_column='score_bps');r['variant']=v;return r
def main():
 p=argparse.ArgumentParser();p.add_argument('--ledger',type=Path,required=True);p.add_argument('--features-json',type=Path,required=True);p.add_argument('--c4-predictions',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.output.exists():raise FileExistsError(a.output)
 raw=[x for x in META_CANDIDATES if x in validate_feature_columns(json.loads(a.features_json.read_text())['raw_feature_columns'])]
 need={'candidate_id','side_name','__ts__','__symbol__','execution_gross_ev_12h','execution_cost_return','execution_net_ev_12h','atr_percentile'}|set(raw)
 led=_add_causal_context(pd.read_parquet(a.ledger,columns=sorted(need)))
 pred=pd.read_parquet(a.c4_predictions);pred=pred[pred.variant.eq('C4_contract_ensemble')][['candidate_id','score_bps','p_upper','p_lower','p_timeout']]
 d=led.merge(pred,on='candidate_id',validate='one_to_one');d['__ts__']=pd.to_datetime(d.__ts__,utc=True)
 # Apr--May ranker fit; June calibrates rank score to bps; July is evaluation.
 fit=d[d.__ts__<'2024-06-01'];cal=d[(d.__ts__>='2024-06-01')&(d.__ts__<'2024-07-01')];test=d[d.__ts__>='2024-07-01']
 edges=[np.quantile(fit.causal_entry_cost_bps,[.25,.5,.75]),np.quantile(fit.atr_percentile,[.25,.5,.75]),np.quantile(fit.score_bps,[.25,.5,.75])]
 fit,cal,test=[context(z,edges) for z in (fit,cal,test)]
 # R0 uses the identical early fit window, targetting residual common bps.
 target=fit.execution_net_ev_12h.to_numpy()*1e4-fit.score_bps.to_numpy();r0=_huber(matrix(fit,raw),target,matrix(test,raw));
 # R1: pair relevance differs only when at least 100 bps apart. Context groups
 # with no separation are discarded, avoiding uncontrolled all-pair creation.
 groups=[];keep=[]
 for _,g in fit.groupby('group',sort=True):
  y=g.execution_net_ev_12h.to_numpy()*1e4
  rel=np.floor((y-y.min())/100.).astype(int)
  if len(g)>=2 and rel.max()>0:groups.append(len(g));keep.append(g.assign(relevance=rel))
 ranked=pd.concat(keep).sort_values('group');ranker=lgb.LGBMRanker(objective='lambdarank',metric='ndcg',n_estimators=250,learning_rate=.04,num_leaves=15,min_child_samples=30,random_state=20260801,n_jobs=1,verbosity=-1)
 ranker.fit(matrix(ranked,raw),ranked.relevance.to_numpy(),group=ranked.groupby('group',sort=True).size().to_numpy())
 # Causal bps calibration: ranker trained Apr--May, predictions emitted in June.
 cal_rank=ranker.predict(matrix(cal,raw));cal_res=cal.execution_net_ev_12h.to_numpy()*1e4-cal.score_bps.to_numpy();mapper=Ridge(alpha=20.).fit(cal_rank[:,None],cal_res);r1=mapper.predict(ranker.predict(matrix(test,raw))[:,None])
 stage=Path(tempfile.mkdtemp(prefix='.'+a.output.name+'.',dir=a.output.parent))
 try:
  pd.concat((metric(test,test.score_bps.to_numpy(),'R0_base_only'),metric(test,test.score_bps.to_numpy()+r0,'R0_pointwise_residual'),metric(test,test.score_bps.to_numpy()+r1,'R1_rank_correction'))).to_parquet(stage/'ranking_loss_ablation.parquet',index=False)
  (stage/'ranking_pair_manifest.json').write_text(json.dumps({'pair_context':['side','UTC week','cost quartile','ATR percentile quartile','base opportunity quartile'],'minimum_net_separation_bps':100,'rank_train':'Apr-May 2024','calibration':'June 2024 rank score to residual bps','evaluation':'July 2024','pooled_global_final_ranking':True,'ranked_train_rows':len(ranked)},indent=2)+'\n');os.replace(stage,a.output)
 except Exception:
  import shutil;shutil.rmtree(stage,ignore_errors=True);raise
if __name__=='__main__':main()
