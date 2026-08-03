#!/usr/bin/env python3
"""Native-only old-24h versus retrained-12h score comparison on identical rows."""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
OLD_SCORE=ROOT/'data_perp/artifacts/febapr2025_canonical_base_oof_20260727_v1/oof_predictions.parquet';FEB=ROOT/'data_perp/artifacts/feb2025_native12h_base_oof_20260729_v1/oof_predictions.parquet';MA=ROOT/'data_perp/artifacts/febapr2025_native12h_partial_marapr_base_oof_20260729_v1/oof_predictions.parquet';OLD=ROOT/'data_perp/artifacts/20260720_s59_h5_signalclose_causal_trailing_cost100bps_labels_v2/labels';NEW=ROOT/'data_perp/artifacts/febapr2025_native_first_touch_full_12h_labels_20260729_v1';OUT=ROOT/'data_perp/artifacts/febapr2025_native12h_matched_score_divergence_20260729_v1'
def ic(x,s,t):
 return {'pearson':float(x[s].corr(x[t])),'spearman':float(x[s].corr(x[t],method='spearman')),'timestamp_local_spearman':float(x.groupby('__ts__').apply(lambda g:g[s].corr(g[t],method='spearman'),include_groups=False).mean()),'symbol_neutral_spearman':float((x[s]-x.groupby('__symbol__')[s].transform('mean')).corr(x[t]-x.groupby('__symbol__')[t].transform('mean'),method='spearman'))}
def tail(x,s,net):
 k=max(1,int(len(x)*.1));p=x.nlargest(k,s);o=x.nlargest(k,net);hit=set(p.candidate_id)&set(o.candidate_id)
 return {'rows':len(x),'top10_rows':k,'gross_mean':float((p[net]+.01).mean()),'cost_mean':.01,'net_mean':float(p[net].mean()),'positive_net_precision':float((p[net]>0).mean()),'oracle_tail_overlap_rate':len(hit)/k,'oracle_tail_recall':len(hit)/k}
def metrics(x):
 out={}
 for score in ('old_score','new_score'):
  out[score]={'ic_vs_native_24h':ic(x,score,'target24'),'ic_vs_native_12h':ic(x,score,'target12'),'native_24h_top10':tail(x,score,'net24'),'native_12h_top10':tail(x,score,'net12')}
 return out
def main():
 oldscore=pd.read_parquet(OLD_SCORE,columns=['candidate_id','base_oof_score']).rename(columns={'base_oof_score':'old_score'});newscore=pd.concat([pd.read_parquet(FEB,columns=['candidate_id','side_name','__symbol__','__ts__','base_oof_score']).rename(columns={'base_oof_score':'new_score'}),pd.read_parquet(MA,columns=['candidate_id','side_name','__symbol__','__ts__','base_oof_score']).rename(columns={'base_oof_score':'new_score'})],ignore_index=True)
 old=pd.concat([pd.read_parquet(OLD/f'train_global_{s}_5_2025_{m:02d}.parquet',columns=['candidate_id','__first_touch_target_soft__','__first_touch_capture_net__']) for m in (2,3,4) for s in ('long','short')],ignore_index=True).rename(columns={'__first_touch_target_soft__':'target24','__first_touch_capture_net__':'net24'})
 new=pd.concat([pd.read_parquet(p,columns=['candidate_id','__native_12h_first_touch_target_soft__','__native_12h_first_touch_capture_net__']) for p in sorted((NEW/'shards').glob('*_labels.parquet'))],ignore_index=True).rename(columns={'__native_12h_first_touch_target_soft__':'target12','__native_12h_first_touch_capture_net__':'net12'})
 x=newscore.merge(oldscore,on='candidate_id',how='inner',validate='one_to_one').merge(old,on='candidate_id',how='inner',validate='one_to_one').merge(new,on='candidate_id',how='inner',validate='one_to_one');x['__ts__']=pd.to_datetime(x.__ts__,utc=True);x['month']=x.__ts__.dt.strftime('%Y-%m')
 OUT.mkdir();x.to_parquet(OUT/'identical_rows.parquet',index=False,compression='zstd');result={'schema':'native12h_retrain_vs_24h_score_divergence_v1','scope':'identical Feb-Apr rows; native targets/capture only; execution EV not joined','rows':len(x),'overall':metrics(x),'by_month_side':[{**{'month':m,'side':s},**metrics(g)} for (m,s),g in x.groupby(['month','side_name'],sort=True)]};(OUT/'report.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
