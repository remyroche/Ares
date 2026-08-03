#!/usr/bin/env python3
"""Report native 12h label change relative to the archived 24h target."""
from __future__ import annotations
import json
from pathlib import Path
import sys
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
NEW=ROOT/'data_perp/artifacts/febapr2025_native_first_touch_full_12h_labels_20260729_v1'
OLD=ROOT/'data_perp/artifacts/20260720_s59_h5_signalclose_causal_trailing_cost100bps_labels_v2/labels'
def stats(x:pd.DataFrame)->dict:
 d=x.new-x.old;event=(x.new_hit.ne(x.old_hit)|x.new_stop.ne(x.old_stop)|x.new_timeout.ne(x.old_timeout))
 return {'rows':len(x),'pearson':float(x.new.corr(x.old,method='pearson')),'spearman':float(x.new.corr(x.old,method='spearman')),'mean_abs_change':float(d.abs().mean()),'soft_change_ge_0_10_rate':float((d.abs()>=.10).mean()),'outcome_state_change_rate':float(event.mean()),'new_mean':float(x.new.mean()),'old_mean':float(x.old.mean())}
def main():
 n=pd.concat([pd.read_parquet(p,columns=['candidate_id','side_name','__ts__','__native_12h_first_touch_target_soft__','__native_12h_first_touch_hit__','__native_12h_first_touch_stop__','__native_12h_first_touch_timeout__']) for p in sorted((NEW/'shards').glob('*_labels.parquet'))],ignore_index=True)
 o=pd.concat([pd.read_parquet(OLD/f'train_global_{s}_5_2025_{m:02d}.parquet',columns=['candidate_id','__first_touch_target_soft__','__first_touch_hit__','__first_touch_stop__','__first_touch_timeout__']) for m in (2,3,4) for s in ('long','short')],ignore_index=True)
 x=n.merge(o,on='candidate_id',how='inner',validate='one_to_one').rename(columns={'__native_12h_first_touch_target_soft__':'new','__first_touch_target_soft__':'old','__native_12h_first_touch_hit__':'new_hit','__first_touch_hit__':'old_hit','__native_12h_first_touch_stop__':'new_stop','__first_touch_stop__':'old_stop','__native_12h_first_touch_timeout__':'new_timeout','__first_touch_timeout__':'old_timeout'})
 x['month']=pd.to_datetime(x.__ts__,utc=True).dt.strftime('%Y-%m')
 result={'schema':'native_12h_vs_archived_24h_label_diagnostic_v1','scope':'native labels only; no execution EV evaluation','overall':stats(x),'by_side_month':[{**{'side':str(side),'month':str(month)},**stats(g)} for (side,month),g in x.groupby(['side_name','month'],sort=True)]}
 (NEW/'horizon_change_diagnostic.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
