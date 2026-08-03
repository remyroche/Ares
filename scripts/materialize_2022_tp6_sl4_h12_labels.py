#!/usr/bin/env python3
"""Materialise exact TP6/SL4/H12 labels on the stored 2022 causal candidates."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.materialize_exact_geometry_sidecar import _first_touch
from scripts.materialize_full_universe_tp6_sl4_h12_sidecar import _minute_path, ROUND_TRIP_COST_BPS
from scripts.materialize_tp6_sl4_robust_clear_labels import _pre_adverse_mfe

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False);out=a.out/'parts';out.mkdir()
 sources=[ROOT/'data_perp/artifacts/jan_jul_2022_inverse_pi_causal_features_20260730_v3/candidate_shards',ROOT/'data_perp/artifacts/aug2022_inverse_pi_causal_features_20260730_v1/candidate_shards'];rows=[]
 for source in sources:
  for part in sorted(source.glob('*.parquet')):
   x=pd.read_parquet(part);x['__ts__']=pd.to_datetime(x.__ts__,utc=True);x['__decision_ts__']=x.__ts__+pd.Timedelta(hours=1);x['candidate_id']=x.__symbol__.astype(str)+'|'+x.__ts__.astype(str)+'|1h|'+x.side_name.astype(str)
   result=[]
   for old_symbol,g in x.groupby('__symbol__',sort=True):
    symbol=str(old_symbol).replace('/','_');start=g.__decision_ts__.min();end=g.__decision_ts__.max()+pd.Timedelta(hours=12);minute=_minute_path(ROOT/'data_perp/exchanges/krakenfutures/execution_1m/ohlcv',symbol,start,end);starts=minute.index.get_indexer(g.__decision_ts__);entry=np.full(len(g),np.nan);ok=starts>=0;entry[ok]=minute.open.to_numpy(float)[starts[ok]];atr=g.atr_fraction_14h.to_numpy(float)*entry;side=np.where(g.side_name.eq('long'),1.,-1.);event,exit_min,pnl=_first_touch(minute.high.to_numpy(float),minute.low.to_numpy(float),minute.close.to_numpy(float),starts.astype('int64'),entry,atr,side,6.,4.,720)
    finite=np.isfinite(minute[['open','high','low','close']].to_numpy(float)).all(1).astype('int64');cum=np.r_[0,np.cumsum(finite)];complete=np.zeros(len(g),bool);inside=ok&(starts+720<=len(minute));ss=starts[inside];complete[inside]=(cum[ss+720]-cum[ss])==720;valid=complete&np.isfinite(pnl);_,pre_mfe,lower=_pre_adverse_mfe(minute.high.to_numpy(float),minute.low.to_numpy(float),minute.close.to_numpy(float),starts.astype('int64'),entry,atr,side);atr_bps=atr/entry*1e4;margin=pre_mfe*atr_bps-125.;robust=valid&(margin>0);z=g.copy();z['entry_price']=entry;z['atr_1h']=atr;z['label_valid']=valid;z['event']=np.where(valid,event,np.nan);z['gross_bps']=np.where(valid,pnl*atr/entry*1e4,np.nan);z['net_bps']=np.where(valid,z.gross_bps-ROUND_TRIP_COST_BPS,np.nan);z['pre_adverse_mfe_atr']=np.where(valid,pre_mfe,np.nan);z['lower_touch_minute']=np.where(valid,lower,-1);z['robust_clear_event_b25']=np.where(valid,robust,np.nan);z['r3_class']=np.select([robust,valid&(lower>=0)],[2,0],default=1).astype('int8');z['__label_available_at__']=z.__decision_ts__+pd.Timedelta(hours=12);result.append(z)
   z=pd.concat(result,ignore_index=True);dst=out/part.name;z.to_parquet(dst,index=False,compression='zstd');rows.append({'part':part.name,'rows':len(z),'valid':int(z.label_valid.sum())});print(rows[-1],flush=True)
 (a.out/'manifest.json').write_text(json.dumps({'contract':'TP6/SL4/H12; entry signal+1h exact minute open; adverse tie; 100bps fixed cost','sources':[str(x) for x in sources],'parts':rows},indent=2))
if __name__=='__main__':main()
