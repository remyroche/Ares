#!/usr/bin/env python3
"""Resolved H12 full-universe market-dynamics labels; never inference inputs."""
from __future__ import annotations
import argparse, hashlib, json, os
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
H=48; PRE=96; MIN_ASSETS=80

def _sha(p: Path)->str:
 h=hashlib.sha256()
 for q in sorted(p.glob('*_15m.parquet')):
  h.update(q.name.encode()); h.update(str(q.stat().st_size).encode())
 return h.hexdigest()

def _write(path: Path,obj: object)->None:
 fd=os.open(path,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o644)
 with os.fdopen(fd,'w') as f: json.dump(obj,f,indent=2,sort_keys=True,default=str)

def _panel(root: Path, index: pd.DatetimeIndex)->tuple[np.ndarray,np.ndarray]:
 closes=[]; vols=[]
 for p in sorted(root.glob('*_15m.parquet')):
  try:
   # Some historical files persist ``ts`` as an index, others as a physical
   # column.  Treat both forms identically; rejecting either would silently
   # shrink the market universe and bias every cross-sectional target.
   try: x=pd.read_parquet(p,columns=['close','volume','ts'])
   except Exception: x=pd.read_parquet(p,columns=['close','volume'])
   t=pd.to_datetime(x.pop('ts'),utc=True,errors='coerce') if 'ts' in x else pd.to_datetime(x.index,utc=True,errors='coerce')
   x.index=t; x=x.loc[~x.index.duplicated(keep='last')].reindex(index)
   closes.append(pd.to_numeric(x.close,errors='coerce').to_numpy('float32')); vols.append(pd.to_numeric(x.volume,errors='coerce').to_numpy('float32'))
  except Exception: continue
 return np.column_stack(closes),np.column_stack(vols)

def _labels(close: np.ndarray,vol: np.ndarray)->pd.DataFrame:
 ret=np.full_like(close,np.nan); ret[1:]=close[1:]/close[:-1]-1
 valid=np.isfinite(ret); n=valid.sum(1); mret=np.nanmedian(ret,1); breadth=np.nanmean(ret>0,1); disp=np.nanstd(ret,1)
 dollar=close*vol; total=np.nansum(dollar,1); share=dollar/np.maximum(total[:,None],1e-12); hhi=np.nansum(share**2,1)
 rv=np.sqrt(np.nanmean(ret**2,1)); out={}
 for t in range(PRE,len(ret)-H):
  if n[t:t+H].min()<MIN_ASSETS or not np.isfinite(mret[t-PRE:t+H]).all(): continue
  fut=mret[t+1:t+H+1]; pre=mret[t-PRE:t]; trend=np.sign(np.nansum(pre[-16:])); cum=np.cumsum(fut); atr=max(rv[t]*np.sqrt(16),1e-8)
  out[t]={
   'market_label_valid':True,
   'market_trend_continuation_12h':trend*cum[-1]/atr,
   'market_signed_directional_efficiency_12h':trend*cum[-1]/max(np.abs(fut).sum(),1e-8),
   'market_time_to_trend_break_12h':next((i/4 for i,x in enumerate(trend*cum,1) if x<-atr),12.),
   'market_vol_change_12h':np.log(max(np.sqrt(np.mean(fut*fut)),1e-8)/max(rv[t],1e-8)),
   'market_vol_acceleration_12h':np.log(max(np.sqrt(np.mean(fut[24:]**2)),1e-8)/max(np.sqrt(np.mean(fut[:24]**2)),1e-8)),
   'market_breadth_change_12h':float(np.nanmean(breadth[t+1:t+H+1])-breadth[t]),
   'cross_sectional_dispersion_change_12h':np.log(max(np.nanmean(disp[t+1:t+H+1]),1e-8)/max(disp[t],1e-8)),
   'market_turnover_change_12h':np.log(max(np.nanmean(total[t+1:t+H+1]),1e-8)/max(total[t],1e-8)),
   'market_volume_concentration_change_12h':float(np.nanmean(hhi[t+1:t+H+1])-hhi[t]),
   'market_future_max_drawdown_12h':float((np.maximum.accumulate(cum)-cum).max()/atr),
   'market_jump_asymmetry_12h':float((abs(fut.min())-abs(fut.max()))/(abs(fut.min())+abs(fut.max())+1e-8)),
  }
 return pd.DataFrame.from_dict(out,orient='index')

def run(ledger:Path,bars:Path,out:Path)->Path:
 if out.exists(): raise FileExistsError(out)
 x=pd.read_parquet(ledger,columns=['candidate_id','__decision_ts__','side_name']); x['__decision_ts__']=pd.to_datetime(x['__decision_ts__'],utc=True)
 idx=pd.date_range(x.__decision_ts__.min().floor('15min')-pd.Timedelta(hours=24),x.__decision_ts__.max().ceil('15min')+pd.Timedelta(hours=12),freq='15min',tz='UTC')
 close,vol=_panel(bars,idx); labels=_labels(close,vol); labels.index=idx[labels.index]
 z=x.merge(labels,left_on='__decision_ts__',right_index=True,how='left'); z['market_label_available_ts']=z['__decision_ts__']+pd.Timedelta(hours=12)
 valid=z.market_label_valid.fillna(False); cols=[c for c in z if c.startswith('market_') or c.startswith('cross_sectional_')]
 z.loc[~valid,[c for c in cols if c not in ('market_label_valid','market_label_available_ts')]]=np.nan
 out.mkdir(parents=True); z.to_parquet(out/'market_dynamics_labels.parquet',index=False,compression='zstd')
 _write(out/'run_manifest.json',{'schema':'strict_r3_o3v2_market_dynamics_labels_v1','scope':'resolved H12 labels only; prohibited from scoring/inference','ledger':str(ledger),'bars':str(bars),'bars_hash':_sha(bars),'horizon_hours':12,'min_assets':MIN_ASSETS,'valid_rows':int(valid.sum()),'rows':len(z)})
 return out
def main():
 p=argparse.ArgumentParser(); p.add_argument('--ledger',type=Path,required=True);p.add_argument('--bars',type=Path,default=ROOT/'15m_ohlcv_perp');p.add_argument('--out',type=Path,required=True);a=p.parse_args();print(run(a.ledger,a.bars,a.out))
if __name__=='__main__': main()
