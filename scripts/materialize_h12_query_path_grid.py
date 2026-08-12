#!/usr/bin/env python3
"""Materialise H12 path primitives for all predeclared query grades.

One 15-minute pass per symbol yields first favourable/adverse touch times for
ATR thresholds 1..6 and the terminal H12 return.  It is label-only and marks
the coarser OHLC entry/path convention explicitly as a proxy.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from extreme_price_movements.query_path_grid_labels import first_touch_grid_h12, first_touch_grid_horizon, path_extrema_h12, path_extrema_horizon
from scripts.materialize_full_universe_tp6_sl4_h12_sidecar import (
    DEFAULT_PANEL, HORIZON_MINUTES, ONE_MINUTE, _complete_paths, _load_candidates, _minute_path,
)

# Supports the requested 1/1.5/2 ATR grade spacings and every 2..6 ATR
# triple-barrier contract in one path pass.
THRESHOLDS=np.asarray((1.,1.5,2.,3.,4.,4.5,5.,6.),dtype=np.float64)
COST_BPS=100.
# This is the historical, exchange-native coarse source.  It deliberately
# supersedes the shallow recent raw cache: every barrier calculation below is
# performed on its 15-minute bars, never on one-minute data.
FIFTEEN_MINUTE=ROOT/'15m_ohlcv_perp'


def _args() -> argparse.Namespace:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--panel',type=Path,default=DEFAULT_PANEL)
    p.add_argument('--minute-root',type=Path,default=ONE_MINUTE)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--symbol',action='append',default=[])
    return p.parse_args()


def _existing_ok(path: Path) -> bool:
    if not path.exists(): return False
    x=pd.read_parquet(path,columns=['candidate_id','label_valid','terminal_atr'])
    return len(x)>0 and not x.candidate_id.duplicated().any() and x.label_valid.notna().all()


def _fallback_15m(symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Return the historical 15-minute grid for a candidate instrument.

    This is explicitly a coarse proxy: entry is the decision-time 15m open
    and a 12-hour path has 48 bars.  Its provenance is stored per output row.
    """
    stem=symbol.lower().replace('_','')+'_15m.parquet'; path=FIFTEEN_MINUTE/stem
    if not path.exists(): raise FileNotFoundError(f'no historical 15m path for {symbol}')
    # Older downloaded files retain their DatetimeIndex as
    # ``__index_level_0__`` whereas newer files use ``ts``.  Read the small
    # schema first rather than assuming either representation.
    raw=pd.read_parquet(path)
    time_col=next((c for c in ('ts','timestamp','__index_level_0__') if c in raw.columns),None)
    if time_col is None:
        if not isinstance(raw.index,pd.DatetimeIndex):
            raise ValueError(f'15m source lacks timestamp column: {path}')
    else:
        raw=raw.set_index(time_col)
    raw=raw.loc[:,['open','high','low','close']]
    raw.index=pd.to_datetime(raw.index,utc=True); raw=raw[~raw.index.duplicated(keep='last')].sort_index()
    grid=pd.date_range(start.floor('15min'),(end-pd.Timedelta(minutes=15)).floor('15min'),freq='15min',tz='UTC')
    return raw.reindex(grid)


def _part(part: Path, minute_root: Path) -> pd.DataFrame:
    c=_load_candidates(part)
    symbol=str(c.__symbol__.iloc[0]); start=c.__decision_ts__.min(); end=c.__decision_ts__.max()+pd.Timedelta(minutes=HORIZON_MINUTES)
    # The query-ablation contract deliberately uses the available coarse path
    # source, not minute bars.  This avoids changing target semantics merely
    # because a symbol happens to have deeper execution history.
    try:
        minute=_fallback_15m(symbol,start,end); resolution=15; horizon_bars=HORIZON_MINUTES//15; source='proxy_15m'
    except FileNotFoundError:
        out=c[['candidate_id','__ts__','__decision_ts__','__symbol__','side_name']].copy()
        out['label_valid']=False; out['path_resolution_minutes']=np.int8(0); out['path_source']='unavailable'; out['entry_is_proxy']=False
        for name in ('entry_price','atr_bps','terminal_atr','terminal_gross_bps','terminal_net_bps','mfe_atr','mae_atr'): out[name]=np.nan
        for t in THRESHOLDS:
            n=(f'{t:g}').replace('.','p'); out[f'fav_touch_{n}atr_minute']=-1; out[f'adv_touch_{n}atr_minute']=-1
        out['__label_available_at__']=out.__decision_ts__+pd.Timedelta(hours=12)
        return out
    starts=minute.index.get_indexer(c.__decision_ts__)
    if (starts<0).any(): raise ValueError(f'{symbol}: entry minute missing')
    finite=np.isfinite(minute[['open','high','low','close']].to_numpy(float)).all(axis=1).astype(np.int64); cumulative=np.r_[0,np.cumsum(finite)]
    valid=(starts>=0)&(starts+horizon_bars<=len(minute)); valid[valid]=cumulative[starts[valid]+horizon_bars]-cumulative[starts[valid]]==horizon_bars
    entry=np.full(len(c),np.nan,float); entry[valid]=minute.open.to_numpy(float)[starts[valid]]
    stored=c.decision_price.to_numpy(float)
    side=np.where(c.side_name.eq('long').to_numpy(),1.,-1.)
    high=minute.high.to_numpy(float); low=minute.low.to_numpy(float); close=minute.close.to_numpy(float)
    fav,adv,terminal=first_touch_grid_horizon(high,low,close,starts.astype(np.int64),entry,c.atr_1h.to_numpy(float),side,THRESHOLDS,horizon_bars)
    mfe,mae=path_extrema_horizon(high,low,starts.astype(np.int64),entry,c.atr_1h.to_numpy(float),side,horizon_bars)
    label_valid=valid & np.isfinite(terminal)
    atr_bps=c.atr_1h.to_numpy(float)/entry*10_000.
    out=c[['candidate_id','__ts__','__decision_ts__','__symbol__','side_name']].copy()
    out['label_valid']=label_valid
    out['path_resolution_minutes']=np.int8(resolution)
    out['path_source']=source
    out['entry_is_proxy']=resolution!=1
    out['entry_price']=entry.astype(np.float64)
    out['atr_bps']=atr_bps.astype(np.float32)
    out['terminal_atr']=terminal.astype(np.float32)
    out['terminal_gross_bps']=(terminal*atr_bps).astype(np.float32)
    out['terminal_net_bps']=(terminal*atr_bps-COST_BPS).astype(np.float32)
    out['mfe_atr']=mfe.astype(np.float32); out['mae_atr']=mae.astype(np.float32)
    for j,t in enumerate(THRESHOLDS):
        name=(f'{t:g}').replace('.','p')
        out[f'fav_touch_{name}atr_minute']=fav[:,j]
        out[f'adv_touch_{name}atr_minute']=adv[:,j]
    out['__label_available_at__']=out.__decision_ts__+pd.Timedelta(hours=12)
    return out


def main() -> None:
    a=_args(); parts=sorted((a.panel/'parts').glob('*.parquet')); a.out.mkdir(parents=True,exist_ok=True)
    wanted=set(a.symbol); completed=[]
    for part in parts:
        symbol=part.name.removeprefix('symbol=').removesuffix('.parquet')
        if wanted and symbol not in wanted: continue
        target=a.out/f'symbol={symbol}.parquet'
        if a.resume and _existing_ok(target): completed.append(symbol); continue
        frame=_part(part,a.minute_root)
        frame.to_parquet(target,index=False,compression='zstd')
        completed.append(symbol)
        (a.out/'progress.json').write_text(json.dumps({'status':'running','completed_symbols':completed},indent=2)+'\n')
    (a.out/'manifest.json').write_text(json.dumps({'schema':'h12_query_path_grid_v1','thresholds_atr':THRESHOLDS.tolist(),'horizon_minutes':HORIZON_MINUTES,'cost_bps':COST_BPS,'entry':'decision_timestamp_15m_open_proxy','path_resolution_minutes':15,'tie_break':'adverse','label_only':True,'symbols':completed},indent=2)+'\n')
    (a.out/'progress.json').write_text(json.dumps({'status':'complete','completed_symbols':completed},indent=2)+'\n')


if __name__=='__main__': main()
