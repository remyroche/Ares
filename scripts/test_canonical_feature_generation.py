from __future__ import annotations

import pathlib
import sys
import pandas as pd
import numpy as np
sys.path.insert(0, '/Users/remyroche/Documents/Ares')
from extreme_price_movements.features import compute_market_features, add_regime_gates, compute_features_hourly
from extreme_price_movements.config import CFG

ROOT = pathlib.Path('/Users/remyroche/Documents/Ares')
def read_one(symbol):
    p = ROOT/'data_perp/ohlcv'/f'symbol={symbol}'/'year=2026'
    files=sorted(p.glob('*.parquet'))
    if not files: return None
    x=pd.concat([pd.read_parquet(f) for f in files],ignore_index=True)
    x['ts']=pd.to_datetime(x['ts'],utc=True); x=x.drop_duplicates('ts').set_index('ts').sort_index()
    return x.resample('1h').agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum','mark_price':'last','index_price':'last','funding_rate':'last','open_interest':'last','spot_close':'last'}).dropna(subset=['close'])

syms=['AAVE_USDT','ADA_USDT','BTC_USDT','ETH_USDT','SOL_USDT']
frames={s:read_one(s) for s in syms}; frames={s:x for s,x in frames.items() if x is not None}
idx=pd.date_range('2026-01-01','2026-02-01',freq='1h',tz='UTC')
panel={}
for c in ['open','high','low','close','volume','mark_price','index_price','funding_rate','open_interest','spot_close']:
    panel[c]=pd.concat({s:x[c] for s,x in frames.items()},axis=1).reindex(idx)
panel['quote_volume']=panel['close']*panel['volume']
panel['spot_open']=panel['spot_close']; panel['spot_high']=panel['spot_close']; panel['spot_low']=panel['spot_close']
for field in ['best_bid','best_ask','mid','bid_qty_1','ask_qty_1','cum_bid_qty_l10','cum_ask_qty_l10','cum_bid_qty_l20','cum_ask_qty_l20','notional_1h','mean_trade_qty_1h']:
    panel['orderbook_'+field]=pd.concat({s:pd.read_parquet(ROOT/'data_perp/orderbook_hourly'/f'{s.replace("_USDT","_USDC")}.parquet')[field] for s in frames if (ROOT/'data_perp/orderbook_hourly'/f'{s.replace("_USDT","_USDC")}.parquet').exists()},axis=1).reindex(idx)
mkt=compute_market_features(panel,list(frames))
gates=add_regime_gates(mkt, gate_vol_lookback_hours=24*7, gate_trend_thr=0.0)
print('mkt',mkt.shape,'gates',gates.shape)
req=['mark_perp_dislocation','mkt_close_location_1h','xasset_mkt_ob_stress_z_24h','mkt_oi_chg_accel_1h','breadth_recovery_from_6h_min']
cfg=dict(CFG); cfg.update({'atr_n':14,'enable_orderbook_features':False,'enable_orderbook_wall_features':False,'feature_portability_mode':'off','feature_portability_strict':False,'live_lgbm_mask_feature_fast_path_enabled':False})
res=compute_features_hourly(panel,gates,cfg,requested_feature_keys=req)
print('res type',type(res),len(res)); feats=res[0] if isinstance(res,tuple) else res
print('out',len(feats),[k for k in req if k in feats]);
for k in req:
    print(k, type(feats.get(k)), None if k not in feats else feats[k].shape, None if k not in feats else float(np.isfinite(feats[k].to_numpy()).mean()))
