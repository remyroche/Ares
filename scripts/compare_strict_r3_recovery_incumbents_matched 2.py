#!/usr/bin/env python3
"""Replay recovery incumbents on the exact LambdaRank scored-date support."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts.replay_strict_r3_forward_portfolio import CAUSAL_AUCTION_CURVE,_auction_candidates
from scripts.replay_strict_r3_policy_portfolio_2025_2026 import _run

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--funnel-root',type=Path,required=True);ap.add_argument('--lr-root',type=Path,required=True);ap.add_argument('--out-dir',type=Path,required=True);a=ap.parse_args()
 if a.out_dir.exists():raise FileExistsError(a.out_dir)
 a.out_dir.mkdir(parents=True)
 maps=pd.read_parquet(a.funnel_root/'multiwindow_ev_maps.parquet');maps['__decision_ts__']=pd.to_datetime(maps['__decision_ts__'],utc=True);maps['day']=maps.__decision_ts__.dt.normalize()
 p1=pd.concat([pd.read_parquet(a.lr_root/'h1_oof_predictions.parquet'),pd.read_parquet(a.lr_root/'h1_2026_predictions.parquet')]);p2=pd.concat([pd.read_parquet(a.lr_root/'h2_oof_predictions.parquet'),pd.read_parquet(a.lr_root/'h2_2026_predictions.parquet')])
 support=set(p1.snapshot_utc).intersection(set(p2.snapshot_utc));maps=maps.loc[maps.day.isin(support)].copy()
 expressions={'R21_t50':maps.m21,'R28_t50':maps.m28,'prior_W22_t50':maps.m21+(maps.m28-maps.m21)/7,'prior_W23_t50':maps.m21+2*(maps.m28-maps.m21)/7,'prior_W24_t50':maps.m21+3*(maps.m28-maps.m21)/7,'R21_t40':maps.m21,'R21_t30':maps.m21}
 thresholds={'R21_t50':50.,'R28_t50':50.,'prior_W22_t50':50.,'prior_W23_t50':50.,'prior_W24_t50':50.,'R21_t40':40.,'R21_t30':30.}
 rows=[];monthly=[];weekly=[]
 for arm,ev in expressions.items():
  frame=maps.copy();threshold=thresholds[arm];frame['causal_21d_side_expected_net_bps']=ev;frame['causal_21d_side_admitted_ge_50bps']=ev.ge(threshold);frame['auction_tie_break_score']=frame.final_score
  for year in (2025,2026):
   block=frame.loc[frame.__decision_ts__.dt.year.eq(year)];c=_auction_candidates(block,strategy_prefix='matched_'+arm);d,e,m,s=_run(c,0.,f'{arm}_{year}',initial_wallet=1000.,perp_leverage=7.,margin_slot_wallet_fraction=.10,ev_curve=CAUSAL_AUCTION_CURVE)
   ac=d.loc[d.accepted.fillna(False)].copy();ac['net_bps']=pd.to_numeric(ac.position_net_return,errors='coerce')*10000
   rows.append({'arm':arm,'year':year,'scored_days':block.day.nunique(),'accepted_trades':len(ac),'net_bps_per_trade':ac.net_bps.mean(),'total_net_bps':ac.net_bps.sum(),'trades_per_day':s.get('trades_per_day'),'final_wallet':s.get('final_wallet')});monthly.append(m.assign(arm=arm,year=year,threshold_bps=threshold))
   if len(ac):ac['week']=pd.to_datetime(ac.timestamp,utc=True).dt.to_period('W-SUN').astype(str);weekly.append(ac.groupby('week',as_index=False).agg(trades=('net_bps','size'),net_bps_per_trade=('net_bps','mean'),total_net_bps=('net_bps','sum')).assign(arm=arm,year=year,threshold_bps=threshold))
 pd.DataFrame(rows).to_csv(a.out_dir/'matched_portfolio_summary.csv',index=False);pd.concat(monthly,ignore_index=True).to_csv(a.out_dir/'matched_monthly.csv',index=False);pd.concat(weekly,ignore_index=True).to_csv(a.out_dir/'matched_weekly.csv',index=False)
 (a.out_dir/'run_manifest.json').write_text(json.dumps({'status':'complete','support':'intersection of selected H1/H2 daily predictions','arms':list(expressions),'thresholds':thresholds},indent=2)+'\n')
if __name__=='__main__':main()
