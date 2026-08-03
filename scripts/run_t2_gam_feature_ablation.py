#!/usr/bin/env python3
"""G0/G1 structural spline baseline as strict OOF features for T2."""
from __future__ import annotations
import argparse,json,os,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import numpy as np,pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer,StandardScaler
from extreme_price_movements.feature_provenance_gate import validate_feature_columns
from extreme_price_movements.t2_atr_funnel import BarrierGeometry,soft_event_targets,top_book_metrics
from scripts.run_t2_atr_sequential_funnel import _add_causal_context,_conditional_mean,_huber,_score_frame
CORE=("side_is_long","causal_entry_cost_bps","atr_percentile","atr_change_rate","rv_24h","mkt_rv_24h","mkt_rv_ratio_1h_24h","market_breadth_1h","market_breadth_24h","market_dispersion_4h","volume_percentile","amihud_z","liquidity_ratio_peer_resid","ob_spread_z_24h","mkt_oi_chg_4h","mkt_funding_dispersion","mkt_ret_4h","trend_strength_percentile","loc_session_pos_24","loc_prev_day_range_pos_24")
def lab(base,event):
 d=base.merge(event,on='candidate_id',how='left',validate='one_to_one');d[['t2_upper_soft','t2_lower_soft','t2_timeout_soft']]=soft_event_targets(d,BarrierGeometry(2,1),temperature_atr=.25);return d
def gam_fit(x,y,z):
 return np.column_stack([make_pipeline(SimpleImputer(),StandardScaler(),SplineTransformer(n_knots=4,degree=2),Ridge(alpha=20.)).fit(x,y[:,j]).predict(z) for j in range(3)])
def base_fit(tr,te,raw,extra=0):
 cols=raw+['side_is_long','causal_entry_cost_bps']+([f'gam_{j}' for j in range(3)] if extra else [])
 x,z=tr[cols].to_numpy(np.float32),te[cols].to_numpy(np.float32);y=tr[['t2_upper_soft','t2_lower_soft','t2_timeout_soft']].to_numpy(float)
 p=np.column_stack([np.maximum(_huber(x,y[:,j],z),0) for j in range(3)]);p/=np.maximum(p.sum(1,keepdims=True),1e-8);means=np.array([_conditional_mean(y[:,j],tr.execution_net_ev_12h.to_numpy(float)) for j in range(3)])*1e4;return p@means,p
def metric(f,s,v):
 r=top_book_metrics(_score_frame(f,s,v,'gam_development','TP2_SL1',.25),score_column='score_bps');r['variant']=v;return r
def main():
 p=argparse.ArgumentParser();p.add_argument('--ledger',type=Path,required=True);p.add_argument('--features-json',type=Path,required=True);p.add_argument('--events',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.output.exists():raise FileExistsError(a.output)
 raw=list(validate_feature_columns(json.loads(a.features_json.read_text())['raw_feature_columns']));req={'candidate_id','side_name','__ts__','__symbol__','execution_gross_ev_12h','execution_cost_return','execution_net_ev_12h','oof_fold'}
 d=_add_causal_context(pd.read_parquet(a.ledger,columns=sorted(req|set(raw))));d=lab(d,pd.read_parquet(a.events));tr=d[d.oof_fold.eq('base_train')].sort_values('__ts__').reset_index(drop=True);te=d[d.oof_fold.eq('meta_train')].copy();core=[x for x in CORE if x in tr]
 if len(core)<15:raise ValueError('insufficient core fields')
 # Strict chronological GAM OOF: the first inner block is warmup and excluded
 # from the nonlinear base fit rather than given an in-sample structural score.
 blocks=np.array_split(np.arange(len(tr)),3);usable=np.concatenate(blocks[1:]);oof=np.full((len(tr),3),np.nan)
 for i in (1,2):oof[blocks[i]]=gam_fit(tr.loc[np.concatenate(blocks[:i]),core],tr.loc[np.concatenate(blocks[:i]),['t2_upper_soft','t2_lower_soft','t2_timeout_soft']].to_numpy(),tr.loc[blocks[i],core])
 te_gam=gam_fit(tr[core],tr[['t2_upper_soft','t2_lower_soft','t2_timeout_soft']].to_numpy(),te[core])
 tr_u=tr.loc[usable].copy();tr_u[[f'gam_{j}' for j in range(3)]]=oof[usable];te[[f'gam_{j}' for j in range(3)]]=te_gam
 g0,_=base_fit(tr_u,te,raw,0);g1,pred=base_fit(tr_u,te,raw,1)
 stage=Path(tempfile.mkdtemp(prefix='.'+a.output.name+'.',dir=a.output.parent))
 try:
  pd.concat((metric(te,g0,'G0_no_gam'),metric(te,g1,'G1_gam_feature'))).to_parquet(stage/'gam_residualization_ablation.parquet',index=False)
  pd.DataFrame({'candidate_id':tr.candidate_id,'gam_p_upper':oof[:,0],'gam_p_lower':oof[:,1],'gam_p_timeout':oof[:,2],'gam_strict_oof':np.isfinite(oof).all(1)}).to_parquet(stage/'gam_oof_predictions.parquet',index=False)
  (stage/'gam_feature_manifest.json').write_text(json.dumps({'model':'regularized spline regression, one-versus-rest soft probabilities','core_features':core,'inner_protocol':'three chronological blocks; warmup excluded from nonlinear base fit','base_rows_after_oof_warmup':len(tr_u),'final_oos_untouched':True},indent=2)+'\n');os.replace(stage,a.output)
 except Exception:
  import shutil;shutil.rmtree(stage,ignore_errors=True);raise
if __name__=='__main__':main()
