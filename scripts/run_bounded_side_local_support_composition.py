#!/usr/bin/env python3
"""Bounded side-local support composition over the robust-decomposed lineage."""
from __future__ import annotations
import argparse,itertools,json,math,os,sys,tempfile
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.isotonic import IsotonicRegression
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts import run_bounded_robust_auxiliary_contribution_ablation as base
MAE=ROOT/'data_perp/artifacts/febapr2025_historical_auxiliary_oof_20260729_v2_timing_mae'
V2=ROOT/'data_perp/artifacts/bounded_direct_tail_repair_20260730_v2'
ID=list(base.ID);FRACS=(.01,.05,.1,.2);GRID=(0.,.15)
def bounds(x,col,f):
 q=base.order(x,col,f);n=len(q);cut=float(q[col].iloc[-1]);above=x[x[col]>cut];tie=x[np.isclose(x[col].to_numpy(float),cut,rtol=0,atol=1e-14)];need=n-len(above);a=above[base.Y].to_numpy(float);t=tie[base.Y].to_numpy(float)
 def s(v):return float(v.mean()*1e4),float((v>0).mean())
 exp=np.r_[a,np.repeat(t.mean(),need)];best=np.r_[a,np.sort(t)[-need:]];worst=np.r_[a,np.sort(t)[:need]]
 return {'top_fraction':f,'rows':n,'cutoff':cut,'above_cutoff_rows':len(above),'cutoff_tie_rows':len(tie),'cutoff_tie_fraction_of_book':len(tie)/n,'required_from_tie_rows':need,'deterministic_net_bps':s(q[base.Y].to_numpy(float))[0],'deterministic_precision':s(q[base.Y].to_numpy(float))[1],'random_tie_expected_net_bps':s(exp)[0],'random_tie_expected_precision':s(exp)[1],'best_tie_net_bps':s(best)[0],'best_tie_precision':s(best)[1],'worst_tie_net_bps':s(worst)[0],'worst_tie_precision':s(worst)[1]}
def z(ref,val):
 m=float(ref.median());sd=float(ref.std(ddof=0));sd=sd if np.isfinite(sd) and sd>1e-12 else 1.;return (val.to_numpy(float)-m)/sd,{'median':m,'std':sd}
def strict_mae(root):
 m=json.loads((root/'manifest.json').read_text());q=root/'oof_predictions.parquet'
 roles=m.get('roles',{});need=['mae_before_meaningful_mfe_atr.p_hit','mae_before_meaningful_mfe_atr.if_hit','mae_before_meaningful_mfe_atr.if_no_hit']
 if m.get('status')!='STRICT_SIDE_LOCAL_MARCH_APRIL_AUXILIARY_OOF_COMPLETE' or not q.is_file() or any(k not in roles for k in need):return None,{'status':'FAIL_CLOSED_NO_STRICT_OOF_ADVERSE_SEVERITY'}
 x=pd.read_parquet(q,columns=ID+['pred_mae_before_meaningful_mfe_atr__p_hit','pred_mae_before_meaningful_mfe_atr__if_hit','pred_mae_before_meaningful_mfe_atr__if_no_hit'])
 if len(x)!=140682 or x.duplicated(ID).any() or not np.isfinite(x.drop(columns=ID)).all().all():return None,{'status':'FAIL_CLOSED_ADVERSE_LEDGER_CONTRACT'}
 return x,{'status':'AVAILABLE_STRICT_OOF_ADVERSE_SEVERITY','manifest_sha256':base.hs(root/'manifest.json'),'predictions_sha256':base.hs(q),'roles':need,'semantic':'predicted mixture MAE severity used as ranking risk support; no MAE action'}
def load(a):
 x=base.load(a);mae,status=strict_mae(a.mae)
 if mae is None:return x,status
 mae['__ts__']=pd.to_datetime(mae['__ts__'],utc=True);x=x.merge(mae,on=ID,validate='one_to_one');return x,status
def score(frame,config,scales):
 out=frame.robust_decomposed.to_numpy(float).copy()
 for side in ('long','short'):
  ix=frame.side_name.eq(side).to_numpy();wp,ws,wa=config[side];out[ix]+=scales[side]['base']['std']*(wp*frame.loc[ix,'peak_z'].to_numpy(float)+ws*frame.loc[ix,'slope_z'].to_numpy(float)-wa*frame.loc[ix,'adverse_z'].to_numpy(float))
 return out
def run(a):
 if a.output_dir.exists():raise FileExistsError(a.output_dir)
 x,adverse=load(a)
 if adverse['status'].startswith('FAIL_CLOSED'):
  # The requested peak/slope arms still proceed only when no adverse ledger is
  # available; this repository currently has a valid severity mixture, so this
  # branch is deliberately sealed rather than silently substituted.
  raise RuntimeError(adverse['status'])
 dev,conf,cuts=base.reconstruct(x);dev=dev[np.isfinite(dev.robust_decomposed)].copy()
 prior=pd.read_parquet(a.v2/'confirmation_predictions.parquet');prior=prior[prior.arm.eq('robust_decomposed')][ID+['robust_decomposed']].rename(columns={'robust_decomposed':'reference'})
 chk=conf.merge(prior,on=ID,validate='one_to_one');assert float(np.abs(chk.robust_decomposed-chk.reference).max())==0.
 for z0 in (dev,conf):
  z0['peak']=z0.pred_peak_mfe_12h_atr__p_hit*z0.pred_peak_mfe_12h_atr__conditional_mean
  z0['adverse']=z0.pred_mae_before_meaningful_mfe_atr__p_hit*z0.pred_mae_before_meaningful_mfe_atr__if_hit+(1-z0.pred_mae_before_meaningful_mfe_atr__p_hit)*z0.pred_mae_before_meaningful_mfe_atr__if_no_hit
 scales={}
 for side in ('long','short'):
  d=dev[dev.side_name.eq(side)];c=conf[conf.side_name.eq(side)];scales[side]={}
  for src,dst in [('robust_decomposed','base'),('peak','peak'),('pred_future_slope_atr_per_hour__diagnostic','slope'),('adverse','adverse')]:
   d[dst+'_z'],scales[side][dst]=z(d[src],d[src]);c[dst+'_z'],_=z(d[src],c[src]);dev.loc[d.index,dst+'_z']=d[dst+'_z'];conf.loc[c.index,dst+'_z']=c[dst+'_z']
 configs=[]
 for l in itertools.product(GRID,repeat=3):
  for s in itertools.product(GRID,repeat=3):configs.append({'long':l,'short':s})
 rows=[]
 for i,cfg in enumerate(configs):
  dev['raw_score']=score(dev,cfg,scales);q=base.order(dev,'raw_score',.1);rows.append({'config_id':i,'long_peak_weight':cfg['long'][0],'long_slope_weight':cfg['long'][1],'long_adverse_weight':cfg['long'][2],'short_peak_weight':cfg['short'][0],'short_slope_weight':cfg['short'][1],'short_adverse_weight':cfg['short'][2],'march_oof_global_raw_top10_net_bps':float(q[base.Y].mean()*1e4),'complexity':sum(cfg['long'])+sum(cfg['short'])})
 table=pd.DataFrame(rows).sort_values(['march_oof_global_raw_top10_net_bps','complexity','config_id'],ascending=[False,True,True],kind='mergesort');winner=table.iloc[0].to_dict();cfg=configs[int(winner['config_id'])]
 dev['raw_score']=score(dev,cfg,scales);conf['raw_score']=score(conf,cfg,scales);mapper=IsotonicRegression(out_of_bounds='clip').fit(dev.raw_score,dev[base.Y]);conf['mapped_score']=mapper.predict(conf.raw_score)
 metrics,sides,assets=base.metrics(conf,'side_local_support_composition',0.,'confirmation','2025-04');tie=[];weeks=[]
 for kind,col in [('raw','raw_score'),('mapped','mapped_score')]:
  for f in FRACS:
   b=bounds(conf,col,f);b['score_kind']=kind;tie.append(b);week=conf.__ts__.max().floor('D')-pd.Timedelta(days=6);q=base.order(conf[conf.__ts__>=week],col,f);weeks.append({'score_kind':kind,'top_fraction':f,'latest_week_start_utc':week,'rows':len(q),'net_bps':float(q[base.Y].mean()*1e4),'positive_rate':float(q[base.Y].gt(0).mean())})
 gates=[]
 for kind in ('raw','mapped'):
  q=base.order(conf,kind+'_score',.1);b=next(v for v in tie if v['score_kind']==kind and v['top_fraction']==.1);w=next(v for v in weeks if v['score_kind']==kind and v['top_fraction']==.1);side=q.groupby('side_name').agg(rows=(base.Y,'size'),net=(base.Y,'mean'));asset=q.groupby('__symbol__').size();m=next(v for v in metrics if v['score_kind']==kind and v['top_fraction']==.1)
  gates += [{'gate':kind+' expected top10 economics','pass':b['random_tie_expected_net_bps']>0,'detail':b['random_tie_expected_net_bps']},{'gate':kind+' latest week top10','pass':w['net_bps']>0,'detail':w['net_bps']},{'gate':kind+' ties <=5% book','pass':b['cutoff_tie_fraction_of_book']<=.05,'detail':b['cutoff_tie_fraction_of_book']},{'gate':kind+' materially-selected sides positive','pass':bool((side.loc[side.rows>=.1*len(q),'net']>0).all()),'detail':{k:float(v*1e4) for k,v in side.net.items()}},{'gate':kind+' asset max <=10%','pass':float(asset.max()/len(q))<=.1,'detail':float(asset.max()/len(q))},{'gate':kind+' |bias|/ECE<=25bps','pass':abs(m['prediction_bias_bps'])<=25 and m['calibration_ece_bps']<=25,'detail':{'bias':m['prediction_bias_bps'],'ece':m['calibration_ece_bps']}}]
 st=Path(tempfile.mkdtemp(prefix='.'+a.output_dir.name+'.',dir=a.output_dir.parent));table.to_csv(st/'march_oof_side_local_weight_grid.csv',index=False);pd.DataFrame(metrics).to_csv(st/'global_metrics.csv',index=False);pd.DataFrame(sides).to_csv(st/'side_metrics.csv',index=False);pd.DataFrame(assets).to_csv(st/'asset_metrics.csv',index=False);pd.DataFrame(tie).to_csv(st/'tie_bounds.csv',index=False);pd.DataFrame(weeks).to_csv(st/'latest_week_metrics.csv',index=False);pd.DataFrame(gates).to_csv(st/'promotion_gates.csv',index=False);conf[ID+[base.TIME,base.END,base.Y,'robust_decomposed','raw_score','mapped_score']].to_parquet(st/'april_confirmation_predictions.parquet',index=False);base.wj(st/'adverse_availability.json',adverse);base.wj(st/'control_parity.json',{'bit_identical':True,'max_abs_delta':0.,'v2_manifest_sha256':base.hs(a.v2/'manifest.json')});outs={p.name:base.hs(p) for p in st.iterdir() if p.is_file()};man={'schema':'bounded_side_local_support_composition_v1','status':'CONFIRMATION_GATES_FAIL_NO_REPLAY','promotion_eligible':False,'contract':{'lineage':'exact v2 robust_decomposed, April control parity asserted','selection':'64 predeclared paired side-local weights; March chronological OOF global raw top10 selection; no side quota','weights':{'grid':GRID,'components':'positive peak, positive slope, negative predicted MAE severity'},'map':'fit March OOF only, applied only April','actions':'timing/target-price/wait actions excluded; MAE is predicted severity ranking support only','population':'exact IDs; global top-K stable ties','portfolio_replay':'NOT_RUN'},'adverse_support':adverse,'frozen_winner':winner,'march_oof_blocks_utc':[str(v) for v in cuts],'scales':scales,'input_sha256':{str(a.source):base.hs(a.source),str(a.peak):base.hs(a.peak),str(a.slope):base.hs(a.slope),str(a.mae/'oof_predictions.parquet'):base.hs(a.mae/'oof_predictions.parquet'),str(a.v2/'manifest.json'):base.hs(a.v2/'manifest.json')},'outputs_sha256':outs,'runner_sha256':base.hs(Path(__file__))};base.wj(st/'manifest.json',man);(st/'manifest.sha256').write_text(base.hs(st/'manifest.json')+'  manifest.json\n');os.replace(st,a.output_dir);return man
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--source',type=Path,default=base.SRC);p.add_argument('--peak',type=Path,default=base.PEAK);p.add_argument('--slope',type=Path,default=base.SLOPE);p.add_argument('--mae',type=Path,default=MAE);p.add_argument('--v2',type=Path,default=V2);p.add_argument('--output-dir',type=Path,required=True);print(json.dumps(run(p.parse_args()),indent=2,default=str))
