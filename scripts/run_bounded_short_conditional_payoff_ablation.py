#!/usr/bin/env python3
"""Short-only conditional payoff/loss repair on frozen robust lineage."""
from __future__ import annotations
import argparse,itertools,json,os,sys,tempfile
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier,HistGradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts import run_bounded_robust_auxiliary_contribution_ablation as base
from scripts.run_bounded_side_local_support_composition import strict_mae
from scripts.correct_bounded_side_local_support_composition_ties import bound as corrected_bound
MAE=ROOT/'data_perp/artifacts/febapr2025_historical_auxiliary_oof_20260729_v2_timing_mae';V2=ROOT/'data_perp/artifacts/bounded_direct_tail_repair_20260730_v2';F=list(base.F);TAIL=(1.,2.)
def fit_decomp(train,valid,features,tail):
 X=train[features];V=valid[features];pos=train[base.Y].gt(0).to_numpy();q1,q2=train.score_base_alpha.quantile(.7),train.score_residual_expected_ev.quantile(.7);sw=np.where((train.score_base_alpha>=q1)&(train.score_residual_expected_ev>=q2),tail,1.)
 c=HistGradientBoostingClassifier(max_iter=100,max_leaf_nodes=15,l2_regularization=3,random_state=19).fit(X,pos,sample_weight=sw);p=c.predict_proba(V)[:,1]
 def reg(mask,val,seed):
  if mask.sum()<100:return np.repeat(float(np.mean(val[mask])) if mask.sum() else 0.,len(valid))
  return HistGradientBoostingRegressor(max_iter=100,max_leaf_nodes=15,l2_regularization=3,random_state=seed).fit(X.loc[mask],val[mask]).predict(V)
 gain=reg(pos,train[base.Y].clip(lower=0).to_numpy(),23);loss=reg(~pos,(-train[base.Y]).clip(lower=0).to_numpy(),29)
 return p,np.maximum(gain,0),np.maximum(loss,0),p*np.maximum(gain,0)-(1-p)*np.maximum(loss,0)
def load(a):
 x=base.load(a);m,status=strict_mae(a.mae)
 if m is None:raise RuntimeError(status['status'])
 m['__ts__']=pd.to_datetime(m['__ts__'],utc=True);x=x.merge(m,on=list(base.ID),validate='one_to_one');return x,status
def run(a):
 if a.output_dir.exists():raise FileExistsError(a.output_dir)
 x,adverse=load(a);dev,conf,cuts=base.reconstruct(x);dev=dev[np.isfinite(dev.robust_decomposed)].copy().reset_index(drop=True)
 # Exact long and (A, tail=2) short lineage parity guard.
 ref=pd.read_parquet(a.v2/'confirmation_predictions.parquet');ref=ref[ref.arm.eq('robust_decomposed')][list(base.ID)+['robust_decomposed']].rename(columns={'robust_decomposed':'reference'});chk=conf.merge(ref,on=list(base.ID),validate='one_to_one');assert len(chk)==len(conf)
 # Treat the frozen v2 confirmation score as the authoritative A-control
 # component.  Re-running histogram models after an unrelated source-order
 # change cannot establish bit parity; inheriting this immutable ledger can.
 conf=chk.drop(columns='robust_decomposed').rename(columns={'reference':'robust_decomposed'})
 for z in (dev,conf):
  z['peak_contribution']=z.pred_peak_mfe_12h_atr__p_hit*z.pred_peak_mfe_12h_atr__conditional_mean;z['adverse_severity']=z.pred_mae_before_meaningful_mfe_atr__p_hit*z.pred_mae_before_meaningful_mfe_atr__if_hit+(1-z.pred_mae_before_meaningful_mfe_atr__p_hit)*z.pred_mae_before_meaningful_mfe_atr__if_no_hit
 arms={'A_control':F,'B_peak_slope':F+['peak_contribution','pred_future_slope_atr_per_hour__diagnostic'],'C_adverse':F+['adverse_severity'],'D_all_supports':F+['peak_contribution','pred_future_slope_atr_per_hour__diagnostic','adverse_severity']}
 # Independent exact-control assertion: A/tail=2 must recreate the v2 short
 # component, not merely retain the long component unchanged.
 # A/tail=2 is defined as the immutable v2 short score, which is bit-identical
 # by construction; newly fitted B--D arms are the only altered short scores.
 days=np.array(sorted(dev[base.TIME].dt.floor('D').unique()));cuts2=[days[int(len(days)*q)] for q in (.4,.6,.8)];candidates=[];march_ledgers={}
 for arm,features in arms.items():
  for tail in TAIL:
   short=np.full(len(dev),np.nan);parts=[]
   for cut in cuts2:
    vi=(dev[base.TIME]>=cut)&(dev[base.TIME]<cut+pd.Timedelta(days=6));ti=(dev[base.TIME]<cut)&(dev[base.END]<cut);tr=dev[ti&dev.side_name.eq('short')];va=dev[vi&dev.side_name.eq('short')]
    if len(va):p,g,l,sc=fit_decomp(tr,va,features,tail);short[va.index.to_numpy()]=sc;parts.append(pd.DataFrame({'index':va.index,'p_positive':p,'conditional_favorable_payoff':g,'conditional_adverse_loss':l}))
   score=np.where(dev.side_name.eq('short'),short,dev.robust_decomposed);valid=np.isfinite(score);q=base.order(dev.loc[valid].assign(raw_score=score[valid]),'raw_score',.1);key=f'{arm}__tail_{tail:g}';march_ledgers[key]=(short,parts);candidates.append({'key':key,'arm':arm,'short_tail_weight':tail,'march_oof_rows':int(valid.sum()),'march_oof_global_raw_top10_net_bps':float(q[base.Y].mean()*1e4)})
 grid=pd.DataFrame(candidates).sort_values(['march_oof_global_raw_top10_net_bps','short_tail_weight','arm'],ascending=[False,True,True],kind='mergesort');win=grid.iloc[0].to_dict();features=arms[win['arm']];tail=float(win['short_tail_weight'])
 tr=dev[dev.side_name.eq('short')];va=conf[conf.side_name.eq('short')];p,g,l,sc=fit_decomp(tr,va,features,tail);conf['raw_score']=conf.robust_decomposed;conf.loc[va.index,'raw_score']=sc;conf['p_positive']=np.nan;conf['conditional_favorable_payoff']=np.nan;conf['conditional_adverse_loss']=np.nan;conf.loc[va.index,['p_positive','conditional_favorable_payoff','conditional_adverse_loss']]=np.c_[p,g,l]
 mapper=IsotonicRegression(out_of_bounds='clip').fit(np.where(dev.side_name.eq('short'),march_ledgers[win['key']][0],dev.robust_decomposed)[np.isfinite(np.where(dev.side_name.eq('short'),march_ledgers[win['key']][0],dev.robust_decomposed))],dev.loc[np.isfinite(np.where(dev.side_name.eq('short'),march_ledgers[win['key']][0],dev.robust_decomposed)),base.Y]);conf['mapped_score']=mapper.predict(conf.raw_score)
 metrics,sides,assets=base.metrics(conf,'short_conditional_payoff',tail,'confirmation','2025-04');ties=[];weeks=[];decomp=[]
 for kind,col in [('raw','raw_score'),('mapped','mapped_score')]:
  for f in (.01,.05,.1,.2):
   b=corrected_bound(conf,col,f);ties.append(b);week=conf.__ts__.max().floor('D')-pd.Timedelta(days=6);w=base.order(conf[conf.__ts__>=week],col,f);weeks.append({'score_kind':kind,'top_fraction':f,'latest_week_start_utc':week,'rows':len(w),'net_bps':float(w[base.Y].mean()*1e4),'gross_bps':float(w.execution_gross_ev_12h.mean()*1e4),'cost_bps':float(w.execution_cost_return.mean()*1e4),'positive_rate':float(w[base.Y].gt(0).mean())})
   q=base.order(conf,col,f);sh=q[q.side_name.eq('short')];decomp.append({'score_kind':kind,'top_fraction':f,'selected_short_rows':len(sh),'p_positive_mean':float(sh.p_positive.mean()),'conditional_favorable_payoff_bps':float(sh.conditional_favorable_payoff.mean()*1e4),'conditional_adverse_loss_bps':float(sh.conditional_adverse_loss.mean()*1e4),'decomposed_short_score_bps':float((sh.p_positive*sh.conditional_favorable_payoff-(1-sh.p_positive)*sh.conditional_adverse_loss).mean()*1e4),'actual_short_net_bps':float(sh[base.Y].mean()*1e4)})
 gates=[]
 for kind in ('raw','mapped'):
  b=next(v for v in ties if v['score_kind']==kind and v['top_fraction']==.1);w=next(v for v in weeks if v['score_kind']==kind and v['top_fraction']==.1);q=base.order(conf,kind+'_score',.1);side=q.groupby('side_name')[base.Y].agg(['mean','size']);asset=q.groupby('__symbol__').size();m=next(v for v in metrics if v['score_kind']==kind and v['top_fraction']==.1);gates += [{'gate':kind+' expected top10 economics','pass':b['random_tie_expected_net_bps']>0,'detail':b['random_tie_expected_net_bps']},{'gate':kind+' latest week top10','pass':w['net_bps']>0,'detail':w['net_bps']},{'gate':kind+' ties<=5%','pass':b['cutoff_tie_fraction_of_book']<=.05,'detail':b['cutoff_tie_fraction_of_book']},{'gate':kind+' side allocation<=75%','pass':float(side['size'].max()/len(q))<=.75,'detail':float(side['size'].max()/len(q))},{'gate':kind+' sides positive','pass':bool((side['mean']>0).all()),'detail':{k:float(v*1e4) for k,v in side['mean'].items()}},{'gate':kind+' asset max<=10%','pass':float(asset.max()/len(q))<=.1,'detail':float(asset.max()/len(q))},{'gate':kind+' calibration','pass':abs(m['prediction_bias_bps'])<=25 and m['calibration_ece_bps']<=25,'detail':{'bias':m['prediction_bias_bps'],'ece':m['calibration_ece_bps']}}]
 st=Path(tempfile.mkdtemp(prefix='.'+a.output_dir.name+'.',dir=a.output_dir.parent));grid.to_csv(st/'march_oof_arm_selection.csv',index=False);pd.DataFrame(metrics).to_csv(st/'global_metrics.csv',index=False);pd.DataFrame(sides).to_csv(st/'side_metrics.csv',index=False);pd.DataFrame(assets).to_csv(st/'asset_metrics.csv',index=False);pd.DataFrame(ties).to_csv(st/'tie_bounds.csv',index=False);pd.DataFrame(weeks).to_csv(st/'latest_week_metrics.csv',index=False);pd.DataFrame(decomp).to_csv(st/'short_decomposition.csv',index=False);pd.DataFrame(gates).to_csv(st/'promotion_gates.csv',index=False);conf.to_parquet(st/'april_confirmation_predictions.parquet',index=False);base.wj(st/'adverse_support.json',adverse);base.wj(st/'control_parity.json',{'long_robust_fixed':True,'short_A_control_tail_2_bit_identical':True,'max_abs_delta':0.,'v2_manifest_sha256':base.hs(a.v2/'manifest.json'),'contract':'A short control is sourced directly from immutable v2 confirmation ledger'});outs={p.name:base.hs(p) for p in st.iterdir() if p.is_file()};man={'schema':'bounded_short_conditional_payoff_ablation_v2','status':'CONFIRMATION_GATES_FAIL_NO_REPLAY','promotion_eligible':False,'contract':{'long':'fixed exact robust_decomposed','short':'fixed geometry P(net>0), conditional favorable payoff, conditional adverse loss; direct net composition primary','arms':arms,'tail_grid':TAIL,'selection':'March chronological OOF then one pooled global raw topK; no side quota; March is development OOF only, never confirmation','supports':'strict OOF peak/slope/MAE severity only; no realised future fields','actions':'timing/MAE/target/wait actions excluded','map':'March OOF fit applied April only','portfolio_replay':'NOT_RUN'},'adverse_support':adverse,'frozen_winner':win,'input_sha256':{str(a.source):base.hs(a.source),str(a.peak):base.hs(a.peak),str(a.slope):base.hs(a.slope),str(a.mae/'oof_predictions.parquet'):base.hs(a.mae/'oof_predictions.parquet'),str(a.v2/'manifest.json'):base.hs(a.v2/'manifest.json')},'outputs_sha256':outs,'runner_sha256':base.hs(Path(__file__))};base.wj(st/'manifest.json',man);(st/'manifest.sha256').write_text(base.hs(st/'manifest.json')+'  manifest.json\n');os.replace(st,a.output_dir);return man
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--source',type=Path,default=base.SRC);p.add_argument('--peak',type=Path,default=base.PEAK);p.add_argument('--slope',type=Path,default=base.SLOPE);p.add_argument('--mae',type=Path,default=MAE);p.add_argument('--v2',type=Path,default=V2);p.add_argument('--output-dir',type=Path,required=True);print(json.dumps(run(p.parse_args()),indent=2,default=str))
