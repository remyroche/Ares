#!/usr/bin/env python3
"""Bounded CatBoost context continuation of frozen gross-hurdle v2."""
from __future__ import annotations
import argparse,hashlib,json,os,time,sys
from pathlib import Path
import numpy as np,pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parent))
from run_historical_execution_ev_gross_hurdle_decomposition import (_load_frozen_population,_arm_features,_purged_before,_features_by_rank,_fit_score,_common_unit,_causal_map,_online_causal_21d_map,_economics,_spearman,sha256,atomic_json,atomic_parquet,ID,OOF_FOLDS)
BASELINES={"risk_peak_direct_net":("plus_risk_peak","direct_net"),"risk_direct_gross":("plus_risk","direct_gross")}
def choose(train,features,method,seed,threads):
 c=pd.Timestamp(train['__ts__'].quantile(.75));a=_purged_before(train,c);b=train[train['__ts__']>=c]
 best=None;board=[]
 for n,(depth,l2) in enumerate(((4,6.),(6,10.))):
  target=a.execution_net_ev_12h.to_numpy() if method=='direct_net' else a.execution_gross_ev_12h.to_numpy();fs=_features_by_rank(a,features,target,.65)
  score,_=_fit_score(a,b,fs,method,0.,None,depth,l2,seed+n,threads,40);tr,_=_fit_score(a,a,fs,method,0.,None,depth,l2,seed+n,threads,40)
  nb=int(np.ceil(.1*len(b)));na=int(np.ceil(.1*len(a)));ev=float(b.iloc[np.argsort(score)[-nb:]]['execution_net_ev_12h'].mean()*1e4);ic=_spearman(score,b.execution_net_ev_12h);tev=float(a.iloc[np.argsort(tr)[-na:]]['execution_net_ev_12h'].mean()*1e4);obj=ev+25*ic+.05*tev
  board.append({'depth':depth,'l2':l2,'objective':obj,'validation_top10_net_bps':ev,'validation_ic':ic,'training_top10_net_bps':tev,'features':fs})
  if best is None or obj>best['objective']:best=board[-1]
 return best,board
def score(train,ev,features,method,seed,threads):
 h,board=choose(train,features,method,seed,threads);v,p=_fit_score(train,ev,h['features'],method,0.,None,h['depth'],h['l2'],seed+100,threads,120)
 q=ev[ID+['execution_label_end_utc','execution_gross_ev_12h','execution_cost_return','execution_net_ev_12h']].copy();q['raw_score']=v;q['hpo']=json.dumps(h,default=str);return q,board
def main():
 p=argparse.ArgumentParser();p.add_argument('--gate-root',type=Path,required=True);p.add_argument('--context-root',type=Path,required=True);p.add_argument('--output-root',type=Path,required=True);p.add_argument('--threads',type=int,default=4);a=p.parse_args();part=a.output_root.with_name(a.output_root.name+'.partial')
 if a.output_root.exists() or part.exists():raise FileExistsError(a.output_root)
 x,gm=_load_frozen_population(a.gate_root);cm=json.loads((a.context_root/'manifest.json').read_text());panel=a.context_root/'panel.parquet'
 seal=a.context_root/'manifest.sha256'
 if cm.get('status')!='IMMUTABLE_PREENTRY_ONLY_INPUT_PANEL' or sha256(panel)!=cm['outputs_sha256']['panel.parquet'] or not seal.exists():raise ValueError('sealed repaired v3 context required')
 ctx=pd.read_parquet(panel,columns=[*ID,*cm['feature_columns']]);x=x.merge(ctx,on=ID,validate='one_to_one');base=_arm_features(x);core=cm['feature_groups']['core_gross_opportunity'];reg=cm['feature_groups']['archived_regime_source_composites'];d3=[z for z in cm['feature_groups']['past_only_transition_deltas'] if z.endswith('delta_3h')];d12=[z for z in cm['feature_groups']['past_only_transition_deltas'] if z.endswith('delta_12h')]
 groups={'no_context':[],'static_core':core,'compact_static_regime':reg[:10],'exact_3h':d3,'exact_12h':d12,'compact_transition_regime':reg[:8]+d3[:4]+d12[:4]};identity=hashlib.sha256(x[ID].sort_values(ID).to_csv(index=False,lineterminator='\n').encode()).hexdigest()
 if not np.allclose(x.execution_net_ev_12h,x.execution_gross_ev_12h-x.execution_cost_return,atol=1e-10,rtol=0):raise ValueError('economics identity')
 v2=a.gate_root.parent/'historical_execution_ev_gross_hurdle_decomposition_20260729_v2';part.mkdir(parents=True);report={'schema':'historical_execution_ev_catboost_context_continuation_v1','status':'research_only_pending_eligibility','identity_sha256':identity,'context_groups':groups,'contract':'March-purged side-local CatBoost HPO; April untouched; pooled global top10 only.','fit_plan':{'hpo_iterations':40,'refit_iterations':120,'planned_model_fits':len(BASELINES)*len(groups)*2*3*5},'frozen_v2':{'manifest_sha256':sha256(v2/'manifest.json'),'report_sha256':sha256(v2/'report.json')},'arms':{}};hashes={};t=time.monotonic()
 for bn,(family,method) in BASELINES.items():
  for gn,add in groups.items():
   name=f'{bn}__{gn}';features=list(dict.fromkeys(base[family]+add));inner=[];outer=[];hpo={}
   for si,side in enumerate(('long','short')):
    march=x[(x.m=='2025-03')&(x.side_name==side)];april=x[(x.m=='2025-04')&(x.side_name==side)];hpo[side]=[]
    for fi,(ss,ee) in enumerate(OOF_FOLDS):
     s=pd.Timestamp(ss,tz='UTC');e=pd.Timestamp(ee,tz='UTC');q,bo=score(_purged_before(march,s),march[(march['__ts__']>=s)&(march['__ts__']<e)],features,method,10000*si+fi,a.threads);q['fold_start']=s;inner.append(q);hpo[side].append(bo)
    q,bo=score(_purged_before(march,pd.Timestamp('2025-04-01',tz='UTC')),april,features,method,20000+si,a.threads);outer.append(q);hpo[side].append(bo)
   inn,out,rel=_common_unit(pd.concat(inner,ignore_index=True),pd.concat(outer,ignore_index=True));out['march_only_mapped_score']=_causal_map(inn,out);out['online_causal_21d_mapped_score']=_online_causal_21d_map(inn,out)
   d=part/name;d.mkdir();ip=d/'march_inner_oof.parquet';op=d/'april_predictions.parquet';atomic_parquet(ip,inn);atomic_parquet(op,out);hashes[str(ip.relative_to(part))]=sha256(ip);hashes[str(op.relative_to(part))]=sha256(op)
   latest=out[pd.to_datetime(out['__ts__'],utc=True)>=pd.Timestamp('2025-04-24',tz='UTC')]
   report['arms'][name]={'baseline':bn,'context_group':gn,'features_requested':features,'hpo':hpo,'reliability':rel,'april':{'raw':_economics(out,out.raw_score.to_numpy()),'common_unit':_economics(out,out.common_unit_score.to_numpy()),'march_only_map':_economics(out,out.march_only_mapped_score.to_numpy()),'online_causal_21d_map':_economics(out,out.online_causal_21d_mapped_score.to_numpy()),'latest_week_common_unit':_economics(latest,latest.common_unit_score.to_numpy())}}
 eligible=[name for name,item in report['arms'].items() if item['april']['common_unit']['net_bps']>0 and all(abs(float(v['inner_raw_net_pearson']))>=.02 for v in item['reliability'].values())]
 report['status']='research_only_replay_eligible' if eligible else 'research_only_no_portfolio';report['portfolio_eligibility']={'eligible_arms':eligible,'actual_completed_model_fits':report['fit_plan']['planned_model_fits'],'replay_run':False}
 rp=part/'report.json';atomic_json(rp,report);hashes['report.json']=sha256(rp);manifest={'schema':'historical_execution_ev_catboost_context_continuation_manifest_v1','status':report['status'],'runner_sha256':sha256(Path(__file__)),'source_gate_manifest_sha256':sha256(a.gate_root/'manifest.json'),'context_manifest_sha256':sha256(a.context_root/'manifest.json'),'context_detached_seal_sha256':sha256(seal),'output_sha256':hashes,'elapsed_seconds':time.monotonic()-t};atomic_json(part/'manifest.json',manifest);part.replace(a.output_root);print(json.dumps({'output':str(a.output_root),'arms':len(report['arms']),'eligible':eligible}))
if __name__=='__main__':main()
