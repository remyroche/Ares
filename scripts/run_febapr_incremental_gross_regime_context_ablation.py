#!/usr/bin/env python3
"""Bounded March-development/April-holdout incremental context ablation."""
from __future__ import annotations
import hashlib,json,os,tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import Ridge

ROOT=Path(__file__).resolve().parents[1]
RES=ROOT/'data_perp/artifacts/febapr2025_canonical_residual_oof_20260727_v1/oof_predictions.parquet'
CTX=ROOT/'data_perp/artifacts/febapr2025_strict_residual_gross_regime_context_20260729_v3/panel.parquet'
POP=ROOT/'data_perp/artifacts/febapr2025_canonical_exact_policy_base_population_20260727_v2/population.parquet'
AUX=ROOT/'data_perp/artifacts/febapr2025_historical_auxiliary_oof_20260729_v2/oof_predictions.parquet'
RISK=ROOT/'data_perp/artifacts/febapr2025_historical_competing_risk_catboost_20260729_v2'
DESIGN=ROOT/'data_perp/artifacts/febapr2025_gross_regime_context_incremental_ablation_design_20260729_v1/design.json'
OUT=ROOT/'data_perp/artifacts/febapr2025_incremental_gross_regime_context_ablation_20260729_v3'
ID=['candidate_id','side_name','__symbol__','__ts__']

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def top(x,c):
 return x.sort_values([c,'candidate_id'],ascending=[False,True],kind='stable').head(max(1,int(np.ceil(len(x)*.1))))
def prep(a,b,fs):
 med=a[fs].median().fillna(0); xa=a[fs].fillna(med);xb=b[fs].fillna(med);sd=xa.std().replace(0,1).fillna(1);return (xa-xa.mean())/sd,(xb-xa.mean())/sd
def map_causal(inner,test):
 out=pd.Series(index=test.index,dtype=float)
 for day in sorted(test.__ts__.dt.floor('D').unique()):
  m=test.__ts__.dt.floor('D').eq(day);h=inner[(inner.__ts__<day)&(inner.execution_label_end_utc<day)&(inner.__ts__>=day-pd.Timedelta(days=21))]
  if len(h)<300 or h.raw_score.nunique()<2:out.loc[m]=test.loc[m,'raw_score']
  else:out.loc[m]=IsotonicRegression(out_of_bounds='clip').fit(h.raw_score,h.execution_net_ev_12h).predict(test.loc[m,'raw_score'])
 return out
def score(x):
 return {'rows':len(x),'gross_bps':float(x.execution_gross_ev_12h.mean()*1e4),'cost_bps':float(x.execution_cost_return.mean()*1e4),'net_bps':float(x.execution_net_ev_12h.mean()*1e4),'positive_rate':float(x.execution_net_ev_12h.gt(0).mean())}
def run_side(train,test,features):
 cut=pd.Timestamp('2025-03-21T00:00:00Z');fit=train[(train.__ts__<cut)&(train.execution_label_end_utc<cut)].copy();val=train[train.__ts__>=cut].copy()
 # Fold-local FS: only rows resolving before the validation boundary rank fields.
 rank=fit[features].corrwith(fit.execution_net_ev_12h,method='spearman').abs().fillna(0).sort_values(ascending=False)
 fs=rank.head(min(max(4,int(np.ceil(len(features)*.75))),len(features))).index.tolist();best=None
 for alpha in (.1,1.,10.,100.):
  xa,xv=prep(fit,val,fs);m=Ridge(alpha=alpha).fit(xa,fit.execution_net_ev_12h);loss=float(np.mean((m.predict(xv)-val.execution_net_ev_12h)**2));best=min(best,(loss,alpha)) if best else (loss,alpha)
 xv0,xv=prep(fit,val,fs);m=Ridge(alpha=best[1]).fit(xv0,fit.execution_net_ev_12h);inner=val[ID+['execution_label_end_utc','execution_net_ev_12h']].copy();inner['raw_score']=m.predict(xv)
 xa,xb=prep(train,test,fs);m=Ridge(alpha=best[1]).fit(xa,train.execution_net_ev_12h);outer=test.copy();outer['raw_score']=m.predict(xb)
 return inner,outer,{'selected_features':fs,'alpha':best[1],'inner_mse':best[0],'purge':'fit execution_label_end_utc < 2025-03-21T00:00:00Z'}
def main():
 if OUT.exists():raise FileExistsError(OUT)
 design=json.loads(DESIGN.read_text());groups={x['name']:x['fields'] for x in design['incremental_context_groups']}
 r=pd.read_parquet(RES);r=r[r.residual_is_oof].copy();r=r[ID+['__decision_ts__','execution_label_end_utc','base_oof_score','base_expected_ev','residual_expected_ev','residual_delta_ev']]
 p=pd.read_parquet(POP,columns=['candidate_id','execution_gross_ev_12h','execution_cost_return','execution_net_ev_12h']);x=r.merge(p,on='candidate_id',validate='one_to_one')
 c=pd.read_parquet(CTX);x=x.merge(c.drop(columns=['__decision_ts__']),on=ID,validate='one_to_one')
 a=pd.read_parquet(AUX,columns=ID+['pred_peak_mfe_12h_atr__p_hit','pred_peak_mfe_12h_atr__conditional_mean']);x=x.merge(a,on=ID,validate='one_to_one')
 rr=[]
 for side in ('long','short'):rr.append(pd.read_parquet(RISK/side/'oof.parquet',columns=ID+['prob_timeout','prob_adverse_first_or_conflict','prob_favorable_first']))
 x=x.merge(pd.concat(rr),on=ID,validate='one_to_one');x.__ts__=pd.to_datetime(x.__ts__,utc=True);x.execution_label_end_utc=pd.to_datetime(x.execution_label_end_utc,utc=True);x['month']=x.__ts__.dt.strftime('%Y-%m')
 if len(x)!=140682 or set(x.month)!={'2025-03','2025-04'} or not np.allclose(x.execution_gross_ev_12h-x.execution_cost_return,x.execution_net_ev_12h,atol=1e-12,rtol=0):raise ValueError('source contract fails')
 core=['base_oof_score','base_expected_ev','residual_expected_ev','residual_delta_ev'];risk=['prob_timeout','prob_adverse_first_or_conflict','prob_favorable_first'];peak=['pred_peak_mfe_12h_atr__p_hit','pred_peak_mfe_12h_atr__conditional_mean']
 arms={'A0_base_residual':core,'A1_static_core':core+groups['static_core'],'A2_static_regime':core+groups['static_core']+groups['static_regime'],'A3_core_3h':core+groups['static_core']+groups['static_regime']+groups['transition_core_3h'],'A4_core_12h':core+groups['static_core']+groups['static_regime']+groups['transition_core_12h']}
 # Predeclared March-inner gate chooses the strongest context arm before risk/peak additions.
 allrows=[];ledgers={}
 for name,fs in arms.items():
  inn=[];out=[];contract={}
  for side in ('long','short'):
   tr=x[(x.month=='2025-03')&(x.side_name==side)];te=x[(x.month=='2025-04')&(x.side_name==side)];i,o,q=run_side(tr,te,fs);inn.append(i);out.append(o);contract[side]=q
  inn=pd.concat(inn);out=pd.concat(out);out['mapped_score']=map_causal(inn,out);sel=top(out,'mapped_score');allrows.append({'arm':name,'stage':'context','features':fs,'contract':contract,'inner_mse':float(np.mean([q['inner_mse'] for q in contract.values()])),'april_pooled_global_top10':score(sel)});ledgers[name]=(inn,out)
 winner=min([r for r in allrows if r['stage']=='context'],key=lambda q:q['inner_mse'])['arm'];winner_features=arms[winner]+groups['transition_regime']
 for name,fs in {'A5_transition_regime':winner_features,'B1_plus_risk':winner_features+risk,'B2_plus_peak':winner_features+peak,'B3_plus_risk_peak':winner_features+risk+peak}.items():
  inn=[];out=[];contract={}
  for side in ('long','short'):
   tr=x[(x.month=='2025-03')&(x.side_name==side)];te=x[(x.month=='2025-04')&(x.side_name==side)];i,o,q=run_side(tr,te,fs);inn.append(i);out.append(o);contract[side]=q
  inn=pd.concat(inn);out=pd.concat(out);out['mapped_score']=map_causal(inn,out);sel=top(out,'mapped_score');allrows.append({'arm':name,'stage':'risk_peak','features':fs,'contract':contract,'inner_mse':float(np.mean([q['inner_mse'] for q in contract.values()])),'april_pooled_global_top10':score(sel)});ledgers[name]=(inn,out)
 temp=Path(tempfile.mkdtemp(dir=OUT.parent,prefix=f'.{OUT.name}.'))
 rows=[]
 for r0 in allrows:
  arm=r0['arm'];i,o=ledgers[arm];i.to_parquet(temp/f'{arm}_march_inner_oof.parquet',index=False,compression='zstd');o[ID+['raw_score','mapped_score','execution_gross_ev_12h','execution_cost_return','execution_net_ev_12h']].to_parquet(temp/f'{arm}_april_predictions.parquet',index=False,compression='zstd');rows.append(r0)
 (temp/'report.json').write_text(json.dumps(rows,default=str,indent=2)+'\n');output_hashes={q.name:sha(q) for q in sorted(temp.glob('*.parquet'))};output_hashes['report.json']=sha(temp/'report.json')
 positive=[r for r in rows if r['april_pooled_global_top10']['net_bps']>0 and r['april_pooled_global_top10']['rows']>=6900]
 manifest={'schema':'febapr_incremental_gross_regime_context_ablation_v1','status':'RESEARCH_ONLY_NO_PORTFOLIO' if not positive else 'RESEARCH_ONLY_PORTFOLIO_ELIGIBLE','rows':len(x),'winner_from_context_inner_mse':winner,'selection':'April one pooled global top10 after causal map; never per timestamp','holdout':'April untouched by March-only fold-local FS/HPO and map fit','sources_sha256':{str(q):sha(q) for q in (RES,CTX,POP,AUX,DESIGN)},'risk_sources_sha256':{str(RISK/s/'oof.parquet'):sha(RISK/s/'oof.parquet') for s in ('long','short')},'runner_sha256':sha(Path(__file__)),'outputs_sha256':output_hashes,'portfolio_replay':'NOT_RUN' if not positive else 'ELIGIBLE_BUT_NOT_RUN'}
 (temp/'manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n');(temp/'manifest.sha256').write_text(f'{sha(temp/"manifest.json")}  manifest.json\n');os.replace(temp,OUT)
if __name__=='__main__':main()
