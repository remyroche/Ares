#!/usr/bin/env python3
"""Three-era common-30 regime/transition category stability audit; no gate."""
from __future__ import annotations
import hashlib,json,math,os,shutil,tempfile
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];ART=ROOT/'data_perp/artifacts';J=ART/'july2025_common30_final_base_residual_oof_bridge_20260730_v1';A=ART/'augnov2025_common30_frozen_july_base_residual_oos_bridge_20260730_v1';S=ART/'authoritative_soft_regime_transition_sidecars_20260730_v1';V3=ART/'final_identical_row_regime_stack_gam_ablation_20260730_v3';OUT=ART/'h2_common30_regime_category_performance_stability_20260730_v3'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):
 q=Path(p).with_name('.'+Path(p).name+'.partial');q.write_text(json.dumps(x,indent=2,sort_keys=True,default=str)+'\n');os.replace(q,p)
def join(x,ctx):return x.merge(ctx,left_on='__ts__',right_on='source_utc',how='left',validate='many_to_one').drop(columns='source_utc')
def cat(x):
 r=np.where(x.regime_change_probability_mean>=.10,'change_high',np.where(x.regime_state_age_hours<6,'fresh_lowchange','established_lowchange'))
 t=np.where(x.transition_lgbm_probability>=.50,'transition_high','transition_low')
 o=[]
 for layer,v in [('regime',r),('transition',t),('combined',np.char.add(np.char.add(r,'|'),t))]:
  z=x[['candidate_id','__ts__','side_name','execution_net_ev_12h','era']].copy();z['layer']=layer;z['category']=v;o.append(z)
 return pd.concat(o,ignore_index=True)
def top(x,score):
 z=x.sort_values([score,'candidate_id'],ascending=[False,True],kind='stable').copy();z['selected_global_top10']=False;z.loc[z.index[:math.ceil(len(z)*.1)],'selected_global_top10']=True;return z
def run(output=OUT):
 output=Path(output)
 if output.exists():raise RuntimeError(output)
 jm=json.loads((J/'bridge_contract.json').read_text());am=json.loads((A/'bridge_contract.json').read_text());jp=J/'oof_predictions.parquet';ap=A/'oos_predictions.parquet'
 if jm['outputs'][jp.name]!=sha(jp) or am['outputs_sha256'][ap.name]!=sha(ap):raise RuntimeError('bridge hash')
 j=pd.read_parquet(jp,columns=['candidate_id','__ts__','side_name','execution_net_ev_12h','score_residual_expected_ev']);a=pd.read_parquet(ap,columns=['candidate_id','__ts__','side_name','execution_net_ev_12h','score_residual_expected_ev']);x=pd.concat([j,a],ignore_index=True);x['__ts__']=pd.to_datetime(x.__ts__,utc=True)
 x['era']=np.select([x.__ts__.lt(pd.Timestamp('2025-08-01',tz='UTC')),x.__ts__.lt(pd.Timestamp('2025-10-01',tz='UTC'))],['2025-07','2025-08_09'],'2025-10_11')
 cm=json.loads((S/'manifest.json').read_text());rp=S/'soft_regime_hourly.parquet';tp=S/'soft_transition_hourly.parquet'
 if cm['outputs_sha256'][rp.name]!=sha(rp) or cm['outputs_sha256'][tp.name]!=sha(tp):raise RuntimeError('context hash')
 r=pd.read_parquet(rp);t=pd.read_parquet(tp);r['source_utc']=pd.to_datetime(r.source_utc,utc=True);t['source_utc']=pd.to_datetime(t.source_utc,utc=True);ctx=r.merge(t,on='source_utc',validate='one_to_one',suffixes=('','_t')).rename(columns={'bocpd__change_probability_mean':'regime_change_probability_mean','bocpd__state_age_hours':'regime_state_age_hours','lgbm_transition_probability':'transition_lgbm_probability'})
 x=join(x,ctx);req=['regime_change_probability_mean','regime_state_age_hours','transition_lgbm_probability']
 if x[req].isna().any().any() or not (x.__ts__.astype('int64')%pd.Timedelta(hours=1).value==0).all():raise RuntimeError('context/cadence')
 x=top(x,'score_residual_expected_ev');pre=cat(x[x.selected_global_top10]);pre['day']=pre.__ts__.dt.floor('D');daily=pre.groupby(['layer','category','era','side_name','day'],as_index=False).agg(net_bps=('execution_net_ev_12h',lambda z:z.mean()*1e4),rows=('candidate_id','size'))
 cells=daily.groupby(['layer','category','era','side_name'],as_index=False).agg(days=('day','nunique'),rows=('rows','sum'),mean_bps=('net_bps','mean'),se_bps=('net_bps',lambda z:z.std(ddof=1)/math.sqrt(len(z)) if len(z)>1 else np.nan))
 loo=[]
 for (layer,categ,side,held),z in daily.groupby(['layer','category','side_name','era']):
  tr=daily[(daily.layer==layer)&(daily.category==categ)&(daily.side_name==side)&(daily.era!=held)].net_bps;te=z.net_bps;loo.append({'layer':layer,'category':categ,'side_name':side,'heldout_era':held,'train_days':len(tr),'heldout_days':len(te),'train_mean_bps':tr.mean(),'heldout_mean_bps':te.mean(),'same_sign':bool(np.sign(tr.mean())==np.sign(te.mean())) if len(tr) and len(te) else False,'positive_transfer':bool(tr.mean()>5 and te.mean()>0) if len(tr) and len(te) else False})
 loo=pd.DataFrame(loo);qual=[]
 for (layer,categ),z in loo.groupby(['layer','category']):qual.append({'layer':layer,'category':categ,'both_sides':z.side_name.nunique()==2,'independent_eras':z.heldout_era.nunique(),'all_same_sign':z.same_sign.all(),'all_positive_transfer':z.positive_transfer.all(),'stable_good':bool(z.side_name.nunique()==2 and z.heldout_era.nunique()>=3 and z.positive_transfer.all()),'promotion_eligible':False})
 # Untouched 2026 category assessment uses its frozen baseline raw score only.
 fm=json.loads((V3/'manifest.json').read_text());fp=V3/'frozen_2026_candidate_scores.parquet'
 if fm['outputs_sha256'][fp.name]!=sha(fp):raise RuntimeError('2026 hash')
 f=pd.read_parquet(fp,filters=[('arm','==','baseline')]);f['__ts__']=pd.to_datetime(f.__ts__,utc=True);f=f.drop(columns=[c for c in ctx.columns if c!='source_utc' and c in f.columns],errors='ignore');f['era']='2026_untouched';f=join(f,ctx);f=top(f,'raw_score');future=cat(f[f.selected_global_top10]);future['day']=future.__ts__.dt.floor('D');future=future.groupby(['layer','category','side_name'],as_index=False).agg(rows=('candidate_id','size'),mean_bps=('execution_net_ev_12h',lambda z:z.mean()*1e4))
 stage=Path(tempfile.mkdtemp(dir=output.parent,prefix='.'+output.name+'.'))
 try:
  x.to_parquet(stage/'pre2026_global_top10_context.parquet',index=False);daily.to_parquet(stage/'category_daily_economics.parquet',index=False);cells.to_csv(stage/'category_era_summary.csv',index=False);loo.to_csv(stage/'category_leave_era_out.csv',index=False);pd.DataFrame(qual).to_csv(stage/'category_stability_qualification.csv',index=False);future.to_csv(stage/'untouched_2026_category_assessment.csv',index=False)
  rep={'schema':'h2_common30_regime_category_performance_stability_v3','status':'SEALED_COMMON30_THREE_ERA_CATEGORY_AUDIT_NON_PROMOTION','promotion_eligible':False,'decision_cadence':'1h','exact_replay_bar_cadence':'1m_labels_only','selection':'one pooled global top10 across all pre-2026 H2 rows before category attribution; separate untouched 2026 frozen baseline global top10 assessment','eras':['2025-07','2025-08_09','2025-10_11'],'both_side_support':True,'layers':'regime and transition separate plus fixed combined interaction','no_ex_post_phase_gate':True,'no_2026_tuning':True,'findings':{'stable_good_categories':int(pd.DataFrame(qual).stable_good.sum()),'qualified_categories':len(qual),'pre2026_selected_rows':int(x.selected_global_top10.sum())},'limitation':'three common30 H2 eras are independent evaluation windows but not population-identical to wider v3; diagnostic only'};dump(stage/'report.json',rep);files=[p for p in stage.iterdir() if p.is_file()];m={**rep,'inputs':{str((J/'manifest.json').resolve()):sha(J/'manifest.json'),str((A/'manifest.json').resolve()):sha(A/'manifest.json'),str((S/'manifest.json').resolve()):sha(S/'manifest.json'),str((V3/'manifest.json').resolve()):sha(V3/'manifest.json')},'outputs_sha256':{p.name:sha(p) for p in files}};dump(stage/'manifest.json',m);(stage/'manifest.sha256').write_text(f"{sha(stage/'manifest.json')}  manifest.json\n");os.replace(stage,output);return output
 except Exception:shutil.rmtree(stage,ignore_errors=True);raise
if __name__=='__main__':print(run())
