#!/usr/bin/env python3
"""Tie-aware April confirmation gates for the raw-March-selected support arm."""
from __future__ import annotations
import argparse,hashlib,json,math,os,tempfile
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];SRC=ROOT/'data_perp/artifacts/bounded_robust_auxiliary_contribution_ablation_20260730_v2';SIDE=ROOT/'data_perp/artifacts/bounded_robust_auxiliary_contribution_ablation_20260730_v2_provenance_20260730_v1';FRACS=(.01,.05,.1,.2);ID=['candidate_id','side_name','__symbol__','__ts__'];Y='execution_net_ev_12h'
def h(p):
 d=hashlib.sha256()
 with Path(p).open('rb') as x:
  for b in iter(lambda:x.read(1<<20),b''):d.update(b)
 return d.hexdigest()
def w(p,x):
 t=p.with_name('.'+p.name+'.tmp');t.write_text(json.dumps(x,indent=2,sort_keys=True,default=str)+'\n');os.replace(t,p)
def select(x,col,f):
 n=max(1,math.ceil(len(x)*f));return x.sort_values([col,'candidate_id','__ts__','__symbol__','side_name'],ascending=[False,True,True,True,True],kind='mergesort').iloc[:n]
def bounds(x,col,f):
 q=select(x,col,f);n=len(q);cut=float(q[col].iloc[-1]);above=x[x[col]>cut];ties=x[np.isclose(x[col].to_numpy(float),cut,rtol=0,atol=1e-14)];need=n-len(above);tv=ties[Y].to_numpy(float);av=above[Y].to_numpy(float)
 def stat(v):return float(np.mean(v)*1e4),float(np.mean(v>0))
 expected=np.r_[av,np.repeat(tv.mean(),need)]
 return {'top_fraction':f,'rows':n,'cutoff':cut,'above_cutoff_rows':len(above),'cutoff_tie_rows':len(ties),'cutoff_tie_fraction_of_book':len(ties)/n,'required_from_tie_rows':need,'deterministic_net_bps':stat(q[Y].to_numpy(float))[0],'deterministic_precision':stat(q[Y].to_numpy(float))[1],'random_tie_expected_net_bps':stat(expected)[0],'random_tie_expected_precision':stat(expected)[1],'best_tie_net_bps':stat(np.r_[av,np.sort(tv)[-need:]])[0],'best_tie_precision':stat(np.r_[av,np.sort(tv)[-need:]])[1],'worst_tie_net_bps':stat(np.r_[av,np.sort(tv)[:need]])[0],'worst_tie_precision':stat(np.r_[av,np.sort(tv)[:need]])[1]}
def cal(q,col):
 p=q[col].to_numpy(float);y=q[Y].to_numpy(float);edges=np.unique(np.quantile(p,np.linspace(0,1,11)));ece=0.
 for lo,hi in zip(edges[:-1],edges[1:]):
  m=(p>=lo)&((p<hi)|(hi==edges[-1]));ece+=m.mean()*abs(p[m].mean()-y[m].mean()) if m.any() else 0.
 return {'bias_bps':float((p-y).mean()*1e4),'mae_bps':float(np.abs(p-y).mean()*1e4),'ece_bps':float(ece*1e4)}
def run(a):
 if a.output_dir.exists():raise FileExistsError(a.output_dir)
 man=json.loads((a.source/'manifest.json').read_text());win=man['frozen_april_winner_from_raw_march_oof'];assert win['arm']=='future_slope' and float(win['weight'])==.25
 all_pred=pd.read_parquet(a.source/'april_confirmation_predictions.parquet');all_pred['__ts__']=pd.to_datetime(all_pred['__ts__'],utc=True)
 # Per-arm March winners are diagnostic context only.  The global March winner
 # below remains the sole confirmation-authoritative arm and the only one that
 # can enter the promotion gates.
 arm_choices=pd.read_csv(a.source/'march_oof_raw_weight_selection.csv').sort_values(['arm','march_oof_raw_top10_net_bps','weight'],ascending=[True,False,True],kind='mergesort').groupby('arm',as_index=False).head(1)
 x=all_pred[(all_pred.arm==win['arm'])&(all_pred.weight==win['weight'])].copy();assert len(x)==69258
 ties=[];weeks=[];gates=[]
 for _,choice in arm_choices.iterrows():
  arm_x=all_pred[(all_pred.arm.eq(choice.arm))&(all_pred.weight.eq(choice.weight))]
  for kind,col in [('raw','raw_score'),('mapped','mapped_score')]:
   for f in FRACS:
    week=arm_x['__ts__'].max().floor('D')-pd.Timedelta(days=6);last=arm_x[arm_x.__ts__>=week];wq=select(last,col,f)
    weeks.append({'arm':choice.arm,'weight':choice.weight,'confirmation_authoritative':bool(choice.arm==win['arm'] and float(choice.weight)==float(win['weight'])),'score_kind':kind,'top_fraction':f,'latest_week_start_utc':week,'rows':len(wq),'net_bps':float(wq[Y].mean()*1e4),'positive_rate':float(wq[Y].gt(0).mean())})
 for kind,col in [('raw','raw_score'),('mapped','mapped_score')]:
  for f in FRACS:
   b=bounds(x,col,f);b.update({'score_kind':kind});ties.append(b)
   q=select(x,col,f);week=x['__ts__'].max().floor('D')-pd.Timedelta(days=6);last=x[x.__ts__>=week];wq=select(last,col,f)
   if f==.1:
    c=cal(q,col);side=q.groupby('side_name').agg(rows=(Y,'size'),net=(Y,'mean'));asset=q.groupby('__symbol__').size();tie=b
    gates += [
     {'gate':'April confirmation coverage','pass':len(x)==69258,'detail':len(x)},
     {'gate':f'{kind} top10 expected economics','pass':tie['random_tie_expected_net_bps']>0,'detail':tie['random_tie_expected_net_bps']},
     {'gate':f'{kind} latest-week top10 economics','pass':float(wq[Y].mean()*1e4)>0,'detail':float(wq[Y].mean()*1e4)},
     {'gate':f'{kind} cutoff ties <= 5% of book','pass':tie['cutoff_tie_fraction_of_book']<=.05,'detail':tie['cutoff_tie_fraction_of_book']},
     {'gate':f'{kind} side allocation <= 75%','pass':float(side.rows.max()/len(q))<=.75,'detail':float(side.rows.max()/len(q))},
     {'gate':f'{kind} each materially selected side positive','pass':bool((side.loc[side.rows>=.1*len(q),'net']>0).all()),'detail':{k:float(v*1e4) for k,v in side.net.items()}},
     {'gate':f'{kind} top asset <= 10%','pass':float(asset.max()/len(q))<=.10,'detail':float(asset.max()/len(q))},
     {'gate':f'{kind} top10 |bias| <= 25 bps and ECE <= 25 bps','pass':abs(c['bias_bps'])<=25 and c['ece_bps']<=25,'detail':c},]
 st=Path(tempfile.mkdtemp(prefix='.'+a.output_dir.name+'.',dir=a.output_dir.parent));pd.DataFrame(ties).to_csv(st/'tie_bounds.csv',index=False);pd.DataFrame(weeks).to_csv(st/'latest_week_metrics.csv',index=False);pd.DataFrame(gates).to_csv(st/'promotion_gates.csv',index=False);out={p.name:h(p) for p in st.iterdir() if p.is_file()};m={'schema':'bounded_robust_auxiliary_contribution_gates_v1','status':'CONFIRMATION_GATES_FAIL_NO_REPLAY','frozen_arm':win,'scope':'only raw-March-selected future_slope weight .25 is confirmation-authoritative','tie_contract':'deterministic ordering is diagnostic only; expected/best/worst tie allocation bounds determine economic gate','gates':'all must pass for replay','sources':{'v2_manifest':h(a.source/'manifest.json'),'v2_predictions':h(a.source/'april_confirmation_predictions.parquet'),'slope_seal':h(a.side/'slope_detached_seal.json'),'v1_invalidation':h(a.side/'v1_invalidation.json')},'outputs_sha256':out};w(st/'manifest.json',m);(st/'manifest.sha256').write_text(h(st/'manifest.json')+'  manifest.json\n');os.replace(st,a.output_dir);return m
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--source',type=Path,default=SRC);p.add_argument('--side',type=Path,default=SIDE);p.add_argument('--output-dir',type=Path,required=True);print(json.dumps(run(p.parse_args()),indent=2,default=str))
