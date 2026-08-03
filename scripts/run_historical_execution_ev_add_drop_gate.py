#!/usr/bin/env python3
"""March-development / April-outer historical execution-EV add/drop gate."""
from __future__ import annotations
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import Ridge
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
ID=["candidate_id","side_name","__symbol__","__ts__"]
def sha256_file(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for block in iter(lambda:f.read(1024*1024),b""):h.update(block)
 return h.hexdigest()
def identity_sha256(x):
 q=x[ID].copy();q["__ts__"]=pd.to_datetime(q["__ts__"],utc=True).astype(str)
 q=q.sort_values(ID,kind="stable")
 return hashlib.sha256(q.to_csv(index=False,lineterminator="\n").encode()).hexdigest()
def wj(p,x):
 t=p.with_suffix(p.suffix+".partial");t.write_text(json.dumps(x,default=str,indent=2));t.replace(p)
def validate_execution_target(x):
 for c in ("execution_gross_ev_12h","execution_cost_return","execution_net_ev_12h"):
  x[c]=pd.to_numeric(x[c],errors="coerce")
  if not np.isfinite(x[c]).all():raise ValueError(f"non-finite execution target component: {c}")
 if not np.allclose(x.execution_net_ev_12h,x.execution_gross_ev_12h-x.execution_cost_return,rtol=0.,atol=1e-10):
  raise ValueError("execution net EV is not exactly gross EV minus realized cost")
def load(a):
 r=pd.read_parquet(a.residual);d=pd.read_parquet(a.context);u=pd.read_parquet(a.aux);s=pd.concat([pd.read_parquet(a.six_root/x/"oof.parquet") for x in ("long","short")]);k=pd.concat([pd.read_parquet(a.risk_root/x/"oof.parquet") for x in ("long","short")]);pop=pd.read_parquet(a.population,columns=[*ID,"execution_gross_ev_12h","execution_cost_return"])
 for x in (r,d,u,s,k,pop):x["__ts__"]=pd.to_datetime(x["__ts__"],utc=True);x["candidate_id"]=x.candidate_id.astype(str)
 keep=["historical_base_soft_oof","base_margin_to_cutoff","base_margin_to_cutoff_z","base_score_z_within_timestamp","base_score_rank_pct_within_timestamp","candidate_group_size"]
 s=s.rename(columns={c:"sixprob_"+c[5:] for c in s if c.startswith("prob_")});k=k.rename(columns={c:"riskprob_"+c[5:] for c in k if c.startswith("prob_")})
 x=r.merge(d[ID+keep],on=ID,validate="one_to_one").merge(u[ID+["pred_peak_mfe_12h_atr__p_hit","pred_peak_mfe_12h_atr__conditional_mean"]],on=ID,validate="one_to_one").merge(s[ID+[c for c in s if c.startswith("sixprob_")]],on=ID,validate="one_to_one").merge(k[ID+[c for c in k if c.startswith("riskprob_")]],on=ID,validate="one_to_one").merge(pop,on=ID,validate="one_to_one")
 if len(x)!=140682:raise ValueError(f"strict residual identity mismatch {len(x)}")
 validate_execution_target(x)
 x["execution_label_end_utc"]=pd.to_datetime(x.execution_label_end_utc,utc=True,errors="raise")
 x["m"]=x.__ts__.dt.strftime("%Y-%m");
 if set(x.m)!={"2025-03","2025-04"}:raise ValueError("need exact March/April rows")
 six=[c for c in x if c.startswith("sixprob_")];x["six_pred_arch"]=x[six].idxmax(axis=1);g=x.groupby(["__ts__","side_name","six_pred_arch"])["historical_base_soft_oof"];x["archetype_relative_alpha_z"]=((x.historical_base_soft_oof-g.transform("mean"))/g.transform("std").replace(0,np.nan)).fillna(0.);x["rank_decile"]=(x.base_score_rank_pct_within_timestamp*10).clip(0,9.999).astype(int)
 numeric=[c for c in x.columns if c.startswith(("sixprob_","riskprob_","pred_peak_"))]+["historical_base_soft_oof","base_expected_ev","residual_expected_ev","residual_delta_ev","base_margin_to_cutoff","base_margin_to_cutoff_z","base_score_z_within_timestamp","base_score_rank_pct_within_timestamp","candidate_group_size","archetype_relative_alpha_z","rank_decile"]
 if not np.isfinite(x[numeric].to_numpy(float)).all():raise ValueError("nonfinite frozen model/context feature")
 return x
def score(q):
 v=q.execution_net_ev_12h;return {"rows":len(q),"gross_bps":float(q.execution_gross_ev_12h.mean()*1e4),"cost_bps":float(q.execution_cost_return.mean()*1e4),"net_bps":float(v.mean()*1e4),"net_median_bps":float(v.median()*1e4),"positive_rate":float(v.gt(0).mean())}
def diagnostic(allq,sel):
 n=len(sel);gross=set(allq.nlargest(n,"execution_gross_ev_12h").candidate_id);net=set(allq.nlargest(n,"execution_net_ev_12h").candidate_id);picked=set(sel.candidate_id)
 selected_by_hour={t:set(q.__symbol__.astype(str)) for t,q in sel.groupby("__ts__",sort=True)}
 hourly=[selected_by_hour.get(t,set()) for t in sorted(allq.__ts__.unique())]
 hourly_turn=[1-len(a&b)/max(1,len(a|b)) for a,b in zip(hourly,hourly[1:]) if a or b]
 selected_daily=sel.assign(day=sel.__ts__.dt.floor("D")).groupby("day",sort=True).__symbol__.agg(lambda s:set(s.astype(str)))
 daily=list(selected_daily);daily_turn=[1-len(a&b)/max(1,len(a|b)) for a,b in zip(daily,daily[1:]) if a or b]
 return {"selected":score(sel),"all_candidates":score(allq),"gross_exceeds_cost_rate":float(sel.execution_gross_ev_12h.gt(sel.execution_cost_return).mean()),"positive_net_precision":float(sel.execution_net_ev_12h.gt(0).mean()),"gross_oracle_top_decile_recall":float(len(picked&gross)/max(1,len(gross))),"net_oracle_top_decile_recall":float(len(picked&net)/max(1,len(net))),"adjacent_hour_selected_asset_turnover":float(np.mean(hourly_turn)) if hourly_turn else 0.,"adjacent_day_selected_asset_turnover":float(np.mean(daily_turn)) if daily_turn else 0.,"selected_coverage":[{"side":str(side),"month":str(month),"rows":int(rows)} for (side,month),rows in sel.groupby(["side_name","m"]).size().items()]}
def prep(tr,te,fs):
 med=tr[fs].apply(pd.to_numeric,errors="coerce").median().fillna(0);a=tr[fs].apply(pd.to_numeric,errors="coerce").fillna(med);b=te[fs].apply(pd.to_numeric,errors="coerce").fillna(med);scale=a.std().replace(0,1).fillna(1);return (a-a.mean())/scale,(b-a.mean())/scale
def fit_arm(train,test,features,side):
 cut=train.__ts__.quantile(.75);a=train.loc[(train.__ts__<cut)&(pd.to_datetime(train.execution_label_end_utc,utc=True)<cut)];b=train.loc[train.__ts__>=cut];rank=a[features].corrwith(a.execution_net_ev_12h,method="spearman").abs().fillna(0).sort_values(ascending=False);fs=rank.head(max(1,min(len(features),max(3,int(len(features)*.8))))).index.tolist();best=None
 for alpha in (.1,1.,10.,100.):
  xa,xb=prep(a,b,fs);m=Ridge(alpha=alpha).fit(xa,a.execution_net_ev_12h);loss=float(np.mean((m.predict(xb)-b.execution_net_ev_12h)**2));best=min((loss,alpha),best) if best else (loss,alpha)
 xt,xv=prep(train,test,fs);m=Ridge(alpha=best[1]).fit(xt,train.execution_net_ev_12h);return m.predict(xv),fs,best
def inner_scores(train,features,alpha,fs):
 out=[]
 for c,e in ((pd.Timestamp("2025-03-11",tz="UTC"),pd.Timestamp("2025-03-21",tz="UTC")),(pd.Timestamp("2025-03-21",tz="UTC"),pd.Timestamp("2025-04-01",tz="UTC"))):
  a=train.loc[(train.__ts__<c)&(train.execution_label_end_utc<c)];b=train.loc[(train.__ts__>=c)&(train.__ts__<e)];xa,xb=prep(a,b,fs);m=Ridge(alpha=alpha).fit(xa,a.execution_net_ev_12h);q=b[ID+["execution_label_end_utc","execution_gross_ev_12h","execution_cost_return","execution_net_ev_12h"]].copy();q["score"]=m.predict(xb);out.append(q)
 return pd.concat(out)
def mapped(inner,test,raw):
 out=np.empty(len(test));day=test.__ts__.dt.floor("D")
 for d in day.unique():
  mask=day.eq(d);hist=inner.loc[(inner.__ts__<d)&(inner.execution_label_end_utc<d)&(inner.__ts__>=d-pd.Timedelta(days=21))]
  if len(hist)<300:out[mask]=raw[mask]
  else:out[mask]=IsotonicRegression(out_of_bounds="clip").fit(hist.score,hist.execution_net_ev_12h).predict(raw[mask])
 return out
def main():
 p=argparse.ArgumentParser();p.add_argument("--residual",type=Path,required=True);p.add_argument("--context",type=Path,required=True);p.add_argument("--aux",type=Path,required=True);p.add_argument("--six-root",type=Path,required=True);p.add_argument("--risk-root",type=Path,required=True);p.add_argument("--population",type=Path,required=True);p.add_argument("--output",type=Path,required=True);a=p.parse_args()
 if a.output.exists():raise FileExistsError(a.output)
 partial=a.output.with_name(a.output.name+".partial")
 if partial.exists():raise FileExistsError(partial)
 partial.mkdir(parents=True)
 x=load(a);six=[c for c in x if c.startswith("sixprob_")];risk=[c for c in x if c.startswith("riskprob_")];base=["historical_base_soft_oof"];res=["base_expected_ev","residual_expected_ev","residual_delta_ev"];peak=["pred_peak_mfe_12h_atr__p_hit","pred_peak_mfe_12h_atr__conditional_mean"];ctx=["historical_base_soft_oof","base_margin_to_cutoff","base_margin_to_cutoff_z","base_score_z_within_timestamp","base_score_rank_pct_within_timestamp","archetype_relative_alpha_z","rank_decile","candidate_group_size"]
 arms={"base_only":base,"residual_only":res,"base_residual":base+res,"plus_six":base+res+six,"plus_risk":base+res+risk,"plus_peak":base+res+peak,"all_non_timing":list(dict.fromkeys(base+res+six+risk+peak+ctx))}
 for n,fam in (("loo_six",six),("loo_risk",risk),("loo_peak",peak),("loo_context",ctx)):arms[n]=[f for f in arms["all_non_timing"] if f not in fam]
 for c in ctx:arms["base_residual_plus_context_"+c]=list(dict.fromkeys(base+res+[c]))
 for c in ctx:arms["all_minus_context_"+c]=[f for f in arms["all_non_timing"] if f!=c]
 rows=[];output_index={}
 for arm,features in arms.items():
  pred=[];inner_pred=[];contract={}
  for side in ("long","short"):
   tr=x.loc[(x.m.eq("2025-03"))&(x.side_name.eq(side))&(x.execution_label_end_utc<pd.Timestamp("2025-04-01",tz="UTC"))].copy();te=x.loc[(x.m.eq("2025-04"))&(x.side_name.eq(side))].copy();raw,fs,best=fit_arm(tr,te,features,side);inn=inner_scores(tr,features,best[1],fs);te["raw_score"]=raw;te["mapped_score"]=mapped(inn,te,raw);pred.append(te);inner_pred.append(inn);contract[side]={"selected_features":fs,"hpo_alpha":best[1],"inner_validation_mse":best[0],"inner_oof_rows":len(inn),"label_purge":"feature selection/HPO and every inner OOF train block require execution_label_end_utc before validation start"}
  q=pd.concat(pred);rawsel=q.nlargest(int(np.ceil(len(q)*.1)),"raw_score");sel=q.nlargest(int(np.ceil(len(q)*.1)),"mapped_score");r={"arm":arm,"features_requested":features,"contract":contract,"raw_score_global_top10_diagnostic":diagnostic(q.assign(mapped_score=q.raw_score),rawsel),"causal_mapped_global_top10_diagnostic":diagnostic(q,sel),"april_long_selected":score(sel.loc[sel.side_name.eq("long")]),"april_short_selected":score(sel.loc[sel.side_name.eq("short")]),"selection":"one pooled global top10 across both sides after causal rolling 21d mapping; never per timestamp","research_status":"diagnostic_march_dev_non_nested_fs_hpo_mapping","outer_status":"diagnostic_non_promotion_march_reused_for_fs_hpo_mapping"};rows.append(r)
  arm_dir=partial/arm;arm_dir.mkdir()
  inner_path=arm_dir/"march_inner_oof_scores.parquet";outer_path=arm_dir/"april_outer_predictions.parquet"
  inner_frame=pd.concat(inner_pred,ignore_index=True)
  inner_frame.to_parquet(inner_path,index=False,compression="zstd")
  q[ID+["execution_label_end_utc","execution_gross_ev_12h","execution_cost_return","execution_net_ev_12h","raw_score","mapped_score"]].to_parquet(outer_path,index=False,compression="zstd")
  output_index[arm]={"march_inner_oof_scores":{"path":str(inner_path.relative_to(partial)),"sha256":sha256_file(inner_path),"rows":int(len(inner_frame))},"april_outer_predictions":{"path":str(outer_path.relative_to(partial)),"sha256":sha256_file(outer_path),"rows":int(len(q))}}
 wj(partial/"report.json",rows)
 sources={"residual":a.residual,"context":a.context,"aux":a.aux,"population":a.population}
 for side in ("long","short"):
  sources[f"six_{side}"]=a.six_root/side/"oof.parquet"
  sources[f"risk_{side}"]=a.risk_root/side/"oof.parquet"
 manifest={"schema":"historical_execution_ev_add_drop_gate_v6","status":"research_only_diagnostic","rows":int(len(x)),"april_outer_rows":int(x.m.eq("2025-04").sum()),"strict_identity_sha256":identity_sha256(x),"sources":{k:{"path":str(v),"sha256":sha256_file(v)} for k,v in sources.items()},"contracts":{"development":"March only; every fit block purges labels resolving at or after its validation start","outer":"April predictions are untouched by fitting, selection, HPO and mapping fit","selection":"one pooled global top 10%; never per timestamp","mapping":"causal rolling 21-day mapping from resolved inner-OOF rows only","durable_score_ledgers":"per-arm purged March inner-OOF calibration scores and untouched April predictions","turnover":"Jaccard turnover of selected asset sets across adjacent candidate hours/days; never candidate-ID turnover","side_local_ridge_hpo_alpha_grid":[0.1,1.0,10.0,100.0],"timing_mae_wait_excluded":True,"execution_target_identity":"execution_net_ev_12h == execution_gross_ev_12h - execution_cost_return, atol=1e-10"},"arms":len(rows),"outputs":output_index}
 wj(partial/"manifest.json",manifest)
 partial.replace(a.output)
 print(json.dumps({"manifest":manifest,"arms":rows},default=str))
if __name__=="__main__":main()
