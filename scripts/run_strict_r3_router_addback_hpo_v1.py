#!/usr/bin/env python3
"""Short pruned strict-OOF HPO for Router incremental-addback finalists."""
from __future__ import annotations
import argparse, gc, hashlib, json, os, sys
from pathlib import Path
import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMRanker, early_stopping

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT/"scripts") not in sys.path: sys.path.insert(0,str(ROOT/"scripts"))
import run_strict_r3_economic_recall_router as router
import run_strict_r3_router_full_universe_stability_v1 as stability
import run_strict_r3_router_subset_ladder_v1 as ladder

SCHEMA="strict_r3_router_addback_hpo_v1"; SEED=1729
GAINS={"moderate":[0,1,2,4,7,11],"tail":[0,1,3,6,11,18],"clipped":[0,.5,2,3,6,8]}

def _once(path:Path,payload:object)->None:
 path.parent.mkdir(parents=True,exist_ok=True); fd=os.open(path,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o644)
 with os.fdopen(fd,"w",encoding="utf-8") as f: json.dump(payload,f,indent=2,sort_keys=True,default=str)
def _hash(v:tuple[str,...])->str:return hashlib.sha256("\n".join(v).encode()).hexdigest()

def _fit_eval(train,held,xtrain,xheld,fields,policy,primary,scheme,params,seed,n_jobs,score_path):
 target=router._primary_target(train,primary).astype(np.int32); work=train.copy(); work["__target__"]=target
 utility=router._primary_weight_utility(work,primary,target)
 order,groups,weights,weight_summary=router._query_weights(work,scheme=scheme,primary_utility=utility)
 rows=order["__row__"].to_numpy(np.int64); y=work.iloc[rows]["__target__"].to_numpy(np.int32)
 depth=params["depth"]; leaves=min(params["leaves"],2**depth-1)
 spec=dict(objective=params["objective"],metric="ndcg",label_gain=GAINS[params["gains"]],
  n_estimators=2000,learning_rate=params["lr"],max_depth=depth,num_leaves=leaves,
  min_child_samples=max(params["floor"],int(params["frac"]*len(rows))),min_split_gain=params["gain"],
  subsample=params["subsample"],subsample_freq=1,colsample_bytree=params["feature_fraction"],
  reg_alpha=params["l1"],reg_lambda=params["l2"],max_bin=params["max_bin"],
  lambdarank_truncation_level=params["truncation"],random_state=seed,n_jobs=n_jobs,deterministic=True,force_col_wise=True,verbosity=-1)
 cut=min(max(int(len(groups)*.8),400),len(groups)-100); split=int(groups[:cut].sum())
 # Groups are chronologically contiguous after _query_weights.  Fit the
 # early-stopping probe on the same prefix represented by groups[:cut], then
 # evaluate strictly on the later query groups.  Do not hand LightGBM a full
 # matrix with prefix-only weights.
 probe=LGBMRanker(**spec).fit(xtrain[rows][:split],y[:split],group=groups[:cut],sample_weight=weights[:split],eval_set=[(xtrain[rows][split:],y[split:])],eval_group=[groups[cut:]],callbacks=[early_stopping(30,verbose=False)])
 spec["n_estimators"]=int(probe.best_iteration_ or 2000)
 model=LGBMRanker(**spec).fit(xtrain[rows],y,group=groups,sample_weight=weights)
 train_raw=model.predict(xtrain[rows]).astype(np.float32); held_raw=model.predict(xheld).astype(np.float32)
 rank=__import__('screen_strict_r3_router_full_universe_v1')._rank_reference(train_raw,held_raw)
 # The scorer is deliberately materialised before it is allowed to see an
 # outcome column.  This receipt makes the strict target-free handoff
 # independently auditable for every HPO fold/trial.
 target_free=held.loc[:,list(stability.IDENTITY)].copy()
 target_free["router_primary_rank"]=rank
 target_free.to_parquet(score_path,index=False,compression="zstd")
 metric=stability._metric(held,rank,policy)
 return metric, int(spec["n_estimators"]), weight_summary

def _params(trial):
 d=trial.suggest_int("depth",3,5); requested=trial.suggest_categorical("leaves",[7,15,31]); leaves=max(v for v in (7,15,31) if v<=min(requested,2**d-1))
 return {"objective":trial.suggest_categorical("objective",["rank_xendcg","lambdarank"]),"gains":trial.suggest_categorical("gains",list(GAINS)),"depth":d,"leaves":leaves,"lr":trial.suggest_float("lr",.02,.08,log=True),"frac":trial.suggest_float("frac",.004,.020,log=True),"floor":trial.suggest_int("floor",200,1000,step=100),"gain":trial.suggest_float("gain",1e-4,.02,log=True),"feature_fraction":trial.suggest_float("feature_fraction",.70,.92),"subsample":trial.suggest_float("subsample",.70,.92),"l1":trial.suggest_float("l1",1e-4,5,log=True),"l2":trial.suggest_float("l2",.1,30,log=True),"max_bin":trial.suggest_categorical("max_bin",[63,127]),"truncation":trial.suggest_categorical("truncation",[10,12,16,20,24])}

def run_candidate(args, candidate, fields, roots, policy, months, root):
 cache=[]
 for month in months:
  cache.append(stability._prepare_fold(roots=roots,fields=fields,policy=policy,held_month=month,train_months=args.train_months,reserve_days=args.reserve_days,train_cap=args.train_cap,held_cap=args.held_cap))
 trials=[]
 def objective(trial):
  p=_params(trial); scores=[]; fold_rows=[]
  for i,(month,data) in enumerate(zip(months,cache,strict=True)):
   train,held,xt,xh,_=data
   score_path=(root/"target_free_trial_scores"/candidate/f"trial={trial.number:02d}"/f"fold={month:%Y-%m}.parquet")
   score_path.parent.mkdir(parents=True,exist_ok=True)
   metric,trees,weight=_fit_eval(train,held,xt,xh,fields,policy,args.primary_target,args.row_weight_scheme,p,SEED+trial.number*100+i,args.n_jobs,score_path)
   scores.append(metric["s_router"]); fold_rows.append({"candidate":candidate,"trial":trial.number,"held_month":f"{month:%Y-%m}","trees":trees,**metric,**weight})
   current=np.asarray(scores,float); value=float(.65*current.mean()+.25*np.quantile(current,.25)+.10*current.min()); trial.report(value,i)
   if i>=1 and trial.should_prune(): trial.set_user_attr("fold_rows",fold_rows); raise optuna.TrialPruned()
  trial.set_user_attr("fold_rows",fold_rows); return float(.65*np.mean(scores)+.25*np.quantile(scores,.25)+.10*np.min(scores))
 sampler=optuna.samplers.TPESampler(seed=SEED+len(fields),multivariate=True); pruner=optuna.pruners.MedianPruner(n_startup_trials=3,n_warmup_steps=1)
 study=optuna.create_study(direction="maximize",sampler=sampler,pruner=pruner); study.optimize(objective,n_trials=args.trials,n_jobs=1,gc_after_trial=True)
 for t in study.trials:
  for row in t.user_attrs.get("fold_rows",[]): trials.append({**row,"state":t.state.name,"value":t.value,"params_json":json.dumps(t.params,sort_keys=True)})
 return study, pd.DataFrame(trials)

def main():
 p=argparse.ArgumentParser(); p.add_argument("--feature-roots",required=True);p.add_argument("--finalists",type=Path,required=True);p.add_argument("--policy",type=Path,required=True);p.add_argument("--out",type=Path,required=True);p.add_argument("--held-months",default="2025-10,2026-02,2026-06");p.add_argument("--primary-target",default="U50_p050_c300",choices=router.ALL_PRIMARY_TARGETS);p.add_argument("--row-weight-scheme",default="positive_125",choices=router._ROW_WEIGHT_SCHEMES);p.add_argument("--train-months",type=int,default=3);p.add_argument("--reserve-days",type=int,default=28);p.add_argument("--train-cap",type=int,default=120000);p.add_argument("--held-cap",type=int,default=12000);p.add_argument("--trials",type=int,default=8);p.add_argument("--n-jobs",type=int,default=4);p.add_argument("--control-only",action="store_true",help="run the frozen 30-field contract through the identical HPO protocol");a=p.parse_args()
 if a.out.exists() or a.trials<6: raise ValueError("immutable HPO output exists or insufficient short-HPO trials")
 if a.train_cap < 90000: raise ValueError("HPO train cap must retain at least 500 hourly timestamp queries")
 roots=ladder._roots(a.feature_roots); months=ladder._parse_months(a.held_months); payload=json.loads(a.finalists.read_text()); finalists=payload["finalists"]
 if a.control_only:
  finalists=[{"candidate":"frozen30_control", "feature_contract":payload["control"]["feature_contract"], "feature_contract_sha256":payload["control"]["feature_contract_sha256"]}]
 if not finalists: raise AssertionError("no add-back finalist passed advance guard")
 policy=router._policy_window(a.policy.resolve(),months[0]-pd.DateOffset(months=a.train_months+2),months[-1]+pd.offsets.MonthBegin(1)).loc[:,["candidate_id","policy_path_valid","policy_net_bps","policy_gross_bps","policy_label_available_ts"]]
 a.out.mkdir(parents=True);_once(a.out/"run_contract.json",{"schema":SCHEMA,"scope":"offline strict-OOF Router HPO; no live/exchange mutation","finalists":a.finalists.resolve().as_posix(),"control_only":a.control_only,"folds":[f"{x:%Y-%m}" for x in months],"trials":a.trials,"early_stopping":30,"objective_candidates":["rank_xendcg","lambdarank"],"target_free_held_scores_before_metric_join":True})
 summaries=[];allrows=[]
 for item in finalists:
  name=item["candidate"]; fields=tuple(item["feature_contract"]); study,rows=run_candidate(a,name,fields,roots,policy,months,a.out);allrows.append(rows);best=study.best_trial
  summaries.append({"candidate":name,"features":len(fields),"feature_hash":_hash(fields),"best_trial":best.number,"s_stable":best.value,"params_json":json.dumps(best.params,sort_keys=True),"fold_scores_json":json.dumps([r["s_router"] for r in best.user_attrs["fold_rows"]])})
 frame=pd.concat(allrows,ignore_index=True);summary=pd.DataFrame(summaries).sort_values("s_stable",ascending=False,kind="stable");frame.to_parquet(a.out/"trial_fold_metrics.parquet",index=False,compression="zstd");summary.to_parquet(a.out/"candidate_summary.parquet",index=False,compression="zstd");_once(a.out/"hpo_winner.json",{"schema":SCHEMA,"winner":summary.iloc[0].to_dict(),"scope":"research-only HPO winner; requires fresh frozen-forward scoring and downstream replay"});_once(a.out/"run_manifest.json",{"schema":SCHEMA,"status":"complete","candidates":len(finalists),"trials":a.trials,"scope":"offline Router HPO complete; no live/exchange mutation"});print(json.dumps({"event":"complete","winner":summary.iloc[0].candidate,"s_stable":float(summary.iloc[0].s_stable)}))
if __name__=="__main__":main()
