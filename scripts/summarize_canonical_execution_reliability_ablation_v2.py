#!/usr/bin/env python3
"""Seal the decision summary for the canonical reliability ablation v2."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/"data_perp/artifacts/canonical_execution_reliability_ablation_20260730_v2"
CONFIG=ROOT/"configs/canonical_execution_reliability_workstream_20260730_v2.json"
OUTPUT=ROOT/"data_perp/artifacts/canonical_execution_reliability_ablation_summary_20260730_v1"
NET="execution_net_ev_12h"

class SummaryError(RuntimeError): pass

def sha(path:Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1<<20),b""): h.update(block)
    return h.hexdigest()

def safe(value:Any)->Any:
    if isinstance(value,Mapping): return {str(k):safe(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [safe(v) for v in value]
    if isinstance(value,(Path,pd.Timestamp)): return str(value)
    if isinstance(value,np.generic): return value.item()
    if isinstance(value,float) and not np.isfinite(value): return None
    return value

def write_json(path:Path,value:Mapping[str,Any])->None:
    tmp=path.with_name("."+path.name+".tmp")
    tmp.write_text(json.dumps(safe(value),indent=2,sort_keys=True)+"\n")
    os.replace(tmp,path)

def verify_source(root:Path)->dict[str,Any]:
    manifest=root/"manifest.json"; seal=root/"manifest.sha256"
    if not manifest.is_file() or not seal.is_file(): raise FileNotFoundError(root)
    if sha(manifest)!=seal.read_text().split()[0]: raise SummaryError("source manifest seal mismatch")
    payload=json.loads(manifest.read_text())
    if payload.get("schema")!="canonical_execution_reliability_ablation_v2": raise SummaryError("wrong source schema")
    for name,expected in payload.get("outputs_sha256",{}).items():
        path=root/name
        if not path.is_file() or sha(path)!=expected: raise SummaryError(f"source output hash mismatch: {name}")
    return payload

def random_tie(frame:pd.DataFrame,score:str,fraction:float)->dict[str,float]:
    n=max(1,int(math.ceil(len(frame)*fraction)))
    ordered=frame.sort_values([score,"candidate_id"],ascending=[False,True],kind="mergesort")
    cutoff=float(ordered[score].iloc[n-1])
    above=frame.loc[frame[score].gt(cutoff)]
    tie=frame.loc[frame[score].eq(cutoff)]
    need=n-len(above)
    net=float((above[NET].sum()+need*tie[NET].mean())/n*1e4)
    return {"rows":len(frame),"selected_rows":n,"net_bps":net,"cutoff":cutoff,"tie_selected_share":float(need/n)}

def selection_objective(values:Sequence[float])->dict[str,float]:
    x=np.asarray(values,float)
    return {"selection_mean_bps":float(x.mean()),"selection_std_bps":float(x.std()),"selection_worst_bps":float(x.min()),"selection_latest_bps":float(x[-1]),"selection_objective_bps":float(x.mean()-.5*x.std()+.25*x.min())}

def summarize(source:Path,config_path:Path)->tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame,dict[str,Any]]:
    config=json.loads(config_path.read_text())
    selection=pd.read_csv(source/"selection.csv")
    economics=pd.read_csv(source/"economics.csv")
    attribution=pd.read_csv(source/"global_book_attribution.csv")
    scores=pd.read_parquet(source/"scores.parquet")
    fold_names=[f["name"] for f in config["outer_folds"] if str(f["role"]).startswith("architecture")]
    evaluated=sorted(economics["config"].unique())
    arms=[]
    for name in evaluated:
        row={"config":name}
        selected=selection.loc[selection.config.eq(name)]
        if len(selected):
            row["architecture"]=selected.iloc[0].get("architecture")
            row["variant"]=selected.iloc[0].get("variant")
        fold=economics.loc[economics.config.eq(name)&economics.stage.eq("march_oof")&economics.score_kind.eq("mapped")&economics.top_fraction.eq(.1)&economics.window.isin(fold_names)].set_index("window")
        if set(fold_names)-set(fold.index): raise SummaryError(f"incomplete selection folds: {name}")
        row.update(selection_objective([float(fold.loc[f,"random_tie_expected_net_bps"]) for f in fold_names]))
        for stage,prefix in (("march_oof","march"),("april_frozen_diagnostic","april")):
            for kind in ("raw","mapped"):
                q=economics.loc[economics.config.eq(name)&economics.stage.eq(stage)&economics.window.eq("aggregate")&economics.score_kind.eq(kind)&economics.top_fraction.eq(.1)]
                if len(q)!=1: raise SummaryError(f"missing aggregate economics: {name}/{stage}/{kind}")
                q=q.iloc[0]
                row[f"{prefix}_{kind}_top10_gross_bps"]=q["random_tie_expected_gross_bps"]
                row[f"{prefix}_{kind}_top10_cost_bps"]=q["random_tie_expected_cost_bps"]
                row[f"{prefix}_{kind}_top10_net_bps"]=q["random_tie_expected_net_bps"]
                row[f"{prefix}_{kind}_top10_tie_selected_share"]=q["cutoff_tie_selected_share"]
        q=scores.loc[scores.config.eq(name)&scores.mapped_eligible&scores.execution_decision_utc.dt.month.eq(4)].copy()
        end=q.execution_decision_utc.max().floor("D")+pd.Timedelta(days=1)
        q=q.loc[q.execution_decision_utc.ge(end-pd.Timedelta(days=7))&q.execution_decision_utc.lt(end)]
        row["april_latest_7d_start_utc"]=end-pd.Timedelta(days=7)
        row["april_latest_7d_end_utc"]=end
        row["april_latest_7d_mapped_top10_net_bps"]=random_tie(q,"mapped_score",.1)["net_bps"]
        fold_ties=fold["cutoff_tie_selected_share"].astype(float)
        row["march_selection_max_tie_selected_share"]=float(fold_ties.max())
        side=attribution.loc[attribution.config.eq(name)&attribution.stage.eq("march_oof")&attribution.window.eq("aggregate")&attribution.score_kind.eq("mapped")&attribution.top_fraction.eq(.1)&attribution.dimension.eq("side")].set_index("value")
        row["march_long_net_contribution_bps"]=float(side.loc["long","net_bps_contribution"])
        row["march_short_net_contribution_bps"]=float(side.loc["short","net_bps_contribution"])
        arms.append(row)
    arms=pd.DataFrame(arms).sort_values("selection_objective_bps",ascending=False).reset_index(drop=True)
    support=selection.loc[selection.architecture.eq("A1")&selection.objective.notna()].sort_values("objective",ascending=False).iloc[0]["config"]
    context=selection.loc[selection.architecture.eq("A1_context")&selection.objective.notna()].sort_values("objective",ascending=False).iloc[0]["config"]
    target=selection.loc[selection.architecture.isin(["A2","A3","A4"])&selection.objective.notna()].sort_values("objective",ascending=False).iloc[0]["config"]
    final=selection.loc[selection.architecture.eq("A5")&selection.objective.notna()].sort_values("objective",ascending=False).iloc[0]["config"]
    control="A0__score_residual_expected_ev"
    f=arms.set_index("config").loc[final]; c=arms.set_index("config").loc[control]
    gates=[
        ("march_aggregate_positive",f.march_mapped_top10_net_bps>0,f.march_mapped_top10_net_bps,"> 0 bps"),
        ("march_latest_fold_positive",f.selection_latest_bps>0,f.selection_latest_bps,"> 0 bps"),
        ("march_worst_fold_positive",f.selection_worst_bps>0,f.selection_worst_bps,"> 0 bps"),
        ("april_reused_diagnostic_positive",f.april_mapped_top10_net_bps>0,f.april_mapped_top10_net_bps,"> 0 bps"),
        ("april_latest_7d_positive",f.april_latest_7d_mapped_top10_net_bps>0,f.april_latest_7d_mapped_top10_net_bps,"> 0 bps"),
        ("both_sides_positive",f.march_long_net_contribution_bps>0 and f.march_short_net_contribution_bps>0,min(f.march_long_net_contribution_bps,f.march_short_net_contribution_bps),"long and short > 0 bps"),
        ("aggregate_tie_safe",f.march_mapped_top10_tie_selected_share<=.05 and f.april_mapped_top10_tie_selected_share<=.05,max(f.march_mapped_top10_tie_selected_share,f.april_mapped_top10_tie_selected_share),"<= 5%"),
        ("all_selection_folds_tie_safe",f.march_selection_max_tie_selected_share<=.05,f.march_selection_max_tie_selected_share,"<= 5%"),
        ("beats_residual_control_march",f.march_mapped_top10_net_bps>c.march_mapped_top10_net_bps,f.march_mapped_top10_net_bps-c.march_mapped_top10_net_bps,"> 0 bps delta"),
        ("beats_residual_control_objective",f.selection_objective_bps>c.selection_objective_bps,f.selection_objective_bps-c.selection_objective_bps,"> 0 bps delta"),
        ("untouched_forward_evidence",False,np.nan,"required")
    ]
    gates=pd.DataFrame([{"gate":g,"pass":bool(ok),"observed":value,"requirement":req} for g,ok,value,req in gates])
    heads=pd.read_csv(source/"head_metrics.csv")
    metric_cols=["support","prevalence","ROC_AUC","average_precision","log_loss","Brier","ECE","macro_one_vs_rest_AUC","conditional_support","MAE","RMSE","rank_IC","bias"]
    head_summary=heads.groupby(["architecture","side","head"],dropna=False)[metric_cols].mean(numeric_only=True).reset_index()
    decision={"schema":"canonical_execution_reliability_ablation_decision_v1","support_winner":support,"context_winner":context,"target_architecture_winner":target,"final_frozen_challenger":final,"mapped_control":control,"promotion_eligible":bool(gates["pass"].all()),"portfolio_replay_authorized":False,"known_evidence_limit":config["mandatory_ic_ev_divergence_diagnostic"]["known_coverage_limit"]}
    return arms,head_summary,gates,decision

def run(source:Path,output:Path,config:Path)->dict[str,Any]:
    if output.exists(): raise FileExistsError(output)
    source_manifest=verify_source(source)
    arms,heads,gates,decision=summarize(source,config)
    stage=Path(tempfile.mkdtemp(prefix="."+output.name+".",dir=output.parent))
    try:
        arms.to_csv(stage/"arm_summary.csv",index=False)
        heads.to_csv(stage/"head_summary.csv",index=False)
        gates.to_csv(stage/"gate_evaluation.csv",index=False)
        write_json(stage/"decision.json",decision)
        outputs={name:sha(stage/name) for name in ("arm_summary.csv","head_summary.csv","gate_evaluation.csv","decision.json")}
        manifest={"schema":"canonical_execution_reliability_ablation_summary_v1","status":"SEALED_RESEARCH_DECISION_NO_PROMOTION_NO_PORTFOLIO_REPLAY","promotion_eligible":False,"source":{"path":str(source),"manifest_sha256":sha(source/"manifest.json"),"schema":source_manifest["schema"]},"config":{"path":str(config),"sha256":sha(config)},"outputs_sha256":outputs,"runner":{"path":str(Path(__file__).resolve()),"sha256":sha(Path(__file__))}}
        write_json(stage/"manifest.json",manifest)
        (stage/"manifest.sha256").write_text(sha(stage/"manifest.json")+"  manifest.json\n")
        os.replace(stage,output)
    except Exception:
        shutil.rmtree(stage,ignore_errors=True)
        raise
    return manifest

def main(argv:Sequence[str]|None=None)->int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",type=Path,default=SOURCE)
    parser.add_argument("--output",type=Path,default=OUTPUT)
    parser.add_argument("--config",type=Path,default=CONFIG)
    args=parser.parse_args(argv)
    print(json.dumps(safe(run(args.source,args.output,args.config)),indent=2))
    return 0

if __name__=="__main__": raise SystemExit(main())
