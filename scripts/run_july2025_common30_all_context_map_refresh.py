#!/usr/bin/env python3
"""Sensitivity-only causal EV-map refresh for compatible July context raw scores.

For each final-v3 baseline/residual/GAM arm, this compares the frozen v3 OOF
map against the same map with sealed July-common30 raw OOF scores appended.
It never changes model scores, arm definitions, selection parameters or 2026
data; common30 is deliberately not treated as population-identical promotion
evidence.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
import sys
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from scripts import run_july_common30_baseline_map_refresh as base

SCHEMA = "july2025_common30_all_context_map_refresh_v1"
STACK = ROOT / "data_perp/artifacts/final_identical_row_regime_stack_gam_ablation_20260730_v3"
JULY_CONTEXT = ROOT / "data_perp/artifacts/july2025_common30_regime_context_raw_score_extension_20260730_v1"
OUT = ROOT / "data_perp/artifacts/july2025_common30_all_context_map_refresh_20260730_v1"
IDENTITY = base.IDENTITY
TARGET, GROSS, COST, ALPHA = base.TARGET, base.GROSS, base.COST, base.ALPHA

ARM_MAP = {
    "baseline": "baseline_raw_residual",
    "residual_regime_only": "residual_trust_regime_raw",
    "residual_transition_only": "residual_trust_transition_raw",
    "residual_combined": "residual_trust_combined_raw",
    "gam_regime_only": "additive_bounded_gam_regime_raw",
    "gam_transition_only": "additive_bounded_gam_transition_raw",
    "gam_combined": "additive_bounded_gam_combined_raw",
}

class RefreshError(RuntimeError): pass

def sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1<<20),b""): h.update(block)
    return h.hexdigest()

def write_json(path: Path, value: Any) -> None:
    tmp=path.with_name(f".{path.name}.{os.getpid()}.partial")
    tmp.write_text(json.dumps(value,indent=2,sort_keys=True,default=str)+"\n")
    os.replace(tmp,path)

def _sealed_stack(root: Path) -> tuple[pd.DataFrame,pd.DataFrame,dict[str,Any]]:
    manifest_path,marker=root/"manifest.json",root/"manifest.sha256"
    if not marker.is_file() or marker.read_text().split(maxsplit=1)[0]!=sha(manifest_path): raise RefreshError("v3 manifest is not sealed")
    manifest=json.loads(manifest_path.read_text())
    if manifest.get("schema")!="final_identical_row_regime_stack_gam_ablation_v3" or manifest.get("status")!="SEALED_STRICT_FORWARD_IDENTICAL_ROW_ABLATION_NON_PROMOTION": raise RefreshError("requires sealed corrected v3 stack")
    history_path,forward_path=root/"historical_oof_scores.parquet",root/"frozen_2026_candidate_scores.parquet"
    for p in (history_path,forward_path):
        if manifest.get("outputs_sha256",{}).get(p.name)!=sha(p): raise RefreshError(f"v3 output checksum mismatch: {p.name}")
    history=pd.read_parquet(history_path);forward=pd.read_parquet(forward_path)
    for name,frame in (("history",history),("forward",forward)):
        frame["__ts__"]=pd.to_datetime(frame["__ts__"],utc=True,errors="raise")
        frame["execution_label_end_utc"]=pd.to_datetime(frame["execution_label_end_utc"],utc=True,errors="raise")
        if frame.duplicated([*IDENTITY,"arm"]).any() or not (frame.__ts__.astype("int64")%pd.Timedelta(hours=1).value==0).all(): raise RefreshError(f"{name} is not exact hourly arm ledger")
    if not history.__ts__.lt(pd.Timestamp("2026-01-01",tz="UTC")).all() or not forward.__ts__.ge(pd.Timestamp("2026-01-01",tz="UTC")).all(): raise RefreshError("v3 split invalid")
    return history,forward,manifest

def _sealed_july(root: Path) -> tuple[pd.DataFrame,dict[str,Any],Path]:
    manifest_path,marker,scores=root/"manifest.json",root/"manifest.sha256",root/"july_raw_context_scores.parquet"
    if not marker.is_file() or marker.read_text().split(maxsplit=1)[0]!=sha(manifest_path): raise RefreshError("July context manifest is not sealed")
    manifest=json.loads(manifest_path.read_text())
    if manifest.get("schema")!="july2025_common30_regime_context_raw_score_extension_v1" or not str(manifest.get("status","")).startswith("SEALED_STRICT_PREJULY"): raise RefreshError("requires sealed strict pre-July context extension")
    if manifest.get("outputs_sha256",{}).get(scores.name)!=sha(scores): raise RefreshError("July context score checksum mismatch")
    data=pd.read_parquet(scores)
    data["__ts__"]=pd.to_datetime(data["__ts__"],utc=True,errors="raise");data["execution_label_end_utc"]=pd.to_datetime(data["execution_label_end_utc"],utc=True,errors="raise")
    if data.duplicated([*IDENTITY,"arm"]).any() or not (data.__ts__.astype("int64")%pd.Timedelta(hours=1).value==0).all(): raise RefreshError("July context data not exact hourly unique")
    if not data.__ts__.between(pd.Timestamp("2025-07-01",tz="UTC"),pd.Timestamp("2025-07-31 23:00",tz="UTC")).all() or not data.execution_label_end_utc.gt(data.__ts__).all(): raise RefreshError("July context dates/endpoints invalid")
    if set(data.arm.unique())!=set(ARM_MAP.values()) or data.groupby("arm").size().ne(44640).any(): raise RefreshError("July arm coverage incomplete")
    return data,manifest,scores

def run(*, stack: Path=STACK, july_context: Path=JULY_CONTEXT, output: Path=OUT) -> Path:
    output=Path(output)
    if output.exists(): raise RefreshError(f"immutable output exists: {output}")
    old,forward,stack_manifest=_sealed_stack(Path(stack));july,july_manifest,july_scores=_sealed_july(Path(july_context))
    result_rows=[]; period_rows=[]; side_rows=[]; calibration_rows=[]; selected_rows=[]; audit=[]
    for v3_arm,july_arm in ARM_MAP.items():
        old_fit=old.loc[old.arm.eq(v3_arm),[*IDENTITY,"raw_score",TARGET,"execution_label_end_utc"]].copy()
        extra=july.loc[july.arm.eq(july_arm),[*IDENTITY,"raw_score",TARGET,"execution_label_end_utc"]].copy()
        app=pd.concat([old_fit,extra],ignore_index=True)
        if old_fit.empty or len(extra)!=44640 or app.duplicated(IDENTITY).any() or not old_fit.execution_label_end_utc.lt(pd.Timestamp("2026-01-01",tz="UTC")).all(): raise RefreshError(f"{v3_arm}: invalid fit support")
        current=forward.loc[forward.arm.eq(v3_arm)].copy()
        if len(current)!=127777: raise RefreshError(f"{v3_arm}: forward common universe differs")
        for fit_name,fit in (("old_cutoff",old_fit),("july_refreshed_common30",app)):
            mapper=base._fit(fit)
            for rank_mode in (False,True):
                name=f"{v3_arm}__{fit_name}__{'rank_preserving' if rank_mode else 'isotonic'}"
                scored=current.copy(); mapped=mapper.predict(scored.raw_score.to_numpy(float)); scored["mapped_score"]=base._strict_rank(mapped,scored.raw_score.to_numpy(float)) if rank_mode else mapped
                summary,period,side,cal=base._evaluate(scored,name)
                summary.update({"v3_arm":v3_arm,"july_raw_arm":july_arm,"fit_window":fit_name,"rank_preserving":rank_mode})
                result_rows.append(summary);period["v3_arm"]=v3_arm;period["fit_window"]=fit_name;period["rank_preserving"]=rank_mode;period_rows.append(period)
                side["v3_arm"]=v3_arm;side["fit_window"]=fit_name;side["rank_preserving"]=rank_mode;side_rows.append(side)
                cal["v3_arm"]=v3_arm;cal["fit_window"]=fit_name;cal["rank_preserving"]=rank_mode;calibration_rows.append(cal)
                picked=base._select(scored);selected_rows.append(picked.loc[picked.selected_global_top10,[*IDENTITY,"raw_score","mapped_score",TARGET,GROSS,COST,"execution_label_end_utc","selected_global_top10"]].assign(map=name,v3_arm=v3_arm,fit_window=fit_name,rank_preserving=rank_mode))
                audit.append({"map":name,"v3_arm":v3_arm,"july_raw_arm":july_arm,"fit_window":fit_name,"rank_preserving":rank_mode,"fit_rows":len(fit),"v3_oof_fit_rows":len(old_fit),"july_common30_fit_rows":len(extra) if fit_name.startswith("july_") else 0,"fit_label_end_max":fit.execution_label_end_utc.max(),"fit_strictly_pre2026":bool(fit.execution_label_end_utc.lt(pd.Timestamp("2026-01-01",tz="UTC")).all()),"no_2026_fit_tuning_or_selection":True})
    summary=pd.DataFrame(result_rows)
    controls=summary.loc[summary.fit_window.eq("old_cutoff")&~summary.rank_preserving,["v3_arm","top10_net_ev","execution_rank_ic"]].rename(columns={"top10_net_ev":"old_top10_net_ev","execution_rank_ic":"old_execution_rank_ic"})
    refreshed=summary.fit_window.eq("july_refreshed_common30")&~summary.rank_preserving
    summary=summary.merge(controls,on="v3_arm",how="left",validate="many_to_one")
    summary["delta_vs_old_isotonic_top10_net_ev"]=summary.top10_net_ev-summary.old_top10_net_ev
    summary["delta_vs_old_isotonic_execution_rank_ic"]=summary.execution_rank_ic-summary.old_execution_rank_ic
    stage=Path(tempfile.mkdtemp(dir=output.parent,prefix=f".{output.name}."))
    try:
        summary.to_csv(stage/"metrics_summary.csv",index=False);pd.concat(period_rows,ignore_index=True).to_parquet(stage/"period_metrics.parquet",index=False);pd.concat(side_rows,ignore_index=True).to_parquet(stage/"side_metrics.parquet",index=False);pd.concat(calibration_rows,ignore_index=True).to_parquet(stage/"calibration_deciles.parquet",index=False);pd.concat(selected_rows,ignore_index=True).to_parquet(stage/"frozen_2026_selected_scores.parquet",index=False);write_json(stage/"mapping_fit_audit.json",audit)
        contract={"sample_cadence":"1h","exact_replay_bar_cadence":"1m_labels_only","arms":"baseline plus all compatible final-v3 residual/GAM context arms; raw July names explicitly cross-walked","fit":"each arm's frozen historical OOF raw scores/labels versus the same ledger with sealed July common30 blocked-OOF raw scores/labels appended","forward_assessment":"same exact 127777 2026 hourly candidates per arm; one pooled global top10 after arm-local map","maps":"ordinary/rank-preserving monotone increasing isotonic; fixed old and July-refreshed windows before reading 2026 outcomes","no_2026_fit_tuning_or_selection":True,"scope_limitation":"July is a frozen common30 cohort, not population-identical to v3; sensitivity-only, cannot promote a map/context/policy","source_stack_manifest_sha256":sha(Path(stack)/"manifest.json"),"source_july_context_manifest_sha256":sha(Path(july_context)/"manifest.json")}
        write_json(stage/"contract.json",contract);files=[p for p in stage.iterdir() if p.is_file()]
        manifest={"schema":SCHEMA,"status":"SEALED_ALL_CONTEXT_CAUSAL_MAP_REFRESH_COMMON30_LIMITED_NON_PROMOTION","promotion_eligible":False,"inputs":{str((Path(stack)/"manifest.json").resolve()):sha(Path(stack)/"manifest.json"),str((Path(july_context)/"manifest.json").resolve()):sha(Path(july_context)/"manifest.json"),str(july_scores.resolve()):sha(july_scores)},"contract":contract,"outputs_sha256":{p.name:sha(p) for p in files}}
        write_json(stage/"manifest.json",manifest);(stage/"manifest.sha256").write_text(f"{sha(stage/'manifest.json')}  manifest.json\n");os.replace(stage,output);return output
    except Exception: shutil.rmtree(stage,ignore_errors=True);raise

if __name__=="__main__": print(run())
