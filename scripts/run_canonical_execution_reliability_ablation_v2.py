#!/usr/bin/env python3
"""Frozen v2 execution-reliability ablation; no policy or replay path.

The runner intentionally has one purpose: produce strict March OOF and one
post-freeze April diagnostic score ledger for A0--A5.  Every learned head is
side-local, HPO/feature selection is fold-local, and the only rankable score is
that arm's own causal recent-EV map.  It refuses to run if the eventual v3
input does not contain the sealed pre-exit labels.
"""
from __future__ import annotations

import argparse, hashlib, json, math, os, shutil, tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, mean_absolute_error, mean_squared_error, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/canonical_execution_reliability_workstream_20260730_v2.json"
INPUT = ROOT / "data_perp/artifacts/canonical_execution_reliability_input_20260730_v3"
DEFAULT_OUTPUT = ROOT / "data_perp/artifacts/canonical_execution_reliability_ablation_20260730_v2"
SIDES = ("long", "short")
TOPS = (.01, .05, .10, .20)
END = "execution_label_end_utc"
TIME = "execution_decision_utc"
NET = "execution_net_ev_12h"
GROSS = "execution_gross_ev_12h"
COST = "execution_cost_return"
IDENTITY = ("candidate_id", "side_name", "__symbol__", "__ts__")
CAPTURE = (
    "target_pre_exit_meaningful_mfe", "target_pre_exit_capture_valid",
    "target_pre_exit_capture_net_positive", "target_pre_exit_economic_capture_ratio",
)
REGIME_SOURCES = (
    "__regime_source_shock_impulse_score__",
    "__regime_source_execution_quality_score__",
    "__regime_source_execution_risk_score__",
    "__regime_source_oi_agreement_score__",
    "__regime_source_compression_score__",
    "__regime_source_loud_breakout_impulse_score__",
    "__regime_source_dirty_shock_avoid_score__",
    "__regime_source_clean_execution_context_score__",
)
PERSISTED_ROW_FIELDS = (
    *IDENTITY, TIME, END, NET, GROSS, COST, "candidate_month",
    "target_pre_exit_meaningful_mfe", "target_pre_exit_economic_opportunity",
    "opportunity_gross_above_cost_0bps", "opportunity_gross_above_cost_25bps",
    "execution_exit_class", "execution_exit_reason", "regime_execution_risk_quintile",
)

class ContractError(RuntimeError): pass

@dataclass(frozen=True)
class Geometry:
    name: str; iterations: int; depth: int; learning_rate: float; l2_leaf_reg: float

def sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()

def safe(x: Any) -> Any:
    if isinstance(x, Mapping): return {str(k):safe(v) for k,v in x.items()}
    if isinstance(x, (list,tuple)): return [safe(v) for v in x]
    if isinstance(x, (Path,pd.Timestamp)): return str(x)
    if isinstance(x,np.generic): return x.item()
    if isinstance(x,float) and not np.isfinite(x): return None
    return x

def write_json(path: Path, value: Mapping[str,Any]) -> None:
    tmp=path.with_name("."+path.name+".tmp"); tmp.write_text(json.dumps(safe(value),indent=2,sort_keys=True)+"\n"); os.replace(tmp,path)

def load_contract(path: Path=CONFIG) -> dict[str,Any]:
    c=json.loads(path.read_text())
    if c.get("schema")!="canonical_execution_reliability_workstream_v2": raise ContractError("v2 frozen contract required")
    if not isinstance(c.get("input_artifact"),str) or not c["input_artifact"]: raise ContractError("contract must bind an input artifact")
    # A declared schema wins; otherwise derive only the artifact's explicit vN suffix.
    # This keeps a v3/v4 wrapper switch contract-driven without a runner rewrite.
    if not c.get("input_schema"):
        version=c["input_artifact"].rsplit("_",1)[-1]
        if version.startswith("v") and version[1:].isdigit(): c["input_schema"]=f"canonical_execution_reliability_input_{version}"
    if not isinstance(c.get("input_schema"),str) or not c["input_schema"]: raise ContractError("contract must bind an input schema")
    if len(c["feature_arms"]["transition_interaction_sources"])!=5: raise ContractError("A5 must have exactly five interaction sources")
    return c

def _context_fields(value: Any) -> list[str]:
    """Accept future context variants as a list or {fields: [...], ...}."""
    if isinstance(value, Mapping): value=value.get("fields",[])
    if not isinstance(value, Sequence) or isinstance(value, (str,bytes)):
        raise ContractError("context ablation variant must declare a feature list")
    return [str(x) for x in value]

def configured_feature_fields(contract: Mapping[str, Any]) -> set[str]:
    arms=contract["feature_arms"]
    fields=set(arms["score4"]) | set(arms["support_S0"]) | set(arms["support_S1B"])
    fields.update(arms.get("candidate_context",[]))
    fields.update(arms["transition_interaction_sources"])
    for value in arms.get("context_ablation_variants",{}).values(): fields.update(_context_fields(value))
    return fields

def verify_evidence_artifacts(contract: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Evidence is external but must be sealed before this runner can cite it."""
    spec=contract.get("mandatory_ic_ev_divergence_diagnostic",{})
    artifacts=spec.get("evidence_artifacts")
    if not isinstance(artifacts,list) or not artifacts:
        raise ContractError("mandatory IC/EV evidence_artifacts must be declared in config")
    sealed=[]
    for item in artifacts:
        if not isinstance(item,Mapping) or not all(k in item for k in ("path","schema","role")):
            raise ContractError("each IC/EV evidence artifact needs path, schema and role")
        root=Path(str(item["path"])); root=root if root.is_absolute() else ROOT/root
        manifest=root/"manifest.json"; seal=root/"manifest.sha256"
        if not manifest.is_file() or not seal.is_file(): raise FileNotFoundError(f"missing required evidence artifact: {root}")
        if sha(manifest)!=seal.read_text().split()[0]: raise ContractError(f"evidence manifest seal mismatch: {root}")
        payload=json.loads(manifest.read_text())
        if payload.get("schema")!=item["schema"]: raise ContractError(f"evidence schema mismatch: {root}")
        outputs=payload.get("outputs_sha256")
        if isinstance(outputs,Mapping) and outputs:
            declared=[(root/str(name),str(expected)) for name,expected in outputs.items()]
        else:
            legacy=payload.get("outputs")
            if not isinstance(legacy,Mapping) or not legacy: raise ContractError(f"evidence output hashes absent: {root}")
            declared=[]
            for name,spec in legacy.items():
                if not isinstance(spec,Mapping) or not spec.get("path") or not spec.get("sha256"): raise ContractError(f"evidence output declaration invalid: {root}/{name}")
                candidate=Path(str(spec["path"])); declared.append((candidate if candidate.is_absolute() else ROOT/candidate,str(spec["sha256"])))
        for candidate,expected in declared:
            if not candidate.is_file() or sha(candidate)!=expected: raise ContractError(f"evidence output hash mismatch: {candidate}")
        sealed.append({"path":str(root),"schema":item["schema"],"role":item["role"],"manifest_sha256":sha(manifest),"outputs_sha256":{str(path):digest for path,digest in declared}})
    return sealed

def verify_input(root: Path, contract: Mapping[str,Any]) -> tuple[pd.DataFrame,dict[str,Any]]:
    manifest=root/"manifest.json"; seal=root/"manifest.sha256"; panel=root/"panel.parquet"; roles_path=root/"feature_roles.json"
    if not all(p.is_file() for p in (manifest,seal,panel,roles_path)): raise FileNotFoundError("sealed configured reliability input is unavailable")
    if sha(manifest)!=seal.read_text().split()[0]: raise ContractError("configured input manifest seal mismatch")
    m=json.loads(manifest.read_text())
    if m.get("schema")!=contract["input_schema"]: raise ContractError("canonical reliability input schema mismatch")
    declared_outputs=m.get("outputs_sha256",{})
    if not isinstance(declared_outputs,Mapping) or not declared_outputs: raise ContractError("configured input has no output hash ledger")
    for name,expected in declared_outputs.items():
        candidate=root/str(name)
        if not candidate.is_file() or sha(candidate)!=str(expected): raise ContractError(f"configured input output hash mismatch: {name}")
    if declared_outputs.get("panel.parquet")!=sha(panel): raise ContractError("configured input panel hash mismatch")
    if declared_outputs.get("feature_roles.json")!=sha(roles_path): raise ContractError("configured input feature-role hash mismatch")
    role=json.loads(roles_path.read_text())
    alias_fields={field for spec in contract["feature_arms"].get("candidate_context_aliases",{}).values() for field in (spec.get("canonical_field"),spec.get("alias")) if field}
    cols=configured_feature_fields(contract) | alias_fields | set(CAPTURE) | set(IDENTITY) | set(REGIME_SOURCES) | {TIME,END,NET,GROSS,COST,"target_clean_favorable_first","target_competing_class","target_positive_net_magnitude","target_adverse_loss_magnitude","base_oof_score","execution_exit_class","execution_exit_reason","target_pre_exit_economic_opportunity","opportunity_gross_above_cost_0bps","opportunity_gross_above_cost_25bps","candidate_month","__decision_ts__","__label_end_ts__"}
    missing=cols-set(pd.read_parquet(panel,columns=None).columns)
    if missing: raise ContractError("v3 input missing contract fields: "+str(sorted(missing)))
    x=pd.read_parquet(panel,columns=sorted(cols))
    for c in ("__ts__",TIME,END,"__decision_ts__","__label_end_ts__"): x[c]=pd.to_datetime(x[c],utc=True,errors="raise")
    if len(x)!=110730 or x.duplicated(["candidate_id","side_name"]).any(): raise ContractError("110730 candidate+side v3 identity required")
    if not x.side_name.isin(SIDES).all() or not x[END].eq(x[TIME]+pd.Timedelta(hours=12)).all(): raise ContractError("side or exact H12 contract failed")
    if not np.allclose(x[GROSS]-x[COST],x[NET],atol=1e-7): raise ContractError("cost counted incorrectly")
    forbidden=tuple(contract["forbidden_inputs"])+tuple(contract["action_layer_exclusions"])
    features=list(configured_feature_fields(contract))
    bad=[f for f in features if any(t.lower() in f.lower() for t in ("mapped","dae","gmm","full_horizon","time_to","mae","target_price","wait"))]
    if bad: raise ContractError("forbidden configured input: "+str(bad))
    allowed=set(role.get("default_ev_inputs",[])) | set(role.get("transition_interaction_sources",[]))
    unapproved=set(features)-allowed
    if unapproved: raise ContractError("configured feature not approved by sealed feature roles: "+str(sorted(unapproved)))
    risk="__regime_source_execution_risk_score__"
    reference=x.loc[x.candidate_month.eq("2025-03"),risk].astype(float)
    edges=np.unique(reference.quantile(np.linspace(0,1,6)).to_numpy())
    if len(edges)!=6: raise ContractError("cannot form frozen five-bin execution-risk regime attribution")
    # March learns the internal cut points; open outer bounds keep April
    # attribution exhaustive without refitting its regime geometry.
    edges[0],edges[-1]=-np.inf,np.inf
    x["regime_execution_risk_quintile"]=pd.cut(x[risk].astype(float),bins=edges,labels=["Q1","Q2","Q3","Q4","Q5"],include_lowest=True).astype(str)
    if not x["regime_execution_risk_quintile"].isin(["Q1","Q2","Q3","Q4","Q5"]).all():
        raise ContractError("frozen execution-risk regime attribution is not exhaustive")
    aliases=contract["feature_arms"].get("candidate_context_aliases",{})
    for name,spec in aliases.items():
        canonical=spec.get("canonical_field"); alias=spec.get("alias")
        if canonical in x and alias in x and not np.array_equal(x[canonical].to_numpy(),x[alias].to_numpy(),equal_nan=True):
            raise ContractError(f"candidate-context alias mismatch: {name}")
    return x,m

def outer_masks(frame: pd.DataFrame, fold: Mapping[str,Any]) -> tuple[np.ndarray,np.ndarray]:
    start=pd.Timestamp(fold["validation_start_utc"]); end=pd.Timestamp(fold["validation_end_utc"])
    valid=frame[TIME].ge(start).to_numpy() & frame[TIME].lt(end).to_numpy()
    train=frame[TIME].lt(start).to_numpy() & frame[END].lt(start).to_numpy()
    if not valid.any() or not train.any() or not frame.loc[train,END].lt(start).all(): raise ContractError("strict decision-time H12 purge failed")
    return train,valid

def inner_masks(train: pd.DataFrame) -> tuple[np.ndarray,np.ndarray]:
    times=np.sort(train[TIME].drop_duplicates().to_numpy())
    if len(times)<4: raise ContractError("insufficient inner timestamps")
    start=pd.Timestamp(times[max(1,int(math.floor(len(times)*.75)))])
    valid=train[TIME].ge(start).to_numpy()
    fit=train[TIME].lt(start).to_numpy() & train[END].lt(start).to_numpy()
    if not fit.any() or not valid.any() or not train.loc[fit,END].lt(start).all(): raise ContractError("inner exact H12 purge failed")
    return fit,valid

def interaction_features(train: pd.DataFrame, score: pd.Series, sources: Sequence[str], target: pd.DataFrame) -> pd.DataFrame:
    """Training-side means/std only; exactly five bounded interactions."""
    if len(sources)!=5: raise ContractError("A5 requires exactly five interactions")
    out=pd.DataFrame(index=target.index)
    for source in sources:
        mean=float(train[source].mean()); std=float(train[source].std(ddof=0))
        z=(target[source].astype(float)-mean)/(std if std>1e-12 else 1.)
        out[f"interaction__base_oof_score_x_{source}"]=(target[score.name].astype(float)*z).clip(-1.,1.)
    return out

def feature_arm(contract: Mapping[str,Any], variant: str) -> list[str]:
    a=contract["feature_arms"]; base=list(a["score4"])
    if variant=="score4+support_S0": return base+list(a["support_S0"])
    if variant=="score4+support_S1B": return base+list(a["support_S1B"])
    if variant=="score4+support_S0+support_S1B": return base+list(a["support_S0"])+list(a["support_S1B"])
    raise KeyError(variant)

def context_variants(contract: Mapping[str, Any], base_features: Sequence[str] | None = None) -> list[tuple[str, list[str]]]:
    """Optional future config support for one-field/group-at-a-time context arms."""
    declared=contract["feature_arms"].get("context_ablation_variants",{})
    architecture=contract.get("architectures",{}).get("A1_context")
    if architecture is None:
        return []
    if not isinstance(declared,Mapping): raise ContractError("A1_context requires feature_arms.context_ablation_variants mapping")
    if isinstance(architecture,Mapping):
        names=architecture.get("feature_variants")
        if names is None:
            source=architecture.get("feature_variants_from")
            if source!="feature_arms.context_ablation_variants": raise ContractError("A1_context must bind its declared variant mapping")
            names=list(declared)
        base_variant=architecture.get("base_variant","score4+support_S0+support_S1B")
    elif isinstance(architecture,list):
        names=architecture; base_variant="score4+support_S0+support_S1B"
    else: raise ContractError("architectures.A1_context must be a mapping or list")
    base=list(base_features) if base_features is not None else feature_arm(contract,str(base_variant)); out=[]
    for name in names:
        if name not in declared: raise ContractError(f"A1_context missing declared fields for {name}")
        fields=_context_fields(declared[name])
        if not fields and str(name)!="context_none": raise ContractError(f"A1_context {name} has no fields")
        out.append((str(name),list(dict.fromkeys([*base,*fields]))))
    return out

def _model(task: str, geometry: Geometry, seed: int):
    from catboost import CatBoostClassifier, CatBoostRegressor
    common=dict(iterations=geometry.iterations,depth=geometry.depth,learning_rate=geometry.learning_rate,l2_leaf_reg=geometry.l2_leaf_reg,random_seed=seed,verbose=False,allow_writing_files=False,thread_count=1)
    if task=="binary": return CatBoostClassifier(loss_function="Logloss",**common)
    if task=="multi": return CatBoostClassifier(loss_function="MultiClass",**common)
    return CatBoostRegressor(loss_function="RMSE",**common)

def _prediction(model: Any, x: pd.DataFrame, task: str) -> np.ndarray:
    if task=="reg": return np.asarray(model.predict(x),dtype=float)
    p=np.asarray(model.predict_proba(x),dtype=float)
    if task=="multi": return p
    classes=np.asarray(model.classes_).astype(str)
    hit=np.flatnonzero(classes=="1")
    if len(hit)!=1: raise ContractError(f"binary model lacks unique positive class 1: {classes.tolist()}")
    return p[:,int(hit[0])]

def _loss(y: np.ndarray,p:np.ndarray,task:str, *, classes: Sequence[str] | None = None) -> tuple[float,float]:
    if task=="binary":
        q=np.clip(p,1e-6,1-1e-6); return float(log_loss(y,q,labels=[0,1])),float(brier_score_loss(y,q))
    if task=="multi":
        if classes is None: raise ContractError("multiclass loss requires model class order")
        return float(log_loss(np.asarray(y).astype(str),np.clip(p,1e-6,1-1e-6),labels=list(np.asarray(classes).astype(str)))),np.nan
    return float(mean_absolute_error(y,p)), -rank_ic(y,p)

def ece_10bin(y: np.ndarray, p: np.ndarray) -> tuple[float,list[dict[str,Any]]]:
    """Canonical 10 equal-width-bin expected calibration error."""
    y=np.asarray(y,float); p=np.clip(np.asarray(p,float),0.,1.)
    bucket=np.minimum((p*10).astype(int),9); diagnostics=[]; ece=0.
    for b in range(10):
        mask=bucket==b; support=int(mask.sum())
        if not support:
            diagnostics.append({"bin":b,"support":0,"mean_prediction":None,"event_rate":None,"absolute_gap":None})
            continue
        mean=float(p[mask].mean()); rate=float(y[mask].mean()); gap=abs(mean-rate); ece+=support/len(y)*gap
        diagnostics.append({"bin":b,"support":support,"mean_prediction":mean,"event_rate":rate,"absolute_gap":gap})
    return float(ece),diagnostics

def head_metrics(y: np.ndarray, p: np.ndarray, task: str, *, classes: Sequence[str] | None = None) -> dict[str, Any]:
    """Per-head diagnostics; economics are reported only after causal mapping."""
    if task=="binary":
        q=np.clip(np.asarray(p,float),1e-6,1-1e-6); y=np.asarray(y,int)
        ece,bins=ece_10bin(y,q)
        return {"prevalence":float(y.mean()),"ROC_AUC":float(roc_auc_score(y,q)) if np.unique(y).size>1 else np.nan,
                "average_precision":float(average_precision_score(y,q)) if np.unique(y).size>1 else np.nan,
                "log_loss":float(log_loss(y,q,labels=[0,1])),"Brier":float(brier_score_loss(y,q)),"ECE":ece,"ECE_bins":json.dumps(bins,sort_keys=True),"bias":float((q-y).mean())}
    if task=="multi":
        q=np.clip(np.asarray(p,float),1e-6,1-1e-6)
        labels=list(classes) if classes is not None else list(sorted(pd.Series(y).astype(str).unique()))
        if q.shape[1] != len(labels): raise ContractError("multiclass probability/class-order mismatch")
        y_text=np.asarray(y).astype(str)
        one_vs_rest=[]; briers={}
        for i,label in enumerate(labels):
            event=(y_text==label).astype(int)
            one_vs_rest.append(float(roc_auc_score(event,q[:,i])) if event.min()!=event.max() else np.nan)
            briers[label]=float(brier_score_loss(event,q[:,i]))
        class_ece={}; class_bins={}
        for i,label in enumerate(labels):
            class_ece[label],class_bins[label]=ece_10bin((y_text==label).astype(int),q[:,i])
        return {"class_support":json.dumps(pd.Series(y_text).value_counts().to_dict(),sort_keys=True),"log_loss":float(log_loss(y_text,q,labels=labels)),
                "macro_one_vs_rest_AUC":float(np.nanmean(one_vs_rest)) if np.isfinite(one_vs_rest).any() else np.nan,
                "per_class_Brier":json.dumps(briers,sort_keys=True),"ECE":float(np.mean(list(class_ece.values()))),"ECE_bins":json.dumps(class_bins,sort_keys=True)}
    y=np.asarray(y,float);q=np.asarray(p,float)
    return {"conditional_support":int(len(y)),"MAE":float(mean_absolute_error(y,q)),"RMSE":float(mean_squared_error(y,q)**.5),"rank_IC":rank_ic(y,q),"bias":float((q-y).mean())}

def rank_ic(a: Sequence[float],b: Sequence[float])->float:
    x=pd.DataFrame({"a":a,"b":b}).dropna()
    return float(x.a.corr(x.b,method="spearman")) if len(x)>2 and x.a.nunique()>1 and x.b.nunique()>1 else np.nan

def fit_head(train: pd.DataFrame, valid: pd.DataFrame, features: Sequence[str], ycol: str, task: str, geometries: Sequence[Geometry], seed:int) -> tuple[np.ndarray,dict[str,Any],Any,list[str]]:
    """Geometry on a legal inner tail; PVC selection then outer-train refit."""
    y=train[ycol].to_numpy(); fit,iv=inner_masks(train); candidates=[]
    for g in geometries:
        model=_model(task,g,seed); model.fit(train.loc[fit,list(features)],y[fit])
        p=_prediction(model,train.loc[iv,list(features)],task); primary,secondary=_loss(y[iv],p,task,classes=(model.classes_ if task=="multi" else None))
        candidates.append((primary,secondary,g.depth,g,model))
    primary,secondary,_,g,inner=sorted(candidates,key=lambda z:(z[0],z[1],z[2]))[0]
    imp=np.maximum(np.asarray(inner.get_feature_importance(type="PredictionValuesChange"),dtype=float),0.)
    features=list(dict.fromkeys(features));
    if len(features)!=len(imp): raise ContractError("PVC feature/importance alignment failed")
    anchors=[f for f in features if f in ("raw_score","score_base_alpha","score_residual_expected_ev","direct_q25_return","base_oof_score")]
    if len(anchors)>24: raise ContractError("score anchors exceed maximum selected features")
    order=np.argsort(-imp); total=float(imp.sum()); max_features=min(24,len(features)); minimum=min(8,max_features)
    chosen=[]; selected_importance=0.
    for pos in order:
        if imp[pos]<=0: continue
        field=features[int(pos)]
        if field not in chosen: chosen.append(field); selected_importance+=float(imp[pos])
        if len(chosen)>=minimum and (total<=0 or selected_importance/total>=.9): break
        if len(chosen)>=max_features: break
    chosen=list(dict.fromkeys([*anchors,*chosen]))
    for pos in order:
        if len(chosen)>=minimum: break
        field=features[int(pos)]
        if field not in chosen: chosen.append(field)
    chosen=chosen[:max_features]
    if len(chosen)<minimum: raise ContractError("cannot satisfy minimum feature-selection count")
    final=_model(task,g,seed+1); final.fit(train.loc[:,chosen],y)
    actual_importance=float(sum(imp[features.index(f)] for f in chosen))
    return _prediction(final,valid.loc[:,chosen],task),{"geometry":g.name,"features":chosen,"inner_primary":primary,"inner_secondary":secondary,"pvc_cumulative_fraction":(actual_importance/total if total>0 else 0.),"pvc_zero_importance_backfill":bool(any(imp[features.index(f)]<=0 for f in chosen))},final,chosen

def combine(architecture:str, heads: Mapping[str,np.ndarray], classes: Sequence[str]|None=None)->np.ndarray:
    if architecture=="A1": return heads["direct"]
    if architecture=="A2":
        return heads["mfe"]*(heads["capture"]*np.maximum(heads["gain"],0)-(1-heads["capture"])*np.maximum(heads["loss"],0))
    if architecture=="A3": return heads["clean"]*(heads["clean_payoff"])-(1-heads["clean"])*np.maximum(heads["adverse"],0)
    if architecture=="A4":
        return sum(heads[f"class_{name}"]*heads[f"payoff_{name}"] for name in (classes or ()))
    raise KeyError(architecture)

def head_specs(architecture:str)->list[tuple[str,str,str,str|None]]:
    if architecture=="A1": return [("direct",NET,"reg",None)]
    if architecture=="A2": return [("mfe","target_pre_exit_meaningful_mfe","binary",None),("capture","target_pre_exit_capture_net_positive","binary","capture_valid_and_meaningful"),("gain","target_positive_net_magnitude","reg","capture_positive_pre_exit"),("loss","target_adverse_loss_magnitude","reg","capture_nonpositive_pre_exit")]
    if architecture=="A3": return [("clean","target_clean_favorable_first","binary",None),("clean_payoff",NET,"reg","target_clean_favorable_first"),("adverse","target_adverse_loss_magnitude","reg","not_clean_favorable_first")]
    if architecture=="A4":
        return [("class", "target_competing_class","multi",None),("payoff_favorable_first",NET,"reg","target_competing_class=favorable_first"),("payoff_adverse_first_or_conflict",NET,"reg","target_competing_class=adverse_first_or_conflict"),("payoff_timeout",NET,"reg","target_competing_class=timeout")]
    raise KeyError(architecture)

def condition_mask(frame: pd.DataFrame, condition: str | None) -> np.ndarray:
    if not condition:return np.ones(len(frame),dtype=bool)
    if condition=="capture_valid_and_meaningful":
        return (frame.target_pre_exit_capture_valid.astype(bool)&frame.target_pre_exit_meaningful_mfe.astype(bool)).to_numpy()
    if condition=="capture_positive_pre_exit":
        return (frame.target_pre_exit_capture_valid.astype(bool)&frame.target_pre_exit_meaningful_mfe.astype(bool)&frame.target_pre_exit_capture_net_positive.astype(bool)).to_numpy()
    if condition=="capture_nonpositive_pre_exit":
        return (frame.target_pre_exit_capture_valid.astype(bool)&frame.target_pre_exit_meaningful_mfe.astype(bool)&~frame.target_pre_exit_capture_net_positive.astype(bool)).to_numpy()
    if condition=="not_clean_favorable_first":
        return (~frame.target_clean_favorable_first.astype(bool)).to_numpy()
    if "=" in condition:
        c,v=condition.split("=",1); return frame[c].astype(str).eq(v).to_numpy()
    return frame[condition].astype(bool).to_numpy()

def mask_for(train:pd.DataFrame, condition:str|None)->pd.DataFrame:
    return train.loc[condition_mask(train,condition)].copy()

def global_book_weights(frame: pd.DataFrame, score: str, fraction: float) -> tuple[pd.Series,dict[str,float]]:
    """Expected fractional membership of one pooled-global random-tie book."""
    if not len(frame): raise ContractError("cannot select an empty global book")
    n=max(1,int(math.ceil(len(frame)*fraction))); ordered=frame.sort_values([score,"candidate_id"],ascending=[False,True],kind="mergesort")
    cut=float(ordered[score].iloc[n-1]); above=frame[score].gt(cut); tie=frame[score].eq(cut); need=int(n-above.sum()); population=int(tie.sum())
    if need<0 or not population: raise ContractError("invalid cutoff tie accounting")
    weights=pd.Series(0.,index=frame.index); weights.loc[above]=1.
    weights.loc[tie]=need/population
    return weights,{"selected_rows":n,"cutoff":cut,"boundary_tie_population":population,"boundary_tie_population_fraction":float(population/len(frame)),"cutoff_tie_selected_share":float(need/n),"cutoff_tie_fraction":float(need/n)}

def random_tie_expected(frame:pd.DataFrame, score:str, fraction:float)->dict[str,float]:
    weights,meta=global_book_weights(frame,score,fraction); n=meta["selected_rows"]
    result=dict(meta)
    for field,label in ((NET,"net"),(GROSS,"gross"),(COST,"cost")):
        if field in frame:
            result[f"random_tie_expected_{label}_bps"]=float((weights*frame[field]).sum()/n*1e4)
    for field,label in (("target_pre_exit_economic_opportunity","pre_exit_economic_opportunity_rate"),("target_pre_exit_meaningful_mfe","pre_exit_meaningful_mfe_rate"),("opportunity_gross_above_cost_0bps","opportunity_gross_above_cost_0bps_rate")):
        if field in frame: result[f"random_tie_expected_{label}"]=float((weights*frame[field].astype(float)).sum()/n)
    selected=frame.loc[weights.gt(0)]
    result["score_rank_IC"]=rank_ic(frame[score],frame[NET])
    result["tail_rank_IC"]=rank_ic(selected[score],selected[NET])
    result["tail_rank_IC_support"]=int(len(selected))
    return result

def causal_map(scores:pd.DataFrame, *, pooled_min:int=2000, side_min:int=1000)->tuple[pd.DataFrame,pd.DataFrame]:
    """Score-specific 21d pooled isotonic plus side residual, daily and causal."""
    out=scores.copy();out["mapped_score"]=np.nan;out["mapped_eligible"]=False; audits=[]
    for day,idx in out.groupby(out[TIME].dt.floor("D"),sort=True).groups.items():
        day=pd.Timestamp(day); ref=out.loc[out[END].ge(day-pd.Timedelta(days=21)) & out[END].lt(day) & out["score_available_utc"].lt(day)].copy(); pos=np.asarray(list(idx))
        if not (ref[END].lt(day).all() if len(ref) else True): raise ContractError("mapper used unresolved label")
        ready=len(ref)>=pooled_min and ref.raw_score.nunique()>1
        audits.append({"snapshot_utc":day,"evaluation_rows":len(pos),"reference_rows":len(ref),"strict_causal_window_pass":True,"pooled_support_pass":ready})
        if not ready:continue
        pooled=IsotonicRegression(out_of_bounds="clip").fit(ref.raw_score,ref[NET]); raw=out.loc[pos,"raw_score"].to_numpy(float); val=pooled.predict(raw)
        for side in SIDES:
            bm=out.loc[pos,"side_name"].eq(side).to_numpy(); sub=ref.loc[ref.side_name.eq(side)]
            if bm.any() and len(sub)>=side_min and sub.raw_score.nunique()>1:
                sm=IsotonicRegression(out_of_bounds="clip").fit(sub.raw_score,sub[NET]); w=len(sub)/(len(sub)+side_min); val[bm]+=w*(sm.predict(raw[bm])-pooled.predict(raw[bm]))
        out.loc[pos,"mapped_score"]=val;out.loc[pos,"mapped_eligible"]=True
    return out,pd.DataFrame(audits)

def metrics(frame:pd.DataFrame, config:str, stage:str)->list[dict[str,Any]]:
    rows=[]
    windows=[("aggregate",frame)]
    if "outer_fold" in frame:
        windows.extend((str(name),part) for name,part in frame.groupby("outer_fold",sort=True))
    for window,part in windows:
        for mapped,col in (("raw","raw_score"),("mapped","mapped_score")):
            x=part if mapped=="raw" else part.loc[part.mapped_eligible].copy()
            if not len(x): continue
            for f in TOPS:
                r={"config":config,"stage":stage,"window":window,"score_kind":mapped,"top_fraction":f,**random_tie_expected(x,col,f)}
                r["candidate_rows"]=len(x)
                r["score_bias_bps"]=float((x[col]-x[NET]).mean()*1e4)
                r["score_MAE_bps"]=float((x[col]-x[NET]).abs().mean()*1e4)
                r["tail_rank_IC_scope"]="boundary-inclusive when the random-tie cutoff has multiple candidates"
                rows.append(r)
    return rows

def global_book_attribution(frame: pd.DataFrame, config: str, stage: str, contract: Mapping[str,Any]) -> list[dict[str,Any]]:
    """Attribute a globally admitted mapped book; it never creates group quotas."""
    rows=[]
    windows=[("aggregate",frame)]
    if "outer_fold" in frame: windows.extend((str(name),part) for name,part in frame.groupby("outer_fold",sort=True))
    dimensions=contract.get("global_book_attribution",{}).get("dimensions",{})
    if not isinstance(dimensions,Mapping): raise ContractError("global-book attribution dimensions must be a mapping")
    for window,part in windows:
        x=part.loc[part.mapped_eligible].copy()
        if not len(x): continue
        for fraction in TOPS:
            weights,meta=global_book_weights(x,"mapped_score",fraction); denominator=meta["selected_rows"]
            for dimension_name,dimension in dimensions.items():
                if dimension_name=="regime" and "execution_risk_score" in str(dimension): dimension="regime_execution_risk_quintile"
                if dimension_name=="exit" and "execution_exit_class" in str(dimension): dimension="execution_exit_class"
                if dimension not in x: continue
                for value,index in x.groupby(dimension,dropna=False,sort=True).groups.items():
                    w=weights.loc[index]; contribution=float((w*x.loc[index,NET]).sum()/denominator*1e4)
                    selected=x.loc[index]
                    positive=(selected[NET]>0).astype(float)
                    opportunity=selected["target_pre_exit_economic_opportunity"].astype(float)
                    total_opportunity=float(x["target_pre_exit_economic_opportunity"].astype(float).sum())
                    rows.append({"config":config,"stage":stage,"window":window,"score_kind":"mapped","top_fraction":fraction,"dimension":dimension_name,"field":dimension,"value":str(value),"candidate_rows_group":int(len(selected)),"expected_selected_rows":float(w.sum()),"selected_book_share":float(w.sum()/denominator),"net_bps_contribution":contribution,"gross_bps_contribution":float((w*selected[GROSS]).sum()/denominator*1e4),"cost_bps_contribution":float((w*selected[COST]).sum()/denominator*1e4),"positive_net_rate":float((w*positive).sum()/w.sum()) if w.sum() else np.nan,"economic_opportunity_precision":float((w*opportunity).sum()/w.sum()) if w.sum() else np.nan,"economic_opportunity_recall_contribution":float((w*opportunity).sum()/total_opportunity) if total_opportunity else np.nan,"conditional_positive_payoff":float((w*selected.loc[selected[NET]>0,NET]).sum()/w.loc[selected[NET]>0].sum()) if (w.loc[selected[NET]>0].sum()) else np.nan,"conditional_adverse_payoff":float((w*(-selected.loc[selected[NET]<0,NET])).sum()/w.loc[selected[NET]<0].sum()) if (w.loc[selected[NET]<0].sum()) else np.nan,**meta})
    return rows

def run_architecture(dev:pd.DataFrame, april:pd.DataFrame, contract:Mapping[str,Any], architecture:str, variant:str, geometries:Sequence[Geometry], *, interactions:bool=False, features_override: Sequence[str] | None=None)->tuple[pd.DataFrame,pd.DataFrame,list[dict[str,Any]],dict[str,Any]]:
    features=list(features_override) if features_override is not None else feature_arm(contract,variant); folds=contract["outer_folds"]; oof=[]; heads=[]; recipes={}
    for fi,fold in enumerate(folds):
        tr,va=outer_masks(dev,fold)
        for side in SIDES:
            train=dev.loc[tr & dev.side_name.eq(side)].copy(); valid=dev.loc[va & dev.side_name.eq(side)].copy()
            if interactions:
                inter_train=interaction_features(train,train.base_oof_score,contract["feature_arms"]["transition_interaction_sources"],train); inter_valid=interaction_features(train,train.base_oof_score,contract["feature_arms"]["transition_interaction_sources"],valid)
                for c in inter_train: train[c]=inter_train[c];valid[c]=inter_valid[c]
                local_features=[*features,*inter_train.columns]
            else: local_features=features
            pred={}; class_names=("favorable_first","adverse_first_or_conflict","timeout")
            for hn,target,task,condition in head_specs(architecture):
                local=mask_for(train,condition)
                if len(local)<100 or (task=="binary" and local[target].nunique()<2): raise ContractError(f"insufficient {architecture}/{fold['name']}/{side}/{hn} support")
                p,detail,model,chosen=fit_head(local,valid,local_features,target,task,geometries,20260730+fi)
                metric_mask=condition_mask(valid,condition)
                metric_y=valid.loc[metric_mask,target].to_numpy()
                metric_p=p[metric_mask]
                metric_classes=None
                if hn=="class":
                    classes=np.asarray(model.classes_).astype(str)
                    if set(classes)!=set(class_names): raise ContractError("A4 multiclass classes drifted")
                    for name in class_names: pred[f"class_{name}"]=p[:,int(np.flatnonzero(classes==name)[0])]
                    metric_classes=classes
                else: pred[hn]=p
                heads.append({"architecture":architecture,"variant":variant,"fold":fold["name"],"side":side,"head":hn,"target":target,"task":task,"support":len(local),**detail,**head_metrics(metric_y,metric_p,task,classes=metric_classes)})
                recipes[(side,hn)]={"source_fold":fold["name"],"geometry":detail["geometry"],"features":chosen}
            valid=valid.loc[:,list(PERSISTED_ROW_FIELDS)].copy();valid["raw_score"]=combine(architecture,pred,class_names);valid["score_available_utc"]=valid[TIME];valid["outer_fold"]=fold["name"];valid["candidate_score_is_oof"]=True;oof.append(valid)
    march=pd.concat(oof,ignore_index=True)
    # Freeze a single production recipe using all resolved pre-April March rows.
    # Its geometry and PVC selection use that training set's own chronological inner tail;
    # April is prediction-only and never participates in HPO/feature selection.
    scored=[]
    freeze_recipes={}
    for side in SIDES:
        train=dev.loc[dev.side_name.eq(side)&dev[END].lt(pd.Timestamp("2025-04-01T00:00:00Z"))].copy(); valid=april.loc[april.side_name.eq(side)].copy(); pred={}; class_names=("favorable_first","adverse_first_or_conflict","timeout")
        if interactions:
            it=interaction_features(train,train.base_oof_score,contract["feature_arms"]["transition_interaction_sources"],train); iv=interaction_features(train,train.base_oof_score,contract["feature_arms"]["transition_interaction_sources"],valid)
            for c in it: train[c]=it[c];valid[c]=iv[c]
            frozen_features=[*features,*it.columns]
        else:
            frozen_features=features
        for hn,target,task,condition in head_specs(architecture):
            local=mask_for(train,condition)
            p,detail,model,chosen=fit_head(local,valid,frozen_features,target,task,geometries,20269999)
            freeze_recipes[(side,hn)]={"source":"pre_april_full_march_inner_hpo_pvc","geometry":detail["geometry"],"features":chosen,"inner_primary":detail["inner_primary"],"inner_secondary":detail["inner_secondary"]}
            if hn=="class":
                classes=np.asarray(model.classes_).astype(str)
                if set(classes)!=set(class_names): raise ContractError("A4 frozen classes drifted")
                for name in class_names:pred[f"class_{name}"]=p[:,int(np.flatnonzero(classes==name)[0])]
            else:pred[hn]=p
        valid=valid.loc[:,list(PERSISTED_ROW_FIELDS)].copy();valid["raw_score"]=combine(architecture,pred,class_names);valid["score_available_utc"]=valid[TIME];valid["outer_fold"]="april_frozen";valid["candidate_score_is_oof"]=False;scored.append(valid)
    return march,pd.concat(scored,ignore_index=True),heads,{"architecture":architecture,"variant":variant,"features":features,"oof_recipes":recipes,"freeze_recipes":freeze_recipes,"freeze_recipe_source":"pre_april_full_march_inner_hpo_pvc"}

def selection_score(march:pd.DataFrame,contract:Mapping[str,Any])->dict[str,float]:
    mapped,audit=causal_map(march); vals=[]
    for f in contract["outer_folds"]:
        if f["role"].startswith("architecture"):
            raw=mapped.loc[mapped.outer_fold.eq(f["name"])]
            if not len(raw) or not raw.mapped_eligible.all():
                raise ContractError(f"selection fold lacks full causal-map coverage: {f['name']}")
            vals.append(random_tie_expected(raw,"mapped_score",.10)["random_tie_expected_net_bps"])
    if not vals or not np.isfinite(vals).all(): raise ContractError("selection economics coverage is incomplete")
    a=np.asarray(vals,float); return {"mean":float(a.mean()),"std":float(a.std()),"worst":float(a.min()),"latest_fold":float(a[-1]),"fold_top10_net_bps":json.dumps(dict(zip([f["name"] for f in contract["outer_folds"] if f["role"].startswith("architecture")],a.tolist())),sort_keys=True),"mapping_coverage_pass":True,"objective":float(a.mean()-.5*a.std()+.25*a.min())}

def selection_key(config: str, march: pd.DataFrame, head_rows: Sequence[Mapping[str,Any]], contract: Mapping[str,Any]) -> tuple[float,float,float,str]:
    """Frozen tie-break: objective, then fewer selected features, shallower, lexical."""
    objective=selection_score(march,contract)["objective"]
    if not np.isfinite(objective): objective=-np.inf
    sizes=[len(x.get("features",[])) for x in head_rows if x.get("features") is not None]
    depth={g["name"]:g["depth"] for g in contract["catboost_hpo"]["geometries"]}
    depths=[depth[x["geometry"]] for x in head_rows if x.get("geometry") in depth]
    return (-objective,float(np.mean(sizes)) if sizes else np.inf,float(np.mean(depths)) if depths else np.inf,config)

def canonical_oof_control(dev: pd.DataFrame, score: str, contract: Mapping[str, Any]) -> pd.DataFrame:
    """A0 must use the same mapping-calibration/selection rows as learned arms."""
    parts=[]
    for fold in contract["outer_folds"]:
        _,valid=outer_masks(dev,fold)
        x=dev.loc[valid,list(PERSISTED_ROW_FIELDS)+[score]].copy().rename(columns={score:"raw_score"})
        x["score_available_utc"]=x[TIME]; x["outer_fold"]=fold["name"]; x["candidate_score_is_oof"]=True
        parts.append(x)
    out=pd.concat(parts,ignore_index=True)
    if out.duplicated(["candidate_id","side_name"]).any(): raise ContractError("A0 OOF cohort identity duplicated")
    return out

def run(input_root:Path,output_dir:Path,config_path:Path=CONFIG)->dict[str,Any]:
    if output_dir.exists():raise FileExistsError(output_dir)
    c=load_contract(config_path)
    configured_input=ROOT/c["input_artifact"]
    if input_root==INPUT: input_root=configured_input
    if input_root.resolve()!=configured_input.resolve(): raise ContractError("input root differs from the frozen contract")
    evidence=verify_evidence_artifacts(c)
    panel,m=verify_input(input_root,c); dev=panel.loc[panel.candidate_month.eq("2025-03")].copy(); april=panel.loc[panel.candidate_month.eq("2025-04")].copy()
    if len(dev)!=41472 or len(april)!=69258:raise ContractError("frozen March/April row counts changed")
    geoms=tuple(Geometry(**g) for g in c["catboost_hpo"]["geometries"]); all_heads=[]; choice=[]; controls=[]
    # A0 controls have no learned head, but still receive their own causal map.
    for score in c["feature_arms"]["score4"]:
        q=canonical_oof_control(dev,score,c)
        z=april.loc[:,list(PERSISTED_ROW_FIELDS)+[score]].copy().rename(columns={score:"raw_score"});z["score_available_utc"]=z[TIME];z["outer_fold"]="april_frozen";z["candidate_score_is_oof"]=False
        controls.append({"config":"A0__"+score,"architecture":"A0","variant":"individual","features":[score],"march":q,"april":z,"heads":[],"recipe":{}});choice.append({"config":"A0__"+score,"architecture":"A0","variant":"individual","objective":np.nan,"mapping_cohort":"exact_outer_oof_cohort"})
    a1=[]
    for v in c["architectures"]["A1"]["feature_variants"]:
        fields=feature_arm(c,v); o,a,h,r=run_architecture(dev,april,c,"A1",v,geoms,features_override=fields); s=selection_score(o,c); row={"config":"A1__"+v,"architecture":"A1","runner_architecture":"A1","variant":v,"features":fields,"march":o,"april":a,"heads":h,"recipe":r};a1.append(row);choice.append({"config":row["config"],"architecture":"A1","variant":v,**s});all_heads.extend(h)
    best_support=min(a1,key=lambda z:selection_key(z["config"],z["march"],z["heads"],c))
    context_arms=[]
    for name,fields in context_variants(c,best_support["features"]):
        v="context__"+name; o,a,h,r=run_architecture(dev,april,c,"A1",v,geoms,features_override=fields); s=selection_score(o,c); row={"config":"A1_context__"+name,"architecture":"A1_context","runner_architecture":"A1","variant":v,"features":fields,"march":o,"april":a,"heads":h,"recipe":r};a1.append(row);choice.append({"config":row["config"],"architecture":"A1_context","variant":v,**s});all_heads.extend(h)
        context_arms.append(row)
    if not context_arms: raise ContractError("A1_context variants are required for the configured reliability workstream")
    for name,spec in c["feature_arms"].get("candidate_context_aliases",{}).items():
        choice.append({"config":"A1_context__"+name,"architecture":"A1_context","variant":"alias_parity","status":"ALREADY_PRESENT_ALIAS_NOT_DUPLICATED","canonical_field":spec.get("canonical_field"),"alias":spec.get("alias"),"objective":np.nan})
    for name,reason in c["feature_arms"].get("unavailable_candidate_context",{}).items():
        choice.append({"config":"A1_context__"+name,"architecture":"A1_context","variant":"unavailable","status":"UNAVAILABLE_NOT_SUBSTITUTED","reason":reason,"objective":np.nan})
    best_a1=min(context_arms,key=lambda z:selection_key(z["config"],z["march"],z["heads"],c)); best_variant=best_a1["variant"]; best_features=best_a1["features"]
    candidates=[best_a1]
    for architecture in ("A2","A3","A4"):
        o,a,h,r=run_architecture(dev,april,c,architecture,best_variant,geoms,features_override=best_features);s=selection_score(o,c);row={"config":architecture+"__"+best_variant,"architecture":architecture,"runner_architecture":architecture,"variant":best_variant,"features":best_features,"march":o,"april":a,"heads":h,"recipe":r};choice.append({"config":row["config"],"architecture":architecture,"variant":best_variant,**s});all_heads.extend(h);candidates.append(row)
    winner=min(candidates,key=lambda z:selection_key(z["config"],z["march"],z["heads"],c))
    parent=winner["architecture"]; o,a,h,r=run_architecture(dev,april,c,winner["runner_architecture"],winner["variant"],geoms,interactions=True,features_override=winner["features"])
    for row in h: row["architecture"]="A5"
    s=selection_score(o,c);a5_name="A5__"+parent+"__"+winner["variant"];a5={"config":a5_name,"architecture":"A5","runner_architecture":winner["runner_architecture"],"variant":winner["variant"],"features":winner["features"],"march":o,"april":a,"heads":h,"recipe":r};choice.append({"config":a5_name,"architecture":"A5","variant":winner["variant"],"parent":parent,**s});all_heads.extend(h);candidates.append(a5)
    score_parts=[];audit_parts=[];metric_parts=[]; attribution_parts=[]
    evaluated={}
    for item in [*controls,*a1,*candidates]:
        evaluated[item["config"]]=item
    for item in evaluated.values():
        config=item["config"]; march=item["march"]; ap=item["april"]
        combined=pd.concat([march,ap],ignore_index=True);mapped,audit=causal_map(combined);mapped["config"]=config;score_parts.append(mapped);audit["config"]=config;audit_parts.append(audit);metric_parts.extend(metrics(mapped.loc[mapped.candidate_month.eq("2025-03")],config,"march_oof"));metric_parts.extend(metrics(mapped.loc[mapped.candidate_month.eq("2025-04")],config,"april_frozen_diagnostic"))
        attribution_parts.extend(global_book_attribution(mapped.loc[mapped.candidate_month.eq("2025-03")],config,"march_oof",c)); attribution_parts.extend(global_book_attribution(mapped.loc[mapped.candidate_month.eq("2025-04")],config,"april_frozen_diagnostic",c))
    stage=Path(tempfile.mkdtemp(prefix="."+output_dir.name+".",dir=output_dir.parent))
    try:
        outputs={"scores.parquet":pd.concat(score_parts,ignore_index=True),"head_metrics.csv":pd.DataFrame(all_heads),"selection.csv":pd.DataFrame(choice),"mapping_audit.csv":pd.concat(audit_parts,ignore_index=True),"economics.csv":pd.DataFrame(metric_parts),"global_book_attribution.csv":pd.DataFrame(attribution_parts),"freeze_recipes.json":safe({item["config"]:item["recipe"] for item in evaluated.values() if item["recipe"]})}
        for n,x in outputs.items():
            if n.endswith(".parquet"): x.to_parquet(stage/n,index=False,compression="zstd")
            elif n.endswith(".json"): write_json(stage/n,x)
            else: x.to_csv(stage/n,index=False)
        report={"schema":"canonical_execution_reliability_ablation_v2","status":"RESEARCH_ONLY_NO_PROMOTION_NO_PORTFOLIO_REPLAY","promotion_eligible":False,"input":{"artifact":c["input_artifact"],"schema":c["input_schema"],"manifest_sha256":sha(input_root/"manifest.json"),"panel_sha256":sha(input_root/"panel.parquet"),"feature_roles_sha256":sha(input_root/"feature_roles.json"),"rows":len(panel)},"config":{"path":str(config_path),"sha256":sha(config_path)},"evidence_artifacts":evidence,"contract":{"outer_folds":c["outer_folds"],"selection":c["selection"],"mapping":c["mapping"],"A5":"exactly five train-standardized bounded interactions","april":"pre-April full-March inner HPO/PVC freeze; April prediction only"},"outputs_sha256":{n:sha(stage/n) for n in outputs},"runner":{"path":str(Path(__file__).resolve()),"sha256":sha(Path(__file__))}}
        write_json(stage/"manifest.json",report);(stage/"manifest.sha256").write_text(sha(stage/"manifest.json")+"  manifest.json\n");os.replace(stage,output_dir)
    except Exception:shutil.rmtree(stage,ignore_errors=True);raise
    return report

def main(argv:Sequence[str]|None=None)->int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-root",type=Path,default=INPUT)
    p.add_argument("--output-dir",type=Path,default=DEFAULT_OUTPUT)
    p.add_argument("--config",type=Path,default=CONFIG)
    a=p.parse_args(argv)
    print(json.dumps(safe(run(a.input_root,a.output_dir,a.config)),indent=2))
    return 0
if __name__=="__main__":raise SystemExit(main())
