#!/usr/bin/env python3
"""Strict OOF R3 base plus same-side R3 meta-layer ablation.

The base handoff is the direct strict-OOF R3 probability simplex.  The meta
layer receives that same-side simplex unchanged, its entropy/margin, and
causal meta-context fields.  Both layers predict the canonical TP6/SL4 R3
three-state target; no bps conversion or test-period mapping is introduced.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, log_loss

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from extreme_price_movements.config import CFG  # noqa: E402
from extreme_price_movements.transport_supervised_archetypes import configured_available_meta_features, training_univariate_screen  # noqa: E402

SOURCE = ROOT / "data_perp/artifacts/r3_tp6_sl4_meta_target_ablation_20260803_v1/r3_meta_target_oof_predictions.parquet"
OUT = ROOT / "data_perp/artifacts/r3_plus_meta_tp6_sl4_ablation_20260803_v1"
BASE = ("r3_meta_p_adverse", "r3_meta_p_weak", "r3_meta_p_clear", "r3_meta_opportunity_score", "base_r3_entropy", "base_r3_top2_margin")


def _matrix(train: pd.DataFrame, test: pd.DataFrame, fields: list[str]) -> tuple[np.ndarray, np.ndarray]:
    median=train.loc[:,fields].replace([np.inf,-np.inf],np.nan).median().fillna(0.)
    return train.loc[:,fields].replace([np.inf,-np.inf],np.nan).fillna(median).to_numpy(np.float32),test.loc[:,fields].replace([np.inf,-np.inf],np.nan).fillna(median).to_numpy(np.float32)


def _economic(frame: pd.DataFrame, score: str, arm: str, fold: int) -> list[dict[str,object]]:
    ranked=frame.sort_values([score,"candidate_id"],ascending=[False,True],kind="stable");rows=[]
    for top in (.01,.05,.10):
      chosen=ranked.head(max(1,int(np.ceil(len(ranked)*top))))
      for scope,part in (("global",chosen),("long",chosen[chosen.side_name.eq("long")]),("short",chosen[chosen.side_name.eq("short")])):
       if len(part):rows.append({"fold":fold,"arm":arm,"scope":scope,"top_fraction":top,"rows":len(part),"net_bps":float(part.exact_net_bps.mean()),"gross_bps":float(part.exact_gross_bps.mean()),"robust_clear_rate":float(part.r3_class.eq(2).mean()),"adverse_rate":float(part.r3_class.eq(0).mean()),"long_share":float(chosen.side_name.eq("long").mean())})
    return rows


def run() -> None:
    OUT.mkdir(parents=True,exist_ok=True)
    # The frozen base OOF handoff deliberately includes exact identities,
    # labels and the raw causal selector fields used to create it. Re-joining
    # a feature table would duplicate columns and risk an accidental source
    # mismatch; consume the handoff as the single meta input contract.
    frame=pd.read_parquet(SOURCE);frame["__ts__"]=pd.to_datetime(frame["__ts__"],utc=True);frame["label_available_ts"]=pd.to_datetime(frame["label_available_ts"],utc=True)
    available=configured_available_meta_features(CFG,frame.columns.tolist())
    p=frame[["r3_meta_p_adverse","r3_meta_p_weak","r3_meta_p_clear"]].to_numpy(float);frame["base_r3_entropy"]=-(p*np.log(np.maximum(p,1e-12))).sum(axis=1);q=np.sort(p,axis=1);frame["base_r3_top2_margin"]=q[:,-1]-q[:,-2]
    coverage=1-frame.loc[:,available].isna().mean();usable=coverage[coverage.ge(.90)].index.tolist();pd.DataFrame({"feature":available,"coverage":coverage.reindex(available),"usable":pd.Index(available).isin(usable)}).to_parquet(OUT/"meta_feature_coverage.parquet",index=False)
    predictions=[];metrics=[];economics=[]
    for fold in (3,4):
      test=frame.loc[frame.fold.eq(fold)].copy();start=test.__ts__.min();train=frame.loc[frame.fold.lt(fold)&frame.label_available_ts.lt(start)].copy()
      for side in ("long","short"):
        tr=train.loc[train.side_name.eq(side)].copy();te=test.loc[test.side_name.eq(side)].copy();y=tr.r3_class.to_numpy(int)
        context=training_univariate_screen(tr,usable,tr.r3_metric_target.to_numpy(float),maximum=48);fields=[*BASE,*context];xtr,xte=_matrix(tr,te,fields);counts=np.bincount(y,minlength=3).astype(float);w=np.sqrt(len(y)/np.maximum(3*counts[y],1.));w=np.clip(w/w.mean(),.5,2.)
        model=lgb.LGBMClassifier(objective="multiclass",num_class=3,n_estimators=140,learning_rate=.04,num_leaves=20,min_child_samples=350,colsample_bytree=.8,reg_lambda=30.,random_state=20260803+fold,n_jobs=1,verbosity=-1).fit(xtr,y,sample_weight=w)
        pm=np.clip(model.predict_proba(xte),1e-6,1.);pm/=pm.sum(axis=1,keepdims=True);z=te.copy();z[["r3_meta2_p_adverse","r3_meta2_p_weak","r3_meta2_p_clear"]]=pm;z["r3_meta2_opportunity_score"]=pm[:,2]-pm[:,0];predictions.append(z)
        observed=te.r3_class.to_numpy(int);metrics.append({"fold":fold,"side_name":side,"train_rows":len(tr),"test_rows":len(te),"feature_count":len(fields),"selected_context_features":context,"base_log_loss":float(log_loss(observed,te[["r3_meta_p_adverse","r3_meta_p_weak","r3_meta_p_clear"]],labels=[0,1,2])),"meta_log_loss":float(log_loss(observed,pm,labels=[0,1,2])),"base_accuracy":float(accuracy_score(observed,te[["r3_meta_p_adverse","r3_meta_p_weak","r3_meta_p_clear"]].to_numpy().argmax(1))),"meta_accuracy":float(accuracy_score(observed,pm.argmax(1))),"base_r3_ic":float(spearmanr(te.r3_meta_opportunity_score,te.r3_metric_target).statistic),"meta_r3_ic":float(spearmanr(z.r3_meta2_opportunity_score,te.r3_metric_target).statistic)})
    output=pd.concat(predictions,ignore_index=True)
    for fold,part in output.groupby("fold",observed=True):
      economics.extend(_economic(part,"r3_meta_opportunity_score","R3_base",int(fold)));economics.extend(_economic(part,"r3_meta2_opportunity_score","R3_plus_meta",int(fold)))
    output.to_parquet(OUT/"r3_plus_meta_oof_predictions.parquet",index=False);pd.DataFrame(metrics).to_parquet(OUT/"r3_plus_meta_metrics.parquet",index=False);pd.DataFrame(economics).to_parquet(OUT/"r3_plus_meta_economics.parquet",index=False)
    (OUT/"run_manifest.json").write_text(json.dumps({"schema":"r3_plus_meta_tp6_sl4_v1","base":"strict-OOF direct same-side R3 TP6/SL4 simplex from prior ablation","meta_target":"same canonical R3 three-state target","meta_inputs":"direct base simplex + entropy/margin + training-fold-screened causal meta fields","no_conversion":"meta uses raw direct R3 output; final score is P(clear)-P(adverse)","label_availability":"all meta training labels resolve before held-out fold start","evaluation":"fold3 and fold4, global top-k","status":"COMPLETED_DIAGNOSTIC_NO_PROMOTION"},indent=2)+"\n")


if __name__=="__main__":run()
