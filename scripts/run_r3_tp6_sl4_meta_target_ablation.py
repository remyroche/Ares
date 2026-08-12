#!/usr/bin/env python3
"""Try the native base R3 TP6/SL4 target with the causal meta contract.

This is not a replacement for the base model.  It asks whether the available
meta/context features can learn the *same* R3 TP6/SL4 three-state semantics
(adverse / weak / robust-clear) in a strict chronological setting.  It avoids
the noisier exact-net residual target used in the previous archetype run.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, log_loss

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from extreme_price_movements.config import CFG  # noqa: E402
from extreme_price_movements.stage_i_r3_contract import r3_label_economics_contract, selector_validity_mask  # noqa: E402
from extreme_price_movements.transport_supervised_archetypes import configured_available_meta_features, training_univariate_screen  # noqa: E402

LEDGER = ROOT / "data_perp/artifacts/stage_i_selector_sample_20260803_v3/selector_ledger.parquet"
FEATURES = ROOT / "data_perp/artifacts/stage_i_selector_sample_20260803_v3/selector_features.parquet"
OUT = ROOT / "data_perp/artifacts/r3_tp6_sl4_meta_target_ablation_20260803_v1"


def _folds(timestamp: pd.Series) -> pd.Series:
    values = pd.Index(timestamp.drop_duplicates().sort_values())
    mapping = {value: min(4, int(5 * position / max(len(values), 1))) for position, value in enumerate(values)}
    return timestamp.map(mapping).astype(np.int8)


def _matrix(train: pd.DataFrame, test: pd.DataFrame, fields: list[str]) -> tuple[np.ndarray, np.ndarray]:
    median = train.loc[:, fields].replace([np.inf, -np.inf], np.nan).median().fillna(0.)
    return train.loc[:, fields].replace([np.inf, -np.inf], np.nan).fillna(median).to_numpy(np.float32), test.loc[:, fields].replace([np.inf, -np.inf], np.nan).fillna(median).to_numpy(np.float32)


def _economic(frame: pd.DataFrame, score: str, fold: int) -> list[dict[str, object]]:
    ranked = frame.sort_values([score, "candidate_id"], ascending=[False, True], kind="stable")
    records = []
    for fraction in (.01, .05, .10):
        chosen = ranked.head(max(1, int(np.ceil(len(ranked) * fraction))))
        for scope, part in (("global", chosen), ("long", chosen[chosen.side_name.eq("long")]), ("short", chosen[chosen.side_name.eq("short")])):
            if len(part): records.append({"fold":fold,"scope":scope,"top_fraction":fraction,"rows":len(part),"net_bps":float(part.exact_net_bps.mean()),"gross_bps":float(part.exact_gross_bps.mean()),"robust_clear_rate":float(part.r3_class.eq(2).mean()),"adverse_rate":float(part.r3_class.eq(0).mean()),"long_share":float(chosen.side_name.eq("long").mean())})
    return records


def run() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(config={"threads":"2","memory_limit":"512MB","temp_directory":"/tmp"})
    panel_columns = con.execute("DESCRIBE SELECT * FROM read_parquet(?)", [FEATURES.as_posix()]).fetchdf().column_name.tolist()
    available = configured_available_meta_features(CFG, panel_columns)
    selected = ", ".join(f'f."{name}"' for name in available)
    frame = con.execute(f'''SELECT l.*, {selected} FROM read_parquet('{LEDGER.as_posix()}') l JOIN read_parquet('{FEATURES.as_posix()}') f USING(candidate_id)''').fetchdf()
    con.close()
    frame["__ts__"] = pd.to_datetime(frame["__ts__"], utc=True)
    frame["label_available_ts"] = pd.to_datetime(frame["label_available_ts"], utc=True)
    contract = r3_label_economics_contract(frame)
    valid = selector_validity_mask(frame)
    frame = frame.loc[valid].sort_values("__ts__", kind="stable").reset_index(drop=True)
    frame["fold"] = _folds(frame["__ts__"])
    coverage = 1. - frame.loc[:, available].isna().mean()
    usable = coverage[coverage.ge(.90)].index.tolist()
    pd.DataFrame({"feature":available,"coverage":coverage.reindex(available),"usable":pd.Index(available).isin(usable)}).to_parquet(OUT/"r3_meta_feature_coverage.parquet",index=False)
    rows, metrics, economics = [], [], []
    for fold in (2,3,4):
        test = frame.loc[frame.fold.eq(fold)].copy(); start = test.__ts__.min()
        train = frame.loc[frame.label_available_ts.lt(start)].copy()
        scored=[]
        for side in ("long","short"):
            tr=train.loc[train.side_name.eq(side)].copy();te=test.loc[test.side_name.eq(side)].copy()
            y=tr.r3_class.to_numpy(int)
            if set(np.unique(y)) != {0,1,2}: raise ValueError(f"{fold}/{side} training lacks a R3 class")
            context=training_univariate_screen(tr,usable,tr.r3_metric_target.to_numpy(float),maximum=64)
            xtr,xte=_matrix(tr,te,context)
            counts=np.bincount(y,minlength=3).astype(float); weight=np.sqrt(len(y)/np.maximum(3*counts[y],1.));weight=np.clip(weight/weight.mean(),.5,2.)
            model=lgb.LGBMClassifier(objective="multiclass",num_class=3,n_estimators=180,learning_rate=.035,num_leaves=24,min_child_samples=400,colsample_bytree=.8,reg_lambda=30.,random_state=20260803+fold,n_jobs=1,verbosity=-1).fit(xtr,y,sample_weight=weight)
            p=np.clip(model.predict_proba(xte),1e-6,1.);p/=p.sum(axis=1,keepdims=True)
            z=te.copy();z["r3_meta_p_adverse"],z["r3_meta_p_weak"],z["r3_meta_p_clear"]=p[:,0],p[:,1],p[:,2];z["r3_meta_opportunity_score"]=p[:,2]-p[:,0];scored.append(z)
            observed=te.r3_class.to_numpy(int)
            metrics.append({"fold":fold,"side_name":side,"train_rows":len(tr),"test_rows":len(te),"selected_feature_count":len(context),"selected_features":context,"test_log_loss":float(log_loss(observed,p,labels=[0,1,2])),"test_accuracy":float(accuracy_score(observed,p.argmax(1))),"r3_metric_score_ic":float(spearmanr(z.r3_meta_opportunity_score,te.r3_metric_target).statistic)})
        scored_frame=pd.concat(scored,ignore_index=True); rows.append(scored_frame); economics.extend(_economic(scored_frame,"r3_meta_opportunity_score",fold))
    predictions=pd.concat(rows,ignore_index=True)
    predictions.to_parquet(OUT/"r3_meta_target_oof_predictions.parquet",index=False);pd.DataFrame(metrics).to_parquet(OUT/"r3_meta_target_metrics.parquet",index=False);pd.DataFrame(economics).to_parquet(OUT/"r3_meta_target_economics.parquet",index=False)
    (OUT/"run_manifest.json").write_text(json.dumps({"schema":"r3_tp6_sl4_meta_target_ablation_v1","target":"canonical base R3 TP6/SL4 H12: 0=adverse-first, 1=weak/unresolved, 2=cost+25bps robust-clear","soft_metric":"robust_clear_soft_b25_t50 - adverse_indicator","features":"all available configured causal meta features as candidates; training-fold-only 64-field screen","base_feature_contract":"not used; this is a meta-context learnability challenger","label_availability":"strict label_available_ts < test decision start","ranking":"global top-k by P(robust_clear)-P(adverse), never per timestamp","source_contract":contract,"status":"COMPLETED_DIAGNOSTIC_NO_PROMOTION"},indent=2,default=str)+"\n")


if __name__=="__main__": run()
