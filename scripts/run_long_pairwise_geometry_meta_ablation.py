#!/usr/bin/env python3
"""Long-only base/meta ranking ablation with exact geometry labels and AE/GMM.

Evaluation is pooled-global after each score.  ``timestamp x long`` groups are
used only inside LambdaRank.  Path-derived conditions restrict supervised
training support only; they never filter OOF/OOS candidates.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_long_only_executable_net_lambdarank import (
    _fit_bps_map, _fit_ranker, _predict_ranker, _relevance, _tail_metrics, _write_json, _folds,
)

SIDE = "long"


def _fixed_grade(value: np.ndarray, edges: list[float]) -> np.ndarray:
    return np.searchsorted(np.asarray(edges, dtype=float), value, side="right").astype(np.int8)


def _rank_all_grade(value: np.ndarray, group: pd.Series, classes: int = 5) -> np.ndarray:
    out = np.zeros(len(value), dtype=np.int8)
    for _, idx in pd.Series(np.arange(len(value))).groupby(group.to_numpy(), sort=False).groups.items():
        loc = np.asarray(list(idx), dtype=np.int64)
        q = pd.Series(value[loc]).rank(method="average", pct=True).to_numpy()
        out[loc] = np.minimum(classes - 1, np.floor(q * classes)).astype(np.int8)
    return out


def _eligible(frame: pd.DataFrame, mask: np.ndarray) -> pd.DataFrame:
    out = frame.loc[np.asarray(mask, dtype=bool)].copy()
    sizes = out.groupby("query_id", observed=True).size()
    return out.loc[out["query_id"].isin(sizes[sizes >= 2].index)].copy()


def _fit_weighted_ranker(frame: pd.DataFrame, fields: list[str], labels: np.ndarray, weights: np.ndarray, seed: int):
    from lightgbm import LGBMRanker
    order = np.argsort(frame["query_id"].to_numpy(), kind="stable")
    x = frame.iloc[order].loc[:, fields].replace([np.inf, -np.inf], np.nan)
    y = labels[order]; w = weights[order]
    group = frame.iloc[order].groupby("query_id", sort=False, observed=True).size().to_numpy(dtype=np.int32)
    model = LGBMRanker(
        objective="lambdarank", metric="ndcg", label_gain=list(range(6)), n_estimators=400,
        learning_rate=0.04, num_leaves=24, min_child_samples=max(50, int(.015 * len(x))),
        subsample=.8, subsample_freq=1, colsample_bytree=.8, reg_alpha=1.5, reg_lambda=4.,
        lambdarank_truncation_level=10, random_state=seed, n_jobs=4, verbosity=-1,
    )
    model.fit(x, y, group=group, sample_weight=w)
    return model


def _ordering_error_weights(frame: pd.DataFrame, score: str) -> tuple[np.ndarray, np.ndarray]:
    """Row-weighted implementation of the declared wrong-order pair objective.

    LightGBM LambdaRank accepts row weights rather than explicit pair weights;
    the accumulated pair weights are therefore assigned to both endpoints.
    This is recorded as an approximation, not described as native pair-weight
    support.
    """
    n = len(frame); weights = np.full(n, .25, dtype=np.float32)
    value = frame["net_bps"].to_numpy(float); base = frame[score].to_numpy(float)
    for _, positions in pd.Series(np.arange(n)).groupby(frame["query_id"].to_numpy(), sort=False).groups.items():
        pos = np.asarray(list(positions), dtype=np.int64)
        ordered = pos[np.argsort(-base[pos], kind="stable")][:16]
        for ai in range(len(ordered)):
            i = ordered[ai]
            for bj in range(ai + 1, len(ordered)):
                j = ordered[bj]
                gap = value[j] - value[i]
                if gap <= 50.0:
                    continue
                pair_weight = np.log1p(gap / 50.0) * (1.0 + min(abs(base[i] - base[j]) / 100.0, 2.0))
                weights[i] += pair_weight; weights[j] += pair_weight
    weights = np.clip(weights / np.mean(weights), .25, 4.0).astype(np.float32)
    label = _rank_all_grade(value, frame["query_id"], classes=5)
    return label, weights


def _load(args):
    ledger = pd.read_parquet(args.ledger, columns=["candidate_id", "__ts__", "side_name", "gross_bps", "net_bps", "prequential_base_expected_net_bps", "soft_regime_prior_residual_bps", "m6_contract_complete", "shared_regime_contract_complete"])
    ledger = ledger.loc[ledger.side_name.astype(str).str.lower().eq(SIDE)].copy()
    complete = ledger.m6_contract_complete.fillna(False) & ledger.shared_regime_contract_complete.fillna(False)
    ledger = ledger.loc[complete & np.isfinite(ledger.net_bps) & np.isfinite(ledger.gross_bps)].copy()
    ledger["__ts__"] = pd.to_datetime(ledger["__ts__"], utc=True)
    ledger["label_available_ts"] = ledger["__ts__"] + pd.Timedelta(hours=12)
    ledger["current_base_bps"] = ledger.prequential_base_expected_net_bps.fillna(0.).astype("float32")
    ledger["existing_nonranking_bps"] = (ledger.current_base_bps + ledger.soft_regime_prior_residual_bps.fillna(0.)).astype("float32")
    raw = pd.read_parquet(args.raw_features)
    ae = pd.read_parquet(args.aegmm)
    paths = duckdb.sql("SELECT * FROM read_parquet(?)", params=[str(args.path_labels / "parts/*.parquet")]).df()
    paths["__ts__"] = pd.to_datetime(paths["__ts__"], utc=True)
    for item in (raw, ae, paths):
        if item.candidate_id.duplicated().any(): raise ValueError("input sidecar has duplicate candidates")
    fields = [c for c in raw.columns if c != "candidate_id"] + [c for c in ae.columns if c != "candidate_id"]
    work = ledger.merge(raw, on="candidate_id", how="left", validate="one_to_one").merge(ae, on="candidate_id", how="left", validate="one_to_one").merge(paths.drop(columns=["__ts__", "side_name", "__decision_ts__", "__symbol__"]), on="candidate_id", how="left", validate="one_to_one")
    coverage = work.loc[:, fields].notna().mean()
    bad = coverage.loc[coverage < .90].index.tolist()
    if bad:
        raise ValueError(f"raw or AE/GMM fields below 90% full-universe coverage: {bad}")
    if work.label_valid.isna().any(): raise ValueError("path label sidecar does not cover the full long universe")
    if "support_h12_mfe_mae" not in work.columns:
        # Legacy v1 sidecars use this exact same conjunctive condition under
        # the MFE/MAE validity name.  The new explicit column is emitted on
        # future materialisations; accepting the alias avoids a needless
        # path replay solely to rename an already-correct condition.
        work["support_h12_mfe_mae"] = work["mfe_mae_label_valid"].fillna(False)
    if not work.loc[work.support_h12_mfe_mae.fillna(False), "mfe_mae_label_valid"].all():
        raise ValueError("H12 support must be a subset of valid MFE/MAE labels")
    work["query_id"] = work["__ts__"].astype(str) + "|long"; work["month"] = work["__ts__"].dt.strftime("%Y-%m")
    return work, fields


def run(args):
    out_dir = args.output_dir; out_dir.mkdir(parents=True, exist_ok=True)
    work, fields = _load(args)
    prediction_parts=[]; fold_log=[]
    base_names=("A0_current_base", "A1_net_rank_aegmm", "A3_barrier_rank_aegmm", "A4_mfe_mae_rank_aegmm")
    for fold_no,(fold,start,end) in enumerate(_folds(),1):
        start_ts,end_ts=pd.Timestamp(start,tz="UTC"),pd.Timestamp(end,tz="UTC")
        test=work.loc[(work.__ts__>=start_ts)&(work.__ts__<end_ts)].copy()
        history=work.loc[work.label_available_ts<start_ts].copy()
        cut_a,cut_b=history.__ts__.quantile([.60,.80]).to_list()
        base_train=history.loc[history.__ts__<=cut_a].copy(); meta_train=history.loc[(history.__ts__>cut_a)&(history.__ts__<=cut_b)].copy(); meta_cal=history.loc[history.__ts__>cut_b].copy()
        base_scores={"A0_current_base": (meta_train.current_base_bps.to_numpy(),meta_cal.current_base_bps.to_numpy(),test.current_base_bps.to_numpy())}
        definitions={
            "A1_net_rank_aegmm": (base_train, _relevance(base_train.net_bps.to_numpy(),base_train.query_id,margin=35.,classes=6)),
            "A3_barrier_rank_aegmm": (_eligible(base_train,base_train.label_valid.to_numpy()), None),
            "A4_mfe_mae_rank_aegmm": (_eligible(base_train,base_train.mfe_mae_label_valid.to_numpy()), None),
        }
        definitions["A3_barrier_rank_aegmm"]=(definitions["A3_barrier_rank_aegmm"][0],definitions["A3_barrier_rank_aegmm"][0].barrier_relevance_0_5.to_numpy(np.int8))
        definitions["A4_mfe_mae_rank_aegmm"]=(definitions["A4_mfe_mae_rank_aegmm"][0],definitions["A4_mfe_mae_rank_aegmm"][0].mfe_mae_relevance_0_5.to_numpy(np.int8))
        for base_name,(train,label) in definitions.items():
            model,audit=_fit_ranker(train,fields,label,seed=20261000+fold_no)
            raw_train=_predict_ranker(model,meta_train,fields); raw_cal=_predict_ranker(model,meta_cal,fields); raw_test=_predict_ranker(model,test,fields)
            mapper=_fit_bps_map(raw_cal,meta_cal.net_bps.to_numpy())
            base_scores[base_name]=(mapper.predict(raw_train),mapper.predict(raw_cal),mapper.predict(raw_test))
            fold_log.append({"fold":fold,"arm":base_name,"stage":"base","train_rows":len(train),"raw_feature_count":len(fields),**audit})
        result=test.loc[:,["candidate_id","__ts__","month","gross_bps","net_bps"]].copy(); result["fold"]=fold
        result["A0_current_base"] = base_scores["A0_current_base"][2]; result["A0_current_nonranking_residual"] = test.existing_nonranking_bps.to_numpy()
        for base_name,(_,_,score) in base_scores.items(): result[base_name]=score
        for base_name,(mt_score,mc_score,te_score) in base_scores.items():
            anchor="__base_anchor__"; meta_train[anchor]=mt_score; meta_cal[anchor]=mc_score; test[anchor]=te_score
            meta_fields=[*fields,anchor]
            residual=meta_train.net_bps.to_numpy()-mt_score
            targets={
              "B1_current_rank_residual": (meta_train, _relevance(residual,meta_train.query_id,margin=50.,classes=5),np.ones(len(meta_train),np.float32)),
              "B2_realised_net_residual": (meta_train, _fixed_grade(residual,[-150.,-50.,50.,150.]),np.ones(len(meta_train),np.float32)),
              "B3_ordering_error": (meta_train,None,None),
              "B4_atr_residual": (_eligible(meta_train,meta_train.support_h12_mfe_mae.fillna(False).to_numpy()),None,None),
            }
            oe_label,oe_weight=_ordering_error_weights(meta_train,anchor); targets["B3_ordering_error"]=(meta_train,oe_label,oe_weight)
            atr_frame=targets["B4_atr_residual"][0]; atr_res=(atr_frame.net_bps.to_numpy()-atr_frame[anchor].to_numpy())/atr_frame.atr_bps.to_numpy(); targets["B4_atr_residual"]=(atr_frame,_fixed_grade(atr_res,[-1.5,-.5,.75,2.]),np.ones(len(atr_frame),np.float32))
            for meta_name,(train,label,weight) in targets.items():
                if meta_name=="B1_current_rank_residual": model,_=_fit_ranker(train,meta_fields,label,seed=20262000+fold_no)
                else: model=_fit_weighted_ranker(train,meta_fields,label,weight,seed=20262000+fold_no)
                cal_raw=_predict_ranker(model,meta_cal,meta_fields); test_raw=_predict_ranker(model,test,meta_fields)
                mapper=_fit_bps_map(cal_raw,meta_cal.net_bps.to_numpy()-meta_cal[anchor].to_numpy())
                result[f"{base_name}__{meta_name}"]=te_score+mapper.predict(test_raw)
                fold_log.append({"fold":fold,"arm":f"{base_name}__{meta_name}","stage":"meta","train_rows":len(train),"raw_feature_count":len(meta_fields),"weighted":meta_name=="B3_ordering_error"})
        prediction_parts.append(result)
    pred=pd.concat(prediction_parts,ignore_index=True); pred.to_parquet(out_dir/"oof_oos_predictions.parquet",index=False,compression="zstd")
    pd.DataFrame(fold_log).to_parquet(out_dir/"arm_fold_audit.parquet",index=False)
    metrics=[]
    for score in [c for c in pred if c.startswith("A")]: metrics.extend(_tail_metrics(pred,score))
    pd.DataFrame(metrics).to_parquet(out_dir/"ablation_metrics.parquet",index=False)
    _write_json(out_dir/"run_manifest.json",{"schema":"long_pairwise_geometry_meta_ablation_v1","status":"complete","side":SIDE,"raw_mda_features":92,"aegmm_features":len(fields)-92,"feature_scope":"MDA raw + frozen AE/GMM only; no other regime representations","base_arms":list(base_names),"meta_arms":["B1_current_rank_residual","B2_realised_net_residual","B3_ordering_error_row_weighted_pair_approximation","B4_atr_residual"],"evaluation":"pooled-global top-k after score; no outcome condition at evaluation"})


def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir",type=Path,required=True)
    p.add_argument("--ledger",type=Path,default=ROOT/"data_perp/artifacts/tp6_m6_shared_regime_residual_features_20260809_v3/shared_regime_residual_ledger.parquet")
    p.add_argument("--raw-features",type=Path,default=ROOT/"data_perp/artifacts/long_only_executable_net_lambdarank_20260804_v1/long_mda92_features.parquet")
    p.add_argument("--aegmm",type=Path,default=ROOT/"data_perp/artifacts/long_pairwise_aegmm_20260804_v1.parquet")
    p.add_argument("--path-labels",type=Path,default=ROOT/"data_perp/artifacts/long_pairwise_path_labels_20260804_v1")
    return p.parse_args()


if __name__=="__main__": run(parse_args())
