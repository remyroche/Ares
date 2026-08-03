#!/usr/bin/env python3
"""D1 causal-student / future-path-teacher distillation ablation for T2."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_full_universe_t2_t4_target_screen import _read, _soft, _subset  # noqa: E402


def matrix(frame: pd.DataFrame, cols: list[str]) -> np.ndarray:
    return frame[cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy("float32")


def model() -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(objective="huber", alpha=.90, n_estimators=120, learning_rate=.06, num_leaves=24, min_child_samples=400, colsample_bytree=.8, subsample=.8, reg_lambda=8., random_state=20260802, n_jobs=1, verbosity=-1)


def simplex(raw: np.ndarray) -> np.ndarray:
    raw = np.maximum(raw, 1e-7)
    return raw / raw.sum(axis=1, keepdims=True)


def fit_side(train: pd.DataFrame, evaluation: pd.DataFrame, base: list[str], geometry: str, tau: float, alpha: float, temperature: float):
    target = _soft(train, geometry, tau)
    proxy = train[f"t4_{geometry}_net_bps"].to_numpy(float)
    selected = _subset(train, base, proxy, 36)
    # These columns are strictly future-path supervision.  They are used only
    # while cross-fitting teacher targets within the outer training fold.
    teacher_cols = ["t2_path_mfe_atr", "t2_path_mae_atr", f"t2_{geometry}_event", f"t2_{geometry}_exit_minute"]
    tx = matrix(train, teacher_cols)
    oof = np.zeros_like(target)
    for fit_idx, hold_idx in KFold(n_splits=2, shuffle=True, random_state=20260802).split(tx):
        for j in range(3):
            oof[hold_idx, j] = model().fit(tx[fit_idx], target[fit_idx, j]).predict(tx[hold_idx])
    teacher = simplex(oof)
    if temperature != 1.0:
        teacher = simplex(np.power(teacher, 1.0 / temperature))
    student_target = alpha * target + (1.0 - alpha) * teacher
    xtr, xev = matrix(train, selected), matrix(evaluation, selected)
    pred = np.column_stack([model().fit(xtr, student_target[:, j]).predict(xev) for j in range(3)])
    probability = simplex(pred)
    net = train[f"t4_{geometry}_net_bps"].to_numpy(float)
    means = (target * net[:, None]).sum(axis=0) / np.maximum(target.sum(axis=0), 1.0)
    return probability @ means, probability, {"student_features": selected, "teacher_future_only_features": teacher_cols, "alpha": alpha, "temperature": temperature, "teacher_oof_rows": len(train), "conditional_net_means_bps": means.tolist()}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--panel", type=Path, required=True); p.add_argument("--audit", type=Path, required=True); p.add_argument("--out", type=Path, required=True)
    p.add_argument("--train-end", default="2024-08-01"); p.add_argument("--eval-end", default="2024-12-01"); p.add_argument("--geometry", default="tp3_sl2"); p.add_argument("--tau", type=float, default=.25); p.add_argument("--alpha", type=float, default=.50); p.add_argument("--temperature", type=float, default=2.0)
    a = p.parse_args()
    if not 0.0 < a.alpha <= 1.0 or a.temperature < 1.0: p.error("alpha must be in (0,1], temperature >= 1")
    base = json.loads(a.audit.read_text())["base"]["coverage_ge_90pct"]
    cols = ["candidate_id", "__ts__", "side_name", "t2_path_mfe_atr", "t2_path_mae_atr"]
    for g in ("tp2_sl1", "tp2_sl2", "tp3_sl1", "tp3_sl2"): cols += [f"t2_{g}_event", f"t2_{g}_exit_minute", f"t4_{g}_gross_bps", f"t4_{g}_net_bps"]
    data = _read(a.panel, list(dict.fromkeys(cols + base))); data.__ts__ = pd.to_datetime(data.__ts__, utc=True)
    train = data[data.__ts__ < pd.Timestamp(a.train_end, tz="UTC")]; evaluation = data[(data.__ts__ >= pd.Timestamp(a.train_end, tz="UTC")) & (data.__ts__ < pd.Timestamp(a.eval_end, tz="UTC"))]
    results=[]; contracts={}
    for side in ("long", "short"):
        score, probability, contract = fit_side(train[train.side_name.eq(side)].copy(), evaluation[evaluation.side_name.eq(side)].copy(), base, a.geometry, a.tau, a.alpha, a.temperature)
        z=evaluation[evaluation.side_name.eq(side)][["candidate_id","__ts__","side_name",f"t4_{a.geometry}_gross_bps",f"t4_{a.geometry}_net_bps"]].copy();z.columns=["candidate_id","__ts__","side_name","gross_bps","net_bps"];z["score_bps"]=score;z[["p_upper","p_lower","p_timeout"]]=probability;results.append(z);contracts[side]=contract
    output=pd.concat(results,ignore_index=True).sort_values(["score_bps","candidate_id"],ascending=[False,True]); metrics=[]
    for q in (.01,.05,.1,.2):
        z=output.head(int(len(output)*q+.999));metrics.append({"top_fraction":q,"n":len(z),"gross_bps":float(z.gross_bps.mean()),"net_bps":float(z.net_bps.mean()),"long_n":int(z.side_name.eq("long").sum()),"short_n":int(z.side_name.eq("short").sum())})
    a.out.mkdir(parents=True,exist_ok=True);output.to_parquet(a.out/"target_screen_predictions.parquet",index=False);pd.DataFrame(metrics).to_parquet(a.out/"target_screen_metrics.parquet",index=False)
    (a.out/"target_family_manifest.json").write_text(json.dumps({"schema":"full_universe_t2_distillation_v1","placement":"D1 base only","train_window":[str(train.__ts__.min()),a.train_end],"evaluation_window":[a.train_end,a.eval_end],"teacher_future_features_never_in_inference":True,"geometry":a.geometry,"tau":a.tau,"alpha":a.alpha,"temperature":a.temperature,"feature_contract":contracts,"metrics":metrics},indent=2));print(pd.DataFrame(metrics).to_string(index=False))

if __name__ == "__main__": main()
