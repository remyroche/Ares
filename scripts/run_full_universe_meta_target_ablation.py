#!/usr/bin/env python3
"""Alternative per-row economic/reliability targets for the raw-probability meta head."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_full_universe_residual_meta import fam, select  # noqa: E402


def features(frame: pd.DataFrame, chosen: list[str]) -> np.ndarray:
    context = frame[chosen].replace([np.inf, -np.inf], np.nan).fillna(0.).to_numpy("float32")
    base = frame[["p_upper", "p_lower", "p_timeout"]].to_numpy("float32")
    side = frame.side_name.eq("long").to_numpy("float32")[:, None]
    return np.column_stack([context, base, side])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--panel", type=Path, required=True); p.add_argument("--audit", type=Path, required=True); p.add_argument("--base-root", type=Path, required=True); p.add_argument("--out", type=Path, required=True)
    p.add_argument("--geometry", default="tp3_sl2", choices=("tp2_sl1", "tp2_sl2", "tp3_sl1", "tp3_sl2")); p.add_argument("--days", type=int, default=120); p.add_argument("--train-end", default="2024-08-01"); p.add_argument("--eval-end", default="2024-12-01")
    p.add_argument("--target", choices=("direct_net", "residual_net", "class_correct", "cost_clear", "upper_first"), required=True)
    a = p.parse_args()
    gross, net = f"t4_{a.geometry}_gross_bps", f"t4_{a.geometry}_net_bps"
    meta = json.loads(a.audit.read_text())["meta"]["coverage_ge_90pct"]
    columns = ["candidate_id", "__ts__", "__label_available_at__", "side_name", gross, net, f"t2_{a.geometry}_event"] + meta
    raw = pd.concat([pd.read_parquet(x, columns=columns) for x in sorted((a.panel / "parts").glob("*.parquet"))], ignore_index=True)
    raw.__ts__ = pd.to_datetime(raw.__ts__, utc=True); raw.__label_available_at__ = pd.to_datetime(raw.__label_available_at__, utc=True)
    predictions=[]
    for side in ("long", "short"):
        path = a.base_root / side / "target_screen_predictions.parquet"
        if not path.exists(): path = a.base_root / f"t2_{a.geometry}_{side}" / "target_screen_predictions.parquet"
        predictions.append(pd.read_parquet(path, columns=["candidate_id", "score_bps", "p_upper", "p_lower", "p_timeout"]))
    data = raw.merge(pd.concat(predictions, ignore_index=True), on="candidate_id", validate="one_to_one")
    end = pd.Timestamp(a.train_end, tz="UTC"); start = end - pd.Timedelta(days=a.days); eval_end = pd.Timestamp(a.eval_end, tz="UTC")
    train = data[data.__label_available_at__.lt(end) & data.__ts__.ge(start) & data.__ts__.lt(end)].copy()
    evaluation = data[(data.__ts__.ge(end)) & (data.__ts__.lt(eval_end))].copy()
    if a.target == "direct_net": y = train[net].to_numpy(float); classification = False
    elif a.target == "residual_net": y = train[net].to_numpy(float) - train.score_bps.to_numpy(float); classification = False
    elif a.target == "class_correct":
        y = (train[f"t2_{a.geometry}_event"].to_numpy(int) == np.argmax(train[["p_upper", "p_lower", "p_timeout"]].to_numpy(float), axis=1)).astype(int); classification = True
    elif a.target == "cost_clear": y = train[net].gt(0).to_numpy(int); classification = True
    else: y = train[f"t2_{a.geometry}_event"].eq(0).to_numpy(int); classification = True
    chosen = select(train, meta, y, n=36)
    xtr, xev = features(train, chosen), features(evaluation, chosen)
    if classification:
        model = lgb.LGBMClassifier(objective="binary", n_estimators=160, learning_rate=.05, num_leaves=24, min_child_samples=350, colsample_bytree=.8, subsample=.8, reg_lambda=10., random_state=20260803, n_jobs=1, verbosity=-1).fit(xtr, y)
        raw_score = model.predict_proba(xev)[:, 1]
        if a.target == "class_correct": actual = (evaluation[f"t2_{a.geometry}_event"].to_numpy(int) == np.argmax(evaluation[["p_upper", "p_lower", "p_timeout"]].to_numpy(float), axis=1)).astype(int)
        elif a.target == "cost_clear": actual = evaluation[net].gt(0).to_numpy(int)
        else: actual = evaluation[f"t2_{a.geometry}_event"].eq(0).to_numpy(int)
        diagnostics = {"oos_auc": float(roc_auc_score(actual, raw_score)), "oos_brier": float(brier_score_loss(actual, raw_score)), "oos_base_rate": float(actual.mean())}
        final = raw_score
    else:
        model = lgb.LGBMRegressor(objective="huber", alpha=.9, n_estimators=160, learning_rate=.05, num_leaves=24, min_child_samples=350, colsample_bytree=.8, subsample=.8, reg_lambda=10., random_state=20260803, n_jobs=1, verbosity=-1).fit(xtr, y)
        residual = model.predict(xev); residual += float(np.mean(y - model.predict(xtr)))
        final = residual if a.target == "direct_net" else evaluation.score_bps.to_numpy(float) + residual
        diagnostics = {"oos_prediction_std": float(np.std(final))}
    out = evaluation[["candidate_id", "__ts__", "side_name", gross, net]].copy(); out["final_score"] = final
    out = out.sort_values(["final_score", "candidate_id"], ascending=[False, True]); metrics=[]
    for q in (.01, .05, .10, .20):
        z=out.head(int(len(out)*q+.999)); metrics.append({"top_fraction":q,"n":len(z),"gross_bps":float(z[gross].mean()),"net_bps":float(z[net].mean()),"long_n":int(z.side_name.eq("long").sum()),"short_n":int(z.side_name.eq("short").sum())})
    a.out.mkdir(parents=True, exist_ok=True); out.to_parquet(a.out / "meta_target_predictions.parquet", index=False); pd.DataFrame(metrics).to_parquet(a.out / "meta_target_metrics.parquet", index=False)
    manifest={"schema":"full_universe_meta_target_ablation_v1","target":a.target,"target_definition":{"direct_net":"realised barrier-exit net bps","residual_net":"realised barrier-exit net bps minus frozen base score","class_correct":"stored first-touch class equals base argmax probability","cost_clear":"realised barrier-exit net bps > 0","upper_first":"TP/upper barrier is first-touch"}[a.target],"base_inputs":"same-side p_upper, p_lower, p_timeout","meta_context_feature_count":len(chosen),"meta_context_features":chosen,"side_indicator":"side_is_long","train_window":[str(start),str(end)],"evaluation_window":[str(end),str(eval_end)],"global_selection":"pooled long/short/global rank","diagnostics":diagnostics,"metrics":metrics}
    (a.out / "manifest.json").write_text(json.dumps(manifest, indent=2)); print(json.dumps({"target":a.target,"diagnostics":diagnostics,"top10":metrics[2]}))


if __name__ == "__main__": main()
