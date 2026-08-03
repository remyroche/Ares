#!/usr/bin/env python3
"""Causal GAM-plus-nonlinear residual base ablation for full-universe T2.

The additive spline model learns smooth broad opportunity structure.  A
side-local nonlinear learner then predicts residual log probabilities; neither
stage sees realised-path fields as inputs.  The result remains three base
probabilities suitable for the stopped-gradient economic meta layer.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_full_universe_t2_t4_target_screen import _read, _soft, _subset  # noqa: E402


def _x(frame: pd.DataFrame, cols: list[str]) -> np.ndarray:
    return frame[cols].replace([np.inf, -np.inf], np.nan).to_numpy("float32")


def _fit_side(train: pd.DataFrame, evaluation: pd.DataFrame, cols: list[str], geometry: str, tau: float, mode: str) -> tuple[np.ndarray, np.ndarray, list[str], dict]:
    # The spline component intentionally uses a compact, diverse prefix of
    # the training-only rank-selected base contract.  It is an additive
    # structural baseline, not a second high-capacity nonlinear model.
    proxy = train[f"t4_{geometry}_net_bps"].to_numpy(float)
    selected = _subset(train, cols, proxy, 36)
    gam_cols = selected[:12]
    y = _soft(train, geometry, tau)
    log_y = np.log(np.clip(y, 1e-6, 1.0))
    xt_gam, xe_gam = _x(train, gam_cols), _x(evaluation, gam_cols)
    gam = make_pipeline(SimpleImputer(strategy="median"), SplineTransformer(n_knots=5, degree=2, extrapolation="linear"), Ridge(alpha=20.0))
    gam.fit(xt_gam, log_y)
    base_train = gam.predict(xt_gam)
    base_eval = gam.predict(xe_gam)
    residual = log_y - base_train
    xt, xe = _x(train, selected), _x(evaluation, selected)
    nonlinear_pred = []
    if mode == "augmentation":
        # G1: ordinary nonlinear target fit, with additive baseline
        # probabilities as purely causal extra inputs.
        xt = np.column_stack([np.nan_to_num(xt, nan=0.0, posinf=0.0, neginf=0.0), softmax(base_train, axis=1)])
        xe = np.column_stack([np.nan_to_num(xe, nan=0.0, posinf=0.0, neginf=0.0), softmax(base_eval, axis=1)])
    for j in range(3):
        model = lgb.LGBMRegressor(objective="huber", alpha=.90, n_estimators=200, learning_rate=.05, num_leaves=24, min_child_samples=400, colsample_bytree=.8, subsample=.8, reg_lambda=8., random_state=20260802, n_jobs=1, verbosity=-1)
        model.fit(np.nan_to_num(xt, nan=0.0, posinf=0.0, neginf=0.0), y[:, j] if mode == "augmentation" else residual[:, j])
        nonlinear_pred.append(model.predict(np.nan_to_num(xe, nan=0.0, posinf=0.0, neginf=0.0)))
    if mode == "augmentation":
        raw = np.maximum(np.column_stack(nonlinear_pred), 0.0)
        prob = raw / np.maximum(raw.sum(axis=1, keepdims=True), 1e-8)
    else:
        prob = softmax(base_eval + np.column_stack(nonlinear_pred), axis=1)
    net = train[f"t4_{geometry}_net_bps"].to_numpy(float)
    means = (y * net[:, None]).sum(axis=0) / np.maximum(y.sum(axis=0), 1.0)
    score = prob @ means
    return score, prob, selected, {"gam_features": gam_cols, "conditional_net_means_bps": means.tolist()}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--panel", type=Path, required=True)
    p.add_argument("--audit", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--train-end", default="2024-08-01")
    p.add_argument("--eval-end", default="2024-12-01")
    p.add_argument("--geometry", default="tp3_sl2", choices=("tp2_sl1", "tp2_sl2", "tp3_sl1", "tp3_sl2"))
    p.add_argument("--tau", type=float, default=.25)
    p.add_argument("--mode", choices=("residual", "augmentation"), default="residual")
    args = p.parse_args()
    audit = json.loads(args.audit.read_text())
    base = audit["base"]["coverage_ge_90pct"]
    labels = ["candidate_id", "__ts__", "side_name", "t2_path_mfe_atr", "t2_path_mae_atr"]
    for geometry in ("tp2_sl1", "tp2_sl2", "tp3_sl1", "tp3_sl2"):
        labels += [f"t2_{geometry}_event", f"t2_{geometry}_exit_minute", f"t4_{geometry}_gross_bps", f"t4_{geometry}_net_bps"]
    data = _read(args.panel, list(dict.fromkeys(labels + base)))
    data["__ts__"] = pd.to_datetime(data["__ts__"], utc=True)
    train = data[data.__ts__.lt(pd.Timestamp(args.train_end, tz="UTC"))]
    evaluation = data[(data.__ts__ >= pd.Timestamp(args.train_end, tz="UTC")) & (data.__ts__ < pd.Timestamp(args.eval_end, tz="UTC"))]
    all_predictions = []
    contract: dict[str, object] = {}
    for side in ("long", "short"):
        tr = train[train.side_name.eq(side)].copy()
        ev = evaluation[evaluation.side_name.eq(side)].copy()
        score, probabilities, selected, details = _fit_side(tr, ev, base, args.geometry, args.tau, args.mode)
        out = ev[["candidate_id", "__ts__", "side_name", f"t4_{args.geometry}_gross_bps", f"t4_{args.geometry}_net_bps"]].copy()
        out.columns = ["candidate_id", "__ts__", "side_name", "gross_bps", "net_bps"]
        out["score_bps"] = score
        out[["p_upper", "p_lower", "p_timeout"]] = probabilities
        all_predictions.append(out)
        contract[side] = {"nonlinear_features": selected, **details}
    pred = pd.concat(all_predictions, ignore_index=True).sort_values(["score_bps", "candidate_id"], ascending=[False, True])
    metrics = []
    for fraction in (.01, .05, .10, .20):
        chosen = pred.head(int(len(pred) * fraction + .999))
        metrics.append({"top_fraction": fraction, "n": len(chosen), "gross_bps": float(chosen.gross_bps.mean()), "net_bps": float(chosen.net_bps.mean()), "long_n": int(chosen.side_name.eq("long").sum()), "short_n": int(chosen.side_name.eq("short").sum())})
    args.out.mkdir(parents=True, exist_ok=True)
    pred.to_parquet(args.out / "target_screen_predictions.parquet", index=False)
    pd.DataFrame(metrics).to_parquet(args.out / "target_screen_metrics.parquet", index=False)
    architecture = "side-local additive spline log-probability baseline plus nonlinear residual logits" if args.mode == "residual" else "side-local nonlinear T2 base augmented by causal additive spline probabilities"
    (args.out / "target_family_manifest.json").write_text(json.dumps({"schema": "full_universe_t2_gam_residual_v1", "train_window": [str(train.__ts__.min()), args.train_end], "evaluation_window": [args.train_end, args.eval_end], "geometry": args.geometry, "tau": args.tau, "base_architecture": architecture, "entry": "next hourly open", "exit": "first TP/SL then H12 timeout", "global_selection": "pooled across sides and timestamps after common-bps mapping", "feature_contract": contract, "metrics": metrics}, indent=2))
    print(pd.DataFrame(metrics).to_string(index=False))


if __name__ == "__main__":
    main()
