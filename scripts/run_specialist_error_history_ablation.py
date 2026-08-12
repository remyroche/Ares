#!/usr/bin/env python3
"""Frozen residual ablation with causal specialist error-history features.

The history features are computed strictly from labels available before each
row.  Calibration rows use prior calibration outcomes; test rows use the
calibration history only (test outcomes are never fed back into features).
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.funnel_selection import global_tail_metrics, monthly_stability
from scripts.run_frozen_residual_query_hpo import (
    _fit_residual,
    _fit_specialists,
    _load,
    _make_features,
    _utc,
)
from scripts.run_market_spine_covariance_meta import LONG_HISTORY_FOLDS

OUT = ROOT / "data_perp/artifacts/specialist_error_history_ablation_20260810_v1"
PARAMS = {
    "n_estimators": 220,
    "learning_rate": 0.03,
    "max_depth": 4,
    "num_leaves": 31,
    "min_child_samples": 893,
    "min_sum_hessian_in_leaf": 1.13,
    "min_gain_to_split": 0.00893,
    "colsample_bytree": 0.79,
    "subsample": 0.87,
    "subsample_freq": 1,
    "reg_alpha": 0.03,
    "reg_lambda": 0.17,
    "max_bin": 63,
    "label_gain": [0.0, 0.25, 1.0, 3.0, 7.0, 12.0],
    "verbosity": -1,
    "random_state": 20260810,
    "n_jobs": 1,
}
BASE_OUTPUTS = ["p_clear", "p_adverse", "p_weak", "base_score", "prequential_base_expected_net_bps"]
LOOKBACKS = (3, 7, 14)


def _side_month_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for arm, x in frame.groupby("arm", sort=False):
        for side, y in x.groupby("side_name", sort=False):
            rows.append({"arm": arm, "scope": "side", "side": side, **global_tail_metrics(y), **monthly_stability(y)})
        for month, y in x.assign(_month=pd.to_datetime(x["__ts__"], utc=True).dt.strftime("%Y-%m")).groupby("_month", sort=True):
            rows.append({"arm": arm, "scope": "month", "month": month, **global_tail_metrics(y), **monthly_stability(y)})
    return pd.DataFrame(rows)


def _rolling_features(frame: pd.DataFrame, score_cols: list[str], known_labels: pd.Series) -> pd.DataFrame:
    """Return prior-only hit-rate, surprise and IC fields for each specialist."""
    x = frame[["candidate_id", "__ts__", "side_name", *score_cols]].copy()
    x["__order__"] = np.arange(len(x))
    x = x.sort_values(["side_name", "__ts__", "candidate_id"], kind="stable").reset_index(drop=True)
    ts = pd.to_datetime(x["__ts__"], utc=True)
    out = x[["candidate_id"]].copy()
    # Rank within the decision timestamp is causal and removes arbitrary head scale.
    for col in score_cols:
        x[f"__prob__{col}"] = x.groupby(["side_name", "__ts__"], sort=False)[col].rank(pct=True, method="average").astype(float)
    # The caller passes labels in the original order. Reindex by the retained
    # original row positions after the stable sort.
    y = pd.to_numeric(known_labels.iloc[x["__order__"].to_numpy()].to_numpy(), errors="coerce")
    y_series = pd.Series(y, index=ts)
    for col in score_cols:
        prob = pd.Series(x[f"__prob__{col}"].to_numpy(float), index=ts)
        for days in LOOKBACKS:
            window = f"{days}D"
            prior_n = y_series.notna().astype(float).rolling(window, closed="left", min_periods=1).sum()
            hit = y_series.rolling(window, closed="left", min_periods=1).mean()
            surprise = (y_series - prob).rolling(window, closed="left", min_periods=1).mean()
            xx = prob
            yy = y_series
            xy = (xx * yy).rolling(window, closed="left", min_periods=1).sum()
            sx = xx.rolling(window, closed="left", min_periods=1).sum()
            sy = yy.rolling(window, closed="left", min_periods=1).sum()
            sxx = (xx * xx).rolling(window, closed="left", min_periods=1).sum()
            syy = (yy * yy).rolling(window, closed="left", min_periods=1).sum()
            cov = xy - sx * sy / prior_n.replace(0, np.nan)
            vx = sxx - sx * sx / prior_n.replace(0, np.nan)
            vy = syy - sy * sy / prior_n.replace(0, np.nan)
            ic = cov / np.sqrt((vx * vy).clip(lower=1e-12))
            prefix = f"err__{col.removeprefix('mv__')}__{days}d__"
            vals = pd.DataFrame({
                "candidate_id": x["candidate_id"].to_numpy(),
                prefix + "hit_rate": hit.to_numpy(float),
                prefix + "hit_surprise": surprise.to_numpy(float),
                prefix + "ic": ic.to_numpy(float),
            })
            out = out.merge(vals, on="candidate_id", validate="one_to_one")
    return out


def run(out: Path = OUT) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    base, views, ae, ctx = _load()
    predictions: list[pd.DataFrame] = []
    for fold in LONG_HISTORY_FOLDS[3:]:
        a, b, c, e = map(_utc, (fold.train_start, fold.calibration_start, fold.test_start, fold.test_end))
        tr = base[base.__ts__.between(a, b, inclusive="left") & base.label_available_ts.lt(b)]
        cal = base[base.__ts__.between(b, c, inclusive="left") & base.label_available_ts.lt(c)]
        test = base[base.__ts__.between(c, e, inclusive="left")]
        for side in ("long", "short"):
            train = tr[tr.side_name.eq(side)].copy()
            ca = cal[cal.side_name.eq(side)].copy()
            te = test[test.side_name.eq(side)].copy()
            cal_scores, test_scores = _fit_specialists(train, ca, te, views[side], "q4h_side")
            calx, fields = _make_features(ca, cal_scores, ae + ctx)
            testx, _ = _make_features(te, test_scores, ae + ctx)
            score_cols = [c for c in cal_scores.columns if c.startswith("mv__")]
            # Calibration labels are known; test labels are deliberately hidden.
            cal_known = (ca.net_bps > 50.0).astype(float).reset_index(drop=True)
            test_unknown = pd.Series(np.nan, index=np.arange(len(te)), dtype=float)
            cal_hist_frame = ca[["candidate_id", "__ts__", "side_name"]].merge(cal_scores, on="candidate_id", validate="one_to_one")
            test_hist_frame = te[["candidate_id", "__ts__", "side_name"]].merge(test_scores, on="candidate_id", validate="one_to_one")
            all_frame = pd.concat([cal_hist_frame[["candidate_id", "__ts__", "side_name", *score_cols]], test_hist_frame[["candidate_id", "__ts__", "side_name", *score_cols]]], ignore_index=True)
            known = pd.concat([cal_known, test_unknown], ignore_index=True)
            hist = _rolling_features(all_frame, score_cols, known)
            calx = calx.merge(hist[hist.candidate_id.isin(ca.candidate_id)], on="candidate_id", validate="one_to_one")
            testx = testx.merge(hist[hist.candidate_id.isin(te.candidate_id)], on="candidate_id", validate="one_to_one")
            history_by_arm = {
                "no_error": [],
                "lookback_3d": [c for c in hist.columns if "__3d__" in c],
                "lookback_7d": [c for c in hist.columns if "__7d__" in c],
                "lookback_14d": [c for c in hist.columns if "__14d__" in c],
                "all_3_7_14d": [c for c in hist.columns if c.startswith("err__")],
            }
            for arm, extra in history_by_arm.items():
                use = list(dict.fromkeys([f for f in fields if f in calx.columns and f in testx.columns] + extra))
                use = [f for f in use if f in calx.columns and f in testx.columns]
                raw = _fit_residual(calx, testx, use, "q4h_side", PARAMS)
                z = te[["candidate_id", "__ts__", "side_name", "net_bps", "gross_bps"]].copy()
                z["score"] = te.prequential_base_expected_net_bps.to_numpy(float) + np.clip(raw, -50.0, 50.0)
                z["fold"], z["arm"] = fold.name, arm
                predictions.append(z)
                del raw, z, use
                gc.collect()
            del calx, testx, hist, all_frame
            gc.collect()
    pred = pd.concat(predictions, ignore_index=True)
    rows = [{"arm": arm, **global_tail_metrics(x), **monthly_stability(x)} for arm, x in pred.groupby("arm", sort=False)]
    metrics = pd.DataFrame(rows)
    metrics.to_parquet(out / "metrics.parquet", index=False)
    _side_month_metrics(pred).to_parquet(out / "side_month_metrics.parquet", index=False)
    pred.to_parquet(out / "predictions.parquet", index=False)
    (out / "manifest.json").write_text(json.dumps({
        "schema": "specialist_error_history_ablation_v1",
        "specialist_contract": "frozen 7 views x 68 fields",
        "specialist_target": "ATR spacing 2.0",
        "query": "q4h_side",
        "history_features": ["hit_rate", "hit_surprise", "IC"],
        "lookbacks_days": list(LOOKBACKS),
        "causality": "calibration labels only; test labels hidden; closed-left rolling windows",
        "max_depth": 4,
        "selection": "global top5 net, then monthly stability, then top1 net",
    }, indent=2) + "\n")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    print(run(args.out))
