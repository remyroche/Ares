#!/usr/bin/env python3
"""Base-tail and top-40% specialist gates through the matched residual stack."""
from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.funnel_selection import global_tail_metrics, monthly_stability
from scripts.run_frozen_residual_query_hpo import (
    BASE_FEATURES, CONTRACT, _fit_residual, _fit_specialists, _load, _make_features,
)
from scripts.run_frozen_multiview_specialist_input_ablation import _base, _store_rows, _utc
from scripts.run_market_spine_covariance_meta import LONG_HISTORY_FOLDS

OUT = ROOT / "data_perp/artifacts/frozen_base_tail_top40_gate_20260810_v1"
LABELS = ROOT / "data_perp/artifacts/tp6_sl4_robust_clear_labels_20260802_v1/parts/*.parquet"
LINEAGE = ROOT / "data_perp/artifacts/feature_leaf_reasoning_portability_f0_transport_a_20260803_v4/base_feature_arm_lineage.json"
BASE_PARAMS = dict(n_estimators=140, learning_rate=.05, num_leaves=31, min_child_samples=350, subsample=.8, colsample_bytree=.8, reg_lambda=8., n_jobs=1, verbosity=-1, random_state=20260810)
RESIDUAL_PARAMS = dict(n_estimators=220, learning_rate=.03, max_depth=5, num_leaves=52, min_child_samples=893, min_sum_hessian_in_leaf=1.1298052513600887, min_gain_to_split=.0089300561896448, colsample_bytree=.7882182037573211, subsample=.8666554346312396, subsample_freq=1, reg_alpha=.030925476912139326, reg_lambda=.16986488135579808, max_bin=63, label_gain=[0., .25, 1., 3., 7., 12.], verbosity=-1, random_state=20260810, n_jobs=1)


def _top40(frame: pd.DataFrame) -> pd.Series:
    rank = frame.groupby(["__ts__", "side_name"], sort=False)["base_score"].rank(method="first", ascending=False, pct=True)
    return rank.le(.40)


def _r3_labels() -> pd.DataFrame:
    con = duckdb.connect()
    x = con.execute("SELECT candidate_id, robust_clear_event_b25, lower_touch_minute, label_valid FROM read_parquet(?)", [str(LABELS)]).df()
    con.close()
    x = x.drop_duplicates("candidate_id")
    x["r3_class"] = np.select([x.robust_clear_event_b25.eq(1), x.lower_touch_minute.ge(0)], [2, 0], default=1).astype(np.int8)
    return x[["candidate_id", "r3_class", "label_valid"]]


def _base_features() -> dict[str, list[str]]:
    d = json.loads(LINEAGE.read_text())
    return {str(x["side"]): list(x["features"]) for x in d}


def _fit_base_tail(train: pd.DataFrame, cal: pd.DataFrame, test: pd.DataFrame, fields: list[str], labels: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    tr = train.merge(labels, on="candidate_id", how="inner").loc[lambda x: x.label_valid.eq(True)].copy()
    tr = tr.loc[_top40(tr)].copy()
    cx, tx = cal.merge(_store_rows(cal, fields), on="candidate_id", validate="one_to_one"), test.merge(_store_rows(test, fields), on="candidate_id", validate="one_to_one")
    fx = tr.merge(_store_rows(tr, fields), on="candidate_id", validate="one_to_one")
    med = fx[fields].apply(pd.to_numeric, errors="coerce").median()
    model = lgb.LGBMClassifier(objective="multiclass", num_class=3, **BASE_PARAMS)
    model.fit(fx[fields].apply(pd.to_numeric, errors="coerce").fillna(med).fillna(0.), fx.r3_class.to_numpy(np.int8))
    pcal = model.predict_proba(cx[fields].apply(pd.to_numeric, errors="coerce").fillna(med).fillna(0.))
    ptest = model.predict_proba(tx[fields].apply(pd.to_numeric, errors="coerce").fillna(med).fillna(0.))
    return pcal[:, 2] - pcal[:, 0], ptest[:, 2] - ptest[:, 0]


def _run_fold(base, views, ae, ctx, base_fields, labels, fold) -> list[pd.DataFrame]:
    a, b, c, e = map(_utc, (fold.train_start, fold.calibration_start, fold.test_start, fold.test_end))
    tr = base[base.__ts__.between(a, b, inclusive="left") & base.label_available_ts.lt(b)]
    ca = base[base.__ts__.between(b, c, inclusive="left") & base.label_available_ts.lt(c)]
    te = base[base.__ts__.between(c, e, inclusive="left")]
    out = []
    for side in ("long", "short"):
        train, cal, test = (x[x.side_name.eq(side)].copy() for x in (tr, ca, te))
        train_gate = _top40(train)
        variants = {
            "control": train,
            "base_tail": train,
            "specialists_top40": train.loc[train_gate],
            "base_tail_specialists_top40": train.loc[train_gate],
        }
        for arm, specialist_train in variants.items():
            cal_scores, test_scores = _fit_specialists(specialist_train, cal, test, views[side], "q4h_side")
            calx, fields = _make_features(cal, cal_scores, ae + ctx)
            testx, _ = _make_features(test, test_scores, ae + ctx)
            if "base_tail" in arm:
                pcal, ptest = _fit_base_tail(train, cal, test, base_fields[side], labels)
                calx["base_tail_score"], testx["base_tail_score"] = pcal, ptest
                fields = fields + ["base_tail_score"]
            raw = _fit_residual(calx, testx, fields, "q4h_side", RESIDUAL_PARAMS)
            z = test[["candidate_id", "__ts__", "side_name", "net_bps", "gross_bps"]].copy()
            z["score"] = test.prequential_base_expected_net_bps.to_numpy(float) + raw
            z["fold"], z["arm"] = fold.name, arm
            out.append(z)
            del cal_scores, test_scores, calx, testx
            gc.collect()
    return out


def run(out: Path = OUT) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    base, views, ae, ctx = _load()
    labels = _r3_labels()
    base_fields = _base_features()
    preds = []
    for fold in LONG_HISTORY_FOLDS[3:]:
        preds.extend(_run_fold(base, views, ae, ctx, base_fields, labels, fold))
    p = pd.concat(preds, ignore_index=True)
    p.to_parquet(out / "predictions.parquet", index=False)
    rows = []
    for arm, d in p.groupby("arm", sort=False):
        rows.append({"arm": arm, **global_tail_metrics(d), **monthly_stability(d)})
    pd.DataFrame(rows).to_parquet(out / "metrics.parquet", index=False)
    (out / "manifest.json").write_text(json.dumps({"schema": "frozen_base_tail_top40_gate_v1", "specialist_contract": str(CONTRACT), "base_target": "R3 clear/adverse/weak; clear=robust_clear_event_b25, adverse=lower_touch_first", "base_feature_lineage": str(LINEAGE), "specialist_target": "ATR spacing 2.0", "specialist_query": "4h x side", "residual_query": "4h x side", "arms": ["control", "base_tail", "specialists_top40", "base_tail_specialists_top40"], "selection": "global top5 net, then monthly stability, then top1 net"}, indent=2) + "\n")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--out", type=Path, default=OUT); args = ap.parse_args(); print(run(args.out))
