#!/usr/bin/env python3
"""Bounded strict-OOS model/loss HPO for the 15m stateful continuation head."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.p8u_15m_features import FIFTEEN_MINUTE_FEATURE_KEYS
from extreme_price_movements.p8u_continuation_state import CONTINUATION_STATE_FEATURE_KEYS
from scripts import run_strict_r3_p8u_15m_continuation_c1_ablation as c1


DEFAULT_OUTPUT = ROOT / "data_perp/artifacts/strict_r3_p8u_15m_continuation_hpo_20260830_v1"
SPECS = c1.MODEL_SPECS
FLOORS = (20.0, 30.0, 40.0, 50.0)
FEATURES = (*FIFTEEN_MINUTE_FEATURE_KEYS, *CONTINUATION_STATE_FEATURE_KEYS)


def _predict_vector(model, spec: str, x: pd.DataFrame) -> np.ndarray:
    raw = np.asarray(model.booster_.predict(x) if hasattr(model, "booster_") else model.predict(x), dtype=float)
    if spec == "lgb_l1_bps":
        raw = np.interp(raw, [-1e9, -100., -25., 25., 100., 1e9], [0., 0., 1., 2., 3., 4.])
    return np.clip(raw, 0., 4.)


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--output",type=Path,default=DEFAULT_OUTPUT); args=parser.parse_args()
    out=args.output.resolve(); out.mkdir(parents=True,exist_ok=False)
    panel=c1._load_panel(c1.STATE_ROOT,c1.PARITY_ROOT)
    rows=[]; metrics=[]
    months=pd.date_range("2026-04-01","2026-08-01",freq="MS",tz="UTC")
    for floor in FLOORS:
        scoped=panel.loc[pd.to_numeric(panel.MC1_expected_bps,errors="coerce").ge(floor)].copy()
        for held in months:
            start=held-pd.DateOffset(months=2); end=held+pd.offsets.MonthBegin(1)
            train=scoped.loc[scoped.entry_decision_ts.ge(start)&scoped.entry_decision_ts.lt(held)&scoped.policy_label_available_ts.lt(held)].copy()
            test=scoped.loc[scoped.entry_decision_ts.ge(held)&scoped.entry_decision_ts.lt(end)].copy()
            if train.candidate_id.nunique()<100 or test.empty: continue
            for spec in SPECS:
                model,_=c1._fit(train,spec)
                pred=_predict_vector(model,spec,test.loc[:,FEATURES])
                frame=test.loc[:,["candidate_id","entry_decision_ts","state_decision_ts","state_bar_15m","continuation_delta_bps"]].copy()
                frame["floor_bps"],frame["held_month"],frame["model_spec"],frame["prediction"]=floor,held.strftime("%Y-%m"),spec,pred
                frame["quintile"]=pd.qcut(frame.prediction.rank(method="first"),5,labels=False,duplicates="drop")
                buckets=frame.groupby("quintile",observed=True).continuation_delta_bps.mean()
                metrics.append({"floor_bps":floor,"held_month":held.strftime("%Y-%m"),"model_spec":spec,"states":len(frame),"entries":frame.candidate_id.nunique(),"spearman":frame[["prediction","continuation_delta_bps"]].corr(method="spearman").iloc[0,1],"bottom_quintile_bps":buckets.iloc[0] if len(buckets) else np.nan,"top_quintile_bps":buckets.iloc[-1] if len(buckets) else np.nan,"spread_bps":(buckets.iloc[-1]-buckets.iloc[0]) if len(buckets)>1 else np.nan})
                rows.append(frame)
    pred=pd.concat(rows,ignore_index=True); monthly=pd.DataFrame(metrics)
    agg=monthly.groupby(["floor_bps","model_spec"],as_index=False).agg(held_months=("held_month","nunique"),states=("states","sum"),entries=("entries","sum"),mean_spearman=("spearman","mean"),worst_spearman=("spearman","min"),mean_spread_bps=("spread_bps","mean"),worst_spread_bps=("spread_bps","min"))
    pred.to_parquet(out/"walkforward_predictions.parquet",index=False);monthly.to_parquet(out/"monthly_metrics.parquet",index=False);agg.to_parquet(out/"aggregate_metrics.parquet",index=False)
    (out/"run_manifest.json").write_text(json.dumps({"schema":"p8u-continuation-hpo-v1","scope":"offline strict-OOS predictive only","state_update":"one prediction after every completed 15m bar; action only next interval","fold":"previous two calendar months with labels resolved before held boundary","floors_bps":FLOORS,"specs":SPECS,"model_hpo":{"depth":4,"min_leaf_fraction":.02,"lgb_losses":["L1","L2"],"catboost_losses":["MAE","RMSE"],"target_variants":["ordinal continuation grade","continuation delta bps"]}},indent=2)+"\n")

if __name__=="__main__": main()
