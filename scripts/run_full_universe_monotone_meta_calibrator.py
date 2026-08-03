#!/usr/bin/env python3
"""Fit a shallow monotone base-value/reliability calibrator on development only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd


def _metrics(frame: pd.DataFrame, score: np.ndarray, fraction: float) -> dict:
    z=frame.assign(_score=score).sort_values(["_score","candidate_id"],ascending=[False,True]).head(int(np.ceil(len(frame)*fraction)))
    return {"n":int(len(z)),"gross_bps":float(z.gross_bps.mean()),"net_bps":float(z.net_bps.mean()),"long_n":int(z.side_name.eq("long").sum()),"short_n":int(z.side_name.eq("short").sum())}


def main() -> None:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--development",type=Path,required=True);ap.add_argument("--oos",type=Path,required=True);ap.add_argument("--out",type=Path,required=True);ap.add_argument("--top-fraction",type=float,default=.10)
    a=ap.parse_args()
    dev,oos=pd.read_parquet(a.development),pd.read_parquet(a.oos)
    fields=["base_expected_net_bps","meta_score"]
    xdev=dev[fields].to_numpy(float);xoos=oos[fields].to_numpy(float)
    # A fixed, deliberately low-capacity monotone surface: expected value and
    # reliability can only help, never hurt, the final expected-net estimate.
    model=lgb.LGBMRegressor(objective="huber",alpha=.9,n_estimators=80,learning_rate=.05,num_leaves=4,max_depth=2,min_child_samples=1500,reg_lambda=30.,monotone_constraints=[1,1],random_state=20260803,n_jobs=1,verbosity=-1)
    model.fit(xdev,dev.net_bps.to_numpy(float))
    pdev,poos=model.predict(xdev),model.predict(xoos)
    result={"schema":"full_universe_monotone_meta_calibrator_v1","inputs":fields,"contract":"development fit only; monotone increasing in base expected net and cost-clear probability","model":{"num_leaves":4,"max_depth":2,"min_child_samples":1500,"monotone_constraints":[1,1]},"top_fraction":a.top_fraction,"development":{"base":_metrics(dev,dev.base_expected_net_bps.to_numpy(float),a.top_fraction),"calibrated":_metrics(dev,pdev,a.top_fraction)},"oos":{"base":_metrics(oos,oos.base_expected_net_bps.to_numpy(float),a.top_fraction),"calibrated":_metrics(oos,poos,a.top_fraction)}}
    a.out.mkdir(parents=True,exist_ok=True);pd.DataFrame({"candidate_id":oos.candidate_id,"monotone_score":poos}).to_parquet(a.out/"oos_predictions.parquet",index=False);(a.out/"manifest.json").write_text(json.dumps(result,indent=2));print(json.dumps(result))


if __name__=="__main__":main()
