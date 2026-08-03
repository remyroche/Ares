#!/usr/bin/env python3
"""Prequential side-local calibration of a shared high-base reliability score."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

def _fit_predict(train:pd.DataFrame, apply:pd.DataFrame)->np.ndarray:
    if len(train)<2000 or train.target.nunique()<2:return apply.raw_probability.to_numpy(float)
    model=IsotonicRegression(y_min=.001,y_max=.999,out_of_bounds='clip').fit(train.raw_probability,train.target)
    return model.predict(apply.raw_probability)
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--development',type=Path,required=True);p.add_argument('--oos',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--threshold',type=float,default=25.);p.add_argument('--minimum-rows',type=int,default=2000);a=p.parse_args()
    dev=pd.read_parquet(a.development);oos=pd.read_parquet(a.oos)
    for d in (dev,oos):
        d['__ts__']=pd.to_datetime(d.__ts__,utc=True);d['raw_probability']=d.reliability_score;d['target']=(d.net_bps>a.threshold).astype(int);d['available_at']=d.__ts__+pd.Timedelta(hours=12)
    dev['side_calibrated_probability']=np.nan;records=[]
    # Each development day uses only resolved earlier predictions/outcomes.
    for day,idx in dev.groupby(dev.__ts__.dt.floor('D'),sort=True).groups.items():
        for side in ('long','short'):
            apply=dev.loc[idx];apply=apply[(apply.side_name==side)&apply.raw_probability.notna()]
            history=dev[(dev.side_name==side)&dev.raw_probability.notna()&(dev.available_at<day)]
            if len(history)>=a.minimum_rows and history.target.nunique()==2:
                pred=_fit_predict(history,apply);status='isotonic'
            else:pred=apply.raw_probability.to_numpy(float);status='identity_warmup'
            dev.loc[apply.index,'side_calibrated_probability']=pred;records.append({'day':str(day),'side':side,'history_rows':int(len(history)),'mode':status})
    boundary=oos.__ts__.min();oos['side_calibrated_probability']=np.nan
    for side in ('long','short'):
        train=dev[(dev.side_name==side)&dev.raw_probability.notna()&(dev.available_at<boundary)]
        apply=oos[(oos.side_name==side)&oos.raw_probability.notna()]
        oos.loc[apply.index,'side_calibrated_probability']=_fit_predict(train,apply)
    for d in (dev,oos):
        d['reliability_score']=d.side_calibrated_probability;d['meta_score']=d.side_calibrated_probability
    a.out.mkdir(parents=True,exist_ok=True);dev.to_parquet(a.out/'development.parquet',index=False);oos.to_parquet(a.out/'oos.parquet',index=False)
    manifest={'schema':'full_universe_side_reliability_calibration_v1','target':f'I(net > {a.threshold:g} bps)','contract':'shared target-model probabilities; per-side isotonic calibrators with prequential development scores and final pre-OOS calibration fit','development_rows':int(len(dev)),'oos_rows':int(len(oos)),'development_calibration_records':records,'oos_calibration_training_rows':{s:int(len(dev[(dev.side_name==s)&dev.raw_probability.notna()&(dev.available_at<boundary)])) for s in ('long','short')}};(a.out/'manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps({'oos_calibration_training_rows':manifest['oos_calibration_training_rows']}))
if __name__=='__main__':main()
