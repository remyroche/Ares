#!/usr/bin/env python3
"""Pool independently checkpointed target-screen side runs into global books."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd

def metrics(x):
    x=x.sort_values(['score_bps','candidate_id'],ascending=[False,True],kind='mergesort'); rows=[]
    for fraction in (.01,.05,.10,.20):
        y=x.head(int(len(x)*fraction+.999))
        rows.append(dict(top_fraction=fraction,n=len(y),gross_bps=float(y.gross_bps.mean()),net_bps=float(y.net_bps.mean()),long_n=int(y.side_name.eq('long').sum()),short_n=int(y.side_name.eq('short').sum())))
    return rows

def main():
    p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,required=True); p.add_argument('--out',type=Path,required=True); a=p.parse_args()
    runs=[]
    for d in a.root.iterdir():
        f=d/'target_screen_predictions.parquet'
        if not f.exists(): continue
        x=pd.read_parquet(f)
        if len(x) and x.side_name.nunique()==1: runs.append(x)
    grouped={}
    for x in runs:
        key=(x.family.iloc[0],x.geometry.iloc[0]); grouped.setdefault(key,[]).append(x)
    records=[]
    for (family,geometry), parts in grouped.items():
        if {z.side_name.iloc[0] for z in parts}!={'long','short'}: continue
        x=pd.concat(parts,ignore_index=True)
        for row in metrics(x): records.append(dict(family=family,geometry=geometry,month='all',**row))
        for month,y in x.groupby(pd.to_datetime(x.__ts__,utc=True).dt.to_period('M').astype(str)):
            for row in metrics(y): records.append(dict(family=family,geometry=geometry,month=month,**row))
    out=pd.DataFrame(records); a.out.parent.mkdir(parents=True,exist_ok=True); out.to_parquet(a.out,index=False)
    print(out[(out.month=='all')&(out.top_fraction==.10)].sort_values('net_bps',ascending=False).to_string(index=False))
if __name__=='__main__': main()
