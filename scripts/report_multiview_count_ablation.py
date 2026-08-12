#!/usr/bin/env python3
"""Build a compact, per-count specialist ablation report against control."""
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,required=True); p.add_argument('--control',type=Path,required=True); p.add_argument('--out',type=Path,required=True); a=p.parse_args()
    control=pd.read_parquet(a.control/'metrics.parquet'); control=control[(control.arm=='meta_lambdarank')&(control.side=='pooled')][['fold','tail','net_bps','gross_bps']].rename(columns={'net_bps':'control_net_bps','gross_bps':'control_gross_bps'})
    results=[]
    for location in sorted(a.root.glob('broad_multiview_binary_h12net50_*views_isolated_*')):
        manifest=location/'manifest.json'
        metrics=location/'metrics.parquet'
        if not manifest.exists() or not metrics.exists(): continue
        count=int(location.name.split('_')[4].removesuffix('views'))
        x=pd.read_parquet(metrics); x=x[(x.arm=='meta_lambdarank')&(x.side=='pooled')][['fold','tail','net_bps','gross_bps','rank_ic']]
        x=x.merge(control,on=['fold','tail'],validate='one_to_one'); x['specialist_count']=count; x['net_uplift_bps']=x.net_bps-x.control_net_bps; results.append(x)
    out=pd.concat(results,ignore_index=True) if results else pd.DataFrame()
    a.out.parent.mkdir(parents=True,exist_ok=True); out.to_parquet(a.out.with_suffix('.parquet'),index=False); out.to_csv(a.out.with_suffix('.csv'),index=False)
    if not out.empty:
        summary=out.groupby(['specialist_count','tail'],as_index=False)[['net_bps','net_uplift_bps','rank_ic']].mean()
        # Keep report generation dependency-free on minimal research hosts.
        a.out.with_suffix('.md').write_text('# Specialist-count ablation\n\n```text\n' + summary.to_string(index=False) + '\n```\n')


if __name__=='__main__': main()
