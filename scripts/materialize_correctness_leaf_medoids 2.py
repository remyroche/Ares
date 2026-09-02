#!/usr/bin/env python3
"""Materialise the deterministic medoid of every discovered leaf-rule cluster."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
DEFAULT=ROOT/'data_perp/artifacts/correctness_leaf_regime_oof_20260803_v3'

def run(source: Path=DEFAULT) -> pd.DataFrame:
 from extreme_price_movements.performance_regimes.correctness_leaf_regimes import medoid
 rules=pd.read_parquet(source/'rule_clusters.parquet')
 sim=pd.read_parquet(source/'rule_similarity.parquet')
 rows=[]
 for keys,group in rules.groupby(['target','fold','side_name','cluster'],observed=True):
  target,fold,side,cluster=keys; members=group.rule_id.tolist()
  s=sim[(sim.target.eq(target))&(sim.fold.eq(fold))&(sim.side_name.eq(side))]
  representative=medoid(members,s)
  row=group[group.rule_id.eq(representative)].iloc[0]
  rows.append({'target':target,'fold':fold,'side_name':side,'cluster':cluster,'medoid_rule_id':representative,'cluster_size':len(members),'conditions_json':row.conditions_json,'economic_effect':float(row.economic_effect)})
 out=pd.DataFrame(rows).sort_values(['target','fold','side_name','cluster'])
 out.to_parquet(source/'rule_cluster_medoids.parquet',index=False)
 print(f'{len(out)} medoids -> {source}/rule_cluster_medoids.parquet')
 return out

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--source',type=Path,default=DEFAULT);a=p.parse_args();run(a.source)
