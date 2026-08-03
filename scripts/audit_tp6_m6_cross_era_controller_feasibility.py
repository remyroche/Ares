#!/usr/bin/env python3
"""Fail-closed readiness audit for TP6 M6 cross-era MDA/OOD work.

It deliberately does not impute missing context/base outputs.  A cross-era
error model needs all named causal fields on every era; otherwise its MDA and
controller results would be a one-era result disguised as transport.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd

FIELDS = [
 "base_p_upper","base_p_lower","base_p_timeout","base_margin","base_entropy","m6_probability",
 "market_median_ret_1h","market_median_ret_4h","market_median_ret_24h","market_dispersion_1h",
 "market_dispersion_4h","market_median_rv_24h","market_negative_breadth_4h","market_average_pair_corr_24h",
]
def main():
 p=argparse.ArgumentParser();p.add_argument('--ledger',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 d=pd.read_parquet(a.ledger,columns=['era','candidate_id','__ts__','net_bps',*FIELDS]); coverage=(1-d[FIELDS].isna().groupby(d.era).mean()).T
 eras={era:{field:float(coverage.loc[field,era]) for field in FIELDS} for era in coverage.columns}
 complete={era:all(v>=.90 for v in fields.values()) for era,fields in eras.items()}
 status='FEASIBLE' if len(complete)>=2 and all(complete.values()) else 'BLOCKED_SCHEMA_MISSING_COMMON_14_FIELD_CONTRACT'
 report={'schema':'tp6_m6_cross_era_controller_feasibility_v1','status':status,'rows_by_era':d.groupby('era').size().astype(int).to_dict(),'coverage_by_era':eras,'complete_era_contract':complete,'required_controller_label':'causal false-positive outcome derived only after H12 label resolution','blocker':None if status=='FEASIBLE' else 'Materialise/replay every listed base/M6/context field for each missing era keyed by candidate_id; do not impute or substitute a different stack.'}
 a.out.mkdir(parents=True,exist_ok=False);(a.out/'audit.json').write_text(json.dumps(report,indent=2)+'\n');coverage.to_csv(a.out/'coverage.csv');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
