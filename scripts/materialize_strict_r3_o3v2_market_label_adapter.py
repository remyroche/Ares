#!/usr/bin/env python3
"""Adapt resolved market labels to the strict O3-v2 auxiliary-label schema."""
from __future__ import annotations
import argparse,json,os
from pathlib import Path
import pandas as pd

IDENTITY=['candidate_id','__ts__','__decision_ts__','__symbol__','side_name']
CONTROL=['aux_label_available_ts','aux_path_valid','aux_path_efficiency']
def _write(p,o):
 fd=os.open(p,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o644)
 with os.fdopen(fd,'w') as f: json.dump(o,f,indent=2,sort_keys=True,default=str)
def main():
 a=argparse.ArgumentParser();a.add_argument('--path-root',type=Path,required=True);a.add_argument('--market',type=Path,required=True);a.add_argument('--out',type=Path,required=True);z=a.parse_args()
 if z.out.exists():raise FileExistsError(z.out)
 m=pd.read_parquet(z.market);m['__decision_ts__']=pd.to_datetime(m['__decision_ts__'],utc=True)
 labels=[c for c in m if c.startswith('market_') or c.startswith('cross_sectional_')]
 z.out.mkdir(parents=True); cov=[]
 for src in sorted(z.path_root.glob('parts/month=*/auxiliary_path_labels.parquet')):
  token=src.parent.name.split('=',1)[1];x=pd.read_parquet(src,columns=IDENTITY+CONTROL);x['__decision_ts__']=pd.to_datetime(x['__decision_ts__'],utc=True)
  y=x.merge(m[['candidate_id',*labels]],on='candidate_id',how='left',validate='one_to_one')
  if y.candidate_id.duplicated().any():raise AssertionError('duplicate identity')
  d=z.out/'parts'/f'month={token}';d.mkdir(parents=True);y.to_parquet(d/'auxiliary_path_labels.parquet',index=False,compression='zstd')
  cov.append({'month':token,'rows':len(y),'market_valid':float(y.market_label_valid.fillna(False).mean())})
 pd.DataFrame(cov).to_parquet(z.out/'coverage_by_month.parquet',index=False)
 _write(z.out/'run_manifest.json',{'schema':'strict_r3_o3v2_market_label_adapter_v1','scope':'resolved labels only; no inference use','source_path':str(z.path_root),'source_market':str(z.market),'labels':labels,'coverage':cov})
 print(z.out)
if __name__=='__main__':main()
