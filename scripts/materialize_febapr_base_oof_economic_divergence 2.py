#!/usr/bin/env python3
"""Materialize the frozen base-label/execution-EV divergence diagnostic."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from extreme_price_movements.base_oof_economic_divergence import build_divergence_diagnostic
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def safe(x):
 if isinstance(x,dict):return {str(k):safe(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [safe(v) for v in x]
 if isinstance(x,(np.generic,pd.Timestamp,Path)):return x.item() if isinstance(x,np.generic) else str(x)
 if isinstance(x,float) and not np.isfinite(x):return None
 return x
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--base-oof',type=Path,default=ROOT/'data_perp/artifacts/febapr2025_canonical_base_oof_20260727_v1/oof_predictions.parquet');p.add_argument('--execution-labels',type=Path,default=ROOT/'data_perp/artifacts/febapr2025_execution_ev_current_spread_12h_labels_20260727_v1/labels.parquet');p.add_argument('--population',type=Path,default=ROOT/'data_perp/artifacts/febapr2025_canonical_exact_policy_base_population_20260727_v2/population.parquet');p.add_argument('--output-dir',type=Path,default=ROOT/'data_perp/artifacts/febapr2025_base_oof_economic_divergence_20260727_v1');a=p.parse_args()
 if a.output_dir.exists():raise FileExistsError(a.output_dir)
 tables,manifest=build_divergence_diagnostic(pd.read_parquet(a.base_oof),pd.read_parquet(a.execution_labels),pd.read_parquet(a.population));a.output_dir.mkdir(parents=True)
 manifest['sources']={k:{'path':str(v),'sha256':sha(v)} for k,v in {'base_oof':a.base_oof,'exact_execution_labels':a.execution_labels,'transition_population':a.population}.items()};manifest['outputs']={}
 for name,table in tables.items():
  path=a.output_dir/f'{name}.parquet';table.to_parquet(path,index=False);manifest['outputs'][name]=sha(path)
 (a.output_dir/'manifest.json').write_text(json.dumps(safe(manifest),indent=2,sort_keys=True)+'\n');print(json.dumps(safe(manifest['headline']),indent=2))
if __name__=='__main__':main()
