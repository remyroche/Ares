#!/usr/bin/env python3
"""Materialize strict residual-OOF transition comparison; never route trades."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from extreme_price_movements.residual_oof_transition_diagnostic import build_residual_transition_diagnostic

def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
def safe(value):
    if isinstance(value,dict): return {str(k):safe(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [safe(v) for v in value]
    if isinstance(value,(Path,pd.Timestamp)): return str(value)
    if isinstance(value,np.generic): return value.item()
    if isinstance(value,float) and not np.isfinite(value): return None
    return value
def main() -> None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--residual-oof',type=Path,default=ROOT/'data_perp/artifacts/febapr2025_canonical_residual_oof_20260727_v1/oof_predictions.parquet')
    p.add_argument('--windows',type=Path,default=ROOT/'data_perp/artifacts/febapr2025_canonical_exact_policy_base_population_20260727_v2/transition_event_windows.parquet')
    p.add_argument('--active-hours',type=Path,default=ROOT/'data_perp/artifacts/regime_transition_active_head_20260726_v1/grouped_oof.parquet')
    p.add_argument('--output-dir',type=Path,default=ROOT/'data_perp/artifacts/marapr2025_strict_residual_oof_transition_diagnostic_20260727_v1')
    a=p.parse_args()
    if a.output_dir.exists(): raise FileExistsError(a.output_dir)
    cov,metrics,manifest=build_residual_transition_diagnostic(pd.read_parquet(a.residual_oof),pd.read_parquet(a.windows),pd.read_parquet(a.active_hours))
    a.output_dir.mkdir(parents=True);cov.to_parquet(a.output_dir/'event_coverage.parquet',index=False);metrics.to_parquet(a.output_dir/'event_phase_side_metrics.parquet',index=False)
    manifest['sources']={k:{'path':str(v),'sha256':sha(v)} for k,v in {'strict_residual_oof':a.residual_oof,'frozen_event_windows':a.windows,'expost_active_hours':a.active_hours}.items()};manifest['outputs']={'event_coverage':sha(a.output_dir/'event_coverage.parquet'),'event_phase_side_metrics':sha(a.output_dir/'event_phase_side_metrics.parquet')}
    (a.output_dir/'manifest.json').write_text(json.dumps(safe(manifest),indent=2,sort_keys=True)+'\n');print(json.dumps(safe(manifest),indent=2))
if __name__=='__main__': main()
