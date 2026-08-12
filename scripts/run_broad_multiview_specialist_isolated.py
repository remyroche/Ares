#!/usr/bin/env python3
"""Run broad specialist folds in isolated processes and merge checkpoints.

LightGBM, Arrow and pandas retain allocator arenas after each large fold on
macOS.  This wrapper preserves the exact per-fold experiment, but makes the
operating-system reclaim that memory before the next chronological fold.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
RUNNER=ROOT/'scripts/run_broad_multiview_specialist_lambdarank.py'
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))


def _args() -> argparse.Namespace:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--specialist-count',type=int,required=True,choices=range(6,13))
    p.add_argument('--max-meta-heads',type=int,default=6)
    p.add_argument('--fold-count',type=int,default=None)
    return p.parse_args()


def main() -> Path:
    args=_args(); args.out.mkdir(parents=True,exist_ok=True)
    from scripts.run_market_spine_covariance_meta import LONG_HISTORY_FOLDS
    folds=LONG_HISTORY_FOLDS[3:]
    if args.fold_count is not None: folds=folds[:args.fold_count]
    completed=[]
    for index, fold in enumerate(folds):
        cell=args.out/'folds'/f'{index:02d}_{fold.name}'
        cell.mkdir(parents=True,exist_ok=True)
        # A fold is self-contained.  Retain a finished cell on restart instead
        # of rerunning it, which keeps interrupted transport experiments both
        # reproducible and economical.
        required=(cell/'metrics.parquet', cell/'predictions.parquet', cell/'view_discovery.parquet', cell/'routing.parquet')
        if not all(path.exists() for path in required):
            command=[sys.executable,str(RUNNER),'--out',str(cell),'--specialist-count',str(args.specialist_count),'--max-meta-heads',str(args.max_meta_heads),'--fold-index',str(index)]
            log=cell/'run.log'
            with log.open('w') as handle:
                subprocess.run(command,cwd=ROOT,stdout=handle,stderr=subprocess.STDOUT,check=True)
        completed.append(fold.name)
        (args.out/'progress.json').write_text(json.dumps({'status':'running','completed_folds':completed},indent=2)+'\n')
    outputs={'metrics.parquet':'metrics','predictions.parquet':'predictions','view_discovery.parquet':'view_discovery','routing.parquet':'routing'}
    for output, stem in outputs.items():
        parts=[pd.read_parquet(args.out/'folds'/f'{index:02d}_{fold.name}'/output) for index,fold in enumerate(folds)]
        pd.concat(parts,ignore_index=True).to_parquet(args.out/output,index=False)
    (args.out/'manifest.json').write_text(json.dumps({'schema':'broad_multiview_specialist_isolated_v1','specialist_count':args.specialist_count,'max_meta_heads':args.max_meta_heads,'folds':completed,'target':'exact_h12_net_bps_gt_50','execution':'one fresh Python process per fold'},indent=2)+'\n')
    (args.out/'progress.json').write_text(json.dumps({'status':'complete','completed_folds':completed},indent=2)+'\n')
    return args.out


if __name__=='__main__': print(main())
