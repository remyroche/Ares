#!/usr/bin/env python3
"""Attach all predeclared H12 query relevance grades to path primitives."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from extreme_price_movements.query_grade_grid import grade_columns


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--path-grid',type=Path,required=True); p.add_argument('--out',type=Path,required=True); p.add_argument('--resume',action='store_true'); a=p.parse_args()
    parts=sorted(a.path_grid.glob('symbol=*.parquet'))
    if not parts: raise FileNotFoundError('no path-grid symbol partitions found')
    a.out.mkdir(parents=True,exist_ok=True); columns=[]
    for part in parts:
        target=a.out/part.name
        if a.resume and target.exists():
            existing=pd.read_parquet(target,columns=None)
            columns=[c for c in existing if c.startswith('grade_')]
            if len(columns)==18 and 'label_valid' in existing: continue
        frame=grade_columns(pd.read_parquet(part)); frame.to_parquet(target,index=False,compression='zstd'); columns=[c for c in frame if c.startswith('grade_')]
    (a.out/'manifest.json').write_text(json.dumps({'schema':'h12_query_grade_grid_v1','source_path_grid':str(a.path_grid),'grade_columns':columns,'label_only':True,'invalid_rows_remain_zero_and_are_marked_label_valid_false':True,'completed_partitions':len(list(a.out.glob('symbol=*.parquet'))),'source_partitions_seen':len(parts),'status':'complete' if (a.path_grid/'manifest.json').exists() else 'partial'},indent=2)+'\n')


if __name__=='__main__': main()
