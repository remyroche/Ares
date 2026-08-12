#!/usr/bin/env python3
"""Join complete H12 grade labels to the frozen residual candidate universe."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb

ROOT=Path(__file__).resolve().parents[1]
DEFAULT_LEDGER=ROOT/'data_perp/artifacts/tp6_m6_shared_regime_residual_features_20260809_v3/shared_regime_residual_ledger.parquet'


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--grades',type=Path,required=True); p.add_argument('--out',type=Path,required=True); p.add_argument('--ledger',type=Path,default=DEFAULT_LEDGER); a=p.parse_args()
    manifest=json.loads((a.grades/'manifest.json').read_text())
    if manifest.get('status')!='complete': raise ValueError('refusing partial grade grid for query screening')
    grade_glob=str(a.grades/'symbol=*.parquet')
    con=duckdb.connect(); con.execute('PRAGMA threads=2')
    grade_cols=', '.join(f'g."{name}"' for name in manifest['grade_columns'])
    # Grade path economics are the realised H12 values used in query proxies;
    # frozen residual economics are retained separately for later model replay.
    query=f'''SELECT l.candidate_id, l.__ts__, l.side_name, l.era,
                     g.__decision_ts__, g.__label_available_at__, g.label_valid,
                     g.atr_bps, g.terminal_gross_bps AS gross_bps,
                     g.terminal_net_bps AS net_bps,
                     l.gross_bps AS residual_gross_bps, l.net_bps AS residual_net_bps,
                     l.p_clear, l.p_adverse, l.p_weak, l.prequential_base_expected_net_bps,
                     {grade_cols}
              FROM read_parquet(?) l INNER JOIN read_parquet(?) g USING(candidate_id)
              WHERE g.label_valid AND l.shared_regime_contract_complete'''
    a.out.parent.mkdir(parents=True,exist_ok=True)
    escaped=str(a.out).replace("'", "''")
    con.execute(f"COPY ({query}) TO '{escaped}' (FORMAT PARQUET, COMPRESSION ZSTD)",[str(a.ledger),grade_glob])
    count=con.execute('SELECT count(*) FROM read_parquet(?)',[str(a.out)]).fetchone()[0]; con.close()
    (a.out.with_suffix('.manifest.json')).write_text(json.dumps({'schema':'query_screen_population_v1','rows':count,'source_ledger':str(a.ledger),'source_grades':str(a.grades),'path_economics':'exact_terminal_h12','residual_economics_retained_separately':True},indent=2)+'\n')


if __name__=='__main__': main()
