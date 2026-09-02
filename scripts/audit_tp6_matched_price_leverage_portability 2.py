#!/usr/bin/env python3
"""Matched, transport-separated portability audit for stored price–leverage fields."""
from __future__ import annotations

import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
LEDGER=ROOT/'data_perp/artifacts/tp6_m6_shared_regime_residual_features_20260809_v3/shared_regime_residual_ledger.parquet'
PANEL=ROOT/'data_perp/artifacts/full_universe_t2_t4_panel_20260801_v3/parts/*.parquet'
OUT=ROOT/'data_perp/artifacts/tp6_score_portability_admission_20260803_v1'
TRANSPORTS=(('transport_2023q4_to_2024h1',("2024-01_02","2024-05_06")),('transport_2024h1_to_h2',("2024-07_08","2024-09_10","2024-11")))
FIELDS=('price_x_oi_1d','price_x_oi_3d','price_x_oi_7d','volume_price_corr_ts_resid')


def run() -> Path:
    con=duckdb.connect(config={'threads':'2','memory_limit':'512MB','temp_directory':'/tmp'})
    records=[]
    try:
        for transport,eras in TRANSPORTS:
            era_sql=', '.join(repr(x) for x in eras)
            unions=' UNION ALL '.join(f"SELECT side_name, month, base_bucket, clear_bucket, adverse_bucket, atr_bucket, net_bps, feature, value FROM ranked_{field}" for field in FIELDS)
            ranked='\n,'.join(f"""ranked_{field} AS (
                SELECT *, '{field}' AS feature, {field} AS value
                FROM bucketed WHERE {field} IS NOT NULL
            )""" for field in FIELDS)
            query=f"""
            WITH base AS (
                SELECT l.side_name, l.__ts__, l.net_bps, l.prequential_base_expected_net_bps,
                       l.p_clear, l.p_adverse, p.atr_percentile,
                       p.price_x_oi_1d, p.price_x_oi_3d, p.price_x_oi_7d, p.volume_price_corr_ts_resid
                FROM read_parquet('{LEDGER.as_posix()}') l
                JOIN read_parquet('{PANEL.as_posix()}') p USING(candidate_id)
                WHERE l.shared_regime_contract_complete
                  AND l.prequential_base_expected_net_bps IS NOT NULL
                  AND l.era IN ({era_sql})
            ), bucketed AS (
                SELECT *,
                    date_trunc('month', __ts__) AS month,
                    ntile(10) OVER (PARTITION BY side_name, date_trunc('month', __ts__) ORDER BY prequential_base_expected_net_bps) AS base_bucket,
                    ntile(5) OVER (PARTITION BY side_name, date_trunc('month', __ts__) ORDER BY p_clear) AS clear_bucket,
                    ntile(5) OVER (PARTITION BY side_name, date_trunc('month', __ts__) ORDER BY p_adverse) AS adverse_bucket,
                    ntile(5) OVER (PARTITION BY side_name, date_trunc('month', __ts__) ORDER BY atr_percentile) AS atr_bucket
                FROM base
            ), {ranked}, stacked AS ({unions}), classified AS (
                SELECT *, value >= median(value) OVER (PARTITION BY side_name, month, base_bucket, clear_bucket, adverse_bucket, atr_bucket) AS high
                FROM stacked
            )
            SELECT '{transport}' AS transport, side_name, feature, high,
                   count(*) AS rows, avg(net_bps) AS net_bps, avg(CAST(net_bps > 0 AS DOUBLE)) AS cost_clearing_rate
            FROM classified GROUP BY ALL
            """
            data=con.execute(query).fetchdf()
            for (side,field),group in data.groupby(['side_name','feature']):
                low=group[group.high.eq(False)].iloc[0]; high=group[group.high.eq(True)].iloc[0]
                records.append({'transport':transport,'side_name':side,'feature':field,'matched_rows':int(group.rows.sum()),'high_minus_low_net_bps':float(high.net_bps-low.net_bps),'high_minus_low_cost_clearing_rate':float(high.cost_clearing_rate-low.cost_clearing_rate),'role':'INVARIANT_CONDITIONAL' if abs(high.net_bps-low.net_bps)>=10 else 'REJECTED_WEAK_CONDITIONAL'})
    finally:
        con.close()
    result=pd.DataFrame(records)
    for (_side,_feature), index in result.groupby(['side_name','feature']).groups.items():
        values=result.loc[index,'high_minus_low_net_bps'].to_numpy(float)
        stable=len(values)==2 and np.sign(values[0])==np.sign(values[1]) and np.min(np.abs(values))>=10.
        result.loc[index,'effect_sign_consistent']=len(values)==2 and np.sign(values[0])==np.sign(values[1])
        result.loc[index,'cross_transport_role']='INVARIANT_CONDITIONAL' if stable else 'REJECTED_CROSS_TRANSPORT'
    output=OUT/'matched_price_leverage_portability.parquet'; result.to_parquet(output,index=False)
    (OUT/'matched_price_leverage_manifest.json').write_text(json.dumps({'contract':'side × decision month × base-score decile × base probability quintiles × ATR-percentile quintile; high/low feature comparison','fields':list(FIELDS),'transports':[x[0] for x in TRANSPORTS],'inference_safe':False,'purpose':'matched conditional portability diagnostic only'},indent=2)+'\n')
    return output


if __name__=='__main__': print(run())
