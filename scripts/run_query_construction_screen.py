#!/usr/bin/env python3
"""Run the no-model stages of the sequential LambdaRank query funnel.

The input must already contain a predeclared relevance-grade column.  This is
intentional: path-derived labels are materialised separately and this script
never opens final OOS data to choose a query definition.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from extreme_price_movements.query_candidate_definitions import (
    materialize_query_membership,
    query_definitions_by_name,
    recommended_query_definitions,
)
from extreme_price_movements.query_construction_pipeline import query_common_shock_metrics, query_geometry, query_oracle_metrics, query_pair_metrics
from extreme_price_movements.query_funnel import aggregate_portability, portability_metrics, select_pareto_shortlist, validity_audit


def args() -> argparse.Namespace:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--grade-column",required=True)
    p.add_argument("--fold-column",default="fold")
    p.add_argument("--development-end",default=None,help="exclusive UTC timestamp; omit only for a development-only input")
    p.add_argument("--query-names", nargs="*", default=None,
                   help="optional frozen subset of the predeclared query grammar")
    return p.parse_args()


def main() -> None:
    a=args(); x=pd.read_parquet(a.input)
    required={"candidate_id","__ts__","side_name","net_bps","gross_bps",a.grade_column}
    missing=required.difference(x); 
    if missing: raise KeyError(f"query screen input missing {sorted(missing)}")
    x["__ts__"]=pd.to_datetime(x["__ts__"],utc=True,errors="raise")
    if a.development_end:
        end=pd.Timestamp(a.development_end)
        if end.tzinfo is None: end=end.tz_localize("UTC")
        x=x[x.__ts__.lt(end)].copy()
    if x.candidate_id.duplicated().any(): raise ValueError("input candidate IDs must be unique")
    if "atr_bps" in x:
        # Do not convert a field already expressed in bps a second time.
        x["atr_bps"] = pd.to_numeric(x["atr_bps"], errors="coerce")
    elif "atr_1h" in x and "decision_price" in x:
        x["atr_bps"] = (
            pd.to_numeric(x["atr_1h"], errors="coerce")
            / pd.to_numeric(x["decision_price"], errors="coerce")
            * 10_000.0
        )
    else:
        x["atr_bps"] = float("nan")
    definitions = (
        query_definitions_by_name(a.query_names)
        if a.query_names else recommended_query_definitions()
    )
    membership=materialize_query_membership(x,definitions)
    validity=validity_audit(x,membership,fold_column=a.fold_column)
    if (validity.future_membership_violation_count.ne(0)|validity.query_boundary_violation_count.ne(0)|validity.candidate_duplicate_membership_rate.ne(0)).any():
        raise ValueError("inference-validity audit failed")
    geometry=query_geometry(x,membership,grade_column=a.grade_column)
    pair=query_pair_metrics(x,membership,grade_column=a.grade_column)
    oracle=query_oracle_metrics(x,membership)
    shock=query_common_shock_metrics(x,membership)
    era=portability_metrics(x,membership,grade_column=a.grade_column)
    portable=aggregate_portability(era)
    summary=geometry.merge(pair,on="query_candidate").merge(oracle,on="query_candidate").merge(shock,on="query_candidate").merge(portable,on="query_candidate")
    shortlist=select_pareto_shortlist(summary)
    a.out.mkdir(parents=True,exist_ok=True)
    membership.to_parquet(a.out/'candidate_query_membership.parquet',index=False)
    validity.to_parquet(a.out/'query_validity_audit.parquet',index=False)
    geometry.to_parquet(a.out/'query_geometry_metrics.parquet',index=False)
    pair.to_parquet(a.out/'query_pair_metrics.parquet',index=False)
    oracle.to_parquet(a.out/'query_oracle_metrics.parquet',index=False)
    shock.to_parquet(a.out/'query_common_shock_metrics.parquet',index=False)
    era.to_parquet(a.out/'query_portability_metrics.parquet',index=False)
    shortlist.to_parquet(a.out/'query_pareto_frontier.parquet',index=False)
    shortlisted=shortlist.loc[shortlist.shortlisted,"query_candidate"].tolist()
    (a.out/'query_shortlist.json').write_text(json.dumps({"definitions":[q.manifest() for q in definitions],"shortlist":shortlisted,"grade_column":a.grade_column,"development_end":a.development_end},indent=2)+'\n')
    (a.out/'query_candidate_manifest.json').write_text(json.dumps({"schema":"query_construction_screen_v2","definitions":[q.manifest() for q in definitions],"evaluation_query":"decision_timestamp × side","side_local_training":True,"atr_bps_lineage":"direct atr_bps when available; otherwise atr_1h / decision_price"},indent=2)+'\n')


if __name__=="__main__": main()
