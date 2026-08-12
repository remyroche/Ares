#!/usr/bin/env python3
"""Screen every predeclared query grade in one read of the label population."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from extreme_price_movements.query_candidate_definitions import materialize_query_membership, recommended_query_definitions
from extreme_price_movements.query_construction_pipeline import query_common_shock_metrics, query_geometry, query_oracle_metrics, query_pair_metrics
from extreme_price_movements.query_funnel import aggregate_portability, portability_metrics, select_pareto_shortlist, validity_audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fold-column", default="era")
    parser.add_argument("--development-end", required=True)
    args = parser.parse_args()
    frame = pd.read_parquet(args.input)
    frame["__ts__"] = pd.to_datetime(frame["__ts__"], utc=True, errors="raise")
    end = pd.Timestamp(args.development_end)
    end = end.tz_localize("UTC") if end.tzinfo is None else end.tz_convert("UTC")
    frame = frame[frame.__ts__.lt(end)].copy()
    if frame.candidate_id.duplicated().any():
        raise ValueError("candidate IDs must be unique")
    grades = [column for column in frame if column.startswith("grade_")]
    if not grades:
        raise ValueError("no predeclared grade columns")
    membership = materialize_query_membership(frame, recommended_query_definitions())
    validity = validity_audit(frame, membership, fold_column=args.fold_column)
    if (validity.future_membership_violation_count.ne(0) | validity.query_boundary_violation_count.ne(0) | validity.candidate_duplicate_membership_rate.ne(0)).any():
        raise ValueError("query validity audit failed")
    summaries: list[pd.DataFrame] = []
    eras: list[pd.DataFrame] = []
    shortlists: list[pd.DataFrame] = []
    for grade in grades:
        geometry = query_geometry(frame, membership, grade_column=grade)
        pairs = query_pair_metrics(frame, membership, grade_column=grade)
        oracle = query_oracle_metrics(frame, membership)
        shock = query_common_shock_metrics(frame, membership)
        era = portability_metrics(frame, membership, grade_column=grade)
        portable = aggregate_portability(era)
        summary = geometry.merge(pairs, on="query_candidate").merge(oracle, on="query_candidate").merge(shock, on="query_candidate").merge(portable, on="query_candidate")
        summary["grade_column"] = grade
        shortlist = select_pareto_shortlist(summary).assign(grade_column=grade)
        summaries.append(summary)
        eras.append(era.assign(grade_column=grade))
        shortlists.append(shortlist.loc[shortlist.shortlisted, ["grade_column", "query_candidate", "query_score", "pareto_frontier"]])
    args.out.mkdir(parents=True, exist_ok=True)
    validity.to_parquet(args.out / "query_validity_audit.parquet", index=False)
    pd.concat(summaries, ignore_index=True).to_parquet(args.out / "query_grade_grid_summary.parquet", index=False)
    pd.concat(eras, ignore_index=True).to_parquet(args.out / "query_grade_grid_portability.parquet", index=False)
    pd.concat(shortlists, ignore_index=True).to_parquet(args.out / "query_grade_grid_shortlist.parquet", index=False)
    (args.out / "manifest.json").write_text(json.dumps({"schema": "query_grade_grid_screen_v1", "grade_columns": grades, "development_end": end.isoformat(), "query_definitions": [q.manifest() for q in recommended_query_definitions()], "membership_materialised_once": True}, indent=2) + "\n")


if __name__ == "__main__":
    main()
