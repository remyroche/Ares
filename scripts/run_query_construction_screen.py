#!/usr/bin/env python3
"""Run the no-model, chronological LambdaRank query-construction screen."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.query_candidate_definitions import (
    materialize_query_membership,
    query_definitions_by_name,
    recommended_query_definitions,
)
from extreme_price_movements.query_construction_pipeline import (
    query_common_shock_metrics,
    query_geometry,
    query_oracle_metrics,
    query_pair_metrics,
)
from extreme_price_movements.query_funnel import (
    aggregate_portability,
    portability_metrics,
    select_pareto_shortlist,
    validity_audit,
)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--grade-column", required=True)
    parser.add_argument("--fold-column", default="fold")
    parser.add_argument("--development-end", default=None, help="exclusive UTC decision timestamp")
    parser.add_argument("--query-names", nargs="*", default=None)
    parser.add_argument(
        "--max-p90-group-size", type=int, default=None,
        help=("optional ranker-capacity gate applied before Pareto selection; "
              "prevents oversized queries from entering a LambdaRank HPO"),
    )
    return parser.parse_args()


def run(
    *, input_path: Path, out: Path, grade_column: str, fold_column: str,
    development_end: str | None, query_names: list[str] | None,
    max_p90_group_size: int | None = None,
) -> Path:
    if out.exists():
        raise FileExistsError(f"query-screen output must be new: {out}")
    frame = pd.read_parquet(input_path)
    required = {"candidate_id", "__ts__", "side_name", "net_bps", "gross_bps", grade_column}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise KeyError(f"query screen input missing {missing}")
    frame["__ts__"] = pd.to_datetime(frame["__ts__"], utc=True, errors="raise")
    if development_end:
        end = pd.to_datetime(development_end, utc=True)
        frame = frame.loc[frame.__ts__.lt(end)].copy()
    if frame.empty or frame.candidate_id.duplicated().any():
        raise ValueError("query screen input is empty or candidate identities are not unique")
    if "atr_bps" not in frame:
        frame["atr_bps"] = float("nan")
    else:
        frame["atr_bps"] = pd.to_numeric(frame["atr_bps"], errors="coerce")
    definitions = query_definitions_by_name(query_names) if query_names else recommended_query_definitions()
    membership = materialize_query_membership(frame, definitions)
    validity = validity_audit(frame, membership, fold_column=fold_column, executable_column="entry_executable")
    bad = (
        validity.future_membership_violation_count.ne(0)
        | validity.query_boundary_violation_count.ne(0)
        | validity.candidate_duplicate_membership_rate.ne(0)
    )
    if bad.any():
        raise ValueError("inference-valid query membership audit failed")
    geometry = query_geometry(frame, membership, grade_column=grade_column)
    pair = query_pair_metrics(frame, membership, grade_column=grade_column)
    oracle = query_oracle_metrics(frame, membership)
    shock = query_common_shock_metrics(frame, membership)
    era = portability_metrics(frame, membership, grade_column=grade_column)
    portable = aggregate_portability(era)
    summary = (
        geometry.merge(pair, on="query_candidate")
        .merge(oracle, on="query_candidate")
        .merge(shock, on="query_candidate")
        .merge(portable, on="query_candidate")
    )
    # A no-model screen can legitimately favour an all-day group because its
    # oracle has more candidates to choose from.  That does *not* make it a
    # viable LambdaRank group: NDCG gradient work grows sharply with query
    # size.  Gate such groups on the label-only development population before
    # the Pareto step, never after seeing an OOS model result.
    if max_p90_group_size is not None:
        if max_p90_group_size < 2:
            raise ValueError("max_p90_group_size must be at least two")
        summary["ranker_capacity_eligible"] = (
            pd.to_numeric(summary["p90_group_size"], errors="coerce")
            .le(float(max_p90_group_size))
            .fillna(False)
        )
        eligible = summary.loc[summary.ranker_capacity_eligible].copy()
        if eligible.empty:
            raise ValueError("rank capacity gate excluded every query candidate")
        shortlist = select_pareto_shortlist(eligible)
        shortlist = summary.merge(
            shortlist[["query_candidate", "pareto_frontier", "query_score", "shortlisted"]],
            on="query_candidate", how="left",
        )
        shortlist["pareto_frontier"] = shortlist.pareto_frontier.fillna(False)
        shortlist["shortlisted"] = shortlist.shortlisted.fillna(False)
    else:
        summary["ranker_capacity_eligible"] = True
        shortlist = select_pareto_shortlist(summary)
    out.mkdir(parents=True)
    for name, value in {
        "candidate_query_membership.parquet": membership,
        "query_validity_audit.parquet": validity,
        "query_geometry_metrics.parquet": geometry,
        "query_pair_metrics.parquet": pair,
        "query_oracle_metrics.parquet": oracle,
        "query_common_shock_metrics.parquet": shock,
        "query_portability_metrics.parquet": era,
        "query_pareto_frontier.parquet": shortlist,
    }.items():
        value.to_parquet(out / name, index=False, compression="zstd")
    selected = shortlist.loc[shortlist.shortlisted, "query_candidate"].tolist()
    (out / "query_shortlist.json").write_text(json.dumps({
        "schema": "query_construction_screen_v3",
        "definitions": [definition.manifest() for definition in definitions],
        "shortlist": selected,
        "grade_column": grade_column,
        "development_end": development_end,
        "side_local_training": True,
        "max_p90_group_size": max_p90_group_size,
        "source": str(input_path),
    }, indent=2) + "\n")
    return out


def main() -> None:
    args = _args()
    print(run(
        input_path=args.input, out=args.out, grade_column=args.grade_column,
        fold_column=args.fold_column, development_end=args.development_end,
        query_names=args.query_names, max_p90_group_size=args.max_p90_group_size,
    ))


if __name__ == "__main__":
    main()
