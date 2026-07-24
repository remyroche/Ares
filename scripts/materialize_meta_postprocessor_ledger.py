#!/usr/bin/env python3
"""Materialize the compact score/outcome contract consumed by v9 and MLP."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq


KEYS = ("__ts__", "__symbol__", "side_name")


def _column(names: set[str], *candidates: str) -> str:
    for candidate in candidates:
        if candidate in names:
            return candidate
    raise ValueError(f"Missing required column; tried {candidates}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--score-col", default="score_meta_base_soft_label"
    )
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument(
        "--rank-contract",
        choices=("causal_expanding", "global_percentile"),
        default="causal_expanding",
    )
    args = parser.parse_args()

    source = args.predictions
    if source.is_dir():
        paths = sorted(source.glob("*.parquet"))
    else:
        paths = [source]
    if not paths:
        raise FileNotFoundError(f"No prediction parquet files under {source}")
    names = set(pq.read_schema(paths[0]).names)
    for key in KEYS:
        if key not in names:
            raise ValueError(f"Prediction source is missing key {key}")
    score = _column(names, args.score_col)
    fold = _column(names, "oos_fold")
    valid_start = _column(names, "valid_start")
    archetype = _column(
        names,
        "archetype_policy_key",
        "__archetype_policy_key__",
        "policy_archetype",
        "local_side_archetype",
    )
    outcome_aliases = {
        "ev_after_1pct": ("ev_after_1pct",),
        "clean_exec": ("clean_exec",),
        "dirty_positive": ("dirty_positive",),
        "full_path_bad_mae_1r": ("full_path_bad_mae_1r",),
        "timeout": ("timeout",),
    }
    outcomes = {
        alias: _column(names, *candidates)
        for alias, candidates in outcome_aliases.items()
    }
    quoted_paths = ", ".join(
        "'" + str(path.resolve()).replace("'", "''") + "'" for path in paths
    )
    input_expr = quoted_paths if len(paths) > 1 else quoted_paths
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute(f"PRAGMA threads={max(1, int(args.threads))}")
    con.execute("PRAGMA memory_limit='3GB'")
    con.execute(
        f"CREATE VIEW src AS SELECT * FROM read_parquet([{input_expr}], union_by_name=true)"
    )
    outcome_sql = ",\n                ".join(
        f'CAST("{source_name}" AS FLOAT) AS "{alias}"'
        for alias, source_name in outcomes.items()
    )
    query = f"""
        COPY (
            WITH compact AS (
                SELECT
                    __ts__, __symbol__, lower(CAST(side_name AS VARCHAR)) AS side_name,
                    CAST("{fold}" AS VARCHAR) AS oos_fold,
                    CAST("{valid_start}" AS TIMESTAMP) AS valid_start,
                    CAST("{archetype}" AS VARCHAR) AS archetype_policy_key,
                    CAST("{score}" AS FLOAT) AS hit_probability,
                    {outcome_sql}
                FROM src
                WHERE isfinite(CAST("{score}" AS DOUBLE))
                QUALIFY row_number() OVER (
                    PARTITION BY __ts__, __symbol__, side_name,
                                 CAST("{archetype}" AS VARCHAR)
                    ORDER BY __ts__ DESC
                ) = 1
            ), ranked AS (
                SELECT *,
                    CAST(percent_rank() OVER (ORDER BY hit_probability) AS FLOAT)
                        AS historical_rank
                FROM compact
            )
            SELECT *,
                historical_rank AS historical_rank_adjusted,
                hit_probability AS hit_prob_adjusted
            FROM ranked
        ) TO '{str(output).replace("'", "''")}'
        (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)
    """
    con.execute(query)
    rank_contract = "global_percent_rank_of_OOF_meta_score"
    if args.rank_contract == "causal_expanding":
        frame = pd.read_parquet(output)
        frame["valid_start"] = pd.to_datetime(
            frame["valid_start"], utc=True, errors="coerce"
        )
        rank = np.full(len(frame), 0.5, dtype=np.float32)
        reference = np.empty(0, dtype=np.float32)
        ordered_folds = (
            frame.loc[:, ["oos_fold", "valid_start"]]
            .drop_duplicates()
            .sort_values(["valid_start", "oos_fold"], kind="stable")
        )
        scores = pd.to_numeric(
            frame["hit_probability"], errors="coerce"
        ).to_numpy(dtype=np.float32)
        for fold_row in ordered_folds.itertuples(index=False):
            idx = np.flatnonzero(frame["oos_fold"].eq(fold_row.oos_fold).to_numpy())
            values = scores[idx]
            finite = np.isfinite(values)
            if reference.size == 0:
                local_reference = np.sort(values[finite])
                left = np.searchsorted(local_reference, values[finite], side="left")
                right = np.searchsorted(local_reference, values[finite], side="right")
                rank[idx[finite]] = (
                    (left + right) / (2.0 * max(local_reference.size, 1))
                ).astype(np.float32)
            else:
                left = np.searchsorted(reference, values[finite], side="left")
                right = np.searchsorted(reference, values[finite], side="right")
                rank[idx[finite]] = (
                    (left + right) / (2.0 * reference.size)
                ).astype(np.float32)
            reference = np.sort(
                np.concatenate((reference, values[finite].astype(np.float32, copy=False)))
            )
        frame["historical_rank"] = rank
        frame["historical_rank_adjusted"] = rank
        frame.to_parquet(output, index=False, compression="zstd")
        rank_contract = "causal_expanding_prior_fold_empirical_cdf"
    rows = int(con.execute(f"SELECT count(*) FROM read_parquet('{output}')").fetchone()[0])
    bounds = con.execute(
        f"SELECT min(__ts__), max(__ts__) FROM read_parquet('{output}')"
    ).fetchone()
    manifest = {
        "schema": "meta_postprocessor_compact_ledger_v1",
        "source": [str(path) for path in paths],
        "output": str(output),
        "score_col": score,
        "rank_contract": rank_contract,
        "rows": rows,
        "min_timestamp": str(bounds[0]),
        "max_timestamp": str(bounds[1]),
        "cost_contract": "ev_after_1pct already includes exactly 1% round-trip cost",
    }
    output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
