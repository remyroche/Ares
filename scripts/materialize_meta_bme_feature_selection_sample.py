#!/usr/bin/env python3
"""Materialize a deterministic wide meta sample for one-shot feature selection.

Fresh meta MDA needs the broad pre-selection feature universe, but it does not
need that universe for every historical candidate. This utility samples equal
support from the beginning, middle, and end thirds using only row keys during
sampling, then projects the selected keys onto the wide handoff and outcome
ledger. Full model fitting can subsequently reload only the selected columns.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb

from extreme_price_movements.lgbm_pipeline import materialize_bme_parquet_sample


def _sql_path(path: Path) -> str:
    return str(path.resolve()).replace("'", "''")


def materialize_sample(
    *,
    handoff: Path,
    ledger: Path,
    out_dir: Path,
    rows: int,
    seed: int,
) -> dict[str, object]:
    if rows < 300:
        raise ValueError("Feature-selection sample must contain at least 300 rows")
    out_dir.mkdir(parents=True, exist_ok=True)
    sampled_handoff = out_dir / "train_meta_regime_handoff.parquet"
    sampled_ledger = out_dir / "s52_trailing_regime_scored_ledger.parquet"
    sample_contract = materialize_bme_parquet_sample(
        handoff,
        sampled_handoff,
        max_rows=int(rows),
        seed=int(seed),
        timestamp_column="__ts__",
        identity_columns=("__symbol__", "side_name"),
    )
    band_limits = list(sample_contract["band_limits"])
    handoff_sql = _sql_path(handoff)
    ledger_sql = _sql_path(ledger)
    sampled_handoff_sql = _sql_path(sampled_handoff)
    sampled_ledger_sql = _sql_path(sampled_ledger)
    connection = duckdb.connect()
    connection.execute("SET TimeZone='UTC'")
    connection.execute("PRAGMA threads=4")
    connection.execute("PRAGMA memory_limit='8GB'")
    connection.execute(
        f"""
        COPY (
            SELECT l.*
            FROM read_parquet('{ledger_sql}') AS l
            INNER JOIN read_parquet('{sampled_handoff_sql}') AS h
              ON l.__ts__ = h.__ts__
             AND l.__symbol__ = h.__symbol__
             AND lower(l.side_name) = lower(h.side_name)
            ORDER BY l.__ts__, l.__symbol__, lower(l.side_name)
        ) TO '{sampled_ledger_sql}'
        (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 10000)
        """
    )
    summary = connection.execute(
        f"""
        WITH source AS (
            SELECT
                __ts__, __symbol__, lower(side_name) AS side_name,
                ntile(3) OVER (
                    ORDER BY __ts__, __symbol__, lower(side_name)
                ) AS time_band
            FROM read_parquet('{handoff_sql}')
        ), sampled AS (
            SELECT __ts__, __symbol__, lower(side_name) AS side_name
            FROM read_parquet('{sampled_handoff_sql}')
        )
        SELECT
            s.time_band,
            count(*) AS rows,
            min(s.__ts__) AS min_ts,
            max(s.__ts__) AS max_ts
        FROM source AS s
        INNER JOIN sampled AS x
          ON s.__ts__ = x.__ts__
         AND s.__symbol__ = x.__symbol__
         AND s.side_name = x.side_name
        GROUP BY s.time_band
        ORDER BY s.time_band
        """
    ).fetchdf()
    handoff_rows = int(
        connection.execute(
            f"SELECT count(*) FROM read_parquet('{sampled_handoff_sql}')"
        ).fetchone()[0]
    )
    ledger_rows = int(
        connection.execute(
            f"SELECT count(*) FROM read_parquet('{sampled_ledger_sql}')"
        ).fetchone()[0]
    )
    if handoff_rows != ledger_rows or handoff_rows != int(rows):
        raise RuntimeError(
            f"Sample alignment failed: handoff={handoff_rows} ledger={ledger_rows} "
            f"expected={rows}"
        )
    manifest: dict[str, object] = {
        "schema": "meta_bme_feature_selection_sample_v1",
        "timestamp_contract": "UTC",
        "source_handoff": str(handoff.resolve()),
        "source_ledger": str(ledger.resolve()),
        "sampled_handoff": str(sampled_handoff.resolve()),
        "sampled_ledger": str(sampled_ledger.resolve()),
        "sampling": {
            "method": "lgbm_pipeline_two_phase_bme_parquet_sample",
            "requested_rows": int(rows),
            "seed": int(seed),
            "band_limits": band_limits,
            "contract": sample_contract,
        },
        "rows": handoff_rows,
        "bands": summary.assign(
            min_ts=summary["min_ts"].astype(str),
            max_ts=summary["max_ts"].astype(str),
        ).to_dict(orient="records"),
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--handoff", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=45_000)
    parser.add_argument("--seed", type=int, default=20260705)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    manifest = materialize_sample(
        handoff=args.handoff,
        ledger=args.ledger,
        out_dir=args.out_dir,
        rows=int(args.rows),
        seed=int(args.seed),
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
