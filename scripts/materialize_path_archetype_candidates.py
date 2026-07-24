#!/usr/bin/env python3
"""Join the canonical base candidate population to path-label geometry inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import duckdb
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def materialize(
    population_path: Path,
    labels_path: Path,
    output_dir: Path,
    *,
    cost_return: float = 0.01,
) -> dict[str, Any]:
    if not 0.0 <= float(cost_return) < 1.0:
        raise ValueError("cost_return must be in [0, 1)")
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "candidates.parquet"
    label_glob = str(labels_path / "*.parquet") if labels_path.is_dir() else str(labels_path)
    population_columns = set(pq.read_schema(population_path).names)
    required_population = {"__ts__", "__symbol__", "side_name", "score", "selected_top40"}
    missing = sorted(required_population.difference(population_columns))
    if missing:
        raise ValueError(f"candidate population is missing columns: {missing}")
    label_files = sorted(labels_path.glob("*.parquet")) if labels_path.is_dir() else [labels_path]
    if not label_files:
        raise FileNotFoundError(f"no path-label parquet files under {labels_path}")
    label_columns = set(pq.read_schema(label_files[0]).names)
    required_labels = {
        "__ts__",
        "__symbol__",
        "side_name",
        "side",
        "__barrier_pct__",
        "__path_auxiliary_atr_fraction__",
        "candidate_id",
    }
    missing = sorted(required_labels.difference(label_columns))
    if missing:
        raise ValueError(f"path-label source is missing columns: {missing}")
    con = duckdb.connect()
    try:
        con.execute("SET TimeZone='UTC'")
        duplicate_population = con.execute(
            """
            SELECT count(*) - count(DISTINCT (epoch_ns(__ts__), __symbol__, side_name))
            FROM read_parquet(?) WHERE coalesce(selected_top40, false)
            """,
            [str(population_path)],
        ).fetchone()[0]
        if int(duplicate_population) != 0:
            raise ValueError("candidate population has duplicate UTC symbol-side identities")
        duplicate_labels = con.execute(
            """
            SELECT count(*) - count(DISTINCT (epoch_ns(__ts__), __symbol__, lower(side_name)))
            FROM read_parquet(?)
            """,
            [label_glob],
        ).fetchone()[0]
        if int(duplicate_labels) != 0:
            raise ValueError("path-label source has duplicate UTC symbol-side identities")
        quoted_output = str(output).replace("'", "''")
        con.execute(
            f"""
            COPY (
                SELECT
                    p.__ts__, p.__symbol__, l.side, lower(p.side_name) AS side_name,
                    l.__barrier_pct__, l.__path_auxiliary_atr_fraction__,
                    l.candidate_id, CAST(? AS DOUBLE) AS path_cost_return,
                    p.score AS base_oof_score,
                    p.base_candidate_rank_timestamp_side,
                    p.base_candidate_rank_pct_timestamp_side,
                    true AS selected_top40
                FROM read_parquet(?) p
                INNER JOIN read_parquet(?) l
                  ON epoch_ns(p.__ts__) = epoch_ns(l.__ts__)
                 AND p.__symbol__ = l.__symbol__
                 AND lower(p.side_name) = lower(l.side_name)
                WHERE coalesce(p.selected_top40, false)
                ORDER BY p.__ts__, p.__symbol__, lower(p.side_name)
            ) TO '{quoted_output}' (FORMAT PARQUET, COMPRESSION ZSTD)
            """,
            [float(cost_return), str(population_path), label_glob],
        )
        population_rows = int(
            con.execute(
                "SELECT count(*) FROM read_parquet(?) WHERE coalesce(selected_top40, false)",
                [str(population_path)],
            ).fetchone()[0]
        )
        rows = int(con.execute("SELECT count(*) FROM read_parquet(?)", [str(output)]).fetchone()[0])
        side_rows = {
            str(side): int(count)
            for side, count in con.execute(
                "SELECT side_name, count(*) FROM read_parquet(?) GROUP BY side_name ORDER BY side_name",
                [str(output)],
            ).fetchall()
        }
    finally:
        con.close()
    manifest = {
        "schema": "path_archetype_candidates_v3_base_top40",
        "population_source": str(population_path),
        "population_source_sha256": _sha256(population_path),
        "path_label_source": str(labels_path),
        "output": str(output),
        "output_sha256": _sha256(output),
        "population_rows": population_rows,
        "rows": rows,
        "coverage_vs_population": float(rows / population_rows) if population_rows else 0.0,
        "side_rows": side_rows,
        "cost_return": float(cost_return),
        "join_key": ["UTC epoch_ns", "__symbol__", "side_name"],
        "downstream_population_contract": (
            "same exact joined rows for alpha residual, five auxiliary heads, and CatBoost"
        ),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--population", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cost-return", type=float, default=0.01)
    args = parser.parse_args()
    print(
        json.dumps(
            materialize(
                args.population,
                args.labels,
                args.output_dir,
                cost_return=args.cost_return,
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
