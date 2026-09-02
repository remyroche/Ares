#!/usr/bin/env python3
"""Materialize a causal candidate-stream parent for residual-episode research.

This creates a new research baseline from the frozen candidate shards.  It is
not a V9 backfill: each UTC day is ranked only against the previous eight
days' raw candidate scores.  The resulting parent can be joined with one or
more separately frozen OOS residual-state artifacts, including an early-March
state fit trained only through February.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.materialize_complete_meta_residual_parent import (
    KEYS,
    OUTCOMES,
    _causal_trailing_day_ranks,
    _read_prediction_shards,
)


def _utc(value: str) -> pd.Timestamp:
    return pd.Timestamp(value, tz="UTC")


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _finalize_existing(
    candidate_root: Path,
    output_dir: Path,
    *,
    start: str,
    end: str,
    lookback_days: int,
    min_reference_rows: int,
    recovered: bool,
) -> dict[str, object]:
    """Seal an already-written parent after an interrupted reporting phase."""

    parent_path = output_dir / "causal_parent_predictions.parquet"
    cache_path = output_dir / "causal_train_rank_cache.parquet"
    if not parent_path.exists() or not cache_path.exists():
        raise FileNotFoundError("Both causal parent and rank cache are required to finalize")
    parent = pd.read_parquet(parent_path, columns=["__ts__", "historical_rank"])
    parent["__ts__"] = pd.to_datetime(parent["__ts__"], utc=True, errors="coerce")
    coverage = float(parent["historical_rank"].notna().mean())
    month_coverage = (
        parent.assign(month=parent["__ts__"].dt.strftime("%Y-%m"))
        .groupby("month", observed=True)
        .agg(
            rows=("historical_rank", "size"),
            rank_coverage=("historical_rank", lambda values: float(values.notna().mean())),
            rank_p10=("historical_rank", lambda values: float(values.quantile(0.10))),
            rank_p90=("historical_rank", lambda values: float(values.quantile(0.90))),
        )
        .reset_index()
    )
    month_coverage.to_csv(output_dir / "rank_coverage_by_month.csv", index=False)
    manifest: dict[str, object] = {
        "schema": "causal_residual_episode_parent_v1",
        "candidate_root": str(candidate_root),
        "start": _utc(start).isoformat(),
        "end_exclusive": _utc(end).isoformat(),
        "rows": int(len(parent)),
        "causal_rank_contract": (
            "global trailing prior-day empirical CDF; current UTC day excluded; "
            f"lookback_days={int(lookback_days)}"
        ),
        "rank_coverage": coverage,
        "rank_coverage_by_month": month_coverage.to_dict("records"),
        "min_reference_rows": int(min_reference_rows),
        "parent_path": str(parent_path),
        "rank_cache_path": str(cache_path),
        "production_status": "research_only_new_parent_contract",
        "finalized_after_interrupted_reporting": bool(recovered),
    }
    _write_json(output_dir / "manifest.json", manifest)
    return manifest


def materialize(
    candidate_root: Path,
    output_dir: Path,
    *,
    start: str,
    end: str,
    lookback_days: int,
    min_reference_rows: int,
) -> dict[str, object]:
    """Write a rank cache and thin parent ledger under one causal contract."""

    start_ts = _utc(start)
    end_ts = _utc(end)
    if end_ts <= start_ts:
        raise ValueError("end must be after start")
    source = _read_prediction_shards(candidate_root)
    source["__ts__"] = pd.to_datetime(source["__ts__"], utc=True, errors="coerce")
    source = source.loc[source["__ts__"].notna() & source["__ts__"].lt(end_ts)]
    prior = source.loc[source["__ts__"].lt(start_ts)].copy()
    evaluation = source.loc[source["__ts__"].ge(start_ts)].copy().reset_index(drop=True)
    if evaluation.empty:
        raise ValueError("candidate stream has no rows in the requested evaluation range")
    ranks, rank_manifest = _causal_trailing_day_ranks(
        prior,
        evaluation,
        lookback_days=int(lookback_days),
        min_reference_rows=int(min_reference_rows),
    )
    evaluation["historical_rank"] = ranks
    evaluation["historical_rank_strict_extreme_local"] = ranks
    evaluation["hit_probability"] = pd.to_numeric(
        evaluation["score_meta_base_soft_label"], errors="coerce"
    ).astype(np.float32)
    coverage = float(np.isfinite(ranks).mean())
    if coverage < 0.99:
        raise ValueError(
            f"causal rank coverage is incomplete: {coverage:.2%}; "
            "provide sufficient strictly-prior score history"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    cache = evaluation.loc[:, [*KEYS, "historical_rank"]]
    parent = evaluation.loc[
        :, [
            *KEYS,
            "historical_rank",
            "historical_rank_strict_extreme_local",
            "hit_probability",
            *OUTCOMES,
        ]
    ]
    cache.to_parquet(output_dir / "causal_train_rank_cache.parquet", index=False, compression="zstd")
    parent.to_parquet(output_dir / "causal_parent_predictions.parquet", index=False, compression="zstd")
    # The source frame carries every candidate shard field. Release it before
    # building diagnostics so reporting cannot double the peak memory of an
    # otherwise completed parent materialization.
    del source, prior, evaluation, ranks, cache
    gc.collect()
    manifest = _finalize_existing(
        candidate_root,
        output_dir,
        start=start,
        end=end,
        lookback_days=lookback_days,
        min_reference_rows=min_reference_rows,
        recovered=False,
    )
    manifest["rank_folds"] = rank_manifest
    _write_json(output_dir / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start", default="2025-03-01")
    parser.add_argument("--end", default="2026-07-01")
    parser.add_argument("--lookback-days", type=int, default=8)
    parser.add_argument("--min-reference-rows", type=int, default=2_000)
    parser.add_argument(
        "--finalize-only",
        action="store_true",
        help="Write coverage/manifest for an already materialized parent and rank cache.",
    )
    args = parser.parse_args()
    if args.finalize_only:
        print(
            json.dumps(
                _finalize_existing(
                    args.candidate_root,
                    args.output_dir,
                    start=args.start,
                    end=args.end,
                    lookback_days=args.lookback_days,
                    min_reference_rows=args.min_reference_rows,
                    recovered=True,
                ),
                indent=2,
            )
        )
        return
    print(
        json.dumps(
            materialize(
                args.candidate_root,
                args.output_dir,
                start=args.start,
                end=args.end,
                lookback_days=args.lookback_days,
                min_reference_rows=args.min_reference_rows,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
