#!/usr/bin/env python3
"""Append resolved July base candidates to a compact meta residual handoff.

The source handoff is already the authorised pre-July top-30 base candidate
stream.  The external candidate frame contains frozen July base scores only;
this utility joins it to causal labels, derives the same execution outcomes as
the meta-handoff generator, and writes a compact, resolved handoff usable by
``run_meta_v9_ev_mapped_side_residual_ablation.py``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from extreme_price_movements.features_gmm_ae import AE_GMM_FEATURE_COLUMNS
from scripts.report_s52_trailing_regime_meta_handoff import _enrich_ledger


KEYS = ("__ts__", "__symbol__", "side_name")
HANDOFF_COLUMNS = (
    "__ts__",
    "__symbol__",
    "side_name",
    "__decision_ts__",
    "__first_path_ts__",
    "__label_path_end_ts__",
    "score",
    "base_margin_to_cutoff",
    "base_margin_to_cutoff_z",
    "base_signal_zscore_within_archetype",
    "archetype_label_family",
    "archetype_policy_key",
    *AE_GMM_FEATURE_COLUMNS,
)
LEDGER_COLUMNS = (
    "__ts__",
    "__symbol__",
    "side_name",
    "ev_after_1pct",
    "clean_exec",
    "dirty_positive",
    "full_path_bad_mae_1r",
    "timeout",
)


def _q(path: Path) -> str:
    return "'" + str(path.resolve()).replace("'", "''") + "'"


def _read_labels(labels_dir: Path) -> pd.DataFrame:
    parts = []
    for side in ("long", "short"):
        path = labels_dir / f"train_global_{side}_5_2026_07.parquet"
        if not path.is_file():
            raise FileNotFoundError(path)
        part = pd.read_parquet(path)
        part["side_name"] = part.get("side_name", side).astype(str).str.lower()
        part["__ts__"] = pd.to_datetime(part["__ts__"], utc=True, errors="raise")
        parts.append(part)
    labels = pd.concat(parts, ignore_index=True, copy=False)
    if labels.duplicated(list(KEYS)).any():
        raise ValueError("Causal July labels contain duplicate timestamp/symbol/side keys")
    return labels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-handoff", type=Path, required=True)
    parser.add_argument("--source-ledger", type=Path, required=True)
    parser.add_argument("--july-candidates", type=Path, required=True)
    parser.add_argument("--labels-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--end-exclusive", default="2026-07-21")
    args = parser.parse_args()

    end = pd.Timestamp(args.end_exclusive, tz="UTC")
    source_ts = duckdb.sql(
        f"SELECT max(__ts__) FROM read_parquet({_q(args.source_handoff)})"
    ).fetchone()[0]
    cutoff = pd.Timestamp(source_ts)
    cutoff = (
        cutoff.tz_localize("UTC")
        if cutoff.tzinfo is None
        else cutoff.tz_convert("UTC")
    )
    labels = _read_labels(args.labels_dir)
    candidates = pd.read_parquet(args.july_candidates)
    candidates["__ts__"] = pd.to_datetime(candidates["__ts__"], utc=True, errors="raise")
    candidates["side_name"] = candidates["side_name"].astype(str).str.lower()
    candidates = candidates.loc[
        candidates["__ts__"].gt(cutoff) & candidates["__ts__"].lt(end)
    ].copy()
    if candidates.duplicated(list(KEYS)).any():
        raise ValueError("July candidate scorer emitted duplicate timestamp/symbol/side keys")

    needed_candidate = {
        "score",
        "base_margin_to_cutoff",
        "base_margin_to_cutoff_z",
        "base_signal_zscore_within_archetype",
        "__archetype_policy_key__",
    }
    missing = sorted(needed_candidate - set(candidates.columns))
    if missing:
        raise ValueError(f"July candidates lack required meta anchors: {missing}")
    joined = candidates.merge(labels, on=list(KEYS), how="inner", validate="one_to_one", suffixes=("", "__label"))
    if joined.empty:
        raise ValueError("No July base candidates resolved against materialized labels")
    joined["month"] = joined["__ts__"].dt.strftime("%Y-%m")
    outcomes = _enrich_ledger(
        joined,
        embedded_round_trip_cost=0.003,
        executable_cost_floor=0.010,
    )
    for column in LEDGER_COLUMNS[3:]:
        if column not in outcomes or not pd.to_numeric(outcomes[column], errors="coerce").notna().all():
            raise ValueError(f"Cannot derive finite {column} for every July candidate")
    first_path = pd.to_datetime(outcomes["__first_path_ts__"], utc=True, errors="raise")
    decision = pd.to_datetime(outcomes["__decision_ts__"], utc=True, errors="raise")
    if (first_path < decision).any():
        raise ValueError("July label violates first_path_ts >= decision_ts")
    # This is the same resolution horizon used in the source handoff contract.
    outcomes["__label_path_end_ts__"] = first_path + pd.Timedelta(days=1)
    outcomes["archetype_policy_key"] = outcomes["__archetype_policy_key__"].astype(str)
    outcomes["archetype_label_family"] = outcomes[
        "__archetype_label_family__"
    ].astype(str)
    tail_handoff = outcomes.loc[:, list(HANDOFF_COLUMNS)].copy()
    tail_ledger = outcomes.loc[:, list(LEDGER_COLUMNS)].copy()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    source_handoff_out = args.output_dir / "train_meta_regime_handoff.parquet"
    source_ledger_out = args.output_dir / "s52_trailing_regime_scored_ledger.parquet"
    hcols = ", ".join(HANDOFF_COLUMNS)
    lcols = ", ".join(LEDGER_COLUMNS)
    source_handoff_compact = args.output_dir / "_source_handoff_compact.parquet"
    source_ledger_compact = args.output_dir / "_source_ledger_compact.parquet"
    duckdb.sql(
        f"COPY (SELECT {hcols} FROM read_parquet({_q(args.source_handoff)})) "
        f"TO {_q(source_handoff_compact)} (FORMAT PARQUET, COMPRESSION ZSTD)"
    )
    duckdb.sql(
        f"COPY (SELECT {lcols} FROM read_parquet({_q(args.source_ledger)})) "
        f"TO {_q(source_ledger_compact)} (FORMAT PARQUET, COMPRESSION ZSTD)"
    )
    tail_handoff_path = args.output_dir / "_tail_handoff.parquet"
    tail_ledger_path = args.output_dir / "_tail_ledger.parquet"
    tail_handoff.to_parquet(tail_handoff_path, index=False, compression="zstd")
    tail_ledger.to_parquet(tail_ledger_path, index=False, compression="zstd")
    duckdb.sql(
        f"COPY (SELECT * FROM read_parquet({_q(source_handoff_compact)}) "
        f"UNION ALL SELECT * FROM read_parquet({_q(tail_handoff_path)})) "
        f"TO {_q(source_handoff_out)} (FORMAT PARQUET, COMPRESSION ZSTD)"
    )
    duckdb.sql(
        f"COPY (SELECT * FROM read_parquet({_q(source_ledger_compact)}) "
        f"UNION ALL SELECT * FROM read_parquet({_q(tail_ledger_path)})) "
        f"TO {_q(source_ledger_out)} (FORMAT PARQUET, COMPRESSION ZSTD)"
    )
    for temporary in (source_handoff_compact, source_ledger_compact, tail_handoff_path, tail_ledger_path):
        temporary.unlink(missing_ok=True)

    handoff_rows = duckdb.sql(f"SELECT count(*) FROM read_parquet({_q(source_handoff_out)})").fetchone()[0]
    ledger_rows = duckdb.sql(f"SELECT count(*) FROM read_parquet({_q(source_ledger_out)})").fetchone()[0]
    duplicate_rows = duckdb.sql(
        f"SELECT count(*) FROM (SELECT {', '.join(KEYS)}, count(*) n FROM read_parquet({_q(source_handoff_out)}) GROUP BY {', '.join(KEYS)} HAVING n > 1)"
    ).fetchone()[0]
    if int(handoff_rows) != int(ledger_rows) or duplicate_rows:
        raise ValueError(f"Extended handoff integrity failed rows={handoff_rows}/{ledger_rows} duplicates={duplicate_rows}")
    manifest = {
        "schema": "compact_extended_meta_residual_handoff_v1",
        "source_handoff": str(args.source_handoff),
        "source_ledger": str(args.source_ledger),
        "july_candidates": str(args.july_candidates),
        "labels_dir": str(args.labels_dir),
        "source_cutoff_inclusive": cutoff.isoformat(),
        "tail_eval_end_exclusive": end.isoformat(),
        "tail_candidate_rows": int(len(candidates)),
        "tail_resolved_rows": int(len(outcomes)),
        "rows": int(handoff_rows),
        "columns": {"handoff": list(HANDOFF_COLUMNS), "ledger": list(LEDGER_COLUMNS)},
        "outcome_contract": "same _enrich_ledger economics; embedded_cost=0.003, executable_cost_floor=0.010",
        "label_resolution_contract": "__first_path_ts__ + 1 day, matching source handoff",
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
