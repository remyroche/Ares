#!/usr/bin/env python3
"""Materialise row-level inputs for the frozen structural-family specialists.

The coverage audit freezes family IDs and medoid feature digests.  This utility
turns the complete structural contribution stream into one causal row per
fold/candidate with signed family contribution shares, absolute contribution
shares, activity flags, selected mass and unassigned mass.  DuckDB performs the
large aggregation directly over Parquet so the 36-million-row contribution
stream is never loaded into a Python DataFrame.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import duckdb
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "data_perp/artifacts/long_structural_tree_meta_sidecar_20260804_v4"
DEFAULT_AUDIT = ROOT / "data_perp/artifacts/frozen_family_coverage_audit_20260808_v1"
DEFAULT_OUT = ROOT / "data_perp/artifacts/frozen_family_inputs_20260808_v1"
DEFAULT_THRESHOLD = 0.60
DEFAULT_TOP_N = 64


def _digest(values: list[str]) -> str:
    return hashlib.sha256("\n".join(values).encode()).hexdigest()


def _sql_path(path: Path) -> str:
    return str(path).replace("'", "''")


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


def _family_columns(top_n: int) -> tuple[list[str], list[str]]:
    signed = [f"sf__{i:03d}__signed_share" for i in range(1, top_n + 1)]
    absolute = [f"sf__{i:03d}__abs_share" for i in range(1, top_n + 1)]
    active = [f"sf__{i:03d}__active" for i in range(1, top_n + 1)]
    return signed, absolute + active


def run(args: argparse.Namespace) -> Path:
    source = Path(args.source)
    audit = Path(args.audit)
    out = Path(args.out)
    threshold = float(args.threshold)
    top_n = int(args.top_n)
    if out.exists() and any(out.iterdir()) and not args.resume:
        raise FileExistsError(f"refusing to overwrite populated output: {out}")
    out.mkdir(parents=True, exist_ok=True)

    mapping_path = audit / "frozen_family_rule_mapping.parquet"
    summary_path = audit / "frozen_family_superfamily_summary.parquet"
    if not mapping_path.exists() or not summary_path.exists():
        raise FileNotFoundError("coverage audit mapping/summary is required")

    summary = pd.read_parquet(summary_path)
    selected_summary = summary[
        summary.threshold.eq(threshold) & summary.development_mass_rank.le(top_n)
    ].sort_values("development_mass_rank")
    if len(selected_summary) != top_n:
        raise ValueError(
            f"requested top_n={top_n} but only {len(selected_summary)} families exist at threshold {threshold}"
        )
    if selected_summary.development_fold_count.lt(2).any():
        raise ValueError("selected family lacks both development folds")
    digests = selected_summary.frozen_feature_digest.astype(str).tolist()
    if len(set(digests)) != len(digests):
        raise ValueError("selected family feature digests are not unique")

    signed_names = [f"sf__{i:03d}__signed_share" for i in range(1, top_n + 1)]
    abs_names = [f"sf__{i:03d}__abs_share" for i in range(1, top_n + 1)]
    active_names = [f"sf__{i:03d}__active" for i in range(1, top_n + 1)]
    select_fields = [
        "p.fold_id",
        "p.candidate_id",
        "p.__ts__",
        "p.side_name",
        "p.meta_partition",
        "COALESCE(g.family_total_abs_contribution, 0.0) AS family_total_abs_contribution",
        "COALESCE(g.family_selected_abs_contribution, 0.0) AS family_selected_abs_contribution",
    ]
    for i, name in enumerate(signed_names, 1):
        select_fields.append(
            f"CASE WHEN g.family_total_abs_contribution > 0 THEN COALESCE(g.sf_{i:03d}_signed, 0.0) / g.family_total_abs_contribution ELSE 0.0 END AS {name}"
        )
    for i, name in enumerate(abs_names, 1):
        select_fields.append(
            f"CASE WHEN g.family_total_abs_contribution > 0 THEN COALESCE(g.sf_{i:03d}_abs, 0.0) / g.family_total_abs_contribution ELSE 0.0 END AS {name}"
        )
    for i, name in enumerate(active_names, 1):
        select_fields.append(f"CAST(CASE WHEN COALESCE(g.sf_{i:03d}_abs, 0.0) > 1e-12 THEN 1 ELSE 0 END AS UTINYINT) AS {name}")

    rank_cases_signed = []
    rank_cases_abs = []
    for i in range(1, top_n + 1):
        rank_cases_signed.append(
            f"SUM(CASE WHEN m.assigned_to_frozen_contract AND m.nearest_mass_rank = {i} THEN c.contribution ELSE 0.0 END) AS sf_{i:03d}_signed"
        )
        rank_cases_abs.append(
            f"SUM(CASE WHEN m.assigned_to_frozen_contract AND m.nearest_mass_rank = {i} THEN ABS(c.contribution) ELSE 0.0 END) AS sf_{i:03d}_abs"
        )
    query = f"""
    COPY (
      WITH partitions AS (
        SELECT candidate_id, CAST(fold AS VARCHAR) AS fold_id, __ts__, side_name, meta_partition
        FROM read_parquet('{_sql_path(source / 'fold_evaluations' / '*.parquet')}')
      ),
      contributions AS (
        SELECT candidate_id, CAST(fold_id AS VARCHAR) AS fold_id,
               fold_id || '::' || rule_signature AS rule_key,
               CAST(family_ensemble_tree_contribution AS DOUBLE) AS contribution
        FROM read_parquet('{_sql_path(source / 'family_contributions' / '*.parquet')}')
      ),
      frozen_mapping AS (
        SELECT rule_key, nearest_mass_rank, assigned_to_frozen_contract
        FROM read_parquet('{_sql_path(mapping_path)}')
        WHERE threshold = {threshold}
      ),
      grouped AS (
        SELECT c.fold_id, c.candidate_id,
               SUM(ABS(c.contribution)) AS family_total_abs_contribution,
               SUM(CASE WHEN m.assigned_to_frozen_contract AND m.nearest_mass_rank <= {top_n} THEN ABS(c.contribution) ELSE 0.0 END) AS family_selected_abs_contribution,
               {', '.join(rank_cases_signed)},
               {', '.join(rank_cases_abs)}
        FROM contributions c
        LEFT JOIN frozen_mapping m USING (rule_key)
        GROUP BY c.fold_id, c.candidate_id
      )
      SELECT {', '.join(select_fields)}
      FROM partitions p
      LEFT JOIN grouped g USING (fold_id, candidate_id)
    ) TO '{_sql_path(out / 'frozen_family_inputs.parquet')}' (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)
    """
    con = duckdb.connect()
    con.execute("PRAGMA threads=4")
    con.execute(query)
    con.close()

    inputs = pd.read_parquet(out / "frozen_family_inputs.parquet")
    if inputs.duplicated(["fold_id", "candidate_id"]).any():
        raise ValueError("duplicate fold/candidate output rows")
    if len(inputs) != sum(
        len(pd.read_parquet(source / "fold_evaluations" / f"{fold}.parquet", columns=["candidate_id"]))
        for fold in ("oof_jul_aug", "oof_may_jun", "oos_sep_nov")
    ):
        raise ValueError("row count does not match fold-evaluation population")
    if not inputs[signed_names + abs_names + active_names].apply(pd.to_numeric, errors="coerce").notna().all().all():
        raise ValueError("non-finite family input values")

    feature_contract = {
        "schema": "frozen_family_specialist_input_contract_v1",
        "source_coverage_audit": str(audit),
        "threshold": threshold,
        "top_n": top_n,
        "family_ids": selected_summary.superfamily_id.astype(str).tolist(),
        "family_feature_digests": digests,
        "signed_columns": signed_names,
        "absolute_share_columns": abs_names,
        "active_columns": active_names,
        "feature_contract_sha256": _digest(digests + signed_names + abs_names + active_names),
        "outcome_columns_not_used_as_inputs": ["gross_bps", "net_bps", "label_valid", "barrier_relevance_0_5", "mfe_mae_label_valid"],
    }
    _write_json(out / "frozen_specialist_input_contract.json", feature_contract)
    checks = {
        "status": "passed",
        "row_identity_unique": True,
        "population_row_count_matches": True,
        "family_contract_fixed": True,
        "selected_rank_le_top_n": True,
        "outcome_columns_absent_from_family_inputs": not any(c in inputs.columns for c in ["gross_bps", "net_bps", "label_valid"]),
        "finite_family_inputs": True,
    }
    if not checks["outcome_columns_absent_from_family_inputs"]:
        checks["status"] = "failed"
    _write_json(out / "correctness_test_report.json", checks)
    _write_json(
        out / "run_manifest.json",
        {
            "schema": "materialize_frozen_family_inputs_v1",
            "status": checks["status"],
            "source": str(source),
            "coverage_audit": str(audit),
            "threshold": threshold,
            "top_n": top_n,
            "rows": len(inputs),
            "columns": len(inputs.columns),
            "feature_contract_sha256": feature_contract["feature_contract_sha256"],
            "aggregation": "DuckDB batch-parquet contribution aggregation; one row per fold/candidate",
        },
    )
    return out


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    p.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    p.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
    p.add_argument("--resume", action="store_true")
    return p


if __name__ == "__main__":
    run(_parser().parse_args())
