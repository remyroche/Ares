#!/usr/bin/env python3
"""Materialise a target-free B0 router control on a frozen candidate universe.

This is a small, immutable adapter for the economic-router downstream test.
It deliberately contains no policy, path, outcome, or label-availability
columns.  It projects the sealed strict-OOF B0 score onto the exact candidate
identities of a supplied frozen 120-field source, one score panel per month.

The output has the same minimal score contract consumed by
``run_strict_r3_router_downstream.py``:

    candidate_id, __decision_ts__, side_name, router_primary_rank

Despite the compatibility name, ``router_primary_rank`` is the unmodified
strict-OOF B0 score.  The downstream runner forms timestamp-local ranks from
that score itself, before any consensus or outcome join.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "strict_r3_b0_router_control_v1"
IDENTITY = ("candidate_id", "__decision_ts__", "side_name")
SCORE = "base_score"
PROHIBITED = frozenset({
    "policy_path_valid", "policy_gross_bps", "policy_net_bps", "policy_exit_bar_15m",
    "policy_entry_price", "policy_exit_price", "policy_exit_reason", "policy_label_available_ts",
    "policy_cost_bps", "semantic_path_valid", "semantic_sequence", "semantic_speed_bin",
    "semantic_persistence_bin", "semantic_pre_adverse_bin", "semantic_policy_conversion_bin",
    "semantic_exit_reason", "semantic_composite", "semantic_tbm_event",
})


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json_exclusive(path: Path, payload: object) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)


def _tokens(raw: str) -> tuple[str, ...]:
    result = tuple(token.strip() for token in raw.split(",") if token.strip())
    if not result:
        raise ValueError("at least one month is required")
    return result


def run(*, source_root: Path, b0_control: Path, out: Path, months: tuple[str, ...]) -> None:
    if out.exists():
        raise FileExistsError(out)
    schema = pq.ParquetFile(b0_control).schema_arrow.names
    leaked = sorted(PROHIBITED.intersection(schema))
    if leaked:
        raise AssertionError(f"B0 control contains prohibited target/outcome columns: {leaked}")
    required = set((*IDENTITY, SCORE))
    missing = sorted(required - set(schema))
    if missing:
        raise AssertionError(f"B0 control is missing columns: {missing}")
    b0 = pd.read_parquet(b0_control, columns=[*IDENTITY, SCORE])
    b0["__decision_ts__"] = pd.to_datetime(b0["__decision_ts__"], utc=True, errors="raise")
    if b0["candidate_id"].duplicated().any():
        raise AssertionError("B0 control contains duplicate candidate IDs")
    out.mkdir(parents=True)
    target = out / "target_free_scores"
    target.mkdir()
    audits: list[dict[str, object]] = []
    for token in months:
        path = source_root / "target_free_monthly" / f"month={token}" / "scores_features.parquet"
        if not path.exists():
            raise FileNotFoundError(path)
        source_schema = pq.ParquetFile(path).schema_arrow.names
        leaked = sorted(PROHIBITED.intersection(source_schema))
        if leaked:
            raise AssertionError(f"{token}: source receipt contains prohibited columns: {leaked}")
        source = pd.read_parquet(path, columns=list(IDENTITY))
        source["__decision_ts__"] = pd.to_datetime(source["__decision_ts__"], utc=True, errors="raise")
        if source["candidate_id"].duplicated().any():
            raise AssertionError(f"{token}: source has duplicate candidate IDs")
        expected = source.loc[:, list(IDENTITY)].merge(
            b0, on=list(IDENTITY), how="left", validate="one_to_one", indicator=True,
        )
        if not expected["_merge"].eq("both").all() or expected[SCORE].isna().any():
            bad = expected.loc[expected["_merge"].ne("both") | expected[SCORE].isna(), "candidate_id"].head(5).tolist()
            raise AssertionError(f"{token}: B0 does not cover source identity; examples={bad}")
        result = expected.loc[:, [*IDENTITY, SCORE]].rename(columns={SCORE: "router_primary_rank"})
        result.to_parquet(target / f"month={token}.parquet", index=False, compression="zstd")
        audits.append({
            "month": token, "source_rows": int(len(source)), "scored_rows": int(len(result)),
            "identity_exact": True, "non_null_score_fraction": float(result["router_primary_rank"].notna().mean()),
        })
    audit = pd.DataFrame(audits)
    if audit["scored_rows"].ne(audit["source_rows"]).any() or audit["non_null_score_fraction"].lt(1.0).any():
        raise AssertionError("B0 control did not exactly cover the frozen source population")
    audit.to_parquet(out / "coverage_audit.parquet", index=False, compression="zstd")
    contract = {
        "schema": SCHEMA,
        "scope": "offline research only; target-free B0 score projection, no labels or live changes",
        "score": "sealed strict-OOF B0 base_score; downstream creates timestamp-local ranks",
        "source_root": str(source_root), "b0_control": str(b0_control), "months": list(months),
        "identity": list(IDENTITY), "prohibited_columns": sorted(PROHIBITED),
    }
    _write_json_exclusive(out / "run_contract.json", contract)
    _write_json_exclusive(out / "run_manifest.json", {
        "schema": SCHEMA, "status": "complete", "contract": contract,
        "source_hashes": {"b0_control": _sha256(b0_control)},
        "coverage": audit.to_dict(orient="records"),
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--b0-control", type=Path, default=ROOT / "data_perp/artifacts/strict_r3_long_base_recall_funnel_2025dev_holdout_2026oos_20260822_v1/b0_target_free_reconstruction.parquet")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--months", required=True, help="comma-separated YYYY-MM months")
    args = parser.parse_args()
    run(source_root=args.source_root, b0_control=args.b0_control, out=args.out, months=_tokens(args.months))


if __name__ == "__main__":
    main()
