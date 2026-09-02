#!/usr/bin/env python3
"""Attach the canonical source-aligned policy outcome contract to score rows.

This is intentionally a post-score materialisation step: none of the policy
outcome fields may take part in model inference.  It replaces the complete
policy contract from one authoritative parent materialisation, rather than
merging selected economics fields and retaining a stale validity mask.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds


POLICY_COLUMNS = (
    "policy_path_valid",
    "policy_gross_bps",
    "policy_net_bps",
    "policy_exit_bar_15m",
    "policy_entry_price",
    "policy_exit_price",
    "policy_exit_reason",
    "policy_label_available_ts",
    "policy_outcome_source",
    "policy_cost_bps",
)
ECONOMIC_COLUMNS = (
    "policy_gross_bps",
    "policy_net_bps",
    "policy_exit_bar_15m",
    "policy_entry_price",
    "policy_exit_price",
    "policy_cost_bps",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _score_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in frame.columns if column not in POLICY_COLUMNS]


def _load_policy_contract(
    path: Path, *, candidate_ids: Iterable[str] | None = None
) -> pd.DataFrame:
    required = ["candidate_id", *POLICY_COLUMNS]
    if candidate_ids is None:
        policy = pd.read_parquet(path, columns=required)
    else:
        identities = pd.Index(candidate_ids, dtype="string").dropna().unique().tolist()
        if not identities:
            return pd.DataFrame(columns=required)
        dataset = ds.dataset(path, format="parquet")
        policy = dataset.to_table(
            columns=required,
            filter=pc.field("candidate_id").isin(pa.array(identities, type=pa.string())),
        ).to_pandas()
    if policy["candidate_id"].duplicated().any():
        raise ValueError("canonical policy materialisation has duplicate candidate_id values")
    policy["policy_path_valid"] = policy["policy_path_valid"].fillna(False).astype(bool)
    policy["policy_label_available_ts"] = pd.to_datetime(
        policy["policy_label_available_ts"], utc=True, errors="coerce"
    )
    finite_net = np.isfinite(pd.to_numeric(policy["policy_net_bps"], errors="coerce"))
    valid = policy["policy_path_valid"].to_numpy(bool)
    if (valid & ~finite_net.to_numpy(bool)).any():
        raise ValueError("canonical policy marks a non-finite outcome valid")
    if (valid & policy["policy_label_available_ts"].isna().to_numpy(bool)).any():
        raise ValueError("canonical policy marks an outcome valid without availability time")
    return policy


def materialize_policy_contract(
    scores: pd.DataFrame, policy: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Replace every policy field with the authoritative source contract."""
    if "candidate_id" not in scores:
        raise ValueError("score ledger lacks candidate_id")
    if scores["candidate_id"].duplicated().any():
        raise ValueError("score ledger has duplicate candidate_id values")

    score_columns = _score_columns(scores)
    # A target-free route may retain its source-row index after filtering.  The
    # outcome merge naturally builds a fresh RangeIndex, so normalise only the
    # index before the exact value/identity parity assertion.
    score_only = scores.loc[:, score_columns].copy().reset_index(drop=True)
    output = score_only.merge(policy, on="candidate_id", how="left", validate="one_to_one", sort=False)
    if not output.loc[:, score_columns].equals(score_only):
        raise AssertionError("post-score policy join modified score or identity fields")

    source_found = output["policy_path_valid"].notna()
    output["policy_path_valid"] = (
        output["policy_path_valid"].astype("boolean").fillna(False).astype(bool)
    )
    valid = output["policy_path_valid"].to_numpy(bool)
    for column in ECONOMIC_COLUMNS:
        output.loc[~valid, column] = np.nan
    output.loc[~valid, "policy_label_available_ts"] = pd.NaT
    output.loc[~valid, "policy_exit_reason"] = "unavailable"
    output.loc[~valid, "policy_outcome_source"] = "unavailable"

    finite_net = np.isfinite(pd.to_numeric(output["policy_net_bps"], errors="coerce"))
    available = pd.to_datetime(output["policy_label_available_ts"], utc=True, errors="coerce").notna()
    if not np.array_equal(valid, (finite_net & available).to_numpy(bool)):
        raise AssertionError("policy validity must be equivalent to finite, available outcome")
    if "__decision_ts__" in output:
        decision = pd.to_datetime(output["__decision_ts__"], utc=True, errors="raise")
        if (pd.to_datetime(output.loc[valid, "policy_label_available_ts"], utc=True) <= decision.loc[valid]).any():
            raise AssertionError("policy label is available at or before its decision")

    audit = {
        "score_rows": int(len(scores)),
        "canonical_policy_rows": int(len(policy)),
        "candidate_ids_found_in_policy": int(source_found.sum()),
        "valid_policy_rows": int(valid.sum()),
        "invalid_or_missing_policy_rows": int((~valid).sum()),
    }
    return output, audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", required=True, type=Path)
    parser.add_argument("--canonical-policy", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()

    if args.out_dir.exists():
        raise FileExistsError(f"immutable output already exists: {args.out_dir}")
    args.out_dir.mkdir(parents=True)
    scores = pd.read_parquet(args.scores)
    policy = _load_policy_contract(args.canonical_policy)
    output, audit = materialize_policy_contract(scores, policy)
    output_path = args.out_dir / "current_v5_scored_label_ledger_canonical_policy.parquet"
    output.to_parquet(output_path, index=False)
    manifest = {
        "schema": "strict_r3_current_v5_canonical_policy_materialisation_v1",
        "purpose": "authoritative post-score policy-outcome replacement; invalid paths excluded from fitting and realised evaluation",
        "scores": str(args.scores),
        "scores_sha256": _sha256(args.scores),
        "canonical_policy": str(args.canonical_policy),
        "canonical_policy_sha256": _sha256(args.canonical_policy),
        "output": str(output_path),
        "output_sha256": _sha256(output_path),
        "policy_columns_replaced": list(POLICY_COLUMNS),
        "outcome_fields_not_used_for_scoring": True,
        "audit": audit,
    }
    (args.out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
