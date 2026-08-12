#!/usr/bin/env python3
"""Fail-closed lineage audit for a strict-R3 monthly prequential ledger."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _utc(value: pd.Series) -> pd.Series:
    return pd.to_datetime(value, utc=True, errors="raise")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--fold-audit", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    ledger = pd.read_parquet(args.ledger)
    audit = pd.read_parquet(args.fold_audit)
    required = {
        "candidate_id", "__decision_ts__", "held_month", "side_name",
        "stack_is_prequential", "policy_label_available_ts", "r3_label_available_ts",
        "prequential_base_rank42", "prequential_base_anchor_bps",
        "prequential_consensus_rank", "prequential_residual_rank", "prequential_upstream",
    }
    missing = sorted(required.difference(ledger.columns))
    checks: dict[str, object] = {"rows": int(len(ledger)), "missing_columns": missing}
    if missing:
        raise ValueError(f"ledger lacks required columns: {missing}")
    decision = _utc(ledger["__decision_ts__"])
    policy_available = _utc(ledger["policy_label_available_ts"])
    r3_available = _utc(ledger["r3_label_available_ts"])
    expected_month = decision.dt.strftime("%Y-%m")
    checks["candidate_ids_unique"] = bool(not ledger["candidate_id"].duplicated().any())
    checks["held_month_matches_decision_month"] = bool(ledger["held_month"].astype(str).eq(expected_month).all())
    checks["all_rows_flagged_prequential"] = bool(ledger["stack_is_prequential"].fillna(False).astype(bool).all())
    checks["all_policy_labels_resolve_after_decision"] = bool(policy_available.ge(decision).all())
    checks["all_r3_labels_resolve_after_decision"] = bool(r3_available.ge(decision).all())
    finite_columns = [
        "prequential_base_rank42", "prequential_base_anchor_bps",
        "prequential_consensus_rank", "prequential_residual_rank", "prequential_upstream",
    ]
    checks["complete_stack_outputs"] = bool(all(
        np.isfinite(pd.to_numeric(ledger[column], errors="coerce")).all()
        for column in finite_columns
    ))
    checks["single_side"] = int(ledger["side_name"].astype(str).str.lower().nunique())
    base = audit.loc[audit["pass"].eq("base")].copy()
    downstream = audit.loc[audit["pass"].eq("map_residual")].copy()
    checks["base_completed_folds"] = int(base["status"].eq("complete").sum())
    checks["downstream_completed_folds"] = int(downstream["status"].eq("complete").sum())
    checks["downstream_has_no_held_outcomes_consumed"] = bool(
        downstream.loc[downstream["status"].eq("complete"), "held_outcomes_consumed"].fillna(True).eq(False).all()
        if "held_outcomes_consumed" in downstream else False
    )
    checks["all_base_references_same_model"] = bool(
        base.loc[base["status"].eq("complete"), "same_model_reference"].fillna(False).astype(bool).all()
        if "same_model_reference" in base else False
    )
    if "reference_start" in base and "reference_end_exclusive" in base:
        start = _utc(base.loc[base["status"].eq("complete"), "reference_start"])
        end = _utc(base.loc[base["status"].eq("complete"), "reference_end_exclusive"])
        checks["all_base_references_exactly_42_days"] = bool((end - start).eq(pd.Timedelta(days=42)).all())
    else:
        checks["all_base_references_exactly_42_days"] = False
    checks["month_rows"] = {
        month: int(rows) for month, rows in ledger.groupby("held_month", sort=True).size().items()
    }
    failed = [name for name, value in checks.items() if isinstance(value, bool) and not value]
    checks["passed"] = not failed
    checks["failed_checks"] = failed
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(checks, indent=2, default=str) + "\n")
    print(json.dumps(checks, default=str))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
