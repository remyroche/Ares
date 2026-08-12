#!/usr/bin/env python3
"""Recover an MDA feature contract after a post-selection metadata failure.

This is deliberately narrow: it never fits a model, recomputes MDA, or edits
checkpoint results.  It derives the exact accepted field set from the persisted
backward-elimination path and reproduces the runner's documented proposal
selection from the immutable MDA summaries.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.run_strict_r3_k9weighted_mda as runner  # noqa: E402
import scripts.run_strict_r3_n5_canonical_selection as selection  # noqa: E402


def _proposed(detail: pd.DataFrame, group_detail: pd.DataFrame, folds: int) -> list[str]:
    config = selection._feature_group_config()
    group_summary = selection._portable_summary(
        group_detail.loc[group_detail["environment_kind"].eq("fold")],
        ("group",), int(folds), config,
    )
    minimum = float(config["portability"]["minimum_positive_environment_fraction"])
    passing = set(group_summary.loc[
        group_summary["mda_median"].gt(0.0)
        & group_summary["positive_fold_recurrence"].ge(minimum), "group"
    ].astype(str))
    if not passing:
        passing = set(group_summary.head(2)["group"].astype(str))
    summary = detail.drop_duplicates("field").loc[:, [
        "field", "family", "group", "mda_median", "mda_mad", "mda_worst_fold",
        "positive_fold_recurrence", "portable_mda_score",
    ]].sort_values("portable_mda_score", ascending=False, kind="stable")
    selected: list[str] = []
    for group in sorted(passing):
        block = summary.loc[
            summary["group"].eq(group)
            & summary["positive_fold_recurrence"].ge(minimum)
            & summary["mda_median"].ge(0.0)
        ].head(5)
        if block.empty:
            block = summary.loc[summary["group"].eq(group)].head(1)
        selected.extend(block["field"].astype(str).tolist())
    selected = list(dict.fromkeys(selected))[:40]
    if len(selected) < 12:
        selected = summary.head(min(20, len(summary)))["field"].astype(str).tolist()
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--surface", type=Path, required=True)
    parser.add_argument("--partial-dir", type=Path, required=True)
    parser.add_argument(
        "--out", type=Path,
        help="Write a separately versioned recovered contract; preserves an earlier recovery.",
    )
    args = parser.parse_args()
    root = args.partial_dir
    required = [
        root / "portable_mda_detail.parquet", root / "portable_group_mda_detail.parquet",
        root / "backward_elimination_path.parquet",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"recovery requires completed selection artifacts: {missing}")
    contract_path = args.out or (root / "mda_feature_contract.json")
    if contract_path.exists():
        raise FileExistsError(f"contract already exists: {contract_path}")
    admitted, admission = runner._load_admitted_surface(args.surface)
    # The persisted detail is authoritative for the original run's eligible
    # universe.  Re-running today's coverage gate can incorrectly admit fields
    # whose coverage changed after the original fold construction.
    detail = pd.read_parquet(root / "portable_mda_detail.parquet")
    fields = detail.drop_duplicates("field")["field"].astype(str).tolist()
    current = set(runner._fields(admitted))
    missing_from_surface = sorted(set(fields).difference(current))
    if missing_from_surface:
        raise AssertionError(
            "persisted MDA fields are absent from the current recovered surface: "
            f"{missing_from_surface[:10]}"
        )
    if admitted["geometry_bundle_sha256"].nunique() != 1:
        raise AssertionError("recovered MDA contract requires one frozen geometry bundle")
    path = pd.read_parquet(root / "backward_elimination_path.parquet")
    if path.empty or not path.iloc[0]["stage"] == "full":
        raise AssertionError("backward elimination path lacks its full-contract baseline")
    active = list(fields)
    for _, row in path.iloc[1:].iterrows():
        if bool(row.get("accepted", False)):
            removed = list(row.get("removed", []))
            active = [field for field in active if field not in set(map(str, removed))]
    group_detail = pd.read_parquet(root / "portable_group_mda_detail.parquet")
    proposed = _proposed(detail, group_detail, folds=int(detail["fold"].nunique()))
    payload = {
        "schema": "strict_r3_schema_v2_additive_k9weighted_mda_v3_neutral_all_fields",
        "fields": fields, "field_count": len(fields),
        "mda_proposed_fields": proposed, "mda_proposed_field_count": len(proposed),
        "compact_fields": active, "compact_field_count": len(active),
        "selection_rule": "all existing and newly-derived fields are equal MDA and retrained elimination candidates; no protected feature tier",
        "frozen_geometry_bundle_sha256": str(admitted["geometry_bundle_sha256"].iloc[0]),
        "history_rule": "cluster correctness/residuals use only resolved labels strictly before decision timestamp",
        "raw_k9_memberships_used": False,
        "folds": sorted(detail["cutoff"].astype(str).unique().tolist()),
        "admission_rows": int(len(admission)),
        "recovery": "metadata-only reconstruction from completed immutable MDA and backward-elimination artifacts; no fitting or score recomputation",
    }
    contract_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"event": "recovered", "compact_fields": len(active), "fields": len(fields)}))


if __name__ == "__main__":
    main()
