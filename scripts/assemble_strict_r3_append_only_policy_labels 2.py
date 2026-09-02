#!/usr/bin/env python3
"""Assemble an append-only strict-R3 exact-policy outcome ledger.

Previously valid resolved labels are immutable. A later sparse market-data
refresh may add newly resolved candidates or upgrade a previously invalid path,
but it can never downgrade or rewrite an already valid economic outcome. The
target-free prediction population is authoritative for row identity; labels
cannot widen or shrink that population.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


LABEL_COLUMNS = (
    "candidate_id", "policy_path_valid", "policy_gross_bps", "policy_net_bps",
    "policy_exit_bar_15m", "policy_exit_reason", "policy_entry_price",
    "policy_exit_price", "policy_label_available_ts", "policy_cost_bps",
)
ECONOMIC_COLUMNS = LABEL_COLUMNS[1:]


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _unique(frame: pd.DataFrame, name: str) -> None:
    if frame["candidate_id"].isna().any() or frame["candidate_id"].duplicated().any():
        raise ValueError(f"{name} has null or duplicate candidate IDs")


def _equal_series(left: pd.Series, right: pd.Series) -> np.ndarray:
    if pd.api.types.is_numeric_dtype(left) and not pd.api.types.is_bool_dtype(left):
        return np.isclose(
            pd.to_numeric(left, errors="coerce").to_numpy(float),
            pd.to_numeric(right, errors="coerce").to_numpy(float),
            atol=1e-9, rtol=0.0, equal_nan=True,
        )
    return (
        left.astype(str).eq(right.astype(str)) | (left.isna() & right.isna())
    ).to_numpy(bool)


def assemble_append_only_labels(
    *, immutable_prefix: pd.DataFrame, refreshed_labels: pd.DataFrame,
    target_free_population: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object]]:
    prefix = immutable_prefix.loc[:, LABEL_COLUMNS].copy()
    refreshed = refreshed_labels.loc[:, LABEL_COLUMNS].copy()
    target = target_free_population.loc[:, ["candidate_id"]].copy()
    for frame, name in (
        (prefix, "immutable prefix"), (refreshed, "refreshed labels"),
        (target, "target-free population"),
    ):
        frame["candidate_id"] = frame["candidate_id"].astype(str)
        _unique(frame, name)

    target_ids = set(target["candidate_id"])
    prefix_ids = set(prefix["candidate_id"])
    refreshed_ids = set(refreshed["candidate_id"])
    if not prefix_ids.issubset(target_ids):
        raise ValueError("immutable prefix contains IDs outside target-free population")
    missing_refresh = target_ids.difference(refreshed_ids)
    extra_refresh = refreshed_ids.difference(target_ids)
    if missing_refresh or extra_refresh:
        raise ValueError(
            "refreshed labels must exactly cover target-free population: "
            f"missing={len(missing_refresh)}, extra={len(extra_refresh)}"
        )

    prefix_indexed = prefix.set_index("candidate_id")
    refreshed_indexed = refreshed.set_index("candidate_id")
    overlap_ids = sorted(prefix_ids.intersection(refreshed_ids))
    old_overlap = prefix_indexed.loc[overlap_ids]
    new_overlap = refreshed_indexed.loc[overlap_ids]
    old_valid = old_overlap["policy_path_valid"].fillna(False).astype(bool)
    new_valid = new_overlap["policy_path_valid"].fillna(False).astype(bool)
    both_valid = old_valid & new_valid
    for field in ECONOMIC_COLUMNS:
        if both_valid.any() and not _equal_series(
            old_overlap.loc[both_valid, field], new_overlap.loc[both_valid, field],
        ).all():
            raise ValueError(f"refreshed valid label conflicts with immutable {field}")

    # Start from the complete refreshed population, then restore immutable valid
    # rows and allow only invalid-to-valid upgrades on old identities.
    output = refreshed_indexed.copy()
    immutable_valid_ids = old_overlap.index[old_valid]
    output.loc[immutable_valid_ids, list(ECONOMIC_COLUMNS)] = old_overlap.loc[
        immutable_valid_ids, list(ECONOMIC_COLUMNS)
    ]
    still_invalid_ids = old_overlap.index[~old_valid & ~new_valid]
    output.loc[still_invalid_ids, list(ECONOMIC_COLUMNS)] = old_overlap.loc[
        still_invalid_ids, list(ECONOMIC_COLUMNS)
    ]
    output = output.reindex(target["candidate_id"]).reset_index()
    _unique(output, "append-only output")

    valid = output["policy_path_valid"].fillna(False).astype(bool)
    finite = np.isfinite(pd.to_numeric(output["policy_net_bps"], errors="coerce"))
    if not finite.loc[valid].all():
        raise ValueError("valid append-only labels contain non-finite net outcomes")
    if valid.any() and not np.allclose(
        output.loc[valid, "policy_net_bps"].to_numpy(float),
        output.loc[valid, "policy_gross_bps"].to_numpy(float) - 100.0,
        atol=1e-9, rtol=0.0,
    ):
        raise ValueError("append-only labels do not subtract 100 bps exactly once")

    audit = {
        "rows": int(len(output)),
        "valid_rows": int(valid.sum()),
        "immutable_prefix_rows": int(len(prefix)),
        "immutable_valid_rows": int(prefix["policy_path_valid"].fillna(False).sum()),
        "new_identity_rows": int(len(target_ids.difference(prefix_ids))),
        "invalid_to_valid_upgrades": int((~old_valid & new_valid).sum()),
        "valid_to_invalid_downgrades_prevented": int((old_valid & ~new_valid).sum()),
        "overlapping_valid_rows_verified_identical": int(both_valid.sum()),
        "immutable_valid_rows_preserved": True,
        "target_free_identity_exact": True,
        "cost_bps_once": 100.0,
    }
    return output, audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--immutable-prefix", type=Path, required=True)
    parser.add_argument("--refreshed-labels", type=Path, required=True)
    parser.add_argument("--target-free-population", type=Path, required=True)
    parser.add_argument("--policy-json", type=Path, required=True)
    parser.add_argument("--start")
    parser.add_argument("--end-exclusive")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(f"immutable output exists: {args.out_dir}")
    prefix = pd.read_parquet(args.immutable_prefix)
    refreshed = pd.read_parquet(args.refreshed_labels)
    target_columns = ["candidate_id"]
    if args.start or args.end_exclusive:
        target_columns.append("__decision_ts__")
    target = pd.read_parquet(args.target_free_population, columns=target_columns)
    if bool(args.start) != bool(args.end_exclusive):
        raise ValueError("start and end-exclusive must be supplied together")
    if args.start:
        target["__decision_ts__"] = pd.to_datetime(
            target["__decision_ts__"], utc=True, errors="raise",
        )
        start = pd.Timestamp(args.start)
        end = pd.Timestamp(args.end_exclusive)
        start = start.tz_localize("UTC") if start.tzinfo is None else start.tz_convert("UTC")
        end = end.tz_localize("UTC") if end.tzinfo is None else end.tz_convert("UTC")
        if end <= start:
            raise ValueError("end-exclusive must be after start")
        target = target.loc[
            target["__decision_ts__"].ge(start)
            & target["__decision_ts__"].lt(end),
            ["candidate_id"],
        ].copy()
    output, audit = assemble_append_only_labels(
        immutable_prefix=prefix,
        refreshed_labels=refreshed,
        target_free_population=target,
    )
    args.out_dir.mkdir(parents=True)
    output_path = args.out_dir / "frozen_policy_labels.parquet"
    output.to_parquet(output_path, index=False, compression="zstd")
    manifest = {
        "schema": "strict_r3_append_only_frozen_policy_labels_v1",
        **audit,
        "policy_json": str(args.policy_json),
        "policy_json_sha256": _sha(args.policy_json),
        "evaluation_start": args.start,
        "evaluation_end_exclusive": args.end_exclusive,
        "sources": {
            "immutable_prefix": {"path": str(args.immutable_prefix), "sha256": _sha(args.immutable_prefix)},
            "refreshed_labels": {"path": str(args.refreshed_labels), "sha256": _sha(args.refreshed_labels)},
            "target_free_population": {"path": str(args.target_free_population), "sha256": _sha(args.target_free_population)},
        },
        "output": str(output_path),
        "output_sha256": _sha(output_path),
    }
    (args.out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"event": "complete", **manifest}))


if __name__ == "__main__":
    main()
