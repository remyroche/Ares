#!/usr/bin/env python3
"""Assemble the long-only schema-v2 feature/label source panel.

Rows are retained on point-in-time feature eligibility.  Invalid or incomplete
future paths remain present with null supervised targets; they are never
encoded as ordinary economic failures.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.strict_r3_canonical_v2 import SCHEMA  # noqa: E402


def _tree_sha(path: Path) -> str:
    digest = hashlib.sha256()
    paths = [path] if path.is_file() else sorted(value for value in path.rglob("*") if value.is_file())
    for value in paths:
        digest.update(str(value.relative_to(path) if path.is_dir() else value.name).encode())
        with value.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def _load_feature(path: Path, fields: list[str]) -> pd.DataFrame:
    columns = ["__ts__", "__symbol__", *fields]
    frame = pd.read_parquet(path, columns=columns)
    frame["__ts__"] = pd.to_datetime(frame["__ts__"], utc=True)
    return frame


def _load_exact(root: Path) -> pd.DataFrame:
    paths = sorted(root.glob("parts/month=*/side=long.parquet"))
    if not paths:
        raise FileNotFoundError(f"no long exact-label parts under {root}")
    columns = [
        "candidate_id", "__ts__", "__decision_ts__", "__symbol__", "side_name",
        "__label_available_at__", "label_valid", "target_invalid",
        "t2_tp6_sl4_event", "robust_clear_event_b25",
        "t4_tp6_sl4_gross_bps", "t4_tp6_sl4_net_bps",
    ]
    return pd.concat(
        [pd.read_parquet(path, columns=columns) for path in paths], ignore_index=True
    )


def _load_policy(path: Path) -> pd.DataFrame:
    if path.is_dir():
        path = path / "frozen_policy_labels.parquet"
    columns = [
        "candidate_id", "policy_path_valid", "policy_gross_bps", "policy_net_bps",
        "policy_label_available_ts", "policy_exit_bar_15m", "policy_exit_reason",
        "policy_entry_price", "policy_exit_price", "policy_cost_bps",
    ]
    return pd.read_parquet(path, columns=columns)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature", type=Path, action="append", required=True)
    parser.add_argument("--exact-root", type=Path, action="append", required=True)
    parser.add_argument("--policy", type=Path, action="append", required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument(
        "--min-base-feature-fraction",
        type=float,
        default=1.0,
        help=(
            "strict canonical default: require every frozen base field. "
            "A lower value is a separately named research contract and cannot "
            "claim replay/live feature parity."
        ),
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(f"immutable output already exists: {args.out_dir}")
    contract = json.loads(args.contract.read_text())
    base_fields = list(contract["base_fields_by_side"]["long"])
    context_fields = list(contract["severe_context_fields"])
    all_fields = list(dict.fromkeys([*base_fields, *context_fields]))
    feature_frames = []
    for precedence, path in enumerate(args.feature):
        source = _load_feature(path, all_fields)
        source["__feature_source_precedence__"] = precedence
        feature_frames.append(source)
    features = pd.concat(feature_frames, ignore_index=True)
    overlap_rows = int(features.duplicated(["__ts__", "__symbol__"], keep=False).sum())
    # Later sources intentionally supersede earlier broad historical stores.  This
    # allows a freshly materialised year to repair an older, incomplete feature
    # panel without creating two candidate identities.
    features = (
        features.sort_values(
            ["__ts__", "__symbol__", "__feature_source_precedence__"],
            kind="stable",
        )
        .drop_duplicates(["__ts__", "__symbol__"], keep="last")
        .drop(columns="__feature_source_precedence__")
    )
    exact = pd.concat([_load_exact(root) for root in args.exact_root], ignore_index=True)
    if exact["candidate_id"].duplicated().any():
        raise ValueError("exact-label sources overlap or duplicate candidate IDs")
    policy = pd.concat([_load_policy(path) for path in args.policy], ignore_index=True)
    if policy["candidate_id"].duplicated().any():
        raise ValueError("policy-label sources overlap or duplicate candidate IDs")
    for column in ("__ts__", "__decision_ts__", "__label_available_at__"):
        exact[column] = pd.to_datetime(exact[column], utc=True)
    policy["policy_label_available_ts"] = pd.to_datetime(
        policy["policy_label_available_ts"], utc=True
    )
    frame = exact.merge(policy, on="candidate_id", how="left", validate="one_to_one")
    frame = frame.merge(
        features, on=["__ts__", "__symbol__"], how="left", validate="many_to_one"
    )
    if not 0.0 < float(args.min_base_feature_fraction) <= 1.0:
        raise ValueError("--min-base-feature-fraction must be in (0, 1]")
    available = frame[base_fields].replace([np.inf, -np.inf], np.nan).notna()
    feature_fraction = available.mean(axis=1)
    frame["base_contract_complete"] = available.all(axis=1)
    frame["base_feature_available_fraction"] = feature_fraction.astype(np.float32)
    eligible = feature_fraction.ge(float(args.min_base_feature_fraction))
    feature_rejections = frame.loc[
        ~eligible,
        ["candidate_id", "__ts__", "__decision_ts__", "__symbol__", "side_name",
         "base_feature_available_fraction"],
    ].copy()
    feature_rejections["rejection_reason"] = "base_feature_fraction_below_declared_gate"
    frame = frame.loc[eligible].copy()
    valid = (
        frame["label_valid"].fillna(False).astype(bool)
        & ~frame["target_invalid"].fillna(True).astype(bool)
    )
    event = pd.to_numeric(frame["t2_tp6_sl4_event"], errors="coerce")
    robust = pd.to_numeric(frame["robust_clear_event_b25"], errors="coerce")
    frame["r3_class"] = np.nan
    frame.loc[valid & event.eq(1.0), "r3_class"] = 0
    frame.loc[valid & event.ne(1.0) & robust.ne(1.0), "r3_class"] = 1
    frame.loc[valid & event.ne(1.0) & robust.eq(1.0), "r3_class"] = 2
    frame["r3_label_available_ts"] = frame["__label_available_at__"]
    frame["h12_label_available_ts"] = frame["__label_available_at__"]
    frame["h12_label_valid"] = valid
    frame["h12_tp6_sl4_gross_bps"] = pd.to_numeric(
        frame["t4_tp6_sl4_gross_bps"], errors="coerce"
    ).where(valid)
    frame["h12_tp6_sl4_net_bps"] = pd.to_numeric(
        frame["t4_tp6_sl4_net_bps"], errors="coerce"
    ).where(valid)
    frame["policy_path_valid"] = frame["policy_path_valid"].fillna(False).astype(bool)
    for column in ("policy_gross_bps", "policy_net_bps"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce").where(
            frame["policy_path_valid"]
        )
    frame["geometry_definition_population_complete"] = True
    keep = [
        "candidate_id", "__ts__", "__decision_ts__", "__symbol__", "side_name",
        "r3_class", "r3_label_available_ts", "policy_path_valid",
        "policy_label_available_ts", "policy_gross_bps", "policy_net_bps",
        "policy_exit_bar_15m", "policy_exit_reason", "policy_entry_price",
        "policy_exit_price", "policy_cost_bps", "h12_label_valid",
        "h12_label_available_ts", "h12_tp6_sl4_gross_bps",
        "h12_tp6_sl4_net_bps", "geometry_definition_population_complete",
        "base_contract_complete", "base_feature_available_fraction",
        *all_fields,
    ]
    frame = frame.loc[:, keep].sort_values(
        ["__decision_ts__", "candidate_id"], kind="stable"
    ).reset_index(drop=True)
    if frame["candidate_id"].duplicated().any():
        raise AssertionError("assembled panel duplicated candidate identities")
    frame["month"] = frame["__decision_ts__"].dt.strftime("%Y-%m")
    audit = frame.groupby("month", as_index=False).agg(
        rows=("candidate_id", "size"),
        r3_valid_rows=("r3_class", "count"),
        h12_valid_rows=("h12_label_valid", "sum"),
        policy_valid_rows=("policy_path_valid", "sum"),
        symbols=("__symbol__", "nunique"),
    )
    frame = frame.drop(columns="month")
    args.out_dir.mkdir(parents=True)
    panel_path = args.out_dir / "canonical_source_panel.parquet"
    frame.to_parquet(panel_path, index=False, compression="zstd")
    feature_rejections.to_parquet(
        args.out_dir / "feature_eligibility_rejections.parquet",
        index=False, compression="zstd",
    )
    audit.to_parquet(args.out_dir / "source_panel_coverage.parquet", index=False)
    manifest = {
        "schema": f"{SCHEMA}_source_panel",
        "side_name": "long", "rows": len(frame),
        "base_fields": len(base_fields), "context_fields": len(context_fields),
        "overlapping_feature_rows_resolved_by_later_source": overlap_rows,
        "feature_source_precedence": [str(path) for path in args.feature],
        "future_invalid_rows_retained": True,
        "invalid_rows_encoded_as_failures": False,
        "min_base_feature_fraction": float(args.min_base_feature_fraction),
        "feature_eligibility_rejected_rows": int(len(feature_rejections)),
        "remaining_incomplete_rows_use_train_only_imputation": bool(
            float(args.min_base_feature_fraction) < 1.0
        ),
        "strict_full_base_contract_required": bool(
            float(args.min_base_feature_fraction) == 1.0
        ),
        "source_hashes": {
            "features": {str(path): _tree_sha(path) for path in args.feature},
            "exact": {str(path): _tree_sha(path) for path in args.exact_root},
            "policy": {str(path): _tree_sha(path) for path in args.policy},
            "contract": _tree_sha(args.contract),
        },
    }
    (args.out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({"event": "complete", "output": str(args.out_dir), **manifest}))


if __name__ == "__main__":
    main()
