#!/usr/bin/env python3
"""Build a strict BCF prequential ledger from one target-free feature lineage.

The feature matrix is constructed first from the frozen 170-symbol universe.
This program then attaches resolved labels from a separate source and builds
new chronological base/map/residual handoffs.  It intentionally consumes no
old feature or score columns from the legacy prequential ledger.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.strict_r3_canonical_v2 import build_prequential_stack_ledger


LABEL_COLUMNS = [
    "candidate_id", "__decision_ts__", "side_name", "r3_class",
    "r3_label_available_ts", "policy_net_bps", "policy_label_available_ts",
    "h12_label_valid", "h12_label_available_ts", "h12_tp6_sl4_net_bps",
]


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _contract(path: Path) -> tuple[list[str], list[str]]:
    payload = json.loads(path.read_text())
    base = [str(name) for name in payload["base_fields_by_side"]["long"]]
    context = [str(name) for name in payload["severe_context_fields"]]
    if len(base) != 120 or len(set(base)) != 120:
        raise ValueError("same-lineage BCF requires the 120-field long base contract")
    if len(context) != 73 or len(set(context)) != 73:
        raise ValueError("same-lineage BCF requires the 73-field Severe context contract")
    return base, context


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--label-source", type=Path, required=True)
    parser.add_argument("--feature-contract", type=Path, required=True)
    parser.add_argument("--first-held-month", required=True)
    parser.add_argument("--last-held-month", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(args.out_dir)

    base, context = _contract(args.feature_contract)
    feature_columns = [
        "candidate_id", "__decision_ts__", "__symbol__", "side_name", *base, *context,
    ]
    feature_columns = list(dict.fromkeys(feature_columns))
    features = pd.read_parquet(args.features, columns=feature_columns)
    features["__decision_ts__"] = pd.to_datetime(features["__decision_ts__"], utc=True)
    if features.empty or features["candidate_id"].duplicated().any():
        raise ValueError("target-free feature matrix must be nonempty and candidate-unique")
    if not features["side_name"].astype(str).str.lower().eq("long").all():
        raise ValueError("same-lineage BCF retrain is long-only")
    if features.loc[:, list(dict.fromkeys([*base, *context]))].isna().all(axis=1).any():
        raise ValueError("feature matrix contains rows with no model-contract inputs")

    labels = pd.read_parquet(args.label_source, columns=LABEL_COLUMNS)
    labels["__decision_ts__"] = pd.to_datetime(labels["__decision_ts__"], utc=True)
    if labels["candidate_id"].duplicated().any():
        raise ValueError("label source must be candidate-unique")
    if not labels["side_name"].astype(str).str.lower().eq("long").all():
        raise ValueError("label source is not long-only")
    labels = labels.drop(columns=["side_name"])

    panel = features.merge(labels, on="candidate_id", how="left", suffixes=("", "__label"), validate="one_to_one")
    for column in ("__decision_ts__",):
        label_column = f"{column}__label"
        if label_column in panel.columns:
            mismatch = panel[label_column].notna() & panel[column].ne(panel[label_column])
            if mismatch.any():
                raise ValueError("label source decision timestamps disagree with target-free feature identities")
            panel = panel.drop(columns=label_column)
    for column in ("r3_label_available_ts", "policy_label_available_ts", "h12_label_available_ts"):
        panel[column] = pd.to_datetime(panel[column], utc=True)

    ledger, audit = build_prequential_stack_ledger(
        panel,
        base_fields=base,
        first_held_month=args.first_held_month,
        last_held_month=args.last_held_month,
        reference_days=42,
    )
    if ledger.empty or not ledger["stack_is_prequential"].fillna(False).astype(bool).all():
        raise ValueError("same-lineage construction produced no fully prequential ledger rows")
    args.out_dir.mkdir(parents=True)
    panel.to_parquet(args.out_dir / "target_free_features_then_labels.parquet", index=False, compression="zstd")
    ledger.to_parquet(args.out_dir / "prequential_stack_ledger.parquet", index=False, compression="zstd")
    audit.to_parquet(args.out_dir / "prequential_audit.parquet", index=False, compression="zstd")
    manifest = {
        "schema": "strict_r3_bcf_same_lineage_prequential_ledger_v1",
        "status": "complete",
        "feature_lineage": "target_free_170_universe_stateful_materialization",
        "features": {"path": str(args.features), "sha256": _sha(args.features)},
        "label_source": {"path": str(args.label_source), "sha256": _sha(args.label_source)},
        "feature_contract": {"path": str(args.feature_contract), "sha256": _sha(args.feature_contract)},
        "feature_rows": int(len(features)),
        "label_rows_joined": int(panel["r3_class"].notna().sum()),
        "prequential_rows": int(len(ledger)),
        "prequential_decisions": int(ledger["__decision_ts__"].nunique()),
        "first_held_month": str(args.first_held_month),
        "last_held_month": str(args.last_held_month),
        "reference_days": 42,
        "outcomes_joined_only_after_target_free_feature_materialization": True,
        "legacy_feature_or_score_columns_consumed": [],
    }
    (args.out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"event": "complete", **manifest}))


if __name__ == "__main__":
    main()
