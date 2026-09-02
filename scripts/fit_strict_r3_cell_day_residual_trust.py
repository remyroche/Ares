#!/usr/bin/env python3
"""Fit and persist one canonical R5 Cell-day residual-trust bundle."""

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

from extreme_price_movements.strict_r3_cell_day_trust import (  # noqa: E402
    POSTERIOR_CONTRACT_PATH,
    persist_cell_day_residual_trust_bundle,
    train_cell_day_residual_trust_bundle,
)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prequential-ledger", type=Path, action="append", required=True,
        help="Repeat for contiguous immutable ledger partitions.",
    )
    parser.add_argument(
        "--cell-day-provenance", type=Path, action="append", required=True,
        help="Repeat for candidate-disjoint causal 28-day map partitions.",
    )
    parser.add_argument(
        "--expected-map-field",
        default="causal_21d_side_expected_net_bps",
        help="Explicit expected-net field in the supplied causal map provenance.",
    )
    parser.add_argument(
        "--mapping-status-field",
        default="causal_21d_side_mapping_status",
        help="Mapping-status field; pass an empty string only for audited ablation ledgers.",
    )
    parser.add_argument("--cutoff", required=True)
    parser.add_argument("--side", choices=("long", "short"), default="long")
    parser.add_argument(
        "--integration-contract", type=Path, default=POSTERIOR_CONTRACT_PATH,
        help="R5 integration contract; canonical default is the 9-month posterior admission.",
    )
    parser.add_argument(
        "--model-contract", type=Path, default=None,
        help="Explicit side-local R5 model contract. Required for a short R5 fit.",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    # The R5 learner has a frozen feature contract.  Reading unrelated wide
    # base/geometry columns from a multi-year ledger multiplies memory without
    # changing either the selected training rows or the fitted model.
    ledger_columns: list[str] | None = None
    if args.model_contract is not None:
        model_payload = json.loads(args.model_contract.read_text())
        fields = [str(value) for value in model_payload.get("features", [])]
        ledger_columns = list(dict.fromkeys([
            "candidate_id", "__decision_ts__", "side_name",
            "policy_label_available_ts", "policy_path_valid", "policy_net_bps",
            "final_score", *fields,
        ]))
    ledger = pd.concat(
        [pd.read_parquet(path, columns=ledger_columns) for path in args.prequential_ledger],
        ignore_index=True,
    )
    map_columns = ["candidate_id", "__decision_ts__", args.expected_map_field]
    if args.mapping_status_field:
        map_columns.append(args.mapping_status_field)
    mapped = pd.concat(
        [pd.read_parquet(path, columns=list(dict.fromkeys(map_columns))) for path in args.cell_day_provenance],
        ignore_index=True,
    )
    if ledger["candidate_id"].duplicated().any():
        raise ValueError("prequential ledger partitions contain duplicate candidate IDs")
    if "side_name" not in ledger.columns:
        raise ValueError("prequential ledger lacks mandatory side_name")
    observed_side = ledger["side_name"].astype(str).str.strip().str.lower()
    if not observed_side.eq(args.side).all():
        raise ValueError(
            "prequential ledger must be side-local to --side; "
            f"observed={observed_side.value_counts(dropna=False).to_dict()}",
        )
    required_map = {"candidate_id", "__decision_ts__", args.expected_map_field}
    if args.mapping_status_field:
        required_map.add(args.mapping_status_field)
    missing = sorted(required_map.difference(mapped.columns))
    if missing:
        raise ValueError(f"Cell-day provenance lacks: {missing}")
    if mapped["candidate_id"].duplicated().any():
        raise ValueError("Cell-day provenance has duplicate candidate IDs")
    # A downstream evaluation ledger may already carry map-lineage columns
    # from an earlier application.  The separately supplied, hashed
    # provenance is authoritative for this fit; retaining both would create
    # pandas suffixes and, more importantly, could silently reuse the wrong
    # producer's expected-net value.
    ledger = ledger.drop(
        columns=["__map_decision_ts__", "raw_expected_bps", "__map_status__"],
        errors="ignore",
    )
    mapped = mapped.rename(columns={
        args.expected_map_field: "raw_expected_bps",
        "__decision_ts__": "__map_decision_ts__",
    })
    selected_columns = ["candidate_id", "__map_decision_ts__", "raw_expected_bps"]
    if args.mapping_status_field:
        mapped = mapped.rename(columns={args.mapping_status_field: "__map_status__"})
        selected_columns.append("__map_status__")
    joined = ledger.merge(
        mapped.loc[:, selected_columns],
        on="candidate_id", how="left", validate="one_to_one",
    )
    joined["__decision_ts__"] = pd.to_datetime(joined["__decision_ts__"], utc=True)
    joined["__map_decision_ts__"] = pd.to_datetime(joined["__map_decision_ts__"], utc=True)
    overlap = joined["__map_decision_ts__"].notna()
    if not joined.loc[overlap, "__decision_ts__"].eq(joined.loc[overlap, "__map_decision_ts__"]).all():
        raise ValueError("Cell-day provenance identity/timestamp mismatch")
    bundle = train_cell_day_residual_trust_bundle(
        joined,
        cutoff=args.cutoff,
        side=args.side,
        integration_contract_path=args.integration_contract,
        model_contract_path=args.model_contract,
        source_hashes={
            "prequential_ledgers": [
                {"path": str(path), "sha256": _sha(path)}
                for path in args.prequential_ledger
            ],
            "cell_day_provenance": [
                {"path": str(path), "sha256": _sha(path)}
                for path in args.cell_day_provenance
            ],
            "expected_map_field": args.expected_map_field,
            "mapping_status_field": args.mapping_status_field or None,
            "model_contract": None if args.model_contract is None else str(args.model_contract),
            "side": args.side,
        },
    )
    manifest = persist_cell_day_residual_trust_bundle(bundle, args.out_dir)
    print(json.dumps({"event": "complete", "out_dir": str(args.out_dir), **manifest}))


if __name__ == "__main__":
    main()
