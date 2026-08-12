#!/usr/bin/env python3
"""Materialise selector-ready Stage-A/F4 evidence from a candidate panel.

Example contract JSON (side contracts may differ)::

  {"F0_current_frozen":{"long":["x"],"short":["x"]},
   "F3_plus_relative":{"long":["x","x__causal_rank_w90",...],"short":[...]}}

No panel is generated here.  The caller supplies a pre-materialised CSV or
Parquet candidate panel with decision timestamps, label availability timestamps,
side, candidate identity, net-bps target, and every declared feature column.
"""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.feature_portability_mda import (  # noqa: E402
    ChronologicalTransport,
    FrozenR3ModelContract,
    R3CostContract,
    materialize_feature_portability_f4_evidence,
)


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"unsupported candidate panel format: {path} (use parquet or csv)")


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _transports(payload: Any) -> tuple[ChronologicalTransport, ...]:
    if not isinstance(payload, list):
        raise ValueError("transports JSON must be a list")
    rows = []
    for item in payload:
        if not isinstance(item, dict):
            raise ValueError("every transport must be an object")
        rows.append(ChronologicalTransport(
            name=item["name"], train_start=item["train_start"],
            evaluation_start=item["evaluation_start"], evaluation_end=item["evaluation_end"],
        ))
    return tuple(rows)


def _r3_cost_contract(payload: Any) -> R3CostContract:
    if not isinstance(payload, dict):
        raise ValueError("R3/cost contract must be a JSON object")
    required = {"class_column", "gross_bps_column", "net_bps_column", "expected_cost_bps"}
    missing = sorted(required.difference(payload))
    if missing:
        raise ValueError(f"R3/cost contract lacks: {missing}")
    if "sample_weight_column" not in payload and "robust_clear_columns" not in payload:
        raise ValueError("R3/cost contract requires sample_weight_column or robust_clear_columns")
    robust = payload.get("robust_clear_columns", ())
    return R3CostContract(
        class_column=str(payload["class_column"]), gross_bps_column=str(payload["gross_bps_column"]),
        net_bps_column=str(payload["net_bps_column"]), expected_cost_bps=float(payload["expected_cost_bps"]),
        sample_weight_column=payload.get("sample_weight_column"), robust_clear_columns=tuple(robust),
    )


def _r3_model_contract(payload: Any) -> FrozenR3ModelContract:
    if not isinstance(payload, dict):
        raise ValueError("frozen R3 model contract must be a JSON object")
    required = {"model_id", "params", "model_hpo_performed"}
    missing = sorted(required.difference(payload))
    if missing:
        raise ValueError(f"frozen R3 model contract lacks: {missing}")
    return FrozenR3ModelContract(
        model_id=str(payload["model_id"]), params=payload["params"],
        random_seed=int(payload.get("random_seed", 17)), model_hpo_performed=bool(payload["model_hpo_performed"]),
    )


def _callback(import_path: str):
    module_name, separator, attribute = import_path.partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError("--r3-fit-predict must be an import path in module:callable form")
    callback = getattr(importlib.import_module(module_name), attribute, None)
    if not callable(callback):
        raise ValueError("--r3-fit-predict must resolve to a callable")
    return callback


def _selector_lineage(
    *,
    representation_contracts: dict[str, object],
    compact_contracts: dict[str, object],
    transports: tuple[ChronologicalTransport, ...],
    control_representation: str,
) -> dict[str, object]:
    """Return the post-fit lineage required by the fail-closed F4 selector.

    This file is written only after the frozen R3 transport fits complete, so
    ``oof_materialised`` describes actual chronological scoring rather than a
    pre-fit panel declaration.  The per-side contracts remain distinct even
    though the selector conservatively uses the larger side feature count.
    """
    records: list[dict[str, object]] = []
    for representation, side_contracts in representation_contracts.items():
        if representation == control_representation:
            continue
        if not isinstance(side_contracts, dict):
            raise ValueError("representation contracts must map representations to side contracts")
        for transport in transports:
            for side in ("long", "short"):
                fields = side_contracts.get(side)
                if not isinstance(fields, list) or not all(isinstance(field, str) for field in fields):
                    raise ValueError(f"{representation}/{side} must be a string feature list")
                records.append({
                    "representation": representation,
                    "run": transport.name,
                    "side_name": side,
                    "oof_materialised": True,
                    "feature_count": len(fields),
                    "features": fields,
                    "source": "frozen_r3_outer_transport_fit",
                })
    compact_by_representation = compact_contracts.get("representations")
    if not isinstance(compact_by_representation, dict):
        raise ValueError("F4 compact-contract payload lacks representations")
    for representation, record in compact_by_representation.items():
        if not isinstance(record, dict) or not isinstance(record.get("by_transport"), dict):
            raise ValueError(f"F4 compact contract lacks per-transport lineage: {representation}")
        for transport in transports:
            side_contracts = record["by_transport"].get(transport.name)
            if not isinstance(side_contracts, dict):
                raise ValueError(f"F4 compact contract omits {transport.name}: {representation}")
            for side in ("long", "short"):
                fields = side_contracts.get(side)
                if not isinstance(fields, list) or not all(isinstance(field, str) for field in fields):
                    raise ValueError(f"F4 compact {representation}/{transport.name}/{side} is not a string feature list")
                records.append({
                    "representation": representation,
                    "run": transport.name,
                    "side_name": side,
                    "oof_materialised": True,
                    "feature_count": len(fields),
                    "features": fields,
                    "source": "inner_chronological_f4_transform_family_mda_then_frozen_r3_outer_transport_fit",
                })
    return {"schema": "stage_a_f4_selector_lineage_v1", "records": records}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Materialise development-only F4 grouped-MDA evidence")
    parser.add_argument("--panel", type=Path, required=True, help="pre-materialised CSV or Parquet candidate panel")
    parser.add_argument("--representation-contracts", type=Path, required=True, help="JSON representation -> long/short feature lists")
    parser.add_argument("--transports", type=Path, required=True, help="JSON list of development chronological transports")
    parser.add_argument("--control-representation", required=True)
    parser.add_argument("--f3-representation", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--r3-cost-contract", type=Path, required=True, help="required class/gross/net/cost/weight JSON contract")
    parser.add_argument("--frozen-r3-model-contract", type=Path, required=True, help="required fixed R3 model-id/params/no-HPO JSON contract")
    parser.add_argument("--r3-fit-predict", required=True, help="frozen R3 callback import path: module:callable")
    parser.add_argument("--inner-folds", type=int, default=2)
    parser.add_argument("--min-coverage", type=float, default=0.99)
    parser.add_argument(
        "--f4-group-count", action="append", type=int,
        help="Predeclared nested compact transform-family count; repeat in increasing order (default: 1,2,3).",
    )
    args = parser.parse_args(argv)

    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite F4 evidence directory: {args.output_dir}")
    representation_contracts = _read_json(args.representation_contracts)
    transports = _transports(_read_json(args.transports))
    result = materialize_feature_portability_f4_evidence(
        _read_table(args.panel), representation_features=representation_contracts,
        control_representation=args.control_representation, f3_representation=args.f3_representation,
        transports=transports,
        r3_cost=_r3_cost_contract(_read_json(args.r3_cost_contract)),
        r3_model=_r3_model_contract(_read_json(args.frozen_r3_model_contract)),
        r3_fit_predict=_callback(args.r3_fit_predict), inner_folds=args.inner_folds, min_coverage=args.min_coverage,
        f4_group_counts=tuple(args.f4_group_count) if args.f4_group_count else (1, 2, 3),
    )
    args.output_dir.mkdir(parents=True)
    result.evidence.to_parquet(args.output_dir / "f4_evidence.parquet", index=False, compression="zstd")
    result.transformed_coverage.to_parquet(args.output_dir / "f4_actual_f3_transformed_coverage.parquet", index=False, compression="zstd")
    result.source_intersection_coverage.to_parquet(args.output_dir / "f4_source_intersection_coverage.parquet", index=False, compression="zstd")
    result.representation_coverage.to_parquet(args.output_dir / "f4_representation_coverage.parquet", index=False, compression="zstd")
    result.fold_mda.to_parquet(args.output_dir / "f4_grouped_chronological_mda.parquet", index=False, compression="zstd")
    result.feature_group_mda.to_parquet(args.output_dir / "f4_transform_family_inner_mda.parquet", index=False, compression="zstd")
    result.transport_audit.to_parquet(args.output_dir / "f4_transport_audit.parquet", index=False, compression="zstd")
    (args.output_dir / "f4_compact_contracts.json").write_text(
        json.dumps(result.compact_contracts, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "f4_selector_lineage.json").write_text(
        json.dumps(
            _selector_lineage(
                representation_contracts=representation_contracts, transports=transports,
                compact_contracts=dict(result.compact_contracts), control_representation=args.control_representation,
            ),
            indent=2, sort_keys=True,
        ) + "\n", encoding="utf-8"
    )
    (args.output_dir / "f4_evidence_manifest.json").write_text(
        json.dumps(result.manifest, indent=2, default=str) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
