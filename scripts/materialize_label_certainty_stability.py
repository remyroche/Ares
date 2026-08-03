#!/usr/bin/env python3
"""Materialize train-only label-certainty diagnostics from exact H12 paths.

The command fails closed when its requested perturbation contracts are not
supported by the immutable path source.  In particular, a 16h neighbour cannot
be silently fabricated from an H12 path pack.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.label_certainty_stability import (  # noqa: E402
    DEFAULT_PERTURBATION_CONTRACTS,
    build_label_certainty,
    contracts_payload,
    materialize_perturbed_barrier_targets,
)


def _decode(values: pd.Series) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    raw = [json.loads(value) for value in values]
    try:
        result = tuple(np.asarray([item[name] for item in raw], dtype=np.float64) for name in ("open", "high", "low", "close"))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("execution_future_path must contain complete numeric OHLC JSON") from exc
    return result  # type: ignore[return-value]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary-labels", type=Path, required=True)
    parser.add_argument("--paths", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--atr-long-column", default=None, help="Pre-entry long ATR column materialized alongside each path.")
    parser.add_argument("--max-rows", type=int, default=None, help="Diagnostic smoke limit only; never use as a production artifact.")
    parser.add_argument("--allow-incomplete-neighborhood", action="store_true", help="Write a blocked diagnostic manifest instead of raising when the frozen paths cannot support every predeclared contract.")
    args = parser.parse_args()
    primary = pd.read_parquet(args.primary_labels, columns=["candidate_id", "side", "execution_exact_h12_net_bps", "execution_exact_h12_cost_bps", "path_auxiliary_atr_fraction", "label_available_ts"])
    if primary.candidate_id.isna().any() or primary.candidate_id.duplicated().any():
        raise ValueError("primary labels have invalid candidate_id")
    if primary.side.isna().any():
        raise ValueError("primary labels have invalid side")
    if args.max_rows is not None:
        primary = primary.sort_values("candidate_id", kind="stable").head(args.max_rows).copy()
    required = {"candidate_id", "execution_future_path"}
    if args.atr_long_column:
        required.add(args.atr_long_column)
    path_columns = ["candidate_id", "execution_future_path"] + ([args.atr_long_column] if args.atr_long_column else [])
    wanted_ids = primary.candidate_id.astype(str).tolist()
    # Predicate pushdown is important: full JSON path columns are large and a
    # smoke materialization must not deserialize unrelated paths.
    paths = pd.concat([pd.read_parquet(path, columns=path_columns, filters=[("candidate_id", "in", wanted_ids)]) for path in args.paths], ignore_index=True)
    if paths.candidate_id.duplicated().any():
        raise ValueError("path candidate identity must be unique across supplied path packs")
    joined = primary.merge(paths, on="candidate_id", how="inner", validate="one_to_one")
    if joined.empty:
        raise ValueError("no primary labels could be joined to exact paths")
    open_, high, low, close = _decode(joined.execution_future_path)
    available_minutes = close.shape[1]
    unsupported = [item.contract_id for item in DEFAULT_PERTURBATION_CONTRACTS if item.entry_delay_minutes + item.horizon_minutes > available_minutes]
    atr_missing = any(item.atr_source != "reference" for item in DEFAULT_PERTURBATION_CONTRACTS) and not args.atr_long_column
    payload = contracts_payload()
    payload.update({"primary_labels": str(args.primary_labels), "primary_labels_sha256": _sha256(args.primary_labels), "path_sources": [str(path) for path in args.paths], "available_path_minutes": int(available_minutes), "unsupported_contract_ids": unsupported, "atr_long_materialized": bool(args.atr_long_column), "row_count_joined": int(len(joined)), "max_rows": args.max_rows})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "label_stability_contracts.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if unsupported or atr_missing:
        payload["status"] = "BLOCKED_INCOMPLETE_FROZEN_PATH_CONTRACT"
        payload["blockers"] = (["requested contracts exceed exact path horizon: " + ", ".join(unsupported)] if unsupported else []) + (["long pre-entry ATR is not materialized in the exact path source"] if atr_missing else [])
        (args.output_dir / "manifest.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if not args.allow_incomplete_neighborhood:
            raise ValueError("; ".join(payload["blockers"]))
        return
    sign = np.where(joined.side.astype(str).str.lower().eq("long"), 1.0, -1.0)
    alternatives = {"long": pd.to_numeric(joined[args.atr_long_column], errors="raise").to_numpy(dtype=float)}
    variants, contract_report = materialize_perturbed_barrier_targets(open_=open_, high=high, low=low, close=close, side_sign=sign, atr_reference=pd.to_numeric(joined.path_auxiliary_atr_fraction, errors="raise").to_numpy(dtype=float), cost_return=pd.to_numeric(joined.execution_exact_h12_cost_bps, errors="raise").to_numpy(dtype=float) / 10000.0, atr_alternatives=alternatives)
    certainty = build_label_certainty(variants, reference_target=pd.to_numeric(joined.execution_exact_h12_net_bps, errors="raise").to_numpy(dtype=float) / 10000.0)
    identity = joined.loc[:, ["candidate_id", "label_available_ts"]].reset_index(drop=True)
    identity.join(certainty).to_parquet(args.output_dir / "label_certainty_diagnostics.parquet", index=False)
    contract_report.to_parquet(args.output_dir / "label_stability_contract_report.parquet", index=False)
    payload["status"] = "COMPLETED_TRAINING_ONLY"
    payload["outputs"] = {"diagnostics": "label_certainty_diagnostics.parquet", "contract_report": "label_stability_contract_report.parquet"}
    (args.output_dir / "manifest.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
