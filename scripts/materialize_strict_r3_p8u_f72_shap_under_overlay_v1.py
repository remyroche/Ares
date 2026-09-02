#!/usr/bin/env python3
"""Materialise a narrow, target-free F72-SHAP overlay for Under-F120 research.

The canonical P8U Under head consumes a frozen 120-field causal contract.
This producer makes one *matched* extension containing those exact fields plus
only predeclared strict-OOF, per-row F72 SHAP coordinates.  It never opens a
policy/path label, computes no raw-feature CMI/IC, and fixes every held
identity before any downstream model can be fitted.

Research only: no admission, portfolio, inference, live, or exchange action.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "strict_r3_p8u_f72_shap_under_overlay_v1"
IDENTITY = ("candidate_id", "__decision_ts__", "side_name")
PROHIBITED = frozenset({
    "policy_path_valid", "policy_gross_bps", "policy_net_bps", "policy_exit_bar_15m",
    "policy_entry_price", "policy_exit_price", "policy_exit_reason",
    "policy_label_available_ts", "policy_cost_bps", "policy_outcome_source",
    "label_source_complete_1m_path", "supportive_path_valid",
    "supportive_label_available_ts", "path_arch_peak_mfe_atr", "path_arch_atr_fraction",
})


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    members = sorted(path.rglob("*.parquet")) if path.is_dir() else [path]
    for member in members:
        digest.update(str(member).encode())
        with member.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
    return digest.hexdigest()


def _once_json(path: Path, payload: object) -> None:
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)


def _months(text: str) -> tuple[pd.Timestamp, ...]:
    values = tuple(pd.Timestamp(f"{item.strip()}-01", tz="UTC") for item in text.split(",") if item.strip())
    if not values or len(values) != len(set(values)) or tuple(sorted(values)) != values:
        raise ValueError("--months must be unique chronological YYYY-MM values")
    return values


def _contract(path: Path) -> tuple[str, ...]:
    raw = json.loads(path.read_text())
    fields = raw.get("selected_features", raw.get("features"))
    if not isinstance(fields, list) or len(fields) != 120 or len(fields) != len(set(fields)):
        raise AssertionError(f"{path}: expected exact 120 unique Under fields")
    return tuple(str(value) for value in fields)


def _assert_target_free(path: Path) -> None:
    names = set(pq.ParquetFile(path).schema_arrow.names)
    leaked = sorted(names.intersection(PROHIBITED))
    if leaked:
        raise AssertionError(f"{path}: target-free source leaks label/outcome fields {leaked}")


def _raw_path(roots: Iterable[Path], month: pd.Timestamp) -> Path:
    candidates = [root / f"month={month:%Y-%m}" / "causal_feature_universe.parquet" for root in roots]
    existing = [path for path in candidates if path.is_file()]
    if len(existing) != 1:
        raise AssertionError(f"{month:%Y-%m}: expected exactly one raw feature owner, found {len(existing)}")
    return existing[0]


def _coverage(frame: pd.DataFrame, fields: tuple[str, ...], month: pd.Timestamp) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for field in fields:
        values = pd.to_numeric(frame[field], errors="coerce")
        rows.append({
            "month": f"{month:%Y-%m}", "feature": field, "rows": int(len(values)),
            "finite_rows": int(values.notna().sum()), "finite_fraction": float(values.notna().mean()),
            "n_unique": int(values.nunique(dropna=True)),
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shap-root", type=Path, required=True, help="completed target_free_shap_features root")
    parser.add_argument("--raw-feature-roots", type=Path, nargs="+", required=True)
    parser.add_argument("--under-contract", type=Path, required=True)
    parser.add_argument("--derived-features", nargs="+", required=True)
    parser.add_argument("--months", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    months = _months(args.months)
    shap_root = args.shap_root.resolve()
    raw_roots = tuple(path.resolve() for path in args.raw_feature_roots)
    parent_contract = args.under_contract.resolve()
    base_fields = _contract(parent_contract)
    derived = tuple(str(item) for item in args.derived_features)
    if len(derived) != len(set(derived)) or not derived or any(not field.startswith("shap_f72_") for field in derived):
        raise AssertionError("derived features must be non-empty unique F72 SHAP fields")
    if set(derived).intersection(base_fields):
        raise AssertionError("derived feature collides with frozen Under-F120 contract")
    fields = (*base_fields, *derived)
    out = args.out.resolve()
    if out.exists():
        raise FileExistsError(f"immutable output exists: {out}")
    out.mkdir(parents=True)

    coverage: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    source_receipts: list[dict[str, str]] = []
    for month in months:
        shap_path = shap_root / f"month={month:%Y-%m}.parquet"
        if not shap_path.is_file():
            raise FileNotFoundError(shap_path)
        raw_path = _raw_path(raw_roots, month)
        for path in (shap_path, raw_path):
            _assert_target_free(path)
        shap_names = set(pq.ParquetFile(shap_path).schema_arrow.names)
        raw_names = set(pq.ParquetFile(raw_path).schema_arrow.names)
        missing_shap, missing_raw = sorted(set(derived).difference(shap_names)), sorted(set(base_fields).difference(raw_names))
        if missing_shap or missing_raw:
            raise AssertionError(f"{month:%Y-%m}: missing SHAP={missing_shap}, Under={missing_raw[:8]}")
        shap = pd.read_parquet(shap_path, columns=[*IDENTITY, *derived])
        raw = pd.read_parquet(raw_path, columns=[*IDENTITY, *base_fields])
        for piece, name in ((shap, "shap"), (raw, "under")):
            piece["__decision_ts__"] = pd.to_datetime(piece["__decision_ts__"], utc=True, errors="raise")
            if piece.duplicated(IDENTITY).any() or not piece.side_name.eq("long").all():
                raise AssertionError(f"{month:%Y-%m}: invalid target-free {name} identity")
        # The Under causal panel is intentionally the complete point-in-time
        # feature population.  F72 is the Router-selected subset.  The
        # correct contract is therefore *left coverage of every F72 identity*,
        # rather than equality of the two population sizes.
        result = shap.merge(raw, on=list(IDENTITY), how="left", validate="one_to_one", indicator=True)
        if len(result) != len(shap) or not result["_merge"].eq("both").all() or result.duplicated(IDENTITY).any():
            raise AssertionError(f"{month:%Y-%m}: F72 target-free identities are not fully covered by Under features")
        result = result.drop(columns="_merge")
        result = result.loc[:, [*IDENTITY, *fields]].sort_values(["__decision_ts__", "candidate_id"], kind="stable").reset_index(drop=True)
        target = out / f"month={month:%Y-%m}"
        target.mkdir()
        result.to_parquet(target / "causal_feature_universe.parquet", index=False, compression="zstd")
        coverage.extend(_coverage(result, fields, month))
        audit.append({
            "month": f"{month:%Y-%m}", "rows": int(len(result)), "under_population_rows": int(len(raw)), "under_f120_fields": len(base_fields),
            "f72_shap_fields": len(derived), "identity_exact": True, "target_free": True,
            "mean_finite_fraction": float(result.loc[:, list(fields)].notna().mean().mean()),
        })
        source_receipts.append({"month": f"{month:%Y-%m}", "shap": str(shap_path), "shap_sha256": _sha(shap_path), "under": str(raw_path), "under_sha256": _sha(raw_path)})
        print(json.dumps({"event": "month_complete", **audit[-1]}, sort_keys=True), flush=True)

    pd.DataFrame(coverage).to_parquet(out / "feature_coverage.parquet", index=False, compression="zstd")
    pd.DataFrame(audit).to_parquet(out / "identity_audit.parquet", index=False, compression="zstd")
    contract_root = out / "contracts"
    contract_root.mkdir()
    extension_name = f"under_f{len(fields)}_f72_shap{len(derived)}"
    contract = {
        "schema": f"strict_r3_p8u_{extension_name}_contract_v1",
        "scope": "research-only matched Under-F120 extension; predeclared strict-OOF F72 SHAP features only",
        "parent_under_f120_contract": str(parent_contract), "parent_under_f120_sha256": _sha(parent_contract),
        "selected_features": list(fields), "feature_count": len(fields), "parent_feature_count": len(base_fields),
        "derived_features": list(derived), "selected_features_sha256": hashlib.sha256("\n".join(fields).encode()).hexdigest(),
        "selection": "predeclared only after nine strict-OOF monthly F72 SHAP diagnostics: positive timestamp IC and positive timestamp-top10 economics in every held fold",
    }
    _once_json(contract_root / f"{extension_name}.json", contract)
    correctness = {
        "base_under_fields_match_frozen_f120": True,
        "only_predeclared_shap_features_added": True,
        "target_free_sources_only": True,
        "identities_exact_before_model_fitting": True,
        "no_raw_feature_cmi_or_ic_computed": True,
        "no_policy_or_path_label_opened": True,
        "no_live_admission_portfolio_or_exchange_mutation": True,
    }
    _once_json(out / "correctness_report.json", correctness)
    _once_json(out / "run_manifest.json", {
        "schema": SCHEMA, "scope": "offline target-free overlay only; no labels/models/MC1/admission/portfolio/inference/live/exchange mutation",
        "months": [f"{month:%Y-%m}" for month in months], "shap_root": str(shap_root),
        "raw_feature_roots": [str(path) for path in raw_roots], "parent_under_contract": str(parent_contract),
        "source_receipts": source_receipts, "audit": audit, "correctness": correctness,
    })


if __name__ == "__main__":
    main()
