#!/usr/bin/env python3
"""Build an exact-ID, target-free F72 + SHAP/state Meta feature universe.

The incumbent causal panel owns the frozen raw F72 values, while the P8U
enriched bridge owns the target-free SHAP and market-state overlays.  Neither
is a replacement for the other.  This producer freezes their identity join
before any policy or path outcomes are opened and writes one causal feature
owner per month for downstream Meta selection only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "strict_r3_p8u_meta_f72_enriched_universe_v1"
IDENTITY = ("candidate_id", "__decision_ts__", "side_name")
PROHIBITED_SUBSTRINGS = (
    "policy_", "path_arch_", "supportive_", "h12_", "label_", "outcome_",
    "future_", "target_",
)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    members = sorted(path.rglob("*.parquet")) if path.is_dir() else [path]
    for member in members:
        digest.update(str(member).encode())
        with member.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
    return digest.hexdigest()


def _once(path: Path, payload: object) -> None:
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)


def _months(values: list[str]) -> tuple[pd.Timestamp, ...]:
    parsed = tuple(pd.Timestamp(f"{value}-01", tz="UTC") for value in values)
    if not parsed or len(parsed) != len(set(parsed)):
        raise ValueError("months must be distinct YYYY-MM values")
    return tuple(sorted(parsed))


def _read_features(contract: Path) -> tuple[str, ...]:
    payload = json.loads(contract.read_text())
    fields = payload.get("selected_features", payload.get("features"))
    if not isinstance(fields, list) or len(fields) != 72 or len(fields) != len(set(fields)):
        raise AssertionError(f"{contract}: expected exactly 72 unique frozen raw Meta fields")
    return tuple(str(item) for item in fields)


def _target_free(path: Path) -> None:
    text = path.parent / "run_manifest.json"
    if not text.exists():
        raise FileNotFoundError(f"missing target-free source manifest: {text}")
    # File schema itself is enforced below.  The manifest is retained as a
    # receipt rather than interpreted as a model-selection source.


def _month_path(root: Path, month: pd.Timestamp, name: str) -> Path:
    path = root / f"month={month:%Y-%m}" / name
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def run(
    *, candidate_root: Path, raw_f72_root: Path, enriched_root: Path,
    f72_contract: Path, months: tuple[pd.Timestamp, ...], out: Path,
) -> Path:
    if out.exists():
        raise FileExistsError(f"immutable output exists: {out}")
    fields = _read_features(f72_contract)
    out.mkdir(parents=True)
    audit: list[dict[str, Any]] = []
    feature_coverage: list[dict[str, Any]] = []
    append_fields: tuple[str, ...] | None = None
    for month in months:
        candidate_path = candidate_root / f"month={month:%Y-%m}.parquet"
        raw_path = _month_path(raw_f72_root, month, "causal_feature_universe.parquet")
        enriched_path = _month_path(enriched_root, month, "causal_feature_universe.parquet")
        _target_free(candidate_path)
        candidates = pd.read_parquet(candidate_path, columns=list(IDENTITY)).copy()
        raw = pd.read_parquet(raw_path, columns=[*IDENTITY, *fields]).copy()
        enriched = pd.read_parquet(enriched_path).copy()
        for frame in (candidates, raw, enriched):
            frame["__decision_ts__"] = pd.to_datetime(frame["__decision_ts__"], utc=True, errors="raise")
            if frame.duplicated(list(IDENTITY)).any() or not frame.side_name.eq("long").all():
                raise AssertionError(f"{month:%Y-%m}: non-unique or non-long causal identity")
        absent_raw = candidates.merge(raw.loc[:, list(IDENTITY)], on=list(IDENTITY), how="left", indicator=True)
        absent_enriched = candidates.merge(enriched.loc[:, list(IDENTITY)], on=list(IDENTITY), how="left", indicator=True)
        if not absent_raw._merge.eq("both").all() or not absent_enriched._merge.eq("both").all():
            raise AssertionError(f"{month:%Y-%m}: raw F72 or enriched source does not cover frozen target-free IDs")
        forbidden = [
            column for column in enriched.columns
            if column not in IDENTITY and any(token in column.lower() for token in PROHIBITED_SUBSTRINGS)
        ]
        if forbidden:
            raise AssertionError(f"{month:%Y-%m}: enriched input exposes outcome-like columns: {forbidden[:16]}")
        extras = tuple(column for column in enriched.columns if column not in IDENTITY and column not in fields)
        if append_fields is None:
            append_fields = extras
        elif extras != append_fields:
            raise AssertionError(f"{month:%Y-%m}: enriched append feature ordering drift")
        raw_selected = candidates.merge(raw.loc[:, [*IDENTITY, *fields]], on=list(IDENTITY), how="left", validate="one_to_one")
        final = raw_selected.merge(enriched.loc[:, [*IDENTITY, *extras]], on=list(IDENTITY), how="left", validate="one_to_one")
        if final.columns.tolist() != [*IDENTITY, *fields, *extras]:
            raise AssertionError(f"{month:%Y-%m}: output column ordering drift")
        target = out / f"month={month:%Y-%m}" / "causal_feature_universe.parquet"
        target.parent.mkdir(parents=True, exist_ok=True)
        final.to_parquet(target, index=False, compression="zstd")
        # The downstream selector needs a month-local, target-free hygiene
        # receipt.  Compute it from this just-materialized causal panel—not
        # from labels or a later selection universe.
        for field in (*fields, *extras):
            values = pd.to_numeric(final[field], errors="coerce").to_numpy(dtype=float, copy=False)
            finite = np.isfinite(values)
            feature_coverage.append({
                "month": f"{month:%Y-%m}",
                "feature": field,
                "finite_fraction": float(finite.mean()),
                "nunique": int(pd.Series(values[finite]).nunique(dropna=True)),
            })
        audit.append({
            "month": f"{month:%Y-%m}", "candidate_rows": len(candidates),
            "raw_f72_rows": len(raw), "enriched_rows": len(enriched),
            "f72_fields": len(fields), "enriched_append_fields": len(extras),
            "output_fields": len(fields) + len(extras),
            "identity_exact": True,
            "outcome_like_input_fields": 0,
        })
    coverage = pd.DataFrame(audit)
    coverage.to_parquet(out / "source_coverage_audit.parquet", index=False, compression="zstd")
    pd.DataFrame(feature_coverage).to_parquet(
        out / "feature_coverage.parquet", index=False, compression="zstd",
    )
    _once(out / "feature_universe_contract.json", {
        "schema": SCHEMA,
        "frozen_raw_f72": list(fields),
        "frozen_raw_f72_count": len(fields),
        "target_free_append_pool": list(append_fields or ()),
        "target_free_append_pool_count": len(append_fields or ()),
        "selection_rule": "Downstream feature selection may retain frozen F72 fields and append only this declared target-free pool; no outcomes are fields.",
    })
    _once(out / "correctness_report.json", {
        "candidate_identity_is_frozen_target_free_owner": True,
        "raw_f72_and_enriched_panels_cover_every_candidate_exactly": True,
        "raw_f72_values_remain_the_frozen_parent_owner": True,
        "shap_and_state_fields_are_append_only_target_free_candidates": True,
        "month_local_feature_coverage_is_computed_from_the_target_free_owner": True,
        "outcome_like_columns_rejected_from_feature_inputs": True,
        "no_policy_path_label_or_mc1_input": True,
    })
    _once(out / "run_manifest.json", {
        "schema": SCHEMA,
        "scope": "offline target-free P8U Meta feature universe; no admission, portfolio, live, or exchange authority",
        "candidate_root": str(candidate_root.resolve()),
        "raw_f72_root": str(raw_f72_root.resolve()),
        "enriched_root": str(enriched_root.resolve()),
        "f72_contract": str(f72_contract.resolve()),
        "candidate_sha256": _sha(candidate_root),
        "raw_f72_sha256": _sha(raw_f72_root),
        "enriched_sha256": _sha(enriched_root),
        "f72_contract_sha256": _sha(f72_contract),
        "months": [f"{month:%Y-%m}" for month in months],
        "coverage": audit,
    })
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--raw-f72-root", type=Path, required=True)
    parser.add_argument("--enriched-root", type=Path, required=True)
    parser.add_argument("--f72-contract", type=Path, required=True)
    parser.add_argument("--months", nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(run(
        candidate_root=args.candidate_root.resolve(), raw_f72_root=args.raw_f72_root.resolve(),
        enriched_root=args.enriched_root.resolve(), f72_contract=args.f72_contract.resolve(),
        months=_months(list(args.months)), out=args.out.resolve(),
    ))


if __name__ == "__main__":
    main()
