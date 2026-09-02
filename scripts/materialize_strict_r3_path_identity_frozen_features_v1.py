#!/usr/bin/env python3
"""Materialise one frozen causal feature contract from target-free path identities.

The archived path ledger carries realised fields, but this producer reads only
the decision-time identity columns.  It fixes the entire candidate population
before invoking the causal feature engine; no label, future bar, path
completeness, or outcome column can decide whether a row is materialised.

It is intended for historical warm-up gaps where an existing frozen feature
contract must be recovered without changing its field order or semantics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_tp6_sl4_exact170_canonical_consensus import (  # noqa: E402
    FROZEN_GENERATION_DEPENDENCIES,
    materialize_features,
)


SCHEMA = "strict_r3_path_identity_frozen_features_v1"
GENERATION_ONLY = ("rv_120h",)
IDENTITY_COLUMNS = ("candidate_id", "__decision_ts__", "side_name", "__symbol__")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json_exclusive(path: Path, payload: object) -> None:
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)


def _months(text: str) -> tuple[pd.Timestamp, ...]:
    values = tuple(pd.Timestamp(f"{token.strip()}-01", tz="UTC") for token in text.split(",") if token.strip())
    if not values or tuple(sorted(values)) != values or len(set(values)) != len(values):
        raise ValueError("--months must contain unique chronological YYYY-MM values")
    return values


def _fields(path: Path, key: str | None) -> tuple[tuple[str, ...], str]:
    payload: Any = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError("feature-contract JSON must be an object")
    keys = (key,) if key else ("selected_features", "feature_contract", "features")
    used = next((candidate for candidate in keys if isinstance(payload.get(candidate), list)), None)
    if used is None:
        raise ValueError(f"{path}: no list under {keys}")
    fields = tuple(str(value) for value in payload[used])
    if not fields or len(fields) != len(set(fields)):
        raise AssertionError("frozen field contract must be non-empty and unique")
    return fields, used


def _identities(path_root: Path, month: pd.Timestamp) -> tuple[pd.DataFrame, list[Path]]:
    parts: list[pd.DataFrame] = []
    used: list[Path] = []
    for token in (month - pd.offsets.MonthBegin(1), month):
        path = path_root / f"month={token:%Y-%m}" / "side=long.parquet"
        if not path.is_file():
            raise FileNotFoundError(path)
        part = pd.read_parquet(path, columns=list(IDENTITY_COLUMNS))
        parts.append(part)
        used.append(path)
    source = pd.concat(parts, ignore_index=True)
    source["__decision_ts__"] = pd.to_datetime(source["__decision_ts__"], utc=True, errors="raise")
    start, end = month, month + pd.offsets.MonthBegin(1)
    source = source.loc[source["__decision_ts__"].ge(start) & source["__decision_ts__"].lt(end)].copy()
    if source.empty or source["candidate_id"].duplicated().any():
        raise AssertionError(f"{month:%Y-%m}: invalid target-free identity population")
    if not source["side_name"].astype(str).str.lower().eq("long").all():
        raise AssertionError(f"{month:%Y-%m}: expected long-only path identities")
    output = source.loc[:, list(IDENTITY_COLUMNS)].copy()
    output["__ts__"] = output["__decision_ts__"] - pd.Timedelta(hours=1)
    output["__symbol__"] = output["__symbol__"].astype(str)
    return output, used


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path-identity-root", type=Path, required=True)
    parser.add_argument("--feature-contract-json", type=Path, required=True)
    parser.add_argument("--feature-key")
    parser.add_argument("--months", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--warmup-days", type=int, default=180)
    args = parser.parse_args()
    if args.warmup_days < 30:
        raise ValueError("--warmup-days must be at least 30")
    months = _months(args.months)
    fields, field_key = _fields(args.feature_contract_json.resolve(), args.feature_key)
    out = args.out.resolve()
    if out.exists():
        raise FileExistsError(f"immutable output exists: {out}")
    out.mkdir(parents=True)
    requested = tuple(dict.fromkeys((*fields, *GENERATION_ONLY)))
    audits: list[dict[str, object]] = []
    source_files: list[Path] = []
    for month in months:
        identities, used = _identities(args.path_identity_root.resolve(), month)
        source_files.extend(used)
        month_root = out / f"month={month:%Y-%m}"
        generated_path = materialize_features(
            month_root,
            identities,
            {"long": list(requested), "short": []},
            month - pd.Timedelta(days=args.warmup_days),
            month + pd.offsets.MonthBegin(1),
            full_feature_universe=False,
        )
        canonical = month_root / "causal_feature_universe.parquet"
        os.replace(generated_path, canonical)
        generated = pd.read_parquet(canonical, columns=["__ts__", "__symbol__", *fields])
        restored = identities.loc[:, ["candidate_id", "__decision_ts__", "side_name", "__ts__", "__symbol__"]].merge(
            generated, on=["__ts__", "__symbol__"], how="inner", validate="one_to_one",
        )
        if len(restored) != len(identities) or restored["candidate_id"].duplicated().any():
            raise AssertionError(f"{month:%Y-%m}: causal feature engine changed target-free identities")
        restored.to_parquet(canonical, index=False, compression="zstd")
        audit = {
            "month": f"{month:%Y-%m}", "identity_rows": int(len(identities)),
            "feature_rows": int(len(restored)), "feature_count": len(fields),
            "finite_coverage_mean": float(restored.loc[:, list(fields)].notna().mean().mean()),
            "target_free": True,
        }
        audits.append(audit)
        print(json.dumps({"event": "month_complete", **audit}), flush=True)
    pd.DataFrame(audits).to_parquet(out / "identity_and_coverage_audit.parquet", index=False, compression="zstd")
    _write_json_exclusive(out / "run_manifest.json", {
        "schema": SCHEMA,
        "scope": "offline target-free causal feature materialisation; no labels, models, MC1, admission, portfolio, inference, live, or exchange mutation",
        "candidate_contract": "archived path source is read only for identity/timestamp/symbol/side before features are generated; every identity is retained",
        "path_identity_root": str(args.path_identity_root.resolve()),
        "path_identity_sha256": hashlib.sha256("".join(_sha256(path) for path in sorted(set(source_files))).encode()).hexdigest(),
        "feature_contract_json": str(args.feature_contract_json.resolve()),
        "feature_contract_sha256": _sha256(args.feature_contract_json.resolve()),
        "feature_key": field_key,
        "feature_contract": list(fields),
        "feature_contract_sha256_ordered": hashlib.sha256("\n".join(fields).encode()).hexdigest(),
        "generation_only_dependencies": list(GENERATION_ONLY),
        "generation_dependencies": list(FROZEN_GENERATION_DEPENDENCIES),
        "warmup_days": int(args.warmup_days),
        "months": [f"{month:%Y-%m}" for month in months],
        "audit": audits,
        "correctness": {"identities_fixed_before_generation": True, "no_outcome_column_read": True, "no_candidate_filter_from_future_path": True},
    })


if __name__ == "__main__":
    main()
