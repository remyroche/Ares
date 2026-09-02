#!/usr/bin/env python3
"""Freeze a causal full-universe feature contract for Router50 research.

The Router selector must not discover features from labels, outcomes, or a
held-period score panel.  This utility therefore consumes only the immutable,
target-free feature panels and their per-month coverage sidecars.  It enforces
the Phase-J hygiene gates before any target-aware screen is allowed to start:

* every requested month has exactly one explicit causal source;
* the source schema is stable and contains no outcome-like field;
* global coverage is at least 95% and every available month is at least 90%;
* constants are rejected without treating intentionally discrete state fields
  as automatically invalid.

It is deliberately label-free and research-only.  The resulting ordered
contract is an input to later fold-local, target-aware Router selection; it is
not an inference bundle and never mutates Base, consensus, MC1, or live state.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Iterable

import pandas as pd


IDENTITY_COLUMNS = {
    "candidate_id", "__decision_ts__", "side_name", "__ts__", "__symbol__",
}
# These are an explicit fail-closed check rather than a name-based feature
# selector.  The source manifest is the primary provenance proof; the tokens
# catch an accidental label merge before a target-aware model ever sees it.
# Do not reject causal market-language fields such as ``realized_volatility``
# or ``reversion_target_distance`` merely because of an ambiguous word.  The
# materialisation manifest proves that source panels are target-free; this
# additional guard only blocks identifiers that unambiguously describe our
# supervised outcome ledger or future path labels.
PROHIBITED_OUTCOME_TOKENS = (
    "policy_net", "policy_gross", "policy_label", "label_available", "path_valid",
    "future_path", "h12_outcome", "exact_net", "realized_net", "realised_net",
    "mfe_h12", "mae_h12", "aux_time_to_", "aux_reached_", "first_policy_hit",
)
SCHEMA = "strict_r3_router_full_universe_hygiene_v1"


def _write_once(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)


def _hash_lines(values: Iterable[str]) -> str:
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


def _parse_roots(value: str) -> tuple[Path, ...]:
    roots = tuple(Path(token.strip()).resolve() for token in value.split(",") if token.strip())
    if not roots or len(set(roots)) != len(roots):
        raise ValueError("feature roots must be a non-empty unique comma-separated list")
    for root in roots:
        if not root.exists():
            raise FileNotFoundError(root)
    return roots


def _available_months(roots: tuple[Path, ...]) -> list[str]:
    months: set[str] = set()
    for root in roots:
        for path in root.glob("month=*"):
            token = path.name.removeprefix("month=")
            if (path / "causal_feature_universe.parquet").exists() and (path / "feature_coverage.parquet").exists():
                months.add(token)
    result = sorted(months)
    if not result:
        raise FileNotFoundError("no causal_feature_universe/feature_coverage month pairs found")
    return result


def _resolve_month(roots: tuple[Path, ...], month: str) -> Path:
    matches = [root / f"month={month}" for root in roots
               if (root / f"month={month}" / "causal_feature_universe.parquet").exists()
               and (root / f"month={month}" / "feature_coverage.parquet").exists()]
    if len(matches) != 1:
        raise AssertionError(f"{month}: expected exactly one causal source, found {len(matches)}")
    return matches[0]


def _schema(month_root: Path) -> tuple[str, ...]:
    columns = pd.read_parquet(month_root / "causal_feature_universe.parquet").head(0).columns.tolist()
    fields = tuple(column for column in columns if column not in IDENTITY_COLUMNS)
    if not fields:
        raise AssertionError(f"{month_root}: no non-identity causal features")
    bad = [field for field in fields if any(token in field.lower() for token in PROHIBITED_OUTCOME_TOKENS)]
    if bad:
        raise AssertionError(f"{month_root}: outcome-like fields unexpectedly present: {bad[:10]}")
    return fields


def run(*, roots: tuple[Path, ...], months: list[str], out: Path,
        min_global_coverage: float, min_month_coverage: float) -> None:
    if out.exists():
        raise FileExistsError(f"immutable artifact already exists: {out}")
    if not (0.0 < min_month_coverage <= min_global_coverage <= 1.0):
        raise ValueError("coverage gates must satisfy 0 < monthly <= global <= 1")

    resolved = {month: _resolve_month(roots, month) for month in months}
    schemas = {month: _schema(path) for month, path in resolved.items()}
    first_month = months[0]
    canonical = schemas[first_month]
    canonical_set = set(canonical)
    for month, fields in schemas.items():
        if set(fields) != canonical_set:
            missing = sorted(canonical_set - set(fields))
            extra = sorted(set(fields) - canonical_set)
            raise AssertionError(
                f"{month}: causal feature schema mismatch; missing={missing[:8]} extra={extra[:8]}"
            )

    records: list[pd.DataFrame] = []
    for month, path in resolved.items():
        coverage = pd.read_parquet(path / "feature_coverage.parquet")
        required = {"feature", "rows", "finite_rows", "finite_fraction", "n_unique"}
        if not required.issubset(coverage.columns):
            raise AssertionError(f"{month}: invalid coverage sidecar columns")
        coverage = coverage.loc[coverage["feature"].isin(canonical_set), list(required)].copy()
        if len(coverage) != len(canonical_set) or coverage["feature"].duplicated().any():
            raise AssertionError(f"{month}: coverage sidecar does not cover the frozen schema exactly")
        coverage["month"] = month
        records.append(coverage)
    table = pd.concat(records, ignore_index=True)
    table["rows"] = pd.to_numeric(table["rows"], errors="raise").astype("int64")
    table["finite_rows"] = pd.to_numeric(table["finite_rows"], errors="raise").astype("int64")
    table["finite_fraction"] = pd.to_numeric(table["finite_fraction"], errors="raise").astype(float)
    table["n_unique"] = pd.to_numeric(table["n_unique"], errors="raise").astype("int64")
    if table["finite_rows"].gt(table["rows"]).any() or table["finite_fraction"].lt(0).any() or table["finite_fraction"].gt(1).any():
        raise AssertionError("invalid finite coverage counts")

    summary = table.groupby("feature", sort=False).agg(
        rows_total=("rows", "sum"), finite_rows_total=("finite_rows", "sum"),
        coverage_min=("finite_fraction", "min"), coverage_median=("finite_fraction", "median"),
        coverage_max=("finite_fraction", "max"), n_unique_min=("n_unique", "min"),
        n_unique_max=("n_unique", "max"),
    ).reindex(canonical).reset_index()
    summary["coverage_global"] = summary["finite_rows_total"] / summary["rows_total"].clip(lower=1)
    # Do not throw away a valid binary/ordinal state feature merely because it
    # has few values.  A true constant is always unusable, while all remaining
    # low-variance checks are performed fold-locally by the selector.
    summary["non_constant"] = summary["n_unique_max"].gt(1)
    summary["hygiene_keep"] = (
        summary["coverage_global"].ge(min_global_coverage)
        & summary["coverage_min"].ge(min_month_coverage)
        & summary["non_constant"]
    )
    kept = summary.loc[summary["hygiene_keep"], "feature"].tolist()
    if len(kept) < 100:
        raise AssertionError(f"hygiene left only {len(kept)} features; source contract is unexpectedly incomplete")

    out.mkdir(parents=True)
    table.to_parquet(out / "monthly_feature_coverage.parquet", index=False, compression="zstd")
    summary.to_parquet(out / "feature_hygiene.parquet", index=False, compression="zstd")
    _write_once(out / "feature_contract.json", {
        "schema": SCHEMA,
        "scope": "offline Router50 causal feature hygiene only; no labels/outcomes/model scores/live state",
        "feature_contract": kept,
        "feature_contract_sha256": _hash_lines(kept),
        "source_schema_sha256": _hash_lines(canonical),
        "source_feature_count": len(canonical),
        "eligible_feature_count": len(kept),
        "months": months,
        "source_roots": [str(root) for root in roots],
        "coverage_gates": {
            "global": min_global_coverage,
            "every_month": min_month_coverage,
            "constant": "n_unique_max <= 1 rejected; discrete state fields retained",
        },
        "causality": {
            "source": "target-free causal feature panels only",
            "outcome_like_name_check": list(PROHIBITED_OUTCOME_TOKENS),
            "ambiguous_month_sources": "fail closed",
        },
    })
    _write_once(out / "run_manifest.json", {
        "schema": SCHEMA,
        "status": "complete",
        "scope": "offline Router50 feature hygiene only; no live/exchange mutation",
        "months": months,
        "source_feature_count": len(canonical),
        "eligible_feature_count": len(kept),
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-roots", required=True, help="comma-separated disjoint causal feature roots")
    parser.add_argument("--months", help="optional chronological comma-separated YYYY-MM list; default all available")
    parser.add_argument("--min-global-coverage", type=float, default=.95)
    parser.add_argument("--min-month-coverage", type=float, default=.90)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    roots = _parse_roots(args.feature_roots)
    available = _available_months(roots)
    months = [token.strip() for token in args.months.split(",") if token.strip()] if args.months else available
    if not months or months != sorted(set(months)) or any(month not in available for month in months):
        raise ValueError("months must be unique chronological source months")
    run(roots=roots, months=months, out=args.out.resolve(),
        min_global_coverage=args.min_global_coverage, min_month_coverage=args.min_month_coverage)


if __name__ == "__main__":
    main()
