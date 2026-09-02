#!/usr/bin/env python3
"""Freeze a coverage-audited broad-market feature contract for P8u magnitude.

The magnitude head is a slow market-conversion correction.  It keeps the F72
Base context but augments it only with named, causal registry families from
``extreme_price_movements.config``.  This utility never reads labels or
outcomes; it emits an immutable feature-contract receipt plus per-month
coverage evidence for the target-free panels used by the OOF screen.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import pyarrow.parquet as pq

from extreme_price_movements import config as feature_config
import run_strict_r3_p8u_meta_target_query_grid_v1 as screen


SCHEMA = "strict_r3_p8u_magnitude_broad_market_contract_v1"


def _once(path: Path, payload: object) -> None:
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _roots(raw: dict[str, object]) -> tuple[Path, ...]:
    source = raw["source"]
    return tuple(ROOT / str(value) for value in source["full_feature_roots"])


def _full_path(roots: tuple[Path, ...], month: pd.Timestamp) -> Path:
    found = [root / f"month={month:%Y-%m}" / "causal_feature_universe.parquet" for root in roots]
    existing = [path for path in found if path.exists()]
    if len(existing) != 1:
        raise AssertionError(f"{month:%Y-%m}: expected exactly one full causal feature owner, got {len(existing)}")
    return existing[0]


def run(*, config_path: Path, out: Path, minimum_coverage: float, last_covered_month: str) -> Path:
    if out.exists():
        raise FileExistsError(f"immutable output exists: {out}")
    raw = json.loads(config_path.read_text())
    source = raw["source"]
    inherited_f72_fields = screen._read_selection(ROOT / str(source["base_f72_contract"]))
    # This is a market-state magnitude head, not a funding or order-book
    # context head.  Remove inherited aliases as well as refusing new ones;
    # otherwise the stated feature boundary would be misleading.
    def _is_excluded_context(field: str) -> bool:
        lowered = field.lower()
        return (
            "fund" in lowered
            or "orderbook" in lowered
            or lowered.startswith("ob_")
            or "__ob_" in lowered
        )

    excluded_legacy_funding = tuple(
        field for field in inherited_f72_fields if "fund" in field.lower()
    )
    excluded_legacy_orderbook = tuple(
        field for field in inherited_f72_fields
        if _is_excluded_context(field) and "fund" not in field.lower()
    )
    base_fields = tuple(field for field in inherited_f72_fields if not _is_excluded_context(field))
    if not excluded_legacy_funding:
        raise AssertionError("expected explicit inherited funding fields to exclude")
    registered = tuple(feature_config.MAGNITUDE_BROAD_MARKET_META_FEATURE_KEYS)
    # Continuous-regime fields have a separate strict-prequential candidate
    # sidecar.  Soft five-state/transition outputs are registered for
    # inference, but were not materialised for this historical panel; do not
    # substitute or impute them in an OOF research result.
    continuous_fields = tuple(
        field for field in feature_config.CAUSAL_CONTINUOUS_REGIME_META_FEATURE_KEYS
        if "funding" not in field.lower()
    )
    soft_transition_fields = tuple(feature_config.CAUSAL_SOFT_REGIME_TRANSITION_META_FEATURE_KEYS)
    panel_registered = tuple(
        field for field in registered
        if field not in set(continuous_fields)
        and field not in set(soft_transition_fields)
        and not _is_excluded_context(field)
    )
    candidate_fields = tuple(dict.fromkeys((*base_fields, *panel_registered)))
    all_months = tuple(pd.Timestamp(value, tz="UTC") for value in (
        "2025-08-01", "2025-09-01", "2025-10-01", "2025-11-01", "2025-12-01",
        "2026-01-01", "2026-02-01", "2026-03-01", "2026-04-01", "2026-05-01",
        "2026-06-01", "2026-07-01",
    ))
    last = pd.Timestamp(f"{last_covered_month}-01", tz="UTC")
    months = tuple(month for month in all_months if month <= last)
    if not months or months[-1] != last:
        raise ValueError("--last-covered-month is outside the declared historical panel")
    roots = _roots(raw)
    continuous_sidecar = ROOT / "data_perp/artifacts/strict_r3_continuous_context_long_dec2024_jul2026_20260811_v1/candidate_causal_continuous_context.parquet"
    if not continuous_sidecar.exists():
        raise FileNotFoundError(continuous_sidecar)
    schema_presence: dict[str, bool] = {field: True for field in candidate_fields}
    coverage: dict[str, list[float]] = {field: [] for field in candidate_fields}
    audits: list[dict[str, object]] = []
    sidecar_coverage: dict[str, list[float]] = {field: [] for field in continuous_fields}
    for month in months:
        path = _full_path(roots, month)
        names = set(pq.ParquetFile(path).schema_arrow.names)
        present = [field for field in candidate_fields if field in names]
        absent = [field for field in candidate_fields if field not in names]
        for field in absent:
            schema_presence[field] = False
        frame = pd.read_parquet(path, columns=list(screen.IDENTITY) + present)
        row: dict[str, object] = {"month": f"{month:%Y-%m}", "rows": int(len(frame)), "source": str(path)}
        for field in candidate_fields:
            value = float(frame[field].notna().mean()) if field in frame else 0.0
            coverage[field].append(value)
            row[field] = value
        audits.append(row)
        sidecar = pd.read_parquet(
            continuous_sidecar,
            columns=["candidate_id", "__ts__", "__symbol__", "side_name", "source_utc", *continuous_fields],
            filters=[("__ts__", ">=", month), ("__ts__", "<", month + pd.offsets.MonthBegin(1))],
        )
        sidecar["__ts__"] = pd.to_datetime(sidecar["__ts__"], utc=True, errors="raise")
        sidecar["source_utc"] = pd.to_datetime(sidecar["source_utc"], utc=True, errors="raise")
        decision_identity = ["__symbol__", "__ts__", "side_name"]
        if sidecar.duplicated(decision_identity).any() or not sidecar.source_utc.le(sidecar.__ts__).all():
            raise AssertionError(f"{month:%Y-%m}: invalid or post-date continuous-regime sidecar")
        base_identity = pd.read_parquet(screen._base_path(ROOT / str(source["base_target_free_root"]), month), columns=list(screen.IDENTITY))
        base_identity["__symbol__"] = base_identity.candidate_id.astype(str).str.split("|", n=1).str[0]
        base_identity = base_identity.rename(columns={"__decision_ts__": "__ts__"})
        if base_identity.duplicated(decision_identity).any():
            raise AssertionError(f"{month:%Y-%m}: non-unique Base decision identity")
        sidecar = base_identity.loc[:, decision_identity].merge(
            sidecar.loc[:, [*decision_identity, *continuous_fields]], on=decision_identity, how="left", validate="one_to_one",
        )
        sidecar_row: dict[str, object] = {"month": f"{month:%Y-%m}", "rows": int(len(sidecar)), "source": str(continuous_sidecar)}
        for field in continuous_fields:
            value = float(sidecar[field].notna().mean())
            sidecar_coverage[field].append(value)
            sidecar_row[field] = value
        audits.append({f"sidecar_{key}": value for key, value in sidecar_row.items()})
    audit = pd.DataFrame(audits)
    # F72 is the pre-existing frozen Base context: retain it byte-for-byte,
    # even where its historical availability is below the stricter addition
    # gate.  The Meta learner's fold-local imputer handles those predeclared
    # legacy values exactly as it did in the F72 control.  The 90% rule is for
    # *new* broad-market additions only, so a sparse legacy field can never
    # smuggle a new context into this market-state experiment.
    broad_candidates = tuple(field for field in candidate_fields if field not in set(base_fields))
    broad_kept = tuple(
        field for field in broad_candidates
        if schema_presence[field] and min(coverage[field]) >= float(minimum_coverage)
    )
    continuous_kept = tuple(
        field for field in continuous_fields
        if min(sidecar_coverage[field]) >= float(minimum_coverage)
    )
    kept = tuple(dict.fromkeys((*base_fields, *broad_kept, *continuous_kept)))
    base_below_gate = sorted(
        field for field in base_fields
        if not schema_presence[field] or min(coverage[field]) < float(minimum_coverage)
    )
    if not broad_kept:
        raise AssertionError("no broad market fields survived the coverage contract")
    out.mkdir(parents=True)
    audit.to_parquet(out / "field_coverage_by_month.parquet", index=False, compression="zstd")
    selection = {
        "schema": SCHEMA,
        "scope": "target-free causal broad-market magnitude Meta feature contract",
        "selected_features": list(kept),
        "base_context_fields": list(base_fields),
        "inherited_f72_fields": list(inherited_f72_fields),
        "excluded_legacy_funding_fields": list(excluded_legacy_funding),
        "excluded_legacy_orderbook_fields": list(excluded_legacy_orderbook),
        "broad_market_fields": list(broad_kept),
        "candidate_feature_count": len(candidate_fields),
        "selected_feature_count": len(kept),
        "selected_broad_market_feature_count": len(broad_kept),
        "selected_continuous_regime_feature_count": len(continuous_kept),
        "continuous_regime_sidecar_fields": list(continuous_kept),
        "continuous_regime_sidecar_source": str(continuous_sidecar),
        "continuous_regime_fields_unmaterialized_for_comparable_panel": list(
            field for field in continuous_fields if field not in set(continuous_kept)
        ),
        "soft_transition_fields_unmaterialized_for_historical_panel": list(soft_transition_fields),
        "legacy_base_fields_below_broad_addition_coverage_gate": base_below_gate,
        "minimum_coverage_each_month": float(minimum_coverage),
        "coverage_months": [f"{month:%Y-%m}" for month in months],
        "registry_family": "MAGNITUDE_BROAD_MARKET_META_FEATURE_KEYS",
        "causality": "named decision-time feature registry only; labels/outcomes are not read by this utility",
        "config_source": str(config_path),
        "config_sha256": _sha(config_path),
    }
    _once(out / "feature_contract.json", selection)
    _once(out / "correctness_report.json", {
        "target_free_feature_panels_only": True,
        "named_config_registry_only": True,
        "all_new_broad_fields_present_every_coverage_month": True,
        "all_new_broad_fields_meet_minimum_coverage_each_month": True,
        "continuous_regime_sidecar_is_exact_identity_and_prior_only": True,
        "soft_transition_outputs_not_silently_imputed": True,
        "frozen_nonfunding_base_context_retained": True,
        "no_funding_field_in_final_contract": not any("fund" in field.lower() for field in kept),
        "no_orderbook_field_in_final_contract": not any(
            _is_excluded_context(field) and "fund" not in field.lower()
            for field in kept
        ),
        "no_policy_path_or_outcome_read": True,
    })
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--minimum-coverage", type=float, default=0.90)
    parser.add_argument("--last-covered-month", default="2026-06")
    args = parser.parse_args()
    print(run(
        config_path=args.config.resolve(), out=args.out.resolve(), minimum_coverage=float(args.minimum_coverage),
        last_covered_month=str(args.last_covered_month),
    ))


if __name__ == "__main__":
    main()
