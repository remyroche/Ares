#!/usr/bin/env python3
"""Audit observable residual-state coverage and local feature retention."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from scripts.run_global_residual_latent_state import (
    CALENDAR_OBSERVABLE_FAMILY_PATTERNS,
)


def _safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _family_features(columns: list[str]) -> dict[str, list[str]]:
    output: dict[str, list[str]] = {}
    for family, patterns in CALENDAR_OBSERVABLE_FAMILY_PATTERNS.items():
        output[family] = [
            name
            for name in columns
            if any(pattern in name.lower() for pattern in patterns)
            and not name.startswith(("target_", "placebo_target_"))
        ]
    return output


def audit_state_contract(
    states_path: Path,
    manifest_path: Path,
    *,
    local_relevance_path: Path | None = None,
    min_month_coverage: float = 0.50,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    schema = [str(name) for name in pq.ParquetFile(states_path).schema_arrow.names]
    family_columns = _family_features(schema)
    requested = sorted(
        {
            name
            for names in family_columns.values()
            for name in names
            if name in schema
        }
    )
    columns = [name for name in ("__ts__", "side_name") if name in schema] + requested
    states = pd.read_parquet(states_path, columns=columns)
    states["month"] = pd.to_datetime(states["__ts__"], utc=True).dt.strftime("%Y-%m")
    coverage_rows: list[dict[str, Any]] = []
    for month, local in states.groupby("month", observed=True, sort=True):
        for family, names in family_columns.items():
            values = (
                local[names].apply(pd.to_numeric, errors="coerce")
                if names
                else pd.DataFrame(index=local.index)
            )
            finite = np.isfinite(values.to_numpy(dtype=np.float32, copy=False))
            coverage_rows.append(
                {
                    "month": str(month),
                    "family": family,
                    "state_rows": int(len(local)),
                    "feature_count": int(len(names)),
                    "mean_feature_coverage": float(finite.mean()) if finite.size else 0.0,
                    "max_feature_coverage": float(finite.mean(axis=0).max())
                    if finite.size
                    else 0.0,
                    "features": "|".join(names),
                }
            )
    coverage = pd.DataFrame(coverage_rows)

    local_rows: list[dict[str, Any]] = []
    if local_relevance_path is not None and local_relevance_path.exists():
        relevance = pd.read_csv(local_relevance_path)
        selected_mask = (
            relevance["selected"].fillna(False).astype(bool)
            if "selected" in relevance
            else pd.Series(False, index=relevance.index)
        )
        selected = relevance.loc[selected_mask].copy()
        partition_cols = [
            name
            for name in ("side_name", "archetype_policy_key", "state_partition_token")
            if name in selected
        ]
        if partition_cols:
            for keys, part in selected.groupby(partition_cols, observed=True, sort=True):
                keys = keys if isinstance(keys, tuple) else (keys,)
                identity = dict(zip(partition_cols, keys, strict=True))
                selected_names = set(part["feature"].astype(str))
                for family, names in family_columns.items():
                    retained = sorted(selected_names.intersection(names))
                    local_rows.append(
                        {
                            **identity,
                            "family": family,
                            "retained_feature_count": int(len(retained)),
                            "retained_features": "|".join(retained),
                        }
                    )
    local_retention = pd.DataFrame(local_rows)

    family_summary: dict[str, Any] = {}
    for family in CALENDAR_OBSERVABLE_FAMILY_PATTERNS:
        rows = coverage.loc[coverage["family"].eq(family)]
        failed_months = rows.loc[
            rows["max_feature_coverage"].lt(float(min_month_coverage)), "month"
        ].tolist()
        family_summary[family] = {
            "state_feature_count": int(len(family_columns.get(family, []))),
            "minimum_month_max_coverage": float(rows["max_feature_coverage"].min())
            if not rows.empty
            else 0.0,
            "failed_months": failed_months,
            "pass": bool(len(family_columns.get(family, [])) and not failed_months),
        }
    summary = {
        "schema": "residual_state_observable_contract_audit_v1",
        "states": str(states_path),
        "manifest": str(manifest_path),
        "rows": int(len(states)),
        "start": str(pd.to_datetime(states["__ts__"], utc=True).min()),
        "end": str(pd.to_datetime(states["__ts__"], utc=True).max()),
        "fit_partition": (manifest.get("raw_feature_preselection") or {}).get(
            "fit_partition"
        ),
        "required_family_min_month_coverage": float(min_month_coverage),
        "families": family_summary,
        "all_required_families_pass": bool(
            family_summary and all(row["pass"] for row in family_summary.values())
        ),
    }
    return coverage, local_retention, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--states", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--local-relevance", type=Path, default=None)
    parser.add_argument("--min-month-coverage", type=float, default=0.50)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = args.manifest or args.states.with_suffix(".manifest.json")
    args.output.mkdir(parents=True, exist_ok=True)
    coverage, retention, summary = audit_state_contract(
        args.states,
        manifest,
        local_relevance_path=args.local_relevance,
        min_month_coverage=float(args.min_month_coverage),
    )
    coverage.to_csv(args.output / "observable_family_monthly_coverage.csv", index=False)
    retention.to_csv(args.output / "local_partition_family_retention.csv", index=False)
    (args.output / "manifest.json").write_text(
        json.dumps(_safe(summary), indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(_safe(summary), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
