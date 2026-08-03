#!/usr/bin/env python3
"""Seal a score-only PIT readiness audit for the December 2025 common30 bridge.

The candidate rows are hourly.  Exact 1m paths are read only to verify their
identity; no label value is used by this preflight.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from audit_augnov2025_pit_scoring_preflight import (
    AE,
    IDENTITY,
    NATIVE,
    PROMOTION,
    PreflightError,
    _feature_coverage,
    _identity_digest,
    _identity_match,
    _load_contracts,
    _sha256,
)

INPUT = ROOT / "data_perp/artifacts/dec2025_common30_exact1m_backfill_inputs_20260730_v1/candidates.parquet"
EXECUTION = ROOT / "data_perp/artifacts/dec2025_execution_ev_common30_exact1m_labels_20260730_v1/labels.parquet"
OUT = ROOT / "data_perp/artifacts/dec2025_common30_pit_scoring_preflight_20260730_v1"


def _write(path: Path, value: dict) -> None:
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False, default=str)
        handle.write("\n")
    os.replace(temporary, path)


def run(output: Path = OUT) -> Path:
    output = Path(output)
    if output.exists():
        raise PreflightError(f"refusing to overwrite sealed audit: {output}")
    candidates = pd.read_parquet(INPUT, columns=list(IDENTITY))
    candidates["__ts__"] = pd.to_datetime(candidates["__ts__"], utc=True, errors="raise")
    if (len(candidates) != 44_640 or candidates["candidate_id"].duplicated().any()
            or not candidates["__ts__"].dt.strftime("%Y-%m").eq("2025-12").all()
            or not candidates["__ts__"].dt.minute.eq(0).all()):
        raise PreflightError("December common30 candidate identity is not exact hourly 44,640-row scope")
    execution = pd.read_parquet(EXECUTION, columns=list(IDENTITY))
    execution["__ts__"] = pd.to_datetime(execution["__ts__"], utc=True, errors="raise")
    native = pd.concat([
        pd.read_parquet(NATIVE / f"train_global_{side}_5_2025_12.parquet", columns=list(IDENTITY))
        for side in ("long", "short")
    ], ignore_index=True)
    native["__ts__"] = pd.to_datetime(native["__ts__"], utc=True, errors="raise")
    native_exact, native_missing = _identity_match(candidates, native)
    execution_exact, execution_missing = _identity_match(candidates, execution)
    routes = _load_contracts(PROMOTION, AE)
    coverage = {
        side: _feature_coverage(candidates.loc[candidates.side_name.eq(side)].reset_index(drop=True), side=side, route=routes[side])
        for side in ("long", "short")
    }
    records = []
    for side in ("long", "short"):
        for item in coverage[side]["coverage_rows"]:
            sample = candidates.loc[(candidates.side_name.eq(side)) & (candidates.__symbol__.eq(item["symbol"]))]
            n_ok, n_bad = _identity_match(sample, native)
            e_ok, e_bad = _identity_match(sample, execution)
            records.append({**item, "native_identity_rows": n_ok, "native_identity_missing_rows": n_bad,
                            "execution_identity_rows": e_ok, "execution_identity_missing_rows": e_bad})
    all_pit = all(coverage[s]["exact_pit_key_fraction"] == 1.0 for s in coverage)
    all_base = all(coverage[s]["base_joint_complete_fraction"] == 1.0 for s in coverage)
    all_residual = all(coverage[s]["residual_raw_joint_complete_fraction"] == 1.0 for s in coverage)
    ready = native_missing == 0 and execution_missing == 0 and all_pit and all_base and all_residual
    report = {
        "schema": "dec2025_common30_pit_scoring_preflight_v1",
        "status": "SEALED_SCORE_ONLY_PIT_PREFLIGHT_NON_PROMOTION",
        "scope": "December 2025 common30; strict hourly score identity; 1m exact execution label identity only",
        "model_sample_cadence": "1h",
        "exact_replay_bar_cadence": "1m_labels_only",
        "candidate_rows": int(len(candidates)),
        "candidate_identity_sha256": _identity_digest(candidates),
        "native_base_identity": {"exact_rows": native_exact, "missing_or_mismatched_rows": native_missing},
        "exact_execution_label_identity": {"exact_rows": execution_exact, "missing_or_mismatched_rows": execution_missing},
        "feature_coverage": coverage,
        "score_materialization": {
            "data_and_pit_feasible": ready,
            "frozen_model_requirement": "score only with a base/residual fit frozen before 2025-12-01; no December native or execution outcome may enter that fit",
            "not_authorized": "no mapping, calibration, policy selection, or model fitting on December execution labels",
        },
        "inputs_sha256": {"candidates": _sha256(INPUT), "execution_labels": _sha256(EXECUTION), "promotion": _sha256(PROMOTION)},
    }
    output.mkdir(parents=True)
    pd.DataFrame(records).sort_values(["month", "side_name", "symbol"], kind="stable").to_csv(output / "coverage_by_month_side_symbol.csv", index=False)
    _write(output / "readiness_report.json", report)
    manifest = {"schema": report["schema"], "status": report["status"], "model_sample_cadence": "1h", "exact_replay_bar_cadence": "1m_labels_only",
                "outputs_sha256": {name: _sha256(output / name) for name in ("coverage_by_month_side_symbol.csv", "readiness_report.json")}}
    _write(output / "manifest.json", manifest)
    (output / "manifest.sha256").write_text(f"{_sha256(output / 'manifest.json')}  manifest.json\n")
    return output


if __name__ == "__main__":
    print(run())
