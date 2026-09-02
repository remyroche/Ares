#!/usr/bin/env python3
"""Materialize the frozen causal raw-feature universe for capture screening."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_conditional_gmm_feature_selection import (  # noqa: E402
    _load_feature_store_columns,
)

IDENTITY = ("__ts__", "__symbol__", "side_name", "candidate_id")
PREFIX = "capture_candidate__"
SCHEMA = "exact_policy_capture_causal_feature_universe_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def prefixed_feature_names(features: list[str]) -> list[str]:
    return [f"{PREFIX}{feature}" for feature in features]


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    frame = pd.read_parquet(args.input)
    if frame.duplicated(list(IDENTITY)).any():
        raise ValueError("canonical input contains duplicate identities")
    universe = json.loads(args.frozen_feature_contract.read_text())
    features = [str(value) for value in universe["feature_columns"]]
    if len(features) != len(set(features)):
        raise ValueError("raw feature universe contains duplicate names")
    matrix, store_report = _load_feature_store_columns(
        frame,
        feature_dir=args.feature_store,
        selected_features=features,
    )
    matrix = matrix.rename(
        columns=dict(zip(features, prefixed_feature_names(features)))
    )
    result = pd.concat(
        [frame.reset_index(drop=True), matrix.reset_index(drop=True)], axis=1
    )
    coverage_rows = []
    for side in ("long", "short"):
        side_mask = result["side_name"].astype(str).eq(side)
        for raw, column in zip(features, prefixed_feature_names(features)):
            values = pd.to_numeric(
                result.loc[side_mask, column], errors="coerce"
            ).to_numpy(dtype=float)
            finite = np.isfinite(values)
            coverage_rows.append(
                {
                    "side_name": side,
                    "raw_feature": raw,
                    "column": column,
                    "rows": int(len(values)),
                    "finite_rows": int(finite.sum()),
                    "finite_fraction": float(finite.mean()),
                    "finite_variance": (
                        float(np.var(values[finite])) if finite.any() else np.nan
                    ),
                }
            )
    coverage = pd.DataFrame(coverage_rows)
    eligible = (
        coverage.groupby("column")["finite_fraction"].min()
        >= float(args.minimum_full_period_coverage)
    )
    eligible_columns = eligible.index[eligible].tolist()
    args.output_dir.mkdir(parents=True)
    output = args.output_dir / "capture_feature_universe.parquet"
    coverage_output = args.output_dir / "feature_coverage.csv"
    feature_manifest = args.output_dir / "feature_universe_manifest.json"
    manifest_output = args.output_dir / "manifest.json"
    result.to_parquet(output, index=False, compression="zstd")
    coverage.to_csv(coverage_output, index=False)
    feature_payload = {
        "schema": "capture_candidate_feature_universe_v1",
        "candidate_feature_columns": prefixed_feature_names(features),
        "eligible_full_period_feature_columns": eligible_columns,
        "minimum_full_period_coverage": float(args.minimum_full_period_coverage),
        "selection_contract": "eligibility is outcome-free; final selection occurs inside each temporal training fit",
    }
    _write_json(feature_manifest, feature_payload)
    manifest = {
        "schema": SCHEMA,
        "status": "completed_outcome_free_feature_universe",
        "rows": int(len(result)),
        "raw_feature_count": int(len(features)),
        "eligible_full_period_feature_count": int(len(eligible_columns)),
        "coverage": {
            "minimum": float(coverage["finite_fraction"].min()),
            "median": float(coverage["finite_fraction"].median()),
            "maximum": float(coverage["finite_fraction"].max()),
        },
        "contract": {
            "point_in_time": "immutable causal feature-store value at candidate signal __ts__",
            "outcome_used": False,
            "selection_used": False,
            "missing_values": "preserved for train-only coverage screening and CatBoost native handling",
            "side_scope": "coverage reported per side; feature selection deferred to side-local temporal fits",
        },
        "feature_store_report": store_report,
        "inputs": {
            "canonical": {"path": str(args.input), "sha256": _sha256(args.input)},
            "frozen_feature_contract": {
                "path": str(args.frozen_feature_contract),
                "sha256": _sha256(args.frozen_feature_contract),
            },
            "feature_store": str(args.feature_store),
        },
        "outputs": {
            "universe": {"path": str(output), "sha256": _sha256(output)},
            "coverage": {
                "path": str(coverage_output),
                "sha256": _sha256(coverage_output),
            },
            "feature_manifest": {
                "path": str(feature_manifest),
                "sha256": _sha256(feature_manifest),
            },
        },
    }
    _write_json(manifest_output, manifest)
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "data_perp/artifacts/execution_ev_canonical_exact_policy_regime_input_20260727_v3/joined.parquet"
        ),
    )
    parser.add_argument(
        "--frozen-feature-contract",
        type=Path,
        default=Path(
            "data_perp/artifacts/packb_side_local_ae_20260724_v1/long/loader_evidence/frozen_feature_contract.json"
        ),
    )
    parser.add_argument(
        "--feature-store",
        type=Path,
        default=Path("data_perp/features/20260711_070000"),
    )
    parser.add_argument("--minimum-full-period-coverage", type=float, default=0.99)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


if __name__ == "__main__":
    print(json.dumps(run(_parser().parse_args()), indent=2))
