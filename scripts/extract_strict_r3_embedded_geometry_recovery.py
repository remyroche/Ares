#!/usr/bin/env python3
"""Extract and certify the frozen Geometry/K9 parent embedded in a sealed bundle.

This is deliberately not a geometry refit.  It copies the already-fitted
October--December 2024 parent object from a sealed conversion bundle, retains
its historical semantic identity, and writes a new content-addressed payload
plus an output-parity receipt.  A consumer must bind both the new payload hash
and the preserved semantic parent identity; the latter is what the conversion
bundle's frozen K9 view uses.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
EXPECTED_PARENT_SHA256 = (
    "ad7eae631da909feddee7349d07fd8ef377db173067d971bee33d24d82f20eb4"
)
DEFAULT_SOURCE = (
    ROOT
    / "data_perp/artifacts/strict_r3_lockstep_successor28_homogeneous28_long_aug1_7_"
    "20260813_v1/bundles/cutoff=20260801/conversion/four_week_conversion_bundle.joblib"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode()).hexdigest()


def _write_new(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"immutable output exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


def _output_parity(
    *, parent: Any, extracted: Any, features_path: Path, temperature_scale: float,
) -> dict[str, Any]:
    features = pd.read_parquet(features_path)
    required = ["candidate_id", "__decision_ts__", *parent.encoder_fields]
    missing = [field for field in required if field not in features]
    if missing:
        raise ValueError(f"verification features miss required fields: {missing[:5]}")
    left = parent.transform(features, temperature_scale=temperature_scale)
    right = extracted.transform(features, temperature_scale=temperature_scale)
    if list(left.columns) != list(right.columns):
        raise AssertionError("extracted geometry output columns differ")
    left_values = left.to_numpy(dtype=np.float64)
    right_values = right.to_numpy(dtype=np.float64)
    if not np.array_equal(np.isnan(left_values), np.isnan(right_values)):
        raise AssertionError("extracted geometry NaN mask differs")
    finite = np.isfinite(left_values) & np.isfinite(right_values)
    maximum = float(np.max(np.abs(left_values[finite] - right_values[finite]))) if finite.any() else 0.0
    if maximum != 0.0:
        raise AssertionError(f"extracted geometry output differs: max delta={maximum}")
    return {
        "rows": int(len(features)),
        "temperature_scale": float(temperature_scale),
        "output_columns": int(len(left.columns)),
        "output_columns_sha256": _json_sha256(list(left.columns)),
        "source_output_sha256_float32": hashlib.sha256(
            left.to_numpy(dtype=np.float32).tobytes()
        ).hexdigest(),
        "extracted_output_sha256_float32": hashlib.sha256(
            right.to_numpy(dtype=np.float32).tobytes()
        ).hexdigest(),
        "maximum_absolute_delta": maximum,
        "nan_mask_exact": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--source-conversion-bundle", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--verification-features", type=Path, required=True)
    parser.add_argument("--temperature-scale", type=float, default=0.25)
    args = parser.parse_args()

    output = args.out_dir.resolve()
    source = args.source_conversion_bundle.resolve()
    verification_features = args.verification_features.resolve()
    if output.exists():
        raise FileExistsError(f"immutable output directory already exists: {output}")
    if not source.is_file() or not verification_features.is_file():
        raise FileNotFoundError("source conversion bundle or verification features are missing")

    source_bundle = joblib.load(source)
    parent = source_bundle.geometry.parent
    if str(parent.bundle_sha256) != EXPECTED_PARENT_SHA256:
        raise ValueError("source conversion bundle does not embed the required geometry parent")
    if str(source_bundle.geometry.bundle_sha256) != (
        "dbf7de6da6bad6927bcbe577d7ad2d2118ecc24a6bdfd35fb7fa190be13d7638"
    ):
        raise ValueError("source conversion bundle does not embed the required frozen K9 view")

    # Deep-copy so the preserved sealed source never changes in memory.
    extracted = copy.deepcopy(parent)
    payload = output / "frozen_geometry_k9.joblib"
    output.mkdir(parents=True)
    joblib.dump(extracted, payload, compress=3)
    content_sha = _sha256(payload)
    restored = joblib.load(payload)
    if str(restored.bundle_sha256) != EXPECTED_PARENT_SHA256:
        raise AssertionError("extracted payload lost its original semantic identity")
    parity = _output_parity(
        parent=parent,
        extracted=restored,
        features_path=verification_features,
        temperature_scale=args.temperature_scale,
    )

    manifest = {
        **restored.fit_audit,
        "schema": "strict_r3_geometry_k9_oct_dec_2024_v2_embedded_recovery",
        "side": "long",
        "bundle_file": payload.name,
        "bundle_sha256": content_sha,
        "semantic_parent_bundle_sha256": EXPECTED_PARENT_SHA256,
        "semantic_k9_view_sha256": str(source_bundle.geometry.bundle_sha256),
        "recovery_mode": "sealed_conversion_bundle_embedded_parent_extraction_no_refit",
        "source_conversion_bundle": str(source.relative_to(ROOT)),
        "source_conversion_bundle_sha256": _sha256(source),
        "leaf_categories_sha256": _json_sha256(restored.leaf_categories),
        "leaf_support_sha256": _json_sha256([
            hashlib.sha256(np.asarray(values).tobytes()).hexdigest()
            for values in restored.leaf_support_counts
        ]),
        "cluster_centres_sha256": hashlib.sha256(
            np.asarray(restored.kmeans.cluster_centers_, dtype=np.float32).tobytes()
        ).hexdigest(),
        "cluster_order": np.asarray(restored.cluster_order).tolist(),
        "input_order": list(restored.encoder_fields),
        "imputation_medians": np.asarray(restored.medians, dtype=float).tolist(),
        "verification": parity,
    }
    _write_new(output / "run_manifest.json", manifest)
    print(json.dumps({
        "status": "pass",
        "out_dir": str(output),
        "content_bundle_sha256": content_sha,
        "semantic_parent_bundle_sha256": EXPECTED_PARENT_SHA256,
        "semantic_k9_view_sha256": str(source_bundle.geometry.bundle_sha256),
        "verification": parity,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
