#!/usr/bin/env python3
"""Attach the frozen 31/8 alpha feature lists to exact-policy candidates."""

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
SIDES = ("long", "short")
GENERATED_PREFIXES = ("dae_", "gmm_")
SCHEMA = "exact_policy_capture_side_feature_expansion_v1"


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


def build_side_feature_contract(
    manifests: Mapping[str, Mapping[str, Any]],
    core_features: list[str],
    *,
    include_generated: bool = True,
) -> tuple[dict[str, list[str]], list[str], list[str]]:
    side_features: dict[str, list[str]] = {}
    raw_features: list[str] = []
    generated_features: list[str] = []
    for side in SIDES:
        selected = [str(value) for value in manifests[side]["features"]]
        raw = [
            feature
            for feature in selected
            if not feature.startswith(GENERATED_PREFIXES)
        ]
        generated = [
            feature
            for feature in selected
            if feature.startswith(GENERATED_PREFIXES) and include_generated
        ]
        names = [
            *core_features,
            *(f"capture_raw__{feature}" for feature in raw),
            *(f"capture_repr__{feature}" for feature in generated),
        ]
        side_features[side] = list(dict.fromkeys(names))
        raw_features.extend(raw)
        generated_features.extend(generated)
    return (
        side_features,
        list(dict.fromkeys(raw_features)),
        list(dict.fromkeys(generated_features)),
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    frame = pd.read_parquet(args.input)
    if frame.duplicated(list(IDENTITY)).any():
        raise ValueError("canonical capture input contains duplicate identities")
    core_manifest = json.loads(args.core_feature_manifest.read_text())
    core_features = [str(value) for value in core_manifest["feature_columns"]]
    alpha_manifests = {
        "long": json.loads(args.long_alpha_manifest.read_text()),
        "short": json.loads(args.short_alpha_manifest.read_text()),
    }
    side_features, raw_features, generated_features = build_side_feature_contract(
        alpha_manifests,
        core_features,
        include_generated=not bool(args.exclude_generated),
    )
    raw_matrix, raw_report = _load_feature_store_columns(
        frame,
        feature_dir=args.feature_store,
        selected_features=raw_features,
    )
    raw_matrix = raw_matrix.rename(
        columns={name: f"capture_raw__{name}" for name in raw_features}
    )
    frame = pd.concat(
        [frame.reset_index(drop=True), raw_matrix.reset_index(drop=True)], axis=1
    )
    if generated_features:
        context = pd.read_parquet(
            args.representation_context,
            columns=[*IDENTITY, *generated_features],
        )
        if context.duplicated(list(IDENTITY)).any():
            raise ValueError("representation context contains duplicate identities")
        context = context.rename(
            columns={
                name: f"capture_repr__{name}" for name in generated_features
            }
        )
        frame = frame.merge(
            context,
            on=list(IDENTITY),
            how="left",
            validate="one_to_one",
        )
    coverage_rows = []
    for side in SIDES:
        mask = frame["side_name"].astype(str).eq(side)
        additions = [
            name
            for name in side_features[side]
            if name.startswith(("capture_raw__", "capture_repr__"))
        ]
        numeric = frame.loc[mask, additions].apply(pd.to_numeric, errors="coerce")
        for name in additions:
            values = numeric[name].to_numpy(dtype=float)
            coverage_rows.append(
                {
                    "side_name": side,
                    "feature": name,
                    "rows": int(mask.sum()),
                    "finite_rows": int(np.isfinite(values).sum()),
                    "finite_fraction": float(np.isfinite(values).mean()),
                }
            )
        if not np.isfinite(numeric.to_numpy(dtype=float)).all():
            bad = [
                name
                for name in additions
                if not np.isfinite(numeric[name].to_numpy(dtype=float)).all()
            ]
            raise ValueError(
                f"{side} frozen selected capture features are incomplete: {bad}"
            )
    args.output_dir.mkdir(parents=True)
    output = args.output_dir / "expanded_capture_input.parquet"
    feature_manifest = args.output_dir / "capture_feature_manifest.json"
    coverage_output = args.output_dir / "feature_coverage.csv"
    manifest_output = args.output_dir / "manifest.json"
    frame.to_parquet(output, index=False, compression="zstd")
    pd.DataFrame(coverage_rows).to_csv(coverage_output, index=False)
    feature_payload = {
        "schema": "capture_feature_columns_by_side_v1",
        "feature_columns_by_side": side_features,
        "feature_columns": list(
            dict.fromkeys(side_features["long"] + side_features["short"])
        ),
        "source_alpha_feature_counts": {
            side: int(alpha_manifests[side]["feature_count"]) for side in SIDES
        },
        "contract": "core execution context plus exact frozen alpha winner list per side",
    }
    _write_json(feature_manifest, feature_payload)
    manifest = {
        "schema": SCHEMA,
        "status": "completed_frozen_side_feature_add_one_screen_input",
        "rows": int(len(frame)),
        "identity_unique": bool(not frame.duplicated(list(IDENTITY)).any()),
        "side_feature_counts": {
            side: len(side_features[side]) for side in SIDES
        },
        "added_raw_features": raw_features,
        "added_generated_features": generated_features,
        "feature_store_report": raw_report,
        "contract": {
            "side_local": True,
            "feature_selection": "reuse frozen 31/8 alpha winners; no capture-outcome selection",
            "feature_availability": "immutable causal feature-store point lookup at signal __ts__",
            "generated_representation": "frozen DAE/GMM outputs selected by the long alpha winner only",
            "generated_representation_included": bool(generated_features),
            "missing_value_policy": "fail closed for every side-local selected feature",
            "outcome_columns_added": [],
        },
        "inputs": {
            "canonical": {"path": str(args.input), "sha256": _sha256(args.input)},
            "core_feature_manifest": {
                "path": str(args.core_feature_manifest),
                "sha256": _sha256(args.core_feature_manifest),
            },
            "long_alpha_manifest": {
                "path": str(args.long_alpha_manifest),
                "sha256": _sha256(args.long_alpha_manifest),
            },
            "short_alpha_manifest": {
                "path": str(args.short_alpha_manifest),
                "sha256": _sha256(args.short_alpha_manifest),
            },
            "representation_context": {
                "path": str(args.representation_context),
                "sha256": _sha256(args.representation_context),
            },
            "feature_store": str(args.feature_store),
        },
        "outputs": {
            "expanded_input": {"path": str(output), "sha256": _sha256(output)},
            "feature_manifest": {
                "path": str(feature_manifest),
                "sha256": _sha256(feature_manifest),
            },
            "coverage": {
                "path": str(coverage_output),
                "sha256": _sha256(coverage_output),
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
        "--core-feature-manifest",
        type=Path,
        default=Path(
            "data_perp/artifacts/execution_ev_context_clean_regime_diagnosis_forward_july19_20260726_v1/regime_diagnosis_manifest.json"
        ),
    )
    parser.add_argument(
        "--long-alpha-manifest",
        type=Path,
        default=Path(
            "data_perp/artifacts/packb_side_local_outer_oof_july20_20260726_v1_31_8/long/manifest.json"
        ),
    )
    parser.add_argument(
        "--short-alpha-manifest",
        type=Path,
        default=Path(
            "data_perp/artifacts/packb_side_local_outer_oof_july20_20260726_v1_31_8/short/manifest.json"
        ),
    )
    parser.add_argument(
        "--representation-context",
        type=Path,
        default=Path(
            "data_perp/artifacts/packb_downstream_representation_july20_20260726_v1_31_8/context.parquet"
        ),
    )
    parser.add_argument(
        "--feature-store",
        type=Path,
        default=Path("data_perp/features/20260711_070000"),
    )
    parser.add_argument("--exclude-generated", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


if __name__ == "__main__":
    print(json.dumps(run(_parser().parse_args()), indent=2))
