#!/usr/bin/env python3
"""Apply pre-2026 frozen contextual score arms to sealed 2026 inputs.

This command is deliberately separate from fitting.  It accepts only an exact
candidate stream and authoritative v2 current-context sidecars; it does not
perform selection, tuning, calibration or causal EV mapping.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd

from scripts.train_freeze_contextual_score_arms import (
    ARMS, BASE_FEATURES, FREEZE, IDENTITY, REGIME_FEATURES, SCHEMA as FROZEN_SCHEMA,
    TRANSITION_FEATURES, FrozenContextError, _canonical, _sha, feature_sets,
)

SCHEMA = "precomputed_post_freeze_contextual_arm_scores_v1"
AUTHORITATIVE = "SEALED_POST_FREEZE_2026_AUTHORITATIVE"


class ApplyFrozenContextError(RuntimeError):
    """Raised when post-freeze scoring provenance is incomplete."""


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ApplyFrozenContextError(f"JSON object required: {path}")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        temp = Path(handle.name)
        json.dump(value, handle, indent=2, sort_keys=True, default=str, allow_nan=False)
        handle.write("\n")
    os.replace(temp, path)


def _contains_hash(value: Any, digest: str) -> bool:
    if isinstance(value, Mapping):
        return any(_contains_hash(v, digest) for v in value.values())
    if isinstance(value, list):
        return any(_contains_hash(v, digest) for v in value)
    return isinstance(value, str) and value == digest


def _verify_binding(path: Path, manifest_path: Path, checksum_path: Path, *, role: str,
                    expected_status: str | None = None, expected_schema: str | None = None) -> dict[str, Any]:
    manifest = _json(manifest_path)
    if not checksum_path.is_file() or checksum_path.read_text(encoding="utf-8").split()[0:1] != [_sha(manifest_path)]:
        raise ApplyFrozenContextError(f"{role} manifest detached checksum fails")
    if not _contains_hash(manifest, _sha(path)):
        raise ApplyFrozenContextError(f"{role} manifest does not hash-bind {path.name}")
    if expected_status and manifest.get("status") != expected_status:
        raise ApplyFrozenContextError(f"{role} status must be {expected_status}")
    if expected_schema and manifest.get("schema") != expected_schema:
        raise ApplyFrozenContextError(f"{role} schema must be {expected_schema}")
    return manifest


def _exact_hourly_join(candidates: pd.DataFrame, sidecar: pd.DataFrame, *, role: str,
                       fields: tuple[str, ...], available: str) -> pd.DataFrame:
    required = {"__ts__", available, *fields}
    missing = sorted(required.difference(sidecar.columns))
    if missing:
        raise ApplyFrozenContextError(f"{role} sidecar lacks {missing}")
    payload = sidecar.loc[:, ["__ts__", available, *fields]].copy()
    payload["__ts__"] = pd.to_datetime(payload["__ts__"], utc=True, errors="raise")
    payload[available] = pd.to_datetime(payload[available], utc=True, errors="raise")
    if payload.duplicated("__ts__").any() or payload[available].gt(payload["__ts__"]).any():
        raise ApplyFrozenContextError(f"{role} sidecar has duplicate or unavailable state")
    payload.loc[:, list(fields)] = payload.loc[:, list(fields)].apply(pd.to_numeric, errors="coerce")
    marked = candidates.copy(); marked["__order__"] = np.arange(len(marked))
    joined = marked.merge(payload, how="left", on="__ts__", validate="many_to_one", sort=False).sort_values("__order__", kind="stable").drop(columns="__order__")
    if len(joined) != len(candidates) or joined[list(IDENTITY)].reset_index(drop=True).equals(candidates[list(IDENTITY)].reset_index(drop=True)) is False:
        raise ApplyFrozenContextError(f"{role} exact join altered candidate support")
    if joined[available].isna().any() or joined[available].gt(joined["__ts__"]).any():
        raise ApplyFrozenContextError(f"{role} lacks available exact timestamp coverage")
    return joined


def apply(*, frozen_dir: Path, candidates_path: Path, candidates_manifest_path: Path,
          candidates_checksum_path: Path, regime_path: Path, regime_manifest_path: Path,
          regime_checksum_path: Path, transition_path: Path, transition_manifest_path: Path,
          transition_checksum_path: Path, output_dir: Path) -> dict[str, Any]:
    """Score all four frozen arms, preserving exact candidate identity and labels."""
    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError(destination)
    frozen_dir = Path(frozen_dir)
    frozen_manifest = _verify_binding(frozen_dir / "blocked_oof_training_panel.parquet", frozen_dir / "manifest.json",
                                      frozen_dir / "manifest.sha256", role="frozen arm artifact", expected_schema=FROZEN_SCHEMA,
                                      expected_status="FROZEN_PRE_2026_CONTEXTUAL_SCORE_ARMS_READY")
    contract = frozen_manifest.get("frozen_contextual_coefficients", {})
    if contract.get("training_end_exclusive_utc") != FREEZE.isoformat() or contract.get("arms") != list(ARMS):
        raise ApplyFrozenContextError("frozen arm artifact does not prove the approved pre-2026 contract")
    # Candidate score input must itself be sealed.  Its own producer need not
    # share this script's schema, but it must be immutable before evaluation.
    candidate_manifest = _verify_binding(Path(candidates_path), Path(candidates_manifest_path), Path(candidates_checksum_path), role="2026 candidate score source")
    if not str(candidate_manifest.get("status", "")).startswith("SEALED_POST_FREEZE_2026"):
        raise ApplyFrozenContextError("2026 candidate score source is not sealed")
    for path, manifest, checksum, role in ((regime_path, regime_manifest_path, regime_checksum_path, "regime"),
                                            (transition_path, transition_manifest_path, transition_checksum_path, "transition")):
        sidecar_manifest = _verify_binding(Path(path), Path(manifest), Path(checksum), role=role, expected_status=AUTHORITATIVE)
        if not str(sidecar_manifest.get("schema", "")).endswith("_v2"):
            raise ApplyFrozenContextError(f"{role} must be an authoritative v2 sidecar")
    candidates = _canonical(pd.read_parquet(candidates_path), role="2026 candidate score source")
    if not candidates["__ts__"].dt.year.eq(2026).all():
        raise ApplyFrozenContextError("this post-freeze applicator accepts 2026 candidate timestamps only")
    if "baseline_context_free_raw_score" not in candidates.columns:
        raise ApplyFrozenContextError("2026 candidate source lacks baseline_context_free_raw_score")
    candidates["baseline_context_free_raw_score"] = pd.to_numeric(candidates["baseline_context_free_raw_score"], errors="coerce")
    if candidates["baseline_context_free_raw_score"].isna().any():
        raise ApplyFrozenContextError("baseline score has non-finite values")
    panel = _exact_hourly_join(candidates, pd.read_parquet(regime_path), role="regime", fields=REGIME_FEATURES, available="regime_available_utc")
    panel = _exact_hourly_join(panel, pd.read_parquet(transition_path), role="transition", fields=TRANSITION_FEATURES, available="transition_available_utc")
    panel["side_is_long"] = panel["side_name"].eq("long").astype(float)
    for field in (*REGIME_FEATURES, *TRANSITION_FEATURES):
        panel[field] = pd.to_numeric(panel[field], errors="coerce")
    for arm, fields in feature_sets().items():
        model_path = frozen_dir / f"{arm}.joblib"
        if not model_path.is_file():
            raise ApplyFrozenContextError(f"frozen arm model absent: {model_path.name}")
        panel[f"{arm}_raw_score"] = np.asarray(joblib.load(model_path).predict(panel.loc[:, fields]), dtype=float)
    stage = Path(tempfile.mkdtemp(dir=destination.parent, prefix=f".{destination.name}.staging-"))
    try:
        output = stage / "precomputed_2026_arm_scores.parquet"
        panel.to_parquet(output, index=False, compression="zstd")
        manifest = {"schema": SCHEMA, "status": "SEALED_POST_FREEZE_2026_SCORES",
                    "scope": "2026 raw score application only; causal EV mapping and global top10 selection are deferred to evaluator",
                    "frozen_contextual_coefficients": contract,
                    "inputs": {str(path): _sha(Path(path)) for path in (candidates_path, regime_path, transition_path,
                        frozen_dir / "manifest.json", *(frozen_dir / f"{arm}.joblib" for arm in ARMS))},
                    "outputs": {output.name: _sha(output)}, "rows": int(len(panel)), "arms": list(ARMS)}
        _write_json(stage / "manifest.json", manifest)
        (stage / "manifest.sha256").write_text(f"{_sha(stage / 'manifest.json')}  manifest.json\n", encoding="utf-8")
        os.replace(stage, destination)
        return manifest
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-dir", required=True, type=Path)
    parser.add_argument("--candidates", required=True, type=Path); parser.add_argument("--candidates-manifest", required=True, type=Path); parser.add_argument("--candidates-checksum", required=True, type=Path)
    parser.add_argument("--regime", required=True, type=Path); parser.add_argument("--regime-manifest", required=True, type=Path); parser.add_argument("--regime-checksum", required=True, type=Path)
    parser.add_argument("--transition", required=True, type=Path); parser.add_argument("--transition-manifest", required=True, type=Path); parser.add_argument("--transition-checksum", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(apply(frozen_dir=args.frozen_dir, candidates_path=args.candidates, candidates_manifest_path=args.candidates_manifest, candidates_checksum_path=args.candidates_checksum,
                           regime_path=args.regime, regime_manifest_path=args.regime_manifest, regime_checksum_path=args.regime_checksum,
                           transition_path=args.transition, transition_manifest_path=args.transition_manifest, transition_checksum_path=args.transition_checksum,
                           output_dir=args.output_dir), indent=2, default=str))


if __name__ == "__main__":
    main()
