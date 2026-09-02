#!/usr/bin/env python3
"""Union independently trained long/short CatBoost OOF streams for execution EV."""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

CLASS_ORDER = (
    "immediate_adverse_path",
    "early_mfe_full_reversal",
    "fast_realization_winner",
    "late_breakout",
    "slow_grinder",
    "noisy_timeout_usable_mfe",
    "dead_timeout",
)
JOIN_KEYS = ("__ts__", "__symbol__", "side_name", "candidate_id")
PREDICTION_ROLE = "path_archetype_oof"
ROLE_MANIFEST = "oof_probabilities.role_manifest.json"
OOF_PARQUET = "oof_probabilities.parquet"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_hash(
    payload: Mapping[str, Any], *, excluded: Sequence[str] = ()
) -> str:
    canonical = {key: value for key, value in payload.items() if key not in excluded}
    encoded = json.dumps(
        canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_json(path: Path, *, source: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{source}: cannot read valid JSON from {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{source}: expected a JSON object")
    return payload


def _load_side(
    directory: Path, *, expected_side: str
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    parquet = directory / OOF_PARQUET
    manifest_path = directory / ROLE_MANIFEST
    if not parquet.is_file() or not manifest_path.is_file():
        raise ValueError(f"{expected_side}: missing CatBoost OOF parquet/role manifest")
    manifest = _load_json(manifest_path, source=f"{expected_side} role manifest")
    signed = manifest.get("prediction_role_manifest_sha256")
    expected_signed = _canonical_json_hash(
        manifest, excluded=("prediction_role_manifest_sha256",)
    )
    if not isinstance(signed, str) or not hmac.compare_digest(signed, expected_signed):
        raise ValueError(f"{expected_side}: role manifest signature does not verify")
    if manifest.get("prediction_role") != PREDICTION_ROLE:
        raise ValueError(f"{expected_side}: wrong prediction role")
    if manifest.get("source_artifact_sha256") != _sha256(parquet):
        raise ValueError(f"{expected_side}: role manifest does not bind OOF parquet")
    declarations = manifest.get("prediction_columns")
    if not isinstance(declarations, Mapping):
        raise ValueError(f"{expected_side}: prediction columns are not declared")
    probability_columns = tuple(
        f"probability__{class_name}" for class_name in CLASS_ORDER
    )
    missing = sorted(set(probability_columns).difference(declarations))
    if missing:
        raise ValueError(
            f"{expected_side}: incomplete CatBoost probability vector {missing!r}"
        )
    frame = pd.read_parquet(parquet)
    missing_frame = sorted(set((*JOIN_KEYS, *probability_columns)).difference(frame))
    if missing_frame:
        raise ValueError(f"{expected_side}: OOF frame missing {missing_frame!r}")
    side = frame["side_name"].astype("string").str.strip().str.lower()
    if set(side) != {expected_side}:
        raise ValueError(f"{expected_side}: OOF frame contains the wrong side")
    frame = frame.copy()
    frame["side_name"] = side.astype(str)
    frame["__ts__"] = pd.to_datetime(frame["__ts__"], utc=True, errors="coerce")
    if frame[list(JOIN_KEYS)].isna().any().any():
        raise ValueError(f"{expected_side}: invalid OOF identity")
    if frame.duplicated(list(JOIN_KEYS)).any():
        raise ValueError(f"{expected_side}: duplicate OOF identity")
    evidence = {
        "side": expected_side,
        "directory": str(directory.resolve()),
        "oof_parquet": {
            "path": str(parquet.resolve()),
            "sha256": _sha256(parquet),
            "rows": int(len(frame)),
        },
        "role_manifest": {
            "path": str(manifest_path.resolve()),
            "sha256": _sha256(manifest_path),
            "signed_sha256": signed,
        },
    }
    return frame, manifest, evidence


def run(args: argparse.Namespace) -> dict[str, Path]:
    manifest_output = args.manifest or args.output.with_suffix(".manifest.json")
    if args.output.exists() or manifest_output.exists():
        raise ValueError("refusing to overwrite existing CatBoost union output")
    long, long_manifest, long_evidence = _load_side(args.long_dir, expected_side="long")
    short, short_manifest, short_evidence = _load_side(
        args.short_dir, expected_side="short"
    )
    if set(long.columns) != set(short.columns):
        raise ValueError("long and short CatBoost OOF schemas disagree")
    union = pd.concat([long, short], ignore_index=True)
    if union.duplicated(list(JOIN_KEYS)).any():
        raise ValueError("CatBoost side union contains duplicate identities")
    union = union.sort_values(list(JOIN_KEYS), kind="stable").reset_index(drop=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    union.to_parquet(args.output, index=False)

    prediction_columns = long_manifest.get("prediction_columns")
    if prediction_columns != short_manifest.get("prediction_columns"):
        raise ValueError("long and short prediction declarations disagree")
    manifest: dict[str, Any] = {
        "schema": "catboost_path_archetype_side_union_oof_v1",
        "prediction_role": PREDICTION_ROLE,
        "class_names": list(CLASS_ORDER),
        "prediction_columns": prediction_columns,
        "model_side_scope": "per_side",
        "shared_fitted_state": False,
        "final_refit_predictions_used_in_oof": False,
        "identity_columns": list(JOIN_KEYS),
        "rows": int(len(union)),
        "rows_by_side": {
            "long": int(len(long)),
            "short": int(len(short)),
        },
        "per_side_sources": {
            "long": long_evidence,
            "short": short_evidence,
        },
        "source_artifact": str(args.output.resolve()),
        "source_artifact_sha256": _sha256(args.output),
    }
    manifest["prediction_role_manifest_sha256"] = _canonical_json_hash(manifest)
    manifest_output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {"output": args.output, "manifest": manifest_output}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--long-dir", required=True, type=Path)
    parser.add_argument("--short-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", type=Path)
    return parser


def main() -> None:
    outputs = run(_parser().parse_args())
    print(json.dumps({key: str(value) for key, value in outputs.items()}))


if __name__ == "__main__":
    main()
