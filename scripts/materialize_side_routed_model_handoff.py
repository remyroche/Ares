#!/usr/bin/env python3
"""Materialize an explicit side-routed base/meta/residual model handoff.

This adapter is intentionally provenance-only: it does not rewrite or merge
models whose frozen AE/GMM representations differ.  Consumers must select the
complete route for the requested side, including that route's AE/GMM state.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import joblib

from extreme_price_movements.inference.side_routed_model import (
    BaseScorePassthroughRegressor,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return payload


def _require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _artifact(path: Path) -> dict[str, Any]:
    path = _require(path)
    return {"path": str(path), "sha256": _sha256(path), "bytes": path.stat().st_size}


def _positional_feature_mapping(features: list[str]) -> dict[str, str]:
    """Persist the exact positional-to-raw feature order expected by inference."""

    return {f"f{index}": str(feature) for index, feature in enumerate(features)}


def _rewrite_policy_paths(value: Any, source_root: Path, output_root: Path) -> Any:
    """Relocate policy-local paths while preserving unrelated provenance paths."""

    if isinstance(value, dict):
        return {
            str(key): _rewrite_policy_paths(item, source_root, output_root)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_rewrite_policy_paths(item, source_root, output_root) for item in value]
    if not isinstance(value, str):
        return value

    candidates = (str(source_root), str(source_root.resolve()))
    for prefix in candidates:
        if value == prefix:
            return str(output_root)
        if value.startswith(prefix + "/"):
            return str(output_root / value[len(prefix) + 1 :])
    return value


def _materialize_side_routed_policy(
    *,
    source_root: Path,
    output_root: Path,
    routed_residual_path: Path,
) -> dict[str, Any]:
    """Copy the shared policy and restrict the V9 route to shorts.

    A source training/live parity contract is intentionally not copied: its
    model hashes describe the source bundle, not this side-routed bundle. Live
    inference therefore remains fail-closed until combined parity evidence is
    generated.
    """

    source_root = _require(source_root)
    output_root.mkdir(parents=True, exist_ok=True)
    skipped = {"training_live_parity_contract.json", "side_residual_expert.joblib"}
    stale_parity = output_root / "training_live_parity_contract.json"
    if stale_parity.exists():
        stale_parity.unlink()
    copied: list[str] = []
    for source in sorted(source_root.iterdir()):
        if not source.is_file() or source.name in skipped:
            continue
        destination = output_root / source.name
        if source.suffix.lower() == ".json":
            payload = _read_json(source)
            payload = _rewrite_policy_paths(payload, source_root, output_root)
            if source.name == "optimized_portfolio_policy_config.json":
                payload["canonical_meta_postprocessor_sides"] = ["short"]
                payload["side_residual_expert_enabled"] = True
                payload["side_residual_expert_artifact_path"] = str(
                    routed_residual_path.resolve()
                )
            destination.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        else:
            shutil.copy2(source, destination)
        copied.append(source.name)

    return {
        "source_root": str(source_root),
        "output_root": str(output_root),
        "copied_files": copied,
        "copied_file_count": len(copied),
        "skipped_files": sorted(skipped),
        "canonical_meta_postprocessor_sides": ["short"],
        "side_residual_expert_artifact_path": str(routed_residual_path.resolve()),
        "training_live_parity_contract_materialized": False,
    }


def _state_contract(path: Path) -> dict[str, Any]:
    state = joblib.load(_require(path))
    if not isinstance(state, dict) or not bool(state.get("enabled", False)):
        raise ValueError(f"AE/GMM state is not an enabled mapping: {path}")
    return {
        **_artifact(path),
        "input_feature_order_hash": str(state.get("input_feature_order_hash") or ""),
        "input_feature_count": len(state.get("feature_columns") or []),
        "latent_feature_count": len(state.get("latent_columns") or []),
        "gmm_n_components": int(state.get("gmm_n_components") or 0),
        "gmm_reg_covar": float(state.get("gmm_reg_covar") or 0.0),
        "ae_fit_rows": int(state.get("ae_fit_rows") or 0),
        "gmm_fit_rows": int(state.get("gmm_fit_rows") or 0),
    }


def _base_contract(model_path: Path, columns_path: Path, side: str) -> dict[str, Any]:
    model_payload = joblib.load(_require(model_path))
    columns = _read_json(_require(columns_path))
    if isinstance(model_payload, dict):
        if side not in model_payload:
            raise ValueError(f"Base model dictionary has no {side!r} head: {model_path}")
        model_kind = type(model_payload[side]).__name__
        model_key: str | None = side
    else:
        model_kind = type(model_payload).__name__
        model_key = None
    by_side = columns.get("feature_names_by_side") or {}
    features = by_side.get(side) or by_side.get("shared") or columns.get("feature_names")
    if not isinstance(features, list) or not features:
        raise ValueError(f"No base feature contract for {side!r}: {columns_path}")
    return {
        "model": _artifact(model_path),
        "model_key": model_key,
        "model_kind": model_kind,
        "columns": _artifact(columns_path),
        "feature_count": len(features),
        "feature_names": [str(value) for value in features],
        "feature_contract_hash": str(columns.get("feature_contract_hash") or ""),
    }


def _meta_contract(model_path: Path | None, columns_path: Path | None, side: str) -> dict[str, Any]:
    if model_path is None:
        return {"mode": "direct_base_residual", "model": None}
    model = joblib.load(_require(model_path))
    if isinstance(model, dict):
        raise TypeError(f"Expected one {side} meta model, got a mapping: {model_path}")
    result: dict[str, Any] = {
        "mode": "frozen_meta_head",
        "model": _artifact(model_path),
        "model_kind": type(model).__name__,
    }
    if columns_path is not None:
        columns = _read_json(_require(columns_path))
        by_model = columns.get("feature_names_by_model") or {}
        features = by_model.get(f"base_soft_label_{side}") or columns.get("feature_names")
        if not isinstance(features, list) or not features:
            raise ValueError(f"No meta feature contract for {side!r}: {columns_path}")
        result.update(
            {
                "columns": _artifact(columns_path),
                "feature_count": len(features),
                "feature_names": [str(value) for value in features],
                "feature_contract_hash": str(
                    (columns.get("feature_contract_hash_by_model") or {}).get(
                        f"base_soft_label_{side}", columns.get("feature_contract_hash") or ""
                    )
                ),
            }
        )
    return result


def _residual_contract(path: Path, side: str) -> dict[str, Any]:
    payload = joblib.load(_require(path))
    if not isinstance(payload, dict):
        raise TypeError(f"Residual expert is not a mapping: {path}")
    for key in ("feature_contract", "residual_models", "alpha_by_side"):
        value = payload.get(key)
        if not isinstance(value, dict) or side not in value:
            raise ValueError(f"Residual expert has no {side!r} entry in {key}: {path}")
    return {
        **_artifact(path),
        "schema": str(payload.get("schema") or ""),
        "backbone_score": str(payload.get("backbone_score") or ""),
        "backbone_score_col": str(payload.get("backbone_score_col") or ""),
        "feature_count": len(payload["feature_contract"][side]),
        "feature_names": [str(value) for value in payload["feature_contract"][side]],
        "alpha": float(payload["alpha_by_side"][side]),
        "round_trip_cost": float(payload.get("round_trip_cost")),
    }


def materialize(
    *,
    output_root: Path,
    long_base_model: Path,
    long_base_columns: Path,
    long_residual_expert: Path,
    long_ae_gmm_state: Path,
    short_bundle_manifest: Path,
) -> dict[str, Any]:
    short_handoff = _read_json(_require(short_bundle_manifest))
    short_base_dir = Path(short_handoff["base_model_dir"])
    short_meta_dir = Path(short_handoff["meta_model_dir"])
    short_residual = Path(short_handoff["side_residual_expert"])
    short_state = Path(short_handoff["ae_gmm_state"])
    short_policy = Path(short_handoff["policy_root"])
    routes = {
        "long": {
            "architecture": "current_base_plus_direct_residual_expert",
            "base": _base_contract(long_base_model, long_base_columns, "long"),
            "meta": _meta_contract(None, None, "long"),
            "residual_expert": _residual_contract(long_residual_expert, "long"),
            "ae_gmm": _state_contract(long_ae_gmm_state),
            "policy": {"status": "retain_current_long_policy_contract"},
        },
        "short": {
            "architecture": "packb_short_meta_plus_direct_residual_expert_v9only",
            "source_handoff": _artifact(short_bundle_manifest),
            "base": _base_contract(
                short_base_dir / "base_model.joblib",
                short_base_dir / "columns.json",
                "short",
            ),
            "meta": _meta_contract(
                short_meta_dir / "base_soft_label_short.joblib",
                short_meta_dir / "columns.json",
                "short",
            ),
            "residual_expert": _residual_contract(short_residual, "short"),
            "ae_gmm": _state_contract(short_state),
            "policy": {
                "mode": "v9_only",
                "root": str(_require(short_policy)),
            },
        },
    }
    long_base_payload = joblib.load(long_base_model)
    if isinstance(long_base_payload, dict):
        long_base_payload = long_base_payload.get("long")
    if long_base_payload is None:
        raise ValueError("Current long base model could not be resolved")
    short_base_payload = joblib.load(short_base_dir / "base_model.joblib")
    if not isinstance(short_base_payload, dict) or "short" not in short_base_payload:
        raise ValueError("Requested short base model dictionary has no short head")
    short_meta_payload = joblib.load(short_meta_dir / "base_soft_label_short.joblib")
    trained_state = {
        "schema": "side_routed_trained_state_v1",
        "bundle": {
            "alpha_models": {
                "long_s52_meta_threshold_handoff": {
                    "model": long_base_payload,
                    "feat_cols": routes["long"]["base"]["feature_names"],
                },
                "short_s52_meta_threshold_handoff": {
                    "model": short_base_payload["short"],
                    "feat_cols": routes["short"]["base"]["feature_names"],
                },
            },
            "meta_models": {
                "long_s52_meta_threshold_handoff": BaseScorePassthroughRegressor(),
                "short_s52_meta_threshold_handoff": short_meta_payload,
            },
        },
    }
    model_root = output_root / "models"
    model_root.mkdir(parents=True, exist_ok=True)
    trained_state_path = model_root / "trained_state.pkl"
    joblib.dump(trained_state, trained_state_path)

    long_residual_payload = joblib.load(long_residual_expert)
    short_residual_payload = joblib.load(short_residual)
    routed_residual_path = output_root / "policy_params" / "side_residual_expert.joblib"
    routed_residual_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "schema": "side_routed_side_residual_expert_v1",
            "routes": {
                "long": long_residual_payload,
                "short": short_residual_payload,
            },
        },
        routed_residual_path,
    )

    policy_contract = _materialize_side_routed_policy(
        source_root=short_policy,
        output_root=output_root / "policy_params",
        routed_residual_path=routed_residual_path,
    )

    meta_contract_root = output_root / "meta_oof"
    meta_contract_root.mkdir(parents=True, exist_ok=True)
    (meta_contract_root / "meta_feature_contract.json").write_text(
        json.dumps(
            {
                "schema": "side_routed_meta_feature_contract_v1",
                "meta_models": {
                    "long_s52_meta_threshold_handoff": {
                        "feature_columns": ["base_score_raw"],
                        "n_features": 1,
                        "positional_feature_mapping": {
                            "f0": "base_score_raw",
                        },
                        "mode": "base_score_passthrough_before_direct_residual",
                    },
                    "short_s52_meta_threshold_handoff": {
                        "feature_columns": routes["short"]["meta"]["feature_names"],
                        "n_features": len(routes["short"]["meta"]["feature_names"]),
                        "positional_feature_mapping": _positional_feature_mapping(
                            routes["short"]["meta"]["feature_names"]
                        ),
                        "feature_contract_hash": routes["short"]["meta"].get(
                            "feature_contract_hash", ""
                        ),
                    },
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    state_hashes = {side: route["ae_gmm"]["sha256"] for side, route in routes.items()}
    manifest = {
        "schema": "side_routed_model_handoff_v1",
        "routes": routes,
        "routing_contract": {
            "key": "side_name",
            "allowed": ["long", "short"],
            "select_entire_route_before_ae_gmm_transform": True,
            "shared_ae_gmm_state": len(set(state_hashes.values())) == 1,
            "ae_gmm_state_sha256_by_side": state_hashes,
        },
        "status": {
            "model_components_loadable": True,
            "side_routed_model_bundle_ready": True,
            "side_routed_residual_expert_ready": True,
            "model_replay_handoff_ready": True,
            "full_policy_replay_ready": False,
            "native_inference_ready": False,
            "native_inference_blockers": [
                "combined training/live parity contract is not materialized",
                (
                    "combined side-routed execution evidence is pending; labels already "
                    "use signal_ts + timeframe and delayed fills remain a downstream "
                    "policy/inference adjustment"
                ),
            ],
        },
        "packaged": {
            "trained_state": _artifact(trained_state_path),
            "side_residual_expert": _artifact(routed_residual_path),
            "side_routed_policy": policy_contract,
            "meta_feature_contract": _artifact(
                meta_contract_root / "meta_feature_contract.json"
            ),
        },
    }
    output_root.mkdir(parents=True, exist_ok=True)
    path = output_root / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--long-base-model", type=Path, required=True)
    parser.add_argument("--long-base-columns", type=Path, required=True)
    parser.add_argument("--long-residual-expert", type=Path, required=True)
    parser.add_argument("--long-ae-gmm-state", type=Path, required=True)
    parser.add_argument("--short-bundle-manifest", type=Path, required=True)
    args = parser.parse_args()
    manifest = materialize(
        output_root=args.output_root,
        long_base_model=args.long_base_model,
        long_base_columns=args.long_base_columns,
        long_residual_expert=args.long_residual_expert,
        long_ae_gmm_state=args.long_ae_gmm_state,
        short_bundle_manifest=args.short_bundle_manifest,
    )
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
