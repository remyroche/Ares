#!/usr/bin/env python3
"""Materialize the frozen 39-field execution-EV pre-entry table.

Consumes the future Pack-B base/residual context, scores the frozen clean-event,
conditional Peak-MFE, and seven-class path CatBoost models per side, and emits
the exact raw feature contract required by the final direct/capture heads.
Missing or non-finite inputs fail the whole cohort; no outcome columns are read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.path_auxiliary_timing_training import (  # noqa: E402
    predict_side_local_timing_cdf_family,
)
from scripts.run_catboost_path_archetype_classifier import _entropy  # noqa: E402


SCHEMA = "execution_ev_forward_preentry_v1"
SIDES = ("long", "short")
IDENTITY = ("candidate_id", "__ts__", "__symbol__", "side_name")
PATH_CLASSES = (
    "immediate_adverse_path",
    "early_mfe_full_reversal",
    "fast_realization_winner",
    "late_breakout",
    "slow_grinder",
    "noisy_timeout_usable_mfe",
    "dead_timeout",
)
DEFAULT_SPEC = Path(
    "configs/execution_ev_forward_confirmation_candidate_20260728_v1.json"
)
DEFAULT_ROLE_ROOT = Path(
    "data_perp/artifacts/packb_path_auxiliary_role_bundles_20260725_v1_31_8"
)
DEFAULT_CATBOOST_ROOT = Path(
    "data_perp/reports/catboost_path_archetype_packb31_8_structural_balance_"
    "20260725_v1"
)
DEFAULT_HEAD_CONTRACT = Path(
    "data_perp/artifacts/execution_ev_forward_final_heads_20260728_v1/"
    "feature_contract.json"
)
DEFAULT_OUTPUT = Path(
    "data_perp/artifacts/execution_ev_forward_preentry_20260728_v1"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(_safe(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _role_record(spec: Mapping[str, Any], role: str) -> Mapping[str, Any]:
    records = [record for record in spec["models"] if record["role"] == role]
    if len(records) != 1:
        raise ValueError(f"expected exactly one {role} model record")
    record = records[0]
    path = Path(record["path"])
    if not path.is_absolute():
        path = ROOT / path
    if _sha256(path) != record["sha256"]:
        raise ValueError(f"{role} model hash mismatch")
    return record


def _finite_matrix(frame: pd.DataFrame, features: Sequence[str], *, name: str) -> pd.DataFrame:
    missing = sorted(set(features).difference(frame.columns))
    if missing:
        raise ValueError(f"{name} features missing: {missing}")
    matrix = frame.loc[:, list(features)].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(matrix.to_numpy(dtype=np.float32)).all():
        raise ValueError(f"{name} features are not finite")
    return matrix.astype(np.float32)


def _timing_features(state: Mapping[str, Any]) -> list[str]:
    by_horizon = state.get("selected_features_by_horizon")
    if by_horizon is None:
        return list(state["selected_features"])
    return list(
        dict.fromkeys(
            feature
            for hour in sorted(by_horizon, key=lambda value: int(value))
            for feature in by_horizon[hour]
        )
    )


def score_supporting_heads(
    frame: pd.DataFrame,
    *,
    spec: Mapping[str, Any],
    role_root: Path,
    catboost_root: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    parts: list[pd.DataFrame] = []
    reports: dict[str, Any] = {}
    for side in SIDES:
        local = frame.loc[frame["side_name"].astype(str).eq(side)].copy()
        if local.empty:
            raise ValueError(f"future pre-entry context has no {side} rows")
        _role_record(spec, f"clean_favorable_event_{side}")
        _role_record(spec, f"peak_mfe_{side}")
        _role_record(spec, f"path_catboost_{side}")
        timing = joblib.load(
            role_root
            / "shared/meaningful_mfe_event"
            / side
            / "timing_cdf_family.joblib"
        )
        mean = joblib.load(
            role_root
            / "roles/peak_mfe_12h_atr__conditional_mean"
            / side
            / "role_bundle.joblib"
        )
        timing_state = timing["side_models"][side]
        peak_features = list(
            dict.fromkeys([*_timing_features(timing_state), *mean["selected_features"]])
        )
        peak_x = _finite_matrix(local, peak_features, name=f"peak_clean_{side}")
        timing_scores = predict_side_local_timing_cdf_family(
            timing,
            peak_x,
            sides=[side] * len(peak_x),
        )
        # The frozen timing-family scorer returns NumPy arrays.  Accept a
        # Series-like result as well, but do not require pandas semantics at
        # this model boundary.
        p_hit = np.clip(
            np.asarray(timing_scores["p_hit_12h"], dtype=float), 0.0, 1.0
        )
        conditional_mean = np.maximum(
            np.asarray(
                mean["final_inference_model"].predict(
                    peak_x.loc[:, mean["selected_features"]]
                ),
                dtype=float,
            ),
            0.0,
        )
        local["oof_clean_favorable_probability"] = p_hit
        local["pred_peak_MFE_12h_ATR"] = p_hit * conditional_mean

        classifier = joblib.load(
            catboost_root / f"side={side}" / "path_archetype_classifier.joblib"
        )
        classes = tuple(map(str, classifier.class_names))
        if classes != PATH_CLASSES:
            raise ValueError(f"{side} path CatBoost class order changed")
        path_x = _finite_matrix(
            local,
            list(classifier.feature_columns),
            name=f"path_catboost_{side}",
        )
        probability_frame = classifier.predict_proba(path_x)
        probabilities = probability_frame.loc[:, list(classes)].to_numpy(dtype=float)
        if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6, rtol=0.0):
            raise ValueError(f"{side} path probabilities do not sum to one")
        for index, name in enumerate(classes):
            local[f"catboost_p_{index}"] = probabilities[:, index]
        local["catboost_entropy"] = _entropy(probabilities)
        local["catboost_archetype"] = np.asarray(classes, dtype=object)[
            np.argmax(probabilities, axis=1)
        ]
        local["peak_mfe_available_at"] = local["execution_decision_utc"]
        local["path_catboost_available_at"] = local["execution_decision_utc"]
        local["clean_probability_available_at"] = local["execution_decision_utc"]
        reports[side] = {
            "rows": int(len(local)),
            "peak_clean_feature_count": len(peak_features),
            "path_catboost_feature_count": len(classifier.feature_columns),
        }
        parts.append(local)
    return pd.concat(parts, ignore_index=True), reports


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    spec = json.loads(args.contract_spec.read_text(encoding="utf-8"))
    frame = pd.read_parquet(args.packb_context)
    if frame.duplicated(list(IDENTITY)).any():
        raise ValueError("Pack-B forward context contains duplicate identities")
    frame["__ts__"] = pd.to_datetime(frame["__ts__"], utc=True, errors="raise")
    frame["execution_decision_utc"] = pd.to_datetime(
        frame["execution_decision_utc"], utc=True, errors="raise"
    )
    scored, reports = score_supporting_heads(
        frame,
        spec=spec,
        role_root=args.role_root,
        catboost_root=args.catboost_root,
    )
    head_contract = json.loads(args.head_feature_contract.read_text(encoding="utf-8"))
    raw_features = list(
        dict.fromkeys(
            feature
            for side in SIDES
            for feature in head_contract["feature_columns_by_side"][side]
        )
    )
    # Archetype one-hots are deterministically reconstructed by the final-head
    # scorer from this pre-entry predicted path archetype.
    required_raw = [
        feature
        for feature in raw_features
        if not feature.startswith("catboost_archetype__")
    ]
    missing = sorted(set(required_raw).difference(scored.columns))
    if missing:
        raise ValueError(f"canonical execution-EV pre-entry fields missing: {missing}")
    if not np.isfinite(
        scored.loc[:, required_raw].apply(pd.to_numeric, errors="raise").to_numpy(float)
    ).all():
        raise ValueError("canonical execution-EV pre-entry fields are not finite")
    keep = list(
        dict.fromkeys(
            [
                *IDENTITY,
                "execution_decision_utc",
                "catboost_archetype",
                *required_raw,
                "feature_available_at",
                "base_available_at",
                "residual_available_at",
                "peak_mfe_available_at",
                "path_catboost_available_at",
                "clean_probability_available_at",
            ]
        )
    )
    output = scored.loc[:, keep].copy()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    output_path = args.output_dir / "preentry.parquet"
    output.to_parquet(output_path, index=False, compression="zstd")
    manifest = {
        "schema": SCHEMA,
        "status": "materialized_frozen_preentry_without_outcomes",
        "rows": int(len(output)),
        "columns": len(output.columns),
        "raw_final_head_feature_count": len(raw_features),
        "derived_at_final_head_stage": [
            feature
            for feature in raw_features
            if feature.startswith("catboost_archetype__")
        ],
        "side_reports": reports,
        "inputs": {
            "packb_context": {
                "path": args.packb_context,
                "sha256": _sha256(args.packb_context),
            },
            "contract_spec": {
                "path": args.contract_spec,
                "sha256": _sha256(args.contract_spec),
            },
            "head_feature_contract": {
                "path": args.head_feature_contract,
                "sha256": _sha256(args.head_feature_contract),
            },
        },
        "output": {"path": output_path, "sha256": _sha256(output_path)},
    }
    _write_json(args.output_dir / "manifest.json", manifest)
    return manifest


def _parser(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packb-context", type=Path, required=True)
    parser.add_argument("--contract-spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--role-root", type=Path, default=DEFAULT_ROLE_ROOT)
    parser.add_argument("--catboost-root", type=Path, default=DEFAULT_CATBOOST_ROOT)
    parser.add_argument(
        "--head-feature-contract",
        type=Path,
        default=DEFAULT_HEAD_CONTRACT,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


if __name__ == "__main__":
    print(json.dumps(run(_parser()), indent=2, default=str))
