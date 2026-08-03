#!/usr/bin/env python3
"""Score frozen Pack-B base and residual final refits on future candidates.

Input rows are the complete point-in-time hourly candidate feature matrix for
both sides.  This scorer applies the frozen 31/8 side-local base models, selects
the deterministic top 40% within UTC timestamp x side, constructs the exact
base-margin/rank context, and then applies the frozen side-local residual model
and EV map.  It never consumes labels or outcomes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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

from extreme_price_movements.base_candidate_population import (  # noqa: E402
    BaseCandidatePopulationContract,
    deterministic_candidate_ids,
    select_base_candidate_population,
)
from extreme_price_movements.path_auxiliary_lgbm import (  # noqa: E402
    transform_base_archetype_label_features,
)
from scripts.run_packb_side_local_residual_oof import _predict_ev_map  # noqa: E402


SCHEMA = "packb_final_refits_forward_v1"
SIDES = ("long", "short")
IDENTITY = ("candidate_id", "__ts__", "__symbol__", "side_name")
DEFAULT_SPEC = Path(
    "configs/execution_ev_forward_confirmation_candidate_20260728_v1.json"
)
DEFAULT_ALPHA_MANIFEST = Path(
    "data_perp/artifacts/execution_ev_alpha_oof_july20_20260726_v1/manifest.json"
)
DEFAULT_SUPPORT_CONTEXT = Path(
    "data_perp/artifacts/packb_downstream_context_july20_20260726_v1_31_8/"
    "context.parquet"
)
DEFAULT_RESIDUAL_ROOT = Path(
    "data_perp/artifacts/packb_side_local_residual_oof_july20_20260726_v1_31_8"
)
DEFAULT_OUTPUT = Path(
    "data_perp/artifacts/packb_final_refits_forward_20260728_v1"
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


def build_base_context(
    population: pd.DataFrame,
    *,
    top_fraction: float = 0.40,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select the canonical top fraction and construct downstream rank fields."""

    contract = BaseCandidatePopulationContract(
        top_fraction=top_fraction,
        score_col="prediction",
    )
    selected = select_base_candidate_population(population, contract)
    group_keys = ["__ts__", "side_name"]
    full = population.copy()
    full["base_rank_decile"] = np.clip(
        np.floor(
            full.groupby(group_keys, sort=False)["prediction"].rank(
                method="first", ascending=False, pct=True
            )
            * 10.0
        ).astype(int),
        0,
        9,
    )
    group_stats = full.groupby(group_keys, sort=False)["prediction"].agg(
        group_score_mean="mean",
        group_score_std="std",
    )
    decile_stats = full.groupby(
        [*group_keys, "base_rank_decile"], sort=False
    )["prediction"].agg(
        decile_score_mean="mean",
        decile_score_std="std",
    )
    cutoff = (
        selected.groupby(group_keys, sort=False)["prediction"]
        .min()
        .rename("base_cutoff_score")
    )
    output = selected.copy()
    output["base_rank_decile"] = np.clip(
        np.floor(output["base_candidate_rank_pct_timestamp_side"] * 10.0).astype(int),
        0,
        9,
    )
    output = output.join(group_stats, on=group_keys)
    output = output.join(decile_stats, on=[*group_keys, "base_rank_decile"])
    output = output.join(cutoff, on=group_keys)
    output["score"] = output["prediction"].astype(np.float32)
    output["base_oof_score"] = output["prediction"].astype(np.float32)
    output["base_margin_to_cutoff"] = (
        output["prediction"] - output["base_cutoff_score"]
    ).astype(np.float32)
    safe_group_std = output["group_score_std"].where(
        output["group_score_std"].gt(1e-12)
    )
    output["base_margin_to_cutoff_z"] = (
        output["base_margin_to_cutoff"].div(safe_group_std).fillna(0.0).astype(np.float32)
    )
    safe_decile_std = output["decile_score_std"].where(
        output["decile_score_std"].gt(1e-12)
    )
    output["base_signal_zscore_within_archetype"] = (
        (output["prediction"] - output["decile_score_mean"])
        .div(safe_decile_std)
        .fillna(0.0)
        .astype(np.float32)
    )
    output["base_score_z_timestamp_side"] = (
        (output["prediction"] - output["group_score_mean"])
        .div(safe_group_std)
        .fillna(0.0)
        .astype(np.float32)
    )
    output["archetype_label_family"] = output["base_rank_decile"].map(
        lambda value: f"base_rank_decile_{int(value)}"
    )
    output["archetype_policy_key"] = output["archetype_label_family"]
    return output, full


def _load_role(spec: Mapping[str, Any], role: str) -> tuple[Path, Mapping[str, Any]]:
    matches = [record for record in spec["models"] if record["role"] == role]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one {role} record")
    record = matches[0]
    path = Path(record["path"])
    if not path.is_absolute():
        path = ROOT / path
    if _sha256(path) != record["sha256"]:
        raise ValueError(f"{role} model hash mismatch")
    contract_path = Path(record["feature_contract"]["path"])
    if not contract_path.is_absolute():
        contract_path = ROOT / contract_path
    if _sha256(contract_path) != record["feature_contract"]["sha256"]:
        raise ValueError(f"{role} feature contract hash mismatch")
    return path, json.loads(contract_path.read_text(encoding="utf-8"))


def _matrix(frame: pd.DataFrame, columns: Sequence[str], *, name: str) -> pd.DataFrame:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"{name} features missing: {missing}")
    matrix = frame.loc[:, list(columns)].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(matrix.to_numpy(dtype=np.float32)).all():
        raise ValueError(f"{name} features must be finite")
    return matrix.astype(np.float32)


def _support_counts(
    forward: pd.DataFrame,
    historical_context: pd.DataFrame,
) -> np.ndarray:
    columns = ("archetype_label_family", "archetype_policy_key")
    counts = (
        historical_context.groupby(list(columns), dropna=False)
        .size()
        .rename("support")
        .reset_index()
    )
    aligned = forward.loc[:, list(columns)].merge(
        counts,
        on=list(columns),
        how="left",
        sort=False,
    )
    return np.log1p(aligned["support"].fillna(0).to_numpy(dtype=float))


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    spec = json.loads(args.contract_spec.read_text(encoding="utf-8"))
    frame = pd.read_parquet(args.candidate_features)
    required = {"__ts__", "__symbol__", "side_name", "execution_decision_utc", "feature_available_at"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"future candidate matrix missing: {missing}")
    frame["__ts__"] = pd.to_datetime(frame["__ts__"], utc=True, errors="raise")
    frame["execution_decision_utc"] = pd.to_datetime(
        frame["execution_decision_utc"], utc=True, errors="raise"
    )
    frame["feature_available_at"] = pd.to_datetime(
        frame["feature_available_at"], utc=True, errors="raise"
    )
    if (frame["feature_available_at"] > frame["execution_decision_utc"]).any():
        raise ValueError("future candidate features occur after decision")
    frame["side_name"] = frame["side_name"].astype(str).str.lower()
    if set(frame["side_name"]) != set(SIDES):
        raise ValueError("future candidate matrix must contain both sides")
    if "candidate_id" not in frame:
        frame["candidate_id"] = deterministic_candidate_ids(frame, timeframe="1h")
    if frame.duplicated(list(IDENTITY)).any() or frame["candidate_id"].duplicated().any():
        raise ValueError("future candidate identity is not unique")

    from lightgbm import Booster

    base_parts: list[pd.DataFrame] = []
    model_records: dict[str, Any] = {}
    for side in SIDES:
        path, manifest = _load_role(spec, f"base_{side}")
        features = list(manifest["features"])
        local = frame.loc[frame["side_name"].eq(side)].copy()
        model = Booster(model_file=str(path))
        local["prediction"] = np.asarray(
            model.predict(_matrix(local, features, name=f"base_{side}")),
            dtype=float,
        )
        local["prediction_source"] = "frozen_final_refit"
        base_parts.append(local)
        model_records[f"base_{side}"] = {
            "path": path,
            "sha256": _sha256(path),
            "feature_count": len(features),
        }
    population = pd.concat(base_parts, ignore_index=True)
    context, _ = build_base_context(population, top_fraction=0.40)
    context["base_prediction"] = context["prediction"].astype(np.float32)
    context["base_rank_timestamp_side"] = context[
        "base_candidate_rank_timestamp_side"
    ].astype(np.float32)
    context["base_rank_pct_timestamp_side"] = context[
        "base_candidate_rank_pct_timestamp_side"
    ].astype(np.float32)

    residual_parts: list[pd.DataFrame] = []
    for side in SIDES:
        path, contract = _load_role(spec, f"residual_{side}")
        features = list(contract["features"])
        local = context.loc[context["side_name"].eq(side)].copy()
        model = Booster(model_file=str(path))
        residual_root = args.residual_root / side / "final_refit"
        ev_map_path = residual_root / "baseline_ev_map.joblib"
        ev_map = joblib.load(ev_map_path)
        base_expected = _predict_ev_map(
            ev_map, local["prediction"].to_numpy(dtype=float)
        )
        delta = np.asarray(
            model.predict(_matrix(local, features, name=f"residual_{side}")),
            dtype=float,
        )
        alpha = float(contract["alpha"])
        local["base_alpha_ev"] = base_expected
        local["residual_delta_ev"] = delta
        local["existing_alpha_ev"] = base_expected + alpha * delta
        local["alpha_prediction_uncertainty"] = np.abs(
            local["existing_alpha_ev"] - local["base_alpha_ev"]
        )
        local["base_available_at"] = local["execution_decision_utc"]
        local["residual_available_at"] = local["execution_decision_utc"]
        residual_parts.append(local)
        model_records[f"residual_{side}"] = {
            "path": path,
            "sha256": _sha256(path),
            "ev_map_path": ev_map_path,
            "ev_map_sha256": _sha256(ev_map_path),
            "feature_count": len(features),
            "alpha": alpha,
        }
    output = pd.concat(residual_parts, ignore_index=True)
    alpha_manifest = json.loads(args.alpha_manifest.read_text(encoding="utf-8"))
    alpha_contract = alpha_manifest["definitions"][
        "base_archetype_label_feature_contract"
    ]
    onehot = transform_base_archetype_label_features(output, alpha_contract)
    output = pd.concat([output, onehot], axis=1)
    support_context = pd.read_parquet(
        args.support_context,
        columns=["archetype_label_family", "archetype_policy_key"],
    )
    output["alpha_leaf_support"] = _support_counts(output, support_context).astype(
        np.float32
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    output_path = args.output_dir / "packb_forward_context.parquet"
    output.to_parquet(output_path, index=False, compression="zstd")
    manifest = {
        "schema": SCHEMA,
        "status": "frozen_final_refit_preentry_context_not_oos_metrics",
        "contract": {
            "base": "side-local frozen 31/8 final refits",
            "candidate_selection": "top 40% within UTC timestamp x side",
            "residual": "side-local frozen residual model plus frozen base EV map",
            "base_archetype": "pre-entry base rank decile only",
            "outcomes_used": False,
        },
        "rows": {
            "input": int(len(frame)),
            "selected_top40": int(len(output)),
            "by_side": {
                side: int(output["side_name"].eq(side).sum()) for side in SIDES
            },
        },
        "decision_bounds": {
            "min": output["execution_decision_utc"].min(),
            "max": output["execution_decision_utc"].max(),
        },
        "models": model_records,
        "inputs": {
            "candidate_features": {
                "path": args.candidate_features,
                "sha256": _sha256(args.candidate_features),
            },
            "contract_spec": {
                "path": args.contract_spec,
                "sha256": _sha256(args.contract_spec),
            },
            "alpha_manifest": {
                "path": args.alpha_manifest,
                "sha256": _sha256(args.alpha_manifest),
            },
            "support_context": {
                "path": args.support_context,
                "sha256": _sha256(args.support_context),
            },
        },
        "output": {"path": output_path, "sha256": _sha256(output_path)},
    }
    _write_json(args.output_dir / "manifest.json", manifest)
    return manifest


def _parser(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-features", type=Path, required=True)
    parser.add_argument("--contract-spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--alpha-manifest", type=Path, default=DEFAULT_ALPHA_MANIFEST)
    parser.add_argument("--support-context", type=Path, default=DEFAULT_SUPPORT_CONTEXT)
    parser.add_argument("--residual-root", type=Path, default=DEFAULT_RESIDUAL_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


if __name__ == "__main__":
    print(json.dumps(run(_parser()), indent=2, default=str))
