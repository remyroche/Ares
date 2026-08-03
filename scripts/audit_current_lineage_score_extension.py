#!/usr/bin/env python3
"""Materialise the valid current-lineage transition overlap and its extension contract.

This is deliberately a *lineage audit*, not a score backfill.  It publishes
the largest existing strict-OOF current-lineage panel with exact candidate
identity, base/residual/alpha/execution-EV scores and grouped-OOF transition
probabilities.  It then proves, field by field, whether an older historical
extension can be reconstructed without silently substituting a legacy score
route or a future-trained model.

In particular, the historical 2025 two-layer score archives are never copied
into the output panel: they use a fold-local top-40 raw route and their
economics are explicitly invalidated.  They are retained only as exact-label
and identity evidence for a future, fresh, chronological reconstruction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
IDENTITY = ("__ts__", "__symbol__", "side_name", "candidate_id")
SIDES = ("long", "short")
MAPPED_SCORE = "causal_recent_side_isotonic_ev"
MAPPED_OOF = f"{MAPPED_SCORE}__is_oof"

DEFAULT_MAPPED = ROOT / (
    "data_perp/artifacts/execution_ev_context_clean_recent_mapping_forward_"
    "july19_20260726_v1/mapped_oof.parquet"
)
DEFAULT_BASE = ROOT / (
    "data_perp/artifacts/packb_side_local_outer_oof_july20_20260726_v1_31_8/"
    "oof_predictions.parquet"
)
DEFAULT_RESIDUAL = ROOT / (
    "data_perp/artifacts/packb_side_local_residual_oof_july20_20260726_v1_31_8/"
    "oof_predictions.parquet"
)
DEFAULT_ALPHA = ROOT / (
    "data_perp/artifacts/execution_ev_alpha_oof_july20_20260726_v1/"
    "alpha_oof.parquet"
)
DEFAULT_ACTIVE = ROOT / (
    "data_perp/artifacts/regime_transition_active_head_20260726_v1/"
    "grouped_oof.parquet"
)
DEFAULT_HISTORICAL = ROOT / (
    "data_perp/artifacts/janapr2025_execution_ev_exact1m_two_layer_oof_"
    "20260727_v1/two_layer_direct_ev_strict_oof.parquet"
)
DEFAULT_HISTORICAL_LABELS = ROOT / (
    "data_perp/artifacts/janapr2025_execution_ev_exact1m_two_layer_oof_"
    "20260727_v1/exact_1m_execution_ev_12h_labels.parquet"
)
DEFAULT_HISTORICAL_INVALIDATION = DEFAULT_HISTORICAL.parent / "ECONOMIC_INVALIDATION.json"
DEFAULT_FEATURE_STORE = ROOT / "data_perp/features/20260711_070000"
DEFAULT_BASE_PROMOTION = ROOT / "docs/pipeline_roadmap/20260724/r3/packb_side_fs_hpo_promotion_v1.json"
DEFAULT_RESIDUAL_ROOT = ROOT / "data_perp/artifacts/packb_side_local_residual_oof_july20_20260726_v1_31_8"
DEFAULT_AE_ROOT = ROOT / "data_perp/artifacts/packb_side_local_ae_20260724_v1"
DEFAULT_EXECUTION_BUNDLE = ROOT / (
    "data_perp/artifacts/execution_ev_context_head_clean_20260726_v1/"
    "execution_ev_model_ablation_bundle.joblib"
)
DEFAULT_FORWARD_MANIFEST = ROOT / (
    "data_perp/artifacts/execution_ev_context_head_clean_forward_july19_"
    "20260726_v2/manifest.json"
)
DEFAULT_OUTPUT = ROOT / (
    "data_perp/artifacts/current_lineage_score_extension_readiness_20260727_v2"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_safe(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(_safe(dict(payload)), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _normalise_identity(frame: pd.DataFrame, *, name: str) -> pd.DataFrame:
    missing = sorted(set(IDENTITY).difference(frame.columns))
    if missing:
        raise ValueError(f"{name} missing exact identity fields: {missing}")
    result = frame.copy()
    result["__ts__"] = pd.to_datetime(result["__ts__"], utc=True, errors="raise")
    result["__symbol__"] = result["__symbol__"].astype(str)
    result["side_name"] = result["side_name"].astype(str).str.lower()
    result["candidate_id"] = result["candidate_id"].astype(str)
    if not result["side_name"].isin(SIDES).all():
        raise ValueError(f"{name} has noncanonical side values")
    if result.duplicated(list(IDENTITY), keep=False).any():
        raise ValueError(f"{name} has duplicate exact candidate identities")
    return result


def _identity_columns(frame: pd.DataFrame, requested: Sequence[str]) -> pd.DataFrame:
    columns = list(dict.fromkeys([*IDENTITY, *requested]))
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"source is missing required columns: {missing}")
    return _normalise_identity(frame.loc[:, columns], name="score source")


def _date_summary(frame: pd.DataFrame, *, time_column: str = "__ts__") -> dict[str, Any]:
    values = pd.to_datetime(frame[time_column], utc=True, errors="raise")
    return {
        "rows": int(len(frame)),
        "hours": int(values.nunique()),
        "start_utc": values.min(),
        "end_utc": values.max(),
        "observed_utc_days": int(values.dt.floor("D").nunique()),
    }


def build_current_score_panel(
    mapped: pd.DataFrame,
    base: pd.DataFrame,
    residual: pd.DataFrame,
    alpha: pd.DataFrame,
) -> pd.DataFrame:
    """Return only existing strict current-lineage OOF scores, never forward rows."""

    required_mapped = (
        "execution_decision_utc",
        "execution_label_end_utc",
        "execution_gross_ev_12h",
        "execution_net_ev_12h",
        "evaluation_origin",
        "catboost__residual__without_hpo__all_features",
        MAPPED_SCORE,
        MAPPED_OOF,
    )
    work = _identity_columns(mapped, required_mapped)
    if not work[MAPPED_OOF].fillna(False).astype(bool).any():
        raise ValueError("mapped score source has no strict OOF rows")
    work = work.loc[work[MAPPED_OOF].fillna(False).astype(bool)].copy()
    work["execution_decision_utc"] = pd.to_datetime(
        work["execution_decision_utc"], utc=True, errors="raise"
    )
    work["execution_label_end_utc"] = pd.to_datetime(
        work["execution_label_end_utc"], utc=True, errors="raise"
    )
    if not work["execution_decision_utc"].eq(work["__ts__"] + pd.Timedelta(hours=1)).all():
        raise ValueError("current mapped scores violate the signal-to-decision contract")
    if work["execution_label_end_utc"].lt(work["execution_decision_utc"]).any():
        raise ValueError("current mapped scores have an invalid label resolution time")

    base_columns = ("prediction", "outer_fold", "prediction_source")
    base_work = _identity_columns(base, base_columns).rename(
        columns={
            "prediction": "base_31_8_oof_score",
            "outer_fold": "base_31_8_oof_fold",
            "prediction_source": "base_31_8_score_provenance",
        }
    )
    residual_columns = (
        "prediction",
        "residual_oof_fold",
        "base_expected_ev",
        "residual_delta_ev",
        "residual_expected_ev",
        "residual_is_oof",
    )
    residual_work = _identity_columns(residual, residual_columns).rename(
        columns={
            "prediction": "residual_31_8_oof_prediction",
            "residual_oof_fold": "residual_31_8_oof_fold",
            "base_expected_ev": "residual_base_expected_ev",
            "residual_delta_ev": "residual_delta_ev",
            "residual_expected_ev": "residual_expected_ev",
            "residual_is_oof": "residual_is_oof",
        }
    )
    alpha_columns = (
        "existing_alpha_ev",
        "base_alpha_ev",
        "alpha_prediction_uncertainty",
        "alpha_leaf_support",
        "oof_fold",
    )
    alpha_work = _identity_columns(alpha, alpha_columns).rename(
        columns={"oof_fold": "alpha_oof_fold"}
    )
    for name, right in (("base", base_work), ("residual", residual_work), ("alpha", alpha_work)):
        work = work.merge(right, on=list(IDENTITY), how="left", validate="one_to_one")
        if work[right.columns.difference(IDENTITY)].isna().all(axis=1).any():
            missing = int(work[right.columns.difference(IDENTITY)].isna().all(axis=1).sum())
            raise ValueError(f"current strict mapped OOF lacks {name} lineage rows: {missing}")
    if not work["residual_is_oof"].fillna(False).astype(bool).all():
        raise ValueError("current strict mapped OOF joins a non-OOF residual score")
    numeric = (
        "catboost__residual__without_hpo__all_features",
        MAPPED_SCORE,
        "base_31_8_oof_score",
        "residual_31_8_oof_prediction",
        "residual_expected_ev",
        "existing_alpha_ev",
    )
    for column in numeric:
        work[column] = pd.to_numeric(work[column], errors="coerce")
    if not np.isfinite(work.loc[:, list(numeric)].to_numpy(dtype=np.float64)).all():
        raise ValueError("current score panel contains non-finite score values")
    return work.sort_values(["__ts__", "__symbol__", "side_name", "candidate_id"], kind="stable").reset_index(drop=True)


def attach_transition_scores(panel: pd.DataFrame, active: pd.DataFrame) -> pd.DataFrame:
    required = {"source_utc", "target__event_id", "target__transition_active", "prediction"}
    missing = sorted(required.difference(active.columns))
    if missing:
        raise ValueError(f"active transition source misses fields: {missing}")
    transition = active.loc[:, sorted(required)].copy()
    transition["source_utc"] = pd.to_datetime(transition["source_utc"], utc=True, errors="raise")
    if transition["source_utc"].duplicated().any():
        raise ValueError("grouped active OOF must have exactly one row per source hour")
    transition = transition.rename(columns={"prediction": "active_transition_probability_grouped_oof"})
    output = panel.merge(
        transition,
        left_on="__ts__",
        right_on="source_utc",
        how="left",
        validate="many_to_one",
    )
    return output.sort_values(["__ts__", "__symbol__", "side_name", "candidate_id"], kind="stable").reset_index(drop=True)


def transition_coverage(frame: pd.DataFrame) -> dict[str, Any]:
    matched = frame.loc[frame["active_transition_probability_grouped_oof"].notna()].copy()
    matched_events = matched.loc[matched["target__event_id"].notna()].copy()
    events = (
        matched_events.groupby("target__event_id", sort=True)
        .agg(
            start_utc=("__ts__", "min"),
            end_utc=("__ts__", "max"),
            candidate_rows=("candidate_id", "size"),
            active_candidate_rows=("target__transition_active", "sum"),
        )
        .reset_index()
    )
    return {
        "transition_score_rows": int(len(matched)),
        "transition_score_row_fraction": float(len(matched) / max(len(frame), 1)),
        "transition_hours": int(matched["__ts__"].nunique()),
        "events_overlapped": int(len(events)),
        "events_with_active_hours": int((events["active_candidate_rows"] > 0).sum()),
        "event_rows": events.to_dict("records"),
        "meets_five_independent_event_target": bool(
            int((events["active_candidate_rows"] > 0).sum()) >= 5
        ),
    }


def _schema_columns(feature_store: Path) -> set[str]:
    examples = sorted(feature_store.glob("symbol=*.parquet"))
    example = examples[0] if examples else None
    if example is None:
        raise FileNotFoundError(f"no feature-store symbol files under {feature_store}")
    return set(pq.read_schema(example).names)


def _load_promotion_routes(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    routes: dict[str, dict[str, Any]] = {}
    for side in SIDES:
        route = payload.get("sides", {}).get(side, {})
        root = ROOT / str(route.get("source_root", ""))
        feature_path = root / "feature_contract.json"
        parameter_path = root / "hpo_parameters.json"
        if not feature_path.is_file() or not parameter_path.is_file():
            raise FileNotFoundError(f"canonical {side} feature/HPO contract is absent")
        feature_contract = json.loads(feature_path.read_text(encoding="utf-8"))
        routes[side] = {
            "features": list(map(str, feature_contract.get("selected_features", []))),
            "feature_contract": feature_path,
            "hpo_parameters": parameter_path,
            "selected_trial_id": route.get("selected_trial_id"),
        }
    return routes


def historical_reconstruction_readiness(
    *,
    historical: pd.DataFrame,
    feature_store: Path,
    promotion_path: Path,
    residual_root: Path,
    ae_root: Path,
    execution_bundle: Path,
    forward_manifest: Path,
    invalidation_path: Path,
) -> dict[str, Any]:
    """Describe the reconstructible base/residual surface and hard downstream blockers."""

    historical = _normalise_identity(historical, name="historical strict OOF archive")
    source_symbols = set(historical["__symbol__"].astype(str))
    feature_symbols = {
        path.name[len("symbol=") : -len(".parquet")]
        for path in feature_store.glob("symbol=*.parquet")
    }
    raw_columns = _schema_columns(feature_store)
    routes = _load_promotion_routes(promotion_path)
    base: dict[str, Any] = {}
    residual: dict[str, Any] = {}
    for side in SIDES:
        features = routes[side]["features"]
        generated = [name for name in features if name.startswith(("dae_", "gmm_"))]
        raw = [name for name in features if name not in generated]
        state_path = ae_root / side / "ae_gmm" / "ae_gmm_state.pkl"
        base[side] = {
            "feature_count": len(features),
            "features": features,
            "raw_feature_columns_missing_from_store_schema": sorted(set(raw).difference(raw_columns)),
            "generated_features": generated,
            "frozen_side_ae_gmm_state": str(state_path),
            "frozen_side_ae_gmm_state_present": state_path.is_file(),
            "frozen_hpo_contract": str(routes[side]["hpo_parameters"]),
            "frozen_hpo_contract_present": routes[side]["hpo_parameters"].is_file(),
            "selected_trial_id": routes[side]["selected_trial_id"],
        }
        residual_feature_path = residual_root / side / "feature_contract.json"
        residual_hpo_path = residual_root / side / "hpo_contract.json"
        residual_features = json.loads(residual_feature_path.read_text(encoding="utf-8")).get("features", [])
        static = [name for name in residual_features if not str(name).startswith("base_")]
        residual[side] = {
            "feature_count": len(residual_features),
            "frozen_feature_contract": str(residual_feature_path),
            "frozen_hpo_contract": str(residual_hpo_path),
            "frozen_hpo_contract_present": residual_hpo_path.is_file(),
            "derivable_base_geometry_fields": [name for name in residual_features if str(name).startswith("base_")],
            "static_feature_columns_missing_from_store_schema": sorted(set(static).difference(raw_columns)),
        }
    invalidation = json.loads(invalidation_path.read_text(encoding="utf-8")) if invalidation_path.is_file() else {}
    forward = json.loads(forward_manifest.read_text(encoding="utf-8"))
    model_inputs = list(map(str, forward.get("raw_feature_columns", [])))
    missing_historical_heads = [
        "historical strict-OOF CatBoost path-archetype predictions",
        "historical strict-OOF peak-MFE predictions",
        "historical deployed-policy path targets for CatBoost/peak retraining",
        "historical deployed-policy exact 12h execution-EV labels",
        "historical fold-local execution-EV meta bundles fitted only on prior resolved rows",
        "historical causal recent-EV mapping state fitted only on prior resolved rows",
    ]
    base_ready = all(
        not item["raw_feature_columns_missing_from_store_schema"]
        and item["frozen_side_ae_gmm_state_present"]
        and item["frozen_hpo_contract_present"]
        for item in base.values()
    ) and source_symbols.issubset(feature_symbols)
    residual_ready = all(
        not item["static_feature_columns_missing_from_store_schema"]
        and item["frozen_hpo_contract_present"]
        for item in residual.values()
    )
    return {
        "historical_identity_and_label_foundation": {
            **_date_summary(historical),
            "candidate_symbols": int(len(source_symbols)),
            "feature_store_symbol_coverage": float(len(source_symbols.intersection(feature_symbols)) / max(len(source_symbols), 1)),
            "missing_feature_store_symbols": sorted(source_symbols.difference(feature_symbols)),
            "strict_oof_archive_route": "legacy_fold_local_top40_raw_two_layer",
            "allowed_use": "identity, chronological fold and candidate-label foundation only",
            "prohibited_use": "do not pool/archive score as canonical current-lineage or deployed-policy economics",
            "economic_invalidation": invalidation,
        },
        "canonical_31_8_base_reconstruction": {
            "ready_to_materialize_after_chronological_training": base_ready,
            "side_contracts": base,
            "required_training": (
                "Fit fresh side-local chronological base models using the frozen 31/8 "
                "features and frozen side AE/GMM transform; do not score with a later final refit."
            ),
        },
        "canonical_residual_reconstruction": {
            "feature_surface_ready_after_base_oof": residual_ready,
            "side_contracts": residual,
            "required_training": (
                "Rebuild top-40 base admission and fit fresh side-local residual folds from "
                "only prior resolved first-touch labels; do not reuse 2026 residual boosters."
            ),
        },
        "execution_ev_reconstruction": {
            "frozen_2026_bundle_present": execution_bundle.is_file(),
            "frozen_2026_bundle_forbidden_for_historical_oof": True,
            "current_winner_input_columns": model_inputs,
            "missing_historical_requirements": missing_historical_heads,
            "blocked": True,
            "block_reason": (
                "The available Jan-Apr exact labels are fee-only and explicitly invalidated for "
                "deployed-policy economics.  The complete CatBoost/peak/execution-EV head stack "
                "cannot be made historical OOF until deployed-policy targets and chronological "
                "per-head OOF training are materialized."
            ),
        },
        "required_sequence": [
            "Materialize deployed-policy exact 1m 12h labels for the historical exact identities.",
            "Recreate frozen 31/8 side-local base OOF from point-in-time feature-store values.",
            "Recreate base top-40 geometry and frozen-contract side-local residual OOF.",
            "Materialize historical per-side CatBoost and peak-head OOF from the repaired path targets.",
            "Fit execution-EV meta folds only on prior resolved head OOF and labels, then fit a causal recent-EV map.",
            "Join grouped transition OOF by exact source hour; require at least five independent active events before policy economics.",
        ],
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True)
    mapped = pd.read_parquet(args.mapped)
    panel = build_current_score_panel(
        mapped,
        pd.read_parquet(args.base),
        pd.read_parquet(args.residual),
        pd.read_parquet(args.alpha),
    )
    joined = attach_transition_scores(panel, pd.read_parquet(args.active))
    overlap = joined.loc[joined["active_transition_probability_grouped_oof"].notna()].copy()
    panel_path = output / "current_lineage_strict_oof_scores.parquet"
    overlap_path = output / "current_lineage_strict_oof_transition_overlap.parquet"
    panel.to_parquet(panel_path, index=False)
    overlap.to_parquet(overlap_path, index=False)
    historical = pd.read_parquet(args.historical, columns=list(IDENTITY))
    readiness = historical_reconstruction_readiness(
        historical=historical,
        feature_store=Path(args.feature_store),
        promotion_path=Path(args.base_promotion),
        residual_root=Path(args.residual_root),
        ae_root=Path(args.ae_root),
        execution_bundle=Path(args.execution_bundle),
        forward_manifest=Path(args.forward_manifest),
        invalidation_path=Path(args.historical_invalidation),
    )
    coverage = transition_coverage(joined)
    manifest = {
        "schema": "current_lineage_score_extension_readiness_v1",
        "status": (
            "PARTIAL_CURRENT_LINEAGE_PANEL_MATERIALIZED_FULL_HISTORICAL_EXTENSION_BLOCKED"
        ),
        "strict_oof_only": True,
        "current_lineage_panel": {
            **_date_summary(panel),
            "score_columns": [
                "base_31_8_oof_score",
                "residual_31_8_oof_prediction",
                "residual_expected_ev",
                "existing_alpha_ev",
                "catboost__residual__without_hpo__all_features",
                MAPPED_SCORE,
            ],
            "path": str(panel_path),
            "sha256": _sha256(panel_path),
        },
        "transition_overlap": {
            **coverage,
            "path": str(overlap_path),
            "sha256": _sha256(overlap_path),
        },
        "historical_extension": readiness,
        "source_hashes": {
            str(path): _sha256(path)
            for path in (
                Path(args.mapped), Path(args.base), Path(args.residual), Path(args.alpha),
                Path(args.active), Path(args.historical), Path(args.historical_labels),
                Path(args.base_promotion), Path(args.forward_manifest), Path(args.historical_invalidation),
            )
            if path.is_file()
        },
    }
    _write_json(output / "readiness.json", manifest)
    return {
        "output_dir": str(output),
        "current_oof_rows": int(len(panel)),
        "transition_overlap_events": int(coverage["events_with_active_hours"]),
        "full_historical_extension_blocked": True,
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--mapped", type=Path, default=DEFAULT_MAPPED)
    result.add_argument("--base", type=Path, default=DEFAULT_BASE)
    result.add_argument("--residual", type=Path, default=DEFAULT_RESIDUAL)
    result.add_argument("--alpha", type=Path, default=DEFAULT_ALPHA)
    result.add_argument("--active", type=Path, default=DEFAULT_ACTIVE)
    result.add_argument("--historical", type=Path, default=DEFAULT_HISTORICAL)
    result.add_argument("--historical-labels", type=Path, default=DEFAULT_HISTORICAL_LABELS)
    result.add_argument("--historical-invalidation", type=Path, default=DEFAULT_HISTORICAL_INVALIDATION)
    result.add_argument("--feature-store", type=Path, default=DEFAULT_FEATURE_STORE)
    result.add_argument("--base-promotion", type=Path, default=DEFAULT_BASE_PROMOTION)
    result.add_argument("--residual-root", type=Path, default=DEFAULT_RESIDUAL_ROOT)
    result.add_argument("--ae-root", type=Path, default=DEFAULT_AE_ROOT)
    result.add_argument("--execution-bundle", type=Path, default=DEFAULT_EXECUTION_BUNDLE)
    result.add_argument("--forward-manifest", type=Path, default=DEFAULT_FORWARD_MANIFEST)
    result.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return result


if __name__ == "__main__":
    print(json.dumps(_safe(run(parser().parse_args())), indent=2, sort_keys=True))
