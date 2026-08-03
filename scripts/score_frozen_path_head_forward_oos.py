#!/usr/bin/env python3
"""Score frozen Peak-MFE and CatBoost path heads on a later, immutable stream.

This is deliberately an inference-only bridge.  Historical OOF files are read
only to bind their hashes and to establish the boundary; they are never
rewritten or appended to.  Rows between an OOF end and the final-refit label
availability time are retained only as clearly labelled retrospective overlap.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.catboost_archetype_classifier import (
    path_archetype_probability_contract,
)
from extreme_price_movements.path_auxiliary_lgbm import (
    fit_base_archetype_label_feature_contract,
    transform_base_archetype_label_features,
)
from extreme_price_movements.path_auxiliary_model_families import (
    compose_peak_predictions,
)
from extreme_price_movements.path_auxiliary_timing_training import (
    predict_side_local_timing_cdf_family,
)
from scripts.run_catboost_path_archetype_classifier import _entropy
from scripts.run_path_auxiliary_lgbm_models import (
    ARCHETYPE_COLUMNS,
    _load_static_features,
    _overlay_handoff_model_features,
)


SIDES = ("long", "short")
IDENTITY = ("__ts__", "__symbol__", "side_name", "candidate_id")
DEFAULT_ROLE_ROOT = ROOT / "data_perp/artifacts/packb_path_auxiliary_role_bundles_20260725_v1_31_8"
DEFAULT_CATBOOST_ROOT = ROOT / "data_perp/reports/catboost_path_archetype_packb31_8_structural_balance_20260725_v1"
DEFAULT_CONTEXT = ROOT / "data_perp/artifacts/packb_downstream_representation_july20_20260726_v1_31_8/context.parquet"
DEFAULT_TARGETS = ROOT / "data_perp/artifacts/packb_path_auxiliary_targets_july20_20260726_v1_31_8/targets.parquet"
DEFAULT_ARCH_LABELS = ROOT / "data_perp/artifacts/path_archetype_labels_july20_20260726_v1/path_archetype_labels.parquet"
DEFAULT_FEATURE_STORE = ROOT / "data_perp/features/20260711_070000"
DEFAULT_OUTPUT = ROOT / "data_perp/artifacts/path_head_forward_oos_july19_20260726_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utc(value: Any) -> pd.Timestamp:
    return pd.Timestamp(value, tz="UTC")


def _normalise(frame: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(set(IDENTITY).difference(frame.columns))
    if missing:
        raise ValueError(f"missing forward identity columns: {missing}")
    output = frame.copy()
    output["__ts__"] = pd.to_datetime(output["__ts__"], utc=True, errors="raise")
    output["__symbol__"] = output["__symbol__"].astype(str)
    output["side_name"] = output["side_name"].astype(str).str.lower()
    output["candidate_id"] = output["candidate_id"].astype(str)
    if output.duplicated(list(IDENTITY)).any() or output["candidate_id"].duplicated().any():
        raise ValueError("forward identity must be unique, including candidate_id")
    if set(output["side_name"].unique()).difference(SIDES):
        raise ValueError("unexpected side in forward stream")
    return output


def _join_exact(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    right = _normalise(right)
    out = left.merge(right, on=list(IDENTITY), how="inner", validate="one_to_one", suffixes=("", "__right"))
    if len(out) != len(left):
        raise ValueError("forward labels/context do not cover the same UTC identity")
    return out.drop(columns=[c for c in out if c.endswith("__right")])


def _refit_available_from_peak(bundle: Mapping[str, Any]) -> pd.Timestamp:
    bound = bundle["final_refit_contract"]["label_resolved_bounds"]["max_utc"]
    return _utc(bound)


def classify_forward_rows(
    timestamps: pd.Series,
    *,
    oof_last_input: pd.Timestamp,
    final_refit_available_at: pd.Timestamp,
) -> pd.DataFrame:
    """Return causal classification for rows already after the frozen OOF stream."""
    ts = pd.to_datetime(timestamps, utc=True, errors="raise")
    if (ts <= oof_last_input).any():
        raise ValueError("forward scorer received a historical OOF-or-earlier row")
    origin = np.where(
        ts >= final_refit_available_at,
        "forward_frozen_final_refit",
        "retrospective_post_oof_overlap",
    )
    return pd.DataFrame({
        "prediction_origin": origin,
        "is_oof": False,
        "is_forward_oos": ts >= final_refit_available_at,
        "promotion_eligible": False,
        "prediction_available_at": final_refit_available_at,
    }, index=timestamps.index)


def _matrix(
    frame: pd.DataFrame,
    *,
    features: Sequence[str],
    feature_store: Path,
    handoff_columns: Sequence[str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    selected = list(dict.fromkeys(map(str, features)))
    matrix, report = _load_static_features(
        frame, feature_dir=feature_store, requested_features=selected, read_cache=None
    )
    matrix, report = _overlay_handoff_model_features(
        matrix, frame, requested_features=selected, static_report=report,
        handoff_feature_columns=handoff_columns,
    )
    # The immutable static store deliberately carries some model-context names
    # as all-null schema placeholders.  Preserve non-null static values, but
    # source an entirely-null placeholder from the exact frozen handoff.
    filled_from_placeholder: list[str] = []
    for feature in selected:
        if feature not in handoff_columns or feature not in frame:
            continue
        values = pd.to_numeric(frame[feature], errors="coerce").to_numpy(dtype=np.float32, copy=False)
        if np.isfinite(values).any() and not np.isfinite(matrix[feature].to_numpy(dtype=np.float32, copy=False)).any():
            matrix[feature] = values
            filled_from_placeholder.append(feature)
    report["handoff_all_null_static_placeholder_features"] = sorted(filled_from_placeholder)
    matrix = matrix.reindex(columns=selected).astype(np.float32, copy=False)
    return matrix, report


def _finite_mask(matrix: pd.DataFrame) -> np.ndarray:
    return np.isfinite(matrix.to_numpy(dtype=np.float32, copy=False)).all(axis=1)


def _peak_feature_contract(
    role_root: Path, side: str) -> tuple[list[str], Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    timing = joblib.load(role_root / "shared/meaningful_mfe_event" / side / "timing_cdf_family.joblib")
    mean = joblib.load(role_root / "roles/peak_mfe_12h_atr__conditional_mean" / side / "role_bundle.joblib")
    q80 = joblib.load(role_root / "roles/peak_mfe_12h_atr__conditional_q80" / side / "role_bundle.joblib")
    state = timing["side_models"][side]
    timing_features = state.get("selected_features_by_horizon")
    if timing_features is None:
        timing_selected = list(state["selected_features"])
    else:
        timing_selected = [f for hour in sorted(timing_features) for f in timing_features[hour]]
    features = list(dict.fromkeys([*timing_selected, *mean["selected_features"], *q80["selected_features"]]))
    return features, timing, mean, q80


def _peak_scores(
    frame: pd.DataFrame,
    *,
    role_root: Path,
    feature_store: Path,
    handoff_columns: Sequence[str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    parts: list[pd.DataFrame] = []
    report: dict[str, Any] = {}
    for side in SIDES:
        local = frame.loc[frame["side_name"].eq(side)].copy()
        features, timing, mean, q80 = _peak_feature_contract(role_root, side)
        matrix, source = _matrix(local, features=features, feature_store=feature_store, handoff_columns=handoff_columns)
        good = _finite_mask(matrix)
        local["peak_feature_complete"] = good
        local["pred_p_meaningful_mfe_12h"] = np.nan
        local["pred_peak_mfe_if_hit_mean_atr"] = np.nan
        local["pred_peak_mfe_if_hit_q80_atr"] = np.nan
        local["pred_expected_peak_mfe_atr"] = np.nan
        if good.any():
            X = matrix.loc[good]
            timing_score = predict_side_local_timing_cdf_family(timing, X, sides=[side] * len(X))
            p_hit = np.clip(timing_score["p_hit_12h"], 0.0, 1.0)
            composed = compose_peak_predictions(
                p_hit,
                np.maximum(np.asarray(mean["final_inference_model"].predict(X.loc[:, mean["selected_features"]]), dtype=float), 0.0),
                np.maximum(np.asarray(q80["final_inference_model"].predict(X.loc[:, q80["selected_features"]]), dtype=float), 0.0),
            )
            local.loc[good, "pred_p_meaningful_mfe_12h"] = composed["p_hit"]
            local.loc[good, "pred_peak_mfe_if_hit_mean_atr"] = composed["conditional_mean_atr"]
            local.loc[good, "pred_peak_mfe_if_hit_q80_atr"] = composed["conditional_q80_atr"]
            local.loc[good, "pred_expected_peak_mfe_atr"] = composed["expected_peak_mfe_atr"]
        report[side] = {"rows": int(len(local)), "scored_rows": int(good.sum()), "features": len(features), "feature_source": source}
        parts.append(local)
        del matrix
        gc.collect()
    return pd.concat(parts, ignore_index=True), report


def _catboost_scores(
    frame: pd.DataFrame,
    *,
    catboost_root: Path,
    feature_store: Path,
    handoff_columns: Sequence[str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    parts: list[pd.DataFrame] = []
    report: dict[str, Any] = {}
    for side in SIDES:
        local = frame.loc[frame["side_name"].eq(side)].copy()
        classifier = joblib.load(catboost_root / f"side={side}" / "path_archetype_classifier.joblib")
        features = list(classifier.feature_columns)
        matrix, source = _matrix(local, features=features, feature_store=feature_store, handoff_columns=handoff_columns)
        good = _finite_mask(matrix)
        local["catboost_feature_complete"] = good
        classes = tuple(map(str, classifier.class_names))
        for name in classes:
            local[f"probability__{name}"] = np.nan
        for name in ("predicted_path_archetype", "predicted_path_shape_archetype"):
            local[name] = None
        for name in ("probability_entropy", "max_probability", "normalized_entropy", "top2_probability_margin", "adverse_probability_mass", "favorable_probability_mass"):
            local[name] = np.nan
        if good.any():
            probability_frame = classifier.predict_proba(matrix.loc[good, features])
            # The persisted classifier returns the seven raw probabilities plus
            # contract diagnostics; score only the named raw taxonomy columns.
            probabilities = probability_frame.loc[:, list(classes)].to_numpy(dtype=np.float64, copy=False)
            for i, name in enumerate(classes):
                local.loc[good, f"probability__{name}"] = probabilities[:, i]
            labels = np.asarray(classes, dtype=object)[np.argmax(probabilities, axis=1)]
            local.loc[good, "predicted_path_archetype"] = labels
            local.loc[good, "predicted_path_shape_archetype"] = labels
            contract = path_archetype_probability_contract(probabilities, classes, index=local.index[good])
            local.loc[good, "probability_entropy"] = _entropy(probabilities)
            for name in ("max_probability", "normalized_entropy", "top2_probability_margin", "adverse_probability_mass", "favorable_probability_mass"):
                local.loc[good, name] = contract[name].to_numpy()
        report[side] = {"rows": int(len(local)), "scored_rows": int(good.sum()), "features": len(features), "feature_source": source}
        parts.append(local)
        del matrix
        gc.collect()
    return pd.concat(parts, ignore_index=True), report


def _economics(frame: pd.DataFrame, *, score: str, target: str) -> dict[str, Any]:
    if target not in frame:
        return {"evaluated_rows": 0, "reason": f"target_not_available:{target}"}
    valid = frame.loc[frame[score].notna() & pd.to_numeric(frame[target], errors="coerce").notna()].copy()
    if valid.empty:
        return {"evaluated_rows": 0}
    valid[target] = pd.to_numeric(valid[target], errors="raise")
    summary: dict[str, Any] = {"evaluated_rows": int(len(valid)), "mean_realized_net_return": float(valid[target].mean())}
    top_n = int(np.ceil(len(valid) * 0.10))
    top = valid.sort_values(score, ascending=False, kind="stable").head(top_n)
    summary["pooled_global_top10"] = {"rows": int(len(top)), "mean_realized_net_return": float(top[target].mean())}
    return summary


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--context", type=Path, default=DEFAULT_CONTEXT)
    p.add_argument("--peak-targets", type=Path, default=DEFAULT_TARGETS)
    p.add_argument("--archetype-labels", type=Path, default=DEFAULT_ARCH_LABELS)
    p.add_argument("--role-root", type=Path, default=DEFAULT_ROLE_ROOT)
    p.add_argument("--catboost-root", type=Path, default=DEFAULT_CATBOOST_ROOT)
    p.add_argument("--feature-store", type=Path, default=DEFAULT_FEATURE_STORE)
    p.add_argument("--end-exclusive", default="2026-07-20T00:00:00+00:00")
    p.add_argument("--head", choices=("all", "peak", "catboost"), default="all")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return p


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    context = _normalise(pd.read_parquet(args.context))
    end = _utc(args.end_exclusive)
    handoff_columns = list(context.columns)
    peak_scored: pd.DataFrame | None = None
    peak_report: dict[str, Any] | None = None
    old_peak = args.role_root / "peak_mfe_12h_atr/oof_bundle.parquet"
    if args.head in ("all", "peak"):
        old_peak_df = pd.read_parquet(old_peak, columns=["__ts__"])
        peak_oof_last = pd.to_datetime(old_peak_df["__ts__"], utc=True).max()
        peak_all = _join_exact(_normalise(pd.read_parquet(args.peak_targets)), context)
        # Recreate only the frozen input encoding; reference rows are strictly pre-May.
        reference = peak_all.loc[peak_all["__ts__"] < _utc("2026-05-01")]
        sources = [name for name in ARCHETYPE_COLUMNS if name in peak_all.columns]
        contract = fit_base_archetype_label_feature_contract(reference, source_columns=sources, canonical_source=sources[0])
        peak_inputs = peak_all.loc[(peak_all["__ts__"] > peak_oof_last) & (peak_all["__ts__"] < end)].reset_index(drop=True)
        peak_features = transform_base_archetype_label_features(peak_inputs, contract)
        peak_inputs = pd.concat([peak_inputs.reset_index(drop=True), peak_features.reset_index(drop=True)], axis=1)
        availability = max(
            _refit_available_from_peak(joblib.load(args.role_root / "roles/peak_mfe_12h_atr__conditional_mean" / side / "role_bundle.joblib"))
            for side in SIDES
        )
        peak_classification = classify_forward_rows(peak_inputs["__ts__"], oof_last_input=peak_oof_last, final_refit_available_at=availability)
        peak_scored, peak_report = _peak_scores(peak_inputs, role_root=args.role_root, feature_store=args.feature_store, handoff_columns=list(peak_inputs.columns))
        peak_scored = pd.concat([peak_scored, peak_classification.reset_index(drop=True)], axis=1)

    cat_scored: pd.DataFrame | None = None
    cat_report: dict[str, Any] | None = None
    old_cat_paths = {side: args.catboost_root / f"side={side}" / "oof_probabilities.parquet" for side in SIDES}
    cat_label_source: Path | None = None
    if args.head in ("all", "catboost"):
        cat_oof_last = max(pd.to_datetime(pd.read_parquet(path, columns=["__ts__"])["__ts__"], utc=True).max() for path in old_cat_paths.values())
        cat_inputs = _join_exact(_normalise(pd.read_parquet(args.archetype_labels)), context)
        cat_inputs = cat_inputs.loc[(cat_inputs["__ts__"] > cat_oof_last) & (cat_inputs["__ts__"] < end)].reset_index(drop=True)
        # Bind final-refit availability to the immutable label source declared by
        # the persisted model rather than guessing a horizon from the new stream.
        cat_run = json.loads((args.catboost_root / "side=long" / "run_manifest.json").read_text(encoding="utf-8"))
        cat_label_source = Path(cat_run["source"])
        if not cat_label_source.is_absolute():
            cat_label_source = ROOT / cat_label_source
        cat_available = pd.to_datetime(
            pd.read_parquet(cat_label_source, columns=["__label_end_ts__"])["__label_end_ts__"], utc=True, errors="raise"
        ).max()
        cat_classification = classify_forward_rows(cat_inputs["__ts__"], oof_last_input=cat_oof_last, final_refit_available_at=cat_available)
        cat_scored, cat_report = _catboost_scores(cat_inputs, catboost_root=args.catboost_root, feature_store=args.feature_store, handoff_columns=list(cat_inputs.columns))
        cat_scored = pd.concat([cat_scored, cat_classification.reset_index(drop=True)], axis=1)

    args.output_dir.mkdir(parents=True, exist_ok=False)
    peak_path = args.output_dir / "peak_mfe_forward_oos_predictions.parquet"
    cat_path = args.output_dir / "catboost_archetype_forward_oos_predictions.parquet"
    if peak_scored is not None:
        peak_scored.to_parquet(peak_path, index=False, compression="zstd")
    if cat_scored is not None:
        cat_scored.to_parquet(cat_path, index=False, compression="zstd")
    result = {
        "schema": "frozen_path_head_forward_oos_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "contract": {
            "historical_oof": "read-only source artifacts; never re-scored or appended",
            "forward_boundary": "rows before final-refit label availability are retrospective overlap, never forward OOS",
            "promotion_eligible": False,
            "evaluation": "monitor-only frozen-final-refit diagnostics; not OOF promotion evidence",
        },
        "sources": {
            "context": {"path": str(args.context), "sha256": _sha256(args.context)},
            "peak_old_oof": ({"path": str(old_peak), "sha256": _sha256(old_peak), "last_input": str(peak_oof_last)} if peak_scored is not None else None),
            "catboost_old_oof": ({side: {"path": str(path), "sha256": _sha256(path)} for side, path in old_cat_paths.items()} if cat_scored is not None else None),
            "catboost_final_refit_label_source": ({"path": str(cat_label_source), "sha256": _sha256(cat_label_source)} if cat_label_source is not None else None),
        },
        "peak_mfe": None if peak_scored is None else {"path": str(peak_path), "availability_at": str(availability), "report": peak_report, "economics": _economics(peak_scored, score="pred_expected_peak_mfe_atr", target="path_final_return_net_1pct")},
        "catboost_archetype": None if cat_scored is None else {"path": str(cat_path), "availability_at": str(cat_available), "report": cat_report, "economics": _economics(cat_scored, score="favorable_probability_mass", target="path_arch_final_return_net_1pct")},
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    outcome = run(_parser().parse_args())
    print(json.dumps({"peak": None if outcome["peak_mfe"] is None else outcome["peak_mfe"]["report"], "catboost": None if outcome["catboost_archetype"] is None else outcome["catboost_archetype"]["report"]}, indent=2, default=str))
