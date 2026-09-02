#!/usr/bin/env python3
"""Stable-geometry follow-up for the strict-R3 long supportive-label funnel.

Stage 1 deliberately refits the realised-path representation per outer fold to
answer a label-quality question.  Such fold-local clusters are valid for that
diagnostic, but their nominal component IDs are not a deployable contract.

This runner therefore freezes a compact GMM path geometry once, on valid
October--December 2024 labels, and keeps that representation fixed for every
2025--2026 fold.  The geometry-definition rows are *never* used by the
supervised membership or residual learners.  Every later outer-fold model is
fit only on labels resolved before the held fold, with the usual H12 embargo.

The runner compares only a small, predeclared set of uses for the causal path
recogniser:

* S2 frozen-GMM expected policy value;
* S3 equal-bps blend of the frozen-GMM value and frozen upstream value;
* S3 one shared Huber policy-residual model using strict-inner-OOF GMM
  memberships.  This is not a routed local expert and it has no authority to
  alter live artifacts.

All realised path coordinates remain target-side quantities.  At transform
time models receive only the frozen causal base contract, permitted
prequential base outputs, and causal probabilities emitted by a prior-trained
membership recogniser.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
import pandas as pd

from run_strict_r3_long_supportive_label_funnel import (
    DEFAULT_LABELS,
    DEFAULT_LEDGER,
    EMBARGO,
    FOLDS,
    IDENTITY,
    MAX_CLUSTER_ROWS,
    MAX_TRAIN_ROWS,
    P1_FUTURE_FIELDS,
    SEED,
    _align_probabilities,
    _class_policy_priors,
    _score_eligible,
    _train_path_eligible,
    _fit_gmm,
    _fit_p1_transform,
    _joined_population,
    _ledger_fields,
    _matrix,
    _model_classifier,
    _model_regressor,
    _quality_metrics,
    _sample_month_balanced,
    _sha256,
    _transform_p1,
)


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "strict_r3_long_supportive_label_stage2_frozen_gmm_v1"
GEOMETRY_START = pd.Timestamp("2024-10-01T00:00:00Z")
GEOMETRY_END = pd.Timestamp("2025-01-01T00:00:00Z")
SUPERVISED_START = pd.Timestamp("2025-01-01T00:00:00Z")
K = 8
INNER_BLOCKS = 4
MIN_INNER_TRAIN_ROWS = 20_000

BASE_AUX_FIELDS = (
    "prequential_p_adverse",
    "prequential_p_weak",
    "prequential_p_clear",
    "prequential_base_score",
    "prequential_base_rank42",
)


@dataclass(frozen=True)
class FrozenGeometry:
    fields: tuple[str, ...]
    lower: np.ndarray
    upper: np.ndarray
    medians: pd.Series
    scaler: Any
    pca: Any
    gmm: Any
    original_component_order: np.ndarray
    geometry_component_policy_priors: np.ndarray


def _finite(values: pd.Series | np.ndarray) -> np.ndarray:
    return pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).to_numpy(float) if isinstance(values, pd.Series) else np.asarray(values, dtype=float)


def _safe_entropy(probability: np.ndarray) -> np.ndarray:
    clipped = np.clip(probability, 1e-9, 1.0)
    return (-clipped * np.log(clipped)).sum(axis=1).astype(np.float32)


def _p1_probability(frame: pd.DataFrame, geometry: FrozenGeometry) -> np.ndarray:
    latent = _transform_p1(
        frame,
        geometry.fields,
        geometry.scaler,
        geometry.pca,
        geometry.medians,
        geometry.lower,
        geometry.upper,
    )
    raw = geometry.gmm.predict_proba(latent).astype(np.float32)
    return raw[:, geometry.original_component_order]


def _frozen_labels(frame: pd.DataFrame, geometry: FrozenGeometry) -> np.ndarray:
    return _p1_probability(frame, geometry).argmax(axis=1).astype(np.int16)


def _build_geometry(
    *,
    ledger: Path,
    labels_root: Path,
    fields: Sequence[str],
    geometry_start: pd.Timestamp = GEOMETRY_START,
    geometry_end: pd.Timestamp = GEOMETRY_END,
) -> tuple[FrozenGeometry, pd.DataFrame, dict[str, Any]]:
    raw = _joined_population(
        ledger,
        labels_root,
        start=geometry_start,
        end=geometry_end,
        fields=fields,
        p1_fields=P1_FUTURE_FIELDS,
    )
    valid = _train_path_eligible(raw)
    availability = pd.to_datetime(valid["supportive_label_available_ts"], utc=True)
    valid = valid.loc[availability.lt(geometry_end)].copy()
    if valid.empty:
        raise RuntimeError("no geometry-definition rows available before 2025-01-01")
    sample = _sample_month_balanced(valid, MAX_CLUSTER_ROWS, seed=SEED)
    scaler, pca, medians, sample_pca, lower, upper = _fit_p1_transform(sample, P1_FUTURE_FIELDS)
    gmm, original_labels, _ = _fit_gmm(sample_pca, k=K)
    original_priors = _class_policy_priors(sample, original_labels, k=K)
    order = np.argsort(original_priors, kind="stable").astype(np.int16)
    geometry = FrozenGeometry(
        fields=tuple(P1_FUTURE_FIELDS),
        lower=lower,
        upper=upper,
        medians=medians,
        scaler=scaler,
        pca=pca,
        gmm=gmm,
        original_component_order=order,
        geometry_component_policy_priors=original_priors[order].astype(np.float32),
    )
    probability = _p1_probability(valid, geometry)
    labels = probability.argmax(axis=1)
    audit = []
    for component in range(K):
        local = valid.loc[labels == component]
        audit.append({
            "component": component,
            "geometry_definition_rows": int(len(local)),
            "geometry_definition_fraction": float(len(local) / len(valid)),
            "geometry_definition_policy_net_bps": float(pd.to_numeric(local["policy_net_bps"], errors="coerce").mean()),
            "geometry_definition_peak_mfe_atr": float(pd.to_numeric(local["supportive_peak_mfe_atr_h12"], errors="coerce").mean()),
            "geometry_definition_efficiency": float(pd.to_numeric(local["supportive_path_efficiency_h12"], errors="coerce").mean()),
        })
    by_month = valid.assign(__month__=pd.to_datetime(valid["__decision_ts__"], utc=True).dt.to_period("M").astype(str)).groupby("__month__", observed=True).size().to_dict()
    manifest = {
        "geometry_definition_start": str(geometry_start),
        "geometry_definition_end_exclusive": str(geometry_end),
        "geometry_rows": int(len(valid)),
        "geometry_sample_rows": int(len(sample)),
        "geometry_month_rows": {str(key): int(value) for key, value in by_month.items()},
        "components": K,
        "p1_field_count": len(P1_FUTURE_FIELDS),
        "component_order": [int(value) for value in order],
        "component_ordering": "ascending train-only policy-net prior on October--December 2024 geometry sample",
        "geometry_definition_excluded_from_supervised_training": True,
    }
    return geometry, pd.DataFrame(audit), manifest


def _fit_membership(
    train: pd.DataFrame,
    held: pd.DataFrame,
    predictor_fields: Sequence[str],
    geometry: FrozenGeometry,
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.Series]:
    labels = _frozen_labels(train, geometry)
    priors = _class_policy_priors(train, labels, k=K)
    x_train, medians = _matrix(train, predictor_fields)
    x_held, _ = _matrix(held, predictor_fields, medians=medians)
    classifier = _model_classifier(classes=K, seed=seed)
    classifier.fit(x_train, labels)
    probability = _align_probabilities(classifier, classifier.predict_proba(x_held), k=K)
    return probability, priors, labels, medians


def _strict_inner_oof(
    train: pd.DataFrame,
    predictor_fields: Sequence[str,
    ],
    geometry: FrozenGeometry,
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Create prequential frozen-GMM outputs for residual fitting.

    Every inner score is generated by a classifier/prior fit only on rows whose
    H12 labels had resolved before that validation block.  The returned scores
    are deliberately absent during the burn-in block.
    """
    # Return arrays in the caller's original row order.  The chronological
    # sorting below is only for constructing prior-only inner folds; residual
    # targets and inputs must remain aligned one-for-one afterwards.
    ordered = train.assign(__original_position__=np.arange(len(train), dtype=np.int64)).sort_values(
        ["__decision_ts__", "candidate_id"], kind="stable"
    ).reset_index(drop=True)
    score = np.full(len(ordered), np.nan, dtype=np.float32)
    probability = np.full((len(ordered), K), np.nan, dtype=np.float32)
    points = np.linspace(0, len(ordered), INNER_BLOCKS + 1, dtype=int)
    decision = pd.to_datetime(ordered["__decision_ts__"], utc=True)
    available = pd.to_datetime(ordered["supportive_label_available_ts"], utc=True)
    for block in range(1, INNER_BLOCKS):
        left, right = int(points[block]), int(points[block + 1])
        valid = ordered.iloc[left:right]
        if valid.empty:
            continue
        cutoff = pd.to_datetime(valid["__decision_ts__"].iloc[0], utc=True)
        train_mask = available.lt(cutoff - EMBARGO) & decision.lt(cutoff)
        inner = ordered.loc[train_mask].copy()
        if len(inner) < MIN_INNER_TRAIN_ROWS:
            continue
        p, priors, _, _ = _fit_membership(inner, valid, predictor_fields, geometry, seed=seed + block)
        score[left:right] = p @ priors
        probability[left:right] = p
    aligned_score = np.full(len(train), np.nan, dtype=np.float32)
    aligned_probability = np.full((len(train), K), np.nan, dtype=np.float32)
    original_position = ordered["__original_position__"].to_numpy(np.int64)
    aligned_score[original_position] = score
    aligned_probability[original_position] = probability
    return aligned_score, aligned_probability


def _residual_matrix(frame: pd.DataFrame, predictor_fields: Sequence[str], probability: np.ndarray, p1_score: np.ndarray, *, medians: pd.Series | None = None) -> tuple[np.ndarray, pd.Series]:
    base, base_medians = _matrix(frame, predictor_fields, medians=medians)
    summary = np.column_stack((
        probability,
        _safe_entropy(probability),
        probability.max(axis=1),
        p1_score,
    )).astype(np.float32)
    return np.column_stack((base, summary)).astype(np.float32), base_medians


def _run_fold(
    *,
    fold: Any,
    fold_index: int,
    ledger: Path,
    labels_root: Path,
    source_fields: Sequence[str],
    geometry: FrozenGeometry,
    supervised_start: pd.Timestamp,
    out_parts: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    predictor_fields = tuple((*source_fields, *BASE_AUX_FIELDS))
    train_raw = _joined_population(ledger, labels_root, start=supervised_start, end=fold.start, fields=source_fields, p1_fields=P1_FUTURE_FIELDS)
    held_raw = _joined_population(ledger, labels_root, start=fold.start, end=fold.end, fields=source_fields, p1_fields=P1_FUTURE_FIELDS)
    train = _train_path_eligible(train_raw, cutoff=fold.start)
    held = _score_eligible(held_raw)
    del train_raw, held_raw
    gc.collect()
    if len(train) < MIN_INNER_TRAIN_ROWS or len(held) < 5_000:
        return [], {"fold": fold.name, "status": "insufficient_support", "train_rows": int(len(train)), "held_rows": int(len(held))}
    train = _sample_month_balanced(train, MAX_TRAIN_ROWS, seed=SEED + fold_index)
    # Full-fold causal membership score.  Geometry itself is frozen and lies
    # strictly before the supervised population.
    held_probability, priors, _, _ = _fit_membership(train, held, predictor_fields, geometry, seed=SEED + 10_000 + fold_index)
    held_p1_score = (held_probability @ priors).astype(np.float32)
    metrics: list[dict[str, Any]] = []
    part = 0

    def write_arm(arm: str, score: np.ndarray, probability: np.ndarray | None = None, extra: dict[str, np.ndarray] | None = None) -> None:
        nonlocal part
        metrics.extend(_quality_metrics(fold=fold, arm=arm, feature_mode="causal120_plus_base", score=score, held=held))
        frame = pd.DataFrame({
            "candidate_id": held["candidate_id"].to_numpy(),
            "__decision_ts__": held["__decision_ts__"].to_numpy(),
            "fold": fold.name,
            "cohort": fold.cohort,
            "arm": arm,
            "feature_mode": "causal120_plus_base",
            "predicted_policy_net_bps": score.astype(np.float32),
            "realised_policy_net_bps": pd.to_numeric(held["policy_net_bps"], errors="coerce").to_numpy(np.float32),
        })
        if probability is not None:
            frame["path_entropy"] = _safe_entropy(probability)
            frame["path_max_probability"] = probability.max(axis=1)
            for idx in range(K):
                frame[f"frozen_path_p_{idx:02d}"] = probability[:, idx]
        if extra:
            for key, value in extra.items():
                frame[key] = value
        frame.to_parquet(out_parts / f"part={part:03d}.parquet", index=False, compression="zstd")
        part += 1

    write_arm("S2_frozen_gmm_k8", held_p1_score, held_probability)
    base = pd.to_numeric(held["prequential_upstream"], errors="coerce").to_numpy(np.float32)
    write_arm("S3_equal_bps_base_path_blend", 0.5 * base + 0.5 * held_p1_score, held_probability)

    # One shared residual head.  Its training inputs use inner-OOF GMM
    # recogniser probabilities, not in-sample memberships or outcome labels.
    oof_score, oof_probability = _strict_inner_oof(train, predictor_fields, geometry, seed=SEED + 20_000 + fold_index * 101)
    target = pd.to_numeric(train["policy_net_bps"], errors="coerce").to_numpy(float) - oof_score.astype(float)
    usable = np.isfinite(oof_score) & np.isfinite(target) & np.isfinite(oof_probability).all(axis=1)
    if int(usable.sum()) >= MIN_INNER_TRAIN_ROWS:
        residual_x, base_medians = _residual_matrix(train.loc[usable], predictor_fields, oof_probability[usable], oof_score[usable])
        held_x, _ = _residual_matrix(held, predictor_fields, held_probability, held_p1_score, medians=base_medians)
        residual = _model_regressor(seed=SEED + 30_000 + fold_index)
        residual.fit(residual_x, target[usable])
        shared_score = held_p1_score + residual.predict(held_x).astype(np.float32)
        write_arm(
            "S3_frozen_gmm_k8_shared_policy_residual",
            shared_score,
            held_probability,
            {"residual_oof_rows": np.full(len(held), int(usable.sum()), dtype=np.int32)},
        )
        residual_status = "ok"
    else:
        residual_status = "insufficient_inner_oof"
    audit = {
        "fold": fold.name,
        "status": "ok",
        "train_rows": int(len(train)),
        "held_rows": int(len(held)),
        "supervised_start": str(supervised_start),
        "train_label_cutoff": str(fold.start),
        "embargo_hours": int(EMBARGO / pd.Timedelta(hours=1)),
        "residual_oof_rows": int(usable.sum()),
        "residual_status": residual_status,
        "class_priors_bps": [float(value) for value in priors],
    }
    return metrics, audit


def _aggregate(metrics: pd.DataFrame) -> pd.DataFrame:
    kept = metrics.loc[metrics["metric"].isin(("top_1%_net_ev_bps", "top_5%_net_ev_bps", "global_score_policy_residual_spearman"))].copy()
    return kept.groupby(["arm", "feature_mode", "cohort", "metric"], as_index=False).agg(
        mean_value=("value", "mean"), median_value=("value", "median"), worst_value=("value", "min"), folds=("fold", "nunique")
    )


def _rolling_geometry_window(fold: Any) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return the predeclared three-month geometry and later training windows.

    The geometry is always six-to-three months before the held fold.  The
    immediately subsequent three months are the *only* supervised population
    for that bundle.  Therefore no shared residual consumes memberships from
    a different geometry state.
    """
    geometry_end = fold.start - pd.DateOffset(months=3)
    geometry_start = fold.start - pd.DateOffset(months=6)
    return pd.Timestamp(geometry_start), pd.Timestamp(geometry_end)


def run(*, ledger: Path, labels_root: Path, out: Path, geometry_mode: str = "frozen") -> Path:
    if out.exists():
        raise FileExistsError(f"immutable output already exists: {out}")
    out.mkdir(parents=True, exist_ok=False)
    if geometry_mode not in {"frozen", "rolling_three_month"}:
        raise ValueError(f"unsupported geometry mode: {geometry_mode}")
    source_fields = _ledger_fields(ledger)
    frozen_geometry: FrozenGeometry | None = None
    geometry_manifest: dict[str, Any] | None = None
    if geometry_mode == "frozen":
        frozen_geometry, geometry_audit, geometry_manifest = _build_geometry(ledger=ledger, labels_root=labels_root, fields=source_fields)
        joblib.dump(frozen_geometry, out / "frozen_p1_gmm_k8_geometry.joblib")
        geometry_audit.to_parquet(out / "frozen_geometry_component_audit.parquet", index=False, compression="zstd")
        (out / "frozen_geometry_contract.json").write_text(json.dumps(geometry_manifest, indent=2) + "\n")
    else:
        (out / "geometry_bundles").mkdir(parents=True, exist_ok=False)
    parts_root = out / "oof_prediction_parts"
    parts_root.mkdir(parents=True, exist_ok=False)
    all_metrics: list[dict[str, Any]] = []
    fold_audit: list[dict[str, Any]] = []
    geometry_by_fold: list[dict[str, Any]] = []
    for index, fold in enumerate(FOLDS):
        if geometry_mode == "frozen":
            assert frozen_geometry is not None and geometry_manifest is not None
            geometry = frozen_geometry
            supervised_start = SUPERVISED_START
            current_geometry_manifest = geometry_manifest
        else:
            geometry_start, geometry_end = _rolling_geometry_window(fold)
            geometry, current_geometry_audit, current_geometry_manifest = _build_geometry(
                ledger=ledger,
                labels_root=labels_root,
                fields=source_fields,
                geometry_start=geometry_start,
                geometry_end=geometry_end,
            )
            bundle_root = out / "geometry_bundles" / f"fold={index:02d}_{fold.name}"
            bundle_root.mkdir(parents=True, exist_ok=False)
            joblib.dump(geometry, bundle_root / "p1_gmm_k8_geometry.joblib")
            current_geometry_audit.to_parquet(bundle_root / "component_audit.parquet", index=False, compression="zstd")
            (bundle_root / "contract.json").write_text(json.dumps(current_geometry_manifest, indent=2) + "\n")
            supervised_start = geometry_end
        geometry_by_fold.append({"fold": fold.name, "supervised_start": str(supervised_start), **current_geometry_manifest})
        fold_parts = parts_root / f"fold={index:02d}_{fold.name}"
        fold_parts.mkdir(parents=True, exist_ok=False)
        metrics, audit = _run_fold(
            fold=fold,
            fold_index=index,
            ledger=ledger,
            labels_root=labels_root,
            source_fields=source_fields,
            geometry=geometry,
            supervised_start=supervised_start,
            out_parts=fold_parts,
        )
        all_metrics.extend(metrics)
        fold_audit.append(audit)
        print(json.dumps(audit, sort_keys=True), flush=True)
    metrics_frame = pd.DataFrame(all_metrics)
    metrics_frame.to_parquet(out / "stage2_metrics.parquet", index=False, compression="zstd")
    _aggregate(metrics_frame).to_parquet(out / "stage2_summary.parquet", index=False, compression="zstd")
    pd.DataFrame(fold_audit).to_parquet(out / "stage2_fold_audit.parquet", index=False, compression="zstd")
    parts = sorted(parts_root.rglob("*.parquet"))
    (out / "stage2_oof_predictions_manifest.json").write_text(json.dumps({
        "format": "partitioned_parquet",
        "root": str(parts_root),
        "parts": [str(item.relative_to(parts_root)) for item in parts],
    }, indent=2) + "\n")
    manifest = {
        "schema": SCHEMA,
        "scope": "offline long-only supportive-label research; no inference/live mutation",
        "ledger": str(ledger.resolve()),
        "ledger_sha256": _sha256(ledger),
        "labels_root": str(labels_root.resolve()),
        "labels_manifest_sha256": _sha256(labels_root / "run_manifest.json"),
        "geometry_mode": geometry_mode,
        "geometry": geometry_manifest if geometry_mode == "frozen" else None,
        "geometry_by_fold": geometry_by_fold,
        "outer_folds": [{"name": fold.name, "start": str(fold.start), "end_exclusive": str(fold.end), "cohort": fold.cohort} for fold in FOLDS],
        "predictor_contract": list((*source_fields, *BASE_AUX_FIELDS)),
        "inference_contract": "frozen-GMM causal membership probabilities, entropy, confidence and train-derived class-policy priors only; realised P1 labels are prohibited",
        "residual_contract": "one shared Huber residual trained only on strict-inner-OOF causal membership outputs; no local experts or routing",
        "geometry_definition_rows_excluded_from_supervised_training": True,
        "h12_embargo_hours": int(EMBARGO / pd.Timedelta(hours=1)),
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--labels-root", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--geometry-mode", choices=("frozen", "rolling_three_month"), default="frozen")
    args = parser.parse_args()
    result = run(ledger=args.ledger.resolve(), labels_root=args.labels_root.resolve(), out=args.out.resolve(), geometry_mode=args.geometry_mode)
    print(json.dumps({"status": "ok", "out": str(result)}, sort_keys=True))


if __name__ == "__main__":
    main()
