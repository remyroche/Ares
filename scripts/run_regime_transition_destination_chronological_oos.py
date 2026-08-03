#!/usr/bin/env python3
"""Chronological label-purged destination probabilities with abstention.

Destination is conditional on a canonical transition event.  Expanding-month
folds use only labels available before the evaluation month and additionally
purge any event ID appearing in that month's evaluation rows.  Upstream state
geometry remains pooled research evidence and is disclosed explicitly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score, log_loss

from run_regime_transition_active_head_chronological_oos import (
    conservative_label_available_utc,
)


CLASSES = tuple(f"state_{index}" for index in range(5))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, (Path, pd.Timestamp, pd.Timedelta)):
        return str(value)
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


def destination_frame(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.loc[
        frame["target__destination_state"].notna()
        & frame["target__event_id"].notna()
    ].copy()
    work["source_utc"] = pd.to_datetime(
        work["source_utc"], utc=True, errors="raise"
    )
    work["destination_label"] = (
        "state_"
        + work["target__destination_state"].astype(int).astype(str)
    )
    if not set(work["destination_label"]).issubset(CLASSES):
        raise ValueError("destination target contains an unsupported state")
    return work


def destination_month_folds(
    frame: pd.DataFrame,
    *,
    first_evaluation_month: str,
    last_evaluation_month: str,
    minimum_train_months: int,
) -> list[tuple[pd.Timestamp, np.ndarray, np.ndarray]]:
    source = pd.to_datetime(frame["source_utc"], utc=True, errors="raise")
    available = conservative_label_available_utc(frame)
    first = pd.Timestamp(first_evaluation_month, tz="UTC")
    last = pd.Timestamp(last_evaluation_month, tz="UTC")
    folds: list[tuple[pd.Timestamp, np.ndarray, np.ndarray]] = []
    for start in pd.date_range(first, last, freq="MS", tz="UTC"):
        end = start + pd.offsets.MonthBegin(1)
        evaluation_mask = source.ge(start) & source.lt(end)
        evaluation = np.flatnonzero(evaluation_mask.to_numpy())
        if not len(evaluation):
            continue
        evaluation_events = set(
            frame.iloc[evaluation]["target__event_id"].dropna().astype(str)
        )
        train_mask = available.lt(start) & ~frame["target__event_id"].astype(
            str
        ).isin(evaluation_events)
        train = np.flatnonzero(train_mask.to_numpy())
        train_months = source.iloc[train].dt.tz_localize(None).dt.to_period("M")
        if train_months.nunique() < int(minimum_train_months):
            continue
        train_events = set(
            frame.iloc[train]["target__event_id"].dropna().astype(str)
        )
        if train_events.intersection(evaluation_events):
            raise AssertionError("destination fold leaks an evaluation event")
        if len(train) and available.iloc[train].max() >= start:
            raise AssertionError("destination fold contains unavailable labels")
        folds.append((start, train, evaluation))
    return folds


def _model(seed: int) -> CatBoostClassifier:
    return CatBoostClassifier(
        iterations=280,
        depth=6,
        learning_rate=0.055,
        l2_leaf_reg=6.0,
        random_seed=seed,
        loss_function="MultiClass",
        auto_class_weights="Balanced",
        verbose=False,
        allow_writing_files=False,
        thread_count=4,
    )


def _probability_in_global_order(
    model: CatBoostClassifier, x: pd.DataFrame
) -> np.ndarray:
    local = np.asarray(model.predict_proba(x), dtype=float)
    output = np.zeros((len(x), len(CLASSES)), dtype=float)
    for local_index, trained_class in enumerate(model.classes_):
        output[:, int(trained_class)] = local[:, local_index]
    output /= np.maximum(output.sum(axis=1, keepdims=True), 1e-12)
    return output


def _metrics(y: np.ndarray, probability: np.ndarray) -> dict[str, Any]:
    hard = np.argmax(probability, axis=1)
    return {
        "rows": int(len(y)),
        "events": None,
        "balanced_accuracy": float(balanced_accuracy_score(y, hard)),
        "macro_f1": float(f1_score(y, hard, average="macro")),
        "log_loss": float(
            log_loss(y, np.clip(probability, 1e-7, 1.0), labels=range(5))
        ),
        "accuracy": float(np.mean(hard == y)),
    }


def abstention_curve(
    prediction: pd.DataFrame, thresholds: Sequence[float]
) -> pd.DataFrame:
    probability_columns = [f"p_destination__{label}" for label in CLASSES]
    probabilities = prediction[probability_columns].to_numpy(float)
    confidence = probabilities.max(axis=1)
    y = prediction["destination_label"].map(
        {label: index for index, label in enumerate(CLASSES)}
    ).to_numpy(int)
    hard = probabilities.argmax(axis=1)
    rows: list[dict[str, Any]] = []
    for threshold in thresholds:
        accepted = confidence >= float(threshold)
        rows.append(
            {
                "minimum_confidence": float(threshold),
                "coverage": float(accepted.mean()),
                "accepted_rows": int(accepted.sum()),
                "accepted_events": int(
                    prediction.loc[accepted, "target__event_id"].nunique()
                ),
                "accuracy": float(np.mean(hard[accepted] == y[accepted]))
                if accepted.any()
                else np.nan,
                "macro_f1": float(
                    f1_score(y[accepted], hard[accepted], average="macro")
                )
                if accepted.any()
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


def fit_predict(
    frame: pd.DataFrame,
    *,
    features: Sequence[str],
    first_evaluation_month: str,
    last_evaluation_month: str,
    minimum_train_months: int,
    confidence_thresholds: Sequence[float],
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    work = destination_frame(frame)
    missing = sorted(set(features).difference(work.columns))
    if missing:
        raise ValueError(f"destination frame lacks features: {missing[:5]}")
    x = work[list(features)].apply(pd.to_numeric, errors="coerce")
    class_map = {label: index for index, label in enumerate(CLASSES)}
    y = work["destination_label"].map(class_map).to_numpy(np.int16)
    folds = destination_month_folds(
        work,
        first_evaluation_month=first_evaluation_month,
        last_evaluation_month=last_evaluation_month,
        minimum_train_months=minimum_train_months,
    )
    if not folds:
        raise ValueError("no eligible chronological destination folds")
    predictions: list[pd.DataFrame] = []
    fold_rows: list[dict[str, Any]] = []
    for fold_index, (start, train, evaluation) in enumerate(folds):
        if len(np.unique(y[train])) < 2:
            raise ValueError(f"{start:%Y-%m} destination train has one class")
        model = _model(seed + fold_index)
        model.fit(x.iloc[train], y[train])
        probability = _probability_in_global_order(model, x.iloc[evaluation])
        local = work.iloc[evaluation][
            ["source_utc", "target__event_id", "destination_label"]
        ].copy()
        hard = probability.argmax(axis=1)
        local["predicted_destination"] = [CLASSES[index] for index in hard]
        for index, label in enumerate(CLASSES):
            local[f"p_destination__{label}"] = probability[:, index].astype(
                np.float32
            )
        local["destination_confidence"] = probability.max(axis=1).astype(
            np.float32
        )
        local["destination_entropy"] = (
            -np.sum(
                np.where(
                    probability > 0,
                    probability * np.log(np.clip(probability, 1e-12, 1.0)),
                    0.0,
                ),
                axis=1,
            )
        ).astype(np.float32)
        local["evaluation_month"] = start.strftime("%Y-%m")
        local["train_rows"] = int(len(train))
        predictions.append(local)
        metrics = _metrics(y[evaluation], probability)
        metrics["events"] = int(local["target__event_id"].nunique())
        fold_rows.append(
            {
                "evaluation_month": start.strftime("%Y-%m"),
                "train_rows": int(len(train)),
                "train_events": int(
                    work.iloc[train]["target__event_id"].nunique()
                ),
                "train_end_label_available_utc": conservative_label_available_utc(
                    work.iloc[train]
                ).max(),
                **metrics,
            }
        )
    oos = pd.concat(predictions, ignore_index=True)
    if oos.duplicated(["source_utc", "target__event_id"]).any():
        raise AssertionError("duplicate destination prediction identity")
    return (
        oos,
        pd.DataFrame(fold_rows),
        abstention_curve(oos, confidence_thresholds),
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    dataset_path = Path(args.dataset)
    features_path = Path(args.winning_features)
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    frame = pd.read_parquet(dataset_path)
    features = list(joblib.load(features_path))
    oos, fold_metrics, abstention = fit_predict(
        frame,
        features=features,
        first_evaluation_month=args.first_evaluation_month,
        last_evaluation_month=args.last_evaluation_month,
        minimum_train_months=int(args.minimum_train_months),
        confidence_thresholds=args.confidence_thresholds,
        seed=int(args.seed),
    )
    probability = oos[
        [f"p_destination__{label}" for label in CLASSES]
    ].to_numpy(float)
    y = oos["destination_label"].map(
        {label: index for index, label in enumerate(CLASSES)}
    ).to_numpy(int)
    output.mkdir(parents=True, exist_ok=False)
    prediction_path = output / "destination_chronological_oos.parquet"
    fold_path = output / "fold_metrics.csv"
    abstention_path = output / "abstention_curve.csv"
    feature_output_path = output / "features.json"
    oos.to_parquet(prediction_path, index=False, compression="zstd")
    fold_metrics.to_csv(fold_path, index=False)
    abstention.to_csv(abstention_path, index=False)
    _write_json(feature_output_path, {"features": features})
    metrics = _metrics(y, probability)
    metrics["events"] = int(oos["target__event_id"].nunique())
    manifest = {
        "schema": "destination_chronological_oos_v1",
        "status": "RESEARCH_ONLY_CHRONOLOGICAL_LABEL_OOS_COMPLETE",
        "promotion_eligible": False,
        "promotion_blocker": (
            "destination folds are chronological, label-purged and event-purged, "
            "but upstream five-state geometry is pooled research"
        ),
        "validation_contract": {
            "fold": "expanding monthly",
            "label_purge": (
                "train max(max(source+12h,target__available_utc)) < month start"
            ),
            "event_purge": "evaluation event IDs excluded from training",
            "upstream_geometry": "pooled research; non-production-causal",
            "first_evaluation_month": args.first_evaluation_month,
            "last_evaluation_month": args.last_evaluation_month,
            "minimum_train_months": int(args.minimum_train_months),
        },
        "metrics": metrics,
        "feature_count": len(features),
        "sources": {
            "dataset": {
                "path": str(dataset_path),
                "sha256": _sha256(dataset_path),
            },
            "winning_features": {
                "path": str(features_path),
                "sha256": _sha256(features_path),
            },
        },
        "outputs": {},
        "runner": {
            "path": str(Path(__file__).resolve()),
            "sha256": _sha256(Path(__file__).resolve()),
        },
    }
    outputs = {
        "predictions": prediction_path,
        "fold_metrics": fold_path,
        "abstention_curve": abstention_path,
        "features": feature_output_path,
    }
    manifest["outputs"] = {
        name: {"path": str(path), "sha256": _sha256(path)}
        for name, path in outputs.items()
    }
    manifest_path = output / "manifest.json"
    _write_json(manifest_path, manifest)
    return manifest


def _parser() -> argparse.ArgumentParser:
    root = Path("/Users/remyroche/Documents/Ares")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=root
        / (
            "data_perp/artifacts/regime_transition_research_20260726_v3/"
            "hourly_transition_dataset.parquet"
        ),
    )
    parser.add_argument(
        "--winning-features",
        type=Path,
        default=root
        / (
            "data_perp/artifacts/regime_transition_classifier_ablation_20260726_v2/"
            "winning_feature_list.joblib"
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--first-evaluation-month", default="2024-01-01")
    parser.add_argument("--last-evaluation-month", default="2026-07-01")
    parser.add_argument("--minimum-train-months", type=int, default=12)
    parser.add_argument(
        "--confidence-thresholds",
        type=float,
        nargs="+",
        default=(0.0, 0.50, 0.60, 0.70, 0.80),
    )
    parser.add_argument("--seed", type=int, default=1829)
    return parser


def main() -> None:
    print(json.dumps(_safe(run(_parser().parse_args())), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
