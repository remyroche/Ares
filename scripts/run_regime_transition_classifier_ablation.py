#!/usr/bin/env python3
"""Grouped pooled ablations for onset, phase and destination classification.

Validation is intentionally not walk-forward.  It is nevertheless
leakage-controlled: every transition window is kept in one fold and stable
controls are grouped into seven-day blocks.  Random row splitting is forbidden.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Callable

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_DATASET = Path(
    "data_perp/artifacts/regime_transition_research_20260726_v2/"
    "hourly_transition_dataset.parquet"
)
DEFAULT_EVENTS = Path(
    "data_perp/artifacts/regime_transition_research_20260726_v2/"
    "transition_events.parquet"
)
DEFAULT_OUTPUT = Path(
    "data_perp/artifacts/regime_transition_classifier_ablation_20260726_v2"
)


def _safe(value: Any) -> Any:
    if isinstance(value, (Path, pd.Timestamp, pd.Timedelta)):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(_safe(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _groups(frame: pd.DataFrame) -> np.ndarray:
    timestamp = pd.to_datetime(frame["source_utc"], utc=True)
    control = (
        "control_"
        + frame["segment_id"].astype(str)
        + "_"
        + timestamp.dt.to_period("W").astype(str)
    )
    return frame["target__event_id"].fillna(control).astype(str).to_numpy()


def _catboost(seed: int, classes: int = 2) -> CatBoostClassifier:
    loss = "Logloss" if classes == 2 else "MultiClass"
    return CatBoostClassifier(
        iterations=280,
        depth=6,
        learning_rate=0.055,
        l2_leaf_reg=6.0,
        random_seed=seed,
        loss_function=loss,
        auto_class_weights="Balanced",
        verbose=False,
        allow_writing_files=False,
        thread_count=4,
    )


def _lightgbm(seed: int, classes: int = 2) -> LGBMClassifier:
    return LGBMClassifier(
        objective="binary" if classes == 2 else "multiclass",
        n_estimators=300,
        learning_rate=0.045,
        num_leaves=31,
        min_child_samples=40,
        subsample=0.85,
        colsample_bytree=0.75,
        reg_lambda=5.0,
        class_weight="balanced",
        random_state=seed,
        n_jobs=4,
        verbosity=-1,
    )


def _logistic(seed: int) -> object:
    return make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True),
        RobustScaler(quantile_range=(10, 90)),
        LogisticRegression(
            C=0.3,
            class_weight="balanced",
            max_iter=1000,
            random_state=seed,
        ),
    )


def _cross_validated_binary(
    frame: pd.DataFrame,
    *,
    features: list[str],
    target: str,
    model_factory: Callable[[int], object],
    folds: int,
    seed: int,
) -> np.ndarray:
    x = frame[features].apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(frame[target], errors="raise").astype(int).to_numpy()
    groups = _groups(frame)
    splitter = StratifiedGroupKFold(
        n_splits=folds, shuffle=True, random_state=seed
    )
    prediction = np.full(len(frame), np.nan, dtype=np.float32)
    for fold, (train, evaluation) in enumerate(
        splitter.split(x, y, groups=groups)
    ):
        model = model_factory(seed + fold)
        model.fit(x.iloc[train], y[train])
        prediction[evaluation] = np.asarray(
            model.predict_proba(x.iloc[evaluation])
        )[:, 1]
    if not np.isfinite(prediction).all():
        raise ValueError("grouped CV failed to produce complete predictions")
    return prediction


def _binary_metrics(y: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    hard = prediction >= 0.5
    return {
        "average_precision": float(average_precision_score(y, prediction)),
        "roc_auc": float(roc_auc_score(y, prediction)),
        "brier": float(brier_score_loss(y, prediction)),
        "log_loss": float(log_loss(y, np.clip(prediction, 1e-6, 1 - 1e-6))),
        "precision_at_0_5": float(precision_score(y, hard, zero_division=0)),
        "recall_at_0_5": float(recall_score(y, hard, zero_division=0)),
        "f1_at_0_5": float(f1_score(y, hard, zero_division=0)),
        "mcc_at_0_5": float(matthews_corrcoef(y, hard)),
        "prevalence": float(np.mean(y)),
    }


def _alert_episodes(
    timestamp: pd.Series,
    score: np.ndarray,
    threshold: float,
    *,
    refractory_hours: int = 6,
) -> int:
    order = np.argsort(pd.to_datetime(timestamp, utc=True).to_numpy())
    stamps = pd.to_datetime(timestamp.iloc[order], utc=True)
    active = score[order] >= threshold
    alerts = 0
    last: pd.Timestamp | None = None
    for stamp, flag in zip(stamps, active, strict=True):
        if not flag:
            continue
        if last is None or stamp - last >= pd.Timedelta(hours=refractory_hours):
            alerts += 1
            last = stamp
    return alerts


def _event_metrics(
    frame: pd.DataFrame,
    prediction: np.ndarray,
) -> list[dict[str, float]]:
    work = frame[
        [
            "source_utc",
            "target__event_id",
            "target__time_to_onset_hours",
            "target__onset_within_3h",
        ]
    ].copy()
    work["prediction"] = prediction
    negative = work["target__onset_within_3h"].eq(0)
    days = max(
        (
            pd.to_datetime(work["source_utc"], utc=True).max()
            - pd.to_datetime(work["source_utc"], utc=True).min()
        ).total_seconds()
        / 86_400,
        1,
    )
    event_leads = work.loc[
        work["target__onset_within_3h"].eq(1)
        & work["target__event_id"].notna()
    ]
    events = int(event_leads["target__event_id"].nunique())
    rows: list[dict[str, float]] = []
    for threshold in (0.25, 0.40, 0.50, 0.65, 0.80):
        false_alerts = _alert_episodes(
            work.loc[negative, "source_utc"],
            work.loc[negative, "prediction"].to_numpy(),
            threshold,
        )
        detected = (
            event_leads.groupby("target__event_id", observed=True)["prediction"]
            .max()
            .ge(threshold)
        )
        lead = event_leads.loc[event_leads["prediction"].ge(threshold)]
        first_lead = (
            -lead.groupby("target__event_id", observed=True)[
                "target__time_to_onset_hours"
            ].min()
            if len(lead)
            else pd.Series(dtype=float)
        )
        rows.append(
            {
                "threshold": threshold,
                "event_count": events,
                "event_recall": float(detected.mean()) if len(detected) else np.nan,
                "false_alerts_per_30d": float(false_alerts / days * 30),
                "median_lead_hours": float(first_lead.median())
                if len(first_lead)
                else np.nan,
            }
        )
    return rows


def _multiclass_oof(
    frame: pd.DataFrame,
    *,
    features: list[str],
    target: str,
    folds: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    target_values = frame[target].astype(str)
    classes = sorted(target_values.unique())
    mapping = {label: index for index, label in enumerate(classes)}
    y = target_values.map(mapping).to_numpy(np.int16)
    x = frame[features].apply(pd.to_numeric, errors="coerce")
    groups = _groups(frame)
    splitter = StratifiedGroupKFold(
        n_splits=folds, shuffle=True, random_state=seed
    )
    probability = np.full((len(frame), len(classes)), np.nan, dtype=np.float32)
    for fold, (train, evaluation) in enumerate(
        splitter.split(x, y, groups=groups)
    ):
        model = _catboost(seed + fold, classes=len(classes))
        model.fit(x.iloc[train], y[train])
        fold_probability = np.asarray(model.predict_proba(x.iloc[evaluation]))
        trained_classes = np.asarray(model.classes_, dtype=int)
        probability[np.ix_(evaluation, trained_classes)] = fold_probability
        missing = sorted(set(range(len(classes))).difference(trained_classes))
        if missing:
            probability[np.ix_(evaluation, missing)] = 0.0
    probability = np.nan_to_num(probability, nan=0.0)
    probability /= np.maximum(probability.sum(axis=1, keepdims=True), 1e-12)
    return y, probability, classes


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--events", type=Path, default=DEFAULT_EVENTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1729)
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True)
    frame = pd.read_parquet(args.dataset)
    frame["source_utc"] = pd.to_datetime(frame["source_utc"], utc=True)
    metadata = {
        "source_utc",
        "execution_decision_utc",
        "segment_id",
        "target__pooled_state",
    }
    numeric = [
        name
        for name in frame.columns
        if name not in metadata
        and not name.startswith("target__")
        and pd.api.types.is_numeric_dtype(frame[name])
    ]
    existing = [
        name for name in numeric if name.startswith("mkt_regime_change__")
    ]
    new = [name for name in numeric if name.startswith("transition_new__")]
    state_context = [
        name for name in numeric if name.startswith("state_context__")
    ]
    levels = [
        name
        for name in numeric
        if name not in set(existing + new + state_context)
    ]
    feature_sets = {
        "levels_only": levels,
        "existing_transition_only": existing,
        "state_context_only": state_context,
        "levels_plus_state_context": levels + state_context,
        "existing_plus_state_context": existing + state_context,
        "existing_plus_new_transition": existing + new,
        "levels_plus_existing_transition": levels + existing,
        "all_market_without_state_context": levels + existing + new,
        "all_market_with_state_context": levels
        + existing
        + new
        + state_context,
    }
    y = frame["target__onset_within_3h"].astype(int).to_numpy()
    metric_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    oof_parts: list[pd.DataFrame] = []
    specifications: list[
        tuple[str, str, Callable[[int], object]]
    ] = [
        ("logistic", "all_market_without_state_context", _logistic),
        (
            "lightgbm",
            "all_market_without_state_context",
            lambda seed: _lightgbm(seed),
        ),
    ]
    specifications.extend(
        ("catboost", feature_set, lambda seed: _catboost(seed))
        for feature_set in feature_sets
    )
    for model_name, feature_set, factory in specifications:
        features = feature_sets[feature_set]
        prediction = _cross_validated_binary(
            frame,
            features=features,
            target="target__onset_within_3h",
            model_factory=factory,
            folds=int(args.folds),
            seed=int(args.seed),
        )
        metrics = _binary_metrics(y, prediction)
        metric_rows.append(
            {
                "model": model_name,
                "feature_set": feature_set,
                "feature_count": len(features),
                "rows": len(frame),
                **metrics,
            }
        )
        for row in _event_metrics(frame, prediction):
            event_rows.append(
                {"model": model_name, "feature_set": feature_set, **row}
            )
        oof_parts.append(
            frame[
                [
                    "source_utc",
                    "segment_id",
                    "target__event_id",
                    "target__onset_within_3h",
                    "target__time_to_onset_hours",
                ]
            ].assign(
                model=model_name,
                feature_set=feature_set,
                prediction=prediction,
            )
        )
    metrics = pd.DataFrame(metric_rows).sort_values(
        ["average_precision", "roc_auc"], ascending=False
    )
    winner = metrics.iloc[0]
    winning_features = feature_sets[str(winner["feature_set"])]
    event_rows_frame = pd.DataFrame(event_rows)

    # Destination is defined only for lead/active rows and uses the stabilized
    # [+6,+12) class, never the first changed hour.
    destination = frame.loc[
        frame["target__destination_state"].notna()
        & frame["target__event_id"].notna()
    ].copy()
    destination["destination_label"] = (
        "state_"
        + destination["target__destination_state"].astype(int).astype(str)
    )
    dy, dp, destination_classes = _multiclass_oof(
        destination,
        features=winning_features,
        target="destination_label",
        folds=int(args.folds),
        seed=int(args.seed) + 100,
    )
    dhard = np.argmax(dp, axis=1)
    destination_metrics = {
        "rows": len(destination),
        "events": int(destination["target__event_id"].nunique()),
        "classes": destination_classes,
        "balanced_accuracy": float(balanced_accuracy_score(dy, dhard)),
        "macro_f1": float(f1_score(dy, dhard, average="macro")),
        "log_loss": float(log_loss(dy, np.clip(dp, 1e-7, 1.0))),
    }
    destination_oof = destination[
        ["source_utc", "target__event_id", "destination_label"]
    ].copy()
    destination_oof["predicted_destination"] = [
        destination_classes[index] for index in dhard
    ]
    for index, label in enumerate(destination_classes):
        destination_oof[f"p_destination__{label}"] = dp[:, index]

    phase = frame.copy()
    py, pp, phase_classes = _multiclass_oof(
        phase,
        features=winning_features,
        target="target__phase",
        folds=int(args.folds),
        seed=int(args.seed) + 200,
    )
    phard = np.argmax(pp, axis=1)
    phase_metrics = {
        "rows": len(phase),
        "classes": phase_classes,
        "balanced_accuracy": float(balanced_accuracy_score(py, phard)),
        "macro_f1": float(f1_score(py, phard, average="macro")),
        "log_loss": float(log_loss(py, np.clip(pp, 1e-7, 1.0))),
    }
    metrics.to_csv(output / "onset_ablation_metrics.csv", index=False)
    event_rows_frame.to_csv(output / "onset_event_metrics.csv", index=False)
    pd.concat(oof_parts, ignore_index=True).to_parquet(
        output / "onset_grouped_oof.parquet", index=False
    )
    destination_oof.to_parquet(
        output / "destination_grouped_oof.parquet", index=False
    )
    pd.DataFrame(
        {
            "source_utc": phase["source_utc"],
            "target_phase": phase["target__phase"],
            "predicted_phase": [phase_classes[index] for index in phard],
        }
    ).to_parquet(output / "phase_grouped_oof.parquet", index=False)
    report = {
        "schema": "pooled_grouped_transition_classifier_ablation_v1",
        "research_only": True,
        "walk_forward_validation": False,
        "validation": (
            "5-fold stratified grouped CV; transition window=event_id; "
            "stable controls=calendar-week block"
        ),
        "rows": len(frame),
        "events": int(frame["target__event_id"].dropna().nunique()),
        "positive_onset_rows": int(y.sum()),
        "feature_sets": {key: len(value) for key, value in feature_sets.items()},
        "winner": winner.to_dict(),
        "destination": destination_metrics,
        "phase": phase_metrics,
    }
    _write_json(output / "report.json", report)
    joblib.dump(winning_features, output / "winning_feature_list.joblib")
    return report


def main() -> None:
    report = run(_parser().parse_args())
    print(json.dumps(_safe(report), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
