#!/usr/bin/env python3
"""Grouped research ablation of compact current-lineage health features.

This is deliberately a research-only, grouped evaluation.  It refuses to
invent an OOF result when the overlap contains fewer independent onset events
than requested folds.  In that case it still writes a coverage artifact that
states why incremental classification is not yet identifiable.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.regime_transition_current_model_health import CURRENT_MODEL_HEALTH_COLUMNS  # noqa: E402
from scripts.run_regime_transition_classifier_ablation import _groups  # noqa: E402


DEFAULT_DATASET = Path("data_perp/artifacts/regime_transition_research_20260726_v3/hourly_transition_dataset.parquet")
DEFAULT_HEALTH = Path("data_perp/artifacts/regime_transition_current_model_health_20260727_v1/hourly_model_health.parquet")
DEFAULT_OUTPUT = Path("data_perp/artifacts/regime_transition_current_model_health_ablation_20260727_v1")


def _safe(value: Any) -> Any:
    if isinstance(value, (pd.Timestamp, Path)):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_safe(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _model(seed: int) -> LGBMClassifier:
    return LGBMClassifier(
        objective="binary",
        n_estimators=320,
        learning_rate=0.035,
        num_leaves=31,
        min_child_samples=30,
        subsample=0.85,
        colsample_bytree=0.75,
        reg_alpha=0.5,
        reg_lambda=8.0,
        class_weight="balanced",
        random_state=seed,
        n_jobs=4,
        verbosity=-1,
    )


def _alert_count(timestamp: pd.Series, segment: pd.Series, score: np.ndarray, threshold: float, *, refractory_hours: int = 6) -> int:
    work = pd.DataFrame({
        "timestamp": pd.to_datetime(timestamp, utc=True),
        "segment": segment.to_numpy(),
        "score": score,
    })
    total = 0
    for _, local in work.loc[work["score"] >= threshold].sort_values("timestamp").groupby("segment", observed=True, sort=False):
        last: pd.Timestamp | None = None
        for stamp in local["timestamp"]:
            if last is None or stamp - last >= pd.Timedelta(hours=refractory_hours):
                total += 1
                last = stamp
    return total


def _recall_at_fixed_false_alerts(frame: pd.DataFrame, prediction: np.ndarray) -> list[dict[str, float | int | None]]:
    """Select thresholds on this grouped-OOF research panel for reporting only."""

    y = frame["target__onset_within_3h"].astype(int).to_numpy()
    negative = y == 0
    event = (y == 1) & frame["target__event_id"].notna().to_numpy()
    start = pd.to_datetime(frame["source_utc"], utc=True).min()
    end = pd.to_datetime(frame["source_utc"], utc=True).max()
    days = max(float((end - start) / pd.Timedelta(days=1)), 1.0)
    event_max = pd.DataFrame({
        "event": frame.loc[event, "target__event_id"].astype(str).to_numpy(),
        "prediction": prediction[event],
    }).groupby("event", observed=True)["prediction"].max()
    thresholds = np.r_[np.inf, np.sort(np.unique(prediction[negative]))[::-1], -np.inf]
    rows: list[dict[str, float | int | None]] = []
    for budget in (1.0, 2.0, 4.0):
        limit = budget * days / 30.0
        chosen = float("inf")
        actual = 0
        for threshold in thresholds:
            count = _alert_count(
                frame.loc[negative, "source_utc"], frame.loc[negative, "segment_id"], prediction[negative], float(threshold)
            )
            if count <= limit:
                chosen, actual = float(threshold), count
            else:
                break
        detected = event_max.ge(chosen)
        rows.append({
            "false_alert_budget_per_30d": budget,
            "threshold": chosen if np.isfinite(chosen) else None,
            "false_alerts_per_30d": float(actual / days * 30.0),
            "event_count": int(len(event_max)),
            "event_recall": float(detected.mean()) if len(detected) else None,
        })
    return rows


def _cross_validated(frame: pd.DataFrame, features: list[str], *, folds: int, seed: int) -> tuple[np.ndarray, list[dict[str, int]]]:
    x = frame.loc[:, features].apply(pd.to_numeric, errors="coerce")
    y = frame["target__onset_within_3h"].astype(int).to_numpy()
    groups = _groups(frame)
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    prediction = np.full(len(frame), np.nan, dtype=np.float32)
    provenance: list[dict[str, int]] = []
    for fold, (train, valid) in enumerate(splitter.split(x, y, groups)):
        overlap = set(groups[train]).intersection(groups[valid])
        if overlap:
            raise AssertionError("event/control groups overlap between train and validation")
        model = _model(seed + fold)
        model.fit(x.iloc[train], y[train])
        prediction[valid] = model.predict_proba(x.iloc[valid])[:, 1]
        provenance.append({
            "fold": fold,
            "train_rows": int(len(train)),
            "validation_rows": int(len(valid)),
            "train_positive_rows": int(y[train].sum()),
            "validation_positive_rows": int(y[valid].sum()),
            "group_overlap": 0,
        })
    if not np.isfinite(prediction).all():
        raise AssertionError("grouped OOF prediction is incomplete")
    return prediction, provenance


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--health", type=Path, default=DEFAULT_HEALTH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2837)
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True)
    market = pd.read_parquet(args.dataset)
    health = pd.read_parquet(args.health)
    for frame in (market, health):
        frame["source_utc"] = pd.to_datetime(frame["source_utc"], utc=True, errors="raise")
        if frame["source_utc"].duplicated().any():
            raise ValueError("hourly source must have exactly one row per source_utc")
    frame = market.merge(health, on="source_utc", how="inner", validate="one_to_one", suffixes=("", "__health"))
    if "execution_decision_utc__health" in frame:
        if not frame["execution_decision_utc"].eq(frame["execution_decision_utc__health"]).all():
            raise ValueError("market/health execution timestamp mismatch")
        frame = frame.drop(columns="execution_decision_utc__health")
    health_features = [name for name in CURRENT_MODEL_HEALTH_COLUMNS if name in frame]
    if len(health_features) != len(CURRENT_MODEL_HEALTH_COLUMNS):
        raise ValueError("health schema does not match the compact current-lineage contract")
    excluded = {"source_utc", "execution_decision_utc", "segment_id", "target__pooled_state"}
    market_features = [
        name for name in market.columns
        if name not in excluded and not name.startswith("target__") and pd.api.types.is_numeric_dtype(market[name])
    ]
    positive_events = frame.loc[frame["target__onset_within_3h"].astype(bool), "target__event_id"].dropna().nunique()
    coverage = {
        "rows": int(len(frame)),
        "start": frame["source_utc"].min(),
        "end": frame["source_utc"].max(),
        "onset_positive_rows": int(frame["target__onset_within_3h"].sum()),
        "independent_onset_events": int(positive_events),
        "health_feature_count": len(health_features),
        "market_feature_count": len(market_features),
    }
    frame.loc[:, ["source_utc", "target__event_id", "target__onset_within_3h"]].to_parquet(output / "overlap_labels.parquet", index=False)
    if positive_events < int(args.folds):
        report = {
            "schema": "current_lineage_model_health_transition_ablation_v1",
            "status": "INSUFFICIENT_INDEPENDENT_EVENTS_FOR_GROUPED_OOF",
            "research_only": True,
            "coverage": coverage,
            "requested_folds": int(args.folds),
            "metrics": None,
            "reason": (
                "Current-lineage overlap contains fewer independent onset events than grouped folds. "
                "No row-level split, in-sample metric, or cross-lineage backfill is substituted."
            ),
            "next_requirement": "materialize current-lineage OOF scores across at least five independent transition events",
        }
        (output / "report.json").write_text(json.dumps(_safe(report), indent=2, sort_keys=True) + "\n")
        return report
    y = frame["target__onset_within_3h"].astype(int).to_numpy()
    sets = {
        "market_only_same_current_period": market_features,
        "current_model_health_only": health_features,
        "market_plus_current_model_health": market_features + health_features,
    }
    metrics: list[dict[str, Any]] = []
    predictions = frame.loc[:, ["source_utc", "target__event_id", "target__onset_within_3h"]].copy()
    fold_report: dict[str, list[dict[str, int]]] = {}
    for index, (name, features) in enumerate(sets.items()):
        prediction, folds = _cross_validated(frame, features, folds=int(args.folds), seed=int(args.seed) + index * 11)
        predictions[f"prediction__{name}"] = prediction
        fold_report[name] = folds
        metrics.append({
            "feature_set": name,
            "feature_count": len(features),
            "pr_auc": float(average_precision_score(y, prediction)),
            "roc_auc": float(roc_auc_score(y, prediction)),
            "brier": float(brier_score_loss(y, prediction)),
            "fixed_false_alert_event_recall": _recall_at_fixed_false_alerts(frame, prediction),
        })
    pd.DataFrame([{key: value for key, value in row.items() if key != "fixed_false_alert_event_recall"} for row in metrics]).to_csv(output / "metrics.csv", index=False)
    predictions.to_parquet(output / "grouped_oof.parquet", index=False)
    report = {
        "schema": "current_lineage_model_health_transition_ablation_v1",
        "status": "GROUPED_RESEARCH_OOF_COMPLETE",
        "research_only": True,
        "validation_contract": "event windows remain whole; stable controls are grouped into segment-week blocks; shuffled row-level CV is forbidden",
        "future_use_caveat": "OOF grouping is research evidence only, not chronological policy OOS or a promotion result",
        "coverage": coverage,
        "folds": fold_report,
        "metrics": metrics,
    }
    (output / "report.json").write_text(json.dumps(_safe(report), indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    print(json.dumps(_safe(run(_parser().parse_args())), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
