#!/usr/bin/env python3
"""Expanding-time OOS active-transition predictions on the research panel.

The model and feature contract intentionally match the grouped-OOF active head
so validation geometry can be compared directly.  Each evaluation month is
predicted by a model trained only on rows whose conservative label-availability
timestamp precedes that month.  Upstream pooled state geometry remains
research-only and is disclosed in the manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    roc_auc_score,
)


EXCLUDED_COLUMNS = {
    "source_utc",
    "execution_decision_utc",
    "segment_id",
    "target__pooled_state",
}


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


def feature_columns(frame: pd.DataFrame) -> list[str]:
    return [
        name
        for name in frame
        if name not in EXCLUDED_COLUMNS
        and not name.startswith("target__")
        and pd.api.types.is_numeric_dtype(frame[name])
    ]


def conservative_label_available_utc(frame: pd.DataFrame) -> pd.Series:
    source = pd.to_datetime(frame["source_utc"], utc=True, errors="raise")
    floor = source + pd.Timedelta(hours=12)
    declared = pd.to_datetime(
        frame.get("target__available_utc"), utc=True, errors="coerce"
    )
    available = pd.concat(
        [floor.rename("floor"), declared.rename("declared")], axis=1
    ).max(axis=1)
    return pd.to_datetime(available, utc=True, errors="raise")


def chronological_month_folds(
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
    months = pd.date_range(first, last, freq="MS", tz="UTC")
    folds: list[tuple[pd.Timestamp, np.ndarray, np.ndarray]] = []
    for start in months:
        end = start + pd.offsets.MonthBegin(1)
        evaluation = np.flatnonzero(
            source.ge(start).to_numpy() & source.lt(end).to_numpy()
        )
        train = np.flatnonzero(available.lt(start).to_numpy())
        if not len(evaluation):
            continue
        train_months = source.iloc[train].dt.tz_localize(None).dt.to_period("M")
        if train_months.nunique() < int(minimum_train_months):
            continue
        if len(train) and available.iloc[train].max() >= start:
            raise AssertionError("chronological fold contains unavailable labels")
        folds.append((start, train, evaluation))
    return folds


def _metrics(y: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    if len(np.unique(y)) < 2:
        return {
            "rows": int(len(y)),
            "positive_rows": int(y.sum()),
            "prevalence": float(y.mean()) if len(y) else np.nan,
            "average_precision": np.nan,
            "roc_auc": np.nan,
            "brier": float(brier_score_loss(y, prediction)) if len(y) else np.nan,
            "f1_at_0_5": float(f1_score(y, prediction >= 0.5, zero_division=0))
            if len(y)
            else np.nan,
        }
    return {
        "rows": int(len(y)),
        "positive_rows": int(y.sum()),
        "prevalence": float(y.mean()),
        "average_precision": float(average_precision_score(y, prediction)),
        "roc_auc": float(roc_auc_score(y, prediction)),
        "brier": float(brier_score_loss(y, prediction)),
        "f1_at_0_5": float(
            f1_score(y, prediction >= 0.5, zero_division=0)
        ),
    }


def _episode_count(mask: np.ndarray, timestamps: pd.Series) -> int:
    selected = pd.to_datetime(timestamps.loc[mask], utc=True).sort_values()
    if selected.empty:
        return 0
    return int(1 + selected.diff().gt(pd.Timedelta(hours=1)).iloc[1:].sum())


def active_operating_curve(
    oos: pd.DataFrame, thresholds: Sequence[float]
) -> pd.DataFrame:
    active_events = oos.loc[oos["target__transition_active"].eq(1)].copy()
    event_ids = active_events["target__event_id"].dropna().astype(str).unique()
    total_days = (
        oos["source_utc"].max() - oos["source_utc"].min()
    ) / pd.Timedelta(days=1)
    operating_rows: list[dict[str, Any]] = []
    for threshold in thresholds:
        event_detected = []
        for event_id in event_ids:
            local = active_events.loc[
                active_events["target__event_id"].astype(str).eq(event_id)
            ]
            event_detected.append(bool(local["prediction"].ge(threshold).any()))
        false_mask = (
            oos["prediction"].ge(threshold)
            & ~oos["target__transition_active"].astype(bool)
        ).to_numpy()
        false_episodes = _episode_count(false_mask, oos["source_utc"])
        operating_rows.append(
            {
                "threshold": float(threshold),
                "event_count": int(len(event_ids)),
                "event_recall": float(np.mean(event_detected))
                if event_detected
                else np.nan,
                "false_alert_episodes": int(false_episodes),
                "false_alert_episodes_per_30d": float(
                    false_episodes * 30.0 / max(float(total_days), 1.0)
                ),
            }
        )
    return pd.DataFrame(operating_rows)


def fit_predict(
    frame: pd.DataFrame,
    *,
    first_evaluation_month: str,
    last_evaluation_month: str,
    minimum_train_months: int,
    thresholds: Sequence[float],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    work = frame.copy()
    work["source_utc"] = pd.to_datetime(
        work["source_utc"], utc=True, errors="raise"
    )
    features = feature_columns(work)
    x = work[features].apply(pd.to_numeric, errors="coerce")
    y = work["target__transition_active"].astype(int).to_numpy()
    folds = chronological_month_folds(
        work,
        first_evaluation_month=first_evaluation_month,
        last_evaluation_month=last_evaluation_month,
        minimum_train_months=minimum_train_months,
    )
    if not folds:
        raise ValueError("no eligible chronological evaluation folds")
    predictions: list[pd.DataFrame] = []
    fold_rows: list[dict[str, Any]] = []
    for fold_index, (start, train, evaluation) in enumerate(folds):
        if len(np.unique(y[train])) < 2:
            raise ValueError(f"{start:%Y-%m} training data has one class")
        model = LGBMClassifier(
            objective="binary",
            n_estimators=320,
            learning_rate=0.035,
            num_leaves=63,
            min_child_samples=30,
            subsample=0.85,
            colsample_bytree=0.75,
            reg_alpha=0.5,
            reg_lambda=8.0,
            class_weight="balanced",
            random_state=2219 + fold_index,
            n_jobs=4,
            verbosity=-1,
        )
        model.fit(x.iloc[train], y[train])
        prediction = model.predict_proba(x.iloc[evaluation])[:, 1]
        local = work.iloc[evaluation][
            ["source_utc", "target__event_id", "target__transition_active"]
        ].copy()
        local["prediction"] = prediction.astype(np.float32)
        local["evaluation_month"] = start.strftime("%Y-%m")
        local["train_rows"] = int(len(train))
        local["train_end_label_available_utc"] = conservative_label_available_utc(
            work.iloc[train]
        ).max()
        predictions.append(local)
        fold_rows.append(
            {
                "evaluation_month": start.strftime("%Y-%m"),
                "train_rows": int(len(train)),
                "train_positive_rows": int(y[train].sum()),
                "train_start_utc": work.iloc[train]["source_utc"].min(),
                "train_end_source_utc": work.iloc[train]["source_utc"].max(),
                "train_end_label_available_utc": local[
                    "train_end_label_available_utc"
                ].iloc[0],
                **_metrics(y[evaluation], prediction),
            }
        )
    oos = pd.concat(predictions, ignore_index=True)
    if oos["source_utc"].duplicated().any():
        raise AssertionError("chronological predictions contain duplicate hours")
    fold_metrics = pd.DataFrame(fold_rows)
    return oos, fold_metrics, active_operating_curve(oos, thresholds), features


def run(args: argparse.Namespace) -> dict[str, Any]:
    dataset_path = Path(args.dataset)
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    frame = pd.read_parquet(dataset_path)
    oos, fold_metrics, operating, features = fit_predict(
        frame,
        first_evaluation_month=args.first_evaluation_month,
        last_evaluation_month=args.last_evaluation_month,
        minimum_train_months=int(args.minimum_train_months),
        thresholds=args.thresholds,
    )
    output.mkdir(parents=True, exist_ok=False)
    prediction_path = output / "chronological_oos.parquet"
    fold_path = output / "fold_metrics.csv"
    operating_path = output / "operating_curve.csv"
    features_path = output / "features.json"
    oos.to_parquet(prediction_path, index=False, compression="zstd")
    fold_metrics.to_csv(fold_path, index=False)
    operating.to_csv(operating_path, index=False)
    _write_json(features_path, {"features": features})
    y = oos["target__transition_active"].to_numpy(int)
    prediction = oos["prediction"].to_numpy(float)
    manifest = {
        "schema": "active_transition_chronological_oos_v1",
        "status": "RESEARCH_ONLY_CHRONOLOGICAL_LABEL_OOS_COMPLETE",
        "promotion_eligible": False,
        "promotion_blocker": (
            "model folds are chronological and label-purged, but the upstream "
            "five-state geometry/research panel is pooled and policy lambdas "
            "have not yet been frozen on a prior event block"
        ),
        "validation_contract": {
            "fold": "expanding monthly",
            "label_purge": (
                "train max(max(source+12h,target__available_utc)) < "
                "evaluation month start"
            ),
            "minimum_train_months": int(args.minimum_train_months),
            "first_evaluation_month": args.first_evaluation_month,
            "last_evaluation_month": args.last_evaluation_month,
            "feature_contract": "identical to grouped active-head v1",
            "upstream_geometry": "pooled research; non-production-causal",
        },
        "metrics": _metrics(y, prediction),
        "feature_count": len(features),
        "event_count": int(
            oos.loc[
                oos["target__transition_active"].eq(1), "target__event_id"
            ].dropna().nunique()
        ),
        "sources": {
            "dataset": {
                "path": str(dataset_path),
                "sha256": _sha256(dataset_path),
            }
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
        "operating_curve": operating_path,
        "features": features_path,
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
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--first-evaluation-month", default="2024-01-01")
    parser.add_argument("--last-evaluation-month", default="2026-07-01")
    parser.add_argument("--minimum-train-months", type=int, default=12)
    parser.add_argument(
        "--thresholds", type=float, nargs="+", default=(0.25, 0.50, 0.75)
    )
    return parser


def main() -> None:
    print(json.dumps(_safe(run(_parser().parse_args())), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
