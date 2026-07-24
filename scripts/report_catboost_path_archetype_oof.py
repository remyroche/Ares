#!/usr/bin/env python3
"""Report bounded, validated OOF metrics for path-archetype CatBoost outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.catboost_archetype_classifier import (  # noqa: E402
    multiclass_classification_diagnostics,
)
from extreme_price_movements.path_archetype_support import (  # noqa: E402
    MERGED_PATH_ARCHETYPE_CLASSES,
)

# Kept as a public alias for report callers; the canonical classifier taxonomy
# is the merged seven-class contract.
PATH_SHAPE_TYPES = MERGED_PATH_ARCHETYPE_CLASSES

DEFAULT_TIMESTAMP_COLUMN = "__ts__"
DEFAULT_SIDE_COLUMN = "side"
DEFAULT_TRUE_CLASS_COLUMN = "path_archetype"
DEFAULT_PREDICTED_CLASS_COLUMN = "predicted_path_archetype"
DEFAULT_PROBABILITY_PREFIX = "probability__"
DEFAULT_FOLD_COLUMN = "oof_fold_id"
PROBABILITY_SUM_TOLERANCE = 1e-6


def _source_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (pd.Timestamp, pd.Timedelta)):
        return str(value)
    return value


def _validate_utc_timestamps(series: pd.Series, column: str) -> None:
    if not isinstance(series.dtype, pd.DatetimeTZDtype):
        raise ValueError(f"{column} must be timezone-aware UTC timestamps")
    if str(series.dtype.tz).upper() != "UTC":
        raise ValueError(f"{column} must use UTC, found {series.dtype.tz}")
    if series.isna().any():
        raise ValueError(f"{column} contains missing timestamps")


def _validate_frame(
    frame: pd.DataFrame,
    *,
    timestamp_column: str,
    side_column: str,
    true_class_column: str,
    predicted_class_column: str,
    probability_columns: list[str],
    class_names: list[str],
    fold_column: str,
) -> None:
    _validate_utc_timestamps(frame[timestamp_column], timestamp_column)
    if frame[side_column].isna().any():
        raise ValueError(f"{side_column} contains missing sides")
    if frame[true_class_column].isna().any():
        raise ValueError(f"{true_class_column} contains missing classes")
    if frame[predicted_class_column].isna().any():
        raise ValueError(f"{predicted_class_column} contains missing classes")
    labels = frame[true_class_column].astype(str)
    if not labels.isin(class_names).all():
        raise ValueError(f"{true_class_column} contains classes without probabilities")
    probabilities = frame.loc[:, probability_columns].to_numpy(dtype=float, copy=False)
    if not np.isfinite(probabilities).all():
        raise ValueError("Probability columns must be finite")
    if np.any((probabilities < 0.0) | (probabilities > 1.0)):
        raise ValueError("Probability columns must be within [0, 1]")
    if not np.allclose(
        probabilities.sum(axis=1), 1.0, atol=PROBABILITY_SUM_TOLERANCE, rtol=0.0
    ):
        raise ValueError("Probability rows must sum to 1 within tolerance")
    expected = np.asarray(class_names, dtype=str)[np.argmax(probabilities, axis=1)]
    if not np.array_equal(frame[predicted_class_column].astype(str).to_numpy(), expected):
        raise ValueError(f"{predicted_class_column} does not match probability argmax")
    folds = pd.to_numeric(frame[fold_column], errors="coerce").to_numpy(dtype=float)
    if (
        not np.isfinite(folds).all()
        or not np.equal(folds, np.floor(folds)).all()
        or np.any(folds < 0)
    ):
        raise ValueError(f"{fold_column} must contain non-negative integer OOF fold ids")


def _metrics(
    frame: pd.DataFrame,
    probability_columns: list[str],
    class_names: list[str],
    true_class_column: str,
    fold_column: str | None = None,
) -> dict[str, Any]:
    label_positions = {name: index for index, name in enumerate(class_names)}
    labels = frame[true_class_column].astype(str).map(label_positions).to_numpy()
    probabilities = frame.loc[:, probability_columns].to_numpy(dtype=float, copy=False)
    report = multiclass_classification_diagnostics(
        labels,
        probabilities,
        fold_ids=None if fold_column is None else frame[fold_column].to_numpy(),
        class_names=class_names,
    )
    confusion = np.asarray(report["confusion_matrix"], dtype=int)
    report["accuracy"] = float(np.trace(confusion) / max(int(confusion.sum()), 1))
    eces = [float(item["ece"]) for item in report["classwise"].values()]
    report["classwise_ece_summary"] = {
        "macro": float(np.mean(eces)),
        "minimum": float(np.min(eces)),
        "maximum": float(np.max(eces)),
    }
    return report


def _grouped_metrics(
    frame: pd.DataFrame,
    group_column: str,
    probability_columns: list[str],
    class_names: list[str],
    true_class_column: str,
) -> dict[str, dict[str, Any]]:
    groups: dict[str, dict[str, Any]] = {}
    for value, indices in frame.groupby(group_column, observed=True, sort=True).indices.items():
        groups[str(value)] = _metrics(
            frame.iloc[indices],
            probability_columns,
            class_names,
            true_class_column,
        )
    return groups


def _csv_rows(metrics: dict[str, Any]) -> Iterable[dict[str, Any]]:
    for grouping, grouped in metrics.items():
        for group, report in grouped.items():
            row = {
                "grouping": grouping,
                "group": group,
                "rows": report["rows"],
                "logloss": report["logloss"],
                "brier_macro": report["brier_macro"],
                "brier_weighted": report["brier_weighted"],
                "classwise_ece_macro": report["classwise_ece_summary"]["macro"],
                "f1_macro": report["f1_macro"],
                "f1_weighted": report["f1_weighted"],
                "accuracy": report["accuracy"],
                "classwise_json": json.dumps(report["classwise"], sort_keys=True),
                "confusion_matrix_json": json.dumps(report["confusion_matrix"]),
            }
            row.update(
                {
                    f"recall__{class_name}": class_metrics["recall"]
                    for class_name, class_metrics in report["classwise"].items()
                }
            )
            yield row


def run_report(
    input_path: Path,
    output_dir: Path,
    *,
    timestamp_column: str = DEFAULT_TIMESTAMP_COLUMN,
    side_column: str = DEFAULT_SIDE_COLUMN,
    true_class_column: str = DEFAULT_TRUE_CLASS_COLUMN,
    predicted_class_column: str = DEFAULT_PREDICTED_CLASS_COLUMN,
    probability_prefix: str = DEFAULT_PROBABILITY_PREFIX,
    fold_column: str = DEFAULT_FOLD_COLUMN,
) -> dict[str, Any]:
    """Read only the required OOF columns, validate them, and write reports."""
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    parquet_schema = pq.ParquetFile(input_path).schema_arrow
    schema = pd.Series(
        {field.name: str(field.type) for field in parquet_schema}, dtype="object"
    )
    probability_columns = sorted(
        column for column in schema.index if column.startswith(probability_prefix)
    )
    required = {
        timestamp_column,
        side_column,
        true_class_column,
        predicted_class_column,
        fold_column,
    }
    missing = sorted(required.difference(schema.index))
    if missing:
        raise ValueError(f"Missing required OOF columns: {missing}")
    expected_probability_columns = {
        f"{probability_prefix}{class_name}" for class_name in PATH_SHAPE_TYPES
    }
    if set(probability_columns) != expected_probability_columns:
        missing_probabilities = sorted(expected_probability_columns - set(probability_columns))
        unexpected_probabilities = sorted(set(probability_columns) - expected_probability_columns)
        raise ValueError(
            "Expected the exact canonical seven-class path-archetype probability vector; "
            f"missing={missing_probabilities}, unexpected={unexpected_probabilities}"
        )
    class_names = [column.removeprefix(probability_prefix) for column in probability_columns]
    if len(set(class_names)) != len(class_names):
        raise ValueError("Probability class names must be unique")
    columns = [
        timestamp_column,
        side_column,
        true_class_column,
        predicted_class_column,
        fold_column,
        *probability_columns,
    ]
    frame = pd.read_parquet(input_path, columns=columns)
    if frame.empty:
        raise ValueError("OOF input has no rows")
    _validate_frame(
        frame,
        timestamp_column=timestamp_column,
        side_column=side_column,
        true_class_column=true_class_column,
        predicted_class_column=predicted_class_column,
        probability_columns=probability_columns,
        class_names=class_names,
        fold_column=fold_column,
    )
    frame["__month__"] = pd.Categorical(frame[timestamp_column].dt.strftime("%Y-%m"))
    frame["__week__"] = pd.Categorical(
        frame[timestamp_column].dt.strftime("%G-W%V")
    )
    frame["__side_true_archetype__"] = pd.Categorical(
        frame[side_column].astype(str) + " x " + frame[true_class_column].astype(str)
    )
    metrics = {
        "overall": {
            "all": _metrics(
                frame,
                probability_columns,
                class_names,
                true_class_column,
                fold_column,
            )
        },
        "month": _grouped_metrics(
            frame, "__month__", probability_columns, class_names, true_class_column
        ),
        "week": _grouped_metrics(
            frame, "__week__", probability_columns, class_names, true_class_column
        ),
        "fold": _grouped_metrics(
            frame, fold_column, probability_columns, class_names, true_class_column
        ),
        "side": _grouped_metrics(
            frame, side_column, probability_columns, class_names, true_class_column
        ),
        "true_path_archetype": _grouped_metrics(
            frame,
            true_class_column,
            probability_columns,
            class_names,
            true_class_column,
        ),
        "side_x_true_path_archetype": _grouped_metrics(
            frame,
            "__side_true_archetype__",
            probability_columns,
            class_names,
            true_class_column,
        ),
    }
    manifest = {
        "schema": "report_catboost_path_archetype_oof_v1",
        "source": {
            "path": str(input_path),
            "sha256": _source_hash(input_path),
            "columns": {column: str(dtype) for column, dtype in schema.items()},
        },
        "columns": {
            "timestamp": timestamp_column,
            "side": side_column,
            "true_class": true_class_column,
            "predicted_class": predicted_class_column,
            "fold": fold_column,
            "probability_columns": probability_columns,
        },
        "class_names": class_names,
        "reported_rows": int(len(frame)),
        "claim": (
            "Metrics are OOF-only and are limited to rows with validated "
            "non-negative integer OOF fold ids."
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "oof_metrics.json").write_text(
        json.dumps(_json_safe(metrics), indent=2, sort_keys=True) + "\n"
    )
    pd.DataFrame(_csv_rows(metrics)).to_csv(output_dir / "oof_metrics.csv", index=False)
    (output_dir / "oof_manifest.json").write_text(
        json.dumps(_json_safe(manifest), indent=2, sort_keys=True) + "\n"
    )
    return {"manifest": manifest, "metrics": metrics}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--timestamp-col", default=DEFAULT_TIMESTAMP_COLUMN)
    parser.add_argument("--side-col", default=DEFAULT_SIDE_COLUMN)
    parser.add_argument("--true-class-col", default=DEFAULT_TRUE_CLASS_COLUMN)
    parser.add_argument("--predicted-class-col", default=DEFAULT_PREDICTED_CLASS_COLUMN)
    parser.add_argument("--prob-prefix", default=DEFAULT_PROBABILITY_PREFIX)
    parser.add_argument("--fold-col", default=DEFAULT_FOLD_COLUMN)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_report(
        args.input,
        args.output_dir,
        timestamp_column=args.timestamp_col,
        side_column=args.side_col,
        true_class_column=args.true_class_col,
        predicted_class_column=args.predicted_class_col,
        probability_prefix=args.prob_prefix,
        fold_column=args.fold_col,
    )


if __name__ == "__main__":
    main()
