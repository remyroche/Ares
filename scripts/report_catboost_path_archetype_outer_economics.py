#!/usr/bin/env python3
"""Score final CatBoost outer OOF probabilities with fold-train-only economics."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.class_balance_oof_economics import (  # noqa: E402
    BalanceArmOOF,
    EconomicOOFConfig,
    score_class_balance_oof_economics,
)
from extreme_price_movements.path_archetype_support import (  # noqa: E402
    merge_fast_realization_winner,
)

OUTCOME_COLUMNS = (
    "path_arch_final_return_net_1pct",
    "path_arch_peak_mfe_atr",
    "path_arch_mae_12h_r",
    "path_arch_mae_before_meaningful_mfe_r",
    "path_arch_stop_before_meaningful_mfe",
    "path_arch_reaches_meaningful_mfe",
    "path_arch_time_to_first_meaningful_mfe_h",
    "path_arch_peak_retention_ratio",
    "path_arch_time_to_trailing_h",
    "path_arch_mfe_to_activation_distance",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _required_columns(path: Path, required: set[str]) -> None:
    observed = set(pq.ParquetFile(path).schema_arrow.names)
    missing = sorted(required.difference(observed))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")


def run_report(
    *,
    oof_path: Path,
    labels_path: Path,
    training_report_path: Path,
    output_path: Path,
    embargo_hours: float = 24.0,
) -> dict[str, Any]:
    """Reconstruct exact outer folds and score one frozen production arm."""

    oof_path = Path(oof_path)
    labels_path = Path(labels_path)
    training_report_path = Path(training_report_path)
    output_path = Path(output_path)
    training_report = _read_json(training_report_path)
    diagnostics = training_report.get("oof_diagnostics")
    if not isinstance(diagnostics, dict):
        raise ValueError("training report is missing OOF diagnostics")
    classes = tuple(map(str, diagnostics.get("class_names", ())))
    if len(classes) < 2 or len(set(classes)) != len(classes):
        raise ValueError("training report has an invalid frozen class order")
    probability_columns = [f"probability__{name}" for name in classes]
    oof_columns = {
        "candidate_id",
        "__ts__",
        "__label_end_ts__",
        "side_name",
        "oof_fold_id",
        "validation_start",
        *probability_columns,
    }
    label_columns = {
        "candidate_id",
        "__ts__",
        "__label_end_ts__",
        "side_name",
        "path_arch_complete_24h",
        "path_shape_archetype",
        *OUTCOME_COLUMNS,
    }
    _required_columns(oof_path, oof_columns)
    _required_columns(labels_path, label_columns)
    oof = pd.read_parquet(oof_path, columns=sorted(oof_columns))
    if oof.empty or oof["candidate_id"].duplicated().any():
        raise ValueError("OOF probabilities must contain unique candidate identities")
    side_values = tuple(pd.unique(oof["side_name"].astype("string")))
    if len(side_values) != 1 or side_values[0] not in {"long", "short"}:
        raise ValueError("OOF probabilities must contain exactly one canonical side")
    side = str(side_values[0])
    labels = pd.read_parquet(labels_path, columns=sorted(label_columns))
    labels = labels.loc[
        labels["side_name"].astype("string").eq(side)
        & pd.to_numeric(labels["path_arch_complete_24h"], errors="coerce").eq(1)
    ].copy()
    if labels.empty or labels["candidate_id"].duplicated().any():
        raise ValueError("complete side-local labels must have unique candidate IDs")
    labels["__ts__"] = pd.to_datetime(labels["__ts__"], utc=True, errors="coerce")
    labels["__label_end_ts__"] = pd.to_datetime(
        labels["__label_end_ts__"], utc=True, errors="coerce"
    )
    if labels[["__ts__", "__label_end_ts__"]].isna().any().any():
        raise ValueError("labels contain invalid decision or label-end timestamps")
    labels["path_archetype"] = merge_fast_realization_winner(
        labels["path_shape_archetype"].astype("string")
    ).astype("string")
    class_codes = labels["path_archetype"].map(
        {name: index for index, name in enumerate(classes)}
    )
    if class_codes.isna().any():
        unexpected = sorted(
            set(labels.loc[class_codes.isna(), "path_archetype"].astype(str))
        )
        raise ValueError(f"labels fall outside the frozen class order: {unexpected}")

    position_by_id = pd.Series(
        np.arange(len(labels), dtype=np.int64),
        index=labels["candidate_id"].astype(str),
    )
    oof_positions = oof["candidate_id"].astype(str).map(position_by_id)
    if oof_positions.isna().any() or oof_positions.duplicated().any():
        raise ValueError("OOF identities are not an exact subset of complete labels")
    oof_positions_array = oof_positions.to_numpy(dtype=np.int64)
    oof_timestamps = pd.to_datetime(oof["__ts__"], utc=True, errors="coerce")
    if not np.array_equal(
        oof_timestamps.to_numpy(),
        labels.iloc[oof_positions_array]["__ts__"].to_numpy(),
    ):
        raise ValueError("OOF identity timestamps do not match canonical labels")

    probabilities = np.full((len(labels), len(classes)), np.nan, dtype=np.float64)
    probabilities[oof_positions_array] = oof[probability_columns].to_numpy(
        dtype=np.float64
    )
    fold_ids = np.full(len(labels), -1, dtype=np.int64)
    fold_ids[oof_positions_array] = pd.to_numeric(
        oof["oof_fold_id"], errors="raise"
    ).to_numpy(dtype=np.int64)
    folds: list[SimpleNamespace] = []
    expected_fold_reports = {
        int(item["fold_id"]): item
        for item in diagnostics.get("fold_fit_reports", ())
        if isinstance(item, dict) and "fold_id" in item
    }
    embargo = pd.Timedelta(hours=float(embargo_hours))
    for fold_id in sorted(pd.unique(fold_ids[fold_ids >= 0])):
        valid = np.flatnonzero(fold_ids == fold_id)
        validation_values = pd.to_datetime(
            oof.loc[
                pd.to_numeric(oof["oof_fold_id"], errors="coerce").eq(fold_id),
                "validation_start",
            ],
            utc=True,
            errors="coerce",
        ).dropna()
        if validation_values.nunique() != 1:
            raise ValueError(f"fold {fold_id} has no unique validation start")
        validation_start = validation_values.iloc[0]
        train = np.flatnonzero(
            labels["__ts__"].lt(validation_start - embargo).to_numpy()
            & labels["__label_end_ts__"].lt(validation_start).to_numpy()
        )
        expected = expected_fold_reports.get(int(fold_id))
        if expected is None:
            raise ValueError(f"fold {fold_id} is absent from the training report")
        if int(expected["train_rows"]) != len(train) or int(
            expected["validation_rows"]
        ) != len(valid):
            raise ValueError(
                f"fold {fold_id} support differs from the frozen training report"
            )
        folds.append(
            SimpleNamespace(
                fold_id=int(fold_id),
                train_indices=train,
                validation_indices=valid,
            )
        )

    balance = training_report.get("class_balance")
    selection = (
        balance.get("selection_provenance") if isinstance(balance, dict) else None
    )
    if not isinstance(selection, dict):
        raise ValueError("training report lacks class-balance selection provenance")
    fingerprints = {
        "structural": str(selection.get("structural_fingerprint", "")),
        "feature": str(selection.get("feature_fingerprint", "")),
        "geometry": str(selection.get("geometry_fingerprint", "")),
    }
    if not all(fingerprints.values()):
        raise ValueError(
            "training report lacks structural/feature/geometry fingerprints"
        )
    arm = BalanceArmOOF(
        probabilities=probabilities,
        fold_ids=fold_ids,
        folds=folds,
        classes=classes,
        structural_fingerprint=fingerprints["structural"],
        feature_fingerprint=fingerprints["feature"],
        geometry_fingerprint=fingerprints["geometry"],
        oof_guard=balance.get("selected_arm_oof_guard"),
        row_ids=labels["candidate_id"].to_numpy(),
    )
    scorer_config = EconomicOOFConfig(
        timestamp_col="__ts__",
        side_col="side_name",
        label_end_col="__label_end_ts__",
        identity_col="candidate_id",
        embargo=embargo,
        expected_arms=("uniform",),
    )
    report = score_class_balance_oof_economics(
        labels,
        class_codes.to_numpy(dtype=np.int64),
        {"uniform": arm},
        config=scorer_config,
    )
    result = {
        "schema": "catboost_path_archetype_outer_oof_economics_v1",
        "side": side,
        "status": "complete",
        "claim": (
            "Every class-outcome prior is fit from the exact corresponding "
            "outer-fold training indices; validation outcomes are evaluation-only."
        ),
        "sources": {
            "oof_probabilities": {
                "path": str(oof_path),
                "sha256": _sha256(oof_path),
            },
            "canonical_labels": {
                "path": str(labels_path),
                "sha256": _sha256(labels_path),
            },
            "training_report": {
                "path": str(training_report_path),
                "sha256": _sha256(training_report_path),
            },
        },
        "rows": {
            "complete_side_labels": int(len(labels)),
            "outer_oof": int((fold_ids >= 0).sum()),
        },
        "folds": [
            {
                "fold_id": int(fold.fold_id),
                "train_rows": int(len(fold.train_indices)),
                "validation_rows": int(len(fold.validation_indices)),
            }
            for fold in folds
        ],
        "scoring": report,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oof", required=True, type=Path)
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--training-report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--embargo-hours", type=float, default=24.0)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = run_report(
        oof_path=args.oof,
        labels_path=args.labels,
        training_report_path=args.training_report,
        output_path=args.output,
        embargo_hours=args.embargo_hours,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
