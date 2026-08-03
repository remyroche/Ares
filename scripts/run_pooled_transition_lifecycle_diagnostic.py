#!/usr/bin/env python3
"""Run the bounded pooled adverse-regime lifecycle diagnostic.

This is deliberately a *research* classifier, not an execution-policy gate.
It builds three lifecycle labels from the immutable H12 global-book panel:

* onset within the next three anchors;
* recovery within the next three anchors, conditional on being active now; and
* a fresh adverse onset after a completed three-anchor recovery.

Every derived label carries the maximum availability time of every source
label it touches.  Source family is never a model input: it is retained as a
domain flag for reporting and used only to balance training mass.  The last
target is expected to fail closed when the event count is inadequate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

try:
    from scripts.run_cross_era_regime_transition_classifier_ablation import (
        RANDOM_STATE,
        _model,
        _purge_near_validation,
        feature_sets,
    )
except ModuleNotFoundError:  # direct execution from scripts/
    from run_cross_era_regime_transition_classifier_ablation import (
        RANDOM_STATE,
        _model,
        _purge_near_validation,
        feature_sets,
    )


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PANEL = ROOT / "data_perp/artifacts/cross_era_global_book_transition_research_panel_20260730_v4"
DEFAULT_OUTPUT = ROOT / "data_perp/artifacts/pooled_transition_lifecycle_diagnostic_20260730_v1"
SCHEMA = "pooled_transition_lifecycle_diagnostic_v1"
HORIZON_HOURS = 12
RECOVERY_CONFIRM_HOURS = 3
ONSET_LEAD_HOURS = 3
REVERSAL_PRIOR_ACTIVE_HOURS = 3
MIN_ROWS = 60
MIN_POSITIVES = 12


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe(value: Any) -> Any:
    if isinstance(value, (Path, pd.Timestamp)):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(_safe(payload), indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _exact_shift(frame: pd.DataFrame, column: str, offsets: Sequence[int]) -> list[pd.Series]:
    indexed = frame.set_index("cohort_anchor_utc")[column]
    return [
        indexed.reindex(frame["cohort_anchor_utc"] + pd.Timedelta(hours=offset)).reset_index(drop=True)
        for offset in offsets
    ]


def _availability_shift(frame: pd.DataFrame, column: str, offsets: Sequence[int]) -> pd.DataFrame:
    return pd.concat(_exact_shift(frame, column, offsets), axis=1)


def derive_lifecycle_targets(panel: pd.DataFrame) -> pd.DataFrame:
    """Add exact availability-bound onset/recovery/reversal targets.

    Recovery is conditional on an active adverse state at the anchor and needs
    three subsequent inactive active-state anchors.  Reversal is a new active
    onset within three anchors after three active anchors followed by a fully
    observed three-anchor inactive recovery.  Undefined conditioning rows are
    null rather than negative examples.
    """

    required = {
        "cohort_anchor_utc", "source_family", "horizon_hours",
        "book_fraction", "target__active_adverse",
        "target__active_adverse_available_utc", "target__adverse_onset_within_3h",
        "target__adverse_onset_within_3h_available_utc",
    }
    missing = sorted(required.difference(panel.columns))
    if missing:
        raise ValueError(f"panel lacks lifecycle dependencies: {missing}")
    pieces: list[pd.DataFrame] = []
    for _, source in panel.groupby(["source_family", "horizon_hours", "book_fraction"], sort=False):
        work = source.sort_values("cohort_anchor_utc").reset_index(drop=True).copy()
        active = pd.to_numeric(work["target__active_adverse"], errors="coerce")
        active_available = pd.to_datetime(work["target__active_adverse_available_utc"], utc=True, errors="coerce")

        # Existing lead-onset target is copied only after proving its declared
        # availability is present; this gives all three heads one contract.
        work["target__lifecycle_onset_within_3h"] = pd.to_numeric(
            work["target__adverse_onset_within_3h"], errors="coerce"
        )
        work["target__lifecycle_onset_within_3h_available_utc"] = pd.to_datetime(
            work["target__adverse_onset_within_3h_available_utc"], utc=True, errors="coerce"
        )

        future_active = _availability_shift(work, "target__active_adverse", (1, 2, 3))
        future_available = _availability_shift(work, "target__active_adverse_available_utc", (0, 1, 2, 3))
        recovery_complete = (
            active.eq(1.0) & future_active.notna().all(axis=1) & future_available.notna().all(axis=1)
        )
        work["target__lifecycle_recovery_within_3h"] = (
            future_active.eq(0.0).all(axis=1).astype(float).where(recovery_complete)
        )
        work["target__lifecycle_recovery_within_3h_available_utc"] = future_available.max(axis=1).where(recovery_complete)

        prior_active = _availability_shift(work, "target__active_adverse", (-6, -5, -4))
        quiet_active = _availability_shift(work, "target__active_adverse", (-3, -2, -1, 0))
        next_active = _availability_shift(work, "target__active_adverse", (1, 2, 3))
        reversal_values = pd.concat([prior_active, quiet_active, next_active], axis=1)
        reversal_available = _availability_shift(
            work, "target__active_adverse_available_utc", (-6, -5, -4, -3, -2, -1, 0, 1, 2, 3)
        )
        reversal_complete = reversal_values.notna().all(axis=1) & reversal_available.notna().all(axis=1)
        prior_is_active = prior_active.eq(1.0).all(axis=1)
        recovered = quiet_active.eq(0.0).all(axis=1)
        work["target__lifecycle_reversal_after_recovery_within_3h"] = (
            (prior_is_active & recovered & next_active.eq(1.0).any(axis=1)).astype(float).where(reversal_complete)
        )
        work["target__lifecycle_reversal_after_recovery_within_3h_available_utc"] = reversal_available.max(axis=1).where(reversal_complete)
        pieces.append(work)
    result = pd.concat(pieces, ignore_index=True)
    for target in lifecycle_target_names():
        availability = f"{target}_available_utc"
        valid = result[target].notna()
        if valid.any() and result.loc[valid, availability].isna().any():
            raise ValueError(f"{target} has resolved values without availability")
    return result


def lifecycle_target_names() -> tuple[str, ...]:
    return (
        "target__lifecycle_onset_within_3h",
        "target__lifecycle_recovery_within_3h",
        "target__lifecycle_reversal_after_recovery_within_3h",
    )


def source_balanced_weights(frame: pd.DataFrame) -> np.ndarray:
    """Equalize source mass without ever letting source family become a feature."""
    counts = frame["source_family"].value_counts()
    weights = frame["source_family"].map(lambda value: 1.0 / float(counts.loc[value])).to_numpy(float)
    return weights / weights.mean()


def _fit_prediction(
    train: pd.DataFrame, y: pd.Series, validation: pd.DataFrame, columns: Sequence[str], model_name: str,
) -> np.ndarray:
    model = _model(model_name, columns)
    model.fit(train, y, model__sample_weight=source_balanced_weights(train))
    return model.predict_proba(validation)[:, 1]


def _nested_shrunk_weighted_prediction(
    train: pd.DataFrame, y: pd.Series, validation: pd.DataFrame, columns: Sequence[str], model_name: str,
) -> tuple[np.ndarray, float]:
    local = train.reset_index(drop=True)
    local_y = y.reset_index(drop=True)
    groups = local["cv_group_id"].astype(str)
    folds = min(3, groups.nunique(), int(local_y.sum()), int((1 - local_y).sum()))
    if folds < 3:
        raise ValueError("insufficient inner grouped support for calibration")
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE + 29)
    prediction = np.full(len(local), np.nan)
    timestamps = pd.to_datetime(local["cohort_anchor_utc"], utc=True)
    for train_index, validation_index in splitter.split(local, local_y, groups):
        train_index = _purge_near_validation(train_index, validation_index, timestamps, embargo_hours=36)
        if len(train_index) < 40 or local_y.iloc[train_index].nunique() < 2:
            raise ValueError("inner 36h embargo leaves insufficient support")
        prediction[validation_index] = _fit_prediction(
            local.iloc[train_index], local_y.iloc[train_index], local.iloc[validation_index], columns, model_name
        )
    if not np.isfinite(prediction).all():
        raise ValueError("inner OOF calibration coverage is incomplete")
    weights = source_balanced_weights(local)
    prior = float(np.average(local_y.to_numpy(float), weights=weights))
    candidates = np.asarray((0.0, 0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.0))
    losses = [
        brier_score_loss(local_y, np.clip(prior + value * (prediction - prior), 1e-8, 1 - 1e-8), sample_weight=weights)
        for value in candidates
    ]
    shrink = float(candidates[int(np.argmin(losses))])
    raw = _fit_prediction(local, local_y, validation, columns, model_name)
    return np.clip(prior + shrink * (raw - prior), 1e-8, 1 - 1e-8), shrink


def _metrics(frame: pd.DataFrame, *, target: str, model: str, scope: str) -> dict[str, Any]:
    y = frame["target"].to_numpy(float)
    p = np.clip(frame["prediction"].to_numpy(float), 1e-8, 1 - 1e-8)
    w = source_balanced_weights(frame)
    selected = frame["selected_top10"].to_numpy(bool)
    prevalence = float(np.average(y, weights=w))
    precision = float(y[selected].mean()) if selected.any() else float("nan")
    return {
        "target": target, "model": model, "scope": scope, "rows": int(len(frame)),
        "positive_rows": int(y.sum()), "prevalence": prevalence,
        "roc_auc": float(roc_auc_score(y, p, sample_weight=w)) if len(np.unique(y)) == 2 else float("nan"),
        "average_precision": float(average_precision_score(y, p, sample_weight=w)) if y.sum() else float("nan"),
        "brier": float(brier_score_loss(y, p, sample_weight=w)),
        "log_loss": float(log_loss(y, p, labels=[0.0, 1.0], sample_weight=w)),
        "top10_selected_rows": int(selected.sum()), "top10_precision": precision,
        "top10_lift": float(precision / prevalence) if prevalence and np.isfinite(precision) else float("nan"),
    }


def grouped_oof(frame: pd.DataFrame, *, target: str, columns: Sequence[str], model_name: str) -> pd.DataFrame:
    availability_column = f"{target}_available_utc"
    valid = frame[target].notna() & frame[availability_column].notna()
    work = frame.loc[valid].reset_index(drop=True).copy()
    y = pd.to_numeric(work[target], errors="raise").astype(int)
    groups = work["cv_group_id"].astype(str)
    folds = min(5, groups.nunique(), int(y.sum()), int((1 - y).sum()))
    if len(work) < MIN_ROWS or int(y.sum()) < MIN_POSITIVES or folds < 3:
        raise ValueError("insufficient_binary_grouped_support")
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
    score = np.full(len(work), np.nan)
    fold_ids = np.full(len(work), -1, dtype=int)
    shrink = np.full(len(work), np.nan)
    timestamp = pd.to_datetime(work["cohort_anchor_utc"], utc=True)
    for fold, (train_index, validation_index) in enumerate(splitter.split(work, y, groups)):
        train_index = _purge_near_validation(train_index, validation_index, timestamp, embargo_hours=36)
        if len(train_index) < 40 or y.iloc[train_index].nunique() < 2:
            raise ValueError("outer_36h_embargo_leaves_insufficient_support")
        prediction, local_shrink = _nested_shrunk_weighted_prediction(
            work.iloc[train_index], y.iloc[train_index], work.iloc[validation_index], columns, model_name
        )
        score[validation_index] = prediction
        fold_ids[validation_index] = fold
        shrink[validation_index] = local_shrink
    if not np.isfinite(score).all():
        raise ValueError("incomplete_grouped_oof_coverage")
    selected = np.zeros(len(work), dtype=bool)
    # One pooled global alert book over all outer-OOF rows.  This is not a
    # timestamp-, side- or source-quota selection.
    count = max(1, int(math.ceil(0.10 * len(work))))
    order = np.lexsort((work["cohort_anchor_utc"].astype("int64").to_numpy(), -score))
    selected[order[:count]] = True
    output = work.loc[:, ["cohort_anchor_utc", "source_family", "economics_tier", "mapping_provenance_role", "cv_group_id"]].copy()
    output["target_available_utc"] = pd.to_datetime(work[availability_column], utc=True)
    output["target"] = y.to_numpy(float)
    output["prediction"] = score
    output["cv_fold"] = fold_ids
    output["selected_top10"] = selected
    output["calibration_shrinkage_weight"] = shrink
    return output


def run_diagnostic(panel: pd.DataFrame, feature_columns: Sequence[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    work = panel.loc[panel["horizon_hours"].eq(HORIZON_HOURS) & panel["context_available"].astype(bool)].copy()
    # The frozen current forward rows cannot participate in a pooled diagnosis.
    work = work.loc[~(work["source_family"].eq("current_exact_spread_mayjul2026") & work["mapping_provenance_role"].ne("strict_oof"))].copy()
    work = derive_lifecycle_targets(work)
    families = feature_sets(feature_columns)
    columns = list(dict.fromkeys([*families["coordinates_only"], *families["raw_state_only"]]))
    # Require usable support in every source; a model must not silently lean on
    # source-specific missingness as a domain identifier.
    usable = [
        column for column in columns
        if all(group[column].notna().sum() >= max(20, int(0.50 * len(group))) and pd.to_numeric(group[column], errors="coerce").nunique(dropna=True) > 1
               for _, group in work.groupby("source_family", sort=False))
    ]
    if not usable:
        raise ValueError("no_cross_source_usable_features")
    metrics: list[dict[str, Any]] = []
    outputs: list[pd.DataFrame] = []
    skipped: list[dict[str, Any]] = []
    coverage: list[dict[str, Any]] = []
    for target in lifecycle_target_names():
        availability = f"{target}_available_utc"
        for source, local in work.groupby("source_family", sort=True):
            valid = local[target].notna() & local[availability].notna()
            coverage.append({"target": target, "source_family": source, "rows": int(len(local)), "resolved_rows": int(valid.sum()), "positive_rows": int(pd.to_numeric(local.loc[valid, target], errors="coerce").sum()), "groups": int(local.loc[valid, "cv_group_id"].nunique())})
        for model_name in ("logistic", "extra_trees"):
            try:
                prediction = grouped_oof(work, target=target, columns=usable, model_name=model_name)
            except ValueError as error:
                skipped.append({"target": target, "model": model_name, "reason": str(error)})
                continue
            prediction["model"] = model_name
            prediction["feature_count"] = len(usable)
            outputs.append(prediction.assign(target_name=target))
            metrics.append(_metrics(prediction, target=target, model=model_name, scope="pooled_source_balanced"))
            for source, local in prediction.groupby("source_family", sort=True):
                metrics.append(_metrics(local, target=target, model=model_name, scope=f"source::{source}"))
    return pd.DataFrame(metrics), (pd.concat(outputs, ignore_index=True) if outputs else pd.DataFrame()), pd.DataFrame(skipped), pd.DataFrame(coverage)


def run(args: argparse.Namespace) -> dict[str, Any]:
    panel_root = Path(args.panel)
    panel_path = panel_root / "transition_research_panel.parquet"
    manifest_path = panel_root / "manifest.json"
    sidecar = panel_root / "manifest.sha256"
    if not all(path.is_file() for path in (panel_path, manifest_path, sidecar)):
        raise FileNotFoundError("cross-era transition panel is incomplete")
    if sidecar.read_text(encoding="utf-8").split()[0] != sha256(manifest_path):
        raise ValueError("cross-era transition panel manifest checksum fails")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite immutable output {output}")
    metrics, predictions, skipped, coverage = run_diagnostic(pd.read_parquet(panel_path), manifest["feature_columns"])
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(dir=output.parent, prefix=f".{output.name}."))
    paths = {"metrics": temporary / "metrics.csv", "predictions": temporary / "grouped_oof_predictions.parquet", "skipped": temporary / "skipped_arms.csv", "coverage": temporary / "coverage.csv"}
    metrics.to_csv(paths["metrics"], index=False)
    predictions.to_parquet(paths["predictions"], index=False, compression="zstd")
    skipped.to_csv(paths["skipped"], index=False)
    coverage.to_csv(paths["coverage"], index=False)
    result = {
        "schema": SCHEMA,
        "status": "GROUPED_POOLED_LIFECYCLE_DIAGNOSTIC_COMPLETE",
        "contracts": {
            "targets": "H12 exact before/after global-book dependencies; derived lifecycle availability is the maximum over every active/onset label dependency",
            "folding": "five-fold shuffled UTC seven-day StratifiedGroupKFold with a two-sided 36h embargo; no walk-forward requirement",
            "inputs": "decision-time common context only; source family, economics tier, provenance and calendar are never model features",
            "domains": "source flags retained for reporting; inverse-source-frequency sample weights balance source training mass and are also used by calibration",
            "calibration": "nested grouped/36h-purged source-balanced OOF Brier chooses shrinkage toward the source-balanced train prevalence",
            "reversal": "reversal requires 3 active anchors, a 4-anchor inactive recovery including the current anchor, then a fresh active state within the next 3 anchors; inadequate support is reported as skipped",
            "promotion": "research-only lifecycle diagnosis; no admission, timing, portfolio, policy or production-score change",
        },
        "feature_count": int(predictions["feature_count"].iloc[0]) if not predictions.empty else 0,
        "metric_rows": int(len(metrics)), "prediction_rows": int(len(predictions)), "skipped_rows": int(len(skipped)),
        "source": {"panel": str(panel_path), "panel_sha256": sha256(panel_path), "manifest": str(manifest_path), "manifest_sha256": sha256(manifest_path)},
        "outputs_sha256": {name: sha256(path) for name, path in paths.items()},
    }
    _write_json(temporary / "manifest.json", result)
    (temporary / "manifest.sha256").write_text(f"{sha256(temporary / 'manifest.json')}  manifest.json\n")
    os.replace(temporary, output)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(_safe(run(parse_args())), indent=2, sort_keys=True))
