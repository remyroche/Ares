#!/usr/bin/env python3
"""Measure whether causal context identifies the break and direct-head trust.

The two targets are deliberately separate:
1. March 20 period membership, which asks whether the market state changed;
2. direct-minus-residual hourly book contribution, which asks whether that
   state is economically useful for routing.

Both are reused-month, grouped-OOF diagnostics.  No fitted output is a gate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
INPUT = (
    ROOT / "data_perp/artifacts/"
    "marapr2025_direct_residual_regime_trust_diagnostic_20260730_v1"
)
OUTPUT = (
    ROOT / "data_perp/artifacts/"
    "marapr2025_direct_residual_regime_break_learnability_20260730_v1"
)
SCHEMA = "marapr2025_direct_residual_regime_break_learnability_v1"
INPUT_SCHEMA = "marapr2025_direct_residual_regime_trust_diagnostic_v1"
NET = "execution_net_ev_12h"

REGIME_FIELDS = (
    "bocpd__change_probability_mean",
    "bocpd__change_probability_max",
    "bocpd__run_length_mean",
    "bocpd__run_length_q05",
    "bocpd__run_length_entropy",
    "bocpd__signal_count",
    "bocpd__state_age_hours",
    "bocpd__is_persistent_24h",
    "bocpd__is_persistent_72h",
)
TRANSITION_FIELDS = (
    "lgbm_transition_probability",
    "lgbm_entropy",
    "lgbm_margin",
    "bocpd_stable_vs_transition_probability",
    "bocpd_onset_h1_probability",
    "bocpd_onset_h3_probability",
    "bocpd_onset_h6_probability",
    "bocpd_onset_h12_probability",
)
TRAJECTORY_FIELDS = (
    "trajectory_transition_probability",
    "trajectory_probability_entropy",
    "trajectory_top2_margin",
)
ARMS: Mapping[str, tuple[str, ...]] = {
    "regime9": REGIME_FIELDS,
    "transition8": TRANSITION_FIELDS,
    "trajectory3": TRAJECTORY_FIELDS,
    "combined20": (*REGIME_FIELDS, *TRANSITION_FIELDS, *TRAJECTORY_FIELDS),
}


class LearnabilityError(RuntimeError):
    """Raised when a diagnostic or grouped-OOF invariant fails."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(item) for item in value]
    if isinstance(value, (Path, pd.Timestamp)):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(safe(payload), indent=2, sort_keys=True) + "\n")


def verify_input(root: Path) -> dict[str, Any]:
    manifest_path = root / "manifest.json"
    seal_path = root / "manifest.sha256"
    if not manifest_path.is_file() or not seal_path.is_file():
        raise LearnabilityError(f"sealed input required: {root}")
    if sha256(manifest_path) != seal_path.read_text().split()[0]:
        raise LearnabilityError("input manifest seal mismatch")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != INPUT_SCHEMA:
        raise LearnabilityError("input schema mismatch")
    for name, digest in manifest.get("outputs_sha256", {}).items():
        if sha256(root / name) != digest:
            raise LearnabilityError(f"input output hash mismatch: {name}")
    if manifest.get("promotion_eligible") is not False:
        raise LearnabilityError("diagnostic input must be non-promotable")
    return manifest


def week_group(timestamp: pd.Series) -> pd.Series:
    values = pd.to_datetime(timestamp, utc=True, errors="raise")
    nanoseconds = values.astype("int64")
    return (nanoseconds // int(pd.Timedelta(days=7).value)).astype("int64")


def day_group(timestamp: pd.Series) -> pd.Series:
    values = pd.to_datetime(timestamp, utc=True, errors="raise")
    nanoseconds = values.astype("int64")
    return (nanoseconds // int(pd.Timedelta(days=1).value)).astype("int64")


def build_hour_side_panel(
    candidates: pd.DataFrame,
    selected_books: pd.DataFrame,
) -> pd.DataFrame:
    context = list(ARMS["combined20"])
    grouped = candidates.groupby(
        ["__ts__", "side_name", "diagnostic_period"], sort=True, observed=True
    )
    for field in context:
        if grouped[field].nunique(dropna=False).max() != 1:
            raise LearnabilityError(f"context is not hourly-side invariant: {field}")
    hourly = grouped[context].first().reset_index()
    selected = (
        selected_books.groupby(
            ["diagnostic_period", "selection_source", "__ts__", "side_name"],
            sort=True,
            observed=True,
        )
        .agg(selected_rows=("candidate_id", "size"), selected_net_sum=(NET, "sum"))
        .reset_index()
    )
    book_sizes = (
        selected_books.groupby(
            ["diagnostic_period", "selection_source"], observed=True
        )
        .size()
        .rename("book_rows")
        .reset_index()
    )
    selected = selected.merge(
        book_sizes,
        on=["diagnostic_period", "selection_source"],
        how="left",
        validate="many_to_one",
    )
    selected["contribution_bps"] = (
        selected["selected_net_sum"] / selected["book_rows"] * 1e4
    )
    for source in ("direct_q25", "residual"):
        part = selected.loc[
            selected["selection_source"].eq(source),
            [
                "diagnostic_period",
                "__ts__",
                "side_name",
                "selected_rows",
                "contribution_bps",
            ],
        ].rename(
            columns={
                "selected_rows": f"{source}_selected_rows",
                "contribution_bps": f"{source}_contribution_bps",
            }
        )
        hourly = hourly.merge(
            part,
            on=["diagnostic_period", "__ts__", "side_name"],
            how="left",
            validate="one_to_one",
        )
    contribution_fields = [
        "direct_q25_selected_rows",
        "direct_q25_contribution_bps",
        "residual_selected_rows",
        "residual_contribution_bps",
    ]
    hourly[contribution_fields] = hourly[contribution_fields].fillna(0.0)
    hourly["either_source_selected"] = (
        hourly["direct_q25_selected_rows"].gt(0)
        | hourly["residual_selected_rows"].gt(0)
    )
    hourly["direct_advantage_bps"] = (
        hourly["direct_q25_contribution_bps"]
        - hourly["residual_contribution_bps"]
    )
    hourly["direct_advantage_positive"] = hourly["direct_advantage_bps"].gt(0).astype(int)
    hourly["post_march20"] = hourly["diagnostic_period"].eq("march20_31").astype(int)
    hourly["cv_day_group"] = day_group(hourly["__ts__"])
    hourly["cv_week_group"] = week_group(hourly["__ts__"])
    return hourly


def _auc(y: np.ndarray, probability: np.ndarray) -> float:
    return float(roc_auc_score(y, probability)) if np.unique(y).size == 2 else np.nan


def grouped_oof(
    frame: pd.DataFrame,
    features: Sequence[str],
    target: str,
    *,
    n_splits: int = 5,
    seed: int = 20260730,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    x = frame.loc[:, list(features)].to_numpy(dtype=float)
    y = pd.to_numeric(frame[target], errors="raise").to_numpy(dtype=int)
    groups = frame["cv_group"].to_numpy()
    if not np.isfinite(x).all() or np.unique(y).size != 2:
        raise LearnabilityError("OOF matrix is non-finite or target has one class")
    splitter = StratifiedGroupKFold(
        n_splits=min(n_splits, np.unique(groups).size),
        shuffle=True,
        random_state=seed,
    )
    probability = np.full(len(frame), np.nan, dtype=float)
    fold_id = np.full(len(frame), -1, dtype=int)
    fold_rows: list[dict[str, Any]] = []
    for fold, (train, valid) in enumerate(splitter.split(x, y, groups)):
        if set(groups[train]) & set(groups[valid]):
            raise LearnabilityError("CV week group overlaps train and validation")
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=1.0,
                class_weight="balanced",
                max_iter=2_000,
                solver="lbfgs",
                random_state=seed + fold,
            ),
        )
        model.fit(x[train], y[train])
        probability[valid] = model.predict_proba(x[valid])[:, 1]
        fold_id[valid] = fold
        fold_rows.append(
            {
                "fold": fold,
                "train_rows": len(train),
                "validation_rows": len(valid),
                "train_groups": len(np.unique(groups[train])),
                "validation_groups": len(np.unique(groups[valid])),
                "train_positive_rate": float(y[train].mean()),
                "validation_positive_rate": float(y[valid].mean()),
                "auc": _auc(y[valid], probability[valid]),
                "average_precision": (
                    float(average_precision_score(y[valid], probability[valid]))
                    if np.unique(y[valid]).size == 2
                    else np.nan
                ),
            }
        )
    if np.isnan(probability).any() or (fold_id < 0).any():
        raise LearnabilityError("grouped OOF did not cover every row")
    predictions = frame[
        [
            "__ts__",
            "side_name",
            "diagnostic_period",
            "cv_group",
            target,
            "direct_advantage_bps",
        ]
    ].copy()
    predictions["fold"] = fold_id
    predictions["probability"] = probability

    full_model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=2_000,
            solver="lbfgs",
            random_state=seed,
        ),
    )
    full_model.fit(x, y)
    coefficients = pd.DataFrame(
        {
            "feature": list(features),
            "standardized_coefficient": full_model[-1].coef_[0],
        }
    )
    coefficients["absolute_coefficient"] = coefficients["standardized_coefficient"].abs()
    return predictions, pd.DataFrame(fold_rows), coefficients


def aggregate_metrics(
    predictions: pd.DataFrame,
    *,
    task: str,
    arm: str,
    target: str,
) -> dict[str, Any]:
    y = predictions[target].to_numpy(int)
    probability = predictions["probability"].to_numpy(float)
    result = {
        "task": task,
        "arm": arm,
        "rows": len(predictions),
        "positive_rate": float(y.mean()),
        "grouped_oof_auc": _auc(y, probability),
        "grouped_oof_average_precision": float(
            average_precision_score(y, probability)
        ),
        "grouped_oof_brier": float(brier_score_loss(y, probability)),
        "grouped_oof_balanced_accuracy_at_0_5": float(
            balanced_accuracy_score(y, probability >= 0.5)
        ),
    }
    if task.startswith("direct_trust_"):
        result["probability_rank_ic_direct_advantage"] = float(
            pd.Series(probability).corr(
                predictions["direct_advantage_bps"].reset_index(drop=True),
                method="spearman",
            )
        )
        threshold = float(np.quantile(probability, 0.90))
        result["top_probability_decile_advantage_bps"] = float(
            predictions.loc[probability >= threshold, "direct_advantage_bps"].mean()
        )
    else:
        result["probability_rank_ic_direct_advantage"] = np.nan
        result["top_probability_decile_advantage_bps"] = np.nan
    return result


def run(input_root: Path = INPUT, output: Path = OUTPUT) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite sealed output: {output}")
    input_manifest = verify_input(input_root)
    candidates = pd.read_parquet(input_root / "candidate_panel.parquet")
    selected = pd.read_parquet(input_root / "selected_books.parquet")
    candidates["__ts__"] = pd.to_datetime(candidates["__ts__"], utc=True)
    selected["__ts__"] = pd.to_datetime(selected["__ts__"], utc=True)
    panel = build_hour_side_panel(candidates, selected)

    predictions: list[pd.DataFrame] = []
    fold_metrics: list[pd.DataFrame] = []
    coefficients: list[pd.DataFrame] = []
    metrics: list[dict[str, Any]] = []
    break_rows = (
        panel.loc[
            panel["diagnostic_period"].isin(("march03_19", "march20_31"))
        ]
        .sort_values(["__ts__", "side_name"], kind="stable")
        .drop_duplicates("__ts__", keep="first")
        .copy()
    )
    break_rows["side_name"] = "global"
    tasks = {
        "break_recognition": break_rows,
        "direct_trust_long": panel.loc[
            panel["either_source_selected"] & panel["side_name"].eq("long")
        ].copy(),
        "direct_trust_short": panel.loc[
            panel["either_source_selected"] & panel["side_name"].eq("short")
        ].copy(),
    }
    tasks["break_recognition"]["cv_group"] = tasks["break_recognition"][
        "cv_day_group"
    ]
    tasks["direct_trust_long"]["cv_group"] = tasks["direct_trust_long"][
        "cv_week_group"
    ]
    tasks["direct_trust_short"]["cv_group"] = tasks["direct_trust_short"][
        "cv_week_group"
    ]
    targets = {
        "break_recognition": "post_march20",
        "direct_trust_long": "direct_advantage_positive",
        "direct_trust_short": "direct_advantage_positive",
    }
    for task, rows in tasks.items():
        target = targets[task]
        for arm, features in ARMS.items():
            pred, folds, coef = grouped_oof(rows, features, target)
            pred["task"] = task
            pred["arm"] = arm
            folds["task"] = task
            folds["arm"] = arm
            coef["task"] = task
            coef["arm"] = arm
            predictions.append(pred)
            fold_metrics.append(folds)
            coefficients.append(coef)
            metrics.append(
                aggregate_metrics(pred, task=task, arm=arm, target=target)
            )

    outputs = {
        "hour_side_panel.parquet": panel,
        "oof_predictions.parquet": pd.concat(predictions, ignore_index=True),
        "aggregate_metrics.parquet": pd.DataFrame(metrics),
        "fold_metrics.parquet": pd.concat(fold_metrics, ignore_index=True),
        "full_diagnostic_coefficients.parquet": pd.concat(
            coefficients, ignore_index=True
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        hashes: dict[str, str] = {}
        rows: dict[str, int] = {}
        for name, frame in outputs.items():
            path = stage / name
            frame.to_parquet(path, index=False, compression="zstd")
            hashes[name] = sha256(path)
            rows[name] = len(frame)
        manifest = {
            "schema": SCHEMA,
            "status": "SEALED_REUSED_MONTH_GROUPED_OOF_LEARNABILITY_DIAGNOSTIC_NO_GATE",
            "promotion_eligible": False,
            "portfolio_replay_authorized": False,
            "contract": {
                "tasks": {
                    "break_recognition": "March03-19 versus March20-31 from causal context only",
                    "direct_trust_long": "positive direct-minus-residual long hourly global-book contribution on rows where either source selects",
                    "direct_trust_short": "positive direct-minus-residual short hourly global-book contribution on rows where either source selects",
                },
                "validation": "fixed C=1 balanced logistic; shuffled stratified UTC-day-group OOF for boundary recognition and seven-day-group OOF for direct trust; no walk-forward requirement; no HPO/feature selection/threshold search",
                "feature_arms": {key: list(value) for key, value in ARMS.items()},
                "prohibited": "calendar/month/time, OOD, state/destination IDs, scores, outcomes, post-entry paths and action fields are not model inputs",
                "interpretation": "reused-month learnability only; even a strong result cannot select or authorize a trading gate",
            },
            "outputs_sha256": hashes,
            "output_rows": rows,
            "source": {
                "input_manifest_sha256": sha256(input_root / "manifest.json"),
                "input_identity_sha256": input_manifest["sources"]["bridge_identity_sha256"],
            },
            "runner": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256(Path(__file__).resolve()),
            },
        }
        write_json(stage / "manifest.json", manifest)
        (stage / "manifest.sha256").write_text(
            f"{sha256(stage / 'manifest.json')}  manifest.json\n"
        )
        os.replace(stage, output)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return manifest


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--input-root", type=Path, default=INPUT)
    result.add_argument("--output", type=Path, default=OUTPUT)
    return result


def main() -> None:
    args = parser().parse_args()
    print(json.dumps(safe(run(args.input_root, args.output)), indent=2))


if __name__ == "__main__":
    main()
