#!/usr/bin/env python3
"""Run grouped, non-walk-forward cross-era transition-classifier ablations."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PANEL = ROOT / (
    "data_perp/artifacts/cross_era_global_book_transition_research_panel_"
    "20260730_v4"
)
DEFAULT_OUTPUT = ROOT / (
    "data_perp/artifacts/cross_era_regime_transition_classifier_ablation_"
    "20260730_v4"
)
SCHEMA = "cross_era_regime_transition_classifier_ablation_v4"
RANDOM_STATE = 20260729
ADVERSE_SENSITIVITY_BPS = (50, 75, 100)
ADVERSE_SENSITIVITY_TARGETS = tuple(
    target
    for threshold_bps in ADVERSE_SENSITIVITY_BPS
    for target in (
        f"target__active_adverse_sensitivity_{threshold_bps}bps",
        f"target__adverse_onset_within_3h_sensitivity_{threshold_bps}bps",
    )
)
CONDITIONAL_MECHANISM_TARGETS = (
    "target__mechanism_upside_collapse",
    "target__mechanism_loss_expansion",
)
ECONOMIC_TARGETS = (*ADVERSE_SENSITIVITY_TARGETS, *CONDITIONAL_MECHANISM_TARGETS)
# The sensitivity screen deliberately uses a bounded, reproducible model grid.
# Nested shrinkage remains implemented for a later calibration-specific study,
# rather than multiplying this label/mechanism ablation by its inner CV cost.
DEFAULT_MODEL_ARMS = ("prior", "logistic", "extra_trees")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe(value: Any) -> Any:
    if isinstance(value, (Path, pd.Timestamp, pd.Timedelta)):
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


def feature_sets(columns: Sequence[str]) -> dict[str, list[str]]:
    columns = list(columns)
    coordinates = [
        column for column in columns if column.startswith("context__mapping_")
    ]
    raw_state = [
        column
        for column in columns
        if column.startswith(("context__state_", "context__past_"))
    ]
    past = [column for column in columns if column.startswith("context__past_")]
    result = {
        "coordinates_only": coordinates,
        "raw_state_only": raw_state,
        "past_transitions_only": past,
        "coordinates_plus_raw_state": [*coordinates, *raw_state],
    }
    return {
        name: list(dict.fromkeys(values))
        for name, values in result.items()
        if values
    }


def _model(name: str, columns: Sequence[str]) -> Pipeline:
    name = name.removesuffix("_shrunk")
    if name == "logistic":
        estimator = LogisticRegression(
            C=0.5,
            penalty="l2",
            solver="lbfgs",
            max_iter=2_000,
            random_state=RANDOM_STATE,
        )
        transform = Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ]
        )
    elif name == "extra_trees":
        estimator = ExtraTreesClassifier(
            n_estimators=200,
            max_depth=6,
            min_samples_leaf=6,
            max_features="sqrt",
            class_weight="balanced",
            n_jobs=1,
            random_state=RANDOM_STATE,
        )
        transform = Pipeline([("impute", SimpleImputer(strategy="median"))])
    else:
        raise ValueError(f"unknown transition model: {name}")
    preprocessor = ColumnTransformer(
        [("numeric", transform, list(columns))],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return Pipeline([("features", preprocessor), ("model", estimator)])


def _nested_shrunk_prediction(
    train: pd.DataFrame,
    train_y: pd.Series,
    validation: pd.DataFrame,
    *,
    columns: Sequence[str],
    base_model_name: str,
) -> tuple[np.ndarray, float]:
    groups = train["cv_group_id"].astype(str).reset_index(drop=True)
    local_y = train_y.reset_index(drop=True)
    local = train.reset_index(drop=True)
    folds = min(3, groups.nunique(), int(local_y.sum()), int((1 - local_y).sum()))
    if folds < 3:
        raise ValueError("insufficient inner grouped support for shrinkage")
    splitter = StratifiedGroupKFold(
        n_splits=folds, shuffle=True, random_state=RANDOM_STATE + 17
    )
    inner_prediction = np.full(len(local), np.nan)
    for inner_train, inner_validation in splitter.split(
        local, local_y, groups
    ):
        inner_train = _purge_near_validation(
            inner_train,
            inner_validation,
            pd.to_datetime(local["cohort_anchor_utc"], utc=True),
            embargo_hours=36,
        )
        if len(inner_train) < 40 or local_y.iloc[inner_train].nunique() < 2:
            raise ValueError("inner 36h embargo leaves insufficient support")
        model = _model(base_model_name, columns)
        model.fit(local.iloc[inner_train], local_y.iloc[inner_train])
        inner_prediction[inner_validation] = model.predict_proba(
            local.iloc[inner_validation]
        )[:, 1]
    if not np.isfinite(inner_prediction).all():
        raise ValueError("inner shrinkage OOF coverage is incomplete")
    prior = float(local_y.mean())
    weights = np.asarray((0.0, 0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.0))
    losses = [
        brier_score_loss(
            local_y,
            np.clip(
                prior + weight * (inner_prediction - prior), 1e-8, 1 - 1e-8
            ),
        )
        for weight in weights
    ]
    selected_weight = float(weights[int(np.argmin(losses))])
    final_model = _model(base_model_name, columns)
    final_model.fit(local, local_y)
    raw = final_model.predict_proba(validation)[:, 1]
    prediction = prior + selected_weight * (raw - prior)
    return np.clip(prediction, 1e-8, 1 - 1e-8), selected_weight


def _metric_record(
    frame: pd.DataFrame,
    *,
    setup: str,
    horizon: int,
    target: str,
    feature_set: str,
    model: str,
    scope: str,
) -> dict[str, Any]:
    y = frame["target"].to_numpy(float)
    prediction = np.clip(frame["prediction"].to_numpy(float), 1e-8, 1 - 1e-8)
    selected = frame["selected_top10"].astype(bool).to_numpy()
    positives = int(y.sum())
    result = {
        "setup": setup,
        "horizon_hours": int(horizon),
        "target": target,
        "feature_set": feature_set,
        "model": model,
        "scope": scope,
        "rows": int(len(frame)),
        "positive_rows": positives,
        "prevalence": float(y.mean()),
        "roc_auc": (
            float(roc_auc_score(y, prediction))
            if len(np.unique(y)) == 2
            else float("nan")
        ),
        "average_precision": (
            float(average_precision_score(y, prediction))
            if positives
            else float("nan")
        ),
        "brier": float(brier_score_loss(y, prediction)),
        "log_loss": float(log_loss(y, prediction, labels=[0.0, 1.0])),
        "top10_selected_rows": int(selected.sum()),
        "top10_precision": (
            float(y[selected].mean()) if selected.any() else float("nan")
        ),
        "top10_recall": (
            float(y[selected].sum() / positives)
            if positives and selected.any()
            else float("nan")
        ),
    }
    result["top10_lift"] = (
        result["top10_precision"] / result["prevalence"]
        if result["prevalence"] > 0 and np.isfinite(result["top10_precision"])
        else float("nan")
    )
    if "actual_onset" in frame:
        ordered = frame.sort_values("cohort_anchor_utc")
        onset_times = ordered.loc[
            ordered["actual_onset"].eq(1.0), "cohort_anchor_utc"
        ]
        alert_times = ordered.loc[
            ordered["selected_top10"], "cohort_anchor_utc"
        ]
        caught = 0
        leads: list[float] = []
        for onset in onset_times:
            local = alert_times[
                alert_times.between(
                    onset - pd.Timedelta(hours=2), onset, inclusive="both"
                )
            ]
            if len(local):
                caught += 1
                leads.append(
                    float((onset - local.min()) / pd.Timedelta(hours=1))
                )
        false_alerts = 0
        for alert in alert_times:
            if not onset_times.between(
                alert, alert + pd.Timedelta(hours=2), inclusive="both"
            ).any():
                false_alerts += 1
        duration_days = max(
            float(
                (
                    ordered["cohort_anchor_utc"].max()
                    - ordered["cohort_anchor_utc"].min()
                )
                / pd.Timedelta(days=1)
            ),
            1.0,
        )
        result.update(
            {
                "onset_events": int(len(onset_times)),
                "event_recall_at_top10": (
                    caught / len(onset_times)
                    if len(onset_times)
                    else float("nan")
                ),
                "median_lead_hours_at_top10": (
                    float(np.median(leads)) if leads else float("nan")
                ),
                "false_alerts_per_30d_at_top10": float(
                    false_alerts * 30.0 / duration_days
                ),
            }
        )
    return result


def _purge_near_validation(
    train_index: np.ndarray,
    validation_index: np.ndarray,
    timestamps: pd.Series,
    *,
    embargo_hours: int,
) -> np.ndarray:
    train_times = timestamps.iloc[train_index].astype("int64").to_numpy()
    validation_times = np.sort(
        timestamps.iloc[validation_index].astype("int64").to_numpy()
    )
    positions = np.searchsorted(validation_times, train_times)
    maximum = np.iinfo(np.int64).max
    left = np.where(
        positions > 0,
        validation_times[np.maximum(positions - 1, 0)],
        maximum,
    )
    right = np.where(
        positions < len(validation_times),
        validation_times[np.minimum(positions, len(validation_times) - 1)],
        maximum,
    )
    distance = np.minimum(
        np.abs(train_times - left), np.abs(train_times - right)
    )
    embargo_ns = int(pd.Timedelta(hours=embargo_hours).value)
    return train_index[distance > embargo_ns]


def grouped_oof_predictions(
    frame: pd.DataFrame,
    *,
    columns: Sequence[str],
    target: str,
    model_name: str,
    n_splits: int = 5,
) -> pd.DataFrame:
    work = frame.copy().reset_index(drop=True)
    y = pd.to_numeric(work[target], errors="coerce")
    availability_column = f"{target}_available_utc"
    if availability_column in work:
        availability = pd.to_datetime(
            work[availability_column], utc=True, errors="coerce"
        )
        valid = y.notna() & availability.notna()
    else:
        # This catches an accidental attempt to fit a derived target without
        # its materialized availability lineage.
        if target.startswith("target__"):
            raise ValueError(f"target lacks exact availability column: {target}")
        availability = pd.Series(pd.NaT, index=work.index, dtype="datetime64[ns, UTC]")
        valid = y.notna()
    work = work.loc[valid].reset_index(drop=True)
    y = y.loc[valid].astype(int).reset_index(drop=True)
    availability = availability.loc[valid].reset_index(drop=True)
    groups = work["cv_group_id"].astype(str)
    unique_groups = groups.nunique()
    positives = int(y.sum())
    negatives = int((1 - y).sum())
    folds = min(int(n_splits), int(unique_groups), positives, negatives)
    if len(work) < 60 or folds < 3:
        raise ValueError("insufficient grouped binary support")
    splitter = StratifiedGroupKFold(
        n_splits=folds, shuffle=True, random_state=RANDOM_STATE
    )
    prediction = np.full(len(work), np.nan)
    fold_id = np.full(len(work), -1, dtype=int)
    selected = np.zeros(len(work), dtype=bool)
    calibration_weight = np.full(len(work), np.nan)
    for fold, (train_index, validation_index) in enumerate(
        splitter.split(work, y, groups)
    ):
        train_index = _purge_near_validation(
            train_index,
            validation_index,
            pd.to_datetime(work["cohort_anchor_utc"], utc=True),
            embargo_hours=36,
        )
        if len(train_index) < 40:
            raise ValueError("36h embargo leaves insufficient training rows")
        train_y = y.iloc[train_index]
        if train_y.nunique() < 2:
            raise ValueError("grouped training fold has one target class")
        if model_name == "prior":
            local_prediction = np.full(
                len(validation_index), float(train_y.mean())
            )
            local_weight = 0.0
        elif model_name.endswith("_shrunk"):
            local_prediction, local_weight = _nested_shrunk_prediction(
                work.iloc[train_index],
                train_y,
                work.iloc[validation_index],
                columns=columns,
                base_model_name=model_name.removesuffix("_shrunk"),
            )
        else:
            model = _model(model_name, columns)
            model.fit(work.iloc[train_index], train_y)
            local_prediction = model.predict_proba(
                work.iloc[validation_index]
            )[:, 1]
            local_weight = 1.0
        prediction[validation_index] = local_prediction
        calibration_weight[validation_index] = local_weight
        fold_id[validation_index] = fold
        count = max(1, int(math.ceil(0.10 * len(validation_index))))
        local_order = np.lexsort(
            (
                work.iloc[validation_index]["cohort_anchor_utc"]
                .astype("int64")
                .to_numpy(),
                -local_prediction,
            )
        )
        selected[validation_index[local_order[:count]]] = True
    if not np.isfinite(prediction).all() or (fold_id < 0).any():
        raise ValueError("grouped OOF prediction coverage is incomplete")
    result = work.loc[
        :,
        [
            "cohort_anchor_utc",
            "horizon_hours",
            "source_family",
            "economics_tier",
            "mapping_provenance_role",
            "cv_group_id",
        ],
    ].copy()
    result["target_available_utc"] = availability
    result["target_name"] = target
    result["target"] = y.to_numpy(float)
    result["prediction"] = prediction
    result["cv_fold"] = fold_id
    result["selected_top10"] = selected
    result["calibration_shrinkage_weight"] = calibration_weight
    onset_target = target.replace("_within_3h", "")
    if (
        target.startswith("target__adverse_onset_within_3h")
        and onset_target in work
    ):
        result["actual_onset"] = pd.to_numeric(
            work[onset_target], errors="coerce"
        ).to_numpy(float)
    return result


def run_ablation(
    panel: pd.DataFrame,
    feature_columns: Sequence[str],
    *,
    model_arms: Sequence[str] = DEFAULT_MODEL_ARMS,
    targets_filter: Sequence[str] | None = None,
    feature_sets_filter: Sequence[str] | None = None,
    setups_filter: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    work = panel.loc[
        panel["context_available"].astype(bool)
        & panel["horizon_hours"].eq(12)
    ].copy()
    families = feature_sets(feature_columns)
    if feature_sets_filter is not None:
        requested = set(feature_sets_filter)
        unknown = sorted(requested.difference(families))
        if unknown:
            raise ValueError(f"unknown feature-set filter: {unknown}")
        families = {name: columns for name, columns in families.items() if name in requested}
    if not families:
        raise ValueError("no feature family remains after filtering")
    allowed_targets = set(targets_filter) if targets_filter is not None else None
    setups: list[tuple[str, pd.DataFrame, tuple[str, ...]]] = [
        (
            "reconstructed_fee_only",
            work.loc[
                work["source_family"].eq(
                    "reconstructed_exact1m_janapr2025"
                )
            ].copy(),
            ECONOMIC_TARGETS,
        ),
        (
            "canonical_spread",
            work.loc[
                work["source_family"].eq("canonical_spread_febapr2025")
            ].copy(),
            ECONOMIC_TARGETS,
        ),
        (
            "current_exact_spread",
            work.loc[
                work["source_family"].eq(
                    "current_exact_spread_mayjul2026"
                )
            ].copy(),
            ECONOMIC_TARGETS,
        ),
        (
            "spread_aware_combined",
            work.loc[
                work["source_family"].isin(
                    (
                        "canonical_spread_febapr2025",
                        "current_exact_spread_mayjul2026",
                    )
                )
            ].copy(),
            ECONOMIC_TARGETS,
        ),
    ]
    if setups_filter is not None:
        requested = set(setups_filter)
        known = {name for name, _, _ in setups}
        unknown = sorted(requested.difference(known))
        if unknown:
            raise ValueError(f"unknown setup filter: {unknown}")
        setups = [setup for setup in setups if setup[0] in requested]
    if not setups:
        raise ValueError("no source setup remains after filtering")
    metric_rows: list[dict[str, Any]] = []
    predictions: list[pd.DataFrame] = []
    skipped: list[dict[str, Any]] = []
    for setup_name, setup_frame, targets in setups:
        if allowed_targets is not None:
            unknown = sorted(allowed_targets.difference(targets))
            if unknown:
                raise ValueError(f"unknown target filter: {unknown}")
            targets = tuple(target for target in targets if target in allowed_targets)
        for horizon, horizon_frame in setup_frame.groupby(
            "horizon_hours", sort=True
        ):
            for target in targets:
                for feature_name, columns in families.items():
                    usable = [
                        column
                        for column in columns
                        if horizon_frame[column].notna().sum() >= max(
                            20, int(0.50 * len(horizon_frame))
                        )
                        and pd.to_numeric(
                            horizon_frame[column], errors="coerce"
                        ).nunique(dropna=True)
                        > 1
                    ]
                    if not usable:
                        skipped.append(
                            {
                                "setup": setup_name,
                                "horizon_hours": int(horizon),
                                "target": target,
                                "feature_set": feature_name,
                                "reason": "no_usable_features",
                            }
                        )
                        continue
                    for model_name in model_arms:
                        try:
                            prediction = grouped_oof_predictions(
                                horizon_frame,
                                columns=usable,
                                target=target,
                                model_name=model_name,
                            )
                        except ValueError as error:
                            skipped.append(
                                {
                                    "setup": setup_name,
                                    "horizon_hours": int(horizon),
                                    "target": target,
                                    "feature_set": feature_name,
                                    "model": model_name,
                                    "reason": str(error),
                                }
                            )
                            continue
                        prediction["setup"] = setup_name
                        prediction["feature_set"] = feature_name
                        prediction["model"] = model_name
                        prediction["feature_count"] = len(usable)
                        predictions.append(prediction)
                        metric_rows.append(
                            _metric_record(
                                prediction,
                                setup=setup_name,
                                horizon=int(horizon),
                                target=target,
                                feature_set=feature_name,
                                model=model_name,
                                scope="all",
                            )
                        )
                        if setup_name == "spread_aware_combined":
                            for source, local in prediction.groupby(
                                "source_family", sort=True
                            ):
                                metric_rows.append(
                                    _metric_record(
                                        local,
                                        setup=setup_name,
                                        horizon=int(horizon),
                                        target=target,
                                        feature_set=feature_name,
                                        model=model_name,
                                        scope=f"source::{source}",
                                    )
                                )
                        if setup_name == "current_exact_spread":
                            for provenance, local in prediction.groupby(
                                "mapping_provenance_role", sort=True
                            ):
                                if len(local) < 20:
                                    continue
                                metric_rows.append(
                                    _metric_record(
                                        local,
                                        setup=setup_name,
                                        horizon=int(horizon),
                                        target=target,
                                        feature_set=feature_name,
                                        model=model_name,
                                        scope=f"provenance::{provenance}",
                                    )
                                )
    if not predictions:
        raise ValueError("no transition-classifier arm was evaluable")
    return (
        pd.DataFrame(metric_rows),
        pd.concat(predictions, ignore_index=True),
        pd.DataFrame(skipped),
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    panel_root = Path(args.panel)
    panel_path = panel_root / "transition_research_panel.parquet"
    catalog_path = panel_root / "field_catalog.csv"
    manifest_path = panel_root / "manifest.json"
    sidecar = panel_root / "manifest.sha256"
    if not all(
        path.is_file()
        for path in (panel_path, catalog_path, manifest_path, sidecar)
    ):
        raise FileNotFoundError("cross-era transition panel is incomplete")
    if sidecar.read_text(encoding="utf-8").split()[0] != sha256(manifest_path):
        raise ValueError("cross-era transition panel manifest checksum fails")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    feature_columns = list(manifest["feature_columns"])
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite immutable output {output}")
    metrics, predictions, skipped = run_ablation(
        pd.read_parquet(panel_path),
        feature_columns,
        model_arms=args.model_arms,
        targets_filter=args.targets or None,
        feature_sets_filter=args.feature_sets or None,
        setups_filter=args.setups or None,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(dir=output.parent, prefix=f".{output.name}.")
    )
    paths = {
        "metrics": temporary / "metrics.csv",
        "predictions": temporary / "grouped_oof_predictions.parquet",
        "skipped": temporary / "skipped_arms.csv",
    }
    metrics.to_csv(paths["metrics"], index=False)
    predictions.to_parquet(
        paths["predictions"], index=False, compression="zstd"
    )
    skipped.to_csv(paths["skipped"], index=False)
    model_metrics = metrics.loc[
        metrics["model"].ne("prior") & metrics["scope"].eq("all")
    ].copy()
    leaders = (
        model_metrics.sort_values(
            ["setup", "horizon_hours", "target", "average_precision"],
            ascending=[True, True, True, False],
        )
        .groupby(["setup", "horizon_hours", "target"], sort=False)
        .head(1)
    )
    result_manifest = {
        "schema": SCHEMA,
        "status": "GROUPED_NON_WALK_FORWARD_DIAGNOSTIC_COMPLETE",
        "contracts": {
            "rows": "primary H12 complete-window rows; active and onset labels use exact hourly persistence dependencies",
            "targets": "predeclared 50/75/100-bps active/onset sensitivity arms plus conditional-on-active upside-collapse and loss-expansion mechanism arms",
            "folding": "five-fold shuffled StratifiedGroupKFold over UTC seven-day groups plus a 36h two-sided train embargo around every held-out anchor; no walk-forward requirement",
            "feature_preprocessing": "median imputation and scaling fit inside each grouped training fold",
            "probability_shrinkage": "shrunk variants select one of {0,.05,.10,.20,.35,.50,.75,1} against inner three-fold grouped/36h-purged OOF Brier, then blend the outer prediction toward the outer-training prevalence",
            "model_arms": "bounded sensitivity screen: prior, logistic and single-process ExtraTrees; nested shrunk variants are reserved for a calibration-specific follow-up",
            "filters": "optional target, feature-family and source-setup filters retain the same source-separated, 36h-purged grouped-CV contract and permit bounded reproducible shards",
            "feature_selection": "predeclared feature families only; constant/low-coverage removal inside each setup, no target-based selection",
            "global_book": "labels come from one pooled global top10 after causal recent EV mapping, never per timestamp or side",
            "source_separation": "fee-only, canonical spread-aware and current spread-aware economics reported separately; combined spread arm also reports each source",
            "current_provenance": "current strict mapped OOF and frozen nonpromotable forward-OOS rows are reported separately",
            "promotion": "research diagnosis only; no portfolio replay or policy change",
        },
        "feature_sets": {
            key: value for key, value in feature_sets(feature_columns).items()
        },
        "metric_rows": int(len(metrics)),
        "prediction_rows": int(len(predictions)),
        "skipped_rows": int(len(skipped)),
        "leaders": leaders.to_dict(orient="records"),
        "source": {
            "panel": str(panel_path),
            "panel_sha256": sha256(panel_path),
            "manifest": str(manifest_path),
            "manifest_sha256": sha256(manifest_path),
        },
        "outputs": {
            name: {"path": path.name, "sha256": sha256(path)}
            for name, path in paths.items()
        },
    }
    (temporary / "manifest.json").write_text(
        json.dumps(_safe(result_manifest), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (temporary / "manifest.sha256").write_text(
        f"{sha256(temporary / 'manifest.json')}  manifest.json\n",
        encoding="utf-8",
    )
    os.replace(temporary, output)
    return {
        "output": str(output),
        "metric_rows": int(len(metrics)),
        "prediction_rows": int(len(predictions)),
        "skipped_rows": int(len(skipped)),
        "leaders": int(len(leaders)),
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    result.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    result.add_argument(
        "--targets",
        nargs="+",
        choices=ECONOMIC_TARGETS,
        help="optional declared targets for a bounded immutable ablation shard",
    )
    result.add_argument(
        "--feature-sets",
        nargs="+",
        choices=(
            "coordinates_only",
            "raw_state_only",
            "past_transitions_only",
            "coordinates_plus_raw_state",
        ),
        help="optional declared decision-time feature families for a shard",
    )
    result.add_argument(
        "--model-arms",
        nargs="+",
        choices=(
            "prior",
            "logistic",
            "logistic_shrunk",
            "extra_trees",
            "extra_trees_shrunk",
        ),
        default=list(DEFAULT_MODEL_ARMS),
        help="bounded model arms; default excludes nested shrinkage",
    )
    result.add_argument(
        "--setups",
        nargs="+",
        choices=(
            "reconstructed_fee_only",
            "canonical_spread",
            "current_exact_spread",
            "spread_aware_combined",
        ),
        help="optional source-separated setup(s) for a bounded shard",
    )
    return result


def main() -> None:
    print(json.dumps(_safe(run(parser().parse_args())), sort_keys=True))


if __name__ == "__main__":
    main()
