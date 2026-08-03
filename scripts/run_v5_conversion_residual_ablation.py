#!/usr/bin/env python3
"""Mapped-policy-aligned feature/target ablation for the v5 conversion layer."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import materialize_v5_conversion_residual_input as materialized
from scripts import run_short_winner_causal_recent_ev_mapping_v5 as mapping

SOURCE = ROOT / "data_perp/artifacts/v5_conversion_residual_input_20260730_v2"
DEFAULT_OUTPUT = ROOT / "data_perp/artifacts/v5_conversion_residual_ablation_20260730_v4"
TIME = "execution_decision_utc"
END = "execution_label_end_utc"
TARGET = "execution_net_ev_12h"
FRACTIONS = (0.01, 0.05, 0.10, 0.20)
MARCH_FOLDS = (
    ("march_23_25", pd.Timestamp("2025-03-23T00:00:00Z"), pd.Timestamp("2025-03-26T00:00:00Z")),
    ("march_26_28", pd.Timestamp("2025-03-26T00:00:00Z"), pd.Timestamp("2025-03-29T00:00:00Z")),
    ("march_29_31", pd.Timestamp("2025-03-29T00:00:00Z"), pd.Timestamp("2025-04-01T00:00:00Z")),
)
CALIBRATION_FOLD = (
    "march_20_22_mapping_calibration",
    pd.Timestamp("2025-03-20T00:00:00Z"),
    pd.Timestamp("2025-03-23T00:00:00Z"),
)
ALL_MARCH_FOLDS = (CALIBRATION_FOLD, *MARCH_FOLDS)
SELECTION_FOLD_NAMES = frozenset(fold[0] for fold in MARCH_FOLDS)
MODEL_SEEDS = {"classifier": 71, "gain": 73, "loss": 79, "residual": 83, "class_regression": 89}
MODEL_KWARGS = {
    "max_iter": 75,
    "max_leaf_nodes": 15,
    "min_samples_leaf": 30,
    "l2_regularization": 3.0,
}


class AblationError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(dict(payload), indent=2, sort_keys=True, default=str) + "\n")
    os.replace(temporary, path)


def verify_source(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = root / "manifest.json"
    seal = root / "manifest.sha256"
    if not manifest.is_file() or not seal.is_file():
        raise AblationError("sealed conversion input is missing")
    if seal.read_text().split()[0] != sha256(manifest):
        raise AblationError("conversion-input manifest seal mismatch")
    roles_path = root / "feature_roles.json"
    payload = json.loads(manifest.read_text())
    if payload.get("schema") != "v5_conversion_residual_input_v2":
        raise AblationError("wrong conversion-input source")
    if payload["outputs_sha256"]["feature_roles.json"] != sha256(roles_path):
        raise AblationError("feature roles hash mismatch")
    return payload, json.loads(roles_path.read_text())


def feature_groups(roles: Mapping[str, Any]) -> dict[str, list[str]]:
    scores = [
        "raw_score",
        "score_base_alpha",
        "score_residual_expected_ev",
        "direct_q25_return",
    ]
    peak_slope = [
        "pred_peak_mfe_12h_atr__p_hit",
        "pred_peak_mfe_12h_atr__conditional_mean",
        "pred_peak_mfe_12h_atr__expected",
        "pred_future_slope_atr_per_hour__diagnostic",
    ]
    levels = list(materialized.CORE_CONTEXT)
    transitions = list(materialized.CORE_TRANSITIONS)
    regimes = list(materialized.REGIME_CONTEXT)
    cohort = list(materialized.COHORT_CONTEXT)
    optional_mae = list(roles["optional_adverse_risk_ablation_only"])
    groups = {
        "scores": scores,
        "scores_peak_slope": [*scores, *peak_slope],
        "scores_peak_slope_levels": [*scores, *peak_slope, *levels],
        "scores_peak_slope_levels_transitions": [
            *scores,
            *peak_slope,
            *levels,
            *transitions,
        ],
        "scores_peak_slope_levels_regimes": [
            *scores,
            *peak_slope,
            *levels,
            *regimes,
            *cohort,
        ],
        "all_compact": [*scores, *peak_slope, *levels, *transitions, *regimes, *cohort],
        "all_compact_optional_mae": [
            *scores,
            *peak_slope,
            *levels,
            *transitions,
            *regimes,
            *cohort,
            *optional_mae,
        ],
    }
    baseline = set(roles["baseline_model_features"])
    optional = set(optional_mae)
    for name, columns in groups.items():
        if len(columns) != len(set(columns)):
            raise AblationError(f"duplicate feature in {name}")
        if not set(columns).issubset(baseline | optional):
            raise AblationError(f"feature group escapes approved contract: {name}")
    return groups


def competing_class(frame: pd.DataFrame) -> pd.Series:
    positive = frame[TARGET].gt(0)
    reason = frame.execution_exit_reason.astype(str)
    values = np.select(
        [
            reason.eq("trailing") & positive,
            reason.eq("timeout") & positive,
            reason.eq("trailing") & ~positive,
            reason.eq("timeout") & ~positive,
        ],
        [
            "trailing_positive",
            "timeout_positive",
            "trailing_nonpositive",
            "timeout_nonpositive",
        ],
        default="full_stop_or_adverse",
    )
    return pd.Series(values, index=frame.index, dtype="object")


def _tail_weight(train: pd.DataFrame) -> np.ndarray:
    cutoff = float(train.raw_score.quantile(0.70))
    return np.where(train.raw_score.ge(cutoff), 2.0, 1.0)


def _conditional_prediction(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    features: Sequence[str],
    mask: np.ndarray,
    values: np.ndarray,
    *,
    seed: int,
) -> np.ndarray:
    if int(mask.sum()) < 100:
        mean = float(np.mean(values[mask])) if mask.any() else 0.0
        return np.repeat(mean, len(valid))
    model = HistGradientBoostingRegressor(
        **MODEL_KWARGS, random_state=seed
    ).fit(train.loc[mask, list(features)], values[mask])
    return model.predict(valid.loc[:, list(features)])


def fit_predict(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    features: Sequence[str],
    architecture: str,
) -> np.ndarray:
    x_train = train.loc[:, list(features)]
    x_valid = valid.loc[:, list(features)]
    weight = _tail_weight(train)
    if architecture == "direct_residual":
        target = train.target_raw_conversion_residual.to_numpy(float)
        model = HistGradientBoostingRegressor(
            **MODEL_KWARGS, random_state=MODEL_SEEDS["residual"]
        ).fit(x_train, target, sample_weight=weight)
        correction = model.predict(x_valid)
        lower, upper = np.quantile(target, (0.02, 0.98))
        return valid.raw_score.to_numpy(float) + np.clip(correction, lower, upper)
    if architecture == "positive_hurdle":
        positive = train[TARGET].gt(0).to_numpy()
        classifier = HistGradientBoostingClassifier(
            **MODEL_KWARGS, random_state=MODEL_SEEDS["classifier"]
        ).fit(x_train, positive, sample_weight=weight)
        probability = classifier.predict_proba(x_valid)[:, 1]
        outcome = train[TARGET].to_numpy(float)
        gain = _conditional_prediction(
            train,
            valid,
            features,
            positive,
            np.maximum(outcome, 0),
            seed=MODEL_SEEDS["gain"],
        )
        loss = _conditional_prediction(
            train,
            valid,
            features,
            ~positive,
            np.maximum(-outcome, 0),
            seed=MODEL_SEEDS["loss"],
        )
        return probability * np.maximum(gain, 0) - (1 - probability) * np.maximum(loss, 0)
    if architecture == "competing_risk":
        labels = competing_class(train)
        classifier = HistGradientBoostingClassifier(
            **MODEL_KWARGS, random_state=MODEL_SEEDS["classifier"]
        ).fit(x_train, labels, sample_weight=weight)
        probabilities = classifier.predict_proba(x_valid)
        outcome = train[TARGET].to_numpy(float)
        expected = np.zeros(len(valid), dtype=float)
        for position, class_name in enumerate(classifier.classes_):
            mask = labels.eq(class_name).to_numpy()
            conditional = _conditional_prediction(
                train,
                valid,
                features,
                mask,
                outcome,
                seed=MODEL_SEEDS["class_regression"] + position,
            )
            expected += probabilities[:, position] * conditional
        return expected
    raise ValueError(architecture)


def chronological_oof(
    march: pd.DataFrame,
    features: Sequence[str],
    architecture: str,
) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for fold_name, start, end in ALL_MARCH_FOLDS:
        valid_mask = march[TIME].ge(start) & march[TIME].lt(end)
        train_mask = march[TIME].lt(start) & march[END].lt(start)
        fold_parts = []
        for side_name in ("long", "short"):
            train = march.loc[train_mask & march.side_name.eq(side_name)].copy()
            valid = march.loc[valid_mask & march.side_name.eq(side_name)].copy()
            if len(train) < 1_000 or len(valid) == 0:
                raise AblationError(
                    f"insufficient {side_name} support for {fold_name}: {len(train)}/{len(valid)}"
                )
            score = fit_predict(train, valid, features, architecture)
            fold_parts.append(
                valid.loc[
                    :,
                    [
                        "candidate_id",
                        "side_name",
                        "__symbol__",
                        "__ts__",
                        TIME,
                        END,
                        TARGET,
                        "execution_gross_ev_12h",
                        "execution_cost_return",
                        "raw_score",
                        "score_available_utc",
                    ],
                ].assign(
                    challenger_score=score,
                    selection_fold=fold_name,
                    fold_role=(
                        "mapping_calibration"
                        if fold_name == CALIBRATION_FOLD[0]
                        else "selection_evaluation"
                    ),
                    fold_train_label_end_max=train[END].max(),
                    fold_validation_start=start,
                )
            )
        fold = pd.concat(fold_parts, ignore_index=True)
        if not fold.fold_train_label_end_max.lt(fold.fold_validation_start).all():
            raise AblationError(f"purge failure in {fold_name}")
        pieces.append(fold)
    result = pd.concat(pieces, ignore_index=True)
    if result.duplicated(["candidate_id", "side_name"]).any():
        raise AblationError("March OOF predictions overlap")
    return result


def causal_selection_map(
    predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Map the OOF stream using only outcomes resolved before each snapshot."""
    source = predictions.drop(columns=["raw_score"]).rename(
        columns={"challenger_score": "raw_score"}
    )
    mapped, audit = mapping.causal_map(
        source,
        source,
        add_side_residual=True,
    )
    mapped = mapped.rename(
        columns={
            "raw_score": "challenger_score",
            "causal_pooled_side_21d": "selection_mapped_score",
            "causal_pooled_side_21d_eligible": "selection_mapped_eligible",
            "causal_pooled_side_21d_status": "selection_mapping_status",
        }
    )
    if not audit.strict_causal_window_pass.all():
        raise AblationError("March selection map contains a non-causal snapshot")
    if mapped.selection_mapped_eligible.sum() == 0:
        raise AblationError("March selection map has no eligible rows")
    selection = mapped.loc[mapped.selection_fold.isin(SELECTION_FOLD_NAMES)]
    for fold_name, part in selection.groupby("selection_fold", sort=True):
        if not part.selection_mapped_eligible.any():
            raise AblationError(f"March selection fold has no mapped rows: {fold_name}")
    return mapped, audit


def selection_metrics(
    predictions: pd.DataFrame,
    *,
    config: str,
    group: str,
    architecture: str,
    feature_count: int,
) -> dict[str, Any]:
    predictions = predictions.loc[
        predictions.selection_fold.isin(SELECTION_FOLD_NAMES)
    ].copy()
    fold_values = []
    for fold_name, part in predictions.groupby("selection_fold", sort=True):
        mapped_part = part.loc[part.selection_mapped_eligible.astype(bool)].copy()
        if mapped_part.empty:
            raise AblationError(f"selection fold has no causally mapped rows: {fold_name}")
        selected = mapping.stable_top(
            mapped_part,
            "selection_mapped_score",
            0.10,
        )
        mapped_tie = mapping.bound(
            mapped_part,
            "selection_mapped_score",
            0.10,
        )
        raw_selected = mapping.stable_top(part, "challenger_score", 0.10)
        raw_tie = mapping.bound(part, "challenger_score", 0.10)
        fold_values.append(
            {
                "fold": fold_name,
                "rows": len(part),
                "mapped_eligible_rows": len(mapped_part),
                "mapped_coverage_fraction": float(len(mapped_part) / len(part)),
                "top10_rows": len(selected),
                "top10_net_bps": float(
                    mapped_tie["random_tie_expected_net_bps"]
                ),
                "deterministic_top10_net_bps": float(
                    selected[TARGET].mean() * 1e4
                ),
                "cutoff_tie_fraction_of_book": float(
                    mapped_tie["cutoff_tie_fraction_of_book"]
                ),
                "raw_diagnostic_top10_net_bps": float(
                    raw_tie["random_tie_expected_net_bps"]
                ),
                "raw_diagnostic_deterministic_top10_net_bps": float(
                    raw_selected[TARGET].mean() * 1e4
                ),
            }
        )
    values = np.array([row["top10_net_bps"] for row in fold_values], dtype=float)
    eligible = predictions.loc[predictions.selection_mapped_eligible.astype(bool)]
    aggregate = mapping.stable_top(
        eligible,
        "selection_mapped_score",
        0.10,
    )
    aggregate_tie = mapping.bound(eligible, "selection_mapped_score", 0.10)
    raw_aggregate = mapping.stable_top(predictions, "challenger_score", 0.10)
    raw_aggregate_tie = mapping.bound(predictions, "challenger_score", 0.10)
    return {
        "config": config,
        "feature_group": group,
        "architecture": architecture,
        "feature_count": feature_count,
        "march_oof_rows": len(predictions),
        "march_oof_mapped_eligible_rows": len(eligible),
        "march_oof_mapped_coverage_fraction": float(len(eligible) / len(predictions)),
        "march_oof_global_top10_net_bps": float(
            aggregate_tie["random_tie_expected_net_bps"]
        ),
        "march_oof_deterministic_global_top10_net_bps": float(
            aggregate[TARGET].mean() * 1e4
        ),
        "march_oof_cutoff_tie_fraction_of_book": float(
            aggregate_tie["cutoff_tie_fraction_of_book"]
        ),
        "march_oof_raw_diagnostic_top10_net_bps": float(
            raw_aggregate_tie["random_tie_expected_net_bps"]
        ),
        "march_oof_raw_diagnostic_deterministic_top10_net_bps": float(
            raw_aggregate[TARGET].mean() * 1e4
        ),
        "fold_mean_top10_net_bps": float(values.mean()),
        "fold_std_top10_net_bps": float(values.std(ddof=0)),
        "fold_worst_top10_net_bps": float(values.min()),
        "stability_objective_bps": float(values.mean() - 0.5 * values.std(ddof=0) + 0.25 * values.min()),
        "fold_metrics": fold_values,
    }


def fit_forward(
    march: pd.DataFrame,
    april: pd.DataFrame,
    features: Sequence[str],
    architecture: str,
) -> np.ndarray:
    result = np.full(len(april), np.nan)
    for side_name in ("long", "short"):
        train = march.loc[march.side_name.eq(side_name)]
        valid = april.loc[april.side_name.eq(side_name)]
        result[valid.index.to_numpy()] = fit_predict(train, valid, features, architecture)
    if not np.isfinite(result).all():
        raise AblationError("forward score contains a non-finite value")
    return result


def causal_forward_map(
    march_oof: pd.DataFrame,
    april: pd.DataFrame,
    forward_score: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    reference = march_oof.copy()
    reference["raw_score"] = reference.pop("challenger_score")
    evaluate = april.loc[
        :,
        [
            "candidate_id",
            "side_name",
            "__symbol__",
            "__ts__",
            TIME,
            END,
            TARGET,
            "execution_gross_ev_12h",
            "execution_cost_return",
            "score_available_utc",
            "execution_exit_reason",
        ],
    ].copy()
    evaluate["raw_score"] = forward_score
    history = pd.concat([reference, evaluate], ignore_index=True, sort=False)
    mapped, audit = mapping.causal_map(history, evaluate, add_side_residual=True)
    mapped = mapped.rename(
        columns={
            "raw_score": "challenger_raw_score",
            "causal_pooled_side_21d": "challenger_mapped_score",
            "causal_pooled_side_21d_eligible": "challenger_mapped_eligible",
        }
    )
    if not mapped.challenger_mapped_eligible.all():
        raise AblationError("precommitted causal map has incomplete April coverage")
    return mapped, audit


def april_metrics(
    predictions: pd.DataFrame, config: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    all_metrics: list[dict[str, Any]] = []
    all_sides: list[dict[str, Any]] = []
    all_assets: list[dict[str, Any]] = []
    all_intervals: list[dict[str, Any]] = []
    for score_kind, score, eligible in (
        ("raw", "challenger_raw_score", None),
        ("mapped", "challenger_mapped_score", "challenger_mapped_eligible"),
    ):
        metrics, sides, assets, intervals = mapping.evaluate_arm(
            predictions,
            arm=f"{config}__{score_kind}",
            score=score,
            eligible=eligible,
        )
        for row in metrics:
            row.update(config=config, score_kind=score_kind)
        for table in (sides, assets, intervals):
            for row in table:
                row.update(config=config, score_kind=score_kind)
        all_metrics.extend(metrics)
        all_sides.extend(sides)
        all_assets.extend(assets)
        all_intervals.extend(intervals)
    return all_metrics, all_sides, all_assets, all_intervals


def choose(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    table = pd.DataFrame(rows)
    return (
        table.sort_values(
            ["stability_objective_bps", "feature_count", "config"],
            ascending=[False, True, True],
            kind="stable",
        )
        .iloc[0]
        .to_dict()
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    source_manifest, roles = verify_source(args.source)
    panel = pd.read_parquet(args.source / "panel.parquet")
    for column in ("__ts__", TIME, END, "score_available_utc"):
        panel[column] = pd.to_datetime(panel[column], utc=True, errors="raise")
    march = panel.loc[panel.model_development_eligible.astype(bool)].copy().reset_index(drop=True)
    april = panel.loc[panel.forward_diagnostic_only.astype(bool)].copy().reset_index(drop=True)
    groups = feature_groups(roles)
    selection_rows: list[dict[str, Any]] = []
    oof_predictions: dict[str, pd.DataFrame] = {}
    mapped_selection_predictions: dict[str, pd.DataFrame] = {}
    selection_mapping_audits: dict[str, pd.DataFrame] = {}
    # Stage 1: feature groups under one residual architecture.
    for group_name, features in groups.items():
        config = f"feature__{group_name}__direct_residual"
        oof = chronological_oof(march, features, "direct_residual")
        oof_predictions[config] = oof
        mapped_oof, selection_audit = causal_selection_map(oof)
        mapped_selection_predictions[config] = mapped_oof
        selection_mapping_audits[config] = selection_audit
        selection_rows.append(
            selection_metrics(
                mapped_oof,
                config=config,
                group=group_name,
                architecture="direct_residual",
                feature_count=len(features),
            )
        )
    feature_winner = choose(selection_rows)
    selected_group = str(feature_winner["feature_group"])
    selected_features = groups[selected_group]
    # Stage 2: target architecture on the frozen feature winner.
    target_rows: list[dict[str, Any]] = []
    for architecture in ("direct_residual", "positive_hurdle", "competing_risk"):
        config = f"target__{selected_group}__{architecture}"
        if architecture == "direct_residual":
            source_config = f"feature__{selected_group}__direct_residual"
            oof = oof_predictions[source_config]
            mapped_oof = mapped_selection_predictions[source_config]
            selection_audit = selection_mapping_audits[source_config]
        else:
            oof = chronological_oof(march, selected_features, architecture)
            mapped_oof, selection_audit = causal_selection_map(oof)
        oof_predictions[config] = oof
        mapped_selection_predictions[config] = mapped_oof
        selection_mapping_audits[config] = selection_audit
        row = selection_metrics(
            mapped_oof,
            config=config,
            group=selected_group,
            architecture=architecture,
            feature_count=len(selected_features),
        )
        target_rows.append(row)
    target_winner = choose(target_rows)
    final_config = str(target_winner["config"])
    target_fold_metrics = list(target_winner["fold_metrics"])
    target_latest_fold = next(
        row
        for row in target_fold_metrics
        if row["fold"] == MARCH_FOLDS[-1][0]
    )
    # Evaluate every distinct stage configuration on reused April.
    configs: dict[str, tuple[str, str, list[str]]] = {}
    for row in selection_rows:
        configs[str(row["config"])] = (
            str(row["feature_group"]),
            str(row["architecture"]),
            groups[str(row["feature_group"])],
        )
    for row in target_rows:
        configs[str(row["config"])] = (
            str(row["feature_group"]),
            str(row["architecture"]),
            selected_features,
        )
    metrics: list[dict[str, Any]] = []
    sides: list[dict[str, Any]] = []
    assets: list[dict[str, Any]] = []
    intervals: list[dict[str, Any]] = []
    audits: list[pd.DataFrame] = []
    prediction_ledgers: list[pd.DataFrame] = []
    for config, (group_name, architecture, features) in configs.items():
        forward_score = fit_forward(march, april, features, architecture)
        mapped, audit = causal_forward_map(
            oof_predictions[config], april, forward_score
        )
        metric, side, asset, interval = april_metrics(mapped, config)
        metrics.extend(metric)
        sides.extend(side)
        assets.extend(asset)
        intervals.extend(interval)
        audits.append(audit.assign(config=config))
        prediction_ledgers.append(
            mapped.loc[
                :,
                [
                    "candidate_id",
                    "side_name",
                    "__symbol__",
                    "__ts__",
                    TIME,
                    END,
                    TARGET,
                    "execution_gross_ev_12h",
                    "execution_cost_return",
                    "challenger_raw_score",
                    "challenger_mapped_score",
                    "challenger_mapped_eligible",
                ],
            ].assign(
                config=config,
                feature_group=group_name,
                architecture=architecture,
            )
        )
    selection_table = pd.DataFrame(
        [
            {
                **{key: value for key, value in row.items() if key != "fold_metrics"},
                "fold_metrics_json": json.dumps(row["fold_metrics"], sort_keys=True),
                "stage": "feature",
            }
            for row in selection_rows
        ]
        + [
            {
                **{key: value for key, value in row.items() if key != "fold_metrics"},
                "fold_metrics_json": json.dumps(row["fold_metrics"], sort_keys=True),
                "stage": "target",
            }
            for row in target_rows
        ]
    )
    metrics_table = pd.DataFrame(metrics)
    sides_table = pd.DataFrame(sides)
    assets_table = pd.DataFrame(assets)
    intervals_table = pd.DataFrame(intervals)
    audit_table = pd.concat(audits, ignore_index=True)
    predictions_table = pd.concat(prediction_ledgers, ignore_index=True)
    selection_audit_table = pd.concat(
        [
            audit.assign(config=config)
            for config, audit in selection_mapping_audits.items()
        ],
        ignore_index=True,
    )
    selection_predictions_table = pd.concat(
        [
            prediction.assign(config=config)
            for config, prediction in mapped_selection_predictions.items()
        ],
        ignore_index=True,
    )
    control_rows: list[dict[str, Any]] = []
    control_audits: list[pd.DataFrame] = []
    for control_name, score in (
        ("v5", "raw_score"),
        ("base", "score_base_alpha"),
        ("residual", "score_residual_expected_ev"),
        ("direct_q25", "direct_q25_return"),
    ):
        control_reference = march.loc[
            :,
            [
                "candidate_id",
                "side_name",
                "__symbol__",
                "__ts__",
                TIME,
                END,
                TARGET,
                "execution_gross_ev_12h",
                "execution_cost_return",
                "score_available_utc",
            ],
        ].copy()
        control_reference["challenger_score"] = march[score].to_numpy(float)
        control_mapped, control_audit = causal_forward_map(
            control_reference,
            april,
            april[score].to_numpy(float),
        )
        control_audits.append(control_audit.assign(control=control_name))
        for score_kind, score_column, eligible_column in (
            ("raw", "challenger_raw_score", None),
            ("mapped", "challenger_mapped_score", "challenger_mapped_eligible"),
        ):
            control_metrics_rows, _, _, _ = mapping.evaluate_arm(
                control_mapped,
                arm=f"{control_name}__{score_kind}",
                score=score_column,
                eligible=eligible_column,
            )
            for row in control_metrics_rows:
                row.update(control=control_name, score_kind=score_kind)
                control_rows.append(row)
    controls_table = pd.DataFrame(control_rows)
    control_audit_table = pd.concat(control_audits, ignore_index=True)
    winner_metric = metrics_table.loc[
        metrics_table.config.eq(final_config)
        & metrics_table.score_kind.eq("mapped")
        & metrics_table.top_fraction.eq(0.10)
    ].iloc[0]
    winner_sides = sides_table.loc[
        sides_table.config.eq(final_config) & sides_table.score_kind.eq("mapped")
    ]
    winner_assets = assets_table.loc[
        assets_table.config.eq(final_config) & assets_table.score_kind.eq("mapped")
    ]
    winner_audit = audit_table.loc[audit_table.config.eq(final_config)]
    winner_selection_audit = selection_audit_table.loc[
        selection_audit_table.config.eq(final_config)
    ]
    winner_selection_predictions = selection_predictions_table.loc[
        selection_predictions_table.config.eq(final_config)
    ]
    winner_scored_selection_predictions = winner_selection_predictions.loc[
        winner_selection_predictions.selection_fold.isin(SELECTION_FOLD_NAMES)
    ]
    best_control_top10 = float(
        controls_table.loc[
            controls_table.top_fraction.eq(0.10)
            & controls_table.score_kind.eq("mapped"),
            "random_tie_expected_net_bps",
        ].max()
    )
    gates = pd.DataFrame(
        [
            {
                "gate": "all March selection snapshots causally legal",
                "pass": bool(
                    winner_selection_audit.strict_causal_window_pass.all()
                ),
                "value": bool(
                    winner_selection_audit.strict_causal_window_pass.all()
                ),
                "threshold": "True",
            },
            {
                "gate": "March mapped selection coverage 100%",
                "pass": math.isclose(
                    float(target_winner["march_oof_mapped_coverage_fraction"]),
                    1.0,
                ),
                "value": float(
                    target_winner["march_oof_mapped_coverage_fraction"]
                ),
                "threshold": "1.0",
            },
            {
                "gate": "unsupported March warmup never receives mapped score",
                "pass": bool(
                    winner_selection_predictions.loc[
                        ~winner_selection_predictions.selection_mapped_eligible,
                        "selection_mapped_score",
                    ].isna().all()
                ),
                "value": int(
                    (
                        ~winner_selection_predictions.selection_mapped_eligible
                    ).sum()
                ),
                "threshold": "all ineligible rows have NaN mapped score",
            },
            {
                "gate": "every March selection fold has mapped support",
                "pass": bool(
                    winner_scored_selection_predictions.groupby("selection_fold")[
                        "selection_mapped_eligible"
                    ].any().all()
                ),
                "value": json.dumps(
                    {
                        str(fold): int(part.selection_mapped_eligible.sum())
                        for fold, part in winner_scored_selection_predictions.groupby(
                            "selection_fold", sort=True
                        )
                    },
                    sort_keys=True,
                ),
                "threshold": ">0 mapped rows in each fold",
            },
            {
                "gate": "March mapped aggregate top10 positive",
                "pass": float(
                    target_winner["march_oof_global_top10_net_bps"]
                ) > 0,
                "value": float(
                    target_winner["march_oof_global_top10_net_bps"]
                ),
                "threshold": ">0 bps",
            },
            {
                "gate": "March latest fold mapped top10 positive",
                "pass": float(target_latest_fold["top10_net_bps"]) > 0,
                "value": float(target_latest_fold["top10_net_bps"]),
                "threshold": ">0 bps",
            },
            {
                "gate": "March worst fold mapped top10 positive",
                "pass": float(target_winner["fold_worst_top10_net_bps"]) > 0,
                "value": float(target_winner["fold_worst_top10_net_bps"]),
                "threshold": ">0 bps",
            },
            {
                "gate": "March aggregate mapped cutoff tie fraction <=5%",
                "pass": float(
                    target_winner["march_oof_cutoff_tie_fraction_of_book"]
                ) <= 0.05,
                "value": float(
                    target_winner["march_oof_cutoff_tie_fraction_of_book"]
                ),
                "threshold": "<=0.05",
            },
            {
                "gate": "every March fold mapped cutoff tie fraction <=5%",
                "pass": all(
                    float(row["cutoff_tie_fraction_of_book"]) <= 0.05
                    for row in target_fold_metrics
                ),
                "value": max(
                    float(row["cutoff_tie_fraction_of_book"])
                    for row in target_fold_metrics
                ),
                "threshold": "<=0.05",
            },
            {
                "gate": "all April mapping snapshots causally legal",
                "pass": bool(winner_audit.strict_causal_window_pass.all()),
                "value": bool(winner_audit.strict_causal_window_pass.all()),
                "threshold": "True",
            },
            {
                "gate": "all April mapping snapshots supported",
                "pass": bool(winner_audit.pooled_support_pass.all()),
                "value": bool(winner_audit.pooled_support_pass.all()),
                "threshold": "True",
            },
            {
                "gate": "mapped coverage 100%",
                "pass": math.isclose(float(winner_metric.coverage_fraction), 1.0),
                "value": float(winner_metric.coverage_fraction),
                "threshold": "1.0",
            },
            {
                "gate": "April is not promotion evidence",
                "pass": False,
                "value": "reused rediagnostic",
                "threshold": "new untouched forward block required",
            },
            {
                "gate": "mapped expected top10 positive",
                "pass": float(winner_metric.random_tie_expected_net_bps) > 0,
                "value": float(winner_metric.random_tie_expected_net_bps),
                "threshold": ">0 bps",
            },
            {
                "gate": "mapped latest week positive",
                "pass": float(winner_metric.latest_week_net_bps) > 0,
                "value": float(winner_metric.latest_week_net_bps),
                "threshold": ">0 bps",
            },
            {
                "gate": "mapped cutoff tie fraction <=5%",
                "pass": float(winner_metric.cutoff_tie_fraction_of_book) <= 0.05,
                "value": float(winner_metric.cutoff_tie_fraction_of_book),
                "threshold": "<=0.05",
            },
            {
                "gate": "largest side share <=75%",
                "pass": bool(
                    len(winner_sides) == 2 and float(winner_sides.share.max()) <= 0.75
                ),
                "value": float(winner_sides.share.max()),
                "threshold": "<=0.75",
            },
            {
                "gate": "both sides positive",
                "pass": bool(
                    len(winner_sides) == 2 and winner_sides.net_bps.gt(0).all()
                ),
                "value": json.dumps(
                    {
                        str(row.side_name): float(row.net_bps)
                        for _, row in winner_sides.iterrows()
                    },
                    sort_keys=True,
                ),
                "threshold": "long>0 and short>0",
            },
            {
                "gate": "largest asset share <=10%",
                "pass": float(winner_assets.share.max()) <= 0.10,
                "value": float(winner_assets.share.max()),
                "threshold": "<=0.10",
            },
            {
                "gate": "absolute mapped top10 bias <=25bps",
                "pass": abs(float(winner_metric.prediction_bias_bps)) <= 25,
                "value": abs(float(winner_metric.prediction_bias_bps)),
                "threshold": "<=25 bps",
            },
            {
                "gate": "beats every identical-ID mapped control at top10",
                "pass": float(winner_metric.random_tie_expected_net_bps) > best_control_top10,
                "value": float(winner_metric.random_tie_expected_net_bps),
                "threshold": f">{best_control_top10} bps",
            },
            {
                "gate": "mapped top10 ECE <=25bps",
                "pass": float(winner_metric.calibration_ece_bps) <= 25,
                "value": float(winner_metric.calibration_ece_bps),
                "threshold": "<=25 bps",
            },
        ]
    )
    stage = Path(
        tempfile.mkdtemp(prefix=f".{args.output_dir.name}.", dir=args.output_dir.parent)
    )
    try:
        selection_table.to_csv(stage / "march_oof_selection.csv", index=False)
        metrics_table.to_csv(stage / "april_global_metrics.csv", index=False)
        sides_table.to_csv(stage / "april_side_top10.csv", index=False)
        assets_table.to_csv(stage / "april_asset_top10.csv", index=False)
        intervals_table.to_csv(stage / "april_day_block_intervals.csv", index=False)
        controls_table.to_csv(stage / "april_control_metrics.csv", index=False)
        control_audit_table.to_parquet(
            stage / "control_mapping_audit.parquet",
            index=False,
            compression="zstd",
        )
        audit_table.to_parquet(stage / "daily_mapping_audit.parquet", index=False, compression="zstd")
        predictions_table.to_parquet(stage / "april_predictions.parquet", index=False, compression="zstd")
        selection_audit_table.to_parquet(
            stage / "march_selection_mapping_audit.parquet",
            index=False,
            compression="zstd",
        )
        selection_predictions_table.to_parquet(
            stage / "march_selection_predictions.parquet",
            index=False,
            compression="zstd",
        )
        gates.to_csv(stage / "promotion_gates.csv", index=False)
        write_json(stage / "feature_groups.json", groups)
        outputs = {
            path.name: sha256(path) for path in stage.iterdir() if path.is_file()
        }
        manifest = {
            "schema": "v5_conversion_residual_ablation_v4",
            "run_id": args.output_dir.name,
            "status": "COMPLETED_REUSED_APRIL_DIAGNOSTIC_NO_PROMOTION_NO_REPLAY",
            "promotion_eligible": False,
            "portfolio_replay": "NOT_RUN",
            "search": {
                "stage_1": "7 fixed feature groups under direct residual",
                "stage_2": "direct residual versus positive hurdle versus five-class competing risk on frozen feature winner",
                "selection": "Each predeclared config first emits separate March 20-22 calibration OOF from prior-resolved March 13-19 history; March 23-31 chronological OOF is then ranked after daily causal 21d pooled+side mapping using that history, with mapped random-tie-expected pooled-global top10 stability objective mean - 0.5*std + 0.25*worst; raw scores diagnostic only",
                "feature_winner": feature_winner,
                "target_winner": target_winner,
                "final_config": final_config,
                "hpo": "none",
            },
            "validation": {
                "March": "one config-specific March 20-22 calibration OOF fold plus three March 23-31 chronological selection folds, all with label_end < validation_start; calibration rows provide only prior-resolved score-specific map history and are never scored in the selection objective",
                "April": "frozen forward reused rediagnostic, never selection and not promotion evidence",
                "mapping": "fixed daily 21d pooled isotonic plus side residual n/(n+500), global top-k",
                "controls": "every base, residual, direct-q25 and v5 comparator receives its own score-specific causal map; mapped challenger is gated only against mapped controls",
            },
            "features": {
                "groups": groups,
                "timing_mae_target_wait_actions": "excluded; predicted MAE appears only in its explicit optional-risk feature arm",
            },
            "models": {
                "geometry": MODEL_KWARGS,
                "seeds": MODEL_SEEDS,
                "side_handling": "separate long/short models, one global downstream ranking",
                "competing_classes": [
                    "trailing_positive",
                    "timeout_positive",
                    "trailing_nonpositive",
                    "timeout_nonpositive",
                    "full_stop_or_adverse",
                ],
            },
            "input_sha256": {
                "source_manifest": sha256(args.source / "manifest.json"),
                "panel": sha256(args.source / "panel.parquet"),
                "roles": sha256(args.source / "feature_roles.json"),
            },
            "outputs_sha256": outputs,
            "runner": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256(Path(__file__).resolve()),
            },
            "limitations": [
                "April has been inspected previously; all April metrics are diagnostic only.",
                "Feature-group and architecture choices require a new forward block for confirmation.",
                "The compact context transform is research-only until strict inference parity exists.",
                "The frozen B/tail2 candidate configuration was developed using March; this remains research/development evidence, never untouched promotion evidence.",
                "March 13-19 changes the amount of genuine prior-resolved training history relative to v3, and March 20-22 is generated independently for every predeclared configuration.",
                "The selected configuration is a diagnostic stability-objective leader, not an admissible winner, unless every development and forward gate passes.",
            ],
        }
        write_json(stage / "manifest.json", manifest)
        (stage / "manifest.sha256").write_text(
            sha256(stage / "manifest.json") + "  manifest.json\n"
        )
        os.replace(stage, args.output_dir)
        for invalidated in args.invalidate_artifact:
            if not invalidated.is_dir():
                raise AblationError(
                    f"artifact to invalidate is missing: {invalidated}"
                )
            write_json(
                invalidated / "INVALIDATION.json",
                {
                    "status": "INVALIDATED",
                    "reason": (
                        "March feature/architecture selection ranked raw challenger "
                        "scores rather than the causally mapped EV used by the live "
                        "pooled-global top-k admission policy."
                    ),
                    "replacement": str(args.output_dir),
                    "preserved_use": (
                        "Data integrity and April mapping remain diagnostic; the v1 "
                        "selected configuration is not mapped-policy winner evidence."
                    ),
                },
            )
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return manifest


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--source", type=Path, default=SOURCE)
    command.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    command.add_argument(
        "--invalidate-artifact",
        type=Path,
        action="append",
        default=[],
    )
    return command


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    print(json.dumps(run(args), indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
