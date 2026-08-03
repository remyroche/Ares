#!/usr/bin/env python3
"""Train a causal 12h-vs-24h timeout action head and evaluate frozen July rows."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.report_execution_ev_july_exact_economics import (  # noqa: E402
    COHORT_FLAGS,
    DEFAULT_POLICY_CONFIG,
    _artifact_record,
    _resolve,
    _sha256,
    portfolio_replays,
)
from scripts.report_execution_ev_timeout_ablation import (  # noqa: E402
    LABEL_VALUE_COLUMNS,
)
from scripts.report_historical_exact_policy_timeout_recurrence import (  # noqa: E402
    _policy_component_hashes,
)

SCHEMA = "timeout_action_head_ablation_v1"
DEFAULT_HISTORICAL = Path(
    "data_perp/artifacts/"
    "historical_exact_policy_timeout_recurrence_may_jul10_20260730_v4/"
    "paired_population.parquet"
)
DEFAULT_HISTORICAL_MANIFEST = DEFAULT_HISTORICAL.with_name("manifest.json")
DEFAULT_HISTORICAL_FEATURES = Path(
    "data_perp/artifacts/execution_ev_joined_handoff_20260725_v1/joined.parquet"
)
DEFAULT_CURRENT_ROOT = Path(
    "data_perp/artifacts/execution_ev_july20_23_retrospective_20260730_v2"
)
DEFAULT_CURRENT_PAIRED = (
    DEFAULT_CURRENT_ROOT
    / "exact_policy_timeout_ablation_12h_vs_24h_v1"
    / "paired_population.parquet"
)
DEFAULT_CURRENT_MANIFEST = DEFAULT_CURRENT_PAIRED.with_name("manifest.json")
DEFAULT_CURRENT_FEATURES = DEFAULT_CURRENT_ROOT / "preentry" / "preentry.parquet"

FEATURES = (
    "mapped_execution_ev",
    "existing_alpha_ev",
    "pred_peak_MFE_12h_ATR",
    "catboost_p_0",
    "catboost_p_1",
    "catboost_p_2",
    "catboost_p_3",
    "catboost_p_4",
    "catboost_p_5",
    "catboost_p_6",
    "catboost_entropy",
    "alpha_prediction_uncertainty",
    "alpha_leaf_support",
)
POLICIES = (
    "no_action_12h",
    "always_24h",
    "classifier_action",
    "regression_action",
    "blend_action",
)
MODEL_POLICIES = POLICIES[2:]
IDENTITY = ("candidate_id", "__ts__", "__symbol__", "side_name")


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _validate_unique(frame: pd.DataFrame, *, role: str) -> pd.DataFrame:
    missing = sorted(set(IDENTITY).difference(frame.columns))
    if missing:
        raise ValueError(f"{role} lacks identity fields: {missing}")
    if frame["candidate_id"].duplicated().any():
        raise ValueError(f"{role} candidate IDs are not unique")
    return frame


def join_causal_features(
    paired: pd.DataFrame,
    features: pd.DataFrame,
    *,
    historical: bool,
) -> pd.DataFrame:
    """Join only the explicitly common, decision-time feature contract."""

    paired = _validate_unique(paired.copy(), role="paired exact paths")
    features = _validate_unique(features.copy(), role="causal features")
    required = {*IDENTITY, "execution_decision_utc", "catboost_archetype"}
    required.update(FEATURES[1:])
    missing = sorted(required.difference(features.columns))
    if missing:
        raise ValueError(f"causal feature handoff is incomplete: {missing}")
    availability = (
        ("alpha_available_at", "peak_mfe_available_at", "catboost_available_at")
        if historical
        else (
            "feature_available_at",
            "residual_available_at",
            "peak_mfe_available_at",
            "path_catboost_available_at",
        )
    )
    missing_availability = sorted(set(availability).difference(features.columns))
    if missing_availability:
        raise ValueError(f"causal availability fields missing: {missing_availability}")
    decision = pd.to_datetime(features["execution_decision_utc"], utc=True, errors="raise")
    for column in availability:
        available = pd.to_datetime(features[column], utc=True, errors="raise")
        if (available > decision).any():
            raise ValueError(f"{column} contains post-decision feature rows")
    selected = features[
        [*IDENTITY, "catboost_archetype", *FEATURES[1:]]
    ].copy()
    output = paired.merge(
        selected, on=list(IDENTITY), how="left", validate="one_to_one", indicator=True
    )
    if not output["_merge"].eq("both").all():
        raise ValueError("causal feature handoff does not cover every paired path")
    output = output.drop(columns="_merge")
    matrix = output.loc[:, FEATURES].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(matrix.to_numpy(dtype=float)).all():
        raise ValueError("action-head features must be finite")
    output["execution_decision_utc"] = pd.to_datetime(
        output["execution_decision_utc"], utc=True, errors="raise"
    )
    output["utc_month"] = output["execution_decision_utc"].dt.strftime("%Y-%m")
    output["late_recovery"] = (
        output["execution_exit_reason__12h"].astype(str).eq("timeout")
        & output["execution_exit_reason__24h"].astype(str).eq("trailing")
        & (output["execution_net_ev_12h__12h"] <= 0.0)
        & (output["execution_net_ev_12h__24h"] > 0.0)
    )
    output["timeout_to_full_stop"] = (
        output["execution_exit_reason__12h"].astype(str).eq("timeout")
        & output["execution_exit_reason__24h"]
        .astype(str)
        .isin(("full_sl", "full_stop"))
    )
    return output


def temporal_folds(frame: pd.DataFrame, *, folds: int = 4) -> list[dict[str, Any]]:
    """Expanding temporal folds with 24h outcome resolution purge."""

    ordered = frame.sort_values("execution_decision_utc", kind="mergesort")
    unique_hours = pd.Index(ordered["execution_decision_utc"].drop_duplicates())
    if len(unique_hours) < 12:
        raise ValueError("not enough decision timestamps for temporal OOF")
    boundaries = sorted(
        {
            min(len(unique_hours) - 1, max(1, int(len(unique_hours) * fraction)))
            for fraction in np.linspace(0.25, 0.80, folds)
        }
    )
    result: list[dict[str, Any]] = []
    for fold, position in enumerate(boundaries):
        start = unique_hours[position]
        end = unique_hours[boundaries[fold + 1]] if fold + 1 < len(boundaries) else None
        validation = ordered["execution_decision_utc"].ge(start)
        if end is not None:
            validation &= ordered["execution_decision_utc"].lt(end)
        train = pd.to_datetime(
            ordered["execution_label_end_utc__24h"], utc=True, errors="raise"
        ).lt(start)
        if train.sum() < 250 or validation.sum() < 50:
            continue
        result.append(
            {
                "fold": len(result),
                "validation_start_utc": start,
                "validation_end_utc": end,
                "train_index": ordered.index[train].to_numpy(),
                "validation_index": ordered.index[validation].to_numpy(),
            }
        )
    if len(result) < 2:
        raise ValueError("fewer than two valid temporal OOF folds")
    return result


def _class_weights(target: np.ndarray) -> np.ndarray:
    positive = int(target.sum())
    negative = int(len(target) - positive)
    if positive == 0 or negative == 0:
        return np.ones(len(target), dtype=float)
    weight = np.ones(len(target), dtype=float)
    weight[target.astype(bool)] = negative / positive
    return weight


def _fit_models(x: np.ndarray, frame: pd.DataFrame) -> dict[str, Any]:
    delta = np.clip(
        frame["paired_delta_net_24h_minus_12h"].to_numpy(dtype=float), -0.08, 0.08
    )
    late = frame["late_recovery"].to_numpy(dtype=int)
    full = frame["timeout_to_full_stop"].to_numpy(dtype=int)
    regression = HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_iter=120,
        max_leaf_nodes=15,
        min_samples_leaf=80,
        l2_regularization=10.0,
        random_state=41,
    ).fit(x, delta)
    late_model = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=100,
        max_leaf_nodes=15,
        min_samples_leaf=80,
        l2_regularization=10.0,
        random_state=42,
    ).fit(x, late, sample_weight=_class_weights(late))
    full_model = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=100,
        max_leaf_nodes=15,
        min_samples_leaf=80,
        l2_regularization=10.0,
        random_state=43,
    ).fit(x, full, sample_weight=_class_weights(full))
    return {"regression": regression, "late": late_model, "full": full_model}


def _predict_models(models: Mapping[str, Any], x: np.ndarray) -> dict[str, np.ndarray]:
    regression = np.asarray(models["regression"].predict(x), dtype=float)
    late = np.asarray(models["late"].predict_proba(x)[:, 1], dtype=float)
    full = np.asarray(models["full"].predict_proba(x)[:, 1], dtype=float)
    classifier = late - full
    blend = regression + 0.02 * classifier
    return {
        "predicted_delta": regression,
        "predicted_late_recovery": late,
        "predicted_timeout_full_stop": full,
        "classifier_action_score": classifier,
        "regression_action_score": regression,
        "blend_action_score": blend,
    }


def temporal_oof_predictions(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Generate expanding temporal OOF scores independently per side."""

    output = frame.copy()
    score_columns = (
        "predicted_delta",
        "predicted_late_recovery",
        "predicted_timeout_full_stop",
        "classifier_action_score",
        "regression_action_score",
        "blend_action_score",
    )
    for column in score_columns:
        output[column] = np.nan
    output["action_oof_fold"] = pd.Series(pd.NA, index=output.index, dtype="Int64")
    fold_records: list[dict[str, Any]] = []
    for side in ("long", "short"):
        local = output.loc[output["side_name"].eq(side)]
        folds = temporal_folds(local)
        for fold in folds:
            train = output.loc[fold["train_index"]]
            validation = output.loc[fold["validation_index"]]
            x_train = train.loc[:, FEATURES].to_numpy(dtype=float)
            x_validation = validation.loc[:, FEATURES].to_numpy(dtype=float)
            models = _fit_models(x_train, train)
            predictions = _predict_models(models, x_validation)
            for column, values in predictions.items():
                output.loc[validation.index, column] = values
            output.loc[validation.index, "action_oof_fold"] = int(fold["fold"])
            fold_records.append(
                {
                    "side": side,
                    "fold": int(fold["fold"]),
                    "validation_start_utc": fold["validation_start_utc"],
                    "validation_end_utc": fold["validation_end_utc"],
                    "train_rows": int(len(train)),
                    "validation_rows": int(len(validation)),
                    "train_label_end_max_utc": train[
                        "execution_label_end_utc__24h"
                    ].max(),
                }
            )
    return output, fold_records


def select_thresholds(oof: pd.DataFrame) -> tuple[dict[str, float], pd.DataFrame]:
    """Select economic action thresholds exclusively on temporal OOF rows."""

    eligible = oof.loc[oof["action_oof_fold"].notna()].copy()
    delta = eligible["paired_delta_net_24h_minus_12h"].to_numpy(dtype=float)
    rows: list[dict[str, Any]] = []
    thresholds: dict[str, float] = {}
    score_map = {
        "classifier_action": "classifier_action_score",
        "regression_action": "regression_action_score",
        "blend_action": "blend_action_score",
    }
    for policy, score_column in score_map.items():
        score = eligible[score_column].to_numpy(dtype=float)
        candidates = np.unique(np.quantile(score, np.linspace(0.50, 0.98, 33)))
        best: tuple[float, float] | None = None
        for threshold in candidates:
            action = score > threshold
            coverage = float(action.mean())
            if action.sum() < 50 or not 0.015 <= coverage <= 0.50:
                continue
            total_increment = float(np.mean(np.where(action, delta, 0.0)))
            side_increment = []
            for side in ("long", "short"):
                is_side = eligible["side_name"].eq(side).to_numpy()
                side_increment.append(
                    float(np.mean(np.where(action[is_side], delta[is_side], 0.0)))
                )
            objective = total_increment + 0.25 * min(side_increment)
            rows.append(
                {
                    "policy": policy,
                    "threshold": float(threshold),
                    "coverage": coverage,
                    "action_rows": int(action.sum()),
                    "mean_increment_all_rows_bps": total_increment * 10_000.0,
                    "long_increment_all_rows_bps": side_increment[0] * 10_000.0,
                    "short_increment_all_rows_bps": side_increment[1] * 10_000.0,
                    "objective": objective,
                }
            )
            candidate = (objective, float(threshold))
            if best is None or candidate > best:
                best = candidate
        if best is None:
            raise ValueError(f"no admissible OOF threshold for {policy}")
        thresholds[policy] = best[1]
    return thresholds, pd.DataFrame(rows)


def router_gate_summary(
    threshold_grid: pd.DataFrame, thresholds: Mapping[str, float]
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for policy, threshold in thresholds.items():
        winner = threshold_grid.loc[
            threshold_grid["policy"].eq(policy)
            & np.isclose(
                threshold_grid["threshold"].to_numpy(dtype=float),
                float(threshold),
                atol=1e-15,
                rtol=0.0,
            )
        ]
        if len(winner) != 1:
            raise ValueError(f"OOF router winner is not unique for {policy}")
        record = winner.iloc[0].to_dict()
        record["passed_positive_oof_increment_gate"] = bool(
            float(record["mean_increment_all_rows_bps"]) > 0.0
            and float(record["long_increment_all_rows_bps"]) >= 0.0
            and float(record["short_increment_all_rows_bps"]) >= 0.0
        )
        summary[policy] = record
    return summary


def action_flags(frame: pd.DataFrame, thresholds: Mapping[str, float]) -> pd.DataFrame:
    output = frame.copy()
    output["action__no_action_12h"] = False
    output["action__always_24h"] = True
    for policy in MODEL_POLICIES:
        output[f"action__{policy}"] = (
            output[f"{policy.replace('_action', '')}_action_score"]
            > float(thresholds[policy])
        )
    return output


def action_metrics(frame: pd.DataFrame, *, evaluation: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    dimensions: list[tuple[str, Sequence[str]]] = [
        ("overall", ()),
        ("side", ("side_name",)),
        ("month", ("utc_month",)),
        ("regime", ("catboost_archetype",)),
        ("side_regime", ("side_name", "catboost_archetype")),
    ]
    cohorts = {"global_top10": frame["global_top10_capacity_member"].astype(bool)}
    cohorts.update(
        {name: frame[column].astype(bool) for name, column in COHORT_FLAGS.items()}
    )
    # COHORT_FLAGS also contains global_top10; dictionary semantics intentionally dedupe.
    for cohort, cohort_mask in cohorts.items():
        selected = frame.loc[cohort_mask]
        for policy in POLICIES:
            action_column = f"action__{policy}"
            for scope, keys in dimensions:
                groups = [((), selected)] if not keys else selected.groupby(list(keys), sort=True)
                for values, group in groups:
                    values = values if isinstance(values, tuple) else (values,)
                    action = group[action_column].astype(bool)
                    delta = group["paired_delta_net_24h_minus_12h"]
                    chosen = np.where(
                        action,
                        group["execution_net_ev_12h__24h"],
                        group["execution_net_ev_12h__12h"],
                    )
                    row = {
                        "evaluation": evaluation,
                        "cohort": cohort,
                        "policy": policy,
                        "scope": scope,
                        "side_name": None,
                        "utc_month": None,
                        "catboost_archetype": None,
                        "rows": int(len(group)),
                        "action_rows": int(action.sum()),
                        "coverage": float(action.mean()),
                        "abstention_rate": float((~action).mean()),
                        "mean_chosen_net_bps": float(np.mean(chosen) * 10_000.0),
                        "increment_vs_12h_all_rows_bps": float(
                            np.mean(np.where(action, delta, 0.0)) * 10_000.0
                        ),
                        "action_mean_delta_bps": float(delta.loc[action].mean() * 10_000.0)
                        if action.any()
                        else np.nan,
                        "action_positive_delta_rate": float((delta.loc[action] > 0).mean())
                        if action.any()
                        else np.nan,
                        "late_recovery_capture_rows": int(
                            (action & group["late_recovery"]).sum()
                        ),
                        "timeout_full_stop_exposure_rows": int(
                            (action & group["timeout_to_full_stop"]).sum()
                        ),
                    }
                    for key, value in zip(keys, values):
                        row[key] = value
                    rows.append(row)
    return pd.DataFrame(rows)


def discrimination_metrics(frame: pd.DataFrame, *, evaluation: str) -> pd.DataFrame:
    rows = []
    covered = frame.loc[frame["predicted_delta"].notna()]
    for side, group in covered.groupby("side_name", sort=True):
        for target, prediction in (
            ("late_recovery", "predicted_late_recovery"),
            ("timeout_to_full_stop", "predicted_timeout_full_stop"),
        ):
            y = group[target].astype(int)
            score = group[prediction]
            rows.append(
                {
                    "evaluation": evaluation,
                    "side_name": side,
                    "target": target,
                    "rows": int(len(group)),
                    "positives": int(y.sum()),
                    "positive_rate": float(y.mean()),
                    "roc_auc": float(roc_auc_score(y, score))
                    if y.nunique() == 2
                    else np.nan,
                    "average_precision": float(average_precision_score(y, score))
                    if y.sum()
                    else np.nan,
                    "delta_rank_ic": float(
                        group["predicted_delta"].corr(
                            group["paired_delta_net_24h_minus_12h"], method="spearman"
                        )
                    ),
                }
            )
    return pd.DataFrame(rows)


def mixed_horizon_frame(frame: pd.DataFrame, policy: str) -> pd.DataFrame:
    output = frame.copy()
    action = output[f"action__{policy}"].astype(bool)
    for column in LABEL_VALUE_COLUMNS:
        output[column] = np.where(
            action, output[f"{column}__24h"], output[f"{column}__12h"]
        )
    output["timeout_action_policy"] = policy
    output["timeout_extended_to_24h"] = action
    return output


def portfolio_action_replays(
    current: pd.DataFrame, *, policy_path: Path, initial_wallet: float
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    summaries = []
    decisions = []
    equities = []
    sides = []
    contracts: dict[str, Any] = {}
    for policy in POLICIES:
        mixed = mixed_horizon_frame(current, policy)
        summary, decision, equity, side, contract = portfolio_replays(
            mixed,
            policy_path=policy_path,
            initial_wallet=initial_wallet,
        )
        decision["timeout_extended_to_24h"] = False
        for cohort, flag in COHORT_FLAGS.items():
            book_action = mixed.loc[
                mixed[flag].astype(bool), f"action__{policy}"
            ].astype(bool).to_numpy()
            cohort_mask = decision["cohort"].eq(cohort)
            candidate_index = pd.to_numeric(
                decision.loc[cohort_mask, "candidate_index"], errors="raise"
            ).to_numpy(dtype=int)
            if len(candidate_index) and (
                candidate_index.min() < 0 or candidate_index.max() >= len(book_action)
            ):
                raise ValueError("portfolio candidate index is outside its frozen cohort")
            decision.loc[cohort_mask, "timeout_extended_to_24h"] = book_action[
                candidate_index
            ]
            summary.loc[summary["cohort"].eq(cohort), "book_action_rows"] = int(
                book_action.sum()
            )
        for (replay_arm, cohort), group in decision.groupby(
            ["replay_arm", "cohort"], sort=False
        ):
            accepted = group["accepted"].astype(bool)
            acted = group["timeout_extended_to_24h"].astype(bool)
            mask = summary["replay_arm"].eq(replay_arm) & summary["cohort"].eq(cohort)
            summary.loc[mask, "accepted_action_rows"] = int((accepted & acted).sum())
            summary.loc[mask, "accepted_action_rate"] = float(
                (accepted & acted).sum() / max(int(accepted.sum()), 1)
            )
        for table in (summary, decision, equity, side):
            table.insert(0, "timeout_action_policy", policy)
        summaries.append(summary)
        decisions.append(decision)
        equities.append(equity)
        sides.append(side)
        contracts[policy] = contract
    return (
        pd.concat(summaries, ignore_index=True),
        pd.concat(decisions, ignore_index=True),
        pd.concat(equities, ignore_index=True),
        pd.concat(sides, ignore_index=True),
        contracts,
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    for field in (
        "historical",
        "historical_manifest",
        "historical_features",
        "current",
        "current_manifest",
        "current_features",
        "policy",
        "output_dir",
    ):
        setattr(args, field, _resolve(getattr(args, field)))
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    historical_manifest = _read_json(args.historical_manifest)
    current_manifest = _read_json(args.current_manifest)
    if (
        historical_manifest.get("schema")
        != "historical_exact_policy_timeout_recurrence_v2"
        or int(
            historical_manifest.get("selection_contract", {}).get("finite_oof_rows", -1)
        )
        != 114_096
    ):
        raise ValueError("historical v4 strict mapped OOF contract is not authoritative")
    historical_bound = historical_manifest.get("outputs", {}).get(
        "paired_population", {}
    )
    current_bound = current_manifest.get("outputs", {}).get("paired_population", {})
    if (
        _sha256(args.historical) != historical_bound.get("sha256")
        or _sha256(args.current) != current_bound.get("sha256")
    ):
        raise ValueError("paired exact-path inputs are not hash-bound by their manifests")
    historical_policy_sha = (
        historical_manifest.get("inputs", {}).get("signed_policy", {}).get("sha256")
    )
    current_policy_sha = (
        current_manifest.get("inputs", {})
        .get("signed_simple_policy", {})
        .get("sha256")
    )
    historical_components = (
        historical_manifest.get("inputs", {})
        .get("signed_policy", {})
        .get("component_hashes")
    )
    current_components = _policy_component_hashes(args.policy)
    if current_policy_sha != _sha256(args.policy):
        raise ValueError("current paired paths are not bound to the replay policy")
    if not historical_components or historical_components != current_components:
        raise ValueError(
            "historical/current paired paths do not share policy core and strategies"
        )
    if (
        current_manifest.get("paired_contract", {}).get("rows", 5_760) != 5_760
        and current_manifest.get("paired_contract", {}).get("rows") is not None
    ):
        raise ValueError("current paired timeout population is not the fixed 5,760 rows")
    historical = join_causal_features(
        pd.read_parquet(args.historical),
        pd.read_parquet(args.historical_features),
        historical=True,
    )
    current = join_causal_features(
        pd.read_parquet(args.current),
        pd.read_parquet(args.current_features),
        historical=False,
    )
    if len(current) != 5_760:
        raise ValueError("current frozen evaluation must contain exactly 5,760 rows")
    training = historical.loc[
        historical["global_top10_capacity_member"].astype(bool)
    ].copy()
    oof, folds = temporal_oof_predictions(training)
    thresholds, threshold_grid = select_thresholds(oof)
    router_gates = router_gate_summary(threshold_grid, thresholds)
    oof = action_flags(oof, thresholds)
    final_models: dict[str, Any] = {}
    for side in ("long", "short"):
        local = training.loc[training["side_name"].eq(side)]
        final_models[side] = _fit_models(
            local.loc[:, FEATURES].to_numpy(dtype=float), local
        )
    for column in (
        "predicted_delta",
        "predicted_late_recovery",
        "predicted_timeout_full_stop",
        "classifier_action_score",
        "regression_action_score",
        "blend_action_score",
    ):
        current[column] = np.nan
    for side, models in final_models.items():
        mask = current["side_name"].eq(side)
        predictions = _predict_models(
            models, current.loc[mask, FEATURES].to_numpy(dtype=float)
        )
        for column, values in predictions.items():
            current.loc[mask, column] = values
    current = action_flags(current, thresholds)
    oof_evaluation = oof.loc[oof["action_oof_fold"].notna()].copy()
    oof_metrics = action_metrics(
        oof_evaluation, evaluation="historical_temporal_oof"
    )
    current_metrics = action_metrics(current, evaluation="july20_23_frozen_once")
    discrimination = pd.concat(
        [
            discrimination_metrics(
                oof_evaluation, evaluation="historical_temporal_oof"
            ),
            discrimination_metrics(
                current.loc[current["global_top10_capacity_member"].astype(bool)],
                evaluation="july20_23_frozen_once",
            ),
        ],
        ignore_index=True,
    )
    (
        portfolio_summary,
        portfolio_decisions,
        portfolio_equity,
        portfolio_side,
        portfolio_contracts,
    ) = portfolio_action_replays(
        current, policy_path=args.policy, initial_wallet=args.initial_wallet
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    outputs = {
        "historical_oof_predictions": args.output_dir
        / "historical_temporal_oof_predictions.parquet",
        "current_frozen_predictions": args.output_dir
        / "july20_23_frozen_action_predictions.parquet",
        "threshold_grid": args.output_dir / "threshold_grid.csv",
        "action_metrics": args.output_dir / "action_metrics.csv",
        "discrimination_metrics": args.output_dir / "discrimination_metrics.csv",
        "portfolio_summary": args.output_dir / "portfolio_summary.csv",
        "portfolio_side_metrics": args.output_dir / "portfolio_side_metrics.csv",
        "portfolio_decisions": args.output_dir / "portfolio_decisions.parquet",
        "portfolio_equity": args.output_dir / "portfolio_equity.parquet",
        "model_bundle": args.output_dir / "model_bundle.joblib",
    }
    oof.to_parquet(outputs["historical_oof_predictions"], index=False, compression="zstd")
    current.to_parquet(outputs["current_frozen_predictions"], index=False, compression="zstd")
    threshold_grid.to_csv(outputs["threshold_grid"], index=False)
    pd.concat([oof_metrics, current_metrics], ignore_index=True).to_csv(
        outputs["action_metrics"], index=False
    )
    discrimination.to_csv(outputs["discrimination_metrics"], index=False)
    portfolio_summary.to_csv(outputs["portfolio_summary"], index=False)
    portfolio_side.to_csv(outputs["portfolio_side_metrics"], index=False)
    portfolio_decisions.to_parquet(
        outputs["portfolio_decisions"], index=False, compression="zstd"
    )
    portfolio_equity.to_parquet(
        outputs["portfolio_equity"], index=False, compression="zstd"
    )
    joblib.dump(
        {
            "schema": SCHEMA,
            "features": FEATURES,
            "models_by_side": final_models,
            "thresholds": thresholds,
            "trained_through_decision_utc": training["execution_decision_utc"].max(),
            "trained_outcomes_available_through_utc": training[
                "execution_label_end_utc__24h"
            ].max(),
        },
        outputs["model_bundle"],
    )
    manifest = {
        "schema": SCHEMA,
        "status": "research_only_frozen_once_nonpromotable",
        "promotion_eligible": False,
        "contracts": {
            "architecture": "frozen execution-EV admission -> separate timeout action head",
            "training_population": "v4 frozen pooled global-top10 only",
            "models": "per-side temporal-OOF regression plus late-recovery/full-stop competing-risk classifiers",
            "features": list(FEATURES),
            "feature_availability": "all feature availability timestamps <= decision; no exit/outcome/calendar shortcut is a feature",
            "folds": folds,
            "thresholds": thresholds,
            "router_oof_gates": router_gates,
            "threshold_selection": "temporal-OOF economic objective only; 1.5%-50% action coverage",
            "current_evaluation": "models and routers frozen before 2026-07-20; evaluated once on identical July20-23 rows",
            "admission": "frozen global top10 and mapped >0/25/50 cohorts; no outcome reselection and no per-timestamp quota",
            "accounting": "choose stored 12h or 24h exact net outcome; no cost reapplied; portfolio consumes stored net verbatim",
            "deployment_change": False,
            "policy_lineage": {
                "historical_wrapper_sha256": historical_policy_sha,
                "current_wrapper_sha256": current_policy_sha,
                "wrapper_hashes_equal": historical_policy_sha == current_policy_sha,
                "core_and_all_strategy_component_hashes_equal": True,
                "component_hashes": current_components,
            },
        },
        "inputs": {
            "historical_paired": _artifact_record(args.historical),
            "historical_manifest": _artifact_record(args.historical_manifest),
            "historical_causal_features": _artifact_record(args.historical_features),
            "current_paired": _artifact_record(args.current),
            "current_manifest": _artifact_record(args.current_manifest),
            "current_causal_features": _artifact_record(args.current_features),
            "signed_policy": _artifact_record(args.policy),
        },
        "portfolio_contracts": portfolio_contracts,
        "outputs": {name: _artifact_record(path) for name, path in outputs.items()},
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical", type=Path, default=DEFAULT_HISTORICAL)
    parser.add_argument(
        "--historical-manifest", type=Path, default=DEFAULT_HISTORICAL_MANIFEST
    )
    parser.add_argument(
        "--historical-features", type=Path, default=DEFAULT_HISTORICAL_FEATURES
    )
    parser.add_argument("--current", type=Path, default=DEFAULT_CURRENT_PAIRED)
    parser.add_argument(
        "--current-manifest", type=Path, default=DEFAULT_CURRENT_MANIFEST
    )
    parser.add_argument(
        "--current-features", type=Path, default=DEFAULT_CURRENT_FEATURES
    )
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY_CONFIG)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--initial-wallet", type=float, default=10_000.0)
    return parser


def main() -> None:
    manifest = run(_parser().parse_args())
    print(json.dumps(manifest["contracts"], indent=2, default=str))


if __name__ == "__main__":
    main()
