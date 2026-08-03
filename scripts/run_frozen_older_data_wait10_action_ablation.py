#!/usr/bin/env python3
"""Train Wait10 action heads on older exact labels and score a frozen book.

February and causally resolved March rows are training sources only.  The
March/April evaluation identities and fractional global-book weights remain
those of the sealed no-rerank handoff.  Model family, feature groups and
action rules are predeclared; reused evaluation months cannot promote a rule.
"""
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
from sklearn.metrics import brier_score_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.materialize_febapr_current_policy_wait10_action import (
    BASE_FEATURES,
    MODEL_FEATURES,
    STATE_FEATURES,
    identity_digest,
    sha256,
)

TRAINING_ROOT = (
    ROOT
    / "data_perp/artifacts/febapr2025_current_policy_wait10_action_20260730_v1"
)
HANDOFF_ROOT = ROOT / "data_perp/artifacts/frozen_entry_action_handoff_20260730_v2"
PRIOR_RESULT_ROOT = (
    ROOT
    / "data_perp/artifacts/frozen_preentry_wait10_action_ablation_20260730_v2"
)
OUT = (
    ROOT
    / "data_perp/artifacts/frozen_older_data_wait10_action_ablation_20260730_v1"
)

SCHEMA = "frozen_older_data_wait10_action_ablation_v1"
IDENTITY = ("candidate_id", "side_name", "__symbol__", "__ts__")
WEIGHTS = ("weight_top_01", "weight_top_05", "weight_top_10", "weight_top_20")
TOPS = (0.01, 0.05, 0.10, 0.20)
TRANSITION_FEATURES = (
    "regime_stability_24h",
    "regime_transition_entropy_12h",
    "regime_transition_entropy_48h",
    "mkt_atr_expansion_1h",
    "breadth_accel_1h",
    "breadth_chg_1h",
    "cross_asset_corr_chg_1h",
    "correlation_breakdown_dispersion",
    "correlation_heterogeneity_dispersion",
    "leverage_build_score",
    "liquidation_onset_score",
    "liquidation_climax_score",
    "fragile_leverage_rebuild",
    "shock_12h",
    "shock_vol_ratio",
    "entropy_jump_24h",
    "complexity_regime_24h",
)
FEATURE_SETS: Mapping[str, tuple[str, ...]] = {
    "base_only": BASE_FEATURES,
    "base_plus_transition": (*BASE_FEATURES, *TRANSITION_FEATURES),
    "all_state_transition": MODEL_FEATURES,
}
POLICIES = (
    "enter_now",
    "always_wait10",
    "oracle_wait10",
    "direct_delta",
    "expected_delta",
    "q25_guard",
    "weighted_q25_fixed",
    "weighted_q25_calibrated",
    "soft_q25",
)


class AblationError(RuntimeError):
    pass


class ConstantRegressor:
    def __init__(self, value: float):
        self.value = float(value)

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        return np.full(len(x), self.value, dtype=float)


class ConstantClassifier:
    def __init__(self, probability: float):
        self.probability = float(np.clip(probability, 0.0, 1.0))

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        return np.column_stack(
            [
                np.full(len(x), 1.0 - self.probability),
                np.full(len(x), self.probability),
            ]
        )


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
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(safe(payload), indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def verify_artifact(root: Path, schema: str) -> dict[str, Any]:
    manifest_path = root / "manifest.json"
    seal_path = root / "manifest.sha256"
    if not manifest_path.is_file() or not seal_path.is_file():
        raise AblationError(f"sealed artifact required: {root}")
    if sha256(manifest_path) != seal_path.read_text().split()[0]:
        raise AblationError(f"manifest seal mismatch: {root}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != schema:
        raise AblationError(f"schema mismatch: {root}")
    for name, digest in manifest.get("outputs_sha256", {}).items():
        if sha256(root / name) != digest:
            raise AblationError(f"output hash mismatch: {root / name}")
    return manifest


def prepare_x(
    frame: pd.DataFrame,
    features: Sequence[str],
    medians: pd.Series | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    x = frame.loc[:, list(features)].apply(pd.to_numeric, errors="coerce")
    if medians is None:
        medians = x.median(axis=0, skipna=True).fillna(0.0)
    x = x.fillna(medians).fillna(0.0).astype("float32")
    if not np.isfinite(x.to_numpy(dtype=float)).all():
        raise AblationError("model inputs remain non-finite after train-only imputation")
    return x, medians


def _fit_regressor(
    x: pd.DataFrame,
    y: Sequence[float],
    *,
    seed: int,
    objective: str = "regression_l1",
    alpha: float | None = None,
) -> Any:
    target = np.asarray(y, dtype=float)
    if len(target) < 40 or float(np.std(target)) <= 1e-12:
        return ConstantRegressor(float(np.mean(target)) if len(target) else 0.0)
    import lightgbm as lgb

    parameters: dict[str, Any] = {
        "objective": objective,
        "n_estimators": 220,
        "learning_rate": 0.03,
        "num_leaves": 15,
        "max_depth": 4,
        "min_child_samples": 80,
        "reg_alpha": 0.15,
        "reg_lambda": 7.0,
        "colsample_bytree": 0.80,
        "subsample": 0.85,
        "subsample_freq": 1,
        "max_bin": 127,
        "random_state": seed,
        "deterministic": True,
        "force_col_wise": True,
        "n_jobs": 4,
        "verbosity": -1,
    }
    if alpha is not None:
        parameters["alpha"] = float(alpha)
    model = lgb.LGBMRegressor(**parameters)
    model.fit(x, target)
    return model


def _fit_classifier(
    x: pd.DataFrame,
    y: Sequence[bool],
    *,
    seed: int,
    sample_weight: Sequence[float] | None = None,
) -> Any:
    target = np.asarray(y, dtype=np.int8)
    if np.unique(target).size < 2:
        return ConstantClassifier(float(target[0]) if len(target) else 0.0)
    import lightgbm as lgb

    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=220,
        learning_rate=0.03,
        num_leaves=15,
        max_depth=4,
        min_child_samples=80,
        reg_alpha=0.15,
        reg_lambda=7.0,
        colsample_bytree=0.80,
        subsample=0.85,
        subsample_freq=1,
        max_bin=127,
        random_state=seed,
        deterministic=True,
        force_col_wise=True,
        n_jobs=4,
        verbosity=-1,
    )
    model.fit(x, target, sample_weight=sample_weight)
    return model


def fit_bundle(x: pd.DataFrame, delta: np.ndarray, seed: int) -> dict[str, Any]:
    positive = delta > 0.0
    economic_weight = np.clip(np.abs(delta) * 10_000.0, 0.25, 20.0)
    soft = 1.0 / (1.0 + np.exp(-np.clip(delta * 10_000.0 / 25.0, -40.0, 40.0)))
    return {
        "direct": _fit_regressor(x, delta, seed=seed),
        "q25": _fit_regressor(
            x, delta, seed=seed + 1, objective="quantile", alpha=0.25
        ),
        "event": _fit_classifier(x, positive, seed=seed + 2),
        "weighted_event": _fit_classifier(
            x,
            positive,
            seed=seed + 3,
            sample_weight=economic_weight,
        ),
        "positive": _fit_regressor(
            x.loc[positive], delta[positive], seed=seed + 4
        ),
        "negative": _fit_regressor(
            x.loc[~positive], -delta[~positive], seed=seed + 5
        ),
        "soft": _fit_regressor(x, soft, seed=seed + 6),
    }


def predict_bundle(models: Mapping[str, Any], x: pd.DataFrame) -> pd.DataFrame:
    event = np.asarray(models["event"].predict_proba(x), dtype=float)[:, 1]
    weighted = np.asarray(
        models["weighted_event"].predict_proba(x), dtype=float
    )[:, 1]
    positive = np.maximum(np.asarray(models["positive"].predict(x), dtype=float), 0.0)
    negative = np.maximum(np.asarray(models["negative"].predict(x), dtype=float), 0.0)
    result = pd.DataFrame(
        {
            "pred_direct_delta": np.asarray(models["direct"].predict(x), dtype=float),
            "pred_q25_delta": np.asarray(models["q25"].predict(x), dtype=float),
            "pred_event_probability": np.clip(event, 0.0, 1.0),
            "pred_weighted_event_score": np.clip(weighted, 0.0, 1.0),
            "pred_positive_delta": positive,
            "pred_negative_delta": negative,
            "pred_soft_score": np.clip(
                np.asarray(models["soft"].predict(x), dtype=float), 0.0, 1.0
            ),
        }
    )
    result["pred_expected_delta"] = event * positive - (1.0 - event) * negative
    return result


def calibration_split(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    ordered = frame.sort_values(["execution_decision_utc", "candidate_id"], kind="stable")
    split_position = max(1, int(math.floor(len(ordered) * 0.80)))
    calibration_start = ordered.iloc[split_position]["execution_decision_utc"]
    calibration = frame["execution_decision_utc"].ge(calibration_start).to_numpy()
    core = frame["execution_label_end_utc"].lt(calibration_start).to_numpy()
    if core.sum() < 500 or calibration.sum() < 100:
        raise AblationError("training source lacks a valid resolved-label calibration split")
    return np.flatnonzero(core), np.flatnonzero(calibration)


def choose_weighted_threshold(
    calibration: pd.DataFrame,
    *,
    minimum_action_rate: float = 0.005,
) -> tuple[float, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    days = pd.to_datetime(
        calibration["execution_decision_utc"], utc=True
    ).dt.floor("D")
    for threshold in np.linspace(0.50, 0.95, 19):
        action = (
            calibration["pred_weighted_event_score"].ge(threshold)
            & calibration["pred_q25_delta"].gt(0.0)
        )
        contribution = calibration["wait_delta"].where(action, 0.0)
        daily = pd.DataFrame({"day": days, "contribution": contribution}).groupby(
            "day", sort=True
        )["contribution"].mean()
        mean = float(contribution.mean())
        se = (
            float(daily.std(ddof=1) / np.sqrt(len(daily)))
            if len(daily) > 1
            else np.inf
        )
        rate = float(action.mean())
        rows.append(
            {
                "threshold": float(threshold),
                "action_rate": rate,
                "delta_bps": mean * 10_000.0,
                "day_cluster_lcb90_bps": (mean - 1.645 * se) * 10_000.0,
                "days": int(len(daily)),
            }
        )
    eligible = [
        row
        for row in rows
        if row["action_rate"] >= minimum_action_rate
        and row["day_cluster_lcb90_bps"] > 0.0
    ]
    if not eligible:
        return np.inf, {
            "selected_threshold": np.inf,
            "selection": "ABSTAIN_NO_POSITIVE_CALIBRATION_LOWER_BOUND",
            "grid": rows,
        }
    winner = max(
        eligible,
        key=lambda row: (row["day_cluster_lcb90_bps"], row["delta_bps"], row["threshold"]),
    )
    return float(winner["threshold"]), {
        "selected_threshold": float(winner["threshold"]),
        "selection": "MAX_POSITIVE_DAY_CLUSTER_LCB90_TRAIN_ONLY",
        "winner": winner,
        "grid": rows,
    }


def route(policy: str, rows: pd.DataFrame, threshold: float) -> np.ndarray:
    if policy == "enter_now":
        return np.zeros(len(rows), dtype=bool)
    if policy == "always_wait10":
        return np.ones(len(rows), dtype=bool)
    if policy == "oracle_wait10":
        return rows["wait_delta"].to_numpy(dtype=float) > 0.0
    if policy == "direct_delta":
        return rows["pred_direct_delta"].to_numpy(dtype=float) > 0.0
    if policy == "expected_delta":
        return rows["pred_expected_delta"].to_numpy(dtype=float) > 0.0
    if policy == "q25_guard":
        return rows["pred_q25_delta"].to_numpy(dtype=float) > 0.0
    if policy == "weighted_q25_fixed":
        return (
            rows["pred_weighted_event_score"].to_numpy(dtype=float) > 0.5
        ) & (rows["pred_q25_delta"].to_numpy(dtype=float) > 0.0)
    if policy == "weighted_q25_calibrated":
        return (
            rows["pred_weighted_event_score"].to_numpy(dtype=float) >= threshold
        ) & (rows["pred_q25_delta"].to_numpy(dtype=float) > 0.0)
    if policy == "soft_q25":
        return (rows["pred_soft_score"].to_numpy(dtype=float) > 0.5) & (
            rows["pred_q25_delta"].to_numpy(dtype=float) > 0.0
        )
    raise AblationError(f"unknown policy: {policy}")


def weighted_mean(values: Sequence[float], weights: Sequence[float]) -> float:
    value = np.asarray(values, dtype=float)
    weight = np.asarray(weights, dtype=float)
    denominator = float(weight.sum())
    return (
        float(np.sum(value * weight) / denominator)
        if denominator > 0.0
        else np.nan
    )


def economics(rows: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    group_keys = ["training_source", "feature_set", "evaluation"]
    for keys, part in rows.groupby(group_keys, sort=True):
        training_source, feature_set, evaluation = keys
        threshold = float(part["calibrated_threshold"].iloc[0])
        for fraction in TOPS:
            weight_col = f"weight_top_{int(fraction * 100):02d}"
            active = part.loc[part[weight_col].gt(0)].copy()
            global_denominator = float(active[weight_col].sum())
            for scope, local in [
                ("global", active),
                *[
                    (f"side_{side}", side_rows)
                    for side, side_rows in active.groupby("side_name", sort=True)
                ],
            ]:
                weights = local[weight_col].to_numpy(dtype=float)
                baseline = weighted_mean(local["enter_now_net"], weights)
                for policy in POLICIES:
                    wait = route(policy, local, threshold)
                    net = np.where(wait, local["wait10_net"], local["enter_now_net"])
                    gross = np.where(
                        wait, local["wait10_gross"], local["enter_now_gross"]
                    )
                    cost = np.where(
                        wait, local["wait10_cost"], local["enter_now_cost"]
                    )
                    numerator = float(np.sum(net * weights))
                    records.append(
                        {
                            "training_source": training_source,
                            "feature_set": feature_set,
                            "evaluation": evaluation,
                            "top_fraction": fraction,
                            "scope": scope,
                            "policy": policy,
                            "rows": int(len(local)),
                            "expected_selected_rows": float(weights.sum()),
                            "net_bps": weighted_mean(net, weights) * 10_000.0,
                            "gross_bps": weighted_mean(gross, weights) * 10_000.0,
                            "cost_bps": weighted_mean(cost, weights) * 10_000.0,
                            "delta_vs_enter_now_bps": (
                                weighted_mean(net, weights) - baseline
                            )
                            * 10_000.0,
                            "wait_rate": weighted_mean(wait.astype(float), weights),
                            "global_book_side_contribution_bps": (
                                numerator / global_denominator * 10_000.0
                                if global_denominator > 0.0
                                else np.nan
                            ),
                        }
                    )
    return pd.DataFrame(records)


def head_metrics(rows: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for keys, part in rows.groupby(
        ["training_source", "feature_set", "evaluation", "side_name"], sort=True
    ):
        source, feature_set, evaluation, side = keys
        event = part["wait_delta"].gt(0.0).to_numpy()
        probability = part["pred_event_probability"].to_numpy(dtype=float)
        delta = part["wait_delta"].to_numpy(dtype=float)
        records.append(
            {
                "training_source": source,
                "feature_set": feature_set,
                "evaluation": evaluation,
                "side_name": side,
                "rows": int(len(part)),
                "better_rate": float(event.mean()),
                "event_auc": (
                    float(roc_auc_score(event, probability))
                    if np.unique(event).size == 2
                    else np.nan
                ),
                "event_brier": float(brier_score_loss(event, probability)),
                "direct_mae_bps": float(
                    np.mean(np.abs(part["pred_direct_delta"] - delta)) * 10_000.0
                ),
                "direct_spearman": float(
                    pd.Series(part["pred_direct_delta"]).corr(
                        pd.Series(delta), method="spearman"
                    )
                ),
                "expected_spearman": float(
                    pd.Series(part["pred_expected_delta"]).corr(
                        pd.Series(delta), method="spearman"
                    )
                ),
                "q25_empirical_below_rate": float(
                    np.mean(delta < part["pred_q25_delta"].to_numpy(dtype=float))
                ),
            }
        )
    return pd.DataFrame(records)


def bootstrap_top10(rows: pd.DataFrame, draws: int = 1_000) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for keys, part in rows.groupby(
        ["training_source", "feature_set", "evaluation"], sort=True
    ):
        source, feature_set, evaluation = keys
        active = part.loc[part["weight_top_10"].gt(0)].copy()
        active["day"] = pd.to_datetime(
            active["execution_decision_utc"], utc=True
        ).dt.floor("D")
        days = sorted(active["day"].unique())
        if not days:
            continue
        seed = 20260730 + sum(map(ord, f"{source}|{feature_set}|{evaluation}"))
        draw_index = np.random.default_rng(seed).integers(
            0, len(days), size=(draws, len(days))
        )
        threshold = float(active["calibrated_threshold"].iloc[0])
        for policy in POLICIES:
            wait = route(policy, active, threshold)
            active["_weighted_delta"] = active["weight_top_10"] * np.where(
                wait, active["wait_delta"], 0.0
            )
            active["_weight"] = active["weight_top_10"]
            daily = (
                active.groupby("day")[["_weighted_delta", "_weight"]]
                .sum()
                .reindex(days, fill_value=0.0)
            )
            samples = (
                daily["_weighted_delta"].to_numpy()[draw_index].sum(axis=1)
                / daily["_weight"].to_numpy()[draw_index].sum(axis=1)
            )
            records.append(
                {
                    "training_source": source,
                    "feature_set": feature_set,
                    "evaluation": evaluation,
                    "policy": policy,
                    "days": len(days),
                    "draws": draws,
                    "delta_ci_low_bps": float(np.quantile(samples, 0.025) * 10_000.0),
                    "delta_ci_high_bps": float(np.quantile(samples, 0.975) * 10_000.0),
                }
            )
    return pd.DataFrame(records)


def training_sources(rows: pd.DataFrame) -> Mapping[str, tuple[pd.DataFrame, tuple[str, ...]]]:
    february = rows.loc[rows["candidate_month"].eq("2025-02")].copy()
    february_tail = february.loc[
        february["base_rank_pct_timestamp_side"].le(0.20)
    ].copy()
    april_start = pd.Timestamp("2025-04-01", tz="UTC")
    march_resolved = rows.loc[
        rows["candidate_month"].eq("2025-03")
        & rows["execution_label_end_utc"].lt(april_start)
    ].copy()
    return {
        "february_all": (february, ("2025-03", "2025-04")),
        "february_base_rank_top_half": (
            february_tail,
            ("2025-03", "2025-04"),
        ),
        "march_all_resolved": (march_resolved, ("2025-04",)),
        "february_plus_march_resolved": (
            pd.concat([february, march_resolved], ignore_index=True),
            ("2025-04",),
        ),
    }


def run(
    training_root: Path = TRAINING_ROOT,
    handoff_root: Path = HANDOFF_ROOT,
    prior_root: Path = PRIOR_RESULT_ROOT,
    output: Path = OUT,
) -> dict[str, Any]:
    training_manifest = verify_artifact(
        training_root, "febapr2025_current_policy_wait10_action_v1"
    )
    handoff_manifest = verify_artifact(
        handoff_root, "frozen_entry_action_handoff_v2"
    )
    prior_manifest = verify_artifact(
        prior_root, "frozen_preentry_wait10_action_ablation_v2"
    )
    if output.exists():
        raise FileExistsError(output)

    labels = pd.read_parquet(training_root / "action_labels.parquet")
    features = pd.read_parquet(training_root / "preentry_features.parquet")
    rows = labels.merge(features, on=list(IDENTITY), how="inner", validate="one_to_one")
    if len(rows) != int(training_manifest["rows"]):
        raise AblationError("training feature/label coverage changed")
    rows["execution_decision_utc"] = pd.to_datetime(
        rows["execution_decision_utc"], utc=True
    )
    rows["execution_label_end_utc"] = pd.to_datetime(
        rows["execution_label_end_utc"], utc=True
    )

    handoff = pd.read_parquet(
        handoff_root / "handoff.parquet",
        columns=[*IDENTITY, "candidate_month", *WEIGHTS],
    )
    evaluation = handoff.merge(
        rows,
        on=list(IDENTITY),
        how="left",
        validate="one_to_one",
        suffixes=("_frozen", ""),
    )
    if len(evaluation) != len(handoff) or evaluation["wait_delta"].isna().any():
        raise AblationError("frozen book is not fully covered by older action labels")
    if not evaluation["candidate_month_frozen"].eq(evaluation["candidate_month"]).all():
        raise AblationError("frozen candidate month changed")
    evaluation = evaluation.drop(columns=["candidate_month_frozen"])

    prior = pd.read_parquet(prior_root / "action_predictions.parquet")
    prior = prior.loc[prior["feature_set"].eq("compact")].drop_duplicates("candidate_id")
    parity = evaluation.loc[
        :, ["candidate_id", "enter_now_net", "wait10_net", "wait_delta"]
    ].merge(
        prior.loc[
            :, ["candidate_id", "enter_now_net", "wait10_net", "wait_delta"]
        ],
        on="candidate_id",
        validate="one_to_one",
        suffixes=("_historical", "_prior"),
    )
    parity_records = []
    for field in ("enter_now_net", "wait10_net", "wait_delta"):
        delta = np.abs(
            parity[f"{field}_historical"].to_numpy(dtype=float)
            - parity[f"{field}_prior"].to_numpy(dtype=float)
        )
        parity_records.append(
            {
                "field": field,
                "rows": len(delta),
                "mismatch_rows": int((delta > 1e-12).sum()),
                "max_abs_delta": float(delta.max(initial=0.0)),
            }
        )
    parity_summary = pd.DataFrame(parity_records)
    if parity_summary["mismatch_rows"].sum() != 0:
        raise AblationError("new historical labels differ from the sealed frozen-book labels")

    predictions: list[pd.DataFrame] = []
    calibration_records: list[dict[str, Any]] = []
    for source_index, (source_name, (source_rows, evaluation_months)) in enumerate(
        training_sources(rows).items()
    ):
        for feature_index, (feature_name, feature_names) in enumerate(FEATURE_SETS.items()):
            for side_index, side in enumerate(("long", "short")):
                train = source_rows.loc[source_rows["side_name"].eq(side)].copy()
                core_indices, calibration_indices = calibration_split(train)
                core = train.iloc[core_indices].copy()
                calibration = train.iloc[calibration_indices].copy()
                x_core, core_medians = prepare_x(core, feature_names)
                x_calibration, _ = prepare_x(
                    calibration, feature_names, core_medians
                )
                seed = (
                    20260730
                    + source_index * 1_000
                    + feature_index * 100
                    + side_index * 10
                )
                calibration_models = fit_bundle(
                    x_core, core["wait_delta"].to_numpy(dtype=float), seed
                )
                calibration_predictions = predict_bundle(
                    calibration_models, x_calibration
                )
                calibration_scored = pd.concat(
                    [
                        calibration.loc[
                            :, ["execution_decision_utc", "wait_delta"]
                        ].reset_index(drop=True),
                        calibration_predictions,
                    ],
                    axis=1,
                )
                threshold, threshold_audit = choose_weighted_threshold(
                    calibration_scored
                )
                calibration_records.append(
                    {
                        "training_source": source_name,
                        "feature_set": feature_name,
                        "side_name": side,
                        "source_rows": int(len(train)),
                        "core_rows": int(len(core)),
                        "calibration_rows": int(len(calibration)),
                        "core_label_end_max_utc": core["execution_label_end_utc"].max(),
                        "calibration_start_utc": calibration[
                            "execution_decision_utc"
                        ].min(),
                        "strict_resolution_before_calibration": bool(
                            core["execution_label_end_utc"].max()
                            < calibration["execution_decision_utc"].min()
                        ),
                        "selected_threshold": threshold,
                        "threshold_selection": threshold_audit["selection"],
                        "threshold_audit": json.dumps(safe(threshold_audit), sort_keys=True),
                    }
                )

                x_train, medians = prepare_x(train, feature_names)
                final_models = fit_bundle(
                    x_train, train["wait_delta"].to_numpy(dtype=float), seed + 50
                )
                for month in evaluation_months:
                    valid = evaluation.loc[
                        evaluation["candidate_month"].eq(month)
                        & evaluation["side_name"].eq(side)
                    ].copy()
                    x_valid, _ = prepare_x(valid, feature_names, medians)
                    prediction = predict_bundle(final_models, x_valid)
                    scored = pd.concat(
                        [valid.reset_index(drop=True), prediction], axis=1
                    )
                    scored["training_source"] = source_name
                    scored["feature_set"] = feature_name
                    scored["evaluation"] = f"{month}_frozen_global_book"
                    scored["calibrated_threshold"] = threshold
                    predictions.append(scored)

    ledger = pd.concat(predictions, ignore_index=True)
    prediction_columns = [
        "pred_direct_delta",
        "pred_q25_delta",
        "pred_event_probability",
        "pred_weighted_event_score",
        "pred_positive_delta",
        "pred_negative_delta",
        "pred_soft_score",
        "pred_expected_delta",
    ]
    if not np.isfinite(ledger[prediction_columns].to_numpy(dtype=float)).all():
        raise AblationError("action predictions contain non-finite values")
    calibration = pd.DataFrame(calibration_records)
    if not calibration["strict_resolution_before_calibration"].all():
        raise AblationError("training calibration uses unresolved labels")
    metric_frame = economics(ledger)
    head_frame = head_metrics(ledger)
    bootstrap = bootstrap_top10(ledger)

    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        parity_summary.to_csv(temporary / "label_parity.csv", index=False)
        calibration.to_csv(temporary / "calibration_audit.csv", index=False)
        metric_frame.to_csv(temporary / "policy_metrics.csv", index=False)
        head_frame.to_csv(temporary / "head_metrics.csv", index=False)
        bootstrap.to_csv(temporary / "daily_bootstrap_ci_top10.csv", index=False)
        retained = [
            *IDENTITY,
            "candidate_month",
            "execution_decision_utc",
            *WEIGHTS,
            "enter_now_gross",
            "enter_now_cost",
            "enter_now_net",
            "wait10_gross",
            "wait10_cost",
            "wait10_net",
            "wait_delta",
            "training_source",
            "feature_set",
            "evaluation",
            "calibrated_threshold",
            *prediction_columns,
        ]
        ledger.loc[:, retained].to_parquet(
            temporary / "action_predictions.parquet",
            index=False,
            compression="zstd",
        )
        outputs = {
            name: sha256(temporary / name)
            for name in (
                "label_parity.csv",
                "calibration_audit.csv",
                "policy_metrics.csv",
                "head_metrics.csv",
                "daily_bootstrap_ci_top10.csv",
                "action_predictions.parquet",
            )
        }
        manifest = {
            "schema": SCHEMA,
            "status": "SEALED_DIAGNOSTIC_OLDER_TRAINING_FROZEN_BOOK_NO_PROMOTION",
            "contract": {
                "training": "side-local February or causally resolved March exact-current-policy labels; train-only chronological calibration split",
                "evaluation": "unchanged March/April frozen pooled-global books and fractional top-1/5/10/20 weights; no rerank, backfill or sizing change",
                "feature_ablation": "base context; base plus explicit regime-transition context; all 34 causal state/transition inputs",
                "heads": "event probability, economically weighted event score, direct delta, conditional positive/negative magnitude, q25 delta lower bound, and soft delta",
                "abstention": "calibration threshold is admitted only when its train-only day-cluster 90% lower bound is positive; otherwise the action rule abstains",
                "selection": "no configuration may be selected or promoted from reused March/April results",
            },
            "training_sources": list(training_sources(rows)),
            "feature_sets": {key: list(value) for key, value in FEATURE_SETS.items()},
            "policies": list(POLICIES),
            "prediction_rows": int(len(ledger)),
            "frozen_rows": int(len(evaluation)),
            "frozen_identity_sha256": identity_digest(evaluation),
            "promotion_eligible": False,
            "portfolio_replay_authorized": False,
            "input_provenance": {
                "training_manifest_sha256": sha256(training_root / "manifest.json"),
                "training_labels_sha256": training_manifest["outputs_sha256"][
                    "action_labels.parquet"
                ],
                "training_features_sha256": training_manifest["outputs_sha256"][
                    "preentry_features.parquet"
                ],
                "handoff_manifest_sha256": sha256(handoff_root / "manifest.json"),
                "handoff_sha256": handoff_manifest["outputs_sha256"]["handoff.parquet"],
                "prior_result_manifest_sha256": sha256(prior_root / "manifest.json"),
                "prior_predictions_sha256": prior_manifest["outputs_sha256"][
                    "action_predictions.parquet"
                ],
            },
            "outputs_sha256": outputs,
            "runner": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256(Path(__file__).resolve()),
            },
        }
        write_json(temporary / "manifest.json", manifest)
        (temporary / "manifest.sha256").write_text(
            f"{sha256(temporary / 'manifest.json')}  manifest.json\n"
        )
        os.replace(temporary, output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return manifest


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--training-root", type=Path, default=TRAINING_ROOT)
    result.add_argument("--handoff-root", type=Path, default=HANDOFF_ROOT)
    result.add_argument("--prior-root", type=Path, default=PRIOR_RESULT_ROOT)
    result.add_argument("--output", type=Path, default=OUT)
    return result


def main() -> None:
    args = parser().parse_args()
    print(
        json.dumps(
            safe(run(args.training_root, args.handoff_root, args.prior_root, args.output)),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
