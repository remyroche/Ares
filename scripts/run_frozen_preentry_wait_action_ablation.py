#!/usr/bin/env python3
"""Learn a side-local wait-10m action without changing the frozen global book.

The current deployed simple-policy simulator is the label engine.  Enter-now
must reproduce the sealed deployed control exactly before any wait label or
model result is accepted.  March is chronological OOF diagnostics; a frozen
March refit scores April forward.  Neither reused month is promotion evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
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
HANDOFF_ROOT = ROOT / "data_perp/artifacts/frozen_entry_action_handoff_20260730_v2"
POLICY_PATH = (
    ROOT
    / "data_perp/reports/simple_policy_1m_joint_trailing_raw_bayesian_champion_20260718_v1"
    / "production_staging/best_policy_params.json"
)
LABEL_ROOT = (
    ROOT
    / "data_perp/artifacts/febapr2025_execution_ev_current_spread_12h_labels_20260727_v1"
)
OUT = ROOT / "data_perp/artifacts/frozen_preentry_wait10_action_ablation_20260730_v2"
SCHEMA = "frozen_preentry_wait10_action_ablation_v2"
IDENTITY = ("candidate_id", "side_name", "__symbol__", "__ts__")
WEIGHTS = ("weight_top_01", "weight_top_05", "weight_top_10", "weight_top_20")
TOPS = (0.01, 0.05, 0.10, 0.20)
WAIT_MINUTES = 10
COMPACT_FEATURES = (
    "raw_score",
    "score_base_alpha",
    "score_residual_expected_ev",
    "direct_q25_return",
    "base_oof_score",
    "base_rank_pct_timestamp_side",
    "base_score_z_timestamp_side",
    "base_margin_to_top40_cutoff_z",
    "base_rank_pct_timestamp_global",
    "base_score_z_timestamp_global",
    "range_24h_pct",
    "__meta_raw__volatility_zscore",
    "trend_r2_24",
    "jump_intensity",
    "__meta_raw__chop_score",
    "preentry_transition__range_24h_pct__delta_12h",
    "preentry_transition__meta_raw__volatility_zscore__delta_12h",
    "preentry_transition__trend_r2_24__delta_12h",
    "preentry_transition__jump_intensity__delta_12h",
    "preentry_transition__meta_raw__chop_score__delta_12h",
    "__regime_source_execution_quality_score__",
    "__regime_source_execution_risk_score__",
    "__regime_source_dirty_shock_avoid_score__",
    "__regime_source_clean_execution_context_score__",
)
POLICIES = ("enter_now", "always_wait10", "oracle_wait10", "direct", "decomposed", "soft")
PARITY_FIELDS = (
    "execution_gross_ev_12h",
    "execution_net_ev_12h",
    "execution_exit_hour",
    "execution_mfe_return_12h",
    "execution_mae_return_12h",
    "execution_entry_price",
    "execution_exit_price",
    "execution_expected_spread_bps",
    "execution_entry_half_spread_bps",
    "execution_exit_half_spread_bps",
)


class ContractError(RuntimeError):
    pass


class ConstantRegressor:
    def __init__(self, value: float):
        self.value = float(value)

    def predict(self, values: pd.DataFrame) -> np.ndarray:
        return np.full(len(values), self.value, dtype=float)


class ConstantClassifier:
    def __init__(self, probability: float):
        self.probability = float(probability)

    def predict_proba(self, values: pd.DataFrame) -> np.ndarray:
        probability = float(np.clip(self.probability, 0.0, 1.0))
        return np.column_stack(
            [np.full(len(values), 1.0 - probability), np.full(len(values), probability)]
        )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(safe(payload), indent=2, sort_keys=True) + "\n")


def verify_handoff(root: Path) -> dict[str, Any]:
    manifest_path = root / "manifest.json"
    seal_path = root / "manifest.sha256"
    if not manifest_path.is_file() or not seal_path.is_file():
        raise ContractError("sealed v2 handoff is required")
    if sha256(manifest_path) != seal_path.read_text().split()[0]:
        raise ContractError("handoff manifest seal mismatch")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != "frozen_entry_action_handoff_v2":
        raise ContractError("handoff schema mismatch")
    for name, expected in manifest["outputs_sha256"].items():
        if sha256(root / name) != expected:
            raise ContractError(f"handoff output mismatch: {name}")
    return manifest


def parse_paths(payloads: pd.Series) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
    timestamps = np.empty((len(payloads), 720), dtype=np.int64)
    arrays = tuple(np.empty((len(payloads), 720), dtype=np.float32) for _ in range(4))
    for position, payload in enumerate(payloads.astype(str)):
        parsed = json.loads(payload)
        timestamp = np.asarray(parsed["timestamp"], dtype=np.int64)
        values = tuple(
            np.asarray(parsed[name], dtype=np.float32)
            for name in ("open", "high", "low", "close")
        )
        if (
            timestamp.shape != (720,)
            or any(value.shape != (720,) for value in values)
            or any(not np.isfinite(value).all() for value in values)
            or not np.all(np.diff(timestamp) == 60_000_000_000)
        ):
            raise ContractError("every action path must be contiguous finite 720x1m")
        for target, value in zip(arrays, values):
            target[position] = value
        timestamps[position] = timestamp
    return timestamps, arrays


def wait_slice(
    timestamps: np.ndarray,
    arrays: tuple[np.ndarray, ...],
    *,
    wait_minutes: int,
) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
    if wait_minutes < 1 or wait_minutes >= timestamps.shape[1]:
        raise ContractError("wait offset must leave at least one executable bar")
    return timestamps[:, wait_minutes:], tuple(
        array[:, wait_minutes:] for array in arrays
    )


def _simulate(
    rows: pd.DataFrame,
    arrays: tuple[np.ndarray, ...],
    policy: Mapping[str, Any],
    *,
    wait_minutes: int,
) -> pd.DataFrame:
    from scripts.materialize_execution_ev_policy_labels import (
        _resolved_geometry,
        _simulate_batch,
    )
    from scripts.run_frozen_exit_state_action_ablation import _strategy_lookup

    candidates = rows.copy()
    candidates["__symbol__"] = candidates["path_symbol"].astype(str)
    candidates["__raw_policy_archetype__"] = candidates["policy_archetype"].astype(str)
    candidates, _ = _resolved_geometry(candidates, policy)
    if wait_minutes:
        candidates["__decision_ts__"] = pd.to_datetime(
            candidates["__decision_ts__"], utc=True
        ) + pd.Timedelta(minutes=wait_minutes)
        action_arrays = tuple(array[:, wait_minutes:] for array in arrays)
    else:
        action_arrays = arrays
    strategies = _strategy_lookup(policy)
    output = pd.DataFrame(index=np.arange(len(candidates)))
    for geometry_key, indices in candidates.groupby(
        "execution_geometry_key", sort=True
    ).groups.items():
        positions = np.asarray(list(indices), dtype=int)
        simulated = _simulate_batch(
            candidates.iloc[positions],
            tuple(array[positions] for array in action_arrays),
            strategies[str(geometry_key)],
        )
        output.loc[positions, simulated.columns] = simulated.to_numpy()
        output.loc[positions, "execution_geometry_key"] = str(geometry_key)
    categorical = {"execution_exit_reason", "execution_geometry_key"}
    for column in output.columns:
        if column not in categorical:
            output[column] = pd.to_numeric(output[column], errors="coerce")
    if output.drop(columns=list(categorical)).isna().any().any():
        raise ContractError("simple-policy action replay produced non-finite output")
    return output


def _fit_regressor(x: pd.DataFrame, y: Sequence[float], seed: int) -> Any:
    target = np.asarray(y, dtype=float)
    finite = np.isfinite(target)
    if finite.sum() < 20 or float(np.nanstd(target[finite])) <= 1e-12:
        return ConstantRegressor(float(np.nanmean(target[finite])) if finite.any() else 0.0)
    import lightgbm as lgb

    model = lgb.LGBMRegressor(
        objective="regression_l1",
        n_estimators=180,
        learning_rate=0.035,
        num_leaves=15,
        max_depth=4,
        min_child_samples=40,
        reg_alpha=0.12,
        reg_lambda=5.0,
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
    model.fit(x.iloc[np.flatnonzero(finite)], target[finite])
    return model


def _fit_classifier(x: pd.DataFrame, y: Sequence[bool], seed: int) -> Any:
    target = np.asarray(y, dtype=np.int8)
    if np.unique(target).size < 2:
        return ConstantClassifier(float(target[0]) if len(target) else 0.0)
    import lightgbm as lgb

    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=180,
        learning_rate=0.035,
        num_leaves=15,
        max_depth=4,
        min_child_samples=40,
        reg_alpha=0.12,
        reg_lambda=5.0,
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
    model.fit(x, target)
    return model


def fit_heads(x: pd.DataFrame, delta: np.ndarray, seed: int) -> dict[str, Any]:
    positive = delta > 0.0
    scale = 25.0
    soft = 1.0 / (1.0 + np.exp(-np.clip(delta * 10_000.0 / scale, -40.0, 40.0)))
    return {
        "direct": _fit_regressor(x, delta, seed),
        "better": _fit_classifier(x, positive, seed + 1),
        "positive": _fit_regressor(x.loc[positive], delta[positive], seed + 2),
        "negative": _fit_regressor(x.loc[~positive], -delta[~positive], seed + 3),
        "soft": _fit_regressor(x, soft, seed + 4),
    }


def predict_heads(models: Mapping[str, Any], x: pd.DataFrame) -> dict[str, np.ndarray]:
    probability = np.asarray(models["better"].predict_proba(x), dtype=float)[:, 1]
    positive = np.maximum(np.asarray(models["positive"].predict(x), dtype=float), 0.0)
    negative = np.maximum(np.asarray(models["negative"].predict(x), dtype=float), 0.0)
    return {
        "pred_direct_delta": np.asarray(models["direct"].predict(x), dtype=float),
        "pred_better_probability": np.clip(probability, 0.0, 1.0),
        "pred_positive_delta": positive,
        "pred_negative_delta": negative,
        "pred_decomposed_delta": probability * positive - (1.0 - probability) * negative,
        "pred_soft_score": np.clip(
            np.asarray(models["soft"].predict(x), dtype=float), 0.0, 1.0
        ),
    }


def route_wait(policy: str, prediction: pd.DataFrame, delta: np.ndarray) -> np.ndarray:
    if policy == "enter_now":
        return np.zeros(len(prediction), dtype=bool)
    if policy == "always_wait10":
        return np.ones(len(prediction), dtype=bool)
    if policy == "oracle_wait10":
        return np.asarray(delta, dtype=float) > 0.0
    if policy == "direct":
        return prediction["pred_direct_delta"].to_numpy(dtype=float) > 0.0
    if policy == "decomposed":
        return prediction["pred_decomposed_delta"].to_numpy(dtype=float) > 0.0
    if policy == "soft":
        return prediction["pred_soft_score"].to_numpy(dtype=float) > 0.5
    raise ContractError(f"unknown routing policy: {policy}")


def weighted_mean(values: Sequence[float], weights: Sequence[float]) -> float:
    value = np.asarray(values, dtype=float)
    weight = np.asarray(weights, dtype=float)
    denominator = float(weight.sum())
    if denominator <= 0.0:
        return np.nan
    return float(np.sum(value * weight) / denominator)


def _metric_records(rows: pd.DataFrame) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for feature_set, feature_rows in rows.groupby("feature_set", sort=True):
        for evaluation, evaluation_rows in feature_rows.groupby("evaluation", sort=True):
            for fraction in TOPS:
                weight_col = f"weight_top_{int(fraction * 100):02d}"
                active = evaluation_rows.loc[evaluation_rows[weight_col].gt(0)]
                scopes = [("global", active)] + [
                    (f"side_{side}", local)
                    for side, local in active.groupby("side_name", sort=True)
                ]
                for scope, local in scopes:
                    weight = local[weight_col].to_numpy(dtype=float)
                    baseline = weighted_mean(local["enter_now_net"], weight)
                    for policy in POLICIES:
                        wait = route_wait(
                            policy, local, local["wait_delta"].to_numpy(dtype=float)
                        )
                        utility = np.where(
                            wait,
                            local["wait10_net"].to_numpy(dtype=float),
                            local["enter_now_net"].to_numpy(dtype=float),
                        )
                        records.append(
                            {
                                "feature_set": feature_set,
                                "evaluation": evaluation,
                                "top_fraction": fraction,
                                "scope": scope,
                                "policy": policy,
                                "rows": int(len(local)),
                                "expected_selected_rows": float(weight.sum()),
                                "net_bps": weighted_mean(utility, weight) * 10_000.0,
                                "gross_bps": weighted_mean(
                                    np.where(
                                        wait,
                                        local["wait10_gross"].to_numpy(dtype=float),
                                        local["enter_now_gross"].to_numpy(dtype=float),
                                    ),
                                    weight,
                                )
                                * 10_000.0,
                                "cost_bps": weighted_mean(
                                    np.where(
                                        wait,
                                        local["wait10_cost"].to_numpy(dtype=float),
                                        local["enter_now_cost"].to_numpy(dtype=float),
                                    ),
                                    weight,
                                )
                                * 10_000.0,
                                "delta_vs_enter_now_bps": (
                                    weighted_mean(utility, weight) - baseline
                                )
                                * 10_000.0,
                                "wait_rate": weighted_mean(wait.astype(float), weight),
                                "positive_rate": weighted_mean(
                                    (utility > 0.0).astype(float), weight
                                ),
                                "fraction_improved": weighted_mean(
                                    (
                                        local["wait_delta"].to_numpy(dtype=float) > 0.0
                                    ).astype(float),
                                    weight,
                                ),
                            }
                        )
    return records


def _head_records(rows: pd.DataFrame) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for keys, part in rows.groupby(
        ["feature_set", "evaluation", "side_name"], sort=True
    ):
        feature_set, evaluation, side = keys
        delta = part["wait_delta"].to_numpy(dtype=float)
        better = delta > 0.0
        probability = part["pred_better_probability"].to_numpy(dtype=float)
        records.append(
            {
                "feature_set": feature_set,
                "evaluation": evaluation,
                "side_name": side,
                "rows": int(len(part)),
                "better_rate": float(better.mean()),
                "better_auc": (
                    float(roc_auc_score(better, probability))
                    if np.unique(better).size == 2
                    else np.nan
                ),
                "better_brier": float(brier_score_loss(better, probability)),
                "direct_delta_mae_bps": float(
                    np.mean(np.abs(part["pred_direct_delta"].to_numpy() - delta))
                    * 10_000.0
                ),
                "direct_delta_spearman": float(
                    pd.Series(part["pred_direct_delta"].to_numpy()).corr(
                        pd.Series(delta), method="spearman"
                    )
                ),
                "decomposed_delta_mae_bps": float(
                    np.mean(np.abs(part["pred_decomposed_delta"].to_numpy() - delta))
                    * 10_000.0
                ),
                "decomposed_delta_spearman": float(
                    pd.Series(part["pred_decomposed_delta"].to_numpy()).corr(
                        pd.Series(delta), method="spearman"
                    )
                ),
                "soft_score_spearman": float(
                    pd.Series(part["pred_soft_score"].to_numpy()).corr(
                        pd.Series(delta), method="spearman"
                    )
                ),
            }
        )
    return records


def _bootstrap_records(rows: pd.DataFrame, draws: int = 2_000) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for keys, part in rows.groupby(["feature_set", "evaluation"], sort=True):
        feature_set, evaluation = keys
        for fraction in TOPS:
            weight_col = f"weight_top_{int(fraction * 100):02d}"
            active = part.loc[part[weight_col].gt(0)].copy()
            scopes = [("global", active)] + [
                (f"side_{side}", local.copy())
                for side, local in active.groupby("side_name", sort=True)
            ]
            for scope, local in scopes:
                local["day"] = pd.to_datetime(
                    local["execution_decision_utc"], utc=True
                ).dt.floor("D")
                days = sorted(local["day"].unique())
                if not days:
                    continue
                seed = (
                    20260730
                    + int(fraction * 100)
                    + sum(map(ord, str(evaluation)))
                    + sum(map(ord, scope))
                )
                draw_index = np.random.default_rng(seed).integers(
                    0, len(days), size=(draws, len(days))
                )
                for policy in POLICIES:
                    wait = route_wait(
                        policy, local, local["wait_delta"].to_numpy(dtype=float)
                    )
                    local["_delta"] = local[weight_col] * np.where(
                        wait, local["wait_delta"].to_numpy(dtype=float), 0.0
                    )
                    local["_den"] = local[weight_col]
                    daily = (
                        local.groupby("day")[["_delta", "_den"]]
                        .sum()
                        .reindex(days, fill_value=0.0)
                    )
                    sample = daily["_delta"].to_numpy()[draw_index].sum(axis=1) / daily[
                        "_den"
                    ].to_numpy()[draw_index].sum(axis=1)
                    records.append(
                        {
                            "feature_set": feature_set,
                            "evaluation": evaluation,
                            "top_fraction": fraction,
                            "scope": scope,
                            "policy": policy,
                            "days": len(days),
                            "draws": draws,
                            "delta_ci_low_bps": float(
                                np.quantile(sample, 0.025) * 10_000.0
                            ),
                            "delta_ci_high_bps": float(
                                np.quantile(sample, 0.975) * 10_000.0
                            ),
                        }
                    )
    return records


def run(
    handoff_root: Path = HANDOFF_ROOT,
    policy_path: Path = POLICY_PATH,
    label_root: Path = LABEL_ROOT,
    output: Path = OUT,
) -> dict[str, Any]:
    handoff_manifest = verify_handoff(handoff_root)
    if output.exists():
        raise FileExistsError(output)
    frame = pd.read_parquet(handoff_root / "handoff.parquet")
    identity_digest = hashlib.sha256(
        pd.util.hash_pandas_object(
            frame.loc[:, [*IDENTITY, *WEIGHTS]], index=False
        ).to_numpy(dtype=np.uint64).tobytes()
    ).hexdigest()
    if identity_digest != handoff_manifest.get("identity_weight_digest"):
        raise ContractError("frozen identity/weight digest changed")
    frame["execution_decision_utc"] = pd.to_datetime(
        frame["execution_decision_utc"], utc=True
    )
    frame = frame.sort_values(
        ["execution_decision_utc", "candidate_id"], kind="stable"
    ).reset_index(drop=True)
    if frame.duplicated(list(IDENTITY), keep=False).any():
        raise ContractError("handoff identities are not unique")
    roles = json.loads((handoff_root / "feature_roles.json").read_text())
    full_features = tuple(roles["model_inputs"])
    if not set(COMPACT_FEATURES).issubset(full_features):
        raise ContractError("compact feature contract is not contained in sealed inputs")
    policy = json.loads(policy_path.read_text())
    timestamps, arrays = parse_paths(frame["execution_future_path"])
    wait_timestamps, wait_arrays = wait_slice(
        timestamps, arrays, wait_minutes=WAIT_MINUTES
    )
    action_entry_utc = pd.to_datetime(
        wait_timestamps[:, 0], unit="ns", utc=True
    )
    expected_action_entry = frame["execution_decision_utc"] + pd.Timedelta(
        minutes=WAIT_MINUTES
    )
    if not pd.Series(action_entry_utc).reset_index(drop=True).eq(
        expected_action_entry.reset_index(drop=True)
    ).all():
        raise ContractError("wait path does not begin exactly at decision + 10m")
    if (
        wait_timestamps.shape[1] != 710
        or any(array.shape[1] != 710 for array in wait_arrays)
        or not np.all(np.diff(wait_timestamps, axis=1) == 60_000_000_000)
    ):
        raise ContractError("wait action must retain exactly 710 contiguous 1m bars")
    enter_now = _simulate(frame, arrays, policy, wait_minutes=0)
    wait10 = _simulate(frame, arrays, policy, wait_minutes=WAIT_MINUTES)

    label_manifest = json.loads((label_root / "manifest.json").read_text())
    label_record = label_manifest.get("output", {})
    if (
        label_manifest.get("schema") != "execution_ev_deployed_policy_1m_labels_v1"
        or sha256(label_root / "labels.parquet") != label_record.get("sha256")
    ):
        raise ContractError("deployed control labels do not verify")
    reference_columns = [*IDENTITY, "execution_geometry_key", *PARITY_FIELDS, "execution_exit_reason"]
    reference = pd.read_parquet(
        label_root / "labels.parquet", columns=reference_columns
    )
    if reference["candidate_id"].duplicated().any():
        raise ContractError("deployed reference candidate_id is not unique")
    reference = frame.loc[:, ["candidate_id"]].merge(
        reference, on="candidate_id", how="left", validate="one_to_one"
    )
    if reference["execution_geometry_key"].isna().any():
        raise ContractError("deployed control reference coverage is incomplete")
    parity_records: list[dict[str, Any]] = []
    for field in PARITY_FIELDS:
        delta = np.abs(
            enter_now[field].to_numpy(dtype=float)
            - reference[field].to_numpy(dtype=float)
        )
        parity_records.append(
            {
                "field": field,
                "max_abs_delta": float(delta.max()),
                "mismatch_rows": int((delta > 1e-12).sum()),
            }
        )
    for field in ("execution_exit_reason", "execution_geometry_key"):
        mismatch = (
            enter_now[field].astype(str).to_numpy()
            != reference[field].astype(str).to_numpy()
        )
        parity_records.append(
            {
                "field": field,
                "max_abs_delta": np.nan,
                "mismatch_rows": int(mismatch.sum()),
            }
        )
    parity = pd.DataFrame(parity_records)
    parity["passed"] = parity["mismatch_rows"].eq(0)
    if not parity["passed"].all():
        raise ContractError(f"enter-now current-policy parity failed:\n{parity}")

    frame["enter_now_gross"] = enter_now["execution_gross_ev_12h"].to_numpy(dtype=float)
    frame["enter_now_net"] = enter_now["execution_net_ev_12h"].to_numpy(dtype=float)
    frame["enter_now_cost"] = enter_now["execution_cost_return"].to_numpy(dtype=float)
    frame["wait10_gross"] = wait10["execution_gross_ev_12h"].to_numpy(dtype=float)
    frame["wait10_net"] = wait10["execution_net_ev_12h"].to_numpy(dtype=float)
    frame["wait10_cost"] = wait10["execution_cost_return"].to_numpy(dtype=float)
    for prefix in ("enter_now", "wait10"):
        reconciliation = np.abs(
            frame[f"{prefix}_gross"]
            - frame[f"{prefix}_cost"]
            - frame[f"{prefix}_net"]
        )
        if (reconciliation > 1e-12).any():
            raise ContractError(f"{prefix} cost is not deducted exactly once")
    frame["wait_delta"] = frame["wait10_net"] - frame["enter_now_net"]
    frame["wait_action_entry_utc"] = action_entry_utc
    frame["wait10_exit_reason"] = wait10["execution_exit_reason"].astype(str).to_numpy()
    frame["wait10_exit_hour_after_entry"] = wait10["execution_exit_hour"].to_numpy(dtype=float)

    march = frame.loc[frame["candidate_month"].eq("2025-03")].copy().reset_index(drop=True)
    april = frame.loc[frame["candidate_month"].eq("2025-04")].copy().reset_index(drop=True)
    if march.empty or april.empty:
        raise ContractError("both March training and April forward rows are required")
    from extreme_price_movements.execution_ev_meta import chronological_purged_splits

    folds = chronological_purged_splits(
        march,
        n_splits=3,
        min_train_size=500,
        decision_time_col="execution_decision_utc",
        label_end_time_col="execution_label_end_utc",
        horizon_hours=12.0,
        embargo_hours=12.0,
    )
    if not folds:
        raise ContractError("no March chronological purged OOF folds")

    ledgers: list[pd.DataFrame] = []
    audits: list[dict[str, Any]] = []
    for feature_set, features in (
        ("compact", COMPACT_FEATURES),
        ("full_authorized", full_features),
    ):
        march_prediction = pd.DataFrame(index=march.index)
        march_prediction["oof_fold"] = pd.Series(pd.NA, index=march.index, dtype="Int64")
        for split in folds:
            for side in ("long", "short"):
                train = np.asarray(
                    [
                        index
                        for index in split.train_indices
                        if march.iloc[index]["side_name"] == side
                    ],
                    dtype=int,
                )
                valid = np.asarray(
                    [
                        index
                        for index in split.validation_indices
                        if march.iloc[index]["side_name"] == side
                    ],
                    dtype=int,
                )
                if len(valid):
                    validation_start = pd.to_datetime(
                        march.loc[valid, "execution_decision_utc"], utc=True
                    ).min()
                    train = np.asarray(
                        [
                            index
                            for index in train
                            if pd.Timestamp(
                                march.iloc[index]["execution_label_end_utc"]
                            )
                            < validation_start
                        ],
                        dtype=int,
                    )
                if not len(valid) or len(train) < 40:
                    continue
                x_train = march.loc[train, list(features)].astype("float32")
                x_valid = march.loc[valid, list(features)].astype("float32")
                models = fit_heads(
                    x_train,
                    march.loc[train, "wait_delta"].to_numpy(dtype=float),
                    20260730 + int(split.fold) * 10 + (0 if side == "long" else 1),
                )
                prediction = predict_heads(models, x_valid)
                for name, values in prediction.items():
                    march_prediction.loc[valid, name] = values
                march_prediction.loc[valid, "oof_fold"] = int(split.fold)
                audits.append(
                    {
                        "feature_set": feature_set,
                        "fold": int(split.fold),
                        "side_name": side,
                        "train_rows": int(len(train)),
                        "validation_rows": int(len(valid)),
                        "train_label_resolved_max_utc": pd.to_datetime(
                            march.loc[train, "execution_label_end_utc"], utc=True
                        ).max(),
                        "validation_start_utc": pd.to_datetime(
                            march.loc[valid, "execution_decision_utc"], utc=True
                        ).min(),
                        "strict_label_resolution_before_validation": bool(
                            pd.to_datetime(
                                march.loc[train, "execution_label_end_utc"], utc=True
                            ).max()
                            < pd.to_datetime(
                                march.loc[valid, "execution_decision_utc"], utc=True
                            ).min()
                        ),
                    }
                )
        valid_oof = march_prediction["oof_fold"].notna()
        march_ledger = pd.concat(
            [
                march.loc[valid_oof].reset_index(drop=True),
                march_prediction.loc[valid_oof].reset_index(drop=True),
            ],
            axis=1,
        )
        march_ledger["feature_set"] = feature_set
        march_ledger["evaluation"] = "march_chronological_oof"
        ledgers.append(march_ledger)

        april_prediction = pd.DataFrame(index=april.index)
        for side in ("long", "short"):
            train = np.flatnonzero(march["side_name"].eq(side).to_numpy())
            valid = np.flatnonzero(april["side_name"].eq(side).to_numpy())
            models = fit_heads(
                march.loc[train, list(features)].astype("float32"),
                march.loc[train, "wait_delta"].to_numpy(dtype=float),
                20260830 + (0 if side == "long" else 1),
            )
            prediction = predict_heads(
                models, april.loc[valid, list(features)].astype("float32")
            )
            for name, values in prediction.items():
                april_prediction.loc[valid, name] = values
        april_ledger = pd.concat(
            [april.reset_index(drop=True), april_prediction.reset_index(drop=True)],
            axis=1,
        )
        april_ledger["feature_set"] = feature_set
        april_ledger["evaluation"] = "april_frozen_march_forward"
        ledgers.append(april_ledger)

    ledger = pd.concat(ledgers, ignore_index=True)
    prediction_columns = [
        "pred_direct_delta",
        "pred_better_probability",
        "pred_positive_delta",
        "pred_negative_delta",
        "pred_decomposed_delta",
        "pred_soft_score",
    ]
    if not np.isfinite(ledger[prediction_columns].to_numpy(dtype=float)).all():
        raise ContractError("action head predictions contain non-finite values")
    metrics = pd.DataFrame(_metric_records(ledger))
    heads = pd.DataFrame(_head_records(ledger))
    bootstrap = pd.DataFrame(_bootstrap_records(ledger))
    if not pd.DataFrame(audits)[
        "strict_label_resolution_before_validation"
    ].all():
        raise ContractError("March OOF labels are not strictly resolved before validation")
    promotion = pd.DataFrame(
        [
            {
                "gate": "reused_months_not_promotion_evidence",
                "passed": False,
                "reason": "March/April diagnose action learnability only; no policy/model/threshold may be promoted.",
            },
            {
                "gate": "portfolio_replay",
                "passed": False,
                "reason": "No candidate action has passed multi-block economics and uncertainty gates.",
            },
        ]
    )

    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        parity.to_csv(temporary / "control_parity.csv", index=False)
        metrics.to_csv(temporary / "policy_metrics.csv", index=False)
        heads.to_csv(temporary / "head_metrics.csv", index=False)
        bootstrap.to_csv(temporary / "daily_bootstrap_ci.csv", index=False)
        pd.DataFrame(audits).to_csv(temporary / "fold_audit.csv", index=False)
        promotion.to_csv(temporary / "promotion_gate.csv", index=False)
        retained = [
            *IDENTITY,
            "execution_decision_utc",
            "candidate_month",
            *WEIGHTS,
            "feature_set",
            "evaluation",
            "oof_fold",
            "enter_now_gross",
            "enter_now_net",
            "enter_now_cost",
            "wait10_gross",
            "wait10_net",
            "wait10_cost",
            "wait_delta",
            "wait_action_entry_utc",
            "wait10_exit_reason",
            "wait10_exit_hour_after_entry",
            *prediction_columns,
        ]
        ledger.loc[:, [name for name in retained if name in ledger]].to_parquet(
            temporary / "action_predictions.parquet", index=False, compression="zstd"
        )
        outputs = {
            name: sha256(temporary / name)
            for name in (
                "control_parity.csv",
                "policy_metrics.csv",
                "head_metrics.csv",
                "daily_bootstrap_ci.csv",
                "fold_audit.csv",
                "promotion_gate.csv",
                "action_predictions.parquet",
            )
        }
        manifest = {
            "schema": SCHEMA,
            "status": "SEALED_DIAGNOSTIC_ONLY_CURRENT_POLICY_PARITY_NO_RERANK_NO_PROMOTION",
            "contract": {
                "selection": "exact frozen pooled-global monthly top-1/5/10/20 identities and fractional weights; no rerank, backfill, or sizing change",
                "action": "enter-now versus wait-market 10m; wait enters at minute-10 open and retains the exact original barrier/side strategy over the remaining 710 minutes",
                "absolute_deadline_remaining_minutes": 710,
                "exit_policy": "current deployed simple-policy simulator; enter-now exact row-level gross/net/cost/reason parity is mandatory",
                "training": "side-local fixed-geometry models; March chronological OOF with 12h label resolution purge and embargo; full March frozen refit scores April",
                "heads": "direct delta regression; better-action classifier; positive/negative magnitude decomposition; 25bps-temperature soft binary regression",
                "thresholds": "predeclared zero expected-delta or 0.5 soft-score; no threshold tuning on reused months",
            },
            "rows": int(len(frame)),
            "march_rows": int(len(march)),
            "april_rows": int(len(april)),
            "identity_weight_digest": identity_digest,
            "feature_sets": {
                "compact": list(COMPACT_FEATURES),
                "full_authorized": list(full_features),
            },
            "promotion_eligible": False,
            "limitations": [
                "March/April are reused diagnostics and cannot select a deployable action.",
                "Wait10 is a market-delay action; adverse-limit price targeting remains a separate unimplemented action.",
                "No portfolio concurrency/exposure/asset-limit replay is claimed.",
            ],
            "input_provenance": {
                "handoff_manifest_sha256": sha256(handoff_root / "manifest.json"),
                "handoff_sha256": handoff_manifest["outputs_sha256"]["handoff.parquet"],
                "policy_sha256": sha256(policy_path),
                "label_manifest_sha256": sha256(label_root / "manifest.json"),
                "labels_sha256": label_record["sha256"],
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
    result.add_argument("--handoff-root", type=Path, default=HANDOFF_ROOT)
    result.add_argument("--policy-path", type=Path, default=POLICY_PATH)
    result.add_argument("--label-root", type=Path, default=LABEL_ROOT)
    result.add_argument("--output", type=Path, default=OUT)
    return result


def main() -> None:
    args = parser().parse_args()
    print(
        json.dumps(
            safe(run(args.handoff_root, args.policy_path, args.label_root, args.output)),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
