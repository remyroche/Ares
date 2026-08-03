#!/usr/bin/env python3
"""Bounded full-base opportunity ablation with an untouched April holdout.

This runner deliberately keeps the frozen base model in its intended alpha role.
It tests whether side-local opportunity heads can translate the frozen alpha
score into a common, cost-aware expected-net ranking.

Research contract
-----------------
* Input is the immutable 509,868-row canonical panel v2.
* Development rows are February and March execution labels resolved before
  2025-04-01.  The base model's longer native-label resolution is irrelevant
  here because its OOF score is an already-frozen decision-time input, not a
  target of this meta-head.
* April is scored once after feature, target, geometry, and mapper selection.
* Model OOF uses five contiguous timestamp blocks with 12-hour path purging.
  The complementary pre-April blocks may be on either side of a held block;
  this is static blocked CV, not walk-forward evidence.
* Models are side-local.  Their scores are converted to a common expected-net
  unit by a cross-fitted side-aware isotonic map shrunk toward a pooled map.
* Selection is one pooled global top-k with candidate_id tie-breaking.
* Timing, MAE, target-price, wait-action, causal mapped fields, and outcomes are
  forbidden model inputs.
* No portfolio replay is performed.  The output only records whether the
  economic gates would authorize a later frozen replay.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PANEL_ROOT = (
    ROOT
    / "data_perp/artifacts/canonical_opportunity_payoff_trust_panel_20260729_v2"
)
DEFAULT_OUTPUT = (
    ROOT
    / "data_perp/artifacts/canonical_full_base_opportunity_ablation_20260729_v1"
)

SCHEMA = "canonical_full_base_opportunity_ablation_v1"
SIDES = ("long", "short")
APRIL_FREEZE = pd.Timestamp("2025-04-01T00:00:00Z")
PURGE_HOURS = 12
N_FOLDS = 5
SEED = 20260729
FRACTIONS = (0.01, 0.05, 0.10, 0.20)
PRIMARY_ARMS = ("S0", "S1", "S1+R", "S1+B", "S1+R+B")
SENSITIVITY_ARMS = ("S1+R+B-no-DAE-GMM",)
TARGETS = ("hard0", "hard25", "soft", "direct_net")

IDENTITY = ("candidate_id", "side_name", "__symbol__", "__ts__")
TIME_COLUMNS = (
    "__decision_ts__",
    "execution_label_end_utc",
    "effective_label_resolution_utc",
)
TARGET_COLUMNS = (
    "opportunity_gross_above_cost_0bps",
    "opportunity_gross_above_cost_25bps",
    "execution_soft_positive_12h",
    "execution_net_ev_12h",
    "execution_gross_ev_12h",
    "execution_cost_return",
)

SCORE_CONTEXT = (
    "base_rank_pct_timestamp_side",
    "base_score_z_timestamp_side",
    "base_group_rows_timestamp_side",
    "base_margin_to_top40_cutoff_z",
    "base_rank_pct_timestamp_global",
    "base_score_z_timestamp_global",
    "base_group_rows_timestamp_global",
)

REGIME_LEVELS = (
    "range_24h_pct",
    "__meta_raw__volatility_zscore",
    "trend_r2_24",
    "jump_intensity",
    "__meta_raw__chop_score",
)

REGIME_TRANSITIONS = (
    "preentry_transition__range_24h_pct__delta_3h",
    "preentry_transition__range_24h_pct__delta_12h",
    "preentry_transition__meta_raw__volatility_zscore__delta_3h",
    "preentry_transition__meta_raw__volatility_zscore__delta_12h",
    "preentry_transition__trend_r2_24__delta_3h",
    "preentry_transition__trend_r2_24__delta_12h",
    "preentry_transition__jump_intensity__delta_3h",
    "preentry_transition__jump_intensity__delta_12h",
    "preentry_transition__meta_raw__chop_score__delta_3h",
    "preentry_transition__meta_raw__chop_score__delta_12h",
)

BASE_LONG = (
    "base_input__climax_decay",
    "base_input__cross_asset_corr_1h",
    "base_input__delta_stall_6",
    "base_input__dow_cos",
    "base_input__dow_sin",
    "base_input__eig_effective_rank__breakout_all",
    "base_input__eig_participation_ratio__breakout_all",
    "base_input__eth_btc_ret_1h",
    "base_input__fragmented_flush_recovery",
    "base_input__giveback_vol_units",
    "base_input__hour_cos",
    "base_input__hour_sin",
    "base_input__liquidation_onset_score",
    "base_input__mark_perp_dislocation",
    "base_input__mark_vs_perp_bps",
    "base_input__market_breadth_1h",
    "base_input__median_volume_z",
    "base_input__mkt_atr_expansion_1h",
    "base_input__pct_assets_above_ema_fast",
    "base_input__pct_assets_above_vwap",
    "base_input__prog_eff_12",
    "base_input__prog_eff_24",
    "base_input__q_iqr__amihud_z_peer_resid",
    "base_input__qv",
    "base_input__range_12h_pct",
    "base_input__regime_transition_entropy_48h",
    "base_input__rejection_proxy",
    "base_input__rvol_z_peer_resid",
    "base_input__z_r_24",
    "base_input__dae_b16_02",
    "base_input__gmm_ood_score",
)

BASE_SHORT = (
    "base_input__mark_perp_dislocation",
    "base_input__mark_vs_perp_bps",
    "base_input__climax_decay",
    "base_input__impact_12",
    "base_input__post_flush_leverage_rebuild",
    "base_input__shock_12h",
    "base_input__bb_pos_12",
    "base_input__liquidation_onset_score",
)

GEOMETRY_FEATURES = frozenset(
    ("base_input__dae_b16_02", "base_input__gmm_ood_score")
)

# Fields that may exist in the panel but can never be model inputs.
FORBIDDEN_FEATURE_PREFIXES = (
    "mapped_",
    "causal_score_",
    "opportunity_",
    "execution_",
    "__first_touch_",
    "exit_",
)
FORBIDDEN_FEATURE_TOKENS = (
    "target_price",
    "wait_action",
    "timing",
    "label_resolution",
    "mfe",
    "mae",
)


@dataclass(frozen=True)
class Geometry:
    name: str
    iterations: int
    depth: int
    learning_rate: float
    l2_leaf_reg: float
    random_strength: float = 0.5
    bagging_temperature: float = 1.0


GEOMETRIES = (
    Geometry("fixed_d5", 300, 5, 0.04, 12.0),
    Geometry("compact_d4", 240, 4, 0.05, 20.0),
    Geometry("deep_d6", 360, 6, 0.03, 16.0),
)


@dataclass(frozen=True)
class Fold:
    fold_id: int
    validation_timestamps: tuple[pd.Timestamp, ...]
    validation_start: pd.Timestamp
    validation_end: pd.Timestamp


@dataclass
class IsotonicState:
    x: list[float]
    y: list[float]

    @classmethod
    def fit(cls, score: np.ndarray, target: np.ndarray) -> "IsotonicState":
        x = np.asarray(score, dtype=np.float64)
        y = np.asarray(target, dtype=np.float64)
        valid = np.isfinite(x) & np.isfinite(y)
        if valid.sum() < 2 or np.unique(x[valid]).size < 2:
            level = float(np.nanmean(y[valid])) if valid.any() else 0.0
            return cls(x=[-1.0e30, 1.0e30], y=[level, level])
        model = IsotonicRegression(increasing=True, out_of_bounds="clip")
        model.fit(x[valid], y[valid])
        return cls(
            x=[float(value) for value in model.X_thresholds_],
            y=[float(value) for value in model.y_thresholds_],
        )

    def predict(self, score: Sequence[float]) -> np.ndarray:
        values = np.asarray(score, dtype=np.float64)
        return np.interp(values, np.asarray(self.x), np.asarray(self.y))


@dataclass
class ShrunkMapper:
    pooled: IsotonicState
    by_side: dict[str, IsotonicState]
    side_support: dict[str, int]
    shrinkage_rows: int

    @classmethod
    def fit(
        cls,
        score: np.ndarray,
        side: Sequence[str],
        net: np.ndarray,
        *,
        shrinkage_rows: int,
    ) -> "ShrunkMapper":
        side_values = np.asarray(side, dtype=str)
        return cls(
            pooled=IsotonicState.fit(score, net),
            by_side={
                name: IsotonicState.fit(
                    np.asarray(score)[side_values == name],
                    np.asarray(net)[side_values == name],
                )
                for name in SIDES
            },
            side_support={
                name: int((side_values == name).sum()) for name in SIDES
            },
            shrinkage_rows=int(shrinkage_rows),
        )

    def predict(self, score: np.ndarray, side: Sequence[str]) -> np.ndarray:
        raw = np.asarray(score, dtype=np.float64)
        side_values = np.asarray(side, dtype=str)
        pooled = self.pooled.predict(raw)
        result = pooled.copy()
        for name in SIDES:
            mask = side_values == name
            if not mask.any():
                continue
            support = self.side_support[name]
            weight = support / (support + self.shrinkage_rows)
            side_prediction = self.by_side[name].predict(raw[mask])
            result[mask] = weight * side_prediction + (1.0 - weight) * pooled[mask]
        return result

    def to_dict(self) -> dict[str, Any]:
        return {
            "pooled": asdict(self.pooled),
            "by_side": {
                side: asdict(state) for side, state in self.by_side.items()
            },
            "side_support": self.side_support,
            "shrinkage_rows": self.shrinkage_rows,
        }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    return value


def identity_sha256(frame: pd.DataFrame) -> str:
    values = frame.loc[:, list(IDENTITY)].copy()
    values["__ts__"] = pd.to_datetime(values["__ts__"], utc=True).astype(str)
    values = values.astype(str).sort_values(list(IDENTITY), kind="stable")
    return hashlib.sha256(
        values.to_csv(index=False, lineterminator="\n").encode()
    ).hexdigest()


def target_values(frame: pd.DataFrame, target: str) -> np.ndarray:
    if target == "hard0":
        return frame["opportunity_gross_above_cost_0bps"].to_numpy(dtype=np.float64)
    if target == "hard25":
        return frame["opportunity_gross_above_cost_25bps"].to_numpy(
            dtype=np.float64
        )
    if target == "soft":
        return frame["execution_soft_positive_12h"].to_numpy(dtype=np.float64)
    if target == "direct_net":
        return frame["execution_net_ev_12h"].to_numpy(dtype=np.float64)
    raise KeyError(target)


def is_classifier_target(target: str) -> bool:
    return target != "direct_net"


def arm_features(arm: str, side: str) -> tuple[str, ...]:
    if side not in SIDES:
        raise ValueError(f"unknown side: {side}")
    base = BASE_LONG if side == "long" else BASE_SHORT
    features: list[str] = ["base_oof_score"]
    if arm != "S0":
        features.extend(SCORE_CONTEXT)
    if "+R" in arm:
        features.extend(REGIME_LEVELS)
        features.extend(REGIME_TRANSITIONS)
    if "+B" in arm:
        if arm.endswith("no-DAE-GMM"):
            base = tuple(name for name in base if name not in GEOMETRY_FEATURES)
        features.extend(base)
    result = tuple(dict.fromkeys(features))
    validate_feature_names(result)
    return result


def validate_feature_names(features: Sequence[str]) -> None:
    for name in features:
        lower = name.lower()
        if name.startswith(FORBIDDEN_FEATURE_PREFIXES) or any(
            token in lower for token in FORBIDDEN_FEATURE_TOKENS
        ):
            raise ValueError(f"forbidden model feature: {name}")


def required_columns() -> tuple[str, ...]:
    features: list[str] = []
    for arm in (*PRIMARY_ARMS, *SENSITIVITY_ARMS):
        for side in SIDES:
            features.extend(arm_features(arm, side))
    return tuple(
        dict.fromkeys((*IDENTITY, *TIME_COLUMNS, *TARGET_COLUMNS, *features))
    )


def load_panel(panel_root: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    manifest_path = panel_root / "manifest.json"
    sidecar = panel_root / "manifest.sha256"
    panel_path = panel_root / "panel.parquet"
    if not (manifest_path.exists() and sidecar.exists() and panel_path.exists()):
        raise FileNotFoundError("panel root lacks panel, manifest, or sidecar")
    manifest = json.loads(manifest_path.read_text())
    expected_manifest_hash = sidecar.read_text().split()[0]
    if sha256_file(manifest_path) != expected_manifest_hash:
        raise ValueError("panel manifest detached SHA256 mismatch")
    if manifest.get("schema") != "canonical_opportunity_payoff_trust_panel_v2":
        raise ValueError("unexpected input panel schema")
    if manifest.get("outputs_sha256", {}).get("panel.parquet") != sha256_file(
        panel_path
    ):
        raise ValueError("input panel SHA256 mismatch")
    frame = pd.read_parquet(panel_path, columns=list(required_columns()))
    for column in ("__ts__", *TIME_COLUMNS):
        frame[column] = pd.to_datetime(frame[column], utc=True, errors="raise")
    validate_panel(frame, manifest)
    return frame, manifest


def validate_panel(frame: pd.DataFrame, manifest: Mapping[str, Any]) -> None:
    if len(frame) != 509_868 or frame["candidate_id"].duplicated().any():
        raise ValueError("canonical full-base identity contract failed")
    if identity_sha256(frame) != manifest["identity_sha256"]:
        raise ValueError("canonical identity SHA256 mismatch")
    if not frame["side_name"].isin(SIDES).all():
        raise ValueError("unknown side")
    if not frame["__decision_ts__"].eq(
        frame["__ts__"] + pd.Timedelta(hours=1)
    ).all():
        raise ValueError("decision timestamp mismatch")
    if not frame["execution_label_end_utc"].eq(
        frame["__decision_ts__"] + pd.Timedelta(hours=12)
    ).all():
        raise ValueError("execution label horizon mismatch")
    if not np.allclose(
        frame["execution_gross_ev_12h"] - frame["execution_cost_return"],
        frame["execution_net_ev_12h"],
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("gross-cost-net reconciliation failed")
    if not np.allclose(
        frame["execution_soft_positive_12h"],
        1.0
        / (
            1.0
            + np.exp(
                -np.clip(
                    frame["execution_net_ev_12h"].to_numpy(dtype=float) / 0.01,
                    -60.0,
                    60.0,
                )
            )
        ),
        rtol=0.0,
        atol=5e-8,
    ):
        raise ValueError("existing soft target is not sigmoid(net/1pct)")
    for arm in (*PRIMARY_ARMS, *SENSITIVITY_ARMS):
        for side in SIDES:
            missing = set(arm_features(arm, side)).difference(frame.columns)
            if missing:
                raise ValueError(f"{arm}/{side} missing features: {sorted(missing)}")


def split_development_april(
    frame: pd.DataFrame,
    *,
    freeze: pd.Timestamp = APRIL_FREEZE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    month = frame["__ts__"].dt.strftime("%Y-%m")
    development = frame.loc[
        month.isin(("2025-02", "2025-03"))
        & frame["execution_label_end_utc"].lt(freeze)
    ].copy()
    april = frame.loc[month.eq("2025-04")].copy()
    if len(frame) == 509_868:
        if len(development) != 334_298:
            raise ValueError(
                f"expected 334298 resolved execution-label rows, got {len(development)}"
            )
        if len(april) != 172_450:
            raise ValueError(f"expected 172450 April rows, got {len(april)}")
        counts = development.groupby("side_name").size().to_dict()
        if counts != {"long": 167_149, "short": 167_149}:
            raise ValueError(f"unexpected development side counts: {counts}")
    if not development["execution_label_end_utc"].lt(freeze).all():
        raise ValueError("unresolved execution target entered development")
    return (
        development.sort_values(list(IDENTITY), kind="stable").reset_index(drop=True),
        april.sort_values(list(IDENTITY), kind="stable").reset_index(drop=True),
    )


def make_blocked_folds(
    frame: pd.DataFrame,
    *,
    n_folds: int = N_FOLDS,
) -> tuple[Fold, ...]:
    timestamps = pd.DatetimeIndex(frame["__ts__"].drop_duplicates().sort_values())
    if len(timestamps) < n_folds:
        raise ValueError("fewer timestamps than folds")
    blocks = np.array_split(timestamps.to_numpy(), n_folds)
    folds: list[Fold] = []
    for fold_id, values in enumerate(blocks):
        block = pd.DatetimeIndex(values)
        folds.append(
            Fold(
                fold_id=fold_id,
                validation_timestamps=tuple(pd.Timestamp(value) for value in block),
                validation_start=pd.Timestamp(block.min()),
                validation_end=pd.Timestamp(block.max()),
            )
        )
    return tuple(folds)


def fold_masks(
    frame: pd.DataFrame,
    fold: Fold,
    *,
    purge_hours: int = PURGE_HOURS,
) -> tuple[np.ndarray, np.ndarray]:
    timestamp = frame["__ts__"]
    validation = timestamp.isin(fold.validation_timestamps).to_numpy()
    if not validation.any():
        raise ValueError(f"fold {fold.fold_id} has no validation rows")
    if int(purge_hours) != 12:
        raise ValueError("the canonical execution target has an exact 12-hour path")
    # Use the actual decision/label timestamps instead of subtracting 12 hours
    # from the signal timestamp.  Decision is signal+1h, so a symmetric
    # signal-time 12h gap would incorrectly admit one boundary row whose label
    # resolves exactly at validation start.
    resolution = pd.to_datetime(
        frame["execution_label_end_utc"], utc=True, errors="raise"
    )
    decision = pd.to_datetime(frame["__decision_ts__"], utc=True, errors="raise")
    validation_label_end = resolution.loc[validation].max()
    earlier_resolved = resolution.lt(fold.validation_start).to_numpy()
    later_starts_after_path = decision.gt(validation_label_end).to_numpy()
    training = (~validation) & (earlier_resolved | later_starts_after_path)
    if not training.any():
        raise ValueError(f"fold {fold.fold_id} has no purged training rows")
    if not (
        resolution.loc[training].lt(fold.validation_start)
        | decision.loc[training].gt(validation_label_end)
    ).all():
        raise AssertionError("exact execution-path purge contract failed")
    return training, validation


def numeric_features(frame: pd.DataFrame, features: Sequence[str]) -> pd.DataFrame:
    # CatBoost handles NaN natively.  No global imputer or scaler is fitted.
    result = frame.loc[:, list(features)].apply(pd.to_numeric, errors="raise")
    if np.isinf(result.to_numpy(dtype=np.float64)).any():
        raise ValueError("infinite model feature")
    return result


def fit_predict_model(
    train_x: pd.DataFrame,
    train_y: np.ndarray,
    evaluation_x: pd.DataFrame,
    *,
    target: str,
    geometry: Geometry,
    seed: int,
    threads: int,
) -> tuple[np.ndarray, Any]:
    common = {
        "iterations": geometry.iterations,
        "depth": geometry.depth,
        "learning_rate": geometry.learning_rate,
        "l2_leaf_reg": geometry.l2_leaf_reg,
        "random_strength": geometry.random_strength,
        "bagging_temperature": geometry.bagging_temperature,
        "bootstrap_type": "Bayesian",
        "random_seed": seed,
        "thread_count": threads,
        "verbose": False,
        "allow_writing_files": False,
    }
    if is_classifier_target(target):
        from catboost import CatBoostClassifier

        model = CatBoostClassifier(loss_function="CrossEntropy", **common)
        model.fit(train_x, train_y)
        prediction = model.predict_proba(evaluation_x)[:, 1]
    else:
        from catboost import CatBoostRegressor

        model = CatBoostRegressor(loss_function="RMSE", **common)
        model.fit(train_x, train_y)
        prediction = model.predict(evaluation_x)
    return np.asarray(prediction, dtype=np.float64), model


def generate_side_local_oof(
    development: pd.DataFrame,
    folds: Sequence[Fold],
    *,
    arm: str,
    target: str,
    geometry: Geometry,
    threads: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    prediction = np.full(len(development), np.nan, dtype=np.float64)
    fold_assignment = np.full(len(development), -1, dtype=np.int16)
    y = target_values(development, target)
    for fold in folds:
        training, validation = fold_masks(development, fold)
        for side_index, side in enumerate(SIDES):
            train_side = training & development["side_name"].eq(side).to_numpy()
            valid_side = validation & development["side_name"].eq(side).to_numpy()
            if not train_side.any() or not valid_side.any():
                raise ValueError(f"empty {side} rows in fold {fold.fold_id}")
            features = arm_features(arm, side)
            pred, _ = fit_predict_model(
                numeric_features(development.loc[train_side], features),
                y[train_side],
                numeric_features(development.loc[valid_side], features),
                target=target,
                geometry=geometry,
                seed=seed + fold.fold_id * 100 + side_index,
                threads=threads,
            )
            prediction[valid_side] = pred
            fold_assignment[valid_side] = fold.fold_id
    if not np.isfinite(prediction).all() or (fold_assignment < 0).any():
        raise ValueError("OOF generation left uncovered rows")
    return prediction, fold_assignment


def crossfit_expected_net_mapping(
    score: np.ndarray,
    side: Sequence[str],
    fold_id: np.ndarray,
    net: np.ndarray,
    *,
    shrinkage_rows: int,
) -> tuple[np.ndarray, ShrunkMapper]:
    mapped = np.full(len(score), np.nan, dtype=np.float64)
    for held_fold in sorted(np.unique(fold_id)):
        fit = fold_id != held_fold
        held = fold_id == held_fold
        mapper = ShrunkMapper.fit(
            score[fit],
            np.asarray(side)[fit],
            net[fit],
            shrinkage_rows=shrinkage_rows,
        )
        mapped[held] = mapper.predict(score[held], np.asarray(side)[held])
    if not np.isfinite(mapped).all():
        raise ValueError("cross-fitted mapping left non-finite rows")
    final_mapper = ShrunkMapper.fit(
        score,
        side,
        net,
        shrinkage_rows=shrinkage_rows,
    )
    return mapped, final_mapper


def stable_global_top_mask(
    frame: pd.DataFrame,
    score: Sequence[float],
    fraction: float,
) -> np.ndarray:
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must lie in (0,1]")
    ranking = pd.DataFrame(
        {
            "position": np.arange(len(frame), dtype=np.int64),
            "candidate_id": frame["candidate_id"].astype(str).to_numpy(),
            "score": np.asarray(score, dtype=np.float64),
        }
    ).sort_values(
        ["score", "candidate_id"],
        ascending=[False, True],
        kind="stable",
    )
    count = max(1, int(math.ceil(len(frame) * fraction)))
    mask = np.zeros(len(frame), dtype=bool)
    mask[ranking["position"].to_numpy()[:count]] = True
    return mask


def safe_auc(y: np.ndarray, score: np.ndarray) -> float:
    if np.unique(y).size < 2:
        return float("nan")
    return float(roc_auc_score(y, score))


def sigmoid_net(net: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(net / 0.01, -60.0, 60.0)))


def calibration_errors(
    mapped_net: np.ndarray,
    realised_net: np.ndarray,
    hard0: np.ndarray,
    *,
    bins: int = 10,
) -> dict[str, float]:
    order = pd.Series(mapped_net).rank(method="first", pct=True).to_numpy()
    bucket = np.minimum((order * bins).astype(int), bins - 1)
    net_error = 0.0
    probability_error = 0.0
    total = 0
    probability = sigmoid_net(mapped_net)
    for index in range(bins):
        mask = bucket == index
        if not mask.any():
            continue
        weight = int(mask.sum())
        total += weight
        net_error += weight * abs(
            float(mapped_net[mask].mean() - realised_net[mask].mean())
        )
        probability_error += weight * abs(
            float(probability[mask].mean() - hard0[mask].mean())
        )
    return {
        "net_calibration_mae": net_error / max(total, 1),
        "hard0_ece": probability_error / max(total, 1),
        "hard0_brier": float(np.mean((probability - hard0) ** 2)),
    }


def score_metrics(
    frame: pd.DataFrame,
    raw_score: np.ndarray,
    mapped_net: np.ndarray,
    *,
    split: str,
    arm: str,
    target: str,
    geometry: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    net = frame["execution_net_ev_12h"].to_numpy(dtype=np.float64)
    hard0 = frame["opportunity_gross_above_cost_0bps"].to_numpy(dtype=np.float64)
    hard25 = frame["opportunity_gross_above_cost_25bps"].to_numpy(dtype=np.float64)
    calibration = calibration_errors(mapped_net, net, hard0)
    overall = {
        "split": split,
        "arm": arm,
        "target": target,
        "geometry": geometry,
        "rows": len(frame),
        "hard0_auc_raw_score": safe_auc(hard0, raw_score),
        "hard25_auc_raw_score": safe_auc(hard25, raw_score),
        "mapped_net_mae": float(np.mean(np.abs(mapped_net - net))),
        **calibration,
    }
    tails: list[dict[str, Any]] = []
    for fraction in FRACTIONS:
        mask = stable_global_top_mask(frame, mapped_net, fraction)
        selected = frame.loc[mask]
        tails.append(
            {
                "split": split,
                "arm": arm,
                "target": target,
                "geometry": geometry,
                "fraction": fraction,
                "rows": int(mask.sum()),
                "mean_net_bps": float(
                    selected["execution_net_ev_12h"].mean() * 10_000.0
                ),
                "sum_net": float(selected["execution_net_ev_12h"].sum()),
                "hard0_precision": float(
                    selected["opportunity_gross_above_cost_0bps"].mean()
                ),
                "hard25_precision": float(
                    selected["opportunity_gross_above_cost_25bps"].mean()
                ),
                "long_share": float(selected["side_name"].eq("long").mean()),
                "symbols": int(selected["__symbol__"].nunique()),
                "days": int(selected["__ts__"].dt.floor("D").nunique()),
            }
        )
    return overall, tails


def week_side_contributions(
    frame: pd.DataFrame,
    mapped_net: np.ndarray,
    *,
    arm: str,
    target: str,
    geometry: str,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    week = frame["__ts__"].dt.to_period("W-SUN").dt.start_time.dt.tz_localize("UTC")
    for fraction in FRACTIONS:
        selected = stable_global_top_mask(frame, mapped_net, fraction)
        subset = frame.loc[selected].copy()
        subset["week_start_utc"] = week[selected].to_numpy()
        for (week_start, side), group in subset.groupby(
            ["week_start_utc", "side_name"], sort=True
        ):
            records.append(
                {
                    "arm": arm,
                    "target": target,
                    "geometry": geometry,
                    "fraction": fraction,
                    "week_start_utc": week_start,
                    "side_name": side,
                    "rows": len(group),
                    "mean_net_bps": float(
                        group["execution_net_ev_12h"].mean() * 10_000.0
                    ),
                    "sum_net": float(group["execution_net_ev_12h"].sum()),
                    "hard0_precision": float(
                        group["opportunity_gross_above_cost_0bps"].mean()
                    ),
                }
            )
    return pd.DataFrame.from_records(records)


def day_block_bootstrap(
    frame: pd.DataFrame,
    mapped_net: np.ndarray,
    *,
    arm: str,
    target: str,
    geometry: str,
    repetitions: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    day = frame["__ts__"].dt.floor("D").to_numpy()
    records: list[dict[str, Any]] = []
    for fraction in FRACTIONS:
        selected = stable_global_top_mask(frame, mapped_net, fraction)
        selected_days = day[selected]
        selected_net = frame.loc[selected, "execution_net_ev_12h"].to_numpy(float)
        unique_days = np.unique(selected_days)
        grouped = {
            value: selected_net[selected_days == value] for value in unique_days
        }
        samples = np.empty(repetitions, dtype=np.float64)
        for index in range(repetitions):
            draw = rng.choice(unique_days, size=len(unique_days), replace=True)
            values = np.concatenate([grouped[value] for value in draw])
            samples[index] = values.mean() * 10_000.0
        records.append(
            {
                "arm": arm,
                "target": target,
                "geometry": geometry,
                "fraction": fraction,
                "repetitions": repetitions,
                "mean_net_bps": float(samples.mean()),
                "lower_95_bps": float(np.quantile(samples, 0.025)),
                "upper_95_bps": float(np.quantile(samples, 0.975)),
            }
        )
    return pd.DataFrame.from_records(records)


def rank_feature_arms(tails: pd.DataFrame) -> dict[str, list[str]]:
    top10 = tails.loc[np.isclose(tails["fraction"], 0.10)].copy()
    result: dict[str, list[str]] = {}
    for target in TARGETS:
        candidates = top10.loc[
            top10["target"].eq(target)
            & top10["arm"].isin(PRIMARY_ARMS)
            & top10["geometry"].eq(GEOMETRIES[0].name)
        ].sort_values(
            ["mean_net_bps", "hard0_precision", "arm"],
            ascending=[False, False, True],
            kind="stable",
        )
        if len(candidates) != len(PRIMARY_ARMS):
            raise ValueError(f"incomplete fixed-arm OOF metrics for {target}")
        result[target] = candidates["arm"].tolist()[:2]
    return result


def choose_geometry(tails: pd.DataFrame, selected_arms: Mapping[str, Sequence[str]]) -> pd.DataFrame:
    top10 = tails.loc[np.isclose(tails["fraction"], 0.10)].copy()
    rows: list[pd.Series] = []
    for target, arms in selected_arms.items():
        for arm in arms:
            candidates = top10.loc[
                top10["target"].eq(target) & top10["arm"].eq(arm)
            ].sort_values(
                ["mean_net_bps", "hard0_precision", "geometry"],
                ascending=[False, False, True],
                kind="stable",
            )
            rows.append(candidates.iloc[0])
    return pd.DataFrame(rows).reset_index(drop=True)


def fit_final_side_models(
    development: pd.DataFrame,
    april: pd.DataFrame,
    *,
    arm: str,
    target: str,
    geometry: Geometry,
    threads: int,
    seed: int,
    model_dir: Path,
) -> np.ndarray:
    prediction = np.full(len(april), np.nan, dtype=np.float64)
    y = target_values(development, target)
    for side_index, side in enumerate(SIDES):
        train_mask = development["side_name"].eq(side).to_numpy()
        april_mask = april["side_name"].eq(side).to_numpy()
        features = arm_features(arm, side)
        pred, model = fit_predict_model(
            numeric_features(development.loc[train_mask], features),
            y[train_mask],
            numeric_features(april.loc[april_mask], features),
            target=target,
            geometry=geometry,
            seed=seed + side_index,
            threads=threads,
        )
        prediction[april_mask] = pred
        destination = model_dir / f"{target}__{arm}__{geometry.name}__{side}.cbm"
        model.save_model(destination)
    if not np.isfinite(prediction).all():
        raise ValueError("final side-local model left non-finite April predictions")
    return prediction


def promotion_gate(
    april_tail: pd.DataFrame,
    weekly: pd.DataFrame,
    bootstrap: pd.DataFrame,
    *,
    arm: str,
    target: str,
    geometry: str,
) -> dict[str, Any]:
    key = (
        april_tail["arm"].eq(arm)
        & april_tail["target"].eq(target)
        & april_tail["geometry"].eq(geometry)
        & np.isclose(april_tail["fraction"], 0.10)
    )
    tail = april_tail.loc[key].iloc[0]
    boot = bootstrap.loc[
        bootstrap["arm"].eq(arm)
        & bootstrap["target"].eq(target)
        & bootstrap["geometry"].eq(geometry)
        & np.isclose(bootstrap["fraction"], 0.10)
    ].iloc[0]
    week = weekly.loc[
        weekly["arm"].eq(arm)
        & weekly["target"].eq(target)
        & weekly["geometry"].eq(geometry)
        & np.isclose(weekly["fraction"], 0.10)
    ]
    weekly_total = week.groupby("week_start_utc", sort=True).agg(
        sum_net=("sum_net", "sum"),
        rows=("rows", "sum"),
    )
    latest_week_net_bps = (
        float(weekly_total.iloc[-1]["sum_net"] / weekly_total.iloc[-1]["rows"] * 10_000)
        if len(weekly_total)
        else float("nan")
    )
    checks = {
        "april_top10_positive": bool(tail["mean_net_bps"] > 0.0),
        "day_block_lower_95_positive": bool(boot["lower_95_bps"] > 0.0),
        "latest_week_nonnegative": bool(latest_week_net_bps >= 0.0),
        "both_sides_selected": bool(
            0.0 < float(tail["long_share"]) < 1.0
        ),
    }
    return {
        "arm": arm,
        "target": target,
        "geometry": geometry,
        "latest_week_net_bps": latest_week_net_bps,
        "checks": checks,
        "portfolio_replay_authorized": bool(all(checks.values())),
        "portfolio_replay_performed": False,
    }


def fit_budget() -> dict[str, int]:
    primary_and_sensitivity = len(PRIMARY_ARMS) + len(SENSITIVITY_ARMS)
    fixed_oof = primary_and_sensitivity * len(TARGETS) * len(SIDES) * N_FOLDS
    hpo_oof = (
        2
        * len(TARGETS)
        * (len(GEOMETRIES) - 1)
        * len(SIDES)
        * N_FOLDS
    )
    fixed_final = primary_and_sensitivity * len(TARGETS) * len(SIDES)
    tuned_final = 2 * len(TARGETS) * len(SIDES)
    return {
        "fixed_oof_model_fits": fixed_oof,
        "additional_hpo_oof_model_fits_max": hpo_oof,
        "fixed_april_final_model_fits": fixed_final,
        "selected_hpo_april_final_model_fits_max": tuned_final,
        "maximum_model_fits": fixed_oof + hpo_oof + fixed_final + tuned_final,
    }


def write_immutable_outputs(
    output: Path,
    temporary: Path,
    manifest: dict[str, Any],
) -> None:
    if output.exists():
        raise FileExistsError(f"immutable output already exists: {output}")
    hashes = {
        str(path.relative_to(temporary)): sha256_file(path)
        for path in sorted(temporary.rglob("*"))
        if path.is_file() and path.name not in ("manifest.json", "manifest.sha256")
    }
    manifest["outputs_sha256"] = hashes
    manifest_path = temporary / "manifest.json"
    manifest_path.write_text(
        json.dumps(json_safe(manifest), indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    )
    (temporary / "manifest.sha256").write_text(
        f"{sha256_file(manifest_path)}  manifest.json\n"
    )
    os.replace(temporary, output)


def run(args: argparse.Namespace) -> Path:
    frame, panel_manifest = load_panel(args.panel_root)
    development, april = split_development_april(frame)
    folds = make_blocked_folds(development)
    if args.plan_only:
        print(
            json.dumps(
                {
                    "development_rows": len(development),
                    "april_rows": len(april),
                    "feature_arms": [*PRIMARY_ARMS, *SENSITIVITY_ARMS],
                    "targets": list(TARGETS),
                    "geometries": [asdict(item) for item in GEOMETRIES],
                    "fit_budget": fit_budget(),
                },
                indent=2,
            )
        )
        return args.output

    if args.output.exists():
        raise FileExistsError(f"immutable output already exists: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(dir=args.output.parent, prefix=f".{args.output.name}.")
    )
    try:
        model_dir = temporary / "models"
        mapper_dir = temporary / "mappers"
        model_dir.mkdir()
        mapper_dir.mkdir()
        oof_wide = development.loc[
            :, [*IDENTITY, "execution_net_ev_12h"]
        ].copy()
        april_wide = april.loc[:, [*IDENTITY, "execution_net_ev_12h"]].copy()
        oof_overall: list[dict[str, Any]] = []
        oof_tails: list[dict[str, Any]] = []
        cached: dict[tuple[str, str, str], tuple[np.ndarray, np.ndarray, ShrunkMapper]] = {}

        # Stage 1: every fixed-geometry primary arm and the no-geometry sensitivity.
        for target in TARGETS:
            for arm in (*PRIMARY_ARMS, *SENSITIVITY_ARMS):
                geometry = GEOMETRIES[0]
                raw, fold_id = generate_side_local_oof(
                    development,
                    folds,
                    arm=arm,
                    target=target,
                    geometry=geometry,
                    threads=args.threads,
                    seed=args.seed,
                )
                mapped, mapper = crossfit_expected_net_mapping(
                    raw,
                    development["side_name"].to_numpy(),
                    fold_id,
                    development["execution_net_ev_12h"].to_numpy(float),
                    shrinkage_rows=args.mapper_shrinkage_rows,
                )
                key = (target, arm, geometry.name)
                cached[key] = (raw, mapped, mapper)
                column = "__".join(key)
                oof_wide[f"raw__{column}"] = raw
                oof_wide[f"mapped_net__{column}"] = mapped
                overall, tails = score_metrics(
                    development,
                    raw,
                    mapped,
                    split="development_oof",
                    arm=arm,
                    target=target,
                    geometry=geometry.name,
                )
                oof_overall.append(overall)
                oof_tails.extend(tails)

        oof_tail_frame = pd.DataFrame(oof_tails)
        selected_arms = rank_feature_arms(oof_tail_frame)

        # Stage 2: only the two best fixed arms per target receive two extra geometries.
        for target, arms in selected_arms.items():
            for arm in arms:
                for geometry in GEOMETRIES[1:]:
                    raw, fold_id = generate_side_local_oof(
                        development,
                        folds,
                        arm=arm,
                        target=target,
                        geometry=geometry,
                        threads=args.threads,
                        seed=args.seed,
                    )
                    mapped, mapper = crossfit_expected_net_mapping(
                        raw,
                        development["side_name"].to_numpy(),
                        fold_id,
                        development["execution_net_ev_12h"].to_numpy(float),
                        shrinkage_rows=args.mapper_shrinkage_rows,
                    )
                    key = (target, arm, geometry.name)
                    cached[key] = (raw, mapped, mapper)
                    column = "__".join(key)
                    oof_wide[f"raw__{column}"] = raw
                    oof_wide[f"mapped_net__{column}"] = mapped
                    overall, tails = score_metrics(
                        development,
                        raw,
                        mapped,
                        split="development_oof",
                        arm=arm,
                        target=target,
                        geometry=geometry.name,
                    )
                    oof_overall.append(overall)
                    oof_tails.extend(tails)

        oof_tail_frame = pd.DataFrame(oof_tails)
        winners = choose_geometry(oof_tail_frame, selected_arms)

        # Fixed arms plus OOF-selected tuned configurations are fitted before April
        # metrics are inspected.  Duplicate fixed winners are fitted only once.
        final_configs = {
            (target, arm, GEOMETRIES[0].name)
            for target in TARGETS
            for arm in (*PRIMARY_ARMS, *SENSITIVITY_ARMS)
        }
        final_configs.update(
            (str(row.target), str(row.arm), str(row.geometry))
            for row in winners.itertuples(index=False)
        )
        geometry_by_name = {item.name: item for item in GEOMETRIES}
        april_overall: list[dict[str, Any]] = []
        april_tails: list[dict[str, Any]] = []
        weekly_frames: list[pd.DataFrame] = []
        bootstrap_frames: list[pd.DataFrame] = []
        for config_index, (target, arm, geometry_name) in enumerate(
            sorted(final_configs)
        ):
            geometry = geometry_by_name[geometry_name]
            raw_april = fit_final_side_models(
                development,
                april,
                arm=arm,
                target=target,
                geometry=geometry,
                threads=args.threads,
                seed=args.seed,
                model_dir=model_dir,
            )
            mapper = cached[(target, arm, geometry_name)][2]
            mapped_april = mapper.predict(
                raw_april, april["side_name"].to_numpy()
            )
            column = "__".join((target, arm, geometry_name))
            april_wide[f"raw__{column}"] = raw_april
            april_wide[f"mapped_net__{column}"] = mapped_april
            (mapper_dir / f"{column}.json").write_text(
                json.dumps(mapper.to_dict(), indent=2, sort_keys=True) + "\n"
            )
            overall, tails = score_metrics(
                april,
                raw_april,
                mapped_april,
                split="untouched_april",
                arm=arm,
                target=target,
                geometry=geometry_name,
            )
            april_overall.append(overall)
            april_tails.extend(tails)
            weekly_frames.append(
                week_side_contributions(
                    april,
                    mapped_april,
                    arm=arm,
                    target=target,
                    geometry=geometry_name,
                )
            )
            bootstrap_frames.append(
                day_block_bootstrap(
                    april,
                    mapped_april,
                    arm=arm,
                    target=target,
                    geometry=geometry_name,
                    repetitions=args.bootstrap_repetitions,
                    seed=args.seed + config_index,
                )
            )

        april_tail_frame = pd.DataFrame(april_tails)
        weekly = pd.concat(weekly_frames, ignore_index=True)
        bootstrap = pd.concat(bootstrap_frames, ignore_index=True)
        # Both predeclared HPO-eligible arms per target receive independent gates.
        gates = [
            promotion_gate(
                april_tail_frame,
                weekly,
                bootstrap,
                arm=str(row.arm),
                target=str(row.target),
                geometry=str(row.geometry),
            )
            for row in winners.itertuples(index=False)
        ]

        oof_wide.to_parquet(
            temporary / "development_oof_predictions.parquet",
            index=False,
            compression="zstd",
        )
        april_wide.to_parquet(
            temporary / "untouched_april_predictions.parquet",
            index=False,
            compression="zstd",
        )
        pd.DataFrame(oof_overall).to_parquet(
            temporary / "development_oof_overall_metrics.parquet",
            index=False,
        )
        oof_tail_frame.to_parquet(
            temporary / "development_oof_tail_metrics.parquet", index=False
        )
        pd.DataFrame(april_overall).to_parquet(
            temporary / "untouched_april_overall_metrics.parquet", index=False
        )
        april_tail_frame.to_parquet(
            temporary / "untouched_april_tail_metrics.parquet", index=False
        )
        weekly.to_parquet(
            temporary / "untouched_april_week_side_contributions.parquet",
            index=False,
        )
        bootstrap.to_parquet(
            temporary / "untouched_april_day_block_bootstrap.parquet", index=False
        )
        winners.to_parquet(
            temporary / "development_oof_selected_geometries.parquet", index=False
        )
        manifest = {
            "schema": SCHEMA,
            "status": "COMPLETED_UNTOUCHED_APRIL_RESEARCH_ABLATION",
            "source": {
                "panel_root": str(args.panel_root),
                "panel_sha256": panel_manifest["outputs_sha256"]["panel.parquet"],
                "panel_manifest_sha256": sha256_file(
                    args.panel_root / "manifest.json"
                ),
                "identity_sha256": panel_manifest["identity_sha256"],
                "runner_sha256": sha256_file(Path(__file__).resolve()),
            },
            "population": {
                "development_rows": len(development),
                "development_side_rows": development.groupby("side_name")
                .size()
                .to_dict(),
                "resolution_cutoff_exclusive": APRIL_FREEZE.isoformat(),
                "april_rows": len(april),
                "april_side_rows": april.groupby("side_name").size().to_dict(),
            },
            "validation": {
                "kind": "five_contiguous_timestamp_block_complement_cv",
                "not_walk_forward_evidence": True,
                "folds": [
                    {
                        "fold_id": fold.fold_id,
                        "validation_start": fold.validation_start.isoformat(),
                        "validation_end": fold.validation_end.isoformat(),
                        "validation_timestamps": len(fold.validation_timestamps),
                    }
                    for fold in folds
                ],
                "purge_hours": PURGE_HOURS,
                "april_untouched_until_all_selection_frozen": True,
            },
            "features": {
                "primary_arms": {
                    arm: {
                        side: list(arm_features(arm, side)) for side in SIDES
                    }
                    for arm in PRIMARY_ARMS
                },
                "sensitivity_arms": {
                    arm: {
                        side: list(arm_features(arm, side)) for side in SIDES
                    }
                    for arm in SENSITIVITY_ARMS
                },
                "native_missing_no_global_preprocessing": True,
                "mapped_fields_forbidden": True,
                "approved_31_8_representation_selection_exception": True,
            },
            "targets": {
                "hard0": "opportunity_gross_above_cost_0bps",
                "hard25": "opportunity_gross_above_cost_25bps",
                "soft": "execution_soft_positive_12h = sigmoid(net/0.01)",
                "direct_net": "execution_net_ev_12h",
            },
            "geometry": {
                "fixed": asdict(GEOMETRIES[0]),
                "hpo_challengers": [asdict(item) for item in GEOMETRIES[1:]],
                "hpo_only_best_two_fixed_oof_arms_per_target": selected_arms,
                "selected": winners.to_dict(orient="records"),
            },
            "mapping": {
                "kind": "side_isotonic_shrunk_to_pooled_isotonic",
                "development_metrics_use_leave_fold_out_mapper": True,
                "april_mapper_fitted_only_on_development_oof_predictions": True,
                "shrinkage_rows": args.mapper_shrinkage_rows,
            },
            "selection": {
                "scope": "one pooled global top-k",
                "fractions": list(FRACTIONS),
                "tie_break": "candidate_id ascending",
                "never_per_timestamp_or_side": True,
            },
            "bootstrap": {
                "unit": "UTC day among frozen selected rows",
                "repetitions": args.bootstrap_repetitions,
                "seed": args.seed,
            },
            "promotion_gates": gates,
            "portfolio_replay": {
                "performed": False,
                "contract": "run later only for an OOF-frozen candidate whose April gates all pass",
            },
            "fit_budget": fit_budget(),
            "seed": args.seed,
            "threads": args.threads,
            "cost_contract": "current-spread counterfactual exact-policy cost subtracted exactly once",
            "checksum_convention": (
                "Every material output is SHA256-listed; manifest.json is "
                "verified by detached manifest.sha256."
            ),
        }
        write_immutable_outputs(args.output, temporary, manifest)
        return args.output
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel-root", type=Path, default=DEFAULT_PANEL_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--mapper-shrinkage-rows", type=int, default=5_000)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2_000)
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="validate the full input contract and print the fit budget only",
    )
    args = parser.parse_args(argv)
    if args.threads < 1:
        parser.error("--threads must be positive")
    if args.mapper_shrinkage_rows < 1:
        parser.error("--mapper-shrinkage-rows must be positive")
    if args.bootstrap_repetitions < 100:
        parser.error("--bootstrap-repetitions must be at least 100")
    return args


if __name__ == "__main__":
    run(parse_args())
