#!/usr/bin/env python3
"""Cross-era coherent tail-payoff challenger, frozen before July 20--23 labels.

This is a research-only challenger.  It learns a side-local, mutually
exclusive economic event distribution and conditional payoff quantiles from
the materialised February--April 2025 and May--July 19 2026 feature/label
population.  Current July features are scored and frozen before the current
exact outcome table is opened.

The score is intentionally a global candidate score.  No timestamp-local
selection, timing action, target-price action, or timeout action is included.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import log_loss


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_july_exact_preentry_heads import IDENTITY, sha256


SCHEMA = "cross_era_tail_payoff_challenger_v1"
SIDES = ("long", "short")
CLASS_NAMES = ("positive", "adverse_negative", "timeout_negative", "other_negative")
POSITIVE, ADVERSE, TIMEOUT, OTHER = range(4)
CURRENT_START = pd.Timestamp("2026-07-20T00:00:00Z")
MAP_DAYS = 21
MAP_SHRINK_ROWS = 2_000
TIMEOUT_SHRINK_ROWS = 2_000

# The folds deliberately cover both data eras, with no full walk-forward
# requirement.  Every training label resolves before the relevant block.
FOLD_WINDOWS = (
    ("old_march", "2025-03-01T00:00:00Z", "2025-04-01T00:00:00Z"),
    ("old_april", "2025-04-01T00:00:00Z", "2025-05-01T00:00:00Z"),
    ("recent_may", "2026-05-01T00:00:00Z", "2026-06-01T00:00:00Z"),
    ("recent_june", "2026-06-01T00:00:00Z", "2026-07-01T00:00:00Z"),
    ("recent_july", "2026-07-01T00:00:00Z", "2026-07-20T00:00:00Z"),
)

# This is deliberately bounded: two regularised LightGBM settings, selected
# only by the historical mapped global top-decile economics below.
HPO_CONFIGS: tuple[dict[str, Any], ...] = (
    {
        "name": "shallow_24",
        "feature_count": 24,
        "num_leaves": 15,
        "max_depth": 5,
        "min_child_samples": 300,
        "reg_lambda": 20.0,
        "n_estimators": 120,
        "learning_rate": 0.045,
    },
    {
        "name": "regularised_40",
        "feature_count": 40,
        "num_leaves": 23,
        "max_depth": 6,
        "min_child_samples": 500,
        "reg_lambda": 35.0,
        "n_estimators": 150,
        "learning_rate": 0.035,
    },
)

QUANTILE_SPECS = (
    ("positive", POSITIVE, 0.25),
    ("positive", POSITIVE, 0.50),
    ("adverse", ADVERSE, 0.50),
    ("adverse", ADVERSE, 0.85),
    ("timeout", TIMEOUT, 0.75),
    ("other", OTHER, 0.75),
)

REGIME_SOURCES = (
    "regime_transition_entropy_48h",
    "regime_stability_24h",
    "market_breadth_24h",
    "negative_breadth_pct",
    "eth_btc_ret_24h",
    "xs_dispersion__amihud_illiq",
    "volatility_of_volatility_48",
)


@dataclass(frozen=True)
class Fold:
    name: str
    start: pd.Timestamp
    end: pd.Timestamp
    train: np.ndarray
    valid: np.ndarray


def _safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, (Path, pd.Timestamp)):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(_safe(payload), indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _binding(path: Path) -> dict[str, Any]:
    return {"path": str(path), "sha256": sha256(path)}


def economic_event_code(frame: pd.DataFrame) -> np.ndarray:
    """Return a mutually exclusive, exhaustive economic four-class target."""
    positive = pd.to_numeric(frame["positive_net"], errors="raise").astype(bool).to_numpy()
    negative = ~positive
    adverse = pd.to_numeric(frame["adverse_first"], errors="raise").astype(bool).to_numpy()
    timeout = pd.to_numeric(frame["timeout_event"], errors="raise").astype(bool).to_numpy()
    result = np.full(len(frame), OTHER, dtype=np.int8)
    result[negative & timeout & ~adverse] = TIMEOUT
    result[negative & adverse] = ADVERSE
    result[positive] = POSITIVE
    return result


def add_regime_composites(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Add small explicit causal regime composites; no outcome-derived fields."""
    result = frame.copy()
    missing = [column for column in REGIME_SOURCES if column not in result]
    if missing:
        raise ValueError(f"regime sources unavailable: {missing}")
    result["regime_transition_instability"] = (
        result["regime_transition_entropy_48h"]
        * (1.0 - result["regime_stability_24h"].clip(0.0, 1.0))
    )
    result["regime_breadth_volatility"] = (
        (result["market_breadth_24h"] - result["negative_breadth_pct"])
        * result["volatility_of_volatility_48"]
    )
    result["regime_eth_btc_liquidity"] = (
        result["eth_btc_ret_24h"] * result["xs_dispersion__amihud_illiq"]
    )
    result["regime_transition_liquidity"] = (
        result["regime_transition_entropy_48h"]
        * result["xs_dispersion__amihud_illiq"]
    )
    # v3 harmonises both eras to spread-adjusted executable exact-1m paths.
    # The domain feature/calibration remains because market and candidate-score
    # distributions drift across eras, not because label resolution differs.
    if "era" in result:
        result["__era_is_2026__"] = result["era"].astype(str).str.startswith("2026").astype(float)
    else:
        result["__era_is_2026__"] = 1.0
    # Explicit interactions are restricted to stable relative score context
    # and regime state.  Raw rank/margin/group-count geometry is never used.
    if "base_rank_pct_timestamp_side" in result:
        result["__base_rank_pct_x_era__"] = result["base_rank_pct_timestamp_side"] * result["__era_is_2026__"]
    if "base_oof_score" in result:
        result["__base_oof_score_x_era__"] = result["base_oof_score"] * result["__era_is_2026__"]
    if "base_score_z_timestamp_side" in result:
        result["__base_score_z_x_era__"] = result["base_score_z_timestamp_side"] * result["__era_is_2026__"]
    result["__transition_entropy_x_era__"] = result["regime_transition_entropy_48h"] * result["__era_is_2026__"]
    return result, [
        "regime_transition_instability",
        "regime_breadth_volatility",
        "regime_eth_btc_liquidity",
        "regime_transition_liquidity",
    ]


def feature_arms(contract: Mapping[str, Any], composites: Sequence[str]) -> dict[str, list[str]]:
    raw = list(map(str, contract["feature_columns"]))
    # The materialiser documents an era shift for raw score/rank margin/group
    # geometry.  Percentile rank is the one invariant candidate-relative input
    # used in this v1; raw rank, margin, group-size and raw score are excluded.
    context = ["base_oof_score", "base_rank_pct_timestamp_side", "base_score_z_timestamp_side"]
    mandatory_domain = ["__era_is_2026__"]
    return {
        "raw": [*raw, *mandatory_domain],
        "raw_context": [*raw, *context, "__base_rank_pct_x_era__", "__base_oof_score_x_era__", "__base_score_z_x_era__", *mandatory_domain],
        "raw_context_regime": [*raw, *context, *composites, "__base_rank_pct_x_era__", "__base_oof_score_x_era__", "__base_score_z_x_era__", "__transition_entropy_x_era__", *mandatory_domain],
    }


def _normalise_matrix(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"required causal features missing: {missing}")
    matrix = frame.loc[:, list(columns)].apply(pd.to_numeric, errors="coerce")
    # The feature-contract inputs are materially complete, but median imputation
    # makes model fitting/prediction deterministic for a sparse exceptional field.
    return matrix.replace([np.inf, -np.inf], np.nan)


def chronological_folds(frame: pd.DataFrame) -> list[Fold]:
    ts = pd.to_datetime(frame["__ts__"], utc=True, errors="raise")
    resolved = pd.to_datetime(frame["label_resolution_utc"], utc=True, errors="raise")
    folds: list[Fold] = []
    for name, start_s, end_s in FOLD_WINDOWS:
        start, end = pd.Timestamp(start_s), pd.Timestamp(end_s)
        train = np.flatnonzero(((ts < start) & (resolved < start)).to_numpy())
        valid = np.flatnonzero(((ts >= start) & (ts < end)).to_numpy())
        if len(train) < 10_000 or len(valid) < 1_000:
            raise ValueError(f"insufficient support for {name}: train={len(train)} valid={len(valid)}")
        if not bool((resolved.iloc[train] < start).all()):
            raise AssertionError(f"label chronology violated for {name}")
        folds.append(Fold(name, start, end, train, valid))
    return folds


def _rank_score(values: np.ndarray, target: np.ndarray) -> float:
    finite = np.isfinite(values) & np.isfinite(target)
    if finite.sum() < 100 or np.unique(values[finite]).size < 2:
        return 0.0
    value = spearmanr(values[finite], target[finite]).statistic
    return float(value) if np.isfinite(value) else 0.0


def screen_features(
    matrix: pd.DataFrame,
    target: np.ndarray,
    positions: np.ndarray,
    count: int,
    *,
    multiclass: bool,
) -> list[str]:
    """Nested, target-specific univariate screen with correlation pruning."""
    if len(positions) < 100:
        return list(matrix.columns[: min(count, len(matrix.columns))])
    local = matrix.iloc[positions]
    y = np.asarray(target)[positions]
    coverage = local.notna().mean()
    variance = local.var()
    candidates = list(coverage.index[(coverage >= .99) & (variance > 1e-12)])
    if not candidates:
        raise ValueError("target-specific feature screen has no usable candidates")
    values = local.loc[:, candidates]
    midpoint = max(1, len(values) // 2)

    def stable_linear_score(y_value: np.ndarray) -> pd.Series:
        # Vectorised Pearson IC is materially faster than per-column Spearman
        # while retaining target-specific, early/late-sign-stable screening.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            full = values.corrwith(pd.Series(y_value, index=values.index)).fillna(0.0)
            early = values.iloc[:midpoint].corrwith(pd.Series(y_value[:midpoint], index=values.index[:midpoint])).fillna(0.0)
            late = values.iloc[midpoint:].corrwith(pd.Series(y_value[midpoint:], index=values.index[midpoint:])).fillna(0.0)
        stable = np.where(early.to_numpy() * late.to_numpy() > 0.0, np.minimum(np.abs(early.to_numpy()), np.abs(late.to_numpy())), .10 * np.abs(full.to_numpy()))
        return pd.Series(stable, index=values.columns)

    if multiclass:
        score = pd.concat(
            [stable_linear_score((y == klass).astype(float)) for klass in (POSITIVE, ADVERSE, TIMEOUT, OTHER)], axis=1
        ).max(axis=1)
    else:
        score = stable_linear_score(y.astype(float))
    ordered = sorted(((str(column), float(score.loc[column]), float(coverage.loc[column])) for column in candidates), key=lambda row: (-row[1], row[0]))
    selected: list[str] = []
    for column, _, _ in ordered:
        if len(selected) >= count:
            break
        candidate = local[column]
        candidate_std = float(candidate.std())
        correlated = False
        for previous in selected:
            if candidate_std <= 1e-12 or float(local[previous].std()) <= 1e-12:
                continue
            correlation = candidate.corr(local[previous])
            if np.isfinite(correlation) and abs(correlation) >= 0.95:
                correlated = True
                break
        if not correlated:
            selected.append(column)
    if len(selected) < min(8, count):
        raise ValueError("target-specific feature screen returned insufficient features")
    return selected


def _fit_event(matrix: pd.DataFrame, target: np.ndarray, config: Mapping[str, Any], seed: int) -> lgb.LGBMClassifier:
    counts = np.bincount(target.astype(int), minlength=4).astype(float)
    weights = np.divide(len(target), 4.0 * np.maximum(counts[target.astype(int)], 1.0))
    model = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=4,
        n_estimators=int(config["n_estimators"]),
        learning_rate=float(config["learning_rate"]),
        num_leaves=int(config["num_leaves"]),
        max_depth=int(config["max_depth"]),
        min_child_samples=int(config["min_child_samples"]),
        reg_lambda=float(config["reg_lambda"]),
        colsample_bytree=0.8,
        subsample=0.85,
        subsample_freq=1,
        random_state=seed,
        n_jobs=4,
        verbosity=-1,
    )
    model.fit(matrix, target.astype(int), sample_weight=weights)
    return model


def _fit_quantile(
    matrix: pd.DataFrame,
    target_loss_bps: np.ndarray,
    config: Mapping[str, Any],
    alpha: float,
    seed: int,
) -> Any:
    if len(target_loss_bps) < 120:
        return float(np.quantile(target_loss_bps, alpha))
    model = lgb.LGBMRegressor(
        objective="quantile",
        alpha=float(alpha),
        n_estimators=int(config["n_estimators"]),
        learning_rate=float(config["learning_rate"]),
        num_leaves=max(7, int(config["num_leaves"])),
        max_depth=int(config["max_depth"]),
        min_child_samples=max(100, int(config["min_child_samples"]) // 2),
        reg_lambda=float(config["reg_lambda"]),
        colsample_bytree=0.8,
        subsample=0.85,
        subsample_freq=1,
        random_state=seed,
        n_jobs=4,
        verbosity=-1,
    )
    # log1p stabilises the severe adverse tail; values are inverted at scoring.
    transformed = np.log1p(np.maximum(np.asarray(target_loss_bps, dtype=float), 0.0) / 100.0)
    model.fit(matrix, transformed)
    return model


def _predict_quantile(model: Any, matrix: pd.DataFrame) -> np.ndarray:
    if isinstance(model, (float, int, np.floating)):
        return np.full(len(matrix), float(model), dtype=float)
    return np.maximum(np.expm1(np.asarray(model.predict(matrix), dtype=float)) * 100.0, 0.0)


def _predict_proba(model: lgb.LGBMClassifier, matrix: pd.DataFrame) -> np.ndarray:
    result = np.asarray(model.predict_proba(matrix), dtype=float)
    if result.shape[1] != 4:
        raise ValueError("event model did not emit four class probabilities")
    return np.clip(result, 1e-8, 1.0)


def normalise_probabilities(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(probabilities, dtype=float), 1e-8, np.inf)
    return clipped / clipped.sum(axis=1, keepdims=True)


def fit_class_calibrators(raw: np.ndarray, target: np.ndarray, eras: Sequence[str]) -> dict[str, list[Any]]:
    """Fit domain-specific calibrators to control cross-era distribution drift."""
    values = np.asarray(eras, dtype=str)
    result: dict[str, list[Any]] = {}
    for era in sorted(set(values)):
        mask = values == era
        calibrators: list[Any] = []
        for klass in range(4):
            y = (target[mask] == klass).astype(int)
            if y.min() == y.max():
                calibrators.append(float(y.mean()))
                continue
            model = IsotonicRegression(y_min=1e-6, y_max=1.0 - 1e-6, out_of_bounds="clip")
            model.fit(raw[mask, klass], y)
            calibrators.append(model)
        result[era] = calibrators
    return result


def apply_class_calibrators(raw: np.ndarray, calibrators: Mapping[str, Sequence[Any]], eras: Sequence[str]) -> np.ndarray:
    values = np.asarray(eras, dtype=str)
    result = np.full((len(raw), 4), np.nan, dtype=float)
    for era in sorted(set(values)):
        if era not in calibrators:
            raise ValueError(f"no calibrated event distribution for domain {era}")
        mask = values == era
        columns = []
        for klass, calibrator in enumerate(calibrators[era]):
            if isinstance(calibrator, (float, int, np.floating)):
                columns.append(np.full(mask.sum(), float(calibrator)))
            else:
                columns.append(np.asarray(calibrator.predict(raw[mask, klass]), dtype=float))
        result[mask] = normalise_probabilities(np.column_stack(columns))
    return result


def compose_tail_scores(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["ev50_bps"] = (
        result["p_positive"] * result["q50_positive_bps"]
        - result["p_adverse_negative"] * result["q50_adverse_bps"]
        - result["p_timeout_negative"] * result["q75_timeout_bps"]
        - result["p_other_negative"] * result["q75_other_bps"]
    )
    result["tail_ev_bps"] = (
        result["p_positive"] * result["q25_positive_bps"]
        - result["p_adverse_negative"] * result["q85_adverse_bps"]
        - result["p_timeout_negative"] * result["q75_timeout_bps"]
        - result["p_other_negative"] * result["q75_other_bps"]
    )
    return result


def _side_timeout_prediction(
    pooled_prediction: np.ndarray,
    side: str,
    side_target: np.ndarray,
    alpha: float,
) -> np.ndarray:
    if len(side_target) == 0:
        return pooled_prediction
    side_value = float(np.quantile(side_target, alpha))
    weight = len(side_target) / (len(side_target) + TIMEOUT_SHRINK_ROWS)
    return weight * pooled_prediction + (1.0 - weight) * side_value


def _inner_side_domain_calibrator(
    frame: pd.DataFrame,
    matrix: pd.DataFrame,
    event_code: np.ndarray,
    train: np.ndarray,
    valid: np.ndarray,
    config: Mapping[str, Any],
    seed: int,
) -> tuple[dict[str, list[Any]] | None, dict[str, Any]]:
    """Fit calibrator on an inner chronological train block only.

    The outer validation labels are never inspected.  The calibration block is
    restricted to the outer validation's era domain, so calibration controls
    cross-era distribution drift despite harmonised exact-1m measurements.
    """
    domain_values = frame.iloc[valid]["era"].astype(str).unique()
    if len(domain_values) != 1:
        raise ValueError("an outer validation block must have one event-label domain")
    domain = str(domain_values[0])
    candidates = train[frame.iloc[train]["era"].astype(str).eq(domain).to_numpy()]
    if len(candidates) < 1_000:
        return None, {"domain": domain, "status": "raw_probability_no_prior_domain_calibration", "rows": int(len(candidates))}
    ordered = candidates[np.argsort(frame.iloc[candidates]["__ts__"].to_numpy())]
    calibration_count = max(500, int(math.ceil(0.20 * len(ordered))))
    calibration = ordered[-calibration_count:]
    inner_train = np.setdiff1d(train, calibration, assume_unique=False)
    if len(inner_train) < 5_000:
        return None, {"domain": domain, "status": "raw_probability_insufficient_inner_train", "rows": int(len(inner_train))}
    selected = screen_features(matrix, event_code, inner_train, int(config["feature_count"]), multiclass=True)
    median = matrix.iloc[inner_train][selected].median()
    model = _fit_event(matrix.iloc[inner_train][selected].fillna(median), event_code[inner_train], config, seed)
    raw_cal = _predict_proba(model, matrix.iloc[calibration][selected].fillna(median))
    return fit_class_calibrators(raw_cal, event_code[calibration], np.full(len(calibration), domain)), {
        "domain": domain,
        "status": "inner_train_chronological_calibration",
        "inner_train_rows": int(len(inner_train)),
        "calibration_rows": int(len(calibration)),
        "features": selected,
    }


def _fit_fold_arm(
    frame: pd.DataFrame,
    matrix: pd.DataFrame,
    fold: Fold,
    config: Mapping[str, Any],
    seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fit one arm/config on one chronological block and return OOF scores."""
    event_code = economic_event_code(frame)
    net_bps = pd.to_numeric(frame["execution_net_ev_12h"], errors="raise").to_numpy(float) * 1e4
    output = frame.iloc[fold.valid].loc[:, [*IDENTITY, "era", "label_resolution_utc"]].copy()
    output["event_code"] = event_code[fold.valid]
    output["execution_net_ev_12h"] = net_bps[fold.valid] / 1e4
    feature_records: dict[str, Any] = {"event": {}, "payoff": {}, "calibration": {}}

    for side_index, side in enumerate(SIDES):
        train = fold.train[frame.iloc[fold.train]["side_name"].astype(str).to_numpy() == side]
        valid = fold.valid[frame.iloc[fold.valid]["side_name"].astype(str).to_numpy() == side]
        if len(train) < 5_000 or len(valid) < 100:
            raise ValueError(f"insufficient {side} support in {fold.name}")
        selected_event = screen_features(
            matrix, event_code, train, int(config["feature_count"]), multiclass=True
        )
        event_model = _fit_event(
            matrix.iloc[train].loc[:, selected_event].fillna(matrix.iloc[train][selected_event].median()),
            event_code[train], config, seed + side_index * 10,
        )
        train_median = matrix.iloc[train][selected_event].median()
        raw = _predict_proba(event_model, matrix.iloc[valid].loc[:, selected_event].fillna(train_median))
        positions = output.index[output["side_name"].astype(str).eq(side)]
        output.loc[positions, ["raw_p_positive", "raw_p_adverse_negative", "raw_p_timeout_negative", "raw_p_other_negative"]] = raw
        calibrators, calibration_record = _inner_side_domain_calibrator(
            frame, matrix, event_code, train, valid, config, seed + side_index * 10 + 50
        )
        domain = frame.iloc[valid]["era"].astype(str).to_numpy()
        calibrated = raw if calibrators is None else apply_class_calibrators(raw, calibrators, domain)
        output.loc[positions, ["p_positive", "p_adverse_negative", "p_timeout_negative", "p_other_negative"]] = calibrated
        feature_records["event"][side] = selected_event
        feature_records["calibration"][side] = calibration_record

        for target_name, klass, alpha in QUANTILE_SPECS:
            label = f"q{int(alpha * 100):02d}_{target_name}_bps"
            conditional = train[event_code[train] == klass]
            # A positive payoff is already positive; all non-positive class
            # heads use the positive loss magnitude.
            values = net_bps[conditional] if klass == POSITIVE else -net_bps[conditional]
            selected = screen_features(
                matrix, net_bps if klass == POSITIVE else -net_bps,
                conditional, min(int(config["feature_count"]), 24), multiclass=False,
            )
            if target_name == "timeout":
                # A pooled model is fitted below, not a fragile side-local
                # nonlinear timeout model.  Record the intended target-specific
                # screen for audit consistency.
                feature_records["payoff"].setdefault(label, {})[side] = selected
                continue
            train_median = matrix.iloc[conditional][selected].median()
            model = _fit_quantile(
                matrix.iloc[conditional].loc[:, selected].fillna(train_median), values, config, alpha,
                seed + 100 + side_index * 20 + int(alpha * 100),
            )
            prediction = _predict_quantile(model, matrix.iloc[valid].loc[:, selected].fillna(train_median))
            output.loc[positions, label] = prediction
            feature_records["payoff"].setdefault(label, {})[side] = selected

    # Timeout is pooled with a side indicator and then explicitly shrunk to the
    # side conditional quantile.  This protects the 2026 short timeout support.
    timeout_train = fold.train[event_code[fold.train] == TIMEOUT]
    timeout_valid = fold.valid
    timeout_target = -net_bps[timeout_train]
    timeout_matrix = matrix.copy()
    timeout_matrix["__side_long__"] = frame["side_name"].astype(str).eq("long").astype(float)
    timeout_selected = screen_features(
        timeout_matrix, -net_bps, timeout_train,
        min(int(config["feature_count"]), 16), multiclass=False,
    )
    timeout_median = timeout_matrix.iloc[timeout_train][timeout_selected].median()
    timeout_model = _fit_quantile(
        timeout_matrix.iloc[timeout_train].loc[:, timeout_selected].fillna(timeout_median),
        timeout_target, config, 0.75, seed + 400,
    )
    pooled_timeout = _predict_quantile(
        timeout_model,
        timeout_matrix.iloc[timeout_valid].loc[:, timeout_selected].fillna(timeout_median),
    )
    for side in SIDES:
        valid_mask = frame.iloc[timeout_valid]["side_name"].astype(str).eq(side).to_numpy()
        side_train = timeout_train[frame.iloc[timeout_train]["side_name"].astype(str).eq(side).to_numpy()]
        values = -net_bps[side_train]
        positions = output.index[output["side_name"].astype(str).eq(side)]
        output.loc[positions, "q75_timeout_bps"] = _side_timeout_prediction(
            pooled_timeout[valid_mask], side, values, 0.75
        )
    feature_records["payoff"]["q75_timeout_bps"] = {"pooled": timeout_selected}
    return compose_tail_scores(output), feature_records


def causal_side_shrunk_isotonic(
    frame: pd.DataFrame,
    score_column: str,
    *,
    current: pd.DataFrame | None = None,
) -> tuple[pd.Series, dict[str, Any]]:
    """Causal 21d pooled/side-shrunk isotonic map for OOF or current scores."""
    source = frame.copy()
    source["__ts__"] = pd.to_datetime(source["__ts__"], utc=True)
    source["label_resolution_utc"] = pd.to_datetime(source["label_resolution_utc"], utc=True)
    source["_net_bps"] = pd.to_numeric(source["execution_net_ev_12h"], errors="raise") * 1e4
    target = source if current is None else current.copy()
    target["__ts__"] = pd.to_datetime(target["__ts__"], utc=True)
    mapped = pd.Series(index=target.index, dtype=float)
    records: list[dict[str, Any]] = []
    for day, local in target.groupby(target["__ts__"].dt.floor("D"), sort=True):
        lower = day - pd.Timedelta(days=MAP_DAYS)
        available = source.loc[
            source["__ts__"].lt(day)
            & source["__ts__"].ge(lower)
            & source["label_resolution_utc"].lt(day)
            & source[score_column].notna()
        ]
        if len(available) < 200:
            mapped.loc[local.index] = local[score_column].to_numpy(float)
            records.append({"day": day, "rows": len(available), "mapped": False})
            continue
        pooled = IsotonicRegression(out_of_bounds="clip")
        pooled.fit(available[score_column], available["_net_bps"])
        pooled_values = pooled.predict(local[score_column])
        values = np.asarray(pooled_values, dtype=float)
        for side in SIDES:
            idx = local.index[local["side_name"].astype(str).eq(side)]
            side_available = available.loc[available["side_name"].astype(str).eq(side)]
            if len(idx) == 0 or len(side_available) < 100:
                continue
            model = IsotonicRegression(out_of_bounds="clip")
            model.fit(side_available[score_column], side_available["_net_bps"])
            side_values = model.predict(local.loc[idx, score_column])
            weight = len(side_available) / (len(side_available) + MAP_SHRINK_ROWS)
            position = local.index.get_indexer(idx)
            values[position] = weight * side_values + (1.0 - weight) * values[position]
        mapped.loc[local.index] = values
        records.append({"day": day, "rows": len(available), "mapped": True})
    return mapped, {"days": records, "window_days": MAP_DAYS, "side_shrink_rows": MAP_SHRINK_ROWS}


def _top_economics(frame: pd.DataFrame, score_column: str) -> tuple[dict[str, Any], pd.DataFrame]:
    work = frame.loc[np.isfinite(frame[score_column])].copy()
    work["month"] = pd.to_datetime(work["__ts__"], utc=True).dt.strftime("%Y-%m")
    rows: list[dict[str, Any]] = []
    for month, local in work.groupby("month", sort=True):
        take = max(1, int(math.ceil(0.10 * len(local))))
        chosen = local.sort_values([score_column, "candidate_id"], ascending=[False, True], kind="stable").iloc[:take]
        net = chosen["execution_net_ev_12h"].to_numpy(float) * 1e4
        rows.append({
            "month": month,
            "rows": len(chosen),
            "net_ev_bps": float(np.mean(net)),
            "positive_precision": float((net > 0).mean()),
            "cvar05_bps": float(np.mean(np.sort(net)[: max(1, int(math.ceil(.05 * len(net))))])),
            "long_rows": int(chosen["side_name"].astype(str).eq("long").sum()),
            "short_rows": int(chosen["side_name"].astype(str).eq("short").sum()),
        })
    monthly = pd.DataFrame(rows)
    global_take = max(1, int(math.ceil(0.10 * len(work))))
    global_top = work.sort_values([score_column, "candidate_id"], ascending=[False, True], kind="stable").iloc[:global_take]
    global_net = global_top["execution_net_ev_12h"].to_numpy(float) * 1e4
    metrics = {
        "global_rows": len(global_top),
        "global_top10_net_ev_bps": float(np.mean(global_net)),
        "global_top10_positive_precision": float((global_net > 0).mean()),
        "global_top10_cvar05_bps": float(np.mean(np.sort(global_net)[: max(1, int(math.ceil(.05 * len(global_net))))])),
        "worst_month_top10_net_ev_bps": float(monthly["net_ev_bps"].min()),
        "latest_month_top10_net_ev_bps": float(monthly.sort_values("month").iloc[-1]["net_ev_bps"]),
        "mean_month_top10_net_ev_bps": float(monthly["net_ev_bps"].mean()),
    }
    return metrics, monthly


def _trial_key(metrics: Mapping[str, Any]) -> tuple[float, float, float, float]:
    # The only selector: global top10 exact economics, then worst/latest month
    # and tail loss.  Predictive metrics are diagnostic only.
    return (
        float(metrics["global_top10_net_ev_bps"]),
        float(metrics["worst_month_top10_net_ev_bps"]),
        float(metrics["latest_month_top10_net_ev_bps"]),
        float(metrics["global_top10_cvar05_bps"]),
    )


def run_oof_trials(
    frame: pd.DataFrame,
    arms: Mapping[str, Sequence[str]],
    seed: int,
    configs: Sequence[Mapping[str, Any]] = HPO_CONFIGS,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    folds = chronological_folds(frame)
    targets = economic_event_code(frame)
    trials: list[dict[str, Any]] = []
    prediction_frames: dict[str, pd.DataFrame] = {}
    month_frames: dict[str, pd.DataFrame] = {}
    for arm_index, (arm_name, columns) in enumerate(arms.items()):
        matrix = _normalise_matrix(frame, columns)
        for config_index, config in enumerate(configs):
            key = f"{arm_name}__{config['name']}"
            print(f"tail-payoff: start trial={key}", flush=True)
            parts: list[pd.DataFrame] = []
            records: list[dict[str, Any]] = []
            for fold_index, fold in enumerate(folds):
                print(f"tail-payoff: trial={key} fold={fold.name}", flush=True)
                part, feature_record = _fit_fold_arm(
                    frame, matrix, fold, config, seed + arm_index * 10_000 + config_index * 1_000 + fold_index * 100,
                )
                part["trial_key"] = key
                parts.append(part)
                records.append({"fold": fold.name, "features": feature_record, "train_rows": len(fold.train), "validation_rows": len(fold.valid)})
            raw_oof = pd.concat(parts, ignore_index=True)
            raw_columns = ["raw_p_positive", "raw_p_adverse_negative", "raw_p_timeout_negative", "raw_p_other_negative"]
            raw_oof = compose_tail_scores(raw_oof)
            raw_oof["mapped_tail_ev_bps"], map_record = causal_side_shrunk_isotonic(raw_oof, "tail_ev_bps")
            metrics, monthly = _top_economics(raw_oof, "mapped_tail_ev_bps")
            # Diagnostics deliberately do not participate in selection.
            calibrated = raw_oof[["p_positive", "p_adverse_negative", "p_timeout_negative", "p_other_negative"]].to_numpy(float)
            metrics["event_multiclass_log_loss"] = float(log_loss(raw_oof["event_code"], calibrated, labels=[0, 1, 2, 3]))
            metrics["probability_simplex_max_error"] = float(np.abs(calibrated.sum(axis=1) - 1.0).max())
            trials.append({
                "trial_key": key,
                "arm": arm_name,
                "config": dict(config),
                "metrics": metrics,
                "folds": records,
                "mapping": map_record,
            })
            prediction_frames[key] = raw_oof
            month_frames[key] = monthly.assign(trial_key=key)
            print(
                f"tail-payoff: completed trial={key} global_top10_bps={metrics['global_top10_net_ev_bps']:.3f}",
                flush=True,
            )
    winner = max(trials, key=lambda record: _trial_key(record["metrics"]))
    oof = prediction_frames[winner["trial_key"]].copy()
    monthly = month_frames[winner["trial_key"]].copy()
    return winner, oof, pd.DataFrame([
        {"trial_key": row["trial_key"], "arm": row["arm"], **row["metrics"]} for row in trials
    ]).sort_values(["global_top10_net_ev_bps", "worst_month_top10_net_ev_bps"], ascending=False, kind="stable")


def _fit_final_models(
    frame: pd.DataFrame,
    columns: Sequence[str],
    config: Mapping[str, Any],
    oof: pd.DataFrame,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    matrix = _normalise_matrix(frame, columns)
    event_code = economic_event_code(frame)
    net_bps = pd.to_numeric(frame["execution_net_ev_12h"], errors="raise").to_numpy(float) * 1e4
    bundle: dict[str, Any] = {"event": {}, "payoff": {}, "columns": list(columns)}
    state: dict[str, Any] = {"sides": {}, "timeout": {}}
    for side_index, side in enumerate(SIDES):
        positions = np.flatnonzero(frame["side_name"].astype(str).eq(side).to_numpy())
        selected_event = screen_features(matrix, event_code, positions, int(config["feature_count"]), multiclass=True)
        median = matrix.iloc[positions][selected_event].median()
        bundle["event"][side] = {
            "features": selected_event,
            "median": median,
            "model": _fit_event(matrix.iloc[positions][selected_event].fillna(median), event_code[positions], config, seed + side_index),
        }
        state["sides"][side] = {"event_features": selected_event, "payoff_features": {}}
        for target_name, klass, alpha in QUANTILE_SPECS:
            label = f"q{int(alpha * 100):02d}_{target_name}_bps"
            if target_name == "timeout":
                continue
            conditional = positions[event_code[positions] == klass]
            selected = screen_features(matrix, net_bps if klass == POSITIVE else -net_bps, conditional, min(int(config["feature_count"]), 24), multiclass=False)
            median = matrix.iloc[conditional][selected].median()
            target = net_bps[conditional] if klass == POSITIVE else -net_bps[conditional]
            bundle["payoff"].setdefault(label, {})[side] = {
                "features": selected,
                "median": median,
                "model": _fit_quantile(matrix.iloc[conditional][selected].fillna(median), target, config, alpha, seed + 100 + side_index * 20 + int(alpha * 100)),
            }
            state["sides"][side]["payoff_features"][label] = selected
    timeout_positions = np.flatnonzero(event_code == TIMEOUT)
    timeout_matrix = matrix.copy()
    timeout_matrix["__side_long__"] = frame["side_name"].astype(str).eq("long").astype(float)
    selected = screen_features(timeout_matrix, -net_bps, timeout_positions, min(int(config["feature_count"]), 16), multiclass=False)
    median = timeout_matrix.iloc[timeout_positions][selected].median()
    model = _fit_quantile(timeout_matrix.iloc[timeout_positions][selected].fillna(median), -net_bps[timeout_positions], config, .75, seed + 400)
    side_quantiles = {}
    for side in SIDES:
        values = -net_bps[timeout_positions[frame.iloc[timeout_positions]["side_name"].astype(str).eq(side).to_numpy()]]
        side_quantiles[side] = {"rows": int(len(values)), "q75": float(np.quantile(values, .75)) if len(values) else 0.0}
    bundle["timeout"] = {"features": selected, "median": median, "model": model, "side_quantiles": side_quantiles}
    state["timeout"] = {"features": selected, "side_quantiles": side_quantiles, "shrink_rows": TIMEOUT_SHRINK_ROWS}
    # The final-current calibrator is derived from a fixed recent historical
    # inner calibration block (July 1--19) after a model fit only through June.
    # It never sees current July 20--23 labels and is side/domain local.
    cutoff = pd.Timestamp("2026-07-01T00:00:00Z")
    bundle["calibrators"] = {}
    state["calibration"] = {}
    for side_index, side in enumerate(SIDES):
        calibration = np.flatnonzero(
            frame["side_name"].astype(str).eq(side).to_numpy()
            & frame["era"].astype(str).eq("2026_may_jul19").to_numpy()
            & frame["__ts__"].ge(cutoff).to_numpy()
        )
        inner = np.flatnonzero(
            frame["side_name"].astype(str).eq(side).to_numpy()
            & frame["label_resolution_utc"].lt(cutoff).to_numpy()
        )
        if len(calibration) < 500 or len(inner) < 5_000:
            raise ValueError(f"insufficient frozen 2026 calibration support for {side}")
        selected = screen_features(matrix, event_code, inner, int(config["feature_count"]), multiclass=True)
        median = matrix.iloc[inner][selected].median()
        inner_model = _fit_event(matrix.iloc[inner][selected].fillna(median), event_code[inner], config, seed + 900 + side_index)
        raw_calibration = _predict_proba(inner_model, matrix.iloc[calibration][selected].fillna(median))
        bundle["calibrators"][side] = fit_class_calibrators(
            raw_calibration, event_code[calibration], np.full(len(calibration), "2026_may_jul19")
        )
        state["calibration"][side] = {
            "domain": "2026_may_jul19",
            "inner_train_end_exclusive": cutoff,
            "inner_train_rows": int(len(inner)),
            "calibration_rows": int(len(calibration)),
            "features": selected,
        }
    return bundle, state


def _prepare_current_features(packb_path: Path, columns: Sequence[str]) -> pd.DataFrame:
    current = pd.read_parquet(packb_path)
    current["era"] = "2026_may_jul19"
    rename = {
        "base_candidate_group_rows": "candidate_group_size",
        "base_margin_to_cutoff": "base_margin_to_candidate_cutoff",
    }
    current = current.rename(columns={source: target for source, target in rename.items() if source in current})
    current, _ = add_regime_composites(current)
    _normalise_matrix(current, columns)
    if current.duplicated(list(IDENTITY)).any():
        raise ValueError("current feature population has duplicate identities")
    return current


def score_current(bundle: Mapping[str, Any], current: pd.DataFrame) -> pd.DataFrame:
    matrix = _normalise_matrix(current, bundle["columns"])
    result = current.loc[:, list(IDENTITY)].copy()
    raw_columns = ["raw_p_positive", "raw_p_adverse_negative", "raw_p_timeout_negative", "raw_p_other_negative"]
    for side in SIDES:
        positions = np.flatnonzero(current["side_name"].astype(str).eq(side).to_numpy())
        event = bundle["event"][side]
        raw = _predict_proba(event["model"], matrix.iloc[positions][event["features"]].fillna(event["median"]))
        result.loc[result.index[positions], raw_columns] = raw
        for label, side_models in bundle["payoff"].items():
            record = side_models[side]
            result.loc[result.index[positions], label] = _predict_quantile(record["model"], matrix.iloc[positions][record["features"]].fillna(record["median"]))
    timeout_matrix = matrix.copy()
    timeout_matrix["__side_long__"] = current["side_name"].astype(str).eq("long").astype(float)
    timeout = bundle["timeout"]
    pooled = _predict_quantile(timeout["model"], timeout_matrix[timeout["features"]].fillna(timeout["median"]))
    for side in SIDES:
        positions = np.flatnonzero(current["side_name"].astype(str).eq(side).to_numpy())
        record = timeout["side_quantiles"][side]
        weight = record["rows"] / (record["rows"] + TIMEOUT_SHRINK_ROWS)
        result.loc[result.index[positions], "q75_timeout_bps"] = (
            weight * pooled[positions] + (1.0 - weight) * record["q75"]
        )
    for side in SIDES:
        positions = np.flatnonzero(current["side_name"].astype(str).eq(side).to_numpy())
        calibrated = apply_class_calibrators(
            result.iloc[positions][raw_columns].to_numpy(float), bundle["calibrators"][side], current.iloc[positions]["era"].to_numpy(str)
        )
        result.loc[result.index[positions], ["p_positive", "p_adverse_negative", "p_timeout_negative", "p_other_negative"]] = calibrated
    return compose_tail_scores(result)


def _load_current_exact_labels(labels_path: Path, stage1_current_path: Path) -> pd.DataFrame:
    labels = pd.read_parquet(labels_path)
    stage1 = pd.read_parquet(stage1_current_path)
    needed = [*IDENTITY, "adverse_1atr_reached"]
    current = labels.merge(stage1.loc[:, needed], on=list(IDENTITY), how="inner", validate="one_to_one")
    if len(current) != len(labels):
        raise ValueError("current exact label/adverse join incomplete")
    net = pd.to_numeric(current["execution_net_ev_12h"], errors="raise")
    current["positive_net"] = net.gt(0).astype(np.int8)
    current["adverse_first"] = (current["adverse_1atr_reached"].astype(bool) & net.le(0)).astype(np.int8)
    current["timeout_event"] = (
        current["execution_exit_reason"].astype(str).eq("timeout")
        & net.le(0)
        & ~current["adverse_first"].astype(bool)
    ).astype(np.int8)
    return current


def current_economics(scored: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    top_count = max(1, int(math.ceil(.10 * len(scored))))
    selected = scored.sort_values(["mapped_tail_ev_bps", "candidate_id"], ascending=[False, True], kind="stable").iloc[:top_count]
    for scope, local in (("global_top10", selected), ("all", scored)):
        net = local["execution_net_ev_12h"].to_numpy(float) * 1e4
        rows.append({
            "scope": scope,
            "rows": len(local),
            "net_ev_bps": float(net.mean()),
            "positive_precision": float((net > 0).mean()),
            "cvar05_bps": float(np.mean(np.sort(net)[:max(1, int(math.ceil(.05 * len(net))))])),
            "long_rows": int(local["side_name"].astype(str).eq("long").sum()),
            "short_rows": int(local["side_name"].astype(str).eq("short").sum()),
        })
    support = []
    for side in SIDES:
        local = selected.loc[selected["side_name"].astype(str).eq(side)]
        support.append({"side": side, "top10_rows": len(local), "coverage": len(local) / len(scored), "support_safeguard_pass": bool(len(local) >= 50 and len(local) / len(scored) >= .01)})
    return pd.DataFrame(rows), pd.DataFrame(support)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    input_manifest = json.loads(args.dataset_dir.joinpath("manifest.json").read_text())
    dataset_path = args.dataset_dir / "cross_era_tail_payoff_dataset.parquet"
    if sha256(dataset_path) != input_manifest["outputs"]["dataset"]["sha256"]:
        raise ValueError("cross-era dataset hash mismatch")
    contract = json.loads(args.dataset_dir.joinpath("feature_contract.json").read_text())
    history = pd.read_parquet(dataset_path)
    history["__ts__"] = pd.to_datetime(history["__ts__"], utc=True)
    history["label_resolution_utc"] = pd.to_datetime(history["label_resolution_utc"], utc=True)
    # The materialised source may contain final July 19 signals whose 12h label
    # resolves on July 20.  Exclude them rather than treating them as historical
    # support for a July 20 frozen-current evaluation.
    history = history.loc[history["label_resolution_utc"].lt(CURRENT_START)].reset_index(drop=True)
    if history.empty:
        raise ValueError("strict historical cutoff removed every source row")
    history, composites = add_regime_composites(history)
    arms = feature_arms(contract, composites)
    if args.arm:
        if args.arm not in arms:
            raise ValueError(f"unknown arm {args.arm}; choices={sorted(arms)}")
        arms = {args.arm: arms[args.arm]}
    configs = tuple(config for config in HPO_CONFIGS if not args.hpo_name or config["name"] == args.hpo_name)
    if not configs:
        raise ValueError(f"unknown hpo configuration {args.hpo_name}")
    winner, oof, trials = run_oof_trials(history, arms, args.seed, configs)
    winner_arm = winner["arm"]
    winner_config = winner["config"]
    bundle, final_state = _fit_final_models(history, arms[winner_arm], winner_config, oof, args.seed + 1_000_000)
    model_path = args.output_dir / "frozen_models.joblib"
    joblib.dump(bundle, model_path)
    frozen = {
        "schema": SCHEMA,
        "selection_status": "historical_oof_global_economics_only",
        "current_outcomes_used_for_selection": False,
        "dataset": _binding(dataset_path),
        "dataset_manifest": _binding(args.dataset_dir / "manifest.json"),
        "feature_contract": _binding(args.dataset_dir / "feature_contract.json"),
        "winner": {"trial_key": winner["trial_key"], "arm": winner_arm, "config": winner_config, "metrics": winner["metrics"]},
        "event_classes": list(CLASS_NAMES),
        "feature_arms": {name: len(columns) for name, columns in arms.items()},
        "final_state": final_state,
        "model": _binding(model_path),
        "mapping": {"type": "causal_21d_side_shrunk_isotonic", "window_days": MAP_DAYS, "side_shrink_rows": MAP_SHRINK_ROWS},
        "contract": {
            "ranking": "one pooled global top-k after causal mapping; never timestamp-local",
            "costs": "exact execution_net_ev_12h is already cost-aware; no cost is deducted again",
            "actions": "no timing, target-price, wait, or timeout action heads",
            "support": "support safeguard is reporting only, never a fixed side threshold",
        },
    }
    frozen_path = args.output_dir / "frozen_before_current_evaluation.json"
    _write_json(frozen_path, frozen)
    frozen_sha = sha256(frozen_path)

    # Current causal features are opened only after all arm/HPO/model/mapping
    # decisions are frozen.  The exact current label files are not opened yet.
    current_features = _prepare_current_features(args.current_packb, arms[winner_arm])
    current_scores = score_current(bundle, current_features)
    current_scores["mapped_tail_ev_bps"], current_mapping = causal_side_shrunk_isotonic(oof, "tail_ev_bps", current=current_scores)
    current_score_path = args.output_dir / "current_predictions_before_outcomes.parquet"
    current_scores.to_parquet(current_score_path, index=False)
    score_sha = sha256(current_score_path)

    # One exact evaluation, after the state and score artifact are sealed.
    labels = _load_current_exact_labels(args.current_labels, args.current_stage1)
    scored = current_scores.merge(labels, on=list(IDENTITY), how="inner", validate="one_to_one")
    if len(scored) != len(current_scores):
        raise ValueError("current score/exact label join incomplete")
    economics, support = current_economics(scored)
    outputs: dict[str, Any] = {}
    tables: dict[str, pd.DataFrame] = {
        "historical_oof_winner": oof,
        "historical_trial_metrics": trials,
        "current_predictions_before_outcomes": current_scores,
        "current_scored_exact": scored,
        "current_economics": economics,
        "current_support_safeguard": support,
    }
    for name, table in tables.items():
        suffix = ".parquet" if name in {"historical_oof_winner", "current_predictions_before_outcomes", "current_scored_exact"} else ".csv"
        path = args.output_dir / f"{name}{suffix}"
        if name == "current_predictions_before_outcomes":
            # Keep the sealed original and avoid changing its lineage.
            path = current_score_path
        elif suffix == ".parquet":
            table.to_parquet(path, index=False)
        else:
            table.to_csv(path, index=False)
        outputs[name] = {**_binding(path), "rows": len(table)}
    report = {
        "schema": SCHEMA,
        "status": "completed_research_only_no_promotion",
        "promotion_eligible": False,
        "current_outcomes_used_for_selection": False,
        "frozen_state": {"path": str(frozen_path), "sha256_before_current_features": frozen_sha},
        "current_score_before_outcomes": {"path": str(current_score_path), "sha256_before_current_labels": score_sha},
        "winner": frozen["winner"],
        "current_mapping": current_mapping,
        "outputs": outputs,
    }
    _write_json(args.output_dir / "report.json", report)
    _write_json(args.output_dir / "manifest.json", {
        "schema": SCHEMA,
        "status": report["status"],
        "promotion_eligible": False,
        "frozen_state_sha256": frozen_sha,
        "report": _binding(args.output_dir / "report.json"),
        "outputs": outputs,
    })
    return report


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-dir", type=Path, default=Path("data_perp/artifacts/cross_era_tail_payoff_dataset_20260730_v3"))
    p.add_argument("--current-packb", type=Path, default=Path("data_perp/artifacts/execution_ev_july20_23_retrospective_20260730_v2/packb/packb_forward_context.parquet"))
    p.add_argument("--current-labels", type=Path, default=Path("data_perp/artifacts/execution_ev_july20_23_retrospective_20260730_v2/labels_12h/execution_ev_policy_labels.parquet"))
    p.add_argument("--current-stage1", type=Path, default=Path("data_perp/artifacts/historical_to_july_meaningful_mfe_gate_challenger_20260730_v2/current_predictions.parquet"))
    p.add_argument("--output-dir", type=Path, default=Path("data_perp/artifacts/cross_era_tail_payoff_challenger_20260730_v2"))
    p.add_argument("--arm", choices=("raw", "raw_context", "raw_context_regime"))
    p.add_argument("--hpo-name", choices=tuple(config["name"] for config in HPO_CONFIGS))
    p.add_argument("--seed", type=int, default=20260730)
    return p


if __name__ == "__main__":
    run(parser().parse_args())
