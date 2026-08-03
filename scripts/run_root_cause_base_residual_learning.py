#!/usr/bin/env python3
"""Stage 3--4 base/residual learning-efficiency and metric-concordance audit.

This is deliberately a diagnostic runner.  It neither ranks a live book nor
changes entry, execution, policy, sizing, or portfolio code.  It uses the raw
causal base matrix to predict the native soft-alpha label, then fits a separate
*stopped-gradient* residual learner for gross H12 economics.  The residual
learner only sees chronological inner-OOF base predictions on its training
rows; it never consumes a same-fit base score.

The runner is intentionally strict about the three-way identity join requested
by the root-cause roadmap:
  diagnostic substrate <-> raw causal feature panel <-> existing OOF stack.
The existing stack is recorded as a frozen reference only, never used as a new
model input.  All newly trained base and residual models use raw causal fields
from ``raw_feature_contract.json`` only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, LinearRegression, Ridge
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

try:
    import lightgbm as lgb
except ImportError:  # pragma: no cover - an actionable runtime failure
    lgb = None

try:
    from catboost import CatBoostRegressor
except ImportError:  # pragma: no cover - optional generic ladder baseline
    CatBoostRegressor = None


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
ART = ROOT / "data_perp" / "artifacts"
# v4 is the pointer-pinned canonical Stage-0 substrate.  Its ledger is
# byte-identical to v3; v4 fixes the provenance/runner seal.
DEFAULT_SUBSTRATE = ART / "root_cause_diagnostic_substrate_20260731_v4" / "diagnostic_row_ledger.parquet"
DEFAULT_RAW_PANEL = ART / "long_exact_h12_raw_base_panel_20260730_v2" / "raw_base_panel.parquet"
DEFAULT_FEATURE_CONTRACT = ART / "long_exact_h12_raw_base_panel_20260730_v2" / "raw_feature_contract.json"
DEFAULT_STACK = ART / "reconstructed_base_residual_stack_2022_2024_20260730_v3" / "oof_scores.parquet"
DEFAULT_OUTPUT = ART / "root_cause_base_residual_learning_20260731_v1"

ID = "candidate_id"
SIDE = "side_name"
TIME = "__ts__"
ALPHA = "__reconstructed_soft_alpha_12h__"
GROSS_CANDIDATES = ("gross_h12_bps", "execution_gross_ev_12h", "exact_h12_gross_bps")
NET_CANDIDATES = ("net_h12_bps", "execution_net_ev_12h", "exact_h12_net_bps")
LABEL_END_CANDIDATES = ("label_available_ts", "__label_available_at__", "execution_label_available_at")
SEEDS = (20260731, 20260801, 20260802)
DETERMINISTIC_FAMILIES = frozenset(("prior", "ridge", "gam_additive_boosted_stumps"))
HORIZON = pd.Timedelta(hours=12)
EMBARGO = pd.Timedelta(hours=12)
TOPS = (0.01, 0.05, 0.10, 0.20)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _dump(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")


def _utc(value: pd.Series) -> pd.Series:
    return pd.to_datetime(value, utc=True, errors="raise")


def _one(frame: pd.DataFrame, names: Iterable[str], role: str) -> str:
    found = [name for name in names if name in frame]
    if len(found) != 1:
        raise ValueError(f"{role} must resolve to exactly one column; found {found}")
    return found[0]


def _first(frame: pd.DataFrame, names: Iterable[str], role: str) -> str:
    """Choose the explicitly ordered canonical alias, never an arbitrary set."""
    for name in names:
        if name in frame:
            return name
    raise ValueError(f"{role} has none of the canonical aliases: {list(names)}")


def _stable_ids(values: pd.Series) -> str:
    return hashlib.sha256("\n".join(values.astype(str)).encode()).hexdigest()


def _safe_float_matrix(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    return frame.loc[:, features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)


def _progress(message: str) -> None:
    """Unbuffered runner progress: long research runs must be observable."""
    print(f"[base-residual-diagnostic] {message}", flush=True)


def _bounded_chronological_sample(frame: pd.DataFrame, maximum_rows: int | None) -> pd.DataFrame:
    """Bound compute without silently discarding older regimes.

    Uniform positions preserve chronological support; unlike a recent tail this
    leaves every historical regime represented.  The cap is recorded in
    lineage, so it cannot be mistaken for a full-data result.
    """
    if maximum_rows is None or len(frame) <= maximum_rows:
        return frame
    ordered = frame.sort_values([TIME, ID], kind="stable").reset_index(drop=True)
    positions = np.linspace(0, len(ordered) - 1, num=maximum_rows, dtype=int)
    return ordered.iloc[np.unique(positions)].copy()


def _raw_feature_names(contract: Path, frame: pd.DataFrame) -> list[str]:
    payload = json.loads(contract.read_text())
    names = list(map(str, payload.get("raw_feature_columns", [])))
    if not names:
        raise ValueError("raw feature contract has no raw_feature_columns")
    missing = sorted(set(names).difference(frame.columns))
    if missing:
        raise ValueError(f"raw causal matrix missing contract fields: {missing[:10]}")
    # Guard only against semantic *tokens*, not arbitrary substrings.  For
    # example ``volume_zscore`` and ``oi_recovery`` are legitimate causal raw
    # fields; the former is not a model score and the latter is not an outcome.
    # The contract is otherwise the authority for this runner.  Stage 2 owns
    # field-level availability/provenance auditing.
    forbidden = re.compile(
        r"(?:^|_)(?:future|label|target|outcome|execution|residual)(?:_|$)",
        flags=re.IGNORECASE,
    )
    exact_targets = {ALPHA, *GROSS_CANDIDATES, *NET_CANDIDATES, *LABEL_END_CANDIDATES}
    bad = [name for name in names if forbidden.search(name) or name in exact_targets]
    if bad:
        raise ValueError(f"raw feature contract contains target/proxy field(s): {bad[:10]}")
    return names


@dataclass(frozen=True)
class Fold:
    name: str
    start: pd.Timestamp
    end: pd.Timestamp
    split: str


def fixed_folds() -> list[Fold]:
    """Two sealed four-month outer windows; monthly slices remain evaluation-only."""
    return [
        Fold("development_2024-04_to_07", pd.Timestamp("2024-04-01", tz="UTC"), pd.Timestamp("2024-08-01", tz="UTC"), "development_oof"),
        Fold("later_2024-08_to_11", pd.Timestamp("2024-08-01", tz="UTC"), pd.Timestamp("2024-12-01", tz="UTC"), "later_oos"),
    ]


def purged_train_mask(frame: pd.DataFrame, fold: Fold, *, time_col: str, label_available_col: str) -> pd.Series:
    """Use a true 12h label purge plus a separate 12h embargo."""
    decision = _utc(frame[time_col]); available = _utc(frame[label_available_col])
    return decision.lt(fold.start - HORIZON - EMBARGO) & available.le(fold.start - EMBARGO)


def _preprocess(features: pd.DataFrame, *, linear: bool) -> Pipeline:
    steps: list[tuple[str, Any]] = [("impute", SimpleImputer(strategy="median", add_indicator=False))]
    if linear:
        steps.append(("scale", StandardScaler()))
    return Pipeline(steps)


def make_estimator(family: str, seed: int) -> tuple[Any | None, bool]:
    """Return estimator and whether its input requires standardisation."""
    if family == "prior":
        return None, False
    if family == "ridge":
        return Ridge(alpha=30.0, random_state=seed), True
    if family == "gam_additive_boosted_stumps":
        if lgb is None: raise RuntimeError("lightgbm is required for gam_additive_boosted_stumps")
        # All-feature additive approximation.  Depth-1 stumps preserve the
        # GAM diagnostic intent without materialising a dense spline expansion.
        return lgb.LGBMRegressor(objective="huber", alpha=0.9, n_estimators=160, learning_rate=0.04, num_leaves=2, max_depth=1, min_child_samples=180, colsample_bytree=1.0, subsample=1.0, reg_lambda=20.0, reg_alpha=0.2, random_state=0, n_jobs=1, deterministic=True, force_col_wise=True, verbosity=-1), False
    if family == "shallow_lgbm":
        if lgb is None: raise RuntimeError("lightgbm is required for shallow_lgbm")
        return lgb.LGBMRegressor(objective="huber", alpha=0.9, n_estimators=120, learning_rate=0.05, num_leaves=15, max_depth=3, min_child_samples=180, colsample_bytree=0.80, subsample=0.85, reg_lambda=10.0, reg_alpha=0.10, random_state=seed, n_jobs=1, deterministic=True, force_col_wise=True, verbosity=-1), False
    if family == "shallow_catboost":
        if CatBoostRegressor is None: return None, False
        return CatBoostRegressor(loss_function="RMSE", iterations=150, depth=4, learning_rate=0.05, l2_leaf_reg=10.0, random_seed=seed, thread_count=1, verbose=False, allow_writing_files=False), False
    if family == "production_like_lgbm":
        if lgb is None: raise RuntimeError("lightgbm is required for production_like_lgbm")
        return lgb.LGBMRegressor(objective="huber", alpha=0.9, n_estimators=260, learning_rate=0.04, num_leaves=23, max_depth=5, min_child_samples=180, colsample_bytree=0.80, subsample=0.85, reg_lambda=15.0, reg_alpha=0.15, random_state=seed, n_jobs=1, deterministic=True, force_col_wise=True, verbosity=-1), False
    if family == "causal_capacity_oracle":
        if lgb is None: raise RuntimeError("lightgbm is required for causal_capacity_oracle")
        return lgb.LGBMRegressor(objective="huber", alpha=0.9, n_estimators=550, learning_rate=0.03, num_leaves=63, max_depth=-1, min_child_samples=80, colsample_bytree=0.95, subsample=0.90, reg_lambda=2.0, reg_alpha=0.0, random_state=seed, n_jobs=1, deterministic=True, force_col_wise=True, verbosity=-1), False
    if family == "future_feature_oracle":
        if lgb is None: raise RuntimeError("lightgbm is required for future_feature_oracle")
        return lgb.LGBMRegressor(objective="huber", alpha=0.9, n_estimators=260, learning_rate=0.04, num_leaves=31, max_depth=6, min_child_samples=120, colsample_bytree=1.0, subsample=0.9, reg_lambda=5.0, reg_alpha=0.0, random_state=seed, n_jobs=1, deterministic=True, force_col_wise=True, verbosity=-1), False
    if family == "hist_gradient":
        return HistGradientBoostingRegressor(max_iter=220, max_leaf_nodes=31, l2_regularization=10.0, learning_rate=0.05, random_state=seed), False
    raise ValueError(f"unknown model family: {family}")


def _fit_predict(family: str, seed: int, x_train: pd.DataFrame, y_train: np.ndarray, x_eval: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    estimator, linear = make_estimator(family, seed)
    if estimator is None:
        value = float(np.mean(y_train))
        return np.full(len(x_train), value), np.full(len(x_eval), value), {"kind": "prior", "constant": value}
    transform = _preprocess(x_train, linear=linear)
    train_matrix = transform.fit_transform(x_train)
    eval_matrix = transform.transform(x_eval)
    estimator.fit(train_matrix, y_train)
    # LightGBM can emit a feature-name warning after the intentionally shared
    # preprocessing transform returns an anonymous numeric matrix.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="X does not have valid feature names", category=UserWarning)
        train_prediction = estimator.predict(train_matrix)
        eval_prediction = estimator.predict(eval_matrix)
    return train_prediction, eval_prediction, {"kind": family, "preprocess": {"imputer": "median", "scaled": linear}}


def _inner_oof_base(
    train: pd.DataFrame, features: list[str], alpha_col: str, label_available_col: str,
    family: str, seed: int, minimum_rows: int,
) -> pd.DataFrame:
    """Return OOF alpha rows only; early rows remain unavailable, never in-sample."""
    ordered = train.sort_values([TIME, ID], kind="stable").reset_index(drop=True)
    cuts = np.quantile(ordered[TIME].astype("int64"), [0.45, 0.62, 0.79])
    parts: list[pd.DataFrame] = []
    for index, cut in enumerate(np.unique(cuts.astype("int64"))):
        start = pd.Timestamp(int(cut), tz="UTC")
        inner_train = ordered.loc[_utc(ordered[label_available_col]).le(start - EMBARGO)].copy()
        inner_test = ordered.loc[(_utc(ordered[TIME]).ge(start)) & (_utc(ordered[TIME]).lt(start + pd.Timedelta(days=60)))].copy()
        if len(inner_train) < minimum_rows or len(inner_test) < 20:
            continue
        xtr, xte = _safe_float_matrix(inner_train, features), _safe_float_matrix(inner_test, features)
        _, pred, _ = _fit_predict(family, seed + 100 + index, xtr, inner_train[alpha_col].to_numpy(float), xte)
        part = inner_test[[ID, TIME, SIDE]].copy()
        part["stopped_gradient_base_alpha_oof"] = pred
        part["inner_fold"] = index
        parts.append(part)
    if not parts:
        return pd.DataFrame(columns=[ID, TIME, SIDE, "stopped_gradient_base_alpha_oof", "inner_fold"])
    output = pd.concat(parts, ignore_index=True).sort_values([TIME, ID], kind="stable")
    return output.drop_duplicates(ID, keep="first")


def _cached_prequential_base_oof(
    frame: pd.DataFrame, features: list[str], alpha_col: str, label_available_col: str,
    family: str, seed: int, minimum_rows: int, max_training_rows: int | None = None,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Build one reusable strictly-prequential base-OOF series per side/family.

    Each predicted row is fitted only on labels available before that quarter's
    start.  Outer folds merely subset this frozen prequential series, avoiding
    repeated fits of the same earlier-time base model across outer folds.
    """
    ordered = frame.sort_values([TIME, ID], kind="stable").reset_index(drop=True)
    first = _utc(ordered[TIME]).min() + pd.DateOffset(months=6)
    first = pd.Timestamp(first.year, first.month, 1, tz="UTC")
    starts = pd.date_range(first, _utc(ordered[TIME]).max(), freq="QS", tz="UTC")
    parts: list[pd.DataFrame] = []; states: list[dict[str, Any]] = []
    for index, start in enumerate(starts):
        inner_train = ordered.loc[_utc(ordered[label_available_col]).le(start - EMBARGO)].copy()
        uncapped_rows = len(inner_train)
        inner_train = _bounded_chronological_sample(inner_train, max_training_rows)
        inner_test = ordered.loc[(_utc(ordered[TIME]).ge(start)) & (_utc(ordered[TIME]).lt(start + pd.DateOffset(months=3)))].copy()
        if len(inner_train) < minimum_rows or len(inner_test) < 20:
            continue
        xtr, xte = _safe_float_matrix(inner_train, features), _safe_float_matrix(inner_test, features)
        _, pred, _ = _fit_predict(family, seed + 100 + index, xtr, inner_train[alpha_col].to_numpy(float), xte)
        part = inner_test[[ID, TIME, SIDE]].copy()
        part["stopped_gradient_base_alpha_oof"] = pred
        part["inner_fold"] = f"prequential_q_{start:%Y%m}"
        parts.append(part)
        states.append({"family": family, "seed": seed, "side": str(ordered[SIDE].iloc[0]), "prequential_start": str(start), "prequential_train_rows": int(len(inner_train)), "prequential_uncapped_train_rows": int(uncapped_rows), "prequential_training_row_cap": max_training_rows, "prequential_test_rows": int(len(inner_test)), "fit_role": "cached_inner_base_oof"})
    if not parts:
        return pd.DataFrame(columns=[ID, TIME, SIDE, "stopped_gradient_base_alpha_oof", "inner_fold"]), states
    output = pd.concat(parts, ignore_index=True).sort_values([TIME, ID], kind="stable")
    if output[ID].duplicated().any():
        raise AssertionError("prequential OOF windows must be non-overlapping")
    return output, states


def _normal_probability(prediction: np.ndarray, train_prediction: np.ndarray, train_alpha: np.ndarray) -> np.ndarray:
    # The probability is a fixed train-scale diagnostic transform, not a
    # learned policy calibrator.  It remains strictly train-derived.
    scale = max(float(np.std(train_alpha - train_prediction)), 1e-4)
    z = np.clip((prediction - 0.5) / scale, -12.0, 12.0)
    return 1.0 / (1.0 + np.exp(-z))


def _ece(y: np.ndarray, probability: np.ndarray, bins: int = 10) -> float:
    cuts = np.linspace(0.0, 1.0, bins + 1)
    total = len(y); result = 0.0
    for lo, hi in zip(cuts[:-1], cuts[1:]):
        mask = (probability >= lo) & ((probability < hi) if hi < 1.0 else (probability <= hi))
        if mask.any(): result += mask.mean() * abs(float(y[mask].mean()) - float(probability[mask].mean()))
    return float(result)


def _tail_metric(y_gross: np.ndarray, y_net: np.ndarray, score: np.ndarray, fraction: float) -> tuple[float, float, int]:
    count = max(1, int(np.ceil(len(score) * fraction)))
    index = np.argsort(-np.nan_to_num(score, nan=-np.inf), kind="stable")[:count]
    return float(np.mean(y_gross[index])), float(np.mean(y_net[index])), int(count)


def base_directional_metrics(frame: pd.DataFrame, *, alpha_true: str, alpha_pred: str) -> dict[str, Any]:
    """Metrics owned by the directional/soft-alpha base layer only."""
    alpha = frame[alpha_true].to_numpy(float)
    predicted_alpha = frame[alpha_pred].to_numpy(float)
    binary = (alpha >= 0.5).astype(int)
    probability = np.clip(frame["direction_probability"].to_numpy(float), 1e-6, 1.0 - 1e-6)
    return {
        "base_directional__rows": int(len(frame)),
        "base_directional__mae": float(np.mean(np.abs(alpha - predicted_alpha))),
        "base_directional__huber": float(np.mean(np.where(np.abs(alpha - predicted_alpha) <= .1, .5 * (alpha - predicted_alpha) ** 2, .1 * (np.abs(alpha - predicted_alpha) - .05)))),
        "base_directional__spearman_ic": float(spearmanr(alpha, predicted_alpha).statistic) if np.std(predicted_alpha) > 0 else np.nan,
        "base_directional__prediction_dispersion": float(np.std(predicted_alpha)),
        "base_directional__roc_auc": float(roc_auc_score(binary, probability)) if len(np.unique(binary)) > 1 else np.nan,
        "base_directional__pr_auc": float(average_precision_score(binary, probability)) if len(np.unique(binary)) > 1 else np.nan,
        "base_directional__log_loss": float(log_loss(binary, probability, labels=[0, 1])),
        "base_directional__brier": float(brier_score_loss(binary, probability)),
        "base_directional__ece": _ece(binary, probability),
        # Calibration is of the train-derived directional probability against
        # the soft-alpha target (not against later economics).
        "base_directional__calibration_slope": float(np.polyfit(probability, alpha, 1)[0]) if np.std(probability) > 1e-9 else np.nan,
        "base_directional__calibration_intercept": float(np.polyfit(probability, alpha, 1)[1]) if np.std(probability) > 1e-9 else np.nan,
    }


def residual_economic_metrics(frame: pd.DataFrame, *, gross_true: str, net_true: str, gross_pred: str) -> dict[str, Any]:
    """Metrics owned by the stopped-gradient residual economic layer only."""
    gross = frame[gross_true].to_numpy(float); net = frame[net_true].to_numpy(float)
    predicted_gross = frame[gross_pred].to_numpy(float)
    result: dict[str, Any] = {
        "residual_economic__rows": int(len(frame)),
        "residual_economic__mae_bps": float(np.mean(np.abs(gross - predicted_gross))),
        "residual_economic__huber_bps": float(np.mean(np.where(np.abs(gross - predicted_gross) <= 100., .5 * (gross - predicted_gross) ** 2, 100. * (np.abs(gross - predicted_gross) - 50.)))),
        "residual_economic__spearman_ic": float(spearmanr(gross, predicted_gross).statistic) if np.std(predicted_gross) > 0 else np.nan,
        "residual_economic__prediction_dispersion_bps": float(np.std(predicted_gross)),
        "residual_economic__threshold_gross_bps": float(gross[predicted_gross > 0].mean()) if (predicted_gross > 0).any() else np.nan,
        "residual_economic__threshold_net_bps": float(net[predicted_gross > 0].mean()) if (predicted_gross > 0).any() else np.nan,
        "residual_economic__threshold_rows": int((predicted_gross > 0).sum()),
        "residual_economic__calibration_slope": float(np.polyfit(predicted_gross, gross, 1)[0]) if np.std(predicted_gross) > 1e-9 else np.nan,
        "residual_economic__calibration_intercept_bps": float(np.polyfit(predicted_gross, gross, 1)[1]) if np.std(predicted_gross) > 1e-9 else np.nan,
        "residual_economic__gross_mean_bps": float(gross.mean()),
        "residual_economic__net_mean_bps": float(net.mean()),
    }
    for fraction in TOPS:
        top_gross, top_net, count = _tail_metric(gross, net, predicted_gross, fraction)
        token = str(int(fraction * 100))
        result[f"residual_economic__gross_top{token}_bps"] = top_gross; result[f"residual_economic__net_top{token}_bps"] = top_net; result[f"residual_economic__top{token}_rows"] = count
    return result


def frozen_net_mapping_reference_metrics(frame: pd.DataFrame, *, net_true: str, net_pred: str) -> dict[str, Any]:
    """Descriptive metrics for the sealed stack's explicitly net-mapped score."""
    net = frame[net_true].to_numpy(float); prediction = frame[net_pred].to_numpy(float)
    result: dict[str, Any] = {
        "frozen_net_mapping_reference__rows": int(len(frame)),
        "frozen_net_mapping_reference__mae_bps": float(np.mean(np.abs(net - prediction))),
        "frozen_net_mapping_reference__huber_bps": float(np.mean(np.where(np.abs(net - prediction) <= 100., .5 * (net - prediction) ** 2, 100. * (np.abs(net - prediction) - 50.)))),
        "frozen_net_mapping_reference__spearman_ic": float(spearmanr(net, prediction).statistic) if np.std(prediction) > 0 else np.nan,
        "frozen_net_mapping_reference__prediction_dispersion_bps": float(np.std(prediction)),
        "frozen_net_mapping_reference__threshold_net_bps": float(net[prediction > 0].mean()) if (prediction > 0).any() else np.nan,
        "frozen_net_mapping_reference__threshold_rows": int((prediction > 0).sum()),
        "frozen_net_mapping_reference__calibration_slope": float(np.polyfit(prediction, net, 1)[0]) if np.std(prediction) > 1e-9 else np.nan,
        "frozen_net_mapping_reference__calibration_intercept_bps": float(np.polyfit(prediction, net, 1)[1]) if np.std(prediction) > 1e-9 else np.nan,
        "frozen_net_mapping_reference__net_mean_bps": float(net.mean()),
    }
    for fraction in TOPS:
        count = max(1, int(np.ceil(len(prediction) * fraction)))
        index = np.argsort(-np.nan_to_num(prediction, nan=-np.inf), kind="stable")[:count]
        token = str(int(fraction * 100))
        result[f"frozen_net_mapping_reference__net_top{token}_bps"] = float(np.mean(net[index]))
        result[f"frozen_net_mapping_reference__top{token}_rows"] = count
    return result


def _economic_mapper(base_oof: pd.DataFrame, gross_col: str) -> LinearRegression:
    model = LinearRegression()
    model.fit(base_oof[["stopped_gradient_base_alpha_oof"]], base_oof[gross_col])
    return model


def _score_family_side(
    train: pd.DataFrame, test: pd.DataFrame, *, family: str, seed: int, features: list[str],
    alpha_col: str, gross_col: str, label_available_col: str, minimum_rows: int, cached_inner_oof: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    x_train, x_test = _safe_float_matrix(train, features), _safe_float_matrix(test, features)
    base_train, base_test, state = _fit_predict(family, seed, x_train, train[alpha_col].to_numpy(float), x_test)
    inner = _inner_oof_base(train, features, alpha_col, label_available_col, family, seed, minimum_rows) if cached_inner_oof is None else cached_inner_oof.loc[cached_inner_oof[ID].isin(train[ID])].copy()
    train_for_residual = train.merge(inner[[ID, "stopped_gradient_base_alpha_oof"]], on=ID, how="inner", validate="one_to_one")
    if len(train_for_residual) < minimum_rows:
        raise ValueError(f"{family}: insufficient stopped-gradient inner OOF rows ({len(train_for_residual)})")
    mapper = _economic_mapper(train_for_residual, gross_col)
    train_for_residual["base_economic_oof_bps"] = mapper.predict(train_for_residual[["stopped_gradient_base_alpha_oof"]])
    train_for_residual["economic_residual_target_bps"] = train_for_residual[gross_col] - train_for_residual["base_economic_oof_bps"]
    residual_features = [*features, "stopped_gradient_base_alpha_oof"]
    res_x_train = _safe_float_matrix(train_for_residual, residual_features)
    residual_test_input = test.loc[:, features].copy(); residual_test_input["stopped_gradient_base_alpha_oof"] = base_test
    res_x_test = _safe_float_matrix(residual_test_input, residual_features)
    residual_train, residual_test, residual_state = _fit_predict(family, seed + 10_000, res_x_train, train_for_residual["economic_residual_target_bps"].to_numpy(float), res_x_test)
    base_economic_train = mapper.predict(pd.DataFrame({"stopped_gradient_base_alpha_oof": base_train}))
    base_economic_test = mapper.predict(pd.DataFrame({"stopped_gradient_base_alpha_oof": base_test}))
    out = test[[ID, TIME, SIDE]].copy()
    out["base_alpha_prediction"] = base_test
    out["base_economic_prediction_bps"] = base_economic_test
    out["residual_prediction_bps"] = residual_test
    out["combined_economic_prediction_bps"] = base_economic_test + residual_test
    out["direction_probability"] = _normal_probability(base_test, base_train, train[alpha_col].to_numpy(float))
    out["evaluation_scope"] = "outer_heldout"
    # Train diagnostics are deliberately marked in-sample.  They quantify the
    # train--heldout gap and are never treated as OOF performance.  Residual
    # training is evaluated only on the rows with cached prequential base OOF.
    base_train_frame = train[[ID, TIME, SIDE]].copy()
    base_train_frame["base_alpha_prediction"] = base_train
    base_train_frame["base_economic_prediction_bps"] = base_economic_train
    train_lookup = base_train_frame.set_index(ID)
    train_out = train_for_residual[[ID, TIME, SIDE]].copy()
    train_out["base_alpha_prediction"] = train_lookup.loc[train_out[ID], "base_alpha_prediction"].to_numpy(float)
    train_out["base_economic_prediction_bps"] = train_lookup.loc[train_out[ID], "base_economic_prediction_bps"].to_numpy(float)
    train_out["residual_prediction_bps"] = residual_train
    train_out["combined_economic_prediction_bps"] = train_out["base_economic_prediction_bps"].to_numpy(float) + residual_train
    train_out["direction_probability"] = _normal_probability(train_out["base_alpha_prediction"].to_numpy(float), base_train, train[alpha_col].to_numpy(float))
    train_out["evaluation_scope"] = "train_in_sample"
    state.update({"residual_state": residual_state, "inner_oof_rows": int(len(train_for_residual)), "inner_oof_source": "cached_prequential" if cached_inner_oof is not None else "fold_local", "mapper_coef": float(mapper.coef_[0]), "mapper_intercept": float(mapper.intercept_), "stopped_gradient": True, "residual_target": "gross_h12_bps - frozen_train_oof_base_to_gross_map"})
    return pd.concat([out, train_out], ignore_index=True), state


def _run_family(
    frame: pd.DataFrame, folds: list[Fold], *, family: str, seed: int, features: list[str], alpha_col: str,
    gross_col: str, net_col: str, label_available_col: str, minimum_rows: int, max_training_rows: int | None = None, cached_inner_by_side: dict[str, pd.DataFrame] | None = None,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    outputs: list[pd.DataFrame] = []; states: list[dict[str, Any]] = []
    for fold in folds:
        test_all = frame.loc[_utc(frame[TIME]).ge(fold.start) & _utc(frame[TIME]).lt(fold.end)]
        train_all = frame.loc[purged_train_mask(frame, fold, time_col=TIME, label_available_col=label_available_col)]
        for side in ("long", "short"):
            train, test = train_all.loc[train_all[SIDE].eq(side)].copy(), test_all.loc[test_all[SIDE].eq(side)].copy()
            uncapped_train_rows = len(train)
            train = _bounded_chronological_sample(train, max_training_rows)
            if len(train) < minimum_rows or test.empty: continue
            scored, state = _score_family_side(train, test, family=family, seed=seed, features=features, alpha_col=alpha_col, gross_col=gross_col, label_available_col=label_available_col, minimum_rows=minimum_rows, cached_inner_oof=(cached_inner_by_side or {}).get(side))
            source = pd.concat([train, test], ignore_index=True).drop_duplicates(ID, keep="last").set_index(ID)
            scored[alpha_col] = source.loc[scored[ID], alpha_col].to_numpy(float)
            scored[gross_col] = source.loc[scored[ID], gross_col].to_numpy(float)
            scored[net_col] = source.loc[scored[ID], net_col].to_numpy(float)
            scored["model_family"] = family; scored["seed"] = seed; scored["fold"] = fold.name; scored["split"] = fold.split
            state.update({"model_family": family, "seed": seed, "side": side, "fold": fold.name, "split": fold.split, "train_rows": int(len(train)), "uncapped_train_rows": int(uncapped_train_rows), "training_row_cap": max_training_rows, "seed_dispersion_structural_zero": family in DETERMINISTIC_FAMILIES, "test_rows": int(len(test)), "train_max_label_available": str(_utc(train[label_available_col]).max()), "test_start": str(fold.start), "purge_hours": 12, "embargo_hours": 12})
            outputs.append(scored); states.append(state)
    return (pd.concat(outputs, ignore_index=True) if outputs else pd.DataFrame()), states


def _reference_predictions(frame: pd.DataFrame, alpha_col: str, gross_col: str, net_col: str) -> pd.DataFrame:
    needed = {"score_base_alpha", "score_base_expected_ev", "score_residual_expected_ev"}
    if not needed.issubset(frame): return pd.DataFrame()
    result = frame.loc[frame.score_base_alpha.notna(), [ID, TIME, SIDE, alpha_col, gross_col, net_col, "score_base_alpha", "score_base_expected_ev", "score_residual_expected_ev"]].copy()
    result["base_alpha_prediction"] = result.score_base_alpha
    # These frozen stack fields were mapped to execution *net* EV.  They are
    # retained as an explicitly net-mapped reference, never relabelled gross.
    result["base_net_reference_prediction_bps"] = result.score_base_expected_ev * 10_000.0
    result["residual_net_reference_prediction_bps"] = (result.score_residual_expected_ev - result.score_base_expected_ev) * 10_000.0
    result["combined_net_reference_prediction_bps"] = result.score_residual_expected_ev * 10_000.0
    result["direction_probability"] = np.clip(result.base_alpha_prediction, 1e-6, 1-1e-6)
    result["evaluation_scope"] = "frozen_oof_reference"
    result["model_family"] = "frozen_oof_reference"; result["seed"] = -1; result["fold"] = "stack_oof"; result["split"] = "reference_oof"
    return result.drop(columns=["score_base_alpha", "score_base_expected_ev", "score_residual_expected_ev"])


def _metrics_table(predictions: pd.DataFrame, alpha_col: str, gross_col: str, net_col: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, part in predictions.groupby(["model_family", "seed", "split", "fold", "evaluation_scope", SIDE], observed=True):
        family, seed, split, fold, evaluation_scope, side = keys
        common = {"model_family": family, "seed": seed, "split": split, "fold": fold, "evaluation_scope": evaluation_scope, "side": side}
        base = {**common, "component": "base_directional"}
        base.update(base_directional_metrics(part, alpha_true=alpha_col, alpha_pred="base_alpha_prediction"))
        rows.append(base)
        if family == "frozen_oof_reference":
            reference = {**common, "component": "frozen_net_mapping_reference"}
            # Reference expected-EV is net; report its economics separately
            # rather than mixing it with the new gross-residual layer.
            reference.update(frozen_net_mapping_reference_metrics(part, net_true=net_col, net_pred="combined_net_reference_prediction_bps"))
            rows.append(reference)
        else:
            residual = {**common, "component": "residual_economic"}
            residual.update(residual_economic_metrics(part, gross_true=gross_col, net_true=net_col, gross_pred="combined_economic_prediction_bps"))
            rows.append(residual)
    return pd.DataFrame(rows)


def _learning_curves(frame: pd.DataFrame, features: list[str], alpha_col: str, gross_col: str, net_col: str, label_available_col: str, seed: int, minimum_rows: int) -> pd.DataFrame:
    """Predeclared curves on July, always using rows chronologically before it."""
    fold = Fold("curve_2024-07", pd.Timestamp("2024-07-01", tz="UTC"), pd.Timestamp("2024-08-01", tz="UTC"), "development_oof")
    train_all = frame.loc[purged_train_mask(frame, fold, time_col=TIME, label_available_col=label_available_col)]
    test_all = frame.loc[_utc(frame[TIME]).ge(fold.start) & _utc(frame[TIME]).lt(fold.end)]
    rows: list[dict[str, Any]] = []
    curve_row_cap = 18_000
    feature_values = tuple(dict.fromkeys((min(32, len(features)), min(96, len(features)), len(features))))
    # Curves isolate data support, so use one stable low-variance learner here.
    # Re-running every capacity rung at every curve point is a combinatorial
    # compute expansion rather than new evidence; the separate capacity block
    # below supplies that comparison on the identical July holdout.
    for family in ("ridge",):
        for kind, values in (("sample_rows", (3_000, 9_000, 18_000)), ("history_months", (3, 6, 12)), ("feature_count", feature_values)):
            for value in values:
                for side in ("long", "short"):
                    train, test = train_all.loc[train_all[SIDE].eq(side)].copy(), test_all.loc[test_all[SIDE].eq(side)].copy()
                    if kind == "sample_rows": train = train.sort_values([TIME, ID], kind="stable").tail(min(int(value), len(train)))
                    elif kind == "history_months": train = train.loc[_utc(train[TIME]).ge(fold.start - pd.DateOffset(months=int(value)))]
                    elif kind == "feature_count": pass
                    train = _bounded_chronological_sample(train, curve_row_cap)
                    if len(train) < minimum_rows or len(test) < 20: continue
                    use = features[: min(int(value), len(features))] if kind == "feature_count" else features
                    try:
                        pred, _ = _score_family_side(train, test, family=family, seed=seed, features=use, alpha_col=alpha_col, gross_col=gross_col, label_available_col=label_available_col, minimum_rows=minimum_rows)
                    except ValueError as error:
                        if "insufficient stopped-gradient inner OOF rows" not in str(error): raise
                        rows.append({"curve_type": kind, "curve_value": int(value), "model_family": family, "side": side, "train_rows": len(train), "curve_row_cap": curve_row_cap, "features": len(use), "status": "INSUFFICIENT_SUPPORT", "minimum_rows": minimum_rows, "detail": str(error)})
                        continue
                    pred = pred.loc[pred.evaluation_scope.eq("outer_heldout")].copy()
                    pred[alpha_col] = test.set_index(ID).loc[pred[ID], alpha_col].to_numpy(float); pred[gross_col] = test.set_index(ID).loc[pred[ID], gross_col].to_numpy(float); pred[net_col] = test.set_index(ID).loc[pred[ID], net_col].to_numpy(float)
                    row = {"curve_type": kind, "curve_value": int(value), "model_family": family, "side": side, "train_rows": len(train), "curve_row_cap": curve_row_cap, "features": len(use)}
                    row.update(residual_economic_metrics(pred, gross_true=gross_col, net_true=net_col, gross_pred="combined_economic_prediction_bps")); rows.append(row)
    # Capacity is varied independently of feature/history support.  This is
    # a predeclared shallow/current-like/high-capacity LGBM ladder, not HPO.
    for capacity_rank, family in enumerate(("shallow_lgbm", "production_like_lgbm", "causal_capacity_oracle"), start=1):
        for side in ("long", "short"):
            train, test = train_all.loc[train_all[SIDE].eq(side)].copy(), test_all.loc[test_all[SIDE].eq(side)].copy()
            train = _bounded_chronological_sample(train, curve_row_cap)
            if len(train) < minimum_rows or len(test) < 20: continue
            try:
                pred, _ = _score_family_side(train, test, family=family, seed=seed, features=features, alpha_col=alpha_col, gross_col=gross_col, label_available_col=label_available_col, minimum_rows=minimum_rows)
            except ValueError as error:
                if "insufficient stopped-gradient inner OOF rows" not in str(error): raise
                rows.append({"curve_type": "capacity", "curve_value": capacity_rank, "capacity_label": family, "model_family": family, "side": side, "train_rows": len(train), "curve_row_cap": curve_row_cap, "features": len(features), "status": "INSUFFICIENT_SUPPORT", "minimum_rows": minimum_rows, "detail": str(error)})
                continue
            pred = pred.loc[pred.evaluation_scope.eq("outer_heldout")].copy()
            pred[alpha_col] = test.set_index(ID).loc[pred[ID], alpha_col].to_numpy(float); pred[gross_col] = test.set_index(ID).loc[pred[ID], gross_col].to_numpy(float); pred[net_col] = test.set_index(ID).loc[pred[ID], net_col].to_numpy(float)
            row = {"curve_type": "capacity", "curve_value": capacity_rank, "capacity_label": family, "model_family": family, "side": side, "train_rows": len(train), "curve_row_cap": curve_row_cap, "features": len(features)}
            row.update(residual_economic_metrics(pred, gross_true=gross_col, net_true=net_col, gross_pred="combined_economic_prediction_bps")); rows.append(row)
    return pd.DataFrame(rows)


def _synthetic_recovery(frame: pd.DataFrame, features: list[str], alpha_col: str, label_available_col: str, seed: int, minimum_rows: int) -> pd.DataFrame:
    """Recovery is measured on a held-out fold; the injected function is causal."""
    fold = Fold("synthetic_2024-07", pd.Timestamp("2024-07-01", tz="UTC"), pd.Timestamp("2024-08-01", tz="UTC"), "development_oof")
    train_all = frame.loc[purged_train_mask(frame, fold, time_col=TIME, label_available_col=label_available_col)]
    test_all = frame.loc[_utc(frame[TIME]).ge(fold.start) & _utc(frame[TIME]).lt(fold.end)]
    selected = features[: min(8, len(features))]
    result: list[dict[str, Any]] = []
    # Zero/medium/high injected strengths demonstrate the recovery threshold
    # without turning a diagnostic sanity check into an expensive HPO loop.
    for alpha in (0.0, 0.25, 0.50):
        for family in ("ridge", "production_like_lgbm"):
            for side in ("long", "short"):
                train, test = train_all.loc[train_all[SIDE].eq(side)].copy(), test_all.loc[test_all[SIDE].eq(side)].copy()
                train = _bounded_chronological_sample(train, 18_000)
                if len(train) < minimum_rows or len(test) < 20: continue
                # Fixed causal function: clipped first two causal columns after
                # train-only standardisation.  No target participates in it.
                trans = _preprocess(_safe_float_matrix(train, selected), linear=True); tr = trans.fit_transform(_safe_float_matrix(train, selected)); te = trans.transform(_safe_float_matrix(test, selected))
                signal_train = np.tanh(np.asarray(tr)[:, 0] + .5 * np.asarray(tr)[:, min(1, tr.shape[1] - 1)])
                signal_test = np.tanh(np.asarray(te)[:, 0] + .5 * np.asarray(te)[:, min(1, te.shape[1] - 1)])
                target = train[alpha_col].to_numpy(float) + alpha * signal_train
                _, pred, _ = _fit_predict(family, seed, _safe_float_matrix(train, selected), target, _safe_float_matrix(test, selected))
                recovered = float(np.corrcoef(pred - test[alpha_col].to_numpy(float), signal_test)[0, 1]) if np.std(pred) > 1e-9 and np.std(signal_test) > 1e-9 else np.nan
                result.append({"alpha": alpha, "model_family": family, "side": side, "rows": len(test), "known_causal_function": "tanh(z(feature_0)+0.5*z(feature_1))", "recovery_correlation": recovered, "recovery_slope": float(np.polyfit(signal_test, pred - test[alpha_col].to_numpy(float), 1)[0]) if np.std(signal_test) > 1e-9 else np.nan})
    return pd.DataFrame(result)


def _concordance(metrics_frame: pd.DataFrame) -> pd.DataFrame:
    """Does development base quality align with later residual economics?"""
    dev = metrics_frame.loc[metrics_frame.split.eq("development_oof") & metrics_frame.evaluation_scope.eq("outer_heldout") & metrics_frame.component.eq("base_directional")]
    later = metrics_frame.loc[metrics_frame.split.eq("later_oos") & metrics_frame.evaluation_scope.eq("outer_heldout") & metrics_frame.component.eq("residual_economic")]
    keys = ["model_family", "seed", "side"]
    left = dev.groupby(keys, as_index=False).mean(numeric_only=True)
    right = later.groupby(keys, as_index=False).mean(numeric_only=True)
    joined = left.merge(right, on=keys, suffixes=("_development", "_later"), validate="one_to_one")
    outcomes = ["residual_economic__gross_top10_bps_later", "residual_economic__net_top10_bps_later", "residual_economic__gross_top20_bps_later", "residual_economic__net_mean_bps_later", "residual_economic__gross_mean_bps_later"]
    candidates = ["base_directional__roc_auc", "base_directional__pr_auc", "base_directional__log_loss", "base_directional__brier", "base_directional__ece", "base_directional__spearman_ic", "base_directional__mae", "base_directional__huber", "base_directional__prediction_dispersion"]
    rows: list[dict[str, Any]] = []
    for metric in candidates:
        x = joined.get(f"{metric}_development")
        if x is None: continue
        for outcome in outcomes:
            y = joined.get(outcome)
            if y is None: continue
            valid = x.notna() & y.notna()
            rows.append({"development_base_metric": metric, "later_residual_economic_metric": outcome.removesuffix("_later"), "arms": int(valid.sum()), "pearson": float(x[valid].corr(y[valid], method="pearson")) if valid.sum() > 2 else np.nan, "spearman": float(x[valid].corr(y[valid], method="spearman")) if valid.sum() > 2 else np.nan})
    return pd.DataFrame(rows)


def _gap_and_seed_summary(metrics_frame: pd.DataFrame) -> pd.DataFrame:
    """Explicit train--heldout gaps, stochastic seed dispersion and named gaps."""
    held = metrics_frame.loc[metrics_frame.evaluation_scope.eq("outer_heldout")].copy()
    train = metrics_frame.loc[metrics_frame.evaluation_scope.eq("train_in_sample")].copy()
    rows: list[dict[str, Any]] = []
    measures = ("base_directional__mae", "base_directional__spearman_ic", "residual_economic__mae_bps", "residual_economic__spearman_ic", "residual_economic__net_top10_bps")
    keys = ["model_family", "seed", "split", "fold", "side", "component"]
    joined = train.merge(held, on=keys, suffixes=("_train", "_heldout"), validate="one_to_one")
    for measure in measures:
        left, right = f"{measure}_train", f"{measure}_heldout"
        if left not in joined or right not in joined:
            continue
        valid = joined[left].notna() & joined[right].notna()
        for _, row in joined.loc[valid].iterrows():
            rows.append({"record_type": "train_heldout_gap", "model_family": row.model_family, "seed": row.seed, "split": row.split, "fold": row.fold, "side": row.side, "component": row.component, "metric": measure, "train_value": float(row[left]), "heldout_value": float(row[right]), "train_minus_heldout": float(row[left] - row[right])})
    # Dispersion is computed only for genuinely repeated stochastic arms.
    aggregate = held.groupby(["model_family", "seed", "side", "component"], as_index=False).mean(numeric_only=True)
    for (family, side, component), part in aggregate.groupby(["model_family", "side", "component"], observed=True):
        for measure in measures:
            if measure not in part or part[measure].notna().sum() < 2:
                continue
            rows.append({"record_type": "seed_dispersion", "model_family": family, "side": side, "component": component, "metric": measure, "seed_count": int(part[measure].notna().sum()), "seed_std": float(part[measure].std(ddof=0)), "seed_min": float(part[measure].min()), "seed_max": float(part[measure].max()), "structural_zero": family in DETERMINISTIC_FAMILIES})
    # Named later-OOS economic gaps in bps.  Ratios are allowed only when the
    # null-to-causal denominator is strictly positive.
    later = held.loc[(held.split.eq("later_oos")) & (held.component.eq("residual_economic"))]
    means = later.groupby("model_family", as_index=True)["residual_economic__net_top10_bps"].mean()
    def add_gap(name: str, left: str, right: str) -> None:
        if left in means and right in means:
            rows.append({"record_type": "named_economic_gap", "comparison": name, "metric": "later_residual_economic__net_top10_bps", "left_bps": float(means[left]), "right_bps": float(means[right]), "right_minus_left_bps": float(means[right] - means[left]), "economic_regret_bps": float(means[right] - means[left])})
    add_gap("null_to_causal", "prior", "causal_capacity_oracle")
    add_gap("production_to_causal", "production_like_lgbm", "causal_capacity_oracle")
    add_gap("causal_to_future", "causal_capacity_oracle", "future_feature_oracle")
    if {"prior", "causal_capacity_oracle", "production_like_lgbm"}.issubset(means.index):
        denominator = float(means["causal_capacity_oracle"] - means["prior"])
        numerator = float(means["production_like_lgbm"] - means["prior"])
        rows.append({"record_type": "safe_efficiency_ratio", "comparison": "production_capture_of_null_to_causal_gap", "denominator_bps": denominator, "numerator_bps": numerator, "ratio": numerator / denominator if denominator > 0 else np.nan, "ratio_reported": denominator > 0})
    return pd.DataFrame(rows)


def _stage4_arm_outcomes(predictions: pd.DataFrame, net_col: str, gross_col: str) -> pd.DataFrame:
    """Later-OOS arm outcomes and paired whole-UTC-day bootstrap evidence."""
    later = predictions.loc[(predictions.split.eq("later_oos")) & (predictions.evaluation_scope.eq("outer_heldout"))].copy()
    rows: list[dict[str, Any]] = []
    selected_by_arm: dict[tuple[str, int], pd.DataFrame] = {}
    for (family, seed), part in later.groupby(["model_family", "seed"], observed=True):
        count = max(1, int(np.ceil(len(part) * .10)))
        selected = part.nlargest(count, "combined_economic_prediction_bps", keep="first").copy()
        selected_by_arm[(family, int(seed))] = selected
        monthly = selected.assign(month=_utc(selected[TIME]).dt.to_period("M").astype(str)).groupby("month")[[net_col, gross_col]].mean()
        side_values = selected.groupby(SIDE)[[net_col, gross_col]].mean()
        threshold = part.loc[part.combined_economic_prediction_bps.gt(0)]
        rows.append({"record_type": "arm_outcome", "model_family": family, "seed": seed, "rows": int(len(part)), "selected_global_top10_rows": count, "later_net_top10_bps": float(selected[net_col].mean()), "later_gross_top10_bps": float(selected[gross_col].mean()), "later_worst_month_net_bps": float(monthly[net_col].min()), "later_worst_month_gross_bps": float(monthly[gross_col].min()), "later_worst_side_net_bps": float(side_values[net_col].min()), "later_worst_side_gross_bps": float(side_values[gross_col].min()), "later_causal_threshold_gross_bps": float(threshold[gross_col].mean()) if len(threshold) else np.nan, "later_causal_threshold_rows": int(len(threshold))})
    baseline_key = next((key for key in selected_by_arm if key[0] == "prior"), None)
    baseline = selected_by_arm.get(baseline_key) if baseline_key is not None else None
    if baseline is not None:
        base_day = baseline.assign(day=_utc(baseline[TIME]).dt.floor("D")).groupby("day")[net_col].mean()
        rng = np.random.default_rng(20260731)
        for (family, seed), selected in selected_by_arm.items():
            if family == "prior":
                continue
            arm_day = selected.assign(day=_utc(selected[TIME]).dt.floor("D")).groupby("day")[net_col].mean()
            days = base_day.index.union(arm_day.index)
            delta = arm_day.reindex(days, fill_value=0.0).to_numpy(float) - base_day.reindex(days, fill_value=0.0).to_numpy(float)
            draws = np.array([delta[rng.integers(0, len(delta), len(delta))].mean() for _ in range(500)])
            rows.append({"record_type": "paired_day_bootstrap_vs_prior", "model_family": family, "seed": seed, "baseline": "prior", "paired_days": int(len(days)), "delta_net_top10_bps": float(delta.mean()), "bootstrap_ci_low_bps": float(np.quantile(draws, .025)), "bootstrap_ci_high_bps": float(np.quantile(draws, .975)), "bootstrap_probability_positive": float((draws > 0).mean())})
    return pd.DataFrame(rows)


def _load_causal_allowlist(path: Path | None, raw_features: list[str]) -> list[str]:
    """Read the Stage-2 approved list; absent means the sealed raw contract.

    The default is intentionally conservative: the raw contract already
    excludes outcomes, OOF scores and action fields.  Once Stage 2 provides a
    stricter allowlist, it can only remove raw fields, never add one.
    """
    if path is None:
        return raw_features
    if path.suffix == ".parquet":
        inventory = pd.read_parquet(path)
        if not {"feature_name", "causal_probe_eligible"}.issubset(inventory):
            raise ValueError("causal allowlist parquet must contain feature_name and causal_probe_eligible")
        requested = inventory.loc[inventory.causal_probe_eligible.astype(bool), "feature_name"].astype(str).tolist()
    else:
        payload = json.loads(path.read_text())
        if isinstance(payload, list):
            requested = list(map(str, payload))
        elif isinstance(payload, dict):
            requested = list(map(str, payload.get("causal_features", payload.get("approved_features", []))))
        else:
            raise ValueError("causal allowlist must be a JSON list/object or Stage-2 inventory Parquet")
    if not requested:
        raise ValueError("causal allowlist is empty")
    illegal = sorted(set(requested).difference(raw_features))
    if illegal:
        raise ValueError(f"Stage-2 allowlist expands raw contract: {illegal[:10]}")
    return [name for name in raw_features if name in set(requested)]


def _read_join(substrate: Path, raw_panel: Path, feature_contract: Path, stack: Path, causal_allowlist: Path | None = None) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    ledger = pd.read_parquet(substrate)
    raw = pd.read_parquet(raw_panel)
    stack_frame = pd.read_parquet(stack)
    for name, part in (("substrate", ledger), ("raw panel", raw), ("OOF stack", stack_frame)):
        if ID not in part or part[ID].duplicated().any(): raise ValueError(f"{name} must contain unique candidate_id")
    if bool(stack_frame.get("residual_is_oof", pd.Series(True, index=stack_frame.index)).eq(False).any()):
        raise ValueError("stack contains non-OOF rows")
    raw_columns = [ID, TIME, SIDE, ALPHA, *[name for name in GROSS_CANDIDATES + NET_CANDIDATES + LABEL_END_CANDIDATES if name in raw]]
    raw_columns = list(dict.fromkeys(raw_columns + _raw_feature_names(feature_contract, raw)))
    raw = raw.loc[:, raw_columns]
    common = set(ledger[ID].astype(str)).intersection(raw[ID].astype(str)).intersection(stack_frame[ID].astype(str))
    if len(common) != len(ledger):
        raise ValueError(f"three-way join is not exact: ledger rows={len(ledger)}, common={len(common)}")
    ledger = ledger.loc[ledger[ID].astype(str).isin(common)].copy()
    frame = ledger.merge(raw, on=ID, how="inner", validate="one_to_one", suffixes=("", "_raw"))
    reference = stack_frame.loc[stack_frame[ID].astype(str).isin(common), [ID, "score_base_alpha", "score_base_expected_ev", "score_residual_expected_ev"]].copy()
    # Stage 0 already carries some OOF columns.  Verify any overlap against
    # the third input, then merge only fields absent from the substrate.
    score_names = ["score_base_alpha", "score_base_expected_ev", "score_residual_expected_ev"]
    overlap = [name for name in score_names if name in frame]
    if overlap:
        joined_reference = frame[[ID, *overlap]].merge(reference[[ID, *overlap]], on=ID, suffixes=("_ledger", "_stack"), validate="one_to_one")
        for name in overlap:
            if not np.allclose(joined_reference[f"{name}_ledger"].to_numpy(float), joined_reference[f"{name}_stack"].to_numpy(float), equal_nan=True):
                raise ValueError(f"Stage-0 and OOF stack {name} values differ on exact candidate IDs")
    missing_scores = [name for name in score_names if name not in frame]
    if missing_scores:
        frame = frame.merge(reference[[ID, *missing_scores]], on=ID, how="inner", validate="one_to_one")
    # Prefer the substrate's separately-reconciled economic fields when present.
    alpha_col = ALPHA if ALPHA in frame else _one(frame, (ALPHA + "_raw",), "alpha target")
    gross_col = _first(frame, GROSS_CANDIDATES, "gross target")
    net_col = _first(frame, NET_CANDIDATES, "net target")
    label_col = _first(frame, LABEL_END_CANDIDATES, "label availability")
    time_col = TIME if TIME in frame else _one(frame, (TIME + "_raw",), "decision time")
    side_col = SIDE if SIDE in frame else _one(frame, (SIDE + "_raw",), "side")
    if time_col != TIME: frame[TIME] = frame[time_col]
    if side_col != SIDE: frame[SIDE] = frame[side_col]
    frame[TIME] = _utc(frame[TIME]); frame[label_col] = _utc(frame[label_col]); frame = frame.sort_values([TIME, ID], kind="stable").reset_index(drop=True)
    if not frame[SIDE].isin(["long", "short"]).all(): raise ValueError("canonical long/short sides required")
    raw_features = _raw_feature_names(feature_contract, frame)
    features = _load_causal_allowlist(causal_allowlist, raw_features)
    provenance = {"joined_rows": len(frame), "ordered_candidate_sha256": _stable_ids(frame[ID]), "alpha_target": alpha_col, "gross_target": gross_col, "net_target": net_col, "label_available": label_col, "raw_feature_count": len(raw_features), "approved_causal_feature_count": len(features), "causal_allowlist": str(causal_allowlist) if causal_allowlist else "sealed_raw_feature_contract_default", "stack_reference_oof_only": True}
    return frame, features, provenance


def _future_oracle_features(frame: pd.DataFrame) -> list[str]:
    """Pinned hindsight-only Stage-0 fields for M7 diagnostic headroom.

    These values resolve after entry and are never allowed into M0--M6 or any
    deployable artifact.  Their only purpose is to quantify remaining
    non-causal information value on the exact same folds and rows.
    """
    candidates = (
        "postcost_h0_event", "postcost_h0_favorable_minute", "postcost_h0_adverse_minute", "postcost_h0_resolved_minute",
        "postcost_h25_event", "postcost_h25_favorable_minute", "postcost_h25_adverse_minute", "postcost_h25_resolved_minute",
        "postcost_h0_retained_net", "postcost_h0_giveback_after_clear", "exit_hour",
    )
    selected = [name for name in candidates if name in frame and pd.api.types.is_numeric_dtype(frame[name])]
    if len(selected) < 4:
        raise ValueError("Stage-0 lacks enough numeric future/event fields for M7 future-feature oracle")
    return selected


def run(*, substrate: Path = DEFAULT_SUBSTRATE, raw_panel: Path = DEFAULT_RAW_PANEL, feature_contract: Path = DEFAULT_FEATURE_CONTRACT, stack: Path = DEFAULT_STACK, causal_allowlist: Path | None = None, output: Path = DEFAULT_OUTPUT, seeds: tuple[int, ...] = SEEDS, families: tuple[str, ...] = ("prior", "ridge", "gam_additive_boosted_stumps", "shallow_lgbm", "shallow_catboost", "production_like_lgbm", "causal_capacity_oracle", "future_feature_oracle"), minimum_rows: int = 2_000, max_features: int | None = None, max_training_rows: int | None = None) -> dict[str, Any]:
    if output.exists(): raise FileExistsError(output)
    _progress("loading and identity-validating Stage-0/raw/stack inputs")
    frame, features, provenance = _read_join(substrate, raw_panel, feature_contract, stack, causal_allowlist)
    if max_features is not None: features = features[:max_features]
    oracle_features = _future_oracle_features(frame) if "future_feature_oracle" in families else []
    provenance["main_training_row_cap"] = max_training_rows
    alpha_col, gross_col, net_col, label_col = provenance["alpha_target"], provenance["gross_target"], provenance["net_target"], provenance["label_available"]
    folds = fixed_folds(); all_predictions: list[pd.DataFrame] = []; states: list[dict[str, Any]] = []
    expected_cache_windows = len(pd.date_range(_utc(frame[TIME]).min() + pd.DateOffset(months=6), _utc(frame[TIME]).max(), freq="QS"))
    fit_units = sum((1 if family in DETERMINISTIC_FAMILIES else len(seeds)) * (len(folds) * 2 * 2 + expected_cache_windows * 2) for family in families)
    _progress(f"fit plan: about {fit_units} base/residual/prequential fit units; deterministic families use one seed, trees use {len(seeds)}")
    start_monotonic = time.monotonic()
    completed_family_seeds = 0
    for family in families:
        if family == "shallow_catboost" and CatBoostRegressor is None: continue
        family_features = oracle_features if family == "future_feature_oracle" else features
        family_seeds = seeds[:1] if family in DETERMINISTIC_FAMILIES else seeds
        for seed in family_seeds:
            _progress(f"precomputing cached prequential base OOF: {family}, seed={seed}, features={'future_oracle' if family == 'future_feature_oracle' else 'raw_causal'}")
            cached_inner_by_side: dict[str, pd.DataFrame] = {}
            cache_states: list[dict[str, Any]] = []
            for side in ("long", "short"):
                cached, cache_lineage = _cached_prequential_base_oof(frame.loc[frame[SIDE].eq(side)].copy(), family_features, alpha_col, label_col, family, int(seed), minimum_rows, max_training_rows=max_training_rows)
                cached_inner_by_side[side] = cached; cache_states.extend(cache_lineage)
            _progress(f"main ladder {family}, seed={seed}, folds={len(folds)}, cap={max_training_rows or 'full'}")
            predicted, family_states = _run_family(frame, folds, family=family, seed=int(seed), features=family_features, alpha_col=alpha_col, gross_col=gross_col, net_col=net_col, label_available_col=label_col, minimum_rows=minimum_rows, max_training_rows=max_training_rows, cached_inner_by_side=cached_inner_by_side)
            elapsed = time.monotonic() - start_monotonic; completed_family_seeds += 1
            _progress(f"completed {family}, seed={seed}: {len(predicted):,} scored rows; elapsed={elapsed:.0f}s, completed family-seeds={completed_family_seeds}")
            all_predictions.append(predicted); states.extend(cache_states); states.extend(family_states)
    predictions = pd.concat(all_predictions, ignore_index=True)
    _progress("scoring frozen OOF reference and layer-specific metrics")
    reference = _reference_predictions(frame, alpha_col, gross_col, net_col)
    if not reference.empty: predictions = pd.concat([predictions, reference], ignore_index=True)
    metric_frame = _metrics_table(predictions, alpha_col, gross_col, net_col)
    _progress("running bounded sample/history/feature/capacity curves")
    curves = _learning_curves(frame, features, alpha_col, gross_col, net_col, label_col, int(seeds[0]), minimum_rows)
    _progress("running semi-synthetic causal-signal recovery")
    synthetic = _synthetic_recovery(frame, features, alpha_col, label_col, int(seeds[0]), minimum_rows)
    association = _concordance(metric_frame)
    association.insert(0, "record_type", "base_to_later_residual_association")
    arm_outcomes = _stage4_arm_outcomes(predictions.loc[predictions.model_family.ne("frozen_oof_reference")].copy(), net_col, gross_col)
    concordance = pd.concat([association, arm_outcomes], ignore_index=True, sort=False)
    gaps = _gap_and_seed_summary(metric_frame)
    # Absolute gaps are primary.  Ratios are reported only if their denominator
    # is economically separated from null, avoiding misleading percentages.
    summary = metric_frame.groupby(["model_family", "component", "split", "evaluation_scope"], as_index=False).mean(numeric_only=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        predictions.to_parquet(stage / "base_residual_oof_predictions.parquet", index=False, compression="zstd")
        metric_frame.to_parquet(stage / "model_learning_efficiency.parquet", index=False, compression="zstd")
        curves.to_parquet(stage / "learning_capacity_feature_history_curves.parquet", index=False, compression="zstd")
        synthetic.to_parquet(stage / "semi_synthetic_recovery.parquet", index=False, compression="zstd")
        concordance.to_parquet(stage / "metric_concordance.parquet", index=False, compression="zstd")
        gaps.to_parquet(stage / "model_learning_gaps_and_seed_dispersion.parquet", index=False, compression="zstd")
        summary.to_parquet(stage / "model_learning_summary.parquet", index=False, compression="zstd")
        _dump(stage / "fold_model_lineage.json", states)
        report = ["# Stage 3–4 base/residual learning diagnostic", "", "Research-only. No EV meta, timing/action, auxiliary/CatBoost head, ranking, sizing, portfolio, threshold, or policy logic is invoked.", "", "## Layer contract", "", f"- Base target: `{alpha_col}` (directional/soft alpha).", f"- Residual target: `{gross_col} - map_train_oof(base_alpha)` in bps.", "- The residual learner consumes only stopped-gradient chronological inner-OOF base alpha on training rows; all test base scores are outer-OOS.", f"- Raw causal feature contract: {len(features)} fields; existing stack is reference-only and never a newly-trained input.", f"- Main ladder training row cap: `{max_training_rows or 'none (full eligible fold)'}`. Curves and synthetic recovery use a separate, recorded 18,000-row chronological-support cap.", "", "## Ladder boundary", "", "- M0–M6 are represented by prior, ridge, additive boosted stumps, shallow LGBM, generic shallow CatBoost baseline, production-like LGBM, and high-capacity causal LGBM.", "- M7 is a strictly diagnostic hindsight oracle using only sealed Stage-0 post-entry event fields. It shares rows/folds/targets with M0–M6 but is non-causal and cannot be promoted or used as a pipeline input.", "", "## Interpretation", "", "Base directional metrics and residual economic metrics use disjoint namespaces. Metric-concordance links development base quality to later residual economics. Treat future-oracle, action, policy, and head conclusions as out of scope for this artifact."]
        (stage / "STAGE_3_4_BASE_RESIDUAL_DIAGNOSTIC.md").write_text("\n".join(report) + "\n")
        invariants = {"m0_m6_raw_causal_features_only": True, "m7_isolated_hindsight_diagnostic_only": "future_feature_oracle" in families, "stack_is_reference_only": True, "residual_training_base_is_inner_oof": True, "outer_predictions_are_chronological_oof_or_later_oos": True, "no_auxiliary_or_policy_layers": True}
        _dump(stage / "correctness_test_report.json", {"schema": "root_cause_base_residual_learning_correctness_v1", "invariants": invariants, "focused_pytest": "tests/test_run_root_cause_base_residual_learning.py"})
        outputs = {item.name: _sha(item) for item in stage.iterdir()}
        input_paths = [substrate, raw_panel, feature_contract, stack, *([causal_allowlist] if causal_allowlist else [])]
        manifest = {"schema": "root_cause_base_residual_learning_v1", "status": "COMPLETED_RESEARCH_ONLY_NO_PROMOTION", "scope": "base_directional_alpha_and_stopped_gradient_gross_residual_only", "runner": {"path": str(Path(__file__).relative_to(ROOT)), "sha256": _sha(Path(__file__))}, "provenance": provenance, "folds": [{"name": fold.name, "start": str(fold.start), "end": str(fold.end), "split": fold.split, "purge_hours": 12, "embargo_hours": 12} for fold in folds], "models": list(families), "seeds": list(seeds), "features": features, "future_oracle_features": oracle_features, "ladder_disposition": {"M0": "prior", "M1": "ridge", "M2": "gam_additive_boosted_stumps", "M3": "shallow_lgbm", "M4": "shallow_catboost_generic_baseline_not_head", "M5": "production_like_lgbm", "M6": "causal_capacity_oracle_raw_causal_features", "M7": "future_feature_oracle_hindsight_only"}, "invariants": invariants, "inputs": {str(path): _sha(path) for path in input_paths}, "outputs_sha256": outputs}
        _dump(stage / "run_manifest.json", manifest)
        os.replace(stage, output)
        return manifest
    except Exception:
        shutil.rmtree(stage, ignore_errors=True); raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--substrate", type=Path, default=DEFAULT_SUBSTRATE)
    parser.add_argument("--raw-panel", type=Path, default=DEFAULT_RAW_PANEL)
    parser.add_argument("--feature-contract", type=Path, default=DEFAULT_FEATURE_CONTRACT)
    parser.add_argument("--stack", type=Path, default=DEFAULT_STACK)
    parser.add_argument("--causal-allowlist", type=Path, help="Stage-2 approved raw-causal feature subset JSON")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--minimum-rows", type=int, default=2_000)
    parser.add_argument("--max-features", type=int)
    parser.add_argument("--max-training-rows", type=int, help="Optional deterministic per-side full-support cap for a bounded main ladder run")
    parser.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    parser.add_argument("--families", default="prior,ridge,gam_additive_boosted_stumps,shallow_lgbm,shallow_catboost,production_like_lgbm,causal_capacity_oracle,future_feature_oracle")
    args = parser.parse_args()
    result = run(substrate=args.substrate, raw_panel=args.raw_panel, feature_contract=args.feature_contract, stack=args.stack, causal_allowlist=args.causal_allowlist, output=args.output, seeds=tuple(int(item) for item in args.seeds.split(",") if item), families=tuple(item for item in args.families.split(",") if item), minimum_rows=args.minimum_rows, max_features=args.max_features, max_training_rows=args.max_training_rows)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
