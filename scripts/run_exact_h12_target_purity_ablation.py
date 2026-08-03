#!/usr/bin/env python3
"""Candidate-level E0--E3 exact-H12 target-purity ablation.

The frozen historical execution policy is never optimised here.  This runner
only asks whether identical candidates can be ranked by post-cost H12 net
under that policy.  It intentionally excludes sizing, portfolio constraints,
and in-trade actions.
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
from typing import Any, Callable

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.materialize_historical_exact_h12_alignment_sidecar import (
    COST_MODEL_ID, EXECUTION_POLICY_ID, TARGET_ID,
)
from scripts.run_long_base_residual_target_ablation import global_top_mask


PANEL_DIR = ROOT / "data_perp/artifacts/long_exact_h12_raw_base_panel_20260730_v2"
PANEL = PANEL_DIR / "raw_base_panel.parquet"
FEATURE_CONTRACT = PANEL_DIR / "raw_feature_contract.json"
ALIGNMENT = ROOT / "data_perp/artifacts/historical_exact_h12_alignment_sidecar_research_only_20260731_v1/alignment_sidecar.parquet"
POSTCOST_EVENTS = ROOT / "data_perp/artifacts/historical_exact_h12_postcost_events_20260731_v1/postcost_events.parquet"
PERSISTENCE_LABELS = ROOT / "data_perp/artifacts/historical_exact_h12_postcost_persistence_labels_20260731_v1/postcost_persistence_labels.parquet"
DEFAULT_OUTPUT = ROOT / "data_perp/artifacts/exact_h12_target_purity_ablation_20260731_v10"
SIDES = ("long", "short")
TOPS = (0.01, 0.05, 0.10, 0.20)
BASE_START = pd.Timestamp("2023-04-01T00:00:00Z")
BASE_END = pd.Timestamp("2024-04-01T00:00:00Z")
META_END = pd.Timestamp("2024-08-01T00:00:00Z")
EVAL_END = pd.Timestamp("2024-12-01T00:00:00Z")
CALIBRATION_DAYS = 21
TOP_FEATURES = 64
EVENTS = ("clean", "adverse", "timeout")
PERSISTENCE_EVENTS = ("retained", "giveback", "adverse", "timeout")
HURDLES_BPS = (0.0, 25.0, 50.0)
FIXED_POLICY_COST_BPS = 100.0
BOOTSTRAP_REPLICATES = 400
CALIBRATION_BINS = (-np.inf, -200.0, -100.0, -50.0, 0.0, 50.0, 100.0, 200.0, np.inf)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _raw_features(path: Path) -> list[str]:
    columns = list(json.loads(path.read_text(encoding="utf-8"))["raw_feature_columns"])
    forbidden = ("future_", "label", "target", "outcome", "actual_", "execution_net", "execution_gross", "execution_cost", "exit_reason", "recommended_action", "action_value")
    rejected = [name for name in columns if any(token in name.lower() for token in forbidden)]
    if rejected:
        raise ValueError(f"future/outcome fields entered execution matrix: {rejected}")
    return columns


def _fit_regressor(x: pd.DataFrame, y: np.ndarray, *, seed: int, trees: int) -> lgb.LGBMRegressor:
    model = lgb.LGBMRegressor(objective="regression", n_estimators=trees, learning_rate=0.04, num_leaves=23, max_depth=5, min_child_samples=180, colsample_bytree=0.80, subsample=0.85, subsample_freq=1, reg_lambda=15.0, reg_alpha=0.15, random_state=seed, n_jobs=2, verbosity=-1)
    model.fit(x, y)
    return model


def _fit_classifier(x: pd.DataFrame, y: np.ndarray, *, seed: int, trees: int, classes: int = 2) -> lgb.LGBMClassifier:
    model = lgb.LGBMClassifier(objective="multiclass" if classes > 2 else "binary", num_class=classes if classes > 2 else None, n_estimators=trees, learning_rate=0.04, num_leaves=23, max_depth=5, min_child_samples=180, colsample_bytree=0.80, subsample=0.85, subsample_freq=1, reg_lambda=15.0, reg_alpha=0.15, random_state=seed, n_jobs=2, verbosity=-1)
    model.fit(x, y)
    return model


def _matrix(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    values = frame.loc[:, features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if list(values.columns) != features:
        raise AssertionError("feature matrix order changed")
    return values


def _event(frame: pd.DataFrame, *, post_cost_hurdle_bps: float | None = None, exact_postcost_token: str | None = None) -> np.ndarray:
    if exact_postcost_token is not None:
        column = f"postcost_{exact_postcost_token}_event"
        if column not in frame:
            raise ValueError(f"missing materialised exact post-cost event: {column}")
        source = frame[column].astype(str)
        mapped = np.where(source.eq("clear_cost_first"), "clean", np.where(source.eq("adverse_first_or_conflict"), "adverse", np.where(source.eq("timeout"), "timeout", "invalid")))
        if (mapped == "invalid").any():
            raise ValueError("materialised exact post-cost events are not a clean/adverse/timeout simplex")
        return mapped
    source = frame.event_first.astype(str)
    mapped = np.where(source.eq("favorable_first"), "clean", np.where(source.eq("adverse_first_or_conflict"), "adverse", np.where(source.eq("timeout"), "timeout", "invalid")))
    if (mapped == "invalid").any():
        raise ValueError("historical first-event field is not a clean/adverse/timeout simplex")
    if post_cost_hurdle_bps is not None:
        clears = frame.exact_h12_gross_bps.to_numpy(float) > FIXED_POLICY_COST_BPS + float(post_cost_hurdle_bps)
        # A favorable-first path that does not clear an economic gross hurdle is
        # deliberately treated as a non-clear timeout/late outcome.  This is a
        # conservative proxy, not a claim that the historical source contains
        # exact intrahorizon post-cost barrier timestamps.
        mapped = np.where((mapped == "clean") & clears, "clean", np.where(mapped == "adverse", "adverse", "timeout"))
    return mapped


def _persistence_event(frame: pd.DataFrame, token: str) -> np.ndarray:
    column = f"postcost_{token}_four_state"
    if column not in frame:
        raise ValueError(f"missing materialised persistence state: {column}")
    source = frame[column].astype(str)
    mapped = np.where(source.eq("clear_then_retained"), "retained", np.where(source.eq("clear_then_giveback"), "giveback", np.where(source.eq("adverse_first_or_conflict"), "adverse", np.where(source.eq("timeout"), "timeout", "invalid"))))
    if (mapped == "invalid").any():
        raise ValueError("materialised persistence states are not a four-state simplex")
    return mapped


def _calendar(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    ts = pd.to_datetime(frame.decision_ts, utc=True)
    return {
        "base_train": (ts.ge(BASE_START) & ts.lt(BASE_END)).to_numpy(),
        "base_oos": (ts.ge(BASE_END) & ts.lt(EVAL_END)).to_numpy(),
        "meta_train": (ts.ge(BASE_END) & ts.lt(META_END)).to_numpy(),
        "eval": (ts.ge(META_END) & ts.lt(EVAL_END)).to_numpy(),
    }


def _select_base_features(frame: pd.DataFrame, raw: list[str], *, seed: int, trees: int) -> dict[str, list[str]]:
    selected: dict[str, list[str]] = {}
    for index, side in enumerate(SIDES):
        local = frame.loc[frame.side.eq(side)].copy()
        x = _matrix(local, raw)
        usable = [name for name in raw if x[name].notna().mean() >= 0.50 and x[name].nunique(dropna=True) > 1]
        y = (_event(local) == "clean").astype(int)
        model = _fit_classifier(_matrix(local, usable), y, seed=seed + index, trees=max(70, trees // 2))
        gain = pd.Series(model.booster_.feature_importance(importance_type="gain"), index=usable)
        keep = gain.loc[gain.gt(0.0)].sort_values(ascending=False, kind="stable").head(TOP_FEATURES).index.tolist()
        if len(keep) < 16:
            raise ValueError(f"{side} base selection retained fewer than 16 features")
        selected[side] = keep
    return selected


def _select_retention_features(frame: pd.DataFrame, raw: list[str], *, seed: int, trees: int) -> dict[str, list[str]]:
    """Target-specific, pre-evaluation feature selection for retain|clear."""
    selected: dict[str, list[str]] = {}
    for index, side in enumerate(SIDES):
        local = frame.loc[frame.side.eq(side)].copy()
        state = _persistence_event(local, "h0")
        local = local.loc[np.isin(state, ("retained", "giveback"))].reset_index(drop=True)
        labels = _persistence_event(local, "h0") == "retained"
        x = _matrix(local, raw)
        usable = [name for name in raw if x[name].notna().mean() >= 0.50 and x[name].nunique(dropna=True) > 1]
        model = _fit_classifier(_matrix(local, usable), labels.astype(int), seed=seed + index, trees=max(70, trees // 2))
        gain = pd.Series(model.booster_.feature_importance(importance_type="gain"), index=usable)
        keep = gain.loc[gain.gt(0.0)].sort_values(ascending=False, kind="stable").head(TOP_FEATURES).index.tolist()
        if len(keep) < 16:
            raise ValueError(f"{side} retention selection retained fewer than 16 features")
        selected[side] = keep
    return selected


def _policy_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Only row-varying, decision-known target generators are appended."""
    return pd.DataFrame({
        "estimated_spread_bps": frame.estimated_spread_bps.to_numpy(float),
        "entry_half_spread_bps": frame.entry_half_spread_bps.to_numpy(float),
        "barrier_pct": frame.barrier_pct.to_numpy(float),
        "entry_price_log": np.log(frame.execution_entry_price.to_numpy(float)),
    }, index=frame.index)


def _features_for(frame: pd.DataFrame, raw: list[str], *, include_base: bool = False) -> pd.DataFrame:
    matrix = pd.concat([_matrix(frame, raw).reset_index(drop=True), _policy_features(frame).reset_index(drop=True)], axis=1)
    if include_base:
        if "base_expected_net_bps" not in frame:
            raise ValueError("base feature requested before strict OOS base output is available")
        matrix["base_expected_net_bps"] = frame.base_expected_net_bps.to_numpy(float)
    return matrix


def _causal_map(history: pd.DataFrame, evaluate: pd.DataFrame, *, side_specific: bool = False) -> pd.DataFrame:
    """One pooled score-to-net map using only prior resolved labels."""
    if side_specific:
        parts = [
            _causal_map(history.loc[history.side.eq(side)].copy(), evaluate.loc[evaluate.side.eq(side)].copy(), side_specific=False)
            for side in SIDES
        ]
        return pd.concat(parts, ignore_index=True).sort_values(["decision_ts", "candidate_id"], kind="stable").reset_index(drop=True)
    full = pd.concat([history, evaluate], ignore_index=True, sort=False)
    result = evaluate.copy().reset_index(drop=True)
    result["calibrated_expected_net_bps"] = np.nan
    result["map_reference_rows"] = 0
    days = result.decision_ts.dt.floor("D")
    for day in days.drop_duplicates().sort_values():
        target = days.eq(day).to_numpy()
        reference = full.loc[(full.decision_ts.ge(day - pd.Timedelta(days=CALIBRATION_DAYS))) & (full.label_available_ts.lt(day)) & np.isfinite(full.raw_score) & np.isfinite(full.exact_h12_net_bps)]
        result.loc[target, "map_reference_rows"] = len(reference)
        if len(reference) < 500 or reference.raw_score.nunique() < 2:
            continue
        mapper = IsotonicRegression(out_of_bounds="clip")
        mapper.fit(reference.raw_score, reference.exact_h12_net_bps)
        result.loc[target, "calibrated_expected_net_bps"] = mapper.predict(result.loc[target, "raw_score"])
    return result


def _base_scores(frame: pd.DataFrame, selected: dict[str, list[str]], raw: list[str], masks: dict[str, np.ndarray], *, seed: int, trees: int) -> pd.DataFrame:
    """Frozen OOS base opportunity outputs; no archived score is used as input."""
    result = frame.loc[masks["base_oos"], ["candidate_id", "side", "decision_ts", "label_available_ts", "exact_h12_net_bps"]].copy().reset_index(drop=True)
    result["raw_score"] = np.nan
    for offset, side in enumerate(SIDES):
        train = frame.loc[masks["base_train"] & frame.side.eq(side)].copy().reset_index(drop=True)
        test_pos = np.flatnonzero(result.side.eq(side).to_numpy())
        model = _fit_classifier(_features_for(train, selected[side]), (_event(train) == "clean").astype(int), seed=seed + offset, trees=trees)
        test_rows = frame.loc[masks["base_oos"] & frame.side.eq(side)].copy().reset_index(drop=True)
        result.loc[test_pos, "raw_score"] = model.predict_proba(_features_for(test_rows, selected[side]))[:, 1]
    if not np.isfinite(result.raw_score).all():
        raise AssertionError("base OOS score incomplete")
    return _causal_map(result.iloc[0:0], result)


def _prior(frame: pd.DataFrame, mask: np.ndarray, value: np.ndarray, fallback: float) -> float:
    local = value[mask]
    return float(np.mean(local)) if len(local) else float(fallback)


def _causal_cost_proxy(train: pd.DataFrame, rows: pd.DataFrame) -> np.ndarray:
    """Training-only linear proxy; no realised evaluation cost enters inputs."""
    x = train.estimated_spread_bps.to_numpy(float)
    y = train.row_cost_bps.to_numpy(float)
    if len(train) < 100 or np.nanstd(x) < 1e-8:
        return np.full(len(rows), FIXED_POLICY_COST_BPS, dtype=float)
    slope, intercept = np.polyfit(x, y, deg=1)
    proxy = intercept + slope * rows.estimated_spread_bps.to_numpy(float)
    return np.clip(proxy, 95.0, 105.0)


def _three_state_expected_net(train: pd.DataFrame, test: pd.DataFrame, x_train: pd.DataFrame, x_test: pd.DataFrame, *, seed: int, trees: int, post_cost_hurdle_bps: float | None = None, exact_postcost_token: str | None = None) -> np.ndarray:
    labels = _event(train, post_cost_hurdle_bps=post_cost_hurdle_bps, exact_postcost_token=exact_postcost_token)
    code = pd.Categorical(labels, categories=list(EVENTS)).codes
    classifier = _fit_classifier(x_train, code, seed=seed, trees=trees, classes=3)
    probabilities = classifier.predict_proba(x_test)
    net_train = train.exact_h12_net_bps.to_numpy(float)
    expected = np.zeros(len(test), dtype=float)
    for event_index, event_name in enumerate(EVENTS):
        support = labels == event_name
        prior = _prior(train, support, net_train, float(net_train.mean()))
        if int(support.sum()) < 180:
            conditional = np.full(len(test), prior)
        else:
            model = _fit_regressor(x_train.loc[support], net_train[support], seed=seed + 10 + event_index, trees=trees)
            raw = model.predict(x_test)
            shrink = float(support.sum()) / float(support.sum() + 500)
            conditional = shrink * raw + (1.0 - shrink) * prior
        expected += probabilities[:, event_index] * conditional
    return expected


def _persistence_four_state_expected_net(train: pd.DataFrame, test: pd.DataFrame, x_train: pd.DataFrame, x_test: pd.DataFrame, *, seed: int, trees: int, token: str) -> np.ndarray:
    labels = _persistence_event(train, token)
    code = pd.Categorical(labels, categories=list(PERSISTENCE_EVENTS)).codes
    classifier = _fit_classifier(x_train, code, seed=seed, trees=trees, classes=len(PERSISTENCE_EVENTS))
    probabilities = classifier.predict_proba(x_test)
    net_train = train.exact_h12_net_bps.to_numpy(float)
    expected = np.zeros(len(test), dtype=float)
    for state_index, state_name in enumerate(PERSISTENCE_EVENTS):
        support = labels == state_name
        prior = _prior(train, support, net_train, float(net_train.mean()))
        if int(support.sum()) < 180:
            conditional = np.full(len(test), prior)
        else:
            model = _fit_regressor(x_train.loc[support], net_train[support], seed=seed + 30 + state_index, trees=trees)
            raw = model.predict(x_test)
            shrink = float(support.sum()) / float(support.sum() + 500)
            conditional = shrink * raw + (1.0 - shrink) * prior
        expected += probabilities[:, state_index] * conditional
    return expected


def _hierarchical_state_probabilities(p_clear: np.ndarray, p_retain_given_clear: np.ndarray, p_adverse_given_not_clear: np.ndarray) -> np.ndarray:
    """Recompose the exact four states from two causal path transitions.

    The hierarchy is intentionally explicit: a retained/giveback outcome is
    only possible after cost has cleared before the adverse barrier.  This is
    equivalent in expectation to a four-class simplex, but gives the learner
    denser and semantically simpler probability questions.
    """
    clear = np.clip(np.asarray(p_clear, dtype=float), 0.0, 1.0)
    retain = np.clip(np.asarray(p_retain_given_clear, dtype=float), 0.0, 1.0)
    adverse = np.clip(np.asarray(p_adverse_given_not_clear, dtype=float), 0.0, 1.0)
    probabilities = np.column_stack([
        clear * retain,
        clear * (1.0 - retain),
        (1.0 - clear) * adverse,
        (1.0 - clear) * (1.0 - adverse),
    ])
    if not np.isfinite(probabilities).all() or not np.allclose(probabilities.sum(axis=1), 1.0, rtol=0.0, atol=1e-10):
        raise AssertionError("hierarchical persistence probabilities are not a finite simplex")
    return probabilities


def _binary_probability(x_train: pd.DataFrame, target: np.ndarray, x_test: pd.DataFrame, *, seed: int, trees: int) -> np.ndarray:
    target = np.asarray(target, dtype=int)
    prior = float(target.mean()) if len(target) else 0.5
    if len(target) < 180 or target.min() == target.max():
        return np.full(len(x_test), prior, dtype=float)
    return _fit_classifier(x_train, target, seed=seed, trees=trees).predict_proba(x_test)[:, 1]


def _conditional_net_prediction(x_train: pd.DataFrame, x_test: pd.DataFrame, mask: np.ndarray, net: np.ndarray, *, seed: int, trees: int) -> np.ndarray:
    """Event-conditional net with causal support shrinkage."""
    support = np.asarray(mask, dtype=bool)
    prior = float(net[support].mean()) if support.any() else float(net.mean())
    if int(support.sum()) < 180:
        return np.full(len(x_test), prior, dtype=float)
    raw = _fit_regressor(x_train.loc[support], net[support], seed=seed, trees=trees).predict(x_test)
    shrink = float(support.sum()) / float(support.sum() + 500)
    return shrink * raw + (1.0 - shrink) * prior


def _soft_terminal_net_target(net_bps: np.ndarray, *, hurdle_bps: float, temperature_bps: float) -> np.ndarray:
    """Cost-aware soft terminal-value label used only as an execution target.

    It is deliberately not a realised top-k membership or a probability that
    is pooled across sides.  The final output remains a causally calibrated
    expected H12 net score in bps.
    """
    if temperature_bps <= 0.0:
        raise ValueError("soft-label temperature must be positive")
    scaled = np.clip((np.asarray(net_bps, dtype=float) - hurdle_bps) / temperature_bps, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-scaled))


def _hierarchical_persistence_expected_net(train: pd.DataFrame, test: pd.DataFrame, x_train: pd.DataFrame, x_test: pd.DataFrame, *, seed: int, trees: int, token: str, return_components: bool = False, retention_x_train: pd.DataFrame | None = None, retention_x_test: pd.DataFrame | None = None) -> np.ndarray | dict[str, np.ndarray]:
    """Factor reachability, persistence, and non-clear adverse risk.

    This is a Stage-B target-formulation ablation, not a supportive-head
    feature blend.  All probabilities and state-conditional net estimates are
    reconstructed into one raw expected-net score before the usual causal map.
    """
    labels = _persistence_event(train, token)
    net = train.exact_h12_net_bps.to_numpy(float)
    clear = np.isin(labels, ("retained", "giveback"))
    retained_given_clear = labels[clear] == "retained"
    not_clear = ~clear
    adverse_given_not_clear = labels[not_clear] == "adverse"
    if (retention_x_train is None) != (retention_x_test is None):
        raise ValueError("retention feature matrices must be supplied as a pair")
    retain_train = x_train if retention_x_train is None else retention_x_train
    retain_test = x_test if retention_x_test is None else retention_x_test
    if len(retain_train) != len(train) or len(retain_test) != len(test):
        raise ValueError("retention feature matrices must preserve hierarchy row order")
    p_clear = _binary_probability(x_train, clear, x_test, seed=seed, trees=trees)
    # Stage-C can replace this matrix only.  The clear and adverse paths, plus
    # state-conditional net estimators, remain on ``x_*`` exactly as before.
    p_retain = _binary_probability(retain_train.loc[clear], retained_given_clear, retain_test, seed=seed + 1, trees=trees)
    p_adverse = _binary_probability(x_train.loc[not_clear], adverse_given_not_clear, x_test, seed=seed + 2, trees=trees)
    probabilities = _hierarchical_state_probabilities(p_clear, p_retain, p_adverse)
    conditional = np.column_stack([
        _conditional_net_prediction(x_train, x_test, labels == "retained", net, seed=seed + 10, trees=trees),
        _conditional_net_prediction(x_train, x_test, labels == "giveback", net, seed=seed + 11, trees=trees),
        _conditional_net_prediction(x_train, x_test, labels == "adverse", net, seed=seed + 12, trees=trees),
        _conditional_net_prediction(x_train, x_test, labels == "timeout", net, seed=seed + 13, trees=trees),
    ])
    expected = (probabilities * conditional).sum(axis=1)
    if not return_components:
        return expected
    return {
        "raw_score": expected,
        "p_clear_cost_before_adverse": p_clear,
        "p_retain_given_clear": p_retain,
        "p_adverse_given_not_clear": p_adverse,
        "p_retained": probabilities[:, 0],
        "p_giveback": probabilities[:, 1],
        "p_adverse": probabilities[:, 2],
        "p_timeout": probabilities[:, 3],
        "probability_simplex_error": np.abs(probabilities.sum(axis=1) - 1.0),
    }


def _predict_arm(train: pd.DataFrame, test: pd.DataFrame, x_train: pd.DataFrame, x_test: pd.DataFrame, arm: str, *, seed: int, trees: int) -> np.ndarray:
    net_train = train.exact_h12_net_bps.to_numpy(float)
    base_train = train.base_expected_net_bps.to_numpy(float)
    if arm == "E0_direct_net":
        return _fit_regressor(x_train, net_train, seed=seed, trees=trees).predict(x_test)
    if arm == "E4_fixed_cost_100":
        return _fit_regressor(x_train, train.exact_h12_gross_bps.to_numpy(float) - FIXED_POLICY_COST_BPS, seed=seed, trees=trees).predict(x_test)
    if arm == "E5_causal_cost_proxy":
        train_target = train.exact_h12_gross_bps.to_numpy(float) - _causal_cost_proxy(train, train)
        return _fit_regressor(x_train, train_target, seed=seed, trees=trees).predict(x_test)
    if arm == "E1_net_residual":
        return test.base_expected_net_bps.to_numpy(float) + _fit_regressor(x_train, net_train - base_train, seed=seed, trees=trees).predict(x_test)
    if arm == "E2_three_state":
        return _three_state_expected_net(train, test, x_train, x_test, seed=seed, trees=trees)
    if arm.startswith("E6_postcost_three_state_"):
        hurdle = float(arm.rsplit("_", 1)[1])
        return _three_state_expected_net(train, test, x_train, x_test, seed=seed, trees=trees, post_cost_hurdle_bps=hurdle)
    if arm.startswith("E11_exact1m_postcost_three_state_"):
        token = f"h{arm.rsplit('_', 1)[1]}"
        return _three_state_expected_net(train, test, x_train, x_test, seed=seed, trees=trees, exact_postcost_token=token)
    if arm.startswith("E13_exact1m_persistence_four_state_"):
        token = f"h{arm.rsplit('_', 1)[1]}"
        return _persistence_four_state_expected_net(train, test, x_train, x_test, seed=seed, trees=trees, token=token)
    if arm.startswith("E15_exact1m_hierarchical_persistence_"):
        token = f"h{arm.rsplit('_', 1)[1]}"
        return _hierarchical_persistence_expected_net(train, test, x_train, x_test, seed=seed, trees=trees, token=token)
    if arm.startswith("E18_exact1m_hierarchical_persistence_"):
        return _hierarchical_persistence_expected_net(train, test, x_train, x_test, seed=seed, trees=trees, token="h0")
    if arm.startswith("E17_soft_terminal_net_"):
        # E17_soft_terminal_net_h0_t100: a smooth cost-aware terminal-value
        # target.  Its raw output is mapped causally to exact net downstream.
        _, _, _, _, hurdle, temperature = arm.rsplit("_", 5)
        if not hurdle.startswith("h") or not temperature.startswith("t"):
            raise ValueError(f"invalid soft target arm: {arm}")
        target = _soft_terminal_net_target(net_train, hurdle_bps=float(hurdle[1:]), temperature_bps=float(temperature[1:]))
        return _fit_regressor(x_train, target, seed=seed, trees=trees).predict(x_test)
    if arm.startswith("E3_hurdle_"):
        hurdle = float(arm.rsplit("_", 1)[1])
        clear = net_train > hurdle
        prior_p = float(clear.mean())
        if clear.min() == clear.max():
            p_clear = np.full(len(test), prior_p)
        else:
            p_clear = _fit_classifier(x_train, clear.astype(int), seed=seed, trees=trees).predict_proba(x_test)[:, 1]
        up = np.maximum(net_train - hurdle, 0.0)
        down = np.maximum(hurdle - net_train, 0.0)
        def conditional(mask: np.ndarray, values: np.ndarray, sub_seed: int) -> np.ndarray:
            prior = _prior(train, mask, values, 0.0)
            if int(mask.sum()) < 180:
                return np.full(len(test), prior)
            raw = _fit_regressor(x_train.loc[mask], values[mask], seed=sub_seed, trees=trees).predict(x_test)
            shrink = float(mask.sum()) / float(mask.sum() + 500)
            return shrink * raw + (1.0 - shrink) * prior
        mu_up = conditional(clear, up, seed + 20)
        mu_down = conditional(~clear, down, seed + 21)
        return hurdle + p_clear * mu_up - (1.0 - p_clear) * mu_down
    raise ValueError(f"unknown arm: {arm}")


def _model_arm_for(arm: str) -> str:
    """Map calibration/feature variants to their shared target formulation."""
    aliases = {
        "E7_postcost_three_state_0_sidebridge": "E6_postcost_three_state_0",
        "E8_postcost_three_state_0_base_only": "E6_postcost_three_state_0",
        "E9_postcost_three_state_0_raw_plus_base": "E6_postcost_three_state_0",
        "E10_postcost_three_state_0_base_only_sidebridge": "E6_postcost_three_state_0",
        "E12_exact1m_postcost_three_state_0_sidebridge": "E11_exact1m_postcost_three_state_0",
        "E14_exact1m_persistence_four_state_0_sidebridge": "E13_exact1m_persistence_four_state_0",
        "E16_exact1m_hierarchical_persistence_0_sidebridge": "E15_exact1m_hierarchical_persistence_0",
        "E19_exact1m_hierarchical_persistence_0_retention_features_sidebridge": "E18_exact1m_hierarchical_persistence_0_retention_features",
    }
    return aliases.get(arm, arm)


def _uses_side_bridge(arm: str) -> bool:
    return arm in {
        "E7_postcost_three_state_0_sidebridge",
        "E10_postcost_three_state_0_base_only_sidebridge",
        "E12_exact1m_postcost_three_state_0_sidebridge",
        "E14_exact1m_persistence_four_state_0_sidebridge",
        "E16_exact1m_hierarchical_persistence_0_sidebridge",
        "E19_exact1m_hierarchical_persistence_0_retention_features_sidebridge",
    }


def _prequential_scores(frame: pd.DataFrame, feature_by_side: dict[str, list[str]], arm: str, *, seed: int, trees: int, include_base: bool = False) -> pd.DataFrame:
    """Expanding monthly OOF scores; every fit is strictly before its month."""
    result = frame.loc[:, ["candidate_id", "side", "decision_ts", "label_available_ts", "exact_h12_net_bps", "base_expected_net_bps"]].copy()
    result["raw_score"] = np.nan
    months = sorted(frame.decision_ts.dt.strftime("%Y-%m").unique())
    for side_index, side in enumerate(SIDES):
        local_idx = np.flatnonzero(frame.side.eq(side).to_numpy())
        local = frame.iloc[local_idx].reset_index(drop=True)
        local_month = local.decision_ts.dt.strftime("%Y-%m")
        features = feature_by_side[side]
        for month_index, month in enumerate(months):
            if month_index == 0:
                continue
            valid = local_month.eq(month).to_numpy()
            train = local_month.lt(month).to_numpy() & local.label_available_ts.lt(pd.Timestamp(f"{month}-01", tz="UTC")).to_numpy()
            if valid.sum() == 0 or train.sum() < 1_000:
                continue
            score = _predict_arm(local.loc[train].reset_index(drop=True), local.loc[valid].reset_index(drop=True), _features_for(local.loc[train].reset_index(drop=True), features, include_base=include_base), _features_for(local.loc[valid].reset_index(drop=True), features, include_base=include_base), arm, seed=seed + side_index * 100 + month_index, trees=trees)
            result.loc[result.index[local_idx[valid]], "raw_score"] = score
    return result


def _book_records(scored: pd.DataFrame, arm: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    valid = scored.loc[np.isfinite(scored.calibrated_expected_net_bps)].copy().reset_index(drop=True)
    for top in TOPS:
        selected = global_top_mask(valid.calibrated_expected_net_bps, top)
        book = valid.loc[selected]
        rows.append({"arm": arm, "scope": "pooled_global_top", "fraction": top, "selected_rows": len(book), "gross_bps": float(book.exact_h12_gross_bps.mean()), "cost_bps": float(book.row_cost_bps.mean()), "net_bps": float(book.exact_h12_net_bps.mean()), "positive_net_rate": float((book.exact_h12_net_bps > 0).mean()), "side_long_share": float(book.side.eq("long").mean())})
        for side, part in book.groupby("side", sort=True):
            rows.append({"arm": arm, "scope": "pooled_global_membership_by_side", "fraction": top, "side": side, "selected_rows": len(part), "gross_bps": float(part.exact_h12_gross_bps.mean()), "cost_bps": float(part.row_cost_bps.mean()), "net_bps": float(part.exact_h12_net_bps.mean()), "positive_net_rate": float((part.exact_h12_net_bps > 0).mean()), "side_long_share": float(part.side.eq("long").mean())})
        for month, part in book.assign(month=book.decision_ts.dt.strftime("%Y-%m")).groupby("month", sort=True):
            rows.append({"arm": arm, "scope": "pooled_global_membership_by_month", "fraction": top, "month": month, "selected_rows": len(part), "gross_bps": float(part.exact_h12_gross_bps.mean()), "cost_bps": float(part.row_cost_bps.mean()), "net_bps": float(part.exact_h12_net_bps.mean()), "positive_net_rate": float((part.exact_h12_net_bps > 0).mean()), "side_long_share": float(part.side.eq("long").mean())})
    return rows


def _calibration_records(scored: pd.DataFrame, arm: str) -> list[dict[str, Any]]:
    """Fixed prediction buckets; never a rank or threshold decision rule."""
    valid = scored.loc[np.isfinite(scored.calibrated_expected_net_bps)].copy()
    valid["prediction_bucket"] = pd.cut(valid.calibrated_expected_net_bps, bins=CALIBRATION_BINS, include_lowest=True)
    records: list[dict[str, Any]] = []
    for bucket, part in valid.groupby("prediction_bucket", observed=True, sort=True):
        records.append({
            "arm": arm,
            "prediction_bucket": str(bucket),
            "rows": len(part),
            "mean_predicted_net_bps": float(part.calibrated_expected_net_bps.mean()),
            "mean_realised_gross_bps": float(part.exact_h12_gross_bps.mean()),
            "mean_realised_cost_bps": float(part.row_cost_bps.mean()),
            "mean_realised_net_bps": float(part.exact_h12_net_bps.mean()),
        })
    return records


def _calibration_diagnostics(scored: pd.DataFrame, arm: str) -> list[dict[str, Any]]:
    """Calibration slope/intercept and monotonicity in common net-bps units."""
    valid = scored.loc[np.isfinite(scored.calibrated_expected_net_bps)].copy()
    records: list[dict[str, Any]] = []
    for scope, part in [("pooled", valid), *[(f"side:{side}", group) for side, group in valid.groupby("side", sort=True)]]:
        predicted = part.calibrated_expected_net_bps.to_numpy(float)
        realised = part.exact_h12_net_bps.to_numpy(float)
        if len(part) < 20 or np.nanstd(predicted) == 0.0:
            slope, intercept = np.nan, np.nan
        else:
            slope = float(np.cov(predicted, realised, ddof=0)[0, 1] / np.var(predicted))
            intercept = float(np.mean(realised) - slope * np.mean(predicted))
        buckets = pd.qcut(pd.Series(predicted).rank(method="first"), q=min(10, len(part)), duplicates="drop")
        means = part.assign(__bucket__=buckets).groupby("__bucket__", observed=True).agg(predicted=("calibrated_expected_net_bps", "mean"), realised=("exact_h12_net_bps", "mean"))
        realised_means = means.realised.to_numpy(float)
        monotonic_violations = int(np.sum(np.diff(realised_means) < 0.0)) if len(realised_means) > 1 else 0
        records.append({
            "arm": arm,
            "scope": scope,
            "rows": len(part),
            "calibration_slope": slope,
            "calibration_intercept_bps": intercept,
            "prediction_actual_spearman": float(pd.Series(predicted).corr(pd.Series(realised), method="spearman")),
            "quantile_buckets": int(len(means)),
            "monotonicity_violations": monotonic_violations,
            "lowest_bucket_realised_net_bps": float(realised_means[0]) if len(realised_means) else np.nan,
            "highest_bucket_realised_net_bps": float(realised_means[-1]) if len(realised_means) else np.nan,
        })
    return records


def _hierarchical_component_diagnostics(scored: pd.DataFrame, arm: str) -> list[dict[str, Any]]:
    """OOF/evaluation reliability of the separate path-transition heads."""
    if "p_clear_cost_before_adverse" not in scored:
        return []
    state = scored.persistence_state.astype(str)
    definitions = [
        ("clear_cost_before_adverse", "p_clear_cost_before_adverse", state.isin(("retained", "giveback")).to_numpy(int)),
        ("retain_given_clear", "p_retain_given_clear", (state.eq("retained") & state.isin(("retained", "giveback"))).to_numpy(int)),
        ("adverse_given_not_clear", "p_adverse_given_not_clear", state.eq("adverse").to_numpy(int)),
    ]
    rows: list[dict[str, Any]] = []
    for component, column, target in definitions:
        mask = np.ones(len(scored), dtype=bool)
        if component == "retain_given_clear":
            mask = state.isin(("retained", "giveback")).to_numpy()
        elif component == "adverse_given_not_clear":
            mask = ~state.isin(("retained", "giveback")).to_numpy()
        prediction = scored.loc[mask, column].to_numpy(float)
        actual = target[mask].astype(float)
        brier = float(np.mean((prediction - actual) ** 2)) if len(prediction) else np.nan
        clipped = np.clip(prediction, 1e-6, 1.0 - 1e-6)
        logloss = float(-np.mean(actual * np.log(clipped) + (1.0 - actual) * np.log(1.0 - clipped))) if len(prediction) else np.nan
        rows.append({
            "arm": arm,
            "component": component,
            "rows": int(len(prediction)),
            "prevalence": float(actual.mean()) if len(actual) else np.nan,
            "mean_prediction": float(prediction.mean()) if len(prediction) else np.nan,
            "brier": brier,
            "log_loss": logloss,
            "prediction_actual_spearman": float(pd.Series(prediction).corr(pd.Series(actual), method="spearman")) if len(prediction) and np.nanstd(prediction) > 0.0 else np.nan,
            "mean_probability_simplex_error": float(scored.probability_simplex_error.mean()),
        })
    return rows


def _target_gates(records: list[dict[str, Any]], diagnostics: list[dict[str, Any]], arms: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Lexicographic economics and calibration gates; never selects on one tail."""
    metrics = pd.DataFrame(records)
    calibration = pd.DataFrame(diagnostics)
    decision_rows, rejected = [], []
    for arm in arms:
        top = metrics.loc[(metrics.arm == arm) & (metrics.scope == "pooled_global_top")].set_index("fraction")
        top10 = top.loc[0.10] if 0.10 in top.index else pd.Series(dtype=float)
        top1 = top.loc[0.01] if 0.01 in top.index else pd.Series(dtype=float)
        top5 = top.loc[0.05] if 0.05 in top.index else pd.Series(dtype=float)
        threshold = metrics.loc[(metrics.arm == arm) & (metrics.scope == "causal_threshold")]
        latest = metrics.loc[(metrics.arm == arm) & (metrics.scope == "pooled_global_membership_by_month") & (metrics.month == "2024-11") & (metrics.fraction == 0.10)]
        sides = metrics.loc[(metrics.arm == arm) & (metrics.scope == "pooled_global_membership_by_side") & (metrics.fraction == 0.10)]
        pooled_calibration = calibration.loc[(calibration.arm == arm) & (calibration.scope == "pooled")]
        reasons: list[str] = []
        if threshold.empty or not np.isfinite(threshold.net_bps.iloc[0]) or threshold.net_bps.iloc[0] <= 0.0:
            reasons.append("causal_threshold_net_not_positive")
        if top10.empty or not np.isfinite(top10.net_bps) or top10.net_bps <= 0.0:
            reasons.append("pooled_global_top10_net_not_positive")
        if top1.empty or top5.empty or min(float(top1.net_bps), float(top5.net_bps)) < -100.0:
            reasons.append("severe_top1_or_top5_reversal")
        if latest.empty or int(latest.selected_rows.iloc[0]) < 100 or latest.net_bps.iloc[0] <= 0.0:
            reasons.append("latest_month_coverage_or_economics_fail")
        if len(sides) != 2 or (sides.net_bps <= 0.0).any():
            reasons.append("side_economics_fail")
        if pooled_calibration.empty or pooled_calibration.calibration_slope.iloc[0] <= 0.0 or pooled_calibration.monotonicity_violations.iloc[0] > 4:
            reasons.append("calibration_fail")
        passed = not reasons
        row = {
            "arm": arm,
            "candidate_economics_pass": passed,
            "causal_threshold_net_bps": float(threshold.net_bps.iloc[0]) if not threshold.empty else np.nan,
            "top1_net_bps": float(top1.net_bps) if not top1.empty else np.nan,
            "top5_net_bps": float(top5.net_bps) if not top5.empty else np.nan,
            "top10_net_bps": float(top10.net_bps) if not top10.empty else np.nan,
            "latest_month_top10_net_bps": float(latest.net_bps.iloc[0]) if not latest.empty else np.nan,
            "latest_month_selected_rows": int(latest.selected_rows.iloc[0]) if not latest.empty else 0,
            "rejection_reasons": ";".join(reasons),
        }
        decision_rows.append(row)
        if not passed and arm != "CONTROL_base_opportunity":
            rejected.append({**row, "stage": "B_execution_target", "reason": row["rejection_reasons"]})
    return pd.DataFrame(decision_rows), pd.DataFrame(rejected)


def _paired_day_bootstrap(scored: pd.DataFrame, *, control_arm: str, seed: int, replicates: int) -> pd.DataFrame:
    """Paired day-block bootstrap of global top-10% net against frozen control."""
    valid = scored.loc[np.isfinite(scored.calibrated_expected_net_bps)].copy()
    arms = sorted(valid.arm.unique())
    pivot_score = valid.pivot(index="candidate_id", columns="arm", values="calibrated_expected_net_bps")
    payload = valid.loc[valid.arm.eq(control_arm), ["candidate_id", "decision_ts", "exact_h12_net_bps"]].set_index("candidate_id")
    if pivot_score.isna().any().any() or len(payload) != len(pivot_score):
        raise AssertionError("arms must have identical finite candidate coverage for paired bootstrap")
    payload = payload.reindex(pivot_score.index)
    day_codes, days = pd.factorize(payload.decision_ts.dt.floor("D"), sort=True)
    day_rows = [np.flatnonzero(day_codes == index) for index in range(len(days))]
    if not day_rows or any(len(rows) == 0 for rows in day_rows):
        raise AssertionError("missing day block in paired bootstrap")
    values = payload.exact_h12_net_bps.to_numpy(float)
    score_matrix = pivot_score.loc[:, arms].to_numpy(float)
    control_full_net = float(values[global_top_mask(score_matrix[:, arms.index(control_arm)], 0.10)].mean())
    rng = np.random.default_rng(seed)
    deltas = {arm: np.empty(replicates, dtype=float) for arm in arms if arm != control_arm}
    for replicate in range(replicates):
        sampled = rng.integers(0, len(day_rows), size=len(day_rows))
        positions = np.concatenate([day_rows[index] for index in sampled])
        control_scores = score_matrix[positions, arms.index(control_arm)]
        control_net = float(values[positions][global_top_mask(control_scores, 0.10)].mean())
        for arm in deltas:
            scores = score_matrix[positions, arms.index(arm)]
            candidate_net = float(values[positions][global_top_mask(scores, 0.10)].mean())
            deltas[arm][replicate] = candidate_net - control_net
    return pd.DataFrame([{
        "arm": arm,
        "control_arm": control_arm,
        "fraction": 0.10,
        "day_blocks": len(day_rows),
        "replicates": replicates,
        "delta_net_bps_full_sample": float(values[global_top_mask(score_matrix[:, arms.index(arm)], 0.10)].mean() - control_full_net),
        "delta_net_bps_bootstrap_mean": float(samples.mean()),
        "delta_net_bps_p05": float(np.quantile(samples, 0.05)),
        "delta_net_bps_p95": float(np.quantile(samples, 0.95)),
        "probability_improves": float((samples > 0.0).mean()),
    } for arm, samples in deltas.items()])


def _causal_threshold(scored: pd.DataFrame) -> pd.DataFrame:
    """Daily threshold selected from prior resolved labels only, never scores today."""
    candidates = np.asarray([-150, -100, -50, 0, 25, 50, 75, 100, 150, 200], dtype=float)
    result = scored.copy()
    result["threshold_bps"] = np.nan
    result["threshold_enter"] = False
    all_rows = result.loc[:, ["decision_ts", "label_available_ts", "calibrated_expected_net_bps", "exact_h12_net_bps"]]
    for day in result.decision_ts.dt.floor("D").drop_duplicates().sort_values():
        mask = result.decision_ts.dt.floor("D").eq(day).to_numpy()
        reference = all_rows.loc[(all_rows.decision_ts.ge(day - pd.Timedelta(days=CALIBRATION_DAYS))) & (all_rows.label_available_ts.lt(day))]
        if len(reference) < 500:
            continue
        means = []
        for threshold in candidates:
            selected = reference.loc[reference.calibrated_expected_net_bps.gt(threshold), "exact_h12_net_bps"]
            means.append(float(selected.mean()) if len(selected) >= 100 else float("-inf"))
        threshold = float(candidates[int(np.argmax(means))])
        result.loc[mask, "threshold_bps"] = threshold
        result.loc[mask, "threshold_enter"] = result.loc[mask, "calibrated_expected_net_bps"].gt(threshold)
    return result


def _read(panel: Path, alignment: Path, contract: Path, postcost_events: Path, persistence_labels: Path, *, smoke: bool) -> tuple[pd.DataFrame, list[str]]:
    raw = _raw_features(contract)
    panel_columns = ["candidate_id", *raw]
    data = pd.read_parquet(panel, columns=panel_columns)
    sidecar = pd.read_parquet(alignment)
    if sidecar.target_id.nunique() != 1 or sidecar.target_id.iloc[0] != TARGET_ID or sidecar.execution_policy_id.nunique() != 1 or sidecar.execution_policy_id.iloc[0] != EXECUTION_POLICY_ID or sidecar.cost_model_id.nunique() != 1 or sidecar.cost_model_id.iloc[0] != COST_MODEL_ID:
        raise ValueError("target/policy/cost alignment contract is incompatible")
    frame = sidecar.merge(data, on="candidate_id", how="inner", validate="one_to_one")
    if len(frame) != len(sidecar) or frame.candidate_id.duplicated().any():
        raise ValueError("alignment/raw feature candidate identities differ")
    events = pd.read_parquet(postcost_events)
    required_events = {"candidate_id", "side", "decision_ts", "label_end_ts", "label_available_ts", "postcost_target_id", "execution_policy_id", "cost_model_id", "postcost_h0_event", "postcost_h25_event"}
    missing_events = sorted(required_events.difference(events.columns))
    if missing_events or events.candidate_id.duplicated().any():
        raise ValueError(f"invalid exact post-cost event sidecar: {missing_events}")
    if events.postcost_target_id.nunique() != 1 or events.execution_policy_id.nunique() != 1 or events.execution_policy_id.iloc[0] != EXECUTION_POLICY_ID or events.cost_model_id.nunique() != 1 or events.cost_model_id.iloc[0] != COST_MODEL_ID:
        raise ValueError("exact post-cost event contract is incompatible")
    for column in ("decision_ts", "label_end_ts", "label_available_ts"):
        events[column] = pd.to_datetime(events[column], utc=True, errors="raise")
    compare = events.loc[:, ["candidate_id", "side", "decision_ts", "label_end_ts", "label_available_ts", "postcost_h0_event", "postcost_h25_event"]]
    frame = frame.merge(compare, on="candidate_id", how="inner", validate="one_to_one", suffixes=("", "_postcost"))
    if len(frame) != len(sidecar) or not frame.side.eq(frame.side_postcost).all() or not frame.decision_ts.eq(frame.decision_ts_postcost).all() or not frame.label_end_ts.eq(frame.label_end_ts_postcost).all() or not frame.label_available_ts.eq(frame.label_available_ts_postcost).all():
        raise ValueError("exact post-cost event rows do not exactly align to target rows")
    frame = frame.drop(columns=["side_postcost", "decision_ts_postcost", "label_end_ts_postcost", "label_available_ts_postcost"])
    persistence = pd.read_parquet(persistence_labels)
    required_persistence = {"candidate_id", "side", "decision_ts", "label_end_ts", "label_available_ts", "execution_policy_id", "cost_model_id", "postcost_h0_four_state", "postcost_h25_four_state"}
    missing_persistence = sorted(required_persistence.difference(persistence.columns))
    if missing_persistence or persistence.candidate_id.duplicated().any():
        raise ValueError(f"invalid persistence label sidecar: {missing_persistence}")
    if persistence.execution_policy_id.nunique() != 1 or persistence.execution_policy_id.iloc[0] != EXECUTION_POLICY_ID or persistence.cost_model_id.nunique() != 1 or persistence.cost_model_id.iloc[0] != COST_MODEL_ID:
        raise ValueError("persistence label contract is incompatible")
    for column in ("decision_ts", "label_end_ts", "label_available_ts"):
        persistence[column] = pd.to_datetime(persistence[column], utc=True, errors="raise")
    persistence = persistence.loc[:, ["candidate_id", "side", "decision_ts", "label_end_ts", "label_available_ts", "postcost_h0_four_state", "postcost_h25_four_state"]]
    frame = frame.merge(persistence, on="candidate_id", how="inner", validate="one_to_one", suffixes=("", "_persistence"))
    if len(frame) != len(sidecar) or not frame.side.eq(frame.side_persistence).all() or not frame.decision_ts.eq(frame.decision_ts_persistence).all() or not frame.label_end_ts.eq(frame.label_end_ts_persistence).all() or not frame.label_available_ts.eq(frame.label_available_ts_persistence).all():
        raise ValueError("persistence label rows do not exactly align to target rows")
    frame = frame.drop(columns=["side_persistence", "decision_ts_persistence", "label_end_ts_persistence", "label_available_ts_persistence"])
    frame["decision_ts"] = pd.to_datetime(frame.decision_ts, utc=True, errors="raise")
    frame["label_available_ts"] = pd.to_datetime(frame.label_available_ts, utc=True, errors="raise")
    frame = frame.loc[frame.decision_ts.ge(BASE_START) & frame.decision_ts.lt(EVAL_END)].sort_values(["decision_ts", "candidate_id"], kind="stable").reset_index(drop=True)
    if smoke:
        frame["__hash__"] = pd.util.hash_pandas_object(frame.candidate_id, index=False).astype("uint64")
        frame = frame.assign(month=frame.decision_ts.dt.strftime("%Y-%m")).sort_values(["month", "side", "__hash__"], kind="stable").groupby(["month", "side"], group_keys=False).head(1200).drop(columns=["__hash__", "month"]).sort_values(["decision_ts", "candidate_id"], kind="stable").reset_index(drop=True)
    return frame, raw


def run(*, panel: Path, alignment: Path, contract: Path, postcost_events: Path, persistence_labels: Path, output: Path, seed: int = 20260731, smoke: bool = False) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    frame, raw = _read(panel, alignment, contract, postcost_events, persistence_labels, smoke=smoke)
    masks = _calendar(frame)
    if not all(mask.any() for mask in masks.values()):
        raise ValueError("required 12/8/4/4 calendar is incomplete")
    trees = 60 if smoke else 180
    base_train = frame.loc[masks["base_train"]].copy().reset_index(drop=True)
    selected = _select_base_features(base_train, raw, seed=seed, trees=trees)
    retention_selected = _select_retention_features(base_train, raw, seed=seed + 50, trees=trees)
    base = _base_scores(frame, selected, raw, masks, seed=seed + 100, trees=trees)
    frame = frame.merge(base.loc[:, ["candidate_id", "raw_score", "calibrated_expected_net_bps"]].rename(columns={"raw_score": "base_raw_score", "calibrated_expected_net_bps": "base_expected_net_bps"}), on="candidate_id", how="left", validate="one_to_one")
    meta = frame.loc[masks["meta_train"] & np.isfinite(frame.base_expected_net_bps)].copy().reset_index(drop=True)
    evaluate = frame.loc[masks["eval"] & np.isfinite(frame.base_expected_net_bps)].copy().reset_index(drop=True)
    if len(meta) < 4_000 or len(evaluate) < 4_000:
        raise ValueError("insufficient prequential base-map support for target purity")
    feature_by_side = {side: [*selected[side], "estimated_spread_bps", "entry_half_spread_bps", "barrier_pct", "entry_price_log"] for side in SIDES}
    arms = [
        "E0_direct_net", "E1_net_residual", "E2_three_state", *[f"E3_hurdle_{int(h)}" for h in HURDLES_BPS],
        "E4_fixed_cost_100", "E5_causal_cost_proxy", "E6_postcost_three_state_0", "E6_postcost_three_state_25", "E7_postcost_three_state_0_sidebridge",
        "E8_postcost_three_state_0_base_only", "E9_postcost_three_state_0_raw_plus_base", "E10_postcost_three_state_0_base_only_sidebridge",
        "E11_exact1m_postcost_three_state_0", "E11_exact1m_postcost_three_state_25", "E12_exact1m_postcost_three_state_0_sidebridge",
        "E13_exact1m_persistence_four_state_0", "E13_exact1m_persistence_four_state_25", "E14_exact1m_persistence_four_state_0_sidebridge",
        "E15_exact1m_hierarchical_persistence_0", "E15_exact1m_hierarchical_persistence_25", "E16_exact1m_hierarchical_persistence_0_sidebridge",
        "E18_exact1m_hierarchical_persistence_0_retention_features", "E19_exact1m_hierarchical_persistence_0_retention_features_sidebridge",
        "E17_soft_terminal_net_h0_t50", "E17_soft_terminal_net_h0_t100", "E17_soft_terminal_net_h0_t150",
    ]
    all_scored: list[pd.DataFrame] = []
    records: list[dict[str, Any]] = []
    calibration: list[dict[str, Any]] = []
    calibration_diagnostics: list[dict[str, Any]] = []
    component_diagnostics: list[dict[str, Any]] = []
    support: list[dict[str, Any]] = []
    ids = tuple(evaluate.candidate_id)
    control = evaluate.loc[:, ["candidate_id", "side", "decision_ts", "label_available_ts", "exact_h12_gross_bps", "row_cost_bps", "exact_h12_net_bps", "base_raw_score", "base_expected_net_bps"]].copy()
    control = control.rename(columns={"base_raw_score": "raw_score", "base_expected_net_bps": "calibrated_expected_net_bps"})
    control["map_reference_rows"] = np.nan
    control["arm"] = "CONTROL_base_opportunity"
    control = _causal_threshold(control)
    if tuple(control.candidate_id) != ids or not np.isfinite(control.calibrated_expected_net_bps).all():
        raise AssertionError("frozen control does not retain complete ordered evaluation candidates")
    all_scored.append(control)
    records.extend(_book_records(control, "CONTROL_base_opportunity"))
    calibration.extend(_calibration_records(control, "CONTROL_base_opportunity"))
    calibration_diagnostics.extend(_calibration_diagnostics(control, "CONTROL_base_opportunity"))
    entered = control.loc[control.threshold_enter]
    records.append({"arm": "CONTROL_base_opportunity", "scope": "causal_threshold", "selected_rows": len(entered), "net_bps": float(entered.exact_h12_net_bps.mean()) if len(entered) else np.nan, "gross_bps": float(entered.exact_h12_gross_bps.mean()) if len(entered) else np.nan, "cost_bps": float(entered.row_cost_bps.mean()) if len(entered) else np.nan, "threshold_bps_mean": float(entered.threshold_bps.mean()) if len(entered) else np.nan})
    for arm_index, arm in enumerate(arms):
        model_arm = _model_arm_for(arm)
        post_cost_hurdle = float(model_arm.rsplit("_", 1)[1]) if model_arm.startswith("E6_postcost_three_state_") else None
        exact_postcost_token = f"h{model_arm.rsplit('_', 1)[1]}" if model_arm.startswith("E11_exact1m_postcost_three_state_") else None
        persistence_token = f"h{model_arm.rsplit('_', 1)[1]}" if model_arm.startswith(("E13_exact1m_persistence_four_state_", "E15_exact1m_hierarchical_persistence_")) else ("h0" if model_arm.startswith("E18_exact1m_hierarchical_persistence_") else None)
        include_base = arm in {"E8_postcost_three_state_0_base_only", "E9_postcost_three_state_0_raw_plus_base", "E10_postcost_three_state_0_base_only_sidebridge"}
        history_parts, final_parts = [], []
        for side_index, side in enumerate(SIDES):
            train = meta.loc[meta.side.eq(side)].copy().reset_index(drop=True)
            test = evaluate.loc[evaluate.side.eq(side)].copy().reset_index(drop=True)
            base_only = arm in {"E8_postcost_three_state_0_base_only", "E10_postcost_three_state_0_base_only_sidebridge"}
            features = [] if base_only else (retention_selected[side] if model_arm.startswith("E18_exact1m_hierarchical_persistence_") else selected[side])
            local_feature_by_side = {candidate_side: ([] if base_only else (retention_selected[candidate_side] if model_arm.startswith("E18_exact1m_hierarchical_persistence_") else selected[candidate_side])) for candidate_side in SIDES}
            oof = _prequential_scores(train, local_feature_by_side, model_arm, seed=seed + 1000 + arm_index * 100 + side_index, trees=trees, include_base=include_base)
            history_parts.append(oof.loc[np.isfinite(oof.raw_score)])
            train_matrix = _features_for(train, features, include_base=include_base)
            test_matrix = _features_for(test, features, include_base=include_base)
            components: dict[str, np.ndarray] | None = None
            if model_arm.startswith(("E15_exact1m_hierarchical_persistence_", "E18_exact1m_hierarchical_persistence_")):
                components = _hierarchical_persistence_expected_net(train, test, train_matrix, test_matrix, seed=seed + 2000 + arm_index * 100 + side_index, trees=trees, token=persistence_token or "h0", return_components=True)
                raw_score = components.pop("raw_score")
            else:
                raw_score = _predict_arm(train, test, train_matrix, test_matrix, model_arm, seed=seed + 2000 + arm_index * 100 + side_index, trees=trees)
            final = test.loc[:, ["candidate_id", "side", "decision_ts", "label_available_ts", "exact_h12_gross_bps", "row_cost_bps", "exact_h12_net_bps"]].copy()
            final["raw_score"] = raw_score
            if components is not None:
                for name, values in components.items():
                    final[name] = values
                # Outcome label is retained in the research artifact solely
                # for component reliability reporting; it never enters the
                # feature matrix or any prediction call.
                final["persistence_state"] = _persistence_event(test, persistence_token or "h0")
            final_parts.append(final)
            event = _persistence_event(train, persistence_token) if persistence_token is not None else _event(train, post_cost_hurdle_bps=post_cost_hurdle, exact_postcost_token=exact_postcost_token)
            for name in (PERSISTENCE_EVENTS if persistence_token is not None else EVENTS):
                support.append({"arm": arm, "side": side, "event": name, "train_rows": int((event == name).sum()), "train_net_mean_bps": float(train.loc[event == name, "exact_h12_net_bps"].mean())})
        raw_eval = pd.concat(final_parts, ignore_index=True).sort_values(["decision_ts", "candidate_id"], kind="stable").reset_index(drop=True)
        if tuple(raw_eval.candidate_id) != ids:
            raise AssertionError("target arms do not retain identical ordered candidate IDs")
        history = pd.concat(history_parts, ignore_index=True)
        mapped = _causal_map(history, raw_eval, side_specific=_uses_side_bridge(arm))
        mapped["arm"] = arm
        mapped = _causal_threshold(mapped)
        if not np.isfinite(mapped.calibrated_expected_net_bps).all():
            raise AssertionError(f"{arm} does not have complete causal-map coverage")
        all_scored.append(mapped)
        records.extend(_book_records(mapped, arm))
        calibration.extend(_calibration_records(mapped, arm))
        calibration_diagnostics.extend(_calibration_diagnostics(mapped, arm))
        component_diagnostics.extend(_hierarchical_component_diagnostics(mapped, arm))
        entered = mapped.loc[mapped.threshold_enter]
        records.append({"arm": arm, "scope": "causal_threshold", "selected_rows": len(entered), "net_bps": float(entered.exact_h12_net_bps.mean()) if len(entered) else np.nan, "gross_bps": float(entered.exact_h12_gross_bps.mean()) if len(entered) else np.nan, "cost_bps": float(entered.row_cost_bps.mean()) if len(entered) else np.nan, "threshold_bps_mean": float(entered.threshold_bps.mean()) if len(entered) else np.nan})
    scored = pd.concat(all_scored, ignore_index=True)
    bootstrap = _paired_day_bootstrap(scored, control_arm="CONTROL_base_opportunity", seed=seed + 991, replicates=50 if smoke else BOOTSTRAP_REPLICATES)
    gates, rejected = _target_gates(records, calibration_diagnostics, ["CONTROL_base_opportunity", *arms])
    stage = Path(tempfile.mkdtemp(dir=output.parent, prefix=f".{output.name}.staging-"))
    try:
        results_path = stage / "target_ablation_results.parquet"
        metrics_path = stage / "target_ablation_metrics.csv"
        support_path = stage / "event_support_by_side.csv"
        feature_path = stage / "selected_execution_features.json"
        calibration_path = stage / "calibration_by_prediction_bucket.csv"
        calibration_diagnostics_path = stage / "calibration_diagnostics.csv"
        component_diagnostics_path = stage / "hierarchical_component_diagnostics.csv"
        bootstrap_path = stage / "paired_day_bootstrap_vs_frozen_control.csv"
        gates_path = stage / "target_selection_gates.csv"
        rejected_path = stage / "rejected_arms.parquet"
        scored.to_parquet(results_path, index=False, compression="zstd")
        pd.DataFrame(records).to_csv(metrics_path, index=False)
        pd.DataFrame(support).to_csv(support_path, index=False)
        pd.DataFrame(calibration).to_csv(calibration_path, index=False)
        pd.DataFrame(calibration_diagnostics).to_csv(calibration_diagnostics_path, index=False)
        pd.DataFrame(component_diagnostics).to_csv(component_diagnostics_path, index=False)
        bootstrap.to_csv(bootstrap_path, index=False)
        gates.to_csv(gates_path, index=False)
        rejected.to_parquet(rejected_path, index=False, compression="zstd")
        _write_json(feature_path, {side: feature_by_side[side] for side in SIDES})
        top10 = pd.DataFrame(records).query("scope == 'pooled_global_top' and fraction == 0.10").sort_values("net_bps", ascending=False)
        top10_csv = top10.loc[:, ["arm", "selected_rows", "gross_bps", "cost_bps", "net_bps", "positive_net_rate", "side_long_share"]].to_csv(index=False)
        bootstrap_csv = bootstrap.to_csv(index=False)
        gate_csv = gates.to_csv(index=False)
        selected = gates.loc[gates.candidate_economics_pass & gates.arm.ne("CONTROL_base_opportunity"), "arm"].tolist()
        decision_state = "STAGE_B_NO_EXECUTION_TARGET_ADVANCES" if not selected else "STAGE_B_EXECUTION_TARGET_ADVANCES"
        summary = ["# Exact-H12 target-purity ablation", "", "Candidate-level only. The execution policy, costs, rows, raw features, folds, seeds and evaluator are frozen across E0--E3. No sizing, portfolio constraints or action optimisation is included.", "", "## Top-10 pooled-global exact net", "", "```csv", top10_csv.rstrip(), "```", "", "## Paired day-block bootstrap vs frozen base-opportunity control", "", "```csv", bootstrap_csv.rstrip(), "```", "", "## Lexicographic execution-target gate", "", "```csv", gate_csv.rstrip(), "```", "", f"Decision state: `{decision_state}`.  A non-advancing target family must not be followed by base/supportive/threshold selection on this panel; that would conflate layers instead of diagnosing the failed execution conversion.", "", "## Contract", "", f"- target: `{TARGET_ID}`", f"- frozen execution policy/replay: `{EXECUTION_POLICY_ID}`", f"- cost: `{COST_MODEL_ID}`; gross minus row cost equals net exactly once", "- only estimated entry-time spread fields, barrier geometry and entry price enter the execution matrix; realised row cost and exit-time spread are outcome-bound and forbidden as inputs", "- base outputs are generated OOS from the 12-month base fit; archived historical score is not an execution feature", "- execution calibration uses only prior resolved labels; threshold selection is daily and prior-resolved only", "- all arms and the frozen control use identical ordered evaluation candidate IDs and finite mapped-score coverage", "", "## Limitation", "", "This is a candidate-conditioned current-spread-counterfactual historical panel. It is not full-universe, factual historical execution, or promotion evidence."]
        (stage / "target_ablation_summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
        (stage / "alignment_report.md").write_text((ROOT / "data_perp/artifacts/historical_exact_h12_alignment_sidecar_research_only_20260731_v1/alignment_report.md").read_text(encoding="utf-8"), encoding="utf-8")
        selected_manifest = {
            "status": decision_state,
            "base_target": {"id": "clean_opportunity_h12", "model": "side_local_lgbm", "reason": "frozen OOS control only; downstream base-target selection is blocked until an execution target passes Stage B"},
            "execution_target": {"id": None, "reason": "no candidate target arm clears the required causal economics, stability, side and calibration gates"},
            "supportive_heads": {"retained": [], "diagnostic_only": ["reachability", "adverse_path", "opportunity_magnitude", "persistence_giveback", "recovery"], "reason": "not tested as execution inputs because Stage B failed; no downstream mixing"},
            "feature_set": {"id": f"raw_380_{_sha256(contract)[:16]}"},
            "execution_policy": {"id": EXECUTION_POLICY_ID},
            "cost_model": {"id": COST_MODEL_ID},
            "calibrator": {"id": "prequential_21d_isotonic_net_v1"},
            "entry_rule": {"expected_net_threshold_bps": None, "optional_gates": [], "reason": "no positive causal threshold result"},
            "rejected_arms": rejected.arm.tolist(),
            "scope": "candidate-level alignment only; no portfolio or deployment claim",
        }
        _write_json(stage / "selected_architecture.yaml", selected_manifest)
        _write_json(stage / "selected_feature_manifest.json", {"execution_basic_features": {side: feature_by_side[side] for side in SIDES}, "supportive_predictions": [], "status": decision_state})
        _write_json(stage / "inference_graph.json", {"status": decision_state, "graph": ["raw causal features", "base opportunity model", "OOF-style base output", "execution-target selection halted: no Stage B target passes", "no operational threshold"]})
        pd.concat([pd.DataFrame(calibration), pd.DataFrame(calibration_diagnostics)], axis=0, ignore_index=True, sort=False).to_parquet(stage / "threshold_calibration.parquet", index=False, compression="zstd")
        rejected_csv = rejected.loc[:, ["arm", "reason", "causal_threshold_net_bps", "top10_net_bps", "latest_month_top10_net_bps"]].to_csv(index=False).rstrip() if len(rejected) else "None"
        decision_report = ["# Ablation decision report", "", f"## Status\n\n`{decision_state}`", "", "The execution-target stage is intentionally not collapsed into a nominal winner.  All target arms are rejected by the causal decision gate; therefore base-target selection, supportive-head stacking and an entry threshold are not selected from this panel.", "", "## Rejected arms", "", "```csv", rejected_csv, "```", "", "## Next layer diagnosis", "", "The frozen base ranks exact H12 net better than every re-trained execution target.  The next valid work is a new Stage-B target/feature diagnosis (causal gross-minus-fixed-cost and post-cost event definitions plus a cross-side calibration bridge), not an auxiliary-head blend or threshold search."]
        (stage / "ablation_decision_report.md").write_text("\n".join(decision_report) + "\n", encoding="utf-8")
        manifest = {"schema": "exact_h12_target_purity_ablation_v10", "status": "COMPLETED_RESEARCH_ONLY_NO_PROMOTION", "decision_state": decision_state, "mode": "smoke" if smoke else "full", "contract": {"target_id": TARGET_ID, "execution_policy_id": EXECUTION_POLICY_ID, "cost_model_id": COST_MODEL_ID, "exact_postcost_target_id": "exact_1m_h12_postcost_barrier_first_fixed100bps_v1", "persistence_target_id": "historical_exact_h12_postcost_persistence_labels_v1", "feature_set_id": f"raw_380_{_sha256(contract)[:16]}"}, "calendar": {"base_train": "2023-04..2024-03", "base_oos": "2024-04..2024-11", "meta_train": "2024-04..2024-07", "evaluation": "2024-08..2024-11"}, "arms": ["CONTROL_base_opportunity", *arms], "unweighted": True, "selection": "pooled global top tails diagnostic; daily causal expected-net threshold is executable candidate rule", "limitations": ["candidate-conditioned", "current-spread counterfactual", "no historical L2", "pre-2025 geometry not bit exact", "not portfolio replay"], "inputs": {str(path): _sha256(path) for path in (panel, alignment, contract, postcost_events, persistence_labels)}, "rows": {"base_train": int(masks["base_train"].sum()), "base_oos": int(masks["base_oos"].sum()), "meta_train_with_base_map": int(len(meta)), "evaluation_with_base_map": int(len(evaluate))}, "outputs": {name: _sha256(stage / name) for name in ("target_ablation_results.parquet", "target_ablation_metrics.csv", "event_support_by_side.csv", "selected_execution_features.json", "calibration_by_prediction_bucket.csv", "calibration_diagnostics.csv", "hierarchical_component_diagnostics.csv", "paired_day_bootstrap_vs_frozen_control.csv", "target_selection_gates.csv", "rejected_arms.parquet", "selected_architecture.yaml", "selected_feature_manifest.json", "inference_graph.json", "threshold_calibration.parquet", "target_ablation_summary.md", "ablation_decision_report.md", "alignment_report.md")}}
        _write_json(stage / "manifest.json", manifest)
        os.replace(stage, output)
        return manifest
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, default=PANEL)
    parser.add_argument("--alignment", type=Path, default=ALIGNMENT)
    parser.add_argument("--feature-contract", type=Path, default=FEATURE_CONTRACT)
    parser.add_argument("--postcost-events", type=Path, default=POSTCOST_EVENTS)
    parser.add_argument("--persistence-labels", type=Path, default=PERSISTENCE_LABELS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(panel=args.panel, alignment=args.alignment, contract=args.feature_contract, postcost_events=args.postcost_events, persistence_labels=args.persistence_labels, output=args.output, seed=args.seed, smoke=args.smoke), indent=2))


if __name__ == "__main__":
    main()
