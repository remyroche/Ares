#!/usr/bin/env python3
"""Direct exact-net tail challenger, frozen before July 20--23 outcomes.

Unlike the prior competing-risk mixture, this script learns direct cost-aware
12h net-return quantiles and a separate calibrated severe-loss expectation.
It has no timing, target-price, wait, or timeout action layer.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_july_exact_preentry_heads import IDENTITY, sha256
from scripts.run_cross_era_tail_payoff_challenger import (
    CURRENT_START,
    MAP_DAYS,
    MAP_SHRINK_ROWS,
    Fold,
    _binding,
    _normalise_matrix,
    _safe,
    _top_economics,
    _trial_key,
    _write_json,
    add_regime_composites,
    causal_side_shrunk_isotonic,
    chronological_folds,
    feature_arms,
    screen_features,
)


SCHEMA = "cross_era_direct_net_quantile_challenger_v1"
SIDES = ("long", "short")
QUANTILES = (0.10, 0.25, 0.50, 0.75)
SEVERE_THRESHOLDS = (-100.0, -200.0, -400.0)
CURRENT_ERA = "2026_may_jul19"
SCORE_FORMS = {
    "q10": "q10_net_bps",
    "q25": "q25_net_bps",
    "q50": "q50_net_bps",
    "q50_minus_severe": "score_median_minus_severe_bps",
}

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


def _fit_quantile(matrix: pd.DataFrame, target: np.ndarray, config: Mapping[str, Any], alpha: float, seed: int) -> Any:
    if len(target) < 120:
        return float(np.quantile(target, alpha))
    model = lgb.LGBMRegressor(
        objective="quantile", alpha=float(alpha), n_estimators=int(config["n_estimators"]),
        learning_rate=float(config["learning_rate"]), num_leaves=int(config["num_leaves"]),
        max_depth=int(config["max_depth"]), min_child_samples=max(100, int(config["min_child_samples"]) // 2),
        reg_lambda=float(config["reg_lambda"]), colsample_bytree=.8, subsample=.85, subsample_freq=1,
        random_state=seed, n_jobs=4, verbosity=-1,
    )
    model.fit(matrix, np.asarray(target, dtype=float))
    return model


def _predict_quantile(model: Any, matrix: pd.DataFrame) -> np.ndarray:
    if isinstance(model, (float, int, np.floating)):
        return np.full(len(matrix), float(model), dtype=float)
    return np.asarray(model.predict(matrix), dtype=float)


def _fit_binary(matrix: pd.DataFrame, target: np.ndarray, config: Mapping[str, Any], seed: int) -> Any:
    y = np.asarray(target, dtype=int)
    if y.min() == y.max():
        return float(y.mean())
    counts = np.bincount(y, minlength=2).astype(float)
    weights = len(y) / (2.0 * np.maximum(counts[y], 1.0))
    model = lgb.LGBMClassifier(
        objective="binary", n_estimators=int(config["n_estimators"]), learning_rate=float(config["learning_rate"]),
        num_leaves=int(config["num_leaves"]), max_depth=int(config["max_depth"]),
        min_child_samples=int(config["min_child_samples"]), reg_lambda=float(config["reg_lambda"]),
        colsample_bytree=.8, subsample=.85, subsample_freq=1, random_state=seed, n_jobs=4, verbosity=-1,
    )
    model.fit(matrix, y, sample_weight=weights)
    return model


def _predict_binary(model: Any, matrix: pd.DataFrame) -> np.ndarray:
    if isinstance(model, (float, int, np.floating)):
        return np.full(len(matrix), float(model), dtype=float)
    return np.clip(np.asarray(model.predict_proba(matrix)[:, 1], dtype=float), 1e-6, 1.0 - 1e-6)


def monotone_severe_probabilities(frame: pd.DataFrame) -> pd.DataFrame:
    """Project independently calibrated threshold probabilities monotonically."""
    result = frame.copy()
    p100 = np.clip(result["p_loss_le_100"].to_numpy(float), 0.0, 1.0)
    p200 = np.clip(result["p_loss_le_200"].to_numpy(float), 0.0, 1.0)
    p400 = np.clip(result["p_loss_le_400"].to_numpy(float), 0.0, 1.0)
    p200 = np.maximum(p200, p400)
    p100 = np.maximum(p100, p200)
    result["p_loss_le_100"], result["p_loss_le_200"], result["p_loss_le_400"] = p100, p200, p400
    return result


def monotone_quantiles(frame: pd.DataFrame) -> pd.DataFrame:
    """Project direct quantiles to their required row-wise order."""
    result = frame.copy()
    columns = ["q10_net_bps", "q25_net_bps", "q50_net_bps", "q75_net_bps"]
    result[columns] = np.sort(result[columns].to_numpy(float), axis=1)
    return result


def compose_scores(frame: pd.DataFrame) -> pd.DataFrame:
    """Direct quantile utilities plus non-overlapping calibrated tail buckets."""
    result = monotone_quantiles(monotone_severe_probabilities(frame))
    p100 = result["p_loss_le_100"] - result["p_loss_le_200"]
    p200 = result["p_loss_le_200"] - result["p_loss_le_400"]
    p400 = result["p_loss_le_400"]
    result["severe_expected_loss_bps"] = (
        p100 * result["q75_loss_100_200_bps"]
        + p200 * result["q75_loss_200_400_bps"]
        + p400 * result["q75_loss_400_plus_bps"]
    )
    result["score_median_bps"] = result["q50_net_bps"]
    result["score_lower_quantile_bps"] = result["q25_net_bps"]
    result["score_median_minus_severe_bps"] = result["q50_net_bps"] - result["severe_expected_loss_bps"]
    return result


def _inner_calibrator(
    frame: pd.DataFrame, matrix: pd.DataFrame, target: np.ndarray, train: np.ndarray,
    config: Mapping[str, Any], seed: int, count: int,
) -> tuple[Any, dict[str, Any]]:
    """Train-only chronological binary probability calibration for one side/head."""
    ordered = train[np.argsort(frame.iloc[train]["__ts__"].to_numpy())]
    calibration_count = max(500, int(math.ceil(.20 * len(ordered))))
    calibration = ordered[-calibration_count:]
    inner = np.setdiff1d(train, calibration, assume_unique=False)
    if len(inner) < 5_000 or np.unique(target[calibration]).size < 2:
        return None, {"status": "raw_probability", "inner_rows": int(len(inner)), "calibration_rows": int(len(calibration))}
    features = screen_features(matrix, target.astype(float), inner, count, multiclass=False)
    median = matrix.iloc[inner][features].median()
    model = _fit_binary(matrix.iloc[inner][features].fillna(median), target[inner], config, seed)
    raw = _predict_binary(model, matrix.iloc[calibration][features].fillna(median))
    calibration_model = IsotonicRegression(y_min=1e-6, y_max=1.0 - 1e-6, out_of_bounds="clip")
    calibration_model.fit(raw, target[calibration])
    return calibration_model, {"status": "inner_train_chronological", "inner_rows": int(len(inner)), "calibration_rows": int(len(calibration)), "features": features}


def _apply_calibrator(raw: np.ndarray, calibrator: Any) -> np.ndarray:
    return np.asarray(raw if calibrator is None else calibrator.predict(raw), dtype=float)


def _bucket_target(net_bps: np.ndarray, bucket: str) -> tuple[np.ndarray, np.ndarray]:
    if bucket == "100_200":
        mask = (net_bps <= -100.0) & (net_bps > -200.0)
    elif bucket == "200_400":
        mask = (net_bps <= -200.0) & (net_bps > -400.0)
    elif bucket == "400_plus":
        mask = net_bps <= -400.0
    else:
        raise ValueError(bucket)
    return mask, -net_bps[mask]


def _fit_fold(
    frame: pd.DataFrame, matrix: pd.DataFrame, fold: Fold, config: Mapping[str, Any], seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    net_bps = pd.to_numeric(frame["execution_net_ev_12h"], errors="raise").to_numpy(float) * 1e4
    output = frame.iloc[fold.valid].loc[:, [*IDENTITY, "era", "label_resolution_utc"]].copy()
    output["execution_net_ev_12h"] = net_bps[fold.valid] / 1e4
    records: dict[str, Any] = {"quantiles": {}, "severe_probability": {}, "severe_magnitude": {}}
    for side_index, side in enumerate(SIDES):
        train = fold.train[frame.iloc[fold.train]["side_name"].astype(str).eq(side).to_numpy()]
        valid = fold.valid[frame.iloc[fold.valid]["side_name"].astype(str).eq(side).to_numpy()]
        positions = output.index[output["side_name"].astype(str).eq(side)]
        if len(train) < 5_000 or len(valid) < 100:
            raise ValueError(f"insufficient {side} support for {fold.name}")
        quantile_features = screen_features(matrix, net_bps, train, int(config["feature_count"]), multiclass=False)
        quantile_median = matrix.iloc[train][quantile_features].median()
        for alpha in QUANTILES:
            label = f"q{int(alpha * 100):02d}_net_bps"
            model = _fit_quantile(matrix.iloc[train][quantile_features].fillna(quantile_median), net_bps[train], config, alpha, seed + side_index * 100 + int(alpha * 100))
            output.loc[positions, label] = _predict_quantile(model, matrix.iloc[valid][quantile_features].fillna(quantile_median))
            records["quantiles"].setdefault(side, {})[label] = quantile_features
        for threshold in SEVERE_THRESHOLDS:
            label = f"p_loss_le_{abs(int(threshold))}"
            target = (net_bps <= threshold).astype(np.int8)
            features = screen_features(matrix, target.astype(float), train, int(config["feature_count"]), multiclass=False)
            median = matrix.iloc[train][features].median()
            model = _fit_binary(matrix.iloc[train][features].fillna(median), target[train], config, seed + side_index * 100 + abs(int(threshold)))
            raw = _predict_binary(model, matrix.iloc[valid][features].fillna(median))
            calibrator, calibration_record = _inner_calibrator(frame, matrix, target, train, config, seed + side_index * 1000 + abs(int(threshold)), int(config["feature_count"]))
            output.loc[positions, f"raw_{label}"] = raw
            output.loc[positions, label] = _apply_calibrator(raw, calibrator)
            records["severe_probability"].setdefault(side, {})[label] = {"features": features, "calibration": calibration_record}
        for bucket in ("100_200", "200_400", "400_plus"):
            label = f"q75_loss_{bucket}_bps"
            mask, _ = _bucket_target(net_bps, bucket)
            conditional = train[mask[train]]
            features = screen_features(matrix, -net_bps, conditional, min(int(config["feature_count"]), 24), multiclass=False)
            median = matrix.iloc[conditional][features].median()
            model = _fit_quantile(matrix.iloc[conditional][features].fillna(median), -net_bps[conditional], config, .75, seed + side_index * 300 + len(bucket))
            output.loc[positions, label] = _predict_quantile(model, matrix.iloc[valid][features].fillna(median))
            records["severe_magnitude"].setdefault(side, {})[label] = features
    return compose_scores(output), records


def _side_diagnostics(frame: pd.DataFrame, score_column: str) -> dict[str, float]:
    """Report allocation and side-local tail economics; never use a side gate."""
    selected_count = max(1, int(math.ceil(.10 * len(frame))))
    selected = frame.sort_values([score_column, "candidate_id"], ascending=[False, True], kind="stable").iloc[:selected_count]
    result: dict[str, float] = {}
    for side in SIDES:
        global_side = selected.loc[selected["side_name"].astype(str).eq(side)]
        local = frame.loc[frame["side_name"].astype(str).eq(side)]
        local_take = max(1, int(math.ceil(.10 * len(local))))
        local_top = local.sort_values([score_column, "candidate_id"], ascending=[False, True], kind="stable").iloc[:local_take]
        net = local_top["execution_net_ev_12h"].to_numpy(float) * 1e4
        result[f"global_top10_{side}_rows"] = float(len(global_side))
        result[f"global_top10_{side}_coverage"] = float(len(global_side) / len(frame))
        result[f"side_top10_{side}_net_ev_bps"] = float(net.mean())
        result[f"side_top10_{side}_cvar05_bps"] = float(np.sort(net)[:max(1, int(math.ceil(.05 * len(net))))].mean())
    return result


def run_oof_trials(
    frame: pd.DataFrame,
    arms: Mapping[str, Sequence[str]],
    seed: int,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    folds = chronological_folds(frame)
    trials, stored, months = [], {}, {}
    for arm_index, (arm, columns) in enumerate(arms.items()):
        matrix = _normalise_matrix(frame, columns)
        for config_index, config in enumerate(HPO_CONFIGS):
            key = f"{arm}__{config['name']}"
            print(f"direct-net: start {key}", flush=True)
            parts, fold_records = [], []
            for fold_index, fold in enumerate(folds):
                print(f"direct-net: {key} {fold.name}", flush=True)
                part, record = _fit_fold(frame, matrix, fold, config, seed + arm_index * 10000 + config_index * 1000 + fold_index * 100)
                parts.append(part)
                fold_records.append({"fold": fold.name, "train_rows": len(fold.train), "validation_rows": len(fold.valid), "features": record})
            oof = pd.concat(parts, ignore_index=True)
            for score_name, score_column in SCORE_FORMS.items():
                trial_key = f"{key}__{score_name}"
                mapped_column = f"mapped_{score_name}_bps"
                scored = oof.copy()
                scored[mapped_column], mapping = causal_side_shrunk_isotonic(scored, score_column)
                metrics, monthly = _top_economics(scored, mapped_column)
                metrics.update(_side_diagnostics(scored, mapped_column))
                trials.append({"trial_key": trial_key, "arm": arm, "config": dict(config), "score_name": score_name, "score_column": score_column, "mapped_column": mapped_column, "metrics": metrics, "folds": fold_records, "mapping": mapping})
                stored[trial_key], months[trial_key] = scored, monthly.assign(trial_key=trial_key)
                print(f"direct-net: completed {trial_key} top10={metrics['global_top10_net_ev_bps']:.3f}", flush=True)
    winner = max(trials, key=lambda x: _trial_key(x["metrics"]))
    table = pd.DataFrame([{"trial_key": x["trial_key"], "arm": x["arm"], "score_name": x["score_name"], **x["metrics"]} for x in trials]).sort_values(["global_top10_net_ev_bps", "worst_month_top10_net_ev_bps"], ascending=False, kind="stable")
    return (
        winner,
        stored[winner["trial_key"]].copy(),
        table,
        months[winner["trial_key"]].copy(),
    )


def _fit_final(frame: pd.DataFrame, columns: Sequence[str], config: Mapping[str, Any], seed: int) -> tuple[dict[str, Any], dict[str, Any]]:
    matrix = _normalise_matrix(frame, columns)
    net_bps = pd.to_numeric(frame["execution_net_ev_12h"], errors="raise").to_numpy(float) * 1e4
    bundle: dict[str, Any] = {"columns": list(columns), "quantiles": {}, "probabilities": {}, "magnitudes": {}, "calibrators": {}}
    state: dict[str, Any] = {"sides": {}, "calibration": {}}
    cutoff = pd.Timestamp("2026-07-01T00:00:00Z")
    for side_index, side in enumerate(SIDES):
        positions = np.flatnonzero(frame["side_name"].astype(str).eq(side).to_numpy())
        state["sides"][side] = {"quantiles": {}, "probabilities": {}, "magnitudes": {}}
        quantile_features = screen_features(matrix, net_bps, positions, int(config["feature_count"]), multiclass=False)
        quantile_median = matrix.iloc[positions][quantile_features].median()
        for alpha in QUANTILES:
            label = f"q{int(alpha * 100):02d}_net_bps"
            bundle["quantiles"].setdefault(side, {})[label] = {"features": quantile_features, "median": quantile_median, "model": _fit_quantile(matrix.iloc[positions][quantile_features].fillna(quantile_median), net_bps[positions], config, alpha, seed + side_index * 100 + int(alpha * 100))}
            state["sides"][side]["quantiles"][label] = quantile_features
        calibration_inner = np.flatnonzero(frame["side_name"].astype(str).eq(side).to_numpy() & frame["label_resolution_utc"].lt(cutoff).to_numpy())
        calibration_rows = np.flatnonzero(frame["side_name"].astype(str).eq(side).to_numpy() & frame["era"].astype(str).eq(CURRENT_ERA).to_numpy() & frame["__ts__"].ge(cutoff).to_numpy())
        bundle["calibrators"][side] = {}
        state["calibration"][side] = {}
        for threshold in SEVERE_THRESHOLDS:
            label = f"p_loss_le_{abs(int(threshold))}"
            target = (net_bps <= threshold).astype(np.int8)
            features = screen_features(matrix, target.astype(float), positions, int(config["feature_count"]), multiclass=False)
            median = matrix.iloc[positions][features].median()
            bundle["probabilities"].setdefault(side, {})[label] = {"features": features, "median": median, "model": _fit_binary(matrix.iloc[positions][features].fillna(median), target[positions], config, seed + side_index * 100 + abs(int(threshold)))}
            inner_features = screen_features(matrix, target.astype(float), calibration_inner, int(config["feature_count"]), multiclass=False)
            inner_median = matrix.iloc[calibration_inner][inner_features].median()
            inner_model = _fit_binary(matrix.iloc[calibration_inner][inner_features].fillna(inner_median), target[calibration_inner], config, seed + 5000 + side_index * 100 + abs(int(threshold)))
            raw = _predict_binary(inner_model, matrix.iloc[calibration_rows][inner_features].fillna(inner_median))
            calibrator = IsotonicRegression(y_min=1e-6, y_max=1.0 - 1e-6, out_of_bounds="clip")
            calibrator.fit(raw, target[calibration_rows])
            bundle["calibrators"][side][label] = calibrator
            state["sides"][side]["probabilities"][label] = features
            state["calibration"][side][label] = {"inner_train_rows": int(len(calibration_inner)), "calibration_rows": int(len(calibration_rows)), "inner_train_end_exclusive": cutoff, "features": inner_features}
        for bucket in ("100_200", "200_400", "400_plus"):
            label = f"q75_loss_{bucket}_bps"
            mask, _ = _bucket_target(net_bps, bucket)
            conditional = positions[mask[positions]]
            features = screen_features(matrix, -net_bps, conditional, min(int(config["feature_count"]), 24), multiclass=False)
            median = matrix.iloc[conditional][features].median()
            bundle["magnitudes"].setdefault(side, {})[label] = {"features": features, "median": median, "model": _fit_quantile(matrix.iloc[conditional][features].fillna(median), -net_bps[conditional], config, .75, seed + 600 + side_index * 100 + len(bucket))}
            state["sides"][side]["magnitudes"][label] = features
    return bundle, state


def _prepare_current(path: Path, columns: Sequence[str]) -> pd.DataFrame:
    current = pd.read_parquet(path)
    current["era"] = CURRENT_ERA
    current = current.rename(columns={"base_candidate_group_rows": "candidate_group_size", "base_margin_to_cutoff": "base_margin_to_candidate_cutoff"})
    current, _ = add_regime_composites(current)
    _normalise_matrix(current, columns)
    return current


def score_current(bundle: Mapping[str, Any], current: pd.DataFrame) -> pd.DataFrame:
    matrix = _normalise_matrix(current, bundle["columns"])
    output = current.loc[:, list(IDENTITY)].copy()
    for side in SIDES:
        pos = np.flatnonzero(current["side_name"].astype(str).eq(side).to_numpy())
        for label, record in bundle["quantiles"][side].items():
            output.loc[output.index[pos], label] = _predict_quantile(record["model"], matrix.iloc[pos][record["features"]].fillna(record["median"]))
        for label, record in bundle["probabilities"][side].items():
            raw = _predict_binary(record["model"], matrix.iloc[pos][record["features"]].fillna(record["median"]))
            output.loc[output.index[pos], f"raw_{label}"] = raw
            output.loc[output.index[pos], label] = _apply_calibrator(raw, bundle["calibrators"][side][label])
        for label, record in bundle["magnitudes"][side].items():
            output.loc[output.index[pos], label] = _predict_quantile(record["model"], matrix.iloc[pos][record["features"]].fillna(record["median"]))
    return compose_scores(output)


def _current_economics(scored: pd.DataFrame, score_column: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    take = max(1, int(math.ceil(.10 * len(scored))))
    selected = scored.sort_values([score_column, "candidate_id"], ascending=[False, True], kind="stable").iloc[:take]
    rows = []
    for scope, local in (("global_top10", selected), ("all", scored)):
        net = local["execution_net_ev_12h"].to_numpy(float) * 1e4
        rows.append({"scope": scope, "rows": len(local), "net_ev_bps": float(net.mean()), "positive_precision": float((net > 0).mean()), "cvar05_bps": float(np.sort(net)[:max(1, int(math.ceil(.05 * len(net))))].mean()), "long_rows": int(local["side_name"].astype(str).eq("long").sum()), "short_rows": int(local["side_name"].astype(str).eq("short").sum())})
    support = [{"side": side, "top10_rows": int(selected["side_name"].astype(str).eq(side).sum()), "coverage": float(selected["side_name"].astype(str).eq(side).sum() / len(scored))} for side in SIDES]
    return pd.DataFrame(rows), pd.DataFrame(support)


def _tail_economics_by_period(
    scored: pd.DataFrame,
    score_column: str,
    split: str,
) -> pd.DataFrame:
    """Persist global-book and side-local tail evidence without side quotas."""
    work = scored.loc[np.isfinite(scored[score_column])].copy()
    work["month"] = pd.to_datetime(work["__ts__"], utc=True).dt.strftime("%Y-%m")
    work["day"] = pd.to_datetime(work["__ts__"], utc=True).dt.strftime("%Y-%m-%d")
    rows: list[dict[str, Any]] = []
    periods: list[tuple[str, str, pd.DataFrame]] = [("aggregate", "all", work)]
    periods.extend(("month", str(key), local) for key, local in work.groupby("month", sort=True))
    if split == "current":
        periods.extend(("day", str(key), local) for key, local in work.groupby("day", sort=True))
    for level, period, local in periods:
        take = max(1, int(math.ceil(.10 * len(local))))
        selected = local.sort_values(
            [score_column, "candidate_id"],
            ascending=[False, True],
            kind="stable",
        ).iloc[:take]
        for scope, cohort in (("global", selected), *(
            (
                f"side_local_{side}",
                side_frame.sort_values(
                    [score_column, "candidate_id"],
                    ascending=[False, True],
                    kind="stable",
                ).iloc[: max(1, int(math.ceil(.10 * len(side_frame))))],
            )
            for side in SIDES
            for side_frame in [local.loc[local["side_name"].astype(str).eq(side)]]
            if len(side_frame)
        )):
            net = cohort["execution_net_ev_12h"].to_numpy(float) * 1e4
            rows.append({
                "split": split,
                "level": level,
                "period": period,
                "scope": scope,
                "rows": int(len(cohort)),
                "net_ev_bps": float(net.mean()),
                "positive_precision": float((net > 0).mean()),
                "cvar05_bps": float(np.sort(net)[: max(1, int(math.ceil(.05 * len(net))))].mean()),
                "long_rows": int(cohort["side_name"].astype(str).eq("long").sum()),
                "short_rows": int(cohort["side_name"].astype(str).eq("short").sum()),
            })
    return pd.DataFrame(rows)


def _probability_calibration_metrics(scored: pd.DataFrame, split: str) -> pd.DataFrame:
    """Compare raw and calibrated severe-loss probabilities by side/month."""
    work = scored.copy()
    work["month"] = pd.to_datetime(work["__ts__"], utc=True).dt.strftime("%Y-%m")
    net_bps = pd.to_numeric(work["execution_net_ev_12h"], errors="raise").to_numpy(float) * 1e4
    rows: list[dict[str, Any]] = []
    for side in SIDES:
        side_mask = work["side_name"].astype(str).eq(side).to_numpy()
        for month in ("all", *sorted(work.loc[side_mask, "month"].unique())):
            mask = side_mask if month == "all" else side_mask & work["month"].eq(month).to_numpy()
            for threshold in SEVERE_THRESHOLDS:
                label = f"p_loss_le_{abs(int(threshold))}"
                truth = (net_bps[mask] <= threshold).astype(float)
                for calibration, column in (("raw", f"raw_{label}"), ("calibrated", label)):
                    if column not in work:
                        continue
                    probability = np.clip(work.loc[mask, column].to_numpy(float), 0.0, 1.0)
                    bins = np.minimum((probability * 10).astype(int), 9)
                    ece = 0.0
                    for bin_index in range(10):
                        local = bins == bin_index
                        if local.any():
                            ece += local.mean() * abs(probability[local].mean() - truth[local].mean())
                    rows.append({
                        "split": split,
                        "side_name": side,
                        "month": month,
                        "head": label,
                        "calibration": calibration,
                        "rows": int(mask.sum()),
                        "predicted_mean": float(probability.mean()),
                        "actual_rate": float(truth.mean()),
                        "brier": float(np.mean((probability - truth) ** 2)),
                        "ece10": float(ece),
                    })
    return pd.DataFrame(rows)


def _assert_complete_current_labels(
    predictions: pd.DataFrame,
    labels: pd.DataFrame,
) -> dict[str, Any]:
    """Fail closed unless current predictions and labels have identical IDs."""
    if predictions.duplicated(list(IDENTITY)).any():
        raise ValueError("current predictions contain duplicate identities")
    if labels.duplicated(list(IDENTITY)).any():
        raise ValueError("current labels contain duplicate identities")
    prediction_ids = predictions.loc[:, list(IDENTITY)].sort_values(list(IDENTITY)).reset_index(drop=True)
    label_ids = labels.loc[:, list(IDENTITY)].sort_values(list(IDENTITY)).reset_index(drop=True)
    if len(prediction_ids) != len(label_ids) or not prediction_ids.equals(label_ids):
        missing = prediction_ids.merge(label_ids, on=list(IDENTITY), how="left", indicator=True)
        extra = label_ids.merge(prediction_ids, on=list(IDENTITY), how="left", indicator=True)
        raise ValueError(
            "current label identity coverage mismatch: "
            f"predictions={len(prediction_ids)} labels={len(label_ids)} "
            f"missing={(missing['_merge'] == 'left_only').sum()} "
            f"extra={(extra['_merge'] == 'left_only').sum()}"
        )
    return {
        "prediction_rows": int(len(prediction_ids)),
        "label_rows": int(len(label_ids)),
        "identity_complete_one_to_one": True,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    manifest = json.loads((args.dataset_dir / "manifest.json").read_text())
    dataset = args.dataset_dir / "cross_era_tail_payoff_dataset.parquet"
    if sha256(dataset) != manifest["outputs"]["dataset"]["sha256"]:
        raise ValueError("dataset hash mismatch")
    contract = json.loads((args.dataset_dir / "feature_contract.json").read_text())
    history = pd.read_parquet(dataset)
    history["__ts__"] = pd.to_datetime(history["__ts__"], utc=True)
    history["label_resolution_utc"] = pd.to_datetime(history["label_resolution_utc"], utc=True)
    history = history.loc[history["label_resolution_utc"].lt(CURRENT_START)].reset_index(drop=True)
    history, composites = add_regime_composites(history)
    arms = feature_arms(contract, composites)
    winner, oof, trials, winner_monthly = run_oof_trials(history, arms, args.seed)
    bundle, final_state = _fit_final(history, arms[winner["arm"]], winner["config"], args.seed + 1_000_000)
    model_path = args.output_dir / "frozen_models.joblib"
    joblib.dump(bundle, model_path)
    frozen = {
        "schema": SCHEMA,
        "selection_status": "historical_oof_global_economics_only",
        "current_outcomes_used_for_selection": False,
        "dataset": _binding(dataset),
        "dataset_manifest": _binding(args.dataset_dir / "manifest.json"),
        "feature_contract": _binding(args.dataset_dir / "feature_contract.json"),
        "current_feature_pack_declared": _binding(args.current_packb),
        "winner": {
            "trial_key": winner["trial_key"],
            "arm": winner["arm"],
            "config": winner["config"],
            "score_name": winner["score_name"],
            "score_column": winner["score_column"],
            "mapped_column": winner["mapped_column"],
            "metrics": winner["metrics"],
        },
        "final_state": final_state,
        "model": _binding(model_path),
        "mapping": {
            "type": "causal_21d_side_shrunk_isotonic",
            "window_days": MAP_DAYS,
            "side_shrink_rows": MAP_SHRINK_ROWS,
        },
        "calibration_caveat": (
            "The final severe-probability calibrator is learned from a pre-July "
            "inner model on July 1--19, then applied to an all-history refit. "
            "This is causal but not exact prediction-model parity; raw and "
            "calibrated diagnostics are mandatory."
        ),
        "contract": {
            "ranking": "one pooled global top-k after causal mapping",
            "costs": "exact net labels are cost-aware",
            "actions": "no timing/action heads",
            "support": "reporting only; no fixed side threshold",
        },
    }
    frozen_path = args.output_dir / "frozen_before_current_evaluation.json"
    _write_json(frozen_path, frozen)
    frozen_sha = sha256(frozen_path)
    current = _prepare_current(args.current_packb, arms[winner["arm"]])
    current_scores = score_current(bundle, current)
    current_mapped_column = winner["mapped_column"]
    current_scores[current_mapped_column], current_mapping = causal_side_shrunk_isotonic(oof, winner["score_column"], current=current_scores)
    current_path = args.output_dir / "current_predictions_before_outcomes.parquet"
    current_scores.to_parquet(current_path, index=False)
    current_sha = sha256(current_path)
    labels = pd.read_parquet(args.current_labels)
    coverage = _assert_complete_current_labels(current_scores, labels)
    scored = current_scores.merge(labels.loc[:, [*IDENTITY, "execution_net_ev_12h"]], on=list(IDENTITY), how="inner", validate="one_to_one")
    if len(scored) != len(current_scores):
        raise ValueError("current evaluation is not complete one-to-one coverage")
    economics, support = _current_economics(scored, current_mapped_column)
    historical_period_economics = _tail_economics_by_period(oof, winner["mapped_column"], "historical_oof")
    current_period_economics = _tail_economics_by_period(scored, current_mapped_column, "current")
    historical_calibration = _probability_calibration_metrics(oof, "historical_oof")
    current_calibration = _probability_calibration_metrics(scored, "current")
    coverage = {
        **coverage,
        "scored_rows": int(len(scored)),
        "current_feature_pack": _binding(args.current_packb),
        "current_labels": _binding(args.current_labels),
    }
    outputs = {}
    for name, table, suffix in (
        ("historical_oof_winner", oof, ".parquet"),
        ("historical_trial_metrics", trials, ".csv"),
        ("historical_winner_monthly", winner_monthly, ".csv"),
        ("historical_period_economics", historical_period_economics, ".csv"),
        ("historical_probability_calibration", historical_calibration, ".csv"),
        ("current_predictions_before_outcomes", current_scores, ".parquet"),
        ("current_scored_exact", scored, ".parquet"),
        ("current_economics", economics, ".csv"),
        ("current_support", support, ".csv"),
        ("current_period_economics", current_period_economics, ".csv"),
        ("current_probability_calibration", current_calibration, ".csv"),
    ):
        path = current_path if name == "current_predictions_before_outcomes" else args.output_dir / f"{name}{suffix}"
        if name != "current_predictions_before_outcomes":
            table.to_parquet(path, index=False) if suffix == ".parquet" else table.to_csv(path, index=False)
        outputs[name] = {**_binding(path), "rows": len(table)}
    report = {
        "schema": SCHEMA,
        "status": "completed_research_only_no_promotion",
        "promotion_eligible": False,
        "current_outcomes_used_for_selection": False,
        "frozen_state": {
            "path": str(frozen_path),
            "sha256_before_current_features": frozen_sha,
        },
        "current_score_before_outcomes": {
            "path": str(current_path),
            "sha256_before_current_labels": current_sha,
        },
        "current_coverage": coverage,
        "winner": frozen["winner"],
        "current_mapping": current_mapping,
        "outputs": outputs,
    }
    _write_json(args.output_dir / "report.json", report)
    _write_json(args.output_dir / "manifest.json", {"schema": SCHEMA, "status": report["status"], "promotion_eligible": False, "frozen_state_sha256": frozen_sha, "report": _binding(args.output_dir / "report.json"), "outputs": outputs})
    return report


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-dir", type=Path, default=Path("data_perp/artifacts/cross_era_tail_payoff_dataset_20260730_v3"))
    p.add_argument("--current-packb", type=Path, default=Path("data_perp/artifacts/execution_ev_july20_23_retrospective_20260730_v2/packb/packb_forward_context.parquet"))
    p.add_argument("--current-labels", type=Path, default=Path("data_perp/artifacts/execution_ev_july20_23_retrospective_20260730_v2/labels_12h/execution_ev_policy_labels.parquet"))
    p.add_argument("--output-dir", type=Path, default=Path("data_perp/artifacts/cross_era_direct_net_quantile_challenger_20260730_v1"))
    p.add_argument("--seed", type=int, default=20260730)
    return p


if __name__ == "__main__":
    run(parser().parse_args())
