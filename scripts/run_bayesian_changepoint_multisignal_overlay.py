#!/usr/bin/env python3
"""Evaluate multi-signal BOCPD guards with train-only Bayesian prototype matching.

Inputs are deliberately restricted to causal, frozen outputs: the market-state
BOCPD score, prior local LGBM/MLP risk percentiles, and activations of prior
Bayesian Rule List terms.  For each side x archetype, a nearest-prototype
likelihood score compares the current three-channel state with train-only
adverse timestamps and matched benign controls.  It is an overlay diagnostic;
it cannot refit or alter the V9 parent model.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import lightgbm as lgb

from extreme_price_movements.bayesian_changepoint import BOCPDConfig, bocpd_student_t
from scripts import run_bayesian_changepoint_matched_control_overlay as base


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data_perp/reports/meta_residual_interpretable_rule_overlay_20260714_v18_episode_intervention"
CHANGEPOINTS = ROOT / "data_perp/reports/bayesian_market_state_changepoints_20260714_v5_strict/hourly_changepoint_scores.csv.gz"
STATE = ROOT / "data_perp/reports/meta_residual_interpretable_rule_overlay_20260714_v10_corrected_clock/negative_residual_market_features_20250301_20260712.parquet"
RULES = SOURCE / "extracted_rules.csv"
OUTPUT = ROOT / "data_perp/reports/bayesian_changepoint_multisignal_overlay_20260715_v1"
FOLDS = base.FOLDS
LEAF_STATE_DEFAULTS = (
    "negative_breadth_pct", "breadth_dispersion", "downside_breadth_intensity",
    "short_covering_score_market", "funding_deleveraging_divergence",
    "flush_recovery_state", "market_state_persistence_5d",
    "compression_quality_consistency", "breakout_confirmation_ratio",
    "peer_volatility_decoupling",
)
BRL_TERM = re.compile(r"IF\s+([A-Za-z0-9_]+)__(high20|low20)\s+>\s+0\.5")


def _load_rows() -> pd.DataFrame:
    requested = [
        "__ts__", "side_name", "archetype_policy_key", "parent_rank_v9",
        "ev_after_1pct", "clean_exec", "dirty_positive", "full_path_bad_mae_1r",
        "timeout", "top10_adverse_period_target", "residual_error_risk_percentile",
        "interpretable_rule_risk_percentile",
    ]
    frames: list[pd.DataFrame] = []
    for filename in ("train_oof_predictions.parquet", "oos_predictions.parquet"):
        path = SOURCE / filename
        available = set(pq.read_schema(path).names)
        frame = pd.read_parquet(path, columns=[name for name in requested if name in available])
        for name in requested:
            if name not in frame:
                frame[name] = np.nan
        frame["__ts__"] = pd.to_datetime(frame["__ts__"], utc=True)
        frame = frame.loc[pd.to_numeric(frame["parent_rank_v9"], errors="coerce").ge(0.90)].copy()
        frame["local_model_risk_pct"] = pd.to_numeric(
            frame["residual_error_risk_percentile"], errors="coerce"
        )
        missing_risk = frame["local_model_risk_pct"].isna()
        frame.loc[missing_risk, "local_model_risk_pct"] = pd.to_numeric(
            frame.loc[missing_risk, "interpretable_rule_risk_percentile"], errors="coerce"
        )
        frames.append(frame.reindex(columns=[*requested, "local_model_risk_pct"]))
    return pd.concat(frames, ignore_index=True, copy=False).sort_values("__ts__", kind="stable")


def _load_state() -> pd.DataFrame:
    frame = pd.read_parquet(STATE)
    if "__index_level_0__" in frame:
        timestamp = frame.pop("__index_level_0__")
    elif frame.index.name in {"__index_level_0__", "__ts__"} or isinstance(frame.index, pd.DatetimeIndex):
        timestamp = frame.index.to_series(index=frame.index)
    else:
        timestamp = frame.pop("__ts__")
    frame["__ts__"] = pd.to_datetime(timestamp, utc=True)
    return frame.drop_duplicates("__ts__", keep="last")


def _load_market_score() -> pd.DataFrame:
    frame = pd.read_csv(CHANGEPOINTS, parse_dates=["__ts__"])
    frame["__ts__"] = pd.to_datetime(frame["__ts__"], utc=True)
    frame = frame.loc[frame["method"].eq("bocpd_h48_sync4"), ["__ts__", "synchronized_break_score"]]
    return frame.drop_duplicates("__ts__", keep="last").rename(columns={"synchronized_break_score": "market_bocpd_score"})


def _panel(rows: pd.DataFrame, state: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    panel = (
        rows.groupby(["__ts__", "side_name", "archetype_policy_key"], observed=True, sort=True)
        .agg(
            candidate_rows=("parent_rank_v9", "size"),
            parent_rank_mean=("parent_rank_v9", "mean"),
            target=("top10_adverse_period_target", "max"),
            local_model_risk_pct=("local_model_risk_pct", "mean"),
        )
        .reset_index()
        .merge(market, on="__ts__", how="inner", validate="many_to_one")
        .merge(state, on="__ts__", how="left", validate="many_to_one")
    )
    panel["hour"] = panel["__ts__"].dt.hour.astype(np.int8)
    panel["day"] = panel["__ts__"].dt.floor("D")
    return panel


def _prior_brl_terms(rules: pd.DataFrame, side: str, archetype: str, start: pd.Timestamp) -> list[tuple[str, str]]:
    frame = rules.copy()
    frame["fold_start"] = pd.to_datetime(frame["fold_start"], utc=True, errors="coerce")
    frame = frame.loc[
        frame["model_arm"].eq("brl__top10_adverse_period")
        & frame["stage"].eq("oof")
        & frame["side_name"].eq(side)
        & frame["archetype_policy_key"].eq(archetype)
        & frame["fold_start"].lt(start)
    ].sort_values("fold_start", kind="stable")
    if frame.empty:
        return []
    latest = str(frame.iloc[-1]["rule_list"])
    return list(dict.fromkeys(BRL_TERM.findall(latest)))


def _brl_activation(train: pd.DataFrame, score: pd.DataFrame, terms: list[tuple[str, str]]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    active_terms = [(name, tail) for name, tail in terms if name in train and name in score]
    if not active_terms:
        return np.zeros(len(train), np.float32), np.zeros(len(score), np.float32), []
    train_score = np.zeros(len(train), dtype=np.float32)
    out_score = np.zeros(len(score), dtype=np.float32)
    names: list[str] = []
    for name, tail in active_terms:
        values = pd.to_numeric(train[name], errors="coerce").to_numpy(float)
        ref = values[np.isfinite(values)]
        if len(ref) < 64:
            continue
        threshold = float(np.quantile(ref, 0.80 if tail == "high20" else 0.20))
        direction = 1.0 if tail == "high20" else -1.0
        train_score += (direction * (np.nan_to_num(values, nan=threshold) - threshold) >= 0.0).astype(np.float32)
        target = pd.to_numeric(score[name], errors="coerce").to_numpy(float)
        out_score += (direction * (np.nan_to_num(target, nan=threshold) - threshold) >= 0.0).astype(np.float32)
        names.append(f"{name}__{tail}")
    denom = max(len(names), 1)
    return train_score / denom, out_score / denom, names


def _causal_bocpd(train: np.ndarray, score: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Use an early historical anchor only; no later value affects scaling."""
    finite = np.asarray(train, dtype=np.float64)
    finite = finite[np.isfinite(finite)][:720]
    if len(finite) < 64:
        return np.zeros(len(train), np.float32), np.zeros(len(score), np.float32)
    median = float(np.median(finite))
    q25, q75 = np.quantile(finite, [0.25, 0.75])
    scale = max(float(q75 - q25), 1e-4)
    full = np.concatenate([train, score]).astype(np.float64, copy=False)
    scaled = np.clip((np.nan_to_num(full, nan=median) - median) / scale, -8.0, 8.0)
    values = bocpd_student_t(scaled, BOCPDConfig(expected_run_hours=48, max_run_hours=96))
    return values[: len(train)], values[len(train):]


def _leaf_features(frame: pd.DataFrame, active_terms: list[str]) -> list[str]:
    """Compact, causal market-state basis for a local LGBM leaf channel."""
    from_rules = [term.rsplit("__", 1)[0] for term in active_terms]
    names = list(dict.fromkeys([*from_rules, *LEAF_STATE_DEFAULTS]))
    selected: list[str] = []
    for name in names:
        if name not in frame:
            continue
        values = pd.to_numeric(frame[name], errors="coerce")
        if values.notna().sum() >= 64 and values.std(skipna=True) > 1e-8:
            selected.append(name)
    return selected[:12]


def _leaf_target_rate(model: Any, x_fit: np.ndarray, y_fit: np.ndarray, x_score: np.ndarray) -> np.ndarray:
    """Map frozen tree leaves to Laplace-smoothed adverse rates learned on train rows."""
    fit_leaves = np.asarray(model.predict(x_fit, pred_leaf=True), dtype=np.int32)
    score_leaves = np.asarray(model.predict(x_score, pred_leaf=True), dtype=np.int32)
    if fit_leaves.ndim == 1:
        fit_leaves, score_leaves = fit_leaves[:, None], score_leaves[:, None]
    y = np.asarray(y_fit, dtype=np.float64)
    global_rate = (float(y.sum()) + 2.0) / (float(len(y)) + 4.0)
    result = np.zeros(len(x_score), dtype=np.float64)
    for tree in range(fit_leaves.shape[1]):
        unique, inverse = np.unique(fit_leaves[:, tree], return_inverse=True)
        count = np.bincount(inverse, minlength=len(unique)).astype(np.float64)
        hits = np.bincount(inverse, weights=y, minlength=len(unique))
        rate = (hits + 12.0 * global_rate) / (count + 12.0)
        location = np.searchsorted(unique, score_leaves[:, tree])
        clipped = np.minimum(location, len(unique) - 1)
        valid = (location < len(unique)) & (unique[clipped] == score_leaves[:, tree])
        result += np.where(valid, rate[clipped], global_rate)
    return (result / fit_leaves.shape[1]).astype(np.float32)


def _tree_leaf_paths(node: dict[str, Any], path: list[str], names: list[str], output: dict[int, list[str]]) -> None:
    """Decode a LightGBM leaf into its observable feature conjunction."""
    if "leaf_index" in node:
        output[int(node["leaf_index"])] = path
        return
    feature = names[int(node["split_feature"])]
    threshold = float(node["threshold"])
    _tree_leaf_paths(node["left_child"], [*path, f"{feature} <= {threshold:.6g}"], names, output)
    _tree_leaf_paths(node["right_child"], [*path, f"{feature} > {threshold:.6g}"], names, output)


def _high_risk_leaf_path_pressure(
    model: Any,
    x_fit: np.ndarray,
    y_fit: np.ndarray,
    x_score: np.ndarray,
    names: list[str],
    high_quantile: float,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Combine supported adverse LGBM leaf paths into one causal state score."""
    fit_leaves = np.asarray(model.predict(x_fit, pred_leaf=True), dtype=np.int32)
    score_leaves = np.asarray(model.predict(x_score, pred_leaf=True), dtype=np.int32)
    if fit_leaves.ndim == 1:
        fit_leaves, score_leaves = fit_leaves[:, None], score_leaves[:, None]
    y = np.asarray(y_fit, dtype=np.float64)
    global_rate = (float(y.sum()) + 2.0) / (float(len(y)) + 4.0)
    tree_paths: list[dict[int, list[str]]] = []
    for tree in model.dump_model()["tree_info"]:
        decoded: dict[int, list[str]] = {}
        _tree_leaf_paths(tree["tree_structure"], [], names, decoded)
        tree_paths.append(decoded)
    leaf_statistics: list[tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    for tree_idx in range(fit_leaves.shape[1]):
        unique, inverse = np.unique(fit_leaves[:, tree_idx], return_inverse=True)
        count = np.bincount(inverse, minlength=len(unique)).astype(np.float64)
        hits = np.bincount(inverse, weights=y, minlength=len(unique))
        rate = (hits + 12.0 * global_rate) / (count + 12.0)
        leaf_statistics.append((tree_idx, unique, count, rate, count >= 12.0))
    supported_rates = [rate[supported] for _, _, _, rate, supported in leaf_statistics if supported.any()]
    pooled_supported_rates = np.concatenate(supported_rates) if supported_rates else np.empty(0, dtype=np.float64)
    if len(pooled_supported_rates) == 0:
        return np.zeros(len(x_score), dtype=np.float32), []
    cutoff = float(np.quantile(pooled_supported_rates, high_quantile))
    pressure = np.zeros(len(x_score), dtype=np.float64)
    weight_total = 0.0
    rules: list[dict[str, Any]] = []
    for tree_idx, unique, count, rate, supported in leaf_statistics:
        chosen = np.flatnonzero(supported & (rate >= cutoff) & (rate > global_rate))
        for pos in chosen:
            leaf_id = int(unique[pos])
            weight = float(rate[pos] - global_rate)
            pressure += weight * (score_leaves[:, tree_idx] == leaf_id)
            weight_total += weight
            rules.append({
                "tree_index": tree_idx,
                "leaf_index": leaf_id,
                "train_support": int(count[pos]),
                "train_adverse_rate": float(rate[pos]),
                "global_adverse_rate": global_rate,
                "global_leaf_rate_cutoff": cutoff,
                "path": " AND ".join(tree_paths[tree_idx].get(leaf_id, [])) or "ALL",
            })
    if weight_total <= 0.0:
        return np.zeros(len(x_score), dtype=np.float32), rules
    return (pressure / weight_total).astype(np.float32), rules


def _fit_leaf_state(
    train: pd.DataFrame,
    score: pd.DataFrame,
    positives: pd.DataFrame,
    controls: pd.DataFrame,
    features: list[str],
    seed: int,
    leaf_path_quantile: float,
) -> tuple[dict[str, np.ndarray], dict[str, Any], list[dict[str, Any]]]:
    """Causal OOF leaf-risk history plus frozen OOS leaf-risk mapping.

    The OOF history is produced by two expanding chronological fits. The full
    pre-fold fit then produces OOS leaves. This avoids a row scoring on a tree
    that was trained with that row's outcome.
    """
    if len(features) < 2 or len(positives) < 12 or len(controls) < 24:
        empty_train = np.full(len(train), np.nan, dtype=np.float32)
        return {
            "rate_train": empty_train, "rate_oos": np.full(len(score), np.nan, dtype=np.float32),
            "path_train": empty_train.copy(), "path_oos": np.full(len(score), np.nan, dtype=np.float32),
        }, {"status": "insufficient_leaf_support"}, []
    def matrix(frame: pd.DataFrame) -> np.ndarray:
        value = frame.reindex(columns=features).apply(pd.to_numeric, errors="coerce").to_numpy(np.float32, copy=True)
        med = np.nanmedian(value, axis=0)
        value[~np.isfinite(value)] = np.take(med, np.nonzero(~np.isfinite(value))[1])
        return np.clip(value, -20.0, 20.0)
    x_train, x_score = matrix(train), matrix(score)
    train_key = pd.MultiIndex.from_frame(train.loc[:, ["__ts__", "side_name", "archetype_policy_key"]])
    pos_idx = train_key.get_indexer(pd.MultiIndex.from_frame(positives.loc[:, ["__ts__", "side_name", "archetype_policy_key"]]))
    ctl_idx = train_key.get_indexer(pd.MultiIndex.from_frame(controls.loc[:, ["__ts__", "side_name", "archetype_policy_key"]]))
    pos_idx, ctl_idx = pos_idx[pos_idx >= 0], ctl_idx[ctl_idx >= 0]
    def fit_at(indices: np.ndarray, random_state: int) -> tuple[Any, np.ndarray, np.ndarray]:
        local_pos = np.intersect1d(pos_idx, indices, assume_unique=False)
        local_ctl = np.intersect1d(ctl_idx, indices, assume_unique=False)
        selected = np.concatenate([local_pos, local_ctl])
        y = np.concatenate([np.ones(len(local_pos), dtype=np.int8), np.zeros(len(local_ctl), dtype=np.int8)])
        if len(local_pos) < 8 or len(local_ctl) < 16:
            raise ValueError("insufficient expanding leaf support")
        model = lgb.train(
            {
                "objective": "binary", "learning_rate": 0.045, "num_leaves": 7,
                "max_depth": 3, "min_data_in_leaf": 24, "min_gain_to_split": 0.02,
                "lambda_l1": 0.25, "lambda_l2": 1.5, "feature_fraction": 0.8,
                "bagging_fraction": 0.85, "bagging_freq": 1, "seed": random_state,
                "num_threads": 1, "verbosity": -1,
            },
            lgb.Dataset(x_train[selected], label=y, feature_name=features),
            num_boost_round=96,
        )
        return model, selected, y
    n = len(train)
    oof_rate = np.full(n, np.nan, dtype=np.float32)
    oof_path = np.full(n, np.nan, dtype=np.float32)
    for part, boundary in enumerate((n // 3, (2 * n) // 3)):
        next_boundary = (2 * n) // 3 if part == 0 else n
        if boundary < 96:
            continue
        try:
            model, selected, y = fit_at(np.arange(boundary, dtype=np.int64), seed + part)
        except ValueError:
            continue
        oof_rate[boundary:next_boundary] = _leaf_target_rate(model, x_train[selected], y, x_train[boundary:next_boundary])
        oof_path[boundary:next_boundary], _ = _high_risk_leaf_path_pressure(
            model, x_train[selected], y, x_train[boundary:next_boundary], features, leaf_path_quantile
        )
    try:
        final, selected, y = fit_at(np.arange(n, dtype=np.int64), seed + 100)
        oos_rate = _leaf_target_rate(final, x_train[selected], y, x_score)
        oos_path, rules = _high_risk_leaf_path_pressure(
            final, x_train[selected], y, x_score, features, leaf_path_quantile
        )
    except ValueError:
        oos_rate = np.full(len(score), np.nan, dtype=np.float32)
        oos_path = np.full(len(score), np.nan, dtype=np.float32)
        rules = []
    return {
        "rate_train": oof_rate, "rate_oos": oos_rate,
        "path_train": oof_path, "path_oos": oos_path,
    }, {
        "status": "ok", "features": "|".join(features),
        "oof_coverage": float(np.isfinite(oof_rate).mean()),
        "path_rule_count": len(rules),
    }, rules


def _prototype_scores(
    train_positive: pd.DataFrame,
    train_controls: pd.DataFrame,
    score: pd.DataFrame,
    features: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Leave-one-out train and frozen OOS log-distance evidence scores."""
    all_train = pd.concat([train_positive, train_controls], ignore_index=True, copy=False)
    reference = all_train[features].to_numpy(np.float64, copy=True)
    median = np.nanmedian(reference, axis=0)
    q25, q75 = np.nanquantile(reference, [0.25, 0.75], axis=0)
    scale = np.maximum(q75 - q25, 1e-4)
    def transform(frame: pd.DataFrame) -> np.ndarray:
        values = frame[features].to_numpy(np.float64, copy=True)
        values[~np.isfinite(values)] = np.take(median, np.nonzero(~np.isfinite(values))[1])
        return np.clip((values - median) / scale, -8.0, 8.0)
    pos = transform(train_positive)
    ctl = transform(train_controls)
    test = transform(score)
    k_pos = min(5, max(len(pos) - 1, 1))
    k_ctl = min(5, max(len(ctl) - 1, 1))
    def evidence(query: np.ndarray, own: np.ndarray, other: np.ndarray, leave_self: bool) -> np.ndarray:
        d_pos = np.sqrt(((query[:, None, :] - pos[None, :, :]) ** 2).sum(axis=2))
        d_ctl = np.sqrt(((query[:, None, :] - ctl[None, :, :]) ** 2).sum(axis=2))
        if leave_self and own is pos:
            np.fill_diagonal(d_pos, np.inf)
        if leave_self and own is ctl:
            np.fill_diagonal(d_ctl, np.inf)
        near_pos = np.partition(d_pos, min(k_pos - 1, d_pos.shape[1] - 1), axis=1)[:, :k_pos].mean(axis=1)
        near_ctl = np.partition(d_ctl, min(k_ctl - 1, d_ctl.shape[1] - 1), axis=1)[:, :k_ctl].mean(axis=1)
        return (near_ctl - near_pos).astype(np.float32)
    pos_train = evidence(pos, pos, ctl, True)
    ctl_train = evidence(ctl, ctl, pos, True)
    return pos_train, ctl_train, evidence(test, test, pos, False)


def _centroid_scores(
    train_positive: pd.DataFrame,
    train_controls: pd.DataFrame,
    score: pd.DataFrame,
    features: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Shrunken centroid log-distance evidence with leave-one-out train scores.

    This is deliberately a different matching geometry from nearest prototypes:
    it rewards agreement with an adverse *distribution*, not a single historical
    episode. Diagonal variance is pooled and shrunk for sparse event support.
    """
    reference = pd.concat([train_positive[features], train_controls[features]], ignore_index=True, copy=False).to_numpy(np.float64, copy=True)
    median = np.nanmedian(reference, axis=0)
    q25, q75 = np.nanquantile(reference, [0.25, 0.75], axis=0)
    scale = np.maximum(q75 - q25, 1e-4)
    def transform(frame: pd.DataFrame) -> np.ndarray:
        values = frame[features].to_numpy(np.float64, copy=True)
        values[~np.isfinite(values)] = np.take(median, np.nonzero(~np.isfinite(values))[1])
        return np.clip((values - median) / scale, -8.0, 8.0)
    pos, ctl, test = transform(train_positive), transform(train_controls), transform(score)
    pooled_var = np.maximum(np.var(np.vstack([pos, ctl]), axis=0), 0.05)
    prior_mean = np.vstack([pos, ctl]).mean(axis=0)
    def score_rows(rows: np.ndarray, own: np.ndarray, other: np.ndarray, leave_self: bool) -> np.ndarray:
        own_sum, other_sum = own.sum(axis=0), other.sum(axis=0)
        own_n, other_n = len(own), len(other)
        # 12 pseudo-observations shrink sparse adverse centroids to the pooled state.
        if leave_self:
            own_mean = (own_sum[None, :] - rows + 12.0 * prior_mean) / max(own_n - 1 + 12, 1)
        else:
            own_mean = np.repeat(((own_sum + 12.0 * prior_mean) / (own_n + 12))[None, :], len(rows), axis=0)
        other_mean = (other_sum + 12.0 * prior_mean) / (other_n + 12)
        own_dist = (((rows - own_mean) ** 2) / pooled_var).sum(axis=1)
        other_dist = (((rows - other_mean[None, :]) ** 2) / pooled_var).sum(axis=1)
        return (other_dist - own_dist).astype(np.float32)
    return score_rows(pos, pos, ctl, True), score_rows(ctl, ctl, pos, True), score_rows(test, pos, ctl, False)


def _threshold(pos: np.ndarray, ctl: np.ndarray, quantiles: tuple[float, ...]) -> dict[str, Any]:
    if len(pos) < 12 or len(ctl) < 24:
        return {"status": "insufficient_matched_support"}
    candidates = []
    for q in quantiles:
        cut = float(np.quantile(ctl, q))
        recall = float((pos >= cut).mean())
        fpr = float((ctl >= cut).mean())
        lift = recall / max(fpr, 1e-6)
        candidates.append({"quantile": q, "threshold": cut, "train_recall": recall, "train_matched_fpr": fpr, "train_lift": lift, "objective": recall - 0.75 * fpr})
    eligible = [x for x in candidates if x["train_recall"] >= .20 and x["train_matched_fpr"] <= .15 and x["train_lift"] >= 1.5]
    best = max(candidates, key=lambda x: (x["objective"], x["train_lift"]))
    return {"status": "accepted" if eligible else "no_train_only_discriminative_threshold", **(max(eligible, key=lambda x: (x["objective"], x["train_lift"])) if eligible else best)}


def run(args: argparse.Namespace) -> dict[str, Any]:
    args.output.mkdir(parents=True, exist_ok=True)
    rows = _load_rows()
    panel = _panel(rows, _load_state(), _load_market_score())
    rules = pd.read_csv(RULES)
    reports: list[dict[str, Any]] = []
    actions: list[pd.DataFrame] = []
    term_rows: list[dict[str, Any]] = []
    leaf_rule_rows: list[dict[str, Any]] = []
    for fold, (start_raw, end_raw) in enumerate(FOLDS):
        start, end = base._utc(start_raw), base._utc(end_raw)
        cutoff = start - pd.Timedelta(hours=args.embargo_hours)
        score_candidates = rows.loc[rows["__ts__"].ge(start) & rows["__ts__"].lt(end)].copy()
        for (side, archetype), local_score in panel.loc[panel["__ts__"].ge(start) & panel["__ts__"].lt(end)].groupby(["side_name", "archetype_policy_key"], observed=True):
            local_train = panel.loc[(panel["side_name"].eq(side)) & (panel["archetype_policy_key"].eq(archetype)) & panel["__ts__"].lt(cutoff)].copy()
            local_score = local_score.copy()
            if local_train.empty or len(local_score) < 16:
                continue
            terms = _prior_brl_terms(rules, side, archetype, start)
            brl_train, brl_score, active_terms = _brl_activation(local_train, local_score, terms)
            local_train["brl_activation"] = brl_train
            local_score["brl_activation"] = brl_score
            risk_train, risk_score = _causal_bocpd(local_train["local_model_risk_pct"].to_numpy(float), local_score["local_model_risk_pct"].to_numpy(float))
            brl_cp_train, brl_cp_score = _causal_bocpd(brl_train, brl_score)
            local_train["local_model_bocpd"] = risk_train
            local_score["local_model_bocpd"] = risk_score
            local_train["brl_bocpd"] = brl_cp_train
            local_score["brl_bocpd"] = brl_cp_score
            positives, controls = base._matched_controls(local_train, controls_per_positive=args.controls_per_positive)
            leaf_features = _leaf_features(local_train, active_terms)
            leaf_state, leaf_manifest, leaf_rules = _fit_leaf_state(
                local_train, local_score, positives, controls, leaf_features,
                seed=71_000 + fold * 1_000 + sum(map(ord, f"{side}|{archetype}")),
                leaf_path_quantile=args.leaf_path_risk_quantile,
            )
            local_train["leaf_state_risk"] = leaf_state["rate_train"]
            local_score["leaf_state_risk"] = leaf_state["rate_oos"]
            local_train["leaf_path_pressure"] = leaf_state["path_train"]
            local_score["leaf_path_pressure"] = leaf_state["path_oos"]
            leaf_cp_train, leaf_cp_score = _causal_bocpd(leaf_state["rate_train"], leaf_state["rate_oos"])
            leaf_path_cp_train, leaf_path_cp_score = _causal_bocpd(leaf_state["path_train"], leaf_state["path_oos"])
            local_train["leaf_state_bocpd"] = leaf_cp_train
            local_score["leaf_state_bocpd"] = leaf_cp_score
            local_train["leaf_path_bocpd"] = leaf_path_cp_train
            local_score["leaf_path_bocpd"] = leaf_path_cp_score
            # Recreate the deterministic matched sets after adding leaf fields.
            positives, controls = base._matched_controls(local_train, controls_per_positive=args.controls_per_positive)
            blocks = base._event_block_count(local_train)
            features = [
                "market_bocpd_score", "local_model_bocpd", "brl_bocpd",
                "brl_activation", "leaf_state_bocpd", "leaf_path_bocpd",
            ]
            varied = [name for name in features if np.nanstd(local_train[name].to_numpy(float)) > 1e-6]
            if len(varied) < 2 or positives.empty or controls.empty:
                decision: dict[str, Any] = {"status": "insufficient_multisignal_support"}
                score_risk = np.zeros(len(local_score), np.float32)
            else:
                candidates: list[tuple[str, dict[str, Any], np.ndarray]] = []
                for method, scorer in (
                    ("nearest_prototype", _prototype_scores),
                    ("shrunken_centroid", _centroid_scores),
                ):
                    pos_risk, ctl_risk, candidate_score = scorer(positives, controls, local_score, varied)
                    candidate_decision = _threshold(pos_risk, ctl_risk, args.decision_quantiles)
                    candidates.append((method, candidate_decision, candidate_score))
                eligible = [item for item in candidates if item[1].get("status") == "accepted"]
                pool = eligible if eligible else candidates
                method, decision, score_risk = max(
                    pool,
                    key=lambda item: (
                        float(item[1].get("objective", -np.inf)),
                        float(item[1].get("train_lift", -np.inf)),
                    ),
                )
                decision = {**decision, "matching_method": method}
            accepted = decision.get("status") == "accepted" and blocks >= args.min_event_blocks
            local_candidates = score_candidates.loc[
                score_candidates["side_name"].eq(side)
                & score_candidates["archetype_policy_key"].eq(archetype)
            ].merge(
                local_score.loc[:, ["__ts__", "side_name", "archetype_policy_key"]].assign(prototype_risk=score_risk),
                on=["__ts__", "side_name", "archetype_policy_key"], how="inner", validate="many_to_one",
            )
            blocked = (
                local_candidates["prototype_risk"].to_numpy(float) >= float(decision.get("threshold", np.inf))
                if accepted else np.zeros(len(local_candidates), bool)
            )
            base_metrics, guard_metrics = base._metrics(local_candidates), base._metrics(local_candidates.loc[~blocked])
            reports.append({
                "fold": fold, "fold_start": start, "fold_end": end, "side_name": side, "archetype_policy_key": archetype,
                "overlay_status": "accepted" if accepted else str(decision.get("status")), "train_event_blocks": blocks,
                "train_positive_timestamps": len(positives), "train_matched_controls": len(controls), "active_brl_terms": "|".join(active_terms), "signal_features": "|".join(varied),
                "leaf_state_status": leaf_manifest.get("status"), "leaf_state_features": leaf_manifest.get("features", ""), "leaf_state_oof_coverage": leaf_manifest.get("oof_coverage", np.nan), "leaf_path_rule_count": leaf_manifest.get("path_rule_count", 0),
                "matching_method": decision.get("matching_method", ""),
                **{f"decision_{k}": v for k, v in decision.items() if k != "status"},
                "baseline_selected_rows": base_metrics.get("selected_rows", 0), "guarded_selected_rows": guard_metrics.get("selected_rows", 0), "removed_rows": int(blocked.sum()), "activity_retained": float((~blocked).mean()),
                **{f"baseline_{k}": v for k, v in base_metrics.items() if k != "selected_rows"}, **{f"guarded_{k}": v for k, v in guard_metrics.items() if k != "selected_rows"},
                "delta_mean_ev_after_1pct": guard_metrics.get("mean_ev_after_1pct", np.nan) - base_metrics.get("mean_ev_after_1pct", np.nan),
            })
            actions.append(local_candidates.loc[:, ["__ts__", "side_name", "archetype_policy_key", "ev_after_1pct", "clean_exec", "top10_adverse_period_target", "prototype_risk"]].assign(
                fold=fold, guard_accepted=int(accepted), guard_blocked=blocked.astype(np.int8), guard_threshold=float(decision.get("threshold", np.nan))))
            term_rows.extend({"fold": fold, "side_name": side, "archetype_policy_key": archetype, "brl_term": term} for term in active_terms)
            leaf_rule_rows.extend({"fold": fold, "side_name": side, "archetype_policy_key": archetype, **rule} for rule in leaf_rules)
    report = pd.DataFrame(reports)
    action = pd.concat(actions, ignore_index=True, copy=False) if actions else pd.DataFrame()
    report.to_csv(args.output / "side_archetype_multisignal_metrics.csv", index=False)
    action.to_csv(args.output / "oos_multisignal_actions.csv.gz", index=False, compression="gzip")
    pd.DataFrame(term_rows).to_csv(args.output / "brl_terms_used.csv", index=False)
    pd.DataFrame(leaf_rule_rows).to_csv(args.output / "lgbm_leaf_path_rules.csv", index=False)
    manifest = {
        "purpose": "research-only multi-signal BOCPD + Bayesian prototype overlay",
        "channels": ["market_bocpd_h48_sync4", "prior_local_lgbm_or_mlp_risk_bocpd", "prior_brl_activation_bocpd", "fold_local_lgbm_leaf_state_bocpd", "fold_local_lgbm_high_risk_leaf_path_bocpd"],
        "matching": "train-only selection between leave-one-out nearest-prototype and shrunken-centroid adverse-versus-matched-benign evidence",
        "leaf_arm": "fold-local shallow LGBM leaf-risk plus high-risk leaf-path pressure: chronological OOF leaves for train prototypes and a frozen pre-fold leaf map for OOS; path rules are exported; saved final-model leaves remain intentionally excluded",
        "outcome_inputs_at_inference": False,
        "policy_wiring": False,
        "embargo_hours": args.embargo_hours,
        "leaf_path_risk_quantile": args.leaf_path_risk_quantile,
        "decision_quantiles": list(args.decision_quantiles),
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--controls-per-positive", type=int, default=3)
    parser.add_argument("--min-event-blocks", type=int, default=3)
    parser.add_argument("--embargo-hours", type=int, default=48)
    parser.add_argument("--leaf-path-risk-quantile", type=float, default=0.85)
    parser.add_argument("--min-decision-quantile", type=float, default=0.70)
    args = parser.parse_args()
    if not 0.50 <= args.leaf_path_risk_quantile < 1.0:
        parser.error("--leaf-path-risk-quantile must be in [0.50, 1.0)")
    args.decision_quantiles = tuple(q for q in (*base.QUANTILES, 0.97, 0.98, 0.99) if q >= args.min_decision_quantile)
    if not args.decision_quantiles:
        parser.error("--min-decision-quantile leaves no candidate thresholds")
    return args


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, default=str))
