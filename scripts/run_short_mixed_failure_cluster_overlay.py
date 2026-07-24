#!/usr/bin/env python3
"""Discover adverse short-mixed mechanisms and fit one OOS overlay per cluster."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN

from extreme_price_movements.features_negative_residuals import (
    NEGATIVE_RESIDUAL_META_FEATURE_KEYS,
    NEGATIVE_RESIDUAL_TEMPORAL_MECHANISM_FEATURE_KEYS,
)
from scripts import run_meta_residual_event_balanced_error_overlay as base


SIDE = "short"
ARCHETYPE = "short_mixed_clean_path"
TARGET = "bad_residual_event_target"
RISK = "short_mixed_cluster_risk"
RISK_PCT = "short_mixed_cluster_risk_percentile"
FOLD_STARTS = (
    pd.Timestamp("2025-09-01", tz="UTC"),
    pd.Timestamp("2025-11-01", tz="UTC"),
    pd.Timestamp("2026-01-01", tz="UTC"),
)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _join_context(frame: pd.DataFrame, path: Path) -> pd.DataFrame:
    available = pd.read_parquet(path)
    if "ts" in available.columns:
        ts = pd.to_datetime(available.pop("ts"), utc=True, errors="coerce")
    else:
        ts = pd.to_datetime(available.index, utc=True, errors="coerce")
    available.index = ts
    columns = [name for name in NEGATIVE_RESIDUAL_META_FEATURE_KEYS if name in available]
    context = available.loc[~available.index.duplicated(keep="last"), columns]
    missing = [name for name in columns if name not in frame]
    if not missing:
        return frame
    return frame.merge(
        context[missing].reset_index(names="__ts__"),
        on="__ts__",
        how="left",
        validate="many_to_one",
    )


def _state(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    local = frame.loc[
        frame["side_name"].astype(str).eq(SIDE)
        & frame["archetype_policy_key"].astype(str).eq(ARCHETYPE)
        & frame["parent_rank_v9"].ge(0.80)
    ].copy()
    return base._timestamp_training_frame(
        local,
        features,
        target_column=TARGET,
        event_column="adverse_calendar_cell",
    )


def _robust_matrix(
    fit: pd.DataFrame,
    score: pd.DataFrame,
    features: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_fit = fit[features].to_numpy(np.float32, copy=True)
    x_score = score[features].to_numpy(np.float32, copy=True)
    median = np.nanmedian(x_fit, axis=0).astype(np.float32)
    q25 = np.nanquantile(x_fit, 0.25, axis=0).astype(np.float32)
    q75 = np.nanquantile(x_fit, 0.75, axis=0).astype(np.float32)
    scale = np.maximum(q75 - q25, np.float32(1e-4))
    median = np.nan_to_num(median, nan=0.0)
    for matrix in (x_fit, x_score):
        missing = ~np.isfinite(matrix)
        if missing.any():
            matrix[missing] = np.take(median, np.nonzero(missing)[1])
        matrix -= median
        matrix /= scale
        np.clip(matrix, -5.0, 5.0, out=matrix)
    return x_fit, x_score, median, scale


def _cluster_features(adverse: pd.DataFrame, candidates: list[str]) -> list[str]:
    rows: list[tuple[str, float]] = []
    for name in candidates:
        values = pd.to_numeric(adverse[name], errors="coerce").to_numpy(np.float32)
        finite = np.isfinite(values)
        if finite.mean() < 0.70 or np.unique(values[finite]).size < 4:
            continue
        q25, q75 = np.nanquantile(values, [0.25, 0.75])
        rows.append((name, float(q75 - q25)))
    ordered = [name for name, _ in sorted(rows, key=lambda item: item[1], reverse=True)]
    selected: list[str] = []
    for name in ordered:
        if len(selected) >= 10:
            break
        values = pd.to_numeric(adverse[name], errors="coerce")
        if any(abs(values.corr(pd.to_numeric(adverse[other], errors="coerce"), method="spearman")) > 0.90 for other in selected):
            continue
        selected.append(name)
    return selected


def _fit_cluster_ensemble(
    fit: pd.DataFrame,
    score: pd.DataFrame,
    candidates: list[str],
    seed: int,
) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    adverse = fit.loc[fit[TARGET].eq(1)]
    features = _cluster_features(adverse, candidates)
    if len(adverse) < 20 or len(features) < 3:
        raise RuntimeError("Insufficient adverse short-mixed support for clustering")
    x_fit, x_score, median, scale = _robust_matrix(fit, score, features)
    adverse_positions = np.flatnonzero(fit[TARGET].to_numpy(np.int8) > 0)
    minimum_cluster = max(5, min(20, len(adverse_positions) // 5))
    cluster_attempts = (
        {
            "min_cluster_size": minimum_cluster,
            "min_samples": 3,
            "cluster_selection_method": "leaf",
            "allow_single_cluster": False,
        },
        {
            "min_cluster_size": max(5, minimum_cluster // 2),
            "min_samples": 2,
            "cluster_selection_method": "eom",
            "allow_single_cluster": False,
        },
        {
            "min_cluster_size": max(5, minimum_cluster // 2),
            "min_samples": 2,
            "cluster_selection_method": "eom",
            "allow_single_cluster": True,
        },
    )
    labels = np.full(len(adverse_positions), -1, dtype=np.int16)
    cluster_params: dict[str, Any] = {}
    best_key = (-1, -1.0)
    for params in cluster_attempts:
        candidate = HDBSCAN(**params).fit(x_fit[adverse_positions])
        candidate_labels = np.asarray(candidate.labels_, dtype=np.int16)
        candidate_ids = [int(value) for value in np.unique(candidate_labels) if value >= 0]
        coverage = float(np.mean(candidate_labels >= 0))
        key = (len(candidate_ids), coverage)
        if key > best_key:
            best_key = key
            labels = candidate_labels
            cluster_params = dict(params)
    cluster_ids = sorted(int(value) for value in np.unique(labels) if value >= 0)
    if not cluster_ids:
        raise RuntimeError("HDBSCAN found no supported adverse short-mixed cluster")
    scores: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    for cluster_id in cluster_ids:
        target = np.zeros(len(fit), dtype=np.float32)
        members = adverse_positions[labels == cluster_id]
        target[members] = 1.0
        if len(members) < 5:
            continue
        model = lgb.train(
            {
                "objective": "binary",
                "metric": "None",
                "learning_rate": 0.035,
                "max_depth": 3,
                "num_leaves": 7,
                "min_data_in_leaf": 40,
                "min_gain_to_split": 0.03,
                "lambda_l1": 1.5,
                "lambda_l2": 10.0,
                "feature_fraction": 0.85,
                "bagging_fraction": 0.85,
                "bagging_freq": 1,
                "seed": seed + cluster_id,
                "verbosity": -1,
                "force_col_wise": True,
            },
            lgb.Dataset(
                x_fit,
                label=target,
                weight=np.where(target > 0, len(target) / (2 * len(members)), len(target) / (2 * max(len(target) - len(members), 1))).astype(np.float32),
                feature_name=features,
            ),
            num_boost_round=140,
        )
        score_values = np.asarray(model.predict(x_score), dtype=np.float32)
        scores.append(score_values)
        center = np.nanmedian(x_fit[members], axis=0)
        leading = np.argsort(np.abs(center))[::-1][:4]
        rows.append(
            {
                "cluster_id": cluster_id,
                "support_timestamps": int(len(members)),
                "support_days": int(fit.iloc[members]["day"].nunique()),
                "leading_state_features": "|".join(features[index] for index in leading),
                "leading_state_directions": "|".join(
                    f"{features[index]}:{center[index]:+.3f}" for index in leading
                ),
            }
        )
    if not scores:
        raise RuntimeError("No supported short-mixed cluster classifier")
    return np.max(np.column_stack(scores), axis=1).astype(np.float32), rows, {
        "features": features,
        "median": median,
        "scale": scale,
        "clusters": len(scores),
        "noise_fraction": float(np.mean(labels < 0)),
        "cluster_params": cluster_params,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    args.output.mkdir(parents=True, exist_ok=True)
    train = pd.read_parquet(args.train_predictions)
    valid = pd.read_parquet(args.eval_predictions)
    for frame in (train, valid):
        frame["__ts__"] = pd.to_datetime(frame["__ts__"], utc=True)
        frame["day"] = pd.to_datetime(frame["day"], utc=True).dt.floor("D")
    train = _join_context(train, args.market_features)
    valid = _join_context(valid, args.market_features)
    candidates = [
        name
        for name in NEGATIVE_RESIDUAL_TEMPORAL_MECHANISM_FEATURE_KEYS
        if name in train
    ]
    train_state = _state(train, candidates)
    valid_state = _state(valid, candidates)
    oof_parts: list[pd.DataFrame] = []
    catalog: list[dict[str, Any]] = []
    for fold_index, start in enumerate(FOLD_STARTS):
        end = FOLD_STARTS[fold_index + 1] if fold_index + 1 < len(FOLD_STARTS) else pd.Timestamp("2026-04-01", tz="UTC")
        fit = train_state.loc[train_state["__ts__"].lt(start - pd.Timedelta(days=2))]
        score = train_state.loc[train_state["__ts__"].ge(start) & train_state["__ts__"].lt(end)]
        if fit.empty or score.empty:
            continue
        values, rows, state = _fit_cluster_ensemble(fit, score, candidates, args.seed + 100 * fold_index)
        reference_values, _, _ = _fit_cluster_ensemble(fit, fit, candidates, args.seed + 100 * fold_index)
        part = score.loc[:, ["__ts__", "day", "ev_after_1pct", "clean_exec", "adverse_calendar_cell", TARGET]].copy()
        part["parent_rank_v9"] = train.loc[train["__ts__"].isin(part["__ts__"]), ["__ts__", "parent_rank_v9"]].groupby("__ts__")["parent_rank_v9"].median().reindex(part["__ts__"]).to_numpy(np.float32)
        part[RISK] = values
        part[RISK_PCT] = base._midrank(values, np.sort(reference_values))
        part["fold_start"] = start
        oof_parts.append(part)
        for row in rows:
            catalog.append({"stage": "oof", "fold_start": start, **state, **row})
    if not oof_parts:
        raise RuntimeError("No short-mixed cluster OOF predictions")
    oof = pd.concat(oof_parts, ignore_index=True)
    search, accepted = base._search_local_overlay(
        oof,
        base.Config(max_features=len(candidates), seed=args.seed),
        risk_column=RISK_PCT,
    )
    final_score, rows, final_state = _fit_cluster_ensemble(
        train_state,
        valid_state,
        candidates,
        args.seed + 10_000,
    )
    train_reference, _, _ = _fit_cluster_ensemble(
        train_state,
        train_state,
        candidates,
        args.seed + 10_000,
    )
    valid_state[RISK] = final_score
    valid_state[RISK_PCT] = base._midrank(final_score, np.sort(train_reference))
    for row in rows:
        catalog.append({"stage": "final", "fold_start": "final", **final_state, **row})
    search.to_csv(args.output / "overlay_search.csv", index=False)
    pd.DataFrame(catalog).to_csv(args.output / "cluster_catalog.csv", index=False)
    oof.to_parquet(args.output / "train_oof_predictions.parquet", index=False)
    valid_state.to_parquet(args.output / "eval_state_predictions.parquet", index=False)
    manifest = {
        "schema": "short_mixed_failure_cluster_overlay_v1",
        "accepted": accepted,
        "candidate_features": candidates,
        "oof_rows": len(oof),
        "eval_state_rows": len(valid_state),
        "clusters_final": final_state["clusters"],
        "leakage_contract": (
            "HDBSCAN sees adverse short-mixed timestamps from each chronological fit only. "
            "One shallow classifier per cluster predicts frozen cluster membership from "
            "pre-entry temporal mechanisms. Two days are purged before every OOF fold."
        ),
    }
    (args.output / "manifest.json").write_text(
        json.dumps(_json_safe(manifest), indent=2, sort_keys=True) + "\n"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-predictions", type=Path, required=True)
    parser.add_argument("--eval-predictions", type=Path, required=True)
    parser.add_argument("--market-features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260713)
    args = parser.parse_args()
    print(json.dumps(_json_safe(run(args)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
